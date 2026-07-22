#!/usr/bin/env python3
"""Generate one breed-correct canonical animal image for target-native 3D work.

The identity image owns breed identity, proportions, silhouette and camera.  A
real-photo board supplies breed appearance evidence only.  The result always
stops at project-owner 2D review; this tool never authorizes Pixel3D, rigging,
asset registration or instance generation.

The undistilled FLUX.2 Klein Base model is loaded fully onto one CUDA device.
CPU/model offload and low-VRAM execution are intentionally unsupported.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

from PIL import Image, ImageDraw, ImageFont


MODEL_ID = "black-forest-labs/FLUX.2-klein-base-4B"
MODEL_REVISION = "a3b4f4849157f664bdbc776fd7453c2783562f4d"
SNAPSHOT_ENV = "AVENGINE_FLUX2_KLEIN_BASE_SNAPSHOT"
SCHEMA = "avengine_target_native_animal_canonical_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-image", required=True, type=Path)
    parser.add_argument("--appearance-reference-board", type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--negative-prompt-file", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-name", default="canonical.png")
    parser.add_argument("--species", required=True)
    parser.add_argument("--breed", required=True)
    parser.add_argument("--coat-profile", required=True)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=(Path(os.environ[SNAPSHOT_ENV]) if SNAPSHOT_ENV in os.environ else None),
        help=f"Undistilled FLUX Base snapshot; defaults to ${SNAPSHOT_ENV}.",
    )
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    return parser.parse_args()


def _require_regular_image(path: Path, label: str) -> tuple[Path, Image.Image]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or is a symlink: {path}")
    with Image.open(path) as opened:
        opened.load()
        if opened.size != (1024, 1024):
            raise RuntimeError(f"{label} must be 1024x1024: {path}")
        return path, opened.convert("RGB")


def _read_prompt(path: Path, label: str) -> tuple[Path, str]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} is missing or is a symlink: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"{label} is empty: {path}")
    return path, text


def _require_base_snapshot(path: Path | None) -> Path:
    if path is None:
        raise RuntimeError(
            f"provide --snapshot or set {SNAPSHOT_ENV} to the local model snapshot"
        )
    path = path.resolve()
    index_path = path / "model_index.json"
    if path.is_symlink() or not path.is_dir() or not index_path.is_file():
        raise RuntimeError(f"FLUX Base snapshot is missing: {path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("_class_name") != "Flux2KleinPipeline":
        raise RuntimeError("snapshot has the wrong pipeline class")
    if index.get("is_distilled", False) is not False:
        raise RuntimeError("canonical generation requires undistilled FLUX Base")
    return path


def _token_count(tokenizer: Any, prompt: str) -> int:
    chat = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return len(tokenizer(chat, add_special_tokens=False)["input_ids"])


def _contact_sheet(
    identity: Image.Image,
    appearance: Image.Image | None,
    output: Image.Image,
    target: Path,
) -> None:
    panels = [("identity authority", identity)]
    if appearance is not None:
        panels.append(("real breed references", appearance))
    panels.append(("canonical candidate", output))
    panel = 512
    sheet = Image.new("RGB", (panel * len(panels), panel), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (label, image) in enumerate(panels):
        resized = image.resize((panel, panel), Image.Resampling.LANCZOS)
        x = panel * index
        sheet.paste(resized, (x, 0))
        bounds = draw.textbbox((0, 0), label, font=font)
        draw.rectangle((x + 8, 8, x + bounds[2] + 18, bounds[3] + 18), fill=(0, 0, 0))
        draw.text((x + 13, 13), label, fill=(255, 255, 255), font=font)
    sheet.save(target, format="PNG", optimize=False, compress_level=6)


def run(args: argparse.Namespace) -> Path:
    if args.gpu < 0:
        raise ValueError("--gpu must be non-negative")
    if Path(args.output_name).name != args.output_name or not args.output_name.endswith(".png"):
        raise ValueError("--output-name must be one PNG filename")

    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError(f"refusing to replace output root: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    snapshot = _require_base_snapshot(args.snapshot)
    identity_path, identity = _require_regular_image(args.identity_image, "identity image")
    if args.appearance_reference_board is None:
        appearance_path = None
        appearance = None
    else:
        appearance_path, appearance = _require_regular_image(
            args.appearance_reference_board, "appearance reference board"
        )
    prompt_path, prompt = _read_prompt(args.prompt_file, "prompt file")
    negative_path, negative_prompt = _read_prompt(
        args.negative_prompt_file, "negative prompt file"
    )

    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )

    import torch
    from diffusers import Flux2KleinPipeline
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)
    positive_tokens = _token_count(tokenizer, prompt)
    negative_tokens = _token_count(tokenizer, negative_prompt)
    if max(positive_tokens, negative_tokens) > args.max_sequence_length:
        raise RuntimeError(
            "prompt exceeds max sequence length: "
            f"positive={positive_tokens}, negative={negative_tokens}, "
            f"limit={args.max_sequence_length}"
        )

    load_started = time.perf_counter()
    pipeline = Flux2KleinPipeline.from_pretrained(
        str(snapshot), torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    model_load_seconds = time.perf_counter() - load_started
    pipeline.set_progress_bar_config(disable=False)

    try:
        negative_embeds, _ = pipeline.encode_prompt(
            prompt=negative_prompt,
            device=pipeline._execution_device,
            num_images_per_prompt=1,
            max_sequence_length=args.max_sequence_length,
        )
        generator = torch.Generator("cuda").manual_seed(args.seed)
        inference_started = time.perf_counter()
        conditioning_images = [identity]
        if appearance is not None:
            conditioning_images.append(appearance)
        result = pipeline(
            image=conditioning_images,
            prompt=prompt,
            negative_prompt_embeds=negative_embeds,
            width=1024,
            height=1024,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            max_sequence_length=args.max_sequence_length,
        )
        inference_seconds = time.perf_counter() - inference_started
        if not getattr(result, "images", None) or len(result.images) != 1:
            raise RuntimeError("FLUX must return exactly one canonical candidate")
        candidate = result.images[0].convert("RGB")
        if candidate.size != (1024, 1024):
            raise RuntimeError(f"FLUX changed output resolution: {candidate.size}")

        output_root.mkdir()
        output_path = output_root / args.output_name
        candidate.save(output_path, format="PNG", optimize=False, compress_level=6)
        contact_path = output_root / "review_contact_sheet.png"
        _contact_sheet(identity, appearance, candidate, contact_path)
        manifest = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "rendered_pending_project_owner_review",
            "state_classification": "research_candidate",
            "formal_asset_registration_authorized": False,
            "pixel3d_authorized": False,
            "identity": {
                "species": args.species,
                "breed": args.breed,
                "coat_profile": args.coat_profile,
            },
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "snapshot": str(snapshot),
                "is_distilled": False,
                "execution": "full_single_gpu_no_cpu_offload",
                "gpu": args.gpu,
            },
            "inputs": {
                "identity_image": {
                    "path": str(identity_path),
                    "role": "breed_identity_proportions_silhouette_and_camera_authority",
                },
                "appearance_reference_board": {
                    "path": str(appearance_path) if appearance_path is not None else None,
                    "role": (
                        "real_photo_breed_appearance_evidence_only"
                        if appearance_path is not None
                        else None
                    ),
                    "rights_review": (
                        "required_before_formal_promotion"
                        if appearance_path is not None
                        else None
                    ),
                },
            },
            "prompt_files": {
                "positive": str(prompt_path),
                "negative": str(negative_path),
            },
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "prompt_token_counts": {
                "positive": positive_tokens,
                "negative": negative_tokens,
            },
            "inference": {
                "seed": args.seed,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "max_sequence_length": args.max_sequence_length,
                "negative_conditioning": "native_negative_prompt_embeddings_with_cfg",
                "model_load_seconds": model_load_seconds,
                "inference_seconds": inference_seconds,
            },
            "outputs": {
                "canonical_image": str(output_path),
                "review_contact_sheet": str(contact_path),
            },
            "next_gate": (
                "project_owner_accepts_breed_identity_and_four_complete_limbs_"
                "before_any_pixel3d_execution"
            ),
        }
        manifest_path = output_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest_path
    finally:
        del pipeline
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    try:
        manifest = run(args)
    except Exception as error:
        print(f"TARGET_NATIVE_ANIMAL_CANONICAL_FAILED {type(error).__name__}: {error}")
        return 2
    print(f"TARGET_NATIVE_ANIMAL_CANONICAL_OK manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
