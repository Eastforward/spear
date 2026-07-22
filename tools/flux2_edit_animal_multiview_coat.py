#!/usr/bin/env python3
"""Edit a fixed animal four-view montage with undistilled FLUX.2 Klein Base.

This is a research-only, single-invocation coat edit.  Geometry, rigging, and
animation remain outside the model's authority; accepted pixels are projected
onto the unchanged source mesh by a separate Blender step.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


VIEW_ORDER = ("front", "back", "left", "right")
MODEL_REVISION = "a3b4f4849157f664bdbc776fd7453c2783562f4d"
DEFAULT_SNAPSHOT = os.environ.get("AVENGINE_FLUX2_KLEIN_BASE_SNAPSHOT")
SCHEMA = "avengine_flux2_base_animal_multiview_coat_edit_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-view-dir", type=Path, required=True)
    parser.add_argument(
        "--appearance-reference-board",
        type=Path,
        help=(
            "Optional 1024x1024 real-photo board used only as breed/coat "
            "appearance evidence. The source montage remains geometry authority."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-description", required=True)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(DEFAULT_SNAPSHOT) if DEFAULT_SNAPSHOT else None,
        help=(
            "Undistilled FLUX.2 Klein Base snapshot. Defaults to "
            "AVENGINE_FLUX2_KLEIN_BASE_SNAPSHOT."
        ),
    )
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--max-sequence-length", type=int, default=512)
    return parser.parse_args()


def require_new_directory(path: Path) -> Path:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace output root: {path}")
    path.mkdir(parents=True)
    return path


def require_base_snapshot(path: Path) -> Path:
    path = path.resolve()
    index_path = path / "model_index.json"
    if path.is_symlink() or not path.is_dir() or not index_path.is_file():
        raise RuntimeError(f"FLUX Base snapshot is missing: {path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("_class_name") != "Flux2KleinPipeline":
        raise RuntimeError("snapshot has the wrong pipeline class")
    if index.get("is_distilled", False) is not False:
        raise RuntimeError("coat editor requires undistilled FLUX.2 Klein Base")
    return path


def main() -> int:
    args = parse_args()
    output_root = require_new_directory(args.output_root)
    if args.snapshot is None:
        raise RuntimeError(
            "provide --snapshot or AVENGINE_FLUX2_KLEIN_BASE_SNAPSHOT"
        )
    snapshot = require_base_snapshot(args.snapshot)
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
    from PIL import Image
    from transformers import AutoTokenizer

    source_views = []
    for name in VIEW_ORDER:
        path = (args.input_view_dir / f"{name}.png").resolve()
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"missing source view: {path}")
        with Image.open(path) as opened:
            opened.load()
            if opened.size != (512, 512):
                raise RuntimeError(f"source view must be 512x512: {path}")
            source_views.append(opened.convert("RGB"))

    boxes = {
        "front": (0, 0, 512, 512),
        "back": (512, 0, 1024, 512),
        "left": (0, 512, 512, 1024),
        "right": (512, 512, 1024, 1024),
    }
    montage = Image.new("RGB", (1024, 1024), (128, 128, 128))
    for name, image in zip(VIEW_ORDER, source_views):
        montage.paste(image, boxes[name][:2])
    source_montage = output_root / "source_montage.png"
    montage.save(source_montage, format="PNG", optimize=False, compress_level=6)

    appearance_reference = None
    if args.appearance_reference_board is not None:
        appearance_reference_path = args.appearance_reference_board.resolve()
        if appearance_reference_path.is_symlink() or not appearance_reference_path.is_file():
            raise RuntimeError(
                f"appearance reference board is missing: {appearance_reference_path}"
            )
        with Image.open(appearance_reference_path) as opened:
            opened.load()
            if opened.size != (1024, 1024):
                raise RuntimeError(
                    "appearance reference board must be 1024x1024: "
                    f"{appearance_reference_path}"
                )
            appearance_reference = opened.convert("RGB")
        role_instruction = (
            "Image 1 is the edit target and sole geometry, identity, pose, camera, "
            "lighting, background, crop, and 2x2 layout authority. Image 2 is a "
            "real-photo appearance board of genuine examples; use it only for "
            "breed-appropriate coat colour, ticking distribution, and facial coat "
            "accents. Do not copy any Image 2 animal's pose, body shape, silhouette, "
            "background, lighting, collar, or individual identity. The output must "
            "remain exactly Image 1's 2x2 orthographic montage. "
        )
    else:
        appearance_reference_path = None
        role_instruction = (
            "The input image is the edit target and sole geometry, identity, pose, "
            "camera, lighting, background, crop, and 2x2 layout authority. "
        )

    prompt = (
        role_instruction
        + "Precise object edit of a fixed 2x2 orthographic montage of one 3D animal: "
        "front at top-left, back at top-right, left side at bottom-left, right "
        "side at bottom-right. Change only the coat pixels in all four panels "
        f"to this exact target: {args.target_description}. A ticked coat means "
        "microscopic alternating colour bands on each individual hair; from a "
        "normal viewing distance the body must look smooth, even, and finely "
        "salt-and-pepper blue-grey over warm beige, never striped. Keep the "
        "same individual, exact mesh silhouette, body and head shape, ears, "
        "muzzle, eyes, four legs, paws, one tail, pose, camera, crop, panel "
        "layout, neutral lighting, and background unchanged. Keep corresponding "
        "body regions consistent across all views. Photorealistic short fur."
    )
    negative_prompt = (
        "saturated blue fur, pure blue fur, cyan fur, navy fur, blue-dyed animal, "
        "monochrome blue coat, body stripes, parallel lines, contour lines, mackerel tabby, classic "
        "tabby, spots, leg bars, tail rings, necklace markings, white patches, "
        "pink-dyed face, pink-dyed paws, geometry change, silhouette change, "
        "pose change, camera change, crop, panel reordering, extra animal, "
        "extra limb, missing limb, fused legs, extra tail, missing tail, text, "
        "watermark, illustration, cartoon"
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot / "tokenizer", local_files_only=True)
    for label, text in (("positive", prompt), ("negative", negative_prompt)):
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        token_count = len(tokenizer(chat, add_special_tokens=False)["input_ids"])
        if token_count > args.max_sequence_length:
            raise RuntimeError(f"{label} prompt is too long: {token_count}")

    load_started = time.perf_counter()
    pipeline = Flux2KleinPipeline.from_pretrained(
        str(snapshot),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        device_map="balanced",
        max_memory={0: "46GiB"},
    )
    model_load_seconds = time.perf_counter() - load_started
    pipeline.set_progress_bar_config(disable=False)

    negative_embeds, _ = pipeline.encode_prompt(
        prompt=negative_prompt,
        device=pipeline._execution_device,
        num_images_per_prompt=1,
        max_sequence_length=args.max_sequence_length,
    )
    generator = torch.Generator("cuda").manual_seed(args.seed)
    conditioning_images = [montage]
    if appearance_reference is not None:
        conditioning_images.append(appearance_reference)
    inference_started = time.perf_counter()
    result = pipeline(
        image=conditioning_images,
        prompt=prompt,
        negative_prompt_embeds=negative_embeds,
        height=1024,
        width=1024,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        max_sequence_length=args.max_sequence_length,
    )
    inference_seconds = time.perf_counter() - inference_started
    if not getattr(result, "images", None) or len(result.images) != 1:
        raise RuntimeError("FLUX returned no edited image")
    edited = result.images[0].convert("RGB")
    if edited.size != (1024, 1024):
        raise RuntimeError(f"FLUX output resolution changed: {edited.size}")
    edited_montage = output_root / "edited_montage.png"
    edited.save(edited_montage, format="PNG", optimize=False, compress_level=6)
    edited_views = output_root / "edited_views"
    edited_views.mkdir()
    for name in VIEW_ORDER:
        edited.crop(boxes[name]).save(edited_views / f"{name}.png")

    manifest = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": "technical_spike_only",
        "formal_dataset_registration_authorized": False,
        "model": "black-forest-labs/FLUX.2-klein-base-4B",
        "revision": MODEL_REVISION,
        "snapshot": str(snapshot),
        "is_distilled": False,
        "pipeline": "Flux2KleinPipeline",
        "device_map": "balanced_single_gpu_direct_load",
        "max_memory": {"0": "46GiB"},
        "pipeline_hf_device_map": getattr(pipeline, "hf_device_map", None),
        "one_model_invocation": True,
        "gpu": args.gpu,
        "seed": args.seed,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "negative_conditioning": "native_negative_prompt_embeddings_with_cfg",
        "model_load_seconds": model_load_seconds,
        "inference_seconds": inference_seconds,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "target_description": args.target_description,
        "input_view_dir": str(args.input_view_dir.resolve()),
        "appearance_reference_board": (
            str(appearance_reference_path) if appearance_reference_path is not None else None
        ),
        "reference_image_count": len(conditioning_images),
        "image_roles": {
            "image_1": "edit_target_and_geometry_authority",
            "image_2": (
                "real_photo_coat_appearance_only"
                if appearance_reference_path is not None
                else None
            ),
        },
        "source_montage": str(source_montage),
        "edited_montage": str(edited_montage),
        "edited_view_dir": str(edited_views),
        "view_order": list(VIEW_ORDER),
        "geometry_rig_or_animation_edit_authorized": False,
        "next_gate": "multiview_pose_silhouette_and_coat_visual_qa",
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"FLUX2_ANIMAL_MULTIVIEW_COAT_EDIT_OK output={output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
