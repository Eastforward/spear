#!/usr/bin/env python3
"""Render authenticated multiview evidence for a controlled Pixal animal batch."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts
from tools import rocketbox_native_material_canary as immutable
from tools import run_controlled_animal_pixal_jobs as pixal_runner


REVIEW_BATCH_SCHEMA = "avengine_controlled_animal_static_review_batch_v1"
REVIEW_SCHEMA = "avengine_controlled_animal_static_review_v1"
SPEAR_ROOT = Path(__file__).resolve().parents[1]
BLENDER = Path("/data/jzy/.local/bin/blender")
RENDERER = SPEAR_ROOT / "tools/blender_render_i23d_review.py"
VIEWS = ("front", "back", "side", "top", "quarter")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256(
        {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    )


def _relative(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_pixal_batch(path: Path) -> tuple[Path, dict[str, Any]]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"Pixal batch is missing: {path}")
    payload = contracts.load_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != pixal_runner.BATCH_SCHEMA
        or payload.get("status") != "passed_generation_and_glb_readback"
        or payload.get("batch_sha256") != _hash_without(payload, "batch_sha256")
        or payload.get("automatic_checks", {}).get("overall") != "passed"
        or payload.get("job_count") != len(payload.get("attempts", []))
    ):
        raise contracts.ContractError("Pixal batch contract/hash is invalid")
    root = path.parent
    identifiers = set()
    for attempt in payload["attempts"]:
        instance_id = attempt.get("instance_id")
        if not instance_id or instance_id in identifiers:
            raise contracts.ContractError("Pixal attempts contain duplicate IDs")
        identifiers.add(instance_id)
        output = (root / attempt["output"]["path"]).resolve()
        try:
            output.relative_to(root)
        except ValueError as error:
            raise contracts.ContractError("Pixal output escaped its batch root") from error
        if (
            output.is_symlink()
            or not output.is_file()
            or output.stat().st_size != attempt["output"]["size_bytes"]
            or _sha256_file(output) != attempt["output"]["sha256"]
        ):
            raise contracts.ContractError(f"Pixal output changed: {instance_id}")
    return path, payload


def _panel(image: Image.Image, label: str) -> Image.Image:
    panel = image.convert("RGB").resize((320, 320), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    bounds = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((8, 8, bounds[2] + 18, bounds[3] + 18), fill=(0, 0, 0))
    draw.text((13, 13), label, font=font, fill=(255, 255, 255))
    return panel


def build_contact_sheet(reference_path: Path, view_root: Path, output: Path) -> None:
    with Image.open(reference_path) as opened:
        opened.load()
        reference = opened.convert("RGBA")
    backdrop = Image.new("RGB", reference.size, (205, 205, 205))
    backdrop.paste(reference.convert("RGB"), mask=reference.getchannel("A"))
    panels = [("approved FLUX.2", backdrop)]
    for view in VIEWS:
        with Image.open(view_root / f"{view}.png") as opened:
            opened.load()
            panels.append((view, opened.convert("RGB")))
    canvas = Image.new("RGB", (960, 640), (28, 28, 28))
    for index, (label, image) in enumerate(panels):
        canvas.paste(_panel(image, label), ((index % 3) * 320, (index // 3) * 320))
    canvas.save(output, format="PNG", optimize=False, compress_level=6)


def _render_one(
    attempt: dict[str, Any], pixal_root: Path, staging: Path
) -> dict[str, Any]:
    instance_id = attempt["instance_id"]
    destination = staging / instance_id
    views = destination / "views"
    log_path = destination / "blender.log"
    destination.mkdir(parents=True, exist_ok=False)
    glb = (pixal_root / attempt["output"]["path"]).resolve()
    command = [
        str(BLENDER),
        "-b",
        "--python",
        str(RENDERER),
        "--",
        "--input",
        str(glb),
        "--output-dir",
        str(views),
        "--width",
        "480",
        "--height",
        "480",
        "--front-axis",
        "negative-x",
        "--include-top",
        "--animal-material-preview",
    ]
    with log_path.open("xb") as log:
        completed = subprocess.run(
            command,
            cwd=SPEAR_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=1800,
            check=False,
        )
        log.flush()
        os.fsync(log.fileno())
    if completed.returncode != 0:
        raise contracts.ContractError(f"Blender static review failed: {instance_id}")
    render_manifest_path = views / "render_manifest.json"
    render_manifest = contracts.load_json(render_manifest_path)
    if (
        Path(render_manifest.get("input", "")).resolve() != glb
        or render_manifest.get("front_axis") != "negative-x"
        or set(render_manifest.get("views", {})) != set(VIEWS)
        or render_manifest.get("resolution") != [480, 480]
        or render_manifest.get("material_preview", {}).get("mode")
        != "ue_animal_nonmetallic_roughness_preview_v1"
        or render_manifest.get("material_preview", {}).get(
            "principled_nodes_changed", 0
        )
        <= 0
        or render_manifest.get("lighting")
        != {
            "area_light_scale": 0.25,
            "world_strength": 0.25,
            "exposure": -0.5,
        }
    ):
        raise contracts.ContractError(f"static render manifest mismatch: {instance_id}")
    for view in VIEWS:
        image = views / f"{view}.png"
        if not image.is_file():
            raise contracts.ContractError(f"missing static view {view}: {instance_id}")
        with Image.open(image) as opened:
            opened.load()
            if opened.size != (480, 480):
                raise contracts.ContractError("static view resolution changed")
    reference_path = Path(attempt["pixal_input"]["path"]).resolve()
    if (
        reference_path.is_symlink()
        or not reference_path.is_file()
        or reference_path.stat().st_size != attempt["pixal_input"]["size_bytes"]
        or _sha256_file(reference_path) != attempt["pixal_input"]["sha256"]
    ):
        raise contracts.ContractError(f"approved FLUX/Pixal input changed: {instance_id}")
    contact_path = destination / "contact_sheet.png"
    build_contact_sheet(reference_path, views, contact_path)
    review: dict[str, Any] = {
        "schema": REVIEW_SCHEMA,
        "instance_id": instance_id,
        "request_sha256": attempt["request_sha256"],
        "profile_schema_id": attempt["profile_schema_id"],
        "sampled_attributes": attempt["sampled_attributes"],
        "target_physical_profile": attempt["target_physical_profile"],
        "front_axis": "negative-x",
        "up_axis": "positive-z",
        "pixal_output": attempt["output"],
        "mesh_readback": attempt["mesh_readback"],
        "reference_rgba": attempt["pixal_input"],
        "render_manifest": _relative(render_manifest_path, staging),
        "views": {
            view: _relative(views / f"{view}.png", staging) for view in VIEWS
        },
        "contact_sheet": _relative(contact_path, staging),
        "blender_log": _relative(log_path, staging),
        "automatic_checks": {
            "pixal_glb_reauthenticated": True,
            "reference_reauthenticated": True,
            "front_back_side_top_quarter_rendered": True,
            "pbr_material_and_texture_readback": True,
            "ue_animal_material_preview_applied": True,
            "overall": "passed",
        },
        "visual_qa": "pending",
        "next_gate": "agent_or_human_static_visual_decision",
    }
    review["review_sha256"] = _hash_without(review, "review_sha256")
    review_path = destination / "static_review_manifest.json"
    contracts.write_json_no_replace(review_path, review)
    return {
        "instance_id": instance_id,
        "request_sha256": attempt["request_sha256"],
        "review": _relative(review_path, staging),
        "review_sha256": review["review_sha256"],
        "contact_sheet": _relative(contact_path, staging),
        "status": "rendered_pending_visual_qa",
    }


def run_reviews(
    pixal_batch_path: Path, output_root: Path, workers: int
) -> Path:
    pixal_batch_path, batch = load_pixal_batch(pixal_batch_path)
    if not 1 <= workers <= 4:
        raise contracts.ContractError("workers must be in [1, 4]")
    if not BLENDER.is_file() or not RENDERER.is_file():
        raise contracts.ContractError("pinned Blender/static renderer is missing")
    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(f"refusing to replace output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    try:
        reviews = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_render_one, attempt, pixal_batch_path.parent, staging):
                attempt["instance_id"]
                for attempt in batch["attempts"]
            }
            for future in as_completed(futures):
                review = future.result()
                reviews.append(review)
                print(
                    "CONTROLLED_ANIMAL_STATIC_RENDERED "
                    f"instance={review['instance_id']}",
                    flush=True,
                )
        manifest: dict[str, Any] = {
            "schema": REVIEW_BATCH_SCHEMA,
            "status": "rendered_pending_visual_qa",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "pixal_batch": {
                "path": str(pixal_batch_path),
                "sha256": _sha256_file(pixal_batch_path),
                "batch_sha256": batch["batch_sha256"],
            },
            "review_count": len(reviews),
            "reviews": sorted(reviews, key=lambda item: item["instance_id"]),
            "automatic_checks": {
                "all_pixal_outputs_reauthenticated": True,
                "all_multiview_renders_passed": True,
                "all_visual_decisions_pending": True,
                "overall": "passed",
            },
        }
        manifest["review_batch_sha256"] = _hash_without(
            manifest, "review_batch_sha256"
        )
        contracts.write_json_no_replace(staging / "static_review_batch_manifest.json", manifest)
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError("static review output appeared concurrently")
        os.rename(staging, output_root)
        return output_root / "static_review_batch_manifest.json"
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixal-batch", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest = run_reviews(args.pixal_batch, args.output_root, args.workers)
    except (contracts.ContractError, OSError, subprocess.SubprocessError) as error:
        print(f"CONTROLLED_ANIMAL_STATIC_REVIEW_FAILED {error}", file=sys.stderr)
        return 2
    print(f"CONTROLLED_ANIMAL_STATIC_REVIEW_OK output={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
