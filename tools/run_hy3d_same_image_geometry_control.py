#!/usr/bin/env python3
"""Run a Hunyuan3D-Shape technical control on an exact Pixal input image.

This intentionally skips FLUX and background removal.  The input RGBA is
copied byte-for-byte so a Pixal/Hunyuan comparison changes only the image-to-3D
backend.  Outputs are permanently classified as ``technical_spike_only`` and
must never be registered as formal AVEngine dataset assets.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = SCRIPT_DIR.parent
TMP_ROOT = SPEAR_ROOT / "tmp"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import hy3d_generate_human_candidates as hy3d  # noqa: E402


SCHEMA = "avengine_hy3d_same_image_geometry_control_v1"
CONTROL_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,127}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return hashlib.sha256(
        _canonical_json(
            {name: copy.deepcopy(item) for name, item in value.items() if name != key}
        ).encode("utf-8")
    ).hexdigest()


def _git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _validate_input(path: Path) -> tuple[Path, dict[str, Any]]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"input image must be a non-empty regular file: {path}")
    with Image.open(path) as image:
        image.load()
        if image.mode != "RGBA":
            raise ValueError(f"control input must be the exact segmented RGBA: {image.mode}")
        width, height = image.size
        alpha = image.getchannel("A")
        alpha_extrema = alpha.getextrema()
        if alpha_extrema is None or alpha_extrema[0] == alpha_extrema[1]:
            raise ValueError("control input alpha channel must contain foreground/background")
    return path, {
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "mode": "RGBA",
        "width": width,
        "height": height,
        "alpha_extrema": list(alpha_extrema),
    }


def _prepare_output(output_root: Path, control_id: str) -> Path:
    if not CONTROL_ID_RE.fullmatch(control_id):
        raise ValueError(f"invalid control_id: {control_id!r}")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        output_root.relative_to(TMP_ROOT.resolve())
    except ValueError as error:
        raise ValueError("output_root must remain under external/SPEAR/tmp") from error
    asset_dir = output_root / control_id
    asset_dir.mkdir(exist_ok=False)
    return asset_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-image", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--control-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps != 50 or args.guidance_scale != 5.0:
        raise ValueError("the historical Hunyuan control contract is fixed at 50 steps/guidance 5")
    source, input_record = _validate_input(args.input_image)
    asset_dir = _prepare_output(args.output_root, args.control_id)
    copied_input = asset_dir / "input_rgba_exact.png"
    shutil.copy2(source, copied_input)
    if _sha256_file(copied_input) != input_record["sha256"]:
        raise RuntimeError("byte-for-byte input copy changed")

    shape_path = asset_dir / "shape.glb"
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    pipeline = hy3d.load_shape_pipeline()
    model_load_seconds = time.perf_counter() - start
    inference_start = time.perf_counter()
    hy3d.generate_shape(
        pipeline,
        {
            "seed": args.seed,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
        },
        copied_input,
        shape_path,
    )
    inference_seconds = time.perf_counter() - inference_start
    if not shape_path.is_file() or shape_path.stat().st_size <= 0:
        raise RuntimeError("Hunyuan shape output is missing")

    import torch

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "control_id": args.control_id,
        "state_classification": "technical_spike_only",
        "formal_dataset_registration_authorized": False,
        "comparison_contract": {
            "controlled_variable": "image_to_3d_backend",
            "fixed_upstream_image": "exact_pixal_isnet_rgba_byte_copy",
            "flux_rerun": False,
            "background_removal_rerun": False,
            "paint_stage_run": False,
        },
        "input": {
            "source_absolute_path": str(source),
            "copied_absolute_path": str(copied_input.resolve()),
            **input_record,
        },
        "hunyuan": {
            "checkout": str(hy3d.HUNYUAN_CHECKOUT),
            "git_head": _git_head(hy3d.HUNYUAN_CHECKOUT),
            "model_root": str(hy3d.CANONICAL_MODEL_ROOT),
            "weight_manifest": str(hy3d.WEIGHT_ROOT_HASH_MANIFEST),
            "weight_manifest_sha256": hy3d.current_weight_manifest_sha256(),
            "local_files_only": True,
            "offline": True,
            "seed": args.seed,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
        },
        "runtime": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "logical_cuda_device": 0,
            "gpu_name": torch.cuda.get_device_name(0),
            "python": sys.executable,
        },
        "output": {
            "shape_glb": str(shape_path.resolve()),
            "sha256": _sha256_file(shape_path),
            "size_bytes": shape_path.stat().st_size,
        },
        "timings_seconds": {
            "model_load": round(model_load_seconds, 6),
            "shape_inference": round(inference_seconds, 6),
            "total": round(time.perf_counter() - start, 6),
        },
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "license_policy": {
            "hunyuan_output_use": "comparison_and_visual_debug_only",
            "training_or_evaluation_use_prohibited": True,
        },
    }
    manifest["manifest_sha256"] = _hash_without(manifest, "manifest_sha256")
    manifest_path = asset_dir / "control_manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        "HY3D_SAME_IMAGE_GEOMETRY_CONTROL_OK "
        f"control_id={args.control_id} manifest={manifest_path.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
