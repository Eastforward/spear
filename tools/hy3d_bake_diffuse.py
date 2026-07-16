"""Run Hunyuan3D Paint with explicit, canonical local dependencies."""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
CONTRACT_DIR = SCRIPT_DIR / "spike_rlr"
if str(CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTRACT_DIR))

from hy3d_human_candidate import (  # noqa: E402
    CANONICAL_DINOV2_ROOT,
    CANONICAL_MODEL_PARENT,
    CANONICAL_MODEL_ROOT,
    CANONICAL_REALESRGAN_CKPT,
    WEIGHT_ROOT_HASH_MANIFEST,
    verify_weight_manifest,
)


CANONICAL_HY3D_ROOT = Path("/data/jzy/code/AVEngine/external/Hunyuan3D-2.1")
DINOV2_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
HY3D_ROOT = str(CANONICAL_HY3D_ROOT)
HY3D_CUSTOM_RASTERIZER_ROOT = CANONICAL_HY3D_ROOT / "hy3dpaint" / "custom_rasterizer"
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DIFFUSERS_OFFLINE": "1",
}

for import_root in (
    HY3D_CUSTOM_RASTERIZER_ROOT,
    CANONICAL_HY3D_ROOT,
    CANONICAL_HY3D_ROOT / "hy3dshape",
    CANONICAL_HY3D_ROOT / "hy3dpaint",
):
    sys.path.insert(0, str(import_root))


def _absolute_without_symlinks(path: Path, description: str) -> Path:
    provided = Path(path)
    if not provided.is_absolute():
        raise ValueError(f"{description} must be an absolute canonical path")
    absolute = provided.absolute()
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component) and stat.S_ISLNK(os.lstat(component).st_mode):
            raise ValueError(f"{description} path must not contain a symlink: {component}")
    return absolute


def _canonical_regular_file(path: Path, expected: Path, description: str) -> Path:
    absolute = _absolute_without_symlinks(path, description)
    expected = Path(expected).absolute()
    if absolute != expected:
        raise ValueError(f"{description} must be the canonical local path: {expected}")
    if not os.path.lexists(absolute) or not stat.S_ISREG(os.lstat(absolute).st_mode):
        raise FileNotFoundError(f"{description} is not a regular file: {absolute}")
    if absolute.stat().st_size <= 0 or absolute.resolve() != absolute:
        raise ValueError(f"{description} must be a non-empty regular file: {absolute}")
    return absolute


def _canonical_directory(path: Path, expected: Path, description: str) -> Path:
    absolute = _absolute_without_symlinks(path, description)
    expected = Path(expected).absolute()
    if absolute != expected:
        raise ValueError(f"{description} must be the canonical local path: {expected}")
    if not os.path.lexists(absolute) or not stat.S_ISDIR(os.lstat(absolute).st_mode):
        raise FileNotFoundError(f"{description} is not a directory: {absolute}")
    if absolute.resolve() != absolute:
        raise ValueError(f"{description} resolved path is not exact: {absolute}")
    return absolute


def _resolve_realesrgan_ckpt_path(explicit_path: Path | str | None = None) -> str:
    """Require the explicit canonical RealESRGAN checkpoint; never search."""
    if explicit_path is None:
        raise ValueError("--realesrgan-ckpt is required; fallback is forbidden")
    return str(
        _canonical_regular_file(
            Path(explicit_path), CANONICAL_REALESRGAN_CKPT, "RealESRGAN checkpoint"
        )
    )


def _resolve_dinov2_root(explicit_path: Path | str | None = None) -> str:
    """Require a complete explicit canonical DINOv2 directory."""
    if explicit_path is None:
        raise ValueError("--dinov2-root is required; HF cache fallback is forbidden")
    root = _canonical_directory(
        Path(explicit_path), CANONICAL_DINOV2_ROOT, "DINOv2 root"
    )
    for filename in DINOV2_FILES:
        _canonical_regular_file(root / filename, root / filename, f"DINOv2 {filename}")
    return str(root)


def _resolve_multiview_pretrained_path() -> str:
    """Return the absolute canonical model root for the local paint loader."""
    configured_parent = os.environ.get("HY3DGEN_MODELS")
    if not configured_parent:
        raise ValueError("HY3DGEN_MODELS must name the canonical local model parent")
    parent = _canonical_directory(
        Path(configured_parent), CANONICAL_MODEL_PARENT, "HY3DGEN_MODELS"
    )
    model_root = _canonical_directory(
        CANONICAL_MODEL_ROOT, CANONICAL_MODEL_ROOT, "canonical model root"
    )
    paint_index = model_root / "hunyuan3d-paintpbr-v2-1" / "model_index.json"
    _canonical_regular_file(paint_index, paint_index, "local paint model_index.json")
    if model_root.parent != parent:
        raise ValueError("local paint model escaped HY3DGEN_MODELS containment")
    return str(model_root)


def verify_canonical_weight_manifest(explicit_path: Path | str | None) -> str:
    """Re-hash the exact canonical manifest and every physical model file."""
    if explicit_path is None:
        raise ValueError("--weight-manifest is required; fallback is forbidden")
    manifest = _canonical_regular_file(
        Path(explicit_path),
        WEIGHT_ROOT_HASH_MANIFEST,
        "weight SHA-256 manifest",
    )
    return verify_weight_manifest(CANONICAL_MODEL_ROOT, manifest)


def configure_paint_dependencies(
    config: Any, realesrgan_ckpt: Path | str, dinov2_root: Path | str
) -> None:
    """Apply only verified absolute local dependency paths to a paint config."""
    config.realesrgan_ckpt_path = _resolve_realesrgan_ckpt_path(realesrgan_ckpt)
    config.dino_ckpt_path = _resolve_dinov2_root(dinov2_root)


def validate_paint_dependencies(
    realesrgan_ckpt: Path | str, dinov2_root: Path | str
) -> tuple[str, str]:
    """Return canonical dependency paths after complete local validation."""
    return (
        _resolve_realesrgan_ckpt_path(realesrgan_ckpt),
        _resolve_dinov2_root(dinov2_root),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glb", required=True)
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--realesrgan-ckpt", required=True, type=Path)
    parser.add_argument("--dinov2-root", required=True, type=Path)
    parser.add_argument("--weight-manifest", required=True, type=Path)
    parser.add_argument("--max-num-view", type=int, default=6, choices=[6, 7, 8, 9])
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--texture-size", type=int, default=4096)
    return parser.parse_args()


def _run(args: argparse.Namespace) -> None:
    os.environ.update(OFFLINE_ENVIRONMENT)
    configured_root = os.environ.get("HY3D_ROOT", str(CANONICAL_HY3D_ROOT))
    _canonical_directory(Path(configured_root), CANONICAL_HY3D_ROOT, "HY3D_ROOT")
    verify_canonical_weight_manifest(args.weight_manifest)
    validate_paint_dependencies(args.realesrgan_ckpt, args.dinov2_root)
    multiview_pretrained_path = _resolve_multiview_pretrained_path()

    workdir = Path(args.workdir).absolute()
    workdir.mkdir(parents=True, exist_ok=True)
    work_glb = workdir / "input.glb"
    shutil.copy2(args.input_glb, work_glb)
    output_obj = workdir / "hy3d_output_mesh.obj"

    from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline

    config = Hunyuan3DPaintConfig(
        max_num_view=args.max_num_view, resolution=args.resolution
    )
    configure_paint_dependencies(config, args.realesrgan_ckpt, args.dinov2_root)
    config.multiview_pretrained_path = multiview_pretrained_path
    config.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    config.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    config.texture_size = args.texture_size

    started = time.time()
    print(
        f"[hy3d_bake] loading pipeline ({args.max_num_view} views, "
        f"{args.resolution} res)...",
        flush=True,
    )
    pipeline = Hunyuan3DPaintPipeline(config)
    print(
        f"[hy3d_bake] pipeline loaded in {time.time() - started:.1f}s", flush=True
    )

    paint_started = time.time()
    pipeline(
        mesh_path=str(work_glb),
        image_path=str(Path(args.reference_image).absolute()),
        output_mesh_path=str(output_obj),
        use_remesh=True,
        save_glb=True,
    )
    elapsed = time.time() - paint_started

    for source_suffix, destination_name in (
        (".jpg", "hy3d_diffuse.jpg"),
        ("_metallic.jpg", "hy3d_metallic.jpg"),
        ("_roughness.jpg", "hy3d_roughness.jpg"),
    ):
        source = output_obj.with_suffix("")
        source = Path(f"{source}{source_suffix}")
        destination = workdir / destination_name
        if source.exists():
            shutil.move(source, destination)
    textured_obj = workdir / "hy3d_textured.obj"
    if output_obj.exists():
        shutil.move(output_obj, textured_obj)

    required = ("white_mesh_remesh.obj", "hy3d_textured.obj", "hy3d_diffuse.jpg")
    missing = [filename for filename in required if not (workdir / filename).exists()]
    if missing:
        raise RuntimeError(f"HY3D_BAKE_FAIL missing {missing} in {workdir}")
    if "\nvt " not in textured_obj.read_text(encoding="utf-8"):
        raise RuntimeError(f"HY3D_BAKE_FAIL {textured_obj} contains no vt (UV) lines")
    print(
        f"HY3D_BAKE_OK workdir={workdir} elapsed={elapsed:.1f}s "
        f"textured_obj={textured_obj}"
    )


def main() -> None:
    args = parse_args()
    previous_cwd = Path.cwd()
    try:
        os.chdir(CANONICAL_HY3D_ROOT)
        _run(args)
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
