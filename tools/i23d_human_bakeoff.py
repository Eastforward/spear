#!/usr/bin/env python3

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image


MODEL_SPECS = {
    "trellis2": {
        "root": Path("/data/models/hub/models--microsoft--TRELLIS.2-4B"),
        "revision": "af44b45f2e35a493886929c6d786e563ec68364d",
        "required": [
            "pipeline.json",
            "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
            "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
            "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
            "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_512_bf16.safetensors",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
        ],
    },
    "pixal3d": {
        "root": Path("/data/models/hub/models--TencentARC--Pixal3D"),
        "revision": "0b31f9160aa400719af409098bff7936a932f726",
        "required": [
            "pipeline.json",
            "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
            "ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors",
            "ckpts/shape_dec_next_dc_f16c32_fp16.safetensors",
            "ckpts/slat_flow_img2shape_dit_1_3B_512_bf16.safetensors",
            "ckpts/slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors",
            "ckpts/tex_dec_next_dc_f16c32_fp16.safetensors",
            "ckpts/slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors",
        ],
    },
}

DINO_SPEC = {
    "root": Path(
        "/data/models/hub/models--camenduru--dinov3-vitl16-pretrain-lvd1689m"
    ),
    "revision": "3c276edd87d6f6e569ff0c4400e086807d0f3881",
    "required": ["config.json", "model.safetensors", "LICENSE.md"],
}

EXTERNAL_ROOT = Path(__file__).resolve().parents[2]
TRELLIS_ROOT = EXTERNAL_ROOT / "TRELLIS.2"
PIXAL_ROOT = EXTERNAL_ROOT / "Pixal3D"


def verify_pinned_file(path, expected_sha256):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Pinned file is missing: {path}")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"Pinned file SHA-256 mismatch for {path}: {actual} != {expected_sha256}"
        )
    return path


def patch_trellis_conditioning(extractor_module, rembg_module, dino_snapshot):
    original_extractor = extractor_module.DinoV3FeatureExtractor
    pinned_model = str(dino_snapshot)

    class PinnedDinoV3FeatureExtractor(original_extractor):
        def __init__(self, *args, **kwargs):
            kwargs["model_name"] = pinned_model
            super().__init__(*args, **kwargs)

    extractor_module.DinoV3FeatureExtractor = PinnedDinoV3FeatureExtractor
    rembg_module.BiRefNet = lambda *args, **kwargs: None


def patch_pixal_conditioning(rembg_module, image_cond_configs, dino_snapshot):
    pinned_model = str(dino_snapshot)
    for config in image_cond_configs.values():
        config["model_name"] = pinned_model
    rembg_module.BiRefNet = lambda *args, **kwargs: None


def resolve_backend_assets(backend):
    if backend not in MODEL_SPECS:
        raise ValueError(f"Unsupported backend: {backend}")
    model = MODEL_SPECS[backend]
    return {
        "model": resolve_snapshot(
            model["root"], model["revision"], model["required"]
        ),
        "dino": resolve_snapshot(
            DINO_SPEC["root"], DINO_SPEC["revision"], DINO_SPEC["required"]
        ),
    }


def build_runtime_env(backend, gpu, base_env=None):
    attention_backends = {"trellis2": "xformers", "pixal3d": "sdpa"}
    if backend not in attention_backends:
        raise ValueError(f"Unsupported backend: {backend}")
    env = dict(os.environ if base_env is None else base_env)
    env.update(
        {
            "ATTN_BACKEND": attention_backends[backend],
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "HF_HUB_CACHE": "/data/models/hub",
            "HF_HUB_OFFLINE": "1",
            "OPENCV_IO_ENABLE_OPENEXR": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_HOME": "/data/models/torch",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    return env


def resolve_snapshot(model_root, revision, required_files):
    model_root = Path(model_root)
    snapshot = model_root / "snapshots" / revision
    if model_root.is_symlink() or snapshot.is_symlink():
        raise ValueError("Model cache root and pinned snapshot must not be symlinks")
    if not snapshot.is_dir():
        raise ValueError(f"Pinned snapshot is missing: {snapshot}")
    incomplete = sorted(model_root.rglob("*.incomplete"))
    if incomplete:
        raise ValueError(f"Model cache contains incomplete files: {incomplete[0]}")
    for relative_path in required_files:
        path = snapshot / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Pinned snapshot required file is missing: {path}")
    return snapshot


def inspect_rgba_input(path):
    path = Path(path)
    with Image.open(path) as image:
        if image.mode != "RGBA":
            raise ValueError(f"Input must be a transparent RGBA image: {path}")
        alpha = np.asarray(image.getchannel("A"))
        alpha_min = int(alpha.min())
        alpha_max = int(alpha.max())
        if alpha_min == 255 or alpha_max == 0:
            raise ValueError(f"Input must be a transparent RGBA image: {path}")
        size = list(image.size)

    return {
        "mode": "RGBA",
        "size": size,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _import_trellis_runtime():
    sys.path.insert(0, str(TRELLIS_ROOT))
    import o_voxel
    import torch
    from trellis2.modules import image_feature_extractor
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.pipelines import rembg

    return SimpleNamespace(
        extractor_module=image_feature_extractor,
        o_voxel=o_voxel,
        pipeline_class=Trellis2ImageTo3DPipeline,
        rembg_module=rembg,
        torch=torch,
    )


def run_trellis2(
    *,
    image_path,
    output_path,
    model_snapshot,
    dino_snapshot,
    seed,
    resolution,
    low_vram,
):
    if resolution not in {1024, 1536}:
        raise ValueError("TRELLIS.2 resolution must be 1024 or 1536")
    runtime = _import_trellis_runtime()
    patch_trellis_conditioning(
        runtime.extractor_module, runtime.rembg_module, dino_snapshot
    )
    pipeline = runtime.pipeline_class.from_pretrained(str(model_snapshot))
    if low_vram:
        pipeline.low_vram = True
        pipeline._device = runtime.torch.device("cuda")
    else:
        pipeline.low_vram = False
        pipeline.cuda()

    with Image.open(image_path) as source:
        image = source.copy()
    mesh = pipeline.run(
        image, seed=seed, pipeline_type=f"{resolution}_cascade"
    )[0]
    glb = runtime.o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=1_000_000,
        texture_size=4096,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(str(output_path), extension_webp=True)


def _import_pixal_runtime():
    sys.path.insert(0, str(PIXAL_ROOT))
    from pixal3d.pipelines import rembg

    spec = importlib.util.spec_from_file_location(
        "avengine_pixal3d_inference", PIXAL_ROOT / "inference.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load Pixal3D inference module")
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)
    return SimpleNamespace(inference=inference, rembg_module=rembg)


def run_pixal3d(
    *,
    image_path,
    output_path,
    model_snapshot,
    dino_snapshot,
    seed,
    resolution,
    manual_fov,
    low_vram,
):
    if resolution not in {1024, 1536}:
        raise ValueError("Pixal3D resolution must be 1024 or 1536")
    if manual_fov <= 0:
        raise ValueError("Pixal3D requires a positive manual FOV")
    runtime = _import_pixal_runtime()
    patch_pixal_conditioning(
        runtime.rembg_module,
        runtime.inference.IMAGE_COND_CONFIGS,
        dino_snapshot,
    )
    runtime.inference.run_inference(
        image_path=str(image_path),
        output_path=str(output_path),
        seed=seed,
        manual_fov=manual_fov,
        model_path=str(model_snapshot),
        low_vram=low_vram,
        resolution=resolution,
    )


def execute_job(
    *,
    backend,
    image_path,
    output_path,
    seed,
    resolution,
    manual_fov,
    low_vram,
):
    total_started = time.perf_counter()
    image_path = Path(image_path)
    output_path = Path(output_path)
    input_metadata = inspect_rgba_input(image_path)
    assets = resolve_backend_assets(backend)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    common = {
        "image_path": image_path,
        "output_path": output_path,
        "model_snapshot": assets["model"],
        "dino_snapshot": assets["dino"],
        "seed": seed,
        "resolution": resolution,
        "low_vram": low_vram,
    }
    inference_started = time.perf_counter()
    if backend == "trellis2":
        run_trellis2(**common)
    elif backend == "pixal3d":
        run_pixal3d(**common, manual_fov=manual_fov)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    inference_seconds = time.perf_counter() - inference_started

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Inference did not create a non-empty output: {output_path}")

    manifest = {
        "backend": backend,
        "input": {"path": str(image_path.resolve()), **input_metadata},
        "output": {
            "bytes": output_path.stat().st_size,
            "path": str(output_path.resolve()),
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        },
        "model": {
            "snapshot": str(assets["model"]),
            "revision": MODEL_SPECS[backend]["revision"],
        },
        "dino": {
            "snapshot": str(assets["dino"]),
            "revision": DINO_SPEC["revision"],
        },
        "parameters": {
            "low_vram": low_vram,
            "manual_fov": manual_fov,
            "resolution": resolution,
            "seed": seed,
        },
        "timings": {
            "inference_seconds": inference_seconds,
            "total_before_manifest_seconds": time.perf_counter() - total_started,
        },
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the pinned TRELLIS.2/Pixal3D human image-to-3D bake-off"
    )
    parser.add_argument("--backend", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, choices=(1024, 1536), default=1024)
    parser.add_argument("--manual-fov", type=float, default=0.2)
    parser.add_argument("--low-vram", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ.update(build_runtime_env(args.backend, args.gpu))
    manifest = execute_job(
        backend=args.backend,
        image_path=args.image,
        output_path=args.output,
        seed=args.seed,
        resolution=args.resolution,
        manual_fov=args.manual_fov,
        low_vram=args.low_vram,
    )
    print(manifest)
    return manifest


if __name__ == "__main__":
    main()
