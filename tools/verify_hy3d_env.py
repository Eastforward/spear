"""Gate 0 for the animated-dog-hunyuan-paint spec.

Confirms the local Hunyuan3D-Paint pipeline can be imported and
constructed with the local pretrained weights. Prints HY3D_ENV_OK on
success. Any other output = STOP, fix env before continuing.

Usage:
  HY3DGEN_MODELS=/data/jzy/code/Hunyuan3D-2.1/pretrained_models \\
  LD_LIBRARY_PATH=/data/jzy/miniconda3/envs/hunyuan3d/lib/python3.10/site-packages/torch/lib \\
  CUDA_VISIBLE_DEVICES=0 \\
    /data/jzy/miniconda3/envs/hunyuan3d/bin/python tools/verify_hy3d_env.py
"""
import os
import sys

HY3D_ROOT = "/data/jzy/code/Hunyuan3D-2.1"
REQUIRED_ENV = [
    "HY3DGEN_MODELS",
    "LD_LIBRARY_PATH",   # must contain torch/lib for libc10.so
]

for var in REQUIRED_ENV:
    if not os.environ.get(var):
        print(f"HY3D_ENV_FAIL missing env {var}")
        sys.exit(1)

sys.path.insert(0, HY3D_ROOT)
sys.path.insert(0, os.path.join(HY3D_ROOT, "hy3dshape"))
sys.path.insert(0, os.path.join(HY3D_ROOT, "hy3dpaint"))
os.chdir(HY3D_ROOT)  # required — pipeline uses relative paths for cfgs/ckpts

try:
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig  # noqa: F401
except ImportError as e:
    print(f"HY3D_ENV_FAIL import failed: {e}")
    sys.exit(1)

try:
    import custom_rasterizer as cr
    assert hasattr(cr, "rasterize"), "custom_rasterizer.rasterize missing"
except (ImportError, AssertionError) as e:
    print(f"HY3D_ENV_FAIL custom_rasterizer bad: {e}")
    sys.exit(1)

# Weights presence check (no HF download attempt)
expected_weights = os.path.join(
    os.environ["HY3DGEN_MODELS"], "tencent", "Hunyuan3D-2.1",
    "hunyuan3d-paintpbr-v2-1", "unet",
)
if not os.path.isdir(expected_weights):
    print(f"HY3D_ENV_FAIL weights not found at {expected_weights}")
    print("  Fix: create the tencent/Hunyuan3D-2.1 symlink pointing at pretrained_models/hunyuan3d-2.1")
    sys.exit(1)

print("HY3D_ENV_OK")
