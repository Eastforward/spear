import subprocess
import sys
from pathlib import Path
import os


REPO = Path(__file__).resolve().parents[2]


def test_hy3d_bake_diffuse_resolves_hunyuan_sibling_in_monorepo():
    code = f"""
import sys
sys.path.insert(0, {str(REPO / "tools")!r})
import hy3d_bake_diffuse
print(hy3d_bake_diffuse.HY3D_ROOT)
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(REPO.parent / "Hunyuan3D-2.1")


def test_hy3d_bake_diffuse_prefers_root_realesrgan_checkpoint():
    code = f"""
import sys
sys.path.insert(0, {str(REPO / "tools")!r})
import hy3d_bake_diffuse
print(hy3d_bake_diffuse._resolve_realesrgan_ckpt_path())
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == str(
        REPO.parent / "Hunyuan3D-2.1" / "ckpt" / "RealESRGAN_x4plus.pth"
    )


def test_hy3d_bake_diffuse_prefers_local_paint_pretrained_path():
    code = f"""
import sys
sys.path.insert(0, {str(REPO / "tools")!r})
import hy3d_bake_diffuse
print(hy3d_bake_diffuse._resolve_multiview_pretrained_path())
"""
    env = {
        **os.environ,
        "HY3DGEN_MODELS": str(REPO.parent / "Hunyuan3D-2.1" / "pretrained_models"),
    }
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "hunyuan3d-2.1"


def test_hy3d_bake_diffuse_prioritizes_local_custom_rasterizer_path():
    code = f"""
import sys
sys.path.insert(0, {str(REPO / "tools")!r})
import hy3d_bake_diffuse
print(sys.path.index(str(hy3d_bake_diffuse.HY3D_CUSTOM_RASTERIZER_ROOT)))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert int(proc.stdout.strip()) < 4
