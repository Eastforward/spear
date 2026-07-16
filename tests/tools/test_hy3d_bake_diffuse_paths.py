import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "hy3d_bake_diffuse.py"
MULTIVIEW_UTILS = (
    REPO.parent / "Hunyuan3D-2.1" / "hy3dpaint" / "utils" / "multiview_utils.py"
)


def _load_wrapper():
    spec = importlib.util.spec_from_file_location("hy3d_bake_diffuse_paths", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_manifest(model_root: Path, manifest: Path) -> None:
    lines = []
    for path in sorted(candidate for candidate in model_root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(model_root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  ./{relative}\n")
    manifest.write_text("".join(lines), encoding="utf-8")


def _model_fixture(root: Path) -> tuple[Path, Path]:
    model_root = root / "hunyuan3d-2.1"
    required = (
        "hunyuan3d-dit-v2-1/config.yaml",
        "hunyuan3d-dit-v2-1/model.fp16.ckpt",
        "hunyuan3d-paintpbr-v2-1/model_index.json",
        "hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin",
        "hunyuan3d-vae-v2-1/config.yaml",
        "hunyuan3d-vae-v2-1/model.fp16.ckpt",
        "dependencies/realesrgan/RealESRGAN_x4plus.pth",
        "dependencies/dinov2-giant/config.json",
        "dependencies/dinov2-giant/preprocessor_config.json",
        "dependencies/dinov2-giant/model.safetensors",
    )
    for relative in required:
        path = model_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    manifest = root / "weights.sha256"
    _write_manifest(model_root, manifest)
    return model_root, manifest


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


def test_hy3d_bake_diffuse_cli_requires_canonical_weight_arguments(
    tmp_path, monkeypatch
):
    wrapper = _load_wrapper()
    base = [
        "hy3d_bake_diffuse.py",
        "--input-glb",
        str(tmp_path / "shape.glb"),
        "--reference-image",
        str(tmp_path / "reference.png"),
        "--workdir",
        str(tmp_path / "work"),
        "--realesrgan-ckpt",
        str(wrapper.CANONICAL_REALESRGAN_CKPT),
        "--dinov2-root",
        str(wrapper.CANONICAL_DINOV2_ROOT),
    ]
    monkeypatch.setattr(sys, "argv", base)

    with pytest.raises(SystemExit):
        wrapper.parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        base + ["--weight-manifest", str(wrapper.WEIGHT_ROOT_HASH_MANIFEST)],
    )
    args = wrapper.parse_args()
    assert args.weight_manifest == wrapper.WEIGHT_ROOT_HASH_MANIFEST


def test_hy3d_bake_diffuse_reverifies_exact_canonical_manifest(
    tmp_path, monkeypatch
):
    wrapper = _load_wrapper()
    model_root, manifest = _model_fixture(tmp_path / "models")
    monkeypatch.setattr(wrapper, "CANONICAL_MODEL_ROOT", model_root)
    monkeypatch.setattr(wrapper, "WEIGHT_ROOT_HASH_MANIFEST", manifest)

    assert wrapper.verify_canonical_weight_manifest(manifest) == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()

    alternate = tmp_path / "alternate.sha256"
    alternate.write_bytes(manifest.read_bytes())
    with pytest.raises(ValueError, match="canonical"):
        wrapper.verify_canonical_weight_manifest(alternate)

    (model_root / "dependencies/dinov2-giant/model.safetensors").write_bytes(
        b"changed"
    )
    with pytest.raises(RuntimeError, match="SHA-256|hash"):
        wrapper.verify_canonical_weight_manifest(manifest)


def test_hy3d_bake_diffuse_uses_absolute_canonical_multiview_root(
    tmp_path, monkeypatch
):
    wrapper = _load_wrapper()
    model_root, _ = _model_fixture(tmp_path / "models")
    monkeypatch.setattr(wrapper, "CANONICAL_MODEL_PARENT", model_root.parent)
    monkeypatch.setattr(wrapper, "CANONICAL_MODEL_ROOT", model_root)
    monkeypatch.setenv("HY3DGEN_MODELS", str(model_root.parent))

    resolved = wrapper._resolve_multiview_pretrained_path()

    assert resolved == str(model_root)
    assert Path(resolved).is_absolute()


def test_checkout_multiview_loader_has_no_repo_or_cache_fallback():
    source = MULTIVIEW_UTILS.read_text(encoding="utf-8")
    compact = "".join(source.split())

    assert "huggingface_hub" not in source
    assert "snapshot_download" not in source
    assert "local_files_only=True" in compact
    assert "os.path.isabs" in source
    assert "hunyuan3d-paintpbr-v2-1" in source


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
