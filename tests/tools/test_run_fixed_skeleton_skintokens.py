import json
from pathlib import Path
import struct

import pytest

from tools import run_fixed_skeleton_skintokens as runner


def _write_glb(path, payload):
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((4 - len(raw) % 4) % 4)
    total = 12 + 8 + len(raw)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(raw), runner.GLB_JSON_CHUNK)
        + raw
    )


def test_glb_summary_requires_one_skin_mesh_pbr_and_no_animation(tmp_path):
    valid = tmp_path / "valid.glb"
    _write_glb(
        valid,
        {
            "asset": {"version": "2.0"},
            "meshes": [{}],
            "skins": [{}],
            "nodes": [{}, {}],
            "materials": [{}],
            "images": [{}],
        },
    )
    assert runner.glb_summary(valid)["skins"] == 1

    animated = tmp_path / "animated.glb"
    _write_glb(
        animated,
        {
            "asset": {"version": "2.0"},
            "meshes": [{}],
            "skins": [{}],
            "nodes": [{}, {}],
            "materials": [{}],
            "images": [{}],
            "animations": [{}],
        },
    )
    with pytest.raises(RuntimeError, match="must not contain animation"):
        runner.glb_summary(animated)


def test_runner_uses_fixed_skeleton_mode_and_immutable_ledger():
    text = Path(runner.__file__).read_text(encoding="utf-8")

    assert '"--use_skeleton"' in text
    assert '"--use_transfer"' in text
    assert '"mode": "fixed_skeleton_generate_skin_only"' in text
    assert 'log_path.open("x"' in text
    assert 'path.open("x"' in text
    assert "refusing to replace output directory" in text
    assert '"TOKENRIG_BPY_PORT"' in text
    assert '"bpy_port": bpy_port' in text


def test_loopback_port_allocator_returns_unprivileged_port():
    port = runner.allocate_loopback_port()
    assert 1024 <= port <= 65535


def test_venv_python_accepts_pinned_launcher_symlink(tmp_path, monkeypatch):
    root = tmp_path / "SkinTokens"
    bin_dir = root / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    real = tmp_path / "cpython"
    real.write_bytes(b"python")
    real.chmod(0o755)
    launcher = bin_dir / "python"
    launcher.symlink_to(real)
    monkeypatch.setattr(runner, "SKINTOKENS_ROOT", root)

    assert runner.require_venv_python(launcher) == launcher.absolute()


def test_venv_python_rejects_launcher_outside_pinned_venv(tmp_path, monkeypatch):
    root = tmp_path / "SkinTokens"
    root.mkdir()
    launcher = tmp_path / "python"
    launcher.write_bytes(b"python")
    launcher.chmod(0o755)
    monkeypatch.setattr(runner, "SKINTOKENS_ROOT", root)

    with pytest.raises(SystemExit, match="escaped its pinned venv"):
        runner.require_venv_python(launcher)


def test_pinned_model_file_requires_exact_logical_checkpoint(tmp_path, monkeypatch):
    expected = tmp_path / "checkpoint.ckpt"
    other = tmp_path / "other.ckpt"
    other.write_bytes(b"weights")
    monkeypatch.setattr(runner, "CHECKPOINT", expected)

    with pytest.raises(SystemExit, match="path is not pinned"):
        runner.require_pinned_model_file(other)
