import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time

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


def test_runtime_hygiene_forces_the_unauthenticated_bpy_api_to_loopback():
    text = runner.PATCH_PATH.read_text(encoding="utf-8")

    assert '_bpy_bind_host = "127.0.0.1"' in text
    assert "def _install_bottle_loopback():" in text
    assert 'kwargs["host"] = _bpy_bind_host' in text
    assert '"bpy_bind_host": _bpy_bind_host' in text
    assert '"/avengine-health"' in text
    assert "0.0.0.0" not in text


def test_runtime_hygiene_really_binds_loopback_and_publishes_generation_health(
    tmp_path,
):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "numpy.py").write_text(
        """
class _Random:
    @staticmethod
    def seed(_value):
        pass

random = _Random()
""".lstrip(),
        encoding="utf-8",
    )
    (stubs / "torch.py").write_text(
        """
def manual_seed(_value):
    pass

class _Cuda:
    @staticmethod
    def manual_seed_all(_value):
        pass

cuda = _Cuda()
""".lstrip(),
        encoding="utf-8",
    )
    evidence = tmp_path / "bind_evidence.json"
    (stubs / "bottle.py").write_text(
        """
import json
import os
from pathlib import Path
import socket

def run(app=None, server="unused", host="127.0.0.1", port=8080, **_kwargs):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen()
    Path(os.environ["BIND_EVIDENCE"]).write_text(
        json.dumps(
            {
                "host": listener.getsockname()[0],
                "health": app.routes["/avengine-health"](),
            }
        ),
        encoding="utf-8",
    )
    while True:
        connection, _address = listener.accept()
        connection.close()
""".lstrip(),
        encoding="utf-8",
    )
    generation = "a" * 32
    patch_sha = hashlib.sha256(runner.PATCH_PATH.read_bytes()).hexdigest()
    port = runner.allocate_loopback_port()
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(runner.PATCH_DIR), str(stubs))),
        "TOKENRIG_CANARY_SEED": "42",
        "TOKENRIG_SERVICE_GENERATION": generation,
        "TOKENRIG_LOAD_AUDIT_PATH": str(tmp_path / "audit.jsonl"),
        "TOKENRIG_HYGIENE_MARKER_DIR": str(marker_dir),
        "TOKENRIG_SERVER_HYGIENE_SHA256": patch_sha,
        "TOKENRIG_BPY_BIND_HOST": "127.0.0.1",
        "TOKENRIG_BPY_PORT": str(port),
        "BIND_EVIDENCE": str(evidence),
    }
    command = """
import bottle

class App:
    def __init__(self):
        self.routes = {}

    def route(self, path, method="GET", callback=None):
        self.routes[path] = callback

bottle.run(App(), host="0.0.0.0", port=int(__import__("os").environ["TOKENRIG_BPY_PORT"]))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", command],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not evidence.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise AssertionError(
                    f"patched bottle process exited: {process.returncode}\n{stdout}\n{stderr}"
                )
            time.sleep(0.02)
        assert evidence.exists()
        payload = json.loads(evidence.read_text(encoding="utf-8"))
        assert payload["host"] == "127.0.0.1"
        health = json.loads(payload["health"])
        assert health == {
            "bind_host": "127.0.0.1",
            "generation": generation,
            "patch_sha256": patch_sha,
            "pid": process.pid,
            "port": port,
            "schema": "avengine_tokenrig_warm_health_v1",
        }
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass
        marker = json.loads(next(marker_dir.glob("*.json")).read_text(encoding="utf-8"))
        assert marker["generation"] == generation
        assert marker["bpy_bind_host"] == "127.0.0.1"
    finally:
        process.terminate()
        process.wait(timeout=3)


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
