"""Lifecycle contracts for the authenticated TokenRig warm-service helper."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys
import threading
import time


SCRIPT = Path(__file__).resolve().parents[2] / "tools/dev_warm_services.sh"


def _unused_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as stream:
        stream.bind(("127.0.0.1", 0))
        return int(stream.getsockname()[1])


def _run(environment: dict[str, str], action: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(SCRIPT), action],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=15,
    )


def _make_fake_skintokens(tmp_path: Path) -> Path:
    root = tmp_path / "SkinTokens"
    bin_dir = root / ".venv/bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(Path(sys.executable).resolve())
    (root / "numpy.py").write_text(
        """
class _Random:
    @staticmethod
    def seed(_value):
        pass

random = _Random()
""".lstrip(),
        encoding="utf-8",
    )
    (root / "torch.py").write_text(
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
    server = root / "bpy_server.py"
    server.write_text(
        """
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import sys

if os.environ.get("FAKE_TOKENRIG_EXIT"):
    print("FAKE_TOKENRIG_EXIT requested", flush=True)
    raise SystemExit(int(os.environ["FAKE_TOKENRIG_EXIT"]))

host = os.environ["TOKENRIG_BPY_BIND_HOST"]
port = int(os.environ["TOKENRIG_BPY_PORT"])
health = json.dumps(
    {
        "bind_host": host,
        "generation": os.environ["TOKENRIG_SERVICE_GENERATION"],
        "patch_sha256": os.environ["TOKENRIG_SERVER_HYGIENE_SHA256"],
        "pid": os.getpid(),
        "port": port,
        "schema": "avengine_tokenrig_warm_health_v1",
    },
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/avengine-health":
            body = health
            status = 200
        elif self.path == "/ping":
            body = b"pong"
            status = 200
        else:
            body = b"not found"
            status = 404
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass

print(f"FAKE_TOKENRIG_LISTENING {host}:{port}", flush=True)
HTTPServer((host, port), Handler).serve_forever()
""".lstrip(),
        encoding="utf-8",
    )
    server.chmod(0o755)
    return root


def _environment(tmp_path: Path, skintokens: Path, port: int) -> dict[str, str]:
    return {
        **os.environ,
        "AVENGINE_DEV_WARM_ROOT": str(tmp_path / "warm"),
        "AVENGINE_SKINTOKENS_ROOT": str(skintokens),
        "AVENGINE_WARM_TOKENRIG_PORT": str(port),
        "AVENGINE_WARM_TOKENRIG_GPU": "3",
    }


def _wait_ready(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    deadline = time.monotonic() + 8
    last = None
    while time.monotonic() < deadline:
        last = _run(environment, "ready")
        if last.returncode == 0:
            return last
        assert last.returncode in {2, 3}, (last.stdout, last.stderr)
        time.sleep(0.05)
    assert last is not None
    raise AssertionError(f"service did not become ready: {last.stdout} {last.stderr}")


def _state(environment: dict[str, str]) -> tuple[Path, dict[str, str]]:
    path = Path(environment["AVENGINE_DEV_WARM_ROOT"]) / "tokenrig/server.state"
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        values[key] = value
    return path, values


def _stop_if_owned(environment: dict[str, str]) -> None:
    state_path = Path(environment["AVENGINE_DEV_WARM_ROOT"]) / "tokenrig/server.state"
    if state_path.exists():
        result = _run(environment, "stop")
        assert result.returncode == 0, (result.stdout, result.stderr)


def test_isolated_status_is_nonzero_down_without_creating_state(tmp_path):
    environment = {
        **os.environ,
        "AVENGINE_DEV_WARM_ROOT": str(tmp_path / "warm"),
        "AVENGINE_WARM_TOKENRIG_PORT": str(_unused_loopback_port()),
    }

    result = _run(environment, "status")

    assert result.returncode == 1
    assert result.stdout.strip() == "tokenrig: DOWN"
    assert not (tmp_path / "warm").exists()


def test_invalid_port_fails_before_service_probe(tmp_path):
    result = _run(
        {
            **os.environ,
            "AVENGINE_DEV_WARM_ROOT": str(tmp_path / "warm"),
            "AVENGINE_WARM_TOKENRIG_PORT": "1",
        },
        "status",
    )

    assert result.returncode == 2
    assert "invalid port" in result.stderr


def test_start_ready_status_and_stop_authenticate_one_private_generation(tmp_path):
    skintokens = _make_fake_skintokens(tmp_path)
    environment = _environment(tmp_path, skintokens, _unused_loopback_port())
    try:
        started = _run(environment, "start")
        assert started.returncode == 0, (started.stdout, started.stderr)
        ready = _wait_ready(environment)
        assert "tokenrig: READY generation=" in ready.stdout

        state_path, state = _state(environment)
        pid = int(state["pid"])
        assert len(state["generation"]) == 32
        assert state["patch_sha256"] in ready.stdout or state["generation"] in ready.stdout
        assert state["bind_host"] == "127.0.0.1"
        assert state["port"] == environment["AVENGINE_WARM_TOKENRIG_PORT"]
        assert Path(f"/proc/{pid}").is_dir()

        socket_result = subprocess.run(
            ["ss", "-H", "-ltnp", f"sport = :{state['port']}"],
            check=True,
            capture_output=True,
            text=True,
        )
        assert f"127.0.0.1:{state['port']}" in socket_result.stdout
        assert f"0.0.0.0:{state['port']}" not in socket_result.stdout
        assert f"pid={pid}," in socket_result.stdout

        warm_root = Path(environment["AVENGINE_DEV_WARM_ROOT"])
        for directory in (
            warm_root,
            warm_root / "tokenrig",
            Path(state["generation_dir"]),
            Path(state["marker_dir"]),
        ):
            assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(state_path.stat().st_mode) == 0o600
        assert stat.S_IMODE(Path(state["log_path"]).stat().st_mode) == 0o600
        assert not list((warm_root / "tokenrig").glob(".server.state.*"))

        changed = {**environment, "AVENGINE_WARM_TOKENRIG_GPU": "4"}
        stale = _run(changed, "status")
        assert stale.returncode == 3
        assert "configuration changed" in stale.stdout

        # stop authenticates the recorded generation, not the caller's current
        # GPU preference, so an old but exact generation remains manageable.
        stopped = _run(changed, "stop")
        assert stopped.returncode == 0, (stopped.stdout, stopped.stderr)
        assert not Path(f"/proc/{pid}").exists()
    finally:
        _stop_if_owned(environment)


def test_plain_pong_listener_is_external_and_never_reused_or_stopped(tmp_path):
    port = _unused_loopback_port()

    class PongHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b"pong"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), PongHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    skintokens = _make_fake_skintokens(tmp_path)
    environment = _environment(tmp_path, skintokens, port)
    try:
        status = _run(environment, "status")
        assert status.returncode == 3
        assert "STALE/EXTERNAL" in status.stdout

        start = _run(environment, "start")
        assert start.returncode == 3
        assert "unauthenticated listener" in start.stderr

        stop = _run(environment, "stop")
        assert stop.returncode == 3
        assert "refusing to stop" in stop.stderr
        assert thread.is_alive()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_concurrent_start_publishes_only_one_generation(tmp_path):
    skintokens = _make_fake_skintokens(tmp_path)
    environment = _environment(tmp_path, skintokens, _unused_loopback_port())
    processes = [
        subprocess.Popen(
            [str(SCRIPT), "start"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        for _ in range(2)
    ]
    try:
        results = [process.communicate(timeout=15) + (process.returncode,) for process in processes]
        assert [result[2] for result in results] == [0, 0], results
        _wait_ready(environment)
        generation_root = (
            Path(environment["AVENGINE_DEV_WARM_ROOT"]) / "tokenrig/generations"
        )
        assert len([path for path in generation_root.iterdir() if path.is_dir()]) == 1
    finally:
        _stop_if_owned(environment)


def test_immediate_child_exit_is_nonzero_and_preserves_generation_log(tmp_path):
    skintokens = _make_fake_skintokens(tmp_path)
    environment = {
        **_environment(tmp_path, skintokens, _unused_loopback_port()),
        "FAKE_TOKENRIG_EXIT": "23",
    }

    result = _run(environment, "start")

    assert result.returncode == 1
    assert "exited" in result.stderr
    logs = list(
        (Path(environment["AVENGINE_DEV_WARM_ROOT"]) / "tokenrig/generations").glob(
            "*/server.log"
        )
    )
    assert len(logs) == 1
    assert "FAKE_TOKENRIG_EXIT requested" in logs[0].read_text(encoding="utf-8")
    assert not (
        Path(environment["AVENGINE_DEV_WARM_ROOT"]) / "tokenrig/server.state"
    ).exists()


def test_stop_refuses_tampered_pid_without_signalling_it(tmp_path):
    skintokens = _make_fake_skintokens(tmp_path)
    environment = _environment(tmp_path, skintokens, _unused_loopback_port())
    sleeper = subprocess.Popen(["sleep", "30"])
    try:
        assert _run(environment, "start").returncode == 0
        _wait_ready(environment)
        state_path, _ = _state(environment)
        original = state_path.read_text(encoding="utf-8")
        sleeper_starttime = Path(f"/proc/{sleeper.pid}/stat").read_text(
            encoding="utf-8"
        ).split()[21]
        tampered = []
        for line in original.splitlines():
            if line.startswith("pid="):
                line = f"pid={sleeper.pid}"
            elif line.startswith("starttime="):
                line = f"starttime={sleeper_starttime}"
            tampered.append(line)
        state_path.write_text("\n".join(tampered) + "\n", encoding="utf-8")
        state_path.chmod(0o600)

        result = _run(environment, "stop")

        assert result.returncode == 3
        assert "refusing to signal" in result.stderr
        assert sleeper.poll() is None
        state_path.write_text(original, encoding="utf-8")
        state_path.chmod(0o600)
    finally:
        _stop_if_owned(environment)
        sleeper.terminate()
        sleeper.wait(timeout=2)


def test_script_contains_no_broad_process_kill_or_eval():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "flock -x" in text
    assert "write_state_atomic" in text
    assert "process_matches_loaded_state" in text
    assert "pkill" not in text
    assert "killall" not in text
    assert "eval " not in text
