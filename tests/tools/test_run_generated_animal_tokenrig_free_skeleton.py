#
# Copyright (c) 2025 The SPEAR Development Team
# Copyright (c) 2022 Intel
#

import argparse
import ast
import json
from pathlib import Path
import shlex
import signal
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import generated_animal_tokenrig_closure as closure
from tools import run_generated_animal_tokenrig_free_skeleton as runner


def production_argv(tmp_path):
    return [
        "--input",
        str(tmp_path / "input.glb"),
        "--output",
        str(tmp_path / "output.glb"),
        "--top_k",
        "5",
        "--top_p",
        "0.95",
        "--temperature",
        "1.0",
        "--repetition_penalty",
        "2.0",
        "--num_beams",
        "10",
        "--use_transfer",
        "--model_ckpt",
        str(tmp_path / "grpo_1400.ckpt"),
        "--use_postprocess",
    ]


def write_glb(path, *, animations=0, images=1, materials=1, meshes=1, skins=1):
    payload = {
        "asset": {"version": "2.0"},
        "animations": [{} for _ in range(animations)],
        "images": [{} for _ in range(images)],
        "materials": [{} for _ in range(materials)],
        "meshes": [{} for _ in range(meshes)],
        "skins": [{} for _ in range(skins)],
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    raw = (
        b"glTF"
        + struct.pack("<II", 2, 20 + len(encoded))
        + struct.pack("<II", len(encoded), runner.GLB_JSON_CHUNK)
        + encoded
    )
    path.write_bytes(raw)


def marker_payload(pid, argv, runtime):
    return {
        "argv": argv,
        "bpy_bind_host": "127.0.0.1",
        "bpy_port": runtime["port"],
        "generation": runtime["generation"],
        "patch_sha256": runtime["patch_sha256"],
        "pid": pid,
        "seed": runner.EXPECTED_SEED,
    }


def inventory(objects, meshes, materials, images):
    return {
        "objects": objects,
        "mesh_count": meshes,
        "material_count": materials,
        "image_count": images,
    }


def write_load_audit(path, source, server_marker):
    events = []
    for sequence in (1, 2):
        for phase, scene in (
            ("before_clean", inventory([{"name": "old", "type": "MESH"}], 1, 1, 1)),
            ("after_clean", inventory([], 0, 0, 0)),
            (
                "after_import",
                inventory([{"name": "animal", "type": "MESH"}], 1, 1, 1),
            ),
        ):
            events.append(
                {
                    "filepath": str(source),
                    "generation": server_marker["generation"],
                    "inventory": scene,
                    "patch_sha256": server_marker["patch_sha256"],
                    "phase": phase,
                    "pid": server_marker["pid"],
                    "sequence": sequence,
                }
            )
    path.write_text(
        "".join(json.dumps(event, sort_keys=True) + "\n" for event in events),
        encoding="utf-8",
    )


def test_parser_accepts_only_the_free_skeleton_surface(tmp_path):
    args = runner.parse_argv(production_argv(tmp_path))
    runner.require_production_sampling(args)
    assert args.use_skeleton is False
    assert args.hf_path is None

    with pytest.raises(SystemExit):
        runner.parse_argv(production_argv(tmp_path) + ["--use_skeleton"])
    with pytest.raises(SystemExit):
        runner.parse_argv(production_argv(tmp_path) + ["--hf_path", "model"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("top_k", 6),
        ("top_p", 0.9),
        ("temperature", 0.9),
        ("repetition_penalty", 1.9),
        ("num_beams", 9),
        ("use_transfer", False),
        ("use_postprocess", False),
        ("use_skeleton", True),
        ("hf_path", "other"),
    ],
)
def test_sampling_contract_fails_closed(field, value):
    args = argparse.Namespace(
        top_k=5,
        top_p=0.95,
        temperature=1.0,
        repetition_penalty=2.0,
        num_beams=10,
        use_transfer=True,
        use_postprocess=True,
        use_skeleton=False,
        hf_path=None,
    )
    setattr(args, field, value)
    with pytest.raises(runner.RunnerError, match="production contract"):
        runner.require_production_sampling(args)


def test_input_output_gate_rejects_symlink_and_existing_output(tmp_path):
    source = tmp_path / "source.glb"
    write_glb(source)
    link = tmp_path / "link.glb"
    link.symlink_to(source)
    with pytest.raises(runner.RunnerError, match="regular non-empty GLB"):
        runner.require_regular_glb(link, "input")

    output = tmp_path / "output.glb"
    output.write_bytes(b"existing")
    with pytest.raises(runner.RunnerError, match="refusing to replace"):
        runner.require_new_glb(output, source.resolve())


def test_model_gate_requires_pinned_snapshot_path_and_payload(tmp_path, monkeypatch):
    revision = runner.EXPECTED_MODEL_SNAPSHOT_REVISION
    models = tmp_path / "models"
    snapshot = models / "snapshots" / revision / "experiments"
    model_dir = snapshot / "articulation_xl_quantization_256_token_4"
    blobs = models / "blobs"
    model_dir.mkdir(parents=True)
    blobs.mkdir()
    payload = blobs / "payload"
    payload.write_bytes(b"pinned checkpoint")
    (model_dir / "grpo_1400.ckpt").symlink_to("../../../../blobs/payload")
    skintokens = tmp_path / "SkinTokens"
    skintokens.mkdir()
    (skintokens / "experiments").symlink_to(snapshot, target_is_directory=True)
    invocation = skintokens / runner.MODEL_RELATIVE_PATH
    monkeypatch.setattr(runner, "EXPECTED_MODEL_SHA256", runner.sha256_file(payload))

    observed, resolved, observed_revision = runner.require_model(invocation, skintokens)
    assert observed == invocation.absolute()
    assert resolved == payload
    assert observed_revision == revision

    other = tmp_path / "other.ckpt"
    other.write_bytes(payload.read_bytes())
    with pytest.raises(runner.RunnerError, match="invocation path"):
        runner.require_model(other, skintokens)


def test_runtime_environment_authenticates_the_main_marker(tmp_path, monkeypatch):
    main_pid = 123
    generation = "1" * 32
    patch_sha256 = runner.sha256_file(runner.RUNTIME_PATCH)
    marker_dir = tmp_path / "runtime_markers"
    marker_dir.mkdir()
    audit_path = tmp_path / "load_audit.jsonl"
    monkeypatch.setattr(runner.os, "getpid", lambda: main_pid)
    values = {
        "CUDA_VISIBLE_DEVICES": runner.EXPECTED_GPU,
        "PYTHONUNBUFFERED": "1",
        "TOKENRIG_BPY_BIND_HOST": "127.0.0.1",
        "TOKENRIG_BPY_PORT": "23456",
        "TOKENRIG_CANARY_SEED": str(runner.EXPECTED_SEED),
        "TOKENRIG_HYGIENE_MARKER_DIR": str(marker_dir),
        "TOKENRIG_LOAD_AUDIT_PATH": str(audit_path),
        "TOKENRIG_SERVER_HYGIENE_SHA256": patch_sha256,
        "TOKENRIG_SERVICE_GENERATION": generation,
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)
    runtime = {
        "generation": generation,
        "patch_sha256": patch_sha256,
        "port": 23456,
    }
    marker = marker_payload(main_pid, list(sys.argv), runtime)
    (marker_dir / f"{main_pid}.json").write_text(json.dumps(marker), encoding="utf-8")

    observed = runner.require_runtime_environment()
    assert observed["audit_path"] == audit_path
    assert observed["marker_dir"] == marker_dir

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(runner.RunnerError, match="exactly 3"):
        runner.require_runtime_environment()


def test_owned_health_can_become_ready_after_upstream_thirty_second_limit():
    expected = {
        "bind_host": "127.0.0.1",
        "generation": "1" * 32,
        "patch_sha256": "2" * 64,
        "pid": 4321,
        "port": 23456,
        "schema": runner.HEALTH_SCHEMA,
    }

    class Process:
        pid = 4321

        @staticmethod
        def poll():
            return None

    now = [0.0]
    attempts = [0]

    def health_reader(_url):
        attempts[0] += 1
        if attempts[0] == 1:
            raise runner.HealthUnavailable
        return expected

    def sleeper(_seconds):
        now[0] = 31.0

    runner.wait_for_owned_server(
        Process(),
        expected,
        health_reader=health_reader,
        monotonic=lambda: now[0],
        sleeper=sleeper,
    )
    assert now[0] > 30.0


def test_owned_health_timeout_and_early_exit_fail_closed(monkeypatch):
    expected = {
        "bind_host": "127.0.0.1",
        "generation": "1" * 32,
        "patch_sha256": "2" * 64,
        "pid": 4321,
        "port": 23456,
        "schema": runner.HEALTH_SCHEMA,
    }

    class RunningProcess:
        pid = 4321

        @staticmethod
        def poll():
            return None

    now = [0.0]
    monkeypatch.setattr(runner, "STARTUP_TIMEOUT_SECONDS", 1.0)
    with pytest.raises(runner.RunnerError, match="authenticated readiness"):
        runner.wait_for_owned_server(
            RunningProcess(),
            expected,
            health_reader=lambda _url: (_ for _ in ()).throw(
                runner.HealthUnavailable()
            ),
            monotonic=lambda: now[0],
            sleeper=lambda seconds: now.__setitem__(0, now[0] + seconds),
        )

    class ExitedProcess:
        pid = 4321

        @staticmethod
        def poll():
            return 7

    with pytest.raises(runner.RunnerError, match="code=7"):
        runner.wait_for_owned_server(
            ExitedProcess(),
            expected,
            health_reader=lambda _url: expected,
        )


def test_owned_health_rejects_stale_service_identity():
    expected = {
        "bind_host": "127.0.0.1",
        "generation": "1" * 32,
        "patch_sha256": "2" * 64,
        "pid": 4321,
        "port": 23456,
        "schema": runner.HEALTH_SCHEMA,
    }

    class Process:
        pid = 4321

        @staticmethod
        def poll():
            return None

    stale = dict(expected)
    stale["pid"] = 9999
    with pytest.raises(runner.RunnerError, match="different service"):
        runner.wait_for_owned_server(
            Process(),
            expected,
            health_reader=lambda _url: stale,
        )


def test_owned_server_cleanup_escalates_and_reaps(monkeypatch):
    class Process:
        pid = 4321

        def __init__(self):
            self.wait_count = 0

        @staticmethod
        def poll():
            return None

        def wait(self, timeout=None):
            self.wait_count += 1
            if self.wait_count == 1:
                raise subprocess.TimeoutExpired("server", timeout)
            return -signal.SIGKILL

    signals = []
    monkeypatch.setattr(runner.os, "getpgid", lambda _pid: 4321)
    monkeypatch.setattr(
        runner.os,
        "killpg",
        lambda process_group, sent: signals.append((process_group, sent)),
    )
    process = Process()
    runner.terminate_owned_server(process)
    assert signals == [
        (4321, signal.SIGTERM),
        (4321, signal.SIGKILL),
    ]
    assert process.wait_count == 2


def test_glb_gate_requires_one_skin_no_animation_and_pbr(tmp_path):
    valid = tmp_path / "valid.glb"
    write_glb(valid)
    assert runner.glb_summary(valid) == {
        "animations": 0,
        "images": 1,
        "materials": 1,
        "meshes": 1,
        "skins": 1,
    }

    animated = tmp_path / "animated.glb"
    write_glb(animated, animations=1)
    with pytest.raises(runner.RunnerError, match="must not contain animation"):
        runner.glb_summary(animated)

    no_pbr = tmp_path / "no_pbr.glb"
    write_glb(no_pbr, images=0)
    with pytest.raises(runner.RunnerError, match="lost its PBR"):
        runner.glb_summary(no_pbr)


def test_runtime_evidence_requires_two_markers_and_two_clean_loads(
    tmp_path, monkeypatch
):
    main_pid = 100
    server_pid = 200
    monkeypatch.setattr(runner.os, "getpid", lambda: main_pid)
    marker_dir = tmp_path / "runtime_markers"
    marker_dir.mkdir()
    source = tmp_path / "source.glb"
    write_glb(source)
    runtime = {
        "audit_path": tmp_path / "load_audit.jsonl",
        "generation": "1" * 32,
        "marker_dir": marker_dir,
        "patch_sha256": "2" * 64,
        "port": 23456,
    }
    main_marker = marker_payload(main_pid, list(sys.argv), runtime)
    server_marker = marker_payload(server_pid, ["bpy_server.py"], runtime)
    (marker_dir / f"{main_pid}.json").write_text(
        json.dumps(main_marker), encoding="utf-8"
    )
    (marker_dir / f"{server_pid}.json").write_text(
        json.dumps(server_marker), encoding="utf-8"
    )
    write_load_audit(runtime["audit_path"], source.resolve(), server_marker)

    observed = runner.validate_runtime_evidence(runtime, source.resolve(), server_pid)
    assert len(observed) == 2

    runtime["audit_path"].write_text(
        runtime["audit_path"]
        .read_text(encoding="utf-8")
        .replace('"after_clean"', '"wrong_phase"', 1),
        encoding="utf-8",
    )
    with pytest.raises(runner.RunnerError, match="phase ordering"):
        runner.validate_runtime_evidence(runtime, source.resolve(), server_pid)


def test_runner_is_one_shot_and_cross_contracts_match():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    run_cli_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_cli"
    ]
    assert len(run_cli_calls) == 1
    assert runner.STARTUP_TIMEOUT_SECONDS == 600.0
    assert runner.EXPECTED_SEED == closure.EXPECTED_SEED
    assert (
        runner.EXECUTION_IDENTITY_SCHEMA
        == closure.FREE_SKELETON_EXECUTION_IDENTITY_SCHEMA
    )
    assert runner.EXPECTED_MODEL_SHA256 == closure.EXPECTED_MODEL_SHA256
    assert (
        runner.EXPECTED_MODEL_SNAPSHOT_REVISION
        == closure.EXPECTED_MODEL_SNAPSHOT_REVISION
    )
    readback = (
        Path(runner.__file__).parent
        / "blender_readback_generated_animal_tokenrig_binding.py"
    ).read_text(encoding="utf-8")
    assert f'SCHEMA = "{closure.READBACK_SCHEMA}"' in readback
    closure_source = Path(closure.__file__).read_text(encoding="utf-8")
    expected_flags = closure_source[
        closure_source.index("expected_flags = [") : closure_source.index(
            "if exact_argv != expected_flags:",
            closure_source.index("expected_flags = ["),
        )
    ]
    assert '"--use_skeleton"' not in expected_flags


def test_closure_evidence_writer_retains_the_exact_runner_bytes(tmp_path):
    staging = tmp_path / "closure.staging"
    staging.mkdir()
    writer = closure.EvidenceWriter(staging)
    runner_path = Path(runner.__file__).resolve()
    record = writer.copy(
        "free-skeleton runner",
        runner_path,
        "execution/runner.py",
    )
    copied = staging / record["closure_copy"]["path"]
    assert record["original"]["sha256"] == runner.sha256_file(runner_path)
    assert record["closure_copy"]["sha256"] == runner.sha256_file(copied)
    assert copied.read_bytes() == runner_path.read_bytes()


def test_main_calls_demo_once_and_publishes_execution_identity(
    tmp_path, monkeypatch, capsys
):
    source = tmp_path / "source.glb"
    write_glb(source)
    output = tmp_path / "attempt" / "tokenrig_native.glb"
    output.parent.mkdir()
    marker_dir = output.parent / "runtime_markers"
    marker_dir.mkdir()
    audit_path = output.parent / "load_audit.jsonl"
    runtime = {
        "audit_path": audit_path,
        "generation": "1" * 32,
        "marker_dir": marker_dir,
        "patch_sha256": runner.sha256_file(runner.RUNTIME_PATCH),
        "port": 23456,
    }
    skintokens = tmp_path / "SkinTokens"
    skintokens.mkdir()
    demo_path = skintokens / "demo.py"
    demo_path.write_text("# fake demo\n", encoding="utf-8")
    model = skintokens / runner.MODEL_RELATIVE_PATH
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    process = SimpleNamespace(pid=4567, poll=lambda: None)
    calls = []

    class Requests:
        @staticmethod
        def post(*_args, **_kwargs):
            raise AssertionError("fake inference must not issue HTTP")

    def run_cli(args):
        calls.append(args)
        write_glb(Path(args.output))

    demo = SimpleNamespace(
        BPY_SERVER="http://127.0.0.1:23456",
        __file__=str(demo_path),
        requests=Requests,
        run_cli=run_cli,
        start_bpy_server=lambda: process,
    )
    monkeypatch.chdir(skintokens)
    monkeypatch.setattr(runner, "require_runtime_environment", lambda: runtime)
    monkeypatch.setattr(runner, "port_is_available", lambda _port: True)
    monkeypatch.setattr(
        runner,
        "require_skintokens_checkout",
        lambda _root: runner.EXPECTED_SKINTOKENS_REVISION,
    )
    monkeypatch.setattr(
        runner,
        "require_checked_in_runner",
        lambda: ("2" * 40, Path(runner.__file__).resolve()),
    )
    monkeypatch.setattr(
        runner,
        "require_model",
        lambda _path, _root: (
            model.absolute(),
            model.resolve(),
            runner.EXPECTED_MODEL_SNAPSHOT_REVISION,
        ),
    )
    monkeypatch.setattr(runner.importlib, "import_module", lambda _name: demo)
    monkeypatch.setattr(
        runner,
        "wait_for_owned_server",
        lambda _process, _health: None,
    )
    monkeypatch.setattr(
        runner,
        "validate_runtime_evidence",
        lambda _runtime, _source, _pid: [],
    )
    terminated = []
    monkeypatch.setattr(
        runner,
        "terminate_owned_server",
        lambda observed: terminated.append(observed.pid),
    )

    result = runner.main(
        [
            "--input",
            str(source),
            "--output",
            str(output),
            "--top_k",
            "5",
            "--top_p",
            "0.95",
            "--temperature",
            "1.0",
            "--repetition_penalty",
            "2.0",
            "--num_beams",
            "10",
            "--use_transfer",
            "--model_ckpt",
            str(model),
            "--use_postprocess",
        ]
    )
    assert result == 0
    assert len(calls) == 1
    assert calls[0].use_skeleton is False
    assert terminated == [4567]
    lines = capsys.readouterr().out.splitlines()
    identity_lines = [
        line
        for line in lines
        if line.startswith("TOKENRIG_FREE_SKELETON_EXECUTION_IDENTITY ")
    ]
    assert len(identity_lines) == 1
    identity = json.loads(identity_lines[0].split(" ", 1)[1])
    assert identity["runner"]["sha256"] == runner.sha256_file(Path(runner.__file__))
    assert identity["production_contract"]["use_skeleton"] is False


def test_checked_in_mode_run_log_is_accepted_by_closure(tmp_path, monkeypatch):
    tokenrig_input = tmp_path / "source.glb"
    tokenrig_output = tmp_path / "tokenrig_native.glb"
    model = tmp_path / "model.ckpt"
    tokenrig_input.write_bytes(b"source")
    tokenrig_output.write_bytes(b"rig")
    model.write_bytes(b"model")
    marker_dir = tmp_path / "runtime_markers"
    marker_dir.mkdir()
    load_audit = tmp_path / "load_audit.jsonl"
    patch_sha256 = runner.sha256_file(runner.RUNTIME_PATCH)
    generation = "1" * 32
    port = 23456
    exact_argv = [
        str(Path(runner.__file__).resolve()),
        "--input",
        str(tokenrig_input),
        "--output",
        str(tokenrig_output),
        "--top_k",
        "5",
        "--top_p",
        "0.95",
        "--temperature",
        "1.0",
        "--repetition_penalty",
        "2.0",
        "--num_beams",
        "10",
        "--use_transfer",
        "--model_ckpt",
        str(model),
        "--use_postprocess",
    ]
    main_marker = {
        "argv": exact_argv,
        "bpy_bind_host": "127.0.0.1",
        "bpy_port": port,
        "generation": generation,
        "patch_sha256": patch_sha256,
        "pid": 100,
        "seed": closure.EXPECTED_SEED,
    }
    server_marker = {
        **main_marker,
        "argv": ["bpy_server.py"],
        "pid": 200,
    }
    main_path = marker_dir / "100.json"
    server_path = marker_dir / "200.json"
    main_path.write_text(json.dumps(main_marker), encoding="utf-8")
    server_path.write_text(json.dumps(server_marker), encoding="utf-8")
    write_load_audit(load_audit, tokenrig_input.resolve(), server_marker)
    model_hash = runner.sha256_file(model)
    snapshot = "3" * 40
    monkeypatch.setattr(closure, "EXPECTED_MODEL_SHA256", model_hash)
    monkeypatch.setattr(closure, "EXPECTED_MODEL_SNAPSHOT_REVISION", snapshot)
    monkeypatch.setattr(
        closure,
        "model_link_chain",
        lambda _path: (model.resolve(), [], snapshot),
    )
    identity = {
        "schema": runner.EXECUTION_IDENTITY_SCHEMA,
        "spear_revision": "2" * 40,
        "runner": runner.file_record(Path(runner.__file__).resolve()),
        "runtime_patch": runner.file_record(runner.RUNTIME_PATCH),
        "skintokens_revision": runner.EXPECTED_SKINTOKENS_REVISION,
        "input": runner.file_record(tokenrig_input.resolve()),
        "model_checkpoint": {
            "invocation_path": str(model),
            "resolved_payload": runner.file_record(model.resolve()),
            "snapshot_revision": snapshot,
        },
        "production_contract": {
            "demo_run_cli_invocations": 1,
            "input_count": 1,
            "output_count": 1,
            "retry_or_ranking": False,
            "use_skeleton": False,
        },
        "formal_dataset_registration_authorized": False,
    }
    environment = [
        "env",
        "CUDA_VISIBLE_DEVICES=3",
        f"TOKENRIG_CANARY_SEED={closure.EXPECTED_SEED}",
        f"TOKENRIG_SERVER_HYGIENE_SHA256={patch_sha256}",
        f"TOKENRIG_BPY_PORT={port}",
        f"TOKENRIG_HYGIENE_MARKER_DIR={marker_dir}",
        f"TOKENRIG_LOAD_AUDIT_PATH={load_audit}",
        f"TOKENRIG_SERVICE_GENERATION={generation}",
        "TOKENRIG_BPY_BIND_HOST=127.0.0.1",
        "PYTHONUNBUFFERED=1",
        sys.executable,
        *exact_argv,
    ]
    run_log = tmp_path / "run.log"
    run_log.write_text(
        "\n".join(
            [
                "TOKENRIG_FREE_SKELETON_EXECUTION_IDENTITY "
                + json.dumps(identity, sort_keys=True, separators=(",", ":")),
                f"[OK] Exported: {tokenrig_output}",
                f'Command being timed: "{shlex.join(environment)}"',
                "\tExit status: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    markers = closure.marker_payloads(marker_dir)

    observed = closure.validate_run_evidence(
        run_log,
        markers,
        marker_dir,
        tokenrig_input.resolve(),
        tokenrig_output.resolve(),
        closure.CHECKED_IN_FREE_SKELETON_MODE,
        load_audit.resolve(),
    )
    assert observed["spear_execution_identity"] == identity

    with pytest.raises(closure.ClosureError, match="dedicated evidence mode"):
        closure.validate_run_evidence(
            run_log,
            markers,
            marker_dir,
            tokenrig_input.resolve(),
            tokenrig_output.resolve(),
            closure.COMPLETE_MODE,
            load_audit.resolve(),
        )


def test_spear_git_identity_rejects_post_inference_runner_or_patch_mutation(
    tmp_path,
):
    spear = tmp_path / "SPEAR"
    checked_runner = spear / closure.FREE_SKELETON_RUNNER_RELATIVE
    checked_patch = spear / closure.RUNTIME_PATCH_RELATIVE
    checked_runner.parent.mkdir(parents=True)
    checked_patch.parent.mkdir(parents=True)
    checked_runner.write_bytes(Path(runner.__file__).read_bytes())
    checked_patch.write_bytes(runner.RUNTIME_PATCH.read_bytes())
    commands = [
        ["git", "init", "-q"],
        ["git", "config", "user.email", "test@example.com"],
        ["git", "config", "user.name", "Test"],
        ["git", "add", "."],
        ["git", "commit", "-qm", "fixture"],
    ]
    for command in commands:
        subprocess.run(command, cwd=spear, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=spear,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    identity = {
        "spear_revision": revision,
        "runner": runner.file_record(checked_runner),
        "runtime_patch": runner.file_record(checked_patch),
    }

    observed = closure.validate_spear_execution_identity(
        identity,
        runner_path=checked_runner,
        patch_path=checked_patch,
        spear_root=spear,
    )
    assert observed["free-skeleton runner"]["git_revision"] == revision

    original_runner = checked_runner.read_bytes()
    checked_runner.write_bytes(original_runner + b"\nchanged\n")
    with pytest.raises(closure.ClosureError, match="changed after inference"):
        closure.validate_spear_execution_identity(
            identity,
            runner_path=checked_runner,
            patch_path=checked_patch,
            spear_root=spear,
        )
    checked_runner.write_bytes(original_runner)
    checked_patch.write_bytes(checked_patch.read_bytes() + b"\nchanged\n")
    with pytest.raises(closure.ClosureError, match="changed after inference"):
        closure.validate_spear_execution_identity(
            identity,
            runner_path=checked_runner,
            patch_path=checked_patch,
            spear_root=spear,
        )
