#!/usr/bin/env python3
#
# Copyright (c) 2025 The SPEAR Development Team
# Copyright (c) 2022 Intel
#
"""Run one authenticated free-skeleton TokenRig invocation.

This is the checked-in replacement for the temporary extended-start wrappers
used by the first generated-animal canaries.  It deliberately preserves the
exact TokenRig production argv consumed by
``generated_animal_tokenrig_closure.py`` while replacing the upstream
identity-free 30-second ``/ping`` gate with a 600-second, owned-generation
health check.

The runner accepts exactly one GLB and performs exactly one ``demo.run_cli``
call.  It never accepts ``--use_skeleton``: TokenRig remains responsible for
generating both the hierarchy and skin weights.  Output, runtime markers, and
load-audit files are never overwritten.
"""

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import struct
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


STARTUP_TIMEOUT_SECONDS = 600.0
SERVER_REQUEST_TIMEOUT_SECONDS = 600.0
POLL_INTERVAL_SECONDS = 0.5
SHUTDOWN_TIMEOUT_SECONDS = 10.0
EXPECTED_GPU = "3"
EXPECTED_SEED = 42
EXPECTED_SKINTOKENS_REVISION = "273b691d35989d71cd17ff2895fdc735097b92d1"
EXPECTED_MODEL_SHA256 = (
    "f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692"
)
EXPECTED_MODEL_SNAPSHOT_REVISION = "79736cad0fd84de384d5eede659b4ebd24effe33"
HEALTH_SCHEMA = "avengine_tokenrig_warm_health_v1"
EXECUTION_IDENTITY_SCHEMA = (
    "avengine_generated_animal_tokenrig_free_skeleton_execution_identity_v1"
)
SERVICE_GENERATION_PATTERN = re.compile(r"^[0-9a-f]{32}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GLB_JSON_CHUNK = 0x4E4F534A
TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
RUNTIME_PATCH = TOOLS_DIR / "runtime_patches/fixed_skeleton_skintokens/sitecustomize.py"
MODEL_RELATIVE_PATH = Path(
    "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
)


class RunnerError(RuntimeError):
    pass


class HealthUnavailable(RuntimeError):
    pass


def parse_argv(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top_k", type=int, required=True)
    parser.add_argument("--top_p", type=float, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--repetition_penalty", type=float, required=True)
    parser.add_argument("--num_beams", type=int, required=True)
    parser.add_argument("--use_transfer", action="store_true", required=True)
    parser.add_argument("--model_ckpt", required=True)
    parser.add_argument("--use_postprocess", action="store_true", required=True)
    args = parser.parse_args(argv)
    args.use_skeleton = False
    args.hf_path = None
    return args


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path):
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def require_production_sampling(args):
    expected = {
        "top_k": 5,
        "top_p": 0.95,
        "temperature": 1.0,
        "repetition_penalty": 2.0,
        "num_beams": 10,
        "use_transfer": True,
        "use_postprocess": True,
        "use_skeleton": False,
        "hf_path": None,
    }
    observed = {key: getattr(args, key) for key in expected}
    if observed != expected:
        raise RunnerError(
            "TokenRig arguments differ from the free-skeleton seed42 "
            "production contract"
        )


def require_regular_glb(path, label):
    unresolved = Path(path).absolute()
    if (
        unresolved.is_symlink()
        or not unresolved.is_file()
        or unresolved.stat().st_size <= 0
        or unresolved.suffix.lower() != ".glb"
    ):
        raise RunnerError(f"{label} must be one regular non-empty GLB: {unresolved}")
    return unresolved.resolve(strict=True)


def require_new_glb(path, source):
    unresolved = Path(path).absolute()
    if unresolved.exists() or unresolved.is_symlink():
        raise RunnerError(f"refusing to replace TokenRig output: {unresolved}")
    if unresolved.suffix.lower() != ".glb":
        raise RunnerError("TokenRig output must use the .glb suffix")
    try:
        candidate = unresolved.parent.resolve(strict=True) / unresolved.name
    except OSError as error:
        raise RunnerError("TokenRig output parent is missing") from error
    if candidate == source:
        raise RunnerError("TokenRig input and output paths are identical")
    return candidate


def path_snapshot_revision(path):
    current = Path(path.absolute().anchor)
    revisions = []
    for part in path.absolute().parts[1:]:
        current = current / part
        if not current.is_symlink():
            continue
        target = Path(os.readlink(current))
        if not target.is_absolute():
            target = current.parent / target
        match = re.search(r"/snapshots/([0-9a-f]{40})(?:/|$)", str(target.resolve()))
        if match is not None:
            revisions.append(match.group(1))
    if revisions != [EXPECTED_MODEL_SNAPSHOT_REVISION]:
        raise RunnerError(
            "TokenRig checkpoint does not resolve through one pinned snapshot"
        )
    return revisions[0]


def require_model(path, skintokens_root):
    invocation = Path(path).absolute()
    expected = (skintokens_root / MODEL_RELATIVE_PATH).absolute()
    try:
        same_file = os.path.samefile(invocation, expected)
    except OSError as error:
        raise RunnerError("TokenRig checkpoint is missing") from error
    if not same_file or invocation != expected:
        raise RunnerError("TokenRig checkpoint invocation path is not pinned")
    resolved = invocation.resolve(strict=True)
    if (
        not resolved.is_file()
        or resolved.stat().st_size <= 0
        or sha256_file(resolved) != EXPECTED_MODEL_SHA256
    ):
        raise RunnerError("TokenRig checkpoint payload hash changed")
    snapshot = path_snapshot_revision(invocation)
    return invocation, resolved, snapshot


def run_git(root, arguments, label):
    command = ["git", "-C", str(root), *arguments]
    print(f"TOKENRIG_FREE_SKELETON_COMMAND {shlex.join(command)}", flush=True)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip()
        raise RunnerError(f"{label} failed: {detail}")
    return result.stdout.strip()


def run_git_bytes(root, arguments, label):
    command = ["git", "-C", str(root), *arguments]
    print(f"TOKENRIG_FREE_SKELETON_COMMAND {shlex.join(command)}", flush=True)
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RunnerError(f"{label} failed: {detail}")
    return result.stdout


def require_skintokens_checkout(root):
    revision = run_git(root, ["rev-parse", "HEAD"], "SkinTokens revision")
    if revision != EXPECTED_SKINTOKENS_REVISION:
        raise RunnerError("SkinTokens revision changed")
    dirty = run_git(
        root,
        ["status", "--porcelain", "--untracked-files=no"],
        "SkinTokens tracked status",
    )
    if dirty:
        raise RunnerError("SkinTokens has tracked modifications")
    return revision


def require_checked_in_runner():
    runner = Path(__file__).resolve()
    software = (
        ("free-skeleton runner", runner),
        ("TokenRig runtime patch", RUNTIME_PATCH.resolve()),
    )
    for label, path in software:
        relative = path.relative_to(SPEAR_ROOT)
        tracked = run_git(
            SPEAR_ROOT,
            ["ls-files", "--error-unmatch", str(relative)],
            f"{label} tracking",
        )
        if tracked != str(relative):
            raise RunnerError(f"{label} is not tracked at its canonical path")
        dirty = run_git(
            SPEAR_ROOT,
            ["status", "--porcelain", "--untracked-files=no", "--", str(relative)],
            f"{label} tracked status",
        )
        if dirty:
            raise RunnerError(f"{label} has uncommitted changes")
    revision = run_git(SPEAR_ROOT, ["rev-parse", "HEAD"], "SPEAR revision")
    for label, path in software:
        relative = path.relative_to(SPEAR_ROOT)
        committed = run_git_bytes(
            SPEAR_ROOT,
            ["show", f"{revision}:{relative}"],
            f"{label} committed bytes",
        )
        if sha256_bytes(committed) != sha256_file(path):
            raise RunnerError(f"{label} differs from the SPEAR revision")
    return revision, runner


def load_json_file(path, label):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunnerError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise RunnerError(f"{label} must be a JSON object")
    return payload


def require_runtime_environment():
    required = {
        "CUDA_VISIBLE_DEVICES",
        "TOKENRIG_BPY_BIND_HOST",
        "TOKENRIG_BPY_PORT",
        "TOKENRIG_CANARY_SEED",
        "TOKENRIG_HYGIENE_MARKER_DIR",
        "TOKENRIG_LOAD_AUDIT_PATH",
        "TOKENRIG_SERVER_HYGIENE_SHA256",
        "TOKENRIG_SERVICE_GENERATION",
        "PYTHONUNBUFFERED",
    }
    missing = sorted(key for key in required if key not in os.environ)
    if missing:
        raise RunnerError(f"missing TokenRig environment: {missing}")
    if os.environ["CUDA_VISIBLE_DEVICES"] != EXPECTED_GPU:
        raise RunnerError("CUDA_VISIBLE_DEVICES must be exactly 3")
    if os.environ["TOKENRIG_CANARY_SEED"] != str(EXPECTED_SEED):
        raise RunnerError("TOKENRIG_CANARY_SEED must be exactly 42")
    if os.environ["PYTHONUNBUFFERED"] != "1":
        raise RunnerError("PYTHONUNBUFFERED must be exactly 1")
    if os.environ["TOKENRIG_BPY_BIND_HOST"] != "127.0.0.1":
        raise RunnerError("TOKENRIG_BPY_BIND_HOST must be 127.0.0.1")
    generation = os.environ["TOKENRIG_SERVICE_GENERATION"]
    if SERVICE_GENERATION_PATTERN.fullmatch(generation) is None:
        raise RunnerError("TOKENRIG_SERVICE_GENERATION must be 32 lowercase hex digits")
    patch_sha256 = os.environ["TOKENRIG_SERVER_HYGIENE_SHA256"]
    if SHA256_PATTERN.fullmatch(patch_sha256) is None:
        raise RunnerError("TOKENRIG_SERVER_HYGIENE_SHA256 is invalid")
    if sha256_file(RUNTIME_PATCH) != patch_sha256:
        raise RunnerError("runtime patch hash differs from the checked-in patch")
    try:
        port = int(os.environ["TOKENRIG_BPY_PORT"])
    except ValueError as error:
        raise RunnerError("TOKENRIG_BPY_PORT is not an integer") from error
    if not 1024 <= port <= 65535:
        raise RunnerError("TOKENRIG_BPY_PORT must be in [1024, 65535]")
    marker_dir = Path(os.environ["TOKENRIG_HYGIENE_MARKER_DIR"])
    audit_path = Path(os.environ["TOKENRIG_LOAD_AUDIT_PATH"])
    if not marker_dir.is_absolute() or not audit_path.is_absolute():
        raise RunnerError("TokenRig marker and audit paths must be absolute")
    if marker_dir.is_symlink() or not marker_dir.is_dir():
        raise RunnerError("TokenRig marker directory is missing or unsafe")
    if audit_path.exists() or audit_path.is_symlink():
        raise RunnerError("refusing to append to an existing TokenRig load audit")
    expected_marker = marker_dir / f"{os.getpid()}.json"
    entries = sorted(marker_dir.iterdir())
    if entries != [expected_marker] or expected_marker.is_symlink():
        raise RunnerError("TokenRig marker directory is not a fresh one-shot directory")
    marker = load_json_file(expected_marker, "main runtime marker")
    expected_fields = {
        "argv": list(sys.argv),
        "bpy_bind_host": "127.0.0.1",
        "bpy_port": port,
        "generation": generation,
        "patch_sha256": patch_sha256,
        "pid": os.getpid(),
        "seed": EXPECTED_SEED,
    }
    if marker != expected_fields:
        raise RunnerError("main runtime marker does not authenticate this invocation")
    return {
        "audit_path": audit_path,
        "generation": generation,
        "marker_dir": marker_dir,
        "patch_sha256": patch_sha256,
        "port": port,
    }


def port_is_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def read_health(url):
    try:
        with urlopen(url, timeout=1.0) as response:
            raw = response.read()
    except (HTTPError, URLError, TimeoutError, OSError) as error:
        raise HealthUnavailable from error
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunnerError("owned TokenRig health response is invalid") from error
    if not isinstance(payload, dict):
        raise RunnerError("owned TokenRig health response is not an object")
    return payload


def wait_for_owned_server(
    process,
    expected_health,
    health_reader=None,
    monotonic=None,
    sleeper=None,
):
    health_reader = health_reader or read_health
    monotonic = monotonic or time.monotonic
    sleeper = sleeper or time.sleep
    started = monotonic()
    url = f"http://127.0.0.1:{expected_health['port']}/avengine-health"
    while True:
        returncode = process.poll()
        if returncode is not None:
            raise RunnerError(
                "owned bpy_server exited before readiness: "
                f"pid={process.pid} code={returncode}"
            )
        try:
            payload = health_reader(url)
        except HealthUnavailable:
            payload = None
        if payload is not None:
            if payload != expected_health:
                raise RunnerError(
                    "TokenRig health endpoint belongs to a different service"
                )
            print(
                "TOKENRIG_FREE_SKELETON_SERVER_READY "
                f"pid={process.pid} generation={expected_health['generation']}",
                flush=True,
            )
            return
        elapsed = monotonic() - started
        if elapsed >= STARTUP_TIMEOUT_SECONDS:
            raise RunnerError(
                "owned bpy_server failed authenticated readiness within "
                f"{STARTUP_TIMEOUT_SECONDS:.0f} seconds"
            )
        sleeper(POLL_INTERVAL_SECONDS)


def terminate_owned_server(process):
    if process.poll() is not None:
        process.wait()
        return
    try:
        process_group = os.getpgid(process.pid)
    except ProcessLookupError:
        process.wait()
        return
    if process_group == process.pid:
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        process.terminate()
    try:
        process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    if process_group == process.pid:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        process.kill()
    process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)


def install_bounded_server_requests(demo):
    original_post = demo.requests.post

    def bounded_post(*args, **kwargs):
        kwargs.setdefault("timeout", SERVER_REQUEST_TIMEOUT_SECONDS)
        return original_post(*args, **kwargs)

    demo.requests.post = bounded_post


def glb_summary(path):
    raw = path.read_bytes()
    if len(raw) < 20 or raw[:4] != b"glTF":
        raise RunnerError("TokenRig output is not a GLB")
    version, declared_length = struct.unpack_from("<II", raw, 4)
    if version != 2 or declared_length != len(raw):
        raise RunnerError("TokenRig output GLB header is invalid")
    chunk_length, chunk_type = struct.unpack_from("<II", raw, 12)
    if chunk_type != GLB_JSON_CHUNK or 20 + chunk_length > len(raw):
        raise RunnerError("TokenRig output has no valid leading JSON chunk")
    try:
        payload = json.loads(
            raw[20 : 20 + chunk_length].rstrip(b" \t\r\n\0").decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RunnerError("TokenRig output GLB JSON is invalid") from error
    if not isinstance(payload, dict):
        raise RunnerError("TokenRig output GLB JSON root is not an object")
    summary = {
        "animations": len(payload.get("animations", [])),
        "images": len(payload.get("images", [])),
        "materials": len(payload.get("materials", [])),
        "meshes": len(payload.get("meshes", [])),
        "skins": len(payload.get("skins", [])),
    }
    if summary["meshes"] != 1 or summary["skins"] != 1:
        raise RunnerError(f"unexpected TokenRig mesh/skin counts: {summary}")
    if summary["animations"] != 0:
        raise RunnerError("free-skeleton TokenRig output must not contain animation")
    if summary["materials"] < 1 or summary["images"] < 1:
        raise RunnerError("free-skeleton TokenRig output lost its PBR container")
    return summary


def validate_load_audit(path, source, expected_server):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RunnerError("TokenRig load audit is missing or invalid") from error
    events = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise RunnerError("TokenRig load audit contains invalid JSON") from error
        if not isinstance(event, dict):
            raise RunnerError("TokenRig load audit event is not an object")
        events.append(event)
    sequences = {}
    for event in events:
        sequence = event.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int):
            raise RunnerError("TokenRig load audit sequence is invalid")
        sequences.setdefault(sequence, []).append(event)
        filepath = event.get("filepath")
        if (
            not isinstance(filepath, str)
            or not filepath
            or Path(filepath).resolve() != source
            or event.get("generation") != expected_server["generation"]
            or event.get("patch_sha256") != expected_server["patch_sha256"]
            or event.get("pid") != expected_server["pid"]
        ):
            raise RunnerError("TokenRig load audit lineage changed")
    if sorted(sequences) != [1, 2]:
        raise RunnerError("TokenRig load audit must contain exactly two load sequences")
    for records in sequences.values():
        if [record.get("phase") for record in records] != [
            "before_clean",
            "after_clean",
            "after_import",
        ]:
            raise RunnerError("TokenRig load audit phase ordering changed")
        clean = records[1].get("inventory")
        imported = records[2].get("inventory")
        if (
            not isinstance(clean, dict)
            or clean.get("objects") != []
            or clean.get("mesh_count") != 0
            or clean.get("material_count") != 0
            or clean.get("image_count") != 0
        ):
            raise RunnerError("TokenRig load audit did not prove a clean Blender scene")
        if not isinstance(imported, dict) or imported.get("mesh_count") != 1:
            raise RunnerError("TokenRig load audit did not import exactly one mesh")
        objects = imported.get("objects")
        mesh_objects = (
            [
                item
                for item in objects
                if isinstance(item, dict) and item.get("type") == "MESH"
            ]
            if isinstance(objects, list)
            else []
        )
        empty_objects = (
            [
                item
                for item in objects
                if isinstance(item, dict) and item.get("type") == "EMPTY"
            ]
            if isinstance(objects, list)
            else []
        )
        if (
            not isinstance(objects, list)
            or len(mesh_objects) != 1
            or empty_objects
            not in ([], [{"name": "world", "type": "EMPTY"}])
            or mesh_objects[0].get("name") in {"Camera", "Cube", "Light"}
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or not item["name"]
                or item.get("type") not in {"MESH", "EMPTY"}
                for item in objects
            )
        ):
            raise RunnerError("TokenRig load audit imported a contaminated scene")
    return events


def validate_runtime_evidence(runtime, source, server_pid):
    paths = sorted(runtime["marker_dir"].glob("*.json"))
    expected_paths = sorted(
        [
            runtime["marker_dir"] / f"{os.getpid()}.json",
            runtime["marker_dir"] / f"{server_pid}.json",
        ]
    )
    if paths != expected_paths or any(path.is_symlink() for path in paths):
        raise RunnerError("TokenRig did not publish exactly two owned runtime markers")
    markers = [load_json_file(path, "runtime marker") for path in paths]
    server = [marker for marker in markers if marker.get("argv") == ["bpy_server.py"]]
    main = [marker for marker in markers if marker.get("argv") != ["bpy_server.py"]]
    if len(server) != 1 or len(main) != 1:
        raise RunnerError("TokenRig runtime markers do not identify main and server")
    common = {
        "bpy_bind_host": "127.0.0.1",
        "bpy_port": runtime["port"],
        "generation": runtime["generation"],
        "patch_sha256": runtime["patch_sha256"],
        "seed": EXPECTED_SEED,
    }
    expected_main = {
        **common,
        "argv": list(sys.argv),
        "pid": os.getpid(),
    }
    expected_server = {
        **common,
        "argv": ["bpy_server.py"],
        "pid": server_pid,
    }
    if main[0] != expected_main or server[0] != expected_server:
        raise RunnerError("TokenRig runtime marker execution identity changed")
    validate_load_audit(runtime["audit_path"], source, server[0])
    return markers


def main(argv=None):
    args = parse_argv(argv)
    require_production_sampling(args)
    runtime = require_runtime_environment()
    source = require_regular_glb(args.input, "TokenRig input")
    output = require_new_glb(args.output, source)
    attempt_root = output.parent.resolve(strict=True)
    if (
        runtime["marker_dir"].parent.resolve(strict=True) != attempt_root
        or runtime["audit_path"].parent.resolve(strict=True) != attempt_root
    ):
        raise RunnerError(
            "TokenRig output, markers, and load audit must share one attempt root"
        )
    if not port_is_available(runtime["port"]):
        raise RunnerError("TokenRig bpy port already has a listener")
    skintokens_root = Path.cwd().resolve(strict=True)
    skintokens_revision = require_skintokens_checkout(skintokens_root)
    spear_revision, runner = require_checked_in_runner()
    model, resolved_model, snapshot = require_model(args.model_ckpt, skintokens_root)
    input_sha256 = sha256_file(source)
    input_identity = (source.stat().st_dev, source.stat().st_ino)
    patch_identity = (RUNTIME_PATCH.stat().st_dev, RUNTIME_PATCH.stat().st_ino)
    args.input = str(source)
    args.output = str(output)
    args.model_ckpt = str(model)
    demo = importlib.import_module("demo")
    demo_path = Path(demo.__file__).resolve(strict=True)
    if not os.path.samefile(demo_path, skintokens_root / "demo.py"):
        raise RunnerError(
            "imported demo.py is outside the authenticated SkinTokens root"
        )
    expected_server_url = f"http://127.0.0.1:{runtime['port']}"
    if demo.BPY_SERVER != expected_server_url:
        raise RunnerError("SkinTokens demo imported a different bpy server endpoint")
    install_bounded_server_requests(demo)
    execution_identity = {
        "schema": EXECUTION_IDENTITY_SCHEMA,
        "spear_revision": spear_revision,
        "runner": file_record(runner),
        "runtime_patch": file_record(RUNTIME_PATCH),
        "skintokens_revision": skintokens_revision,
        "input": file_record(source),
        "model_checkpoint": {
            "invocation_path": str(model),
            "resolved_payload": file_record(resolved_model),
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
    print(
        "TOKENRIG_FREE_SKELETON_EXECUTION_IDENTITY "
        + json.dumps(execution_identity, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    print(
        "TOKENRIG_FREE_SKELETON_STARTED "
        f"input={source} input_sha256={input_sha256} output={output} "
        f"runner={runner} runner_sha256={sha256_file(runner)} "
        f"spear_revision={spear_revision} "
        f"skintokens_revision={skintokens_revision} "
        f"checkpoint={resolved_model} checkpoint_sha256={EXPECTED_MODEL_SHA256} "
        f"checkpoint_snapshot={snapshot}",
        flush=True,
    )
    process = None
    try:
        process = demo.start_bpy_server()
        expected_health = {
            "bind_host": "127.0.0.1",
            "generation": runtime["generation"],
            "patch_sha256": runtime["patch_sha256"],
            "pid": process.pid,
            "port": runtime["port"],
            "schema": HEALTH_SCHEMA,
        }
        wait_for_owned_server(process, expected_health)
        demo.run_cli(args)
        if process.poll() is not None:
            raise RunnerError("owned bpy_server exited during TokenRig inference")
        current_identity = (source.stat().st_dev, source.stat().st_ino)
        if current_identity != input_identity or sha256_file(source) != input_sha256:
            raise RunnerError("TokenRig input changed during inference")
        current_patch_identity = (
            RUNTIME_PATCH.stat().st_dev,
            RUNTIME_PATCH.stat().st_ino,
        )
        if (
            current_patch_identity != patch_identity
            or sha256_file(RUNTIME_PATCH) != runtime["patch_sha256"]
        ):
            raise RunnerError("TokenRig runtime patch changed during inference")
        generated = require_regular_glb(output, "TokenRig output")
        summary = glb_summary(generated)
        validate_runtime_evidence(runtime, source, process.pid)
        print(
            "TOKENRIG_FREE_SKELETON_OK "
            f"output={generated} output_sha256={sha256_file(generated)} "
            f"summary={json.dumps(summary, sort_keys=True)}",
            flush=True,
        )
        return 0
    finally:
        if process is not None:
            terminate_owned_server(process)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as error:
        print(f"TOKENRIG_FREE_SKELETON_FAILED {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
