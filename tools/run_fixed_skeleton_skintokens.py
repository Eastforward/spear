#!/usr/bin/env python3
"""Run deterministic SkinTokens skinning for a prepared fixed skeleton.

The input GLB must already contain one generated mesh and one fitted,
single-root skeleton.  TokenRig is conditioned with ``--use_skeleton`` so it
generates weights only; it is not allowed to invent a new hierarchy or bone
axes.  Every invocation publishes an immutable attempt ledger and live log.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time


SCHEMA = "avengine_fixed_skeleton_skintokens_attempt_v1"
TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
SKINTOKENS_ROOT = SPEAR_ROOT.parent / "SkinTokens"
SKINTOKENS_PYTHON = SKINTOKENS_ROOT / ".venv/bin/python"
SKINTOKENS_DEMO = SKINTOKENS_ROOT / "demo.py"
CHECKPOINT = (
    SKINTOKENS_ROOT
    / "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
)
PATCH_DIR = TOOLS_DIR / "runtime_patches/fixed_skeleton_skintokens"
PATCH_PATH = PATCH_DIR / "sitecustomize.py"
GLB_JSON_CHUNK = 0x4E4F534A


def parse_argv(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--bpy-port",
        type=int,
        default=0,
        help="Dedicated SkinTokens Blender-service port; 0 allocates a free loopback port.",
    )
    parser.add_argument("--no-postprocess", action="store_true")
    return parser.parse_args(argv)


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path):
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def require_direct_file(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe {label}: {path}")
    return path


def require_venv_python(path: Path) -> Path:
    """Accept a normal venv launcher symlink without accepting an arbitrary path."""
    path = path.absolute()
    expected_bin = (SKINTOKENS_ROOT / ".venv/bin").absolute()
    try:
        path.relative_to(expected_bin)
    except ValueError as error:
        raise SystemExit(
            f"SkinTokens Python escaped its pinned venv bin directory: {path}"
        ) from error
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SystemExit(f"missing SkinTokens Python: {path}") from error
    if not path.is_file() or not resolved.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(f"missing or non-executable SkinTokens Python: {path}")
    return path


def require_pinned_model_file(path: Path) -> Path:
    """Allow the pinned HF snapshot link only when it resolves under /data/models."""
    path = path.absolute()
    if path != CHECKPOINT.absolute():
        raise SystemExit(f"SkinTokens checkpoint path is not pinned: {path}")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(Path("/data/models").resolve())
    except (OSError, ValueError) as error:
        raise SystemExit(
            f"SkinTokens checkpoint did not resolve inside /data/models: {path}"
        ) from error
    if not path.is_file() or not resolved.is_file() or resolved.stat().st_size <= 0:
        raise SystemExit(f"missing or empty SkinTokens checkpoint: {path}")
    return path


def git_head(root: Path):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def glb_summary(path: Path):
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise RuntimeError("SkinTokens output is not a GLB")
    declared = struct.unpack_from("<I", data, 8)[0]
    if declared != len(data):
        raise RuntimeError("SkinTokens output GLB length does not match its header")
    chunk_length, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != GLB_JSON_CHUNK:
        raise RuntimeError("SkinTokens output GLB has no leading JSON chunk")
    payload = json.loads(data[20 : 20 + chunk_length].rstrip(b" \t\r\n\x00"))
    summary = {
        "meshes": len(payload.get("meshes", [])),
        "skins": len(payload.get("skins", [])),
        "nodes": len(payload.get("nodes", [])),
        "animations": len(payload.get("animations", [])),
        "materials": len(payload.get("materials", [])),
        "images": len(payload.get("images", [])),
        "extensions_used": sorted(payload.get("extensionsUsed", [])),
    }
    if summary["meshes"] != 1 or summary["skins"] != 1:
        raise RuntimeError(f"unexpected SkinTokens output container: {summary}")
    if summary["animations"] != 0:
        raise RuntimeError("fixed-skeleton skin output must not contain animation")
    if summary["materials"] < 1 or summary["images"] < 1:
        raise RuntimeError("fixed-skeleton skin output lost its PBR container")
    return summary


def write_json_exclusive(path: Path, payload):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def allocate_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def main(argv=None):
    args = parse_argv(argv)
    if args.bpy_port and not 1024 <= args.bpy_port <= 65535:
        raise SystemExit("--bpy-port must be 0 or in [1024, 65535]")
    bpy_port = args.bpy_port or allocate_loopback_port()
    source = require_direct_file(args.input, "conditioning GLB")
    python = require_venv_python(SKINTOKENS_PYTHON)
    demo = require_direct_file(SKINTOKENS_DEMO, "SkinTokens demo")
    checkpoint = require_pinned_model_file(CHECKPOINT)
    patch = require_direct_file(PATCH_PATH, "runtime hygiene patch")
    output_dir = args.output_dir.absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise SystemExit(f"refusing to replace output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    marker_dir = output_dir / "runtime_markers"
    marker_dir.mkdir()
    output = output_dir / "fixed_skeleton_skin.glb"
    log_path = output_dir / "inference.log"
    ledger_path = output_dir / "attempt.json"
    audit_path = output_dir / "load_audit.jsonl"

    command = [
        str(python),
        str(demo),
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
        "--use_skeleton",
        "--model_ckpt",
        str(checkpoint),
    ]
    if not args.no_postprocess:
        command.append("--use_postprocess")

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "TOKENRIG_CANARY_SEED": str(args.seed),
            "TOKENRIG_LOAD_AUDIT_PATH": str(audit_path),
            "TOKENRIG_HYGIENE_MARKER_DIR": str(marker_dir),
            "TOKENRIG_SERVER_HYGIENE_SHA256": sha256_file(patch),
            "TOKENRIG_BPY_PORT": str(bpy_port),
            "PYTHONPATH": os.pathsep.join(
                [str(PATCH_DIR), str(SKINTOKENS_ROOT), environment.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep),
            "MPLCONFIGDIR": "/tmp/avengine_matplotlib",
            "NUMBA_CACHE_DIR": "/tmp/avengine_numba_cache",
        }
    )

    started_at = now_utc()
    start = time.monotonic()
    print(
        "FIXED_SKELETON_SKINTOKENS_STARTED "
        f"gpu={args.gpu} bpy_port={bpy_port} seed={args.seed} "
        f"input={source} output={output}",
        flush=True,
    )
    returncode = None
    error = None
    with log_path.open("x", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=SKINTOKENS_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            print(line, end="", flush=True)
        returncode = process.wait()
        log.flush()
        os.fsync(log.fileno())

    summary = None
    try:
        if returncode != 0:
            raise RuntimeError(f"SkinTokens exited with code {returncode}")
        if not output.is_file() or output.stat().st_size <= 0:
            raise RuntimeError("SkinTokens returned zero without publishing its output")
        summary = glb_summary(output)
    except Exception as caught:
        error = f"{type(caught).__name__}: {caught}"

    payload = {
        "schema": SCHEMA,
        "started_at": started_at,
        "ended_at": now_utc(),
        "duration_seconds": time.monotonic() - start,
        "status": "succeeded" if error is None else "failed",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "input": file_record(source),
        "output": file_record(output) if output.is_file() else None,
        "output_readback": summary,
        "runtime": {
            "python_launcher": str(python),
            "python_resolved": str(python.resolve()),
            "gpu_physical_index": args.gpu,
            "bpy_port": bpy_port,
            "seed": args.seed,
            "postprocess": not args.no_postprocess,
            "command": command,
            "returncode": returncode,
            "log": file_record(log_path),
            "load_audit": file_record(audit_path) if audit_path.is_file() else None,
            "marker_files": [
                file_record(path) for path in sorted(marker_dir.glob("*.json"))
            ],
        },
        "skintokens": {
            "root": str(SKINTOKENS_ROOT),
            "git_revision": git_head(SKINTOKENS_ROOT),
            "demo": file_record(demo),
            "checkpoint": file_record(checkpoint),
            "checkpoint_resolved": str(checkpoint.resolve()),
            "runtime_hygiene_patch": file_record(patch),
            "mode": "fixed_skeleton_generate_skin_only",
        },
        "error": error,
        "next_stage": (
            "attach_compatible_walk_idle_and_run_deformation_contact_media_gates"
            if error is None
            else "preserve_failure_and_diagnose"
        ),
    }
    write_json_exclusive(ledger_path, payload)
    if error is not None:
        print(
            f"FIXED_SKELETON_SKINTOKENS_FAILED ledger={ledger_path} error={error}",
            flush=True,
        )
        raise SystemExit(1)
    print(
        "FIXED_SKELETON_SKINTOKENS_OK "
        f"output={output} sha256={payload['output']['sha256']} ledger={ledger_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
