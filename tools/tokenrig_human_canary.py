#!/usr/bin/env python3
"""Run one hash-locked Pixal male TokenRig transfer without network access."""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import hashlib
import json
import os
import shutil
import socket
import stat
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ASSET_ID = "rocketbox_male_adult_01"
SCHEMA = "pixal_tokenrig_canary_v1"
PIXAL3D_REVISION = "0b31f9160aa400719af409098bff7936a932f726"
TOKENRIG_CHECKPOINT_RELATIVE = (
    "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
)
SKIN_VAE_CHECKPOINT_RELATIVE = "experiments/skin_vae_2_10_32768/last.ckpt"
OFFLINE_ENVIRONMENT = {
    "DIFFUSERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}
SAMPLING_PARAMETERS = {
    "do_sample": True,
    "max_length": 2048,
    "num_beams": 10,
    "num_return_sequences": 1,
    "repetition_penalty": 2.0,
    "temperature": 1.0,
    "top_k": 5,
    "top_p": 0.95,
}
PHYSICAL_GPU = "3"
EXPECTED_PYTHON_PREFIX = "3.11."
EXPECTED_CUDA_VERSION = "12.6"
EXPECTED_RUNTIME_PACKAGES = {
    "bpy": "5.0.1",
    "diffusers": "0.39.0",
    "flash_attn": "2.8.3.post1",
    "lightning": "2.6.5",
    "numpy": "1.26.4",
    "omegaconf": "2.3.1",
    "open3d": "0.19.0",
    "torch": "2.7.1+cu126",
    "transformers": "5.13.1",
    "trimesh": "4.12.2",
}
BPY_HOST = "127.0.0.1"
BPY_PORT = 59876
TRAINING_PROVENANCE_SOURCES = (
    "ArticulationXL 2.0",
    "VRoid Hub",
    "ModelsResource",
)
SKINTOKENS_MODEL_CARD_URL = (
    "https://huggingface.co/VAST-AI/SkinTokens/tree/main/experiments/"
    "articulation_xl_quantization_256_token_4"
)

RUNNER_PATH = Path(__file__).resolve()
_REPO_ROOT = RUNNER_PATH.parents[1]
_SKINTOKENS_ROOT = _REPO_ROOT.parent / "SkinTokens"
_INPUT_ROOT = (
    _REPO_ROOT
    / "tmp/i23d_human_bakeoff_v1/pixal3d/rocketbox_male_adult_01"
)
_OUTPUT_DIR = _REPO_ROOT / "tmp/pixal_tokenrig_route2_v1" / ASSET_ID
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1
_GLB_JSON_CHUNK = 0x4E4F534A


class CanaryError(ValueError):
    """Raised when the canary contract cannot be proven or safely published."""


class CanarySubprocessError(CanaryError):
    """A checked subprocess failure with its complete captured result."""

    def __init__(self, result: Any):
        self.result = result
        returncode = getattr(result, "returncode", None)
        stderr = getattr(result, "stderr", "") or ""
        super().__init__(f"subprocess failed with exit {returncode}: {stderr}")


@dataclass(frozen=True)
class CanaryContract:
    asset_id: str
    input_glb: Path
    input_manifest: Path
    output_dir: Path
    skintokens_root: Path
    input_glb_sha256: str
    input_manifest_sha256: str
    skintokens_commit: str
    model_revision: str
    code_hashes: Mapping[str, str]
    checkpoint_hashes: Mapping[str, str]
    checkpoint_sizes: Mapping[str, int]
    weight_cache_root: Path | None
    qwen_revision: str
    qwen_cache_root: Path | None
    qwen_file_hashes: Mapping[str, str]
    qwen_file_sizes: Mapping[str, int]
    license_hash: str


@dataclass(frozen=True)
class RecoveryContract:
    asset_id: str
    attempt_id: str
    attempt_ledger: Path
    attempt_ledger_sha256: str
    attempt_ledger_size: int
    failed_evidence_dir: Path
    failed_inventory_sha256: str
    failed_glb: Path
    failed_glb_sha256: str
    failed_glb_size: int
    inference_log: Path
    inference_log_sha256: str
    inference_log_size: int
    sitecustomize_path: Path
    sitecustomize_sha256: str
    sitecustomize_size: int
    input_glb: Path
    input_glb_sha256: str
    input_glb_size: int
    input_manifest: Path
    input_manifest_sha256: str
    input_manifest_size: int
    skintokens_root: Path
    skintokens_commit: str
    parser_path: Path
    parser_sha256: str
    parser_size: int
    bpyparser_load_sha256: str
    clean_bpy_sha256: str
    recovery_output_dir: Path


PINNED_CONTRACT = CanaryContract(
    asset_id=ASSET_ID,
    input_glb=_INPUT_ROOT / "canary_1024_seed42.glb",
    input_manifest=_INPUT_ROOT / "canary_1024_seed42.manifest.json",
    output_dir=_OUTPUT_DIR,
    skintokens_root=_SKINTOKENS_ROOT,
    input_glb_sha256="1df2490d6b83e52fa3b7c4e9d6b69207fa59cad0deae80e3dc3f894dfc443c42",
    input_manifest_sha256="f0658fbcf84d3505d5ea08fcf3011c9070de46933da50c7304d44875a3b038e3",
    skintokens_commit="273b691d35989d71cd17ff2895fdc735097b92d1",
    model_revision="79736cad0fd84de384d5eede659b4ebd24effe33",
    code_hashes={
        "demo.py": "8e6d058225c39caad0fccf7c4d6942f8e7e32e3f57c5b14cdc60cf2d6cb5d316",
        "bpy_server.py": "0764aa1436130bdf32ffc2892a4497b77e9e22399c7ed1af5cf11a1f32500130",
    },
    checkpoint_hashes={
        TOKENRIG_CHECKPOINT_RELATIVE: "f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692",
        SKIN_VAE_CHECKPOINT_RELATIVE: "4843f49e58afff88345806b94ca82e6cc9d8def6e7432e2853c677b154de0ed4",
    },
    checkpoint_sizes={
        TOKENRIG_CHECKPOINT_RELATIVE: 1131603979,
        SKIN_VAE_CHECKPOINT_RELATIVE: 487311745,
    },
    weight_cache_root=Path("/data/models/hub/models--VAST-AI--SkinTokens"),
    qwen_revision="c1899de289a04d12100db370d81485cdf75e47ca",
    qwen_cache_root=Path("/data/models/hub/models--Qwen--Qwen3-0.6B"),
    qwen_file_hashes={
        "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
        "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
        "README.md": "1ab64a26fcb3b461423b89a433a8c858f1bf8d4086f979cbb3ff878d47cf20e9",
    },
    qwen_file_sizes={
        "config.json": 726,
        "generation_config.json": 239,
        "README.md": 13965,
    },
    license_hash="4f818b00ed33ed1772236c8b0acfd40e740e11e93f4d6f4f846b506b5b690789",
)

_ROUTE_ROOT = _OUTPUT_DIR.parent
_FAILED_EVIDENCE_DIR = _ROUTE_ROOT / f"{ASSET_ID}.tokenrig_failed_attempt"
PINNED_RECOVERY_CONTRACT = RecoveryContract(
    asset_id=ASSET_ID,
    attempt_id="22ff2d5d4b1181eb728a74c38f95dadf43bf3e45c963a3d3b31beb472759e4ed",
    attempt_ledger=_ROUTE_ROOT / f"{ASSET_ID}.tokenrig_attempt.json",
    attempt_ledger_sha256="b76e3e65733151e394fbfe59a219162deba8bed9ca85faa4718181fa32b15d29",
    attempt_ledger_size=17979,
    failed_evidence_dir=_FAILED_EVIDENCE_DIR,
    failed_inventory_sha256="ac97e4b5293ae6cb073d39fbe4af598c7d43854572ea6b77ce00a4c248b0a9a8",
    failed_glb=_FAILED_EVIDENCE_DIR / "tokenrig_transfer.glb",
    failed_glb_sha256="8606c013fba02f722e1d5c65accddc4398eab1fa925467a9233aaf458d93f01c",
    failed_glb_size=50843552,
    inference_log=_FAILED_EVIDENCE_DIR / "inference.log",
    inference_log_sha256="c78cc5d4665a2794fe662bbd6a5abbfe05fcfd4942489aaf030c840a19f7f29d",
    inference_log_size=3372,
    sitecustomize_path=_FAILED_EVIDENCE_DIR / "runtime_patch/sitecustomize.py",
    sitecustomize_sha256="d643aa31ee39a12c356802bf659864328fa7ce7b51df9d6a3e7b8d85acbf207d",
    sitecustomize_size=2309,
    input_glb=PINNED_CONTRACT.input_glb,
    input_glb_sha256=PINNED_CONTRACT.input_glb_sha256,
    input_glb_size=36430844,
    input_manifest=PINNED_CONTRACT.input_manifest,
    input_manifest_sha256=PINNED_CONTRACT.input_manifest_sha256,
    input_manifest_size=1135,
    skintokens_root=_SKINTOKENS_ROOT,
    skintokens_commit=PINNED_CONTRACT.skintokens_commit,
    parser_path=_SKINTOKENS_ROOT / "src/rig_package/parser/bpy.py",
    parser_sha256="ac186556d424b2581d0127e579440e605909f2420ae9e48bf8f401db31114a39",
    parser_size=32134,
    bpyparser_load_sha256="26bde20077ca21e34822e6fbab5f6395924f097a94e7b21d0b2e656bd648d4a0",
    clean_bpy_sha256="af3707436fa42f3c2388b65f0522684bff0c28eee9ac5334a77639e8b035cb62",
    recovery_output_dir=_OUTPUT_DIR,
)


SEED_BOOTSTRAP = """\
import random
import runpy
import sys
seed = int(sys.argv[1])
random.seed(seed)
import numpy as np
np.random.seed(seed)
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
demo = sys.argv[2]
sys.argv = [demo, *sys.argv[3:]]
runpy.run_path(demo, run_name="__main__")
"""


SERVER_HYGIENE_SOURCE = """\
\"\"\"Injected fail-closed hygiene for the SkinTokens bpy child process.\"\"\"
import hashlib
import json
import os
import random
import sys
from pathlib import Path

seed = int(os.environ["TOKENRIG_CANARY_SEED"])
random.seed(seed)
import numpy as np
np.random.seed(seed)
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

from src.rig_package.parser import bpy as parser

_original_load = parser.BpyParser.load.__func__
_load_sequence = 0
_load_audit_path = Path(os.environ["TOKENRIG_LOAD_AUDIT_PATH"])
_load_audit_path.parent.mkdir(parents=True, exist_ok=True)

def _inventory():
    objects = sorted(
        ({"name": obj.name, "type": obj.type} for obj in parser.bpy.data.objects),
        key=lambda item: (item["name"], item["type"]),
    )
    return {
        "objects": objects,
        "mesh_count": len(parser.bpy.data.meshes),
        "material_count": len(parser.bpy.data.materials),
        "image_count": len(parser.bpy.data.images),
    }

def _record_load_event(sequence, phase, filepath):
    event = {
        "sequence": sequence,
        "phase": phase,
        "filepath": str(Path(filepath).resolve()),
        "inventory": _inventory(),
    }
    payload = (json.dumps(event, sort_keys=True) + "\\n").encode("utf-8")
    with _load_audit_path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())

def _clean_load(cls, filepath, **kwargs):
    global _load_sequence
    _load_sequence += 1
    sequence = _load_sequence
    _record_load_event(sequence, "before_clean", filepath)
    parser.clean_bpy()
    _record_load_event(sequence, "after_clean", filepath)
    result = _original_load(cls, filepath, **kwargs)
    _record_load_event(sequence, "after_import", filepath)
    return result

parser.BpyParser.load = classmethod(_clean_load)

marker_dir = Path(os.environ["TOKENRIG_HYGIENE_MARKER_DIR"])
marker_dir.mkdir(parents=True, exist_ok=True)
marker = {
    "argv": list(sys.argv),
    "patch_sha256": os.environ["TOKENRIG_SERVER_HYGIENE_SHA256"],
    "pid": os.getpid(),
    "seed": seed,
}
marker_bytes = (json.dumps(marker, sort_keys=True) + "\\n").encode("utf-8")
with (marker_dir / f"{os.getpid()}.json").open("xb") as stream:
    stream.write(marker_bytes)
    stream.flush()
    os.fsync(stream.fileno())
"""


RUNTIME_PROBE_SOURCE = """\
import importlib.metadata
import json
import os
import platform
import torch

distributions = {
    "bpy": "bpy",
    "diffusers": "diffusers",
    "flash_attn": "flash-attn",
    "lightning": "lightning",
    "numpy": "numpy",
    "omegaconf": "omegaconf",
    "open3d": "open3d",
    "torch": "torch",
    "transformers": "transformers",
    "trimesh": "trimesh",
}
packages = {name: importlib.metadata.version(dist) for name, dist in distributions.items()}
packages["torch"] = torch.__version__
available = torch.cuda.is_available()
gpu = {
    "available": available,
    "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "logical_index": torch.cuda.current_device() if available else None,
    "name": torch.cuda.get_device_name(torch.cuda.current_device()) if available else None,
}
print(json.dumps({
    "python": platform.python_version(),
    "packages": packages,
    "cuda": torch.version.cuda,
    "gpu": gpu,
}, sort_keys=True))
"""


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, public_path: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(public_path if public_path is not None else path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _require_real_directory(path: Path, description: str) -> Path:
    path = _absolute(path)
    if not path.is_dir() or path.is_symlink() or path.resolve() != path:
        raise CanaryError(f"{description} must be a direct real directory: {path}")
    return path


def _require_regular_file(path: Path, root: Path, description: str) -> Path:
    path = _absolute(path)
    root = _require_real_directory(root, f"approved {description} root")
    if not os.path.lexists(path):
        raise CanaryError(f"{description} is missing: {path}")
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise CanaryError(
            f"{description} symlink resolves outside its approved root"
        ) from error
    mode = os.lstat(path).st_mode
    if path != resolved or stat.S_ISLNK(mode):
        raise CanaryError(f"{description} must not be a symlink")
    if not stat.S_ISREG(mode):
        raise CanaryError(f"{description} must be a regular file")
    if path.stat().st_size <= 0:
        raise CanaryError(f"{description} is empty")
    return path


def _require_hash(path: Path, expected: str, description: str) -> None:
    if sha256_file(path) != expected:
        raise CanaryError(f"{description} SHA-256 does not match the pinned contract")


def inspect_glb(path: Path, *, require_skin: bool = True) -> dict[str, Any]:
    """Parse the complete GLB container and its JSON document."""
    try:
        raw = Path(path).read_bytes()
        if len(raw) < 20:
            raise CanaryError("GLB is shorter than its header and first chunk")
        magic, version, declared_length = struct.unpack_from("<4sII", raw, 0)
        if magic != b"glTF" or version != 2 or declared_length != len(raw):
            raise CanaryError("GLB header is invalid or has a mismatched length")
        offset = 12
        chunks: list[tuple[int, bytes]] = []
        while offset < len(raw):
            if offset + 8 > len(raw):
                raise CanaryError("GLB chunk header is truncated")
            length, kind = struct.unpack_from("<II", raw, offset)
            offset += 8
            end = offset + length
            if length % 4 or end > len(raw):
                raise CanaryError("GLB chunk is truncated or not four-byte aligned")
            chunks.append((kind, raw[offset:end]))
            offset = end
        if offset != len(raw) or not chunks or chunks[0][0] != _GLB_JSON_CHUNK:
            raise CanaryError("GLB does not begin with a readable JSON chunk")
        document = json.loads(chunks[0][1].decode("utf-8").rstrip(" \t\r\n\x00"))
        if not isinstance(document, dict) or document.get("asset", {}).get("version") != "2.0":
            raise CanaryError("GLB JSON does not declare asset version 2.0")
        meshes = document.get("meshes")
        skins = document.get("skins")
        if not isinstance(meshes, list) or not meshes:
            raise CanaryError("GLB JSON contains no readable mesh")
        if require_skin and (not isinstance(skins, list) or not skins):
            raise CanaryError("GLB JSON contains no readable skin")
        return {
            "asset_version": "2.0",
            "chunk_count": len(chunks),
            "mesh_count": len(meshes),
            "skin_count": len(skins) if isinstance(skins, list) else 0,
        }
    except CanaryError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, struct.error) as error:
        raise CanaryError(f"GLB parse/readability check failed: {error}") from error


def _read_input_contract(
    input_glb: Path, input_manifest: Path, contract: CanaryContract
) -> dict[str, Any]:
    if input_glb.name == "cleaned.obj" or input_glb.suffix.lower() != ".glb":
        raise CanaryError("only the original packed-PBR male GLB is allowed; cleaned.obj is rejected")
    if _absolute(input_glb) != _absolute(contract.input_glb):
        raise CanaryError("input GLB path is not the exact pinned male input")
    if _absolute(input_manifest) != _absolute(contract.input_manifest):
        raise CanaryError("input manifest path is not the exact pinned male manifest")
    root = _absolute(contract.input_glb).parent
    input_glb = _require_regular_file(input_glb, root, "input GLB")
    input_manifest = _require_regular_file(input_manifest, root, "input manifest")
    _require_hash(input_glb, contract.input_glb_sha256, "input GLB")
    _require_hash(input_manifest, contract.input_manifest_sha256, "input manifest")
    try:
        manifest = json.loads(input_manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanaryError(f"input manifest is not readable JSON: {error}") from error
    output = manifest.get("output") if isinstance(manifest, dict) else None
    parameters = manifest.get("parameters") if isinstance(manifest, dict) else None
    model = manifest.get("model") if isinstance(manifest, dict) else None
    if manifest.get("backend") != "pixal3d" or not isinstance(output, dict):
        raise CanaryError("input manifest is not the authenticated Pixal3D output")
    if not isinstance(model, dict) or model.get("revision") != PIXAL3D_REVISION:
        raise CanaryError("input manifest Pixal3D revision is not pinned")
    if output.get("path") != str(input_glb) or output.get("bytes") != input_glb.stat().st_size:
        raise CanaryError("input manifest output descriptor does not identify the exact GLB")
    if output.get("sha256") != contract.input_glb_sha256:
        raise CanaryError("input manifest output hash does not identify the exact GLB")
    required_parameters = {
        "low_vram": True,
        "manual_fov": 0.2,
        "resolution": 1024,
        "seed": 42,
    }
    if not isinstance(parameters, dict) or any(
        parameters.get(key) != value for key, value in required_parameters.items()
    ):
        raise CanaryError("input manifest parameters are not the reviewed male canary")
    glb_readback = inspect_glb(input_glb, require_skin=False)
    return {
        "glb": {**_file_record(input_glb), "readback": glb_readback},
        "manifest": _file_record(input_manifest),
    }


def _run_process(
    process_runner: Callable[..., Any], command: Sequence[str], **kwargs: Any
) -> Any:
    result = process_runner([str(value) for value in command], **kwargs)
    returncode = getattr(result, "returncode", 0)
    if returncode != 0:
        raise CanarySubprocessError(result)
    return result


def _validate_checkpoint(
    *,
    root: Path,
    relative: str,
    expected_hash: str,
    expected_size: int,
    contract: CanaryContract,
    label: str,
) -> dict[str, Any]:
    logical = _absolute(root / relative)
    if contract.weight_cache_root is None:
        path = _require_regular_file(logical, root, label)
        _require_hash(path, expected_hash, label)
        if path.stat().st_size != expected_size:
            raise CanaryError(f"{label} size does not match the pinned contract")
        return {
            **_file_record(path),
            "logical_path": str(path),
            "resolved_path": str(path),
            "link_chain": [],
        }

    cache_root = _require_real_directory(contract.weight_cache_root, "SkinTokens weight cache root")
    snapshot = cache_root / "snapshots" / contract.model_revision
    snapshot_experiments = _require_real_directory(
        snapshot / "experiments", "pinned SkinTokens snapshot experiments"
    )
    experiments_link = root / "experiments"
    if not os.path.lexists(experiments_link) or not experiments_link.is_symlink():
        raise CanaryError("SkinTokens experiments must be the pinned snapshot symlink")
    if experiments_link.resolve() != snapshot_experiments:
        raise CanaryError("SkinTokens experiments symlink does not resolve to the pinned revision")
    if not os.path.lexists(logical):
        raise CanaryError(f"{label} is missing: {logical}")
    try:
        logical.relative_to(experiments_link)
    except ValueError as error:
        raise CanaryError(f"{label} logical path is outside experiments") from error
    if not logical.is_symlink():
        raise CanaryError(f"{label} must be a Hugging Face blob symlink")
    resolved = logical.resolve()
    expected_blob = cache_root / "blobs" / expected_hash
    if resolved != expected_blob:
        raise CanaryError(f"{label} does not resolve to the exact pinned blob")
    blobs_root = _require_real_directory(cache_root / "blobs", "SkinTokens weight blobs root")
    resolved = _require_regular_file(resolved, blobs_root, f"{label} blob")
    _require_hash(resolved, expected_hash, label)
    if resolved.stat().st_size != expected_size:
        raise CanaryError(f"{label} size does not match the pinned contract")
    return {
        "logical_path": str(logical),
        "resolved_path": str(resolved),
        "sha256": expected_hash,
        "size_bytes": expected_size,
        "link_chain": [
            {
                "path": str(experiments_link),
                "target": os.readlink(experiments_link),
                "resolved_path": str(snapshot_experiments),
            },
            {
                "path": str(logical),
                "target": os.readlink(logical),
                "resolved_path": str(resolved),
            },
        ],
    }


def _validate_qwen(root: Path, contract: CanaryContract) -> dict[str, Any]:
    logical_model = _absolute(root / "models/Qwen3-0.6B")
    expected_names = set(contract.qwen_file_hashes)
    if set(contract.qwen_file_sizes) != expected_names:
        raise CanaryError("Qwen file hash and size contracts differ")
    if contract.qwen_cache_root is None:
        model_dir = _require_real_directory(logical_model, "Qwen logical model directory")
        if {path.name for path in model_dir.iterdir()} != expected_names:
            raise CanaryError("Qwen config-only snapshot file set is not pinned")
        files: dict[str, Any] = {}
        for name in sorted(expected_names):
            path = _require_regular_file(model_dir / name, model_dir, f"Qwen {name}")
            _require_hash(path, contract.qwen_file_hashes[name], f"Qwen {name}")
            if path.stat().st_size != contract.qwen_file_sizes[name]:
                raise CanaryError(f"Qwen {name} size does not match the pinned contract")
            files[name] = {
                **_file_record(path),
                "logical_path": str(path),
                "resolved_path": str(path),
                "link_chain": [],
            }
        return {
            "revision": contract.qwen_revision,
            "logical_path": str(model_dir),
            "resolved_path": str(model_dir),
            "link_chain": [],
            "files": files,
        }

    cache_root = _require_real_directory(contract.qwen_cache_root, "Qwen cache root")
    snapshot = _require_real_directory(
        cache_root / "snapshots" / contract.qwen_revision, "pinned Qwen snapshot"
    )
    if not os.path.lexists(logical_model) or not logical_model.is_symlink():
        raise CanaryError("Qwen logical model path must be the pinned snapshot symlink")
    if logical_model.resolve() != snapshot:
        raise CanaryError("Qwen logical model symlink does not resolve to the pinned revision")
    if {path.name for path in snapshot.iterdir()} != expected_names:
        raise CanaryError("Qwen config-only snapshot file set is not pinned")
    blobs_root = _require_real_directory(cache_root / "blobs", "Qwen blobs root")
    files = {}
    for name in sorted(expected_names):
        logical_file = logical_model / name
        if not os.path.lexists(logical_file) or not logical_file.is_symlink():
            raise CanaryError(f"Qwen {name} must be a Hugging Face blob symlink")
        resolved = logical_file.resolve()
        resolved = _require_regular_file(resolved, blobs_root, f"Qwen {name} blob")
        _require_hash(resolved, contract.qwen_file_hashes[name], f"Qwen {name}")
        if resolved.stat().st_size != contract.qwen_file_sizes[name]:
            raise CanaryError(f"Qwen {name} size does not match the pinned contract")
        files[name] = {
            "logical_path": str(logical_file),
            "resolved_path": str(resolved),
            "sha256": contract.qwen_file_hashes[name],
            "size_bytes": contract.qwen_file_sizes[name],
            "link_chain": [
                {
                    "path": str(logical_model),
                    "target": os.readlink(logical_model),
                    "resolved_path": str(snapshot),
                },
                {
                    "path": str(logical_file),
                    "target": os.readlink(logical_file),
                    "resolved_path": str(resolved),
                },
            ],
        }
    return {
        "revision": contract.qwen_revision,
        "logical_path": str(logical_model),
        "resolved_path": str(snapshot),
        "link_chain": [
            {
                "path": str(logical_model),
                "target": os.readlink(logical_model),
                "resolved_path": str(snapshot),
            }
        ],
        "files": files,
    }


def _validate_skintokens(
    root: Path,
    contract: CanaryContract,
    process_runner: Callable[..., Any],
) -> dict[str, Any]:
    if _absolute(root) != _absolute(contract.skintokens_root):
        raise CanaryError("SkinTokens root is not the pinned checkout")
    root = _require_real_directory(root, "SkinTokens root")
    revision = _run_process(
        process_runner,
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if str(getattr(revision, "stdout", "")).strip() != contract.skintokens_commit:
        raise CanaryError("SkinTokens commit is not the pinned commit")
    status = _run_process(
        process_runner,
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
        check=False,
        capture_output=True,
        text=True,
    )
    if str(getattr(status, "stdout", "")).strip():
        raise CanaryError("SkinTokens checkout has tracked changes")

    code: dict[str, Any] = {}
    for relative, expected in contract.code_hashes.items():
        path = _require_regular_file(root / relative, root, f"SkinTokens code {relative}")
        _require_hash(path, expected, f"SkinTokens code {relative}")
        code[relative] = _file_record(path)
    weights: dict[str, Any] = {}
    for relative, expected in contract.checkpoint_hashes.items():
        label = "TokenRig checkpoint" if relative == TOKENRIG_CHECKPOINT_RELATIVE else "Skin VAE checkpoint"
        if relative not in contract.checkpoint_sizes:
            raise CanaryError(f"{label} has no pinned size")
        weights[relative] = _validate_checkpoint(
            root=root,
            relative=relative,
            expected_hash=expected,
            expected_size=contract.checkpoint_sizes[relative],
            contract=contract,
            label=label,
        )
    license_path = _require_regular_file(root / "LICENSE", root, "SkinTokens license")
    _require_hash(license_path, contract.license_hash, "SkinTokens license")
    readme_path = _require_regular_file(root / "README.md", root, "SkinTokens README")
    readme_record = _file_record(readme_path)
    training_provenance_risks = [
        {
            "training_source": source,
            "status": "unresolved_for_formal_registration",
            "upstream_reference": {
                "model_card_url": SKINTOKENS_MODEL_CARD_URL,
                "readme": readme_record,
                "readme_statement": (
                    "TokenRig checkpoint trained on ArticulationXL 2.0 + "
                    "VRoid Hub + ModelsResource and refined with GRPO"
                ),
            },
        }
        for source in TRAINING_PROVENANCE_SOURCES
    ]
    qwen = _validate_qwen(root, contract)
    python = root / ".venv/bin/python"
    if not python.is_file() or not os.access(python, os.X_OK):
        raise CanaryError(f"SkinTokens venv Python is missing or not executable: {python}")
    return {
        "root": str(root),
        "commit": contract.skintokens_commit,
        "model_revision": contract.model_revision,
        "code_hashes": code,
        "weight_hashes": weights,
        "license_hash": contract.license_hash,
        "license": _file_record(license_path),
        "training_provenance_risks": training_provenance_risks,
        "qwen": qwen,
        "python": str(python),
    }


def _offline_environment(base_env: Mapping[str, str] | None, *, seed: int) -> dict[str, str]:
    environment = dict(os.environ if base_env is None else base_env)
    environment.update(OFFLINE_ENVIRONMENT)
    environment["CUDA_VISIBLE_DEVICES"] = PHYSICAL_GPU
    environment["PYTHONHASHSEED"] = str(seed)
    environment["TOKENIZERS_PARALLELISM"] = "false"
    return environment


def default_runtime_probe(
    *,
    python_executable: Path,
    cwd: Path,
    env: Mapping[str, str],
    subprocess_runner: Callable[..., Any],
) -> dict[str, Any]:
    result = _run_process(
        subprocess_runner,
        [str(python_executable), "-c", RUNTIME_PROBE_SOURCE],
        cwd=str(cwd),
        env=dict(env),
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(str(getattr(result, "stdout", "")).strip())
    except json.JSONDecodeError as error:
        raise CanaryError("SkinTokens runtime probe did not return JSON") from error
    if not isinstance(payload, dict):
        raise CanaryError("SkinTokens runtime probe returned a non-object")
    return payload


def _validate_runtime_info(info: Mapping[str, Any]) -> dict[str, Any]:
    python_version = info.get("python")
    packages = info.get("packages")
    if not isinstance(python_version, str) or not isinstance(packages, Mapping):
        raise CanaryError("runtime probe is missing environment versions")
    if not python_version.startswith(EXPECTED_PYTHON_PREFIX):
        raise CanaryError(f"Python version must start with {EXPECTED_PYTHON_PREFIX}")
    for name, expected in EXPECTED_RUNTIME_PACKAGES.items():
        if packages.get(name) != expected:
            raise CanaryError(f"{name} version is not the pinned runtime version {expected}")
    if info.get("cuda") != EXPECTED_CUDA_VERSION:
        raise CanaryError(f"CUDA version must be {EXPECTED_CUDA_VERSION}")
    gpu = info.get("gpu")
    if not isinstance(gpu, Mapping):
        raise CanaryError("runtime probe is missing GPU information")
    if (
        gpu.get("available") is not True
        or gpu.get("visible_devices") != PHYSICAL_GPU
        or gpu.get("logical_index") != 0
        or not isinstance(gpu.get("name"), str)
        or not gpu.get("name")
    ):
        raise CanaryError("GPU must be available as logical index 0 with CUDA_VISIBLE_DEVICES=3")
    return json.loads(json.dumps(info))


def build_inference_command(
    *,
    python_executable: Path,
    demo_path: Path,
    input_glb: Path,
    output_glb: Path,
    checkpoint: Path,
    seed: int,
    use_skeleton_input: bool,
) -> list[str]:
    return _expected_inference_command(
        python_executable=python_executable,
        demo_path=demo_path,
        input_glb=input_glb,
        output_glb=output_glb,
        checkpoint=checkpoint,
        seed=seed,
        use_skeleton_input=use_skeleton_input,
    )


def _expected_inference_command(
    *,
    python_executable: Path,
    demo_path: Path,
    input_glb: Path,
    output_glb: Path,
    checkpoint: Path,
    seed: int,
    use_skeleton_input: bool,
) -> list[str]:
    command = [
        str(python_executable),
        "-c",
        SEED_BOOTSTRAP,
        str(seed),
        str(demo_path),
        "--input",
        str(input_glb),
        "--output",
        str(output_glb),
        "--top_k",
        str(SAMPLING_PARAMETERS["top_k"]),
        "--top_p",
        str(SAMPLING_PARAMETERS["top_p"]),
        "--temperature",
        str(SAMPLING_PARAMETERS["temperature"]),
        "--repetition_penalty",
        str(SAMPLING_PARAMETERS["repetition_penalty"]),
        "--num_beams",
        str(SAMPLING_PARAMETERS["num_beams"]),
        "--use_transfer",
    ]
    if use_skeleton_input:
        command.append("--use_skeleton")
    command.extend(["--model_ckpt", str(checkpoint)])
    return command


def _validate_command_recording(command: Sequence[str], expected: Sequence[str]) -> None:
    if list(command) != list(expected):
        raise CanaryError("unrecorded inference parameter or command mutation is rejected")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _snapshot_orchestrator(path: Path) -> dict[str, Any]:
    path = _absolute(path)
    path = _require_regular_file(path, path.parent, "orchestrator runner")
    return _file_record(path)


def assert_bpy_port_available(
    *, host: str = BPY_HOST, port: int = BPY_PORT, timeout: float = 0.25
) -> None:
    """Fail closed when the fixed SkinTokens bpy server port has a listener."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.settimeout(timeout)
        result = probe.connect_ex((host, port))
    finally:
        probe.close()
    if result == 0:
        raise CanaryError(f"BPY port {port} is already listening at {host}")
    if result != errno.ECONNREFUSED:
        raise CanaryError(
            f"BPY port {port} availability could not be proven: errno {result}"
        )


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _create_attempt_ledger(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        _write_fsync(path, _json_bytes(payload))
        _fsync_directory(path.parent)
    except FileExistsError as error:
        raise CanaryError(f"attempt ledger already exists: {path}") from error


def _replace_attempt_ledger(path: Path, payload: Mapping[str, Any]) -> None:
    current = _require_regular_file(path, path.parent, "attempt ledger")
    try:
        current_payload = json.loads(current.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanaryError("attempt ledger became unreadable") from error
    if current_payload.get("attempt_id") != payload.get("attempt_id"):
        raise CanaryError("attempt ledger identity changed")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_fsync(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as error:
        raise CanaryError("atomic no-replace publication is unavailable") from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise CanaryError(f"stale output already exists: {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _write_server_hygiene(
    staging: Path, seed: int
) -> tuple[Path, str, Path, Path]:
    patch_dir = staging / "runtime_patch"
    patch_dir.mkdir()
    patch_path = patch_dir / "sitecustomize.py"
    content = SERVER_HYGIENE_SOURCE.encode("utf-8")
    _write_fsync(patch_path, content)
    marker_dir = patch_dir / "markers"
    marker_dir.mkdir()
    load_audit_path = patch_dir / "load_audit.jsonl"
    return patch_dir, hashlib.sha256(content).hexdigest(), marker_dir, load_audit_path


def _validate_hygiene_markers(marker_dir: Path, patch_hash: str, seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    roles: set[str] = set()
    for path in sorted(marker_dir.glob("*.json")):
        path = _require_regular_file(path, marker_dir, "server hygiene marker")
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CanaryError("server hygiene marker is unreadable") from error
        argv = record.get("argv") if isinstance(record, dict) else None
        if record.get("patch_sha256") != patch_hash or record.get("seed") != seed:
            raise CanaryError("server hygiene marker does not match the patch and seed")
        if not isinstance(argv, list) or not argv or not isinstance(argv[0], str):
            raise CanaryError("server hygiene marker has no process argv")
        if argv[0] == "-c":
            role = "demo"
        elif Path(argv[0]).name == "bpy_server.py":
            role = "bpy_server"
        else:
            continue
        roles.add(role)
        records.append({
            "role": role,
            "pid": record.get("pid"),
            "marker": _file_record(path, public_path=Path("runtime_patch/markers") / path.name),
        })
    if roles != {"demo", "bpy_server"}:
        raise CanaryError("server hygiene patch did not run in both demo and bpy server processes")
    return records


def _validate_load_audit(
    audit_path: Path, patch_dir: Path, input_glb: Path
) -> dict[str, Any]:
    audit_path = _require_regular_file(
        audit_path, patch_dir, "bpy load hygiene audit"
    )
    try:
        events = [
            json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanaryError("bpy load hygiene audit is unreadable") from error
    expected = [
        (1, "before_clean"),
        (1, "after_clean"),
        (1, "after_import"),
        (2, "before_clean"),
        (2, "after_clean"),
        (2, "after_import"),
    ]
    if len(events) != 6 or not all(isinstance(event, Mapping) for event in events):
        raise CanaryError(
            "bpy load hygiene audit must contain exactly source then transfer target"
        )
    if [
        (event.get("sequence"), event.get("phase"))
        for event in events
    ] != expected:
        raise CanaryError(
            "bpy load hygiene audit must contain exactly source then transfer target"
        )
    default_names = {"Cube", "Camera", "Light"}
    expected_path = str(input_glb)
    for event in events:
        if event.get("filepath") != expected_path:
            raise CanaryError("bpy load hygiene audit contains an unexpected input")
        inventory = event.get("inventory")
        if not isinstance(inventory, Mapping) or not isinstance(
            inventory.get("objects"), list
        ):
            raise CanaryError("bpy load hygiene inventory is missing")
        objects = inventory["objects"]
        if not all(
            isinstance(item, Mapping)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("type"), str)
            for item in objects
        ):
            raise CanaryError("bpy load hygiene object inventory is invalid")
        if event["phase"] == "after_clean" and (
            objects
            or inventory.get("mesh_count") != 0
            or inventory.get("material_count") != 0
            or inventory.get("image_count") != 0
        ):
            raise CanaryError("bpy scene was not empty after clean")
        if (
            event["sequence"] == 2
            and event["phase"] == "before_clean"
            and inventory.get("mesh_count", 0) < 1
        ):
            raise CanaryError(
                "bpy transfer cleanup did not observe the prior source scene"
            )
        if event["phase"] == "after_import":
            names = {item["name"] for item in objects}
            if names & default_names or inventory.get("mesh_count", 0) < 1:
                raise CanaryError(
                    "bpy imported scene contains a default object or no mesh"
                )
    return {
        "relative_path": "runtime_patch/load_audit.jsonl",
        "sha256": sha256_file(audit_path),
        "size_bytes": audit_path.stat().st_size,
        "loads": [
            {
                "sequence": 1,
                "role": "source",
                "filepath": expected_path,
            },
            {
                "sequence": 2,
                "role": "transfer_target",
                "filepath": expected_path,
            },
        ],
    }


def _fsync_tree(root: Path) -> None:
    root = _require_real_directory(root, "staging evidence root")
    directories = [root]
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CanaryError(f"staging evidence contains a symlink: {path}")
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            descriptor = os.open(str(path), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        else:
            raise CanaryError(f"staging evidence contains a special file: {path}")
    for directory in reversed(directories):
        _fsync_directory(directory)


def _failure_evidence_inventory(root: Path) -> dict[str, Any]:
    root = _require_real_directory(root, "failed attempt evidence root")
    files: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if path.is_dir() and not path.is_symlink():
            continue
        path = _require_regular_file(path, root, "failed attempt evidence file")
        relative = path.relative_to(root).as_posix()
        files[relative] = _file_record(path)
    digest_payload = {
        relative: {
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
        for relative, record in files.items()
    }
    return {
        "path": str(root),
        "inventory_sha256": hashlib.sha256(_json_bytes(digest_payload)).hexdigest(),
        "files": files,
    }


def _require_recovery_file(
    path: Path,
    root: Path,
    expected_sha256: str,
    expected_size: int,
    description: str,
) -> Path:
    path = _require_regular_file(path, root, description)
    _require_hash(path, expected_sha256, description)
    if path.stat().st_size != expected_size:
        raise CanaryError(f"{description} size does not match the recovery contract")
    return path


def _ast_source_hash(source: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise CanaryError("could not extract pinned parser source")
    return hashlib.sha256(segment.encode("utf-8")).hexdigest()


def _upstream_clean_bpy_evidence(contract: RecoveryContract) -> dict[str, Any]:
    parser_path = _require_recovery_file(
        contract.parser_path,
        contract.skintokens_root,
        contract.parser_sha256,
        contract.parser_size,
        "pinned SkinTokens bpy parser",
    )
    try:
        source = parser_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise CanaryError("pinned SkinTokens bpy parser is unreadable") from error
    parser_class = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "BpyParser"
        ),
        None,
    )
    clean_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "clean_bpy"
        ),
        None,
    )
    load_node = (
        next(
            (
                node
                for node in parser_class.body
                if isinstance(node, ast.FunctionDef) and node.name == "load"
            ),
            None,
        )
        if parser_class is not None
        else None
    )
    if load_node is None or clean_node is None:
        raise CanaryError("pinned parser lacks BpyParser.load or clean_bpy")
    load_hash = _ast_source_hash(source, load_node)
    clean_hash = _ast_source_hash(source, clean_node)
    if load_hash != contract.bpyparser_load_sha256:
        raise CanaryError("BpyParser.load source hash does not match recovery contract")
    if clean_hash != contract.clean_bpy_sha256:
        raise CanaryError("clean_bpy source hash does not match recovery contract")
    calls = [node for node in ast.walk(load_node) if isinstance(node, ast.Call)]

    def call_lines(name: str) -> list[int]:
        return [
            node.lineno
            for node in calls
            if isinstance(node.func, ast.Name) and node.func.id == name
        ]

    clean_lines = call_lines("clean_bpy")
    import_lines = call_lines("load")
    if not clean_lines or not import_lines or min(clean_lines) >= min(import_lines):
        raise CanaryError("BpyParser.load must call clean_bpy before import")
    return {
        "skintokens_commit": contract.skintokens_commit,
        "parser": _file_record(parser_path),
        "bpyparser_load_sha256": load_hash,
        "clean_bpy_sha256": clean_hash,
        "bpyparser_load_calls_clean_before_import": True,
    }


def _descriptor_matches(
    value: Any, expected: Mapping[str, Any], description: str
) -> None:
    if not isinstance(value, Mapping):
        raise CanaryError(f"{description} descriptor is missing")
    for key in ("path", "sha256", "size_bytes"):
        if value.get(key) != expected.get(key):
            raise CanaryError(f"{description} descriptor does not match")


def _authenticate_failed_attempt(contract: RecoveryContract) -> dict[str, Any]:
    route_root = _require_real_directory(
        _absolute(contract.attempt_ledger).parent, "recovery route root"
    )
    expected_output_dir = route_root / contract.asset_id
    expected_failed_dir = route_root / f"{contract.asset_id}.tokenrig_failed_attempt"
    expected_attempt = route_root / f"{contract.asset_id}.tokenrig_attempt.json"
    if _absolute(contract.recovery_output_dir) != expected_output_dir:
        raise CanaryError("recovery output directory is not canonical")
    if _absolute(contract.failed_evidence_dir) != expected_failed_dir:
        raise CanaryError("failed evidence directory is not the exact sibling")
    if _absolute(contract.attempt_ledger) != expected_attempt:
        raise CanaryError("attempt ledger is not the exact route ledger")
    failed_dir = _require_real_directory(
        contract.failed_evidence_dir, "failed attempt evidence directory"
    )
    if _absolute(contract.failed_glb) != failed_dir / "tokenrig_transfer.glb":
        raise CanaryError("failed TokenRig GLB path is not canonical evidence")
    if _absolute(contract.inference_log) != failed_dir / "inference.log":
        raise CanaryError("failed inference log path is not canonical evidence")
    if _absolute(contract.sitecustomize_path) != failed_dir / "runtime_patch/sitecustomize.py":
        raise CanaryError("failed sitecustomize path is not canonical evidence")

    attempt_path = _require_recovery_file(
        contract.attempt_ledger,
        route_root,
        contract.attempt_ledger_sha256,
        contract.attempt_ledger_size,
        "direct attempt ledger",
    )
    try:
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanaryError("direct attempt ledger is unreadable") from error
    expected_error = {
        "type": "CanaryError",
        "message": "server hygiene patch did not run in both demo and bpy server processes",
    }
    if (
        not isinstance(attempt, Mapping)
        or attempt.get("schema") != "pixal_tokenrig_attempt_v1"
        or attempt.get("attempt_id") != contract.attempt_id
        or attempt.get("asset_id") != contract.asset_id
        or attempt.get("status") != "failed"
        or attempt.get("returncode") != 0
        or attempt.get("failure_stage") != "output_validation"
        or attempt.get("error") != expected_error
    ):
        raise CanaryError("direct attempt is not the exact recoverable hygiene failure")
    stderr = attempt.get("stderr")
    site_error = "Error in sitecustomize; set PYTHONVERBOSE for traceback:"
    module_error = "ModuleNotFoundError: No module named 'src'"
    if (
        not isinstance(stderr, str)
        or stderr.count(site_error) != 2
        or stderr.count(module_error) != 2
        or stderr.count("ModuleNotFoundError") != 2
    ):
        raise CanaryError(
            "recovery requires exactly two src sitecustomize ModuleNotFoundError records"
        )

    failed_glb = _require_recovery_file(
        contract.failed_glb,
        failed_dir,
        contract.failed_glb_sha256,
        contract.failed_glb_size,
        "failed TokenRig GLB",
    )
    inference_log = _require_recovery_file(
        contract.inference_log,
        failed_dir,
        contract.inference_log_sha256,
        contract.inference_log_size,
        "failed inference log",
    )
    sitecustomize_path = _require_recovery_file(
        contract.sitecustomize_path,
        failed_dir,
        contract.sitecustomize_sha256,
        contract.sitecustomize_size,
        "failed sitecustomize",
    )
    log_text = inference_log.read_text(encoding="utf-8")
    if str(attempt.get("stdout", "")) not in log_text or stderr not in log_text:
        raise CanaryError("inference log does not contain exact ledger stdout/stderr")
    markers_dir = _require_real_directory(
        failed_dir / "runtime_patch/markers", "failed hygiene markers directory"
    )
    if list(markers_dir.iterdir()):
        raise CanaryError("failed hygiene markers directory must be empty")
    if os.path.lexists(failed_dir / "runtime_patch/load_audit.jsonl"):
        raise CanaryError("failed attempt unexpectedly contains a load audit")

    inventory = _failure_evidence_inventory(failed_dir)
    if inventory.get("inventory_sha256") != contract.failed_inventory_sha256:
        raise CanaryError("failed evidence inventory hash does not match")
    if attempt.get("failed_evidence") != inventory:
        raise CanaryError("attempt ledger failed evidence does not match current bytes")
    patch_record = inventory["files"].get("runtime_patch/sitecustomize.py")
    _descriptor_matches(
        patch_record, _file_record(sitecustomize_path), "failed sitecustomize"
    )

    input_glb = _require_recovery_file(
        contract.input_glb,
        _absolute(contract.input_glb).parent,
        contract.input_glb_sha256,
        contract.input_glb_size,
        "recovery input GLB",
    )
    input_manifest = _require_recovery_file(
        contract.input_manifest,
        _absolute(contract.input_manifest).parent,
        contract.input_manifest_sha256,
        contract.input_manifest_size,
        "recovery input manifest",
    )
    input_records = {
        "glb": _file_record(input_glb),
        "manifest": _file_record(input_manifest),
    }
    authenticated = attempt.get("authenticated_hashes")
    if not isinstance(authenticated, Mapping):
        raise CanaryError("attempt ledger authenticated hashes are missing")
    recorded_input = authenticated.get("input")
    if not isinstance(recorded_input, Mapping):
        raise CanaryError("attempt ledger input provenance is missing")
    _descriptor_matches(recorded_input.get("glb"), input_records["glb"], "input GLB")
    _descriptor_matches(
        recorded_input.get("manifest"), input_records["manifest"], "input manifest"
    )
    recorded_skintokens = authenticated.get("skintokens")
    if (
        not isinstance(recorded_skintokens, Mapping)
        or recorded_skintokens.get("commit") != contract.skintokens_commit
    ):
        raise CanaryError("attempt ledger SkinTokens commit is not pinned")

    parser_evidence = _upstream_clean_bpy_evidence(contract)
    output_record = {
        **_file_record(failed_glb),
        "readback": inspect_glb(failed_glb),
    }
    return {
        "attempt": attempt,
        "attempt_ledger": _file_record(attempt_path),
        "failed_evidence": inventory,
        "inference_log": _file_record(inference_log),
        "input": input_records,
        "output": output_record,
        "sitecustomize": _file_record(sitecustomize_path),
        "upstream_clean_bpy": parser_evidence,
        "sitecustomize_import_failure": {
            "exception": module_error,
            "occurrences": 2,
        },
    }


def recover_failed_attempt(
    *, contract: RecoveryContract = PINNED_RECOVERY_CONTRACT
) -> Path:
    """Publish an honest, no-replace recovery manifest without rerunning inference."""
    evidence = _authenticate_failed_attempt(contract)
    recovery_output_dir = _absolute(contract.recovery_output_dir)
    recovery_output_dir.parent.mkdir(parents=True, exist_ok=True)
    _require_real_directory(recovery_output_dir.parent, "recovery output root")
    if os.path.lexists(recovery_output_dir):
        raise CanaryError(f"stale output already exists: {recovery_output_dir}")
    builder_snapshot = _snapshot_orchestrator(RUNNER_PATH)
    staging: Path | None = Path(
        tempfile.mkdtemp(
            prefix=f".{recovery_output_dir.name}.recovery.",
            suffix=".staging",
            dir=str(recovery_output_dir.parent),
        )
    )
    try:
        attempt = evidence["attempt"]
        manifest = {
            "schema": "pixal_tokenrig_recovery_v1",
            "asset_id": contract.asset_id,
            "attempt": "direct_transfer_recovered_from_hygiene_assertion",
            "state_classification": "research_candidate_recovered_from_hygiene_assertion",
            "task3_gate_status": "failed",
            "pbr_validation_status": "pending_static_audit",
            "source_front": "positive-y",
            "canonical_front": "negative-y",
            "input": evidence["input"],
            "output": evidence["output"],
            "recovery": {
                "returncode": attempt["returncode"],
                "failure_stage": attempt["failure_stage"],
                "error": attempt["error"],
                "sitecustomize_import_failure": evidence[
                    "sitecustomize_import_failure"
                ],
                "attempt_ledger": evidence["attempt_ledger"],
                "inference_log": evidence["inference_log"],
                "failed_evidence": evidence["failed_evidence"],
                "failed_sitecustomize": evidence["sitecustomize"],
                "upstream_clean_bpy": evidence["upstream_clean_bpy"],
                "recovery_builder": builder_snapshot,
                "task3_passed": False,
            },
        }
        _write_fsync(staging / "tokenrig_manifest.json", _json_bytes(manifest))
        _fsync_directory(staging)
        if _authenticate_failed_attempt(contract) != evidence:
            raise CanaryError("failed attempt evidence changed during recovery staging")
        if _snapshot_orchestrator(RUNNER_PATH) != builder_snapshot:
            raise CanaryError("recovery builder changed during staging")
        _rename_noreplace(staging, recovery_output_dir)
        staging = None
        _fsync_directory(recovery_output_dir.parent)
        return recovery_output_dir / "tokenrig_manifest.json"
    except BaseException:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)
        raise


def _prepare_output(output_dir: Path, contract: CanaryContract) -> Path:
    output_dir = _absolute(output_dir)
    if output_dir != _absolute(contract.output_dir):
        raise CanaryError("output directory is outside the approved canary root")
    if output_dir == _absolute(contract.input_glb) or output_dir / "tokenrig_transfer.glb" == _absolute(contract.input_glb):
        raise CanaryError("output aliases the input GLB")
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    _require_real_directory(parent, "approved output root")
    if os.path.lexists(output_dir):
        raise CanaryError(f"stale output already exists: {output_dir}")
    failed_evidence = (
        output_dir.parent / f"{output_dir.name}.tokenrig_failed_attempt"
    )
    if os.path.lexists(failed_evidence):
        raise CanaryError(f"stale failed-attempt evidence already exists: {failed_evidence}")
    return output_dir


def run_canary(
    *,
    input_glb: Path,
    input_manifest: Path,
    output_dir: Path,
    skintokens_root: Path,
    model_revision: str,
    seed: int = 42,
    use_skeleton_input: bool = False,
    contract: CanaryContract = PINNED_CONTRACT,
    subprocess_runner: Callable[..., Any] = subprocess.run,
    runtime_probe: Callable[..., Mapping[str, Any]] = default_runtime_probe,
    bpy_port_probe: Callable[[], None] = assert_bpy_port_available,
    orchestrator_path: Path | None = None,
    base_env: Mapping[str, str] | None = None,
) -> Path:
    """Validate, execute into staging, then atomically publish one canary."""
    input_glb = _absolute(input_glb)
    input_manifest = _absolute(input_manifest)
    output_dir = _prepare_output(output_dir, contract)
    if model_revision != contract.model_revision:
        raise CanaryError("model revision is not the pinned SkinTokens weight revision")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 2**32:
        raise CanaryError("seed must be an integer in [0, 2**32)")

    input_snapshot = _read_input_contract(input_glb, input_manifest, contract)
    orchestrator_snapshot = _snapshot_orchestrator(
        RUNNER_PATH if orchestrator_path is None else orchestrator_path
    )
    skintokens = _validate_skintokens(skintokens_root, contract, subprocess_runner)
    environment = _offline_environment(base_env, seed=seed)
    runtime = _validate_runtime_info(
        runtime_probe(
            python_executable=Path(skintokens["python"]),
            cwd=_absolute(skintokens_root),
            env=environment,
            subprocess_runner=subprocess_runner,
        )
    )

    staging: Path | None = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".staging",
            dir=str(output_dir.parent),
        )
    )
    attempt_path = (
        output_dir.parent / f"{output_dir.name}.tokenrig_attempt.json"
    )
    attempt: dict[str, Any] | None = None
    attempt_created = False
    child_started = False
    failure_stage = "staging_setup"
    inference_result: Any = None
    failed_evidence_dir = (
        output_dir.parent / f"{output_dir.name}.tokenrig_failed_attempt"
    )
    try:
        patch_dir, patch_hash, marker_dir, load_audit_path = _write_server_hygiene(
            staging, seed
        )
        environment = dict(environment)
        previous_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(patch_dir), str(_absolute(skintokens_root)))
        ) + (
            os.pathsep + previous_pythonpath if previous_pythonpath else ""
        )
        environment["TOKENRIG_CANARY_SEED"] = str(seed)
        environment["TOKENRIG_HYGIENE_MARKER_DIR"] = str(marker_dir)
        environment["TOKENRIG_SERVER_HYGIENE_SHA256"] = patch_hash
        environment["TOKENRIG_LOAD_AUDIT_PATH"] = str(load_audit_path)

        staged_output = staging / "tokenrig_transfer.glb"
        checkpoint = _absolute(skintokens_root) / TOKENRIG_CHECKPOINT_RELATIVE
        command_kwargs = {
            "python_executable": Path(skintokens["python"]),
            "demo_path": _absolute(skintokens_root) / "demo.py",
            "input_glb": input_glb,
            "output_glb": staged_output,
            "checkpoint": checkpoint,
            "seed": seed,
            "use_skeleton_input": use_skeleton_input,
        }
        command = build_inference_command(**command_kwargs)
        expected_command = _expected_inference_command(**command_kwargs)
        _validate_command_recording(command, expected_command)

        started_at = _utc_now()
        attempt_id = hashlib.sha256(
            _json_bytes(
                {
                    "command": command,
                    "orchestrator": orchestrator_snapshot,
                    "started_at_utc": started_at,
                }
            )
        ).hexdigest()
        attempt = {
            "schema": "pixal_tokenrig_attempt_v1",
            "attempt_id": attempt_id,
            "asset_id": contract.asset_id,
            "status": "started",
            "started_at_utc": started_at,
            "ended_at_utc": None,
            "command": command,
            "working_directory": str(_absolute(skintokens_root)),
            "output_dir": str(output_dir),
            "returncode": None,
            "stdout": "",
            "stderr": "",
            "failure_stage": "bpy_port_preflight",
            "error": None,
            "authenticated_hashes": {
                "input": input_snapshot,
                "skintokens": skintokens,
                "orchestrator": orchestrator_snapshot,
                "server_hygiene_patch": {
                    "sha256": patch_hash,
                    "size_bytes": len(SERVER_HYGIENE_SOURCE.encode("utf-8")),
                },
            },
        }
        _create_attempt_ledger(attempt_path, attempt)
        attempt_created = True

        failure_stage = "bpy_port_preflight"
        bpy_port_probe()
        failure_stage = "inference_subprocess"
        child_started = True
        try:
            inference_result = _run_process(
                subprocess_runner,
                command,
                cwd=str(_absolute(skintokens_root)),
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
        except CanarySubprocessError as error:
            inference_result = error.result
            raise
        stdout = getattr(inference_result, "stdout", "") or ""
        stderr = getattr(inference_result, "stderr", "") or ""
        inference_log = staging / "inference.log"
        log_bytes = (
            "=== stdout ===\n"
            + str(stdout)
            + ("\n" if stdout and not str(stdout).endswith("\n") else "")
            + "=== stderr ===\n"
            + str(stderr)
            + ("\n" if stderr and not str(stderr).endswith("\n") else "")
        ).encode("utf-8", errors="replace")
        _write_fsync(inference_log, log_bytes)

        failure_stage = "output_validation"
        staged_output = _require_regular_file(staged_output, staging, "TokenRig output GLB")
        if os.path.samefile(staged_output, input_glb):
            raise CanaryError("TokenRig output aliases the input GLB")
        glb_readback = inspect_glb(staged_output)
        processes = _validate_hygiene_markers(marker_dir, patch_hash, seed)
        load_audit = _validate_load_audit(load_audit_path, patch_dir, input_glb)

        failure_stage = "provenance_revalidation"
        if _read_input_contract(input_glb, input_manifest, contract) != input_snapshot:
            raise CanaryError("authenticated input changed during inference")
        current_skintokens = _validate_skintokens(skintokens_root, contract, subprocess_runner)
        if current_skintokens != skintokens:
            raise CanaryError("SkinTokens code, weights, or license changed during inference")
        if _snapshot_orchestrator(
            RUNNER_PATH if orchestrator_path is None else orchestrator_path
        ) != orchestrator_snapshot:
            raise CanaryError("orchestrator changed during inference")

        attempt.update(
            {
                "status": "succeeded",
                "ended_at_utc": _utc_now(),
                "returncode": int(getattr(inference_result, "returncode", 0)),
                "stdout": str(stdout),
                "stderr": str(stderr),
                "failure_stage": None,
                "error": None,
            }
        )
        failure_stage = "attempt_finalization"
        _replace_attempt_ledger(attempt_path, attempt)
        attempt_record = _file_record(attempt_path)

        public_output = output_dir / "tokenrig_transfer.glb"
        inference_parameters = {
            "hf_path": None,
            "input": str(input_glb),
            "model_checkpoint": str(checkpoint),
            "staging_output": str(staged_output),
            "use_postprocess": False,
            "use_skeleton": use_skeleton_input,
            "use_transfer": True,
        }
        manifest = {
            "schema": SCHEMA,
            "asset_id": contract.asset_id,
            "attempt": "fitted_skeleton_transfer" if use_skeleton_input else "direct_transfer",
            "state_classification": "research_candidate",
            "source_front": "positive-y",
            "canonical_front": "negative-y",
            "input": input_snapshot,
            "orchestrator": {
                "runner": orchestrator_snapshot,
                "provenance_schema": SCHEMA,
            },
            "skintokens": skintokens,
            "command": command,
            "working_directory": str(_absolute(skintokens_root)),
            "offline_environment": {key: environment[key] for key in OFFLINE_ENVIRONMENT},
            "inference_parameters": inference_parameters,
            "sampling_parameters": dict(SAMPLING_PARAMETERS),
            "random_parameters": {
                "seed": seed,
                "seed_bootstrap_before_demo_import": True,
                "seeded_libraries": ["random", "numpy", "torch", "torch.cuda"],
            },
            "environment": runtime,
            "gpu": runtime["gpu"],
            "inference_log": _file_record(
                inference_log, public_path=output_dir / "inference.log"
            ),
            "attempt_ledger": attempt_record,
            "server_hygiene": {
                "mechanism": "injected_sitecustomize_v1",
                "relative_path": "runtime_patch/sitecustomize.py",
                "sha256": patch_hash,
                "cleans_before_every_bpyparser_load": True,
                "pbr_preservation_requested_by_use_transfer": True,
                "pbr_validation_status": "pending_static_audit",
                "server_runtime": {
                    "bpy": runtime["packages"]["bpy"],
                    "purpose": "SkinTokens embedded transfer server",
                },
                "processes": processes,
                "load_audit": {
                    key: value
                    for key, value in load_audit.items()
                    if key != "loads"
                },
                "loads": load_audit["loads"],
            },
            "output": {
                **_file_record(staged_output, public_path=public_output),
                "readback": glb_readback,
            },
        }
        failure_stage = "manifest_publication"
        manifest_bytes = _json_bytes(manifest)
        _write_fsync(staging / "tokenrig_manifest.json", manifest_bytes)
        _fsync_directory(marker_dir)
        _fsync_directory(patch_dir)
        _fsync_directory(staging)
        failure_stage = "canonical_publication"
        _rename_noreplace(staging, output_dir)
        staging = None
        _fsync_directory(output_dir.parent)
        return output_dir / "tokenrig_manifest.json"
    except BaseException as error:
        failed_evidence: dict[str, Any] | None = None
        if child_started and staging is not None and staging.exists():
            try:
                _fsync_tree(staging)
                _rename_noreplace(staging, failed_evidence_dir)
                staging = None
                _fsync_directory(failed_evidence_dir.parent)
                failed_evidence = _failure_evidence_inventory(
                    failed_evidence_dir
                )
            except BaseException as preservation_error:
                failed_evidence = {
                    "path": str(staging),
                    "preservation_error": {
                        "type": type(preservation_error).__name__,
                        "message": str(preservation_error),
                    },
                }
        if attempt_created and attempt is not None:
            returncode = (
                getattr(inference_result, "returncode", None)
                if inference_result is not None
                else None
            )
            stdout = (
                getattr(inference_result, "stdout", "") or ""
                if inference_result is not None
                else ""
            )
            stderr = (
                getattr(inference_result, "stderr", "") or ""
                if inference_result is not None
                else ""
            )
            attempt.update(
                {
                    "status": "failed",
                    "ended_at_utc": _utc_now(),
                    "returncode": returncode,
                    "stdout": str(stdout),
                    "stderr": str(stderr),
                    "failure_stage": failure_stage,
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                    "failed_evidence": failed_evidence,
                }
            )
            try:
                _replace_attempt_ledger(attempt_path, attempt)
            except BaseException:
                pass
        if (
            not child_started
            and staging is not None
            and staging.exists()
        ):
            shutil.rmtree(staging)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skintokens-root", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-skeleton-input", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_canary(
        input_glb=args.input_glb,
        input_manifest=args.input_manifest,
        output_dir=args.output_dir,
        skintokens_root=args.skintokens_root,
        model_revision=args.model_revision,
        seed=args.seed,
        use_skeleton_input=args.use_skeleton_input,
    )
    print(f"TOKENRIG_CANARY_PUBLISHED {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
