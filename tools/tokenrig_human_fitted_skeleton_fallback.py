#!/usr/bin/env python3
"""Run the one required skeleton-conditioned TokenRig fallback on GPU 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import blender_tokenrig_human_static_audit as static_audit
from tools import tokenrig_human_canary as base


ASSET_ID = "rocketbox_male_adult_01"
SCHEMA = "pixal_tokenrig_fitted_skeleton_v1"
RUNNER_PATH = Path(__file__).resolve()
_REPO_ROOT = RUNNER_PATH.parents[1]
_ROUTE_ROOT = _REPO_ROOT / "tmp/pixal_tokenrig_route2_v1"
_ASSET_ROOT = _ROUTE_ROOT / ASSET_ID
_DIRECT_FAILED_ROOT = _ROUTE_ROOT / f"{ASSET_ID}.tokenrig_failed_attempt"


class FallbackError(base.CanaryError):
    """Raised when the required fitted-skeleton fallback is not authenticated."""


@dataclass(frozen=True)
class FilePin:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class FallbackContract:
    asset_id: str
    input_glb: Path
    input_glb_sha256: str
    input_glb_size: int
    recovery_manifest: Path
    recovery_manifest_sha256: str
    recovery_manifest_size: int
    original_source_glb: Path
    original_source_glb_sha256: str
    static_failures: tuple[FilePin, FilePin]
    output_dir: Path


PINNED_FALLBACK = FallbackContract(
    asset_id=ASSET_ID,
    input_glb=_DIRECT_FAILED_ROOT / "tokenrig_transfer.glb",
    input_glb_sha256="8606c013fba02f722e1d5c65accddc4398eab1fa925467a9233aaf458d93f01c",
    input_glb_size=50_843_552,
    recovery_manifest=_ASSET_ROOT / "tokenrig_manifest.json",
    recovery_manifest_sha256="cae4aac8f6472b893ce695173ad9a1766ef8f2ecf86cbbc7a80440b2ad949e96",
    recovery_manifest_size=5_071,
    original_source_glb=base.PINNED_CONTRACT.input_glb,
    original_source_glb_sha256=base.PINNED_CONTRACT.input_glb_sha256,
    static_failures=(
        FilePin(
            path=_ASSET_ROOT
            / "static_audit_v1.failed.554c2e6ca92a409497c80b39673b60b7.json",
            sha256="0eab61c2dfcb5a7fe0a05ee8f5109c60a9c756a6ae048a0545df6d9e64c590ce",
            size_bytes=1_127,
        ),
        FilePin(
            path=_ASSET_ROOT
            / "static_audit_v1.failed.f868faf6c0ea4870aef5e114c403eb9c.json",
            sha256="c78d5f1e4d7c127781f8b24b5a9bed906f96b5034d3b5bda9509c23c8f492e26",
            size_bytes=1_154,
        ),
    ),
    output_dir=_ASSET_ROOT / "fitted_skeleton_v1",
)


PINNED_BASE_CONTRACT = replace(
    base.PINNED_CONTRACT,
    input_glb=PINNED_FALLBACK.input_glb,
    input_manifest=PINNED_FALLBACK.recovery_manifest,
    input_glb_sha256=PINNED_FALLBACK.input_glb_sha256,
    input_manifest_sha256=PINNED_FALLBACK.recovery_manifest_sha256,
    output_dir=PINNED_FALLBACK.output_dir,
)


DEFERRED_HYGIENE_SOURCE = r'''\
"""Deferred fail-closed scene hygiene for SkinTokens child processes."""
import builtins
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

_load_sequence = 0
_parser_hook_installed = False
_load_audit_path = Path(os.environ["TOKENRIG_LOAD_AUDIT_PATH"])
_load_audit_path.parent.mkdir(parents=True, exist_ok=True)

def _inventory(parser):
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

def _record_load_event(parser, sequence, phase, filepath):
    event = {
        "sequence": sequence,
        "phase": phase,
        "filepath": str(Path(filepath).resolve()),
        "inventory": _inventory(parser),
    }
    payload = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
    with _load_audit_path.open("ab") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())

def _install_parser_hook():
    global _parser_hook_installed
    if _parser_hook_installed:
        return
    parser = sys.modules.get("src.rig_package.parser.bpy")
    if parser is None or not hasattr(parser, "BpyParser"):
        return
    original_load = parser.BpyParser.load.__func__

    def _clean_load(cls, filepath, **kwargs):
        global _load_sequence
        _load_sequence += 1
        sequence = _load_sequence
        _record_load_event(parser, sequence, "before_clean", filepath)
        parser.clean_bpy()
        _record_load_event(parser, sequence, "after_clean", filepath)
        result = original_load(cls, filepath, **kwargs)
        _record_load_event(parser, sequence, "after_import", filepath)
        return result

    parser.BpyParser.load = classmethod(_clean_load)
    _parser_hook_installed = True

_original_import = builtins.__import__
def _deferred_import(name, globals=None, locals=None, fromlist=(), level=0):
    result = _original_import(name, globals, locals, fromlist, level)
    _install_parser_hook()
    return result

builtins.__import__ = _deferred_import
_install_parser_hook()

marker_dir = Path(os.environ["TOKENRIG_HYGIENE_MARKER_DIR"])
marker_dir.mkdir(parents=True, exist_ok=True)
marker = {
    "argv": list(sys.argv),
    "patch_sha256": os.environ["TOKENRIG_SERVER_HYGIENE_SHA256"],
    "pid": os.getpid(),
    "seed": seed,
}
marker_bytes = (json.dumps(marker, sort_keys=True) + "\n").encode("utf-8")
with (marker_dir / f"{os.getpid()}.json").open("xb") as stream:
    stream.write(marker_bytes)
    stream.flush()
    os.fsync(stream.fileno())
'''


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": base.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _require_pin(pin: FilePin, description: str) -> Path:
    path = _absolute(pin.path)
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise FallbackError(f"{description} must be a direct regular file")
    if path.stat().st_size != pin.size_bytes or base.sha256_file(path) != pin.sha256:
        raise FallbackError(f"{description} hash/size changed")
    return path


def validate_static_failure_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(payloads) != 2:
        raise FallbackError("exactly two direct static failure records are required")
    messages = []
    for payload in payloads:
        if (
            payload.get("decision") != "rejected"
            or payload.get("readiness_bundle_published") is not False
            or not isinstance(payload.get("failure"), Mapping)
        ):
            raise FallbackError("direct static evidence is not a preserved rejection")
        messages.append(str(payload["failure"].get("message", "")))
    if "raw GLB triangle count changed" not in messages[0]:
        raise FallbackError("strict topology failure evidence is missing")
    if "opposite-limb contamination" not in messages[1]:
        raise FallbackError("opposite-limb failure evidence is missing")
    return {
        "strict_topology_attempt": "rejected",
        "bounded_skin_attempt": "rejected",
        "animation_authorized": False,
    }


def _read_fitted_input_contract(
    input_glb: Path, input_manifest: Path, contract: base.CanaryContract
) -> dict[str, Any]:
    pinned = PINNED_FALLBACK
    if (
        _absolute(input_glb) != _absolute(pinned.input_glb)
        or _absolute(input_manifest) != _absolute(pinned.recovery_manifest)
        or _absolute(contract.input_glb) != _absolute(pinned.input_glb)
        or _absolute(contract.input_manifest) != _absolute(pinned.recovery_manifest)
        or _absolute(contract.output_dir) != _absolute(pinned.output_dir)
    ):
        raise FallbackError("fitted fallback paths are not the exact pinned contract")
    direct_glb = _require_pin(
        FilePin(pinned.input_glb, pinned.input_glb_sha256, pinned.input_glb_size),
        "preserved direct TokenRig GLB",
    )
    recovery_manifest = _require_pin(
        FilePin(
            pinned.recovery_manifest,
            pinned.recovery_manifest_sha256,
            pinned.recovery_manifest_size,
        ),
        "honest recovery manifest",
    )
    try:
        recovery = json.loads(recovery_manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FallbackError(f"recovery manifest is invalid: {exc}") from exc
    output = recovery.get("output") if isinstance(recovery, Mapping) else None
    if (
        recovery.get("schema") != "pixal_tokenrig_recovery_v1"
        or recovery.get("asset_id") != ASSET_ID
        or recovery.get("task3_gate_status") != "failed"
        or not isinstance(output, Mapping)
        or any(
            output.get(field) != _file_record(direct_glb).get(field)
            for field in ("path", "sha256", "size_bytes")
        )
    ):
        raise FallbackError("recovery manifest does not authenticate the direct GLB")

    failure_payloads = []
    failure_records = []
    for pin in pinned.static_failures:
        path = _require_pin(pin, "immutable direct static failure")
        if path.stat().st_mode & 0o222:
            raise FallbackError("direct static failure evidence is mutable")
        try:
            failure_payloads.append(json.loads(path.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FallbackError(f"direct static failure evidence is invalid: {exc}")
        failure_records.append(_file_record(path))
    failure_summary = validate_static_failure_payloads(tuple(failure_payloads))

    original_source = _require_pin(
        FilePin(
            pinned.original_source_glb,
            pinned.original_source_glb_sha256,
            _absolute(pinned.original_source_glb).stat().st_size,
        ),
        "original Pixal PBR source",
    )
    original_source_manifest = _require_pin(
        FilePin(
            base.PINNED_CONTRACT.input_manifest,
            base.PINNED_CONTRACT.input_manifest_sha256,
            _absolute(base.PINNED_CONTRACT.input_manifest).stat().st_size,
        ),
        "original Pixal source manifest",
    )
    source_parsed = static_audit.read_glb(original_source)
    direct_parsed = static_audit.read_glb(direct_glb)
    pbr = static_audit.compare_pbr_payloads(
        static_audit.pbr_payload_contract(source_parsed),
        static_audit.pbr_payload_contract(direct_parsed),
    )
    surface = static_audit.analyze_raw_serialization_equivalence(
        source_parsed, direct_parsed
    )
    skins = direct_parsed.document.get("skins")
    if (
        not isinstance(skins, list)
        or len(skins) != 1
        or not isinstance(skins[0], Mapping)
        or not isinstance(skins[0].get("joints"), list)
        or len(skins[0]["joints"]) != 52
    ):
        raise FallbackError("conditioning derivative must contain the exact 52-bone skin")
    return {
        "glb": {**_file_record(direct_glb), "readback": base.inspect_glb(direct_glb)},
        "manifest": _file_record(recovery_manifest),
        "fallback_provenance": {
            "policy": "required_fitted_skeleton_after_direct_skin_rejection",
            "conditioning_joint_count": 52,
            "conditioning_uses_pixal_derivative_surface": True,
            "rocketbox_mesh_used": False,
            "original_source_glb": _file_record(original_source),
            "original_source_manifest": _file_record(original_source_manifest),
            "pbr": pbr,
            "bounded_surface_equivalence": surface,
            "static_failures": failure_records,
            "static_failure_summary": failure_summary,
            "animation_authorized": False,
        },
    }


def augment_fitted_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(manifest))
    parameters = result.get("inference_parameters")
    input_record = result.get("input")
    if (
        result.get("schema") != base.SCHEMA
        or result.get("asset_id") != ASSET_ID
        or result.get("attempt") != "fitted_skeleton_transfer"
        or not isinstance(parameters, Mapping)
        or parameters.get("use_skeleton") is not True
        or parameters.get("use_transfer") is not True
        or not isinstance(input_record, Mapping)
        or not isinstance(input_record.get("glb"), Mapping)
        or not isinstance(input_record.get("fallback_provenance"), Mapping)
    ):
        raise FallbackError(
            "base manifest is not the forced use_skeleton skeleton-conditioned fallback"
        )
    result.update(
        {
            "schema": SCHEMA,
            "base_runner_schema": base.SCHEMA,
            "state_classification": "research_candidate_fitted_skeleton_fallback",
            "task3_direct_gate_status": "failed",
            "static_audit_status": "pending_fitted_static_audit",
            "animation_authorized": False,
            "pbr_validation_status": "pending_static_audit",
            "fitted_skeleton": {
                "use_skeleton_input": True,
                "conditioning_source": input_record["glb"],
                "provenance": input_record["fallback_provenance"],
            },
        }
    )
    return result


def _replace_json_fsync(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(
                (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        base._fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_base_call(*, seed: int = 42) -> dict[str, Any]:
    return {
        "input_glb": PINNED_FALLBACK.input_glb,
        "input_manifest": PINNED_FALLBACK.recovery_manifest,
        "output_dir": PINNED_FALLBACK.output_dir,
        "skintokens_root": PINNED_BASE_CONTRACT.skintokens_root,
        "model_revision": PINNED_BASE_CONTRACT.model_revision,
        "seed": seed,
        "use_skeleton_input": True,
        "contract": PINNED_BASE_CONTRACT,
        "orchestrator_path": RUNNER_PATH,
    }


def run_fitted_skeleton(
    *,
    seed: int = 42,
    subprocess_runner: Any = subprocess.run,
    runtime_probe: Any = base.default_runtime_probe,
    bpy_port_probe: Any = base.assert_bpy_port_available,
) -> Path:
    original_reader = base._read_input_contract
    original_hygiene = base.SERVER_HYGIENE_SOURCE
    original_rename = base._rename_noreplace
    output_dir = _absolute(PINNED_FALLBACK.output_dir)

    def fitted_rename(source: Path, destination: Path) -> None:
        source = Path(source)
        destination = _absolute(destination)
        manifest_path = source / "tokenrig_manifest.json"
        if destination == output_dir and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise FallbackError(f"staged base manifest is invalid: {exc}") from exc
            _replace_json_fsync(manifest_path, augment_fitted_manifest(manifest))
        original_rename(source, destination)

    base._read_input_contract = _read_fitted_input_contract
    base.SERVER_HYGIENE_SOURCE = DEFERRED_HYGIENE_SOURCE
    base._rename_noreplace = fitted_rename
    try:
        manifest_path = base.run_canary(
            **build_base_call(seed=seed),
            subprocess_runner=subprocess_runner,
            runtime_probe=runtime_probe,
            bpy_port_probe=bpy_port_probe,
        )
    finally:
        base._read_input_contract = original_reader
        base.SERVER_HYGIENE_SOURCE = original_hygiene
        base._rename_noreplace = original_rename
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FallbackError(f"published fitted manifest is unreadable: {exc}") from exc
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("animation_authorized") is not False
        or manifest.get("inference_parameters", {}).get("use_skeleton") is not True
    ):
        raise FallbackError("published fitted manifest lost its forced fallback contract")
    return manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(seed=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_fitted_skeleton(seed=args.seed)
    print(f"TOKENRIG_FITTED_SKELETON_PUBLISHED {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
