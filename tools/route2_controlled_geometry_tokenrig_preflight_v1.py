#!/usr/bin/env python3
"""Run an agent-passed controlled-geometry Pixal GLB through TokenRig transfer."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import stat
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import tokenrig_human_canary as base
from tools import route2_controlled_geometry_pixal_static_decision_v1 as static_qa


SCHEMA = "route2_controlled_geometry_tokenrig_lineage_v1"
TOKENRIG_SCHEMA = "route2_controlled_geometry_tokenrig_candidate_v1"
RUNNER_PATH = Path(__file__).resolve()
SPEAR_ROOT = RUNNER_PATH.parents[1]
PIXAL_ROOT = SPEAR_ROOT / "tmp/i23d_controlled_geometry_v3/pixal3d"
OUTPUT_ROOT = SPEAR_ROOT / "tmp/route2_controlled_geometry_tokenrig_v1"
EXECUTION_CONTRACT = PIXAL_ROOT / "execution_contract_v1.json"
STATIC_DECISION_SCHEMA = "route2_controlled_geometry_pixal_static_agent_qa_v1"
SKINTOKENS_REVISION = base.PINNED_CONTRACT.model_revision
BPY_PORT = 59876
ALLOWED_GPUS = ("0", "1", "2", "3")
ALLOWED_PORTS = (59876, 59877, 59878, 59879)
PORT_OVERRIDE_SOURCE = '''\
import os as _route2_os
from src.server import spec as _route2_server_spec
_route2_port_text = _route2_os.environ.get("TOKENRIG_BPY_PORT", "")
if not _route2_port_text.isascii() or not _route2_port_text.isdecimal():
    raise RuntimeError("TOKENRIG_BPY_PORT must be an ASCII decimal integer")
_route2_port = int(_route2_port_text)
if _route2_port not in (59876, 59877, 59878, 59879):
    raise RuntimeError("TOKENRIG_BPY_PORT is outside the authenticated Route-2 port set")
_route2_server_spec.BPY_PORT = _route2_port
_route2_server_spec.BPY_SERVER = f"http://localhost:{_route2_port}"
'''


class ControlledTokenRigError(RuntimeError):
    pass


def _record(path: Path, *, require_mode: int | None = None) -> dict[str, Any]:
    path = Path(path).absolute()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
        or not stat.S_ISREG(os.lstat(path).st_mode)
        or path.stat().st_size <= 0
    ):
        raise ControlledTokenRigError(f"artifact must be a direct nonempty file: {path}")
    if require_mode is not None and stat.S_IMODE(path.stat().st_mode) != require_mode:
        raise ControlledTokenRigError(f"artifact mode changed: {path}")
    return {
        "path": str(path),
        "sha256": base.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ControlledTokenRigError(f"{description} is invalid: {error}") from error
    if not isinstance(payload, dict):
        raise ControlledTokenRigError(f"{description} must contain an object")
    return payload


def allowed_assets() -> tuple[str, ...]:
    contract = _json(EXECUTION_CONTRACT, "controlled Pixal execution contract")
    assets = tuple(str(item.get("asset_id")) for item in contract.get("jobs", ()))
    if (
        contract.get("schema") != "route2_controlled_geometry_pixal_execution_contract_v1"
        or len(assets) != 8
        or len(set(assets)) != 8
        or any(not asset.startswith("route2_v3_") for asset in assets)
    ):
        raise ControlledTokenRigError("controlled Pixal execution contract changed")
    return assets


def authenticate_static_gate(asset_id: str) -> dict[str, Any]:
    if asset_id not in allowed_assets():
        raise ControlledTokenRigError(f"asset is not in the controlled Pixal contract: {asset_id}")
    review_root, review, review_record = static_qa.authenticate_review(asset_id)
    decision_path = review_root / "agent_static_visual_qa.json"
    decision = _json(decision_path, "static agent decision")
    if (
        decision.get("schema") != STATIC_DECISION_SCHEMA
        or decision.get("asset_id") != asset_id
        or decision.get("status") != "agent_static_visual_passed"
        or decision.get("state_classification") != "research_candidate"
        or decision.get("review_manifest") != review_record
        or decision.get("contact_sheet") != review["artifacts"]["contact_sheet.png"]
        or decision.get("tokenrig_preflight_authorized") is not True
        or decision.get("formal_dataset_registration_authorized") is not False
        or decision.get("user_acceptance") != "not_claimed"
        or not all(value is True for value in decision.get("checks", {}).values())
    ):
        raise ControlledTokenRigError("static agent decision lineage or gate changed")
    return {
        "review_manifest": review_record,
        "decision": _record(decision_path, require_mode=0o444),
        "contact_sheet": review["artifacts"]["contact_sheet.png"],
    }


def read_controlled_input(
    input_glb: Path, input_manifest: Path, contract: base.CanaryContract
) -> dict[str, Any]:
    input_glb = Path(input_glb).absolute()
    input_manifest = Path(input_manifest).absolute()
    if input_glb != Path(contract.input_glb).absolute() or input_manifest != Path(
        contract.input_manifest
    ).absolute():
        raise base.CanaryError("controlled Pixal input paths are not the exact contract paths")
    if input_glb.suffix.lower() != ".glb" or input_glb.name != "canary_1024_seed42.glb":
        raise base.CanaryError("TokenRig requires the original packed-PBR controlled Pixal GLB")
    glb_record = _record(input_glb, require_mode=0o444)
    manifest_record = _record(input_manifest, require_mode=0o444)
    if (
        glb_record["sha256"] != contract.input_glb_sha256
        or manifest_record["sha256"] != contract.input_manifest_sha256
    ):
        raise base.CanaryError("controlled Pixal input hashes changed")
    manifest = _json(input_manifest, "controlled Pixal manifest")
    output = manifest.get("output")
    if (
        manifest.get("schema") != "route2_controlled_geometry_pixal_candidate_v1"
        or manifest.get("asset_id") != contract.asset_id
        or manifest.get("backend") != "pixal3d"
        or manifest.get("state_classification") != "research_candidate"
        or manifest.get("formal_registration_authorized") is not False
        or manifest.get("model", {}).get("revision") != base.PIXAL3D_REVISION
        or manifest.get("parameters")
        != {"seed": 42, "manual_fov": 0.2, "resolution": 1024, "low_vram": True}
        or output != glb_record
        or manifest.get("pbr_glb_readback", {}).get("passed") is not True
        or manifest.get("pbr_glb_readback", {}).get("packed_pbr") is not True
        or int(manifest.get("pbr_glb_readback", {}).get("image_count", 0)) < 2
    ):
        raise base.CanaryError("controlled Pixal manifest identity/PBR contract changed")
    return {
        "glb": {**glb_record, "readback": base.inspect_glb(input_glb, require_skin=False)},
        "manifest": manifest_record,
        "pbr_glb_readback": manifest["pbr_glb_readback"],
    }


def _contract(asset_id: str) -> base.CanaryContract:
    asset_root = PIXAL_ROOT / asset_id
    glb = asset_root / "canary_1024_seed42.glb"
    manifest = asset_root / "canary_1024_seed42.manifest.json"
    return replace(
        base.PINNED_CONTRACT,
        asset_id=asset_id,
        input_glb=glb,
        input_manifest=manifest,
        output_dir=OUTPUT_ROOT / asset_id / "direct_transfer_seed42",
        input_glb_sha256=base.sha256_file(glb),
        input_manifest_sha256=base.sha256_file(manifest),
    )


def _publish_lineage(
    *, asset_id: str, gate: Mapping[str, Any], tokenrig_manifest: Path, gpu: str, port: int
) -> Path:
    output_dir = Path(tokenrig_manifest).absolute().parent
    tokenrig = _json(tokenrig_manifest, "controlled TokenRig manifest")
    if (
        tokenrig.get("schema") != TOKENRIG_SCHEMA
        or tokenrig.get("asset_id") != asset_id
        or tokenrig.get("attempt") != "direct_transfer"
        or tokenrig.get("state_classification") != "research_candidate"
        or tokenrig.get("inference_parameters", {}).get("use_transfer") is not True
        or tokenrig.get("inference_parameters", {}).get("use_skeleton") is not False
        or tokenrig.get("output", {}).get("readback", {}).get("skin_count", 0) < 1
    ):
        raise ControlledTokenRigError("TokenRig output manifest is not a direct transfer candidate")
    destination = output_dir / "controlled_geometry_lineage_v1.json"
    payload = {
        "schema": SCHEMA,
        "asset_id": asset_id,
        "state_classification": "research_candidate",
        "static_pixal_gate": dict(gate),
        "tokenrig_manifest": _record(tokenrig_manifest),
        "tokenrig_output": tokenrig["output"],
        "runner": _record(RUNNER_PATH),
        "binding_mode": {"use_transfer": True, "use_skeleton": False},
        "runtime_assignment": {
            "physical_gpu": gpu,
            "bpy_port": port,
            "bpy_server": f"http://localhost:{port}",
            "client_and_server_override": "authenticated_sitecustomize",
        },
        "static_binding_audit": "pending",
        "animation_authorized": False,
        "formal_dataset_registration_authorized": False,
        "user_acceptance": "not_claimed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o444,
    )
    try:
        os.write(
            descriptor,
            (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    for path in output_dir.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(0o444)
    attempt = output_dir.parent / f"{output_dir.name}.tokenrig_attempt.json"
    if attempt.is_file() and not attempt.is_symlink():
        attempt.chmod(0o444)
    return destination


def run(asset_id: str, gpu: str, port: int) -> Path:
    if gpu not in ALLOWED_GPUS:
        raise ControlledTokenRigError(f"GPU must be one of {ALLOWED_GPUS}")
    if port not in ALLOWED_PORTS:
        raise ControlledTokenRigError(f"bpy port must be one of {ALLOWED_PORTS}")
    gate = authenticate_static_gate(asset_id)
    contract = _contract(asset_id)
    # SkinTokens uses one fixed embedded bpy HTTP port.  A host lock permits
    # aggressive parallel upstream work without corrupting concurrent binds.
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    lock_path = OUTPUT_ROOT / f".bpy_port_{port}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        base.PHYSICAL_GPU = gpu
        base.SCHEMA = TOKENRIG_SCHEMA
        original_reader = base._read_input_contract
        original_hygiene = base.SERVER_HYGIENE_SOURCE
        base._read_input_contract = read_controlled_input
        base.SERVER_HYGIENE_SOURCE = PORT_OVERRIDE_SOURCE + "\n" + original_hygiene
        try:
            tokenrig_manifest = base.run_canary(
                input_glb=contract.input_glb,
                input_manifest=contract.input_manifest,
                output_dir=contract.output_dir,
                skintokens_root=contract.skintokens_root,
                model_revision=contract.model_revision,
                seed=42,
                use_skeleton_input=False,
                contract=contract,
                orchestrator_path=RUNNER_PATH,
                base_env={**os.environ, "TOKENRIG_BPY_PORT": str(port)},
                bpy_port_probe=lambda: base.assert_bpy_port_available(port=port),
            )
        finally:
            base._read_input_contract = original_reader
            base.SERVER_HYGIENE_SOURCE = original_hygiene
        if authenticate_static_gate(asset_id) != gate:
            raise ControlledTokenRigError("static Pixal gate changed during TokenRig inference")
        return _publish_lineage(
            asset_id=asset_id,
            gate=gate,
            tokenrig_manifest=tokenrig_manifest,
            gpu=gpu,
            port=port,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--gpu", choices=ALLOWED_GPUS, required=True)
    parser.add_argument("--bpy-port", type=int, choices=ALLOWED_PORTS, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        "CONTROLLED_GEOMETRY_TOKENRIG_OK "
        f"{run(args.asset_id, args.gpu, args.bpy_port)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
