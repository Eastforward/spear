#!/usr/bin/env python3
"""Authorize research-only animation when fixed-skeleton lineage proves direction.

This is deliberately not a human-approval writer.  It authenticates the
generated target, the unchanged animated carrier, fixed-skeleton conditioning,
SkinTokens attempt, and independent static rig audit.  It may emit an
agent-delegated research decision only when the user has explicitly delegated
autonomous review, both assets use the same reviewed front axis, and every
upstream artifact remains research-only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


SCHEMA = "avengine_fixed_skeleton_motion_basis_agent_decision_v1"
FRONT_AXES = ("positive-x", "negative-x", "positive-y", "negative-y")


def parse_argv(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--target-glb", type=Path, required=True)
    parser.add_argument("--animated-carrier-glb", type=Path, required=True)
    parser.add_argument("--conditioning-manifest", type=Path, required=True)
    parser.add_argument("--skintokens-attempt", type=Path, required=True)
    parser.add_argument("--static-rig-audit", type=Path, required=True)
    parser.add_argument("--target-front-axis", choices=FRONT_AXES, required=True)
    parser.add_argument("--carrier-front-axis", choices=FRONT_AXES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--user-delegated-autonomous-review",
        action="store_true",
        help="Required explicit acknowledgement; never implies human approval.",
    )
    parser.add_argument("--delegation-note", required=True)
    parser.add_argument("--agent-id", default="codex-root")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    path = path.absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"missing or unsafe {label}: {path}")
    return path


def file_record(path: Path):
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_json(path: Path, label: str):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid {label}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a JSON object")
    return value


def same_file(record, path: Path, label: str):
    if (
        not isinstance(record, dict)
        or record.get("sha256") != sha256_file(path)
        or int(record.get("size_bytes", -1)) != path.stat().st_size
    ):
        raise SystemExit(f"{label} identity mismatch")


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def main(argv=None):
    args = parse_argv(argv)
    if not args.user_delegated_autonomous_review:
        raise SystemExit("explicit user delegation is required for agent research review")
    if not args.delegation_note.strip():
        raise SystemExit("--delegation-note must not be empty")
    if args.target_front_axis != args.carrier_front_axis:
        raise SystemExit("target and unchanged animation carrier front axes differ")

    target = require_file(args.target_glb, "target GLB")
    carrier = require_file(args.animated_carrier_glb, "animated carrier GLB")
    conditioning_path = require_file(args.conditioning_manifest, "conditioning manifest")
    attempt_path = require_file(args.skintokens_attempt, "SkinTokens attempt")
    audit_path = require_file(args.static_rig_audit, "static rig audit")
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace decision: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    conditioning = load_json(conditioning_path, "conditioning manifest")
    attempt = load_json(attempt_path, "SkinTokens attempt")
    audit = load_json(audit_path, "static rig audit")
    if (
        conditioning.get("schema")
        != "avengine_fixed_quadruped_skeleton_conditioning_v3"
        or conditioning.get("formal_dataset_registration_authorized") is not False
        or conditioning.get("skeleton_conditioning", {}).get(
            "reparenting_preserved_armature_space_rest_coordinates"
        )
        is not True
    ):
        raise SystemExit("conditioning manifest is not the strict v3 contract")
    same_file(conditioning.get("input"), carrier, "conditioning carrier input")
    if (
        attempt.get("schema") != "avengine_fixed_skeleton_skintokens_attempt_v1"
        or attempt.get("status") != "succeeded"
        or attempt.get("formal_dataset_registration_authorized") is not False
        or attempt.get("skintokens", {}).get("mode")
        != "fixed_skeleton_generate_skin_only"
    ):
        raise SystemExit("SkinTokens attempt is not a successful fixed-skeleton run")
    conditioning_output = Path(str(conditioning.get("output", {}).get("path", "")))
    conditioning_output = require_file(conditioning_output, "conditioning output")
    same_file(conditioning.get("output"), conditioning_output, "conditioning output")
    same_file(attempt.get("input"), conditioning_output, "SkinTokens input")
    same_file(attempt.get("output"), target, "SkinTokens target output")
    if (
        audit.get("schema") != "avengine_generated_animal_rig_audit_v1"
        or audit.get("automatic_checks", {}).get("overall") != "passed"
        or audit.get("animation_authorized") is not False
        or audit.get("formal_dataset_registration_authorized") is not False
        or audit.get("coordinate_contract", {}).get("reviewed_front_axis")
        != args.target_front_axis
    ):
        raise SystemExit("independent static rig audit did not pass the requested axis")
    same_file(audit.get("input"), target, "static rig audit target")

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "asset_id": args.asset_id,
        "status": "agent_research_approved",
        "decision_authority": "agent_delegated_research",
        "human_approved": False,
        "human_approved_by": None,
        "agent_approved_for_research": True,
        "agent_id": args.agent_id,
        "user_delegated_autonomous_review": True,
        "delegation_note": args.delegation_note.strip(),
        "target_animation_generation_authorized": True,
        "formal_dataset_registration_authorized": False,
        "target": {
            **file_record(target),
            "reviewed_front_axis": args.target_front_axis,
        },
        "animated_template_carrier": {
            **file_record(carrier),
            "reviewed_front_axis": args.carrier_front_axis,
            "actions_are_unchanged_template_authority": True,
        },
        "authenticated_lineage": {
            "conditioning_manifest": file_record(conditioning_path),
            "skintokens_attempt": file_record(attempt_path),
            "static_rig_audit": file_record(audit_path),
            "conditioning_input_is_carrier": True,
            "conditioning_output_is_skintokens_input": True,
            "skintokens_output_is_target": True,
            "rest_coordinate_reparenting_only": True,
        },
        "manual_cardinal_motion_basis_yaw_deg": 0,
        "side_chain_mode": "matched",
        "rotation_transfer_mode": "unchanged_template_actions",
        "basis_reason": (
            "target and animated carrier share authenticated fixed-skeleton lineage "
            "and the same reviewed front axis; no cardinal rotation or side swap"
        ),
    }
    payload["decision_sha256"] = hashlib.sha256(
        canonical_json(payload).encode("utf-8")
    ).hexdigest()
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "FIXED_SKELETON_AGENT_MOTION_DECISION_OK "
        f"asset={args.asset_id} yaw=0 side=matched output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
