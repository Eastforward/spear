#!/usr/bin/env python3
"""Freeze one explicit user decision for an authenticated target-native v4 review.

This tool never infers a verdict.  The caller must supply the user's explicit
decision, all six explicit check values, and the SHA-256 of the exact review
covered by that instruction.  It reauthenticates the complete source and review
lineage before publishing a new, immutable decision record.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts
from tools import prepare_user_approved_generated_animal_ue_imports as bridge
from tools import rocketbox_native_material_canary as immutable


RECEIPT_SCHEMA = bridge.DECISION_FREEZE_RECEIPT_SCHEMA
APPROVED = "approved_for_ue_apartment"
REJECTED = "rejected"
DECISIONS = (APPROVED, REJECTED)
CHECK_ARGUMENTS = {
    "walking_direction": "--walking-direction",
    "walking_limb_deformation": "--walking-limb-deformation",
    "walking_ground_contact": "--walking-ground-contact",
    "idle_ground_contact": "--idle-ground-contact",
    "body_stability": "--body-stability",
    "detached_geometry_absent": "--detached-geometry-absent",
}


def _explicit_bool(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise argparse.ArgumentTypeError("expected the literal true or false")


def authenticate_review(
    *,
    source_registry_manifest_path: Path,
    expected_source_registry_sha256: str,
    source_asset_path: Path,
    animation_review_path: Path,
    expected_animation_review_sha256: str,
    artifact_roots: Mapping[str, Path] | None = None,
) -> tuple[Path, str, Path, dict[str, Any], Path, dict[str, Any]]:
    expected_animation_review_sha256 = bridge._require_sha256(
        expected_animation_review_sha256,
        "expected animation review file sha256",
    )
    review_path = bridge._direct_file(
        animation_review_path, "target-native animation review"
    )
    if bridge._sha256_file(review_path) != expected_animation_review_sha256:
        raise contracts.ContractError(
            "animation review does not match the external expected SHA-256"
        )
    roots = {
        name: bridge._uses_only_exact_tmp_bridge(
            Path(path),
            f"artifact root {name}",
        )[0]
        for name, path in (artifact_roots or bridge.DEFAULT_ARTIFACT_ROOTS).items()
    }
    (
        registry_path,
        _registry,
        source_request,
        source_profile,
        registry_mode,
    ) = bridge.load_source_registry_anchor(
        source_registry_manifest_path,
        source_asset_path,
        expected_file_sha256=expected_source_registry_sha256,
    )
    source_path, source_asset, source_artifacts = bridge.load_source_asset(
        source_asset_path,
        roots,
        request=source_request,
        profile=source_profile,
    )
    (
        authenticated_review_path,
        review,
        _animated_glb,
        review_artifacts,
    ) = bridge.load_animation_review(
        review_path,
        source_asset=source_asset,
        source_artifacts=source_artifacts,
    )
    if (
        authenticated_review_path != review_path
        or review.get("schema") != bridge.BRANCHED_GENERATED_REVIEW_SCHEMA
        or review.get("status") != "research_candidate_pending_human_review"
        or review.get("formal_dataset_registration_authorized") is not False
        or review.get("automatic_admission_gates", {}).get("all_automatic_gates_passed")
        is not True
    ):
        raise contracts.ContractError(
            "only a fully authenticated target-native v4 review can be decided"
        )
    return (
        registry_path,
        registry_mode,
        source_path,
        source_asset,
        review_path,
        review_artifacts,
    )


def freeze_decision(
    *,
    source_registry_manifest_path: Path,
    expected_source_registry_sha256: str,
    source_asset_path: Path,
    animation_review_path: Path,
    expected_animation_review_sha256: str,
    decision: str,
    checks: Mapping[str, bool],
    caveats: Sequence[str],
    notes: str,
    user_explicit_decision: str,
    user_explicit_review_sha256: str,
    output_root: Path,
    artifact_roots: Mapping[str, Path] | None = None,
) -> Path:
    if decision not in DECISIONS or user_explicit_decision != decision:
        raise contracts.ContractError(
            "the frozen verdict must match the user's explicit decision"
        )
    bridge._require_sha256(
        user_explicit_review_sha256,
        "user-explicit animation review sha256",
    )
    if user_explicit_review_sha256 != expected_animation_review_sha256:
        raise contracts.ContractError(
            "the user's explicit instruction is not bound to the expected review"
        )
    if (
        not isinstance(checks, Mapping)
        or set(checks) != bridge.DECISION_CHECK_FIELDS
        or any(not isinstance(value, bool) for value in checks.values())
    ):
        raise contracts.ContractError(
            "all six animation decision checks must be explicit booleans"
        )
    if decision == APPROVED and not all(checks.values()):
        raise contracts.ContractError("approval requires all six checks to pass")
    if (
        not isinstance(notes, str)
        or not notes.strip()
        or not isinstance(caveats, Sequence)
        or isinstance(caveats, (str, bytes))
        or any(not isinstance(value, str) or not value for value in caveats)
        or len(caveats) != len(set(caveats))
    ):
        raise contracts.ContractError("decision notes/caveats are invalid")

    (
        registry_path,
        registry_mode,
        source_path,
        source_asset,
        review_path,
        review_artifacts,
    ) = authenticate_review(
        source_registry_manifest_path=source_registry_manifest_path,
        expected_source_registry_sha256=expected_source_registry_sha256,
        source_asset_path=source_asset_path,
        animation_review_path=animation_review_path,
        expected_animation_review_sha256=expected_animation_review_sha256,
        artifact_roots=artifact_roots,
    )
    output_root = bridge._new_output_path(Path(output_root), "output")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    bridge._new_output_path(output_root, "output")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.",
            suffix=".staging",
            dir=output_root.parent,
        )
    )
    try:
        state = "research_candidate" if decision == APPROVED else "rejected"
        next_gate = (
            "ue_import_metric_trajectory_audio_and_apartment_media"
            if decision == APPROVED
            else "stop"
        )
        record: dict[str, Any] = {
            "schema": bridge.DECISION_SCHEMA,
            "asset_id": source_asset["asset_id"],
            "review_sha256": expected_animation_review_sha256,
            "decision": decision,
            "checks": {
                name: checks[name] for name in sorted(bridge.DECISION_CHECK_FIELDS)
            },
            "caveats": list(caveats),
            "notes": notes,
            "review": bridge._absolute_record(review_path),
            "state_classification": state,
            "formal_dataset_registration_authorized": False,
            "next_gate": next_gate,
        }
        record["decision_sha256"] = bridge._hash_without(record, "decision_sha256")
        decision_path = contracts.write_json_no_replace(
            staging / "animation_decision.json", record
        )
        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "status": "frozen",
            "state_classification": state,
            "formal_dataset_registration_authorized": False,
            "source_asset_registry": bridge._absolute_record(registry_path),
            "expected_source_asset_registry_file_sha256": (
                expected_source_registry_sha256
            ),
            "source_asset_registry_validation_mode": registry_mode,
            "source_asset": bridge._absolute_record(source_path),
            "animation_review": bridge._absolute_record(review_path),
            "expected_animation_review_file_sha256": (expected_animation_review_sha256),
            "user_instruction_binding": {
                "decision": decision,
                "review_sha256": user_explicit_review_sha256,
                "all_six_checks_explicit": True,
            },
            "user_instruction_authority": dict(bridge.USER_INSTRUCTION_AUTHORITY),
            "authenticated_review_artifact_count": len(review_artifacts),
            "animation_decision": bridge._relative_record(decision_path, staging),
            "decision_sha256": record["decision_sha256"],
        }
        receipt["receipt_sha256"] = bridge._hash_without(receipt, "receipt_sha256")
        contracts.write_json_no_replace(
            staging / "decision_freeze_receipt.json", receipt
        )
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError(
                "animation decision output appeared concurrently"
            )
        os.rename(staging, output_root)
        return output_root / decision_path.name
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry-manifest", required=True, type=Path)
    parser.add_argument("--expected-source-registry-sha256", required=True)
    parser.add_argument("--source-asset", required=True, type=Path)
    parser.add_argument("--animation-review", required=True, type=Path)
    parser.add_argument("--expected-animation-review-sha256", required=True)
    parser.add_argument("--decision", choices=DECISIONS, required=True)
    parser.add_argument(
        "--user-explicit-decision",
        choices=DECISIONS,
        required=True,
        help=(
            "Pass only after the user explicitly issued this exact verdict; "
            "it must equal --decision."
        ),
    )
    parser.add_argument(
        "--user-explicit-review-sha256",
        required=True,
        help=(
            "SHA-256 of the exact review covered by the user's instruction; "
            "it must equal --expected-animation-review-sha256."
        ),
    )
    for name, flag in CHECK_ARGUMENTS.items():
        parser.add_argument(
            flag,
            dest=name,
            type=_explicit_bool,
            choices=(True, False),
            required=True,
        )
    parser.add_argument("--notes", required=True)
    parser.add_argument("--caveat", action="append", default=[])
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--artifact-root",
        action="append",
        default=[],
        metavar="ROOT_ID=PATH",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        decision_path = freeze_decision(
            source_registry_manifest_path=args.source_registry_manifest,
            expected_source_registry_sha256=args.expected_source_registry_sha256,
            source_asset_path=args.source_asset,
            animation_review_path=args.animation_review,
            expected_animation_review_sha256=(args.expected_animation_review_sha256),
            decision=args.decision,
            checks={name: getattr(args, name) for name in CHECK_ARGUMENTS},
            caveats=args.caveat,
            notes=args.notes,
            user_explicit_decision=args.user_explicit_decision,
            user_explicit_review_sha256=args.user_explicit_review_sha256,
            output_root=args.output_root,
            artifact_roots=bridge.parse_artifact_roots(args.artifact_root),
        )
        payload = bridge._load_finite_json(
            decision_path,
            "frozen animation decision output",
        )
    except (contracts.ContractError, OSError) as error:
        print(
            f"TARGET_NATIVE_ANIMATION_DECISION_FREEZE_FAILED {error}",
            file=sys.stderr,
        )
        return 2
    print(
        "TARGET_NATIVE_ANIMATION_DECISION_FREEZE_OK "
        f"decision={decision_path} "
        f"decision_file_sha256={bridge._sha256_file(decision_path)} "
        f"decision_sha256={payload['decision_sha256']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
