#!/usr/bin/env python3
"""Audit a sealed controlled-animal FLUX batch for no-seed-lottery evidence."""

from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts
from tools import prepare_controlled_animal_pixal_inputs as pixal_inputs
from tools import review_controlled_animal_flux2_candidates as review
from tools import run_controlled_animal_pixal_jobs as pixal_runner


SCHEMA = "avengine_controlled_animal_one_shot_batch_audit_v2"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_without(value: dict[str, Any], field: str) -> str:
    return contracts.manifest_sha256(
        {key: copy.deepcopy(item) for key, item in value.items() if key != field}
    )


def _checked_relative_artifact(root: Path, record: Any, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise contracts.ContractError(f"{label} record is invalid")
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise contracts.ContractError(f"{label} path is unsafe")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise contracts.ContractError(f"{label} escaped its batch root") from error
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record["size_bytes"]
        or _sha256_file(path) != record["sha256"]
    ):
        raise contracts.ContractError(f"{label} artifact changed")
    return path


def _audit_pixal_batch(
    pixal_batch_path: Path, flux_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    path = Path(pixal_batch_path).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"Pixal batch is missing: {path}")
    batch = contracts.load_json(path)
    attempts = batch.get("attempts")
    expected = {row["instance_id"]: row for row in flux_rows}
    if (
        batch.get("schema") != "avengine_controlled_animal_pixal_batch_v1"
        or batch.get("status") != "passed_generation_and_glb_readback"
        or batch.get("batch_sha256") != _hash_without(batch, "batch_sha256")
        or batch.get("job_count") != len(expected)
        or batch.get("passed_count") != len(expected)
        or batch.get("failed_count") != 0
        or not isinstance(attempts, list)
        or len(attempts) != len(expected)
        or len({item.get("instance_id") for item in attempts}) != len(attempts)
    ):
        raise contracts.ContractError("Pixal batch one-shot coverage/hash is invalid")
    root = path.parent
    rows = []
    for attempt in attempts:
        instance_id = attempt["instance_id"]
        flux = expected.get(instance_id)
        if (
            flux is None
            or attempt.get("execution_job_id") != flux["execution_job_id"]
            or attempt.get("request_sha256") != flux["request_sha256"]
            or attempt.get("seed") != flux["generation_seed"]
            or attempt.get("status") != "passed_generation_and_glb_readback"
        ):
            raise contracts.ContractError("Pixal attempt differs from frozen FLUX request")
        output = _checked_relative_artifact(root, attempt.get("output"), "Pixal output")
        model_manifest_path = _checked_relative_artifact(
            root, attempt.get("attempt_manifest"), "Pixal attempt manifest"
        )
        model_manifest = contracts.load_json(model_manifest_path)
        if (
            model_manifest.get("backend") != "pixal3d"
            or model_manifest.get("parameters", {}).get("seed")
            != flux["generation_seed"]
            or model_manifest.get("output", {}).get("sha256") != _sha256_file(output)
        ):
            raise contracts.ContractError("Pixal model manifest seed/output changed")
        rows.append(
            {
                "instance_id": instance_id,
                "request_sha256": flux["request_sha256"],
                "generation_seed": flux["generation_seed"],
                "recorded_pixal_invocations": 1,
                "output_sha256": _sha256_file(output),
                "attempt_manifest_sha256": _sha256_file(model_manifest_path),
            }
        )
    scheduling = batch.get("scheduling", {})
    if (
        scheduling.get("mode") != "shared_claim_queue_v1"
        or scheduling.get("claim_count") != len(expected)
        or len(scheduling.get("claims", [])) != len(expected)
        or {item.get("instance_id") for item in scheduling.get("claims", [])}
        != set(expected)
    ):
        raise contracts.ContractError("Pixal dynamic claim evidence is not one-per-request")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "batch_sha256": batch["batch_sha256"],
        "recorded_attempts_per_request": 1,
        "dynamic_claims_per_request": 1,
        "attempts": sorted(rows, key=lambda item: item["instance_id"]),
    }


def _authenticate_pixal_inputs(
    pixal_inputs_path: Path,
    *,
    flux_batch: Mapping[str, Any],
    candidates: Mapping[str, Mapping[str, Any]],
    bounded_exploration: bool,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path, payload = pixal_runner.load_pixal_inputs(pixal_inputs_path)
    freeze = payload.get("bounded_exploration_freeze")
    upstream_evidence = payload.get("upstream_flux_one_shot_evidence")
    if bounded_exploration:
        if (
            not isinstance(freeze, Mapping)
            or not isinstance(upstream_evidence, Mapping)
            or upstream_evidence.get("mode")
            != "native_bounded_exploration_frozen"
        ):
            raise contracts.ContractError(
                "bounded FLUX audit requires Pixal inputs with a "
                "reauthenticated exploration freeze"
            )
        expected_evidence = pixal_inputs._flux_one_shot_evidence(
            flux_batch,
            candidates,
            bounded_exploration_freeze=freeze,
        )
    else:
        if freeze is not None:
            raise contracts.ContractError(
                "ordinary one-shot Pixal inputs contain an unexpected "
                "bounded exploration freeze"
            )
        expected_evidence = pixal_inputs._flux_one_shot_evidence(
            flux_batch, candidates
        )
    if contracts.canonical_json(upstream_evidence) != contracts.canonical_json(
        expected_evidence
    ):
        raise contracts.ContractError(
            "Pixal input evidence differs from the audited FLUX batch"
        )
    return path, payload, expected_evidence


def _pixal_selected_flux_rows(
    payload: Mapping[str, Any],
    flux_rows: Sequence[dict[str, Any]],
    *,
    bounded_exploration: bool,
) -> list[dict[str, Any]]:
    by_instance = {row["instance_id"]: row for row in flux_rows}
    selected = []
    for job in payload["jobs"]:
        controlled = job["controlled_request"]
        instance_id = controlled["instance_id"]
        row = by_instance.get(instance_id)
        if row is None:
            raise contracts.ContractError(
                "Pixal inputs select a candidate outside the audited FLUX batch"
            )
        expected = {
            "execution_job_id": row["execution_job_id"],
            "request_sha256": row["request_sha256"],
            "profile_schema_id": row["profile_schema_id"],
            "generation_seed": row["generation_seed"],
        }
        if (
            any(controlled.get(key) != value for key, value in expected.items())
            or job.get("reference", {}).get("source", {}).get("sha256")
            != row["candidate_sha256"]
        ):
            raise contracts.ContractError(
                "Pixal input job differs from its audited FLUX candidate"
            )
        selected.append(row)
    if not bounded_exploration and {
        row["instance_id"] for row in selected
    } != set(by_instance):
        raise contracts.ContractError(
            "ordinary one-shot Pixal inputs do not cover every FLUX request"
        )
    return selected


def build_audit(
    flux_batch_path: Path,
    pixal_batch_path: Path | None = None,
    pixal_inputs_path: Path | None = None,
) -> dict[str, Any]:
    path = Path(flux_batch_path).resolve()
    _root, batch, candidates = review.load_flux_batch(path)
    bounded_exploration = (
        batch.get("selection", {}).get("bounded_exploration") is not None
    )
    if bounded_exploration and pixal_inputs_path is None:
        raise contracts.ContractError(
            "bounded FLUX audit requires --pixal-inputs so its frozen "
            "selection is reauthenticated"
        )
    authenticated_pixal_inputs = None
    if pixal_inputs_path is None:
        evidence = pixal_inputs._flux_one_shot_evidence(batch, candidates)
        pixal_payload = None
    else:
        (
            authenticated_pixal_inputs,
            pixal_payload,
            evidence,
        ) = _authenticate_pixal_inputs(
            pixal_inputs_path,
            flux_batch=batch,
            candidates=candidates,
            bounded_exploration=bounded_exploration,
        )
    rows = []
    for instance_id, candidate in sorted(candidates.items()):
        manifest = candidate["manifest"]
        generation = manifest["generation"]
        rows.append(
            {
                "instance_id": instance_id,
                "execution_job_id": manifest["execution_job_id"],
                "request_sha256": manifest["request_sha256"],
                "profile_schema_id": manifest["profile_schema_id"],
                "generation_seed": generation["seed"],
                "recorded_flux_invocations": generation["flux_invocations"],
                "candidate_sha256": manifest["output"]["sha256"],
                "candidate_manifest_sha256": candidate["index"][
                    "candidate_manifest"
                ]["sha256"],
            }
        )
    pixal_coverage_rows = rows
    if pixal_payload is not None:
        pixal_coverage_rows = _pixal_selected_flux_rows(
            pixal_payload,
            rows,
            bounded_exploration=bounded_exploration,
        )
    audit: dict[str, Any] = {
        "schema": SCHEMA,
        "status": (
            "passed_native_policy"
            if evidence["profile_qualification_authorized"]
            else "passed_legacy_batch_only_profile_qualification_blocked"
        ),
        "flux_batch": {
            "path": str(path),
            "sha256": _sha256_file(path),
            "batch_sha256": batch["batch_sha256"],
        },
        "one_shot_evidence": evidence,
        "candidate_count": len(rows),
        "candidates": rows,
        "automatic_checks": {
            "batch_and_candidate_hashes_reauthenticated": True,
            "one_recorded_candidate_per_request": True,
            "one_recorded_flux_invocation_per_candidate": True,
            "profile_qualification_authorized": evidence[
                "profile_qualification_authorized"
            ],
            "overall": "passed_with_legacy_limitation"
            if not evidence["profile_qualification_authorized"]
            else "passed",
        },
        "formal_dataset_registration_authorized": False,
    }
    if authenticated_pixal_inputs is not None:
        audit["pixal_inputs"] = {
            "path": str(authenticated_pixal_inputs),
            "sha256": _sha256_file(authenticated_pixal_inputs),
            "manifest_sha256": pixal_payload["manifest_sha256"],
        }
        if bounded_exploration:
            audit["automatic_checks"][
                "bounded_exploration_freeze_reauthenticated_by_pixal_input_loader"
            ] = True
    if pixal_batch_path is not None:
        audit["pixal_batch"] = _audit_pixal_batch(
            pixal_batch_path, pixal_coverage_rows
        )
        audit["automatic_checks"]["one_recorded_pixal_invocation_per_request"] = True
    audit["audit_sha256"] = contracts.manifest_sha256(audit)
    return audit


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flux-batch", required=True, type=Path)
    parser.add_argument("--pixal-inputs", type=Path)
    parser.add_argument("--pixal-batch", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        audit = build_audit(
            args.flux_batch,
            pixal_batch_path=args.pixal_batch,
            pixal_inputs_path=args.pixal_inputs,
        )
        contracts.write_json_no_replace(args.output.resolve(), audit)
    except (contracts.ContractError, OSError, ValueError) as error:
        print(f"CONTROLLED_ANIMAL_ONE_SHOT_AUDIT_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_ANIMAL_ONE_SHOT_AUDIT_OK "
        f"status={audit['status']} output={args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
