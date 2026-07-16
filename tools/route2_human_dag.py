#!/usr/bin/env python3
"""Immutable serial/resume ledger for the nine Route-2 human instances."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from tools import route2_human_contract_common as common


RUN_SCHEMA = "route2_human_serial_run_v1"
EVENT_SCHEMA = "route2_human_serial_event_v1"
RUN_MANIFEST_NAME = "route2_run_manifest_v1.json"
EVENTS_DIRNAME = "events"

CASE_ORDER = (
    "rocketbox_male_adult_01",
    "rocketbox_female_adult_01",
    "tall_man",
    "short_woman",
    "glasses",
    "hat",
    "short_sleeve_color",
    "trousers",
    "shoes",
)
ATTRIBUTE_CASES = CASE_ORDER[2:]
BASE_FOR_CASE = {
    "rocketbox_male_adult_01": "rocketbox_male_adult_01",
    "rocketbox_female_adult_01": "rocketbox_female_adult_01",
    "tall_man": "rocketbox_male_adult_01",
    "short_woman": "rocketbox_female_adult_01",
    "glasses": "rocketbox_male_adult_01",
    "hat": "rocketbox_female_adult_01",
    "short_sleeve_color": "rocketbox_male_adult_01",
    "trousers": "rocketbox_female_adult_01",
    "shoes": "rocketbox_male_adult_01",
}
BASE_STAGES = (
    "pixal_source",
    "tokenrig_static",
    "retarget_walk_idle",
    "dynamic_media",
    "agent_visual_qa",
    "qualified_candidate",
)
ATTRIBUTE_STAGES = (
    "flux2_edit",
    "agent_2d_qa",
    "pixal3d",
    "tokenrig_static",
    "retarget_walk_idle",
    "dynamic_media",
    "agent_visual_qa",
    "qualified_candidate",
)
QUALIFIED_STATUS = "agent_qa_passed_pending_user_acceptance"
_EVENT_NAME = re.compile(r"([0-9]{6})\.([0-9a-f]{64})\.json")


class DagError(RuntimeError):
    """Raised when the serial Route-2 run cannot be authenticated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _embedded_file_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in ("path", "sha256", "size_bytes", "mode")
    }


def _stages(case_id: str) -> tuple[str, ...]:
    if case_id in CASE_ORDER[:2]:
        return BASE_STAGES
    if case_id in ATTRIBUTE_CASES:
        return ATTRIBUTE_STAGES
    raise DagError(f"unknown Route-2 case_id: {case_id}")


def _manifest_payload() -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "created_at_utc": _utc_now(),
        "case_order": list(CASE_ORDER),
        "base_for_case": dict(BASE_FOR_CASE),
        "stage_profiles": {
            "base_avatar": list(BASE_STAGES),
            "attribute_instance": list(ATTRIBUTE_STAGES),
        },
        "execution": {
            "serial": True,
            "resume_without_repeating_succeeded_stage": True,
            "rejected_instance_advances_to_next_case": True,
        },
        "state_classification": "research_candidate",
        "publication": {"event_files": "immutable_no_replace", "mode": "0444"},
        "user_acceptance": "pending_user_review",
    }


def create_run(run_root: Path) -> Path:
    """Create the immutable run header and empty append-only event directory."""

    root = common.absolute(run_root)
    if os.path.lexists(root):
        raise FileExistsError(root)
    parent = root.parent
    common.require_real_directory(parent, "Route-2 run parent", DagError)
    os.mkdir(root, 0o755)
    events = root / EVENTS_DIRNAME
    os.mkdir(events, 0o755)
    common.fsync_directory(events)
    common.fsync_directory(root)
    common.fsync_directory(parent)
    payload = _manifest_payload()
    common.reject_user_approval(payload, DagError, "Route-2 run manifest")
    destination = root / RUN_MANIFEST_NAME
    common.write_json_immutable_noreplace(
        destination,
        payload,
        DagError,
        "Route-2 run manifest",
    )
    common.fsync_directory(root)
    return destination


def _load_manifest(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, record = common.load_json_mapping_record(
        root / RUN_MANIFEST_NAME,
        root=root,
        description="Route-2 run manifest",
        error_type=DagError,
        require_mode=0o444,
    )
    common.reject_user_approval(payload, DagError, "Route-2 run manifest")
    expected_fields = {
        "schema",
        "created_at_utc",
        "case_order",
        "base_for_case",
        "stage_profiles",
        "execution",
        "state_classification",
        "publication",
        "user_acceptance",
    }
    if set(payload) != expected_fields:
        raise DagError("Route-2 run manifest fields changed")
    if (
        payload.get("schema") != RUN_SCHEMA
        or payload.get("case_order") != list(CASE_ORDER)
        or payload.get("base_for_case") != BASE_FOR_CASE
        or payload.get("stage_profiles")
        != {
            "base_avatar": list(BASE_STAGES),
            "attribute_instance": list(ATTRIBUTE_STAGES),
        }
        or payload.get("execution")
        != {
            "serial": True,
            "resume_without_repeating_succeeded_stage": True,
            "rejected_instance_advances_to_next_case": True,
        }
        or payload.get("state_classification") != "research_candidate"
        or payload.get("publication")
        != {"event_files": "immutable_no_replace", "mode": "0444"}
        or payload.get("user_acceptance") != "pending_user_review"
        or not isinstance(payload.get("created_at_utc"), str)
        or not payload["created_at_utc"].endswith("Z")
    ):
        raise DagError("Route-2 run manifest policy or order changed")
    return payload, _embedded_file_record(record)


def _initial_state(
    *, root: Path, manifest: Mapping[str, Any], manifest_record: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema": RUN_SCHEMA,
        "run_root": str(root),
        "run_manifest": dict(manifest_record),
        "manifest": dict(manifest),
        "events": [],
        "next_sequence": 1,
        "head_event_sha256": None,
        "cases": {
            case_id: {
                "base_avatar_id": BASE_FOR_CASE[case_id],
                "succeeded_stages": [],
                "terminal_status": None,
                "terminal_stage": None,
                "reason_code": None,
            }
            for case_id in CASE_ORDER
        },
        "complete": False,
        "blocked_reason": None,
    }


def _qualified(state: Mapping[str, Any], case_id: str) -> bool:
    return state["cases"][case_id]["terminal_status"] == QUALIFIED_STATUS


def next_action(state: Mapping[str, Any]) -> dict[str, str] | None:
    """Return exactly one serial stage, or None for complete/dependency-blocked."""

    cases = state.get("cases")
    if not isinstance(cases, Mapping) or set(cases) != set(CASE_ORDER):
        raise DagError("Route-2 scheduler state has an invalid case set")
    for case_id in CASE_ORDER:
        case = cases[case_id]
        if case.get("terminal_status") is not None:
            continue
        if case_id == "rocketbox_female_adult_01" and not _qualified(
            state, "rocketbox_male_adult_01"
        ):
            return None
        if case_id in ATTRIBUTE_CASES and not (
            _qualified(state, "rocketbox_male_adult_01")
            and _qualified(state, "rocketbox_female_adult_01")
        ):
            return None
        succeeded = case.get("succeeded_stages")
        if not isinstance(succeeded, list):
            raise DagError("Route-2 scheduler case stages are invalid")
        for stage in _stages(case_id):
            if stage not in succeeded:
                return {
                    "case_id": case_id,
                    "base_avatar_id": BASE_FOR_CASE[case_id],
                    "stage": stage,
                }
        raise DagError(f"case exhausted stages without a terminal result: {case_id}")
    return None


def _validate_evidence_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
        "mode",
    }:
        raise DagError("event evidence descriptor is incomplete")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise DagError("event evidence path must be absolute")
    path = common.absolute(Path(raw_path))
    record = common.hash_file_snapshot(
        path,
        root=path.parent,
        description="Route-2 event evidence",
        error_type=DagError,
        require_mode=0o444,
    )
    expected = _embedded_file_record(record)
    if dict(value) != expected:
        raise DagError("Route-2 event evidence changed or hash differs")
    return expected


def _apply_event(state: dict[str, Any], event: Mapping[str, Any]) -> None:
    expected = next_action(state)
    if expected is None:
        raise DagError("no serial Route-2 stage is currently authorized")
    if any(event.get(key) != expected[key] for key in expected):
        raise DagError(
            "Route-2 event does not match the expected serial case/stage"
        )
    status = event.get("status")
    stage = expected["stage"]
    reason_code = event.get("reason_code")
    if status == "rejected":
        if (
            not isinstance(reason_code, str)
            or common.CANONICAL_ID_RE.fullmatch(reason_code) is None
        ):
            raise DagError("rejected Route-2 event requires a canonical reason_code")
    elif reason_code is not None:
        raise DagError("non-rejected Route-2 event cannot carry a reason_code")
    if stage == "qualified_candidate":
        if status not in {QUALIFIED_STATUS, "rejected"}:
            raise DagError(
                "qualified_candidate requires "
                "agent_qa_passed_pending_user_acceptance or rejected"
            )
    elif status not in {"succeeded", "rejected"}:
        raise DagError("non-terminal Route-2 stage must succeed or reject")

    case = state["cases"][expected["case_id"]]
    if status == "succeeded":
        case["succeeded_stages"].append(stage)
    else:
        if status == QUALIFIED_STATUS:
            case["succeeded_stages"].append(stage)
        case["terminal_status"] = status
        case["terminal_stage"] = stage
        case["reason_code"] = reason_code


def _load_event(
    path: Path,
    *,
    root: Path,
    expected_sequence: int,
    expected_previous: str | None,
    manifest_record: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    match = _EVENT_NAME.fullmatch(path.name)
    if match is None or int(match.group(1)) != expected_sequence:
        raise DagError("Route-2 event sequence/name is invalid or has a gap")
    payload, file_record = common.load_json_mapping_record(
        path,
        root=root,
        description="Route-2 serial event",
        error_type=DagError,
        require_mode=0o444,
    )
    actual_hash = file_record["sha256"]
    if actual_hash != match.group(2):
        raise DagError("Route-2 event filename hash changed")
    expected_fields = {
        "schema",
        "sequence",
        "previous_event_sha256",
        "run_manifest",
        "case_id",
        "base_avatar_id",
        "stage",
        "status",
        "reason_code",
        "evidence",
        "recorded_at_utc",
    }
    common.reject_user_approval(payload, DagError, "Route-2 serial event")
    if set(payload) != expected_fields:
        raise DagError("Route-2 event fields changed")
    if (
        payload.get("schema") != EVENT_SCHEMA
        or payload.get("sequence") != expected_sequence
        or payload.get("previous_event_sha256") != expected_previous
        or payload.get("run_manifest") != dict(manifest_record)
        or payload.get("case_id") not in CASE_ORDER
        or payload.get("base_avatar_id")
        != BASE_FOR_CASE.get(str(payload.get("case_id")))
        or payload.get("stage") not in _stages(str(payload.get("case_id")))
        or not isinstance(payload.get("recorded_at_utc"), str)
        or not payload["recorded_at_utc"].endswith("Z")
        or not isinstance(payload.get("evidence"), list)
        or not payload["evidence"]
    ):
        raise DagError("Route-2 event identity, chain, or stage changed")
    evidence = [_validate_evidence_record(value) for value in payload["evidence"]]
    if len({value["path"] for value in evidence}) != len(evidence):
        raise DagError("Route-2 event repeats an evidence path")
    payload = dict(payload)
    payload["evidence"] = evidence
    event_record = _embedded_file_record(file_record)
    payload["event_path"] = event_record["path"]
    payload["event_sha256"] = event_record["sha256"]
    return payload, event_record


def _validate_run(
    run_root: Path, *, allowed_staging_for: str | None = None
) -> dict[str, Any]:
    """Recompute the complete event chain and return the next resumable stage."""

    root = common.require_real_directory(
        common.absolute(run_root), "Route-2 run root", DagError
    )
    manifest, manifest_record = _load_manifest(root)
    events_root = common.require_real_directory(
        root / EVENTS_DIRNAME, "Route-2 events directory", DagError
    )
    raw_entries = sorted(events_root.iterdir(), key=lambda value: value.name)
    entries: list[Path] = []
    skipped_staging = 0
    staging_prefix = (
        None if allowed_staging_for is None else f".{allowed_staging_for}."
    )
    for entry in raw_entries:
        if (
            staging_prefix is not None
            and entry.name.startswith(staging_prefix)
            and entry.name.endswith(".staging")
        ):
            skipped_staging += 1
            continue
        entries.append(entry)
    if skipped_staging > 1:
        raise DagError("multiple Route-2 event staging files exist")
    state = _initial_state(
        root=root, manifest=manifest, manifest_record=manifest_record
    )
    previous: str | None = None
    for sequence, path in enumerate(entries, start=1):
        if path.is_symlink() or not path.is_file():
            raise DagError("Route-2 events directory contains a non-event entry")
        event, record = _load_event(
            path,
            root=events_root,
            expected_sequence=sequence,
            expected_previous=previous,
            manifest_record=manifest_record,
        )
        _apply_event(state, event)
        state["events"].append(event)
        previous = record["sha256"]
    state["next_sequence"] = len(entries) + 1
    state["head_event_sha256"] = previous
    action = next_action(state)
    state["complete"] = action is None and all(
        state["cases"][case_id]["terminal_status"] is not None
        for case_id in CASE_ORDER
    )
    if action is None and not state["complete"]:
        if state["cases"]["rocketbox_male_adult_01"]["terminal_status"] == "rejected":
            state["blocked_reason"] = "male_base_rejected"
        elif (
            state["cases"]["rocketbox_female_adult_01"]["terminal_status"]
            == "rejected"
        ):
            state["blocked_reason"] = "female_base_rejected"
        else:
            state["blocked_reason"] = "base_qualification_required"
    return state


def validate_run(run_root: Path) -> dict[str, Any]:
    """Recompute the complete event chain and return the next resumable stage."""

    return _validate_run(run_root)


def _snapshot_evidence(paths: Sequence[Path]) -> list[dict[str, Any]]:
    if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)) or not paths:
        raise DagError("Route-2 event requires at least one immutable evidence file")
    records = []
    for supplied in paths:
        path = common.absolute(Path(supplied))
        if path.is_symlink():
            raise DagError("Route-2 evidence must be a direct regular file, not a symlink")
        try:
            record = common.hash_file_snapshot(
                path,
                root=path.parent,
                description="Route-2 event evidence",
                error_type=DagError,
                require_mode=0o444,
            )
        except DagError as error:
            if "mode 0444" in str(error):
                raise DagError(
                    "Route-2 event evidence must be immutable/read-only mode 0444"
                ) from error
            raise
        records.append(_embedded_file_record(record))
    if len({value["path"] for value in records}) != len(records):
        raise DagError("Route-2 event evidence paths must be unique")
    return records


def append_stage_event(
    run_root: Path,
    *,
    case_id: str,
    stage: str,
    status: str,
    evidence_paths: Sequence[Path],
    reason_code: str | None = None,
) -> Path:
    """Append exactly the currently authorized stage as a no-replace event."""

    root = common.absolute(run_root)
    before = validate_run(root)
    expected = next_action(before)
    if expected is None or (case_id, stage) != (
        expected["case_id"],
        expected["stage"],
    ):
        raise DagError("event does not match the expected serial Route-2 stage")
    evidence = common.stable_mapping_snapshot(
        lambda: {"records": _snapshot_evidence(evidence_paths)},
        DagError,
        "Route-2 event evidence",
    )["records"]
    payload = {
        "schema": EVENT_SCHEMA,
        "sequence": before["next_sequence"],
        "previous_event_sha256": before["head_event_sha256"],
        "run_manifest": before["run_manifest"],
        "case_id": case_id,
        "base_avatar_id": BASE_FOR_CASE[case_id],
        "stage": stage,
        "status": status,
        "reason_code": reason_code,
        "evidence": evidence,
        "recorded_at_utc": _utc_now(),
    }
    common.reject_user_approval(payload, DagError, "Route-2 serial event")
    trial = json.loads(json.dumps(before))
    _apply_event(trial, payload)
    digest = hashlib.sha256(_json_bytes(payload)).hexdigest()
    destination = root / EVENTS_DIRNAME / (
        f"{before['next_sequence']:06d}.{digest}.json"
    )

    def validate_prelink() -> None:
        current = _validate_run(root, allowed_staging_for=destination.name)
        if (
            current["next_sequence"] != before["next_sequence"]
            or current["head_event_sha256"] != before["head_event_sha256"]
            or next_action(current) != expected
            or _snapshot_evidence(evidence_paths) != evidence
        ):
            raise DagError("Route-2 run or evidence changed before event publication")

    def validate_postlink() -> None:
        current = _validate_run(root, allowed_staging_for=destination.name)
        if (
            current["next_sequence"] != before["next_sequence"] + 1
            or current["head_event_sha256"] != digest
            or _snapshot_evidence(evidence_paths) != evidence
        ):
            raise DagError("Route-2 run or evidence changed after event publication")

    common.write_json_immutable_noreplace(
        destination,
        payload,
        DagError,
        "Route-2 serial event",
        prelink_validator=validate_prelink,
        postlink_validator=validate_postlink,
    )
    state = validate_run(root)
    if (
        state["next_sequence"] != before["next_sequence"] + 1
        or state["head_event_sha256"] != digest
    ):
        raise DagError("Route-2 event publication did not advance the authenticated chain")
    return destination


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create_parser = subparsers.add_parser("create")
    create_parser.add_argument("--run-root", type=Path, required=True)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--run-root", type=Path, required=True)
    append_parser = subparsers.add_parser("append")
    append_parser.add_argument("--run-root", type=Path, required=True)
    append_parser.add_argument("--case-id", required=True)
    append_parser.add_argument("--stage", required=True)
    append_parser.add_argument(
        "--status",
        required=True,
        choices=("succeeded", "rejected", QUALIFIED_STATUS),
    )
    append_parser.add_argument("--evidence", type=Path, action="append", required=True)
    append_parser.add_argument("--reason-code")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "create":
        path = create_run(args.run_root)
        print(f"ROUTE2_DAG_CREATED {path}")
        return 0
    if args.command == "append":
        path = append_stage_event(
            args.run_root,
            case_id=args.case_id,
            stage=args.stage,
            status=args.status,
            evidence_paths=args.evidence,
            reason_code=args.reason_code,
        )
        print(f"ROUTE2_DAG_EVENT_APPENDED {path}")
        return 0
    state = validate_run(args.run_root)
    action = next_action(state)
    print(json.dumps(state, indent=2, sort_keys=True))
    if action is None:
        terminal = "complete" if state["complete"] else state["blocked_reason"]
        print(f"ROUTE2_DAG_TERMINAL {terminal}")
    else:
        print(
            "ROUTE2_DAG_NEXT "
            f"case={action['case_id']} stage={action['stage']}"
        )
    return 0


__all__ = [
    "ATTRIBUTE_CASES",
    "ATTRIBUTE_STAGES",
    "BASE_FOR_CASE",
    "BASE_STAGES",
    "CASE_ORDER",
    "DagError",
    "EVENT_SCHEMA",
    "QUALIFIED_STATUS",
    "RUN_MANIFEST_NAME",
    "RUN_SCHEMA",
    "append_stage_event",
    "create_run",
    "next_action",
    "main",
    "parse_args",
    "validate_run",
]


if __name__ == "__main__":
    raise SystemExit(main())
