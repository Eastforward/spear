#!/usr/bin/env python3
"""Authenticate one-factor-at-a-time stable-animal instance review evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence


SPEAR_ROOT = Path(__file__).resolve().parents[1]
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import controlled_source_asset_schema as contracts  # noqa: E402
from tools import prepare_controlled_source_asset_execution as preflight_lib  # noqa: E402


SCHEMA = "avengine_stable_animal_ofat_review_v1"
REALIZATION_SCHEMA = "avengine_stable_animal_instance_realization_v1"
DEFORMATION_SCHEMA = "avengine_skinned_deformation_audit_v1"
REQUIRED_ACTIONS = {"Walking", "Idle"}


class ReviewError(RuntimeError):
    """Raised when an OFAT artifact or invariant cannot be authenticated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ReviewError(f"missing or unsafe {label}: {path}")
    return path


def load_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    path = regular_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ReviewError(f"{label} must be a JSON object: {path}")
    return path, payload


def file_record(path: Path) -> dict[str, Any]:
    path = regular_file(path, "artifact")
    return {
        "absolute_path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def parse_selection(values: Sequence[str]) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    labels: set[str] = set()
    instances: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ReviewError("--selection must use LABEL=INSTANCE_ID")
        label, instance_id = value.split("=", 1)
        if not label or not instance_id or label in labels or instance_id in instances:
            raise ReviewError(f"duplicate or empty selection: {value}")
        labels.add(label)
        instances.add(instance_id)
        result.append((label, instance_id))
    if "baseline" not in labels:
        raise ReviewError("one --selection must be named baseline")
    return result


def ffprobe_video(path: Path) -> dict[str, Any]:
    path = regular_file(path, "review video")
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise ReviewError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    try:
        probe = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ReviewError(f"invalid ffprobe JSON for {path}") from error
    streams = probe.get("streams", [])
    if len(streams) != 1:
        raise ReviewError(f"exactly one video stream required: {path}")
    stream = streams[0]
    duration = float(probe.get("format", {}).get("duration", 0.0))
    frames = int(stream.get("nb_frames", 0))
    if (
        stream.get("codec_name") != "h264"
        or int(stream.get("width", 0)) <= 0
        or int(stream.get("height", 0)) <= 0
        or duration <= 0.0
        or frames <= 0
    ):
        raise ReviewError(f"review video readback is incomplete: {path}")
    return {
        **file_record(path),
        "codec": stream["codec_name"],
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream.get("r_frame_rate"),
        "frame_count": frames,
        "duration_seconds": duration,
    }


def validate_realization(
    path: Path,
    *,
    instance_id: str,
    profile_id: str,
    profile_hash: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path, payload = load_json(path, "realization manifest")
    glb_record = payload.get("artifacts", {}).get("glb", {})
    glb_path = regular_file(Path(str(glb_record.get("path", ""))), "instance GLB")
    if (
        payload.get("schema") != REALIZATION_SCHEMA
        or payload.get("instance_id") != instance_id
        or payload.get("state_classification") != "research_candidate"
        or payload.get("formal_dataset_registration_authorized") is not False
        or payload.get("request", {}).get("instance_id") != instance_id
        or payload.get("realization", {}).get("topology_uv_skin_unchanged") is not True
        or payload.get("realization", {}).get("actions_unchanged") is not True
        or glb_record.get("sha256") != sha256_file(glb_path)
        or glb_record.get("size_bytes") != glb_path.stat().st_size
    ):
        raise ReviewError(f"realization contract failed: {instance_id}")
    # The authenticated preflight/request carries the profile ID/hash.  Keep
    # both in the final batch so a profile file cannot be swapped later.
    if payload.get("request", {}).get("request_sha256") is None:
        raise ReviewError(f"realization request hash missing: {instance_id}")
    return payload, {
        "manifest": file_record(path),
        "glb": file_record(glb_path),
        "profile_schema_id": profile_id,
        "profile_sha256": profile_hash,
    }


def validate_deformation(path: Path, instance_id: str) -> dict[str, Any]:
    path, payload = load_json(path, "deformation audit")
    actions = payload.get("actions", [])
    requested = {item.get("requested_action") for item in actions}
    if (
        payload.get("schema") != DEFORMATION_SCHEMA
        or payload.get("overall") != "passed"
        or requested != REQUIRED_ACTIONS
        or any(
            item.get("decision") != "passed_automatic_deformation_measurements"
            for item in actions
        )
    ):
        raise ReviewError(f"Walk/Idle deformation failed: {instance_id}")
    return {
        "artifact": file_record(path),
        "overall": payload["overall"],
        "actions": {
            item["requested_action"]: {
                "resolved_action": item["resolved_action"],
                "decision": item["decision"],
                "worst_case": item["worst_case"],
            }
            for item in actions
        },
    }


def validate_ofat_coverage(
    profile: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    domains = profile["sampled_attribute_domains"]
    baseline_rows = [item for item in entries if item["label"] == "baseline"]
    if len(baseline_rows) != 1:
        raise ReviewError("exactly one baseline selection is required")
    baseline = baseline_rows[0]["sampled_attributes"]
    coverage: dict[str, list[str]] = {}
    for attribute, values in domains.items():
        observed = sorted({item["sampled_attributes"][attribute] for item in entries})
        if set(observed) != set(values):
            raise ReviewError(
                f"attribute domain is not fully covered: {attribute} {observed} != {values}"
            )
        coverage[attribute] = observed
    for item in entries:
        if item["label"] == "baseline":
            continue
        differences = [
            key
            for key in sorted(domains)
            if item["sampled_attributes"][key] != baseline[key]
        ]
        if len(differences) != 1:
            raise ReviewError(
                f"selection is not one-factor-at-a-time: {item['label']} {differences}"
            )
        item["changed_attribute_from_baseline"] = differences[0]
    return {
        "baseline_attributes": baseline,
        "domain_coverage": coverage,
        "all_domain_values_realized": True,
        "every_nonbaseline_entry_changes_exactly_one_attribute": True,
    }


def finalize(args: argparse.Namespace) -> Path:
    profile_path, raw_profile = load_json(args.profile, "attribute profile")
    profile = contracts.validate_attribute_profile(raw_profile)
    profile_hash = contracts.profile_sha256(profile)
    preflight_path, raw_preflight = load_json(args.preflight, "execution preflight")
    preflight = preflight_lib.validate_execution_preflight(raw_preflight)
    selections = parse_selection(args.selection)
    jobs = {
        consumer["instance_id"]: job
        for job in preflight["routes"]["stable_animal_template_v1"]
        for consumer in job.get("consumer_requests", [])
    }
    entries: list[dict[str, Any]] = []
    for label, instance_id in selections:
        job = jobs.get(instance_id)
        if (
            job is None
            or job.get("profile_schema_id") != profile["profile_schema_id"]
            or job.get("profile_sha256") != profile_hash
        ):
            raise ReviewError(f"instance is not authenticated by this profile: {instance_id}")
        realization_dir = args.realizations_root / instance_id
        realization, realization_records = validate_realization(
            realization_dir / "manifest.json",
            instance_id=instance_id,
            profile_id=profile["profile_schema_id"],
            profile_hash=profile_hash,
        )
        deformation = validate_deformation(
            args.deformation_root / f"{instance_id}.json", instance_id
        )
        static_image = file_record(args.static_root / label / "frame_0000.png")
        videos = {
            action: ffprobe_video(
                args.animation_root
                / label
                / action.lower()
                / f"{action.lower()}_side_review.mp4"
            )
            for action in REQUIRED_ACTIONS
        }
        entries.append(
            {
                "label": label,
                "instance_id": instance_id,
                "sampled_attributes": realization["sampled_attributes"],
                "fixed_attributes": realization["fixed_attributes"],
                "target_physical_profile": realization["target_physical_profile"],
                "attribute_operations": realization["attribute_operations"],
                "realization": realization_records,
                "deformation": deformation,
                "static_fixed_scale_review": static_image,
                "videos": videos,
                "qa": {
                    "glb_readback": "passed",
                    "topology_uv_skin_preserved": "passed",
                    "actions_preserved": "passed",
                    "walking_deformation": "passed",
                    "idle_deformation": "passed",
                    "media_readback": "passed",
                    "human_visual_review": "pending",
                    "ue_apartment": "pending",
                    "audio": "pending",
                },
            }
        )
    coverage = validate_ofat_coverage(profile, entries)
    contact_sheets = [file_record(path) for path in args.contact_sheet]
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state_classification": "research_candidate_pending_human_and_ue_review",
        "formal_dataset_registration_authorized": False,
        "profile": {
            **file_record(profile_path),
            "profile_schema_id": profile["profile_schema_id"],
            "profile_sha256": profile_hash,
        },
        "preflight": {
            **file_record(preflight_path),
            "preflight_sha256": preflight["preflight_sha256"],
        },
        "entry_count": len(entries),
        "ofat": coverage,
        "entries": entries,
        "contact_sheets": contact_sheets,
        "automatic_checks": {
            "profile_and_preflight_authenticated": True,
            "all_instance_manifests_and_glbs_authenticated": True,
            "all_topology_uv_skin_and_action_invariants_preserved": True,
            "all_walk_idle_deformation_audits_passed": True,
            "all_static_and_video_media_read_back": True,
            "all_domain_values_realized": True,
            "overall": "passed_pending_human_ue_audio_review",
        },
    }
    payload["manifest_sha256"] = contracts.manifest_sha256(payload)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise ReviewError(f"refusing to replace output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    observed = json.loads(output.read_text(encoding="utf-8"))
    if observed.get("manifest_sha256") != contracts.manifest_sha256(observed):
        raise ReviewError("published manifest readback hash mismatch")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--realizations-root", type=Path, required=True)
    parser.add_argument("--deformation-root", type=Path, required=True)
    parser.add_argument("--static-root", type=Path, required=True)
    parser.add_argument("--animation-root", type=Path, required=True)
    parser.add_argument("--selection", action="append", required=True)
    parser.add_argument("--contact-sheet", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = finalize(args)
    except (ReviewError, contracts.ContractError, OSError, ValueError) as error:
        print(f"STABLE_ANIMAL_OFAT_REVIEW_FAILED {error}", file=sys.stderr)
        return 2
    print(f"STABLE_ANIMAL_OFAT_REVIEW_OK output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
