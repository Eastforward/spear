#!/usr/bin/env python3
"""Build authenticated Walk/Idle Apartment specs for stable native animals."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any


SCHEMA = "stable_animal_walk_idle_apartment_specs_v1"
IMPORT_SCHEMA = "stable_animal_ue_import_batch_v1"
IMPORT_RESULT_SCHEMA = "stable_animal_ue_import_result_v1"
REGISTRY_SCHEMA = "avengine_quaternius_stable_template_registry_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, *, published_path: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or unsafe artifact: {path}")
    return {
        "path": str((published_path or path).resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _artifact_matches(record: dict[str, Any]) -> bool:
    try:
        path = Path(record["path"]).resolve()
        return bool(
            not path.is_symlink()
            and path.is_file()
            and path.stat().st_size == record["size_bytes"]
            and _sha256(path) == record["sha256"]
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def authenticate(
    *, jobs_path: Path, result_path: Path, registry_path: Path
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    jobs = _load(jobs_path)
    result = _load(result_path)
    registry = _load(registry_path)
    if (
        jobs.get("schema") != IMPORT_SCHEMA
        or jobs.get("job_count") != len(jobs.get("jobs", []))
        or not _artifact_matches(jobs.get("registry", {}))
        or Path(jobs["registry"]["path"]).resolve() != registry_path.resolve()
        or registry.get("schema") != REGISTRY_SCHEMA
    ):
        raise ValueError("stable import jobs or registry changed")
    if (
        result.get("schema") != IMPORT_RESULT_SCHEMA
        or Path(result.get("input_manifest", "")).resolve() != jobs_path.resolve()
        or result.get("input_manifest_sha256") != _sha256(jobs_path)
        or result.get("passed_count") != len(result.get("results", []))
    ):
        raise ValueError("stable UE import result is incomplete")
    entries = {entry["template_id"]: entry for entry in registry.get("entries", [])}
    results = {item["template_id"]: item for item in result.get("results", [])}
    selected = jobs.get("jobs", [])
    if not selected or len(results) != len(selected):
        raise ValueError("stable UE import coverage changed")
    for job in selected:
        template_id = job["template_id"]
        entry = entries.get(template_id)
        imported = results.get(template_id)
        if (
            entry is None
            or imported is None
            or job.get("asset_id") != template_id
            or imported.get("asset_id") != template_id
            or imported.get("tag") != job.get("tag")
            or imported.get("source_sha256") != job.get("rigged_glb_sha256")
            or set(imported.get("actions", [])) != {"Idle", "Walking"}
            or entry.get("runtime_glb", {}).get("sha256")
            != job.get("rigged_glb_sha256")
            or entry.get("direction", {}).get("review_status")
            != "agent_selected_pending_human_review"
            or entry.get("formal_dataset_registration_authorized") is not False
        ):
            raise ValueError(f"stable UE identity changed: {template_id}")
    return sorted(selected, key=lambda item: item["template_id"]), entries


def build_pair(
    template: dict[str, Any], *, job: dict[str, Any], gate: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    walking = copy.deepcopy(template)
    walking["description"] = (
        f"Stable native {job['breed']} starts camera-right/rear, passes "
        "camera-left/front, then walks one counter-clockwise table loop."
    )
    walking["usage_scope"] = "research_candidate"
    walking["formal_registration_authorized"] = False
    source = walking["sources"][0]
    source.update(
        {
            "tag": job["tag"],
            "asset_id": job["asset_id"],
            "template_id": job["template_id"],
            "asset_class": "animal",
            "species": job["species"],
            "breed": job["breed"],
            "kind": "moving",
            "motion": "explicit_camera_pass_table_loop",
            "motion_style": "camera_pass_then_round_table_loop",
            "wanted_anim": "Walking",
            "walking_forward_yaw_offset_deg": job[
                "walking_forward_yaw_offset_deg"
            ],
            "actor_scale": job["actor_scale"],
            "actor_z_lift_cm": 0.0,
            "animation_play_rate": 1.0,
            "ground_snap_to_floor": True,
            "ground_snap_max_abs_correction_cm": max(
                25.0, round(float(job["actor_scale"]) * 200.0, 3)
            ),
            "audio_lookup": job["audio_lookup"],
            "audio_source_height_offset_m": job[
                "audio_source_height_offset_m"
            ],
            "adaptive_repeat_short_calls": True,
            "strict_audio": True,
            "mute_audio": False,
            "stable_animal_gate": copy.deepcopy(gate),
        }
    )
    walking["camera_pass_table_loop_contract"]["animal_scale_rationale"] = {
        "template_id": job["template_id"],
        "actor_scale": job["actor_scale"],
        "policy": (
            "versioned stable-template canary scale; packaged UE bounds and "
            "physical measurement remain authoritative"
        ),
    }

    idle = copy.deepcopy(walking)
    trajectory = walking["sources"][0]["trajectory_m"]
    idle_frame = int(
        walking["camera_pass_table_loop_contract"]["left_front_nearest_frame"]
    )
    idle_position = copy.deepcopy(trajectory[idle_frame])
    idle_source = idle["sources"][0]
    idle_source.update(
        {
            "kind": "stationary",
            "wanted_anim": "Idle",
            "motion": "stationary_left_front",
            "motion_style": "stationary_visible_left_front",
            "start_pos_m": idle_position,
            "end_pos_m": idle_position,
            "trajectory_m": [copy.deepcopy(idle_position) for _ in trajectory],
        }
    )
    idle["description"] = (
        f"Stable native {job['breed']} holds its authored Idle at the visible "
        "camera-left/front waypoint without actor translation."
    )
    idle["trajectory_profile"] = "stationary_camera_left_front_v1"
    idle.pop("rig_direction_check_windows", None)
    idle["stationary_idle_contract"] = {
        "status": "passed",
        "frame_count": len(trajectory),
        "position_m": idle_position,
        "maximum_position_delta_m": 0.0,
        "source_waypoint_frame": idle_frame,
    }
    return {"Walking": walking, "Idle": idle}


def build_specs(
    *,
    jobs_path: Path,
    result_path: Path,
    registry_path: Path,
    template_path: Path,
    output_root: Path,
) -> Path:
    paths = [jobs_path, result_path, registry_path, template_path]
    for path in paths:
        if path.resolve().is_symlink() or not path.resolve().is_file():
            raise ValueError(f"missing stable Apartment input: {path}")
    selected, entries = authenticate(
        jobs_path=jobs_path.resolve(),
        result_path=result_path.resolve(),
        registry_path=registry_path.resolve(),
    )
    template = _load(template_path.resolve())
    output_root = output_root.absolute()
    if output_root.exists() or output_root.is_symlink():
        raise ValueError(f"refusing to replace output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    try:
        records = []
        registry_artifact = _artifact(registry_path)
        import_artifact = _artifact(result_path)
        for job in selected:
            entry = entries[job["template_id"]]
            gate = {
                "schema": "stable_animal_apartment_gate_v1",
                "status": "approved_for_automated_research_candidate_apartment",
                "asset_id": job["asset_id"],
                "template_id": job["template_id"],
                "tag": job["tag"],
                "species": job["species"],
                "breed": job["breed"],
                "template_registry": registry_artifact,
                "ue_import_result": import_artifact,
                "source_sha256": job["rigged_glb_sha256"],
                "deformation_audit": copy.deepcopy(entry["deformation_audit"]),
                "direction": copy.deepcopy(entry["direction"]),
                "human_visual_review": "pending",
                "formal_dataset_registration_authorized": False,
            }
            action_records = {}
            for action, spec in build_pair(template, job=job, gate=gate).items():
                motion = action.lower()
                clip_name = f"camera_pass_table_loop_{motion}"
                relative_spec = Path("specs") / job["tag"] / f"{clip_name}.json"
                staged_spec = staging / relative_spec
                staged_spec.parent.mkdir(parents=True, exist_ok=True)
                staged_spec.write_text(
                    json.dumps(spec, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                published_spec = output_root / relative_spec
                action_records[action] = {
                    "motion": motion,
                    "spec": str(published_spec),
                    "spec_evidence": _artifact(
                        staged_spec, published_path=published_spec
                    ),
                    "output_dir": str(
                        output_root / "clips" / job["tag"] / clip_name
                    ),
                    "clip_id": f"{job['tag']}_{clip_name}_v1",
                }
            records.append(
                {
                    "base_avatar_id": job["asset_id"],
                    "asset_id": job["asset_id"],
                    "template_id": job["template_id"],
                    "tag": job["tag"],
                    "species": job["species"],
                    "breed": job["breed"],
                    "source_glb": {
                        "path": job["rigged_glb"],
                        "sha256": job["rigged_glb_sha256"],
                    },
                    "actions": action_records,
                }
            )
        manifest = {
            "schema": SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "usage_scope": "research_candidate",
            "formal_registration_authorized": False,
            "human_visual_review": "pending",
            "trajectory_policy": (
                "Walking uses camera right/rear -> left/front -> one table loop; "
                "Idle is stationary at left/front"
            ),
            "audio_policy": (
                "species-matched short calls are energy-segmented and repeated "
                "with silent gaps"
            ),
            "avatar_count": len(records),
            "clip_count": len(records) * 2,
            "inputs": {
                "ue_import_jobs": _artifact(jobs_path),
                "ue_import_result": import_artifact,
                "template_registry": registry_artifact,
                "scene_template": _artifact(template_path),
            },
            "records": records,
        }
        (staging / "spec_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, output_root)
        return output_root / "spec_manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = build_specs(
        jobs_path=args.jobs,
        result_path=args.result,
        registry_path=args.registry,
        template_path=args.template,
        output_root=args.output_root,
    )
    print(f"STABLE_ANIMAL_APARTMENT_SPECS_OK manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
