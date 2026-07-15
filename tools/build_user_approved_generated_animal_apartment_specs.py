#!/usr/bin/env python3
"""Build authenticated Apartment Walk/Idle specs for one user-approved generated animal."""

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
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts


CONFIG_SCHEMA = "user_approved_generated_animal_apartment_config_v1"
OUTPUT_SCHEMA = "controlled_animal_walk_idle_apartment_specs_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash_without(value: Mapping[str, Any], key: str) -> str:
    payload = {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    return hashlib.sha256(contracts.canonical_json(payload).encode("utf-8")).hexdigest()


def _artifact(path: Path, *, published_path: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise contracts.ContractError(f"missing or unsafe artifact: {path}")
    return {
        "path": str((published_path or path).resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _descriptor_matches(record: Mapping[str, Any]) -> bool:
    try:
        path = Path(str(record["path"])).resolve()
        return bool(
            not path.is_symlink()
            and path.is_file()
            and path.stat().st_size == int(record["size_bytes"])
            and _sha256(path) == record["sha256"]
        )
    except (KeyError, TypeError, ValueError, OSError):
        return False


def _load(path: Path) -> dict[str, Any]:
    value = contracts.load_json(path.resolve())
    if not isinstance(value, dict):
        raise contracts.ContractError(f"JSON object required: {path}")
    return value


def _authenticate(
    *, config_path: Path, jobs_path: Path, result_path: Path, decision_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load(config_path)
    jobs = _load(jobs_path)
    result = _load(result_path)
    decision = _load(decision_path)
    required_config = {
        "schema",
        "asset_id",
        "tag",
        "profile_schema_id",
        "species",
        "breed",
        "sampled_attributes",
        "actor_scale",
        "walking_forward_yaw_offset_deg",
        "ground_snap_max_abs_correction_cm",
        "audio_lookup",
        "audio_source_height_offset_m",
        "scale_rationale",
        "state_classification",
        "formal_dataset_registration_authorized",
    }
    if set(config) != required_config or config.get("schema") != CONFIG_SCHEMA:
        raise contracts.ContractError("generated animal Apartment config fields changed")
    if (
        config.get("state_classification") != "research_candidate"
        or config.get("formal_dataset_registration_authorized") is not False
        or not str(config.get("tag", "")).startswith("pixal_")
        or not all(
            isinstance(config.get(name), str) and config[name]
            for name in ("asset_id", "profile_schema_id", "species", "breed", "audio_lookup")
        )
        or not isinstance(config.get("sampled_attributes"), dict)
        or not config["sampled_attributes"]
        or not all(
            isinstance(name, str) and name and isinstance(value, str) and value
            for name, value in config["sampled_attributes"].items()
        )
    ):
        raise contracts.ContractError("generated animal Apartment config is invalid")
    actor_scale = float(config["actor_scale"])
    yaw = float(config["walking_forward_yaw_offset_deg"])
    ground_limit = float(config["ground_snap_max_abs_correction_cm"])
    audio_height = float(config["audio_source_height_offset_m"])
    if (
        not 0.01 <= actor_scale <= 2.0
        or abs(yaw / 90.0 - round(yaw / 90.0)) > 1e-9
        or not 0.0 < ground_limit <= 200.0
        or not 0.0 <= audio_height <= 5.0
    ):
        raise contracts.ContractError("scale, cardinal direction, ground snap, or audio height is unsafe")

    job_list = jobs.get("jobs", [])
    result_list = result.get("results", [])
    if (
        jobs.get("schema") != "pixal_animal_ue_import_batch_v1"
        or len(job_list) != 1
        or result.get("schema") != "pixal_animal_ue_import_result_v1"
        or Path(str(result.get("input_manifest", ""))).resolve() != jobs_path.resolve()
        or result.get("passed_count") != 1
        or len(result_list) != 1
    ):
        raise contracts.ContractError("generated animal UE import coverage is incomplete")
    job = job_list[0]
    imported = result_list[0]
    runtime = Path(str(job.get("rigged_glb", ""))).resolve()
    if (
        job.get("legacy_tag") != config["asset_id"]
        or job.get("asset_id") != config["asset_id"]
        or job.get("tag") != config["tag"]
        or job.get("profile_schema_id") != config["profile_schema_id"]
        or job.get("sampled_attributes") != config["sampled_attributes"]
        or set(job.get("expected_actions", [])) != {"Walking", "Idle"}
        or runtime.is_symlink()
        or not runtime.is_file()
        or _sha256(runtime) != job.get("rigged_glb_sha256")
        or imported.get("legacy_tag") != config["asset_id"]
        or imported.get("tag") != config["tag"]
        or imported.get("source_sha256") != job.get("rigged_glb_sha256")
        or set(imported.get("actions", [])) != {"Walking", "Idle"}
        or imported.get("status") != "passed"
    ):
        raise contracts.ContractError("generated animal UE identity or action readback changed")

    if (
        decision.get("schema") != "avengine_controlled_animal_animation_decision_v1"
        or decision.get("asset_id") != config["asset_id"]
        or decision.get("decision") != "approved_for_ue_apartment"
        or decision.get("state_classification") != "research_candidate"
        or decision.get("formal_dataset_registration_authorized") is not False
        or decision.get("decision_sha256")
        != _canonical_hash_without(decision, "decision_sha256")
        or not _descriptor_matches(decision.get("review", {}))
    ):
        raise contracts.ContractError("user animation decision is missing, changed, or not approved")
    return config, job, decision


def _build_pair(
    template: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    walking = copy.deepcopy(template)
    walking.update(
        {
            "description": (
                f"User-approved generated {config['breed']} starts camera-right/rear, "
                "passes camera-left/front, then walks one counter-clockwise table loop."
            ),
            "usage_scope": "research_candidate",
            "formal_registration_authorized": False,
        }
    )
    source = walking["sources"][0]
    source.update(
        {
            "tag": config["tag"],
            "asset_id": config["asset_id"],
            "asset_class": "animal",
            "species": config["species"],
            "breed": config["breed"],
            "profile_schema_id": config["profile_schema_id"],
            "sampled_attributes": copy.deepcopy(config["sampled_attributes"]),
            "kind": "moving",
            "motion": "explicit_camera_pass_table_loop",
            "motion_style": "camera_pass_then_round_table_loop",
            "wanted_anim": "Walking",
            "walking_forward_yaw_offset_deg": config["walking_forward_yaw_offset_deg"],
            "actor_scale": config["actor_scale"],
            "actor_z_lift_cm": 0.0,
            "animation_play_rate": 1.0,
            "ground_snap_to_floor": True,
            "ground_snap_max_abs_correction_cm": config["ground_snap_max_abs_correction_cm"],
            "audio_lookup": config["audio_lookup"],
            "audio_source_height_offset_m": config["audio_source_height_offset_m"],
            "adaptive_repeat_short_calls": True,
            "strict_audio": True,
            "mute_audio": False,
            "controlled_animal_gate": copy.deepcopy(gate),
        }
    )
    walking["camera_pass_table_loop_contract"]["animal_scale_rationale"] = {
        "actor_scale": config["actor_scale"],
        "policy": config["scale_rationale"],
        "profile_schema_id": config["profile_schema_id"],
    }

    idle = copy.deepcopy(walking)
    trajectory = walking["sources"][0]["trajectory_m"]
    idle_frame = int(walking["camera_pass_table_loop_contract"]["left_front_nearest_frame"])
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
        f"User-approved generated {config['breed']} holds Idle at the visible "
        "camera-left/front waypoint without translation."
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
    config_path: Path,
    ue_jobs: Path,
    ue_result: Path,
    animation_decision: Path,
    template: Path,
    output_root: Path,
) -> Path:
    for path in (config_path, ue_jobs, ue_result, animation_decision, template):
        _artifact(path)
    config, job, decision = _authenticate(
        config_path=config_path.resolve(),
        jobs_path=ue_jobs.resolve(),
        result_path=ue_result.resolve(),
        decision_path=animation_decision.resolve(),
    )
    template_payload = _load(template)
    output_root = output_root.absolute()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(f"refusing to replace output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    try:
        decision_artifact = _artifact(animation_decision)
        decision_artifact["decision_sha256"] = decision["decision_sha256"]
        gate = {
            "schema": "controlled_animal_apartment_gate_v1",
            "status": "approved_for_research_candidate_apartment",
            "asset_id": config["asset_id"],
            "tag": config["tag"],
            "animation_decision": decision_artifact,
            "ue_import_result": _artifact(ue_result),
            "ue_source_sha256": job["rigged_glb_sha256"],
            "formal_dataset_registration_authorized": False,
        }
        actions: dict[str, Any] = {}
        for action, spec in _build_pair(template_payload, config=config, gate=gate).items():
            motion = action.lower()
            clip_name = f"camera_pass_table_loop_{motion}"
            relative_spec = Path("specs") / config["tag"] / f"{clip_name}.json"
            staged_spec = staging / relative_spec
            contracts.write_json_no_replace(staged_spec, spec)
            published_spec = output_root / relative_spec
            actions[action] = {
                "motion": motion,
                "spec": str(published_spec),
                "spec_evidence": _artifact(staged_spec, published_path=published_spec),
                "output_dir": str(output_root / "clips" / config["tag"] / clip_name),
                "clip_id": f"{config['tag']}_{clip_name}_v1",
            }
        manifest: dict[str, Any] = {
            "schema": OUTPUT_SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "usage_scope": "research_candidate",
            "formal_registration_authorized": False,
            "trajectory_policy": "Walking uses camera right/rear -> left/front -> one table loop; Idle is stationary at left/front",
            "audio_policy": "species-matched short calls are segmented and repeated with silent gaps",
            "avatar_count": 1,
            "clip_count": 2,
            "inputs": {
                "config": _artifact(config_path),
                "ue_import_jobs": _artifact(ue_jobs),
                "ue_import_result": _artifact(ue_result),
                "animation_decision": decision_artifact,
                "template": _artifact(template),
            },
            "records": [
                {
                    "base_avatar_id": config["asset_id"],
                    "asset_id": config["asset_id"],
                    "tag": config["tag"],
                    "profile_schema_id": config["profile_schema_id"],
                    "species": config["species"],
                    "breed": config["breed"],
                    "sampled_attributes": copy.deepcopy(config["sampled_attributes"]),
                    "source_glb": {
                        "path": job["rigged_glb"],
                        "sha256": job["rigged_glb_sha256"],
                    },
                    "actions": actions,
                }
            ],
        }
        manifest["manifest_sha256"] = contracts.manifest_sha256(manifest)
        contracts.write_json_no_replace(staging / "spec_manifest.json", manifest)
        os.rename(staging, output_root)
        return output_root / "spec_manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--ue-jobs", required=True, type=Path)
    parser.add_argument("--ue-result", required=True, type=Path)
    parser.add_argument("--animation-decision", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_specs(
            config_path=args.config,
            ue_jobs=args.ue_jobs,
            ue_result=args.ue_result,
            animation_decision=args.animation_decision,
            template=args.template,
            output_root=args.output_root,
        )
    except (contracts.ContractError, OSError, KeyError, TypeError, ValueError) as error:
        print(f"USER_APPROVED_GENERATED_ANIMAL_APARTMENT_SPECS_FAILED {error}")
        return 2
    print(f"USER_APPROVED_GENERATED_ANIMAL_APARTMENT_SPECS_OK manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
