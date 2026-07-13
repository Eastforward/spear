#!/usr/bin/env python3
"""Build authenticated Walk/Idle Apartment specs for approved controlled animals."""

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
from tools import review_controlled_animal_animation_candidates as animation_decisions
from tools import run_controlled_animal_lod_binding as lod_binding


SCHEMA = "controlled_animal_walk_idle_apartment_specs_v1"
SPEAR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = (
    SPEAR_ROOT
    / "tmp/pixal_animal_camera_pass_table_loop_apartment_review_v1/specs/"
    "dog_pug_pixal_canary_v2_100k/camera_pass_table_loop_walking.json"
)

PROFILE: dict[str, dict[str, Any]] = {
    "dog_beagle_v1": {
        "species": "dog",
        "breed": "beagle",
        "base_actor_scale": 0.08,
        "audio_lookup": "dog_bark",
        "audio_height_m": 0.35,
    },
    "dog_golden_retriever_v1": {
        "species": "dog",
        "breed": "golden_retriever",
        "base_actor_scale": 0.13,
        "audio_lookup": "dog_bark",
        "audio_height_m": 0.45,
    },
    "dog_pug_v1": {
        "species": "dog",
        "breed": "pug",
        "base_actor_scale": 0.15,
        "audio_lookup": "dog_bark",
        "audio_height_m": 0.30,
    },
    "cat_siamese_bindpose_v2": {
        "species": "cat",
        "breed": "siamese",
        "base_actor_scale": 0.09,
        "audio_lookup": "cat_meow",
        "audio_height_m": 0.25,
    },
    "cat_tabby_bindpose_v2": {
        "species": "cat",
        "breed": "tabby",
        "base_actor_scale": 0.09,
        "audio_lookup": "cat_meow",
        "audio_height_m": 0.25,
    },
    "cat_tabby_four_limb_rest_side_v5": {
        "species": "cat",
        "breed": "tabby",
        "base_actor_scale": 0.09,
        "audio_lookup": "cat_meow",
        "audio_height_m": 0.25,
    },
    "dog_beagle_four_limb_rest_side_clay_v6": {
        "species": "dog",
        "breed": "beagle",
        "base_actor_scale": 0.08,
        "audio_lookup": "dog_bark",
        "audio_height_m": 0.35,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256({name: copy.deepcopy(item) for name, item in value.items() if name != key})


def _record(path: Path, *, published_path: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str((published_path or path).resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = contracts.load_json(path.resolve())
    if not isinstance(payload, dict):
        raise contracts.ContractError(f"JSON object required: {path}")
    return payload


def _authenticate_inputs(
    jobs_path: Path,
    result_path: Path,
    lod_path: Path,
    decisions_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    jobs = _load_json(jobs_path)
    result = _load_json(result_path)
    lod = _load_json(lod_path)
    decisions = _load_json(decisions_path)
    if jobs.get("schema") != "pixal_animal_ue_import_batch_v1":
        raise contracts.ContractError("UE import jobs schema changed")
    if (
        result.get("schema") != "pixal_animal_ue_import_result_v1"
        or Path(result.get("input_manifest", "")).resolve() != jobs_path.resolve()
        or result.get("passed_count") != len(result.get("results", []))
    ):
        raise contracts.ContractError("UE import result is incomplete")
    if (
        lod.get("schema") != lod_binding.BATCH_SCHEMA
        or lod.get("batch_sha256") != _hash_without(lod, "batch_sha256")
        or lod.get("failed_count") != 0
        or lod.get("passed_count") != len(lod.get("attempts", []))
    ):
        raise contracts.ContractError("LOD/binding batch is invalid")
    if (
        decisions.get("schema") != animation_decisions.DECISION_BATCH_SCHEMA
        or decisions.get("decision_batch_sha256")
        != _hash_without(decisions, "decision_batch_sha256")
    ):
        raise contracts.ContractError("animation decision batch is invalid")

    lod_by_id = {item["asset_id"]: item for item in lod["attempts"]}
    approved: dict[str, dict[str, Any]] = {}
    decision_root = decisions_path.resolve().parent
    for index in decisions.get("decisions", []):
        record_path = (decision_root / index["record"]["path"]).resolve()
        if (
            record_path.is_symlink()
            or not record_path.is_file()
            or record_path.stat().st_size != index["record"]["size_bytes"]
            or _sha256(record_path) != index["record"]["sha256"]
        ):
            raise contracts.ContractError("animation decision record changed")
        record = _load_json(record_path)
        if (
            record.get("decision_sha256") != index.get("decision_sha256")
            or record.get("decision_sha256") != _hash_without(record, "decision_sha256")
        ):
            raise contracts.ContractError("animation decision record hash changed")
        if record.get("decision") == "approved_for_ue_apartment":
            approved[index["asset_id"]] = {
                "payload": record,
                "path": record_path,
            }

    jobs_by_id = {item["legacy_tag"]: item for item in jobs.get("jobs", [])}
    results_by_id = {item["legacy_tag"]: item for item in result.get("results", [])}
    if not approved or set(jobs_by_id) != set(results_by_id) or set(jobs_by_id) != set(approved):
        raise contracts.ContractError("approved/imported controlled animal coverage changed")
    if len(jobs_by_id) != len(jobs.get("jobs", [])):
        raise contracts.ContractError("duplicate controlled animal import job")
    for asset_id, job in jobs_by_id.items():
        attempt = lod_by_id.get(asset_id)
        imported = results_by_id[asset_id]
        source = Path(job["rigged_glb"]).resolve()
        if (
            attempt is None
            or job.get("profile_schema_id") != attempt.get("profile_schema_id")
            or job.get("sampled_attributes") != attempt.get("sampled_attributes")
            or set(job.get("expected_actions", [])) != {"Idle", "Walking"}
            or imported.get("tag") != job.get("tag")
            or set(imported.get("actions", [])) != {"Idle", "Walking"}
            or imported.get("source_sha256") != job.get("rigged_glb_sha256")
            or not source.is_file()
            or _sha256(source) != job.get("rigged_glb_sha256")
            or attempt.get("profile_schema_id") not in PROFILE
        ):
            raise contracts.ContractError(f"controlled animal identity changed: {asset_id}")
    return sorted(jobs["jobs"], key=lambda item: item["legacy_tag"]), lod_by_id, approved


def build_pair(
    template: dict[str, Any], *, job: Mapping[str, Any], attempt: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    profile = PROFILE[str(attempt["profile_schema_id"])]
    physical = copy.deepcopy(attempt["target_physical_profile"])
    scale_ratio = float(physical["scale_ratio"])
    actor_scale = round(float(profile["base_actor_scale"]) * scale_ratio, 5)
    if not 0.01 <= actor_scale <= 1.0:
        raise contracts.ContractError(f"unsafe controlled animal scale: {actor_scale}")
    ground_snap_limit_cm = round(max(25.0, actor_scale * 200.0), 3)

    walking = copy.deepcopy(template)
    walking["description"] = (
        f"Controlled {profile['breed']} {job['legacy_tag']} starts camera-right/rear, "
        "passes camera-left/front, then walks one counter-clockwise loop around the round table."
    )
    walking["usage_scope"] = "research_candidate"
    walking["formal_registration_authorized"] = False
    source = walking["sources"][0]
    source.update(
        {
            "tag": job["tag"],
            "asset_id": job["legacy_tag"],
            "asset_class": "animal",
            "species": profile["species"],
            "breed": profile["breed"],
            "profile_schema_id": attempt["profile_schema_id"],
            "sampled_attributes": copy.deepcopy(attempt["sampled_attributes"]),
            "kind": "moving",
            "wanted_anim": "Walking",
            "actor_scale": actor_scale,
            "actor_z_lift_cm": 0.0,
            "walking_forward_yaw_offset_deg": 0.0,
            "animation_play_rate": 1.0,
            "ground_snap_to_floor": True,
            # Pixal animal assets retain a mesh-origin offset that scales with
            # the actor. Keep every correction in runtime evidence, but use a
            # scale-aware sanity limit so a valid large pug is not rejected by
            # a fixed threshold intended for a smaller cat.
            "ground_snap_max_abs_correction_cm": ground_snap_limit_cm,
            "audio_lookup": profile["audio_lookup"],
            "audio_source_height_offset_m": profile["audio_height_m"] * scale_ratio,
            "adaptive_repeat_short_calls": True,
            "strict_audio": True,
            "mute_audio": False,
        }
    )
    walking["camera_pass_table_loop_contract"]["animal_scale_rationale"] = {
        "profile_schema_id": attempt["profile_schema_id"],
        "base_actor_scale": profile["base_actor_scale"],
        "sampled_size": attempt["sampled_attributes"][physical["control_attribute"]],
        "physical_scale_ratio": scale_ratio,
        "actor_scale": actor_scale,
        "target_measurement": physical["measurement"],
        "target_value_cm": physical["target_value_cm"],
        "policy": "breed-calibrated UE scale multiplied by the sampled physical size ratio",
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
        f"Controlled {profile['breed']} {job['legacy_tag']} holds Idle at the visible "
        "camera-left/front waypoint without actor translation."
    )
    idle["trajectory_profile"] = "stationary_camera_left_front_v1"
    # Direction windows compare actor forward against translational motion and
    # therefore do not apply to a stationary Idle clip.  Omitting the field is
    # the renderer's explicit no-check contract; an empty list is invalid.
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
    ue_jobs: Path,
    ue_result: Path,
    lod_binding_batch: Path,
    animation_decision_batch: Path,
    template: Path,
    output_root: Path,
) -> Path:
    paths = [ue_jobs, ue_result, lod_binding_batch, animation_decision_batch, template]
    for path in paths:
        if path.resolve().is_symlink() or not path.resolve().is_file():
            raise contracts.ContractError(f"missing controlled animal input: {path}")
    jobs, lod_by_id, approved = _authenticate_inputs(
        ue_jobs.resolve(), ue_result.resolve(), lod_binding_batch.resolve(), animation_decision_batch.resolve()
    )
    template_payload = _load_json(template)
    output_root = output_root.absolute()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(f"refusing to replace output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent))
    try:
        records = []
        for job in jobs:
            asset_id = job["legacy_tag"]
            attempt = lod_by_id[asset_id]
            action_records = {}
            decision = approved[asset_id]
            controlled_gate = {
                "schema": "controlled_animal_apartment_gate_v1",
                "status": "approved_for_research_candidate_apartment",
                "asset_id": asset_id,
                "tag": job["tag"],
                "animation_decision": {
                    **_record(decision["path"]),
                    "decision_sha256": decision["payload"]["decision_sha256"],
                },
                "ue_import_result": _record(ue_result),
                "ue_source_sha256": job["rigged_glb_sha256"],
                "formal_dataset_registration_authorized": False,
            }
            for action, spec in build_pair(template_payload, job=job, attempt=attempt).items():
                spec["sources"][0]["controlled_animal_gate"] = copy.deepcopy(
                    controlled_gate
                )
                motion = "walking" if action == "Walking" else "idle"
                clip_name = f"camera_pass_table_loop_{motion}"
                relative_spec = Path("specs") / job["tag"] / f"{clip_name}.json"
                staged_spec = staging / relative_spec
                contracts.write_json_no_replace(staged_spec, spec)
                published_spec = output_root / relative_spec
                action_records[action] = {
                    "motion": motion,
                    "spec": str(published_spec),
                    "spec_evidence": _record(staged_spec, published_path=published_spec),
                    "output_dir": str(output_root / "clips" / job["tag"] / clip_name),
                    "clip_id": f"{job['tag']}_{clip_name}_v1",
                }
            records.append(
                {
                    "base_avatar_id": asset_id,
                    "asset_id": asset_id,
                    "tag": job["tag"],
                    "profile_schema_id": job["profile_schema_id"],
                    "sampled_attributes": copy.deepcopy(job["sampled_attributes"]),
                    "target_physical_profile": copy.deepcopy(attempt["target_physical_profile"]),
                    "source_glb": {"path": job["rigged_glb"], "sha256": job["rigged_glb_sha256"]},
                    "actions": action_records,
                }
            )
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "usage_scope": "research_candidate",
            "formal_registration_authorized": False,
            "trajectory_policy": "Walking uses camera right/rear -> left/front -> one table loop; Idle is stationary at left/front",
            "audio_policy": "species-matched short calls are energy-segmented and repeated with silent gaps",
            "physical_scale_policy": "breed base scale multiplied by immutable sampled target_physical_profile.scale_ratio",
            "avatar_count": len(records),
            "clip_count": len(records) * 2,
            "inputs": {
                "ue_import_jobs": _record(ue_jobs),
                "ue_import_result": _record(ue_result),
                "lod_binding_batch": _record(lod_binding_batch),
                "animation_decision_batch": _record(animation_decision_batch),
                "template": _record(template),
            },
            "records": records,
        }
        manifest["manifest_sha256"] = _hash_without(manifest, "manifest_sha256")
        contracts.write_json_no_replace(staging / "spec_manifest.json", manifest)
        os.rename(staging, output_root)
        return output_root / "spec_manifest.json"
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ue-jobs", required=True, type=Path)
    parser.add_argument("--ue-result", required=True, type=Path)
    parser.add_argument("--lod-binding-batch", required=True, type=Path)
    parser.add_argument("--animation-decisions", required=True, type=Path)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        manifest = build_specs(
            ue_jobs=args.ue_jobs,
            ue_result=args.ue_result,
            lod_binding_batch=args.lod_binding_batch,
            animation_decision_batch=args.animation_decisions,
            template=args.template,
            output_root=args.output_root,
        )
        payload = _load_json(manifest)
    except (contracts.ContractError, OSError, json.JSONDecodeError, KeyError, ValueError) as error:
        print(f"CONTROLLED_ANIMAL_APARTMENT_SPECS_FAILED {error}")
        return 2
    print(
        f"CONTROLLED_ANIMAL_APARTMENT_SPECS_OK animals={payload['avatar_count']} "
        f"clips={payload['clip_count']} output={manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
