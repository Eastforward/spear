import copy
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess

import pytest

from tools import build_controlled_source_asset_inputs as input_builder
from tools import controlled_source_asset_schema as contracts
from tools import prepare_controlled_source_asset_execution as execution_preparation
from tools import prepare_user_approved_generated_animal_ue_imports as preparation
from tools import register_controlled_animal_source_assets as source_registry


PROFILE = (
    Path(__file__).resolve().parents[2]
    / "data/controlled_source_attributes_v1/candidate_profiles/animal"
    / "dog_shiba_inu_four_limb_rest_side_clay_v1.json"
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path):
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _relative_record(path, root):
    record = _record(path)
    record["path"] = path.resolve().relative_to(root.resolve()).as_posix()
    return record


def _root_record(root_id, path, root):
    return {
        "root_id": root_id,
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _inject_duplicate_schema(path):
    payload = contracts.load_json(path)
    encoded = json.dumps(payload, ensure_ascii=False)
    duplicate = json.dumps(payload["schema"], ensure_ascii=False)
    path.write_text(
        f'{{"schema":{duplicate},' + encoded[1:] + "\n",
        encoding="utf-8",
    )


def _write_glb(path):
    document = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"mesh": 0, "skin": 0},
            {"name": "root"},
            {"name": "joint_1"},
            {"name": "joint_2"},
            {"name": "joint_3"},
            {"name": "joint_4"},
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "JOINTS_0": 1,
                            "WEIGHTS_0": 2,
                        }
                    }
                ]
            }
        ],
        "skins": [{"joints": [1, 2, 3, 4, 5], "inverseBindMatrices": 3}],
        "animations": [
            {
                "name": name,
                "samplers": [{"input": 4, "output": 5}],
                "channels": [
                    {
                        "sampler": 0,
                        "target": {"node": 1, "path": "rotation"},
                    }
                ],
            }
            for name in ("Idle", "Walking")
        ],
        "buffers": [{"byteLength": 512}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 512}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 1,
                "type": "VEC3",
            },
            {
                "bufferView": 0,
                "componentType": 5123,
                "count": 1,
                "type": "VEC4",
            },
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 1,
                "type": "VEC4",
            },
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 5,
                "type": "MAT4",
            },
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 1,
                "type": "SCALAR",
            },
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 1,
                "type": "VEC4",
            },
        ],
    }
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((4 - len(json_chunk) % 4) % 4)
    binary_chunk = bytes(512)
    total_length = 12 + 8 + len(json_chunk) + 8 + len(binary_chunk)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total_length)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
        + struct.pack("<II", len(binary_chunk), 0x004E4942)
        + binary_chunk
    )


def _write_fake_unskinned_glb(path):
    document = {
        "asset": {"version": "2.0"},
        "nodes": [{"mesh": 0, "skin": 0}],
        "meshes": [{"primitives": [{}]}],
        "skins": [{"joints": [0]}],
        "animations": [{"name": "Idle"}, {"name": "Walking"}],
    }
    payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
    payload += b" " * ((4 - len(payload) % 4) % 4)
    total_length = 12 + 8 + len(payload)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total_length)
        + struct.pack("<II", len(payload), 0x4E4F534A)
        + payload
    )


def _build_frozen_preflight(tmp_path):
    profiles = input_builder.load_profiles([PROFILE])
    roots = execution_preparation.default_artifact_roots()
    authentication = {
        profile["profile_schema_id"]: input_builder.authenticate_profile_artifacts(
            profile, roots
        )
        for profile in profiles
    }
    files = input_builder.compile_inputs(
        profiles=profiles,
        count_per_profile=1,
        seed=20260728,
        plan_id="approved_generated_animal_bridge_test_v1",
        split_salt="approved_generated_animal_bridge_test_v1",
        max_qa_pairs_per_split=None,
        artifact_authentication=authentication,
    )
    files["execution_jobs.json"] = input_builder.build_execution_jobs(
        files["instance_requests.json"],
        route_names={
            "flux2_pixal3d_animal_v1",
            "stable_animal_template_v1",
            "rocketbox_material_v1",
        },
    )
    input_dir = tmp_path / "inputs"
    input_builder.publish_output(input_dir, files)
    preflight = execution_preparation.build_execution_preflight(input_dir, roots)
    preflight["routes"].pop("flux2_pixal3d_static_v1")
    preflight["execution_summary"].pop("static_object_job_count")
    preflight["preflight_sha256"] = execution_preparation.preflight_sha256(preflight)
    preflight_path = tmp_path / "execution_preflight.json"
    contracts.write_json_no_replace(preflight_path, preflight)
    request = files["instance_requests.json"]["requests"][0]
    return preflight_path, preflight, profiles[0], request


def _make_video(path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=512x384:r=8",
            "-frames:v",
            str(preparation.REQUIRED_REVIEW_FRAMES),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def _build_media_lineage(root, *, input_glb, media):
    media_root = root / "05_review"
    media_root.mkdir(parents=True, exist_ok=True)
    lineage = {}
    for label, action, view, yaw in preparation.generated_review.REVIEW_MEDIA_SPECS:
        paths = preparation.generated_review.review_media_paths(
            {"review_root": media_root}, label
        )
        frame_dir = paths["frame_dir"]
        frame_dir.mkdir(exist_ok=True)
        frame_range = [1.0, 9.0]
        frames = []
        for index in range(preparation.REQUIRED_REVIEW_FRAMES):
            frame = frame_dir / f"frame_{index:04d}.png"
            frame.write_bytes(f"{label}-png-{index}".encode())
            fraction = index / (preparation.REQUIRED_REVIEW_FRAMES - 1)
            frames.append(
                {
                    "index": index,
                    "sample_fraction": fraction,
                    "source_action_frame": float(
                        int(
                            round(
                                frame_range[0]
                                + (frame_range[1] - frame_range[0]) * fraction
                            )
                        )
                    ),
                    "artifact": _record(frame),
                }
            )
        render_payload = {
            "schema": preparation.generated_review.RENDER_MANIFEST_SCHEMA,
            "status": "frames_rendered",
            "formal_dataset_registration_authorized": False,
            "input_glb": _record(input_glb),
            "request": {
                "action": action,
                "resolved_action": action,
                "rest_pose": False,
                "view": view,
                "asset_yaw_deg": yaw,
                "n_frames": preparation.REQUIRED_REVIEW_FRAMES,
                "resolution": {
                    "width": preparation.generated_review.REVIEW_MEDIA_WIDTH,
                    "height": preparation.generated_review.REVIEW_MEDIA_HEIGHT,
                },
                "fps": preparation.generated_review.REVIEW_MEDIA_FPS,
                "output_dir": str(frame_dir.resolve()),
            },
            "render_config": copy.deepcopy(
                preparation.generated_review.REVIEW_RENDER_CONFIG
            ),
            "action_frame_range": frame_range,
            "frames": frames,
        }
        _write_json(paths["render_manifest"], render_payload)
        encode_payload = {
            "schema": preparation.generated_review.ENCODE_MANIFEST_SCHEMA,
            "status": "video_encoded_and_probed",
            "formal_dataset_registration_authorized": False,
            "media_identity": {
                "label": label,
                "action": action,
                "view": view,
                "asset_yaw_deg": yaw,
            },
            "render_manifest": _record(paths["render_manifest"]),
            "frame_set": preparation.generated_review.render_frame_set(render_payload),
            "ffmpeg": preparation.generated_review.expected_review_ffmpeg_config(
                frame_dir,
                preparation.REQUIRED_REVIEW_FRAMES,
            ),
            "video": copy.deepcopy(media[label]),
        }
        _write_json(paths["encode_manifest"], encode_payload)
        lineage[label] = {
            "render_manifest": _record(paths["render_manifest"]),
            "encode_manifest": _record(paths["encode_manifest"]),
        }
    return lineage


def _fake_tokenrig_lineage(root, *, raw_pixal_glb, target_rig_glb):
    closure_root = root / "tokenrig_closure"
    closure_root.mkdir(exist_ok=True)
    closure_manifest = closure_root / "manifest.json"
    closure_manifest.write_text('{"fixture":"closure"}\n', encoding="utf-8")
    readback = closure_root / "geometry_readback.json"
    readback.write_text('{"fixture":"readback"}\n', encoding="utf-8")
    descriptor = {
        "schema": "avengine_generated_animal_tokenrig_closure_descriptor_v1",
        "status": "passed",
        "asset_id": "fixture_workspace",
        "evidence_mode": "complete_load_audit_v1",
        "closure_manifest": {
            **_record(closure_manifest),
            "manifest_sha256": "1" * 64,
        },
        "lineage": {
            "raw_pixal_glb": _record(raw_pixal_glb),
            "tokenrig_input": _record(raw_pixal_glb),
            "tokenrig_output": _record(target_rig_glb),
            "upstream_kind": "watertight_runtime_proxy",
        },
        "geometry_readback": {
            **_record(readback),
            "manifest_sha256": "2" * 64,
        },
        "execution": {
            "seed": 42,
            "exact_argv_sha256": "3" * 64,
            "model_checkpoint_sha256": "4" * 64,
            "model_snapshot_revision": "5" * 40,
            "runtime_patch_sha256": "6" * 64,
            "skintokens_revision": "7" * 40,
        },
        "formal_dataset_registration_authorized": False,
    }
    descriptor["descriptor_sha256"] = preparation._hash_without(
        descriptor, "descriptor_sha256"
    )
    return descriptor


@pytest.fixture
def approved_generated_animal(tmp_path, monkeypatch):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    candidate = artifact_root / "candidate.bin"
    candidate.write_bytes(b"authenticated source candidate")
    license_path = artifact_root / "LICENSE.txt"
    license_path.write_bytes(b"fixture research license")
    preflight_path, preflight, profile, request = _build_frozen_preflight(tmp_path)

    source_asset = contracts.build_source_asset_v2(
        request,
        artifacts={
            "pixal_raw_glb": _root_record("fixture_root", candidate, artifact_root)
        },
        physical_measurements={"status": "pending"},
        provenance={
            "attempt_id": "fixture_approved_generated_animal_v1",
            "request_sha256": request["request_sha256"],
            "models": copy.deepcopy(request["generation_plan"]["model_revisions"]),
        },
        rights={
            "status": "review_required",
            "licenses": [_root_record("fixture_root", license_path, artifact_root)],
            "blockers": ["research_candidate_only"],
        },
        qa={
            "reference_2d": "passed",
            "static_mesh": "passed",
            "binding": "pending",
            "walking": "pending",
            "idle": "pending",
            "ue_import_readback": "pending",
            "apartment_media": "pending",
            "audio": "pending",
        },
        state_classification="research_candidate",
    )
    registry_root = tmp_path / "source_registry"
    source_dir = registry_root / "source_assets"
    source_dir.mkdir(parents=True)
    source_path = source_dir / f"{source_asset['asset_id']}.json"
    contracts.write_json_no_replace(source_path, source_asset)
    attribute_evidence = {
        name: (
            "deferred_to_metric_3d"
            if name == source_asset["target_physical_profile"]["control_attribute"]
            else "passed_static_visual"
        )
        for name in source_asset["sampled_attributes"]
    }
    source_index = {
        "asset_id": source_asset["asset_id"],
        "profile_schema_id": source_asset["profile_schema_id"],
        "request_sha256": source_asset["request_sha256"],
        "sampled_attributes": copy.deepcopy(source_asset["sampled_attributes"]),
        "attribute_evidence": attribute_evidence,
        "source_asset": {
            "path": source_path.relative_to(registry_root).as_posix(),
            "sha256": _sha256(source_path),
            "size_bytes": source_path.stat().st_size,
        },
        "state_classification": "research_candidate",
        "next_gate": "lod_then_species_rig_binding",
    }
    pixal_inputs_path = tmp_path / "pixal_inputs_manifest.json"
    pixal_inputs = {
        "manifest_sha256": "5" * 64,
        "jobs": [{"fixture": source_asset["asset_id"]}],
    }
    contracts.write_json_no_replace(pixal_inputs_path, pixal_inputs)
    pixal_batch = {
        "schema": source_registry.pixal_runner.BATCH_SCHEMA,
        "status": "passed_generation_and_glb_readback",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "pixal_inputs": {
            "path": str(pixal_inputs_path.resolve()),
            "sha256": _sha256(pixal_inputs_path),
            "manifest_sha256": pixal_inputs["manifest_sha256"],
        },
        "automatic_checks": copy.deepcopy(preparation.PIXAL_BATCH_AUTOMATIC_CHECKS),
    }
    pixal_batch["batch_sha256"] = source_registry._hash_without(
        pixal_batch, "batch_sha256"
    )
    pixal_batch_path = tmp_path / "pixal_batch_manifest.json"
    contracts.write_json_no_replace(pixal_batch_path, pixal_batch)

    static_decision_batch = {
        "schema": source_registry.static_decisions.DECISION_BATCH_SCHEMA,
        "status": "completed",
        "automatic_checks": copy.deepcopy(
            preparation.STATIC_DECISION_BATCH_AUTOMATIC_CHECKS
        ),
    }
    static_decision_batch["decision_batch_sha256"] = source_registry._hash_without(
        static_decision_batch, "decision_batch_sha256"
    )
    static_decision_batch_path = tmp_path / "static_decision_batch_manifest.json"
    contracts.write_json_no_replace(static_decision_batch_path, static_decision_batch)

    input_jobs = {source_asset["asset_id"]: {"fixture": "input"}}
    attempts = {source_asset["asset_id"]: {"instance_id": source_asset["asset_id"]}}
    decisions = {
        source_asset["asset_id"]: {
            "payload": {
                "decision": "approved_for_lod_and_binding",
                "attribute_evidence": copy.deepcopy(attribute_evidence),
            }
        }
    }

    def load_pixal_inputs(path):
        assert path == pixal_inputs_path.resolve()
        return path, copy.deepcopy(pixal_inputs)

    def validate_pixal_request_identity(batch, manifest, requests):
        assert batch["batch_sha256"] == pixal_batch["batch_sha256"]
        assert manifest["manifest_sha256"] == pixal_inputs["manifest_sha256"]
        assert source_asset["asset_id"] in requests
        return copy.deepcopy(input_jobs), copy.deepcopy(attempts)

    def load_decision_batch(path):
        assert path == static_decision_batch_path.resolve()
        return path, copy.deepcopy(static_decision_batch), copy.deepcopy(decisions)

    monkeypatch.setattr(
        source_registry.pixal_runner, "load_pixal_inputs", load_pixal_inputs
    )
    monkeypatch.setattr(
        source_registry,
        "validate_pixal_request_identity",
        validate_pixal_request_identity,
    )
    monkeypatch.setattr(source_registry, "load_decision_batch", load_decision_batch)

    registry = {
        "schema": source_registry.REGISTRY_SCHEMA,
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "preflight": {
            "path": str(preflight_path.resolve()),
            "sha256": _sha256(preflight_path),
            "preflight_sha256": preflight["preflight_sha256"],
            "validation_mode": "frozen_historical_preflight_v1",
        },
        "pixal_batch": {
            "path": str(pixal_batch_path.resolve()),
            "sha256": _sha256(pixal_batch_path),
            "batch_sha256": pixal_batch["batch_sha256"],
        },
        "static_decision_batch": {
            "path": str(static_decision_batch_path.resolve()),
            "sha256": _sha256(static_decision_batch_path),
            "decision_batch_sha256": static_decision_batch["decision_batch_sha256"],
        },
        "source_asset_count": 1,
        "source_assets": [source_index],
        "automatic_checks": copy.deepcopy(preparation.REGISTRY_AUTOMATIC_CHECKS),
    }
    registry["registry_sha256"] = source_registry._hash_without(
        registry, "registry_sha256"
    )
    registry_path = registry_root / "registry_manifest.json"
    contracts.write_json_no_replace(registry_path, registry)

    animated_glb = tmp_path / "target_animated.glb"
    _write_glb(animated_glb)
    input_glb = tmp_path / "target_rig.glb"
    _write_glb(input_glb)
    target_rig_lineage = _fake_tokenrig_lineage(
        tmp_path,
        raw_pixal_glb=candidate,
        target_rig_glb=input_glb,
    )
    heading_evidence = tmp_path / "forward_declaration.json"
    heading_evidence.write_text('{"fixture":true}\n', encoding="utf-8")
    source_motion = tmp_path / "source_motion.glb"
    _write_glb(source_motion)
    stage_records = {}
    for name in preparation.REQUIRED_REVIEW_OUTPUTS - {
        "animated_glb",
        "retargeted_animated_glb",
    }:
        evidence = tmp_path / f"{name}.json"
        evidence.write_text(
            json.dumps({"status": "passed", "name": name}), encoding="utf-8"
        )
        stage_records[name] = _record(evidence)
    stage_records["animated_glb"] = _record(animated_glb)
    stage_records["retargeted_animated_glb"] = _record(animated_glb)
    stage_records["gait_direction_audit"] = copy.deepcopy(
        stage_records["gait_direction_audit_initial"]
    )
    stage_records["deformation_audit"] = copy.deepcopy(
        stage_records["deformation_audit_initial"]
    )
    stage_records["gait_direction_status"] = "pass"
    stage_records["deformation_overall"] = "passed"

    review_media_root = tmp_path / "05_review"
    review_media_root.mkdir()
    first_video = review_media_root / "walking_side.mp4"
    _make_video(first_video)
    media = {}
    for name in preparation.REQUIRED_MEDIA:
        path = review_media_root / f"{name}.mp4"
        if path != first_video:
            shutil.copyfile(first_video, path)
        media[name] = preparation.generated_review.verify_video(
            path, preparation.REQUIRED_REVIEW_FRAMES
        )
    stage_records["media"] = media
    stage_records["media_lineage"] = _build_media_lineage(
        tmp_path,
        input_glb=animated_glb,
        media=media,
    )

    pipeline = preparation._review_pipeline_order(repair_branch="not_needed")
    review = {
        "schema": preparation.BRANCHED_GENERATED_REVIEW_SCHEMA,
        "created_at": "2026-07-28T00:00:00+00:00",
        "status": "research_candidate_pending_human_review",
        "formal_dataset_registration_authorized": False,
        "forward_contract": {},
        "pipeline_order": pipeline,
        "automatic_admission_gates": {
            "heading": "positive-x",
            "rig": "passed",
            "support_plane": "mesh-foot-bottoms",
            "retarget_export_front_axis": "positive-x",
            "gait_initial": "pass",
            "deformation_initial": "passed",
            "weight_repair_policy": "auto",
            "weight_repair_triggered": False,
            "weight_repair": "not_needed",
            "weight_repair_strategy": "not_needed",
            "weight_repair_branch": "not_needed",
            "weight_repair_attempts": [],
            "weight_repair_final_artifact": (
                preparation.generated_review.weight_repair_final_artifact("not_needed")
            ),
            "gait_final": "pass",
            "deformation_final": "passed",
            "all_automatic_gates_passed": True,
        },
        "inputs": {
            "target_rig_glb": _record(input_glb),
            "target_rig_lineage": target_rig_lineage,
            "heading_review_evidence": _record(heading_evidence),
            "source_motion_glb": _record(source_motion),
        },
        "outputs": stage_records,
        "timings_seconds": {name: 0.1 for name in pipeline},
    }
    review_path = tmp_path / "review_run.json"
    contracts.write_json_no_replace(review_path, review)

    decision = {
        "schema": preparation.DECISION_SCHEMA,
        "asset_id": source_asset["asset_id"],
        "review_sha256": _sha256(review_path),
        "decision": "approved_for_ue_apartment",
        "checks": {name: True for name in preparation.DECISION_CHECK_FIELDS},
        "caveats": [],
        "notes": "User approved all authenticated Walk and Idle views.",
        "review": _record(review_path),
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "next_gate": "ue_import_metric_trajectory_audio_and_apartment_media",
    }
    decision["decision_sha256"] = preparation._hash_without(decision, "decision_sha256")
    decision_path = tmp_path / "animation_decision.json"
    contracts.write_json_no_replace(decision_path, decision)

    def fixture_lineage(**kwargs):
        return kwargs["authenticated"]["output:animated_glb"]

    monkeypatch.setattr(preparation, "_validate_review_lineage", fixture_lineage)
    fixture = {
        "artifact_root": artifact_root,
        "registry_path": registry_path,
        "source_asset": source_asset,
        "source_path": source_path,
        "review_path": review_path,
        "decision_path": decision_path,
        "receipt_path": tmp_path / "decision_freeze_receipt.json",
        "media_path": Path(next(iter(media.values()))["path"]),
    }
    _rewrite_freeze_receipt(fixture)
    return fixture


def _prepare(
    fixture,
    output,
    *,
    expected_registry_sha256=None,
    expected_decision_sha256=None,
    expected_receipt_sha256=None,
):
    return preparation.prepare_import(
        source_registry_manifest_path=fixture["registry_path"],
        expected_source_registry_sha256=(
            expected_registry_sha256 or _sha256(fixture["registry_path"])
        ),
        source_asset_path=fixture["source_path"],
        animation_review_path=fixture["review_path"],
        animation_decision_path=fixture["decision_path"],
        expected_animation_decision_sha256=(
            expected_decision_sha256 or _sha256(fixture["decision_path"])
        ),
        animation_decision_freeze_receipt_path=fixture["receipt_path"],
        expected_animation_decision_freeze_receipt_sha256=(
            expected_receipt_sha256 or _sha256(fixture["receipt_path"])
        ),
        output_root=output,
        artifact_roots={"fixture_root": fixture["artifact_root"]},
    )


def _rewrite_review_and_rebind_decision(fixture, review):
    _write_json(fixture["review_path"], review)
    decision = contracts.load_json(fixture["decision_path"])
    decision["review"] = _record(fixture["review_path"])
    decision["review_sha256"] = decision["review"]["sha256"]
    decision["decision_sha256"] = preparation._hash_without(decision, "decision_sha256")
    _write_json(fixture["decision_path"], decision)
    _rewrite_freeze_receipt(fixture, review=review)


def _rewrite_freeze_receipt(fixture, *, review=None):
    registry = contracts.load_json(fixture["registry_path"])
    if review is None:
        review = contracts.load_json(fixture["review_path"])
    decision = contracts.load_json(fixture["decision_path"])
    gates = review["automatic_admission_gates"]
    legacy = review["schema"] == preparation.LEGACY_GENERATED_REVIEW_SCHEMA
    try:
        branch = preparation._repair_branch_contract(
            gates,
            review_schema=review["schema"],
        )[1]
        descriptor_names = set(preparation.REQUIRED_REVIEW_OUTPUTS)
        descriptor_names.update(
            preparation._repair_output_names(legacy=legacy, branch=branch)
        )
        review_artifact_count = (
            len(preparation.REVIEW_FILE_INPUTS)
            + len(descriptor_names)
            + len(preparation.REQUIRED_MEDIA)
            + (2 * len(preparation.REQUIRED_MEDIA) if not legacy else 0)
        )
    except contracts.ContractError:
        if not fixture["receipt_path"].is_file():
            raise
        review_artifact_count = contracts.load_json(fixture["receipt_path"])[
            "authenticated_review_artifact_count"
        ]
    receipt = {
        "schema": preparation.DECISION_FREEZE_RECEIPT_SCHEMA,
        "status": "frozen",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "source_asset_registry": _record(fixture["registry_path"]),
        "expected_source_asset_registry_file_sha256": _sha256(fixture["registry_path"]),
        "source_asset_registry_validation_mode": registry["preflight"][
            "validation_mode"
        ],
        "source_asset": _record(fixture["source_path"]),
        "animation_review": _record(fixture["review_path"]),
        "expected_animation_review_file_sha256": _sha256(fixture["review_path"]),
        "user_instruction_binding": {
            "decision": "approved_for_ue_apartment",
            "review_sha256": _sha256(fixture["review_path"]),
            "all_six_checks_explicit": True,
        },
        "user_instruction_authority": copy.deepcopy(
            preparation.USER_INSTRUCTION_AUTHORITY
        ),
        "authenticated_review_artifact_count": review_artifact_count,
        "animation_decision": _relative_record(
            fixture["decision_path"],
            fixture["receipt_path"].parent,
        ),
        "decision_sha256": decision["decision_sha256"],
    }
    receipt["receipt_sha256"] = preparation._hash_without(
        receipt,
        "receipt_sha256",
    )
    _write_json(fixture["receipt_path"], receipt)


def _reauthenticate_registry_source(fixture):
    registry = contracts.load_json(fixture["registry_path"])
    descriptor = registry["source_assets"][0]["source_asset"]
    descriptor["sha256"] = _sha256(fixture["source_path"])
    descriptor["size_bytes"] = fixture["source_path"].stat().st_size
    registry["registry_sha256"] = source_registry._hash_without(
        registry, "registry_sha256"
    )
    _write_json(fixture["registry_path"], registry)
    _rewrite_freeze_receipt(fixture)


def _write_weight_repair_manifest(
    path,
    *,
    stage,
    input_glb,
    output_glb,
    status,
):
    parameters = copy.deepcopy(preparation._weight_repair_parameters(stage))
    authority = {
        name: True for name in preparation.generated_review.REPAIR_AUTHORITY_FIELDS
    }
    authority.update(
        {
            "source_animation_fingerprints": {"Walking": "fixture"},
            "post_repair_animation_fingerprints": {"Walking": "fixture"},
            "source_animation_curve_stats": {"Walking": {"curves": 1}},
            "post_repair_animation_curve_stats": {"Walking": {"curves": 1}},
            "rest_geometry_topology_fingerprint_before": "fixture-topology",
            "rest_geometry_topology_fingerprint_after": "fixture-topology",
            "maximum_rest_geometry_delta": 0.0,
            "maximum_allowed_rest_geometry_delta": 0.0,
        }
    )
    ready = status == preparation.generated_review.WEIGHT_REPAIR_READY_STATUS
    payload = {
        "schema": preparation.generated_review.WEIGHT_REPAIR_SCHEMA,
        "status": status,
        "input": _record(input_glb),
        "output": _record(output_glb),
        "front_axis": "positive-x",
        "parameters": parameters,
        "formal_dataset_registration_authorized": False,
        "authority_contract": authority,
        "final_measurements": {
            "maximum_extension_ratio_of_rest_diagonal": (0.019 if ready else 0.03),
            "remaining_seed_edge_count": 0 if ready else 2,
        },
    }
    if parameters["cross_limb_authority"] == "low-slice-components":
        chains = (
            "front_side_negative",
            "front_side_positive",
            "hind_side_negative",
            "hind_side_positive",
        )
        payload["cross_limb_preclean"] = {
            "entries_after": 0,
            "weight_mass_after": 0,
            "final_forbidden_entries": 0,
            "final_forbidden_weight_mass": 0,
            "skipped_for_residual_pass": parameters["skip_cross_limb_preclean"],
            "spatial_authority": {
                "method": "four_largest_disconnected_low_slice_components",
                "immutable_through_motion_repair": True,
                "blender_up_axis": "positive-z",
                "height_fraction": parameters["limb_slice_height_fraction"],
                "substantial_component_sizes_descending": [4, 3, 2, 1],
                "assignment_counts": {name: 1 for name in chains},
                "components": [{"semantic_chain": name} for name in chains],
            },
        }
    _write_json(path, payload)
    return payload


def _build_weight_repair_lineage(tmp_path, branch):
    retargeted = tmp_path / "retargeted.glb"
    _write_glb(retargeted)
    statuses = {
        "primary": {
            "primary": preparation.generated_review.WEIGHT_REPAIR_READY_STATUS,
        },
        "fallback_a": {
            "primary": preparation.generated_review.WEIGHT_REPAIR_INCOMPLETE_STATUS,
            "fallback_a": preparation.generated_review.WEIGHT_REPAIR_READY_STATUS,
        },
        "fallback_a_b": {
            "primary": preparation.generated_review.WEIGHT_REPAIR_INCOMPLETE_STATUS,
            "fallback_a": preparation.generated_review.WEIGHT_REPAIR_INCOMPLETE_STATUS,
            "fallback_b": preparation.generated_review.WEIGHT_REPAIR_READY_STATUS,
        },
    }[branch]
    outputs = {}
    authenticated = {}
    attempts = []
    payloads = {}
    stage_outputs = {}
    stage_manifests = {}
    for stage in preparation.generated_review.WEIGHT_REPAIR_BRANCH_STAGES[branch]:
        contract = preparation.generated_review.WEIGHT_REPAIR_STAGE_CONTRACTS[stage]
        output_path = tmp_path / f"{stage}.glb"
        _write_glb(output_path)
        manifest_path = tmp_path / f"{stage}.json"
        input_path = (
            stage_outputs["fallback_a"] if stage == "fallback_b" else retargeted
        )
        payload = _write_weight_repair_manifest(
            manifest_path,
            stage=stage,
            input_glb=input_path,
            output_glb=output_path,
            status=statuses[stage],
        )
        output_name = contract["output_glb_output_descriptor"]
        manifest_name = contract["manifest_output_descriptor"]
        outputs[output_name] = _record(output_path)
        outputs[manifest_name] = _record(manifest_path)
        authenticated[f"output:{output_name}"] = output_path.resolve()
        authenticated[f"output:{manifest_name}"] = manifest_path.resolve()
        stage_outputs[stage] = output_path
        stage_manifests[stage] = manifest_path
        payloads[stage] = payload
        attempts.append(
            preparation.generated_review.weight_repair_attempt_record(
                stage,
                payload,
                manifest_path,
                output_path,
            )
        )
    final_artifact = preparation.generated_review.weight_repair_final_artifact(branch)
    final_stage = preparation.generated_review.WEIGHT_REPAIR_BRANCH_FINAL_STAGE[branch]
    final_glb = stage_outputs[final_stage]
    final_manifest = stage_manifests[final_stage]
    outputs["animated_glb"] = copy.deepcopy(
        outputs[final_artifact["glb_output_descriptor"]]
    )
    outputs["weight_repair_manifest"] = copy.deepcopy(
        outputs[final_artifact["manifest_output_descriptor"]]
    )
    authenticated["output:animated_glb"] = final_glb.resolve()
    authenticated["output:weight_repair_manifest"] = final_manifest.resolve()
    gates = {
        "heading": "positive-x",
        "rig": "passed",
        "support_plane": "mesh-foot-bottoms",
        "retarget_export_front_axis": "positive-x",
        "gait_initial": "pass",
        "deformation_initial": "rejected",
        "weight_repair_policy": "auto",
        "weight_repair_triggered": True,
        "weight_repair": preparation.generated_review.WEIGHT_REPAIR_READY_STATUS,
        "weight_repair_strategy": (
            preparation.generated_review.WEIGHT_REPAIR_BRANCH_STRATEGY[branch]
        ),
        "weight_repair_branch": branch,
        "weight_repair_attempts": (
            preparation.generated_review.weight_repair_gate_attempts(attempts)
        ),
        "weight_repair_final_artifact": final_artifact,
        "gait_final": "pass",
        "deformation_final": "passed",
        "all_automatic_gates_passed": True,
    }
    return {
        "gates": gates,
        "outputs": outputs,
        "authenticated": authenticated,
        "retargeted": retargeted.resolve(),
        "final_glb": final_glb.resolve(),
        "payloads": payloads,
        "stage_outputs": stage_outputs,
        "stage_manifests": stage_manifests,
    }


def _validate_weight_repair_fixture(fixture):
    legacy, branch = preparation._repair_branch_contract(
        fixture["gates"],
        review_schema=preparation.BRANCHED_GENERATED_REVIEW_SCHEMA,
    )
    preparation._validate_weight_repair_lineage(
        gates=fixture["gates"],
        outputs=fixture["outputs"],
        authenticated=fixture["authenticated"],
        retargeted_glb=fixture["retargeted"],
        final_glb=fixture["final_glb"],
        legacy=legacy,
        branch=branch,
    )


def _v4_not_needed_review(fixture):
    review = contracts.load_json(fixture["review_path"])
    review["schema"] = preparation.BRANCHED_GENERATED_REVIEW_SCHEMA
    review["automatic_admission_gates"].update(
        {
            "weight_repair_strategy": "not_needed",
            "weight_repair_branch": "not_needed",
            "weight_repair_attempts": [],
            "weight_repair_final_artifact": (
                preparation.generated_review.weight_repair_final_artifact("not_needed")
            ),
        }
    )
    review["outputs"]["media_lineage"] = _build_media_lineage(
        fixture["review_path"].parent,
        input_glb=Path(review["outputs"]["animated_glb"]["path"]),
        media=review["outputs"]["media"],
    )
    return review


def test_prepares_fresh_canonical_job_and_does_not_rewrite_old_job(
    approved_generated_animal, tmp_path
):
    old_job = tmp_path / "historical_ue_import_jobs.json"
    old_bytes = b'{"sampled_attributes":{"coat_tone":"red"}}\n'
    old_job.write_bytes(old_bytes)

    manifest_path = _prepare(
        approved_generated_animal, tmp_path / "new_canonical_ue_import"
    )

    manifest = contracts.load_json(manifest_path)
    jobs = contracts.load_json(manifest_path.parent / "ue_import_jobs.json")
    source = approved_generated_animal["source_asset"]
    job = jobs["jobs"][0]
    assert set(jobs) == {
        "schema",
        "status",
        "state_classification",
        "formal_dataset_registration_authorized",
        "job_type",
        "job_count",
        "jobs",
        "non_destructive_policy",
        "batch_sha256",
    }
    assert jobs["schema"] == preparation.IMPORT_SCHEMA
    assert jobs["status"] == "ready_for_new_ue_import"
    assert jobs["state_classification"] == "research_candidate"
    assert jobs["formal_dataset_registration_authorized"] is False
    assert jobs["job_type"] == preparation.IMPORT_JOB_TYPE
    assert jobs["job_count"] == 1
    assert jobs["batch_sha256"] == preparation._hash_without(jobs, "batch_sha256")
    assert set(job) == {
        "job_type",
        "asset_id",
        "legacy_tag",
        "tag",
        "profile_schema_id",
        "sampled_attributes",
        "expected_actions",
        "rigged_glb",
        "rigged_glb_sha256",
        "source_registry_sha256",
        "source_asset_sha256",
        "request_sha256",
        "animation_decision_file_sha256",
        "animation_decision_sha256",
    }
    assert job["job_type"] == preparation.IMPORT_JOB_TYPE
    assert job["legacy_tag"] == source["asset_id"]
    assert job["expected_actions"] == ["Idle", "Walking"]
    assert manifest["automatic_checks"]["overall"] == "passed"
    assert manifest["automatic_checks"]["source_registry_and_preflight_reauthenticated"]
    assert manifest["automatic_checks"][
        "source_registry_matched_external_expected_sha256"
    ]
    assert manifest["automatic_checks"][
        "human_animation_approval_matched_external_expected_sha256"
    ]
    assert manifest["schema"] == preparation.SCHEMA
    assert manifest["animation_decision_freeze_receipt"] == _record(
        approved_generated_animal["receipt_path"]
    )
    assert manifest[
        "expected_animation_decision_freeze_receipt_file_sha256"
    ] == _sha256(approved_generated_animal["receipt_path"])
    assert manifest["user_instruction_authority"] == (
        preparation.USER_INSTRUCTION_AUTHORITY
    )
    assert (
        manifest["user_instruction_authority"]["cryptographic_user_identity_verified"]
        is False
    )
    assert manifest["automatic_checks"][
        "animation_decision_freeze_receipt_reauthenticated"
    ]
    assert manifest["automatic_checks"][
        "user_instruction_authority_preserved_without_cryptographic_upgrade"
    ]
    assert set(manifest["ue_import_jobs"]) == {"path", "sha256", "size_bytes"}
    assert manifest["ue_import_jobs"]["path"] == "ue_import_jobs.json"
    assert manifest["ue_import_jobs"]["sha256"] == _sha256(
        manifest_path.parent / "ue_import_jobs.json"
    )
    assert job["asset_id"] == source["asset_id"]
    assert job["tag"] == f"pixal_{source['asset_id']}"
    assert job["profile_schema_id"] == source["profile_schema_id"]
    assert job["sampled_attributes"] == source["sampled_attributes"]
    assert job["request_sha256"] == source["request_sha256"]
    assert old_job.read_bytes() == old_bytes


def test_rejects_source_registry_without_external_expected_file_hash(
    approved_generated_animal, tmp_path
):
    with pytest.raises(contracts.ContractError, match="externally expected SHA-256"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_unanchored_registry",
            expected_registry_sha256="f" * 64,
        )


def test_rejects_vacuous_source_registry_automatic_checks(
    approved_generated_animal, tmp_path
):
    registry = contracts.load_json(approved_generated_animal["registry_path"])
    registry["automatic_checks"] = {"overall": "passed"}
    registry["registry_sha256"] = source_registry._hash_without(
        registry, "registry_sha256"
    )
    _write_json(approved_generated_animal["registry_path"], registry)

    with pytest.raises(contracts.ContractError, match="automatic checks are invalid"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_vacuous_registry_checks",
        )


@pytest.mark.parametrize(
    ("authority_name", "internal_sha_name"),
    (
        ("pixal_batch", "batch_sha256"),
        ("static_decision_batch", "decision_batch_sha256"),
    ),
)
def test_rejects_missing_rebound_source_registry_authority(
    approved_generated_animal,
    tmp_path,
    authority_name,
    internal_sha_name,
):
    registry = contracts.load_json(approved_generated_animal["registry_path"])
    registry[authority_name] = {
        "path": str((tmp_path / f"missing_{authority_name}.json").resolve()),
        "sha256": "a" * 64,
        internal_sha_name: "b" * 64,
    }
    registry["registry_sha256"] = source_registry._hash_without(
        registry, "registry_sha256"
    )
    _write_json(approved_generated_animal["registry_path"], registry)

    with pytest.raises(contracts.ContractError, match="missing or unsafe"):
        _prepare(
            approved_generated_animal,
            tmp_path / f"rejected_missing_{authority_name}",
        )


@pytest.mark.parametrize(
    ("authority_name", "internal_sha_name"),
    (
        ("pixal_batch", "batch_sha256"),
        ("static_decision_batch", "decision_batch_sha256"),
    ),
)
def test_rejects_rebound_source_registry_internal_authority_hash(
    approved_generated_animal,
    tmp_path,
    authority_name,
    internal_sha_name,
):
    registry = contracts.load_json(approved_generated_animal["registry_path"])
    registry[authority_name][internal_sha_name] = "b" * 64
    registry["registry_sha256"] = source_registry._hash_without(
        registry, "registry_sha256"
    )
    _write_json(approved_generated_animal["registry_path"], registry)

    with pytest.raises(contracts.ContractError, match="contract/hash is invalid"):
        _prepare(
            approved_generated_animal,
            tmp_path / f"rejected_internal_{authority_name}",
        )


@pytest.mark.parametrize("branch", ("primary", "fallback_a", "fallback_a_b"))
def test_accepts_exact_authenticated_weight_repair_branch(tmp_path, branch):
    fixture = _build_weight_repair_lineage(tmp_path, branch)

    _validate_weight_repair_fixture(fixture)


def test_target_rig_lineage_binds_source_raw_pixal_and_exact_target(
    tmp_path, monkeypatch
):
    spear_root = tmp_path / "SPEAR"
    workspace = spear_root / "tmp/new_animal_assets/fixture_workspace"
    workspace.mkdir(parents=True)
    raw_pixal = workspace / "pixal_raw.glb"
    raw_pixal.write_bytes(b"raw Pixal geometry")
    target_rig = workspace / "tokenrig_output.glb"
    target_rig.write_bytes(b"TokenRig output")
    descriptor = _fake_tokenrig_lineage(
        tmp_path,
        raw_pixal_glb=raw_pixal,
        target_rig_glb=target_rig,
    )

    def validate(path, *, expected_manifest_sha256, expected_target_rig_glb):
        assert path == Path(descriptor["closure_manifest"]["path"])
        assert expected_manifest_sha256 == descriptor["closure_manifest"]["sha256"]
        assert expected_target_rig_glb == target_rig
        return copy.deepcopy(descriptor)

    monkeypatch.setattr(preparation, "SPEAR_ROOT", spear_root)
    monkeypatch.setattr(preparation, "validate_tokenrig_closure_manifest", validate)
    authenticated = {}

    observed = preparation._validate_target_rig_lineage(
        descriptor,
        target_rig_glb=target_rig,
        source_asset={"asset_class": "animal"},
        source_artifacts={"artifact:pixal_raw_glb": raw_pixal},
        authenticated=authenticated,
    )

    assert observed == descriptor
    assert set(authenticated) == {
        "target_rig_lineage:closure_manifest",
        "target_rig_lineage:geometry_readback",
    }


def test_target_rig_lineage_rejects_rebound_raw_pixal(tmp_path, monkeypatch):
    spear_root = tmp_path / "SPEAR"
    workspace = spear_root / "tmp/new_animal_assets/fixture_workspace"
    workspace.mkdir(parents=True)
    canonical_raw = workspace / "canonical_raw.glb"
    canonical_raw.write_bytes(b"canonical raw Pixal geometry")
    rebound_raw = workspace / "rebound_raw.glb"
    rebound_raw.write_bytes(b"other raw Pixal geometry")
    target_rig = workspace / "tokenrig_output.glb"
    target_rig.write_bytes(b"TokenRig output")
    descriptor = _fake_tokenrig_lineage(
        tmp_path,
        raw_pixal_glb=rebound_raw,
        target_rig_glb=target_rig,
    )
    monkeypatch.setattr(preparation, "SPEAR_ROOT", spear_root)
    monkeypatch.setattr(
        preparation,
        "validate_tokenrig_closure_manifest",
        lambda *args, **kwargs: copy.deepcopy(descriptor),
    )

    with pytest.raises(contracts.ContractError, match="raw Pixal source rejected"):
        preparation._validate_target_rig_lineage(
            descriptor,
            target_rig_glb=target_rig,
            source_asset={"asset_class": "animal"},
            source_artifacts={"artifact:pixal_raw_glb": canonical_raw},
            authenticated={},
        )


@pytest.mark.parametrize("branch", ("primary", "fallback_a", "fallback_a_b"))
def test_v4_review_accepts_only_exact_branch_pipeline_and_outputs(
    approved_generated_animal,
    tmp_path,
    branch,
):
    repair_root = tmp_path / f"repair_{branch}"
    repair_root.mkdir()
    repair = _build_weight_repair_lineage(repair_root, branch)
    review = contracts.load_json(approved_generated_animal["review_path"])
    review["schema"] = preparation.BRANCHED_GENERATED_REVIEW_SCHEMA
    review["automatic_admission_gates"] = repair["gates"]
    review["pipeline_order"] = preparation._review_pipeline_order(repair_branch=branch)
    review["timings_seconds"] = {name: 0.1 for name in review["pipeline_order"]}
    review["outputs"].update(repair["outputs"])
    review["outputs"]["media_lineage"] = _build_media_lineage(
        approved_generated_animal["review_path"].parent,
        input_glb=repair["final_glb"],
        media=review["outputs"]["media"],
    )
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    manifest_path = _prepare(
        approved_generated_animal,
        tmp_path / f"accepted_v4_{branch}",
    )

    manifest = contracts.load_json(manifest_path)
    assert manifest["animation_review_schema"] == (
        preparation.BRANCHED_GENERATED_REVIEW_SCHEMA
    )
    assert manifest["reviewed_animated_glb"]["sha256"] == _sha256(repair["final_glb"])


def test_primary_branch_rejects_compatibility_manifest_not_equal_to_final(
    tmp_path,
):
    fixture = _build_weight_repair_lineage(tmp_path, "primary")
    unrelated = tmp_path / "unrelated_manifest.json"
    unrelated.write_text('{"unrelated":true}\n', encoding="utf-8")
    fixture["outputs"]["weight_repair_manifest"] = _record(unrelated)
    fixture["authenticated"]["output:weight_repair_manifest"] = unrelated.resolve()

    with pytest.raises(contracts.ContractError, match="final weight-repair lineage"):
        _validate_weight_repair_fixture(fixture)


def test_fallback_a_branch_rejects_a_input_other_than_retarget(tmp_path):
    fixture = _build_weight_repair_lineage(tmp_path, "fallback_a")
    payload = fixture["payloads"]["fallback_a"]
    payload["input"] = _record(fixture["stage_outputs"]["primary"])
    _write_json(fixture["stage_manifests"]["fallback_a"], payload)

    with pytest.raises(
        contracts.ContractError, match="fallback_a weight repair manifest rejected"
    ):
        _validate_weight_repair_fixture(fixture)


def test_fallback_a_b_branch_rejects_b_input_other_than_a_output(tmp_path):
    fixture = _build_weight_repair_lineage(tmp_path, "fallback_a_b")
    payload = fixture["payloads"]["fallback_b"]
    payload["input"] = _record(fixture["retargeted"])
    _write_json(fixture["stage_manifests"]["fallback_b"], payload)

    with pytest.raises(
        contracts.ContractError, match="fallback_b weight repair manifest rejected"
    ):
        _validate_weight_repair_fixture(fixture)


def test_rejects_v3_review_with_v4_repair_fields(approved_generated_animal, tmp_path):
    review = contracts.load_json(approved_generated_animal["review_path"])
    review["schema"] = preparation.LEGACY_GENERATED_REVIEW_SCHEMA
    review["automatic_admission_gates"].update(
        {
            "weight_repair_strategy": "not_needed",
            "weight_repair_branch": "not_needed",
            "weight_repair_attempts": [],
            "weight_repair_final_artifact": (
                preparation.generated_review.weight_repair_final_artifact("not_needed")
            ),
        }
    )
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(contracts.ContractError, match="legacy v3.*fields"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_v3_v4_field_union",
        )


def test_rejects_v4_review_with_legacy_repair_fields(
    approved_generated_animal, tmp_path
):
    review = contracts.load_json(approved_generated_animal["review_path"])
    review["schema"] = preparation.BRANCHED_GENERATED_REVIEW_SCHEMA
    for name in (
        "weight_repair_strategy",
        "weight_repair_branch",
        "weight_repair_attempts",
        "weight_repair_final_artifact",
    ):
        review["automatic_admission_gates"].pop(name)
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(contracts.ContractError, match="branched v4.*fields"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_v4_legacy_fields",
        )


def test_legacy_v3_review_is_audit_only_and_cannot_publish_v2_batch(
    approved_generated_animal, tmp_path
):
    review = contracts.load_json(approved_generated_animal["review_path"])
    review["schema"] = preparation.LEGACY_GENERATED_REVIEW_SCHEMA
    for name in (
        "weight_repair_strategy",
        "weight_repair_branch",
        "weight_repair_attempts",
        "weight_repair_final_artifact",
    ):
        review["automatic_admission_gates"].pop(name)
    review["outputs"].pop("media_lineage")
    review["inputs"].pop("target_rig_lineage")
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(contracts.ContractError, match="legacy v3 is audit-only"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_v3_prepare",
        )


def test_v4_not_needed_branch_rejects_unrun_repair_output(
    approved_generated_animal, tmp_path
):
    review = _v4_not_needed_review(approved_generated_animal)
    review["outputs"]["weight_repair_primary_glb"] = copy.deepcopy(
        review["outputs"]["retargeted_animated_glb"]
    )
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(contracts.ContractError, match="review I/O is invalid"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_unrun_repair_output",
        )


def test_v4_review_rejects_missing_media_lineage(approved_generated_animal, tmp_path):
    review = _v4_not_needed_review(approved_generated_animal)
    review["outputs"].pop("media_lineage")
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(contracts.ContractError, match="review I/O is invalid"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_missing_media_lineage",
        )


def test_v4_review_rejects_tampered_render_frame_artifact(
    approved_generated_animal, tmp_path
):
    review = _v4_not_needed_review(approved_generated_animal)
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)
    frame = (
        approved_generated_animal["review_path"].parent
        / "05_review/walking_side_frames/frame_0003.png"
    )
    frame.write_bytes(b"tampered render frame")

    with pytest.raises(
        contracts.ContractError,
        match="review render/encode media lineage rejected",
    ):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_tampered_render_frame",
        )


def test_v4_review_rejects_swapped_encode_media_identity(
    approved_generated_animal, tmp_path
):
    review = _v4_not_needed_review(approved_generated_animal)
    encode_manifest = (
        approved_generated_animal["review_path"].parent
        / "05_review/walking_side_encode_manifest.json"
    )
    encode = contracts.load_json(encode_manifest)
    encode["media_identity"]["action"] = "Idle"
    _write_json(encode_manifest, encode)
    review["outputs"]["media_lineage"]["walking_side"]["encode_manifest"] = _record(
        encode_manifest
    )
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(
        contracts.ContractError,
        match="review render/encode media lineage rejected",
    ):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_swapped_encode_identity",
        )


def test_rejects_empty_required_gate_descriptor(approved_generated_animal, tmp_path):
    review = contracts.load_json(approved_generated_animal["review_path"])
    review["outputs"]["support_plane_manifest"] = {}
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(contracts.ContractError, match="descriptor"):
        _prepare(approved_generated_animal, tmp_path / "rejected_empty_gate")


def test_rejects_tampered_animation_review_media(approved_generated_animal, tmp_path):
    approved_generated_animal["media_path"].write_bytes(b"tampered")

    with pytest.raises(contracts.ContractError, match="review media .* changed"):
        _prepare(approved_generated_animal, tmp_path / "rejected_tamper")


def test_rejects_external_nonmedia_masquerading_as_h264(
    approved_generated_animal, tmp_path
):
    review = contracts.load_json(approved_generated_animal["review_path"])
    hosts = Path("/etc/hosts")
    fake = {
        **_record(hosts),
        "codec": "h264",
        "width": 512,
        "height": 384,
        "frame_count": preparation.REQUIRED_REVIEW_FRAMES,
        "frame_rate": "8/1",
        "duration_seconds": 1,
    }
    review["outputs"]["media"]["walking_side"] = fake
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(contracts.ContractError, match="escaped its artifact root"):
        _prepare(approved_generated_animal, tmp_path / "rejected_fake_media")


def test_rejects_minimal_fake_glb_without_real_skin_accessors(
    approved_generated_animal, tmp_path
):
    fake = tmp_path / "fake_minimal.glb"
    _write_fake_unskinned_glb(fake)
    review = contracts.load_json(approved_generated_animal["review_path"])
    fake_record = _record(fake)
    review["outputs"]["animated_glb"] = fake_record
    review["outputs"]["retargeted_animated_glb"] = copy.deepcopy(fake_record)
    review["outputs"]["media_lineage"] = _build_media_lineage(
        approved_generated_animal["review_path"].parent,
        input_glb=fake,
        media=review["outputs"]["media"],
    )
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(
        contracts.ContractError, match="complete embedded skinned-mesh payload"
    ):
        _prepare(approved_generated_animal, tmp_path / "rejected_fake_glb")


def test_rejects_self_claimed_source_request_and_profile_hashes(
    approved_generated_animal, tmp_path
):
    source = contracts.load_json(approved_generated_animal["source_path"])
    source["request_sha256"] = "a" * 64
    source["profile_sha256"] = "b" * 64
    source["provenance"]["request_sha256"] = "a" * 64
    _write_json(approved_generated_animal["source_path"], source)
    _reauthenticate_registry_source(approved_generated_animal)

    with pytest.raises(contracts.ContractError, match="does not match its request"):
        _prepare(approved_generated_animal, tmp_path / "rejected_self_claim")


def test_rejects_nonfinite_anywhere_in_review(approved_generated_animal, tmp_path):
    review = contracts.load_json(approved_generated_animal["review_path"])
    review["timings_seconds"]["heading"] = float("nan")
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)

    with pytest.raises(contracts.ContractError, match="non-finite"):
        _prepare(approved_generated_animal, tmp_path / "rejected_nan")


def test_rejects_decision_without_external_expected_file_hash(
    approved_generated_animal, tmp_path
):
    with pytest.raises(contracts.ContractError, match="external expected SHA-256"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_unanchored_decision",
            expected_decision_sha256="f" * 64,
        )


def test_rejects_decision_bound_only_to_review_self_claimed_hash(
    approved_generated_animal, tmp_path
):
    review = contracts.load_json(approved_generated_animal["review_path"])
    review["review_sha256"] = "f" * 64
    _rewrite_review_and_rebind_decision(approved_generated_animal, review)
    decision = contracts.load_json(approved_generated_animal["decision_path"])
    decision["review_sha256"] = "f" * 64
    decision["decision_sha256"] = preparation._hash_without(decision, "decision_sha256")
    _write_json(approved_generated_animal["decision_path"], decision)

    with pytest.raises(contracts.ContractError, match="generated animation review"):
        _prepare(approved_generated_animal, tmp_path / "rejected_self_hash")


def test_rejects_existing_output(approved_generated_animal, tmp_path):
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(contracts.ContractError, match="refusing to replace output"):
        _prepare(approved_generated_animal, output)


def test_rejects_freeze_receipt_without_external_expected_file_hash(
    approved_generated_animal,
    tmp_path,
):
    with pytest.raises(contracts.ContractError, match="external expected SHA-256"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_unanchored_receipt",
            expected_receipt_sha256="f" * 64,
        )


def test_cannot_upgrade_caller_assertion_to_cryptographic_identity(
    approved_generated_animal,
    tmp_path,
):
    receipt = contracts.load_json(approved_generated_animal["receipt_path"])
    receipt["user_instruction_authority"]["cryptographic_user_identity_verified"] = True
    receipt["receipt_sha256"] = preparation._hash_without(
        receipt,
        "receipt_sha256",
    )
    _write_json(approved_generated_animal["receipt_path"], receipt)

    with pytest.raises(contracts.ContractError, match="authority contract"):
        _prepare(
            approved_generated_animal,
            tmp_path / "rejected_crypto_upgrade",
        )


@pytest.mark.parametrize(
    "artifact",
    (
        "registry",
        "source_asset",
        "pixal_batch",
        "static_decision_batch",
        "preflight",
        "review",
        "decision",
        "receipt",
    ),
)
def test_rejects_duplicate_keys_across_security_critical_json_entrypoints(
    approved_generated_animal,
    tmp_path,
    artifact,
):
    fixture = approved_generated_animal
    registry = contracts.load_json(fixture["registry_path"])
    paths = {
        "registry": fixture["registry_path"],
        "source_asset": fixture["source_path"],
        "pixal_batch": Path(registry["pixal_batch"]["path"]),
        "static_decision_batch": Path(registry["static_decision_batch"]["path"]),
        "preflight": Path(registry["preflight"]["path"]),
        "review": fixture["review_path"],
        "decision": fixture["decision_path"],
        "receipt": fixture["receipt_path"],
    }
    target = paths[artifact]
    _inject_duplicate_schema(target)
    if artifact == "source_asset":
        registry["source_assets"][0]["source_asset"].update(
            {
                "sha256": _sha256(target),
                "size_bytes": target.stat().st_size,
            }
        )
    elif artifact in {"pixal_batch", "static_decision_batch", "preflight"}:
        registry_field = {
            "pixal_batch": "pixal_batch",
            "static_decision_batch": "static_decision_batch",
            "preflight": "preflight",
        }[artifact]
        registry[registry_field]["sha256"] = _sha256(target)
    if artifact in {
        "source_asset",
        "pixal_batch",
        "static_decision_batch",
        "preflight",
    }:
        registry["registry_sha256"] = source_registry._hash_without(
            registry,
            "registry_sha256",
        )
        _write_json(fixture["registry_path"], registry)
        _rewrite_freeze_receipt(fixture)

    with pytest.raises(contracts.ContractError, match="duplicate JSON object key"):
        _prepare(
            fixture,
            tmp_path / f"rejected_duplicate_{artifact}",
        )


def test_direct_files_reject_arbitrary_symlink_parent_and_allow_only_exact_tmp_bridge(
    tmp_path,
    monkeypatch,
):
    real_root = tmp_path / "real"
    real_root.mkdir()
    evidence = real_root / "evidence.json"
    evidence.write_text('{"status":"passed"}\n', encoding="utf-8")
    arbitrary = tmp_path / "arbitrary"
    arbitrary.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(contracts.ContractError, match="symlink path component"):
        preparation._direct_file(arbitrary / evidence.name, "arbitrary alias")

    trusted_bridge = tmp_path / "tmp"
    trusted_bridge.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setattr(preparation, "SPEAR_TMP_BRIDGE", trusted_bridge.absolute())
    assert (
        preparation._direct_file(
            trusted_bridge / evidence.name,
            "trusted tmp evidence",
        )
        == evidence.resolve()
    )

    nested_real = real_root / "nested_real"
    nested_real.mkdir()
    nested_evidence = nested_real / "nested.json"
    nested_evidence.write_text('{"status":"passed"}\n', encoding="utf-8")
    nested_alias = real_root / "nested_alias"
    nested_alias.symlink_to(nested_real, target_is_directory=True)
    with pytest.raises(contracts.ContractError, match="symlink path component"):
        preparation._direct_file(
            trusted_bridge / "nested_alias" / nested_evidence.name,
            "nested alias",
        )


def test_output_and_artifact_roots_reject_arbitrary_symlink_parents(
    tmp_path,
):
    real_root = tmp_path / "real"
    real_root.mkdir()
    evidence = real_root / "evidence.bin"
    evidence.write_bytes(b"evidence")
    alias = tmp_path / "alias"
    alias.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(contracts.ContractError, match="symlink path component"):
        preparation._new_output_path(alias / "new_output", "output")
    with pytest.raises(contracts.ContractError, match="symlink path component"):
        preparation._resolve_root_artifact(
            _root_record("custom", evidence, real_root),
            {"custom": alias},
            "custom rooted evidence",
        )
