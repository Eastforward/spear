from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tests.tools import (
    test_build_user_approved_generated_animal_apartment_specs as apartment_builder_support,
)
from tools import (
    build_user_approved_generated_animal_apartment_specs as apartment_builder,
)
from tools import measure_controlled_animal_physical_attributes as subject
from tools.measure_controlled_animal_physical_attributes import (
    LEGACY_FRONT_UPPER_GROUPS,
    PHYSICAL_MEASUREMENT,
    RETARGET_SCHEMA,
    RIG_SEMANTIC_EVIDENCE_SCHEMA,
    WEIGHT_REPAIR_SCHEMA,
    MeasurementError,
    _load_records,
    authenticate_record_front_upper_groups,
    build_admission_checks,
    build_runtime_measurement,
    quantile,
    resolve_front_upper_groups,
    summarize_size_ordering,
)


def test_measurement_entrypoint_exposes_repo_package_from_external_cwd(
    tmp_path: Path,
) -> None:
    launcher = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "measure_controlled_animal_physical_attributes.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy; "
                f"runpy.run_path({str(launcher)!r}, run_name='measure_import'); "
                "import tools"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def _record(size="medium", profile="dog_golden_retriever_v1"):
    return {
        "base_avatar_id": f"dog_{size}",
        "tag": f"pixal_dog_{size}",
        "profile_schema_id": profile,
        "sampled_attributes": {"size": size},
        "target_physical_profile": {
            "control_attribute": "size",
            "measurement": "shoulder_height_cm",
            "target_value_cm": 50.0,
            "tolerance_cm": 3.0,
            "reference_provenance": {"status": "provisional"},
        },
    }


def _visual(tag, height=80.0, scale=0.13):
    frame = {
        "bounds_ue": {
            "minimum_cm": [0, 0, 27.1],
            "maximum_cm": [90, 30, 27.1 + height],
        },
        "root_transform_ue": {"scale": [scale, scale, scale]},
        "floor_contact": {"within_penetration_tolerance": True},
    }
    return {
        "automatic_checks": {"overall": "passed"},
        "sources": [{"tag": tag, "runtime_frames": [frame, frame]}],
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "size_bytes": path.stat().st_size,
    }


def _canonical_hash_without(value: dict, key: str) -> str:
    payload = {name: item for name, item in value.items() if name != key}
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _seal_manifest(path: Path, value: dict) -> Path:
    value["manifest_sha256"] = _canonical_hash_without(value, "manifest_sha256")
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _manifest_fixture(tmp_path: Path) -> tuple[Path, dict]:
    asset_id = "generated_dog_medium"
    tag = "pixal_generated_dog_medium"
    profile = "dog_generated_v1"
    sampled = {"size": "medium"}
    target = {
        "profile_id": "dog_generated_physical_v1",
        "control_attribute": "size",
        "selected_value": "medium",
        "measurement": PHYSICAL_MEASUREMENT,
        "mode": "relative_to_profile_reference",
        "reference_value_cm": 50.0,
        "reference_provenance": {
            "status": "provisional",
            "source_id": "fixture_reference_v1",
            "artifact": None,
            "notes": "Fixture-only provisional physical target.",
        },
        "scale_ratio": 1.0,
        "target_value_cm": 50.0,
        "tolerance_cm": 3.0,
    }
    source_glb = tmp_path / "generated.glb"
    source_glb.write_bytes(b"generated-rig")
    walking_trajectory = [[float(index), 0.0, 0.0] for index in range(5)]
    actions = {}
    for action_name, motion, kind in (
        ("Walking", "walking", "moving"),
        ("Idle", "idle", "stationary"),
    ):
        trajectory = (
            walking_trajectory
            if action_name == "Walking"
            else [walking_trajectory[2]] * len(walking_trajectory)
        )
        spec = {
            "render_config": {"duration_s": 1.0, "fps": 5, "n_frames": 5},
            "audio_config": {"duration_s": 1.0, "sample_rate_hz": 16000},
            "camera_pass_table_loop_contract": {"left_front_nearest_frame": 2},
            "sources": [
                {
                    "asset_id": asset_id,
                    "tag": tag,
                    "profile_schema_id": profile,
                    "sampled_attributes": sampled,
                    "target_physical_profile": target,
                    "wanted_anim": action_name,
                    "kind": kind,
                    "actor_scale": 0.13,
                    "start_pos_m": trajectory[0],
                    "end_pos_m": trajectory[-1],
                    "trajectory_m": trajectory,
                }
            ],
        }
        if action_name == "Walking":
            spec["rig_direction_check_windows"] = [
                {"frame_a": 0, "frame_b": 1, "label": "start"}
            ]
        else:
            spec["stationary_idle_contract"] = {
                "status": "passed",
                "frame_count": 5,
                "position_m": walking_trajectory[2],
                "maximum_position_delta_m": 0.0,
                "source_waypoint_frame": 2,
            }
        spec_path = tmp_path / f"{motion}.json"
        spec_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
        actions[action_name] = {
            "motion": motion,
            "spec": str(spec_path.resolve()),
            "spec_evidence": _descriptor(spec_path),
            "output_dir": str((tmp_path / f"{motion}-output").resolve()),
            "clip_id": f"{tag}_{motion}_v1",
        }
    record = {
        "base_avatar_id": asset_id,
        "asset_id": asset_id,
        "tag": tag,
        "profile_schema_id": profile,
        "sampled_attributes": sampled,
        "target_physical_profile": target,
        "source_glb": {
            "path": str(source_glb.resolve()),
            "sha256": _sha(source_glb),
        },
        "actions": actions,
    }
    manifest = {
        "schema": subject.SCHEMA,
        "avatar_count": 1,
        "clip_count": 2,
        "records": [record],
    }
    path = _seal_manifest(tmp_path / "spec_manifest.json", manifest)
    return path, manifest


def _semantic_record(
    tmp_path: Path,
    *,
    schema: str = WEIGHT_REPAIR_SCHEMA,
    output_path: Path | None = None,
) -> tuple[dict, Path]:
    source = tmp_path / "generated.glb"
    source.write_bytes(b"generated-rig")
    bound_output = output_path or source
    chains = {
        "front_side_negative": [
            "opaque_front_negative_proximal",
            "opaque_front_negative_distal",
        ],
        "front_side_positive": [
            "opaque_front_positive_proximal",
            "opaque_front_positive_distal",
        ],
    }
    if schema == WEIGHT_REPAIR_SCHEMA:
        payload = {
            "schema": schema,
            "output": _descriptor(bound_output),
            "authority_contract": {
                "native_mesh_geometry_preserved": True,
                "native_mesh_topology_preserved": True,
                "pbr_material_preserved": True,
                "fitted_skeleton_rest_matrices_preserved": True,
                "approved_animation_curves_preserved": True,
                "only_vertex_weights_modified_in_memory": True,
            },
            "semantic_rig": {"chains": chains},
        }
        kind = "motion_aware_weight_repair"
    else:
        payload = {
            "schema": schema,
            "export": _descriptor(bound_output),
            "semantic_inference": {
                "bone_name_independent_target": True,
                "complete_target_bone_coverage": True,
                "chains": chains,
            },
        }
        kind = "bone_name_independent_retarget"
    evidence = tmp_path / "semantic_evidence.json"
    evidence.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    record = {
        "source_glb": {"path": str(source.resolve()), "sha256": _sha(source)},
        "rig_semantic_evidence": {
            "schema": RIG_SEMANTIC_EVIDENCE_SCHEMA,
            "kind": kind,
            "artifact": _descriptor(evidence),
            "semantic_schema": schema,
            "source_glb_sha256": _sha(source),
        },
    }
    return record, evidence


def test_runtime_measurement_uses_geometry_ratio_not_prompt_target():
    record = _record()
    result = build_runtime_measurement(
        record=record,
        walking_spec={"sources": [{"tag": record["tag"], "actor_scale": 0.13}]},
        visual=_visual(record["tag"]),
        geometry={
            "bounds_height_units": 4.0,
            "shoulder_height_units": 2.5,
            "nose_to_tail_length_units": 5.0,
        },
    )

    runtime = result["physical_measurements"]["runtime"]
    assert runtime == {
        "actor_scale": 0.13,
        "shoulder_height_cm": 50.0,
        "bounds_height_cm": 80.0,
        "nose_to_tail_length_cm": 100.0,
    }
    assert result["target_comparison"]["status"] == "within_tolerance"


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf"), True])
def test_runtime_measurement_rejects_invalid_ue_root_scale(bad_value):
    record = _record()
    visual = copy.deepcopy(_visual(record["tag"]))
    visual["sources"][0]["runtime_frames"][0]["root_transform_ue"]["scale"][0] = (
        bad_value
    )

    with pytest.raises(MeasurementError, match="finite number"):
        build_runtime_measurement(
            record=record,
            walking_spec={"sources": [{"tag": record["tag"], "actor_scale": 0.13}]},
            visual=visual,
            geometry={
                "bounds_height_units": 4.0,
                "shoulder_height_units": 2.5,
                "nose_to_tail_length_units": 5.0,
            },
        )


@pytest.mark.parametrize(
    ("location", "bad_value"),
    [
        ("actor_scale", True),
        ("actor_scale", float("nan")),
        ("unused_bounds_axis", float("inf")),
        ("geometry", False),
        ("geometry", float("-inf")),
        ("target", float("inf")),
        ("tolerance", False),
    ],
)
def test_runtime_measurement_rejects_other_invalid_numbers(location, bad_value):
    record = _record()
    visual = copy.deepcopy(_visual(record["tag"]))
    walking_spec = {"sources": [{"tag": record["tag"], "actor_scale": 0.13}]}
    geometry = {
        "bounds_height_units": 4.0,
        "shoulder_height_units": 2.5,
        "nose_to_tail_length_units": 5.0,
    }
    if location == "actor_scale":
        walking_spec["sources"][0]["actor_scale"] = bad_value
    elif location == "unused_bounds_axis":
        visual["sources"][0]["runtime_frames"][0]["bounds_ue"]["minimum_cm"][0] = (
            bad_value
        )
    elif location == "geometry":
        geometry["nose_to_tail_length_units"] = bad_value
    elif location == "target":
        record["target_physical_profile"]["target_value_cm"] = bad_value
    elif location == "tolerance":
        record["target_physical_profile"]["tolerance_cm"] = bad_value

    with pytest.raises(MeasurementError, match="finite number"):
        build_runtime_measurement(
            record=record,
            walking_spec=walking_spec,
            visual=visual,
            geometry=geometry,
        )


def test_size_ordering_uses_observed_shoulder_heights():
    rows = []
    for size, height in (("large", 60.0), ("small", 40.0), ("medium", 50.0)):
        rows.append(
            {
                "profile_schema_id": "dog_example_v1",
                "sampled_size": size,
                "physical_measurements": {"runtime": {"shoulder_height_cm": height}},
            }
        )

    summary = summarize_size_ordering(rows)

    assert summary[0]["ordered_sizes"] == ["small", "medium", "large"]
    assert summary[0]["strictly_increasing"] is True
    assert quantile([0.0, 10.0], 0.25) == 2.5


def test_single_outside_tolerance_is_rejected_without_vacuous_ordering_pass():
    record = _record()
    measured = build_runtime_measurement(
        record=record,
        walking_spec={"sources": [{"tag": record["tag"], "actor_scale": 0.13}]},
        visual=_visual(record["tag"]),
        geometry={
            "bounds_height_units": 4.0,
            "shoulder_height_units": 2.0,
            "nose_to_tail_length_units": 5.0,
        },
    )

    assert measured["target_comparison"]["status"] == "outside_tolerance"
    ordering = summarize_size_ordering([measured])
    assert ordering[0]["ordering_applicable"] is False
    assert ordering[0]["strictly_increasing"] is None
    assert ordering[0]["status"] == "not_applicable_single_size"

    checks = build_admission_checks([measured], ordering)
    assert checks["all_applicable_profile_size_medians_strictly_increasing"] is None
    assert checks["size_ordering_status"] == "not_applicable_single_size_only"
    assert checks["size_ordering_admission_satisfied"] is True
    assert checks["all_targets_within_tolerance"] is False
    assert checks["outside_provisional_target_tolerance_count"] == 1
    assert checks["downstream_source_asset_registration_ready"] is False
    assert checks["overall"] == "rejected"


@pytest.mark.parametrize("bad_value", [True, float("nan"), float("inf")])
def test_quantile_and_ordering_reject_invalid_numbers(bad_value):
    with pytest.raises(MeasurementError, match="finite number"):
        quantile([0.0, bad_value], 0.5)
    row = {
        "profile_schema_id": "dog_example_v1",
        "sampled_size": "medium",
        "physical_measurements": {"runtime": {PHYSICAL_MEASUREMENT: bad_value}},
    }
    with pytest.raises(MeasurementError, match="finite number"):
        summarize_size_ordering([row])


def test_admission_checks_reject_empty_or_malformed_evidence():
    with pytest.raises(MeasurementError, match="non-empty"):
        build_admission_checks([], [])
    with pytest.raises(MeasurementError, match="ordering status"):
        build_admission_checks(
            [
                {
                    "target_comparison": {
                        "status": "within_tolerance",
                    }
                }
            ],
            [
                {
                    "ordering_applicable": False,
                    "strictly_increasing": True,
                    "status": "passed",
                }
            ],
        )


def test_load_records_authenticates_manifest_and_record(tmp_path: Path):
    manifest_path, _manifest = _manifest_fixture(tmp_path)

    records = _load_records([manifest_path])

    assert len(records) == 1
    assert records[0]["base_avatar_id"] == "generated_dog_medium"


def test_load_records_reauthenticates_builder_v2_authority_graph(
    tmp_path: Path,
) -> None:
    inputs = apartment_builder_support._fixture(
        tmp_path,
        texture_transcode=True,
    )
    inputs.pop("semantic_evidence")
    manifest = apartment_builder.build_specs(
        **inputs,
        output_root=tmp_path / "apartment",
    )

    records = _load_records([manifest])

    assert len(records) == 1
    assert records[0]["base_avatar_id"] == "horse_candidate_001"
    assert records[0]["emitter_measurement"] == _descriptor(
        inputs["emitter_measurement"]
    )
    assert (
        records[0]["_authenticated_apartment_v2"]["runtime_lineage"]
        == records[0]["runtime_lineage"]
    )


def test_load_records_rejects_resealed_builder_v2_runtime_lineage_change(
    tmp_path: Path,
) -> None:
    inputs = apartment_builder_support._fixture(
        tmp_path,
        texture_transcode=True,
    )
    inputs.pop("semantic_evidence")
    manifest_path = apartment_builder.build_specs(
        **inputs,
        output_root=tmp_path / "apartment",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][0]["runtime_lineage"]["ue_import_glb"] = copy.deepcopy(
        manifest["records"][0]["runtime_lineage"]["reviewed_animated_glb"]
    )
    manifest_path.chmod(0o644)
    _seal_manifest(manifest_path, manifest)

    with pytest.raises(MeasurementError, match="Apartment v2 authority"):
        _load_records([manifest_path])


def test_load_records_rejects_stale_manifest_hash(tmp_path: Path):
    manifest_path, _manifest = _manifest_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["records"][0]["tag"] = "pixal_changed_after_sealing"
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(MeasurementError, match="invalid controlled animal manifest"):
        _load_records([manifest_path])


def test_load_records_rejects_resealed_record_identity_change(tmp_path: Path):
    manifest_path, _manifest = _manifest_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["records"][0]["asset_id"] = "different_asset"
    _seal_manifest(manifest_path, payload)

    with pytest.raises(MeasurementError, match="record identity changed"):
        _load_records([manifest_path])


def test_load_records_rejects_empty_batch_even_with_valid_hash(tmp_path: Path):
    manifest_path, _manifest = _manifest_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["records"] = []
    payload["avatar_count"] = 0
    payload["clip_count"] = 0
    _seal_manifest(manifest_path, payload)

    with pytest.raises(MeasurementError, match="invalid controlled animal manifest"):
        _load_records([manifest_path])


def test_load_records_rejects_nonfinite_values_even_outside_used_fields(
    tmp_path: Path,
):
    manifest_path, _manifest = _manifest_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["unused_diagnostic"] = {"score": float("inf")}
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(MeasurementError, match="non-finite number"):
        _load_records([manifest_path])


def test_load_records_requires_selected_physical_identity(tmp_path: Path):
    manifest_path, _manifest = _manifest_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["records"][0]["target_physical_profile"].pop("selected_value")
    _seal_manifest(manifest_path, payload)

    with pytest.raises(MeasurementError, match="physical identity changed"):
        _load_records([manifest_path])


def test_load_records_rejects_resealed_spec_identity_change(tmp_path: Path):
    manifest_path, _manifest = _manifest_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    action = payload["records"][0]["actions"]["Walking"]
    spec_path = Path(action["spec"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["sources"][0]["tag"] = "pixal_changed_in_spec"
    spec_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    action["spec_evidence"] = _descriptor(spec_path)
    _seal_manifest(manifest_path, payload)

    with pytest.raises(MeasurementError, match="spec identity changed"):
        _load_records([manifest_path])


def test_load_records_requires_target_profile_in_both_specs(tmp_path: Path):
    manifest_path, _manifest = _manifest_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for action in payload["records"][0]["actions"].values():
        spec_path = Path(action["spec"])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec["sources"][0].pop("target_physical_profile")
        spec_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
        action["spec_evidence"] = _descriptor(spec_path)
    _seal_manifest(manifest_path, payload)

    with pytest.raises(MeasurementError, match="spec identity changed"):
        _load_records([manifest_path])


def test_load_records_rejects_boolean_target_reference_and_scale(tmp_path: Path):
    manifest_path, _manifest = _manifest_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    target = payload["records"][0]["target_physical_profile"]
    target["reference_value_cm"] = True
    target["scale_ratio"] = True
    for action in payload["records"][0]["actions"].values():
        spec_path = Path(action["spec"])
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec["sources"][0]["target_physical_profile"] = copy.deepcopy(target)
        spec_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
        action["spec_evidence"] = _descriptor(spec_path)
    _seal_manifest(manifest_path, payload)

    with pytest.raises(MeasurementError, match="finite number"):
        _load_records([manifest_path])


def test_load_records_rejects_boolean_spec_actor_scale(tmp_path: Path):
    manifest_path, _manifest = _manifest_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    action = payload["records"][0]["actions"]["Idle"]
    spec_path = Path(action["spec"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["sources"][0]["actor_scale"] = True
    spec_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    action["spec_evidence"] = _descriptor(spec_path)
    _seal_manifest(manifest_path, payload)

    with pytest.raises(MeasurementError, match="finite number"):
        _load_records([manifest_path])


@pytest.mark.parametrize("location", ["trajectory_component", "render_fps"])
def test_load_records_rejects_boolean_spec_numeric_contract(
    tmp_path: Path, location: str
):
    manifest_path, _manifest = _manifest_fixture(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    action = payload["records"][0]["actions"]["Walking"]
    spec_path = Path(action["spec"])
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if location == "trajectory_component":
        spec["sources"][0]["trajectory_m"][0][0] = True
        error = "finite number"
    else:
        spec["render_config"]["fps"] = True
        error = "integer"
    spec_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    action["spec_evidence"] = _descriptor(spec_path)
    _seal_manifest(manifest_path, payload)

    with pytest.raises(MeasurementError, match=error):
        _load_records([manifest_path])


def test_load_records_rejects_manifest_and_source_symlinks(tmp_path: Path):
    manifest_path, _manifest = _manifest_fixture(tmp_path)
    manifest_link = tmp_path / "manifest-link.json"
    manifest_link.symlink_to(manifest_path)

    with pytest.raises(MeasurementError, match="manifest cannot be a symlink"):
        _load_records([manifest_link])

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = Path(payload["records"][0]["source_glb"]["path"])
    source_link = tmp_path / "source-link.glb"
    source_link.symlink_to(source_path)
    payload["records"][0]["source_glb"]["path"] = str(source_link)
    _seal_manifest(manifest_path, payload)

    with pytest.raises(MeasurementError, match="source GLB.*symlink"):
        _load_records([manifest_path])


def test_measurement_artifacts_and_descriptors_reject_symlinks(tmp_path: Path):
    target = tmp_path / "target.json"
    target.write_text('{"status":"passed"}\n', encoding="utf-8")
    link = tmp_path / "target-link.json"
    link.symlink_to(target)
    descriptor = _descriptor(target)
    descriptor["path"] = str(link)

    with pytest.raises(MeasurementError, match="symlink"):
        subject._artifact(link)
    assert subject._descriptor_binds_file(descriptor, target, _sha(target)) is False


def test_measurement_artifacts_reject_symlink_parent_directories(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    target = real / "target.json"
    target.write_text('{"status":"passed"}\n', encoding="utf-8")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    aliased_target = alias / "target.json"
    descriptor = _descriptor(target)
    descriptor["path"] = str(aliased_target)

    with pytest.raises(MeasurementError, match="symlink"):
        subject._artifact(aliased_target)
    assert subject._descriptor_binds_file(descriptor, target, _sha(target)) is False


def test_batch_api_rejects_boolean_worker_count(tmp_path: Path):
    with pytest.raises(MeasurementError, match="workers"):
        subject.build_measurements(
            manifest_paths=[],
            output_root=tmp_path / "measurement-output",
            blender=Path("/fake/blender"),
            workers=True,
        )


def test_cli_returns_nonzero_for_retained_rejected_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
):
    batch_path = tmp_path / "measurement_batch_manifest.json"
    batch_path.write_text(
        json.dumps(
            {
                "asset_count": 1,
                "automatic_checks": {"overall": "rejected"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        subject,
        "build_measurements",
        lambda **_kwargs: batch_path,
    )

    returncode = subject.main(
        [
            "--manifest",
            str(tmp_path / "input.json"),
            "--output-root",
            str(tmp_path / "output"),
        ]
    )

    assert returncode == 2
    assert "CONTROLLED_ANIMAL_PHYSICAL_MEASUREMENT_REJECTED" in capsys.readouterr().out


def test_rejected_single_asset_batch_is_retained_only_for_recalibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manifest_path, manifest = _manifest_fixture(tmp_path)
    record = manifest["records"][0]
    walking = record["actions"]["Walking"]
    visual_path = Path(walking["output_dir"]) / "videos" / "actor_visual_metadata.json"
    visual_path.parent.mkdir(parents=True)
    visual_path.write_text(
        json.dumps(_visual(record["tag"], height=80.0, scale=0.13)) + "\n",
        encoding="utf-8",
    )

    def fake_blender(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(
                {
                    "bounds_height_units": 4.0,
                    "shoulder_height_units": 2.0,
                    "nose_to_tail_length_units": 5.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "fake blender\n")

    monkeypatch.setattr(subject.subprocess, "run", fake_blender)
    output_root = tmp_path / "measurement-output"
    batch_path = subject.build_measurements(
        manifest_paths=[manifest_path],
        output_root=output_root,
        blender=Path("/fake/blender"),
        workers=1,
    )

    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    assert batch["asset_count"] == 1
    assert batch["measurements"][0]["target_comparison"]["status"] == (
        "outside_tolerance"
    )
    assert batch["admission_policy"] == {
        "target_tolerance": ("required_for_downstream_source_asset_registration"),
        "single_size_ordering": "not_applicable_not_vacuously_passed",
        "rejected_batch_usage": ("measurement_and_scale_recalibration_evidence_only"),
    }
    assert batch["automatic_checks"]["size_ordering_status"] == (
        "not_applicable_single_size_only"
    )
    assert batch["automatic_checks"]["all_targets_within_tolerance"] is False
    assert (
        batch["automatic_checks"]["downstream_source_asset_registration_ready"] is False
    )
    assert batch["automatic_checks"]["overall"] == "rejected"
    assert batch["batch_sha256"] == _canonical_hash_without(batch, "batch_sha256")


def test_foreleg_groups_support_quaternius_and_rocketbox_native_rigs():
    assert resolve_front_upper_groups({"Bone.014", "Bone.017", "Bone"}) == {
        "Bone.014",
        "Bone.017",
    }
    assert LEGACY_FRONT_UPPER_GROUPS == {"Bone.014", "Bone.017"}
    assert resolve_front_upper_groups(
        {"beagle L UpperArm", "beagle R UpperArm", "beagle Pelvis"}
    ) == {"beagle L UpperArm", "beagle R UpperArm"}


@pytest.mark.parametrize("schema", [WEIGHT_REPAIR_SCHEMA, RETARGET_SCHEMA])
def test_hash_bound_semantics_resolve_arbitrary_tokenrig_group_names(
    tmp_path: Path, schema: str
):
    record, _evidence = _semantic_record(tmp_path, schema=schema)

    authenticated = authenticate_record_front_upper_groups(record)

    assert authenticated is not None
    groups, wrapper = authenticated
    assert groups == {
        "opaque_front_negative_proximal",
        "opaque_front_positive_proximal",
    }
    assert (
        resolve_front_upper_groups(
            {
                "opaque_front_negative_proximal",
                "opaque_front_negative_distal",
                "opaque_front_positive_proximal",
                "opaque_front_positive_distal",
            },
            authenticated_semantic_groups=groups,
        )
        == groups
    )
    assert wrapper == record["rig_semantic_evidence"]


def test_opaque_numeric_groups_fail_closed_without_semantic_evidence():
    with pytest.raises(MeasurementError, match="missing or ambiguous"):
        resolve_front_upper_groups({"bone_0", "bone_4", "bone_19", "bone_27"})


def test_rig_semantic_artifact_tampering_fails_closed(tmp_path: Path):
    record, evidence = _semantic_record(tmp_path)
    evidence.write_text("{}\n", encoding="utf-8")

    with pytest.raises(MeasurementError, match="artifact changed"):
        authenticate_record_front_upper_groups(record)


def test_rig_semantics_must_bind_the_exact_measured_glb(tmp_path: Path):
    other = tmp_path / "other.glb"
    other.write_bytes(b"different-rig")
    record, _evidence = _semantic_record(
        tmp_path, schema=WEIGHT_REPAIR_SCHEMA, output_path=other
    )

    with pytest.raises(MeasurementError, match="does not bind"):
        authenticate_record_front_upper_groups(record)


def test_weight_repair_semantics_require_preservation_authority(tmp_path: Path):
    record, evidence = _semantic_record(tmp_path)
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["authority_contract"]["fitted_skeleton_rest_matrices_preserved"] = False
    evidence.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    record["rig_semantic_evidence"]["artifact"] = _descriptor(evidence)

    with pytest.raises(MeasurementError, match="authority did not pass"):
        authenticate_record_front_upper_groups(record)
