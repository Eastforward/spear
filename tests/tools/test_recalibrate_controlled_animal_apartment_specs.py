import hashlib
import json
from pathlib import Path

from tools.recalibrate_controlled_animal_apartment_specs import (
    build_recalibration_batch,
    calibrated_actor_scale,
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def test_calibrated_actor_scale_uses_observed_measurement():
    measurement = {
        "physical_measurements": {
            "runtime": {"actor_scale": 0.15, "shoulder_height_cm": 60.0}
        },
        "target_comparison": {"target_value_cm": 30.0},
    }

    assert calibrated_actor_scale(measurement) == 0.075


def test_batch_publishes_new_specs_without_changing_source(tmp_path):
    source_root = tmp_path / "source"
    output_root = tmp_path / "calibrated"
    actions = {}
    for action, suffix in (("Idle", "idle"), ("Walking", "walking")):
        spec_path = source_root / "specs" / f"{suffix}.json"
        spec = {
            "camera_pass_table_loop_contract": {
                "animal_scale_rationale": {
                    "actor_scale": 0.15,
                    "base_actor_scale": 0.15,
                    "physical_scale_ratio": 1.0,
                    "policy": "old",
                    "target_value_cm": 30.0,
                }
            },
            "runtime_assertions": {
                "ground_snap_max_abs_correction_cm": 30.0,
                "ground_snap_to_floor": True,
            },
            "sources": [
                {
                    "asset_class": "animal",
                    "asset_id": "dog_test",
                    "actor_scale": 0.15,
                    "species": "dog",
                    "tag": "pixal_dog_test",
                    "wanted_anim": action,
                }
            ],
        }
        _write_json(spec_path, spec)
        actions[action] = {
            "clip_id": f"dog_test_{suffix}_v1",
            "motion": suffix,
            "output_dir": str(source_root / "clips" / suffix),
            "spec": str(spec_path),
            "spec_evidence": {},
        }
    manifest_path = source_root / "spec_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema": "controlled_animal_walk_idle_apartment_specs_v1",
            "records": [
                {
                    "actions": actions,
                    "asset_id": "dog_test",
                    "base_avatar_id": "dog_test",
                    "profile_schema_id": "dog_test_v1",
                    "sampled_attributes": {"size": "medium"},
                    "source_glb": {"path": "/tmp/test.glb", "sha256": "0" * 64},
                    "tag": "pixal_dog_test",
                    "target_physical_profile": {
                        "measurement": "shoulder_height_cm",
                        "target_value_cm": 30.0,
                        "tolerance_cm": 3.0,
                    },
                }
            ],
        },
    )
    measurement_path = tmp_path / "measurements.json"
    _write_json(
        measurement_path,
        {
            "schema": "controlled_animal_physical_measurement_batch_v1",
            "measurements": [
                {
                    "asset_id": "dog_test",
                    "physical_measurements": {
                        "runtime": {
                            "actor_scale": 0.15,
                            "shoulder_height_cm": 60.0,
                        }
                    },
                    "target_comparison": {
                        "status": "outside_tolerance",
                        "target_value_cm": 30.0,
                    },
                }
            ],
        },
    )
    original = (source_root / "specs" / "walking.json").read_bytes()

    result = build_recalibration_batch(
        manifest_path=manifest_path,
        measurement_batch_path=measurement_path,
        output_root=output_root,
        max_relative_error=0.25,
    )

    published = json.loads(result.read_text())
    assert published["avatar_count"] == 1
    assert published["clip_count"] == 2
    record = published["records"][0]
    walking_path = Path(record["actions"]["Walking"]["spec"])
    walking = json.loads(walking_path.read_text())
    assert walking["sources"][0]["actor_scale"] == 0.075
    assert walking["runtime_assertions"]["ground_snap_max_abs_correction_cm"] == 25.0
    assert record["actions"]["Walking"]["output_dir"].startswith(str(output_root))
    assert (source_root / "specs" / "walking.json").read_bytes() == original
    evidence = record["actions"]["Walking"]["spec_evidence"]
    assert evidence["sha256"] == hashlib.sha256(walking_path.read_bytes()).hexdigest()
    assert published["manifest_sha256"]
