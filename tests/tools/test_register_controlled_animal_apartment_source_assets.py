import copy
import json
from pathlib import Path

from tools import controlled_source_asset_schema as schema
from tools.register_controlled_animal_apartment_source_assets import upgrade_source_asset


def _artifact():
    return {
        "root_id": "fixture_root",
        "path": "fixture.bin",
        "sha256": "a" * 64,
        "size_bytes": 1,
    }


def test_upgrade_preserves_absolute_identity_rights_and_passes_scene_qa():
    profile = schema.load_json(
        Path("data/controlled_source_attributes_v1/profiles/animal/cat_siamese_bindpose_v2.json")
    )
    request = schema.sample_instance_requests(profile, count=1, batch_seed=13)[0]
    asset = schema.build_source_asset_v2(
        request,
        artifacts={"static_mesh": _artifact()},
        physical_measurements={"status": "pending"},
        provenance={
            "attempt_id": "static_fixture_v1",
            "request_sha256": request["request_sha256"],
            "models": copy.deepcopy(request["generation_plan"]["model_revisions"]),
        },
        rights={
            "status": "review_required",
            "licenses": [_artifact()],
            "blockers": ["fixture_rights_review"],
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
    measured = {
        "status": "measured",
        "method": "fixture_measurement_v1",
        "runtime": {"actor_scale": 0.1, "shoulder_height_cm": 30.0},
    }

    upgraded = upgrade_source_asset(
        asset,
        physical_measurements=measured,
        added_artifacts={"apartment_registry": _artifact()},
    )

    assert upgraded["asset_id"] == asset["asset_id"]
    assert upgraded["sampled_attributes"] == asset["sampled_attributes"]
    assert upgraded["physical_measurements"] == measured
    assert upgraded["rights"] == asset["rights"]
    assert upgraded["qa"] == {
        "reference_2d": "passed",
        "static_mesh": "passed",
        "binding": "passed",
        "walking": "passed",
        "idle": "passed",
        "ue_import_readback": "passed",
        "apartment_media": "passed",
        "audio": "passed",
    }
    assert upgraded["state_classification"] == "research_candidate"
