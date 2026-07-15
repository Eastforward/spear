import hashlib
import json
from pathlib import Path

from tools.prepare_stable_animal_ue_jobs import build_jobs, prepare


REPO_ROOT = Path(__file__).resolve().parents[2]


def _artifact(path):
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


def _inputs(tmp_path):
    glb = tmp_path / "husky.glb"
    glb.write_bytes(b"stable glb")
    deformation = tmp_path / "deformation.json"
    deformation.write_text("{}")
    entry = {
        "template_id": "quaternius_ultimate_husky_v1",
        "taxonomy_label": "Husky",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "runtime_glb": _artifact(glb),
        "deformation_audit": _artifact(deformation),
        "actions": ["Walking", "Idle"],
        "direction": {
            "cardinal_yaw_deg": 90,
            "automatic_fine_yaw_inference": False,
            "review_status": "agent_selected_pending_human_review",
        },
        "qa": {
            "walking_deformation": "passed_automatic_deformation_measurements",
            "idle_deformation": "passed_automatic_deformation_measurements",
            "ue_apartment_media": "pending",
            "human_visual_review": "pending",
        },
    }
    registry = {
        "schema": "avengine_quaternius_stable_template_registry_v1",
        "entries": [entry],
    }
    selection = {
        "schema": "avengine_stable_animal_ue_selection_v1",
        "selections": [
            {
                "template_id": entry["template_id"],
                "species": "dog",
                "breed": "husky",
                "actor_scale": 0.15,
                "audio_lookup": "dog_bark",
                "audio_source_height_offset_m": 0.45,
            }
        ],
    }
    return registry, selection


def test_build_jobs_uses_stable_namespace_and_authored_cardinal_yaw(tmp_path):
    registry, selection = _inputs(tmp_path)

    jobs = build_jobs(registry, selection)

    assert len(jobs) == 1
    assert jobs[0]["tag"].startswith("stable_dog_husky_")
    assert jobs[0]["walking_forward_yaw_offset_deg"] == 90
    assert jobs[0]["expected_actions"] == ["Idle", "Walking"]
    assert jobs[0]["human_review_status"] == (
        "agent_selected_pending_human_review"
    )
    assert jobs[0]["formal_dataset_registration_authorized"] is False


def test_build_jobs_accepts_generic_ofat_registry_and_local_pending_status(tmp_path):
    registry, selection = _inputs(tmp_path)
    registry["schema"] = "avengine_stable_animal_template_registry_v2"
    entry = registry["entries"][0]
    entry["direction"]["review_status"] = "local_ofat_visual_review_pending"
    entry["sampled_attributes"] = {"size": "small"}

    jobs = build_jobs(registry, selection)

    assert jobs[0]["human_review_status"] == "local_ofat_visual_review_pending"
    assert jobs[0]["sampled_attributes"] == {"size": "small"}


def test_prepare_publishes_non_overwriting_authenticated_jobs(tmp_path):
    registry, selection = _inputs(tmp_path)
    registry_path = tmp_path / "registry.json"
    selection_path = tmp_path / "selection.json"
    registry_path.write_text(json.dumps(registry))
    selection_path.write_text(json.dumps(selection))
    output = tmp_path / "prepared"

    result = prepare(
        registry_path=registry_path,
        selection_path=selection_path,
        output_root=output,
    )

    payload = json.loads(result.read_text())
    assert payload["schema"] == "stable_animal_ue_import_batch_v1"
    assert payload["job_count"] == 1
    assert payload["registry"]["sha256"] == hashlib.sha256(
        registry_path.read_bytes()
    ).hexdigest()
    assert (output / "ue_import_preparation_manifest.json").is_file()


def test_remaining_native_batch_selection_is_bounded_and_explicit():
    selection = json.loads(
        (
            REPO_ROOT
            / "data/controlled_source_attributes_v1/"
            "stable_animal_ue_batch_remaining11_v1.json"
        ).read_text(encoding="utf-8")
    )
    records = selection["selections"]

    assert selection["usage_scope"] == "research_candidate"
    assert selection["human_visual_review"] == "pending"
    assert selection["formal_dataset_registration_authorized"] is False
    assert len(records) == 11
    assert len({item["template_id"] for item in records}) == 11
    assert "quaternius_ultimate_husky_v1" not in {
        item["template_id"] for item in records
    }
    assert {item["audio_lookup"] for item in records} >= {
        "cattle_moo",
        "deer_call",
        "fox_call",
        "horse_neigh",
        "wolf_howl",
        "silent",
    }
    for item in records:
        assert 0.01 <= item["actor_scale"] <= 1.0
        assert 0.05 <= item["audio_source_height_offset_m"] <= 3.0
        assert item["scale_rationale"]
