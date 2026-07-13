import hashlib
import json

from tools.prepare_stable_animal_ue_jobs import build_jobs, prepare


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
