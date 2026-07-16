import json

from tools.spike_rlr import generated_animal_motion_basis_review_server as server


def _preview(root, asset_id="animal_x"):
    asset_root = root / asset_id
    asset_root.mkdir(parents=True)
    candidates = []
    for side in ("matched", "swapped"):
        for yaw in (-90, 0, 90, 180):
            candidates.append(
                {
                    "candidate_id": f"yaw_{yaw}_side_{side}",
                    "motion_basis_yaw_deg": yaw,
                    "side_chain_mode": side,
                    "rotation_transfer_mode": "world-left-delta-v2",
                    "source_motion_forward": [1, 0, 0],
                    "target_animation_generated": False,
                    "frames": [],
                }
            )
    payload = {
        "schema": server.PREVIEW_SCHEMA,
        "asset_id": asset_id,
        "target_animation_generated": False,
        "target": {
            "path": "/target.glb",
            "sha256": "target-hash",
            "size_bytes": 10,
            "reviewed_front_axis": "positive-x",
        },
        "source_motion": {
            "path": "/source.glb",
            "sha256": "source-hash",
            "size_bytes": 20,
            "action": "Walking",
        },
        "candidates": candidates,
    }
    payload["preview_sha256"] = server.hash_without(payload, "preview_sha256")
    (asset_root / "preview.json").write_text(json.dumps(payload))
    return payload


def test_approval_is_immutable_and_authorizes_only_selected_candidate(tmp_path):
    preview_root = tmp_path / "previews"
    state_root = tmp_path / "state"
    preview = _preview(preview_root)
    app = server.create_app(preview_root, state_root)
    client = app.test_client()

    assets = client.get("/api/assets")
    assert assets.status_code == 200
    assert assets.get_json() == [
        {
            "asset_id": "animal_x",
            "preview_sha256": preview["preview_sha256"],
            "target_animation_generated": False,
        }
    ]

    response = client.post(
        "/api/decision/animal_x",
        json={
            "status": "motion_basis_approved",
            "candidate_id": "yaw_0_side_matched",
            "notes": "model and source motion both face +X",
        },
    )
    assert response.status_code == 200
    decision = response.get_json()
    assert decision["human_approved"] is True
    assert decision["target_animation_generation_authorized"] is True
    assert decision["manual_cardinal_motion_basis_yaw_deg"] == 0
    assert decision["side_chain_mode"] == "matched"
    assert decision["decision_sha256"] == server.hash_without(
        decision, "decision_sha256"
    )
    assert (state_root / "decisions/animal_x.json").is_file()

    duplicate = client.post(
        "/api/decision/animal_x",
        json={
            "status": "motion_basis_approved",
            "candidate_id": "yaw_90_side_swapped",
        },
    )
    assert duplicate.status_code == 400
    assert "immutable" in duplicate.get_json()["error"]


def test_rejection_requires_notes_and_never_authorizes_animation(tmp_path):
    preview_root = tmp_path / "previews"
    state_root = tmp_path / "state"
    _preview(preview_root)
    client = server.create_app(preview_root, state_root).test_client()

    missing_note = client.post(
        "/api/decision/animal_x",
        json={
            "status": "motion_basis_rejected",
            "candidate_id": "yaw_0_side_matched",
        },
    )
    assert missing_note.status_code == 400

    response = client.post(
        "/api/decision/animal_x",
        json={
            "status": "motion_basis_rejected",
            "candidate_id": "yaw_0_side_matched",
            "notes": "limb swing is lateral",
        },
    )
    assert response.status_code == 200
    decision = response.get_json()
    assert decision["human_approved"] is False
    assert decision["target_animation_generation_authorized"] is False
