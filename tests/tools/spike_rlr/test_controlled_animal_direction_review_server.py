"""Contract tests for the non-destructive animal direction review gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import trimesh


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

import controlled_animal_direction_review_server as server  # noqa: E402


def _entry() -> dict:
    artifacts = {
        name: {
            "absolute_path": f"/not-read/{name}",
            "server_path": f"/external/{name}",
        }
        for name in (
            "pixal_raw_glb",
            "prebind_lod_glb",
            "static_contact_sheet",
            "current_bound_glb",
            "walking_side",
            "walking_front",
            "idle_side",
            "apartment_walking_review",
            "apartment_walking_main",
            "apartment_walking_topdown",
        )
    }
    return {
        "asset_id": "dog_fixture",
        "species": "dog",
        "breed": "fixture",
        "profile_schema_id": "dog_fixture_v1",
        "sampled_attributes": {"size": "medium"},
        "artifacts": artifacts,
        "current_evidence_status": {
            "walking_direction": "rejected_by_user_visual_review"
        },
    }


def test_yaw_normalization_and_preview_matrix_contract():
    assert server._normalize_yaw(0) == 0
    assert server._normalize_yaw(180) == 180
    assert server._normalize_yaw(-180) == 180
    assert server._normalize_yaw(185) == -175
    assert np.isclose(np.linalg.det(server._combined_preview_matrix(47)), -1.0)
    assert np.allclose(
        server._combined_preview_matrix(180) @ np.asarray([1.0, 0.0, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        atol=1e-8,
    )


def test_review_state_is_transform_only_and_decision_is_immutable(
    tmp_path, monkeypatch
):
    entry = _entry()
    manifest = {
        "schema": server.MANIFEST_SCHEMA,
        "manifest_sha256": "a" * 64,
        "asset_count": 1,
        "formal_dataset_registration_authorized": False,
        "entries": [entry],
    }
    monkeypatch.setattr(
        server,
        "_validate_manifest",
        lambda _path: (manifest, {entry["asset_id"]: entry}),
    )
    monkeypatch.setattr(
        server, "_load_preview_mesh", lambda _path: trimesh.creation.box()
    )

    def fake_render(_mesh, destination, *, yaw_deg, max_points=45_000):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"\x89PNG\r\n\x1a\nfixture")

    monkeypatch.setattr(server, "_render_orientation_preview", fake_render)
    state_root = tmp_path / "review-state"
    client = server.create_app(tmp_path / "manifest.json", state_root).test_client()

    assert client.get("/").status_code == 200
    assert client.get("/api/state").get_json()["dog_fixture"]["yaw_deg"] == 0
    preview = client.get("/preview/dog_fixture.png")
    assert preview.status_code == 200 and preview.data.startswith(b"\x89PNG")

    rotated = client.post(
        "/api/rotate/dog_fixture", json={"delta_deg": 180}
    )
    assert rotated.status_code == 200
    assert rotated.get_json()["yaw_deg"] == 180

    decided = client.post(
        "/api/decision/dog_fixture",
        json={"status": "geometry_orientation_approved", "notes": "head is +X"},
    )
    assert decided.status_code == 200
    decision_path = state_root / "decisions/dog_fixture.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["post_mirror_yaw_about_gltf_positive_y_deg"] == 180
    assert decision["downstream_candidate"] == {
        "flip_x": True,
        "target_rotate_z_deg_after_flip_x": 180.0,
        "coordinate_mapping_status": "requires_straight_line_ue_canary",
    }
    assert decision["current_walking_media_status"] == "rejected_by_user_visual_review"
    assert decision["source_assets_modified"] is False
    assert decision["formal_dataset_registration_authorized"] is False

    locked = client.post(
        "/api/rotate/dog_fixture", json={"delta_deg": 5}
    )
    assert locked.status_code == 409
    assert "immutable" in locked.get_json()["error"]


def test_rejection_requires_a_note(tmp_path, monkeypatch):
    entry = _entry()
    manifest = {
        "schema": server.MANIFEST_SCHEMA,
        "manifest_sha256": "b" * 64,
        "asset_count": 1,
        "formal_dataset_registration_authorized": False,
        "entries": [entry],
    }
    monkeypatch.setattr(
        server,
        "_validate_manifest",
        lambda _path: (manifest, {entry["asset_id"]: entry}),
    )
    client = server.create_app(
        tmp_path / "manifest.json", tmp_path / "state"
    ).test_client()
    response = client.post(
        "/api/decision/dog_fixture",
        json={"status": "geometry_orientation_rejected", "notes": ""},
    )
    assert response.status_code == 409
    assert "note" in response.get_json()["error"]
