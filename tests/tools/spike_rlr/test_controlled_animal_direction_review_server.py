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
            "pixal_input_rgba",
            "prebind_lod_glb",
            "static_contact_sheet",
            "static_top_view",
            "current_bound_glb",
            "walking_side",
            "walking_front",
            "idle_side",
            "apartment_walking_review",
            "apartment_walking_main",
            "apartment_walking_topdown",
            "apartment_idle_review",
            "apartment_idle_main",
            "apartment_idle_topdown",
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


def test_cardinal_yaw_normalization_and_preview_matrix_contract():
    assert server._normalize_yaw(0) == 0
    assert server._normalize_yaw(180) == 180
    assert server._normalize_yaw(-180) == 180
    assert server._normalize_yaw(185) == -175
    assert server.ALLOWED_DELTAS == {-90.0, 90.0, 180.0}
    assert np.isclose(np.linalg.det(server._manual_preview_matrix(90)), 1.0)
    assert np.allclose(
        server._manual_preview_matrix(180) @ np.asarray([1.0, 0.0, 0.0]),
        np.asarray([-1.0, 0.0, 0.0]),
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
    load_count = 0

    def fake_load(_path):
        nonlocal load_count
        load_count += 1
        return trimesh.creation.box()

    monkeypatch.setattr(server, "_load_preview_mesh", fake_load)

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

    fine_rotation = client.post(
        "/api/rotate/dog_fixture", json={"delta_deg": 5}
    )
    assert fine_rotation.status_code == 409
    assert "cardinal" in fine_rotation.get_json()["error"]

    rotated = client.post(
        "/api/rotate/dog_fixture", json={"delta_deg": 180}
    )
    assert rotated.status_code == 200
    assert rotated.get_json()["yaw_deg"] == 180
    assert load_count == 1

    decided = client.post(
        "/api/decision/dog_fixture",
        json={
            "status": "source_pose_and_cardinal_orientation_approved",
            "notes": "torso axis is +X",
            "pose_checks": {
                "spine_is_straight": True,
                "head_is_aligned_with_torso": True,
                "front_and_hind_legs_share_consistent_planes": True,
                "all_paws_share_one_ground_plane": True,
            },
        },
    )
    assert decided.status_code == 200
    decision_path = state_root / "decisions/dog_fixture.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    assert decision["manual_cardinal_yaw_about_gltf_positive_y_deg"] == 180
    assert decision["automatic_orientation_inference_used"] is False
    assert decision["initial_preview_pretransform"] == "identity"
    assert np.isclose(decision["determinant"], 1.0)
    assert decision["downstream_candidate"] == {
        "manual_cardinal_yaw_deg": 180.0,
        "binding_pretransform": "not_authorized_by_this_visual_gate",
        "coordinate_mapping_status": "requires_binding_basis_and_straight_line_ue_canary",
    }
    assert decision["current_walking_media_status"] == "rejected_by_user_visual_review"
    assert decision["source_assets_modified"] is False
    assert decision["formal_dataset_registration_authorized"] is False

    locked = client.post(
        "/api/rotate/dog_fixture", json={"delta_deg": 90}
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
        json={"status": "source_pose_rejected", "notes": ""},
    )
    assert response.status_code == 409
    assert "note" in response.get_json()["error"]


def test_approval_records_optional_pose_hints_without_blocking(tmp_path, monkeypatch):
    entry = _entry()
    manifest = {
        "schema": server.MANIFEST_SCHEMA,
        "manifest_sha256": "c" * 64,
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
        json={
            "status": "source_pose_and_cardinal_orientation_approved",
            "notes": "looks close",
            "pose_checks": {
                "spine_is_straight": True,
                "head_is_aligned_with_torso": True,
                "front_and_hind_legs_share_consistent_planes": False,
                "all_paws_share_one_ground_plane": True,
            },
        },
    )
    assert response.status_code == 200
    decision = json.loads(
        (tmp_path / "state/decisions/dog_fixture.json").read_text(encoding="utf-8")
    )
    assert decision["manual_pose_checks_are_advisory"] is True
    assert decision["manual_pose_checks"] == {
        "all_paws_share_one_ground_plane": True,
        "front_and_hind_legs_share_consistent_planes": False,
        "head_is_aligned_with_torso": True,
        "spine_is_straight": True,
    }


def test_new_canary_entry_can_omit_apartment_media_from_video_tabs():
    entry = _entry()
    entry["current_evidence_status"]["walking_direction"] = (
        "new_canary_pending_manual_review"
    )
    for name in (
        "apartment_walking_review",
        "apartment_walking_main",
        "apartment_walking_topdown",
        "apartment_idle_review",
        "apartment_idle_main",
        "apartment_idle_topdown",
    ):
        del entry["artifacts"][name]

    public = server._public_entry(entry)

    assert "walking_side" in public["artifacts"]
    assert "apartment_walking_review" not in public["artifacts"]
    assert "Object.keys(labels).filter(k=>a.artifacts[k])" in server.HTML


def test_page_exposes_authenticated_apartment_walk_and_idle_tabs():
    html = server.HTML

    assert "apartment_walking_review" in html
    assert "apartment_walking_main" in html
    assert "apartment_walking_topdown" in html
    assert "apartment_idle_review" in html
    assert "apartment_idle_main" in html
    assert "apartment_idle_topdown" in html
    assert "不会自动注册正式资产" in html


def test_page_labels_cardinal_choice_as_current_direction_not_ninety_degrees():
    html = server.HTML

    assert "姿势合格并保存当前方向" in html
    assert "$('approve').textContent=`姿势合格并保存当前方向（${yaw}°）`" in html
    assert "保存当前整90°方向" not in html


def test_page_marks_pose_checks_optional_and_prefetches_neighbor_previews():
    html = server.HTML

    assert "可选检查提示（不勾选也能保存）" in html
    assert "四项源姿势检查必须全部人工确认" not in html
    assert "prefetchNeighbors" in html
    assert "历史失败动画，仅供定位" in html
