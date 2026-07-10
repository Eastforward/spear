"""Tests for the dedicated Rocketbox motion review Flask application."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from rocketbox_motion_review import EXPECTED_ASSET_IDS, REQUIRED_MEDIA  # noqa: E402


REQUIRED_INPUT_HASHES = (
    "avatar_fbx",
    "motion_fbx",
    "source_review",
    "body_color_texture",
    "head_color_texture",
    "opacity_color_texture",
    "retarget_glb",
)


def _write_ready_fixture(review_root: Path, asset_id: str) -> Path:
    review_dir = review_root / asset_id
    review_dir.mkdir(parents=True)
    media = {}
    for name in REQUIRED_MEDIA:
        filename = f"{name}.png" if name == "contact_sheet" else f"{name}.mp4"
        (review_dir / filename).write_bytes(f"{asset_id}:{name}".encode("ascii"))
        media[name] = filename
    manifest = {
        "schema_version": "rocketbox_retarget_manifest_v1",
        "asset_id": asset_id,
        "immutable_input_hashes": {
            name: value * 64
            for name, value in zip(
                REQUIRED_INPUT_HASHES, ("a", "b", "c", "d", "e", "f", "0")
            )
        },
        "binding": {
            "target_asset_id": asset_id,
            "target_mesh_bound": True,
            "official_textures_attached": True,
        },
        "media": media,
        "automatic_checks": {"overall": "passed"},
    }
    (review_dir / "retarget_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return review_dir


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    review_root = tmp_path / "reviews"
    for asset_id in EXPECTED_ASSET_IDS:
        _write_ready_fixture(review_root, asset_id)
    return review_root


def _client(workspace: Path):
    from rocketbox_motion_review_server import create_app

    return create_app(workspace).test_client()


def test_asset_page_shows_bound_video_tabs_and_decision_controls(workspace):
    response = _client(workspace).get("/asset/rocketbox_male_adult_01")

    assert response.status_code == 200
    assert b"<video" in response.data
    for label in (
        b"Front",
        b"Side",
        b"Top",
        b"Joints",
        b"Feet",
        b"Source + Target",
        b"Contact",
    ):
        assert label in response.data
    assert b"Approve" in response.data
    assert b"Reject" in response.data
    assert b"loop" in response.data
    assert b"muted" in response.data
    assert b"playsinline" in response.data


def test_media_route_rejects_unknown_kind_and_path_traversal(workspace):
    client = _client(workspace)

    assert client.get("/media/rocketbox_male_adult_01/front").status_code == 200
    assert client.get("/media/rocketbox_male_adult_01/unknown").status_code == 404
    assert client.get("/media/../front").status_code in {404, 405}


def test_asset_url_must_match_the_manifest_asset_id(workspace):
    manifest_path = workspace / "rocketbox_male_adult_01" / "retarget_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["asset_id"] = "rocketbox_female_adult_01"
    manifest["binding"]["target_asset_id"] = "rocketbox_female_adult_01"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert _client(workspace).get("/asset/rocketbox_male_adult_01").status_code == 409


def test_decision_uses_explicit_reviewer_and_replaces_previous_decision(workspace):
    client = _client(workspace)

    response = client.post(
        "/decision/rocketbox_male_adult_01",
        data={"decision": "approved", "reviewer": "  reviewer-a  ", "notes": "looks good"},
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/asset/rocketbox_male_adult_01")
    review_path = workspace / "rocketbox_male_adult_01" / "motion_review.json"
    assert json.loads(review_path.read_text(encoding="utf-8"))["reviewer"] == "reviewer-a"

    client.post(
        "/decision/rocketbox_male_adult_01",
        data={"decision": "rejected", "notes": "frame 18"},
    )
    record = json.loads(review_path.read_text(encoding="utf-8"))
    assert record["decision"] == "rejected"
    assert record["notes"] == "frame 18"


def test_decision_defaults_reviewer_to_current_user(workspace, monkeypatch):
    import rocketbox_motion_review_server

    monkeypatch.setattr(rocketbox_motion_review_server.getpass, "getuser", lambda: "local-user")

    response = _client(workspace).post(
        "/decision/rocketbox_male_adult_01",
        data={"decision": "approved", "reviewer": "   "},
    )

    assert response.status_code == 302
    record = json.loads(
        (workspace / "rocketbox_male_adult_01" / "motion_review.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["reviewer"] == "local-user"


def test_gate_reports_locked_pending_pair_then_approved_pair(workspace):
    from rocketbox_motion_review import record_decision

    client = _client(workspace)
    locked = client.get("/gate")

    assert locked.status_code == 200
    assert locked.get_json()["state"] == "locked"

    for asset_id in EXPECTED_ASSET_IDS:
        record_decision(workspace / asset_id, "approved", "reviewer", "approved")

    approved = client.get("/gate")
    assert approved.status_code == 200
    assert approved.get_json()["state"] == "approved"
