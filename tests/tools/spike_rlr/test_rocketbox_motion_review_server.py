"""Tests for the dedicated Rocketbox motion review Flask application."""

from __future__ import annotations

import json
import re
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


def _csrf_token(client) -> str:
    response = client.get("/asset/rocketbox_male_adult_01")
    assert response.status_code == 200
    match = re.search(rb'name="csrf_token" value="([^"]+)"', response.data)
    assert match is not None
    return match.group(1).decode("ascii")


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
    assert b'name="csrf_token"' in response.data
    assert b"HttpOnly" in response.headers["Set-Cookie"].encode()
    assert b"SameSite=Strict" in response.headers["Set-Cookie"].encode()


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


def test_unknown_asset_get_and_post_return_404(workspace):
    client = _client(workspace)
    csrf_token = _csrf_token(client)

    assert client.get("/asset/not-a-rocketbox-asset").status_code == 404
    assert (
        client.post(
            "/decision/not-a-rocketbox-asset",
            data={
                "decision": "approved",
                "reviewer": "reviewer",
                "csrf_token": csrf_token,
            },
        ).status_code
        == 404
    )


def test_invalid_decision_with_valid_csrf_does_not_replace_review_record(workspace):
    client = _client(workspace)
    csrf_token = _csrf_token(client)
    review_path = workspace / "rocketbox_male_adult_01" / "motion_review.json"
    before = review_path.read_bytes()

    response = client.post(
        "/decision/rocketbox_male_adult_01",
        data={
            "decision": "invalid",
            "reviewer": "reviewer",
            "csrf_token": csrf_token,
        },
    )

    assert response.status_code == 400
    assert review_path.read_bytes() == before


def test_media_route_rejects_manifest_mapping_front_to_manifest_file(workspace):
    manifest_path = workspace / "rocketbox_male_adult_01" / "retarget_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["media"]["front"] = "retarget_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    response = _client(workspace).get("/media/rocketbox_male_adult_01/front")

    assert response.status_code == 409


def test_media_route_rejects_front_media_symlink(workspace):
    asset_dir = workspace / "rocketbox_male_adult_01"
    (asset_dir / "front.mp4").unlink()
    (asset_dir / "front.mp4").symlink_to("side.mp4")

    response = _client(workspace).get("/media/rocketbox_male_adult_01/front")

    assert response.status_code == 409


def test_decision_uses_explicit_reviewer_and_replaces_previous_decision(workspace):
    client = _client(workspace)
    csrf_token = _csrf_token(client)

    response = client.post(
        "/decision/rocketbox_male_adult_01",
        data={
            "decision": "approved",
            "reviewer": "  reviewer-a  ",
            "notes": "looks good",
            "csrf_token": csrf_token,
        },
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/asset/rocketbox_male_adult_01")
    review_path = workspace / "rocketbox_male_adult_01" / "motion_review.json"
    assert json.loads(review_path.read_text(encoding="utf-8"))["reviewer"] == "reviewer-a"

    client.post(
        "/decision/rocketbox_male_adult_01",
        data={
            "decision": "rejected",
            "reviewer": "reviewer-b",
            "notes": "frame 18",
            "csrf_token": csrf_token,
        },
    )
    record = json.loads(review_path.read_text(encoding="utf-8"))
    assert record["decision"] == "rejected"
    assert record["notes"] == "frame 18"


def test_asset_prefills_current_os_user_when_review_has_no_reviewer(workspace, monkeypatch):
    import rocketbox_motion_review_server

    monkeypatch.setattr(rocketbox_motion_review_server.getpass, "getuser", lambda: "local-user")

    client = _client(workspace)
    response = client.get("/asset/rocketbox_male_adult_01")

    assert response.status_code == 200
    assert (
        b'<input name="reviewer" value="local-user" placeholder="'
        in response.data
    )


def test_decision_rejects_blank_reviewer_with_valid_csrf_token(workspace):
    client = _client(workspace)
    csrf_token = _csrf_token(client)
    review_path = workspace / "rocketbox_male_adult_01" / "motion_review.json"
    before = review_path.read_bytes()

    response = client.post(
        "/decision/rocketbox_male_adult_01",
        data={"decision": "approved", "reviewer": "   ", "csrf_token": csrf_token},
    )

    assert response.status_code == 400
    assert review_path.read_bytes() == before


def test_decision_rejects_missing_csrf_token_without_writing_a_review(workspace):
    client = _client(workspace)

    response = client.post(
        "/decision/rocketbox_male_adult_01",
        data={"decision": "approved", "reviewer": "cross-site"},
    )

    assert response.status_code == 400
    assert not (workspace / "rocketbox_male_adult_01" / "motion_review.json").exists()


def test_decision_rejects_invalid_csrf_token_without_writing_a_review(workspace):
    client = _client(workspace)
    _csrf_token(client)
    review_path = workspace / "rocketbox_male_adult_01" / "motion_review.json"
    before = review_path.read_bytes()

    response = client.post(
        "/decision/rocketbox_male_adult_01",
        data={"decision": "approved", "csrf_token": "not-the-session-token"},
    )

    assert response.status_code == 400
    assert review_path.read_bytes() == before


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
