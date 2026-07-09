"""Tests for the Flask review UI server (v2 — single-card + rotate).

Uses Flask's built-in test_client to avoid needing a real port.
"""
import json
import os
import shutil
import sys
from pathlib import Path
import hashlib

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
sys.path.insert(0, str(REPO / "tests" / "tools" / "spike_rlr"))


@pytest.fixture
def workspace(tmp_path):
    """Set up pending/approved/rejected with one pending tag."""
    from auto_orient_ingest import process_one
    from test_auto_orient_ingest import _write_synth_pending

    pending = tmp_path / "pending"
    approved = tmp_path / "approved"
    rejected = tmp_path / "rejected"
    for d in (pending, approved, rejected):
        d.mkdir(parents=True)

    tag_dir = _write_synth_pending(pending, "dog_test_srv", head_axis="+X")
    process_one(tag_dir)
    return {"pending": pending, "approved": approved, "rejected": rejected}


def test_root_redirects_to_first_pending_tag(workspace):
    from review_ui_server import create_app
    app = create_app(workspace["pending"], workspace["approved"], workspace["rejected"])
    client = app.test_client()
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 301)
    assert "/tag/dog_test_srv" in r.headers["Location"]


def test_tag_view_renders_single_card(workspace):
    from review_ui_server import create_app
    app = create_app(workspace["pending"], workspace["approved"], workspace["rejected"])
    client = app.test_client()
    r = client.get("/tag/dog_test_srv")
    assert r.status_code == 200
    assert b"dog_test_srv" in r.data
    assert b"Approve" in r.data
    # spatial rotate buttons (positioned around preview via CSS grid)
    assert b"yaw -90" in r.data or b"yaw &minus;90" in r.data or b"yaw \xe2\x88\x9290" in r.data
    assert b"yaw +90" in r.data
    assert b"flip 180" in r.data
    assert b"reset" in r.data
    # absolute mesh path shown for reviewer's benefit
    assert b"mesh.glb" in r.data


def test_tag_view_shows_absolute_paths(workspace):
    from review_ui_server import create_app
    app = create_app(workspace["pending"], workspace["approved"], workspace["rejected"])
    client = app.test_client()
    r = client.get("/tag/dog_test_srv")
    # Both pending dir and mesh path should appear as absolute paths
    pending_abs = str(workspace["pending"].resolve())
    assert pending_abs.encode() in r.data


def test_preview_png_served(workspace):
    from review_ui_server import create_app
    app = create_app(workspace["pending"], workspace["approved"], workspace["rejected"])
    client = app.test_client()
    r = client.get("/preview/dog_test_srv.png")
    assert r.status_code == 200
    assert r.data[:4] == b"\x89PNG"


def test_tag_view_regenerates_stale_review_preview(workspace):
    from review_ui_server import create_app
    app = create_app(workspace["pending"], workspace["approved"], workspace["rejected"])
    client = app.test_client()
    tag_dir = workspace["pending"] / "dog_test_srv"
    preview = tag_dir / "direction_preview_review.png"

    assert client.get("/tag/dog_test_srv").status_code == 200
    assert preview.exists()
    os.utime(preview, (1, 1))

    assert client.get("/tag/dog_test_srv").status_code == 200

    assert preview.stat().st_mtime > 1


def test_rotate_updates_rotation_json_and_regens_preview(workspace):
    from review_ui_server import create_app
    app = create_app(workspace["pending"], workspace["approved"], workspace["rejected"])
    client = app.test_client()

    # First rotate — 90° about Y
    r = client.post("/rotate/dog_test_srv",
                     data={"axis": "y", "deg": "90"},
                     follow_redirects=False)
    assert r.status_code == 302
    assert (workspace["pending"] / "dog_test_srv" / "rotation.json").exists()
    rot = json.loads((workspace["pending"] / "dog_test_srv" / "rotation.json").read_text())
    R = np.array(rot["matrix"])
    # 90° rotation about Y: R @ (1,0,0) -> (0, 0, -1)
    assert np.allclose(R @ np.array([1, 0, 0]), np.array([0, 0, -1]), atol=1e-9)
    assert rot["history"] == ["y+90"]

    # Second rotate — another 90° about Y → composed = 180°
    client.post("/rotate/dog_test_srv", data={"axis": "y", "deg": "90"})
    rot = json.loads((workspace["pending"] / "dog_test_srv" / "rotation.json").read_text())
    R = np.array(rot["matrix"])
    # 180° about Y: (1,0,0) -> (-1, 0, 0)
    assert np.allclose(R @ np.array([1, 0, 0]), np.array([-1, 0, 0]), atol=1e-9)
    assert rot["history"] == ["y+90", "y+90"]


def test_reset_clears_rotation(workspace):
    from review_ui_server import create_app
    app = create_app(workspace["pending"], workspace["approved"], workspace["rejected"])
    client = app.test_client()
    client.post("/rotate/dog_test_srv", data={"axis": "y", "deg": "90"})
    r = client.post("/reset/dog_test_srv", follow_redirects=False)
    assert r.status_code == 302
    rot = json.loads((workspace["pending"] / "dog_test_srv" / "rotation.json").read_text())
    assert rot["history"] == []
    assert np.allclose(np.array(rot["matrix"]), np.eye(3))


def test_approve_bakes_rotation_and_moves_to_approved(workspace):
    from review_ui_server import create_app
    app = create_app(workspace["pending"], workspace["approved"], workspace["rejected"])
    client = app.test_client()

    # Rotate 180° so the mesh should now be head-at-negative-X (in mesh coords)
    client.post("/rotate/dog_test_srv", data={"axis": "y", "deg": "180"})
    r = client.post("/approve/dog_test_srv", follow_redirects=False)
    assert r.status_code == 302
    assert not (workspace["pending"] / "dog_test_srv").exists()
    approved_tag = workspace["approved"] / "dog_test_srv"
    assert approved_tag.exists()
    dj = json.loads((approved_tag / "direction.json").read_text())
    assert dj["human_approved"] is True
    assert dj["human_approved_by"] is not None
    assert dj["human_approved_at"] is not None
    # The baked rotation history should be persisted
    assert dj["human_applied_rotation_history"] == ["y+180"]
    # mesh_oriented.glb should have been rewritten
    mesh_oriented = approved_tag / "mesh_oriented.glb"
    assert mesh_oriented.exists()
    assert dj["mesh_source"].startswith(str(approved_tag))
    assert dj["mesh_oriented"] == str(mesh_oriented)
    assert dj["mesh_sha256"] == hashlib.sha256(mesh_oriented.read_bytes()).hexdigest()


def test_reject_moves_to_rejected(workspace):
    from review_ui_server import create_app
    app = create_app(workspace["pending"], workspace["approved"], workspace["rejected"])
    client = app.test_client()
    r = client.post("/reject/dog_test_srv", data={"reason": "test rejection"})
    assert r.status_code in (200, 302)
    assert not (workspace["pending"] / "dog_test_srv").exists()
    rejected_tag = workspace["rejected"] / "dog_test_srv"
    assert rejected_tag.exists()
    dj = json.loads((rejected_tag / "direction.json").read_text())
    assert dj["human_approved"] is False
    assert dj["human_notes"] == "test rejection"


def test_skip_leaves_in_pending(workspace):
    from review_ui_server import create_app
    app = create_app(workspace["pending"], workspace["approved"], workspace["rejected"])
    client = app.test_client()
    r = client.post("/skip/dog_test_srv", follow_redirects=False)
    assert r.status_code == 302
    # Still in pending
    assert (workspace["pending"] / "dog_test_srv").exists()


def test_missing_tag_approve_returns_404(workspace):
    from review_ui_server import create_app
    app = create_app(workspace["pending"], workspace["approved"], workspace["rejected"])
    client = app.test_client()
    r = client.post("/approve/nonexistent_tag")
    assert r.status_code == 404


def test_root_redirects_to_done_when_no_pending(workspace):
    """After approving the only pending tag, / shows a done page (not another card)."""
    from review_ui_server import create_app
    app = create_app(workspace["pending"], workspace["approved"], workspace["rejected"])
    client = app.test_client()
    client.post("/approve/dog_test_srv")
    r = client.get("/")
    assert r.status_code == 200
    assert b"All pending meshes reviewed" in r.data or b"reviewed" in r.data
