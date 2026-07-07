"""Tests for the Flask review UI server.

Uses Flask's built-in test_client to avoid needing a real port. Focuses on
routing + state transitions (pending -> approved / rejected).
"""
import json
import shutil
import sys
from pathlib import Path

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


def test_root_lists_pending(workspace):
    from review_ui_server import create_app
    app = create_app(workspace["pending"], workspace["approved"], workspace["rejected"])
    client = app.test_client()
    r = client.get("/")
    assert r.status_code == 200
    assert b"dog_test_srv" in r.data
    assert b"Approve" in r.data


def test_preview_png_served(workspace):
    from review_ui_server import create_app
    app = create_app(workspace["pending"], workspace["approved"], workspace["rejected"])
    client = app.test_client()
    r = client.get("/preview/dog_test_srv.png")
    assert r.status_code == 200
    assert r.data[:4] == b"\x89PNG"


def test_approve_moves_to_approved(workspace):
    from review_ui_server import create_app
    app = create_app(workspace["pending"], workspace["approved"], workspace["rejected"])
    client = app.test_client()
    r = client.post("/approve/dog_test_srv")
    assert r.status_code in (200, 302)
    # Tag should have moved
    assert not (workspace["pending"] / "dog_test_srv").exists()
    approved_tag = workspace["approved"] / "dog_test_srv"
    assert approved_tag.exists()
    dj = json.loads((approved_tag / "direction.json").read_text())
    assert dj["human_approved"] is True
    assert dj["human_approved_by"] is not None
    assert dj["human_approved_at"] is not None


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


def test_override_records_human_override(workspace):
    from review_ui_server import create_app
    app = create_app(workspace["pending"], workspace["approved"], workspace["rejected"])
    client = app.test_client()
    r = client.post("/override/dog_test_srv",
                    data={"correct_direction_x": "-1",
                          "correct_direction_y": "0",
                          "correct_direction_z": "0",
                          "reason": "auto detected reverse"})
    assert r.status_code in (200, 302)
    rejected_tag = workspace["rejected"] / "dog_test_srv"
    assert rejected_tag.exists()
    dj = json.loads((rejected_tag / "direction.json").read_text())
    assert dj["human_override"] is not None
    assert dj["human_override"]["correct_head_direction_in_original_mesh"] == [-1.0, 0.0, 0.0]
    assert dj["human_override"]["reason"] == "auto detected reverse"


def test_missing_tag_returns_404(workspace):
    from review_ui_server import create_app
    app = create_app(workspace["pending"], workspace["approved"], workspace["rejected"])
    client = app.test_client()
    r = client.post("/approve/nonexistent_tag")
    assert r.status_code == 404
