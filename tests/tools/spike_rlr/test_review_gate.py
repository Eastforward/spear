"""Tests for tools/spike_rlr/review_gate.py."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def _make_direction_json(tag_dir, algorithm_version="auto_orient_v1",
                          human_approved=True, quarantined=False):
    tag_dir.mkdir(parents=True, exist_ok=True)
    (tag_dir / "direction.json").write_text(json.dumps({
        "algorithm_version": algorithm_version,
        "human_approved": human_approved,
        "human_approved_by": "test",
        "human_approved_at": "2026-07-08T00:00:00Z",
        "quarantined": quarantined,
        "detection": {"head_direction_original_mesh_frame": [1, 0, 0],
                       "confidence": 0.9, "signals": {}, "total_votes": 3,
                       "unanimous": True,
                       "rotation_applied_to_align_to_plus_x": [[1,0,0],[0,1,0],[0,0,1]]},
    }))


def test_approved_tag_returns_direction_dict(tmp_path):
    from review_gate import assert_mesh_approved
    approved = tmp_path / "approved"
    _make_direction_json(approved / "dog_x", human_approved=True)
    d = assert_mesh_approved("dog_x", approved_dir=approved)
    assert d["human_approved"] is True


def test_unapproved_tag_raises(tmp_path):
    from review_gate import assert_mesh_approved, MeshNotApprovedError
    approved = tmp_path / "approved"
    _make_direction_json(approved / "dog_x", human_approved=False)
    with pytest.raises(MeshNotApprovedError, match="human_approved"):
        assert_mesh_approved("dog_x", approved_dir=approved)


def test_missing_tag_raises(tmp_path):
    from review_gate import assert_mesh_approved, MeshNotApprovedError
    approved = tmp_path / "approved"
    approved.mkdir()
    with pytest.raises(MeshNotApprovedError, match="not found"):
        assert_mesh_approved("missing", approved_dir=approved)


def test_stale_algorithm_version_raises(tmp_path):
    from review_gate import assert_mesh_approved, MeshNotApprovedError
    approved = tmp_path / "approved"
    _make_direction_json(approved / "dog_x", algorithm_version="auto_orient_v0")
    with pytest.raises(MeshNotApprovedError, match="algorithm_version"):
        assert_mesh_approved("dog_x", approved_dir=approved,
                              required_algorithm_version="auto_orient_v1")


def test_quarantined_tag_raises(tmp_path):
    from review_gate import assert_mesh_approved, MeshNotApprovedError
    approved = tmp_path / "approved"
    _make_direction_json(approved / "dog_x", quarantined=True)
    with pytest.raises(MeshNotApprovedError, match="quarantine"):
        assert_mesh_approved("dog_x", approved_dir=approved)


def test_actionable_error_message(tmp_path):
    from review_gate import assert_mesh_approved, MeshNotApprovedError
    approved = tmp_path / "approved"
    _make_direction_json(approved / "dog_x", human_approved=False)
    try:
        assert_mesh_approved("dog_x", approved_dir=approved)
    except MeshNotApprovedError as e:
        assert "review_ui_server" in str(e), \
            "error message should tell user how to fix (open review UI)"
