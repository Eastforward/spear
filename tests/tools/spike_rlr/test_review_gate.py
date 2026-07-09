"""Tests for tools/spike_rlr/review_gate.py."""
import json
import sys
from pathlib import Path
import hashlib

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def _make_direction_json(tag_dir, algorithm_version="auto_orient_v1",
                          human_approved=True, quarantined=False,
                          mesh_bytes=b"test mesh"):
    tag_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = tag_dir / "mesh_oriented.glb"
    mesh_path.write_bytes(mesh_bytes)
    mesh_sha256 = hashlib.sha256(mesh_bytes).hexdigest()
    (tag_dir / "direction.json").write_text(json.dumps({
        "algorithm_version": algorithm_version,
        "human_approved": human_approved,
        "human_approved_by": "test",
        "human_approved_at": "2026-07-08T00:00:00Z",
        "quarantined": quarantined,
        "mesh_oriented": str(mesh_path),
        "mesh_sha256": mesh_sha256,
        "detection": {"head_direction_original_mesh_frame": [1, 0, 0],
                       "confidence": 0.9, "signals": {}, "total_votes": 3,
                       "unanimous": True,
                       "rotation_applied_to_align_to_plus_x": [[1,0,0],[0,1,0],[0,0,1]]},
    }))


def test_runtime_proxy_default_is_midpoly_not_tiny():
    from runtime_proxy_mesh import DEFAULT_TARGET_FACES
    assert DEFAULT_TARGET_FACES == 40000


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


def test_approved_mesh_record_returns_canonical_mesh_and_hash(tmp_path):
    from review_gate import approved_mesh_record
    approved = tmp_path / "approved"
    _make_direction_json(approved / "dog_x", mesh_bytes=b"canonical")

    rec = approved_mesh_record("dog_x", approved_dir=approved)

    assert rec["tag"] == "dog_x"
    assert rec["mesh_path"] == approved / "dog_x" / "mesh_oriented.glb"
    assert rec["mesh_sha256"] == hashlib.sha256(b"canonical").hexdigest()
    assert rec["runtime_mesh_path"] is None


def test_approved_mesh_record_reports_valid_runtime_proxy(tmp_path):
    from review_gate import approved_mesh_record
    approved = tmp_path / "approved"
    tag_dir = approved / "dog_x"
    _make_direction_json(tag_dir, mesh_bytes=b"canonical")
    runtime = tag_dir / "mesh_runtime.glb"
    runtime.write_bytes(b"runtime")
    (tag_dir / "mesh_runtime.json").write_text(json.dumps({
        "algorithm": "blender_decimate_v1",
        "source_mesh_sha256": hashlib.sha256(b"canonical").hexdigest(),
        "target_faces": 80000,
        "actual_faces": 79999,
        "runtime_mesh_sha256": hashlib.sha256(b"runtime").hexdigest(),
    }))

    rec = approved_mesh_record("dog_x", approved_dir=approved)

    assert rec["mesh_path"] == tag_dir / "mesh_oriented.glb"
    assert rec["runtime_mesh_path"] == runtime
    assert rec["runtime_mesh_sha256"] == hashlib.sha256(b"runtime").hexdigest()


def test_approved_mesh_record_ignores_stale_runtime_proxy(tmp_path):
    from review_gate import approved_mesh_record
    approved = tmp_path / "approved"
    tag_dir = approved / "dog_x"
    _make_direction_json(tag_dir, mesh_bytes=b"canonical")
    (tag_dir / "mesh_runtime.glb").write_bytes(b"runtime")
    (tag_dir / "mesh_runtime.json").write_text(json.dumps({
        "algorithm": "blender_decimate_v1",
        "source_mesh_sha256": "0" * 64,
        "target_faces": 80000,
        "actual_faces": 79999,
    }))

    rec = approved_mesh_record("dog_x", approved_dir=approved)

    assert rec["runtime_mesh_path"] is None


def test_approved_mesh_hash_mismatch_raises(tmp_path):
    from review_gate import approved_mesh_record, MeshNotApprovedError
    approved = tmp_path / "approved"
    _make_direction_json(approved / "dog_x", mesh_bytes=b"canonical")
    d = json.loads((approved / "dog_x" / "direction.json").read_text())
    d["mesh_sha256"] = "0" * 64
    (approved / "dog_x" / "direction.json").write_text(json.dumps(d))

    with pytest.raises(MeshNotApprovedError, match="sha256"):
        approved_mesh_record("dog_x", approved_dir=approved)
