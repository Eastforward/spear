"""End-to-end integration: synth dog mesh -> ingest -> approve -> gate ok.

Verifies the whole Plan 1.5.A pipeline holds together as a unit. Does NOT
run UE — only the Python side of the auto-orient/audit flow.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
sys.path.insert(0, str(REPO / "tests" / "tools" / "spike_rlr"))


def test_synth_dog_full_pipeline(tmp_path):
    from test_auto_orient_ingest import _write_synth_pending
    from review_gate import assert_mesh_approved, MeshNotApprovedError

    pending = tmp_path / "pending"
    approved = tmp_path / "approved"
    rejected = tmp_path / "rejected"
    for d in (pending, approved, rejected):
        d.mkdir()

    # 1. Simulate Hunyuan output
    tag_dir = _write_synth_pending(pending, "synth_dog_e2e", head_axis="+X")

    # 2. Run ingest
    r = subprocess.run(
        ["/data/jzy/miniconda3/envs/ss2/bin/python",
         str(REPO / "tools/spike_rlr/auto_orient_ingest.py"),
         "--pending-dir", str(pending)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr

    # 3. Gate should refuse (tag not yet moved to approved/ — either "not found"
    #    if still in pending/, or "human_approved=False" if pre-copied)
    with pytest.raises(MeshNotApprovedError):
        assert_mesh_approved("synth_dog_e2e", approved_dir=approved)

    # 4. Simulate human approval via the Flask app
    from review_ui_server import create_app
    app = create_app(pending, approved, rejected)
    client = app.test_client()
    resp = client.post("/approve/synth_dog_e2e")
    assert resp.status_code in (200, 302)

    # 5. Gate should now succeed
    d = assert_mesh_approved("synth_dog_e2e", approved_dir=approved)
    assert d["human_approved"] is True
    assert d["algorithm_version"] == "auto_orient_v1"

    # 6. Resolve path returns mesh_oriented.glb
    from review_gate import resolve_approved_mesh_path
    p = resolve_approved_mesh_path("synth_dog_e2e", approved_dir=approved)
    assert p.exists()
    assert p.name == "mesh_oriented.glb"
