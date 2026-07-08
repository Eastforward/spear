"""Tests for tools/spike_rlr/auto_orient_ingest.py."""
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
INGEST = REPO / "tools" / "spike_rlr" / "auto_orient_ingest.py"
PYTHON = "/data/jzy/miniconda3/envs/ss2/bin/python"


def _write_synth_pending(pending_root, tag, head_axis="+X"):
    """Create pending/{tag}/mesh.glb with known head direction."""
    sys.path.insert(0, str(REPO / "tests" / "tools" / "spike_rlr"))
    from test_detect_head_axis import _synth_dog
    import trimesh

    verts = _synth_dog(head_axis=head_axis)
    hull = trimesh.convex.convex_hull(verts)
    tag_dir = pending_root / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    out = tag_dir / "mesh.glb"
    hull.export(str(out))
    return tag_dir


def test_ingest_produces_direction_json_and_preview(tmp_path):
    pending = tmp_path / "pending"
    tag_dir = _write_synth_pending(pending, "dog_test_plus_x", head_axis="+X")

    r = subprocess.run(
        [PYTHON, str(INGEST), "--pending-dir", str(pending)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"ingest failed:\n{r.stdout}\n---\n{r.stderr}"

    # direction.json exists and has required fields
    dj = tag_dir / "direction.json"
    assert dj.exists()
    d = json.loads(dj.read_text())
    assert "algorithm_version" in d
    assert "detection" in d
    assert "head_direction_original_mesh_frame" in d["detection"]
    assert "rotation_applied_to_align_to_plus_x" in d["detection"]
    assert "signals" in d["detection"]
    assert "confidence" in d["detection"]
    assert d["human_approved"] is False
    assert d["human_approved_by"] is None
    assert d["human_approved_at"] is None

    # preview PNG exists
    assert (tag_dir / "direction_preview.png").exists()


def test_ingest_writes_oriented_mesh_with_head_at_plus_x(tmp_path):
    """After ingest, mesh_oriented.glb should have head along +X."""
    pending = tmp_path / "pending"
    tag_dir = _write_synth_pending(pending, "dog_test_minus_x", head_axis="-X")

    r = subprocess.run(
        [PYTHON, str(INGEST), "--pending-dir", str(pending)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"ingest failed:\n{r.stdout}\n---\n{r.stderr}"

    # Re-detect on the oriented mesh — should say head is at +X now
    import trimesh
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from detect_head_axis import detect_head_axis

    m = trimesh.load(str(tag_dir / "mesh_oriented.glb"))
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    result = detect_head_axis(np.array(m.vertices))
    # Head should now be along +X
    assert result.head_direction[0] > 0.7, \
        f"oriented mesh head still not at +X: {result.head_direction}"


def test_ingest_skips_existing_direction_json(tmp_path):
    """If direction.json already exists, ingest should skip that tag by default."""
    pending = tmp_path / "pending"
    tag_dir = _write_synth_pending(pending, "dog_test_skip", head_axis="+X")
    # First run
    r1 = subprocess.run([PYTHON, str(INGEST), "--pending-dir", str(pending)],
                         capture_output=True, text=True)
    assert r1.returncode == 0
    mtime1 = (tag_dir / "direction.json").stat().st_mtime
    # Second run should skip (default behavior)
    r2 = subprocess.run([PYTHON, str(INGEST), "--pending-dir", str(pending)],
                         capture_output=True, text=True)
    assert r2.returncode == 0
    mtime2 = (tag_dir / "direction.json").stat().st_mtime
    assert mtime1 == mtime2, "direction.json was rewritten despite existing"


def test_ingest_help_shows_expected_flags():
    r = subprocess.run([PYTHON, str(INGEST), "--help"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "--pending-dir" in r.stdout
    assert "--force" in r.stdout  # for overwriting existing direction.json


def test_auto_orient_caps_numeric_threads_before_numpy_import():
    text = INGEST.read_text()

    assert text.index('os.environ.setdefault("OMP_NUM_THREADS", "1")') < text.index(
        "import numpy as np"
    )
    assert text.index('os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")') < text.index(
        "import numpy as np"
    )


def test_ingest_preserves_obj_uvs_in_oriented_mesh(tmp_path):
    sys.path.insert(0, str(REPO / "tests" / "tools" / "spike_rlr"))
    from test_detect_head_axis import _synth_dog

    pending = tmp_path / "pending"
    tag_dir = pending / "dog_uv"
    tag_dir.mkdir(parents=True)
    verts = _synth_dog(head_axis="+X")
    obj = tag_dir / "mesh.obj"
    lines = ["mtllib mesh.mtl", "o dog_uv"]
    for x, y, z in verts:
        lines.append(f"v {x:.6f} {y:.6f} {z:.6f}")
    for _ in verts:
        lines.append("vt 0.5 0.5")
    for i in range(1, len(verts) - 2, 3):
        lines.append(f"f {i}/{i} {i+1}/{i+1} {i+2}/{i+2}")
    obj.write_text("\n".join(lines) + "\n")
    (tag_dir / "mesh.mtl").write_text("newmtl Material\nmap_Kd hy3d_diffuse.jpg\n")
    (tag_dir / "hy3d_diffuse.jpg").write_bytes(b"not a real jpg but referenced")

    r = subprocess.run(
        [PYTHON, str(INGEST), "--pending-dir", str(pending)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"ingest failed:\n{r.stdout}\n---\n{r.stderr}"

    glb = (tag_dir / "mesh_oriented.glb").read_bytes()
    assert glb[:4] == b"glTF"
    json_len, json_type = struct.unpack_from("<II", glb, 12)
    assert json_type == 0x4E4F534A
    payload = json.loads(glb[20:20 + json_len].decode("utf-8"))
    attrs = [
        prim.get("attributes", {})
        for mesh in payload.get("meshes", [])
        for prim in mesh.get("primitives", [])
    ]

    assert any("TEXCOORD_0" in attr for attr in attrs)
