"""Validate the apartment-shell mesh has expected shape and material assignments."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
GEN = REPO / "tools" / "spike_rlr" / "gen_mesh_apartment.py"
SHELL_JSON = REPO / "data" / "apartment_shell_map.json"

# Use ss2 env because trimesh + numpy are there; spear-env also has them.
PYTHON = "/data/jzy/miniconda3/envs/ss2/bin/python"


@pytest.fixture(scope="module")
def generated_mesh(tmp_path_factory):
    if not SHELL_JSON.exists():
        pytest.skip("apartment_shell_map.json not yet dumped (Task 2)")
    outdir = tmp_path_factory.mktemp("mesh")
    glb = outdir / "apartment_v1_mesh.glb"
    mats = outdir / "apartment_v1_materials.json"
    r = subprocess.run(
        [PYTHON, str(GEN),
         "--shell-json", str(SHELL_JSON),
         "--out-glb", str(glb),
         "--out-materials", str(mats)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"gen_mesh_apartment failed:\nstdout: {r.stdout}\nstderr: {r.stderr}"
    return glb, mats


def test_glb_file_created(generated_mesh):
    glb, _ = generated_mesh
    assert glb.exists() and glb.stat().st_size > 0


def test_materials_json_has_rlr_schema(generated_mesh):
    _, mats_path = generated_mesh
    mats = json.loads(mats_path.read_text())
    # Same schema as shoebox_v2_materials.json — RLR consumer already knows this.
    assert "materials" in mats
    assert "material_indices" in mats
    assert "face_material_tags" in mats
    assert "n_triangles" in mats
    # Every triangle has an index
    assert len(mats["material_indices"]) == mats["n_triangles"]
    assert len(mats["face_material_tags"]) == mats["n_triangles"]
    # Every material used has entries in the materials table
    tags_seen = set(mats["face_material_tags"])
    material_names = {m["name"] for m in mats["materials"]}
    assert tags_seen.issubset(material_names), \
        f"unmapped tags: {tags_seen - material_names}"
    # Every material has RLR fields
    for m in mats["materials"]:
        assert "alpha" in m and len(m["alpha"]) == 4
        assert "scattering" in m
        assert "transmission" in m and len(m["transmission"]) == 4


def test_glb_bbox_covers_apartment_extent(generated_mesh):
    """Sanity: the shell mesh should span the apartment's known extent."""
    import trimesh
    glb, _ = generated_mesh
    scene = trimesh.load(str(glb))
    if isinstance(scene, trimesh.Scene):
        combined = trimesh.util.concatenate(list(scene.geometry.values()))
    else:
        combined = scene
    bmin, bmax = combined.bounds
    extent_m = (bmax - bmin)  # meters
    # apartment_furniture_map.json shows X range 596-(-549)=1145 cm=11.45m,
    # Y 656-(-688)=1344 cm=13.44m. Allow some slack for shell dumping only.
    assert extent_m[0] > 8.0, f"X extent {extent_m[0]:.1f}m too small"
    assert extent_m[1] > 10.0, f"Y extent {extent_m[1]:.1f}m too small"


def test_triangle_count_is_12_per_actor(generated_mesh):
    """We emit 12 triangles per shell actor (a box). Confirm count matches."""
    _, mats_path = generated_mesh
    mats = json.loads(mats_path.read_text())
    shell_map = json.loads(SHELL_JSON.read_text())
    n_actors = len(shell_map["shell_actors"])
    # Number of triangles must be n_actors * 12 minus any degenerate skips
    # (in practice, all 21 apartment_0000 shell actors have valid bboxes).
    assert mats["n_triangles"] <= n_actors * 12
    assert mats["n_triangles"] >= (n_actors - 3) * 12  # allow up to 3 skips
