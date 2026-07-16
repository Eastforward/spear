"""Tests for gen_mesh.py — verifies mesh topology and material assignment.

These tests intentionally load the SSOT files (spec + db) as they exist in
the repo, not fake fixtures, because a change to shoebox_v2_spec.json is
what would break downstream A/B/C consumers and we want the test to catch it.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "data" / "shoebox_v2_spec.json"
DB_PATH = REPO_ROOT / "data" / "acoustic_material_db.json"
GEN_MESH_PATH = REPO_ROOT / "tools" / "spike_rlr" / "gen_mesh.py"


@pytest.fixture(scope="module")
def gen_mesh_module():
    spec = importlib.util.spec_from_file_location("gen_mesh", GEN_MESH_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def spec():
    with open(SPEC_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def db():
    with open(DB_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def built(gen_mesh_module, spec, db):
    mesh, tags = gen_mesh_module.build_room_mesh(spec, db)
    materials, indices = gen_mesh_module.build_rlr_materials(tags, db)
    return mesh, tags, materials, indices


def test_expected_triangle_count(built):
    """6 room walls × 2 tris + N_furniture × 6 faces × 2 tris."""
    mesh, tags, _, _ = built
    # spec has 1 furniture (sofa) -> 12 + 12 = 24 triangles
    assert len(mesh.faces) == 24, f"got {len(mesh.faces)} triangles"
    assert len(tags) == 24


def test_face_material_tags_align_with_faces(built):
    mesh, tags, _, indices = built
    assert len(tags) == len(mesh.faces)
    assert len(indices) == len(mesh.faces)


def test_unique_material_count(built, spec):
    """spec's surfaces have 3 distinct materials (drywall x4 walls, hardwood, plaster)
    + furniture sofa fabric -> 4 unique."""
    mesh, tags, materials, _ = built
    expected_unique = set(spec["surfaces"].values()) | {f["material"] for f in spec["furniture"]}
    got_unique = set(tags)
    assert got_unique == expected_unique, (
        f"tag set mismatch: got {got_unique}, expected {expected_unique}"
    )
    assert len(materials) == len(expected_unique)


def test_all_indices_in_range(built):
    _, _, materials, indices = built
    assert min(indices) >= 0
    assert max(indices) < len(materials)


def test_room_bounds_contain_all_room_faces(built, spec):
    """The room enclosure vertices should fill the [0, room_size] AABB."""
    mesh, tags, _, _ = built
    rs = spec["room_size_m"]
    # Grab the vertices of the first 12 triangles (the room walls)
    room_face_indices = [i for i, t in enumerate(tags) if not t.startswith("fabric_")]
    # actually, safer: room walls are the first 12 tris by construction
    room_face_indices = list(range(12))
    room_verts = mesh.vertices[mesh.faces[room_face_indices].flatten()]
    for axis in range(3):
        assert room_verts[:, axis].min() >= -1e-6
        assert room_verts[:, axis].max() <= rs[axis] + 1e-6
    # Room verts should span the full room
    for axis in range(3):
        assert room_verts[:, axis].max() - room_verts[:, axis].min() >= rs[axis] - 1e-6


def test_sofa_verts_inside_room(built, spec):
    mesh, tags, _, _ = built
    rs = spec["room_size_m"]
    # Sofa tris are indices 12..23 (after 12 room tris)
    sofa_tri_ids = list(range(12, 24))
    sofa_verts = mesh.vertices[mesh.faces[sofa_tri_ids].flatten()]
    for axis in range(3):
        assert sofa_verts[:, axis].min() >= -1e-6
        assert sofa_verts[:, axis].max() <= rs[axis] + 1e-6


def test_sofa_dimensions_match_spec(built, spec):
    mesh, tags, _, _ = built
    sofa = spec["furniture"][0]
    sofa_tri_ids = list(range(12, 24))
    sofa_verts = mesh.vertices[mesh.faces[sofa_tri_ids].flatten()]
    got_min = sofa_verts.min(axis=0)
    got_max = sofa_verts.max(axis=0)
    c = np.array(sofa["center_m"])
    s = np.array(sofa["size_m"])
    exp_min = c - s / 2
    exp_max = c + s / 2
    np.testing.assert_allclose(got_min, exp_min, atol=1e-5)
    np.testing.assert_allclose(got_max, exp_max, atol=1e-5)


def test_output_files_exist_after_main_run():
    out_dir = REPO_ROOT / "tmp" / "spike_rlr"
    assert (out_dir / "shoebox_v2_mesh.glb").exists()
    assert (out_dir / "shoebox_v2_materials.json").exists()
    assert (out_dir / "shoebox_v2_mesh.obj").exists()


def test_output_materials_json_shape():
    materials_path = REPO_ROOT / "tmp" / "spike_rlr" / "shoebox_v2_materials.json"
    with open(materials_path) as f:
        data = json.load(f)
    assert data["n_triangles"] == 24
    assert len(data["material_indices"]) == 24
    assert len(data["face_material_tags"]) == 24
    for mat in data["materials"]:
        assert len(mat["alpha"]) == 4
        assert len(mat["transmission"]) == 4
