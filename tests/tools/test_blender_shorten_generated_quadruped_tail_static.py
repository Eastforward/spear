"""Static authority checks for generated-mesh local tail shortening."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/blender_shorten_generated_quadruped_tail.py"


def source():
    return SCRIPT.read_text(encoding="utf-8")


def test_tail_shortener_is_taxonomy_independent_and_uses_no_template_geometry():
    text = source()

    assert "map_point" in text
    assert "generated_input_geometry_authority" in text
    assert '"template_geometry_used": False' in text
    assert "mesh_elements_added" in text
    assert "mesh_elements_deleted" in text
    assert "breed" not in text.lower()
    assert "corgi" not in text.lower()
    assert "quaternius" not in text.lower()


def test_tail_shortener_authenticates_inputs_and_fails_closed_on_edit_scope():
    text = source()

    assert "--input-sha256" in text
    assert "--profile-sha256" in text
    assert "os.O_NOFOLLOW" in text
    assert "stage_authenticated_input" in text
    assert "profile source identity does not match authenticated input" in text
    assert "profile.minimum_moved_vertices" in text
    assert "profile.maximum_moved_vertex_fraction" in text
    assert "a non-tail vertex position changed" in text
    assert "refusing to replace" in text


def test_tail_shortener_preserves_topology_uv_pbr_and_rump_continuity():
    text = source()

    assert "mesh_topology_sha256" in text
    assert "uv_sha256" in text
    assert "material_pbr_sha256" in text
    assert "topology_stats" in text
    assert "rump_surface_opening_introduced" in text
    assert "boundary_and_nonmanifold_counts_preserved_in_memory" in text
    assert "pbr_readback" in text
    assert '"formal_dataset_registration_authorized": False' in text
