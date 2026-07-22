"""Static contract for generated quadruped support-plane leveling."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_level_generated_animal_support_plane.py"
)


def test_support_plane_leveling_is_semantic_rigid_and_pre_animation():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "avengine_generated_animal_support_plane_leveling_v1" in text
    assert "SPEAR_ROOT = Path(__file__).resolve().parents[1]" in text
    assert "sys.path.insert(0, str(SPEAR_ROOT))" in text
    assert "infer_quadruped_semantics" in text
    assert "semantics.foot_leaves" in text
    assert "np.linalg.lstsq" in text
    assert "semantic feet do not define one support plane" in text
    assert "maximum_residual_ratio > args.maximum_foot_plane_residual_ratio" in text
    assert "normal.rotation_difference(up)" in text
    assert "support-plane leveling must run before animation" in text
    assert "tilt_deg > args.maximum_tilt_deg" in text
    assert "root.matrix_world = transform @ root.matrix_world" in text
    assert '"mesh_topology_changed": False' in text
    assert '"skeleton_hierarchy_changed": False' in text
    assert '"skin_weights_changed": False' in text
    assert "refusing to replace" in text
