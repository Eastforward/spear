"""Static contract for generated-animal cardinal heading normalization."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_normalize_generated_animal_heading.py"
)


def test_heading_normalization_is_rigid_review_driven_and_fail_closed():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "avengine_generated_animal_heading_normalization_v1" in text
    assert '"--reviewed-source-front-yaw-deg"' in text
    assert '"--review-evidence"' in text
    assert '"skinned_mesh_count"' in text
    assert '"skinned_meshes"' in text
    assert "helper meshes are allowed" in text
    assert 'Matrix.Rotation(math.radians(delta_yaw), 4, "Z")' in text
    assert "root.matrix_world = rotation @ root.matrix_world" in text
    assert '"mesh_topology_changed": False' in text
    assert '"skeleton_hierarchy_changed": False' in text
    assert '"skin_weights_changed": False' in text
    assert "refusing to replace" in text
    assert "technical_spike_only_pending_reaudit_and_animation_qa" in text
