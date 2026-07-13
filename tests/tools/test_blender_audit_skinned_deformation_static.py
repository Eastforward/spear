"""Static contracts for the species-independent skinned deformation gate."""

from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "tools/blender_audit_skinned_deformation.py"


def test_deformation_gate_samples_rest_walk_and_idle_without_mesh_edits():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"--input"' in text
    assert '"--output"' in text
    assert '"--action"' in text
    assert "armature.data.pose_position = \"REST\"" in text
    assert "evaluated_get" in text
    assert "to_mesh" in text
    assert "to_mesh_clear" in text
    assert "bpy.ops.export_scene" not in text


def test_deformation_gate_measures_visible_fan_and_membrane_stretch():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "edge_stretch_ratio" in text
    assert "edge_extension_ratio_of_rest_diagonal" in text
    assert "triangle_area_stretch_ratio" in text
    assert "surface_area_ratio_to_rest" in text
    assert "reject_visible_skinning_fan_or_membrane" in text
    assert "manual_review_local_deformation" in text


def test_deformation_gate_is_species_independent_and_records_thresholds():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "species" not in text.lower()
    assert "thresholds" in text


def test_default_review_extension_threshold_is_calibrated_above_authored_walk():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'default=0.07' in text
    assert 'default=0.08' in text
    assert "sampled_frames" in text
    assert "input_sha256" in text
    assert "formal_dataset_registration_authorized" in text
