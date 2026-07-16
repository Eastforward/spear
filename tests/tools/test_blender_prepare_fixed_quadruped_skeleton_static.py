from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "blender_prepare_fixed_quadruped_skeleton.py"
)


def test_fixed_skeleton_conditioner_is_single_root_and_preserves_generated_mesh():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "avengine_fixed_quadruped_skeleton_conditioning_v3" in text
    assert "positive_weight_bones" in text
    assert "load_reference_deform_authority" in text
    assert '"--reference-rig"' in text
    assert "reference_deform_authority_plus_positive_weight_ancestors_v3" in text
    assert "temporary_transfer_weighted_bones" in text
    assert "reference_required_weighted_bones" in text
    assert "include_ancestors" in text
    assert "parent_retained_roots_under_main" in text
    assert '"--main-root"' in text
    assert "reparented_weight_bearing_roots" in text
    assert "reparenting_preserved_armature_space_rest_coordinates" in text
    assert "len(roots_after) != 1" in text
    assert "remove_unretained_bones" in text
    assert "normalize_armature_name" in text
    assert 'default="Armature"' in text
    assert "skintokens_use_origin_export_name_compatibility" in text
    assert '"generated_topology_preserved": True' in text
    assert '"animation_removed": True' in text
    assert '"next_stage": "skintokens_use_skeleton_use_transfer"' in text
    assert "export_animations=False" in text
    assert "export_skins=True" in text
    assert "bpy.ops.object.modifier_apply" not in text
