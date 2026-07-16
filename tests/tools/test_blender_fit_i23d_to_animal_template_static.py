from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/blender_fit_i23d_to_animal_template.py"


def source():
    return SCRIPT.read_text(encoding="utf-8")


def test_uses_only_explicit_cardinal_front_axes():
    text = source()
    assert '"negative-y": 90.0' in text
    assert '"positive-y": -90.0' in text
    assert '"fine_yaw_inference": False' in text


def test_preserves_stable_template_rig_and_actions():
    text = source()
    assert "BreedTemplateSubdivision" in text
    assert "weld_target_position_duplicates" in text
    assert '"position_weld": position_weld' in text
    assert "weighted_vertices" in text
    assert "keep_canonical_walk_idle_actions" in text
    assert 'export_skins=True' in text
    assert 'canonical_walk_idle=True' in text


def test_projects_guide_pbr_without_copying_generated_topology():
    text = source()
    assert "fit_surface_and_uv" in text
    assert "I23D_Projected_UV" in text
    assert "I23D_Breed_Appearance" in text
    assert "realize_guide_textures" in text
    assert '"realized_texture_files"' in text
    assert '"rerig_per_colour_instance": False' in text


def test_default_appearance_transfer_bakes_to_template_uv():
    text = source()
    assert (
        'choices=("vertex-color", "region-atlas", "bake", "projected-uv")'
        in text
    )
    assert 'default="vertex-color"' in text
    assert "bake_guide_material_to_template" in text
    assert 'use_selected_to_active=True' in text
    assert '"cycles_selected_to_active_template_uv_v1"' in text
    assert "preserve_template_uv_for_selected_to_active_bake" in text
    assert "ensure_template_bake_uv" in text
    assert "deterministic_smart_project" in text


def test_region_atlas_uses_semantic_correspondence_not_cross_surface_rays():
    text = source()
    assert "bake_region_sampled_guide_to_template" in text
    assert '"semantic_nearest_surface_template_uv_atlas_v1"' in text
    assert (
        '"nearest_guide_triangle_within_matching_semantic_region"'
        in text
    )
    assert 'type="EMIT"' in text
    assert "use_selected_to_active=False" in text
    assert 'name="I23D_Region_Sampled_BaseColor"' in text


def test_vertex_color_transfer_avoids_generated_topology_and_uv_islands():
    text = source()
    assert "install_guide_vertex_color_material" in text
    assert "bilinear_sample_image" in text
    assert 'name="I23D_Breed_VertexColor"' in text
    assert 'domain="CORNER"' in text
    assert '"nearest_surface_region_vertex_color_v1"' in text


def test_can_lock_all_four_template_limbs_while_fitting_axial_regions():
    text = source()
    assert 'choices=("all", "axial-only")' in text
    assert "geometry_fit_mode == \"axial-only\"" in text
    assert '"front_left_leg"' in text
    assert '"front_right_leg"' in text
    assert '"hind_left_leg"' in text
    assert '"hind_right_leg"' in text
    assert '"locked_limb_vertices"' in text
    assert '"preserve_subdivided_template_rest_geometry_and_native_weights"' in text


def test_refuses_to_overwrite_outputs():
    text = source()
    assert "refusing to replace" in text
    assert 'manifest_path.open("x"' in text
