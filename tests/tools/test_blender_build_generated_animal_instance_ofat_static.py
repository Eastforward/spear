from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_build_generated_animal_instance_ofat.py"
)


def test_generated_instance_ofat_keeps_generated_asset_authority():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"research_candidate"' in text
    assert '"formal_dataset_registration_authorized": False' in text
    assert '"actual_generated_mesh_preserved": True' in text
    assert '"template_geometry_used": False' in text
    assert "Quaternius" not in text
    assert 'animations != ["Idle", "Walking"]' in text
    assert '{"JOINTS_0", "WEIGHTS_0"}' in text
    assert "asset_authority_signature" in text
    assert "authority_signature_matches_baseline" in text
    assert "idle_walking_keyframe_digest" in text
    assert "stable.action_sha256" in text
    assert "changed protected asset authority" in text
    assert "differences = {" in text


def test_generated_instance_ofat_covers_attributes_without_rgb_coat_tint():
    text = SCRIPT.read_text(encoding="utf-8")

    for value in ("small", "medium", "large"):
        assert f'"{value}"' in text
    for value in ("slim", "standard", "stocky"):
        assert f'"{value}"' in text
    for value in ("young", "adult", "senior"):
        assert f'"{value}"' in text
    assert "--coat-glb" in text
    assert "--coat-projection-manifest" in text
    assert "exactly three breed-scoped coat GLBs are required" in text
    assert "real_reference_flux_multiview_edit_then_uv_projection" in text
    assert '"not_global_rgb_factor": True' in text
    assert '"global_rgb_material_factor_used": False' in text
    assert "baseColorFactor" not in text
    assert "COAT_FACTORS" not in text
    assert "patch_glb_base_color_factor" not in text


def test_generated_instance_ofat_uses_reviewed_visible_shape_ranges():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'BUILD_RATIOS = {"slim": 0.84, "standard": 1.0, "stocky": 1.16}' in text
    assert 'HEAD_RATIOS = {"young": 1.12, "adult": 1.0, "senior": 0.97}' in text
    assert "torso_weighted_lateral_vertical_rms_ratio" in text
    assert "head_weighted_radius_rms_ratio" in text


def test_generated_instance_ofat_senior_cue_is_local_not_global_tint():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "apply_senior_muzzle_surface_cue" in text
    assert "textured.rasterize_muzzle_mask" in text
    assert '"semantic_uv_muzzle_neutral_gray_floor_v1"' in text
    assert '"already_light_fur_luminance_preserved": True' in text
    assert '"global_rgb_material_factor_used": False' in text


def test_generated_instance_ofat_grounds_every_variant_and_derives_emitter():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "def rest_world_coordinates" in text
    assert "armature.data.pose_position = \"REST\"" in text
    assert "def ground_instance_root" in text
    assert '"rest_mesh_minimum_z_to_asset_root_zero_v1"' in text
    assert "abs(minimum_after) > GROUND_TOLERANCE_M" in text
    assert "def derive_muzzle_emitter" in text
    assert '"semantic_head_forward_quantile_rest_mesh_v1"' in text
    assert '"asset_specific_not_species_template": True' in text
    assert '"mouth_animation_required": False' in text
    assert text.index("ground_instance_root(mesh, armature, root)") < text.index(
        "stable.export_instance(output)"
    )
    assert "shutil.copyfile" not in text


def test_generated_instance_ofat_requires_real_reference_flux_evidence():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "validate_baseline_generation" in text
    assert "validate_projection_evidence" in text
    assert 'flux.get("is_distilled") is not False' in text
    assert 'flux.get("reference_image_count") != 2' in text
    assert 'not flux.get("appearance_reference_board")' in text
    assert 'projection.get("not_global_rgb_factor") is not True' in text
