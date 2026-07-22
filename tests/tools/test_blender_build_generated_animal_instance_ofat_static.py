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


def test_generated_instance_ofat_requires_real_reference_flux_evidence():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "validate_baseline_generation" in text
    assert "validate_projection_evidence" in text
    assert 'flux.get("is_distilled") is not False' in text
    assert 'flux.get("reference_image_count") != 2' in text
    assert 'not flux.get("appearance_reference_board")' in text
    assert 'projection.get("not_global_rgb_factor") is not True' in text
