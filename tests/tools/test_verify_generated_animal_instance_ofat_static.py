from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/verify_generated_animal_instance_ofat.py"
)


def test_instance_ofat_verifier_covers_the_complete_attribute_matrix():
    text = SCRIPT.read_text(encoding="utf-8")

    for variant in (
        "baseline",
        "size_small",
        "size_large",
        "build_slim",
        "build_stocky",
        "coat_blue_merle",
        "coat_red_white",
        "age_young",
        "age_senior",
    ):
        assert f'"{variant}"' in text
    assert 'batch.get("variant_count") != 9' in text
    assert 'set(records) != set(EXPECTED_VARIANTS)' in text


def test_instance_ofat_verifier_authenticates_runtime_and_coat_evidence():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "sha256_file(glb)" in text
    assert 'animations != ["Idle", "Walking"]' in text
    assert '{"JOINTS_0", "WEIGHTS_0"}' in text
    assert '"baseColorTexture"' in text
    assert '"semantic_head_forward_quantile_rest_mesh_v1"' in text
    assert '"size-specific emitter did not scale in final asset-root space"' in text
    assert '"size_specific_emitter_offset_m"' in text
    assert '"real_reference_flux_multiview_edit_then_uv_projection"' in text
    assert '"native_negative_prompt_embeddings_with_cfg"' in text
    assert 'projection.get("not_global_rgb_factor") is not True' in text
    assert '"nodes", "meshes", "skins", "accessors", "animations"' in text
    assert '"nonbaseline_coats_use_spatial_uv_projection_not_global_rgb": True' in text
