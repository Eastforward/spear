from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BOARD_BUILDER = ROOT / "tools/build_animal_coat_reference_board.py"
RENDERER = ROOT / "tools/blender_render_generated_animal_coat_views.py"
FLUX_EDITOR = ROOT / "tools/flux2_edit_animal_multiview_coat.py"
PROJECTOR = ROOT / "tools/blender_project_animal_multiview_coat.py"


def test_reference_board_is_deterministic_presentation_not_breed_inference():
    text = BOARD_BUILDER.read_text(encoding="utf-8")
    assert "reference board requires 4 to 9 curated photographs" in text
    assert "ImageOps.fit" in text
    assert '"source_files_copied": False' in text
    assert '"breed_identity_and_rights_must_be_reviewed_separately": True' in text


def test_renderer_is_rest_pose_four_view_and_non_mutating():
    text = RENDERER.read_text(encoding="utf-8")
    assert 'VIEW_ORDER = ("front", "back", "left", "right")' in text
    assert 'armature.data.pose_position = "REST"' in text
    assert '"source_asset_modified": False' in text
    assert 'roughness.default_value = 0.82' in text
    assert 'camera_data.type = "ORTHO"' in text


def test_flux_editor_uses_base_true_cfg_and_forbids_drawn_stripes():
    text = FLUX_EDITOR.read_text(encoding="utf-8")
    assert "Flux2KleinPipeline" in text
    assert 'index.get("is_distilled", False) is not False' in text
    assert "negative_prompt_embeds=negative_embeds" in text
    assert '"one_model_invocation": True' in text
    assert "microscopic alternating colour bands" in text
    assert "never striped" in text
    assert "--appearance-reference-board" in text
    assert "Image 1 is the edit target" in text
    assert "Image 2 is a " in text
    assert '"real-photo appearance board of genuine examples' in text
    assert "conditioning_images = [montage]" in text
    assert "AVENGINE_FLUX2_KLEIN_BASE_SNAPSHOT" in text
    assert "one 3D animal" in text
    assert '"appearance_reference_board"' in text
    assert '"geometry_rig_or_animation_edit_authorized": False' in text


def test_projector_uses_spatial_view_ratios_and_preserves_runtime_contract():
    text = PROJECTOR.read_text(encoding="utf-8")
    assert "visibility_weighted_multiview_log_chroma_linear_luminance_preserved_v5" in text
    assert '"not_global_rgb_factor": True' in text
    assert "absolute_edited_chroma" in text
    assert "--absolute-chroma-strength" in text
    assert "args.absolute_chroma_strength * colour_field[loop_vertex]" in text
    assert '"colour_transfer_mode": args.colour_transfer_mode' in text
    assert "def srgb_to_linear(values):" in text
    assert '"explicit_srgb_to_linear_input_decode": True' in text
    assert '"bake_output_colourspace": "sRGB"' in text
    assert "BVHTree.FromPolygons" in text
    assert "original_base_image" in text
    assert "edited_chroma" in text
    assert "source_chroma" in text
    assert "luminance_transfer_strength" in text
    assert "candidate_rgb * luminance_normalization[:, None]" in text
    assert '"per_uv_corner_rec709_linear_luminance_preserved": True' in text
    assert "base_rgb * np.exp(colour_field[loop_vertex])" in text
    assert "base_geometric_luminance + blended_chroma" in text
    assert 'animations != ["Idle", "Walking"]' in text
    assert '"formal_dataset_registration_authorized": False' in text
    assert "--output-stem" in text
    assert 'f"animated_walk_idle_{args.output_stem}.glb"' in text
