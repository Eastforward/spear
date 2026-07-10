from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "blender_render_rocketbox_source_review.py"


def renderer_source():
    return SCRIPT.read_text(encoding="utf-8")


def test_renderer_never_clears_materials_or_assigns_actions():
    source = renderer_source()

    assert ".materials.clear(" not in source
    assert "animation_data.action" not in source
    assert "animation_data_clear()" in source
    assert 'pose_position = "REST"' in source


def test_renderer_declares_required_outputs_and_eevee():
    source = renderer_source()

    outputs = (
        "front.png",
        "back.png",
        "left.png",
        "right.png",
        "top.png",
        "face_close.png",
        "arms_close.png",
        "feet_close.png",
        "turntable.mp4",
        "render_manifest.json",
    )
    for name in outputs:
        assert name in source
    assert '"BLENDER_EEVEE_NEXT"' in source


def test_renderer_exposes_the_pinned_cli_and_axes():
    source = renderer_source()

    for option in (
        "--asset-id",
        "--fbx",
        "--texture-dir",
        "--output-dir",
        "--forward-axis",
        "--up-axis",
    ):
        assert option in source
    assert 'default="-Y"' in source
    assert 'default="+Z"' in source
    assert '"front": (0.0, -1.0, 0.15)' in source
    assert '"back": (0.0, 1.0, 0.15)' in source
    assert '"left": (-1.0, 0.0, 0.15)' in source
    assert '"right": (1.0, 0.0, 0.15)' in source
    assert '"top": (0.0, 0.0, 1.0)' in source


def test_renderer_validates_the_imported_rocketbox_rig_and_slots():
    source = renderer_source()

    assert "read_factory_settings(use_empty=True)" in source
    assert "import_scene.fbx" in source
    assert "len(obj.data.vertices)" in source
    assert "len(armature.data.bones) != 80" in source
    assert "expected 80 Rocketbox avatar bones" in source
    assert "[slot.material.name for slot in mesh.material_slots]" in source
    assert "unexpected Rocketbox material slots" in source
    for suffix in ("_body", "_head", "_opacity"):
        assert suffix in source


def test_renderer_reconstructs_textured_eevee_materials_in_place():
    source = renderer_source()

    assert "material.node_tree" in source
    assert 'colorspace_settings.name = "sRGB"' in source
    assert 'colorspace_settings.name = "Non-Color"' in source
    assert 'ShaderNodeNormalMap' in source
    assert '"Specular IOR Level"' in source
    assert 'surface_render_method = "DITHERED"' in source
    assert "FileNotFoundError" in source


def test_renderer_bulk_reads_opacity_pixels_and_reuses_legend_material():
    source = renderer_source()

    assert "image.pixels.foreach_get" in source
    assert "image.pixels[" not in source
    assert "bpy.data.materials.get(name)" in source


def test_renderer_keeps_labels_and_direction_guides_review_safe():
    source = renderer_source()

    assert "REVIEW_LIGHT_ENERGIES = (320.0, 140.0, 220.0)" in source
    assert "BODY_FRAME_MARGIN = 1.24" in source
    assert "LEGEND_FONT_SCALE = 0.018" in source
    assert "LEGEND_MARGIN_SCALE = 0.070" in source
    assert 'floor.hide_render = view_name == "top"' in source
    assert 'arrow.hide_render = view_name != "top"' in source
    assert "OPACITY_ROUGHNESS = 0.86" in source


def test_renderer_fits_ortho_camera_from_blender_frame_dimensions():
    source = renderer_source()

    assert "camera.data.view_frame(scene=scene)" in source
    assert "content_width / frame_width" in source
    assert "content_height / frame_height" in source


def test_turntable_legend_uses_blender_frame_bounds():
    source = renderer_source()

    assert "def camera_frame_bounds" in source
    assert "frame_left, frame_right, frame_bottom, frame_top = camera_frame_bounds(" in source
    assert "text.location = (frame_left + margin, frame_top - margin, -1.0)" in source


def test_renderer_preserves_rocketbox_opacity_semantics_and_pixel_labels():
    source = renderer_source()

    assert "def material_uses_color_as_alpha" in source
    assert 'link.from_socket.name == "Color"' in source
    assert 'link.to_socket.name == "Alpha"' in source
    assert "material.use_transparency_overlap = False" in source
    assert 'shader.inputs["Specular IOR Level"].default_value = 0.0' in source
    assert "def annotate_still" in source
    assert "from PIL import Image, ImageDraw, ImageFont" in source
    assert "annotate_still(" in source


def test_renderer_pins_review_dimensions_and_turntable_timing():
    source = renderer_source()

    assert "1200, 1600" in source
    assert "1200, 900" in source
    assert "1280, 720" in source
    assert "fps = 24" in source
    assert "frame_end = 96" in source
    for legend_text in (
        "UP +Z",
        "FRONT -Y",
        "REST POSE / NO ACTION",
    ):
        assert legend_text in source


def test_renderer_manifest_records_the_review_contract():
    source = renderer_source()

    fields = (
        "blender_version",
        "source_fbx_sha256",
        "mesh_name",
        "vertex_count",
        "polygon_count",
        "uv_layer_count",
        "armature_name",
        "bone_count",
        "material_slot_names",
        "view_files",
        "video_file",
        "forward_axis",
        "up_axis",
        "animation_attached",
    )
    for field in fields:
        assert f'"{field}"' in source
    assert '"animation_attached": False' in source
