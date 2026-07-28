from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "blender_readback_generated_animal_tokenrig_binding.py"
)
SOURCE = SCRIPT.read_text(encoding="utf-8")


def test_readback_is_a_fresh_blender_import_and_no_replace_gate():
    assert SOURCE.count("bpy.ops.wm.read_factory_settings(use_empty=True)") >= 1
    assert "bpy.ops.import_scene.gltf" in SOURCE
    assert "refusing to replace TokenRig readback" in SOURCE


def test_readback_requires_one_skin_armature_and_no_animation():
    assert 'output_container["skin_count"] != 1' in SOURCE
    assert "len(armatures) != 1" in SOURCE
    assert "linked != {armatures[0]}" in SOURCE
    assert 'output_container["animation_count"]' in SOURCE
    assert "bpy.data.actions" in SOURCE


def test_readback_compares_logical_vertices_triangles_uvs_and_pbr():
    assert "map_to_source_position_classes" in SOURCE
    assert "output_covered_classes" in SOURCE
    assert "cyclic_triangle_signature" in SOURCE
    assert "input_triangles != output_triangles" in SOURCE
    assert "input_corners != output_corners" in SOURCE
    assert "pbr_payload_sha256" in SOURCE
    assert "embedded image" in SOURCE


def test_readback_has_fixed_tolerances_and_no_cli_bypass():
    assert "POSITION_TOLERANCE_RATIO = 3.0e-7" in SOURCE
    assert "UV_ABSOLUTE_TOLERANCE = 1.0e-7" in SOURCE
    assert "--position-tolerance" not in SOURCE
    assert "--allow-geometry-change" not in SOURCE
