from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_finalize_generated_static_object.py"
)


def test_static_finalization_is_data_driven_and_never_rigs():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "avengine_controlled_static_object_decision_v1" in text
    assert "avengine_static_heading_review_v1" in text
    assert '"measurement") != "height_cm"' in text
    assert "reviewed_source_front_yaw_deg" in text
    assert 'target_front_axis"] != "positive-x"' in text
    assert "Matrix.Scale(uniform_scale, 4)" in text
    assert "Matrix.Translation" in text
    assert "static finalization input contains rig or animation data" in text
    for forbidden in ("telephone", "doorbell", "kettle", "microwave", "alarm_clock"):
        assert forbidden not in text


def test_static_finalization_reauthenticates_lineage_and_topology():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "validate_watertight_manifest" in text
    assert "validate_static_decision" in text
    assert "validate_heading_evidence" in text
    assert "static decision/Pixal/watertight lineage changed" in text
    assert '"boundary_edges"' in text
    assert '"wire_edges"' in text
    assert '"nonmanifold_edges_over_two_faces"' in text
    assert '"formal_dataset_registration_authorized": False' in text
    assert "refusing to replace" in text
    assert "os.O_EXCL" in text
    assert "os.O_NOFOLLOW" in text


def test_static_finalization_requires_export_readback_gates():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "bpy.ops.import_scene.gltf(filepath=str(output))" in text
    assert "static physical-height readback exceeded profile tolerance" in text
    assert "static grounding readback exceeded tolerance" in text
    assert "protected scene field" in text
    assert '"passed_final_scaled_grounded_canonical_glb"' in text
