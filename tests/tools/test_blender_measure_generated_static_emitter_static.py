from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_measure_generated_static_emitter.py"
)


def test_static_emitter_is_data_driven_surface_measurement_not_object_heuristics():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "mesh_surface_barycentric_samples_v1" in text
    assert "reviewed_bbox_fraction_nearest_surface_v1" in text
    assert "BVHTree.FromPolygons" in text
    assert "weighted_emitter" in text
    assert '"asset_specific_not_class_template": True' in text
    assert '"offset_space": "final_scaled_asset_root"' in text
    for forbidden in ("telephone", "doorbell", "kettle", "microwave", "alarm_clock"):
        assert forbidden not in text


def test_static_emitter_requires_finalization_and_review_authority():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "validate_static_finalization" in text
    assert "validate_static_anchor_spec" in text
    assert '"formal_dataset_registration_authorized": False' in text
    assert '"visual_review": "pending"' in text
    assert "refusing to replace" in text
    assert "os.O_EXCL" in text
    assert "os.O_NOFOLLOW" in text
    assert "os.link(temporary, marker_path)" in text


def test_static_emitter_rejects_rigs_and_animation():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'modifier.type == "ARMATURE"' in text
    assert 'item.type == "ARMATURE"' in text
    assert "bpy.data.actions" in text
    assert "static emitter input contains rig data" in text
