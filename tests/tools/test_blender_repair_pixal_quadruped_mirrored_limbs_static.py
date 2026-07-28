"""Static fail-closed contracts for the Blender same-Pixal repair entry."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_repair_pixal_quadruped_mirrored_limbs.py"
)


def _text() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_wrapper_reauthenticates_owner_pixal_and_formal_rejection():
    text = _text()

    for option in (
        "--source",
        "--pixal-manifest",
        "--approved-reference",
        "--owner-review",
        "--static-decision",
        "--expected-source-sha256",
        "--expected-reference-sha256",
    ):
        assert option in text
    assert "authenticate_lineage(" in text
    assert '"approved_for_pixal3d"' in text
    assert '"avengine_controlled_animal_static_decision_v1"' in text
    assert "formal raw-Pixal rejection contract changed" in text


def test_wrapper_has_no_donor_or_rig_input_and_mirrors_only_masked_pixal_copy():
    text = _text()

    assert 'bpy.ops.import_scene.gltf(filepath=str(source_path))' in text
    # The second import is the just-exported output readback, not donor input.
    assert text.count("bpy.ops.import_scene.gltf(") == 2
    assert "bpy.ops.import_scene.gltf(filepath=str(path))" in text
    assert 'mirrored_geometry_source": "same_authenticated_pixal_mesh_only"' in text
    assert '"external_geometry_inputs": []' in text
    assert '"external_skeleton_inputs": []' in text
    assert '"external_weight_inputs": []' in text
    assert "--donor" not in text
    assert "--rig" not in text
    assert "np.all(masks[\"replace_corridor\"][triangles], axis=1)" in text
    assert "np.all(masks[\"donor\"][triangles], axis=1)" in text
    assert "retain_faces(" in text
    assert "retain_vertices(" not in text


def test_wrapper_protects_tail_and_requires_dynamic_geometry_gates():
    text = _text()

    assert "identify_tail_surface(" in text
    assert '"height_independent_source_tail_identified"' in text
    assert '"authenticated_source_tail_surface_preserved"' in text
    assert "tail_protected_donor_overlap_vertex_count" in text
    assert "tail_protected_replacement_overlap_vertex_count" in text
    assert '"tail_geometry_selected_for_mirroring": False' in text
    assert "mirror_points_across_y_with_attachment_taper" in text
    assert '"mirrored_attachment": mirror_attachment' in text
    assert "audit_low_slice_limb_chains(" in text
    assert '"no_low_cross_limb_membrane"' in text
    assert '"one_connected_output_component"' in text
    assert '"watertight_manifold_topology"' in text


def test_wrapper_forbids_global_remesh_and_fails_closed_before_visual_review():
    text = _text()

    assert "bpy.ops.object.voxel_remesh()" not in text
    assert 'type="LAPLACIANSMOOTH"' not in text
    assert 'type="DECIMATE"' not in text
    assert "local_weld(" in text
    assert "audit_immutable_source_topology(" in text
    assert "rejected_bounded_local_repair_preflight" in text
    assert "rejected_unsafe_or_incomplete_local_weld" in text
    assert "raise SystemExit(2)" in text
    assert "write_manifest(manifest, payload)" in text
    assert '"cat_semantic_retarget_authorized": False' in text
    assert '"passed_automatic_geometry_gate_pending_multiview_review"' in text
    assert 'payload["formal_dataset_registration_authorized"] = False' in text


def test_wrapper_reimports_export_and_authenticates_surface_uv_material_and_pbr():
    text = _text()

    assert "canonical_triangle_surface_signatures(" in text
    assert "audit_expected_surface_signatures(" in text
    assert "restore_source_pbr_contract(" in text
    assert "embedded_images(" in text
    assert '"outside_corridor_position_index_uv_material_preserved"' in text
    assert '"embedded_textures_and_pbr_unchanged"' in text
    assert "source_glb_pbr_bindings_and_embedded_bytes_restored_exactly" in text
