"""Static safety contracts for the receipt-hard Blender identity tool."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/blender_realize_bounded_quadruped_identity.py"
CONTRACT = ROOT / "tools/bounded_quadruped_identity_contract.py"


def source(path):
    return path.read_text(encoding="utf-8")


def test_blender_tool_authenticates_all_inputs_and_publishes_create_only_outputs():
    text = source(SCRIPT)
    contract_text = source(CONTRACT)

    assert "os.O_NOFOLLOW" in text
    assert "identity changed while it was read" in text
    assert "refusing to replace output root" in contract_text
    assert "open_secure_publication" in text
    assert "publication.seal(" in text
    assert "publication.publish(" in text
    assert "/proc/self/fd/" in contract_text
    assert "renameat2" in contract_text
    assert "os.walk(" not in text + contract_text
    assert "shutil.rmtree" not in text + contract_text
    assert "os.rename(" not in text + contract_text
    assert "os.unlink(" not in contract_text
    assert "os.rmdir(" not in contract_text
    assert "tempfile.mkdtemp" not in text + contract_text
    assert "strict_json_loads" in text
    assert "require_snapshot_publication_records" in text
    assert "raw_animation_time_signature(\n        source_snapshot_path" in text
    assert "raw_skin_signature(\n        source_snapshot_path" in text
    assert "**file_record(source_path)" not in text


def test_audit_never_mutates_and_realization_requires_machine_authorization():
    text = source(SCRIPT)

    audit_branch = text.index('if args.mode == "audit":')
    authorization_binding = text.index("validate_execution_authorization")
    morph_call = text.index("manifest = realize(")
    assert audit_branch < authorization_binding < morph_call
    assert "pending_machine_execution_authorization" in text
    assert 'preflight.get("toolchain") != toolchain_record()' in text
    assert '"execution_authorized": False' in text
    assert '"formal_dataset_registration_authorized": False' in text
    assert '"native_merge_authorized": False' in text
    assert '"user_approval_inferred": False' in text


def test_morph_is_bounded_to_masks_and_records_all_required_invariants():
    text = source(SCRIPT)

    assert "all_position_groups" in text
    assert "ear masks omit duplicate vertices" in text
    assert "pair_mirrored_groups" in text
    assert "maximum_moved_vertex_fraction" in text
    assert "maximum_displacement_diagonal_ratio" in text
    assert "unselected_position_sha256_before" in text
    assert "unselected_position_sha256_after" in text
    assert "mesh_topology_sha256" in text
    assert "vertex_weight_sha256" in text
    assert "skeleton_sha256" in text
    assert "action_sha256" in text
    assert "raw_skin_signature" in text
    assert "joint_hierarchy_and_order" in text
    assert "inverse_bind_matrices" in text
    assert "inverse_mesh_global_times_joint_global_times_inverse_bind" in text
    assert "maximum_bind_matrix_semantic_delta" in text
    assert "bind_pose_world_surface_sha256" in text
    assert "sampled_animation_semantics" in text
    assert "evaluated_world_surface_sha256" in text
    assert "compare_sampled_animation_semantics" in text
    assert "maximum_sampled_skinned_world_delta" in text
    assert "maximum_animation_duration_delta_frames" in text
    assert "maximum_root_trajectory_delta" in text
    assert "canonical_skinned_surface_sha256" in text
    assert "mesh_object.data.calc_loop_triangles()" in text


def test_inventory_removal_uv_coat_export_and_readback_are_fail_closed():
    text = source(SCRIPT)

    assert "removal target is not the exact unskinned material-free object" in text
    assert "removal target is not the only expected pose-bone custom shape" in text
    assert "bone.custom_shape = None" in text
    assert "bpy.data.objects.remove(removal, do_unlink=True)" in text
    assert "bpy.data.meshes.remove(removal_mesh)" in text
    assert "bpy.ops.uv.smart_project" in text
    assert "normalized_face_region_texture_v1" in source(CONTRACT)
    assert "image.pack()" in text
    assert '"TEXCOORD_0"' in text
    assert '"JOINTS_0"' in text
    assert '"WEIGHTS_0"' in text
    assert "output coat texture is not embedded PNG evidence" in text
    assert "output reimport changed canonical geometry or skin weights" in text
    assert "import_snapshot(output_glb, disable_bone_shape=True)" in text
    assert '"synthetic_bone_shape_object_absent": True' in text
    assert '"glb": relative_file_record(output_glb)' in text
    assert "base = np.asarray(plan.base_color" in text
    assert "white = np.asarray(plan.white_color" in text
    assert "srgb_channel_to_linear" not in text
    assert "contract.png_rgba8_code_values" in text
    assert "external_png_srgb_readback" in text
    assert "embedded_png_srgb_readback" in text
    assert '"base_color_texture_color_space": "sRGB"' in text


def test_export_preserves_fractional_key_times_and_dense_phase_gate():
    text = source(SCRIPT)

    assert "export_force_sampling=False" in text
    assert "export_force_sampling=True" not in text
    assert 'export_animation_mode="ACTIONS"' in text
    assert 'export_animation_mode="NLA_TRACKS"' not in text
    assert "subframe=sample_frame - integer_frame" in text
    assert "raw_animation_time_signature" in text
    assert "output GLB changed raw animation timelines or interpolation" in text
    assert '"raw_animation_timelines_and_interpolation_unchanged": True' in text
    assert "DENSE_ANIMATION_PHASE_INTERVALS = 80" in source(CONTRACT)
    assert "complete uniform 81-phase" in source(CONTRACT)
    assert 'publication.write_evidence(\n            "preflight_receipt.json"' in text
    assert (
        'publication.write_evidence(\n            "machine_execution_authorization.json"'
        in text
    )
    assert '"publication_snapshot": evidence_file_record(' in text
    assert '"blender_realize_bounded_quadruped_identity.py"' in text
    assert '"bounded_quadruped_identity_contract.py"' in text
    assert '"toolchain_publication_snapshots": toolchain_snapshots' in text
    assert '"publication_snapshot": evidence_file_record(source_snapshot_path)' in text
    assert '"publication_snapshot": evidence_file_record(plan_snapshot_path)' in text


def test_publication_boundary_reverifies_final_and_states_same_uid_limit():
    text = source(SCRIPT)
    contract_text = source(CONTRACT)

    assert "parent_entry_name=self.output_root.name" in contract_text
    assert "post_rename_exact_held_fd_reverification_required" in contract_text
    assert "rename_is_publication_boundary_not_readiness_claim" in contract_text
    assert "malicious_same_uid_writer_can_mutate_after_verification" in contract_text
    assert "external_expected_manifest_raw_sha256_required" in contract_text
    assert "rehash_entire_declared_file_closure_before_use" in contract_text
    assert (
        '"publication_security_boundary": contract.publication_security_boundary()'
        in text
    )
    for success_marker in (
        "BOUNDED_QUADRUPED_IDENTITY_PREFLIGHT_OK",
        "BOUNDED_QUADRUPED_IDENTITY_REALIZATION_OK",
    ):
        success_index = text.index(success_marker)
        publish_index = text.rfind("publication.publish(", 0, success_index)
        close_index = text.rfind("publication.close()", 0, success_index)
        assert publish_index < close_index < success_index


def test_new_code_remains_python_39_compatible_and_taxonomy_independent():
    combined = source(SCRIPT) + source(CONTRACT)

    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "zip"
        and any(keyword.arg == "strict" for keyword in node.keywords)
        for tree in (ast.parse(source(SCRIPT)), ast.parse(source(CONTRACT)))
        for node in ast.walk(tree)
    )
    assert " | None" not in combined
    assert "corgi" not in combined.lower()
    assert "pembroke" not in combined.lower()
    assert "unreal" not in combined.lower()
    assert "native merge" in combined.lower()
