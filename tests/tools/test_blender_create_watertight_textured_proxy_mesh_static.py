"""Static contracts for watertight image-to-3D runtime proxies."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_create_watertight_textured_proxy_mesh.py"
)


def test_proxy_regularizes_topology_before_rigging():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "avengine_watertight_textured_runtime_proxy_v1" in text
    assert "bpy.ops.object.voxel_remesh()" in text
    assert 'type="SHRINKWRAP"' in text
    assert 'type="DECIMATE"' in text
    assert "approved_skeleton_or_animation_touched" in text


def test_proxy_preserves_pbr_through_surface_attribute_transfer():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'type="DATA_TRANSFER"' in text
    assert 'available = {"UV"}' in text
    assert 'available.add("COLOR_CORNER")' in text
    assert 'modifier.loop_mapping = "POLYINTERP_NEAREST"' in text
    assert "source.data.materials" in text


def test_proxy_refuses_open_or_nonmanifold_results_and_overwrites():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "refusing to replace" in text
    assert 'with manifest.open("x"' in text
    assert 'final_topology["boundary_edges"]' in text
    assert 'final_topology["nonmanifold_edges_over_two_faces"]' in text


def test_proxy_reports_every_long_phase_without_waiting_for_completion():
    text = SCRIPT.read_text(encoding="utf-8")

    for phase in (
        "import_start",
        "voxel_remesh_start",
        "shrinkwrap_start",
        "post_shrinkwrap_smooth_start",
        "decimate_start",
        "surface_transfer_start",
        "export_start",
    ):
        assert f'"{phase}"' in text
    assert "flush=True" in text


def test_proxy_can_remove_closed_crack_like_folds_after_shrinkwrap():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"--post-shrinkwrap-smooth-iterations"' in text
    assert "post_shrinkwrap_smooth(proxy" in text
    assert 'type="LAPLACIANSMOOTH"' in text
    assert "modifier.use_volume_preserve = True" in text
    assert '"post_shrinkwrap_smooth_iterations"' in text


def test_proxy_can_limit_return_to_a_defective_raw_surface():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"--shrinkwrap-strength"' in text
    assert "clean_positions = [vertex.co.copy()" in text
    assert "clean.lerp(vertex.co, strength)" in text
    assert '"shrinkwrap_strength": args.shrinkwrap_strength' in text


def test_proxy_can_repair_only_the_normalized_mid_torso():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"--torso-fold-repair-iterations"' in text
    assert "repair_normalized_torso_folds" in text
    assert 'name="NormalizedTorsoFoldRepair"' in text
    assert "modifier.vertex_group = group_name" in text
    assert "remaining_group = proxy.vertex_groups.get(group_name)" in text
    assert '"weighted_mid_torso_only_preserve_volume"' in text
    assert '"torso_fold_repair": torso_fold_repair' in text


def test_proxy_can_use_a_lower_face_pbr_copy_only_for_attribute_transfer():
    text = SCRIPT.read_text(encoding="utf-8")

    assert '"--attribute-source"' in text
    assert "attribute_source_path != source_path" in text
    assert "attribute_source," in text
    assert "full_resolution_source_remains_geometry_authority" in text
    assert "shrinkwrap_to_source(proxy, source, args.shrinkwrap_strength)" in text


def test_proxy_defaults_to_pbr_bake_and_retains_diagnostic_transfer_backends():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'choices=("bake", "bvh", "data-transfer")' in text
    assert 'default="bake"' in text
    assert "BVHTree.FromPolygons" in text
    assert "bvh.ray_cast(" in text
    assert "bvh.find_nearest(point)" in text
    assert "nearest_fallback_count" in text
    assert '"query_domain": "face_corner"' in text
    assert "samples[loop.index]" in text
    assert "transfer_surface_attributes_modifier" in text


def test_proxy_bakes_a_new_pbr_atlas_instead_of_reusing_ambiguous_uv_seams():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "bpy.ops.uv.smart_project(" in text
    assert 'type="DIFFUSE"' in text
    assert 'pass_filter={"COLOR"}' in text
    assert 'type="ROUGHNESS"' in text
    assert "base_image.pack()" in text
    assert "roughness_image.pack()" in text
    assert "metallic_policy" in text
    assert "encoded <= 0.04045" in text
    assert "encoded / 12.92" in text
    assert "** 2.4" in text
    assert "base_color_encoding_policy" in text
    assert 'default="preserve-bake"' in text
    assert '"--base-color-gain"' in text
    assert "apply_base_color_gain" in text
    assert "attribute_source_pbr_baked_to_new_uv_atlas" in text
    assert "Cycles circular dependency" in text


def test_proxy_uses_blender_42_vertex_color_export_options():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "export_all_vertex_colors=True" in text
    assert 'export_vertex_color="ACTIVE"' in text
    assert "export_colors=" not in text
