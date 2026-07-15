from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "blender_robust_swap_mesh_keep_rig.py"


def test_blender_robust_swap_exposes_nearest_weight_mode():
    text = SCRIPT.read_text()

    assert 'choices=["region", "auto", "nearest"]' in text
    assert "transfer_weights_by_nearest_surface" in text
    assert 'args.weight_mode == "nearest"' in text


def test_blender_robust_swap_defaults_to_accelerated_bvh_nearest_surface():
    text = SCRIPT.read_text()

    assert '"--nearest-backend"' in text
    assert 'default="bvh"' in text
    assert 'choices=["bvh", "bruteforce"]' in text
    assert "def transfer_weights_by_blender_bvh(" in text
    assert "BVHTree.FromPolygons" in text
    assert 'args.nearest_backend == "bvh"' in text
    assert "barycentric[0] * source_weights[face[0]]" in text


def test_blender_robust_swap_exposes_target_yaw_rotation():
    text = SCRIPT.read_text()

    assert "--target-rotate-z-deg" in text
    assert "rotate_target_z_degrees" in text
    assert "args.target_rotate_z_deg" in text
    assert "Matrix.Rotation" in text
    assert ".data.transform(" in text


def test_blender_robust_swap_supports_explicit_animal_forward_axes():
    text = SCRIPT.read_text()

    assert "--semantic-forward-axis" in text
    assert 'choices=["positive-x", "negative-y"]' in text
    assert 'forward_axis="positive-x"' in text
    assert "-vertices[:, 1]" in text
    assert "args.semantic_forward_axis" in text


def test_blender_robust_swap_can_export_only_canonical_walk_and_idle():
    text = SCRIPT.read_text()

    assert "--export-action-policy" in text
    assert 'choices=["all", "walk-idle"]' in text
    assert "keep_canonical_walk_idle_actions" in text
    assert 'idle.name = "Idle"' in text
    assert 'walking.name = "Walking"' in text
    assert 'gltf["animations"] = [idle, walking]' in text


def test_blender_robust_swap_keeps_approved_animal_dampening_defaults():
    text = SCRIPT.read_text()

    assert 'p.add_argument("--dampen-foot-rotations", type=float, default=1.0' in text
    assert 'p.add_argument("--dampen-head-rotations", type=float, default=0.0' in text
    assert 'p.add_argument("--dampen-tail-rotations", type=float, default=0.0' in text


def test_blender_robust_swap_welds_target_geometry_before_weight_transfer():
    text = SCRIPT.read_text()

    assert "import bmesh" in text
    assert "def weld_target_position_duplicates" in text
    weld_call = text.index("    weld_target_position_duplicates(tgt_mesh)")
    transfer_branch = text.index('    if args.weight_mode == "auto":')
    assert weld_call < transfer_branch
    assert "bmesh.ops.remove_doubles" in text
    assert "coordinates remain stored per face corner" in text


def test_blender_robust_swap_still_recomputes_normals_before_weld_helper():
    text = SCRIPT.read_text()
    start = text.index("def recompute_normals(obj):")
    end = text.index("def weld_target_position_duplicates(obj):")
    function_text = text[start:end]

    assert 'bpy.ops.object.mode_set(mode="EDIT")' in function_text
    assert 'bpy.ops.mesh.select_all(action="SELECT")' in function_text
    assert "bpy.ops.mesh.normals_make_consistent(inside=False)" in function_text
    assert function_text.rstrip().endswith('bpy.ops.object.mode_set(mode="OBJECT")')


def test_blender_robust_swap_accepts_stable_rocketbox_fbx_meshes():
    text = SCRIPT.read_text()
    function_text = text[
        text.index("def import_mesh_only(path):") : text.index("def import_rig_scene(path):")
    ]

    assert 'elif ext == ".fbx":' in function_text
    assert "bpy.ops.import_scene.fbx(filepath=path, use_anim=False)" in function_text


def test_blender_robust_swap_detaches_fbx_parent_and_rejects_zero_weights():
    text = SCRIPT.read_text()

    assert "target_world = tgt_mesh.matrix_world.copy()" in text
    assert "tgt_mesh.parent = None" in text
    assert "tgt_mesh.matrix_world = target_world" in text
    assert "skin transfer left unweighted target vertices" in text


def test_blender_robust_swap_can_apply_nonmetallic_animal_runtime_material():
    text = SCRIPT.read_text()

    assert "--animal-nonmetallic-roughness" in text
    assert "apply_animal_nonmetallic_material_policy" in text
    assert 'metallic.default_value = 0.0' in text
    assert "roughness_input.default_value = roughness" in text
    assert '"preserve_imported_pbr_base_color"' in text
