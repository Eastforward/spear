"""Static contract for generated quadruped support-plane leveling."""

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_level_generated_animal_support_plane.py"
)
NUMPY_AUTHORITY = (
    Path(__file__).resolve().parents[2]
    / "tools/generated_animal_support_plane.py"
)
STDLIB_CONTRACT = (
    Path(__file__).resolve().parents[2]
    / "tools/generated_animal_support_plane_contract.py"
)
OUTPUT_READBACK = (
    Path(__file__).resolve().parents[2]
    / "tools/blender_readback_generated_animal_support_plane.py"
)


def test_support_plane_leveling_is_semantic_rigid_and_pre_animation():
    text = SCRIPT.read_text(encoding="utf-8")

    assert "avengine_generated_animal_support_plane_leveling_v2" in text
    assert "SPEAR_ROOT = Path(__file__).resolve().parents[1]" in text
    assert "sys.path.insert(0, str(SPEAR_ROOT))" in text
    assert "infer_quadruped_semantics" in text
    assert "semantics.foot_leaves" in text
    assert "evaluate_dual_authority_support_plane" in text
    assert "distal_two_weight_scores" in text
    assert 'choices=("mesh-foot-bottoms",)' in text
    assert "bone-endpoints" not in text
    assert "normal.rotation_difference(up)" in text
    assert "support-plane leveling must run before animation" in text
    assert "root.matrix_world = transform @ root.matrix_world" in text
    assert '"dual_authority": dual_authority' in text
    assert '"mesh_topology_changed": False' in text
    assert '"skeleton_hierarchy_changed": False' in text
    assert '"skin_weights_changed": False' in text
    assert "refusing to replace" in text


def test_mesh_foot_authority_rejects_sparse_contact_instead_of_falling_back():
    numpy_text = NUMPY_AUTHORITY.read_text(encoding="utf-8")
    contract_text = STDLIB_CONTRACT.read_text(encoding="utf-8")

    assert "nearest_segment = np.argmin" in numpy_text
    assert "score_owner = np.argmax" in numpy_text
    assert "bottom band captured fewer than" in numpy_text
    assert "foot floors disagree beyond contact band" in numpy_text
    assert "contact centroids disagree beyond the" in numpy_text
    assert '"fallback_used": False' in numpy_text
    assert "support-plane fallback is forbidden" in contract_text
    assert "maximum_contact_centroid_xy_delta_between_authorities" in (
        contract_text
    )
    assert "import numpy" not in contract_text


def test_output_glb_is_reimported_and_binds_semantics_transform_and_weights():
    text = OUTPUT_READBACK.read_text(encoding="utf-8")

    assert "bpy.ops.import_scene.gltf" in text
    assert text.count("snapshot(") >= 3
    assert "directional_vertex_coverage" in text
    assert "maximum_world_vertex_delta_from_declared_transform" in text
    assert "maximum_bone_endpoint_delta_from_declared_transform" in text
    assert "maximum_skin_weight_delta" in text
    assert "distal_owner_bones" in text
    assert "semantic foot/limb ownership changed after export" in text
    assert "support manifest foot leaves do not match independent inference" in text
