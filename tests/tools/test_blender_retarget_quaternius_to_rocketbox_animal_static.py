from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/blender_retarget_quaternius_to_rocketbox_animal.py"


def source():
    return SCRIPT.read_text(encoding="utf-8")


def test_preserves_native_rocketbox_mesh_skin_and_uv_contract():
    text = source()
    assert "mesh_skin_sha256" in text
    assert "mesh_contract_before != mesh_contract_after" in text
    assert '"native_skin_unchanged": True' in text


def test_maps_quaternius_dog_semantics_to_rocketbox_beagle():
    text = source()
    assert '"beagle Pelvis": "Bone"' in text
    assert '"beagle Head": "Bone.003"' in text
    assert '"beagle L Foot": "Bone.010"' in text
    assert '"beagle R Hand": "Bone.016"' in text
    assert '"beagle Tail3": "Bone.007"' in text


def test_uses_cardinal_yaw_and_never_fine_yaw_inference():
    text = source()
    assert "CANONICAL_YAW_DEGREES = 90.0" in text
    assert '"fine_yaw_inference": False' in text


def test_exports_only_walk_idle_with_native_pbr():
    text = source()
    assert 'ACTION_HINTS = {"Walking": "walk", "Idle": "idle"}' in text
    assert 'canonical_walk_idle=True' in text
    assert "Rocketbox Color" in text
    assert "Rocketbox Bump" in text
    assert "Rocketbox Specular" in text


def test_refuses_overwrite_and_records_mit_license():
    text = source()
    assert "refusing to replace" in text
    assert 'manifest_path.open("x"' in text
    assert '"spdx_id": "MIT"' in text


def test_validates_bone_names_instead_of_bpy_bone_objects():
    text = source()
    assert "set(source.data.bones.keys())" in text
    assert "set(target.data.bones.keys())" in text
    assert "bone.name: parent_local_rest(bone)" in text


def test_resamples_fractional_source_keyframes_without_rounding_them():
    text = source()
    assert "source_action_sample_frames" in text
    assert "set_scene_time(source_frame)" in text
    assert '"source_frame": source_frame' in text


def test_quaternion_sign_and_source_helpers_cannot_pollute_evidence():
    text = source()
    assert "abs(float(actual.dot(rotation)))" in text
    assert "remove_export_extras(target, mesh)" in text
    assert '"removed_export_extras": removed_export_extras' in text


def test_can_damp_in_out_limb_swing_without_changing_native_skin():
    text = source()
    assert "--limb-lateral-damping" in text
    assert "def damp_limb_lateral_direction" in text
    assert "TARGET_AUTHORED_LATERAL" in text
    assert "LOCOMOTION_LIMB_BONES" in text
    assert '"method": "rest_lateral_component_damping_v1"' in text
    assert '"native_skin_unchanged": True' in text
