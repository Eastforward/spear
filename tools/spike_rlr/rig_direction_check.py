"""Rig direction runtime assertion via bone query.

Two capabilities:
  1. calibrate_rig_forward_from_velocity(actor, instance) -> observed rig
     forward direction in world frame (used to build rig_calibration.json).
  2. assert_body_forward(actor, expected_yaw_deg, tolerance_deg) -> raises
     AssertionError if observed body forward diverges from expected motion
     direction. Called per-clip or per-frame in run_render_pass_apartment
     to catch coordinate-system bugs.

Query strategy:
  Read Root (or Pelvis, or Spine1) bone WORLD position at frame T and T+N.
  velocity = pos_T+N - pos_T. Direction of velocity = rig's actual forward.

Note: works even when Head/Tail bones are dampened (they follow the root
rigidly during walking). See Plan 1.5.A analysis for the reasoning.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_FILE = REPO_ROOT / "tools" / "spike_rlr" / "rig_calibration.json"
RIG_CALIBRATION_ALGORITHM_VERSION = "rig_calib_v1"

# Preferred body-center bones in order of fallback
_BODY_BONE_CANDIDATES = (
    "Root",
    "Bip01",
    "Bip01 Pelvis",
    "Bip02",
    "Bip02 Pelvis",
    "Pelvis",
    "Hips",
    "Spine1",
    "Spine",
    "Bone",
)

_BODY_BASIS_BONE_CANDIDATES = {
    "pelvis": ("Bip01 Pelvis", "Bip02 Pelvis", "Pelvis", "Hips"),
    "spine": ("Bip01 Spine2", "Bip02 Spine2", "Spine2", "Spine1", "Spine"),
    "left_clavicle": (
        "Bip01 L Clavicle",
        "Bip02 L Clavicle",
        "LeftClavicle",
        "LeftShoulder",
        "mixamorig LeftShoulder",
    ),
    "right_clavicle": (
        "Bip01 R Clavicle",
        "Bip02 R Clavicle",
        "RightClavicle",
        "RightShoulder",
        "mixamorig RightShoulder",
    ),
}

# Quaternius Animal Pack dog/cat rigs use numeric bones.  The semantic
# longitudinal axis is still unambiguous: Bone is the rear torso/root,
# Bone.001/.002 advance toward the neck/head, and .010/.013 are the paired
# rear feet.  UE sanitizes dots to underscores, which the normalized-name
# lookup below intentionally treats as equivalent.
_QUADRUPED_BASIS_BONE_CANDIDATES = {
    "rear": ("Bone", "Hips"),
    "front": ("Bone.002", "Bone.001", "Shoulders"),
    "body": ("Bone", "Hips", "Body"),
    "left_foot": ("Bone.010", "IKBackLeft", "BackFoot.L"),
    "right_foot": ("Bone.013", "IKBackRight", "BackFoot.R"),
}


def _integer_return_value(value) -> int:
    """Normalize direct and as-dict Unreal integer return values."""
    current = value
    for _ in range(2):
        if not isinstance(current, dict):
            return int(current)
        lowered = {str(key).lower(): item for key, item in current.items()}
        if "returnvalue" not in lowered:
            break
        current = lowered["returnvalue"]
    return int(current)


def _name_return_value(value) -> str:
    current = value
    for _ in range(2):
        if not isinstance(current, dict):
            return str(current)
        lowered = {str(key).lower(): item for key, item in current.items()}
        if "returnvalue" not in lowered:
            break
        current = lowered["returnvalue"]
    return str(current)


def _normalized_bone_name(name: str) -> str:
    return "".join(character.lower() for character in str(name) if character.isalnum())


def _component_bone_names(component):
    bone_count = _integer_return_value(component.GetNumBones())
    return [
        _name_return_value(component.GetBoneName(BoneIndex=index))
        for index in range(bone_count)
    ]


def _unit_vector(vector, *, label: str):
    value = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(value))
    if length < 1e-6:
        raise ValueError(f"{label} vector is degenerate")
    return value / length


def body_basis_from_positions(*, pelvis, spine, left_clavicle, right_clavicle):
    """Build a semantic body basis from same-frame world-space bone positions."""
    pelvis = np.asarray(pelvis, dtype=np.float64)
    spine = np.asarray(spine, dtype=np.float64)
    left_clavicle = np.asarray(left_clavicle, dtype=np.float64)
    right_clavicle = np.asarray(right_clavicle, dtype=np.float64)

    up = _unit_vector(spine - pelvis, label="body up")
    right = right_clavicle - left_clavicle
    right = right - up * float(np.dot(right, up))
    right = _unit_vector(right, label="body right")
    # UE's body frame is +X forward, +Y right, +Z up, so right x up
    # recovers forward. Reversing this cross product points through the back.
    forward = _unit_vector(np.cross(right, up), label="body forward")
    forward_xy = forward[:2]
    if float(np.linalg.norm(forward_xy)) < 1e-6:
        raise ValueError("body forward has no horizontal component")

    return {
        "up_vector_ue": up.tolist(),
        "right_vector_ue": right.tolist(),
        "forward_vector_ue": forward.tolist(),
        "forward_yaw_ue_deg": float(np.degrees(np.arctan2(
            forward[1], forward[0]
        ))),
        "up_alignment_z": float(up[2]),
    }


def quadruped_basis_from_positions(
    *, rear, front, body, left_foot, right_foot
):
    """Build a quadruped body basis from torso and paired-foot anchors."""
    rear = np.asarray(rear, dtype=np.float64)
    front = np.asarray(front, dtype=np.float64)
    body = np.asarray(body, dtype=np.float64)
    left_foot = np.asarray(left_foot, dtype=np.float64)
    right_foot = np.asarray(right_foot, dtype=np.float64)

    feet_center = 0.5 * (left_foot + right_foot)
    up = _unit_vector(body - feet_center, label="quadruped up")
    forward = front - rear
    forward = forward - up * float(np.dot(forward, up))
    forward = _unit_vector(forward, label="quadruped forward")
    right = _unit_vector(np.cross(up, forward), label="quadruped right")
    forward_xy = forward[:2]
    if float(np.linalg.norm(forward_xy)) < 1e-6:
        raise ValueError("quadruped forward has no horizontal component")
    anatomical_right = right_foot - left_foot
    anatomical_right = anatomical_right - up * float(np.dot(anatomical_right, up))
    anatomical_alignment = float(
        np.dot(right, _unit_vector(anatomical_right, label="quadruped anatomical right"))
    )
    return {
        "basis_kind": "quadruped_longitudinal_v1",
        "up_vector_ue": up.tolist(),
        "right_vector_ue": right.tolist(),
        "forward_vector_ue": forward.tolist(),
        "forward_yaw_ue_deg": float(
            np.degrees(np.arctan2(forward[1], forward[0]))
        ),
        "up_alignment_z": float(up[2]),
        "anatomical_right_alignment": anatomical_alignment,
    }


def sample_body_basis_in_frame(actor, *, unreal_service=None, diagnostics=None):
    """Sample humanoid or quadruped body axes inside an active SPEAR frame."""
    if unreal_service is None:
        raise RuntimeError(
            "unreal_service is required for safe UClass handle marshalling"
        )
    try:
        component = select_skeletal_mesh_component(
            unreal_service=unreal_service,
            actor=actor,
            diagnostics=diagnostics,
        )
        if component is None:
            return None
        available_names = _component_bone_names(component)
        by_normalized_name = {
            _normalized_bone_name(name): name for name in available_names
        }
        def match_roles(candidate_map):
            matched = {}
            for role, candidates in candidate_map.items():
                for candidate in candidates:
                    actual = by_normalized_name.get(_normalized_bone_name(candidate))
                    if actual is not None:
                        matched[role] = actual
                        break
            return matched

        human_names = match_roles(_BODY_BASIS_BONE_CANDIDATES)
        quadruped_names = match_roles(_QUADRUPED_BASIS_BONE_CANDIDATES)
        if len(human_names) == len(_BODY_BASIS_BONE_CANDIDATES):
            matched_names = human_names
            basis_builder = body_basis_from_positions
            basis_kind = "humanoid_semantic_v1"
        elif len(quadruped_names) == len(_QUADRUPED_BASIS_BONE_CANDIDATES):
            matched_names = quadruped_names
            basis_builder = quadruped_basis_from_positions
            basis_kind = "quadruped_longitudinal_v1"
        else:
            if diagnostics is not None:
                diagnostics.append({
                    "stage": "body_basis_bone_lookup",
                    "candidate_schemes": {
                        "humanoid": {
                            role: list(candidates)
                            for role, candidates in _BODY_BASIS_BONE_CANDIDATES.items()
                        },
                        "quadruped": {
                            role: list(candidates)
                            for role, candidates in _QUADRUPED_BASIS_BONE_CANDIDATES.items()
                        },
                    },
                    "matched_humanoid_roles": sorted(human_names),
                    "matched_quadruped_roles": sorted(quadruped_names),
                    "available_bone_names": available_names,
                })
            return None

        positions = {}
        for role, bone_name in matched_names.items():
            position = sample_body_bone_position_in_frame(
                actor,
                bone_name,
                unreal_service=unreal_service,
                diagnostics=diagnostics,
            )
            if position is None:
                return None
            positions[role] = position

        basis = basis_builder(**positions)
        basis.setdefault("basis_kind", basis_kind)
        basis["bone_names"] = matched_names
        basis["positions_ue_cm"] = {
            role: np.asarray(position, dtype=np.float64).tolist()
            for role, position in positions.items()
        }
        return basis
    except Exception as error:
        if diagnostics is not None:
            diagnostics.append({
                "stage": "body_basis",
                "error_type": type(error).__name__,
                "error": str(error),
            })
        return None


def select_skeletal_mesh_component(*, unreal_service, actor, diagnostics=None):
    """Select the actor's populated skeletal component.

    A SkeletalMeshActor Blueprint can expose an empty inherited component
    before its actual imported 80-bone mesh component. Unreal's singular
    GetComponentByClass-style lookup therefore is not sufficient here.
    """
    components = unreal_service.get_components_by_class(
        actor=actor,
        uclass="USkeletalMeshComponent",
    )
    if components is None:
        components = []
    elif not isinstance(components, (list, tuple)):
        components = [components]

    populated = []
    inventory = []
    for index, component in enumerate(components):
        try:
            bone_count = _integer_return_value(component.GetNumBones())
        except Exception as error:
            inventory.append({
                "component_index": index,
                "error_type": type(error).__name__,
                "error": str(error),
            })
            continue
        inventory.append({
            "component_index": index,
            "bone_count": bone_count,
        })
        if bone_count > 0:
            populated.append((bone_count, -index, component))

    if populated:
        return max(populated, key=lambda item: (item[0], item[1]))[2]

    if diagnostics is not None:
        diagnostics.append({
            "stage": "component_selection",
            "error_type": "MissingRiggedComponent",
            "error": "no USkeletalMeshComponent with one or more bones",
            "component_inventory": inventory,
        })
    return None


def _yaw_difference_deg(a: float, b: float) -> float:
    """Signed shortest angular difference b-a in (-180, 180] degrees."""
    d = ((b - a + 180.0) % 360.0) - 180.0
    return d


def _assert_yaw_ok(observed: float, expected: float, tolerance_deg: float,
                    context: str) -> None:
    diff = abs(_yaw_difference_deg(observed, expected))
    if diff > tolerance_deg:
        raise AssertionError(
            f"[{context}] rig direction check FAILED: observed body-forward "
            f"yaw = {observed:.1f} deg, expected = {expected:.1f} deg, "
            f"diff = {diff:.1f} deg > tolerance {tolerance_deg:.1f} deg. "
            f"This usually means the rig walked in the WRONG direction. "
            f"Root causes: (a) rig walking_forward_yaw_offset_deg is wrong; "
            f"(b) mesh not yet approved-and-oriented in tmp/hy3d_batch/approved/; "
            f"(c) room world<->UE convention (position/rotation) got desynced."
        )


def sample_body_bone_position_in_frame(
    actor,
    bone_name: str,
    *,
    unreal_service=None,
    diagnostics=None,
):
    """Query a bone's world-space location via SPEAR RPC.

    IMPORTANT: caller must be inside an active `instance.begin_frame()`
    context — this function does NOT open its own frame. That was the bug
    in v1: opening a nested begin_frame after the render loop tore down
    frame state triggered engine_service.begin_frame:157 assert False.

    Returns np.ndarray shape (3,) in UE cm world frame, or None if the bone
    doesn't exist.
    """
    if unreal_service is None:
        raise RuntimeError(
            "unreal_service is required for safe UClass handle marshalling"
        )
    try:
        # The service wrapper resolves USkeletalMeshComponent to a real UClass
        # handle. Calling actor.GetComponentByClass with a class-path string
        # reaches SPEAR's native pointer parser and asserts on non-0x input.
        comp = select_skeletal_mesh_component(
            unreal_service=unreal_service,
            actor=actor,
            diagnostics=diagnostics,
        )
        if comp is None:
            return None
        bone_index = int(comp.GetBoneIndex(BoneName=bone_name))
        if bone_index < 0:
            if diagnostics is not None:
                diagnostics.append({
                    "bone_name": str(bone_name),
                    "stage": "bone_lookup",
                    "error_type": "MissingBone",
                    "error": f"GetBoneIndex returned {bone_index}",
                })
            return None
        tf = comp.GetBoneTransform(
            InBoneName=bone_name,
            TransformSpace="RTS_World",
            as_dict=True,
        )
        if isinstance(tf, dict) and "ReturnValue" in tf:
            tf = tf["ReturnValue"]
        if isinstance(tf, dict):
            lowered = {str(key).lower(): value for key, value in tf.items()}
            loc = lowered.get("translation") or lowered.get("location")
        else:
            loc = getattr(tf, "Translation", None) or getattr(tf, "Location", None)
        if loc is None:
            if diagnostics is not None:
                diagnostics.append({
                    "bone_name": str(bone_name),
                    "stage": "parse",
                    "error_type": type(tf).__name__,
                    "error": f"missing Translation in {repr(tf)[:500]}",
                })
            return None
        if isinstance(loc, dict):
            lowered_loc = {str(key).lower(): value for key, value in loc.items()}
            return np.array(
                [lowered_loc["x"], lowered_loc["y"], lowered_loc["z"]],
                dtype=np.float64,
            )
        return np.array([loc.x, loc.y, loc.z], dtype=np.float64)
    except Exception as error:
        if diagnostics is not None:
            diagnostics.append({
                "bone_name": str(bone_name),
                "stage": "query",
                "error_type": type(error).__name__,
                "error": str(error),
            })
        return None


def find_body_bone_in_frame(
    actor,
    *,
    unreal_service=None,
    diagnostics=None,
) -> Optional[str]:
    """Return the first available candidate bone name on this actor.

    Must be called inside an active begin_frame context."""
    available_names = None
    try:
        component = select_skeletal_mesh_component(
            unreal_service=unreal_service,
            actor=actor,
        )
        if component is not None:
            available_names = _component_bone_names(component)
    except Exception:
        available_names = None

    if available_names is not None:
        by_normalized_name = {
            _normalized_bone_name(name): name for name in available_names
        }
        query_names = [
            by_normalized_name[_normalized_bone_name(candidate)]
            for candidate in _BODY_BONE_CANDIDATES
            if _normalized_bone_name(candidate) in by_normalized_name
        ]
    else:
        query_names = list(_BODY_BONE_CANDIDATES)

    for query_name in query_names:
        pos = sample_body_bone_position_in_frame(
            actor,
            query_name,
            unreal_service=unreal_service,
            diagnostics=diagnostics,
        )
        if pos is not None:
            return query_name
    if diagnostics is not None:
        if available_names is not None:
            diagnostics.append({
                "stage": "bone_inventory",
                "available_bone_names": available_names,
            })
        else:
            diagnostics.append({
                "stage": "bone_inventory",
                "error_type": "UnavailableBoneInventory",
                "error": "GetBoneName inventory could not be read",
            })
    return None


def assert_body_yaw_from_positions(pos_start, pos_end, expected_yaw_world_deg: float,
                                     tolerance_deg: float = 15.0,
                                     context: str = "clip") -> None:
    """Compare a velocity yaw (from two already-sampled positions) to expected.

    No SPEAR access — pure math. Caller samples pos_start and pos_end inside
    two different begin_frame windows (e.g. frames T and T+N of the render
    loop) and then passes them here.

    Raises AssertionError if outside tolerance; silently skips if actor
    isn't moving (pos_end - pos_start too small).
    """
    if pos_start is None or pos_end is None:
        raise RuntimeError(f"[{context}] missing bone position sample")
    v = np.asarray(pos_end) - np.asarray(pos_start)
    if np.linalg.norm(v[:2]) < 1e-3:
        return  # not moving — skip (paused / hold segment)
    observed_yaw = float(np.degrees(np.arctan2(v[1], v[0])))
    _assert_yaw_ok(observed=observed_yaw, expected=expected_yaw_world_deg,
                    tolerance_deg=tolerance_deg, context=context)


# ---- Legacy convenience wrappers (open their own begin_frame windows) ----
# Kept for tests + calibration CLI. Do NOT call these from inside a render
# loop that has its own frame management.

def _sample_body_bone_position(actor, instance, bone_name: str):
    unreal_service = instance.get_game().unreal_service
    with instance.begin_frame():
        pos = sample_body_bone_position_in_frame(
            actor,
            bone_name,
            unreal_service=unreal_service,
        )
    with instance.end_frame():
        pass
    return pos


def _find_body_bone(actor, instance) -> Optional[str]:
    for name in _BODY_BONE_CANDIDATES:
        pos = _sample_body_bone_position(actor, instance, name)
        if pos is not None:
            return name
    return None


def calibrate_rig_forward_from_velocity(actor, instance, n_step_frames: int = 30) -> float:
    """Spawn actor at (0,0,0) with body_yaw=0, play walking, return the observed
    forward yaw (world-frame degrees). Caller uses this as offset baseline.

    This is a static-scene calibration: caller must have already spawned the
    actor + set body yaw to 0 before calling. Opens its own begin_frame
    windows — safe to call between clips, NOT inside another begin_frame.
    """
    body_bone = _find_body_bone(actor, instance)
    if body_bone is None:
        raise RuntimeError("no body-center bone found on actor")
    pos_start = _sample_body_bone_position(actor, instance, body_bone)
    instance.step(num_frames=n_step_frames)
    pos_end = _sample_body_bone_position(actor, instance, body_bone)
    if pos_start is None or pos_end is None:
        raise RuntimeError("failed to sample bone positions")
    v = pos_end - pos_start
    if np.linalg.norm(v[:2]) < 1e-3:
        raise RuntimeError(
            f"observed velocity too small ({np.linalg.norm(v):.4f} cm) — "
            f"is the animation actually playing?"
        )
    return float(np.degrees(np.arctan2(v[1], v[0])))


def assert_body_forward(actor, instance, expected_yaw_world_deg: float,
                         tolerance_deg: float = 15.0, n_step_frames: int = 5,
                         context: str = "clip") -> None:
    """DEPRECATED for in-render-loop use.

    Uses its own begin_frame windows — will crash if called after a render
    loop has torn down frame state. For in-render-loop use, call
    sample_body_bone_position_in_frame() + assert_body_yaw_from_positions().
    """
    body_bone = _find_body_bone(actor, instance)
    if body_bone is None:
        raise RuntimeError(f"[{context}] no body-center bone found")
    pos_start = _sample_body_bone_position(actor, instance, body_bone)
    instance.step(num_frames=n_step_frames)
    pos_end = _sample_body_bone_position(actor, instance, body_bone)
    assert_body_yaw_from_positions(
        pos_start=pos_start, pos_end=pos_end,
        expected_yaw_world_deg=expected_yaw_world_deg,
        tolerance_deg=tolerance_deg, context=context,
    )


def write_rig_calibration_json(tag: str, offset_deg: float,
                                algorithm_version: str) -> None:
    """Write/update rig_calibration.json for one tag."""
    p = CALIBRATION_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        db = json.loads(p.read_text())
    else:
        db = {}
    db[tag] = {
        "walking_forward_yaw_offset_deg": float(offset_deg),
        "algorithm_version": algorithm_version,
        "calibrated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    p.write_text(json.dumps(db, indent=2, sort_keys=True))


def read_rig_calibration_json(tag: str) -> Optional[dict]:
    p = CALIBRATION_FILE
    if not p.exists():
        return None
    db = json.loads(p.read_text())
    if not isinstance(db, dict):
        return None
    entry = db.get(tag)
    # Skip doc-only fields (keys starting with "_")
    if not isinstance(entry, dict):
        return None
    return entry
