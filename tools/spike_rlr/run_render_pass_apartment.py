"""SPEAR/UE render pass for apartment_v1_spec.

Flow:
  1. Load apartment_0000 map via configure_instance (SPEAR RPC).
  2. Enumerate all StaticMeshActors; classify each; destroy furniture
     actors NOT in the spec's furniture_mode keep-set.
     - furniture_mode == 'shell'   -> keep no furniture (only structural
       shell stays)
     - furniture_mode == 'subset'  -> keep actors in
       furniture_include_categories (core + decoration by default)
     - furniture_mode == 'full'    -> keep all furniture
     Shell actors are always kept.
  3. Spawn dogs via scene_two_dogs_apartment.
  4. Spawn 1 forward camera glued to mic pose, FOV from spec.
  5. Per-frame: advance dog animation, capture, save png.
  6. ffmpeg png sequence -> mp4.

Must be run under spear-env (has spear_ext + SPEAR RPC bindings) with
DISPLAY=:99 + VK_ICD_FILENAMES set. See tools/spike_rlr/run_apartment.sh
for env setup.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "data" / "apartment_v1_spec.json"
DEFAULT_OUT = REPO_ROOT / "tmp" / "spike_output_apartment"

sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "gpurir_scenes"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))

from render_in_apartment import (  # noqa: E402
    APARTMENT_MAP, configure_instance, spawn_camera, read_frame,
)
from gpurir_scenes.run_render_pass import (  # noqa: E402
    APARTMENT_MIC_ORIGIN_CM, APARTMENT_FLOOR_Z_CM, M2CM,
    _world_from_scene, _yaw_world_to_ue, _spawn_animal, _step_animated,
)
from apartment_actor_classifier import classify_actor, SHELL_LABELS  # noqa: E402
from scene_two_dogs_apartment import (  # noqa: E402
    compose_two_dog_scene_apartment,
    _shell_wall_bboxes,
    _static_obstacle_bboxes,
)
from profiling import StageTimer  # noqa: E402


def _rig_assert_enabled() -> bool:
    """Check if per-clip rig direction assertion should run.

    Enabled by: SPEAR_RIG_ASSERT=1 env var OR --rig-assert CLI flag.
    Opt-in to preserve fast Plan 1 iteration; will be on-by-default in Plan 2.
    """
    return os.environ.get("SPEAR_RIG_ASSERT", "0") == "1" or \
           any("--rig-assert" in a for a in sys.argv)


def _assert_source_review_gates(spec):
    """Route technical-spike humans through their artifact-locked gate."""
    from human_apartment_gate import (
        ALLOWED_HUMAN_SPIKES,
        NATIVE_ROCKETBOX_HUMAN_CANDIDATES,
        assert_batch_native_rocketbox_apartment_ready,
        assert_human_apartment_ready,
        assert_native_rocketbox_apartment_ready,
        is_batch_native_rocketbox_human_candidate,
    )

    skip_review_gate = os.environ.get("SPEAR_SKIP_REVIEW_GATE", "0") == "1"
    human_gate_evidence = []
    for src in spec.get("sources", []):
        tag = src.get("tag")
        if not tag:
            continue
        if tag in NATIVE_ROCKETBOX_HUMAN_CANDIDATES:
            evidence = assert_native_rocketbox_apartment_ready(tag)
            human_gate_evidence.append(evidence)
            print(f"[apt_render] native Rocketbox artifact gate OK for {tag}")
        elif is_batch_native_rocketbox_human_candidate(tag):
            evidence = assert_batch_native_rocketbox_apartment_ready(tag)
            human_gate_evidence.append(evidence)
            print(f"[apt_render] batch native Rocketbox artifact gate OK for {tag}")
        elif tag in ALLOWED_HUMAN_SPIKES:
            evidence = assert_human_apartment_ready(
                tag,
                skip_review_gate=skip_review_gate,
            )
            human_gate_evidence.append(evidence)
            print(f"[apt_render] humanoid artifact gate OK for {tag}")
        elif not skip_review_gate:
            from review_gate import assert_mesh_approved, MeshNotApprovedError
            try:
                assert_mesh_approved(tag)
                print(f"[apt_render] review_gate OK for {tag}")
            except MeshNotApprovedError as exc:
                print(f"[apt_render] review_gate REFUSED {tag}: {exc}")
                raise
    return human_gate_evidence


def _check_no_clipping_apartment(spec_dict, scene, cats):
    """Safety net: raise AssertionError if any animal trajectory point enters
    any kept furniture or shell wall bbox. Called before UE render so we
    catch bad specs cheaply rather than staring at a 30 s render of a dog
    walking through a sofa."""
    policy = spec_dict.get("source_collision_policy", "furniture_and_walls")
    if policy == "walls_only_center":
        obstacles = _shell_wall_bboxes(spec_dict)
        policy_hint = "walls_only_center"
    else:
        obstacles = _static_obstacle_bboxes(spec_dict, cats)
        policy_hint = "furniture_and_walls"
    for a in scene.animals:
        for k, xyz in enumerate(a.trajectory_m):
            x, y = float(xyz[0]), float(xyz[1])
            for x0, y0, x1, y1 in obstacles:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    raise AssertionError(
                        f"{a.tag} clips obstacle at frame {k}: "
                        f"pos=({x:.2f}, {y:.2f}) inside bbox "
                        f"[{x0:.2f},{y0:.2f}]-[{x1:.2f},{y1:.2f}]. "
                        f"Fix spec start/end so planner can route around, "
                        f"or set spec 'furniture_exclude_actors' to remove it. "
                        f"source_collision_policy={policy_hint}."
                    )


def _check_source_separation(spec_dict, scene):
    """Fail before UE launch when synchronized source centers get too close."""
    required_m = float(spec_dict.get("minimum_source_separation_m", 0.0))
    if not np.isfinite(required_m) or required_m < 0.0:
        raise ValueError("minimum_source_separation_m must be finite and non-negative")

    pairs = []
    for source_a, source_b in combinations(scene.animals, 2):
        trajectory_a = np.asarray(source_a.trajectory_m, dtype=np.float64)
        trajectory_b = np.asarray(source_b.trajectory_m, dtype=np.float64)
        if trajectory_a.shape != trajectory_b.shape:
            raise ValueError(
                f"trajectory shape mismatch for {source_a.tag}/{source_b.tag}: "
                f"{trajectory_a.shape} vs {trajectory_b.shape}"
            )
        distances = np.linalg.norm(
            trajectory_a[:, :2] - trajectory_b[:, :2], axis=1
        )
        frame_index = int(np.argmin(distances))
        minimum_m = float(distances[frame_index])
        pair = {
            "tags": [str(source_a.tag), str(source_b.tag)],
            "minimum_distance_m": minimum_m,
            "frame_index": frame_index,
        }
        pairs.append(pair)
        if minimum_m < required_m:
            raise AssertionError(
                f"{source_a.tag}/{source_b.tag} source separation {minimum_m:.3f} m "
                f"at frame {frame_index} is below {required_m:.3f} m"
            )
    return {
        "status": "passed",
        "minimum_required_m": required_m,
        "pairs": pairs,
    }


def _load_categories():
    return json.loads((REPO_ROOT / "tools" / "spike_rlr"
                       / "apartment_furniture_categories.json").read_text())


def _apartment_camera_ue_cm(mic_pos_m):
    """Convert an apartment_v1 absolute SSOT mic pose to UE centimeters."""
    mic_pos_m = np.asarray(mic_pos_m, dtype=np.float64)
    return (
        APARTMENT_MIC_ORIGIN_CM[0] + mic_pos_m[0] * M2CM,
        APARTMENT_MIC_ORIGIN_CM[1] - mic_pos_m[1] * M2CM,
        APARTMENT_FLOOR_Z_CM + mic_pos_m[2] * M2CM,
    )


def _apartment_ue_cm_to_ssot_m(pos_ue_cm):
    """Convert an apartment UE centimeter position to absolute SSOT meters."""
    x_cm, y_cm, z_cm = [float(v) for v in pos_ue_cm]
    return (
        (x_cm - APARTMENT_MIC_ORIGIN_CM[0]) / M2CM,
        -(y_cm - APARTMENT_MIC_ORIGIN_CM[1]) / M2CM,
        (z_cm - APARTMENT_FLOOR_Z_CM) / M2CM,
    )


def _actor_visual_center_ssot_m(actor):
    """Return actor bounds center in absolute SSOT meters."""
    bounds = actor.GetActorBounds(bOnlyCollidingComponents=False, as_dict=True)
    origin = bounds["Origin"]
    return _apartment_ue_cm_to_ssot_m((origin["x"], origin["y"], origin["z"]))


def _dict_components(value, names):
    """Read numeric Unreal struct fields regardless of key capitalization."""
    expected = [name.lower() for name in names]
    current = value
    for _ in range(3):
        if not isinstance(current, dict):
            break
        lowered = {str(key).lower(): item for key, item in current.items()}
        if all(name in lowered for name in expected):
            return [float(lowered[name]) for name in expected]
        if "returnvalue" in lowered and isinstance(lowered["returnvalue"], dict):
            current = lowered["returnvalue"]
            continue
        if len(current) == 1:
            only_value = next(iter(current.values()))
            if isinstance(only_value, dict):
                current = only_value
                continue
        break
    keys = list(current) if isinstance(current, dict) else type(current).__name__
    raise KeyError(f"expected Unreal components {expected}, got {keys}")


def _snap_actor_bounds_bottom_to_floor(
    actor,
    *,
    floor_z_cm: float = APARTMENT_FLOOR_Z_CM,
    maximum_abs_correction_cm: float = 15.0,
    residual_tolerance_cm: float = 0.1,
):
    """Move only actor world Z so the evaluated mesh bottom meets the floor."""
    limit_cm = float(maximum_abs_correction_cm)
    tolerance_cm = float(residual_tolerance_cm)
    if (
        not np.isfinite(limit_cm)
        or limit_cm <= 0.0
        or not np.isfinite(tolerance_cm)
        or tolerance_cm < 0.0
    ):
        raise ValueError("ground snap tolerances must be finite and positive")
    bounds_before = actor.GetActorBounds(
        bOnlyCollidingComponents=False, as_dict=True
    )
    origin_before = _dict_components(bounds_before["Origin"], ("x", "y", "z"))
    extent_before = _dict_components(
        bounds_before["BoxExtent"], ("x", "y", "z")
    )
    bottom_before = float(origin_before[2] - extent_before[2])
    correction_cm = float(floor_z_cm) - bottom_before
    if not np.isfinite(correction_cm) or abs(correction_cm) > limit_cm:
        raise AssertionError(
            f"ground snap correction {correction_cm:.3f} cm exceeds "
            f"{limit_cm:.3f} cm; bounds are implausible"
        )

    location = _dict_components(
        actor.K2_GetActorLocation(as_dict=True), ("x", "y", "z")
    )
    rotation = _dict_components(
        actor.K2_GetActorRotation(as_dict=True), ("roll", "pitch", "yaw")
    )
    actor.K2_SetActorLocationAndRotation(
        NewLocation={
            "X": location[0],
            "Y": location[1],
            "Z": location[2] + correction_cm,
        },
        NewRotation={
            "Roll": rotation[0],
            "Pitch": rotation[1],
            "Yaw": rotation[2],
        },
        bSweep=False,
        bTeleport=True,
    )
    bounds_after = actor.GetActorBounds(
        bOnlyCollidingComponents=False, as_dict=True
    )
    origin_after = _dict_components(bounds_after["Origin"], ("x", "y", "z"))
    extent_after = _dict_components(
        bounds_after["BoxExtent"], ("x", "y", "z")
    )
    bottom_after = float(origin_after[2] - extent_after[2])
    residual_cm = bottom_after - float(floor_z_cm)
    if not np.isfinite(residual_cm) or abs(residual_cm) > tolerance_cm:
        raise AssertionError(
            f"ground snap residual {residual_cm:.3f} cm exceeds "
            f"{tolerance_cm:.3f} cm"
        )
    return {
        "schema": "ue_dynamic_ground_snap_v1",
        "status": "passed",
        "floor_z_cm": float(floor_z_cm),
        "bounds_bottom_before_cm": bottom_before,
        "applied_z_correction_cm": correction_cm,
        "bounds_bottom_after_cm": bottom_after,
        "residual_clearance_cm": residual_cm,
        "maximum_abs_correction_cm": limit_cm,
        "residual_tolerance_cm": tolerance_cm,
        "horizontal_location_unchanged": True,
        "rotation_unchanged": True,
    }


def _actor_runtime_frame_state(
    actor,
    placement,
    frame_i: int,
    *,
    floor_z_cm: float = APARTMENT_FLOOR_Z_CM,
    penetration_tolerance_cm: float = 1.0,
):
    """Capture raw UE transform, bounds, and floor contact for one frame."""
    bounds = actor.GetActorBounds(bOnlyCollidingComponents=False, as_dict=True)
    origin = _dict_components(bounds["Origin"], ("x", "y", "z"))
    extent = _dict_components(bounds["BoxExtent"], ("x", "y", "z"))
    minimum = [origin[i] - extent[i] for i in range(3)]
    maximum = [origin[i] + extent[i] for i in range(3)]

    location = _dict_components(
        actor.K2_GetActorLocation(as_dict=True), ("x", "y", "z")
    )
    rotation = _dict_components(
        actor.K2_GetActorRotation(as_dict=True), ("roll", "pitch", "yaw")
    )
    scale = _dict_components(
        actor.GetActorScale3D(as_dict=True), ("x", "y", "z")
    )

    clearance_cm = float(minimum[2] - float(floor_z_cm))
    penetration_cm = max(0.0, -clearance_cm)
    tolerance_cm = float(penetration_tolerance_cm)
    return {
        "frame_index": int(frame_i),
        "animation_name": str(
            getattr(placement, "wanted_anim", "Walking") or "Walking"
        ),
        "root_transform_ue": {
            "location_cm": location,
            "rotation_deg": rotation,
            "scale": scale,
        },
        "root_location_ssot_m": [
            float(value) for value in _apartment_ue_cm_to_ssot_m(location)
        ],
        "bounds_ue": {
            "origin_cm": origin,
            "extent_cm": extent,
            "minimum_cm": minimum,
            "maximum_cm": maximum,
        },
        "bounds_origin_ssot_m": [
            float(value) for value in _apartment_ue_cm_to_ssot_m(origin)
        ],
        "floor_contact": {
            "floor_z_ue_cm": float(floor_z_cm),
            "bounds_bottom_z_ue_cm": float(minimum[2]),
            "clearance_cm": clearance_cm,
            "penetration_cm": penetration_cm,
            "penetration_tolerance_cm": tolerance_cm,
            "within_penetration_tolerance": penetration_cm <= tolerance_cm,
        },
    }


def _summarize_actor_runtime_frames(
    frames,
    *,
    expected_frame_count: int,
    penetration_tolerance_cm: float = 1.0,
):
    """Summarize the evidence used by the human apartment floor gate."""
    if len(frames) != int(expected_frame_count):
        raise AssertionError(
            f"runtime frame count {len(frames)} != {int(expected_frame_count)}"
        )
    if not frames:
        raise AssertionError("runtime frame evidence must not be empty")

    contacts = [frame["floor_contact"] for frame in frames]
    penetrations = [float(contact["penetration_cm"]) for contact in contacts]
    clearances = [float(contact["clearance_cm"]) for contact in contacts]
    rotations = [frame["root_transform_ue"]["rotation_deg"] for frame in frames]
    heights = [
        float(frame["bounds_ue"]["maximum_cm"][2])
        - float(frame["bounds_ue"]["minimum_cm"][2])
        for frame in frames
    ]
    maximum_penetration = max(penetrations)
    tolerance_cm = float(penetration_tolerance_cm)
    summary = {
        "frame_count": len(frames),
        "animation_names": sorted({frame["animation_name"] for frame in frames}),
        "minimum_bounds_height_cm": min(heights),
        "maximum_bounds_height_cm": max(heights),
        "minimum_floor_clearance_cm": min(clearances),
        "maximum_floor_clearance_cm": max(clearances),
        "maximum_floor_penetration_cm": maximum_penetration,
        "penetration_tolerance_cm": tolerance_cm,
        "maximum_abs_root_roll_deg": max(abs(float(rot[0])) for rot in rotations),
        "maximum_abs_root_pitch_deg": max(abs(float(rot[1])) for rot in rotations),
        "floor_contact_status": (
            "passed" if maximum_penetration <= tolerance_cm else "failed"
        ),
    }
    ground_snaps = [
        frame.get("ground_snap")
        for frame in frames
        if isinstance(frame.get("ground_snap"), dict)
    ]
    if ground_snaps:
        if len(ground_snaps) != len(frames) or any(
            snap.get("status") != "passed" for snap in ground_snaps
        ):
            raise AssertionError("dynamic ground snap evidence is incomplete")
        summary["dynamic_ground_snap"] = {
            "status": "passed",
            "frame_count": len(ground_snaps),
            "minimum_applied_z_correction_cm": min(
                float(snap["applied_z_correction_cm"]) for snap in ground_snaps
            ),
            "maximum_applied_z_correction_cm": max(
                float(snap["applied_z_correction_cm"]) for snap in ground_snaps
            ),
            "maximum_abs_residual_clearance_cm": max(
                abs(float(snap["residual_clearance_cm"])) for snap in ground_snaps
            ),
        }
    else:
        summary["dynamic_ground_snap"] = {"status": "not_requested"}
    return summary


def _build_actor_visual_metadata(
    *,
    scene,
    visual_centers,
    runtime_frames,
    human_gate_evidence,
    n_frames: int,
    penetration_tolerance_cm: float = 1.0,
    rig_direction_evidence=None,
):
    """Build and enforce the per-frame runtime evidence publication."""
    human_tags = {
        str(evidence["tag"])
        for evidence in human_gate_evidence
        if evidence.get("tag")
    }
    sources = []
    checks = {}
    for placement in scene.animals:
        tag = placement.tag
        centers = visual_centers.get(tag, [])
        frames = runtime_frames.get(tag, [])
        if len(centers) != int(n_frames):
            raise AssertionError(
                f"{tag} visual center count {len(centers)} != {int(n_frames)}"
            )
        summary = _summarize_actor_runtime_frames(
            frames,
            expected_frame_count=n_frames,
            penetration_tolerance_cm=penetration_tolerance_cm,
        )
        expected_animation = str(
            getattr(placement, "wanted_anim", "Walking") or "Walking"
        )
        if summary["animation_names"] != [expected_animation]:
            raise AssertionError(
                f"{tag} runtime animation names {summary['animation_names']} "
                f"!= {[expected_animation]}"
            )
        if tag in human_tags:
            if summary["floor_contact_status"] != "passed":
                raise AssertionError(
                    f"{tag} floor penetration "
                    f"{summary['maximum_floor_penetration_cm']:.3f} cm exceeds "
                    f"{summary['penetration_tolerance_cm']:.3f} cm"
                )
            if summary["maximum_abs_root_roll_deg"] > 1.0 or summary[
                "maximum_abs_root_pitch_deg"
            ] > 1.0:
                raise AssertionError(
                    f"{tag} root orientation is not upright: "
                    f"roll={summary['maximum_abs_root_roll_deg']:.3f}, "
                    f"pitch={summary['maximum_abs_root_pitch_deg']:.3f}"
                )

        checks[tag] = summary
        sources.append({
            "tag": tag,
            "animation_name": expected_animation,
            "visual_center_world_xyz_per_frame": centers,
            "runtime_summary": summary,
            "runtime_frames": frames,
        })

    return {
        "schema_version": "apartment_actor_visual_metadata_v2",
        "coordinate_frame": "absolute_ssot_m",
        "source": (
            "UE actor root transform and GetActorBounds captured per rendered frame"
        ),
        "human_gate_evidence": human_gate_evidence,
        "rig_direction_evidence": rig_direction_evidence or {},
        "automatic_checks": {"overall": "passed", "sources": checks},
        "sources": sources,
    }


def _build_rig_direction_evidence(
    scene,
    rig_samples,
    *,
    frame_a: int,
    frame_b: int,
    enabled: bool,
    tolerance_deg: float = 25.0,
):
    """Validate same-frame body forward plus moving-root trajectory direction."""
    from rig_direction_check import (
        _assert_yaw_ok,
        _yaw_difference_deg,
        assert_body_yaw_from_positions,
    )

    evidence = {}
    for placement in scene.animals:
        tag = str(placement.tag)
        if not enabled:
            evidence[tag] = {
                "status": "not_run",
                "reason": "rig direction assertion disabled",
            }
            continue

        samples = rig_samples.get(tag, {})
        body_basis = samples.get("body_basis")
        if body_basis is None:
            diagnostics = samples.get("diagnostics", [])
            raise AssertionError(
                f"{tag} missing body-forward basis at frame {frame_a}; "
                f"diagnostics={json.dumps(diagnostics)}"
            )

        asset_yaw_offset = float(
            getattr(placement, "walking_forward_yaw_offset_deg", 0.0) or 0.0
        )
        semantic_yaw_ssot = (
            float(np.asarray(placement.yaw_deg)[frame_a]) - asset_yaw_offset
        )
        expected_yaw_ue = -semantic_yaw_ssot
        body_forward_yaw_ue = float(body_basis["forward_yaw_ue_deg"])
        up_alignment_z = float(body_basis["up_alignment_z"])
        if up_alignment_z < 0.75:
            raise AssertionError(
                f"{tag} body up alignment {up_alignment_z:.3f} is below 0.750"
            )
        try:
            _assert_yaw_ok(
                observed=body_forward_yaw_ue,
                expected=expected_yaw_ue,
                tolerance_deg=float(tolerance_deg),
                context=f"apartment_v1/{tag}/body_basis",
            )
        except AssertionError as error:
            raise AssertionError(
                f"{error}; body_basis={json.dumps(body_basis, sort_keys=True)}"
            ) from error

        trajectory = np.asarray(placement.trajectory_m, dtype=np.float64)
        displacement = trajectory[frame_b, :2] - trajectory[frame_a, :2]
        moving = float(np.linalg.norm(displacement)) >= 1e-3
        bone_name = samples.get("bone")
        # Actor-root positions are the authority for the explicit world
        # trajectory.  A pelvis/root bone contains gait sway and can diverge
        # by tens of degrees over a short four-frame window on quadrupeds.
        # Keep bone samples as diagnostics and as a compatibility fallback.
        position_a = samples.get("root_a", samples.get("a"))
        position_b = samples.get("root_b", samples.get("b"))
        translation_sample_kind = (
            "actor_root_transform"
            if samples.get("root_a") is not None and samples.get("root_b") is not None
            else "body_bone_fallback"
        )
        if moving:
            if position_a is None or position_b is None:
                diagnostics = samples.get("diagnostics", [])
                raise AssertionError(
                    f"{tag} missing rig direction bone samples at frames "
                    f"{frame_a}/{frame_b}; diagnostics={json.dumps(diagnostics)}"
                )
            position_a = np.asarray(position_a, dtype=np.float64)
            position_b = np.asarray(position_b, dtype=np.float64)
            observed_yaw_ue = float(np.degrees(np.arctan2(
                position_b[1] - position_a[1],
                position_b[0] - position_a[0],
            )))
            expected_root_yaw_ssot = float(np.degrees(np.arctan2(
                displacement[1], displacement[0]
            )))
            expected_root_yaw_ue = -expected_root_yaw_ssot
            assert_body_yaw_from_positions(
                pos_start=position_a,
                pos_end=position_b,
                expected_yaw_world_deg=expected_root_yaw_ue,
                tolerance_deg=float(tolerance_deg),
                context=f"apartment_v1/{tag}/pelvis_translation",
            )
            root_motion_status = "passed"
            position_a_record = position_a.tolist()
            position_b_record = position_b.tolist()
            root_error_deg = abs(_yaw_difference_deg(
                observed_yaw_ue, expected_root_yaw_ue
            ))
        else:
            observed_yaw_ue = None
            expected_root_yaw_ue = None
            root_motion_status = "not_applicable"
            position_a_record = None
            position_b_record = None
            root_error_deg = None

        evidence[tag] = {
            "status": "passed",
            "bone_name": str(bone_name) if bone_name is not None else None,
            "translation_sample_kind": translation_sample_kind,
            "sample_frames": [int(frame_a), int(frame_b)],
            "position_a_ue_cm": position_a_record,
            "position_b_ue_cm": position_b_record,
            "observed_yaw_ue_deg": observed_yaw_ue,
            "expected_yaw_ue_deg": expected_yaw_ue,
            "absolute_error_deg": float(root_error_deg) if root_error_deg is not None else None,
            "root_motion_status": root_motion_status,
            "root_motion_expected_yaw_ue_deg": expected_root_yaw_ue,
            "body_forward_yaw_ue_deg": body_forward_yaw_ue,
            "body_forward_error_deg": abs(_yaw_difference_deg(
                body_forward_yaw_ue, expected_yaw_ue
            )),
            "body_up_alignment_z": up_alignment_z,
            "asset_yaw_offset_ssot_deg": asset_yaw_offset,
            "body_basis": body_basis,
            "tolerance_deg": float(tolerance_deg),
        }
    return evidence


def _resolve_rig_direction_check_windows(spec, n_frames: int):
    """Return validated direction windows, preserving the historical default."""
    configured = spec.get("rig_direction_check_windows")
    if configured is None:
        frame_a = min(30, int(n_frames) // 3)
        return [{
            "label": "primary",
            "frame_a": frame_a,
            "frame_b": min(frame_a + 10, int(n_frames) - 1),
        }]
    if not isinstance(configured, list) or not configured:
        raise ValueError("rig_direction_check_windows must be a non-empty list")

    windows = []
    labels = set()
    for item in configured:
        if not isinstance(item, dict):
            raise ValueError("each rig direction window must be an object")
        label = str(item.get("label", ""))
        if re.fullmatch(r"[a-z0-9_]+", label) is None or label in labels:
            raise ValueError(f"invalid or duplicate rig direction window label: {label!r}")
        frame_a = int(item["frame_a"])
        frame_b = int(item["frame_b"])
        if not 0 <= frame_a < frame_b < int(n_frames):
            raise ValueError(
                f"invalid rig direction window {label}: {frame_a}/{frame_b} "
                f"for {int(n_frames)} frames"
            )
        labels.add(label)
        windows.append({"label": label, "frame_a": frame_a, "frame_b": frame_b})
    return windows


def _build_rig_direction_window_evidence(
    scene,
    rig_samples_by_window,
    *,
    windows,
    enabled: bool,
    tolerance_deg: float = 25.0,
):
    """Aggregate independent body/root direction checks across path segments."""
    by_tag = {str(placement.tag): [] for placement in scene.animals}
    for window in windows:
        label = str(window["label"])
        result = _build_rig_direction_evidence(
            scene,
            rig_samples_by_window.get(label, {}),
            frame_a=int(window["frame_a"]),
            frame_b=int(window["frame_b"]),
            enabled=enabled,
            tolerance_deg=tolerance_deg,
        )
        for tag, evidence in result.items():
            by_tag[str(tag)].append({"label": label, **evidence})

    aggregate = {}
    for tag, window_evidence in by_tag.items():
        statuses = {item["status"] for item in window_evidence}
        aggregate[tag] = {
            "status": "passed" if statuses == {"passed"} else "not_run",
            "window_count": len(window_evidence),
            "windows": window_evidence,
        }
    return aggregate


def _placement_visual_fallback_ssot_m(placement, frame_i: int) -> list[float]:
    if getattr(placement, "trajectory_m", None) is not None:
        point = np.asarray(placement.trajectory_m[frame_i], dtype=np.float64)
    elif getattr(placement, "static_pos_m", None) is not None:
        point = np.asarray(placement.static_pos_m, dtype=np.float64)
    else:
        point = np.zeros(3, dtype=np.float64)
    return [float(v) for v in point[:3]]


def _sanitize_actor_visual_center_ssot_m(center, placement, frame_i: int) -> list[float]:
    """Reject implausible UE actor bounds centers before review overlay.

    Imported Mixamo humanoids can report skeletal bounds far outside the
    apartment even when the rendered mesh is placed correctly. In that case
    the acoustic/source trajectory is a better marker anchor than a bogus
    50-meter-tall bounds center.
    """
    fallback = _placement_visual_fallback_ssot_m(placement, frame_i)
    try:
        arr = np.asarray(center, dtype=np.float64)
    except Exception:
        return fallback
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        return fallback
    horizontal_delta_m = float(np.linalg.norm(arr[:2] - np.asarray(fallback[:2])))
    if horizontal_delta_m > 2.0 or arr[2] < -0.5 or arr[2] > 4.0:
        return fallback
    return [float(v) for v in arr]


def _absolute_apartment_render_scene(scene):
    """Adapt apartment_v1 absolute SSOT trajectories for the shared actor
    spawner, whose apartment transform is relative to scene.mic_pos_m.

    Plan-2 apartment specs store both mic and source positions in the same
    absolute apartment coordinate frame where APARTMENT_MIC_ORIGIN_CM is SSOT
    (0, 0). Setting the render-scene mic anchor to (0, 0) makes the shared
    _world_from_scene helper place actors in that absolute frame.
    """
    return replace(scene, mic_pos_m=(0.0, 0.0, scene.mic_pos_m[2]))


def _compute_keep_set(spec, cats):
    """Given the loaded spec and categories JSON, return the set of actor names
    (from apartment_furniture_map.json) that should be KEPT this clip."""
    mode = spec["furniture_mode"]
    include_cats = set(spec.get("furniture_include_categories", []))
    keep = set()
    if mode == "shell":
        pass  # keep no furniture
    elif mode == "full":
        for c in ("core", "decoration", "misc"):
            keep.update(cats.get(c, []))
    else:  # subset
        for c in include_cats:
            keep.update(cats.get(c, []))
    keep.update(spec.get("furniture_include_actors_extra", []))
    keep.difference_update(spec.get("furniture_exclude_actors", []))
    return keep


def _destroy_unwanted_actors(instance, game, spec, cats):
    """Enumerate all StaticMeshActors; destroy furniture actors NOT in keep.

    Shell actors are always kept. Enumeration + classification + destruction
    all happen inside ONE begin_frame..end_frame pair (SPEAR RPC forbids
    nested begin_frame calls).
    """
    keep_actors = _compute_keep_set(spec, cats)
    n_shell = 0
    n_kept = 0
    n_removed = 0
    with instance.begin_frame():
        actors = game.unreal_service.find_actors_by_class(uclass="AStaticMeshActor")
        print(f"[apt_render] enumerated {len(actors)} StaticMeshActors")

        for actor in actors:
            try:
                name = game.unreal_service.get_stable_name_for_actor(
                    actor=actor, include_unreal_name=True) or "<empty>"
            except Exception:
                name = "<no-stable-name>"
            try:
                b = actor.GetActorBounds(bOnlyCollidingComponents=False, as_dict=True)
                origin = b["Origin"]; ext = b["BoxExtent"]
                bmin_z = origin["z"] - ext["z"]
                bmax_z = origin["z"] + ext["z"]
                x_ext = ext["x"] * 2.0
                y_ext = ext["y"] * 2.0
            except Exception:
                # No bounds -> skip (can't classify safely)
                continue
            label = classify_actor(name, bmin_z, bmax_z, x_ext, y_ext)
            if label in SHELL_LABELS:
                n_shell += 1
                continue  # always keep shell
            # furniture
            if name in keep_actors:
                n_kept += 1
                continue
            # Destroy unwanted furniture actor in place
            try:
                game.unreal_service.destroy_actor(actor=actor)
                n_removed += 1
            except Exception as e:
                print(f"[apt_render] WARN failed to destroy {name!r}: {e}")
    with instance.end_frame():
        pass

    print(f"[apt_render] shell kept: {n_shell}, furniture kept: {n_kept}, "
          f"furniture removed: {n_removed}")


def render_apartment(spec_path: Path, out_dir: Path, csv_path: Path,
                     clip_id: str = "apartment_v1_000"):
    spec = json.loads(spec_path.read_text())
    render_config = spec["render_config"]
    n_frames = int(render_config["n_frames"])
    fps = int(render_config["fps"])
    streaming_warmup_frames = int(
        render_config.get("streaming_warmup_frames", 120)
    )
    camera_warmup_frames = int(render_config.get("camera_warmup_frames", 40))
    cam_cfg = spec["camera_configs"][0]
    fov_deg = float(cam_cfg["fov_deg"])
    yaw_world_deg = float(cam_cfg["yaw_deg"])
    yaw_ue_deg = _yaw_world_to_ue(yaw_world_deg, "apartment")

    # Technical-spike humans cannot use the legacy bypass. Their stable
    # Blender artifacts, UE import hash, and second-process reload must all be
    # current before the packaged runtime is launched.
    human_gate_evidence = _assert_source_review_gates(spec)
    scene = compose_two_dog_scene_apartment(spec_path)
    cats = _load_categories()
    # Safety net: fail fast if any planned trajectory clips a furniture / wall
    # bbox. Cheap (a few hundred us) and saves ~30 s of UE render on bad specs.
    _check_no_clipping_apartment(spec, scene, cats)
    source_separation = _check_source_separation(spec, scene)
    (out_dir / "runtime_gate.json").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "runtime_gate.json").write_text(json.dumps({
        "map": APARTMENT_MAP,
        "clip_id": clip_id,
        "human_gate_evidence": human_gate_evidence,
        "source_separation": source_separation,
    }, indent=2), encoding="utf-8")

    frames_dir = out_dir / "videos" / "apartment_v1_view0"
    frames_dir.mkdir(parents=True, exist_ok=True)

    with StageTimer("ue_render", clip_id=clip_id, csv_path=csv_path):
        rpc_port = int(os.environ.get("SPEAR_APARTMENT_RPC_PORT", "39004"))
        if not 1024 <= rpc_port <= 65535:
            raise ValueError(f"invalid SPEAR_APARTMENT_RPC_PORT: {rpc_port}")
        instance = configure_instance(
            rpc_port=rpc_port,
            fixed_delta_time=1.0 / fps,
        )
        game = instance.get_game()
        try:
            # 1. Destroy unwanted furniture actors
            _destroy_unwanted_actors(instance, game, spec, cats)

            # 2. Spawn camera glued to mic pose
            with instance.begin_frame():
                cam, comp = spawn_camera(
                    game=game,
                    width=int(spec["render_config"]["width"]),
                    height=int(spec["render_config"]["height"]),
                )
                # Set FOV
                try:
                    comp.set_property_value(property_name="FOVAngle",
                                             property_value=fov_deg)
                except Exception as e:
                    print(f"[apt_render] WARN failed to set FOV: {e}")

                # 3. Spawn dogs
                render_scene = _absolute_apartment_render_scene(scene)
                actors = [_spawn_animal(game, a, "apartment", render_scene)
                          for a in scene.animals]

                # Unpause the game
                game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(
                    bPaused=False)
            with instance.end_frame():
                pass

            # Warmup for virtual texture streaming (matches shoebox pipeline)
            instance.step(num_frames=streaming_warmup_frames)

            mic_x_cm, mic_y_cm, mic_z_cm = _apartment_camera_ue_cm(
                spec["mic"]["pos_m"]
            )

            # Set camera to mic pose, one yaw only (Plan-1 single view)
            with instance.begin_frame():
                cam.K2_SetActorLocationAndRotation(
                    NewLocation={"X": mic_x_cm, "Y": mic_y_cm, "Z": mic_z_cm},
                    NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": float(yaw_ue_deg)},
                    bSweep=False, bTeleport=True,
                )
            with instance.end_frame():
                pass
            instance.step(num_frames=camera_warmup_frames)

            # ---- Plan 1.5.B: per-clip rig direction sanity check ----
            # Opt-in via env var (SPEAR_RIG_ASSERT=1) or --rig-assert flag.
            # In-loop implementation: sample Root bone position at frames
            # RIG_SAMPLE_FRAME_A and RIG_SAMPLE_FRAME_B (inside their
            # begin_frame windows), then compare velocity yaw AFTER the
            # loop finishes. No extra begin_frame after teardown, so no
            # engine_service.begin_frame:157 assert False crash.
            rig_assert_on = _rig_assert_enabled()
            rig_windows = _resolve_rig_direction_check_windows(spec, n_frames)
            sample_events = {}
            for window in rig_windows:
                sample_events.setdefault(window["frame_a"], []).append(
                    (window["label"], "a")
                )
                sample_events.setdefault(window["frame_b"], []).append(
                    (window["label"], "b")
                )
            rig_samples_by_window = {
                window["label"]: {} for window in rig_windows
            }
            if rig_assert_on:
                from rig_direction_check import (
                    find_body_bone_in_frame,
                    sample_body_basis_in_frame,
                    sample_body_bone_position_in_frame,
                )
            visual_centers = {placement.tag: [] for placement in scene.animals}
            runtime_frames = {placement.tag: [] for placement in scene.animals}

            # Per-frame render
            import cv2
            for frame_i in range(n_frames):
                with instance.begin_frame():
                    for actor, placement in zip(actors, scene.animals):
                        if placement.is_animated:
                            _step_animated(actor, placement, frame_i,
                                            "apartment", render_scene)
                    for actor, placement in zip(actors, scene.animals):
                        ground_snap = None
                        if getattr(placement, "ground_snap_to_floor", False):
                            ground_snap = _snap_actor_bounds_bottom_to_floor(
                                actor,
                                maximum_abs_correction_cm=float(
                                    getattr(
                                        placement,
                                        "ground_snap_max_abs_correction_cm",
                                        15.0,
                                    )
                                ),
                            )
                        runtime_state = _actor_runtime_frame_state(
                            actor,
                            placement,
                            frame_i,
                        )
                        if ground_snap is not None:
                            runtime_state["ground_snap"] = ground_snap
                        runtime_frames[placement.tag].append(runtime_state)
                        center = _sanitize_actor_visual_center_ssot_m(
                            runtime_state["bounds_origin_ssot_m"],
                            placement,
                            frame_i,
                        )
                        visual_centers[placement.tag].append(
                            [float(v) for v in center]
                        )
                    cam.K2_SetActorLocationAndRotation(
                        NewLocation={"X": mic_x_cm, "Y": mic_y_cm, "Z": mic_z_cm},
                        NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": float(yaw_ue_deg)},
                        bSweep=False, bTeleport=True,
                    )
                    # Sample bone positions inside the SAME begin_frame — Plan
                    # 1.5.B in-loop rig direction check
                    if rig_assert_on and frame_i in sample_events:
                        for label, which in sample_events[frame_i]:
                            samples = rig_samples_by_window[label]
                            for actor, placement in zip(actors, scene.animals):
                                if not placement.is_animated:
                                    continue
                                bucket = samples.setdefault(placement.tag, {})
                                if which == "a":
                                    bucket["diagnostics"] = []
                                    bucket["body_basis"] = sample_body_basis_in_frame(
                                        actor,
                                        unreal_service=game.unreal_service,
                                        diagnostics=bucket["diagnostics"],
                                    )
                                    bucket["bone"] = find_body_bone_in_frame(
                                        actor,
                                        unreal_service=game.unreal_service,
                                        diagnostics=bucket["diagnostics"],
                                    )
                                bone = bucket.get("bone")
                                if bone is None:
                                    continue
                                bucket[which] = sample_body_bone_position_in_frame(
                                    actor,
                                    bone,
                                    unreal_service=game.unreal_service,
                                    diagnostics=bucket["diagnostics"],
                                )
                with instance.end_frame():
                    img = read_frame(comp)
                    cv2.imwrite(str(frames_dir / f"frame_{frame_i:04d}.png"), img)
                if frame_i % 15 == 0:
                    print(f"[apt_render] frame {frame_i}/{n_frames}")

            # Populate direction windows from the exact UE actor transforms
            # captured on every rendered frame. Same-frame body-bone samples
            # remain responsible for facing; actor roots are responsible for
            # explicit world-trajectory direction.
            for window in rig_windows:
                label = str(window["label"])
                frame_a = int(window["frame_a"])
                frame_b = int(window["frame_b"])
                samples = rig_samples_by_window[label]
                for placement in scene.animals:
                    if not placement.is_animated:
                        continue
                    frames = runtime_frames[placement.tag]
                    bucket = samples.setdefault(placement.tag, {})
                    bucket["root_a"] = frames[frame_a]["root_transform_ue"][
                        "location_cm"
                    ]
                    bucket["root_b"] = frames[frame_b]["root_transform_ue"][
                        "location_cm"
                    ]

            rig_direction_evidence = _build_rig_direction_window_evidence(
                scene,
                rig_samples_by_window,
                windows=rig_windows,
                enabled=rig_assert_on,
            )
            for tag, direction in rig_direction_evidence.items():
                print(f"[apt_render] rig direction {direction['status']} for {tag}")

            visual_meta_path = out_dir / "videos" / "actor_visual_metadata.json"
            visual_metadata = _build_actor_visual_metadata(
                scene=scene,
                visual_centers=visual_centers,
                runtime_frames=runtime_frames,
                human_gate_evidence=human_gate_evidence,
                n_frames=n_frames,
                rig_direction_evidence=rig_direction_evidence,
            )
            visual_meta_path.write_text(json.dumps(visual_metadata, indent=2))
            print(f"[apt_render] wrote {visual_meta_path}")

            # ffmpeg -> mp4
            mp4_path = out_dir / "videos" / "apartment_v1_view0.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", str(frames_dir / "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                str(mp4_path),
            ], check=True)
            print(f"[apt_render] wrote {mp4_path}")

        finally:
            instance.close(force=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(DEFAULT_SPEC))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--clip-id", default="apartment_v1_000")
    ap.add_argument("--rig-assert", action="store_true",
                    help="Enable Plan 1.5.B rig direction assertion per clip "
                         "(also enabled by SPEAR_RIG_ASSERT=1 env var).")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    csv_path = out_dir / "profile_per_clip.csv"
    render_apartment(Path(args.spec), out_dir, csv_path, args.clip_id)


if __name__ == "__main__":
    main()
