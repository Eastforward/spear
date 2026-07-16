#!/usr/bin/env python3
"""Build exact shared arm/leg motion-basis Walking review candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
TOOLS_DIR = SCRIPT_PATH.parent
SPEAR_ROOT = TOOLS_DIR.parent
SPIKE_DIR = TOOLS_DIR / "spike_rlr"
for directory in (SPEAR_ROOT, TOOLS_DIR, SPIKE_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tools import blender_render_tokenrig_human_review as dynamic_review
from tools import blender_retarget_rocketbox_to_tokenrig as runner
from retarget_motion_basis_review import (
    BUNDLE_SCHEMA,
    CANDIDATE_ANGLES,
    VIEWS,
    compute_axial_pose_metrics,
    compute_four_limb_motion_metrics,
    file_record,
    sha256_file,
    validate_review_bundle,
    yaw_matrix,
)


FPS = 30
VIDEO_SIZE = (640, 360)
SAMPLED_ROLES = (
    "pelvis",
    "neck",
    "head",
    "left_clavicle",
    "right_clavicle",
    "left_upper_arm",
    "left_forearm",
    "left_hand",
    "right_upper_arm",
    "right_forearm",
    "right_hand",
    "left_thigh",
    "left_calf",
    "left_foot",
    "right_thigh",
    "right_calf",
    "right_foot",
)
LOCKED_BODY_ROLES = ("pelvis", "neck", "head")


class MotionBasisBuildError(RuntimeError):
    """The exact motion-basis candidate build failed."""


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _matrix_rows(value: Any) -> list[list[float]]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise MotionBasisBuildError("locked animation matrix is not finite 4x4")
    return np.round(array, decimals=9).tolist()


def _sample_candidate(
    bpy: Any,
    armature: Any,
    semantic_bones: Mapping[str, Any],
    *,
    frame_start: int,
    frame_end: int,
) -> tuple[list[dict[str, tuple[float, float, float]]], dict[str, Any]]:
    for role in SAMPLED_ROLES:
        name = semantic_bones.get(role)
        if not isinstance(name, str) or armature.pose.bones.get(name) is None:
            raise MotionBasisBuildError(f"candidate is missing semantic pose bone {role}")
    locked_names = [semantic_bones[role] for role in LOCKED_BODY_ROLES]
    spine = semantic_bones.get("spine")
    if not isinstance(spine, list) or not spine:
        raise MotionBasisBuildError("candidate has no locked semantic spine")
    locked_names.extend(spine)
    frames = []
    locked = []
    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        frames.append(
            {
                role: tuple(
                    armature.matrix_world
                    @ armature.pose.bones[semantic_bones[role]].head
                )
                for role in SAMPLED_ROLES
            }
            | {
                "head_tail": tuple(
                    armature.matrix_world
                    @ armature.pose.bones[semantic_bones["head"]].tail
                )
            }
        )
        locked.append(
            {
                "frame": frame,
                "armature_world": _matrix_rows(armature.matrix_world),
                "body_bones": {
                    name: _matrix_rows(
                        armature.matrix_world @ armature.pose.bones[name].matrix
                    )
                    for name in locked_names
                },
            }
        )
    return frames, {
        "schema": "locked_root_body_trajectory_v1",
        "frame_start": frame_start,
        "frame_end": frame_end,
        "frames": locked,
    }


def _clear_actions(bpy: Any) -> None:
    for action in list(bpy.data.actions):
        action.use_fake_user = False
        bpy.data.actions.remove(action, do_unlink=True)


def _export_walking(
    bpy: Any,
    *,
    armature: Any,
    mesh: Any,
    action: Any,
    output_path: Path,
) -> tuple[int, int]:
    armature.animation_data.action = action
    for other in list(bpy.data.actions):
        if other != action:
            other.use_fake_user = False
            bpy.data.actions.remove(other, do_unlink=True)
    frame_start, frame_end = runner._integer_frame_range(action)
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end
    bpy.context.scene.frame_set(frame_start)
    runner._select_target_only(bpy, armature, mesh)
    result = bpy.ops.export_scene.gltf(
        filepath=str(output_path), **runner.gltf_export_parameters("Walking")
    )
    if "FINISHED" not in result or not output_path.is_file():
        raise MotionBasisBuildError("one-action Walking GLB export failed")
    details = dynamic_review._validate_one_action_glb(output_path, "Walking")
    if frame_start != 1 or frame_end != 33:
        raise MotionBasisBuildError("candidate Walking must contain exact frames 1..33")
    if details["animation_name"] != "Walking":
        raise MotionBasisBuildError("candidate GLB Walking action readback failed")
    return frame_start, frame_end


def _external_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(Path(path).resolve()),
        "sha256": sha256_file(path),
        "size_bytes": Path(path).stat().st_size,
    }


def _build_candidate(
    *,
    bpy: Any,
    asset_id: str,
    angle: int,
    candidate_dir: Path,
    bind_pose_glb: Path,
    baseline_blend: Path,
    static_auth: Mapping[str, Any],
    walk_auth: Mapping[str, Any],
) -> dict[str, Any]:
    result = bpy.ops.wm.open_mainfile(filepath=str(baseline_blend))
    if "FINISHED" not in result:
        raise MotionBasisBuildError("could not reopen the sealed baseline blend")
    runner._configure_scene(bpy)
    source = runner._identify_walk_source(bpy, walk_auth["source_animation"])
    semantic = static_auth["semantic_mapping"]
    target, mesh = runner._import_tokenrig_runtime(bpy, bind_pose_glb, semantic)
    target_base = runner.capture_target_base_transform(target)
    all_bones = [bone.name for bone in target.data.bones]
    rest_matrices = runner.capture_rest_matrices(target, all_bones)
    pbr_before = dynamic_review.material_graph_hash(mesh)
    cached = runner.cache_source_motion(bpy, source, action_name="Walking")
    cached["source_armature"] = source.armature
    walking, bake = runner.bake_rest_corrected_action(
        bpy=bpy,
        target_armature=target,
        semantic=semantic,
        cached=cached,
        action_name="Walking",
        target_base_transform=target_base,
        limb_motion_basis_3x3=yaw_matrix(angle),
    )
    frames, locked = _sample_candidate(
        bpy,
        target,
        semantic["semantic_bones"],
        frame_start=1,
        frame_end=33,
    )
    limb_metrics = compute_four_limb_motion_metrics(frames, fps=FPS)
    axial_metrics = compute_axial_pose_metrics(frames, fps=FPS)
    if axial_metrics["automatic_checks"] != "passed":
        raise MotionBasisBuildError(
            "candidate exceeds the anatomical axial pose envelope"
        )
    locked_sha256 = _sha256_json(locked)
    runner.remove_source_objects(bpy, source, [walking])
    runner._remove_everything_except(bpy, [target, mesh], [walking])
    runner.validate_target_only_scene(bpy, target, mesh, all_bones)
    glb_path = candidate_dir / "walking.glb"
    frame_start, frame_end = _export_walking(
        bpy,
        armature=target,
        mesh=mesh,
        action=walking,
        output_path=glb_path,
    )
    pbr_after = dynamic_review.material_graph_hash(mesh)
    if pbr_after != pbr_before:
        raise MotionBasisBuildError("candidate bake/export changed the Pixal PBR graph")
    metrics = {
        **limb_metrics,
        "asset_id": asset_id,
        "candidate_id": next(
            key for key, value in CANDIDATE_ANGLES.items() if value == angle
        ),
        "yaw_degrees": angle,
        "matrix_3x3": yaw_matrix(angle).tolist(),
        "locked_root_body_sha256": locked_sha256,
        "target_rest_matrices_sha256": _sha256_json(rest_matrices),
        "pbr_material_graph_sha256": pbr_before,
        "shared_limb_motion_basis": bake["shared_limb_motion_basis"],
        "anatomical_axial_transfer": bake["anatomical_axial_transfer"],
        "anatomical_axial_pose_gate": axial_metrics,
        "primary_leg_ik_summary": {
            key: bake["primary_leg_ik"][key]
            for key in (
                "method",
                "maximum_endpoint_delta_m",
                "maximum_endpoint_delta_body_height_ratio",
                "maximum_endpoint_delta_leg_length_ratio",
                "minimum_reach_margin_m",
            )
        },
    }
    metrics_path = candidate_dir / "metrics.json"
    _write_exclusive(metrics_path, _json_bytes(metrics))

    _clear_actions(bpy)
    # Some sealed baseline .blend files intentionally have no World assigned.
    # The shared review renderer configures the existing World, so create the
    # neutral container here before it imports and renders the candidate GLB.
    if bpy.context.scene.world is None:
        bpy.context.scene.world = bpy.data.worlds.new("MotionBasisReviewWorld")
    dynamic_review.VIDEO_SIZE = VIDEO_SIZE
    dynamic_review.VIEWS = VIEWS
    motion, media_qa = dynamic_review._render_motion(
        bpy=bpy,
        glb_path=glb_path,
        motion="walking",
        staging=candidate_dir,
        semantic_bones=semantic["semantic_bones"],
    )
    artifacts = {
        "walking.glb": file_record(glb_path, relative_to=candidate_dir.parent),
        "metrics.json": file_record(metrics_path, relative_to=candidate_dir.parent),
    }
    for view in VIEWS:
        for suffix in ("png", "mp4"):
            name = f"walking_{view}.{suffix}"
            artifacts[name] = file_record(
                candidate_dir / name, relative_to=candidate_dir.parent
            )
    return {
        "yaw_degrees": angle,
        "matrix_3x3": yaw_matrix(angle).tolist(),
        "locked_root_body_sha256": locked_sha256,
        "target_rest_matrices_sha256": _sha256_json(rest_matrices),
        "pbr_material_graph_sha256": pbr_before,
        "metrics": metrics,
        "motion": motion,
        "media_qa": media_qa,
        "artifacts": artifacts,
        "automatic_checks": "candidate_generated_and_read_back",
    }


def run(
    *,
    asset_id: str,
    bind_pose_glb: Path,
    static_qa_json: Path,
    baseline_retarget_blend: Path,
    baseline_retarget_manifest: Path,
    output_dir: Path,
    command: Sequence[str],
) -> Path:
    try:
        import bpy
    except ImportError as error:
        raise MotionBasisBuildError("builder must run inside Blender") from error
    if tuple(bpy.app.version) != (4, 2, 1):
        raise MotionBasisBuildError("builder requires exact Blender 4.2.1")
    destination = Path(os.path.abspath(os.fspath(output_dir)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise MotionBasisBuildError(f"no-replace output already exists: {destination}")
    bind_pose_glb = Path(bind_pose_glb).resolve()
    static_qa_json = Path(static_qa_json).resolve()
    baseline_retarget_blend = Path(baseline_retarget_blend).resolve()
    baseline_retarget_manifest = Path(baseline_retarget_manifest).resolve()
    static_auth = runner.authenticate_static_gate(
        asset_id=asset_id,
        bind_pose_glb=bind_pose_glb,
        static_qa_json=static_qa_json,
    )
    walk_auth = runner.authenticate_sealed_walk(
        base_avatar_id=asset_id,
        baseline_retarget_blend=baseline_retarget_blend,
        baseline_retarget_manifest=baseline_retarget_manifest,
    )
    execution_before = {
        "builder": _external_record(SCRIPT_PATH),
        "retarget_runner": _external_record(Path(runner.__file__)),
        "review_renderer": _external_record(Path(dynamic_review.__file__)),
    }
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.", suffix=".staging", dir=destination.parent
        )
    )
    try:
        candidates = {}
        for candidate_id, angle in CANDIDATE_ANGLES.items():
            candidate_dir = staging / candidate_id
            candidate_dir.mkdir()
            candidates[candidate_id] = _build_candidate(
                bpy=bpy,
                asset_id=asset_id,
                angle=angle,
                candidate_dir=candidate_dir,
                bind_pose_glb=bind_pose_glb,
                baseline_blend=baseline_retarget_blend,
                static_auth=static_auth,
                walk_auth=walk_auth,
            )
        locked = {value["locked_root_body_sha256"] for value in candidates.values()}
        rests = {value["target_rest_matrices_sha256"] for value in candidates.values()}
        pbr = {value["pbr_material_graph_sha256"] for value in candidates.values()}
        if len(locked) != 1 or len(rests) != 1 or len(pbr) != 1:
            raise MotionBasisBuildError(
                "basis candidates changed locked root/body, rest matrices, or PBR"
            )
        current_static = runner.authenticate_static_gate(
            asset_id=asset_id,
            bind_pose_glb=bind_pose_glb,
            static_qa_json=static_qa_json,
        )
        current_walk = runner.authenticate_sealed_walk(
            base_avatar_id=asset_id,
            baseline_retarget_blend=baseline_retarget_blend,
            baseline_retarget_manifest=baseline_retarget_manifest,
        )
        execution_after = {
            "builder": _external_record(SCRIPT_PATH),
            "retarget_runner": _external_record(Path(runner.__file__)),
            "review_renderer": _external_record(Path(dynamic_review.__file__)),
        }
        if current_static != static_auth or current_walk != walk_auth:
            raise MotionBasisBuildError("authenticated source changed during candidate build")
        if execution_after != execution_before:
            raise MotionBasisBuildError("builder/runner/renderer changed during execution")
        manifest = {
            "schema": BUNDLE_SCHEMA,
            "asset_id": asset_id,
            "classification": "technical_diagnostic_only",
            "decision": "pending_human_basis_selection",
            "formal_dataset_asset": False,
            "canonical_front": "negative-y",
            "canonical_up": "positive-z",
            "fps": FPS,
            "frame_start": 1,
            "frame_end": 33,
            "candidate_order": list(CANDIDATE_ANGLES),
            "candidates": candidates,
            "authenticated_inputs": {
                "static": static_auth,
                "sealed_walk": walk_auth,
            },
            "locked_invariants": {
                "root_body_sha256": next(iter(locked)),
                "target_rest_matrices_sha256": next(iter(rests)),
                "pbr_material_graph_sha256": next(iter(pbr)),
            },
            "execution": {
                **execution_before,
                "blender_version": list(bpy.app.version),
                "video_size": list(VIDEO_SIZE),
                "command": list(command),
            },
            "automatic_checks": "all_candidates_generated_and_hash_locked",
        }
        manifest_path = staging / "motion_basis_review_manifest.json"
        _write_exclusive(manifest_path, _json_bytes(manifest))
        validate_review_bundle(staging)
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_file():
                path.chmod(0o444)
            elif path.is_dir():
                path.chmod(0o555)
        staging.chmod(0o555)
        runner.rename_directory_noreplace(staging, destination)
        validate_review_bundle(destination)
        return destination / "motion_basis_review_manifest.json"
    except BaseException as error:
        if staging.exists():
            for path in staging.rglob("*"):
                try:
                    path.chmod(0o755 if path.is_dir() else 0o644)
                except OSError:
                    pass
            failure = staging / "failure.json"
            if not failure.exists():
                failure.write_text(
                    json.dumps(
                        {
                            "schema": "shared_limb_motion_basis_build_failure_v1",
                            "asset_id": asset_id,
                            "error_type": type(error).__name__,
                            "error": str(error),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
            failed = destination.with_name(
                f"{destination.name}.failed.{uuid.uuid4().hex}"
            )
            runner.rename_directory_noreplace(staging, failed)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--bind-pose-glb", type=Path, required=True)
    parser.add_argument("--static-qa-json", type=Path, required=True)
    parser.add_argument("--baseline-retarget-blend", type=Path, required=True)
    parser.add_argument("--baseline-retarget-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run(
        asset_id=args.asset_id,
        bind_pose_glb=args.bind_pose_glb,
        static_qa_json=args.static_qa_json,
        baseline_retarget_blend=args.baseline_retarget_blend,
        baseline_retarget_manifest=args.baseline_retarget_manifest,
        output_dir=args.output_dir,
        command=sys.argv,
    )
    print(f"RETARGET_MOTION_BASIS_REVIEW_OK {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
