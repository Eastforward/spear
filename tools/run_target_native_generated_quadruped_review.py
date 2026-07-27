#!/usr/bin/env python3
"""Run the reviewed post-TokenRig quadruped normalization and QA sequence.

This runner deliberately starts from a target-native, unanimated rig.  FLUX,
Pixel3D and topology/PBR repair keep their own review gates and artifacts.  It
then runs, in order: rigid heading normalization, rig audit, four-foot support
plane leveling, semantic motion retarget, rotation-invariant Walk/Idle
deformation audit, conditional deterministic weight repair with mandatory
post-repair gait/deformation audits, six-view rendering, encoding and media
readback.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


SPEAR_ROOT = Path(__file__).resolve().parents[1]
TOOLS = SPEAR_ROOT / "tools"
SCHEMA = "avengine_target_native_generated_quadruped_review_run_v3"
HEADING_SCHEMA = "avengine_generated_animal_heading_normalization_v1"
RIG_AUDIT_SCHEMA = "avengine_generated_animal_rig_audit_v1"
SUPPORT_PLANE_SCHEMA = "avengine_generated_animal_support_plane_leveling_v1"
RETARGET_SCHEMA = "avengine_generated_quadruped_retarget_v5"
GAIT_AUDIT_SCHEMA = "avengine_generated_animal_gait_direction_run_v1"
GAIT_RESULT_SCHEMA = "avengine_quadruped_gait_direction_audit_v1"
DEFORMATION_AUDIT_SCHEMA = "avengine_skinned_deformation_audit_v1"
DEFORMATION_THRESHOLDS = {
    "review_edge_extension_ratio": 0.07,
    "reject_edge_extension_ratio": 0.08,
    "review_edge_stretch_ratio": 2.0,
    "reject_edge_stretch_ratio": 4.0,
    "review_area_stretch_ratio": 3.0,
    "reject_area_stretch_ratio": 8.0,
}
WEIGHT_REPAIR_PARAMETERS = {
    "walking_samples": 21,
    "idle_samples": 9,
    "maximum_passes": 6,
    "inner_iterations": 4,
    "extension_threshold": 0.02,
    "minimum_stretch_ratio": 1.8,
    "maximum_rest_edge_ratio": 0.04,
    "blend": 0.9,
    "component_rings": 4,
    "repair_mode": "component-parent-lock",
    "top_k": 4,
    "minimum_weight": 1.0e-5,
    "maximum_seed_edges": 4096,
    "skip_cross_limb_preclean": False,
    "cross_limb_authority": "largest-bone",
    "limb_slice_height_fraction": 0.4,
}
WEIGHT_REPAIR_SCHEMA = "avengine_motion_aware_quadruped_weight_repair_v2"
WEIGHT_REPAIR_READY_STATUS = (
    "research_candidate_pending_readback_and_visual_qa"
)
REPAIR_AUTHORITY_FIELDS = (
    "native_mesh_geometry_preserved",
    "native_mesh_topology_preserved",
    "pbr_material_preserved",
    "fitted_skeleton_rest_matrices_preserved",
    "approved_animation_curves_preserved",
    "only_vertex_weights_modified_in_memory",
)

if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools.generated_animal_forward_contract import (  # noqa: E402
    assert_declared_motion_basis,
    assert_declared_motion_donor_artifact,
    load_forward_declaration,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-rig-glb", type=Path, required=True)
    parser.add_argument(
        "--forward-declaration",
        type=Path,
        help=(
            "Authenticated single-point forward declaration.  Supplies the "
            "reviewed source front yaw, the canonical target axis and the "
            "constant motion basis of the declared donor; the legacy "
            "per-asset heading/motion-basis flags are then forbidden."
        ),
    )
    parser.add_argument("--heading-review-evidence", type=Path)
    parser.add_argument("--reviewed-source-front-yaw-deg", type=float)
    parser.add_argument(
        "--target-front-axis",
        choices=("positive-x", "negative-x", "positive-y", "negative-y"),
        default="positive-x",
    )
    parser.add_argument("--source-motion-glb", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--blender", default=os.environ.get("AVENGINE_BLENDER", "blender"))
    parser.add_argument("--motion-amplitude", type=float, default=1.0)
    parser.add_argument(
        "--motion-basis-yaw-deg", type=int, choices=(-90, 0, 90, 180)
    )
    parser.add_argument("--side-chain-mode", choices=("matched", "swapped"))
    parser.add_argument(
        "--support-plane-source",
        choices=("bone-endpoints", "mesh-foot-bottoms"),
        default="mesh-foot-bottoms",
        help=(
            "Support-plane authority. Generated TokenRig bone tails can extend "
            "below the visible feet, so the hardened route defaults to the "
            "visible mesh-foot contact bands."
        ),
    )
    parser.add_argument(
        "--weight-repair-policy",
        choices=("auto", "always", "never"),
        default="auto",
        help=(
            "auto runs the deterministic gentle weight repair only when the "
            "initial deformation audit is not passed; always is useful for "
            "controlled revalidation, while never fails closed instead of "
            "rendering a non-passing asset."
        ),
    )
    parser.add_argument("--deformation-samples", type=int, default=24)
    parser.add_argument("--review-frames", type=int, default=8)
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help=(
            "Stop after retarget, gait-direction audit and two cheap Walking "
            "renders so direction mistakes surface in minutes instead of "
            "after the full deformation audit and six-view render."
        ),
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args(argv)


def regular_file(path: Path, label: str) -> Path:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or unsafe {label}: {path}")
    return path


def executable(value: str) -> str:
    candidate = Path(value)
    if candidate.parent != Path("."):
        candidate = candidate.resolve()
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or not os.access(candidate, os.X_OK)
        ):
            raise ValueError(f"missing or unsafe Blender executable: {candidate}")
        return str(candidate)
    resolved = shutil.which(value)
    if resolved is None:
        raise ValueError(f"Blender executable not found on PATH: {value}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    path = regular_file(path, "output artifact")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def output_paths(root: Path) -> dict[str, Path]:
    heading = root / "01_heading"
    rig = root / "02_rig_audit"
    level = root / "03_support_plane"
    motion = root / "04_motion"
    review = root / "05_review"
    return {
        "heading_glb": heading / "target_heading.glb",
        "heading_manifest": heading / "manifest.json",
        "rig_audit": rig / "generated_rig_audit.json",
        "leveled_glb": level / "target_heading_leveled.glb",
        "level_manifest": level / "manifest.json",
        "animated_glb": motion / "target_animated.glb",
        "retarget_manifest": motion / "retarget_manifest.json",
        "gait_audit": motion / "gait_direction_audit.json",
        "deformation_audit": motion / "skinned_deformation_walk_idle.json",
        "repaired_glb": motion / "target_animated_repaired.glb",
        "weight_repair_manifest": motion / "weight_repair_manifest.json",
        "repaired_gait_audit": motion / "gait_direction_audit_repaired.json",
        "repaired_deformation_audit": (
            motion / "skinned_deformation_walk_idle_repaired.json"
        ),
        "review_root": review,
        "result": root / "review_run.json",
        "preview_result": root / "preview_run.json",
    }


def blender_command(blender: str, script: str, arguments: list[str]) -> list[str]:
    # Blender otherwise reports some uncaught Python-script exceptions with a
    # successful process return code and lets the outer pipeline continue with
    # missing artifacts.  New-asset stages must fail closed at the first error.
    return [
        blender,
        "-b",
        "--python-exit-code",
        "2",
        "--python",
        str(TOOLS / script),
        "--",
        *arguments,
    ]


def build_initial_commands(
    args, paths: dict[str, Path], blender: str
) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    commands.append(
        (
            "heading",
            blender_command(
                blender,
                "blender_normalize_generated_animal_heading.py",
                [
                    "--input", str(args.target_rig_glb),
                    "--output", str(paths["heading_glb"]),
                    "--manifest", str(paths["heading_manifest"]),
                    "--reviewed-source-front-yaw-deg",
                    str(args.reviewed_source_front_yaw_deg),
                    "--target-front-axis", args.target_front_axis,
                    "--review-evidence", str(args.heading_review_evidence),
                ],
            ),
        )
    )
    commands.append(
        (
            "rig_audit",
            blender_command(
                blender,
                "blender_audit_generated_animal_rig.py",
                [
                    "--input", str(paths["heading_glb"]),
                    "--output", str(paths["rig_audit"]),
                    "--front-axis", args.target_front_axis,
                ],
            ),
        )
    )
    commands.append(
        (
            "support_plane",
            blender_command(
                blender,
                "blender_level_generated_animal_support_plane.py",
                [
                    "--input", str(paths["heading_glb"]),
                    "--output", str(paths["leveled_glb"]),
                    "--manifest", str(paths["level_manifest"]),
                    "--front-axis", args.target_front_axis,
                    "--review-evidence", str(paths["rig_audit"]),
                    "--plane-source", args.support_plane_source,
                    "--maximum-tilt-deg", "30",
                    "--maximum-foot-plane-residual-ratio", "0.02",
                ],
            ),
        )
    )
    commands.append(
        (
            "retarget",
            blender_command(
                blender,
                "blender_retarget_quaternius_to_generated_quadruped.py",
                [
                    "--target-glb", str(paths["leveled_glb"]),
                    "--source-rig-glb", str(args.source_motion_glb),
                    "--output-glb", str(paths["animated_glb"]),
                    "--manifest", str(paths["retarget_manifest"]),
                    "--technical-spike-only",
                    "--target-front-axis", args.target_front_axis,
                    "--motion-amplitude", str(args.motion_amplitude),
                    "--rotation-transfer-mode", "world-left-delta-v2",
                    "--pose-transfer-mode", "world-rotation-retarget-v2",
                    "--motion-basis-yaw-deg", str(args.motion_basis_yaw_deg),
                    "--side-chain-mode", args.side_chain_mode,
                ],
            ),
        )
    )
    commands.append(
        (
            "gait_direction",
            blender_command(
                blender,
                "blender_audit_gait_direction.py",
                [
                    "--input", str(paths["animated_glb"]),
                    "--output", str(paths["gait_audit"]),
                    "--action", "Walking",
                ],
            ),
        )
    )
    commands.append(
        (
            "deformation",
            blender_command(
                blender,
                "blender_audit_skinned_deformation.py",
                [
                    "--input", str(paths["animated_glb"]),
                    "--output", str(paths["deformation_audit"]),
                    "--action", "Walking",
                    "--action", "Idle",
                    "--samples", str(args.deformation_samples),
                ],
            ),
        )
    )
    return commands


def build_repair_commands(
    args, paths: dict[str, Path], blender: str
) -> list[tuple[str, list[str]]]:
    return [
        (
            "weight_repair",
            blender_command(
                blender,
                "blender_repair_animated_quadruped_weight_stretch.py",
                [
                    "--input", str(paths["animated_glb"]),
                    "--output", str(paths["repaired_glb"]),
                    "--manifest", str(paths["weight_repair_manifest"]),
                    # Retarget export canonicalizes every supported input axis
                    # to +X.  Repair therefore consumes the exported axis, not
                    # the pre-retarget target axis used by legacy mode.
                    "--front-axis", "positive-x",
                    "--repair-mode", "component-parent-lock",
                    "--component-rings", "4",
                    "--extension-threshold", "0.02",
                    "--maximum-passes", "6",
                    "--inner-iterations", "4",
                ],
            ),
        ),
        (
            "gait_direction_repaired",
            blender_command(
                blender,
                "blender_audit_gait_direction.py",
                [
                    "--input", str(paths["repaired_glb"]),
                    "--output", str(paths["repaired_gait_audit"]),
                    "--action", "Walking",
                ],
            ),
        ),
        (
            "deformation_repaired",
            blender_command(
                blender,
                "blender_audit_skinned_deformation.py",
                [
                    "--input", str(paths["repaired_glb"]),
                    "--output", str(paths["repaired_deformation_audit"]),
                    "--action", "Walking",
                    "--action", "Idle",
                    "--samples", str(args.deformation_samples),
                ],
            ),
        ),
    ]


def build_review_commands(
    args,
    paths: dict[str, Path],
    blender: str,
    input_glb: Path,
) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    views = (
        ("walking_side", "Walking", "side", "0"),
        ("walking_front", "Walking", "front", "0"),
        ("walking_rear", "Walking", "front", "180"),
        ("idle_side", "Idle", "side", "0"),
        ("idle_front", "Idle", "front", "0"),
        ("idle_rear", "Idle", "front", "180"),
    )
    for label, action, view, yaw in views:
        frame_dir = paths["review_root"] / f"{label}_frames"
        video = paths["review_root"] / f"{label}.mp4"
        commands.append(
            (
                f"render_{label}",
                blender_command(
                    blender,
                    "blender_render_glb_animation.py",
                    [
                        "--input", str(input_glb),
                        "--action", action,
                        "--output-dir", str(frame_dir),
                        "--n-frames", str(args.review_frames),
                        "--width", "512",
                        "--height", "384",
                        "--samples", "16",
                        "--view", view,
                        "--asset-yaw-deg", yaw,
                        "--trajectory-distance-ratio", "0",
                        "--ground-plane",
                        "--engine", "BLENDER_EEVEE_NEXT",
                    ],
                ),
            )
        )
        commands.append(
            (
                f"encode_{label}",
                [
                    "ffmpeg", "-y", "-loglevel", "error",
                    "-framerate", "8",
                    "-i", str(frame_dir / "frame_%04d.png"),
                    "-c:v", "libx264", "-crf", "18",
                    "-pix_fmt", "yuv420p", str(video),
                ],
            )
        )
    return commands


def build_commands(
    args, paths: dict[str, Path], blender: str
) -> list[tuple[str, list[str]]]:
    """Return the complete nominal plan used by ``--validate-only``.

    Runtime ``auto`` policy may skip the repair branch when the initial
    deformation audit already passes.  The nominal plan still exposes every
    command and exact parameter so it can be reviewed before any Blender work.
    """

    return [
        *build_initial_commands(args, paths, blender),
        *build_repair_commands(args, paths, blender),
        *build_review_commands(args, paths, blender, paths["repaired_glb"]),
    ]


def verify_video(path: Path, expected_frames: int) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration",
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    if (
        stream.get("codec_name") != "h264"
        or int(stream.get("width", 0)) != 512
        or int(stream.get("height", 0)) != 384
        or int(stream.get("nb_frames", 0)) != expected_frames
    ):
        raise RuntimeError(f"review video readback failed: {path}")
    return {
        **file_record(path),
        "codec": "h264",
        "width": 512,
        "height": 384,
        "frame_count": expected_frames,
        "frame_rate": stream.get("r_frame_rate"),
        "duration_seconds": float(payload["format"]["duration"]),
    }


def load_json_artifact(path: Path, label: str) -> dict:
    path = regular_file(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return payload


def require_file_binding(record: object, path: Path, label: str) -> None:
    expected = regular_file(path, label)
    if not isinstance(record, dict):
        raise RuntimeError(f"{label} file binding is missing")
    if (
        record.get("path") != str(expected)
        or record.get("sha256") != sha256_file(expected)
        or record.get("size_bytes") != expected.stat().st_size
    ):
        raise RuntimeError(f"{label} file binding does not match {expected}")


def require_path_hash_binding(record: object, path: Path, label: str) -> None:
    expected = regular_file(path, label)
    if not isinstance(record, dict):
        raise RuntimeError(f"{label} file binding is missing")
    if (
        record.get("path") != str(expected)
        or record.get("sha256") != sha256_file(expected)
    ):
        raise RuntimeError(f"{label} file binding does not match {expected}")


def require_heading_manifest(
    path: Path,
    input_glb: Path,
    review_evidence: Path,
    output_glb: Path,
    *,
    reviewed_source_front_yaw_deg: float,
    target_front_axis: str,
) -> dict:
    payload = load_json_artifact(path, "heading manifest")
    if payload.get("schema") != HEADING_SCHEMA:
        raise RuntimeError("heading schema is missing or unsupported")
    require_file_binding(payload.get("input"), input_glb, "heading input")
    require_path_hash_binding(
        payload.get("review_evidence"), review_evidence, "heading review evidence"
    )
    require_file_binding(payload.get("output"), output_glb, "heading output")
    cardinal_yaws = {
        "positive-x": 0.0,
        "positive-y": 90.0,
        "negative-x": 180.0,
        "negative-y": -90.0,
    }
    target_yaw = cardinal_yaws[target_front_axis]
    heading = payload.get("heading")
    if (
        not isinstance(heading, dict)
        or heading.get("reviewed_source_front_yaw_deg")
        != reviewed_source_front_yaw_deg
        or heading.get("target_front_axis") != target_front_axis
        or heading.get("target_front_yaw_deg") != target_yaw
        or heading.get("applied_world_z_yaw_deg")
        != target_yaw - reviewed_source_front_yaw_deg
        or heading.get("policy")
        != "single_rigid_world_rotation_for_every_scene_root"
    ):
        raise RuntimeError("heading parameters changed during execution")
    preservation = payload.get("preservation_contract")
    if not isinstance(preservation, dict) or any(
        preservation.get(field) is not False
        for field in (
            "mesh_topology_changed",
            "material_changed",
            "skeleton_hierarchy_changed",
            "skin_weights_changed",
            "animation_present_or_changed",
        )
    ):
        raise RuntimeError("heading rigid-preservation contract did not pass")
    if (
        payload.get("scene_before") != payload.get("scene_after")
        or payload.get("status")
        != "technical_spike_only_pending_reaudit_and_animation_qa"
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise RuntimeError("heading scene or authority state changed")
    return payload


def require_retarget_manifest(
    path: Path,
    target_glb: Path,
    source_motion_glb: Path,
    output_glb: Path,
    *,
    target_front_axis: str,
    motion_amplitude: float,
    motion_basis_yaw_deg: int,
    side_chain_mode: str,
) -> dict:
    payload = load_json_artifact(path, "retarget manifest")
    if payload.get("schema") != RETARGET_SCHEMA:
        raise RuntimeError("retarget schema is missing or unsupported")
    require_path_hash_binding(payload.get("target"), target_glb, "retarget target")
    target = payload["target"]
    if (
        target.get("mesh_pbr_skeleton_and_weights_authority") is not True
        or target.get("reviewed_front_axis") != target_front_axis
    ):
        raise RuntimeError("retarget target authority or reviewed axis changed")
    require_path_hash_binding(
        payload.get("source_motion"), source_motion_glb, "retarget motion source"
    )
    source = payload["source_motion"]
    if (
        source.get("geometry_used") is not False
        or source.get("weights_used") is not False
        or source.get("animation_channels_used")
        != ["translation", "rotation", "scale"]
    ):
        raise RuntimeError("retarget motion-source authority contract changed")
    gate = payload.get("motion_basis_gate")
    if not isinstance(gate, dict) or (
        gate.get("mode") != "technical_spike_only_unreviewed"
        or gate.get("human_approved") is not False
        or gate.get("target_animation_generation_authorized") is not False
        or gate.get("formal_dataset_registration_authorized") is not False
        or gate.get("selected_motion_basis_yaw_deg") != motion_basis_yaw_deg
        or gate.get("selected_side_chain_mode") != side_chain_mode
        or gate.get("selected_rotation_solver") != "world-left-delta-v2"
    ):
        raise RuntimeError("retarget motion basis changed during execution")
    pose_transfer = payload.get("runtime_pose_transfer")
    rotation_transfer = payload.get("rotation_transfer")
    if (
        not isinstance(pose_transfer, dict)
        or pose_transfer.get("mode") != "world-rotation-retarget-v2"
        or pose_transfer.get("world_rotation_resampling_used") is not True
        or not isinstance(rotation_transfer, dict)
        or rotation_transfer.get("method") != "world-left-delta-v2"
        or rotation_transfer.get("approved_preview_method")
        != "world-left-delta-v2"
        or rotation_transfer.get("used_for_runtime") is not True
        or rotation_transfer.get("motion_amplitude") != motion_amplitude
        or rotation_transfer.get("motion_basis_yaw_deg")
        != motion_basis_yaw_deg
        or rotation_transfer.get("side_chain_mode") != side_chain_mode
    ):
        raise RuntimeError("retarget solver or amplitude changed during execution")
    semantics = payload.get("semantic_inference")
    if (
        not isinstance(semantics, dict)
        or semantics.get("bone_name_independent_target") is not True
        or semantics.get("complete_target_bone_coverage") is not True
    ):
        raise RuntimeError("retarget semantic coverage contract did not pass")
    export = payload.get("export")
    require_file_binding(export, output_glb, "retarget export")
    if (
        export.get("canonical_front_axis") != "positive-x"
        or export.get("action_names") != ["Walking", "Idle"]
    ):
        raise RuntimeError("retarget export is not the canonical +X Walk/Idle asset")
    if (
        payload.get("status")
        != "technical_spike_only_pending_deformation_and_visual_qa"
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise RuntimeError("retarget output authority state is unsupported")
    return payload


def require_rig_audit(path: Path, input_glb: Path, *, front_axis: str) -> dict:
    payload = load_json_artifact(path, "rig audit")
    if payload.get("schema") != RIG_AUDIT_SCHEMA:
        raise RuntimeError("rig audit schema is missing or unsupported")
    require_file_binding(payload.get("input"), input_glb, "rig audit input")
    coordinate = payload.get("coordinate_contract")
    mesh = payload.get("mesh")
    armature = payload.get("armature")
    skin = payload.get("skin")
    automatic_checks = payload.get("automatic_checks")
    if (
        not isinstance(coordinate, dict)
        or coordinate.get("up_axis") != "positive-z"
        or coordinate.get("reviewed_front_axis") != front_axis
        or coordinate.get("automatic_fine_yaw") is not False
        or not isinstance(mesh, dict)
        or not isinstance(mesh.get("vertices"), int)
        or isinstance(mesh.get("vertices"), bool)
        or mesh.get("vertices") <= 0
        or not isinstance(mesh.get("polygons"), int)
        or isinstance(mesh.get("polygons"), bool)
        or mesh.get("polygons") <= 0
        or not isinstance(mesh.get("materials"), list)
        or not mesh.get("materials")
        or not isinstance(mesh.get("uv_layers"), list)
        or not mesh.get("uv_layers")
        or not isinstance(armature, dict)
        or not isinstance(armature.get("bone_count"), int)
        or isinstance(armature.get("bone_count"), bool)
        or armature.get("bone_count") < 5
        or not isinstance(armature.get("records"), list)
        or len(armature.get("records")) != armature.get("bone_count")
        or not isinstance(armature.get("root_bones"), list)
        or len(armature.get("root_bones")) != 1
        or not isinstance(armature.get("low_leaf_endpoint_candidates"), list)
        or len(armature.get("low_leaf_endpoint_candidates")) != 4
        or not isinstance(skin, dict)
        or skin.get("vertex_count") != mesh.get("vertices")
        or skin.get("vertex_group_count") != armature.get("bone_count")
        or skin.get("maximum_allowed_influences") != 4
        or not isinstance(skin.get("maximum_influences_per_vertex"), int)
        or skin.get("maximum_influences_per_vertex") > 4
        or skin.get("unweighted_vertices") != 0
        or skin.get("non_normalized_vertices") != 0
        or skin.get("hard_failures") != []
        or skin.get("status") != "passed"
        or payload.get("animation_authorized") is not False
        or not isinstance(automatic_checks, dict)
        or automatic_checks.get("hard_failures") != []
        or automatic_checks.get("overall") != "passed"
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise RuntimeError("rig audit rejected the generated animal")
    return payload


def require_support_plane(
    path: Path,
    input_glb: Path,
    output_glb: Path,
    *,
    plane_source: str,
    review_evidence: Path,
    front_axis: str,
) -> dict:
    payload = load_json_artifact(path, "support-plane manifest")
    if payload.get("schema") != SUPPORT_PLANE_SCHEMA:
        raise RuntimeError("support-plane schema is missing or unsupported")
    require_file_binding(
        payload.get("input"), input_glb, "support-plane input"
    )
    require_file_binding(
        payload.get("output"), output_glb, "support-plane output"
    )
    require_path_hash_binding(
        payload.get("review_evidence"),
        review_evidence,
        "support-plane review evidence",
    )
    support = payload.get("support_plane")
    if not isinstance(support, dict):
        raise RuntimeError("support-plane measurements are missing")
    maximum_residual = support.get(
        "maximum_residual_ratio_of_mesh_diagonal"
    )
    tilt_deg = support.get("tilt_deg")
    if (
        support.get("plane_source") != plane_source
        or support.get("front_axis") != front_axis
        or support.get("maximum_reviewed_residual_ratio_of_mesh_diagonal")
        != 0.02
        or support.get("maximum_tilt_deg") != 30.0
        or not isinstance(maximum_residual, (int, float))
        or isinstance(maximum_residual, bool)
        or not math.isfinite(maximum_residual)
        or not 0.0 <= maximum_residual <= 0.02
        or not isinstance(tilt_deg, (int, float))
        or isinstance(tilt_deg, bool)
        or not math.isfinite(tilt_deg)
        or not 0.0 <= tilt_deg <= 30.0
        or support.get("policy")
        != "four_semantic_feet_least_squares_plane_rigid_leveling"
    ):
        raise RuntimeError("support-plane authority changed during execution")
    foot_leaves = support.get("foot_leaves")
    if not isinstance(foot_leaves, list) or len(foot_leaves) != 4:
        raise RuntimeError("support-plane manifest does not bind four semantic feet")
    if plane_source == "mesh-foot-bottoms":
        band_sizes = support.get("mesh_foot_contact_band_sizes")
        if (
            not isinstance(band_sizes, list)
            or len(band_sizes) != 4
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 10
                for value in band_sizes
            )
        ):
            raise RuntimeError(
                "mesh-foot-bottoms did not capture a non-empty contact band "
                "for every semantic foot"
            )
    preservation = payload.get("preservation_contract")
    if not isinstance(preservation, dict) or any(
        preservation.get(field) is not False
        for field in (
            "mesh_topology_changed",
            "material_changed",
            "skeleton_hierarchy_changed",
            "skin_weights_changed",
            "animation_present_or_changed",
        )
    ):
        raise RuntimeError("support-plane rigid-preservation contract did not pass")
    if (
        payload.get("scene_before") != payload.get("scene_after")
        or payload.get("status")
        != "technical_spike_only_pending_retarget_and_visual_qa"
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise RuntimeError("support-plane scene or authority state changed")
    return payload


def require_gait_audit(path: Path, input_glb: Path, label: str) -> dict:
    payload = load_json_artifact(path, label)
    if payload.get("schema") != GAIT_AUDIT_SCHEMA:
        raise RuntimeError(f"{label} schema is missing or unsupported")
    require_file_binding(payload.get("input"), input_glb, f"{label} input")
    action = payload.get("action")
    result = payload.get("result")
    stance_drift_yaw = (
        result.get("stance_drift_yaw_deg") if isinstance(result, dict) else None
    )
    if (
        not isinstance(action, str)
        or "walking" not in action.lower()
        or payload.get("samples") != 12
        or not isinstance(result, dict)
        or result.get("schema") != GAIT_RESULT_SCHEMA
        or result.get("coordinate_frame")
        != "blender_world_z_up_front_positive_x"
        or result.get("classification") != "forward"
        or result.get("walks_head_first") is not True
        or not isinstance(stance_drift_yaw, (int, float))
        or isinstance(stance_drift_yaw, bool)
        or not math.isfinite(stance_drift_yaw)
        or payload.get("status") != "pass"
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise RuntimeError(f"{label} did not pass")
    return payload


def require_deformation_audit(
    path: Path,
    input_glb: Path,
    label: str,
    *,
    expected_samples: int,
    require_pass: bool,
) -> dict:
    payload = load_json_artifact(path, label)
    if payload.get("schema") != DEFORMATION_AUDIT_SCHEMA:
        raise RuntimeError(f"{label} schema is missing or unsupported")
    expected = regular_file(input_glb, f"{label} input")
    if (
        payload.get("input") != str(expected)
        or payload.get("input_sha256") != sha256_file(expected)
        or payload.get("input_size_bytes") != expected.stat().st_size
    ):
        raise RuntimeError(f"{label} input binding does not match {expected}")
    if payload.get("thresholds") != DEFORMATION_THRESHOLDS:
        raise RuntimeError(f"{label} threshold contract changed")
    rest = payload.get("rest_geometry")
    if (
        not isinstance(rest, dict)
        or rest.get("decision_scale") != "centroid_bounding_sphere_diameter"
        or any(
            not isinstance(rest.get(field), int)
            or isinstance(rest.get(field), bool)
            or rest.get(field) <= 0
            for field in ("vertices", "edges", "triangles")
        )
    ):
        raise RuntimeError(f"{label} rest-geometry evidence is incomplete")
    actions = payload.get("actions")
    if not isinstance(actions, list) or len(actions) != 2:
        raise RuntimeError(f"{label} must contain Walking and Idle evidence")
    decisions = []
    for expected_action, action in zip(("Walking", "Idle"), actions):
        if not isinstance(action, dict):
            raise RuntimeError(f"{label} action evidence is malformed")
        resolved_action = action.get("resolved_action")
        sampled_frames = action.get("sampled_frames")
        worst = action.get("worst_case")
        if (
            action.get("requested_action") != expected_action
            or not isinstance(resolved_action, str)
            or expected_action.lower() not in resolved_action.lower()
            or not isinstance(sampled_frames, list)
            or len(sampled_frames) != expected_samples
            or any(
                not isinstance(frame, dict)
                or not isinstance(frame.get("metrics"), dict)
                for frame in sampled_frames
            )
            or not isinstance(worst, dict)
        ):
            raise RuntimeError(
                f"{label} {expected_action} samples or identity are incomplete"
            )
        numeric_fields = (
            "maximum_edge_extension_ratio_of_rest_rotation_invariant_scale",
            "maximum_edge_stretch_ratio",
            "maximum_triangle_area_stretch_ratio",
        )
        values = [worst.get(field) for field in numeric_fields]
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
            for value in values
        ):
            raise RuntimeError(f"{label} {expected_action} worst case is invalid")
        sampled_metric_paths = (
            (
                "edge_extension_ratio_of_rest_rotation_invariant_scale",
                "maximum",
            ),
            ("edge_stretch_ratio", "maximum"),
            ("triangle_area_stretch_ratio", "maximum"),
        )
        recomputed_values = []
        for metric_name, metric_field in sampled_metric_paths:
            observed = []
            for frame in sampled_frames:
                metric = frame["metrics"].get(metric_name)
                value = (
                    metric.get(metric_field) if isinstance(metric, dict) else None
                )
                if (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise RuntimeError(
                        f"{label} {expected_action} sampled metric is invalid"
                    )
                observed.append(value)
            recomputed_values.append(max(observed))
        if any(
            not math.isclose(reported, recomputed, rel_tol=1.0e-12, abs_tol=1.0e-12)
            for reported, recomputed in zip(values, recomputed_values)
        ):
            raise RuntimeError(
                f"{label} {expected_action} worst case contradicts sampled frames"
            )
        extension, edge, area = values
        expected_decision = (
            "reject_visible_skinning_fan_or_membrane"
            if extension > DEFORMATION_THRESHOLDS["reject_edge_extension_ratio"]
            and (
                edge > DEFORMATION_THRESHOLDS["reject_edge_stretch_ratio"]
                or area > DEFORMATION_THRESHOLDS["reject_area_stretch_ratio"]
            )
            else "manual_review_local_deformation"
            if extension > DEFORMATION_THRESHOLDS["review_edge_extension_ratio"]
            and (
                edge > DEFORMATION_THRESHOLDS["review_edge_stretch_ratio"]
                or area > DEFORMATION_THRESHOLDS["review_area_stretch_ratio"]
            )
            else "passed_automatic_deformation_measurements"
        )
        if action.get("decision") != expected_decision:
            raise RuntimeError(
                f"{label} {expected_action} decision contradicts its measurements"
            )
        decisions.append(expected_decision)
    derived_overall = (
        "rejected"
        if any(decision.startswith("reject_") for decision in decisions)
        else "manual_review_required"
        if any(decision.startswith("manual_") for decision in decisions)
        else "passed"
    )
    overall = payload.get("overall")
    if (
        overall != derived_overall
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise RuntimeError(
            f"{label} overall status contradicts its action evidence"
        )
    if require_pass and overall != "passed":
        raise RuntimeError(f"{label} did not pass: overall={overall}")
    return payload


def require_weight_repair(
    path: Path,
    input_glb: Path,
    output_glb: Path,
) -> dict:
    payload = load_json_artifact(path, "weight repair manifest")
    if payload.get("schema") != WEIGHT_REPAIR_SCHEMA:
        raise RuntimeError("weight repair schema is missing or unsupported")
    require_file_binding(payload.get("input"), input_glb, "weight repair input")
    require_file_binding(payload.get("output"), output_glb, "weight repair output")
    if payload.get("status") != WEIGHT_REPAIR_READY_STATUS:
        raise RuntimeError(
            "weight repair did not converge to a readback-ready candidate: "
            f"status={payload.get('status')!r}"
        )
    if (
        payload.get("front_axis") != "positive-x"
        or payload.get("parameters") != WEIGHT_REPAIR_PARAMETERS
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise RuntimeError("weight repair recipe or authority state changed")
    authority = payload.get("authority_contract")
    if not isinstance(authority, dict) or any(
        authority.get(field) is not True for field in REPAIR_AUTHORITY_FIELDS
    ):
        raise RuntimeError("weight repair authority contract did not pass")
    if (
        authority.get("source_animation_fingerprints")
        != authority.get("post_repair_animation_fingerprints")
        or authority.get("source_animation_curve_stats")
        != authority.get("post_repair_animation_curve_stats")
        or authority.get("rest_geometry_topology_fingerprint_before")
        != authority.get("rest_geometry_topology_fingerprint_after")
    ):
        raise RuntimeError("weight repair changed protected animation or topology")
    geometry_delta = authority.get("maximum_rest_geometry_delta")
    allowed_geometry_delta = authority.get("maximum_allowed_rest_geometry_delta")
    final = payload.get("final_measurements")
    final_extension = (
        final.get("maximum_extension_ratio_of_rest_diagonal")
        if isinstance(final, dict)
        else None
    )
    if (
        not isinstance(geometry_delta, (int, float))
        or isinstance(geometry_delta, bool)
        or not math.isfinite(geometry_delta)
        or geometry_delta < 0.0
        or (
            allowed_geometry_delta is not None
            and (
                not isinstance(allowed_geometry_delta, (int, float))
                or isinstance(allowed_geometry_delta, bool)
                or not math.isfinite(allowed_geometry_delta)
                or allowed_geometry_delta < 0.0
                or geometry_delta > allowed_geometry_delta
            )
        )
        or not isinstance(final_extension, (int, float))
        or isinstance(final_extension, bool)
        or not math.isfinite(final_extension)
        or not 0.0 <= final_extension <= 0.02
        or final.get("remaining_seed_edge_count") != 0
    ):
        raise RuntimeError("weight repair convergence evidence did not pass")
    return payload


def weight_repair_required(policy: str, deformation_overall: str) -> bool:
    if policy == "always":
        return True
    if policy == "auto":
        return deformation_overall != "passed"
    if policy == "never":
        if deformation_overall != "passed":
            raise RuntimeError(
                "initial deformation audit did not pass and "
                "--weight-repair-policy=never forbids repair"
            )
        return False
    raise ValueError(f"unsupported weight repair policy: {policy}")


def run_stage(
    label: str,
    command: list[str],
    timings: dict[str, float],
    pipeline_order: list[str],
) -> None:
    started = time.monotonic()
    subprocess.run(command, cwd=SPEAR_ROOT, check=True)
    timings[label] = time.monotonic() - started
    pipeline_order.append(label)


def require_stage_inputs_unchanged(
    label: str,
    target_record: dict,
    target_rig_glb: Path,
    heading_evidence_record: dict,
    heading_review_evidence: Path,
    motion_record: dict,
    source_motion_glb: Path,
) -> None:
    if label == "heading":
        require_file_binding(
            target_record, target_rig_glb, "authenticated target rig"
        )
        require_file_binding(
            heading_evidence_record,
            heading_review_evidence,
            "authenticated heading review evidence",
        )
    elif label == "retarget":
        require_file_binding(
            motion_record, source_motion_glb, "authenticated motion donor"
        )


def main(argv=None):
    args = parse_args(argv)
    args.target_rig_glb = regular_file(args.target_rig_glb, "target rig GLB")
    args.source_motion_glb = regular_file(
        args.source_motion_glb, "source motion GLB"
    )
    target_input_record = file_record(args.target_rig_glb)
    motion_input_record = file_record(args.source_motion_glb)
    if args.forward_declaration is not None:
        if (
            args.heading_review_evidence is not None
            or args.reviewed_source_front_yaw_deg is not None
            or args.motion_basis_yaw_deg is not None
            or args.side_chain_mode is not None
        ):
            raise ValueError(
                "--forward-declaration replaces --heading-review-evidence, "
                "--reviewed-source-front-yaw-deg, --motion-basis-yaw-deg and "
                "--side-chain-mode; the anatomical front is declared exactly "
                "once and the motion basis is the donor constant"
            )
        declaration_path = regular_file(
            args.forward_declaration, "forward declaration"
        )
        heading_input_record = file_record(declaration_path)
        declaration = load_forward_declaration(declaration_path)
        require_file_binding(
            heading_input_record,
            declaration_path,
            "authenticated forward declaration",
        )
        require_file_binding(
            declaration.get("input_glb"),
            args.target_rig_glb,
            "forward declaration target rig",
        )
        assert_declared_motion_donor_artifact(
            declaration["motion_donor_tag"], args.source_motion_glb
        )
        basis = declaration["expected_motion_basis"]
        assert_declared_motion_basis(
            declaration["motion_donor_tag"],
            basis["motion_basis_yaw_deg"],
            basis["side_chain_mode"],
        )
        args.heading_review_evidence = declaration_path
        args.reviewed_source_front_yaw_deg = declaration[
            "reviewed_source_front_yaw_deg"
        ]
        args.target_front_axis = declaration["target_front_axis"]
        args.motion_basis_yaw_deg = basis["motion_basis_yaw_deg"]
        args.side_chain_mode = basis["side_chain_mode"]
        forward_contract = {
            "mode": "forward_declaration_v1",
            "declaration": dict(heading_input_record),
            "motion_donor_tag": declaration["motion_donor_tag"],
            "derived_motion_basis": dict(basis),
        }
    else:
        if (
            args.heading_review_evidence is None
            or args.reviewed_source_front_yaw_deg is None
            or args.motion_basis_yaw_deg is None
            or args.side_chain_mode is None
        ):
            raise ValueError(
                "legacy mode requires --heading-review-evidence, "
                "--reviewed-source-front-yaw-deg, --motion-basis-yaw-deg and "
                "--side-chain-mode together; prefer --forward-declaration"
            )
        args.heading_review_evidence = regular_file(
            args.heading_review_evidence, "heading review evidence"
        )
        heading_input_record = file_record(args.heading_review_evidence)
        forward_contract = {
            "mode": "legacy_free_parameters_deprecated",
            "deprecation": (
                "per-asset motion-basis and heading parameters are a known "
                "orientation-bug source; migrate to --forward-declaration"
            ),
        }
        if not args.validate_only:
            raise ValueError(
                "legacy free-parameter mode is plan-only and cannot execute; "
                "build an authenticated --forward-declaration"
            )
    if not 0.0 < args.motion_amplitude <= 1.0:
        raise ValueError("--motion-amplitude must be in (0, 1]")
    if not 4 <= args.deformation_samples <= 120:
        raise ValueError("--deformation-samples must be in [4, 120]")
    if not 4 <= args.review_frames <= 120:
        raise ValueError("--review-frames must be in [4, 120]")
    root = args.output_root.resolve()
    if root.exists() or root.is_symlink():
        raise ValueError(f"refusing to replace output root: {root}")
    blender = executable(args.blender)
    paths = output_paths(root)
    preview_labels = (
        "heading", "rig_audit", "support_plane", "retarget", "gait_direction",
        "render_walking_side", "encode_walking_side",
        "render_walking_front", "encode_walking_front",
    )
    if args.preview_only:
        commands = [
            *[
                item
                for item in build_initial_commands(args, paths, blender)
                if item[0] in preview_labels
            ],
            *[
                item
                for item in build_review_commands(
                    args, paths, blender, paths["animated_glb"]
                )
                if item[0] in preview_labels
            ],
        ]
    else:
        if args.weight_repair_policy == "never":
            commands = [
                *build_initial_commands(args, paths, blender),
                *build_review_commands(
                    args, paths, blender, paths["animated_glb"]
                ),
            ]
        else:
            commands = build_commands(args, paths, blender)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "weight_repair_policy": args.weight_repair_policy,
                    "conditional_stages": {
                        "weight_repair": (
                            "not_run_preview_only"
                            if args.preview_only
                            else (
                                "run when the initial deformation audit is "
                                "not passed"
                                if args.weight_repair_policy == "auto"
                                else args.weight_repair_policy
                            )
                        )
                    },
                    "commands": commands,
                },
                indent=2,
            )
        )
        return 0

    root.mkdir(parents=True)
    timings: dict[str, float] = {}
    pipeline_order: list[str] = []
    heading = None
    rig = None
    support_plane = None
    retarget = None
    gait = None
    if args.preview_only:
        for label, command in commands:
            require_stage_inputs_unchanged(
                label,
                target_input_record,
                args.target_rig_glb,
                heading_input_record,
                args.heading_review_evidence,
                motion_input_record,
                args.source_motion_glb,
            )
            run_stage(label, command, timings, pipeline_order)
            require_stage_inputs_unchanged(
                label,
                target_input_record,
                args.target_rig_glb,
                heading_input_record,
                args.heading_review_evidence,
                motion_input_record,
                args.source_motion_glb,
            )
            if label == "heading":
                heading = require_heading_manifest(
                    paths["heading_manifest"],
                    args.target_rig_glb,
                    args.heading_review_evidence,
                    paths["heading_glb"],
                    reviewed_source_front_yaw_deg=args.reviewed_source_front_yaw_deg,
                    target_front_axis=args.target_front_axis,
                )
            elif label == "rig_audit":
                rig = require_rig_audit(
                    paths["rig_audit"],
                    paths["heading_glb"],
                    front_axis=args.target_front_axis,
                )
            elif label == "support_plane":
                support_plane = require_support_plane(
                    paths["level_manifest"],
                    paths["heading_glb"],
                    paths["leveled_glb"],
                    plane_source=args.support_plane_source,
                    review_evidence=paths["rig_audit"],
                    front_axis=args.target_front_axis,
                )
            elif label == "retarget":
                retarget = require_retarget_manifest(
                    paths["retarget_manifest"],
                    paths["leveled_glb"],
                    args.source_motion_glb,
                    paths["animated_glb"],
                    target_front_axis=args.target_front_axis,
                    motion_amplitude=args.motion_amplitude,
                    motion_basis_yaw_deg=args.motion_basis_yaw_deg,
                    side_chain_mode=args.side_chain_mode,
                )
            elif label == "gait_direction":
                gait = require_gait_audit(
                    paths["gait_audit"],
                    paths["animated_glb"],
                    "initial gait audit",
                )
        if (
            heading is None
            or rig is None
            or support_plane is None
            or retarget is None
            or gait is None
        ):
            raise RuntimeError("preview omitted a required automatic gate")
        require_file_binding(
            target_input_record, args.target_rig_glb, "final target rig identity"
        )
        require_file_binding(
            motion_input_record,
            args.source_motion_glb,
            "final motion donor identity",
        )
        require_file_binding(
            heading_input_record,
            args.heading_review_evidence,
            "final heading review evidence identity",
        )
        readback_started = time.monotonic()
        media = {
            label: verify_video(
                paths["review_root"] / f"{label}.mp4", args.review_frames
            )
            for label in ("walking_side", "walking_front")
        }
        timings["media_readback"] = time.monotonic() - readback_started
        pipeline_order.append("media_readback")
        result = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "preview_only_pending_full_review",
            "formal_dataset_registration_authorized": False,
            "forward_contract": forward_contract,
            "pipeline_order": pipeline_order,
            "automatic_admission_gates": {
                "heading": heading["heading"]["target_front_axis"],
                "rig": rig["automatic_checks"]["overall"],
                "support_plane": support_plane["support_plane"]["plane_source"],
                "retarget_export_front_axis": retarget["export"][
                    "canonical_front_axis"
                ],
                "gait": gait["status"],
                "full_deformation_review": "not_run_preview_only",
                "all_required_preview_gates_passed": True,
            },
            "inputs": {
                "target_rig_glb": dict(target_input_record),
                "heading_review_evidence": dict(heading_input_record),
                "source_motion_glb": dict(motion_input_record),
            },
            "outputs": {
                "heading_manifest": file_record(paths["heading_manifest"]),
                "rig_audit": file_record(paths["rig_audit"]),
                "support_plane_manifest": file_record(paths["level_manifest"]),
                "animated_glb": file_record(paths["animated_glb"]),
                "retarget_manifest": file_record(paths["retarget_manifest"]),
                "gait_direction_audit": file_record(paths["gait_audit"]),
                "gait_direction_status": gait.get("status"),
                "media": media,
            },
            "timings_seconds": timings,
        }
        with paths["preview_result"].open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(
            "TARGET_NATIVE_QUADRUPED_PREVIEW_OK "
            f"output={paths['preview_result']}"
        )
        return 0

    initial_deformation = None
    for label, command in build_initial_commands(args, paths, blender):
        require_stage_inputs_unchanged(
            label,
            target_input_record,
            args.target_rig_glb,
            heading_input_record,
            args.heading_review_evidence,
            motion_input_record,
            args.source_motion_glb,
        )
        run_stage(label, command, timings, pipeline_order)
        require_stage_inputs_unchanged(
            label,
            target_input_record,
            args.target_rig_glb,
            heading_input_record,
            args.heading_review_evidence,
            motion_input_record,
            args.source_motion_glb,
        )
        if label == "heading":
            heading = require_heading_manifest(
                paths["heading_manifest"],
                args.target_rig_glb,
                args.heading_review_evidence,
                paths["heading_glb"],
                reviewed_source_front_yaw_deg=args.reviewed_source_front_yaw_deg,
                target_front_axis=args.target_front_axis,
            )
        elif label == "rig_audit":
            rig = require_rig_audit(
                paths["rig_audit"],
                paths["heading_glb"],
                front_axis=args.target_front_axis,
            )
        elif label == "support_plane":
            support_plane = require_support_plane(
                paths["level_manifest"],
                paths["heading_glb"],
                paths["leveled_glb"],
                plane_source=args.support_plane_source,
                review_evidence=paths["rig_audit"],
                front_axis=args.target_front_axis,
            )
        elif label == "retarget":
            retarget = require_retarget_manifest(
                paths["retarget_manifest"],
                paths["leveled_glb"],
                args.source_motion_glb,
                paths["animated_glb"],
                target_front_axis=args.target_front_axis,
                motion_amplitude=args.motion_amplitude,
                motion_basis_yaw_deg=args.motion_basis_yaw_deg,
                side_chain_mode=args.side_chain_mode,
            )
        elif label == "gait_direction":
            gait = require_gait_audit(
                paths["gait_audit"],
                paths["animated_glb"],
                "initial gait audit",
            )
        elif label == "deformation":
            initial_deformation = require_deformation_audit(
                paths["deformation_audit"],
                paths["animated_glb"],
                "initial deformation audit",
                expected_samples=args.deformation_samples,
                require_pass=False,
            )
    if (
        heading is None
        or rig is None
        or support_plane is None
        or retarget is None
        or gait is None
        or initial_deformation is None
    ):
        raise RuntimeError("full review omitted a required automatic gate")

    repair_triggered = weight_repair_required(
        args.weight_repair_policy, initial_deformation["overall"]
    )
    repair_manifest = None
    if repair_triggered:
        final_gait = None
        final_deformation = None
        for label, command in build_repair_commands(args, paths, blender):
            run_stage(label, command, timings, pipeline_order)
            if label == "weight_repair":
                repair_manifest = require_weight_repair(
                    paths["weight_repair_manifest"],
                    paths["animated_glb"],
                    paths["repaired_glb"],
                )
            elif label == "gait_direction_repaired":
                final_gait = require_gait_audit(
                    paths["repaired_gait_audit"],
                    paths["repaired_glb"],
                    "repaired gait audit",
                )
            elif label == "deformation_repaired":
                final_deformation = require_deformation_audit(
                    paths["repaired_deformation_audit"],
                    paths["repaired_glb"],
                    "repaired deformation audit",
                    expected_samples=args.deformation_samples,
                    require_pass=True,
                )
        if (
            repair_manifest is None
            or final_gait is None
            or final_deformation is None
        ):
            raise RuntimeError("weight repair omitted a required post-repair gate")
        reviewed_glb = paths["repaired_glb"]
        final_gait_path = paths["repaired_gait_audit"]
        final_deformation_path = paths["repaired_deformation_audit"]
    else:
        final_gait = gait
        final_deformation = require_deformation_audit(
            paths["deformation_audit"],
            paths["animated_glb"],
            "final deformation audit",
            expected_samples=args.deformation_samples,
            require_pass=True,
        )
        reviewed_glb = paths["animated_glb"]
        final_gait_path = paths["gait_audit"]
        final_deformation_path = paths["deformation_audit"]

    for label, command in build_review_commands(
        args, paths, blender, reviewed_glb
    ):
        run_stage(label, command, timings, pipeline_order)

    require_file_binding(
        target_input_record, args.target_rig_glb, "final target rig identity"
    )
    require_file_binding(
        motion_input_record, args.source_motion_glb, "final motion donor identity"
    )
    require_file_binding(
        heading_input_record,
        args.heading_review_evidence,
        "final heading review evidence identity",
    )
    readback_started = time.monotonic()
    media = {
        label: verify_video(paths["review_root"] / f"{label}.mp4", args.review_frames)
        for label in (
            "walking_side", "walking_front", "walking_rear",
            "idle_side", "idle_front", "idle_rear",
        )
    }
    timings["media_readback"] = time.monotonic() - readback_started
    pipeline_order.append("media_readback")
    result = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "research_candidate_pending_human_review",
        "formal_dataset_registration_authorized": False,
        "forward_contract": forward_contract,
        "pipeline_order": pipeline_order,
        "automatic_admission_gates": {
            "heading": heading["heading"]["target_front_axis"],
            "rig": rig["automatic_checks"]["overall"],
            "support_plane": support_plane["support_plane"]["plane_source"],
            "retarget_export_front_axis": retarget["export"][
                "canonical_front_axis"
            ],
            "gait_initial": gait["status"],
            "deformation_initial": initial_deformation["overall"],
            "weight_repair_policy": args.weight_repair_policy,
            "weight_repair_triggered": repair_triggered,
            "weight_repair": (
                repair_manifest["status"] if repair_manifest is not None else "not_needed"
            ),
            "gait_final": final_gait["status"],
            "deformation_final": final_deformation["overall"],
            "all_automatic_gates_passed": True,
        },
        "inputs": {
            "target_rig_glb": dict(target_input_record),
            "heading_review_evidence": dict(heading_input_record),
            "source_motion_glb": dict(motion_input_record),
        },
        "outputs": {
            "heading_manifest": file_record(paths["heading_manifest"]),
            "rig_audit": file_record(paths["rig_audit"]),
            "support_plane_manifest": file_record(paths["level_manifest"]),
            "retargeted_animated_glb": file_record(paths["animated_glb"]),
            "animated_glb": file_record(reviewed_glb),
            "retarget_manifest": file_record(paths["retarget_manifest"]),
            "gait_direction_audit_initial": file_record(paths["gait_audit"]),
            "gait_direction_audit": file_record(final_gait_path),
            "gait_direction_status": final_gait["status"],
            "deformation_audit_initial": file_record(paths["deformation_audit"]),
            "deformation_audit": file_record(final_deformation_path),
            "deformation_overall": final_deformation["overall"],
            **(
                {
                    "weight_repair_manifest": file_record(
                        paths["weight_repair_manifest"]
                    )
                }
                if repair_triggered
                else {}
            ),
            "media": media,
        },
        "timings_seconds": timings,
    }
    with paths["result"].open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"TARGET_NATIVE_QUADRUPED_REVIEW_OK output={paths['result']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"TARGET_NATIVE_QUADRUPED_REVIEW_FAILED {error}", file=sys.stderr)
        raise SystemExit(2)
