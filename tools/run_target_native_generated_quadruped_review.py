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
from fractions import Fraction
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
SCHEMA = "avengine_target_native_generated_quadruped_review_run_v4"
RENDER_MANIFEST_SCHEMA = "avengine_glb_animation_frame_render_v1"
ENCODE_MANIFEST_SCHEMA = "avengine_glb_animation_video_encode_v1"
REVIEW_MEDIA_SPECS = (
    ("walking_side", "Walking", "side", 0.0),
    ("walking_front", "Walking", "front", 0.0),
    ("walking_rear", "Walking", "front", 180.0),
    ("idle_side", "Idle", "side", 0.0),
    ("idle_front", "Idle", "front", 0.0),
    ("idle_rear", "Idle", "front", 180.0),
)
REVIEW_MEDIA_WIDTH = 512
REVIEW_MEDIA_HEIGHT = 384
REVIEW_MEDIA_FPS = 8
REVIEW_RENDER_CONFIG = {
    "engine": "BLENDER_EEVEE_NEXT",
    "samples": 16,
    "image_format": "PNG",
    "film_transparent": False,
    "ground_plane": True,
    "trajectory_distance_ratio": 0.0,
    "orthographic": False,
    "camera_distance_multiplier": 2.0,
    "camera_reference_diagonal": 0.0,
    "quadruped_far_limb_offset_ratio": 0.0,
    "quadruped_far_limb_action_pose_ratio": 0.0,
    "quadruped_pose_action": "Walking",
    "quadruped_pose_samples": 57,
    "quadruped_morphotype_guide_profile": None,
    "pose_template_clay_color": None,
    "pose_template_yaw_deg": 0.0,
    "review_clay_color": None,
    "preserve_volume": False,
}
HEADING_SCHEMA = "avengine_generated_animal_heading_normalization_v1"
RIG_AUDIT_SCHEMA = "avengine_generated_animal_rig_audit_v1"
SUPPORT_PLANE_SCHEMA = "avengine_generated_animal_support_plane_leveling_v2"
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
WEIGHT_REPAIR_FALLBACK_PARAMETERS = {
    **WEIGHT_REPAIR_PARAMETERS,
    "maximum_passes": 12,
    "inner_iterations": 8,
    "repair_mode": "edge-average",
    "cross_limb_authority": "low-slice-components",
    "limb_slice_height_fraction": 0.2,
}
WEIGHT_REPAIR_FALLBACK_RESIDUAL_PARAMETERS = {
    **WEIGHT_REPAIR_FALLBACK_PARAMETERS,
    "skip_cross_limb_preclean": True,
}
WEIGHT_REPAIR_SCHEMA = "avengine_motion_aware_quadruped_weight_repair_v2"
WEIGHT_REPAIR_READY_STATUS = (
    "research_candidate_pending_readback_and_visual_qa"
)
WEIGHT_REPAIR_INCOMPLETE_STATUS = "research_candidate_repair_incomplete"
WEIGHT_REPAIR_STAGE_CONTRACTS = {
    "primary": {
        "pipeline_stage": "weight_repair",
        "strategy": "component_parent_lock",
        "input_glb_output_descriptor": "retargeted_animated_glb",
        "output_glb_output_descriptor": "weight_repair_primary_glb",
        "manifest_output_descriptor": "weight_repair_primary_manifest",
    },
    "fallback_a": {
        "pipeline_stage": "weight_repair_low_slice_edge_average",
        "strategy": "low_slice_edge_average",
        "input_glb_output_descriptor": "retargeted_animated_glb",
        "output_glb_output_descriptor": "weight_repair_fallback_a_glb",
        "manifest_output_descriptor": "weight_repair_fallback_a_manifest",
    },
    "fallback_b": {
        "pipeline_stage": "weight_repair_low_slice_edge_average_residual",
        "strategy": "low_slice_edge_average_residual",
        "input_glb_output_descriptor": "weight_repair_fallback_a_glb",
        "output_glb_output_descriptor": "weight_repair_fallback_b_glb",
        "manifest_output_descriptor": "weight_repair_fallback_b_manifest",
    },
}
WEIGHT_REPAIR_BRANCH_STAGES = {
    "not_needed": (),
    "primary": ("primary",),
    "fallback_a": ("primary", "fallback_a"),
    "fallback_a_b": ("primary", "fallback_a", "fallback_b"),
}
WEIGHT_REPAIR_BRANCH_PIPELINE_STAGES = {
    "not_needed": (),
    "primary": (
        "weight_repair",
        "gait_direction_repaired",
        "deformation_repaired",
    ),
    "fallback_a": (
        "weight_repair",
        "weight_repair_low_slice_edge_average",
        "gait_direction_repaired_low_slice_edge_average",
        "deformation_repaired_low_slice_edge_average",
    ),
    "fallback_a_b": (
        "weight_repair",
        "weight_repair_low_slice_edge_average",
        "weight_repair_low_slice_edge_average_residual",
        "gait_direction_repaired_low_slice_edge_average_residual",
        "deformation_repaired_low_slice_edge_average_residual",
    ),
}
WEIGHT_REPAIR_BRANCH_FINAL_STAGE = {
    "not_needed": None,
    "primary": "primary",
    "fallback_a": "fallback_a",
    "fallback_a_b": "fallback_b",
}
WEIGHT_REPAIR_BRANCH_STRATEGY = {
    "not_needed": "not_needed",
    "primary": "component_parent_lock",
    "fallback_a": "low_slice_edge_average",
    "fallback_a_b": "low_slice_edge_average_residual",
}
WEIGHT_REPAIR_ATTEMPT_GATE_FIELDS = (
    "stage",
    "pipeline_stage",
    "strategy",
    "status",
    "input_glb_output_descriptor",
    "output_glb_output_descriptor",
    "manifest_output_descriptor",
    "maximum_extension_ratio_of_rest_diagonal",
    "remaining_seed_edge_count",
)
REPAIR_AUTHORITY_FIELDS = (
    "native_mesh_geometry_preserved",
    "native_mesh_topology_preserved",
    "pbr_material_preserved",
    "fitted_skeleton_rest_matrices_preserved",
    "approved_animation_curves_preserved",
    "only_vertex_weights_modified_in_memory",
)


class WeightRepairIncomplete(RuntimeError):
    """A well-formed repair attempt that did not reach the strict 0.02 gate."""

    def __init__(self, message: str, payload: dict):
        super().__init__(message)
        self.payload = payload

if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools.controlled_source_asset_schema import (  # noqa: E402
    StrictJSONError,
    strict_json_loads,
)
from tools.generated_animal_forward_contract import (  # noqa: E402
    assert_declared_motion_basis,
    assert_declared_motion_donor_artifact,
    expected_motion_donor_contract,
    load_forward_declaration,
)
from tools.generated_animal_support_plane_contract import (  # noqa: E402
    POLICY as SUPPORT_PLANE_POLICY,
    SupportPlaneContractError,
    validate_output_glb_readback,
    validate_rigid_leveling_transform,
)
from tools.generated_animal_tokenrig_closure import (  # noqa: E402
    validate_tokenrig_closure_manifest,
)
from tools.generated_quadruped_semantics import (  # noqa: E402
    SemanticRigError,
    infer_quadruped_semantics,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-rig-glb", type=Path, required=True)
    parser.add_argument(
        "--tokenrig-closure-manifest",
        type=Path,
        required=True,
        help=(
            "Authenticated source/geometry-to-TokenRig closure for the exact "
            "--target-rig-glb."
        ),
    )
    parser.add_argument(
        "--expected-tokenrig-closure-sha256",
        required=True,
        help=(
            "External SHA-256 authority for the TokenRig closure manifest "
            "file bytes."
        ),
    )
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
        choices=("mesh-foot-bottoms",),
        default="mesh-foot-bottoms",
        help=(
            "V2 visible-foot authority: mutually exclusive nearest complete "
            "leaf-segment corridors plus a mandatory distal-two-bone weight "
            "crosscheck. No endpoint or weight fallback is available."
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
        "level_readback": level / "output_glb_readback.json",
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
        "fallback_repaired_glb": (
            motion / "target_animated_repaired_low_slice_edge_average.glb"
        ),
        "fallback_weight_repair_manifest": (
            motion / "weight_repair_low_slice_edge_average_manifest.json"
        ),
        "fallback_gait_audit": (
            motion / "gait_direction_audit_low_slice_edge_average.json"
        ),
        "fallback_deformation_audit": (
            motion / "skinned_deformation_walk_idle_low_slice_edge_average.json"
        ),
        "fallback_residual_repaired_glb": (
            motion
            / "target_animated_repaired_low_slice_edge_average_residual.glb"
        ),
        "fallback_residual_weight_repair_manifest": (
            motion
            / "weight_repair_low_slice_edge_average_residual_manifest.json"
        ),
        "fallback_residual_gait_audit": (
            motion / "gait_direction_audit_low_slice_edge_average_residual.json"
        ),
        "fallback_residual_deformation_audit": (
            motion
            / "skinned_deformation_walk_idle_low_slice_edge_average_residual.json"
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
            "support_plane_readback",
            blender_command(
                blender,
                "blender_readback_generated_animal_support_plane.py",
                [
                    "--pre-level-input", str(paths["heading_glb"]),
                    "--leveled-output", str(paths["leveled_glb"]),
                    "--support-manifest", str(paths["level_manifest"]),
                    "--rig-audit", str(paths["rig_audit"]),
                    "--front-axis", args.target_front_axis,
                    "--readback-output", str(paths["level_readback"]),
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


def _low_slice_edge_average_arguments(
    *,
    input_glb: Path,
    output_glb: Path,
    manifest: Path,
    residual: bool,
) -> list[str]:
    parameters = (
        WEIGHT_REPAIR_FALLBACK_RESIDUAL_PARAMETERS
        if residual
        else WEIGHT_REPAIR_FALLBACK_PARAMETERS
    )
    arguments = [
        "--input",
        str(input_glb),
        "--output",
        str(output_glb),
        "--manifest",
        str(manifest),
        "--front-axis",
        "positive-x",
        "--walking-samples",
        str(parameters["walking_samples"]),
        "--idle-samples",
        str(parameters["idle_samples"]),
        "--maximum-passes",
        str(parameters["maximum_passes"]),
        "--inner-iterations",
        str(parameters["inner_iterations"]),
        "--extension-threshold",
        str(parameters["extension_threshold"]),
        "--minimum-stretch-ratio",
        str(parameters["minimum_stretch_ratio"]),
        "--maximum-rest-edge-ratio",
        str(parameters["maximum_rest_edge_ratio"]),
        "--blend",
        str(parameters["blend"]),
        "--component-rings",
        str(parameters["component_rings"]),
        "--repair-mode",
        str(parameters["repair_mode"]),
        "--top-k",
        str(parameters["top_k"]),
        "--minimum-weight",
        str(parameters["minimum_weight"]),
        "--maximum-seed-edges",
        str(parameters["maximum_seed_edges"]),
        "--cross-limb-authority",
        str(parameters["cross_limb_authority"]),
        "--limb-slice-height-fraction",
        str(parameters["limb_slice_height_fraction"]),
    ]
    if residual:
        arguments.append("--skip-cross-limb-preclean")
    return arguments


def build_fallback_repair_commands(
    args, paths: dict[str, Path], blender: str
) -> list[tuple[str, list[str]]]:
    """Build an evidence-preserving two-stage low-slice fallback.

    The first fallback always starts from the immutable retarget export.  The
    residual continuation consumes only that fallback output and has its own
    GLB and manifest, so neither the default attempt nor its evidence can be
    overwritten.  A ready first fallback proceeds directly to its own
    gait/deformation audits; the residual stage runs only when that first
    fallback is valid but incomplete, and must then reach the strict
    repair-ready status with zero remaining seed edges.
    """

    return [
        (
            "weight_repair_low_slice_edge_average",
            blender_command(
                blender,
                "blender_repair_animated_quadruped_weight_stretch.py",
                _low_slice_edge_average_arguments(
                    input_glb=paths["animated_glb"],
                    output_glb=paths["fallback_repaired_glb"],
                    manifest=paths["fallback_weight_repair_manifest"],
                    residual=False,
                ),
            ),
        ),
        (
            "gait_direction_repaired_low_slice_edge_average",
            blender_command(
                blender,
                "blender_audit_gait_direction.py",
                [
                    "--input",
                    str(paths["fallback_repaired_glb"]),
                    "--output",
                    str(paths["fallback_gait_audit"]),
                    "--action",
                    "Walking",
                ],
            ),
        ),
        (
            "deformation_repaired_low_slice_edge_average",
            blender_command(
                blender,
                "blender_audit_skinned_deformation.py",
                [
                    "--input",
                    str(paths["fallback_repaired_glb"]),
                    "--output",
                    str(paths["fallback_deformation_audit"]),
                    "--action",
                    "Walking",
                    "--action",
                    "Idle",
                    "--samples",
                    str(args.deformation_samples),
                ],
            ),
        ),
        (
            "weight_repair_low_slice_edge_average_residual",
            blender_command(
                blender,
                "blender_repair_animated_quadruped_weight_stretch.py",
                _low_slice_edge_average_arguments(
                    input_glb=paths["fallback_repaired_glb"],
                    output_glb=paths["fallback_residual_repaired_glb"],
                    manifest=paths["fallback_residual_weight_repair_manifest"],
                    residual=True,
                ),
            ),
        ),
        (
            "gait_direction_repaired_low_slice_edge_average_residual",
            blender_command(
                blender,
                "blender_audit_gait_direction.py",
                [
                    "--input",
                    str(paths["fallback_residual_repaired_glb"]),
                    "--output",
                    str(paths["fallback_residual_gait_audit"]),
                    "--action",
                    "Walking",
                ],
            ),
        ),
        (
            "deformation_repaired_low_slice_edge_average_residual",
            blender_command(
                blender,
                "blender_audit_skinned_deformation.py",
                [
                    "--input",
                    str(paths["fallback_residual_repaired_glb"]),
                    "--output",
                    str(paths["fallback_residual_deformation_audit"]),
                    "--action",
                    "Walking",
                    "--action",
                    "Idle",
                    "--samples",
                    str(args.deformation_samples),
                ],
            ),
        ),
    ]


def review_media_spec(label: str) -> tuple[str, str, float]:
    for candidate, action, view, yaw in REVIEW_MEDIA_SPECS:
        if candidate == label:
            return action, view, yaw
    raise ValueError(f"unsupported review media label: {label!r}")


def review_media_paths(paths: dict[str, Path], label: str) -> dict[str, Path]:
    review_root = paths["review_root"]
    return {
        "frame_dir": review_root / f"{label}_frames",
        "render_manifest": review_root / f"{label}_render_manifest.json",
        "video": review_root / f"{label}.mp4",
        "encode_manifest": review_root / f"{label}_encode_manifest.json",
    }


def expected_review_ffmpeg_config(
    frame_dir: Path, n_frames: int
) -> dict:
    return {
        "overwrite": False,
        "loglevel": "error",
        "input_framerate": REVIEW_MEDIA_FPS,
        "start_number": 0,
        "input_pattern": str(
            (frame_dir / "frame_%04d.png").resolve()
        ),
        "frame_count": n_frames,
        "video_codec": "libx264",
        "crf": 18,
        "pixel_format": "yuv420p",
        "movflags": "+faststart",
    }


def build_review_commands(
    args,
    paths: dict[str, Path],
    blender: str,
    input_glb: Path,
) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    for label, action, view, yaw in REVIEW_MEDIA_SPECS:
        media_paths = review_media_paths(paths, label)
        commands.append(
            (
                f"render_{label}",
                blender_command(
                    blender,
                    "blender_render_glb_animation.py",
                    [
                        "--input", str(input_glb),
                        "--action", action,
                        "--output-dir", str(media_paths["frame_dir"]),
                        "--manifest", str(media_paths["render_manifest"]),
                        "--n-frames", str(args.review_frames),
                        "--width", str(REVIEW_MEDIA_WIDTH),
                        "--height", str(REVIEW_MEDIA_HEIGHT),
                        "--fps", str(REVIEW_MEDIA_FPS),
                        "--samples", "16",
                        "--view", view,
                        "--asset-yaw-deg", str(yaw),
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
                    sys.executable,
                    str(TOOLS / "encode_quadruped_review_media.py"),
                    "--input-glb", str(input_glb),
                    "--render-manifest", str(media_paths["render_manifest"]),
                    "--frame-dir", str(media_paths["frame_dir"]),
                    "--label", label,
                    "--action", action,
                    "--view", view,
                    "--asset-yaw-deg", str(yaw),
                    "--n-frames", str(args.review_frames),
                    "--width", str(REVIEW_MEDIA_WIDTH),
                    "--height", str(REVIEW_MEDIA_HEIGHT),
                    "--fps", str(REVIEW_MEDIA_FPS),
                    "--output", str(media_paths["video"]),
                    "--manifest", str(media_paths["encode_manifest"]),
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
        *build_fallback_repair_commands(args, paths, blender),
        *build_review_commands(
            args,
            paths,
            blender,
            paths["fallback_residual_repaired_glb"],
        ),
    ]


def verify_video(
    path: Path,
    expected_frames: int,
    *,
    expected_width: int = REVIEW_MEDIA_WIDTH,
    expected_height: int = REVIEW_MEDIA_HEIGHT,
    expected_fps: int = REVIEW_MEDIA_FPS,
) -> dict:
    path = regular_file(path, "review video")
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            (
                "stream=codec_name,width,height,nb_frames,"
                "r_frame_rate,avg_frame_rate:format=duration"
            ),
            "-of", "json", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        payload = strict_json_loads(result.stdout)
    except StrictJSONError as error:
        raise RuntimeError(f"ffprobe returned invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"ffprobe JSON must be an object: {path}")
    streams = payload.get("streams")
    if (
        not isinstance(streams, list)
        or len(streams) != 1
        or not isinstance(streams[0], dict)
    ):
        raise RuntimeError(f"review video stream coverage failed: {path}")
    stream = streams[0]
    try:
        frame_rate = Fraction(str(stream.get("r_frame_rate", "0/1")))
        average_rate = Fraction(str(stream.get("avg_frame_rate", "0/1")))
        duration = float(payload["format"]["duration"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise RuntimeError(f"review video metadata is invalid: {path}") from error
    if (
        stream.get("codec_name") != "h264"
        or int(stream.get("width", 0)) != expected_width
        or int(stream.get("height", 0)) != expected_height
        or int(stream.get("nb_frames", 0)) != expected_frames
        or frame_rate != expected_fps
        or average_rate != expected_fps
        or not math.isfinite(duration)
        or abs(duration - expected_frames / expected_fps)
        > 1.0 / expected_fps
    ):
        raise RuntimeError(f"review video readback failed: {path}")
    return {
        **file_record(path),
        "codec": "h264",
        "width": expected_width,
        "height": expected_height,
        "frame_count": expected_frames,
        "frame_rate": stream.get("r_frame_rate"),
        "duration_seconds": duration,
    }


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def require_finite_json(value: object, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            require_finite_json(item, label)
    elif isinstance(value, list):
        for item in value:
            require_finite_json(item, label)


def require_render_manifest(
    manifest_path: Path,
    *,
    input_glb: Path,
    frame_dir: Path,
    action: str,
    view: str,
    asset_yaw_deg: float,
    n_frames: int,
) -> dict:
    payload = load_json_artifact(manifest_path, "review render manifest")
    require_finite_json(payload, "review render manifest")
    if set(payload) != {
        "schema",
        "status",
        "formal_dataset_registration_authorized",
        "input_glb",
        "request",
        "render_config",
        "action_frame_range",
        "frames",
    }:
        raise RuntimeError("review render manifest fields changed")
    if (
        payload.get("schema") != RENDER_MANIFEST_SCHEMA
        or payload.get("status") != "frames_rendered"
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise RuntimeError("review render manifest status/schema is invalid")
    require_file_binding(
        payload.get("input_glb"), input_glb, "review render input GLB"
    )
    frame_dir = frame_dir.resolve()
    request = payload.get("request")
    if (
        not isinstance(request, dict)
        or set(request)
        != {
            "action",
            "resolved_action",
            "rest_pose",
            "view",
            "asset_yaw_deg",
            "n_frames",
            "resolution",
            "fps",
            "output_dir",
        }
        or request.get("action") != action
        or not isinstance(request.get("resolved_action"), str)
        or action.lower() not in request["resolved_action"].lower()
        or request.get("rest_pose") is not False
        or request.get("view") != view
        or request.get("asset_yaw_deg") != float(asset_yaw_deg)
        or request.get("n_frames") != n_frames
        or request.get("resolution")
        != {
            "width": REVIEW_MEDIA_WIDTH,
            "height": REVIEW_MEDIA_HEIGHT,
        }
        or request.get("fps") != REVIEW_MEDIA_FPS
        or request.get("output_dir") != str(frame_dir)
    ):
        raise RuntimeError("review render request identity changed")
    if payload.get("render_config") != REVIEW_RENDER_CONFIG:
        raise RuntimeError("review render configuration changed")
    frame_range = payload.get("action_frame_range")
    frames = payload.get("frames")
    if (
        not isinstance(frame_range, list)
        or len(frame_range) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in frame_range
        )
        or frame_range[1] < frame_range[0]
        or not isinstance(frames, list)
        or len(frames) != n_frames
    ):
        raise RuntimeError("review render frame sampling is invalid")
    for index, frame in enumerate(frames):
        fraction = 0.0 if n_frames == 1 else index / (n_frames - 1)
        source_frame = float(
            int(
                round(
                    frame_range[0]
                    + (frame_range[1] - frame_range[0]) * fraction
                )
            )
        )
        expected_frame = frame_dir / f"frame_{index:04d}.png"
        sample_fraction = (
            frame.get("sample_fraction")
            if isinstance(frame, dict)
            else None
        )
        sampled_source_frame = (
            frame.get("source_action_frame")
            if isinstance(frame, dict)
            else None
        )
        if (
            not isinstance(frame, dict)
            or set(frame)
            != {
                "index",
                "sample_fraction",
                "source_action_frame",
                "artifact",
            }
            or frame.get("index") != index
            or isinstance(sample_fraction, bool)
            or not isinstance(sample_fraction, (int, float))
            or isinstance(sampled_source_frame, bool)
            or not isinstance(sampled_source_frame, (int, float))
            or not math.isclose(
                sample_fraction,
                fraction,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not math.isclose(
                sampled_source_frame,
                source_frame,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise RuntimeError(
                f"review render frame sampling changed at index {index}"
            )
        require_file_binding(
            frame.get("artifact"),
            expected_frame,
            f"review render frame {index}",
        )
    return payload


def render_frame_set(payload: dict) -> dict:
    artifacts = [dict(frame["artifact"]) for frame in payload["frames"]]
    return {
        "count": len(artifacts),
        "sha256": canonical_json_sha256(artifacts),
        "artifacts": artifacts,
    }


def require_encode_manifest(
    manifest_path: Path,
    *,
    label: str,
    render_manifest_path: Path,
    render_payload: dict,
    frame_dir: Path,
    video_path: Path,
    video_record: dict,
    action: str,
    view: str,
    asset_yaw_deg: float,
    n_frames: int,
) -> dict:
    payload = load_json_artifact(manifest_path, "review encode manifest")
    require_finite_json(payload, "review encode manifest")
    if set(payload) != {
        "schema",
        "status",
        "formal_dataset_registration_authorized",
        "media_identity",
        "render_manifest",
        "frame_set",
        "ffmpeg",
        "video",
    }:
        raise RuntimeError("review encode manifest fields changed")
    if (
        payload.get("schema") != ENCODE_MANIFEST_SCHEMA
        or payload.get("status") != "video_encoded_and_probed"
        or payload.get("formal_dataset_registration_authorized") is not False
        or payload.get("media_identity")
        != {
            "label": label,
            "action": action,
            "view": view,
            "asset_yaw_deg": float(asset_yaw_deg),
        }
    ):
        raise RuntimeError("review encode manifest identity/status changed")
    require_file_binding(
        payload.get("render_manifest"),
        render_manifest_path,
        "review encode render manifest",
    )
    if payload.get("frame_set") != render_frame_set(render_payload):
        raise RuntimeError("review encode frame-set binding changed")
    if payload.get("ffmpeg") != expected_review_ffmpeg_config(
        frame_dir, n_frames
    ):
        raise RuntimeError("review encode FFmpeg configuration changed")
    require_file_binding(
        payload.get("video"), video_path, "review encoded video"
    )
    if payload.get("video") != video_record:
        raise RuntimeError(
            "review encode video descriptor contradicts FFprobe readback"
        )
    return payload


def require_review_render_stage(
    paths: dict[str, Path],
    label: str,
    input_glb: Path,
    n_frames: int,
) -> dict:
    action, view, yaw = review_media_spec(label)
    media_paths = review_media_paths(paths, label)
    return require_render_manifest(
        media_paths["render_manifest"],
        input_glb=input_glb,
        frame_dir=media_paths["frame_dir"],
        action=action,
        view=view,
        asset_yaw_deg=yaw,
        n_frames=n_frames,
    )


def require_review_encode_stage(
    paths: dict[str, Path],
    label: str,
    input_glb: Path,
    n_frames: int,
) -> tuple[dict, dict]:
    action, view, yaw = review_media_spec(label)
    media_paths = review_media_paths(paths, label)
    render_payload = require_review_render_stage(
        paths, label, input_glb, n_frames
    )
    video_record = verify_video(media_paths["video"], n_frames)
    require_encode_manifest(
        media_paths["encode_manifest"],
        label=label,
        render_manifest_path=media_paths["render_manifest"],
        render_payload=render_payload,
        frame_dir=media_paths["frame_dir"],
        video_path=media_paths["video"],
        video_record=video_record,
        action=action,
        view=view,
        asset_yaw_deg=yaw,
        n_frames=n_frames,
    )
    return video_record, {
        "render_manifest": file_record(media_paths["render_manifest"]),
        "encode_manifest": file_record(media_paths["encode_manifest"]),
    }


def require_review_media_set(
    paths: dict[str, Path],
    labels: tuple[str, ...],
    input_glb: Path,
    n_frames: int,
) -> tuple[dict, dict]:
    media = {}
    lineage = {}
    for label in labels:
        media[label], lineage[label] = require_review_encode_stage(
            paths, label, input_glb, n_frames
        )
    return media, lineage


def load_json_artifact(path: Path, label: str) -> dict:
    path = regular_file(path, label)
    try:
        payload = strict_json_loads(path.read_bytes())
    except (OSError, StrictJSONError) as error:
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
    motion_donor_contract: dict | None = None,
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
    expected_geometry_used = False
    expected_weights_used = False
    expected_animation_channels = ["translation", "rotation", "scale"]
    expected_actions = ["Walking", "Idle"]
    if motion_donor_contract is not None:
        if (
            motion_donor_contract.get("donor_role")
            != "skeleton_semantics_and_animation_only"
            or motion_donor_contract.get("target_mesh_authority")
            != "generated_target_glb"
        ):
            raise RuntimeError("retarget motion-donor authority contract is invalid")
        expected_geometry_used = motion_donor_contract.get("geometry_used")
        expected_weights_used = motion_donor_contract.get("weights_used")
        expected_animation_channels = motion_donor_contract.get(
            "animation_channels_used"
        )
        expected_actions = motion_donor_contract.get("required_actions")
    if (
        source.get("geometry_used") is not expected_geometry_used
        or source.get("weights_used") is not expected_weights_used
        or source.get("animation_channels_used")
        != expected_animation_channels
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
        or export.get("action_names") != expected_actions
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
    readback_path: Path,
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
    if plane_source != "mesh-foot-bottoms":
        raise RuntimeError("support-plane endpoint authority is forbidden")
    dual_authority = support.get("dual_authority")
    try:
        validated = validate_rigid_leveling_transform(
            dual_authority, support
        )
    except SupportPlaneContractError as exc:
        raise RuntimeError(
            f"support-plane dual authority rejected the evidence: {exc}"
        ) from exc
    thresholds = dual_authority["thresholds"]
    primary = dual_authority["primary"]
    primary_plane = primary["plane"]
    review_payload = load_json_artifact(
        review_evidence, "support-plane review evidence"
    )
    reviewed_diagonal = (
        review_payload.get("mesh", {}).get("world_bbox_diagonal")
        if isinstance(review_payload.get("mesh"), dict)
        else None
    )
    if (
        isinstance(reviewed_diagonal, bool)
        or not isinstance(reviewed_diagonal, (int, float))
        or not math.isfinite(reviewed_diagonal)
        or not math.isclose(
            validated["mesh_diagonal"],
            float(reviewed_diagonal),
            rel_tol=1.0e-7,
            abs_tol=1.0e-9,
        )
    ):
        raise RuntimeError(
            "support-plane mesh diagonal changed after the rig audit"
        )
    if (
        support.get("plane_source") != plane_source
        or support.get("front_axis") != front_axis
        or thresholds.get(
            "maximum_plane_residual_ratio_of_mesh_diagonal"
        )
        != 0.02
        or thresholds.get("maximum_tilt_deg") != 30.0
        or support.get("maximum_reviewed_residual_ratio_of_mesh_diagonal")
        != 0.02
        or support.get("maximum_tilt_deg") != 30.0
        or support.get("policy") != SUPPORT_PLANE_POLICY
    ):
        raise RuntimeError("support-plane authority changed during execution")
    foot_leaves = support.get("foot_leaves")
    if (
        not isinstance(foot_leaves, list)
        or len(foot_leaves) != 4
        or len(set(foot_leaves)) != 4
        or any(not isinstance(value, str) or not value for value in foot_leaves)
    ):
        raise RuntimeError("support-plane manifest does not bind four semantic feet")
    reviewed_mesh = review_payload.get("mesh")
    reviewed_armature = review_payload.get("armature")
    try:
        independently_inferred = infer_quadruped_semantics(
            reviewed_armature.get("records")
            if isinstance(reviewed_armature, dict)
            else None,
            bbox_min=(
                reviewed_mesh.get("world_bbox_min")
                if isinstance(reviewed_mesh, dict)
                else None
            ),
            bbox_extent=(
                reviewed_mesh.get("world_bbox_extent")
                if isinstance(reviewed_mesh, dict)
                else None
            ),
            front_axis=front_axis,
        )
    except (SemanticRigError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"support-plane could not independently infer semantic feet: {exc}"
        ) from exc
    expected_foot_leaves = list(independently_inferred.foot_leaves)
    limb_labels = (
        "front_side_negative",
        "front_side_positive",
        "hind_side_negative",
        "hind_side_positive",
    )
    expected_limb_chains = {
        label: list(getattr(independently_inferred, label))
        for label in limb_labels
    }
    chain_by_leaf = {
        chain[-1]: chain
        for chain in expected_limb_chains.values()
        if len(chain) >= 2
    }
    if (
        foot_leaves != expected_foot_leaves
        or set(chain_by_leaf) != set(expected_foot_leaves)
    ):
        raise RuntimeError(
            "support-plane foot leaves do not match independent rig semantics"
        )
    expected_semantic_rig = {
        "foot_leaves": expected_foot_leaves,
        "limb_chains": expected_limb_chains,
        "distal_owner_bones": [
            chain_by_leaf[leaf][-2:] for leaf in expected_foot_leaves
        ],
    }
    alias_fields = {
        "mesh_foot_capture_counts": primary["capture_counts"],
        "mesh_foot_contact_band_sizes": primary["contact_band_sizes"],
        "foot_points_before": primary["foot_points"],
        "z_equals_ax_plus_by_plus_c": primary_plane[
            "z_equals_ax_plus_by_plus_c"
        ],
        "residual_z": primary_plane["residual_z"],
        "maximum_residual": primary_plane["maximum_residual"],
        "maximum_residual_ratio_of_mesh_diagonal": primary_plane[
            "maximum_residual_ratio_of_mesh_diagonal"
        ],
        "normal_before": primary_plane["normal"],
        "tilt_deg": primary_plane["tilt_deg"],
    }
    if any(
        support.get(field) != expected
        for field, expected in alias_fields.items()
    ):
        raise RuntimeError(
            "support-plane primary-authority aliases are inconsistent"
        )
    for field in (
        "foot_points_after",
        "crosscheck_foot_points_after",
    ):
        points = support.get(field)
        if (
            not isinstance(points, list)
            or len(points) != 4
            or any(
                not isinstance(point, list)
                or len(point) != 3
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in point
                )
                for point in points
            )
        ):
            raise RuntimeError(
                f"support-plane transformed evidence is invalid: {field}"
            )
    for field in ("applied_vertical_translation", "minimum_foot_z_after"):
        value = support.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise RuntimeError(
                f"support-plane transformed scalar is invalid: {field}"
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
    scene_keys = {
        "mesh_count",
        "skinned_mesh_count",
        "armature_count",
        "bone_count",
        "material_count",
        "image_count",
        "action_count",
    }
    scene_before = payload.get("scene_before")
    scene_after = payload.get("scene_after")
    if (
        not isinstance(scene_before, dict)
        or not isinstance(scene_after, dict)
        or set(scene_before) != scene_keys
        or set(scene_after) != scene_keys
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for scene in (scene_before, scene_after)
            for value in scene.values()
        )
        or scene_before != scene_after
    ):
        raise RuntimeError(
            "support-plane scene preservation summary is invalid"
        )
    reviewed_armature = review_payload.get("armature")
    reviewed_bone_count = (
        reviewed_armature.get("bone_count")
        if isinstance(reviewed_armature, dict)
        else None
    )
    if (
        isinstance(reviewed_bone_count, bool)
        or not isinstance(reviewed_bone_count, int)
        or reviewed_bone_count <= 0
        or scene_before["mesh_count"] < scene_before["skinned_mesh_count"]
        or scene_before["mesh_count"] <= 0
        or scene_before["skinned_mesh_count"] != 1
        or scene_before["armature_count"] != 1
        or scene_before["bone_count"] != reviewed_bone_count
        or scene_before["material_count"] <= 0
        or scene_before["action_count"] != 0
    ):
        raise RuntimeError(
            "support-plane scene summary does not match the rig audit"
        )
    if (
        payload.get("status")
        != "technical_spike_only_pending_retarget_and_visual_qa"
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise RuntimeError("support-plane scene or authority state changed")
    readback = load_json_artifact(
        readback_path, "support-plane output GLB readback"
    )
    require_file_binding(
        readback.get("pre_level_input"),
        input_glb,
        "support-plane readback pre-level input",
    )
    require_file_binding(
        readback.get("leveled_output"),
        output_glb,
        "support-plane readback leveled output",
    )
    require_path_hash_binding(
        readback.get("support_manifest"),
        path,
        "support-plane readback manifest binding",
    )
    require_path_hash_binding(
        readback.get("rig_audit"),
        review_evidence,
        "support-plane readback rig-audit binding",
    )
    if readback.get("front_axis") != front_axis:
        raise RuntimeError("support-plane readback front axis changed")
    try:
        validate_output_glb_readback(
            readback,
            source_evidence=dual_authority,
            support_plane=support,
            expected_scene=scene_before,
            expected_semantic_rig=expected_semantic_rig,
            expected_vertex_count=reviewed_mesh.get("vertices"),
            expected_polygon_count=reviewed_mesh.get("polygons"),
        )
    except SupportPlaneContractError as exc:
        raise RuntimeError(
            f"support-plane output GLB readback rejected the evidence: {exc}"
        ) from exc
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
    if (
        not isinstance(expected_samples, int)
        or isinstance(expected_samples, bool)
        or expected_samples < 2
    ):
        raise RuntimeError(f"{label} requires at least two deformation samples")
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
        frame_range = action.get("frame_range")
        sampled_frames = action.get("sampled_frames")
        worst = action.get("worst_case")
        if (
            action.get("requested_action") != expected_action
            or not isinstance(resolved_action, str)
            or expected_action.lower() not in resolved_action.lower()
            or not isinstance(frame_range, list)
            or len(frame_range) != 2
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                for value in frame_range
            )
            or frame_range[1] <= frame_range[0]
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
        expected_source_frames = [
            frame_range[0]
            + (frame_range[1] - frame_range[0])
            * index
            / (expected_samples - 1)
            for index in range(expected_samples)
        ]
        expected_evaluated_frames = [
            int(round(source_frame)) for source_frame in expected_source_frames
        ]
        if len(set(expected_evaluated_frames)) != expected_samples:
            raise RuntimeError(
                f"{label} {expected_action} frame range cannot provide "
                f"{expected_samples} unique evaluated frames"
            )
        for index, frame in enumerate(sampled_frames):
            source_frame = frame.get("source_frame")
            evaluated_frame = frame.get("evaluated_frame")
            if (
                set(frame) != {"source_frame", "evaluated_frame", "metrics"}
                or not isinstance(source_frame, (int, float))
                or isinstance(source_frame, bool)
                or not math.isfinite(source_frame)
                or not math.isclose(
                    source_frame,
                    expected_source_frames[index],
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-9,
                )
                or not isinstance(evaluated_frame, int)
                or isinstance(evaluated_frame, bool)
                or evaluated_frame != expected_evaluated_frames[index]
            ):
                raise RuntimeError(
                    f"{label} {expected_action} sampled frame coverage changed"
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
    *,
    expected_parameters: dict | None = None,
    allow_incomplete: bool = False,
) -> dict:
    payload = load_json_artifact(path, "weight repair manifest")
    if payload.get("schema") != WEIGHT_REPAIR_SCHEMA:
        raise RuntimeError("weight repair schema is missing or unsupported")
    require_file_binding(payload.get("input"), input_glb, "weight repair input")
    require_file_binding(payload.get("output"), output_glb, "weight repair output")
    status = payload.get("status")
    if status == WEIGHT_REPAIR_INCOMPLETE_STATUS and not allow_incomplete:
        raise WeightRepairIncomplete(
            "weight repair did not converge to a readback-ready candidate: "
            f"status={status!r}",
            payload,
        )
    expected_parameters = (
        WEIGHT_REPAIR_PARAMETERS
        if expected_parameters is None
        else expected_parameters
    )
    if (
        payload.get("front_axis") != "positive-x"
        or payload.get("parameters") != expected_parameters
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
    remaining_seed_edges = (
        final.get("remaining_seed_edge_count")
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
        or final_extension < 0.0
        or not isinstance(remaining_seed_edges, int)
        or isinstance(remaining_seed_edges, bool)
        or remaining_seed_edges < 0
    ):
        raise RuntimeError("weight repair convergence evidence did not pass")
    if expected_parameters.get("cross_limb_authority") == "low-slice-components":
        preclean = payload.get("cross_limb_preclean")
        spatial = (
            preclean.get("spatial_authority")
            if isinstance(preclean, dict)
            else None
        )
        component_sizes = (
            spatial.get("substantial_component_sizes_descending")
            if isinstance(spatial, dict)
            else None
        )
        assignment_counts = (
            spatial.get("assignment_counts")
            if isinstance(spatial, dict)
            else None
        )
        component_records = (
            spatial.get("components") if isinstance(spatial, dict) else None
        )
        expected_chains = {
            "front_side_negative",
            "front_side_positive",
            "hind_side_negative",
            "hind_side_positive",
        }
        if (
            not isinstance(preclean, dict)
            or preclean.get("entries_after") != 0
            or preclean.get("weight_mass_after") != 0
            or preclean.get("final_forbidden_entries") != 0
            or preclean.get("final_forbidden_weight_mass") != 0
            or preclean.get("skipped_for_residual_pass")
            is not expected_parameters.get("skip_cross_limb_preclean")
            or not isinstance(spatial, dict)
            or spatial.get("method")
            != "four_largest_disconnected_low_slice_components"
            or spatial.get("immutable_through_motion_repair") is not True
            or spatial.get("blender_up_axis") != "positive-z"
            or spatial.get("height_fraction")
            != expected_parameters.get("limb_slice_height_fraction")
            or not isinstance(component_sizes, list)
            or len(component_sizes) != 4
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in component_sizes
            )
            or not isinstance(assignment_counts, dict)
            or set(assignment_counts) != expected_chains
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                for value in assignment_counts.values()
            )
            or not isinstance(component_records, list)
            or len(component_records) != 4
            or {
                record.get("semantic_chain")
                for record in component_records
                if isinstance(record, dict)
            }
            != expected_chains
        ):
            raise RuntimeError(
                "weight repair low-slice spatial authority did not pass"
            )
    converged = final_extension <= 0.02 and remaining_seed_edges == 0
    if status == WEIGHT_REPAIR_READY_STATUS:
        if not converged:
            raise RuntimeError(
                "weight repair ready status contradicts convergence evidence"
            )
        return payload
    if status == WEIGHT_REPAIR_INCOMPLETE_STATUS:
        if converged:
            raise RuntimeError(
                "weight repair incomplete status contradicts convergence evidence"
            )
        if allow_incomplete:
            return payload
        raise WeightRepairIncomplete(
            "weight repair did not converge to a readback-ready candidate: "
            f"status={status!r}",
            payload,
        )
    raise RuntimeError(f"unsupported weight repair status: {status!r}")


def weight_repair_attempt_record(
    stage: str,
    payload: dict,
    manifest_path: Path,
    output_glb_path: Path,
) -> dict:
    """Bind one repair stage to its unambiguous review-output descriptors."""

    contract = WEIGHT_REPAIR_STAGE_CONTRACTS.get(stage)
    if contract is None:
        raise ValueError(f"unsupported weight repair stage: {stage!r}")
    measurements = payload["final_measurements"]
    return {
        "stage": stage,
        **contract,
        "status": payload["status"],
        "maximum_extension_ratio_of_rest_diagonal": measurements[
            "maximum_extension_ratio_of_rest_diagonal"
        ],
        "remaining_seed_edge_count": measurements["remaining_seed_edge_count"],
        "manifest": file_record(manifest_path),
        "output_glb": file_record(output_glb_path),
    }


def require_weight_repair_branch_consistency(
    branch: str, attempts: list[dict]
) -> None:
    expected_stages = WEIGHT_REPAIR_BRANCH_STAGES.get(branch)
    if expected_stages is None:
        raise ValueError(f"unsupported weight repair branch: {branch!r}")
    if tuple(attempt.get("stage") for attempt in attempts) != expected_stages:
        raise RuntimeError(
            "weight repair attempt sequence contradicts the selected branch"
        )
    for attempt in attempts:
        stage = attempt["stage"]
        contract = WEIGHT_REPAIR_STAGE_CONTRACTS[stage]
        if any(attempt.get(name) != value for name, value in contract.items()):
            raise RuntimeError(
                f"weight repair descriptor contract changed for stage {stage!r}"
            )
        if attempt.get("status") not in {
            WEIGHT_REPAIR_READY_STATUS,
            WEIGHT_REPAIR_INCOMPLETE_STATUS,
        }:
            raise RuntimeError(
                f"weight repair attempt has unsupported status: {stage!r}"
            )
    if branch == "primary":
        if attempts[0]["status"] != WEIGHT_REPAIR_READY_STATUS:
            raise RuntimeError("primary repair branch did not reach ready status")
    elif branch == "fallback_a":
        if (
            attempts[0]["status"] != WEIGHT_REPAIR_INCOMPLETE_STATUS
            or attempts[1]["status"] != WEIGHT_REPAIR_READY_STATUS
        ):
            raise RuntimeError(
                "fallback-A repair branch has contradictory stage status"
            )
    elif branch == "fallback_a_b":
        if (
            attempts[0]["status"] != WEIGHT_REPAIR_INCOMPLETE_STATUS
            or attempts[1]["status"] != WEIGHT_REPAIR_INCOMPLETE_STATUS
            or attempts[-1]["status"] != WEIGHT_REPAIR_READY_STATUS
        ):
            raise RuntimeError(
                "fallback repair branch has contradictory primary/final status"
            )


def require_weight_repair_pipeline_order(
    pipeline_order: list[str], branch: str
) -> None:
    expected = WEIGHT_REPAIR_BRANCH_PIPELINE_STAGES.get(branch)
    if expected is None:
        raise ValueError(f"unsupported weight repair branch: {branch!r}")
    try:
        start = pipeline_order.index("deformation") + 1
        stop = pipeline_order.index("render_walking_side", start)
    except ValueError as error:
        raise RuntimeError(
            "pipeline order lacks deformation/review boundaries"
        ) from error
    if tuple(pipeline_order[start:stop]) != expected:
        raise RuntimeError(
            "pipeline order contradicts the selected weight repair branch"
        )


def weight_repair_gate_attempts(attempts: list[dict]) -> list[dict]:
    return [
        {name: attempt[name] for name in WEIGHT_REPAIR_ATTEMPT_GATE_FIELDS}
        for attempt in attempts
    ]


def weight_repair_output_descriptors(attempts: list[dict]) -> dict[str, dict]:
    outputs: dict[str, dict] = {}
    for attempt in attempts:
        outputs[attempt["output_glb_output_descriptor"]] = dict(
            attempt["output_glb"]
        )
        outputs[attempt["manifest_output_descriptor"]] = dict(
            attempt["manifest"]
        )
    return outputs


def weight_repair_final_artifact(branch: str) -> dict:
    final_stage = WEIGHT_REPAIR_BRANCH_FINAL_STAGE.get(branch)
    if branch not in WEIGHT_REPAIR_BRANCH_FINAL_STAGE:
        raise ValueError(f"unsupported weight repair branch: {branch!r}")
    if final_stage is None:
        return {
            "stage": "not_needed",
            "glb_output_descriptor": "retargeted_animated_glb",
            "manifest_output_descriptor": None,
        }
    contract = WEIGHT_REPAIR_STAGE_CONTRACTS[final_stage]
    return {
        "stage": final_stage,
        "glb_output_descriptor": contract["output_glb_output_descriptor"],
        "manifest_output_descriptor": contract["manifest_output_descriptor"],
    }


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


def require_target_rig_lineage_unchanged(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    target_rig_glb: Path,
    expected_descriptor: dict,
) -> None:
    observed = validate_tokenrig_closure_manifest(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_target_rig_glb=target_rig_glb,
    )
    if observed != expected_descriptor:
        raise RuntimeError(
            "TokenRig source closure changed during quadruped review"
        )


def main(argv=None):
    args = parse_args(argv)
    args.target_rig_glb = regular_file(args.target_rig_glb, "target rig GLB")
    args.source_motion_glb = regular_file(
        args.source_motion_glb, "source motion GLB"
    )
    target_input_record = file_record(args.target_rig_glb)
    motion_input_record = file_record(args.source_motion_glb)
    target_rig_lineage = validate_tokenrig_closure_manifest(
        args.tokenrig_closure_manifest,
        expected_manifest_sha256=args.expected_tokenrig_closure_sha256,
        expected_target_rig_glb=args.target_rig_glb,
    )
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
        donor_readback = assert_declared_motion_donor_artifact(
            declaration["motion_donor_tag"], args.source_motion_glb
        )
        motion_donor_contract = declaration.get(
            "expected_motion_donor_contract"
        ) or expected_motion_donor_contract(declaration["motion_donor_tag"])
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
            "target_species": motion_donor_contract["target_species"],
            "motion_donor_tag": declaration["motion_donor_tag"],
            "motion_donor": donor_readback,
            "motion_donor_contract": dict(motion_donor_contract),
            "derived_motion_basis": dict(basis),
            "target_asset_authority": {
                "input_glb": dict(target_input_record),
                "mesh_geometry_source": "target_rig_glb",
                "motion_donor_geometry_used": False,
            },
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
        motion_donor_contract = None
        if not args.validate_only:
            raise ValueError(
                "legacy free-parameter mode is plan-only and cannot execute; "
                "build an authenticated --forward-declaration"
            )
    forward_contract["target_rig_lineage"] = dict(target_rig_lineage)
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
        "heading", "rig_audit", "support_plane", "support_plane_readback",
        "retarget", "gait_direction",
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
                    "forward_contract": forward_contract,
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
    media = {}
    media_lineage = {}
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
            elif label == "support_plane_readback":
                support_plane = require_support_plane(
                    paths["level_manifest"],
                    paths["heading_glb"],
                    paths["leveled_glb"],
                    readback_path=paths["level_readback"],
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
                    motion_donor_contract=motion_donor_contract,
                )
            elif label == "gait_direction":
                gait = require_gait_audit(
                    paths["gait_audit"],
                    paths["animated_glb"],
                    "initial gait audit",
                )
            elif label.startswith("render_"):
                require_review_render_stage(
                    paths,
                    label.removeprefix("render_"),
                    paths["animated_glb"],
                    args.review_frames,
                )
            elif label.startswith("encode_"):
                media_label = label.removeprefix("encode_")
                media[media_label], media_lineage[media_label] = (
                    require_review_encode_stage(
                        paths,
                        media_label,
                        paths["animated_glb"],
                        args.review_frames,
                    )
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
        require_target_rig_lineage_unchanged(
            manifest_path=args.tokenrig_closure_manifest,
            expected_manifest_sha256=args.expected_tokenrig_closure_sha256,
            target_rig_glb=args.target_rig_glb,
            expected_descriptor=target_rig_lineage,
        )
        readback_started = time.monotonic()
        media, media_lineage = require_review_media_set(
            paths,
            ("walking_side", "walking_front"),
            paths["animated_glb"],
            args.review_frames,
        )
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
                "target_rig_lineage": dict(target_rig_lineage),
                "heading_review_evidence": dict(heading_input_record),
                "source_motion_glb": dict(motion_input_record),
            },
            "outputs": {
                "heading_manifest": file_record(paths["heading_manifest"]),
                "rig_audit": file_record(paths["rig_audit"]),
                "support_plane_manifest": file_record(paths["level_manifest"]),
                "support_plane_output_readback": file_record(
                    paths["level_readback"]
                ),
                "animated_glb": file_record(paths["animated_glb"]),
                "retarget_manifest": file_record(paths["retarget_manifest"]),
                "gait_direction_audit": file_record(paths["gait_audit"]),
                "gait_direction_status": gait.get("status"),
                "media": media,
                "media_lineage": media_lineage,
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
        elif label == "support_plane_readback":
            support_plane = require_support_plane(
                paths["level_manifest"],
                paths["heading_glb"],
                paths["leveled_glb"],
                readback_path=paths["level_readback"],
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
                motion_donor_contract=motion_donor_contract,
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
    final_repair_manifest_path = None
    repair_strategy = "not_needed"
    repair_branch = "not_needed"
    repair_attempts = []
    if repair_triggered:
        primary_commands = build_repair_commands(args, paths, blender)
        primary_label, primary_command = primary_commands[0]
        run_stage(primary_label, primary_command, timings, pipeline_order)
        primary_manifest = require_weight_repair(
            paths["weight_repair_manifest"],
            paths["animated_glb"],
            paths["repaired_glb"],
            expected_parameters=WEIGHT_REPAIR_PARAMETERS,
            allow_incomplete=True,
        )
        repair_attempts.append(
            weight_repair_attempt_record(
                "primary",
                primary_manifest,
                paths["weight_repair_manifest"],
                paths["repaired_glb"],
            )
        )
        if primary_manifest["status"] == WEIGHT_REPAIR_READY_STATUS:
            repair_manifest = primary_manifest
            final_repair_manifest_path = paths["weight_repair_manifest"]
            repair_strategy = "component_parent_lock"
            repair_branch = "primary"
            post_repair_commands = primary_commands[1:]
            reviewed_glb = paths["repaired_glb"]
            final_gait_path = paths["repaired_gait_audit"]
            final_deformation_path = paths["repaired_deformation_audit"]
        else:
            fallback_commands = build_fallback_repair_commands(
                args, paths, blender
            )
            fallback_label, fallback_command = fallback_commands[0]
            run_stage(
                fallback_label,
                fallback_command,
                timings,
                pipeline_order,
            )
            fallback_manifest = require_weight_repair(
                paths["fallback_weight_repair_manifest"],
                paths["animated_glb"],
                paths["fallback_repaired_glb"],
                expected_parameters=WEIGHT_REPAIR_FALLBACK_PARAMETERS,
                allow_incomplete=True,
            )
            repair_attempts.append(
                weight_repair_attempt_record(
                    "fallback_a",
                    fallback_manifest,
                    paths["fallback_weight_repair_manifest"],
                    paths["fallback_repaired_glb"],
                )
            )
            if fallback_manifest["status"] == WEIGHT_REPAIR_READY_STATUS:
                repair_manifest = fallback_manifest
                final_repair_manifest_path = paths[
                    "fallback_weight_repair_manifest"
                ]
                repair_strategy = "low_slice_edge_average"
                repair_branch = "fallback_a"
                post_repair_commands = fallback_commands[1:3]
                reviewed_glb = paths["fallback_repaired_glb"]
                final_gait_path = paths["fallback_gait_audit"]
                final_deformation_path = paths[
                    "fallback_deformation_audit"
                ]
            else:
                residual_label, residual_command = fallback_commands[3]
                run_stage(
                    residual_label,
                    residual_command,
                    timings,
                    pipeline_order,
                )
                repair_manifest = require_weight_repair(
                    paths["fallback_residual_weight_repair_manifest"],
                    paths["fallback_repaired_glb"],
                    paths["fallback_residual_repaired_glb"],
                    expected_parameters=(
                        WEIGHT_REPAIR_FALLBACK_RESIDUAL_PARAMETERS
                    ),
                )
                final_repair_manifest_path = paths[
                    "fallback_residual_weight_repair_manifest"
                ]
                repair_attempts.append(
                    weight_repair_attempt_record(
                        "fallback_b",
                        repair_manifest,
                        paths[
                            "fallback_residual_weight_repair_manifest"
                        ],
                        paths["fallback_residual_repaired_glb"],
                    )
                )
                repair_strategy = "low_slice_edge_average_residual"
                repair_branch = "fallback_a_b"
                post_repair_commands = fallback_commands[4:]
                reviewed_glb = paths["fallback_residual_repaired_glb"]
                final_gait_path = paths[
                    "fallback_residual_gait_audit"
                ]
                final_deformation_path = paths[
                    "fallback_residual_deformation_audit"
                ]
        final_gait = None
        final_deformation = None
        for label, command in post_repair_commands:
            run_stage(label, command, timings, pipeline_order)
            if label in {
                "gait_direction_repaired",
                "gait_direction_repaired_low_slice_edge_average",
                "gait_direction_repaired_low_slice_edge_average_residual",
            }:
                final_gait = require_gait_audit(
                    final_gait_path,
                    reviewed_glb,
                    "repaired gait audit",
                )
            elif label in {
                "deformation_repaired",
                "deformation_repaired_low_slice_edge_average",
                "deformation_repaired_low_slice_edge_average_residual",
            }:
                final_deformation = require_deformation_audit(
                    final_deformation_path,
                    reviewed_glb,
                    "repaired deformation audit",
                    expected_samples=args.deformation_samples,
                    require_pass=True,
                )
        if (
            repair_manifest is None
            or final_repair_manifest_path is None
            or final_gait is None
            or final_deformation is None
        ):
            raise RuntimeError("weight repair omitted a required post-repair gate")
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

    require_weight_repair_branch_consistency(repair_branch, repair_attempts)
    if repair_strategy != WEIGHT_REPAIR_BRANCH_STRATEGY[repair_branch]:
        raise RuntimeError(
            "weight repair strategy contradicts the selected branch"
        )

    for label, command in build_review_commands(
        args, paths, blender, reviewed_glb
    ):
        run_stage(label, command, timings, pipeline_order)
        if label.startswith("render_"):
            require_review_render_stage(
                paths,
                label.removeprefix("render_"),
                reviewed_glb,
                args.review_frames,
            )
        elif label.startswith("encode_"):
            media_label = label.removeprefix("encode_")
            media[media_label], media_lineage[media_label] = (
                require_review_encode_stage(
                    paths,
                    media_label,
                    reviewed_glb,
                    args.review_frames,
                )
            )
    require_weight_repair_pipeline_order(pipeline_order, repair_branch)

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
    require_target_rig_lineage_unchanged(
        manifest_path=args.tokenrig_closure_manifest,
        expected_manifest_sha256=args.expected_tokenrig_closure_sha256,
        target_rig_glb=args.target_rig_glb,
        expected_descriptor=target_rig_lineage,
    )
    readback_started = time.monotonic()
    media, media_lineage = require_review_media_set(
        paths,
        tuple(spec[0] for spec in REVIEW_MEDIA_SPECS),
        reviewed_glb,
        args.review_frames,
    )
    timings["media_readback"] = time.monotonic() - readback_started
    pipeline_order.append("media_readback")
    final_animation_record = file_record(reviewed_glb)
    repair_final_artifact = weight_repair_final_artifact(repair_branch)
    repair_outputs = weight_repair_output_descriptors(repair_attempts)
    if repair_triggered:
        final_glb_descriptor = repair_final_artifact["glb_output_descriptor"]
        final_manifest_descriptor = repair_final_artifact[
            "manifest_output_descriptor"
        ]
        final_manifest_record = file_record(final_repair_manifest_path)
        if (
            repair_outputs.get(final_glb_descriptor) != final_animation_record
            or repair_outputs.get(final_manifest_descriptor)
            != final_manifest_record
        ):
            raise RuntimeError(
                "weight repair final artifact contradicts its named descriptors"
            )
        # This compatibility alias remains hash-identical to the branch-specific
        # final descriptor.  Downstream bridges must resolve and compare both.
        repair_outputs["weight_repair_manifest"] = final_manifest_record
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
            "weight_repair_strategy": repair_strategy,
            "weight_repair_branch": repair_branch,
            "weight_repair_attempts": weight_repair_gate_attempts(
                repair_attempts
            ),
            "weight_repair_final_artifact": repair_final_artifact,
            "gait_final": final_gait["status"],
            "deformation_final": final_deformation["overall"],
            "all_automatic_gates_passed": True,
        },
        "inputs": {
            "target_rig_glb": dict(target_input_record),
            "target_rig_lineage": dict(target_rig_lineage),
            "heading_review_evidence": dict(heading_input_record),
            "source_motion_glb": dict(motion_input_record),
        },
        "outputs": {
            "heading_manifest": file_record(paths["heading_manifest"]),
            "rig_audit": file_record(paths["rig_audit"]),
            "support_plane_manifest": file_record(paths["level_manifest"]),
            "support_plane_output_readback": file_record(
                paths["level_readback"]
            ),
            "retargeted_animated_glb": file_record(paths["animated_glb"]),
            "animated_glb": final_animation_record,
            "retarget_manifest": file_record(paths["retarget_manifest"]),
            "gait_direction_audit_initial": file_record(paths["gait_audit"]),
            "gait_direction_audit": file_record(final_gait_path),
            "gait_direction_status": final_gait["status"],
            "deformation_audit_initial": file_record(paths["deformation_audit"]),
            "deformation_audit": file_record(final_deformation_path),
            "deformation_overall": final_deformation["overall"],
            **repair_outputs,
            "media": media,
            "media_lineage": media_lineage,
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
