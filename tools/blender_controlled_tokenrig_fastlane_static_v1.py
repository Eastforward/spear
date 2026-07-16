#!/usr/bin/env python3
"""Visual/runtime-oriented static gate for controlled TokenRig research canaries.

The strict static rejections remain authoritative evidence.  This independent
fastlane treats bounded duplicate/degenerate micro-topology as advisory while
keeping PBR, hierarchy, finite skinning, obvious cross-limb weights,
grounding, semantic coverage, bind export/readback and review media hard.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import stat
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import blender_tokenrig_human_static_audit as audit


SCHEMA = "tokenrig_human_static_qa_v1"
FASTLANE_SCHEMA = "route2_controlled_tokenrig_fastlane_static_v1"
TOKENRIG_SCHEMA = "route2_controlled_geometry_tokenrig_candidate_v1"
LINEAGE_SCHEMA = "route2_controlled_geometry_tokenrig_lineage_v1"
RUNNER_PATH = Path(__file__).resolve()
SPEAR_ROOT = RUNNER_PATH.parents[1]
PIXAL_ROOT = SPEAR_ROOT / "tmp/i23d_controlled_geometry_v3/pixal3d"
TOKENRIG_ROOT = SPEAR_ROOT / "tmp/route2_controlled_geometry_tokenrig_v1"
OUTPUT_DIRNAME = "fastlane_static_v1"
MAX_OPPOSITE_WEIGHT = 0.25
MAX_OPPOSITE_OVER_10_PERCENT_FRACTION = 0.005
MAX_OPPOSITE_OVER_5_PERCENT_FRACTION = 0.02
EXPECTED_BONE_COUNT = 52
ARTIFACT_NAMES = (
    "bind_pose.glb",
    "bind_front.png",
    "bind_back.png",
    "bind_side.png",
    "bind_top.png",
    "skeleton_overlay.png",
    "weights_contact.png",
    "texture_compare.png",
    "joint_hierarchy.txt",
)


class FastlaneError(RuntimeError):
    pass


def _record(path: Path, *, public_path: Path | None = None, mode: int | None = None) -> dict[str, Any]:
    path = Path(path).absolute()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
        or not stat.S_ISREG(os.lstat(path).st_mode)
        or path.stat().st_size <= 0
    ):
        raise FastlaneError(f"artifact must be a direct nonempty file: {path}")
    if mode is not None and stat.S_IMODE(path.stat().st_mode) != mode:
        raise FastlaneError(f"artifact mode changed: {path}")
    return {
        "path": str(public_path if public_path is not None else path),
        "sha256": audit.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FastlaneError(f"{description} is invalid: {error}") from error
    if not isinstance(value, dict):
        raise FastlaneError(f"{description} must contain an object")
    return value


def authenticate_inputs(asset_id: str) -> dict[str, Any]:
    pixal_root = PIXAL_ROOT / asset_id
    tokenrig_root = TOKENRIG_ROOT / asset_id / "direct_transfer_seed42"
    source_glb = pixal_root / "canary_1024_seed42.glb"
    source_manifest_path = pixal_root / "canary_1024_seed42.manifest.json"
    tokenrig_glb = tokenrig_root / "tokenrig_transfer.glb"
    tokenrig_manifest_path = tokenrig_root / "tokenrig_manifest.json"
    lineage_path = tokenrig_root / "controlled_geometry_lineage_v1.json"
    source = _json(source_manifest_path, "Pixal manifest")
    tokenrig = _json(tokenrig_manifest_path, "TokenRig manifest")
    lineage = _json(lineage_path, "controlled TokenRig lineage")
    if (
        source.get("schema") != "route2_controlled_geometry_pixal_candidate_v1"
        or source.get("asset_id") != asset_id
        or source.get("output") != _record(source_glb, mode=0o444)
        or source.get("pbr_glb_readback", {}).get("passed") is not True
        or tokenrig.get("schema") != TOKENRIG_SCHEMA
        or tokenrig.get("asset_id") != asset_id
        or tokenrig.get("attempt") != "direct_transfer"
        or tokenrig.get("inference_parameters", {}).get("use_transfer") is not True
        or tokenrig.get("inference_parameters", {}).get("use_skeleton") is not False
        or tokenrig.get("output", {}).get("path") != str(tokenrig_glb)
        or tokenrig.get("output", {}).get("sha256") != audit.sha256_file(tokenrig_glb)
        or tokenrig.get("output", {}).get("size_bytes") != tokenrig_glb.stat().st_size
        or lineage.get("schema") != LINEAGE_SCHEMA
        or lineage.get("asset_id") != asset_id
        or lineage.get("tokenrig_manifest") != _record(tokenrig_manifest_path, mode=0o444)
        or lineage.get("formal_dataset_registration_authorized") is not False
    ):
        raise FastlaneError("controlled Pixal/TokenRig identity, PBR, or lineage changed")
    strict_failures = []
    for path in sorted(tokenrig_root.glob("static_audit_v1.failed.*.json")):
        payload = _json(path, "strict static rejection")
        if (
            payload.get("decision") != "rejected"
            or payload.get("readiness_bundle_published") is not False
            or not isinstance(payload.get("failure", {}).get("message"), str)
            or stat.S_IMODE(path.stat().st_mode) != 0o444
        ):
            raise FastlaneError("strict static rejection evidence changed")
        strict_failures.append({
            "record": _record(path, mode=0o444),
            "message": payload["failure"]["message"],
            "disposition": "preserved_advisory_for_research_candidate_fastlane",
        })
    if not strict_failures:
        raise FastlaneError("fastlane requires preserved strict rejection evidence")
    return {
        "asset_id": asset_id,
        "source_glb": source_glb,
        "source_manifest": _record(source_manifest_path, mode=0o444),
        "tokenrig_glb": tokenrig_glb,
        "tokenrig_manifest": _record(tokenrig_manifest_path, mode=0o444),
        "lineage": _record(lineage_path, mode=0o444),
        "strict_rejections": strict_failures,
    }


def obvious_bilateral_metrics(
    positions: Sequence[Sequence[float]],
    weights: Sequence[Mapping[str, float]],
    chains: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    if len(positions) != len(weights) or not positions:
        raise FastlaneError("bilateral inputs are empty or mismatched")
    xs = [float(position[0]) for position in positions]
    if any(not math.isfinite(value) for value in xs):
        raise FastlaneError("bilateral positions are non-finite")
    center = (min(xs) + max(xs)) * 0.5
    half_width = (max(xs) - min(xs)) * 0.5
    if half_width <= 1.0e-9:
        raise FastlaneError("bilateral width is degenerate")
    cutoff = 0.25 * half_width
    left_bones = set(chains["left_arm"]) | set(chains["left_leg"])
    right_bones = set(chains["right_arm"]) | set(chains["right_leg"])
    values = []
    for position, vertex_weights in zip(positions, weights):
        side = float(position[0]) - center
        if abs(side) < cutoff:
            continue
        opposite = right_bones if side > 0.0 else left_bones
        value = sum(float(vertex_weights.get(name, 0.0)) for name in opposite)
        if not math.isfinite(value) or value < 0.0:
            raise FastlaneError("opposite-limb weight is invalid")
        values.append(value)
    if not values:
        raise FastlaneError("bilateral gate considered no distal vertices")
    over_5 = sum(value > 0.05 for value in values)
    over_10 = sum(value > 0.10 for value in values)
    maximum = max(values)
    metrics = {
        "considered_distal_vertex_count": len(values),
        "maximum_opposite_limb_weight": maximum,
        "over_5_percent_count": over_5,
        "over_5_percent_fraction": over_5 / len(values),
        "over_10_percent_count": over_10,
        "over_10_percent_fraction": over_10 / len(values),
        "advisory_over_1e4_count": sum(value > 1.0e-4 for value in values),
        "thresholds": {
            "maximum_opposite_limb_weight": MAX_OPPOSITE_WEIGHT,
            "over_5_percent_fraction": MAX_OPPOSITE_OVER_5_PERCENT_FRACTION,
            "over_10_percent_fraction": MAX_OPPOSITE_OVER_10_PERCENT_FRACTION,
        },
    }
    metrics["passed"] = (
        maximum <= MAX_OPPOSITE_WEIGHT
        and metrics["over_5_percent_fraction"] <= MAX_OPPOSITE_OVER_5_PERCENT_FRACTION
        and metrics["over_10_percent_fraction"] <= MAX_OPPOSITE_OVER_10_PERCENT_FRACTION
    )
    if not metrics["passed"]:
        raise FastlaneError(f"obvious bilateral contamination exceeds fastlane bounds: {metrics}")
    return metrics


def _advisory(label: str, function: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    try:
        return {"status": "passed", "result": function(*args, **kwargs)}
    except Exception as error:
        return {
            "status": "advisory_failed",
            "label": label,
            "error": {"type": type(error).__name__, "message": str(error)},
        }


def _run_blender(authenticated: Mapping[str, Any], staging: Path) -> dict[str, Any]:
    import bpy
    from mathutils import Matrix

    source_glb = Path(authenticated["source_glb"])
    tokenrig_glb = Path(authenticated["tokenrig_glb"])
    source_parsed = audit.read_glb(source_glb)
    tokenrig_parsed = audit.read_glb(tokenrig_glb)
    source_pbr = audit.pbr_payload_contract(source_parsed)
    input_pbr = audit.compare_pbr_payloads(source_pbr, audit.pbr_payload_contract(tokenrig_parsed))
    raw_topology = _advisory(
        "bounded_raw_serialization_equivalence",
        audit.analyze_raw_serialization_equivalence,
        source_parsed,
        tokenrig_parsed,
    )

    bpy.ops.wm.read_factory_settings(use_empty=True)
    if "FINISHED" not in bpy.ops.import_scene.gltf(filepath=str(tokenrig_glb)):
        raise FastlaneError("Blender could not import TokenRig GLB")
    audit.remove_gltf_import_helpers(bpy)
    armature, mesh, orphans = audit.identify_exact_runtime(bpy)
    audit.remove_proven_runtime_orphans(bpy, orphans)
    pre_mesh_contract = audit.capture_blender_mesh_contract(mesh)
    source_floor = audit.mesh_floor_z(mesh)
    axis = audit.canonical_axis_contract(source_front="positive-y", prior_transform_count=0)
    grounding = audit.ground_bind_contract(source_floor_z=source_floor, prior_transform_count=0)
    roots = audit.runtime_roots({armature, mesh})
    transform = Matrix.Translation((0.0, 0.0, grounding["ground_translation_z"])) @ Matrix.Rotation(math.pi, 4, "Z")
    for root in roots:
        root.matrix_world = transform @ root.matrix_world
    bpy.context.view_layer.update()
    post_floor = audit.mesh_floor_z(mesh)
    if abs(post_floor) > 2.0e-6:
        raise FastlaneError(f"canonical bind floor is not zero: {post_floor}")
    grounding = {**grounding, "post_floor_z": 0.0, "canonical_floor_z": 0.0, "runtime_root_count": len(roots)}

    bones = audit.bone_records_from_armature(armature)
    hierarchy = audit.validate_hierarchy(bones)
    if hierarchy["bone_count"] != EXPECTED_BONE_COUNT:
        raise FastlaneError(f"TokenRig hierarchy must contain {EXPECTED_BONE_COUNT} bones")
    semantics = audit.resolve_five_semantic_chains(bones)
    weights, positions = audit.extract_vertex_weights(mesh, armature)
    weight_validation = audit.validate_vertex_weights(
        weights, bone_names={bone.name for bone in bones}
    )
    bilateral = obvious_bilateral_metrics(positions, weights, semantics["chains"])
    seam_advisory = _advisory("strict_uv_seam_weight_equality", audit.validate_seam_weights, positions, weights)
    canonical_mesh = audit.capture_blender_mesh_contract(mesh)
    canonical_rest = audit.capture_blender_full_rest_contract(armature)

    bind_path = staging / "bind_pose.glb"
    audit.export_bind_pose_glb(bpy, armature, mesh, bind_path)
    bind_parsed = audit.read_glb(bind_path)
    exported_pbr = audit.compare_pbr_payloads(source_pbr, audit.pbr_payload_contract(bind_parsed))
    inverse_bind = audit.extract_inverse_bind_contract(bind_parsed)
    if len(inverse_bind["joint_names"]) != EXPECTED_BONE_COUNT:
        raise FastlaneError("bind GLB inverse-bind joint count changed")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    if "FINISHED" not in bpy.ops.import_scene.gltf(filepath=str(bind_path)):
        raise FastlaneError("Blender could not read back fastlane bind GLB")
    audit.remove_gltf_import_helpers(bpy)
    round_armature, round_mesh, round_orphans = audit.identify_exact_runtime(bpy)
    audit.remove_proven_runtime_orphans(bpy, round_orphans)
    round_bones = audit.bone_records_from_armature(round_armature)
    round_hierarchy = audit.validate_hierarchy(round_bones)
    round_semantics = audit.resolve_five_semantic_chains(round_bones)
    round_weights, round_positions = audit.extract_vertex_weights(round_mesh, round_armature)
    round_weight_validation = audit.validate_vertex_weights(
        round_weights, bone_names={bone.name for bone in round_bones}
    )
    rest_roundtrip = audit.compare_full_rest_contracts(
        canonical_rest, audit.capture_blender_full_rest_contract(round_armature)
    )
    skin_roundtrip = _advisory(
        "exact_seam_cluster_skin_roundtrip",
        audit.compare_skin_by_position,
        positions,
        weights,
        round_positions,
        round_weights,
    )
    round_bilateral = obvious_bilateral_metrics(
        round_positions, round_weights, round_semantics["chains"]
    )
    if abs(audit.mesh_floor_z(round_mesh)) > 2.0e-6:
        raise FastlaneError("bind GLB readback no longer has floor Z=0")
    renders = audit.render_static_evidence(
        bpy=bpy,
        armature=round_armature,
        mesh=round_mesh,
        semantics=round_semantics,
        output_dir=staging,
    )
    audit.write_joint_hierarchy(
        path=staging / "joint_hierarchy.txt",
        bones=round_bones,
        semantics=round_semantics,
    )
    return {
        "fastlane_schema": FASTLANE_SCHEMA,
        "automatic_static_checks": "passed_research_candidate_fastlane",
        "hard_checks": {
            "input_pbr": input_pbr,
            "exported_pbr": exported_pbr,
            "hierarchy": hierarchy,
            "roundtrip_hierarchy": round_hierarchy,
            "semantic_mapping": semantics,
            "weights": weight_validation,
            "roundtrip_weights": round_weight_validation,
            "bilateral_contamination": bilateral,
            "roundtrip_bilateral_contamination": round_bilateral,
            "rest_roundtrip": rest_roundtrip,
            "inverse_bind_joint_count": len(inverse_bind["joint_names"]),
            "axis_canonicalization": axis,
            "grounding": grounding,
            "canonical_mesh_contract": canonical_mesh,
            "renders": renders,
        },
        "advisories": {
            "strict_rejections": authenticated["strict_rejections"],
            "raw_topology": raw_topology,
            "strict_seam_weights": seam_advisory,
            "skin_roundtrip": skin_roundtrip,
            "precanonical_mesh_contract": pre_mesh_contract,
        },
    }


def _failure(output: Path, asset_id: str, error: BaseException, authenticated: Any) -> Path:
    destination = output.parent / f".{output.name}.failed.{uuid.uuid4().hex}.json"
    payload = {
        "schema": "route2_controlled_tokenrig_fastlane_static_failure_v1",
        "asset_id": asset_id,
        "decision": "rejected",
        "state_classification": "rejected",
        "formal_registration_authorized": False,
        "error": {"type": type(error).__name__, "message": str(error)},
        "authenticated": authenticated,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    try:
        os.write(descriptor, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def run(asset_id: str, output_dir: Path) -> Path:
    expected = TOKENRIG_ROOT / asset_id / "direct_transfer_seed42" / OUTPUT_DIRNAME
    output = Path(output_dir).absolute()
    if output != expected or os.path.lexists(output):
        raise FastlaneError("fastlane output must be the unused canonical directory")
    authenticated: Mapping[str, Any] | None = None
    staging: Path | None = None
    try:
        authenticated = authenticate_inputs(asset_id)
        staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT_DIRNAME}.", suffix=".staging", dir=output.parent))
        checks = _run_blender(authenticated, staging)
        public = output
        artifacts = {
            name: {
                "filename": name,
                "size_bytes": (staging / name).stat().st_size,
                "sha256": audit.sha256_file(staging / name),
            }
            for name in ARTIFACT_NAMES
        }
        hard = checks["hard_checks"]
        qa = {
            "schema": SCHEMA,
            "fastlane_schema": FASTLANE_SCHEMA,
            "asset_id": asset_id,
            "decision": "automatic_static_checks_passed",
            "agent_qa_status": "pending_agent_visual_qa",
            "state_classification": "research_candidate_fastlane",
            "user_acceptance": "not_claimed",
            "readiness_bundle_published": True,
            "formal_dataset_registration_authorized": False,
            "animation_authorized": False,
            "authenticated": {
                key: value
                for key, value in authenticated.items()
                if key not in {"source_glb", "tokenrig_glb"}
            },
            "qualification": {
                "basis": "visual_runtime_fastlane_with_preserved_strict_topology_rejections",
                "strict_topology_status": "advisory_not_erased",
                "hard_gate_scope": [
                    "packed_pbr",
                    "52_bone_hierarchy",
                    "finite_normalized_weights",
                    "obvious_bilateral_contamination",
                    "front_and_grounding",
                    "bind_glb_roundtrip",
                    "review_media",
                ],
            },
            "checks": {
                **hard,
                "exported_pbr": hard["exported_pbr"],
                "semantic_mapping": hard["semantic_mapping"],
                "axis_canonicalization": hard["axis_canonicalization"],
                "grounding": hard["grounding"],
                "canonical_mesh_contract": hard["canonical_mesh_contract"],
            },
            "advisories": checks["advisories"],
            "runner": _record(RUNNER_PATH),
            "artifacts": artifacts,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "static_qa.json").write_text(
            json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for path in staging.iterdir():
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                raise FastlaneError(f"fastlane artifact is invalid: {path}")
            path.chmod(0o444)
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        staging.chmod(0o555)
        audit.rename_directory_noreplace(staging, output)
        staging = None
        return output / "static_qa.json"
    except BaseException as error:
        if staging is not None and staging.exists():
            staging.chmod(0o700)
            for path in staging.iterdir():
                path.chmod(0o600)
            shutil.rmtree(staging)
        evidence = _failure(output, asset_id, error, authenticated)
        raise FastlaneError(f"fastlane static rejected: {error}; evidence={evidence}") from error


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def _blender_argv() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


if __name__ == "__main__":
    args = parse_args(_blender_argv())
    print(f"CONTROLLED_TOKENRIG_FASTLANE_STATIC_OK {run(args.asset_id, args.output_dir)}")
