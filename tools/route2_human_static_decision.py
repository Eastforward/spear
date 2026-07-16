#!/usr/bin/env python3
"""Immutable hash-locked agent visual decision for a Route-2 static bundle."""

from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Mapping

from tools import route2_human_contract_common as common
from tools import route2_human_instance_contract as instance_contract


SCHEMA = "route2_human_static_agent_visual_qa_v1"
STATIC_QA_SCHEMA = "tokenrig_human_static_qa_v1"
STATIC_BUNDLE_DIRNAME = "static_audit_v1"
DECISION_FILENAME = f"{STATIC_BUNDLE_DIRNAME}.agent_visual_qa.json"
PASS_STATUS = "agent_qa_passed_pending_user_acceptance"
REJECTED_STATUS = "rejected"
STATIC_ARTIFACTS = (
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
ACCESSORY_CASES = frozenset({"glasses", "hat"})
ACCESSORY_ARTIFACTS = ("accessory_head_binding.json", "accessory_head_binding.png")
STATIC_VISUAL_CHECKS = (
    "bind_pose_anatomically_reasonable",
    "front_back_side_top_consistent",
    "skeleton_covers_required_anatomy",
    "skeleton_aligned_with_mesh",
    "weight_distribution_reasonable",
    "pbr_textures_preserved",
    "feet_grounded",
    "garments_intact",
    "head_attachments_stable_if_present",
)
REQUIRED_STATIC_CHECKS = frozenset(
    {
        "automatic_static_checks",
        "raw_triangle_contract",
        "input_pbr",
        "input_mesh",
        "input_surface",
        "source_mesh_contract",
        "source_removed_gltf_import_helpers",
        "tokenrig_mesh_contract_before_canonical_transform",
        "tokenrig_removed_gltf_import_helpers",
        "removed_proven_orphans",
        "axis_canonicalization",
        "grounding",
        "canonical_mesh_contract",
        "hierarchy",
        "semantic_mapping",
        "weights",
        "seams",
        "bilateral_contamination",
        "exported_pbr",
        "glb_roundtrip",
        "renders",
    }
)
REQUIRED_SEMANTIC_BONES = frozenset(
    {
        "pelvis",
        "spine",
        "head",
        "left_upper_arm",
        "left_forearm",
        "left_hand",
        "right_upper_arm",
        "right_forearm",
        "right_hand",
        "left_thigh",
        "left_calf",
        "left_foot",
        "left_toe",
        "right_thigh",
        "right_calf",
        "right_foot",
        "right_toe",
    }
)


class StaticDecisionError(RuntimeError):
    """Static evidence or its immutable agent decision is invalid or stale."""


class StaticDecisionNotAccepted(StaticDecisionError):
    """The exact static snapshot was rejected or has no passing agent decision."""


def _translate_instance_error(error: Exception) -> StaticDecisionError:
    return StaticDecisionError(f"instance contract is invalid: {error}")


def _contract(contract_path: Path) -> tuple[dict[str, Any], Path, Path]:
    try:
        contract = instance_contract.validate_instance_contract(contract_path)
    except instance_contract.InstanceContractError as error:
        raise _translate_instance_error(error) from error
    path = common.absolute(contract_path)
    root = Path(contract["canonical_output_root"])
    return contract, path, root


def _branch_root(contract: Mapping[str, Any], branch_id: str) -> Path:
    try:
        root = instance_contract.resolve_branch_root(contract, branch_id)
    except instance_contract.InstanceContractError as error:
        raise StaticDecisionError(f"branch contract is invalid: {error}") from error
    return common.require_real_directory(root, "final static branch", StaticDecisionError)


def static_bundle_path(contract: Mapping[str, Any], branch_id: str) -> Path:
    return _branch_root(contract, branch_id) / STATIC_BUNDLE_DIRNAME


def static_decision_path(contract: Mapping[str, Any], branch_id: str) -> Path:
    return _branch_root(contract, branch_id) / DECISION_FILENAME


def _artifact_record(path: Path, root: Path, description: str) -> dict[str, Any]:
    return common.file_record(
        path,
        root=root,
        description=description,
        error_type=StaticDecisionError,
        require_mode=0o444,
    )


def _validate_static_artifact_formats(static_root: Path) -> dict[str, Any]:
    from PIL import Image, UnidentifiedImageError

    document, _ = common.load_glb_document_record(
        static_root / "bind_pose.glb",
        root=static_root,
        description="static artifact bind_pose.glb",
        error_type=StaticDecisionError,
        require_mode=0o444,
    )
    meshes = document.get("meshes")
    skins = document.get("skins")
    nodes = document.get("nodes")
    animations = document.get("animations", [])
    if (
        not isinstance(meshes, list)
        or not meshes
        or not isinstance(skins, list)
        or len(skins) != 1
        or not isinstance(skins[0], Mapping)
        or not isinstance(skins[0].get("joints"), list)
        or not skins[0]["joints"]
        or not isinstance(nodes, list)
        or not nodes
        or animations not in (None, [])
    ):
        raise StaticDecisionError(
            "static artifact bind_pose.glb must contain mesh, one non-empty skin, and no animation"
        )
    joints = skins[0]["joints"]
    if (
        any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(nodes)
            for index in joints
        )
        or len(joints) != len(set(joints))
    ):
        raise StaticDecisionError("bind_pose.glb skin joint indices are invalid")
    joint_names: list[str] = []
    for index in joints:
        node = nodes[index]
        name = node.get("name") if isinstance(node, Mapping) else None
        if not isinstance(name, str) or not name.strip():
            raise StaticDecisionError("bind_pose.glb skin joint name is missing")
        joint_names.append(name)
    if len(joint_names) != len(set(joint_names)):
        raise StaticDecisionError("bind_pose.glb skin joint names are not unique")

    node_parents: dict[int, int] = {}
    for parent_index, node in enumerate(nodes):
        children = node.get("children", []) if isinstance(node, Mapping) else []
        if not isinstance(children, list):
            raise StaticDecisionError("bind_pose.glb node children are invalid")
        for child_index in children:
            if (
                not isinstance(child_index, int)
                or isinstance(child_index, bool)
                or not 0 <= child_index < len(nodes)
                or child_index in node_parents
            ):
                raise StaticDecisionError("bind_pose.glb node hierarchy is invalid")
            node_parents[child_index] = parent_index
    joint_set = set(joints)
    joint_name_by_index = dict(zip(joints, joint_names))
    joint_parents: dict[str, str | None] = {}
    for joint_index in joints:
        seen: set[int] = set()
        parent_index = node_parents.get(joint_index)
        while parent_index is not None and parent_index not in joint_set:
            if parent_index in seen:
                raise StaticDecisionError("bind_pose.glb node hierarchy contains a cycle")
            seen.add(parent_index)
            parent_index = node_parents.get(parent_index)
        joint_parents[joint_name_by_index[joint_index]] = (
            joint_name_by_index[parent_index]
            if parent_index is not None
            else None
        )
    roots = [name for name, parent in joint_parents.items() if parent is None]
    if len(roots) != 1:
        raise StaticDecisionError("bind_pose.glb skin hierarchy must have exactly one root")
    position = {name: index for index, name in enumerate(joint_names)}
    if any(
        parent is not None and position[parent] >= position[name]
        for name, parent in joint_parents.items()
    ):
        raise StaticDecisionError("bind_pose.glb skin joints are not parent-first")
    for name in joint_names:
        seen: set[str] = set()
        current: str | None = name
        while current is not None:
            if current in seen:
                raise StaticDecisionError("bind_pose.glb skin hierarchy contains a cycle")
            seen.add(current)
            current = joint_parents[current]
        if roots[0] not in seen:
            raise StaticDecisionError("bind_pose.glb skin hierarchy is disconnected")

    accessors = document.get("accessors")
    serialized_vertex_count = 0
    if isinstance(accessors, list):
        for mesh in meshes:
            primitives = mesh.get("primitives", []) if isinstance(mesh, Mapping) else []
            if not isinstance(primitives, list):
                raise StaticDecisionError("bind_pose.glb mesh primitives are invalid")
            for primitive in primitives:
                attributes = (
                    primitive.get("attributes")
                    if isinstance(primitive, Mapping)
                    else None
                )
                accessor_index = (
                    attributes.get("POSITION")
                    if isinstance(attributes, Mapping)
                    else None
                )
                if (
                    not isinstance(accessor_index, int)
                    or isinstance(accessor_index, bool)
                    or not 0 <= accessor_index < len(accessors)
                    or not isinstance(accessors[accessor_index], Mapping)
                    or not isinstance(accessors[accessor_index].get("count"), int)
                    or isinstance(accessors[accessor_index].get("count"), bool)
                    or accessors[accessor_index]["count"] <= 0
                ):
                    raise StaticDecisionError(
                        "bind_pose.glb mesh has no authenticated POSITION count"
                    )
                serialized_vertex_count += accessors[accessor_index]["count"]
    if serialized_vertex_count <= 0:
        raise StaticDecisionError("bind_pose.glb serialized vertex inventory is empty")
    format_artifacts = list(STATIC_ARTIFACTS)
    if (static_root / "accessory_head_binding.png").exists():
        format_artifacts.append("accessory_head_binding.png")
    for name in format_artifacts:
        path = static_root / name
        if name.endswith(".png"):
            data, _ = common.read_file_snapshot(
                path,
                root=static_root,
                description=f"static artifact {name}",
                error_type=StaticDecisionError,
                require_mode=0o444,
            )
            try:
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                    if image.format != "PNG" or image.width <= 0 or image.height <= 0:
                        raise StaticDecisionError(f"static artifact is not a valid PNG: {name}")
            except (UnidentifiedImageError, OSError) as error:
                raise StaticDecisionError(f"static artifact is not PNG: {name}") from error
    try:
        hierarchy_data, _ = common.read_file_snapshot(
            static_root / "joint_hierarchy.txt",
            root=static_root,
            description="joint hierarchy",
            error_type=StaticDecisionError,
            require_mode=0o444,
        )
        hierarchy = hierarchy_data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise StaticDecisionError("joint hierarchy is not UTF-8") from error
    if not hierarchy.startswith("TokenRig static joint hierarchy\n"):
        raise StaticDecisionError("joint hierarchy does not use the TokenRig hierarchy format")
    lines = hierarchy.splitlines()
    if len(lines) < 5 or not lines[1].startswith("root=") or not lines[2].startswith(
        "bone_count="
    ):
        raise StaticDecisionError("joint hierarchy header is incomplete")
    try:
        declared_count = int(lines[2].split("=", 1)[1])
    except ValueError as error:
        raise StaticDecisionError("joint hierarchy bone_count is invalid") from error
    hierarchy_parents: dict[str, str | None] = {}
    for line in lines[4:]:
        fields = line.split("\t")
        if len(fields) != 4 or not fields[0]:
            raise StaticDecisionError("joint hierarchy row is malformed")
        name, parent = fields[0], fields[1]
        if name in hierarchy_parents:
            raise StaticDecisionError("joint hierarchy contains duplicate bones")
        hierarchy_parents[name] = None if parent == "-" else parent
    if (
        lines[1].split("=", 1)[1] != roots[0]
        or declared_count != len(joint_names)
        or list(hierarchy_parents) != joint_names
        or hierarchy_parents != joint_parents
    ):
        raise StaticDecisionError(
            "joint_hierarchy.txt does not match the bind_pose.glb skin hierarchy"
        )
    return {
        "joint_names": joint_names,
        "joint_parents": joint_parents,
        "root": roots[0],
        "bone_count": len(joint_names),
        "serialized_vertex_count": serialized_vertex_count,
    }


def _finite_number(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StaticDecisionError(f"{description} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise StaticDecisionError(f"{description} must be a finite number")
    return number


def _validate_complete_static_checks(
    checks: Any,
    *,
    owner_authentication: Mapping[str, Any],
    bind_skeleton: Mapping[str, Any],
) -> None:
    if not isinstance(checks, Mapping) or set(checks) != REQUIRED_STATIC_CHECKS:
        missing = sorted(REQUIRED_STATIC_CHECKS - set(checks or {})) if isinstance(checks, Mapping) else []
        raise StaticDecisionError(f"static checks are incomplete or unexpected: missing={missing}")
    if checks.get("automatic_static_checks") != "passed":
        raise StaticDecisionError("automatic static checks are not passed")
    for name in (
        "raw_triangle_contract",
        "input_pbr",
        "input_mesh",
        "input_surface",
        "exported_pbr",
        "glb_roundtrip",
        "renders",
    ):
        value = checks.get(name)
        if not isinstance(value, Mapping) or value.get("passed") is not True:
            raise StaticDecisionError(f"static check is not passed: {name}")
    mesh_contracts: dict[str, Mapping[str, Any]] = {}
    for name in (
        "source_mesh_contract",
        "tokenrig_mesh_contract_before_canonical_transform",
        "canonical_mesh_contract",
    ):
        value = checks.get(name)
        vertex_count = value.get("vertex_count") if isinstance(value, Mapping) else None
        if (
            not isinstance(value, Mapping)
            or not isinstance(vertex_count, int)
            or isinstance(vertex_count, bool)
            or vertex_count <= 0
        ):
            raise StaticDecisionError(f"static mesh contract is incomplete: {name}")
        mesh_contracts[name] = value
    if (
        mesh_contracts["tokenrig_mesh_contract_before_canonical_transform"].get(
            "vertex_count"
        )
        != mesh_contracts["canonical_mesh_contract"].get("vertex_count")
    ):
        raise StaticDecisionError("canonical transform changed the TokenRig vertex count")
    for name in (
        "source_removed_gltf_import_helpers",
        "tokenrig_removed_gltf_import_helpers",
        "removed_proven_orphans",
    ):
        if not isinstance(checks.get(name), list):
            raise StaticDecisionError(f"static removal evidence is malformed: {name}")
    if checks.get("removed_proven_orphans") != []:
        raise StaticDecisionError("static QA retained a proven orphan")
    hierarchy = checks["hierarchy"]
    if (
        not isinstance(hierarchy, Mapping)
        or hierarchy.get("connected") is not True
        or hierarchy.get("parent_first") is not True
        or not isinstance(hierarchy.get("bone_count"), int)
        or isinstance(hierarchy.get("bone_count"), bool)
        or hierarchy["bone_count"] < 19
        or hierarchy.get("bone_count") != bind_skeleton["bone_count"]
        or hierarchy.get("root") != bind_skeleton["root"]
    ):
        raise StaticDecisionError("static hierarchy proof is incomplete")
    semantic = checks["semantic_mapping"]
    semantic_bones = semantic.get("semantic_bones") if isinstance(semantic, Mapping) else None
    if not isinstance(semantic_bones, Mapping) or not REQUIRED_SEMANTIC_BONES.issubset(semantic_bones):
        raise StaticDecisionError("static semantic skeleton coverage is incomplete")
    for role in REQUIRED_SEMANTIC_BONES:
        bone = semantic_bones[role]
        valid = (
            isinstance(bone, str)
            and bool(bone.strip())
            or role == "spine"
            and isinstance(bone, list)
            and bool(bone)
            and all(isinstance(name, str) and name.strip() for name in bone)
        )
        if not valid:
            raise StaticDecisionError(f"static semantic bone mapping is invalid: {role}")
        names = bone if isinstance(bone, list) else [bone]
        if any(name not in bind_skeleton["joint_names"] for name in names):
            raise StaticDecisionError(
                f"static semantic bone is absent from bind_pose.glb: {role}"
            )
    parents = bind_skeleton["joint_parents"]

    def is_ancestor(ancestor: str, descendant: str) -> bool:
        current = parents.get(descendant)
        seen: set[str] = set()
        while current is not None:
            if current == ancestor:
                return True
            if current in seen:
                return False
            seen.add(current)
            current = parents.get(current)
        return False

    spine = list(semantic_bones["spine"])
    semantic_chains = [
        [semantic_bones["pelvis"], *spine, semantic_bones["head"]],
        [
            spine[-1],
            semantic_bones["left_upper_arm"],
            semantic_bones["left_forearm"],
            semantic_bones["left_hand"],
        ],
        [
            spine[-1],
            semantic_bones["right_upper_arm"],
            semantic_bones["right_forearm"],
            semantic_bones["right_hand"],
        ],
        [
            semantic_bones["pelvis"],
            semantic_bones["left_thigh"],
            semantic_bones["left_calf"],
            semantic_bones["left_foot"],
            semantic_bones["left_toe"],
        ],
        [
            semantic_bones["pelvis"],
            semantic_bones["right_thigh"],
            semantic_bones["right_calf"],
            semantic_bones["right_foot"],
            semantic_bones["right_toe"],
        ],
    ]
    if semantic_bones["pelvis"] != bind_skeleton["root"] or any(
        len(chain) != len(set(chain))
        or any(not is_ancestor(parent, child) for parent, child in zip(chain, chain[1:]))
        for chain in semantic_chains
    ):
        raise StaticDecisionError(
            "static semantic hierarchy does not match bind_pose.glb"
        )
    left = {
        semantic_bones[role]
        for role in REQUIRED_SEMANTIC_BONES
        if role.startswith("left_")
    }
    right = {
        semantic_bones[role]
        for role in REQUIRED_SEMANTIC_BONES
        if role.startswith("right_")
    }
    if left & right:
        raise StaticDecisionError("static left/right semantic bones overlap")
    weights = checks["weights"]
    if (
        not isinstance(weights, Mapping)
        or not isinstance(weights.get("vertex_count"), int)
        or isinstance(weights.get("vertex_count"), bool)
        or weights["vertex_count"] <= 0
        or not isinstance(weights.get("maximum_influences"), int)
        or isinstance(weights.get("maximum_influences"), bool)
        or not 1 <= weights["maximum_influences"] <= 4
    ):
        raise StaticDecisionError("static weights proof is incomplete")
    weight_error = _finite_number(
        weights.get("maximum_weight_sum_error"), "maximum weight-sum error"
    )
    weight_tolerance = _finite_number(
        weights.get("weight_sum_tolerance"), "weight-sum tolerance"
    )
    if weight_error < 0.0 or weight_tolerance < 0.0 or weight_error > weight_tolerance:
        raise StaticDecisionError("static weights proof is incomplete")
    seams = checks["seams"]
    if not isinstance(seams, Mapping):
        raise StaticDecisionError("static seam-weight proof is incomplete")
    seam_error = _finite_number(
        seams.get("maximum_weight_l1_error"), "maximum seam weight error"
    )
    seam_tolerance = _finite_number(
        seams.get("weight_l1_tolerance"), "seam weight tolerance"
    )
    if seam_error < 0.0 or seam_tolerance < 0.0 or seam_error > seam_tolerance:
        raise StaticDecisionError("static seam-weight proof is incomplete")
    bilateral = checks["bilateral_contamination"]
    if (
        not isinstance(bilateral, Mapping)
        or bilateral.get("contaminated_vertex_count") != 0
        or isinstance(bilateral.get("contaminated_vertex_count"), bool)
    ):
        raise StaticDecisionError("static bilateral-weight proof is incomplete")
    opposite_weight = _finite_number(
        bilateral.get("maximum_opposite_limb_weight"),
        "maximum opposite-limb weight",
    )
    opposite_tolerance = _finite_number(
        bilateral.get("tolerance"), "opposite-limb tolerance"
    )
    if (
        opposite_weight < 0.0
        or opposite_tolerance < 0.0
        or opposite_weight > opposite_tolerance
    ):
        raise StaticDecisionError("static bilateral-weight proof is incomplete")
    roundtrip = checks["glb_roundtrip"]
    for name in ("mesh", "surface", "pbr", "skin"):
        value = roundtrip.get(name) if isinstance(roundtrip, Mapping) else None
        if not isinstance(value, Mapping) or value.get("passed") is not True:
            raise StaticDecisionError(f"static GLB roundtrip proof is incomplete: {name}")
    full_rest = roundtrip.get("full_rest")
    if owner_authentication.get("sanitized_candidate") is not True:
        if (
            not isinstance(full_rest, Mapping)
            or full_rest.get("passed") is not True
            or not isinstance(full_rest.get("bone_count"), int)
            or isinstance(full_rest.get("bone_count"), bool)
            or full_rest["bone_count"] < 19
        ):
            raise StaticDecisionError("static GLB roundtrip full_rest proof is incomplete")
        rest_tolerance = _finite_number(
            full_rest.get("tolerance"), "full-rest tolerance"
        )
        rest_errors = (
            _finite_number(
                full_rest.get("maximum_matrix_element_error"), "full-rest matrix error"
            ),
            _finite_number(full_rest.get("maximum_tail_error_m"), "full-rest tail error"),
            _finite_number(
                full_rest.get("maximum_roll_error_radians"), "full-rest roll error"
            ),
        )
        if rest_tolerance < 0.0 or any(
            error < 0.0 or error > rest_tolerance for error in rest_errors
        ):
            raise StaticDecisionError("static GLB roundtrip full_rest proof is incomplete")
    inverse_bind = roundtrip.get("inverse_bind")
    if owner_authentication.get("sanitized_candidate") is not True:
        if (
            not isinstance(inverse_bind, Mapping)
            or inverse_bind.get("passed") is not True
            or inverse_bind.get("joint_order_unchanged") is not True
            or not isinstance(inverse_bind.get("joint_count"), int)
            or isinstance(inverse_bind.get("joint_count"), bool)
            or inverse_bind["joint_count"] < 19
        ):
            raise StaticDecisionError("static GLB roundtrip inverse_bind proof is incomplete")
        inverse_tolerance = _finite_number(
            inverse_bind.get("tolerance"), "inverse-bind tolerance"
        )
        inverse_error = _finite_number(
            inverse_bind.get("maximum_matrix_element_error"),
            "inverse-bind matrix error",
        )
        if (
            inverse_tolerance < 0.0
            or inverse_error < 0.0
            or inverse_error > inverse_tolerance
        ):
            raise StaticDecisionError("static GLB roundtrip inverse_bind proof is incomplete")


def _validate_branch_owner(
    *,
    contract: Mapping[str, Any],
    branch_id: str,
    branch_root: Path,
    static_qa: Mapping[str, Any],
) -> dict[str, Any]:
    from tools import blender_tokenrig_human_static_audit as static_audit

    schemas = {
        "direct": {"pixal_tokenrig_canary_v1", "pixal_tokenrig_recovery_v1"},
        "fitted_skeleton": {"pixal_tokenrig_fitted_skeleton_v1"},
        "sanitized_weights": {"pixal_tokenrig_sanitized_weights_v1"},
    }
    manifest_path = branch_root / "tokenrig_manifest.json"
    glb_path = branch_root / "tokenrig_transfer.glb"
    try:
        for path, description in (
            (manifest_path, "branch TokenRig manifest"),
            (glb_path, "branch TokenRig GLB"),
        ):
            common.file_record(
                path,
                root=branch_root,
                description=description,
                error_type=StaticDecisionError,
                require_mode=0o444,
            )
        manifest = common.load_json_mapping(
            manifest_path, "branch TokenRig manifest", StaticDecisionError
        )
        if manifest.get("schema") not in schemas.get(branch_id, set()):
            raise StaticDecisionError("branch TokenRig producer schema does not match branch")
        owner = static_audit.authenticate_task3_inputs(
            asset_id=contract["asset_id"],
            source_glb=Path(contract["source_lineage"]["pixal_pbr_glb"]["path"]),
            tokenrig_glb=glb_path,
            tokenrig_manifest=manifest_path,
        )
    except (static_audit.StaticAuditError, StaticDecisionError) as error:
        raise StaticDecisionError(f"branch TokenRig producer validation failed: {error}") from error
    if static_qa.get("authenticated") != owner:
        raise StaticDecisionError("static QA owner authentication is stale")
    return owner


def _read_glb_document_and_binary(path: Path, root: Path) -> tuple[dict[str, Any], bytes]:
    data, _ = common.read_file_snapshot(
        path,
        root=root,
        description="accessory bind_pose.glb",
        error_type=StaticDecisionError,
        require_mode=0o444,
    )
    if len(data) < 20 or data[:4] != b"glTF":
        raise StaticDecisionError("accessory bind pose is not GLB 2.0")
    version, declared = struct.unpack_from("<II", data, 4)
    if version != 2 or declared != len(data):
        raise StaticDecisionError("accessory bind pose GLB header is invalid")
    offset = 12
    json_chunk: bytes | None = None
    binary_chunk: bytes | None = None
    while offset < len(data):
        if offset + 8 > len(data):
            raise StaticDecisionError("accessory bind pose GLB chunk is truncated")
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        end = offset + length
        if end > len(data):
            raise StaticDecisionError("accessory bind pose GLB payload is truncated")
        chunk = data[offset:end]
        offset = end
        if kind == 0x4E4F534A:
            if json_chunk is not None:
                raise StaticDecisionError("accessory bind pose has duplicate JSON chunks")
            json_chunk = chunk
        elif kind == 0x004E4942:
            if binary_chunk is not None:
                raise StaticDecisionError("accessory bind pose has duplicate BIN chunks")
            binary_chunk = chunk
    if json_chunk is None or binary_chunk is None:
        raise StaticDecisionError("accessory bind pose must contain JSON and BIN chunks")
    try:
        document = json.loads(json_chunk.rstrip(b" \t\r\n\x00").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StaticDecisionError("accessory bind pose JSON is invalid") from error
    if not isinstance(document, dict):
        raise StaticDecisionError("accessory bind pose JSON root is invalid")
    return document, binary_chunk


def _decode_accessor_vec4(
    document: Mapping[str, Any],
    binary: bytes,
    accessor_index: Any,
    *,
    selected_indices: set[int] | None = None,
) -> tuple[int, dict[int, tuple[float, float, float, float]]]:
    accessors = document.get("accessors")
    views = document.get("bufferViews")
    if (
        not isinstance(accessor_index, int)
        or isinstance(accessor_index, bool)
        or not isinstance(accessors, list)
        or not 0 <= accessor_index < len(accessors)
        or not isinstance(views, list)
    ):
        raise StaticDecisionError("accessory GLB skin accessor index is invalid")
    accessor = accessors[accessor_index]
    if (
        not isinstance(accessor, Mapping)
        or accessor.get("type") != "VEC4"
        or "sparse" in accessor
        or not isinstance(accessor.get("bufferView"), int)
        or isinstance(accessor.get("bufferView"), bool)
        or not 0 <= accessor["bufferView"] < len(views)
        or not isinstance(accessor.get("count"), int)
        or isinstance(accessor.get("count"), bool)
        or accessor["count"] <= 0
    ):
        raise StaticDecisionError("accessory GLB skin accessor is incomplete")
    view = views[accessor["bufferView"]]
    if not isinstance(view, Mapping) or view.get("buffer", 0) != 0:
        raise StaticDecisionError("accessory GLB skin accessor is not in the GLB BIN chunk")
    formats = {
        5120: ("b", 1),
        5121: ("B", 1),
        5122: ("h", 2),
        5123: ("H", 2),
        5125: ("I", 4),
        5126: ("f", 4),
    }
    component_type = accessor.get("componentType")
    if component_type not in formats:
        raise StaticDecisionError("accessory GLB skin accessor component type is invalid")
    code, width = formats[component_type]
    element_width = width * 4
    stride = view.get("byteStride", element_width)
    if (
        not isinstance(stride, int)
        or isinstance(stride, bool)
        or stride < element_width
    ):
        raise StaticDecisionError("accessory GLB skin accessor stride is invalid")
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        raise StaticDecisionError("accessory GLB skin accessor offset is invalid")
    normalized = accessor.get("normalized", False)
    if not isinstance(normalized, bool):
        raise StaticDecisionError("accessory GLB accessor normalized flag is invalid")
    if selected_indices is None:
        indices = set(range(accessor["count"]))
    else:
        indices = set(selected_indices)
        if any(
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < accessor["count"]
            for index in indices
        ):
            raise StaticDecisionError("accessory GLB selected vertex index is invalid")
    result: dict[int, tuple[float, float, float, float]] = {}
    for index in sorted(indices):
        offset = start + index * stride
        if offset + element_width > len(binary):
            raise StaticDecisionError("accessory GLB skin accessor exceeds BIN payload")
        values = struct.unpack_from(f"<4{code}", binary, offset)
        if normalized and component_type != 5126:
            if component_type == 5121:
                values = tuple(value / 255.0 for value in values)
            elif component_type == 5123:
                values = tuple(value / 65535.0 for value in values)
            elif component_type == 5120:
                values = tuple(max(-1.0, value / 127.0) for value in values)
            elif component_type == 5122:
                values = tuple(max(-1.0, value / 32767.0) for value in values)
            else:
                raise StaticDecisionError(
                    "accessory GLB normalized skin accessor component is unsupported"
                )
        result[index] = tuple(float(value) for value in values)
    return accessor["count"], result


def _glb_vertex_weights(
    static_root: Path,
    *,
    requested_indices: set[int],
    bind_skeleton: Mapping[str, Any],
) -> dict[int, dict[str, float]]:
    document, binary = _read_glb_document_and_binary(
        static_root / "bind_pose.glb", static_root
    )
    meshes = document.get("meshes")
    skins = document.get("skins")
    nodes = document.get("nodes")
    if (
        not isinstance(meshes, list)
        or not isinstance(skins, list)
        or len(skins) != 1
        or not isinstance(nodes, list)
    ):
        raise StaticDecisionError("accessory bind GLB skin inventory is invalid")
    skin_joints = skins[0].get("joints") if isinstance(skins[0], Mapping) else None
    if not isinstance(skin_joints, list):
        raise StaticDecisionError("accessory bind GLB joint inventory is invalid")
    joint_names = [nodes[index].get("name") for index in skin_joints]
    if joint_names != bind_skeleton["joint_names"]:
        raise StaticDecisionError("accessory bind GLB joint names changed")
    actual: dict[int, dict[str, float]] = {}
    cursor = 0
    accessors = document.get("accessors")
    for mesh in meshes:
        primitives = mesh.get("primitives", []) if isinstance(mesh, Mapping) else []
        for primitive in primitives:
            attributes = primitive.get("attributes") if isinstance(primitive, Mapping) else None
            if not isinstance(attributes, Mapping):
                raise StaticDecisionError("accessory bind GLB primitive attributes are missing")
            position_index = attributes.get("POSITION")
            if (
                not isinstance(accessors, list)
                or not isinstance(position_index, int)
                or isinstance(position_index, bool)
                or not 0 <= position_index < len(accessors)
                or not isinstance(accessors[position_index], Mapping)
            ):
                raise StaticDecisionError("accessory bind GLB POSITION accessor is invalid")
            count = accessors[position_index].get("count")
            selected = [
                index for index in requested_indices if cursor <= index < cursor + count
            ] if isinstance(count, int) else []
            local_indices = {index - cursor for index in selected}
            joint_count, joints = _decode_accessor_vec4(
                document,
                binary,
                attributes.get("JOINTS_0"),
                selected_indices=local_indices,
            )
            weight_count, weights = _decode_accessor_vec4(
                document,
                binary,
                attributes.get("WEIGHTS_0"),
                selected_indices=local_indices,
            )
            if not isinstance(count, int) or count != joint_count or count != weight_count:
                raise StaticDecisionError("accessory bind GLB skin accessor counts differ")
            for global_index in selected:
                local = global_index - cursor
                by_bone: dict[str, float] = {}
                for joint_value, weight in zip(joints[local], weights[local]):
                    if not math.isfinite(weight) or weight < 0.0:
                        raise StaticDecisionError("accessory bind GLB weight is invalid")
                    if weight <= 0.0:
                        continue
                    joint = int(joint_value)
                    if float(joint) != joint_value or not 0 <= joint < len(joint_names):
                        raise StaticDecisionError("accessory bind GLB joint value is invalid")
                    name = joint_names[joint]
                    by_bone[name] = by_bone.get(name, 0.0) + weight
                actual[global_index] = by_bone
            cursor += count
    if set(actual) != requested_indices:
        raise StaticDecisionError("accessory vertex indices are absent from bind GLB")
    return actual


def _validate_accessory_head_binding(
    *,
    contract: Mapping[str, Any],
    static_root: Path,
    records: Mapping[str, Mapping[str, Any]],
    checks: Mapping[str, Any],
    bind_skeleton: Mapping[str, Any],
) -> None:
    case_id = contract["case"]["case_id"]
    if case_id not in ACCESSORY_CASES:
        return
    binding, binding_record = common.load_json_mapping_record(
        static_root / "accessory_head_binding.json",
        root=static_root,
        description="accessory Head binding evidence",
        error_type=StaticDecisionError,
        require_mode=0o444,
    )
    expected_binding = records["accessory_head_binding.json"]
    if any(
        binding_record[key] != expected_binding[key]
        for key in ("path", "sha256", "size_bytes", "mode")
    ):
        raise StaticDecisionError("accessory Head binding changed during parsing")
    common.reject_user_approval(binding, StaticDecisionError, "accessory Head binding")
    candidate = contract["source_lineage"].get("candidate_rgba")
    candidate_manifest = contract["source_lineage"].get("attribute_candidate_manifest")
    mapping = binding.get("pixel_to_vertex_mapping")
    allowed_bones = binding.get("allowed_bones")
    descendants = binding.get("head_descendant_bones")
    vertex_weights = binding.get("vertex_weights")
    semantic = checks.get("semantic_mapping")
    semantic_bones = semantic.get("semantic_bones") if isinstance(semantic, Mapping) else None
    proven_descendants = (
        semantic.get("ignored_proven_head_descendants")
        if isinstance(semantic, Mapping)
        else None
    )
    if (
        set(binding)
        != {
            "schema",
            "asset_id",
            "base_avatar_id",
            "case_id",
            "reviewed_candidate_rgba_sha256",
            "reviewed_candidate_manifest_sha256",
            "head_bone",
            "head_descendant_bones",
            "allowed_bones",
            "accessory_vertex_count",
            "minimum_head_descendant_weight",
            "maximum_non_head_descendant_weight",
            "vertex_weights",
            "pixel_to_vertex_mapping",
            "rigid_binding_passed",
            "overlay",
            "user_acceptance",
        }
        or binding.get("schema") != "route2_accessory_head_binding_v1"
        or binding.get("asset_id") != contract["asset_id"]
        or binding.get("base_avatar_id") != contract["base_avatar_id"]
        or binding.get("case_id") != case_id
        or not isinstance(candidate, Mapping)
        or not isinstance(candidate_manifest, Mapping)
        or binding.get("reviewed_candidate_rgba_sha256") != candidate.get("sha256")
        or binding.get("reviewed_candidate_manifest_sha256")
        != candidate_manifest.get("sha256")
        or not isinstance(semantic_bones, Mapping)
        or binding.get("head_bone") != semantic_bones.get("head")
        or not isinstance(proven_descendants, list)
        or not isinstance(descendants, list)
        or len(descendants) != len(set(descendants))
        or not set(descendants).issubset(set(proven_descendants))
        or not isinstance(allowed_bones, list)
        or not allowed_bones
        or len(allowed_bones) != len(set(allowed_bones))
        or binding.get("head_bone") not in allowed_bones
        or set(allowed_bones) != ({binding.get("head_bone")} | set(descendants))
        or not isinstance(mapping, Mapping)
        or mapping.get("method") != "reviewed_pixels_to_surface_vertices_v1"
        or binding.get("rigid_binding_passed") is not True
        or binding.get("user_acceptance") != "pending_user_review"
    ):
        raise StaticDecisionError("accessory Head/Head-descendant rigid binding proof is invalid")
    head = str(binding["head_bone"])
    joint_names = set(bind_skeleton["joint_names"])
    joint_parents = bind_skeleton["joint_parents"]
    if head not in joint_names or any(name not in joint_names for name in descendants):
        raise StaticDecisionError("accessory Head descendants are absent from bind_pose.glb")
    for descendant in descendants:
        current = joint_parents.get(descendant)
        seen: set[str] = set()
        while current is not None and current != head:
            if current in seen:
                raise StaticDecisionError("accessory Head descendant hierarchy is cyclic")
            seen.add(current)
            current = joint_parents.get(current)
        if current != head:
            raise StaticDecisionError(
                "accessory claimed Head descendant is not below the semantic Head"
            )
    count = binding.get("accessory_vertex_count")
    reviewed_pixels = mapping.get("reviewed_pixel_count")
    mapped_vertices = mapping.get("mapped_vertex_count")
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count <= 0
        or count > bind_skeleton["serialized_vertex_count"]
        or not isinstance(reviewed_pixels, int)
        or isinstance(reviewed_pixels, bool)
        or reviewed_pixels <= 0
        or not isinstance(mapped_vertices, int)
        or isinstance(mapped_vertices, bool)
        or mapped_vertices != count
    ):
        raise StaticDecisionError("accessory vertex/pixel counts must be positive integers")
    coverage = _finite_number(mapping.get("coverage"), "accessory mapping coverage")
    minimum_claim = _finite_number(
        binding.get("minimum_head_descendant_weight"),
        "minimum Head-chain accessory weight",
    )
    maximum_outside_claim = _finite_number(
        binding.get("maximum_non_head_descendant_weight"),
        "maximum non-Head-chain accessory weight",
    )
    if coverage != 1.0 or minimum_claim < 0.999999 or maximum_outside_claim > 1.0e-6:
        raise StaticDecisionError("accessory Head weight/coverage thresholds are not passed")
    if not isinstance(vertex_weights, list) or len(vertex_weights) != count:
        raise StaticDecisionError("accessory per-vertex weight evidence is incomplete")
    seen_vertices: set[int] = set()
    expected_vertex_weights: dict[int, dict[str, float]] = {}
    minimum_actual = 1.0
    maximum_outside_actual = 0.0
    allowed = set(allowed_bones)
    for vertex in vertex_weights:
        if not isinstance(vertex, Mapping) or set(vertex) != {"vertex_index", "weights"}:
            raise StaticDecisionError("accessory per-vertex weight record is malformed")
        index = vertex.get("vertex_index")
        weights = vertex.get("weights")
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < bind_skeleton["serialized_vertex_count"]
            or index in seen_vertices
            or not isinstance(weights, list)
            or not weights
        ):
            raise StaticDecisionError("accessory vertex index/weight coverage is invalid")
        seen_vertices.add(index)
        by_bone: dict[str, float] = {}
        for influence in weights:
            if not isinstance(influence, Mapping) or set(influence) != {"bone", "weight"}:
                raise StaticDecisionError("accessory vertex influence is malformed")
            bone = influence.get("bone")
            if not isinstance(bone, str) or not bone or bone in by_bone:
                raise StaticDecisionError("accessory vertex influence bone is invalid")
            weight = _finite_number(
                influence.get("weight"), "accessory vertex influence weight"
            )
            if weight <= 0.0:
                raise StaticDecisionError("accessory non-zero influence must be positive")
            by_bone[bone] = weight
        expected_vertex_weights[index] = by_bone
        if head not in by_bone:
            raise StaticDecisionError(
                "every accessory vertex must retain direct semantic Head coverage"
            )
        total = sum(by_bone.values())
        if abs(total - 1.0) > 1.0e-6:
            raise StaticDecisionError("accessory vertex weights are not normalized")
        outside = sum(weight for bone, weight in by_bone.items() if bone not in allowed)
        if outside > 0.0:
            raise StaticDecisionError(
                "accessory vertex has a non-zero weight outside Head/Head descendants"
            )
        chain = total - outside
        minimum_actual = min(minimum_actual, chain)
        maximum_outside_actual = max(maximum_outside_actual, outside)
    if (
        abs(minimum_actual - minimum_claim) > 1.0e-9
        or abs(maximum_outside_actual - maximum_outside_claim) > 1.0e-9
    ):
        raise StaticDecisionError("accessory aggregate weight claims do not match vertices")
    actual_vertex_weights = _glb_vertex_weights(
        static_root,
        requested_indices=set(expected_vertex_weights),
        bind_skeleton=bind_skeleton,
    )
    for index, expected_weights in expected_vertex_weights.items():
        actual_weights = actual_vertex_weights[index]
        if set(actual_weights) != set(expected_weights) or any(
            abs(actual_weights[bone] - weight) > 1.0e-6
            for bone, weight in expected_weights.items()
        ):
            raise StaticDecisionError(
                f"accessory GLB vertex weight does not match Head evidence: {index}"
            )
    overlay = binding.get("overlay")
    expected_overlay = records["accessory_head_binding.png"]
    if (
        not isinstance(overlay, Mapping)
        or overlay.get("filename") != "accessory_head_binding.png"
        or overlay.get("sha256") != expected_overlay["sha256"]
        or overlay.get("size_bytes") != expected_overlay["size_bytes"]
    ):
        raise StaticDecisionError("accessory Head binding overlay descriptor changed")


def _validated_static_snapshot_once(
    contract_path: Path, *, branch_id: str
) -> dict[str, Any]:
    contract, contract_path, output_root = _contract(contract_path)
    branch = _branch_root(contract, branch_id)
    static_root = common.require_real_directory(
        branch / STATIC_BUNDLE_DIRNAME,
        "static QA bundle",
        StaticDecisionError,
        mode=0o555,
    )
    accessory_case = contract["case"]["case_id"] in ACCESSORY_CASES
    expected_artifacts = set(STATIC_ARTIFACTS) | (
        set(ACCESSORY_ARTIFACTS) if accessory_case else set()
    )
    expected_names = {"static_qa.json", *expected_artifacts}
    actual_names = {path.name for path in static_root.iterdir()}
    if actual_names != expected_names:
        if accessory_case and not set(ACCESSORY_ARTIFACTS).issubset(actual_names):
            raise StaticDecisionError("accessory Head binding evidence is missing")
        raise StaticDecisionError("static QA bundle inventory is incomplete or unexpected")
    if any(not path.is_file() or path.is_symlink() for path in static_root.iterdir()):
        raise StaticDecisionError("static QA bundle contains a non-regular file or symlink")
    records = {
        name: _artifact_record(
            static_root / name,
            output_root,
            f"static artifact {name}",
        )
        for name in sorted(expected_names)
    }
    bind_skeleton = _validate_static_artifact_formats(static_root)
    static_qa_path = static_root / "static_qa.json"
    static_qa, static_qa_record = common.load_json_mapping_record(
        static_qa_path,
        root=output_root,
        description="static_qa.json",
        error_type=StaticDecisionError,
        require_mode=0o444,
    )
    if static_qa_record != records["static_qa.json"]:
        raise StaticDecisionError("static_qa.json changed between inventory and parsing")
    common.reject_user_approval(static_qa, StaticDecisionError, "static QA")
    if static_qa.get("schema") != STATIC_QA_SCHEMA:
        raise StaticDecisionError(f"static QA schema must be {STATIC_QA_SCHEMA}")
    if static_qa.get("asset_id") != contract["asset_id"]:
        raise StaticDecisionError("static QA asset_id does not match the instance contract")
    if static_qa.get("decision") != "automatic_static_checks_passed":
        raise StaticDecisionError("static QA automatic decision has not passed")
    if static_qa.get("agent_qa_status") != "pending_agent_visual_qa":
        raise StaticDecisionError("static QA must start at pending_agent_visual_qa")
    if static_qa.get("user_acceptance") != "pending_user_review":
        raise StaticDecisionError("static QA may only await user review")
    if static_qa.get("readiness_bundle_published") is not True:
        raise StaticDecisionError("static QA readiness bundle is not published")
    checks = static_qa.get("checks")
    owner = _validate_branch_owner(
        contract=contract,
        branch_id=branch_id,
        branch_root=branch,
        static_qa=static_qa,
    )
    _validate_complete_static_checks(
        checks,
        owner_authentication=owner,
        bind_skeleton=bind_skeleton,
    )
    axis = checks.get("axis_canonicalization")
    ground = checks.get("grounding")
    pbr = checks.get("exported_pbr")
    if (
        not isinstance(axis, Mapping)
        or axis.get("canonical_front") != "negative-y"
        or axis.get("canonical_up_vector") != [0.0, 0.0, 1.0]
        or axis.get("transform_count") != 1
    ):
        raise StaticDecisionError("static QA does not prove one-time FRONT -Y canonicalization")
    if (
        not isinstance(ground, Mapping)
        or ground.get("canonical_floor_z") != 0.0
        or ground.get("post_floor_z") != 0.0
        or ground.get("transform_count") != 1
    ):
        raise StaticDecisionError("static QA does not prove one-time grounding at Z=0")
    if not isinstance(pbr, Mapping) or pbr.get("passed") is not True:
        raise StaticDecisionError("static QA does not prove PBR preservation")
    artifacts = static_qa.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        raise StaticDecisionError("static QA artifact descriptors are incomplete or unexpected")
    for name in expected_artifacts:
        descriptor = artifacts[name]
        expected = records[name]
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {"filename", "sha256", "size_bytes"}
            or descriptor.get("filename") != name
            or descriptor.get("sha256") != expected["sha256"]
            or descriptor.get("size_bytes") != expected["size_bytes"]
        ):
            raise StaticDecisionError(f"static QA artifact descriptor changed: {name}")
    _validate_accessory_head_binding(
        contract=contract,
        static_root=static_root,
        records=records,
        checks=checks,
        bind_skeleton=bind_skeleton,
    )
    try:
        branch_descriptor = instance_contract.branch_descriptor(contract, branch_id)
    except instance_contract.InstanceContractError as error:
        raise StaticDecisionError(f"branch contract is invalid: {error}") from error
    contract_record = common.file_record(
        contract_path,
        root=output_root,
        description="instance contract",
        error_type=StaticDecisionError,
        require_mode=0o444,
    )
    return {
        "contract": contract_record,
        "asset_id": contract["asset_id"],
        "base_avatar_id": contract["base_avatar_id"],
        "case": dict(contract["case"]),
        "branch": {
            "branch_id": branch_id,
            "relative_root": branch_descriptor["relative_root"],
            "path": str(branch),
        },
        "static_bundle": str(static_root),
        "artifacts": records,
    }


def validated_static_snapshot(
    contract_path: Path, *, branch_id: str
) -> dict[str, Any]:
    return common.stable_mapping_snapshot(
        lambda: _validated_static_snapshot_once(contract_path, branch_id=branch_id),
        StaticDecisionError,
        "static QA bundle",
    )


def _validate_decision_fields(
    *,
    status: str,
    reviewer: str,
    notes: str,
    checks: Mapping[str, bool],
) -> tuple[str, str, str, dict[str, bool]]:
    if status not in {PASS_STATUS, REJECTED_STATUS}:
        raise StaticDecisionError("static agent decision status must be pending-user pass or rejected")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise StaticDecisionError("static agent reviewer must be non-empty")
    if not isinstance(notes, str) or not notes.strip():
        raise StaticDecisionError("static agent notes must be non-empty")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != set(STATIC_VISUAL_CHECKS)
        or any(type(value) is not bool for value in checks.values())
    ):
        raise StaticDecisionError("static agent visual checklist is incomplete or unexpected")
    normalized_checks = {name: checks[name] for name in STATIC_VISUAL_CHECKS}
    if status == PASS_STATUS and not all(normalized_checks.values()):
        raise StaticDecisionError("all visual checks must pass before static agent acceptance")
    probe = {
        "status": status,
        "reviewer": reviewer.strip(),
        "notes": notes.strip(),
        "checks": normalized_checks,
    }
    common.reject_user_approval(probe, StaticDecisionError, "static agent decision")
    return status, reviewer.strip(), notes.strip(), normalized_checks


def record_static_agent_visual_decision(
    contract_path: Path,
    *,
    branch_id: str,
    status: str,
    reviewer: str,
    notes: str,
    checks: Mapping[str, bool],
) -> Path:
    status, reviewer, notes, checks = _validate_decision_fields(
        status=status,
        reviewer=reviewer,
        notes=notes,
        checks=checks,
    )
    snapshot = validated_static_snapshot(contract_path, branch_id=branch_id)
    contract, _, _ = _contract(contract_path)
    destination = static_decision_path(contract, branch_id)
    payload = {
        "schema": SCHEMA,
        "asset_id": snapshot["asset_id"],
        "base_avatar_id": snapshot["base_avatar_id"],
        "case": snapshot["case"],
        "branch_id": branch_id,
        "status": status,
        "reviewer_kind": "agent",
        "reviewer": reviewer,
        "notes": notes,
        "checks": checks,
        "snapshot": snapshot,
        "user_acceptance": "pending_user_review",
    }
    common.reject_user_approval(payload, StaticDecisionError, "static agent decision")

    def validate_prelink() -> None:
        current = validated_static_snapshot(contract_path, branch_id=branch_id)
        if current != snapshot:
            raise StaticDecisionError(
                "static snapshot changed during agent-decision pre-publication validation"
            )

    return common.write_json_immutable_noreplace(
        destination,
        payload,
        StaticDecisionError,
        "static agent visual decision",
        prelink_validator=validate_prelink,
    )


def _validate_static_agent_visual_decision_once(
    contract_path: Path,
    *,
    branch_id: str,
    require_pass: bool = False,
) -> dict[str, Any]:
    contract, _, _ = _contract(contract_path)
    destination = static_decision_path(contract, branch_id)
    try:
        snapshot = validated_static_snapshot(contract_path, branch_id=branch_id)
    except StaticDecisionError as error:
        if os.path.lexists(destination):
            raise StaticDecisionError(
                f"static snapshot changed after the agent visual decision: {error}"
            ) from error
        raise
    branch_root = _branch_root(contract, branch_id)
    decision_path = common.require_contained_regular_file(
        destination,
        branch_root,
        "static agent visual decision",
        StaticDecisionError,
        mode=0o444,
    )
    payload, _ = common.load_json_mapping_record(
        decision_path,
        root=branch_root,
        description="static agent visual decision",
        error_type=StaticDecisionError,
        require_mode=0o444,
    )
    common.reject_user_approval(payload, StaticDecisionError, "static agent decision")
    if set(payload) != {
        "schema",
        "asset_id",
        "base_avatar_id",
        "case",
        "branch_id",
        "status",
        "reviewer_kind",
        "reviewer",
        "notes",
        "checks",
        "snapshot",
        "user_acceptance",
    }:
        raise StaticDecisionError("static agent decision fields are incomplete or unexpected")
    if (
        payload.get("schema") != SCHEMA
        or payload.get("asset_id") != contract["asset_id"]
        or payload.get("base_avatar_id") != contract["base_avatar_id"]
        or payload.get("case") != contract["case"]
        or payload.get("branch_id") != branch_id
        or payload.get("reviewer_kind") != "agent"
        or payload.get("user_acceptance") != "pending_user_review"
    ):
        raise StaticDecisionError("static agent decision identity or policy is stale")
    status, reviewer, notes, checks = _validate_decision_fields(
        status=payload.get("status"),
        reviewer=payload.get("reviewer"),
        notes=payload.get("notes"),
        checks=payload.get("checks"),
    )
    if payload.get("snapshot") != snapshot:
        raise StaticDecisionError("static snapshot changed after the agent visual decision")
    normalized = {
        **payload,
        "status": status,
        "reviewer": reviewer,
        "notes": notes,
        "checks": checks,
        "snapshot": snapshot,
    }
    if require_pass and status != PASS_STATUS:
        raise StaticDecisionNotAccepted(f"static agent visual decision is {status}")
    return normalized


def validate_static_agent_visual_decision(
    contract_path: Path,
    *,
    branch_id: str,
    require_pass: bool = False,
) -> dict[str, Any]:
    return common.stable_mapping_snapshot(
        lambda: _validate_static_agent_visual_decision_once(
            contract_path,
            branch_id=branch_id,
            require_pass=require_pass,
        ),
        StaticDecisionError,
        "static agent visual decision",
    )


def static_decision_record(
    contract_path: Path, *, branch_id: str, require_pass: bool = True
) -> dict[str, Any]:
    contract, _, output_root = _contract(contract_path)
    validate_static_agent_visual_decision(
        contract_path, branch_id=branch_id, require_pass=require_pass
    )
    path = static_decision_path(contract, branch_id)
    return common.file_record(
        path,
        root=output_root,
        description="static agent visual decision",
        error_type=StaticDecisionError,
        require_mode=0o444,
    )


__all__ = [
    "ACCESSORY_ARTIFACTS",
    "ACCESSORY_CASES",
    "DECISION_FILENAME",
    "PASS_STATUS",
    "REJECTED_STATUS",
    "SCHEMA",
    "REQUIRED_STATIC_CHECKS",
    "STATIC_ARTIFACTS",
    "STATIC_BUNDLE_DIRNAME",
    "STATIC_VISUAL_CHECKS",
    "StaticDecisionError",
    "StaticDecisionNotAccepted",
    "record_static_agent_visual_decision",
    "static_bundle_path",
    "static_decision_path",
    "static_decision_record",
    "validate_static_agent_visual_decision",
    "validated_static_snapshot",
]
