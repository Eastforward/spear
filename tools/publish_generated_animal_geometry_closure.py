#!/usr/bin/env python3
"""Publish one create-only, generic bounded Pixel3D geometry closure.

The v2 closure does not create a new human approval.  It replays the canonical
raw static approval and carries that exact decision through either a
byte-identical no-op or the oriented-sheet producer's exact-position
degenerate-index filter.  All geometry, topology, clay, PBR-container, and
downstream claims are independently read back before the manifest is created.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import audit_quadruped_i23d_geometry as geometry_audit
from tools import controlled_animal_derived_static_review_contract as review_contract
from tools import controlled_source_asset_schema as contracts
from tools import register_controlled_animal_source_assets as source_registry
from tools import repair_pixal_oriented_sheet_indices as oriented_repair

SCHEMA = "avengine_generated_animal_geometry_closure_v2"
LEGACY_SCHEMA = "avengine_generated_animal_geometry_closure_v1"
STATUS = "passed_bounded_oriented_geometry_closure"
STATE = "research_candidate"
VIEWS = ("front", "back", "side", "top", "quarter")
MUTATION_CLASS = "exact_position_degenerate_index_filter_or_byte_identical_noop_v1"
INHERITANCE_SCOPE = (
    "canonical_raw_static_approval_inherited_through_zero_area_index_filter_"
    "or_byte_identical_noop_v1"
)
AUTOMATIC_CHECK_FIELDS = frozenset(
    {
        "canonical_raw_static_decision_batch_replayed",
        "canonical_raw_static_approval_preserved_without_rewrite",
        "raw_static_review_and_reference_reauthenticated",
        "raw_pixel3d_glb_and_manifest_reauthenticated",
        "oriented_repair_manifest_strictly_replayed",
        "repair_limited_to_zero_area_index_filter_or_byte_identical_noop",
        "repaired_glb_exactly_bound",
        "independent_geometry_audit_v4_reauthenticated",
        "all_exact_position_directed_occurrences_reverse_paired",
        "clay_render_manifest_reauthenticated",
        "five_480x480_clay_pngs_reauthenticated",
        "clay_contact_sheet_reauthenticated",
        "single_mesh_single_primitive_pbr_container_readback",
        "no_skin_or_animation_in_repaired_glb",
        "no_new_human_approval_created",
        "target_native_tokenrig_is_the_only_authorized_next_execution",
        "formal_dataset_registration_not_authorized",
        "overall",
    }
)
DOWNSTREAM_FIELDS = frozenset(
    {
        "tokenrig_entry_authorized",
        "tokenrig_execution_performed",
        "quaternius_rig_swap_authorized",
        "animation_execution_performed",
        "ue_import_executed",
        "native_change_executed",
        "emitter_measurement_executed",
        "formal_dataset_registration_authorized",
    }
)
TOPOLOGY_FIELDS = frozenset(
    {
        "imported_vertices",
        "position_unique_vertices",
        "imported_triangles",
        "position_indexed_triangles",
        "degenerate_triangles_after_position_indexing",
        "boundary_edges",
        "manifold_two_face_edges",
        "two_face_orientation_mismatch_edges",
        "nonmanifold_edges_over_two_faces",
        "balanced_oriented_multicover_edges_over_two_faces",
        "unbalanced_edges_over_two_faces",
        "unpaired_oriented_edges",
        "unpaired_oriented_edge_occurrences",
        "paired_oriented_sheet_edge_occurrences",
        "maximum_edge_face_multiplicity",
        "nonmanifold_edge_ratio_per_triangle",
        "unpaired_oriented_edge_ratio_per_triangle",
        "topology_acceptance_semantics",
    }
)
PRIMARY_MIDLINE_FIELDS = frozenset(
    {
        "coordinate_frame",
        "central_longitudinal_percentiles",
        "torso_floor_fraction_of_robust_height",
        "surface_side_percentiles",
        "section_count",
        "selected_vertex_count",
        "side_slope_per_forward_unit",
        "yaw_degrees",
        "global_axis_yaw_degrees",
        "global_axis_semantics",
        "centerline_bend_p95_degrees",
        "centerline_bend_max_degrees",
        "centerline_lateral_rms_ratio",
        "centerline_lateral_peak_ratio",
        "centerline_curve_degree",
        "centerline_shape_semantics",
        "fit_r_squared",
    }
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
SPEAR_ROOT = Path(__file__).resolve().parents[1]
LOGICAL_TMP_ROOT = SPEAR_ROOT / "tmp"
PHYSICAL_TMP_ROOT = Path(
    "/data/datasets/avengine_workspaces/AVEngine/external/SPEAR/tmp"
)


class GeometryClosureError(ValueError):
    """Fail-closed geometry closure contract error."""


def _canonical(value: Any) -> str:
    try:
        return contracts.canonical_json(value)
    except contracts.ContractError as error:
        raise GeometryClosureError(str(error)) from error


def _hash_without(value: Mapping[str, Any], field: str) -> str:
    payload = {
        name: copy.deepcopy(item) for name, item in value.items() if name != field
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise GeometryClosureError(f"{label} must be a lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GeometryClosureError(f"{label} must be non-empty text")
    return value


def _identifier(value: Any, label: str) -> str:
    value = _text(value, label)
    if not IDENTIFIER_RE.fullmatch(value):
        raise GeometryClosureError(f"{label} is not a canonical identifier")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GeometryClosureError(f"{label} must be an object")
    return value


def _exact(
    value: Any,
    fields: set[str] | frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    value = _mapping(value, label)
    if set(value) != set(fields):
        missing = sorted(set(fields) - set(value))
        extra = sorted(set(value) - set(fields))
        raise GeometryClosureError(
            f"{label} fields are invalid: missing={missing} extra={extra}"
        )
    return value


def _nonnegative(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GeometryClosureError(f"{label} must be a nonnegative integer")
    return value


def _positive(value: Any, label: str) -> int:
    value = _nonnegative(value, label)
    if value == 0:
        raise GeometryClosureError(f"{label} must be positive")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GeometryClosureError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise GeometryClosureError(f"{label} must be a finite number")
    return converted


def _finite_json(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for name, item in value.items():
            if not isinstance(name, str):
                raise GeometryClosureError(f"{label} has a non-string key")
            _finite_json(item, f"{label}.{name}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_json(item, f"{label}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise GeometryClosureError(f"{label} contains a non-finite number")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise GeometryClosureError(f"{label} contains an unsupported value")


def _path_components(path: Path) -> list[Path]:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    result: list[Path] = []
    for part in absolute.parts[1:]:
        current = current / part
        result.append(current)
    return result


def _allowed_tmp_bridge(component: Path) -> bool:
    if component != LOGICAL_TMP_ROOT or not component.is_symlink():
        return False
    try:
        target = Path(os.readlink(component))
    except OSError:
        return False
    if not target.is_absolute():
        target = component.parent / target
    try:
        return os.path.samefile(target, PHYSICAL_TMP_ROOT)
    except OSError:
        return False


def _regular_file(path: Path, label: str) -> Path:
    absolute = Path(path).absolute()
    components = _path_components(absolute)
    for index, component in enumerate(components):
        if not component.is_symlink():
            continue
        if _allowed_tmp_bridge(component) and index != len(components) - 1:
            continue
        raise GeometryClosureError(
            f"{label} contains an unsafe symlink component: {component}"
        )
    if not absolute.is_file() or absolute.stat().st_size <= 0:
        raise GeometryClosureError(f"{label} is missing or empty: {absolute}")
    return absolute.resolve(strict=True)


def _external_file(path: Path, expected_sha256: str, label: str) -> Path:
    expected_sha256 = _sha256(expected_sha256, f"{label} expected SHA-256")
    path = _regular_file(path, label)
    if _sha256_file(path) != expected_sha256:
        raise GeometryClosureError(f"{label} changed from its external SHA-256")
    return path


def _new_file(path: Path) -> Path:
    absolute = Path(path).absolute()
    for component in _path_components(absolute):
        if component.is_symlink() and not _allowed_tmp_bridge(component):
            raise GeometryClosureError(
                f"output contains an unsafe symlink component: {component}"
            )
    if os.path.lexists(absolute):
        raise GeometryClosureError(f"refusing to replace output: {absolute}")
    if not absolute.parent.is_dir():
        raise GeometryClosureError("geometry closure output parent must exist")
    return absolute


def _record(path: Path) -> dict[str, Any]:
    path = _regular_file(path, "recorded geometry closure artifact")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _file_record(value: Any, label: str) -> dict[str, Any]:
    value = _exact(value, {"path", "sha256", "size_bytes"}, label)
    raw_path = _text(value["path"], f"{label}.path")
    if not Path(raw_path).is_absolute():
        raise GeometryClosureError(f"{label}.path must be absolute")
    _sha256(value["sha256"], f"{label}.sha256")
    _positive(value["size_bytes"], f"{label}.size_bytes")
    return copy.deepcopy(dict(value))


def _record_path(value: Any, label: str) -> Path:
    record = _file_record(value, label)
    path = _regular_file(Path(record["path"]), label)
    if (
        path.stat().st_size != record["size_bytes"]
        or _sha256_file(path) != record["sha256"]
    ):
        raise GeometryClosureError(f"{label} changed")
    return path


def _same_file(left: Path, right: Path, label: str) -> None:
    try:
        same = os.path.samefile(left, right)
    except OSError as error:
        raise GeometryClosureError(f"cannot compare {label} identity") from error
    if not same:
        raise GeometryClosureError(f"{label} points to an alternate path")


def _bind_expected(observed: Path, expected: Path | None, label: str) -> None:
    if expected is not None:
        _same_file(observed, _regular_file(expected, f"expected {label}"), label)


def _image(path: Path, label: str, expected_size: tuple[int, int]) -> None:
    try:
        with Image.open(path) as opened:
            opened.load()
            if opened.format != "PNG" or opened.size != expected_size:
                raise GeometryClosureError(
                    f"{label} must be a {expected_size[0]}x{expected_size[1]} PNG"
                )
    except OSError as error:
        raise GeometryClosureError(f"{label} is not a readable PNG") from error


def _image_reference(path: Path) -> None:
    try:
        with Image.open(path) as opened:
            opened.load()
            if opened.format not in {"PNG", "JPEG", "WEBP"}:
                raise GeometryClosureError(
                    "source reference has an unsupported image format"
                )
            if opened.width <= 0 or opened.height <= 0:
                raise GeometryClosureError("source reference is empty")
    except OSError as error:
        raise GeometryClosureError("source reference is unreadable") from error


def _validate_topology(value: Any) -> dict[str, Any]:
    topology = _exact(value, TOPOLOGY_FIELDS, "audit v4 topology")
    for name in (
        "imported_vertices",
        "position_unique_vertices",
        "imported_triangles",
        "position_indexed_triangles",
        "degenerate_triangles_after_position_indexing",
        "boundary_edges",
        "manifold_two_face_edges",
        "two_face_orientation_mismatch_edges",
        "nonmanifold_edges_over_two_faces",
        "balanced_oriented_multicover_edges_over_two_faces",
        "unbalanced_edges_over_two_faces",
        "unpaired_oriented_edges",
        "unpaired_oriented_edge_occurrences",
        "paired_oriented_sheet_edge_occurrences",
        "maximum_edge_face_multiplicity",
    ):
        _nonnegative(topology[name], f"audit v4 topology.{name}")
    for name in (
        "nonmanifold_edge_ratio_per_triangle",
        "unpaired_oriented_edge_ratio_per_triangle",
    ):
        _finite_number(topology[name], f"audit v4 topology.{name}")
    if (
        topology["topology_acceptance_semantics"]
        != (
            "every_exact_position_directed_edge_occurrence_has_one_"
            "oppositely_oriented_partner"
        )
        or topology["position_indexed_triangles"] <= 0
        or topology["degenerate_triangles_after_position_indexing"] != 0
        or topology["unpaired_oriented_edges"] != 0
        or topology["unpaired_oriented_edge_occurrences"] != 0
        or topology["unpaired_oriented_edge_ratio_per_triangle"] != 0.0
    ):
        raise GeometryClosureError(
            "audit v4 does not prove a nondegenerate reverse-paired topology"
        )
    return copy.deepcopy(dict(topology))


def _validate_audit(value: Any, repaired_glb: Path) -> dict[str, Any]:
    audit = _exact(
        value,
        {"schema", "created_at", "purpose", "records"},
        "independent geometry audit v4",
    )
    if (
        audit["schema"] != geometry_audit.SCHEMA
        or audit["purpose"]
        != "prebind_geometry_measurement_without_direction_inference"
        or not isinstance(audit["records"], list)
        or len(audit["records"]) != 1
    ):
        raise GeometryClosureError("independent geometry audit v4 contract changed")
    _text(audit["created_at"], "independent geometry audit created_at")
    record = _exact(
        audit["records"][0],
        {"label", "mesh", "topology", "torso_midline", "decision"},
        "independent geometry audit record",
    )
    _text(record["label"], "independent geometry audit label")
    mesh = _exact(
        record["mesh"],
        {"absolute_path", "sha256", "size_bytes"},
        "independent geometry audit mesh",
    )
    mesh_path = _record_path(
        {
            "path": mesh["absolute_path"],
            "sha256": mesh["sha256"],
            "size_bytes": mesh["size_bytes"],
        },
        "independent geometry audit mesh",
    )
    _same_file(mesh_path, repaired_glb, "independent geometry audit mesh")
    topology = _validate_topology(record["topology"])
    midline = _exact(
        record["torso_midline"],
        {
            "primary",
            "sensitivity_yaw_degrees",
            "sensitivity_global_axis_yaw_degrees",
            "sensitivity_centerline_bend_p95_degrees",
            "sensitivity_central_percentiles",
        },
        "independent geometry audit torso_midline",
    )
    primary = _exact(
        midline["primary"],
        PRIMARY_MIDLINE_FIELDS,
        "independent geometry audit primary midline",
    )
    _finite_json(midline, "independent geometry audit torso_midline")
    for name in (
        "sensitivity_yaw_degrees",
        "sensitivity_global_axis_yaw_degrees",
        "sensitivity_centerline_bend_p95_degrees",
        "sensitivity_central_percentiles",
    ):
        if not isinstance(midline[name], list) or len(midline[name]) != 3:
            raise GeometryClosureError(
                f"independent geometry audit {name} coverage changed"
            )
    decision = _exact(
        record["decision"],
        {
            "status",
            "rejection_reasons",
            "manual_review_reasons",
            "cardinal_orientation_inference",
            "global_yaw_is_a_rejection_criterion",
            "thresholds",
        },
        "independent geometry audit decision",
    )
    expected_decision = geometry_audit.decision(
        dict(topology),
        dict(primary),
    )
    if (
        dict(decision) != expected_decision
        or decision["status"]
        not in {
            "passed_automatic_geometry_measurements",
            "manual_source_geometry_review_required",
        }
        or decision["rejection_reasons"] != []
    ):
        raise GeometryClosureError(
            "independent geometry audit decision is not a valid non-rejection"
        )
    return {
        "label": record["label"],
        "topology": topology,
        "decision": copy.deepcopy(dict(decision)),
    }


def _validate_render_manifest(value: Any, repaired_glb: Path) -> dict[str, Any]:
    render = _exact(
        value,
        {
            "input",
            "bbox_min",
            "bbox_max",
            "extent",
            "front_axis",
            "views",
            "resolution",
            "samples",
            "material_preview",
            "lighting",
        },
        "clay render manifest",
    )
    input_path = _regular_file(Path(_text(render["input"], "clay input")), "clay input")
    _same_file(input_path, repaired_glb, "clay render input")
    if render["front_axis"] not in review_contract.FRONT_AXES:
        raise GeometryClosureError("clay render front axis changed")
    if render["resolution"] != [480, 480]:
        raise GeometryClosureError("clay render resolution must be 480x480")
    samples = _positive(render["samples"], "clay render samples")
    if samples > 256:
        raise GeometryClosureError("clay render sample count is out of range")
    for name in ("bbox_min", "bbox_max", "extent"):
        vector = render[name]
        if not isinstance(vector, list) or len(vector) != 3:
            raise GeometryClosureError(f"clay render {name} must be a vector")
        for index, item in enumerate(vector):
            _finite_number(item, f"clay render {name}[{index}]")
    for minimum, maximum, extent in zip(
        render["bbox_min"], render["bbox_max"], render["extent"]
    ):
        if float(maximum) <= float(minimum) or not math.isclose(
            float(maximum) - float(minimum),
            float(extent),
            rel_tol=1.0e-9,
            abs_tol=1.0e-9,
        ):
            raise GeometryClosureError("clay render bounding box is inconsistent")
    views = _exact(render["views"], set(VIEWS), "clay render views")
    for name, location in views.items():
        if not isinstance(location, list) or len(location) != 3:
            raise GeometryClosureError(f"clay render {name} camera is invalid")
        for index, item in enumerate(location):
            _finite_number(item, f"clay render {name} camera[{index}]")
    material = _exact(
        render["material_preview"],
        {
            "mode",
            "principled_nodes_changed",
            "metallic_links_removed",
            "roughness_links_removed",
        },
        "clay material preview",
    )
    if material != {
        "mode": "neutral_clay_geometry_qa_v1",
        "principled_nodes_changed": 0,
        "metallic_links_removed": 0,
        "roughness_links_removed": 0,
    }:
        raise GeometryClosureError("clay material preview contract changed")
    lighting = _exact(
        render["lighting"],
        {"area_light_scale", "world_strength", "exposure"},
        "clay lighting",
    )
    if dict(lighting) != {
        "area_light_scale": 1.0,
        "world_strength": 0.5,
        "exposure": 0.0,
    }:
        raise GeometryClosureError("clay lighting contract changed")
    return {
        "front_axis": render["front_axis"],
        "resolution": [480, 480],
        "material_mode": material["mode"],
    }


def _glb_readback(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    try:
        document, binary = oriented_repair._parse_glb(data, "repaired GLB")
        authenticated = oriented_repair._authenticate_primitive(document, binary)
    except oriented_repair.OrientedSheetRepairError as error:
        raise GeometryClosureError(
            f"repaired GLB failed strict container readback: {error}"
        ) from error
    skins = document.get("skins", [])
    animations = document.get("animations", [])
    materials = document.get("materials", [])
    textures = document.get("textures", [])
    images = document.get("images", [])
    if (
        not isinstance(skins, list)
        or not isinstance(animations, list)
        or not isinstance(materials, list)
        or not isinstance(textures, list)
        or not isinstance(images, list)
        or len(skins) != 0
        or len(animations) != 0
        or len(materials) != 1
        or len(textures) <= 0
        or len(images) <= 0
    ):
        raise GeometryClosureError(
            "repaired GLB must be one unskinned, unanimated PBR primitive"
        )
    material = _mapping(
        materials[authenticated["material_index"]],
        "repaired GLB material",
    )
    pbr = _mapping(
        material.get("pbrMetallicRoughness"),
        "repaired GLB pbrMetallicRoughness",
    )
    for binding_name in ("baseColorTexture", "metallicRoughnessTexture"):
        binding = _mapping(
            pbr.get(binding_name),
            f"repaired GLB {binding_name}",
        )
        texture_index = binding.get("index")
        if (
            isinstance(texture_index, bool)
            or not isinstance(texture_index, int)
            or not 0 <= texture_index < len(textures)
        ):
            raise GeometryClosureError(f"repaired GLB {binding_name} index is invalid")
        texture = _mapping(
            textures[texture_index],
            f"repaired GLB {binding_name} texture",
        )
        try:
            oriented_repair._texture_image_source(
                document,
                texture,
                len(images),
                f"repaired GLB {binding_name}",
            )
        except oriented_repair.OrientedSheetRepairError as error:
            raise GeometryClosureError(
                f"repaired GLB {binding_name} image binding is invalid"
            ) from error
    return {
        "mesh_count": 1,
        "primitive_count": 1,
        "material_count": len(materials),
        "texture_count": len(textures),
        "image_count": len(images),
        "skin_count": len(skins),
        "animation_count": len(animations),
        "pbr_containers_present": True,
        "index_component_type": authenticated["index_layout"]["component_type"],
    }


def _strict_repair_readback(
    *,
    repair: Mapping[str, Any],
    instance_id: str,
    raw_glb: Path,
    pixal_manifest: Path,
    static_decision: Path,
    repaired_glb: Path,
) -> None:
    """Recompute the oriented producer's bounded mutation from actual bytes."""

    try:
        raw_data = raw_glb.read_bytes()
        source_record = _record(raw_glb)
        pixal_payload = contracts.load_json(pixal_manifest)
        decision_payload = contracts.load_json(static_decision)
        oriented_repair._authenticate_pixal_manifest(
            pixal_payload,
            instance_id=instance_id,
            source_path=raw_glb,
            source_record=source_record,
        )
        controlled_request = _mapping(
            pixal_payload.get("controlled_request"),
            "Pixel3D manifest controlled request",
        )
        oriented_repair._authenticate_static_decision(
            decision_payload,
            instance_id=instance_id,
            source_record=source_record,
            controlled_request=controlled_request,
        )
        source_document, source_binary = oriented_repair._parse_glb(
            raw_data,
            "source Pixel3D GLB",
        )
        source = oriented_repair._authenticate_primitive(
            source_document,
            source_binary,
        )
        readback = oriented_repair._readback(
            source_document=source_document,
            source_binary=source_binary,
            output_data=repaired_glb.read_bytes(),
            source=source,
        )
    except (
        contracts.ContractError,
        OSError,
        oriented_repair.OrientedSheetRepairError,
    ) as error:
        raise GeometryClosureError(
            f"oriented repair byte replay failed: {error}"
        ) from error

    retained = source["retained_triangles"]
    removed = source["removed_ordinals"]
    source_triangles = source["source_triangles"]
    index = source["index_layout"]
    expected_source_contract = {
        "glb_version": 2,
        "mesh_count": 1,
        "primitive_count": 1,
        "primitive_mode": 4,
        "position_accessor_index": source["attributes"]["POSITION"],
        "normal_accessor_index": source["attributes"]["NORMAL"],
        "texcoord_0_accessor_index": source["attributes"]["TEXCOORD_0"],
        "index_accessor_index": index["accessor_index"],
        "index_component_type": index["component_type"],
        "source_index_count": index["count"],
        "source_triangle_count": len(source_triangles),
        "source_exact_position_degenerate_triangle_count": len(removed),
        "source_nondegenerate_triangle_count": len(retained),
    }
    expected_mutation = {
        "method": oriented_repair.MUTATION_METHOD,
        "canonical_signed_zero": True,
        "authorized_json_changes": (
            [] if not removed else [f"accessors[{index['accessor_index']}].count"]
        ),
        "authorized_bin_byte_range": {
            "offset": index["start"],
            "source_length": index["count"] * index["width"],
            "output_used_length": len(retained) * 3 * index["width"],
        },
        "external_geometry_inputs": [],
        "external_skeleton_inputs": [],
        "external_weight_inputs": [],
        "external_material_inputs": [],
        "external_texture_inputs": [],
        "animation_inputs": [],
        "removed_triangle_count": len(removed),
        "removed_triangle_ordinals_sha256": (oriented_repair._ordinal_sha256(removed)),
        "source_nondegenerate_index_triplets_sha256": (
            oriented_repair._triplet_sha256(retained)
        ),
        "output_index_triplets_sha256": (oriented_repair._triplet_sha256(retained)),
        "output_index_count": len(retained) * 3,
        "output_byte_identical_to_source": not removed,
    }
    if (
        repair["source_contract"] != expected_source_contract
        or repair["mutation"] != expected_mutation
        or repair["topology"] != readback["output_topology"]
        or repair["readback"] != readback
        or repair["output"] != _record(repaired_glb)
        or (not removed and raw_data != repaired_glb.read_bytes())
    ):
        raise GeometryClosureError(
            "oriented repair manifest does not replay from the actual GLB bytes"
        )


def validate_manifest(value: Any) -> dict[str, Any]:
    """Validate the exact v2 JSON contract without filesystem access."""

    manifest = _exact(
        value,
        {
            "schema",
            "created_at",
            "status",
            "state_classification",
            "formal_dataset_registration_authorized",
            "identity",
            "source_authorities",
            "bounded_repair",
            "independent_audit",
            "clay_readback",
            "container_readback",
            "inherited_static_judgment",
            "downstream",
            "automatic_checks",
            "manifest_sha256",
        },
        "generated animal geometry closure v2",
    )
    _finite_json(manifest, "generated animal geometry closure v2")
    if (
        manifest["schema"] != SCHEMA
        or manifest["status"] != STATUS
        or manifest["state_classification"] != STATE
        or manifest["formal_dataset_registration_authorized"] is not False
    ):
        raise GeometryClosureError("geometry closure v2 authority boundary changed")
    _text(manifest["created_at"], "geometry closure created_at")
    identity = _exact(manifest["identity"], {"instance_id"}, "closure identity")
    _identifier(identity["instance_id"], "closure instance_id")
    authorities = _exact(
        manifest["source_authorities"],
        {
            "raw_pixal_glb",
            "pixal_manifest",
            "source_reference",
            "raw_static_decision_batch",
            "raw_static_decision",
        },
        "closure source authorities",
    )
    for name in ("raw_pixal_glb", "pixal_manifest", "source_reference"):
        _file_record(authorities[name], f"closure {name}")
    batch = _exact(
        authorities["raw_static_decision_batch"],
        {"file", "decision_batch_sha256"},
        "closure raw static decision batch",
    )
    _file_record(batch["file"], "closure raw static decision batch file")
    _sha256(
        batch["decision_batch_sha256"],
        "closure raw static decision batch internal hash",
    )
    decision = _exact(
        authorities["raw_static_decision"],
        {
            "file",
            "static_review",
            "decision_sha256",
            "decision",
            "state_classification",
            "formal_dataset_registration_authorized",
            "next_gate",
            "checks",
        },
        "closure raw static decision",
    )
    _file_record(decision["file"], "closure raw static decision file")
    _file_record(decision["static_review"], "closure raw static review")
    _sha256(decision["decision_sha256"], "closure raw static decision hash")
    checks = _exact(
        decision["checks"],
        review_contract.RAW_STATIC_CHECK_FIELDS,
        "closure raw static checks",
    )
    if (
        decision["decision"] != "approved_for_lod_and_binding"
        or decision["state_classification"] != STATE
        or decision["formal_dataset_registration_authorized"] is not False
        or decision["next_gate"] != "lod_then_species_rig_binding"
        or any(item is not True for item in checks.values())
    ):
        raise GeometryClosureError(
            "geometry closure requires the canonical all-true raw static approval"
        )
    bounded = _exact(
        manifest["bounded_repair"],
        {
            "manifest",
            "implementation_contract",
            "output",
            "mutation_class",
            "removed_exact_position_degenerate_triangle_count",
            "output_byte_identical_to_raw",
        },
        "closure bounded repair",
    )
    _file_record(bounded["manifest"], "closure repair manifest")
    _file_record(bounded["output"], "closure repaired GLB")
    removed = _nonnegative(
        bounded["removed_exact_position_degenerate_triangle_count"],
        "closure removed exact-position degenerate triangles",
    )
    if (
        bounded["implementation_contract"] != oriented_repair.IMPLEMENTATION_CONTRACT
        or bounded["mutation_class"] != MUTATION_CLASS
        or bounded["output_byte_identical_to_raw"] is not (removed == 0)
    ):
        raise GeometryClosureError("closure repair scope exceeded the bounded contract")
    independent = _exact(
        manifest["independent_audit"],
        {
            "file",
            "schema",
            "record_label",
            "topology_acceptance_semantics",
            "degenerate_triangles_after_position_indexing",
            "unpaired_oriented_edges",
            "unpaired_oriented_edge_occurrences",
            "decision_status",
        },
        "closure independent audit",
    )
    _file_record(independent["file"], "closure independent audit file")
    if (
        independent["schema"] != geometry_audit.SCHEMA
        or independent["topology_acceptance_semantics"]
        != (
            "every_exact_position_directed_edge_occurrence_has_one_"
            "oppositely_oriented_partner"
        )
        or independent["degenerate_triangles_after_position_indexing"] != 0
        or independent["unpaired_oriented_edges"] != 0
        or independent["unpaired_oriented_edge_occurrences"] != 0
        or independent["decision_status"]
        not in {
            "passed_automatic_geometry_measurements",
            "manual_source_geometry_review_required",
        }
    ):
        raise GeometryClosureError("closure independent audit summary changed")
    clay = _exact(
        manifest["clay_readback"],
        {
            "render_manifest",
            "front_axis",
            "resolution",
            "material_mode",
            "views",
            "contact_sheet",
            "human_approval_claimed",
        },
        "closure clay readback",
    )
    _file_record(clay["render_manifest"], "closure clay render manifest")
    clay_views = _exact(clay["views"], set(VIEWS), "closure clay views")
    for name, record in clay_views.items():
        _file_record(record, f"closure clay {name}")
    _file_record(clay["contact_sheet"], "closure clay contact sheet")
    if (
        clay["front_axis"] not in review_contract.FRONT_AXES
        or clay["resolution"] != [480, 480]
        or clay["material_mode"] != "neutral_clay_geometry_qa_v1"
        or clay["human_approval_claimed"] is not False
    ):
        raise GeometryClosureError("closure clay readback claim changed")
    container = _exact(
        manifest["container_readback"],
        {
            "mesh_count",
            "primitive_count",
            "material_count",
            "texture_count",
            "image_count",
            "skin_count",
            "animation_count",
            "pbr_containers_present",
            "index_component_type",
        },
        "closure GLB container readback",
    )
    if (
        container["mesh_count"] != 1
        or container["primitive_count"] != 1
        or container["material_count"] != 1
        or _positive(container["texture_count"], "closure texture_count") <= 0
        or _positive(container["image_count"], "closure image_count") <= 0
        or container["skin_count"] != 0
        or container["animation_count"] != 0
        or container["pbr_containers_present"] is not True
        or container["index_component_type"] not in {5121, 5123, 5125}
    ):
        raise GeometryClosureError("closure GLB container readback changed")
    inherited = _exact(
        manifest["inherited_static_judgment"],
        {
            "authority",
            "decision_sha256",
            "checks",
            "inheritance_scope",
            "new_human_approval_created",
            "clay_render_human_approval_claimed",
        },
        "closure inherited static judgment",
    )
    inherited_checks = _exact(
        inherited["checks"],
        review_contract.RAW_STATIC_CHECK_FIELDS,
        "closure inherited static checks",
    )
    if (
        inherited["authority"] != "canonical_raw_static_approval_v1"
        or inherited["decision_sha256"] != decision["decision_sha256"]
        or dict(inherited_checks) != dict(checks)
        or inherited["inheritance_scope"] != INHERITANCE_SCOPE
        or inherited["new_human_approval_created"] is not False
        or inherited["clay_render_human_approval_claimed"] is not False
    ):
        raise GeometryClosureError("closure fabricated a new human judgment")
    downstream = _exact(
        manifest["downstream"],
        DOWNSTREAM_FIELDS,
        "closure downstream boundary",
    )
    if downstream != {
        "tokenrig_entry_authorized": True,
        "tokenrig_execution_performed": False,
        "quaternius_rig_swap_authorized": False,
        "animation_execution_performed": False,
        "ue_import_executed": False,
        "native_change_executed": False,
        "emitter_measurement_executed": False,
        "formal_dataset_registration_authorized": False,
    }:
        raise GeometryClosureError("closure downstream execution boundary changed")
    automatic = _exact(
        manifest["automatic_checks"],
        AUTOMATIC_CHECK_FIELDS,
        "closure automatic checks",
    )
    if any(item is not True for item in automatic.values()):
        raise GeometryClosureError("all geometry closure automatic checks must pass")
    _sha256(manifest["manifest_sha256"], "geometry closure manifest hash")
    if manifest["manifest_sha256"] != _hash_without(manifest, "manifest_sha256"):
        raise GeometryClosureError("geometry closure embedded self-hash changed")
    return copy.deepcopy(dict(manifest))


def _replay_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_instance_id: str | None = None,
    expected_raw_pixal_glb: Path | None = None,
    expected_pixal_manifest: Path | None = None,
    expected_source_reference: Path | None = None,
    expected_raw_static_decision_batch: Path | None = None,
    expected_raw_static_decision: Path | None = None,
    expected_repair_manifest: Path | None = None,
    expected_repaired_glb: Path | None = None,
    expected_geometry_audit: Path | None = None,
) -> dict[str, Any]:
    manifest = validate_manifest(manifest)
    instance_id = manifest["identity"]["instance_id"]
    if expected_instance_id is not None and instance_id != expected_instance_id:
        raise GeometryClosureError("geometry closure instance identity changed")
    authorities = manifest["source_authorities"]
    raw_glb = _record_path(authorities["raw_pixal_glb"], "closure raw Pixel3D GLB")
    pixal_manifest = _record_path(
        authorities["pixal_manifest"],
        "closure Pixel3D manifest",
    )
    source_reference = _record_path(
        authorities["source_reference"],
        "closure source reference",
    )
    _image_reference(source_reference)
    batch_authority = authorities["raw_static_decision_batch"]
    decision_batch = _record_path(
        batch_authority["file"],
        "closure raw static decision batch",
    )
    decision_authority = authorities["raw_static_decision"]
    decision_path = _record_path(
        decision_authority["file"],
        "closure raw static decision",
    )
    static_review_path = _record_path(
        decision_authority["static_review"],
        "closure raw static review",
    )
    for observed, expected, label in (
        (raw_glb, expected_raw_pixal_glb, "closure raw Pixel3D GLB"),
        (pixal_manifest, expected_pixal_manifest, "closure Pixel3D manifest"),
        (source_reference, expected_source_reference, "closure source reference"),
        (
            decision_batch,
            expected_raw_static_decision_batch,
            "closure raw static decision batch",
        ),
        (
            decision_path,
            expected_raw_static_decision,
            "closure raw static decision",
        ),
    ):
        _bind_expected(observed, expected, label)
    try:
        authenticated_batch, batch_payload, decisions = (
            source_registry.load_decision_batch(decision_batch)
        )
    except contracts.ContractError as error:
        raise GeometryClosureError(
            f"canonical raw static decision batch is invalid: {error}"
        ) from error
    selected = decisions.get(instance_id)
    selected_payload = (
        selected.get("payload") if isinstance(selected, Mapping) else None
    )
    selected_review = (
        selected.get("static_review") if isinstance(selected, Mapping) else None
    )
    if (
        authenticated_batch != decision_batch
        or batch_payload.get("decision_batch_sha256")
        != batch_authority["decision_batch_sha256"]
        or not isinstance(selected_payload, Mapping)
        or not isinstance(selected_review, Mapping)
        or selected.get("path") != decision_path
        or selected_review.get("path") != static_review_path
        or selected_payload.get("decision_sha256")
        != decision_authority["decision_sha256"]
        or any(
            selected_payload.get(name) != decision_authority[name]
            for name in (
                "decision",
                "state_classification",
                "formal_dataset_registration_authorized",
                "next_gate",
                "checks",
            )
        )
    ):
        raise GeometryClosureError(
            "closure raw static approval is not the selected canonical record"
        )
    static_review = selected_review.get("payload")
    if not isinstance(static_review, Mapping):
        raise GeometryClosureError("canonical raw static review is missing")
    reference_descriptor = static_review.get("reference_rgba")
    if not isinstance(reference_descriptor, Mapping):
        raise GeometryClosureError(
            "canonical raw static review lacks its source reference"
        )
    reference_value = Path(
        _text(reference_descriptor.get("path"), "raw static source reference path")
    )
    if reference_value.is_absolute():
        canonical_reference = _record_path(
            reference_descriptor,
            "canonical raw static source reference",
        )
    else:
        review_batch_path = _regular_file(
            Path(batch_payload["static_review_batch"]["path"]),
            "canonical raw static review batch",
        )
        canonical_reference = _record_path(
            {
                **dict(reference_descriptor),
                "path": str(review_batch_path.parent / reference_value),
            },
            "canonical raw static source reference",
        )
    _same_file(
        canonical_reference,
        source_reference,
        "canonical raw static source reference",
    )
    review_output = _mapping(
        static_review.get("pixal_output"),
        "canonical raw static review Pixel3D output",
    )
    if (
        review_output.get("sha256") != authorities["raw_pixal_glb"]["sha256"]
        or review_output.get("size_bytes") != authorities["raw_pixal_glb"]["size_bytes"]
    ):
        raise GeometryClosureError(
            "canonical raw static review is disconnected from the raw Pixel3D GLB"
        )
    bounded = manifest["bounded_repair"]
    repair_path = _record_path(
        bounded["manifest"],
        "closure oriented repair manifest",
    )
    repaired = _record_path(bounded["output"], "closure repaired GLB")
    _bind_expected(repair_path, expected_repair_manifest, "closure repair manifest")
    _bind_expected(repaired, expected_repaired_glb, "closure repaired GLB")
    try:
        repair = oriented_repair.validate_repair_manifest(
            contracts.load_json(repair_path)
        )
    except (
        contracts.ContractError,
        oriented_repair.OrientedSheetRepairError,
    ) as error:
        raise GeometryClosureError(
            f"oriented repair manifest failed strict replay: {error}"
        ) from error
    _strict_repair_readback(
        repair=repair,
        instance_id=instance_id,
        raw_glb=raw_glb,
        pixal_manifest=pixal_manifest,
        static_decision=decision_path,
        repaired_glb=repaired,
    )
    repair_lineage = repair["lineage"]
    for descriptor, expected, label in (
        (repair_lineage["pixal_source"], raw_glb, "repair raw Pixel3D GLB"),
        (repair_lineage["pixal_manifest"], pixal_manifest, "repair Pixel3D manifest"),
        (
            repair_lineage["static_decision_batch"],
            decision_batch,
            "repair raw static decision batch",
        ),
        (
            repair_lineage["static_decision"],
            decision_path,
            "repair raw static decision",
        ),
        (repair["output"], repaired, "repair output GLB"),
    ):
        _same_file(_record_path(descriptor, label), expected, label)
    if (
        bounded["removed_exact_position_degenerate_triangle_count"]
        != repair["mutation"]["removed_triangle_count"]
        or bounded["output_byte_identical_to_raw"]
        is not repair["mutation"]["output_byte_identical_to_source"]
    ):
        raise GeometryClosureError("closure repair summary contradicts its manifest")
    if (
        bounded["output_byte_identical_to_raw"]
        and authorities["raw_pixal_glb"]["sha256"] != bounded["output"]["sha256"]
    ):
        raise GeometryClosureError("closure no-op output is not byte-identical")
    audit_path = _record_path(
        manifest["independent_audit"]["file"],
        "closure independent geometry audit",
    )
    _bind_expected(audit_path, expected_geometry_audit, "closure geometry audit")
    audit = _validate_audit(contracts.load_json(audit_path), repaired)
    audit_summary = manifest["independent_audit"]
    topology = audit["topology"]
    if (
        audit_summary["record_label"] != audit["label"]
        or audit_summary["topology_acceptance_semantics"]
        != topology["topology_acceptance_semantics"]
        or audit_summary["degenerate_triangles_after_position_indexing"]
        != topology["degenerate_triangles_after_position_indexing"]
        or audit_summary["unpaired_oriented_edges"]
        != topology["unpaired_oriented_edges"]
        or audit_summary["unpaired_oriented_edge_occurrences"]
        != topology["unpaired_oriented_edge_occurrences"]
        or audit_summary["decision_status"] != audit["decision"]["status"]
    ):
        raise GeometryClosureError("closure audit summary contradicts audit v4")
    clay = manifest["clay_readback"]
    render_path = _record_path(
        clay["render_manifest"],
        "closure clay render manifest",
    )
    render = _validate_render_manifest(
        contracts.load_json(render_path),
        repaired,
    )
    if any(clay[name] != render[name] for name in ("front_axis", "resolution")):
        raise GeometryClosureError("closure clay summary contradicts render manifest")
    if clay["material_mode"] != render["material_mode"]:
        raise GeometryClosureError("closure clay material mode changed")
    for name in VIEWS:
        view_path = _record_path(clay["views"][name], f"closure clay {name}")
        canonical_view = render_path.parent / f"{name}.png"
        _same_file(view_path, canonical_view, f"closure clay {name}")
        _image(view_path, f"closure clay {name}", (480, 480))
    contact_path = _record_path(
        clay["contact_sheet"],
        "closure clay contact sheet",
    )
    expected_contact = (
        render_path.parent.parent / "contact_sheet.png"
        if render_path.parent.name == "views"
        else render_path.parent / "contact_sheet.png"
    )
    _same_file(contact_path, expected_contact, "closure clay contact sheet")
    _image(contact_path, "closure clay contact sheet", (960, 640))
    observed_container = _glb_readback(repaired)
    if manifest["container_readback"] != observed_container:
        raise GeometryClosureError("closure GLB container readback was resealed")
    return {
        "manifest": manifest,
        "paths": {
            "raw_pixal_glb": raw_glb,
            "pixal_manifest": pixal_manifest,
            "source_reference": source_reference,
            "raw_static_decision_batch": decision_batch,
            "raw_static_decision": decision_path,
            "raw_static_review": static_review_path,
            "repair_manifest": repair_path,
            "repaired_glb": repaired,
            "geometry_audit": audit_path,
            "clay_render_manifest": render_path,
            "clay_contact_sheet": contact_path,
        },
        "repair": repair,
        "audit": audit,
    }


def load_geometry_closure_v2(
    path: Path,
    *,
    expected_manifest_sha256: str | None = None,
    expected_instance_id: str | None = None,
    expected_raw_pixal_glb: Path | None = None,
    expected_pixal_manifest: Path | None = None,
    expected_source_reference: Path | None = None,
    expected_raw_static_decision_batch: Path | None = None,
    expected_raw_static_decision: Path | None = None,
    expected_repair_manifest: Path | None = None,
    expected_repaired_glb: Path | None = None,
    expected_geometry_audit: Path | None = None,
) -> dict[str, Any]:
    """Load a published v2 closure and replay every nested authority."""

    path = _regular_file(path, "generated animal geometry closure v2")
    if expected_manifest_sha256 is not None and _sha256_file(path) != _sha256(
        expected_manifest_sha256, "geometry closure expected SHA-256"
    ):
        raise GeometryClosureError("geometry closure changed from its external SHA-256")
    payload = contracts.load_json(path)
    return _replay_manifest(
        payload,
        expected_instance_id=expected_instance_id,
        expected_raw_pixal_glb=expected_raw_pixal_glb,
        expected_pixal_manifest=expected_pixal_manifest,
        expected_source_reference=expected_source_reference,
        expected_raw_static_decision_batch=expected_raw_static_decision_batch,
        expected_raw_static_decision=expected_raw_static_decision,
        expected_repair_manifest=expected_repair_manifest,
        expected_repaired_glb=expected_repaired_glb,
        expected_geometry_audit=expected_geometry_audit,
    )


def load_geometry_closure(
    path: Path,
    *,
    expected_manifest_sha256: str | None = None,
    **expected: Any,
) -> tuple[str, dict[str, Any]]:
    """Dispatch v2 strictly while leaving the legacy v1 reader explicit."""

    path = _regular_file(path, "generated animal geometry closure")
    if expected_manifest_sha256 is not None and _sha256_file(path) != _sha256(
        expected_manifest_sha256, "geometry closure expected SHA-256"
    ):
        raise GeometryClosureError("geometry closure changed from its external SHA-256")
    payload = contracts.load_json(path)
    schema = payload.get("schema") if isinstance(payload, Mapping) else None
    if schema == SCHEMA:
        return SCHEMA, _replay_manifest(payload, **expected)
    if schema == LEGACY_SCHEMA:
        if not isinstance(payload, dict):
            raise GeometryClosureError("legacy geometry closure must be an object")
        return LEGACY_SCHEMA, copy.deepcopy(payload)
    raise GeometryClosureError(f"unsupported geometry closure schema: {schema!r}")


def publish_geometry_closure(
    *,
    instance_id: str,
    raw_pixal_glb_path: Path,
    expected_raw_pixal_glb_sha256: str,
    pixal_manifest_path: Path,
    expected_pixal_manifest_sha256: str,
    source_reference_path: Path,
    expected_source_reference_sha256: str,
    raw_static_decision_batch_path: Path,
    expected_raw_static_decision_batch_sha256: str,
    raw_static_decision_path: Path,
    expected_raw_static_decision_sha256: str,
    repair_manifest_path: Path,
    expected_repair_manifest_sha256: str,
    repaired_glb_path: Path,
    expected_repaired_glb_sha256: str,
    geometry_audit_path: Path,
    expected_geometry_audit_sha256: str,
    clay_render_manifest_path: Path,
    expected_clay_render_manifest_sha256: str,
    clay_views: Mapping[str, Path],
    expected_clay_view_sha256s: Mapping[str, str],
    clay_contact_sheet_path: Path,
    expected_clay_contact_sheet_sha256: str,
    output_path: Path,
) -> Path:
    """Publish one create-only v2 closure after complete strict replay."""

    instance_id = _identifier(instance_id, "instance_id")
    raw_glb = _external_file(
        raw_pixal_glb_path,
        expected_raw_pixal_glb_sha256,
        "raw Pixel3D GLB",
    )
    pixal_manifest = _external_file(
        pixal_manifest_path,
        expected_pixal_manifest_sha256,
        "Pixel3D manifest",
    )
    source_reference = _external_file(
        source_reference_path,
        expected_source_reference_sha256,
        "source reference",
    )
    decision_batch = _external_file(
        raw_static_decision_batch_path,
        expected_raw_static_decision_batch_sha256,
        "raw static decision batch",
    )
    decision = _external_file(
        raw_static_decision_path,
        expected_raw_static_decision_sha256,
        "raw static decision",
    )
    repair_path = _external_file(
        repair_manifest_path,
        expected_repair_manifest_sha256,
        "oriented repair manifest",
    )
    repaired = _external_file(
        repaired_glb_path,
        expected_repaired_glb_sha256,
        "repaired GLB",
    )
    audit_path = _external_file(
        geometry_audit_path,
        expected_geometry_audit_sha256,
        "independent geometry audit",
    )
    render_path = _external_file(
        clay_render_manifest_path,
        expected_clay_render_manifest_sha256,
        "clay render manifest",
    )
    if set(clay_views) != set(VIEWS) or set(expected_clay_view_sha256s) != set(VIEWS):
        raise GeometryClosureError("exactly five named clay views are required")
    authenticated_views = {
        name: _external_file(
            clay_views[name],
            expected_clay_view_sha256s[name],
            f"clay {name}",
        )
        for name in VIEWS
    }
    contact = _external_file(
        clay_contact_sheet_path,
        expected_clay_contact_sheet_sha256,
        "clay contact sheet",
    )
    try:
        repair = oriented_repair.validate_repair_manifest(
            contracts.load_json(repair_path)
        )
    except (
        contracts.ContractError,
        oriented_repair.OrientedSheetRepairError,
    ) as error:
        raise GeometryClosureError(
            f"oriented repair manifest failed strict replay: {error}"
        ) from error
    audit = _validate_audit(contracts.load_json(audit_path), repaired)
    render = _validate_render_manifest(contracts.load_json(render_path), repaired)
    for name, path in authenticated_views.items():
        _same_file(path, render_path.parent / f"{name}.png", f"clay {name}")
        _image(path, f"clay {name}", (480, 480))
    expected_contact = (
        render_path.parent.parent / "contact_sheet.png"
        if render_path.parent.name == "views"
        else render_path.parent / "contact_sheet.png"
    )
    _same_file(contact, expected_contact, "clay contact sheet")
    _image(contact, "clay contact sheet", (960, 640))
    container = _glb_readback(repaired)
    batch_payload = contracts.load_json(decision_batch)
    batch_internal_sha256 = (
        batch_payload.get("decision_batch_sha256")
        if isinstance(batch_payload, Mapping)
        else None
    )
    _sha256(batch_internal_sha256, "raw static decision batch internal hash")
    decision_payload = contracts.load_json(decision)
    if not isinstance(decision_payload, Mapping):
        raise GeometryClosureError("raw static decision must be an object")
    static_review_record = _file_record(
        decision_payload.get("review"),
        "raw static decision review",
    )
    mutation = repair["mutation"]
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": STATUS,
        "state_classification": STATE,
        "formal_dataset_registration_authorized": False,
        "identity": {"instance_id": instance_id},
        "source_authorities": {
            "raw_pixal_glb": _record(raw_glb),
            "pixal_manifest": _record(pixal_manifest),
            "source_reference": _record(source_reference),
            "raw_static_decision_batch": {
                "file": _record(decision_batch),
                "decision_batch_sha256": batch_internal_sha256,
            },
            "raw_static_decision": {
                "file": _record(decision),
                "static_review": static_review_record,
                "decision_sha256": decision_payload.get("decision_sha256"),
                "decision": decision_payload.get("decision"),
                "state_classification": decision_payload.get("state_classification"),
                "formal_dataset_registration_authorized": decision_payload.get(
                    "formal_dataset_registration_authorized"
                ),
                "next_gate": decision_payload.get("next_gate"),
                "checks": copy.deepcopy(decision_payload.get("checks")),
            },
        },
        "bounded_repair": {
            "manifest": _record(repair_path),
            "implementation_contract": repair["implementation_contract"],
            "output": _record(repaired),
            "mutation_class": MUTATION_CLASS,
            "removed_exact_position_degenerate_triangle_count": mutation[
                "removed_triangle_count"
            ],
            "output_byte_identical_to_raw": mutation["output_byte_identical_to_source"],
        },
        "independent_audit": {
            "file": _record(audit_path),
            "schema": geometry_audit.SCHEMA,
            "record_label": audit["label"],
            "topology_acceptance_semantics": audit["topology"][
                "topology_acceptance_semantics"
            ],
            "degenerate_triangles_after_position_indexing": audit["topology"][
                "degenerate_triangles_after_position_indexing"
            ],
            "unpaired_oriented_edges": audit["topology"]["unpaired_oriented_edges"],
            "unpaired_oriented_edge_occurrences": audit["topology"][
                "unpaired_oriented_edge_occurrences"
            ],
            "decision_status": audit["decision"]["status"],
        },
        "clay_readback": {
            "render_manifest": _record(render_path),
            "front_axis": render["front_axis"],
            "resolution": render["resolution"],
            "material_mode": render["material_mode"],
            "views": {
                name: _record(path)
                for name, path in sorted(authenticated_views.items())
            },
            "contact_sheet": _record(contact),
            "human_approval_claimed": False,
        },
        "container_readback": container,
        "inherited_static_judgment": {
            "authority": "canonical_raw_static_approval_v1",
            "decision_sha256": decision_payload.get("decision_sha256"),
            "checks": copy.deepcopy(decision_payload.get("checks")),
            "inheritance_scope": INHERITANCE_SCOPE,
            "new_human_approval_created": False,
            "clay_render_human_approval_claimed": False,
        },
        "downstream": {
            "tokenrig_entry_authorized": True,
            "tokenrig_execution_performed": False,
            "quaternius_rig_swap_authorized": False,
            "animation_execution_performed": False,
            "ue_import_executed": False,
            "native_change_executed": False,
            "emitter_measurement_executed": False,
            "formal_dataset_registration_authorized": False,
        },
        "automatic_checks": {name: True for name in sorted(AUTOMATIC_CHECK_FIELDS)},
    }
    manifest["manifest_sha256"] = _hash_without(manifest, "manifest_sha256")
    _replay_manifest(
        manifest,
        expected_instance_id=instance_id,
        expected_raw_pixal_glb=raw_glb,
        expected_pixal_manifest=pixal_manifest,
        expected_source_reference=source_reference,
        expected_raw_static_decision_batch=decision_batch,
        expected_raw_static_decision=decision,
        expected_repair_manifest=repair_path,
        expected_repaired_glb=repaired,
        expected_geometry_audit=audit_path,
    )
    output = _new_file(output_path)
    encoded = (
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    created_identity: tuple[int, int] | None = None
    try:
        with output.open("xb") as stream:
            opened_stat = os.fstat(stream.fileno())
            created_identity = (opened_stat.st_dev, opened_stat.st_ino)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        load_geometry_closure_v2(
            output,
            expected_manifest_sha256=_sha256_file(output),
            expected_instance_id=instance_id,
            expected_raw_pixal_glb=raw_glb,
            expected_pixal_manifest=pixal_manifest,
            expected_source_reference=source_reference,
            expected_raw_static_decision_batch=decision_batch,
            expected_raw_static_decision=decision,
            expected_repair_manifest=repair_path,
            expected_repaired_glb=repaired,
            expected_geometry_audit=audit_path,
        )
    except Exception:
        if created_identity is not None:
            try:
                current = output.lstat()
            except FileNotFoundError:
                current = None
            if (
                current is not None
                and stat.S_ISREG(current.st_mode)
                and (current.st_dev, current.st_ino) == created_identity
            ):
                output.unlink()
        raise
    return output.resolve()


def _parse_view_hashes(values: Sequence[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise GeometryClosureError(f"{label} must use VIEW=VALUE")
        name, item = value.split("=", 1)
        if name not in VIEWS or name in result:
            raise GeometryClosureError(f"{label} contains an invalid view")
        result[name] = item
    if set(result) != set(VIEWS):
        raise GeometryClosureError(f"{label} must cover all five views")
    return result


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", required=True)
    pairs = (
        ("raw-pixal-glb", Path),
        ("pixal-manifest", Path),
        ("source-reference", Path),
        ("raw-static-decision-batch", Path),
        ("raw-static-decision", Path),
        ("repair-manifest", Path),
        ("repaired-glb", Path),
        ("geometry-audit", Path),
        ("clay-render-manifest", Path),
        ("clay-contact-sheet", Path),
    )
    for name, value_type in pairs:
        parser.add_argument(f"--{name}", required=True, type=value_type)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument(
        "--clay-view",
        action="append",
        required=True,
        metavar="VIEW=PATH",
    )
    parser.add_argument(
        "--expected-clay-view-sha256",
        action="append",
        required=True,
        metavar="VIEW=SHA256",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        view_values = _parse_view_hashes(args.clay_view, "--clay-view")
        view_hashes = _parse_view_hashes(
            args.expected_clay_view_sha256,
            "--expected-clay-view-sha256",
        )
        output = publish_geometry_closure(
            instance_id=args.instance_id,
            raw_pixal_glb_path=args.raw_pixal_glb,
            expected_raw_pixal_glb_sha256=args.expected_raw_pixal_glb_sha256,
            pixal_manifest_path=args.pixal_manifest,
            expected_pixal_manifest_sha256=args.expected_pixal_manifest_sha256,
            source_reference_path=args.source_reference,
            expected_source_reference_sha256=args.expected_source_reference_sha256,
            raw_static_decision_batch_path=args.raw_static_decision_batch,
            expected_raw_static_decision_batch_sha256=(
                args.expected_raw_static_decision_batch_sha256
            ),
            raw_static_decision_path=args.raw_static_decision,
            expected_raw_static_decision_sha256=(
                args.expected_raw_static_decision_sha256
            ),
            repair_manifest_path=args.repair_manifest,
            expected_repair_manifest_sha256=args.expected_repair_manifest_sha256,
            repaired_glb_path=args.repaired_glb,
            expected_repaired_glb_sha256=args.expected_repaired_glb_sha256,
            geometry_audit_path=args.geometry_audit,
            expected_geometry_audit_sha256=args.expected_geometry_audit_sha256,
            clay_render_manifest_path=args.clay_render_manifest,
            expected_clay_render_manifest_sha256=(
                args.expected_clay_render_manifest_sha256
            ),
            clay_views={name: Path(path) for name, path in view_values.items()},
            expected_clay_view_sha256s=view_hashes,
            clay_contact_sheet_path=args.clay_contact_sheet,
            expected_clay_contact_sheet_sha256=(
                args.expected_clay_contact_sheet_sha256
            ),
            output_path=args.output,
        )
    except (OSError, contracts.ContractError, GeometryClosureError) as error:
        print(f"GENERATED_ANIMAL_GEOMETRY_CLOSURE_FAILED {error}", file=sys.stderr)
        return 2
    print(f"GENERATED_ANIMAL_GEOMETRY_CLOSURE_OK output={output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
