#!/usr/bin/env python3
"""Register static-qualified controlled animal outputs as source_asset_v2 candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import adopt_direct_animal_pixal_attempt as direct_adopter
from tools import build_controlled_source_asset_inputs as input_builder
from tools import controlled_animal_derived_static_review_contract as derived_review_contract
from tools import controlled_source_asset_schema as contracts
from tools import execute_controlled_rocketbox_material_jobs as preflight_tools
from tools import freeze_controlled_animal_derived_static_human_decision as derived_static_decisions
from tools import prepare_controlled_source_asset_execution as preparation
from tools import review_controlled_animal_pixal_static_candidates as static_decisions
from tools import rocketbox_native_material_canary as immutable
from tools import run_controlled_animal_pixal_jobs as pixal_runner


REGISTRY_SCHEMA = "avengine_controlled_animal_source_asset_registry_v1"
LEGACY_DERIVED_REGISTRY_SCHEMA = (
    "avengine_controlled_animal_source_asset_registry_v2"
)
DERIVED_REGISTRY_SCHEMA = "avengine_controlled_animal_source_asset_registry_v3"
DIRECT_GEOMETRY_REGISTRY_SCHEMA = (
    "avengine_controlled_animal_source_asset_registry_v4"
)
DIRECT_GEOMETRY_RAW_DECISION_PROVENANCE_MODEL = (
    "canonical_raw_static_decision_sha256"
)
DIRECT_GEOMETRY_CLOSURE_PROVENANCE_MODEL = (
    "bounded_geometry_closure_manifest_sha256"
)
PHYSICAL_PROFILE_AUTHORITY_REQUEST_MODEL = "physical_profile_authority_request_sha256"
PHYSICAL_PROFILE_AUTHORITY_ARTIFACT_ROLE = "physical_profile_authority_request_batch"
PHYSICAL_PROFILE_BASE_FIELDS = frozenset(
    ("control_attribute", "measurement", "mode", "profile_id",
     "reference_value_cm", "selected_value", "target_value_cm", "tolerance_cm")
)
DERIVED_REGISTRY_AUTOMATIC_CHECKS = {
    "all_requests_reauthenticated": True,
    "all_pixal_input_attempt_request_identities_reauthenticated": True,
    "all_pixal_outputs_reauthenticated": True,
    "raw_static_decisions_preserved_without_rewrite": True,
    "all_derived_static_decisions_reauthenticated": True,
    "all_repaired_glbs_and_geometry_closures_reauthenticated": True,
    "all_source_asset_v2_validated_against_request_and_profile": True,
    "all_physical_measurements_pending": True,
    "all_animation_ue_audio_qa_pending": True,
    "all_rights_blockers_preserved": True,
    "overall": "passed",
}
LEGACY_DERIVED_REGISTRY_AUTOMATIC_CHECKS = {
    **{
        name: value
        for name, value in DERIVED_REGISTRY_AUTOMATIC_CHECKS.items()
        if name != "raw_static_decisions_preserved_without_rewrite"
    },
    "raw_static_rejections_preserved": True,
}
DIRECT_DERIVED_REGISTRY_AUTOMATIC_CHECKS = {
    "direct_source_authority_reauthenticated": True,
    "adopted_pixal_batch_and_source_spec_reauthenticated": True,
    "adopted_pixal_output_and_attempt_reauthenticated": True,
    "raw_static_decision_preserved_without_rewrite": True,
    "derived_static_decision_reauthenticated": True,
    "repaired_glb_and_geometry_closure_reauthenticated": True,
    "source_asset_v2_bare_validated_and_bound_to_direct_authority": True,
    "research_candidate_state_preserved": True,
    "formal_dataset_registration_not_authorized": True,
    "rights_blockers_preserved": True,
    "overall": "passed",
}
DIRECT_GEOMETRY_REGISTRY_AUTOMATIC_CHECKS = {
    "direct_source_authority_reauthenticated": True,
    "adopted_pixal_batch_and_source_spec_reauthenticated": True,
    "original_adopted_and_static_review_byte_copies_reauthenticated": True,
    "raw_static_decision_preserved_without_rewrite": True,
    "exact_geometry_closure_replayed": True,
    "repair_limited_to_zero_area_filter_or_byte_identical_noop": True,
    "canonical_raw_static_approval_inherited_without_new_human_claim": True,
    "derived_static_decision_not_created": True,
    "source_asset_v2_bare_validated_and_bound_to_direct_authority": True,
    "research_candidate_state_preserved": True,
    "formal_dataset_registration_not_authorized": True,
    "rights_blockers_preserved": True,
    "overall": "passed",
}
SPEAR_ROOT = Path(__file__).resolve().parents[1]
MODELS_ROOT = Path("/data/models")
LICENSE_SPECS = (
    {
        "path": "hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/e7b7dc27f91deacad38e78976d1f2b499d76a294/LICENSE.md",
        "sha256": "ca02bc51900ab07789d1b70283329e7137f5af98f5161c23a1c81fc38a4af1fe",
        "size_bytes": 9584,
    },
    {
        "path": "hub/models--TencentARC--Pixal3D/snapshots/0b31f9160aa400719af409098bff7936a932f726/LICENSE",
        "sha256": "31d37e9c4fee1e0cd2196bccd592e8a2c30bfa17ea177d70ad25f977ba6bd9c0",
        "size_bytes": 1064,
    },
    {
        "path": "hub/models--camenduru--dinov3-vitl16-pretrain-lvd1689m/snapshots/3c276edd87d6f6e569ff0c4400e086807d0f3881/LICENSE.md",
        "sha256": "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e",
        "size_bytes": 7503,
    },
    {
        "path": "rembg/isnet-general-use/rembg-2.0.69-LICENSE.txt",
        "sha256": "90a3215072968fd304669c5389f04f1274a587abdd0507d99dead0f5511f8999",
        "size_bytes": 1069,
    },
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256(
        {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    )


def _authenticate_physical_profile_authority_request_batch(
    path: Path,
    expected_file_sha256: str,
    *,
    taxonomy: Mapping[str, Any],
    sampled_attributes: Mapping[str, Any],
    target_physical_profile: Mapping[str, Any],
    expected_request_sha256: str | None = None,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """Recover only missing physical reference provenance from one request."""

    direct_adopter._require_sha256(expected_file_sha256, "authority batch SHA-256")
    literal = Path(path).absolute()
    resolved = literal.resolve()
    if (
        literal.is_symlink()
        or not resolved.is_file()
        or resolved.stat().st_size <= 0
        or _sha256_file(resolved) != expected_file_sha256
    ):
        raise contracts.ContractError("physical-profile authority batch changed")
    batch = contracts.load_json(resolved)
    requests = batch.get("requests") if isinstance(batch, Mapping) else None
    if (not isinstance(batch, Mapping)
            or batch.get("schema") != contracts.REQUEST_BATCH_SCHEMA
            or not isinstance(requests, list) or not requests):
        raise contracts.ContractError("physical-profile authority batch is invalid")
    matching = []
    for value in requests:
        request = contracts.validate_request_integrity(value)
        physical = request.get("target_physical_profile")
        if (
            request.get("taxonomy") == taxonomy
            and request.get("sampled_attributes") == sampled_attributes
            and isinstance(physical, Mapping)
            and all(
                physical.get(field) == target_physical_profile.get(field)
                for field in PHYSICAL_PROFILE_BASE_FIELDS
            )
        ):
            matching.append(request)
    if len(matching) != 1:
        raise contracts.ContractError("physical-profile authority match is not unique")
    request = matching[0]
    if (expected_request_sha256 is not None
            and request["request_sha256"] != direct_adopter._require_sha256(
                expected_request_sha256, "authority request SHA-256")):
        raise contracts.ContractError("physical-profile authority request changed")
    authority_profile = request.get("target_physical_profile")
    authority_without_provenance = (copy.deepcopy(dict(authority_profile))
                                    if isinstance(authority_profile, Mapping) else {})
    reference_provenance = authority_without_provenance.pop(
        "reference_provenance", None
    )
    if (
        "reference_provenance" in target_physical_profile
        or reference_provenance is None
        or authority_without_provenance != target_physical_profile
    ):
        raise contracts.ContractError("authority differs beyond reference_provenance")
    restored = copy.deepcopy(dict(target_physical_profile))
    restored["reference_provenance"] = reference_provenance
    return resolved, request, restored


def spear_artifact(path: Path) -> dict[str, Any]:
    literal = Path(path).absolute()
    path = literal.resolve()
    try:
        relative = path.relative_to(SPEAR_ROOT.resolve())
    except ValueError:
        # SPEAR/tmp is intentionally a logical repository path backed by the
        # external workspace symlink. Preserve that logical root in durable
        # artifact records while authenticating the resolved target bytes.
        try:
            tmp_relative = path.relative_to((SPEAR_ROOT / "tmp").resolve())
        except ValueError as error:
            raise contracts.ContractError(
                f"artifact is outside SPEAR root: {path}"
            ) from error
        relative = Path("tmp") / tmp_relative
    if literal.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise contracts.ContractError(f"artifact is missing/non-direct: {path}")
    return {
        "root_id": "spear_repo",
        "path": relative.as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verified_file_record(
    value: Any,
    label: str,
    *,
    root: Path | None = None,
) -> Path:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256", "size_bytes"}
        or not isinstance(value.get("path"), str)
    ):
        raise contracts.ContractError(f"{label} descriptor is invalid")
    literal = Path(value["path"])
    if root is None:
        if not literal.is_absolute():
            raise contracts.ContractError(f"{label} path must be absolute")
    else:
        if literal.is_absolute() or ".." in literal.parts or not literal.parts:
            raise contracts.ContractError(f"{label} path escaped its root")
        literal = root / literal
    path = literal.resolve()
    if root is not None:
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise contracts.ContractError(f"{label} escaped its root") from error
    size = value.get("size_bytes")
    if (
        literal.is_symlink()
        or not path.is_file()
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or path.stat().st_size != size
        or _sha256_file(path) != value.get("sha256")
    ):
        raise contracts.ContractError(f"{label} changed")
    return path


def _verified_descriptor_target(
    value: Any,
    expected: Path,
    label: str,
    *,
    base: Path = SPEAR_ROOT,
) -> Path:
    """Reauthenticate a nested descriptor and bind it to one expected file."""

    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("path"), str)
        or not value["path"]
    ):
        raise contracts.ContractError(f"{label} descriptor is invalid")
    literal = Path(value["path"])
    candidate = literal if literal.is_absolute() else base / literal
    path = candidate.resolve()
    expected = Path(expected).resolve()
    size = value.get("size_bytes")
    if (
        candidate.is_symlink()
        or not path.is_file()
        or _sha256_file(path) != value.get("sha256")
        or (
            size is not None
            and (
                isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
                or path.stat().st_size != size
            )
        )
    ):
        raise contracts.ContractError(f"{label} changed")
    try:
        identical = os.path.samefile(path, expected)
    except OSError as error:
        raise contracts.ContractError(
            f"cannot compare {label} file identity"
        ) from error
    if not identical:
        raise contracts.ContractError(f"{label} points to a different file")
    return path


def _authenticated_byte_copy_pair(
    source: Path,
    copy_path: Path,
    label: str,
) -> tuple[Path, Path]:
    """Authenticate two direct files and prove that their bytes are identical."""

    authenticated = []
    for role, value in (("source", source), ("copy", copy_path)):
        literal = Path(value).absolute()
        path = literal.resolve()
        if literal.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise contracts.ContractError(
                f"{label} {role} is missing or non-direct"
            )
        authenticated.append(path)
    source_path, copied_path = authenticated
    if (
        source_path.stat().st_size != copied_path.stat().st_size
        or _sha256_file(source_path) != _sha256_file(copied_path)
    ):
        raise contracts.ContractError(f"{label} is not a byte copy")
    return source_path, copied_path


def _require_same_file(first: Path, second: Path, label: str) -> None:
    """Require one canonical authority file, not merely equivalent bytes."""

    try:
        identical = os.path.samefile(Path(first).resolve(), Path(second).resolve())
    except OSError as error:
        raise contracts.ContractError(f"cannot compare {label}") from error
    if not identical:
        raise contracts.ContractError(f"{label} was rebound to another file")


def license_records() -> list[dict[str, Any]]:
    models_root = MODELS_ROOT.absolute()
    if models_root.is_symlink() or not models_root.is_dir():
        raise contracts.ContractError("/data/models root is missing or unsafe")
    resolved_root = models_root.resolve()
    records = []
    for spec in LICENSE_SPECS:
        logical = models_root / spec["path"]
        resolved = logical.resolve(strict=True)
        try:
            relative = resolved.relative_to(resolved_root)
        except ValueError as error:
            raise contracts.ContractError("model license escaped /data/models") from error
        if (
            not resolved.is_file()
            or resolved.stat().st_size != spec["size_bytes"]
            or _sha256_file(resolved) != spec["sha256"]
        ):
            raise contracts.ContractError(f"model license changed: {logical}")
        # Hugging Face snapshot license entries are normally symlink leaves.
        # Preserve the pinned bytes while recording their direct content-addressed
        # blob path so downstream consumers never need to weaken symlink checks.
        direct = resolved_root / relative
        if direct.is_symlink() or direct.resolve(strict=True) != resolved:
            raise contracts.ContractError(
                f"resolved model license is not a direct file: {resolved}"
            )
        records.append(
            {
                "root_id": "models_root",
                "path": relative.as_posix(),
                "sha256": spec["sha256"],
                "size_bytes": spec["size_bytes"],
            }
        )
    return records


def load_decision_batch(path: Path):
    literal = Path(path).absolute()
    path = literal.resolve()
    if literal.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise contracts.ContractError("static decision batch is missing or unsafe")
    payload = contracts.load_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != static_decisions.DECISION_BATCH_SCHEMA
        or payload.get("status") != "completed"
        or payload.get("decision_batch_sha256")
        != _hash_without(payload, "decision_batch_sha256")
        or payload.get("automatic_checks", {}).get("overall") != "passed"
        or payload.get("decision_count") != len(payload.get("decisions", []))
    ):
        raise contracts.ContractError("static decision batch contract/hash is invalid")

    review_batch_descriptor = payload.get("static_review_batch")
    if (
        not isinstance(review_batch_descriptor, Mapping)
        or set(review_batch_descriptor)
        != {"path", "sha256", "review_batch_sha256"}
        or not isinstance(review_batch_descriptor.get("path"), str)
        or not Path(review_batch_descriptor["path"]).is_absolute()
        or not isinstance(review_batch_descriptor.get("sha256"), str)
        or len(review_batch_descriptor["sha256"]) != 64
        or not isinstance(review_batch_descriptor.get("review_batch_sha256"), str)
        or len(review_batch_descriptor["review_batch_sha256"]) != 64
    ):
        raise contracts.ContractError("static review batch descriptor is invalid")
    review_batch_literal = Path(review_batch_descriptor["path"])
    review_batch_path = review_batch_literal.resolve()
    if (
        review_batch_literal.is_symlink()
        or not review_batch_path.is_file()
        or _sha256_file(review_batch_path) != review_batch_descriptor["sha256"]
    ):
        raise contracts.ContractError("static review batch changed")
    (
        authenticated_review_batch_path,
        authenticated_review_batch,
        authenticated_reviews,
    ) = static_decisions.load_review_batch(review_batch_path)
    if (
        authenticated_review_batch_path != review_batch_path
        or authenticated_review_batch["review_batch_sha256"]
        != review_batch_descriptor["review_batch_sha256"]
    ):
        raise contracts.ContractError("static review batch identity changed")

    records = {}
    root = path.parent
    for index in payload["decisions"]:
        expected_index_fields = {
            "instance_id",
            "decision",
            "decision_sha256",
            "record",
        }
        record_descriptor = index.get("record") if isinstance(index, Mapping) else None
        if (
            not isinstance(index, Mapping)
            or set(index) != expected_index_fields
            or index.get("decision")
            not in {"approved_for_lod_and_binding", "rejected"}
            or not isinstance(record_descriptor, Mapping)
            or set(record_descriptor) != {"path", "sha256", "size_bytes"}
            or not isinstance(record_descriptor.get("path"), str)
            or Path(record_descriptor["path"]).is_absolute()
            or ".." in Path(record_descriptor["path"]).parts
            or not isinstance(record_descriptor.get("sha256"), str)
            or len(record_descriptor["sha256"]) != 64
            or isinstance(record_descriptor.get("size_bytes"), bool)
            or not isinstance(record_descriptor.get("size_bytes"), int)
            or record_descriptor["size_bytes"] <= 0
        ):
            raise contracts.ContractError("static decision index is invalid")
        record_literal = root / record_descriptor["path"]
        record_path = record_literal.resolve()
        try:
            record_path.relative_to(root.resolve())
        except ValueError as error:
            raise contracts.ContractError(
                "static decision record escaped its batch root"
            ) from error
        if (
            record_literal.is_symlink()
            or not record_path.is_file()
            or record_path.stat().st_size != record_descriptor["size_bytes"]
            or _sha256_file(record_path) != record_descriptor["sha256"]
        ):
            raise contracts.ContractError("static decision record changed")
        record = contracts.load_json(record_path)
        if not isinstance(record, Mapping):
            raise contracts.ContractError("static decision record is invalid")
        review = authenticated_reviews.get(index["instance_id"])
        review_descriptor = record.get("review")
        checks = record.get("checks")
        evidence = record.get("attribute_evidence")
        expected_state = (
            "research_candidate"
            if index["decision"] == "approved_for_lod_and_binding"
            else "rejected"
        )
        expected_next_gate = (
            "lod_then_species_rig_binding"
            if index["decision"] == "approved_for_lod_and_binding"
            else "stop"
        )
        if (
            record.get("schema") != "avengine_controlled_animal_static_decision_v1"
            or record.get("instance_id") != index["instance_id"]
            or record.get("decision") != index["decision"]
            or record.get("decision_sha256") != index["decision_sha256"]
            or record.get("decision_sha256")
            != _hash_without(record, "decision_sha256")
            or review is None
            or not isinstance(review_descriptor, Mapping)
            or set(review_descriptor) != {"path", "sha256", "size_bytes"}
            or Path(str(review_descriptor.get("path", ""))).resolve()
            != review["path"].resolve()
            or review_descriptor.get("sha256") != _sha256_file(review["path"])
            or review_descriptor.get("size_bytes") != review["path"].stat().st_size
            or record.get("review_sha256")
            != review["payload"].get("review_sha256")
            or record.get("state_classification") != expected_state
            or record.get("formal_dataset_registration_authorized") is not False
            or record.get("next_gate") != expected_next_gate
            or not isinstance(checks, Mapping)
            or set(checks) != static_decisions.CHECK_FIELDS
            or any(not isinstance(value, bool) for value in checks.values())
            or (
                index["decision"] == "approved_for_lod_and_binding"
                and not all(checks.values())
            )
            or (
                index["decision"] == "rejected"
                and all(checks.values())
            )
            or not isinstance(evidence, Mapping)
            or set(evidence) != set(review["payload"]["sampled_attributes"])
            or evidence.get(
                review["payload"]["target_physical_profile"]["control_attribute"]
            )
            != "deferred_to_metric_3d"
            or (
                index["decision"] == "approved_for_lod_and_binding"
                and any(
                    status != "passed_static_visual"
                    for attribute, status in evidence.items()
                    if attribute
                    != review["payload"]["target_physical_profile"][
                        "control_attribute"
                    ]
                )
            )
        ):
            raise contracts.ContractError("static decision identity/hash changed")
        records[index["instance_id"]] = {
            "payload": record,
            "path": record_path,
            "static_review": review,
        }
    approved_count = sum(
        value["payload"]["decision"] == "approved_for_lod_and_binding"
        for value in records.values()
    )
    if (
        len(records) != payload["decision_count"]
        or set(records) != set(authenticated_reviews)
        or payload.get("approved_count") != approved_count
        or payload.get("rejected_count") != len(records) - approved_count
    ):
        raise contracts.ContractError("duplicate static decision records")
    return path, payload, records


def _verify_frozen_source_file(
    input_dir: Path, name: str, descriptor: Any
) -> Path:
    if (
        not isinstance(descriptor, Mapping)
        or set(descriptor) != {"path", "sha256", "size_bytes"}
        or not isinstance(descriptor.get("path"), str)
        or not Path(descriptor["path"]).is_absolute()
        or not isinstance(descriptor.get("sha256"), str)
        or len(descriptor["sha256"]) != 64
        or isinstance(descriptor.get("size_bytes"), bool)
        or not isinstance(descriptor.get("size_bytes"), int)
        or descriptor["size_bytes"] <= 0
    ):
        raise contracts.ContractError(f"frozen preflight descriptor is invalid: {name}")
    literal = Path(descriptor["path"])
    path = literal.resolve()
    if path != (input_dir / name).resolve():
        raise contracts.ContractError(
            f"frozen preflight source path changed: {name}"
        )
    if (
        literal.is_symlink()
        or not path.is_file()
        or path.stat().st_size != descriptor["size_bytes"]
        or _sha256_file(path) != descriptor["sha256"]
    ):
        raise contracts.ContractError(f"frozen preflight source file changed: {name}")
    return path


def _load_frozen_historical_preflight(
    preflight_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Reauthenticate a sealed older preflight without regenerating it.

    Historical evidence is accepted only when every field that existed at
    publication still equals the corresponding current deterministic rebuild.
    Current rebuild fields that were not present historically may be additive;
    no historical bytes are ever rewritten or silently normalized.
    """

    literal = Path(preflight_path).absolute()
    path = literal.resolve()
    if literal.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise contracts.ContractError(f"preflight is missing or unsafe: {path}")
    preflight = preparation.validate_execution_preflight(contracts.load_json(path))
    expected_preflight_fields = {
        "schema",
        "source_bundle",
        "artifact_roots",
        "profile_artifact_authentication",
        "routes",
        "execution_summary",
        "automatic_checks",
        "preflight_sha256",
    }
    if set(preflight) != expected_preflight_fields:
        raise contracts.ContractError("frozen preflight fields are invalid")

    source_bundle = preflight.get("source_bundle")
    expected_source_fields = {
        "input_dir",
        "files",
        "profile_snapshot_sha256",
        "request_batch_id",
        "request_batch_sha256",
        "execution_jobs_sha256",
        "generation_plan_id",
        "generation_plan_sha256",
        "qa_pair_plan_sha256",
        "profile_count",
        "request_count",
        "planned_job_count",
    }
    if (
        not isinstance(source_bundle, Mapping)
        or set(source_bundle) != expected_source_fields
        or not isinstance(source_bundle.get("input_dir"), str)
        or not Path(source_bundle["input_dir"]).is_absolute()
    ):
        raise contracts.ContractError("frozen preflight source bundle is invalid")
    input_dir = Path(source_bundle["input_dir"]).resolve()
    if not input_dir.is_dir():
        raise contracts.ContractError("frozen preflight input directory is missing")
    descriptors = source_bundle.get("files")
    if (
        not isinstance(descriptors, Mapping)
        or set(descriptors) != set(preparation.REQUIRED_INPUT_FILES)
    ):
        raise contracts.ContractError("frozen preflight source file set changed")
    source_paths = {
        name: _verify_frozen_source_file(input_dir, name, descriptors[name])
        for name in preparation.REQUIRED_INPUT_FILES
    }
    bundle = {
        name: contracts.load_json(source_path)
        for name, source_path in source_paths.items()
    }
    if any(not isinstance(value, dict) for value in bundle.values()):
        raise contracts.ContractError("frozen preflight source must contain JSON objects")

    roots_value = preflight.get("artifact_roots")
    if (
        not isinstance(roots_value, Mapping)
        or not roots_value
        or any(
            not isinstance(root_id, str)
            or not root_id
            or not isinstance(root_path, str)
            or not Path(root_path).is_absolute()
            for root_id, root_path in roots_value.items()
        )
    ):
        raise contracts.ContractError("frozen preflight artifact roots are invalid")
    roots = {
        root_id: Path(root_path).resolve()
        for root_id, root_path in roots_value.items()
    }

    snapshot = bundle["profile_snapshot.json"]
    if (
        snapshot.get("schema") != input_builder.PROFILE_SNAPSHOT_SCHEMA
        or set(snapshot) != {"schema", "profiles", "snapshot_sha256"}
        or snapshot.get("snapshot_sha256")
        != _hash_without(snapshot, "snapshot_sha256")
        or not isinstance(snapshot.get("profiles"), list)
        or not snapshot["profiles"]
    ):
        raise contracts.ContractError("frozen profile snapshot contract/hash is invalid")
    profiles_list: list[dict[str, Any]] = []
    for entry in snapshot["profiles"]:
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {
                "profile_schema_id",
                "profile_sha256",
                "profile",
                "artifact_authentication",
            }
        ):
            raise contracts.ContractError("frozen profile snapshot entry is invalid")
        profile = contracts.validate_attribute_profile(entry["profile"])
        if (
            entry["profile_schema_id"] != profile["profile_schema_id"]
            or entry["profile_sha256"] != contracts.profile_sha256(profile)
        ):
            raise contracts.ContractError("frozen profile identity/hash changed")
        profiles_list.append(profile)
    profiles_list.sort(key=lambda item: item["profile_schema_id"])
    profile_ids = [item["profile_schema_id"] for item in profiles_list]
    if len(profile_ids) != len(set(profile_ids)):
        raise contracts.ContractError("frozen profile snapshot contains duplicates")
    fresh_authentication = {
        profile["profile_schema_id"]: input_builder.authenticate_profile_artifacts(
            profile, roots
        )
        for profile in profiles_list
    }
    rebuilt_snapshot = input_builder.build_profile_snapshot(
        profiles_list, fresh_authentication
    )
    if contracts.canonical_json(snapshot) != contracts.canonical_json(
        rebuilt_snapshot
    ):
        raise contracts.ContractError(
            "frozen profile snapshot no longer matches authenticated artifacts"
        )

    request_batch = bundle["instance_requests.json"]
    try:
        contracts.validate_request_batch(request_batch, profiles_list)
    except contracts.ContractError as error:
        raise contracts.ContractError(
            "frozen instance requests no longer match deterministic sampling"
        ) from error
    profiles = {item["profile_schema_id"]: item for item in profiles_list}
    requests: dict[str, dict[str, Any]] = {}
    for request in request_batch["requests"]:
        profile = profiles.get(request.get("profile_schema_id"))
        if profile is None:
            raise contracts.ContractError("frozen request profile is missing")
        validated = contracts.validate_instance_request(request, profile)
        instance_id = validated["instance_id"]
        if instance_id in requests:
            raise contracts.ContractError("frozen requests contain duplicate identities")
        requests[instance_id] = validated

    execution_jobs = bundle["execution_jobs.json"]
    if (
        execution_jobs.get("schema") != input_builder.EXECUTION_JOBS_SCHEMA
        or not isinstance(execution_jobs.get("routes"), dict)
    ):
        raise contracts.ContractError("frozen execution jobs contract is invalid")
    rebuilt_jobs = input_builder.build_execution_jobs(
        request_batch, route_names=execution_jobs["routes"]
    )
    if contracts.canonical_json(execution_jobs) != contracts.canonical_json(
        rebuilt_jobs
    ):
        raise contracts.ContractError(
            "frozen execution jobs no longer match authenticated requests"
        )
    preparation._validate_planning_sidecars(
        request_batch,
        bundle["generation_plan.json"],
        bundle["qa_pair_plan.json"],
    )

    expected_source_values = {
        "profile_snapshot_sha256": snapshot["snapshot_sha256"],
        "request_batch_id": request_batch["batch_id"],
        "request_batch_sha256": request_batch["batch_sha256"],
        "execution_jobs_sha256": execution_jobs["jobs_sha256"],
        "generation_plan_id": bundle["generation_plan.json"]["plan_id"],
        "generation_plan_sha256": bundle["generation_plan.json"]["manifest_sha256"],
        "qa_pair_plan_sha256": bundle["qa_pair_plan.json"]["pair_plan_sha256"],
        "profile_count": len(profiles_list),
        "request_count": len(requests),
        "planned_job_count": execution_jobs["job_count"],
    }
    if any(
        source_bundle.get(name) != expected
        for name, expected in expected_source_values.items()
    ):
        raise contracts.ContractError("frozen preflight source identity/count changed")
    if set(preflight["routes"]) != set(execution_jobs["routes"]):
        raise contracts.ContractError(
            "frozen preflight and execution job route sets differ"
        )

    rebuilt = preparation.build_execution_preflight(input_dir, roots)
    for name in expected_preflight_fields - {
        "routes",
        "execution_summary",
        "preflight_sha256",
    }:
        if contracts.canonical_json(preflight[name]) != contracts.canonical_json(
            rebuilt[name]
        ):
            raise contracts.ContractError(
                f"frozen preflight field no longer matches current rebuild: {name}"
            )
    for route, jobs in preflight["routes"].items():
        if route not in rebuilt["routes"] or contracts.canonical_json(
            jobs
        ) != contracts.canonical_json(rebuilt["routes"][route]):
            raise contracts.ContractError(
                f"frozen preflight route no longer matches current rebuild: {route}"
            )
    historical_summary = preflight["execution_summary"]
    if (
        not isinstance(historical_summary, Mapping)
        or not set(historical_summary).issubset(rebuilt["execution_summary"])
        or any(
            rebuilt["execution_summary"][name] != value
            for name, value in historical_summary.items()
        )
    ):
        raise contracts.ContractError(
            "frozen preflight summary no longer matches current rebuild"
        )
    return preflight, requests, profiles


def load_source_contract(
    preflight_path: Path, *, frozen_historical_preflight: bool = False
):
    if frozen_historical_preflight:
        return _load_frozen_historical_preflight(preflight_path)
    preflight = preflight_tools._load_preflight(preflight_path)
    source_files = preflight["source_bundle"]["files"]
    requests_path = Path(source_files["instance_requests.json"]["path"])
    profiles_path = Path(source_files["profile_snapshot.json"]["path"])
    for name, path in (
        ("instance_requests.json", requests_path),
        ("profile_snapshot.json", profiles_path),
    ):
        expected = source_files[name]
        if (
            not path.is_file()
            or path.stat().st_size != expected["size_bytes"]
            or _sha256_file(path) != expected["sha256"]
        ):
            raise contracts.ContractError(f"preflight source file changed: {name}")
    request_batch = contracts.load_json(requests_path)
    profile_snapshot = contracts.load_json(profiles_path)
    requests = {
        request["instance_id"]: contracts.validate_request_integrity(request)
        for request in request_batch["requests"]
    }
    profiles = {
        item["profile_schema_id"]: contracts.validate_attribute_profile(
            item["profile"]
        )
        for item in profile_snapshot["profiles"]
    }
    return preflight, requests, profiles


def validate_pixal_request_identity(
    pixal_batch: Mapping[str, Any],
    pixal_inputs_manifest: Mapping[str, Any],
    requests: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind every Pixal input and attempt back to its canonical request."""

    jobs = pixal_inputs_manifest.get("jobs")
    attempts_value = pixal_batch.get("attempts")
    if (
        not isinstance(jobs, list)
        or not jobs
        or pixal_inputs_manifest.get("job_count") != len(jobs)
        or not isinstance(attempts_value, list)
        or not attempts_value
        or pixal_batch.get("job_count") != len(attempts_value)
        or pixal_batch.get("passed_count") != len(attempts_value)
        or pixal_batch.get("failed_count") != 0
        or pixal_batch.get("status") != "passed_generation_and_glb_readback"
    ):
        raise contracts.ContractError("Pixal job/attempt coverage is invalid")
    input_jobs: dict[str, Any] = {}
    for job in jobs:
        controlled = job.get("controlled_request")
        if not isinstance(controlled, Mapping):
            raise contracts.ContractError("Pixal controlled request is missing")
        instance_id = controlled.get("instance_id")
        request = requests.get(instance_id)
        if request is None or request["asset_class"] != "animal":
            raise contracts.ContractError("Pixal input has no canonical animal request")
        expected = {
            "execution_job_id": f"animal_{request['request_sha256'][:16]}",
            "instance_id": request["instance_id"],
            "request_sha256": request["request_sha256"],
            "generation_seed": int(request["generation_plan"]["generation_seed"]),
            "profile_schema_id": request["profile_schema_id"],
            "sampled_attributes": request["sampled_attributes"],
            "target_physical_profile": request["target_physical_profile"],
        }
        optional_expected = {
            "profile_sha256": request["profile_sha256"],
            "asset_class": "animal",
            "route": "flux2_pixal3d_animal_v1",
            "rig_profile": request["rig_profile"],
        }
        if any(
            contracts.canonical_json(controlled.get(name))
            != contracts.canonical_json(value)
            for name, value in expected.items()
        ) or any(
            name in controlled
            and contracts.canonical_json(controlled[name])
            != contracts.canonical_json(value)
            for name, value in optional_expected.items()
        ):
            raise contracts.ContractError(
                "Pixal input/canonical request identity changed"
            )
        if (
            instance_id in input_jobs
            or job.get("legacy_tag") != instance_id
            or job.get("seed") != expected["generation_seed"]
        ):
            raise contracts.ContractError("Pixal input identity is duplicate/invalid")
        input_jobs[instance_id] = job

    attempts: dict[str, Any] = {}
    for attempt in attempts_value:
        if not isinstance(attempt, Mapping):
            raise contracts.ContractError("Pixal attempt is invalid")
        instance_id = attempt.get("instance_id")
        request = requests.get(instance_id)
        input_job = input_jobs.get(instance_id)
        if request is None or input_job is None:
            raise contracts.ContractError("Pixal attempt lacks its frozen input/request")
        expected = {
            "execution_job_id": f"animal_{request['request_sha256'][:16]}",
            "request_sha256": request["request_sha256"],
            "profile_schema_id": request["profile_schema_id"],
            "sampled_attributes": request["sampled_attributes"],
            "target_physical_profile": request["target_physical_profile"],
            "seed": int(request["generation_plan"]["generation_seed"]),
            "attempt_ordinal": input_job["attempt_ordinal"],
            "pixal_input": input_job["reference"]["pixal_input"],
        }
        if any(
            contracts.canonical_json(attempt.get(name))
            != contracts.canonical_json(value)
            for name, value in expected.items()
        ):
            raise contracts.ContractError(
                "Pixal attempt/canonical request identity changed"
            )
        if instance_id in attempts:
            raise contracts.ContractError("duplicate Pixal attempt identity")
        attempts[instance_id] = attempt
    if set(input_jobs) != set(attempts):
        raise contracts.ContractError("Pixal input/attempt identity coverage differs")
    return input_jobs, attempts


def approved_attempt_ids(
    decisions: Mapping[str, Any], attempts: Mapping[str, Any]
) -> set[str]:
    """Require a decision for every Pixal attempt and return the approved subset."""
    if set(decisions) != set(attempts):
        raise contracts.ContractError(
            "static decision/Pixal attempt coverage differs"
        )
    approved = {
        instance_id
        for instance_id, value in decisions.items()
        if value["payload"]["decision"] == "approved_for_lod_and_binding"
    }
    if not approved:
        raise contracts.ContractError(
            "source registry requires at least one approved static candidate"
        )
    return approved


def _load_registration_context(
    preflight_path: Path,
    pixal_batch_path: Path,
    *,
    frozen_historical_preflight: bool,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    Path,
    dict[str, Any],
    Path,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    preflight_literal = Path(preflight_path).absolute()
    preflight, requests, profiles = load_source_contract(
        preflight_literal,
        frozen_historical_preflight=frozen_historical_preflight,
    )
    preflight_path = preflight_literal.resolve()
    pixal_batch_path = Path(pixal_batch_path).resolve()
    if pixal_batch_path.is_symlink() or not pixal_batch_path.is_file():
        raise contracts.ContractError("Pixal batch is missing")
    pixal_batch = contracts.load_json(pixal_batch_path)
    if (
        pixal_batch.get("schema") != pixal_runner.BATCH_SCHEMA
        or pixal_batch.get("batch_sha256")
        != _hash_without(pixal_batch, "batch_sha256")
        or pixal_batch.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise contracts.ContractError("Pixal batch contract/hash is invalid")
    pixal_inputs_path = Path(pixal_batch["pixal_inputs"]["path"]).resolve()
    if (
        not pixal_inputs_path.is_file()
        or _sha256_file(pixal_inputs_path)
        != pixal_batch["pixal_inputs"]["sha256"]
    ):
        raise contracts.ContractError("Pixal inputs manifest changed")
    _pixal_inputs_path, pixal_inputs_manifest = pixal_runner.load_pixal_inputs(
        pixal_inputs_path
    )
    if (
        pixal_inputs_manifest.get("manifest_sha256")
        != pixal_batch["pixal_inputs"].get("manifest_sha256")
    ):
        raise contracts.ContractError("Pixal input manifest identity changed")
    input_jobs, attempts = validate_pixal_request_identity(
        pixal_batch, pixal_inputs_manifest, requests
    )
    return (
        preflight_path,
        preflight,
        requests,
        profiles,
        pixal_batch_path,
        pixal_batch,
        pixal_inputs_path,
        pixal_inputs_manifest,
        input_jobs,
        attempts,
    )


def register(
    preflight_path: Path,
    pixal_batch_path: Path,
    decision_batch_path: Path,
    output_root: Path,
    *,
    frozen_historical_preflight: bool = False,
) -> Path:
    (
        preflight_path,
        preflight,
        requests,
        profiles,
        pixal_batch_path,
        pixal_batch,
        pixal_inputs_path,
        _pixal_inputs_manifest,
        input_jobs,
        attempts,
    ) = _load_registration_context(
        preflight_path,
        pixal_batch_path,
        frozen_historical_preflight=frozen_historical_preflight,
    )
    decision_batch_path, decision_batch, decisions = load_decision_batch(
        decision_batch_path
    )
    approved_ids = approved_attempt_ids(decisions, attempts)
    model_licenses = license_records()

    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(f"refusing to replace output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    try:
        entries = []
        for instance_id in sorted(approved_ids):
            request = requests.get(instance_id)
            if request is None or request["asset_class"] != "animal":
                raise contracts.ContractError("approved animal request is missing")
            profile = profiles[request["profile_schema_id"]]
            attempt = attempts[instance_id]
            decision = decisions[instance_id]
            input_job = input_jobs.get(instance_id)
            if input_job is None:
                raise contracts.ContractError(
                    "approved instance lacks a Pixal input job"
                )
            if (
                attempt["request_sha256"] != request["request_sha256"]
                or attempt["sampled_attributes"] != request["sampled_attributes"]
            ):
                raise contracts.ContractError("animal request/attempt identity changed")
            pixal_root = pixal_batch_path.parent
            output_path = (pixal_root / attempt["output"]["path"]).resolve()
            attempt_manifest_path = (
                pixal_root / attempt["attempt_manifest"]["path"]
            ).resolve()
            if (
                not output_path.is_file()
                or output_path.stat().st_size != attempt["output"]["size_bytes"]
                or _sha256_file(output_path) != attempt["output"]["sha256"]
                or not attempt_manifest_path.is_file()
                or attempt_manifest_path.stat().st_size
                != attempt["attempt_manifest"]["size_bytes"]
                or _sha256_file(attempt_manifest_path)
                != attempt["attempt_manifest"]["sha256"]
            ):
                raise contracts.ContractError("Pixal attempt artifacts changed")
            attempt_manifest = contracts.load_json(attempt_manifest_path)
            if (
                attempt_manifest.get("controlled_request")
                != input_job["controlled_request"]
                or attempt_manifest.get("output", {}).get("sha256")
                != attempt["output"]["sha256"]
            ):
                raise contracts.ContractError(
                    "Pixal attempt manifest/request binding changed"
                )
            static_review_path = decision["static_review"]["path"]
            static_review = decision["static_review"]["payload"]
            if (
                static_review.get("instance_id") != request["instance_id"]
                or static_review.get("request_sha256") != request["request_sha256"]
                or static_review.get("profile_schema_id")
                != request["profile_schema_id"]
                or contracts.canonical_json(static_review.get("sampled_attributes"))
                != contracts.canonical_json(request["sampled_attributes"])
                or contracts.canonical_json(
                    static_review.get("target_physical_profile")
                )
                != contracts.canonical_json(request["target_physical_profile"])
                or static_review.get("pixal_output", {}).get("sha256")
                != attempt["output"]["sha256"]
            ):
                raise contracts.ContractError(
                    "static review/canonical request identity changed"
                )
            contact_path = (
                static_review_path.parents[1] / static_review["contact_sheet"]["path"]
            ).resolve()
            pixal_input_path = Path(attempt["pixal_input"]["path"]).resolve()
            candidate_path = Path(input_job["reference"]["source"]["path"]).resolve()
            artifacts = {
                "flux2_candidate_image": spear_artifact(candidate_path),
                "pixal_input_rgba": spear_artifact(pixal_input_path),
                "pixal_inputs_manifest": spear_artifact(pixal_inputs_path),
                "pixal_raw_glb": spear_artifact(output_path),
                "pixal_attempt_manifest": spear_artifact(attempt_manifest_path),
                "static_review_manifest": spear_artifact(static_review_path),
                "static_contact_sheet": spear_artifact(contact_path),
                "static_decision": spear_artifact(decision["path"]),
            }
            source_asset = contracts.build_source_asset_v2(
                request,
                artifacts=artifacts,
                physical_measurements={"status": "pending"},
                provenance={
                    "attempt_id": f"static_{attempt['execution_job_id']}",
                    "request_sha256": request["request_sha256"],
                    # This must remain byte-for-byte equivalent to the immutable
                    # request contract. ISNet preprocessing is authenticated by
                    # pixal_inputs_manifest instead of being injected here.
                    "models": copy.deepcopy(
                        request["generation_plan"]["model_revisions"]
                    ),
                },
                rights={
                    "status": "review_required",
                    "licenses": copy.deepcopy(model_licenses),
                    "blockers": [
                        "legacy_reference_provenance_unknown",
                        "physical_target_reference_provisional",
                        "pixal_research_dependency_export_review_required",
                        "dino_snapshot_origin_review_required",
                    ],
                },
                qa={
                    "reference_2d": "passed",
                    "static_mesh": "passed",
                    "binding": "pending",
                    "walking": "pending",
                    "idle": "pending",
                    "ue_import_readback": "pending",
                    "apartment_media": "pending",
                    "audio": "pending",
                },
                state_classification="research_candidate",
            )
            contracts.validate_source_asset_v2(
                source_asset, request=request, profile=profile
            )
            destination = staging / "source_assets" / f"{instance_id}.json"
            contracts.write_json_no_replace(destination, source_asset)
            entries.append(
                {
                    "asset_id": instance_id,
                    "profile_schema_id": request["profile_schema_id"],
                    "request_sha256": request["request_sha256"],
                    "sampled_attributes": request["sampled_attributes"],
                    "attribute_evidence": decision["payload"]["attribute_evidence"],
                    "source_asset": {
                        "path": destination.relative_to(staging).as_posix(),
                        "sha256": _sha256_file(destination),
                        "size_bytes": destination.stat().st_size,
                    },
                    "state_classification": "research_candidate",
                    "next_gate": "lod_then_species_rig_binding",
                }
            )
        registry: dict[str, Any] = {
            "schema": REGISTRY_SCHEMA,
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "preflight": {
                "path": str(preflight_path),
                "sha256": _sha256_file(preflight_path),
                "preflight_sha256": preflight["preflight_sha256"],
                "validation_mode": (
                    "frozen_historical_preflight_v1"
                    if frozen_historical_preflight
                    else "current_exact_rebuild"
                ),
            },
            "pixal_batch": {
                "path": str(pixal_batch_path),
                "sha256": _sha256_file(pixal_batch_path),
                "batch_sha256": pixal_batch["batch_sha256"],
            },
            "static_decision_batch": {
                "path": str(decision_batch_path),
                "sha256": _sha256_file(decision_batch_path),
                "decision_batch_sha256": decision_batch["decision_batch_sha256"],
            },
            "source_asset_count": len(entries),
            "source_assets": entries,
            "automatic_checks": {
                "all_requests_reauthenticated": True,
                "all_pixal_input_attempt_request_identities_reauthenticated": True,
                "all_pixal_outputs_reauthenticated": True,
                "all_static_decisions_reauthenticated": True,
                "all_source_asset_v2_validated_against_request_and_profile": True,
                "all_physical_measurements_pending": True,
                "all_animation_ue_audio_qa_pending": True,
                "all_rights_blockers_preserved": True,
                "overall": "passed",
            },
        }
        registry["registry_sha256"] = _hash_without(registry, "registry_sha256")
        contracts.write_json_no_replace(staging / "registry_manifest.json", registry)
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError("animal registry output appeared concurrently")
        os.rename(staging, output_root)
        return output_root / "registry_manifest.json"
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def _reauthenticate_bounded_derived_geometry(
    *,
    review: Mapping[str, Any],
    raw_pixal_path: Path,
    reviewed_reference_path: Path,
    raw_decision_path: Path,
    raw_attempt_manifest_path: Path,
    repaired_glb_path: Path,
    geometry_closure_path: Path,
    repair_manifest_path: Path,
    geometry_audit_path: Path,
    raw_decision_batch_path: Path | None = None,
    pixal_input_path: Path | None = None,
) -> None:
    """Replay the bounded-local repair boundary at registration time."""

    repair_payload = contracts.load_json(repair_manifest_path)
    try:
        repair = derived_review_contract.validate_bounded_repair_manifest(
            repair_payload
        )
    except derived_review_contract.DerivedStaticReviewContractError as error:
        raise contracts.ContractError(
            f"derived repair manifest is not the current bounded contract: {error}"
        ) from error
    if (
        review["derived_geometry"]["repair_method"]
        != repair["implementation_contract"]
    ):
        raise contracts.ContractError(
            "derived review repair implementation no longer matches its manifest"
        )

    repair_method = repair["implementation_contract"]
    lineage = repair["lineage"]
    lineage_names = [
        "pixal_manifest",
        "pixal_source",
        "static_decision",
    ]
    if (
        repair_method
        == derived_review_contract.REPAIR_IMPLEMENTATION_CONTRACT
    ):
        lineage_names.extend(("approved_reference", "owner_review"))
    else:
        lineage_names.append("static_decision_batch")
    lineage_paths = {
        name: _verified_file_record(
            lineage[name],
            f"derived repair lineage {name}",
        )
        for name in lineage_names
    }
    expected_lineage = {
        "pixal_manifest": raw_attempt_manifest_path,
        "pixal_source": raw_pixal_path,
        "static_decision": raw_decision_path,
    }
    if (
        repair_method
        == derived_review_contract.REPAIR_IMPLEMENTATION_CONTRACT
    ):
        expected_lineage["approved_reference"] = reviewed_reference_path
    elif raw_decision_batch_path is None:
        raise contracts.ContractError(
            "oriented-sheet repair lacks its canonical raw decision batch"
        )
    else:
        expected_lineage["static_decision_batch"] = raw_decision_batch_path
    if any(
        lineage_paths[name] != Path(expected).resolve()
        for name, expected in expected_lineage.items()
    ):
        raise contracts.ContractError(
            "derived repair lineage no longer binds the frozen Pixal authorities"
        )
    _verified_descriptor_target(
        repair["output"],
        repaired_glb_path,
        "derived repair output",
    )

    closure = contracts.load_json(geometry_closure_path)
    if (
        isinstance(closure, Mapping)
        and closure.get("schema")
        == "avengine_generated_animal_geometry_closure_v2"
    ):
        if raw_decision_batch_path is None or pixal_input_path is None:
            raise contracts.ContractError(
                "geometry closure v2 lacks canonical raw batch/RGBA authority"
            )
        from tools import (
            publish_generated_animal_geometry_closure as geometry_closures,
        )

        try:
            geometry_closures.load_geometry_closure_v2(
                geometry_closure_path,
                expected_manifest_sha256=_sha256_file(geometry_closure_path),
                expected_instance_id=review["instance_identity"]["instance_id"],
                expected_raw_pixal_glb=raw_pixal_path,
                expected_pixal_manifest=raw_attempt_manifest_path,
                expected_source_reference=pixal_input_path,
                expected_raw_static_decision_batch=raw_decision_batch_path,
                expected_raw_static_decision=raw_decision_path,
                expected_repair_manifest=repair_manifest_path,
                expected_repaired_glb=repaired_glb_path,
                expected_geometry_audit=geometry_audit_path,
            )
        except geometry_closures.GeometryClosureError as error:
            raise contracts.ContractError(
                f"derived geometry closure v2 strict replay failed: {error}"
            ) from error
        derived_geometry = review["derived_geometry"]
        if (
            derived_geometry["automatic_gate_statuses"]
            != derived_review_contract.ORIENTED_V2_AUTOMATIC_GATE_STATUSES
            or derived_geometry["inherited_manual_review_statuses"]
            != (
                derived_review_contract
                .ORIENTED_V2_INHERITED_MANUAL_REVIEW_STATUSES
            )
        ):
            raise contracts.ContractError(
                "geometry closure v2 review claims a new manual approval"
            )
        return
    if (
        not isinstance(closure, Mapping)
        or closure.get("schema")
        != "avengine_generated_animal_geometry_closure_v1"
        or closure.get("status") != "pass_geometry_only"
        or closure.get("downstream", {}).get(
            "formal_dataset_registration_authorized"
        )
        is not False
        or closure.get("downstream", {}).get("ue_import_executed") is not False
    ):
        raise contracts.ContractError(
            "derived geometry closure boundary changed at registration"
        )
    _verified_descriptor_target(
        closure.get("candidate", {}).get("source_pixal_glb"),
        raw_pixal_path,
        "derived closure raw Pixal source",
    )
    _verified_descriptor_target(
        closure.get("candidate", {}).get("owner_approved_flux_reference"),
        reviewed_reference_path,
        "derived closure 2D reference",
    )
    output = closure.get("output")
    if not isinstance(output, Mapping):
        raise contracts.ContractError("derived closure output is missing")
    _verified_descriptor_target(
        output.get("glb"),
        repaired_glb_path,
        "derived closure repaired GLB",
    )
    _verified_descriptor_target(
        output.get("repair_manifest"),
        repair_manifest_path,
        "derived closure repair manifest",
    )
    _verified_descriptor_target(
        output.get("independent_geometry_audit"),
        geometry_audit_path,
        "derived closure geometry audit",
    )

    geometry_audit = contracts.load_json(geometry_audit_path)
    records = (
        geometry_audit.get("records")
        if isinstance(geometry_audit, Mapping)
        else None
    )
    if (
        not isinstance(geometry_audit, Mapping)
        or geometry_audit.get("schema")
        != "avengine_quadruped_i23d_geometry_audit_v4"
        or not isinstance(records, list)
        or len(records) != 1
        or not isinstance(records[0], Mapping)
    ):
        raise contracts.ContractError(
            "derived independent geometry audit contract changed"
        )
    mesh = records[0].get("mesh")
    if not isinstance(mesh, Mapping):
        raise contracts.ContractError(
            "derived independent geometry audit mesh is missing"
        )
    _verified_descriptor_target(
        {
            "path": mesh.get("absolute_path"),
            "sha256": mesh.get("sha256"),
            "size_bytes": mesh.get("size_bytes"),
        },
        repaired_glb_path,
        "derived independent geometry audit mesh",
    )
    topology = records[0].get("topology")
    audit_decision = records[0].get("decision")
    if (
        not isinstance(topology, Mapping)
        or topology.get("topology_acceptance_semantics")
        != (
            "every_exact_position_directed_edge_occurrence_has_one_"
            "oppositely_oriented_partner"
        )
        or topology.get("degenerate_triangles_after_position_indexing") != 0
        or topology.get("unpaired_oriented_edges") != 0
        or topology.get("unpaired_oriented_edge_occurrences") != 0
        or topology.get("unpaired_oriented_edge_ratio_per_triangle") != 0.0
        or not isinstance(audit_decision, Mapping)
        or audit_decision.get("status") == "reject_before_lod_and_binding"
        or audit_decision.get("rejection_reasons") != []
    ):
        raise contracts.ContractError(
            "derived independent geometry audit v4 no longer proves "
            "paired oriented sheets"
        )


def _reauthenticate_review_raw_authorities(
    *,
    raw_authority: Mapping[str, Any],
    raw_decision: Mapping[str, Any],
    decision_batch_path: Path,
    decision_batch: Mapping[str, Any],
) -> None:
    """Bind review authority to the exact canonical decision and batch paths."""

    decision_authority = raw_authority.get("raw_static_decision")
    batch_authority = raw_authority.get("raw_static_decision_batch")
    if not isinstance(decision_authority, Mapping) or not isinstance(
        batch_authority, Mapping
    ):
        raise contracts.ContractError(
            "derived review raw static authorities are incomplete"
        )
    decision_file = _verified_file_record(
        decision_authority.get("file"),
        "derived review raw static decision",
    )
    batch_file = _verified_file_record(
        batch_authority.get("file"),
        "derived review raw static decision batch",
    )
    canonical_decision_path = Path(raw_decision["path"]).resolve()
    canonical_batch_path = Path(decision_batch_path).resolve()
    canonical_payload = raw_decision.get("payload")
    if (
        not isinstance(canonical_payload, Mapping)
        or decision_file != canonical_decision_path
        or batch_file != canonical_batch_path
        or batch_authority.get("decision_batch_sha256")
        != decision_batch.get("decision_batch_sha256")
        or decision_authority.get("decision")
        != canonical_payload.get("decision")
        or decision_authority.get("state_classification")
        != canonical_payload.get("state_classification")
        or decision_authority.get("decision_sha256")
        != canonical_payload.get("decision_sha256")
    ):
        raise contracts.ContractError(
            "derived review rebound its canonical raw static authorities"
        )


def _build_direct_source_asset_v2(
    *,
    authority: Mapping[str, Any],
    direct_context: Mapping[str, Any],
    artifacts: Mapping[str, Any],
    canonical_raw_decision: Mapping[str, Any] | None = None,
    bounded_geometry_closure: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and bare-validate a direct asset, then bind every authority field."""

    batch = direct_context["adopted_batch"]
    attempt = direct_context["attempt"]
    source_spec = direct_context["source_spec"]
    controlled = direct_context["adoption_context"]["controlled"]
    if (
        source_spec.get("instance_id") != authority["instance_id"]
        or controlled.get("request_sha256") != authority["request_sha256"]
        or controlled.get("profile_schema_id") != authority["profile_schema_id"]
        or attempt.get("instance_id") != authority["instance_id"]
        or attempt.get("request_sha256") != authority["request_sha256"]
        or attempt.get("profile_schema_id") != authority["profile_schema_id"]
    ):
        raise contracts.ContractError(
            "authenticated direct batch/spec identity changed"
        )
    direct_geometry_provenance = (
        canonical_raw_decision is not None
        or bounded_geometry_closure is not None
    )
    if direct_geometry_provenance and (
        not isinstance(canonical_raw_decision, Mapping)
        or not isinstance(bounded_geometry_closure, Mapping)
    ):
        raise contracts.ContractError(
            "direct geometry provenance requires both canonical raw decision "
            "and bounded geometry closure authorities"
        )
    provenance_models = copy.deepcopy(batch["models"])
    provenance_attempt_id = f"derived_static_{attempt['execution_job_id']}"
    if direct_geometry_provenance:
        inherited = bounded_geometry_closure.get(
            "inherited_static_judgment"
        )
        raw_decision_sha256 = canonical_raw_decision.get("decision_sha256")
        closure_manifest_sha256 = bounded_geometry_closure.get(
            "manifest_sha256"
        )
        if (
            canonical_raw_decision.get("decision")
            != "approved_for_lod_and_binding"
            or canonical_raw_decision.get("state_classification")
            != "research_candidate"
            or canonical_raw_decision.get(
                "formal_dataset_registration_authorized"
            )
            is not False
            or not isinstance(raw_decision_sha256, str)
            or len(raw_decision_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in raw_decision_sha256
            )
            or bounded_geometry_closure.get("state_classification")
            != "research_candidate"
            or bounded_geometry_closure.get(
                "formal_dataset_registration_authorized"
            )
            is not False
            or not isinstance(closure_manifest_sha256, str)
            or len(closure_manifest_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in closure_manifest_sha256
            )
            or not isinstance(inherited, Mapping)
            or inherited.get("authority")
            != "canonical_raw_static_approval_v1"
            or inherited.get("decision_sha256") != raw_decision_sha256
            or inherited.get("new_human_approval_created") is not False
        ):
            raise contracts.ContractError(
                "direct geometry provenance authorities changed"
            )
        reserved_models = {
            DIRECT_GEOMETRY_RAW_DECISION_PROVENANCE_MODEL,
            DIRECT_GEOMETRY_CLOSURE_PROVENANCE_MODEL,
        }
        if not isinstance(provenance_models, Mapping) or (
            set(provenance_models) & reserved_models
        ):
            raise contracts.ContractError(
                "adopted model provenance collides with direct geometry "
                "authority records"
            )
        provenance_models.update(
            {
                DIRECT_GEOMETRY_RAW_DECISION_PROVENANCE_MODEL: (
                    raw_decision_sha256
                ),
                DIRECT_GEOMETRY_CLOSURE_PROVENANCE_MODEL: (
                    closure_manifest_sha256
                ),
            }
        )
        provenance_attempt_id = (
            f"direct_geometry_{attempt['execution_job_id']}"
        )
    semantic_attributes = {
        **copy.deepcopy(authority["taxonomy"]),
        **copy.deepcopy(authority["fixed_attributes"]),
        **copy.deepcopy(attempt["sampled_attributes"]),
    }
    source_asset = {
        "schema": contracts.SOURCE_ASSET_SCHEMA,
        "asset_id": authority["instance_id"],
        "profile_schema_id": authority["profile_schema_id"],
        "profile_sha256": authority["profile_sha256"],
        "request_sha256": authority["request_sha256"],
        "asset_class": "animal",
        "lineage_group_id": authority["lineage_group_id"],
        "taxonomy": copy.deepcopy(authority["taxonomy"]),
        "fixed_attributes": copy.deepcopy(authority["fixed_attributes"]),
        "sampled_attributes": copy.deepcopy(attempt["sampled_attributes"]),
        "semantic_attributes": semantic_attributes,
        "target_physical_profile": copy.deepcopy(
            attempt["target_physical_profile"]
        ),
        "artifacts": copy.deepcopy(dict(artifacts)),
        "physical_measurements": {"status": "pending"},
        "rig": copy.deepcopy(controlled["rig_profile"]),
        "acoustic_profile": copy.deepcopy(authority["acoustic_profile"]),
        "provenance": {
            "attempt_id": provenance_attempt_id,
            "request_sha256": authority["request_sha256"],
            "models": provenance_models,
        },
        "rights": {
            "status": "review_required",
            "licenses": license_records(),
            "blockers": [
                "direct_source_reference_rights_review_required",
                "physical_target_reference_provisional",
                "pixal_research_dependency_export_review_required",
                "dino_snapshot_origin_review_required",
            ],
        },
        "qa": {
            "reference_2d": "passed",
            "static_mesh": "passed",
            "binding": "pending",
            "walking": "pending",
            "idle": "pending",
            "ue_import_readback": "pending",
            "apartment_media": "pending",
            "audio": "pending",
        },
        "state_classification": "research_candidate",
    }
    validated = contracts.validate_source_asset_v2(source_asset)
    expected_bindings = {
        "asset_id": authority["instance_id"],
        "profile_schema_id": authority["profile_schema_id"],
        "profile_sha256": authority["profile_sha256"],
        "request_sha256": authority["request_sha256"],
        "asset_class": "animal",
        "lineage_group_id": authority["lineage_group_id"],
        "taxonomy": authority["taxonomy"],
        "fixed_attributes": authority["fixed_attributes"],
        "sampled_attributes": attempt["sampled_attributes"],
        "target_physical_profile": attempt["target_physical_profile"],
        "rig": controlled["rig_profile"],
        "acoustic_profile": authority["acoustic_profile"],
    }
    for field, expected in expected_bindings.items():
        if contracts.canonical_json(validated[field]) != contracts.canonical_json(
            expected
        ):
            raise contracts.ContractError(
                f"direct source_asset_v2 {field} escaped its authority"
            )
    if (
        validated["provenance"]["attempt_id"] != provenance_attempt_id
        or validated["provenance"]["request_sha256"]
        != authority["request_sha256"]
        or contracts.canonical_json(validated["provenance"]["models"])
        != contracts.canonical_json(provenance_models)
    ):
        raise contracts.ContractError(
            "direct source_asset_v2 provenance escaped its adopted batch"
        )
    return validated


def register_direct_geometry(
    direct_source_authority_path: Path,
    expected_direct_source_authority_sha256: str,
    pixal_batch_path: Path,
    decision_batch_path: Path,
    geometry_closure_path: Path,
    expected_geometry_closure_sha256: str,
    output_root: Path,
    *,
    physical_profile_authority_request_batch_path: Path | None = None,
    expected_physical_profile_authority_request_batch_sha256: str | None = None,
) -> Path:
    """Register one direct asset from its preserved raw approval and v2 closure.

    This mode deliberately does not manufacture a derived static review or a
    second human decision.  It inherits the exact canonical raw approval
    through the geometry closure's bounded no-op/zero-area-filter contract.
    """

    (
        direct_authority_path,
        direct_authority,
        direct_context,
    ) = direct_adopter.load_direct_source_authority(
        direct_source_authority_path,
        expected_sha256=expected_direct_source_authority_sha256,
    )
    supplied_batch_literal = Path(pixal_batch_path).absolute()
    if supplied_batch_literal.is_symlink() or not supplied_batch_literal.is_file():
        raise contracts.ContractError("adopted Pixal batch is missing or unsafe")
    pixal_batch_path, pixal_batch = direct_adopter.load_adopted_batch(
        supplied_batch_literal
    )
    _require_same_file(
        pixal_batch_path,
        direct_context["adopted_batch_path"],
        "direct authority adopted Pixal batch",
    )
    if (
        pixal_batch["batch_sha256"]
        != direct_authority["adopted_batch"]["batch_sha256"]
        or pixal_batch["batch_sha256"]
        != direct_context["adopted_batch"]["batch_sha256"]
    ):
        raise contracts.ContractError(
            "direct authority points to a different adopted Pixal batch"
        )

    attempt = direct_context["attempt"]
    instance_id = direct_authority["instance_id"]
    if (
        attempt.get("instance_id") != instance_id
        or attempt.get("request_sha256") != direct_authority["request_sha256"]
        or attempt.get("profile_schema_id")
        != direct_authority["profile_schema_id"]
    ):
        raise contracts.ContractError(
            "direct authority adopted attempt identity changed"
        )
    if (physical_profile_authority_request_batch_path is None) != (
        expected_physical_profile_authority_request_batch_sha256 is None
    ):
        raise contracts.ContractError("physical-profile authority pair is incomplete")
    physical_profile_authority = None
    if physical_profile_authority_request_batch_path is not None:
        physical_profile_authority = _authenticate_physical_profile_authority_request_batch(
            physical_profile_authority_request_batch_path,
            expected_physical_profile_authority_request_batch_sha256,
            taxonomy=direct_authority["taxonomy"],
            sampled_attributes=attempt["sampled_attributes"],
            target_physical_profile=attempt["target_physical_profile"],
        )

    decision_batch_path, decision_batch, raw_decisions = load_decision_batch(
        decision_batch_path
    )
    if set(raw_decisions) != {instance_id}:
        raise contracts.ContractError(
            "direct geometry registration requires exactly one matching raw "
            "static decision"
        )
    raw_decision = raw_decisions[instance_id]
    raw_payload = raw_decision.get("payload")
    raw_review_record = raw_decision.get("static_review")
    raw_review = (
        raw_review_record.get("payload")
        if isinstance(raw_review_record, Mapping)
        else None
    )
    raw_review_path = (
        raw_review_record.get("path")
        if isinstance(raw_review_record, Mapping)
        else None
    )
    if (
        not isinstance(raw_payload, Mapping)
        or raw_payload.get("decision") != "approved_for_lod_and_binding"
        or raw_payload.get("state_classification") != "research_candidate"
        or raw_payload.get("formal_dataset_registration_authorized") is not False
        or raw_payload.get("next_gate") != "lod_then_species_rig_binding"
        or not isinstance(raw_payload.get("checks"), Mapping)
        or set(raw_payload["checks"]) != static_decisions.CHECK_FIELDS
        or any(value is not True for value in raw_payload["checks"].values())
        or not isinstance(raw_review, Mapping)
        or not isinstance(raw_review_path, Path)
        or raw_review.get("instance_id") != instance_id
        or raw_review.get("request_sha256")
        != direct_authority["request_sha256"]
        or raw_review.get("profile_schema_id")
        != direct_authority["profile_schema_id"]
        or contracts.canonical_json(raw_review.get("sampled_attributes"))
        != contracts.canonical_json(attempt["sampled_attributes"])
        or contracts.canonical_json(raw_review.get("target_physical_profile"))
        != contracts.canonical_json(attempt["target_physical_profile"])
        or raw_review.get("pixal_output") != attempt["output"]
    ):
        raise contracts.ContractError(
            "raw static approval/direct authority identity changed"
        )

    adopted_root = pixal_batch_path.parent
    adopted_raw_path = _verified_file_record(
        attempt["output"],
        "adopted Pixel3D raw GLB",
        root=adopted_root,
    )
    adopted_manifest_path = _verified_file_record(
        attempt["attempt_manifest"],
        "adopted Pixel3D attempt manifest",
        root=adopted_root,
    )
    adopted_input_path = _verified_file_record(
        attempt["pixal_input"],
        "adopted Pixel3D input RGBA",
        root=adopted_root,
    )
    bundle_sources = direct_context["adoption_context"].get("bundle_sources")
    if not isinstance(bundle_sources, Mapping):
        raise contracts.ContractError(
            "direct adoption source byte authorities are missing"
        )
    try:
        original_raw_path = Path(bundle_sources["pixal_raw_glb"])
        original_manifest_path = Path(bundle_sources["pixal_attempt_manifest"])
        original_input_path = Path(bundle_sources["pixal_input_rgba"])
    except (KeyError, TypeError) as error:
        raise contracts.ContractError(
            "direct adoption source byte authorities are incomplete"
        ) from error
    _authenticated_byte_copy_pair(
        original_raw_path,
        adopted_raw_path,
        "original/adopted Pixel3D raw GLB",
    )
    _authenticated_byte_copy_pair(
        original_manifest_path,
        adopted_manifest_path,
        "original/adopted Pixel3D attempt manifest",
    )
    _authenticated_byte_copy_pair(
        original_input_path,
        adopted_input_path,
        "original/adopted Pixel3D input RGBA",
    )

    review_root = Path(decision_batch["static_review_batch"]["path"]).resolve().parent
    review_reference_path = _verified_file_record(
        raw_review["reference_rgba"],
        "raw static review source reference",
        root=review_root,
    )
    raw_contact_path = _verified_file_record(
        raw_review["contact_sheet"],
        "raw static review contact sheet",
        root=review_root,
    )
    _authenticated_byte_copy_pair(
        original_input_path,
        review_reference_path,
        "original/raw-static-review Pixel3D input RGBA",
    )

    closure_literal = Path(geometry_closure_path).absolute()
    geometry_closure_path = closure_literal.resolve()
    if (
        closure_literal.is_symlink()
        or not geometry_closure_path.is_file()
        or geometry_closure_path.stat().st_size <= 0
        or _sha256_file(geometry_closure_path)
        != expected_geometry_closure_sha256
    ):
        raise contracts.ContractError(
            "geometry closure changed from its expected file SHA-256"
        )
    # Local import avoids the geometry closure module's intentional import of
    # this registry module for raw decision replay.
    from tools import publish_generated_animal_geometry_closure as geometry_closures

    try:
        closure_replay = geometry_closures.load_geometry_closure_v2(
            geometry_closure_path,
            expected_manifest_sha256=expected_geometry_closure_sha256,
            expected_instance_id=instance_id,
            expected_raw_pixal_glb=original_raw_path,
            expected_pixal_manifest=original_manifest_path,
            expected_raw_static_decision_batch=decision_batch_path,
            expected_raw_static_decision=raw_decision["path"],
        )
    except geometry_closures.GeometryClosureError as error:
        raise contracts.ContractError(
            f"direct geometry closure strict replay failed: {error}"
        ) from error
    closure_manifest = closure_replay.get("manifest")
    closure_paths = closure_replay.get("paths")
    if not isinstance(closure_manifest, Mapping) or not isinstance(
        closure_paths, Mapping
    ):
        raise contracts.ContractError(
            "direct geometry closure replay result is incomplete"
        )
    try:
        closure_raw_path = Path(closure_paths["raw_pixal_glb"])
        closure_manifest_path = Path(closure_paths["pixal_manifest"])
        closure_reference_path = Path(closure_paths["source_reference"])
        closure_batch_path = Path(closure_paths["raw_static_decision_batch"])
        closure_decision_path = Path(closure_paths["raw_static_decision"])
        closure_review_path = Path(closure_paths["raw_static_review"])
        repair_manifest_path = Path(closure_paths["repair_manifest"])
        repaired_glb_path = Path(closure_paths["repaired_glb"])
        geometry_audit_path = Path(closure_paths["geometry_audit"])
        clay_render_manifest_path = Path(
            closure_paths["clay_render_manifest"]
        )
        clay_contact_path = Path(closure_paths["clay_contact_sheet"])
    except (KeyError, TypeError) as error:
        raise contracts.ContractError(
            "direct geometry closure replay paths are incomplete"
        ) from error
    for observed, expected, label in (
        (closure_raw_path, original_raw_path, "closure raw Pixel3D GLB"),
        (
            closure_manifest_path,
            original_manifest_path,
            "closure Pixel3D attempt manifest",
        ),
        (
            closure_reference_path,
            review_reference_path,
            "closure raw static source reference",
        ),
        (
            closure_batch_path,
            decision_batch_path,
            "closure raw static decision batch",
        ),
        (
            closure_decision_path,
            raw_decision["path"],
            "closure raw static decision",
        ),
        (
            closure_review_path,
            raw_review_path,
            "closure raw static review",
        ),
    ):
        _require_same_file(observed, expected, label)
    _authenticated_byte_copy_pair(
        closure_raw_path,
        adopted_raw_path,
        "closure/adopted Pixel3D raw GLB",
    )
    _authenticated_byte_copy_pair(
        closure_manifest_path,
        adopted_manifest_path,
        "closure/adopted Pixel3D attempt manifest",
    )
    _authenticated_byte_copy_pair(
        closure_reference_path,
        adopted_input_path,
        "closure/adopted Pixel3D input RGBA",
    )

    bounded = closure_manifest.get("bounded_repair")
    inherited = closure_manifest.get("inherited_static_judgment")
    downstream = closure_manifest.get("downstream")
    removed = (
        bounded.get("removed_exact_position_degenerate_triangle_count")
        if isinstance(bounded, Mapping)
        else None
    )
    byte_identical = (
        bounded.get("output_byte_identical_to_raw")
        if isinstance(bounded, Mapping)
        else None
    )
    if (
        closure_manifest.get("schema") != geometry_closures.SCHEMA
        or closure_manifest.get("state_classification") != "research_candidate"
        or closure_manifest.get("formal_dataset_registration_authorized") is not False
        or not isinstance(bounded, Mapping)
        or bounded.get("implementation_contract")
        != geometry_closures.oriented_repair.IMPLEMENTATION_CONTRACT
        or bounded.get("mutation_class") != geometry_closures.MUTATION_CLASS
        or isinstance(removed, bool)
        or not isinstance(removed, int)
        or removed < 0
        or byte_identical is not (removed == 0)
        or not isinstance(inherited, Mapping)
        or inherited.get("authority") != "canonical_raw_static_approval_v1"
        or inherited.get("decision_sha256")
        != raw_payload["decision_sha256"]
        or inherited.get("checks") != raw_payload["checks"]
        or inherited.get("inheritance_scope")
        != geometry_closures.INHERITANCE_SCOPE
        or inherited.get("new_human_approval_created") is not False
        or inherited.get("clay_render_human_approval_claimed") is not False
        or not isinstance(downstream, Mapping)
        or downstream.get("tokenrig_entry_authorized") is not True
        or downstream.get("tokenrig_execution_performed") is not False
        or downstream.get("formal_dataset_registration_authorized") is not False
    ):
        raise contracts.ContractError(
            "direct geometry closure exceeded its research-only inherited "
            "approval boundary"
        )
    raw_hash = _sha256_file(closure_raw_path)
    repaired_hash = _sha256_file(repaired_glb_path)
    if (
        (removed == 0 and raw_hash != repaired_hash)
        or (removed > 0 and raw_hash == repaired_hash)
    ):
        raise contracts.ContractError(
            "direct geometry closure repair bytes contradict its no-op/filter claim"
        )

    artifacts = {
        "source_reference_2d": spear_artifact(closure_reference_path),
        "pixal_input_rgba": spear_artifact(adopted_input_path),
        "pixal_raw_glb": spear_artifact(closure_raw_path),
        "pixal_attempt_manifest": spear_artifact(closure_manifest_path),
        "raw_static_review_manifest": spear_artifact(raw_review_path),
        "raw_static_contact_sheet": spear_artifact(raw_contact_path),
        "raw_static_decision": spear_artifact(raw_decision["path"]),
        "raw_static_decision_batch": spear_artifact(decision_batch_path),
        "derived_repaired_glb": spear_artifact(repaired_glb_path),
        "derived_geometry_closure": spear_artifact(geometry_closure_path),
        "derived_repair_manifest": spear_artifact(repair_manifest_path),
        "derived_geometry_audit": spear_artifact(geometry_audit_path),
        "derived_clay_render_manifest": spear_artifact(
            clay_render_manifest_path
        ),
        "derived_clay_contact_sheet": spear_artifact(clay_contact_path),
        "direct_source_authority": spear_artifact(direct_authority_path),
        "adopted_pixal_batch": spear_artifact(pixal_batch_path),
        "direct_adoption_spec": spear_artifact(
            direct_context["source_spec_path"]
        ),
    }
    effective_direct_context = copy.deepcopy(direct_context)
    if physical_profile_authority is not None:
        (
            physical_authority_path,
            physical_authority_request,
            restored_physical_profile,
        ) = physical_profile_authority
        artifacts[PHYSICAL_PROFILE_AUTHORITY_ARTIFACT_ROLE] = spear_artifact(
            physical_authority_path)
        effective_direct_context["attempt"]["target_physical_profile"] = (
            restored_physical_profile)
        models = effective_direct_context["adopted_batch"]["models"]
        if PHYSICAL_PROFILE_AUTHORITY_REQUEST_MODEL in models:
            raise contracts.ContractError("physical authority model role collision")
        models[PHYSICAL_PROFILE_AUTHORITY_REQUEST_MODEL] = (
            physical_authority_request["request_sha256"])
    source_asset = _build_direct_source_asset_v2(
        authority=direct_authority,
        direct_context=effective_direct_context,
        artifacts=artifacts,
        canonical_raw_decision=raw_payload,
        bounded_geometry_closure=closure_manifest,
    )

    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(f"refusing to replace output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.",
            suffix=".staging",
            dir=output_root.parent,
        )
    )
    try:
        destination = staging / "source_assets" / f"{instance_id}.json"
        contracts.write_json_no_replace(destination, source_asset)
        entry = {
            "asset_id": instance_id,
            "profile_schema_id": direct_authority["profile_schema_id"],
            "request_sha256": direct_authority["request_sha256"],
            "sampled_attributes": copy.deepcopy(attempt["sampled_attributes"]),
            "attribute_evidence": copy.deepcopy(
                raw_payload["attribute_evidence"]
            ),
            "source_asset": {
                "path": destination.relative_to(staging).as_posix(),
                "sha256": _sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            },
            "state_classification": "research_candidate",
            "next_gate": "lod_then_species_rig_binding",
        }
        registry: dict[str, Any] = {
            "schema": DIRECT_GEOMETRY_REGISTRY_SCHEMA,
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "direct_source_authority": {
                "path": str(direct_authority_path),
                "sha256": expected_direct_source_authority_sha256,
                "size_bytes": direct_authority_path.stat().st_size,
                "authority_sha256": direct_authority["authority_sha256"],
            },
            "pixal_batch": {
                "path": str(pixal_batch_path),
                "sha256": _sha256_file(pixal_batch_path),
                "batch_sha256": pixal_batch["batch_sha256"],
            },
            "static_decision_batch": {
                "path": str(decision_batch_path),
                "sha256": _sha256_file(decision_batch_path),
                "decision_batch_sha256": decision_batch[
                    "decision_batch_sha256"
                ],
            },
            "geometry_closure": {
                "path": str(geometry_closure_path),
                "sha256": expected_geometry_closure_sha256,
                "manifest_sha256": closure_manifest["manifest_sha256"],
            },
            "source_asset_count": 1,
            "source_assets": [entry],
            "automatic_checks": copy.deepcopy(
                DIRECT_GEOMETRY_REGISTRY_AUTOMATIC_CHECKS
            ),
        }
        registry["registry_sha256"] = _hash_without(
            registry,
            "registry_sha256",
        )
        contracts.write_json_no_replace(
            staging / "registry_manifest.json",
            registry,
        )
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError(
                "animal registry output appeared concurrently"
            )
        os.rename(staging, output_root)
        return output_root / "registry_manifest.json"
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def register_derived(
    preflight_path: Path | None,
    pixal_batch_path: Path,
    decision_batch_path: Path,
    derived_decision_path: Path,
    expected_derived_decision_sha256: str,
    output_root: Path,
    *,
    frozen_historical_preflight: bool = False,
    direct_source_authority_path: Path | None = None,
    expected_direct_source_authority_sha256: str | None = None,
) -> Path:
    direct_mode = direct_source_authority_path is not None
    if direct_mode != (expected_direct_source_authority_sha256 is not None):
        raise contracts.ContractError(
            "direct source authority path and expected file SHA-256 must be supplied together"
        )
    if direct_mode and preflight_path is not None:
        raise contracts.ContractError(
            "direct derived registration cannot claim a preflight authority"
        )
    direct_authority_path: Path | None = None
    direct_authority: Mapping[str, Any] | None = None
    direct_context: Mapping[str, Any] | None = None
    if direct_mode:
        (
            direct_authority_path,
            direct_authority,
            direct_context,
        ) = direct_adopter.load_direct_source_authority(
            direct_source_authority_path,
            expected_sha256=expected_direct_source_authority_sha256,
        )
        supplied_batch_literal = Path(pixal_batch_path).absolute()
        if supplied_batch_literal.is_symlink() or not supplied_batch_literal.is_file():
            raise contracts.ContractError("adopted Pixal batch is missing or unsafe")
        pixal_batch_path, pixal_batch = direct_adopter.load_adopted_batch(
            supplied_batch_literal
        )
        try:
            same_batch = os.path.samefile(
                pixal_batch_path,
                direct_context["adopted_batch_path"],
            )
        except OSError as error:
            raise contracts.ContractError(
                "cannot compare direct authority adopted batch"
            ) from error
        if (
            not same_batch
            or pixal_batch["batch_sha256"]
            != direct_authority["adopted_batch"]["batch_sha256"]
        ):
            raise contracts.ContractError(
                "direct authority points to a different adopted Pixal batch"
            )
        attempt = direct_context["attempt"]
        instance_id = direct_authority["instance_id"]
        attempts = {instance_id: attempt}
        request = {
            "instance_id": instance_id,
            "profile_schema_id": direct_authority["profile_schema_id"],
            "profile_sha256": direct_authority["profile_sha256"],
            "request_sha256": direct_authority["request_sha256"],
            "taxonomy": copy.deepcopy(direct_authority["taxonomy"]),
            "fixed_attributes": copy.deepcopy(direct_authority["fixed_attributes"]),
            "sampled_attributes": copy.deepcopy(attempt["sampled_attributes"]),
            "target_physical_profile": copy.deepcopy(
                attempt["target_physical_profile"]
            ),
        }
        preflight = None
        profile = None
        pixal_inputs_path = Path(direct_context["source_spec_path"]).resolve()
        input_jobs: dict[str, Any] = {}
        requests: dict[str, Any] = {}
        profiles: dict[str, Any] = {}
    else:
        if preflight_path is None:
            raise contracts.ContractError(
                "legacy derived registration requires a preflight"
            )
        (
            preflight_path,
            preflight,
            requests,
            profiles,
            pixal_batch_path,
            pixal_batch,
            pixal_inputs_path,
            _pixal_inputs_manifest,
            input_jobs,
            attempts,
        ) = _load_registration_context(
            preflight_path,
            pixal_batch_path,
            frozen_historical_preflight=frozen_historical_preflight,
        )
        request = None
        profile = None
    decision_batch_path, decision_batch, raw_decisions = load_decision_batch(
        decision_batch_path
    )
    if set(raw_decisions) != set(attempts):
        raise contracts.ContractError(
            "derived registration requires complete preserved raw static decisions"
        )

    derived_decision_literal = Path(derived_decision_path).absolute()
    derived_decision_path = derived_decision_literal.resolve()
    if (
        derived_decision_literal.is_symlink()
        or not derived_decision_path.is_file()
        or _sha256_file(derived_decision_path)
        != expected_derived_decision_sha256
    ):
        raise contracts.ContractError("derived static decision changed")
    derived_decision = derived_static_decisions.validate_decision(
        contracts.load_json(derived_decision_path)
    )
    if derived_decision["decision"] != derived_static_decisions.APPROVED:
        raise contracts.ContractError(
            "derived static decision does not authorize source registration"
        )
    instance_id = derived_decision["instance_id"]
    if not direct_mode:
        request = requests.get(instance_id)
        profile = (
            profiles.get(request["profile_schema_id"])
            if isinstance(request, Mapping)
            else None
        )
    attempt = attempts.get(instance_id)
    input_job = input_jobs.get(instance_id)
    raw_decision = raw_decisions.get(instance_id)
    required_authorities = (
        (request, attempt, raw_decision)
        if direct_mode
        else (request, profile, attempt, input_job, raw_decision)
    )
    if not all(isinstance(value, Mapping) for value in required_authorities):
        raise contracts.ContractError(
            "derived static decision lacks its frozen request/Pixal authority"
        )

    review_record = derived_decision["review_binding"]["review_file"]
    review_path = _verified_file_record(
        review_record,
        "derived static review",
    )
    review = derived_review_contract.validate_review(
        contracts.load_json(review_path)
    )
    authenticated_review_artifact_count = (
        derived_static_decisions._authenticate_review_artifacts(
            review,
            review_path.parent,
        )
    )
    preserved_raw_decision = (
        derived_static_decisions.stable._validate_raw_static_decision(
            review,
            review_path.parent,
        )
    )
    derived_raw_field = (
        "raw_static_decision"
        if "raw_static_decision" in derived_decision
        else "raw_static_rejection"
    )
    if (
        authenticated_review_artifact_count
        != derived_decision["authenticated_review_artifact_count"]
        or contracts.canonical_json(preserved_raw_decision)
        != contracts.canonical_json(
            derived_decision[derived_raw_field]
        )
    ):
        raise contracts.ContractError(
            "derived decision no longer matches all frozen review artifacts"
        )
    identity = review["instance_identity"]
    if (
        review["review_sha256"]
        != derived_decision["review_binding"]["internal_review_sha256"]
        or (
            direct_mode
            and review.get("schema")
            != derived_review_contract.DIRECT_REVIEW_SCHEMA
        )
        or (
            not direct_mode
            and review.get("schema")
            == derived_review_contract.DIRECT_REVIEW_SCHEMA
        )
        or identity["instance_id"] != instance_id
        or identity["profile_schema_id"] != request["profile_schema_id"]
        or identity["profile_sha256"] != request["profile_sha256"]
        or identity["request_sha256"] != request["request_sha256"]
        or contracts.canonical_json(identity["sampled_attributes"])
        != contracts.canonical_json(request["sampled_attributes"])
        or contracts.canonical_json(identity["target_physical_profile"])
        != contracts.canonical_json(request["target_physical_profile"])
    ):
        raise contracts.ContractError(
            "derived static review/canonical request identity changed"
        )
    raw_authority = review["source_authorities"]
    if direct_mode:
        review_pixal_batch = raw_authority.get("pixal_batch")
        review_direct_authority = raw_authority.get(
            "direct_source_authority"
        )
        if (
            not isinstance(review_pixal_batch, Mapping)
            or _verified_file_record(
                review_pixal_batch.get("file"),
                "derived review adopted Pixal batch",
            )
            != pixal_batch_path
            or review_pixal_batch.get("batch_sha256")
            != pixal_batch.get("batch_sha256")
            or not isinstance(review_direct_authority, Mapping)
            or set(review_direct_authority)
            != {"path", "sha256", "size_bytes", "authority_sha256"}
            or _verified_file_record(
                {
                    name: review_direct_authority[name]
                    for name in ("path", "sha256", "size_bytes")
                },
                "derived review direct source authority",
            )
            != direct_authority_path
            or review_direct_authority.get("authority_sha256")
            != direct_authority.get("authority_sha256")
        ):
            raise contracts.ContractError(
                "derived review rebound its direct source/adopted batch authority"
            )
    _reauthenticate_review_raw_authorities(
        raw_authority=raw_authority,
        raw_decision=raw_decision,
        decision_batch_path=decision_batch_path,
        decision_batch=decision_batch,
    )
    raw_pixal_path = _verified_file_record(
        raw_authority["raw_pixal_glb"],
        "derived review raw Pixal GLB",
    )
    reviewed_reference_path = _verified_file_record(
        raw_authority["reference_2d"],
        "derived review 2D reference",
    )
    if direct_mode:
        input_reference = attempt["pixal_input"]
        input_reference_path = _verified_file_record(
            input_reference,
            "adopted Pixel3D RGBA reference",
            root=pixal_batch_path.parent,
        )
    else:
        input_reference = input_job.get("reference", {}).get("source")
        input_reference_path = _verified_file_record(
            input_reference,
            "Pixal input 2D reference",
        )
    if (
        raw_pixal_path
        != (pixal_batch_path.parent / attempt["output"]["path"]).resolve()
        or raw_authority["raw_pixal_glb"]["sha256"]
        != attempt["output"]["sha256"]
        or reviewed_reference_path != input_reference_path
        or raw_authority["reference_2d"]["sha256"]
        != input_reference["sha256"]
        or raw_authority["reference_2d"]["size_bytes"]
        != input_reference["size_bytes"]
    ):
        raise contracts.ContractError(
            "derived review did not preserve its Pixal source authorities"
        )

    derived_geometry = review["derived_geometry"]
    repaired_glb_path = _verified_file_record(
        derived_geometry["repaired_glb"],
        "derived repaired GLB",
    )
    geometry_closure_path = _verified_file_record(
        derived_geometry["geometry_closure"],
        "derived geometry closure",
    )
    repair_manifest_path = _verified_file_record(
        derived_geometry["repair_manifest"],
        "derived repair manifest",
    )
    geometry_audit_path = _verified_file_record(
        derived_geometry["independent_geometry_audit"],
        "derived geometry audit",
    )
    pbr_contact_path = _verified_file_record(
        review["evidence"]["pbr_five_view"]["contact_sheet"],
        "derived PBR contact sheet",
        root=review_path.parent,
    )
    clay_contact_path = _verified_file_record(
        review["evidence"]["clay_five_view"]["contact_sheet"],
        "derived clay contact sheet",
    )

    pixal_root = pixal_batch_path.parent
    attempt_manifest_path = (
        pixal_root / attempt["attempt_manifest"]["path"]
    ).resolve()
    pixal_input_path = (
        input_reference_path
        if direct_mode
        else Path(attempt["pixal_input"]["path"]).resolve()
    )
    if (
        not attempt_manifest_path.is_file()
        or attempt_manifest_path.stat().st_size
        != attempt["attempt_manifest"]["size_bytes"]
        or _sha256_file(attempt_manifest_path)
        != attempt["attempt_manifest"]["sha256"]
    ):
        raise contracts.ContractError("Pixal attempt manifest changed")
    _reauthenticate_bounded_derived_geometry(
        review=review,
        raw_pixal_path=raw_pixal_path,
        reviewed_reference_path=reviewed_reference_path,
        raw_decision_path=raw_decision["path"],
        raw_attempt_manifest_path=attempt_manifest_path,
        repaired_glb_path=repaired_glb_path,
        geometry_closure_path=geometry_closure_path,
        repair_manifest_path=repair_manifest_path,
        geometry_audit_path=geometry_audit_path,
        raw_decision_batch_path=decision_batch_path,
        pixal_input_path=pixal_input_path,
    )
    artifacts = {
        (
            "source_reference_2d"
            if direct_mode
            else "flux2_candidate_image"
        ): spear_artifact(reviewed_reference_path),
        "pixal_input_rgba": spear_artifact(pixal_input_path),
        "pixal_raw_glb": spear_artifact(raw_pixal_path),
        "pixal_attempt_manifest": spear_artifact(attempt_manifest_path),
        "raw_static_decision": spear_artifact(raw_decision["path"]),
        "derived_repaired_glb": spear_artifact(repaired_glb_path),
        "derived_geometry_closure": spear_artifact(geometry_closure_path),
        "derived_repair_manifest": spear_artifact(repair_manifest_path),
        "derived_geometry_audit": spear_artifact(geometry_audit_path),
        "derived_static_review_manifest": spear_artifact(review_path),
        "derived_static_decision": spear_artifact(derived_decision_path),
        "derived_pbr_contact_sheet": spear_artifact(pbr_contact_path),
        "derived_clay_contact_sheet": spear_artifact(clay_contact_path),
    }
    if direct_mode:
        artifacts.update(
            {
                "direct_source_authority": spear_artifact(
                    direct_authority_path
                ),
                "adopted_pixal_batch": spear_artifact(pixal_batch_path),
                "direct_adoption_spec": spear_artifact(pixal_inputs_path),
            }
        )
        source_asset = _build_direct_source_asset_v2(
            authority=direct_authority,
            direct_context=direct_context,
            artifacts=artifacts,
        )
    else:
        artifacts["pixal_inputs_manifest"] = spear_artifact(
            pixal_inputs_path
        )
        source_asset = contracts.build_source_asset_v2(
            request,
            artifacts=artifacts,
            physical_measurements={"status": "pending"},
            provenance={
                "attempt_id": f"derived_static_{attempt['execution_job_id']}",
                "request_sha256": request["request_sha256"],
                "models": copy.deepcopy(
                    request["generation_plan"]["model_revisions"]
                ),
            },
            rights={
                "status": "review_required",
                "licenses": license_records(),
                "blockers": [
                    "legacy_reference_provenance_unknown",
                    "physical_target_reference_provisional",
                    "pixal_research_dependency_export_review_required",
                    "dino_snapshot_origin_review_required",
                ],
            },
            qa={
                "reference_2d": "passed",
                "static_mesh": "passed",
                "binding": "pending",
                "walking": "pending",
                "idle": "pending",
                "ue_import_readback": "pending",
                "apartment_media": "pending",
                "audio": "pending",
            },
            state_classification="research_candidate",
        )
        contracts.validate_source_asset_v2(
            source_asset,
            request=request,
            profile=profile,
        )

    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(f"refusing to replace output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.",
            suffix=".staging",
            dir=output_root.parent,
        )
    )
    try:
        destination = staging / "source_assets" / f"{instance_id}.json"
        contracts.write_json_no_replace(destination, source_asset)
        entry = {
            "asset_id": instance_id,
            "profile_schema_id": request["profile_schema_id"],
            "request_sha256": request["request_sha256"],
            "sampled_attributes": request["sampled_attributes"],
            "attribute_evidence": derived_decision["attribute_evidence"],
            "source_asset": {
                "path": destination.relative_to(staging).as_posix(),
                "sha256": _sha256_file(destination),
                "size_bytes": destination.stat().st_size,
            },
            "state_classification": "research_candidate",
            "next_gate": "lod_then_species_rig_binding",
        }
        registry: dict[str, Any] = {
            "schema": DERIVED_REGISTRY_SCHEMA,
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "pixal_batch": {
                "path": str(pixal_batch_path),
                "sha256": _sha256_file(pixal_batch_path),
                "batch_sha256": pixal_batch["batch_sha256"],
            },
            "static_decision_batch": {
                "path": str(decision_batch_path),
                "sha256": _sha256_file(decision_batch_path),
                "decision_batch_sha256": decision_batch["decision_batch_sha256"],
            },
            "derived_static_decisions": [
                {
                    "path": str(derived_decision_path),
                    "sha256": expected_derived_decision_sha256,
                    "decision_sha256": derived_decision["decision_sha256"],
                }
            ],
            "source_asset_count": 1,
            "source_assets": [entry],
            "automatic_checks": copy.deepcopy(
                DIRECT_DERIVED_REGISTRY_AUTOMATIC_CHECKS
                if direct_mode
                else DERIVED_REGISTRY_AUTOMATIC_CHECKS
            ),
        }
        if direct_mode:
            registry["direct_source_authority"] = {
                "path": str(direct_authority_path),
                "sha256": expected_direct_source_authority_sha256,
                "size_bytes": direct_authority_path.stat().st_size,
                "authority_sha256": direct_authority["authority_sha256"],
            }
        else:
            registry["preflight"] = {
                "path": str(preflight_path),
                "sha256": _sha256_file(preflight_path),
                "preflight_sha256": preflight["preflight_sha256"],
                "validation_mode": (
                    "frozen_historical_preflight_v1"
                    if frozen_historical_preflight
                    else "current_exact_rebuild"
                ),
            }
        registry["registry_sha256"] = _hash_without(
            registry,
            "registry_sha256",
        )
        contracts.write_json_no_replace(
            staging / "registry_manifest.json",
            registry,
        )
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError(
                "animal registry output appeared concurrently"
            )
        os.rename(staging, output_root)
        return output_root / "registry_manifest.json"
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    authority = parser.add_mutually_exclusive_group(required=True)
    authority.add_argument("--preflight", type=Path)
    authority.add_argument("--direct-source-authority", type=Path)
    parser.add_argument("--expected-direct-source-authority-sha256")
    parser.add_argument("--pixal-batch", required=True, type=Path)
    parser.add_argument("--static-decision-batch", required=True, type=Path)
    parser.add_argument("--derived-static-decision", type=Path)
    parser.add_argument("--expected-derived-static-decision-sha256")
    parser.add_argument("--geometry-closure", type=Path)
    parser.add_argument("--expected-geometry-closure-sha256")
    parser.add_argument("--physical-profile-authority-request-batch", type=Path)
    parser.add_argument("--expected-physical-profile-authority-request-batch-sha256")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--frozen-historical-preflight",
        action="store_true",
        help=(
            "Validate an immutable older preflight as frozen evidence while "
            "allowing only additive fields in the current deterministic rebuild."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        if (args.direct_source_authority is None) != (
            args.expected_direct_source_authority_sha256 is None
        ):
            raise contracts.ContractError(
                "direct source authority path and expected file SHA-256 must be supplied together"
            )
        if (args.derived_static_decision is None) != (
            args.expected_derived_static_decision_sha256 is None
        ):
            raise contracts.ContractError(
                "derived static decision path and expected SHA-256 must be supplied together"
            )
        if (args.geometry_closure is None) != (
            args.expected_geometry_closure_sha256 is None
        ):
            raise contracts.ContractError(
                "geometry closure path and expected SHA-256 must be supplied together"
            )
        authority_batch = args.physical_profile_authority_request_batch
        authority_batch_sha = (
            args.expected_physical_profile_authority_request_batch_sha256)
        if (authority_batch is None) != (authority_batch_sha is None):
            raise contracts.ContractError("physical-profile authority pair is incomplete")
        if authority_batch is not None and args.geometry_closure is None:
            raise contracts.ContractError("physical-profile authority requires geometry")
        if (
            args.derived_static_decision is not None
            and args.geometry_closure is not None
        ):
            raise contracts.ContractError(
                "derived static decision and direct geometry closure modes are "
                "mutually exclusive"
            )
        if args.geometry_closure is not None:
            if args.direct_source_authority is None:
                raise contracts.ContractError(
                    "direct geometry registration requires a direct source authority"
                )
            if args.frozen_historical_preflight:
                raise contracts.ContractError(
                    "direct geometry registration cannot claim a frozen preflight"
                )
            manifest = register_direct_geometry(
                args.direct_source_authority,
                args.expected_direct_source_authority_sha256,
                args.pixal_batch,
                args.static_decision_batch,
                args.geometry_closure,
                args.expected_geometry_closure_sha256,
                args.output_root,
                physical_profile_authority_request_batch_path=authority_batch,
                expected_physical_profile_authority_request_batch_sha256=authority_batch_sha,
            )
        elif args.derived_static_decision is None:
            if args.preflight is None:
                raise contracts.ContractError(
                    "direct source authority requires derived or direct geometry "
                    "registration"
                )
            manifest = register(
                args.preflight,
                args.pixal_batch,
                args.static_decision_batch,
                args.output_root,
                frozen_historical_preflight=args.frozen_historical_preflight,
            )
        else:
            manifest = register_derived(
                args.preflight,
                args.pixal_batch,
                args.static_decision_batch,
                args.derived_static_decision,
                args.expected_derived_static_decision_sha256,
                args.output_root,
                frozen_historical_preflight=args.frozen_historical_preflight,
                direct_source_authority_path=args.direct_source_authority,
                expected_direct_source_authority_sha256=(
                    args.expected_direct_source_authority_sha256
                ),
            )
        payload = contracts.load_json(manifest)
    except (contracts.ContractError, OSError) as error:
        print(f"CONTROLLED_ANIMAL_SOURCE_ASSET_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_ANIMAL_SOURCE_ASSET_OK "
        f"assets={payload['source_asset_count']} output={manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
