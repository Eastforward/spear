#!/usr/bin/env python3
"""Publish and revalidate an immutable qualified Route-2 human candidate pointer."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from tools import route2_human_contract_common as common
from tools import route2_human_instance_contract as instance_contract
from tools import route2_human_static_decision as static_decision


SCHEMA = "route2_human_qualified_candidate_v1"
FILENAME = "qualified_candidate_v1.json"
RETARGET_DIRNAME = "retarget_v1"
DYNAMIC_REVIEW_DIRNAME = "dynamic_review_v1"
RETARGET_FILES = (
    "animated.blend",
    "walking.glb",
    "standing_idle.glb",
    "retarget_metrics.json",
    "retarget_manifest.json",
)
BRANCH_MANIFEST_SCHEMAS = {
    "direct": {"pixal_tokenrig_canary_v1", "pixal_tokenrig_recovery_v1"},
    "fitted_skeleton": {"pixal_tokenrig_fitted_skeleton_v1"},
    "sanitized_weights": {"pixal_tokenrig_sanitized_weights_v1"},
}
FEMALE_BASE_ASSET_ID = "rocketbox_female_adult_01"


class QualificationError(RuntimeError):
    """The candidate is incomplete, rejected, stale, non-canonical, or mutable."""


def _instance(contract_path: Path) -> tuple[dict[str, Any], Path, Path]:
    try:
        contract = instance_contract.validate_instance_contract(contract_path)
    except instance_contract.InstanceContractError as error:
        raise QualificationError(f"instance contract is invalid: {error}") from error
    path = common.absolute(contract_path)
    return contract, path, Path(contract["canonical_output_root"])


def _branch(
    contract: Mapping[str, Any], branch_id: str
) -> tuple[dict[str, str], Path]:
    try:
        descriptor = instance_contract.branch_descriptor(contract, branch_id)
        root = instance_contract.resolve_branch_root(contract, branch_id)
    except instance_contract.InstanceContractError as error:
        raise QualificationError(f"final branch is invalid: {error}") from error
    if branch_id not in contract["allowed_branch_dag"]["finalizable_branches"]:
        raise QualificationError(f"branch is not finalizable: {branch_id}")
    return descriptor, common.require_real_directory(
        root, "actual final branch", QualificationError
    )


def _readonly_inventory_record(path: Path, output_root: Path, description: str) -> dict[str, Any]:
    return common.file_record(
        path,
        root=output_root,
        description=description,
        error_type=QualificationError,
        require_mode=0o444,
    )


def _validate_readonly_bundle(
    root: Path,
    *,
    expected_names: set[str],
    output_root: Path,
    description: str,
) -> list[dict[str, Any]]:
    root = common.require_real_directory(
        root, description, QualificationError, mode=0o555
    )
    actual_names = {path.name for path in root.iterdir()}
    if actual_names != expected_names:
        raise QualificationError(f"{description} inventory is incomplete or unexpected")
    records = []
    for name in sorted(expected_names):
        records.append(
            _readonly_inventory_record(
                root / name, output_root, f"{description} artifact {name}"
            )
        )
    return records


def _validate_retarget_bundle(
    branch_root: Path,
    *,
    output_root: Path,
    contract: Mapping[str, Any],
) -> tuple[Path, list[dict[str, Any]]]:
    from tools import blender_render_tokenrig_human_review as review_renderer
    from tools import blender_retarget_rocketbox_to_tokenrig as retarget_owner

    retarget_root = branch_root / RETARGET_DIRNAME
    records = _validate_readonly_bundle(
        retarget_root,
        expected_names=set(RETARGET_FILES),
        output_root=output_root,
        description="retarget bundle",
    )
    records_by_name = {Path(record["path"]).name: record for record in records}
    manifest_path = retarget_root / "retarget_manifest.json"
    manifest, parsed_manifest_record = common.load_json_mapping_record(
        manifest_path,
        root=output_root,
        description="retarget manifest",
        error_type=QualificationError,
        require_mode=0o444,
    )
    if parsed_manifest_record != records_by_name["retarget_manifest.json"]:
        raise QualificationError("retarget manifest changed between inventory and parsing")
    common.reject_user_approval(manifest, QualificationError, "retarget manifest")
    if (
        manifest.get("schema") != "tokenrig_rocketbox_retarget_v1"
        or manifest.get("asset_id") != contract["asset_id"]
        or manifest.get("base_avatar_id") != contract["base_avatar_id"]
        or manifest.get("state_classification") != "research_candidate"
        or manifest.get("automatic_checks") != "passed"
        or manifest.get("canonical_front") != "negative-y"
        or manifest.get("canonical_up") != "positive-z"
        or manifest.get("user_acceptance") != "pending_user_review"
    ):
        raise QualificationError("retarget manifest identity, axes, or automatic gate is stale")
    metrics, parsed_metrics_record = common.load_json_mapping_record(
        retarget_root / "retarget_metrics.json",
        root=output_root,
        description="retarget metrics",
        error_type=QualificationError,
        require_mode=0o444,
    )
    if parsed_metrics_record != records_by_name["retarget_metrics.json"]:
        raise QualificationError("retarget metrics changed between inventory and parsing")
    actions = metrics.get("actions")
    if (
        metrics.get("schema") != retarget_owner.METRICS_SCHEMA
        or metrics.get("automatic_checks") != "passed"
        or not isinstance(actions, Mapping)
        or set(actions) != set(retarget_owner.ACTION_NAMES.values())
    ):
        raise QualificationError("retarget metrics schema/actions are incomplete")
    try:
        for action_name in retarget_owner.ACTION_NAMES.values():
            retarget_owner.validate_action_metrics(actions[action_name])
    except retarget_owner.RetargetError as error:
        raise QualificationError(f"retarget action metrics failed owner validation: {error}") from error
    try:
        canonical_manifest = retarget_owner.build_retarget_manifest(
            asset_id=contract["asset_id"],
            base_avatar_id=contract["base_avatar_id"],
            authenticated=manifest.get("authenticated_inputs", {}),
            metrics=metrics,
            artifacts=manifest.get("artifacts", {}),
            command=manifest.get("command", []),
            blender_version=str(manifest.get("environment", {}).get("blender_version", "")),
        )
    except retarget_owner.RetargetError as error:
        raise QualificationError(f"canonical retarget manifest owner validation failed: {error}") from error
    if canonical_manifest != manifest:
        raise QualificationError("retarget manifest is not the canonical owner-built manifest")
    try:
        authenticated = review_renderer.authenticate_review_inputs(
            asset_id=contract["asset_id"],
            static_qa_json=branch_root / static_decision.STATIC_BUNDLE_DIRNAME / "static_qa.json",
            retarget_manifest=manifest_path,
            walking_glb=retarget_root / "walking.glb",
            standing_idle_glb=retarget_root / "standing_idle.glb",
        )
    except review_renderer.ReviewRenderError as error:
        raise QualificationError(f"retarget/static owner validation failed: {error}") from error
    if (
        authenticated["glbs"]["walking"]["sha256"]
        == authenticated["glbs"]["standing_idle"]["sha256"]
    ):
        raise QualificationError("Walking and Standing Idle GLBs must be distinct")
    artifacts = manifest.get("artifacts")
    expected_artifacts = set(RETARGET_FILES) - {"retarget_manifest.json"}
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_artifacts:
        raise QualificationError("retarget manifest artifact inventory is incomplete")
    for name in expected_artifacts:
        descriptor = artifacts[name]
        expected = records_by_name[name]
        if (
            not isinstance(descriptor, Mapping)
            or descriptor.get("path") != name
            or descriptor.get("sha256") != expected["sha256"]
            or descriptor.get("size_bytes") != expected["size_bytes"]
        ):
            raise QualificationError(f"retarget artifact descriptor changed: {name}")
    return retarget_root, records


def _dynamic_snapshot(
    branch_root: Path,
    *,
    output_root: Path,
    contract: Mapping[str, Any],
    static_snapshot: Mapping[str, Any],
    retarget_root: Path,
) -> tuple[Path, Path, dict[str, Any], list[dict[str, Any]]]:
    from tools.spike_rlr import tokenrig_human_review

    review_root = branch_root / DYNAMIC_REVIEW_DIRNAME
    if not review_root.exists():
        raise QualificationError(f"dynamic review bundle is missing: {review_root}")
    try:
        snapshot = tokenrig_human_review.validated_review_snapshot(review_root)
        decision = tokenrig_human_review.read_agent_visual_qa(review_root)
    except tokenrig_human_review.ReviewContractError as error:
        raise QualificationError(f"dynamic review snapshot changed or is invalid: {error}") from error
    if decision.get("status") != static_decision.PASS_STATUS:
        raise QualificationError(
            f"dynamic agent decision is not accepted: {decision.get('status')}"
        )
    if snapshot.get("asset_id") != contract["asset_id"]:
        raise QualificationError("dynamic review asset_id does not match the instance contract")
    if snapshot.get("instance_kind") != contract["case"]["kind"]:
        raise QualificationError("dynamic review instance_kind does not match the instance contract")
    expected_static_root = branch_root / static_decision.STATIC_BUNDLE_DIRNAME
    if Path(snapshot["upstream_paths"]["static_qa"]) != expected_static_root / "static_qa.json":
        raise QualificationError("dynamic review points to a different static final branch")
    if Path(snapshot["upstream_paths"]["bind_pose"]) != expected_static_root / "bind_pose.glb":
        raise QualificationError("dynamic review points to a different bind pose")
    expected_retarget_paths = {
        "retarget_manifest": retarget_root / "retarget_manifest.json",
        "retarget_metrics": retarget_root / "retarget_metrics.json",
        "glb:walking": retarget_root / "walking.glb",
        "glb:standing_idle": retarget_root / "standing_idle.glb",
    }
    for key, expected_path in expected_retarget_paths.items():
        if Path(snapshot["upstream_paths"].get(key, "")) != expected_path:
            raise QualificationError(
                f"dynamic review does not bind canonical {expected_path.name}"
            )
        if snapshot["upstream_sha256"].get(key) != common.sha256_file(expected_path):
            raise QualificationError(
                f"dynamic review {expected_path.name} hash does not match retarget bundle"
            )
    if (
        snapshot["upstream_sha256"]["glb:walking"]
        == snapshot["upstream_sha256"]["glb:standing_idle"]
    ):
        raise QualificationError("Walking and Standing Idle GLBs must be distinct")
    if (
        snapshot["upstream_sha256"]["static_qa"]
        != static_snapshot["artifacts"]["static_qa.json"]["sha256"]
        or snapshot["upstream_sha256"]["bind_pose"]
        != static_snapshot["artifacts"]["bind_pose.glb"]["sha256"]
    ):
        raise QualificationError("dynamic review static hashes do not match the static decision")
    review_names = {
        "review_manifest.json",
        "media_qa.json",
        *(
            f"{motion}_{view}.{kind}"
            for motion in tokenrig_human_review.MOTIONS
            for view in tokenrig_human_review.VIEWS
            for kind in ("png", "mp4")
        ),
    }
    records = _validate_readonly_bundle(
        review_root,
        expected_names=review_names,
        output_root=output_root,
        description="dynamic review bundle",
    )
    decision_path = tokenrig_human_review.agent_decision_path(review_root)
    records.append(
        _readonly_inventory_record(
            decision_path, output_root, "dynamic agent visual decision"
        )
    )
    return review_root, decision_path, snapshot, records


def _inventory_digest(inventory: list[dict[str, Any]]) -> str:
    return hashlib.sha256(common.canonical_json(inventory).encode("utf-8")).hexdigest()


def _iter_manifest_file_records(value: Any):
    if isinstance(value, Mapping):
        if (
            isinstance(value.get("path"), str)
            and isinstance(value.get("sha256"), str)
            and isinstance(value.get("size_bytes", value.get("bytes")), int)
        ):
            yield value
        for child in value.values():
            yield from _iter_manifest_file_records(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_manifest_file_records(child)


def _branch_provenance(
    *,
    contract: Mapping[str, Any],
    branch_id: str,
    branch_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from tools import blender_tokenrig_human_static_audit as static_audit

    manifest_path = branch_root / "tokenrig_manifest.json"
    glb_path = branch_root / "tokenrig_transfer.glb"
    manifest_record = _readonly_inventory_record(
        manifest_path, Path(contract["canonical_output_root"]), "branch TokenRig manifest"
    )
    glb_record = _readonly_inventory_record(
        glb_path, Path(contract["canonical_output_root"]), "branch TokenRig GLB"
    )
    manifest, parsed_manifest_record = common.load_json_mapping_record(
        manifest_path,
        root=Path(contract["canonical_output_root"]),
        description="branch TokenRig manifest",
        error_type=QualificationError,
        require_mode=0o444,
    )
    if parsed_manifest_record != manifest_record:
        raise QualificationError("branch TokenRig manifest changed during parsing")
    common.reject_user_approval(manifest, QualificationError, "branch TokenRig manifest")
    expected_schemas = BRANCH_MANIFEST_SCHEMAS.get(branch_id)
    if expected_schemas is None or manifest.get("schema") not in expected_schemas:
        raise QualificationError("branch producer manifest schema does not match final branch")
    attempt = manifest.get("attempt_ledger")
    source_glb = Path(contract["source_lineage"]["pixal_pbr_glb"]["path"])
    try:
        owner = static_audit.authenticate_task3_inputs(
            asset_id=contract["asset_id"],
            source_glb=source_glb,
            tokenrig_glb=glb_path,
            tokenrig_manifest=manifest_path,
        )
    except static_audit.StaticAuditError as error:
        raise QualificationError(f"branch producer owner validation failed: {error}") from error
    static_qa, _ = common.load_json_mapping_record(
        branch_root / static_decision.STATIC_BUNDLE_DIRNAME / "static_qa.json",
        root=Path(contract["canonical_output_root"]),
        description="static QA",
        error_type=QualificationError,
        require_mode=0o444,
    )
    if static_qa.get("authenticated") != owner:
        raise QualificationError("static QA does not bind the current branch producer snapshot")
    records_by_path: dict[str, dict[str, Any]] = {
        manifest_record["path"]: manifest_record,
        glb_record["path"]: glb_record,
    }
    manifest_bound_paths: set[str] = set()
    for descriptor in _iter_manifest_file_records(manifest):
        supplied = Path(descriptor["path"])
        path = supplied if supplied.is_absolute() else branch_root / supplied
        path = common.absolute(path)
        manifest_bound_paths.add(str(path))
        try:
            path.relative_to(
                common.absolute(Path(contract["canonical_output_root"]))
            )
            internal_producer_file = True
        except ValueError:
            internal_producer_file = False
        record = common.file_record(
            path,
            root=path.parent,
            description="branch provenance file",
            error_type=QualificationError,
            require_mode=0o444 if internal_producer_file else None,
        )
        expected_size = descriptor.get("size_bytes", descriptor.get("bytes"))
        if (
            record["sha256"] != descriptor["sha256"]
            or record["size_bytes"] != expected_size
        ):
            raise QualificationError("branch provenance descriptor changed")
        records_by_path[record["path"]] = record
    attempt_path: Path
    if isinstance(attempt, Mapping) and isinstance(attempt.get("path"), str):
        attempt_path = Path(attempt["path"])
    elif branch_id == "direct":
        local = branch_root / "tokenrig_attempt.json"
        legacy = Path(contract["canonical_output_root"]).parent / (
            f"{contract['asset_id']}.tokenrig_attempt.json"
        )
        attempt_path = local if local.exists() else legacy
    elif branch_id == "fitted_skeleton":
        attempt_path = branch_root.with_name(f"{branch_root.name}.tokenrig_attempt.json")
    else:
        # Deterministic sanitation uses its immutable manifest as its attempt ledger.
        attempt_path = manifest_path
    attempt_record = common.file_record(
        attempt_path,
        root=attempt_path.parent,
        description="branch attempt ledger",
        error_type=QualificationError,
        require_mode=0o444,
    )
    records_by_path[attempt_record["path"]] = attempt_record

    if (
        contract.get("asset_id") == FEMALE_BASE_ASSET_ID
        and contract.get("base_avatar_id") == FEMALE_BASE_ASSET_ID
    ):
        from tools import tokenrig_human_female_canary as female_owner

        output_root = common.absolute(Path(contract["canonical_output_root"]))
        direct_manifest_path = output_root / "tokenrig_manifest.json"
        authorization_path = output_root / "tokenrig_female_authorization_v2.json"
        try:
            authorization = female_owner.validate_female_authorization_manifest(
                authorization_path,
                expected_tokenrig_manifest=direct_manifest_path,
                expected_asset_id=FEMALE_BASE_ASSET_ID,
            )
        except female_owner.FemaleGateError as error:
            raise QualificationError(
                f"female authorization or male qualified gate is invalid: {error}"
            ) from error
        owner_records = authorization.get("records")
        if not isinstance(owner_records, Mapping) or set(owner_records) != {
            "authorization",
            "female_gate_record",
            "male_qualified_candidate",
            "tokenrig_manifest",
            "female_wrapper",
            "base_runner",
        }:
            raise QualificationError("female authorization owner records are incomplete")

        direct_manifest, direct_manifest_record = common.load_json_mapping_record(
            direct_manifest_path,
            root=output_root,
            description="female direct TokenRig manifest",
            error_type=QualificationError,
            require_mode=0o444,
        )
        if direct_manifest_record != owner_records["tokenrig_manifest"]:
            raise QualificationError("female direct TokenRig manifest owner record changed")
        direct_attempt = direct_manifest.get("attempt_ledger")
        if isinstance(direct_attempt, Mapping) and isinstance(
            direct_attempt.get("path"), str
        ):
            direct_attempt_path = Path(direct_attempt["path"])
            if not direct_attempt_path.is_absolute():
                direct_attempt_path = output_root / direct_attempt_path
        else:
            local = output_root / "tokenrig_attempt.json"
            legacy = output_root.parent / (
                f"{contract['asset_id']}.tokenrig_attempt.json"
            )
            direct_attempt_path = local if local.exists() else legacy
        direct_attempt_record = common.file_record(
            direct_attempt_path,
            root=direct_attempt_path.parent,
            description="female direct attempt ledger",
            error_type=QualificationError,
            require_mode=0o444,
        )
        if isinstance(direct_attempt, Mapping):
            expected_size = direct_attempt.get(
                "size_bytes", direct_attempt.get("bytes")
            )
            if (
                direct_attempt_record["sha256"] != direct_attempt.get("sha256")
                or direct_attempt_record["size_bytes"] != expected_size
            ):
                raise QualificationError("female direct attempt descriptor changed")

        if branch_id != "direct":
            direct_lineage = manifest.get("direct_female_lineage")
            expected_direct_lineage = {
                "manifest": {
                    key: direct_manifest_record[key]
                    for key in ("path", "sha256", "size_bytes")
                },
                "attempt": {
                    key: direct_attempt_record[key]
                    for key in ("path", "sha256", "size_bytes")
                },
                "authorization": {
                    key: owner_records["authorization"][key]
                    for key in ("path", "sha256", "size_bytes")
                },
            }
            if (
                not isinstance(direct_lineage, Mapping)
                or set(direct_lineage) != set(expected_direct_lineage)
                or any(
                    not isinstance(direct_lineage.get(role), Mapping)
                    or dict(direct_lineage[role]) != expected
                    for role, expected in expected_direct_lineage.items()
                )
            ):
                raise QualificationError(
                    "nested female final branch direct manifest, attempt, and "
                    "authorization require exact direct_female_lineage descriptors"
                )
            required_direct_lineage = {
                record["path"] for record in expected_direct_lineage.values()
            }
            if not required_direct_lineage.issubset(manifest_bound_paths):
                raise QualificationError(
                    "nested female direct_female_lineage was not recursively authenticated"
                )

        for description, record in owner_records.items():
            if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
                raise QualificationError(
                    f"female authorization owner record is invalid: {description}"
                )
            existing = records_by_path.get(record["path"])
            if existing is not None and existing != dict(record):
                raise QualificationError(
                    f"female authorization owner record conflicts with branch provenance: {description}"
                )
            records_by_path[record["path"]] = dict(record)
        records_by_path[direct_attempt_record["path"]] = direct_attempt_record

    records = sorted(records_by_path.values(), key=lambda item: item["path"])
    if not any(record["path"] == attempt_record["path"] for record in records):
        raise QualificationError("branch attempt ledger is not present in provenance inventory")
    return owner, records


def _build_qualified_candidate_once(
    contract_path: Path, *, branch_id: str
) -> dict[str, Any]:
    contract, contract_path, output_root = _instance(contract_path)
    branch_descriptor, branch_root = _branch(contract, branch_id)
    try:
        static_payload = static_decision.validate_static_agent_visual_decision(
            contract_path, branch_id=branch_id, require_pass=True
        )
        static_record = static_decision.static_decision_record(
            contract_path, branch_id=branch_id, require_pass=True
        )
    except static_decision.StaticDecisionError as error:
        raise QualificationError(f"static agent decision is not accepted: {error}") from error
    static_root = branch_root / static_decision.STATIC_BUNDLE_DIRNAME
    static_artifact_names = set(static_decision.STATIC_ARTIFACTS)
    if contract["case"]["case_id"] in static_decision.ACCESSORY_CASES:
        static_artifact_names.update(static_decision.ACCESSORY_ARTIFACTS)
    static_records = _validate_readonly_bundle(
        static_root,
        expected_names={"static_qa.json", *static_artifact_names},
        output_root=output_root,
        description="static QA bundle",
    )
    branch_owner, branch_provenance_inventory = _branch_provenance(
        contract=contract,
        branch_id=branch_id,
        branch_root=branch_root,
    )
    retarget_root, retarget_records = _validate_retarget_bundle(
        branch_root,
        output_root=output_root,
        contract=contract,
    )
    review_root, dynamic_decision_path, _, dynamic_records = _dynamic_snapshot(
        branch_root,
        output_root=output_root,
        contract=contract,
        static_snapshot=static_payload["snapshot"],
        retarget_root=retarget_root,
    )
    contract_record = _readonly_inventory_record(
        contract_path, output_root, "instance contract"
    )
    inventory = [
        contract_record,
        *static_records,
        static_record,
        *retarget_records,
        *dynamic_records,
    ]
    inventory.sort(key=lambda record: record["relative_path"])
    relative_paths = [record["relative_path"] for record in inventory]
    if len(relative_paths) != len(set(relative_paths)):
        raise QualificationError("qualification inventory contains duplicate paths")
    review_manifest = _readonly_inventory_record(
        review_root / "review_manifest.json",
        output_root,
        "dynamic review manifest",
    )
    dynamic_decision_record = _readonly_inventory_record(
        dynamic_decision_path,
        output_root,
        "dynamic agent visual decision",
    )
    retarget_manifest = _readonly_inventory_record(
        retarget_root / "retarget_manifest.json",
        output_root,
        "retarget manifest",
    )
    result = {
        "schema": SCHEMA,
        "asset_id": contract["asset_id"],
        "base_avatar_id": contract["base_avatar_id"],
        "case": dict(contract["case"]),
        "state_classification": "research_candidate",
        "status": static_decision.PASS_STATUS,
        "contract": contract_record,
        "final_branch": {
            "branch_id": branch_id,
            "path": str(branch_root),
            "relative_root": branch_descriptor["relative_root"],
        },
        "branch_owner_validation": branch_owner,
        "branch_provenance_inventory": branch_provenance_inventory,
        "branch_provenance_sha256": _inventory_digest(branch_provenance_inventory),
        "static": {
            "bundle_dir": str(static_root),
            "decision": static_record,
        },
        "retarget": {
            "bundle_dir": str(retarget_root),
            "manifest": retarget_manifest,
        },
        "dynamic": {
            "review_dir": str(review_root),
            "review_manifest": review_manifest,
            "decision": dynamic_decision_record,
        },
        "inventory": inventory,
        "inventory_sha256": _inventory_digest(inventory),
        "user_acceptance": "pending_user_review",
    }
    common.reject_user_approval(result, QualificationError, "qualified candidate")
    return result


def build_qualified_candidate(
    contract_path: Path, *, branch_id: str
) -> dict[str, Any]:
    return common.stable_mapping_snapshot(
        lambda: _build_qualified_candidate_once(contract_path, branch_id=branch_id),
        QualificationError,
        "qualified candidate evidence",
    )


def publish_qualified_candidate(contract_path: Path, *, branch_id: str) -> Path:
    payload = build_qualified_candidate(contract_path, branch_id=branch_id)
    output_root = Path(payload["contract"]["path"]).parent
    destination = output_root / FILENAME

    def validate_prelink() -> None:
        if build_qualified_candidate(contract_path, branch_id=branch_id) != payload:
            raise QualificationError(
                "qualified candidate snapshot changed during pre-publication validation"
            )

    return common.write_json_immutable_noreplace(
        destination,
        payload,
        QualificationError,
        "qualified candidate",
        prelink_validator=validate_prelink,
    )


def _validate_qualified_candidate_once(path: Path) -> dict[str, Any]:
    supplied = common.absolute(path)
    if supplied.name != FILENAME:
        raise QualificationError(f"qualified candidate must be named {FILENAME}")
    output_root = common.require_real_directory(
        supplied.parent, "qualified avatar root", QualificationError
    )
    payload, _ = common.load_json_mapping_record(
        supplied,
        root=output_root,
        description="qualified candidate",
        error_type=QualificationError,
        require_mode=0o444,
    )
    common.reject_user_approval(payload, QualificationError, "qualified candidate")
    contract_descriptor = payload.get("contract")
    if not isinstance(contract_descriptor, Mapping):
        raise QualificationError("qualified candidate contract descriptor is missing")
    contract_path_value = contract_descriptor.get("path")
    if not isinstance(contract_path_value, str):
        raise QualificationError("qualified candidate contract path is missing")
    contract_path = common.absolute(Path(contract_path_value))
    if contract_path != output_root / instance_contract.FILENAME:
        raise QualificationError("qualified candidate contract path is not canonical")
    branch = payload.get("final_branch")
    if not isinstance(branch, Mapping) or not isinstance(branch.get("branch_id"), str):
        raise QualificationError("qualified candidate final branch is missing")
    try:
        expected = build_qualified_candidate(
            contract_path, branch_id=branch["branch_id"]
        )
    except QualificationError as error:
        raise QualificationError(
            f"qualified candidate snapshot changed after publication: {error}"
        ) from error
    if payload != expected:
        raise QualificationError("qualified candidate snapshot changed after publication")
    return expected


def validate_qualified_candidate(path: Path) -> dict[str, Any]:
    return common.stable_mapping_snapshot(
        lambda: _validate_qualified_candidate_once(path),
        QualificationError,
        "qualified candidate pointer",
    )


__all__ = [
    "DYNAMIC_REVIEW_DIRNAME",
    "FILENAME",
    "QualificationError",
    "RETARGET_DIRNAME",
    "RETARGET_FILES",
    "SCHEMA",
    "build_qualified_candidate",
    "publish_qualified_candidate",
    "validate_qualified_candidate",
]
