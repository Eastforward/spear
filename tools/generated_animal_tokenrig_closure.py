"""Build and validate a fail-closed Pixal/geometry -> TokenRig evidence closure.

The builder derives execution identity from retained runtime markers and the
`/usr/bin/time` command in run.log; callers do not self-report seed, argv,
checkpoint, or patch identity.  The resulting directory contains immutable
copies of all small evidence and hashes the large GLBs/model in place.

Validation requires the SHA-256 of the manifest file from an external caller.
Consequently, editing a manifest and recomputing its embedded self-hash cannot
authorize a different asset, command, model, or evidence chain.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
from typing import Any
import uuid

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import (
    controlled_animal_derived_static_review_contract as derived_review_contract,
)
from tools import controlled_source_asset_schema as contracts


SCHEMA = "avengine_generated_animal_tokenrig_source_closure_v1"
READBACK_SCHEMA = "avengine_generated_animal_tokenrig_binding_readback_v1"
DESCRIPTOR_SCHEMA = "avengine_generated_animal_tokenrig_closure_descriptor_v1"
COMPLETE_MODE = "complete_load_audit_v1"
CHECKED_IN_FREE_SKELETON_MODE = "checked_in_free_skeleton_runner_v1"
LEGACY_SHIBA_MODE = "legacy_shiba_20260726_missing_load_audit_v1"
LOAD_AUDIT_MODES = {COMPLETE_MODE, CHECKED_IN_FREE_SKELETON_MODE}
SUPPORTED_MODES = LOAD_AUDIT_MODES | {LEGACY_SHIBA_MODE}
FREE_SKELETON_EXECUTION_IDENTITY_SCHEMA = (
    "avengine_generated_animal_tokenrig_free_skeleton_execution_identity_v1"
)
EXPECTED_SEED = 42
EXPECTED_MODEL_SHA256 = (
    "f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692"
)
EXPECTED_MODEL_SNAPSHOT_REVISION = "79736cad0fd84de384d5eede659b4ebd24effe33"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SNAPSHOT_PATTERN = re.compile(r"/snapshots/([0-9a-f]{40})(?:/|$)")
COMMAND_PATTERN = re.compile(r'^\s*Command being timed: "(.*)"\s*$', re.MULTILINE)
EXPORTED_PATTERN = re.compile(r"^\[OK\] Exported:\s*(.+?)\s*$", re.MULTILINE)
FREE_SKELETON_IDENTITY_PATTERN = re.compile(
    r"^TOKENRIG_FREE_SKELETON_EXECUTION_IDENTITY\s+(.+?)\s*$",
    re.MULTILINE,
)
SPEAR_ROOT = Path(__file__).resolve().parents[1]
FREE_SKELETON_RUNNER_RELATIVE = Path(
    "tools/run_generated_animal_tokenrig_free_skeleton.py"
)
RUNTIME_PATCH_RELATIVE = Path(
    "tools/runtime_patches/fixed_skeleton_skintokens/sitecustomize.py"
)
LOGICAL_TMP_ROOT = SPEAR_ROOT / "tmp"
PHYSICAL_TMP_ROOT = Path(
    "/data/datasets/avengine_workspaces/AVEngine/external/SPEAR/tmp"
)


class ClosureError(RuntimeError):
    pass


def canonical_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ClosureError("payload is not canonical finite JSON") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_without(payload: dict[str, Any], field: str) -> str:
    return sha256_bytes(
        canonical_bytes({key: value for key, value in payload.items() if key != field})
    )


def recursive_finite(value: Any, label: str = "payload") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ClosureError(f"{label} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ClosureError(f"{label} contains a non-string JSON key")
            recursive_finite(child, f"{label}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            recursive_finite(child, f"{label}[{index}]")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise ClosureError(f"{label} contains a non-JSON value")


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
        raise ClosureError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _path_components(path: Path) -> list[Path]:
    absolute = path.absolute()
    components = []
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        components.append(current)
    return components


def _allowed_workspace_tmp_link(component: Path) -> bool:
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


def _reject_unsafe_symlinks(path: Path, label: str, *, file_leaf: bool) -> None:
    components = _path_components(path)
    for index, component in enumerate(components):
        if not component.is_symlink():
            continue
        is_leaf = index == len(components) - 1
        if _allowed_workspace_tmp_link(component) and not (file_leaf and is_leaf):
            continue
        raise ClosureError(f"{label} contains an unsafe symlink component: {component}")


def require_regular_file(
    path: Path,
    label: str,
) -> Path:
    absolute = path.absolute()
    _reject_unsafe_symlinks(absolute, label, file_leaf=True)
    if not absolute.is_file():
        raise ClosureError(f"missing {label}: {absolute}")
    if absolute.stat().st_size <= 0:
        raise ClosureError(f"empty {label}: {absolute}")
    return absolute.resolve(strict=True)


def require_directory(
    path: Path,
    label: str,
) -> Path:
    absolute = path.absolute()
    _reject_unsafe_symlinks(absolute, label, file_leaf=False)
    if not absolute.is_dir():
        raise ClosureError(f"missing {label}: {absolute}")
    return absolute.resolve(strict=True)


def same_file(left: Path, right: Path, label: str) -> None:
    try:
        identical = os.path.samefile(left, right)
    except OSError as error:
        raise ClosureError(f"cannot compare {label} file identity") from error
    if not identical:
        raise ClosureError(
            f"{label} path points to a different file: {left} != {right}"
        )


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ClosureError(f"invalid {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ClosureError(f"{label} must be a JSON object")
    recursive_finite(payload, label)
    return payload


def descriptor_path(
    descriptor: Any,
    label: str,
    *,
    size_fields: tuple[str, ...] = ("size_bytes", "bytes"),
    require_size: bool = True,
) -> Path:
    if not isinstance(descriptor, dict):
        raise ClosureError(f"{label} descriptor must be an object")
    path_value = descriptor.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ClosureError(f"{label} descriptor path is invalid")
    path = require_regular_file(Path(path_value), label)
    expected_hash = require_sha256(descriptor.get("sha256"), f"{label} sha256")
    if sha256_file(path) != expected_hash:
        raise ClosureError(f"{label} SHA-256 mismatch")
    size = next(
        (descriptor.get(field) for field in size_fields if field in descriptor),
        None,
    )
    if size is None and not require_size:
        return path
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or path.stat().st_size != size
    ):
        raise ClosureError(f"{label} size mismatch")
    return path


def require_descriptor_matches(
    descriptor: Any,
    expected: Path,
    label: str,
    *,
    size_fields: tuple[str, ...] = ("size_bytes", "bytes"),
) -> None:
    observed = descriptor_path(descriptor, label, size_fields=size_fields)
    same_file(observed, expected, label)


def require_descriptor_content_matches(
    descriptor: Any,
    expected: Path,
    label: str,
    *,
    size_fields: tuple[str, ...] = ("size_bytes", "bytes"),
) -> None:
    """Bind a historical descriptor to externally selected current bytes."""

    if not isinstance(descriptor, dict):
        raise ClosureError(f"{label} descriptor must be an object")
    path_value = descriptor.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise ClosureError(f"{label} descriptor path is invalid")
    expected = require_regular_file(expected, label)
    expected_hash = require_sha256(descriptor.get("sha256"), f"{label} sha256")
    if sha256_file(expected) != expected_hash:
        raise ClosureError(f"{label} SHA-256 mismatch")
    size = next(
        (descriptor.get(field) for field in size_fields if field in descriptor),
        None,
    )
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or expected.stat().st_size != size
    ):
        raise ClosureError(f"{label} size mismatch")


def git_output(root: Path, arguments: list[str], label: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments],
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise ClosureError(f"{label} failed: {detail}") from error


def git_revision(root: Path) -> str:
    revision = (
        git_output(root, ["rev-parse", "HEAD"], "SkinTokens revision")
        .decode("ascii")
        .strip()
    )
    if not COMMIT_PATTERN.fullmatch(revision):
        raise ClosureError("SkinTokens HEAD is not a full commit hash")
    dirty = git_output(
        root,
        ["status", "--porcelain", "--untracked-files=no"],
        "SkinTokens tracked status",
    )
    if dirty:
        raise ClosureError("SkinTokens has tracked modifications")
    return revision


def git_blob(root: Path, spec: str, label: str) -> bytes:
    if ":" not in spec:
        raise ClosureError(f"{label} git spec must be REVISION:path")
    revision, relative = spec.split(":", 1)
    if not revision or not relative or Path(relative).is_absolute():
        raise ClosureError(f"invalid {label} git spec")
    return git_output(root, ["show", spec], label)


def parse_raw_pixal_output(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("backend") != "pixal3d":
        raise ClosureError("raw Pixal manifest backend must be pixal3d")
    output = payload.get("output")
    if not isinstance(output, dict):
        raise ClosureError("raw Pixal manifest is missing output")
    return output


def validate_upstream_lineage(
    raw_manifest_path: Path,
    upstream_manifest_path: Path,
    tokenrig_input: Path,
    *,
    raw_static_decision_batch_path: Path | None = None,
    expected_raw_static_decision_batch_sha256: str | None = None,
    require_oriented_batch_external_authority: bool = False,
) -> tuple[Path, str, list[Path]]:
    raw_manifest = load_json(raw_manifest_path, "raw Pixal manifest")
    raw_output = parse_raw_pixal_output(raw_manifest)
    upstream = load_json(upstream_manifest_path, "upstream geometry manifest")
    schema = upstream.get("schema")
    raw_glb = (
        None
        if schema == "avengine_generated_animal_geometry_closure_v2"
        else descriptor_path(
            raw_output,
            "raw Pixal GLB",
            size_fields=("bytes", "size_bytes"),
        )
    )
    extra_evidence: list[Path] = []
    if schema == "avengine_watertight_textured_runtime_proxy_v1":
        assert raw_glb is not None
        if (
            raw_static_decision_batch_path is not None
            or expected_raw_static_decision_batch_sha256 is not None
        ):
            raise ClosureError(
                "raw/watertight fallback cannot consume derived repair authority"
            )
        require_descriptor_matches(
            upstream.get("input"), raw_glb, "watertight raw Pixal input"
        )
        require_descriptor_matches(
            upstream.get("output"), tokenrig_input, "watertight TokenRig input"
        )
        if (
            upstream.get("authority_contract", {}).get(
                "approved_skeleton_or_animation_touched"
            )
            is not False
        ):
            raise ClosureError("watertight upstream touched skeleton or animation")
        upstream_kind = "watertight_runtime_proxy"
    elif schema == "avengine_generated_animal_geometry_closure_v2":
        from tools import (
            publish_generated_animal_geometry_closure as geometry_closures,
        )

        if (
            raw_static_decision_batch_path is None
            and require_oriented_batch_external_authority
        ):
            raise ClosureError(
                "geometry closure v2 requires an externally pinned raw "
                "static decision batch"
            )
        supplied_batch: Path | None = None
        if raw_static_decision_batch_path is not None:
            if expected_raw_static_decision_batch_sha256 is None:
                raise ClosureError(
                    "geometry closure v2 raw decision batch needs an "
                    "expected SHA-256"
                )
            supplied_batch = require_regular_file(
                raw_static_decision_batch_path,
                "externally supplied raw static decision batch",
            )
            expected_batch_sha256 = require_sha256(
                expected_raw_static_decision_batch_sha256,
                "expected raw static decision batch SHA-256",
            )
            if sha256_file(supplied_batch) != expected_batch_sha256:
                raise ClosureError(
                    "raw static decision batch changed from external authority"
                )
        elif expected_raw_static_decision_batch_sha256 is not None:
            raise ClosureError(
                "raw static decision batch path/hash must be supplied together"
            )
        try:
            replay = geometry_closures.load_geometry_closure_v2(
                upstream_manifest_path,
                expected_pixal_manifest=raw_manifest_path,
                expected_raw_static_decision_batch=supplied_batch,
                expected_repaired_glb=tokenrig_input,
            )
        except geometry_closures.GeometryClosureError as error:
            raise ClosureError(
                f"geometry closure v2 strict replay failed: {error}"
            ) from error
        replay_paths = replay["paths"]
        raw_glb = require_regular_file(
            replay_paths["raw_pixal_glb"],
            "geometry closure v2 raw Pixel3D GLB",
        )
        require_descriptor_content_matches(
            raw_output,
            raw_glb,
            "raw Pixal GLB",
            size_fields=("bytes", "size_bytes"),
        )
        if (
            supplied_batch is not None
            and replay_paths["raw_static_decision_batch"] != supplied_batch
        ):
            raise ClosureError(
                "geometry closure v2 raw decision batch identity changed"
            )
        extra_evidence.extend(
            (
                replay_paths["repair_manifest"],
                replay_paths["geometry_audit"],
                replay_paths["raw_static_decision_batch"],
            )
        )
        upstream_kind = "bounded_geometry_closure"
    elif schema == "avengine_generated_animal_geometry_closure_v1":
        assert raw_glb is not None
        if upstream.get("status") != "pass_geometry_only":
            raise ClosureError("geometry closure did not pass its geometry-only gate")
        require_descriptor_matches(
            upstream.get("candidate", {}).get("source_pixal_glb"),
            raw_glb,
            "geometry closure raw Pixal source",
        )
        require_descriptor_matches(
            upstream.get("output", {}).get("glb"),
            tokenrig_input,
            "geometry closure TokenRig input",
        )
        if upstream.get("downstream", {}).get("tokenrig_entry_authorized") is not True:
            raise ClosureError("geometry closure did not authorize TokenRig entry")
        repair_descriptor = upstream.get("output", {}).get("repair_manifest")
        repair_path = descriptor_path(
            repair_descriptor,
            "geometry repair manifest",
            size_fields=("size_bytes",),
            require_size=False,
        )
        repair = load_json(repair_path, "geometry repair manifest")
        try:
            repair = derived_review_contract.validate_bounded_repair_manifest(
                repair
            )
        except derived_review_contract.DerivedStaticReviewContractError as error:
            raise ClosureError(
                f"geometry repair manifest failed its strict contract: {error}"
            ) from error
        require_descriptor_matches(
            repair.get("output"), tokenrig_input, "geometry repair output"
        )
        require_descriptor_matches(
            repair.get("lineage", {}).get("pixal_source"),
            raw_glb,
            "geometry repair raw Pixal source",
        )
        repair_raw_manifest = descriptor_path(
            repair.get("lineage", {}).get("pixal_manifest"),
            "geometry repair raw Pixal manifest",
        )
        same_file(
            repair_raw_manifest,
            raw_manifest_path,
            "geometry repair raw Pixal manifest",
        )
        extra_evidence.append(repair_path)
        if (
            repair["implementation_contract"]
            == derived_review_contract.ORIENTED_REPAIR_IMPLEMENTATION_CONTRACT
        ):
            batch_from_repair = descriptor_path(
                repair["lineage"].get("static_decision_batch"),
                "oriented repair canonical raw static decision batch",
            )
            if (
                raw_static_decision_batch_path is None
                and require_oriented_batch_external_authority
            ):
                raise ClosureError(
                    "oriented repair requires an externally pinned raw "
                    "static decision batch"
                )
            if raw_static_decision_batch_path is not None:
                supplied_batch = require_regular_file(
                    raw_static_decision_batch_path,
                    "externally supplied raw static decision batch",
                )
                same_file(
                    supplied_batch,
                    batch_from_repair,
                    "oriented repair raw static decision batch",
                )
                if expected_raw_static_decision_batch_sha256 is None:
                    raise ClosureError(
                        "oriented repair raw decision batch needs an expected SHA-256"
                    )
                expected_batch_sha256 = require_sha256(
                    expected_raw_static_decision_batch_sha256,
                    "expected raw static decision batch SHA-256",
                )
                if sha256_file(supplied_batch) != expected_batch_sha256:
                    raise ClosureError(
                        "raw static decision batch changed from external authority"
                    )
            elif expected_raw_static_decision_batch_sha256 is not None:
                raise ClosureError(
                    "raw static decision batch path/hash must be supplied together"
                )
            from tools import (
                register_controlled_animal_source_assets as source_registry,
            )

            try:
                authenticated_batch, _batch, decisions = (
                    source_registry.load_decision_batch(batch_from_repair)
                )
            except contracts.ContractError as error:
                raise ClosureError(
                    f"oriented repair raw decision batch is invalid: {error}"
                ) from error
            selected = decisions.get(repair["lineage"]["instance_id"])
            selected_path = (
                selected.get("path") if isinstance(selected, dict) else None
            )
            selected_payload = (
                selected.get("payload") if isinstance(selected, dict) else None
            )
            static_decision = descriptor_path(
                repair["lineage"].get("static_decision"),
                "oriented repair raw static decision",
            )
            if (
                authenticated_batch != batch_from_repair
                or not isinstance(selected_payload, dict)
                or selected_path != static_decision
                or selected_payload.get("decision")
                != "approved_for_lod_and_binding"
                or selected_payload.get("state_classification")
                != "research_candidate"
                or selected_payload.get(
                    "formal_dataset_registration_authorized"
                )
                is not False
                or selected_payload.get("next_gate")
                != "lod_then_species_rig_binding"
                or not isinstance(selected_payload.get("checks"), dict)
                or set(selected_payload["checks"])
                != derived_review_contract.RAW_STATIC_CHECK_FIELDS
                or any(
                    value is not True
                    for value in selected_payload["checks"].values()
                )
            ):
                raise ClosureError(
                    "oriented repair is not bound to the canonical approved "
                    "raw static decision"
                )
            extra_evidence.append(batch_from_repair)
        elif (
            raw_static_decision_batch_path is not None
            or expected_raw_static_decision_batch_sha256 is not None
        ):
            raise ClosureError(
                "mirror-v2 repair cannot consume oriented raw decision "
                "batch authority"
            )
        upstream_kind = "bounded_geometry_closure"
    else:
        raise ClosureError(f"unsupported upstream geometry schema: {schema!r}")
    assert raw_glb is not None
    return raw_glb, upstream_kind, extra_evidence


def validate_readback(
    readback_path: Path,
    tokenrig_input: Path,
    tokenrig_output: Path,
) -> dict[str, Any]:
    payload = load_json(readback_path, "TokenRig geometry readback")
    if payload.get("schema") != READBACK_SCHEMA:
        raise ClosureError("unexpected TokenRig geometry readback schema")
    if payload.get("manifest_sha256") != hash_without(payload, "manifest_sha256"):
        raise ClosureError("TokenRig geometry readback self-hash mismatch")
    require_descriptor_matches(
        payload.get("tokenrig_input"), tokenrig_input, "readback TokenRig input"
    )
    require_descriptor_matches(
        payload.get("tokenrig_output"), tokenrig_output, "readback TokenRig output"
    )
    checks = payload.get("automatic_checks")
    required_checks = {
        "world_geometry_unchanged",
        "logical_vertices_bijective",
        "triangles_bijective",
        "uvs_unchanged",
        "pbr_unchanged",
        "exactly_one_output_skin",
        "exactly_one_output_armature",
        "no_animation",
    }
    if (
        not isinstance(checks, dict)
        or checks.get("overall") != "passed"
        or any(checks.get(key) is not True for key in required_checks)
    ):
        raise ClosureError("TokenRig geometry readback did not pass every gate")
    if payload.get("formal_dataset_registration_authorized") is not False:
        raise ClosureError("geometry readback exceeded its claim boundary")
    return payload


def marker_payloads(marker_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(marker_dir.glob("*.json"))
    if len(paths) != 2:
        raise ClosureError("TokenRig evidence requires exactly two runtime markers")
    result = []
    for path in paths:
        regular = require_regular_file(path, "TokenRig runtime marker")
        payload = load_json(regular, "TokenRig runtime marker")
        pid = payload.get("pid")
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or regular.stem != str(pid)
        ):
            raise ClosureError("runtime marker PID does not match its filename")
        if payload.get("seed") != EXPECTED_SEED:
            raise ClosureError("runtime marker seed is not exactly 42")
        require_sha256(payload.get("patch_sha256"), "runtime marker patch sha256")
        argv = payload.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(value, str) and value for value in argv)
        ):
            raise ClosureError("runtime marker argv is invalid")
        result.append((regular, payload))
    server = [item for item in result if item[1]["argv"] == ["bpy_server.py"]]
    main = [item for item in result if item[1]["argv"] != ["bpy_server.py"]]
    if len(server) != 1 or len(main) != 1:
        raise ClosureError("runtime markers must contain one runner and one server")
    common_fields = ("seed", "patch_sha256", "bpy_port")
    for field in common_fields:
        if server[0][1].get(field) != main[0][1].get(field):
            raise ClosureError(f"runtime markers disagree on {field}")
    return [main[0], server[0]]


def option_value(argv: list[str], option: str) -> str:
    if argv.count(option) != 1:
        raise ClosureError(f"exact argv must contain one {option}")
    index = argv.index(option)
    if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
        raise ClosureError(f"exact argv has no value for {option}")
    return argv[index + 1]


def parse_timed_command(run_log: str) -> tuple[dict[str, str], str, list[str]]:
    matches = COMMAND_PATTERN.findall(run_log)
    if len(matches) != 1:
        raise ClosureError("run.log must retain exactly one timed command")
    try:
        tokens = shlex.split(matches[0])
    except ValueError as error:
        raise ClosureError("run.log timed command cannot be parsed") from error
    if not tokens or tokens[0] != "env":
        raise ClosureError("run.log timed command must begin with env")
    environment: dict[str, str] = {}
    index = 1
    while index < len(tokens) and "=" in tokens[index]:
        key, value = tokens[index].split("=", 1)
        if not key or key in environment:
            raise ClosureError("run.log has an invalid or duplicate environment key")
        environment[key] = value
        index += 1
    if index >= len(tokens):
        raise ClosureError("run.log timed command is missing its executable")
    executable = tokens[index]
    exact_argv = tokens[index + 1 :]
    if not exact_argv:
        raise ClosureError("run.log timed command is missing runner argv")
    return environment, executable, exact_argv


def _logged_file_record(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise ClosureError(f"{label} record is invalid")
    path = value.get("path")
    size = value.get("size_bytes")
    if not isinstance(path, str) or not path:
        raise ClosureError(f"{label} path is invalid")
    require_sha256(value.get("sha256"), f"{label} sha256")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ClosureError(f"{label} size is invalid")
    return value


def parse_free_skeleton_execution_identity(
    run_log: str,
    *,
    exact_argv: list[str],
    tokenrig_input: Path,
    model_record: dict[str, Any],
    patch_sha256: str,
) -> dict[str, Any]:
    matches = FREE_SKELETON_IDENTITY_PATTERN.findall(run_log)
    if len(matches) != 1:
        raise ClosureError(
            "checked-in free-skeleton run.log must retain one execution identity"
        )
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError as error:
        raise ClosureError(
            "free-skeleton execution identity is invalid JSON"
        ) from error
    recursive_finite(payload, "free-skeleton execution identity")
    expected_fields = {
        "schema",
        "spear_revision",
        "runner",
        "runtime_patch",
        "skintokens_revision",
        "input",
        "model_checkpoint",
        "production_contract",
        "formal_dataset_registration_authorized",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_fields
        or payload.get("schema") != FREE_SKELETON_EXECUTION_IDENTITY_SCHEMA
        or payload.get("formal_dataset_registration_authorized") is not False
    ):
        raise ClosureError("free-skeleton execution identity contract changed")
    spear_revision = payload.get("spear_revision")
    skintokens_revision = payload.get("skintokens_revision")
    if (
        not isinstance(spear_revision, str)
        or COMMIT_PATTERN.fullmatch(spear_revision) is None
        or not isinstance(skintokens_revision, str)
        or COMMIT_PATTERN.fullmatch(skintokens_revision) is None
    ):
        raise ClosureError("free-skeleton execution revision is invalid")
    runner_record = _logged_file_record(payload.get("runner"), "free-skeleton runner")
    patch_record = _logged_file_record(
        payload.get("runtime_patch"), "free-skeleton runtime patch"
    )
    input_record = _logged_file_record(payload.get("input"), "free-skeleton input")
    if input_record != file_record(tokenrig_input):
        raise ClosureError("free-skeleton logged input changed")
    runner_path = Path(runner_record["path"])
    if (
        not runner_path.is_absolute()
        or not Path(exact_argv[0]).is_absolute()
        or runner_path != Path(exact_argv[0])
    ):
        raise ClosureError("free-skeleton logged runner differs from exact argv")
    expected_runner = (SPEAR_ROOT / FREE_SKELETON_RUNNER_RELATIVE).resolve()
    if runner_path.resolve() != expected_runner:
        raise ClosureError("free-skeleton runner is not the canonical checked-in path")
    expected_patch = (SPEAR_ROOT / RUNTIME_PATCH_RELATIVE).resolve()
    if Path(patch_record["path"]).resolve() != expected_patch:
        raise ClosureError("free-skeleton runtime patch path is not canonical")
    if patch_record["sha256"] != patch_sha256:
        raise ClosureError("free-skeleton logged runtime patch hash changed")
    checkpoint = payload.get("model_checkpoint")
    if (
        not isinstance(checkpoint, dict)
        or set(checkpoint)
        != {"invocation_path", "resolved_payload", "snapshot_revision"}
        or checkpoint.get("invocation_path") != model_record["invocation_path"]
        or checkpoint.get("snapshot_revision") != model_record["snapshot_revision"]
        or _logged_file_record(
            checkpoint.get("resolved_payload"),
            "free-skeleton checkpoint payload",
        )
        != model_record["resolved_payload"]
    ):
        raise ClosureError("free-skeleton logged checkpoint identity changed")
    if payload.get("production_contract") != {
        "demo_run_cli_invocations": 1,
        "input_count": 1,
        "output_count": 1,
        "retry_or_ranking": False,
        "use_skeleton": False,
    }:
        raise ClosureError("free-skeleton one-shot production contract changed")
    return payload


def validate_spear_execution_identity(
    identity: dict[str, Any],
    *,
    runner_path: Path,
    patch_path: Path,
    spear_root: Path = SPEAR_ROOT,
) -> dict[str, Any]:
    spear_root = require_directory(spear_root, "SPEAR root")
    runner_path = require_regular_file(runner_path, "checked-in free-skeleton runner")
    patch_path = require_regular_file(patch_path, "checked-in TokenRig runtime patch")
    expected_runner = spear_root / FREE_SKELETON_RUNNER_RELATIVE
    expected_patch = spear_root / RUNTIME_PATCH_RELATIVE
    same_file(runner_path, expected_runner, "checked-in free-skeleton runner")
    same_file(patch_path, expected_patch, "checked-in TokenRig runtime patch")
    revision = identity.get("spear_revision")
    if not isinstance(revision, str) or COMMIT_PATTERN.fullmatch(revision) is None:
        raise ClosureError("free-skeleton SPEAR revision is invalid")
    git_output(
        spear_root,
        ["cat-file", "-e", f"{revision}^{{commit}}"],
        "free-skeleton SPEAR revision",
    )
    result = {}
    for label, relative, current, logged in (
        (
            "free-skeleton runner",
            FREE_SKELETON_RUNNER_RELATIVE,
            runner_path,
            identity.get("runner"),
        ),
        (
            "TokenRig runtime patch",
            RUNTIME_PATCH_RELATIVE,
            patch_path,
            identity.get("runtime_patch"),
        ),
    ):
        record = _logged_file_record(logged, label)
        if record != file_record(current):
            raise ClosureError(f"{label} changed after inference")
        committed = git_output(
            spear_root,
            ["show", f"{revision}:{relative}"],
            f"{label} git blob",
        )
        if (
            sha256_bytes(committed) != record["sha256"]
            or len(committed) != record["size_bytes"]
        ):
            raise ClosureError(f"{label} does not match the executed SPEAR revision")
        result[label] = {
            "git_revision": revision,
            "relative_path": str(relative),
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
    return result


def require_path_string_same_file(value: str, expected: Path, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise ClosureError(f"{label} path string is invalid")
    # Historical argv commonly traversed SPEAR/tmp, which is the repository's
    # deliberate workspace link.  It is not accepted as evidence authority:
    # the resolved inode must still be the separately authenticated canonical
    # artifact supplied by the closure.
    observed = require_regular_file(Path(value), label)
    same_file(observed, expected, label)


def model_link_chain(invocation_path: Path) -> tuple[Path, list[dict[str, str]], str]:
    absolute = invocation_path.absolute()
    links = []
    for component in _path_components(absolute):
        if component.is_symlink():
            links.append(
                {
                    "path": str(component),
                    "target": os.readlink(component),
                }
            )
    resolved = absolute.resolve(strict=True)
    resolved = require_regular_file(resolved, "resolved TokenRig checkpoint")
    snapshot = None
    for link in links:
        target_path = Path(link["target"])
        if not target_path.is_absolute():
            target_path = Path(link["path"]).parent / target_path
        match = SNAPSHOT_PATTERN.search(str(target_path.resolve(strict=False)))
        if match:
            snapshot = match.group(1)
    if snapshot is None:
        raise ClosureError("model invocation path lacks a pinned snapshot revision")
    return resolved, links, snapshot


def validate_model_link_chain(record: dict[str, Any]) -> Path:
    invocation = record.get("invocation_path")
    if not isinstance(invocation, str) or not invocation:
        raise ClosureError("model invocation path is invalid")
    resolved, links, snapshot = model_link_chain(Path(invocation))
    if links != record.get("symlink_chain"):
        raise ClosureError("model symlink chain changed")
    if snapshot != record.get("snapshot_revision"):
        raise ClosureError("model snapshot revision changed")
    require_descriptor_matches(
        record.get("resolved_payload"), resolved, "TokenRig checkpoint payload"
    )
    return resolved


def validate_load_audit(
    path: Path,
    tokenrig_input: Path,
    server_marker: dict[str, Any],
) -> list[dict[str, Any]]:
    events = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ClosureError("cannot read TokenRig load audit") from error
    if not lines:
        raise ClosureError("TokenRig load audit is empty")
    for number, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ClosureError(f"invalid load audit JSON at line {number}") from error
        if not isinstance(event, dict):
            raise ClosureError("load audit event must be an object")
        recursive_finite(event, f"load audit line {number}")
        events.append(event)
    sequences: dict[int, list[dict[str, Any]]] = {}
    for event in events:
        sequence = event.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            raise ClosureError("load audit sequence is invalid")
        sequences.setdefault(sequence, []).append(event)
        require_path_string_same_file(
            event.get("filepath"), tokenrig_input, "load audit input"
        )
        if event.get("patch_sha256") != server_marker.get("patch_sha256"):
            raise ClosureError("load audit patch hash disagrees with server marker")
        if event.get("pid") != server_marker.get("pid"):
            raise ClosureError("load audit PID disagrees with server marker")
        if event.get("generation") != server_marker.get("generation"):
            raise ClosureError("load audit generation disagrees with server marker")
    if len(sequences) != 2:
        raise ClosureError("load audit must contain exactly two load sequences")
    for sequence, records in sorted(sequences.items()):
        phases = [record.get("phase") for record in records]
        if phases != ["before_clean", "after_clean", "after_import"]:
            raise ClosureError(
                f"load audit sequence {sequence} has invalid phase ordering"
            )
        clean = records[1].get("inventory")
        if (
            not isinstance(clean, dict)
            or clean.get("objects") != []
            or clean.get("mesh_count") != 0
            or clean.get("material_count") != 0
            or clean.get("image_count") != 0
        ):
            raise ClosureError("load audit after_clean inventory is not empty")
        imported = records[2].get("inventory")
        if not isinstance(imported, dict) or imported.get("mesh_count") != 1:
            raise ClosureError("load audit after_import lacks exactly one mesh")
        objects = imported.get("objects")
        mesh_objects = (
            [
                item
                for item in objects
                if isinstance(item, dict) and item.get("type") == "MESH"
            ]
            if isinstance(objects, list)
            else []
        )
        empty_objects = (
            [
                item
                for item in objects
                if isinstance(item, dict) and item.get("type") == "EMPTY"
            ]
            if isinstance(objects, list)
            else []
        )
        if (
            not isinstance(objects, list)
            or len(mesh_objects) != 1
            or empty_objects != [{"name": "world", "type": "EMPTY"}]
            or mesh_objects[0].get("name") in {"Cube", "Camera", "Light"}
            or any(
                not isinstance(item, dict)
                or not isinstance(item.get("name"), str)
                or not item["name"]
                or item.get("type") not in {"MESH", "EMPTY"}
                for item in objects
            )
        ):
            raise ClosureError("load audit after_import inventory is contaminated")
    return events


def validate_run_evidence(
    run_log_path: Path,
    markers: list[tuple[Path, dict[str, Any]]],
    marker_dir: Path,
    tokenrig_input: Path,
    tokenrig_output: Path,
    evidence_mode: str,
    load_audit_path: Path | None,
) -> dict[str, Any]:
    run_log = run_log_path.read_text(encoding="utf-8", errors="strict")
    if "\tExit status: 0" not in run_log:
        raise ClosureError("run.log does not retain Exit status: 0")
    exported = EXPORTED_PATTERN.findall(run_log)
    if len(exported) != 1:
        raise ClosureError("run.log must retain exactly one successful export line")
    require_path_string_same_file(
        exported[0], tokenrig_output, "run.log exported TokenRig output"
    )
    environment, executable, exact_argv = parse_timed_command(run_log)
    main_marker = markers[0][1]
    server_marker = markers[1][1]
    if exact_argv != main_marker["argv"]:
        raise ClosureError("run.log argv disagrees with the main runtime marker")
    expected_environment = {
        "CUDA_VISIBLE_DEVICES": "3",
        "TOKENRIG_CANARY_SEED": str(EXPECTED_SEED),
        "TOKENRIG_SERVER_HYGIENE_SHA256": main_marker["patch_sha256"],
        "TOKENRIG_BPY_PORT": str(main_marker["bpy_port"]),
    }
    for key, expected in expected_environment.items():
        if environment.get(key) != expected:
            raise ClosureError(f"run.log environment mismatch for {key}")
    marker_env = environment.get("TOKENRIG_HYGIENE_MARKER_DIR")
    if not isinstance(marker_env, str):
        raise ClosureError("run.log omitted TOKENRIG_HYGIENE_MARKER_DIR")
    observed_marker_dir = require_directory(
        Path(marker_env),
        "run.log runtime marker directory",
    )
    same_file(observed_marker_dir, marker_dir, "runtime marker directory")
    if environment.get("TOKENRIG_LOAD_AUDIT_PATH") is None:
        raise ClosureError("run.log omitted TOKENRIG_LOAD_AUDIT_PATH")
    audit_invocation_path = Path(environment["TOKENRIG_LOAD_AUDIT_PATH"]).absolute()
    if evidence_mode in LOAD_AUDIT_MODES:
        if load_audit_path is None:
            raise ClosureError("load-audit evidence mode requires load audit")
        same_file(
            require_regular_file(audit_invocation_path, "run.log load audit"),
            load_audit_path,
            "run.log load audit",
        )
        if "generation" not in main_marker or "generation" not in server_marker:
            raise ClosureError(
                "load-audit mode runtime markers lack service generation"
            )
        validate_load_audit(load_audit_path, tokenrig_input, server_marker)
    else:
        if load_audit_path is not None:
            raise ClosureError("legacy missing-load-audit mode cannot attach an audit")
        if audit_invocation_path.exists() or audit_invocation_path.is_symlink():
            raise ClosureError("legacy load audit is not actually missing")
        if "generation" in main_marker or "generation" in server_marker:
            raise ClosureError("legacy Shiba markers unexpectedly claim generation")
    expected_flags = [
        exact_argv[0],
        "--input",
        option_value(exact_argv, "--input"),
        "--output",
        option_value(exact_argv, "--output"),
        "--top_k",
        "5",
        "--top_p",
        "0.95",
        "--temperature",
        "1.0",
        "--repetition_penalty",
        "2.0",
        "--num_beams",
        "10",
        "--use_transfer",
        "--model_ckpt",
        option_value(exact_argv, "--model_ckpt"),
        "--use_postprocess",
    ]
    if exact_argv != expected_flags:
        raise ClosureError(
            "TokenRig argv differs from the exact seed42 production form"
        )
    require_path_string_same_file(
        option_value(exact_argv, "--input"), tokenrig_input, "argv TokenRig input"
    )
    require_path_string_same_file(
        option_value(exact_argv, "--output"), tokenrig_output, "argv TokenRig output"
    )
    if not Path(executable).is_file():
        raise ClosureError("TokenRig Python executable no longer exists")
    model_path = Path(option_value(exact_argv, "--model_ckpt"))
    model_payload, model_links, snapshot = model_link_chain(model_path)
    model_record = {
        "invocation_path": str(model_path),
        "symlink_chain": model_links,
        "snapshot_revision": snapshot,
        "resolved_payload": file_record(model_payload),
    }
    if (
        model_record["resolved_payload"]["sha256"] != EXPECTED_MODEL_SHA256
        or snapshot != EXPECTED_MODEL_SNAPSHOT_REVISION
    ):
        raise ClosureError("TokenRig checkpoint is not the pinned production model")
    spear_execution_identity = None
    if evidence_mode == CHECKED_IN_FREE_SKELETON_MODE:
        spear_execution_identity = parse_free_skeleton_execution_identity(
            run_log,
            exact_argv=exact_argv,
            tokenrig_input=tokenrig_input,
            model_record=model_record,
            patch_sha256=main_marker["patch_sha256"],
        )
    elif FREE_SKELETON_IDENTITY_PATTERN.search(run_log) is not None:
        raise ClosureError(
            "checked-in free-skeleton runner requires its dedicated evidence mode"
        )
    return {
        "exact_argv": exact_argv,
        "exact_argv_sha256": sha256_bytes(canonical_bytes(exact_argv)),
        "environment": environment,
        "python_executable": file_record(
            require_regular_file(
                Path(executable).resolve(strict=True),
                "resolved TokenRig Python executable",
            )
        ),
        "model_checkpoint": model_record,
        "patch_sha256": main_marker["patch_sha256"],
        "main_marker": main_marker,
        "server_marker": server_marker,
        "load_audit_invocation_path": str(audit_invocation_path),
        "spear_execution_identity": spear_execution_identity,
    }


class EvidenceWriter:
    def __init__(self, staging: Path):
        self.staging = staging
        self.evidence = staging / "evidence"
        self.evidence.mkdir()

    def copy(self, label: str, source: Path, relative: str) -> dict[str, Any]:
        source = require_regular_file(source, label)
        destination = self.evidence / relative
        if destination.exists() or destination.is_symlink():
            raise ClosureError(f"duplicate closure evidence destination: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        closure_record = file_record(destination)
        closure_record["path"] = str(destination.relative_to(self.staging))
        return {
            "original": file_record(source),
            "closure_copy": closure_record,
        }

    def bytes(
        self,
        label: str,
        value: bytes,
        relative: str,
        provenance: dict[str, Any],
    ) -> dict[str, Any]:
        if not value:
            raise ClosureError(f"{label} bytes are empty")
        destination = self.evidence / relative
        if destination.exists() or destination.is_symlink():
            raise ClosureError(f"duplicate closure evidence destination: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(value)
        closure_record = file_record(destination)
        closure_record["path"] = str(destination.relative_to(self.staging))
        return {
            "provenance": provenance,
            "closure_copy": closure_record,
        }


def validate_closure_copy(
    closure_root: Path,
    record: dict[str, Any],
    label: str,
) -> Path:
    copy_record = record.get("closure_copy")
    if not isinstance(copy_record, dict):
        raise ClosureError(f"{label} closure copy record is missing")
    relative = copy_record.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ClosureError(f"{label} closure copy path is unsafe")
    path = require_regular_file(closure_root / relative, f"{label} closure copy")
    expected_hash = require_sha256(
        copy_record.get("sha256"), f"{label} closure copy sha256"
    )
    size = copy_record.get("size_bytes")
    if (
        sha256_file(path) != expected_hash
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or path.stat().st_size != size
    ):
        raise ClosureError(f"{label} closure copy hash/size mismatch")
    return path


def validate_original_and_copy(
    closure_root: Path,
    record: dict[str, Any],
    label: str,
) -> tuple[Path, Path]:
    original = descriptor_path(record.get("original"), f"{label} original")
    copy = validate_closure_copy(closure_root, record, label)
    if sha256_file(original) != sha256_file(copy):
        raise ClosureError(f"{label} original and closure copy differ")
    return original, copy


def seal_readonly(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            raise ClosureError(f"closure unexpectedly contains a symlink: {path}")
        if path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        elif path.is_dir():
            path.chmod(
                stat.S_IRUSR
                | stat.S_IXUSR
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH
            )
    root.chmod(
        stat.S_IRUSR
        | stat.S_IXUSR
        | stat.S_IRGRP
        | stat.S_IXGRP
        | stat.S_IROTH
        | stat.S_IXOTH
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.evidence_mode not in SUPPORTED_MODES:
        raise ClosureError("unsupported evidence mode")
    if args.evidence_mode == LEGACY_SHIBA_MODE and args.asset_id != (
        "shiba_inu_20260726_01"
    ):
        raise ClosureError("legacy missing-load-audit mode is restricted to Shiba")
    output_root = args.output_root.absolute()
    if output_root.exists() or output_root.is_symlink():
        raise ClosureError(f"refusing to replace closure root: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = output_root.parent / f".{output_root.name}.staging.{uuid.uuid4().hex}"
    if staging.exists() or staging.is_symlink():
        raise ClosureError(f"unexpected closure staging collision: {staging}")
    tokenrig_input = require_regular_file(args.tokenrig_input, "TokenRig input")
    tokenrig_output = require_regular_file(args.tokenrig_output, "TokenRig output")
    if os.path.samefile(tokenrig_input, tokenrig_output):
        raise ClosureError("TokenRig input and output are the same file")
    raw_manifest = require_regular_file(args.raw_pixal_manifest, "raw Pixal manifest")
    upstream_manifest = require_regular_file(
        args.upstream_manifest, "upstream geometry manifest"
    )
    raw_static_decision_batch = getattr(
        args, "raw_static_decision_batch", None
    )
    expected_raw_static_decision_batch_sha256 = getattr(
        args,
        "expected_raw_static_decision_batch_sha256",
        None,
    )
    if (raw_static_decision_batch is None) != (
        expected_raw_static_decision_batch_sha256 is None
    ):
        raise ClosureError(
            "raw static decision batch path/hash must be supplied together"
        )
    raw_glb, upstream_kind, extra_upstream = validate_upstream_lineage(
        raw_manifest,
        upstream_manifest,
        tokenrig_input,
        raw_static_decision_batch_path=raw_static_decision_batch,
        expected_raw_static_decision_batch_sha256=(
            expected_raw_static_decision_batch_sha256
        ),
        require_oriented_batch_external_authority=True,
    )
    readback = require_regular_file(args.geometry_readback, "geometry readback")
    readback_payload = validate_readback(readback, tokenrig_input, tokenrig_output)
    run_log = require_regular_file(args.run_log, "TokenRig run.log")
    marker_dir = require_directory(args.runtime_marker_dir, "runtime marker directory")
    markers = marker_payloads(marker_dir)
    load_audit = (
        require_regular_file(args.load_audit, "TokenRig load audit")
        if args.load_audit is not None
        else None
    )
    execution = validate_run_evidence(
        run_log,
        markers,
        marker_dir,
        tokenrig_input,
        tokenrig_output,
        args.evidence_mode,
        load_audit,
    )
    skintokens_root = require_directory(args.skintokens_root, "SkinTokens root")
    revision = git_revision(skintokens_root)
    if (
        args.evidence_mode == CHECKED_IN_FREE_SKELETON_MODE
        and execution["spear_execution_identity"]["skintokens_revision"] != revision
    ):
        raise ClosureError(
            "free-skeleton logged SkinTokens revision changed before closure"
        )
    demo = require_regular_file(skintokens_root / "demo.py", "SkinTokens demo.py")
    bpy_server = require_regular_file(
        skintokens_root / "src/server/bpy_server.py", "SkinTokens bpy_server.py"
    )
    runner_path = Path(execution["exact_argv"][0])
    if runner_path.name == "demo.py" and not runner_path.is_absolute():
        runner = demo
    else:
        runner = require_regular_file(
            runner_path,
            "TokenRig runner",
        )
    if args.runtime_patch is not None and args.runtime_patch_git_spec is not None:
        raise ClosureError("choose a runtime patch file or git spec, not both")
    if args.runtime_patch is None and args.runtime_patch_git_spec is None:
        raise ClosureError("runtime patch evidence is required")
    spear_git_identity = None
    checked_runtime_patch = None
    if args.evidence_mode == CHECKED_IN_FREE_SKELETON_MODE:
        if args.runtime_patch is None or args.runtime_patch_git_spec is not None:
            raise ClosureError(
                "checked-in free-skeleton mode requires the canonical runtime patch"
            )
        checked_runtime_patch = require_regular_file(
            args.runtime_patch, "checked-in TokenRig runtime patch"
        )
        spear_git_identity = validate_spear_execution_identity(
            execution["spear_execution_identity"],
            runner_path=runner,
            patch_path=checked_runtime_patch,
        )
    staging.mkdir()
    writer = EvidenceWriter(staging)
    try:
        records: dict[str, Any] = {
            "raw_pixal_manifest": writer.copy(
                "raw Pixal manifest", raw_manifest, "lineage/raw_pixal_manifest.json"
            ),
            "upstream_manifest": writer.copy(
                "upstream geometry manifest",
                upstream_manifest,
                "lineage/upstream_geometry_manifest.json",
            ),
            "geometry_readback": writer.copy(
                "geometry readback", readback, "readback/tokenrig_binding_readback.json"
            ),
            "run_log": writer.copy("TokenRig run.log", run_log, "execution/run.log"),
            "runtime_markers": [
                writer.copy(
                    "runtime marker",
                    path,
                    f"execution/runtime_markers/{path.name}",
                )
                for path, _payload in markers
            ],
            "skintokens_demo": writer.copy(
                "SkinTokens demo.py", demo, "software/skintokens_demo.py"
            ),
            "skintokens_bpy_server": writer.copy(
                "SkinTokens bpy_server.py",
                bpy_server,
                "software/skintokens_bpy_server.py",
            ),
        }
        if runner != demo:
            records["runner"] = writer.copy(
                "TokenRig runner", runner, "execution/runner.py"
            )
        else:
            records["runner"] = records["skintokens_demo"]
        if spear_git_identity is not None:
            records["runner"]["provenance"] = {
                "kind": "spear_checked_in_git_blob_v1",
                **spear_git_identity["free-skeleton runner"],
            }
        if load_audit is not None:
            records["load_audit"] = writer.copy(
                "TokenRig load audit",
                load_audit,
                "execution/load_audit.jsonl",
            )
        else:
            records["load_audit"] = {
                "status": "missing_historical_artifact",
                "invocation_path": execution["load_audit_invocation_path"],
                "fabricated_or_reconstructed": False,
            }
        if extra_upstream:
            records["extra_upstream_manifests"] = [
                writer.copy(
                    "extra upstream manifest",
                    path,
                    f"lineage/extra_upstream_{index:02d}.json",
                )
                for index, path in enumerate(extra_upstream, start=1)
            ]
        else:
            records["extra_upstream_manifests"] = []
        if args.runtime_patch is not None:
            runtime_patch = checked_runtime_patch or require_regular_file(
                args.runtime_patch, "TokenRig runtime patch"
            )
            records["runtime_patch"] = writer.copy(
                "TokenRig runtime patch",
                runtime_patch,
                "software/runtime_patch.py",
            )
            if spear_git_identity is None:
                records["runtime_patch"]["provenance"] = {
                    "kind": "regular_file",
                }
            else:
                records["runtime_patch"]["provenance"] = {
                    "kind": "spear_checked_in_git_blob_v1",
                    **spear_git_identity["TokenRig runtime patch"],
                }
        else:
            patch_bytes = git_blob(
                Path(__file__).resolve().parents[1],
                args.runtime_patch_git_spec,
                "historical runtime patch",
            )
            records["runtime_patch"] = writer.bytes(
                "historical runtime patch",
                patch_bytes,
                "software/runtime_patch.py",
                {
                    "kind": "spear_git_blob",
                    "git_spec": args.runtime_patch_git_spec,
                },
            )
        patch_copy = staging / records["runtime_patch"]["closure_copy"]["path"]
        if sha256_file(patch_copy) != execution["patch_sha256"]:
            raise ClosureError(
                "runtime patch bytes do not match marker/run.log patch SHA-256"
            )
        payload = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "asset_id": args.asset_id,
            "status": "passed_source_to_tokenrig_closure",
            "evidence_mode": {
                "mode": args.evidence_mode,
                "load_audit_present": load_audit is not None,
                "missing_evidence": (
                    [] if load_audit is not None else ["load_audit.jsonl"]
                ),
                "legacy_exception_is_explicit": (
                    args.evidence_mode == LEGACY_SHIBA_MODE
                ),
                "no_missing_evidence_was_fabricated": True,
            },
            "lineage": {
                "upstream_kind": upstream_kind,
                "raw_pixal_glb": file_record(raw_glb),
                "tokenrig_input": file_record(tokenrig_input),
                "tokenrig_output": file_record(tokenrig_output),
                "geometry_readback_manifest_sha256": readback_payload[
                    "manifest_sha256"
                ],
            },
            "execution": {
                "seed": EXPECTED_SEED,
                "exact_argv": execution["exact_argv"],
                "exact_argv_sha256": execution["exact_argv_sha256"],
                "environment": execution["environment"],
                "python_executable": execution["python_executable"],
                "model_checkpoint": execution["model_checkpoint"],
                "runtime_patch_sha256": execution["patch_sha256"],
                "spear_execution_identity": execution["spear_execution_identity"],
                "skintokens": {
                    "root": str(skintokens_root),
                    "revision": revision,
                    "revision_is_full_commit": True,
                },
            },
            "evidence": records,
            "claim_boundary": (
                "This closure authenticates raw Pixal/geometry lineage, the exact "
                "seed42 TokenRig execution evidence, and geometry-preserving static "
                "rig output. It does not approve animation, UE import/runtime, "
                "emitter behavior, owner review, or formal dataset registration."
            ),
            "formal_dataset_registration_authorized": False,
        }
        recursive_finite(payload)
        payload["manifest_sha256"] = hash_without(payload, "manifest_sha256")
        manifest = staging / "manifest.json"
        manifest.write_bytes(canonical_bytes(payload) + b"\n")
        external_hash = sha256_file(manifest)
        validate_closure(
            staging,
            expected_manifest_sha256=external_hash,
            allow_staging_name=True,
        )
        if output_root.exists() or output_root.is_symlink():
            raise ClosureError(f"refusing concurrently-created closure: {output_root}")
        os.replace(staging, output_root)
        seal_readonly(output_root)
        return {
            "status": "passed",
            "closure_root": str(output_root),
            "manifest": str(output_root / "manifest.json"),
            "manifest_file_sha256": external_hash,
            "manifest_sha256": payload["manifest_sha256"],
            "tokenrig_input_sha256": payload["lineage"]["tokenrig_input"]["sha256"],
            "tokenrig_output_sha256": payload["lineage"]["tokenrig_output"]["sha256"],
        }
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def validate_execution_identity_fields(
    execution: dict[str, Any],
    reconstructed: dict[str, Any],
) -> None:
    """Compare serialized execution identity with independently retained evidence."""
    if execution.get("seed") != EXPECTED_SEED:
        raise ClosureError("closure seed is not exactly 42")
    exact_argv = execution.get("exact_argv")
    if exact_argv != reconstructed.get("exact_argv"):
        raise ClosureError("closure exact argv disagrees with retained evidence")
    if execution.get("exact_argv_sha256") != sha256_bytes(canonical_bytes(exact_argv)):
        raise ClosureError("closure exact argv hash is internally inconsistent")
    if execution.get("exact_argv_sha256") != reconstructed.get(
        "exact_argv_sha256"
    ) or execution.get("environment") != reconstructed.get("environment"):
        raise ClosureError(
            "closure execution identity disagrees with retained evidence"
        )
    if execution.get("runtime_patch_sha256") != reconstructed.get("patch_sha256"):
        raise ClosureError("closure runtime patch hash is disconnected")
    if execution.get("model_checkpoint") != reconstructed.get("model_checkpoint"):
        raise ClosureError("closure model identity disagrees with exact argv")
    if execution.get("spear_execution_identity") != reconstructed.get(
        "spear_execution_identity"
    ):
        raise ClosureError("closure SPEAR execution identity disagrees with run.log")


def validate_closure(
    closure_root: Path,
    *,
    expected_manifest_sha256: str,
    allow_staging_name: bool = False,
) -> dict[str, Any]:
    expected_manifest_sha256 = require_sha256(
        expected_manifest_sha256, "expected closure manifest file sha256"
    )
    closure_root = require_directory(closure_root, "TokenRig closure root")
    if not allow_staging_name and ".staging." in closure_root.name:
        raise ClosureError("staging directories are not valid closures")
    manifest_path = require_regular_file(
        closure_root / "manifest.json", "TokenRig closure manifest"
    )
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ClosureError(
            "closure manifest file SHA-256 disagrees with external authority"
        )
    payload = load_json(manifest_path, "TokenRig closure manifest")
    if payload.get("schema") != SCHEMA:
        raise ClosureError("unexpected TokenRig closure schema")
    if payload.get("manifest_sha256") != hash_without(payload, "manifest_sha256"):
        raise ClosureError("TokenRig closure embedded self-hash mismatch")
    if payload.get("status") != "passed_source_to_tokenrig_closure":
        raise ClosureError("TokenRig closure status is not passed")
    if payload.get("formal_dataset_registration_authorized") is not False:
        raise ClosureError("TokenRig closure exceeded its claim boundary")
    asset_id = payload.get("asset_id")
    if not isinstance(asset_id, str) or not asset_id:
        raise ClosureError("TokenRig closure asset_id is invalid")
    mode_record = payload.get("evidence_mode")
    if not isinstance(mode_record, dict):
        raise ClosureError("TokenRig closure evidence_mode is missing")
    mode = mode_record.get("mode")
    if mode not in SUPPORTED_MODES:
        raise ClosureError("unsupported TokenRig closure evidence mode")
    if mode == LEGACY_SHIBA_MODE and asset_id != "shiba_inu_20260726_01":
        raise ClosureError("legacy missing-load-audit closure is not Shiba")
    if mode_record.get("no_missing_evidence_was_fabricated") is not True:
        raise ClosureError("closure does not deny fabricated missing evidence")
    evidence = payload.get("evidence")
    lineage = payload.get("lineage")
    execution = payload.get("execution")
    if not all(isinstance(item, dict) for item in (evidence, lineage, execution)):
        raise ClosureError("TokenRig closure is missing required sections")
    raw_manifest, _raw_copy = validate_original_and_copy(
        closure_root, evidence.get("raw_pixal_manifest"), "raw Pixal manifest"
    )
    upstream_manifest, _upstream_copy = validate_original_and_copy(
        closure_root, evidence.get("upstream_manifest"), "upstream manifest"
    )
    readback, _readback_copy = validate_original_and_copy(
        closure_root, evidence.get("geometry_readback"), "geometry readback"
    )
    run_log, _run_copy = validate_original_and_copy(
        closure_root, evidence.get("run_log"), "TokenRig run.log"
    )
    tokenrig_input = descriptor_path(
        lineage.get("tokenrig_input"), "closure TokenRig input"
    )
    tokenrig_output = descriptor_path(
        lineage.get("tokenrig_output"), "closure TokenRig output"
    )
    raw_glb, upstream_kind, extra_upstream = validate_upstream_lineage(
        raw_manifest, upstream_manifest, tokenrig_input
    )
    require_descriptor_matches(
        lineage.get("raw_pixal_glb"), raw_glb, "closure raw Pixal GLB"
    )
    if upstream_kind != lineage.get("upstream_kind"):
        raise ClosureError("closure upstream kind disagrees with manifest lineage")
    extra_records = evidence.get("extra_upstream_manifests")
    if not isinstance(extra_records, list) or len(extra_records) != len(extra_upstream):
        raise ClosureError("closure extra upstream evidence count mismatch")
    for index, (record, expected_path) in enumerate(
        zip(extra_records, extra_upstream)
    ):
        original, _copy = validate_original_and_copy(
            closure_root, record, f"extra upstream manifest {index}"
        )
        same_file(original, expected_path, f"extra upstream manifest {index}")
    readback_payload = validate_readback(readback, tokenrig_input, tokenrig_output)
    if (
        lineage.get("geometry_readback_manifest_sha256")
        != readback_payload["manifest_sha256"]
    ):
        raise ClosureError("closure readback manifest hash is disconnected")
    marker_records = evidence.get("runtime_markers")
    if not isinstance(marker_records, list) or len(marker_records) != 2:
        raise ClosureError("closure runtime marker evidence is invalid")
    marker_originals = []
    for index, record in enumerate(marker_records):
        original, _copy = validate_original_and_copy(
            closure_root, record, f"runtime marker {index}"
        )
        marker_originals.append(original)
    marker_dir = marker_originals[0].parent
    if any(path.parent != marker_dir for path in marker_originals):
        raise ClosureError("runtime markers do not share one directory")
    markers = marker_payloads(marker_dir)
    expected_marker_names = sorted(path.name for path in marker_originals)
    if sorted(path.name for path, _payload in markers) != expected_marker_names:
        raise ClosureError("runtime marker directory membership changed")
    load_record = evidence.get("load_audit")
    load_audit = None
    if mode in LOAD_AUDIT_MODES:
        if mode_record.get("load_audit_present") is not True:
            raise ClosureError("load-audit mode does not claim a load audit")
        load_audit, _copy = validate_original_and_copy(
            closure_root, load_record, "TokenRig load audit"
        )
        if mode_record.get("missing_evidence") != []:
            raise ClosureError("load-audit mode claims missing evidence")
    else:
        if (
            mode_record.get("load_audit_present") is not False
            or mode_record.get("missing_evidence") != ["load_audit.jsonl"]
            or mode_record.get("legacy_exception_is_explicit") is not True
            or not isinstance(load_record, dict)
            or load_record.get("status") != "missing_historical_artifact"
            or load_record.get("fabricated_or_reconstructed") is not False
        ):
            raise ClosureError("legacy missing-load-audit record is invalid")
        invocation_path = load_record.get("invocation_path")
        if (
            not isinstance(invocation_path, str)
            or Path(invocation_path).exists()
            or Path(invocation_path).is_symlink()
        ):
            raise ClosureError("legacy load audit is not demonstrably missing")
    reconstructed = validate_run_evidence(
        run_log,
        markers,
        marker_dir,
        tokenrig_input,
        tokenrig_output,
        mode,
        load_audit,
    )
    validate_execution_identity_fields(execution, reconstructed)
    model_payload = validate_model_link_chain(execution.get("model_checkpoint", {}))
    same_file(
        model_payload,
        Path(reconstructed["model_checkpoint"]["resolved_payload"]["path"]),
        "TokenRig checkpoint payload",
    )
    patch_record = evidence.get("runtime_patch")
    patch_copy = validate_closure_copy(
        closure_root, patch_record, "TokenRig runtime patch"
    )
    if sha256_file(patch_copy) != reconstructed["patch_sha256"]:
        raise ClosureError("closure runtime patch bytes disagree with markers")
    provenance = patch_record.get("provenance")
    if not isinstance(provenance, dict):
        raise ClosureError("runtime patch provenance is missing")
    runtime_patch_original = None
    if provenance.get("kind") == "regular_file":
        runtime_patch_original = descriptor_path(
            patch_record.get("original"), "runtime patch original"
        )
        if sha256_file(runtime_patch_original) != sha256_file(patch_copy):
            raise ClosureError("runtime patch original and closure copy differ")
    elif provenance.get("kind") == "spear_git_blob":
        spec = provenance.get("git_spec")
        if not isinstance(spec, str):
            raise ClosureError("historical runtime patch git spec is invalid")
        historical = git_blob(
            Path(__file__).resolve().parents[1], spec, "historical runtime patch"
        )
        if sha256_bytes(historical) != sha256_file(patch_copy):
            raise ClosureError("historical runtime patch git blob changed")
    elif provenance.get("kind") == "spear_checked_in_git_blob_v1":
        runtime_patch_original = descriptor_path(
            patch_record.get("original"), "runtime patch original"
        )
        if sha256_file(runtime_patch_original) != sha256_file(patch_copy):
            raise ClosureError("runtime patch original and closure copy differ")
    else:
        raise ClosureError("unsupported runtime patch provenance")
    skintokens = execution.get("skintokens")
    if not isinstance(skintokens, dict):
        raise ClosureError("SkinTokens revision record is missing")
    skintokens_root = require_directory(
        Path(skintokens.get("root", "")), "SkinTokens root"
    )
    revision = skintokens.get("revision")
    if not isinstance(revision, str) or not COMMIT_PATTERN.fullmatch(revision):
        raise ClosureError("SkinTokens revision is not a full commit")
    if (
        mode == CHECKED_IN_FREE_SKELETON_MODE
        and reconstructed["spear_execution_identity"]["skintokens_revision"] != revision
    ):
        raise ClosureError(
            "free-skeleton logged SkinTokens revision disagrees with closure"
        )
    git_output(
        skintokens_root,
        ["cat-file", "-e", f"{revision}^{{commit}}"],
        "SkinTokens pinned revision",
    )
    demo_original, demo_copy = validate_original_and_copy(
        closure_root, evidence.get("skintokens_demo"), "SkinTokens demo.py"
    )
    bpy_original, bpy_copy = validate_original_and_copy(
        closure_root,
        evidence.get("skintokens_bpy_server"),
        "SkinTokens bpy_server.py",
    )
    same_file(demo_original, skintokens_root / "demo.py", "SkinTokens demo.py")
    same_file(
        bpy_original,
        skintokens_root / "src/server/bpy_server.py",
        "SkinTokens bpy_server.py",
    )
    for relative, copy in (
        ("demo.py", demo_copy),
        ("src/server/bpy_server.py", bpy_copy),
    ):
        committed = git_output(
            skintokens_root,
            ["show", f"{revision}:{relative}"],
            f"SkinTokens revision {relative}",
        )
        if sha256_bytes(committed) != sha256_file(copy):
            raise ClosureError(
                f"closure {relative} does not belong to the pinned revision"
            )
    runner_record = evidence.get("runner")
    runner_original, _runner_copy = validate_original_and_copy(
        closure_root, runner_record, "TokenRig runner"
    )
    runner_argv = reconstructed["exact_argv"][0]
    if Path(runner_argv).name == "demo.py" and not Path(runner_argv).is_absolute():
        same_file(runner_original, demo_original, "TokenRig runner")
    else:
        same_file(
            runner_original,
            require_regular_file(
                Path(runner_argv),
                "argv TokenRig runner",
            ),
            "TokenRig runner",
        )
    runner_provenance = runner_record.get("provenance")
    if mode == CHECKED_IN_FREE_SKELETON_MODE:
        if (
            not isinstance(runner_provenance, dict)
            or runner_provenance.get("kind") != "spear_checked_in_git_blob_v1"
            or provenance.get("kind") != "spear_checked_in_git_blob_v1"
            or runtime_patch_original is None
        ):
            raise ClosureError(
                "checked-in free-skeleton closure lacks Git blob provenance"
            )
        git_identity = validate_spear_execution_identity(
            reconstructed["spear_execution_identity"],
            runner_path=runner_original,
            patch_path=runtime_patch_original,
        )
        for record, expected in (
            (runner_provenance, git_identity["free-skeleton runner"]),
            (provenance, git_identity["TokenRig runtime patch"]),
        ):
            if record != {
                "kind": "spear_checked_in_git_blob_v1",
                **expected,
            }:
                raise ClosureError(
                    "checked-in free-skeleton Git provenance record changed"
                )
    elif runner_provenance is not None:
        raise ClosureError("historical TokenRig runner claims unexpected provenance")
    return payload


def validate_tokenrig_closure_manifest(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_target_rig_glb: Path,
) -> dict[str, Any]:
    """Validate one closure and return its canonical downstream descriptor.

    The manifest-file hash is deliberately external authority.  The target rig
    is authenticated by actual file identity plus SHA-256 and size; merely
    placing a same-named or same-byte GLB inside the workspace is insufficient.
    """
    manifest_path = require_regular_file(
        Path(manifest_path), "TokenRig closure manifest"
    )
    closure_root = manifest_path.parent
    canonical_manifest = require_regular_file(
        closure_root / "manifest.json", "TokenRig closure canonical manifest"
    )
    same_file(
        manifest_path,
        canonical_manifest,
        "TokenRig closure canonical manifest",
    )
    payload = validate_closure(
        closure_root,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    target_rig = require_regular_file(
        Path(expected_target_rig_glb), "expected target rig GLB"
    )
    require_descriptor_matches(
        payload["lineage"]["tokenrig_output"],
        target_rig,
        "expected target rig GLB",
    )
    readback_record = payload["evidence"]["geometry_readback"]["original"]
    descriptor = {
        "schema": DESCRIPTOR_SCHEMA,
        "status": "passed",
        "asset_id": payload["asset_id"],
        "evidence_mode": payload["evidence_mode"]["mode"],
        "closure_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
            "manifest_sha256": payload["manifest_sha256"],
        },
        "lineage": {
            "raw_pixal_glb": payload["lineage"]["raw_pixal_glb"],
            "tokenrig_input": payload["lineage"]["tokenrig_input"],
            "tokenrig_output": payload["lineage"]["tokenrig_output"],
            "upstream_kind": payload["lineage"]["upstream_kind"],
        },
        "geometry_readback": {
            "path": readback_record["path"],
            "sha256": readback_record["sha256"],
            "size_bytes": readback_record["size_bytes"],
            "manifest_sha256": payload["lineage"]["geometry_readback_manifest_sha256"],
        },
        "execution": {
            "seed": payload["execution"]["seed"],
            "exact_argv_sha256": payload["execution"]["exact_argv_sha256"],
            "model_checkpoint_sha256": payload["execution"]["model_checkpoint"][
                "resolved_payload"
            ]["sha256"],
            "model_snapshot_revision": payload["execution"]["model_checkpoint"][
                "snapshot_revision"
            ],
            "runtime_patch_sha256": payload["execution"]["runtime_patch_sha256"],
            "skintokens_revision": payload["execution"]["skintokens"]["revision"],
        },
        "formal_dataset_registration_authorized": False,
    }
    descriptor["descriptor_sha256"] = hash_without(descriptor, "descriptor_sha256")
    return descriptor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--asset-id", required=True)
    build_parser.add_argument("--output-root", type=Path, required=True)
    build_parser.add_argument("--raw-pixal-manifest", type=Path, required=True)
    build_parser.add_argument("--upstream-manifest", type=Path, required=True)
    build_parser.add_argument("--raw-static-decision-batch", type=Path)
    build_parser.add_argument(
        "--expected-raw-static-decision-batch-sha256"
    )
    build_parser.add_argument("--tokenrig-input", type=Path, required=True)
    build_parser.add_argument("--tokenrig-output", type=Path, required=True)
    build_parser.add_argument("--geometry-readback", type=Path, required=True)
    build_parser.add_argument("--run-log", type=Path, required=True)
    build_parser.add_argument("--runtime-marker-dir", type=Path, required=True)
    build_parser.add_argument("--load-audit", type=Path)
    build_parser.add_argument(
        "--evidence-mode", choices=sorted(SUPPORTED_MODES), required=True
    )
    build_parser.add_argument("--skintokens-root", type=Path, required=True)
    patch_group = build_parser.add_mutually_exclusive_group(required=True)
    patch_group.add_argument("--runtime-patch", type=Path)
    patch_group.add_argument("--runtime-patch-git-spec")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--closure-root", type=Path, required=True)
    validate_parser.add_argument(
        "--expected-manifest-sha256",
        required=True,
        help="Externally retained SHA-256 of manifest.json bytes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "build":
        result = build(args)
    else:
        payload = validate_closure(
            args.closure_root,
            expected_manifest_sha256=args.expected_manifest_sha256,
        )
        result = {
            "status": "passed",
            "closure_root": str(args.closure_root.absolute()),
            "asset_id": payload["asset_id"],
            "manifest_file_sha256": args.expected_manifest_sha256,
            "manifest_sha256": payload["manifest_sha256"],
        }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ClosureError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
