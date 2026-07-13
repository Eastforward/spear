#!/usr/bin/env python3
"""Build native Walking/Idle runtimes for controlled Rocketbox materials."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts
from tools import execute_controlled_rocketbox_material_jobs as materials
from tools import prepare_controlled_source_asset_execution as preparation


HANDOFF_SCHEMA = "avengine_controlled_rocketbox_runtime_handoff_v1"
EXPECTED_RUNTIME_SCHEMA = "rocketbox_native_material_variant_v1"
EXPECTED_RUNTIME_FILENAME = "runtime.glb"


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256(
        {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    value = path.relative_to(root.resolve()).as_posix() if root is not None else str(path)
    return {
        "path": value,
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _authenticate_relative_record(
    record: Mapping[str, Any], root: Path, label: str
) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise contracts.ContractError(f"{label} record fields are invalid")
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise contracts.ContractError(f"{label} path is not safely relative")
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise contracts.ContractError(f"{label} escapes its root") from error
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"{label} is missing or unsafe: {path}")
    if (
        path.stat().st_size != record["size_bytes"]
        or _sha256_file(path) != record["sha256"]
    ):
        raise contracts.ContractError(f"{label} hash or size changed")
    return path


def load_material_batch(path: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"material batch manifest is missing: {path}")
    root = path.parent
    batch = contracts.load_json(path)
    if (
        not isinstance(batch, dict)
        or batch.get("schema") != materials.BATCH_SCHEMA
        or batch.get("status") != "passed"
        or batch.get("automatic_checks", {}).get("overall") != "passed"
        or batch.get("batch_sha256") != _hash_without(batch, "batch_sha256")
    ):
        raise contracts.ContractError("material batch manifest contract/hash is invalid")
    variants = batch.get("variants")
    if (
        not isinstance(variants, list)
        or len(variants) != batch.get("unique_variant_count")
        or len({item.get("variant_key") for item in variants}) != len(variants)
    ):
        raise contracts.ContractError("material batch variants are invalid")

    records = []
    for variant in variants:
        attempt_path = _authenticate_relative_record(
            variant["attempt_manifest"], root, "material attempt manifest"
        )
        texture_path = _authenticate_relative_record(
            variant["body_color_texture"], root, "body color texture"
        )
        request_path = _authenticate_relative_record(
            variant["variant_request"], root, "variant request"
        )
        attempt = contracts.load_json(attempt_path)
        if (
            not isinstance(attempt, dict)
            or attempt.get("schema") != materials.ATTEMPT_SCHEMA
            or attempt.get("status") != "passed"
            or attempt.get("variant_key") != variant["variant_key"]
            or attempt.get("variant_id") != variant["variant_id"]
            or attempt.get("runtime_tag") != variant["runtime_tag"]
            or attempt.get("attempt_sha256")
            != _hash_without(attempt, "attempt_sha256")
            or attempt.get("automatic_checks", {}).get("overall") != "passed"
        ):
            raise contracts.ContractError("material attempt manifest contract/hash is invalid")
        request = contracts.load_json(request_path)
        if (
            not isinstance(request, dict)
            or request.get("schema_version")
            != "rocketbox_native_body_color_variant_v1"
            or request.get("variant_id") != variant["variant_id"]
            or request.get("tag") != variant["runtime_tag"]
            or request.get("body_color_texture_sha256") != _sha256_file(texture_path)
            or request.get("body_color_texture_size_bytes")
            != texture_path.stat().st_size
            or request.get("controlled_source", {}).get("variant_key")
            != variant["variant_key"]
        ):
            raise contracts.ContractError("runtime variant request is invalid")
        expected_command = [
            str(materials.native.BLENDER_PATH),
            "--background",
            "--factory-startup",
            "--python",
            str(
                Path(materials.__file__).resolve().parent
                / "blender_build_native_rocketbox_runtime.py"
            ),
            "--",
            "--body-color-texture",
            str(texture_path),
            "--variant-manifest",
            str(request_path),
        ]
        recorded_handoff = attempt.get("runtime_handoff", {})
        if (
            recorded_handoff.get("command") != expected_command
            or recorded_handoff.get("status") != "pending"
            or recorded_handoff.get("expected_actions")
            != ["Walking", "Standing_Idle"]
            or recorded_handoff.get("actor_scale") != 1.0
        ):
            raise contracts.ContractError("runtime handoff command changed")
        records.append(
            {
                "variant": variant,
                "attempt": attempt,
                "attempt_path": attempt_path,
                "texture_path": texture_path,
                "request_path": request_path,
                "command": expected_command,
                "expected_output": Path(recorded_handoff["expected_output"]).resolve(),
            }
        )
    return root, batch, records


def _validate_runtime(record: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(record["expected_output"])
    runtime_root = output.parent
    manifest_path = runtime_root / "variant_manifest.json"
    if output.name != EXPECTED_RUNTIME_FILENAME or output.is_symlink() or not output.is_file():
        raise contracts.ContractError(f"native runtime output is missing: {output}")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise contracts.ContractError(f"native runtime manifest is missing: {manifest_path}")
    manifest = contracts.load_json(manifest_path)
    variant = record["variant"]
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != EXPECTED_RUNTIME_SCHEMA
        or manifest.get("tag") != variant["runtime_tag"]
        or manifest.get("variant_id") != variant["variant_id"]
        or manifest.get("asset_id") != "rocketbox_male_adult_01"
        or manifest.get("usage_scope") != "research_candidate"
        or manifest.get("formal_registration_authorized") is not False
        or manifest.get("automatic_checks", {}).get("overall") != "passed"
        or manifest.get("automatic_checks", {}).get("variant_equivalence") != "passed"
    ):
        raise contracts.ContractError("native runtime manifest contract is invalid")
    runtime = manifest.get("runtime_glb", {})
    actions = manifest.get("actions", {})
    if (
        runtime.get("filename") != EXPECTED_RUNTIME_FILENAME
        or runtime.get("sha256") != _sha256_file(output)
        or runtime.get("size_bytes") != output.stat().st_size
        or set(actions) != {"Walking", "Standing_Idle", "set"}
        or actions.get("set", {}).get("action_count") != 2
        or actions.get("set", {}).get("walk_action") != "Walking"
        or actions.get("set", {}).get("idle_action") != "Standing_Idle"
    ):
        raise contracts.ContractError("native runtime GLB/action readback is invalid")
    return {
        "variant_key": variant["variant_key"],
        "variant_id": variant["variant_id"],
        "runtime_tag": variant["runtime_tag"],
        "sampled_attributes": variant["sampled_attributes"],
        "consumer_request_count": variant["consumer_request_count"],
        "status": "passed",
        "runtime_glb": _record(output),
        "runtime_manifest": _record(manifest_path),
        "mesh_skin_action_contract_sha256": manifest["glb_contract"][
            "mesh_skin_action_contract_sha256"
        ],
        "actions": ["Standing_Idle", "Walking"],
        "front_axis": "negative_y",
        "actor_scale": 1.0,
        "variant_equivalence": manifest["variant"]["equivalence_to_original"],
    }


def run_handoffs(
    material_batch_path: Path,
    output_root: Path,
    *,
    workers: int,
    resume: bool,
) -> Path:
    if not 1 <= workers <= 4:
        raise contracts.ContractError("workers must be in [1, 4]")
    material_root, batch, records = load_material_batch(material_batch_path)
    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(
            f"refusing to replace existing output directory: {output_root}"
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )

    def execute(record: Mapping[str, Any]) -> dict[str, Any]:
        variant_id = record["variant"]["variant_id"]
        output = Path(record["expected_output"])
        if output.exists() or output.is_symlink():
            if not resume:
                raise contracts.ContractError(
                    f"native runtime output already exists; use --resume after validation: {output}"
                )
            result = _validate_runtime(record)
            result["execution"] = "reused_validated"
            return result
        log_path = staging / "logs" / f"{variant_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("xb") as log:
            completed = subprocess.run(
                record["command"],
                cwd=materials.SPEAR_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
                check=False,
            )
            log.flush()
            os.fsync(log.fileno())
        if completed.returncode != 0:
            raise contracts.ContractError(
                f"native runtime builder failed for {variant_id} with {completed.returncode}"
            )
        result = _validate_runtime(record)
        result["execution"] = "built"
        result["log"] = _record(log_path, root=staging)
        return result

    try:
        results = []
        with ThreadPoolExecutor(max_workers=min(workers, len(records))) as executor:
            futures = {executor.submit(execute, record): record for record in records}
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item["variant_key"])
        manifest: dict[str, Any] = {
            "schema": HANDOFF_SCHEMA,
            "status": "passed",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "material_batch": _record(Path(material_batch_path)),
            "material_batch_sha256": batch["batch_sha256"],
            "material_root": str(material_root),
            "workers": workers,
            "resume": resume,
            "runtime_count": len(results),
            "runtimes": results,
            "automatic_checks": {
                "all_material_attempts_reauthenticated": True,
                "all_native_runtime_builds_passed": True,
                "all_variants_equivalent_to_original_mesh_skin_actions": True,
                "all_actions_exact": all(
                    result["actions"] == ["Standing_Idle", "Walking"]
                    for result in results
                ),
                "baseline_written": False,
                "overall": "passed",
            },
        }
        manifest["handoff_sha256"] = _hash_without(manifest, "handoff_sha256")
        contracts.write_json_no_replace(staging / "runtime_handoff_manifest.json", manifest)
        materials.native._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError(
                f"refusing to replace concurrently-created output: {output_root}"
            )
        os.rename(staging, output_root)
        return output_root / "runtime_handoff_manifest.json"
    except Exception:
        materials.native._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--material-batch", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest_path = run_handoffs(
            args.material_batch,
            args.output_root,
            workers=args.workers,
            resume=args.resume,
        )
        manifest = contracts.load_json(manifest_path)
    except (contracts.ContractError, OSError, subprocess.SubprocessError) as error:
        print(f"CONTROLLED_ROCKETBOX_RUNTIME_FAILED {error}", file=sys.stderr)
        return 2
    built = sum(item["execution"] == "built" for item in manifest["runtimes"])
    reused = sum(item["execution"] == "reused_validated" for item in manifest["runtimes"])
    print(
        "CONTROLLED_ROCKETBOX_RUNTIME_OK "
        f"runtimes={manifest['runtime_count']} built={built} reused={reused} "
        f"output={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
