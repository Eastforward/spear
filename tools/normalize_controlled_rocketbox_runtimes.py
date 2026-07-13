#!/usr/bin/env python3
"""Publish UE in-place/metric bundles for controlled Rocketbox runtimes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import build_ue_inplace_rocketbox_runtime as ue_runtime
from tools import controlled_source_asset_schema as contracts
from tools import execute_controlled_rocketbox_material_jobs as materials
from tools import run_controlled_rocketbox_runtime_handoffs as native_handoff


NORMALIZED_HANDOFF_SCHEMA = "avengine_controlled_rocketbox_ue_runtime_handoff_v1"
UE_RUNTIME_ROOT = materials.SPEAR_ROOT / "tmp/rocketbox_native_runtime_ue_v3"


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _hash_without(value: Mapping[str, Any], key: str) -> str:
    return _json_sha256(
        {name: copy.deepcopy(item) for name, item in value.items() if name != key}
    )


def _sha256_file(path: Path) -> str:
    return native_handoff._sha256_file(path)


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_native_handoff(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"native handoff manifest is missing: {path}")
    manifest = contracts.load_json(path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != native_handoff.HANDOFF_SCHEMA
        or manifest.get("status") != "passed"
        or manifest.get("handoff_sha256") != _hash_without(manifest, "handoff_sha256")
        or manifest.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise contracts.ContractError("native handoff manifest contract/hash is invalid")
    runtimes = manifest.get("runtimes")
    if (
        not isinstance(runtimes, list)
        or len(runtimes) != manifest.get("runtime_count")
        or len({item.get("variant_key") for item in runtimes}) != len(runtimes)
    ):
        raise contracts.ContractError("native handoff runtime list is invalid")
    for runtime in runtimes:
        for role in ("runtime_glb", "runtime_manifest"):
            record = runtime.get(role)
            if not isinstance(record, dict) or set(record) != {
                "path",
                "sha256",
                "size_bytes",
            }:
                raise contracts.ContractError(f"native {role} record is invalid")
            artifact = Path(record["path"]).resolve()
            if artifact.is_symlink() or not artifact.is_file():
                raise contracts.ContractError(f"native {role} is missing: {artifact}")
            if (
                artifact.stat().st_size != record["size_bytes"]
                or _sha256_file(artifact) != record["sha256"]
            ):
                raise contracts.ContractError(f"native {role} hash/size changed")
        source_manifest = contracts.load_json(runtime["runtime_manifest"]["path"])
        if (
            source_manifest.get("schema") != native_handoff.EXPECTED_RUNTIME_SCHEMA
            or source_manifest.get("tag") != runtime["runtime_tag"]
            or source_manifest.get("variant_id") != runtime["variant_id"]
            or source_manifest.get("automatic_checks", {}).get("overall") != "passed"
        ):
            raise contracts.ContractError("native runtime source manifest changed")
    return manifest


def _validate_normalized(
    runtime: Mapping[str, Any], tag: str, manifest_path: Path
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve()
    bundle_root = manifest_path.parent
    output = bundle_root / "runtime.glb"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise contracts.ContractError("normalized runtime manifest is missing")
    if output.is_symlink() or not output.is_file():
        raise contracts.ContractError("normalized runtime GLB is missing")
    manifest = contracts.load_json(manifest_path)
    normalization = manifest.get("normalization", {})
    checks = manifest.get("automatic_checks", {})
    if (
        manifest.get("schema") != "rocketbox_native_ue_runtime_v3"
        or manifest.get("tag") != tag
        or manifest.get("source_tag") != runtime["runtime_tag"]
        or manifest.get("variant_id") != runtime["variant_id"]
        or manifest.get("asset_id") != "rocketbox_male_adult_01"
        or manifest.get("usage_scope") != "research_candidate"
        or manifest.get("formal_registration_authorized") is not False
        or checks.get("overall") != "passed"
        or checks.get("walking_in_place") != "passed"
        or checks.get("embedded_images") != "unchanged"
        or normalization.get("normalized_joint_count") != 80
        or normalization.get("in_place_actions") != ["Walking"]
    ):
        raise contracts.ContractError("normalized runtime contract is invalid")
    output_record = manifest.get("runtime_glb", {})
    if (
        output_record.get("filename") != "runtime.glb"
        or output_record.get("sha256") != _sha256_file(output)
        or output_record.get("size_bytes") != output.stat().st_size
    ):
        raise contracts.ContractError("normalized runtime GLB hash/size changed")
    return {
        "variant_key": runtime["variant_key"],
        "variant_id": runtime["variant_id"],
        "runtime_tag": runtime["runtime_tag"],
        "ue_runtime_tag": tag,
        "sampled_attributes": runtime["sampled_attributes"],
        "consumer_request_count": runtime["consumer_request_count"],
        "status": "passed",
        "runtime_glb": _record(output),
        "normalization_manifest": _record(manifest_path),
        "source_runtime_glb": runtime["runtime_glb"],
        "actions": ["Standing_Idle", "Walking"],
        "front_axis": "negative_y",
        "actor_scale": 1.0,
        "normalization": normalization,
        "automatic_checks": checks,
    }


def normalize_handoffs(
    native_handoff_path: Path,
    output_root: Path,
    *,
    workers: int,
    resume: bool,
) -> Path:
    if not 1 <= workers <= 4:
        raise contracts.ContractError("workers must be in [1, 4]")
    native_manifest = load_native_handoff(native_handoff_path)
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

    def execute(runtime: Mapping[str, Any]) -> dict[str, Any]:
        tag = f"{runtime['runtime_tag']}_ue_v3"
        destination = UE_RUNTIME_ROOT / tag
        source_root = Path(runtime["runtime_glb"]["path"]).resolve().parent
        source_relative_root = source_root.relative_to(materials.SPEAR_ROOT).as_posix()
        contract = {
            "asset_id": "rocketbox_male_adult_01",
            "variant_id": runtime["variant_id"],
            "source_tag": runtime["runtime_tag"],
            "source_relative_root": source_relative_root,
            "source_manifest": "variant_manifest.json",
            "source_manifest_schema": native_handoff.EXPECTED_RUNTIME_SCHEMA,
            "output_relative_root": destination.relative_to(
                materials.SPEAR_ROOT
            ).as_posix(),
        }
        if destination.exists() or destination.is_symlink():
            if not resume:
                raise contracts.ContractError(
                    f"normalized runtime already exists; use --resume: {destination}"
                )
            result = _validate_normalized(
                runtime, tag, destination / "normalization_manifest.json"
            )
            result["execution"] = "reused_validated"
            return result
        try:
            manifest_path = ue_runtime.publish_bundle(tag, contract, destination)
        except ue_runtime.UeInPlaceRuntimeBundleError as error:
            raise contracts.ContractError(str(error)) from error
        result = _validate_normalized(runtime, tag, manifest_path)
        result["execution"] = "built"
        return result

    try:
        results = []
        runtimes = native_manifest["runtimes"]
        with ThreadPoolExecutor(max_workers=min(workers, len(runtimes))) as executor:
            futures = {executor.submit(execute, runtime): runtime for runtime in runtimes}
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda item: item["variant_key"])
        manifest: dict[str, Any] = {
            "schema": NORMALIZED_HANDOFF_SCHEMA,
            "status": "passed",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "native_handoff": _record(Path(native_handoff_path)),
            "native_handoff_sha256": native_manifest["handoff_sha256"],
            "workers": workers,
            "resume": resume,
            "runtime_count": len(results),
            "runtimes": results,
            "automatic_checks": {
                "all_native_runtimes_reauthenticated": True,
                "all_normalizations_passed": True,
                "all_actions_preserved": True,
                "all_embedded_materials_preserved": True,
                "all_walking_actions_in_place": True,
                "actor_scale_preserved": True,
                "baseline_written": False,
                "overall": "passed",
            },
        }
        manifest["handoff_sha256"] = _hash_without(manifest, "handoff_sha256")
        contracts.write_json_no_replace(
            staging / "ue_runtime_handoff_manifest.json", manifest
        )
        materials.native._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError(
                f"refusing to replace concurrently-created output: {output_root}"
            )
        os.rename(staging, output_root)
        return output_root / "ue_runtime_handoff_manifest.json"
    except Exception:
        materials.native._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-handoff", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest_path = normalize_handoffs(
            args.native_handoff,
            args.output_root,
            workers=args.workers,
            resume=args.resume,
        )
        manifest = contracts.load_json(manifest_path)
    except (contracts.ContractError, OSError) as error:
        print(f"CONTROLLED_ROCKETBOX_UE_RUNTIME_FAILED {error}", file=sys.stderr)
        return 2
    built = sum(item["execution"] == "built" for item in manifest["runtimes"])
    reused = sum(item["execution"] == "reused_validated" for item in manifest["runtimes"])
    print(
        "CONTROLLED_ROCKETBOX_UE_RUNTIME_OK "
        f"runtimes={manifest['runtime_count']} built={built} reused={reused} "
        f"output={manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
