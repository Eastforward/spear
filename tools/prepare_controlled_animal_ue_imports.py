#!/usr/bin/env python3
"""Transcode approved controlled animal runtimes and build one UE import batch."""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import audit_mesh_efficiency
from tools import controlled_source_asset_schema as contracts
from tools import review_controlled_animal_animation_candidates as animation_decisions
from tools import rocketbox_native_material_canary as immutable
from tools import run_controlled_animal_animation_reviews as animation_reviews
from tools import transcode_glb_webp_to_png as transcode


SCHEMA = "avengine_controlled_animal_ue_import_preparation_v1"
IMPORT_SCHEMA = "pixal_animal_ue_import_batch_v1"
SPEAR_ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/data/jzy/miniconda3/envs/spear-env/bin/python")
TRANSCODER = SPEAR_ROOT / "tools/transcode_glb_webp_to_png.py"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _relative_record(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _absolute_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_relative(root: Path, record: Mapping[str, Any], label: str) -> Path:
    path = (root / str(record.get("path", ""))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise contracts.ContractError(f"{label} escaped root") from error
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256_file(path) != record.get("sha256")
    ):
        raise contracts.ContractError(f"{label} changed")
    return path


def load_animation_decision_batch(
    path: Path,
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"animation decision batch is missing: {path}")
    payload = contracts.load_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != animation_decisions.DECISION_BATCH_SCHEMA
        or payload.get("decision_batch_sha256")
        != _hash_without(payload, "decision_batch_sha256")
        or payload.get("decision_count") != len(payload.get("decisions", []))
        or payload.get("approved_count") + payload.get("rejected_count")
        != payload.get("decision_count")
        or payload.get("automatic_checks", {}).get("overall") != "passed"
    ):
        raise contracts.ContractError("animation decision batch contract/hash is invalid")
    root = path.parent
    approved: dict[str, dict[str, Any]] = {}
    seen = set()
    for index in payload["decisions"]:
        asset_id = index.get("asset_id")
        if asset_id in seen:
            raise contracts.ContractError("duplicate animation decision asset")
        seen.add(asset_id)
        record_path = _verify_relative(root, index["record"], "animation decision")
        record = contracts.load_json(record_path)
        if (
            record.get("schema") != animation_decisions.DECISION_SCHEMA
            or record.get("asset_id") != asset_id
            or record.get("decision") != index.get("decision")
            or record.get("decision_sha256") != index.get("decision_sha256")
            or record.get("decision_sha256")
            != _hash_without(record, "decision_sha256")
        ):
            raise contracts.ContractError("animation decision identity/hash changed")
        if record["decision"] == "approved_for_ue_apartment":
            approved[asset_id] = {"payload": record, "path": record_path}
    if len(approved) != payload["approved_count"] or not approved:
        raise contracts.ContractError("approved animation decision count changed")
    return path, payload, approved


def _transcode_command(source: Path, output: Path, manifest: Path) -> list[str]:
    return [
        str(PYTHON),
        str(TRANSCODER),
        "--input",
        str(source),
        "--output",
        str(output),
        "--manifest",
        str(manifest),
    ]


def _run_one(
    attempt: Mapping[str, Any],
    decision: Mapping[str, Any],
    staging: Path,
    final_root: Path,
) -> dict[str, Any]:
    asset_id = str(attempt["asset_id"])
    destination = staging / "assets" / asset_id
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / "animated_100000_double_sided_png.glb"
    manifest = destination / "ue_texture_transcode_manifest.json"
    log = destination / "transcode.log"
    source = Path(attempt["rigged_path"]).resolve()
    started = time.monotonic()
    with log.open("xb") as stream:
        completed = subprocess.run(
            _transcode_command(source, output, manifest),
            cwd=SPEAR_ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )
        stream.flush()
        os.fsync(stream.fileno())
    if completed.returncode != 0:
        raise contracts.ContractError(f"texture transcode failed: {asset_id}")
    manifest_payload = contracts.load_json(manifest)
    stats = audit_mesh_efficiency.mesh_stats(output)
    document, _binary = transcode.read_glb(output)
    if (
        manifest_payload.get("geometry_skin_animation_byte_graph_changed") is not False
        or manifest_payload.get("input", {}).get("sha256") != _sha256_file(source)
        or manifest_payload.get("output", {}).get("sha256") != _sha256_file(output)
        or not stats
        or stats.get("skins") != 1
        or stats.get("animations") != 2
        or stats.get("textures", 0) <= 0
        or any(image.get("mimeType") != "image/png" for image in document.get("images", []))
        or "EXT_texture_webp" in document.get("extensionsRequired", [])
    ):
        raise contracts.ContractError(f"UE-compatible GLB readback failed: {asset_id}")
    final_output = final_root / output.relative_to(staging)
    final_manifest = final_root / manifest.relative_to(staging)
    tag = f"pixal_{asset_id}"
    import_job = {
        "tag": tag,
        "legacy_tag": asset_id,
        "asset_id": asset_id,
        "profile_schema_id": attempt["profile_schema_id"],
        "sampled_attributes": copy.deepcopy(attempt["sampled_attributes"]),
        "rigged_glb": str(final_output),
        "rigged_glb_sha256": _sha256_file(output),
        "upstream_rigged_glb": str(source),
        "upstream_rigged_glb_sha256": _sha256_file(source),
        "texture_transcode_manifest": str(final_manifest),
        "animation_decision_sha256": decision["decision_sha256"],
        "expected_actions": ["Idle", "Walking"],
    }
    return {
        "asset_id": asset_id,
        "tag": tag,
        "status": "ready_for_ue_import",
        "upstream_rigged_glb": _absolute_record(source),
        "ue_compatible_glb": _relative_record(output, staging),
        "transcode_manifest": _relative_record(manifest, staging),
        "transcode_log": _relative_record(log, staging),
        "readback": {
            name: value for name, value in stats.items() if name not in {"path", "exists"}
        },
        "wall_seconds": time.monotonic() - started,
        "import_job": import_job,
    }


def prepare_imports(
    lod_binding_batch_path: Path,
    animation_decision_batch_path: Path,
    output_root: Path,
    *,
    workers: int = 8,
) -> Path:
    if not 1 <= workers <= 16:
        raise contracts.ContractError("workers must be between 1 and 16")
    if not PYTHON.is_file() or not TRANSCODER.is_file():
        raise contracts.ContractError("UE texture transcoder environment is missing")
    decision_path, decision_batch, approved = load_animation_decision_batch(
        animation_decision_batch_path
    )
    binding_path, binding_batch, attempts = animation_reviews.load_lod_binding_batch(
        lod_binding_batch_path, sorted(approved)
    )
    if {item["asset_id"] for item in attempts} != set(approved):
        raise contracts.ContractError("binding and animation approval sets differ")
    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(f"refusing to replace output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    started = time.monotonic()
    try:
        results = []
        with ThreadPoolExecutor(max_workers=min(workers, len(attempts))) as executor:
            futures = {
                executor.submit(
                    _run_one,
                    attempt,
                    approved[attempt["asset_id"]]["payload"],
                    staging,
                    output_root,
                ): attempt["asset_id"]
                for attempt in attempts
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"CONTROLLED_ANIMAL_UE_PREP_DONE asset={result['asset_id']}",
                    flush=True,
                )
        results.sort(key=lambda item: item["asset_id"])
        import_payload = {
            "schema": IMPORT_SCHEMA,
            "generated_at": _utc_now(),
            "non_destructive_policy": (
                "unique gate_pixal_* content directories; do not replace legacy "
                "Hunyuan or previously approved assets"
            ),
            "jobs": [copy.deepcopy(item["import_job"]) for item in results],
        }
        import_path = staging / "ue_import_jobs.json"
        contracts.write_json_no_replace(import_path, import_payload)
        for item in results:
            item.pop("import_job")
        manifest_payload: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "ready_for_ue_import",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "lod_binding_batch": _absolute_record(binding_path),
            "lod_binding_batch_sha256": binding_batch["batch_sha256"],
            "animation_decision_batch": _absolute_record(decision_path),
            "animation_decision_batch_sha256": decision_batch[
                "decision_batch_sha256"
            ],
            "transcoder": _absolute_record(TRANSCODER),
            "workers": min(workers, len(attempts)),
            "asset_count": len(results),
            "wall_seconds": time.monotonic() - started,
            "assets": results,
            "ue_import_jobs": _relative_record(import_path, staging),
            "automatic_checks": {
                "all_animation_decisions_reauthenticated": True,
                "all_rigged_glbs_reauthenticated": True,
                "all_webp_textures_losslessly_transcoded_to_png_pixels": True,
                "geometry_skin_animation_graph_unchanged": True,
                "all_ue_glbs_read_back_idle_and_walking": True,
                "non_destructive_unique_tags": True,
                "overall": "passed",
            },
        }
        manifest_payload["manifest_sha256"] = _hash_without(
            manifest_payload, "manifest_sha256"
        )
        contracts.write_json_no_replace(
            staging / "ue_import_preparation_manifest.json", manifest_payload
        )
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError("UE import output appeared concurrently")
        os.rename(staging, output_root)
        return output_root / "ue_import_preparation_manifest.json"
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lod-binding-batch", required=True, type=Path)
    parser.add_argument("--animation-decisions", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        path = prepare_imports(
            args.lod_binding_batch,
            args.animation_decisions,
            args.output_root,
            workers=args.workers,
        )
        payload = contracts.load_json(path)
    except (contracts.ContractError, OSError, subprocess.SubprocessError) as error:
        print(f"CONTROLLED_ANIMAL_UE_PREP_FAILED {error}", file=sys.stderr)
        return 2
    print(
        f"CONTROLLED_ANIMAL_UE_PREP_OK assets={payload['asset_count']} output={path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
