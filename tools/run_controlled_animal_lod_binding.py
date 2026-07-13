#!/usr/bin/env python3
"""Build authenticated 100k LOD + Walk/Idle runtimes for controlled animals.

This stage consumes one or more immutable controlled ``source_asset_v2``
registries.  It never edits those registries or their Pixal GLBs.  Every job
is written under a new atomic batch root, uses the species-matched Quaternius
Cat/Dog rig, and is read back before the batch is published.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import audit_mesh_efficiency
from tools import audit_quadruped_i23d_geometry
from tools import controlled_source_asset_schema as contracts
from tools import register_controlled_animal_source_assets as source_registry
from tools import rocketbox_native_material_canary as immutable
from tools.spike_rlr import runtime_proxy_mesh


BATCH_SCHEMA = "avengine_controlled_animal_lod_binding_batch_v1"
SPEAR_ROOT = Path(__file__).resolve().parents[1]
AVENGINE_ROOT = SPEAR_ROOT.parents[1]
BLENDER = Path("/data/jzy/.local/bin/blender")
GNU_TIME = Path("/usr/bin/time")
LOD_SCRIPT = SPEAR_ROOT / "tools/blender_create_runtime_proxy_mesh.py"
BIND_SCRIPT = SPEAR_ROOT / "tools/blender_robust_swap_mesh_keep_rig.py"
LICENSE_SNAPSHOT = AVENGINE_ROOT / "assets/mesh_library/README.md"
LICENSE_SNAPSHOT_SHA256 = (
    "5887c71ec9a300997bee4445def8f4fb9014ea4e09b36522c1efb9b8eb3a5aef"
)
RIG_SPECS = {
    "cat": {
        "profile_id": "quadruped_cat_v1",
        "skeleton_family": "quaternius_cat",
        "path": AVENGINE_ROOT / "assets/mesh_library/quaternius_animalpack/Cat.glb",
        "sha256": "af2afb5e92c6d9daae98a918f8bd2bcb13ea4d7cfb880020d0d263e4d2f1277e",
    },
    "dog": {
        "profile_id": "quadruped_dog_v1",
        "skeleton_family": "quaternius_dog",
        "path": AVENGINE_ROOT / "assets/mesh_library/quaternius_animalpack/Dog.glb",
        "sha256": "bf9d2fdaf74a36be453edf4516a0b13b042cfce2d2614e0bf3ee24d40d553032",
    },
}
APPROVED_ACTIONS = ["Idle", "Walking"]


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


def _verify_file(path: Path, record: Mapping[str, Any], *, label: str) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record.get("size_bytes")
        or _sha256_file(path) != record.get("sha256")
    ):
        raise contracts.ContractError(f"{label} changed: {path}")


def _rig_spec(source_asset: Mapping[str, Any]) -> dict[str, Any]:
    taxonomy = source_asset.get("taxonomy", {})
    species = taxonomy.get("species")
    if species not in RIG_SPECS:
        raise contracts.ContractError(f"unsupported controlled animal species: {species}")
    spec = RIG_SPECS[species]
    rig = source_asset.get("rig", {})
    if (
        rig.get("profile_id") != spec["profile_id"]
        or rig.get("skeleton_family") != spec["skeleton_family"]
        or rig.get("front_axis") != "positive_x"
        or set(rig.get("actions", [])) != {"Walking", "Idle"}
    ):
        raise contracts.ContractError(
            f"source asset rig contract changed: {source_asset.get('asset_id')}"
        )
    path = Path(spec["path"]).resolve()
    if not path.is_file() or _sha256_file(path) != spec["sha256"]:
        raise contracts.ContractError(f"pinned {species} rig changed: {path}")
    return {**spec, "path": path, "species": species}


def _load_registry(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise contracts.ContractError(f"source registry is missing: {path}")
    payload = contracts.load_json(path)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != source_registry.REGISTRY_SCHEMA
        or payload.get("registry_sha256") != _hash_without(payload, "registry_sha256")
        or payload.get("automatic_checks", {}).get("overall") != "passed"
        or payload.get("source_asset_count") != len(payload.get("source_assets", []))
    ):
        raise contracts.ContractError(f"source registry contract/hash is invalid: {path}")

    jobs = []
    for index in payload["source_assets"]:
        relative = Path(index.get("source_asset", {}).get("path", ""))
        source_path = (path.parent / relative).resolve()
        try:
            source_path.relative_to(path.parent.resolve())
        except ValueError as error:
            raise contracts.ContractError("source_asset_v2 escaped registry root") from error
        _verify_file(source_path, index["source_asset"], label="source_asset_v2")
        source = contracts.load_json(source_path)
        asset_id = source.get("asset_id")
        if (
            source.get("schema") != contracts.SOURCE_ASSET_SCHEMA
            or source.get("asset_class") != "animal"
            or asset_id != index.get("asset_id")
            or source.get("profile_schema_id") != index.get("profile_schema_id")
            or source.get("request_sha256") != index.get("request_sha256")
            or source.get("sampled_attributes") != index.get("sampled_attributes")
            or source.get("qa", {}).get("static_mesh") != "passed"
            or source.get("qa", {}).get("binding") != "pending"
        ):
            raise contracts.ContractError(f"source_asset_v2 identity changed: {asset_id}")

        raw_record = source.get("artifacts", {}).get("pixal_raw_glb", {})
        if raw_record.get("root_id") != "spear_repo":
            raise contracts.ContractError(f"Pixal GLB root changed: {asset_id}")
        raw_path = (SPEAR_ROOT / raw_record.get("path", "")).resolve()
        try:
            raw_path.relative_to(SPEAR_ROOT.resolve())
        except ValueError as error:
            raise contracts.ContractError("Pixal GLB escaped SPEAR root") from error
        _verify_file(raw_path, raw_record, label="Pixal raw GLB")
        raw_stats = audit_mesh_efficiency.mesh_stats(raw_path)
        if (
            not raw_stats
            or not raw_stats.get("exists")
            or raw_stats.get("triangles", 0) <= 0
            or raw_stats.get("materials", 0) <= 0
            or raw_stats.get("textures", 0) <= 0
            or raw_stats.get("skins") != 0
            or raw_stats.get("animations") != 0
        ):
            raise contracts.ContractError(f"Pixal raw GLB readback failed: {asset_id}")
        jobs.append(
            {
                "asset_id": asset_id,
                "profile_schema_id": source["profile_schema_id"],
                "request_sha256": source["request_sha256"],
                "sampled_attributes": source["sampled_attributes"],
                "target_physical_profile": source["target_physical_profile"],
                "source_asset_path": source_path,
                "source_asset_sha256": index["source_asset"]["sha256"],
                "raw_path": raw_path,
                "raw_record": raw_record,
                "raw_stats": {
                    name: value
                    for name, value in raw_stats.items()
                    if name not in {"path", "exists"}
                },
                "rig": _rig_spec(source),
            }
        )
    return payload, jobs


def load_jobs(
    registry_paths: Sequence[Path], asset_ids: Sequence[str] = ()
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not registry_paths:
        raise contracts.ContractError("at least one source registry is required")
    registries = []
    jobs_by_id: dict[str, dict[str, Any]] = {}
    for path in registry_paths:
        resolved = Path(path).resolve()
        payload, jobs = _load_registry(resolved)
        registries.append(
            {
                "path": str(resolved),
                "sha256": _sha256_file(resolved),
                "registry_sha256": payload["registry_sha256"],
                "source_asset_count": payload["source_asset_count"],
            }
        )
        for job in jobs:
            if job["asset_id"] in jobs_by_id:
                raise contracts.ContractError(
                    f"duplicate asset across registries: {job['asset_id']}"
                )
            jobs_by_id[job["asset_id"]] = job
    selected = set(asset_ids)
    if selected:
        missing = selected - set(jobs_by_id)
        if missing:
            raise contracts.ContractError(f"requested assets are missing: {sorted(missing)}")
        jobs_by_id = {
            asset_id: job for asset_id, job in jobs_by_id.items() if asset_id in selected
        }
    if not jobs_by_id:
        raise contracts.ContractError("no controlled animal jobs selected")
    return registries, [jobs_by_id[key] for key in sorted(jobs_by_id)]


def build_commands(
    job: Mapping[str, Any], job_root: Path, *, target_faces: int
) -> tuple[list[str], list[str]]:
    lod = job_root / "runtime_lod/mesh_runtime_100000_double_sided.glb"
    metadata = job_root / "runtime_lod/mesh_runtime_100000_double_sided.json"
    rigged = job_root / "rigged/animated_100000_double_sided.glb"
    lod_command = [
        str(BLENDER),
        "-b",
        "--python",
        str(LOD_SCRIPT),
        "--",
        "--source",
        str(job["raw_path"]),
        "--output",
        str(lod),
        "--metadata",
        str(metadata),
        "--target-faces",
        str(target_faces),
        "--double-sided",
    ]
    bind_command = [
        str(BLENDER),
        "-b",
        "--python",
        str(BIND_SCRIPT),
        "--",
        "--rig-glb",
        str(job["rig"]["path"]),
        "--new-mesh",
        str(lod),
        "--output",
        str(rigged),
        "--flip-x",
        "--align-mode",
        "uniform",
        "--weight-mode",
        "region",
        "--segmentation-mode",
        "proximity",
        "--semantic-forward-axis",
        "positive-x",
        "--dampen-head-rotations",
        "0",
        "--dampen-tail-rotations",
        "0",
        "--dampen-foot-rotations",
        "1",
        "--remove-limb-bridges",
        "yes",
        "--delete-limb-bridge-faces",
        "no",
        "--export-action-policy",
        "walk-idle",
    ]
    return lod_command, bind_command


def _parse_time_metrics(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    cpu = re.search(r"Percent of CPU this job got: ([0-9.]+)%", text)
    return {
        "max_rss_kib": int(rss.group(1)) if rss else None,
        "cpu_percent": float(cpu.group(1)) if cpu else None,
    }


def _run_timed(command: Sequence[str], log_path: Path, timeout: int) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("xb") as log:
        result = subprocess.run(
            [str(GNU_TIME), "-v", *command],
            cwd=SPEAR_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        log.flush()
        os.fsync(log.fileno())
    return {
        "returncode": result.returncode,
        "wall_seconds": time.monotonic() - started,
        **_parse_time_metrics(log_path),
    }


def _glb_document(path: Path) -> dict[str, Any]:
    return audit_mesh_efficiency._load_glb_json(path)


def validate_lod_and_binding(
    raw_stats: Mapping[str, Any], lod_path: Path, rigged_path: Path, target_faces: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    lod_stats = audit_mesh_efficiency.mesh_stats(lod_path)
    rigged_stats = audit_mesh_efficiency.mesh_stats(rigged_path)
    expected_faces = min(int(raw_stats["triangles"]), int(target_faces))
    if (
        not lod_stats
        or not lod_stats.get("exists")
        or not 0.90 * expected_faces <= lod_stats.get("triangles", 0) <= expected_faces
        or lod_stats.get("materials", 0) <= 0
        or lod_stats.get("textures", 0) <= 0
        or lod_stats.get("skins") != 0
        or lod_stats.get("animations") != 0
    ):
        raise contracts.ContractError("runtime LOD GLB readback failed")
    lod_document = _glb_document(lod_path)
    if not lod_document.get("materials") or not all(
        material.get("doubleSided") is True for material in lod_document["materials"]
    ):
        raise contracts.ContractError("runtime LOD is not double-sided")

    if (
        not rigged_stats
        or not rigged_stats.get("exists")
        or not 0.90 * lod_stats["triangles"]
        <= rigged_stats.get("triangles", 0)
        <= lod_stats["triangles"]
        or rigged_stats.get("materials", 0) <= 0
        or rigged_stats.get("textures", 0) <= 0
        or rigged_stats.get("skins") != 1
        or rigged_stats.get("animations") != 2
    ):
        raise contracts.ContractError("rigged GLB readback failed")
    rigged_document = _glb_document(rigged_path)
    animation_names = [
        animation.get("name") for animation in rigged_document.get("animations", [])
    ]
    channel_paths = {
        channel.get("target", {}).get("path")
        for animation in rigged_document.get("animations", [])
        for channel in animation.get("channels", [])
    }
    if animation_names != APPROVED_ACTIONS or not channel_paths <= {
        "translation",
        "rotation",
    }:
        raise contracts.ContractError(
            f"rigged animation contract changed: {animation_names}/{channel_paths}"
        )
    if not rigged_document.get("materials") or not all(
        material.get("doubleSided") is True
        for material in rigged_document["materials"]
    ):
        raise contracts.ContractError("rigged runtime is not double-sided")
    clean_lod = {
        name: value for name, value in lod_stats.items() if name not in {"path", "exists"}
    }
    clean_rigged = {
        name: value
        for name, value in rigged_stats.items()
        if name not in {"path", "exists"}
    }
    clean_rigged["animation_names"] = animation_names
    clean_rigged["animation_channel_paths"] = sorted(channel_paths)
    return clean_lod, clean_rigged


def _rewrite_runtime_metadata(
    metadata_path: Path, public_runtime_path: Path, *, source_sha256: str
) -> None:
    payload = contracts.load_json(metadata_path)
    physical_runtime = Path(payload.get("runtime_mesh", ""))
    algorithm = payload.get("algorithm")
    if (
        algorithm not in runtime_proxy_mesh.SUPPORTED_RUNTIME_PROXY_ALGORITHMS
        or payload.get("source_mesh_sha256") != source_sha256
        or not physical_runtime.is_file()
        or payload.get("runtime_mesh_sha256") != _sha256_file(physical_runtime)
    ):
        raise contracts.ContractError("runtime LOD metadata contract changed")
    if algorithm == runtime_proxy_mesh.RUNTIME_PROXY_ALGORITHM:
        topology = payload.get("topology", {})
        source_after_weld = topology.get("source_after_position_weld", {})
        runtime_after_decimate = topology.get("runtime_after_decimate", {})
        if (
            not isinstance(topology.get("boundary_cracks_introduced"), int)
            or topology["boundary_cracks_introduced"] > 0
            or not topology.get("position_weld", {}).get("vertices_welded", 0) > 0
            or not isinstance(runtime_after_decimate.get("boundary_edges"), int)
            or runtime_after_decimate["boundary_edges"]
            > source_after_weld.get("boundary_edges", -1)
        ):
            raise contracts.ContractError(
                "welded runtime LOD introduced boundary cracks"
            )
    payload["runtime_mesh"] = str(public_runtime_path.resolve())
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _relative_artifact(path: Path, root: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": path.relative_to(root.resolve()).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _available_artifacts(paths: Mapping[str, Path], root: Path) -> dict[str, Any]:
    return {
        name: _relative_artifact(path, root)
        for name, path in paths.items()
        if path.is_file() and not path.is_symlink()
    }


def _enforce_prebind_geometry_audit(record: Mapping[str, Any]) -> None:
    decision = record.get("decision", {})
    status = decision.get("status")
    if status == "reject_before_lod_and_binding":
        raise contracts.ContractError(
            "Pixal source geometry rejected before LOD/binding: "
            + ", ".join(decision.get("rejection_reasons", []))
        )
    if status not in {
        "passed_automatic_geometry_measurements",
        "manual_source_geometry_review_required",
    }:
        raise contracts.ContractError("prebind geometry audit status is invalid")


def _run_job(
    job: Mapping[str, Any], staging: Path, public_root: Path, target_faces: int
) -> dict[str, Any]:
    asset_id = str(job["asset_id"])
    job_root = staging / "assets" / asset_id
    paths = {
        "prebind_geometry_audit": job_root / "prebind_geometry_audit.json",
        "lod_glb": job_root / "runtime_lod/mesh_runtime_100000_double_sided.glb",
        "lod_metadata": job_root
        / "runtime_lod/mesh_runtime_100000_double_sided.json",
        "lod_log": job_root / "runtime_lod/blender.log",
        "rigged_glb": job_root / "rigged/animated_100000_double_sided.glb",
        "binding_log": job_root / "rigged/rig.log",
    }
    lod_command, bind_command = build_commands(job, job_root, target_faces=target_faces)
    base = {
        "asset_id": asset_id,
        "profile_schema_id": job["profile_schema_id"],
        "request_sha256": job["request_sha256"],
        "sampled_attributes": job["sampled_attributes"],
        "target_physical_profile": job["target_physical_profile"],
        "source_asset": {
            "path": str(job["source_asset_path"]),
            "sha256": job["source_asset_sha256"],
        },
        "pixal_raw_glb": copy.deepcopy(job["raw_record"]),
        "source_rig": {
            "species": job["rig"]["species"],
            "profile_id": job["rig"]["profile_id"],
            "skeleton_family": job["rig"]["skeleton_family"],
            "root_id": "avengine_repo",
            "path": job["rig"]["path"].relative_to(AVENGINE_ROOT).as_posix(),
            "sha256": job["rig"]["sha256"],
        },
        "raw_mesh_readback": job["raw_stats"],
    }
    try:
        paths["prebind_geometry_audit"].parent.mkdir(parents=True, exist_ok=True)
        geometry_record = audit_quadruped_i23d_geometry.audit(
            Path(job["raw_path"]), asset_id
        )
        paths["prebind_geometry_audit"].write_text(
            json.dumps(
                {
                    "schema": audit_quadruped_i23d_geometry.SCHEMA,
                    "created_at": _utc_now(),
                    "purpose": (
                        "prebind_geometry_measurement_without_direction_inference"
                    ),
                    "record": geometry_record,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        base["prebind_geometry_audit"] = copy.deepcopy(geometry_record)
        _enforce_prebind_geometry_audit(geometry_record)
        lod_timing = _run_timed(lod_command, paths["lod_log"], timeout=900)
        if lod_timing["returncode"] != 0:
            return {
                **base,
                "status": "failed_lod",
                "timings": {"lod": lod_timing},
                "artifacts": _available_artifacts(paths, staging),
            }
        public_runtime = (
            public_root
            / "assets"
            / asset_id
            / "runtime_lod/mesh_runtime_100000_double_sided.glb"
        )
        _rewrite_runtime_metadata(
            paths["lod_metadata"],
            public_runtime,
            source_sha256=job["raw_record"]["sha256"],
        )
        bind_timing = _run_timed(bind_command, paths["binding_log"], timeout=1800)
        if bind_timing["returncode"] != 0:
            return {
                **base,
                "status": "failed_binding",
                "timings": {"lod": lod_timing, "binding": bind_timing},
                "artifacts": _available_artifacts(paths, staging),
            }
        lod_stats, rigged_stats = validate_lod_and_binding(
            job["raw_stats"], paths["lod_glb"], paths["rigged_glb"], target_faces
        )
        return {
            **base,
            "status": "passed_lod_binding_glb_readback",
            "runtime_lod_readback": lod_stats,
            "rigged_runtime_readback": rigged_stats,
            "timings": {"lod": lod_timing, "binding": bind_timing},
            "artifacts": _available_artifacts(paths, staging),
            "next_gate": "walking_idle_visual_and_contact_qa",
        }
    except (
        contracts.ContractError,
        OSError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        return {
            **base,
            "status": "failed_validation_or_execution",
            "error": str(error),
            "artifacts": _available_artifacts(paths, staging),
        }


def _tool_record(path: Path) -> dict[str, Any]:
    return {
        "root_id": "spear_repo",
        "path": path.relative_to(SPEAR_ROOT).as_posix(),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _pinned_provenance() -> dict[str, Any]:
    if (
        not LICENSE_SNAPSHOT.is_file()
        or _sha256_file(LICENSE_SNAPSHOT) != LICENSE_SNAPSHOT_SHA256
    ):
        raise contracts.ContractError("Quaternius license snapshot changed")
    rigs = {}
    for species, spec in RIG_SPECS.items():
        path = Path(spec["path"])
        if not path.is_file() or _sha256_file(path) != spec["sha256"]:
            raise contracts.ContractError(f"pinned {species} rig changed")
        rigs[species] = {
            "root_id": "avengine_repo",
            "path": path.relative_to(AVENGINE_ROOT).as_posix(),
            "sha256": spec["sha256"],
            "size_bytes": path.stat().st_size,
            "license": "CC0-1.0",
        }
    return {
        "license_snapshot": {
            "root_id": "avengine_repo",
            "path": LICENSE_SNAPSHOT.relative_to(AVENGINE_ROOT).as_posix(),
            "sha256": LICENSE_SNAPSHOT_SHA256,
            "size_bytes": LICENSE_SNAPSHOT.stat().st_size,
        },
        "source_rigs": rigs,
        "tools": {
            "prebind_geometry_audit": _tool_record(
                SPEAR_ROOT / "tools/audit_quadruped_i23d_geometry.py"
            ),
            "runtime_lod": _tool_record(LOD_SCRIPT),
            "binding": _tool_record(BIND_SCRIPT),
        },
    }


def run_batch(
    registry_paths: Sequence[Path],
    output_root: Path,
    *,
    workers: int = 8,
    target_faces: int = 100_000,
    asset_ids: Sequence[str] = (),
) -> Path:
    if not 1 <= workers <= 16:
        raise contracts.ContractError("workers must be between 1 and 16")
    if target_faces != 100_000:
        raise contracts.ContractError("controlled close LOD is pinned to 100000 faces")
    if not BLENDER.is_file() or not GNU_TIME.is_file():
        raise contracts.ContractError("pinned Blender or GNU time is missing")
    provenance = _pinned_provenance()
    registries, jobs = load_jobs(registry_paths, asset_ids)
    output_root = Path(output_root).absolute()
    if output_root.exists() or output_root.is_symlink():
        raise contracts.ContractError(f"refusing to replace output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.", suffix=".staging", dir=output_root.parent
        )
    )
    started_at = _utc_now()
    started = time.monotonic()
    try:
        attempts = []
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            futures = {
                executor.submit(
                    _run_job, job, staging, output_root, target_faces
                ): job["asset_id"]
                for job in jobs
            }
            for future in as_completed(futures):
                attempt = future.result()
                attempts.append(attempt)
                print(
                    "CONTROLLED_ANIMAL_LOD_BINDING_JOB_DONE "
                    f"asset={attempt['asset_id']} status={attempt['status']}",
                    flush=True,
                )
        attempts.sort(key=lambda item: item["asset_id"])
        passed = sum(
            item["status"] == "passed_lod_binding_glb_readback" for item in attempts
        )
        failed = len(attempts) - passed
        manifest: dict[str, Any] = {
            "schema": BATCH_SCHEMA,
            "status": "passed" if failed == 0 else "completed_with_failures",
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "started_at": started_at,
            "finished_at": _utc_now(),
            "wall_seconds": time.monotonic() - started,
            "source_registries": registries,
            "provenance": provenance,
            "parameters": {
                "target_faces": target_faces,
                "double_sided": True,
                "workers": min(workers, len(jobs)),
                "alignment": "uniform",
                "flip_x": True,
                "weight_mode": "region",
                "segmentation_mode": "proximity",
                "semantic_forward_axis": "positive-x",
                "prebind_geometry_audit": (
                    audit_quadruped_i23d_geometry.SCHEMA
                ),
                "remove_limb_bridges": True,
                "delete_limb_bridge_faces": False,
                "head_rotation_dampening": 0.0,
                "tail_rotation_dampening": 0.0,
                "foot_rotation_dampening": 1.0,
                "export_actions": APPROVED_ACTIONS,
                "ue_animation_channel_paths": ["translation", "rotation"],
            },
            "job_count": len(attempts),
            "passed_count": passed,
            "failed_count": failed,
            "attempts": attempts,
            "automatic_checks": {
                "all_source_registries_reauthenticated": True,
                "all_source_asset_v2_records_reauthenticated": True,
                "all_pixal_raw_glbs_reauthenticated": True,
                "all_successful_sources_passed_prebind_geometry_gate": passed > 0,
                "all_source_rigs_and_license_snapshot_pinned": True,
                "all_successful_lods_glb2_readable_and_double_sided": passed > 0,
                "all_successful_runtimes_have_one_skin": passed > 0,
                "all_successful_runtimes_have_only_idle_and_walking": passed > 0,
                "all_successful_runtimes_have_only_ue_safe_animation_channels": passed
                > 0,
                "visual_animation_and_foot_contact_qa_pending": True,
                "overall": "passed" if failed == 0 else "needs_failure_review",
            },
        }
        manifest["batch_sha256"] = _hash_without(manifest, "batch_sha256")
        contracts.write_json_no_replace(staging / "lod_binding_batch_manifest.json", manifest)
        immutable._seal_readonly_tree(staging)
        if output_root.exists() or output_root.is_symlink():
            raise contracts.ContractError("LOD/binding output appeared concurrently")
        os.rename(staging, output_root)
        return output_root / "lod_binding_batch_manifest.json"
    except Exception:
        immutable._remove_staging_tree(staging)
        raise


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--target-faces", type=int, default=100_000)
    parser.add_argument("--asset-id", action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    try:
        manifest_path = run_batch(
            args.registry,
            args.output_root,
            workers=args.workers,
            target_faces=args.target_faces,
            asset_ids=args.asset_id,
        )
        manifest = contracts.load_json(manifest_path)
    except (contracts.ContractError, OSError, subprocess.SubprocessError) as error:
        print(f"CONTROLLED_ANIMAL_LOD_BINDING_FAILED {error}", file=sys.stderr)
        return 2
    print(
        "CONTROLLED_ANIMAL_LOD_BINDING_OK "
        f"passed={manifest['passed_count']} failed={manifest['failed_count']} "
        f"output={manifest_path}"
    )
    return 0 if manifest["failed_count"] == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
