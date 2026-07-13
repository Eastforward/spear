#!/usr/bin/env python3

#
# Copyright (c) 2025 The SPEAR Development Team. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
# Copyright (c) 2022 Intel. Licensed under the MIT License <http://opensource.org/licenses/MIT>.
#

"""Clean a Hunyuan human mesh and bind it to an approved Rocketbox rig."""

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
SPIKE_RLR_DIR = TOOLS_DIR/"spike_rlr"
for import_dir in (SPEAR_ROOT, SPIKE_RLR_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import hy3d_human_candidate as hy3d_contract
from tools import blender_retarget_rocketbox_walk as retarget
from tools.human_part_transfer import (
    HumanRegion,
    collapse_finger_weights_to_palms,
    cross_limb_bridge_face_mask,
    human_ground_artifact_mask,
    target_regions_from_capsules,
    transfer_human_weights,
)


EXPECTED_ASSET_IDS = (
    "rocketbox_male_adult_01",
    "rocketbox_female_adult_01",
)
IDLE_PINS = {
    "rocketbox_male_adult_01": {
        "filename": "m_idle_neutral_01.max.fbx",
        "size_bytes": 2418544,
        "sha256": "818cc185af21390575f7fbfdeb3012ba2ce5969fbcb220ea725a2617b339a6e2",
        "git_blob_sha1": "a2d92c3326a9c503af677c9fa6082387f060d6c4",
    },
    "rocketbox_female_adult_01": {
        "filename": "f_idle_neutral_01.max.fbx",
        "size_bytes": 2959360,
        "sha256": "fd68b33ea9e290dc734ca8c3a71ef5842bb2dfe719853ff84f6336d06d39fdcb",
        "git_blob_sha1": "aecf1d0089ccfc0c381d5395294bb1c8fe0e63ae",
    },
}
HY3D_FILENAMES = {
    "reference": "reference.png",
    "reference_rembg": "reference_rembg.png",
    "shape": "shape.glb",
    "paint_obj": "hy3d_textured.obj",
    "diffuse": "hy3d_diffuse.jpg",
    "metallic": "hy3d_metallic.jpg",
    "roughness": "hy3d_roughness.jpg",
}
CONSUMED_HY3D_ROLES = (
    "reference",
    "paint_obj",
    "diffuse",
    "metallic",
    "roughness",
)
ASSET_SEEDS = {
    "rocketbox_male_adult_01": 4101,
    "rocketbox_female_adult_01": 7301,
}
CANONICAL_BASELINE_ROOT = Path(
    "/data/datasets/rocketbox/approved_baselines/rocketbox_neutral_walk_v1"
)
BASELINE_MANIFEST_SHA256 = (
    "b6e468e5f0c79d7ecec168e3c2460a7997a8d2916393da9add1ef2b6952fb922"
)
CANONICAL_IDLE_ROOT = Path(
    "/data/datasets/rocketbox/sample/Assets/Animations/all_animations_max_motextr_static"
)
CANONICAL_HY3D_ROOT = SPEAR_ROOT/"tmp"/"hy3d_rocketbox_spike_v1"
CANONICAL_REFERENCE_REVIEW_ROOT = SPEAR_ROOT/"tmp"/"human_reference_review"
CANONICAL_MODEL_ROOT = Path("/data/models/hunyuan3d-2.1/hunyuan3d-2.1")
WEIGHT_ROOT_HASH_MANIFEST = Path("/data/models/hunyuan3d-2.1/weights.sha256")
HY3D_SCHEMA_VERSION = "hy3d_human_candidate_v1"
HY3D_USAGE_SCOPE = "technical_spike_only"
READINESS_FILES = (
    "bind_manifest.json",
    "review_manifest.json",
    "hy3d_rocketbox_review.json",
)
TARGET_ROTATE_Z_DEG = 0.0
RAW_HY3D_AXIS_CONTRACTS = {
    "rocketbox_male_adult_01": {
        "source_up_axis": "Y",
        "source_front_axis": "Z",
        "import_forward_axis": "NEGATIVE_Z",
        "import_up_axis": "Y",
        "expected_basis_matrix": (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0),
        ),
    },
    "rocketbox_female_adult_01": {
        "source_up_axis": "Y",
        "source_front_axis": "Z",
        "import_forward_axis": "NEGATIVE_Z",
        "import_up_axis": "Y",
        "expected_basis_matrix": (
            (1.0, 0.0, 0.0),
            (0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0),
        ),
    },
}
PBR_NODE_CONTRACT = {
    "diffuse": {
        "node_name": "Hunyuan Diffuse",
        "image_name": "hy3d_diffuse",
        "principled_input": "Base Color",
        "colorspace": "sRGB",
    },
    "metallic": {
        "node_name": "Hunyuan Metallic",
        "image_name": "hy3d_metallic",
        "principled_input": "Metallic",
        "colorspace": "Non-Color",
    },
    "roughness": {
        "node_name": "Hunyuan Roughness",
        "image_name": "hy3d_roughness",
        "principled_input": "Roughness",
        "colorspace": "Non-Color",
    },
}
GROUND_MAX_CENTER_HEIGHT_RATIO = 0.12
GROUND_MAX_COMPONENT_HEIGHT_RATIO = 0.05
GROUND_MIN_HORIZONTAL_SPREAD_RATIO = 0.40
GROUND_MIN_VERTICES = 20
GROUND_COPLANAR_HEIGHT_RATIO = 0.05
RESIDUAL_ROBUST_QUANTILE = 0.02
RESIDUAL_ROBUST_MARGIN_RATIO = 0.5
TARGET_BONES = retarget.TARGET_BONES
WEIGHT_SUM_TOLERANCE = 1.0e-5
GLTF_MIN_INFLUENCE = 0.0001
MAX_DISTANCE_HEIGHT_RATIO = 0.20
OUTPUT_FILENAMES = (
    "cleaned.obj",
    "bound.blend",
    "bound_walk.glb",
    "bound_idle.glb",
    "bind_metrics.json",
    "bind_manifest.json",
)

assert len(TARGET_BONES) == 80
assert len(set(TARGET_BONES)) == 80


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--hy3d-dir", type=Path, required=True)
    parser.add_argument("--idle-motion-fbx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1_file(path):
    path = Path(path)
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_copy(source, destination):
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            with source.open("rb") as source_stream:
                shutil.copyfileobj(source_stream, stream, length=1024*1024)
            stream.flush()
            os.fsync(stream.fileno())
        if sha256_file(temporary) != sha256_file(source):
            raise RuntimeError(f"staged copy hash mismatch: {destination}")
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def invalidate_readiness(output_dir):
    for filename in READINESS_FILES:
        try:
            os.unlink(Path(output_dir)/filename)
        except FileNotFoundError:
            pass


def require_regular_file(path, description):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError(f"{description} must be a direct regular file: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{description} is empty: {path}")
    return path


def require_real_directory(path, description):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise ValueError(f"{description} must be a direct real directory: {path}")
    return path


def load_json_object(path, description):
    path = require_regular_file(path, description)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must contain a JSON object")
    return payload


def validate_file_record(path, record, size_key, description):
    path = require_regular_file(path, description)
    if not isinstance(record, dict):
        raise ValueError(f"{description} manifest record is missing")
    if record.get(size_key) != path.stat().st_size:
        raise ValueError(f"{description} size does not match its manifest")
    if record.get("sha256") != sha256_file(path):
        raise ValueError(f"{description} SHA-256 does not match its manifest")
    return path


def validate_baseline_inputs(args):
    baseline_root = require_real_directory(args.baseline_dir, "baseline directory")
    if baseline_root != CANONICAL_BASELINE_ROOT:
        raise ValueError(
            f"baseline directory must be the canonical sealed root: {CANONICAL_BASELINE_ROOT}"
        )
    manifest_path = require_regular_file(
        baseline_root/"baseline_manifest.json", "baseline manifest"
    )
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != BASELINE_MANIFEST_SHA256:
        raise ValueError("baseline manifest SHA-256 does not match the pinned seal")
    manifest = load_json_object(manifest_path, "baseline manifest")
    if manifest.get("schema_version") != "rocketbox_baseline_manifest_v1":
        raise ValueError("baseline manifest schema is not rocketbox_baseline_manifest_v1")
    if manifest.get("baseline_id") != "rocketbox_neutral_walk_v1":
        raise ValueError("baseline manifest baseline_id is not rocketbox_neutral_walk_v1")
    if manifest.get("motion") != "walk_neutral":
        raise ValueError("baseline manifest motion is not walk_neutral")
    assets = manifest.get("assets")
    if not isinstance(assets, dict) or args.asset_id not in assets:
        raise ValueError("baseline manifest does not contain the requested asset")
    files = assets[args.asset_id].get("files")
    if not isinstance(files, dict) or "retarget.blend" not in files:
        raise ValueError("baseline manifest is missing retarget.blend")
    asset_dir = require_real_directory(baseline_root/args.asset_id, "baseline asset directory")
    blend_path = validate_file_record(
        asset_dir/"retarget.blend",
        files["retarget.blend"],
        "size",
        "immutable baseline retarget.blend",
    )
    return {
        "baseline_root": baseline_root,
        "baseline_manifest": manifest,
        "baseline_manifest_path": manifest_path,
        "baseline_manifest_sha256": manifest_sha256,
        "baseline_manifest_size": manifest_path.stat().st_size,
        "baseline_blend_path": blend_path,
        "baseline_blend_sha256": sha256_file(blend_path),
        "baseline_blend_size": blend_path.stat().st_size,
    }


def validate_hy3d_provenance_payload(
    manifest,
    asset_id,
    job,
    runtime,
    current_weight_sha256,
    verified_weight_sha256,
):
    if manifest.get("schema_version") != HY3D_SCHEMA_VERSION:
        raise ValueError(f"Hunyuan manifest schema is not {HY3D_SCHEMA_VERSION}")
    if manifest.get("asset_id") != asset_id:
        raise ValueError("Hunyuan manifest asset_id does not match --asset-id")
    if manifest.get("candidate_sha256") != job.get("candidate_sha256"):
        raise ValueError("Hunyuan manifest candidate_sha256 is not the approved candidate")
    approval_fields = (
        "candidate_manifest_sha256",
        "source_sha256",
        "source_approval_sha256",
        "reference_review_sha256",
    )
    for field in approval_fields:
        if manifest.get(field) != job.get(field):
            raise ValueError(f"Hunyuan manifest {field} does not match current approval")
    expected_runtime = {
        "git_head": job.get("hunyuan_runtime_git_head"),
        "fingerprint": job.get("hunyuan_runtime_fingerprint"),
        "file_count": job.get("hunyuan_runtime_file_count"),
    }
    if runtime != expected_runtime:
        raise ValueError("current Hunyuan runtime provenance differs from approved generation")
    if manifest.get("hunyuan_code_revision") != runtime.get("git_head"):
        raise ValueError("Hunyuan code revision differs from current runtime git head")
    if manifest.get("hunyuan_runtime_git_head") != runtime.get("git_head"):
        raise ValueError("Hunyuan runtime git head differs from current provenance")
    if manifest.get("hunyuan_runtime_fingerprint") != runtime.get("fingerprint"):
        raise ValueError("Hunyuan runtime fingerprint differs from current provenance")
    if manifest.get("hunyuan_runtime_file_count") != runtime.get("file_count"):
        raise ValueError("Hunyuan runtime file count differs from current provenance")
    if job.get("model_root") != CANONICAL_MODEL_ROOT:
        raise ValueError("approved generation job does not use the canonical model root")
    if job.get("weight_root_hash_manifest") != WEIGHT_ROOT_HASH_MANIFEST:
        raise ValueError("approved generation job does not use the canonical weight manifest")
    if manifest.get("weight_root") != str(CANONICAL_MODEL_ROOT):
        raise ValueError("Hunyuan manifest weight_root is not canonical")
    if manifest.get("weight_root_hash_manifest") != str(WEIGHT_ROOT_HASH_MANIFEST):
        raise ValueError("Hunyuan manifest weight manifest path is not canonical")
    weight_sha256 = manifest.get("weight_manifest_sha256")
    if weight_sha256 != current_weight_sha256 or weight_sha256 != verified_weight_sha256:
        raise ValueError("Hunyuan weight manifest hash is not current and fully verified")
    if manifest.get("seed") != ASSET_SEEDS[asset_id]:
        raise ValueError("Hunyuan manifest seed is not canonical")
    if manifest.get("steps") != 50:
        raise ValueError("Hunyuan manifest steps must be 50")
    if manifest.get("guidance_scale") != 5.0:
        raise ValueError("Hunyuan manifest guidance_scale must be 5")
    if manifest.get("usage_scope") != HY3D_USAGE_SCOPE:
        raise ValueError(f"Hunyuan manifest usage_scope must be {HY3D_USAGE_SCOPE}")
    if (
        job.get("seed") != ASSET_SEEDS[asset_id]
        or job.get("steps") != 50
        or job.get("guidance_scale") != 5.0
    ):
        raise ValueError("approved generation job parameters are not canonical")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(HY3D_FILENAMES):
        raise ValueError("Hunyuan manifest outputs do not match the canonical allowlist")
    for role, filename in HY3D_FILENAMES.items():
        record = outputs.get(role)
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size_bytes"}:
            raise ValueError(f"Hunyuan {role} descriptor must be exact")
        if record.get("path") != filename:
            raise ValueError(f"Hunyuan {role} must use canonical filename {filename}")
        if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
            raise ValueError(f"Hunyuan {role} SHA-256 descriptor is invalid")
        if not isinstance(record.get("size_bytes"), int) or record["size_bytes"] <= 0:
            raise ValueError(f"Hunyuan {role} size descriptor is invalid")
    if outputs["reference"]["sha256"] != job.get("candidate_sha256"):
        raise ValueError("Hunyuan reference output is not the approved candidate")
    return True


def validate_hy3d_inputs(args, contract=hy3d_contract):
    hy3d_dir = require_real_directory(args.hy3d_dir, "Hunyuan asset directory")
    expected_dir = CANONICAL_HY3D_ROOT/args.asset_id
    if hy3d_dir != expected_dir:
        raise ValueError(f"Hunyuan directory must be canonical: {expected_dir}")
    jobs = contract.assert_generation_ready(CANONICAL_REFERENCE_REVIEW_ROOT)
    if set(jobs) != set(EXPECTED_ASSET_IDS):
        raise ValueError("Task 1 generation gate did not return the exact approved pair")
    job = jobs[args.asset_id]
    runtime = contract.current_hunyuan_runtime_provenance()
    verified_weight_sha256 = contract.verify_canonical_weights()
    current_weight_sha256 = contract.current_weight_manifest_sha256()
    manifest_path = require_regular_file(hy3d_dir/"hy3d_manifest.json", "Hunyuan manifest")
    manifest = load_json_object(manifest_path, "Hunyuan manifest")
    validate_hy3d_provenance_payload(
        manifest,
        args.asset_id,
        job,
        runtime,
        current_weight_sha256,
        verified_weight_sha256,
    )
    outputs = manifest.get("outputs")
    paths = {}
    for role, filename in HY3D_FILENAMES.items():
        record = outputs.get(role)
        paths[role] = validate_file_record(
            hy3d_dir/filename,
            record,
            "size_bytes",
            f"Hunyuan {role}",
        )
    return {
        "hy3d_dir": hy3d_dir,
        "hy3d_manifest": manifest,
        "hy3d_manifest_path": manifest_path,
        "hy3d_manifest_sha256": sha256_file(manifest_path),
        "hy3d_manifest_size": manifest_path.stat().st_size,
        "hy3d_paths": paths,
        "hy3d_output_sha256": {
            role: sha256_file(path) for role, path in paths.items()
        },
        "hy3d_output_sizes": {
            role: path.stat().st_size for role, path in paths.items()
        },
        "approval_job": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in job.items()
            if key != "review_root"
        },
        "runtime_provenance": runtime,
        "verified_weight_manifest_sha256": verified_weight_sha256,
        "current_weight_manifest_sha256": current_weight_sha256,
    }


def validate_idle_motion(args):
    pin = IDLE_PINS.get(args.asset_id)
    if pin is None:
        raise ValueError(f"unexpected asset_id: {args.asset_id}")
    path = Path(args.idle_motion_fbx).absolute()
    expected_path = CANONICAL_IDLE_ROOT/pin["filename"]
    if path != expected_path:
        raise ValueError(f"idle motion must use the exact official path: {expected_path}")
    path = require_regular_file(path, "Rocketbox idle motion FBX")
    if path.stat().st_size != pin["size_bytes"]:
        raise ValueError("idle motion byte size does not match the official pin")
    sha256 = sha256_file(path)
    if sha256 != pin["sha256"]:
        raise ValueError("idle motion SHA-256 does not match the official pin")
    git_blob_sha1 = git_blob_sha1_file(path)
    if git_blob_sha1 != pin["git_blob_sha1"]:
        raise ValueError("idle motion Git blob SHA-1 does not match the official pin")
    return {
        "idle_motion_fbx_path": path,
        "idle_motion_fbx_sha256": sha256,
        "idle_motion_fbx_size": path.stat().st_size,
        "idle_motion_fbx_git_blob_sha1": git_blob_sha1,
    }


def copy_authenticated_file(
    source,
    destination,
    expected_sha256,
    expected_size,
    on_source_opened=None,
):
    source = Path(source).absolute()
    destination = Path(destination).absolute()
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    source_directory_fd = os.open(source.parent, directory_flags)
    destination_directory_fd = os.open(destination.parent, directory_flags)
    source_fd = None
    destination_fd = None
    published = False
    try:
        source_fd = os.open(
            source.name,
            os.O_RDONLY | os.O_NOFOLLOW,
            dir_fd=source_directory_fd,
        )
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("authenticated snapshot source is not a regular file")
        if before.st_size != expected_size:
            raise RuntimeError("authenticated snapshot source size changed")
        if on_source_opened is not None:
            on_source_opened(source)
        destination_fd = os.open(
            destination.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=destination_directory_fd,
        )
        digest = hashlib.sha256()
        copied_size = 0
        while True:
            chunk = os.read(source_fd, 1024*1024)
            if not chunk:
                break
            digest.update(chunk)
            copied_size += len(chunk)
            pending = memoryview(chunk)
            while pending:
                written = os.write(destination_fd, pending)
                if written <= 0:
                    raise RuntimeError("authenticated snapshot write made no progress")
                pending = pending[written:]
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        before_state = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        after_state = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        actual_sha256 = digest.hexdigest()
        if before_state != after_state:
            raise RuntimeError("authenticated snapshot source changed during copy")
        if copied_size != expected_size or actual_sha256 != expected_sha256:
            raise RuntimeError("authenticated snapshot bytes failed hash or size validation")
        destination_stat = os.fstat(destination_fd)
        if not stat.S_ISREG(destination_stat.st_mode):
            raise RuntimeError("authenticated snapshot destination is not regular")
        os.fchmod(destination_fd, 0o600)
        published = True
        return {
            "filename": destination.name,
            "sha256": actual_sha256,
            "size_bytes": copied_size,
        }
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if source_fd is not None:
            os.close(source_fd)
        if not published:
            try:
                os.unlink(destination.name, dir_fd=destination_directory_fd)
            except FileNotFoundError:
                pass
        os.close(destination_directory_fd)
        os.close(source_directory_fd)


def authenticated_snapshot_sources(baseline, hy3d, idle):
    sources = {
        "baseline_manifest": {
            "path": baseline["baseline_manifest_path"],
            "filename": "baseline_manifest.json",
            "sha256": baseline["baseline_manifest_sha256"],
            "size_bytes": baseline["baseline_manifest_size"],
        },
        "baseline_blend": {
            "path": baseline["baseline_blend_path"],
            "filename": "retarget.blend",
            "sha256": baseline["baseline_blend_sha256"],
            "size_bytes": baseline["baseline_blend_size"],
        },
        "hy3d_manifest": {
            "path": hy3d["hy3d_manifest_path"],
            "filename": "hy3d_manifest.json",
            "sha256": hy3d["hy3d_manifest_sha256"],
            "size_bytes": hy3d["hy3d_manifest_size"],
        },
        "idle_motion_fbx": {
            "path": idle["idle_motion_fbx_path"],
            "filename": idle["idle_motion_fbx_path"].name,
            "sha256": idle["idle_motion_fbx_sha256"],
            "size_bytes": idle["idle_motion_fbx_size"],
        },
    }
    for role in CONSUMED_HY3D_ROLES:
        sources[f"hy3d_{role}"] = {
            "path": hy3d["hy3d_paths"][role],
            "filename": HY3D_FILENAMES[role],
            "sha256": hy3d["hy3d_output_sha256"][role],
            "size_bytes": hy3d["hy3d_output_sizes"][role],
        }
    return sources


def stage_input_snapshot(output_dir, baseline, hy3d, idle):
    output_dir = Path(output_dir).absolute()
    snapshot_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.bind-inputs.",
            dir=output_dir.parent,
        )
    )
    os.chmod(snapshot_root, 0o700)
    paths = {}
    records = {}
    try:
        for label, source in authenticated_snapshot_sources(
            baseline, hy3d, idle
        ).items():
            destination = snapshot_root/source["filename"]
            records[label] = copy_authenticated_file(
                source["path"],
                destination,
                source["sha256"],
                source["size_bytes"],
            )
            paths[label] = destination
        records["idle_motion_fbx"]["git_blob_sha1"] = idle[
            "idle_motion_fbx_git_blob_sha1"
        ]
        return {"root": snapshot_root, "paths": paths, "records": records}
    except BaseException:
        shutil.rmtree(snapshot_root, ignore_errors=True)
        raise


def cleanup_input_snapshot(snapshot):
    if snapshot is None:
        return
    snapshot_root = Path(snapshot["root"])
    if snapshot_root.exists():
        shutil.rmtree(snapshot_root)


def capture_source_hashes(baseline, hy3d, idle):
    return {
        "baseline_manifest_sha256": baseline["baseline_manifest_sha256"],
        "baseline_blend_sha256": baseline["baseline_blend_sha256"],
        "hy3d_manifest_sha256": hy3d["hy3d_manifest_sha256"],
        "hy3d_output_sha256": dict(hy3d["hy3d_output_sha256"]),
        "idle_motion_fbx_sha256": idle["idle_motion_fbx_sha256"],
    }


def verify_source_hashes_current(baseline, hy3d, idle, captured):
    current = {
        "baseline_manifest_current_sha256": sha256_file(
            baseline["baseline_manifest_path"]
        ),
        "baseline_blend_current_sha256": sha256_file(baseline["baseline_blend_path"]),
        "hy3d_manifest_current_sha256": sha256_file(hy3d["hy3d_manifest_path"]),
        "hy3d_output_current_sha256": {
            role: sha256_file(path) for role, path in hy3d["hy3d_paths"].items()
        },
        "idle_motion_fbx_current_sha256": sha256_file(idle["idle_motion_fbx_path"]),
    }
    comparisons = (
        (captured["baseline_manifest_sha256"], current["baseline_manifest_current_sha256"]),
        (captured["baseline_blend_sha256"], current["baseline_blend_current_sha256"]),
        (captured["hy3d_manifest_sha256"], current["hy3d_manifest_current_sha256"]),
        (captured["hy3d_output_sha256"], current["hy3d_output_current_sha256"]),
        (captured["idle_motion_fbx_sha256"], current["idle_motion_fbx_current_sha256"]),
    )
    if any(first != second for first, second in comparisons):
        raise RuntimeError("source inputs changed during binding")
    return current


def identify_target_objects():
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if len(armatures) != 1 or len(meshes) != 1:
        raise RuntimeError(
            f"baseline must contain one armature and one mesh: {len(armatures)}/{len(meshes)}"
        )
    armature = armatures[0]
    source_mesh = meshes[0]
    modifiers = [modifier for modifier in source_mesh.modifiers if modifier.type == "ARMATURE"]
    if len(modifiers) != 1 or modifiers[0].object != armature:
        raise RuntimeError("baseline mesh is not bound to its Rocketbox armature")
    return armature, source_mesh


def mesh_arrays(mesh):
    vertices = np.array([tuple(vertex.co) for vertex in mesh.data.vertices], dtype=np.float64)
    if any(len(polygon.vertices) != 3 for polygon in mesh.data.polygons):
        raise RuntimeError(f"mesh must be triangular before transfer: {mesh.name}")
    faces = np.array([tuple(polygon.vertices) for polygon in mesh.data.polygons], dtype=np.int64)
    return vertices, faces


def source_vertex_weights(mesh):
    group_names = [group.name for group in mesh.vertex_groups]
    if tuple(group_names) != TARGET_BONES:
        raise RuntimeError("source mesh groups do not match the exact Rocketbox bone contract")
    weights = np.zeros((len(mesh.data.vertices), len(group_names)), dtype=np.float64)
    for vertex in mesh.data.vertices:
        for influence in vertex.groups:
            weights[vertex.index, influence.group] = float(influence.weight)
    sums = weights.sum(axis=1)
    if np.any(sums <= 0.0):
        raise RuntimeError("source Rocketbox mesh contains zero-weight vertices")
    weights /= sums[:, None]
    return weights, group_names


def nondegenerate_transfer_faces(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    first_edges = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    second_edges = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    area_vectors = np.cross(first_edges, second_edges)
    usable_mask = np.einsum("ij,ij->i", area_vectors, area_vectors) > 0.0
    usable = faces[usable_mask]
    if len(usable) == 0:
        raise RuntimeError("Rocketbox source has no nondegenerate transfer faces")
    return usable, {
        "original_face_count": int(len(faces)),
        "usable_face_count": int(len(usable)),
        "zero_area_face_count": int((~usable_mask).sum()),
    }


def capsule_record(armature, region, first_name, last_name=None):
    first = armature.data.bones[first_name]
    last = armature.data.bones[last_name or first_name]
    start = np.array(tuple(first.head_local), dtype=np.float64)
    end = np.array(tuple(last.tail_local), dtype=np.float64)
    radius = max(float((last.tail_local - first.head_local).length)*0.18, 1.0e-4)
    return (region, start, end, radius)


def rocketbox_capsules(armature):
    return (
        capsule_record(armature, HumanRegion.TORSO, "Bip01 Pelvis", "Bip01 Spine2"),
        capsule_record(armature, HumanRegion.HEAD, "Bip01 Neck", "Bip01 Head"),
        capsule_record(armature, HumanRegion.LEFT_UPPER_ARM, "Bip01 L UpperArm"),
        capsule_record(armature, HumanRegion.LEFT_FOREARM, "Bip01 L Forearm"),
        capsule_record(armature, HumanRegion.LEFT_PALM, "Bip01 L Hand"),
        capsule_record(armature, HumanRegion.RIGHT_UPPER_ARM, "Bip01 R UpperArm"),
        capsule_record(armature, HumanRegion.RIGHT_FOREARM, "Bip01 R Forearm"),
        capsule_record(armature, HumanRegion.RIGHT_PALM, "Bip01 R Hand"),
        capsule_record(armature, HumanRegion.LEFT_THIGH, "Bip01 L Thigh"),
        capsule_record(armature, HumanRegion.LEFT_CALF, "Bip01 L Calf"),
        capsule_record(armature, HumanRegion.LEFT_FOOT, "Bip01 L Foot", "Bip01 L Toe0"),
        capsule_record(armature, HumanRegion.RIGHT_THIGH, "Bip01 R Thigh"),
        capsule_record(armature, HumanRegion.RIGHT_CALF, "Bip01 R Calf"),
        capsule_record(armature, HumanRegion.RIGHT_FOOT, "Bip01 R Foot", "Bip01 R Toe0"),
    )


def capture_rocketbox_source(armature, source_mesh):
    armature.data.pose_position = "REST"
    bpy.context.scene.frame_set(int(round(armature.animation_data.action.frame_range[0])))
    bpy.context.view_layer.update()
    if len(armature.data.bones) != 80:
        raise RuntimeError(f"expected 80 Rocketbox bones, got {len(armature.data.bones)}")
    if tuple(bone.name for bone in armature.data.bones) != TARGET_BONES:
        raise RuntimeError("baseline Rocketbox bone order or names changed")
    if len(source_mesh.data.uv_layers) < 1 or len(source_mesh.material_slots) < 1:
        raise RuntimeError("baseline Rocketbox source mesh is missing UV or material data")
    local_to_armature = armature.matrix_world.inverted() @ source_mesh.matrix_world
    vertices = np.array(
        [tuple(local_to_armature @ vertex.co) for vertex in source_mesh.data.vertices],
        dtype=np.float64,
    )
    world_vertices = np.array(
        [
            tuple(source_mesh.matrix_world @ vertex.co)
            for vertex in source_mesh.data.vertices
        ],
        dtype=np.float64,
    )
    _, original_faces = mesh_arrays(source_mesh)
    faces, transfer_face_metrics = nondegenerate_transfer_faces(
        vertices, original_faces
    )
    weights, group_names = source_vertex_weights(source_mesh)
    source_bbox = np.vstack((vertices.min(axis=0), vertices.max(axis=0)))
    pelvis_height = float(armature.data.bones["Bip01 Pelvis"].head_local.z)
    return {
        "vertices": vertices,
        "faces": faces,
        "weights": weights,
        "group_names": group_names,
        "capsules": rocketbox_capsules(armature),
        "bbox": source_bbox,
        "floor_z_m": float(world_vertices[:, 2].min()),
        "pelvis_height": pelvis_height,
        "mesh_name": source_mesh.name,
        "vertex_count": len(vertices),
        "face_count": len(original_faces),
        "transfer_face_metrics": transfer_face_metrics,
        "uv_layer_count": len(source_mesh.data.uv_layers),
        "material_slot_count": len(source_mesh.material_slots),
        "bone_count": len(armature.data.bones),
    }


def select_only(obj):
    for candidate in bpy.context.scene.objects:
        candidate.select_set(False)
    obj.hide_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def triangulate_mesh(obj):
    select_only(obj)
    modifier = obj.modifiers.new(name="Transfer Triangulation", type="TRIANGULATE")
    result = bpy.ops.object.modifier_apply(modifier=modifier.name)
    if "FINISHED" not in result:
        raise RuntimeError("could not triangulate imported Hunyuan mesh")


def validate_import_basis_matrix(matrix, contract):
    actual = np.asarray(matrix, dtype=np.float64)
    expected = np.asarray(contract["expected_basis_matrix"], dtype=np.float64)
    if actual.shape != (3, 3) or not np.allclose(actual, expected, atol=1.0e-7):
        raise RuntimeError("Blender OBJ importer did not apply the reviewed basis")
    canonical_front = actual @ np.array((0.0, 0.0, 1.0), dtype=np.float64)
    canonical_up = actual @ np.array((0.0, 1.0, 0.0), dtype=np.float64)
    determinant = float(np.linalg.det(actual))
    if not np.allclose(canonical_front, (0.0, -1.0, 0.0), atol=1.0e-7):
        raise RuntimeError("reviewed basis does not map source front to canonical -Y")
    if not np.allclose(canonical_up, (0.0, 0.0, 1.0), atol=1.0e-7):
        raise RuntimeError("reviewed basis does not map source up to canonical +Z")
    if not np.isclose(determinant, 1.0, atol=1.0e-7):
        raise RuntimeError("reviewed basis must be right-handed")
    return {
        "applied_basis_matrix": actual.tolist(),
        "canonical_front_vector": canonical_front.tolist(),
        "canonical_up_vector": canonical_up.tolist(),
        "basis_determinant": determinant,
    }


def import_hy3d_obj(path, asset_id):
    contract = RAW_HY3D_AXIS_CONTRACTS.get(asset_id)
    if contract is None:
        raise ValueError(f"missing reviewed raw-axis contract for {asset_id}")
    before = set(bpy.data.objects)
    result = bpy.ops.wm.obj_import(
        filepath=str(path),
        forward_axis=contract["import_forward_axis"],
        up_axis=contract["import_up_axis"],
    )
    if "FINISHED" not in result:
        raise RuntimeError("could not import hy3d_textured.obj")
    imported = [obj for obj in bpy.data.objects if obj not in before]
    meshes = [obj for obj in imported if obj.type == "MESH"]
    if len(meshes) != 1 or len(imported) != 1:
        raise RuntimeError("hy3d_textured.obj must import as exactly one mesh object")
    target = meshes[0]
    target.name = "Hunyuan_Rocketbox_Body"
    triangulate_mesh(target)
    if len(target.data.uv_layers) < 1 or len(target.material_slots) < 1:
        raise RuntimeError("hy3d_textured.obj must preserve UV and material data")
    raw_vertices = np.array(
        [tuple(vertex.co) for vertex in target.data.vertices], dtype=np.float64
    )
    raw_extents = np.ptp(raw_vertices, axis=0)
    basis_metrics = validate_import_basis_matrix(
        np.array(target.matrix_world.to_3x3(), dtype=np.float64), contract
    )
    return target, {
        **contract,
        **basis_metrics,
        "raw_extents": raw_extents.tolist(),
        "target_rotate_z_deg": TARGET_ROTATE_Z_DEG,
    }


def canonical_import_mesh_arrays(target):
    vertices = np.array(
        [tuple(target.matrix_world @ vertex.co) for vertex in target.data.vertices],
        dtype=np.float64,
    )
    if any(len(polygon.vertices) != 3 for polygon in target.data.polygons):
        raise RuntimeError("import-space Hunyuan mesh must be triangular")
    faces = np.array(
        [tuple(polygon.vertices) for polygon in target.data.polygons], dtype=np.int64
    )
    return vertices, faces


def move_target_to_armature_space(target, armature):
    to_armature = armature.matrix_world.inverted() @ target.matrix_world
    for vertex in target.data.vertices:
        vertex.co = to_armature @ vertex.co
    target.matrix_world = armature.matrix_world.copy()
    target.data.update()
    bpy.context.view_layer.update()


def connected_vertex_components(faces, vertex_count):
    parent = np.arange(int(vertex_count), dtype=np.int64)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    for face in np.asarray(faces, dtype=np.int64):
        if len(face) < 2:
            continue
        first = int(face[0])
        for index in face[1:]:
            first_root = find(first)
            other_root = find(int(index))
            if first_root != other_root:
                parent[other_root] = first_root
    components = {}
    for index in range(int(vertex_count)):
        components.setdefault(find(index), []).append(index)
    return [np.asarray(component, dtype=np.int64) for component in components.values()]


def ground_artifact_cleanup_mask(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0:
        raise RuntimeError("Hunyuan import contains no vertices")
    initial_mask = human_ground_artifact_mask(
        vertices=vertices,
        faces=faces,
        max_center_height_ratio=GROUND_MAX_CENTER_HEIGHT_RATIO,
        max_component_height_ratio=GROUND_MAX_COMPONENT_HEIGHT_RATIO,
        min_horizontal_spread_ratio=GROUND_MIN_HORIZONTAL_SPREAD_RATIO,
        min_vertices=GROUND_MIN_VERTICES,
    )
    remove_mask = initial_mask.copy()
    ground_plane_z = None
    if initial_mask.any():
        height = max(float(np.ptp(vertices[:, 2])), 1.0e-12)
        ground_plane_z = float(np.median(vertices[initial_mask, 2]))
        for component in connected_vertex_components(faces, len(vertices)):
            if initial_mask[component].any():
                continue
            component_vertices = vertices[component]
            component_min = component_vertices.min(axis=0)
            component_max = component_vertices.max(axis=0)
            component_extent = component_max - component_min
            component_center_z = float((component_min[2] + component_max[2])*0.5)
            if (
                component_extent[2] <= GROUND_COPLANAR_HEIGHT_RATIO*height
                and abs(component_center_z - ground_plane_z)
                <= GROUND_COPLANAR_HEIGHT_RATIO*height
            ):
                remove_mask[component] = True
    if remove_mask.all():
        raise RuntimeError("ground cleanup would remove the complete Hunyuan mesh")
    cleaned = vertices[~remove_mask]
    return remove_mask, {
        "initial_ground_vertices": int(initial_mask.sum()),
        "coplanar_expanded_vertices": int(remove_mask.sum() - initial_mask.sum()),
        "removed_vertices": int(remove_mask.sum()),
        "retained_vertices": int(len(cleaned)),
        "ground_plane_z": ground_plane_z,
        "raw_bbox_extent": np.ptp(vertices, axis=0).tolist(),
        "cleaned_bbox_extent": np.ptp(cleaned, axis=0).tolist(),
    }


def validate_residual_human_components(vertices, faces):
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise RuntimeError("ground cleanup produced an empty Hunyuan mesh")
    flat_mask = human_ground_artifact_mask(
        vertices=vertices,
        faces=faces,
        max_center_height_ratio=GROUND_MAX_CENTER_HEIGHT_RATIO,
        max_component_height_ratio=GROUND_MAX_COMPONENT_HEIGHT_RATIO,
        min_horizontal_spread_ratio=GROUND_MIN_HORIZONTAL_SPREAD_RATIO,
        min_vertices=GROUND_MIN_VERTICES,
    )
    if flat_mask.any():
        raise RuntimeError(
            f"cleanup left a large flat component: {int(flat_mask.sum())} vertices"
        )
    lower = np.quantile(vertices, RESIDUAL_ROBUST_QUANTILE, axis=0)
    upper = np.quantile(vertices, 1.0 - RESIDUAL_ROBUST_QUANTILE, axis=0)
    robust_extent = upper - lower
    robust_scale = float(robust_extent.max())
    if robust_scale <= 0.0:
        raise RuntimeError("cleaned Hunyuan mesh has no robust spatial extent")
    margin = robust_scale*RESIDUAL_ROBUST_MARGIN_RATIO
    allowed_min = lower - margin
    allowed_max = upper + margin
    far_components = []
    components = connected_vertex_components(faces, len(vertices))
    for component in components:
        component_vertices = vertices[component]
        component_min = component_vertices.min(axis=0)
        component_max = component_vertices.max(axis=0)
        if np.any(component_max < allowed_min) or np.any(component_min > allowed_max):
            far_components.append(int(len(component)))
    if far_components:
        raise RuntimeError(
            "cleanup left a far component: vertex counts="
            + ",".join(map(str, sorted(far_components, reverse=True)))
        )
    return {
        "component_count": len(components),
        "far_component_count": 0,
        "large_flat_vertex_count": 0,
        "robust_bbox_min": lower.tolist(),
        "robust_bbox_max": upper.tolist(),
        "robust_margin": margin,
    }


def cleanup_import_ground_artifacts(target):
    before_vertices, before_faces = canonical_import_mesh_arrays(target)
    before_uv_layers = len(target.data.uv_layers)
    before_material_slots = len(target.material_slots)
    remove_mask, metrics = ground_artifact_cleanup_mask(
        before_vertices, before_faces
    )
    removed_vertices = delete_masked_vertices(target, remove_mask)
    after_vertices, after_faces = canonical_import_mesh_arrays(target)
    if removed_vertices != metrics["removed_vertices"]:
        raise RuntimeError("ground cleanup removed an unexpected vertex count")
    if len(target.data.uv_layers) != before_uv_layers:
        raise RuntimeError("ground cleanup changed Hunyuan UV layer count")
    if len(target.material_slots) != before_material_slots:
        raise RuntimeError("ground cleanup changed Hunyuan material slot count")
    metrics["before"] = component_metrics(before_vertices, before_faces)
    metrics["after"] = component_metrics(after_vertices, after_faces)
    metrics["residual_validation"] = validate_residual_human_components(
        after_vertices, after_faces
    )
    return metrics


def component_metrics(vertices, faces):
    parent = list(range(len(vertices)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first, second):
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for face in faces:
        for index in range(1, len(face)):
            union(int(face[0]), int(face[index]))
    counts = {}
    for index in range(len(vertices)):
        root = find(index)
        counts[root] = counts.get(root, 0) + 1
    sizes = sorted(counts.values(), reverse=True)
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "components": len(sizes),
        "component_vertex_counts": sizes,
        "bbox_min": vertices.min(axis=0).tolist() if len(vertices) else [],
        "bbox_max": vertices.max(axis=0).tolist() if len(vertices) else [],
    }


def uniform_bbox_alignment(vertices, source_bbox, floor_z_m=None):
    vertices = np.asarray(vertices, dtype=np.float64)
    source_bbox = np.asarray(source_bbox, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError("target vertices must be a non-empty Nx3 array")
    target_min = vertices.min(axis=0)
    target_max = vertices.max(axis=0)
    target_extent = target_max - target_min
    source_extent = source_bbox[1] - source_bbox[0]
    if floor_z_m is None:
        floor_z_m = float(source_bbox[0, 2])
    if target_extent[2] <= 0.0 or source_extent[2] <= 0.0:
        raise ValueError("source and target bounding boxes must have positive height")
    scale = source_extent[2] / target_extent[2]
    scaled = vertices*scale
    scaled_min = scaled.min(axis=0)
    scaled_max = scaled.max(axis=0)
    source_center_xy = (source_bbox[0, :2] + source_bbox[1, :2])*0.5
    target_center_xy = (scaled_min[:2] + scaled_max[:2])*0.5
    translation = np.array(
        (
            source_center_xy[0] - target_center_xy[0],
            source_center_xy[1] - target_center_xy[1],
            source_bbox[0, 2] - scaled_min[2],
        ),
        dtype=np.float64,
    )
    aligned = scaled + translation
    return aligned, {
        "target_rotate_z_deg": TARGET_ROTATE_Z_DEG,
        "uniform_scale": float(scale),
        "floor_z_m": float(floor_z_m),
        "translation": translation.tolist(),
        "source_bbox": source_bbox.tolist(),
        "aligned_bbox": [aligned.min(axis=0).tolist(), aligned.max(axis=0).tolist()],
        "xy_center_error": (
            ((aligned.min(axis=0)[:2] + aligned.max(axis=0)[:2])*0.5) - source_center_xy
        ).tolist(),
        "floor_z_error": float(aligned[:, 2].min() - source_bbox[0, 2]),
    }


def set_vertex_positions(mesh, vertices):
    if len(mesh.data.vertices) != len(vertices):
        raise RuntimeError("aligned vertex count does not match Blender mesh")
    for index, value in enumerate(vertices):
        mesh.data.vertices[index].co = tuple(map(float, value))
    mesh.data.update()
    bpy.context.view_layer.update()


def align_target_mesh(target, source_bbox, floor_z_m):
    vertices, _ = mesh_arrays(target)
    aligned, metrics = uniform_bbox_alignment(vertices, source_bbox, floor_z_m)
    set_vertex_positions(target, aligned)
    return metrics


def delete_masked_vertices(target, mask):
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(target.data.vertices):
        raise RuntimeError("vertex cleanup mask length does not match target mesh")
    if not mask.any():
        return 0
    if mask.all():
        raise RuntimeError("vertex cleanup mask would remove the complete target mesh")
    bm = bmesh.new()
    try:
        bm.from_mesh(target.data)
        bm.verts.ensure_lookup_table()
        removed = [bm.verts[index] for index in np.flatnonzero(mask)]
        bmesh.ops.delete(bm, geom=removed, context="VERTS")
        bm.to_mesh(target.data)
    finally:
        bm.free()
    target.data.update()
    return int(mask.sum())


def delete_masked_faces(target, mask):
    mask = np.asarray(mask, dtype=bool)
    if len(mask) != len(target.data.polygons):
        raise RuntimeError("face cleanup mask length does not match target mesh")
    if not mask.any():
        return 0
    if mask.all():
        raise RuntimeError("face cleanup mask would remove the complete target mesh")
    bm = bmesh.new()
    try:
        bm.from_mesh(target.data)
        bm.faces.ensure_lookup_table()
        removed = [bm.faces[index] for index in np.flatnonzero(mask)]
        bmesh.ops.delete(bm, geom=removed, context="FACES")
        loose_edges = [edge for edge in bm.edges if not edge.link_faces]
        if loose_edges:
            bmesh.ops.delete(bm, geom=loose_edges, context="EDGES")
        loose_vertices = [vertex for vertex in bm.verts if not vertex.link_faces]
        if loose_vertices:
            bmesh.ops.delete(bm, geom=loose_vertices, context="VERTS")
        bm.to_mesh(target.data)
    finally:
        bm.free()
    target.data.update()
    return int(mask.sum())


def cleanup_target_geometry(target, source):
    vertices, faces = mesh_arrays(target)
    before = component_metrics(vertices, faces)
    before_uv_layers = len(target.data.uv_layers)
    before_material_slots = len(target.material_slots)

    ground_mask = human_ground_artifact_mask(vertices=vertices, faces=faces)
    ground_removed_vertices = delete_masked_vertices(target, ground_mask)
    first_alignment = align_target_mesh(
        target, source["bbox"], source["floor_z_m"]
    )

    vertices, faces = mesh_arrays(target)
    regions = target_regions_from_capsules(vertices, source["capsules"])
    protected_ground_mask = human_ground_artifact_mask(
        vertices=vertices,
        faces=faces,
        vertex_regions=regions,
    )
    protected_ground_removed_vertices = delete_masked_vertices(
        target, protected_ground_mask
    )
    second_alignment = align_target_mesh(
        target, source["bbox"], source["floor_z_m"]
    )

    vertices, faces = mesh_arrays(target)
    regions = target_regions_from_capsules(vertices, source["capsules"])
    bridge_mask = cross_limb_bridge_face_mask(
        vertices,
        faces,
        regions,
        pelvis_height=source["pelvis_height"],
    )
    bridge_removed_faces = delete_masked_faces(target, bridge_mask)
    final_alignment = align_target_mesh(
        target, source["bbox"], source["floor_z_m"]
    )

    vertices, faces = mesh_arrays(target)
    final_regions = target_regions_from_capsules(vertices, source["capsules"])
    after = component_metrics(vertices, faces)
    if len(target.data.uv_layers) != before_uv_layers:
        raise RuntimeError("geometric cleanup changed Hunyuan UV layer count")
    if len(target.material_slots) != before_material_slots:
        raise RuntimeError("geometric cleanup changed Hunyuan material slot count")
    if after["vertices"] <= 0 or after["faces"] <= 0:
        raise RuntimeError("geometric cleanup produced an empty Hunyuan mesh")
    return final_regions, {
        "before": before,
        "after": after,
        "removed_vertices": before["vertices"] - after["vertices"],
        "removed_faces": before["faces"] - after["faces"],
        "ground_mask_vertices": int(ground_mask.sum()),
        "ground_removed_vertices": ground_removed_vertices,
        "protected_ground_mask_vertices": int(protected_ground_mask.sum()),
        "protected_ground_removed_vertices": protected_ground_removed_vertices,
        "bridge_mask_faces": int(bridge_mask.sum()),
        "bridge_removed_faces": bridge_removed_faces,
        "first_alignment": first_alignment,
        "second_alignment": second_alignment,
        "final_alignment": final_alignment,
        "uv_layer_count": len(target.data.uv_layers),
        "material_slot_count": len(target.material_slots),
    }


def load_texture(path, non_color):
    image = bpy.data.images.load(str(path), check_existing=False)
    image.name = path.stem
    if non_color:
        image.colorspace_settings.name = "Non-Color"
    else:
        image.colorspace_settings.name = "sRGB"
    return image


def pack_pbr_images(material):
    packed = {}
    for role, contract in PBR_NODE_CONTRACT.items():
        node = material.node_tree.nodes.get(contract["node_name"])
        if node is None or node.type != "TEX_IMAGE" or node.image is None:
            raise RuntimeError(f"PBR {role} image texture node is missing before pack")
        image = node.image
        image.pack()
        if image.packed_file is None or image.packed_file.size <= 0:
            raise RuntimeError(f"PBR {role} image did not pack into bound.blend")
        if image.size[0] <= 0 or image.size[1] <= 0:
            raise RuntimeError(f"PBR {role} image has invalid pixel dimensions")
        packed[role] = {
            "image_name": image.name,
            "packed_size_bytes": int(image.packed_file.size),
            "pixel_size": [int(image.size[0]), int(image.size[1])],
            "colorspace": image.colorspace_settings.name,
        }
    return packed


def assign_hunyuan_pbr_material(target, paths):
    if len(target.data.uv_layers) < 1:
        raise RuntimeError("Hunyuan target has no UV layer for PBR binding")
    if len(target.data.materials):
        material = target.data.materials[0]
    else:
        material = bpy.data.materials.new(name="Hunyuan_PBR")
        target.data.materials.append(material)
    material.name = "Hunyuan_PBR"
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    diffuse = nodes.new("ShaderNodeTexImage")
    metallic = nodes.new("ShaderNodeTexImage")
    roughness = nodes.new("ShaderNodeTexImage")
    diffuse.name = "Hunyuan Diffuse"
    metallic.name = "Hunyuan Metallic"
    roughness.name = "Hunyuan Roughness"
    diffuse.image = load_texture(paths["diffuse"], False)
    metallic.image = load_texture(paths["metallic"], True)
    roughness.image = load_texture(paths["roughness"], True)
    material.node_tree.links.new(diffuse.outputs["Color"], principled.inputs["Base Color"])
    material.node_tree.links.new(metallic.outputs["Color"], principled.inputs["Metallic"])
    material.node_tree.links.new(roughness.outputs["Color"], principled.inputs["Roughness"])
    material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    target.data.materials.clear()
    target.data.materials.append(material)
    for polygon in target.data.polygons:
        polygon.material_index = 0
    if len(target.material_slots) != 1:
        raise RuntimeError("Hunyuan PBR target must have one used material slot")
    packed_images = pack_pbr_images(material)
    return {
        "material_name": material.name,
        "material_slot_count": len(target.material_slots),
        "uv_layer_count": len(target.data.uv_layers),
        "diffuse_image": diffuse.image.name,
        "metallic_image": metallic.image.name,
        "roughness_image": roughness.image.name,
        "packed_images": packed_images,
    }


def validate_packed_pbr_material(target):
    if len(target.data.materials) != 1 or target.data.materials[0] is None:
        raise RuntimeError("reopened bound.blend must contain one PBR material")
    material = target.data.materials[0]
    if not material.use_nodes or material.node_tree is None:
        raise RuntimeError("reopened bound.blend PBR material has no node tree")
    principled_nodes = [
        node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"
    ]
    if len(principled_nodes) != 1:
        raise RuntimeError("reopened PBR material must contain one Principled node")
    principled = principled_nodes[0]
    validation = {}
    for role, contract in PBR_NODE_CONTRACT.items():
        node = material.node_tree.nodes.get(contract["node_name"])
        if node is None or node.type != "TEX_IMAGE" or node.image is None:
            raise RuntimeError(f"reopened PBR {role} image texture node is missing")
        image = node.image
        if image.name != contract["image_name"]:
            raise RuntimeError(f"reopened PBR {role} image name is not canonical")
        if image.packed_file is None or image.packed_file.size <= 0:
            raise RuntimeError(f"reopened PBR {role} image has no packed bytes")
        if image.size[0] <= 0 or image.size[1] <= 0:
            raise RuntimeError(f"reopened PBR {role} image has invalid pixel dimensions")
        if not image.has_data:
            try:
                _ = image.pixels[0]
            except (IndexError, RuntimeError) as error:
                raise RuntimeError(f"reopened PBR {role} pixels cannot load") from error
        if not image.has_data:
            raise RuntimeError(f"reopened PBR {role} image has no decoded pixel data")
        if image.colorspace_settings.name != contract["colorspace"]:
            raise RuntimeError(f"reopened PBR {role} colorspace is incorrect")
        expected_input = principled.inputs[contract["principled_input"]]
        links = [
            link
            for link in material.node_tree.links
            if link.from_node == node
            and link.from_socket == node.outputs["Color"]
            and link.to_node == principled
            and link.to_socket == expected_input
        ]
        if len(links) != 1:
            raise RuntimeError(f"reopened PBR {role} Principled link is incomplete")
        validation[role] = {
            "image_name": image.name,
            "packed_size_bytes": int(image.packed_file.size),
            "pixel_size": [int(image.size[0]), int(image.size[1])],
            "has_data": bool(image.has_data),
            "colorspace": image.colorspace_settings.name,
            "principled_input": contract["principled_input"],
            "link_count": len(links),
        }
    return {
        "material_name": material.name,
        "packed_images": validation,
        "complete": True,
    }


def export_cleaned_obj(target, path):
    path = Path(path)
    mtl_path = path.with_suffix(".mtl")
    try:
        mtl_path.unlink()
    except FileNotFoundError:
        pass
    select_only(target)
    result = bpy.ops.wm.obj_export(
        filepath=str(path),
        export_selected_objects=True,
        export_animation=False,
        apply_modifiers=False,
        export_uv=True,
        export_normals=True,
        export_materials=False,
        export_pbr_extensions=False,
        export_triangulated_mesh=True,
        forward_axis="NEGATIVE_Y",
        up_axis="Z",
    )
    if "FINISHED" not in result or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"cleaned OBJ export failed: {path}")
    if mtl_path.exists():
        raise RuntimeError("geometry-only cleaned OBJ unexpectedly wrote an MTL sidecar")
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith(("mtllib ", "usemtl ")):
                raise RuntimeError("geometry-only cleaned OBJ contains material references")
    return {
        "role": "geometry_only",
        "materials": False,
        "uv": True,
        "normals": True,
    }


def validate_bound_weights(target):
    group_names = tuple(group.name for group in target.vertex_groups)
    if group_names != TARGET_BONES:
        raise RuntimeError("target vertex groups are not the exact Rocketbox bone list")
    finger_indices = {
        index for index, name in enumerate(group_names) if "Finger" in name
    }
    maximum_influences = 0
    maximum_sum_error = 0.0
    for vertex in target.data.vertices:
        influences = [item for item in vertex.groups if item.weight > 0.0]
        influence_count = len(influences)
        weight_sum = sum(float(item.weight) for item in influences)
        if influence_count == 0:
            raise RuntimeError(f"zero-weight target vertex: {vertex.index}")
        if influence_count > 4:
            raise RuntimeError(
                f"target vertex {vertex.index} has {influence_count} influences, expected at most four"
            )
        if abs(weight_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
            raise RuntimeError(
                f"target vertex {vertex.index} has non-normalized weight sum {weight_sum}"
            )
        if any(item.group in finger_indices for item in influences):
            raise RuntimeError("finger vertex groups must be empty after palm collapse")
        maximum_influences = max(maximum_influences, influence_count)
        maximum_sum_error = max(maximum_sum_error, abs(weight_sum - 1.0))
    return {
        "vertex_count": len(target.data.vertices),
        "vertex_group_count": len(target.vertex_groups),
        "maximum_influences": maximum_influences,
        "maximum_weight_sum_error": maximum_sum_error,
        "finger_groups_empty": True,
    }


def prune_gltf_influences(weights):
    out = np.asarray(weights, dtype=np.float64).copy()
    out[out <= GLTF_MIN_INFLUENCE] = 0.0
    totals = out.sum(axis=1)
    if np.any(totals <= 0.0):
        raise RuntimeError("glTF influence pruning produced a zero-weight vertex")
    out /= totals[:, None]
    return out


def transfer_with_distance_contract(source, target):
    source_vertices = np.asarray(source["vertices"], dtype=np.float64)
    source_height = float(np.ptp(source_vertices[:, 2]))
    if source_height <= 0.0:
        raise RuntimeError("Rocketbox REST source height must be positive")
    max_distance = source_height*MAX_DISTANCE_HEIGHT_RATIO
    weights, stats = transfer_human_weights(
        source,
        target,
        max_distance=max_distance,
        top_k=4,
        require_complete=True,
    )
    stats["distance_contract"] = {
        "source_height": source_height,
        "max_distance_height_ratio": MAX_DISTANCE_HEIGHT_RATIO,
        "max_distance": max_distance,
        "require_complete": True,
    }
    return weights, stats


def bind_target_mesh(target, armature, source, target_regions):
    vertices, faces = mesh_arrays(target)
    weights, transfer_stats = transfer_with_distance_contract(
        source,
        {
            "vertices": vertices,
            "faces": faces,
            "regions": target_regions,
        },
    )
    if transfer_stats["unmatched"] > 0:
        raise RuntimeError(
            "part-aware transfer left unmatched target vertices: "
            f"{transfer_stats['unmatched']}"
        )
    weights = collapse_finger_weights_to_palms(weights, TARGET_BONES)
    weights = prune_gltf_influences(weights)
    target.vertex_groups.clear()
    groups = [target.vertex_groups.new(name=name) for name in TARGET_BONES]
    for vertex_index, row in enumerate(weights):
        for group_index in np.flatnonzero(row > 0.0):
            groups[int(group_index)].add(
                [vertex_index], float(row[int(group_index)]), "REPLACE"
            )
    for modifier in list(target.modifiers):
        target.modifiers.remove(modifier)
    modifier = target.modifiers.new(name="Rocketbox Armature", type="ARMATURE")
    modifier.object = armature
    modifier.use_vertex_groups = True
    target.parent = armature
    target.matrix_parent_inverse = Matrix.Identity(4)
    target.matrix_world = armature.matrix_world.copy()
    modifiers = [item for item in target.modifiers if item.type == "ARMATURE"]
    if len(modifiers) != 1 or modifiers[0].object != armature:
        raise RuntimeError("target must have exactly one Rocketbox armature modifier")
    validation = validate_bound_weights(target)
    transfer_stats["region_counts"] = dict(transfer_stats["region_counts"])
    return {
        "transfer": transfer_stats,
        "weights": validation,
        "armature_modifier_count": len(modifiers),
    }


def remove_original_body(source_mesh):
    mesh_data = source_mesh.data
    bpy.data.objects.remove(source_mesh, do_unlink=True)
    if mesh_data.users == 0:
        bpy.data.meshes.remove(mesh_data)


def action_frame_range(action):
    return retarget.integer_frame_range(action)


def capture_target_base_transform(armature):
    if armature.animation_data is None or armature.animation_data.action is None:
        raise RuntimeError("approved baseline walk action is missing before idle bake")
    frame_start, _ = action_frame_range(armature.animation_data.action)
    bpy.context.scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    return {
        "frame": frame_start,
        "location": armature.matrix_world.translation.copy(),
        "rotation_quaternion": armature.matrix_world.to_quaternion().normalized(),
        "scale": armature.scale.copy(),
    }


def restore_target_base_transform(armature, target_base):
    armature.location = target_base["location"]
    armature.rotation_mode = "QUATERNION"
    armature.rotation_quaternion = target_base["rotation_quaternion"]
    armature.scale = target_base["scale"]
    bpy.context.scene.frame_set(target_base["frame"])
    bpy.context.view_layer.update()


def bake_idle_action(armature, asset_id, idle_motion_path):
    target_base = capture_target_base_transform(armature)
    source = retarget.import_source_motion(idle_motion_path)
    try:
        motion_action = getattr(source, "action")
        frame_start, frame_end = action_frame_range(motion_action)
        source_only_bones, unmapped_target_bones, hierarchy_mismatches = (
            retarget.validate_mapping(source.armature, armature)
        )
        cached_frames, helper_basis = retarget.cache_source_frames(
            source, frame_start, frame_end
        )
        parent_first_bones = retarget.parent_first_names(armature)
        armature.animation_data_create()
        idle_action = bpy.data.actions.new(
            name=f"{asset_id}_idle_neutral_01_retarget"
        )
        armature.animation_data.action = idle_action
        restore_target_base_transform(armature, target_base)
        (
            sampled_positions,
            maximum_pose_error,
            maximum_rotation_error,
            target_base_scale,
            baked_root_locations,
        ) = retarget.bake_target_action(
            armature,
            cached_frames,
            helper_basis,
            parent_first_bones,
            idle_action,
        )
        action_validation = retarget.validate_action_ownership(armature, idle_action)
    finally:
        helper_names = retarget.remove_source_import(source)
    idle_action.use_fake_user = True
    return idle_action, {
        "frame_start": frame_start,
        "frame_end": frame_end,
        "source_only_bones": source_only_bones,
        "unmapped_target_bones": unmapped_target_bones,
        "hierarchy_mismatches": hierarchy_mismatches,
        "helper_names_removed": helper_names,
        "maximum_pose_error": maximum_pose_error,
        "maximum_body_rotation_error_rad": maximum_rotation_error,
        "target_base_scale": target_base_scale,
        "approved_target_base_frame": target_base["frame"],
        "approved_target_base_location_m": retarget.vector_list(
            target_base["location"]
        ),
        "baked_root_frame_count": len(baked_root_locations),
        "sampled_frame_count": len(sampled_positions),
        "action_validation": action_validation,
    }


def validate_two_actions(walk_action, idle_action):
    if len(bpy.data.actions) != 2:
        raise RuntimeError(
            f"bound blend must contain exactly two actions, found {len(bpy.data.actions)}"
        )
    if set(bpy.data.actions) != {walk_action, idle_action}:
        raise RuntimeError("bound blend action set differs from approved walk plus baked idle")
    if walk_action == idle_action or walk_action.name == idle_action.name:
        raise RuntimeError("walk and idle must be different target actions")
    if not walk_action.fcurves:
        raise RuntimeError("approved baseline walk action is empty")
    if not idle_action.fcurves:
        raise RuntimeError("baked idle action is empty")
    return {
        "action_count": len(bpy.data.actions),
        "walk_action": walk_action.name,
        "idle_action": idle_action.name,
    }


def select_target_only(armature, target):
    for obj in bpy.context.scene.objects:
        obj.select_set(False)
    armature.hide_set(False)
    target.hide_set(False)
    armature.select_set(True)
    target.select_set(True)
    bpy.context.view_layer.objects.active = armature
    if set(bpy.context.selected_objects) != {armature, target}:
        raise RuntimeError("target-only selection failed")


def validate_target_only_scene(armature, target):
    if set(bpy.context.scene.objects) != {armature, target}:
        raise RuntimeError(
            "final scene must contain only the Rocketbox armature and Hunyuan target body"
        )
    if len(armature.data.bones) != 80:
        raise RuntimeError("final Rocketbox armature does not contain 80 bones")


def save_bound_blend(armature, target, path):
    validate_target_only_scene(armature, target)
    select_target_only(armature, target)
    bpy.context.preferences.filepaths.save_version = 0
    result = bpy.ops.wm.save_as_mainfile(filepath=str(path))
    if "FINISHED" not in result or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"bound blend save failed: {path}")


def sample_action_positions(armature, action):
    armature.data.pose_position = "POSE"
    armature.animation_data.action = action
    frame_start, frame_end = action_frame_range(action)
    positions = {}
    for frame in retarget.sample_frames(frame_start, frame_end):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        positions[frame] = {
            name: retarget.joint_head_world(armature, name).copy()
            for name in TARGET_BONES
        }
    return frame_start, frame_end, positions


def isolate_action_for_export(armature, action):
    armature.animation_data.action = action
    for other in list(bpy.data.actions):
        if other != action:
            bpy.data.actions.remove(other, do_unlink=True)
    if set(bpy.data.actions) != {action}:
        raise RuntimeError("single-action export isolation failed")


def export_single_action_glb(armature, target, action, path):
    if set(bpy.data.actions) != {action}:
        raise RuntimeError("single-action GLB export requires one isolated action")
    armature.data.pose_position = "POSE"
    armature.animation_data.action = action
    frame_start, frame_end = action_frame_range(action)
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end
    bpy.context.scene.frame_set(frame_start)
    bpy.context.view_layer.update()
    select_target_only(armature, target)
    result = bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        use_selection=True,
        export_animations=True,
        export_animation_mode="ACTIVE_ACTIONS",
        export_force_sampling=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
    )
    if "FINISHED" not in result or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"single-action GLB export failed: {path}")


def inspect_bound_glb(path):
    payload = retarget.read_glb_json(path)
    meshes = payload.get("meshes", [])
    skins = payload.get("skins", [])
    animations = payload.get("animations", [])
    materials = payload.get("materials", [])
    nodes = payload.get("nodes", [])
    if len(meshes) != 1:
        raise RuntimeError("bound GLB must contain exactly one mesh")
    if len(skins) != 1 or len(skins[0].get("joints", [])) != 80:
        raise RuntimeError("bound GLB must contain exactly one 80-joint skin")
    if len(animations) != 1 or not animations[0].get("channels"):
        raise RuntimeError("bound GLB must contain exactly one non-empty action")
    joint_names = [nodes[index].get("name") for index in skins[0]["joints"]]
    if set(joint_names) != set(TARGET_BONES):
        raise RuntimeError("bound GLB joint names differ from the Rocketbox contract")
    primitives = [primitive for mesh in meshes for primitive in mesh.get("primitives", [])]
    if not primitives:
        raise RuntimeError("bound GLB has no mesh primitives")
    required_attributes = {"TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"}
    for primitive in primitives:
        attributes = primitive.get("attributes", {})
        if not required_attributes.issubset(attributes):
            raise RuntimeError("bound GLB primitive is missing UV or skin attributes")
    material_checks = []
    for material in materials:
        pbr = material.get("pbrMetallicRoughness", {})
        material_checks.append(
            {
                "name": material.get("name"),
                "baseColorTexture": pbr.get("baseColorTexture"),
                "metallicRoughnessTexture": pbr.get("metallicRoughnessTexture"),
            }
        )
    if not any(
        item["baseColorTexture"] is not None
        and item["metallicRoughnessTexture"] is not None
        for item in material_checks
    ):
        raise RuntimeError("bound GLB is missing Hunyuan PBR texture bindings")
    return {
        "mesh_count": len(meshes),
        "primitive_count": len(primitives),
        "skin_count": len(skins),
        "skin_joint_count": len(skins[0]["joints"]),
        "animation_count": len(animations),
        "animation_channel_count": len(animations[0]["channels"]),
        "materials": material_checks,
        "uv_skin_attributes_present": True,
    }


def load_saved_target(bound_path, walk_name, idle_name):
    result = bpy.ops.wm.open_mainfile(filepath=str(bound_path))
    if "FINISHED" not in result:
        raise RuntimeError("could not reopen bound.blend")
    armature, target = identify_target_objects()
    actions = {action.name: action for action in bpy.data.actions}
    if set(actions) != {walk_name, idle_name}:
        raise RuntimeError("reopened bound.blend does not retain exactly walk and idle")
    validate_target_only_scene(armature, target)
    return armature, target, actions[walk_name], actions[idle_name]


def file_descriptor(path):
    path = require_regular_file(path, path.name)
    return {"filename": path.name, "sha256": sha256_file(path)}


def build_bind_manifest(
    args,
    output_dir,
    action_metrics,
    source_hashes,
    current_hashes,
    floor_z_m,
    consumed_inputs,
    axis_contract=None,
):
    manifest = {
        "schema_version": "hy3d_rocketbox_bind_v1",
        "asset_id": args.asset_id,
        "floor_z_m": floor_z_m,
        "reference": file_descriptor(output_dir/"reference.png"),
        "glbs": {
            "walk": file_descriptor(output_dir/"bound_walk.glb"),
            "idle": file_descriptor(output_dir/"bound_idle.glb"),
        },
        "bound_blend": file_descriptor(output_dir/"bound.blend"),
        "cleaned_obj_contract": {
            "role": "geometry_only",
            "materials": False,
            "uv": True,
            "normals": True,
        },
        "action_names": {
            "walk": action_metrics["walk"]["action_name"],
            "idle": action_metrics["idle"]["action_name"],
        },
        "artifacts": {
            "cleaned_obj": file_descriptor(output_dir/"cleaned.obj"),
            "bound_blend": file_descriptor(output_dir/"bound.blend"),
            "bound_walk_glb": file_descriptor(output_dir/"bound_walk.glb"),
            "bound_idle_glb": file_descriptor(output_dir/"bound_idle.glb"),
            "bind_metrics": file_descriptor(output_dir/"bind_metrics.json"),
        },
        "source_hashes": {**source_hashes, **current_hashes},
        "consumed_inputs": consumed_inputs,
    }
    if axis_contract is not None:
        manifest["axis_contract"] = axis_contract
    return manifest


def run_binding(args, validators=None):
    if args.asset_id not in EXPECTED_ASSET_IDS:
        raise ValueError(f"unexpected Rocketbox asset_id: {args.asset_id}")
    output_dir = require_real_directory(args.output_dir, "output directory")
    validators = dict(validators or {})
    baseline = validators.get("baseline", validate_baseline_inputs)(args)
    hy3d = validators.get("hy3d", validate_hy3d_inputs)(args)
    idle = validators.get("idle", validate_idle_motion)(args)
    source_hashes = capture_source_hashes(baseline, hy3d, idle)
    snapshot = None
    try:
        snapshot = stage_input_snapshot(output_dir, baseline, hy3d, idle)
        return consume_snapshot_binding(
            args,
            output_dir,
            baseline,
            hy3d,
            idle,
            source_hashes,
            snapshot,
        )
    finally:
        cleanup_input_snapshot(snapshot)


def consume_snapshot_binding(
    args,
    output_dir,
    baseline,
    hy3d,
    idle,
    source_hashes,
    snapshot,
):
    result = bpy.ops.wm.open_mainfile(
        filepath=str(snapshot["paths"]["baseline_blend"])
    )
    if "FINISHED" not in result:
        raise RuntimeError("could not open immutable baseline retarget.blend")
    retarget.configure_animation_scene()
    armature, source_mesh = identify_target_objects()
    if armature.animation_data is None or armature.animation_data.action is None:
        raise RuntimeError("approved baseline walk action is missing")
    walk_action = armature.animation_data.action
    if walk_action.name != f"{args.asset_id}_walk_neutral_retarget":
        raise RuntimeError("approved baseline walk action name changed")
    walk_action.use_fake_user = True
    source = capture_rocketbox_source(armature, source_mesh)
    floor_z_m = float(source["floor_z_m"])
    snapshot_hy3d_paths = {
        role: snapshot["paths"][f"hy3d_{role}"]
        for role in CONSUMED_HY3D_ROLES
    }
    target, axis_contract = import_hy3d_obj(
        snapshot["paths"]["hy3d_paint_obj"], args.asset_id
    )
    import_ground_metrics = cleanup_import_ground_artifacts(target)
    move_target_to_armature_space(target, armature)
    target_regions, cleanup_metrics = cleanup_target_geometry(target, source)
    cleanup_metrics["import_ground_artifact"] = import_ground_metrics
    pbr_metrics = assign_hunyuan_pbr_material(target, snapshot_hy3d_paths)
    armature.data.pose_position = "REST"
    bpy.context.view_layer.update()
    cleaned_obj_metrics = export_cleaned_obj(target, output_dir/"cleaned.obj")
    binding_metrics = bind_target_mesh(
        target, armature, source, target_regions
    )
    remove_original_body(source_mesh)
    validate_target_only_scene(armature, target)

    walk_start, walk_end = action_frame_range(walk_action)
    idle_action, idle_metrics = bake_idle_action(
        armature, args.asset_id, snapshot["paths"]["idle_motion_fbx"]
    )
    idle_action.use_fake_user = True
    action_set_metrics = validate_two_actions(walk_action, idle_action)
    walk_action_name = walk_action.name
    idle_action_name = idle_action.name
    save_bound_blend(armature, target, output_dir/"bound.blend")

    armature, target, walk_action, idle_action = load_saved_target(
        output_dir/"bound.blend", walk_action_name, idle_action_name
    )
    bound_blend_pbr = validate_packed_pbr_material(target)
    action_set_metrics = validate_two_actions(walk_action, idle_action)
    expected_mesh = retarget.mesh_metrics(target, armature)
    expected_skin = retarget.capture_skin_contract(target)
    walk_start, walk_end, walk_positions = sample_action_positions(armature, walk_action)
    idle_start, idle_end, idle_positions = sample_action_positions(armature, idle_action)

    walk_path = output_dir/"bound_walk.glb"
    idle_path = output_dir/"bound_idle.glb"
    isolate_action_for_export(armature, walk_action)
    export_single_action_glb(armature, target, walk_action, walk_path)
    armature, target, walk_action, idle_action = load_saved_target(
        output_dir/"bound.blend", walk_action_name, idle_action_name
    )
    isolate_action_for_export(armature, idle_action)
    export_single_action_glb(armature, target, idle_action, idle_path)
    walk_structure = inspect_bound_glb(walk_path)
    idle_structure = inspect_bound_glb(idle_path)
    atomic_copy(
        snapshot["paths"]["hy3d_reference"], output_dir/"reference.png"
    )

    if os.environ.get("HY3D_ROCKETBOX_BIND_FAIL_AFTER_EXPORT") == "1":
        raise RuntimeError("injected Hunyuan Rocketbox post-export failure")

    walk_roundtrip = retarget.roundtrip_validate(
        walk_path,
        expected_mesh,
        walk_positions,
        expected_skin,
        walk_start,
        walk_end,
    )
    idle_roundtrip = retarget.roundtrip_validate(
        idle_path,
        expected_mesh,
        idle_positions,
        expected_skin,
        idle_start,
        idle_end,
    )
    if not walk_roundtrip["skin_weight_validation"]["passed"]:
        raise RuntimeError("walk GLB skin_weight_validation failed")
    if not idle_roundtrip["skin_weight_validation"]["passed"]:
        raise RuntimeError("idle GLB skin_weight_validation failed")
    for role, roundtrip in (("walk", walk_roundtrip), ("idle", idle_roundtrip)):
        if roundtrip["maximum_world_joint_error_m"] >= roundtrip["joint_tolerance_m"]:
            raise RuntimeError(f"{role} GLB joint roundtrip validation failed")

    current_hashes = verify_source_hashes_current(
        baseline, hy3d, idle, source_hashes
    )
    action_metrics = {
        "walk": {
            "action_name": walk_action_name,
            "frame_start": walk_start,
            "frame_end": walk_end,
            "source": "approved baseline walk action",
        },
        "idle": {
            "action_name": idle_action_name,
            "frame_start": idle_start,
            "frame_end": idle_end,
            "source": "gender-matched idle source-absolute bake",
        },
    }
    metrics = {
        "schema_version": "hy3d_rocketbox_bind_metrics_v1",
        "asset_id": args.asset_id,
        "floor_z_m": floor_z_m,
        "axis_contract": axis_contract,
        "source_capture": {
            key: value
            for key, value in source.items()
            if key
            in {
                "mesh_name",
                "vertex_count",
                "face_count",
                "uv_layer_count",
                "material_slot_count",
                "bone_count",
            }
        },
        "cleanup": cleanup_metrics,
        "cleaned_obj": cleaned_obj_metrics,
        "pbr": pbr_metrics,
        "bound_blend_pbr": bound_blend_pbr,
        "binding": binding_metrics,
        "idle_bake": idle_metrics,
        "actions": action_metrics,
        "bound_action_set": action_set_metrics,
        "glb_structure": {"walk": walk_structure, "idle": idle_structure},
        "glb_roundtrip": {"walk": walk_roundtrip, "idle": idle_roundtrip},
        "source_hashes": {**source_hashes, **current_hashes},
        "consumed_inputs": snapshot["records"],
        "outputs": list(OUTPUT_FILENAMES),
    }
    atomic_write_json(output_dir/"bind_metrics.json", metrics)
    manifest = build_bind_manifest(
        args,
        output_dir,
        action_metrics,
        source_hashes,
        current_hashes,
        floor_z_m,
        snapshot["records"],
        axis_contract,
    )
    atomic_write_json(output_dir/"bind_manifest.json", manifest)
    print(f"HY3D_ROCKETBOX_BIND_OK asset_id={args.asset_id}")
    return manifest


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    invalidate_readiness(args.output_dir)
    try:
        run_binding(args)
    except BaseException:
        invalidate_readiness(args.output_dir)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
