#!/usr/bin/env python3
"""Hash-locked Route-2 agent-decision to Pixal3D job contract (no inference)."""

from __future__ import annotations

import ctypes
import errno
import json
import math
import os
import re
import tempfile
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from PIL import Image

from tools import human_attribute_masks
from tools import route2_human_contract_common as common
from tools import route2_human_instance_contract as route2_instance
from tools.spike_rlr import human_attribute_review as attribute_review


PIXAL3D_REVISION = "0b31f9160aa400719af409098bff7936a932f726"
DINO_REVISION = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
PASS_STATUS = "agent_qa_passed_pending_user_acceptance"
SCHEMA = "pixal3d_human_attribute_job_v1"
PIXAL_WRAPPER_PATH = Path(__file__).resolve().parent / "i23d_human_bakeoff.py"
PIXAL_WRAPPER_SHA256 = "6291e42a4f3ca6957beba4e2cd5749c264347657c98b9e067b66c2b2012fc799"
EXECUTOR_PATH = Path(__file__).resolve()
_ASSET_ID = re.compile(r"route2_[a-z0-9_]+_v1")


class PixalContractError(RuntimeError):
    """Raised when a 2D candidate is not authorized for Pixal3D."""


def sha256_file(path: Path) -> str:
    return common.sha256_file(Path(path))


def _load_json(path: Path, description: str) -> dict[str, Any]:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise PixalContractError(f"{description} must be a direct regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PixalContractError(f"{description} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise PixalContractError(f"{description} must contain an object")
    return value


def _validate_record(path: Path, record: Any, description: str) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise PixalContractError(f"{description} descriptor is missing")
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise PixalContractError(f"{description} must be a direct regular file")
    expected = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if any(record.get(key) != value for key, value in expected.items()):
        raise PixalContractError(f"{description} descriptor changed")
    return expected


def build_pixal_job(
    *,
    candidate_manifest: Path,
    agent_decision: Path,
    output_root: Path,
    wrapper_path: Path,
) -> dict[str, Any]:
    candidate_manifest = Path(candidate_manifest).absolute()
    manifest = _load_json(candidate_manifest, "candidate manifest")
    if manifest.get("schema") != "flux2_human_attribute_candidate_v2":
        raise PixalContractError("candidate manifest schema is not the Route-2 v2 contract")
    case_id = manifest.get("case_id")
    asset_id = manifest.get("downstream_asset_id")
    base_asset_id = manifest.get("base_asset_id")
    if (
        case_id not in human_attribute_masks.CASE_MASK_CONTRACTS
        or asset_id != f"route2_{case_id}_v1"
        or not isinstance(asset_id, str)
        or _ASSET_ID.fullmatch(asset_id) is None
        or base_asset_id
        != human_attribute_masks.CASE_MASK_CONTRACTS[str(case_id)]["base_asset_id"]
    ):
        raise PixalContractError("candidate asset lineage is invalid")
    decision_path = Path(agent_decision).absolute()
    if decision_path != attribute_review.decision_path(candidate_manifest.parent):
        raise PixalContractError("agent 2D QA decision path is not canonical")
    try:
        decision = attribute_review.assert_agent_2d_qa_passed(candidate_manifest.parent)
    except attribute_review.AttributeReviewError as error:
        raise PixalContractError(
            f"agent 2D QA has not passed for the exact candidate: {error}"
        ) from error
    if (
        decision.get("status") != PASS_STATUS
        or decision.get("case_id") != case_id
        or decision.get("base_asset_id") != base_asset_id
        or decision.get("downstream_asset_id") != asset_id
        or decision.get("snapshot", {}).get("candidate_manifest_sha256")
        != sha256_file(candidate_manifest)
    ):
        raise PixalContractError("agent 2D QA lineage or candidate snapshot changed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or "candidate_rgba.png" not in artifacts:
        raise PixalContractError("candidate RGBA artifact is missing")
    rgba = candidate_manifest.parent / "candidate_rgba.png"
    rgba_record = _validate_record(rgba, artifacts["candidate_rgba.png"], "candidate RGBA")
    with Image.open(rgba) as opened:
        opened.load()
        if opened.mode != "RGBA":
            raise PixalContractError("Pixal input must be RGBA")
        minimum, maximum = opened.getchannel("A").getextrema()
        if minimum == 255 or maximum == 0:
            raise PixalContractError("Pixal input must contain transparent and visible pixels")
        rgba_record["mode"] = "RGBA"
        rgba_record["size"] = list(opened.size)
        rgba_record["alpha_min"] = minimum
        rgba_record["alpha_max"] = maximum
    wrapper = Path(wrapper_path).absolute()
    if (
        wrapper != Path(PIXAL_WRAPPER_PATH).absolute()
        or wrapper.is_symlink()
        or not wrapper.is_file()
        or wrapper.resolve() != wrapper
        or sha256_file(wrapper) != PIXAL_WRAPPER_SHA256
    ):
        raise PixalContractError("Pixal job must use the exact pinned Pixal wrapper")
    output_root = Path(output_root).absolute()
    if output_root.is_symlink() or not output_root.is_dir() or output_root.resolve() != output_root:
        raise PixalContractError("Pixal output root must be a direct real directory")
    asset_root = output_root / asset_id
    if os.path.lexists(asset_root):
        raise PixalContractError("Pixal attribute output already exists")
    output_glb = asset_root / "canary_1024_seed42.glb"
    return {
        "schema": SCHEMA,
        "case_id": case_id,
        "asset_id": asset_id,
        "base_asset_id": base_asset_id,
        "state_classification": "research_candidate",
        "input_rgba": rgba_record,
        "candidate_manifest": {
            "path": str(candidate_manifest),
            "sha256": sha256_file(candidate_manifest),
            "size_bytes": candidate_manifest.stat().st_size,
        },
        "agent_2d_decision": {
            "path": str(decision_path),
            "sha256": sha256_file(decision_path),
            "size_bytes": decision_path.stat().st_size,
            "status": PASS_STATUS,
        },
        "model_revision": PIXAL3D_REVISION,
        "dino_revision": DINO_REVISION,
        "parameters": {
            "seed": 42,
            "manual_fov": 0.2,
            "resolution": 1024,
            "low_vram": True,
        },
        "wrapper": {
            "path": str(wrapper),
            "sha256": sha256_file(wrapper),
            "size_bytes": wrapper.stat().st_size,
        },
        "output_glb": str(output_glb),
        "output_manifest": str(asset_root / "canary_1024_seed42.manifest.json"),
        "output_policy": "atomic_no_replace",
        "executor": {
            "kind": "atomic_pixal3d_executor_v1",
            "argv": [
                str(wrapper),
                "--backend",
                "pixal3d",
                "--image",
                str(rgba),
                "--output",
                str(output_glb),
                "--gpu",
                "3",
                "--seed",
                "42",
                "--resolution",
                "1024",
                "--manual-fov",
                "0.2",
                "--low-vram",
            ],
            "execution_authorized": True,
            "atomic_no_replace": True,
            "path": str(EXECUTOR_PATH),
            "sha256": sha256_file(EXECUTOR_PATH),
            "size_bytes": EXECUTOR_PATH.stat().st_size,
        },
    }


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise RuntimeError("atomic no-replace publication requires renameat2")
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    result = function(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(destination)
    raise OSError(number, os.strerror(number), destination)


def publish_pixal_job(payload: Mapping[str, Any], destination: Path) -> Path:
    destination = Path(destination).absolute()
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise PixalContractError("Pixal job parent must be a direct real directory")
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    return common.write_json_immutable_noreplace(
        destination,
        payload,
        PixalContractError,
        "Pixal attribute job",
    )


def _embedded_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: record[key] for key in ("path", "sha256", "size_bytes")}


def _persistent_model_evidence(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project snapshot evidence to fields that must stay equal across a run.

    ``cache_hit`` describes how the evidence was obtained in this process, not the
    model snapshot.  Comparing it across pre/postflight would reject an unchanged
    snapshot merely because the second authentication used the verified cache.
    """
    required = ("path", "revision", "file_count", "inventory_sha256", "license")
    try:
        return {key: value[key] for key in required}
    except KeyError as error:
        raise PixalContractError(
            f"Pixal model evidence is missing persistent field {error.args[0]}"
        ) from error


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _authenticate_pixal_job_once(path: Path) -> dict[str, Any]:
    path = Path(path).absolute()
    payload, job_file = common.load_json_mapping_record(
        path,
        root=path.parent,
        description="Pixal attribute job",
        error_type=PixalContractError,
        require_mode=0o444,
    )
    if set(payload) != set(route2_instance.PIXAL_ATTRIBUTE_JOB_FIELDS):
        raise PixalContractError("Pixal attribute job fields changed")
    case_id = payload.get("case_id")
    asset_id = payload.get("asset_id")
    base_asset_id = payload.get("base_asset_id")
    contract = human_attribute_masks.CASE_MASK_CONTRACTS.get(str(case_id))
    if (
        payload.get("schema") != SCHEMA
        or contract is None
        or asset_id != f"route2_{case_id}_v1"
        or base_asset_id != contract["base_asset_id"]
        or payload.get("state_classification") != "research_candidate"
        or payload.get("model_revision") != PIXAL3D_REVISION
        or payload.get("dino_revision") != DINO_REVISION
        or payload.get("parameters")
        != {"seed": 42, "manual_fov": 0.2, "resolution": 1024, "low_vram": True}
        or payload.get("output_policy") != "atomic_no_replace"
    ):
        raise PixalContractError("Pixal attribute job identity or parameters changed")
    common.reject_user_approval(payload, PixalContractError, "Pixal attribute job")
    output_glb = Path(str(payload.get("output_glb"))).absolute()
    output_manifest = Path(str(payload.get("output_manifest"))).absolute()
    asset_root = output_glb.parent
    if (
        asset_root.name != asset_id
        or output_glb.name != "canary_1024_seed42.glb"
        or output_manifest != output_glb.with_suffix(".manifest.json")
        or not asset_root.parent.is_dir()
        or asset_root.parent.is_symlink()
        or asset_root.parent.resolve() != asset_root.parent
        or os.path.lexists(asset_root)
    ):
        raise PixalContractError("Pixal attribute output path is not an unused canonical root")
    if path != asset_root.parent / f"{asset_id}.pixal_job.json":
        raise PixalContractError("Pixal attribute job path is not canonical")

    candidate_value = payload.get("candidate_manifest")
    decision_value = payload.get("agent_2d_decision")
    rgba_value = payload.get("input_rgba")
    wrapper_value = payload.get("wrapper")
    if not all(
        isinstance(value, Mapping)
        for value in (candidate_value, decision_value, rgba_value, wrapper_value)
    ):
        raise PixalContractError("Pixal attribute job file descriptors are incomplete")
    candidate = Path(str(candidate_value.get("path"))).absolute()
    candidate_payload, candidate_file = common.load_json_mapping_record(
        candidate,
        root=candidate.parent,
        description="attribute candidate manifest",
        error_type=PixalContractError,
        require_mode=0o444,
    )
    if _embedded_record(candidate_file) != {
        key: candidate_value.get(key) for key in ("path", "sha256", "size_bytes")
    }:
        raise PixalContractError("attribute candidate manifest descriptor changed")
    decision = attribute_review.decision_path(candidate.parent)
    decision_file = common.file_record(
        decision,
        root=decision.parent,
        description="attribute agent 2D decision",
        error_type=PixalContractError,
        require_mode=0o444,
    )
    expected_decision = {
        **_embedded_record(decision_file),
        "status": PASS_STATUS,
    }
    if dict(decision_value) != expected_decision:
        raise PixalContractError("attribute agent 2D decision descriptor changed")
    try:
        accepted = attribute_review.assert_agent_2d_qa_passed(candidate.parent)
    except attribute_review.AttributeReviewError as error:
        raise PixalContractError(f"attribute agent 2D decision is not accepted: {error}") from error
    if (
        accepted.get("case_id") != case_id
        or accepted.get("base_asset_id") != base_asset_id
        or accepted.get("downstream_asset_id") != asset_id
    ):
        raise PixalContractError("attribute agent 2D decision lineage changed")
    rgba = Path(str(rgba_value.get("path"))).absolute()
    rgba_file = common.file_record(
        rgba,
        root=candidate.parent,
        description="accepted attribute RGBA",
        error_type=PixalContractError,
        require_mode=0o444,
    )
    if any(
        rgba_value.get(key) != rgba_file[key]
        for key in ("path", "sha256", "size_bytes")
    ):
        raise PixalContractError("accepted attribute RGBA descriptor changed")
    if candidate_payload.get("artifacts", {}).get("candidate_rgba.png") != {
        key: rgba_value[key] for key in ("path", "sha256", "size_bytes")
    }:
        raise PixalContractError("candidate manifest no longer binds accepted RGBA")
    with Image.open(rgba) as opened:
        opened.load()
        minimum, maximum = opened.getchannel("A").getextrema()
        if (
            opened.mode != "RGBA"
            or list(opened.size) != rgba_value.get("size")
            or minimum != rgba_value.get("alpha_min")
            or maximum != rgba_value.get("alpha_max")
            or not 0 <= minimum < 255
            or not 0 < maximum <= 255
        ):
            raise PixalContractError("accepted attribute RGBA metadata changed")
    wrapper = Path(str(wrapper_value.get("path"))).absolute()
    wrapper_file = common.file_record(
        wrapper,
        root=wrapper.parent,
        description="pinned Pixal generator wrapper",
        error_type=PixalContractError,
    )
    if (
        wrapper != Path(PIXAL_WRAPPER_PATH).absolute()
        or wrapper_file["sha256"] != PIXAL_WRAPPER_SHA256
        or _embedded_record(wrapper_file) != dict(wrapper_value)
    ):
        raise PixalContractError("pinned Pixal generator wrapper changed")
    expected_argv = [
        str(wrapper), "--backend", "pixal3d", "--image", str(rgba),
        "--output", str(output_glb), "--gpu", "3", "--seed", "42",
        "--resolution", "1024", "--manual-fov", "0.2", "--low-vram",
    ]
    executor_file = common.file_record(
        EXECUTOR_PATH,
        root=EXECUTOR_PATH.parent,
        description="Pixal atomic executor",
        error_type=PixalContractError,
    )
    if payload.get("executor") != {
        "kind": "atomic_pixal3d_executor_v1",
        "argv": expected_argv,
        "execution_authorized": True,
        "atomic_no_replace": True,
        **_embedded_record(executor_file),
    }:
        raise PixalContractError("Pixal atomic executor contract changed")
    models = {
        "pixal": _persistent_model_evidence(
            route2_instance.model_snapshot_evidence(PIXAL3D_REVISION)
        ),
        "dino": _persistent_model_evidence(
            route2_instance.model_snapshot_evidence(DINO_REVISION)
        ),
    }
    return {
        "payload": payload,
        "job_record": _embedded_record(job_file),
        "model_evidence": models,
    }


def reauthenticate_pixal_job(path: Path) -> dict[str, Any]:
    return common.stable_mapping_snapshot(
        lambda: _authenticate_pixal_job_once(path),
        PixalContractError,
        "Pixal attribute job and all owner inputs",
    )


def probe_pixal_runtime() -> dict[str, Any]:
    executable = common.absolute(route2_instance.PIXAL_PYTHON_EXECUTABLE)
    executable_record = common.file_record(
        executable,
        root=executable.parent,
        description="Pixal Python executable",
        error_type=PixalContractError,
    )
    environment = {
        "cuda_visible_devices": "3",
        "attention_backend": "sdpa",
        "hf_hub_cache": "/data/models/hub",
        "hf_hub_offline": "1",
        "transformers_offline": "1",
        "torch_home": "/data/models/torch",
        "opencv_io_enable_openexr": "1",
        "pytorch_cuda_alloc_conf": "expandable_segments:True",
    }
    try:
        runtime = route2_instance._probe_pixal_python_runtime(executable, environment)
    except route2_instance.InstanceContractError as error:
        raise PixalContractError(f"Pixal runtime probe failed: {error}") from error
    return {
        "python_executable": str(executable),
        "python_executable_record": {
            key: executable_record[key]
            for key in ("path", "sha256", "size_bytes", "mode")
        },
        **runtime,
        **environment,
    }


def _runtime_process_environment(environment: Mapping[str, Any]) -> dict[str, str]:
    result = dict(os.environ)
    mapping = {
        "CUDA_VISIBLE_DEVICES": "cuda_visible_devices",
        "ATTN_BACKEND": "attention_backend",
        "HF_HUB_CACHE": "hf_hub_cache",
        "HF_HUB_OFFLINE": "hf_hub_offline",
        "TRANSFORMERS_OFFLINE": "transformers_offline",
        "TORCH_HOME": "torch_home",
        "OPENCV_IO_ENABLE_OPENEXR": "opencv_io_enable_openexr",
        "PYTORCH_CUDA_ALLOC_CONF": "pytorch_cuda_alloc_conf",
    }
    for target, source in mapping.items():
        result[target] = str(environment[source])
    return result


def validate_staged_pixal_glb(
    path: Path,
    *,
    staging: Path,
    input_rgba: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject parseable-but-empty or textureless Pixal output before publication."""
    path = Path(path).absolute()
    input_rgba = Path(input_rgba).absolute()
    try:
        if os.path.samefile(path, input_rgba):
            raise PixalContractError("staged Pixal GLB aliases its input image")
    except FileNotFoundError:
        pass
    document, binary, record = common.load_glb_document_binary_record(
        path,
        root=staging,
        description="staged Pixal GLB",
        error_type=PixalContractError,
    )
    try:
        route2_instance._validate_pixal_pbr_document(
            document,
            binary,
            "staged Pixal GLB",
        )
    except route2_instance.InstanceContractError as error:
        raise PixalContractError(str(error)) from error
    accessors = document.get("accessors")
    meshes = document.get("meshes")
    materials = document.get("materials")
    textures = document.get("textures")
    images = document.get("images")
    if not all(isinstance(value, list) and value for value in (accessors, meshes, materials, textures, images)):
        raise PixalContractError(
            "staged Pixal GLB requires non-empty mesh, material, texture, image, and accessor arrays"
        )
    used_materials: set[int] = set()
    primitive_count = 0
    for mesh in meshes:
        primitives = mesh.get("primitives") if isinstance(mesh, Mapping) else None
        if not isinstance(primitives, list) or not primitives:
            raise PixalContractError("staged Pixal GLB mesh has no primitive")
        for primitive in primitives:
            primitive_count += 1
            attributes = primitive.get("attributes") if isinstance(primitive, Mapping) else None
            position = attributes.get("POSITION") if isinstance(attributes, Mapping) else None
            material = primitive.get("material") if isinstance(primitive, Mapping) else None
            if (
                not isinstance(position, int)
                or isinstance(position, bool)
                or not 0 <= position < len(accessors)
                or not isinstance(material, int)
                or isinstance(material, bool)
                or not 0 <= material < len(materials)
            ):
                raise PixalContractError("staged Pixal GLB primitive lacks POSITION or material")
            accessor = accessors[position]
            bounds = [
                number
                for key in ("min", "max")
                for number in (accessor.get(key, []) if isinstance(accessor, Mapping) else [])
            ]
            if (
                not isinstance(accessor, Mapping)
                or not isinstance(accessor.get("count"), int)
                or accessor["count"] <= 0
                or len(bounds) != 6
                or any(
                    not isinstance(number, (int, float))
                    or isinstance(number, bool)
                    or not math.isfinite(float(number))
                    for number in bounds
                )
            ):
                raise PixalContractError("staged Pixal GLB POSITION accessor is empty or non-finite")
            used_materials.add(material)
    referenced_images: set[int] = set()
    for material_index in used_materials:
        material = materials[material_index]
        pbr = material.get("pbrMetallicRoughness") if isinstance(material, Mapping) else None
        if not isinstance(pbr, Mapping):
            raise PixalContractError("staged Pixal GLB material has no PBR mapping")
        for key in ("baseColorTexture", "metallicRoughnessTexture"):
            texture_ref = pbr.get(key)
            texture_index = texture_ref.get("index") if isinstance(texture_ref, Mapping) else None
            if (
                not isinstance(texture_index, int)
                or isinstance(texture_index, bool)
                or not 0 <= texture_index < len(textures)
            ):
                raise PixalContractError(
                    f"staged Pixal GLB material has no valid {key} PBR reference"
                )
            texture = textures[texture_index]
            image_index = texture.get("source") if isinstance(texture, Mapping) else None
            if image_index is None and isinstance(texture, Mapping):
                extensions = texture.get("extensions")
                webp = (
                    extensions.get("EXT_texture_webp")
                    if isinstance(extensions, Mapping)
                    else None
                )
                if isinstance(webp, Mapping):
                    if "EXT_texture_webp" not in document.get("extensionsUsed", []):
                        raise PixalContractError(
                            "staged Pixal GLB uses undeclared EXT_texture_webp"
                        )
                    image_index = webp.get("source")
            if (
                not isinstance(image_index, int)
                or isinstance(image_index, bool)
                or not 0 <= image_index < len(images)
            ):
                raise PixalContractError("staged Pixal GLB texture has no valid image source")
            referenced_images.add(image_index)
    buffer_views = document.get("bufferViews")
    for image_index in referenced_images:
        image = images[image_index]
        if not isinstance(image, Mapping):
            raise PixalContractError("staged Pixal GLB image descriptor is invalid")
        uri = image.get("uri")
        buffer_view = image.get("bufferView")
        if isinstance(uri, str):
            if (
                not uri.startswith("data:image/")
                or ";base64," not in uri
                or not uri.rsplit(",", 1)[-1]
            ):
                raise PixalContractError(
                    "staged Pixal GLB PBR image must be packed as a data URI"
                )
        elif (
            isinstance(buffer_view, int)
            and not isinstance(buffer_view, bool)
            and isinstance(buffer_views, list)
            and 0 <= buffer_view < len(buffer_views)
            and isinstance(buffer_views[buffer_view], Mapping)
            and isinstance(buffer_views[buffer_view].get("byteLength"), int)
            and not isinstance(buffer_views[buffer_view].get("byteLength"), bool)
            and buffer_views[buffer_view]["byteLength"] > 0
            and image.get("mimeType") in {"image/png", "image/jpeg", "image/webp"}
        ):
            pass
        else:
            raise PixalContractError(
                "staged Pixal GLB PBR image is not a non-empty packed image"
            )
    if primitive_count <= 0 or not used_materials or not referenced_images:
        raise PixalContractError("staged Pixal GLB has no reviewable packed PBR primitive")
    return document, record


def _fsync_readonly_tree(root: Path) -> None:
    directories = [Path(root)]
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise PixalContractError(f"Pixal staging contains a symlink: {path}")
        if path.is_dir():
            directories.append(path)
            continue
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        common.fsync_directory(directory)


def _seal_failure_bundle(
    evidence: Path,
    *,
    payload: Mapping[str, Any],
) -> Path:
    evidence = common.require_real_directory(
        evidence, "Pixal failure evidence", PixalContractError
    )
    directories = [evidence]
    for path in sorted(evidence.rglob("*")):
        if path.is_symlink():
            raise PixalContractError(f"Pixal failure evidence contains a link: {path}")
        if path.is_dir():
            directories.append(path)
            continue
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    artifacts = []
    for path in sorted(
        (value for value in evidence.rglob("*") if value.is_file()),
        key=lambda value: value.relative_to(evidence).as_posix(),
    ):
        record = common.file_record(
            path,
            root=evidence,
            description="Pixal failure artifact",
            error_type=PixalContractError,
            require_mode=0o444,
        )
        artifacts.append(
            {
                "relative_path": record["relative_path"],
                "sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
                "mode": record["mode"],
            }
        )
    manifest = evidence / "failure_manifest.json"
    common.write_json_immutable_noreplace(
        manifest,
        {**dict(payload), "artifacts": artifacts},
        PixalContractError,
        "Pixal failure manifest",
    )
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        descriptor = os.open(
            directory,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    common.fsync_directory(evidence.parent)
    return manifest


def execute_atomic_pixal_job(
    job_path: Path,
    *,
    subprocess_runner: Callable[..., Any] = subprocess.run,
) -> Path:
    """Run one authorized job into staging and atomically publish all evidence."""
    before = reauthenticate_pixal_job(job_path)
    job = before["payload"]
    execution_guard_before = route2_instance.pixal_execution_guard_evidence()
    public_glb = Path(job["output_glb"]).absolute()
    public_manifest = Path(job["output_manifest"]).absolute()
    asset_root = public_glb.parent
    output_root = asset_root.parent
    attempt_id = f"attempt_{uuid.uuid4().hex}"
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{asset_root.name}.{attempt_id}.",
            suffix=".staging",
            dir=output_root,
        )
    )
    start_root = output_root / ".attempts" / asset_root.name
    failure_root = output_root / ".failed_attempts" / asset_root.name
    start_root.mkdir(parents=True, exist_ok=True)
    failure_root.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    start_path = start_root / f"{attempt_id}.started.json"
    executor_record = {
        key: job["executor"][key] for key in ("path", "sha256", "size_bytes")
    }
    start_staging = {"path": str(staging), "created": True}
    common.write_json_immutable_noreplace(
        start_path,
        {
            "schema": "pixal3d_human_attribute_attempt_start_v1",
            "attempt_id": attempt_id,
            "status": "started",
            "case_id": job["case_id"],
            "asset_id": job["asset_id"],
            "base_avatar_id": job["base_asset_id"],
            "started_at_utc": started_at,
            "job": before["job_record"],
            "executor": executor_record,
            "execution_guard_before": execution_guard_before,
            "argv": list(job["executor"]["argv"]),
            "staging": start_staging,
            "publication_policy": "atomic_no_replace",
        },
        PixalContractError,
        "Pixal attempt start ledger",
    )
    start_file = common.file_record(
        start_path,
        root=start_root,
        description="Pixal attempt start ledger",
        error_type=PixalContractError,
        require_mode=0o444,
    )
    staged_glb = staging / public_glb.name
    staged_manifest = staging / public_manifest.name
    logical_argv = list(job["executor"]["argv"])
    staged_argv = list(logical_argv)
    staged_argv[staged_argv.index("--output") + 1] = str(staged_glb)
    completed: Any = None
    environment: dict[str, Any] | None = None
    command: list[str] | None = None
    publication_moved = False
    failure_stage = "runtime_probe"
    try:
        environment = probe_pixal_runtime()
        command = [environment["python_executable"], *staged_argv]
        failure_stage = "subprocess"
        completed = subprocess_runner(
            command,
            cwd=str(Path(job["wrapper"]["path"]).parent),
            env=_runtime_process_environment(environment),
            check=False,
            capture_output=True,
            text=True,
        )
        stdout = str(getattr(completed, "stdout", "") or "")
        stderr = str(getattr(completed, "stderr", "") or "")
        sentinel = str(staged_manifest)
        execution_log_path = staging / "execution.log"
        failure_stage = "execution_log"
        common.write_json_immutable_noreplace(
            execution_log_path,
            {
                "schema": "pixal3d_human_attribute_execution_log_v1",
                "attempt_id": attempt_id,
                "returncode": int(completed.returncode),
                "logical_argv": logical_argv,
                "staged_command": command,
                "stdout": stdout,
                "stderr": stderr,
                "success_sentinel": sentinel,
            },
            PixalContractError,
            "Pixal execution log",
        )
        if getattr(completed, "returncode", None) != 0:
            failure_stage = "subprocess_returncode"
            raise PixalContractError(
                f"Pixal generator failed with returncode {getattr(completed, 'returncode', None)}"
            )
        failure_stage = "success_sentinel"
        stdout_lines = [line for line in stdout.splitlines() if line.strip()]
        if (
            not stdout_lines
            or stdout_lines[-1] != sentinel
            or stdout_lines.count(sentinel) != 1
        ):
            raise PixalContractError(
                "Pixal generator stdout has no unique staged-manifest success sentinel"
            )
        failure_stage = "glb_readback"
        _, staged_glb_file = validate_staged_pixal_glb(
            staged_glb,
            staging=staging,
            input_rgba=Path(job["input_rgba"]["path"]),
        )
        failure_stage = "manifest_readback"
        generated_manifest, _ = common.load_json_mapping_record(
            staged_manifest,
            root=staging,
            description="staged Pixal manifest",
            error_type=PixalContractError,
        )
        generated_input = generated_manifest.get("input")
        generated_output = generated_manifest.get("output")
        if (
            generated_manifest.get("backend") != "pixal3d"
            or not isinstance(generated_input, Mapping)
            or generated_input.get("path") != job["input_rgba"]["path"]
            or generated_input.get("sha256") != job["input_rgba"]["sha256"]
            or not isinstance(generated_output, Mapping)
            or generated_output.get("path") != str(staged_glb)
            or generated_output.get("sha256") != staged_glb_file["sha256"]
            or generated_output.get("bytes") != staged_glb_file["size_bytes"]
            or generated_manifest.get("model", {}).get("revision") != PIXAL3D_REVISION
            or generated_manifest.get("dino", {}).get("revision") != DINO_REVISION
            or generated_manifest.get("parameters") != job["parameters"]
        ):
            raise PixalContractError("staged Pixal manifest readback changed")
        failure_stage = "postflight_reauthentication"
        after = reauthenticate_pixal_job(job_path)
        if common.canonical_json(after) != common.canonical_json(before):
            raise PixalContractError("Pixal job or owner inputs changed during inference")
        failure_stage = "execution_guard"
        execution_guard_after = route2_instance.pixal_execution_guard_evidence()
        if common.canonical_json(execution_guard_after) != common.canonical_json(
            execution_guard_before
        ):
            raise PixalContractError("Pixal execution guard changed during inference")
        models = after["model_evidence"]
        failure_stage = "final_evidence"
        final_manifest = {
            "backend": "pixal3d",
            "asset_id": job["asset_id"],
            "case_id": job["case_id"],
            "base_avatar_id": job["base_asset_id"],
            "input": {
                key: job["input_rgba"][key]
                for key in ("path", "sha256", "mode", "size", "alpha_min", "alpha_max")
            },
            "output": {
                "path": str(public_glb),
                "sha256": staged_glb_file["sha256"],
                "bytes": staged_glb_file["size_bytes"],
            },
            "model": {
                "snapshot": models["pixal"]["path"],
                "revision": PIXAL3D_REVISION,
            },
            "dino": {
                "snapshot": models["dino"]["path"],
                "revision": DINO_REVISION,
            },
            "parameters": dict(job["parameters"]),
        }
        staged_manifest.unlink()
        common.write_json_immutable_noreplace(
            staged_manifest,
            final_manifest,
            PixalContractError,
            "final staged Pixal manifest",
        )
        manifest_file = common.file_record(
            staged_manifest,
            root=staging,
            description="final staged Pixal manifest",
            error_type=PixalContractError,
            require_mode=0o444,
        )
        execution_log_file = common.file_record(
            execution_log_path,
            root=staging,
            description="Pixal execution log",
            error_type=PixalContractError,
            require_mode=0o444,
        )
        attempt = {
            "schema": route2_instance.PIXAL_ATTRIBUTE_ATTEMPT_SCHEMA,
            "attempt_id": attempt_id,
            "status": "succeeded",
            "case_id": job["case_id"],
            "asset_id": job["asset_id"],
            "base_avatar_id": job["base_asset_id"],
            "job": before["job_record"],
            "argv": logical_argv,
            "environment": environment,
            "wrapper": dict(job["wrapper"]),
            "executor": executor_record,
            "execution_guard": {
                "before": execution_guard_before,
                "after": execution_guard_after,
                "unchanged": True,
            },
            "start_ledger": _embedded_record(start_file),
            "execution_log": {
                "path": str(asset_root / "execution.log"),
                "sha256": execution_log_file["sha256"],
                "size_bytes": execution_log_file["size_bytes"],
            },
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "returncode": int(completed.returncode),
            "preflight_reauthenticated": True,
            "postflight_reauthenticated": True,
            "staging": {
                "path": str(staging),
                "created": True,
                "preserved_after_success": False,
            },
            "publication": {
                "policy": "atomic_no_replace",
                "glb_published": True,
                "manifest_published": True,
            },
            "model_inventory": {
                "pixal_revision": PIXAL3D_REVISION,
                "dino_revision": DINO_REVISION,
                "pixal_snapshot_inventory_sha256": models["pixal"]["inventory_sha256"],
                "dino_snapshot_inventory_sha256": models["dino"]["inventory_sha256"],
            },
            "licenses": {
                "pixal_license_sha256": models["pixal"]["license"]["sha256"],
                "dino_license_sha256": models["dino"]["license"]["sha256"],
            },
            "output_glb": {
                "path": str(public_glb),
                "sha256": staged_glb_file["sha256"],
                "size_bytes": staged_glb_file["size_bytes"],
            },
            "output_manifest": {
                "path": str(public_manifest),
                "sha256": manifest_file["sha256"],
                "size_bytes": manifest_file["size_bytes"],
            },
            "failure_evidence": [],
        }
        if set(attempt) != set(route2_instance.PIXAL_ATTRIBUTE_ATTEMPT_FIELDS):
            raise PixalContractError("Pixal succeeded attempt fields are incomplete")
        attempt_path = staging / "pixal_attempt.json"
        common.write_json_immutable_noreplace(
            attempt_path,
            attempt,
            PixalContractError,
            "final staged Pixal attempt",
        )
        _fsync_readonly_tree(staging)
        failure_stage = "publication"
        _rename_noreplace(staging, asset_root)
        publication_moved = True
        common.fsync_directory(output_root)
        return asset_root / "pixal_attempt.json"
    except BaseException as error:
        evidence = failure_root / attempt_id
        preservation_source: Path | None = None
        if publication_moved and asset_root.exists():
            preservation_source = asset_root
        elif staging.exists():
            preservation_source = staging
        if preservation_source is not None:
            try:
                _rename_noreplace(preservation_source, evidence)
                common.fsync_directory(preservation_source.parent)
                common.fsync_directory(failure_root)
                message = str(error) or f"{type(error).__name__} without message"
                _seal_failure_bundle(
                    evidence,
                    payload={
                        "schema": "pixal3d_human_attribute_failure_bundle_v1",
                        "attempt_id": attempt_id,
                        "status": "failed",
                        "case_id": job["case_id"],
                        "asset_id": job["asset_id"],
                        "base_avatar_id": job["base_asset_id"],
                        "job": before["job_record"],
                        "start_ledger": _embedded_record(start_file),
                        "failure_stage": failure_stage,
                        "error": {
                            "type": type(error).__name__,
                            "message": message,
                        },
                        "returncode": getattr(completed, "returncode", None),
                    },
                )
            except BaseException as preservation_error:
                raise PixalContractError(
                    "Pixal execution failed and failure evidence could not be sealed: "
                    f"{preservation_error}"
                ) from preservation_error
        else:
            raise PixalContractError(
                "Pixal execution failed and its staging tree disappeared before "
                f"failure preservation: {type(error).__name__}: {error}"
            ) from error
        raise
