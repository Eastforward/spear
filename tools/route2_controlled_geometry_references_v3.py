#!/usr/bin/env python3
"""Generate fail-closed FLUX.2 canaries for Route-2 v3 geometry templates.

This is deliberately a research-candidate preflight.  It does not consume or
manufacture the formal male/female qualified-candidate gate, and it never
authorizes Pixal3D by inference success alone.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import flux2_edit_human_attributes as flux2_base
from tools import route2_human_contract_common as route2_common


SCHEMA = "route2_controlled_geometry_reference_jobs_v3"
CANDIDATE_SCHEMA = "route2_controlled_geometry_reference_candidate_v3"
DECISION_SCHEMA = "route2_controlled_geometry_reference_agent_qa_v1"
PIXAL_JOBS_SCHEMA = "route2_controlled_geometry_pixal_jobs_v1"
RUNNER_PATH = Path(__file__).resolve()
SPEAR_ROOT = RUNNER_PATH.parents[1]
OUTPUT_ROOT = SPEAR_ROOT / "tmp/route2_controlled_geometry_references_v3"
REFERENCE_ROOT = SPEAR_ROOT / "tmp/human_reference_review"
ALPHA_ROOT = SPEAR_ROOT / "tmp/i23d_human_bakeoff_v1/inputs"
MODEL_ROOT = Path("/data/models/hub/models--black-forest-labs--FLUX.2-klein-4B")
MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
MODEL_INVENTORY = (
    SPEAR_ROOT / "tmp/human_attribute_instances_v1/flux2_snapshot_inventory_v1.json"
)
MODEL_INVENTORY_SHA256 = "962ec618f2846728da8ac4ccb18fb61bdf6334c729017b3feaa48ae7710f04a4"
ISNET_MODEL = Path("/data/models/rembg/isnet-general-use/isnet-general-use.onnx")
ISNET_MODEL_SHA256 = "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a"
ISNET_PROVENANCE = SPEAR_ROOT / "tmp/human_attribute_instances_v1/isnet_provenance_v1.json"
ISNET_PROVENANCE_SHA256 = "42db586046ba2d11cac285074085439037fb6036b8fc294cbbe291dceedbb798"
ISNET_PYTHON = Path("/data/jzy/miniconda3/envs/hunyuan3d/bin/python")
PIXAL_REVISION = "0b31f9160aa400719af409098bff7936a932f726"
WIDTH = 1152
HEIGHT = 1536
STEPS = 28
GUIDANCE_SCALE = 1.0
MAX_SEQUENCE_LENGTH = 512
_SHA256 = re.compile(r"[0-9a-f]{64}")
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


SOURCE_PINS: dict[str, dict[str, str]] = {
    "male": {
        "asset_id": "rocketbox_male_adult_01",
        "image_sha256": "820abc0edb324bee570614cc901b03112589b28f3ea11e14d971788bc97a0938",
        "manifest_sha256": "3f7fbce0a11a4d92ce058fe43f24aeea45f5296241190ab3d9d49dd77efcee79",
        "review_sha256": "5b55506b1323ad60f60d5ec543992479b840827d7fa47660b23447ed648a6851",
        "alpha_sha256": "72030567eb49571a8fb9d141cd45f93b9c233d043a5786bbbedd162319947b56",
    },
    "female": {
        "asset_id": "rocketbox_female_adult_01",
        "image_sha256": "856df2ca3840cf74c9a48cb1ac2081fc0ac61700f5f2fb47aa4a37eb561fa03c",
        "manifest_sha256": "c1f33ebd417be1991507e1bbdd4fde279d7db0aa1a16205736f2b55894218efc",
        "review_sha256": "78faef9bf3fdc4577ab0ec490ef7795316d48e4e0b2c12ac9ab114dcc44af79a",
        "alpha_sha256": "17dafe1f2e3526cfc9b1297f3f2007e967b133b3db838388641f490c2bd54435",
    },
}


def _prompt(subject: str, target: str, preserve: str, avoid: str) -> tuple[str, str]:
    prompt = (
        f"Edit Image 1, the approved full-body soft T-pose reference of the same adult {subject}. "
        f"Change exactly one geometry attribute: {target}. "
        f"Preserve exactly {preserve}. The result remains a photorealistic front-facing full-body "
        "studio reference on the identical light-gray background."
    )
    negative = (
        f"different identity, face change, age change, body shape change, pose change, camera change, "
        f"cropped body, moved hands, fused limbs, lost limb gaps, text, logo, pattern, extra person, {avoid}"
    )
    return prompt, negative


_PRESERVE_COMMON = (
    "the face, skin tone, body proportions, height, hairstyle except the explicitly allowed "
    "hat-contact hair region, symmetrical soft T-pose, arm and leg gaps, open hands, shoes, "
    "framing, camera, lighting, and every non-target garment"
)


def _case(
    case_id: str,
    sex: str,
    geometry: str,
    seed: int,
    target: str,
    avoid: str,
    *,
    hair_exception: bool = False,
) -> dict[str, Any]:
    preserve = _PRESERVE_COMMON
    if not hair_exception:
        preserve = preserve.replace(
            "hairstyle except the explicitly allowed hat-contact hair region",
            "the complete hairstyle",
        )
    prompt, negative = _prompt("man" if sex == "male" else "woman", target, preserve, avoid)
    return {
        "case_id": case_id,
        "sex": sex,
        "base_asset_id": SOURCE_PINS[sex]["asset_id"],
        "geometry_attribute": geometry,
        "seed": seed,
        "prompt": prompt,
        "negative_prompt": negative,
        "mask_version": "route2_controlled_geometry_mask_v3",
        "target_color_policy": "preserve_base_or_fixed_neutral",
    }


CASE_SPECS: tuple[dict[str, Any], ...] = (
    _case(
        "male_long_sleeve",
        "male",
        "long_sleeve",
        301,
        "extend only the existing plain dark forest-green T-shirt sleeves into fitted full-length sleeves ending exactly at both wrists; keep the torso, collar, fabric, color, hands, and trousers unchanged",
        "short sleeves, rolled sleeves, jacket, hoodie, exposed forearms, changed shirt torso or color",
    ),
    _case(
        "female_long_sleeve",
        "female",
        "long_sleeve",
        302,
        "extend only the existing plain deep-burgundy T-shirt sleeves into fitted full-length sleeves ending exactly at both wrists; keep the torso, collar, fabric, color, hands, and trousers unchanged",
        "short sleeves, rolled sleeves, jacket, hoodie, exposed forearms, changed shirt torso or color",
    ),
    _case(
        "male_shorts",
        "male",
        "shorts",
        303,
        "convert only the charcoal full-length trousers into plain straight knee-length shorts ending just above both kneecaps, revealing the same bare lower legs; retain the waist, color family, fabric, and shoes",
        "full-length trousers, skirt, very short shorts, cropped shoes, socks, changed shirt or legs",
    ),
    _case(
        "female_shorts",
        "female",
        "shorts",
        304,
        "convert only the dark-navy full-length trousers into plain straight knee-length shorts ending just above both kneecaps, revealing the same bare lower legs; retain the waist, color family, fabric, and shoes",
        "full-length trousers, skirt, very short shorts, cropped shoes, tights, changed shirt or legs",
    ),
    _case(
        "male_baseball_cap",
        "male",
        "plain_baseball_cap_hat_compatible_hair",
        305,
        "add one plain neutral navy baseball cap fitted naturally to the head with one short forward brim, and compress only the hair physically under the cap into a hat-compatible state",
        "wide-brim hat, helmet, hood, crown, backward cap, multiple hats, glasses, changed face",
        hair_exception=True,
    ),
    _case(
        "female_baseball_cap",
        "female",
        "plain_baseball_cap_hat_compatible_hair",
        306,
        "add one plain neutral navy baseball cap fitted naturally to the head with one short forward brim, compressing only crown hair under the cap while keeping a tidy hat-compatible blonde ponytail",
        "wide-brim hat, helmet, hood, crown, backward cap, multiple hats, glasses, loose new hairstyle, changed face",
        hair_exception=True,
    ),
    _case(
        "male_rectangular_glasses",
        "male",
        "thin_rectangular_glasses",
        307,
        "add exactly one pair of thin matte dark-gray rectangular prescription eyeglasses, centered over both eyes with a small bridge and natural temples",
        "sunglasses, thick frames, goggles, round frames, extra glasses, hat, changed eyes or hair",
    ),
    _case(
        "female_rectangular_glasses",
        "female",
        "thin_rectangular_glasses",
        308,
        "add exactly one pair of thin matte dark-gray rectangular prescription eyeglasses, centered over both eyes with a small bridge and natural temples",
        "sunglasses, thick frames, goggles, round frames, extra glasses, hat, changed eyes or hair",
    ),
)
CASE_BY_ID = {str(case["case_id"]): case for case in CASE_SPECS}


class GeometryReferenceError(RuntimeError):
    """Raised when a v3 reference cannot be authenticated or published."""


def sha256_file(path: Path) -> str:
    return route2_common.sha256_file(Path(path))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _regular_file(path: Path, description: str, *, mode: int | None = None) -> Path:
    path = Path(path).absolute()
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
        or not stat.S_ISREG(os.lstat(path).st_mode)
        or path.stat().st_size <= 0
    ):
        raise GeometryReferenceError(f"{description} must be a direct nonempty regular file: {path}")
    if mode is not None and stat.S_IMODE(path.stat().st_mode) != mode:
        raise GeometryReferenceError(f"{description} must have mode {mode:04o}")
    return path


def _record(path: Path, *, public_path: Path | None = None) -> dict[str, Any]:
    path = _regular_file(path, "artifact")
    return {
        "path": str(public_path if public_path is not None else path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _require_record(path: Path, record: Any, description: str) -> Path:
    path = _regular_file(path, description)
    expected = _record(path)
    if not isinstance(record, Mapping) or any(
        record.get(key) != expected[key] for key in ("path", "sha256", "size_bytes")
    ):
        raise GeometryReferenceError(f"{description} descriptor changed")
    return path


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise GeometryReferenceError("atomic no-replace publication requires renameat2")
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    result = function(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    number = ctypes.get_errno()
    if number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(destination)
    raise OSError(number, os.strerror(number), destination)


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _readonly_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), reverse=True):
        path.chmod(0o755)
    root.chmod(0o755)


def _source_paths(sex: str) -> dict[str, Path]:
    pin = SOURCE_PINS[sex]
    source_root = REFERENCE_ROOT / pin["asset_id"]
    alpha_root = ALPHA_ROOT / pin["asset_id"]
    return {
        "image": source_root / "candidate.png",
        "candidate_manifest": source_root / "candidate_manifest.json",
        "review": source_root / "reference_review.json",
        "alpha": alpha_root / "alpha_isnet.png",
    }


def authenticate_source(sex: str) -> dict[str, Any]:
    if sex not in SOURCE_PINS:
        raise GeometryReferenceError(f"unknown source sex: {sex}")
    pin = SOURCE_PINS[sex]
    paths = _source_paths(sex)
    for key, path in paths.items():
        _regular_file(path, f"{sex} approved {key}")
        expected = pin[f"{key}_sha256"] if key != "candidate_manifest" else pin["manifest_sha256"]
        if sha256_file(path) != expected:
            raise GeometryReferenceError(f"{sex} approved {key} SHA-256 changed")
    try:
        manifest = json.loads(paths["candidate_manifest"].read_text(encoding="utf-8"))
        review = json.loads(paths["review"].read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GeometryReferenceError(f"{sex} approved source JSON is invalid: {error}") from error
    if (
        manifest.get("schema_version") != "human_reference_candidate_v1"
        or manifest.get("asset_id") != pin["asset_id"]
        or manifest.get("model_revision") != MODEL_REVISION
        or manifest.get("output_sha256") != pin["image_sha256"]
        or (manifest.get("width"), manifest.get("height")) != (WIDTH, HEIGHT)
        or review.get("schema_version") != "human_reference_review_v1"
        or review.get("asset_id") != pin["asset_id"]
        or review.get("decision") != "approved"
        or review.get("candidate_sha256") != pin["image_sha256"]
        or review.get("candidate_manifest_sha256") != pin["manifest_sha256"]
        or not isinstance(review.get("reviewer"), str)
        or not review["reviewer"].strip()
    ):
        raise GeometryReferenceError(f"{sex} soft-T source approval lineage changed")
    with Image.open(paths["image"]) as opened:
        if opened.size != (WIDTH, HEIGHT) or opened.format != "PNG":
            raise GeometryReferenceError(f"{sex} source image canvas/format changed")
    with Image.open(paths["alpha"]) as opened:
        alpha = opened.convert("L")
        if alpha.size != (WIDTH, HEIGHT) or alpha.getextrema() == (0, 0):
            raise GeometryReferenceError(f"{sex} source alpha is empty or has wrong canvas")
    return {
        "sex": sex,
        "asset_id": pin["asset_id"],
        "image": _record(paths["image"]),
        "candidate_manifest": _record(paths["candidate_manifest"]),
        "review": _record(paths["review"]),
        "alpha": _record(paths["alpha"]),
        "approval": {
            "decision": "approved",
            "reviewer": review["reviewer"],
            "reviewed_at": review["reviewed_at"],
        },
    }


def authenticate_isnet() -> dict[str, Any]:
    for path, expected, label in (
        (ISNET_MODEL, ISNET_MODEL_SHA256, "ISNet model"),
        (ISNET_PROVENANCE, ISNET_PROVENANCE_SHA256, "ISNet provenance"),
    ):
        _regular_file(path, label)
        if sha256_file(path) != expected:
            raise GeometryReferenceError(f"{label} SHA-256 changed")
    if not ISNET_PYTHON.is_file() or not os.access(ISNET_PYTHON, os.X_OK):
        raise GeometryReferenceError("pinned ISNet Python is missing")
    return {
        "model": _record(ISNET_MODEL),
        "provenance": _record(ISNET_PROVENANCE),
        "python": str(ISNET_PYTHON),
        "use": "hat_candidate_alpha_only",
    }


def authenticate_model() -> dict[str, Any]:
    try:
        return flux2_base.authenticate_model_snapshot(
            model_root=MODEL_ROOT,
            revision=MODEL_REVISION,
            inventory_path=MODEL_INVENTORY,
            inventory_sha256=MODEL_INVENTORY_SHA256,
            expected_model_name="black-forest-labs/FLUX.2-klein-4B",
        )
    except ValueError as error:
        raise GeometryReferenceError(f"FLUX.2 snapshot authentication failed: {error}") from error


def _xy(point: Sequence[float]) -> tuple[int, int]:
    return (
        int(round(float(point[0]) * (WIDTH - 1))),
        int(round(float(point[1]) * (HEIGHT - 1))),
    )


def _polygon_mask(polygons: Sequence[Sequence[Sequence[float]]]) -> np.ndarray:
    image = Image.new("L", (WIDTH, HEIGHT), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon([_xy(point) for point in polygon], fill=255)
    return np.asarray(image, dtype=np.uint8) == 255


def build_edit_core(case: Mapping[str, Any], source_alpha: Image.Image) -> Image.Image:
    sex = str(case["sex"])
    geometry = str(case["geometry_attribute"])
    foreground = np.asarray(source_alpha.convert("L"), dtype=np.uint8) >= 128
    if geometry == "long_sleeve":
        if sex == "male":
            polygons = (
                ((0.268, 0.285), (0.337, 0.225), (0.405, 0.265), (0.339, 0.365), (0.167, 0.480), (0.137, 0.449)),
                ((0.732, 0.285), (0.663, 0.225), (0.595, 0.265), (0.661, 0.365), (0.833, 0.480), (0.863, 0.449)),
            )
        else:
            polygons = (
                ((0.282, 0.280), (0.342, 0.222), (0.407, 0.266), (0.340, 0.363), (0.169, 0.477), (0.142, 0.447)),
                ((0.718, 0.280), (0.658, 0.222), (0.593, 0.266), (0.660, 0.363), (0.831, 0.477), (0.858, 0.447)),
            )
        values = _polygon_mask(polygons)
    elif geometry == "shorts":
        roi = np.zeros((HEIGHT, WIDTH), dtype=bool)
        roi[int(0.495 * HEIGHT) : int(0.900 * HEIGHT), int(0.335 * WIDTH) : int(0.665 * WIDTH)] = True
        values = foreground & roi
        values = ndimage.binary_closing(values, iterations=2)
    elif geometry == "plain_baseball_cap_hat_compatible_hair":
        cap = (
            ((0.382, 0.132), (0.400, 0.087), (0.430, 0.052), (0.475, 0.036), (0.535, 0.040), (0.585, 0.070), (0.615, 0.120), (0.635, 0.132), (0.575, 0.150), (0.405, 0.147)),
        )
        values = _polygon_mask(cap)
        if sex == "female":
            hair = _polygon_mask(
                (
                    ((0.540, 0.105), (0.600, 0.105), (0.607, 0.245), (0.555, 0.265), (0.535, 0.205)),
                    ((0.395, 0.090), (0.438, 0.080), (0.435, 0.205), (0.397, 0.205)),
                )
            )
            face_guard = _polygon_mask(
                (((0.430, 0.105), (0.570, 0.105), (0.565, 0.224), (0.435, 0.224)),)
            )
            values |= hair & ~face_guard
    elif geometry == "thin_rectangular_glasses":
        image = Image.new("L", (WIDTH, HEIGHT), 0)
        draw = ImageDraw.Draw(image)
        center_y = 0.143 if sex == "male" else 0.137
        left = (0.438, center_y - 0.020, 0.495, center_y + 0.020)
        right = (0.505, center_y - 0.020, 0.562, center_y + 0.020)
        for rectangle in (left, right):
            draw.rounded_rectangle(
                (*_xy(rectangle[:2]), *_xy(rectangle[2:])),
                radius=5,
                fill=255,
            )
        draw.line((_xy((0.492, center_y)), _xy((0.508, center_y))), fill=255, width=10)
        draw.line((_xy((0.438, center_y)), _xy((0.416, center_y + 0.006))), fill=255, width=8)
        draw.line((_xy((0.562, center_y)), _xy((0.584, center_y + 0.006))), fill=255, width=8)
        values = np.asarray(image, dtype=np.uint8) == 255
    else:
        raise GeometryReferenceError(f"unsupported geometry attribute: {geometry}")
    if not np.any(values) or np.all(values):
        raise GeometryReferenceError(f"constructed mask is empty/full for {case['case_id']}")
    return Image.fromarray(np.where(values, 255, 0).astype(np.uint8), "L")


def transition_and_guard(core: Image.Image, radius: int = 8) -> tuple[Image.Image, Image.Image]:
    values = np.asarray(core.convert("L"), dtype=np.uint8) == 255
    expanded = ndimage.binary_dilation(values, iterations=radius)
    band = expanded & ~values
    guard = ~expanded
    if not np.all((values.astype(np.uint8) + band.astype(np.uint8) + guard.astype(np.uint8)) == 1):
        raise GeometryReferenceError("mask core/band/guard are not an exact partition")
    return (
        Image.fromarray(np.where(band, 255, 0).astype(np.uint8), "L"),
        Image.fromarray(np.where(guard, 255, 0).astype(np.uint8), "L"),
    )


def composite_candidate(
    source: Image.Image,
    generated: Image.Image,
    core: Image.Image,
    band: Image.Image,
) -> tuple[Image.Image, dict[str, Any]]:
    source_values = np.asarray(source.convert("RGB"), dtype=np.uint8)
    generated_values = np.asarray(generated.convert("RGB"), dtype=np.uint8)
    core_values = np.asarray(core.convert("L"), dtype=np.uint8) == 255
    band_values = np.asarray(band.convert("L"), dtype=np.uint8) == 255
    if source_values.shape != generated_values.shape or source_values.shape[:2] != core_values.shape:
        raise GeometryReferenceError("candidate/source/mask canvases differ")
    guard = ~(core_values | band_values)
    distance_core = ndimage.distance_transform_edt(~core_values)
    distance_guard = ndimage.distance_transform_edt(~guard)
    weights = np.zeros(core_values.shape, dtype=np.float64)
    weights[core_values] = 1.0
    weights[band_values] = distance_guard[band_values] / np.maximum(
        distance_core[band_values] + distance_guard[band_values], 1.0e-9
    )
    result = np.rint(
        source_values * (1.0 - weights[..., None]) + generated_values * weights[..., None]
    ).astype(np.uint8)
    delta = np.abs(result.astype(np.int16) - source_values.astype(np.int16))
    changed = np.any(delta > 0, axis=2)
    return Image.fromarray(result, "RGB"), {
        "outside_changed_pixels": int(np.count_nonzero(changed & guard)),
        "outside_max_abs_channel_delta": int(delta[guard].max()) if np.any(guard) else 0,
        "core_changed_pixels": int(np.count_nonzero(changed & core_values)),
        "core_changed_fraction": float(
            np.count_nonzero(changed & core_values) / max(1, np.count_nonzero(core_values))
        ),
        "transition_changed_pixels": int(np.count_nonzero(changed & band_values)),
    }


ISNET_SOURCE = """
import sys
from PIL import Image
from rembg import new_session, remove
source = Image.open(sys.argv[1]).convert('RGB')
session = new_session('isnet-general-use')
mask = remove(source, session=session, only_mask=True, post_process_mask=True)
mask.convert('L').save(sys.argv[2], format='PNG')
""".strip()


def predict_isnet_alpha(candidate: Image.Image) -> Image.Image:
    with tempfile.TemporaryDirectory(prefix="route2_v3_isnet_") as temporary_name:
        temporary = Path(temporary_name)
        input_path = temporary / "candidate.png"
        output_path = temporary / "alpha.png"
        candidate.save(input_path, format="PNG")
        environment = dict(os.environ)
        environment.update(
            {
                "U2NET_HOME": str(ISNET_MODEL.parent),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": "",
            }
        )
        result = subprocess.run(
            [str(ISNET_PYTHON), "-c", ISNET_SOURCE, str(input_path), str(output_path)],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not output_path.is_file():
            raise GeometryReferenceError(f"pinned ISNet alpha failed: {result.stderr}")
        with Image.open(output_path) as opened:
            alpha = opened.convert("L")
            alpha.load()
    if alpha.size != candidate.size or alpha.getextrema() == (0, 0):
        raise GeometryReferenceError("pinned ISNet alpha is empty or has wrong canvas")
    return alpha


def blend_alpha(
    source_alpha: Image.Image,
    predicted_alpha: Image.Image,
    core: Image.Image,
    band: Image.Image,
) -> tuple[Image.Image, dict[str, Any]]:
    source = np.asarray(source_alpha.convert("L"), dtype=np.uint8)
    predicted = np.asarray(predicted_alpha.convert("L"), dtype=np.uint8)
    core_values = np.asarray(core.convert("L"), dtype=np.uint8) == 255
    band_values = np.asarray(band.convert("L"), dtype=np.uint8) == 255
    guard = ~(core_values | band_values)
    distance_core = ndimage.distance_transform_edt(~core_values)
    distance_guard = ndimage.distance_transform_edt(~guard)
    weights = np.zeros(core_values.shape, dtype=np.float64)
    weights[core_values] = 1.0
    weights[band_values] = distance_guard[band_values] / np.maximum(
        distance_core[band_values] + distance_guard[band_values], 1.0e-9
    )
    result = np.rint(source * (1.0 - weights) + predicted * weights).astype(np.uint8)
    changed = result != source
    return Image.fromarray(result, "L"), {
        "outside_changed_pixels": int(np.count_nonzero(changed & guard)),
        "added_foreground_pixels": int(np.count_nonzero((source < 128) & (result >= 128))),
        "removed_foreground_pixels": int(np.count_nonzero((source >= 128) & (result < 128))),
    }


def evaluate_metrics(
    case: Mapping[str, Any],
    source: Image.Image,
    candidate: Image.Image,
    source_alpha: Image.Image,
    candidate_alpha: Image.Image,
    core: Image.Image,
    band: Image.Image,
    pixel_proof: Mapping[str, Any],
    alpha_proof: Mapping[str, Any],
) -> dict[str, Any]:
    source_rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)
    candidate_rgb = np.asarray(candidate.convert("RGB"), dtype=np.uint8)
    core_values = np.asarray(core.convert("L"), dtype=np.uint8) == 255
    band_values = np.asarray(band.convert("L"), dtype=np.uint8) == 255
    guard = ~(core_values | band_values)
    changed = np.any(source_rgb != candidate_rgb, axis=2)
    x = np.arange(WIDTH)[None, :]
    left = core_values & (x < WIDTH / 2)
    right = core_values & (x >= WIDTH / 2)

    def fraction(region: np.ndarray) -> float:
        return float(np.count_nonzero(changed & region) / max(1, np.count_nonzero(region)))

    source_a = np.asarray(source_alpha.convert("L"), dtype=np.uint8)
    candidate_a = np.asarray(candidate_alpha.convert("L"), dtype=np.uint8)
    source_rows = np.where(source_a >= 128)[0]
    candidate_rows = np.where(candidate_a >= 128)[0]
    foot_delta = abs(int(source_rows.max()) - int(candidate_rows.max()))
    geometry = str(case["geometry_attribute"])
    checks = {
        "outside_mask_rgb_exact": pixel_proof["outside_changed_pixels"] == 0
        and pixel_proof["outside_max_abs_channel_delta"] == 0,
        "outside_mask_alpha_exact": alpha_proof["outside_changed_pixels"] == 0,
        "target_core_changed": float(pixel_proof["core_changed_fraction"]) >= 0.03,
        "floor_contact_unchanged": foot_delta == 0,
        "non_target_guard_byte_identical": bool(
            np.array_equal(source_rgb[guard], candidate_rgb[guard])
        ),
    }
    metrics: dict[str, Any] = {
        "pixel_proof": dict(pixel_proof),
        "alpha_proof": dict(alpha_proof),
        "left_target_changed_fraction": fraction(left),
        "right_target_changed_fraction": fraction(right),
        "foot_contact_y_delta_px": foot_delta,
        "guard_pixels": int(np.count_nonzero(guard)),
    }
    if geometry in {"long_sleeve", "shorts", "thin_rectangular_glasses"}:
        threshold = 0.02 if geometry == "thin_rectangular_glasses" else 0.03
        checks["bilateral_target_changed"] = min(
            metrics["left_target_changed_fraction"], metrics["right_target_changed_fraction"]
        ) >= threshold
    if geometry == "plain_baseball_cap_hat_compatible_hair":
        checks["headwear_additive_silhouette"] = int(
            alpha_proof["added_foreground_pixels"]
        ) >= 10
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
    }


def _labeled_panel(image: Image.Image, label: str, size: tuple[int, int]) -> Image.Image:
    panel = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    box = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((8, 8, box[2] + 18, box[3] + 18), fill=(0, 0, 0))
    draw.text((13, 13), label, fill=(255, 255, 255), font=font)
    return panel


def make_contact_sheet(images: Sequence[tuple[str, Image.Image]]) -> Image.Image:
    panel_size = (288, 384)
    columns = 3
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * panel_size[0], rows * panel_size[1]), (32, 32, 32))
    for index, (label, image) in enumerate(images):
        panel = _labeled_panel(image, label, panel_size)
        canvas.paste(panel, ((index % columns) * panel_size[0], (index // columns) * panel_size[1]))
    return canvas


def _mask_overlay(source: Image.Image, core: Image.Image, band: Image.Image) -> Image.Image:
    base = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    core_values = np.asarray(core.convert("L"), dtype=np.uint8) == 255
    band_values = np.asarray(band.convert("L"), dtype=np.uint8) == 255
    base[core_values] = np.rint(base[core_values] * 0.55 + np.array([255, 0, 0]) * 0.45)
    base[band_values] = np.rint(base[band_values] * 0.55 + np.array([255, 196, 0]) * 0.45)
    return Image.fromarray(base.astype(np.uint8), "RGB")


def _difference(source: Image.Image, candidate: Image.Image) -> Image.Image:
    delta = np.abs(
        np.asarray(candidate.convert("RGB"), dtype=np.int16)
        - np.asarray(source.convert("RGB"), dtype=np.int16)
    )
    return Image.fromarray(np.clip(delta * 4, 0, 255).astype(np.uint8), "RGB")


def _write_image(path: Path, image: Image.Image) -> None:
    image.save(path, format="PNG")
    with Image.open(path) as opened:
        opened.verify()


def prepare() -> Path:
    output = OUTPUT_ROOT.absolute()
    parent = output.parent
    if parent.is_symlink() or not parent.is_dir() or parent.resolve() != parent:
        raise GeometryReferenceError("v3 output parent must be a direct real directory")
    if os.path.lexists(output):
        raise FileExistsError(output)
    sources = {sex: authenticate_source(sex) for sex in ("male", "female")}
    model = authenticate_model()
    isnet = authenticate_isnet()
    runner = _record(RUNNER_PATH)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".staging", dir=parent))
    try:
        (staging / "masks").mkdir()
        (staging / "cases").mkdir()
        (staging / "failures").mkdir()
        cases = []
        for case in CASE_SPECS:
            source = sources[str(case["sex"])]
            with Image.open(source["image"]["path"]) as opened:
                image = opened.convert("RGB")
            with Image.open(source["alpha"]["path"]) as opened:
                alpha = opened.convert("L")
            core = build_edit_core(case, alpha)
            band, guard = transition_and_guard(core)
            mask_dir = staging / "masks" / str(case["case_id"])
            mask_dir.mkdir()
            public_dir = output / "masks" / str(case["case_id"])
            mask_images = {
                "edit_core.png": core,
                "transition_band.png": band,
                "protected_guard.png": guard,
                "overlay.png": _mask_overlay(image, core, band),
            }
            for filename, mask_image in mask_images.items():
                _write_image(mask_dir / filename, mask_image)
            assets = {
                filename: _record(mask_dir / filename, public_path=public_dir / filename)
                for filename in sorted(mask_images)
            }
            mask_manifest_payload = {
                "schema": "route2_controlled_geometry_mask_bundle_v3",
                "case_id": case["case_id"],
                "base_asset_id": case["base_asset_id"],
                "geometry_attribute": case["geometry_attribute"],
                "source_image": source["image"],
                "source_alpha": source["alpha"],
                "construction_version": case["mask_version"],
                "assets": assets,
                "metrics": {
                    "core_pixels": int(np.count_nonzero(np.asarray(core) == 255)),
                    "transition_pixels": int(np.count_nonzero(np.asarray(band) == 255)),
                    "protected_pixels": int(np.count_nonzero(np.asarray(guard) == 255)),
                    "exact_partition": True,
                },
                "review_state": "agent_constructed_preflight_mask",
                "user_acceptance": "not_claimed",
            }
            mask_manifest = mask_dir / "mask_manifest.json"
            mask_manifest.write_bytes(_json_bytes(mask_manifest_payload))
            cases.append(
                {
                    **case,
                    "source": source,
                    "mask_manifest": _record(
                        mask_manifest,
                        public_path=public_dir / "mask_manifest.json",
                    ),
                    "mask_assets": assets,
                    "inference": {
                        "width": WIDTH,
                        "height": HEIGHT,
                        "steps": STEPS,
                        "guidance_scale": GUIDANCE_SCALE,
                        "max_sequence_length": MAX_SEQUENCE_LENGTH,
                        "local_files_only": True,
                    },
                }
            )
        payload = {
            "schema": SCHEMA,
            "state_classification": "research_candidate_preflight",
            "formal_base_qa_required": False,
            "formal_base_qa_satisfied": False,
            "purpose": "one-time controlled geometry template canaries",
            "ordinary_color_instances_use_flux2": False,
            "prohibited_models": ["Hunyuan3D", "Qwen-Image", "FLUX.1", "other_image_models"],
            "output_root": str(output),
            "created_at_utc": _utc_now(),
            "runner": runner,
            "model": model,
            "isnet": isnet,
            "sources": sources,
            "cases": cases,
        }
        if "user_approved" in json.dumps(payload) or "hunyuan3d" in json.dumps(payload).lower():
            # Hunyuan3D is permitted only in the explicit prohibited_models value.
            if payload["prohibited_models"][0] != "Hunyuan3D":
                raise GeometryReferenceError("v3 contract may not claim approval or Hunyuan execution")
        contract = staging / "geometry_jobs_v3.json"
        contract.write_bytes(_json_bytes(payload))
        _readonly_tree(staging)
        _fsync_tree(staging)
        _rename_noreplace(staging, output)
        return output / "geometry_jobs_v3.json"
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def load_contract() -> dict[str, Any]:
    path = _regular_file(OUTPUT_ROOT / "geometry_jobs_v3.json", "v3 jobs contract", mode=0o444)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GeometryReferenceError(f"v3 jobs contract is invalid: {error}") from error
    if (
        payload.get("schema") != SCHEMA
        or payload.get("state_classification") != "research_candidate_preflight"
        or payload.get("formal_base_qa_required") is not False
        or payload.get("formal_base_qa_satisfied") is not False
        or payload.get("ordinary_color_instances_use_flux2") is not False
        or payload.get("output_root") != str(OUTPUT_ROOT)
        or payload.get("runner") != _record(RUNNER_PATH)
        or not isinstance(payload.get("cases"), list)
        or [item.get("case_id") for item in payload["cases"]] != list(CASE_BY_ID)
    ):
        raise GeometryReferenceError("v3 jobs contract schema, state, runner, or case order changed")
    for sex in ("male", "female"):
        if payload.get("sources", {}).get(sex) != authenticate_source(sex):
            raise GeometryReferenceError(f"{sex} source changed after v3 preparation")
    model = payload.get("model")
    snapshot = MODEL_ROOT / "snapshots" / MODEL_REVISION
    if (
        not isinstance(model, Mapping)
        or model.get("revision") != MODEL_REVISION
        or model.get("inventory", {}).get("sha256") != MODEL_INVENTORY_SHA256
        or model.get("snapshot") != str(snapshot)
        or sha256_file(MODEL_INVENTORY) != MODEL_INVENTORY_SHA256
        or not snapshot.is_dir()
        or any(path.name.endswith(".incomplete") for path in MODEL_ROOT.rglob("*"))
    ):
        raise GeometryReferenceError("pinned FLUX.2 snapshot changed after v3 preparation")
    if payload.get("isnet") != authenticate_isnet():
        raise GeometryReferenceError("pinned ISNet snapshot changed after v3 preparation")
    for case in payload["cases"]:
        mask_manifest = Path(case["mask_manifest"]["path"])
        _require_record(mask_manifest, case["mask_manifest"], "v3 mask manifest")
        for filename, record in case["mask_assets"].items():
            _require_record(mask_manifest.parent / filename, record, f"v3 mask {filename}")
    return payload


def _pipeline(gpu: str):
    if gpu not in {"0", "1", "2", "3"}:
        raise GeometryReferenceError("GPU must be one of 0,1,2,3")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in (None, "", gpu):
        raise GeometryReferenceError(f"CUDA_VISIBLE_DEVICES conflicts with --gpu {gpu}")
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    import torch
    from diffusers import Flux2KleinPipeline

    pipeline = Flux2KleinPipeline.from_pretrained(
        str(MODEL_ROOT / "snapshots" / MODEL_REVISION),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    return pipeline.to("cuda")


def _inference(case: Mapping[str, Any], source: Image.Image, pipeline: Any) -> Image.Image:
    import torch

    generator = torch.Generator("cuda").manual_seed(int(case["seed"]))
    prompt = (
        f"{case['prompt']} Preserve all protected pixels and semantics outside the requested region. "
        f"Avoid: {case['negative_prompt']}."
    )
    result = pipeline(
        image=source,
        prompt=prompt,
        width=WIDTH,
        height=HEIGHT,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )
    if not getattr(result, "images", None):
        raise GeometryReferenceError("FLUX.2 returned no image")
    image = result.images[0].convert("RGB")
    if image.size != (WIDTH, HEIGHT):
        raise GeometryReferenceError("FLUX.2 output canvas changed")
    return image


def _case_snapshot(contract: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    matches = [item for item in contract["cases"] if item["case_id"] == case_id]
    if len(matches) != 1:
        raise GeometryReferenceError(f"case is not unique in contract: {case_id}")
    case = dict(matches[0])
    if case_id not in CASE_BY_ID or any(
        case.get(key) != CASE_BY_ID[case_id].get(key)
        for key in (
            "case_id",
            "sex",
            "base_asset_id",
            "geometry_attribute",
            "seed",
            "prompt",
            "negative_prompt",
            "mask_version",
            "target_color_policy",
        )
    ):
        raise GeometryReferenceError(f"case spec changed: {case_id}")
    return case


def _publish_failure(case: Mapping[str, Any], staging: Path, error: BaseException) -> Path:
    root = OUTPUT_ROOT / "failures"
    destination = root / f"{case['case_id']}.{uuid.uuid4().hex}"
    payload = {
        "schema": "route2_controlled_geometry_generation_failure_v1",
        "case_id": case["case_id"],
        "state_classification": "rejected",
        "formal_base_qa_satisfied": False,
        "error": {"type": type(error).__name__, "message": str(error)},
        "recorded_at_utc": _utc_now(),
    }
    (staging / "failure.json").write_bytes(_json_bytes(payload))
    _readonly_tree(staging)
    _fsync_tree(staging)
    _rename_noreplace(staging, destination)
    return destination


def generate_case(contract: Mapping[str, Any], case_id: str, pipeline: Any, gpu: str) -> dict[str, Any]:
    case = _case_snapshot(contract, case_id)
    destination = OUTPUT_ROOT / "cases" / case_id
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{case_id}.", suffix=".staging", dir=OUTPUT_ROOT / "cases")
    )
    try:
        before = load_contract()
        if _case_snapshot(before, case_id) != case:
            raise GeometryReferenceError("case contract changed before inference")
        source_record = case["source"]
        with Image.open(source_record["image"]["path"]) as opened:
            source = opened.convert("RGB")
        with Image.open(source_record["alpha"]["path"]) as opened:
            source_alpha = opened.convert("L")
        mask_dir = Path(case["mask_manifest"]["path"]).parent
        with Image.open(mask_dir / "edit_core.png") as opened:
            core = opened.convert("L")
        with Image.open(mask_dir / "transition_band.png") as opened:
            band = opened.convert("L")
        raw = _inference(case, source, pipeline)
        candidate, pixel_proof = composite_candidate(source, raw, core, band)
        if case["geometry_attribute"] == "plain_baseball_cap_hat_compatible_hair":
            predicted_alpha = predict_isnet_alpha(candidate)
            candidate_alpha, alpha_proof = blend_alpha(
                source_alpha, predicted_alpha, core, band
            )
        else:
            predicted_alpha = source_alpha.copy()
            candidate_alpha = source_alpha.copy()
            alpha_proof = {
                "outside_changed_pixels": 0,
                "added_foreground_pixels": 0,
                "removed_foreground_pixels": 0,
                "source_alpha_preserved_by_geometry_contract": True,
            }
        metrics = evaluate_metrics(
            case,
            source,
            candidate,
            source_alpha,
            candidate_alpha,
            core,
            band,
            pixel_proof,
            alpha_proof,
        )
        rgba = candidate.convert("RGBA")
        rgba.putalpha(candidate_alpha)
        overlay = _mask_overlay(source, core, band)
        difference = _difference(source, candidate)
        images = {
            "source.png": source,
            "source_alpha.png": source_alpha,
            "raw_candidate.png": raw,
            "candidate.png": candidate,
            "candidate_alpha.png": candidate_alpha,
            "candidate_rgba.png": rgba,
            "mask_overlay.png": overlay,
            "difference.png": difference,
        }
        for filename, image in images.items():
            _write_image(staging / filename, image)
        contact = make_contact_sheet(
            (
                ("approved source", source),
                ("raw FLUX.2", raw),
                ("masked candidate", candidate),
                ("authorized mask", overlay),
                ("4x difference", difference),
                ("candidate RGBA", rgba),
            )
        )
        _write_image(staging / "contact_sheet.png", contact)
        public = destination
        artifacts = {
            filename: _record(staging / filename, public_path=public / filename)
            for filename in sorted((*images, "contact_sheet.png"))
        }
        after = load_contract()
        if _case_snapshot(after, case_id) != case:
            raise GeometryReferenceError("case contract changed after inference")
        manifest = {
            "schema": CANDIDATE_SCHEMA,
            "case_id": case_id,
            "base_asset_id": case["base_asset_id"],
            "geometry_attribute": case["geometry_attribute"],
            "state_classification": "research_candidate",
            "generation_status": "generated_pending_agent_2d_qa",
            "formal_base_qa_satisfied": False,
            "user_acceptance": "not_claimed",
            "created_at_utc": _utc_now(),
            "jobs_contract": _record(OUTPUT_ROOT / "geometry_jobs_v3.json"),
            "runner": _record(RUNNER_PATH),
            "model": {
                "name": "black-forest-labs/FLUX.2-klein-4B",
                "revision": MODEL_REVISION,
                "inventory": _record(MODEL_INVENTORY),
                "local_files_only": True,
            },
            "source": source_record,
            "mask_manifest": case["mask_manifest"],
            "mask_assets": case["mask_assets"],
            "parameters": {
                "prompt": case["prompt"],
                "negative_prompt": case["negative_prompt"],
                "seed": case["seed"],
                "width": WIDTH,
                "height": HEIGHT,
                "steps": STEPS,
                "guidance_scale": GUIDANCE_SCALE,
                "max_sequence_length": MAX_SEQUENCE_LENGTH,
                "physical_gpu": gpu,
            },
            "metrics": metrics,
            "automatic_2d_gate": "passed" if metrics["passed"] else "rejected",
            "artifacts": artifacts,
        }
        if "user_approved" in json.dumps(manifest):
            raise GeometryReferenceError("candidate may not claim user approval")
        (staging / "candidate_manifest.json").write_bytes(_json_bytes(manifest))
        _readonly_tree(staging)
        _fsync_tree(staging)
        _rename_noreplace(staging, destination)
        return {
            "case_id": case_id,
            "status": "generated" if metrics["passed"] else "automatic_2d_rejected",
            "destination": str(destination),
            "manifest": str(destination / "candidate_manifest.json"),
        }
    except BaseException as error:
        if staging.exists():
            evidence = _publish_failure(case, staging, error)
        else:
            evidence = None
        if not isinstance(error, Exception):
            raise
        return {
            "case_id": case_id,
            "status": "generation_failure_rejected",
            "evidence": str(evidence) if evidence is not None else None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }


def generate(case_ids: Sequence[str], gpu: str) -> list[dict[str, Any]]:
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise GeometryReferenceError("--case-id must be nonempty and unique")
    unknown = sorted(set(case_ids) - set(CASE_BY_ID))
    if unknown:
        raise GeometryReferenceError(f"unknown cases: {unknown}")
    contract = load_contract()
    pipeline = _pipeline(gpu)
    return [generate_case(contract, case_id, pipeline, gpu) for case_id in case_ids]


def _load_candidate(case_id: str) -> tuple[Path, dict[str, Any]]:
    root = OUTPUT_ROOT / "cases" / case_id
    manifest_path = _regular_file(root / "candidate_manifest.json", "candidate manifest", mode=0o444)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GeometryReferenceError(f"candidate manifest is invalid: {error}") from error
    if (
        manifest.get("schema") != CANDIDATE_SCHEMA
        or manifest.get("case_id") != case_id
        or manifest.get("formal_base_qa_satisfied") is not False
        or manifest.get("user_acceptance") != "not_claimed"
        or manifest.get("runner") != _record(RUNNER_PATH)
        or manifest.get("jobs_contract") != _record(OUTPUT_ROOT / "geometry_jobs_v3.json")
        or not isinstance(manifest.get("artifacts"), Mapping)
    ):
        raise GeometryReferenceError("candidate schema, state, or producer changed")
    for filename, record in manifest["artifacts"].items():
        _require_record(root / filename, record, f"candidate artifact {filename}")
    return root, manifest


def review(case_id: str, status: str, notes: str) -> Path:
    if case_id not in CASE_BY_ID:
        raise GeometryReferenceError(f"unknown case: {case_id}")
    if status not in {"agent_2d_passed", "rejected"}:
        raise GeometryReferenceError("review status must be agent_2d_passed or rejected")
    if not notes.strip():
        raise GeometryReferenceError("review notes must be nonempty")
    root, manifest = _load_candidate(case_id)
    destination = root / "agent_2d_visual_qa.json"
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    automatic_pass = manifest.get("automatic_2d_gate") == "passed"
    if status == "agent_2d_passed" and not automatic_pass:
        raise GeometryReferenceError("agent cannot pass a candidate rejected by automatic metrics")
    checks = {
        "target_geometry_present": status == "agent_2d_passed",
        "identity_face_preserved": status == "agent_2d_passed",
        "pose_camera_limb_gaps_preserved": status == "agent_2d_passed",
        "non_target_regions_preserved": status == "agent_2d_passed",
        "target_mask_boundary_acceptable": status == "agent_2d_passed",
        "pixal_bindable_silhouette_plausible": status == "agent_2d_passed",
    }
    payload = {
        "schema": DECISION_SCHEMA,
        "case_id": case_id,
        "status": status,
        "state_classification": "research_candidate" if status == "agent_2d_passed" else "rejected",
        "reviewer_kind": "agent",
        "reviewer": "codex_female_route2_base",
        "notes": notes.strip(),
        "checks": checks,
        "candidate_manifest": _record(root / "candidate_manifest.json"),
        "contact_sheet": manifest["artifacts"]["contact_sheet.png"],
        "metrics": manifest["metrics"],
        "pixal_authorized": status == "agent_2d_passed",
        "formal_dataset_registration_authorized": False,
        "user_acceptance": "not_claimed",
        "reviewed_at_utc": _utc_now(),
    }
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, _json_bytes(payload))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return destination


def _load_decision(case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    root, candidate = _load_candidate(case_id)
    path = _regular_file(root / "agent_2d_visual_qa.json", "agent 2D decision", mode=0o444)
    try:
        decision = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GeometryReferenceError(f"agent decision is invalid: {error}") from error
    if (
        decision.get("schema") != DECISION_SCHEMA
        or decision.get("case_id") != case_id
        or decision.get("candidate_manifest") != _record(root / "candidate_manifest.json")
        or decision.get("status") not in {"agent_2d_passed", "rejected"}
        or decision.get("formal_dataset_registration_authorized") is not False
        or decision.get("user_acceptance") != "not_claimed"
    ):
        raise GeometryReferenceError("agent decision lineage/state changed")
    return candidate, decision


def finalize() -> Path:
    destination = OUTPUT_ROOT / "review_summary_v1"
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    records = []
    panels = []
    pixal_jobs = []
    for case_id in CASE_BY_ID:
        candidate, decision = _load_decision(case_id)
        case_root = OUTPUT_ROOT / "cases" / case_id
        with Image.open(case_root / "contact_sheet.png") as opened:
            panels.append((case_id, opened.convert("RGB")))
        records.append(
            {
                "case_id": case_id,
                "candidate_manifest": _record(case_root / "candidate_manifest.json"),
                "decision": _record(case_root / "agent_2d_visual_qa.json"),
                "status": decision["status"],
            }
        )
        if decision["status"] == "agent_2d_passed":
            rgba = candidate["artifacts"]["candidate_rgba.png"]
            pixal_jobs.append(
                {
                    "asset_id": f"route2_v3_{case_id}",
                    "base_asset_id": candidate["base_asset_id"],
                    "geometry_attribute": candidate["geometry_attribute"],
                    "state_classification": "research_candidate",
                    "input_rgba": rgba,
                    "reference_manifest": _record(case_root / "candidate_manifest.json"),
                    "reference_decision": _record(case_root / "agent_2d_visual_qa.json"),
                    "model": {"name": "TencentARC/Pixal3D", "revision": PIXAL_REVISION},
                    "parameters": {
                        "seed": 42,
                        "manual_fov": 0.2,
                        "resolution": 1024,
                        "low_vram": True,
                    },
                    "output_dir": str(
                        SPEAR_ROOT
                        / "tmp/i23d_controlled_geometry_v3/pixal3d"
                        / f"route2_v3_{case_id}"
                    ),
                    "execution_status": "ready_for_pixal_preflight_not_formal_registration",
                }
            )
    staging = Path(
        tempfile.mkdtemp(prefix=".review_summary_v1.", suffix=".staging", dir=OUTPUT_ROOT)
    )
    try:
        all_sheet = make_contact_sheet(panels)
        _write_image(staging / "all_cases_contact_sheet.png", all_sheet)
        pixal_payload = {
            "schema": PIXAL_JOBS_SCHEMA,
            "state_classification": "research_candidate_preflight",
            "formal_registration_authorized": False,
            "source_jobs_contract": _record(OUTPUT_ROOT / "geometry_jobs_v3.json"),
            "jobs": pixal_jobs,
        }
        (staging / "pixal_jobs_v1.json").write_bytes(_json_bytes(pixal_payload))
        summary = {
            "schema": "route2_controlled_geometry_reference_summary_v1",
            "state_classification": "research_candidate_preflight",
            "formal_registration_authorized": False,
            "case_count": len(records),
            "passed_count": sum(item["status"] == "agent_2d_passed" for item in records),
            "rejected_count": sum(item["status"] == "rejected" for item in records),
            "cases": records,
            "pixal_job_count": len(pixal_jobs),
            "created_at_utc": _utc_now(),
        }
        (staging / "summary.json").write_bytes(_json_bytes(summary))
        _readonly_tree(staging)
        _fsync_tree(staging)
        _rename_noreplace(staging, destination)
        return destination / "summary.json"
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--case-id", action="append", required=True)
    generate_parser.add_argument("--gpu", choices=("0", "1", "2", "3"), required=True)
    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("--case-id", choices=tuple(CASE_BY_ID), required=True)
    review_parser.add_argument(
        "--status", choices=("agent_2d_passed", "rejected"), required=True
    )
    review_parser.add_argument("--notes", required=True)
    subparsers.add_parser("finalize")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "prepare":
        print(f"ROUTE2_CONTROLLED_GEOMETRY_PREPARED {prepare()}")
    elif args.command == "generate":
        print(json.dumps(generate(args.case_id, args.gpu), indent=2, sort_keys=True))
    elif args.command == "review":
        print(f"ROUTE2_CONTROLLED_GEOMETRY_REVIEWED {review(args.case_id, args.status, args.notes)}")
    elif args.command == "finalize":
        print(f"ROUTE2_CONTROLLED_GEOMETRY_FINALIZED {finalize()}")
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
