"""Authenticated TRELLIS/Pixal inputs for stable Rocketbox template fitting."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


I23D_USAGE_SCOPE = "noncommercial_research_dataset_candidate"
EXPECTED_ASSET_IDS = (
    "rocketbox_male_adult_01",
    "rocketbox_female_adult_01",
)
DINO_REVISION = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
BACKEND_CONTRACTS = {
    "trellis2": {
        "front_axis": "negative-y",
        "model_revision": "af44b45f2e35a493886929c6d786e563ec68364d",
        "dino_revision": DINO_REVISION,
    },
    "pixal3d": {
        "front_axis": "positive-y",
        "model_revision": "0b31f9160aa400719af409098bff7936a932f726",
        "dino_revision": DINO_REVISION,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, description: str) -> Path:
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError(f"{description} must be a direct regular file: {path}")
    if path.stat().st_size <= 0:
        raise ValueError(f"{description} is empty: {path}")
    return path


def _mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} is missing or is not an object")
    return value


def _manifest_path(value: Any, expected: Path, description: str) -> None:
    if not isinstance(value, str) or Path(value).absolute() != expected:
        raise ValueError(f"{description} path does not match the authenticated file")


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }

def validate_i23d_manifest(
    manifest: Mapping[str, Any],
    *,
    asset_id: str,
    backend: str,
    front_axis: str,
    glb_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    """Validate one fixed-input 1024 bake-off output and normalize provenance."""
    if asset_id not in EXPECTED_ASSET_IDS:
        raise ValueError(f"unsupported Rocketbox asset id: {asset_id}")
    contract = BACKEND_CONTRACTS.get(backend)
    if contract is None:
        raise ValueError(f"unsupported I23D backend: {backend}")
    if manifest.get("backend") != backend:
        raise ValueError("I23D manifest backend does not match the requested backend")
    if front_axis != contract["front_axis"]:
        raise ValueError(
            f"{backend} front axis must be {contract['front_axis']}, got {front_axis}"
        )

    glb_path = _regular_file(glb_path, "guide GLB")
    reference_path = _regular_file(reference_path, "reference RGBA")
    output = _mapping(manifest.get("output"), "I23D output descriptor")
    input_record = _mapping(manifest.get("input"), "I23D input descriptor")
    model = _mapping(manifest.get("model"), "I23D model descriptor")
    dino = _mapping(manifest.get("dino"), "I23D DINO descriptor")
    parameters = _mapping(manifest.get("parameters"), "I23D parameters")

    _manifest_path(output.get("path"), glb_path, "guide GLB")
    if output.get("bytes") != glb_path.stat().st_size:
        raise ValueError("guide GLB size does not match its manifest")
    glb_sha256 = sha256_file(glb_path)
    if output.get("sha256") != glb_sha256:
        raise ValueError("guide GLB SHA-256 does not match its manifest")

    _manifest_path(input_record.get("path"), reference_path, "reference RGBA")
    if input_record.get("sha256") != sha256_file(reference_path):
        raise ValueError("reference RGBA SHA-256 does not match its manifest")
    if input_record.get("mode") != "RGBA":
        raise ValueError("reference input mode must be RGBA")
    if input_record.get("alpha_min") == 255 or input_record.get("alpha_max") == 0:
        raise ValueError("reference RGBA must contain transparent and visible pixels")

    if model.get("revision") != contract["model_revision"]:
        raise ValueError(f"{backend} model revision is not pinned")
    if dino.get("revision") != contract["dino_revision"]:
        raise ValueError("DINO revision is not pinned")
    if parameters.get("seed") != 42:
        raise ValueError("I23D seed must be 42")
    if parameters.get("resolution") != 1024:
        raise ValueError("I23D resolution must be the reviewed 1024 canary")

    return {
        "asset_id": asset_id,
        "backend": backend,
        "front_axis": front_axis,
        "canonical_front_axis": "negative-y",
        "usage_scope": I23D_USAGE_SCOPE,
        "research_release_ok": True,
        "permissive_commercial_ok": False,
        "guide_glb": _file_record(glb_path),
        "reference_rgba": _file_record(reference_path),
        "model": dict(model),
        "dino": dict(dino),
        "parameters": dict(parameters),
    }
