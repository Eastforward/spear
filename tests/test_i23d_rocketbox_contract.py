from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


from tools.i23d_rocketbox_contract import (  # type: ignore[import-not-found]
    BACKEND_CONTRACTS,
    I23D_USAGE_SCOPE,
    validate_i23d_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(
    *,
    asset_id: str,
    backend: str,
    glb_path: Path,
    reference_path: Path,
) -> dict:
    contract = BACKEND_CONTRACTS[backend]
    return {
        "backend": backend,
        "input": {
            "alpha_max": 255,
            "alpha_min": 0,
            "mode": "RGBA",
            "path": str(reference_path),
            "sha256": _sha256(reference_path),
            "size": [1152, 1536],
        },
        "model": {
            "revision": contract["model_revision"],
            "snapshot": f"/data/models/{backend}/{contract['model_revision']}",
        },
        "dino": {
            "revision": contract["dino_revision"],
            "snapshot": f"/data/models/dino/{contract['dino_revision']}",
        },
        "output": {
            "bytes": glb_path.stat().st_size,
            "path": str(glb_path),
            "sha256": _sha256(glb_path),
        },
        "parameters": {
            "low_vram": True,
            "manual_fov": 0.2,
            "resolution": 1024,
            "seed": 42,
        },
    }


@pytest.mark.parametrize(
    ("backend", "front_axis"),
    (("trellis2", "negative-y"), ("pixal3d", "positive-y")),
)
def test_validate_i23d_manifest_pins_files_model_and_license_scope(
    tmp_path: Path,
    backend: str,
    front_axis: str,
):
    asset_id = "rocketbox_male_adult_01"
    glb_path = tmp_path / "canary_1024_seed42.glb"
    reference_path = tmp_path / "input_rgba_isnet.png"
    glb_path.write_bytes(f"{backend}:mesh".encode("ascii"))
    reference_path.write_bytes(f"{asset_id}:rgba".encode("ascii"))
    manifest = _manifest(
        asset_id=asset_id,
        backend=backend,
        glb_path=glb_path,
        reference_path=reference_path,
    )

    provenance = validate_i23d_manifest(
        manifest,
        asset_id=asset_id,
        backend=backend,
        front_axis=front_axis,
        glb_path=glb_path,
        reference_path=reference_path,
    )

    assert provenance["backend"] == backend
    assert provenance["front_axis"] == front_axis
    assert provenance["canonical_front_axis"] == "negative-y"
    assert provenance["usage_scope"] == I23D_USAGE_SCOPE
    assert provenance["research_release_ok"] is True
    assert provenance["permissive_commercial_ok"] is False
    assert provenance["asset_id"] == asset_id
    assert provenance["guide_glb"]["sha256"] == _sha256(glb_path)
    assert provenance["reference_rgba"]["sha256"] == _sha256(reference_path)


def test_validate_i23d_manifest_rejects_wrong_front_axis(tmp_path: Path):
    glb_path = tmp_path / "candidate.glb"
    reference_path = tmp_path / "reference.png"
    glb_path.write_bytes(b"mesh")
    reference_path.write_bytes(b"rgba")
    manifest = _manifest(
        asset_id="rocketbox_male_adult_01",
        backend="pixal3d",
        glb_path=glb_path,
        reference_path=reference_path,
    )

    with pytest.raises(ValueError, match="front axis"):
        validate_i23d_manifest(
            manifest,
            asset_id="rocketbox_male_adult_01",
            backend="pixal3d",
            front_axis="negative-y",
            glb_path=glb_path,
            reference_path=reference_path,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload["output"].update(sha256="0" * 64), "guide GLB"),
        (lambda payload: payload["input"].update(sha256="0" * 64), "reference RGBA"),
        (lambda payload: payload["parameters"].update(seed=7), "seed"),
        (
            lambda payload: payload["model"].update(revision="unpinned"),
            "model revision",
        ),
    ),
)
def test_validate_i23d_manifest_rejects_stale_or_unpinned_inputs(
    tmp_path: Path,
    mutation,
    message: str,
):
    glb_path = tmp_path / "candidate.glb"
    reference_path = tmp_path / "reference.png"
    glb_path.write_bytes(b"mesh")
    reference_path.write_bytes(b"rgba")
    manifest = _manifest(
        asset_id="rocketbox_female_adult_01",
        backend="trellis2",
        glb_path=glb_path,
        reference_path=reference_path,
    )
    mutation(manifest)

    with pytest.raises(ValueError, match=message):
        validate_i23d_manifest(
            manifest,
            asset_id="rocketbox_female_adult_01",
            backend="trellis2",
            front_axis="negative-y",
            glb_path=glb_path,
            reference_path=reference_path,
        )


def test_real_bakeoff_manifests_match_the_pinned_contract():
    root = Path(__file__).resolve().parents[1] / "tmp" / "i23d_human_bakeoff_v1"
    for backend, front_axis in (("trellis2", "negative-y"), ("pixal3d", "positive-y")):
        for asset_id in ("rocketbox_male_adult_01", "rocketbox_female_adult_01"):
            asset_dir = root / backend / asset_id
            glb_path = asset_dir / "canary_1024_seed42.glb"
            manifest_path = asset_dir / "canary_1024_seed42.manifest.json"
            reference_path = root / "inputs" / asset_id / "input_rgba_isnet.png"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            result = validate_i23d_manifest(
                manifest,
                asset_id=asset_id,
                backend=backend,
                front_axis=front_axis,
                glb_path=glb_path,
                reference_path=reference_path,
            )

            assert result["guide_glb"]["size_bytes"] == glb_path.stat().st_size
