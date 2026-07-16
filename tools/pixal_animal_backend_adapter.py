#!/usr/bin/env python3
"""Thin Pixal3D adapter for the existing animal orientation/rig/UE pipeline."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SPEAR_ROOT = Path(__file__).resolve().parents[1]
if __package__ in {None, ""}:
    sys.path.insert(0, str(SPEAR_ROOT))
PIXAL_PYTHON = Path(
    "/data/jzy/miniconda3/envs/avengine-3dgen/bin/python3.10"
)
PIXAL_WRAPPER = SPEAR_ROOT / "tools/i23d_human_bakeoff.py"
PIXAL_REVISION = "0b31f9160aa400719af409098bff7936a932f726"
DINO_REVISION = "3c276edd87d6f6e569ff0c4400e086807d0f3881"
FLUX1_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
PIXAL_PARAMETERS = {
    "low_vram": True,
    "manual_fov": 0.2,
    "resolution": 1024,
}


class PixalAnimalAdapterError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, *, public_path: Path | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise PixalAnimalAdapterError(f"artifact must be a direct nonempty file: {path}")
    return {
        "path": str(Path(public_path).resolve() if public_path else path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _load_json(path: Path, description: str) -> dict[str, Any]:
    path = Path(path).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PixalAnimalAdapterError(f"invalid {description}: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise PixalAnimalAdapterError(f"{description} must be a JSON object")
    return payload


def _manifest_path(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return str(path.relative_to(SPEAR_ROOT))
    except ValueError:
        return str(path)


def build_pixal_command(
    *,
    reference: Path,
    output: Path,
    gpu: int,
    seed: int,
) -> list[str]:
    if int(gpu) < 0 or int(seed) < 0:
        raise ValueError("Pixal animal gpu and seed must be non-negative")
    return [
        str(PIXAL_PYTHON),
        str(PIXAL_WRAPPER),
        "--backend",
        "pixal3d",
        "--image",
        str(Path(reference).resolve()),
        "--output",
        str(Path(output).resolve()),
        "--gpu",
        str(int(gpu)),
        "--seed",
        str(int(seed)),
        "--resolution",
        str(PIXAL_PARAMETERS["resolution"]),
        "--manual-fov",
        str(PIXAL_PARAMETERS["manual_fov"]),
        "--low-vram",
    ]


def validate_reference_lineage(
    old_candidate_manifest: Path,
    reference: Path,
) -> dict[str, Any]:
    old_candidate_manifest = Path(old_candidate_manifest).resolve()
    reference = Path(reference).resolve()
    payload = _load_json(old_candidate_manifest, "historical animal candidate")
    generation = payload.get("generation", {})
    visual = payload.get("visual_assets", {})
    recorded_reference = Path(str(visual.get("reference_image", "")))
    if not recorded_reference.is_absolute():
        recorded_reference = SPEAR_ROOT / recorded_reference
    if (
        payload.get("schema_version") != "source_asset_v1"
        or payload.get("asset_class") != "animal"
        or generation.get("source_pipeline") != "flux+hunyuan3d"
        or generation.get("model") != "flux_dev+hunyuan3d-2.1"
        or not isinstance(generation.get("seed"), int)
        or recorded_reference.resolve() != reference
    ):
        raise PixalAnimalAdapterError(
            "historical animal reference does not have the expected FLUX.1-dev lineage"
        )
    return {
        "generator": "black-forest-labs/FLUX.1-dev",
        "revision": FLUX1_REVISION,
        "seed": generation["seed"],
        "positive_prompt": generation.get("positive_prompt"),
        "negative_prompt": generation.get("negative_prompt"),
        "created_at": generation.get("created_at"),
        "historical_manifest": _record(old_candidate_manifest),
        "reference": _record(reference),
        "provenance_resolution": (
            "The historical label flux_dev resolves to black-forest-labs/"
            f"FLUX.1-dev at the local pinned snapshot {FLUX1_REVISION}."
        ),
        "hunyuan_outputs_reused": False,
    }


def validate_generated_pixal(
    *,
    mesh: Path,
    generated_manifest: Path,
    reference: Path,
    seed: int,
) -> dict[str, Any]:
    mesh = Path(mesh).resolve()
    generated_manifest = Path(generated_manifest).resolve()
    reference = Path(reference).resolve()
    generated = _load_json(generated_manifest, "Pixal generation manifest")
    expected_parameters = {**PIXAL_PARAMETERS, "seed": int(seed)}
    if (
        generated.get("backend") != "pixal3d"
        or generated.get("input", {}).get("path") != str(reference)
        or generated.get("input", {}).get("sha256") != _sha256(reference)
        or generated.get("output", {}).get("path") != str(mesh)
        or generated.get("output", {}).get("sha256") != _sha256(mesh)
        or generated.get("output", {}).get("bytes") != mesh.stat().st_size
        or generated.get("model", {}).get("revision") != PIXAL_REVISION
        or generated.get("dino", {}).get("revision") != DINO_REVISION
        or generated.get("parameters") != expected_parameters
    ):
        raise PixalAnimalAdapterError("Pixal animal output/manifest lineage changed")

    try:
        from tools import human_attribute_pixal_contract as pixal_validation

        document, _file = pixal_validation.validate_staged_pixal_glb(
            mesh,
            staging=mesh.parent,
            input_rgba=reference,
        )
    except Exception as error:
        raise PixalAnimalAdapterError(
            f"Pixal animal packed-PBR GLB readback failed: {error}"
        ) from error
    primitive_count = sum(
        len(mesh_record.get("primitives", []))
        for mesh_record in document["meshes"]
    )
    return {
        "passed": True,
        "mesh_count": len(document["meshes"]),
        "primitive_count": primitive_count,
        "material_count": len(document["materials"]),
        "texture_count": len(document["textures"]),
        "image_count": len(document["images"]),
        "packed_pbr": True,
    }


def build_candidate_manifest(
    *,
    tag: str,
    tag_dir: Path,
    old_candidate: Mapping[str, Any],
    reference_lineage: Mapping[str, Any],
    mesh: Path,
    generated_manifest: Path,
    pbr_readback: Mapping[str, Any],
    public_mesh: Path | None = None,
    public_generated_manifest: Path | None = None,
) -> dict[str, Any]:
    if (
        old_candidate.get("asset_class") != "animal"
        or old_candidate.get("category") not in {"dog", "cat"}
        or pbr_readback.get("passed") is not True
    ):
        raise PixalAnimalAdapterError("animal candidate or Pixal PBR evidence is invalid")
    tag_dir = Path(tag_dir).resolve()
    public_mesh = Path(public_mesh or mesh).resolve()
    public_generated_manifest = Path(
        public_generated_manifest or generated_manifest
    ).resolve()
    variant_index = 1
    if tag.rsplit("_v", 1)[-1].isdigit() and "_v" in tag:
        variant_index = int(tag.rsplit("_v", 1)[-1])
    base = tag.rsplit("_v", 1)[0] if "_v" in tag else tag
    generation = old_candidate["generation"]
    return {
        "schema_version": "source_asset_v1",
        "usage_scope": "research_candidate",
        "formal_registration_authorized": False,
        "asset_id": f"{base}_{variant_index:04d}",
        "legacy_tag": tag,
        "asset_class": "animal",
        "category": old_candidate["category"],
        "family": old_candidate["family"],
        "variant": {
            **copy.deepcopy(old_candidate.get("variant", {})),
            "variant_index": variant_index,
        },
        "generation": {
            "source_pipeline": "historical_flux1_reference+pixal3d",
            "seed": int(reference_lineage["seed"]),
            "positive_prompt": generation.get("positive_prompt"),
            "negative_prompt": generation.get("negative_prompt"),
            "text_description": generation.get("text_description"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "models": {
                "reference_generator": {
                    "name": reference_lineage["generator"],
                    "revision": reference_lineage["revision"],
                },
                "pixal3d": {
                    "name": "TencentARC/Pixal3D",
                    "revision": PIXAL_REVISION,
                },
                "dino": {
                    "name": "camenduru/dinov3-vitl16-pretrain-lvd1689m",
                    "revision": DINO_REVISION,
                },
            },
            "parameters": {**PIXAL_PARAMETERS, "seed": int(reference_lineage["seed"])},
            "historical_reference_lineage": copy.deepcopy(dict(reference_lineage)),
            "hunyuan_mesh_or_textures_reused": False,
        },
        # Historical appearance measurements came from Hunyuan's baked
        # diffuse and must not be attributed to the independent Pixal PBR.
        "appearance": {
            "dominant_colors": [],
            "color_tags": [],
            "lightness": None,
            "saturation": None,
            "color_measurement_status": "pending_pixal_pbr_measurement",
            "historical_hunyuan_texture_measurements_reused": False,
        },
        "visual_assets": {
            "reference_image": _manifest_path(tag_dir / "reference.png"),
            "mesh_original": _manifest_path(public_mesh),
            "mesh_oriented": None,
            "mesh_runtime": None,
            "diffuse": None,
            "roughness": None,
            "metallic": None,
            "review_image": None,
            "direction_json": None,
            "runtime_metadata": None,
            "pixal_generation_manifest": _manifest_path(public_generated_manifest),
        },
        "artifact_integrity": {
            "mesh_original": _record(mesh, public_path=public_mesh),
            "pixal_generation_manifest": _record(
                generated_manifest, public_path=public_generated_manifest
            ),
            "pbr_glb_readback": copy.deepcopy(dict(pbr_readback)),
        },
        "material_contract": {
            "source": "pixal_packed_pbr_glb",
            "embedded_pbr_preserved": True,
            "new_diffuse_override_allowed": False,
            "downstream_rig_swap_mode": "preserve_imported_material",
        },
        "rig": copy.deepcopy(old_candidate["rig"]),
        "audio": copy.deepcopy(old_candidate["audio"]),
        "review": {
            "overall_status": "needs_review",
            "appearance_status": "needs_review",
            "direction_status": "needs_review",
            "texture_status": "needs_review",
            "rig_status": "needs_review",
            "audio_mapping_status": "inherited_pending_runtime_validation",
            "approved_by": None,
            "approved_at": None,
            "notes": (
                "Pixal backend-substitution canary; do not formally register "
                "before static, rig, animation, UE, and license review."
            ),
        },
    }


def stage_candidate(
    *,
    tag: str,
    destination: Path,
    old_candidate_manifest: Path,
    reference: Path,
    mesh: Path,
    generated_manifest: Path,
) -> Path:
    destination = Path(destination).resolve()
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    old_candidate = _load_json(old_candidate_manifest, "historical animal candidate")
    lineage = validate_reference_lineage(old_candidate_manifest, reference)
    pbr = validate_generated_pixal(
        mesh=mesh,
        generated_manifest=generated_manifest,
        reference=reference,
        seed=int(lineage["seed"]),
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{tag}.", suffix=".staging", dir=destination.parent)
    )
    try:
        staged_mesh = staging / "mesh.glb"
        staged_reference = staging / "reference.png"
        staged_generated = staging / "pixal_generation_manifest.json"
        shutil.copy2(mesh, staged_mesh)
        shutil.copy2(reference, staged_reference)
        shutil.copy2(generated_manifest, staged_generated)
        candidate = build_candidate_manifest(
            tag=tag,
            tag_dir=destination,
            old_candidate=old_candidate,
            reference_lineage={
                **lineage,
                "reference": _record(staged_reference, public_path=destination / "reference.png"),
            },
            mesh=staged_mesh,
            generated_manifest=staged_generated,
            pbr_readback=pbr,
            public_mesh=destination / "mesh.glb",
            public_generated_manifest=destination / "pixal_generation_manifest.json",
        )
        (staging / "source_asset_candidate.json").write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination / "source_asset_candidate.json"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    command = subparsers.add_parser("command")
    command.add_argument("--reference", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--gpu", type=int, default=0)
    command.add_argument("--seed", type=int, required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--tag", required=True)
    stage.add_argument("--destination", type=Path, required=True)
    stage.add_argument("--old-candidate-manifest", type=Path, required=True)
    stage.add_argument("--reference", type=Path, required=True)
    stage.add_argument("--mesh", type=Path, required=True)
    stage.add_argument("--generated-manifest", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.action == "command":
        print(
            json.dumps(
                build_pixal_command(
                    reference=args.reference,
                    output=args.output,
                    gpu=args.gpu,
                    seed=args.seed,
                )
            )
        )
        return 0
    path = stage_candidate(
        tag=args.tag,
        destination=args.destination,
        old_candidate_manifest=args.old_candidate_manifest,
        reference=args.reference,
        mesh=args.mesh,
        generated_manifest=args.generated_manifest,
    )
    print(f"PIXAL_ANIMAL_CANDIDATE_STAGED {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
