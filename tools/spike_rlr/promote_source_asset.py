"""Promote a gated approved Hunyuan source into source_assets_v1."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

from source_asset_manifest import CANDIDATE_MANIFEST_NAME
from source_asset_registry import default_registry_root


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_APPROVED_DIR = REPO_ROOT / "tmp" / "hy3d_batch" / "approved"


def _repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _srgb_to_linear(c: float) -> float:
    c = c / 255.0
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _rgb_to_lab(rgb: tuple[int, int, int]) -> list[float]:
    r, g, b = (_srgb_to_linear(v) for v in rgb)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    x /= 0.95047
    z /= 1.08883

    def f(t: float) -> float:
        if t > 0.008856:
            return t ** (1.0 / 3.0)
        return (7.787 * t) + (16.0 / 116.0)

    fx, fy, fz = f(x), f(y), f(z)
    return [
        round((116.0 * fy) - 16.0, 1),
        round(500.0 * (fx - fy), 1),
        round(200.0 * (fy - fz), 1),
    ]


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _color_tags(colors: list[dict[str, Any]]) -> list[str]:
    tags: list[str] = []
    for color in colors:
        r, g, b = color["rgb"]
        mx = max(r, g, b)
        mn = min(r, g, b)
        if mx < 55:
            tag = "black"
        elif mn > 205:
            tag = "white"
        elif mx - mn < 28:
            tag = "gray"
        elif r >= g and r >= b and g > b * 0.9:
            tag = "brown"
        elif r >= g and r >= b:
            tag = "red"
        elif g >= r and g >= b:
            tag = "green"
        else:
            tag = "blue"
        if tag not in tags:
            tags.append(tag)
    return tags


def measure_dominant_colors(diffuse: Path, *, max_colors: int = 3) -> dict[str, Any]:
    img = Image.open(diffuse).convert("RGB")
    img.thumbnail((128, 128))
    quantized = img.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette() or []
    counts = quantized.getcolors(maxcolors=128 * 128) or []
    total = sum(count for count, _idx in counts) or 1
    roles = ("coat_primary", "coat_secondary", "coat_tertiary", "coat_quaternary")
    colors: list[dict[str, Any]] = []
    for role, (count, idx) in zip(roles, sorted(counts, reverse=True)):
        offset = int(idx) * 3
        rgb = tuple(int(v) for v in palette[offset:offset + 3])
        colors.append({
            "role": role,
            "hex": _hex(rgb),
            "rgb": list(rgb),
            "lab": _rgb_to_lab(rgb),
            "coverage": round(count / total, 3),
            "source": "measured_from_texture",
        })

    if colors:
        weighted_lightness = sum(
            color["lab"][0] * color["coverage"] for color in colors
        )
        lightness = round(weighted_lightness / 100.0, 3)
        saturation = round(sum(
            math.hypot(color["lab"][1], color["lab"][2]) * color["coverage"]
            for color in colors
        ) / 100.0, 3)
    else:
        lightness = None
        saturation = None
    return {
        "dominant_colors": colors,
        "color_tags": _color_tags(colors),
        "lightness": lightness,
        "saturation": saturation,
    }


def _validate_gate_files(tag_dir: Path) -> None:
    required = [
        "source_asset_candidate.json",
        "direction.json",
        "mesh_oriented.glb",
        "mesh_runtime.glb",
        "mesh_runtime.json",
        "hy3d_diffuse.jpg",
    ]
    missing = [name for name in required if not (tag_dir / name).exists()]
    if missing:
        raise RuntimeError(f"{tag_dir.name} missing required gate files: {missing}")
    direction = _read_json(tag_dir / "direction.json")
    if not direction.get("human_approved"):
        raise RuntimeError(f"{tag_dir.name} direction is not human approved")


def _visual_assets(tag_dir: Path) -> dict[str, str | None]:
    names = {
        "reference_image": "reference.png",
        "mesh_original": "mesh.obj",
        "mesh_oriented": "mesh_oriented.glb",
        "mesh_runtime": "mesh_runtime.glb",
        "mesh_runtime_walking": "mesh_runtime_walking.glb",
        "mesh_runtime_standing_idle": "mesh_runtime_standing_idle.glb",
        "diffuse": "hy3d_diffuse.jpg",
        "roughness": "hy3d_roughness.jpg",
        "metallic": "hy3d_metallic.jpg",
        "review_image": "direction_preview_review.png",
        "direction_json": "direction.json",
        "runtime_metadata": "mesh_runtime.json",
    }
    out: dict[str, str | None] = {}
    for key, name in names.items():
        path = tag_dir / name
        out[key] = _repo_path(path) if path.exists() else None
    return out


def _apply_runtime_metadata(asset: dict[str, Any], tag_dir: Path) -> None:
    runtime_path = tag_dir / "mesh_runtime.json"
    if not runtime_path.exists():
        return
    runtime = _read_json(runtime_path)
    if (
        asset.get("category") != "human"
        or runtime.get("schema_version") != "human_mixamo_runtime_v1"
    ):
        return

    animations = runtime.get("animations") or {}
    rig = asset.setdefault("rig", {})
    rig["runtime_type"] = runtime.get("runtime_type")
    rig["default_animation"] = runtime.get("default_animation")
    if runtime.get("recommended_actor_scale") is not None:
        rig["actor_scale"] = float(runtime["recommended_actor_scale"])
    if runtime.get("recommended_actor_z_lift_cm") is not None:
        rig["actor_z_lift_cm"] = float(runtime["recommended_actor_z_lift_cm"])
    if runtime.get("recommended_walking_forward_yaw_offset_deg") is not None:
        rig["walking_forward_yaw_offset_deg"] = float(
            runtime["recommended_walking_forward_yaw_offset_deg"]
        )
    else:
        rig.setdefault("walking_forward_yaw_offset_deg", 0.0)
    rig["animation_assets"] = animations


def _registry_entry_for(asset: dict[str, Any], asset_path: Path, registry_root: Path) -> dict[str, Any]:
    return {
        "asset_id": asset["asset_id"],
        "asset_class": asset["asset_class"],
        "category": asset["category"],
        "family": asset["family"],
        "path": str(asset_path.relative_to(registry_root)),
        "overall_status": asset["review"]["overall_status"],
    }


def _update_registry(registry_root: Path, entry: dict[str, Any]) -> None:
    registry_path = registry_root / "registry.json"
    if registry_path.exists():
        registry = _read_json(registry_path)
    else:
        registry = {"schema_version": "source_assets_v1", "assets": []}
    assets = [
        item for item in registry.get("assets", [])
        if item.get("asset_id") != entry["asset_id"]
    ]
    assets.append(entry)
    registry["assets"] = sorted(assets, key=lambda item: item["asset_id"])
    _write_json(registry_path, registry)


def promote_source_asset(
    tag: str,
    *,
    approved_dir: Path | str = DEFAULT_APPROVED_DIR,
    registry_root: Path | str | None = None,
) -> Path:
    approved_dir = Path(approved_dir)
    registry_root = Path(registry_root) if registry_root is not None else default_registry_root()
    tag_dir = approved_dir / tag
    _validate_gate_files(tag_dir)
    candidate_path = tag_dir / CANDIDATE_MANIFEST_NAME
    asset = _read_json(candidate_path)

    appearance = measure_dominant_colors(tag_dir / "hy3d_diffuse.jpg")
    asset["appearance"] = appearance
    asset["visual_assets"] = _visual_assets(tag_dir)
    _apply_runtime_metadata(asset, tag_dir)

    direction = _read_json(tag_dir / "direction.json")
    review = asset.setdefault("review", {})
    for key in (
        "overall_status",
        "appearance_status",
        "direction_status",
        "texture_status",
        "rig_status",
        "audio_mapping_status",
    ):
        review[key] = "approved"
    review["approved_by"] = direction.get("human_approved_by") or review.get("approved_by")
    review["approved_at"] = direction.get("human_approved_at") or review.get("approved_at")
    review.setdefault("notes", None)

    asset_path = (
        registry_root
        / asset["category"]
        / asset["family"]
        / asset["asset_id"]
        / "asset.json"
    )
    _write_json(asset_path, asset)
    _write_json(candidate_path, asset)
    _update_registry(registry_root, _registry_entry_for(asset, asset_path, registry_root))
    return asset_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--approved-dir", default=str(DEFAULT_APPROVED_DIR))
    ap.add_argument("--registry-root", default=str(default_registry_root()))
    args = ap.parse_args()
    path = promote_source_asset(
        args.tag,
        approved_dir=args.approved_dir,
        registry_root=args.registry_root,
    )
    print(f"PROMOTE_SOURCE_ASSET_DONE {path}")


if __name__ == "__main__":
    main()
