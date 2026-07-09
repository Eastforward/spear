"""Registry loader for reusable visual/audio source assets.

The registry owns reusable asset identity. Scene/event code can resolve an
``asset_id`` into the legacy ``tag`` and ``audio_lookup`` fields that current
SPEAR render/audio passes still consume.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "source_assets_v1"
ASSET_SCHEMA_VERSION = "source_asset_v1"


def default_registry_root() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "source_assets_v1"


def _coerce_registry_root(registry_root: Path | str | None = None) -> Path:
    if registry_root is None:
        return default_registry_root()
    return Path(registry_root)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_registry(registry_root: Path | str | None = None) -> dict[str, Any]:
    root = _coerce_registry_root(registry_root)
    registry_path = root / "registry.json"
    if not registry_path.exists():
        raise FileNotFoundError(f"source asset registry not found: {registry_path}")
    registry = _read_json(registry_path)
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported source asset registry schema: "
            f"{registry.get('schema_version')!r}"
        )
    assets = registry.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError("source asset registry must contain an assets list")
    seen: set[str] = set()
    for item in assets:
        asset_id = item.get("asset_id")
        if not asset_id:
            raise RuntimeError("source asset registry entry missing asset_id")
        if asset_id in seen:
            raise RuntimeError(f"duplicate source asset_id in registry: {asset_id}")
        seen.add(asset_id)
        if not item.get("path"):
            raise RuntimeError(f"source asset {asset_id} missing manifest path")
    return registry


def _registry_entry(asset_id: str, registry_root: Path) -> dict[str, Any]:
    registry = load_registry(registry_root)
    for item in registry["assets"]:
        if item["asset_id"] == asset_id:
            return item
    raise KeyError(f"unknown source asset_id: {asset_id}")


def load_asset(
    asset_id: str,
    registry_root: Path | str | None = None,
) -> dict[str, Any]:
    root = _coerce_registry_root(registry_root)
    item = _registry_entry(asset_id, root)
    asset_path = root / item["path"]
    if not asset_path.exists():
        raise FileNotFoundError(f"source asset manifest not found: {asset_path}")
    asset = _read_json(asset_path)
    if asset.get("schema_version") != ASSET_SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported source asset schema for {asset_id}: "
            f"{asset.get('schema_version')!r}"
        )
    if asset.get("asset_id") != asset_id:
        raise RuntimeError(
            f"source asset manifest id mismatch: registry has {asset_id}, "
            f"manifest has {asset.get('asset_id')!r}"
        )
    return asset


def _review_status(asset: dict[str, Any]) -> str | None:
    review = asset.get("review") or {}
    return review.get("overall_status")


def _assert_resolvable_asset(asset: dict[str, Any]) -> None:
    asset_id = asset.get("asset_id")
    if not asset.get("legacy_tag"):
        raise RuntimeError(f"source asset {asset_id} missing legacy_tag")
    audio = asset.get("audio") or {}
    if not audio.get("default_lookup"):
        raise RuntimeError(f"source asset {asset_id} missing audio.default_lookup")


def approved_assets(
    registry_root: Path | str | None = None,
    *,
    asset_class: str | None = None,
    category: str | None = None,
) -> list[dict[str, Any]]:
    root = _coerce_registry_root(registry_root)
    registry = load_registry(root)
    results: list[dict[str, Any]] = []
    for item in registry["assets"]:
        if item.get("overall_status") != "approved":
            continue
        asset = load_asset(item["asset_id"], root)
        if _review_status(asset) != "approved":
            continue
        if asset_class is not None and asset.get("asset_class") != asset_class:
            continue
        if category is not None and asset.get("category") != category:
            continue
        results.append(asset)
    return results


def resolve_source_pool_entry(
    entry: dict[str, Any],
    registry_root: Path | str | None = None,
    *,
    require_approved: bool = True,
) -> dict[str, Any]:
    if "asset_id" not in entry:
        return dict(entry)

    root = _coerce_registry_root(registry_root)
    asset = load_asset(str(entry["asset_id"]), root)
    _assert_resolvable_asset(asset)
    status = _review_status(asset)
    if require_approved and status != "approved":
        raise RuntimeError(
            f"source asset {entry['asset_id']} is not approved "
            f"(overall_status={status!r})"
        )

    audio = asset.get("audio") or {}
    audio_lookup = entry.get("audio_lookup") or audio["default_lookup"]
    allowed = set(audio.get("allowed_lookups") or [])
    if allowed and audio_lookup not in allowed and audio_lookup != "silent":
        raise RuntimeError(
            f"audio_lookup {audio_lookup!r} is not allowed for source asset "
            f"{entry['asset_id']}"
        )

    resolved = dict(entry)
    resolved["tag"] = resolved.get("tag") or asset["legacy_tag"]
    resolved["audio_lookup"] = audio_lookup
    resolved["asset_class"] = resolved.get("asset_class") or asset.get("asset_class")
    resolved["category"] = resolved.get("category") or asset.get("category")
    resolved["family"] = resolved.get("family") or asset.get("family")
    rig = asset.get("rig") or {}
    for key in (
        "actor_scale",
        "actor_z_lift_cm",
        "walking_forward_yaw_offset_deg",
    ):
        if key not in resolved and key in rig:
            resolved[key] = rig[key]
    return resolved


def resolve_source_pool(
    pool: list[dict[str, Any]],
    registry_root: Path | str | None = None,
    *,
    require_approved: bool = True,
) -> list[dict[str, Any]]:
    return [
        resolve_source_pool_entry(
            entry,
            registry_root=registry_root,
            require_approved=require_approved,
        )
        for entry in pool
    ]
