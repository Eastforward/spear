"""Audit approved Hunyuan assets against the source asset registry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from source_asset_registry import default_registry_root, load_asset, load_registry


REPO_ROOT = Path(__file__).resolve().parents[2]


def default_approved_dir() -> Path:
    return REPO_ROOT / "tmp" / "hy3d_batch" / "approved"


def _registry_by_legacy_tag(registry_root: Path) -> dict[str, dict[str, Any]]:
    registry = load_registry(registry_root)
    out: dict[str, dict[str, Any]] = {}
    for item in registry["assets"]:
        asset = load_asset(item["asset_id"], registry_root)
        legacy_tag = asset.get("legacy_tag")
        if legacy_tag:
            out[legacy_tag] = {
                "asset_id": item["asset_id"],
                "asset": asset,
                "registry_entry": item,
            }
    return out


def audit_approved_assets(
    approved_dir: Path | str | None = None,
    registry_root: Path | str | None = None,
) -> list[dict[str, Any]]:
    approved_dir = Path(approved_dir) if approved_dir is not None else default_approved_dir()
    registry_root = (
        Path(registry_root) if registry_root is not None else default_registry_root()
    )
    by_tag = _registry_by_legacy_tag(registry_root)
    rows: list[dict[str, Any]] = []
    if not approved_dir.exists():
        return rows
    for tag_dir in sorted(p for p in approved_dir.iterdir() if p.is_dir()):
        tag = tag_dir.name
        required = [
            "direction.json",
            "mesh_oriented.glb",
            "hy3d_diffuse.jpg",
            "mesh_runtime.glb",
            "mesh_runtime.json",
        ]
        missing = [name for name in required if not (tag_dir / name).exists()]
        direction_approved = False
        direction_path = tag_dir / "direction.json"
        if direction_path.exists():
            try:
                direction_approved = bool(
                    json.loads(direction_path.read_text(encoding="utf-8")).get(
                        "human_approved"
                    )
                )
            except json.JSONDecodeError:
                direction_approved = False

        registry_rec = by_tag.get(tag)
        registered = registry_rec is not None
        rows.append({
            "legacy_tag": tag,
            "asset_id": registry_rec["asset_id"] if registry_rec else None,
            "registered": registered,
            "direction_approved": direction_approved,
            "missing_files": missing,
            "classification_complete": bool(
                registered and direction_approved and not missing
            ),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--approved-dir", default=str(default_approved_dir()))
    ap.add_argument("--registry-root", default=str(default_registry_root()))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    rows = audit_approved_assets(args.approved_dir, args.registry_root)
    if args.json:
        print(json.dumps(rows, indent=2))
        return

    complete = sum(1 for row in rows if row["classification_complete"])
    print(f"approved_assets={len(rows)} classification_complete={complete}")
    for row in rows:
        status = "OK" if row["classification_complete"] else "INCOMPLETE"
        print(
            f"{status} {row['legacy_tag']} "
            f"asset_id={row['asset_id'] or '-'} "
            f"missing={','.join(row['missing_files']) or '-'}"
        )


if __name__ == "__main__":
    main()
