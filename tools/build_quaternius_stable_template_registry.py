"""Build a verified registry for stable Quaternius Walk/Idle templates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path


SCHEMA = "avengine_quaternius_stable_template_registry_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path):
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise SystemExit(f"missing or unsafe artifact: {path}")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    asset_root = args.asset_root.resolve()
    media_root = args.media_root.resolve()
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace output: {output}")

    entries = []
    for manifest_path in sorted(asset_root.glob("*_manifest.json")):
        name = manifest_path.name.removesuffix("_manifest.json")
        runtime = asset_root / f"{name}_walk_idle.glb"
        inventory_path = asset_root / f"{name}_walk_idle_inventory.json"
        deformation_path = asset_root / f"{name}_walk_idle_deformation.json"
        walking_video = media_root / "videos" / f"{name}_walking_side.mp4"
        idle_video = media_root / "videos" / f"{name}_idle_side.mp4"
        manifest = read_json(manifest_path)
        inventory = read_json(inventory_path)
        deformation = read_json(deformation_path)
        if deformation.get("overall") != "passed":
            raise SystemExit(f"{name}: deformation gate did not pass")
        decisions = {
            item["requested_action"]: item["decision"]
            for item in deformation.get("actions", [])
        }
        if set(decisions) != {"Walking", "Idle"} or any(
            not value.startswith("passed_") for value in decisions.values()
        ):
            raise SystemExit(f"{name}: incomplete Walk/Idle deformation evidence")
        primary = max(inventory["meshes"], key=lambda item: item["vertices"])
        entries.append(
            {
                "template_id": f"quaternius_ultimate_{name.lower()}_v1",
                "taxonomy_label": name,
                "route_id": "stable_quaternius_native_template_v1",
                "state_classification": "research_candidate",
                "formal_dataset_registration_authorized": False,
                "runtime_glb": artifact(runtime),
                "source_extract_manifest": artifact(manifest_path),
                "inventory": artifact(inventory_path),
                "deformation_audit": artifact(deformation_path),
                "media": {
                    "walking_side": artifact(walking_video),
                    "idle_side": artifact(idle_video),
                },
                "geometry": {
                    "vertices": primary["vertices"],
                    "polygons": primary["polygons"],
                    "bbox_extent": primary["world_bbox_extent"],
                    "material_slots": primary["materials"],
                },
                "actions": ["Walking", "Idle"],
                "direction": {
                    "authored_front_axis": "negative_y",
                    "runtime_front_axis": "positive_x",
                    "cardinal_yaw_deg": 90,
                    "automatic_fine_yaw_inference": False,
                    "review_status": "agent_selected_pending_human_review",
                },
                "qa": {
                    "glb_readback": "passed",
                    "material_alpha_repair": "passed",
                    "walking_deformation": decisions["Walking"],
                    "idle_deformation": decisions["Idle"],
                    "isolated_media": "passed_agent_visual_check",
                    "ue_apartment_media": "pending",
                    "human_visual_review": "pending",
                },
                "source": manifest["source"],
                "license": manifest["license"],
                "material_alpha_repairs": manifest["output"][
                    "material_alpha_repairs"
                ],
            }
        )
    if len(entries) != 12:
        raise SystemExit(f"expected 12 stable templates, found {len(entries)}")

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "asset_root": str(asset_root),
        "media_root": str(media_root),
        "summary": {
            "templates": len(entries),
            "walk_idle_pairs": len(entries),
            "deformation_passed": len(entries),
            "human_review_pending": len(entries),
            "formal_assets": 0,
        },
        "entries": entries,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"QUATERNIUS_STABLE_REGISTRY_OK output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
