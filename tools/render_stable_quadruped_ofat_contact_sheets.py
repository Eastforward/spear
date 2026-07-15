#!/usr/bin/env python3
"""Render fixed-camera evidence for stable-quadruped OFAT instances."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


SPEAR_ROOT = Path(__file__).resolve().parents[1]
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import controlled_source_asset_schema as contracts  # noqa: E402
from tools import run_stable_quadruped_ofat_batch as ofat  # noqa: E402


SCHEMA = "avengine_stable_quadruped_ofat_visual_review_v1"
BLENDER = Path("/data/jzy/.local/bin/blender")
RENDERER = SPEAR_ROOT / "tools/blender_render_glb_animation.py"
EXPECTED_CHANGED = {
    None: 1,
    "size": 2,
    "body_build": 2,
    "coat_tone": 2,
    "life_stage": 2,
}


class ReviewError(RuntimeError):
    """Raised when authenticated OFAT evidence cannot produce a safe review."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ReviewError(f"missing or unsafe {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReviewError(f"cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ReviewError(f"{label} must be a JSON object")
    return value


def record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ReviewError(f"artifact is missing or unsafe: {path}")
    return {
        "absolute_path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def publication_record(path: Path, staging: Path, output: Path) -> dict[str, Any]:
    value = record(path)
    try:
        relative = path.resolve().relative_to(staging.resolve())
    except ValueError as error:
        raise ReviewError(f"generated artifact escaped staging root: {path}") from error
    value["absolute_path"] = str((output / relative).resolve())
    return value


def visible_meshes(inventory: Mapping[str, Any], *, skinned_only: bool) -> list[dict[str, Any]]:
    result = []
    for mesh in inventory.get("meshes", []):
        skinned = bool(mesh.get("vertices_with_weights", 0)) and bool(
            mesh.get("armature_modifiers")
        )
        has_material = bool(mesh.get("materials"))
        keep = skinned if skinned_only else (skinned or has_material)
        if keep:
            result.append(mesh)
    if not result:
        raise ReviewError("inventory has no visible mesh")
    return result


def union_bounds(meshes: Sequence[Mapping[str, Any]]) -> tuple[list[float], list[float]]:
    minimum = [float("inf")] * 3
    maximum = [float("-inf")] * 3
    for mesh in meshes:
        low = mesh.get("world_bbox_min")
        high = mesh.get("world_bbox_max")
        if not isinstance(low, list) or not isinstance(high, list) or len(low) != 3 or len(high) != 3:
            raise ReviewError("inventory mesh bbox is incomplete")
        for axis in range(3):
            minimum[axis] = min(minimum[axis], float(low[axis]))
            maximum[axis] = max(maximum[axis], float(high[axis]))
    return minimum, maximum


def extent_and_diagonal(inventory: Mapping[str, Any], *, skinned_only: bool) -> tuple[list[float], float]:
    minimum, maximum = union_bounds(visible_meshes(inventory, skinned_only=skinned_only))
    extent = [maximum[index] - minimum[index] for index in range(3)]
    return extent, math.sqrt(sum(value * value for value in extent))


def coat_luminance(manifest: Mapping[str, Any]) -> float:
    realization = manifest.get("realization", {})
    texture = realization.get("texture", {})
    if "mean_nonwhite_coat_luminance_after" in texture:
        return float(texture["mean_nonwhite_coat_luminance_after"])
    materials = realization.get("materials", {}).get("materials", [])
    colors = [
        item.get("after", [])[:3]
        for item in materials
        if item.get("role") == "coat" and len(item.get("after", [])) >= 3
    ]
    if not colors:
        raise ReviewError("realization has no coat material evidence")
    return sum(
        0.2126 * float(color[0]) + 0.7152 * float(color[1]) + 0.0722 * float(color[2])
        for color in colors
    ) / len(colors)


def age_appearance_parameters(manifest: Mapping[str, Any]) -> dict[str, float]:
    realization = manifest.get("realization", {})
    texture = realization.get("texture")
    if isinstance(texture, Mapping):
        return {
            "muzzle_gray_mix": float(texture.get("muzzle_gray_mix", 0.0)),
            "senior_coat_desaturation": float(
                texture.get("coat_desaturation", 0.0)
            ),
        }
    materials = realization.get("materials", {})
    return {
        "muzzle_gray_mix": float(materials.get("muzzle_gray_mix", 0.0)),
        "senior_coat_desaturation": float(
            materials.get("senior_coat_desaturation", 0.0)
        ),
    }


def validate_batch(path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    batch = load_json(path, "OFAT batch")
    if (
        batch.get("schema") != ofat.SCHEMA
        or batch.get("manifest_sha256") != contracts.manifest_sha256(batch)
        or batch.get("failed") != 0
        or batch.get("passed") != batch.get("instance_count")
        or batch.get("formal_dataset_registration_authorized") is not False
    ):
        raise ReviewError("OFAT batch authentication failed")
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in batch.get("entries", []):
        if entry.get("status") != "passed" or set(entry.get("checks", {}).values()) != {True}:
            raise ReviewError(f"OFAT entry is not fully passed: {entry.get('instance_id')}")
        groups.setdefault(str(entry["profile_schema_id"]), []).append(entry)
    if not groups or sum(map(len, groups.values())) != batch.get("instance_count"):
        raise ReviewError("OFAT batch profile grouping is incomplete")
    for profile_id, entries in groups.items():
        changed = {
            value: sum(item.get("changed_attribute_from_baseline") == value for item in entries)
            for value in EXPECTED_CHANGED
        }
        if len(entries) != 9 or changed != EXPECTED_CHANGED:
            raise ReviewError(f"profile does not contain one complete nine-instance OFAT: {profile_id}")
    return batch, groups


def entry_evidence(entry: Mapping[str, Any]) -> dict[str, Any]:
    inventory_path = Path(entry["artifacts"]["inventory"]["path"])
    manifest_path = Path(entry["artifacts"]["manifest"]["path"])
    glb_path = Path(entry["artifacts"]["glb"]["path"])
    inventory = load_json(inventory_path, "instance inventory")
    manifest = load_json(manifest_path, "instance manifest")
    if (
        sha256_file(inventory_path) != entry["artifacts"]["inventory"]["sha256"]
        or sha256_file(manifest_path) != entry["artifacts"]["manifest"]["sha256"]
        or sha256_file(glb_path) != entry["artifacts"]["glb"]["sha256"]
        or manifest.get("instance_id") != entry.get("instance_id")
        or inventory.get("input_sha256") != entry["artifacts"]["glb"]["sha256"]
    ):
        raise ReviewError(f"instance evidence changed: {entry.get('instance_id')}")
    skinned_extent, skinned_diagonal = extent_and_diagonal(inventory, skinned_only=True)
    _visible_extent, visible_diagonal = extent_and_diagonal(inventory, skinned_only=False)
    semantic = manifest.get("realization", {}).get("shape", {}).get(
        "semantic_measurements", {}
    )
    if set(semantic) != {
        "torso_weighted_lateral_rms_before",
        "torso_weighted_lateral_rms_after",
        "torso_weighted_lateral_rms_ratio",
        "head_weighted_radius_rms_before",
        "head_weighted_radius_rms_after",
        "head_weighted_radius_rms_ratio",
    }:
        raise ReviewError("realization lacks exact semantic shape measurements")
    return {
        "entry": dict(entry),
        "inventory": inventory,
        "manifest": manifest,
        "glb_path": glb_path.resolve(),
        "inventory_path": inventory_path.resolve(),
        "manifest_path": manifest_path.resolve(),
        "skinned_extent": skinned_extent,
        "skinned_diagonal": skinned_diagonal,
        "visible_diagonal": visible_diagonal,
        "torso_lateral_rms": float(semantic["torso_weighted_lateral_rms_after"]),
        "head_radius_rms": float(semantic["head_weighted_radius_rms_after"]),
        "coat_luminance": coat_luminance(manifest),
    }


def order_entries(evidence: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = [item for item in evidence if item["entry"]["label"] == "baseline"]
    if len(baseline) != 1:
        raise ReviewError("profile must contain exactly one baseline")

    def alternatives(attribute: str) -> list[dict[str, Any]]:
        return [
            item
            for item in evidence
            if item["entry"].get("changed_attribute_from_baseline") == attribute
        ]

    size = sorted(alternatives("size"), key=lambda item: item["skinned_diagonal"])
    build = sorted(alternatives("body_build"), key=lambda item: item["torso_lateral_rms"])
    coat = sorted(alternatives("coat_tone"), key=lambda item: item["coat_luminance"], reverse=True)
    age = sorted(
        alternatives("life_stage"),
        key=lambda item: {"young": 0, "senior": 1}.get(
            item["entry"]["sampled_attributes"]["life_stage"], 99
        ),
    )
    ordered = [*baseline, *size, *build, *coat, *age]
    if len(ordered) != 9 or len({item["entry"]["instance_id"] for item in ordered}) != 9:
        raise ReviewError("cannot order nine unique OFAT entries")
    return ordered


def automatic_checks(ordered: Sequence[dict[str, Any]]) -> dict[str, Any]:
    baseline = ordered[0]
    by_change = {
        attribute: [
            item for item in ordered if item["entry"].get("changed_attribute_from_baseline") == attribute
        ]
        for attribute in ("size", "body_build", "coat_tone", "life_stage")
    }
    size = sorted([*by_change["size"], baseline], key=lambda item: item["skinned_diagonal"])
    build = sorted(
        [*by_change["body_build"], baseline], key=lambda item: item["torso_lateral_rms"]
    )
    coat = sorted([*by_change["coat_tone"], baseline], key=lambda item: item["coat_luminance"])
    age = sorted(
        [*by_change["life_stage"], baseline], key=lambda item: item["head_radius_rms"]
    )
    size_values = [item["entry"]["sampled_attributes"]["size"] for item in size]
    build_values = [item["entry"]["sampled_attributes"]["body_build"] for item in build]
    coat_values = [item["coat_luminance"] for item in coat]
    age_values = {
        item["entry"]["sampled_attributes"]["life_stage"]: {
            "head_scale": float(item["manifest"]["realization"]["shape"]["head_scale"]),
            **age_appearance_parameters(item["manifest"]),
        }
        for item in [baseline, *by_change["life_stage"]]
    }
    checks = {
        "size_order": size_values == ["small", "medium", "large"],
        "body_build_width_order": build_values == ["slim", "standard", "stocky"],
        "coat_luminance_strict_order": coat_values[0] < coat_values[1] < coat_values[2],
        "young_head_scale_exceeds_adult": age_values["young"]["head_scale"] > age_values["adult"]["head_scale"],
        "head_radius_order": [
            item["entry"]["sampled_attributes"]["life_stage"] for item in age
        ]
        == ["senior", "adult", "young"],
        "senior_appearance_parameter_present": (
            age_values["senior"]["muzzle_gray_mix"] > 0.0
            or age_values["senior"]["senior_coat_desaturation"] > 0.0
        ),
    }
    if set(checks.values()) != {True}:
        raise ReviewError(f"attribute ordering gate failed: {checks}")
    return {
        **checks,
        "measurements": {
            "size_order": [
                {
                    "value": item["entry"]["sampled_attributes"]["size"],
                    "skinned_diagonal": item["skinned_diagonal"],
                }
                for item in size
            ],
            "body_build_order": [
                {
                    "value": item["entry"]["sampled_attributes"]["body_build"],
                    "semantic_torso_lateral_rms": item["torso_lateral_rms"],
                }
                for item in build
            ],
            "coat_order": [
                {
                    "value": item["entry"]["sampled_attributes"]["coat_tone"],
                    "mean_material_luminance": item["coat_luminance"],
                }
                for item in coat
            ],
            "life_stage_parameters": age_values,
            "life_stage_head_radius_order": [
                {
                    "value": item["entry"]["sampled_attributes"]["life_stage"],
                    "semantic_head_radius_rms": item["head_radius_rms"],
                }
                for item in age
            ],
        },
    }


def render_one(
    item: Mapping[str, Any],
    destination: Path,
    camera_reference_diagonal: float,
    blender: Path,
) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=False)
    command = [
        str(blender),
        "--background",
        "--python",
        str(RENDERER),
        "--",
        "--input",
        str(item["glb_path"]),
        "--rest-pose",
        "--output-dir",
        str(destination),
        "--n-frames",
        "1",
        "--width",
        "640",
        "--height",
        "480",
        "--samples",
        "8",
        "--view",
        "side",
        "--orthographic",
        "--camera-reference-diagonal",
        f"{camera_reference_diagonal:.9f}",
        "--ground-plane",
    ]
    started = time.monotonic()
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    log_path = destination / "render.log"
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    image_path = destination / "frame_0000.png"
    if (
        result.returncode != 0
        or "RENDER_GLB_ANIM_OK" not in result.stdout
        or not image_path.is_file()
        or image_path.stat().st_size <= 0
    ):
        raise ReviewError(
            f"Blender review render failed: {item['entry']['instance_id']} log={log_path}"
        )
    with Image.open(image_path) as image:
        if image.size != (640, 480):
            raise ReviewError(f"review image resolution changed: {image_path}")
    return {
        "label": item["entry"]["label"],
        "instance_id": item["entry"]["instance_id"],
        "image_path": image_path,
        "log_path": log_path,
        "elapsed_seconds": time.monotonic() - started,
    }


def font(size: int):
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def build_contact_sheet(
    ordered: Sequence[dict[str, Any]],
    rendered: Mapping[str, Mapping[str, Any]],
    output: Path,
) -> None:
    sheet = Image.new("RGB", (1920, 1440), (18, 22, 28))
    label_font = font(22)
    for index, item in enumerate(ordered):
        label = item["entry"]["label"]
        with Image.open(rendered[item["entry"]["instance_id"]]["image_path"]) as opened:
            panel = opened.convert("RGB")
        draw = ImageDraw.Draw(panel)
        attrs = item["entry"]["sampled_attributes"]
        text = f"{label} | {attrs['size']} {attrs['body_build']} {attrs['coat_tone']} {attrs['life_stage']}"
        bbox = draw.textbbox((0, 0), text, font=label_font)
        draw.rectangle((0, 0, min(640, bbox[2] + 14), bbox[3] + 10), fill=(10, 14, 20))
        draw.text((7, 4), text, font=label_font, fill=(245, 248, 252))
        sheet.paste(panel, ((index % 3) * 640, (index // 3) * 480))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=True)


def run(args: argparse.Namespace) -> Path:
    batch_path = args.batch_status.resolve()
    batch, groups = validate_batch(batch_path)
    blender = args.blender.resolve()
    if not blender.is_file() or not os.access(blender, os.X_OK) or not RENDERER.is_file():
        raise ReviewError("Blender or renderer is unavailable")
    output = args.output_root.resolve()
    if output.exists() or output.is_symlink():
        raise ReviewError(f"refusing to replace output: {output}")
    staging = output.with_name(f".{output.name}.staging.{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise ReviewError(f"staging path already exists: {staging}")
    staging.mkdir(parents=True)
    started = time.monotonic()
    try:
        prepared = {}
        render_jobs = []
        for profile_id, entries in sorted(groups.items()):
            ordered = order_entries([entry_evidence(entry) for entry in entries])
            checks = automatic_checks(ordered)
            baseline = ordered[0]
            camera_reference = baseline["visible_diagonal"] * 1.22
            prepared[profile_id] = {
                "ordered": ordered,
                "checks": checks,
                "camera_reference_diagonal": camera_reference,
            }
            for item in ordered:
                render_jobs.append(
                    (
                        profile_id,
                        item,
                        staging / "profiles" / profile_id / "instances" / item["entry"]["label"],
                        camera_reference,
                    )
                )

        rendered: dict[str, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(render_one, item, destination, camera_reference, blender): (
                    profile_id,
                    item["entry"]["instance_id"],
                )
                for profile_id, item, destination, camera_reference in render_jobs
            }
            completed = 0
            for future in as_completed(futures):
                profile_id, instance_id = futures[future]
                result = future.result()
                rendered[instance_id] = result
                completed += 1
                print(
                    f"STABLE_OFAT_VISUAL_PROGRESS completed={completed}/{len(render_jobs)} "
                    f"profile={profile_id} instance={instance_id}",
                    flush=True,
                )

        profile_records = []
        for profile_id, value in sorted(prepared.items()):
            ordered = value["ordered"]
            sheet_path = staging / "profiles" / profile_id / "contact_sheet.png"
            build_contact_sheet(ordered, rendered, sheet_path)
            instances = []
            for item in ordered:
                entry = item["entry"]
                render = rendered[entry["instance_id"]]
                instances.append(
                    {
                        "label": entry["label"],
                        "changed_attribute_from_baseline": entry[
                            "changed_attribute_from_baseline"
                        ],
                        "instance_id": entry["instance_id"],
                        "sampled_attributes": entry["sampled_attributes"],
                        "glb": record(item["glb_path"]),
                        "inventory": record(item["inventory_path"]),
                        "realization_manifest": record(item["manifest_path"]),
                        "image": publication_record(
                            render["image_path"], staging, output
                        ),
                        "render_log": publication_record(
                            render["log_path"], staging, output
                        ),
                        "render_elapsed_seconds": render["elapsed_seconds"],
                    }
                )
            profile_records.append(
                {
                    "profile_schema_id": profile_id,
                    "taxonomy": ordered[0]["entry"]["taxonomy"],
                    "camera_reference_diagonal": value["camera_reference_diagonal"],
                    "instance_count": len(instances),
                    "instances": instances,
                    "contact_sheet": publication_record(sheet_path, staging, output),
                    "automatic_attribute_checks": value["checks"],
                }
            )

        manifest = {
            "schema": SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "state_classification": "research_candidate_automatic_attribute_and_deformation_passed_pending_human_visual_review",
            "formal_dataset_registration_authorized": False,
            "source_batch": record(batch_path),
            "source_batch_manifest_sha256": batch["manifest_sha256"],
            "profile_count": len(profile_records),
            "instance_count": len(render_jobs),
            "elapsed_seconds": time.monotonic() - started,
            "profiles": profile_records,
            "automatic_checks": {
                "all_source_entries_authenticated": True,
                "all_profiles_cover_nine_ofat_instances": True,
                "all_attribute_orders_passed": True,
                "all_fixed_camera_images_rendered": True,
                "all_visual_accessories_retained": True,
                "human_visual_review": "pending",
            },
        }
        manifest["manifest_sha256"] = contracts.manifest_sha256(manifest)
        manifest_path = staging / "review_manifest.json"
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(staging, output)
        final_manifest = output / "review_manifest.json"
        observed = load_json(final_manifest, "published visual review")
        if observed["manifest_sha256"] != contracts.manifest_sha256(observed):
            raise ReviewError("published visual review hash failed")
        return final_manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-status", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 17))
    parser.add_argument("--blender", type=Path, default=BLENDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run(args)
    except (ReviewError, contracts.ContractError, OSError, ValueError) as error:
        print(f"STABLE_OFAT_VISUAL_FAILED {error}", file=sys.stderr)
        return 2
    print(f"STABLE_OFAT_VISUAL_OK manifest={manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
