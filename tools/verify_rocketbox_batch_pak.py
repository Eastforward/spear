"""Verify that a cooked SPEAR PAK contains every native Rocketbox runtime asset.

The UE import manifests are the source of truth.  This verifier converts their
object paths to cooked package paths, requires the skeletal mesh, skeleton,
physics asset, Walking, Standing_Idle, Blueprint, materials, and textures for
every avatar, and binds the report to the final PAK hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SPEAR_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = SPEAR_ROOT / "tmp/rocketbox_route1_inventory_v1/inventory.json"
DEFAULT_MANIFEST_ROOT = SPEAR_ROOT / "tmp/rocketbox_batch_native_ue_import_v1"
DEFAULT_PAK = (
    SPEAR_ROOT
    / "cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim/Content/Paks/SpearSim-Linux.pak"
)
DEFAULT_UNREAL_PAK = Path("/data/UE_5.5/Engine/Binaries/Linux/UnrealPak")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _load_json_file(path: Path, description: str) -> dict:
    path = path.resolve()
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{description} is not a direct file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {description}: {path}: {error}") from error


def parse_unreal_pak_list(output: str) -> set[str]:
    """Return normalized mount-relative paths from ``UnrealPak -List`` output."""
    entries: set[str] = set()
    for match in re.finditer(r'"(?P<path>[^"\r\n]+)"\s+offset:', output):
        path = match.group("path").replace("\\", "/")
        while path.startswith("../"):
            path = path[3:]
        if path:
            entries.add(path)
    return entries


def _object_path_to_pak_entry(object_path: str) -> str:
    if not isinstance(object_path, str) or not object_path.startswith("/Game/"):
        raise RuntimeError(f"unsupported UE object path: {object_path!r}")
    package = object_path.split(".", 1)[0]
    return "SpearSim/Content/" + package.removeprefix("/Game/") + ".uasset"


def _validate_manifest(payload: dict, avatar_id: str, tag: str) -> None:
    content = payload.get("content", {})
    animations = content.get("animations", {})
    if (
        payload.get("schema") != "rocketbox_batch_native_ue_import_v1"
        or payload.get("base_avatar_id") != avatar_id
        or payload.get("tag") != tag
        or payload.get("reload_verification", {}).get("status") != "passed"
        or set(animations) != {"Walking", "Standing_Idle"}
        or not isinstance(content.get("materials"), list)
        or not isinstance(content.get("textures"), list)
    ):
        raise RuntimeError(f"UE import manifest is not cook-ready: {avatar_id}")


def build_expected_entries(
    inventory_path: Path, manifest_root: Path
) -> tuple[set[str], list[dict]]:
    inventory = _load_json_file(inventory_path, "Rocketbox inventory")
    avatars = inventory.get("avatars")
    declared_total = inventory.get("population", {}).get("total")
    if (
        inventory.get("schema_version") != "rocketbox_human_inventory_v1"
        or inventory.get("automatic_checks", {}).get("overall") != "passed"
        or not isinstance(avatars, list)
        or declared_total != len(avatars)
    ):
        raise RuntimeError("Rocketbox inventory is not PAK-verification ready")

    manifest_root = manifest_root.resolve()
    expected: set[str] = set()
    avatar_reports = []
    for avatar in sorted(avatars, key=lambda item: item["base_avatar_id"]):
        avatar_id = avatar["base_avatar_id"]
        if avatar.get("inventory_status") != "passed":
            raise RuntimeError(f"inventory avatar is not ready: {avatar_id}")
        tag = f"{avatar_id}_original_ue_v1"
        manifest_path = manifest_root / tag / "ue_import_manifest.json"
        payload = _load_json_file(manifest_path, "UE import manifest")
        _validate_manifest(payload, avatar_id, tag)
        content = payload["content"]

        required_paths = [
            content["blueprint"],
            content["skeletal_mesh"],
            content["skeleton"],
            content["animations"]["Walking"],
            content["animations"]["Standing_Idle"],
            *content["materials"],
            *content["textures"],
        ]
        avatar_entries = {_object_path_to_pak_entry(path) for path in required_paths}
        skeletal_mesh_entry = _object_path_to_pak_entry(content["skeletal_mesh"])
        avatar_entries.add(skeletal_mesh_entry.removesuffix(".uasset") + "_PhysicsAsset.uasset")
        expected.update(avatar_entries)
        avatar_reports.append(
            {
                "base_avatar_id": avatar_id,
                "tag": tag,
                "manifest": str(manifest_path.resolve()),
                "expected_asset_count": len(avatar_entries),
                "expected_entries": sorted(avatar_entries),
            }
        )
    return expected, avatar_reports


def verify_pak_inventory(
    pak_entries: set[str],
    inventory_path: Path,
    manifest_root: Path,
    *,
    pak_path: Path | None = None,
) -> dict:
    expected, avatars = build_expected_entries(inventory_path, manifest_root)
    missing = sorted(expected - pak_entries)
    if missing:
        sample = "\n".join(missing[:20])
        raise RuntimeError(
            f"missing {len(missing)} expected Rocketbox entries from PAK:\n{sample}"
        )

    report = {
        "schema": "rocketbox_batch_native_pak_audit_v1",
        "status": "passed",
        "generated_at": _utc_now(),
        "inventory": str(inventory_path.resolve()),
        "manifest_root": str(manifest_root.resolve()),
        "avatar_count": len(avatars),
        "expected_entry_count": len(expected),
        "pak_entry_count": len(pak_entries),
        "walking_count": sum(path.endswith("/Walking.uasset") for path in expected),
        "standing_idle_count": sum(
            path.endswith("/Standing_Idle.uasset") for path in expected
        ),
        "skeletal_mesh_count": sum(path.endswith("/runtime.uasset") for path in expected),
        "skeleton_count": sum(
            path.endswith("/runtime_Skeleton.uasset") for path in expected
        ),
        "physics_asset_count": sum(
            path.endswith("/runtime_PhysicsAsset.uasset") for path in expected
        ),
        "blueprint_count": sum(
            "/Blueprints/" in path and path.endswith(".uasset") for path in expected
        ),
        "missing_entries": [],
        "avatars": avatars,
    }
    if pak_path is not None:
        pak_path = pak_path.resolve()
        if pak_path.is_symlink() or not pak_path.is_file():
            raise RuntimeError(f"PAK is not a direct file: {pak_path}")
        report.update(
            {
                "pak_path": str(pak_path),
                "pak_size_bytes": pak_path.stat().st_size,
                "pak_sha256": _sha256(pak_path),
            }
        )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--manifest-root", type=Path, default=DEFAULT_MANIFEST_ROOT)
    parser.add_argument("--pak", type=Path, default=DEFAULT_PAK)
    parser.add_argument("--unreal-pak", type=Path, default=DEFAULT_UNREAL_PAK)
    parser.add_argument("--list-output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-avatar-count", type=int, default=115)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    for path, description in (
        (args.pak, "PAK"),
        (args.unreal_pak, "UnrealPak executable"),
    ):
        path = path.resolve()
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"{description} is not a direct file: {path}")

    process = subprocess.run(
        [str(args.unreal_pak.resolve()), str(args.pak.resolve()), "-List"],
        cwd=SPEAR_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _atomic_text(args.list_output.resolve(), process.stdout)
    if process.returncode != 0:
        raise RuntimeError(f"UnrealPak -List returned {process.returncode}")
    entries = parse_unreal_pak_list(process.stdout)
    report = verify_pak_inventory(
        entries,
        args.inventory,
        args.manifest_root,
        pak_path=args.pak,
    )
    if report["avatar_count"] != args.expected_avatar_count:
        raise RuntimeError(
            f"expected {args.expected_avatar_count} avatars, got {report['avatar_count']}"
        )
    report["unreal_pak"] = str(args.unreal_pak.resolve())
    report["list_output"] = str(args.list_output.resolve())
    _atomic_json(args.report.resolve(), report)
    print(
        "ROCKETBOX_PAK_AUDIT_ALL_OK "
        f"avatars={report['avatar_count']} entries={report['expected_entry_count']} "
        f"pak_sha256={report['pak_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
