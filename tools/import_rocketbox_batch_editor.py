"""Import or independently reload one shard of normalized Rocketbox humans in UE."""

from __future__ import annotations

import json
import os
import runpy
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import spear
import unreal


SPEAR_ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT = SPEAR_ROOT / "tools/import_gate_rocketbox_native_editor.py"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_inventory(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Rocketbox inventory is not a direct file: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != "rocketbox_human_inventory_v1"
        or payload.get("automatic_checks", {}).get("overall") != "passed"
        or payload.get("population", {}).get("total") != 115
    ):
        raise RuntimeError("Rocketbox inventory is not UE-batch-ready")
    return payload


def main() -> None:
    inventory_path = Path(os.environ["ROCKETBOX_NATIVE_INVENTORY_JSON"]).resolve()
    normalized_root = Path(
        os.environ["ROCKETBOX_NATIVE_BATCH_NORMALIZED_ROOT"]
    ).resolve()
    ue_manifest_root = Path(
        os.environ["ROCKETBOX_NATIVE_BATCH_UE_MANIFEST_ROOT"]
    ).resolve()
    shard_index = int(os.environ.get("ROCKETBOX_BATCH_SHARD_INDEX", "0"))
    shard_count = int(os.environ.get("ROCKETBOX_BATCH_SHARD_COUNT", "1"))
    verify_only = os.environ.get("ROCKETBOX_BATCH_VERIFY_ONLY") == "1"
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise RuntimeError("invalid Rocketbox UE batch shard")
    inventory = _load_inventory(inventory_path)
    avatars = sorted(
        inventory["avatars"], key=lambda item: item["base_avatar_id"]
    )
    avatars = [
        avatar
        for ordinal, avatar in enumerate(avatars)
        if ordinal % shard_count == shard_index
    ]
    os.environ["ROCKETBOX_NATIVE_ENABLE_DYNAMIC_BATCH"] = "1"
    os.environ["ROCKETBOX_NATIVE_BATCH_NORMALIZED_ROOT"] = str(normalized_root)
    os.environ["ROCKETBOX_NATIVE_BATCH_UE_MANIFEST_ROOT"] = str(ue_manifest_root)
    os.environ["ROCKETBOX_NATIVE_INVENTORY_JSON"] = str(inventory_path)
    if verify_only:
        os.environ["ROCKETBOX_NATIVE_VERIFY_ONLY"] = "1"
    else:
        os.environ.pop("ROCKETBOX_NATIVE_VERIFY_ONLY", None)

    results = []
    failures = []
    for ordinal, avatar in enumerate(avatars, start=1):
        avatar_id = avatar["base_avatar_id"]
        tag = f"{avatar_id}_original_ue_v1"
        source_root = normalized_root / tag
        source_glb = source_root / "runtime.glb"
        source_manifest = source_root / "normalization_manifest.json"
        ue_manifest = ue_manifest_root / tag / "ue_import_manifest.json"
        if not source_glb.is_file() or not source_manifest.is_file():
            failures.append(
                {
                    "base_avatar_id": avatar_id,
                    "error": "normalized source is missing",
                }
            )
            continue
        if not verify_only and ue_manifest.exists():
            results.append(
                {
                    "base_avatar_id": avatar_id,
                    "status": "skipped_existing",
                    "ue_manifest": str(ue_manifest),
                }
            )
            continue
        if verify_only and not ue_manifest.is_file():
            failures.append(
                {
                    "base_avatar_id": avatar_id,
                    "error": "UE import manifest is missing before verification",
                }
            )
            continue
        os.environ["ROCKETBOX_NATIVE_TAG"] = tag
        os.environ["ROCKETBOX_NATIVE_GLB"] = str(source_glb)
        os.environ["ROCKETBOX_NATIVE_SOURCE_MANIFEST"] = str(source_manifest)
        os.environ["ROCKETBOX_NATIVE_UE_MANIFEST"] = str(ue_manifest)
        spear.log(
            f"ROCKETBOX_BATCH_UE {ordinal}/{len(avatars)} "
            f"{'verify' if verify_only else 'import'} {avatar_id}"
        )
        started = datetime.now(timezone.utc)
        try:
            runpy.run_path(str(GATE_SCRIPT), run_name="__main__")
            manifest = json.loads(ue_manifest.read_text(encoding="utf-8"))
            expected_reload = "passed" if verify_only else "pending"
            if (
                manifest.get("base_avatar_id", avatar_id) != avatar_id
                or manifest.get("tag") != tag
                or manifest.get("automatic_checks", {}).get("overall", "passed")
                == "failed"
                or manifest.get("reload_verification", {}).get("status")
                != expected_reload
            ):
                raise RuntimeError("UE manifest postcondition changed")
            results.append(
                {
                    "base_avatar_id": avatar_id,
                    "status": "passed",
                    "ue_manifest": str(ue_manifest),
                    "second_process_verification": verify_only,
                    "elapsed_seconds": (
                        datetime.now(timezone.utc) - started
                    ).total_seconds(),
                }
            )
        except BaseException as error:
            failures.append(
                {
                    "base_avatar_id": avatar_id,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            spear.log(f"ROCKETBOX_BATCH_UE_FAILED {avatar_id}: {error}")
    status = {
        "schema_version": "rocketbox_batch_ue_editor_status_v1",
        "mode": "verify" if verify_only else "import",
        "shard_index": shard_index,
        "shard_count": shard_count,
        "avatar_count": len(avatars),
        "passed_count": sum(item["status"] == "passed" for item in results),
        "skipped_existing_count": sum(
            item["status"] == "skipped_existing" for item in results
        ),
        "failed_count": len(failures),
        "results": results,
        "failures": failures,
        "automatic_checks": {
            "overall": "passed" if not failures else "failed",
            "failed_avatar_ids": [item["base_avatar_id"] for item in failures],
        },
    }
    suffix = "verify" if verify_only else "import"
    status_path = (
        ue_manifest_root
        / f"batch_{suffix}_status_shard_{shard_index}_of_{shard_count}.json"
    )
    _atomic_json(status_path, status)
    if failures:
        raise RuntimeError(
            f"Rocketbox UE {suffix} shard has {len(failures)} failures"
        )
    spear.log(
        f"ROCKETBOX_BATCH_UE_ALL_OK mode={suffix} shard={shard_index}/"
        f"{shard_count} total={len(avatars)}"
    )


if __name__ == "__main__":
    main()
