"""UE Editor batch wrapper for unique, non-destructive Pixal animal gates.

Environment:
  PIXAL_ANIMAL_IMPORT_MANIFEST  absolute ``ue_import_jobs.json`` path
  PIXAL_ANIMAL_IMPORT_RESULT    absolute result manifest path
"""
from __future__ import annotations

import hashlib
import json
import os
import runpy
from datetime import datetime, timezone
from pathlib import Path

import unreal


SCRIPT_DIR = Path(__file__).resolve().parent
IMPORT_ONE = SCRIPT_DIR / "import_gate_animal_editor.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest_path = Path(os.environ["PIXAL_ANIMAL_IMPORT_MANIFEST"]).resolve()
    result_path = Path(os.environ["PIXAL_ANIMAL_IMPORT_RESULT"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs", [])
    if not jobs:
        raise RuntimeError("Pixal animal UE batch contains no jobs")
    tags = [job["tag"] for job in jobs]
    if len(tags) != len(set(tags)) or any(not tag.startswith("pixal_") for tag in tags):
        raise RuntimeError(f"unsafe or duplicate Pixal UE tags: {tags}")

    results = []
    for job in jobs:
        source = Path(job["rigged_glb"]).resolve()
        if not source.is_file() or _sha256(source) != job["rigged_glb_sha256"]:
            raise RuntimeError(f"source hash mismatch for {job['tag']}: {source}")
        os.environ["GATE_TAG"] = job["tag"]
        os.environ["GATE_RIGGED_GLB"] = str(source)
        runpy.run_path(str(IMPORT_ONE), run_name="__main__")

        mesh_dir = f"/Game/MyAssets/Audioset/Meshes/gate_{job['tag']}"
        bp_dir = f"/Game/MyAssets/Audioset/Blueprints/gate_{job['tag']}"
        imported = unreal.EditorAssetLibrary.list_assets(
            directory_path=mesh_dir, recursive=True
        )
        names = {
            str(
                unreal.EditorAssetLibrary.find_asset_data(asset_path=asset)
                .get_editor_property(name="asset_name")
            )
            for asset in imported
        }
        missing_actions = set(job["expected_actions"]) - names
        bp_path = f"{bp_dir}/BP_gate_{job['tag']}"
        if missing_actions or not unreal.EditorAssetLibrary.does_asset_exist(
            asset_path=bp_path
        ):
            raise RuntimeError(
                f"UE import readback failed for {job['tag']}: "
                f"missing_actions={sorted(missing_actions)} bp={bp_path}"
            )
        results.append(
            {
                "tag": job["tag"],
                "legacy_tag": job["legacy_tag"],
                "source": str(source),
                "source_sha256": job["rigged_glb_sha256"],
                "mesh_content_dir": mesh_dir,
                "blueprint": bp_path,
                "asset_count": len(imported),
                "assets": sorted(imported),
                "actions": sorted(set(job["expected_actions"])),
                "status": "passed",
            }
        )

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "schema": "pixal_animal_ue_import_result_v1",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "input_manifest": str(manifest_path),
                "non_destructive_policy": manifest["non_destructive_policy"],
                "passed_count": len(results),
                "results": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"PIXAL_ANIMAL_UE_BATCH_IMPORT_OK jobs={len(results)} result={result_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
