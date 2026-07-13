"""UE Editor batch import for authenticated stable native animal templates.

Environment:
  STABLE_ANIMAL_IMPORT_MANIFEST  absolute ``ue_import_jobs.json`` path
  STABLE_ANIMAL_IMPORT_RESULT    absolute result manifest path

The actual per-asset import remains the existing ``import_gate_animal_editor``
implementation.  This wrapper only adds a truthful stable_* namespace and
input/result authentication; it never labels native templates as Pixal.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy

import unreal


SCRIPT_DIR = Path(__file__).resolve().parent
IMPORT_ONE = SCRIPT_DIR / "import_gate_animal_editor.py"
BATCH_SCHEMA = "stable_animal_ue_import_batch_v1"
RESULT_SCHEMA = "stable_animal_ue_import_result_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest_path = Path(os.environ["STABLE_ANIMAL_IMPORT_MANIFEST"]).resolve()
    result_path = Path(os.environ["STABLE_ANIMAL_IMPORT_RESULT"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs", [])
    tags = [job.get("tag") for job in jobs]
    if (
        manifest.get("schema") != BATCH_SCHEMA
        or manifest.get("job_count") != len(jobs)
        or not jobs
        or len(tags) != len(set(tags))
        or any(not isinstance(tag, str) or not tag.startswith("stable_") for tag in tags)
    ):
        raise RuntimeError("stable animal UE import manifest is invalid")

    results = []
    for job in jobs:
        source = Path(job["rigged_glb"]).resolve()
        if (
            source.is_symlink()
            or not source.is_file()
            or _sha256(source) != job["rigged_glb_sha256"]
            or set(job.get("expected_actions", [])) != {"Idle", "Walking"}
            or job.get("formal_dataset_registration_authorized") is not False
        ):
            raise RuntimeError(f"stable source contract mismatch: {job.get('template_id')}")
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
                f"stable UE import readback failed for {job['tag']}: "
                f"missing_actions={sorted(missing_actions)} bp={bp_path}"
            )
        results.append(
            {
                "asset_id": job["asset_id"],
                "template_id": job["template_id"],
                "tag": job["tag"],
                "source": str(source),
                "source_sha256": job["rigged_glb_sha256"],
                "mesh_content_dir": mesh_dir,
                "blueprint": bp_path,
                "asset_count": len(imported),
                "assets": sorted(imported),
                "actions": sorted(set(job["expected_actions"])),
                "human_review_status": job["human_review_status"],
                "formal_dataset_registration_authorized": False,
                "status": "passed",
            }
        )

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "schema": RESULT_SCHEMA,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "input_manifest": str(manifest_path),
                "input_manifest_sha256": _sha256(manifest_path),
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
        f"STABLE_ANIMAL_UE_BATCH_IMPORT_OK jobs={len(results)} result={result_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
