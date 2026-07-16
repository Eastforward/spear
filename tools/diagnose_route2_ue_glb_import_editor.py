"""One-shot UE Interchange GLB import diagnostic in a new isolated directory."""

import hashlib
import json
import os
from pathlib import Path

import unreal


SOURCE = Path(os.environ["ROUTE2_DIAG_SOURCE"]).resolve()
TAG = os.environ["ROUTE2_DIAG_TAG"]
REPORT = Path(os.environ["ROUTE2_DIAG_REPORT"]).resolve()
DESTINATION = f"/Game/MyAssets/Audioset/Diagnostics/{TAG}"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path):
    data = unreal.EditorAssetLibrary.find_asset_data(asset_path=path)
    return {
        "path": str(path),
        "asset_name": str(data.get_editor_property("asset_name")),
        "asset_class": str(
            data.get_editor_property("asset_class_path").get_editor_property(
                "asset_name"
            )
        ),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    allowed = (
        root / "tmp/pixal_tokenrig_route2_diagnostics_v1",
        root / "tmp/route2_tokenrig_ue_fastlane_v1",
        root / "tmp/hy3d_rocketbox_template_fit_v1",
    )
    if not SOURCE.is_file() or not any(SOURCE.is_relative_to(path) for path in allowed):
        raise RuntimeError(f"diagnostic source is not allowed: {SOURCE}")
    if not TAG.startswith("route2_diag_") or "/" in TAG or "\\" in TAG:
        raise RuntimeError(f"invalid diagnostic tag: {TAG!r}")
    if unreal.EditorAssetLibrary.does_directory_exist(directory_path=DESTINATION):
        raise RuntimeError(f"refusing to reuse diagnostic directory: {DESTINATION}")
    if REPORT.exists():
        raise RuntimeError(f"refusing to replace diagnostic report: {REPORT}")

    unreal.EditorAssetLibrary.make_directory(directory_path=DESTINATION)
    task = unreal.AssetImportTask()
    task.set_editor_property("async_", False)
    task.set_editor_property("automated", True)
    task.set_editor_property("destination_path", DESTINATION)
    task.set_editor_property("filename", str(SOURCE))
    task.set_editor_property("replace_existing", False)
    task.set_editor_property("replace_existing_settings", False)
    task.set_editor_property("save", True)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])

    task_paths = [str(path) for path in task.get_editor_property("imported_object_paths")]
    listed_paths = [
        str(path)
        for path in unreal.EditorAssetLibrary.list_assets(
            directory_path=DESTINATION,
            recursive=True,
            include_folder=False,
        )
    ]
    report = {
        "schema": "route2_ue_glb_import_diagnostic_v1",
        "source": str(SOURCE),
        "source_sha256": _sha256(SOURCE),
        "source_size_bytes": SOURCE.stat().st_size,
        "tag": TAG,
        "destination": DESTINATION,
        "task_imported_object_paths": task_paths,
        "listed_assets": [_record(path) for path in listed_paths],
        "status": "assets_created" if listed_paths else "no_assets_created",
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    unreal.log_warning("ROUTE2_UE_GLB_DIAGNOSTIC " + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
