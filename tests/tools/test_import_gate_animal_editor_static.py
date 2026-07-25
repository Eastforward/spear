from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_interchange_glb_import_joins_before_asset_readback():
    text = (ROOT / "tools/import_gate_animal_editor.py").read_text(
        encoding="utf-8"
    )

    import_call = text.index("asset_tools.import_asset_tasks")
    blocking_join = text.index("task.get_objects()", import_call)
    registry_join = text.index("wait_for_completion()", blocking_join)
    directory_readback = text.index(
        "unreal.EditorAssetLibrary.list_assets", registry_join
    )

    assert import_call < blocking_join < registry_join < directory_readback
    assert 'name="async_", value=True' in text
    assert "save_directory(" in text


def test_pixal_batch_preflights_ue_texture_compatibility():
    text = (ROOT / "tools/import_pixal_animal_batch_editor.py").read_text(
        encoding="utf-8"
    )

    preflight = text.index("_validate_ue_compatible_glb(job, source)")
    ue_import = text.index("runpy.run_path", preflight)

    assert preflight < ue_import
    assert '"EXT_texture_webp" in required' in text
    assert 'image.get("mimeType") == "image/webp"' in text
    assert "geometry_skin_animation_byte_graph_changed" in text
