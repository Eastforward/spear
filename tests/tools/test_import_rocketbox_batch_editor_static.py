from __future__ import annotations

import ast
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "tools" / "import_rocketbox_batch_editor.py"


def test_batch_ue_wrapper_is_syntax_valid_and_inventory_bound():
    source = SCRIPT.read_text(encoding="utf-8")
    ast.parse(source)
    assert "rocketbox_human_inventory_v1" in source
    assert "ROCKETBOX_NATIVE_INVENTORY_JSON" in source
    assert "ROCKETBOX_NATIVE_BATCH_NORMALIZED_ROOT" in source
    assert "ROCKETBOX_NATIVE_BATCH_UE_MANIFEST_ROOT" in source
    assert "ROCKETBOX_BATCH_SHARD_INDEX" in source
    assert "ROCKETBOX_BATCH_SHARD_COUNT" in source


def test_batch_ue_wrapper_executes_existing_gate_per_avatar_without_replacing():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "runpy.run_path" in source
    assert "import_gate_rocketbox_native_editor.py" in source
    assert "ROCKETBOX_NATIVE_ENABLE_DYNAMIC_BATCH" in source
    assert "ROCKETBOX_NATIVE_TAG" in source
    assert "ROCKETBOX_NATIVE_GLB" in source
    assert "ROCKETBOX_NATIVE_SOURCE_MANIFEST" in source
    assert "ROCKETBOX_NATIVE_UE_MANIFEST" in source
    assert "skipped_existing" in source


def test_batch_ue_wrapper_supports_separate_second_process_verification():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "ROCKETBOX_BATCH_VERIFY_ONLY" in source
    assert "ROCKETBOX_NATIVE_VERIFY_ONLY" in source
    assert "second_process_verification" in source
    assert "failed_avatar_ids" in source
