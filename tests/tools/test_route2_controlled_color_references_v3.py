from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/route2_controlled_color_references_v3.py"
)
SPEC = importlib.util.spec_from_file_location(
    "route2_controlled_color_references_v3", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
color = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(color)


def test_color_feasibility_matrix_is_two_sexes_by_four_locked_attributes():
    assert len(color.CASE_SPECS) == 8
    assert len(color.CASE_BY_ID) == 8
    assert {case["sex"] for case in color.CASE_SPECS} == {"male", "female"}
    for sex in ("male", "female"):
        assert {case["attribute"] for case in color.CASE_SPECS if case["sex"] == sex} == set(color.ATTRIBUTES)


def test_targets_are_natural_explicit_and_geometry_is_locked():
    assert len({tuple(case["target_color_srgb"]) for case in color.CASE_SPECS}) == 8
    for case in color.CASE_SPECS:
        assert all(18 <= value <= 198 for value in case["target_color_srgb"])
        assert "exact geometry" in case["prompt"]
        assert case["alpha_policy"] == "source_alpha_byte_identical"
        assert case["execution_policy"].endswith("not_ordinary_production")


def test_all_source_specific_semantic_masks_are_nonempty_and_local():
    for case in color.CASE_SPECS:
        source = color.geometry.authenticate_source(case["sex"])
        with Image.open(source["image"]["path"]) as opened:
            image = opened.convert("RGB")
        with Image.open(source["alpha"]["path"]) as opened:
            alpha = opened.convert("L")
        core = color.build_edit_core(case, image, alpha)
        values = np.asarray(core) == 255
        assert 500 < int(values.sum()) < int(values.size * 0.20)
        assert not np.any(values & (np.asarray(alpha) < 128))


def test_contract_keeps_flux2_as_explicit_feasibility_fallback():
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '"flux2_is_ordinary_production_color_backend": False' in source
    assert '"ordinary_production_color_policy": "deterministic_semantic_material_transform"' in source
    assert "Hunyuan3D" in source
    assert "MODEL_REVISION" in source
    assert "candidate_alpha = source_alpha.copy()" in source
