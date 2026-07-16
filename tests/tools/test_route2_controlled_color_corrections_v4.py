from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/route2_controlled_color_corrections_v4.py"
)
SPEC = importlib.util.spec_from_file_location(
    "route2_controlled_color_corrections_v4", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def _at(values: np.ndarray, x: float, y: float) -> bool:
    return bool(values[round(y * (values.shape[0] - 1)), round(x * (values.shape[1] - 1))])


def test_two_corrections_replace_only_the_rejected_female_slots():
    assert tuple(runner.CASE_BY_ID) == (
        "female_top_teal_mask_v4",
        "female_hair_chestnut_mask_v4",
    )
    assert {case["attribute"] for case in runner.CASE_SPECS} == {"top_color", "fixed_hair_color"}


def test_top_mask_contains_shirt_but_excludes_exposed_upper_arm_skin():
    source = runner.base.geometry.authenticate_source("female")
    with Image.open(source["image"]["path"]) as opened:
        image = opened.convert("RGB")
    with Image.open(source["alpha"]["path"]) as opened:
        alpha = opened.convert("L")
    values = np.asarray(runner.base.build_edit_core(runner.CASE_SPECS[0], image, alpha)) == 255
    assert _at(values, 0.50, 0.36)
    assert _at(values, 0.35, 0.285)
    assert _at(values, 0.65, 0.285)
    assert not _at(values, 0.30, 0.36)
    assert not _at(values, 0.70, 0.36)


def test_hair_mask_contains_crown_and_ponytail_but_excludes_face_center():
    source = runner.base.geometry.authenticate_source("female")
    with Image.open(source["image"]["path"]) as opened:
        image = opened.convert("RGB")
    with Image.open(source["alpha"]["path"]) as opened:
        alpha = opened.convert("L")
    values = np.asarray(runner.base.build_edit_core(runner.CASE_SPECS[1], image, alpha)) == 255
    assert _at(values, 0.50, 0.102)
    assert _at(values, 0.542, 0.19)
    assert not _at(values, 0.50, 0.13)
    assert not _at(values, 0.50, 0.16)
