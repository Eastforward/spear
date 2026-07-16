from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/route2_controlled_geometry_references_v3.py"
)
SPEC = importlib.util.spec_from_file_location(
    "route2_controlled_geometry_references_v3", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


EXPECTED_CASES = [
    "male_long_sleeve",
    "female_long_sleeve",
    "male_shorts",
    "female_shorts",
    "male_baseball_cap",
    "female_baseball_cap",
    "male_rectangular_glasses",
    "female_rectangular_glasses",
]


def test_contract_has_exact_controlled_geometry_cross_product():
    assert list(runner.CASE_BY_ID) == EXPECTED_CASES
    assert len(runner.CASE_SPECS) == 8
    assert {(case["sex"], case["geometry_attribute"]) for case in runner.CASE_SPECS} == {
        (sex, geometry)
        for sex in ("male", "female")
        for geometry in (
            "long_sleeve",
            "shorts",
            "plain_baseball_cap_hat_compatible_hair",
            "thin_rectangular_glasses",
        )
    }
    assert [case["seed"] for case in runner.CASE_SPECS] == list(range(301, 309))
    assert all(case["target_color_policy"] == "preserve_base_or_fixed_neutral" for case in runner.CASE_SPECS)


def test_contract_uses_only_approved_flux2_and_exact_soft_t_sources():
    assert runner.MODEL_REVISION == "e7b7dc27f91deacad38e78976d1f2b499d76a294"
    assert runner.MODEL_INVENTORY_SHA256 == (
        "962ec618f2846728da8ac4ccb18fb61bdf6334c729017b3feaa48ae7710f04a4"
    )
    assert runner.SOURCE_PINS["male"]["image_sha256"] == (
        "820abc0edb324bee570614cc901b03112589b28f3ea11e14d971788bc97a0938"
    )
    assert runner.SOURCE_PINS["female"]["image_sha256"] == (
        "856df2ca3840cf74c9a48cb1ac2081fc0ac61700f5f2fb47aa4a37eb561fa03c"
    )
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "Flux2KleinPipeline" in source
    assert "local_files_only=True" in source
    assert "Hunyuan3D" in source  # only an explicit prohibited-model record
    assert "run_hunyuan" not in source
    assert "qualified_candidate" not in source
    assert "jobs_v2.json" not in source


@pytest.mark.parametrize("sex", ("male", "female"))
def test_live_approved_source_lineage_is_current(sex):
    authenticated = runner.authenticate_source(sex)
    assert authenticated["asset_id"] == runner.SOURCE_PINS[sex]["asset_id"]
    assert authenticated["image"]["sha256"] == runner.SOURCE_PINS[sex]["image_sha256"]
    assert authenticated["review"]["sha256"] == runner.SOURCE_PINS[sex]["review_sha256"]
    assert authenticated["approval"]["decision"] == "approved"


@pytest.mark.parametrize("case_id", EXPECTED_CASES)
def test_masks_are_nonempty_exact_partitions_with_protected_pixels(case_id):
    case = runner.CASE_BY_ID[case_id]
    alpha = Image.new("L", (runner.WIDTH, runner.HEIGHT), 255)
    core = runner.build_edit_core(case, alpha)
    band, guard = runner.transition_and_guard(core)
    values = [np.asarray(image, dtype=np.uint8) == 255 for image in (core, band, guard)]
    assert all(np.any(value) for value in values)
    assert np.all(sum(value.astype(np.uint8) for value in values) == 1)
    assert not np.any(values[0] & values[1])


def test_masked_composite_is_byte_exact_outside_authorized_region():
    source = Image.new("RGB", (64, 64), (10, 20, 30))
    generated = Image.new("RGB", (64, 64), (210, 120, 40))
    core = Image.new("L", (64, 64), 0)
    ImageDraw = __import__("PIL.ImageDraw", fromlist=["ImageDraw"]).Draw
    ImageDraw(core).rectangle((20, 20, 43, 43), fill=255)
    band, guard = runner.transition_and_guard(core, radius=3)

    candidate, proof = runner.composite_candidate(source, generated, core, band)

    source_values = np.asarray(source)
    candidate_values = np.asarray(candidate)
    guard_values = np.asarray(guard) == 255
    assert np.array_equal(source_values[guard_values], candidate_values[guard_values])
    assert proof["outside_changed_pixels"] == 0
    assert proof["outside_max_abs_channel_delta"] == 0
    assert proof["core_changed_fraction"] == 1.0


def test_metrics_reject_noop_and_accept_bilateral_masked_change():
    case = dict(runner.CASE_BY_ID["male_long_sleeve"])
    source = Image.new("RGB", (runner.WIDTH, runner.HEIGHT), (50, 50, 50))
    alpha = Image.new("L", source.size, 255)
    core = runner.build_edit_core(case, alpha)
    band, _ = runner.transition_and_guard(core)
    noop, noop_proof = runner.composite_candidate(source, source, core, band)
    noop_metrics = runner.evaluate_metrics(
        case,
        source,
        noop,
        alpha,
        alpha,
        core,
        band,
        noop_proof,
        {"outside_changed_pixels": 0, "added_foreground_pixels": 0, "removed_foreground_pixels": 0},
    )
    assert noop_metrics["passed"] is False
    changed_source = Image.new("RGB", source.size, (200, 100, 20))
    candidate, proof = runner.composite_candidate(source, changed_source, core, band)
    metrics = runner.evaluate_metrics(
        case,
        source,
        candidate,
        alpha,
        alpha,
        core,
        band,
        proof,
        {"outside_changed_pixels": 0, "added_foreground_pixels": 0, "removed_foreground_pixels": 0},
    )
    assert metrics["passed"] is True
    assert metrics["checks"]["bilateral_target_changed"] is True


def test_cli_requires_explicit_gpu_cases_and_review_status():
    args = runner.parse_args(
        ["generate", "--case-id", "male_long_sleeve", "--gpu", "2"]
    )
    assert args.case_id == ["male_long_sleeve"]
    assert args.gpu == "2"
    review = runner.parse_args(
        [
            "review",
            "--case-id",
            "female_shorts",
            "--status",
            "rejected",
            "--notes",
            "visible target drift",
        ]
    )
    assert review.status == "rejected"
    with pytest.raises(SystemExit):
        runner.parse_args(["generate", "--case-id", "male_long_sleeve"])


def test_runner_file_hash_is_stable_lowercase_sha256():
    digest = hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()
    assert runner.sha256_file(MODULE_PATH) == digest
    assert runner._SHA256.fullmatch(digest)
