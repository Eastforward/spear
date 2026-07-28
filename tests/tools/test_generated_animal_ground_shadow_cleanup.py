"""Tests for bounded generated-animal opaque ground-shadow cleanup."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from tools import generated_animal_ground_shadow_cleanup as cleanup


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    height, width = 28, 36
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :] = [210, 210, 210]
    alpha = np.zeros((height, width), dtype=np.uint8)

    # One connected quadruped-like subject.  The head/nose keeps the global
    # right bbox beyond the shadow, so removal must not change whole-subject
    # bounds.
    alpha[4:16, 4:25] = 255
    alpha[7:11, 25:34] = 255
    rgb[alpha == 255] = [180, 80, 40]
    leg_boxes = (
        ("rear_near", cleanup.Box(4, 15, 8, 24)),
        ("rear_far", cleanup.Box(9, 15, 13, 24)),
        ("front_far", cleanup.Box(14, 15, 18, 24)),
        ("front_near", cleanup.Box(19, 15, 23, 24)),
    )
    for _, box in leg_boxes:
        alpha[box.y0 : box.y1, box.x0 : box.x1] = 255
        rgb[box.y0 : box.y1, box.x0 : box.x1] = [235, 230, 220]

    # Two reviewed opaque-shadow components connect at separate front paws.
    # They remain color-disconnected through protected white paw pixels.  One
    # non-gray fringe pixel becomes an allowed one-pixel satellite.
    alpha[20:23, 17:19] = 255
    rgb[20:23, 17:19] = [80, 82, 79]
    alpha[20:23, 23:32] = 255
    rgb[20:23, 23:32] = [80, 82, 79]
    alpha[21, 32] = 255
    rgb[21, 32] = [160, 140, 130]
    paw_probes = (
        ("rear_near", cleanup.Box(4, 15, 8, 24)),
        ("rear_far", cleanup.Box(9, 15, 13, 24)),
        ("front_far", cleanup.Box(14, 15, 17, 24)),
        ("front_near", cleanup.Box(19, 15, 23, 24)),
    )

    candidate = tmp_path / "candidate.png"
    input_alpha = tmp_path / "alpha.png"
    input_rgba = tmp_path / "input.png"
    Image.fromarray(rgb, mode="RGB").save(candidate)
    Image.fromarray(alpha, mode="L").save(input_alpha)
    rgba = np.dstack([rgb, alpha])
    Image.fromarray(rgba, mode="RGBA").save(input_rgba)
    rule = cleanup.CleanupRule(
        cleanup_roi=cleanup.Box(16, 18, 34, 25),
        seed_boxes=(
            cleanup.Box(27, 20, 31, 23),
            cleanup.Box(17, 20, 19, 23),
        ),
        rgb_channel_spread_max=8,
        rgb_sum_min=220,
        rgb_sum_max=260,
        protected_light_rgb_sum_min=600,
        protected_chroma_spread_min=50,
        minimum_selected_component_pixels=5,
        maximum_selected_component_pixels=30,
        minimum_removed_pixels=34,
        maximum_removed_pixels=34,
        maximum_removed_foreground_fraction=0.1,
        maximum_satellite_pixels=1,
        paw_probes=tuple(
            cleanup.PawProbe(name, box, minimum_opaque_pixels=20)
            for name, box in paw_probes
        ),
        body_probes=(
            cleanup.PawProbe(
                "abdomen",
                cleanup.Box(4, 4, 14, 14),
                minimum_opaque_pixels=80,
            ),
        ),
    )
    return candidate, input_alpha, input_rgba, rule


def _run(tmp_path: Path, rule: cleanup.CleanupRule | None = None):
    candidate, input_alpha, input_rgba, default_rule = _fixture(tmp_path)
    output_dir = tmp_path / "output"
    manifest = cleanup.clean_authenticated_ground_shadow(
        source_candidate_path=candidate,
        expected_source_candidate_sha256=_sha(candidate),
        input_alpha_path=input_alpha,
        expected_input_alpha_sha256=_sha(input_alpha),
        input_rgba_path=input_rgba,
        expected_input_rgba_sha256=_sha(input_rgba),
        output_dir=output_dir,
        pixal_seed=1234,
        rule=default_rule if rule is None else rule,
    )
    return manifest, output_dir


def test_cleanup_is_auditable_and_preserves_four_paw_probes(tmp_path):
    manifest, output_dir = _run(tmp_path)

    assert manifest["schema"] == cleanup.SCHEMA
    assert manifest["formal_dataset_registration_authorized"] is False
    assert (
        manifest["source"]["pixal_seed_decimal_frozen_for_controlled_rerun"]
        == "1234"
    )
    assert manifest["measurements"]["removed_alpha_pixel_count"] == 34
    assert len(manifest["measurements"]["selected_shadow_components"]) == 2
    assert manifest["measurements"]["satellite_pixel_count"] == 1
    assert manifest["measurements"]["output_foreground_component_count"] == 1
    assert all(
        item["byte_exact_preserved"]
        for item in manifest["measurements"]["paw_bottom_evidence"]
    )
    assert all(
        item["byte_exact_preserved"]
        for item in manifest["measurements"]["protected_body_evidence"]
    )
    assert (
        manifest["measurements"]["visible_rgb_raw_sha256_before"]
        != manifest["measurements"]["visible_rgb_raw_sha256_after"]
    )

    output_alpha = np.array(
        Image.open(output_dir / cleanup.OUTPUT_ALPHA_NAME)
    )
    output_rgba = np.array(
        Image.open(output_dir / cleanup.OUTPUT_RGBA_NAME)
    )
    removal = np.array(
        Image.open(output_dir / cleanup.OUTPUT_REMOVAL_MASK_NAME)
    )
    assert set(np.unique(output_alpha).tolist()) == {0, 255}
    assert int((removal == 255).sum()) == 34
    assert np.all(output_rgba[output_rgba[:, :, 3] == 0, :3] == 0)
    assert (output_dir / cleanup.OUTPUT_MANIFEST_NAME).is_file()
    for output in manifest["outputs"].values():
        assert _sha(Path(output["path"])) == output["sha256"]


def test_cleanup_rejects_authenticated_input_mismatch(tmp_path):
    candidate, input_alpha, input_rgba, rule = _fixture(tmp_path)

    with pytest.raises(cleanup.GroundShadowCleanupError, match="SHA-256 changed"):
        cleanup.clean_authenticated_ground_shadow(
            source_candidate_path=candidate,
            expected_source_candidate_sha256="0" * 64,
            input_alpha_path=input_alpha,
            expected_input_alpha_sha256=_sha(input_alpha),
            input_rgba_path=input_rgba,
            expected_input_rgba_sha256=_sha(input_rgba),
            output_dir=tmp_path / "output",
            pixal_seed=1234,
            rule=rule,
        )


def test_cleanup_rejects_out_of_bounds_rule(tmp_path):
    candidate, input_alpha, input_rgba, rule = _fixture(tmp_path)
    changed = cleanup.CleanupRule(
        **{
            **rule.__dict__,
            "cleanup_roi": cleanup.Box(20, 18, 100, 25),
        }
    )

    with pytest.raises(cleanup.GroundShadowCleanupError, match="outside"):
        cleanup.clean_authenticated_ground_shadow(
            source_candidate_path=candidate,
            expected_source_candidate_sha256=_sha(candidate),
            input_alpha_path=input_alpha,
            expected_input_alpha_sha256=_sha(input_alpha),
            input_rgba_path=input_rgba,
            expected_input_rgba_sha256=_sha(input_rgba),
            output_dir=tmp_path / "output",
            pixal_seed=1234,
            rule=changed,
        )


def test_cleanup_rejects_large_area_deletion(tmp_path):
    candidate, input_alpha, input_rgba, rule = _fixture(tmp_path)
    changed = cleanup.CleanupRule(
        **{
            **rule.__dict__,
            "minimum_removed_pixels": 1,
            "maximum_removed_pixels": 33,
        }
    )

    with pytest.raises(cleanup.GroundShadowCleanupError, match="outside"):
        cleanup.clean_authenticated_ground_shadow(
            source_candidate_path=candidate,
            expected_source_candidate_sha256=_sha(candidate),
            input_alpha_path=input_alpha,
            expected_input_alpha_sha256=_sha(input_alpha),
            input_rgba_path=input_rgba,
            expected_input_rgba_sha256=_sha(input_rgba),
            output_dir=tmp_path / "output",
            pixal_seed=1234,
            rule=changed,
        )


def test_cleanup_rejects_any_paw_probe_change(tmp_path):
    candidate, input_alpha, input_rgba, rule = _fixture(tmp_path)
    probes = list(rule.paw_probes)
    probes[-1] = cleanup.PawProbe(
        "front_near",
        cleanup.Box(19, 15, 24, 24),
        minimum_opaque_pixels=20,
    )
    changed = cleanup.CleanupRule(
        **{
            **rule.__dict__,
            "paw_probes": tuple(probes),
        }
    )

    with pytest.raises(cleanup.GroundShadowCleanupError, match="paw probe front_near changed"):
        cleanup.clean_authenticated_ground_shadow(
            source_candidate_path=candidate,
            expected_source_candidate_sha256=_sha(candidate),
            input_alpha_path=input_alpha,
            expected_input_alpha_sha256=_sha(input_alpha),
            input_rgba_path=input_rgba,
            expected_input_rgba_sha256=_sha(input_rgba),
            output_dir=tmp_path / "output",
            pixal_seed=1234,
            rule=changed,
        )


def test_cleanup_rejects_light_or_high_chroma_satellite_deletion(tmp_path):
    candidate, input_alpha, input_rgba, rule = _fixture(tmp_path)
    candidate_rgb = np.array(Image.open(candidate))
    rgba = np.array(Image.open(input_rgba))
    candidate_rgb[21, 32] = [160, 90, 80]
    rgba[21, 32, :3] = candidate_rgb[21, 32]
    Image.fromarray(candidate_rgb, mode="RGB").save(candidate)
    Image.fromarray(rgba, mode="RGBA").save(input_rgba)

    with pytest.raises(
        cleanup.GroundShadowCleanupError,
        match="protected light/high-chroma",
    ):
        cleanup.clean_authenticated_ground_shadow(
            source_candidate_path=candidate,
            expected_source_candidate_sha256=_sha(candidate),
            input_alpha_path=input_alpha,
            expected_input_alpha_sha256=_sha(input_alpha),
            input_rgba_path=input_rgba,
            expected_input_rgba_sha256=_sha(input_rgba),
            output_dir=tmp_path / "output",
            pixal_seed=1234,
            rule=rule,
        )
