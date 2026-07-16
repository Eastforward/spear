from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools import human_attribute_masks as masks


SPEAR_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = SPEAR_ROOT / "tmp/human_reference_review"
RGBA_ROOT = SPEAR_ROOT / "tmp/i23d_human_bakeoff_v1/inputs"


def _source(asset_id: str) -> tuple[Image.Image, Image.Image]:
    with Image.open(SOURCE_ROOT / asset_id / "candidate.png") as opened:
        rgb = opened.convert("RGB")
    with Image.open(RGBA_ROOT / asset_id / "alpha_isnet.png") as opened:
        alpha = opened.convert("L")
    return rgb, alpha


@pytest.mark.parametrize("case_id", list(masks.CASE_MASK_CONTRACTS))
def test_three_layer_masks_are_source_specific_binary_partition(case_id):
    contract = masks.CASE_MASK_CONTRACTS[case_id]
    source, alpha = _source(contract["base_asset_id"])

    bundle = masks.construct_mask_bundle(case_id, source, alpha)

    core = np.asarray(bundle["edit_core"], dtype=np.uint8)
    band = np.asarray(bundle["transition_band"], dtype=np.uint8)
    guard = np.asarray(bundle["protected_guard"], dtype=np.uint8)
    assert set(np.unique(core)) == {0, 255}
    assert set(np.unique(band)) == {0, 255}
    assert set(np.unique(guard)) == {0, 255}
    assert np.all((core > 0).astype(np.uint8) + (band > 0) + (guard > 0) == 1)
    assert bundle["metrics"]["source_alpha_sha256"] == hashlib.sha256(
        alpha.tobytes()
    ).hexdigest()
    assert bundle["quantitative_gates"] == contract["quantitative_gates"]
    assert bundle["construction_version"] == masks.MASK_CONSTRUCTION_VERSION


@pytest.mark.parametrize("case_id", ["tall_man", "short_woman"])
def test_height_masks_use_foot_anchored_alpha_union_not_a_canvas_rectangle(case_id):
    contract = masks.CASE_MASK_CONTRACTS[case_id]
    source, alpha = _source(contract["base_asset_id"])
    bundle = masks.construct_mask_bundle(case_id, source, alpha)
    core = np.asarray(bundle["edit_core"]) == 255
    source_fg = np.asarray(alpha) >= 128

    assert bundle["strategy"] == "foot_anchored_height_silhouette_union"
    assert core.mean() < 0.35
    assert np.all(core[source_fg])
    assert bundle["metrics"]["foot_anchor_y_px"] == int(np.where(source_fg)[0].max())
    assert bundle["target_parameters"]["height_ratio"] == contract["target_parameters"][
        "height_ratio"
    ]
    rows, columns = np.where(core)
    margin = min(
        int(rows.min()),
        int(columns.min()),
        source.height - 1 - int(rows.max()),
        source.width - 1 - int(columns.max()),
    )
    assert margin >= bundle["quantitative_gates"]["minimum_canvas_margin_px"]


@pytest.mark.parametrize(
    ("case_id", "maximum_background_fraction"),
    [("short_sleeve_color", 0.03), ("trousers", 0.03), ("shoes", 0.03)],
)
def test_color_masks_are_foreground_lab_connected_semantics(case_id, maximum_background_fraction):
    contract = masks.CASE_MASK_CONTRACTS[case_id]
    source, alpha = _source(contract["base_asset_id"])
    bundle = masks.construct_mask_bundle(case_id, source, alpha)
    core = np.asarray(bundle["edit_core"]) == 255
    foreground = np.asarray(alpha) >= 128

    assert bundle["strategy"] == "source_foreground_lab_connected_component"
    assert np.count_nonzero(core & ~foreground) / np.count_nonzero(core) <= maximum_background_fraction
    assert bundle["metrics"]["connected_component_count"] == contract["expected_components"]


def test_reviewed_accessory_envelopes_protect_non_target_face_and_background():
    male, male_alpha = _source("rocketbox_male_adult_01")
    glasses = masks.construct_mask_bundle("glasses", male, male_alpha)
    female, female_alpha = _source("rocketbox_female_adult_01")
    hat = masks.construct_mask_bundle("hat", female, female_alpha)

    assert glasses["strategy"] == "reviewed_eyewear_stroke_envelope"
    assert glasses["metrics"]["core_fraction"] < 0.02
    assert glasses["metrics"]["reviewed_envelope_id"] == "male_seed41_eyewear_v1"
    assert hat["strategy"] == "reviewed_headwear_add_envelope"
    assert hat["metrics"]["core_fraction"] < 0.04
    assert hat["metrics"]["protected_face_below_y_px"] > 0
    hat_core = np.asarray(hat["edit_core"]) == 255
    _, hat_columns = np.where(hat_core)
    bbox_center_x = (int(hat_columns.min()) + int(hat_columns.max())) / 2.0
    assert abs(bbox_center_x - (female.width - 1) / 2.0) <= 6.0


def test_lower_garment_and_shoe_semantics_do_not_absorb_the_neighboring_item():
    female, female_alpha = _source("rocketbox_female_adult_01")
    trousers = masks.construct_mask_bundle("trousers", female, female_alpha)
    trousers_rows = np.where(np.asarray(trousers["edit_core"]) == 255)[0]
    assert int(trousers_rows.max()) <= round(0.89 * (female.height - 1))

    male, male_alpha = _source("rocketbox_male_adult_01")
    shoes = masks.construct_mask_bundle("shoes", male, male_alpha)
    shoe_rows = np.where(np.asarray(shoes["edit_core"]) == 255)[0]
    assert int(shoe_rows.min()) >= round(0.875 * (male.height - 1))


def test_corrected_trousers_mask_protects_the_shirt_above_reviewed_waist():
    female, female_alpha = _source("rocketbox_female_adult_01")

    trousers = masks.construct_mask_bundle("trousers", female, female_alpha)

    reviewed_waist_y = round(0.50 * (female.height - 1))
    core = np.asarray(trousers["edit_core"]) == 255
    band = np.asarray(trousers["transition_band"]) == 255
    assert not np.any(core[:reviewed_waist_y])
    assert not np.any(band[:reviewed_waist_y])
    assert trousers["metrics"]["protected_above_y_px"] == reviewed_waist_y


def test_feathered_composite_is_exact_outside_band_and_blended_inside_band():
    source = Image.new("RGB", (32, 32), (10, 20, 30))
    generated = Image.new("RGB", (32, 32), (210, 120, 80))
    core = Image.new("L", (32, 32), 0)
    values = np.asarray(core).copy()
    values[10:22, 10:22] = 255
    core = Image.fromarray(values, "L")
    band, guard = masks.transition_and_guard(core, radius_px=4)

    candidate, proof = masks.feathered_composite(source, generated, core, band)

    src = np.asarray(source)
    out = np.asarray(candidate)
    core_values = np.asarray(core) == 255
    band_values = np.asarray(band) == 255
    assert np.array_equal(out[~(core_values | band_values)], src[~(core_values | band_values)])
    assert np.array_equal(out[core_values], np.asarray(generated)[core_values])
    assert np.any((out[band_values] != src[band_values]) & (out[band_values] != np.asarray(generated)[band_values]))
    assert proof["outside_changed_pixels"] == 0
    assert proof["outside_max_abs_channel_delta"] == 0
    assert proof["transition_is_feathered"] is True
    assert set(np.unique(np.asarray(guard))) == {0, 255}


@pytest.mark.parametrize(
    ("case_id", "mode"),
    [
        ("short_sleeve_color", "reuse_source_alpha_exact"),
        ("trousers", "reuse_source_alpha_exact"),
        ("shoes", "reuse_source_alpha_exact"),
        ("glasses", "add_envelope_only"),
        ("hat", "add_envelope_only"),
        ("tall_man", "height_silhouette_rebuild"),
        ("short_woman", "height_silhouette_rebuild"),
    ],
)
def test_candidate_alpha_policy_is_case_specific_and_clamped(case_id, mode):
    contract = masks.CASE_MASK_CONTRACTS[case_id]
    source, alpha = _source(contract["base_asset_id"])
    bundle = masks.construct_mask_bundle(case_id, source, alpha)
    predicted = Image.new("L", alpha.size, 255)

    candidate_alpha, proof = masks.build_candidate_alpha(
        case_id,
        alpha,
        predicted,
        bundle["edit_core"],
        bundle["transition_band"],
    )

    allowed = (np.asarray(bundle["edit_core"]) > 0) | (
        np.asarray(bundle["transition_band"]) > 0
    )
    source_values = np.asarray(alpha)
    candidate_values = np.asarray(candidate_alpha)
    assert proof["policy"] == mode
    if mode == "reuse_source_alpha_exact":
        assert np.array_equal(candidate_values, source_values)
    else:
        assert np.array_equal(candidate_values[~allowed], source_values[~allowed])


def test_mask_bundle_publication_contains_reviewable_overlay_hashes_and_no_replace(tmp_path):
    source, alpha = _source("rocketbox_male_adult_01")
    output = tmp_path / "glasses"

    manifest = masks.publish_mask_bundle(
        case_id="glasses",
        source=source,
        source_alpha=alpha,
        source_image_record={"path": "/approved/candidate.png", "sha256": "a" * 64},
        source_alpha_record={"path": "/approved/alpha.png", "sha256": "b" * 64},
        destination=output,
    )

    payload = json.loads(manifest.read_text())
    assert payload["schema"] == "human_attribute_mask_bundle_v2"
    assert set(payload["assets"]) == {
        "edit_core.png",
        "transition_band.png",
        "protected_guard.png",
        "overlay.png",
    }
    for filename, record in payload["assets"].items():
        path = output / filename
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["size_bytes"] == path.stat().st_size
        assert path.stat().st_mode & 0o222 == 0
    assert manifest.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError):
        masks.publish_mask_bundle(
            case_id="glasses",
            source=source,
            source_alpha=alpha,
            source_image_record={"path": "/approved/candidate.png", "sha256": "a" * 64},
            source_alpha_record={"path": "/approved/alpha.png", "sha256": "b" * 64},
            destination=output,
        )


def test_mask_agent_qa_is_immutable_hash_locked_and_invalidated_by_overlay_change(tmp_path):
    source, alpha = _source("rocketbox_male_adult_01")
    bundle = tmp_path / "glasses"
    masks.publish_mask_bundle(
        case_id="glasses",
        source=source,
        source_alpha=alpha,
        source_image_record={"path": "/approved/candidate.png", "sha256": "a" * 64},
        source_alpha_record={"path": "/approved/alpha.png", "sha256": "b" * 64},
        destination=bundle,
    )
    checks = {name: True for name in masks.MASK_AGENT_VISUAL_CHECKS}

    decision = masks.record_mask_agent_qa(
        bundle,
        status="agent_qa_passed_pending_user_acceptance",
        reviewer="codex-mask-qa",
        notes="Inspected source-specific full-resolution overlay.",
        checks=checks,
    )

    assert decision.stat().st_mode & 0o777 == 0o444
    assert masks.assert_mask_agent_qa_passed(bundle)["status"] == (
        "agent_qa_passed_pending_user_acceptance"
    )
    (bundle / "overlay.png").chmod(0o644)
    (bundle / "overlay.png").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="snapshot changed"):
        masks.assert_mask_agent_qa_passed(bundle)


def test_mask_agent_qa_rejects_mutable_manifest_or_artifact(tmp_path):
    source, alpha = _source("rocketbox_male_adult_01")
    bundle = tmp_path / "glasses"
    masks.publish_mask_bundle(
        case_id="glasses",
        source=source,
        source_alpha=alpha,
        source_image_record={"path": "/approved/candidate.png", "sha256": "a" * 64},
        source_alpha_record={"path": "/approved/alpha.png", "sha256": "b" * 64},
        destination=bundle,
    )
    checks = {name: True for name in masks.MASK_AGENT_VISUAL_CHECKS}
    masks.record_mask_agent_qa(
        bundle,
        status=masks.MASK_AGENT_PASS,
        reviewer="codex-mask-qa",
        notes="Inspected immutable source-specific full-resolution overlay.",
        checks=checks,
    )

    (bundle / "edit_core.png").chmod(0o644)
    with pytest.raises(ValueError, match="mutable|mode 0444"):
        masks.assert_mask_agent_qa_passed(bundle)


def test_mask_correction_rejects_old_snapshot_and_binds_replacement(tmp_path):
    source, alpha = _source("rocketbox_female_adult_01")
    old = tmp_path / "masks_v3" / "trousers"
    replacement = tmp_path / "masks_v4" / "trousers"
    old.parent.mkdir()
    replacement.parent.mkdir()
    source_record = {"path": "/approved/candidate.png", "sha256": "a" * 64}
    alpha_record = {"path": "/approved/alpha.png", "sha256": "b" * 64}
    masks.publish_mask_bundle(
        case_id="trousers",
        source=source,
        source_alpha=alpha,
        source_image_record=source_record,
        source_alpha_record=alpha_record,
        destination=old,
    )
    masks.publish_mask_bundle(
        case_id="trousers",
        source=source,
        source_alpha=alpha,
        source_image_record=source_record,
        source_alpha_record=alpha_record,
        destination=replacement,
    )
    checks = {name: True for name in masks.MASK_AGENT_VISUAL_CHECKS}
    prior = masks.record_mask_agent_qa(
        old,
        status=masks.MASK_AGENT_PASS,
        reviewer="codex-mask-qa",
        notes="Synthetic prior decision for correction contract.",
        checks=checks,
    )

    correction = masks.record_rejected_mask_correction(
        old,
        prior_decision=prior,
        replacement_bundle=replacement,
        reviewer="codex-mask-correction",
        reason="The old trousers core selected the non-target shirt hem.",
    )

    payload = json.loads(correction.read_text())
    assert correction.stat().st_mode & 0o777 == 0o444
    assert payload["status"] == "rejected"
    assert payload["old_snapshot"]["manifest_sha256"] == hashlib.sha256(
        (old / "mask_manifest.json").read_bytes()
    ).hexdigest()
    assert payload["replacement"]["manifest_sha256"] == hashlib.sha256(
        (replacement / "mask_manifest.json").read_bytes()
    ).hexdigest()
    with pytest.raises(FileExistsError):
        masks.record_rejected_mask_correction(
            old,
            prior_decision=prior,
            replacement_bundle=replacement,
            reviewer="codex-mask-correction",
            reason="The old trousers core selected the non-target shirt hem.",
        )
