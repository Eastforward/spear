from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools import controlled_source_asset_schema as contracts
from tools.quadruped_morphotype_guide import load_morphotype_guide_profile


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "data/controlled_source_attributes_v1/candidate_profiles/animal"
    / "dog_pembroke_welsh_corgi_four_limb_rest_side_clay_v1.json"
)
REFERENCE_ROOT = (
    ROOT
    / "data/controlled_source_attributes_v1/references/animal"
    / "quaternius_dog_short_leg_min_tail_side_clay_v2"
)
GUIDE_PATH = (
    ROOT
    / "data/controlled_source_attributes_v1/contracts"
    / "quadruped_morphotype_guide_short_leg_min_tail_v2.json"
)
V1_PROVENANCE = (
    ROOT
    / "data/controlled_source_attributes_v1/references/animal"
    / "quaternius_dog_short_leg_short_tail_side_clay_v1/provenance.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_corgi_v3_is_a_new_frozen_method_reference_not_a_seed_retry():
    profile = contracts.validate_attribute_profile(_load(PROFILE_PATH))
    provenance = _load(REFERENCE_ROOT / "provenance.json")
    reference = profile["base_template"]["artifact"]
    reference_path = ROOT / reference["path"]

    assert profile["profile_revision"] == (
        "2026_07_27_v3_min_tail_four_distinct_paws_guide"
    )
    assert profile["lineage_group_id"] == "quaternius_dog_short_leg_min_tail_d85cfbd1"
    assert reference_path == REFERENCE_ROOT / "frame_0000.png"
    assert _sha256(reference_path) == reference["sha256"]
    assert reference_path.stat().st_size == reference["size_bytes"]
    assert provenance["reference"] == {
        "path": reference["path"],
        "sha256": reference["sha256"],
        "size_bytes": reference["size_bytes"],
        "canvas": [1024, 1024],
    }
    assert provenance["method_revision"]["kind"] == (
        "new_render_only_morphotype_method_revision"
    )
    assert provenance["method_revision"]["triggered_by_rejected_candidate_sha256"] == (
        "d63814fbe4fe4f4951ced196d79d082616bba916a7c47d9d3ea33d347b7b3951"
    )


def test_corgi_v3_uses_minimum_tail_and_explicit_four_paw_guards():
    profile = contracts.validate_attribute_profile(_load(PROFILE_PATH))
    guide = load_morphotype_guide_profile(GUIDE_PATH)
    guard = profile["generation_contract"]["pose_guard_prompt"].lower()
    negative = profile["generation_contract"]["negative_prompt"].lower()

    assert guide.leg_length_ratio == 0.65
    assert guide.tail_length_ratio == 0.25
    assert "tail must be no longer than the guide" in guard
    assert "exactly four distinct lower-limb and paw silhouettes" in guard
    assert "tail longer than guide" in negative
    assert "fewer than four lower limbs" in negative
    assert "indistinct fourth paw" in negative


def test_corgi_previous_reference_provenance_remains_present_and_unchanged():
    v1 = _load(V1_PROVENANCE)

    assert v1["reference_id"] == "quaternius_dog_short_leg_short_tail_side_clay_v1"
    assert v1["reference"]["sha256"] == (
        "ef883249cdbb484752607e8502071f05da88da53f975b56715d5fd2dd27c98b6"
    )
