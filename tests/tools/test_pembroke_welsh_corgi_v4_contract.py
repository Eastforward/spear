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
    / "quaternius_dog_short_leg_tail_stump_side_clay_v3"
)
GUIDE_PATH = (
    ROOT
    / "data/controlled_source_attributes_v1/contracts"
    / "quadruped_morphotype_guide_short_leg_tail_stump_v3.json"
)
GUIDE_IMPLEMENTATION = ROOT / "tools/quadruped_morphotype_guide.py"
V2_PROVENANCE = (
    ROOT
    / "data/controlled_source_attributes_v1/references/animal"
    / "quaternius_dog_short_leg_min_tail_side_clay_v2/provenance.json"
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


def test_corgi_v4_frozen_method_reference_remains_hash_bound():
    provenance = _load(REFERENCE_ROOT / "provenance.json")
    reference = provenance["reference"]
    reference_path = ROOT / reference["path"]

    assert reference_path == REFERENCE_ROOT / "frame_0000.png"
    assert _sha256(reference_path) == reference["sha256"]
    assert reference_path.stat().st_size == reference["size_bytes"]
    assert reference["canvas"] == [1024, 1024]
    assert provenance["method_revision"]["kind"] == (
        "new_render_only_morphotype_method_revision"
    )
    assert provenance["method_revision"]["supersedes_reference_id"] == (
        "quaternius_dog_short_leg_min_tail_side_clay_v2"
    )
    assert provenance["method_revision"]["triggered_by_rejected_candidate_sha256"] == (
        "6682838d09f1d68556b6ac045169b253d7b348e451747eaac6b775377ea1bde2"
    )
    assert "does not retry or select another seed" in (
        provenance["method_revision"]["reason"]
    )


def test_corgi_v4_uses_generic_non_degenerate_tail_stump_and_four_paw_guards():
    provenance = _load(REFERENCE_ROOT / "provenance.json")
    guide = load_morphotype_guide_profile(GUIDE_PATH)
    implementation = GUIDE_IMPLEMENTATION.read_text(encoding="utf-8").lower()

    assert guide.leg_length_ratio == 0.65
    assert guide.tail_length_ratio == 0.05
    assert provenance["morphotype_guide"]["tail_length_ratio"] == 0.05
    assert provenance["visual_inspection"]["checks"][
        "tail_is_root_stump_without_visible_free_shaft"
    ] == "passed"
    assert provenance["visual_inspection"]["checks"][
        "exactly_four_distinct_lower_limb_and_paw_silhouettes"
    ] == "passed"
    assert "corgi" not in implementation


def test_corgi_v4_provenance_binds_generic_contract_implementation_and_visual_gate():
    provenance = _load(REFERENCE_ROOT / "provenance.json")
    guide_record = provenance["morphotype_guide"]
    guide_artifact = guide_record["profile"]
    implementation_artifact = guide_record["implementation"]

    assert _sha256(ROOT / guide_artifact["path"]) == guide_artifact["sha256"]
    assert (ROOT / guide_artifact["path"]).stat().st_size == (
        guide_artifact["size_bytes"]
    )
    assert _sha256(ROOT / implementation_artifact["path"]) == (
        implementation_artifact["sha256"]
    )
    assert (ROOT / implementation_artifact["path"]).stat().st_size == (
        implementation_artifact["size_bytes"]
    )
    assert guide_record["tail_length_ratio"] == 0.05
    assert guide_record["realized_tail_length_ratio"] == (
        0.05000006059315269
    )
    assert guide_record["render_only"] is True
    assert guide_record["source_asset_unchanged"] is True
    assert provenance["visual_inspection"]["result"] == "passed"
    assert provenance["visual_inspection"]["checks"] == {
        "tail_is_root_stump_without_visible_free_shaft": "passed",
        "tail_stump_clear_of_hind_legs_and_paws": "passed",
        "low_set_short_leg_proportions": "passed",
        "exactly_four_distinct_lower_limb_and_paw_silhouettes": "passed",
        "four_paws_on_one_ground_plane": "passed",
    }


def test_corgi_previous_reference_provenance_remains_present_and_unchanged():
    v2 = _load(V2_PROVENANCE)
    v1 = _load(V1_PROVENANCE)

    assert v2["reference_id"] == "quaternius_dog_short_leg_min_tail_side_clay_v2"
    assert v2["reference"]["sha256"] == (
        "d85cfbd15b9a69394102cc8534fbb3326fb07099633f8cf9e0ae8186d281f025"
    )
    assert v1["reference_id"] == "quaternius_dog_short_leg_short_tail_side_clay_v1"
    assert v1["reference"]["sha256"] == (
        "ef883249cdbb484752607e8502071f05da88da53f975b56715d5fd2dd27c98b6"
    )
