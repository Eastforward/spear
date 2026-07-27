from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools import controlled_source_asset_schema as contracts


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "data/controlled_source_attributes_v1/candidate_profiles/animal"
    / "cat_british_shorthair_four_limb_rest_side_v1.json"
)
PROVENANCE_PATH = (
    ROOT
    / "data/controlled_source_attributes_v1/candidate_profile_revisions/animal"
    / "cat_british_shorthair_four_limb_rest_side_v3/provenance.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_british_shorthair_v3_is_new_method_bound_to_rejected_pixal_evidence():
    profile = contracts.validate_attribute_profile(_load(PROFILE_PATH))
    provenance = _load(PROVENANCE_PATH)
    profile_record = provenance["profile"]
    failure = provenance["triggering_failure"]
    revision = provenance["method_revision"]

    assert profile["profile_revision"] == (
        "2026_07_27_v3_reconstruction_safe_whiskerless_four_limb_method"
    )
    assert profile["generation_contract"]["prompt_template_id"] == (
        "quadruped_four_limb_rest_side_i2i_v2_reconstruction_safe_british_shorthair"
    )
    assert _sha256(PROFILE_PATH) == profile_record["sha256"]
    assert PROFILE_PATH.stat().st_size == profile_record["size_bytes"]
    assert hashlib.sha256(
        contracts.canonical_json(profile).encode("utf-8")
    ).hexdigest() == profile_record["canonical_sha256"]

    assert revision["kind"] == "new_reconstruction_safe_prompt_method_revision"
    assert revision["supersedes_profile_revision"] == (
        "2026_07_27_v2_one_shot_blue_adult_base"
    )
    assert revision["superseded_profile_canonical_sha256"] == (
        "a98ccd68dd09dcf37560bc95a21153566b9af7e0928c6168a32e7184e71dc2df"
    )
    assert failure["owner_accepted_2d"]["candidate_sha256"] == (
        "e097fa8f85af7c8aa9fa4ce2d7bf3c74be9ba89351a37e452ef61f139a5524df"
    )
    assert failure["raw_pixal_glb"]["sha256"] == (
        "7570694820c781cbc731ca909c964efa4a4a8b3c9ff388efc68c2108b3dbd192"
    )
    assert failure["formal_rejection_decision"]["sha256"] == (
        "948810bd0fd44fa2b953d082b8b0e2611ebe4cfe04202a89e121c147eb037d7d"
    )
    assert failure["formal_rejection_decision"]["decision_sha256"] == (
        "abc335d5dfa9d89b005a06afa4fb347a98a9039ef3a7d5064cfc83abdab9bd77"
    )
    assert failure["formal_rejection_decision"]["decision"] == "rejected"
    assert failure["formal_rejection_decision"]["checks"] == {
        "complete_silhouette": False,
        "four_limbs_usable": False,
        "pose_riggable": False,
        "texture_coherent": True,
        "no_large_holes": True,
    }


def test_british_shorthair_v3_forbids_same_input_or_seed_retry():
    profile = contracts.validate_attribute_profile(_load(PROFILE_PATH))
    provenance = _load(PROVENANCE_PATH)
    policy = profile["generation_contract"]["base_acquisition_policy"]
    revision = provenance["method_revision"]

    assert policy["policy_id"] == "animal_one_shot_no_seed_lottery_v1"
    assert revision["this_is_not_a_seed_retry"] is True
    assert revision["same_candidate_retry_allowed"] is False
    assert revision["same_generation_seed_retry_allowed"] is False
    assert revision["candidate_ranking_allowed"] is False
    assert revision["next_execution_requires_new_profile_bound_request"] is True
    assert revision["next_execution_requires_new_batch_seed"] is True
    assert revision["forbidden_replay_request_sha256"] == (
        provenance["triggering_failure"]["request_sha256"]
    )
    assert revision["forbidden_replay_candidate_sha256"] == (
        provenance["triggering_failure"]["owner_accepted_2d"]["candidate_sha256"]
    )
    assert revision["forbidden_replay_generation_seed"] == (
        provenance["triggering_failure"]["generation_seed"]
    )


def test_british_shorthair_v3_prompt_is_reconstruction_safe_and_identity_locked():
    profile = contracts.validate_attribute_profile(_load(PROFILE_PATH))
    positive = profile["generation_contract"]["positive_template"].lower()
    guard = profile["generation_contract"]["pose_guard_prompt"].lower()
    negative = profile["generation_contract"]["negative_prompt"].lower()

    assert "whiskers absent or too short" in guard
    assert "do not draw or render any thin line" in guard
    assert "whisker shaft" in guard
    assert "exactly four complete, independently bindable limb chains" in guard
    assert "reconstruction-safe compact horizontal limb offsets" in guard
    assert "are mandatory" in guard
    assert "unmistakable compact offset along the image horizontal axis" in guard
    assert "continuous plain-background channel" in guard
    assert "from torso to paw tip" in guard
    assert "visible whisker" in negative
    assert "floating line geometry" in negative
    assert "incomplete far-side limb" in negative
    assert "fewer than four complete limbs" in negative

    assert "british shorthair" in positive
    assert "stocky" in profile["generation_contract"]["value_labels"]["body_build"][
        "stocky"
    ]
    assert "blue-gray" in profile["generation_contract"]["value_labels"][
        "coat_color"
    ]["blue"]
    assert "short dense plush" in profile["generation_contract"]["value_labels"][
        "coat_length"
    ]["short"]
    assert "exactly one" in positive
    assert "the cat has exactly one" in guard
