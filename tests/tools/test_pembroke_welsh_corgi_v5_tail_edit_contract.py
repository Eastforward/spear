from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools import controlled_source_asset_schema as contracts


ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = (
    ROOT
    / "data/controlled_source_attributes_v1/candidate_profiles/animal"
    / "dog_pembroke_welsh_corgi_four_limb_rest_side_clay_v1.json"
)
REFERENCE_ROOT = (
    ROOT
    / "data/controlled_source_attributes_v1/references/animal"
    / "flux2_corgi_v4_tail_edit_source_a3c6553a"
)
METHOD_PROVENANCE_PATH = (
    ROOT
    / "data/controlled_source_attributes_v1/candidate_profile_revisions/animal"
    / "dog_pembroke_welsh_corgi_four_limb_rest_side_v5/provenance.json"
)
SCHEMA_IMPLEMENTATION = ROOT / "tools/controlled_source_asset_schema.py"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_corgi_v5_uses_rejected_v4_only_as_hash_bound_tail_edit_source():
    profile = contracts.validate_attribute_profile(_load(PROFILE_PATH))
    provenance = _load(REFERENCE_ROOT / "provenance.json")
    source_manifest = _load(REFERENCE_ROOT / "source_candidate_manifest.json")
    source_rejection = _load(REFERENCE_ROOT / "source_objective_rejection.json")
    artifact = profile["base_template"]["artifact"]
    reference_path = ROOT / artifact["path"]

    assert profile["profile_revision"] == (
        "2026_07_27_v5_localized_tail_stump_edit_from_v4"
    )
    assert profile["lineage_group_id"] == (
        "flux2_corgi_v4_tail_edit_source_a3c6553a"
    )
    assert profile["base_template"]["template_id"] == (
        "flux2_corgi_v4_tail_edit_source_a3c6553a"
    )
    assert reference_path == REFERENCE_ROOT / "frame_0000.png"
    assert _sha256(reference_path) == artifact["sha256"]
    assert reference_path.stat().st_size == artifact["size_bytes"]
    assert provenance["reference"]["sha256"] == artifact["sha256"]
    assert provenance["usage"]["source_is_not_2d_approved"] is True
    assert provenance["generation"]["seed"] == 8248597625431610484
    assert provenance["generation"]["seed"] == source_manifest["generation"]["seed"]
    assert source_rejection["decision"] == "rejected"
    assert source_rejection["checks"]["hard_gates"]["species_correct_tail"] == (
        "rejected"
    )


def test_corgi_v5_is_one_localized_tail_edit_without_a_free_tail_guard():
    profile = contracts.validate_attribute_profile(_load(PROFILE_PATH))
    request = contracts.sample_instance_requests(
        profile,
        count=1,
        batch_seed=2026072705,
    )[0]
    prompt = request["generation_plan"]["prompt"].lower()
    negative = request["generation_plan"]["negative_prompt"].lower()

    assert request["sampler"]["algorithm"] == "balanced_quota_sampler_v3"
    assert profile["fixed_attributes"]["tail_shape"] == "stump"
    assert "authoritative pixel-level identity" in prompt
    assert "the only authorized edit is the tail region" in prompt
    assert "erase the substantial red-and-white free tail shaft" in prompt
    assert "tiny attached red-fur tail-root stump" in prompt
    assert "clear plain-background gap" in prompt
    assert "free tail visibly separated from both hind legs" not in prompt
    assert "global redraw" in negative
    assert "visible free tail shaft" in negative


def test_corgi_v5_profile_remains_valid_under_the_frozen_v4_sampler_contract():
    profile = contracts.validate_attribute_profile(_load(PROFILE_PATH))
    batch = contracts.build_request_batch(
        [profile],
        count_per_profile=1,
        batch_seed=2026072704,
        sampler_algorithm=contracts.UNIVERSAL_TAIL_GUARD_SAMPLER_ALGORITHM,
    )

    validated = contracts.validate_request_batch(batch, [profile])

    assert validated["sampler"]["algorithm"] == "balanced_quota_sampler_v2"
    assert "free tail visibly separated from both hind legs" in (
        validated["requests"][0]["generation_plan"]["prompt"]
    )


def test_corgi_v5_method_is_predeclared_and_generic_code_has_no_breed_branch():
    profile = contracts.validate_attribute_profile(_load(PROFILE_PATH))
    provenance = _load(METHOD_PROVENANCE_PATH)
    implementation = SCHEMA_IMPLEMENTATION.read_text(encoding="utf-8").lower()

    assert provenance["profile"]["profile_sha256"] == (
        contracts.profile_sha256(profile)
    )
    assert provenance["method_revision"]["this_is_not_a_seed_retry"] is True
    assert provenance["method_revision"]["same_generation_seed_retry_allowed"] is (
        False
    )
    assert provenance["method_revision"]["candidate_ranking_allowed"] is False
    assert provenance["method_revision"]["forbidden_replay_generation_seed"] == (
        8248597625431610484
    )
    assert provenance["triggering_failure"]["generation_seed"] == (
        8248597625431610484
    )
    implementation_record = provenance["generic_sampler_revision"][
        "implementation"
    ]
    assert implementation_record["path"] == (
        "tools/controlled_source_asset_schema.py"
    )
    assert implementation_record["sha256"] == _sha256(SCHEMA_IMPLEMENTATION)
    assert implementation_record["size_bytes"] == (
        SCHEMA_IMPLEMENTATION.stat().st_size
    )
    assert provenance["execution_freeze"]["batch_seed"] == 2026072705
    assert provenance["execution_freeze"]["generation_seed"] == 7530445765198762053
    assert "corgi" not in implementation
