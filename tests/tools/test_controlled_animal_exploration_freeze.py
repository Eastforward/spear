import json

import pytest

from tools import controlled_animal_exploration_freeze as ef
from tools import controlled_animal_one_shot_policy as one_shot


PLAN_ID = "test_wave_20260728_dog_dachshund"


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _candidates(tmp_path, count):
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ordinal in range(count):
        path = tmp_path / f"cand_{ordinal}.png"
        path.write_bytes(f"candidate-{ordinal}".encode("utf-8"))
        paths[ordinal] = path
    return paths


@pytest.fixture
def frozen_chain(tmp_path):
    plan = ef.build_exploration_plan("dog_dachshund", PLAN_ID)
    plan_path = _write(tmp_path / "plan.json", plan)
    candidates = _candidates(tmp_path, plan["seed_ladder"]["seed_count"])
    batch = ef.record_exploration_batch(plan_path, candidates)
    batch_path = _write(tmp_path / "batch.json", batch)
    declaration = ef.freeze_selection(
        plan_path,
        batch_path,
        selected_index=3,
        selection_criterion="limb_separation_precheck",
        unselected_reasons={
            ordinal: "paired paws touching" for ordinal in candidates if ordinal != 3
        },
    )
    declaration_path = _write(tmp_path / "frozen.json", declaration)
    return {
        "plan": plan,
        "plan_path": plan_path,
        "batch": batch,
        "batch_path": batch_path,
        "declaration": declaration,
        "declaration_path": declaration_path,
        "candidates": candidates,
    }


def test_the_one_shot_policy_is_not_weakened_by_exploration():
    """The freeze phase must inherit the untouched no-seed-lottery policy.

    Exploration buys a seed, not an asset. If this ever starts passing while
    the one-shot policy has been relaxed, the provenance argument is gone.
    """
    policy = ef.load_policy()
    assert policy["relationship_to_one_shot_policy"]["weakens_it"] is False
    # The original policy still refuses best-of-n at the production stage.
    original = one_shot.load_policy()
    assert original["per_request_cardinality"]["candidate_ranking_or_best_of_n_allowed"] is False
    assert original["per_request_cardinality"]["seed_retry_allowed"] is False


def test_seed_ladder_is_a_pure_function_of_the_plan_id():
    first = ef.seed_ladder(PLAN_ID, 6)
    assert first == ef.seed_ladder(PLAN_ID, 6)
    assert first[:3] == ef.seed_ladder(PLAN_ID, 3)
    assert ef.seed_ladder("other_plan", 6) != first
    assert len(set(first)) == 6


def test_seeds_stay_inside_the_flux_job_seed_domain():
    # tools/controlled_animal_one_shot_policy.validate_flux_job requires
    # 0 <= seed < 2**63.
    for seed in ef.seed_ladder(PLAN_ID, 10):
        assert 0 <= seed < (1 << 63)


def test_exploration_budget_comes_from_the_trait_tier(frozen_chain):
    assert frozen_chain["plan"]["risk"]["tier"] == "high"
    assert frozen_chain["plan"]["seed_ladder"]["seed_count"] == 10
    assert frozen_chain["plan"]["usage_scope"] == ef.EXPLORATION_USAGE_SCOPE


def test_seed_count_cannot_be_raised_per_asset():
    with pytest.raises(ef.ExplorationError, match="not a per-asset override"):
        ef.build_exploration_plan("dog_beagle", "beagle_v1", seed_count=9)


def test_every_candidate_including_failures_must_be_recorded(tmp_path):
    plan = ef.build_exploration_plan("dog_beagle", "beagle_v1")
    plan_path = _write(tmp_path / "plan.json", plan)
    candidates = _candidates(tmp_path, 3)
    del candidates[1]
    with pytest.raises(ef.ExplorationError, match="including failures"):
        ef.record_exploration_batch(plan_path, candidates)


def test_freeze_requires_a_reason_for_every_unselected_candidate(frozen_chain):
    with pytest.raises(ef.ExplorationError, match="needs a recorded reason"):
        ef.freeze_selection(
            frozen_chain["plan_path"],
            frozen_chain["batch_path"],
            selected_index=3,
            selection_criterion="limb_separation_precheck",
            unselected_reasons={0: "blurry"},
        )


def test_freeze_criterion_must_come_from_the_declared_vocabulary(frozen_chain):
    with pytest.raises(ef.ExplorationError, match="declared vocabulary"):
        ef.freeze_selection(
            frozen_chain["plan_path"],
            frozen_chain["batch_path"],
            selected_index=3,
            selection_criterion="looked_nicest",
            unselected_reasons={
                ordinal: "rejected" for ordinal in frozen_chain["candidates"] if ordinal != 3
            },
        )


def test_frozen_declaration_carries_the_search_size_and_the_criterion(frozen_chain):
    """The provenance the thirteen-workspace regime did not record."""
    exploration = frozen_chain["declaration"]["exploration"]
    assert exploration["candidates_considered"] == 10
    assert exploration["selected_index"] == 3
    assert exploration["selection_criterion"] == "limb_separation_precheck"
    assert len(exploration["unselected_reasons"]) == 9
    assert exploration["risk_tier"] == "high"


def test_frozen_request_is_a_single_shot_with_the_selected_seed(frozen_chain):
    request = frozen_chain["declaration"]["frozen_request"]
    ladder = frozen_chain["plan"]["seed_ladder"]["seeds"]
    assert request["generation_seed"] == ladder[3]
    assert request["flux_invocations"] == 1
    assert request["flux_images_per_invocation"] == 1
    assert request["seed_retry_allowed"] is False
    one_shot.validate_base_acquisition_record(request["base_acquisition_policy"])
    one_shot.validate_policy_record(request["one_shot_policy"])


def test_a_batch_from_another_plan_is_rejected(tmp_path, frozen_chain):
    other = ef.build_exploration_plan("dog_beagle", "beagle_v1")
    other_path = _write(tmp_path / "other_plan.json", other)
    other_batch = ef.record_exploration_batch(other_path, _candidates(tmp_path / "b", 3))
    other_batch_path = _write(tmp_path / "other_batch.json", other_batch)
    with pytest.raises(ef.ExplorationError, match="does not belong to this plan"):
        ef.freeze_selection(
            frozen_chain["plan_path"],
            other_batch_path,
            selected_index=0,
            selection_criterion="owner_2d_decision",
            unselected_reasons={1: "x", 2: "x"},
        )


def test_a_tampered_seed_ladder_fails_closed(tmp_path, frozen_chain):
    tampered = json.loads(frozen_chain["plan_path"].read_text(encoding="utf-8"))
    tampered["seed_ladder"]["seeds"][0] = 12345
    del tampered["plan_sha256"]
    from tools import controlled_animal_morphotype_routing as routing

    tampered["plan_sha256"] = routing.json_sha256(tampered)
    path = _write(tmp_path / "tampered.json", tampered)
    with pytest.raises(ef.ExplorationError, match="not the declared ladder"):
        ef.record_exploration_batch(path, frozen_chain["candidates"])


def test_reproducibility_passes_when_the_seed_reproduces_bitwise(tmp_path, frozen_chain):
    regenerated = tmp_path / "regen.png"
    regenerated.write_bytes(frozen_chain["candidates"][3].read_bytes())
    record = ef.verify_reproducibility(frozen_chain["declaration_path"], regenerated)
    assert record["state"] == "bitwise_reproduced"


def test_reproducibility_fails_closed_on_mismatch(tmp_path, frozen_chain):
    regenerated = tmp_path / "regen.png"
    regenerated.write_bytes(b"different pixels")
    with pytest.raises(ef.ExplorationError, match="did not reproduce"):
        ef.verify_reproducibility(frozen_chain["declaration_path"], regenerated)


def test_reproducibility_downgrade_must_be_explicit_and_recorded(tmp_path, frozen_chain):
    regenerated = tmp_path / "regen.png"
    regenerated.write_bytes(b"different pixels")
    record = ef.verify_reproducibility(
        frozen_chain["declaration_path"], regenerated, allow_nondeterministic=True
    )
    assert record["state"] == "seed_recorded_bitwise_unverified"
    assert record["regenerated_sha256"] != record["selected_candidate_sha256"]
