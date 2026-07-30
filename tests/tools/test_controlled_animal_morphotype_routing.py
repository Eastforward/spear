import copy

import pytest

from tools import controlled_animal_morphotype_routing as routing


@pytest.fixture(scope="module")
def contract():
    return routing.load_trait_contract()


@pytest.fixture(scope="module")
def roster():
    return routing.load_roster()


def test_trait_contract_and_roster_are_self_consistent(contract, roster):
    axes = contract["axes"]
    for entry in roster["entries"]:
        for axis, value in entry["traits"].items():
            assert axis in axes, f"{entry['asset_key']} declares unknown axis {axis}"
            assert value in axes[axis]["values"], (
                f"{entry['asset_key']} declares unsupported {axis}={value}"
            )


def test_every_roster_entry_declares_the_sound_it_exists_to_voice(roster):
    # An animal asset that no AudioSet class maps onto is a modelling exercise,
    # not a sound source; the roster is the join point between the two.
    for entry in roster["entries"]:
        assert entry["acoustic"]["audioset_classes"]
        assert entry["acoustic"]["event_classes"]


def test_short_leg_knowledge_is_inherited_by_every_short_leg_breed(contract, roster):
    """The Corgi paid for the limb-separation corridors; the Dachshund gets them free.

    This is the property the breed-named prompt templates could not have: the
    trait carries the knowledge, so a breed that was never generated before
    still starts from everything the last short-legged breed learned.
    """
    corgi = routing.routing_plan("dog_pembroke_welsh_corgi", roster=roster, contract=contract)
    dachshund = routing.routing_plan("dog_dachshund", roster=roster, contract=contract)

    short_leg_fragments = [
        fragment
        for attribution in corgi["prompt_blocks"]["trait_attribution"]
        if attribution["axis"] == "leg_length_class"
        for fragment in attribution["pose_guard_fragments"]
    ]
    assert short_leg_fragments
    for fragment in short_leg_fragments:
        assert fragment in dachshund["prompt_blocks"]["pose_guard_fragments"]
    assert "limb_separation_precheck" in dachshund["required_gates"]


def test_tail_singularity_reaches_every_free_tail_breed(contract, roster):
    """The British Shorthair two-tail defect becomes a standing guard."""
    for entry in roster["entries"]:
        plan = routing.routing_plan(entry["asset_key"], roster=roster, contract=contract)
        assert "tail_singularity_precheck" in plan["required_gates"], entry["asset_key"]
        assert any(
            "two tails" in term for term in plan["prompt_blocks"]["negative_prompt_terms"]
        ), entry["asset_key"]


def test_risk_tier_drives_exploration_budget_not_the_breed_name(contract, roster):
    beagle = routing.routing_plan("dog_beagle", roster=roster, contract=contract)
    maine_coon = routing.routing_plan("cat_maine_coon", roster=roster, contract=contract)

    assert beagle["risk"]["tier"] == "low"
    assert beagle["budget"]["exploration_seed_count"] == 3
    assert beagle["budget"]["preview_only_triage_required"] is False

    assert maine_coon["risk"]["tier"] == "high"
    assert maine_coon["budget"]["exploration_seed_count"] == 10
    assert maine_coon["budget"]["preview_only_triage_required"] is True


def test_non_walking_gait_requires_an_owner_decision_but_is_not_banned(contract, roster):
    """The checklist is explicit that this is a readiness gate, not a ban list."""
    rabbit = routing.routing_plan("rabbit_domestic", roster=roster, contract=contract)
    assert rabbit["budget"]["owner_budget_decision_required"] is True
    assert "motion_family_decision" in rabbit["required_gates"]
    assert rabbit["acquisition_state"] != "rejected"


def test_composition_is_deterministic_and_order_independent(contract, roster):
    entry = routing.entry_for("cat_maine_coon", roster)
    forward = routing.compose_prompt_blocks(entry["traits"], contract)
    shuffled = dict(reversed(list(entry["traits"].items())))
    assert routing.compose_prompt_blocks(shuffled, contract) == forward


def test_undeclared_or_unknown_traits_fail_closed(contract, roster):
    entry = routing.entry_for("dog_beagle", roster)

    incomplete = {k: v for k, v in entry["traits"].items() if k != "tail_class"}
    with pytest.raises(routing.RoutingError, match="undeclared required traits"):
        routing.compose_prompt_blocks(incomplete, contract)

    unknown_value = {**entry["traits"], "coat_class": "iridescent"}
    with pytest.raises(routing.RoutingError, match="unsupported coat_class"):
        routing.compose_prompt_blocks(unknown_value, contract)

    unknown_axis = {**entry["traits"], "vibe": "good"}
    with pytest.raises(routing.RoutingError, match="not in the contract"):
        routing.compose_prompt_blocks(unknown_axis, contract)


def test_every_trait_value_cites_the_evidence_it_was_scored_from(contract):
    # Risk weights that nobody can trace back to a failure are how a checklist
    # turns into folklore.
    for axis, spec in contract["axes"].items():
        for value, body in spec["values"].items():
            assert body["evidence"].strip(), f"{axis}.{value} has no evidence"


def test_no_composed_guard_mentions_a_breed(contract, roster):
    """The defect this contract exists to fix, asserted directly.

    Shiba, British Shorthair and Corgi each shipped on a prompt template named
    after the breed, so none of them inherited the others' hard-won guards. If
    a breed name ever reappears inside a composed block, the knowledge has been
    re-keyed to an asset and stops generalizing again.
    """
    breeds = {entry["taxonomy"]["breed"] for entry in roster["entries"]}
    words = {word for breed in breeds for word in breed.split("_") if len(word) > 4}
    for entry in roster["entries"]:
        blocks = routing.compose_prompt_blocks(entry["traits"], contract)
        text = " ".join(
            blocks["pose_guard_fragments"] + blocks["negative_prompt_terms"]
        ).lower()
        for word in words:
            assert word not in text, (
                f"{entry['asset_key']} composed a guard mentioning {word!r}"
            )


def test_short_leg_route_carries_the_pose_fix_not_only_negative_terms(contract, roster):
    """Corgi attempt 13 proved the fix is a pose change, not a word list.

    The operative correction was staggering the near/far limbs longitudinally
    and keeping corridors through the upper attachments; a negative prompt
    alone did not clear the under-chest membrane.
    """
    plan = routing.routing_plan("dog_dachshund", roster=roster, contract=contract)
    guards = " ".join(plan["prompt_blocks"]["pose_guard_fragments"]).lower()
    assert "stagger" in guards
    assert "upper attachment" in guards
    assert plan["declared_view_yaw"], "an angled canonical view must be declared"
    assert "declared_view_yaw_recorded" in plan["required_gates"]
    assert "abdominal membrane" in plan["prompt_blocks"]["negative_prompt_terms"]


def test_wave_plans_are_ordered_cheapest_first(contract, roster):
    plans = routing.wave_plans("T2", roster=roster, contract=contract)
    totals = [plan["risk"]["risk_total"] for plan in plans]
    assert totals == sorted(totals)


def test_unknown_asset_key_fails_closed(roster):
    with pytest.raises(routing.RoutingError, match="not on the roster"):
        routing.routing_plan("dog_definitely_not_a_breed", roster=roster)


def test_contract_loader_rejects_a_weakened_axis(contract):
    broken = copy.deepcopy(contract)
    broken["axes"]["tail_class"]["values"]["long_thin"]["evidence"] = "   "
    with pytest.raises(routing.RoutingError, match="evidence"):
        _validate_via_tmp(broken)


def _validate_via_tmp(contract_value):
    """Round-trip a mutated contract through the loader's validation."""
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "contract.json"
        path.write_text(json.dumps(contract_value), encoding="utf-8")
        return routing.load_trait_contract(path)
