import json
from pathlib import Path

from tools import prepare_controlled_source_asset_execution as preflight_lib
from tools import run_stable_quadruped_ofat_batch as batch


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = (
    ROOT
    / "tmp/controlled_source_asset_execution_v1/"
    "stable_template_attribute_preflight_v2_20260715/execution_preflight.json"
)


def load_preflight():
    return preflight_lib.validate_execution_preflight(
        json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    )


def test_plan_selects_nine_ofat_instances_per_profile():
    plan = batch.build_plan(load_preflight(), set())
    assert len(plan) == 12 * 9
    profiles = {entry["profile_schema_id"] for entry in plan}
    assert len(profiles) == 12
    for profile in profiles:
        entries = [entry for entry in plan if entry["profile_schema_id"] == profile]
        assert len(entries) == 9
        assert len({entry["instance_id"] for entry in entries}) == 9
        assert sum(entry["label"] == "baseline" for entry in entries) == 1
        assert sorted(
            entry["changed_attribute_from_baseline"]
            for entry in entries
            if entry["changed_attribute_from_baseline"] is not None
        ) == sorted([attribute for attribute in batch.ATTRIBUTES for _ in range(2)])


def test_every_ofat_entry_is_absolute_and_differs_in_at_most_one_attribute():
    plan = batch.build_plan(load_preflight(), set())
    for profile in {entry["profile_schema_id"] for entry in plan}:
        entries = [entry for entry in plan if entry["profile_schema_id"] == profile]
        baseline = next(entry for entry in entries if entry["label"] == "baseline")
        for entry in entries:
            assert set(entry["sampled_attributes"]) == set(batch.ATTRIBUTES)
            differences = [
                key
                for key in batch.ATTRIBUTES
                if entry["sampled_attributes"][key]
                != baseline["sampled_attributes"][key]
            ]
            expected = entry["changed_attribute_from_baseline"]
            assert differences == ([] if expected is None else [expected])


def test_profile_filter_is_fail_closed():
    preflight = load_preflight()
    profile = "quaternius_ultimate_husky_v1_controlled_attributes_v1"
    plan = batch.build_plan(preflight, {profile})
    assert len(plan) == 9
    assert {entry["profile_schema_id"] for entry in plan} == {profile}
