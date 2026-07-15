import json
from pathlib import Path

from tools import build_stable_template_attribute_profiles as builder
from tools import controlled_source_asset_schema as contracts


ROOT = Path(__file__).resolve().parents[2]
CATALOG = (
    ROOT
    / "data/controlled_source_attributes_v1/stable_template_attribute_catalog_v1.json"
)


def test_catalog_has_exactly_three_values_and_all_four_domains():
    payload = json.loads(CATALOG.read_text(encoding="utf-8"))
    assert payload["schema"] == builder.CATALOG_SCHEMA
    assert payload["attribute_contract"]["maximum_values_per_randomized_attribute"] == 3
    assert payload["attribute_contract"]["required_domains"] == [
        "size",
        "body_build",
        "coat_tone",
        "life_stage",
    ]
    assert len(payload["entries"]) == 12
    for entry in payload["entries"]:
        assert len(entry["coat_tone_values"]) == 3
        assert len(set(entry["coat_tone_values"])) == 3
        assert "golden" not in " ".join(entry["coat_tone_values"])


def test_published_profiles_validate_and_bind_every_attribute():
    output = (
        ROOT
        / "tmp/controlled_source_asset_execution_v1/"
        "stable_template_attribute_profiles_v2_20260715"
    )
    profiles = sorted(output.glob("*_controlled_attributes_v1.json"))
    assert len(profiles) == 12
    for path in profiles:
        profile = contracts.validate_attribute_profile(
            json.loads(path.read_text(encoding="utf-8"))
        )
        assert set(profile["sampled_attribute_domains"]) == {
            "size",
            "body_build",
            "coat_tone",
            "life_stage",
        }
        assert all(
            len(values) == 3
            for values in profile["sampled_attribute_domains"].values()
        )
        assert set(profile["generation_contract"]["attribute_bindings"]) == {
            "size",
            "body_build",
            "coat_tone",
            "life_stage",
        }
        assert profile["generation_contract"]["route"] == "stable_animal_template_v1"
        for parameters in profile["generation_contract"]["attribute_bindings"]["size"]["values"].values():
            assert parameters["template_cardinal_yaw_deg"] in {-90, 0, 90, 180}
        assert profile["rig_profile"]["actions"] == ["Walking", "Idle"]
        assert profile["state_classification"] == "research_candidate"


def test_full_request_batch_is_cartesian_and_balanced():
    path = (
        ROOT
        / "tmp/controlled_source_asset_execution_v1/"
        "stable_template_attribute_inputs_v2_20260715/instance_requests.json"
    )
    batch = json.loads(path.read_text(encoding="utf-8"))
    assert len(batch["requests"]) == 12 * 81
    assert batch["sampler"]["count_per_profile"] == 81
    for profile_id, distribution in batch["distribution"].items():
        assert set(distribution) == {"size", "body_build", "coat_tone", "life_stage"}
        for counts in distribution.values():
            assert set(counts.values()) == {27}, profile_id
