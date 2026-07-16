import json
from pathlib import Path

from tools.build_audioset_indoor_animal_registry import (
    ANIMAL_ID,
    ONTOLOGY_REVISION,
    SCHEMA,
    build_registry,
    descendants,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/controlled_source_attributes_v1"


def registry():
    return build_registry(
        DATA / "audioset_ontology_v1.json",
        DATA / "audioset_ontology_commit_master.json",
    )


def test_registry_accounts_for_every_official_animal_node():
    result = registry()
    ontology = json.loads((DATA / "audioset_ontology_v1.json").read_text())
    by_id = {node["id"]: node for node in ontology}

    assert result["schema"] == SCHEMA
    assert result["ontology_snapshot"]["revision"] == ONTOLOGY_REVISION
    assert {node["audioset_id"] for node in result["ontology_nodes"]} == descendants(
        ANIMAL_ID, by_id
    )
    assert result["summary"]["visible_source_count"] == 32


def test_sound_events_map_to_sources_instead_of_duplicate_meshes():
    result = registry()
    nodes = {node["name"]: node for node in result["ontology_nodes"]}

    assert nodes["Bark"]["mapped_source_ids"] == ["dog"]
    assert nodes["Meow"]["mapped_source_ids"] == ["cat"]
    assert nodes["Neigh, whinny"]["mapped_source_ids"] == ["horse"]
    assert nodes["Buzz"]["mapped_source_ids"] == ["bee_wasp", "housefly"]
    assert set(nodes["Growling"]["mapped_source_ids"]) == {
        "cat",
        "dog",
        "roaring_big_cat",
        "wild_canid",
    }


def test_every_source_uses_bounded_code_sampled_instance_attributes():
    result = registry()
    for source in result["sources"]:
        contract = source["instance_attribute_contract"]
        assert contract["maximum_values_per_randomized_attribute"] == 3
        assert contract["sampling"] == "code_only_complete_prompt_and_json"
        assert contract["relative_edit_history_forbidden"] is True
        assert source["formal_dataset_registration_authorized"] is False


def test_non_quadrupeds_have_family_specific_actions():
    result = registry()
    by_source = {source["source_id"]: source for source in result["sources"]}

    assert "Flying" in by_source["bird_generic"]["required_actions"]
    assert by_source["snake"]["required_actions"] == ["Idle", "Slither"]
    assert by_source["frog"]["required_actions"] == ["Idle", "Hop"]
    assert by_source["whale"]["required_actions"] == ["IdleSwim", "Swimming"]
    assert by_source["whale"]["scene_domains"] == ["indoor_aquarium"]
