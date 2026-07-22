#!/usr/bin/env python3
"""Build the fail-closed AudioSet indoor animal source registry.

AudioSet is a sound-event ontology, not a list of meshes.  This compiler keeps
the official IDs and graph intact while separating visible source entities
(dog, horse, cricket, ...), acoustic events (bark, neigh, buzz, ...), and pure
ontology containers.  Every node below ``Animal`` must be accounted for.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
from pathlib import Path


SCHEMA = "avengine_audioset_indoor_animal_source_registry_v1"
SPEAR_ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_SHA256 = "9c685f4403eecc3ca9be37fd7285cf212feaaea6ff7229d3e7ca89e0d1f2d15d"
ONTOLOGY_REVISION = "d417d32bf59c711abb5910fd2f76a0eb44697991"
ANIMAL_ID = "/m/0jbk"
CONTAINER_IDS = {ANIMAL_ID, "/m/068hy", "/m/0ch8v", "/m/01280g"}


MOTION_FAMILIES = {
    "quadruped_canid": ("Idle", "Walking"),
    "quadruped_felid": ("Idle", "Walking"),
    "quadruped_equid": ("Idle", "Walking"),
    "quadruped_bovid": ("Idle", "Walking"),
    "quadruped_small_ungulate": ("Idle", "Walking"),
    "quadruped_suid": ("Idle", "Walking"),
    "quadruped_rodent": ("Idle", "Scurry"),
    "bird_terrestrial": ("Idle", "Walking", "Hop"),
    "bird_perching_flight": ("PerchedIdle", "Hop", "Flying"),
    "amphibian_hop": ("Idle", "Hop"),
    "serpent_slither": ("Idle", "Slither"),
    "insect_crawl_flight": ("Idle", "Crawl", "Flying"),
    "aquatic_cetacean": ("IdleSwim", "Swimming"),
}


def source(
    source_id,
    node_id,
    motion_family,
    indoor_scope,
    scene_domains,
    appearance_attribute,
    implementation_status,
    *,
    variants=(),
):
    return {
        "source_id": source_id,
        "audioset_node_id": node_id,
        "motion_family": motion_family,
        "indoor_scope": indoor_scope,
        "scene_domains": list(scene_domains),
        "appearance_attribute": appearance_attribute,
        "implementation_status": implementation_status,
        "taxonomy_variants": list(variants),
    }


SOURCE_SPECS = [
    source("dog", "/m/0bt9lr", "quadruped_canid", "apartment_common", ("apartment", "veterinary_clinic"), "coat_tone", "strict_canary_validated"),
    source("cat", "/m/01yrx", "quadruped_felid", "apartment_common", ("apartment", "veterinary_clinic"), "coat_tone", "strict_canary_validated"),
    source("horse", "/m/03k3r", "quadruped_equid", "specialized_indoor_only", ("barn_stable", "large_animal_clinic"), "coat_tone", "strict_canary_in_progress"),
    source("donkey", "/m/0ffhf", "quadruped_equid", "specialized_indoor_only", ("barn_stable", "large_animal_clinic"), "coat_tone", "native_motion_guide_available"),
    source("cattle", "/m/01xq0k1", "quadruped_bovid", "specialized_indoor_only", ("barn_stable", "large_animal_clinic"), "coat_tone", "native_motion_guide_available", variants=("cow", "bull")),
    source("yak", "/m/01hhp3", "quadruped_bovid", "specialized_indoor_only", ("barn_stable", "large_animal_clinic"), "coat_tone", "strict_profile_and_guide_required"),
    source("pig", "/m/068zj", "quadruped_suid", "specialized_indoor_only", ("barn_stable", "veterinary_clinic"), "skin_coat_tone", "strict_profile_and_guide_required"),
    source("goat", "/m/03fwl", "quadruped_small_ungulate", "specialized_indoor_only", ("barn_stable", "veterinary_clinic"), "coat_tone", "strict_profile_and_guide_required"),
    source("sheep", "/m/07bgp", "quadruped_small_ungulate", "specialized_indoor_only", ("barn_stable", "veterinary_clinic"), "fleece_tone", "strict_profile_and_guide_required"),
    source("fowl_generic", "/m/025rv6n", "bird_terrestrial", "specialized_indoor_only", ("barn_coop", "aviary"), "plumage_tone", "bird_adapter_required"),
    source("chicken_rooster", "/m/09b5t", "bird_terrestrial", "specialized_indoor_only", ("barn_coop", "aviary"), "plumage_tone", "bird_adapter_required", variants=("hen", "rooster")),
    source("turkey", "/m/01rd7k", "bird_terrestrial", "specialized_indoor_only", ("barn_coop", "aviary"), "plumage_tone", "bird_adapter_required"),
    source("duck", "/m/09ddx", "bird_terrestrial", "specialized_indoor_only", ("barn_coop", "aviary", "indoor_waterfowl_habitat"), "plumage_tone", "bird_adapter_required"),
    source("goose", "/m/0dbvp", "bird_terrestrial", "specialized_indoor_only", ("barn_coop", "aviary", "indoor_waterfowl_habitat"), "plumage_tone", "bird_adapter_required"),
    source("roaring_big_cat", "/m/0cdnk", "quadruped_felid", "specialized_indoor_only", ("zoo_enclosure", "wildlife_clinic"), "coat_tone", "strict_profile_and_guide_required", variants=("lion", "tiger")),
    source("bird_generic", "/m/015p6", "bird_perching_flight", "apartment_conditional", ("apartment", "aviary", "wildlife_clinic"), "plumage_tone", "bird_adapter_required", variants=("songbird", "parrot")),
    source("pigeon_dove", "/m/0h0rv", "bird_perching_flight", "apartment_conditional", ("apartment", "aviary", "wildlife_clinic"), "plumage_tone", "bird_adapter_required", variants=("pigeon", "dove")),
    source("crow", "/m/04s8yn", "bird_perching_flight", "apartment_conditional", ("aviary", "wildlife_clinic"), "plumage_tone", "bird_adapter_required"),
    source("owl", "/m/09d5_", "bird_perching_flight", "specialized_indoor_only", ("aviary", "wildlife_clinic", "zoo_enclosure"), "plumage_tone", "bird_adapter_required"),
    source("gull", "/m/01dwxx", "bird_perching_flight", "specialized_indoor_only", ("aviary", "wildlife_clinic"), "plumage_tone", "bird_adapter_required"),
    source("wild_canid", "/m/01z5f", "quadruped_canid", "specialized_indoor_only", ("zoo_enclosure", "wildlife_clinic"), "coat_tone", "native_motion_guide_available", variants=("wolf", "fox")),
    source("rodent_generic", "/m/06hps", "quadruped_rodent", "apartment_conditional", ("apartment", "laboratory", "veterinary_clinic"), "coat_tone", "strict_profile_and_guide_required", variants=("rat", "mouse")),
    source("mouse", "/m/04rmv", "quadruped_rodent", "apartment_common", ("apartment", "laboratory", "veterinary_clinic"), "coat_tone", "strict_profile_and_guide_required"),
    source("chipmunk", "/m/02021", "quadruped_rodent", "apartment_conditional", ("wildlife_clinic", "laboratory"), "coat_tone", "strict_profile_and_guide_required"),
    source("insect_generic", "/m/03vt0", "insect_crawl_flight", "apartment_common", ("apartment", "laboratory", "insectarium"), "exoskeleton_tone", "insect_adapter_required"),
    source("cricket", "/m/09xqv", "insect_crawl_flight", "apartment_common", ("apartment", "laboratory", "insectarium"), "exoskeleton_tone", "insect_adapter_required"),
    source("mosquito", "/m/09f96", "insect_crawl_flight", "apartment_common", ("apartment", "laboratory", "insectarium"), "exoskeleton_tone", "insect_adapter_required"),
    source("housefly", "/m/0h2mp", "insect_crawl_flight", "apartment_common", ("apartment", "laboratory", "insectarium"), "exoskeleton_tone", "insect_adapter_required"),
    source("bee_wasp", "/m/01h3n", "insect_crawl_flight", "apartment_conditional", ("apartment", "laboratory", "insectarium"), "exoskeleton_tone", "insect_adapter_required", variants=("bee", "wasp")),
    source("frog", "/m/09ld4", "amphibian_hop", "apartment_conditional", ("terrarium", "laboratory", "zoo_enclosure"), "skin_tone", "amphibian_adapter_required"),
    source("snake", "/m/078jl", "serpent_slither", "apartment_conditional", ("terrarium", "laboratory", "zoo_enclosure"), "scale_tone", "serpent_adapter_required"),
    source("whale", "/m/032n05", "aquatic_cetacean", "specialized_indoor_only", ("indoor_aquarium",), "skin_tone", "aquatic_adapter_required"),
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descendants(root, by_id):
    result = set()
    pending = [root]
    while pending:
        node_id = pending.pop()
        if node_id in result:
            continue
        if node_id not in by_id:
            raise ValueError(f"ontology references missing node {node_id}")
        result.add(node_id)
        pending.extend(by_id[node_id].get("child_ids", []))
    return result


def nearest_sources(node_id, parents, source_by_node):
    pending = deque([(node_id, 0)])
    seen = {node_id}
    found_distance = None
    found = set()
    while pending:
        current, distance = pending.popleft()
        if found_distance is not None and distance > found_distance:
            break
        if current != node_id and current in source_by_node:
            found_distance = distance
            found.add(source_by_node[current]["source_id"])
            continue
        for parent in sorted(parents.get(current, ())):
            if parent not in seen:
                seen.add(parent)
                pending.append((parent, distance + 1))
    return sorted(found)


def build_registry(ontology_path, commit_path):
    ontology_path = Path(ontology_path).resolve()
    if sha256_file(ontology_path) != ONTOLOGY_SHA256:
        raise ValueError("AudioSet ontology snapshot hash changed")
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))
    commit = json.loads(Path(commit_path).read_text(encoding="utf-8"))
    if commit.get("sha") != ONTOLOGY_REVISION:
        raise ValueError("AudioSet ontology commit revision changed")
    by_id = {node["id"]: node for node in ontology}
    if len(by_id) != len(ontology) or ANIMAL_ID not in by_id:
        raise ValueError("AudioSet ontology IDs are missing or duplicated")
    animal_ids = descendants(ANIMAL_ID, by_id)
    parents = defaultdict(set)
    for parent_id in animal_ids:
        for child_id in by_id[parent_id].get("child_ids", []):
            if child_id in animal_ids:
                parents[child_id].add(parent_id)
    source_by_node = {item["audioset_node_id"]: item for item in SOURCE_SPECS}
    if len(source_by_node) != len(SOURCE_SPECS):
        raise ValueError("visible AudioSet source nodes are duplicated")
    unknown_sources = sorted(set(source_by_node) - animal_ids)
    if unknown_sources:
        raise ValueError(f"visible sources are outside Animal: {unknown_sources}")

    nodes = []
    events_by_source = defaultdict(list)
    for node_id in sorted(animal_ids):
        node = by_id[node_id]
        if node_id in CONTAINER_IDS:
            role = "ontology_container"
            mapped_sources = []
        elif node_id in source_by_node:
            role = "visible_source"
            mapped_sources = [source_by_node[node_id]["source_id"]]
        else:
            role = "acoustic_event"
            mapped_sources = nearest_sources(node_id, parents, source_by_node)
            if not mapped_sources:
                raise ValueError(f"acoustic event has no visible source mapping: {node_id}")
        blacklisted = "blacklist" in node.get("restrictions", [])
        record = {
            "audioset_id": node_id,
            "name": node["name"],
            "role": role,
            "parent_ids": sorted(parents.get(node_id, ())),
            "child_ids": sorted(
                child for child in node.get("child_ids", []) if child in animal_ids
            ),
            "restrictions": sorted(node.get("restrictions", [])),
            "dataset_eligible": not blacklisted,
            "mapped_source_ids": mapped_sources,
        }
        nodes.append(record)
        if role == "acoustic_event":
            for source_id in mapped_sources:
                events_by_source[source_id].append(record)

    source_records = []
    for spec in SOURCE_SPECS:
        node = by_id[spec["audioset_node_id"]]
        actions = MOTION_FAMILIES[spec["motion_family"]]
        event_records = sorted(
            events_by_source.get(spec["source_id"], []),
            key=lambda item: (item["name"], item["audioset_id"]),
        )
        source_records.append(
            {
                **spec,
                "audioset_name": node["name"],
                "route": "strict_native_i2i_i23d_rig_v1",
                "required_actions": list(actions),
                "instance_attribute_contract": {
                    "maximum_values_per_randomized_attribute": 3,
                    "required_domains": [
                        "size",
                        "body_build",
                        "life_stage",
                        spec["appearance_attribute"],
                    ],
                    "sampling": "code_only_complete_prompt_and_json",
                    "relative_edit_history_forbidden": True,
                },
                "acoustic_events": [
                    {
                        "audioset_id": spec["audioset_node_id"],
                        "name": node["name"],
                        "dataset_eligible": "blacklist" not in node.get("restrictions", []),
                        "kind": "source_class",
                    }
                ]
                + [
                    {
                        "audioset_id": event["audioset_id"],
                        "name": event["name"],
                        "dataset_eligible": event["dataset_eligible"],
                        "kind": "sound_event",
                    }
                    for event in event_records
                ],
                "audio_schedule_contract": {
                    "species_authenticated_clip_required": True,
                    "short_event_repeat_threshold": True,
                    "minimum_silence_between_repeats": True,
                    "event_timestamps_recorded": True,
                },
                "formal_dataset_registration_authorized": False,
            }
        )

    accounted = {node["audioset_id"] for node in nodes}
    if accounted != animal_ids:
        raise ValueError("AudioSet Animal coverage is incomplete")
    try:
        ontology_artifact_path = ontology_path.resolve().relative_to(SPEAR_ROOT).as_posix()
        license_snapshot_path = (
            Path(commit_path)
            .with_name("audioset_ontology_README_v1.md")
            .resolve()
            .relative_to(SPEAR_ROOT)
            .as_posix()
        )
    except ValueError as error:
        raise ValueError("AudioSet ontology inputs must stay inside the SPEAR checkout") from error
    return {
        "schema": SCHEMA,
        "state_classification": "research_candidate",
        "ontology_snapshot": {
            "source_url": "https://github.com/audioset/ontology",
            "revision": ONTOLOGY_REVISION,
            "artifact_path": ontology_artifact_path,
            "sha256": ONTOLOGY_SHA256,
            "license": "CC-BY-SA-4.0",
            "license_snapshot_path": license_snapshot_path,
        },
        "scope_contract": {
            "meaning": "all visible source entities below the official AudioSet Animal node, classified by physically plausible indoor venue",
            "apartment_common": "ordinary apartment occurrence is plausible",
            "apartment_conditional": "requires pet, intrusion, terrarium, cage, or rescue context",
            "specialized_indoor_only": "requires barn, stable, clinic, aviary, zoo, insectarium, or aquarium; do not place in a normal apartment",
            "sound_event_nodes_do_not_create_duplicate_meshes": True,
        },
        "motion_family_contracts": {
            family: {
                "required_actions": list(actions),
                "separate_pose_direction_rig_and_deformation_gates": True,
                "formal_dataset_registration_authorized": False,
            }
            for family, actions in sorted(MOTION_FAMILIES.items())
        },
        "summary": {
            "audioset_animal_node_count": len(nodes),
            "visible_source_count": len(source_records),
            "acoustic_event_node_count": sum(node["role"] == "acoustic_event" for node in nodes),
            "ontology_container_count": sum(node["role"] == "ontology_container" for node in nodes),
            "apartment_common_count": sum(source["indoor_scope"] == "apartment_common" for source in source_records),
            "apartment_conditional_count": sum(source["indoor_scope"] == "apartment_conditional" for source in source_records),
            "specialized_indoor_only_count": sum(source["indoor_scope"] == "specialized_indoor_only" for source in source_records),
        },
        "sources": source_records,
        "ontology_nodes": nodes,
        "formal_dataset_registration_authorized": False,
    }


def render_markdown(registry):
    lines = [
        "# AudioSet 室内动物声源覆盖表（严格版 v1）",
        "",
        f"官方 revision：`{registry['ontology_snapshot']['revision']}`；可视声源：{registry['summary']['visible_source_count']} 类。",
        "声学事件（如 bark、meow、neigh）挂到动物实体上，不重复生成网格。所有条目当前均为 research candidate。",
        "",
        "| 声源 | AudioSet 类 | 室内范围 | 运动家族 | 动作 | 当前状态 |",
        "|---|---|---|---|---|---|",
    ]
    for item in registry["sources"]:
        lines.append(
            "| {source_id} | {audioset_name} (`{audioset_node_id}`) | "
            "{indoor_scope} | {motion_family} | {actions} | {status} |".format(
                source_id=item["source_id"],
                audioset_name=item["audioset_name"].replace("|", "/"),
                audioset_node_id=item["audioset_node_id"],
                indoor_scope=item["indoor_scope"],
                motion_family=item["motion_family"],
                actions=" / ".join(item["required_actions"]),
                status=item["implementation_status"],
            )
        )
    lines.extend(
        [
            "",
            "普通 Apartment 只能自动采样 `apartment_common`；`apartment_conditional` 必须由场景语义显式允许；`specialized_indoor_only` 禁止进入普通 Apartment。",
            "鸟类、昆虫、蛇、青蛙和鲸分别使用自己的运动适配器，不能套四足 Walking。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    data = root / "data/controlled_source_attributes_v1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ontology", type=Path, default=data / "audioset_ontology_v1.json")
    parser.add_argument("--commit", type=Path, default=data / "audioset_ontology_commit_master.json")
    parser.add_argument("--output", type=Path, default=data / "audioset_indoor_animal_source_registry_v1.json")
    parser.add_argument("--markdown", type=Path, default=root / "docs/audioset_indoor_animal_source_coverage.md")
    return parser.parse_args()


def main():
    args = parse_args()
    registry = build_registry(args.ontology, args.commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(registry), encoding="utf-8")
    print(json.dumps(registry["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
