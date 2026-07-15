#!/usr/bin/env python3
"""Compile the audited Quaternius template registry into attribute profiles.

The catalog contains species semantics only.  Runtime GLB paths, hashes, action
names, and license evidence remain authoritative in the audited template
registry and are copied from there after re-hashing the registry itself.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence


SPEAR_ROOT = Path(__file__).resolve().parents[1]
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import controlled_source_asset_schema as contracts  # noqa: E402


CATALOG_SCHEMA = "avengine_stable_template_attribute_catalog_v1"
MANIFEST_SCHEMA = "avengine_stable_template_attribute_profile_manifest_v1"
FLUX2_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise contracts.ContractError(f"{label} must be an object")
    return value


def _require_three_strings(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != 3
    ):
        raise contracts.ContractError(f"{label} must contain three unique strings")
    return list(value)


def _csv(values: Any, label: str) -> str:
    if not isinstance(values, list) or not values or any(
        not isinstance(item, str) or not item or "," in item for item in values
    ):
        raise contracts.ContractError(f"{label} must be a non-empty string list")
    return ",".join(values)


def _artifact_from_registry(runtime: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(runtime["path"])).resolve()
    try:
        relative = path.relative_to(SPEAR_ROOT)
    except ValueError as error:
        raise contracts.ContractError(
            f"stable runtime must be under SPEAR root: {path}"
        ) from error
    if not path.is_file():
        raise contracts.ContractError(f"stable runtime is missing: {path}")
    if (
        sha256_file(path) != runtime.get("sha256")
        or path.stat().st_size != runtime.get("size_bytes")
    ):
        raise contracts.ContractError(f"stable runtime no longer matches registry: {path}")
    return {
        "root_id": "spear_repo",
        "path": relative.as_posix(),
        "sha256": runtime["sha256"],
        "size_bytes": runtime["size_bytes"],
    }


def _ordered_qa(attribute: str, values: Sequence[str], subject: str) -> dict[str, Any]:
    labels = {value: value.replace("_", " ") for value in values}
    questions = {
        "size": f"Which {subject} is larger?",
        "body_build": f"Which {subject} has the stockier body?",
        "coat_tone": f"Which {subject} has the darker coat tone?",
        "life_stage": f"Which {subject} is at the later life stage?",
    }
    return {
        "kind": "ordered",
        "label": attribute.replace("_", " "),
        "value_order": list(values),
        "value_labels": labels,
        "comparison_question": questions[attribute],
    }


def build_profile(
    catalog_entry: Mapping[str, Any], registry_entry: Mapping[str, Any]
) -> dict[str, Any]:
    template_id = str(catalog_entry["template_id"])
    if registry_entry.get("template_id") != template_id:
        raise contracts.ContractError(f"template mismatch for {template_id}")
    if registry_entry.get("state_classification") != "research_candidate":
        raise contracts.ContractError(f"unexpected template state for {template_id}")
    if registry_entry.get("formal_dataset_registration_authorized") is not False:
        raise contracts.ContractError(f"template must remain non-formal: {template_id}")
    if set(registry_entry.get("actions", [])) != {"Walking", "Idle"}:
        raise contracts.ContractError(f"template lacks exact Walk/Idle: {template_id}")

    coat_values = _require_three_strings(
        catalog_entry.get("coat_tone_values"), f"{template_id}.coat_tone_values"
    )
    coat_labels = _require_three_strings(
        catalog_entry.get("coat_tone_labels"), f"{template_id}.coat_tone_labels"
    )
    coat_materials = _csv(
        catalog_entry.get("coat_material_names"), f"{template_id}.coat_material_names"
    )
    muzzle_materials = _csv(
        catalog_entry.get("muzzle_material_names", ["none"]),
        f"{template_id}.muzzle_material_names",
    )
    species = str(catalog_entry["species"])
    breed = str(catalog_entry["breed"])
    subject = str(catalog_entry["subject_label"])
    fixed = dict(_require_mapping(catalog_entry["fixed_attributes"], "fixed_attributes"))
    expected_fixed = {"coat_length", "coat_pattern", "ear_shape", "tail_shape"}
    if set(fixed) != expected_fixed or any(not isinstance(v, str) or not v for v in fixed.values()):
        raise contracts.ContractError(
            f"{template_id}.fixed_attributes must be exactly {sorted(expected_fixed)}"
        )

    domains = {
        "size": ["small", "medium", "large"],
        "body_build": ["slim", "standard", "stocky"],
        "coat_tone": coat_values,
        "life_stage": ["young", "adult", "senior"],
    }
    value_labels = {
        "species": {species: species.replace("_", " ")},
        "breed": {breed: breed.replace("_", " ")},
        **{name: {value: value.replace("_", " ")} for name, value in fixed.items()},
        "size": {"small": "small", "medium": "medium-sized", "large": "large"},
        "body_build": {
            "slim": "slim",
            "standard": "standard",
            "stocky": "stocky",
        },
        "coat_tone": dict(zip(coat_values, coat_labels)),
        "life_stage": {"young": "young", "adult": "adult", "senior": "senior"},
    }
    torso_tokens = str(catalog_entry.get("torso_group_tokens_csv", "body,back,torso"))
    head_tokens = str(
        catalog_entry.get("head_group_tokens_csv", "head,neck,ear,eye,mouth,muzzle")
    )
    physical = _require_mapping(catalog_entry["physical_profile"], "physical_profile")
    acoustic = _require_mapping(catalog_entry["acoustic_profile"], "acoustic_profile")
    motion_family = str(catalog_entry["motion_family"])
    direction = _require_mapping(registry_entry["direction"], "registry direction")
    cardinal_yaw = direction.get("cardinal_yaw_deg")
    if cardinal_yaw not in {-90, 0, 90, 180}:
        raise contracts.ContractError(
            f"{template_id} has a non-cardinal runtime direction transform"
        )
    if (
        direction.get("runtime_front_axis") != "positive_x"
        or direction.get("automatic_fine_yaw_inference") is not False
    ):
        raise contracts.ContractError(f"{template_id} direction contract is unsafe")

    profile: dict[str, Any] = {
        "schema": contracts.PROFILE_SCHEMA,
        "profile_schema_id": f"{template_id}_controlled_attributes_v1",
        "profile_revision": "2026_07_15_v1_stable_template_ofat",
        "asset_class": "animal",
        "lineage_group_id": template_id,
        "state_classification": "research_candidate",
        "taxonomy": {"species": species, "breed": breed},
        "base_template": {
            "template_id": template_id,
            "kind": "stable_animal_template",
            "artifact": _artifact_from_registry(registry_entry["runtime_glb"]),
            "provenance_status": "verified",
            "usage_scope": "research_candidate",
        },
        "fixed_attributes": fixed,
        "sampled_attribute_domains": domains,
        "forbidden_combinations": [],
        "generation_contract": {
            "route": "stable_animal_template_v1",
            "plan_schema": "stable_animal_instance_plan_v1",
            "prompt_template_id": f"{template_id}_complete_attributes_v1",
            "positive_template": (
                "A {size} {life_stage} {breed} {species} with a {body_build} "
                "body, a {coat_tone} {coat_length} coat with {coat_pattern}, "
                "{ear_shape} ears, and a {tail_shape} tail."
            ),
            "pose_guard_prompt": (
                "Preserve the audited native topology, skeleton, Walk/Idle actions, "
                "four grounded separated legs, level torso, and separated tail."
            ),
            "negative_prompt": (
                "merged legs, hidden feet, floating feet, holes, open belly, stretched "
                "geometry, extra limbs, tail touching hind legs, changed skeleton, text"
            ),
            "value_labels": value_labels,
            "model_revisions": {"flux2": FLUX2_REVISION},
            "flux_reference_policy": "qa_and_optional_semantic_texture_detail_only",
            "attribute_bindings": {
                "size": {
                    "operation": "uniform_actor_scale_from_physical_profile_v1",
                    "values": {
                        value: {
                            "scale_source": "target_physical_profile",
                            "template_cardinal_yaw_deg": cardinal_yaw,
                        }
                        for value in domains["size"]
                    },
                },
                "body_build": {
                    "operation": "semantic_torso_girth_preserve_topology_and_skin_v1",
                    "values": {
                        "slim": {
                            "torso_girth_scale": 0.92,
                            "torso_group_tokens_csv": torso_tokens,
                            "coordinate_mode": "semantic_head_torso_axis_v1",
                        },
                        "standard": {
                            "torso_girth_scale": 1.0,
                            "torso_group_tokens_csv": torso_tokens,
                            "coordinate_mode": "semantic_head_torso_axis_v1",
                        },
                        "stocky": {
                            "torso_girth_scale": 1.08,
                            "torso_group_tokens_csv": torso_tokens,
                            "coordinate_mode": "semantic_head_torso_axis_v1",
                        },
                    },
                },
                "coat_tone": {
                    "operation": "semantic_coat_luminance_preserve_pattern_pbr_v1",
                    "values": {
                        coat_values[0]: {
                            "coat_luminance_gain": 1.18,
                            "coat_material_names_csv": coat_materials,
                            "surface_mode": "solid_material_pbr",
                        },
                        coat_values[1]: {
                            "coat_luminance_gain": 1.0,
                            "coat_material_names_csv": coat_materials,
                            "surface_mode": "solid_material_pbr",
                        },
                        coat_values[2]: {
                            "coat_luminance_gain": 0.80,
                            "coat_material_names_csv": coat_materials,
                            "surface_mode": "solid_material_pbr",
                        },
                    },
                },
                "life_stage": {
                    "operation": "semantic_age_morph_and_muzzle_tone_preserve_skin_v1",
                    "values": {
                        "young": {
                            "head_scale": 1.06,
                            "muzzle_gray_mix": 0.0,
                            "muzzle_gray_target": 0.52,
                            "senior_coat_desaturation": 0.0,
                            "head_group_tokens_csv": head_tokens,
                            "muzzle_material_names_csv": muzzle_materials,
                        },
                        "adult": {
                            "head_scale": 1.0,
                            "muzzle_gray_mix": 0.0,
                            "muzzle_gray_target": 0.52,
                            "senior_coat_desaturation": 0.0,
                            "head_group_tokens_csv": head_tokens,
                            "muzzle_material_names_csv": muzzle_materials,
                        },
                        "senior": {
                            "head_scale": 0.98,
                            "muzzle_gray_mix": 0.40,
                            "muzzle_gray_target": 0.52,
                            "senior_coat_desaturation": 0.18,
                            "head_group_tokens_csv": head_tokens,
                            "muzzle_material_names_csv": muzzle_materials,
                        },
                    },
                },
            },
        },
        "target_physical_profiles": {
            "profile_id": f"{template_id}_physical_candidate_v1",
            "control_attribute": "size",
            "measurement": str(physical["measurement"]),
            "mode": "relative_to_profile_reference",
            "reference_value_cm": physical["reference_value_cm"],
            "reference_provenance": {
                "status": "provisional",
                "source_id": f"{template_id}_calibration_pending_v1",
                "artifact": None,
                "notes": (
                    "Template-relative canary target; replace with a license-snapshotted "
                    "species reference before formal registration."
                ),
            },
            "values": {
                "small": {"scale_ratio": 0.9, "tolerance_cm": physical["tolerance_cm"]},
                "medium": {"scale_ratio": 1.0, "tolerance_cm": physical["tolerance_cm"]},
                "large": {"scale_ratio": 1.1, "tolerance_cm": physical["tolerance_cm"]},
            },
        },
        "rig_profile": {
            "profile_id": f"{motion_family}_quaternius_v1",
            "skeleton_family": "quaternius_animal_armature",
            "actions": ["Walking", "Idle"],
            "front_axis": "positive_x",
        },
        "acoustic_profile": {
            "profile_id": str(acoustic["profile_id"]),
            "default_event_class": str(acoustic["default_event_class"]),
            "allowed_event_classes": list(acoustic["allowed_event_classes"]),
            "selection_attributes": ["species", "breed", "size", "life_stage"],
        },
        "locked_attributes": [
            "species",
            "breed",
            "coat_length",
            "coat_pattern",
            "ear_shape",
            "tail_shape",
        ],
        "qa_contract": {
            "subject_label": subject,
            "attributes": {
                attribute: _ordered_qa(attribute, values, subject)
                for attribute, values in domains.items()
            },
        },
    }
    return contracts.validate_attribute_profile(profile)


def build_profiles(catalog: Mapping[str, Any], registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    if catalog.get("schema") != CATALOG_SCHEMA:
        raise contracts.ContractError(f"catalog schema must be {CATALOG_SCHEMA}")
    entries = catalog.get("entries")
    if not isinstance(entries, list) or not entries:
        raise contracts.ContractError("catalog entries must be a non-empty list")
    registry_entries = registry.get("entries")
    if not isinstance(registry_entries, list) or not registry_entries:
        raise contracts.ContractError("template registry entries are missing")
    catalog_map = {entry.get("template_id"): entry for entry in entries if isinstance(entry, dict)}
    registry_map = {
        entry.get("template_id"): entry for entry in registry_entries if isinstance(entry, dict)
    }
    if len(catalog_map) != len(entries) or set(catalog_map) != set(registry_map):
        raise contracts.ContractError(
            "catalog must cover every stable registry template exactly once"
        )
    return [
        build_profile(catalog_map[template_id], registry_map[template_id])
        for template_id in sorted(registry_map)
    ]


def publish(profiles: Sequence[Mapping[str, Any]], output_dir: Path, inputs: Mapping[str, Any]) -> None:
    if output_dir.exists() or output_dir.is_symlink():
        raise contracts.ContractError(f"refusing to replace output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        records = []
        for profile in profiles:
            path = output_dir / f"{profile['profile_schema_id']}.json"
            contracts.write_json_no_replace(path, profile)
            records.append(
                {
                    "profile_schema_id": profile["profile_schema_id"],
                    "profile_sha256": contracts.profile_sha256(profile),
                    "path": str(path.resolve()),
                    "file_sha256": sha256_file(path),
                }
            )
        manifest: dict[str, Any] = {
            "schema": MANIFEST_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "state_classification": "research_candidate",
            "formal_dataset_registration_authorized": False,
            "inputs": dict(inputs),
            "profile_count": len(records),
            "profiles": records,
        }
        manifest["manifest_sha256"] = contracts.manifest_sha256(manifest)
        contracts.write_json_no_replace(output_dir / "profile_manifest.json", manifest)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--catalog", required=True, type=Path)
    result.add_argument("--template-registry", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    catalog_path = args.catalog.resolve()
    registry_path = args.template_registry.resolve()
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        expected_registry_sha = catalog.get("template_registry_sha256")
        observed_registry_sha = sha256_file(registry_path)
        if expected_registry_sha != observed_registry_sha:
            raise contracts.ContractError(
                "template registry hash differs from the catalog authority"
            )
        profiles = build_profiles(catalog, registry)
        publish(
            profiles,
            args.output_dir.resolve(),
            {
                "catalog": str(catalog_path),
                "catalog_sha256": sha256_file(catalog_path),
                "template_registry": str(registry_path),
                "template_registry_sha256": observed_registry_sha,
            },
        )
    except (OSError, ValueError, KeyError, contracts.ContractError) as error:
        print(f"STABLE_TEMPLATE_PROFILE_BUILD_FAILED {error}", file=sys.stderr)
        return 2
    print(
        f"STABLE_TEMPLATE_PROFILE_BUILD_OK profiles={len(profiles)} output={args.output_dir.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
