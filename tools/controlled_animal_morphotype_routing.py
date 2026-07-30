#!/usr/bin/env python3
"""Trait-driven route, budget and gate selection for animal sound sources.

The hardened route previously grew one prompt template per breed
(``..._v1_shiba_inu``, ``..._v2_reconstruction_safe_british_shorthair``,
``..._local_tail_stump_edit_v1``).  Each template carried real knowledge --
limb-separation corridors, tail singularity, localized anatomical edits --
but keyed it to a breed name, so the next breed inherited none of it.

This module re-keys that knowledge onto declared morphotype traits.  A roster
row declares what an animal *is*; the trait contract turns that into the
prompt guards it needs, the gates it must pass, and how many exploration
seeds its risk tier is worth.  Adding an animal sound source is a roster row,
never a production code change.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_source_asset_schema as contracts


REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_DIR = REPO_ROOT / "data/controlled_source_attributes_v1/contracts"
TRAIT_CONTRACT_PATH = CONTRACT_DIR / "animal_morphotype_traits_v1.json"
ROSTER_PATH = CONTRACT_DIR / "animal_acquisition_roster_v1.json"

TRAIT_CONTRACT_SCHEMA = "avengine_animal_morphotype_trait_contract_v1"
ROSTER_SCHEMA = "avengine_animal_acquisition_roster_v1"
ROUTING_PLAN_SCHEMA = "avengine_animal_morphotype_routing_plan_v1"

ACQUISITION_STATES = frozenset({"shipped", "queued", "deferred", "rejected"})


class RoutingError(ValueError):
    """Raised when the roster or the trait contract cannot be honoured."""


def json_sha256(value: Any) -> str:
    return hashlib.sha256(contracts.canonical_json(value).encode("utf-8")).hexdigest()


def _load_object(path: Path, schema: str, label: str) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise RoutingError(f"{label} is missing: {path}")
    value = contracts.load_json(path)
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise RoutingError(f"{label} identity changed: {path}")
    return value


def load_trait_contract(path: Path = TRAIT_CONTRACT_PATH) -> dict[str, Any]:
    contract = _load_object(path, TRAIT_CONTRACT_SCHEMA, "morphotype trait contract")
    axes = contract.get("axes")
    order = contract.get("composition_order")
    tier_policy = contract.get("tier_policy")
    if not isinstance(axes, dict) or not axes:
        raise RoutingError("trait contract declares no axes")
    if not isinstance(order, list) or sorted(order) != sorted(axes):
        raise RoutingError("composition_order must cover every axis exactly once")
    if not isinstance(tier_policy, dict) or not isinstance(tier_policy.get("tiers"), list):
        raise RoutingError("trait contract declares no tier policy")
    for axis, spec in axes.items():
        values = spec.get("values")
        if not isinstance(values, dict) or not values:
            raise RoutingError(f"axis declares no values: {axis}")
        for value, body in values.items():
            for field in ("reconstruction_risk", "motion_donor_risk"):
                score = body.get(field)
                if isinstance(score, bool) or not isinstance(score, int) or score < 0:
                    raise RoutingError(f"{axis}.{value}.{field} must be a non-negative int")
            for field in ("pose_guard_fragments", "negative_prompt_terms", "required_gates"):
                if not isinstance(body.get(field), list) or not all(
                    isinstance(item, str) and item.strip() for item in body[field]
                ):
                    raise RoutingError(f"{axis}.{value}.{field} must be a list of non-empty strings")
            if not isinstance(body.get("evidence"), str) or not body["evidence"].strip():
                raise RoutingError(f"{axis}.{value} must cite the evidence it was scored from")
    return contract


def load_roster(path: Path = ROSTER_PATH) -> dict[str, Any]:
    roster = _load_object(path, ROSTER_SCHEMA, "animal acquisition roster")
    entries = roster.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RoutingError("roster declares no entries")
    seen: set[str] = set()
    for entry in entries:
        key = entry.get("asset_key")
        if not isinstance(key, str) or not key:
            raise RoutingError("roster entry is missing asset_key")
        if key in seen:
            raise RoutingError(f"duplicate roster asset_key: {key}")
        seen.add(key)
        taxonomy = entry.get("taxonomy")
        if (
            not isinstance(taxonomy, dict)
            or not isinstance(taxonomy.get("species"), str)
            or not isinstance(taxonomy.get("breed"), str)
        ):
            raise RoutingError(f"roster entry has no species/breed taxonomy: {key}")
        if entry.get("acquisition_state") not in ACQUISITION_STATES:
            raise RoutingError(f"unsupported acquisition_state on {key}")
        height = entry.get("shoulder_height_cm")
        if isinstance(height, bool) or not isinstance(height, (int, float)) or height <= 0:
            raise RoutingError(f"roster entry needs a positive shoulder_height_cm: {key}")
        acoustic = entry.get("acoustic")
        if (
            not isinstance(acoustic, dict)
            or not isinstance(acoustic.get("audioset_classes"), list)
            or not acoustic["audioset_classes"]
            or not isinstance(acoustic.get("event_classes"), list)
            or not acoustic["event_classes"]
        ):
            raise RoutingError(
                f"roster entry must declare the AudioSet classes and event classes it "
                f"exists to voice: {key}"
            )
        if not isinstance(entry.get("traits"), dict):
            raise RoutingError(f"roster entry declares no traits: {key}")
    return roster


def entry_for(asset_key: str, roster: Mapping[str, Any] | None = None) -> dict[str, Any]:
    roster = roster if roster is not None else load_roster()
    for entry in roster["entries"]:
        if entry["asset_key"] == asset_key:
            return copy.deepcopy(entry)
    raise RoutingError(f"asset_key is not on the roster: {asset_key}")


def _tier_for(risk_total: int, tier_policy: Mapping[str, Any]) -> Mapping[str, Any]:
    for tier in tier_policy["tiers"]:
        if risk_total <= tier["maximum_risk_total"]:
            return tier
    raise RoutingError(f"no tier covers risk total {risk_total}")


def resolve_traits(
    traits: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[tuple[str, str, Mapping[str, Any]]]:
    """Return (axis, value, body) in the contract's declared composition order."""
    axes = contract["axes"]
    required = {axis for axis, spec in axes.items() if spec.get("required")}
    declared = set(traits)
    missing = sorted(required - declared)
    if missing:
        raise RoutingError(f"undeclared required traits: {missing}")
    unknown = sorted(declared - set(axes))
    if unknown:
        raise RoutingError(f"traits not in the contract: {unknown}")
    resolved: list[tuple[str, str, Mapping[str, Any]]] = []
    for axis in contract["composition_order"]:
        if axis not in traits:
            continue
        value = traits[axis]
        values = axes[axis]["values"]
        if value not in values:
            raise RoutingError(f"unsupported {axis} value: {value!r}")
        resolved.append((axis, value, values[value]))
    return resolved


def compose_prompt_blocks(
    traits: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Compose the guard prompt and negative terms a morphotype requires.

    Order follows ``composition_order`` and duplicates are dropped on first
    occurrence, so the same declared traits always produce byte-identical
    blocks regardless of how the roster row was written.
    """
    fragments: list[str] = []
    negatives: list[str] = []
    attribution: list[dict[str, Any]] = []
    for axis, value, body in resolve_traits(traits, contract):
        contributed_fragments = [f for f in body["pose_guard_fragments"] if f not in fragments]
        contributed_negatives = [t for t in body["negative_prompt_terms"] if t not in negatives]
        fragments.extend(contributed_fragments)
        negatives.extend(contributed_negatives)
        if contributed_fragments or contributed_negatives:
            attribution.append(
                {
                    "axis": axis,
                    "value": value,
                    "pose_guard_fragments": contributed_fragments,
                    "negative_prompt_terms": contributed_negatives,
                    "evidence": body["evidence"],
                }
            )
    return {
        "pose_guard_fragments": fragments,
        "negative_prompt_terms": negatives,
        "trait_attribution": attribution,
    }


def routing_plan(
    asset_key: str,
    *,
    roster: Mapping[str, Any] | None = None,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    contract = contract if contract is not None else load_trait_contract()
    entry = entry_for(asset_key, roster)
    resolved = resolve_traits(entry["traits"], contract)

    reconstruction_risk = sum(body["reconstruction_risk"] for _, _, body in resolved)
    motion_donor_risk = sum(body["motion_donor_risk"] for _, _, body in resolved)
    risk_total = reconstruction_risk + motion_donor_risk
    tier = _tier_for(risk_total, contract["tier_policy"])

    gates: list[str] = []
    for _, _, body in resolved:
        for gate in body["required_gates"]:
            if gate not in gates:
                gates.append(gate)

    # The readiness checklist requires an angled canonical view to be declared
    # rather than discovered, so heading estimation can cross-check it.
    view_yaw: list[dict[str, Any]] = []
    for axis, value, body in resolved:
        window = body.get("declared_view_yaw_deg")
        if window is not None:
            view_yaw.append({"axis": axis, "value": value, "permitted_yaw_deg": window})

    owner_rule = contract["tier_policy"]["owner_budget_decision_required_when"]
    owner_budget_required = (
        motion_donor_risk >= owner_rule["motion_donor_risk_total_at_least"]
    )

    plan = {
        "schema": ROUTING_PLAN_SCHEMA,
        "asset_key": asset_key,
        "taxonomy": entry["taxonomy"],
        "wave": entry.get("wave"),
        "acquisition_state": entry["acquisition_state"],
        "shoulder_height_cm": entry["shoulder_height_cm"],
        "acoustic": entry["acoustic"],
        "traits": {axis: value for axis, value, _ in resolved},
        "risk": {
            "reconstruction_risk": reconstruction_risk,
            "motion_donor_risk": motion_donor_risk,
            "risk_total": risk_total,
            "tier": tier["tier"],
        },
        "budget": {
            "exploration_seed_count": tier["exploration_seed_count"],
            "preview_only_triage_required": tier["preview_only_triage_required"],
            "owner_budget_decision_required": owner_budget_required,
            "owner_budget_decision_semantics": owner_rule["semantics"],
        },
        "required_gates": gates,
        "declared_view_yaw": view_yaw,
        "prompt_blocks": compose_prompt_blocks(entry["traits"], contract),
        "trait_contract_id": contract["contract_id"],
    }
    plan["plan_sha256"] = json_sha256(plan)
    return plan


def wave_plans(
    wave: str,
    *,
    include_shipped: bool = False,
    roster: Mapping[str, Any] | None = None,
    contract: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    roster = roster if roster is not None else load_roster()
    contract = contract if contract is not None else load_trait_contract()
    plans = []
    for entry in roster["entries"]:
        if entry.get("wave") != wave:
            continue
        if entry["acquisition_state"] == "shipped" and not include_shipped:
            continue
        if entry["acquisition_state"] in {"deferred", "rejected"}:
            continue
        plans.append(routing_plan(entry["asset_key"], roster=roster, contract=contract))
    if not plans:
        raise RoutingError(f"wave has no acquirable entries: {wave}")
    # Cheapest first: a low-tier asset that ships early de-risks the batch.
    plans.sort(key=lambda plan: (plan["risk"]["risk_total"], plan["asset_key"]))
    return plans


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--asset-key", help="single roster asset key")
    group.add_argument("--wave", help="plan every acquirable entry in a wave")
    parser.add_argument(
        "--include-shipped",
        action="store_true",
        help="keep already-shipped entries in a wave plan (backtest use)",
    )
    parser.add_argument("--output", type=Path, help="write the plan JSON here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.asset_key:
        payload: Any = routing_plan(args.asset_key)
    else:
        payload = {
            "schema": "avengine_animal_morphotype_wave_plan_v1",
            "wave": args.wave,
            "plans": wave_plans(args.wave, include_shipped=args.include_shipped),
        }
    if args.output:
        contracts.write_json_no_replace(args.output, payload)
    else:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
