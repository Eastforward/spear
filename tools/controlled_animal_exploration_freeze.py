#!/usr/bin/env python3
"""Bounded seed exploration that ends in a frozen one-shot request.

The no-seed-lottery policy forbids rerolling a seed after seeing the output.
That rule is right for the shipped asset and wrong for the search that finds
it: the Corgi took thirteen sequential workspaces over roughly forty-six
hours, each one a fresh "declared request" whose only real content was a
different prompt clause.  The search happened anyway; it was just recorded in
directory names instead of in evidence.

This module makes the search explicit and bounded:

  plan    -- declare the whole candidate set up front.  The seeds come from a
             sha256 ladder over the plan id, so the operator picks the plan
             and the count but never an individual seed, and "kept rolling
             until it worked" is excluded by construction.
  record  -- hash every generated candidate, including the ones that failed.
  freeze  -- select exactly one index with a criterion drawn from the declared
             gate vocabulary, and emit a frozen one-shot request carrying that
             seed.  The shipped bytes are then produced by the untouched
             one-shot route.
  verify  -- prove the frozen seed reproduces the selected candidate bitwise.

Net effect on provenance: the search size and the criterion that ended it
become machine-readable, which they were not before.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_animal_morphotype_routing as routing
from tools import controlled_animal_one_shot_policy as one_shot
from tools import controlled_source_asset_schema as contracts


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    REPO_ROOT
    / "data/controlled_source_attributes_v1/contracts/animal_exploration_then_freeze_v1.json"
)
POLICY_SCHEMA = "avengine_controlled_animal_exploration_then_freeze_policy_v1"
POLICY_ID = "animal_exploration_then_freeze_v1"

EXPLORATION_PLAN_SCHEMA = "avengine_controlled_animal_exploration_plan_v1"
EXPLORATION_BATCH_SCHEMA = "avengine_controlled_animal_exploration_batch_v1"
FROZEN_DECLARATION_SCHEMA = "avengine_controlled_animal_frozen_request_declaration_v1"
REPRODUCIBILITY_SCHEMA = "avengine_controlled_animal_frozen_reproducibility_v1"

EXPLORATION_USAGE_SCOPE = "exploration_only_not_shippable"
SEED_LADDER_ALGORITHM = "sha256_plan_id_ordinal_ladder_v1"


class ExplorationError(ValueError):
    """Raised when exploration/freeze evidence violates the frozen policy."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy(path: Path = POLICY_PATH) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.is_symlink() or not path.is_file():
        raise ExplorationError(f"exploration policy is missing: {path}")
    value = contracts.load_json(path)
    if not isinstance(value, dict):
        raise ExplorationError("exploration policy must be an object")
    if value.get("schema") != POLICY_SCHEMA or value.get("policy_id") != POLICY_ID:
        raise ExplorationError("exploration policy identity changed")
    exploration = value.get("exploration_phase", {})
    selection = value.get("selection", {})
    freeze = value.get("freeze_phase", {})
    relationship = value.get("relationship_to_one_shot_policy", {})
    required = {
        "does_not_weaken_one_shot": relationship.get("weakens_it") is False,
        "candidate_set_declared_before_generation": exploration.get(
            "candidate_set_declared_before_generation"
        )
        is True,
        "operator_chosen_seeds_forbidden": exploration.get("seed_ladder", {}).get(
            "operator_chosen_seeds_allowed"
        )
        is False,
        "seed_ladder_algorithm": exploration.get("seed_ladder", {}).get("algorithm")
        == SEED_LADDER_ALGORITHM,
        "exploration_outputs_not_shippable": exploration.get(
            "outputs_may_enter_production_route"
        )
        is False,
        "exploration_usage_scope": exploration.get("output_usage_scope")
        == EXPLORATION_USAGE_SCOPE,
        "all_candidates_recorded": exploration.get("all_candidates_recorded") is True,
        "unselected_reasons_required": exploration.get(
            "unselected_candidates_require_recorded_reason"
        )
        is True,
        "single_selection": selection.get("selected_candidate_count") == 1,
        "criterion_required": selection.get("criterion_required") is True,
        "frozen_inherits_one_shot": freeze.get("frozen_request_inherits_one_shot_policy")
        is True,
        "production_regeneration_required": freeze.get("production_regeneration_required")
        is True,
        "hidden_failures_forbidden": value.get("ledger", {}).get(
            "hidden_failures_allowed"
        )
        is False,
    }
    failed = sorted(name for name, passed in required.items() if not passed)
    if failed:
        raise ExplorationError(f"exploration policy weakened: {failed}")
    return copy.deepcopy(value)


def policy_record(path: Path = POLICY_PATH) -> dict[str, Any]:
    path = Path(path).resolve()
    policy = load_policy(path)
    return {
        "schema": "avengine_controlled_animal_exploration_policy_record_v1",
        "policy_id": policy["policy_id"],
        "policy_schema": policy["schema"],
        "path": str(path),
        "sha256": _sha256_file(path),
    }


def seed_ladder(plan_id: str, seed_count: int) -> list[int]:
    """Deterministic declared candidate seeds for a plan.

    The ladder is a pure function of ``plan_id``, so the candidate set is
    fixed the moment the plan is written and cannot be extended after seeing
    an output without changing the plan id.
    """
    if not isinstance(plan_id, str) or not plan_id.strip():
        raise ExplorationError("plan_id must be a non-empty string")
    if isinstance(seed_count, bool) or not isinstance(seed_count, int) or seed_count < 1:
        raise ExplorationError("seed_count must be a positive int")
    seeds: list[int] = []
    for ordinal in range(seed_count):
        digest = hashlib.sha256(f"{plan_id}:{ordinal}".encode("utf-8")).hexdigest()
        seed = int(digest[:15], 16)
        if seed in seeds:
            raise ExplorationError(f"seed ladder collided at ordinal {ordinal}")
        seeds.append(seed)
    return seeds


def build_exploration_plan(
    asset_key: str,
    plan_id: str,
    *,
    seed_count: int | None = None,
    policy_path: Path = POLICY_PATH,
) -> dict[str, Any]:
    policy = load_policy(policy_path)
    plan = routing.routing_plan(asset_key)
    tier_count = plan["budget"]["exploration_seed_count"]
    count = tier_count if seed_count is None else seed_count
    maximum = policy["exploration_phase"]["seed_count_maximum"]
    if count > maximum:
        raise ExplorationError(
            f"seed_count {count} exceeds the declared maximum {maximum}"
        )
    if seed_count is not None and seed_count > tier_count:
        raise ExplorationError(
            f"{asset_key} is tier {plan['risk']['tier']} and is budgeted "
            f"{tier_count} seeds; {seed_count} needs a tier change in the trait "
            "contract, not a per-asset override"
        )
    payload = {
        "schema": EXPLORATION_PLAN_SCHEMA,
        "plan_id": plan_id,
        "created_at": _now(),
        "asset_key": asset_key,
        "taxonomy": plan["taxonomy"],
        "policy": policy_record(policy_path),
        "routing_plan_sha256": plan["plan_sha256"],
        "risk": plan["risk"],
        "budget": plan["budget"],
        "required_gates": plan["required_gates"],
        "prompt_blocks": plan["prompt_blocks"],
        "usage_scope": EXPLORATION_USAGE_SCOPE,
        "seed_ladder": {
            "algorithm": SEED_LADDER_ALGORITHM,
            "seed_count": count,
            "seeds": seed_ladder(plan_id, count),
        },
    }
    payload["plan_sha256"] = routing.json_sha256(payload)
    return payload


def _load_plan(path: Path) -> dict[str, Any]:
    value = contracts.load_json(Path(path))
    if not isinstance(value, dict) or value.get("schema") != EXPLORATION_PLAN_SCHEMA:
        raise ExplorationError(f"not an exploration plan: {path}")
    recorded = value.get("plan_sha256")
    body = {k: v for k, v in value.items() if k != "plan_sha256"}
    if recorded != routing.json_sha256(body):
        raise ExplorationError("exploration plan hash does not cover its body")
    declared = value["seed_ladder"]["seeds"]
    if declared != seed_ladder(value["plan_id"], value["seed_ladder"]["seed_count"]):
        raise ExplorationError("exploration plan seeds are not the declared ladder")
    return value


def record_exploration_batch(
    plan_path: Path, candidates: Mapping[int, Path]
) -> dict[str, Any]:
    """Hash every generated candidate, in ladder order, with no gaps."""
    plan = _load_plan(Path(plan_path))
    seeds = plan["seed_ladder"]["seeds"]
    missing = sorted(set(range(len(seeds))) - set(candidates))
    if missing:
        raise ExplorationError(
            f"every declared candidate must be recorded, including failures; "
            f"missing ordinals: {missing}"
        )
    unknown = sorted(set(candidates) - set(range(len(seeds))))
    if unknown:
        raise ExplorationError(f"candidate ordinals outside the ladder: {unknown}")
    records = []
    for ordinal in range(len(seeds)):
        path = Path(candidates[ordinal]).resolve()
        if path.is_symlink() or not path.is_file():
            raise ExplorationError(f"candidate {ordinal} is missing: {path}")
        records.append(
            {
                "ordinal": ordinal,
                "seed": seeds[ordinal],
                "path": str(path),
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    payload = {
        "schema": EXPLORATION_BATCH_SCHEMA,
        "created_at": _now(),
        "plan_id": plan["plan_id"],
        "plan_sha256": plan["plan_sha256"],
        "asset_key": plan["asset_key"],
        "usage_scope": EXPLORATION_USAGE_SCOPE,
        "candidates": records,
    }
    payload["batch_sha256"] = routing.json_sha256(payload)
    return payload


def _load_batch(path: Path) -> dict[str, Any]:
    value = contracts.load_json(Path(path))
    if not isinstance(value, dict) or value.get("schema") != EXPLORATION_BATCH_SCHEMA:
        raise ExplorationError(f"not an exploration batch: {path}")
    body = {k: v for k, v in value.items() if k != "batch_sha256"}
    if value.get("batch_sha256") != routing.json_sha256(body):
        raise ExplorationError("exploration batch hash does not cover its body")
    return value


def freeze_selection(
    plan_path: Path,
    batch_path: Path,
    *,
    selected_index: int,
    selection_criterion: str,
    unselected_reasons: Mapping[int, str],
    policy_path: Path = POLICY_PATH,
) -> dict[str, Any]:
    policy = load_policy(policy_path)
    plan = _load_plan(Path(plan_path))
    batch = _load_batch(Path(batch_path))
    if batch["plan_sha256"] != plan["plan_sha256"]:
        raise ExplorationError("exploration batch does not belong to this plan")

    vocabulary = policy["selection"]["criterion_vocabulary"]
    if selection_criterion not in vocabulary:
        raise ExplorationError(
            f"selection_criterion must come from the declared vocabulary {vocabulary}"
        )
    ordinals = [record["ordinal"] for record in batch["candidates"]]
    if selected_index not in ordinals:
        raise ExplorationError(f"selected_index {selected_index} is not a candidate")
    expected_unselected = sorted(set(ordinals) - {selected_index})
    if sorted(unselected_reasons) != expected_unselected:
        raise ExplorationError(
            f"every unselected candidate needs a recorded reason; expected "
            f"ordinals {expected_unselected}"
        )
    for ordinal, reason in unselected_reasons.items():
        if not isinstance(reason, str) or not reason.strip():
            raise ExplorationError(f"unselected reason {ordinal} is empty")

    selected = next(r for r in batch["candidates"] if r["ordinal"] == selected_index)
    payload = {
        "schema": FROZEN_DECLARATION_SCHEMA,
        "created_at": _now(),
        "asset_key": plan["asset_key"],
        "taxonomy": plan["taxonomy"],
        "plan_id": plan["plan_id"],
        "exploration": {
            "policy": policy_record(policy_path),
            "plan_sha256": plan["plan_sha256"],
            "batch_sha256": batch["batch_sha256"],
            "candidates_considered": len(ordinals),
            "risk_tier": plan["risk"]["tier"],
            "selected_index": selected_index,
            "selection_criterion": selection_criterion,
            "unselected_reasons": {
                str(ordinal): unselected_reasons[ordinal]
                for ordinal in expected_unselected
            },
            "selected_candidate_sha256": selected["sha256"],
        },
        "frozen_request": {
            "generation_seed": selected["seed"],
            "seed_source": "selected_exploration_ladder_index",
            "one_shot_policy": one_shot.policy_record(),
            "base_acquisition_policy": one_shot.base_acquisition_record(),
            "flux_invocations": 1,
            "flux_images_per_invocation": 1,
            "seed_retry_allowed": False,
        },
        "reproducibility": {
            "state": "pending_production_regeneration",
            "verified_at": None,
            "regenerated_sha256": None,
        },
        "required_gates": plan["required_gates"],
        "prompt_blocks": plan["prompt_blocks"],
    }
    payload["declaration_sha256"] = routing.json_sha256(payload)
    return payload


def verify_reproducibility(
    declaration_path: Path,
    regenerated_image: Path,
    *,
    allow_nondeterministic: bool = False,
) -> dict[str, Any]:
    declaration = contracts.load_json(Path(declaration_path))
    if (
        not isinstance(declaration, dict)
        or declaration.get("schema") != FROZEN_DECLARATION_SCHEMA
    ):
        raise ExplorationError(f"not a frozen declaration: {declaration_path}")
    body = {k: v for k, v in declaration.items() if k != "declaration_sha256"}
    if declaration.get("declaration_sha256") != routing.json_sha256(body):
        raise ExplorationError("frozen declaration hash does not cover its body")

    path = Path(regenerated_image).resolve()
    if path.is_symlink() or not path.is_file():
        raise ExplorationError(f"regenerated image is missing: {path}")
    digest = _sha256_file(path)
    expected = declaration["exploration"]["selected_candidate_sha256"]
    if digest == expected:
        state = "bitwise_reproduced"
    elif allow_nondeterministic:
        state = "seed_recorded_bitwise_unverified"
    else:
        raise ExplorationError(
            "the frozen seed did not reproduce the selected exploration "
            f"candidate ({digest} != {expected}); generation is not "
            "deterministic on this configuration, so the seed alone does not "
            "carry the provenance claim. Re-run with "
            "--allow-nondeterministic-regeneration to record the downgrade "
            "explicitly."
        )
    payload = {
        "schema": REPRODUCIBILITY_SCHEMA,
        "created_at": _now(),
        "asset_key": declaration["asset_key"],
        "declaration_sha256": declaration["declaration_sha256"],
        "generation_seed": declaration["frozen_request"]["generation_seed"],
        "selected_candidate_sha256": expected,
        "regenerated_path": str(path),
        "regenerated_sha256": digest,
        "state": state,
    }
    payload["record_sha256"] = routing.json_sha256(payload)
    return payload


def _parse_indexed(values: list[str], label: str) -> dict[int, str]:
    parsed: dict[int, str] = {}
    for item in values or []:
        head, separator, tail = item.partition("=")
        if not separator or not head.strip().isdigit() or not tail.strip():
            raise ExplorationError(f"{label} must be ORDINAL=text, got {item!r}")
        ordinal = int(head.strip())
        if ordinal in parsed:
            raise ExplorationError(f"duplicate {label} ordinal {ordinal}")
        parsed[ordinal] = tail.strip()
    return parsed


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the result JSON here")
    sub = parser.add_subparsers(dest="command", required=True)

    plan_parser = sub.add_parser("plan", help="declare the candidate set")
    plan_parser.add_argument("--asset-key", required=True)
    plan_parser.add_argument("--plan-id", required=True)
    plan_parser.add_argument(
        "--seed-count",
        type=int,
        help="override downward only; the tier budget is the ceiling",
    )

    record_parser = sub.add_parser("record", help="hash every generated candidate")
    record_parser.add_argument("--plan", type=Path, required=True)
    record_parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="ORDINAL=PATH",
        help="repeat for every declared ordinal, failures included",
    )

    freeze_parser = sub.add_parser("freeze", help="select one candidate and freeze it")
    freeze_parser.add_argument("--plan", type=Path, required=True)
    freeze_parser.add_argument("--batch", type=Path, required=True)
    freeze_parser.add_argument("--selected-index", type=int, required=True)
    freeze_parser.add_argument("--selection-criterion", required=True)
    freeze_parser.add_argument(
        "--unselected-reason",
        action="append",
        required=True,
        metavar="ORDINAL=REASON",
        help="repeat for every candidate that was not selected",
    )

    verify_parser = sub.add_parser("verify", help="prove the frozen seed reproduces")
    verify_parser.add_argument("--declaration", type=Path, required=True)
    verify_parser.add_argument("--regenerated-image", type=Path, required=True)
    verify_parser.add_argument(
        "--allow-nondeterministic-regeneration",
        action="store_true",
        help="record a provenance downgrade instead of failing closed",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.command == "plan":
        payload: Any = build_exploration_plan(
            args.asset_key, args.plan_id, seed_count=args.seed_count
        )
    elif args.command == "record":
        candidates = {
            ordinal: Path(value)
            for ordinal, value in _parse_indexed(args.candidate, "--candidate").items()
        }
        payload = record_exploration_batch(args.plan, candidates)
    elif args.command == "freeze":
        payload = freeze_selection(
            args.plan,
            args.batch,
            selected_index=args.selected_index,
            selection_criterion=args.selection_criterion,
            unselected_reasons=_parse_indexed(
                args.unselected_reason, "--unselected-reason"
            ),
        )
    else:
        payload = verify_reproducibility(
            args.declaration,
            args.regenerated_image,
            allow_nondeterministic=args.allow_nondeterministic_regeneration,
        )
    if args.output:
        contracts.write_json_no_replace(args.output, payload)
    else:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
