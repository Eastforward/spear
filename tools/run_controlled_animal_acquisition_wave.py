#!/usr/bin/env python3
"""Plan one batched acquisition wave over the animal sound-source roster.

Two costs dominate the hardened route and neither is per-image: the Pixal3D
cold model load is roughly twenty minutes against roughly one minute of
inference, and every owner gate is a wall-clock stall.  Running breeds one at
a time pays both costs once per breed.

This driver plans a whole wave at once.  It resolves every roster entry in the
wave through the trait contract, declares each one's exploration candidate
set, and emits a single ordered execution plan whose FLUX and Pixal stages are
one batch each.  Entries that need an owner decision before GPU spend --
a motion-family decision for a non-walking gait, a declared budget for a
far-from-donor body plan -- are reported up front instead of being discovered
after the model is warm.

The driver plans; it does not generate.  It emits the exact commands the
runbook already documents so the execution path stays the reviewed one.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import controlled_animal_exploration_freeze as exploration
from tools import controlled_animal_morphotype_routing as routing
from tools import controlled_source_asset_schema as contracts


WAVE_PLAN_SCHEMA = "avengine_controlled_animal_acquisition_wave_plan_v1"

IMAGEGEN_PY = os.environ.get("AVENGINE_IMAGEGEN_PYTHON", sys.executable)


class WaveError(ValueError):
    """Raised when a wave cannot be planned as a single batch."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _blocking_decisions(plan: dict[str, Any]) -> list[dict[str, str]]:
    """Owner decisions that must land before the batch burns GPU time."""
    blocking = []
    if plan["budget"]["owner_budget_decision_required"]:
        blocking.append(
            {
                "asset_key": plan["asset_key"],
                "decision": "declared_attempt_budget",
                "reason": (
                    f"motion-donor risk {plan['risk']['motion_donor_risk']} is at or "
                    "above the declared threshold; the attempt is allowed once the "
                    "budget is declared"
                ),
            }
        )
    if "motion_family_decision" in plan["required_gates"]:
        blocking.append(
            {
                "asset_key": plan["asset_key"],
                "decision": "motion_family_decision",
                "reason": (
                    "the declared gait or rest pose is not the walking quadruped "
                    "donor; retargeting cannot be planned until a motion family is "
                    "chosen"
                ),
            }
        )
    return blocking


def plan_wave(
    wave: str,
    *,
    plan_id_prefix: str,
    workspace_root: Path,
    include_shipped: bool = False,
) -> dict[str, Any]:
    plans = routing.wave_plans(wave, include_shipped=include_shipped)

    items: list[dict[str, Any]] = []
    blocking: list[dict[str, str]] = []
    total_seeds = 0
    for plan in plans:
        asset_key = plan["asset_key"]
        plan_id = f"{plan_id_prefix}_{asset_key}"
        item_blocking = _blocking_decisions(plan)
        blocking.extend(item_blocking)
        exploration_plan = exploration.build_exploration_plan(asset_key, plan_id)
        seeds = exploration_plan["seed_ladder"]["seed_count"]
        total_seeds += seeds
        items.append(
            {
                "asset_key": asset_key,
                "taxonomy": plan["taxonomy"],
                "acquisition_state": plan["acquisition_state"],
                "risk_tier": plan["risk"]["tier"],
                "risk_total": plan["risk"]["risk_total"],
                "exploration_plan_id": plan_id,
                "exploration_seed_count": seeds,
                "preview_only_triage_required": plan["budget"][
                    "preview_only_triage_required"
                ],
                "required_gates": plan["required_gates"],
                "acoustic": plan["acoustic"],
                "blocked_on_owner_decision": [b["decision"] for b in item_blocking],
                "workspace": str(Path(workspace_root) / plan_id),
                "exploration_plan": exploration_plan,
            }
        )

    ready = [item for item in items if not item["blocked_on_owner_decision"]]
    payload = {
        "schema": WAVE_PLAN_SCHEMA,
        "created_at": _now(),
        "wave": wave,
        "plan_id_prefix": plan_id_prefix,
        "workspace_root": str(Path(workspace_root)),
        "batch": {
            "entry_count": len(items),
            "ready_entry_count": len(ready),
            "total_exploration_images": total_seeds,
            "ready_exploration_images": sum(
                item["exploration_seed_count"] for item in ready
            ),
            "model_loads_if_batched": 1,
            "model_loads_if_sequential": len(ready),
            "amortization_note": (
                "one FLUX load and one Pixal3D load cover the whole wave; running "
                "the same entries one at a time pays the roughly twenty-minute "
                "Pixal cold load once per entry"
            ),
        },
        "blocking_owner_decisions": blocking,
        "entries": items,
        "execution": _execution_commands(ready, workspace_root),
    }
    payload["wave_plan_sha256"] = routing.json_sha256(payload)
    return payload


def _execution_commands(
    ready: list[dict[str, Any]], workspace_root: Path
) -> dict[str, Any]:
    """The runbook commands for this wave, in order, as one batch per stage."""
    root = Path(workspace_root)
    inputs = [
        {
            "asset_key": item["asset_key"],
            "workspace": item["workspace"],
            "build_inputs": (
                f"{IMAGEGEN_PY} tools/build_controlled_source_asset_inputs.py "
                f"--profile <profile.json> --count-per-profile "
                f"{item['exploration_seed_count']} "
                f"--plan-id {item['exploration_plan_id']} "
                f"--split-salt avengine_{item['asset_key']} "
                f"--output-dir {item['workspace']}/inputs"
            ),
        }
        for item in ready
    ]
    return {
        "stage_1_declare": [
            "tools/run_controlled_animal_acquisition_wave.py already wrote every "
            "exploration plan; the seed ladders are fixed from this point on"
        ],
        "stage_2_flux_batch": {
            "note": "one preflight and one FLUX invocation set for the whole wave",
            "per_asset_inputs": inputs,
            "preflight": (
                f"{IMAGEGEN_PY} tools/prepare_controlled_source_asset_execution.py "
                f"--input-dir {root}/<asset>/inputs --output-dir {root}/<asset>/preflight"
            ),
            "run": (
                f"{IMAGEGEN_PY} tools/run_controlled_animal_flux2_jobs.py "
                f"--preflight {root}/<asset>/preflight/execution_preflight.json "
                f"--output-root {root}/<asset>/flux --gpu <free-gpu>"
            ),
        },
        "stage_3_record_and_freeze": [
            "tools/controlled_animal_exploration_freeze.py record --plan <plan.json> "
            "--candidate ORDINAL=PATH ...   # every ordinal, failures included",
            "tools/controlled_animal_exploration_freeze.py freeze --plan <plan.json> "
            "--batch <batch.json> --selected-index N --selection-criterion <gate> "
            "--unselected-reason ORDINAL=REASON ...",
        ],
        "stage_4_production_regeneration": [
            "re-run the one-shot FLUX route with the frozen seed, then",
            "tools/controlled_animal_exploration_freeze.py verify "
            "--declaration <frozen.json> --regenerated-image <candidate.png>",
        ],
        "stage_5_pixal_batch": (
            f"{IMAGEGEN_PY} tools/run_controlled_animal_pixal_jobs.py "
            f"--pixal-inputs <manifest> --output-root <ws>/pixal --gpu <free-gpu>   "
            f"# batch every frozen asset in one load"
        ),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", required=True, help="roster wave, e.g. T2")
    parser.add_argument(
        "--plan-id-prefix",
        required=True,
        help="declared prefix; the seed ladder is derived from the full plan id",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path("tmp/new_animal_assets"),
        help="where per-asset workspaces will be created",
    )
    parser.add_argument(
        "--include-shipped",
        action="store_true",
        help="keep shipped entries in the plan (backtest use)",
    )
    parser.add_argument(
        "--write-exploration-plans",
        action="store_true",
        help="write each asset's exploration plan into its workspace",
    )
    parser.add_argument("--output", type=Path, help="write the wave plan JSON here")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    payload = plan_wave(
        args.wave,
        plan_id_prefix=args.plan_id_prefix,
        workspace_root=args.workspace_root,
        include_shipped=args.include_shipped,
    )
    if args.write_exploration_plans:
        for item in payload["entries"]:
            if item["blocked_on_owner_decision"]:
                continue
            workspace = Path(item["workspace"])
            workspace.mkdir(parents=True, exist_ok=True)
            contracts.write_json_no_replace(
                workspace / "exploration_plan.json", item["exploration_plan"]
            )
    if args.output:
        contracts.write_json_no_replace(args.output, payload)
    else:
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
