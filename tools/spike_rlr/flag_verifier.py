"""Orchestrator: given a spec + per-source trajectories, compute all 12 flags.

Aggregation policy:
  - Per-source flags (occlusion, FOV, spatial, motion) are OR-ed across sources:
    True iff ANY source triggers the flag. (Rationale: if any source is
    occluded, the clip is "occluded".)
  - never_occluded / stays_in_camera_fov are AND-ed: True iff ALL sources are.
  - Multi-source flags (sources_pass_each_other) return True iff ANY PAIR
    triggers the pairwise check.
  - Zero-source clips: all flags False (nothing to observe).
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from flag_definitions import (  # noqa: E402
    ALL_FLAGS,
    is_occluded_by_furniture, is_occluded_by_wall, is_never_occluded,
    is_leaves_camera_fov, is_stays_in_camera_fov,
    is_crosses_azimuth_zero, is_passes_close_to_mic, is_far_from_mic_whole_clip,
    is_stationary, is_steady_walk, is_stop_and_go,
    is_sources_pass_each_other,
)


# per-source, OR-aggregated flags
_OR_FLAGS = [
    ("occluded_by_furniture", is_occluded_by_furniture),
    ("occluded_by_wall", is_occluded_by_wall),
    ("leaves_camera_fov", is_leaves_camera_fov),
    ("crosses_azimuth_zero", is_crosses_azimuth_zero),
    ("passes_close_to_mic", is_passes_close_to_mic),
    ("stationary", is_stationary),
    ("stop_and_go", is_stop_and_go),
]

# per-source, AND-aggregated flags (all sources satisfy)
_AND_FLAGS = [
    ("never_occluded", is_never_occluded),
    ("stays_in_camera_fov", is_stays_in_camera_fov),
    ("far_from_mic_whole_clip", is_far_from_mic_whole_clip),
    ("steady_walk", is_steady_walk),
]


def verify_all_flags(spec_dict: dict, trajectories: list,
                      furniture_bboxes, wall_bboxes) -> dict:
    if not trajectories:
        return {name: False for name in ALL_FLAGS}

    mic_pos = tuple(spec_dict["mic"]["pos_m"])
    mic_yaw = float(spec_dict["mic"]["yaw_deg"])
    fov_h = float(spec_dict["camera_configs"][0]["fov_deg"])
    fov_v = float(spec_dict["camera_configs"][0].get("fov_v_deg", 60.0))
    fps = int(spec_dict["render_config"]["fps"])

    result = {}
    kw = dict(
        mic_pos=mic_pos, mic_yaw_deg=mic_yaw,
        fov_h_deg=fov_h, fov_v_deg=fov_v,
        furniture_bboxes=furniture_bboxes, wall_bboxes=wall_bboxes,
        fps=fps,
    )
    for name, fn in _OR_FLAGS:
        result[name] = any(fn(traj_xyz=t, **kw) for t in trajectories)
    for name, fn in _AND_FLAGS:
        result[name] = all(fn(traj_xyz=t, **kw) for t in trajectories)

    # Multi-source: OR over all pairs
    if len(trajectories) >= 2:
        result["sources_pass_each_other"] = any(
            is_sources_pass_each_other(traj_xyz_a=a, traj_xyz_b=b)
            for a, b in combinations(trajectories, 2)
        )
    else:
        result["sources_pass_each_other"] = False

    assert set(result.keys()) == set(ALL_FLAGS), (
        f"missing/extra flags in result: "
        f"missing={set(ALL_FLAGS) - set(result.keys())}, "
        f"extra={set(result.keys()) - set(ALL_FLAGS)}"
    )
    return result


def _source_context(spec_dict: dict, furniture_bboxes, wall_bboxes) -> dict:
    return dict(
        mic_pos=tuple(spec_dict["mic"]["pos_m"]),
        mic_yaw_deg=float(spec_dict["mic"]["yaw_deg"]),
        fov_h_deg=float(spec_dict["camera_configs"][0]["fov_deg"]),
        fov_v_deg=float(spec_dict["camera_configs"][0].get("fov_v_deg", 60.0)),
        furniture_bboxes=furniture_bboxes,
        wall_bboxes=wall_bboxes,
        fps=int(spec_dict["render_config"]["fps"]),
    )


def _single_source_flags(traj_xyz, context: dict) -> dict:
    flags = {}
    for name, fn in _OR_FLAGS + _AND_FLAGS:
        flags[name] = bool(fn(traj_xyz=traj_xyz, **context))
    return flags


def verify_flag_details(
    spec_dict: dict,
    trajectories: list,
    furniture_bboxes,
    wall_bboxes,
    source_tags: list[str] | None = None,
) -> dict:
    """Return aggregate flags plus per-source and pairwise details.

    `verify_all_flags()` remains the backward-compatible flat clip-level API
    used by dataset coverage. This helper makes that aggregation auditable:
    source-local flags are reported per tag, and pairwise-only flags live under
    `pairwise`.
    """
    aggregate = verify_all_flags(
        spec_dict=spec_dict,
        trajectories=trajectories,
        furniture_bboxes=furniture_bboxes,
        wall_bboxes=wall_bboxes,
    )
    if source_tags is None:
        source_tags = [f"source_{i:04d}" for i in range(len(trajectories))]
    if len(source_tags) != len(trajectories):
        raise ValueError(
            f"source_tags length {len(source_tags)} does not match "
            f"trajectories length {len(trajectories)}"
        )

    context = _source_context(spec_dict, furniture_bboxes, wall_bboxes)
    per_source = {
        str(tag): _single_source_flags(np.asarray(traj), context)
        for tag, traj in zip(source_tags, trajectories)
    }

    pairs = []
    for (tag_a, traj_a), (tag_b, traj_b) in combinations(zip(source_tags, trajectories), 2):
        pairs.append({
            "tags": [str(tag_a), str(tag_b)],
            "sources_pass_each_other": bool(
                is_sources_pass_each_other(traj_xyz_a=traj_a, traj_xyz_b=traj_b)
            ),
        })

    return {
        "aggregate": aggregate,
        "per_source": per_source,
        "pairwise": {
            "sources_pass_each_other": bool(aggregate["sources_pass_each_other"]),
            "pairs": pairs,
        },
    }


def set_flags(flag_dict: dict) -> set:
    """Return the set of flag names that are True."""
    return {k for k, v in flag_dict.items() if v}
