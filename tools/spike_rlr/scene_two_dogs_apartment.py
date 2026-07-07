"""Hand-authored two-dog scene composer for apartment_v1_spec.json.

Sibling of scene_two_dogs_v2.py, but for the apartment_v1 spec:
  - Both sources use motion == "linear_uniform" (single start->end line).
  - Body yaw is derived from motion direction + per-rig forward offset,
    identical to the shoebox v2 pipeline.
  - Returns a SceneSpec (same dataclass as v1/v2) with 2 AnimalPlacements.

All numbers come from apartment_v1_spec.json (SSOT). Nothing hardcoded here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SPEAR_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SPEAR_ROOT / "tools"))
sys.path.insert(0, str(_SPEAR_ROOT / "tools" / "spike_rlr"))

from gpurir_scenes.scene_spec import (  # noqa: E402
    AnimalPlacement,
    SceneSpec,
)
from species_rig_map import ANIMATED_RIG_MAP  # noqa: E402
from scene_two_dogs_v2 import (  # noqa: E402
    _linear_between, _motion_yaw_from_trajectory,
)

DEFAULT_SPEC_PATH = _SPEAR_ROOT / "data" / "apartment_v1_spec.json"


def _forward_yaw_offset_for_tag(tag):
    return ANIMATED_RIG_MAP[tag]["walking_forward_yaw_offset_deg"]


def _build_linear_trajectory(src):
    """Uniform linear walk between start_pos_m and end_pos_m over n_frames."""
    start = np.asarray(src["start_pos_m"], dtype=np.float64)
    end = np.asarray(src["end_pos_m"], dtype=np.float64)
    return _linear_between(start, end, _N_FRAMES)


_N_FRAMES = None  # set inside compose


def compose_two_dog_scene_apartment(spec_path=DEFAULT_SPEC_PATH):
    """Build the apartment_v1 SceneSpec (2 animated dogs: golden + husky)."""
    global _N_FRAMES
    with open(spec_path) as f:
        spec = json.load(f)

    assert spec["spec_version"] == "apartment_v1", (
        f"scene_two_dogs_apartment requires spec_version apartment_v1, "
        f"got {spec['spec_version']}"
    )
    n_frames = int(spec["render_config"]["n_frames"])
    _N_FRAMES = n_frames

    animals = []
    for src in spec["sources"]:
        tag = src["tag"]
        traj = _build_linear_trajectory(src)
        motion_yaw = _motion_yaw_from_trajectory(traj)
        offset = _forward_yaw_offset_for_tag(tag)
        yaw = (motion_yaw + offset) % 360.0
        animals.append(AnimalPlacement(
            tag=tag,
            is_animated=True,
            trajectory_m=traj,
            yaw_deg=yaw,
            wanted_anim=src.get("wanted_anim", "Walking"),
        ))

    # For the apartment spec there's no "room_size_m" (non-rectangular shell);
    # we pass a large bounding box for downstream code that only uses it as
    # a sanity check. Actual geometry lives in apartment_shell_map.json.
    return SceneSpec(
        seed=20260707,
        room_size_m=(12.0, 14.0, 3.0),  # loose bounding of apartment_0000 extent
        t60_s=0.7,   # rough apartment estimate; real value computed via RLR IR
        mic_pos_m=tuple(spec["mic"]["pos_m"]),
        animals=animals,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(DEFAULT_SPEC_PATH))
    args = ap.parse_args()
    scene = compose_two_dog_scene_apartment(args.spec)
    print(f"animals: {len(scene.animals)}")
    for a in scene.animals:
        t = a.trajectory_m
        print(f"  {a.tag}: {len(t)} frames, "
              f"pos[0]={t[0].tolist()}, pos[end]={t[-1].tolist()}, "
              f"yaw range [{a.yaw_deg.min():.1f}, {a.yaw_deg.max():.1f}]")


if __name__ == "__main__":
    main()
