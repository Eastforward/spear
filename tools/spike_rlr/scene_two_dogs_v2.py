"""Hand-authored spike scene loaded from shoebox_v2_spec.json (SSOT).

Composition:
  - dog_husky: 4-segment detour around the sofa. Ends fully occluded behind
    sofa (mic->husky line-of-sight blocked). Full trajectory over 75 frames.
  - dog_golden: uniform L->R walk BEHIND the camera (Y=1.5m, mic Y=2.2m).
    Purpose is a purely audio spatial cue -- video shows nothing.

Both dogs are `is_animated=True` (Quaternius Dog rig, "Walking" anim). Their
per-frame body_yaw is derived from motion direction + rig forward offset,
identical to the convention in scene_two_dogs.py.

All numbers come from shoebox_v2_spec.json (SSOT). Nothing hardcoded here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# The v1 scene_spec module is the canonical SceneSpec/AnimalPlacement source.
# We only override the room dimensions via a v2 SceneSpec instance -- we do
# not want to fork the whole module.
_SPEAR_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SPEAR_ROOT / "tools"))
from gpurir_scenes.scene_spec import (  # noqa: E402
    AnimalPlacement,
    SceneSpec,
    ANIM_FORWARD_YAW_OFFSET_DEG,
    N_FRAMES as SPEC_N_FRAMES,
)
from species_rig_map import ANIMATED_RIG_MAP  # noqa: E402

DEFAULT_SPEC_PATH = _SPEAR_ROOT / "data" / "shoebox_v2_spec.json"


def _forward_yaw_offset_for_tag(tag):
    return ANIMATED_RIG_MAP[tag]["walking_forward_yaw_offset_deg"]


def _linear_between(a, b, n_frames):
    """Frame-uniform linear interpolation between 2 3-tuples."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, n_frames)
    return a[None, :] + (b - a)[None, :] * ts[:, None]


def _hold(pos, n_frames):
    """Return a length-n_frames trajectory that just holds `pos`."""
    p = np.asarray(pos, dtype=np.float64)
    return np.tile(p[None, :], (n_frames, 1))


def _shortest_angular_step(a, b):
    """Signed shortest angular step from a to b (deg), in (-180, 180]."""
    return ((b - a + 180.0) % 360.0) - 180.0


def _motion_yaw_from_trajectory(traj):
    """Per-frame motion direction (deg) from 3D trajectory via xy gradient.

    Uses np.gradient so plateaus (hold segments) give a stable neighbor-based
    estimate; the hold caller separately manages what happens during zero-motion
    frames (see _build_husky_trajectory).
    """
    dx = np.gradient(traj[:, 0])
    dy = np.gradient(traj[:, 1])
    return np.degrees(np.arctan2(dy, dx))


def _build_husky_trajectory(spec, tag="dog_husky"):
    """Piecewise linear trajectory across the 4 segments in the SSOT.

    Motion yaw is set per-segment (constant during travel, held during D).
    Body yaw = motion_yaw + rig offset (Quaternius Dog "Walking" faces -X_local).
    """
    husky = next(s for s in spec["sources"] if s["tag"] == tag)
    segments = husky["trajectory_m"]
    n_frames = spec["render_config"]["n_frames"]
    traj = np.zeros((n_frames, 3))
    motion_yaw = np.zeros(n_frames)

    last_motion_yaw = None
    for seg in segments:
        f0 = seg["frame_start"]
        f1 = seg["frame_end"]  # inclusive
        seg_len = f1 - f0 + 1
        start = np.asarray(seg["start_m"])
        end = np.asarray(seg["end_m"])
        if np.allclose(start, end):
            # Hold segment: keep whatever yaw the animal had before.
            traj[f0:f1 + 1] = _hold(start, seg_len)
            if last_motion_yaw is None:
                # Only true if the very first segment is a hold; unlikely, but
                # default to +X.
                last_motion_yaw = 0.0
            motion_yaw[f0:f1 + 1] = last_motion_yaw
        else:
            traj[f0:f1 + 1] = _linear_between(start, end, seg_len)
            direction = end[:2] - start[:2]
            seg_yaw = float(np.degrees(np.arctan2(direction[1], direction[0])))
            motion_yaw[f0:f1 + 1] = seg_yaw
            last_motion_yaw = seg_yaw

    offset = _forward_yaw_offset_for_tag(tag)
    body_yaw = (motion_yaw + offset) % 360.0
    return traj, body_yaw


def _build_golden_trajectory(spec, tag="dog_golden"):
    """Uniform linear walk between start_pos_m and end_pos_m."""
    golden = next(s for s in spec["sources"] if s["tag"] == tag)
    n_frames = spec["render_config"]["n_frames"]
    traj = _linear_between(golden["start_pos_m"], golden["end_pos_m"], n_frames)
    motion_yaw = _motion_yaw_from_trajectory(traj)
    offset = _forward_yaw_offset_for_tag(tag)
    body_yaw = (motion_yaw + offset) % 360.0
    return traj, body_yaw


def compose_two_dog_scene_v2(spec_path=DEFAULT_SPEC_PATH):
    """Build the shoebox v2 SceneSpec (2 animated dogs: husky + golden)."""
    with open(spec_path) as f:
        spec = json.load(f)

    assert spec["spec_version"] == "v2", (
        f"scene_two_dogs_v2 requires spec_version v2, got {spec['spec_version']}"
    )
    assert spec["render_config"]["n_frames"] == SPEC_N_FRAMES, (
        f"spec n_frames={spec['render_config']['n_frames']} differs from "
        f"scene_spec.N_FRAMES={SPEC_N_FRAMES}; align them before rerunning."
    )

    husky_traj, husky_yaw = _build_husky_trajectory(spec, tag="dog_husky")
    golden_traj, golden_yaw = _build_golden_trajectory(spec, tag="dog_golden")

    husky = AnimalPlacement(
        tag="dog_husky",
        is_animated=True,
        trajectory_m=husky_traj,
        yaw_deg=husky_yaw,
        wanted_anim="Walking",
    )
    golden = AnimalPlacement(
        tag="dog_golden",
        is_animated=True,
        trajectory_m=golden_traj,
        yaw_deg=golden_yaw,
        wanted_anim="Walking",
    )

    return SceneSpec(
        seed=20260707,  # spike date; deterministic tag
        room_size_m=tuple(spec["room_size_m"]),
        t60_s=0.45,  # legacy A-group GPURIR uses this; B/C compute their own via RLR
        mic_pos_m=tuple(spec["mic"]["pos_m"]),
        animals=[golden, husky],  # golden first for A-group compat (listed first is drawn first)
    )


def spec_to_json_summary(spec):
    """Compact JSON dump for logging / trajectory.json output."""
    return {
        "seed": spec.seed,
        "room_size_m": list(spec.room_size_m),
        "t60_s": spec.t60_s,
        "mic_pos_m": list(spec.mic_pos_m),
        "animals": [
            {
                "tag": a.tag,
                "is_animated": a.is_animated,
                "wanted_anim": a.wanted_anim,
                "trajectory_m": a.trajectory_m.tolist(),
                "yaw_deg": a.yaw_deg.tolist(),
                "n_frames": len(a.trajectory_m),
            }
            for a in spec.animals
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(DEFAULT_SPEC_PATH))
    ap.add_argument("--out", default=None,
                    help="If set, write trajectory JSON to this path.")
    args = ap.parse_args()

    scene = compose_two_dog_scene_v2(args.spec)
    summary = spec_to_json_summary(scene)
    print(json.dumps({k: v for k, v in summary.items() if k != "animals"}, indent=2))
    print(f"animals: {len(scene.animals)}")
    for a in scene.animals:
        traj = a.trajectory_m
        print(f"  {a.tag}: {len(traj)} frames, "
              f"pos[0]={traj[0].tolist()}, "
              f"pos[end]={traj[-1].tolist()}, "
              f"yaw range [{a.yaw_deg.min():.1f}, {a.yaw_deg.max():.1f}]")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[scene_two_dogs_v2] wrote {args.out}")


if __name__ == "__main__":
    main()
