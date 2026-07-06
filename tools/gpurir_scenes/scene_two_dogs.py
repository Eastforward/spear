"""Fixed hand-authored scene: dog_golden static + dog_husky walking an
L-shaped path with a hard 90-degree pivot in the middle. Produces a
SceneSpec compatible with run_audio_pass and run_render_pass.

Two rooms (apartment + shoebox), 4 camera yaws each, muxed with 4-ch GPURIR
audio (dogs bark).

Design notes (v13, 2026-07-06):
  - No runtime collision — SPEAR spawn_actor + K2_SetActorLocationAndRotation
    with bTeleport=True bypasses all physics. To prevent visible clipping we
    ENFORCE positional constraints in the trajectory itself:
      * footprint-aware wall slack for every trajectory point
      * footprint-aware separation between static golden and every walker point
    An assertion in compose_two_dog_scene() blows up if these break.
  - Walker path is a strict L:
      Phase A (frames 0..24):  walk -X from (2.85, 3.1) to (2.55, 3.1)
      Phase B (frames 25..34): pivot IN PLACE at (2.55, 3.1), rotate motion
                               yaw 180deg -> 90deg over 10 frames (LEFT turn)
      Phase C (frames 35..59): walk +Y from (2.55, 3.1) to (2.55, 3.35)
      Phase D (frames 60..74): pivot IN PLACE to face the camera.
    The pivot period holds position so the anim shows a real body rotation.
  - Body yaw = motion_yaw + 180 (Quaternius Dog "Walking" local-forward is
    -X_local, so body must rotate 180 relative to travel direction).
  - Static golden placed at (1.06, 3.75), body_yaw_world = 270. Its Idle asset
    is 180 degrees opposite the Walking-facing convention used by the husky.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpurir_scenes.scene_spec import (
    AnimalPlacement, SceneSpec,
    ROOM_SIZE_M, MIC_POS_M, T60_S, N_FRAMES,
    SOURCE_HEIGHT_M,
    check_no_clipping,
)
from gpurir_scenes.furniture_map import load_apartment_furniture

SPEAR_PY = "/data/jzy/miniconda3/envs/spear-env/bin/python"
SAO_PY = "/data/jzy/miniconda3/envs/sao-env/bin/python"
TOOLS = os.path.dirname(os.path.abspath(__file__))


# ---- L-path segment definitions --------------------------------------------
# Husky walks a 4-segment path:
#   A: enter from right, walk -X toward mic centerline
#   B: pivot 90 (in-place) — motion yaw 180 -> 90
#   C: walk +Y toward window (away from camera, back-view)
#   D: pivot 180 (in-place) — motion yaw 90 -> -90 to FACE THE CAMERA
# All positions have >=0.5m clearance from every wall AND from golden's fixed
# spot; verified by scene_spec.check_no_clipping() at compose time.
_A_START = (2.85, 3.10)
_A_END = (2.55, 3.10)        # pivot 1
_C_END = (2.55, 3.35)        # pivot 2 (also final position)

_A_END_FRAME = 25            # frames 0..24 walking A
_B_END_FRAME = 35            # frames 25..34 pivot B
_C_END_FRAME = 60            # frames 35..59 walking C, 60..74 pivot D

_A_MOTION_YAW = 180.0        # walking -X_world
_C_MOTION_YAW = 90.0         # walking +Y_world (LEFT turn from A)
_D_END_MOTION_YAW = -90.0    # FACE CAMERA: motion=-90 -> body_yaw=90 -> head at 270w = -Y

# ---- Static golden ----------------------------------------------------------
# Placed forward-left of mic and facing the view0 camera. If it faces sideways,
# the head extends toward the image edge and gets clipped.
_STATIC_XY = (1.06, 3.75)
_STATIC_MOTION_YAW = 90.0    # body_yaw=270; Idle asset faces camera at this yaw

# Anim body-yaw offset: Quaternius Dog "Walking" local-forward = -X_local.
# body_yaw_world = motion_yaw + 180 so the dog walks head-first.
_ANIM_FORWARD_YAW_OFFSET = 180.0

def _shortest_angular_step(a, b):
    """Linear interp along the shorter arc from angle a to angle b (deg)."""
    d = ((b - a + 180.0) % 360.0) - 180.0
    return d


def _l_path_trajectory():
    """Build the piecewise L-path trajectory and per-frame yaw for husky.

    Returns (traj[N,3], yaw_deg[N]). Yaw is BODY yaw in world (already
    offset by +180 to align with the anim's -X_local forward).
    """
    xs = np.empty(N_FRAMES)
    ys = np.empty(N_FRAMES)
    motion_yaw = np.empty(N_FRAMES)

    # Phase A: frames 0 .. _A_END_FRAME - 1 : walk A_START -> A_END
    a_len = _A_END_FRAME
    for i in range(a_len):
        t = i / max(a_len - 1, 1)
        xs[i] = _A_START[0] + t * (_A_END[0] - _A_START[0])
        ys[i] = _A_START[1] + t * (_A_END[1] - _A_START[1])
        motion_yaw[i] = _A_MOTION_YAW

    # Phase B: frames _A_END_FRAME .. _B_END_FRAME - 1 : pivot in place at A_END
    b_len = _B_END_FRAME - _A_END_FRAME
    yaw_delta = _shortest_angular_step(_A_MOTION_YAW, _C_MOTION_YAW)
    for j in range(b_len):
        i = _A_END_FRAME + j
        t = (j + 1) / b_len  # step to next yaw over b_len frames
        xs[i] = _A_END[0]
        ys[i] = _A_END[1]
        motion_yaw[i] = _A_MOTION_YAW + t * yaw_delta

    # Phase C: frames _B_END_FRAME .. _C_END_FRAME - 1 : walk A_END -> C_END
    c_len = _C_END_FRAME - _B_END_FRAME
    for k in range(c_len):
        i = _B_END_FRAME + k
        t = k / max(c_len - 1, 1)
        xs[i] = _A_END[0] + t * (_C_END[0] - _A_END[0])
        ys[i] = _A_END[1] + t * (_C_END[1] - _A_END[1])
        motion_yaw[i] = _C_MOTION_YAW

    # Phase D: frames _C_END_FRAME .. N_FRAMES - 1 : pivot at C_END so the
    # husky turns from +Y back toward -Y (facing the camera).
    d_len = N_FRAMES - _C_END_FRAME
    yaw_delta_d = _shortest_angular_step(_C_MOTION_YAW, _D_END_MOTION_YAW)
    for j in range(d_len):
        i = _C_END_FRAME + j
        t = (j + 1) / d_len
        xs[i] = _C_END[0]
        ys[i] = _C_END[1]
        motion_yaw[i] = _C_MOTION_YAW + t * yaw_delta_d

    zs = np.full(N_FRAMES, SOURCE_HEIGHT_M)
    traj = np.stack([xs, ys, zs], axis=1)
    body_yaw = (motion_yaw + _ANIM_FORWARD_YAW_OFFSET) % 360.0
    return traj, body_yaw


def _static_dog_placement():
    """dog_golden stands still; is_animated=True with constant trajectory so
    run_render_pass loads the animated BP but swaps to Idle anim."""
    x_still, y_still = _STATIC_XY
    traj = np.tile(
        np.array([x_still, y_still, SOURCE_HEIGHT_M]),
        (N_FRAMES, 1),
    )
    body_yaw = np.full(
        N_FRAMES, (_STATIC_MOTION_YAW + _ANIM_FORWARD_YAW_OFFSET) % 360.0
    )
    return AnimalPlacement(
        tag="dog_golden", is_animated=True,
        trajectory_m=traj, yaw_deg=body_yaw,
        wanted_anim="Idle",
    )


def compose_two_dog_scene():
    """Compose the scene AND run the shared per-frame vs per-frame collision
    check BEFORE returning. If anything clips, this raises immediately so
    the render pass never sees a bad spec.
    """
    traj, yaw = _l_path_trajectory()
    animated = AnimalPlacement(
        tag="dog_husky", is_animated=True,
        trajectory_m=traj, yaw_deg=yaw,
    )
    static = _static_dog_placement()
    spec = SceneSpec(
        seed=999,
        room_size_m=ROOM_SIZE_M,
        t60_s=T60_S,
        mic_pos_m=MIC_POS_M,
        animals=[static, animated],
    )
    check_no_clipping(spec)
    # Extra apartment-furniture check: this hand-authored scene targets both
    # shoebox and apartment; ensure the layout also survives apartment furniture.
    apt_furniture = load_apartment_furniture()
    if apt_furniture:
        try:
            check_no_clipping(spec, furniture_bboxes=apt_furniture)
        except AssertionError as e:
            raise RuntimeError(
                f"scene_two_dogs layout clips apartment furniture: {e}\n"
                f"Retune _A_START / _A_END / _C_END / _STATIC_XY constants "
                f"in scene_two_dogs.py to avoid the reported furniture."
            ) from e
    return spec


def _spec_to_json(spec):
    return {
        "seed": spec.seed,
        "room_size_m": list(spec.room_size_m),
        "t60_s": spec.t60_s,
        "mic_pos_m": list(spec.mic_pos_m),
        "animals": [
            {"tag": a.tag, "is_animated": a.is_animated,
             "wanted_anim": getattr(a, "wanted_anim", "Walking"),
             "static_pos_m": list(a.static_pos_m) if a.static_pos_m else None,
             "static_yaw_deg": a.static_yaw_deg,
             "trajectory_m": a.trajectory_m.tolist() if a.trajectory_m is not None else None,
             "yaw_deg": a.yaw_deg.tolist() if a.yaw_deg is not None else None}
            for a in spec.animals
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", default="/data/jzy/code/SPEAR/tmp/gpurir_scenes_v1")
    p.add_argument("--skip-audio", action="store_true",
                   help="Reuse existing audio.wav (skip GPURIR pass).")
    args = p.parse_args()

    scene_dir = os.path.join(args.out_root, "two_dogs")
    os.makedirs(scene_dir, exist_ok=True)

    spec = compose_two_dog_scene()
    with open(os.path.join(scene_dir, "trajectory.json"), "w") as f:
        json.dump(_spec_to_json(spec), f, indent=2)

    audio_wav = os.path.join(scene_dir, "audio.wav")

    if not args.skip_audio:
        audio_env = dict(os.environ)
        audio_env["CUDA_VISIBLE_DEVICES"] = "1"
        wrapper = os.path.join(TOOLS, "_two_dogs_audio_wrapper.py")
        with open(wrapper, "w") as f:
            f.write(f"""\
import sys, numpy as np
sys.path.insert(0, "{os.path.dirname(TOOLS)}")
from gpurir_scenes.run_audio_pass import run_audio_pass
from gpurir_scenes.scene_two_dogs import compose_two_dog_scene
spec = compose_two_dog_scene()
rng = np.random.default_rng(999 + 10000)
meta = run_audio_pass(spec, "{audio_wav}", rng)
print("AUDIO_OK", meta)
""")
        subprocess.run([SAO_PY, wrapper], env=audio_env, check=True)
    else:
        print(f"[two_dogs] --skip-audio: reusing {audio_wav}")

    render_env = dict(os.environ)
    render_env.setdefault("DISPLAY", ":99")
    render_env.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    render_wrap = os.path.join(TOOLS, "_two_dogs_render_wrapper.py")
    with open(render_wrap, "w") as f:
        f.write(f"""\
import argparse, sys
sys.path.insert(0, "{os.path.dirname(TOOLS)}")
from gpurir_scenes.run_render_pass import run_render_pass
from gpurir_scenes.scene_two_dogs import compose_two_dog_scene
p = argparse.ArgumentParser()
p.add_argument("--room", required=True)
p.add_argument("--out-dir", required=True)
args = p.parse_args()
spec = compose_two_dog_scene()
run_render_pass(spec, args.room, args.out_dir)
print("RENDER_DONE", args.out_dir)
""")
    for room in ("apartment", "shoebox"):
        room_dir = os.path.join(scene_dir, room)
        os.makedirs(room_dir, exist_ok=True)
        subprocess.run([
            SPEAR_PY, render_wrap, "--room", room, "--out-dir", room_dir,
        ], env=render_env, check=True)

    subprocess.run([
        SPEAR_PY, os.path.join(TOOLS, "mux_audio_video.py"),
        "--scene-dir", scene_dir,
    ], check=True)

    print(f"TWO_DOGS_DONE {scene_dir}")


if __name__ == "__main__":
    main()
