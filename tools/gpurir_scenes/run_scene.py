"""End-to-end for one scene: spec -> audio -> renders -> mux.

Note the multi-env dance:
  * scene_spec + run_scene driver: spear-env python
  * run_audio_pass (needs gpuRIR + soundfile + SAO fallback): sao-env python
  * run_render_pass (needs spear + UE RPC): spear-env python
  * mux: any python + ffmpeg
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpurir_scenes.scene_spec import compose_scene  # noqa: E402


SPEAR_PY = "/data/jzy/miniconda3/envs/spear-env/bin/python"
SAO_PY = "/data/jzy/miniconda3/envs/sao-env/bin/python"
TOOLS = os.path.dirname(os.path.abspath(__file__))


def _spec_to_json(spec):
    return {
        "seed": spec.seed,
        "room_size_m": list(spec.room_size_m),
        "t60_s": spec.t60_s,
        "mic_pos_m": list(spec.mic_pos_m),
        "animals": [
            {"tag": a.tag, "is_animated": a.is_animated,
             "static_pos_m": list(a.static_pos_m) if a.static_pos_m else None,
             "static_yaw_deg": a.static_yaw_deg,
             "trajectory_m": a.trajectory_m.tolist() if a.trajectory_m is not None else None,
             "yaw_deg": a.yaw_deg.tolist() if a.yaw_deg is not None else None}
            for a in spec.animals
        ],
    }


def main():
    _default_out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tmp/gpurir_scenes_v1",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-root", default=_default_out)
    p.add_argument("--skip-audio", action="store_true")
    p.add_argument("--skip-render", action="store_true", help="skip both UE renders")
    p.add_argument("--skip-mux", action="store_true")
    args = p.parse_args()

    scene_dir = os.path.join(args.out_root, f"scene_{args.seed:02d}")
    os.makedirs(scene_dir, exist_ok=True)

    spec = compose_scene(seed=args.seed)
    with open(os.path.join(scene_dir, "trajectory.json"), "w") as f:
        json.dump(_spec_to_json(spec), f, indent=2)

    env = dict(os.environ)
    env.setdefault("DISPLAY", ":99")
    env.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

    if not args.skip_audio:
        audio_env = dict(env)
        audio_env["CUDA_VISIBLE_DEVICES"] = "1"  # keep SAO/gpuRIR off GPU 0 (UE)
        subprocess.run([
            SAO_PY, os.path.join(TOOLS, "run_audio_pass.py"),
            "--seed", str(args.seed),
            "--out-wav", os.path.join(scene_dir, "audio.wav"),
        ], env=audio_env, check=True)

    if not args.skip_render:
        for room in ("apartment", "shoebox"):
            room_dir = os.path.join(scene_dir, room)
            os.makedirs(room_dir, exist_ok=True)
            subprocess.run([
                SPEAR_PY, os.path.join(TOOLS, "run_render_pass.py"),
                "--seed", str(args.seed), "--room", room,
                "--out-dir", room_dir,
            ], env=env, check=True)

    if not args.skip_mux:
        subprocess.run([
            SPEAR_PY, os.path.join(TOOLS, "mux_audio_video.py"),
            "--scene-dir", scene_dir,
        ], check=True)

    print(f"SCENE_DONE {scene_dir}")


if __name__ == "__main__":
    main()
