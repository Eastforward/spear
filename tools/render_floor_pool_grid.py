"""Render 5 turntables, seed 0..4, each pulls a different pool floor material,
all with the new Floor_400x400 mesh (realistic plank scale) + wall-overlap
+ 6000lm ceiling light + sun pitch -30.

Output dirs under /data/jzy/code/SPEAR/tmp/render_gpurir_room/claude_pool_seed{N}/
"""
import subprocess, sys

for seed in range(5):
    r = subprocess.run(
        [
            "/data/jzy/miniconda3/envs/spear-env/bin/python",
            "/data/jzy/code/SPEAR/examples/render_in_gpurir_room.py",
            "--animal", "dog",
            "--run-name", f"claude_pool_seed{seed}",
            "--floor-material-seed", str(seed),
            "--ceiling-light-lumens", "6000",
            "--directional-light-pitch-deg", "-30",
        ],
        env={"DISPLAY": ":99",
             "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
             "PATH": "/usr/bin:/bin"},
    )
    if r.returncode != 0:
        print(f"seed={seed} returncode={r.returncode}", flush=True)
print("ALL_DONE", flush=True)
