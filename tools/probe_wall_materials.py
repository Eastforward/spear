"""Render 6 candidate wall materials in the shoebox to see actual visual quality.
Uses the same Cube.Cube walls (5.2x2.8 stretched) so we can see whether each
material's texture stretches badly at room scale, before deciding on a strategy
(pool of stretch-tolerant materials, or move walls to Wall_400x300 meshes)."""
import subprocess, sys

CANDIDATES = [
    "M_Basic_Wall",
    "M_Brick_Clay_New",
    "M_Brick_Clay_Beveled",
    "M_Concrete_Panels",
    "M_Concrete_Grime",
    "M_Concrete_Poured",
    "M_Rock_Slate",
    "M_Wood_Pine",       # bonus, sometimes people use wood panels on walls
]

env = {"DISPLAY": ":99", "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json", "PATH": "/usr/bin:/bin"}

# Reuse the existing floor pool sampler (seed 0 = oak); we only vary the wall.
# The main script doesn't expose --wall-material yet, so we monkey-patch by
# editing WALL_MATERIAL through env-less: emit a tiny wrapper.
WRAPPER = "/tmp/probe_wall_wrapper.py"
open(WRAPPER, "w").write("""import sys, os
sys.path.insert(0, '/data/jzy/code/SPEAR/examples')
import render_in_gpurir_room as R
wall = os.environ['WALL_MAT']
R.WALL_MATERIAL = wall
if __name__ == '__main__':
    args = R.parse_args(sys.argv[1:])
    R.render_gpurir_room(args)
""")

for cand in CANDIDATES:
    path = f"/Game/StarterContent/Materials/{cand}.{cand}"
    run = f"claude_wall_{cand.lower()}"
    print(f"\n=== {cand} ===", flush=True)
    e = dict(env, WALL_MAT=path)
    r = subprocess.run(
        ["/data/jzy/miniconda3/envs/spear-env/bin/python", WRAPPER,
         "--animal", "dog", "--run-name", run,
         "--floor-material-seed", "0",
         "--ceiling-light-lumens", "6000",
         "--directional-light-pitch-deg", "-30"],
        env=e,
    )
    print(f"[{cand}] rc={r.returncode}", flush=True)
print("ALL_DONE", flush=True)
