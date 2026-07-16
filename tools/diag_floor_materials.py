"""Render 4 turntables, each with a different floor material, holding
all lighting/geometry constant. Isolates whether the dark bands/blotches on
frame 0027 come from the MI_Floor material (UV/normal/roughness behavior)
or from the scene lighting/geometry."""

import subprocess
import sys

CANDIDATES = [
    # (short_run_suffix, full material path)
    ("mi_floor",          "/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Floor.MI_Floor"),
    ("walnut_polished",   "/Game/StarterContent/Materials/M_Wood_Floor_Walnut_Polished.M_Wood_Floor_Walnut_Polished"),
    ("oak",               "/Game/StarterContent/Materials/M_Wood_Oak.M_Wood_Oak"),
    ("pine",              "/Game/StarterContent/Materials/M_Wood_Pine.M_Wood_Pine"),
    ("basic_floor",       "/Game/StarterContent/Materials/M_Basic_Floor.M_Basic_Floor"),
]

for suffix, mat in CANDIDATES:
    run = f"claude_floor_{suffix}"
    print(f"\n=== RENDERING {run}  material={mat} ===", flush=True)
    r = subprocess.run(
        [
            "/data/jzy/miniconda3/envs/spear-env/bin/python",
            "/data/jzy/code/SPEAR/examples/render_in_gpurir_room.py",
            "--animal", "dog",
            "--run-name", run,
            "--ceiling-light-lumens", "6000",
            "--directional-light-pitch-deg", "-30",
            "--floor-material", mat,
        ],
        env={
            "DISPLAY": ":99",
            "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
            "PATH": "/usr/bin:/bin",
        },
    )
    if r.returncode != 0:
        print(f"[{suffix}] returncode={r.returncode}", flush=True)

print("\nALL_DONE", flush=True)
