"""Batch pipeline: for each (species, breed) tuple below, run the full
Flux -> Hunyuan-Shape -> Hunyuan-Paint -> UV extract -> compile step
and stash the outputs under a per-breed workdir.

We DO NOT swap-into-rig yet — that's a per-species step because the
existing rig is dog-specific. This batch just answers 'does the mesh
+ texture generation work uniformly across species?' — the first thing
we need before designing a per-species rig-transfer strategy.

Output layout:
  /data/jzy/code/SPEAR/tmp/hy3d_batch/<species>/<breed>/
    reference.png            (Flux output)
    reference_rembg.png      (background stripped — Hunyuan input)
    shape.glb                (Hunyuan-3D-Shape output)
    hy3d_textured.obj        (Hunyuan-Paint output with UVs)
    hy3d_diffuse.jpg
    turntable_grid.png       (4-view Blender render)
    meta.json                (prompt, seed, timing, verts, colors_ok)

Usage:
  /data/jzy/miniconda3/envs/hunyuan3d/bin/python tools/batch_animal_pipeline.py
"""
import json
import os
import subprocess
import sys
import time

# (species, breed, prompt) — prompt template must include:
#   * side profile view (Hunyuan multiview works best from side)
#   * tail up or clearly separated from legs (weight-transfer requirement)
#   * all four legs (or feet) spread apart with gaps
#   * plain white background (rembg + Hunyuan cleanness)
# For non-quadrupeds (birds, mice), adapt gently — they have their own
# body plan considerations.
PROMPT_TEMPLATE_QUADRUPED = (
    "a {breed} {species} in perfect side profile view, its tail held "
    "clearly above the horizontal at about 45 degrees upward (not vertical), "
    "all four legs spread wide apart with visible gaps between them, "
    "standing on a level surface, plain solid white background, "
    "product photography, isolated on white"
)
PROMPT_TEMPLATE_BIRD = (
    "a {breed} {species} in perfect side profile view, wings tucked at "
    "sides but slightly separated from body, tail feathers held clearly "
    "away from the body horizontally, both legs visible and apart, standing "
    "upright, plain solid white background, product photography, "
    "isolated on white"
)
PROMPT_TEMPLATE_SMALL_QUADRUPED = (
    "a {breed} {species} in perfect side profile view, tail held clearly "
    "up and away from the body, all four legs visible and separated, "
    "standing on a level surface, plain solid white background, product "
    "photography, isolated on white"
)

SPECIES_LIST = [
    # (species, breed, prompt_template, tag)
    ("dog", "golden retriever", PROMPT_TEMPLATE_QUADRUPED, "dog_golden"),
    ("dog", "beagle",            PROMPT_TEMPLATE_QUADRUPED, "dog_beagle_v2"),
    ("cat", "persian",           PROMPT_TEMPLATE_QUADRUPED, "cat_persian"),
    ("cat", "orange tabby",      PROMPT_TEMPLATE_QUADRUPED, "cat_tabby"),
    ("cat", "british shorthair", PROMPT_TEMPLATE_QUADRUPED, "cat_british_shorthair_v2"),
    ("mouse", "white lab",       PROMPT_TEMPLATE_SMALL_QUADRUPED, "mouse_white"),
    ("mouse", "gray field",      PROMPT_TEMPLATE_SMALL_QUADRUPED, "mouse_gray"),
    ("chicken", "brown hen",     PROMPT_TEMPLATE_BIRD, "chicken_hen"),
    ("chicken", "red rooster",   PROMPT_TEMPLATE_BIRD, "chicken_rooster"),
    ("duck", "mallard drake",    PROMPT_TEMPLATE_BIRD, "duck_mallard"),
    ("bird", "small yellow canary", PROMPT_TEMPLATE_BIRD, "bird_canary"),
]

BATCH_ROOT = "/data/jzy/code/SPEAR/tmp/hy3d_batch"
REPO = "/data/jzy/code/SPEAR"
SPEAR_PY = "/data/jzy/miniconda3/envs/spear-env/bin/python"
HY3D_PY = "/data/jzy/miniconda3/envs/hunyuan3d/bin/python"
BLENDER = "/data/jzy/.local/bin/blender"
HUNYUAN_ROOT = "/data/jzy/code/Hunyuan3D-2.1"

HY3D_ENV = {
    "HY3DGEN_MODELS": f"{HUNYUAN_ROOT}/pretrained_models",
    "LD_LIBRARY_PATH": "/data/jzy/miniconda3/envs/hunyuan3d/lib/python3.10/site-packages/torch/lib",
    "CUDA_VISIBLE_DEVICES": "0",
    "HF_ENDPOINT": "https://huggingface.co",
}

TURNTABLE_SCRIPT = "/tmp/render_360.py"


def run(cmd, env_extra=None, check=True, log_path=None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    print(f"$ {' '.join(cmd[:3])}...", flush=True)
    with open(log_path, "w") if log_path else open(os.devnull, "w") as f:
        r = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
    if check and r.returncode != 0:
        return False
    return True


def process_one(species, breed, template, tag, seed):
    wd = os.path.join(BATCH_ROOT, tag)
    os.makedirs(wd, exist_ok=True)
    meta = {"species": species, "breed": breed, "tag": tag, "seed": seed}
    prompt = template.format(species=species, breed=breed)
    meta["prompt"] = prompt

    # 1. Flux gen
    ref = os.path.join(wd, "reference.png")
    if not os.path.exists(ref):
        t0 = time.time()
        ok = run([HY3D_PY, f"{REPO}/tools/flux_generate_reference.py",
                  "--prompt", prompt, "--output", ref,
                  "--model", "flux_dev", "--seed", str(seed)],
                 env_extra=HY3D_ENV,
                 log_path=os.path.join(wd, "flux.log"))
        meta["flux_seconds"] = round(time.time()-t0, 1)
        meta["flux_ok"] = ok
        if not ok:
            with open(os.path.join(wd, "meta.json"), "w") as f: json.dump(meta, f, indent=2)
            return meta
    else:
        meta["flux_ok"] = True
        meta["flux_seconds"] = 0

    # 2. Hunyuan-Shape (uses rembg internally? No — but our flux script already ran rembg
    #    since --no-rembg was NOT passed, and rembg is default)
    shape_glb = os.path.join(wd, "shape.glb")
    if not os.path.exists(shape_glb):
        script = os.path.join(wd, "shape_gen.py")
        with open(script, "w") as f:
            f.write(f"""
import sys, os
os.chdir('{HUNYUAN_ROOT}')
sys.path.insert(0, 'hy3dshape')
sys.path.insert(0, 'hy3dpaint')
import torch
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
p = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('hunyuan3d-2.1')
mesh = p(image='{ref}')[0]
mesh.export('{shape_glb}')
print('SHAPE_OK', flush=True)
""")
        t0 = time.time()
        ok = run([HY3D_PY, script], env_extra=HY3D_ENV,
                 log_path=os.path.join(wd, "shape.log"))
        meta["shape_seconds"] = round(time.time()-t0, 1)
        meta["shape_ok"] = ok and os.path.exists(shape_glb)
        if not meta["shape_ok"]:
            with open(os.path.join(wd, "meta.json"), "w") as f: json.dump(meta, f, indent=2)
            return meta
    else:
        meta["shape_ok"] = True
        meta["shape_seconds"] = 0

    # 3. Hunyuan-Paint (uses our fixed hy3d_bake_diffuse.py)
    textured = os.path.join(wd, "hy3d_textured.obj")
    diffuse = os.path.join(wd, "hy3d_diffuse.jpg")
    if not (os.path.exists(textured) and os.path.exists(diffuse)):
        t0 = time.time()
        ok = run([HY3D_PY, f"{REPO}/tools/hy3d_bake_diffuse.py",
                  "--input-glb", shape_glb,
                  "--reference-image", ref,
                  "--workdir", wd],
                 env_extra=HY3D_ENV,
                 log_path=os.path.join(wd, "paint.log"))
        meta["paint_seconds"] = round(time.time()-t0, 1)
        meta["paint_ok"] = ok and os.path.exists(textured)
        if not meta["paint_ok"]:
            with open(os.path.join(wd, "meta.json"), "w") as f: json.dump(meta, f, indent=2)
            return meta
    else:
        meta["paint_ok"] = True
        meta["paint_seconds"] = 0

    # 4. Turntable render (Blender)
    tt_dir = os.path.join(wd, "turntable")
    if not os.path.exists(os.path.join(wd, "turntable_grid.png")):
        if not os.path.exists(tt_dir):
            os.makedirs(tt_dir)
            t0 = time.time()
            ok = run([BLENDER, "--background", "--python", TURNTABLE_SCRIPT, "--",
                      "--input", textured,
                      "--diffuse-override", diffuse,
                      "--output-dir", tt_dir,
                      "--n-frames", "8", "--width", "400", "--height", "400"],
                     log_path=os.path.join(wd, "turntable.log"))
            meta["turntable_seconds"] = round(time.time()-t0, 1)
        # Build grid
        import cv2, numpy as np
        tiles = []
        for i in range(8):
            fp = os.path.join(tt_dir, f"frame_{i:04d}.png")
            im = cv2.imread(fp) if os.path.exists(fp) else np.zeros((400,400,3), dtype=np.uint8)
            cv2.putText(im, f"ang{i*45}", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 3)
            cv2.putText(im, f"ang{i*45}", (5,25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            tiles.append(im)
        row1 = np.concatenate(tiles[:4], axis=1); row2 = np.concatenate(tiles[4:], axis=1)
        cv2.imwrite(os.path.join(wd, "turntable_grid.png"), np.concatenate([row1, row2], axis=0))
    else:
        meta["turntable_seconds"] = 0

    # 5. Quick color sanity: sample the diffuse's non-black pixel mean
    import cv2, numpy as np
    atl = cv2.imread(diffuse)
    mask = atl.max(axis=2) > 30
    if mask.sum() > 100:
        mean_bgr = tuple(round(float(atl[..., c][mask].mean()), 1) for c in range(3))
        # count "colored" pixels — any channel differs from grayscale
        maxc, minc = atl.max(axis=2), atl.min(axis=2)
        colored_frac = round(float(((maxc.astype(int) - minc.astype(int)) > 15).mean()), 3)
        meta["diffuse_mean_bgr"] = list(mean_bgr)
        meta["diffuse_colored_fraction"] = colored_frac

    # Verts count
    with open(textured) as f:
        content = f.read()
    meta["textured_verts"] = content.count("\nv ")
    meta["textured_uvs"] = content.count("\nvt ")
    meta["textured_faces"] = content.count("\nf ")

    with open(os.path.join(wd, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    os.makedirs(BATCH_ROOT, exist_ok=True)
    all_meta = []
    for i, (sp, br, tmpl, tag) in enumerate(SPECIES_LIST):
        print(f"\n===== [{i+1}/{len(SPECIES_LIST)}] {sp} / {br} =====", flush=True)
        t0 = time.time()
        meta = process_one(sp, br, tmpl, tag, seed=42 + i)
        meta["total_seconds"] = round(time.time()-t0, 1)
        all_meta.append(meta)
        # save aggregate
        with open(os.path.join(BATCH_ROOT, "batch_summary.json"), "w") as f:
            json.dump(all_meta, f, indent=2)

    # Print summary table
    print(f"\n{'='*60}\nBATCH SUMMARY:")
    for m in all_meta:
        ok = "OK" if all(m.get(k, False) for k in ["flux_ok","shape_ok","paint_ok"]) else "FAIL"
        print(f"  {m['tag']:22s} {ok}  verts={m.get('textured_verts',0):>7d}  "
              f"colored_frac={m.get('diffuse_colored_fraction','?'):>6}")


if __name__ == "__main__":
    main()
