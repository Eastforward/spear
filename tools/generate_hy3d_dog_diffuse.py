"""End-to-end orchestrator for the animated-dog-hunyuan-paint pipeline.

Given a reference image (or a text prompt), produces a new diffuse
painted on Dog_textured.glb's UV layout using Hunyuan3D-Paint as the
core texture generator, then (optionally) installs it as the active
Dog_textured.glb for downstream UE cook + SPEAR render.

Runs in spear-env; shells out to hunyuan3d env for Hunyuan-side tools
and to Blender bundled Python for the glb retexture step.

Usage examples:
  # Reference-image path
  tools/generate_hy3d_dog_diffuse.py \\
    --reference-image demos/collie_clean.png \\
    --breed border_collie \\
    --install

  # Prompt path (requires Task 7 to have created flux_generate_reference.py)
  tools/generate_hy3d_dog_diffuse.py \\
    --prompt "golden retriever" \\
    --breed golden \\
    --install
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

REPO = "/data/jzy/code/SPEAR"
SPEAR_PY = "/data/jzy/miniconda3/envs/spear-env/bin/python"
HY3D_PY = "/data/jzy/miniconda3/envs/hunyuan3d/bin/python"
BLENDER = "/data/jzy/.local/bin/blender"

# Env vars required for any Hunyuan-side subprocess
HY3D_ENV = {
    "HY3DGEN_MODELS": "/data/jzy/code/Hunyuan3D-2.1/pretrained_models",
    "LD_LIBRARY_PATH": "/data/jzy/miniconda3/envs/hunyuan3d/lib/python3.10/site-packages/torch/lib",
    "CUDA_VISIBLE_DEVICES": "0",
}


def run(cmd, env_extra=None, check=True):
    """Run a shell command; merge env_extra onto os.environ. Print + assert on failure."""
    env = os.environ.copy()
    if env_extra:
        # append LD_LIBRARY_PATH rather than replace
        for k, v in env_extra.items():
            if k == "LD_LIBRARY_PATH" and env.get(k):
                env[k] = v + ":" + env[k]
            else:
                env[k] = v
    print(f"[generate] $ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.stdout:
        print(r.stdout, flush=True)
    if r.stderr:
        print(r.stderr, flush=True, file=sys.stderr)
    if check and r.returncode != 0:
        print(f"[generate] STEP FAILED exit={r.returncode}", flush=True)
        sys.exit(r.returncode)
    return r


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-glb", default=f"{REPO}/tmp/animated_dog/Dog_textured.glb")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--reference-image")
    src.add_argument("--prompt")
    p.add_argument("--breed", required=True, help="Slug; used in output paths")
    p.add_argument("--install", action="store_true",
                   help="If set, replace tmp/animated_dog/Dog_textured.glb with the retextured version")
    p.add_argument("--diffuse-size", type=int, default=1024)
    p.add_argument("--dilate", type=int, default=3)
    p.add_argument("--flux-model", default="flux_dev",
                   help="Model tag passed through to flux_generate_reference.py (Task 7)")
    p.add_argument("--flux-seed", type=int, default=None)
    p.add_argument("--subdivision", type=int, default=0, choices=[0, 1, 2],
                   help="Blender Catmull-Clark subdiv level (F4 escape hatch). Default 0.")
    p.add_argument("--skip-gate2", action="store_true",
                   help="Bypass Gate 2 (UV coverage) if you know it's low and want to inspect the result anyway")
    # Post-hoc color correction — Hunyuan diffusion prior consistently
    # bakes a cool-sky bias into 'should-be-black' surfaces. See
    # tools/color_correct_diffuse.py for the fix. Defaults are tuned to
    # a border-collie ref image; may need loosening for lighter breeds.
    p.add_argument("--no-color-correct", action="store_true",
                   help="Skip the post-hoc Hunyuan color-bias correction on the baked diffuse.")
    p.add_argument("--cc-strength", type=float, default=1.0,
                   help="Gray-world strength (0..1) for color correction.")
    p.add_argument("--cc-saturation", type=float, default=0.3,
                   help="HSV saturation multiplier for color correction. 1=keep, <1=desaturate.")
    p.add_argument("--cc-contrast", type=float, default=1.25,
                   help="Contrast lift for color correction. 1=none, >1=crush blacks.")
    return p.parse_args()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    args = parse_args()

    workroot = f"{REPO}/tmp/hy3d/{args.breed}"
    os.makedirs(workroot, exist_ok=True)
    workdir = f"{workroot}/hy3d_workdir"
    os.makedirs(workdir, exist_ok=True)
    metadata = {"breed": args.breed, "started_at_utc": datetime.datetime.utcnow().isoformat()}

    # ---------- Gate 0 ----------
    print("=== Gate 0: Hunyuan env sanity ===", flush=True)
    r = run([HY3D_PY, f"{REPO}/tools/verify_hy3d_env.py"], env_extra=HY3D_ENV, check=False)
    if "HY3D_ENV_OK" not in r.stdout:
        print("[generate] Gate 0 FAILED — abort")
        sys.exit(1)

    # ---------- Reference image ----------
    if args.reference_image:
        if not os.path.isabs(args.reference_image):
            args.reference_image = os.path.abspath(args.reference_image)
        ref_path = args.reference_image
        metadata["source"] = "reference-image"
        metadata["reference_image_path"] = ref_path
        metadata["reference_image_sha"] = sha256_file(ref_path)
    else:
        # Prompt path — shell out to Task 7's flux generator
        ref_path = f"{workroot}/reference.png"
        print(f"=== Text-to-image (Flux): {args.prompt!r} -> {ref_path} ===", flush=True)
        cmd = [HY3D_PY, f"{REPO}/tools/flux_generate_reference.py",
               "--prompt", args.prompt, "--output", ref_path, "--model", args.flux_model]
        if args.flux_seed is not None:
            cmd += ["--seed", str(args.flux_seed)]
        run(cmd, env_extra=HY3D_ENV)
        metadata["source"] = "prompt"
        metadata["prompt"] = args.prompt
        metadata["flux_model"] = args.flux_model
        metadata["reference_image_path"] = ref_path
        metadata["reference_image_sha"] = sha256_file(ref_path)

    # ---------- Optional subdivision escape hatch ----------
    input_glb = args.input_glb
    if args.subdivision > 0:
        subdiv_glb = f"{workroot}/Dog_textured_subdiv{args.subdivision}.glb"
        print(f"=== Subdivision level {args.subdivision} -> {subdiv_glb} ===", flush=True)
        run([BLENDER, "--background", "--python", f"{REPO}/tools/blender_subdivide_dog.py",
             "--", "--input", input_glb, "--output", subdiv_glb,
             "--levels", str(args.subdivision)])
        input_glb = subdiv_glb
        metadata["subdivision_level"] = args.subdivision

    # ---------- Stage 1: Hunyuan bake ----------
    print("=== Stage 1: Hunyuan bake ===", flush=True)
    t0 = time.time()
    run([HY3D_PY, f"{REPO}/tools/hy3d_bake_diffuse.py",
         "--input-glb", input_glb,
         "--reference-image", ref_path,
         "--workdir", workdir],
        env_extra=HY3D_ENV)
    metadata["stage1_bake_seconds"] = round(time.time() - t0, 1)

    # ---------- Gate 1: mesh diff ----------
    print("=== Gate 1: verify_hy3d_remesh ===", flush=True)
    run([HY3D_PY, f"{REPO}/tools/verify_hy3d_remesh.py",
         "--original", input_glb,
         "--hy3d-mesh", f"{workdir}/white_mesh_remesh.obj"],
        env_extra=HY3D_ENV)

    # ---------- Stage 1.5: color correction of the raw Hunyuan atlas ----------
    hy3d_diffuse_src = f"{workdir}/hy3d_diffuse.jpg"
    if not args.no_color_correct:
        print("=== Stage 1.5: color_correct_diffuse (Hunyuan blue-bias fix) ===", flush=True)
        corrected = f"{workdir}/hy3d_diffuse_corrected.png"
        run([HY3D_PY, f"{REPO}/tools/color_correct_diffuse.py",
             "--input", hy3d_diffuse_src,
             "--output", corrected,
             "--strength", str(args.cc_strength),
             "--saturation-boost", str(args.cc_saturation),
             "--contrast-lift", str(args.cc_contrast)],
            env_extra=HY3D_ENV)
        hy3d_diffuse_src = corrected
        metadata["color_corrected"] = True
        metadata["cc_params"] = {"strength": args.cc_strength,
                                 "saturation": args.cc_saturation,
                                 "contrast": args.cc_contrast}
    else:
        metadata["color_corrected"] = False

    # ---------- Stage 2: UV transfer ----------
    print("=== Stage 2: transfer_uv_texture ===", flush=True)
    diffuse = f"{workroot}/transferred_diffuse.png"
    t0 = time.time()
    run([HY3D_PY, f"{REPO}/tools/transfer_uv_texture.py",
         "--orig-mesh", input_glb,
         "--hy3d-mesh", f"{workdir}/white_mesh_remesh.obj",
         "--hy3d-diffuse", hy3d_diffuse_src,
         "--output", diffuse,
         "--size", str(args.diffuse_size),
         "--dilate", str(args.dilate)],
        env_extra=HY3D_ENV)
    metadata["stage2_transfer_seconds"] = round(time.time() - t0, 1)
    metadata["transferred_diffuse"] = diffuse

    # ---------- Gate 2: UV coverage ----------
    if not args.skip_gate2:
        print("=== Gate 2: verify_uv_coverage ===", flush=True)
        run([HY3D_PY, f"{REPO}/tools/verify_uv_coverage.py",
             "--diffuse", diffuse,
             "--original-mesh", input_glb],
            env_extra=HY3D_ENV)
    else:
        print("=== Gate 2 SKIPPED (--skip-gate2) ===", flush=True)

    # ---------- Install: retexture glb + backup ----------
    if args.install:
        print("=== Install: retexture Dog_textured.glb ===", flush=True)
        active_glb = f"{REPO}/tmp/animated_dog/Dog_textured.glb"
        if os.path.exists(active_glb):
            ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            bak = f"{active_glb}.bak.{ts}"
            shutil.copy2(active_glb, bak)
            print(f"[generate] backed up existing glb -> {bak}", flush=True)
            metadata["previous_glb_backup"] = bak

        retextured = f"{workroot}/Dog_{args.breed}.glb"
        # NOTE: use blender_replace_diffuse.py, not blender_add_uv_and_texture.py.
        # The latter re-runs Smart UV Project which would misalign the
        # transferred diffuse (painted according to the mesh's *existing*
        # UVs from Task 4). blender_replace_diffuse.py preserves UVs.
        run([BLENDER, "--background", "--python",
             f"{REPO}/tools/blender_replace_diffuse.py", "--",
             "--input", input_glb,
             "--output", retextured,
             "--diffuse-texture", diffuse])
        metadata["retextured_glb"] = retextured

        # Gate 3: verify glb integrity
        # (Only meaningful if we DIDN'T subdivide; a subdivided variant will have different vert count.)
        if args.subdivision == 0:
            print("=== Gate 3: verify_dog_textured_glb ===", flush=True)
            run([SPEAR_PY, f"{REPO}/tools/verify_dog_textured_glb.py",
                 "--input", input_glb,
                 "--output", retextured])
        # Copy retextured -> active slot
        shutil.copy2(retextured, active_glb)
        metadata["installed_at"] = active_glb

    # ---------- Write metadata ----------
    metadata["finished_at_utc"] = datetime.datetime.utcnow().isoformat()
    with open(f"{workroot}/metadata.json", "w") as fp:
        json.dump(metadata, fp, indent=2)

    print(f"GENERATE_HY3D_OK breed={args.breed} diffuse={diffuse}")


if __name__ == "__main__":
    main()
