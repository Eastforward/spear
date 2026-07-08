"""Generate 2 new Hunyuan3D meshes (beagle, british_shorthair), drop them
into tmp/hy3d_batch/pending/, run auto-orient, then print instructions on
how to start the web-review UI.

This is a small demo driver so a human can exercise the whole Plan 1.5.A
loop end-to-end: mesh generation -> orient -> preview -> manual approve
in browser -> downstream review_gate accepts.

Usage:
    /data/jzy/miniconda3/envs/hunyuan3d/bin/python \\
        tools/spike_rlr/hy3d_generate_and_audit.py

If Hunyuan3D env is not available, use --skip-hunyuan and drop mesh.glb
files into pending/{tag}/ manually before running.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Config ---
BATCH_ROOT = REPO_ROOT / "tmp" / "hy3d_batch"
PENDING_ROOT = BATCH_ROOT / "pending"

HUNYUAN_ROOT = REPO_ROOT.parent / "Hunyuan3D-2.1"          # /data/jzy/code/AVEngine/external/Hunyuan3D-2.1
HY3D_PY = "/data/jzy/miniconda3/envs/hunyuan3d/bin/python"
SS2_PY = "/data/jzy/miniconda3/envs/ss2/bin/python"

FLUX_SCRIPT = REPO_ROOT / "tools" / "flux_generate_reference.py"

HY3D_ENV = {
    "HY3DGEN_MODELS": f"{HUNYUAN_ROOT}/pretrained_models",
    "LD_LIBRARY_PATH": "/data/jzy/miniconda3/envs/hunyuan3d/lib/python3.10/site-packages/torch/lib",
    "CUDA_VISIBLE_DEVICES": "0",
    "HF_ENDPOINT": "https://huggingface.co",
}

PROMPT_TEMPLATE = (
    "a {breed} {species} in perfect side profile view, its tail held "
    "clearly above the horizontal at about 45 degrees upward (not vertical), "
    "all four legs spread wide apart with visible gaps between them, "
    "standing on a level surface, plain solid white background, "
    "product photography, isolated on white"
)

# Two new tags to generate for the demo
NEW_RIGS = [
    {"tag": "dog_beagle", "species": "dog", "breed": "beagle", "seed": 4001},
    {"tag": "cat_british_shorthair", "species": "cat", "breed": "british shorthair", "seed": 4002},
]


def _run(cmd, env_extra=None, cwd=None, timeout=None, check=True):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    print(f"$ {' '.join(str(c) for c in cmd[:6])}...", flush=True)
    r = subprocess.run(cmd, env=env, cwd=cwd, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed with rc={r.returncode}")
    return r.returncode == 0


def _run_flux(prompt: str, out_png: Path, seed: int) -> bool:
    """Generate a reference image via Flux."""
    if out_png.exists():
        print(f"  [flux] reference exists, skipping: {out_png}")
        return True
    t0 = time.time()
    ok = _run(
        [HY3D_PY, str(FLUX_SCRIPT),
         "--prompt", prompt, "--output", str(out_png),
         "--model", "flux_dev", "--seed", str(seed)],
        env_extra=HY3D_ENV, timeout=600, check=False,
    )
    print(f"  [flux] {'OK' if ok else 'FAIL'} in {time.time()-t0:.1f}s")
    return ok


def _run_hy3d_shape(ref_png: Path, out_glb: Path) -> bool:
    """Run Hunyuan-3D-Shape via a tiny inline Python driver."""
    if out_glb.exists():
        print(f"  [hy3d] shape exists, skipping: {out_glb}")
        return True
    driver = f"""
import sys, os
sys.path.insert(0, '{HUNYUAN_ROOT}/hy3dshape')
sys.path.insert(0, '{HUNYUAN_ROOT}/hy3dpaint')
os.chdir('{HUNYUAN_ROOT}')
from PIL import Image
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.rembg import BackgroundRemover
img = Image.open('{ref_png}').convert('RGBA')
img = BackgroundRemover()(img)
p = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('hunyuan3d-2.1')
mesh = p(image=img, num_inference_steps=50)[0]
mesh.export('{out_glb}')
print('[hy3d] wrote', '{out_glb}')
"""
    driver_path = Path("/tmp") / f"hy3d_gen_{out_glb.stem}.py"
    driver_path.write_text(driver)
    t0 = time.time()
    ok = _run([HY3D_PY, str(driver_path)],
              env_extra=HY3D_ENV, timeout=1200, check=False)
    print(f"  [hy3d] {'OK' if ok else 'FAIL'} in {time.time()-t0:.1f}s")
    return ok


def _drop_into_pending(tag: str, source_glb: Path):
    """Move the generated .glb into tmp/hy3d_batch/pending/{tag}/mesh.glb."""
    tag_dir = PENDING_ROOT / tag
    tag_dir.mkdir(parents=True, exist_ok=True)
    dst = tag_dir / "mesh.glb"
    if dst.exists():
        print(f"  pending mesh already exists: {dst}")
        return
    import shutil
    shutil.copy2(source_glb, dst)
    print(f"  -> pending/{tag}/mesh.glb")


def _run_auto_orient():
    """Run auto_orient_ingest on the pending dir — writes direction.json +
    preview.png + mesh_oriented.glb per tag."""
    print("\n== Running auto_orient_ingest ==")
    _run(
        [SS2_PY, str(REPO_ROOT / "tools/spike_rlr/auto_orient_ingest.py"),
         "--pending-dir", str(PENDING_ROOT), "--force"],
        check=True,
    )


def _print_review_instructions():
    print("\n" + "=" * 70)
    print("           HUMAN AUDIT STEP REQUIRED")
    print("=" * 70)
    print()
    print("Auto-orient has run. Each pending/{tag}/direction.json is human_approved=False.")
    print("Downstream renders will REFUSE these tags until you approve them.")
    print()
    print("To approve them:")
    print()
    print("  1. On the server, start the web review UI:")
    print()
    print(f"     {SS2_PY} \\")
    print(f"         {REPO_ROOT}/tools/spike_rlr/review_ui_server.py \\")
    print( "         --port 8080")
    print()
    print("  2. On your LOCAL machine, forward the port over SSH:")
    print()
    print("     ssh -N -L 8080:localhost:8080 <this-server-host>")
    print()
    print("  3. Open your browser at http://localhost:8080/")
    print()
    print("  4. For each pending tag:")
    print("       - Look at the red arrow in the 4-view preview.")
    print("       - If it points at the animal's HEAD -> click [Approve].")
    print("       - If it points at the TAIL -> click [Head is at opposite end].")
    print("       - If the mesh is unusable -> click [Reject].")
    print()
    print("  5. Confirmed tags automatically move to:")
    print(f"       {BATCH_ROOT}/approved/{{tag}}/")
    print("     The next dataset_runner invocation (with SPEAR_SKIP_REVIEW_GATE unset)")
    print("     will accept them.")
    print()
    print("Currently pending tags awaiting your review:")
    for tag_dir in sorted(PENDING_ROOT.iterdir()):
        if tag_dir.is_dir() and (tag_dir / "direction.json").exists():
            preview = tag_dir / "direction_preview.png"
            print(f"     - {tag_dir.name}   preview: {preview}")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-hunyuan", action="store_true",
                     help="Skip Flux+Hunyuan3D generation (assume mesh.glb "
                          "already placed under pending/{tag}/).")
    ap.add_argument("--tags", nargs="*",
                     help="Only process these tag names (default: dog_beagle + cat_british_shorthair).")
    args = ap.parse_args()

    target = NEW_RIGS
    if args.tags:
        target = [r for r in NEW_RIGS if r["tag"] in args.tags]
        if not target:
            raise SystemExit(f"no matching rigs in NEW_RIGS: {args.tags}")

    PENDING_ROOT.mkdir(parents=True, exist_ok=True)

    if not args.skip_hunyuan:
        if not FLUX_SCRIPT.exists():
            raise SystemExit(f"flux script not found: {FLUX_SCRIPT}")
        print("== Generating fresh meshes ==")
        for rig in target:
            print(f"\n-- {rig['tag']} ({rig['breed']} {rig['species']}) --")
            wd = BATCH_ROOT / rig["tag"]
            wd.mkdir(parents=True, exist_ok=True)
            prompt = PROMPT_TEMPLATE.format(species=rig["species"], breed=rig["breed"])
            ref_png = wd / "reference.png"
            shape_glb = wd / "hy3d_output_mesh.glb"
            if not _run_flux(prompt, ref_png, rig["seed"]):
                print(f"  [!] Flux failed for {rig['tag']} — skipping")
                continue
            if not _run_hy3d_shape(ref_png, shape_glb):
                print(f"  [!] Hunyuan3D failed for {rig['tag']} — skipping")
                continue
            _drop_into_pending(rig["tag"], shape_glb)
    else:
        print("== Skipping Hunyuan3D (--skip-hunyuan) ==")
        for rig in target:
            expected = PENDING_ROOT / rig["tag"] / "mesh.glb"
            if not expected.exists():
                print(f"  [WARN] no {expected} — put your mesh here first.")

    _run_auto_orient()
    _print_review_instructions()


if __name__ == "__main__":
    main()
