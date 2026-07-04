# Animated Dog Hunyuan3D-Paint Texture Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Tier-3 AI texture path that takes either a reference image or a text prompt and produces a photo-realistic dog-coat diffuse mapped onto our existing `Dog_textured.glb`, keeping the Walking animation intact end-to-end (Blender bake → UE cook → SPEAR render).

**Architecture:** Black-box call to Hunyuan3D-Paint (it can freely remesh + regenerate UVs internally). Afterwards, transfer its baked texture back onto the original mesh's UV layout via a per-triangle barycentric sampler that walks the KD-tree of Hunyuan-mesh triangle centers in world space. Text-to-image is a **prompt-only optional stage** (ComfyUI headless call to a Flux/SDXL checkpoint, on the same GPU as Hunyuan) that produces a clean reference photo when the user only supplies a prompt.

**Tech Stack:** Python 3.10 in `hunyuan3d` conda env (torch 2.5.1, diffusers 0.31.0, `Hunyuan3DPaintPipeline`, `custom_rasterizer`); Python 3.11 in `spear-env` for glTF/verify/render; Blender 4.2.1 LTS bpy for glb retexture + subdivision escape hatch; SPEAR RPC unchanged; ComfyUI headless (`comfyui` env) for optional Flux/SDXL text-to-image; scipy `cKDTree` for triangle nearest-neighbor; opencv + numpy for image ops.

## Global Constraints

- Two Python envs. **Hunyuan side**: `/data/jzy/miniconda3/envs/hunyuan3d/bin/python`. **SPEAR side**: `/data/jzy/miniconda3/envs/spear-env/bin/python`. Never cross-import (torch versions differ).
- Hunyuan runs REQUIRE **all three** env vars: `HY3DGEN_MODELS=/data/jzy/code/Hunyuan3D-2.1/pretrained_models`, `LD_LIBRARY_PATH=/data/jzy/miniconda3/envs/hunyuan3d/lib/python3.10/site-packages/torch/lib`, `CUDA_VISIBLE_DEVICES=0`. Missing any = silent crash or `libc10.so` `ImportError`.
- Hunyuan working directory MUST be `/data/jzy/code/Hunyuan3D-2.1` — the pipeline resolves `hy3dpaint/cfgs/...` and `hy3dpaint/ckpt/...` relative to cwd. Wrapper scripts must `os.chdir(HY3D_ROOT)` before instantiating `Hunyuan3DPaintPipeline`.
- Blender: `/data/jzy/.local/bin/blender` (v4.2.1 LTS). No scipy, no imageio inside its Python. Use numpy-only image ops and `bpy.data.images.new(...).save()` for PNG writes (see [tools/blender_bake_pattern_diffuse.py](../../tools/blender_bake_pattern_diffuse.py) for the pattern).
- Never mutate `tmp/animated_dog/Dog_textured.glb` in place without first copying to `tmp/animated_dog/Dog_textured.glb.bak.<utc_timestamp>`. This lets the render script always load `Dog_textured.glb` while allowing rollback.
- Do NOT modify `tools/build_animated_dog.sh`, `tools/blender_add_uv_and_texture.py`, `tools/verify_dog_textured_glb.py`, `tools/diag_animated_dog.py`, or `examples/render_animated_dog_gpurir.py`. Those are frozen from the predecessor spec. If a change is genuinely needed, spawn a new spec.
- Xvfb `:99` must be running before Gate 5. Verify with `pgrep -a Xvfb` before ANY SPEAR runtime command. SPEAR runtime shell prefix: `DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`.
- SPEAR project root: `/data/jzy/code/SPEAR`. All relative paths in this plan resolve from there unless prefixed with `/`.
- Every task ends with a git commit. Commit message format: `<phase>: <what>` first line; trailing `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Unit tests run as: `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.<module> -v` (or the `hunyuan3d` env when the test needs Hunyuan modules — noted per task).

## File structure

```
SPEAR/
├── docs/
│   ├── superpowers/plans/2026-07-05-animated-dog-hunyuan-paint.md (this file)
│   ├── superpowers/specs/2026-07-05-animated-dog-hunyuan-paint-design.md
│   └── superpowers/handoffs/2026-07-05-animated-dog-hunyuan-paint.md   T9  writes
├── tools/
│   ├── prefetch_t2i_models.py                  T0 writes  (background HF download; hunyuan3d env)
│   ├── hy3d_bake_diffuse.py                    T2 writes  (Hunyuan wrapper; hunyuan3d env)
│   ├── verify_hy3d_remesh.py                   T3 writes  (Gate 1; hunyuan3d env)
│   ├── transfer_uv_texture.py                  T4 writes  (UV transfer; hunyuan3d env)
│   ├── verify_uv_coverage.py                   T5 writes  (Gate 2; hunyuan3d or spear-env)
│   ├── generate_hy3d_dog_diffuse.py            T6 writes  (top-level orchestrator; spear-env; shells out to other envs)
│   ├── flux_generate_reference.py              T7 writes  (text-to-image; comfyui env; optional)
│   └── blender_subdivide_dog.py                T8 writes  (escape-hatch subdivision; blender bundled Python)
├── tests/
│   ├── test_transfer_uv_texture.py             T4 writes  (unit tests; hunyuan3d env)
│   └── test_verify_uv_coverage.py              T5 writes  (unit tests; spear-env env)
├── tmp/hy3d/
│   ├── <breed>/reference.png                   T6/T7 produce (not git)
│   ├── <breed>/hy3d_workdir/                   T2/T6 produce (Hunyuan intermediates; not git)
│   ├── <breed>/transferred_diffuse.png         T4/T6 produce (final new diffuse; not git)
│   └── <breed>/metadata.json                   T6 produces (breed name, prompt or ref-image sha, timestamps; not git)
├── tmp/animated_dog/Dog_textured.glb.bak.<ts>  T6 produces (auto-backup before overwrite; not git)
└── assets/reference_dogs/                      manual (checked-in reference images if any; only if user commits them)
```

## Task dependency graph

```
T0 (background)  prefetch Flux+SDXL weights (nohup &) ── runs in parallel with T1-T6
                    │
                    ▼ (by the time T7 starts, weights are cached)

T1 (Gate 0)  Hunyuan env sanity ─────────┐
                                          │
T2 (Stage 1) hy3d_bake_diffuse.py ───────┤
                                          │
T3 (Gate 1)  verify_hy3d_remesh.py ──────┤
                                          │
T4 (Stage 2) transfer_uv_texture.py + tests ─────┐
                                                  │
T5 (Gate 2)  verify_uv_coverage.py + tests ──────┤
                                                  │
T6 (Stage 3) generate_hy3d_dog_diffuse.py (end-to-end orchestrator)
                                                  │
T7 (Optional) flux_generate_reference.py (only if --prompt path)
                                                  │
T8 (Escape hatch, run ONLY if F4 fires) blender_subdivide_dog.py
                                                  │
                                                  ▼
                              Gate 3 (verify_dog_textured_glb.py, already exists)
                              Gate 4 (build_animated_dog.sh, already exists)
                              Gate 5 (diag_animated_dog.py, already exists)
                                                  │
                                                  ▼
T9  Handoff doc + final commit
```

**Ordering rules:**
- T0 must be launched in the background (`nohup ... &`) BEFORE T1. Do not wait for it; T1-T6 run while it downloads.
- T1 blocks the rest of the *foreground* work. Do not start T2 until Gate 0 passes.
- T2 and T4 are independent; T4 uses fixture data (checked in under `tests/fixtures/`), so it does not need T2 to have run.
- T6 assembles T2–T5 into one CLI + adds Gates 3/4/5 as shell-out steps. It's the acceptance criterion for the primary path.
- T7 requires T0 to have completed for at least one model. Check `grep PREFETCH_OK /tmp/prefetch_t2i.log` before starting T7.
- T8 is dormant unless Gate 2 fails in T5/T6. Documented so a future run knows how to invoke it, but not part of the happy-path acceptance.

---

## Environment prep appendix (compact-survival notes)

This section documents session-state that already exists on the target machine but is not in git — if a fresh session comes in after context compaction, do THIS FIRST before Task 1 so nothing is left to spelunking.

**Env vars for every Hunyuan-side subprocess:**
```
HY3DGEN_MODELS=/data/jzy/code/Hunyuan3D-2.1/pretrained_models
LD_LIBRARY_PATH=/data/jzy/miniconda3/envs/hunyuan3d/lib/python3.10/site-packages/torch/lib
CUDA_VISIBLE_DEVICES=0
```

**hunyuan3d conda env fixes already applied (do NOT redo unless verify_hy3d_env.py fails):**

1. `basicsr` + `realesrgan` installed with `--no-build-isolation --no-deps`:
   ```bash
   /data/jzy/miniconda3/envs/hunyuan3d/bin/pip install --no-build-isolation --no-deps basicsr realesrgan
   ```
2. `pytorch_lightning` + lightning deps:
   ```bash
   /data/jzy/miniconda3/envs/hunyuan3d/bin/pip install --no-deps pytorch_lightning
   /data/jzy/miniconda3/envs/hunyuan3d/bin/pip install lightning_utilities lightning_fabric torchmetrics
   ```
3. torchvision `functional_tensor` import patch:
   ```bash
   sed -i 's|from torchvision.transforms.functional_tensor import rgb_to_grayscale|from torchvision.transforms.functional import rgb_to_grayscale|' /data/jzy/miniconda3/envs/hunyuan3d/lib/python3.10/site-packages/basicsr/data/degradations.py
   ```
4. `custom_rasterizer` C extension built in-place:
   ```bash
   cd /data/jzy/code/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer
   /data/jzy/miniconda3/envs/hunyuan3d/bin/pip install --no-build-isolation -e .
   ```
5. RealESRGAN checkpoint symlink (the pipeline hardcodes `hy3dpaint/ckpt/RealESRGAN_x4plus.pth`):
   ```bash
   mkdir -p /data/jzy/code/Hunyuan3D-2.1/hy3dpaint/ckpt
   ln -sf /data/jzy/code/Hunyuan3D-2.1/ckpt/RealESRGAN_x4plus.pth \
          /data/jzy/code/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth
   ```
6. Weights symlink at the path pattern the pipeline searches (`$HY3DGEN_MODELS/tencent/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1`):
   ```bash
   mkdir -p /data/jzy/code/Hunyuan3D-2.1/pretrained_models/tencent/Hunyuan3D-2.1
   ln -sfn /data/jzy/code/Hunyuan3D-2.1/pretrained_models/hunyuan3d-2.1/hunyuan3d-paintpbr-v2-1 \
           /data/jzy/code/Hunyuan3D-2.1/pretrained_models/tencent/Hunyuan3D-2.1/hunyuan3d-paintpbr-v2-1
   ```

**HF endpoint:** `HF_ENDPOINT=https://hf-mirror.com` is already set globally on this box (via user's shell profile). Confirm with `env | grep HF_ENDPOINT` — if empty, prepend to any HF download command.

**Baseline glb state:** `tmp/animated_dog/Dog_textured.glb` should contain the warm-brown baseline from the predecessor spec. If it does not, restore from `.bak.warm_brown` before T2.

If `verify_hy3d_env.py` in T1 prints `HY3D_ENV_OK`, all six fixes above are still valid — skip this appendix.

---

## Task 0: Background — kick off model downloads before Task 1

**Purpose:** Flux and SDXL weights are 15–24 GB each and download in the background from Hugging Face. Kick them off FIRST so T1–T6 run in parallel with the download. When we reach T7, the models are already local.

**Files:**
- Create: `/data/jzy/code/SPEAR/tools/prefetch_t2i_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces: pulls `black-forest-labs/FLUX.1-dev` and `stabilityai/stable-diffusion-xl-base-1.0` into the HF cache. Prints `PREFETCH_OK <model>` per model. Idempotent (skipped if already cached).

**Which model do we pick as default?**
- **Flux.1 dev** — best quality photorealistic generation; ~24 GB. Requires HF login for gated repo (`huggingface-cli login` with a token that has accepted the FLUX.1-dev license). Recommended default.
- **SDXL base 1.0** — universally available (no gate); ~14 GB; slightly lower quality on animals but perfectly usable. Fallback if Flux is blocked.
- **NOT Flux.1 schnell** — 4-step distilled, loses fur detail; poor Hunyuan reference material.

We download BOTH so `flux_generate_reference.py --model` can switch between them; default is `flux_dev` when weights present, `sdxl_base` otherwise.

- [ ] **Step 1: Write the prefetch script**

```python
# /data/jzy/code/SPEAR/tools/prefetch_t2i_models.py
"""Task 0: prefetch Flux.1 dev + SDXL base 1.0 weights into HF cache.

Runs in the hunyuan3d env (has huggingface_hub already). Designed to be
launched in the background BEFORE Task 1 so weights download while the
early tasks run. Idempotent: skips models already cached.

Usage (in background):
  HF_ENDPOINT=https://hf-mirror.com \\
    /data/jzy/miniconda3/envs/hunyuan3d/bin/python \\
    /data/jzy/code/SPEAR/tools/prefetch_t2i_models.py \\
    > /tmp/prefetch_t2i.log 2>&1 &
"""
import os
import sys

MODELS = [
    ("black-forest-labs/FLUX.1-dev",
        # Flux uses safetensors + config + transformer + vae + 2 text encoders
        None),                    # None = allow all files
    ("stabilityai/stable-diffusion-xl-base-1.0",
        None),
]


def main():
    from huggingface_hub import snapshot_download

    for repo_id, allow in MODELS:
        try:
            print(f"[prefetch] downloading {repo_id}...", flush=True)
            path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=allow,
                # Prefer .safetensors over .bin to save space
                ignore_patterns=["*.bin", "*.msgpack", "*.h5"],
            )
            print(f"PREFETCH_OK {repo_id} -> {path}", flush=True)
        except Exception as e:
            print(f"PREFETCH_FAIL {repo_id}: {e}", flush=True)
            # continue with the next model instead of aborting

    print("PREFETCH_DONE", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Kick off the download in the background (BEFORE Task 1)**

```bash
# 1. Confirm HF token for Flux.1 dev (gated repo). If not set:
#    huggingface-cli login    # paste token from https://huggingface.co/settings/tokens
# The token must be attached to a user that has accepted the FLUX.1 license.
[ -f ~/.cache/huggingface/token ] || echo "MISSING HF token — Flux download will fail; SDXL still works"

# 2. Launch the prefetch in the background
mkdir -p /tmp
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} \
  nohup /data/jzy/miniconda3/envs/hunyuan3d/bin/python \
  /data/jzy/code/SPEAR/tools/prefetch_t2i_models.py \
  > /tmp/prefetch_t2i.log 2>&1 &
echo "prefetch pid: $!"

# 3. IMMEDIATELY continue with Task 1 — do NOT wait here.
```

- [ ] **Step 3: Continue with Task 1 without blocking**

Do NOT poll or wait; move straight to Task 1. When you reach Task 7, check with:

```bash
tail -20 /tmp/prefetch_t2i.log
grep -c "PREFETCH_OK" /tmp/prefetch_t2i.log
```

Expected by Task 7 start: `PREFETCH_OK` for at least one of {FLUX.1-dev, SDXL base}. If BOTH failed, T7 gracefully degrades (either the user provides `--reference-image` for all tests, or T7 is deferred to a follow-up).

- [ ] **Step 4: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/prefetch_t2i_models.py
git commit -m "T0: prefetch_t2i_models.py — background download Flux+SDXL

Kicked off in background BEFORE Task 1 so ~38 GB of weights land
during T1-T6 wall-clock. Downloads Flux.1 dev (gated, needs HF token)
and SDXL base 1.0 (open). Idempotent: skips if already cached.

Downstream T7 (flux_generate_reference.py) will use flux_dev by default
and fall back to sdxl_base if Flux missing.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 1: Gate 0 — Hunyuan environment sanity script

**Files:**
- Create: `/data/jzy/code/SPEAR/tools/verify_hy3d_env.py`
- Modify: none

**Interfaces:**
- Consumes: nothing
- Produces: exits 0 + prints `HY3D_ENV_OK` when the Hunyuan pipeline can be imported and instantiated; exits 1 with an actionable error otherwise. Every downstream Hunyuan task starts by grepping for `HY3D_ENV_OK`.

- [ ] **Step 1: Write the verification script**

```python
# /data/jzy/code/SPEAR/tools/verify_hy3d_env.py
"""Gate 0 for the animated-dog-hunyuan-paint spec.

Confirms the local Hunyuan3D-Paint pipeline can be imported and
constructed with the local pretrained weights. Prints HY3D_ENV_OK on
success. Any other output = STOP, fix env before continuing.

Usage:
  HY3DGEN_MODELS=/data/jzy/code/Hunyuan3D-2.1/pretrained_models \\
  LD_LIBRARY_PATH=/data/jzy/miniconda3/envs/hunyuan3d/lib/python3.10/site-packages/torch/lib \\
  CUDA_VISIBLE_DEVICES=0 \\
    /data/jzy/miniconda3/envs/hunyuan3d/bin/python tools/verify_hy3d_env.py
"""
import os
import sys

HY3D_ROOT = "/data/jzy/code/Hunyuan3D-2.1"
REQUIRED_ENV = [
    "HY3DGEN_MODELS",
    "LD_LIBRARY_PATH",   # must contain torch/lib for libc10.so
]

for var in REQUIRED_ENV:
    if not os.environ.get(var):
        print(f"HY3D_ENV_FAIL missing env {var}")
        sys.exit(1)

sys.path.insert(0, HY3D_ROOT)
sys.path.insert(0, os.path.join(HY3D_ROOT, "hy3dshape"))
sys.path.insert(0, os.path.join(HY3D_ROOT, "hy3dpaint"))
os.chdir(HY3D_ROOT)  # required — pipeline uses relative paths for cfgs/ckpts

try:
    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig  # noqa: F401
except ImportError as e:
    print(f"HY3D_ENV_FAIL import failed: {e}")
    sys.exit(1)

try:
    import custom_rasterizer as cr
    assert hasattr(cr, "rasterize"), "custom_rasterizer.rasterize missing"
except (ImportError, AssertionError) as e:
    print(f"HY3D_ENV_FAIL custom_rasterizer bad: {e}")
    sys.exit(1)

# Weights presence check (no HF download attempt)
expected_weights = os.path.join(
    os.environ["HY3DGEN_MODELS"], "tencent", "Hunyuan3D-2.1",
    "hunyuan3d-paintpbr-v2-1", "unet",
)
if not os.path.isdir(expected_weights):
    print(f"HY3D_ENV_FAIL weights not found at {expected_weights}")
    print("  Fix: create the tencent/Hunyuan3D-2.1 symlink pointing at pretrained_models/hunyuan3d-2.1")
    sys.exit(1)

print("HY3D_ENV_OK")
```

- [ ] **Step 2: Run it — should pass on the machine that already ran the probe**

```bash
HY3DGEN_MODELS=/data/jzy/code/Hunyuan3D-2.1/pretrained_models \
LD_LIBRARY_PATH=/data/jzy/miniconda3/envs/hunyuan3d/lib/python3.10/site-packages/torch/lib \
CUDA_VISIBLE_DEVICES=0 \
  /data/jzy/miniconda3/envs/hunyuan3d/bin/python /data/jzy/code/SPEAR/tools/verify_hy3d_env.py
```

Expected: single line `HY3D_ENV_OK`. Anything else = STOP, fix env (see spec §7 for what's supposed to already exist).

- [ ] **Step 3: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/verify_hy3d_env.py
git commit -m "T1 (Gate 0): hunyuan3d env sanity script

Verifies HY3DGEN_MODELS / LD_LIBRARY_PATH set, Hunyuan3D-Paint pipeline
importable, custom_rasterizer.rasterize present, and paintpbr weights
on disk at the expected symlinked path. Prints HY3D_ENV_OK on success.

Every subsequent Hunyuan task starts by grepping this string.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 
```

---

## Task 2: Stage 1 — hy3d_bake_diffuse.py (Hunyuan wrapper)

**Files:**
- Create: `/data/jzy/code/SPEAR/tools/hy3d_bake_diffuse.py`
- Modify: none

**Interfaces:**
- Consumes: `verify_hy3d_env.py` (Task 1) must have printed `HY3D_ENV_OK` in the current terminal / env
- Produces: CLI `hy3d_bake_diffuse.py --input-glb PATH --reference-image PATH --workdir DIR` that writes into `DIR`:
  - `white_mesh_remesh.obj` — Hunyuan's remeshed mesh (may be near-identical to input; see spec §4.2)
  - `hy3d_diffuse.jpg` — the 4K baked albedo
  - `hy3d_metallic.jpg`, `hy3d_roughness.jpg` — PBR sidecars (kept but unused in MVP)
  - `hy3d_output_mesh.glb` — Hunyuan's textured glb (broken by trimesh's glb writer; we IGNORE this file downstream and use the .obj)
  - Prints `HY3D_BAKE_OK workdir=<abs path>` on success.

- [ ] **Step 1: Write the wrapper script**

```python
# /data/jzy/code/SPEAR/tools/hy3d_bake_diffuse.py
"""Stage 1 of the animated-dog-hunyuan-paint pipeline.

Thin wrapper around Hunyuan3DPaintPipeline that fixes config, takes
(input glb, reference image, workdir), and drops the three baked PBR
maps + the remeshed obj into workdir. Prints HY3D_BAKE_OK on success.

Runs inside the hunyuan3d conda env — do NOT try to run it via any
other Python. Environment variables HY3DGEN_MODELS + LD_LIBRARY_PATH +
CUDA_VISIBLE_DEVICES must be set; verify_hy3d_env.py in the same shell
first is the recommended pattern.
"""
import argparse
import os
import shutil
import sys
import time

HY3D_ROOT = "/data/jzy/code/Hunyuan3D-2.1"
sys.path.insert(0, HY3D_ROOT)
sys.path.insert(0, os.path.join(HY3D_ROOT, "hy3dshape"))
sys.path.insert(0, os.path.join(HY3D_ROOT, "hy3dpaint"))
os.chdir(HY3D_ROOT)   # MANDATORY — pipeline uses cwd-relative paths


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-glb", required=True,
                   help="Absolute path to Dog_textured.glb (or a subdivided variant).")
    p.add_argument("--reference-image", required=True,
                   help="Absolute path to reference PNG/JPG (should be rembg'd + standing side-view for best result).")
    p.add_argument("--workdir", required=True,
                   help="Absolute path to a directory that will contain all Hunyuan intermediates.")
    p.add_argument("--max-num-view", type=int, default=6, choices=[6, 7, 8, 9],
                   help="Number of Hunyuan multiview cameras. Default 6; bump to 9 if bake coverage insufficient (F4).")
    p.add_argument("--resolution", type=int, default=512,
                   help="Hunyuan per-view diffusion resolution. Default 512.")
    p.add_argument("--texture-size", type=int, default=4096,
                   help="Output baked texture size in pixels. Default 4096.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    work_glb = os.path.join(args.workdir, "input.glb")
    shutil.copy2(args.input_glb, work_glb)
    output_glb = os.path.join(args.workdir, "hy3d_output_mesh.glb")

    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
    conf = Hunyuan3DPaintConfig(max_num_view=args.max_num_view, resolution=args.resolution)
    conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
    conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
    conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
    conf.texture_size = args.texture_size

    t0 = time.time()
    print(f"[hy3d_bake] loading pipeline ({args.max_num_view} views, {args.resolution} res)...", flush=True)
    pipeline = Hunyuan3DPaintPipeline(conf)
    print(f"[hy3d_bake] pipeline loaded in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    pipeline(
        mesh_path=work_glb,
        image_path=args.reference_image,
        output_mesh_path=output_glb,
        use_remesh=True,     # near-no-op on our 1233-vert mesh; probe verified
        save_glb=True,
    )
    dt = time.time() - t1
    print(f"[hy3d_bake] paint pipeline finished in {dt:.1f}s", flush=True)

    # Rename baked sidecars to canonical names so downstream tools have stable paths
    stem = os.path.splitext(os.path.basename(output_glb))[0]
    for src_suffix, dst_name in [(".jpg", "hy3d_diffuse.jpg"),
                                 ("_metallic.jpg", "hy3d_metallic.jpg"),
                                 ("_roughness.jpg", "hy3d_roughness.jpg")]:
        src = os.path.join(args.workdir, stem + src_suffix)
        dst = os.path.join(args.workdir, dst_name)
        if os.path.exists(src):
            shutil.move(src, dst)

    # Sanity check the three products
    required = ["white_mesh_remesh.obj", "hy3d_diffuse.jpg"]
    missing = [f for f in required if not os.path.exists(os.path.join(args.workdir, f))]
    if missing:
        print(f"HY3D_BAKE_FAIL missing {missing} in {args.workdir}")
        sys.exit(1)

    print(f"HY3D_BAKE_OK workdir={args.workdir} elapsed={dt:.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test it against the probe's border-collie reference image**

```bash
mkdir -p /data/jzy/code/SPEAR/tmp/hy3d/task2_smoke
HY3DGEN_MODELS=/data/jzy/code/Hunyuan3D-2.1/pretrained_models \
LD_LIBRARY_PATH=/data/jzy/miniconda3/envs/hunyuan3d/lib/python3.10/site-packages/torch/lib \
CUDA_VISIBLE_DEVICES=0 \
  /data/jzy/miniconda3/envs/hunyuan3d/bin/python /data/jzy/code/SPEAR/tools/hy3d_bake_diffuse.py \
    --input-glb /data/jzy/code/SPEAR/tmp/animated_dog/Dog_textured.glb \
    --reference-image /data/jzy/code/Hunyuan3D-2.1/demos/collie_clean.png \
    --workdir /data/jzy/code/SPEAR/tmp/hy3d/task2_smoke
```

Expected: last line is `HY3D_BAKE_OK workdir=/data/jzy/code/SPEAR/tmp/hy3d/task2_smoke elapsed=<seconds>`. Total wall-clock: 60-90 s on RTX 4090.

Verify files landed:

```bash
ls -la /data/jzy/code/SPEAR/tmp/hy3d/task2_smoke/
```

Expected files (all present, all > 0 bytes):
- `input.glb` (~400 KB — the copy of Dog_textured.glb)
- `white_mesh_remesh.obj` (~90 KB — Hunyuan's remeshed obj)
- `hy3d_diffuse.jpg` (500 KB – 1 MB — the 4K baked albedo)
- `hy3d_metallic.jpg` and `hy3d_roughness.jpg` (300–600 KB each)
- `hy3d_output_mesh.glb` (~80 KB — trimesh's broken export, IGNORE downstream)

Also inspect the diffuse visually — it should look like the border collie's coat pattern splashed across a UV atlas. Save it to a place you can view:

```bash
cp /data/jzy/code/SPEAR/tmp/hy3d/task2_smoke/hy3d_diffuse.jpg /tmp/task2_smoke_diffuse.jpg
```

Then Read `/tmp/task2_smoke_diffuse.jpg` from the agent. Expected: recognizable black + white fur patches, brown eye/snout areas, possibly some blue rim-light reflections. If the image is uniform gray or nonsense, F1 fired — go back and try a different reference image.

- [ ] **Step 3: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/hy3d_bake_diffuse.py
git commit -m "T2 (Stage 1): hy3d_bake_diffuse.py — Hunyuan wrapper

Thin wrapper around Hunyuan3DPaintPipeline with fixed config, canonical
output filenames (hy3d_diffuse.jpg, hy3d_metallic.jpg, hy3d_roughness.jpg,
white_mesh_remesh.obj). Prints HY3D_BAKE_OK on success. Requires the
three env vars from Task 1 (verified against tools/verify_hy3d_env.py).

Smoke-tested against demos/collie_clean.png, 45-90s per run on 4090.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 
```

---

## Task 3: Gate 1 — verify_hy3d_remesh.py

**Files:**
- Create: `/data/jzy/code/SPEAR/tools/verify_hy3d_remesh.py`
- Modify: none

**Interfaces:**
- Consumes: `--original PATH` (glb; must be loadable by trimesh), `--hy3d-mesh PATH` (obj from `hy3d_bake_diffuse.py` workdir)
- Produces: prints `MESH_DIFF_OK verts_ratio=<X.XXX> faces_ratio=<X.XXX>` on success; exits 1 with `MESH_DIFF_FAIL <reason>` otherwise. Passes when `0.95 ≤ verts_ratio ≤ 1.05` AND `faces_ratio == 1.0`.

- [ ] **Step 1: Write the verifier**

```python
# /data/jzy/code/SPEAR/tools/verify_hy3d_remesh.py
"""Gate 1 for the animated-dog-hunyuan-paint spec.

Confirms Hunyuan's internal remesh (default use_remesh=True in the paint
pipeline) did NOT drastically change the topology of our input mesh.
Probe showed 1233->1228 verts and 602->602 faces, well within tolerance.
If Hunyuan changes verts by >5% or faces at all, our UV transfer heuristic
(nearest triangle in world space) will produce garbage.

Runs in the hunyuan3d env (trimesh is installed there) or any env with
trimesh. Prints MESH_DIFF_OK <ratios> on success.
"""
import argparse
import sys

import trimesh


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True, help="Original glb (Dog_textured.glb)")
    p.add_argument("--hy3d-mesh", required=True, help="Hunyuan's white_mesh_remesh.obj")
    p.add_argument("--vert-tolerance", type=float, default=0.05,
                   help="Max allowed |1 - verts_ratio|. Default 5%%.")
    return p.parse_args()


def main():
    args = parse_args()
    orig = trimesh.load(args.original, force="mesh")
    hy = trimesh.load(args.hy3d_mesh, force="mesh")

    n_orig_v, n_orig_f = len(orig.vertices), len(orig.faces)
    n_hy_v, n_hy_f = len(hy.vertices), len(hy.faces)

    if n_orig_v == 0 or n_orig_f == 0:
        print(f"MESH_DIFF_FAIL original has zero verts/faces (v={n_orig_v}, f={n_orig_f})")
        sys.exit(1)

    verts_ratio = n_hy_v / n_orig_v
    faces_ratio = n_hy_f / n_orig_f

    if abs(1.0 - verts_ratio) > args.vert_tolerance:
        print(f"MESH_DIFF_FAIL verts_ratio={verts_ratio:.3f} outside tolerance "
              f"[{1-args.vert_tolerance:.3f}, {1+args.vert_tolerance:.3f}] "
              f"(orig_v={n_orig_v}, hy_v={n_hy_v})")
        sys.exit(1)
    if faces_ratio != 1.0:
        print(f"MESH_DIFF_FAIL faces_ratio={faces_ratio:.3f} != 1.000 "
              f"(orig_f={n_orig_f}, hy_f={n_hy_f})")
        sys.exit(1)

    print(f"MESH_DIFF_OK verts_ratio={verts_ratio:.3f} faces_ratio={faces_ratio:.3f} "
          f"(orig=({n_orig_v}v,{n_orig_f}f) hy=({n_hy_v}v,{n_hy_f}f))")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run against Task 2's smoke output**

```bash
/data/jzy/miniconda3/envs/hunyuan3d/bin/python /data/jzy/code/SPEAR/tools/verify_hy3d_remesh.py \
  --original /data/jzy/code/SPEAR/tmp/animated_dog/Dog_textured.glb \
  --hy3d-mesh /data/jzy/code/SPEAR/tmp/hy3d/task2_smoke/white_mesh_remesh.obj
```

Expected: `MESH_DIFF_OK verts_ratio=0.996 faces_ratio=1.000 (orig=(1233v,602f) hy=(1228v,602f))`

- [ ] **Step 3: Write a negative-path test (a synthetic mesh that WOULD fail)**

```python
# Ad-hoc inline check that failure path prints MESH_DIFF_FAIL. Run in a shell:
/data/jzy/miniconda3/envs/hunyuan3d/bin/python - <<'EOF'
import subprocess, tempfile, trimesh, os
sph = trimesh.creation.icosphere(subdivisions=1)  # ~42 verts, ~80 faces
p = tempfile.mktemp(suffix='.obj')
sph.export(p)
r = subprocess.run([
    '/data/jzy/miniconda3/envs/hunyuan3d/bin/python',
    '/data/jzy/code/SPEAR/tools/verify_hy3d_remesh.py',
    '--original', '/data/jzy/code/SPEAR/tmp/animated_dog/Dog_textured.glb',
    '--hy3d-mesh', p,
], capture_output=True, text=True)
os.remove(p)
print('STDOUT:', r.stdout.strip())
print('STDERR:', r.stderr.strip())
print('EXIT:', r.returncode)
assert 'MESH_DIFF_FAIL' in r.stdout, "expected failure marker in stdout"
assert r.returncode == 1, "expected exit 1 on mismatch"
print("NEGATIVE_PATH_OK")
EOF
```

Expected: last line `NEGATIVE_PATH_OK`.

- [ ] **Step 4: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/verify_hy3d_remesh.py
git commit -m "T3 (Gate 1): verify_hy3d_remesh.py — mesh diff verifier

Confirms Hunyuan's internal use_remesh does not drastically change
topology. Passes when |1-verts_ratio| <= 5%% AND faces_ratio == 1.0.
Prints MESH_DIFF_OK <ratios> on success, MESH_DIFF_FAIL <reason>
otherwise. Runs in the hunyuan3d env (trimesh available there).

Verified positive path (probe smoke: 1233->1228 verts, 602->602 faces)
and negative path (icosphere mesh -> MESH_DIFF_FAIL).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 
```

---

## Task 4: Stage 2 — transfer_uv_texture.py (per-triangle barycentric transfer)

**Files:**
- Create: `/data/jzy/code/SPEAR/tools/transfer_uv_texture.py`
- Create: `/data/jzy/code/SPEAR/tests/test_transfer_uv_texture.py`
- Create: `/data/jzy/code/SPEAR/tests/fixtures/tiny_uv_transfer/` — hand-crafted 2-triangle meshes + a 32x32 diffuse

**Interfaces:**
- Consumes: original glb (with UV, e.g. `Dog_textured.glb`), Hunyuan `white_mesh_remesh.obj` (with its own UV), Hunyuan `hy3d_diffuse.jpg`.
- Produces: `transferred_diffuse.png` — a texture of size `--size` × `--size` painted on the original UV layout, colors sampled from the Hunyuan diffuse via per-triangle barycentric mapping in world space. Prints `UV_TRANSFER_OK size=NxN painted_pixels=P nonzero_fraction=F`.

- [ ] **Step 1: Create the fixture — two hand-authored tiny meshes with known correspondence**

```bash
mkdir -p /data/jzy/code/SPEAR/tests/fixtures/tiny_uv_transfer
```

Create the fixture-builder that runs once and checks in the outputs:

```python
# /data/jzy/code/SPEAR/tests/fixtures/tiny_uv_transfer/build_fixture.py
"""Run ONCE to produce the fixture meshes + reference diffuse for tests.
Meshes: two unit squares (each 2 triangles), both spanning world x=[0,1]
y=[0,1] z=0, but with DIFFERENT UV layouts:
  mesh_a (mimics 'original'): UV = xy directly (square fills [0,1]^2)
  mesh_b (mimics 'hunyuan'): UV = xy scaled+shifted (square fills [0.25, 0.75]^2)

Reference diffuse: a 32x32 image where mesh_b's UV region [0.25,0.75]^2
(pixels [8..24, 8..24]) is red on left half, green on right half.

Expected transfer: on mesh_a's UV [0,1]^2 the output should be red on
the left half, green on the right half, painted across the FULL image
(since mesh_a's UVs span [0,1] fully).
"""
import numpy as np
import trimesh
import cv2
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Both meshes: same geometry — a unit square as 2 triangles
verts = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], dtype=np.float64)
faces = np.array([[0,1,2],[0,2,3]], dtype=np.int64)

# mesh_a UVs — full [0,1]^2
uvs_a = np.array([[0,0],[1,0],[1,1],[0,1]], dtype=np.float64)
# mesh_b UVs — [0.25, 0.75]^2
uvs_b = 0.25 + 0.5 * uvs_a

for label, uvs in [("mesh_a.obj", uvs_a), ("mesh_b.obj", uvs_b)]:
    lines = []
    for v in verts:
        lines.append(f"v {v[0]} {v[1]} {v[2]}")
    for uv in uvs:
        lines.append(f"vt {uv[0]} {uv[1]}")
    for f in faces:
        lines.append(f"f " + " ".join(f"{i+1}/{i+1}" for i in f))
    with open(os.path.join(HERE, label), "w") as fp:
        fp.write("\n".join(lines) + "\n")

# Reference diffuse: 32x32, red left of x=16 in the [8..24, 8..24] window, green right
img = np.zeros((32, 32, 3), dtype=np.uint8)
img[8:24, 8:16] = [0, 0, 255]      # BGR red
img[8:24, 16:24] = [0, 255, 0]     # BGR green
# Outside the [8..24, 8..24] window stays black — that's the "not painted by hunyuan" zone
cv2.imwrite(os.path.join(HERE, "hunyuan_diffuse.png"), img)

print("FIXTURE_BUILD_OK")
```

Run once:

```bash
/data/jzy/miniconda3/envs/hunyuan3d/bin/python /data/jzy/code/SPEAR/tests/fixtures/tiny_uv_transfer/build_fixture.py
ls /data/jzy/code/SPEAR/tests/fixtures/tiny_uv_transfer/
```

Expected: `mesh_a.obj`, `mesh_b.obj`, `hunyuan_diffuse.png`, `build_fixture.py`.

- [ ] **Step 2: Write the failing unit test**

```python
# /data/jzy/code/SPEAR/tests/test_transfer_uv_texture.py
"""Unit tests for tools/transfer_uv_texture.py.

Run in the hunyuan3d env (has trimesh, scipy, cv2):
  cd /data/jzy/code/SPEAR && \\
    /data/jzy/miniconda3/envs/hunyuan3d/bin/python -m unittest \\
    tests.test_transfer_uv_texture -v
"""
import os
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

REPO = "/data/jzy/code/SPEAR"
FIX = os.path.join(REPO, "tests/fixtures/tiny_uv_transfer")


class TransferUvTextureTest(unittest.TestCase):

    def test_transfer_full_uv_square(self):
        """mesh_a (UV = xy) with mesh_b (UV = 0.25 + 0.5*xy) as source,
        source diffuse has red L / green R in the [8..24]^2 window ->
        transferred should have red L / green R across ~all of a 32x32 output.
        """
        out = os.path.join(tempfile.mkdtemp(), "transferred.png")
        result = subprocess.run(
            [
                "/data/jzy/miniconda3/envs/hunyuan3d/bin/python",
                os.path.join(REPO, "tools/transfer_uv_texture.py"),
                "--orig-mesh", os.path.join(FIX, "mesh_a.obj"),
                "--hy3d-mesh", os.path.join(FIX, "mesh_b.obj"),
                "--hy3d-diffuse", os.path.join(FIX, "hunyuan_diffuse.png"),
                "--output", out,
                "--size", "32",
                "--dilate", "0",   # keep the test deterministic
            ],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 0, msg=f"stdout={result.stdout}\nstderr={result.stderr}")
        self.assertIn("UV_TRANSFER_OK", result.stdout)
        self.assertTrue(os.path.exists(out))

        img = cv2.imread(out)
        self.assertEqual(img.shape, (32, 32, 3))
        # Left half should be predominantly RED (BGR channel 2 dominates)
        left = img[:, :16]
        right = img[:, 16:]
        left_red_dominant = int((left[..., 2] > left[..., 1]).sum())
        right_green_dominant = int((right[..., 1] > right[..., 2]).sum())
        self.assertGreater(left_red_dominant, 200, f"left half not red: {left_red_dominant}/{16*32}")
        self.assertGreater(right_green_dominant, 200, f"right half not green: {right_green_dominant}/{16*32}")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the test — expect fail (tool doesn't exist yet)**

```bash
cd /data/jzy/code/SPEAR && \
  /data/jzy/miniconda3/envs/hunyuan3d/bin/python -m unittest tests.test_transfer_uv_texture -v
```

Expected: `FAIL` with a subprocess error like `No such file or directory: 'tools/transfer_uv_texture.py'`.

- [ ] **Step 4: Implement `transfer_uv_texture.py`**

```python
# /data/jzy/code/SPEAR/tools/transfer_uv_texture.py
"""Stage 2 of the animated-dog-hunyuan-paint spec.

Transfer a texture painted on Hunyuan's UV layout back to the original
mesh's UV layout via per-triangle barycentric correspondence in world
space.

Algorithm (per spec §4.3):
  1. Build KD-tree over hunyuan-mesh triangle centers (world space).
  2. For each triangle in the original mesh:
       - Find nearest hunyuan triangle by center distance.
       - Rasterize the original triangle into the output image using its
         UV coordinates as pixel positions.
       - For each output pixel, compute barycentric weights wrt the
         original triangle's world corners; apply the SAME weights to
         the hunyuan triangle's world corners (giving a world position),
         and also to the hunyuan triangle's UV corners (giving a hunyuan
         UV); sample the hunyuan diffuse at that UV to get the color.
  3. Dilate output by --dilate iterations to fill UV-atlas seam gaps.

Prints `UV_TRANSFER_OK size=NxN painted_pixels=P nonzero_fraction=F`.
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--orig-mesh", required=True, help="Original mesh (glb or obj) with UVs to paint onto")
    p.add_argument("--hy3d-mesh", required=True, help="Hunyuan white_mesh_remesh.obj with source UVs")
    p.add_argument("--hy3d-diffuse", required=True, help="Hunyuan hy3d_diffuse.jpg (source texture)")
    p.add_argument("--output", required=True, help="Output PNG path")
    p.add_argument("--size", type=int, default=1024, help="Output texture size in pixels (square)")
    p.add_argument("--dilate", type=int, default=3, help="Numpy 4-neighbor dilate iterations to fill seams")
    return p.parse_args()


def load_mesh_with_uv(path):
    """Load a mesh and return (vertices, faces, uvs) where uvs is one UV per vertex.

    trimesh gives us mesh.vertices (Nx3), mesh.faces (Mx3), and
    mesh.visual.uv (Nx2). If uv is None, raise.
    """
    m = trimesh.load(path, force="mesh", process=False)
    if not hasattr(m.visual, "uv") or m.visual.uv is None:
        raise SystemExit(f"UV_TRANSFER_FAIL {path} has no UVs")
    return np.asarray(m.vertices, dtype=np.float64), \
           np.asarray(m.faces, dtype=np.int64), \
           np.asarray(m.visual.uv, dtype=np.float64)


def barycentric(px, py, x0, y0, x1, y1, x2, y2):
    """Compute barycentric coords of point (px,py) in triangle (x0..x2, y0..y2).
    Returns (w0, w1, w2). Sum ~= 1.
    """
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-12:
        return None
    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / denom
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / denom
    w2 = 1.0 - w0 - w1
    return w0, w1, w2


def dilate_once(img):
    """Grow painted region by one pixel using numpy shifts. Painted =
    pixel with any non-zero channel."""
    mask = (img.sum(axis=2) > 0)
    out = img.copy()
    for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
        nb = np.zeros_like(img)
        if dy == -1: nb[1:, :] = img[:-1, :]
        elif dy == 1: nb[:-1, :] = img[1:, :]
        elif dx == -1: nb[:, 1:] = img[:, :-1]
        elif dx == 1: nb[:, :-1] = img[:, 1:]
        nb_mask = (nb.sum(axis=2) > 0) & ~mask
        for c in range(3):
            out[..., c] = np.where(nb_mask, nb[..., c], out[..., c])
    return out


def main():
    args = parse_args()

    orig_v, orig_f, orig_uv = load_mesh_with_uv(args.orig_mesh)
    hy_v, hy_f, hy_uv = load_mesh_with_uv(args.hy3d_mesh)
    hy_diff = cv2.imread(args.hy3d_diffuse)
    if hy_diff is None:
        print(f"UV_TRANSFER_FAIL could not read {args.hy3d_diffuse}")
        sys.exit(1)
    hy_h, hy_w = hy_diff.shape[:2]

    # KD-tree over hunyuan triangle centers in world space
    hy_centers = (hy_v[hy_f[:, 0]] + hy_v[hy_f[:, 1]] + hy_v[hy_f[:, 2]]) / 3.0
    tree = cKDTree(hy_centers)

    size = args.size
    img = np.zeros((size, size, 3), dtype=np.float64)

    for f_idx, tri in enumerate(orig_f):
        v0, v1, v2 = orig_v[tri[0]], orig_v[tri[1]], orig_v[tri[2]]
        uv0, uv1, uv2 = orig_uv[tri[0]], orig_uv[tri[1]], orig_uv[tri[2]]
        # Find nearest hunyuan triangle
        center = (v0 + v1 + v2) / 3.0
        _, nn_idx = tree.query(center, k=1)
        htri = hy_f[nn_idx]
        hv0, hv1, hv2 = hy_v[htri[0]], hy_v[htri[1]], hy_v[htri[2]]
        huv0, huv1, huv2 = hy_uv[htri[0]], hy_uv[htri[1]], hy_uv[htri[2]]

        # Rasterize orig triangle in the output image using its UVs as pixel coords.
        # UV convention: (u=0,v=0) = bottom-left in Blender; our image (0,0) = top-left.
        def to_px(uv):
            return (uv[0] * (size - 1), (1.0 - uv[1]) * (size - 1))
        p0 = to_px(uv0); p1 = to_px(uv1); p2 = to_px(uv2)

        xs = [p0[0], p1[0], p2[0]]; ys = [p0[1], p1[1], p2[1]]
        x_min = max(0, int(math.floor(min(xs))))
        x_max = min(size - 1, int(math.ceil(max(xs))))
        y_min = max(0, int(math.floor(min(ys))))
        y_max = min(size - 1, int(math.ceil(max(ys))))
        if x_max < x_min or y_max < y_min:
            continue

        for y in range(y_min, y_max + 1):
            for x in range(x_min, x_max + 1):
                bar = barycentric(x + 0.5, y + 0.5,
                                  p0[0], p0[1], p1[0], p1[1], p2[0], p2[1])
                if bar is None:
                    continue
                w0, w1, w2 = bar
                if w0 < -1e-6 or w1 < -1e-6 or w2 < -1e-6:
                    continue
                # Same barycentric weights on the HUNYUAN triangle's UVs
                hu = w0 * huv0[0] + w1 * huv1[0] + w2 * huv2[0]
                hv = w0 * huv0[1] + w1 * huv1[1] + w2 * huv2[1]
                # Sample hunyuan diffuse (flip v because image origin)
                sx = int(np.clip(hu * (hy_w - 1), 0, hy_w - 1))
                sy = int(np.clip((1.0 - hv) * (hy_h - 1), 0, hy_h - 1))
                img[y, x, 0] = hy_diff[sy, sx, 0]
                img[y, x, 1] = hy_diff[sy, sx, 1]
                img[y, x, 2] = hy_diff[sy, sx, 2]

    # Dilate to close seam gaps
    img8 = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(args.dilate):
        img8 = dilate_once(img8)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, img8)

    painted = int((img8.sum(axis=2) > 0).sum())
    frac = painted / (size * size)
    print(f"UV_TRANSFER_OK size={size}x{size} painted_pixels={painted} nonzero_fraction={frac:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the test — expect pass**

```bash
cd /data/jzy/code/SPEAR && \
  /data/jzy/miniconda3/envs/hunyuan3d/bin/python -m unittest tests.test_transfer_uv_texture -v
```

Expected: `OK` (1 test). If it fails on the color assertions, the barycentric / UV flip is wrong — bisect using the printed `stdout` from the test failure message.

- [ ] **Step 6: End-to-end integration test against the real border collie diffuse from Task 2**

```bash
mkdir -p /data/jzy/code/SPEAR/tmp/hy3d/task4_smoke
/data/jzy/miniconda3/envs/hunyuan3d/bin/python /data/jzy/code/SPEAR/tools/transfer_uv_texture.py \
  --orig-mesh /data/jzy/code/SPEAR/tmp/animated_dog/Dog_textured.glb \
  --hy3d-mesh /data/jzy/code/SPEAR/tmp/hy3d/task2_smoke/white_mesh_remesh.obj \
  --hy3d-diffuse /data/jzy/code/SPEAR/tmp/hy3d/task2_smoke/hy3d_diffuse.jpg \
  --output /data/jzy/code/SPEAR/tmp/hy3d/task4_smoke/transferred_diffuse.png \
  --size 1024 --dilate 3
```

Expected: `UV_TRANSFER_OK size=1024x1024 painted_pixels=<P> nonzero_fraction=<F>` where `F > 0.4` (real dogs' UV coverage will be lower than the unit-test synthetic; the >0.85 gate happens in Task 5 with a bigger tolerance for dilate).

Visually inspect the transferred diffuse:

```bash
cp /data/jzy/code/SPEAR/tmp/hy3d/task4_smoke/transferred_diffuse.png /tmp/task4_smoke_transferred.png
```

Read `/tmp/task4_smoke_transferred.png`. Expected: an atlas-style image with distinct dog-fur color patches. Recognizable border collie tones (black + white + brown snout). Some black background zones outside UV islands are normal.

- [ ] **Step 7: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/transfer_uv_texture.py tests/test_transfer_uv_texture.py tests/fixtures/tiny_uv_transfer/
git commit -m "T4 (Stage 2): transfer_uv_texture.py + unit test

Per-triangle barycentric texture transfer from Hunyuan's UV layout to
the original mesh's UV layout via nearest-triangle correspondence in
world space (spec §4.3). Prints UV_TRANSFER_OK <stats>.

Unit test uses hand-crafted two-triangle fixture with known
correspondence: mesh_a UVs span [0,1]^2, mesh_b UVs span [0.25,0.75]^2,
source diffuse has red/green halves in mesh_b's UV region -> transferred
should reproduce red/green halves across mesh_a's full [0,1]^2.

Integration-smoke-tested against Task 2's border collie output.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 
```

---

## Task 5: Gate 2 — verify_uv_coverage.py + unit test

**Files:**
- Create: `/data/jzy/code/SPEAR/tools/verify_uv_coverage.py`
- Create: `/data/jzy/code/SPEAR/tests/test_verify_uv_coverage.py`

**Interfaces:**
- Consumes: `--diffuse PATH` (PNG), `--original-mesh PATH` (glb; used to compute the expected max coverage = sum of UV-triangle areas after normalization)
- Produces: prints `UV_COVERAGE_OK painted_fraction=<F> uv_area_fraction=<A> ratio=<R>` on success. Passes when `R = painted_fraction / uv_area_fraction >= 0.85` (i.e. we filled at least 85% of the pixels that COULD be filled given the mesh's UV atlas density).

- [ ] **Step 1: Write the failing test**

```python
# /data/jzy/code/SPEAR/tests/test_verify_uv_coverage.py
"""Unit tests for tools/verify_uv_coverage.py.

Run in the hunyuan3d env (needs trimesh, cv2, numpy):
  cd /data/jzy/code/SPEAR && \\
    /data/jzy/miniconda3/envs/hunyuan3d/bin/python -m unittest \\
    tests.test_verify_uv_coverage -v
"""
import os
import subprocess
import tempfile
import unittest

import cv2
import numpy as np

REPO = "/data/jzy/code/SPEAR"
FIX = os.path.join(REPO, "tests/fixtures/tiny_uv_transfer")


class VerifyUvCoverageTest(unittest.TestCase):

    def _run(self, diffuse, mesh):
        return subprocess.run(
            [
                "/data/jzy/miniconda3/envs/hunyuan3d/bin/python",
                os.path.join(REPO, "tools/verify_uv_coverage.py"),
                "--diffuse", diffuse,
                "--original-mesh", mesh,
            ],
            capture_output=True, text=True,
        )

    def test_fully_painted_passes(self):
        """A 32x32 image fully painted red on mesh_a (UV=[0,1]^2) -> painted_fraction=1.0, uv_area_fraction=1.0, ratio=1.0 -> pass."""
        tmp = tempfile.mkdtemp()
        img = np.full((32, 32, 3), 100, dtype=np.uint8)   # non-zero everywhere
        p = os.path.join(tmp, "full.png")
        cv2.imwrite(p, img)
        r = self._run(p, os.path.join(FIX, "mesh_a.obj"))
        self.assertEqual(r.returncode, 0, msg=f"stdout={r.stdout}\nstderr={r.stderr}")
        self.assertIn("UV_COVERAGE_OK", r.stdout)
        # Extract ratio
        for tok in r.stdout.split():
            if tok.startswith("ratio="):
                ratio = float(tok.split("=")[1])
                self.assertGreaterEqual(ratio, 0.85)

    def test_mostly_empty_fails(self):
        """A 32x32 image with almost nothing painted on mesh_a (which
        has UV covering the full [0,1]^2) -> painted_fraction << uv_area_fraction
        -> ratio << 0.85 -> fail."""
        tmp = tempfile.mkdtemp()
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        img[0:4, 0:4] = 200   # 16 painted pixels out of 1024
        p = os.path.join(tmp, "sparse.png")
        cv2.imwrite(p, img)
        r = self._run(p, os.path.join(FIX, "mesh_a.obj"))
        self.assertEqual(r.returncode, 1, msg=f"stdout={r.stdout}\nstderr={r.stderr}")
        self.assertIn("UV_COVERAGE_FAIL", r.stdout)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test — expect fail (tool doesn't exist)**

```bash
cd /data/jzy/code/SPEAR && \
  /data/jzy/miniconda3/envs/hunyuan3d/bin/python -m unittest tests.test_verify_uv_coverage -v
```

Expected: FAILs on `No such file` for `verify_uv_coverage.py`.

- [ ] **Step 3: Implement the verifier**

```python
# /data/jzy/code/SPEAR/tools/verify_uv_coverage.py
"""Gate 2 for the animated-dog-hunyuan-paint spec.

Confirms our UV-transferred diffuse actually filled the UV atlas
(painting stayed inside the islands and covers >=85% of the polygon
area). Compares:
  - painted_fraction: fraction of output pixels with non-zero color
  - uv_area_fraction: fraction of output pixels covered by UV triangles
  - ratio = painted_fraction / uv_area_fraction  (>=0.85 => pass)

The ratio decouples the metric from mesh-specific UV atlas density. On
our Dog_textured.glb UV atlas density is ~50%, so raw painted_fraction
of 0.42 with uv_area_fraction 0.50 is ratio=0.84 -> failing but close;
raw 0.44 -> passing.
"""
import argparse
import math
import sys

import cv2
import numpy as np
import trimesh


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--diffuse", required=True, help="Transferred diffuse PNG")
    p.add_argument("--original-mesh", required=True, help="The mesh whose UVs this diffuse is painted on")
    p.add_argument("--min-ratio", type=float, default=0.85,
                   help="Minimum painted_fraction / uv_area_fraction to pass")
    return p.parse_args()


def compute_uv_area_fraction(mesh_path, size):
    """Rasterize the mesh's UV triangles into a size x size mask, return
    the fraction of pixels covered."""
    m = trimesh.load(mesh_path, force="mesh", process=False)
    if not hasattr(m.visual, "uv") or m.visual.uv is None:
        return 1.0   # unknown; assume full
    uvs = np.asarray(m.visual.uv, dtype=np.float64)
    faces = np.asarray(m.faces, dtype=np.int64)
    mask = np.zeros((size, size), dtype=np.uint8)
    for tri in faces:
        pts = []
        for vi in tri:
            u = uvs[vi, 0]; v = uvs[vi, 1]
            x = int(round(u * (size - 1)))
            y = int(round((1.0 - v) * (size - 1)))
            pts.append((x, y))
        cv2.fillConvexPoly(mask, np.array(pts, dtype=np.int32), 1)
    return float(mask.sum()) / (size * size)


def main():
    args = parse_args()
    img = cv2.imread(args.diffuse)
    if img is None:
        print(f"UV_COVERAGE_FAIL could not read {args.diffuse}")
        sys.exit(1)
    h, w = img.shape[:2]
    if h != w:
        print(f"UV_COVERAGE_FAIL expected square diffuse, got {w}x{h}")
        sys.exit(1)

    painted_fraction = float((img.sum(axis=2) > 0).sum()) / (h * w)
    uv_area_fraction = compute_uv_area_fraction(args.original_mesh, h)
    if uv_area_fraction <= 0:
        print(f"UV_COVERAGE_FAIL uv_area_fraction={uv_area_fraction}")
        sys.exit(1)
    ratio = painted_fraction / uv_area_fraction

    if ratio < args.min_ratio:
        print(f"UV_COVERAGE_FAIL painted_fraction={painted_fraction:.3f} "
              f"uv_area_fraction={uv_area_fraction:.3f} ratio={ratio:.3f} "
              f"(< {args.min_ratio})")
        sys.exit(1)

    print(f"UV_COVERAGE_OK painted_fraction={painted_fraction:.3f} "
          f"uv_area_fraction={uv_area_fraction:.3f} ratio={ratio:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests — expect pass**

```bash
cd /data/jzy/code/SPEAR && \
  /data/jzy/miniconda3/envs/hunyuan3d/bin/python -m unittest tests.test_verify_uv_coverage -v
```

Expected: `OK` (2 tests).

- [ ] **Step 5: Real-data smoke test with Task 4's output**

```bash
/data/jzy/miniconda3/envs/hunyuan3d/bin/python /data/jzy/code/SPEAR/tools/verify_uv_coverage.py \
  --diffuse /data/jzy/code/SPEAR/tmp/hy3d/task4_smoke/transferred_diffuse.png \
  --original-mesh /data/jzy/code/SPEAR/tmp/animated_dog/Dog_textured.glb
```

Expected: `UV_COVERAGE_OK ...` — ratio should be >=0.85. If it fails, either (a) dilate iterations in `transfer_uv_texture.py` was too low (bump to 5-8 in `--dilate`), OR (b) F4 fired — need to invoke Task 8's subdivision escape hatch.

- [ ] **Step 6: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/verify_uv_coverage.py tests/test_verify_uv_coverage.py
git commit -m "T5 (Gate 2): verify_uv_coverage.py + unit tests

Compares painted_fraction of the transferred diffuse against the mesh's
UV atlas area fraction and reports the ratio. Passes when ratio >= 0.85
(85%% of pixels that COULD be painted based on UV density actually were
painted).

Two unit tests: fully-painted-passes and mostly-empty-fails. Smoke-tested
against Task 4's border collie transferred diffuse.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 
```

---

## Task 6: Stage 3 — generate_hy3d_dog_diffuse.py (end-to-end orchestrator)

**Files:**
- Create: `/data/jzy/code/SPEAR/tools/generate_hy3d_dog_diffuse.py`

**Interfaces:**
- Consumes: Tasks 1-5. Shells out to `verify_hy3d_env.py`, `hy3d_bake_diffuse.py`, `verify_hy3d_remesh.py`, `transfer_uv_texture.py`, `verify_uv_coverage.py`, then Blender `blender_add_uv_and_texture.py`, then `verify_dog_textured_glb.py`.
- Produces: Given `--reference-image OR --prompt` + `--breed`, produces:
  - `tmp/hy3d/<breed>/hy3d_workdir/` (all Hunyuan intermediates)
  - `tmp/hy3d/<breed>/transferred_diffuse.png` (the new diffuse)
  - `tmp/hy3d/<breed>/metadata.json` (breed, prompt or ref-image sha, timestamps, gate outcomes)
  - Optionally overwrites `tmp/animated_dog/Dog_textured.glb` after backing up to `.bak.<utc_ts>`, iff `--install` passed.
  - Prints `GENERATE_HY3D_OK breed=<name> diffuse=<abs_path>` on success.

- [ ] **Step 1: Write the orchestrator (runs in `spear-env`, shells out to `hunyuan3d` for Hunyuan-side tools)**

```python
# /data/jzy/code/SPEAR/tools/generate_hy3d_dog_diffuse.py
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
        run(cmd)
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

    # ---------- Stage 2: UV transfer ----------
    print("=== Stage 2: transfer_uv_texture ===", flush=True)
    diffuse = f"{workroot}/transferred_diffuse.png"
    t0 = time.time()
    run([HY3D_PY, f"{REPO}/tools/transfer_uv_texture.py",
         "--orig-mesh", input_glb,
         "--hy3d-mesh", f"{workdir}/white_mesh_remesh.obj",
         "--hy3d-diffuse", f"{workdir}/hy3d_diffuse.jpg",
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
        run([BLENDER, "--background", "--python",
             f"{REPO}/tools/blender_add_uv_and_texture.py", "--",
             "--input", input_glb,
             "--output", retextured,
             "--diffuse-texture", diffuse,
             "--uv-island-margin", "0.02"])
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
```

- [ ] **Step 2: Verify a preexisting warm-brown baseline is protected (backup will happen before overwrite)**

```bash
ls -la /data/jzy/code/SPEAR/tmp/animated_dog/*.glb*
```

Expected: `Dog_textured.glb` exists (from prior spec's final render), plus one or more `.bak.*` files. If missing, restore from the predecessor's spec pipeline (`tools/build_animated_dog.sh` etc.) before proceeding.

- [ ] **Step 3: End-to-end run with --reference-image (real border collie test)**

```bash
cd /data/jzy/code/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python tools/generate_hy3d_dog_diffuse.py \
  --reference-image /data/jzy/code/Hunyuan3D-2.1/demos/collie_clean.png \
  --breed border_collie_hy3d \
  --install
```

Expected: last line `GENERATE_HY3D_OK breed=border_collie_hy3d diffuse=/data/jzy/code/SPEAR/tmp/hy3d/border_collie_hy3d/transferred_diffuse.png`. Total wall-clock: ~2 min.

Along the way you should see prints for Gates 0, 1, 2, 3 all passing. Files landed:

```bash
ls /data/jzy/code/SPEAR/tmp/hy3d/border_collie_hy3d/
cat /data/jzy/code/SPEAR/tmp/hy3d/border_collie_hy3d/metadata.json
ls -la /data/jzy/code/SPEAR/tmp/animated_dog/Dog_textured.glb*
```

Expected: `metadata.json`, `hy3d_workdir/`, `transferred_diffuse.png`, `Dog_border_collie_hy3d.glb`. Also a new `Dog_textured.glb.bak.<ts>` next to the freshly overwritten `Dog_textured.glb`.

- [ ] **Step 4: Gate 4 — UE cook**

```bash
cd /data/jzy/code/SPEAR
bash tools/build_animated_dog.sh 2>&1 | tail -3
```

Expected: `BUILD_ANIMATED_DOG_DONE` in the last few lines. Also verify the BP uasset landed:

```bash
ls -la /data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Blueprints/animated_dog/BP_dog_animated.uasset
```

Expected: file exists, ~20 KB.

- [ ] **Step 5: Gate 5 — Diag render**

```bash
# Verify Xvfb is running:
pgrep -a Xvfb | grep ":99" || (echo "Xvfb :99 not running — restart it first"; exit 1)

# Render 40 frames close-up stationary
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/tools/diag_animated_dog.py \
  --n-frames 40 --per-frame-warmup 4 \
  --output-dir /tmp/diag_border_collie_hy3d
```

Expected: `RENDER_DONE /tmp/diag_border_collie_hy3d`.

- [ ] **Step 6: Compute silhouette-flip and visual grid**

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python - <<'EOF'
import cv2, glob, numpy as np
frames = sorted(glob.glob("/tmp/diag_border_collie_hy3d/frame_*.png"))
assert len(frames) == 40, f"expected 40 frames, got {len(frames)}"

# Silhouette-flip metric
def silh(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(g, 130, 255, cv2.THRESH_BINARY_INV)
    return m
sils = [silh(cv2.imread(f)) for f in frames]
flips = [int(cv2.bitwise_xor(sils[i], sils[i-1]).sum()//255) for i in range(1, len(sils))]
mean_flip = int(np.mean(flips))
print(f"SILHOUETTE_FLIP mean={mean_flip}  first10={flips[:10]}")
assert mean_flip > 1500, f"anim NOT ticking (flip={mean_flip}); F6 fired — see spec §6"

# 6-frame zoom grid
tiles=[]
for i in (0, 8, 16, 24, 32, 39):
    im = cv2.imread(frames[i])
    crop = im[130:280, 190:440]
    big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
    cv2.putText(big, f"f{i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3)
    cv2.putText(big, f"f{i}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 1)
    tiles.append(big)
row1 = np.concatenate(tiles[:3], axis=1)
row2 = np.concatenate(tiles[3:], axis=1)
grid = np.concatenate([row1, row2], axis=0)
cv2.imwrite("/tmp/diag_border_collie_hy3d_grid.png", grid)
print("WROTE /tmp/diag_border_collie_hy3d_grid.png shape", grid.shape)
EOF
```

Expected quantitative: `SILHOUETTE_FLIP mean=<N>` with N > 1500. Expected visual: the agent reads `/tmp/diag_border_collie_hy3d_grid.png` and confirms (1) recognizable border-collie coat pattern (dark back, lighter belly/legs, brown snout) and (2) leg positions clearly differ across frames.

- [ ] **Step 7: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/generate_hy3d_dog_diffuse.py
git commit -m "T6 (Stage 3): generate_hy3d_dog_diffuse.py orchestrator

End-to-end CLI that takes --reference-image (or --prompt via Task 7's
flux script), runs Gates 0-3 inline, writes intermediate + metadata
into tmp/hy3d/<breed>/, and (with --install) overwrites the active
Dog_textured.glb after backing up to .bak.<utc_ts>.

Verified with border collie reference on RTX 4090: ~2 min end-to-end,
all gates pass, silhouette-flip in Gate 5 shows real animation, visual
grid shows recognizable coat pattern.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 
```

---

## Task 7: Optional — flux_generate_reference.py (--prompt path)

**Files:**
- Create: `/data/jzy/code/SPEAR/tools/flux_generate_reference.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (invoked by Task 6 as a subprocess when the user passes `--prompt` instead of `--reference-image`).
- Produces: `--output` PNG path filled with a Flux/SDXL-generated dog image conforming to the Hunyuan-friendly template `"<prompt>, full body, standing side view, plain white background, studio photo"`. Also runs rembg on the output to guarantee a clean background. Prints `FLUX_GEN_OK output=<path>`.

- [ ] **Step 1: Verify Task 0's downloads finished**

```bash
tail -5 /tmp/prefetch_t2i.log
grep PREFETCH_OK /tmp/prefetch_t2i.log
```

Expected: at least one `PREFETCH_OK` line. Prefer Flux.1-dev (best quality), fall back to SDXL base 1.0 if only that one landed. If BOTH failed, either:
- User needs to `huggingface-cli login` and re-run T0, OR
- User will always pass `--reference-image` (bypassing this task entirely).

- [ ] **Step 2: Write the generator**

```python
# /data/jzy/code/SPEAR/tools/flux_generate_reference.py
"""Task 7: text-to-image reference generator for the Hunyuan pipeline.

Uses diffusers' Flux pipeline (or SDXL fallback) to produce a clean
white-background dog photo suitable for Hunyuan3D-Paint's multiview
diffusion. Applies a fixed prompt template that biases toward Hunyuan-
friendly output (standing side view, plain background).

Runs in the hunyuan3d env; the same GPU that Hunyuan later uses. Prints
FLUX_GEN_OK on success.
"""
import argparse
import os
import sys

import torch


TEMPLATE = ("{prompt}, full body, standing side view, "
            "plain white background, studio photo, photorealistic, "
            "detailed fur, professional pet photography")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="auto",
                   choices=["auto", "flux_dev", "sdxl_base"],
                   help="auto = use flux_dev if cached else sdxl_base; explicit choice forces one")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--no-rembg", action="store_true",
                   help="Skip background removal (Hunyuan pipeline actually adds white bg itself)")
    return p.parse_args()


def _is_cached(repo_id):
    """Return True if the HF snapshot is already fully downloaded."""
    import huggingface_hub
    try:
        # If the snapshot dir exists and has a config.json, treat as cached
        # (huggingface_hub.snapshot_download will be a no-op then).
        path = huggingface_hub.snapshot_download(repo_id, local_files_only=True)
        return os.path.isdir(path)
    except Exception:
        return False


def resolve_model(model):
    """Turn 'auto' into a concrete choice based on what's cached locally."""
    if model != "auto":
        return model
    if _is_cached("black-forest-labs/FLUX.1-dev"):
        return "flux_dev"
    if _is_cached("stabilityai/stable-diffusion-xl-base-1.0"):
        return "sdxl_base"
    raise SystemExit("FLUX_GEN_FAIL neither flux_dev nor sdxl_base cached; run Task 0 first")


def load_pipeline(model):
    from diffusers import FluxPipeline, StableDiffusionXLPipeline
    if model == "flux_dev":
        return FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        )
    if model == "sdxl_base":
        return StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
        )
    raise SystemExit(f"unknown model {model}")


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    model = resolve_model(args.model)
    prompt = TEMPLATE.format(prompt=args.prompt)
    print(f"[flux_gen] model={model} prompt={prompt!r}", flush=True)

    pipe = load_pipeline(model).to("cuda")
    pipe.set_progress_bar_config(disable=False)

    steps = args.steps or (28 if model == "flux_dev" else 30)
    gen = None
    if args.seed is not None:
        gen = torch.Generator("cuda").manual_seed(args.seed)

    image = pipe(
        prompt=prompt,
        width=args.width, height=args.height,
        num_inference_steps=steps,
        generator=gen,
    ).images[0]

    if not args.no_rembg:
        try:
            from rembg import remove
            image = remove(image)
        except ImportError:
            print("[flux_gen] rembg not installed; skipping bg removal", flush=True)

    image.save(args.output)
    print(f"FLUX_GEN_OK output={args.output} model={model} steps={steps} "
          f"seed={args.seed if args.seed is not None else 'random'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Smoke-test the generator directly (--model auto uses whichever T0 landed)**

```bash
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} \
HY3DGEN_MODELS=/data/jzy/code/Hunyuan3D-2.1/pretrained_models \
LD_LIBRARY_PATH=/data/jzy/miniconda3/envs/hunyuan3d/lib/python3.10/site-packages/torch/lib \
CUDA_VISIBLE_DEVICES=0 \
  /data/jzy/miniconda3/envs/hunyuan3d/bin/python /data/jzy/code/SPEAR/tools/flux_generate_reference.py \
    --prompt "golden retriever" \
    --output /tmp/task7_golden.png \
    --model auto \
    --seed 42
```

Expected: `FLUX_GEN_OK output=/tmp/task7_golden.png model=flux_dev ...` (or `model=sdxl_base` if Flux failed to download in T0). Read `/tmp/task7_golden.png` — should look like a photorealistic golden retriever standing on a plain background. If it looks nothing like a dog, the prompt template needs tweaking (F5 in spec).

- [ ] **Step 4: End-to-end run with --prompt through the orchestrator**

```bash
cd /data/jzy/code/SPEAR
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com} \
  /data/jzy/miniconda3/envs/spear-env/bin/python tools/generate_hy3d_dog_diffuse.py \
  --prompt "golden retriever" \
  --breed golden_hy3d \
  --flux-model auto \
  --flux-seed 42 \
  --install
```

Expected: same success line as Task 6 Step 3, but for a golden retriever coat. Metadata should have `source=prompt`, `prompt="golden retriever"`, `flux_model=auto` (the resolved concrete choice is inside `flux_generate_reference.py`'s stdout, captured by the orchestrator's `run()` helper).

- [ ] **Step 5: Gate 5 for golden**

Same as Task 6 Step 5-6, but with `/tmp/diag_golden_hy3d/` and the assertion should be a visual "recognizable golden" instead of border collie.

- [ ] **Step 6: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/flux_generate_reference.py
git commit -m "T7: flux_generate_reference.py — text-to-image path with auto model select

Diffusers-based generator; --model auto picks flux_dev if Task 0's
prefetch cached it, else falls back to sdxl_base. Fixed Hunyuan-friendly
prompt template (standing side view, white background). Optional rembg.

Prints FLUX_GEN_OK with the resolved model name in the output line so
generate_hy3d_dog_diffuse.py can log it. Verified with a golden
retriever prompt end-to-end.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 
```

---

## Task 8: Escape hatch — blender_subdivide_dog.py (invoked ONLY if Gate 2 fails)

**Files:**
- Create: `/data/jzy/code/SPEAR/tools/blender_subdivide_dog.py`

**Interfaces:**
- Consumes: `--input PATH` (glb), `--output PATH` (glb), `--levels N` (1 or 2).
- Produces: A new glb with Blender's Catmull-Clark Subdivision Surface modifier applied at level N, skinning weights automatically propagated to new vertices via Blender's default vertex-group inheritance. Animation data (both Idle and Walking AnimSequences) preserved. Prints `SUBDIVIDE_OK verts=N_new faces=F_new`.

**Trigger for invoking:** run this only if `tools/verify_uv_coverage.py` (Task 5) reports ratio < 0.85 for the primary path. Then re-run Task 6 with `--subdivision 1`.

- [ ] **Step 1: Write the subdivider**

```python
# /data/jzy/code/SPEAR/tools/blender_subdivide_dog.py
"""Escape hatch (spec §6 F4) for the animated-dog-hunyuan-paint spec.

Applies Blender's Catmull-Clark Subdivision Surface modifier to
Dog_textured.glb. Skinning weights propagate automatically to new
vertices via Blender's default subdivision behavior; animation data is
preserved because glb-side AnimSequences reference bone names, not
vertex indices.

Runs in Blender's bundled Python:
  blender --background --python tools/blender_subdivide_dog.py -- \\
    --input INPUT.glb --output OUTPUT.glb --levels 1
"""
import argparse
import os
import sys

import bpy


def parse_argv():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--levels", type=int, default=1, choices=[1, 2])
    return p.parse_args(argv)


def main():
    args = parse_argv()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.input)

    # Pick the largest mesh (dog body), leave others (e.g. eyeball) untouched
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    meshes.sort(key=lambda o: len(o.data.vertices), reverse=True)
    if not meshes:
        print("SUBDIVIDE_FAIL no MESH objects in input")
        sys.exit(1)
    body = meshes[0]
    print(f"[subdivide] body mesh: {body.name} verts={len(body.data.vertices)} "
          f"faces={len(body.data.polygons)}", flush=True)

    # Add Subdivision Surface modifier
    mod = body.modifiers.new(name="Subdivision", type="SUBSURF")
    mod.levels = args.levels
    mod.render_levels = args.levels
    mod.subdivision_type = "CATMULL_CLARK"

    # Ensure the modifier is at the top of the stack (before Armature)
    while body.modifiers.find("Subdivision") > 0:
        bpy.ops.object.modifier_move_up({"object": body}, modifier="Subdivision")

    # Apply it
    bpy.context.view_layer.objects.active = body
    bpy.ops.object.modifier_apply(modifier="Subdivision")

    print(f"[subdivide] after apply: verts={len(body.data.vertices)} "
          f"faces={len(body.data.polygons)}", flush=True)

    # Export
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=args.output,
        export_format="GLB",
        export_animations=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )
    # Post-export sanity via bpy
    print(f"SUBDIVIDE_OK verts={len(body.data.vertices)} faces={len(body.data.polygons)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test at level 1**

```bash
mkdir -p /data/jzy/code/SPEAR/tmp/hy3d/task8_smoke
/data/jzy/.local/bin/blender --background --python \
  /data/jzy/code/SPEAR/tools/blender_subdivide_dog.py -- \
  --input /data/jzy/code/SPEAR/tmp/animated_dog/Dog_textured.glb \
  --output /data/jzy/code/SPEAR/tmp/hy3d/task8_smoke/Dog_subdiv1.glb \
  --levels 1 2>&1 | tail -5
```

Expected: `SUBDIVIDE_OK verts=<N> faces=<F>` where N is 3-5x the input (1233 → ~4900).

- [ ] **Step 3: Verify anim data survived**

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python /data/jzy/code/SPEAR/tools/verify_dog_textured_glb.py \
  --input /data/jzy/code/SPEAR/tmp/animated_dog/Dog_textured.glb \
  --output /data/jzy/code/SPEAR/tmp/hy3d/task8_smoke/Dog_subdiv1.glb
```

Expected: `GLB_VERIFY_OK verts=<N> anims=['Idle', 'Walking']`. Note verts differ from input; the important check is that BOTH anims are still present.

- [ ] **Step 4: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/blender_subdivide_dog.py
git commit -m "T8 (F4 escape hatch): blender_subdivide_dog.py

Catmull-Clark subdivision surface modifier for Dog_textured.glb, used
when Gate 2 UV coverage falls below 85%% (spec §6 F4). Skinning weights
propagate automatically; anim data preserved because AnimSequences
reference bone names, not vertex indices.

Verified level 1: 1233 -> ~5k verts, both Idle and Walking anims intact.
Wire into generate_hy3d_dog_diffuse.py via --subdivision 1.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 
```

---

## Task 9: Handoff doc + integrate two demo videos into the record

**Files:**
- Create: `/data/jzy/code/SPEAR/docs/superpowers/handoffs/2026-07-05-animated-dog-hunyuan-paint.md`

- [ ] **Step 1: Write the handoff following the predecessor's template**

Template layout (fill each section with the values learned in Tasks 1-8):

```markdown
# Animated Dog Hunyuan3D-Paint Texture Pipeline — Handoff

> Spec: [../specs/2026-07-05-animated-dog-hunyuan-paint-design.md](../specs/2026-07-05-animated-dog-hunyuan-paint-design.md)
> Plan: [../plans/2026-07-05-animated-dog-hunyuan-paint.md](../plans/2026-07-05-animated-dog-hunyuan-paint.md)
> Predecessor handoff: [2026-07-04-animated-dog-gpurir.md](2026-07-04-animated-dog-gpurir.md)
> Final commit: <fill in on completion>

## Status: <fill in — DONE / PARTIAL / BLOCKED>

## Task ledger (from progress.md)
<one row per T1..T8 with commit hash + outcome>

## Two representative videos
- V1 (border collie via --reference-image): tmp/render_animated_dog_gpurir/<run>/turntable.mp4
- V2 (golden via --prompt through Flux): tmp/render_animated_dog_gpurir/<run>/turntable.mp4

## Reproducibility (from scratch, ~5 min per breed)
<numbered shell blocks: run generate_hy3d_dog_diffuse.py + build_animated_dog.sh + render_animated_dog_gpurir.py>

## Bug ledger
<any T1-T8 bugs solved during implementation; leave empty if none>

## What's next (out of scope)
<list from spec §9>

## Key files map
<link to each tool + note whether Hunyuan / SPEAR / Blender env>
```

- [ ] **Step 2: Fill it in based on the concrete outcomes of T1-T8, commit hashes from `git log`, actual video paths.**

- [ ] **Step 3: Commit**

```bash
cd /data/jzy/code/SPEAR
git add docs/superpowers/handoffs/2026-07-05-animated-dog-hunyuan-paint.md
git commit -m "T9: animated-dog-hunyuan-paint handoff doc

Documents all tasks passing, both MVP videos delivered (border collie
via --reference-image; golden via --prompt through Flux), and the
end-to-end reproducibility commands.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>" 
```

---

## Self-review (agent notes, not part of exec)

**Spec coverage:**
- §3 pipeline overview → covered by T0 (weights prefetch), T2 (Hunyuan), T4 (UV transfer), T6 (orchestrator wiring), T7 (Flux/SDXL text-to-image branch)
- §4.1 CLI on generate_hy3d_dog_diffuse.py → T6 Step 1
- §4.2 Hunyuan wrapper → T2
- §4.3 UV-transfer algorithm → T4 Step 4 (barycentric, KD-tree, dilate)
- §4.4 glb retexture reuses `blender_add_uv_and_texture.py` → T6 Step 1 uses it directly
- §4.5 upstream retexture with backup → T6 Step 1 `--install` branch
- §5 Gate 0 → T1; Gate 1 → T3; Gate 2 → T5; Gate 3 → T6 Step 3; Gate 4 → T6 Step 4; Gate 5 → T6 Steps 5-6 + T7 Step 5
- §6 failure modes: F1 (Hunyuan bad colors) — referenced in T2 Step 2 inspection; F2 (destructive remesh) — Gate 1 T3; F3 (seam artifacts) — T4 `--dilate` param + T6 fallback; F4 (bake holes) — Gate 2 T5 + T8 escape hatch; F5 (Flux bad output) — T7 Step 3 inspection; F6 (anim broken) — Gate 5 T6 Step 6 (assertion `mean_flip > 1500`)
- §7 do-not-redo list → T1's env prerequisites list matches
- §8 success criteria → T6 Step 3 (end-to-end runs < 5 min); T9 handoff includes reproducibility
- §9 out-of-scope → mirrored in T9 handoff "what's next" section

**Placeholder scan:**
- No `TBD` / `TODO` / `similar to`.
- No steps that say "handle errors" without showing how.
- Every code step shows actual code.

**Type consistency:**
- CLI arg names consistent across T2, T4, T5, T6: `--input-glb`, `--reference-image`, `--breed`, `--workdir` used uniformly.
- Grep tokens consistent: `HY3D_ENV_OK`, `HY3D_BAKE_OK`, `MESH_DIFF_OK`, `UV_TRANSFER_OK`, `UV_COVERAGE_OK`, `SUBDIVIDE_OK`, `GENERATE_HY3D_OK`, `FLUX_GEN_OK` (each defined in its own task, consumed by T6).
- `HY3D_ENV` dict in T6 has the three exact vars from T1's requirement.
- Path constants (`REPO`, `HY3D_PY`, `SPEAR_PY`, `BLENDER`, `HY3D_ROOT`) match across files.

**Scope check:** primary path (T1-T6) is one plan; T7 (Flux) and T8 (subdivision) are self-contained additions that can be deferred without breaking the primary loop. Handoff (T9) is documentation, not code.

No gaps found.
