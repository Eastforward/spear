# Animated Dog Hunyuan3D-Paint Texture Pipeline — Design Spec

> Date: 2026-07-05
> Predecessor: [2026-07-04-animated-dog-gpurir-design.md](2026-07-04-animated-dog-gpurir-design.md)
> Predecessor handoff: [../handoffs/2026-07-04-animated-dog-gpurir.md](../handoffs/2026-07-04-animated-dog-gpurir.md)
> Status: draft — awaiting user approval

## 1. Goal

Upgrade the animated-dog texture pipeline from Tier-1 (single-tint breed presets) and Tier-2 procedural (back-black / belly-white patterns) to **Tier-3 AI-generated realistic-fur diffuse**. Any user prompt or reference image should yield a plausible dog coat texture (Border Collie, Golden Retriever, Husky, individual-level variations) on the same `Dog_textured.glb` mesh, with the Walking animation fully preserved.

The end deliverable is a new CLI arg on `examples/render_animated_dog_gpurir.py` (or a sibling script) that swaps in an AI-baked breed texture without touching the room / trajectory / anim / mesh pipeline.

## 2. Non-goals

- Not adding mesh detail beyond what's already in Quaternius Dog.glb (subdivision is available as an escape hatch — see §6, but the primary path uses the same 1233-vertex mesh).
- Not generating new poses / anims — Walking loop stays as-is.
- Not integrating with the RIR audio side of the dataset — that's a separate spec.
- Not building a text-to-image system from scratch — we call an existing local Flux/SDXL install if needed.

## 3. Pipeline overview

```
[user provides]                    [system generates internally]
     ↓                                        ↓
  --prompt "border collie"                (only if no image)
     ↓                                        ↓
[Flux/SDXL local inference] ──────── reference_image.png ────────┐
                                                                  │
                                     [user provides --reference-image] ─┤
                                                                        ↓
[Dog_textured.glb (1233v, Smart-UV, Walking anim)] ──────────┐  reference.png
                                                              ↓          ↓
                                              [Hunyuan3D-Paint pipeline (~45s on 4090)]
                                                              ↓
                                              [Hunyuan mesh (1228v, its own UV)
                                               + baked diffuse.jpg (4K)
                                               + metallic.jpg + roughness.jpg]
                                                              ↓
                              [UV transfer: per-vertex nearest-neighbor
                               from original mesh → Hunyuan mesh → sample its texture]
                                                              ↓
                              [new_diffuse.png painted on original UV layout]
                                                              ↓
                     [blender_add_uv_and_texture.py replaces diffuse in original glb]
                                                              ↓
                     [Dog_<breed>_hy3d.glb — 1233v, Smart-UV, Walking anim, real fur]
                                                              ↓
                     [tools/build_animated_dog.sh (unchanged) → UE cook]
                                                              ↓
                     [render_animated_dog_gpurir.py (unchanged) → mp4]
```

## 4. Components

### 4.1 Reference-image supplier (two interfaces, both optional-but-one-required)

**CLI on the new orchestrator script `tools/generate_hy3d_dog_diffuse.py`**:

```
tools/generate_hy3d_dog_diffuse.py
    --input-glb PATH        # default: tmp/animated_dog/Dog_textured.glb
    (--reference-image PATH | --prompt "..." )   # one of these required
    --breed NAME            # slug used in output filename (defaults to derived from prompt)
    --output-diffuse PATH   # where to write the new diffuse.png
    --output-glb PATH       # where to write the retextured glb
    [--flux-model NAME]     # if --prompt: which local Flux/SDXL variant
    [--subdivision LEVEL]   # 0/1/2 escape hatch — see §6
```

When `--reference-image` is given, the Flux/SDXL step is entirely skipped.

When `--prompt` is given but no reference image, run text-to-image locally (Flux.1 dev on the same 4090 that Hunyuan uses) to produce `tmp/hy3d/<breed>/reference.png`, then feed that into Hunyuan.

If both are given, `--reference-image` wins and `--prompt` is stored only in metadata.

### 4.2 Hunyuan3D-Paint wrapper (`tools/hy3d_bake_diffuse.py`)

Thin wrapper around `Hunyuan3DPaintPipeline` with fixed config (max_num_view=6, resolution=512, texture_size=4096). Uses the local weights at `/data/jzy/code/Hunyuan3D-2.1/pretrained_models/` (verified working in probe). Called with `use_remesh=True` (the default — probe showed it's nearly a no-op for our 1233-vert mesh; safer than patching Hunyuan internals).

Runs entirely in the `hunyuan3d` conda env with `HY3DGEN_MODELS` and `LD_LIBRARY_PATH` set correctly. Exit contract: writes `hy3d_mesh.obj` + `hy3d_diffuse.jpg` to a working directory, plus prints a summary line the caller can grep.

### 4.3 UV-transfer step (`tools/transfer_uv_texture.py`)

Given:
- **Original mesh** (`Dog_textured.glb`): 1233 verts, our Smart-UV layout, has bones/anim
- **Hunyuan mesh** (`hy3d_mesh.obj`): 1228 verts, Hunyuan's own UV layout, no bones
- **Hunyuan diffuse** (`hy3d_diffuse.jpg`): 4K image painted on Hunyuan's UV

Produce:
- **New diffuse** (`hy3d_transferred_diffuse.png`): 4K image painted on original mesh's UV layout

**Algorithm — per triangle in original mesh**:

1. Build a KD-tree over Hunyuan-mesh triangle centers (in 3D world space) once at startup.
2. For each triangle T_orig in original mesh:
   - Compute T_orig's center in world space
   - Query the KD-tree for the nearest Hunyuan-mesh triangle T_hy (single nearest, no k-NN needed)
   - Rasterize T_orig into the output image using its original-UV coordinates (same pixel-walk as `blender_bake_pattern_diffuse.py`)
   - For each output pixel inside T_orig: compute the pixel's barycentric coords wrt T_orig's world-space corners, apply the SAME barycentric coords to T_hy's world-space corners to get a world position, then apply them to T_hy's UV corners to get a Hunyuan UV — sample Hunyuan diffuse at that UV
3. Dilate output by 3 pixels (same numpy 4-neighbor approach as `blender_bake_pattern_diffuse.py`) to fill UV-atlas seam gaps.

**Why nearest-triangle rather than nearest-vertex**: probe showed the two meshes have identical face count (602) and topology is basically the same; triangle-level correspondence is more stable than vertex-level (bones near legs/tail have close vertices in different triangles). Rasterizing per-triangle also matches how `blender_bake_pattern_diffuse.py` walks polygons, so the two tools' code can share a helper.

**Why this is safe**: probe showed both meshes have the same 602 faces and near-identical vertex count (1233 vs 1228). Topologically they are basically the same mesh; only UV parameterization differs.

### 4.4 GLB retexture step (reuses existing tools)

`tools/blender_add_uv_and_texture.py` (from the prior spec) already knows how to load a glb + replace its diffuse texture + export a new glb keeping all animation data. Reuse it verbatim; the only new inputs are (glb=Dog_textured.glb, diffuse=hy3d_transferred_diffuse.png).

### 4.5 UE cook + render (unchanged from prior spec)

`tools/build_animated_dog.sh` + `examples/render_animated_dog_gpurir.py` are untouched.

**Chosen approach for glb swapping**: the retexture happens **upstream in `generate_hy3d_dog_diffuse.py`**, which overwrites `tmp/animated_dog/Dog_textured.glb` with the new AI-baked variant (after backing up the current one to `.bak.<timestamp>`). The render script keeps loading `Dog_textured.glb` unchanged. This is simpler than plumbing a `--diffuse-override` arg through the render script + UE cook.

Between renders of different breeds, run `generate_hy3d_dog_diffuse.py --breed <name>` to swap, then run `render_animated_dog_gpurir.py` as usual. The `--breed` runtime tint on the render script (Tier-1 solid colors) becomes a no-op when the baked diffuse already encodes the breed pattern — user is expected to pass `--tint-linear-rgb 1 1 1` (or add a new `--breed hy3d_baked` sentinel) to disable Tier-1 tinting.

## 5. Verification gates (executed in this exact order)

Every gate must PASS before the next runs. Each gate has a specific shell command and a specific expected output.

### Gate 0 — Hunyuan environment sanity

```bash
HY3DGEN_MODELS=/data/jzy/code/Hunyuan3D-2.1/pretrained_models \
LD_LIBRARY_PATH=/data/jzy/miniconda3/envs/hunyuan3d/lib/python3.10/site-packages/torch/lib \
CUDA_VISIBLE_DEVICES=0 \
  /data/jzy/miniconda3/envs/hunyuan3d/bin/python -c \
  "from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig; \
   print('IMPORT_OK')"
```

Expected: prints `IMPORT_OK`. Anything else = fix env before proceeding.

### Gate 1 — Hunyuan mesh diff (proves remesh is near-identity)

After a full pipeline run, check that Hunyuan's remeshed mesh has NOT drastically restructured the geometry:

```bash
/data/jzy/miniconda3/envs/hunyuan3d/bin/python tools/verify_hy3d_remesh.py \
    --original tmp/animated_dog/Dog_textured.glb \
    --hy3d-mesh /tmp/hy3d_workdir/white_mesh_remesh.obj
```

Expected: `MESH_DIFF_OK verts_ratio=0.996 faces_ratio=1.000` (verts within ±5%, faces identical).

### Gate 2 — UV transfer coverage

After UV transfer, check the output diffuse doesn't have huge unpainted regions:

```bash
/data/jzy/miniconda3/envs/hunyuan3d/bin/python tools/verify_uv_coverage.py \
    --diffuse tmp/hy3d/border_collie/transferred_diffuse.png
```

Expected: `UV_COVERAGE_OK painted_fraction=0.87` (>0.85 of pixels have non-zero color; the rest are unavoidable UV atlas gaps).

### Gate 3 — Retextured glb integrity

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python tools/verify_dog_textured_glb.py \
    --input tmp/animated_dog/Dog_textured.glb.bak.warm_brown \
    --output tmp/animated_dog/Dog_border_collie_hy3d.glb
```

Expected: `GLB_VERIFY_OK verts=1233 anims=['Idle', 'Walking']`. Vertex count must equal original (proves no mesh mutation). Both anims must be present.

### Gate 4 — UE cook produces valid pak

Reused: `tools/build_animated_dog.sh` prints `BUILD_ANIMATED_DOG_DONE` and BP uasset lands on disk. Same as prior spec.

### Gate 5 — Diag render shows real breed + real animation

```bash
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python tools/diag_animated_dog.py \
  --n-frames 40 --per-frame-warmup 4 --output-dir /tmp/diag_border_collie_hy3d
```

Then compute silhouette-flip metric AND save a 6-frame zoom grid for visual inspection:

Expected quantitative: `flip mean > 1500` (empirically established threshold: <1000 = frozen, >1500 = real walking cycle).

Expected qualitative: 6-frame zoom grid saved to `/tmp/diag_border_collie_hy3d_grid.png`, inspected by the agent (or user), showing:
- Warm-brown eyes / snout, black back, white belly (or whatever the breed looks like — user validates against expectation)
- Visible pose change across frames (leg positions clearly differ)

Only after Gate 5 passes both metrics AND the human/agent visual inspection is the retextured dog "done".

## 6. Failure modes and escape hatches

### F1 — Hunyuan generates completely wrong colors / doesn't understand the reference image

**Trigger**: Gate 5 visual inspection fails — dog is blue, or all-uniform gray, or the pattern doesn't match the reference.

**Cause**: Hunyuan's multiview diffusion is sensitive to reference-image quality (background clutter, unclear pose, extreme perspective). Our probe used a clean rembg'd standing side-view; that's the sweet spot.

**Fix**: On the `--reference-image` path, add a preprocessing step: run rembg + center-crop to a standing side-view (rotate if needed based on detected orientation). On the `--prompt` path, pin the Flux prompt template to `"<breed>, full body, standing side view, plain white background, studio photo"` to bias toward Hunyuan-friendly output.

### F2 — Hunyuan remeshes destructively (verts change >>5%)

**Trigger**: Gate 1 fails — verts_ratio outside 0.95–1.05 or faces_ratio ≠ 1.0.

**Cause**: We haven't tested this on subdivided meshes yet; Hunyuan's `remesh_mesh()` may behave differently on higher-poly inputs.

**Fix**: The current probe run had `use_remesh=True` and got 1233→1228 verts. Should this fail on a variant input, fall back to `use_remesh=False` (also probe-tested — see prior spec's textureGenPipeline reading). Alternately, patch Hunyuan to skip the `mesh_uv_wrap` call if the input already has UVs (documented as a possible option in brainstorming; risky because it may break Hunyuan's bake step).

### F3 — UV transfer leaves visible seams / artifacts

**Trigger**: Gate 5 visual inspection shows sharp color discontinuities at UV-island boundaries.

**Cause**: Two different UV parameterizations are inherently discontinuous at their seams; our transfer samples the "wrong side" of a seam.

**Fix**: Increase dilate iterations in `transfer_uv_texture.py` from 3 → 8; add a Gaussian blur pass over the mask boundaries (5-px sigma). If artifacts persist, add a proper texture inpaint step (`cv2.inpaint` with `INPAINT_TELEA`).

### F4 — Bake has holes because UV islands are too big for Hunyuan's view sampling

**Trigger**: Gate 2 fails — `painted_fraction < 0.85`.

**Cause**: Low-poly mesh means each UV island covers a large surface region; Hunyuan's 6-view bake may miss surface pixels not visible from any of the 6 cameras.

**Fix**: Bump `max_num_view` from 6 → 9 (Hunyuan supports up to 9). If still insufficient, invoke the subdivision escape hatch:
- Blender pre-processes Dog_textured.glb with a 1-level Catmull-Clark subdivision modifier, applied non-destructively (skinning weights inherit automatically via Blender's built-in vertex-group propagation).
- Output: `Dog_textured_subdiv1.glb` with ~5k verts, same anim, same skinning.
- Feed that into Hunyuan; UV atlas becomes denser and bake coverage improves.
- After UV transfer, the retextured mesh keeps the subdivided topology (which is a genuine mesh change) — verify Gate 3 still passes (verts != 1233, but anim data intact).

### F5 — Flux/SDXL text-to-image produces wrong-looking dog

**Trigger**: User prompt path yields a reference image that Hunyuan then paints "wrong".

**Cause**: Flux may hallucinate weird poses or backgrounds.

**Fix**: Iterate on Flux prompt template. Also allow `--flux-seed N` for determinism. Long-term: consider ControlNet-conditioned Flux with a standing-dog pose skeleton as conditioning.

### F6 — Animation broken in final render (silhouette flip <1000)

**Trigger**: Gate 5 quantitative fails.

**Cause**: Somewhere in the pipeline we mutated mesh topology despite our checks (Gate 3 should have caught this).

**Fix**: This is a bug, not a design issue — bisect which step introduced the mesh mutation, add stricter assertions there. The prior spec already documented `SetGamePaused(bPaused=False)` as the root cause of anim-not-ticking; that fix is already in place. Anim-breaking due to mesh mutation is a new failure mode we haven't seen yet.

## 7. What already exists (do NOT re-do)

- Hunyuan3D-2.1 checkout at `/data/jzy/code/Hunyuan3D-2.1/` with local weights (verified working in probe)
- `hunyuan3d` conda env with all deps installed (torch 2.5.1, diffusers 0.31.0, custom_rasterizer built, basicsr/realesrgan/pytorch_lightning installed)
- `tools/probe_hunyuan3d_paint.py` — the probe script, keep as reference for correct env / config
- `tools/blender_add_uv_and_texture.py` — reuse for glb retexture
- `tools/build_animated_dog.sh` — reuse for UE cook (already has stale-uasset cleanup fix)
- `tools/diag_animated_dog.py` — reuse for Gate 5
- `tools/verify_dog_textured_glb.py` — reuse for Gate 3

## 8. Success criteria (what "done" looks like)

- CLI `tools/generate_hy3d_dog_diffuse.py --prompt "border collie" --breed border_collie` runs end-to-end in <5 min (Flux ~30s + Hunyuan ~45s + UV transfer ~10s + Blender ~10s + UE cook ~90s)
- Gates 0-5 all pass; specifically Gate 5 grid shows visually distinct breed AND legs animating
- Adding a new breed = one shell command, no code changes
- Reproducibility documented in a new handoff `docs/superpowers/handoffs/2026-07-05-animated-dog-hunyuan-paint.md`

## 9. Out of scope for THIS spec

- Batch generation of a dog breed library (obvious follow-up)
- Metallic + roughness maps from Hunyuan (available but we only need diffuse for MVP)
- Interactive breed picker / gradio UI (nice-to-have)
- Fine-tuning Hunyuan on dog-specific data (heavy)
- Individual-level variations (age, injury, collar) via Flux prompt — mentioned in brainstorming but treat as a natural side-effect of the prompt interface; no explicit engineering needed beyond what's already designed
