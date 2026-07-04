# Animated Dog GPURIR — Handoff (2026-07-04)

> Spec: [docs/superpowers/specs/2026-07-04-animated-dog-gpurir-design.md](../specs/2026-07-04-animated-dog-gpurir-design.md)
> Plan: [docs/superpowers/plans/2026-07-04-animated-dog-gpurir.md](../plans/2026-07-04-animated-dog-gpurir.md)
> Progress ledger: `.superpowers/sdd/progress.md`
> Final commit: `a737d26d`

## Status: ✅ COMPLETE

All 9 tasks passed all gates. Two MVP videos delivered.

| # | Task | Gate | Result |
|---|---|---|---|
| T1 | Gate 0 SPEAR RPC PlayAnimation probe | `PROBE_OK` | ✅ |
| T2 | Blender headless: UV + procedural warm-brown fur diffuse | `GLB_VERIFY_OK` (1233 verts) | ✅ |
| T3 | `gpurir_trajectory` byte-identical to v77 `get_pos_traj` | 5/5 tests | ✅ |
| T4 | `waypoint_trajectory` + `compute_yaw_from_positions` | 10/10 tests | ✅ |
| T5 | Headless UE editor commandlet import (no GUI) | BP uasset on disk | ✅ |
| T6 | Cook + package via `run_uat.py` | `COOK_VERIFY_OK` via SPEAR RPC | ✅ |
| T7 | `render_animated_dog_gpurir.py` | `COMPILE_OK` + runtime | ✅ |
| T8 | Render both videos + iterate on user feedback | 3/3 issues fixed | ✅ |
| T9 | This handoff doc | — | ✅ |

## The two videos

Both re-runnable from scratch in ~10 minutes (see Reproducibility below).

- **V1 — GPURIR trajectory (seed=42, speed bucket B)**:
  `tmp/render_animated_dog_gpurir/animated_dog_gpurir_seed42_iter4/turntable.mp4`
  - 36 frames at 12 fps, 3 seconds
  - Trajectory is byte-identical to `Spatial/v77_4ch_S2L/gen_rir_multiscene_v77.get_pos_traj(seed=42)` — same seed will produce the same audio & video positions.
- **V2 — Waypoint L-shape**:
  `tmp/render_animated_dog_gpurir/animated_dog_waypoint_Lshape_iter4/turntable.mp4`
  - Same 36-frame length
  - Cubic-spline through `(0.5,0.5) → (2.6,0.5) → (2.6,4.0)` metres

Each video is accompanied by a sibling `trajectory.json` with the full per-frame positions, yaw, room size, material choices, mic position, and speed bucket — cross-modal alignment metadata for future RIR audio integration.

## T8 — the three iterations to reach a clean render

User's first-review feedback (from V1 iter 1):

1. Fur looked like a red carpet (Carpet013 from ambientCG was chosen after PolyHaven had 0 fur textures)
2. Dog floated 45 cm above the floor
3. Legs didn't animate — the dog translated as a rigid mesh

**Iter 2 (commit `42bacace`)** — Fixed fur + float:

- **Fur**: replaced Carpet013 with a procedural warm-brown Voronoi × Noise diffuse baked by [tools/blender_generate_dog_fur.py](../../tools/blender_generate_dog_fur.py). Emission-shader plane render, 1024², 32 samples, ~2 s. CC0 by construction (our pixels).
- **Float**: `positions_m[:,2]` is the *audio-side* source height (mouth ~0.45 m), NOT the actor transform Z. Actor Z now = `args.z_offset_m` default 0 (foot on floor). Audio height stays in `trajectory.json` as metadata.

**Iter 3 (commit `a737d26d`, first half)** — Fixed leg animation freeze:

Root cause: `USkeletalMeshComponent`'s default `VisibilityBasedAnimTickOption` in cooked builds is `OnlyTickPoseWhenRendered`. This checks `LastRenderTime` before ticking the anim clock. When SPEAR captures frames via `USceneCaptureComponent2D`, `LastRenderTime` is not reliably updated before the pre-tick pass, so the anim clock never advances — the dog freezes in the first-frame pose while its transform continues to translate.

Runtime fix attempted first (`smc.SetVisibilityBasedAnimTickOption(...)`) fails with `'UnrealObject' not callable` — that setter is not a UFUNCTION, so the SPEAR RPC layer can't reach it.

**The correct fix is at editor time**, in [tools/import_animated_dog_editor.py](../../tools/import_animated_dog_editor.py):

```python
smc.set_editor_property(
    name="visibility_based_anim_tick_option",
    value=unreal.VisibilityBasedAnimTickOption.ALWAYS_TICK_POSE_AND_REFRESH_BONES,
)
```

This is baked into the BP's SMC default subobject at import time, survives cook + package, and doesn't need any runtime call.

**Iter 4 (commit `a737d26d`, second half)** — Fixed pale-beige color regression:

After iter 3 the dog rendered as a pale near-white silhouette even though the diffuse texture bind was correctly `baseColorTexture -> dog_fur_diffuse`. Diagnosis: the procedural fur diffuse averages to RGB (156, 132, 106) — a light tan — which reads as near-white under the room's ambient + point light.

Fix: force a rich warm-brown tint on top of the texture via `baseColorFactor`:

```python
unreal.MaterialEditingLibrary.set_material_instance_vector_parameter_value(
    instance=fur_mat, parameter_name="baseColorFactor",
    value=unreal.LinearColor(0.35, 0.20, 0.11, 1.0),
)
```

Also pinned `metallicFactor=0` and `roughnessFactor=0.95` for a matte fur read.

Result: dog is now clearly warm brown, on the floor, with a walking cycle whose legs move (verified by side-by-side frame extraction; f=0 shows one pose, f=12 a distinctly different mid-stride pose).

## Reproducibility (~10 min from scratch)

```bash
# 0. Env sanity
/data/jzy/miniconda3/envs/spear-env/bin/python -c "import spear; print(spear.__can_import_spear_ext__)"  # True
which blender  # /data/jzy/.local/bin/blender (v4.2.1 LTS)

# 1. Regenerate procedural fur diffuse (JPG, 1024x1024)
/data/jzy/.local/bin/blender --background --python \
  /data/jzy/code/SPEAR/tools/blender_generate_dog_fur.py -- \
  --output /data/jzy/code/SPEAR/assets/textures/animal_fur/dog_fur_diffuse.jpg \
  --size 1024 --base-color 0.42 0.28 0.16

# 2. Bake UV + apply fur diffuse onto skinned Dog.glb
/data/jzy/.local/bin/blender --background --python \
  /data/jzy/code/SPEAR/tools/blender_add_uv_and_texture.py -- \
  --input  /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack/Dog.glb \
  --output /data/jzy/code/SPEAR/tmp/animated_dog/Dog_textured.glb \
  --diffuse-texture /data/jzy/code/SPEAR/assets/textures/animal_fur/dog_fur_diffuse.jpg \
  --uv-island-margin 0.02

# 3. Out-of-Blender verify
/data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/tools/verify_dog_textured_glb.py \
  --input  /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack/Dog.glb \
  --output /data/jzy/code/SPEAR/tmp/animated_dog/Dog_textured.glb   # -> GLB_VERIFY_OK

# 4. Headless UE editor import + cook (produces BP_dog_animated + cooked pak)
bash /data/jzy/code/SPEAR/tools/build_animated_dog.sh   # -> BUILD_ANIMATED_DOG_DONE

# 5. Cook verify via SPEAR RPC
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/tools/verify_animated_dog_cook.py   # -> COOK_VERIFY_OK

# 6. Unit tests
cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python \
  -m unittest tests.test_trajectory -v   # -> 10/10 OK

# 7. V1 render (GPURIR seed 42, bucket B)
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/examples/render_animated_dog_gpurir.py \
  --trajectory-mode gpurir --trajectory-seed 42 --speed-bucket B \
  --run-name animated_dog_gpurir_seed42_iter4

# 8. V2 render (waypoint L-shape)
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/examples/render_animated_dog_gpurir.py \
  --trajectory-mode waypoints \
  --waypoints "0.5,0.5;2.6,0.5;2.6,4.0" \
  --run-name animated_dog_waypoint_Lshape_iter4
```

## Cross-modal alignment contract (unchanged from spec)

`examples/trajectory.py::gpurir_trajectory` is a byte-identical replica of `Spatial/v77_4ch_S2L/gen_rir_multiscene_v77.get_pos_traj`, verified in `tests/test_trajectory.py::test_gpurir_matches_v77`. The replica:

- Uses the same 10-anchor + cubic-spline algorithm
- Uses the same `SPEED_BUCKET_STEP` table (A=5, B=15, C=30, D=50 cm/step)
- Saves/restores `np.random.get_state()` internally with an explicit `seed` param — does NOT depend on caller's global RNG state
- Sub-samples the full-resolution 200-point trajectory to `n_frames` at video-generation time — critical: audio should ALWAYS use `traj_pts_full=200`, video uses whatever `n_frames` is chosen, they land on the same underlying spline.

## Key files

| Path | Purpose |
|---|---|
| [examples/render_animated_dog_gpurir.py](../../examples/render_animated_dog_gpurir.py) | Stage 3 render script (V1 + V2 modes) |
| [examples/trajectory.py](../../examples/trajectory.py) | GPURIR-byte-identical replica + waypoint interp + yaw |
| [tests/test_trajectory.py](../../tests/test_trajectory.py) | 10 unit tests (all green) |
| [tools/probe_skeletal_playanimation.py](../../tools/probe_skeletal_playanimation.py) | Gate 0 SPEAR RPC probe (reference for correct UFUNCTION kwargs) |
| [tools/blender_generate_dog_fur.py](../../tools/blender_generate_dog_fur.py) | Procedural CC0 fur diffuse generator |
| [tools/blender_add_uv_and_texture.py](../../tools/blender_add_uv_and_texture.py) | Smart UV Project + apply diffuse to skinned glb |
| [tools/verify_dog_textured_glb.py](../../tools/verify_dog_textured_glb.py) | Post-Blender pygltflib check |
| [tools/import_animated_dog_editor.py](../../tools/import_animated_dog_editor.py) | UE Editor commandlet: import glb + create BP with VisibilityBasedAnimTickOption + warm-brown baseColorFactor |
| [tools/build_animated_dog.sh](../../tools/build_animated_dog.sh) | One-shot Stage 2 pipeline (import + cook) |
| [tools/verify_animated_dog_cook.py](../../tools/verify_animated_dog_cook.py) | Runtime load check for BP + SKM + AnimSequence |

## Bug ledger (all closed)

| Date | Bug | Root cause | Fix |
|---|---|---|---|
| 2026-07-04 | T1 probe silently false-passed | SPEAR RPC catches C-side asserts and returns default-null on subsequent calls; `except Exception` never sees them | Use single `PlayAnimation(NewAnimToPlay, bLooping)` (verified against UE 5.5 `SkeletalMeshComponent.h:1126`) |
| 2026-07-04 | Blender ASSERT_VERT_COUNT_MISMATCH (1200→1233) | Smart UV Project splits verts at UV seams | Verifier allows up to 3× vertex increase |
| 2026-07-04 | Cook was 2-second no-op | `run_uat.py BuildCookRun` needs explicit `-build -cook -stage -package -archive -pak` | Added flags |
| 2026-07-04 | `--cook-dirs /Game/...` yielded bogus paths | `run_uat.py` prepends `unreal_project_dir` | Don't pass `--cook-dirs`; `DirectoriesToAlwaysCook` covers it |
| 2026-07-04 | Local variable shadowing (`yaw_deg` scalar vs array) | Camera scope shadowed per-frame array | Renamed to `cam_yaw_deg` |
| 2026-07-04 | Editor commandlet exits nonzero even on success | Nonzero on any warning (Interchange ensure is non-fatal) | `build_animated_dog.sh` verifies via BP uasset presence |
| 2026-07-04 | Dog float 45cm off floor | Used `source_height_m` (audio metadata) as actor Z | Actor Z = `args.z_offset_m` (default 0) |
| 2026-07-04 | Legs frozen | Cooked-build SMC default `OnlyTickPoseWhenRendered` + SceneCaptureComponent2D doesn't update `LastRenderTime` | Bake `AlwaysTickPoseAndRefreshBones` into BP via editor-time `set_editor_property` (runtime setter is not a UFUNCTION) |
| 2026-07-04 | Pale-beige dog after Iter 3 | Procedural diffuse mean is light tan; unlit texture bind was correct but ambient washed it out | Force `baseColorFactor = LinearColor(0.35, 0.20, 0.11, 1.0)` via `MaterialEditingLibrary.set_material_instance_vector_parameter_value` |

## What's next (follow-up specs, OUT OF SCOPE here)

From `Spatial/v77_4ch_S2L/数据集生成探索.md`:

1. **AI motion generation** (Q4=D deferred): AI4Animation / OmniMotionGPT for Sit / Jump / Bark beyond Idle / Walking. Quaternius glb only ships those two anims.
2. **Material Anything AI texture** (Q5=c deferred): compare procedural fur vs an AI-generated PBR material for realism.
3. **Scene 1 / 2 / 3**: multi-instance (animated + static dogs coexisting), add human / appliance / instrument, full QA metadata schema (`instance_id`, `source_anchor`, `answer_json`).
4. **RIR audio integration**: use the same seed + `traj_pts_full=200` to drive `gen_rir_multiscene_v77.get_pos_traj` for the audio side; each rendered video will have a matching 4-channel RIR-convolved audio clip.
5. **Bulk render**: parameterize (room material, dog scale, trajectory family, camera pose) into a batch renderer so this becomes a real dataset generator.
