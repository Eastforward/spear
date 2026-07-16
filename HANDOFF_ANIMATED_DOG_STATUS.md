# Animated-Dog-GPURIR — Session Status Snapshot (2026-07-04)

> Written to survive Claude context compaction. All source is on `main`
> branch of `/data/jzy/code/SPEAR`. Rendered videos are in `tmp/` (not git,
> reproducible via commands below).

## Spec + Plan (both committed)

- Spec: `docs/superpowers/specs/2026-07-04-animated-dog-gpurir-design.md` (commit `8bf526d9`)
- Plan: `docs/superpowers/plans/2026-07-04-animated-dog-gpurir.md` (commit `8b23771e`)
- Progress ledger: `.superpowers/sdd/progress.md` (gitignored, but readable)

## Task progress — 7 of 9 done, T8 half-done, T9 not started

| # | Task | Status | Commit range | Notes |
|---|---|---|---|---|
| T1 | Gate 0 probe SkeletalMesh RPC | ✅ DONE | `8b23771e..a419fa46` | `PROBE_OK` on Mannequin+MF_Walk_Fwd (probe originally had a silent-false-pass bug; fixed in `cc97182c`) |
| T2 | Stage 1 Blender UV + fur | ✅ DONE | `a419fa46..0bcefe2f` | GLB_VERIFY_OK, 1233 verts, Idle+Walking preserved |
| T3 | Gate 1a `gpurir_trajectory` + 5 tests | ✅ DONE | `0bcefe2f..8d1eb4c0` | Includes v77 byte-identical alignment test |
| T4 | Gate 1b `waypoint_trajectory` + yaw + 5 tests | ✅ DONE | `8d1eb4c0..60b3b3c7` | **10/10 unit tests green** |
| T5+T6 | Stage 2 headless UE import + cook + verify | ✅ DONE | `60b3b3c7..039a2afb` | Replaces the originally-planned manual GUI step; `COOK_VERIFY_OK` |
| T7 | Stage 3 render script | ✅ DONE | `039a2afb..d7c4633c` (+ fixes `cc97182c`) | Compiles + runs end-to-end |
| T8 | Render both videos + user review | ⚠️ **PARTIAL** — see below | `cc97182c..42bacace` | 3 issues remain, 2 fixed, 1 open |
| T9 | HANDOFF_ANIMATED_DOG_GPURIR.md | ⏸ NOT STARTED | — | Wait until T8 fully passes |

## T8 status — 3 issues from user's video review

Both videos rendered end-to-end. User watched both and reported:

| # | Issue | Status | Notes |
|---|---|---|---|
| 1 | Fur looked like red carpet (Carpet013) | ✅ FIXED (`42bacace`) | Replaced with procedural warm-brown Voronoi+Noise via `tools/blender_generate_dog_fur.py`. ambientCG has 0 real fur assets (verified via API); PolyHaven fur category empty. Procedural is safest CC0 route. |
| 2 | Dog floated 45cm above the floor | ✅ FIXED (`42bacace`) | Root cause: I used `positions_m[:,2]` (audio-side source height = dog mouth 0.45m) as the ACTOR Z. Actor Z should be foot Z. Now `args.z_offset_m` default 0.0. Audio height stays in `trajectory.json` as metadata. |
| 3 | Legs don't animate — dog translates as rigid mesh | ❌ **OPEN** | Attempted: `SetActorTickEnabled(True)` + `SetComponentTickEnabled(True)` + `SetPlayRate(1.0)` after `PlayAnimation`. `SetVisibilityBasedAnimTickOption` fails with `'UnrealObject' not callable`. Dog silhouette centroid moves 10px across 36 frames but pixel count stable → shape didn't change → anim clock NOT advancing. Root cause not yet found. |

## Latest videos (both from `42bacace`)

- **V1 GPURIR seed=42**: `/data/jzy/code/SPEAR/tmp/render_animated_dog_gpurir/animated_dog_gpurir_seed42/turntable.mp4`
  - Warm-brown speckled fur ✓, on floor ✓, LEGS FROZEN ✗
  - seed=42 in bucket B is a low-motion trajectory — dog barely moves in the frame
- **V2 waypoint L-shape**: `/data/jzy/code/SPEAR/tmp/render_animated_dog_gpurir/animated_dog_waypoint_Lshape/turntable.mp4`
  - Warm-brown speckled fur ✓, on floor ✓, LEGS FROZEN ✗
  - Clear L-shape traversal: `(0.5,0.5) → (2.6,0.5) → (2.6,4.0)` m

## The open question the user paused on

User was about to compact. Before compact they need to decide:

- **A. Accept "moves but legs frozen" as MVP** → I write `HANDOFF_ANIMATED_DOG_GPURIR.md` (T9), spec is done modulo the known open issue documented in handoff. Follow-up spec can fix leg anim later.
- **B. Keep debugging leg anim** — probable 1-2h more work. Directions worth trying:
  - Add a Tick Event in `BP_dog_animated` that calls `smc.TickAnimation(DeltaTime, bNeedsValidRootMotion=False)` explicitly each frame.
  - Investigate whether `SetVisibilityBasedAnimTickOption` needs a different UFUNCTION name; SPEAR RPC caller pattern (see `examples/control_character/run.py` for a working skel-comp usage).
  - Check whether `smc.Play(bLooping=True)` (not `PlayAnimation`) after `SetAnimation` behaves differently — the probe originally called this 3-step chain but with wrong kwargs; the "right kwargs" version was never actually tried.
  - Verify by rendering a longer sequence with a still camera and observing whether pose ever changes.

## Full pipeline is reproducible from scratch

If tmp/ is wiped, this recreates everything (~10-15 min):

```bash
# 0. Confirm env
/data/jzy/miniconda3/envs/spear-env/bin/python -c "import spear; print(spear.__can_import_spear_ext__)"  # True
which blender  # /data/jzy/.local/bin/blender (v4.2.1 LTS)

# 1. Regenerate fur diffuse
/data/jzy/.local/bin/blender --background --python \
  /data/jzy/code/SPEAR/tools/blender_generate_dog_fur.py -- \
  --output /data/jzy/code/SPEAR/assets/textures/animal_fur/dog_fur_diffuse.jpg \
  --size 1024 --base-color 0.42 0.28 0.16

# 2. Bake UV + fur onto skinned dog
/data/jzy/.local/bin/blender --background --python \
  /data/jzy/code/SPEAR/tools/blender_add_uv_and_texture.py -- \
  --input  /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack/Dog.glb \
  --output /data/jzy/code/SPEAR/tmp/animated_dog/Dog_textured.glb \
  --diffuse-texture /data/jzy/code/SPEAR/assets/textures/animal_fur/dog_fur_diffuse.jpg \
  --uv-island-margin 0.02

# 3. Verify glb attributes
/data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/tools/verify_dog_textured_glb.py \
  --input  /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack/Dog.glb \
  --output /data/jzy/code/SPEAR/tmp/animated_dog/Dog_textured.glb  # -> GLB_VERIFY_OK

# 4. Headless UE import + cook
/data/jzy/code/SPEAR/tools/build_animated_dog.sh  # -> BUILD_ANIMATED_DOG_DONE (~2 min if reusing existing SKM, ~15 min fresh)

# 5. Runtime cook verify
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/tools/verify_animated_dog_cook.py  # -> COOK_VERIFY_OK

# 6. Unit tests
cd /data/jzy/code/SPEAR && \
  /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_trajectory -v  # -> 10/10 OK

# 7. Render V1 (GPURIR seed 42)
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/examples/render_animated_dog_gpurir.py \
  --trajectory-mode gpurir --trajectory-seed 42 --speed-bucket B \
  --run-name animated_dog_gpurir_seed42

# 8. Render V2 (waypoint L-shape)
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/examples/render_animated_dog_gpurir.py \
  --trajectory-mode waypoints \
  --waypoints "0.5,0.5;2.6,0.5;2.6,4.0" \
  --run-name animated_dog_waypoint_Lshape
```

## Bigger picture — where this fits and what's next

This spec is one of a series toward user's dataset engine (`Spatial/v77_4ch_S2L/数据集生成探索.md`):

**Done in this session** (foundational):
- SPEAR-based parameterized shoebox room (floor + wall material pools) — was done in prior session before this one
- One animated dog moving through the room along a trajectory that's byte-identical to future GPURIR audio at same seed

**Follow-up specs (not started, out of scope for this session)**:
1. **AI motion generation** (Q4=D deferred): AI4Animation / OmniMotionGPT for Sit/Jump/Bark beyond Idle/Walking.
2. **Material Anything AI texture** (Q5=c deferred): Compare procedural fur vs AI-generated PBR.
3. **Scene 1/2/3 from `数据集生成探索.md`**: multi-instance (animated + static dogs coexisting), add human/appliance/instrument, full QA metadata schema (`instance_id`, `source_anchor`, etc.).
4. **RIR audio integration**: hook `gen_rir_multiscene_v77.get_pos_traj` at same seed → same trajectory as this video pipeline.

## Bug ledger (session-scoped)

| Date | Bug | Root cause | Fix |
|---|---|---|---|
| 2026-07-04 | T1 probe silently false-passed | SPEAR RPC catches C-side asserts and returns default-null on subsequent calls; `except Exception` never sees them | Use single `PlayAnimation(NewAnimToPlay, bLooping)` (verified against UE 5.5 `SkeletalMeshComponent.h:1126`) instead of the 3-call chain with wrong kwargs |
| 2026-07-04 | Blender ASSERT_VERT_COUNT_MISMATCH (1200→1233) | Smart UV Project introduces UV seams; glTF exporter must split verts at seams | Verifier: allow up to 3× vertex increase (fail on shrink or explode) |
| 2026-07-04 | Cook was 2-second no-op | `run_uat.py BuildCookRun` needs explicit `-build -cook -stage -package -archive -pak` flags; without them it just checks paths | Added flags to `build_animated_dog.sh` |
| 2026-07-04 | `--cook-dirs /Game/...` yielded bogus paths | `run_uat.py` prepends `unreal_project_dir` to each cook dir | Don't pass `--cook-dirs`; `+DirectoriesToAlwaysCook` in `DefaultGame.ini` already covers `/Game/MyAssets/Audioset` |
| 2026-07-04 | Local variable shadowing (`yaw_deg` scalar vs array) | I named camera pose scalar `yaw_deg`, shadowing the per-frame array from `_compute_trajectory` | Renamed camera-scope to `cam_yaw_deg` |
| 2026-07-04 | Editor commandlet exits nonzero even on success | Emits nonzero when any warnings/errors logged, including harmless Interchange `Ensure` | `build_animated_dog.sh` ignores editor exit code, verifies success by checking BP uasset on disk |
| 2026-07-04 | Dog float 45cm off floor | Used `source_height_m` (audio metadata) as actor Z | Actor Z = `args.z_offset_m` (default 0), audio height stays in `trajectory.json` |
| 2026-07-04 | Legs frozen | Unknown — SetActorTickEnabled + SetComponentTickEnabled + SetPlayRate didn't help | **OPEN** |

## Key file map for the next agent

| File | Purpose |
|---|---|
| `examples/render_animated_dog_gpurir.py` | Stage 3 main render script |
| `examples/trajectory.py` | GPURIR-byte-identical replica + waypoint interp + yaw |
| `tests/test_trajectory.py` | 10 unit tests (all green) |
| `tools/probe_skeletal_playanimation.py` | Gate 0 SPEAR RPC probe (use as reference for correct UFUNCTION kwargs) |
| `tools/blender_generate_dog_fur.py` | Procedural CC0 fur diffuse generator |
| `tools/blender_add_uv_and_texture.py` | Smart UV Project + apply diffuse to skinned glb |
| `tools/verify_dog_textured_glb.py` | Post-Blender pygltflib check |
| `tools/import_animated_dog_editor.py` | UE Editor commandlet: import glb + create BP |
| `tools/build_animated_dog.sh` | One-shot Stage 2 pipeline (import + cook) |
| `tools/verify_animated_dog_cook.py` | Runtime load check for BP + SKM + AnimSequence |
| `docs/animated_dog_ue_import.md` | Fallback manual-GUI path (not needed since headless works) |
| `docs/superpowers/specs/2026-07-04-animated-dog-gpurir-design.md` | Full design spec |
| `docs/superpowers/plans/2026-07-04-animated-dog-gpurir.md` | 9-task TDD plan |
| `.superpowers/sdd/progress.md` | Ledger — do NOT re-run completed tasks per this file |

## Untracked files worth knowing about (prior sessions' work)

- `examples/render_in_gpurir_room.py` — the STATIC-dog script we intentionally didn't touch (Q12=A). Not committed, live in repo.
- `examples/render_in_apartment.py` — provides `spawn_camera`, `read_frame`, `clean_frames` we import
- Many `tools/diag_*.py` files — earlier debugging tooling from prior sessions
- Many `HANDOFF_*.md` files at repo root — earlier session handoffs (mystery cube, visual render, etc.)
