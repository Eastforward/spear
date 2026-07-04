# Design: Animated Dog with GPURIR-aligned Trajectory in SPEAR Room

**Date**: 2026-07-04
**Scope**: Single MVP feature — one moving animated dog with fur texture in the existing GPURIR shoebox room, along two trajectory modes (GPURIR-style random + user-specified waypoints).
**Non-goals** (out of scope for this spec, will be follow-up specs):
- Scene 1/2/3 from `数据集生成探索.md` (multi-instance mixing, static+animated coexistence, human/appliance/instrument)
- AI motion generation (AI4Animation, OmniMotionGPT) — Q4 chose D = walking now, AI later
- Material Anything AI texture — Q5 chose c, Blender+PolyHaven now, MA compare later
- Full metadata schema for QA generation
- Any changes to `examples/render_in_gpurir_room.py` (static dog script stays untouched — Q12=A)

## User-locked decisions (grill Q4–Q13)

| Q | Decision | Meaning |
|---|---|---|
| Q4 | D | Walking loop + SPEAR move actor now; AI motion generation is a follow-up spec |
| Q5 | c | Blender auto-UV + PolyHaven fur texture now; Material Anything AI texture is a follow-up spec |
| Q7 | B | GPURIR-style random trajectory default; `--waypoints "x,y;x,y;..."` manual override |
| Q8 | D | This spec delivers exactly 2 videos (GPURIR + waypoint). Scene 1/2/3 = follow-up specs |
| Q9 | A | Texture is a MUST-HAVE, not a stretch goal. Dog must visibly have fur color |
| Q10 | B | Blender headless script (`--background --python`). No GUI on Linux server |
| Q11 | B | Artifacts go under `tmp/`, not git; Blender script reproduces them on demand |
| Q12 | A | New standalone `examples/render_animated_dog_gpurir.py`. Don't modify the static-dog script |
| Q13 | C | Video `n_frames=36` samples from a full-resolution GPURIR `traj_pts=200` trajectory (`positions_video[i] = gpurir_traj_200[i * 200 // 36]`). Preserves the "same seed → same shape" contract with GPURIR audio |

## Success criteria (E2E acceptance gates)

Two videos, both must pass user visual review:

**Video 1 — GPURIR trajectory**:
```bash
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  spear-env/python examples/render_animated_dog_gpurir.py \
  --trajectory-mode gpurir --trajectory-seed 42 --speed-bucket B \
  --run-name animated_dog_gpurir_seed42
```
Delivered to `tmp/render_animated_dog_gpurir/animated_dog_gpurir_seed42/turntable.mp4`.

**Video 2 — Waypoint L-shape**:
```bash
spear-env/python examples/render_animated_dog_gpurir.py \
  --trajectory-mode waypoints \
  --waypoints "0.5,0.5;2.6,0.5;2.6,4.0" \
  --run-name animated_dog_waypoint_Lshape
```
Delivered to `tmp/render_animated_dog_gpurir/animated_dog_waypoint_Lshape/turntable.mp4`.

**Human review checklist** (both videos):
- Dog has visible fur texture (not solid grey)
- Dog's 4 legs animate (Walking cycle plays)
- Dog moves through the room (not stuck at start)
- Dog faces along its walking direction (yaw follows tangent)
- Video 2 additionally: L-shape is visible, 90° turn happens at the corner
- Room lighting/camera/materials identical to the static-dog baseline (floor pool + wall pool + 6000 lm ceiling + pitch -30 sun)

## Architecture

Three decoupled stages:

```
┌──────────────────────────────────────────────────────────────────┐
│ Stage 1 — Blender headless (one-shot per asset)                  │
│   tools/blender_add_uv_and_texture.py                            │
│                                                                  │
│   IN:  Spatial/v77_4ch_S2L/assets/mesh_library/                  │
│          quaternius_animalpack/Dog.glb   (no UV, no texture)     │
│        SPEAR/assets/textures/animal_fur/dog_fur_diffuse.jpg      │
│          (PolyHaven CC0, downloaded by                           │
│           tools/download_polyhaven_dog_fur.py)                   │
│   OUT: SPEAR/tmp/animated_dog/Dog_textured.glb                   │
│         (UV-unwrapped, textured, skeleton + 2 anims preserved)   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage 2 — UE Editor import + cook (one-shot per asset)           │
│   docs/animated_dog_ue_import.md  (human-executed GUI steps)     │
│   tools/cook_animated_dog.sh                                     │
│                                                                  │
│   IN:  tmp/animated_dog/Dog_textured.glb                         │
│   OUT: cpp/.../SpearSim/Content/MyAssets/Audioset/               │
│          Meshes/animated_dog/{SKM_dog, SK_dog_Skeleton,          │
│                               anim_dog_Walking, anim_dog_Idle,   │
│                               M_dog}.uasset                      │
│          Blueprints/animated_dog/BP_dog_animated.uasset          │
│        cooked into SpearSim pak → runtime load OK                │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│ Stage 3 — SPEAR runtime render (fast per-render loop)            │
│   examples/trajectory.py  (pure numpy, no SPEAR dep)             │
│   examples/render_animated_dog_gpurir.py                         │
│                                                                  │
│   IN:  Stage 2's BP_dog_animated                                 │
│        CLI: --trajectory-mode {gpurir,waypoints}                 │
│             --trajectory-seed N                                  │
│             --waypoints "x1,y1;x2,y2;..."                        │
│             --speed-bucket {A,B,C,D}                             │
│             --n-frames 36  --framerate 12                        │
│             (all room/light/camera params reused from baseline)  │
│   OUT: tmp/render_animated_dog_gpurir/{run_name}/                │
│          turntable.mp4                                           │
│          trajectory.json  (per-frame world pos + yaw + metadata) │
│          frame_0000.png ... frame_0035.png                       │
└──────────────────────────────────────────────────────────────────┘
```

### Rationale for the three-stage split

1. **Stage 1/2 are offline one-shots**: run once per asset. When iterating on Stage 3 (rendering, trajectory), we don't re-invoke Blender or UE Editor.
2. **Product boundaries are clean**: Stage 1 outputs glb; Stage 2 outputs uasset+pak; Stage 3 outputs mp4+json. Any layer's bug can be caught by looking at that layer's product.
3. **Static dog pipeline is untouched**: Q12=A. `render_in_gpurir_room.py` is a stable, user-approved file with `floor_material_pool` + `wall_material_pool` recently landed. Zero risk of regression there.

## Components

### Component A — `tools/blender_add_uv_and_texture.py`

**Purpose**: add UV coordinates and apply a diffuse texture to a skinned mesh, without touching bones or animations.

**Interface**:
```bash
/data/jzy/.local/bin/blender --background --python tools/blender_add_uv_and_texture.py -- \
  --input  /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack/Dog.glb \
  --output /data/jzy/code/SPEAR/tmp/animated_dog/Dog_textured.glb \
  --diffuse-texture /data/jzy/code/SPEAR/assets/textures/animal_fur/dog_fur_diffuse.jpg \
  --uv-island-margin 0.02
```

**Internal flow** (bpy API):
1. `bpy.ops.wm.read_factory_settings(use_empty=True)` — clean scene
2. `bpy.ops.import_scene.gltf(filepath=input)`
3. Locate mesh object (Quaternius names it "Cylinder"), enter edit mode, select-all
4. `bpy.ops.uv.smart_project(island_margin=0.02, angle_limit=66)`
5. Create material with Principled BSDF, ImageTexture node loading `--diffuse-texture`
6. Assign the new material to the mesh (replacing the original grey `Dog` material)
7. `bpy.ops.export_scene.gltf(filepath=output, export_animations=True, export_skins=True, export_texcoords=True)`

**Script-internal assertions** (script exits 1 on any failure):
- Output mesh vertex count == input mesh vertex count (no unintended remesh)
- Output glTF primitive has `TEXCOORD_0` attribute
- Output glTF primitive has `JOINTS_0` attribute
- Output glTF primitive has `WEIGHTS_0` attribute
- Output has exactly 2 animations named `Idle` and `Walking`

**Dependency**: `tools/download_polyhaven_dog_fur.py` must run first to produce `assets/textures/animal_fur/dog_fur_diffuse.jpg`. That helper is trivial (`urllib` with a real User-Agent header; PolyHaven requires no login). If the diffuse file is missing, this script exits 1 with the exact download URL printed.

### Component B — `docs/animated_dog_ue_import.md` + `tools/cook_animated_dog.sh`

**Purpose**: turn `Dog_textured.glb` into SpearSim-loadable uassets and cook them into the pak.

**Documented human GUI steps** (run once per asset):
1. Open UE 5.5 Editor, open the SpearSim project
2. Content Browser → `/Game/MyAssets/Audioset/Meshes/`, create folder `animated_dog`
3. Drag `SPEAR/tmp/animated_dog/Dog_textured.glb` into that folder
4. Import dialog — set exactly:
   - Skeletal Mesh: Yes
   - Skeleton: create new (name `SK_dog_Skeleton`)
   - Physics Asset: create if not exist
   - Import Materials: Yes
   - Import Textures: Yes
   - Import Animations: Yes
   - Animation Length: Exported Time
5. Verify 5 uassets created: `SKM_dog`, `SK_dog_Skeleton`, `anim_dog_Walking`, `anim_dog_Idle`, `M_dog`
6. In `/Game/MyAssets/Audioset/Blueprints/`, create folder `animated_dog`
7. Right-click `SKM_dog` → Asset Actions → Create Blueprint based on `AActor`. Save as `BP_dog_animated` in `Blueprints/animated_dog/`
8. In the Blueprint, add a `USkeletalMeshComponent`:
   - Skeletal Mesh: `SKM_dog`
   - Animation Mode: `Use Animation Asset`
   - Anim to Play: `anim_dog_Walking`
9. Compile + Save all → close Editor

**Cook script `tools/cook_animated_dog.sh`**:
```bash
#!/bin/bash
set -euo pipefail
/data/jzy/miniconda3/envs/spear-env/bin/python /data/jzy/code/SPEAR/tools/run_uat.py \
  --unreal-engine-dir /data/UE_5.5 \
  --cook-dirs /Game/MyAssets/Audioset/Meshes/animated_dog \
              /Game/MyAssets/Audioset/Blueprints/animated_dog \
  --skip-cook-default-maps
```

`--skip-cook-default-maps` avoids re-cooking the apartment map (10+ minutes). UE's cook dependency tracker pulls in `SK_dog_Skeleton`, `anim_dog_Walking`, `anim_dog_Idle`, `M_dog`, and the fur texture automatically.

**Post-cook verification** — `tools/verify_animated_dog_cook.py`:
```python
inst = configure_gpurir_instance(rpc_port=39002)
g = inst.get_game()
with inst.begin_frame():
    bp = g.unreal_service.load_class(
        uclass="AActor",
        name="/Game/MyAssets/Audioset/Blueprints/animated_dog/BP_dog_animated.BP_dog_animated_C",
    )
    anim = g.unreal_service.load_object(
        uclass="UAnimationAsset",
        name="/Game/MyAssets/Audioset/Meshes/animated_dog/anim_dog_Walking",
    )
    assert bp is not None
    assert anim is not None
print("COOK_VERIFY_OK")
```

### Component C — `examples/trajectory.py`

**Purpose**: pure numpy library. Computes world-space positions and yaw for the moving source. No SPEAR dependency → unit-testable.

**Interface**:
```python
def gpurir_trajectory(
    *,
    room_size_m: tuple[float, float, float],
    n_frames: int,
    speed_bucket: str = "B",       # "A"|"B"|"C"|"D"
    source_height_m: float = 0.45, # dog mouth height
    traj_aug: bool = True,         # cubic interp + random pause
    seed: int,                     # required — use RandomState(seed), NOT global
    traj_pts_full: int = 200,      # GPURIR-native resolution (Q13=C)
) -> np.ndarray:
    """Return shape (n_frames, 3), world-frame meters.
    Internally computes a full-resolution trajectory of length traj_pts_full
    (matching gen_rir_multiscene_v77.get_pos_traj), then subsamples to n_frames
    via `positions_full[i * traj_pts_full // n_frames]`. Same seed → same trajectory
    shape as GPURIR audio for that seed. This is the Q13=C contract."""


def waypoint_trajectory(
    *,
    waypoints_m: list[tuple[float, ...]],  # each (x,y) or (x,y,z); min 2
    n_frames: int,
    room_size_m: tuple[float, float, float],
    source_height_m: float = 0.45,  # fills z when waypoints are 2D
    kind: str = "cubic",             # "linear" | "cubic"
    wall_margin_m: float = 0.1,      # clip to [margin, room_dim - margin]
) -> np.ndarray:
    """(n_frames, 3). Waypoints < 2 → raise ValueError. Points outside
    [wall_margin, room_dim - wall_margin] are clipped with a warn log."""


def compute_yaw_from_positions(
    positions_m: np.ndarray,      # (n_frames, 3)
    smoothing_window: int = 3,    # forward/backward avg to de-jitter
) -> np.ndarray:  # (n_frames,) yaw degrees
    """Yaw = atan2(dy, dx) of forward direction. Frame 0's yaw uses positions[0]→[1]."""
```

**Unit tests** — see Testing section.

### Component D — `examples/render_animated_dog_gpurir.py`

**Purpose**: SPEAR integration layer. Spawns the room + camera + lights (reusing helpers copied from `render_in_gpurir_room.py`), spawns `BP_dog_animated`, plays `anim_dog_Walking` on loop, drives per-frame position + yaw from `trajectory.py`.

**CLI**:
```bash
--trajectory-mode {gpurir,waypoints}   # default: gpurir
--trajectory-seed INT                  # default: 0 (only used in gpurir mode)
--speed-bucket {A,B,C,D}               # default: B (only used in gpurir mode)
--waypoints "x1,y1;x2,y2;..."          # only used in waypoints mode
--source-height-m FLOAT                # default: 0.45 (dog mouth)
--z-offset-m FLOAT                     # default: 0.0 (adjust if pivot ≠ foot)
--n-frames INT                         # default: 36
--framerate INT                        # default: 12
# ...all existing --floor-material-seed / --wall-material-seed /
#    --ceiling-light-lumens / --directional-light-pitch-deg etc. copied
#    from render_in_gpurir_room.py so both scripts produce visually
#    matched shots.
```

**Flow inside `render_animated_dog(args)`**:
1. Configure SPEAR instance, launch SpearSim
2. `with instance.begin_frame():`
   - Destroy Entry-map default actors (same defensive sweep as the static script)
   - Spawn room pieces (`floor_400x400` + walls + ceiling + window + glass + frame + sky + dir light + point light) using the same layout/material APIs as the static script
   - Spawn camera
   - `bp = load_class("BP_dog_animated_C")`, `actor = spawn_actor(bp, location=(0,0,0), ...)`
   - `smc = get_component_by_class(actor, "USkeletalMeshComponent")`
   - `anim = load_object("UAnimationAsset", ".../anim_dog_Walking")`
   - `smc.SetAnimationMode(NewAnimationMode="AnimationSingleNode")`
   - `smc.SetAnimation(NewAnimToPlay=anim)`
   - `smc.Play(bLooping=True)`
3. `instance.step(num_frames=warmup_frames)` — VT warmup
4. Compute trajectory (call `trajectory.py`) + yaw
5. Per-frame loop:
   ```python
   for i in range(n_frames):
       with instance.begin_frame():
           actor.K2_SetActorLocationAndRotation(
               NewLocation={"X": pos_cm[i,0], "Y": pos_cm[i,1], "Z": pos_cm[i,2]},
               NewRotation={"Roll": 0, "Pitch": 0, "Yaw": yaw_deg[i]},
               bSweep=False, bTeleport=True,
           )
       with instance.end_frame():
           pass
       instance.step(num_frames=per_frame_warmup_frames)
       with instance.begin_frame(): pass
       with instance.end_frame():
           cv2.imwrite(frame_path(i), read_frame(comp))
   ```
6. ffmpeg-mux frames → `turntable.mp4`
7. Write `trajectory.json`:
   ```json
   {
     "trajectory_mode": "gpurir",
     "trajectory_seed": 42,
     "speed_bucket": "B",
     "source_height_m": 0.45,
     "room_size_m": [5.2, 4.4, 2.8],
     "n_frames": 36,
     "traj_pts_full": 200,
     "positions_m": [[x0,y0,z0], ...],
     "yaw_deg": [y0, y1, ...],
     "mic_pos_m": [2.6, 2.2, 1.2]
   }
   ```

## Data flow — GPURIR alignment contract (Q13=C)

The purpose of Q13=C is that video and future GPURIR audio, given the same seed, describe **the same trajectory shape** — the video is just a temporal subsample.

```python
# Audio side (future gpuRIR)
from Spatial.v77_4ch_S2L.data_gen.gen_rir_multiscene_v77 import get_pos_traj
np.random.seed(42)  # global seed (v77 convention)
pos_audio, az, el = get_pos_traj(
    room_sz=[5.2, 4.4, 2.8], traj_pts=200, large_angle=360,
    traj_aug=True, speed_bucket='B', el_range=None, source_height=0.45,
)  # shape (200, 3)

# Video side (this spec)
from examples.trajectory import gpurir_trajectory
pos_video = gpurir_trajectory(
    room_size_m=(5.2, 4.4, 2.8), n_frames=36, speed_bucket='B',
    source_height_m=0.45, traj_aug=True, seed=42, traj_pts_full=200,
)  # shape (36, 3)

# CONTRACT: pos_video[i] == pos_audio[i * 200 // 36]
# — enforced by unit test test_gpurir_downsample_matches_c_choice
```

**Implementation constraint**: `gpurir_trajectory` is a line-by-line replica of v77's `get_pos_traj`, using `np.random.RandomState(seed)` (not global) so it doesn't interfere with the caller's random state. The random consumption order is identical to v77.

## Failure modes and mitigations

**F1 — Blender export loses skin**: `bpy.ops.export_scene.gltf(export_skins=True, export_animations=True)` explicit. Script's post-export assertion catches, exits 1.

**F2 — PolyHaven fur texture missing**: `blender_add_uv_and_texture.py` checks the file first; exits 1 with the download URL printed. `tools/download_polyhaven_dog_fur.py` handles the actual download (User-Agent header, no login).

**F3 — UE Editor imports partially (missing anim, wrong skeleton)**: md doc pins every dialog option. Stage 2 verification script asserts `anim_dog_Walking` loads.

**F4 — Cook succeeds but pak missing asset**: `--cook-dirs` passes both `Meshes/animated_dog` and `Blueprints/animated_dog` so UE's dependency tracker pulls in Skeleton + Anim + Material + Texture. If verification fails, fall back to full cook (slow but reliable).

**F5 — `PlayAnimation` not RPC-callable** (Gate 0 blocker): resolved BEFORE spec starts implementation via `tools/probe_skeletal_playanimation.py`. If it fails, define a `BlueprintCallable` UFUNCTION wrapper inside `BP_dog_animated` (e.g. `SetAndPlayWalking()`) and call that from Python instead.

**F6 — Dog spawns in T-pose (no anim plays)**: `spawn_animated_dog()` helper enforces the sequence `SetAnimationMode(AnimationSingleNode)` → `SetAnimation(...)` → `Play(bLooping=True)`. Verified by looking at the first rendered frame — 4 legs must be off-ground.

**F7 — Video trajectory ≠ audio trajectory** (music-picture desync): the hard test `test_gpurir_matches_v77` in the unit suite blocks merge. Random consumption order in `gpurir_trajectory` is byte-identical to v77.

**F8 — Waypoint edge cases**: `<2` points raises `ValueError`. Points outside `[wall_margin, room_dim - wall_margin]` are clipped with a warn log.

**F9 — Dog foot penetrates floor / floats**: after Gate 5, user visually inspects `frame_0000.png`. If pivot ≠ foot, tune `--z-offset-m`. Default 0.0 (assumes pivot is at foot, standard glTF convention).

## Testing strategy

### Unit tests — `tests/test_trajectory.py` (TDD-first)

| # | Test | Assertion |
|---|---|---|
| 1 | `test_gpurir_returns_correct_shape` | `output.shape == (n_frames, 3)` |
| 2 | `test_gpurir_seed_reproducible` | Two calls with same seed produce identical output |
| 3 | `test_gpurir_seed_different` | Different seeds produce different outputs |
| 4 | `test_gpurir_matches_v77` ★ | Same seed + same params vs. v77's `get_pos_traj` → `np.allclose(..., atol=1e-6)` |
| 5 | `test_gpurir_downsample_matches_c_choice` | `traj(n_frames=36)[i] == traj_full_200[i * 200 // 36]` (Q13=C contract) |
| 6 | `test_waypoint_endpoints` | `waypoints=[(0,0),(5,4)]` → `output[0]≈(0,0)`, `output[-1]≈(5,4)` |
| 7 | `test_waypoint_min_points` | `waypoints=[(0,0)]` → `raises ValueError` |
| 8 | `test_waypoint_clip_to_room` | Out-of-bounds waypoint gets clipped, warn logged |
| 9 | `test_yaw_straight_line` | `(0,0)→(5,0)` → `yaw≈0` for all frames |
| 10 | `test_yaw_curve` | Circular arc → monotonic yaw |

**Run**: `cd SPEAR && spear-env/python -m unittest tests.test_trajectory -v`. All 10 must be green before any Stage 2/3 work.

### Integration tests

- **Stage 1**: 5 assertions embedded in `blender_add_uv_and_texture.py`. Exit 1 on failure.
- **Stage 2**: `tools/verify_animated_dog_cook.py` — loads BP + Anim via SPEAR RPC, prints `COOK_VERIFY_OK`.
- **Stage 3**: `tools/probe_skeletal_playanimation.py` — smoke test that `PlayAnimation` is RPC-callable. If it fails, F5 fallback.

### E2E acceptance (user-reviewed)

Video 1 (GPURIR) + Video 2 (waypoint L-shape) — see "Success criteria" above.

### Gate order (each blocks the next)

```
Gate 0: probe_skeletal_playanimation.py passes  ← else trigger F5 fallback
Gate 1: 10/10 unit tests green
Gate 2: Blender script's 5 assertions all pass
Gate 3: UE Editor human import + cook_animated_dog.sh exit 0
Gate 4: verify_animated_dog_cook.py prints COOK_VERIFY_OK
Gate 5: Video 1 (GPURIR) — user says PASS
Gate 6: Video 2 (waypoint) — user says PASS
Gate 7: Handoff doc HANDOFF_ANIMATED_DOG_GPURIR.md archived (mirrors HANDOFF_GPURIR_ROOM.md style)
```

## Directory layout after this spec ships

```
SPEAR/
├── assets/
│   └── textures/
│       └── animal_fur/
│           └── dog_fur_diffuse.jpg      (PolyHaven CC0, downloaded)
├── docs/
│   ├── animated_dog_ue_import.md        (Stage 2 human GUI steps)
│   └── superpowers/specs/
│       └── 2026-07-04-animated-dog-gpurir-design.md   (this file)
├── examples/
│   ├── render_in_gpurir_room.py         (untouched — Q12=A)
│   ├── render_animated_dog_gpurir.py    (new)
│   └── trajectory.py                    (new)
├── tests/
│   └── test_trajectory.py               (new)
├── tmp/
│   └── animated_dog/
│       ├── Dog_textured.glb             (Blender product, not in git)
│       └── debug/                       (only if F1 assertion trips)
├── tools/
│   ├── blender_add_uv_and_texture.py    (new)
│   ├── cook_animated_dog.sh             (new)
│   ├── download_polyhaven_dog_fur.py    (new)
│   ├── probe_skeletal_playanimation.py  (new)
│   └── verify_animated_dog_cook.py      (new)
└── HANDOFF_ANIMATED_DOG_GPURIR.md       (Gate 7 handoff, mirrors existing HANDOFF_* style)
```

## Notes for the implementation plan (writing-plans)

- **Task 0 must be `probe_skeletal_playanimation.py`**. If Gate 0 fails, plan branches to F5's BP-wrapper approach; that changes Stage 2's md doc (add "define BP function `SetAndPlayWalking`") and changes Stage 3's PlayAnimation call. The plan must accommodate either branch.
- **Task order is Gate order 0→7**. Each task hard-blocks the next.
- **Follow existing SPEAR patterns**: `configure_gpurir_instance`, `spawn_camera`, `read_frame`, `write_video_from_frames` all exist in `render_in_apartment.py` / `render_in_gpurir_room.py`. Import, don't rewrite.
- **Room spawn helpers**: Q12=A means we can't refactor them out of `render_in_gpurir_room.py`. Copy the small chunks (~200 lines) verbatim into `render_animated_dog_gpurir.py`. Marked with a `# copied verbatim from render_in_gpurir_room.py::render_gpurir_room` comment so future refactoring can find them.
- **Session env**: `spear-env` (`/data/jzy/miniconda3/envs/spear-env/bin/python`) for everything Python-side. Blender uses its own `/data/jzy/.local/bin/blender` (bundled python).
