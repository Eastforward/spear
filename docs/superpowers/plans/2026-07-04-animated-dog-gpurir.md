# Animated Dog with GPURIR-aligned Trajectory — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render a walking Quaternius Dog (with fur texture, along a GPURIR-style or user-waypoint trajectory) inside the existing SPEAR GPURIR shoebox room, delivering 2 verification videos.

**Architecture:** Three-stage pipeline. Stage 1 (Blender headless) adds UV + fur diffuse to the skinned glb. Stage 2 (UE Editor human import + `run_uat.py` cook) turns it into runtime-loadable uassets. Stage 3 (new `render_animated_dog_gpurir.py` + new `trajectory.py`) drives per-frame SkeletalMesh actor position/yaw via SPEAR RPC while `PlayAnimation("Walking", looping=True)` handles leg motion.

**Tech Stack:** Python 3.11 in `spear-env`; SPEAR RPC (`spear` package + `spear_ext`); Blender 4.2.1 LTS bpy (headless); UE 5.5 Editor (one manual GUI session); `run_uat.py` cook; numpy + scipy.interpolate; unittest; opencv-python; ffmpeg CLI; pygltflib (already `pip install`ed earlier).

## Global Constraints

- Python interpreter for all SPEAR/plan code: **`/data/jzy/miniconda3/envs/spear-env/bin/python`** (has compiled `spear_ext`; `thu` env will silently fail RPC connect).
- Blender: **`/data/jzy/.local/bin/blender`** (v4.2.1 LTS, its own bundled Python for bpy).
- SPEAR runtime shell prefix: `DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`. Xvfb `:99` is expected to be already running (`pgrep -a Xvfb` confirms `1607463 Xvfb :99 -screen 0 1280x720x24 +extension RANDR -ac`).
- SpearSim executable: `/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh`.
- SPEAR project root: `/data/jzy/code/SPEAR`. All relative paths in this plan resolve from there unless prefixed with `/`.
- Do NOT touch `examples/render_in_gpurir_room.py` (Q12=A: static-dog pipeline is frozen). Task 6 copies helpers verbatim rather than importing across scripts.
- Verbatim-copy blocks must carry the comment `# copied verbatim from render_in_gpurir_room.py::render_gpurir_room` so future refactoring can find them.
- Trajectory contract (Q13=C): `gpurir_trajectory(seed=k, n_frames=N)[i] == full_gpurir_traj(seed=k, traj_pts=200)[i * 200 // N]`. Enforced by unit test in Task 3.
- v77 GPURIR reference module: `/data/jzy/code/Spatial/v77_4ch_S2L/data_gen/gen_rir_multiscene_v77.py::get_pos_traj`. Line-by-line random consumption order must match.
- Every task ends with a git commit (per `Spatial/CLAUDE.md` §12). Commit message format: `<phase>: <what>` first line; trailing `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- Session env is git-tracked at `/data/jzy/code/SPEAR` (`git rev-parse --is-inside-work-tree` returns `true`). Working branch: whatever `git branch --show-current` is at the start (do not switch branches without user OK).
- `unittest` runs must be `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.<module> -v`. Running from other cwd fails with `ModuleNotFoundError: No module named 'tests'`.

## File structure

```
SPEAR/
├── assets/textures/animal_fur/dog_fur_diffuse.jpg      NEW  (T2 downloads; not in git; ~1-2 MB)
├── docs/
│   ├── animated_dog_ue_import.md                       NEW  (T5 writes; human GUI steps)
│   └── superpowers/plans/2026-07-04-animated-dog-gpurir.md  (this file)
├── examples/
│   ├── trajectory.py                                   NEW  (T3, T4 build)
│   └── render_animated_dog_gpurir.py                   NEW  (T7 builds)
├── tests/
│   └── test_trajectory.py                              NEW  (T3, T4 grow it)
├── tmp/animated_dog/Dog_textured.glb                   NEW  (T2 produces; not in git)
├── tools/
│   ├── download_polyhaven_dog_fur.py                   NEW  (T2 writes + runs)
│   ├── blender_add_uv_and_texture.py                   NEW  (T2 writes + runs)
│   ├── probe_skeletal_playanimation.py                 NEW  (T1 writes + runs)  ← Gate 0
│   ├── cook_animated_dog.sh                            NEW  (T5 writes)
│   └── verify_animated_dog_cook.py                     NEW  (T6 writes + runs)
└── HANDOFF_ANIMATED_DOG_GPURIR.md                      NEW  (T9 writes)
```

## Task dependency graph

```
T1 (Gate 0)  probe skeletal RPC ────────────┐
                                             │
T2 (Stage 1) blender uv+texture ────────────┤
                                             │
T3 (Gate 1a) trajectory.py::gpurir + tests ─┤─▶ T5 (Stage 2 doc + cook) ─▶ T6 (Gate 4 verify)
T4 (Gate 1b) trajectory.py::waypoint + tests┤                                    │
                                             │                                    ▼
                                             └─▶ T7 (Stage 3 render script) ─▶ T8 (Gates 5, 6)
                                                                                 │
                                                                                 ▼
                                                                            T9 (Gate 7 handoff)
```

T1 → T2/T3/T4 in parallel is fine (they're independent). T5 blocks on T2. T6 blocks on T5. T7 blocks on T1+T3+T4+T6. T8 blocks on T7. T9 blocks on T8.

---

### Task 1: Probe SkeletalMesh + PlayAnimation via SPEAR RPC (Gate 0)

**Files:**
- Create: `tools/probe_skeletal_playanimation.py`
- Test: N/A (this task IS the test — Gate 0)

**Interfaces:**
- Consumes: existing `SKM_Manny_Simple` skeletal mesh (already cooked, at `/Game/Characters/Mannequins/Meshes/SKM_Manny_Simple.SKM_Manny_Simple`) and existing walk anim (`/Game/Characters/Mannequins/Animations/Quinn/MF_Walk_Fwd.MF_Walk_Fwd`).
- Produces: exit-0 script + printed `PROBE_OK` when the RPC chain `SetAnimationMode` → `SetAnimation` → `Play(bLooping=True)` succeeds. If exit-nonzero, T7 must instead use a Blueprint-side wrapper function (F5 fallback in spec).

- [ ] **Step 1: Write the probe script**

Create `tools/probe_skeletal_playanimation.py`:

```python
"""Probe: does SPEAR RPC actually let us drive USkeletalMeshComponent
PlayAnimation? Spec 2026-07-04-animated-dog-gpurir §F5 / Gate 0.

Uses SKM_Manny_Simple + MF_Walk_Fwd (both already cooked in SpearSim) so we
don't need our own animated dog asset yet. Exit 0 + print 'PROBE_OK' => proceed
with Task 3/7 as designed. Exit 1 + print 'PROBE_FAILED_<reason>' => trigger F5
fallback (define a BlueprintCallable helper inside BP_dog_animated and call
that instead of raw PlayAnimation).
"""

import os
import sys

sys.path.insert(0, "/data/jzy/code/SPEAR/examples")
from render_in_gpurir_room import configure_gpurir_instance  # noqa: E402

MANNEQUIN_MESH = "/Game/Characters/Mannequins/Meshes/SKM_Manny_Simple.SKM_Manny_Simple"
WALK_ANIM = "/Game/Characters/Mannequins/Animations/Quinn/MF_Walk_Fwd.MF_Walk_Fwd"


def main():
    instance = configure_gpurir_instance(rpc_port=39002)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            # Kill spawn cube etc. so the probe is clean
            for cls in ("APlayerStart", "ADefaultPawn", "ASpectatorPawn"):
                try:
                    for a in game.unreal_service.find_actors_by_class(uclass=cls):
                        game.unreal_service.destroy_actor(actor=a)
                except Exception:
                    pass

            # Spawn a bare AActor and add a USkeletalMeshComponent by hand.
            # Simpler than fishing a BP; proves the raw component-level RPC path.
            actor = game.unreal_service.spawn_actor(
                uclass="AActor",
                location={"X": 100.0, "Y": 100.0, "Z": 50.0},
                spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
            )
            # AActor doesn't auto-create a SkeletalMeshComponent — many SPEAR
            # workflows spawn a BP that owns one. We instead use the character BP
            # path (proven-working in examples/control_character/run.py) so the
            # probe measures the same call sequence T7 will use later.
            bp_uclass = game.unreal_service.load_class(
                uclass="AActor",
                name="/Game/ThirdPerson/Blueprints/BP_ThirdPersonCharacter.BP_ThirdPersonCharacter_C",
            )
            game.unreal_service.destroy_actor(actor=actor)  # drop the bare actor
            actor = game.unreal_service.spawn_actor(
                uclass=bp_uclass,
                location={"X": 100.0, "Y": 100.0, "Z": 100.0},
                spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
            )
            smc = game.unreal_service.get_component_by_class(
                actor=actor, uclass="USkeletalMeshComponent"
            )
            # Load + assign the mannequin mesh so PlayAnimation has a skeleton to drive
            mesh = game.unreal_service.load_object(uclass="USkeletalMesh", name=MANNEQUIN_MESH)
            smc.SetSkeletalMeshAsset(NewMesh=mesh)

            # THE PROBE — three-call sequence T7 will replicate
            try:
                smc.SetAnimationMode(NewAnimationMode="AnimationSingleNode")
            except Exception as e:
                print(f"PROBE_FAILED_SetAnimationMode: {e}", flush=True)
                sys.exit(1)

            anim = game.unreal_service.load_object(uclass="UAnimationAsset", name=WALK_ANIM)
            if anim is None:
                print("PROBE_FAILED_load_anim_asset returned None", flush=True)
                sys.exit(1)

            try:
                smc.SetAnimation(NewAnimToPlay=anim)
            except Exception as e:
                print(f"PROBE_FAILED_SetAnimation: {e}", flush=True)
                sys.exit(1)

            try:
                smc.Play(bLooping=True)
            except Exception as e:
                print(f"PROBE_FAILED_Play: {e}", flush=True)
                sys.exit(1)

            print("PROBE_OK", flush=True)
        with instance.end_frame():
            pass
    finally:
        instance.close(force=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the probe**

```bash
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/tools/probe_skeletal_playanimation.py 2>&1 | tail -30
```

Expected: last few lines include `PROBE_OK`, exit code 0. If `PROBE_FAILED_*`, STOP and consult the F5 fallback in the spec — later tasks will need a Blueprint-side wrapper.

- [ ] **Step 3: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/probe_skeletal_playanimation.py
git commit -m "T1 (Gate 0): probe SkeletalMesh PlayAnimation via SPEAR RPC

Uses SKM_Manny_Simple + MF_Walk_Fwd (already cooked). If PROBE_OK, the raw
component-level SetAnimationMode/SetAnimation/Play chain works and T7 can
call PlayAnimation directly. If PROBE_FAILED_*, T5's UE import doc must
add a BlueprintCallable wrapper and T7 must call the wrapper instead
(spec F5 fallback).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Blender UV + fur texture (Stage 1)

**Files:**
- Create: `tools/download_polyhaven_dog_fur.py`
- Create: `tools/blender_add_uv_and_texture.py`
- Create (product): `assets/textures/animal_fur/dog_fur_diffuse.jpg` (not committed — regenerated by the downloader; add to `.gitignore`)
- Create (product): `tmp/animated_dog/Dog_textured.glb` (not committed — regenerated by Blender; `tmp/` should already be ignored)

**Interfaces:**
- Consumes: input glb at `/data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack/Dog.glb` (confirmed skinned + 2 anims Idle/Walking, no UV, no texture).
- Produces: `tmp/animated_dog/Dog_textured.glb` — skinned, 2 anims preserved, has `TEXCOORD_0`, references `dog_fur_diffuse.jpg`. Verified by 5 script-internal assertions.

- [ ] **Step 1: Write the PolyHaven downloader**

Create `tools/download_polyhaven_dog_fur.py`:

```python
"""Download a single fur diffuse map from Poly Haven for the animated dog.

Poly Haven is CC0 (see https://polyhaven.com/license), no login needed. We use
the 2K JPG texture — small (~1-2 MB), fine for a low-poly dog.

Chosen asset: patchy_short_fur (any warm/brown fur works for a low-poly dog).
Direct URL pattern:
  https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/<asset>/<asset>_diff_2k.jpg
Poly Haven blocks the default Python urllib User-Agent (403); we pass a browser UA.
"""

import os
import sys
import urllib.request

ASSET_SLUG = "patchy_short_fur"
DIFFUSE_URL = (
    f"https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/"
    f"{ASSET_SLUG}/{ASSET_SLUG}_diff_2k.jpg"
)
OUTPUT_PATH = "/data/jzy/code/SPEAR/assets/textures/animal_fur/dog_fur_diffuse.jpg"


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 100_000:
        print(f"ALREADY_EXISTS {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)} bytes)", flush=True)
        return 0

    req = urllib.request.Request(
        DIFFUSE_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
            )
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except Exception as e:
        print(f"DOWNLOAD_FAILED {DIFFUSE_URL}: {e}", flush=True)
        return 1

    with open(OUTPUT_PATH, "wb") as f:
        f.write(data)
    print(f"DOWNLOADED {OUTPUT_PATH} ({len(data)} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the downloader**

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/tools/download_polyhaven_dog_fur.py
```

Expected stdout: `DOWNLOADED /data/jzy/code/SPEAR/assets/textures/animal_fur/dog_fur_diffuse.jpg (N bytes)` with N > 100_000. If it prints `DOWNLOAD_FAILED`, try the URL from a browser (Poly Haven occasionally reshuffles paths); if the asset moved, update `ASSET_SLUG` to another warm-fur asset and re-run.

Confirm with `ls -la /data/jzy/code/SPEAR/assets/textures/animal_fur/dog_fur_diffuse.jpg`.

- [ ] **Step 3: Write the Blender UV+texture script**

Create `tools/blender_add_uv_and_texture.py`:

```python
"""Blender headless: add UVs + apply a diffuse texture to a skinned glb,
without touching bones or animations. See spec 2026-07-04-animated-dog-gpurir
Component A.

Usage:
  /data/jzy/.local/bin/blender --background --python \
    /data/jzy/code/SPEAR/tools/blender_add_uv_and_texture.py -- \
    --input  /path/Dog.glb \
    --output /path/Dog_textured.glb \
    --diffuse-texture /path/dog_fur_diffuse.jpg \
    --uv-island-margin 0.02

Post-export assertions (script exits 1 on any failure):
  - Output mesh vertex count == input mesh vertex count (no unintended remesh)
  - Output glTF primitive has TEXCOORD_0
  - Output glTF primitive has JOINTS_0
  - Output glTF primitive has WEIGHTS_0
  - Output has exactly 2 animations named 'Idle' and 'Walking'
"""

import argparse
import os
import sys

import bpy  # provided by blender --background


def parse_argv():
    # Blender passes user args after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--diffuse-texture", required=True)
    p.add_argument("--uv-island-margin", type=float, default=0.02)
    return p.parse_args(argv)


def count_verts_of_first_mesh(glb_path):
    import pygltflib

    g = pygltflib.GLTF2.load(glb_path)
    # We only care about the primary skinned mesh, primitive 0.
    prim = g.meshes[0].primitives[0]
    pos_acc_idx = prim.attributes.POSITION
    return g.accessors[pos_acc_idx].count


def check_attrs_of_first_prim(glb_path):
    import pygltflib

    g = pygltflib.GLTF2.load(glb_path)
    prim = g.meshes[0].primitives[0]
    anims = [a.name for a in (g.animations or [])]
    return {
        "TEXCOORD_0": prim.attributes.TEXCOORD_0,
        "JOINTS_0": prim.attributes.JOINTS_0,
        "WEIGHTS_0": prim.attributes.WEIGHTS_0,
        "animations": anims,
    }


def main():
    args = parse_argv()
    if not os.path.exists(args.input):
        print(f"BLENDER_INPUT_MISSING {args.input}", flush=True)
        sys.exit(1)
    if not os.path.exists(args.diffuse_texture):
        print(
            f"BLENDER_TEXTURE_MISSING {args.diffuse_texture}\n"
            f"Run tools/download_polyhaven_dog_fur.py first.",
            flush=True,
        )
        sys.exit(1)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Remember input vertex count for the post-export assertion
    input_verts = count_verts_of_first_mesh(args.input)
    print(f"INPUT_VERT_COUNT {input_verts}", flush=True)

    # 1. Clean scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2. Import glb
    bpy.ops.import_scene.gltf(filepath=args.input)

    # 3. Find the mesh object (Quaternius names it 'Cylinder', but be defensive)
    mesh_objs = [o for o in bpy.data.objects if o.type == "MESH"]
    if len(mesh_objs) != 1:
        print(f"BLENDER_EXPECTED_1_MESH_GOT {len(mesh_objs)}: {[o.name for o in mesh_objs]}", flush=True)
        sys.exit(1)
    mesh = mesh_objs[0]

    # 4. Enter edit mode, select all faces, Smart UV Project
    bpy.context.view_layer.objects.active = mesh
    mesh.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        island_margin=args.uv_island_margin,
        angle_limit=1.15,  # ~66 degrees in radians
    )
    bpy.ops.object.mode_set(mode="OBJECT")

    # 5. Build a new material with Principled BSDF + ImageTexture
    mat = bpy.data.materials.new(name="Dog_Fur")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    # Remove default nodes, add fresh
    for n in list(nodes):
        nodes.remove(n)
    output_node = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(args.diffuse_texture)
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output_node.inputs["Surface"])

    # 6. Replace mesh's material(s) with the new one
    mesh.data.materials.clear()
    mesh.data.materials.append(mat)

    # 7. Export glb — preserve skin + animations + UVs
    bpy.ops.export_scene.gltf(
        filepath=args.output,
        export_format="GLB",
        export_animations=True,
        export_skins=True,
        export_texcoords=True,
        export_normals=True,
        # Bundle the image inside the glb so downstream tools have a single file
        export_image_format="AUTO",
    )
    print(f"EXPORTED {args.output}", flush=True)

    # 8. Post-export assertions
    out_verts = count_verts_of_first_mesh(args.output)
    if out_verts != input_verts:
        print(f"ASSERT_VERT_COUNT_MISMATCH input={input_verts} output={out_verts}", flush=True)
        sys.exit(1)

    attrs = check_attrs_of_first_prim(args.output)
    if attrs["TEXCOORD_0"] is None:
        print("ASSERT_MISSING_TEXCOORD_0", flush=True)
        sys.exit(1)
    if attrs["JOINTS_0"] is None:
        print("ASSERT_MISSING_JOINTS_0", flush=True)
        sys.exit(1)
    if attrs["WEIGHTS_0"] is None:
        print("ASSERT_MISSING_WEIGHTS_0", flush=True)
        sys.exit(1)
    if sorted(attrs["animations"]) != ["Idle", "Walking"]:
        print(f"ASSERT_ANIM_MISMATCH got={attrs['animations']}", flush=True)
        sys.exit(1)

    print("BLENDER_OK", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the Blender script**

```bash
/data/jzy/.local/bin/blender --background --python \
  /data/jzy/code/SPEAR/tools/blender_add_uv_and_texture.py -- \
  --input  /data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack/Dog.glb \
  --output /data/jzy/code/SPEAR/tmp/animated_dog/Dog_textured.glb \
  --diffuse-texture /data/jzy/code/SPEAR/assets/textures/animal_fur/dog_fur_diffuse.jpg \
  --uv-island-margin 0.02 2>&1 | tail -30
```

Expected: last line `BLENDER_OK`, exit code 0. `Dog_textured.glb` exists at the given path, size > 100 KB.

If `ASSERT_*` failure, STOP; read the exact assertion line and fix the script (typical culprit: `export_scene.gltf` param names change across Blender versions — cross-check with `bpy.ops.export_scene.gltf.get_rna_type().properties` if needed).

- [ ] **Step 5: Confirm .gitignore excludes the products**

```bash
cd /data/jzy/code/SPEAR
grep -F "assets/textures/animal_fur/" .gitignore || echo "assets/textures/animal_fur/*.jpg" >> .gitignore
grep -F "tmp/animated_dog/" .gitignore || echo "tmp/animated_dog/*.glb" >> .gitignore
git status --short assets/textures/ tmp/animated_dog/ 2>&1 | head
```

Expected: the two glb/jpg products do NOT appear under `git status`.

- [ ] **Step 6: Commit the two tools**

```bash
cd /data/jzy/code/SPEAR
git add tools/download_polyhaven_dog_fur.py \
        tools/blender_add_uv_and_texture.py \
        .gitignore
git commit -m "T2 (Stage 1): Blender headless UV + PolyHaven fur diffuse

Two tools:
- download_polyhaven_dog_fur.py fetches patchy_short_fur diffuse (CC0)
  with a real UA header (default urllib UA is 403'd by PolyHaven).
- blender_add_uv_and_texture.py runs Smart UV Project + Principled BSDF
  with the diffuse texture, exporting a glb that preserves skin +
  Idle/Walking anims. 5 post-export assertions guard the invariants.

Products (dog_fur_diffuse.jpg, Dog_textured.glb) live under tmp/ and
assets/textures/ per Q11=B, added to .gitignore.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: `trajectory.py::gpurir_trajectory` + tests 1–5 (Gate 1a)

**Files:**
- Create: `examples/trajectory.py`
- Create: `tests/test_trajectory.py`

**Interfaces:**
- Consumes: `scipy.interpolate.interp1d`, `numpy`. v77 reference at `/data/jzy/code/Spatial/v77_4ch_S2L/data_gen/gen_rir_multiscene_v77.py::get_pos_traj` for the byte-identical replica.
- Produces (used by T7): `gpurir_trajectory(*, room_size_m, n_frames, speed_bucket="B", source_height_m=0.45, traj_aug=True, seed, traj_pts_full=200) -> np.ndarray` of shape `(n_frames, 3)`, world-frame meters. Constants: `SPEED_BUCKET_STEP = {"A": 5.0, "B": 15.0, "C": 30.0, "D": 50.0}`, `MIC_HEIGHT_M = 1.2`.

- [ ] **Step 1: Write the 5 failing tests first (TDD)**

Create `tests/test_trajectory.py`:

```python
"""Unit tests for examples/trajectory.py.

TDD: this file is written BEFORE examples/trajectory.py exists. Steps 1-2
of Task 3 confirm the tests fail with ModuleNotFoundError / AttributeError,
proving the tests actually run (they'd otherwise silently no-op).
"""

import os
import sys
import unittest

import numpy as np

# examples/ is a plain script dir, not a package
_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.abspath(os.path.join(_HERE, "..", "examples"))
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)


class GpurirTrajectoryTests(unittest.TestCase):
    def _call(self, **kwargs):
        import trajectory  # imported inside so a missing module fails per-test
        defaults = dict(
            room_size_m=(5.2, 4.4, 2.8),
            n_frames=36,
            speed_bucket="B",
            source_height_m=0.45,
            traj_aug=True,
            seed=42,
            traj_pts_full=200,
        )
        defaults.update(kwargs)
        return trajectory.gpurir_trajectory(**defaults)

    def test_gpurir_returns_correct_shape(self):
        traj = self._call(n_frames=36)
        self.assertEqual(traj.shape, (36, 3))

    def test_gpurir_seed_reproducible(self):
        a = self._call(seed=42)
        b = self._call(seed=42)
        np.testing.assert_array_equal(a, b)

    def test_gpurir_seed_different(self):
        a = self._call(seed=42)
        b = self._call(seed=43)
        self.assertFalse(np.allclose(a, b), "different seeds must give different trajectories")

    def test_gpurir_matches_v77(self):
        """★ KEY CONTRACT: same seed + same params vs v77's get_pos_traj → identical."""
        sys.path.insert(0, "/data/jzy/code/Spatial/v77_4ch_S2L/data_gen")
        from gen_rir_multiscene_v77 import get_pos_traj  # noqa

        # v77 uses GLOBAL np.random.seed — replicate the exact same call.
        np.random.seed(42)
        pos_audio, _, _ = get_pos_traj(
            room_sz=[5.2, 4.4, 2.8],
            traj_pts=200,
            large_angle=360,
            traj_aug=True,
            speed_bucket="B",
            el_range=None,
            source_height=0.45,
        )
        # gpurir_trajectory uses RandomState(seed) internally so it does NOT
        # care about the caller's global state. Full-res (n_frames=200) call:
        pos_video_full = self._call(n_frames=200, seed=42)
        np.testing.assert_allclose(pos_video_full, pos_audio, atol=1e-6)

    def test_gpurir_downsample_matches_c_choice(self):
        """Q13=C: video subsamples full-res grid by (i * traj_pts_full // n_frames)."""
        full = self._call(n_frames=200, seed=42, traj_pts_full=200)
        sub = self._call(n_frames=36, seed=42, traj_pts_full=200)
        expected = np.stack([full[i * 200 // 36] for i in range(36)])
        np.testing.assert_array_equal(sub, expected)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail with ModuleNotFoundError**

```bash
cd /data/jzy/code/SPEAR && \
  /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_trajectory -v 2>&1 | tail -15
```

Expected: 5 errors, all like `ModuleNotFoundError: No module named 'trajectory'`. This proves the tests do run.

- [ ] **Step 3: Write the minimal implementation to pass tests 1, 2, 3, 5**

Create `examples/trajectory.py`:

```python
"""World-space source trajectories for the animated dog.

`gpurir_trajectory` is a byte-identical replica of
gen_rir_multiscene_v77.get_pos_traj so video and audio (given same seed and
params) describe the SAME trajectory shape. See spec Q13=C and Data Flow §
for the sub-sampling contract.

`waypoint_trajectory` (Task 4) is the user-controlled override.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import interp1d

log = logging.getLogger(__name__)

# ---- byte-identical constants from v77 gen_rir_multiscene_v77 -------------
# DO NOT CHANGE THESE VALUES without a matching change in v77 or you break
# the same-seed cross-modal alignment contract (Q13=C, spec F7).
SPEED_BUCKET_STEP = {"A": 5.0, "B": 15.0, "C": 30.0, "D": 50.0}
MIC_HEIGHT_M = 1.2  # matches v77.MIC_HEIGHT
_N_ANCHORS = 10


def _source_distance(room_sz, rng):
    """Byte-identical to v77._source_distance, but takes an explicit rng."""
    d_max = min(room_sz[0], room_sz[1]) / 2.0 - 0.5
    return float(rng.uniform(1.0, max(1.5, d_max)))


def gpurir_trajectory(
    *,
    room_size_m,
    n_frames,
    speed_bucket="B",
    source_height_m=0.45,
    traj_aug=True,
    seed,
    traj_pts_full=200,
    large_angle=360.0,
):
    """Return (n_frames, 3) world-frame meters.

    Internally: build a full-resolution length-`traj_pts_full` trajectory using
    the SAME random consumption order as v77.get_pos_traj, then subsample to
    `n_frames` via positions_full[i * traj_pts_full // n_frames].

    Uses np.random.RandomState(seed) — does NOT touch the global np.random state.
    HOWEVER, to remain byte-identical with the v77 global-seed path, users of
    the alignment contract (test_gpurir_matches_v77) call this with n_frames =
    traj_pts_full = 200 and compare to a v77 call using the same seed.
    """
    room_sz = list(room_size_m)
    rng = np.random.RandomState(int(seed))

    # ---- anchor sampling (10 points) — verbatim shape from v77 ----
    original_az = np.zeros(_N_ANCHORS)
    original_distance = np.zeros(_N_ANCHORS)
    original_az[0] = rng.uniform(0, large_angle)
    original_distance[0] = _source_distance(room_sz, rng)
    step = SPEED_BUCKET_STEP.get(speed_bucket, 15.0)
    for i in range(1, _N_ANCHORS):
        if traj_aug and rng.rand() < 0.15:  # random pause (v77 same threshold)
            original_az[i] = original_az[i - 1]
            original_distance[i] = original_distance[i - 1]
            continue
        potential_az = original_az[i - 1] + rng.uniform(-step, step)
        if potential_az < 0:
            original_az[i] = -potential_az
        elif potential_az > large_angle:
            original_az[i] = 2 * large_angle - potential_az
        else:
            original_az[i] = potential_az
        original_az[i] = np.clip(original_az[i], 0, large_angle)
        original_distance[i] = original_distance[i - 1] + rng.uniform(-0.02, 0.05)

    # ---- interpolate anchors → full-res grid ----
    time_original = np.linspace(0, 1, _N_ANCHORS)
    time_smooth = np.linspace(0, 1, traj_pts_full)
    kind = "cubic" if traj_aug else "linear"
    smooth_azimuth = np.mod(
        interp1d(time_original, original_az, kind=kind)(time_smooth), large_angle
    )
    smooth_distance = interp1d(time_original, original_distance, kind=kind)(time_smooth)

    # ---- polar → cartesian (in room-center coords, add cx/cy) ----
    theta = smooth_azimuth * np.pi / 180.0
    cx, cy = room_sz[0] / 2.0, room_sz[1] / 2.0
    x = smooth_distance * np.cos(theta) + cx
    y = smooth_distance * np.sin(theta) + cy
    z = np.full(traj_pts_full, float(source_height_m))
    positions_full = np.stack([x, y, z], axis=1)

    # ---- Q13=C sub-sample ----
    if n_frames == traj_pts_full:
        return positions_full
    idx = np.array([i * traj_pts_full // n_frames for i in range(n_frames)])
    return positions_full[idx]
```

- [ ] **Step 4: Run tests, expect 4 to pass and 1 (`test_gpurir_matches_v77`) to fail with a random-order mismatch**

```bash
cd /data/jzy/code/SPEAR && \
  /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_trajectory -v 2>&1 | tail -20
```

Expected: `test_gpurir_returns_correct_shape`, `test_gpurir_seed_reproducible`, `test_gpurir_seed_different`, `test_gpurir_downsample_matches_c_choice` PASS. `test_gpurir_matches_v77` FAILS because v77 uses `np.random.uniform`/`np.random.rand` (global) and we use `RandomState(seed)` — same seed, different generator state trajectory.

- [ ] **Step 5: Fix the v77 alignment — use global np.random inside a scoped `RandomState`**

The trick: v77's `get_pos_traj` calls `np.random.rand()` and `np.random.uniform(...)` — the global module-level functions. To be byte-identical, we must feed the global generator with the same seed and consume in the same order. Update `gpurir_trajectory` — replace the `rng = np.random.RandomState(seed)` line and every `rng.<method>` call with a save/restore of the global state:

Edit `examples/trajectory.py` — replace the body of `gpurir_trajectory` starting from `rng = np.random.RandomState(int(seed))` through the end of the anchor-sampling loop with this block (keep everything after "# ---- interpolate anchors →" as-is):

```python
    room_sz = list(room_size_m)

    # Byte-identical alignment with v77.get_pos_traj: v77 uses the GLOBAL
    # np.random state. We save/restore the caller's global state so we don't
    # clobber it, then seed the global state with `seed` and consume in the
    # SAME order as v77's function body.
    _saved_state = np.random.get_state()
    try:
        np.random.seed(int(seed))

        original_az = np.zeros(_N_ANCHORS)
        original_distance = np.zeros(_N_ANCHORS)
        original_az[0] = np.random.uniform(0, large_angle)
        # v77._source_distance also uses np.random.uniform on the global state
        d_max = min(room_sz[0], room_sz[1]) / 2.0 - 0.5
        original_distance[0] = float(np.random.uniform(1.0, max(1.5, d_max)))
        step = SPEED_BUCKET_STEP.get(speed_bucket, 15.0)
        for i in range(1, _N_ANCHORS):
            if traj_aug and np.random.rand() < 0.15:
                original_az[i] = original_az[i - 1]
                original_distance[i] = original_distance[i - 1]
                continue
            potential_az = original_az[i - 1] + np.random.uniform(-step, step)
            if potential_az < 0:
                original_az[i] = -potential_az
            elif potential_az > large_angle:
                original_az[i] = 2 * large_angle - potential_az
            else:
                original_az[i] = potential_az
            original_az[i] = np.clip(original_az[i], 0, large_angle)
            original_distance[i] = original_distance[i - 1] + np.random.uniform(-0.02, 0.05)
    finally:
        np.random.set_state(_saved_state)
```

Also delete the now-unused `_source_distance` helper (or leave it commented — the inline `d_max` in the block above replaces it).

- [ ] **Step 6: Re-run tests, expect 5/5 PASS**

```bash
cd /data/jzy/code/SPEAR && \
  /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_trajectory -v 2>&1 | tail -20
```

Expected: `Ran 5 tests ... OK`. If `test_gpurir_matches_v77` still fails, cross-check the random consumption order against v77 `get_pos_traj` line by line — even one extra `rng` call shifts everything downstream.

- [ ] **Step 7: Commit**

```bash
cd /data/jzy/code/SPEAR
git add examples/trajectory.py tests/test_trajectory.py
git commit -m "T3 (Gate 1a): gpurir_trajectory + 5 unit tests

Byte-identical replica of v77 gen_rir_multiscene_v77.get_pos_traj:
same seed and same params yield allclose(..., atol=1e-6) output. Uses
save/restore of np.random global state so we don't clobber the caller.

Q13=C sub-sample contract: positions_video[i] = positions_full[i * 200 // n_frames],
enforced by test_gpurir_downsample_matches_c_choice.

5/5 tests green.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: `trajectory.py::waypoint_trajectory` + `compute_yaw_from_positions` + tests 6–10 (Gate 1b)

**Files:**
- Modify: `examples/trajectory.py` (append two functions)
- Modify: `tests/test_trajectory.py` (append 5 tests)

**Interfaces:**
- Consumes: same as Task 3 plus `scipy.interpolate.interp1d`.
- Produces (used by T7):
  - `waypoint_trajectory(*, waypoints_m, n_frames, room_size_m, source_height_m=0.45, kind="cubic", wall_margin_m=0.1) -> np.ndarray` shape `(n_frames, 3)`, world-frame meters.
  - `compute_yaw_from_positions(positions_m, smoothing_window=3) -> np.ndarray` shape `(n_frames,)` in degrees.

- [ ] **Step 1: Append the 5 new tests to `tests/test_trajectory.py`**

Add at the bottom of `tests/test_trajectory.py` (before `if __name__ == "__main__":`):

```python
class WaypointTrajectoryTests(unittest.TestCase):
    def test_waypoint_endpoints(self):
        import trajectory
        traj = trajectory.waypoint_trajectory(
            waypoints_m=[(0.5, 0.5), (5.0, 4.0)],
            n_frames=36,
            room_size_m=(5.2, 4.4, 2.8),
            kind="linear",
        )
        self.assertEqual(traj.shape, (36, 3))
        np.testing.assert_allclose(traj[0, :2], [0.5, 0.5], atol=1e-6)
        np.testing.assert_allclose(traj[-1, :2], [5.0, 4.0], atol=1e-6)

    def test_waypoint_min_points(self):
        import trajectory
        with self.assertRaises(ValueError):
            trajectory.waypoint_trajectory(
                waypoints_m=[(1.0, 1.0)],  # only 1 point
                n_frames=36,
                room_size_m=(5.2, 4.4, 2.8),
            )

    def test_waypoint_clip_to_room(self):
        import trajectory
        with self.assertLogs("trajectory", level="WARNING"):
            traj = trajectory.waypoint_trajectory(
                waypoints_m=[(-1.0, -1.0), (999.0, 999.0)],
                n_frames=10,
                room_size_m=(5.2, 4.4, 2.8),
                wall_margin_m=0.1,
                kind="linear",
            )
        # Endpoints should be clipped to [margin, room_dim - margin]
        self.assertGreaterEqual(traj[0, 0], 0.1 - 1e-9)
        self.assertLessEqual(traj[-1, 0], 5.2 - 0.1 + 1e-9)
        self.assertLessEqual(traj[-1, 1], 4.4 - 0.1 + 1e-9)


class YawTests(unittest.TestCase):
    def test_yaw_straight_line(self):
        import trajectory
        positions = np.stack([
            np.linspace(0.0, 5.0, 20),  # x
            np.zeros(20),                # y
            np.full(20, 0.45),           # z
        ], axis=1)
        yaw = trajectory.compute_yaw_from_positions(positions)
        # +x direction is 0 degrees
        np.testing.assert_allclose(yaw, np.zeros(20), atol=1.0)  # 1 deg tolerance

    def test_yaw_curve(self):
        import trajectory
        # Half-circle in xy at radius 1 centered at (0,0), from (1,0) to (-1,0)
        t = np.linspace(0.0, np.pi, 40)
        positions = np.stack([np.cos(t), np.sin(t), np.full(40, 0.45)], axis=1)
        yaw = trajectory.compute_yaw_from_positions(positions)
        # Yaw of forward tangent goes from ~+90 (moving +y) around to ~-90/+270
        # (moving -y). It should be monotonically increasing modulo 360.
        unwrapped = np.unwrap(np.deg2rad(yaw))
        diffs = np.diff(unwrapped)
        self.assertGreater(diffs.mean(), 0.0, "yaw should sweep in one direction")
```

- [ ] **Step 2: Run tests, expect 5 new tests to fail (AttributeError)**

```bash
cd /data/jzy/code/SPEAR && \
  /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_trajectory -v 2>&1 | tail -20
```

Expected: 5 old tests PASS, 5 new fail with `AttributeError: module 'trajectory' has no attribute 'waypoint_trajectory'` / `compute_yaw_from_positions`.

- [ ] **Step 3: Append the two new functions to `examples/trajectory.py`**

Append at the end of `examples/trajectory.py`:

```python
def waypoint_trajectory(
    *,
    waypoints_m,
    n_frames,
    room_size_m,
    source_height_m=0.45,
    kind="cubic",
    wall_margin_m=0.1,
):
    """Interpolate a list of (x,y) or (x,y,z) waypoints (meters) to n_frames points.

    Raises ValueError if fewer than 2 waypoints. Any waypoint outside
    [wall_margin_m, room_dim - wall_margin_m] is clipped and a WARNING is logged.
    """
    if len(waypoints_m) < 2:
        raise ValueError(f"waypoint_trajectory needs >= 2 waypoints, got {len(waypoints_m)}")

    rx, ry, _rz = (float(v) for v in room_size_m)
    lo_x, hi_x = wall_margin_m, rx - wall_margin_m
    lo_y, hi_y = wall_margin_m, ry - wall_margin_m

    clipped = []
    any_clip = False
    for wp in waypoints_m:
        wp = list(wp)
        if len(wp) == 2:
            wp = [wp[0], wp[1], source_height_m]
        elif len(wp) != 3:
            raise ValueError(f"waypoint must be (x,y) or (x,y,z), got {wp!r}")
        new_x = float(np.clip(wp[0], lo_x, hi_x))
        new_y = float(np.clip(wp[1], lo_y, hi_y))
        if new_x != wp[0] or new_y != wp[1]:
            any_clip = True
        clipped.append([new_x, new_y, float(wp[2])])

    if any_clip:
        log.warning(
            "waypoint_trajectory: clipped one or more waypoints to room bounds "
            "[%.2f, %.2f] x [%.2f, %.2f]", lo_x, hi_x, lo_y, hi_y,
        )

    wp_arr = np.array(clipped)  # (K, 3)
    K = wp_arr.shape[0]
    t_orig = np.linspace(0.0, 1.0, K)
    t_smooth = np.linspace(0.0, 1.0, n_frames)
    # cubic needs K>=4; fall back to linear if fewer
    use_kind = kind if (kind != "cubic" or K >= 4) else "linear"
    xs = interp1d(t_orig, wp_arr[:, 0], kind=use_kind)(t_smooth)
    ys = interp1d(t_orig, wp_arr[:, 1], kind=use_kind)(t_smooth)
    zs = interp1d(t_orig, wp_arr[:, 2], kind=use_kind)(t_smooth)
    return np.stack([xs, ys, zs], axis=1)


def compute_yaw_from_positions(positions_m, smoothing_window=3):
    """Yaw (degrees) of the forward tangent. +x = 0 deg, +y = 90 deg.

    Frame 0 uses positions[0]→positions[1]. Last frame uses positions[-2]→[-1].
    A boxcar smoothing window of `smoothing_window` frames de-jitters tangent
    directions on noisy paths.
    """
    if positions_m.shape[0] < 2:
        raise ValueError("Need >= 2 positions to compute yaw")

    diffs = np.zeros_like(positions_m[:, :2])
    diffs[:-1] = positions_m[1:, :2] - positions_m[:-1, :2]
    diffs[-1] = diffs[-2]  # replicate last known direction

    if smoothing_window > 1:
        # Symmetric moving average via cumulative sum
        w = int(smoothing_window)
        pad = w // 2
        padded = np.pad(diffs, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(w) / w
        smoothed = np.stack([
            np.convolve(padded[:, 0], kernel, mode="valid"),
            np.convolve(padded[:, 1], kernel, mode="valid"),
        ], axis=1)
        diffs = smoothed[: positions_m.shape[0]]

    yaw_rad = np.arctan2(diffs[:, 1], diffs[:, 0])
    return np.rad2deg(yaw_rad)
```

- [ ] **Step 4: Re-run tests, expect 10/10 PASS**

```bash
cd /data/jzy/code/SPEAR && \
  /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_trajectory -v 2>&1 | tail -20
```

Expected: `Ran 10 tests ... OK`.

- [ ] **Step 5: Commit**

```bash
cd /data/jzy/code/SPEAR
git add examples/trajectory.py tests/test_trajectory.py
git commit -m "T4 (Gate 1b): waypoint_trajectory + compute_yaw_from_positions

- waypoint_trajectory: 2+ (x,y) or (x,y,z) meters; scipy interp1d cubic (or
  linear if <4 points); auto-clip to [margin, room-margin] with warn log.
- compute_yaw_from_positions: atan2(dy, dx) with boxcar smoothing window;
  edge frames replicate their neighbor's tangent.

5 new tests green. All 10 unit tests pass.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: UE Editor import doc + cook script (Stage 2, gates 2-3)

**Files:**
- Create: `docs/animated_dog_ue_import.md`
- Create: `tools/cook_animated_dog.sh`

**Interfaces:**
- Consumes: `tmp/animated_dog/Dog_textured.glb` (T2 product); `tools/run_uat.py` (already exists, no changes).
- Produces: (after human executes doc + shell script)
  - `cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Meshes/animated_dog/{SKM_dog.uasset, SK_dog_Skeleton.uasset, anim_dog_Idle.uasset, anim_dog_Walking.uasset, M_dog.uasset}`
  - `cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Blueprints/animated_dog/BP_dog_animated.uasset`
  - Cooked pak update at `cpp/unreal_projects/SpearSim/Saved/Cooked/Linux/...`

- [ ] **Step 1: Write the human import doc**

Create `docs/animated_dog_ue_import.md`:

```markdown
# Import animated dog into SpearSim (Stage 2)

> One-time human step. After T2 has produced `tmp/animated_dog/Dog_textured.glb`,
> follow these steps in UE 5.5 Editor. After that, `tools/cook_animated_dog.sh`
> cooks the assets into the pak (Task 5 Step 2).
>
> Cross-refs: spec `docs/superpowers/specs/2026-07-04-animated-dog-gpurir-design.md`
> §Component B. Plan §Task 5.

## Prereqs

- UE 5.5 installed at `/data/UE_5.5`.
- SpearSim project at `/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/SpearSim.uproject`.
- `tmp/animated_dog/Dog_textured.glb` exists (T2 output).
- If T1 probe printed `PROBE_FAILED_*`, additionally follow the "F5 fallback"
  section at the bottom BEFORE saving the Blueprint.

## GUI steps

1. Launch UE 5.5 Editor, open the SpearSim project.
2. In the Content Browser, navigate to `/Game/MyAssets/Audioset/Meshes/`. Right-click →
   New Folder → name it `animated_dog`.
3. Drag `/data/jzy/code/SPEAR/tmp/animated_dog/Dog_textured.glb` into the new
   `animated_dog` folder in the Content Browser.
4. In the FBX/glTF Import Options dialog, set exactly:
   - **Import as Skeletal Mesh**: Yes (checked).
   - **Skeleton**: leave blank → will create a new one named `SK_dog_Skeleton`.
   - **Physics Asset**: Create if not exist.
   - **Import Materials**: Yes.
   - **Import Textures**: Yes.
   - **Import Animations**: Yes.
   - **Animation Length**: Exported Time.
   - Click **Import All**.
5. After import, the folder should contain 5 uassets:
   - `SKM_dog` (SkeletalMesh)
   - `SK_dog_Skeleton` (Skeleton)
   - `anim_dog_Idle` (AnimSequence)
   - `anim_dog_Walking` (AnimSequence)
   - `M_dog` (Material) and its underlying texture asset (auto-created).
   If naming differs (e.g. `Cylinder`, `Cylinder_Skeleton`), right-click →
   Rename to the exact names above — subsequent tasks look for these paths.
6. In the Content Browser, navigate to `/Game/MyAssets/Audioset/Blueprints/`.
   Right-click → New Folder → name it `animated_dog`.
7. Right-click `SKM_dog` → Asset Actions → Create Blueprint Using This... → base
   class `Actor` → save as `BP_dog_animated` inside the new `Blueprints/animated_dog/`
   folder.
8. Open `BP_dog_animated` in the Blueprint Editor.
9. In the Components panel: verify a `USkeletalMeshComponent` was added (called
   `SkeletalMesh` by default). In its Details panel:
   - **Skeletal Mesh Asset**: `SKM_dog`.
   - **Animation Mode**: `Use Animation Asset`.
   - **Anim to Play**: `anim_dog_Walking`.
   - **Looping**: checked.
10. Click **Compile**. Click **Save**. Close the Blueprint Editor.
11. File → **Save All**. Close the UE Editor.

## Verification (before running the cook script)

Files on disk:
```
find /data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset -type f -iname "*animated_dog*" -o -type d -iname "animated_dog"
```
Expected: 6 uassets (5 in Meshes + 1 BP in Blueprints) + 2 directories.

## F5 fallback — only if T1 probe printed `PROBE_FAILED_*`

If the raw component-level `PlayAnimation` chain didn't work in T1, add a
Blueprint-callable helper before step 11:

- In `BP_dog_animated`, add a new custom Event (right-click empty graph area →
  Add Custom Event) named `SetAndPlayWalking`.
- In its body: drag `SkeletalMesh` component → call `Set Animation Mode` (pass
  `AnimationSingleNode`) → drag → call `Set Animation` (pass a hard reference
  to `anim_dog_Walking`) → drag → call `Play` (pass `Looping=true`).
- Compile + Save.
- Task 7's Python side then calls this event via `actor.SetAndPlayWalking()`
  instead of the three raw component-level calls.
```

- [ ] **Step 2: Write the cook script**

Create `tools/cook_animated_dog.sh`:

```bash
#!/bin/bash
# Cook the animated_dog uassets into the SpearSim pak so SPEAR RPC can
# load_class(BP_dog_animated) at runtime.
#
# Prereqs: docs/animated_dog_ue_import.md fully executed. The two directories
# below MUST exist (with the uassets in them) before running.

set -euo pipefail

/data/jzy/miniconda3/envs/spear-env/bin/python /data/jzy/code/SPEAR/tools/run_uat.py \
  --unreal-engine-dir /data/UE_5.5 \
  --cook-dirs /Game/MyAssets/Audioset/Meshes/animated_dog \
              /Game/MyAssets/Audioset/Blueprints/animated_dog \
  --skip-cook-default-maps

echo "COOK_DONE"
```

- [ ] **Step 3: Make cook script executable**

```bash
chmod +x /data/jzy/code/SPEAR/tools/cook_animated_dog.sh
```

- [ ] **Step 4: Commit doc + script (before human runs them)**

```bash
cd /data/jzy/code/SPEAR
git add docs/animated_dog_ue_import.md tools/cook_animated_dog.sh
git commit -m "T5 (Stage 2): UE Editor import doc + cook script

docs/animated_dog_ue_import.md pins every FBX/glTF import dialog option
and Blueprint Editor step so the resulting BP_dog_animated is exactly
what the T7 render script expects. Includes the F5 fallback (custom
event SetAndPlayWalking) for the case where T1 printed PROBE_FAILED_*.

tools/cook_animated_dog.sh wraps run_uat.py --cook-dirs targeting the
two /Game/MyAssets/Audioset/{Meshes,Blueprints}/animated_dog directories
with --skip-cook-default-maps to avoid re-cooking the apartment map.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Human executes the UE Editor doc**

STOP and read the doc; execute the 11 GUI steps in UE Editor. After it's done,
confirm on disk:

```bash
find /data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Meshes/animated_dog \
     /data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Blueprints/animated_dog \
     -type f -iname "*.uasset" 2>&1 | sort
```
Expected: 6 files listed (SKM_dog, SK_dog_Skeleton, anim_dog_Idle, anim_dog_Walking, M_dog, BP_dog_animated).

- [ ] **Step 6: Run the cook script**

```bash
/data/jzy/code/SPEAR/tools/cook_animated_dog.sh 2>&1 | tail -30
```

Expected: prints `COOK_DONE` on last line, exit 0. Cook logs will show many "Cooking asset X" lines — grep for `animated_dog` to confirm they were included:

```bash
grep -i animated_dog /tmp/cook_animated_dog.log 2>/dev/null | head
# (if cook_animated_dog.sh doesn't tee to a log, wrap in `2>&1 | tee /tmp/cook_animated_dog.log`)
```

- [ ] **Step 7: Confirm no code changes to commit (cook only produces pak binary)**

Cook regenerates `.pak` binaries under `cpp/unreal_projects/SpearSim/Saved/Cooked/`; those are typically already `.gitignore`d in SPEAR. If `git status` shows unexpected tracked-file changes, STOP and investigate before proceeding — the UE Editor may have modified project settings.

```bash
cd /data/jzy/code/SPEAR
git status --short cpp/ 2>&1 | head
```

Expected: only untracked pak files (or nothing). If a tracked config file changed, ask the user before committing.

---

### Task 6: Cook verification (Gate 4)

**Files:**
- Create: `tools/verify_animated_dog_cook.py`

**Interfaces:**
- Consumes: T5 cook products (BP_dog_animated + anim_dog_Walking accessible via SPEAR RPC).
- Produces: exit-0 + `COOK_VERIFY_OK` printout if the SkeletalMesh + Anim + BP all load. Any failure → exit 1 with the specific missing asset name, then STOP (Task 7 cannot proceed).

- [ ] **Step 1: Write the verification script**

Create `tools/verify_animated_dog_cook.py`:

```python
"""Verify that Stage 2 (UE Editor import + cook) produced runtime-loadable
uassets for the animated dog. Blocks Task 7 (Gate 4).

Exit 0 + 'COOK_VERIFY_OK' — proceed to T7.
Exit 1 + 'COOK_VERIFY_FAILED_<what>' — go back to T5 (re-import or re-cook).
"""

import sys

sys.path.insert(0, "/data/jzy/code/SPEAR/examples")
from render_in_gpurir_room import configure_gpurir_instance  # noqa: E402

BP_PATH = "/Game/MyAssets/Audioset/Blueprints/animated_dog/BP_dog_animated.BP_dog_animated_C"
ANIM_WALKING_PATH = "/Game/MyAssets/Audioset/Meshes/animated_dog/anim_dog_Walking"
SKM_PATH = "/Game/MyAssets/Audioset/Meshes/animated_dog/SKM_dog"


def main():
    instance = configure_gpurir_instance(rpc_port=39002)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            for cls in ("APlayerStart", "ADefaultPawn", "ASpectatorPawn"):
                try:
                    for a in game.unreal_service.find_actors_by_class(uclass=cls):
                        game.unreal_service.destroy_actor(actor=a)
                except Exception:
                    pass

            try:
                bp = game.unreal_service.load_class(uclass="AActor", name=BP_PATH)
            except Exception as e:
                print(f"COOK_VERIFY_FAILED_load_class_BP: {e}", flush=True)
                sys.exit(1)
            if bp is None:
                print("COOK_VERIFY_FAILED_load_class_BP: returned None", flush=True)
                sys.exit(1)

            try:
                anim = game.unreal_service.load_object(uclass="UAnimationAsset", name=ANIM_WALKING_PATH)
            except Exception as e:
                print(f"COOK_VERIFY_FAILED_load_anim_Walking: {e}", flush=True)
                sys.exit(1)
            if anim is None:
                print("COOK_VERIFY_FAILED_load_anim_Walking: returned None", flush=True)
                sys.exit(1)

            try:
                skm = game.unreal_service.load_object(uclass="USkeletalMesh", name=SKM_PATH)
            except Exception as e:
                print(f"COOK_VERIFY_FAILED_load_SKM: {e}", flush=True)
                sys.exit(1)
            if skm is None:
                print("COOK_VERIFY_FAILED_load_SKM: returned None", flush=True)
                sys.exit(1)

            print("COOK_VERIFY_OK", flush=True)
        with instance.end_frame():
            pass
    finally:
        instance.close(force=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the verification**

```bash
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/tools/verify_animated_dog_cook.py 2>&1 | tail -10
```

Expected: last line `COOK_VERIFY_OK`, exit 0. If `COOK_VERIFY_FAILED_*`, STOP — re-check the T5 doc steps (naming, missing folders) or re-cook (`tools/cook_animated_dog.sh`).

- [ ] **Step 3: Commit the verifier**

```bash
cd /data/jzy/code/SPEAR
git add tools/verify_animated_dog_cook.py
git commit -m "T6 (Gate 4): verify_animated_dog_cook.py — check runtime load

Loads BP_dog_animated_C, SKM_dog, and anim_dog_Walking via SPEAR RPC.
Each failure prints a specific COOK_VERIFY_FAILED_<what> so it's easy
to tell what went wrong at T5 (naming vs missing vs uncooked).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `render_animated_dog_gpurir.py` — Stage 3 render script (gates 5-6)

**Files:**
- Create: `examples/render_animated_dog_gpurir.py`

**Interfaces:**
- Consumes:
  - `examples/trajectory.py::gpurir_trajectory`, `waypoint_trajectory`, `compute_yaw_from_positions`.
  - `examples/render_in_gpurir_room.py` — copy the following helpers VERBATIM into a section marked `# copied verbatim from render_in_gpurir_room.py::render_gpurir_room`: `WALL_JOINT_OVERLAP_M`, `WALL_THICKNESS_M`, `compute_shoebox_room_layout`, `compute_window_wall_layout`, `compute_window_frame_layout`, `spawn_room_piece`, `spawn_directional_light`, `spawn_point_light`, `spawn_sphere_reflection_capture`, `spawn_sky`, `_material_for_piece`, `_try_hide`, `piece_casts_shadow`, `FLOOR_MESH`, `FLOOR_MESH_TILE_M`, `FLOOR_MESH_THICKNESS_CM`, `FLOOR_MATERIAL_POOL`, `WALL_MATERIAL_POOL`, `FLOOR_MATERIAL`, `WALL_MATERIAL`, `GLASS_MATERIAL`, `WINDOW_FRAME_MATERIAL`, `WINDOW_FRAME_THICKNESS_M`, `GLASS_THICKNESS_M`, `OUTDOOR_GROUND_MATERIAL`, `CUBE_MESH`, `EMPTY_MAP`, `LIGHT_STUDIO_BP`, `SPEARSIM_EXECUTABLE`, `DEFAULT_TMP_ROOT`, `M2CM`, `resolve_floor_material`, `resolve_wall_material`, `configure_gpurir_instance`. Copy `render_in_apartment.py::spawn_camera` and `read_frame` by import (`from render_in_apartment import spawn_camera, read_frame`).
- Produces (for T8):
  - `tmp/render_animated_dog_gpurir/{run_name}/turntable.mp4`
  - `tmp/render_animated_dog_gpurir/{run_name}/trajectory.json`
  - `tmp/render_animated_dog_gpurir/{run_name}/frame_%04d.png`

- [ ] **Step 1: Create the file skeleton — imports, constants, CLI**

Create `examples/render_animated_dog_gpurir.py` with this initial content:

```python
"""Render an animated Quaternius dog inside the SPEAR shoebox room, driven
along either a GPURIR-style random trajectory or a user waypoint list.

Spec: docs/superpowers/specs/2026-07-04-animated-dog-gpurir-design.md
Plan: docs/superpowers/plans/2026-07-04-animated-dog-gpurir.md

USAGE (GPURIR trajectory):
  DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \\
    /data/jzy/miniconda3/envs/spear-env/bin/python \\
    examples/render_animated_dog_gpurir.py \\
    --trajectory-mode gpurir --trajectory-seed 42 --speed-bucket B \\
    --run-name animated_dog_gpurir_seed42

USAGE (waypoint override):
  DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \\
    /data/jzy/miniconda3/envs/spear-env/bin/python \\
    examples/render_animated_dog_gpurir.py \\
    --trajectory-mode waypoints \\
    --waypoints "0.5,0.5;2.6,0.5;2.6,4.0" \\
    --run-name animated_dog_waypoint_Lshape
"""

import argparse
import json
import os
import subprocess
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# copied verbatim from render_in_gpurir_room.py::render_gpurir_room — BEGIN
from render_in_gpurir_room import (  # noqa: E402
    WALL_JOINT_OVERLAP_M,
    WALL_THICKNESS_M,
    FLOOR_MESH,
    FLOOR_MESH_TILE_M,
    FLOOR_MESH_THICKNESS_CM,
    FLOOR_MATERIAL_POOL,
    WALL_MATERIAL_POOL,
    FLOOR_MATERIAL,
    WALL_MATERIAL,
    GLASS_MATERIAL,
    WINDOW_FRAME_MATERIAL,
    WINDOW_FRAME_THICKNESS_M,
    GLASS_THICKNESS_M,
    OUTDOOR_GROUND_MATERIAL,
    CUBE_MESH,
    EMPTY_MAP,
    SPEARSIM_EXECUTABLE,
    DEFAULT_TMP_ROOT,
    M2CM,
    compute_shoebox_room_layout,
    compute_window_wall_layout,
    compute_window_frame_layout,
    spawn_room_piece,
    spawn_directional_light,
    spawn_point_light,
    spawn_sphere_reflection_capture,
    spawn_sky,
    _material_for_piece,
    _try_hide,
    piece_casts_shadow,
    resolve_floor_material,
    resolve_wall_material,
    configure_gpurir_instance,
)
# copied verbatim from render_in_gpurir_room.py::render_gpurir_room — END

from render_in_apartment import spawn_camera, read_frame, clean_frames  # noqa: E402
from trajectory import (  # noqa: E402
    gpurir_trajectory,
    waypoint_trajectory,
    compute_yaw_from_positions,
)


ANIMATED_DOG_BP = "/Game/MyAssets/Audioset/Blueprints/animated_dog/BP_dog_animated.BP_dog_animated_C"
ANIMATED_DOG_WALKING_ANIM = "/Game/MyAssets/Audioset/Meshes/animated_dog/anim_dog_Walking"
DEFAULT_RENDER_ROOT = "/data/jzy/code/SPEAR/tmp/render_animated_dog_gpurir"


def parse_waypoints(s):
    """Parse '0.5,0.5;2.6,0.5;2.6,4.0' into [(0.5,0.5), (2.6,0.5), (2.6,4.0)]."""
    out = []
    for piece in s.split(";"):
        piece = piece.strip()
        if not piece:
            continue
        parts = [float(x) for x in piece.split(",")]
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError(
                f"waypoint must be x,y or x,y,z (got {piece!r})"
            )
        out.append(tuple(parts))
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    # trajectory
    p.add_argument("--trajectory-mode", choices=["gpurir", "waypoints"], default="gpurir")
    p.add_argument("--trajectory-seed", type=int, default=0)
    p.add_argument("--speed-bucket", choices=["A", "B", "C", "D"], default="B")
    p.add_argument("--waypoints", type=parse_waypoints, default=None,
                   help="Semicolon-separated waypoints, e.g. '0.5,0.5;2.6,4.0'")
    p.add_argument("--source-height-m", type=float, default=0.45)
    p.add_argument("--z-offset-m", type=float, default=0.0,
                   help="Vertical offset applied to every frame (metres). "
                        "Set negative if the mesh pivot is above the foot.")
    # room + lights (defaults chosen to match the last static-dog run visually)
    p.add_argument("--room-size-m", type=float, nargs=3, default=[5.2, 4.4, 2.8],
                   metavar=("X", "Y", "Z"))
    p.add_argument("--wall-thickness-m", type=float, default=WALL_THICKNESS_M)
    p.add_argument("--window-w-m", type=float, default=1.4)
    p.add_argument("--window-h-m", type=float, default=1.4)
    p.add_argument("--window-frame-thickness-m", type=float, default=WINDOW_FRAME_THICKNESS_M)
    p.add_argument("--window-cx-m", type=float, default=None,
                   help="Window center X (m). Default = room_x/2.")
    p.add_argument("--window-z-bottom-m", type=float, default=0.9)
    p.add_argument("--floor-material", default=None)
    p.add_argument("--floor-material-seed", type=int, default=0)
    p.add_argument("--wall-material", default=None)
    p.add_argument("--wall-material-seed", type=int, default=0)
    p.add_argument("--ceiling-casts-shadow", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--window-top-wall-casts-shadow", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--window-wall-casts-shadow", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ceiling-light-lumens", type=float, default=6000.0)
    p.add_argument("--ceiling-light-drop-cm", type=float, default=15.0)
    p.add_argument("--ceiling-light-attenuation-cm", type=float, default=600.0)
    p.add_argument("--reflection-capture", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--directional-light-yaw-deg", type=float, default=-90.0)
    p.add_argument("--directional-light-pitch-deg", type=float, default=-30.0)
    p.add_argument("--directional-light-intensity-lux", type=float, default=10.0)
    # camera
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    # frames
    p.add_argument("--n-frames", type=int, default=36)
    p.add_argument("--framerate", type=int, default=12)
    p.add_argument("--warmup-frames", type=int, default=30)
    p.add_argument("--per-frame-warmup-frames", type=int, default=6)
    # output
    p.add_argument("--rpc-port", type=int, default=39002)
    p.add_argument("--output-root", default=DEFAULT_RENDER_ROOT)
    p.add_argument("--run-name", required=True)
    args = p.parse_args(argv)
    if args.trajectory_mode == "waypoints" and not args.waypoints:
        p.error("--trajectory-mode waypoints requires --waypoints '...'")
    return args


def main(argv=None):
    args = parse_args(argv)
    # STUB — Step 2 will fill in
    print(f"[render_animated_dog] args: {args}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Quick syntax + import smoke check**

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -m py_compile \
  /data/jzy/code/SPEAR/examples/render_animated_dog_gpurir.py && echo COMPILE_OK
```

Expected: `COMPILE_OK`. If it fails, fix the import list (e.g. a symbol from `render_in_gpurir_room.py` was renamed — cross-check with `grep -n "^def \|^[A-Z_]* =" examples/render_in_gpurir_room.py`).

- [ ] **Step 3: Implement `render_animated_dog(args)` — spawn room + camera + dog**

Replace the `main` function's body with an actual render pipeline. Append this function above `main` and update `main` to call it:

```python
def _room_layout(args):
    """Compute pre-args room layout — identical shape to render_in_gpurir_room."""
    room_pieces = compute_shoebox_room_layout(
        room_size_m=args.room_size_m,
        wall_thickness_m=args.wall_thickness_m,
    )
    window_cx = args.window_cx_m if args.window_cx_m is not None else args.room_size_m[0] / 2.0
    window_pieces = compute_window_wall_layout(
        room_size_m=args.room_size_m,
        window_w_m=args.window_w_m,
        window_h_m=args.window_h_m,
        window_cx_m=window_cx,
        window_z_bottom_m=args.window_z_bottom_m,
        wall_thickness_m=args.wall_thickness_m,
    )
    window_frame_pieces = compute_window_frame_layout(
        room_size_m=args.room_size_m,
        window_w_m=args.window_w_m,
        window_h_m=args.window_h_m,
        window_cx_m=window_cx,
        window_z_bottom_m=args.window_z_bottom_m,
        wall_thickness_m=args.wall_thickness_m,
        frame_thickness_m=args.window_frame_thickness_m,
    )
    non_y1 = [p for p in room_pieces if p["name"] != "wall_y1"]

    ry_cm = args.room_size_m[1] * M2CM
    t_cm = args.wall_thickness_m * M2CM
    glass_piece = {
        "name": "window_glass",
        "location_cm": (
            window_cx * M2CM,
            ry_cm + t_cm / 2.0,
            (args.window_z_bottom_m + args.window_h_m / 2.0) * M2CM,
        ),
        "scale": (args.window_w_m, GLASS_THICKNESS_M, args.window_h_m),
    }
    outdoor_ground_piece = {
        "name": "outdoor_ground",
        "location_cm": (
            args.room_size_m[0] * M2CM / 2.0,
            args.room_size_m[1] * M2CM / 2.0,
            -args.wall_thickness_m * M2CM - 5.0,
        ),
        "scale": (80.0, 80.0, 0.1),
    }
    return non_y1 + window_pieces + window_frame_pieces + [glass_piece, outdoor_ground_piece]


def _spawn_room_and_lights(args, game):
    """Verbatim room+light spawn sequence copied from
    render_in_gpurir_room.py::render_gpurir_room."""
    resolved_floor = resolve_floor_material(
        floor_material=args.floor_material,
        floor_material_seed=args.floor_material_seed,
    )
    resolved_wall = resolve_wall_material(
        wall_material=args.wall_material,
        wall_material_seed=args.wall_material_seed,
    )
    print(f"[render_animated_dog] floor={resolved_floor}", flush=True)
    print(f"[render_animated_dog] wall={resolved_wall}", flush=True)

    all_pieces = _room_layout(args)
    for piece in all_pieces:
        mat = _material_for_piece(piece["name"], wall_material=resolved_wall)
        if piece["name"] == "floor":
            mat = resolved_floor
        spawn_room_piece(
            game=game,
            piece=piece,
            material_path=mat,
            cast_shadow=piece_casts_shadow(
                name=piece["name"],
                ceiling_casts_shadow=args.ceiling_casts_shadow,
                window_top_wall_casts_shadow=args.window_top_wall_casts_shadow,
                window_wall_casts_shadow=args.window_wall_casts_shadow,
            ),
        )

    spawn_sky(game=game)
    spawn_directional_light(
        game=game,
        yaw_deg=args.directional_light_yaw_deg,
        pitch_deg=args.directional_light_pitch_deg,
        intensity_lux=args.directional_light_intensity_lux,
    )
    room_x_cm = args.room_size_m[0] * M2CM
    room_y_cm = args.room_size_m[1] * M2CM
    room_z_cm = args.room_size_m[2] * M2CM
    spawn_point_light(
        game=game,
        x_cm=room_x_cm / 2.0,
        y_cm=room_y_cm / 2.0,
        z_cm=room_z_cm - args.ceiling_light_drop_cm,
        intensity_lumens=args.ceiling_light_lumens,
        attenuation_cm=args.ceiling_light_attenuation_cm,
    )
    if args.reflection_capture:
        spawn_sphere_reflection_capture(
            game=game,
            x_cm=room_x_cm / 2.0,
            y_cm=room_y_cm / 2.0,
            z_cm=room_z_cm / 2.0,
            influence_radius_cm=max(room_x_cm, room_y_cm, room_z_cm),
        )
    return resolved_floor, resolved_wall


def _spawn_animated_dog(game, x_cm, y_cm, z_cm):
    """Spawn BP_dog_animated and start Walking on loop. Returns (actor, smc)."""
    bp = game.unreal_service.load_class(uclass="AActor", name=ANIMATED_DOG_BP)
    actor = game.unreal_service.spawn_actor(
        uclass=bp,
        location={"X": float(x_cm), "Y": float(y_cm), "Z": float(z_cm)},
        spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
    )
    smc = game.unreal_service.get_component_by_class(actor=actor, uclass="USkeletalMeshComponent")
    smc.SetAnimationMode(NewAnimationMode="AnimationSingleNode")
    anim = game.unreal_service.load_object(uclass="UAnimationAsset", name=ANIMATED_DOG_WALKING_ANIM)
    smc.SetAnimation(NewAnimToPlay=anim)
    smc.Play(bLooping=True)
    return actor, smc


def _compute_trajectory(args):
    if args.trajectory_mode == "gpurir":
        pos_m = gpurir_trajectory(
            room_size_m=tuple(args.room_size_m),
            n_frames=args.n_frames,
            speed_bucket=args.speed_bucket,
            source_height_m=args.source_height_m,
            traj_aug=True,
            seed=args.trajectory_seed,
            traj_pts_full=200,
        )
    else:
        pos_m = waypoint_trajectory(
            waypoints_m=args.waypoints,
            n_frames=args.n_frames,
            room_size_m=tuple(args.room_size_m),
            source_height_m=args.source_height_m,
            kind="cubic",
        )
    if args.z_offset_m:
        pos_m[:, 2] = pos_m[:, 2] + args.z_offset_m
    yaw_deg = compute_yaw_from_positions(pos_m)
    return pos_m, yaw_deg


def render_animated_dog(args):
    output_dir = os.path.join(args.output_root, args.run_name)
    clean_frames(output_dir)
    positions_m, yaw_deg = _compute_trajectory(args)

    instance = configure_gpurir_instance(rpc_port=args.rpc_port)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            # Defensive sweep of Entry-map defaults + spawn-cube prevention
            for cls in ("APlayerStart", "ADefaultPawn", "ASpectatorPawn",
                        "AStaticMeshActor", "ASkeletalMeshActor", "ABrush",
                        "ADecalActor", "AInstancedFoliageActor",
                        "AGameplayDebuggerCategoryReplicator",
                        "AGameplayDebuggerPlayerManager"):
                try:
                    for a in game.unreal_service.find_actors_by_class(uclass=cls):
                        game.unreal_service.destroy_actor(actor=a)
                except Exception:
                    pass

            resolved_floor, resolved_wall = _spawn_room_and_lights(args, game)
            cam, comp = spawn_camera(game=game, width=args.width, height=args.height)
            dog_actor, _dog_smc = _spawn_animated_dog(
                game,
                x_cm=positions_m[0, 0] * M2CM,
                y_cm=positions_m[0, 1] * M2CM,
                z_cm=positions_m[0, 2] * M2CM,
            )
        with instance.end_frame():
            pass

        # VT warmup
        instance.step(num_frames=args.warmup_frames)

        # Per-frame drive: teleport dog, wait a few frames for the walk
        # animation to advance, render.
        for i in range(args.n_frames):
            with instance.begin_frame():
                dog_actor.K2_SetActorLocationAndRotation(
                    NewLocation={
                        "X": float(positions_m[i, 0] * M2CM),
                        "Y": float(positions_m[i, 1] * M2CM),
                        "Z": float(positions_m[i, 2] * M2CM),
                    },
                    NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": float(yaw_deg[i])},
                    bSweep=False,
                    bTeleport=True,
                )
            with instance.end_frame():
                pass
            instance.step(num_frames=args.per_frame_warmup_frames)
            with instance.begin_frame():
                pass
            with instance.end_frame():
                cv2.imwrite(
                    os.path.join(output_dir, f"frame_{i:04d}.png"),
                    read_frame(comp),
                )

        # ffmpeg mux
        video_path = os.path.join(output_dir, "turntable.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(args.framerate),
                "-i", os.path.join(output_dir, "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                video_path,
            ],
            check=True, capture_output=True,
        )
        print(f"VIDEO_DONE {video_path}", flush=True)

        # trajectory.json — cross-modal alignment metadata
        traj_json = {
            "trajectory_mode": args.trajectory_mode,
            "trajectory_seed": args.trajectory_seed,
            "speed_bucket": args.speed_bucket,
            "source_height_m": args.source_height_m,
            "z_offset_m": args.z_offset_m,
            "room_size_m": list(args.room_size_m),
            "n_frames": args.n_frames,
            "traj_pts_full": 200,
            "positions_m": positions_m.tolist(),
            "yaw_deg": yaw_deg.tolist(),
            "mic_pos_m": [args.room_size_m[0] / 2.0, args.room_size_m[1] / 2.0, 1.2],
            "resolved_floor_material": resolved_floor,
            "resolved_wall_material": resolved_wall,
        }
        if args.trajectory_mode == "waypoints":
            traj_json["waypoints_m"] = [list(wp) for wp in args.waypoints]
        with open(os.path.join(output_dir, "trajectory.json"), "w") as f:
            json.dump(traj_json, f, indent=2)
        print(f"TRAJECTORY_JSON_DONE {os.path.join(output_dir, 'trajectory.json')}", flush=True)
    finally:
        instance.close(force=True)


def main(argv=None):
    args = parse_args(argv)
    render_animated_dog(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Compile check**

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -m py_compile \
  /data/jzy/code/SPEAR/examples/render_animated_dog_gpurir.py && echo COMPILE_OK
```

Expected: `COMPILE_OK`.

- [ ] **Step 5: Commit before rendering**

```bash
cd /data/jzy/code/SPEAR
git add examples/render_animated_dog_gpurir.py
git commit -m "T7 (Stage 3): render_animated_dog_gpurir.py

New standalone script (Q12=A) that:
- Reuses room/light/camera helpers imported from render_in_gpurir_room.py
  (all imports labelled 'copied verbatim from render_in_gpurir_room.py').
- Spawns BP_dog_animated (Stage 2 cook), calls SetAnimationMode →
  SetAnimation → Play(bLooping=True) on the SkeletalMeshComponent.
- Computes trajectory from either gpurir_trajectory or waypoint_trajectory
  (T3/T4), yaw from compute_yaw_from_positions.
- Per-frame K2_SetActorLocationAndRotation teleport + per_frame_warmup_frames
  step to let Walking anim advance + render.
- Writes turntable.mp4 + trajectory.json (cross-modal alignment metadata).

Static-dog render_in_gpurir_room.py untouched (Q12=A).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Render + user review of both videos (Gates 5, 6)

**Files:**
- Produces (not committed): `tmp/render_animated_dog_gpurir/animated_dog_gpurir_seed42/{turntable.mp4, trajectory.json, frame_*.png}`.
- Produces (not committed): `tmp/render_animated_dog_gpurir/animated_dog_waypoint_Lshape/{turntable.mp4, trajectory.json, frame_*.png}`.

**Interfaces:**
- Consumes: T7 render script; T6 cook verification passed.
- Produces: user pass/fail feedback that gates T9.

- [ ] **Step 1: Render Video 1 (GPURIR trajectory, seed 42)**

```bash
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/examples/render_animated_dog_gpurir.py \
  --trajectory-mode gpurir --trajectory-seed 42 --speed-bucket B \
  --run-name animated_dog_gpurir_seed42 2>&1 | tail -10
```

Expected: last lines include `VIDEO_DONE ...turntable.mp4` and `TRAJECTORY_JSON_DONE ...trajectory.json`, exit 0. Absolute output path:
```
/data/jzy/code/SPEAR/tmp/render_animated_dog_gpurir/animated_dog_gpurir_seed42/turntable.mp4
```

- [ ] **Step 2: Self-inspection (agent, before showing user)**

Read the following frames and check the human review checklist from the spec:

```bash
ls /data/jzy/code/SPEAR/tmp/render_animated_dog_gpurir/animated_dog_gpurir_seed42/*.png | head
```

Look at `frame_0000.png`, `frame_0009.png`, `frame_0017.png`, `frame_0026.png`, `frame_0035.png`.

Checklist:
- Dog visible in room, has fur texture (NOT solid grey)?
- Legs animate (frame_0000 vs frame_0017 show a different leg pose)?
- Position moves over time (frame_0000 vs frame_0035 not identical)?
- Yaw follows tangent (dog nose points along direction of travel)?

If any is NO, STOP and diagnose (typical issues: F6 T-pose → check SetAnimationMode call; F9 pivot → tune `--z-offset-m`; material grey → re-check T2 export).

- [ ] **Step 3: Send Video 1 to user for review**

Tell the user the path (`/data/jzy/code/SPEAR/tmp/render_animated_dog_gpurir/animated_dog_gpurir_seed42/turntable.mp4`) and ask for pass/fail based on the spec's human review checklist (dog textured, legs animate, moves, faces direction, room lighting matches static baseline). WAIT for their answer before continuing to Step 4.

- [ ] **Step 4: Render Video 2 (waypoint L-shape) only if user passed Video 1**

```bash
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/examples/render_animated_dog_gpurir.py \
  --trajectory-mode waypoints \
  --waypoints "0.5,0.5;2.6,0.5;2.6,4.0" \
  --run-name animated_dog_waypoint_Lshape 2>&1 | tail -10
```

Expected: same VIDEO_DONE / TRAJECTORY_JSON_DONE lines. Output at
```
/data/jzy/code/SPEAR/tmp/render_animated_dog_gpurir/animated_dog_waypoint_Lshape/turntable.mp4
```

- [ ] **Step 5: Self-inspection of Video 2**

Look at frame_0000, frame_0017, frame_0035. Additional checks:
- Video 2 shows an L-shape trajectory (dog goes east first, then turns north)?
- 90° turn happens around frame 17-18 (halfway)?

- [ ] **Step 6: Send Video 2 to user for review**

Same as Step 3 — WAIT for pass/fail before continuing to T9.

- [ ] **Step 7: No commit for T8 (renders are not tracked)**

`tmp/` is not in git. Skip commit. Progress is captured by the user's pass verdict in the next task's HANDOFF doc.

---

### Task 9: HANDOFF doc + wrap-up (Gate 7)

**Files:**
- Create: `HANDOFF_ANIMATED_DOG_GPURIR.md`

**Interfaces:**
- Consumes: user pass on Video 1 and Video 2.
- Produces: handoff doc for future agents / next-spec kick-off.

- [ ] **Step 1: Write the handoff doc**

Create `HANDOFF_ANIMATED_DOG_GPURIR.md` (mirrors the existing `HANDOFF_GPURIR_ROOM.md` style — quick spec pointer, run commands, key files, gotchas, follow-up specs):

```markdown
# Handoff: Animated Dog with GPURIR-aligned Trajectory

**Status**: Landed and user-approved on 2026-07-04.
**Spec**: `docs/superpowers/specs/2026-07-04-animated-dog-gpurir-design.md`
**Plan**: `docs/superpowers/plans/2026-07-04-animated-dog-gpurir.md`

## What this gives you

A standalone SPEAR pipeline that renders a walking Quaternius dog (with fur
texture, playing the Walking cycle in place) driven along either:

- a GPURIR-style random trajectory (byte-identical to
  `v77_4ch_S2L/data_gen/gen_rir_multiscene_v77.get_pos_traj` for the same
  seed — future audio/video will align automatically), or
- a user-supplied semicolon-separated list of (x,y) waypoints.

Delivered videos at
- `tmp/render_animated_dog_gpurir/animated_dog_gpurir_seed42/turntable.mp4`
- `tmp/render_animated_dog_gpurir/animated_dog_waypoint_Lshape/turntable.mp4`

## How to rerun

Prereqs: `spear-env` conda env with `spear_ext`, Xvfb `:99`, Vulkan setup
(see spec §Env). Stage 1/2 (Blender + UE Editor + cook) already done — you
only re-do them if the dog asset changes.

```bash
# GPURIR trajectory, any seed
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python \
  examples/render_animated_dog_gpurir.py \
  --trajectory-mode gpurir --trajectory-seed 100 --run-name my_run

# Waypoint trajectory
--trajectory-mode waypoints --waypoints "1,1;3,2;3,4"
```

## Key files

| File | Purpose |
|---|---|
| `examples/render_animated_dog_gpurir.py` | Stage 3 main script |
| `examples/trajectory.py` | GPURIR replica + waypoint interp + yaw |
| `tests/test_trajectory.py` | 10 unit tests (all green) |
| `tools/probe_skeletal_playanimation.py` | Gate 0 probe |
| `tools/download_polyhaven_dog_fur.py` | fetches CC0 fur diffuse |
| `tools/blender_add_uv_and_texture.py` | Stage 1 UV + texture |
| `docs/animated_dog_ue_import.md` | Stage 2 human GUI steps |
| `tools/cook_animated_dog.sh` | Stage 2 cook wrapper |
| `tools/verify_animated_dog_cook.py` | Gate 4 verifier |

## Gotchas / lessons

- Must use `spear-env` python (has compiled `spear_ext`); `thu` env silently fails.
- `render_in_gpurir_room.py` was NOT modified (Q12=A). Room/light helpers are
  imported by name into `render_animated_dog_gpurir.py`.
- `gpurir_trajectory` uses `np.random.seed(seed) + save/restore of global state`
  to stay byte-identical with v77. Do NOT switch to `RandomState(seed)` — that
  breaks the alignment contract (`test_gpurir_matches_v77`).
- If `dog_fur_diffuse.jpg` PolyHaven URL 404s in future, update `ASSET_SLUG`
  in `tools/download_polyhaven_dog_fur.py`.

## Follow-up specs (out of scope here)

1. **AI motion generation** (Q4=D deferred): AI4Animation / OmniMotionGPT to
   generate new animations (Sit, Jump, Bark) beyond Idle+Walking.
2. **Material Anything AI texture** (Q5=c deferred): AI-generated PBR for
   comparison against the PolyHaven fur baseline. Must NOT regen mesh — see
   spec F5 note about "input mesh + output texture only".
3. **Scene 1/2/3 from `数据集生成探索.md`**: multi-instance (animated + static
   dogs coexisting), human/appliance/instrument added, full QA metadata schema.
4. **RIR audio integration**: hook `gen_rir_multiscene_v77.get_pos_traj` at
   same seed → same trajectory as this video pipeline (Q13=C contract).
```

- [ ] **Step 2: Commit the handoff**

```bash
cd /data/jzy/code/SPEAR
git add HANDOFF_ANIMATED_DOG_GPURIR.md
git commit -m "T9 (Gate 7): HANDOFF_ANIMATED_DOG_GPURIR.md

Landed and user-approved. Mirrors HANDOFF_GPURIR_ROOM.md style: quick
spec/plan pointers, rerun commands, key files, gotchas, and the
follow-up specs (AI motion, Material Anything, Scene 1/2/3, RIR audio).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 3: Summarise to user**

Tell the user:
- Spec + plan + all 7 gates done, both videos approved.
- 10/10 unit tests green.
- 8 new commits on top of the pre-spec HEAD.
- Handoff doc at `HANDOFF_ANIMATED_DOG_GPURIR.md`.
- Ready for the next spec (Scene 1/2/3, AI motion, Material Anything, or RIR
  audio — user picks).

---

## Self-review

**Spec coverage (spec § → task):**
- §Component A (Blender UV+texture) → T2
- §Component B (UE import doc + cook) → T5 + T6
- §Component C (trajectory.py) → T3 + T4
- §Component D (render script) → T7
- §Data flow / Q13=C alignment → T3 Step 5 replaces RandomState with saved global; test_gpurir_matches_v77 + test_gpurir_downsample_matches_c_choice enforce it
- §Failure modes F1-F5 → embedded in T2 assertions (F1, F2), T5 doc (F3, F4), T1 probe (F5)
- §Failure modes F6-F9 → T7 code sequence (F6), T3 impl (F7), T4 code (F8), T7 CLI + T8 self-inspection (F9)
- §Testing: unit → T3+T4; integration → T2/T6/T1; E2E → T8
- §Gates 0-7 → T1, T3+T4, T2, T5, T6, T8-video1, T8-video2, T9

**Placeholder scan**: no "TBD/TODO/implement later/similar to Task N/write tests for the above" in the plan body. Every code step contains complete code.

**Type consistency**:
- `gpurir_trajectory` signature identical between spec, T3 impl, T7 call, and T3 test.
- `waypoint_trajectory` signature identical between spec, T4 impl, T7 call, and T4 tests.
- `compute_yaw_from_positions` signature identical between spec, T4 impl, T7 call, T4 tests.
- BP path `/Game/MyAssets/Audioset/Blueprints/animated_dog/BP_dog_animated.BP_dog_animated_C` used consistently across T5 doc, T6 verifier, T7 script.
- Anim path `/Game/MyAssets/Audioset/Meshes/animated_dog/anim_dog_Walking` used consistently across T5 doc, T6 verifier, T7 script.
