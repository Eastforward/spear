# 12-Quadruped Species-Matched Rig Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce UE orbit walking videos for 12 audioset quadruped animals, using 3 species-matched source rigs (Quaternius Cat/Dog/Wolf) instead of the previous "everyone uses Dog" strategy.

**Architecture:** Hunyuan3D generates each species' textured mesh, then `robust_skin_transfer` swaps that mesh onto the anatomically-nearest Quaternius source rig (Cat rig for small felids, Dog rig for medium canid/pig/goat/sheep, Wolf rig for large ungulates). All three source rigs share identical 34-bone Quaternius template so no code changes are needed in the transfer/segmentation logic. A "gate check" stage renders one animal per rig family into UE first and requires human video approval before running the full batch of 12.

**Tech Stack:** Blender 4.2 (rig transfer), Hunyuan3D-2.1 (mesh+texture generation), Unreal Engine 5.5 + SPEAR (cook + orbit render), Flux/SDXL (reference image generation).

## Global Constraints

- Python interpreter: `/data/jzy/miniconda3/envs/spear-env/bin/python` (SPEAR env with compiled spear_ext, py3.11). Do NOT use the `thu` env — it silently fails RPC connect.
- Never commit HF token `<REDACTED_HUGGINGFACE_TOKEN>` to git or share externally.
- All source rigs live at `/data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack/{Cat,Dog,Wolf}.glb`.
- All flags default to the tuning Codex arrived at last session EXCEPT:
  - `--reverse-actions no` (always — this plan explicitly removes reverse-based direction hacks)
  - `--remove-limb-bridge-components yes` stays (auto-gated by `--limb-bridge-component-min-direct-faces=200`, so it only activates when a mesh actually needs it)
- Orbit videos: 640x480, 72 frames, 15 fps side view, camera radius 200 cm, height 60 cm.
- User verification gate: after Task 3 (Cat gate check) and Task 4 (Wolf gate check), STOP and present videos to user; do not proceed to Task 5+ without user's explicit approval.

## Scope

**In scope (12 quadrupeds):**

| Tag              | Species | Breed             | Source Rig | Existing Hunyuan mesh? |
|------------------|---------|-------------------|------------|------------------------|
| cat_persian      | cat     | persian           | Cat.glb    | YES (`SPEAR/tmp/hy3d_batch/cat_persian/hy3d_textured.obj`) |
| cat_tabby        | cat     | orange tabby      | Cat.glb    | YES (`SPEAR/tmp/hy3d_batch/cat_tabby/hy3d_textured.obj`) |
| chipmunk         | chipmunk| generic           | Cat.glb    | NO — need Flux + Hunyuan |
| dog_golden       | dog     | golden retriever  | Dog.glb    | YES (`SPEAR/tmp/hy3d_batch/dog_golden/hy3d_textured.obj`) |
| dog_husky        | dog     | siberian husky    | Dog.glb    | YES (`SPEAR/tmp/hy3d_batch/dog_husky/hy3d_textured.obj`) |
| goat             | goat    | generic           | Dog.glb    | YES (`Hunyuan3D-2.1/outputs/audioset_assets/goat/goat_textured.glb`) |
| sheep            | sheep   | generic           | Dog.glb    | YES (`.../sheep/sheep_textured.glb`) |
| pig              | pig     | generic           | Dog.glb    | YES (`.../pig/pig_textured.glb`) |
| horse            | horse   | generic           | Wolf.glb   | YES (`.../horse/horse_textured.glb`) |
| cattle_bovinae   | cattle  | bovinae           | Wolf.glb   | YES (`.../cattle_bovinae/cattle_bovinae_textured.glb`) |
| yak              | yak     | generic           | Wolf.glb   | YES (`.../yak/yak_textured.glb`) |
| donkey_ass       | donkey  | ass               | Wolf.glb   | YES (`.../donkey_ass/donkey_ass_textured.glb`) |

**Out of scope this plan:**
- All birds (chicken, duck, goose, turkey, pigeon, crow, owl, gull, bird_animal) — separate future plan.
- Rodents other than chipmunk (mouse, rat) — user did not include.
- Frog/snake/insects — not walking-animation-appropriate.

---

### Task 1: Add species→rig mapping table + gate check runner script

**Files:**
- Create: `SPEAR/tools/species_rig_map.py`
- Create: `SPEAR/tools/gate_check_animal.sh`

**Interfaces:**
- Consumes: existing `robust_skin_transfer.py`, `blender_robust_swap_mesh_keep_rig.py`, `build_animated_dog.sh` conventions, `/tmp/orbit_dog.py`.
- Produces: `species_rig_map.py::RIG_MAP` (dict of tag → dict{source_rig_glb, hy3d_mesh_path, hy3d_diffuse_path}). `gate_check_animal.sh <tag>` — end-to-end pipeline for one tag, output at `/tmp/gate_check_v4/<tag>_side.mp4`.

- [ ] **Step 1: Create the species→rig mapping module**

Write `/data/jzy/code/SPEAR/tools/species_rig_map.py`:

```python
"""Species → source rig mapping for 12-quadruped batch pipeline.

Cat rig (Quaternius) — small felids and rodents:  cat, chipmunk
Dog rig — medium canid/pig/goat/sheep:            dog, goat, sheep, pig
Wolf rig — large ungulates:                       horse, cattle, yak, donkey
"""

import os

QUATERNIUS_DIR = "/data/jzy/code/Spatial/v77_4ch_S2L/assets/mesh_library/quaternius_animalpack"
HY3D_BATCH_DIR = "/data/jzy/code/SPEAR/tmp/hy3d_batch"
HY3D_AUDIOSET_DIR = "/data/jzy/code/Hunyuan3D-2.1/outputs/audioset_assets"


def _batch_mesh(tag):
    return {
        "mesh": f"{HY3D_BATCH_DIR}/{tag}/hy3d_textured.obj",
        "diffuse": f"{HY3D_BATCH_DIR}/{tag}/hy3d_textured.jpg",
    }


def _audioset_mesh(dirname):
    base = f"{HY3D_AUDIOSET_DIR}/{dirname}/{dirname}_textured"
    return {"mesh": f"{base}.obj", "diffuse": f"{base}.jpg"}


CAT_RIG = f"{QUATERNIUS_DIR}/Cat.glb"
DOG_RIG = f"{QUATERNIUS_DIR}/Dog.glb"
WOLF_RIG = f"{QUATERNIUS_DIR}/Wolf.glb"


RIG_MAP = {
    "cat_persian":     {"rig": CAT_RIG,  **_batch_mesh("cat_persian")},
    "cat_tabby":       {"rig": CAT_RIG,  **_batch_mesh("cat_tabby")},
    "chipmunk":        {"rig": CAT_RIG,  **_batch_mesh("chipmunk")},
    "dog_golden":      {"rig": DOG_RIG,  **_batch_mesh("dog_golden")},
    "dog_husky":       {"rig": DOG_RIG,  **_batch_mesh("dog_husky")},
    "goat":            {"rig": DOG_RIG,  **_audioset_mesh("goat")},
    "sheep":           {"rig": DOG_RIG,  **_audioset_mesh("sheep")},
    "pig":             {"rig": DOG_RIG,  **_audioset_mesh("pig")},
    "horse":           {"rig": WOLF_RIG, **_audioset_mesh("horse")},
    "cattle_bovinae":  {"rig": WOLF_RIG, **_audioset_mesh("cattle_bovinae")},
    "yak":             {"rig": WOLF_RIG, **_audioset_mesh("yak")},
    "donkey_ass":      {"rig": WOLF_RIG, **_audioset_mesh("donkey_ass")},
}


def assert_inputs_exist(tag):
    entry = RIG_MAP[tag]
    missing = [k for k in ("rig", "mesh", "diffuse") if not os.path.exists(entry[k])]
    if missing:
        raise SystemExit(f"[species_rig_map] {tag} missing: {[(k, entry[k]) for k in missing]}")


if __name__ == "__main__":
    import sys
    tag = sys.argv[1] if len(sys.argv) > 1 else None
    if tag:
        assert_inputs_exist(tag)
        print(f"[species_rig_map] {tag}: OK", flush=True)
    else:
        for t in RIG_MAP:
            print(t, "->", RIG_MAP[t])
```

- [ ] **Step 2: Verify all 12 rigs' source-rig paths exist (mesh paths may not yet)**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -c "from SPEAR.tools import species_rig_map; import os; [print(t, os.path.exists(e['rig']), os.path.exists(e['mesh']), os.path.exists(e['diffuse'])) for t, e in species_rig_map.RIG_MAP.items()]"`

Expected: all `rig=True`. `mesh/diffuse=True` for at least: cat_persian, cat_tabby, dog_golden, dog_husky, goat, sheep, pig, horse, cattle_bovinae, yak, donkey_ass. Only chipmunk may be missing (Task 6 generates it).

- [ ] **Step 3: Create the per-animal gate check shell script**

Write `/data/jzy/code/SPEAR/tools/gate_check_animal.sh`:

```bash
#!/bin/bash
# Gate check runner for one species tag:
#   1. Rig-swap Hunyuan mesh onto species-matched Quaternius source rig
#   2. UE headless import + cook
#   3. Render 72-frame side-view orbit video
#
# Output: /tmp/gate_check_v4/<tag>_side.mp4
set -uo pipefail

if [ -z "${1:-}" ]; then
    echo "usage: gate_check_animal.sh <tag>"
    exit 1
fi
TAG="$1"

SPEAR_DIR=/data/jzy/code/SPEAR
UE_DIR=/data/UE_5.5
PY=/data/jzy/miniconda3/envs/spear-env/bin/python
BLENDER=/data/jzy/blender-4.2.0-linux-x64/blender

# --- Task 1: verify inputs
$PY -c "import sys; sys.path.insert(0, '$SPEAR_DIR/..'); from SPEAR.tools import species_rig_map as m; m.assert_inputs_exist('$TAG'); import json; print(json.dumps(m.RIG_MAP['$TAG']))" \
    > "/tmp/gate_check_v4_${TAG}_inputs.json"
RIG=$($PY -c "import json; print(json.load(open('/tmp/gate_check_v4_${TAG}_inputs.json'))['rig'])")
MESH=$($PY -c "import json; print(json.load(open('/tmp/gate_check_v4_${TAG}_inputs.json'))['mesh'])")
DIFF=$($PY -c "import json; print(json.load(open('/tmp/gate_check_v4_${TAG}_inputs.json'))['diffuse'])")
echo "[gate_check] tag=$TAG rig=$RIG mesh=$MESH diff=$DIFF"

# --- Task 2: rig swap (Blender)
mkdir -p /tmp/gate_check_v4
RIGGED_GLB=/tmp/gate_check_v4/${TAG}_rigged.glb
$BLENDER --background --python $SPEAR_DIR/tools/blender_robust_swap_mesh_keep_rig.py -- \
    --rig-glb "$RIG" \
    --new-mesh "$MESH" \
    --new-diffuse "$DIFF" \
    --output "$RIGGED_GLB" \
    --reverse-actions no \
    2>&1 | tail -40

if [ ! -f "$RIGGED_GLB" ]; then
    echo "GATE_CHECK_FAIL rig swap did not produce $RIGGED_GLB"
    exit 1
fi

# --- Task 3: UE cook (reuse build_animated_dog.sh style but per-tag)
MESH_DIR="$SPEAR_DIR/cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Meshes/gate_${TAG}"
BP_DIR="$SPEAR_DIR/cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Blueprints/gate_${TAG}"
BP_PATH="$BP_DIR/BP_gate_${TAG}.uasset"

echo "=== wipe stale gate_${TAG} uassets ==="
rm -rf "$MESH_DIR" "$BP_DIR"

echo "=== UE headless import ==="
GATE_TAG=$TAG GATE_RIGGED_GLB=$RIGGED_GLB \
    $PY $SPEAR_DIR/tools/run_editor_script.py \
    --script $SPEAR_DIR/tools/import_gate_animal_editor.py \
    --unreal-engine-dir $UE_DIR \
    --launch-mode commandlet \
    || echo "(editor commandlet returned nonzero — checking BP presence)"

if [ ! -f "$BP_PATH" ]; then
    echo "GATE_CHECK_FAIL BP_uasset missing at $BP_PATH"
    exit 1
fi
echo "GATE_CHECK_IMPORT_OK $BP_PATH"

echo "=== UE cook ==="
$PY $SPEAR_DIR/tools/run_uat.py \
    --unreal-engine-dir $UE_DIR \
    --skip-cook-default-maps \
    -build -cook -stage -package -archive -pak

# --- Task 4: orbit render
ORBIT_DIR=/tmp/gate_check_v4/orbit_${TAG}
rm -rf "$ORBIT_DIR"
GATE_TAG=$TAG $PY /tmp/orbit_animal.py \
    --tag "$TAG" \
    --n-frames 72 \
    --output-dir "$ORBIT_DIR"

# --- Task 5: encode
OUTMP4=/tmp/gate_check_v4/${TAG}_side.mp4
ffmpeg -y -framerate 15 -i "$ORBIT_DIR/frame_%04d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 "$OUTMP4" 2>&1 | tail -5

if [ ! -s "$OUTMP4" ]; then
    echo "GATE_CHECK_FAIL video empty at $OUTMP4"
    exit 1
fi
echo "GATE_CHECK_DONE $OUTMP4"
```

Then make executable:
```bash
chmod +x /data/jzy/code/SPEAR/tools/gate_check_animal.sh
```

- [ ] **Step 4: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/species_rig_map.py tools/gate_check_animal.sh
git commit -m "add species→rig map + per-animal gate check runner"
```

---

### Task 2: Create per-tag UE editor import script + generic orbit renderer

**Files:**
- Create: `SPEAR/tools/import_gate_animal_editor.py`
- Create: `/tmp/orbit_animal.py`

**Interfaces:**
- Consumes: `$GATE_TAG` env var (species tag), `$GATE_RIGGED_GLB` env var (path to rigged GLB from Task 1).
- Produces: uasset `.../Blueprints/gate_{tag}/BP_gate_{tag}.uasset` + Walking anim asset under `.../Meshes/gate_{tag}/Walking`; PNG frames in `--output-dir`.

- [ ] **Step 1: Adapt import_animated_dog_editor.py to be tag-parameterized**

First read the reference: `cat /data/jzy/code/SPEAR/tools/import_animated_dog_editor.py`. It hard-codes `animated_dog`, `Dog_textured.glb`, `dog_animated` — we need a version that reads `os.environ["GATE_TAG"]` and `os.environ["GATE_RIGGED_GLB"]` and substitutes those into the paths.

Write `/data/jzy/code/SPEAR/tools/import_gate_animal_editor.py`:

```python
"""Headless UE editor script: import <GATE_RIGGED_GLB> for tag <GATE_TAG>.

Runs inside UE editor via commandlet (run_editor_script.py). Reads:
  GATE_TAG          — species tag, e.g. "cat_persian"
  GATE_RIGGED_GLB   — absolute path to rigged skeletal-mesh GLB

Emits Blueprint at /Game/MyAssets/Audioset/Blueprints/gate_{TAG}/BP_gate_{TAG}.
"""
import os
import unreal


TAG = os.environ["GATE_TAG"]
RIGGED_GLB = os.environ["GATE_RIGGED_GLB"]
assert os.path.exists(RIGGED_GLB), RIGGED_GLB

MESH_DIR = f"/Game/MyAssets/Audioset/Meshes/gate_{TAG}"
BP_DIR = f"/Game/MyAssets/Audioset/Blueprints/gate_{TAG}"
BP_NAME = f"BP_gate_{TAG}"


def _delete_dir_if_exists(pkg_dir):
    if unreal.EditorAssetLibrary.does_directory_exist(directory_path=pkg_dir):
        unreal.EditorAssetLibrary.delete_directory(directory_path=pkg_dir)


def import_glb():
    _delete_dir_if_exists(MESH_DIR)
    _delete_dir_if_exists(BP_DIR)
    unreal.EditorAssetLibrary.make_directory(directory_path=MESH_DIR)
    unreal.EditorAssetLibrary.make_directory(directory_path=BP_DIR)

    task = unreal.AssetImportTask()
    task.filename = RIGGED_GLB
    task.destination_path = MESH_DIR
    task.replace_existing = True
    task.automated = True
    task.save = True
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])


def create_blueprint():
    imported = unreal.EditorAssetLibrary.list_assets(MESH_DIR, recursive=True)
    skeletal_mesh = None
    for a in imported:
        obj = unreal.EditorAssetLibrary.load_asset(a)
        if isinstance(obj, unreal.SkeletalMesh):
            skeletal_mesh = obj
            break
    assert skeletal_mesh is not None, f"no SkeletalMesh imported into {MESH_DIR}"

    factory = unreal.BlueprintFactory()
    factory.set_editor_property("parent_class", unreal.Actor)
    bp = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
        BP_NAME, BP_DIR, unreal.Blueprint, factory,
    )
    subobj_ds = unreal.SubobjectDataSubsystem.get()
    root = subobj_ds.k2_gather_subobject_data_for_blueprint(context=bp)[0]
    params = unreal.AddNewSubobjectParams(
        parent_handle=root,
        new_class=unreal.SkeletalMeshComponent,
        blueprint_context=bp,
    )
    handle, fail = subobj_ds.add_new_subobject(params=params)
    subobj = subobj_ds.k2_find_subobject_data_from_handle(handle)
    smc = unreal.SubobjectDataBlueprintFunctionLibrary.get_object(subobj)
    smc.set_editor_property("skeletal_mesh_asset", skeletal_mesh)
    unreal.EditorAssetLibrary.save_asset(bp.get_path_name())


import_glb()
create_blueprint()
print(f"GATE_ANIMAL_IMPORT_OK tag={TAG}", flush=True)
```

**Note:** If your `import_animated_dog_editor.py` uses different UE Python API idioms (e.g., older `KismetSystemLibrary` calls), copy those verbatim into `import_gate_animal_editor.py` instead — the goal is byte-for-byte behavior parity, just with `TAG` substitution.

- [ ] **Step 2: Sanity-check the tag substitution**

Cross-check: `diff <(sed 's/animated_dog/GATE_TAG/g; s/Dog_textured/GATE_RIGGED/g; s/dog_animated/GATE_TAG_BP/g' /data/jzy/code/SPEAR/tools/import_animated_dog_editor.py) /data/jzy/code/SPEAR/tools/import_gate_animal_editor.py`

Expected: differences are only structural (env-var loading, template strings), not logic changes. Fix any drift.

- [ ] **Step 3: Adapt orbit_dog.py to be tag-parameterized**

Write `/tmp/orbit_animal.py` (start by copying `/tmp/orbit_dog.py` and parameterizing):

```python
"""SPEAR: render N frames of one gate-check animal with the camera orbiting.

Usage:
  DISPLAY=:99 python /tmp/orbit_animal.py --tag cat_persian --n-frames 72 --output-dir /tmp/gate_check_v4/orbit_cat_persian
"""
import argparse, math, os, sys
import cv2, spear
REPO = "/data/jzy/code/SPEAR"
sys.path.insert(0, os.path.join(REPO, "examples"))
from render_in_apartment import spawn_camera, read_frame
from render_in_gpurir_room import (
    configure_gpurir_instance, spawn_sky, spawn_directional_light, spawn_point_light,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--n-frames", type=int, default=72)
    p.add_argument("--per-frame-warmup", type=int, default=4)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--radius", type=float, default=200.0)
    p.add_argument("--height-cm", type=float, default=60.0)
    p.add_argument("--full-turns", type=float, default=1.0)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    bp_path = f"/Game/MyAssets/Audioset/Blueprints/gate_{args.tag}/BP_gate_{args.tag}.BP_gate_{args.tag}_C"
    anim_path = f"/Game/MyAssets/Audioset/Meshes/gate_{args.tag}/Walking"

    instance = configure_gpurir_instance(rpc_port=39002)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            for cls in ("APlayerStart", "ADefaultPawn", "ASpectatorPawn",
                        "AStaticMeshActor", "ASkeletalMeshActor",
                        "ABrush", "ADecalActor", "AInstancedFoliageActor"):
                try:
                    for a in game.unreal_service.find_actors_by_class(uclass=cls):
                        game.unreal_service.destroy_actor(actor=a)
                except Exception:
                    pass
            spawn_sky(game=game)
            spawn_directional_light(game=game, yaw_deg=-30.0, pitch_deg=-45.0, intensity_lux=8.0)
            spawn_point_light(game=game, x_cm=0.0, y_cm=0.0, z_cm=300.0,
                              intensity_lumens=4000.0, attenuation_cm=800.0)
            cam, comp = spawn_camera(game=game, width=args.width, height=args.height)
            bp = game.unreal_service.load_class(uclass="AActor", name=bp_path)
            actor = game.unreal_service.spawn_actor(
                uclass=bp, location={"X": 0.0, "Y": 0.0, "Z": 0.0},
                spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
            )
            actor.SetActorScale3D(NewScale3D={"X": args.scale, "Y": args.scale, "Z": args.scale})
            actor.K2_SetActorLocationAndRotation(
                NewLocation={"X": 0.0, "Y": 0.0, "Z": 0.0},
                NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": 90.0},
                bSweep=False, bTeleport=True,
            )
            actor.SetActorTickEnabled(bEnabled=True)
            smc = game.unreal_service.get_component_by_class(actor=actor, uclass="USkeletalMeshComponent")
            smc.SetComponentTickEnabled(bEnabled=True)
            anim = game.unreal_service.load_object(uclass="UAnimationAsset", name=anim_path)
            smc.PlayAnimation(NewAnimToPlay=anim, bLooping=True)
            try:
                smc.SetPlayRate(Rate=1.0)
            except Exception:
                pass
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(bPaused=False)
        with instance.end_frame():
            pass
        instance.step(num_frames=args.warmup)

        for i in range(args.n_frames):
            theta = 2.0 * math.pi * args.full_turns * i / args.n_frames
            cx = args.radius * math.cos(theta)
            cy = args.radius * math.sin(theta)
            cz = args.height_cm
            look_x, look_y, look_z = 0.0, 0.0, 30.0
            yaw = math.degrees(math.atan2(look_y - cy, look_x - cx))
            pitch = -math.degrees(math.atan2(cz - look_z, math.hypot(look_x - cx, look_y - cy)))
            instance.step(num_frames=args.per_frame_warmup)
            with instance.begin_frame():
                cam.K2_SetActorLocationAndRotation(
                    NewLocation={"X": cx, "Y": cy, "Z": cz},
                    NewRotation={"Roll": 0.0, "Pitch": pitch, "Yaw": yaw},
                    bSweep=False, bTeleport=True,
                )
            with instance.end_frame():
                img = read_frame(comp)
                cv2.imwrite(os.path.join(args.output_dir, f"frame_{i:04d}.png"), img)
        print(f"ORBIT_DONE {args.output_dir}", flush=True)
    finally:
        instance.close(force=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit editor script (orbit_animal.py lives under /tmp per project convention)**

```bash
cd /data/jzy/code/SPEAR
git add tools/import_gate_animal_editor.py
git commit -m "add tag-parameterized UE import script for gate check"
```

---

### Task 3: Gate Check A — Cat rig on cat_persian

**Files:**
- Consumes: `/data/jzy/code/SPEAR/tmp/hy3d_batch/cat_persian/hy3d_textured.obj` + `.jpg`, `Quaternius/Cat.glb`.
- Produces: `/tmp/gate_check_v4/cat_persian_side.mp4` + sheet PNG.

- [ ] **Step 1: Run the full gate check for cat_persian**

```bash
cd /data/jzy/code/SPEAR
DISPLAY=:99 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    bash tools/gate_check_animal.sh cat_persian 2>&1 | tee /tmp/gate_check_v4_cat_persian.log
```

Expected final line: `GATE_CHECK_DONE /tmp/gate_check_v4/cat_persian_side.mp4`

If it fails at "STAGE2A_FAILED" or a Blender error, do NOT retry blindly — investigate the log, then decide whether the Cat.glb source rig path works at all before continuing.

- [ ] **Step 2: Produce a 3x3 sheet from the video for quick review**

```bash
mkdir -p /tmp/gate_check_v4/sheets
ffmpeg -y -i /tmp/gate_check_v4/cat_persian_side.mp4 \
    -vf "select='not(mod(n\,8))',scale=320:240,tile=3x3" \
    -frames:v 1 /tmp/gate_check_v4/sheets/cat_persian_sheet.png
```

- [ ] **Step 3: Verify output exists and is non-trivial**

```bash
ls -la /tmp/gate_check_v4/cat_persian_side.mp4 /tmp/gate_check_v4/sheets/cat_persian_sheet.png
ffprobe -v error -show_entries stream=nb_read_packets -select_streams v \
    -of csv=p=0 -count_packets /tmp/gate_check_v4/cat_persian_side.mp4
```

Expected: mp4 file >100 KB, sheet PNG exists, packet count = 72.

- [ ] **Step 4: STOP — present the video to the user for approval**

Do not proceed to Task 4 without explicit user OK on the cat_persian video.

Present:
- `/tmp/gate_check_v4/cat_persian_side.mp4`
- `/tmp/gate_check_v4/sheets/cat_persian_sheet.png`

Ask: "Cat gate check — video attached. Does the cat's walking animation look natural (no leg-webbing, tail behaves like a cat's, no wrong-direction walk)? Proceed to horse gate check?"

---

### Task 4: Gate Check B — Wolf rig on horse

**Files:**
- Consumes: `/data/jzy/code/Hunyuan3D-2.1/outputs/audioset_assets/horse/horse_textured.{obj,jpg}`, `Quaternius/Wolf.glb`.
- Produces: `/tmp/gate_check_v4/horse_side.mp4` + sheet PNG.

- [ ] **Step 1: Run the full gate check for horse**

```bash
cd /data/jzy/code/SPEAR
DISPLAY=:99 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
    bash tools/gate_check_animal.sh horse 2>&1 | tee /tmp/gate_check_v4_horse.log
```

Expected final line: `GATE_CHECK_DONE /tmp/gate_check_v4/horse_side.mp4`

If Wolf.glb's proportions don't fit horse mesh well (Wolf is much shorter than horse), the `robust_skin_transfer`'s `--align-mode uniform` should still scale the horse to match the Wolf's bounding box for weight transfer. If the resulting walking animation looks warped (horse's proportions squashed), the fallback is `--align-mode nonuniform` in a re-run — but do NOT change it yet; try uniform first and let the user judge.

- [ ] **Step 2: Sheet + verify**

```bash
ffmpeg -y -i /tmp/gate_check_v4/horse_side.mp4 \
    -vf "select='not(mod(n\,8))',scale=320:240,tile=3x3" \
    -frames:v 1 /tmp/gate_check_v4/sheets/horse_sheet.png
ls -la /tmp/gate_check_v4/horse_side.mp4 /tmp/gate_check_v4/sheets/horse_sheet.png
```

- [ ] **Step 3: STOP — present the video to the user for approval**

Do not proceed to Task 5 without explicit user OK on horse.

Ask: "Wolf gate check — horse walking video attached. Does the horse's gait look plausible (four legs cycling, direction consistent, no severe warping from Wolf-to-horse proportion mismatch)? Proceed to batch of remaining 10?"

---

### Task 5: Generate chipmunk Hunyuan mesh (the only missing input)

**Files:**
- Consumes: `SPEAR/tools/flux_generate_reference.py`, `SPEAR/tools/batch_animal_pipeline.py`.
- Produces: `SPEAR/tmp/hy3d_batch/chipmunk/hy3d_textured.{obj,jpg,mtl}`.

- [ ] **Step 1: Ad-hoc single-species Flux+Hunyuan run for chipmunk**

Reuse the batch script's per-species subroutine by patching a minimal one-off species list. Create `/tmp/gen_chipmunk.py`:

```python
"""One-off chipmunk mesh generation using batch_animal_pipeline's building blocks.
Chipmunk fits SMALL_QUADRUPED prompt template (small, tail up, four legs visible).
"""
import sys
sys.path.insert(0, "/data/jzy/code/SPEAR")
from tools.batch_animal_pipeline import (
    PROMPT_TEMPLATE_SMALL_QUADRUPED, run_one_species,
)
run_one_species(
    species="chipmunk",
    breed="wild eastern",
    prompt_template=PROMPT_TEMPLATE_SMALL_QUADRUPED,
    tag="chipmunk",
    workdir_root="/data/jzy/code/SPEAR/tmp/hy3d_batch",
)
```

If `run_one_species` isn't already a public function, refactor `batch_animal_pipeline.py`:

```bash
grep -n "def " /data/jzy/code/SPEAR/tools/batch_animal_pipeline.py
```

Look for the body of the `for i, (sp, br, tmpl, tag) in enumerate(SPECIES_LIST)` loop at line ~229. Extract that body into `def run_one_species(species, breed, prompt_template, tag, workdir_root):` and call it from the loop. Keep the loop's control flow (progress prints, `continue` on failure) but delegate the actual work.

- [ ] **Step 2: Run it**

```bash
DISPLAY=:99 /data/jzy/miniconda3/envs/spear-env/bin/python /tmp/gen_chipmunk.py 2>&1 | tee /tmp/gen_chipmunk.log
```

Expected: final line reports `hy3d_textured.obj` written, and `ls /data/jzy/code/SPEAR/tmp/hy3d_batch/chipmunk/hy3d_textured.obj` returns the file.

- [ ] **Step 3: Commit refactor (if refactor happened)**

```bash
cd /data/jzy/code/SPEAR
git add tools/batch_animal_pipeline.py
git commit -m "extract run_one_species helper for ad-hoc single-species runs"
```

---

### Task 6: Batch process the remaining 10 quadrupeds

**Files:**
- Create: `SPEAR/tools/batch_gate_check_all.sh`
- Consumes: everything from Tasks 1-5.
- Produces: `/tmp/gate_check_v4/{tag}_side.mp4` for all 12 tags, plus sheet PNGs.

- [ ] **Step 1: Create the batch driver script**

Write `/data/jzy/code/SPEAR/tools/batch_gate_check_all.sh`:

```bash
#!/bin/bash
# Batch runner: process a list of species tags sequentially through
# gate_check_animal.sh. Sequential (not parallel) because:
#   - UE cook holds a global project lock (parallel cooks fight)
#   - RPC port 39002 is single-instance
#
# Usage:
#   ./batch_gate_check_all.sh cat_tabby chipmunk dog_golden ...
#   ./batch_gate_check_all.sh --all   # runs all 12 in RIG_MAP order
set -uo pipefail
SPEAR_DIR=/data/jzy/code/SPEAR
PY=/data/jzy/miniconda3/envs/spear-env/bin/python

if [ "${1:-}" = "--all" ]; then
    TAGS=$($PY -c "import sys; sys.path.insert(0,'$SPEAR_DIR/..'); from SPEAR.tools import species_rig_map as m; print(' '.join(m.RIG_MAP.keys()))")
else
    TAGS="$@"
fi

: > /tmp/gate_check_v4/batch_summary.txt
mkdir -p /tmp/gate_check_v4
for TAG in $TAGS; do
    echo ""
    echo "########################################"
    echo "# $TAG"
    echo "########################################"
    if [ -f "/tmp/gate_check_v4/${TAG}_side.mp4" ]; then
        echo "[skip] already produced /tmp/gate_check_v4/${TAG}_side.mp4"
        echo "$TAG OK (cached)" >> /tmp/gate_check_v4/batch_summary.txt
        continue
    fi
    if DISPLAY=:99 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
        bash $SPEAR_DIR/tools/gate_check_animal.sh "$TAG" 2>&1 \
        | tee "/tmp/gate_check_v4/${TAG}_run.log"; then
        # verify final artifact
        if [ -s "/tmp/gate_check_v4/${TAG}_side.mp4" ]; then
            echo "$TAG OK" >> /tmp/gate_check_v4/batch_summary.txt
        else
            echo "$TAG FAIL (no video)" >> /tmp/gate_check_v4/batch_summary.txt
        fi
    else
        echo "$TAG FAIL (script exit)" >> /tmp/gate_check_v4/batch_summary.txt
    fi
done
echo ""
echo "=== BATCH SUMMARY ==="
cat /tmp/gate_check_v4/batch_summary.txt
```

Make executable:
```bash
chmod +x /data/jzy/code/SPEAR/tools/batch_gate_check_all.sh
```

- [ ] **Step 2: Run for the 10 remaining tags (cat_persian + horse already done in gate check)**

```bash
cd /data/jzy/code/SPEAR
bash tools/batch_gate_check_all.sh \
    cat_tabby chipmunk \
    dog_golden dog_husky goat sheep pig \
    cattle_bovinae yak donkey_ass \
    2>&1 | tee /tmp/gate_check_v4/batch_10.log
```

Expected duration: ~30-60 min (rig-swap ~2 min + UE cook ~3-5 min + render ~2 min per animal × 10).

- [ ] **Step 3: Generate a combined sheet for user review**

```bash
mkdir -p /tmp/gate_check_v4/sheets
for TAG in cat_persian cat_tabby chipmunk dog_golden dog_husky goat sheep pig horse cattle_bovinae yak donkey_ass; do
    if [ -f /tmp/gate_check_v4/${TAG}_side.mp4 ] && [ ! -f /tmp/gate_check_v4/sheets/${TAG}_sheet.png ]; then
        ffmpeg -y -i /tmp/gate_check_v4/${TAG}_side.mp4 \
            -vf "select='not(mod(n\,8))',scale=240:180,tile=3x3" \
            -frames:v 1 /tmp/gate_check_v4/sheets/${TAG}_sheet.png 2>/dev/null
    fi
done
ls /tmp/gate_check_v4/sheets/
```

- [ ] **Step 4: Report summary to user**

Report the `batch_summary.txt` contents plus the list of produced videos and sheets. Do NOT claim all 12 are "good" — only claim "produced" (files exist). Quality assessment is the user's call.

---

## Rollback plan

If Cat gate check (Task 3) fails visually (spikes, misaligned bones):
- The Cat.glb path is not viable → fall back to using Dog.glb for the two cats + chipmunk. Update `species_rig_map.py::RIG_MAP` (set their `rig` value to `DOG_RIG`).
- Re-run Task 3 with cat_persian, then the batch.

If Wolf gate check (Task 4) fails visually (severe proportion warping on horse):
- Retry once with `--align-mode nonuniform` (edit `gate_check_animal.sh` to add the flag on the Blender call), verify horse alone.
- If still bad, fall back to Dog.glb for horse/cattle/yak/donkey. Update `RIG_MAP` accordingly.

If UE cook hangs >5 min without progress:
- Kill UnrealEditor processes: `pkill -9 -f UnrealEditor`
- The known stale-uassets bug — verify `rm -rf $MESH_DIR $BP_DIR` ran in `gate_check_animal.sh`.

## Self-review notes

- **Reverse-actions**: intentionally set to `no` everywhere. If the resulting animation walks backward, that means the source rig's Walking action already walks in `+Y_blender = -Y_ue` direction and the fix is in the orbit camera setup (which orbits 360°, so it's shown from all sides anyway — walking direction becomes ambiguous by design).
- **All 12 tasks share one gate_check_animal.sh** — DRY. Species-specific logic lives only in `species_rig_map.py`.
- **User verification checkpoints at Tasks 3 & 4** — mandatory, this is the whole point.
- **chipmunk mesh generation isolated to Task 5** — it's the only mesh not pre-generated; keeps the batch task (Task 6) focused on rigging + rendering.
