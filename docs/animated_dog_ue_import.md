# Import animated dog into SpearSim (Stage 2)

> **PRIMARY (headless, no GUI)** — this whole step is automated:
> ```bash
> /data/jzy/code/SPEAR/tools/build_animated_dog.sh
> ```
> That script runs `tools/import_animated_dog_editor.py` inside a UE 5.5
> Editor commandlet (`-run=pythonscript`), then cooks the new content into
> the pak. No GUI, no display, safe on a Linux server.
>
> **FALLBACK (manual GUI)** — the step-by-step below is kept only for when
> the commandlet path breaks (UE version bump, API rename, missing plugin).
> On this box the headless path is the tested route.
>
> Cross-refs: spec `docs/superpowers/specs/2026-07-04-animated-dog-gpurir-design.md`
> §Component B. Plan `docs/superpowers/plans/2026-07-04-animated-dog-gpurir.md` §Task 5.

## Prereqs

- UE 5.5 installed at `/data/UE_5.5`.
- SpearSim project at `/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/SpearSim.uproject`.
- `tmp/animated_dog/Dog_textured.glb` exists (T2 output, ~2 MB).
- T1 (`tools/probe_skeletal_playanimation.py`) printed `PROBE_OK` on this machine
  — if instead it printed `PROBE_FAILED_*`, additionally follow the "F5 fallback"
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
   If naming differs (e.g. the mesh is called `Cylinder`, skeleton
   `Cylinder_Skeleton`), right-click → Rename to the exact names above —
   subsequent tasks look for these paths.
6. In the Content Browser, navigate to `/Game/MyAssets/Audioset/Blueprints/`.
   Right-click → New Folder → name it `animated_dog`.
7. Right-click `SKM_dog` → Asset Actions → Create Blueprint Using This... → base
   class `Actor` → save as `BP_dog_animated` inside the new
   `Blueprints/animated_dog/` folder.
8. Open `BP_dog_animated` in the Blueprint Editor.
9. In the Components panel: verify a `USkeletalMeshComponent` was added
   (called `SkeletalMesh` by default). In its Details panel:
   - **Skeletal Mesh Asset**: `SKM_dog`.
   - **Animation Mode**: `Use Animation Asset`.
   - **Anim to Play**: `anim_dog_Walking`.
   - **Looping**: checked.
10. Click **Compile**. Click **Save**. Close the Blueprint Editor.
11. File → **Save All**. Close the UE Editor.

## Verification (before running the cook script)

```bash
find /data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Meshes/animated_dog \
     /data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Blueprints/animated_dog \
     -type f -iname "*.uasset" 2>&1 | sort
```
Expected: 6 files (5 in Meshes + 1 BP in Blueprints).

## F5 fallback — only if T1 probe printed `PROBE_FAILED_*`

(Not needed on this machine — T1 printed `PROBE_OK`.) If a future migration
breaks raw `PlayAnimation` RPC, add a Blueprint-callable helper before step 11:

- In `BP_dog_animated`, add a new custom Event (right-click empty graph area →
  Add Custom Event) named `SetAndPlayWalking`.
- In its body: drag `SkeletalMesh` component → call `Set Animation Mode`
  (pass `AnimationSingleNode`) → drag → call `Set Animation` (pass a hard
  reference to `anim_dog_Walking`) → drag → call `Play` (pass `Looping=true`).
- Compile + Save.
- Task 7's Python side then calls this event via `actor.SetAndPlayWalking()`
  instead of the three raw component-level calls.
