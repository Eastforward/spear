# Realistic-Room + Imported-Asset + Anim-Character Render Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task in this session. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the SPEAR pipeline can render a **photorealistic-looking interior** (using the real `apartment_0000` PBR materials `MI_Walls`/`MI_Floor`) with **imported Hunyuan3D meshes placed at right scale/position**, and **skeletal-mesh characters animated in-place** — yielding a 30-second review video per asset for the user to skim when they wake.

**Architecture:**
- **Editor-time**: re-run `my_build_room.py` (already updated) so the BP_Room* actors reference the SPEAR apartment MIs; one pass yields 6 BP assets and 1 light.
- **Game-time**: a small orchestrator spawns the room + the asset + the camera inside a SpearSim session; reads pixels each frame into PNG; on completion, `ffmpeg` joins PNGs to mp4.
- **Vision-QA**: a multimodal vision call I make myself (not an HTTP API) inspects sampled frames against a fix-list and either passes, or applies one of three preset parameter nudges (re-grade camera elevation → orbit distance → material swap) up to **3 retries** per asset before declaring it `tmp/failed/`.
- **Single autonomous loop**: tasks 5–10 run in this same session with no user interruption; videos are pushed to `/data/jzy/code/SPEAR/tmp/render_<NAME>/turntable.mp4` and a `tmp/QA_log.md` records every decision.

**Tech Stack:**
- Unreal Engine 5.5 (binary at `/data/UE_5.5`)
- SpearSim game shell (`/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh`)
- spear-sim Python package (installed in `/data/jzy/miniconda3/envs/thu/`)
- `tools/run_editor_script.py` (editor commandlet)
- `ffmpeg` 4.4 for PNG→mp4 muxing
- Xvfb :99 (headless rendering server already started; check `ps`)
- NVIDIA RTX 4090 D + Vulkan ICD at `/etc/vulkan/icd.d/nvidia_icd.json`

---

## Global Constraints

- **Never use sudo.** All UE/Python/file ops run as `jzy`.
- **Editor commandlets**: scripts invoked with absolute `--script "$(pwd)/examples/foo.py"`; relative paths fail silently because UE resolves against the editor binary dir, not cwd. Run via `--launch-mode full --render-offscreen`.
- **Game sessions**: run under `DISPLAY=:99` (Xvfb). Set `VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`. Mandatory `instance.step(num_frames=30)` warm-up before first read_pixels() (auto-exposure settles).
- **Numeric accuracy**: SPEAR world units = cm. The 5.2 × 4.4 × 2.8 m room is **520 × 440 × 280 cm**. FLOOR_Z = 1000.0 lifts it well above the minimal_default map so no other geometry intrudes.
- **NEVER delete cooked apartment assets** under `/Game/SPEAR/Scenes/apartment_0000/` — they're read-only references.
- **All created BPs go under `/Game/MyAssets/...`** and are auto-deleted/recreated by their build script (already in code).
- **Vision QA**: use `mcp__MiniMax__understand_image` with `image_source` set to the absolute frame path. Use Chinese OR English prompts — both work.
- **Per-asset retry budget = 3.** Failure on the 3rd retry ⇒ write `tmp/failed/<NAME>.json` and move to the next asset.

---

## File Structure

| Path | Modified | Responsibility |
|---|---|---|
| `examples/my_build_room.py` | yes (already done in this session) | Builds 7 BP assets. Imports `MI_Walls` (wall+ceiling) and `MI_Floor` (floor). Single Light. |
| `examples/my_import_asset.py` | read-only | Imports a Hunyuan3D `.glb` and writes `tmp/asset_meta/<NAME>.json`. Already runs for `Clock`. |
| `examples/preview_room.py` | yes (already done — created in this session) | Empty-room 6-angle snapshot. Used by Task 5. |
| `examples/render_asset_in_room.py` | yes (this plan Task 7) | Per-asset turntable using MI_Walls/MI_Floor room. Reads `tmp/asset_meta/<NAME>.json`. ffmpeg → mp4. |
| `examples/spawn_animated_character.py` | yes (this plan Task 9) | Spawns `SKM_Manny` at room center, plays `MM_Walk_Fwd` on loop. Records 100 frames. |
| `tmp/preview_room/{wallN,wallS,...}.png` | produced | T1 verification artifacts. |
| `tmp/render_<NAME>/turntable.mp4` | produced | Final per-asset render. |
| `tmp/QA_log.md` | produced | Append-only log of every vision-decision and parameter change. |
| `tmp/failed/<NAME>.json` | produced | Failure report when 3 retries exhausted. |

---

## Task 0 — Confirm runtime environment is intact (1 min)

**Files:** none.

**Interfaces:** consumes nothing; produces a green light.

- [ ] **Step 1: Smoke-check that nothing crashed since last attempt**

Run:
```bash
ps -ef | grep -E "Xvfb|SpearSim|UnrealEditor" | grep -v grep
```

Expected: Xvfb on `:99` present, no leftover SpearSim/UnrealEditor (any zombies need `kill -9` first).

- [ ] **Step 2: Verify Vulkan + GPU are reachable**

```bash
ls -l /etc/vulkan/icd.d/nvidia_icd.json && \
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
```

Expected: `nvidia_icd.json` exists; GPU 0 RTX 4090 D idle (<5% util); GPU 1 may be busy (other tenant).

- [ ] **Step 3: Verify the conda env still has `spear`**

```bash
/data/jzy/miniconda3/envs/thu/bin/python -c "import spear; print(spear.__file__)"
```

Expected: prints a path under `/data/jzy/miniconda3/envs/thu/lib/python3.12/site-packages/spear/...`. If `ModuleNotFoundError`, run `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/thu/bin/pip install -e python` once.

- [ ] **Step 4: If any of the above fails, fix it (no advance)**.

---

## Task 1 — Verify BP_Room* assets were built with MI materials (1 min)

**Files:** none modified. Read-only check.

**Interfaces:** consumes `/Game/MyAssets/Room/BP_Room*` uassets; produces textual confirmation.

- [ ] **Step 1: List the BP files**

```bash
ls -la /data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Content/MyAssets/Room/
```

Expected: 7 files, mtime today, total size ~ 190 KB (≈ 4 BP walls + floor + ceiling + light).

- [ ] **Step 2: If BP files are missing or old** (older than today's first session)

If mtime is stale OR files are missing → re-run:
```bash
cd /data/jzy/code/SPEAR && \
/data/jzy/miniconda3/envs/thu/bin/python \
  tools/run_editor_script.py --unreal-engine-dir /data/UE_5.5 \
  --script "$(pwd)/examples/my_build_room.py" \
  --launch-mode full --render-offscreen 2>&1 | tail -30
```

Expected: the last 10 lines contain `AssetCheck: /Game/MyAssets/Room/BP_RoomFloor 正在验证资产` ... `BP_RoomLight` and finally `[SPEAR | run_editor_script.py:0069] Done.`. Editor exit ~30s.

- [ ] **Step 3: Append a one-liner to `tmp/QA_log.md` (creates file on first run)**

```bash
mkdir -p /data/jzy/code/SPEAR/tmp && \
echo "$(date -Is) Task1 BP_Room assets present: $(ls /data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Content/MyAssets/Room/ | tr '\n' ' ')" \
  >> /data/jzy/code/SPEAR/tmp/QA_log.md
```

---

## Task 2 — Re-run empty-room T1 preview (5 min wall-clock)

**Files:** `examples/preview_room.py` (already exists). Output: `tmp/preview_room/*.png`.

**Interfaces:** consumes BP_Room*; produces 6 PNG files.

- [ ] **Step 1: Make sure Xvfb is up**

```bash
pgrep -a Xvfb || (XAUTHORITY= Xvfb :99 -screen 0 1280x720x24 +extension RANDR -ac &) && sleep 1 && pgrep -a Xvfb
```

Expected: at least one Xvfb PID listed.

- [ ] **Step 2: Run the preview headless**

```bash
cd /data/jzy/code/SPEAR && \
DISPLAY=:99 \
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
/data/jzy/miniconda3/envs/thu/bin/python examples/preview_room.py
```

Expected stdout ends with: `PREVIEW DONE (6 frames) -> /data/jzy/code/SPEAR/tmp/preview_room`. Wall-clock: ~90 s. (Editor cold start + scene init + 6 frames + warmup.)

- [ ] **Step 3: Confirm 6 PNGs exist and have non-trivial size**

```bash
ls -la /data/jzy/code/SPEAR/tmp/preview_room/
```

Expected: 6 PNGs (`wallN.png wallS.png wallE.png wallW.png ceiling.png floor.png`), each ≥ 200 KB.

- [ ] **Step 4: Append to log**

```bash
echo "$(date -Is) Task2 T1 preview rendered: $(ls /data/jzy/code/SPEAR/tmp/preview_room/ | wc -l) frames" \
  >> /data/jzy/code/SPEAR/tmp/QA_log.md
```

---

## Task 3 — Vision-QA the T1 frames (1 min wall-clock)

**Files:** none modified. Read PNG headers via vision tool.

**Interfaces:** consumes 6 PNG paths; produces a verdict per frame and one Pass/Fail overall.

- [ ] **Step 1: For each preview frame, run a vision prompt**

For every file in `tmp/preview_room/`:
- Tool: `mcp__MiniMax__understand_image`
- Prompt (Chinese works, paste verbatim):
  ```
  这是一张空房间的渲染图(无任何家具、无人)。判断并用 JSON 回答:
  1) "material_ok": true/false  —— 墙的 PBR 材质看起来是否像真实公寓墙(不是占位符/马赛克/纯灰/无纹理)
  2) "lighting_ok": true/false —— 是否有合理光照(不是全黑/全白/纯色调异常)
  3) "geometry_ok": true/false —— 墙面是否平整对齐、有没有错位/接缝/穿透/翻转法线
  4) "issues": 中文一句话总结最差的问题
  ```

- [ ] **Step 2: Aggregate verdicts**

If `material_ok && lighting_ok && geometry_ok` for **at least 4 of 6** frames ⇒ **PASS**. Otherwise ⇒ **FAIL** and proceed to Task 4.

- [ ] **Step 3: Append verdict to `tmp/QA_log.md`**

```bash
{
  echo "$(date -Is) Task3 T1 QA:"
  for f in /data/jzy/code/SPEAR/tmp/preview_room/*.png; do
    bn="$(basename "$f")"
    echo "  $bn: PASS-or-FAIL one-line note"
  done
} >> /data/jzy/code/SPEAR/tmp/QA_log.md
```

---

## Task 4 — If T1 failed, swap wall material candidates and re-render (5 min wall-clock)

**Files:** `examples/preview_room.py` no edit; we use **environment override** instead.

**Interfaces:** consumes the same BP_Room*; produces 6 new PNGs at `tmp/preview_room_v2/`.

- [ ] **Step 1: Copy preview script with substituted material names**

```bash
sed 's|MI_Walls.MI_Walls|MI_Door.MI_Door|g; s|MI_Floor.MI_Floor|MI_Carpet.MI_Carpet|g' \
  /data/jzy/code/SPEAR/examples/preview_room.py \
  > /tmp/preview_room_v2.py
mkdir -p /tmp/v2
sed -i 's|/data/jzy/code/SPEAR/tmp/preview_room|/tmp/v2|' /tmp/preview_room_v2.py
```

- [ ] **Step 2: Run the v2 preview**

```bash
cd /data/jzy/code/SPEAR && \
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
/data/jzy/miniconda3/envs/thu/bin/python /tmp/preview_room_v2.py
```

Expected: prints `PREVIEW DONE (6 frames) -> /tmp/v2`.

- [ ] **Step 3: Re-run vision-QA on `/tmp/v2/*.png`** (same prompt as Task 3 Step 1).

- [ ] **Step 4: Append verdict**

```
$(date -Is) Task4 v2 (MI_Door+MI_Carpet) QA: <verdict> <issue>
```

- [ ] **Step 5: If still failing** — escalate by editing `my_build_room.py`:

Change `WALL_MAT` to `MI_Props.MI_Props`, `FLOOR_MAT` keeps `MI_Floor`. Re-run editor commandlet (Task 1 Step 2). Then re-run preview (Task 2 Step 2). Then re-QA.

If still failing after this third attempt ⇒ write `tmp/failed/T1_materials.json` and stop the plan (user notification required). See Task 11.

---

## Task 5 — Lock in good turntable camera parameters (5 min wall-clock)

**Files:** `examples/render_asset_in_room.py` (will create here).

**Interfaces:** consumes `tmp/asset_meta/<NAME>.json`; produces a video at `tmp/render_<NAME>/turntable.mp4`.

- [ ] **Step 1: Settle on initial camera params**

Based on T1 passing, codify these as starting parameters:

```python
R        = 3.0 * TARGET         # was 2.5 — slightly farther to clear room walls
CAM_Z_OFF = 40.0                # 40 cm above asset center, NOT 0.20*R
PITCH    = -math.degrees(math.atan2(CAM_Z_OFF, R))
N_FRAMES = 36
MAP      = "/Game/StarterContent/Maps/Minimal_Default"
FLOOR_Z  = 1000.0
ROOM_LIGHT_LUMENS = 800.0
OUT      = f"/data/jzy/code/SPEAR/tmp/render_{NAME}"
```

- [ ] **Step 2: Create `examples/render_asset_in_room.py` from `my_render_in_room.py` skeleton**

The file:
- Reads `tmp/asset_meta/<NAME>.json` for `ext`, `bmin_z`, `height`, `TARGET`.
- Spawns the 7 BP_Room actors + the BP_<NAME> actor.
- Centres the asset, scales to `TARGET/ext`, grounds at `Z = FLOOR_Z - bmin_z * scale`.
- Camera orbits at `(cx + R cosθ, cy + R sinθ, cz + CAM_Z_OFF)` with the PITCH above; yaw points back at the asset center.
- 36 frames, each frame `instance.step(num_frames=1)` then `read_pixels()`.
- Writes PNGs + invokes `ffmpeg` once to produce `turntable.mp4`.

(Code placeholder is identical to `examples/my_render_in_room.py` except for the constants above and the OPTIONAL `instance.close(force=True)` followed by ffmpeg.)

- [ ] **Step 3: Run on Clock as the first asset**

```bash
cd /data/jzy/code/SPEAR && \
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
/data/jzy/miniconda3/envs/thu/bin/python \
  examples/render_asset_in_room.py /Game/MyAssets/Blueprints/BP_Clock.BP_Clock_C Clock
```

Expected stdout ends with: `VIDEO DONE /data/jzy/code/SPEAR/tmp/render_Clock/turntable.mp4`. Wall-clock: ~2 min.

- [ ] **Step 4: Append log**

```
$(date -Is) Task5 Clock rendered with params R=3.0T cam_z=40cm pitch=-22deg
```

---

## Task 6 — Vision-QA the Clock render with auto-iterate budget (3 retries, ~6 min total)

**Files:** `examples/render_asset_in_room.py` no edit — a small wrapper script `/tmp/iterate_clock.sh` does the iteration.

- [ ] **Step 1: For each retry `k = 0,1,2`**

Read 4 sample frames: `{floor((N_FRAMES/4)*{0,1,2,3})}`. Run vision prompt:

```
这张是 SPEAR Unreal Engine 渲染的 360° 转盘视频的第 {i}/36 帧。房屋 5.2×4.4×2.8 米,中央放着一只钟表(BP_Clock)。
请用 JSON 回答:
1) "object_in_frame": true/false  —— 钟表是否在画面内可见(不被墙/天花板/桌面遮挡)
2) "object_size_ok": true/false   —— 钟表在画面中占的纵向像素比例是否 ≥ 30%
3) "object_legibility": true/false —— 钟表细节(指针/外壳)是否清晰可辨(不是一团糊/全黑/全噪)
4) "material_legibility": true/false —— 墙/地 PBR 材质是否像真实公寓(不是占位符/纯色/马赛克)
5) "issues": 中文一句话最大问题
6) "retry_action": 在 ["raise_cam", "raise_cam_then_far", "raise_cam_then_far_then_rot", "give_up"] 里挑一个
   - raise_cam: 钟太低/下半截被地板切 → 把 camera.z 再 +20 cm 重跑
   - raise_cam_then_far: 钟太小 → 把 camera.z +20 cm AND R ×= 1.2 重跑
   - raise_cam_then_far_then_rot: 钟在不同角度消失 → 把相机抬高同时绕到斜俯视角(pitch -25°)
   - give_up: 三次都不行
```

- [ ] **Step 2: Act on `retry_action`**

- `raise_cam`: bump `CAM_Z_OFF += 20`, re-run Task 5 Step 3.
- `raise_cam_then_far`: bump `CAM_Z_OFF += 20`, `R *= 1.2`, re-run.
- `raise_cam_then_far_then_rot`: bump `CAM_Z_OFF += 20`, `R *= 1.2`, set `PITCH = -25°`, re-run.
- `give_up` OR already on retry k=2 ⇒ write `tmp/failed/Clock.json` and abort this asset.

- [ ] **Step 3: Append every decision to log**

```bash
echo "$(date -Is) Task6 Clock retry=$k verdict=$(jq '{action: .retry_action, issues: .issues}' <<<"$JSON")" \
  >> /data/jzy/code/SPEAR/tmp/QA_log.md
```

- [ ] **Step 4: On PASS** with no retry needed (k=0 and at least 3 of 4 sampled frames PASS), append "Clock OK k=0" and exit Task 6.

- [ ] **Step 5: On final PASS** (any k), the deliverable is `tmp/render_Clock/turntable.mp4` + a one-line log entry.

---

## Task 7 — Wrap render_asset_in_room.py into a loop over the catalog (10 min wall-clock if first asset ok)

**Files:** new `examples/batch_render_in_room.py`.

- [ ] **Step 1: Discover all meta JSONs**

```python
metas = sorted(glob.glob("/data/jzy/code/SPEAR/tmp/asset_meta/*.json"))
```

- [ ] **Step 2: For each meta, follow the same flow as Task 5+6** (skip if `tmp/render_<NAME>/turntable.mp4` already exists & mtime < 1 day, AND not in `tmp/failed/`).

- [ ] **Step 3: Run as a single batch in the same SpearSim session** (don't quit-and-relaunch per asset; that's slow). Use one `instance.begin_frame/end_frame` per asset placement, **single** `instance.close(force=True)` at the end.

- [ ] **Step 4: At end of batch, append summary**

```
$(date -Is) Batch done: rendered $N, failed $M, skipped $S — $(cat tmp/failed/*.json 2>/dev/null | wc -l) failed entries
```

- [ ] **Step 5: Stop here**, **even mid-batch**, if we hit 10 sequential failed assets in a row. The pipeline likely has a deeper problem worth a user hand-off.

---

## Task 8 — Optional: Spawn an animated character (Manny walk) (5 min wall-clock)

> Skip until Tasks 5–7 succeed. This is the "see whether UE skeletal animation plays through SPEAR" feasibility test.

**Files:** new `examples/spawn_animated_character.py`.

- [ ] **Step 1: Locate the assets**

- Skeletal mesh: `/Game/Characters/Mannequins/Meshes/SKM_Manny`
- Animation sequence: `/Game/Characters/Mannequins/Animations/Manny/MM_Walk_Fwd`
- Animation BP: `/Game/Characters/Mannequins/Animations/ABP_Manny`

- [ ] **Step 2: Spawn Manny at room center, scale to ~1.8 m (180 cm)**

```python
sk_uclass = game.unreal_service.load_class(uclass="AActor", name="/Game/Characters/Mannequins/Meshes/SKM_Manny")
m = game.unreal_service.spawn_actor(uclass=sk_uclass, location={"X":0,"Y":0,"Z":FLOOR_Z})
m.SetActorScale3D(NewScale3D={"X":1.0,"Y":1.0,"Z":1.0})
```

- [ ] **Step 3: Play `MM_Walk_Fwd` on loop**

```python
anim = game.unreal_service.load_class(uclass="UAnimSequence", name="/Game/Characters/Mannequins/Animations/Manny/MM_Walk_Fwd")
skel_comp = game.unreal_service.get_component_by_class(actor=m, uclass="USkeletalMeshComponent")
skel_comp.SetAnimationMode(0)  # 0 = AnimationBlueprint (alternative)
skel_comp.PlayAnimation(AnimSequence=anim, Looping=True)
```

If `PlayAnimation` doesn't exist on the typed proxy, use `call_async.PlayAnimation` (per SPEAR agents.md, async avoids frame sync) and call `.get()` after `step`.

- [ ] **Step 4: Render 100 frames over ~10 seconds, save to `tmp/manny_walk/frame_{i:04d}.png`, then ffmpeg → `mp4`**

- [ ] **Step 5: Vision-QA "is the figure actually moving across frames"**

Sample 4 frames at i ∈ {10, 40, 70, 99}. Compare positions of the limbs' bounding box centre across frames. If center moves < 1 cm between frames ⇒ animation is not playing ⇒ mark `tmp/failed/manny_anim.json` and stop.

- [ ] **Step 6: Append to log**

---

## Task 9 — Final wrap-up (1 min)

- [ ] **Step 1: Compile `tmp/QA_log.md` summary**

```
$(date -Is) Final summary: rendered=$R, failed=$F, queue remaining=$Q
cat tmp/failed/*.json | jq -s 'map({name, last_reason})'
```

- [ ] **Step 2: Ensure 1 file per rendered asset** at `tmp/render_<NAME>/turntable.mp4` and that all `tmp/preview_room/*.png` stay for sanity-check.

- [ ] **Step 3: Produce the user-facing digest**

Append to `tmp/QA_log.md` a final block "USER DIGEST" listing:
- which assets rendered (mp4 paths)
- which failed (json paths)
- one-line summary of failures

The user will read `tmp/QA_log.md` + sample mp4s upon waking.

---

## Self-Review

**Spec coverage** ✅ — goal, materials, room params, asset choice, camera, retry budget, QA, animated character all have tasks.

**Placeholder scan:** — Task 5 Step 2 says "(Code placeholder is identical to ...)" — replace with full code in the script before running (see `examples/render_asset_in_room.py` sketch in the conversation log).

**Type consistency:** the script's `game.unreal_service.load_class(uclass=...)` matches SPEAR API conventions.

**Gaps:**
- Task 6 vision-prompt JSON keys — the multimodal tool returns free text, not strict JSON. If it doesn't JSON-format cleanly, take the LAST sentence starting with 1)/2)/3)/4)/5)/6) and parse those values.
- Task 7 batch loop should EXCLUDE `Clock.json` (capital C, the legacy entry) when `clock.json` (lowercase) exists, since they're the same asset. Detect by name lowering with first occurrence of each.

---

## Execution Handoff

Plan complete and saved to `/data/jzy/code/SPEAR/docs/superpowers/plans/2026-07-03-realistic-room-render.md`. **In this session (user is offline), I execute inline with batched checkpoints using `superpowers:executing-plans`.**

I'll begin with Task 0 immediately and only interrupt the user with the final `tmp/QA_log.md` digest + a sample mp4 list when the run is finished.
