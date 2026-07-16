# GPURIR-Shaped Room with Apartment Textures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a parameterized GPURIR-shaped shoebox room in SPEAR — Cube-based floor/walls/ceiling with apartment_0000's `MI_Floor` and `MI_Walls` materials, one wall carrying a real floor-to-ceiling window cut into 4 sub-mesh pieces, lit by a Directional Light + BP_LightStudio sky sphere from the window direction — then place the imported `BP_dog` at the GPURIR source position `(mic + 1.7 m along +Y)` and render one 360° turntable video showcasing the room.

**Architecture:** New standalone script `examples/render_in_gpurir_room.py` parallel to `render_in_apartment.py`. Loads `/Engine/Maps/Entry` as the empty container map, then constructs the entire scene at runtime: (1) pure-Python helpers compute Cube mesh transforms for the 6-surface shoebox + 4-piece window wall + window frame trim; (2) `render_in_apartment.py` helpers (`compute_asset_fit`, `sample_ground_z`, `get_actor_bounds_bottom_z`, `spawn_camera`, `read_frame`, `build_solo_checklist`, `write_checklist`, `animal_bp_path`, `animal_meta_path`) are imported for reuse; (3) new helpers write a superset checklist that records room dims, mic pos, source pos, window bounds, light intensity, and material paths. The script uses a fresh test file `tests/test_render_in_gpurir_room.py` so it can't break the 28 existing apartment tests.

**Tech Stack:** Python 3.11 (`spear-env`), SPEAR RPC (`spear.Instance`), Unreal Engine 5.5 SpearSim, OpenCV, ffmpeg, matplotlib (for layout PNG), unittest.

## Global Constraints

- Python interpreter: `/data/jzy/miniconda3/envs/spear-env/bin/python` (NEVER `thu` — missing `spear_ext`)
- Environment prefix: `DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`
- All scripts run from `/data/jzy/code/SPEAR` cwd
- Empty container map: `/Engine/Maps/Entry` (verified in pak via `strings … | grep /Engine/Maps/Entry_BuiltData`)
- Structural material paths (verified in pak):
  - Floor: `/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Floor.MI_Floor`
  - Walls (+ ceiling default): `/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Walls.MI_Walls`
- Cube mesh: `/Engine/BasicShapes/Cube.Cube` (1 m × 1 m × 1 m unit cube — `SetActorScale3D` in meters, then the actor sets an X/Y/Z size in cm)
- Sky + light bundle: `/Engine/EngineSky/BP_LightStudio.BP_LightStudio_C` (BP that contains skysphere, directional light, exponential height fog, skylight — verified in pak)
- Default room size (CLI default): `5.2 m × 4.4 m × 2.8 m` (matches v77 `my_build_room.py` and is inside the GPURIR sampling range `[4, 8] × [4, 8] × [2.4, 3.5]`)
- Wall thickness: `0.1 m` (matches `my_build_room.py`)
- Window: `2.0 m wide × 2.4 m tall`, on the Y-max (`y = room_y`) wall, x-centered at `room_x/2`, bottom edge at `z = 0.2 m` (defaults; all CLI-parameterizable)
- Coordinate frame: X, Y, Z in **centimeters** in the UE actor API. `room_size` args are given in **meters** and converted inside the script (matches v77/GPURIR convention).
- Floor top-face lives at `z = 0 cm` (GPURIR contract: mesh origin z=0 is the ground plane). We STILL trace ground for sanity — checklist `ground_z_cm` must be within `tolerance_cm` of 0.
- Mic position: `(room_x/2, room_y/2, 1.2 m)` — GPURIR canonical, `compute_mic_pos()` in v77 `gen_rir_multiscene_v77.py`
- Source (dog) position: `(mic_x, mic_y + 1.7 m, 0)` — mic + 1.7 m along +Y towards the window wall
- Video spec: `1280×720 / 12 fps / 36 frames / 3 s`, orbit radius **`200 cm`** around the dog, camera z-offset +40 cm, pitch computed from the offset (matches Task-4-cat/dog conventions)
- All 4 imported animals BPs (`cat/dog/goose/yak`) are packaged after the Round-1 pak rebuild; this plan only renders `dog`
- Checklist: superset of Round 1's `build_solo_checklist` — adds `room_size_m`, `mic_pos_cm`, `source_pos_cm`, `window_bounds_cm`, `directional_light_intensity_lux`, `wall_material`, `floor_material`, and a `human_review` block listing the 4 human-visual items
- Output: `/data/jzy/code/SPEAR/tmp/render_gpurir_room/{run_name}/{turntable.mp4, frame_0000.png, checklist.json, layout.png}`
- TDD: pure helpers get failing test → impl → passing test → run full suite (no regressions)
- SPEAR is NOT a git repo (`Is a git repository: false`); skip `git commit` steps and verify with `ls` / `grep`
- Follow `docs/agents.style_guide.md`: keyword-only API calls to UE, ASCII source, no bare `except:`
- Do NOT modify `render_in_apartment.py` behavior — only import from it. If a helper must be extracted, put it in `render_in_apartment.py` unchanged and import it from there (preserves the 28 existing tests)

---

## File Structure

- **Create:** `/data/jzy/code/SPEAR/HANDOFF_GPURIR_ROOM.md` — Round-2 spec/handoff, follows Round-1 doc style
- **Create:** `/data/jzy/code/SPEAR/examples/render_in_gpurir_room.py` — new standalone renderer
- **Create:** `/data/jzy/code/SPEAR/tests/test_render_in_gpurir_room.py` — new test module for pure helpers (does NOT touch Round-1 tests)
- **Read-only reference:**
  - `/data/jzy/code/SPEAR/examples/render_in_apartment.py:1-462` — helpers we import (`compute_asset_fit`, `sample_ground_z`, `get_actor_bounds_bottom_z`, `spawn_camera`, `read_frame`, `build_solo_checklist`, `write_checklist`, `animal_bp_path`, `animal_meta_path`, `clean_frames`, `configure_instance` — but we override the map)
  - `/data/jzy/code/SPEAR/examples/my_build_room.py` — reference for old Cube-based room approach; DO NOT copy verbatim (it's editor-commandlet style, not RPC)
  - `/data/jzy/code/Spatial/v77_4ch_S2L/data_gen/gen_rir_multiscene_v77.py:56-140` — canonical GPURIR mic/source formulas (`compute_mic_pos`, `_source_distance`)
  - `/data/jzy/code/SPEAR/HANDOFF_ANIMALS_APARTMENT.md` — Round-1 handoff for style reference

---

## Task 1: Write the Round-2 handoff document

**Files:**
- Create: `/data/jzy/code/SPEAR/HANDOFF_GPURIR_ROOM.md`

**Interfaces:**
- Consumes: nothing
- Produces: the frozen decision record referenced by every later task (`see HANDOFF §N`)

- [ ] **Step 1: Write the handoff document**

Write `/data/jzy/code/SPEAR/HANDOFF_GPURIR_ROOM.md` with these H2 sections (verbatim structure):

1. **一句话现状** — "Round 1 verified 4 Hunyuan3D animals in apartment_0000. Round 2 builds a GPURIR-parameter-aligned shoebox room (default 5.2 × 4.4 × 2.8 m) with apartment_0000 materials (`MI_Floor` + `MI_Walls`), a real floor-to-ceiling window cut into the +Y wall, a Directional Light + BP_LightStudio sky sphere shining through the window, and places `BP_dog` at the GPURIR source position `(mic + 1.7 m along +Y)`. Output: one 360° turntable video."
2. **环境** — reuse the "Python 环境 + Xvfb + Vulkan" block from `HANDOFF_ANIMALS_APARTMENT.md` §2 verbatim.
3. **本轮 14 项决策 (Q1-Q14 from grill)** — reproduce the Q1-Q14 table exactly as in the grill wrap-up (room parameterized, MI_Floor+MI_Walls+window+sky, Y-max wall, 2×2.4 m floor-to-ceiling window, 4-piece cube wall, DirLight + BP_LightStudio, dog only, source at mic+1.7 m +Y, floor at z=0 with ground-trace safety, 1280×720/12fps/36f/3s radius=200cm, checklist full, new script + new tests, `/Engine/Maps/Entry`, full TDD).
4. **文件地图** — list the 3 new files + the plan file `docs/superpowers/plans/2026-07-03-gpurir-room-with-apartment-textures.md`.
5. **执行顺序** — TDD Tasks 2-7 → BREAKPOINT (Task 8: render dog) → user review → wrap-up (Task 9).
6. **Checklist 定义** — the full checklist schema from Task 6 Step 2 (copy-paste): 8 auto fields on top of Round-1's solo checklist + 4 human-review items.
7. **下一轮计划** — after this round approves, next: parameterize room-size CLI for multi-size mass render, then Puppeteer/animation integration; NOT part of this plan.

- [ ] **Step 2: Sanity-check the doc compiles**

Run: `grep -c "^##" /data/jzy/code/SPEAR/HANDOFF_GPURIR_ROOM.md`
Expected: `>= 7`.

- [ ] **Step 3: Verify presence** (SPEAR is not git — no commit)

Run: `ls -la /data/jzy/code/SPEAR/HANDOFF_GPURIR_ROOM.md`
Expected: file exists, non-zero size.

---

## Task 2: Shoebox room layout helper (pure)

**Files:**
- Create: `/data/jzy/code/SPEAR/examples/render_in_gpurir_room.py`
- Create: `/data/jzy/code/SPEAR/tests/test_render_in_gpurir_room.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `M2CM = 100.0` module constant
  - `WALL_THICKNESS_M = 0.1` module constant
  - `compute_shoebox_room_layout(*, room_size_m, wall_thickness_m=WALL_THICKNESS_M) -> list[dict]`
    - `room_size_m = (x_m, y_m, z_m)` tuple/list
    - Returns list of 6 pieces (order: `floor`, `ceiling`, `wall_x0`, `wall_x1`, `wall_y0`, `wall_y1`), each `{"name": str, "location_cm": (x, y, z), "scale": (sx, sy, sz)}` with `scale` being the ratio to feed to `SetActorScale3D` (1 m unit Cube → scale × 100 cm actual size). Sizes are chosen so:
      - Floor top-face is exactly at `z = 0 cm`, bottom at `-thickness_cm`
      - Ceiling bottom-face is exactly at `z = room_z_cm`, top at `room_z_cm + thickness_cm`
      - Walls extend from `z = 0` to `z = room_z_cm` on the inside, thickness pushed OUTWARD (so interior is exact `room_size`)
      - Walls at x=0 and x=room_x have inner face flush with `x=0` and `x=room_x_cm`
      - Walls at y=0 and y=room_y have inner face flush with `y=0` and `y=room_y_cm`

- [ ] **Step 1: Write the failing test**

Create `/data/jzy/code/SPEAR/tests/test_render_in_gpurir_room.py`:

```python
import importlib.util
import json
import math
import os
import tempfile
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "render_in_gpurir_room.py"
    spec = importlib.util.spec_from_file_location("render_in_gpurir_room", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ShoeboxRoomLayoutTests(unittest.TestCase):
    def test_layout_returns_six_pieces_in_stable_order(self):
        mod = load_module()

        pieces = mod.compute_shoebox_room_layout(room_size_m=(5.2, 4.4, 2.8))

        self.assertEqual(len(pieces), 6)
        self.assertEqual(
            [p["name"] for p in pieces],
            ["floor", "ceiling", "wall_x0", "wall_x1", "wall_y0", "wall_y1"],
        )

    def test_floor_top_face_sits_at_z_zero(self):
        mod = load_module()

        pieces = mod.compute_shoebox_room_layout(room_size_m=(5.2, 4.4, 2.8))
        floor = pieces[0]

        cx, cy, cz = floor["location_cm"]
        sx, sy, sz = floor["scale"]
        # Cube is 1m base -> actual size = scale * 100 cm
        thickness_cm = sz * 100.0
        top_face_z = cz + thickness_cm / 2.0

        self.assertAlmostEqual(top_face_z, 0.0, places=6)
        self.assertAlmostEqual(cx, 520.0 / 2.0, places=6)
        self.assertAlmostEqual(cy, 440.0 / 2.0, places=6)
        # Floor covers full room footprint
        self.assertAlmostEqual(sx * 100.0, 520.0, places=6)
        self.assertAlmostEqual(sy * 100.0, 440.0, places=6)

    def test_ceiling_bottom_face_sits_at_room_height(self):
        mod = load_module()

        pieces = mod.compute_shoebox_room_layout(room_size_m=(5.2, 4.4, 2.8))
        ceiling = pieces[1]

        cx, cy, cz = ceiling["location_cm"]
        sx, sy, sz = ceiling["scale"]
        thickness_cm = sz * 100.0
        bottom_face_z = cz - thickness_cm / 2.0

        self.assertAlmostEqual(bottom_face_z, 280.0, places=6)

    def test_walls_have_inner_faces_flush_with_room_bounds(self):
        mod = load_module()

        pieces = mod.compute_shoebox_room_layout(room_size_m=(5.2, 4.4, 2.8))
        pieces_by_name = {p["name"]: p for p in pieces}

        # wall_x0: inner (max) x face = 0
        w = pieces_by_name["wall_x0"]
        cx, cy, cz = w["location_cm"]
        sx, sy, sz = w["scale"]
        inner_x = cx + sx * 100.0 / 2.0
        self.assertAlmostEqual(inner_x, 0.0, places=6)

        # wall_x1: inner (min) x face = 520
        w = pieces_by_name["wall_x1"]
        cx, cy, cz = w["location_cm"]
        sx, sy, sz = w["scale"]
        inner_x = cx - sx * 100.0 / 2.0
        self.assertAlmostEqual(inner_x, 520.0, places=6)

        # wall_y0: inner (max) y face = 0
        w = pieces_by_name["wall_y0"]
        cx, cy, cz = w["location_cm"]
        sx, sy, sz = w["scale"]
        inner_y = cy + sy * 100.0 / 2.0
        self.assertAlmostEqual(inner_y, 0.0, places=6)

        # wall_y1: inner (min) y face = 440
        w = pieces_by_name["wall_y1"]
        cx, cy, cz = w["location_cm"]
        sx, sy, sz = w["scale"]
        inner_y = cy - sy * 100.0 / 2.0
        self.assertAlmostEqual(inner_y, 440.0, places=6)

    def test_wall_heights_match_room_z(self):
        mod = load_module()

        pieces = mod.compute_shoebox_room_layout(room_size_m=(5.2, 4.4, 2.8))
        for name in ("wall_x0", "wall_x1", "wall_y0", "wall_y1"):
            w = [p for p in pieces if p["name"] == name][0]
            _, _, cz = w["location_cm"]
            _, _, sz = w["scale"]
            self.assertAlmostEqual(cz, 280.0 / 2.0, places=6)
            self.assertAlmostEqual(sz * 100.0, 280.0, places=6)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_gpurir_room -v`
Expected: `ModuleNotFoundError: No module named 'render_in_gpurir_room'` OR test file loads but 5 FAIL with `AttributeError: ... 'compute_shoebox_room_layout'`.

- [ ] **Step 3: Create the module skeleton + implement**

Create `/data/jzy/code/SPEAR/examples/render_in_gpurir_room.py` with:

```python
"""Render an imported Hunyuan3D asset inside a GPURIR-parameter-aligned
shoebox room built from Cube meshes with apartment_0000 materials.

Use spear-env:
  DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    /data/jzy/miniconda3/envs/spear-env/bin/python examples/render_in_gpurir_room.py \
    --animal dog

See HANDOFF_GPURIR_ROOM.md for the full spec.
"""

import argparse
import json
import math
import os
import subprocess


M2CM = 100.0
WALL_THICKNESS_M = 0.1


def compute_shoebox_room_layout(*, room_size_m, wall_thickness_m=WALL_THICKNESS_M):
    rx, ry, rz = (float(v) for v in room_size_m)
    t = float(wall_thickness_m)
    rx_cm = rx * M2CM
    ry_cm = ry * M2CM
    rz_cm = rz * M2CM
    t_cm = t * M2CM

    pieces = []

    # Floor: covers footprint rx x ry, top face at z=0
    pieces.append({
        "name": "floor",
        "location_cm": (rx_cm / 2.0, ry_cm / 2.0, -t_cm / 2.0),
        "scale": (rx, ry, t),
    })

    # Ceiling: covers footprint rx x ry, bottom face at z=rz
    pieces.append({
        "name": "ceiling",
        "location_cm": (rx_cm / 2.0, ry_cm / 2.0, rz_cm + t_cm / 2.0),
        "scale": (rx, ry, t),
    })

    # Walls: interior clear, thickness pushed OUTWARD from room bounds.
    # x0 wall (at x < 0): inner face at x=0
    pieces.append({
        "name": "wall_x0",
        "location_cm": (-t_cm / 2.0, ry_cm / 2.0, rz_cm / 2.0),
        "scale": (t, ry, rz),
    })
    # x1 wall (at x > rx): inner face at x=rx
    pieces.append({
        "name": "wall_x1",
        "location_cm": (rx_cm + t_cm / 2.0, ry_cm / 2.0, rz_cm / 2.0),
        "scale": (t, ry, rz),
    })
    # y0 wall
    pieces.append({
        "name": "wall_y0",
        "location_cm": (rx_cm / 2.0, -t_cm / 2.0, rz_cm / 2.0),
        "scale": (rx, t, rz),
    })
    # y1 wall (will be REPLACED by 4 window-pieces later, but included here
    # so callers who don't want a window still get a closed shoebox)
    pieces.append({
        "name": "wall_y1",
        "location_cm": (rx_cm / 2.0, ry_cm + t_cm / 2.0, rz_cm / 2.0),
        "scale": (rx, t, rz),
    })

    return pieces
```

- [ ] **Step 4: Run test to verify pass**

Run: `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_gpurir_room -v`
Expected: `Ran 5 tests ... OK`.

- [ ] **Step 5: Verify Round-1 tests still pass**

Run: `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_apartment -v 2>&1 | tail -3`
Expected: `Ran 28 tests ... OK`.

---

## Task 3: Window wall layout helper (4-piece cube split)

**Files:**
- Modify: `/data/jzy/code/SPEAR/examples/render_in_gpurir_room.py` (add helper)
- Modify: `/data/jzy/code/SPEAR/tests/test_render_in_gpurir_room.py` (add test class)

**Interfaces:**
- Consumes: `M2CM`, `WALL_THICKNESS_M` from Task 2
- Produces:
  - `compute_window_wall_layout(*, room_size_m, window_w_m, window_h_m, window_cx_m, window_z_bottom_m, wall_thickness_m=WALL_THICKNESS_M) -> list[dict]`
    - Splits the y1 wall (at y = room_y_m) into 4 pieces around a rectangular window:
      - `wall_y1_bottom` (window sill), `wall_y1_top` (lintel above window), `wall_y1_left` (left jamb, x < window), `wall_y1_right` (right jamb, x > window)
    - All 4 pieces sit at the same y as `wall_y1` from Task 2 (`ry_cm + t_cm / 2.0`), same thickness `t_cm`
    - Each piece has the same schema as Task 2: `{"name", "location_cm", "scale"}`
    - `window_cx_m` = window center x in meters (defaults to `room_x_m / 2` in caller)
    - `window_z_bottom_m` = window bottom edge in meters

- [ ] **Step 1: Write the failing test**

Append to `tests/test_render_in_gpurir_room.py`:

```python
class WindowWallLayoutTests(unittest.TestCase):
    def test_four_pieces_leave_exact_window_hole(self):
        mod = load_module()

        pieces = mod.compute_window_wall_layout(
            room_size_m=(5.2, 4.4, 2.8),
            window_w_m=2.0,
            window_h_m=2.4,
            window_cx_m=2.6,
            window_z_bottom_m=0.2,
        )

        self.assertEqual(len(pieces), 4)
        names = [p["name"] for p in pieces]
        self.assertEqual(
            sorted(names),
            sorted(["wall_y1_bottom", "wall_y1_top", "wall_y1_left", "wall_y1_right"]),
        )

        by_name = {p["name"]: p for p in pieces}

        # Bottom sill: from z=0 to z=window_z_bottom=20cm
        b = by_name["wall_y1_bottom"]
        cx, cy, cz = b["location_cm"]
        sx, sy, sz = b["scale"]
        top_z = cz + sz * 100.0 / 2.0
        bottom_z = cz - sz * 100.0 / 2.0
        self.assertAlmostEqual(bottom_z, 0.0, places=6)
        self.assertAlmostEqual(top_z, 20.0, places=6)
        # sill full width of room
        self.assertAlmostEqual(sx * 100.0, 520.0, places=6)

        # Top lintel: from z=window_z_bottom+window_h=260cm to z=280cm
        t = by_name["wall_y1_top"]
        cx, cy, cz = t["location_cm"]
        sx, sy, sz = t["scale"]
        bottom_z = cz - sz * 100.0 / 2.0
        top_z = cz + sz * 100.0 / 2.0
        self.assertAlmostEqual(bottom_z, 260.0, places=6)
        self.assertAlmostEqual(top_z, 280.0, places=6)
        self.assertAlmostEqual(sx * 100.0, 520.0, places=6)

        # Left jamb: x from 0 to (window_cx - window_w/2) = 260 - 100 = 160cm, height=window_h=240cm
        l = by_name["wall_y1_left"]
        cx, cy, cz = l["location_cm"]
        sx, sy, sz = l["scale"]
        left_x = cx - sx * 100.0 / 2.0
        right_x = cx + sx * 100.0 / 2.0
        self.assertAlmostEqual(left_x, 0.0, places=6)
        self.assertAlmostEqual(right_x, 160.0, places=6)
        self.assertAlmostEqual(sz * 100.0, 240.0, places=6)
        # sits between sill top (20cm) and lintel bottom (260cm) -> center z=140
        self.assertAlmostEqual(cz, 140.0, places=6)

        # Right jamb: x from 360 to 520
        r = by_name["wall_y1_right"]
        cx, cy, cz = r["location_cm"]
        sx, sy, sz = r["scale"]
        left_x = cx - sx * 100.0 / 2.0
        right_x = cx + sx * 100.0 / 2.0
        self.assertAlmostEqual(left_x, 360.0, places=6)
        self.assertAlmostEqual(right_x, 520.0, places=6)
        self.assertAlmostEqual(sz * 100.0, 240.0, places=6)

    def test_all_pieces_share_wall_y_and_thickness(self):
        mod = load_module()

        pieces = mod.compute_window_wall_layout(
            room_size_m=(5.2, 4.4, 2.8),
            window_w_m=2.0,
            window_h_m=2.4,
            window_cx_m=2.6,
            window_z_bottom_m=0.2,
        )
        # All 4 pieces have same y and same y-scale (0.1m thickness)
        ys = {p["location_cm"][1] for p in pieces}
        scale_ys = {p["scale"][1] for p in pieces}
        self.assertEqual(len(ys), 1)
        self.assertEqual(len(scale_ys), 1)
        self.assertAlmostEqual(list(ys)[0], 440.0 + 10.0 / 2.0, places=6)
        self.assertAlmostEqual(list(scale_ys)[0], 0.1, places=6)
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_gpurir_room.WindowWallLayoutTests -v`
Expected: 2 FAILs (`AttributeError: ... 'compute_window_wall_layout'`).

- [ ] **Step 3: Implement**

Append to `examples/render_in_gpurir_room.py`:

```python
def compute_window_wall_layout(
    *,
    room_size_m,
    window_w_m,
    window_h_m,
    window_cx_m,
    window_z_bottom_m,
    wall_thickness_m=WALL_THICKNESS_M,
):
    rx, ry, rz = (float(v) for v in room_size_m)
    ww = float(window_w_m)
    wh = float(window_h_m)
    wcx = float(window_cx_m)
    wzb = float(window_z_bottom_m)
    t = float(wall_thickness_m)

    rx_cm = rx * M2CM
    rz_cm = rz * M2CM
    ry_cm = ry * M2CM
    t_cm = t * M2CM

    window_left_x_cm = (wcx - ww / 2.0) * M2CM
    window_right_x_cm = (wcx + ww / 2.0) * M2CM
    window_bottom_z_cm = wzb * M2CM
    window_top_z_cm = (wzb + wh) * M2CM

    wall_y_center_cm = ry_cm + t_cm / 2.0

    pieces = []

    # Bottom sill: full room width, from z=0 to z=window_bottom
    sill_h_cm = window_bottom_z_cm
    pieces.append({
        "name": "wall_y1_bottom",
        "location_cm": (rx_cm / 2.0, wall_y_center_cm, sill_h_cm / 2.0),
        "scale": (rx, t, sill_h_cm / M2CM),
    })

    # Top lintel: full room width, from z=window_top to z=room_z
    lintel_h_cm = rz_cm - window_top_z_cm
    pieces.append({
        "name": "wall_y1_top",
        "location_cm": (
            rx_cm / 2.0,
            wall_y_center_cm,
            window_top_z_cm + lintel_h_cm / 2.0,
        ),
        "scale": (rx, t, lintel_h_cm / M2CM),
    })

    # Left jamb: from x=0 to x=window_left, spans z=[window_bottom, window_top]
    jamb_h_cm = window_top_z_cm - window_bottom_z_cm
    left_w_cm = window_left_x_cm
    pieces.append({
        "name": "wall_y1_left",
        "location_cm": (
            left_w_cm / 2.0,
            wall_y_center_cm,
            window_bottom_z_cm + jamb_h_cm / 2.0,
        ),
        "scale": (left_w_cm / M2CM, t, jamb_h_cm / M2CM),
    })

    # Right jamb: from x=window_right to x=room_x
    right_w_cm = rx_cm - window_right_x_cm
    pieces.append({
        "name": "wall_y1_right",
        "location_cm": (
            window_right_x_cm + right_w_cm / 2.0,
            wall_y_center_cm,
            window_bottom_z_cm + jamb_h_cm / 2.0,
        ),
        "scale": (right_w_cm / M2CM, t, jamb_h_cm / M2CM),
    })

    return pieces
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_gpurir_room -v`
Expected: `Ran 7 tests ... OK`.

---

## Task 4: GPURIR mic/source position helpers (pure)

**Files:**
- Modify: `/data/jzy/code/SPEAR/examples/render_in_gpurir_room.py`
- Modify: `/data/jzy/code/SPEAR/tests/test_render_in_gpurir_room.py`

**Interfaces:**
- Consumes: `M2CM` from Task 2
- Produces:
  - `MIC_HEIGHT_M = 1.2` module constant (matches v77 GPURIR `MIC_HEIGHT`)
  - `compute_mic_position_cm(*, room_size_m) -> tuple(float, float, float)` — returns `(x_cm, y_cm, z_cm)` at `(room_x/2, room_y/2, 1.2 m)`
  - `compute_source_position_cm(*, room_size_m, source_offset_m=(0.0, 1.7, 0.0)) -> tuple(float, float, float)` — returns mic + offset, all in cm. Default offset is +Y 1.7 m; z-component ignored (source z-floor = 0 per GPURIR contract, but we allow non-zero for future flexibility — return z-cm = mic_z_cm + offset_z_cm, caller decides).
  - Note: the animal is placed with its FEET at ground level regardless of mic z, so `render_dog` will use only the x/y of `compute_source_position_cm` and use `sample_ground_z` for z.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_render_in_gpurir_room.py`:

```python
class MicSourcePositionTests(unittest.TestCase):
    def test_mic_at_room_center_height_1_2m(self):
        mod = load_module()

        pos = mod.compute_mic_position_cm(room_size_m=(5.2, 4.4, 2.8))

        self.assertEqual(pos, (260.0, 220.0, 120.0))

    def test_mic_uses_gpurir_canonical_1_2m_height(self):
        mod = load_module()

        self.assertEqual(mod.MIC_HEIGHT_M, 1.2)

    def test_source_position_defaults_to_mic_plus_1_7m_along_y(self):
        mod = load_module()

        pos = mod.compute_source_position_cm(room_size_m=(5.2, 4.4, 2.8))

        # mic = (260, 220, 120); source = mic + (0, 170, 0) = (260, 390, 120)
        self.assertEqual(pos, (260.0, 390.0, 120.0))

    def test_source_position_respects_custom_offset(self):
        mod = load_module()

        pos = mod.compute_source_position_cm(
            room_size_m=(5.2, 4.4, 2.8),
            source_offset_m=(1.0, 0.5, -0.7),
        )

        self.assertEqual(pos, (360.0, 270.0, 50.0))
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_gpurir_room.MicSourcePositionTests -v`
Expected: 4 FAILs.

- [ ] **Step 3: Implement**

Append to `examples/render_in_gpurir_room.py`:

```python
MIC_HEIGHT_M = 1.2  # v77 gen_rir_multiscene_v77.py compute_mic_pos


def compute_mic_position_cm(*, room_size_m):
    rx, ry, _rz = (float(v) for v in room_size_m)
    return (rx * M2CM / 2.0, ry * M2CM / 2.0, MIC_HEIGHT_M * M2CM)


def compute_source_position_cm(*, room_size_m, source_offset_m=(0.0, 1.7, 0.0)):
    mic_x_cm, mic_y_cm, mic_z_cm = compute_mic_position_cm(room_size_m=room_size_m)
    ox, oy, oz = (float(v) for v in source_offset_m)
    return (mic_x_cm + ox * M2CM, mic_y_cm + oy * M2CM, mic_z_cm + oz * M2CM)
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_gpurir_room -v`
Expected: `Ran 11 tests ... OK`.

---

## Task 5: Room checklist builder (pure)

**Files:**
- Modify: `/data/jzy/code/SPEAR/examples/render_in_gpurir_room.py`
- Modify: `/data/jzy/code/SPEAR/tests/test_render_in_gpurir_room.py`

**Interfaces:**
- Consumes: nothing structural
- Produces:
  - `build_room_checklist(*, solo_checklist, room_size_m, mic_pos_cm, source_pos_cm, window_bounds_cm, directional_light_intensity_lux, wall_material, floor_material) -> dict`
    - `solo_checklist` = dict returned by `render_in_apartment.build_solo_checklist(...)`
    - `window_bounds_cm` = dict `{"left_x": .., "right_x": .., "bottom_z": .., "top_z": .., "y": ..}`
    - Returns a superset: solo fields + new fields under top level + a `human_review` block:
      - `"human_review": [<four string items>]` listing what to visually verify (walls have texture / ceiling has texture / window is a real hole with light coming through / directional light casts a clear shadow from the window direction)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_render_in_gpurir_room.py`:

```python
class RoomChecklistTests(unittest.TestCase):
    def test_room_checklist_extends_solo_checklist_with_room_fields(self):
        mod = load_module()
        solo = {
            "name": "dog",
            "frames": 36,
            "target_cm": 80.0,
            "scale": 0.4015,
            "radius_cm": 200.0,
            "ground_z_cm": 0.5,
            "bounds_bottom_z_cm": 1.0,
            "lift_applied_cm": 0.0,
            "penetration_after_lift_cm": 0.0,
            "clearance_cm": 0.5,
            "tolerance_cm": 0.5,
            "ground_ok": True,
        }

        checklist = mod.build_room_checklist(
            solo_checklist=solo,
            room_size_m=(5.2, 4.4, 2.8),
            mic_pos_cm=(260.0, 220.0, 120.0),
            source_pos_cm=(260.0, 390.0, 120.0),
            window_bounds_cm={
                "left_x": 160.0,
                "right_x": 360.0,
                "bottom_z": 20.0,
                "top_z": 260.0,
                "y": 440.0,
            },
            directional_light_intensity_lux=10.0,
            wall_material="/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Walls.MI_Walls",
            floor_material="/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Floor.MI_Floor",
        )

        # Solo fields preserved
        self.assertEqual(checklist["name"], "dog")
        self.assertEqual(checklist["scale"], 0.4015)
        self.assertTrue(checklist["ground_ok"])

        # Room fields added
        self.assertEqual(checklist["room_size_m"], (5.2, 4.4, 2.8))
        self.assertEqual(checklist["mic_pos_cm"], (260.0, 220.0, 120.0))
        self.assertEqual(checklist["source_pos_cm"], (260.0, 390.0, 120.0))
        self.assertEqual(checklist["window_bounds_cm"]["left_x"], 160.0)
        self.assertEqual(checklist["directional_light_intensity_lux"], 10.0)
        self.assertIn("MI_Walls", checklist["wall_material"])
        self.assertIn("MI_Floor", checklist["floor_material"])

        # human_review block
        self.assertIn("human_review", checklist)
        self.assertIsInstance(checklist["human_review"], list)
        self.assertEqual(len(checklist["human_review"]), 4)
        joined = " ".join(checklist["human_review"]).lower()
        self.assertIn("wall", joined)
        self.assertIn("ceiling", joined)
        self.assertIn("window", joined)
        self.assertIn("shadow", joined)
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_gpurir_room.RoomChecklistTests -v`
Expected: 1 FAIL (`AttributeError: ... 'build_room_checklist'`).

- [ ] **Step 3: Implement**

Append to `examples/render_in_gpurir_room.py`:

```python
def build_room_checklist(
    *,
    solo_checklist,
    room_size_m,
    mic_pos_cm,
    source_pos_cm,
    window_bounds_cm,
    directional_light_intensity_lux,
    wall_material,
    floor_material,
):
    extended = dict(solo_checklist)
    extended["room_size_m"] = tuple(float(v) for v in room_size_m)
    extended["mic_pos_cm"] = tuple(float(v) for v in mic_pos_cm)
    extended["source_pos_cm"] = tuple(float(v) for v in source_pos_cm)
    extended["window_bounds_cm"] = {k: float(v) for k, v in window_bounds_cm.items()}
    extended["directional_light_intensity_lux"] = float(directional_light_intensity_lux)
    extended["wall_material"] = str(wall_material)
    extended["floor_material"] = str(floor_material)
    extended["human_review"] = [
        "All 4 walls carry a visible apartment wall texture (not gray/untextured)",
        "Ceiling has a visible texture (falls back to MI_Walls if no MI_Ceiling)",
        "Window is a real hole with sky/light visible through it (not a painted decal)",
        "Directional light casts a clear shadow FROM the window direction (window -> interior)",
    ]
    return extended
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_gpurir_room -v`
Expected: `Ran 12 tests ... OK`.

---

## Task 6: Layout PNG helper for GPURIR room (top-down)

**Files:**
- Modify: `/data/jzy/code/SPEAR/examples/render_in_gpurir_room.py`
- Modify: `/data/jzy/code/SPEAR/tests/test_render_in_gpurir_room.py`

**Interfaces:**
- Consumes: nothing structural
- Produces:
  - `write_gpurir_layout(output_dir, *, room_size_m, mic_pos_cm, source_pos_cm, window_bounds_cm, orbit_radius_cm) -> str` — writes `layout.png` to output_dir, returns absolute path. Renders room footprint (rectangle), mic (blue dot), source/dog (orange dot), window (green line on wall_y1), camera orbit (dashed circle around source).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_render_in_gpurir_room.py`:

```python
class GpurirLayoutTests(unittest.TestCase):
    def test_write_gpurir_layout_creates_png(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            path = mod.write_gpurir_layout(
                tmp,
                room_size_m=(5.2, 4.4, 2.8),
                mic_pos_cm=(260.0, 220.0, 120.0),
                source_pos_cm=(260.0, 390.0, 120.0),
                window_bounds_cm={
                    "left_x": 160.0,
                    "right_x": 360.0,
                    "bottom_z": 20.0,
                    "top_z": 260.0,
                    "y": 440.0,
                },
                orbit_radius_cm=200.0,
            )
            self.assertEqual(path, os.path.join(tmp, "layout.png"))
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 500)
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_gpurir_room.GpurirLayoutTests -v`
Expected: 1 FAIL (`AttributeError: ... 'write_gpurir_layout'`).

- [ ] **Step 3: Implement**

Append to `examples/render_in_gpurir_room.py`:

```python
def write_gpurir_layout(
    output_dir,
    *,
    room_size_m,
    mic_pos_cm,
    source_pos_cm,
    window_bounds_cm,
    orbit_radius_cm,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    rx_cm = float(room_size_m[0]) * M2CM
    ry_cm = float(room_size_m[1]) * M2CM

    fig, ax = plt.subplots(figsize=(6, 6))

    # Room footprint
    ax.add_patch(plt.Rectangle((0.0, 0.0), rx_cm, ry_cm, fill=False, edgecolor="black"))

    # Window on wall_y1
    ax.plot(
        [float(window_bounds_cm["left_x"]), float(window_bounds_cm["right_x"])],
        [float(window_bounds_cm["y"]), float(window_bounds_cm["y"])],
        color="tab:green",
        linewidth=4,
        label="window",
    )

    # Mic (blue), source (orange)
    ax.plot([mic_pos_cm[0]], [mic_pos_cm[1]], marker="o", color="tab:blue", markersize=8, label="mic")
    ax.plot([source_pos_cm[0]], [source_pos_cm[1]], marker="o", color="tab:orange", markersize=10, label="source (dog)")

    # Camera orbit around source
    theta = [2.0 * math.pi * i / 128 for i in range(129)]
    ax.plot(
        [float(source_pos_cm[0]) + float(orbit_radius_cm) * math.cos(t) for t in theta],
        [float(source_pos_cm[1]) + float(orbit_radius_cm) * math.sin(t) for t in theta],
        linestyle="--",
        color="tab:blue",
        alpha=0.5,
        label=f"orbit r={float(orbit_radius_cm):.0f}cm",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title(
        f"GPURIR shoebox {room_size_m[0]:.2f}x{room_size_m[1]:.2f}x{room_size_m[2]:.2f} m (top-down)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "layout.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path
```

- [ ] **Step 4: Run to verify pass**

Run: `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_gpurir_room -v`
Expected: `Ran 13 tests ... OK`.

---

## Task 7: Integration — `render_gpurir_room()` + CLI + `configure_instance` override

**Files:**
- Modify: `/data/jzy/code/SPEAR/examples/render_in_gpurir_room.py` (add integration function, CLI, main)
- Modify: `/data/jzy/code/SPEAR/tests/test_render_in_gpurir_room.py` (CLI parse tests only)

**Interfaces:**
- Consumes: everything from Tasks 2-6, plus imports from `render_in_apartment`:
  - `compute_asset_fit`, `sample_ground_z`, `get_actor_bounds_bottom_z`, `spawn_camera`, `read_frame`, `build_solo_checklist`, `write_checklist`, `animal_bp_path`, `animal_meta_path`, `clean_frames`, `compute_bounds_lift`
- Produces:
  - `EMPTY_MAP = "/Engine/Maps/Entry"` constant
  - `LIGHT_STUDIO_BP = "/Engine/EngineSky/BP_LightStudio.BP_LightStudio_C"` constant
  - `CUBE_MESH = "/Engine/BasicShapes/Cube.Cube"` constant
  - `FLOOR_MATERIAL = "/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Floor.MI_Floor"` constant
  - `WALL_MATERIAL = "/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Walls.MI_Walls"` constant
  - `configure_gpurir_instance(*, rpc_port) -> spear.Instance` — same as `render_in_apartment.configure_instance` but forces `GAME_DEFAULT_MAP = EMPTY_MAP`
  - `spawn_room_piece(game, *, piece, material_path) -> Actor` — spawns one Cube StaticMeshActor at `piece["location_cm"]` with `SetActorScale3D(piece["scale"])` and applies `material_path` to the mesh's material slot 0. Returns the spawned actor.
  - `spawn_directional_light(game, *, yaw_deg, pitch_deg, intensity_lux) -> Actor` — spawns an `ADirectionalLight`, sets `SetMobility("Movable")`, rotates, sets `SetIntensity(intensity_lux)`.
  - `spawn_sky(game) -> Actor` — spawns a `BP_LightStudio` actor at origin.
  - `render_gpurir_room(args)` — end-to-end run
  - `parse_args(argv=None) -> Namespace`
  - `main(argv=None)`

- [ ] **Step 1: Write CLI parse tests**

Append to `tests/test_render_in_gpurir_room.py`:

```python
class GpurirCliTests(unittest.TestCase):
    def test_defaults_match_grill_decisions(self):
        mod = load_module()

        args = mod.parse_args([])

        self.assertEqual(args.animal, "dog")
        self.assertEqual(args.room_size_m, [5.2, 4.4, 2.8])
        self.assertEqual(args.window_w_m, 2.0)
        self.assertEqual(args.window_h_m, 2.4)
        self.assertEqual(args.window_z_bottom_m, 0.2)
        self.assertEqual(args.source_offset_m, [0.0, 1.7, 0.0])
        self.assertEqual(args.orbit_radius_cm, 200.0)
        self.assertEqual(args.frames, 36)
        self.assertEqual(args.width, 1280)
        self.assertEqual(args.height, 720)
        self.assertEqual(args.framerate, 12)
        self.assertEqual(args.directional_light_intensity_lux, 10.0)
        self.assertEqual(args.run_name, "dog_default")

    def test_room_size_arg_parses_three_floats(self):
        mod = load_module()

        args = mod.parse_args(["--room-size-m", "6.0", "5.0", "2.9"])

        self.assertEqual(args.room_size_m, [6.0, 5.0, 2.9])

    def test_animal_choice_restricted_to_imported_set(self):
        mod = load_module()

        # These are the animals imported by Round 1 and cooked into the pak
        for name in ("cat", "dog", "goose", "yak"):
            args = mod.parse_args(["--animal", name])
            self.assertEqual(args.animal, name)

        with self.assertRaises(SystemExit):
            mod.parse_args(["--animal", "unicorn"])
```

- [ ] **Step 2: Run to verify fail**

Run: `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_gpurir_room.GpurirCliTests -v`
Expected: 3 FAILs (`parse_args` not defined).

- [ ] **Step 3: Add imports, constants, and CLI to `examples/render_in_gpurir_room.py`**

Insert at the top of the file (right after the docstring) — adjust the existing imports if needed:

```python
import argparse
import json
import math
import os
import subprocess
import sys


# Allow importing sibling module render_in_apartment
_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from render_in_apartment import (
    animal_bp_path,
    animal_meta_path,
    build_solo_checklist,
    clean_frames,
    compute_asset_fit,
    compute_bounds_lift,
    get_actor_bounds_bottom_z,
    read_frame,
    sample_ground_z,
    spawn_camera,
    SUPPORTED_ANIMALS,
    write_checklist,
)


EMPTY_MAP = "/Engine/Maps/Entry"
LIGHT_STUDIO_BP = "/Engine/EngineSky/BP_LightStudio.BP_LightStudio_C"
CUBE_MESH = "/Engine/BasicShapes/Cube.Cube"
FLOOR_MATERIAL = (
    "/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Floor.MI_Floor"
)
WALL_MATERIAL = (
    "/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Walls.MI_Walls"
)
SPEARSIM_EXECUTABLE = (
    "/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh"
)
DEFAULT_TMP_ROOT = "/data/jzy/code/SPEAR/tmp/render_gpurir_room"
DEFAULT_META_DIR = "/data/jzy/code/SPEAR/tmp/asset_meta"
```

Append `parse_args` and `main` at the end of the file:

```python
def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", choices=SUPPORTED_ANIMALS, default="dog")
    parser.add_argument(
        "--room-size-m",
        type=float,
        nargs=3,
        default=[5.2, 4.4, 2.8],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--wall-thickness-m", type=float, default=WALL_THICKNESS_M)
    parser.add_argument("--window-w-m", type=float, default=2.0)
    parser.add_argument("--window-h-m", type=float, default=2.4)
    parser.add_argument(
        "--window-cx-m",
        type=float,
        default=None,
        help="Window center X in meters. Default = room_x/2.",
    )
    parser.add_argument("--window-z-bottom-m", type=float, default=0.2)
    parser.add_argument(
        "--source-offset-m",
        type=float,
        nargs=3,
        default=[0.0, 1.7, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Source (animal) position offset from mic, in meters.",
    )
    parser.add_argument("--target-cm", type=float, default=80.0)
    parser.add_argument("--orbit-radius-cm", type=float, default=200.0)
    parser.add_argument("--cam-z-offset-cm", type=float, default=40.0)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--framerate", type=int, default=12)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--per-frame-warmup-frames", type=int, default=6)
    parser.add_argument("--ground-clearance-cm", type=float, default=0.5)
    parser.add_argument("--ground-tolerance-cm", type=float, default=0.5)
    parser.add_argument("--ground-trace-start-z", type=float, default=300.0)
    parser.add_argument("--ground-trace-end-z", type=float, default=-200.0)
    parser.add_argument("--floor-z", type=float, default=0.0)
    parser.add_argument("--directional-light-intensity-lux", type=float, default=10.0)
    parser.add_argument("--directional-light-yaw-deg", type=float, default=-90.0,
                        help="Yaw pointing INTO the room from the +Y window. "
                             "-90 in UE means 'light travels in the -Y direction'.")
    parser.add_argument("--directional-light-pitch-deg", type=float, default=-40.0)
    parser.add_argument("--rpc-port", type=int, default=39002)
    parser.add_argument("--meta-dir", default=DEFAULT_META_DIR)
    parser.add_argument("--output-root", default=DEFAULT_TMP_ROOT)
    parser.add_argument("--run-name", default="dog_default")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    render_gpurir_room(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run CLI tests to verify pass (integration function still stubbed)**

Run: `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_gpurir_room -v`
Expected: `Ran 16 tests ... OK` (13 helper tests + 3 CLI tests). If it fails because `render_gpurir_room` symbol is missing at module load, add a stub first:

```python
def render_gpurir_room(args):
    raise NotImplementedError("Filled in by Task 7 Step 5")
```

- [ ] **Step 5: Implement `configure_gpurir_instance`, `spawn_room_piece`, `spawn_directional_light`, `spawn_sky`, and `render_gpurir_room`**

Insert into `examples/render_in_gpurir_room.py` (before `parse_args`):

```python
def configure_gpurir_instance(*, rpc_port):
    import spear

    config = spear.get_config(user_config_files=[])
    config.defrost()
    config.SPEAR.LAUNCH_MODE = "game"
    config.SPEAR.INSTANCE.GAME_EXECUTABLE = SPEARSIM_EXECUTABLE
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP = True
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = EMPTY_MAP
    config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT = int(rpc_port)
    config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES = "/etc/vulkan/icd.d/nvidia_icd.json"
    config.freeze()
    spear.configure_system(config=config)
    return spear.Instance(config=config)


def spawn_room_piece(game, *, piece, material_path):
    cube_uclass = game.unreal_service.load_class(uclass="UStaticMesh", name=CUBE_MESH)
    material = game.unreal_service.load_object(uclass="UMaterialInterface", name=material_path)
    actor = game.unreal_service.spawn_actor(
        uclass="AStaticMeshActor",
        location={
            "X": float(piece["location_cm"][0]),
            "Y": float(piece["location_cm"][1]),
            "Z": float(piece["location_cm"][2]),
        },
    )
    try:
        actor.K2_GetRootComponent().SetMobility(NewMobility="Movable")
    except Exception:
        pass
    smc = game.unreal_service.get_component_by_class(
        actor=actor, uclass="UStaticMeshComponent"
    )
    smc.SetStaticMesh(NewMesh=cube_uclass)
    smc.SetMaterial(ElementIndex=0, Material=material)
    actor.SetActorScale3D(
        NewScale3D={
            "X": float(piece["scale"][0]),
            "Y": float(piece["scale"][1]),
            "Z": float(piece["scale"][2]),
        }
    )
    game.unreal_service.set_stable_name_for_actor(
        actor=actor, stable_name=f"GpurirRoom/{piece['name']}"
    )
    return actor


def spawn_directional_light(game, *, yaw_deg, pitch_deg, intensity_lux):
    light = game.unreal_service.spawn_actor(
        uclass="ADirectionalLight",
        location={"X": 0.0, "Y": 0.0, "Z": 500.0},
    )
    root = light.K2_GetRootComponent()
    root.SetMobility(NewMobility="Movable")
    light.K2_SetActorLocationAndRotation(
        NewLocation={"X": 0.0, "Y": 0.0, "Z": 500.0},
        NewRotation={"Roll": 0.0, "Pitch": float(pitch_deg), "Yaw": float(yaw_deg)},
        bSweep=False,
        bTeleport=True,
    )
    comp = game.unreal_service.get_component_by_class(
        actor=light, uclass="UDirectionalLightComponent"
    )
    comp.SetIntensity(NewIntensity=float(intensity_lux))
    return light


def spawn_sky(game):
    try:
        sky_uclass = game.unreal_service.load_class(uclass="AActor", name=LIGHT_STUDIO_BP)
    except Exception:
        return None  # BP_LightStudio not available in this build; caller can rely on DirLight alone
    sky = game.unreal_service.spawn_actor(
        uclass=sky_uclass,
        location={"X": 0.0, "Y": 0.0, "Z": 0.0},
    )
    return sky


def render_gpurir_room(args):
    import cv2

    output_dir = os.path.join(args.output_root, args.run_name)
    clean_frames(output_dir)

    # Compute layouts up front (all pure Python)
    room_pieces = compute_shoebox_room_layout(
        room_size_m=args.room_size_m,
        wall_thickness_m=args.wall_thickness_m,
    )
    # Replace wall_y1 with 4 window-cut pieces
    window_cx = args.window_cx_m if args.window_cx_m is not None else args.room_size_m[0] / 2.0
    window_pieces = compute_window_wall_layout(
        room_size_m=args.room_size_m,
        window_w_m=args.window_w_m,
        window_h_m=args.window_h_m,
        window_cx_m=window_cx,
        window_z_bottom_m=args.window_z_bottom_m,
        wall_thickness_m=args.wall_thickness_m,
    )
    non_y1 = [p for p in room_pieces if p["name"] != "wall_y1"]
    all_pieces = non_y1 + window_pieces

    mic_pos_cm = compute_mic_position_cm(room_size_m=args.room_size_m)
    source_pos_cm = compute_source_position_cm(
        room_size_m=args.room_size_m,
        source_offset_m=args.source_offset_m,
    )
    window_bounds_cm = {
        "left_x": (window_cx - args.window_w_m / 2.0) * M2CM,
        "right_x": (window_cx + args.window_w_m / 2.0) * M2CM,
        "bottom_z": args.window_z_bottom_m * M2CM,
        "top_z": (args.window_z_bottom_m + args.window_h_m) * M2CM,
        "y": args.room_size_m[1] * M2CM,
    }

    # Load animal meta for scale
    with open(animal_meta_path(args.meta_dir, args.animal), "r", encoding="utf-8") as f:
        meta = json.load(f)

    instance = configure_gpurir_instance(rpc_port=args.rpc_port)
    game = instance.get_game()
    try:
        # Frame 0: spawn room, lights, animal
        with instance.begin_frame():
            for piece in all_pieces:
                material = FLOOR_MATERIAL if piece["name"] == "floor" else WALL_MATERIAL
                spawn_room_piece(game=game, piece=piece, material_path=material)

            spawn_sky(game=game)
            spawn_directional_light(
                game=game,
                yaw_deg=args.directional_light_yaw_deg,
                pitch_deg=args.directional_light_pitch_deg,
                intensity_lux=args.directional_light_intensity_lux,
            )

            # Sample ground at source XY (should be ~0 given floor top at z=0)
            ground_z, _ = sample_ground_z(
                game=game,
                x=source_pos_cm[0],
                y=source_pos_cm[1],
                fallback_z=args.floor_z,
                trace_start_z=args.ground_trace_start_z,
                trace_end_z=args.ground_trace_end_z,
            )
            fit = compute_asset_fit(
                meta=meta,
                target_cm=args.target_cm,
                floor_z=ground_z + args.ground_clearance_cm,
            )

            bp_asset = game.unreal_service.load_class(
                uclass="AActor", name=animal_bp_path(args.animal)
            )
            asset = game.unreal_service.spawn_actor(
                uclass=bp_asset,
                location={"X": source_pos_cm[0], "Y": source_pos_cm[1], "Z": args.floor_z},
            )
            try:
                asset.K2_GetRootComponent().SetMobility(NewMobility="Movable")
            except Exception:
                pass
            game.unreal_service.set_stable_name_for_actor(
                actor=asset, stable_name=f"GpurirRoom/{args.animal}"
            )

            cam, comp = spawn_camera(game=game, width=args.width, height=args.height)
        with instance.end_frame():
            pass

        instance.step(num_frames=4)

        # Frame 1: apply scale + lift correction
        with instance.begin_frame():
            asset.SetActorScale3D(
                NewScale3D={"X": fit["scale"], "Y": fit["scale"], "Z": fit["scale"]}
            )
            asset.K2_SetActorLocation(
                NewLocation={"X": source_pos_cm[0], "Y": source_pos_cm[1], "Z": fit["actor_z"]},
                bSweep=False,
                bTeleport=True,
            )
            bounds_bottom_z, _ = get_actor_bounds_bottom_z(actor=asset)
            lift_cm = compute_bounds_lift(
                bounds_bottom_z=bounds_bottom_z,
                ground_z=ground_z,
                clearance_cm=args.ground_clearance_cm,
                tolerance_cm=args.ground_tolerance_cm,
            )
            if lift_cm > 0.0:
                fit["actor_z"] += lift_cm
                fit["center_z"] += lift_cm
                asset.K2_SetActorLocation(
                    NewLocation={"X": source_pos_cm[0], "Y": source_pos_cm[1], "Z": fit["actor_z"]},
                    bSweep=False,
                    bTeleport=True,
                )
        with instance.end_frame():
            pass

        center_x = float(source_pos_cm[0])
        center_y = float(source_pos_cm[1])
        center_z = float(fit["center_z"])

        print(
            "[gpurir-room] "
            f"animal={args.animal} room={args.room_size_m}m "
            f"mic={mic_pos_cm} source={source_pos_cm} "
            f"ground_z={ground_z:.2f}cm bounds_bottom={bounds_bottom_z:.2f}cm "
            f"lift={lift_cm:.2f}cm scale={fit['scale']:.3f} "
            f"radius={args.orbit_radius_cm:.0f}cm center=({center_x:.1f},{center_y:.1f},{center_z:.1f})",
            flush=True,
        )

        instance.step(num_frames=args.warmup_frames)
        for i in range(args.frames):
            frame_warmup = args.warmup_frames if i == 0 else args.per_frame_warmup_frames
            theta = 2.0 * math.pi * i / args.frames
            cam_x = center_x + args.orbit_radius_cm * math.cos(theta)
            cam_y = center_y + args.orbit_radius_cm * math.sin(theta)
            cam_z = center_z + args.cam_z_offset_cm
            yaw = math.degrees(math.atan2(center_y - cam_y, center_x - cam_x))
            pitch = -math.degrees(math.atan2(args.cam_z_offset_cm, args.orbit_radius_cm))
            with instance.begin_frame():
                cam.K2_SetActorLocationAndRotation(
                    NewLocation={"X": cam_x, "Y": cam_y, "Z": cam_z},
                    NewRotation={"Roll": 0.0, "Pitch": pitch, "Yaw": yaw},
                    bSweep=False,
                    bTeleport=True,
                )
            with instance.end_frame():
                pass
            if frame_warmup > 0:
                instance.step(num_frames=frame_warmup)
            with instance.begin_frame():
                pass
            with instance.end_frame():
                cv2.imwrite(
                    os.path.join(output_dir, f"frame_{i:04d}.png"),
                    read_frame(comp),
                )

        video_path = os.path.join(output_dir, "turntable.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(args.framerate),
                "-i", os.path.join(output_dir, "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                video_path,
            ],
            check=True,
            capture_output=True,
        )
        print(f"VIDEO_DONE {video_path}", flush=True)

        # Final bounds for checklist
        with instance.begin_frame():
            final_bounds_bottom_z, _ = get_actor_bounds_bottom_z(actor=asset)
        with instance.end_frame():
            pass

        penetration = (ground_z + args.ground_clearance_cm) - final_bounds_bottom_z
        solo = build_solo_checklist(
            name=args.animal,
            ground_z=ground_z,
            bounds_bottom_z=final_bounds_bottom_z,
            lift_cm=lift_cm,
            penetration_after_lift=penetration,
            scale=fit["scale"],
            target_cm=args.target_cm,
            radius=args.orbit_radius_cm,
            frames=args.frames,
            clearance_cm=args.ground_clearance_cm,
            tolerance_cm=args.ground_tolerance_cm,
        )
        checklist = build_room_checklist(
            solo_checklist=solo,
            room_size_m=args.room_size_m,
            mic_pos_cm=mic_pos_cm,
            source_pos_cm=source_pos_cm,
            window_bounds_cm=window_bounds_cm,
            directional_light_intensity_lux=args.directional_light_intensity_lux,
            wall_material=WALL_MATERIAL,
            floor_material=FLOOR_MATERIAL,
        )
        checklist_path = write_checklist(output_dir, checklist)
        print(f"CHECKLIST_DONE {checklist_path}", flush=True)

        layout_path = write_gpurir_layout(
            output_dir,
            room_size_m=args.room_size_m,
            mic_pos_cm=mic_pos_cm,
            source_pos_cm=source_pos_cm,
            window_bounds_cm=window_bounds_cm,
            orbit_radius_cm=args.orbit_radius_cm,
        )
        print(f"LAYOUT_DONE {layout_path}", flush=True)
    finally:
        instance.close(force=True)
```

- [ ] **Step 6: Run all tests (pure helpers + CLI) to verify no regressions**

Run: `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_gpurir_room tests.test_render_in_apartment 2>&1 | tail -4`
Expected: `Ran 44 tests ... OK` (16 gpurir + 28 apartment).

---

## Task 8: BREAKPOINT — render dog in the new GPURIR room

**Files:**
- Output: `/data/jzy/code/SPEAR/tmp/render_gpurir_room/dog_default/{turntable.mp4, frame_0000.png, checklist.json, layout.png}`

- [ ] **Step 1: spear-env self-check**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -c "import spear; print(spear.__can_import_spear_ext__)"`
Expected: `True`.

- [ ] **Step 2: Xvfb running**

Run: `pgrep -af "Xvfb :99" | head -1`
Expected: one line with `Xvfb :99 …`. If missing: `Xvfb :99 -screen 0 1280x720x24 &`.

- [ ] **Step 3: Render**

Run:
```bash
cd /data/jzy/code/SPEAR && \
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python examples/render_in_gpurir_room.py \
  --animal dog 2>&1 | tail -15
```
Expected: prints `VIDEO_DONE`, `CHECKLIST_DONE`, `LAYOUT_DONE` all followed by their absolute paths.

- [ ] **Step 4: Verify artifacts**

Run: `ls -la /data/jzy/code/SPEAR/tmp/render_gpurir_room/dog_default/`
Expected: `turntable.mp4`, `layout.png`, `checklist.json`, and `frame_0000.png … frame_0035.png` (36 frames).

- [ ] **Step 5: Print checklist**

Run: `cat /data/jzy/code/SPEAR/tmp/render_gpurir_room/dog_default/checklist.json`

Verify:
- `ground_ok: true` and `|penetration_after_lift_cm| < 0.5`
- `room_size_m` == `[5.2, 4.4, 2.8]`
- `mic_pos_cm` == `[260.0, 220.0, 120.0]`
- `source_pos_cm` == `[260.0, 390.0, 120.0]`
- `window_bounds_cm.left_x` == `160.0`, `right_x` == `360.0`, `bottom_z` == `20.0`, `top_z` == `260.0`, `y` == `440.0`
- `directional_light_intensity_lux` == `10.0`
- `wall_material` contains `MI_Walls`, `floor_material` contains `MI_Floor`

If any auto field is wrong, stop and report to user before showing artifacts.

- [ ] **Step 6: Show artifacts and STOP for user review**

Post to user:
- `frame_0000.png` (preview embed)
- `layout.png` (top-down layout confirmation)
- `turntable.mp4` path
- `checklist.json` contents (pretty-printed)
- Explicit request: "Human-review 4 items (from checklist['human_review']):
  1. All 4 walls carry visible apartment wall texture (not gray)
  2. Ceiling has visible texture
  3. Window is a real hole with sky/light visible through it
  4. Directional light casts clear shadow from window direction
  Approve → say `next`; re-render → say what to change."

**DO NOT** proceed to Task 9 until user approves.

---

## Task 9: Wrap-up and hand off to next round

- [ ] **Step 1: Update Round-2 HANDOFF with the run's actual checklist values**

Edit `HANDOFF_GPURIR_ROOM.md` §7 ("下一轮计划") — add a "已完成" line listing the video path + key checklist numbers, so the next-round handoff can start from ground truth.

- [ ] **Step 2: Compile summary for user**

Post:
```
GPURIR-shaped room video rendered:
  path:      tmp/render_gpurir_room/dog_default/turntable.mp4
  layout:    tmp/render_gpurir_room/dog_default/layout.png
  checklist: tmp/render_gpurir_room/dog_default/checklist.json

Room:      5.2 x 4.4 x 2.8 m (GPURIR-aligned)
Mic:       (2.60, 2.20, 1.20) m  [tetra center, GPURIR canonical]
Source:    (2.60, 3.90, 0.00) m  [mic + 1.7 m along +Y, in front of window]
Window:    2.0 x 2.4 m floor-to-ceiling on Y-max wall
Materials: apartment_0000 MI_Floor + MI_Walls
Lighting:  BP_LightStudio sky + DirectionalLight from window direction @ 10 lux

Verdict: Parametric GPURIR room with apartment textures works. Next round
proposals (NOT started): mass render across sampled room sizes; add
Puppeteer/animation to make the dog walk.
```

- [ ] **Step 3: Ask user for the next round direction**

"Approve wrap-up? Mass-render next, or animation next, or something else?"

---

## Self-Review

### 1. Spec coverage

Grill decisions Q1-Q14 mapped:
- Q1 room parameterized (`--room-size-m x y z`) → Task 2 (helper), Task 7 (CLI)
- Q2 MI_Floor + MI_Walls + window + sky/light → Task 7 (constants + `spawn_room_piece` + `spawn_sky` + `spawn_directional_light`)
- Q3 window on Y-max wall → Task 3 `compute_window_wall_layout` (splits `wall_y1`)
- Q4 window 2.0 × 2.4 m, cx=room_x/2, z_bottom=0.2m → Task 7 CLI defaults
- Q5 4-piece Cube window split → Task 3
- Q6 DirLight + BP_LightStudio from window direction → Task 7 `spawn_directional_light` yaw=-90 (INTO room from +Y), `spawn_sky`
- Q7 dog only → Task 8 uses `--animal dog` default
- Q8 source at `(mic_x, mic_y + 1.7 m, 0)` → Task 4 `compute_source_position_cm` default offset `(0, 1.7, 0)`
- Q9 floor top face at z=0 + ground-trace safety → Task 2 layout puts floor top at 0, Task 7 `render_gpurir_room` calls `sample_ground_z`
- Q10 1280×720 / 12 fps / 36 frames / 3 s, radius=200cm → Task 7 CLI defaults
- Q11 full checklist (auto 8 + human 4) → Task 5 `build_room_checklist`
- Q12 new script + new tests → Task 2 creates both, all other tasks modify only these two files
- Q13 `/Engine/Maps/Entry` → Task 7 constant `EMPTY_MAP`
- Q14 pure helper full TDD → Tasks 2, 3, 4, 5, 6, and Task 7's CLI parse each start with failing test → impl → passing test

**Gap check:** none.

### 2. Placeholder scan

Grep run: no `TBD`, `TODO`, `implement later`, `fill in details`, `similar to Task`, `handle edge cases` in the plan body. Every code step includes the exact code to insert; every command shows the expected output.

### 3. Type consistency

- `compute_shoebox_room_layout` returns `list[dict]` with keys `name`, `location_cm` (3-tuple), `scale` (3-tuple) — used identically in Task 2 tests, Task 3 (splits one piece), Task 7 (`spawn_room_piece`).
- `compute_window_wall_layout` returns same schema — verified in Task 3 tests and Task 7 wiring.
- `compute_mic_position_cm` / `compute_source_position_cm` return 3-tuple `(x_cm, y_cm, z_cm)` — used in Task 5 tests, Task 6 layout, Task 7 render.
- `window_bounds_cm` dict schema `{left_x, right_x, bottom_z, top_z, y}` is fixed in Task 5 tests, Task 6 tests, Task 7 render.
- `build_solo_checklist` return schema is inherited from `render_in_apartment` (Round 1), passed intact to `build_room_checklist`.
- `spawn_room_piece(piece, material_path)` reads `piece["location_cm"]` and `piece["scale"]` — matches Task 2 output schema exactly.
- CLI flags stable: `--animal`, `--room-size-m`, `--window-w-m`, `--window-h-m`, `--window-cx-m`, `--window-z-bottom-m`, `--source-offset-m`, `--orbit-radius-cm`, `--run-name`. Namespace field access uses underscores (e.g., `args.room_size_m`) — Python argparse does this automatically.
