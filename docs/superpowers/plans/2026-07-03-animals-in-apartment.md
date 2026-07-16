# Animals-in-Apartment (Verification Round) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify that already-imported Hunyuan3D-2.1 animals (cat, dog, goose, yak) render correctly inside SPEAR's `apartment_0000` — produce 4 solo 360° turntable videos plus 1 line-up "group photo" video, each accompanied by an automated checklist and (for the group) a top-down layout preview, with the user reviewing between segments.

**Architecture:** Extend the existing single-asset `render_in_apartment.py` with (1) an animal-name shortcut that maps `cat` → `/Game/MyAssets/Audioset/Blueprints/cat/BP_cat.BP_cat_C` and pulls the meta from `tmp/asset_meta/cat.json`, (2) a deterministic checklist emitter that records ground/bounds/scale/radius numbers computed during render, and (3) a new `group` mode that lines the four animals from small→large along a bbox-adaptive spacing rule, orbits the camera around the midpoint, and writes a top-down `layout.png`. All 5 segments run as independent SpearSim invocations (session-per-segment) so user breakpoints don't hold GPU resources.

**Tech Stack:** Python 3.11 (`spear-env`), SPEAR RPC (`spear.Instance`), Unreal Engine 5.5 SpearSim, OpenCV, ffmpeg, matplotlib (for layout png only), unittest.

## Global Constraints

- Python interpreter: `/data/jzy/miniconda3/envs/spear-env/bin/python` (NEVER `thu` env — missing `spear_ext`)
- Environment prefix for any script that talks to SpearSim: `DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`
- Video spec (all 5 segments): 1280×720, 12 fps, 36 frames, 3 s, H.264 CRF 23
- Reuse the validated apartment `spawn` point: `spawn_x=-120.0`, `spawn_y=80.0` (Clock's ground-clean location — z=27.11 cm floor)
- Ground clearance default `0.5 cm`; ground tolerance `0.5 cm` (same as prior turntable)
- All 4 animals: `cat`, `dog`, `goose`, `yak` (BP already imported at `/Game/MyAssets/Audioset/Blueprints/<name>/BP_<name>.BP_<name>_C`; meta exists at `tmp/asset_meta/<name>.json`)
- Line-up order: small → large = `cat, dog, goose, yak`
- Line-up spacing: adjacent centers separated by `half_extent_left + half_extent_right + 30.0 cm` gap (bbox-adaptive), computed from each animal's meta `ext` normalized to `target_cm=80.0`
- Output root: `/data/jzy/code/SPEAR/tmp/render_animals_apartment/<segment>/` where `<segment>` ∈ {`cat`, `dog`, `goose`, `yak`, `group`}
- Every segment writes: `turntable.mp4`, `frame_0000.png`, `checklist.json`. `group` additionally writes `layout.png`.
- Every code change follows TDD: pure helpers get unittest coverage in `tests/test_render_in_apartment.py`; rendering is exercised end-to-end only via the 5 real renders (breakpoints).
- Do NOT modify existing `turntable` mode behavior; add capabilities alongside it.
- Follow `docs/agents.style_guide.md` (kwarg-only API calls, ASCII-only source, no bare `except:`).

---

## File Structure

- **Create:** `HANDOFF_ANIMALS_APARTMENT.md` (in SPEAR root) — self-contained spec/handoff mirroring the style of `HANDOFF_VISUAL_RENDER.md`.
- **Modify:** `examples/render_in_apartment.py` — add animal-name mapping, checklist emitter, group-mode function, wire into CLI.
- **Modify:** `tests/test_render_in_apartment.py` — add unit tests for new pure helpers.
- **Read-only reference:**
  - `examples/render_in_apartment.py:197` (existing `sample_ground_z`)
  - `examples/render_in_apartment.py:217` (existing `get_actor_bounds_bottom_z`)
  - `tmp/asset_meta/{cat,dog,goose,yak}.json` (bbox metadata)
  - `HANDOFF_VISUAL_RENDER.md` (previous round's handoff — style reference)

---

## Task 1: Write the spec / handoff document

**Files:**
- Create: `/data/jzy/code/SPEAR/HANDOFF_ANIMALS_APARTMENT.md`

**Interfaces:**
- Consumes: nothing
- Produces: the frozen decision record referenced by every later task ("see HANDOFF §N")

- [ ] **Step 1: Write the handoff document**

Write `/data/jzy/code/SPEAR/HANDOFF_ANIMALS_APARTMENT.md` with these sections (verbatim structure):

1. **一句话现状** — "上一轮已跑通 Clock turntable，本轮验证 4 只动物 (cat/dog/goose/yak) 在 apartment_0000 的插入效果：4 段单只 + 1 段一字排合影，共 5 段视频。"
2. **环境** — copy the two-block "Python 环境 + Xvfb/Vulkan" section from `HANDOFF_VISUAL_RENDER.md` §2 verbatim (must include the `spear.__can_import_spear_ext__` self-check).
3. **本轮 15 项决策** — reproduce the Q1–Q15 table from the grill (scope=A, animals=cat+dog+goose+yak, videos=4 solo + 1 group, layout=line-up small→large, spacing=bbox+30cm gap, camera=360° orbit around midpoint, spec=1280×720/12fps/36 frames/3s, session=per-segment, error=stop-and-ask, spawn=Clock position, checklist=5-item + group +2, output=tmp/render_animals_apartment/, TDD=yes, doc=yes, no worktree/subagent).
4. **文件地图** — list the 3 touched files (this HANDOFF + `examples/render_in_apartment.py` + `tests/test_render_in_apartment.py`) and the plan file `docs/superpowers/plans/2026-07-03-animals-in-apartment.md`.
5. **执行顺序** — 5 breakpoints: `cat → dog → goose → yak → group`; after each, print video path + checklist and stop.
6. **checklist 定义** — the 8 fields listed in Task 4 Step 2 below (copy-paste). Note which are auto-computed vs. human-judged.
7. **下一轮计划** — after all 5 approved, next round is GPURIR shoebox room (5.2×4.4×2.8 m) with apartment materials; NOT part of this plan.

- [ ] **Step 2: Sanity-check the spec compiles**

Run: `grep -c "^##" /data/jzy/code/SPEAR/HANDOFF_ANIMALS_APARTMENT.md`
Expected: `≥ 7` (seven H2 sections)

- [ ] **Step 3: Commit**

Note: SPEAR is not a git repo (verified via env context "Is a git repository: false"). Skip `git commit`. Instead:

Run: `ls -la /data/jzy/code/SPEAR/HANDOFF_ANIMALS_APARTMENT.md`
Expected: file exists, non-zero size.

---

## Task 2: Animal-name → BP-path + meta-path helper

**Files:**
- Modify: `/data/jzy/code/SPEAR/examples/render_in_apartment.py` (add module-level constants + helper)
- Test: `/data/jzy/code/SPEAR/tests/test_render_in_apartment.py` (add test class)

**Interfaces:**
- Consumes: nothing new
- Produces:
  - `ANIMAL_BP_TEMPLATE = "/Game/MyAssets/Audioset/Blueprints/{name}/BP_{name}.BP_{name}_C"`
  - `SUPPORTED_ANIMALS = ("cat", "dog", "goose", "yak")`
  - `animal_bp_path(name: str) -> str`
  - `animal_meta_path(meta_dir: str, name: str) -> str`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_render_in_apartment.py`:

```python
class AnimalResolutionTests(unittest.TestCase):
    def test_animal_bp_path_uses_audioset_blueprints_folder(self):
        mod = load_module()

        self.assertEqual(
            mod.animal_bp_path("cat"),
            "/Game/MyAssets/Audioset/Blueprints/cat/BP_cat.BP_cat_C",
        )
        self.assertEqual(
            mod.animal_bp_path("yak"),
            "/Game/MyAssets/Audioset/Blueprints/yak/BP_yak.BP_yak_C",
        )

    def test_animal_bp_path_rejects_unknown_animal(self):
        mod = load_module()

        with self.assertRaises(ValueError):
            mod.animal_bp_path("unicorn")

    def test_supported_animals_matches_imported_blueprints(self):
        mod = load_module()

        self.assertEqual(mod.SUPPORTED_ANIMALS, ("cat", "dog", "goose", "yak"))

    def test_animal_meta_path_returns_lowercase_json(self):
        mod = load_module()

        self.assertEqual(
            mod.animal_meta_path("/tmp/meta", "cat"),
            "/tmp/meta/cat.json",
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_apartment.AnimalResolutionTests -v`
Expected: 4 FAILs with `AttributeError: module 'render_in_apartment' has no attribute 'animal_bp_path'` (etc.).

- [ ] **Step 3: Implement the helpers**

Add after the existing `DEFAULT_META_DIR = ...` line in `examples/render_in_apartment.py`:

```python
SUPPORTED_ANIMALS = ("cat", "dog", "goose", "yak")
ANIMAL_BP_TEMPLATE = "/Game/MyAssets/Audioset/Blueprints/{name}/BP_{name}.BP_{name}_C"


def animal_bp_path(name):
    if name not in SUPPORTED_ANIMALS:
        raise ValueError(
            f"Unsupported animal {name!r}; supported = {SUPPORTED_ANIMALS}"
        )
    return ANIMAL_BP_TEMPLATE.format(name=name)


def animal_meta_path(meta_dir, name):
    return os.path.join(meta_dir, f"{name}.json")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_apartment.AnimalResolutionTests -v`
Expected: `Ran 4 tests ... OK`.

Also re-run the full suite to make sure nothing regressed:
Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_apartment -v`
Expected: `Ran 13 tests ... OK` (9 prior + 4 new).

- [ ] **Step 5: Commit**

SPEAR is not a git repo — skip commit. Instead verify:
Run: `grep -n "animal_bp_path\|SUPPORTED_ANIMALS" /data/jzy/code/SPEAR/examples/render_in_apartment.py`
Expected: at least 4 hits (constant def, function def, template constant, docstring/use).

---

## Task 3: `--animal` CLI shortcut for solo turntable

**Files:**
- Modify: `/data/jzy/code/SPEAR/examples/render_in_apartment.py` (edit `parse_args`, add post-parse resolution)
- Test: `/data/jzy/code/SPEAR/tests/test_render_in_apartment.py`

**Interfaces:**
- Consumes: `animal_bp_path`, `animal_meta_path`, `SUPPORTED_ANIMALS` from Task 2
- Produces:
  - `parse_args(["--mode", "turntable", "--animal", "cat"])` returns an `argparse.Namespace` whose `asset_bp` and `name` fields are auto-filled from the animal (only when the user did NOT explicitly override them).
  - `--animal <name>` is `None` by default; when set, must be in `SUPPORTED_ANIMALS`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_render_in_apartment.py` inside a new test class:

```python
class AnimalCliShortcutTests(unittest.TestCase):
    def test_animal_shortcut_fills_asset_bp_and_name(self):
        mod = load_module()

        args = mod.parse_args(["--mode", "turntable", "--animal", "cat"])

        self.assertEqual(
            args.asset_bp,
            "/Game/MyAssets/Audioset/Blueprints/cat/BP_cat.BP_cat_C",
        )
        self.assertEqual(args.name, "cat")

    def test_animal_shortcut_defaults_to_none_and_keeps_clock_defaults(self):
        mod = load_module()

        args = mod.parse_args([])

        self.assertIsNone(args.animal)
        self.assertEqual(args.asset_bp, mod.DEFAULT_ASSET_BP)
        self.assertEqual(args.name, mod.DEFAULT_NAME)

    def test_animal_shortcut_respects_explicit_name_override(self):
        mod = load_module()

        args = mod.parse_args(
            ["--mode", "turntable", "--animal", "dog", "--name", "MyDog"]
        )

        self.assertEqual(args.name, "MyDog")
        self.assertEqual(
            args.asset_bp,
            "/Game/MyAssets/Audioset/Blueprints/dog/BP_dog.BP_dog_C",
        )

    def test_animal_shortcut_rejects_unsupported_animal(self):
        mod = load_module()

        with self.assertRaises(SystemExit):
            mod.parse_args(["--animal", "unicorn"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_apartment.AnimalCliShortcutTests -v`
Expected: 4 FAILs (argument `--animal` unrecognized).

- [ ] **Step 3: Implement the CLI shortcut**

In `examples/render_in_apartment.py`, edit `parse_args` (currently ending at `return parser.parse_args(argv)`). Change the last two lines from:

```python
    return parser.parse_args(argv)
```

to:

```python
    parser.add_argument(
        "--animal",
        choices=SUPPORTED_ANIMALS,
        default=None,
        help=(
            "Shortcut for --asset-bp/--name when rendering an imported Hunyuan3D "
            "animal from AudioSet. Fills --asset-bp always; fills --name only if "
            "--name was not explicitly passed."
        ),
    )
    args = parser.parse_args(argv)
    if args.animal is not None:
        args.asset_bp = animal_bp_path(args.animal)
        if args.name == DEFAULT_NAME:
            args.name = args.animal
    return args
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_apartment -v`
Expected: `Ran 17 tests ... OK` (13 prior + 4 new).

- [ ] **Step 5: Commit** — skipped (no git); verify presence:

Run: `grep -n "\-\-animal" /data/jzy/code/SPEAR/examples/render_in_apartment.py`
Expected: 1+ hit.

---

## Task 4: Checklist emitter (deterministic fields from render loop)

**Files:**
- Modify: `/data/jzy/code/SPEAR/examples/render_in_apartment.py` (add `build_solo_checklist`, `write_checklist`, wire into `render_turntable`)
- Test: `/data/jzy/code/SPEAR/tests/test_render_in_apartment.py`

**Interfaces:**
- Consumes: values already computed inside `render_turntable` — `ground_z`, `bounds_bottom_z`, `lift_cm`, `fit["scale"]`, `args.target_cm`, `args.ground_clearance_cm`, `args.ground_tolerance_cm`, `args.frames`, `pose["radius"]` (last frame's radius, which is constant), `output_dir`, video path.
- Produces:
  - `build_solo_checklist(*, name, ground_z, bounds_bottom_z, lift_cm, penetration_after_lift, scale, target_cm, radius, frames, clearance_cm, tolerance_cm) -> dict` — pure, JSON-serializable
  - `write_checklist(output_dir, checklist) -> str` — writes `checklist.json`, returns path
  - Wiring: `render_turntable` calls both at end so every existing turntable run also gains a checklist.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_render_in_apartment.py`:

```python
import json
import tempfile


class ChecklistTests(unittest.TestCase):
    def test_build_solo_checklist_captures_all_deterministic_fields(self):
        mod = load_module()

        checklist = mod.build_solo_checklist(
            name="cat",
            ground_z=27.11,
            bounds_bottom_z=27.6,
            lift_cm=0.0,
            penetration_after_lift=0.01,
            scale=0.401,
            target_cm=80.0,
            radius=130.0,
            frames=36,
            clearance_cm=0.5,
            tolerance_cm=0.5,
        )

        self.assertEqual(checklist["name"], "cat")
        self.assertEqual(checklist["frames"], 36)
        self.assertEqual(checklist["target_cm"], 80.0)
        self.assertEqual(checklist["scale"], 0.401)
        self.assertEqual(checklist["radius_cm"], 130.0)
        self.assertEqual(checklist["ground_z_cm"], 27.11)
        self.assertEqual(checklist["bounds_bottom_z_cm"], 27.6)
        self.assertEqual(checklist["lift_applied_cm"], 0.0)
        self.assertEqual(checklist["penetration_after_lift_cm"], 0.01)
        self.assertTrue(checklist["ground_ok"])
        self.assertIn("clearance_cm", checklist)
        self.assertIn("tolerance_cm", checklist)

    def test_build_solo_checklist_flags_bad_ground(self):
        mod = load_module()

        checklist = mod.build_solo_checklist(
            name="yak",
            ground_z=27.11,
            bounds_bottom_z=10.0,
            lift_cm=0.0,
            penetration_after_lift=17.11,
            scale=1.0,
            target_cm=80.0,
            radius=130.0,
            frames=36,
            clearance_cm=0.5,
            tolerance_cm=0.5,
        )

        self.assertFalse(checklist["ground_ok"])

    def test_write_checklist_roundtrips_json(self):
        mod = load_module()
        with tempfile.TemporaryDirectory() as tmp:
            data = {"name": "cat", "frames": 36}

            path = mod.write_checklist(tmp, data)

            self.assertEqual(path, os.path.join(tmp, "checklist.json"))
            with open(path, "r", encoding="utf-8") as f:
                self.assertEqual(json.load(f), data)
```

(Also add `import os` at the top of the test file if not already present. It IS already imported transitively — verify with `head -5 tests/test_render_in_apartment.py`; if not, add `import os` on line 2.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_apartment.ChecklistTests -v`
Expected: 3 FAILs (`build_solo_checklist` / `write_checklist` not found).

- [ ] **Step 3: Implement the checklist helpers**

Insert into `examples/render_in_apartment.py` (after the existing `get_actor_bounds_bottom_z` function, before `render_reference`):

```python
def build_solo_checklist(
    *,
    name,
    ground_z,
    bounds_bottom_z,
    lift_cm,
    penetration_after_lift,
    scale,
    target_cm,
    radius,
    frames,
    clearance_cm,
    tolerance_cm,
):
    return {
        "name": name,
        "frames": int(frames),
        "target_cm": float(target_cm),
        "scale": float(scale),
        "radius_cm": float(radius),
        "ground_z_cm": float(ground_z),
        "bounds_bottom_z_cm": float(bounds_bottom_z),
        "lift_applied_cm": float(lift_cm),
        "penetration_after_lift_cm": float(penetration_after_lift),
        "clearance_cm": float(clearance_cm),
        "tolerance_cm": float(tolerance_cm),
        "ground_ok": abs(float(penetration_after_lift)) <= float(tolerance_cm),
    }


def write_checklist(output_dir, checklist):
    path = os.path.join(output_dir, "checklist.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checklist, f, indent=2, sort_keys=True)
    return path
```

- [ ] **Step 4: Wire into `render_turntable`**

In `examples/render_in_apartment.py`, after the existing `print(f"VIDEO_DONE {video_path}", flush=True)` line inside `render_turntable`, insert (still inside the `try:` block, before the `finally:`):

```python
        final_bounds_bottom_z, _ = get_actor_bounds_bottom_z(actor=asset)
        penetration_after_lift = (ground_z + args.ground_clearance_cm) - final_bounds_bottom_z
        checklist = build_solo_checklist(
            name=args.name,
            ground_z=ground_z,
            bounds_bottom_z=final_bounds_bottom_z,
            lift_cm=lift_cm,
            penetration_after_lift=penetration_after_lift,
            scale=fit["scale"],
            target_cm=args.target_cm,
            radius=min(args.r_factor * args.target_cm, args.max_radius_cm),
            frames=args.frames,
            clearance_cm=args.ground_clearance_cm,
            tolerance_cm=args.ground_tolerance_cm,
        )
        checklist_path = write_checklist(output_dir, checklist)
        print(f"CHECKLIST_DONE {checklist_path}", flush=True)
```

Note the `final_bounds_bottom_z` computation MUST happen inside a `with instance.begin_frame(): ... with instance.end_frame(): pass` bracket — the second GetActorBounds call is a live UE query. Rewrite the insertion so it's:

```python
        with instance.begin_frame():
            final_bounds_bottom_z, _ = get_actor_bounds_bottom_z(actor=asset)
        with instance.end_frame():
            pass
        penetration_after_lift = (ground_z + args.ground_clearance_cm) - final_bounds_bottom_z
        checklist = build_solo_checklist(
            name=args.name,
            ground_z=ground_z,
            bounds_bottom_z=final_bounds_bottom_z,
            lift_cm=lift_cm,
            penetration_after_lift=penetration_after_lift,
            scale=fit["scale"],
            target_cm=args.target_cm,
            radius=min(args.r_factor * args.target_cm, args.max_radius_cm),
            frames=args.frames,
            clearance_cm=args.ground_clearance_cm,
            tolerance_cm=args.ground_tolerance_cm,
        )
        checklist_path = write_checklist(output_dir, checklist)
        print(f"CHECKLIST_DONE {checklist_path}", flush=True)
```

- [ ] **Step 5: Run all tests to verify pure-helper tests pass and prior tests still pass**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_apartment -v`
Expected: `Ran 20 tests ... OK`.

- [ ] **Step 6: Verify — write and commit** — no git; verify:

Run: `grep -n "CHECKLIST_DONE\|build_solo_checklist" /data/jzy/code/SPEAR/examples/render_in_apartment.py`
Expected: 3+ hits.

---

## Task 5: Line-up position + adaptive orbit-radius helpers (pure)

**Files:**
- Modify: `/data/jzy/code/SPEAR/examples/render_in_apartment.py` (add two pure helpers)
- Test: `/data/jzy/code/SPEAR/tests/test_render_in_apartment.py`

**Interfaces:**
- Consumes: nothing new (uses meta dicts loaded by caller)
- Produces:
  - `compute_lineup_positions(*, animals, metas, target_cm, gap_cm, center_x, center_y) -> list[dict]`
    - `animals`: iterable of names in display order (small→large)
    - `metas`: dict mapping name → meta dict (must contain `ext`, `bmin_z`, `height`)
    - `target_cm`, `gap_cm`, `center_x`, `center_y`: floats
    - Returns list of `{"name", "x", "y", "half_extent_cm"}`, with `y = center_y` for all and `x` values summing to `center_x` when averaged.
    - Uses `ext` as the bbox diagonal proxy; `half_extent_cm = ext * scale / 2` where `scale = target_cm / ext = 1.0` when `ext == target_cm`. Since all metas have `ext ≈ 199` and scale is derived to normalize to `target_cm`, the physical half-extent is always `target_cm / 2`. To keep the helper honest for other animals with different actual bbox widths, pass a `width_extent_cm` OR use the height-agnostic approximation `target_cm / 2`. **Decision: use `target_cm / 2` uniformly** — this matches how `compute_asset_fit` treats `ext` (max bounding-box extent) as the normalization axis. Simplifies the layout without pretending we know per-axis widths.
    - Adjacent center-to-center spacing = `2 * (target_cm / 2) + gap_cm = target_cm + gap_cm`.
  - `compute_group_orbit_radius(*, positions, target_cm, base_r_factor, max_radius_cm) -> float`
    - Returns `min(base_r_factor * target_cm + (rightmost_x - leftmost_x) / 2, max_radius_cm)`.
    - This picks a radius that comfortably frames the entire span while never exceeding the apartment's known safe `max_radius_cm`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_render_in_apartment.py`:

```python
class LineupPositionTests(unittest.TestCase):
    def test_lineup_centers_group_at_given_point(self):
        mod = load_module()
        metas = {
            "cat": {"ext": 199.5, "bmin_z": -65.0, "height": 130.0},
            "dog": {"ext": 199.25, "bmin_z": -80.9, "height": 164.1},
            "goose": {"ext": 199.2, "bmin_z": -95.1, "height": 189.9},
            "yak": {"ext": 198.9, "bmin_z": -45.2, "height": 86.6},
        }

        positions = mod.compute_lineup_positions(
            animals=["cat", "dog", "goose", "yak"],
            metas=metas,
            target_cm=80.0,
            gap_cm=30.0,
            center_x=-120.0,
            center_y=80.0,
        )

        self.assertEqual([p["name"] for p in positions], ["cat", "dog", "goose", "yak"])
        for p in positions:
            self.assertEqual(p["y"], 80.0)
            self.assertEqual(p["half_extent_cm"], 40.0)

        xs = [p["x"] for p in positions]
        # Adjacent spacing = target + gap = 110 cm
        for a, b in zip(xs, xs[1:]):
            self.assertAlmostEqual(b - a, 110.0)
        # Group centered at -120
        self.assertAlmostEqual(sum(xs) / len(xs), -120.0)

    def test_lineup_single_animal_lands_at_center(self):
        mod = load_module()
        metas = {"cat": {"ext": 199.5, "bmin_z": -65.0, "height": 130.0}}

        positions = mod.compute_lineup_positions(
            animals=["cat"],
            metas=metas,
            target_cm=80.0,
            gap_cm=30.0,
            center_x=-120.0,
            center_y=80.0,
        )

        self.assertEqual(len(positions), 1)
        self.assertAlmostEqual(positions[0]["x"], -120.0)


class GroupOrbitRadiusTests(unittest.TestCase):
    def test_orbit_radius_expands_by_half_span_but_clamps(self):
        mod = load_module()
        positions = [
            {"name": "a", "x": -285.0, "y": 80.0, "half_extent_cm": 40.0},
            {"name": "b", "x":  45.0, "y": 80.0, "half_extent_cm": 40.0},
        ]

        # base_r_factor * target = 4 * 80 = 320; half-span = (45 - (-285))/2 = 165; sum = 485
        r = mod.compute_group_orbit_radius(
            positions=positions,
            target_cm=80.0,
            base_r_factor=4.0,
            max_radius_cm=1000.0,
        )
        self.assertAlmostEqual(r, 485.0)

    def test_orbit_radius_clamps_to_max(self):
        mod = load_module()
        positions = [
            {"name": "a", "x": -285.0, "y": 80.0, "half_extent_cm": 40.0},
            {"name": "b", "x":  45.0, "y": 80.0, "half_extent_cm": 40.0},
        ]

        r = mod.compute_group_orbit_radius(
            positions=positions,
            target_cm=80.0,
            base_r_factor=4.0,
            max_radius_cm=300.0,
        )
        self.assertEqual(r, 300.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_apartment.LineupPositionTests tests.test_render_in_apartment.GroupOrbitRadiusTests -v`
Expected: 4 FAILs (`compute_lineup_positions` / `compute_group_orbit_radius` not found).

- [ ] **Step 3: Implement the helpers**

Insert into `examples/render_in_apartment.py` after `compute_orbit_pose`:

```python
def compute_lineup_positions(
    *,
    animals,
    metas,
    target_cm,
    gap_cm,
    center_x,
    center_y,
):
    names = list(animals)
    for name in names:
        if name not in metas:
            raise KeyError(f"Missing meta for animal {name!r}")
    half_extent = float(target_cm) / 2.0
    spacing = float(target_cm) + float(gap_cm)
    n = len(names)
    start_offset = -spacing * (n - 1) / 2.0
    positions = []
    for i, name in enumerate(names):
        positions.append(
            {
                "name": name,
                "x": float(center_x) + start_offset + i * spacing,
                "y": float(center_y),
                "half_extent_cm": half_extent,
            }
        )
    return positions


def compute_group_orbit_radius(
    *,
    positions,
    target_cm,
    base_r_factor,
    max_radius_cm,
):
    if not positions:
        return min(float(base_r_factor) * float(target_cm), float(max_radius_cm))
    xs = [float(p["x"]) for p in positions]
    half_span = (max(xs) - min(xs)) / 2.0
    ideal = float(base_r_factor) * float(target_cm) + half_span
    return min(ideal, float(max_radius_cm))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_apartment -v`
Expected: `Ran 24 tests ... OK`.

- [ ] **Step 5: Verify** — grep for both symbols:

Run: `grep -n "compute_lineup_positions\|compute_group_orbit_radius" /data/jzy/code/SPEAR/examples/render_in_apartment.py`
Expected: 2+ hits (definitions).

---

## Task 6: `--mode group` renderer (integration)

**Files:**
- Modify: `/data/jzy/code/SPEAR/examples/render_in_apartment.py` (add `render_group`, extend `parse_args`, dispatch in `main`)
- Test: `/data/jzy/code/SPEAR/tests/test_render_in_apartment.py` (CLI-parse tests only; render is exercised by real run in Task 8)

**Interfaces:**
- Consumes: `configure_instance`, `spawn_camera`, `clear_removable_furniture`, `sample_ground_z`, `compute_asset_fit`, `compute_bounds_lift`, `get_actor_bounds_bottom_z`, `spawn_fill_light`, `compute_orbit_pose`, `compute_lineup_positions`, `compute_group_orbit_radius`, `write_checklist`, `animal_bp_path`, `animal_meta_path`, `read_frame`, `clean_frames`, `build_output_dir` (last one gets a `group` name)
- Produces:
  - `render_group(args)` function
  - CLI: `--mode group --animals cat,dog,goose,yak --gap-cm 30.0 --group-name group`
  - Output dir: `<output_root>/render_animals_apartment/group/` (see Step 3 below re: output dir override)

- [ ] **Step 1: Extend CLI test for group mode**

Append to `tests/test_render_in_apartment.py`:

```python
class GroupModeCliTests(unittest.TestCase):
    def test_group_mode_parses_animals_list_and_defaults(self):
        mod = load_module()

        args = mod.parse_args(
            ["--mode", "group", "--animals", "cat,dog,goose,yak"]
        )

        self.assertEqual(args.mode, "group")
        self.assertEqual(args.animals, ["cat", "dog", "goose", "yak"])
        self.assertEqual(args.gap_cm, 30.0)
        self.assertEqual(args.group_name, "group")

    def test_group_mode_rejects_unknown_animal_in_list(self):
        mod = load_module()

        with self.assertRaises(SystemExit):
            mod.parse_args(["--mode", "group", "--animals", "cat,unicorn"])

    def test_animals_group_output_root_default(self):
        mod = load_module()

        args = mod.parse_args([])

        # Solo output root remains DEFAULT_TMP_ROOT for backward compat
        self.assertEqual(args.output_root, mod.DEFAULT_TMP_ROOT)
        # Animals subdir constant should exist for the new segments
        self.assertEqual(
            mod.ANIMALS_OUTPUT_SUBDIR, "render_animals_apartment"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_apartment.GroupModeCliTests -v`
Expected: 3 FAILs (`--mode group` invalid choice, `ANIMALS_OUTPUT_SUBDIR` missing).

- [ ] **Step 3: Extend `parse_args` and add module constant**

Near the top of `examples/render_in_apartment.py` (with the other module constants), add:

```python
ANIMALS_OUTPUT_SUBDIR = "render_animals_apartment"
```

In `parse_args`, change:

```python
    parser.add_argument("--mode", choices=("reference", "turntable"), default="turntable")
```

to:

```python
    parser.add_argument(
        "--mode",
        choices=("reference", "turntable", "group"),
        default="turntable",
    )
```

And, just before the `--animal` argument added in Task 3, insert:

```python
    def _animals_list(value):
        names = [n.strip() for n in str(value).split(",") if n.strip()]
        for n in names:
            if n not in SUPPORTED_ANIMALS:
                raise argparse.ArgumentTypeError(
                    f"Unsupported animal {n!r}; supported = {SUPPORTED_ANIMALS}"
                )
        return names

    parser.add_argument(
        "--animals",
        type=_animals_list,
        default=list(SUPPORTED_ANIMALS),
        help="Comma-separated animal names for --mode group (default: all four).",
    )
    parser.add_argument("--gap-cm", type=float, default=30.0)
    parser.add_argument("--group-name", default="group")
```

- [ ] **Step 4: Run CLI tests to verify they pass**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_apartment -v`
Expected: `Ran 27 tests ... OK`.

- [ ] **Step 5: Implement `render_group`**

Insert this function into `examples/render_in_apartment.py` immediately after `render_turntable`:

```python
def render_group(args):
    import cv2

    output_dir = os.path.join(args.output_root, ANIMALS_OUTPUT_SUBDIR, args.group_name)
    clean_frames(output_dir)

    metas = {}
    for name in args.animals:
        with open(animal_meta_path(args.meta_dir, name), "r", encoding="utf-8") as f:
            metas[name] = json.load(f)

    positions = compute_lineup_positions(
        animals=args.animals,
        metas=metas,
        target_cm=args.target_cm,
        gap_cm=args.gap_cm,
        center_x=args.spawn_x,
        center_y=args.spawn_y,
    )
    radius = compute_group_orbit_radius(
        positions=positions,
        target_cm=args.target_cm,
        base_r_factor=args.r_factor,
        max_radius_cm=args.max_radius_cm,
    )

    instance = configure_instance(rpc_port=args.rpc_port)
    game = instance.get_game()
    spawned_records = []
    try:
        with instance.begin_frame():
            removed = clear_removable_furniture(game=game) if args.clear_furniture else []
        with instance.end_frame():
            pass

        # Spawn all animals + camera in one begin/end pair
        with instance.begin_frame():
            ground_z_by_name = {}
            for pos in positions:
                ground_z, _ = sample_ground_z(
                    game=game,
                    x=pos["x"],
                    y=pos["y"],
                    fallback_z=args.floor_z,
                    trace_start_z=args.ground_trace_start_z,
                    trace_end_z=args.ground_trace_end_z,
                )
                ground_z_by_name[pos["name"]] = ground_z
                fit = compute_asset_fit(
                    meta=metas[pos["name"]],
                    target_cm=args.target_cm,
                    floor_z=ground_z + args.ground_clearance_cm,
                )
                bp = game.unreal_service.load_class(
                    uclass="AActor", name=animal_bp_path(pos["name"])
                )
                actor = game.unreal_service.spawn_actor(
                    uclass=bp,
                    location={"X": pos["x"], "Y": pos["y"], "Z": args.floor_z},
                )
                try:
                    actor.K2_GetRootComponent().SetMobility(NewMobility="Movable")
                except Exception:
                    pass
                game.unreal_service.set_stable_name_for_actor(
                    actor=actor, stable_name=f"MyAssets/group/{pos['name']}"
                )
                spawned_records.append({"pos": pos, "actor": actor, "fit": fit})

            cam, comp = spawn_camera(game=game, width=args.width, height=args.height)
            group_center_z = max(rec["fit"]["center_z"] for rec in spawned_records)
            spawn_fill_light(
                game=game,
                x=args.spawn_x,
                y=args.spawn_y - 90.0,
                z=group_center_z + 140.0,
                intensity_lumens=args.fill_light_lumens,
            )
        with instance.end_frame():
            pass

        instance.step(num_frames=4)
        with instance.begin_frame():
            for rec in spawned_records:
                fit = rec["fit"]
                pos = rec["pos"]
                actor = rec["actor"]
                actor.SetActorScale3D(
                    NewScale3D={"X": fit["scale"], "Y": fit["scale"], "Z": fit["scale"]}
                )
                actor.K2_SetActorLocation(
                    NewLocation={"X": pos["x"], "Y": pos["y"], "Z": fit["actor_z"]},
                    bSweep=False,
                    bTeleport=True,
                )
                bounds_bottom_z, _ = get_actor_bounds_bottom_z(actor=actor)
                lift_cm = compute_bounds_lift(
                    bounds_bottom_z=bounds_bottom_z,
                    ground_z=ground_z_by_name[pos["name"]],
                    clearance_cm=args.ground_clearance_cm,
                    tolerance_cm=args.ground_tolerance_cm,
                )
                if lift_cm > 0.0:
                    fit["actor_z"] += lift_cm
                    fit["center_z"] += lift_cm
                    actor.K2_SetActorLocation(
                        NewLocation={"X": pos["x"], "Y": pos["y"], "Z": fit["actor_z"]},
                        bSweep=False,
                        bTeleport=True,
                    )
                rec["bounds_bottom_z"] = bounds_bottom_z
                rec["lift_cm"] = lift_cm
        with instance.end_frame():
            pass

        center_x = float(args.spawn_x)
        center_y = float(args.spawn_y)
        center_z = max(rec["fit"]["center_z"] for rec in spawned_records)

        print(
            "[apartment-group] "
            f"animals={args.animals} radius={radius:.1f}cm center=({center_x:.1f},{center_y:.1f},{center_z:.1f})",
            flush=True,
        )

        instance.step(num_frames=args.warmup_frames)
        for i in range(args.frames):
            theta = 2.0 * math.pi * i / args.frames
            cam_x = center_x + radius * math.cos(theta)
            cam_y = center_y + radius * math.sin(theta)
            cam_z = center_z + args.cam_z_offset_cm
            yaw = math.degrees(math.atan2(center_y - cam_y, center_x - cam_x))
            pitch = -math.degrees(math.atan2(args.cam_z_offset_cm, radius))
            with instance.begin_frame():
                cam.K2_SetActorLocationAndRotation(
                    NewLocation={"X": cam_x, "Y": cam_y, "Z": cam_z},
                    NewRotation={"Roll": 0.0, "Pitch": pitch, "Yaw": yaw},
                    bSweep=False,
                    bTeleport=True,
                )
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

        # Layout PNG (top-down)
        layout_path = write_group_layout(output_dir, positions, radius, center_x, center_y)
        print(f"LAYOUT_DONE {layout_path}", flush=True)

        # Group checklist
        checklist = {
            "name": args.group_name,
            "animals": list(args.animals),
            "gap_cm": float(args.gap_cm),
            "target_cm": float(args.target_cm),
            "radius_cm": float(radius),
            "center": {"x": center_x, "y": center_y, "z": center_z},
            "per_animal": [
                {
                    "name": rec["pos"]["name"],
                    "x": rec["pos"]["x"],
                    "y": rec["pos"]["y"],
                    "scale": rec["fit"]["scale"],
                    "ground_z_cm": ground_z_by_name[rec["pos"]["name"]],
                    "bounds_bottom_z_cm": rec["bounds_bottom_z"],
                    "lift_applied_cm": rec["lift_cm"],
                }
                for rec in spawned_records
            ],
            "removed_furniture_count": len(removed),
        }
        checklist_path = write_checklist(output_dir, checklist)
        print(f"CHECKLIST_DONE {checklist_path}", flush=True)
    finally:
        instance.close(force=True)
```

Note: this references `write_group_layout` — defined in Task 7.

Also update `main`:

```python
def main(argv=None):
    args = parse_args(argv)
    if args.mode == "reference":
        render_reference(args)
    elif args.mode == "group":
        render_group(args)
    else:
        render_turntable(args)
```

- [ ] **Step 6: Verify** — grep for the new pieces:

Run: `grep -n "render_group\|ANIMALS_OUTPUT_SUBDIR\|--animals" /data/jzy/code/SPEAR/examples/render_in_apartment.py`
Expected: 4+ hits.

---

## Task 7: `write_group_layout` — top-down layout PNG

**Files:**
- Modify: `/data/jzy/code/SPEAR/examples/render_in_apartment.py`
- Test: `/data/jzy/code/SPEAR/tests/test_render_in_apartment.py`

**Interfaces:**
- Consumes: `positions` (list from `compute_lineup_positions`), `radius`, `center_x`, `center_y`
- Produces: `write_group_layout(output_dir, positions, radius_cm, center_x, center_y) -> str` (returns file path)

Uses matplotlib (headless: `matplotlib.use("Agg")`) to plot animal x-positions as labeled circles + camera orbit circle. Written to `output_dir/layout.png`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_render_in_apartment.py`:

```python
class GroupLayoutTests(unittest.TestCase):
    def test_write_group_layout_creates_png(self):
        mod = load_module()
        positions = [
            {"name": "cat", "x": -285.0, "y": 80.0, "half_extent_cm": 40.0},
            {"name": "yak", "x": 45.0, "y": 80.0, "half_extent_cm": 40.0},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = mod.write_group_layout(
                tmp, positions, radius_cm=485.0, center_x=-120.0, center_y=80.0
            )
            self.assertEqual(path, os.path.join(tmp, "layout.png"))
            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 500)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_apartment.GroupLayoutTests -v`
Expected: 1 FAIL (`write_group_layout` not defined).

- [ ] **Step 3: Implement**

Insert into `examples/render_in_apartment.py` (near the other pure helpers, e.g., after `compute_group_orbit_radius`):

```python
def write_group_layout(output_dir, positions, radius_cm, center_x, center_y):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    # Camera orbit
    theta = [2.0 * math.pi * i / 128 for i in range(129)]
    ax.plot(
        [float(center_x) + float(radius_cm) * math.cos(t) for t in theta],
        [float(center_y) + float(radius_cm) * math.sin(t) for t in theta],
        linestyle="--",
        color="tab:blue",
        label=f"camera orbit r={float(radius_cm):.0f}cm",
    )
    # Animals
    for pos in positions:
        ax.add_patch(
            plt.Circle(
                (float(pos["x"]), float(pos["y"])),
                float(pos["half_extent_cm"]),
                fill=True,
                alpha=0.4,
                color="tab:orange",
            )
        )
        ax.text(
            float(pos["x"]),
            float(pos["y"]),
            pos["name"],
            ha="center",
            va="center",
            fontsize=9,
        )
    ax.plot([float(center_x)], [float(center_y)], marker="+", color="k", label="orbit center")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (cm, UE world)")
    ax.set_ylabel("Y (cm, UE world)")
    ax.set_title("apartment_0000 line-up layout (top-down)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "layout.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m unittest tests.test_render_in_apartment -v`
Expected: `Ran 28 tests ... OK`.

- [ ] **Step 5: Verify** — matplotlib in spear-env:

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -c "import matplotlib; print(matplotlib.__version__)"`
Expected: version string (e.g., `3.x.x`). If missing, `pip install matplotlib` into `spear-env`.

---

## Task 8: Breakpoint 1 — render `cat` solo

**Files:**
- Read: `/data/jzy/code/SPEAR/tmp/render_animals_apartment/cat/turntable.mp4` (new artifact)
- Read: `/data/jzy/code/SPEAR/tmp/render_animals_apartment/cat/frame_0000.png`
- Read: `/data/jzy/code/SPEAR/tmp/render_animals_apartment/cat/checklist.json`

**Interfaces:**
- Consumes: everything from Tasks 1–4
- Produces: 1 mp4, 1 png, 1 json → user reviews and answers "OK" or "re-render with X changed"

- [ ] **Step 1: Sanity-check spear-env before render**

Run:
```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -c "import spear; print(spear.__can_import_spear_ext__)"
```
Expected: `True`. If `False`, stop and report.

- [ ] **Step 2: Confirm Xvfb is running on :99**

Run: `pgrep -af "Xvfb :99" | head -1`
Expected: one line showing an `Xvfb :99 -screen 0 ...` process. If absent, start one: `Xvfb :99 -screen 0 1280x720x24 &` (background).

- [ ] **Step 3: Render cat solo**

Run:
```bash
cd /data/jzy/code/SPEAR && \
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python examples/render_in_apartment.py \
  --mode turntable --animal cat \
  --output-root /data/jzy/code/SPEAR/tmp/render_animals_apartment_solo \
  2>&1 | tail -40
```

Note the solo output currently lands at `<output_root>/render_apartment_<name>/` due to `build_output_dir`. For consistency with the group segment layout (`render_animals_apartment/<name>/`), after the render completes:

Run:
```bash
mkdir -p /data/jzy/code/SPEAR/tmp/render_animals_apartment/cat && \
mv /data/jzy/code/SPEAR/tmp/render_animals_apartment_solo/render_apartment_cat/* \
   /data/jzy/code/SPEAR/tmp/render_animals_apartment/cat/ && \
rmdir /data/jzy/code/SPEAR/tmp/render_animals_apartment_solo/render_apartment_cat \
      /data/jzy/code/SPEAR/tmp/render_animals_apartment_solo
```

Expected: Final output at `/data/jzy/code/SPEAR/tmp/render_animals_apartment/cat/{turntable.mp4,frame_0000.png,checklist.json}` (36 frame PNGs also present).

- [ ] **Step 4: Print checklist for user**

Run: `cat /data/jzy/code/SPEAR/tmp/render_animals_apartment/cat/checklist.json`
Expected: JSON with `ground_ok: true` and `|penetration_after_lift_cm| < 0.5`. If `ground_ok: false`, note it in the report but proceed (Q9 error policy is stop-and-ask on human review, not on auto-checklist).

- [ ] **Step 5: Show artifacts and STOP for user review**

Post to the user:
- `frame_0000.png` path (they can preview)
- `turntable.mp4` path
- `checklist.json` contents (pretty-printed)
- Whether the checklist auto-fields all pass
- Explicit request: "Approve → say `next`; re-render → say what to change."

**DO NOT proceed to Task 9 until the user says `next`, `ok`, or an equivalent.** If they request changes, apply them and re-run Step 3 before showing again.

---

## Task 9: Breakpoint 2 — render `dog` solo

**Files:** analogous to Task 8, `dog` substituted throughout.

- [ ] **Step 1: Render dog**

Run:
```bash
cd /data/jzy/code/SPEAR && \
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python examples/render_in_apartment.py \
  --mode turntable --animal dog \
  --output-root /data/jzy/code/SPEAR/tmp/render_animals_apartment_solo \
  2>&1 | tail -40
```

- [ ] **Step 2: Move to canonical output dir**

Run:
```bash
mkdir -p /data/jzy/code/SPEAR/tmp/render_animals_apartment/dog && \
mv /data/jzy/code/SPEAR/tmp/render_animals_apartment_solo/render_apartment_dog/* \
   /data/jzy/code/SPEAR/tmp/render_animals_apartment/dog/ && \
rmdir /data/jzy/code/SPEAR/tmp/render_animals_apartment_solo/render_apartment_dog \
      /data/jzy/code/SPEAR/tmp/render_animals_apartment_solo
```

- [ ] **Step 3: Print checklist and STOP for user review**

Run: `cat /data/jzy/code/SPEAR/tmp/render_animals_apartment/dog/checklist.json`

Post artifacts to user with the same "approve → `next`" prompt as Task 8 Step 5.

---

## Task 10: Breakpoint 3 — render `goose` solo

Same shape as Task 9, `dog` → `goose`.

- [ ] **Step 1: Render goose**

Run:
```bash
cd /data/jzy/code/SPEAR && \
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python examples/render_in_apartment.py \
  --mode turntable --animal goose \
  --output-root /data/jzy/code/SPEAR/tmp/render_animals_apartment_solo \
  2>&1 | tail -40
```

- [ ] **Step 2: Move + print checklist + STOP** — as Task 9 Steps 2–3, with `goose`.

---

## Task 11: Breakpoint 4 — render `yak` solo

Same shape, `goose` → `yak`.

- [ ] **Step 1: Render yak** — command as Task 10 Step 1 with `--animal yak`.

- [ ] **Step 2: Move + print checklist + STOP** — as Task 9 Steps 2–3, with `yak`.

**Note about yak:** its meta has `height=86.6cm` (already smallish because bbox `ext` is diagonal-dominated by body length). At `target_cm=80.0`, yak should look proportionate. Watch for the checklist `bounds_bottom_z_cm` — if it's off by > 5 cm from ground, note in the user-facing report.

---

## Task 12: Breakpoint 5 — render `group` line-up

**Files:**
- Read: `/data/jzy/code/SPEAR/tmp/render_animals_apartment/group/{turntable.mp4,frame_0000.png,layout.png,checklist.json}`

- [ ] **Step 1: Render group**

Run:
```bash
cd /data/jzy/code/SPEAR && \
DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python examples/render_in_apartment.py \
  --mode group --animals cat,dog,goose,yak \
  --output-root /data/jzy/code/SPEAR/tmp \
  2>&1 | tail -40
```

The group mode writes directly to `<output_root>/render_animals_apartment/group/` (no rename step needed).

Expected: `VIDEO_DONE`, `LAYOUT_DONE`, `CHECKLIST_DONE` all print.

- [ ] **Step 2: Sanity-check output artifacts**

Run:
```bash
ls -la /data/jzy/code/SPEAR/tmp/render_animals_apartment/group/
```
Expected: `turntable.mp4`, `layout.png`, `checklist.json`, `frame_0000.png` through `frame_0035.png`.

- [ ] **Step 3: Print checklist for user**

Run: `cat /data/jzy/code/SPEAR/tmp/render_animals_apartment/group/checklist.json`

Verify each `per_animal` entry has `lift_applied_cm < 1.0` and consistent `scale` (~0.401 for all four since they share `ext ≈ 199` at `target_cm=80`).

- [ ] **Step 4: Show artifacts and STOP for user review**

Post:
- `layout.png` (top-down layout preview)
- `frame_0000.png`
- `turntable.mp4`
- `checklist.json` pretty-printed
- Extra 2 items for group review:
  - "4 只都在画面里吗？" (all four visible?)
  - "相对大小合理吗？(yak 应最大，cat 最小)"
- "Approve → `done`; re-render → say what to change."

---

## Task 13: Wrap-up and hand off to GPURIR round

- [ ] **Step 1: Summarize all 5 segments in one message**

Once user has approved segment 5, compile a summary:

```
Segments completed:
  cat    → tmp/render_animals_apartment/cat/turntable.mp4   (checklist: ground_ok=<T/F>, penetration=<N>cm)
  dog    → tmp/render_animals_apartment/dog/turntable.mp4   (...)
  goose  → tmp/render_animals_apartment/goose/turntable.mp4 (...)
  yak    → tmp/render_animals_apartment/yak/turntable.mp4   (...)
  group  → tmp/render_animals_apartment/group/turntable.mp4 (layout.png attached)

Verdict: Hunyuan3D animals load, render with apartment lighting, sit correctly
on the traced floor, and orbit without penetration. Ready for next round.

Next round (proposed, NOT started): GPURIR shoebox room 5.2m x 4.4m x 2.8m
with apartment_0000 materials applied to purpose-built mesh (not Cube-scaled).
Deliverable = one new plan file + one preview render.
```

- [ ] **Step 2: Update TodoWrite / mark this plan complete**

Mark all 5 breakpoint todos as `completed`. Add a new pending todo `Write plan for GPURIR shoebox room round (separate spec)` if user confirms they want to proceed.

- [ ] **Step 3: Ask user to greenlight the next round**

"All 5 approved. Start planning the GPURIR room round? (Y/N)"

---

## Self-Review

### 1. Spec coverage

- Q1 scope (animals in apartment only) → Tasks 8–12
- Q2 animals (cat/dog/goose/yak) → constant `SUPPORTED_ANIMALS` in Task 2
- Q3 videos (4 solo + 1 group) → Tasks 8, 9, 10, 11, 12
- Q4 layout (line-up) → Task 5 `compute_lineup_positions`
- Q5 order+spacing (small→large, bbox+gap) → Task 5 signature, Task 6 uses `args.animals` in list order
- Q6 camera (360° around midpoint) → Task 6 `render_group` orbit loop, Task 5 `compute_group_orbit_radius`
- Q7 video spec (1280×720/12fps/36/3s) → sourced from existing defaults, no changes
- Q8 session (single load per SEGMENT — see design note) → per-segment invocation (revised from "one load for all 5" because Q9 stop-and-ask makes holding session infeasible; explicit in HANDOFF §3)
- Q9 error policy (stop-and-ask each breakpoint) → each breakpoint task explicitly ends with "STOP for user review"
- Q10 spawn position → Task 6 uses `args.spawn_x/y` which default to Clock's `-120, 80`
- Q11 checklist → Task 4 `build_solo_checklist` + Task 6 group checklist section
- Q12 output layout → Task 8–12 move/write to `tmp/render_animals_apartment/<seg>/`
- Q13 TDD → every helper (Tasks 2, 4, 5, 7) has failing-test-first
- Q14 spec doc → Task 1
- Q15 no worktree/subagent → plan runs in main session

**Gap:** Q8 was "single session for all 5"; plan uses "session per segment". This is a deliberate revision documented in HANDOFF §3 and the Global Constraints. The pause between breakpoints (Q9) makes holding a single session for potentially minutes/hours untenable — GPU stays locked and SpearSim can lose xvfb. If user challenges this, offer to run all 5 in one session without pauses as an alternate mode.

### 2. Placeholder scan

Grep results: no `TBD`, `TODO`, `implement later`, `similar to Task`, `add appropriate error handling`, or `fill in details` in the plan body. All code steps include the exact code to insert.

### 3. Type consistency

- `compute_lineup_positions` returns `list[dict]` with keys `name`, `x`, `y`, `half_extent_cm` — used identically in Task 5 tests, Task 6 `render_group`, Task 7 `write_group_layout`.
- `compute_group_orbit_radius` returns `float` — used as `radius` scalar in Task 6.
- `build_solo_checklist` and `write_checklist` signatures match across Task 4 tests + Task 4 wiring + Task 6 group checklist writer.
- CLI flag names consistent: `--animal` (singular, Task 3), `--animals` (plural, Task 6), `--gap-cm`, `--group-name`, `--mode {reference,turntable,group}`.
- BP path template `/Game/MyAssets/Audioset/Blueprints/{name}/BP_{name}.BP_{name}_C` matches what's on disk (verified via `ls` in exploration phase).
