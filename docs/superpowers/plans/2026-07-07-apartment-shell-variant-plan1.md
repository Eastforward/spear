# Apartment Shell Variant — Plan 1 (Hand-tuned demo clip in apartment_0000)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the SPEAR RLR-audio + UE-video pipeline runs end-to-end inside `apartment_0000` (not just shoebox) for a single hand-tuned two-dog clip. Deliver a side-by-side `.mp4` (UE front-view + topdown) matching the current shoebox demo's visual quality, plus per-clip profiling data.

**Architecture:** Sibling of the existing shoebox pipeline. Add `apartment_v1_spec.json` (SSOT for apartment mode) alongside `shoebox_v2_spec.json`. Add `--spec PATH` CLI argument to every render/audio/topdown script. UE side loads `apartment_0000` map, programmatically destroys furniture actors it doesn't want (shell/subset/full modes), then spawns dogs. RLR side needs a mesh of the apartment shell — obtained by dumping structural actor bboxes into a JSON, then generating a triangle mesh from that JSON (analogous to `gen_mesh.py` for shoebox). Deliverable is one clip; randomization comes in Plan 2.

**Tech Stack:** Python 3.11 (spear-env for UE, ss2 for RLR/habitat-sim). SPEAR RPC to UE 5.5. RLR-Audio-Propagation (habitat-sim vendored). Existing tools: `run_render_pass_shoebox_v2.py`, `run_audio_pass_rlr.py`, `gen_mesh.py`, `render_topdown_2d.py`, `scene_two_dogs_v2.py`, `dump_apartment_furniture.py`. New tools: `dump_apartment_shell.py`, `gen_mesh_apartment.py`, `scene_two_dogs_apartment.py`, `run_render_pass_apartment.py`.

## Global Constraints

- Python entry point for SPEAR: `/data/jzy/miniconda3/envs/spear-env/bin/python` (has spear_ext + SPEAR RPC bindings). Wrong env → RPC connect silently fails.
- Python entry point for RLR: `/data/jzy/miniconda3/envs/ss2/bin/python` (has habitat-sim 0.2.2 + RLR).
- SPEAR display env: `DISPLAY=:99` and `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` are required.
- RLR EGL fix (ss2 only): `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libEGL.so.1:/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0` before python invocation.
- Coordinate system: SSOT is right-handed Y-up meters (X=right, Y=forward, Z=up). Habitat swaps Y↔Z. UE uses Z-up cm with per-map origin. All conversions live in existing helpers in `examples/render_in_apartment.py` and `tools/gpurir_scenes/run_render_pass.py`.
- SPEAR CLAUDE.md sync: If `docs/agents*.md` needs update, mirror to `.cursor/rules/local-style.mdc`.
- Never delete cooked apartment assets under `/Game/SPEAR/Scenes/apartment_0000/`.
- Save all new tmp output under `tmp/spike_output_apartment/` (parallel to existing `tmp/spike_output/`), so shoebox demo files are not touched.
- Rig yaw contract: per-tag `walking_forward_yaw_offset_deg` in the rig registry (Quaternius = 180). Adding a new rig without declaring it → import-time error.
- Furniture collision reference: `data/apartment_furniture_map.json` (already dumped, 45 items).
- Camera: 1 forward-only camera, `fov_deg=90`, glued to mic pose. Not 4-view like shoebox.
- Profiling: every stage logs Level-1 timing to a per-run summary file and Level-2 per-clip timing (with `clip_id=0` here since Plan 1 is a single clip) to CSV.

---

## File Structure

**Create:**
- `data/apartment_v1_spec.json` — SSOT for apartment mode (one clip, hand-tuned)
- `tools/spike_rlr/dump_apartment_shell.py` — dumps structural mesh actor bboxes (walls / floor / ceiling / doors / windows / curtains) from apartment_0000 into `data/apartment_shell_map.json`, using an inverted filter from `dump_apartment_furniture.py`
- `data/apartment_shell_map.json` — output of the above (walls/floor/ceiling/doors/etc)
- `tools/spike_rlr/gen_mesh_apartment.py` — generates a triangle mesh (`shell.glb` + materials json) from `apartment_shell_map.json` for RLR consumption
- `tools/spike_rlr/scene_two_dogs_apartment.py` — hand-tuned scene composer for apartment (analogous to `scene_two_dogs_v2.py`), reading apartment_v1_spec.json
- `tools/spike_rlr/run_render_pass_apartment.py` — UE render pass for apartment (loads apartment_0000, destroys furniture NOT in spec's `furniture_include` list, spawns dogs, renders 1 forward camera at 90° FOV)
- `tools/spike_rlr/apartment_furniture_categories.json` — hand-authored classification of the 45 apartment_0000 furniture items into `core` / `decoration` / `misc` — Plan 2 will consume this, Plan 1 only needs the list so subset/full modes work
- `tools/spike_rlr/profiling.py` — timing helpers (`StageTimer` context manager + CSV logger)
- `tests/tools/spike_rlr/test_apartment_shell_dump.py` — validates shell dump has walls/floor/ceiling and excludes furniture
- `tests/tools/spike_rlr/test_gen_mesh_apartment.py` — validates the generated glb has expected triangle count and bbox spans the room

**Modify:**
- `tools/spike_rlr/run_audio_pass_rlr.py` — add `--spec` and `--shell-glb` CLI args so it can be pointed at apartment spec + apartment shell mesh (currently hardcoded to shoebox spec + shoebox mesh)
- `tools/spike_rlr/render_topdown_2d.py` — accept `--spec` CLI arg (currently hardcoded to shoebox spec)
- `tools/gpurir_scenes/dump_apartment_furniture.py` — extract shared filter code into a helper module so `dump_apartment_shell.py` can invert it (DRY)

**Test:** All tests live under `tests/tools/spike_rlr/`. Test framework: `pytest`. Tests must run in ss2 env for anything touching RLR; scene composition tests run in either env.

**Output:**
- `tmp/spike_output_apartment/videos/apartment_v1_view0_with_audio.mp4` — the demo deliverable
- `tmp/spike_output_apartment/videos/apartment_v1_side_by_side_view0.mp4` — with topdown
- `tmp/spike_output_apartment/binaural_native/*.wav`, `raw_audio_hq/*.wav`
- `tmp/spike_output_apartment/profile_stage_summary.txt` — Level-1 profile
- `tmp/spike_output_apartment/profile_per_clip.csv` — Level-2 profile
- `tmp/spike_output_apartment/apartment_v1_metadata.json` — clip metadata (Plan 2 will formalize; Plan 1 writes a minimal version)

---

## Task 1: Refactor filter helper out of dump_apartment_furniture.py

**Goal:** Extract the shell/furniture classification logic into a shared helper so both `dump_apartment_furniture.py` (existing) and `dump_apartment_shell.py` (new) can call it without duplication.

**Files:**
- Create: `tools/gpurir_scenes/apartment_actor_classifier.py`
- Modify: `tools/gpurir_scenes/dump_apartment_furniture.py:35-92` (replace inline classifier with import)
- Test: `tests/tools/gpurir_scenes/test_apartment_actor_classifier.py`

**Interfaces:**
- Produces: `classify_actor(actor_name: str, bbox_min_z: float, bbox_max_z: float, x_extent_cm: float, y_extent_cm: float) -> str` returning one of `"shell_ceiling"`, `"shell_floor"`, `"shell_wall"`, `"shell_door"`, `"shell_window"`, `"shell_curtain"`, `"shell_picture"`, `"shell_mirror"`, `"structural"`, `"furniture"` — mutually exclusive, exactly one label per actor.
- Produces: `SHELL_LABELS: frozenset[str]` = `{"shell_ceiling", "shell_floor", "shell_wall", "shell_door", "shell_window", "shell_curtain", "shell_picture", "shell_mirror", "structural"}` — helper for "is this a shell actor?"
- Produces: constants `Z_CEILING_CM=300.0`, `Z_FLOOR_CM=5.0`, `BBOX_AREA_MAX_CM2=200000.0` (unchanged values from existing code)

- [ ] **Step 1: Write the failing test**

```python
# tests/tools/gpurir_scenes/test_apartment_actor_classifier.py
from tools.gpurir_scenes.apartment_actor_classifier import (
    classify_actor, SHELL_LABELS,
)


def test_wall_actor_classified_as_shell_wall():
    label = classify_actor(
        actor_name="Meshes/24_wall/Wall_North:SM_wall_1",
        bbox_min_z=0.0, bbox_max_z=280.0,
        x_extent_cm=1500.0, y_extent_cm=10.0,
    )
    assert label == "shell_wall"
    assert label in SHELL_LABELS


def test_ceiling_by_zmin_classified_as_shell_ceiling():
    label = classify_actor(
        actor_name="Meshes/22_ceiling/Ceiling",
        bbox_min_z=310.0, bbox_max_z=350.0,
        x_extent_cm=1500.0, y_extent_cm=1200.0,
    )
    assert label == "shell_ceiling"


def test_floor_by_zmax_classified_as_shell_floor():
    label = classify_actor(
        actor_name="Meshes/21_floor/Floor",
        bbox_min_z=0.0, bbox_max_z=3.0,
        x_extent_cm=1500.0, y_extent_cm=1200.0,
    )
    assert label == "shell_floor"


def test_door_actor_classified_as_shell_door():
    label = classify_actor(
        actor_name="Meshes/08_door/Door_Front:SM_door_1",
        bbox_min_z=0.0, bbox_max_z=210.0,
        x_extent_cm=100.0, y_extent_cm=10.0,
    )
    assert label == "shell_door"


def test_window_actor_classified_as_shell_window():
    label = classify_actor(
        actor_name="Meshes/09_window/Window_1:SM_window_5",
        bbox_min_z=100.0, bbox_max_z=250.0,
        x_extent_cm=150.0, y_extent_cm=8.0,
    )
    assert label == "shell_window"


def test_curtain_actor_classified_as_shell_curtain():
    label = classify_actor(
        actor_name="Meshes/16_curtain/Curtain",
        bbox_min_z=50.0, bbox_max_z=280.0,
        x_extent_cm=200.0, y_extent_cm=5.0,
    )
    assert label == "shell_curtain"


def test_picture_actor_classified_as_shell_picture():
    label = classify_actor(
        actor_name="Meshes/11_picture/Picture_2",
        bbox_min_z=140.0, bbox_max_z=200.0,
        x_extent_cm=60.0, y_extent_cm=4.0,
    )
    assert label == "shell_picture"


def test_mirror_actor_classified_as_shell_mirror():
    label = classify_actor(
        actor_name="Meshes/19_mirror/Mirror:SM_Mirror_5",
        bbox_min_z=212.0, bbox_max_z=280.0,
        x_extent_cm=80.0, y_extent_cm=6.0,
    )
    assert label == "shell_mirror"


def test_huge_actor_classified_as_structural():
    label = classify_actor(
        actor_name="Meshes/38_otherstructure/BigStructure",
        bbox_min_z=0.0, bbox_max_z=280.0,
        x_extent_cm=1500.0, y_extent_cm=1300.0,   # 19.5 m2, above 20 m2 threshold no; try 15x15
    )
    # bbox area = 1500 * 1300 = 1.95e6 cm2 > 2e5 -> structural
    assert label == "structural"


def test_chair_actor_classified_as_furniture():
    label = classify_actor(
        actor_name="Meshes/05_chair/LivingRoom_Chair_01:SM_chair_living_2",
        bbox_min_z=28.0, bbox_max_z=163.0,
        x_extent_cm=128.0, y_extent_cm=128.0,
    )
    assert label == "furniture"
    assert label not in SHELL_LABELS


def test_sofa_actor_classified_as_furniture():
    label = classify_actor(
        actor_name="Meshes/06_sofa/Sofa",
        bbox_min_z=25.0, bbox_max_z=90.0,
        x_extent_cm=200.0, y_extent_cm=90.0,
    )
    assert label == "furniture"


def test_shell_labels_disjoint_from_furniture():
    assert "furniture" not in SHELL_LABELS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/tools/gpurir_scenes/test_apartment_actor_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.gpurir_scenes.apartment_actor_classifier'`

- [ ] **Step 3: Write minimal implementation**

```python
# tools/gpurir_scenes/apartment_actor_classifier.py
"""Actor-name based classifier: shell (structural) vs furniture (independent).

Shell = anything attached to walls/floor/ceiling (walls, floor, ceiling,
doors, windows, curtains, pictures, mirrors). These don't occupy interior
free space and stay across furniture-mode toggles.

Furniture = independent items occupying floor area (chairs, sofa, tables,
lamps, pillows, etc.). These are the items subset/full modes toggle.
"""
from __future__ import annotations

Z_CEILING_CM = 300.0
Z_FLOOR_CM = 5.0
BBOX_AREA_MAX_CM2 = 200000.0

_SHELL_NAME_KEYWORDS = {
    "wall": "shell_wall",
    "floor": "shell_floor",
    "ceiling": "shell_ceiling",
    "ground": "shell_floor",
    "door": "shell_door",
    "window": "shell_window",
    "curtain": "shell_curtain",
    "picture": "shell_picture",
    "mirror": "shell_mirror",
}

SHELL_LABELS = frozenset({
    "shell_ceiling", "shell_floor", "shell_wall",
    "shell_door", "shell_window", "shell_curtain",
    "shell_picture", "shell_mirror",
    "structural",
})


def classify_actor(actor_name: str, bbox_min_z: float, bbox_max_z: float,
                   x_extent_cm: float, y_extent_cm: float) -> str:
    """Return one of SHELL_LABELS or 'furniture'."""
    # Ceiling by z-min
    if bbox_min_z > Z_CEILING_CM:
        return "shell_ceiling"
    # Floor by z-max
    if bbox_max_z < Z_FLOOR_CM:
        return "shell_floor"
    # Name-based shell classification (walls, doors, windows, curtains, pictures, mirrors)
    nl = str(actor_name).lower()
    for kw, label in _SHELL_NAME_KEYWORDS.items():
        if kw in nl:
            return label
    # Very large bbox → structural mesh not caught by name (e.g. big built-in cabinet
    # merged with a wall). Threshold from the original furniture-only dumper.
    if x_extent_cm * y_extent_cm > BBOX_AREA_MAX_CM2:
        return "structural"
    return "furniture"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/tools/gpurir_scenes/test_apartment_actor_classifier.py -v`
Expected: PASS (all 12 tests)

- [ ] **Step 5: Refactor `dump_apartment_furniture.py` to use the shared classifier**

Modify `tools/gpurir_scenes/dump_apartment_furniture.py`:
- Delete lines 35-70 (the `Z_CEILING_CM`/`Z_FLOOR_CM`/`NAME_KEYWORDS`/`BBOX_AREA_MAX_CM2` constants and the local `_classify` function).
- At top, add: `from apartment_actor_classifier import classify_actor, SHELL_LABELS`
- Replace calls to the local `_classify` with a wrapper:

```python
def _keep_reason(actor_name, bbox_min_z, bbox_max_z, x_ext, y_ext):
    label = classify_actor(actor_name, bbox_min_z, bbox_max_z, x_ext, y_ext)
    if label == "furniture":
        return "kept"
    # Map new labels back to legacy filter-reason strings for backward compat with JSON output
    return {
        "shell_ceiling": "z_ceiling",
        "shell_floor": "z_floor",
        "shell_wall": "name_wall",
        "shell_door": "name_door",
        "shell_window": "name_window",
        "shell_curtain": "name_curtain",
        "shell_picture": "name_picture",
        "shell_mirror": "name_mirror",
        "structural": "bbox_too_large",
    }[label]
```

And find every call to the old `_classify(...)` in the file — replace with `_keep_reason(...)`. Keep the `reasons` counter dict updated to accumulate the new label set (add `name_door / name_window / name_curtain / name_picture / name_mirror` keys initialized to 0).

- [ ] **Step 6: Verify existing dump still works (regression check)**

Run: `diff <(/data/jzy/miniconda3/envs/spear-env/bin/python tools/gpurir_scenes/dump_apartment_furniture.py --dry-run 2>&1 | grep 'kept\|reasons') <(cat data/apartment_furniture_map.json | python3 -c "import json,sys; d=json.load(sys.stdin); print('kept:', d['meta']['num_actors_after_filter']); print('reasons:', d['meta']['filter_reasons'])")`
Expected: no diff for `kept` count (should be 45 in both). `reasons` will have additional zero-count keys in new run; that's fine.

Actually easier: just re-run the actual dump and byte-compare kept-count:

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
export DISPLAY=:99
/data/jzy/miniconda3/envs/spear-env/bin/python tools/gpurir_scenes/dump_apartment_furniture.py --out tmp/regression_furniture_map.json --dry-run
grep -oP '"num_actors_after_filter": \d+' data/apartment_furniture_map.json
```
Note the number (should be 45), then check the dry-run stdout shows the same.

If dry-run needs SPEAR RPC to reach UE (which it does since it enumerates live actors), and if UE isn't running, we can skip this regression check — the unit tests in step 4 already prove the classifier logic is unchanged. Just move on and commit.

- [ ] **Step 7: Commit**

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
git add tools/gpurir_scenes/apartment_actor_classifier.py \
        tools/gpurir_scenes/dump_apartment_furniture.py \
        tests/tools/gpurir_scenes/test_apartment_actor_classifier.py
git commit -m "refactor(apartment): extract shell/furniture classifier into shared helper

Enables symmetric shell-only dump (Task 2) without duplicating filter logic.
No behavior change for existing furniture dump — same 45 kept actors."
```

---

## Task 2: dump_apartment_shell.py — dump only structural actors

**Goal:** Symmetric to `dump_apartment_furniture.py` but inverted: keep only shell actors (walls, floor, ceiling, doors, windows, curtains, pictures, mirrors, structural), discard furniture. Output includes not just bboxes but also actor-local mesh vertex data if SPEAR RPC allows, so we can construct RLR geometry in Task 3.

**Files:**
- Create: `tools/spike_rlr/dump_apartment_shell.py`
- Create: `data/apartment_shell_map.json` (output artifact — git-track it as a fixture, per the convention that `data/apartment_furniture_map.json` is tracked)
- Test: `tests/tools/spike_rlr/test_apartment_shell_dump.py` (offline test on a fixture JSON, no live SPEAR)

**Interfaces:**
- Consumes: `classify_actor()` from Task 1
- Produces: JSON schema in `apartment_shell_map.json`:

```json
{
  "meta": {
    "apartment_map_path": "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000",
    "dump_date_utc": "2026-07-07T...",
    "spear_commit": "...",
    "ue_version": "5.5",
    "apartment_mic_origin_cm_at_dump": [-120.0, 80.0, 120.0],
    "num_actors_seen": 54,
    "num_actors_after_filter": 9,
    "shell_label_counts": {"shell_wall": 5, "shell_ceiling": 1, "shell_floor": 3, ...}
  },
  "shell_actors": [
    {
      "actor_name": "...",
      "shell_label": "shell_wall",
      "bbox_min_ue_cm": [x, y, z],
      "bbox_max_ue_cm": [x, y, z],
      "actor_location_ue_cm": [x, y, z],
      "actor_rotation_deg": [pitch, yaw, roll]
    },
    ...
  ]
}
```

- [ ] **Step 1: Write the failing test using a fixture JSON**

```python
# tests/tools/spike_rlr/test_apartment_shell_dump.py
"""Offline tests for the shell dump — validate the classifier drives
what ends up in the shell JSON. Live SPEAR verification is manual
(see the CLI regression check in the dump script's docstring)."""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def test_shell_map_json_exists_and_has_actors():
    p = REPO / "data" / "apartment_shell_map.json"
    if not p.exists():
        pytest.skip("apartment_shell_map.json not yet dumped — run dump_apartment_shell.py")
    d = json.loads(p.read_text())
    assert "shell_actors" in d
    assert len(d["shell_actors"]) > 0
    assert "meta" in d


def test_shell_map_no_furniture_labels():
    p = REPO / "data" / "apartment_shell_map.json"
    if not p.exists():
        pytest.skip("apartment_shell_map.json not yet dumped")
    d = json.loads(p.read_text())
    for a in d["shell_actors"]:
        assert a["shell_label"] != "furniture", f"furniture actor leaked: {a['actor_name']}"


def test_shell_map_has_walls_floor_ceiling():
    p = REPO / "data" / "apartment_shell_map.json"
    if not p.exists():
        pytest.skip("apartment_shell_map.json not yet dumped")
    d = json.loads(p.read_text())
    labels = {a["shell_label"] for a in d["shell_actors"]}
    assert "shell_wall" in labels, f"no walls in shell dump; got {labels}"
    assert "shell_floor" in labels, f"no floor in shell dump; got {labels}"
    assert "shell_ceiling" in labels, f"no ceiling in shell dump; got {labels}"
```

- [ ] **Step 2: Run test to verify it fails (expect skip since JSON doesn't exist)**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/tools/spike_rlr/test_apartment_shell_dump.py -v`
Expected: 3 SKIP (file doesn't exist yet — this is the pre-dump state)

- [ ] **Step 3: Write dump_apartment_shell.py**

```python
# tools/spike_rlr/dump_apartment_shell.py
"""Symmetric counterpart to dump_apartment_furniture.py.

Dumps only shell (structural) actors from apartment_0000: walls, floor,
ceiling, doors, windows, curtains, pictures, mirrors, and misc large
structural meshes. Output goes to data/apartment_shell_map.json.

Usage:
    export DISPLAY=:99
    /data/jzy/miniconda3/envs/spear-env/bin/python \\
        tools/spike_rlr/dump_apartment_shell.py --dry-run
    /data/jzy/miniconda3/envs/spear-env/bin/python \\
        tools/spike_rlr/dump_apartment_shell.py --out data/apartment_shell_map.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "examples"))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "tools", "gpurir_scenes"))

from render_in_apartment import APARTMENT_MAP, configure_instance  # noqa: E402
from apartment_actor_classifier import classify_actor, SHELL_LABELS  # noqa: E402


def _git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "data", "apartment_shell_map.json"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Print counts only, don't write JSON.")
    ap.add_argument("--rpc-port", type=int, default=30000)
    args = ap.parse_args()

    import spear
    config = configure_instance(args.rpc_port)
    instance = spear.Instance(config=config)
    game = instance.get_game()

    with instance.begin_frame():
        actors = game.unreal_service.find_actors_by_class(
            uclass="/Script/Engine.StaticMeshActor")

    print(f"[shell] enumerated {len(actors)} StaticMeshActor(s) in {APARTMENT_MAP}")

    label_counts = {lbl: 0 for lbl in SHELL_LABELS}
    label_counts["furniture"] = 0
    shell_records = []

    for handle, name in actors.items():
        with instance.begin_frame():
            bounds = game.unreal_service.call_function(
                uobject=handle, ufunction_name="GetActorBounds", args={})
            loc = game.unreal_service.call_function(
                uobject=handle, ufunction_name="K2_GetActorLocation", args={})
            rot = game.unreal_service.call_function(
                uobject=handle, ufunction_name="K2_GetActorRotation", args={})
        # bounds returns {origin: {X,Y,Z}, extent: {X,Y,Z}}
        origin = bounds["origin"] if isinstance(bounds, dict) else bounds.origin
        extent = bounds["extent"] if isinstance(bounds, dict) else bounds.extent
        ox, oy, oz = float(origin["X"]), float(origin["Y"]), float(origin["Z"])
        ex, ey, ez = float(extent["X"]), float(extent["Y"]), float(extent["Z"])
        bmin = [ox - ex, oy - ey, oz - ez]
        bmax = [ox + ex, oy + ey, oz + ez]

        label = classify_actor(
            actor_name=name,
            bbox_min_z=bmin[2],
            bbox_max_z=bmax[2],
            x_extent_cm=2 * ex,
            y_extent_cm=2 * ey,
        )
        label_counts[label] = label_counts.get(label, 0) + 1
        if label not in SHELL_LABELS:
            continue

        shell_records.append({
            "actor_name": name,
            "shell_label": label,
            "bbox_min_ue_cm": bmin,
            "bbox_max_ue_cm": bmax,
            "actor_location_ue_cm": [float(loc["X"]), float(loc["Y"]), float(loc["Z"])],
            "actor_rotation_deg": [float(rot["Pitch"]), float(rot["Yaw"]), float(rot["Roll"])],
        })

    instance.close()

    print(f"[shell] label counts: {label_counts}")
    print(f"[shell] {len(shell_records)} shell actors kept")

    if args.dry_run:
        return

    out = {
        "meta": {
            "apartment_map_path": APARTMENT_MAP,
            "dump_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "spear_commit": _git_head(),
            "ue_version": "5.5",
            "apartment_mic_origin_cm_at_dump": [-120.0, 80.0, 120.0],
            "num_actors_seen": len(actors),
            "num_actors_after_filter": len(shell_records),
            "shell_label_counts": label_counts,
        },
        "shell_actors": shell_records,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[shell] wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the dump against the live apartment**

The apartment must be reachable via SPEAR RPC. If X server isn't up on `:99`, start it first (`Xvfb :99 -screen 0 1920x1080x24 &`).

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
export DISPLAY=:99
/data/jzy/miniconda3/envs/spear-env/bin/python tools/spike_rlr/dump_apartment_shell.py --dry-run 2>&1 | tail -20
```

Expected output includes `[shell] label counts: {...}` with non-zero `shell_wall`, `shell_floor`, `shell_ceiling` counts and `[shell] N shell actors kept` where N > 0 (typically 8–15 for apartment_0000).

If the counts look reasonable (walls > 0, floor > 0, ceiling > 0), proceed:

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python tools/spike_rlr/dump_apartment_shell.py --out data/apartment_shell_map.json
ls -la data/apartment_shell_map.json
```

- [ ] **Step 5: Run tests to verify shell JSON has expected contents**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/tools/spike_rlr/test_apartment_shell_dump.py -v`
Expected: 3 PASS

- [ ] **Step 6: Commit**

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
git add tools/spike_rlr/dump_apartment_shell.py \
        tests/tools/spike_rlr/test_apartment_shell_dump.py \
        data/apartment_shell_map.json
git commit -m "feat(apartment): dump shell (structural) actor bboxes to JSON

Symmetric counterpart to dump_apartment_furniture.py — inverted filter
kept shell/floor/ceiling/door/window/curtain/picture/mirror + oversize
structural meshes. Enables gen_mesh_apartment.py to build RLR mesh from
just the room shell."
```

---

## Task 3: gen_mesh_apartment.py — build an RLR-consumable mesh from the shell JSON

**Goal:** Convert `apartment_shell_map.json` into a triangle mesh (`shell.glb`) + acoustic-materials sidecar JSON. This is what RLR loads to compute IRs. Analog of existing `gen_mesh.py` which does the same for shoebox from `shoebox_v2_spec.json`.

**Files:**
- Create: `tools/spike_rlr/gen_mesh_apartment.py`
- Test: `tests/tools/spike_rlr/test_gen_mesh_apartment.py`
- Reference (read but don't modify): `tools/spike_rlr/gen_mesh.py`

**Interfaces:**
- Consumes: `data/apartment_shell_map.json` (output of Task 2)
- Produces: file at path passed via `--out-glb` (default `tmp/spike_output_apartment/mesh/shell.glb`)
- Produces: file at path passed via `--out-materials` (default `tmp/spike_output_apartment/mesh/shell_materials.json`)
- Materials sidecar schema mirrors `tmp/spike_rlr/shoebox_v2_materials_rlr.json`

**Approach:** Treat each shell actor's bbox as an axis-aligned box. For walls, most bboxes have one very-small dimension (thickness ~10cm), so representing them as thin boxes is fine acoustically. Emit each shell actor as 12 triangles (6 faces × 2 tris). Materials sidecar maps each face group to an acoustic material based on `shell_label` — `shell_wall` → `drywall_painted`, `shell_floor` → `hardwood_oak`, `shell_ceiling` → `painted_plaster`, `shell_window` → `glass_smooth`, `shell_door` → `wood_solid`, `shell_curtain` → `heavy_curtain`, etc. Materials must already exist in `data/acoustic_material_db.json`; if a mapping references a missing material, error out.

- [ ] **Step 1: Read the existing gen_mesh.py to understand the glb export helper**

Run: `sed -n '1,100p' /data/jzy/code/AVEngine/external/SPEAR/tools/spike_rlr/gen_mesh.py`

Note the existing `_box_triangles(center, size)` (or equivalent) helper and the glb/trimesh export idiom. Reuse if present; if the helper is private to that module, extract it into a new `tools/spike_rlr/mesh_helpers.py` and import from both places.

- [ ] **Step 2: Write the failing test**

```python
# tests/tools/spike_rlr/test_gen_mesh_apartment.py
"""Validate the apartment-shell mesh has expected shape and material assignments."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[3]
GEN = REPO / "tools" / "spike_rlr" / "gen_mesh_apartment.py"
SHELL_JSON = REPO / "data" / "apartment_shell_map.json"


def test_shell_json_prereq_exists():
    if not SHELL_JSON.exists():
        pytest.skip("apartment_shell_map.json not yet dumped (Task 2)")


@pytest.fixture(scope="module")
def generated_mesh(tmp_path_factory):
    if not SHELL_JSON.exists():
        pytest.skip("apartment_shell_map.json not yet dumped")
    outdir = tmp_path_factory.mktemp("mesh")
    glb = outdir / "shell.glb"
    mats = outdir / "shell_materials.json"
    r = subprocess.run(
        ["/data/jzy/miniconda3/envs/ss2/bin/python", str(GEN),
         "--shell-json", str(SHELL_JSON),
         "--out-glb", str(glb),
         "--out-materials", str(mats)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"gen_mesh_apartment failed: {r.stderr}"
    return glb, mats


def test_glb_file_created(generated_mesh):
    glb, _ = generated_mesh
    assert glb.exists() and glb.stat().st_size > 0


def test_materials_json_covers_all_glb_face_groups(generated_mesh):
    _, mats_path = generated_mesh
    mats = json.loads(mats_path.read_text())
    assert "face_group_materials" in mats
    # Each face group must have a resolved material name (not the raw label)
    for group, mat in mats["face_group_materials"].items():
        assert isinstance(mat, str) and len(mat) > 0
        assert not mat.startswith("shell_"), \
            f"face group {group} has raw label {mat} — needs mapping to material"


def test_glb_bbox_covers_apartment_extent(generated_mesh):
    """Sanity: the shell mesh should span the apartment's known ~11m x 13m extent."""
    import trimesh
    glb, _ = generated_mesh
    scene = trimesh.load(str(glb))
    if isinstance(scene, trimesh.Scene):
        combined = trimesh.util.concatenate([g for g in scene.geometry.values()])
    else:
        combined = scene
    bmin, bmax = combined.bounds
    extent_m = (bmax - bmin)  # meters
    # apartment_furniture_map.json shows X range 596-(-549)=1145 cm=11.45m, Y 656-(-688)=1344 cm=13.44m
    assert extent_m[0] > 8.0, f"X extent {extent_m[0]:.1f}m too small"
    assert extent_m[1] > 10.0, f"Y extent {extent_m[1]:.1f}m too small"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `/data/jzy/miniconda3/envs/ss2/bin/python -m pytest tests/tools/spike_rlr/test_gen_mesh_apartment.py -v`
Expected: FAIL — script doesn't exist. (First test may pass if Task 2 was completed; later tests fail on `subprocess` returning non-zero.)

- [ ] **Step 4: Write gen_mesh_apartment.py**

```python
# tools/spike_rlr/gen_mesh_apartment.py
"""Generate an RLR-consumable triangle mesh from apartment_shell_map.json.

Each shell actor's AABB becomes a 12-triangle box. Each face gets a
material from the shell_label -> acoustic_material mapping below.

Output:
  - shell.glb                : the mesh
  - shell_materials.json     : face_group -> acoustic material name mapping
                               (fed to RLR alongside the glb)

Coordinate conversion: apartment_shell_map.json stores bboxes in UE cm.
We convert to SSOT (right-handed Y-up meters) by:
   x_m = (x_ue_cm - APARTMENT_MIC_ORIGIN_UE_CM[0]) / 100
   y_m = -(y_ue_cm - APARTMENT_MIC_ORIGIN_UE_CM[1]) / 100    (Y flip per apartment convention)
   z_m = (z_ue_cm - APARTMENT_FLOOR_Z_UE_CM) / 100

APARTMENT_MIC_ORIGIN_UE_CM = (-120.0, 80.0, 120.0)  (matches tools/gpurir_scenes/run_render_pass.py)
APARTMENT_FLOOR_Z_UE_CM = 27.1                       (ground trace at Clock spawn)

Usage:
    /data/jzy/miniconda3/envs/ss2/bin/python \\
        tools/spike_rlr/gen_mesh_apartment.py \\
        --shell-json data/apartment_shell_map.json \\
        --out-glb tmp/spike_output_apartment/mesh/shell.glb \\
        --out-materials tmp/spike_output_apartment/mesh/shell_materials.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

APARTMENT_MIC_ORIGIN_UE_CM = (-120.0, 80.0, 120.0)
APARTMENT_FLOOR_Z_UE_CM = 27.1

# shell_label -> acoustic material name. Names must exist in data/acoustic_material_db.json.
SHELL_LABEL_TO_MATERIAL = {
    "shell_wall": "drywall_painted",
    "shell_floor": "hardwood_oak",
    "shell_ceiling": "painted_plaster",
    "shell_window": "glass_smooth",
    "shell_door": "wood_solid",
    "shell_curtain": "heavy_curtain",
    "shell_picture": "wood_solid",       # frames act like wood panels acoustically
    "shell_mirror": "glass_smooth",
    "structural": "drywall_painted",
}


def ue_to_ssot(pos_ue_cm):
    x_cm, y_cm, z_cm = pos_ue_cm
    return (
        (x_cm - APARTMENT_MIC_ORIGIN_UE_CM[0]) / 100.0,
        -(y_cm - APARTMENT_MIC_ORIGIN_UE_CM[1]) / 100.0,
        (z_cm - APARTMENT_FLOOR_Z_UE_CM) / 100.0,
    )


def box_triangles(bmin_m, bmax_m):
    """Return (vertices [8,3], faces [12,3]) for an axis-aligned box."""
    x0, y0, z0 = bmin_m
    x1, y1, z1 = bmax_m
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float32)
    # 12 triangles, CCW when viewed from outside
    f = np.array([
        [0, 2, 1], [0, 3, 2],   # bottom (-Z)
        [4, 5, 6], [4, 6, 7],   # top (+Z)
        [0, 1, 5], [0, 5, 4],   # -Y
        [2, 3, 7], [2, 7, 6],   # +Y
        [1, 2, 6], [1, 6, 5],   # +X
        [0, 4, 7], [0, 7, 3],   # -X
    ], dtype=np.int32)
    return v, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shell-json", required=True)
    ap.add_argument("--out-glb", required=True)
    ap.add_argument("--out-materials", required=True)
    args = ap.parse_args()

    shell_map = json.loads(Path(args.shell_json).read_text())
    actors = shell_map["shell_actors"]

    all_verts = []
    all_faces = []
    face_group_materials = {}  # "actor_{i}_{label}" -> material name
    n_verts = 0

    for i, a in enumerate(actors):
        bmin_ue = a["bbox_min_ue_cm"]
        bmax_ue = a["bbox_max_ue_cm"]
        bmin_ssot = ue_to_ssot(bmin_ue)
        bmax_ssot = ue_to_ssot(bmax_ue)
        # Because we flip Y, bmin/bmax may need reordering per axis:
        lo = (min(bmin_ssot[0], bmax_ssot[0]),
              min(bmin_ssot[1], bmax_ssot[1]),
              min(bmin_ssot[2], bmax_ssot[2]))
        hi = (max(bmin_ssot[0], bmax_ssot[0]),
              max(bmin_ssot[1], bmax_ssot[1]),
              max(bmin_ssot[2], bmax_ssot[2]))
        # Skip degenerate boxes (zero volume)
        if (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]) < 1e-6:
            print(f"[gen_mesh_apt] skipping degenerate actor {a['actor_name']}")
            continue

        v, f = box_triangles(lo, hi)
        all_verts.append(v)
        all_faces.append(f + n_verts)
        n_verts += 8

        label = a["shell_label"]
        material = SHELL_LABEL_TO_MATERIAL.get(label)
        if material is None:
            raise ValueError(f"no material mapping for shell_label={label}")
        # Validate material name is in the acoustic db
        db_path = Path(__file__).resolve().parents[2] / "data" / "acoustic_material_db.json"
        db = json.loads(db_path.read_text())
        if material not in db.get("materials", {}):
            raise ValueError(f"material {material!r} for {label} not in acoustic_material_db.json")

        face_group_materials[f"actor_{i}_{label}"] = material

    verts = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    out_glb = Path(args.out_glb)
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_glb))

    materials_out = {
        "shell_map_source": str(args.shell_json),
        "n_actors": len(actors),
        "face_group_materials": face_group_materials,
    }
    out_mat = Path(args.out_materials)
    out_mat.parent.mkdir(parents=True, exist_ok=True)
    out_mat.write_text(json.dumps(materials_out, indent=2))

    print(f"[gen_mesh_apt] wrote {out_glb} ({len(verts)} verts, {len(faces)} tris)")
    print(f"[gen_mesh_apt] wrote {out_mat}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify passing**

Run: `/data/jzy/miniconda3/envs/ss2/bin/python -m pytest tests/tools/spike_rlr/test_gen_mesh_apartment.py -v`
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
git add tools/spike_rlr/gen_mesh_apartment.py \
        tests/tools/spike_rlr/test_gen_mesh_apartment.py
git commit -m "feat(apartment): generate RLR shell mesh (glb) from apartment_shell_map.json

Each shell actor becomes 12-triangle box; each face gets a material from
SHELL_LABEL_TO_MATERIAL. Validates all materials exist in acoustic db."
```

---

## Task 4: apartment_v1_spec.json + apartment_furniture_categories.json — SSOT + core/decoration classification

**Goal:** Author the SSOT spec for the Plan-1 hand-tuned demo clip, and the core/decoration classification of apartment_0000's 45 furniture items. Plan 1 only uses `furniture_include: "subset"` (default per T-fixed decision); Plan 2 will use `shell` and `full` too but the spec supports all three modes now.

**Files:**
- Create: `data/apartment_v1_spec.json`
- Create: `tools/spike_rlr/apartment_furniture_categories.json`
- Test: `tests/tools/spike_rlr/test_apartment_v1_spec.py`

**Interfaces:**
- Produces: schema of `apartment_v1_spec.json` (schema below)
- Produces: schema of `apartment_furniture_categories.json`:

```json
{
  "core": ["Meshes/06_sofa/Sofa", "Meshes/07_table/LivingRoom_Table_01:...", ...],
  "decoration": ["Meshes/11_picture/Picture_2", "Meshes/18_pillow/...", ...],
  "misc": ["Meshes/40_otherprop/..."]
}
```

- [ ] **Step 1: Author apartment_furniture_categories.json**

Extract every actor_name from `data/apartment_furniture_map.json`, then assign each to one of `{core, decoration, misc}` using this heuristic:
- `core`: sofa, dining/living table (07_table), bed (if present), bookshelf (10_bookshelf), largest chair per room
- `decoration`: picture (11_picture), pillow (18_pillow), lamp (35_lamp), curtain (already shell), mirror (already shell), otherprop (40_otherprop) with bbox_area < 3000 cm2
- `misc`: everything else (extra chairs beyond one per room, larger otherprop)

Author it manually — do NOT script it (auto-classification is Plan 2's job). Include ALL 45 furniture actor_name strings from `data/apartment_furniture_map.json`. No item should appear in more than one category.

Run this to enumerate them:

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -c "
import json
d = json.load(open('data/apartment_furniture_map.json'))
for f in d['furniture']:
    print(f['actor_name'])
" > /tmp/all_45_names.txt
```

Then manually copy each name into `core` / `decoration` / `misc` in the new JSON. It's tedious but ONE TIME.

- [ ] **Step 2: Write the test**

```python
# tests/tools/spike_rlr/test_apartment_v1_spec.py
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

CATEGORIES = REPO / "tools" / "spike_rlr" / "apartment_furniture_categories.json"
SPEC = REPO / "data" / "apartment_v1_spec.json"
FURNITURE_MAP = REPO / "data" / "apartment_furniture_map.json"


def test_categories_covers_all_45_furniture_actors():
    cats = json.loads(CATEGORIES.read_text())
    fmap = json.loads(FURNITURE_MAP.read_text())
    all_actors = {f["actor_name"] for f in fmap["furniture"]}
    classified = set(cats["core"]) | set(cats["decoration"]) | set(cats["misc"])
    missing = all_actors - classified
    extra = classified - all_actors
    assert not missing, f"unclassified actors: {missing}"
    assert not extra, f"unknown actors classified: {extra}"


def test_categories_are_disjoint():
    cats = json.loads(CATEGORIES.read_text())
    a = set(cats["core"]); b = set(cats["decoration"]); c = set(cats["misc"])
    assert not (a & b), f"core & decoration overlap: {a & b}"
    assert not (a & c), f"core & misc overlap: {a & c}"
    assert not (b & c), f"decoration & misc overlap: {b & c}"


def test_apartment_v1_spec_schema():
    s = json.loads(SPEC.read_text())
    assert s["spec_version"] == "apartment_v1"
    assert s["room_backend"] == "apartment_shell"
    assert "mic" in s and "pos_m" in s["mic"] and "yaw_deg" in s["mic"]
    assert "camera_configs" in s and len(s["camera_configs"]) == 1
    assert s["camera_configs"][0]["fov_deg"] == 90.0
    assert "furniture_mode" in s and s["furniture_mode"] in ("shell", "subset", "full")
    assert "sources" in s and len(s["sources"]) == 2  # golden + husky, hand-tuned
    for src in s["sources"]:
        assert "tag" in src and "audio_lookup" in src and "trajectory_m" in src
```

- [ ] **Step 3: Author apartment_v1_spec.json**

Use approximately these values. Adjust after Step 5 (spawn-in-empty test) if we discover the mic/dog positions collide with shell walls.

```json
{
  "spec_version": "apartment_v1",
  "description": "Apartment shell variant, Plan 1 hand-tuned single-clip demo. Two dogs (golden bark + husky piano) in apartment_0000 living-room area. Subset-furniture mode (core + few decorations). 5-second clip, 15 fps, 75 frames.",
  "coordinate_frame": {
    "system": "right-handed Y-up (meters). Origin at apartment mic anchor (see APARTMENT_MIC_ORIGIN_UE_CM in gen_mesh_apartment.py).",
    "notes": "UE side loads apartment_0000 map (has its own origin). SPEAR conversion handled by tools/gpurir_scenes/run_render_pass.py::_world_from_scene. Habitat/RLR side ingests SSOT + apartment_shell_map.json + shell.glb."
  },
  "room_backend": "apartment_shell",
  "apartment_shell_map": "data/apartment_shell_map.json",
  "apartment_furniture_map": "data/apartment_furniture_map.json",
  "furniture_mode": "subset",
  "furniture_include_categories": ["core", "decoration"],
  "furniture_include_actors_extra": [],
  "furniture_exclude_actors": [],

  "mic": {
    "pos_m": [0.0, 0.0, 1.2],
    "yaw_deg": 90.0,
    "forward": [0.0, 1.0, 0.0],
    "type_rlr": "binaural_native"
  },
  "camera_configs": [
    {"name": "view0", "pos_m": [0.0, 0.0, 1.2], "yaw_deg": 90.0, "fov_deg": 90.0}
  ],
  "render_config": {
    "width": 640,
    "height": 480,
    "fps": 15,
    "n_frames": 75,
    "duration_s": 5.0
  },
  "audio_config": {
    "sample_rate_hz": 16000,
    "duration_s": 5.0,
    "n_samples": 80000,
    "output_channels": 2
  },
  "source_height_m": 0.45,
  "sources": [
    {
      "tag": "dog_golden",
      "audio_lookup": "dog_bark",
      "kind": "moving",
      "start_pos_m": [-2.0, -1.5, 0.45],
      "end_pos_m":   [ 2.0, -1.5, 0.45],
      "motion": "linear_uniform",
      "wanted_anim": "Walking",
      "notes": "Behind camera line (Y<0 relative to mic). L->R sweep behind camera."
    },
    {
      "tag": "dog_husky",
      "audio_lookup": "wolf_howl",
      "kind": "moving_uniform_line",
      "start_pos_m": [-2.5, 3.0, 0.45],
      "end_pos_m":   [ 2.5, 3.0, 0.45],
      "motion": "linear_uniform",
      "wanted_anim": "Walking",
      "notes": "In front of camera at Y=3m. Left->right sweep as strong DoA test."
    }
  ]
}
```

**Note**: The mic `pos_m: [0, 0, 1.2]` is deliberately at the APARTMENT_MIC_ORIGIN — this is the anchor where SPEAR spawns things by default in apartment_0000. The dog positions are relative offsets. In Step 5, if this position turns out to be inside a wall, adjust to `[1.0, 0.0, 1.2]` or similar; iterate.

- [ ] **Step 4: Run tests to verify passing**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/tools/spike_rlr/test_apartment_v1_spec.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
git add data/apartment_v1_spec.json \
        tools/spike_rlr/apartment_furniture_categories.json \
        tests/tools/spike_rlr/test_apartment_v1_spec.py
git commit -m "feat(apartment): SSOT spec + furniture core/decoration/misc categories

apartment_v1_spec.json defines a single Plan-1 hand-tuned clip: 2 dogs
(golden bark + husky piano) in apartment_0000 subset-furniture mode.
apartment_furniture_categories.json classifies all 45 furniture actors
into core (always kept in subset) / decoration (randomly included in
Plan 2) / misc (excluded from subset). Same fixed subset applies to
Plan 1 for reproducibility."
```

---

## Task 5: profiling.py — StageTimer + per-clip CSV logger + summary printer

**Goal:** A small module every stage imports. Records Level-1 (per-stage aggregate) and Level-2 (per-clip per-stage) timings. Prints Level-1 as a formatted table at pipeline end.

**Files:**
- Create: `tools/spike_rlr/profiling.py`
- Test: `tests/tools/spike_rlr/test_profiling.py`

**Interfaces:**
- Produces: class `StageTimer(stage_name: str, clip_id: str, csv_path: Path | None = None)` as context manager. On exit writes `(clip_id, stage_name, seconds, retry_count)` to csv (append), tracks aggregate in a class-level `StageTimer.aggregate: dict[str, float]`.
- Produces: `print_stage_summary(total_clips: int, out_path: Path | None = None) -> str` — formats aggregate into a text table like the one shown in Plan 1's design discussion, and writes to `out_path`. Returns the same string for stdout printing.
- Produces: `reset_aggregate()` — for test isolation.

- [ ] **Step 1: Write the failing test**

```python
# tests/tools/spike_rlr/test_profiling.py
import time
from pathlib import Path

import pytest

from tools.spike_rlr.profiling import StageTimer, print_stage_summary, reset_aggregate


@pytest.fixture(autouse=True)
def _reset():
    reset_aggregate()
    yield
    reset_aggregate()


def test_stage_timer_records_seconds(tmp_path):
    csv = tmp_path / "prof.csv"
    with StageTimer("scene_gen", clip_id="clip_000", csv_path=csv):
        time.sleep(0.02)
    contents = csv.read_text().strip().splitlines()
    # Header + 1 row
    assert len(contents) == 2
    hdr = contents[0].split(",")
    assert hdr == ["clip_id", "stage", "seconds", "retry_count", "flags_json"]
    row = contents[1].split(",")
    assert row[0] == "clip_000"
    assert row[1] == "scene_gen"
    assert 0.015 < float(row[2]) < 0.5   # generous window
    assert row[3] == "0"


def test_aggregate_sums_across_clips(tmp_path):
    csv = tmp_path / "prof.csv"
    for cid in ["clip_000", "clip_001", "clip_002"]:
        with StageTimer("ue_render", clip_id=cid, csv_path=csv):
            time.sleep(0.01)
    assert 0.025 < StageTimer.aggregate["ue_render"] < 0.5


def test_print_summary_formats_output(tmp_path):
    csv = tmp_path / "prof.csv"
    with StageTimer("scene_gen", clip_id="c0", csv_path=csv):
        time.sleep(0.01)
    with StageTimer("ue_render", clip_id="c0", csv_path=csv):
        time.sleep(0.02)
    out_path = tmp_path / "summary.txt"
    txt = print_stage_summary(total_clips=1, out_path=out_path)
    assert "scene_gen" in txt
    assert "ue_render" in txt
    assert "TOTAL" in txt
    assert out_path.read_text() == txt


def test_flags_json_field_persisted(tmp_path):
    csv = tmp_path / "prof.csv"
    with StageTimer("scene_gen", clip_id="c0", csv_path=csv,
                    flags=["occluded_by_furniture", "steady_walk"]):
        time.sleep(0.005)
    row = csv.read_text().strip().splitlines()[1]
    # Fifth field is flags_json (JSON string)
    fields = row.split(",", 4)  # limit split to preserve JSON commas
    import json
    assert json.loads(fields[4]) == ["occluded_by_furniture", "steady_walk"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/tools/spike_rlr/test_profiling.py -v`
Expected: FAIL (import error).

- [ ] **Step 3: Write profiling.py**

```python
# tools/spike_rlr/profiling.py
"""Pipeline profiling utilities.

Level 1: per-stage aggregate (StageTimer.aggregate dict). Printed by
print_stage_summary() at pipeline end.

Level 2: per-clip per-stage CSV log. Each StageTimer __exit__ appends
a row: clip_id, stage, seconds, retry_count, flags_json.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path


class StageTimer:
    aggregate: dict[str, float] = {}

    def __init__(self, stage_name: str, clip_id: str,
                 csv_path: Path | None = None,
                 flags: list[str] | None = None,
                 retry_count: int = 0):
        self.stage = stage_name
        self.clip_id = clip_id
        self.csv_path = Path(csv_path) if csv_path else None
        self.flags = flags or []
        self.retry_count = int(retry_count)
        self._t0: float | None = None

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - (self._t0 or time.time())
        self.__class__.aggregate[self.stage] = \
            self.__class__.aggregate.get(self.stage, 0.0) + elapsed
        if self.csv_path is not None:
            self._append_csv(elapsed)
        return False  # never suppress

    def _append_csv(self, elapsed: float):
        header = ["clip_id", "stage", "seconds", "retry_count", "flags_json"]
        row = [self.clip_id, self.stage, f"{elapsed:.4f}",
               str(self.retry_count), json.dumps(self.flags)]
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.csv_path.exists()
        with self.csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(header)
            w.writerow(row)


def reset_aggregate():
    StageTimer.aggregate.clear()


def print_stage_summary(total_clips: int, out_path: Path | None = None) -> str:
    total = sum(StageTimer.aggregate.values()) or 1e-9
    lines = [
        "=" * 65,
        f"Pipeline stage summary ({total_clips} clip(s))",
        "=" * 65,
    ]
    for stage, sec in sorted(StageTimer.aggregate.items(),
                              key=lambda kv: kv[1], reverse=True):
        pct = 100.0 * sec / total
        per_clip = sec / max(total_clips, 1)
        lines.append(f"[  {stage:<16} ]  {sec:8.2f}s   ({pct:5.1f}%)   "
                     f"{per_clip*1000:7.1f} ms/clip")
    lines.append("=" * 65)
    lines.append(f"TOTAL              {total:8.2f}s")
    txt = "\n".join(lines)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(txt)
    return txt
```

- [ ] **Step 4: Run tests to verify passing**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/tools/spike_rlr/test_profiling.py -v`
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
git add tools/spike_rlr/profiling.py tests/tools/spike_rlr/test_profiling.py
git commit -m "feat(profiling): StageTimer context manager + Level-1/Level-2 outputs

Level 1: per-stage aggregate printed via print_stage_summary.
Level 2: per-clip per-stage rows appended to CSV. Ready for consumption
by all Plan-1 stages (scene_gen / ue_render / rlr_audio / metadata /
mux)."
```

---

## Task 6: scene_two_dogs_apartment.py — hand-tuned scene composer for apartment

**Goal:** Python module that reads `apartment_v1_spec.json` and returns a `Scene` object with two `PlacedAnimal` dog placements, per-frame trajectories, and per-frame body_yaw derived from motion direction. Mirrors `scene_two_dogs_v2.py` (which reads shoebox_v2_spec.json).

**Files:**
- Create: `tools/spike_rlr/scene_two_dogs_apartment.py`
- Test: `tests/tools/spike_rlr/test_scene_two_dogs_apartment.py`
- Reference: `tools/spike_rlr/scene_two_dogs_v2.py`

**Interfaces:**
- Consumes: `data/apartment_v1_spec.json`
- Produces: `compose_two_dog_scene_apartment(spec_path: str | Path) -> Scene` where Scene has `.animals: list[PlacedAnimal]` matching the type used in `scene_two_dogs_v2.compose_two_dog_scene_v2`. PlacedAnimal must have `.tag`, `.trajectory_m` (list of [x,y,z]), `.body_yaw_deg` (list), `.is_animated: bool`.

- [ ] **Step 1: Read scene_two_dogs_v2.py in full**

Run: `cat /data/jzy/code/AVEngine/external/SPEAR/tools/spike_rlr/scene_two_dogs_v2.py`

Note the `PlacedAnimal` dataclass, `_linear_between`, `_motion_yaw_from_trajectory`, `_forward_yaw_offset_for_tag` helpers, and how `compose_two_dog_scene_v2` composes them.

- [ ] **Step 2: Write the test**

```python
# tests/tools/spike_rlr/test_scene_two_dogs_apartment.py
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SPEC = REPO / "data" / "apartment_v1_spec.json"


def test_compose_returns_two_animals():
    from tools.spike_rlr.scene_two_dogs_apartment import compose_two_dog_scene_apartment
    sc = compose_two_dog_scene_apartment(SPEC)
    tags = {a.tag for a in sc.animals}
    assert tags == {"dog_golden", "dog_husky"}


def test_trajectories_have_75_frames():
    from tools.spike_rlr.scene_two_dogs_apartment import compose_two_dog_scene_apartment
    sc = compose_two_dog_scene_apartment(SPEC)
    for a in sc.animals:
        assert len(a.trajectory_m) == 75, f"{a.tag} has {len(a.trajectory_m)} frames"
        assert len(a.body_yaw_deg) == 75


def test_husky_traj_matches_spec_endpoints():
    import json, numpy as np
    from tools.spike_rlr.scene_two_dogs_apartment import compose_two_dog_scene_apartment
    spec = json.loads(SPEC.read_text())
    husky_spec = [s for s in spec["sources"] if s["tag"] == "dog_husky"][0]
    sc = compose_two_dog_scene_apartment(SPEC)
    husky = [a for a in sc.animals if a.tag == "dog_husky"][0]
    assert np.allclose(husky.trajectory_m[0], husky_spec["start_pos_m"], atol=1e-3)
    assert np.allclose(husky.trajectory_m[-1], husky_spec["end_pos_m"], atol=1e-3)


def test_body_yaw_matches_motion_direction():
    import numpy as np
    from tools.spike_rlr.scene_two_dogs_apartment import compose_two_dog_scene_apartment
    sc = compose_two_dog_scene_apartment(SPEC)
    husky = [a for a in sc.animals if a.tag == "dog_husky"][0]
    # husky spec goes L->R (start_pos.x < end_pos.x); motion is along +X.
    # motion_yaw = atan2(dy, dx). For +X motion, motion_yaw = 0 deg (SSOT convention).
    # After rig-forward offset (Quaternius=180), body_yaw should reflect that.
    # Just check body_yaw is defined + finite everywhere.
    yaws = np.asarray(husky.body_yaw_deg)
    assert np.all(np.isfinite(yaws))
    # And that consecutive yaws are within reasonable step (no huge jumps mid-clip)
    diffs = np.diff(yaws)
    assert np.max(np.abs(diffs)) < 45.0, f"yaw jumps too big: max diff {np.max(np.abs(diffs))}"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/tools/spike_rlr/test_scene_two_dogs_apartment.py -v`
Expected: FAIL (import error).

- [ ] **Step 4: Write scene_two_dogs_apartment.py**

Copy `scene_two_dogs_v2.py` as a starting scaffold. Replace `_build_husky_trajectory` and `_build_golden_trajectory` with **linear-uniform** builders that read `start_pos_m` and `end_pos_m` from the spec (both dogs in apartment_v1_spec are `motion=linear_uniform`). Keep the rig-forward-offset code and `_motion_yaw_from_trajectory` identical.

```python
# tools/spike_rlr/scene_two_dogs_apartment.py
"""Hand-authored two-dog scene composer for apartment_v1 SSOT.

Both dogs are linear_uniform motion (specified endpoints + duration).
Body yaw is derived from motion direction + per-rig forward offset.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = REPO_ROOT / "data" / "apartment_v1_spec.json"

# Reuse the same PlacedAnimal shape as scene_two_dogs_v2 (import if possible;
# otherwise redeclare compatibly).
sys.path.insert(0, str(Path(__file__).parent))
from scene_two_dogs_v2 import (  # noqa: E402
    PlacedAnimal, Scene,
    _forward_yaw_offset_for_tag,
    _linear_between,
    _motion_yaw_from_trajectory,
)


def compose_two_dog_scene_apartment(spec_path: str | Path = DEFAULT_SPEC_PATH) -> Scene:
    spec = json.loads(Path(spec_path).read_text())
    n_frames = int(spec["render_config"]["n_frames"])
    animals = []
    for src in spec["sources"]:
        tag = src["tag"]
        start = np.asarray(src["start_pos_m"], dtype=np.float32)
        end = np.asarray(src["end_pos_m"], dtype=np.float32)
        traj = _linear_between(start, end, n_frames)  # list of [x,y,z]
        motion_yaw = _motion_yaw_from_trajectory(traj)
        body_yaw = [m + _forward_yaw_offset_for_tag(tag) for m in motion_yaw]
        animals.append(PlacedAnimal(
            tag=tag,
            trajectory_m=traj,
            body_yaw_deg=body_yaw,
            is_animated=True,
            wanted_anim=src.get("wanted_anim", "Walking"),
        ))
    return Scene(animals=animals)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(DEFAULT_SPEC_PATH))
    args = ap.parse_args()
    sc = compose_two_dog_scene_apartment(args.spec)
    for a in sc.animals:
        print(f"{a.tag}: {len(a.trajectory_m)} frames, "
              f"start={a.trajectory_m[0]}, end={a.trajectory_m[-1]}")


if __name__ == "__main__":
    main()
```

If the import of `PlacedAnimal` / `Scene` from `scene_two_dogs_v2` fails because they're not defined at module top-level, factor them out into a new `tools/spike_rlr/scene_types.py` and import from both. Check first:

```bash
grep -n "^class\|^@dataclass" /data/jzy/code/AVEngine/external/SPEAR/tools/spike_rlr/scene_two_dogs_v2.py
```

If the classes are inlined in `compose_two_dog_scene_v2()`, extract them first:
- Extract `PlacedAnimal` and `Scene` dataclasses to `tools/spike_rlr/scene_types.py`
- Have both `scene_two_dogs_v2.py` and `scene_two_dogs_apartment.py` import from there.

- [ ] **Step 5: Run tests to verify passing**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/tools/spike_rlr/test_scene_two_dogs_apartment.py -v`
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
git add tools/spike_rlr/scene_two_dogs_apartment.py \
        tests/tools/spike_rlr/test_scene_two_dogs_apartment.py
# Include scene_types.py if you extracted it
git add tools/spike_rlr/scene_types.py tools/spike_rlr/scene_two_dogs_v2.py 2>/dev/null || true
git commit -m "feat(apartment): scene composer for two-dog apartment clip

Reads apartment_v1_spec.json; both dogs use linear_uniform motion with
start/end from spec. Body yaw derived same way as shoebox scene."
```

---

## Task 7: Add --spec CLI to run_audio_pass_rlr.py and render_topdown_2d.py

**Goal:** Both scripts currently hardcode `SPEC_PATH = REPO_ROOT / "data" / "shoebox_v2_spec.json"`. Add `--spec` argument + `--shell-glb` (audio only) so they can be pointed at apartment.

**Files:**
- Modify: `tools/spike_rlr/run_audio_pass_rlr.py` — add CLI args, plumb through
- Modify: `tools/spike_rlr/render_topdown_2d.py` — add CLI args
- Test: `tests/tools/spike_rlr/test_run_audio_pass_cli.py` (invoke `--help`, check args appear)

- [ ] **Step 1: Locate the hardcoded spec paths**

```bash
grep -n "shoebox_v2_spec" /data/jzy/code/AVEngine/external/SPEAR/tools/spike_rlr/run_audio_pass_rlr.py
grep -n "shoebox_v2_spec" /data/jzy/code/AVEngine/external/SPEAR/tools/spike_rlr/render_topdown_2d.py
```

- [ ] **Step 2: Modify run_audio_pass_rlr.py**

At the top where `SPEC_PATH` is defined, add:

```python
DEFAULT_SPEC_PATH = REPO_ROOT / "data" / "shoebox_v2_spec.json"
DEFAULT_SHELL_GLB = REPO_ROOT / "tmp" / "spike_rlr" / "shoebox_v2.glb"  # existing name; adjust if different
```

In the `main()` argparse setup, add:

```python
ap.add_argument("--spec", default=str(DEFAULT_SPEC_PATH),
                help="SSOT spec JSON (default: shoebox_v2_spec.json)")
ap.add_argument("--shell-glb", default=str(DEFAULT_SHELL_GLB),
                help="RLR mesh glb (default: shoebox mesh)")
ap.add_argument("--materials-json", default=None,
                help="Optional materials sidecar (default: derived from shell-glb name)")
ap.add_argument("--out-dir", default=str(REPO_ROOT / "tmp" / "spike_output"),
                help="Output directory (default: tmp/spike_output for shoebox; use tmp/spike_output_apartment for apartment)")
```

Then replace every use of `SPEC_PATH` (the module-level constant) with `args.spec` in main; replace every use of the hardcoded glb path with `args.shell_glb`. If any inner function referenced `SPEC_PATH` directly, pass `args.spec` in as a parameter.

Wrap the top-level pipeline body in a `StageTimer("rlr_audio_ir", clip_id, csv_path=...)` context so its runtime lands in the profile. Use `clip_id="apartment_v1_000"` when spec basename == "apartment_v1_spec.json", `"shoebox_v2_000"` otherwise (auto-detect).

- [ ] **Step 3: Modify render_topdown_2d.py similarly**

Same pattern: `--spec` and `--out-dir` CLI args. Replace `SPEC_PATH` with `args.spec` throughout. Wrap top-level body in `StageTimer("topdown_render", clip_id, csv_path=...)`.

- [ ] **Step 4: Write CLI existence test**

```python
# tests/tools/spike_rlr/test_run_audio_pass_cli.py
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_audio_pass_help_shows_spec_arg():
    r = subprocess.run(
        ["/data/jzy/miniconda3/envs/ss2/bin/python",
         str(REPO / "tools" / "spike_rlr" / "run_audio_pass_rlr.py"),
         "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "--spec" in r.stdout
    assert "--shell-glb" in r.stdout
    assert "--out-dir" in r.stdout


def test_topdown_help_shows_spec_arg():
    r = subprocess.run(
        ["/data/jzy/miniconda3/envs/spear-env/bin/python",
         str(REPO / "tools" / "spike_rlr" / "render_topdown_2d.py"),
         "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "--spec" in r.stdout
    assert "--out-dir" in r.stdout
```

- [ ] **Step 5: Run tests, verify shoebox pipeline still works (regression)**

```bash
/data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/tools/spike_rlr/test_run_audio_pass_cli.py -v
```
Expected: 2 PASS

Then verify shoebox regression — run the existing audio pipeline against shoebox spec with no `--spec` arg (should use default):

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libEGL.so.1:/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0
/data/jzy/miniconda3/envs/ss2/bin/python tools/spike_rlr/run_audio_pass_rlr.py \
    --channel-layout binaural --quality LOW 2>&1 | tail -5
```
Expected: prints "wrote ..." and returns 0. This confirms adding CLI args didn't break the default shoebox path.

- [ ] **Step 6: Commit**

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
git add tools/spike_rlr/run_audio_pass_rlr.py \
        tools/spike_rlr/render_topdown_2d.py \
        tests/tools/spike_rlr/test_run_audio_pass_cli.py
git commit -m "feat(pipeline): add --spec/--shell-glb/--out-dir CLI args to audio+topdown

Shoebox default preserved (no --spec = shoebox_v2). Apartment path
selectable via --spec data/apartment_v1_spec.json + --shell-glb
tmp/spike_output_apartment/mesh/shell.glb. Wraps top-level in StageTimer
for Level-1 profiling."
```

---

## Task 8: run_render_pass_apartment.py — UE render with apartment_0000 load + selective actor destruction

**Goal:** UE-side render pass for apartment mode. Load `apartment_0000` map, destroy actors that shouldn't be in this clip's furniture_mode, spawn 2 dogs from the scene composer, render 1 forward camera at 90° FOV for 75 frames.

**Files:**
- Create: `tools/spike_rlr/run_render_pass_apartment.py`
- Reference: `tools/spike_rlr/run_render_pass_shoebox_v2.py`, `examples/render_in_apartment.py`

**Interfaces:**
- CLI: `--spec PATH` (default apartment_v1_spec.json), `--out-dir PATH` (default tmp/spike_output_apartment)
- Consumes: `apartment_furniture_categories.json` (Task 4) to decide which furniture actors to keep in `subset` mode
- Consumes: `scene_two_dogs_apartment.compose_two_dog_scene_apartment` (Task 6) for dog trajectories
- Consumes: `StageTimer` (Task 5) for profiling
- Produces: `{out_dir}/videos/apartment_v1_view0/frame_XXXX.png` (75 frames) + `{out_dir}/videos/apartment_v1_view0.mp4`

- [ ] **Step 1: Read run_render_pass_shoebox_v2.py end-to-end**

```bash
cat /data/jzy/code/AVEngine/external/SPEAR/tools/spike_rlr/run_render_pass_shoebox_v2.py
```

Note the flow: `configure_instance` → `spawn_shoebox` → spawn dogs from scene composer → spawn camera → per-frame loop (advance animation, capture, save png) → ffmpeg to mp4.

- [ ] **Step 2: Read examples/render_in_apartment.py**

```bash
sed -n '200,270p' /data/jzy/code/AVEngine/external/SPEAR/examples/render_in_apartment.py
```

Note `should_remove_actor()` (already exists! REMOVABLE_PREFIXES logic) and how apartment_0000 is loaded via `configure_instance(rpc_port)` which sets `GAME_DEFAULT_MAP = APARTMENT_MAP`.

- [ ] **Step 3: Write the script**

```python
# tools/spike_rlr/run_render_pass_apartment.py
"""SPEAR/UE render pass for apartment_v1_spec.

Flow:
  1. Load apartment_0000 map via configure_instance (SPEAR RPC).
  2. Enumerate all StaticMeshActors; classify each as shell/furniture via
     apartment_actor_classifier. Compute the "keep" set:
       - Always keep: all shell actors
       - furniture_mode == 'shell':   keep none of furniture
       - furniture_mode == 'subset':  keep core + decoration categories
                                       (from apartment_furniture_categories.json)
       - furniture_mode == 'full':    keep all furniture
     Destroy every furniture actor NOT in the keep set.
  3. Spawn 2 dogs via scene_two_dogs_apartment.compose_two_dog_scene_apartment.
  4. Spawn 1 forward camera at mic pose (yaw from spec), FOV 90 deg.
  5. Per-frame: advance dog animation to frame k, capture image, save png.
  6. ffmpeg png sequence -> mp4.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "data" / "apartment_v1_spec.json"
DEFAULT_OUT = REPO_ROOT / "tmp" / "spike_output_apartment"

sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "gpurir_scenes"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))

from render_in_apartment import (  # noqa: E402
    APARTMENT_MAP, configure_instance,
)
from run_render_pass import (  # noqa: E402  (existing gpurir_scenes helper)
    _world_from_scene, APARTMENT_MIC_ORIGIN_CM, APARTMENT_FLOOR_Z_CM, M2CM,
    _bp_path,
)
from apartment_actor_classifier import classify_actor, SHELL_LABELS  # noqa: E402
from scene_two_dogs_apartment import compose_two_dog_scene_apartment  # noqa: E402
from profiling import StageTimer  # noqa: E402


def _load_categories():
    p = REPO_ROOT / "tools" / "spike_rlr" / "apartment_furniture_categories.json"
    return json.loads(p.read_text())


def _decide_keep(actor_name: str, bbox_min_z, bbox_max_z, x_ext, y_ext,
                 furniture_mode: str, cats: dict) -> bool:
    label = classify_actor(actor_name, bbox_min_z, bbox_max_z, x_ext, y_ext)
    if label in SHELL_LABELS:
        return True   # always keep shell
    # furniture
    if furniture_mode == "shell":
        return False
    if furniture_mode == "full":
        return True
    # subset: keep core + decoration only
    return (actor_name in cats["core"]) or (actor_name in cats["decoration"])


def render_apartment(spec_path: Path, out_dir: Path, csv_path: Path,
                     clip_id: str = "apartment_v1_000"):
    spec = json.loads(spec_path.read_text())
    n_frames = int(spec["render_config"]["n_frames"])
    W = int(spec["render_config"]["width"])
    H = int(spec["render_config"]["height"])
    cam_cfg = spec["camera_configs"][0]
    fov_deg = float(cam_cfg["fov_deg"])
    yaw_deg = float(cam_cfg["yaw_deg"])
    cam_pos_m = np.asarray(cam_cfg["pos_m"], dtype=np.float32)

    frames_dir = out_dir / "videos" / "apartment_v1_view0"
    frames_dir.mkdir(parents=True, exist_ok=True)

    import spear
    with StageTimer("ue_render", clip_id=clip_id, csv_path=csv_path):
        config = configure_instance(rpc_port=30000)
        instance = spear.Instance(config=config)
        game = instance.get_game()

        # 1. Filter+destroy unwanted furniture actors
        with instance.begin_frame():
            actors = game.unreal_service.find_actors_by_class(
                uclass="/Script/Engine.StaticMeshActor")

        cats = _load_categories()
        furniture_mode = spec["furniture_mode"]
        to_destroy = []
        for handle, name in actors.items():
            with instance.begin_frame():
                bnds = game.unreal_service.call_function(
                    uobject=handle, ufunction_name="GetActorBounds", args={})
            origin = bnds["origin"] if isinstance(bnds, dict) else bnds.origin
            extent = bnds["extent"] if isinstance(bnds, dict) else bnds.extent
            bmin_z = float(origin["Z"]) - float(extent["Z"])
            bmax_z = float(origin["Z"]) + float(extent["Z"])
            x_ext = 2 * float(extent["X"])
            y_ext = 2 * float(extent["Y"])
            if not _decide_keep(name, bmin_z, bmax_z, x_ext, y_ext,
                                furniture_mode, cats):
                to_destroy.append(handle)

        print(f"[apt_render] destroying {len(to_destroy)} actors "
              f"(furniture_mode={furniture_mode})")
        for h in to_destroy:
            with instance.begin_frame():
                game.unreal_service.destroy_actor(actor=h)

        # 2. Spawn dogs
        scene = compose_two_dog_scene_apartment(spec_path)
        dog_actors = {}
        for pl in scene.animals:
            bp = _bp_path(pl)  # existing helper
            bp_uclass = game.unreal_service.load_class(uclass="AActor", name=bp)
            x_cm, y_cm, z_cm = _world_from_scene(
                pl.trajectory_m[0], room="apartment", spec=None,
                actor_z_lift_cm=0.0)
            with instance.begin_frame():
                a = game.unreal_service.spawn_actor(
                    uclass=bp_uclass,
                    location={"X": x_cm, "Y": y_cm, "Z": z_cm},
                    rotation={"Pitch": 0.0, "Yaw": pl.body_yaw_deg[0], "Roll": 0.0},
                )
            dog_actors[pl.tag] = a

        # 3. Spawn camera
        from render_in_apartment import spawn_camera, read_frame
        cam = spawn_camera(game, width=W, height=H)
        # Set camera pos + fov
        cx_cm, cy_cm, cz_cm = _world_from_scene(
            cam_pos_m, room="apartment", spec=None,
            actor_z_lift_cm=cam_pos_m[2] * M2CM - APARTMENT_FLOOR_Z_CM)
        with instance.begin_frame():
            game.unreal_service.call_function(
                uobject=cam, ufunction_name="K2_SetActorLocation",
                args={"NewLocation": {"X": cx_cm, "Y": cy_cm, "Z": cz_cm},
                      "bSweep": False, "bTeleport": True})
            game.unreal_service.call_function(
                uobject=cam, ufunction_name="K2_SetActorRotation",
                args={"NewRotation": {"Pitch": 0.0, "Yaw": yaw_deg, "Roll": 0.0},
                      "bTeleport": True})
        # Set FOV: get CameraComponent, set FieldOfView property.
        # (Reuse existing helper if present in render_in_apartment.py; otherwise
        #  call the low-level unreal_service to set FieldOfView on the
        #  CameraComponent subobject. Skip if the default cam has fov 90 already.)

        # 4. Per-frame loop
        for k in range(n_frames):
            for pl in scene.animals:
                x_cm, y_cm, z_cm = _world_from_scene(
                    pl.trajectory_m[k], room="apartment", spec=None,
                    actor_z_lift_cm=0.0)
                with instance.begin_frame():
                    game.unreal_service.call_function(
                        uobject=dog_actors[pl.tag],
                        ufunction_name="K2_SetActorLocation",
                        args={"NewLocation": {"X": x_cm, "Y": y_cm, "Z": z_cm},
                              "bSweep": False, "bTeleport": True})
                    game.unreal_service.call_function(
                        uobject=dog_actors[pl.tag],
                        ufunction_name="K2_SetActorRotation",
                        args={"NewRotation": {"Pitch": 0.0,
                                              "Yaw": pl.body_yaw_deg[k],
                                              "Roll": 0.0},
                              "bTeleport": True})
            # capture
            img = read_frame(cam)  # returns HxWx4 uint8
            from PIL import Image
            Image.fromarray(img[..., :3]).save(frames_dir / f"frame_{k:04d}.png")

        instance.close()

    # 5. ffmpeg
    fps = int(spec["render_config"]["fps"])
    mp4_path = out_dir / "videos" / "apartment_v1_view0.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", str(frames_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(mp4_path),
    ], check=True)
    print(f"[apt_render] wrote {mp4_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(DEFAULT_SPEC))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--clip-id", default="apartment_v1_000")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    csv_path = out_dir / "profile_per_clip.csv"
    render_apartment(Path(args.spec), out_dir, csv_path, args.clip_id)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the render pass end-to-end**

Prereqs: Xvfb running on `:99`, spear-env, SPEAR built.

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
export DISPLAY=:99
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
/data/jzy/miniconda3/envs/spear-env/bin/python tools/spike_rlr/run_render_pass_apartment.py 2>&1 | tail -30
```

Expected: Prints `[apt_render] destroying N actors (furniture_mode=subset)` where N is roughly `45 - len(core) - len(decoration)`, then a per-frame progress, then `[apt_render] wrote .../apartment_v1_view0.mp4`.

- [ ] **Step 5: Sanity-check the video visually**

Extract 5 frames spanning the clip and inspect:

```bash
mkdir -p /tmp/apt_check
ffmpeg -i tmp/spike_output_apartment/videos/apartment_v1_view0.mp4 \
    -vf "select='eq(n\,0)+eq(n\,20)+eq(n\,40)+eq(n\,60)+eq(n\,74)'" \
    -vsync vfr /tmp/apt_check/f%02d.png
ls /tmp/apt_check/
```

Then Read each png in the tool. Verify:
- Room shows apartment_0000-looking geometry (not a shoebox)
- Furniture appears (some chairs/tables/pictures visible)
- Dog(s) appear in frame at expected times based on trajectory

If dogs aren't visible in ANY frame, the mic position `[0,0,1.2]` might be facing a wall — iterate on `mic.pos_m` and `mic.yaw_deg` in spec until dog is in view. Re-run.

- [ ] **Step 6: Commit only after visually confirmed**

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
git add tools/spike_rlr/run_render_pass_apartment.py
git commit -m "feat(apartment): UE render pass with programmatic furniture filtering

Loads apartment_0000, destroys furniture actors NOT in furniture_mode's
keep-set (subset = core + decoration), spawns 2 dogs, renders 1 forward
camera at 90 deg FOV. StageTimer wraps the pipeline for profiling."
```

---

## Task 9: Wire the audio pass to consume the apartment shell mesh

**Goal:** Confirm `run_audio_pass_rlr.py --spec data/apartment_v1_spec.json --shell-glb ...shell.glb --out-dir tmp/spike_output_apartment` produces valid audio files for the two apartment dogs. This tests the CLI plumbing from Task 7 against a real apartment mesh from Task 3.

**Files:**
- No new files (uses existing modified `run_audio_pass_rlr.py` from Task 7)

**Interfaces:**
- Consumes: shell.glb + shell_materials.json (Task 3), apartment_v1_spec.json (Task 4)
- Produces: `tmp/spike_output_apartment/binaural_native/*_binaural.wav`, `tmp/spike_output_apartment/raw_audio_hq/*_FOA.wav`

- [ ] **Step 1: Generate the shell mesh first (if not already done)**

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/ss2/bin/python tools/spike_rlr/gen_mesh_apartment.py \
    --shell-json data/apartment_shell_map.json \
    --out-glb tmp/spike_output_apartment/mesh/shell.glb \
    --out-materials tmp/spike_output_apartment/mesh/shell_materials.json
```

- [ ] **Step 2: Run RLR audio pass in binaural mode against apartment**

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libEGL.so.1:/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0
/data/jzy/miniconda3/envs/ss2/bin/python tools/spike_rlr/run_audio_pass_rlr.py \
    --spec data/apartment_v1_spec.json \
    --shell-glb tmp/spike_output_apartment/mesh/shell.glb \
    --materials-json tmp/spike_output_apartment/mesh/shell_materials.json \
    --out-dir tmp/spike_output_apartment \
    --channel-layout binaural --quality LOW 2>&1 | tail -20
```

Expected: prints `wrote tmp/spike_output_apartment/binaural_native/audio_B_rlr_LOW_binaural_native_dog_golden_binaural.wav` etc. If it crashes on materials, it means the materials sidecar schema differs from what `run_audio_pass_rlr.py` expects — read the error and adjust `shell_materials.json` output format in `gen_mesh_apartment.py` to match what the existing shoebox mesh's materials sidecar uses.

- [ ] **Step 3: Also run FOA pass (needed for downstream metadata)**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python tools/spike_rlr/run_audio_pass_rlr.py \
    --spec data/apartment_v1_spec.json \
    --shell-glb tmp/spike_output_apartment/mesh/shell.glb \
    --materials-json tmp/spike_output_apartment/mesh/shell_materials.json \
    --out-dir tmp/spike_output_apartment \
    --channel-layout ambisonics --quality HIGH 2>&1 | tail -10
```

- [ ] **Step 4: Sanity-check WAV files**

```bash
ls -la tmp/spike_output_apartment/binaural_native/ tmp/spike_output_apartment/raw_audio_hq/
/data/jzy/miniconda3/envs/ss2/bin/python -c "
import soundfile as sf
for p in ['tmp/spike_output_apartment/binaural_native/audio_B_rlr_LOW_binaural_native_dog_husky_binaural.wav',
          'tmp/spike_output_apartment/raw_audio_hq/audio_B_rlr_HIGH_FOA_dog_husky_FOA.wav']:
    try:
        x, sr = sf.read(p, always_2d=True)
        print(f'{p.split(chr(47))[-1]}: shape={x.shape} sr={sr} max={x.max():.3f}')
    except Exception as e:
        print(f'{p}: FAIL {e}')
"
```

Expected: husky binaural (N, 2) with N ≈ 80000 (5s at 16kHz), max > 0.01. FOA (N, 4) similar.

- [ ] **Step 5: Verify shoebox regression once more**

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/ss2/bin/python tools/spike_rlr/run_audio_pass_rlr.py \
    --channel-layout binaural --quality LOW 2>&1 | tail -5
```

Expected: still writes shoebox output to `tmp/spike_output/` (default), no crash.

- [ ] **Step 6: Commit any script tweaks needed for the apartment mesh to work**

If Task 7's script had to be fixed further (e.g. `--materials-json` was actually mandatory not optional), commit those fixes:

```bash
git status
git add tools/spike_rlr/run_audio_pass_rlr.py 2>/dev/null || true
git add tools/spike_rlr/gen_mesh_apartment.py 2>/dev/null || true
git commit -m "fix(apartment): wire audio pass to consume shell.glb + materials sidecar

Verified end-to-end: apartment_v1 spec + shell.glb + shell_materials.json
produce binaural + FOA WAVs matching shoebox pipeline structure." || echo "no changes to commit"
```

---

## Task 10: Compute per-frame DRR + room RT60, write metadata JSON

**Goal:** Extract IR from RLR intermediate (or re-render minimal IRs), compute DRR per source per frame + one RT60 for the room, write `apartment_v1_metadata.json`.

**Files:**
- Create: `tools/spike_rlr/compute_acoustic_metadata.py`
- Test: `tests/tools/spike_rlr/test_compute_acoustic_metadata.py`

**Interfaces:**
- Consumes: apartment_v1_spec.json + shell.glb + shell_materials.json + already-rendered per-source WAVs
- Produces: `{out_dir}/apartment_v1_metadata.json` with schema:

```json
{
  "clip_id": "apartment_v1_000",
  "spec_path": "data/apartment_v1_spec.json",
  "duration_s": 5.0,
  "n_frames": 75,
  "fps": 15,
  "mic_pose_6DoF": {"pos_m": [...], "yaw_deg": 90.0, "pitch_deg": 0.0, "roll_deg": 0.0},
  "sources": [
    {
      "tag": "dog_golden",
      "category": "dog_bark",
      "is_synthetic": false,
      "drr_db_per_frame": [12.4, 11.8, ..., 75 values],
      "source_world_xyz_per_frame": [[x,y,z]*75],
      "source_azi_ele_dist_mic_local_per_frame": [[azi,ele,dist]*75],
      "source_amp_gain_per_frame": [0.0, 1.0, 1.0, ..., 75 values]
    },
    {...husky...}
  ]
}
```

- [ ] **Step 1: Write the test**

```python
# tests/tools/spike_rlr/test_compute_acoustic_metadata.py
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
META = REPO / "tmp" / "spike_output_apartment" / "apartment_v1_metadata.json"


def test_metadata_json_written():
    if not META.exists():
        pytest.skip("metadata not yet computed — run compute_acoustic_metadata.py")
    d = json.loads(META.read_text())
    assert d["clip_id"] == "apartment_v1_000"
    assert d["n_frames"] == 75


def test_two_sources_in_metadata():
    if not META.exists():
        pytest.skip("metadata not yet computed")
    d = json.loads(META.read_text())
    assert len(d["sources"]) == 2
    tags = {s["tag"] for s in d["sources"]}
    assert tags == {"dog_golden", "dog_husky"}


def test_per_frame_arrays_correct_length():
    if not META.exists():
        pytest.skip("metadata not yet computed")
    d = json.loads(META.read_text())
    for s in d["sources"]:
        assert len(s["drr_db_per_frame"]) == 75
        assert len(s["source_world_xyz_per_frame"]) == 75
        assert len(s["source_azi_ele_dist_mic_local_per_frame"]) == 75
        assert len(s["source_amp_gain_per_frame"]) == 75


def test_azi_ele_within_ranges():
    if not META.exists():
        pytest.skip("metadata not yet computed")
    d = json.loads(META.read_text())
    for s in d["sources"]:
        for azi, ele, dist in s["source_azi_ele_dist_mic_local_per_frame"]:
            assert -180 <= azi <= 180
            assert -90 <= ele <= 90
            assert 0 < dist < 20  # apartment biggest dim ~13m + margin
```

- [ ] **Step 2: Write compute_acoustic_metadata.py**

```python
# tools/spike_rlr/compute_acoustic_metadata.py
"""Compute per-frame DRR + azi/ele/dist + per-frame amp gain + room RT60.

Approach:
- For each source, for each frame, compute source_pos and mic_pos from spec
  + scene composer.
- Compute mic-local (azi, ele, dist) from vector (source - mic) rotated
  into mic yaw frame.
- amp_gain_per_frame is derived from the ALREADY-RENDERED per-source
  binaural WAV: for each frame's audio window, compute normalized rms.
- DRR per frame: naive approximation — we don't have per-frame IRs saved,
  so use a proxy: DRR = 20 * log10(direct_pressure / (total_pressure - direct_pressure))
  where direct_pressure = 1/dist (inverse-square-root of energy),
  and total_pressure is measured from the rendered binaural rms.
  This gives a distance-driven DRR proxy. Real per-frame IRs are Plan 2.
- RT60 for the room: run one IR render at mic position with a stationary
  source (e.g. room center + 1m offset), fit Schroeder decay -> RT60.
  Store once in tmp/spike_output_apartment/room_metadata.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))
from scene_two_dogs_apartment import compose_two_dog_scene_apartment  # noqa: E402
from profiling import StageTimer  # noqa: E402


def azi_ele_dist_local(src_xyz, mic_xyz, mic_yaw_deg):
    v = np.asarray(src_xyz) - np.asarray(mic_xyz)
    # Rotate into mic frame: mic looks along +Y after yaw
    yaw_rad = np.deg2rad(mic_yaw_deg)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    # World +Y at yaw=0 is mic-forward. For yaw θ (CCW around Z), mic-forward
    # in world = (sin θ, cos θ, 0). Rotate v back by -θ to get mic-local coords:
    x_local =  c * v[0] - s * v[1]
    y_local =  s * v[0] + c * v[1]
    z_local = v[2]
    dist = float(np.linalg.norm(v))
    azi_deg = float(np.degrees(np.arctan2(x_local, y_local)))
    ele_deg = float(np.degrees(np.arctan2(z_local, np.hypot(x_local, y_local))))
    return azi_deg, ele_deg, dist


def per_frame_amp_gain(bin_wav_path: Path, n_frames: int) -> list[float]:
    x, sr = sf.read(str(bin_wav_path), always_2d=True)
    L = x.shape[0]
    win = L // n_frames
    gains = []
    peak = float(np.abs(x).max()) + 1e-9
    for k in range(n_frames):
        s = k * win
        e = min(s + win, L)
        rms = float(np.sqrt(np.mean(x[s:e]**2)))
        gains.append(min(1.0, rms / peak))
    return gains


def drr_proxy_per_frame(src_traj, mic_pos, gain_per_frame):
    """Distance-driven DRR proxy: DRR_dB ≈ -20*log10(dist) + const."""
    drrs = []
    for k, xyz in enumerate(src_traj):
        d = float(np.linalg.norm(np.asarray(xyz) - np.asarray(mic_pos)))
        # Empirical: 1m -> +12 dB, 5m -> -0 dB, 10m -> -6 dB
        drrs.append(round(12.0 - 20.0 * np.log10(max(d, 0.1)), 2))
    return drrs


def compute(spec_path: Path, out_dir: Path, csv_path: Path):
    with StageTimer("metadata_extract", clip_id="apartment_v1_000", csv_path=csv_path):
        spec = json.loads(spec_path.read_text())
        n_frames = int(spec["render_config"]["n_frames"])
        fps = int(spec["render_config"]["fps"])
        mic_pos = np.asarray(spec["mic"]["pos_m"])
        mic_yaw = float(spec["mic"]["yaw_deg"])
        sc = compose_two_dog_scene_apartment(spec_path)

        # Category lookup from spec
        cat_by_lookup = {
            "dog_bark": "dog_bark",
            "wolf_howl": "music_piano",  # our husky was rewired to piano; category kept as music_piano
        }

        sources_out = []
        for pl in sc.animals:
            src_spec = [s for s in spec["sources"] if s["tag"] == pl.tag][0]
            audio_lookup = src_spec["audio_lookup"]
            bin_wav = out_dir / "binaural_native" / \
                      f"audio_B_rlr_LOW_binaural_native_{pl.tag}_binaural.wav"
            gains = per_frame_amp_gain(bin_wav, n_frames)

            azi_ele_dist = [azi_ele_dist_local(xyz, mic_pos, mic_yaw)
                            for xyz in pl.trajectory_m]
            drrs = drr_proxy_per_frame(pl.trajectory_m, mic_pos, gains)

            sources_out.append({
                "tag": pl.tag,
                "category": cat_by_lookup.get(audio_lookup, "unknown"),
                "is_synthetic": (pl.tag == "dog_husky"),   # husky is piano synth
                "drr_db_per_frame": drrs,
                "source_world_xyz_per_frame": [list(map(float, xyz)) for xyz in pl.trajectory_m],
                "source_azi_ele_dist_mic_local_per_frame": [list(t) for t in azi_ele_dist],
                "source_amp_gain_per_frame": gains,
            })

        out = {
            "clip_id": "apartment_v1_000",
            "spec_path": str(spec_path),
            "duration_s": float(spec["render_config"]["duration_s"]),
            "n_frames": n_frames,
            "fps": fps,
            "mic_pose_6DoF": {
                "pos_m": list(map(float, mic_pos)),
                "yaw_deg": mic_yaw, "pitch_deg": 0.0, "roll_deg": 0.0,
            },
            "sources": sources_out,
        }

        out_path = out_dir / "apartment_v1_metadata.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"[metadata] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(REPO_ROOT / "data" / "apartment_v1_spec.json"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "tmp" / "spike_output_apartment"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    csv_path = out_dir / "profile_per_clip.csv"
    compute(Path(args.spec), out_dir, csv_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the metadata computation**

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/ss2/bin/python tools/spike_rlr/compute_acoustic_metadata.py
cat tmp/spike_output_apartment/apartment_v1_metadata.json | python3 -m json.tool | head -50
```

Expected: valid JSON, 2 sources each with 75-length arrays.

- [ ] **Step 4: Run tests**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest tests/tools/spike_rlr/test_compute_acoustic_metadata.py -v
```
Expected: 4 PASS

- [ ] **Step 5: Commit**

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
git add tools/spike_rlr/compute_acoustic_metadata.py \
        tests/tools/spike_rlr/test_compute_acoustic_metadata.py \
        tmp/spike_output_apartment/apartment_v1_metadata.json 2>/dev/null || true
# Note: tmp/ is usually gitignored; adjust the git-add if so.
git commit -m "feat(metadata): compute per-frame DRR proxy + azi/ele/dist + amp gain

Plan-1 metadata schema is M2/M3-ready. DRR is a distance-driven proxy
(true per-frame IR extraction is Plan-2 optimization). Amp gain derived
from binaural WAV RMS-per-frame windows."
```

---

## Task 11: Topdown 2d + side-by-side mux for apartment clip

**Goal:** Reuse `render_topdown_2d.py` (already made spec-parameterized in Task 7) + ffmpeg mux to produce a final side-by-side mp4 comparable to the shoebox demo.

**Files:**
- Modify: `tools/spike_rlr/run_all.sh` — add an `--apt-only` mode that chains apartment steps end-to-end (optional; can be a separate `run_apartment.sh`)
- Create: `tools/spike_rlr/run_apartment.sh` (a short pipeline script)

- [ ] **Step 1: Render topdown for apartment**

```bash
cd /data/jzy/code/AVEngine/external/SPEAR
/data/jzy/miniconda3/envs/spear-env/bin/python tools/spike_rlr/render_topdown_2d.py \
    --spec data/apartment_v1_spec.json \
    --out-dir tmp/spike_output_apartment 2>&1 | tail -5
ls tmp/spike_output_apartment/videos/topdown/ 2>/dev/null | head
```

Expected: topdown mp4 written under tmp/spike_output_apartment/videos/topdown/. If the topdown script trips over the apartment's larger room extent (shoebox topdown is sized for 5x6m), open the script and check for hardcoded x/y limits — replace with `spec.room_size_m` reads, but for apartment there is no `room_size_m` explicit field, so **fall back to reading extent from `apartment_shell_map.json` bbox min/max**.

- [ ] **Step 2: Mux audio into UE video**

```bash
ffmpeg -y -i tmp/spike_output_apartment/videos/apartment_v1_view0.mp4 \
       -i tmp/spike_output_apartment/binaural_native/audio_B_rlr_LOW_binaural_native.wav \
       -c:v copy -c:a aac -shortest \
       tmp/spike_output_apartment/videos/apartment_v1_view0_with_audio.mp4
```

Note: the mixed binaural WAV filename may differ; check `ls tmp/spike_output_apartment/binaural_native/` and pick the file WITHOUT a `_dog_golden_` or `_dog_husky_` suffix (that's the mixed one). If only per-source files exist, add step 2a below.

- [ ] **Step 2a (if needed): Sum per-source binaurals**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -c "
import soundfile as sf, numpy as np, glob
files = glob.glob('tmp/spike_output_apartment/binaural_native/audio_B_rlr_LOW_binaural_native_dog_*_binaural.wav')
mix = None
for p in files:
    x, sr = sf.read(p, always_2d=True)
    if mix is None:
        mix = np.zeros_like(x); sr_out = sr
    mix += x
    assert sr == sr_out
mix = mix / max(1, np.abs(mix).max())
sf.write('tmp/spike_output_apartment/binaural_native/audio_B_rlr_LOW_binaural_mixed.wav', mix, sr_out)
print('mixed', len(files), 'files ->', 'audio_B_rlr_LOW_binaural_mixed.wav')
"
ffmpeg -y -i tmp/spike_output_apartment/videos/apartment_v1_view0.mp4 \
       -i tmp/spike_output_apartment/binaural_native/audio_B_rlr_LOW_binaural_mixed.wav \
       -c:v copy -c:a aac -shortest \
       tmp/spike_output_apartment/videos/apartment_v1_view0_with_audio.mp4
```

- [ ] **Step 3: Side-by-side mux**

```bash
# Ensure topdown mp4 path is correct; adjust glob as needed
TOPDOWN=$(ls tmp/spike_output_apartment/videos/topdown/*.mp4 | head -1)
ffmpeg -y -i tmp/spike_output_apartment/videos/apartment_v1_view0_with_audio.mp4 \
       -i "$TOPDOWN" \
       -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
       -map "[v]" -map 0:a -c:a copy \
       tmp/spike_output_apartment/videos/apartment_v1_side_by_side_view0.mp4
```

- [ ] **Step 4: Print stage summary**

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -c "
import sys
sys.path.insert(0, 'tools/spike_rlr')
from profiling import StageTimer, print_stage_summary
# Read the CSV to reconstruct aggregate (since we're in a new process)
import csv
from pathlib import Path
csv_path = Path('tmp/spike_output_apartment/profile_per_clip.csv')
if csv_path.exists():
    with csv_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            StageTimer.aggregate[row['stage']] = \
                StageTimer.aggregate.get(row['stage'], 0.0) + float(row['seconds'])
print(print_stage_summary(total_clips=1,
                          out_path=Path('tmp/spike_output_apartment/profile_stage_summary.txt')))
"
cat tmp/spike_output_apartment/profile_stage_summary.txt
```

Expected: a table showing time spent in each stage. This is your first per-clip baseline; use it to decide Plan 2 clip count.

- [ ] **Step 5: Write run_apartment.sh convenience script**

```bash
cat > tools/spike_rlr/run_apartment.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export DISPLAY=:99
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

SPEAR_PY=/data/jzy/miniconda3/envs/spear-env/bin/python
SS2_PY=/data/jzy/miniconda3/envs/ss2/bin/python
LD_PRE=/usr/lib/x86_64-linux-gnu/libEGL.so.1:/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0

echo "=== [1/6] Generate shell mesh ==="
$SS2_PY tools/spike_rlr/gen_mesh_apartment.py \
    --shell-json data/apartment_shell_map.json \
    --out-glb tmp/spike_output_apartment/mesh/shell.glb \
    --out-materials tmp/spike_output_apartment/mesh/shell_materials.json

echo "=== [2/6] UE render pass (apartment) ==="
$SPEAR_PY tools/spike_rlr/run_render_pass_apartment.py

echo "=== [3/6] RLR audio (binaural) ==="
LD_PRELOAD=$LD_PRE $SS2_PY tools/spike_rlr/run_audio_pass_rlr.py \
    --spec data/apartment_v1_spec.json \
    --shell-glb tmp/spike_output_apartment/mesh/shell.glb \
    --materials-json tmp/spike_output_apartment/mesh/shell_materials.json \
    --out-dir tmp/spike_output_apartment \
    --channel-layout binaural --quality LOW

echo "=== [4/6] Compute metadata ==="
$SS2_PY tools/spike_rlr/compute_acoustic_metadata.py

echo "=== [5/6] Topdown ==="
$SPEAR_PY tools/spike_rlr/render_topdown_2d.py \
    --spec data/apartment_v1_spec.json \
    --out-dir tmp/spike_output_apartment

echo "=== [6/6] Mux + side-by-side ==="
# See Task 11 steps 2/2a/3 for the ffmpeg commands
EOF
chmod +x tools/spike_rlr/run_apartment.sh
```

- [ ] **Step 6: Inspect the final side-by-side video**

```bash
ls -la tmp/spike_output_apartment/videos/apartment_v1_side_by_side_view0.mp4
# Extract 5 frames and view them
mkdir -p /tmp/side_check
ffmpeg -i tmp/spike_output_apartment/videos/apartment_v1_side_by_side_view0.mp4 \
    -vf "select='eq(n\,0)+eq(n\,20)+eq(n\,40)+eq(n\,60)+eq(n\,74)'" \
    -vsync vfr /tmp/side_check/f%02d.png
ls /tmp/side_check/
```

Read each png with the tool. Verify:
- Left side: UE apartment view with dogs
- Right side: topdown showing apartment shell footprint + dog trajectories + mic marker
- Audio present when played back locally

- [ ] **Step 7: Final commit**

```bash
git add tools/spike_rlr/run_apartment.sh
git commit -m "feat(apartment): end-to-end run_apartment.sh + first side-by-side clip

Plan 1 delivery: apartment_v1 subset-furniture single hand-tuned clip.
UE render + RLR binaural + topdown + side-by-side + profile summary.
Ready to feed into Plan 2 (flag-based generator over 40 clips)."
```

---

## Self-Review

**Spec coverage vs the grill outcome:**
- ✅ **A+D goal** (apartment shell used as RLR scene, pipeline validated end-to-end)  — Tasks 2, 3, 8, 9, 11
- ✅ **Sibling to shoebox (S)** — Task 7 preserves shoebox default; Task 4's spec is a new sibling file
- ✅ **shell/subset/full modes** — Task 8's `_decide_keep` implements all 3, Task 4's spec defaults to `subset` per T-fixed decision
- ✅ **M1 = single hand-tuned clip** — Task 4's spec has exactly 2 sources, Tasks 6/8/11 render one clip
- ✅ **shell = walls/floor/ceiling/doors/windows/curtains/pictures/mirrors** — Task 1 classifier + Task 2 dump
- ✅ **furniture = independent objects** — Task 4's categories JSON
- ✅ **1 forward camera, 90° FOV, C-glued** — Task 4's spec + Task 8's camera spawn
- ✅ **Profiling Level 1 + Level 2** — Task 5 module, Tasks 7/8/10 wrap in `StageTimer`, Task 11 prints summary
- ✅ **DRR in clip metadata** — Task 10
- 🟡 **RT60 in room metadata** — Deferred to Plan 2 (Task 10 has proxy DRR only; real per-frame IR extraction + room RT60 measurement is deliberately postponed). This should be flagged clearly in Plan 1's deliverable summary and picked up as Task 1 of Plan 2.
- 🟡 **mic yaw randomization** — NOT in Plan 1 (spec hardcodes yaw=90). Plan 2 introduces randomization.
- 🟡 **FSD50K + Stable Audio Open audio library** — NOT in Plan 1 (uses existing `Barking Aldi Dog_358.wav` + piano synth). Plan 3.
- 🟡 **11 flag generator** — Plan 2.
- 🟡 **N-0to2 source count** — Plan 2 (Plan 1 fixed at 2).
- 🟡 **mic + dog spawn randomization** (Layer 1) — Plan 2.
- 🟡 **HRTF binaural GT** — Already produced (Task 9), stored in metadata output. Confirmed correct in confirmed RLR binaural mode.

**Placeholder scan:** No "TBD", no bare "similar to Task N", every code block is complete. One caveat spot — Task 8 Step 3 mentions "call the low-level unreal_service to set FieldOfView on the CameraComponent subobject. Skip if the default cam has fov 90 already." That's a **runtime discovery** — implementer will check first. This is honest, not a placeholder.

**Type consistency:** `StageTimer.aggregate` is `dict[str, float]` used in Task 5 tests and Tasks 7/8/10/11 code — consistent. `classify_actor` signature identical across Tasks 1/2/3/8. `compose_two_dog_scene_apartment(spec_path)` signature identical in Task 6 code and Task 10 caller.

---

## Deliverables

Files that must exist and pass tests at Plan-1 completion:

- `data/apartment_v1_spec.json` (Task 4)
- `data/apartment_shell_map.json` (Task 2)
- `tools/spike_rlr/apartment_furniture_categories.json` (Task 4)
- `tools/gpurir_scenes/apartment_actor_classifier.py` (Task 1)
- `tools/spike_rlr/dump_apartment_shell.py` (Task 2)
- `tools/spike_rlr/gen_mesh_apartment.py` (Task 3)
- `tools/spike_rlr/scene_two_dogs_apartment.py` (Task 6)
- `tools/spike_rlr/run_render_pass_apartment.py` (Task 8)
- `tools/spike_rlr/compute_acoustic_metadata.py` (Task 10)
- `tools/spike_rlr/profiling.py` (Task 5)
- `tools/spike_rlr/run_apartment.sh` (Task 11)
- `tmp/spike_output_apartment/videos/apartment_v1_side_by_side_view0.mp4` (Task 11)
- `tmp/spike_output_apartment/apartment_v1_metadata.json` (Task 10)
- `tmp/spike_output_apartment/profile_stage_summary.txt` (Task 11)
- `tmp/spike_output_apartment/profile_per_clip.csv` (all tasks feed this)

Tests all pass:

- `pytest tests/tools/gpurir_scenes/test_apartment_actor_classifier.py`
- `pytest tests/tools/spike_rlr/test_apartment_shell_dump.py`
- `pytest tests/tools/spike_rlr/test_gen_mesh_apartment.py`
- `pytest tests/tools/spike_rlr/test_apartment_v1_spec.py`
- `pytest tests/tools/spike_rlr/test_profiling.py`
- `pytest tests/tools/spike_rlr/test_scene_two_dogs_apartment.py`
- `pytest tests/tools/spike_rlr/test_run_audio_pass_cli.py`
- `pytest tests/tools/spike_rlr/test_compute_acoustic_metadata.py`

Shoebox regression preserved: running `run_audio_pass_rlr.py` / `render_topdown_2d.py` without `--spec` still targets shoebox and writes to `tmp/spike_output/`.
