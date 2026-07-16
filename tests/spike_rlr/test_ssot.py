"""SSOT (shoebox_v2_spec.json + acoustic_material_db.json) schema + consistency tests.

Guards: any surface/furniture material tag referenced by the scene spec
must resolve in the material DB; geometry constraints (sofa fully inside
room, wall clearance) must hold; per-tag audio contract fields must be
present. These invariants are what let UE/Habitat/RLR all read the same
files and produce a comparable A/B/C spike output.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO_ROOT / "data" / "shoebox_v2_spec.json"
DB_PATH = REPO_ROOT / "data" / "acoustic_material_db.json"


@pytest.fixture(scope="module")
def spec():
    with open(SPEC_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def db():
    with open(DB_PATH) as f:
        return json.load(f)


# ---------- schema: top-level keys ------------------------------------------

def test_spec_has_required_top_level_keys(spec):
    required = {
        "spec_version", "coordinate_frame", "room_size_m", "wall_clearance_m",
        "mic", "camera_configs", "render_config", "audio_config",
        "surfaces", "furniture", "source_height_m", "sources",
        "occlusion_event",
    }
    missing = required - set(spec.keys())
    assert not missing, f"spec missing required keys: {missing}"


def test_db_has_meta_and_at_least_one_material(db):
    assert "_meta" in db
    materials = {k for k in db.keys() if not k.startswith("_")}
    assert len(materials) >= 4, "DB should contain at least 4 materials for shoebox v2"


# ---------- geometry --------------------------------------------------------

def test_room_size_is_3_tuple_positive(spec):
    rs = spec["room_size_m"]
    assert len(rs) == 3
    assert all(x > 0 for x in rs), f"room_size_m has non-positive dim: {rs}"


def test_mic_inside_room(spec):
    rs = spec["room_size_m"]
    mic = spec["mic"]["pos_m"]
    for i, (m, r) in enumerate(zip(mic, rs)):
        assert 0 <= m <= r, f"mic[{i}]={m} outside [0, {r}]"


def test_sofa_fully_inside_room_with_clearance(spec):
    rs = spec["room_size_m"]
    clr = spec["wall_clearance_m"]
    for f in spec["furniture"]:
        assert f["shape"] == "box", f"only box shape supported for spike, got {f['shape']}"
        c = f["center_m"]
        s = f["size_m"]
        for axis in range(3):
            lo = c[axis] - s[axis] / 2
            hi = c[axis] + s[axis] / 2
            # X/Y need wall clearance; Z=0 allowed (sitting on floor)
            if axis in (0, 1):
                assert lo >= clr - 1e-6, (
                    f"furniture {f['name']} axis {axis} lo={lo} violates "
                    f"wall_clearance={clr}"
                )
                assert hi <= rs[axis] - clr + 1e-6, (
                    f"furniture {f['name']} axis {axis} hi={hi} violates "
                    f"wall_clearance={clr} (room {rs[axis]})"
                )
            else:
                # Z: allow sofa to sit on floor, only enforce inside ceiling
                assert lo >= -1e-6, f"furniture {f['name']} sinks below floor: lo={lo}"
                assert hi <= rs[axis] + 1e-6, (
                    f"furniture {f['name']} pokes through ceiling: hi={hi} > {rs[axis]}"
                )


def test_husky_trajectory_endpoints_inside_room_with_clearance(spec):
    rs = spec["room_size_m"]
    clr = spec["wall_clearance_m"]
    husky = next(s for s in spec["sources"] if s["tag"] == "dog_husky")
    for seg in husky["trajectory_m"]:
        for pos in (seg["start_m"], seg["end_m"]):
            for axis in range(2):
                assert clr - 1e-6 <= pos[axis] <= rs[axis] - clr + 1e-6, (
                    f"husky segment {seg['phase']} axis {axis} pos={pos[axis]} "
                    f"outside [{clr}, {rs[axis]-clr}]"
                )


def test_husky_end_position_behind_sofa_center_line(spec):
    """Husky end must be on the far side of sofa center from mic (occlusion sanity)."""
    mic = spec["mic"]["pos_m"]
    sofa = next(f for f in spec["furniture"] if f["name"] == "sofa")
    husky = next(s for s in spec["sources"] if s["tag"] == "dog_husky")
    end = husky["trajectory_m"][-1]["end_m"]
    # Y axis: mic Y < sofa Y < husky end Y (sofa between mic and husky)
    assert mic[1] < sofa["center_m"][1] < end[1], (
        f"expected mic_Y({mic[1]}) < sofa_Y({sofa['center_m'][1]}) < husky_end_Y({end[1]}); "
        f"sofa is not between mic and husky end"
    )


def test_golden_behind_camera(spec):
    """Golden's linear path should be on the mic's back side (Y < mic Y)."""
    mic = spec["mic"]["pos_m"]
    golden = next(s for s in spec["sources"] if s["tag"] == "dog_golden")
    for pos in (golden["start_pos_m"], golden["end_pos_m"]):
        assert pos[1] < mic[1], (
            f"golden position Y={pos[1]} should be behind mic Y={mic[1]} "
            f"(smaller Y = behind camera facing +Y)"
        )


# ---------- surface -> material tag consistency -----------------------------

def test_all_surface_materials_resolve(spec, db):
    referenced = set(spec["surfaces"].values())
    known = {k for k in db.keys() if not k.startswith("_")}
    missing = referenced - known
    assert not missing, f"surfaces reference unknown materials: {missing}"


def test_all_furniture_materials_resolve(spec, db):
    referenced = {f["material"] for f in spec["furniture"]}
    known = {k for k in db.keys() if not k.startswith("_")}
    missing = referenced - known
    assert not missing, f"furniture reference unknown materials: {missing}"


# ---------- material DB shape -----------------------------------------------

def test_db_materials_have_correct_alpha_length(db):
    for name, mat in db.items():
        if name.startswith("_"):
            continue
        assert "alpha" in mat, f"material {name} missing alpha"
        assert len(mat["alpha"]) == 4, (
            f"material {name} alpha has {len(mat['alpha'])} bands, expected 4"
        )
        assert all(0.0 <= a <= 1.0 for a in mat["alpha"]), (
            f"material {name} alpha out of [0,1]: {mat['alpha']}"
        )
        assert "scattering" in mat and 0.0 <= mat["scattering"] <= 1.0
        assert "transmission" in mat and len(mat["transmission"]) == 4


# ---------- source contract -------------------------------------------------

def test_sources_have_required_fields(spec):
    for s in spec["sources"]:
        assert "tag" in s and "audio_lookup" in s and "kind" in s


def test_occlusion_event_time_math(spec):
    ev = spec["occlusion_event"]
    fps = spec["render_config"]["fps"]
    assert abs(ev["start_time_s"] - ev["start_frame"] / fps) < 1e-6
    assert abs(ev["end_time_s"] - (ev["end_frame"] + 1) / fps) < 1e-6 or \
           abs(ev["end_time_s"] - ev["end_frame"] / fps) < 0.1  # allow off-by-one
    assert ev["fully_occluded_from_frame"] >= ev["start_frame"]
    assert ev["expected_energy_drop_dB_min"] > 0
