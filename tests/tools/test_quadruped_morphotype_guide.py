"""Contracts and an optional Blender smoke for render-only morphotype guides."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import subprocess

import pytest

from tools.quadruped_morphotype_guide import (
    MORPHOTYPE_GUIDE_SCHEMA,
    MorphotypeGuideError,
    build_morphotype_guide_plan,
    load_morphotype_guide_profile,
    parse_morphotype_guide_profile,
)


SPEAR_ROOT = Path(__file__).resolve().parents[2]
AVENGINE_ROOT = SPEAR_ROOT.parents[1]
RENDERER = SPEAR_ROOT / "tools/blender_render_glb_animation.py"
PROFILE = (
    SPEAR_ROOT
    / "tests/fixtures/quadruped_morphotype_guide/short_leg_short_tail_v1.json"
)
PRODUCTION_PROFILE = (
    SPEAR_ROOT
    / "data/controlled_source_attributes_v1/contracts"
    / "quadruped_morphotype_guide_short_leg_short_tail_v1.json"
)
DOG_FIXTURE = AVENGINE_ROOT / "assets/mesh_library/quaternius_animalpack/Dog.glb"


def _bone(name, parent, head, tail):
    return {
        "name": name,
        "parent": parent,
        "children": [],
        "head_world": list(head),
        "tail_world": list(tail),
    }


def _synthetic_rig():
    records = [
        _bone("q0", None, (-0.2, 0.0, 0.65), (0.0, 0.0, 0.65)),
        _bone("q1", "q0", (0.0, 0.0, 0.65), (0.3, 0.0, 0.65)),
        _bone("n0", "q1", (0.3, 0.0, 0.65), (0.45, 0.0, 0.72)),
        _bone("n1", "n0", (0.45, 0.0, 0.72), (0.62, 0.0, 0.72)),
        _bone("t0", "q0", (-0.2, 0.0, 0.65), (-0.42, 0.0, 0.67)),
        _bone("t1", "t0", (-0.42, 0.0, 0.67), (-0.68, 0.0, 0.72)),
        _bone("a0", "q1", (0.25, -0.15, 0.55), (0.24, -0.15, 0.13)),
        _bone("a1", "a0", (0.24, -0.15, 0.05), (0.36, -0.15, 0.05)),
        _bone("b0", "q1", (0.25, 0.15, 0.55), (0.24, 0.15, 0.13)),
        _bone("b1", "b0", (0.24, 0.15, 0.05), (0.36, 0.15, 0.05)),
        _bone("c0", "q0", (-0.15, -0.15, 0.57), (-0.14, -0.15, 0.13)),
        _bone("c1", "c0", (-0.14, -0.15, 0.05), (-0.02, -0.15, 0.05)),
        _bone("d0", "q0", (-0.15, 0.15, 0.57), (-0.14, 0.15, 0.13)),
        _bone("d1", "d0", (-0.14, 0.15, 0.05), (-0.02, 0.15, 0.05)),
    ]
    by_name = {record["name"]: record for record in records}
    for record in records:
        parent = record["parent"]
        if parent is not None:
            by_name[parent]["children"].append(record["name"])
    semantics = SimpleNamespace(
        axial=("q0", "q1"),
        head_chain=("n0", "n1"),
        tail_chain=("t0", "t1"),
        front_side_negative=("a0", "a1"),
        front_side_positive=("b0", "b1"),
        hind_side_negative=("c0", "c1"),
        hind_side_positive=("d0", "d1"),
        foot_leaves=("a1", "b1", "c1", "d1"),
    )
    return semantics, records


def _profile_document():
    return json.loads(PROFILE.read_text(encoding="utf-8"))


def _segment_vector(segment):
    return tuple(
        segment.tail_world[index] - segment.head_world[index]
        for index in range(3)
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_profile_fixture_is_strict_and_bounded():
    profile = load_morphotype_guide_profile(PROFILE)

    assert profile.leg_length_ratio == 0.65
    assert profile.tail_length_ratio == 0.45
    assert profile.maximum_ground_residual_height_ratio == 0.005


def test_production_profile_matches_the_validated_fixture():
    production = load_morphotype_guide_profile(PRODUCTION_PROFILE)
    fixture = load_morphotype_guide_profile(PROFILE)

    assert production == fixture


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.update({"schema": "unrecognized_schema"}),
            "profile.schema must be",
        ),
        (lambda value: value.update({"breed": "example"}), "keys must be exactly"),
        (
            lambda value: value["transforms"].update({"leg_length_ratio": 0.2}),
            "leg_length_ratio must be in",
        ),
        (
            lambda value: value["transforms"].update({"tail_length_ratio": 0.1}),
            "tail_length_ratio must be in",
        ),
        (
            lambda value: value["transforms"].update(
                {"leg_length_ratio": True}
            ),
            "must be a finite number",
        ),
        (
            lambda value: value["validation"].update(
                {"maximum_ground_residual_height_ratio": 0.02}
            ),
            "maximum_ground_residual_height_ratio must be in",
        ),
        (
            lambda value: value["transforms"].update(
                {"leg_length_ratio": 1.0, "tail_length_ratio": 1.0}
            ),
            "cannot be a no-op",
        ),
    ],
)
def test_profile_rejects_unknown_unbounded_or_implicit_values(mutation, message):
    document = _profile_document()
    mutation(document)

    with pytest.raises(MorphotypeGuideError, match=message):
        parse_morphotype_guide_profile(document)


def test_profile_loader_rejects_duplicate_json_keys(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        (
            '{"schema":"avengine_quadruped_morphotype_guide_v1",'
            '"schema":"avengine_quadruped_morphotype_guide_v1",'
            '"transforms":{"leg_length_ratio":0.65,"tail_length_ratio":0.45},'
            '"validation":{"maximum_ground_residual_height_ratio":0.005}}'
        ),
        encoding="utf-8",
    )

    with pytest.raises(MorphotypeGuideError, match="duplicate JSON key"):
        load_morphotype_guide_profile(duplicate)


def test_plan_keeps_four_feet_and_translates_torso_without_squashing():
    semantics, records = _synthetic_rig()
    profile = load_morphotype_guide_profile(PROFILE)
    plan = build_morphotype_guide_plan(
        profile,
        semantics,
        records,
        bbox_height=1.0,
    )
    by_name = {record["name"]: record for record in records}

    assert len(plan.limb_chains) == 4
    assert set(plan.foot_leaves) == {"a1", "b1", "c1", "d1"}
    assert plan.maximum_target_foot_displacement_height_ratio == 0.0
    for foot in plan.foot_leaves:
        assert plan.targets[foot].head_world == tuple(by_name[foot]["head_world"])
    for name in plan.body_bones:
        assert _segment_vector(plan.targets[name]) == pytest.approx(
            tuple(
                by_name[name]["tail_world"][index]
                - by_name[name]["head_world"][index]
                for index in range(3)
            ),
            abs=1.0e-12,
        )
    assert plan.maximum_torso_segment_scale_error <= 1.0e-12
    assert plan.body_drop > 0.0
    assert plan.body_drop_height_ratio <= 0.25
    assert all(
        abs(value - profile.leg_length_ratio) <= 0.05
        for value in plan.effective_leg_length_ratios.values()
    )
    assert plan.target_tail_length / plan.source_tail_length == pytest.approx(
        profile.tail_length_ratio,
        abs=1.0e-12,
    )


def test_plan_fails_closed_on_broken_limb_chain():
    semantics, records = _synthetic_rig()
    profile = load_morphotype_guide_profile(PROFILE)
    broken = [dict(record) for record in records]
    next(record for record in broken if record["name"] == "a1")["parent"] = "q0"

    with pytest.raises(MorphotypeGuideError, match="not parent-contiguous"):
        build_morphotype_guide_plan(
            profile,
            semantics,
            broken,
            bbox_height=1.0,
        )


def test_plan_fails_closed_on_missing_tail_chain():
    semantics, records = _synthetic_rig()
    profile = load_morphotype_guide_profile(PROFILE)
    semantics = SimpleNamespace(**vars(semantics))
    semantics.tail_chain = ()

    with pytest.raises(MorphotypeGuideError, match="at least 1 bone"):
        build_morphotype_guide_plan(
            profile,
            semantics,
            records,
            bbox_height=1.0,
        )


def test_plan_supports_one_validated_tail_segment():
    semantics, records = _synthetic_rig()
    profile = load_morphotype_guide_profile(PROFILE)
    semantics = SimpleNamespace(**vars(semantics))
    semantics.tail_chain = ("t0",)
    records = [record for record in records if record["name"] != "t1"]
    next(record for record in records if record["name"] == "t0")["children"] = []

    plan = build_morphotype_guide_plan(
        profile,
        semantics,
        records,
        bbox_height=1.0,
    )

    assert plan.tail_chain == ("t0",)
    assert plan.target_tail_length / plan.source_tail_length == pytest.approx(
        profile.tail_length_ratio,
        abs=1.0e-12,
    )


def test_plan_fails_closed_when_four_feet_are_not_grounded():
    semantics, records = _synthetic_rig()
    profile = load_morphotype_guide_profile(PROFILE)
    lifted = [dict(record) for record in records]
    next(record for record in lifted if record["name"] == "d1")[
        "head_world"
    ] = [-0.14, 0.15, 0.08]

    with pytest.raises(MorphotypeGuideError, match="not on one ground plane"):
        build_morphotype_guide_plan(
            profile,
            semantics,
            lifted,
            bbox_height=1.0,
        )


def test_renderer_wires_profile_only_into_explicit_rest_pose_path():
    source = RENDERER.read_text(encoding="utf-8")

    assert "--quadruped-morphotype-guide-profile" in source
    assert "--quadruped-morphotype-guide-profile requires --rest-pose" in source
    assert "load_morphotype_guide_profile" in source
    assert "apply_quadruped_morphotype_guide" in source
    assert "infer_canonical_quadruped" in source
    assert "torso_transform" in source
    assert "source_asset_unchanged" in source
    assert "bpy.ops.export_scene" not in source
    assert "corgi" not in source.lower()


def test_blender_renders_morphotype_fixture_without_changing_source(tmp_path):
    blender = shutil.which("blender")
    if blender is None:
        pytest.skip("Blender executable is unavailable on PATH")
    if not DOG_FIXTURE.is_file():
        pytest.skip("repository Quaternius Dog fixture is unavailable")
    if not Path(blender).resolve().name == "blender":
        pytest.skip("resolved Blender executable is invalid")
    output = tmp_path / "render"
    source_sha256 = _sha256(DOG_FIXTURE)
    command = [
        blender,
        "--background",
        "--factory-startup",
        "--python",
        str(RENDERER),
        "--",
        "--input",
        str(DOG_FIXTURE),
        "--rest-pose",
        "--orthographic",
        "--view",
        "side",
        "--quadruped-morphotype-guide-profile",
        str(PROFILE),
        "--quadruped-far-limb-offset-ratio",
        "0.30",
        "--pose-template-clay-color",
        "#b88b6a",
        "--ground-plane",
        "--output-dir",
        str(output),
        "--n-frames",
        "1",
        "--width",
        "96",
        "--height",
        "96",
        "--samples",
        "1",
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "MORPHOTYPE_GUIDE_OK" in result.stdout
    line = next(
        item
        for item in result.stdout.splitlines()
        if item.startswith("[morphotype-guide] ")
    )
    diagnostics = json.loads(line.split(" ", 1)[1])
    assert diagnostics["schema"] == MORPHOTYPE_GUIDE_SCHEMA
    assert diagnostics["limb_chain_lengths"] == {
        "front_side_negative": 4,
        "front_side_positive": 4,
        "hind_side_negative": 4,
        "hind_side_positive": 4,
    }
    assert diagnostics["tail_chain_length"] == 5
    assert diagnostics["maximum_foot_ground_residual_height_ratio"] <= 0.005
    assert diagnostics["mesh_ground_residual_height_ratio"] <= 0.005
    assert diagnostics["maximum_cross_section_scale_error"] <= 1.0e-6
    assert diagnostics["maximum_torso_segment_scale_error"] <= 1.0e-6
    assert diagnostics["realized_tail_length_ratio"] == pytest.approx(
        0.45,
        abs=1.0e-6,
    )
    assert diagnostics["render_only"] is True
    assert diagnostics["source_asset_unchanged"] is True
    assert (output / "frame_0000.png").is_file()
    assert not list(output.glob("*.glb"))
    assert _sha256(DOG_FIXTURE) == source_sha256
