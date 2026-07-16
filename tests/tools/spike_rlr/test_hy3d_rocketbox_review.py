"""Contract tests for the hash-locked Hunyuan/Rocketbox motion review."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
TASK3_PRODUCER = REPO / "tools" / "blender_bind_hy3d_to_rocketbox.py"
TASK4_PRODUCER = REPO / "tools" / "blender_render_hy3d_rocketbox_review.py"
PIXEL_QA_CHECKS = (
    "hands_attached",
    "hands_not_duplicated",
    "pieces_nonblank",
    "arm_torso_regions_clean",
    "thigh_regions_clean",
    "sleeves_seam_free",
    "feet_not_inverted",
    "floor_cards_absent",
    "leg_gap_fans_absent",
    "mesh_explosions_absent",
)

import hy3d_rocketbox_review as review_contract  # noqa: E402
from hy3d_rocketbox_review import (  # noqa: E402
    EXPECTED_ASSET_IDS,
    REQUIRED_MOTIONS,
    REQUIRED_VIEWS,
    Hy3DRocketboxNotApproved,
    assert_asset_approved,
    assert_pair_approved,
    read_review_state,
    record_decision,
    sha256_file,
    validated_review_snapshot,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _media_name(motion: str, view: str) -> str:
    return f"{motion}_{view}.mp4"


def _glb_name(motion: str) -> str:
    return f"bound_{motion}.glb"


def _descriptor(path: Path) -> dict[str, str]:
    return {"filename": path.name, "sha256": sha256_file(path)}


def _task3_manifest_builder():
    source = TASK3_PRODUCER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        "sha256_file",
        "require_regular_file",
        "file_descriptor",
        "build_bind_manifest",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    namespace = {"Path": Path, "hashlib": hashlib}
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, str(TASK3_PRODUCER), "exec"), namespace)
    return namespace["build_bind_manifest"]


def _task4_ready_builder():
    source = TASK4_PRODUCER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_direct_attempt_payload"
    )
    constant_names = {
        "DIRECT_ATTEMPT_READY_SCHEMA",
        "DIRECT_ATTEMPT_REJECTED_SCHEMA",
        "PIXEL_QA_FILENAME",
    }
    namespace = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in constant_names
    }
    assert set(namespace) == constant_names
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(TASK4_PRODUCER), "exec"), namespace)
    return namespace["build_direct_attempt_payload"]


def _artifact_snapshot(asset_dir: Path) -> dict[str, object]:
    bind_manifest = json.loads(
        (asset_dir / "bind_manifest.json").read_text(encoding="utf-8")
    )
    return {
        "schema_version": "hy3d_rocketbox_artifact_snapshot_v1",
        "asset_id": bind_manifest["asset_id"],
        "bind_manifest_sha256": sha256_file(asset_dir / "bind_manifest.json"),
        "review_manifest_sha256": sha256_file(asset_dir / "review_manifest.json"),
        "bound_blend": _descriptor(asset_dir / "bound.blend"),
        "glbs": {
            motion: _descriptor(asset_dir / _glb_name(motion))
            for motion in REQUIRED_MOTIONS
        },
        "videos": {
            motion: {
                view: _descriptor(asset_dir / _media_name(motion, view))
                for view in REQUIRED_VIEWS
            }
            for motion in REQUIRED_MOTIONS
        },
        "bind_metrics": _descriptor(asset_dir / "bind_metrics.json"),
        "contact_sheet": _descriptor(asset_dir / "bind_contact_sheet.png"),
    }


def _write_pixel_qa(asset_dir: Path) -> None:
    qa = {
        "schema_version": "hy3d_rocketbox_pixel_qa_v1",
        "asset_id": json.loads(
            (asset_dir / "bind_manifest.json").read_text(encoding="utf-8")
        )["asset_id"],
        "decision": "ready",
        "reviewer": "pixel-reviewer",
        "reviewed_at": "2026-07-10T12:00:00+00:00",
        "notes": "all rendered views inspected",
        "checks": {check: True for check in PIXEL_QA_CHECKS},
        "expected_artifact_snapshot": _artifact_snapshot(asset_dir),
    }
    (asset_dir / "pixel_qa.json").write_text(
        json.dumps(qa, sort_keys=True), encoding="utf-8"
    )


def _write_ready_record(asset_dir: Path, *, refresh_pixel_qa: bool = True) -> None:
    bind_manifest = json.loads(
        (asset_dir / "bind_manifest.json").read_text(encoding="utf-8")
    )
    if refresh_pixel_qa:
        _write_pixel_qa(asset_dir)
    pixel_qa_path = asset_dir / "pixel_qa.json"
    pixel_qa = json.loads(pixel_qa_path.read_text(encoding="utf-8"))
    ready = _task4_ready_builder()(
        bind_manifest["asset_id"],
        "ready",
        pixel_qa,
        sha256_file(pixel_qa_path),
        pixel_qa_path,
        _artifact_snapshot(asset_dir),
    )
    (asset_dir / "direct_attempt_ready.json").write_text(
        json.dumps(ready, sort_keys=True), encoding="utf-8"
    )


def _write_ready_fixture(review_root: Path, asset_id: str) -> Path:
    asset_dir = review_root / asset_id
    asset_dir.mkdir(parents=True)
    reference = f"{asset_id}:approved-flux-reference".encode("ascii")
    (asset_dir / "reference.png").write_bytes(reference)
    (asset_dir / "bound.blend").write_bytes(
        f"{asset_id}:bound-blend".encode("ascii")
    )
    (asset_dir / "cleaned.obj").write_bytes(f"{asset_id}:cleaned".encode("ascii"))
    (asset_dir / "bind_metrics.json").write_text(
        json.dumps({"asset_id": asset_id}), encoding="utf-8"
    )
    (asset_dir / "bind_contact_sheet.png").write_bytes(
        f"{asset_id}:contact-sheet".encode("ascii")
    )
    glbs = {}
    for motion in REQUIRED_MOTIONS:
        payload = f"{asset_id}:{motion}:glb".encode("ascii")
        filename = _glb_name(motion)
        (asset_dir / filename).write_bytes(payload)
        glbs[motion] = {"filename": filename, "sha256": _sha256(payload)}
    action_metrics = {
        "walk": {
            "action_name": f"{asset_id}_walk_retarget",
            "frame_start": 1,
            "frame_end": 31,
        },
        "idle": {
            "action_name": f"{asset_id}_idle_neutral_01_retarget",
            "frame_start": 1,
            "frame_end": 61,
        },
    }
    bind_manifest = _task3_manifest_builder()(
        SimpleNamespace(asset_id=asset_id),
        asset_dir,
        action_metrics,
        {"source_sha256": "a" * 64},
        {"source_current_sha256": "a" * 64},
        0.0,
        {
            "baseline_blend": {
                "filename": "retarget.blend",
                "sha256": "b" * 64,
                "size_bytes": 42,
            }
        },
    )
    (asset_dir / "bind_manifest.json").write_text(
        json.dumps(bind_manifest, sort_keys=True), encoding="utf-8"
    )
    videos = {}
    for motion in REQUIRED_MOTIONS:
        videos[motion] = {}
        for view in REQUIRED_VIEWS:
            payload = f"{asset_id}:{motion}:{view}:video".encode("ascii")
            filename = _media_name(motion, view)
            (asset_dir / filename).write_bytes(payload)
            videos[motion][view] = {"filename": filename, "sha256": _sha256(payload)}
    review_manifest = {
        "schema_version": "hy3d_rocketbox_review_manifest_v1",
        "asset_id": asset_id,
        "bind_manifest_sha256": sha256_file(asset_dir / "bind_manifest.json"),
        "glbs": glbs,
        "videos": videos,
    }
    (asset_dir / "review_manifest.json").write_text(
        json.dumps(review_manifest, sort_keys=True), encoding="utf-8"
    )
    _write_ready_record(asset_dir)
    return asset_dir


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "reviews"
    for asset_id in EXPECTED_ASSET_IDS:
        _write_ready_fixture(root, asset_id)
    return root


def _snapshot(asset_dir: Path) -> dict[str, str]:
    return validated_review_snapshot(asset_dir)[3]


def _replace_video_fixture(
    asset_dir: Path, motion: str, view: str, payload: bytes
) -> None:
    video_path = asset_dir / _media_name(motion, view)
    video_path.write_bytes(payload)
    manifest_path = asset_dir / "review_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["videos"][motion][view]["sha256"] = sha256_file(video_path)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    _write_ready_record(asset_dir)


def test_contract_has_exact_two_assets_two_motions_and_three_views():
    assert EXPECTED_ASSET_IDS == (
        "rocketbox_male_adult_01",
        "rocketbox_female_adult_01",
    )
    assert REQUIRED_MOTIONS == ("walk", "idle")
    assert REQUIRED_VIEWS == ("front", "side", "feet")


def test_bind_manifest_uses_task3_names_and_separate_action_names(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]

    bind_manifest, _, _, _ = validated_review_snapshot(asset_dir)

    assert bind_manifest["glbs"] == {
        "walk": _descriptor(asset_dir / "bound_walk.glb"),
        "idle": _descriptor(asset_dir / "bound_idle.glb"),
    }
    assert set(bind_manifest["glbs"]["walk"]) == {"filename", "sha256"}
    assert set(bind_manifest["glbs"]["idle"]) == {"filename", "sha256"}
    assert set(bind_manifest["bound_blend"]) == {"filename", "sha256"}
    assert set(bind_manifest["reference"]) == {"filename", "sha256"}
    assert set(bind_manifest["action_names"]) == {"walk", "idle"}
    assert bind_manifest["action_names"]["walk"] != bind_manifest["action_names"]["idle"]
    assert bind_manifest["floor_z_m"] == 0.0
    assert bind_manifest["artifacts"]["bound_walk_glb"] == bind_manifest["glbs"]["walk"]
    assert bind_manifest["source_hashes"] == {
        "source_sha256": "a" * 64,
        "source_current_sha256": "a" * 64,
    }


def test_fixture_manifest_is_built_by_the_current_task3_producer(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    actual = json.loads(
        (asset_dir / "bind_manifest.json").read_text(encoding="utf-8")
    )

    assert actual == _task3_manifest_builder()(
        SimpleNamespace(asset_id=EXPECTED_ASSET_IDS[0]),
        asset_dir,
        {
            "walk": {
                "action_name": f"{EXPECTED_ASSET_IDS[0]}_walk_retarget",
                "frame_start": 1,
                "frame_end": 31,
            },
            "idle": {
                "action_name": f"{EXPECTED_ASSET_IDS[0]}_idle_neutral_01_retarget",
                "frame_start": 1,
                "frame_end": 61,
            },
        },
        {"source_sha256": "a" * 64},
        {"source_current_sha256": "a" * 64},
        0.0,
        {
            "baseline_blend": {
                "filename": "retarget.blend",
                "sha256": "b" * 64,
                "size_bytes": 42,
            }
        },
    )


def test_ready_fixture_is_built_by_the_current_task4_producer(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    pixel_qa_path = asset_dir / "pixel_qa.json"
    pixel_qa = json.loads(pixel_qa_path.read_text(encoding="utf-8"))
    actual = json.loads(
        (asset_dir / "direct_attempt_ready.json").read_text(encoding="utf-8")
    )

    assert actual == _task4_ready_builder()(
        EXPECTED_ASSET_IDS[0],
        "ready",
        pixel_qa,
        sha256_file(pixel_qa_path),
        pixel_qa_path,
        _artifact_snapshot(asset_dir),
    )


def test_contract_rejects_legacy_unprefixed_glb_names(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    (asset_dir / "bound_walk.glb").rename(asset_dir / "walk.glb")

    with pytest.raises(ValueError, match="bound walk GLB.*missing"):
        validated_review_snapshot(asset_dir)


def test_contract_rejects_action_name_inside_a_file_descriptor(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    manifest_path = asset_dir / "bind_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["glbs"]["walk"]["action_name"] = "must-be-independent"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly filename and sha256"):
        validated_review_snapshot(asset_dir)


def test_contract_rejects_identical_walk_and_idle_action_names(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    manifest_path = asset_dir / "bind_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["action_names"]["idle"] = manifest["action_names"]["walk"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="action names.*different"):
        validated_review_snapshot(asset_dir)


def test_snapshot_binds_all_task4_pixel_qa_artifacts(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]

    bind_manifest, review_manifest, captured, snapshot = validated_review_snapshot(
        asset_dir
    )

    assert bind_manifest["asset_id"] == EXPECTED_ASSET_IDS[0]
    assert review_manifest["bind_manifest_sha256"] == snapshot["bind_manifest_sha256"]
    assert set(captured) == {
        "bind_manifest",
        "review_manifest",
        "direct_attempt_ready",
        "pixel_qa",
        "reference",
        "bound_blend",
        "bind_metrics",
        "contact_sheet",
        "walk_glb",
        "idle_glb",
        "walk_front",
        "walk_side",
        "walk_feet",
        "idle_front",
        "idle_side",
        "idle_feet",
    }
    assert captured["bind_manifest"] == (asset_dir / "bind_manifest.json").read_bytes()
    assert captured["review_manifest"] == (
        asset_dir / "review_manifest.json"
    ).read_bytes()
    assert captured["reference"] == (asset_dir / "reference.png").read_bytes()
    assert captured["direct_attempt_ready"] == (
        asset_dir / "direct_attempt_ready.json"
    ).read_bytes()
    assert captured["pixel_qa"] == (asset_dir / "pixel_qa.json").read_bytes()
    assert captured["bound_blend"] == (asset_dir / "bound.blend").read_bytes()
    assert captured["bind_metrics"] == (asset_dir / "bind_metrics.json").read_bytes()
    assert captured["contact_sheet"] == (
        asset_dir / "bind_contact_sheet.png"
    ).read_bytes()
    assert captured["walk_glb"] == (asset_dir / "bound_walk.glb").read_bytes()
    assert captured["idle_glb"] == (asset_dir / "bound_idle.glb").read_bytes()
    with pytest.raises(TypeError):
        captured["reference"] = b"replacement"
    assert snapshot["reference_sha256"] == sha256_file(asset_dir / "reference.png")
    assert snapshot["direct_attempt_ready_sha256"] == sha256_file(
        asset_dir / "direct_attempt_ready.json"
    )
    assert snapshot["pixel_qa_sha256"] == sha256_file(asset_dir / "pixel_qa.json")
    assert snapshot["bound_blend_sha256"] == sha256_file(asset_dir / "bound.blend")
    assert snapshot["bind_metrics_sha256"] == sha256_file(
        asset_dir / "bind_metrics.json"
    )
    assert snapshot["bind_contact_sheet_sha256"] == sha256_file(
        asset_dir / "bind_contact_sheet.png"
    )
    assert snapshot["walk_glb_sha256"] == sha256_file(asset_dir / "bound_walk.glb")
    assert snapshot["idle_glb_sha256"] == sha256_file(asset_dir / "bound_idle.glb")
    for motion in REQUIRED_MOTIONS:
        for view in REQUIRED_VIEWS:
            assert captured[f"{motion}_{view}"] == (
                asset_dir / _media_name(motion, view)
            ).read_bytes()
            assert snapshot[f"{motion}_{view}_sha256"] == sha256_file(
                asset_dir / _media_name(motion, view)
            )


def test_captured_bytes_survive_paths_becoming_symlinks(workspace, tmp_path):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    _, _, captured, snapshot = validated_review_snapshot(asset_dir)
    reference_a = captured["reference"]
    video_a = captured["walk_front"]
    external_reference = tmp_path / "external-reference.png"
    external_video = tmp_path / "external-walk-front.mp4"
    external_reference.write_bytes(b"secret reference B")
    external_video.write_bytes(b"secret video B")
    (asset_dir / "reference.png").unlink()
    (asset_dir / "reference.png").symlink_to(external_reference)
    (asset_dir / "walk_front.mp4").unlink()
    (asset_dir / "walk_front.mp4").symlink_to(external_video)

    assert captured["reference"] == reference_a
    assert captured["walk_front"] == video_a
    assert hashlib.sha256(captured["reference"]).hexdigest() == snapshot[
        "reference_sha256"
    ]
    assert hashlib.sha256(captured["walk_front"]).hexdigest() == snapshot[
        "walk_front_sha256"
    ]


@pytest.mark.parametrize(
    "path",
    (
        "bound_walk.glb",
        "bound_idle.glb",
        "bound.blend",
        "walk_front.mp4",
        "idle_feet.mp4",
        "bind_manifest.json",
        "review_manifest.json",
        "direct_attempt_ready.json",
        "pixel_qa.json",
        "bind_metrics.json",
        "bind_contact_sheet.png",
    ),
)
def test_contract_rejects_symlinked_files(workspace, tmp_path, path):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    external = tmp_path / path.replace("/", "_")
    external.write_bytes((asset_dir / path).read_bytes())
    (asset_dir / path).unlink()
    (asset_dir / path).symlink_to(external)

    with pytest.raises(ValueError, match="regular file|symlink|asset root"):
        validated_review_snapshot(asset_dir)


def test_trusted_reads_use_directory_fd_nofollow_and_reject_swap_to_symlink(
    workspace, tmp_path, monkeypatch
):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    target = asset_dir / "walk_front.mp4"
    external = tmp_path / "external-walk-front.mp4"
    external.write_bytes(target.read_bytes())
    real_open = review_contract.os.open
    swapped = False

    def swap_before_fd_open(path, flags, *args, **kwargs):
        nonlocal swapped
        if path == "walk_front.mp4" and kwargs.get("dir_fd") is not None and not swapped:
            swapped = True
            target.unlink()
            target.symlink_to(external)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(review_contract.os, "open", swap_before_fd_open)

    with pytest.raises(ValueError, match="regular|symlink|nofollow"):
        validated_review_snapshot(asset_dir)
    assert swapped


def test_contract_source_uses_fd_relative_nofollow_reads_only():
    source = Path(review_contract.__file__).read_text(encoding="utf-8")

    assert "os.O_NOFOLLOW" in source
    assert "dir_fd=" in source
    assert "os.fstat(" in source
    assert "stat.S_ISREG" in source
    assert ".read_bytes()" not in source


def test_contract_rejects_noncanonical_manifest_paths(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    manifest_path = asset_dir / "review_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["videos"]["walk"]["front"]["filename"] = "../walk_front.mp4"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical media filename"):
        validated_review_snapshot(asset_dir)


def test_contract_requires_a_current_ready_record(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    (asset_dir / "direct_attempt_ready.json").unlink()

    with pytest.raises(ValueError, match="direct attempt ready.*missing"):
        validated_review_snapshot(asset_dir)


@pytest.mark.parametrize(
    ("filename", "description"),
    (
        ("bind_metrics.json", "bind metrics"),
        ("bind_contact_sheet.png", "contact sheet"),
    ),
)
def test_contract_requires_each_pixel_qa_auxiliary_artifact(
    workspace, filename, description
):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    (asset_dir / filename).unlink()

    with pytest.raises(ValueError, match=f"{description}.*missing"):
        validated_review_snapshot(asset_dir)


def test_ready_record_requires_exact_pixel_qa_descriptor(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    ready_path = asset_dir / "direct_attempt_ready.json"
    ready = json.loads(ready_path.read_text(encoding="utf-8"))

    assert ready["pixel_qa"] == _descriptor(asset_dir / "pixel_qa.json")
    ready["pixel_qa"]["source"] = "must-not-be-embedded"
    ready_path.write_text(json.dumps(ready, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="pixel QA.*exactly filename and sha256"):
        validated_review_snapshot(asset_dir)


@pytest.mark.parametrize("field", ("bind_metrics", "contact_sheet"))
def test_ready_record_requires_each_auxiliary_descriptor(workspace, field):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    ready_path = asset_dir / "direct_attempt_ready.json"
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    ready.pop(field)
    ready_path.write_text(json.dumps(ready, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="ready record.*missing required fields"):
        validated_review_snapshot(asset_dir)


@pytest.mark.parametrize("field", ("bind_metrics", "contact_sheet"))
def test_ready_auxiliary_descriptors_have_exact_shape(workspace, field):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    ready_path = asset_dir / "direct_attempt_ready.json"
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    ready[field]["unexpected"] = "not allowed"
    ready_path.write_text(json.dumps(ready, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly filename and sha256"):
        validated_review_snapshot(asset_dir)


@pytest.mark.parametrize("failure", ("missing", "stale_hash"))
def test_ready_record_requires_current_pixel_qa_file(workspace, failure):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    if failure == "missing":
        (asset_dir / "pixel_qa.json").unlink()
    else:
        qa_path = asset_dir / "pixel_qa.json"
        qa_path.write_bytes(qa_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="pixel QA.*missing|pixel QA.*hash"):
        validated_review_snapshot(asset_dir)


def test_old_pixel_qa_cannot_validate_rerendered_snapshot_b(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    qa_a = (asset_dir / "pixel_qa.json").read_bytes()
    video_path = asset_dir / "walk_front.mp4"
    video_path.write_bytes(b"rerendered snapshot B")
    manifest_path = asset_dir / "review_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["videos"]["walk"]["front"]["sha256"] = sha256_file(video_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    _write_ready_record(asset_dir, refresh_pixel_qa=False)

    assert (asset_dir / "pixel_qa.json").read_bytes() == qa_a
    with pytest.raises(ValueError, match="pixel QA expected artifact snapshot is stale"):
        validated_review_snapshot(asset_dir)


@pytest.mark.parametrize("field", ("bind_metrics", "contact_sheet"))
def test_pixel_qa_expected_snapshot_cannot_drop_auxiliary_fields(workspace, field):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    pixel_qa_path = asset_dir / "pixel_qa.json"
    pixel_qa = json.loads(pixel_qa_path.read_text(encoding="utf-8"))
    pixel_qa["expected_artifact_snapshot"].pop(field)
    pixel_qa_path.write_text(json.dumps(pixel_qa, sort_keys=True), encoding="utf-8")
    _write_ready_record(asset_dir, refresh_pixel_qa=False)

    with pytest.raises(ValueError, match="expected artifact snapshot.*complete"):
        validated_review_snapshot(asset_dir)


def test_contract_fails_closed_when_a_rejected_record_exists(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    (asset_dir / "direct_attempt_rejected.json").write_text(
        json.dumps({"status": "rejected"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="direct attempt is rejected"):
        validated_review_snapshot(asset_dir)


def test_contract_rejects_a_symlinked_rejected_record(workspace, tmp_path):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    external = tmp_path / "rejected.json"
    external.write_text(json.dumps({"status": "rejected"}), encoding="utf-8")
    (asset_dir / "direct_attempt_rejected.json").symlink_to(external)

    with pytest.raises(ValueError, match="direct attempt is rejected"):
        validated_review_snapshot(asset_dir)


@pytest.mark.parametrize(
    "field_path",
    (
        ("bind_manifest_sha256",),
        ("review_manifest_sha256",),
        ("bound_blend", "sha256"),
        ("glbs", "walk", "sha256"),
        ("videos", "idle", "feet", "sha256"),
        ("bind_metrics", "sha256"),
        ("contact_sheet", "sha256"),
    ),
)
def test_contract_rejects_each_stale_ready_hash(workspace, field_path):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    ready_path = asset_dir / "direct_attempt_ready.json"
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    target = ready
    for field in field_path[:-1]:
        target = target[field]
    target[field_path[-1]] = "0" * 64
    ready_path.write_text(json.dumps(ready, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="ready.*hash|hash.*ready"):
        validated_review_snapshot(asset_dir)


def test_contract_rejects_ready_descriptor_path_escape(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    ready_path = asset_dir / "direct_attempt_ready.json"
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    ready["videos"]["walk"]["front"]["filename"] = "../walk_front.mp4"
    ready_path.write_text(json.dumps(ready, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical media filename"):
        validated_review_snapshot(asset_dir)


def test_contract_rejects_stale_bind_manifest_hash(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    bind_path = asset_dir / "bind_manifest.json"
    manifest = json.loads(bind_path.read_text(encoding="utf-8"))
    manifest["reference"]["sha256"] = "0" * 64
    bind_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="reference.*hash"):
        validated_review_snapshot(asset_dir)


def test_contract_retries_to_a_stable_multifile_snapshot(workspace, monkeypatch):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    original = review_contract._read_review_snapshot_once
    calls = 0

    def regenerate_after_first_read(path):
        nonlocal calls
        result = original(path)
        calls += 1
        if calls == 1:
            _replace_video_fixture(asset_dir, "walk", "front", b"walk front B")
        return result

    monkeypatch.setattr(review_contract, "_read_review_snapshot_once", regenerate_after_first_read)

    _, _, _, snapshot = validated_review_snapshot(asset_dir)

    assert snapshot["walk_front_sha256"] == sha256_file(asset_dir / "walk_front.mp4")


def test_record_decision_pins_the_expected_full_snapshot(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    snapshot = _snapshot(asset_dir)

    decision = record_decision(
        asset_dir, "approved", "  reviewer-a ", "  ready ", expected_snapshot=snapshot
    )

    assert decision["schema_version"] == "hy3d_rocketbox_review_v1"
    assert decision["decision"] == "approved"
    assert decision["reviewer"] == "reviewer-a"
    assert decision["notes"] == "ready"
    assert decision["snapshot"] == snapshot
    assert decision["snapshot"]["direct_attempt_ready_sha256"] == sha256_file(
        asset_dir / "direct_attempt_ready.json"
    )
    assert decision["snapshot"]["pixel_qa_sha256"] == sha256_file(
        asset_dir / "pixel_qa.json"
    )
    assert decision["snapshot"]["bind_metrics_sha256"] == sha256_file(
        asset_dir / "bind_metrics.json"
    )
    assert decision["snapshot"]["bind_contact_sheet_sha256"] == sha256_file(
        asset_dir / "bind_contact_sheet.png"
    )
    assert not (asset_dir / "hy3d_rocketbox_review.json.tmp").exists()
    assert_asset_approved(asset_dir)


def test_a_changed_video_invalidates_an_approval(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    record_decision(asset_dir, "approved", "reviewer", "ready", expected_snapshot=_snapshot(asset_dir))
    _replace_video_fixture(asset_dir, "idle", "side", b"rerendered")

    with pytest.raises(Hy3DRocketboxNotApproved, match="snapshot is stale"):
        assert_asset_approved(asset_dir)


@pytest.mark.parametrize(
    ("filename", "payload"),
    (
        ("bind_metrics.json", b'{"regenerated":true}'),
        ("bind_contact_sheet.png", b"regenerated contact sheet"),
    ),
)
def test_changed_pixel_qa_auxiliary_artifact_invalidates_approval(
    workspace, filename, payload
):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    record_decision(
        asset_dir,
        "approved",
        "reviewer",
        "ready",
        expected_snapshot=_snapshot(asset_dir),
    )
    (asset_dir / filename).write_bytes(payload)

    with pytest.raises(Hy3DRocketboxNotApproved, match="hash|stale"):
        assert_asset_approved(asset_dir)


def test_stale_expected_snapshot_cannot_overwrite_current_decision(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    old_snapshot = _snapshot(asset_dir)
    record_decision(asset_dir, "rejected", "first", "A", expected_snapshot=old_snapshot)
    review_path = asset_dir / "hy3d_rocketbox_review.json"
    before = review_path.read_bytes()
    (asset_dir / "bound_walk.glb").write_bytes(b"walk glb B")
    bind_path = asset_dir / "bind_manifest.json"
    bind = json.loads(bind_path.read_text(encoding="utf-8"))
    bind["glbs"]["walk"]["sha256"] = sha256_file(asset_dir / "bound_walk.glb")
    bind_path.write_text(json.dumps(bind, sort_keys=True), encoding="utf-8")
    review_manifest_path = asset_dir / "review_manifest.json"
    review_manifest = json.loads(review_manifest_path.read_text(encoding="utf-8"))
    review_manifest["bind_manifest_sha256"] = sha256_file(bind_path)
    review_manifest["glbs"]["walk"]["sha256"] = sha256_file(
        asset_dir / "bound_walk.glb"
    )
    review_manifest_path.write_text(json.dumps(review_manifest, sort_keys=True), encoding="utf-8")
    _write_ready_record(asset_dir)

    with pytest.raises(ValueError, match="snapshot changed"):
        record_decision(asset_dir, "approved", "second", "B", expected_snapshot=old_snapshot)
    assert review_path.read_bytes() == before


def test_record_decision_restores_previous_record_if_snapshot_changes_after_write(
    workspace, monkeypatch
):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    snapshot = _snapshot(asset_dir)
    record_decision(
        asset_dir,
        "approved",
        "first-reviewer",
        "previous",
        expected_snapshot=snapshot,
    )
    decision_path = asset_dir / "hy3d_rocketbox_review.json"
    previous = decision_path.read_bytes()
    real_write = review_contract._atomic_write_json
    writes = 0

    def write_then_regenerate(path, payload):
        nonlocal writes
        real_write(path, payload)
        writes += 1
        if writes == 1:
            _replace_video_fixture(asset_dir, "walk", "front", b"post-write B")

    monkeypatch.setattr(review_contract, "_atomic_write_json", write_then_regenerate)

    with pytest.raises(ValueError, match="snapshot changed"):
        record_decision(
            asset_dir,
            "rejected",
            "second-reviewer",
            "must roll back",
            expected_snapshot=snapshot,
        )

    assert decision_path.read_bytes() == previous


def test_read_only_state_derives_pending_without_writing(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]

    state = read_review_state(asset_dir)

    assert state["decision"] == "pending"
    assert not (asset_dir / "hy3d_rocketbox_review.json").exists()


def test_review_state_for_snapshot_does_not_recapture_changed_artifacts(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    snapshot_a = _snapshot(asset_dir)
    record_decision(
        asset_dir,
        "approved",
        "reviewer",
        "snapshot A",
        expected_snapshot=snapshot_a,
    )
    bind_manifest, _, _, captured_snapshot = validated_review_snapshot(asset_dir)
    _replace_video_fixture(asset_dir, "walk", "front", b"snapshot B video")

    state = review_contract.read_review_state_for_snapshot(
        asset_dir, bind_manifest, captured_snapshot
    )

    assert state["decision"] == "approved"
    assert state["snapshot"] == snapshot_a


def test_pair_gate_requires_both_current_approvals(workspace):
    for asset_id in EXPECTED_ASSET_IDS:
        asset_dir = workspace / asset_id
        record_decision(asset_dir, "approved", "reviewer", "ready", expected_snapshot=_snapshot(asset_dir))

    approvals = assert_pair_approved(workspace)

    assert set(approvals) == set(EXPECTED_ASSET_IDS)


def test_pair_gate_rejects_female_capture_in_male_directory_slot(workspace):
    for asset_id in EXPECTED_ASSET_IDS:
        asset_dir = workspace / asset_id
        record_decision(
            asset_dir,
            "approved",
            "reviewer",
            "ready",
            expected_snapshot=_snapshot(asset_dir),
        )
    male_dir = workspace / EXPECTED_ASSET_IDS[0]
    female_dir = workspace / EXPECTED_ASSET_IDS[1]
    shutil.rmtree(male_dir)
    shutil.copytree(female_dir, male_dir)

    with pytest.raises(Hy3DRocketboxNotApproved, match="directory slot.*asset_id"):
        assert_pair_approved(workspace)


def test_pair_gate_recaptures_both_assets_after_male_changes_during_female_read(
    workspace, monkeypatch
):
    for asset_id in EXPECTED_ASSET_IDS:
        asset_dir = workspace / asset_id
        record_decision(
            asset_dir,
            "approved",
            "reviewer",
            "ready",
            expected_snapshot=_snapshot(asset_dir),
        )
    male_dir = workspace / EXPECTED_ASSET_IDS[0]
    female_dir = workspace / EXPECTED_ASSET_IDS[1]
    original = review_contract._read_review_snapshot_once
    changed = False

    def change_male_after_it_was_captured(path):
        nonlocal changed
        if Path(path) == female_dir and not changed:
            changed = True
            _replace_video_fixture(male_dir, "walk", "side", b"male snapshot B")
        return original(path)

    monkeypatch.setattr(
        review_contract, "_read_review_snapshot_once", change_male_after_it_was_captured
    )

    with pytest.raises(Hy3DRocketboxNotApproved, match="pair.*changed|snapshot.*stale"):
        assert_pair_approved(workspace)
    assert changed


def test_pair_gate_rejects_a_symlinked_asset_directory(workspace, tmp_path):
    asset_id = EXPECTED_ASSET_IDS[0]
    original = workspace / asset_id
    external = tmp_path / asset_id
    original.rename(external)
    original.symlink_to(external, target_is_directory=True)

    with pytest.raises(Hy3DRocketboxNotApproved, match="must not be a symlink"):
        assert_pair_approved(workspace)


def test_gate_rejects_an_approval_after_direct_attempt_is_rejected(workspace):
    asset_dir = workspace / EXPECTED_ASSET_IDS[0]
    record_decision(
        asset_dir,
        "approved",
        "reviewer",
        "ready",
        expected_snapshot=_snapshot(asset_dir),
    )
    (asset_dir / "direct_attempt_rejected.json").write_text(
        json.dumps({"status": "rejected"}), encoding="utf-8"
    )

    with pytest.raises(Hy3DRocketboxNotApproved, match="rejected"):
        assert_asset_approved(asset_dir)
