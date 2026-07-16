"""Flask tests for the Hunyuan/Rocketbox Walk/Idle review UI."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import re
import shutil
import sys
from types import MappingProxyType
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
    record_decision,
    sha256_file,
    validated_review_snapshot,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    bind = json.loads(
        (asset_dir / "bind_manifest.json").read_text(encoding="utf-8")
    )
    return {
        "schema_version": "hy3d_rocketbox_artifact_snapshot_v1",
        "asset_id": bind["asset_id"],
        "bind_manifest_sha256": sha256_file(asset_dir / "bind_manifest.json"),
        "review_manifest_sha256": sha256_file(asset_dir / "review_manifest.json"),
        "bound_blend": _descriptor(asset_dir / "bound.blend"),
        "glbs": {
            motion: _descriptor(asset_dir / _glb_name(motion))
            for motion in REQUIRED_MOTIONS
        },
        "videos": {
            motion: {
                view: _descriptor(asset_dir / f"{motion}_{view}.mp4")
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
    bind = json.loads(
        (asset_dir / "bind_manifest.json").read_text(encoding="utf-8")
    )
    if refresh_pixel_qa:
        _write_pixel_qa(asset_dir)
    pixel_qa_path = asset_dir / "pixel_qa.json"
    pixel_qa = json.loads(pixel_qa_path.read_text(encoding="utf-8"))
    ready = _task4_ready_builder()(
        bind["asset_id"],
        "ready",
        pixel_qa,
        sha256_file(pixel_qa_path),
        pixel_qa_path,
        _artifact_snapshot(asset_dir),
    )
    (asset_dir / "direct_attempt_ready.json").write_text(
        json.dumps(ready, sort_keys=True), encoding="utf-8"
    )


def _write_ready_fixture(root: Path, asset_id: str) -> Path:
    asset_dir = root / asset_id
    asset_dir.mkdir(parents=True)
    reference = b"approved reference"
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
        data = f"{asset_id}:{motion}:glb".encode("ascii")
        filename = _glb_name(motion)
        (asset_dir / filename).write_bytes(data)
        glbs[motion] = {"filename": filename, "sha256": _sha256(data)}
    bind = _task3_manifest_builder()(
        SimpleNamespace(asset_id=asset_id),
        asset_dir,
        {
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
    (asset_dir / "bind_manifest.json").write_text(json.dumps(bind, sort_keys=True), encoding="utf-8")
    videos = {}
    for motion in REQUIRED_MOTIONS:
        videos[motion] = {}
        for view in REQUIRED_VIEWS:
            name = f"{motion}_{view}.mp4"
            data = f"{asset_id}:{motion}:{view}:video".encode("ascii")
            (asset_dir / name).write_bytes(data)
            videos[motion][view] = {"filename": name, "sha256": _sha256(data)}
    review = {
        "schema_version": "hy3d_rocketbox_review_manifest_v1",
        "asset_id": asset_id,
        "bind_manifest_sha256": sha256_file(asset_dir / "bind_manifest.json"),
        "glbs": glbs,
        "videos": videos,
    }
    (asset_dir / "review_manifest.json").write_text(json.dumps(review, sort_keys=True), encoding="utf-8")
    _write_ready_record(asset_dir)
    return asset_dir


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "reviews"
    for asset_id in EXPECTED_ASSET_IDS:
        _write_ready_fixture(root, asset_id)
    return root


def _client(workspace: Path):
    from hy3d_rocketbox_review_server import create_app

    return create_app(workspace).test_client()


def _csrf_token(client, asset_id: str) -> str:
    response = client.get(f"/asset/{asset_id}")
    assert response.status_code == 200
    match = re.search(rb'name="csrf_token" value="([^"]+)"', response.data)
    assert match is not None
    return match.group(1).decode("ascii")


def _snapshot_fields(workspace: Path, asset_id: str) -> dict[str, str]:
    asset_dir = workspace / asset_id
    result = {
        "bind_manifest_sha256": sha256_file(asset_dir / "bind_manifest.json"),
        "review_manifest_sha256": sha256_file(asset_dir / "review_manifest.json"),
        "direct_attempt_ready_sha256": sha256_file(
            asset_dir / "direct_attempt_ready.json"
        ),
        "pixel_qa_sha256": sha256_file(asset_dir / "pixel_qa.json"),
        "reference_sha256": sha256_file(asset_dir / "reference.png"),
        "bound_blend_sha256": sha256_file(asset_dir / "bound.blend"),
        "bind_metrics_sha256": sha256_file(asset_dir / "bind_metrics.json"),
        "bind_contact_sheet_sha256": sha256_file(
            asset_dir / "bind_contact_sheet.png"
        ),
    }
    for motion in REQUIRED_MOTIONS:
        result[f"{motion}_glb_sha256"] = sha256_file(
            asset_dir / _glb_name(motion)
        )
        for view in REQUIRED_VIEWS:
            result[f"{motion}_{view}_sha256"] = sha256_file(asset_dir / f"{motion}_{view}.mp4")
    return result


def _form_data(client, workspace: Path, asset_id: str, decision: str) -> dict[str, str]:
    return {
        "decision": decision,
        "reviewer": "reviewer",
        "notes": "review notes",
        "csrf_token": _csrf_token(client, asset_id),
        **_snapshot_fields(workspace, asset_id),
    }


def _media_url(workspace: Path, asset_id: str, motion: str, view: str) -> str:
    expected = _snapshot_fields(workspace, asset_id)[f"{motion}_{view}_sha256"]
    return f"/media/{asset_id}/{motion}/{view}?expected_sha256={expected}"


def _decision_records(workspace: Path) -> dict[str, bytes | None]:
    return {
        asset_id: ((workspace / asset_id / "hy3d_rocketbox_review.json").read_bytes() if (workspace / asset_id / "hy3d_rocketbox_review.json").exists() else None)
        for asset_id in EXPECTED_ASSET_IDS
    }


def _replace_video_fixture(
    asset_dir: Path, motion: str, view: str, payload: bytes
) -> None:
    video_path = asset_dir / f"{motion}_{view}.mp4"
    video_path.write_bytes(payload)
    manifest_path = asset_dir / "review_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["videos"][motion][view]["sha256"] = sha256_file(video_path)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True), encoding="utf-8"
    )
    _write_ready_record(asset_dir)


@pytest.mark.parametrize("route", ("/asset/rocketbox_male_adult_01", "/gate"))
def test_get_routes_are_read_only(workspace, route):
    before = _decision_records(workspace)

    response = _client(workspace).get(route)

    assert response.status_code == 200
    assert _decision_records(workspace) == before


def test_asset_page_has_identity_cue_motion_tabs_and_all_video_columns(workspace):
    response = _client(workspace).get("/asset/rocketbox_male_adult_01")

    assert response.status_code == 200
    for label in (b"Male", b"Female", b"Walk", b"Idle", b"Front", b"Side", b"Feet"):
        assert label in response.data
    assert b"Approved FLUX reference" in response.data
    assert b"bind_manifest.json" not in response.data
    assert b"review_manifest.json" not in response.data
    assert response.data.count(b"<video") == 3
    assert b"Approve" in response.data
    assert b"Reject" in response.data
    assert b'name="csrf_token"' in response.data
    for field, value in _snapshot_fields(workspace, "rocketbox_male_adult_01").items():
        assert f'name="{field}" value="{value}"'.encode("ascii") in response.data


def test_asset_page_uses_versioned_urls_for_every_video(workspace):
    response = _client(workspace).get("/asset/rocketbox_male_adult_01")

    assert response.status_code == 200
    for motion in REQUIRED_MOTIONS:
        for view in REQUIRED_VIEWS:
            assert _media_url(workspace, "rocketbox_male_adult_01", motion, view).encode("ascii") in response.data


def test_asset_page_fails_closed_when_ready_snapshot_changes_after_capture(
    workspace, monkeypatch
):
    import hy3d_rocketbox_review_server as review_server

    asset_id = "rocketbox_male_adult_01"
    asset_dir = workspace / asset_id
    snapshot_a = _snapshot_fields(workspace, asset_id)
    record_decision(
        asset_dir,
        "approved",
        "snapshot-a-reviewer",
        "approved A",
        expected_snapshot=snapshot_a,
    )
    original = review_server.validated_review_snapshot
    changed = False

    def capture_a_then_publish_b(path):
        nonlocal changed
        result = original(path)
        if Path(path) == asset_dir and not changed:
            changed = True
            _replace_video_fixture(asset_dir, "walk", "front", b"snapshot B video")
        return result

    monkeypatch.setattr(
        review_server, "validated_review_snapshot", capture_a_then_publish_b
    )

    response = review_server.create_app(workspace).test_client().get(
        f"/asset/{asset_id}"
    )

    assert response.status_code == 409
    assert b"Current decision" not in response.data
    assert snapshot_a["walk_front_sha256"].encode("ascii") not in response.data


def test_identity_uses_captured_reference_after_path_becomes_symlink(
    workspace, tmp_path, monkeypatch
):
    import hy3d_rocketbox_review_server as review_server

    asset_id = "rocketbox_male_adult_01"
    asset_dir = workspace / asset_id
    reference_a = (asset_dir / "reference.png").read_bytes()
    secret_b = b"secret external identity B"
    external = tmp_path / "external-reference.png"
    external.write_bytes(secret_b)
    original = review_server.validated_review_snapshot
    changed = False

    def capture_a_then_swap_path(path):
        nonlocal changed
        result = original(path)
        if Path(path) == asset_dir and not changed:
            changed = True
            (asset_dir / "reference.png").unlink()
            (asset_dir / "reference.png").symlink_to(external)
        return result

    monkeypatch.setattr(
        review_server, "validated_review_snapshot", capture_a_then_swap_path
    )

    response = review_server.create_app(workspace).test_client().get(
        f"/asset/{asset_id}"
    )

    assert response.status_code == 409
    assert base64.b64encode(reference_a) not in response.data
    assert base64.b64encode(secret_b) not in response.data


def test_media_is_allowlisted_versioned_and_range_capable(workspace):
    client = _client(workspace)
    response = client.get(_media_url(workspace, "rocketbox_male_adult_01", "walk", "front"), headers={"Range": "bytes=0-3"})

    assert response.status_code == 206
    assert response.data == b"rock"
    assert response.headers["Cache-Control"] == "no-store, max-age=0"
    assert "bytes" in response.headers["Accept-Ranges"]
    assert client.get("/media/rocketbox_male_adult_01/walk/unknown").status_code == 404
    assert client.get("/media/rocketbox_male_adult_01/unknown/front").status_code == 404
    assert client.get("/media/../walk/front").status_code in {404, 405}


def test_media_range_uses_captured_bytes_after_path_replacement(
    workspace, tmp_path, monkeypatch
):
    import hy3d_rocketbox_review_server as review_server

    asset_id = "rocketbox_male_adult_01"
    asset_dir = workspace / asset_id
    url = _media_url(workspace, asset_id, "walk", "front")
    secret_b = b"LEAKED EXTERNAL VIDEO B"
    external = tmp_path / "external-walk-front.mp4"
    external.write_bytes(secret_b)
    original = review_server.validated_review_snapshot
    changed = False

    def capture_a_then_swap_path(path):
        nonlocal changed
        result = original(path)
        if Path(path) == asset_dir and not changed:
            changed = True
            (asset_dir / "walk_front.mp4").unlink()
            (asset_dir / "walk_front.mp4").symlink_to(external)
        return result

    monkeypatch.setattr(
        review_server, "validated_review_snapshot", capture_a_then_swap_path
    )

    response = review_server.create_app(workspace).test_client().get(
        url, headers={"Range": "bytes=0-3"}
    )

    assert response.status_code == 409
    assert response.data != b"rock"
    assert secret_b[:4] not in response.data


def test_media_rechecks_captured_bytes_against_expected_hash(workspace, monkeypatch):
    import hy3d_rocketbox_review_server as review_server

    asset_id = "rocketbox_male_adult_01"
    url = _media_url(workspace, asset_id, "walk", "front")
    secret_b = b"tampered captured video B"
    original = validated_review_snapshot

    def return_tampered_capture(path):
        bind, review, captured, snapshot = original(path)
        tampered = dict(captured)
        tampered["walk_front"] = secret_b
        return bind, review, MappingProxyType(tampered), snapshot

    monkeypatch.setattr(
        review_server, "validated_review_snapshot", return_tampered_capture
    )

    response = review_server.create_app(workspace).test_client().get(url)

    assert response.status_code == 409
    assert secret_b not in response.data


@pytest.mark.parametrize(
    "readiness_state",
    (
        "missing",
        "rejected",
        "stale",
        "old_pixel_qa",
        "bind_metrics_stale",
        "contact_sheet_missing",
    ),
)
def test_server_refuses_review_media_and_decisions_without_current_ready(
    workspace, readiness_state
):
    asset_id = "rocketbox_male_adult_01"
    asset_dir = workspace / asset_id
    client = _client(workspace)
    media_url = _media_url(workspace, asset_id, "walk", "front")
    form = _form_data(client, workspace, asset_id, "approved")
    if readiness_state == "missing":
        (asset_dir / "direct_attempt_ready.json").unlink()
    elif readiness_state == "rejected":
        (asset_dir / "direct_attempt_rejected.json").write_text(
            json.dumps({"status": "rejected"}), encoding="utf-8"
        )
    elif readiness_state == "stale":
        (asset_dir / "bound.blend").write_bytes(b"regenerated bound blend")
    elif readiness_state == "bind_metrics_stale":
        (asset_dir / "bind_metrics.json").write_bytes(b'{"regenerated":true}')
    elif readiness_state == "contact_sheet_missing":
        (asset_dir / "bind_contact_sheet.png").unlink()
    else:
        video_path = asset_dir / "walk_front.mp4"
        video_path.write_bytes(b"rerendered snapshot B")
        manifest_path = asset_dir / "review_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["videos"]["walk"]["front"]["sha256"] = sha256_file(
            video_path
        )
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        _write_ready_record(asset_dir, refresh_pixel_qa=False)

    page = client.get(f"/asset/{asset_id}")
    media = client.get(media_url)
    decision = client.post(f"/decision/{asset_id}", data=form)
    gate = client.get("/gate")

    assert page.status_code == 409
    assert media.status_code == 409
    assert decision.status_code == 409
    assert gate.status_code == 200
    assert gate.get_json()["state"] == "locked"
    assert not (asset_dir / "hy3d_rocketbox_review.json").exists()


def test_media_rejects_a_stale_page_url_after_a_to_b_regeneration(workspace):
    asset_id = "rocketbox_male_adult_01"
    asset_dir = workspace / asset_id
    old_url = _media_url(workspace, asset_id, "walk", "front")
    (asset_dir / "walk_front.mp4").write_bytes(b"walk front B")
    manifest_path = asset_dir / "review_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["videos"]["walk"]["front"]["sha256"] = sha256_file(asset_dir / "walk_front.mp4")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    assert _client(workspace).get(old_url).status_code == 409


def test_posted_decisions_are_independent_and_redirect_to_the_asset(workspace):
    client = _client(workspace)
    asset_id = "rocketbox_male_adult_01"

    response = client.post(f"/decision/{asset_id}?next=https://example.test/", data=_form_data(client, workspace, asset_id, "approved"), follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["Location"].endswith(f"/asset/{asset_id}")
    male = json.loads((workspace / asset_id / "hy3d_rocketbox_review.json").read_text(encoding="utf-8"))
    assert male["decision"] == "approved"
    assert not (workspace / "rocketbox_female_adult_01" / "hy3d_rocketbox_review.json").exists()


def test_stale_post_does_not_write_or_replace_a_decision(workspace):
    asset_id = "rocketbox_male_adult_01"
    asset_dir = workspace / asset_id
    client = _client(workspace)
    old_form = _form_data(client, workspace, asset_id, "approved")
    (asset_dir / "idle_feet.mp4").write_bytes(b"idle feet B")
    manifest_path = asset_dir / "review_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["videos"]["idle"]["feet"]["sha256"] = sha256_file(asset_dir / "idle_feet.mp4")
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    response = client.post(f"/decision/{asset_id}", data=old_form)

    assert response.status_code == 409
    assert not (asset_dir / "hy3d_rocketbox_review.json").exists()


def test_change_in_the_race_between_server_check_and_record_is_rejected(workspace, monkeypatch):
    import hy3d_rocketbox_review_server as review_server

    asset_id = "rocketbox_male_adult_01"
    asset_dir = workspace / asset_id
    client = _client(workspace)
    form = _form_data(client, workspace, asset_id, "approved")
    real_record = review_server.record_decision

    def regenerate_then_record(*args, **kwargs):
        (asset_dir / "walk_side.mp4").write_bytes(b"walk side B")
        manifest_path = asset_dir / "review_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["videos"]["walk"]["side"]["sha256"] = sha256_file(asset_dir / "walk_side.mp4")
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
        return real_record(*args, **kwargs)

    monkeypatch.setattr(review_server, "record_decision", regenerate_then_record)

    response = client.post(f"/decision/{asset_id}", data=form)

    assert response.status_code == 409
    assert not (asset_dir / "hy3d_rocketbox_review.json").exists()


def test_stale_post_after_atomic_write_removes_the_new_decision(workspace, monkeypatch):
    asset_id = "rocketbox_male_adult_01"
    asset_dir = workspace / asset_id
    client = _client(workspace)
    form = _form_data(client, workspace, asset_id, "approved")
    real_write = review_contract._atomic_write_json
    writes = 0

    def write_then_regenerate(path, payload):
        nonlocal writes
        real_write(path, payload)
        writes += 1
        if writes == 1:
            _replace_video_fixture(asset_dir, "idle", "front", b"post-write B")

    monkeypatch.setattr(review_contract, "_atomic_write_json", write_then_regenerate)

    response = client.post(f"/decision/{asset_id}", data=form)

    assert response.status_code == 409
    assert not (asset_dir / "hy3d_rocketbox_review.json").exists()


def test_invalid_csrf_and_unknown_assets_do_not_write(workspace):
    client = _client(workspace)

    assert client.get("/asset/not-a-rocketbox-asset").status_code == 404
    assert client.post("/decision/rocketbox_male_adult_01", data={"decision": "approved", "reviewer": "cross-site"}).status_code == 400
    assert not (workspace / "rocketbox_male_adult_01" / "hy3d_rocketbox_review.json").exists()


def test_pair_gate_only_approves_after_both_current_decisions(workspace):
    client = _client(workspace)

    assert client.get("/gate").get_json()["state"] == "locked"
    for asset_id in EXPECTED_ASSET_IDS:
        response = client.post(f"/decision/{asset_id}", data=_form_data(client, workspace, asset_id, "approved"))
        assert response.status_code == 302

    assert client.get("/gate").get_json() == {"state": "approved", "reason": ""}


def test_pair_gate_locks_when_female_artifacts_fill_the_male_slot(workspace):
    client = _client(workspace)
    for asset_id in EXPECTED_ASSET_IDS:
        response = client.post(
            f"/decision/{asset_id}",
            data=_form_data(client, workspace, asset_id, "approved"),
        )
        assert response.status_code == 302
    male_dir = workspace / EXPECTED_ASSET_IDS[0]
    female_dir = workspace / EXPECTED_ASSET_IDS[1]
    shutil.rmtree(male_dir)
    shutil.copytree(female_dir, male_dir)

    response = client.get("/gate")

    assert response.get_json()["state"] == "locked"
    assert "directory slot" in response.get_json()["reason"]
