from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from urllib.parse import quote

import pytest


TOOLS = Path(__file__).resolve().parents[3] / "tools" / "spike_rlr"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import tokenrig_human_review as contract

SERVER_PATH = TOOLS / "tokenrig_human_review_server.py"
SPEC = importlib.util.spec_from_file_location("tokenrig_human_review_server_under_test", SERVER_PATH)
assert SPEC and SPEC.loader
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _external(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": _sha(path), "size_bytes": path.stat().st_size}


def _local(path: Path) -> dict:
    return {"filename": path.name, "sha256": _sha(path), "size_bytes": path.stat().st_size}


def _make_site(tmp_path: Path) -> tuple[Path, Path]:
    asset = tmp_path / "person_01"
    static = asset / "static_audit_v1"
    dynamic = asset / "dynamic_review_v1"
    retarget = asset / "retarget_v1"
    static.mkdir(parents=True)
    dynamic.mkdir()
    retarget.mkdir()
    static_qa = static / "static_qa.json"
    static_qa.write_text("{}", encoding="utf-8")
    bind = static / "bind_pose.glb"
    bind.write_bytes(b"bind")
    evidence = {}
    for filename in contract.STATIC_EVIDENCE:
        path = static / filename
        path.write_bytes((filename + " data").encode())
        evidence[filename] = _external(path)
    retarget_manifest = retarget / "retarget_manifest.json"
    retarget_manifest.write_text("{}", encoding="utf-8")
    retarget_metrics = retarget / "retarget_metrics.json"
    retarget_metrics.write_text("{}", encoding="utf-8")
    glbs = {}
    for motion, filename in (("walking", "walking.glb"), ("standing_idle", "standing_idle.glb")):
        path = retarget / filename
        path.write_bytes(filename.encode())
        glbs[motion] = _external(path)
    execution = asset / "execution"
    execution.mkdir()
    renderer_path = execution / "blender_render_tokenrig_human_review.py"
    ffmpeg_path = execution / "ffmpeg"
    ffprobe_path = execution / "ffprobe"
    renderer_path.write_bytes(b"renderer code")
    ffmpeg_path.write_bytes(b"ffmpeg binary")
    ffprobe_path.write_bytes(b"ffprobe binary")
    actions = {}
    for motion, action_name in contract.MOTIONS.items():
        views = {}
        for view in contract.VIEWS:
            paths = {}
            for kind in ("png", "mp4"):
                path = dynamic / f"{motion}_{view}.{kind}"
                path.write_bytes((path.name + " media").encode())
                paths[kind] = _local(path)
            views[view] = paths
        actions[motion] = {
            "action_name": action_name,
            "frame_start": 1,
            "frame_end": 31,
            "frame_count": 31,
            "fps": 30,
            "duration_s": 31 / 30,
            "views": views,
        }
    media_qa = dynamic / "media_qa.json"
    media_qa.write_text(
        json.dumps(
            {
                "schema": contract.MEDIA_QA_SCHEMA,
                "asset_id": "person_01",
                "automatic_checks": "passed",
                "actions": {
                    motion: {view: {"png": {}, "mp4": {}} for view in contract.VIEWS}
                    for motion in contract.MOTIONS
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema": contract.REVIEW_SCHEMA,
        "asset_id": "person_01",
        "display_label": "Male route-2 canary",
        "instance_kind": "male_canary",
        "state_classification": "research_candidate",
        "canonical_front": "negative-y",
        "canonical_up": "positive-z",
        "fixed_floor_z_m": 0.0,
        "upstream": {
            "asset_id": "person_01",
            "static_qa": _external(static_qa),
            "bind_pose": _external(bind),
            "static_evidence": evidence,
            "retarget_manifest": _external(retarget_manifest),
            "retarget_metrics": _external(retarget_metrics),
            "glbs": glbs,
        },
        "actions": actions,
        "media_qa": _local(media_qa),
        "execution": {
            "renderer": _external(renderer_path),
            "ffmpeg": _external(ffmpeg_path) | {"version": "ffmpeg version test"},
            "ffprobe": _external(ffprobe_path) | {"version": "ffprobe version test"},
        },
        "automatic_checks": "passed",
        "agent_visual_qa": contract.PENDING_STATUS,
        "user_acceptance": "pending_user_review",
        "environment": {"blender_version": "4.2.1", "fps": 30, "resolution": [1280, 720]},
    }
    (dynamic / "review_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    site = tmp_path / "route2_review_site_v1"
    contract.build_consolidated_bundle([dynamic], site)
    return site, dynamic


@pytest.fixture()
def app_and_site(tmp_path):
    site, dynamic = _make_site(tmp_path)
    app = server.create_app(site)
    app.config.update(TESTING=True)
    return app, site, dynamic


def _catalog(site: Path) -> dict:
    return json.loads((site / "review_catalog.json").read_text())


def test_page_is_read_only_human_media_surface_with_all_views(app_and_site):
    app, _, _ = app_and_site
    response = app.test_client().get("/")
    assert response.status_code == 200
    body = response.data
    for label in (b"Route 2 Acceptance Review", b"Male route-2 canary", b"Walk", b"Idle", b"Front", b"Side", b"Top", b"Feet", b"Skeleton", b"Static bind"):
        assert label in body
    assert b"pending_agent_visual_qa" in body
    assert b"user_approved" not in body
    assert b"Approve" not in body
    assert b".fbx" not in body.lower()
    assert b"review_manifest.json" not in body
    catalog = _catalog(app_and_site[1])
    assert hashlib.sha256(response.data).hexdigest() == catalog["review_html"]["sha256"]
    assert response.headers["Cache-Control"] == "no-store, max-age=0"


def test_page_uses_responsive_grid_and_mobile_viewport(app_and_site):
    app, _, _ = app_and_site
    body = app.test_client().get("/").data
    assert b'name="viewport"' in body
    assert b"@media" in body
    assert b"playsinline" in body
    assert b"preload=\"metadata\"" in body


def test_page_displays_agent_reviewer_and_visual_notes(tmp_path):
    _, dynamic = _make_site(tmp_path)
    contract.record_agent_visual_qa(
        dynamic,
        status=contract.PASS_STATUS,
        reviewer="codex-route2-visual-qa",
        notes="All ten videos inspected; feet and skeleton tracking are reasonable.",
        checks={name: True for name in contract.AGENT_VISUAL_CHECKS},
    )
    site = tmp_path / "route2_review_site_after_agent_qa"
    contract.build_consolidated_bundle([dynamic], site)
    app = server.create_app(site)
    app.config.update(TESTING=True)
    body = app.test_client().get("/").data
    assert b"codex-route2-visual-qa" in body
    assert b"All ten videos inspected" in body


def test_exact_hash_locked_mp4_and_png_are_served_no_store(app_and_site):
    app, site, _ = app_and_site
    entry = _catalog(site)["entries"][0]
    client = app.test_client()
    for kind, mimetype in (("mp4", "video/mp4"), ("png", "image/png")):
        record = entry["media"]["walking"]["front"][kind]
        response = client.get(
            f"/media/person_01/walking/front/{kind}?expected_sha256={record['sha256']}"
        )
        assert response.status_code == 200
        assert response.mimetype == mimetype
        assert response.headers["Cache-Control"] == "no-store, max-age=0"
        assert hashlib.sha256(response.data).hexdigest() == record["sha256"]


def test_media_requests_do_not_rehash_every_other_route2_video(tmp_path, monkeypatch):
    site, _ = _make_site(tmp_path)
    real_validate = server.validate_review_catalog
    calls = []

    def counted_validate(path):
        calls.append(Path(path))
        return real_validate(path)

    monkeypatch.setattr(server, "validate_review_catalog", counted_validate)
    app = server.create_app(site)
    app.config.update(TESTING=True)
    assert len(calls) == 1
    entry = _catalog(site)["entries"][0]
    record = entry["media"]["walking"]["front"]["mp4"]
    response = app.test_client().get(
        f"/media/person_01/walking/front/mp4?expected_sha256={record['sha256']}"
    )
    assert response.status_code == 200
    assert len(calls) == 1


def test_media_rejects_missing_wrong_or_malformed_expected_hash(app_and_site):
    app, site, _ = app_and_site
    expected = _catalog(site)["entries"][0]["media"]["walking"]["front"]["mp4"]["sha256"]
    client = app.test_client()
    assert client.get("/media/person_01/walking/front/mp4").status_code == 400
    assert client.get("/media/person_01/walking/front/mp4?expected_sha256=bad").status_code == 400
    assert client.get("/media/person_01/walking/front/mp4?expected_sha256=" + "0" * 64).status_code == 409
    assert client.get(f"/media/person_01/running/front/mp4?expected_sha256={expected}").status_code == 404
    assert client.get(f"/media/person_01/walking/rear/mp4?expected_sha256={expected}").status_code == 404


def test_static_evidence_and_hierarchy_are_hash_locked(app_and_site):
    app, site, _ = app_and_site
    entry = _catalog(site)["entries"][0]
    client = app.test_client()
    for filename, mimetype in (("bind_front.png", "image/png"), ("joint_hierarchy.txt", "text/plain")):
        record = entry["static_evidence"][filename]
        response = client.get(
            f"/static-evidence/person_01/{quote(filename)}?expected_sha256={record['sha256']}"
        )
        assert response.status_code == 200
        assert response.mimetype == mimetype
        assert response.headers["Cache-Control"] == "no-store, max-age=0"


def test_any_source_tamper_locks_page_and_media(app_and_site):
    app, site, dynamic = app_and_site
    entry = _catalog(site)["entries"][0]
    record = entry["media"]["standing_idle"]["feet"]["mp4"]
    (dynamic / "standing_idle_feet.mp4").write_bytes(b"changed")
    client = app.test_client()
    assert client.get("/").status_code == 409
    assert client.get(
        f"/media/person_01/standing_idle/feet/mp4?expected_sha256={record['sha256']}"
    ).status_code == 409


def test_unknown_assets_and_path_like_static_names_are_not_served(app_and_site):
    app, site, _ = app_and_site
    entry = _catalog(site)["entries"][0]
    expected = entry["static_evidence"]["bind_front.png"]["sha256"]
    client = app.test_client()
    assert client.get(f"/static-evidence/unknown/bind_front.png?expected_sha256={expected}").status_code == 404
    assert client.get(f"/static-evidence/person_01/not_allowed.png?expected_sha256={expected}").status_code == 404


def test_cli_requires_existing_catalog_directory_and_loopback_default():
    args = server.parse_args(["--catalog-dir", "/tmp/route2_review_site_v1"])
    assert args.catalog_dir == Path("/tmp/route2_review_site_v1")
    assert args.host == "127.0.0.1"
    assert args.port == 8097


def test_server_rejects_catalog_path_substitution_before_startup(tmp_path):
    site, _ = _make_site(tmp_path)
    catalog_path = site / "review_catalog.json"
    catalog_path.chmod(0o644)
    catalog = json.loads(catalog_path.read_text())
    original = Path(catalog["entries"][0]["media"]["walking"]["front"]["mp4"]["path"])
    substitute = tmp_path / "outside_same_bytes.mp4"
    substitute.write_bytes(original.read_bytes())
    catalog["entries"][0]["media"]["walking"]["front"]["mp4"]["path"] = str(substitute)
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    with pytest.raises(ValueError, match="entry does not match"):
        server.create_app(site)
