from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SERVER_PATH = (
    Path(__file__).resolve().parents[3]
    / "tools"
    / "spike_rlr"
    / "second_retarget_facing_review_server.py"
)


def _server():
    spec = importlib.util.spec_from_file_location("second_facing_server", SERVER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _local(path: Path) -> dict:
    return {"filename": path.name, "sha256": _sha(path), "size_bytes": path.stat().st_size}


def _external(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": _sha(path), "size_bytes": path.stat().st_size}


@pytest.fixture()
def site(tmp_path: Path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    source_media = {}
    for view in ("front", "side", "feet"):
        path = source / f"walking_{view}.mp4"
        path.write_bytes((view + " video bytes").encode())
        source_media[view] = {"mp4": _external(path)}
    bundle = tmp_path / "second_attempt_facing_review_v1"
    bundle.mkdir()
    html = bundle / "review.html"
    html.write_text("<!doctype html><title>facing review</title>", encoding="utf-8")
    top = bundle / "top_facing.mp4"
    top.write_bytes(b"top video bytes")
    metrics = bundle / "facing_metrics.json"
    metrics.write_text('{"frame_count":33}', encoding="utf-8")
    manifest = {
        "schema": "second_retarget_facing_review_render_v1",
        "asset_id": "rocketbox_male_adult_01",
        "classification": "technical_diagnostic_only",
        "decision": "rejected",
        "formal_dataset_asset": False,
        "source": {"media": source_media},
        "derived_artifacts": {
            "review.html": _local(html),
            "top_facing.mp4": _local(top),
            "facing_metrics.json": _local(metrics),
        },
    }
    manifest_path = bundle / "facing_review_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    server = _server()
    monkeypatch.setattr(server, "validate_facing_bundle", lambda root: manifest)
    app = server.create_app(bundle)
    app.config.update(TESTING=True)
    return server, app, bundle, source, manifest


def test_index_and_four_media_views_are_read_only_no_store(site):
    _, app, _, _, _ = site
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
    assert response.mimetype == "text/html"
    assert response.headers["Cache-Control"] == "no-store, max-age=0"
    for view in ("front", "side", "feet", "top"):
        response = client.get(f"/media/{view}")
        assert response.status_code == 200
        assert response.mimetype == "video/mp4"
        assert response.headers["Cache-Control"] == "no-store, max-age=0"
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.mimetype == "application/json"


@pytest.mark.parametrize("method", ("post", "put", "delete"))
def test_mutating_methods_have_no_route(site, method):
    _, app, _, _, _ = site
    response = getattr(app.test_client(), method)("/")
    assert response.status_code == 405


def test_range_request_is_supported_for_video(site):
    _, app, _, _, _ = site
    response = app.test_client().get("/media/front", headers={"Range": "bytes=0-4"})
    assert response.status_code == 206
    assert response.data == b"front"


def test_unknown_view_and_path_traversal_are_rejected(site):
    _, app, _, _, _ = site
    client = app.test_client()
    assert client.get("/media/unknown").status_code == 404
    assert client.get("/media/../front").status_code == 404


def test_requested_source_tamper_is_refused(site):
    _, app, _, source, _ = site
    (source / "walking_side.mp4").write_bytes(b"tampered")
    response = app.test_client().get("/media/side")
    assert response.status_code == 409
    assert b"changed" in response.data


def test_manifest_change_after_startup_is_refused(site):
    _, app, bundle, _, _ = site
    (bundle / "facing_review_manifest.json").write_text("{}", encoding="utf-8")
    response = app.test_client().get("/")
    assert response.status_code == 409
    assert b"manifest changed" in response.data
