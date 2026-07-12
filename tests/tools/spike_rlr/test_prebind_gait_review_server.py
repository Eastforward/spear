from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SERVER = Path(__file__).resolve().parents[3] / "tools" / "spike_rlr" / "prebind_gait_review_server.py"


def _load():
    spec = importlib.util.spec_from_file_location("prebind_gait_server", SERVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(path: Path, *, local=False):
    return {
        "filename" if local else "path": path.name if local else str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }


@pytest.fixture()
def site(tmp_path: Path, monkeypatch):
    source_dir = tmp_path / "source"; source_dir.mkdir()
    target_dir = tmp_path / "target"; target_dir.mkdir()
    source = {}
    target = {}
    for view in ("front", "side", "top", "source_target"):
        path = source_dir / f"{view}.mp4"; path.write_bytes(("source-" + view).encode())
        source[view] = _record(path)
    for view in ("front", "side", "feet", "top"):
        path = target_dir / f"{view}.mp4"; path.write_bytes(("target-" + view).encode())
        target[view] = _record(path)
    bundle = tmp_path / "bundle"; bundle.mkdir()
    html = bundle / "review.html"; html.write_text("<!doctype html>review", encoding="utf-8")
    metrics = bundle / "prebind_gait_metrics.json"; metrics.write_text("{}", encoding="utf-8")
    manifest = {
        "schema": "prebind_vs_second_retarget_gait_review_v1",
        "source_prebind": {"media": source},
        "target_second_retarget": {"media": target},
        "local_artifacts": {"review.html": _record(html, local=True), "prebind_gait_metrics.json": _record(metrics, local=True)},
    }
    manifest_path = bundle / "prebind_gait_review_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    module = _load()
    monkeypatch.setattr(module, "validate_bundle", lambda root: manifest)
    app = module.create_app(bundle); app.config.update(TESTING=True)
    return app, bundle, source_dir


def test_serves_eight_videos_html_and_metrics_read_only(site):
    app, _, _ = site; client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/metrics").status_code == 200
    for stage, views in (("source", ("front", "side", "top", "source_target")), ("target", ("front", "side", "feet", "top"))):
        for view in views:
            response = client.get(f"/{stage}/{view}")
            assert response.status_code == 200
            assert response.headers["Cache-Control"] == "no-store, max-age=0"
    assert client.post("/").status_code == 405


def test_refuses_unknown_routes_and_source_tamper(site):
    app, _, source_dir = site; client = app.test_client()
    assert client.get("/source/feet").status_code == 404
    (source_dir / "front.mp4").write_bytes(b"tampered")
    assert client.get("/source/front").status_code == 409


def test_refuses_manifest_change_after_startup(site):
    app, bundle, _ = site
    (bundle / "prebind_gait_review_manifest.json").write_text("{}", encoding="utf-8")
    response = app.test_client().get("/")
    assert response.status_code == 409
    assert b"manifest changed" in response.data
