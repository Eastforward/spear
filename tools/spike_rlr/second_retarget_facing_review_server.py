#!/usr/bin/env python3
"""Read-only hash-locked server for rejected retarget attempt-two facing review."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from flask import Flask, abort, send_file

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from second_retarget_facing_review import (
    FacingReviewError,
    authenticate_second_attempt,
    sha256_file,
)


BUNDLE_SCHEMA = "second_retarget_facing_review_render_v1"
VIEWS = ("front", "side", "feet", "top")


def _real_directory(path: Path, description: str) -> Path:
    path = Path(os.path.abspath(os.fspath(path)))
    if path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise FacingReviewError(f"{description} must be a direct real directory")
    return path


def _real_file(path: Path, description: str) -> Path:
    path = Path(os.path.abspath(os.fspath(path)))
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise FacingReviewError(f"{description} must be a direct regular file")
    if path.stat().st_size <= 0:
        raise FacingReviewError(f"{description} is empty")
    return path


def _load_object(path: Path, description: str) -> dict[str, Any]:
    path = _real_file(path, description)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FacingReviewError(f"{description} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise FacingReviewError(f"{description} root must be an object")
    return value


def _check_record(path: Path, record: Any, description: str) -> Path:
    path = _real_file(path, description)
    if not isinstance(record, Mapping):
        raise FacingReviewError(f"{description} record is missing")
    if record.get("size_bytes") != path.stat().st_size:
        raise FacingReviewError(f"{description} size changed")
    if record.get("sha256") != sha256_file(path):
        raise FacingReviewError(f"{description} hash changed")
    return path


def validate_facing_bundle(bundle_dir: Path | str) -> dict[str, Any]:
    root = _real_directory(Path(bundle_dir), "facing review bundle")
    manifest_path = _real_file(root / "facing_review_manifest.json", "facing review manifest")
    manifest = _load_object(manifest_path, "facing review manifest")
    if (
        manifest.get("schema") != BUNDLE_SCHEMA
        or manifest.get("asset_id") != "rocketbox_male_adult_01"
        or manifest.get("classification") != "technical_diagnostic_only"
        or manifest.get("decision") != "rejected"
        or manifest.get("formal_dataset_asset") is not False
        or manifest.get("readiness_bundle_published") is not False
    ):
        raise FacingReviewError("facing review bundle classification is invalid")
    if "user_approved" in json.dumps(manifest, sort_keys=True):
        raise FacingReviewError("facing review bundle may not claim user approval")
    derived = manifest.get("derived_artifacts")
    expected = {
        "top_facing.png",
        "top_facing.mp4",
        "facing_metrics.json",
        "review.html",
    }
    if not isinstance(derived, Mapping) or set(derived) != expected:
        raise FacingReviewError("facing review derived artifact inventory is invalid")
    for filename in expected:
        record = derived[filename]
        if not isinstance(record, Mapping) or record.get("filename") != filename:
            raise FacingReviewError(f"{filename} local record is invalid")
        _check_record(root / filename, record, filename)
    metrics = _load_object(root / "facing_metrics.json", "facing metrics")
    if (
        metrics.get("schema") != "second_retarget_facing_metrics_v1"
        or metrics.get("fps") != 30
        or metrics.get("frame_count") != 33
        or len(metrics.get("frames", [])) != 33
        or metrics.get("summary") != manifest.get("metrics_summary")
    ):
        raise FacingReviewError("facing metrics do not match the bundle manifest")
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise FacingReviewError("facing bundle source snapshot is missing")
    source_manifest = source.get("manifest")
    if not isinstance(source_manifest, Mapping) or not isinstance(source_manifest.get("path"), str):
        raise FacingReviewError("facing bundle source manifest path is missing")
    current_source = authenticate_second_attempt(Path(source_manifest["path"]).parent)
    if current_source != source:
        raise FacingReviewError("facing bundle source snapshot changed")
    environment = manifest.get("environment")
    if not isinstance(environment, Mapping) or (
        environment.get("blender_version"),
        environment.get("fps"),
        environment.get("frame_count"),
        environment.get("resolution"),
    ) != ("4.2.1", 30, 33, [640, 360]):
        raise FacingReviewError("facing bundle environment is not pinned")
    return manifest


def _record_path(root: Path, record: Mapping[str, Any], description: str) -> Path:
    if isinstance(record.get("path"), str):
        return Path(record["path"])
    filename = record.get("filename")
    if not isinstance(filename, str) or Path(filename).name != filename:
        raise FacingReviewError(f"{description} record path is invalid")
    path = root / filename
    try:
        path.resolve().relative_to(root)
    except ValueError as error:
        raise FacingReviewError(f"{description} escaped the bundle") from error
    return path


def create_app(bundle_dir: Path | str) -> Flask:
    app = Flask(__name__)
    root = _real_directory(Path(bundle_dir), "facing review bundle")
    try:
        manifest = validate_facing_bundle(root)
    except FacingReviewError as error:
        raise ValueError(f"invalid second-retarget facing review bundle: {error}") from error
    manifest_path = root / "facing_review_manifest.json"
    manifest_sha256 = sha256_file(manifest_path)

    def unchanged_manifest() -> None:
        try:
            path = _real_file(manifest_path, "facing review manifest")
        except FacingReviewError as error:
            abort(409, description=str(error))
        if not hmac.compare_digest(manifest_sha256, sha256_file(path)):
            abort(409, description="facing review manifest changed after startup")

    def serve_record(record: Any, *, description: str, mimetype: str, download_name: str):
        unchanged_manifest()
        if not isinstance(record, Mapping):
            abort(409, description=f"{description} record is missing")
        try:
            path = _record_path(root, record, description)
            _check_record(path, record, description)
        except FacingReviewError as error:
            abort(409, description=str(error))
        response = send_file(
            path,
            mimetype=mimetype,
            download_name=download_name,
            conditional=True,
            max_age=0,
        )
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @app.get("/")
    def index():
        return serve_record(
            manifest.get("derived_artifacts", {}).get("review.html"),
            description="review.html",
            mimetype="text/html",
            download_name="review.html",
        )

    @app.get("/media/<view>")
    def media(view: str):
        if view not in VIEWS:
            abort(404)
        if view == "top":
            record = manifest.get("derived_artifacts", {}).get("top_facing.mp4")
        else:
            record = manifest.get("source", {}).get("media", {}).get(view, {}).get("mp4")
        return serve_record(
            record,
            description=f"{view} video",
            mimetype="video/mp4",
            download_name=f"walking_{view}.mp4",
        )

    @app.get("/metrics")
    def metrics():
        return serve_record(
            manifest.get("derived_artifacts", {}).get("facing_metrics.json"),
            description="facing metrics",
            mimetype="application/json",
            download_name="facing_metrics.json",
        )

    return app


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8098)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    create_app(args.bundle_dir).run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
