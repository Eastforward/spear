#!/usr/bin/env python3
"""Serve the hash-locked pre-bind versus second-retarget gait review."""

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
SPEAR_ROOT = MODULE_DIR.parents[1]
for directory in (MODULE_DIR, SPEAR_ROOT):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from tools import blender_retarget_rocketbox_to_tokenrig as runner
from second_retarget_facing_review import (
    FacingReviewError,
    authenticate_second_attempt,
    sha256_file,
)
from second_retarget_facing_review_server import validate_facing_bundle


SOURCE_VIEWS = ("front", "side", "top", "source_target")
TARGET_VIEWS = ("front", "side", "feet", "top")


def _real_dir(path: Path, description: str) -> Path:
    path = Path(os.path.abspath(os.fspath(path)))
    if path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise FacingReviewError(f"{description} must be a direct directory")
    return path


def _real_file(path: Path, description: str) -> Path:
    path = Path(os.path.abspath(os.fspath(path)))
    if path.is_symlink() or not path.is_file() or path.resolve() != path or path.stat().st_size <= 0:
        raise FacingReviewError(f"{description} must be a nonempty direct file")
    return path


def _load(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(_real_file(path, description).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FacingReviewError(f"{description} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise FacingReviewError(f"{description} root must be an object")
    return value


def _record_path(root: Path, record: Any, description: str) -> Path:
    if not isinstance(record, Mapping):
        raise FacingReviewError(f"{description} record is missing")
    if isinstance(record.get("path"), str):
        path = Path(record["path"])
    elif isinstance(record.get("filename"), str) and Path(record["filename"]).name == record["filename"]:
        path = root / record["filename"]
    else:
        raise FacingReviewError(f"{description} record path is invalid")
    path = _real_file(path, description)
    if record.get("size_bytes") != path.stat().st_size:
        raise FacingReviewError(f"{description} size changed")
    if record.get("sha256") != sha256_file(path):
        raise FacingReviewError(f"{description} hash changed")
    return path


def validate_bundle(bundle_dir: Path | str) -> dict[str, Any]:
    root = _real_dir(Path(bundle_dir), "prebind gait review bundle")
    manifest_path = root / "prebind_gait_review_manifest.json"
    manifest = _load(manifest_path, "prebind gait review manifest")
    if (
        manifest.get("schema") != "prebind_vs_second_retarget_gait_review_v1"
        or manifest.get("asset_id") != "rocketbox_male_adult_01"
        or manifest.get("classification") != "technical_diagnostic_only"
        or manifest.get("decision") != "rejected"
        or manifest.get("formal_dataset_asset") is not False
        or manifest.get("user_authority") != "human_visual_review_required"
    ):
        raise FacingReviewError("prebind gait review classification is invalid")
    if "user_approved" in json.dumps(manifest, sort_keys=True):
        raise FacingReviewError("prebind gait review may not claim user approval")
    local = manifest.get("local_artifacts")
    if not isinstance(local, Mapping) or set(local) != {"review.html", "prebind_gait_metrics.json"}:
        raise FacingReviewError("prebind gait local artifact inventory is invalid")
    for name in local:
        _record_path(root, local[name], name)
    metrics = _load(root / "prebind_gait_metrics.json", "prebind gait metrics")
    if (
        metrics.get("schema") != "prebind_vs_second_retarget_gait_plane_v1"
        or metrics.get("source_prebind", {}).get("overall_classification") != "sagittal_forward_gait"
        or metrics.get("target_second_retarget", {}).get("overall_classification") != "sideways_leg_swing"
    ):
        raise FacingReviewError("prebind gait metrics do not prove the stage transition")
    source = manifest.get("source_prebind")
    target = manifest.get("target_second_retarget")
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise FacingReviewError("prebind gait stage records are missing")
    for view in SOURCE_VIEWS:
        _record_path(root, source.get("media", {}).get(view), f"source {view}")
    for view in TARGET_VIEWS:
        _record_path(root, target.get("media", {}).get(view), f"target {view}")
    baseline_auth = source.get("authentication")
    if not isinstance(baseline_auth, Mapping):
        raise FacingReviewError("baseline authentication is missing")
    current_baseline = runner.authenticate_sealed_walk(
        base_avatar_id="rocketbox_male_adult_01",
        baseline_retarget_blend=Path(baseline_auth["retarget_blend"]["path"]),
        baseline_retarget_manifest=Path(baseline_auth["retarget_manifest"]["path"]),
    )
    if current_baseline != baseline_auth:
        raise FacingReviewError("sealed prebind baseline changed")
    second_auth = target.get("authentication")
    if not isinstance(second_auth, Mapping):
        raise FacingReviewError("second retarget authentication is missing")
    current_second = authenticate_second_attempt(Path(second_auth["manifest"]["path"]).parent)
    if current_second != second_auth:
        raise FacingReviewError("second retarget source changed")
    top_path = Path(target["media"]["top"]["path"])
    facing = validate_facing_bundle(top_path.parent)
    if sha256_file(top_path.parent / "facing_review_manifest.json") != target.get("facing_bundle_manifest_sha256"):
        raise FacingReviewError("second facing bundle manifest changed")
    if facing.get("source") != current_second:
        raise FacingReviewError("second facing bundle no longer binds the second attempt")
    return manifest


def create_app(bundle_dir: Path | str) -> Flask:
    app = Flask(__name__)
    root = _real_dir(Path(bundle_dir), "prebind gait review bundle")
    try:
        manifest = validate_bundle(root)
    except FacingReviewError as error:
        raise ValueError(f"invalid prebind gait review: {error}") from error
    manifest_path = root / "prebind_gait_review_manifest.json"
    initial_hash = sha256_file(manifest_path)

    def unchanged() -> None:
        try:
            current = _real_file(manifest_path, "prebind gait review manifest")
        except FacingReviewError as error:
            abort(409, description=str(error))
        if not hmac.compare_digest(initial_hash, sha256_file(current)):
            abort(409, description="prebind gait review manifest changed after startup")

    def serve(record: Any, description: str, mimetype: str, name: str):
        unchanged()
        try:
            path = _record_path(root, record, description)
        except FacingReviewError as error:
            abort(409, description=str(error))
        response = send_file(path, mimetype=mimetype, download_name=name, conditional=True, max_age=0)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @app.get("/")
    def index():
        return serve(manifest["local_artifacts"]["review.html"], "review.html", "text/html", "review.html")

    @app.get("/metrics")
    def metrics():
        return serve(manifest["local_artifacts"]["prebind_gait_metrics.json"], "prebind gait metrics", "application/json", "prebind_gait_metrics.json")

    @app.get("/source/<view>")
    def source_media(view: str):
        if view not in SOURCE_VIEWS:
            abort(404)
        return serve(manifest["source_prebind"]["media"][view], f"source {view}", "video/mp4", f"source_{view}.mp4")

    @app.get("/target/<view>")
    def target_media(view: str):
        if view not in TARGET_VIEWS:
            abort(404)
        return serve(manifest["target_second_retarget"]["media"][view], f"target {view}", "video/mp4", f"target_{view}.mp4")

    return app


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8099)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    create_app(args.bundle_dir).run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
