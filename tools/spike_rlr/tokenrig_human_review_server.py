#!/usr/bin/env python3
"""Read-only Flask server for a hash-locked Route-2 review catalog."""

from __future__ import annotations

import argparse
import hmac
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from flask import Flask, abort, request, send_file

from tokenrig_human_review import (
    MOTIONS,
    STATIC_EVIDENCE,
    VIEWS,
    ReviewContractError,
    sha256_file,
    validate_review_catalog,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _entry(catalog: Mapping[str, Any], asset_id: str) -> Mapping[str, Any]:
    for entry in catalog.get("entries", []):
        if isinstance(entry, Mapping) and entry.get("asset_id") == asset_id:
            return entry
    abort(404)


def _expected_sha256(value: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        abort(400, description="expected_sha256 must be 64 lowercase hexadecimal characters")
    return value


def _serve_record(record: Any, *, expected_sha256: str, mimetype: str, download_name: str):
    if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
        abort(409, description="catalog artifact descriptor is missing")
    expected = _expected_sha256(expected_sha256)
    recorded = record.get("sha256")
    if not isinstance(recorded, str) or not hmac.compare_digest(expected, recorded):
        abort(409, description="requested hash does not match the current review catalog")
    path = Path(record["path"])
    if path.is_symlink() or not path.is_file() or path.resolve() != path or path.stat().st_size <= 0:
        abort(409, description="review artifact is no longer a direct regular file")
    if record.get("size_bytes") != path.stat().st_size:
        abort(409, description="review artifact size changed")
    actual = sha256_file(path)
    if not hmac.compare_digest(expected, actual):
        abort(409, description="review artifact hash changed")
    response = send_file(path, mimetype=mimetype, download_name=download_name, conditional=True, max_age=0)
    response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


def create_app(catalog_dir: Path | str) -> Flask:
    app = Flask(__name__)
    root = Path(catalog_dir).absolute()

    try:
        initial_catalog = validate_review_catalog(root)
    except ReviewContractError as error:
        raise ValueError(f"invalid Route-2 review catalog: {error}") from error
    catalog_path = root / "review_catalog.json"
    initial_catalog_sha256 = sha256_file(catalog_path)

    def unchanged_catalog() -> dict[str, Any]:
        if (
            catalog_path.is_symlink()
            or not catalog_path.is_file()
            or catalog_path.resolve() != catalog_path
            or not hmac.compare_digest(
                initial_catalog_sha256, sha256_file(catalog_path)
            )
        ):
            abort(409, description="review catalog bytes changed after server startup")
        try:
            current = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            abort(409, description="review catalog is no longer readable")
        if current != initial_catalog:
            abort(409, description="review catalog snapshot changed after server startup")
        return current

    def current_catalog(*, validate_sources: bool) -> dict[str, Any]:
        unchanged_catalog()
        if not validate_sources:
            return initial_catalog
        try:
            return validate_review_catalog(root)
        except ReviewContractError as error:
            abort(409, description=str(error))

    @app.get("/")
    def index():
        catalog = current_catalog(validate_sources=True)
        record = catalog.get("review_html")
        html_path = root / "review.html"
        if (
            not isinstance(record, Mapping)
            or record.get("filename") != "review.html"
            or html_path.is_symlink()
            or not html_path.is_file()
            or html_path.resolve() != html_path
            or record.get("sha256") != sha256_file(html_path)
            or record.get("size_bytes") != html_path.stat().st_size
        ):
            abort(409, description="hash-locked review.html changed")
        response = send_file(
            html_path,
            mimetype="text/html",
            download_name="review.html",
            conditional=True,
            max_age=0,
        )
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @app.get("/media/<asset_id>/<motion>/<view>/<kind>")
    def media(asset_id: str, motion: str, view: str, kind: str):
        if motion not in MOTIONS or view not in VIEWS or kind not in {"png", "mp4"}:
            abort(404)
        catalog = current_catalog(validate_sources=False)
        entry = _entry(catalog, asset_id)
        record = entry.get("media", {}).get(motion, {}).get(view, {}).get(kind)
        return _serve_record(
            record,
            expected_sha256=request.args.get("expected_sha256", ""),
            mimetype="video/mp4" if kind == "mp4" else "image/png",
            download_name=f"{motion}_{view}.{kind}",
        )

    @app.get("/static-evidence/<asset_id>/<filename>")
    def static_evidence(asset_id: str, filename: str):
        if filename not in STATIC_EVIDENCE:
            abort(404)
        catalog = current_catalog(validate_sources=False)
        entry = _entry(catalog, asset_id)
        record = entry.get("static_evidence", {}).get(filename)
        return _serve_record(
            record,
            expected_sha256=request.args.get("expected_sha256", ""),
            mimetype="text/plain" if filename.endswith(".txt") else "image/png",
            download_name=filename,
        )

    return app


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8097)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    create_app(args.catalog_dir).run(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
