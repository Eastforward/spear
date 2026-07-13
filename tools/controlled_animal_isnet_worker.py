#!/usr/bin/env python3
"""Persistent local ISNet foreground worker for approved animal candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

from PIL import Image
import numpy as np


MODEL_PATH = Path("/data/models/rembg/isnet-general-use/isnet-general-use.onnx")
MODEL_SHA256 = "60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a"
JOBS_SCHEMA = "avengine_controlled_animal_isnet_jobs_v1"
STATUS_SCHEMA = "avengine_controlled_animal_isnet_status_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _save_png(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace ISNet output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            image.save(handle, format="PNG", optimize=False, compress_level=6)
            handle.flush()
            os.fsync(handle.fileno())
        os.rename(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_jobs(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != JOBS_SCHEMA:
        raise ValueError(f"ISNet jobs schema must be {JOBS_SCHEMA}")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("ISNet jobs must be a non-empty list")
    if len({item.get("instance_id") for item in jobs}) != len(jobs):
        raise ValueError("ISNet jobs contain duplicate instance IDs")
    for job in jobs:
        if set(job) != {
            "instance_id",
            "candidate_path",
            "candidate_sha256",
            "alpha_path",
            "rgba_path",
        }:
            raise ValueError("ISNet job fields are invalid")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    args = parser.parse_args(argv)
    if MODEL_PATH.is_symlink() or not MODEL_PATH.is_file() or _sha256_file(MODEL_PATH) != MODEL_SHA256:
        raise RuntimeError("pinned ISNet model is missing or changed")
    os.environ.update(
        {
            "U2NET_HOME": str(MODEL_PATH.parent),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        }
    )
    from rembg import new_session, remove

    payload = load_jobs(args.jobs)
    status: dict[str, Any] = {
        "schema": STATUS_SCHEMA,
        "status": "running",
        "model": {
            "path": str(MODEL_PATH),
            "sha256": MODEL_SHA256,
            "name": "isnet-general-use",
        },
        "model_load_seconds": None,
        "jobs": [],
    }
    _atomic_json(args.status, status)
    load_started = time.perf_counter()
    session = new_session("isnet-general-use")
    status["model_load_seconds"] = time.perf_counter() - load_started
    _atomic_json(args.status, status)
    for job in payload["jobs"]:
        started = time.perf_counter()
        record = {"instance_id": job["instance_id"], "status": "running"}
        try:
            candidate_path = Path(job["candidate_path"]).resolve()
            if (
                candidate_path.is_symlink()
                or not candidate_path.is_file()
                or _sha256_file(candidate_path) != job["candidate_sha256"]
            ):
                raise RuntimeError("approved candidate changed before ISNet")
            with Image.open(candidate_path) as opened:
                opened.load()
                rgb = opened.convert("RGB")
            mask = remove(
                rgb,
                session=session,
                only_mask=True,
                post_process_mask=True,
            ).convert("L")
            if mask.size != rgb.size:
                raise RuntimeError("ISNet alpha canvas changed")
            alpha = np.asarray(mask, dtype=np.uint8)
            foreground = alpha >= 128
            fraction = float(foreground.mean())
            if not 0.05 <= fraction <= 0.85 or alpha.min() != 0 or alpha.max() != 255:
                raise RuntimeError(
                    f"ISNet alpha foreground/extrema are implausible: fraction={fraction}"
                )
            rows, columns = np.nonzero(foreground)
            bbox = [
                int(columns.min()),
                int(rows.min()),
                int(columns.max()) + 1,
                int(rows.max()) + 1,
            ]
            rgba = rgb.convert("RGBA")
            rgba.putalpha(mask)
            alpha_path = Path(job["alpha_path"]).resolve()
            rgba_path = Path(job["rgba_path"]).resolve()
            _save_png(alpha_path, mask)
            _save_png(rgba_path, rgba)
            if _sha256_file(candidate_path) != job["candidate_sha256"]:
                raise RuntimeError("approved candidate changed during ISNet")
            record.update(
                {
                    "status": "passed",
                    "alpha_path": str(alpha_path),
                    "alpha_sha256": _sha256_file(alpha_path),
                    "rgba_path": str(rgba_path),
                    "rgba_sha256": _sha256_file(rgba_path),
                    "foreground_fraction_at_128": fraction,
                    "foreground_bbox_xyxy": bbox,
                    "alpha_extrema": [int(alpha.min()), int(alpha.max())],
                    "wall_seconds": time.perf_counter() - started,
                }
            )
        except BaseException as error:
            record.update(
                {
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "wall_seconds": time.perf_counter() - started,
                }
            )
        status["jobs"].append(record)
        _atomic_json(args.status, status)
        print(
            f"CONTROLLED_ANIMAL_ISNET {job['instance_id']} {record['status']} "
            f"wall={record['wall_seconds']:.1f}s",
            flush=True,
        )
    status["passed_count"] = sum(item["status"] == "passed" for item in status["jobs"])
    status["failed_count"] = sum(item["status"] == "failed" for item in status["jobs"])
    status["status"] = "passed" if status["failed_count"] == 0 else "failed"
    _atomic_json(args.status, status)
    return 1 if status["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
