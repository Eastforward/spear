"""Generate reviewed FLUX.2 Klein human-reference candidates from approved images."""

import argparse
import hashlib
import hmac
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image


PINNED_MODEL_ROOT = Path("/data/models/hub/models--black-forest-labs--FLUX.2-klein-4B")
MODEL_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
OUTPUT_WIDTH = 1152
OUTPUT_HEIGHT = 1536
MAX_SEQUENCE_LENGTH = 512
EXPECTED_ASSET_IDS = frozenset(
    {"rocketbox_male_adult_01", "rocketbox_female_adult_01"}
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
REQUIRED_JOB_FIELDS = frozenset(
    {
        "asset_id",
        "source_image",
        "source_image_sha256",
        "source_review",
        "prompt",
        "seed",
        "width",
        "height",
        "steps",
        "guidance_scale",
    }
)


def write_candidate_manifest(candidate_dir: Path, **kwargs: Any) -> Path:
    """Delegate manifest creation to the pure human-reference contract module."""
    contract_dir = Path(__file__).resolve().parent / "spike_rlr"
    if str(contract_dir) not in sys.path:
        sys.path.insert(0, str(contract_dir))
    from human_reference_review import write_candidate_manifest as write_manifest

    return write_manifest(candidate_dir, **kwargs)


def assert_source_review_approved(path: Path) -> dict[str, Any]:
    """Delegate approved-source validation to the Rocketbox review contract."""
    review_dir = Path(__file__).resolve().parent / "spike_rlr"
    if str(review_dir) not in sys.path:
        sys.path.insert(0, str(review_dir))
    from rocketbox_human_review import assert_source_review_approved as assert_approved

    return assert_approved(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model-root", type=Path, default=PINNED_MODEL_ROOT)
    parser.add_argument("--local-files-only", action="store_true", required=True)
    return parser.parse_args()


def validate_job(job: Mapping[str, Any]) -> None:
    missing = REQUIRED_JOB_FIELDS.difference(job)
    if missing:
        raise ValueError(f"job missing required fields: {', '.join(sorted(missing))}")
    if job["asset_id"] not in EXPECTED_ASSET_IDS:
        raise ValueError(f"unexpected asset_id: {job['asset_id']!r}")
    if not isinstance(job["source_image"], str) or not job["source_image"]:
        raise ValueError("source_image must be a non-empty string")
    if not isinstance(job["prompt"], str) or not job["prompt"]:
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(job["source_review"], str) or not job["source_review"]:
        raise ValueError("source_review must be a non-empty string")
    source_image_sha256 = job["source_image_sha256"]
    if (
        not isinstance(source_image_sha256, str)
        or _SHA256_RE.fullmatch(source_image_sha256) is None
    ):
        raise ValueError(
            "source_image_sha256 must be a 64-character lowercase hex value"
        )
    if job["width"] != OUTPUT_WIDTH or job["height"] != OUTPUT_HEIGHT:
        raise ValueError(f"jobs must produce {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} images")
    if not isinstance(job["seed"], int):
        raise ValueError("seed must be an integer")
    if not isinstance(job["steps"], int) or job["steps"] <= 0:
        raise ValueError("steps must be a positive integer")
    if not isinstance(job["guidance_scale"], (int, float)):
        raise ValueError("guidance_scale must be numeric")


def load_jobs(path: Path) -> list[dict[str, Any]]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read jobs JSON {path}: {exc}") from exc
    jobs = loaded["jobs"] if isinstance(loaded, dict) and "jobs" in loaded else loaded
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs JSON must contain a non-empty list of jobs")
    asset_ids = set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError("each job must be a JSON object")
        validate_job(job)
        if job["asset_id"] in asset_ids:
            raise ValueError(f"duplicate asset_id: {job['asset_id']}")
        asset_ids.add(job["asset_id"])
    return jobs


def _absolute_without_symlinks(path: Path, description: str) -> Path:
    absolute = Path(path).absolute()
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component) and stat.S_ISLNK(os.lstat(component).st_mode):
            raise ValueError(f"{description} path must not contain a symlink: {component}")
    return absolute


def _real_directory(path: Path, description: str) -> Path:
    absolute = _absolute_without_symlinks(path, description)
    if not os.path.lexists(absolute):
        raise ValueError(f"{description} is missing: {absolute}")
    if not stat.S_ISDIR(os.lstat(absolute).st_mode):
        raise ValueError(f"{description} must be a real directory: {absolute}")
    if absolute.resolve() != absolute:
        raise ValueError(f"{description} resolved path is not exact: {absolute}")
    return absolute


def _regular_file_without_symlinks(path: Path, description: str) -> Path:
    absolute = _absolute_without_symlinks(path, description)
    if not os.path.lexists(absolute):
        raise ValueError(f"{description} is missing: {absolute}")
    if not stat.S_ISREG(os.lstat(absolute).st_mode):
        raise ValueError(f"{description} must be a regular file: {absolute}")
    if absolute.resolve() != absolute:
        raise ValueError(f"{description} resolved path is not exact: {absolute}")
    return absolute


def _validated_source_contract(job: Mapping[str, Any]) -> tuple[Path, Path]:
    source = _regular_file_without_symlinks(
        Path(job["source_image"]), "source image"
    )
    review = _regular_file_without_symlinks(
        Path(job["source_review"]), "source review"
    )
    if source.name != "front.png":
        raise ValueError("source image must be the direct front.png file")
    if review.name != "source_review.json":
        raise ValueError("source review must be the direct source_review.json file")
    if source.parent != review.parent:
        raise ValueError(
            "source image and source review must be directly under the same directory"
        )
    return source, review


def _prepare_output_root(output_root: Path) -> Path:
    output_root = _absolute_without_symlinks(output_root, "output_root")
    if not os.path.lexists(output_root):
        output_root.mkdir(parents=True)
    return _real_directory(output_root, "output_root")


def _prepare_candidate_dir(output_root: Path, asset_id: str) -> Path:
    candidate_dir = output_root / asset_id
    if candidate_dir.parent != output_root:
        raise ValueError("candidate directory must be directly under output_root")
    _absolute_without_symlinks(candidate_dir, "candidate directory")
    if not os.path.lexists(candidate_dir):
        candidate_dir.mkdir()
    candidate_dir = _real_directory(candidate_dir, "candidate directory")
    if candidate_dir.parent != output_root:
        raise ValueError("candidate directory escaped output_root containment")
    return candidate_dir


def load_pipeline(model_root: Path, *, local_files_only: bool):
    model_root = Path(model_root).absolute()
    pinned_model_root = Path(PINNED_MODEL_ROOT).absolute()
    if model_root != pinned_model_root:
        raise ValueError(f"model_root must be the pinned local snapshot: {PINNED_MODEL_ROOT}")
    if not local_files_only:
        raise ValueError("local_files_only must be True")
    model_root = _real_directory(model_root, "model cache root")
    snapshot = model_root / "snapshots" / MODEL_REVISION
    snapshot = _real_directory(snapshot, "pinned model snapshot")
    expected_snapshot = model_root / "snapshots" / MODEL_REVISION
    if snapshot != expected_snapshot or snapshot.resolve() != expected_snapshot:
        raise ValueError(
            f"pinned model snapshot resolved path must be exact: {expected_snapshot}"
        )
    if not (snapshot / "model_index.json").is_file():
        raise ValueError(f"pinned model snapshot is missing model_index.json: {snapshot}")
    import torch
    from diffusers import Flux2KleinPipeline

    return Flux2KleinPipeline.from_pretrained(
        str(snapshot),
        torch_dtype=torch.bfloat16,
        local_files_only=local_files_only,
    ).to("cuda")


def validate_png(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"expected PNG, got {image.format!r}")
            if image.size != (OUTPUT_WIDTH, OUTPUT_HEIGHT):
                raise ValueError(
                    f"expected {OUTPUT_WIDTH}x{OUTPUT_HEIGHT} PNG, got {image.size[0]}x{image.size[1]}"
                )
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid candidate PNG {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_png_atomically(image: Image.Image, target: Path) -> None:
    target = Path(target).absolute()
    _real_directory(target.parent, "PNG output parent")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            image.save(handle, format="PNG")
            handle.flush()
            os.fsync(handle.fileno())
        validate_png(temporary)
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def copy_source_image(source: Path, destination: Path) -> None:
    source = _regular_file_without_symlinks(source, "source image")
    destination = Path(destination).absolute()
    _real_directory(destination.parent, "copied source parent")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle, source.open("rb") as input_handle:
            shutil.copyfileobj(input_handle, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def run_jobs(jobs: Sequence[Mapping[str, Any]], output_root: Path, pipeline: Any) -> None:
    import torch

    for job in jobs:
        validate_job(job)
    output_root = _prepare_output_root(output_root)

    for job in jobs:
        source_path, source_review_path = _validated_source_contract(job)
        source_review = assert_source_review_approved(source_review_path)
        if source_review["asset_id"] != job["asset_id"]:
            raise RuntimeError(
                f"source review asset_id {source_review['asset_id']!r} "
                f"does not match {job['asset_id']!r}"
            )
        if not hmac.compare_digest(
            sha256_file(source_path), job["source_image_sha256"]
        ):
            raise ValueError("source image hash does not match source_image_sha256 pin")
        source_approval_sha256 = sha256_file(source_review_path)
        candidate_dir = _prepare_candidate_dir(output_root, job["asset_id"])
        copied_source = candidate_dir / "source.png"
        copy_source_image(source_path, copied_source)
        if not hmac.compare_digest(
            sha256_file(copied_source), job["source_image_sha256"]
        ):
            raise ValueError("copied source image hash does not match source_image_sha256 pin")
        with Image.open(copied_source) as source_image:
            source = source_image.convert("RGB")
            generator = torch.Generator("cuda").manual_seed(job["seed"])
            result = pipeline(
                image=source,
                prompt=job["prompt"],
                width=job["width"],
                height=job["height"],
                num_inference_steps=job["steps"],
                guidance_scale=job["guidance_scale"],
                generator=generator,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
            )
        candidate_path = candidate_dir / "candidate.png"
        save_png_atomically(result.images[0], candidate_path)
        validate_png(candidate_path)
        write_candidate_manifest(
            candidate_dir,
            asset_id=job["asset_id"],
            model_revision=MODEL_REVISION,
            prompt=job["prompt"],
            seed=job["seed"],
            width=job["width"],
            height=job["height"],
            steps=job["steps"],
            guidance_scale=job["guidance_scale"],
            source_approval_sha256=source_approval_sha256,
        )


def main() -> None:
    args = parse_args()
    if not args.local_files_only:
        raise SystemExit("--local-files-only is required")
    if args.model_root.resolve() != PINNED_MODEL_ROOT.resolve():
        raise SystemExit(f"--model-root must be the pinned local snapshot: {PINNED_MODEL_ROOT}")
    if not args.model_root.is_dir():
        raise SystemExit(f"pinned model snapshot is missing: {args.model_root}")
    jobs = load_jobs(args.jobs_json)
    pipeline = load_pipeline(args.model_root, local_files_only=True)
    run_jobs(jobs, args.output_root, pipeline)


if __name__ == "__main__":
    main()
