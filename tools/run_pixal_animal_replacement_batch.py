"""Generate resumable Pixal3D replacements for every legacy animal source."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


Image.MAX_IMAGE_PIXELS = None
SPEAR_ROOT = Path(__file__).resolve().parents[1]
AVENGINE_ROOT = SPEAR_ROOT.parents[1]
PIXAL_PYTHON = Path("/data/jzy/miniconda3/envs/avengine-3dgen/bin/python3.10")
PIXAL_LAUNCHER = SPEAR_ROOT / "tools/i23d_human_bakeoff.py"
DEFAULT_OUT_ROOT = (
    SPEAR_ROOT / "tmp/pixal_animal_backend_substitution_v1/generated_batch_v1"
)
EXISTING_PUG_MANIFEST = (
    SPEAR_ROOT
    / "tmp/pixal_animal_backend_substitution_v1/dog_pug_pixal_canary_v1/"
    "pixal_raw_1024_seed5101.manifest.json"
)

ANIMATED_TAGS = (
    "cat_british_shorthair_v2",
    "cat_persian",
    "cat_siamese_v1",
    "cat_tabby",
    "chipmunk",
    "dog_beagle_v2",
    "dog_golden",
    "dog_pug_v1",
)
STATIC_TAGS = (
    "cattle_bovinae",
    "donkey_ass",
    "goat",
    "horse",
    "pig",
    "sheep",
    "yak",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor(path: Path) -> dict:
    path = path.resolve()
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def source_reference(tag: str) -> Path:
    if tag in ANIMATED_TAGS:
        return SPEAR_ROOT / "tmp/hy3d_batch" / tag / "reference.png"
    if tag in STATIC_TAGS:
        return (
            AVENGINE_ROOT
            / "external/Hunyuan3D-2.1/outputs/audioset_assets"
            / tag
            / "source.png"
        )
    raise ValueError(f"unknown legacy animal tag: {tag}")


def normalize_reference(source: Path, destination: Path) -> dict:
    """Create a bounded RGBA input while preserving the source and its hash."""
    source = source.resolve()
    destination = destination.resolve()
    with Image.open(source) as opened:
        image = opened.convert("RGBA")
        original_size = list(image.size)
        alpha = image.getchannel("A")
        alpha_min, alpha_max = alpha.getextrema()
        if alpha_min == 255 or alpha_max == 0:
            raise ValueError(f"animal reference lacks transparent foreground: {source}")
        if max(image.size) <= 1024:
            # Existing FLUX references are already the approved 1024 input.
            return {
                "source": _descriptor(source),
                "pixal_input": _descriptor(source),
                "original_size": original_size,
                "normalized_size": original_size,
                "normalization": "none_existing_rgba_at_or_below_1024",
                "alpha_range": [int(alpha_min), int(alpha_max)],
            }
        scale = 922.0 / max(image.size)
        resized_size = tuple(max(1, int(round(value * scale))) for value in image.size)
        resized = image.resize(resized_size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (1024, 1024), (0, 0, 0, 0))
        offset = ((1024 - resized.width) // 2, (1024 - resized.height) // 2)
        canvas.alpha_composite(resized, dest=offset)
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, format="PNG", optimize=False, compress_level=6)
    return {
        "source": _descriptor(source),
        "pixal_input": _descriptor(destination),
        "original_size": original_size,
        "normalized_size": [1024, 1024],
        "foreground_size": list(resized_size),
        "foreground_offset": list(offset),
        "normalization": "rgba_lanczos_contain_90pct_transparent_canvas_v1",
        "alpha_range": [int(alpha_min), int(alpha_max)],
    }


def build_jobs(out_root: Path, include_tags: set[str] | None = None) -> list[dict]:
    if include_tags is None:
        include_tags = set(ANIMATED_TAGS + STATIC_TAGS)
        if EXISTING_PUG_MANIFEST.is_file():
            include_tags.discard("dog_pug_v1")
    else:
        include_tags = set(include_tags)
    unknown = include_tags - set(ANIMATED_TAGS + STATIC_TAGS)
    if unknown:
        raise ValueError(f"unknown animal tags: {sorted(unknown)}")
    jobs = []
    for index, tag in enumerate(sorted(include_tags)):
        source = source_reference(tag).resolve()
        if not source.is_file():
            raise ValueError(f"missing stable animal reference: {source}")
        seed = 5301 + index
        tag_dir = out_root / f"{tag}_pixal_v1"
        normalized = out_root / "references_1024" / f"{tag}.png"
        reference = normalize_reference(source, normalized)
        output = tag_dir / f"pixal_raw_1024_seed{seed}.glb"
        jobs.append(
            {
                "legacy_tag": tag,
                "candidate_tag": f"{tag}_pixal_v1",
                "rig_mode": "animated_transfer" if tag in ANIMATED_TAGS else "rig_required",
                "seed": seed,
                "reference": reference,
                "output": str(output.resolve()),
                "manifest": str(output.with_suffix(".manifest.json").resolve()),
                "log": str((tag_dir / "pixal.log").resolve()),
            }
        )
    return jobs


def job_is_complete(job: dict) -> bool:
    output = Path(job["output"])
    manifest_path = Path(job["manifest"])
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return (
        output.is_file()
        and output.stat().st_size > 0
        and manifest.get("backend") == "pixal3d"
        and manifest.get("parameters", {}).get("seed") == job["seed"]
        and manifest.get("input", {}).get("sha256")
        == job["reference"]["pixal_input"]["sha256"]
        and manifest.get("output", {}).get("sha256") == _sha256(output)
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--gpu", type=int, action="append", default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be in [1, 4]")
    gpus = args.gpu or [0, 1, 2, 3]
    if len(gpus) < args.workers or len(set(gpus[: args.workers])) != args.workers:
        raise ValueError("provide one unique GPU per worker")
    out_root = args.out_root.resolve()
    jobs = build_jobs(out_root, set(args.tag) if args.tag else None)
    resources = queue.Queue()
    for gpu in gpus[: args.workers]:
        resources.put(gpu)
    started_at = _utc_now()

    def run(job):
        if args.resume and job_is_complete(job):
            return {**job, "status": "reused", "gpu": None, "wall_seconds": 0.0}
        gpu = resources.get()
        started = time.perf_counter()
        command = [
            str(PIXAL_PYTHON),
            str(PIXAL_LAUNCHER),
            "--backend", "pixal3d",
            "--image", job["reference"]["pixal_input"]["path"],
            "--output", job["output"],
            "--gpu", str(gpu),
            "--seed", str(job["seed"]),
            "--resolution", "1024",
            "--manual-fov", "0.2",
            "--low-vram",
        ]
        try:
            Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
            with Path(job["log"]).open("ab") as stream:
                result = subprocess.run(
                    command,
                    cwd=SPEAR_ROOT,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                )
            if result.returncode != 0:
                raise RuntimeError(f"Pixal3D returned {result.returncode}")
            if not job_is_complete(job):
                raise RuntimeError("Pixal3D output manifest contract is incomplete")
            return {
                **job,
                "status": "passed",
                "gpu": gpu,
                "wall_seconds": time.perf_counter() - started,
                "output_evidence": _descriptor(Path(job["output"])),
                "generation_manifest": _descriptor(Path(job["manifest"])),
            }
        finally:
            resources.put(gpu)

    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except BaseException as error:
                failures.append(
                    {
                        "legacy_tag": job["legacy_tag"],
                        "candidate_tag": job["candidate_tag"],
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "log": job["log"],
                    }
                )
                print(f"PIXAL_ANIMAL_FAILED {job['legacy_tag']}: {error}", flush=True)
            else:
                results.append(result)
                print(
                    f"PIXAL_ANIMAL_OK {job['legacy_tag']} status={result['status']} "
                    f"gpu={result['gpu']} wall={result['wall_seconds']:.1f}s",
                    flush=True,
                )

    status = {
        "schema": "pixal_animal_replacement_batch_v1",
        "started_at": started_at,
        "finished_at": _utc_now(),
        "usage_scope": "research_candidate",
        "formal_registration_authorized": False,
        "legacy_hunyuan_assets_remain": "technical_spike_only",
        "model_revision": "0b31f9160aa400719af409098bff7936a932f726",
        "dino_revision": "3c276edd87d6f6e569ff0c4400e086807d0f3881",
        "parameters": {
            "resolution": 1024,
            "manual_fov": 0.2,
            "low_vram": True,
        },
        "workers": args.workers,
        "gpus": gpus[: args.workers],
        "job_count": len(jobs),
        "passed_count": len(results),
        "failed_count": len(failures),
        "preexisting_pixal_pug": (
            _descriptor(EXISTING_PUG_MANIFEST)
            if EXISTING_PUG_MANIFEST.is_file()
            else None
        ),
        "results": sorted(results, key=lambda item: item["legacy_tag"]),
        "failures": sorted(failures, key=lambda item: item["legacy_tag"]),
    }
    _atomic_json(out_root / "batch_status.json", status)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
