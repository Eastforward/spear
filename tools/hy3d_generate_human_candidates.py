"""Generate Hunyuan3D technical-spike candidates from approved references."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import importlib.machinery
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
CONTRACT_DIR = SCRIPT_DIR / "spike_rlr"
if str(CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTRACT_DIR))

from hy3d_human_candidate import (  # noqa: E402
    CANONICAL_DINOV2_ROOT,
    CANONICAL_MODEL_PARENT,
    CANONICAL_MODEL_ROOT,
    CANONICAL_REALESRGAN_CKPT,
    EXPECTED_ASSET_IDS,
    HUNYUAN_CHECKOUT,
    OUTPUT_FILENAMES,
    REQUIRED_MODEL_FILES,
    WEIGHT_ROOT_HASH_MANIFEST,
    Hy3DHumanNotReady,
    assert_generation_ready,
    current_weight_manifest_sha256,
    verify_canonical_weights,
    write_hy3d_manifest,
)


HUNYUAN_IMPORT_PATHS = (
    HUNYUAN_CHECKOUT / "hy3dshape",
    HUNYUAN_CHECKOUT / "hy3dpaint",
)
OFFLINE_ENVIRONMENT = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "DIFFUSERS_OFFLINE": "1",
}
APPROVAL_JOB_FIELDS = (
    "asset_id",
    "review_root",
    "candidate_path",
    "candidate_sha256",
    "candidate_manifest_sha256",
    "source_sha256",
    "source_approval_sha256",
    "reference_review_sha256",
    "seed",
    "steps",
    "guidance_scale",
    "model_root",
    "weight_root_hash_manifest",
    "hunyuan_runtime_git_head",
    "hunyuan_runtime_fingerprint",
    "hunyuan_runtime_file_count",
)
_MISSING = object()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _absolute_without_symlinks(path: Path, description: str) -> Path:
    absolute = Path(path).absolute()
    for component in (absolute, *absolute.parents):
        if os.path.lexists(component) and stat.S_ISLNK(os.lstat(component).st_mode):
            raise ValueError(f"{description} path must not contain a symlink: {component}")
    return absolute


def _real_directory(path: Path, description: str) -> Path:
    absolute = _absolute_without_symlinks(path, description)
    if not os.path.lexists(absolute) or not stat.S_ISDIR(os.lstat(absolute).st_mode):
        raise ValueError(f"{description} must be a real directory: {absolute}")
    if absolute.resolve() != absolute:
        raise ValueError(f"{description} resolved path is not exact: {absolute}")
    return absolute


def _regular_file_without_symlinks(path: Path, description: str) -> Path:
    absolute = _absolute_without_symlinks(path, description)
    if not os.path.lexists(absolute) or not stat.S_ISREG(os.lstat(absolute).st_mode):
        raise ValueError(f"{description} must be a regular file: {absolute}")
    if absolute.resolve() != absolute or absolute.stat().st_size <= 0:
        raise ValueError(f"{description} must be a non-empty regular file: {absolute}")
    return absolute


def _prepare_output_root(path: Path) -> Path:
    output_root = _absolute_without_symlinks(path, "output_root")
    if not os.path.lexists(output_root):
        output_root.mkdir(parents=True)
    return _real_directory(output_root, "output_root")


def _prepare_asset_dir(output_root: Path, asset_id: str) -> Path:
    asset_dir = output_root / asset_id
    if asset_dir.parent != output_root:
        raise ValueError("asset directory escaped output_root containment")
    _absolute_without_symlinks(asset_dir, "asset directory")
    if not os.path.lexists(asset_dir):
        asset_dir.mkdir()
    return _real_directory(asset_dir, "asset directory")


def _validate_job_asset_dir(job: Mapping[str, Any]) -> Path:
    asset_id = job.get("asset_id")
    if asset_id not in EXPECTED_ASSET_IDS:
        raise ValueError(f"asset_id is outside the approved allowlist: {asset_id!r}")
    asset_dir = _real_directory(Path(job["asset_dir"]), "asset directory")
    if asset_dir.name != asset_id:
        raise ValueError("asset directory name must equal asset_id")
    _real_directory(asset_dir.parent, "output_root")
    return asset_dir


def _copy_atomically(source: Path, destination: Path) -> None:
    source = _regular_file_without_symlinks(source, "approved candidate image")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination_stream, source.open(
            "rb"
        ) as source_stream:
            shutil.copyfileobj(source_stream, destination_stream)
            destination_stream.flush()
            os.fsync(destination_stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_hunyuan_checkout() -> Path:
    checkout = _real_directory(HUNYUAN_CHECKOUT, "Hunyuan checkout")
    required = (
        checkout / "hy3dshape" / "hy3dshape" / "pipelines.py",
        checkout / "hy3dshape" / "hy3dshape" / "rembg.py",
        checkout / "hy3dpaint" / "textureGenPipeline.py",
        checkout / "hy3dpaint" / "cfgs" / "hunyuan-paint-pbr.yaml",
    )
    for path in required:
        _regular_file_without_symlinks(path, "Hunyuan checkout component")
    return checkout


def _validate_local_model_layout(model_root: Path = CANONICAL_MODEL_ROOT) -> Path:
    model_root = _real_directory(model_root, "canonical model root")
    for relative in REQUIRED_MODEL_FILES:
        _regular_file_without_symlinks(
            model_root / relative, f"required local model file {relative}"
        )
    model_index = model_root / "hunyuan3d-paintpbr-v2-1" / "model_index.json"
    try:
        parsed = json.loads(model_index.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"paint model_index.json is invalid: {model_index}") from error
    if not isinstance(parsed, dict):
        raise ValueError("paint model_index.json must contain a JSON object")
    return model_root


@contextmanager
def _hunyuan_call_scope() -> Iterator[None]:
    checkout = _validate_hunyuan_checkout()
    original_cwd = Path.cwd()
    original_sys_path = list(sys.path)
    scoped_environment = {
        **OFFLINE_ENVIRONMENT,
        "HY3DGEN_MODELS": str(CANONICAL_MODEL_PARENT),
    }
    original_environment = {
        name: os.environ.get(name, _MISSING) for name in scoped_environment
    }
    try:
        sys.path[:0] = [str(path) for path in HUNYUAN_IMPORT_PATHS]
        os.environ.update(scoped_environment)
        os.chdir(checkout)
        yield
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_sys_path
        for name, old_value in original_environment.items():
            if old_value is _MISSING:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


def smoke_local_configuration(
    model_root: Path = CANONICAL_MODEL_ROOT,
) -> dict[str, str]:
    """Check real import roots and lightweight local configs without loading models."""
    checkout = _validate_hunyuan_checkout()
    model_root = _validate_local_model_layout(model_root)
    with _hunyuan_call_scope():
        shape_spec = importlib.machinery.PathFinder.find_spec(
            "hy3dshape", [str(checkout / "hy3dshape")]
        )
        paint_spec = importlib.machinery.PathFinder.find_spec(
            "textureGenPipeline", [str(checkout / "hy3dpaint")]
        )
        if shape_spec is None or paint_spec is None:
            raise RuntimeError("Hunyuan checkout import roots are incomplete")
    return {
        "checkout": str(checkout),
        "model_root": str(model_root),
        "shape_import": shape_spec.name,
        "paint_import": paint_spec.name,
    }


def load_shape_pipeline() -> Any:
    """Verify all local weights, then load the absolute local shape model offline."""
    verify_canonical_weights()
    _validate_local_model_layout()
    with _hunyuan_call_scope():
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            str(CANONICAL_MODEL_ROOT), local_files_only=True
        )


def remove_background(reference: Path, destination: Path) -> None:
    with _hunyuan_call_scope():
        from hy3dshape.rembg import BackgroundRemover

        with Image.open(reference) as image:
            cutout = BackgroundRemover()(image.convert("RGBA"))
            cutout.save(destination)


def generate_shape(
    pipeline: Any, job: Mapping[str, Any], reference: Path, shape: Path
) -> None:
    with _hunyuan_call_scope():
        import torch

        generator = torch.Generator(device="cuda").manual_seed(job["seed"])
        result = pipeline(
            image=str(reference),
            generator=generator,
            num_inference_steps=job["steps"],
            guidance_scale=job["guidance_scale"],
        )
        mesh = result[0] if isinstance(result, (list, tuple)) else result.meshes[0]
        mesh.export(str(shape))


def run_paint(shape: Path, reference: Path, workdir: Path) -> None:
    _validate_hunyuan_checkout()
    _validate_local_model_layout()
    workdir = _real_directory(workdir, "paint staging directory")
    command = [
        sys.executable,
        str((SCRIPT_DIR / "hy3d_bake_diffuse.py").resolve()),
        "--input-glb",
        str(Path(shape).absolute()),
        "--reference-image",
        str(Path(reference).absolute()),
        "--workdir",
        str(workdir),
        "--realesrgan-ckpt",
        str(CANONICAL_REALESRGAN_CKPT),
        "--dinov2-root",
        str(CANONICAL_DINOV2_ROOT),
        "--weight-manifest",
        str(WEIGHT_ROOT_HASH_MANIFEST),
    ]
    environment = os.environ.copy()
    environment.update(OFFLINE_ENVIRONMENT)
    environment["HY3D_ROOT"] = str(HUNYUAN_CHECKOUT)
    environment["HY3DGEN_MODELS"] = str(CANONICAL_MODEL_PARENT)
    offline_cache = Path(
        tempfile.mkdtemp(prefix=".hf-offline.", dir=workdir)
    )
    for name in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "DIFFUSERS_CACHE",
    ):
        environment[name] = str(offline_cache)
    try:
        subprocess.run(command, check=True, env=environment)
    finally:
        shutil.rmtree(offline_cache, ignore_errors=True)


def build_generation_job(
    review_root: Path, asset_id: str, output_root: Path
) -> dict[str, Any]:
    jobs = assert_generation_ready(review_root)
    if asset_id not in jobs:
        raise ValueError(f"asset_id is outside the approved allowlist: {asset_id!r}")
    asset_dir = _prepare_asset_dir(_prepare_output_root(output_root), asset_id)
    return {
        **jobs[asset_id],
        "weight_manifest_sha256": current_weight_manifest_sha256(),
        "asset_dir": asset_dir,
    }


def _assert_job_current(job: Mapping[str, Any]) -> None:
    try:
        current = assert_generation_ready(Path(job["review_root"]))[job["asset_id"]]
    except (KeyError, TypeError) as error:
        raise Hy3DHumanNotReady("generation job is missing approval fields") from error
    for field in APPROVAL_JOB_FIELDS:
        if field not in job or field not in current or job[field] != current[field]:
            raise Hy3DHumanNotReady(
                f"generation job field {field} changed since approval validation"
            )


def _invalidate_current_manifest(asset_dir: Path) -> None:
    manifest = asset_dir / "hy3d_manifest.json"
    if not os.path.lexists(manifest):
        return
    mode = os.lstat(manifest).st_mode
    if stat.S_ISREG(mode) or stat.S_ISLNK(mode):
        manifest.unlink()
        return
    raise ValueError("existing canonical manifest is non-regular")


def _create_staging_dir(asset_dir: Path) -> Path:
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{asset_dir.name}.", suffix=".staging", dir=asset_dir.parent
        )
    )
    staging = _real_directory(staging, "generation staging directory")
    if staging.parent != asset_dir.parent or list(staging.iterdir()):
        raise ValueError("generation staging directory must be an empty sibling")
    return staging


def _prepare_canonical_destination(path: Path, description: str) -> None:
    if not os.path.lexists(path):
        return
    mode = os.lstat(path).st_mode
    if stat.S_ISLNK(mode):
        path.unlink()
    elif not stat.S_ISREG(mode):
        raise ValueError(f"existing {description} is non-regular")


def _publish_staging(staging: Path, asset_dir: Path) -> Path:
    staged_outputs = {
        label: _regular_file_without_symlinks(
            staging / filename, f"staged {label} output"
        )
        for label, filename in OUTPUT_FILENAMES.items()
    }
    staged_manifest = _regular_file_without_symlinks(
        staging / "hy3d_manifest.json", "staged Hunyuan manifest"
    )

    _invalidate_current_manifest(asset_dir)
    destinations = {
        label: asset_dir / filename for label, filename in OUTPUT_FILENAMES.items()
    }
    for label, destination in destinations.items():
        _prepare_canonical_destination(destination, f"canonical output {label}")

    for label in OUTPUT_FILENAMES:
        os.replace(staged_outputs[label], destinations[label])
    final_manifest = asset_dir / "hy3d_manifest.json"
    os.replace(staged_manifest, final_manifest)
    return final_manifest


def run_job(pipeline: Any, job: Mapping[str, Any]) -> Path:
    asset_dir = _validate_job_asset_dir(job)
    _invalidate_current_manifest(asset_dir)
    staging: Path | None = None
    try:
        _assert_job_current(job)
        verified_weight_manifest = verify_canonical_weights()
        if verified_weight_manifest != job.get("weight_manifest_sha256"):
            raise Hy3DHumanNotReady(
                "weight manifest changed since generation job construction"
            )

        staging = _create_staging_dir(asset_dir)
        reference = staging / OUTPUT_FILENAMES["reference"]
        cutout = staging / OUTPUT_FILENAMES["reference_rembg"]
        shape = staging / OUTPUT_FILENAMES["shape"]
        _copy_atomically(Path(job["candidate_path"]), reference)
        if not hmac.compare_digest(
            _sha256_file(reference), job["candidate_sha256"]
        ):
            raise Hy3DHumanNotReady(
                "copied candidate hash does not match the approved generation job"
            )
        remove_background(reference, cutout)
        generate_shape(pipeline, job, cutout, shape)
        run_paint(shape, cutout, staging)

        _assert_job_current(job)
        outputs = {
            label: staging / filename for label, filename in OUTPUT_FILENAMES.items()
        }
        write_hy3d_manifest(staging, job=job, outputs=outputs)
        return _publish_staging(staging, asset_dir)
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--asset-id", required=True, choices=EXPECTED_ASSET_IDS)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    asset_dir = _prepare_asset_dir(
        _prepare_output_root(args.output_root), args.asset_id
    )
    _invalidate_current_manifest(asset_dir)
    job = build_generation_job(args.review_root, args.asset_id, args.output_root)
    if Path(job["asset_dir"]) != asset_dir:
        raise ValueError("generation job asset directory changed during construction")
    pipeline = load_shape_pipeline()
    run_job(pipeline, job)


if __name__ == "__main__":
    main()
