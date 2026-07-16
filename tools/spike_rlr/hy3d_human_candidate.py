"""Hash-locked provenance for approved human references sent to Hunyuan3D."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping

from human_reference_review import (
    EXPECTED_ASSET_IDS,
    HumanReferenceNotApproved,
    assert_pair_approved,
    validated_candidate_snapshot,
)


CANONICAL_MODEL_PARENT = Path("/data/models/hunyuan3d-2.1")
CANONICAL_MODEL_ROOT = CANONICAL_MODEL_PARENT / "hunyuan3d-2.1"
WEIGHT_ROOT_HASH_MANIFEST = CANONICAL_MODEL_PARENT / "weights.sha256"
CANONICAL_REALESRGAN_CKPT = (
    CANONICAL_MODEL_ROOT
    / "dependencies"
    / "realesrgan"
    / "RealESRGAN_x4plus.pth"
)
CANONICAL_DINOV2_ROOT = CANONICAL_MODEL_ROOT / "dependencies" / "dinov2-giant"
HUNYUAN_CHECKOUT = Path("/data/jzy/code/AVEngine/external/Hunyuan3D-2.1")
SPEAR_HY3D_BAKE_DIFFUSE = Path(
    "/data/jzy/code/AVEngine/external/SPEAR/tools/hy3d_bake_diffuse.py"
)
ASSET_SEEDS = {
    "rocketbox_male_adult_01": 4101,
    "rocketbox_female_adult_01": 7301,
}
INFERENCE_STEPS = 50
GUIDANCE_SCALE = 5.0
SCHEMA_VERSION = "hy3d_human_candidate_v1"
USAGE_SCOPE = "technical_spike_only"
OUTPUT_FILENAMES = {
    "reference": "reference.png",
    "reference_rembg": "reference_rembg.png",
    "shape": "shape.glb",
    "paint_obj": "hy3d_textured.obj",
    "diffuse": "hy3d_diffuse.jpg",
    "metallic": "hy3d_metallic.jpg",
    "roughness": "hy3d_roughness.jpg",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GIT_HEAD_RE = re.compile(r"[0-9a-f]{40}")
_SHA256SUM_LINE_RE = re.compile(r"([0-9a-f]{64}) ([ *])(.+)")
_RUNTIME_SUFFIXES = frozenset(
    {".py", ".yaml", ".yml", ".json", ".so", ".cpp", ".cu", ".h", ".c"}
)
_RUNTIME_TREE_ROOTS = (
    "hy3dshape/hy3dshape",
    "hy3dpaint/DifferentiableRenderer",
    "hy3dpaint/utils",
    "hy3dpaint/hunyuanpaintpbr",
    "hy3dpaint/custom_rasterizer",
)
_RUNTIME_FILES = (
    "hy3dpaint/textureGenPipeline.py",
    "hy3dpaint/convert_utils.py",
    "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml",
)
_REQUIRED_RUNTIME_PATCHES = frozenset(
    {
        "hy3dpaint/DifferentiableRenderer/mesh_utils.py",
        "hy3dpaint/utils/multiview_utils.py",
        "hy3dpaint/utils/simplify_mesh_utils.py",
    }
)
REQUIRED_MODEL_FILES = frozenset(
    {
        "hunyuan3d-dit-v2-1/config.yaml",
        "hunyuan3d-dit-v2-1/model.fp16.ckpt",
        "hunyuan3d-paintpbr-v2-1/model_index.json",
        "hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin",
        "hunyuan3d-vae-v2-1/config.yaml",
        "hunyuan3d-vae-v2-1/model.fp16.ckpt",
        "dependencies/realesrgan/RealESRGAN_x4plus.pth",
        "dependencies/dinov2-giant/config.json",
        "dependencies/dinov2-giant/preprocessor_config.json",
        "dependencies/dinov2-giant/model.safetensors",
    }
)


class Hy3DHumanNotReady(RuntimeError):
    """Raised when the approved reference pair cannot safely be generated."""


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


def _regular_file_directly_under(path: Path, root: Path, description: str) -> Path:
    absolute = _absolute_without_symlinks(path, description)
    root = _real_directory(root, "asset directory")
    if not os.path.lexists(absolute) or not stat.S_ISREG(os.lstat(absolute).st_mode):
        raise ValueError(f"{description} must be a regular file directly under asset directory")
    if absolute.resolve() != absolute or absolute.parent != root:
        raise ValueError(f"{description} must be directly under asset directory containment")
    return absolute


def _regular_file_without_symlinks(path: Path, description: str) -> Path:
    absolute = _absolute_without_symlinks(path, description)
    if not os.path.lexists(absolute) or not stat.S_ISREG(os.lstat(absolute).st_mode):
        raise ValueError(f"{description} must be a regular file")
    if absolute.resolve() != absolute:
        raise ValueError(f"{description} resolved path is not exact")
    return absolute


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _require_hash(value: Any, description: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{description} must be a 64-character lowercase hex value")
    return value


def _read_git_head(checkout: Path) -> str:
    git_dir = checkout / ".git"
    if git_dir.is_file():
        marker = _regular_file_without_symlinks(git_dir, "Hunyuan gitdir marker")
        marker_text = marker.read_text(encoding="utf-8").strip()
        if not marker_text.startswith("gitdir: "):
            raise ValueError("Hunyuan .git file is not a gitdir marker")
        configured = Path(marker_text.removeprefix("gitdir: "))
        git_dir = configured if configured.is_absolute() else checkout / configured
    git_dir = _real_directory(git_dir, "Hunyuan git metadata")
    head_path = _regular_file_without_symlinks(git_dir / "HEAD", "Hunyuan git HEAD")
    head_text = head_path.read_text(encoding="ascii").strip()
    if _GIT_HEAD_RE.fullmatch(head_text):
        return head_text
    if not head_text.startswith("ref: "):
        raise ValueError("Hunyuan git HEAD is invalid")
    reference = Path(head_text.removeprefix("ref: "))
    if reference.is_absolute() or ".." in reference.parts:
        raise ValueError("Hunyuan git HEAD reference escaped git metadata")
    reference_path = git_dir / reference
    if os.path.lexists(reference_path):
        revision = _regular_file_without_symlinks(
            reference_path, "Hunyuan git HEAD reference"
        ).read_text(encoding="ascii").strip()
    else:
        packed_refs = _regular_file_without_symlinks(
            git_dir / "packed-refs", "Hunyuan packed refs"
        )
        revisions = {
            name: revision
            for line in packed_refs.read_text(encoding="ascii").splitlines()
            if line and not line.startswith(("#", "^"))
            for revision, name in (line.split(" ", 1),)
        }
        revision = revisions.get(reference.as_posix(), "")
    if _GIT_HEAD_RE.fullmatch(revision) is None:
        raise ValueError("Hunyuan git HEAD revision is invalid")
    return revision


def _stable_file_record(path: Path, label: str) -> dict[str, Any]:
    path = _regular_file_without_symlinks(path, f"Hunyuan runtime file {label}")
    before = path.stat()
    sha256 = _sha256_file(path)
    after = path.stat()
    before_state = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_state = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_state != after_state:
        raise ValueError(f"Hunyuan runtime file changed while hashing: {label}")
    return {"path": label, "sha256": sha256, "size_bytes": after.st_size}


def _runtime_paths(checkout: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for relative_root in _RUNTIME_TREE_ROOTS:
        root = _real_directory(checkout / relative_root, "Hunyuan runtime root")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"Hunyuan runtime path must not be a symlink: {path}")
            if not path.is_file() or path.suffix not in _RUNTIME_SUFFIXES:
                continue
            relative = path.relative_to(checkout).as_posix()
            paths[relative] = path
    for relative in _RUNTIME_FILES:
        paths[relative] = checkout / relative
    missing_patches = _REQUIRED_RUNTIME_PATCHES.difference(paths)
    if missing_patches:
        raise ValueError(
            "Hunyuan runtime fingerprint is missing required patched files: "
            + ", ".join(sorted(missing_patches))
        )
    return paths


def _current_hunyuan_runtime_provenance(
    checkout: Path, wrapper_path: Path
) -> dict[str, Any]:
    checkout = _real_directory(checkout, "Hunyuan checkout")
    records = [
        _stable_file_record(path, relative)
        for relative, path in sorted(_runtime_paths(checkout).items())
    ]
    records.append(
        _stable_file_record(
            wrapper_path, "external/SPEAR/tools/hy3d_bake_diffuse.py"
        )
    )
    payload = {
        "schema_version": "hy3d_runtime_fingerprint_v1",
        "git_head": _read_git_head(checkout),
        "files": records,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "git_head": payload["git_head"],
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
        "file_count": len(records),
    }


def current_hunyuan_runtime_provenance(
    checkout: Path = HUNYUAN_CHECKOUT,
    wrapper_path: Path = SPEAR_HY3D_BAKE_DIFFUSE,
) -> dict[str, Any]:
    """Fingerprint the actual checkout bytes, dirty patches, binaries, and wrapper."""
    try:
        return _current_hunyuan_runtime_provenance(
            Path(checkout), Path(wrapper_path)
        )
    except Hy3DHumanNotReady:
        raise
    except (OSError, UnicodeError, ValueError) as error:
        raise Hy3DHumanNotReady(str(error)) from error


def _verify_weight_manifest(model_root: Path, manifest_path: Path) -> str:
    model_root = _real_directory(model_root, "canonical model root")
    manifest_path = _regular_file_without_symlinks(
        manifest_path, "weight SHA-256 manifest"
    )
    manifest_bytes = manifest_path.read_bytes()
    try:
        lines = manifest_bytes.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("weight SHA-256 manifest must be UTF-8 text") from error
    if not lines:
        raise ValueError("weight SHA-256 manifest must not be empty")

    verified_paths: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        match = _SHA256SUM_LINE_RE.fullmatch(line)
        if match is None:
            raise ValueError(
                f"weight SHA-256 manifest line {line_number} is not sha256sum format"
            )
        expected_sha256, _, raw_relative = match.groups()
        if "\\" in raw_relative:
            raise ValueError("weight path must use an unescaped relative path")
        relative = Path(raw_relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("weight path violates model-root containment")
        normalized = relative.as_posix()
        if normalized.startswith("./"):
            normalized = normalized[2:]
        if not normalized or normalized in verified_paths:
            raise ValueError("weight manifest paths must be unique relative files")
        verified_paths.add(normalized)

        weight_path = _regular_file_without_symlinks(
            model_root / relative, f"weight file {normalized}"
        )
        try:
            weight_path.relative_to(model_root)
        except ValueError as error:
            raise ValueError("weight path violates model-root containment") from error
        if weight_path.stat().st_size <= 0:
            raise ValueError(f"weight file is empty: {normalized}")
        actual_sha256 = _sha256_file(weight_path)
        if not hmac.compare_digest(expected_sha256, actual_sha256):
            raise ValueError(f"weight file SHA-256 hash mismatch: {normalized}")

    physical_paths: set[str] = set()
    pending_directories = [model_root]
    while pending_directories:
        directory = pending_directories.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                entry_path = Path(entry.path)
                relative = entry_path.relative_to(model_root).as_posix()
                mode = entry.stat(follow_symlinks=False).st_mode
                if stat.S_ISLNK(mode):
                    raise ValueError(f"model tree contains a symlink: {relative}")
                if stat.S_ISDIR(mode):
                    pending_directories.append(entry_path)
                    continue
                if not stat.S_ISREG(mode):
                    raise ValueError(
                        f"model tree contains a non-regular entry: {relative}"
                    )
                if entry.stat(follow_symlinks=False).st_size <= 0:
                    raise ValueError(f"weight file is empty: {relative}")
                physical_paths.add(relative)

    unlisted = physical_paths.difference(verified_paths)
    if unlisted:
        raise ValueError(
            "model tree contains files unlisted by weight manifest: "
            + ", ".join(sorted(unlisted))
        )
    absent = verified_paths.difference(physical_paths)
    if absent:
        raise ValueError(
            "weight manifest lists files absent from model tree: "
            + ", ".join(sorted(absent))
        )
    missing = REQUIRED_MODEL_FILES.difference(verified_paths)
    if missing:
        raise ValueError(
            "weight manifest is missing required model files: "
            + ", ".join(sorted(missing))
        )
    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("weight SHA-256 manifest changed during verification")
    return hashlib.sha256(manifest_bytes).hexdigest()


def verify_weight_manifest(model_root: Path, manifest_path: Path) -> str:
    """Verify every sha256sum entry and required local Hunyuan model file."""
    try:
        return _verify_weight_manifest(Path(model_root), Path(manifest_path))
    except Hy3DHumanNotReady:
        raise
    except (OSError, ValueError) as error:
        raise Hy3DHumanNotReady(str(error)) from error


def verify_canonical_weights() -> str:
    """Verify the complete canonical weight tree before any generation starts."""
    return verify_weight_manifest(CANONICAL_MODEL_ROOT, WEIGHT_ROOT_HASH_MANIFEST)


def current_weight_manifest_sha256() -> str:
    """Hash the canonical text manifest without re-reading the large weights."""
    try:
        manifest = _regular_file_without_symlinks(
            WEIGHT_ROOT_HASH_MANIFEST, "weight SHA-256 manifest"
        )
        return _sha256_file(manifest)
    except (OSError, ValueError) as error:
        raise Hy3DHumanNotReady(str(error)) from error


def assert_generation_ready(review_root: Path) -> dict[str, dict[str, Any]]:
    """Return immutable generation jobs for the exact currently approved pair."""
    try:
        review_root = _real_directory(Path(review_root), "review root")
        approvals = assert_pair_approved(review_root)
        runtime = current_hunyuan_runtime_provenance()
        jobs: dict[str, dict[str, Any]] = {}
        for asset_id in EXPECTED_ASSET_IDS:
            candidate_dir = _real_directory(Path(review_root) / asset_id, "candidate directory")
            manifest, images, snapshot = validated_candidate_snapshot(candidate_dir)
            candidate_path = _regular_file_directly_under(
                images["candidate"], candidate_dir, "approved candidate image"
            )
            review_path = _regular_file_directly_under(
                candidate_dir / "reference_review.json", candidate_dir, "reference review"
            )
            approval = approvals[asset_id]
            if approval.get("candidate_manifest_sha256") != snapshot["candidate_manifest_sha256"]:
                raise Hy3DHumanNotReady(f"{asset_id} reference review manifest hash is stale")
            if approval.get("candidate_sha256") != snapshot["candidate_sha256"]:
                raise Hy3DHumanNotReady(f"{asset_id} reference review candidate hash is stale")
            source_approval_sha256 = _require_hash(
                manifest.get("source_approval_sha256"), "source_approval_sha256"
            )
            jobs[asset_id] = {
                "asset_id": asset_id,
                "review_root": review_root,
                "candidate_path": candidate_path,
                "candidate_sha256": snapshot["candidate_sha256"],
                "candidate_manifest_sha256": snapshot[
                    "candidate_manifest_sha256"
                ],
                "source_sha256": snapshot["source_sha256"],
                "source_approval_sha256": source_approval_sha256,
                "reference_review_sha256": _sha256_file(review_path),
                "seed": ASSET_SEEDS[asset_id],
                "steps": INFERENCE_STEPS,
                "guidance_scale": GUIDANCE_SCALE,
                "model_root": CANONICAL_MODEL_ROOT,
                "weight_root_hash_manifest": WEIGHT_ROOT_HASH_MANIFEST,
                "hunyuan_runtime_git_head": runtime["git_head"],
                "hunyuan_runtime_fingerprint": runtime["fingerprint"],
                "hunyuan_runtime_file_count": runtime["file_count"],
            }
        return jobs
    except (HumanReferenceNotApproved, ValueError, KeyError) as error:
        raise Hy3DHumanNotReady(str(error)) from error


def write_hy3d_manifest(
    asset_dir: Path, *, job: Mapping[str, Any], outputs: Mapping[str, Path]
) -> Path:
    """Atomically publish provenance only after all canonical outputs exist."""
    asset_dir = _real_directory(asset_dir, "asset directory")
    asset_id = job.get("asset_id")
    if asset_id not in EXPECTED_ASSET_IDS:
        raise ValueError("asset_id must be in the exact approved asset allowlist")
    if job.get("seed") != ASSET_SEEDS[asset_id] or job.get("steps") != INFERENCE_STEPS:
        raise ValueError("generation job must use the fixed seed and 50 steps")
    if job.get("guidance_scale") != GUIDANCE_SCALE:
        raise ValueError("generation job must use guidance_scale=5")
    if job.get("model_root") != CANONICAL_MODEL_ROOT:
        raise ValueError("generation job must use the canonical model root")
    if job.get("weight_root_hash_manifest") != WEIGHT_ROOT_HASH_MANIFEST:
        raise ValueError("generation job must use the canonical weight manifest")
    expected_runtime_git_head = job.get("hunyuan_runtime_git_head")
    if (
        not isinstance(expected_runtime_git_head, str)
        or _GIT_HEAD_RE.fullmatch(expected_runtime_git_head) is None
    ):
        raise ValueError("hunyuan_runtime_git_head must be a 40-character git revision")
    expected_runtime_fingerprint = _require_hash(
        job.get("hunyuan_runtime_fingerprint"), "hunyuan_runtime_fingerprint"
    )
    expected_runtime_file_count = job.get("hunyuan_runtime_file_count")
    if (
        not isinstance(expected_runtime_file_count, int)
        or isinstance(expected_runtime_file_count, bool)
        or expected_runtime_file_count <= 0
    ):
        raise ValueError("hunyuan_runtime_file_count must be a positive integer")
    current_runtime = current_hunyuan_runtime_provenance()
    if current_runtime != {
        "git_head": expected_runtime_git_head,
        "fingerprint": expected_runtime_fingerprint,
        "file_count": expected_runtime_file_count,
    }:
        raise ValueError("Hunyuan runtime fingerprint changed before publication")
    expected_weight_manifest_sha256 = _require_hash(
        job.get("weight_manifest_sha256"), "weight_manifest_sha256"
    )
    current_weight_manifest = _regular_file_without_symlinks(
        WEIGHT_ROOT_HASH_MANIFEST, "weight SHA-256 manifest"
    )
    if not hmac.compare_digest(
        expected_weight_manifest_sha256, _sha256_file(current_weight_manifest)
    ):
        raise ValueError("weight manifest SHA-256 changed before publication")
    if set(outputs) != set(OUTPUT_FILENAMES):
        raise ValueError("outputs must contain exactly the canonical output allowlist")
    output_records: dict[str, dict[str, Any]] = {}
    for label, filename in OUTPUT_FILENAMES.items():
        path = _regular_file_directly_under(outputs[label], asset_dir, f"{label} output")
        if path.name != filename:
            raise ValueError(f"{label} output must use canonical filename {filename}")
        if path.stat().st_size <= 0:
            raise ValueError(f"{label} output must not be empty")
        output_records[label] = {
            "path": filename,
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "asset_id": asset_id,
        "candidate_sha256": _require_hash(job.get("candidate_sha256"), "candidate_sha256"),
        "candidate_manifest_sha256": _require_hash(
            job.get("candidate_manifest_sha256"), "candidate_manifest_sha256"
        ),
        "source_sha256": _require_hash(job.get("source_sha256"), "source_sha256"),
        "source_approval_sha256": _require_hash(
            job.get("source_approval_sha256"), "source_approval_sha256"
        ),
        "reference_review_sha256": _require_hash(
            job.get("reference_review_sha256"), "reference_review_sha256"
        ),
        "hunyuan_code_revision": expected_runtime_git_head,
        "hunyuan_runtime_git_head": expected_runtime_git_head,
        "hunyuan_runtime_fingerprint": expected_runtime_fingerprint,
        "hunyuan_runtime_file_count": expected_runtime_file_count,
        "weight_root": str(CANONICAL_MODEL_ROOT),
        "weight_root_hash_manifest": str(WEIGHT_ROOT_HASH_MANIFEST),
        "weight_manifest_sha256": expected_weight_manifest_sha256,
        "seed": ASSET_SEEDS[asset_id],
        "steps": INFERENCE_STEPS,
        "guidance_scale": job["guidance_scale"],
        "usage_scope": USAGE_SCOPE,
        "outputs": output_records,
    }
    return _atomic_write_json(asset_dir / "hy3d_manifest.json", payload)
