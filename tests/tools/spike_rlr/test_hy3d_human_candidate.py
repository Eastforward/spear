"""Tests for the approved human-reference to Hunyuan3D provenance contract."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
CONTRACT_DIR = REPO / "tools" / "spike_rlr"
if str(CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTRACT_DIR))

from human_reference_review import (  # noqa: E402
    EXPECTED_ASSET_IDS,
    record_review,
    write_candidate_manifest,
)
import hy3d_human_candidate as contract  # noqa: E402
from hy3d_human_candidate import (  # noqa: E402
    ASSET_SEEDS,
    CANONICAL_MODEL_ROOT,
    INFERENCE_STEPS,
    Hy3DHumanNotReady,
    assert_generation_ready,
    write_hy3d_manifest,
)


SOURCE_APPROVAL_SHA256 = "a" * 64
DEPENDENCY_FILES = frozenset(
    {
        "dependencies/realesrgan/RealESRGAN_x4plus.pth",
        "dependencies/dinov2-giant/config.json",
        "dependencies/dinov2-giant/preprocessor_config.json",
        "dependencies/dinov2-giant/model.safetensors",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(candidate_dir: Path) -> dict[str, str]:
    return {
        "candidate_manifest_sha256": _sha256(candidate_dir / "candidate_manifest.json"),
        "source_sha256": _sha256(candidate_dir / "source.png"),
        "candidate_sha256": _sha256(candidate_dir / "candidate.png"),
    }


def _approved_review_root(root: Path) -> Path:
    for asset_id in EXPECTED_ASSET_IDS:
        candidate_dir = root / asset_id
        candidate_dir.mkdir(parents=True)
        (candidate_dir / "source.png").write_bytes(f"{asset_id}:source".encode())
        (candidate_dir / "candidate.png").write_bytes(
            f"{asset_id}:candidate".encode()
        )
        write_candidate_manifest(
            candidate_dir,
            asset_id=asset_id,
            model_revision="e7b7dc27f91deacad38e78976d1f2b499d76a294",
            prompt=f"Exact prompt for {asset_id}.",
            seed=4242,
            width=1024,
            height=1536,
            steps=28,
            guidance_scale=4.0,
            source_approval_sha256=SOURCE_APPROVAL_SHA256,
        )
        record_review(
            candidate_dir,
            "approved",
            "reviewer",
            "ready",
            expected_snapshot=_snapshot(candidate_dir),
        )
    return root


def _write_outputs(asset_dir: Path) -> dict[str, Path]:
    outputs = {
        "reference": asset_dir / "reference.png",
        "reference_rembg": asset_dir / "reference_rembg.png",
        "shape": asset_dir / "shape.glb",
        "paint_obj": asset_dir / "hy3d_textured.obj",
        "diffuse": asset_dir / "hy3d_diffuse.jpg",
        "metallic": asset_dir / "hy3d_metallic.jpg",
        "roughness": asset_dir / "hy3d_roughness.jpg",
    }
    for label, path in outputs.items():
        path.write_bytes(f"{label}-bytes".encode())
    return outputs


def _weight_fixture(root: Path) -> tuple[Path, Path]:
    model_root = root / "hunyuan3d-2.1"
    entries = {
        "hunyuan3d-dit-v2-1/config.yaml": b"model: {}\n",
        "hunyuan3d-dit-v2-1/model.fp16.ckpt": b"shape-weights",
        "hunyuan3d-paintpbr-v2-1/model_index.json": b"{}\n",
        "hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin": b"paint-weights",
        "hunyuan3d-vae-v2-1/config.yaml": b"vae: {}\n",
        "hunyuan3d-vae-v2-1/model.fp16.ckpt": b"vae-weights",
        "dependencies/realesrgan/RealESRGAN_x4plus.pth": b"realesrgan-weights",
        "dependencies/dinov2-giant/config.json": b"{}\n",
        "dependencies/dinov2-giant/preprocessor_config.json": b"{}\n",
        "dependencies/dinov2-giant/model.safetensors": b"dinov2-weights",
    }
    lines = []
    for relative, content in entries.items():
        path = model_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        lines.append(f"{_sha256(path)}  ./{relative}\n")
    manifest_path = root / "weights.sha256"
    manifest_path.write_text("".join(lines), encoding="utf-8")
    return model_root, manifest_path


def _job_for_manifest(tmp_path: Path, monkeypatch) -> tuple[dict, Path, Path]:
    review_root = _approved_review_root(tmp_path / "reviews")
    job = assert_generation_ready(review_root)["rocketbox_male_adult_01"]
    _, weight_manifest = _weight_fixture(tmp_path / "weights")
    monkeypatch.setattr(contract, "WEIGHT_ROOT_HASH_MANIFEST", weight_manifest)
    job["weight_root_hash_manifest"] = weight_manifest
    job["weight_manifest_sha256"] = _sha256(weight_manifest)
    staging_dir = tmp_path / "out" / ".rocketbox_male_adult_01.staging"
    staging_dir.mkdir(parents=True)
    return job, weight_manifest, staging_dir


def _runtime_fixture(root: Path) -> tuple[Path, Path]:
    checkout = root / "Hunyuan3D-2.1"
    (checkout / ".git").mkdir(parents=True)
    (checkout / ".git" / "HEAD").write_text("1" * 40 + "\n", encoding="ascii")
    runtime_files = {
        "hy3dshape/hy3dshape/pipelines.py": b"shape pipeline\n",
        "hy3dshape/hy3dshape/rembg.py": b"background remover\n",
        "hy3dpaint/textureGenPipeline.py": b"paint pipeline\n",
        "hy3dpaint/convert_utils.py": b"conversion helpers\n",
        "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml": b"model: local\n",
        "hy3dpaint/hunyuanpaintpbr/pipeline.py": b"custom pipeline\n",
        "hy3dpaint/DifferentiableRenderer/mesh_utils.py": b"bpy fallback patch\n",
        "hy3dpaint/utils/multiview_utils.py": b"local weights patch\n",
        "hy3dpaint/utils/simplify_mesh_utils.py": b"trimesh patch\n",
        "hy3dpaint/custom_rasterizer/custom_rasterizer_kernel.so": b"binary\n",
    }
    for relative, content in runtime_files.items():
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    wrapper = root / "hy3d_bake_diffuse.py"
    wrapper.write_bytes(b"paint wrapper\n")
    return checkout, wrapper


def test_generation_requires_the_exact_approved_pair_and_pins_hashes(tmp_path):
    review_root = _approved_review_root(tmp_path / "reviews")

    jobs = assert_generation_ready(review_root)

    assert set(jobs) == set(EXPECTED_ASSET_IDS)
    for asset_id, job in jobs.items():
        candidate_dir = review_root / asset_id
        assert job["asset_id"] == asset_id
        assert job["candidate_path"] == candidate_dir / "candidate.png"
        assert job["candidate_sha256"] == _sha256(candidate_dir / "candidate.png")
        assert job["candidate_manifest_sha256"] == _sha256(
            candidate_dir / "candidate_manifest.json"
        )
        assert job["reference_review_sha256"] == _sha256(
            candidate_dir / "reference_review.json"
        )
        assert job["source_approval_sha256"] == SOURCE_APPROVAL_SHA256
        assert job["seed"] == ASSET_SEEDS[asset_id]
        assert job["steps"] == INFERENCE_STEPS == 50
        assert job["model_root"] == CANONICAL_MODEL_ROOT
        assert len(job["hunyuan_runtime_fingerprint"]) == 64
        assert len(job["hunyuan_runtime_git_head"]) == 40
        assert job["hunyuan_runtime_file_count"] > 0


@pytest.mark.parametrize(
    "tracked_patch",
    (
        "hy3dpaint/DifferentiableRenderer/mesh_utils.py",
        "hy3dpaint/utils/multiview_utils.py",
        "hy3dpaint/utils/simplify_mesh_utils.py",
    ),
)
def test_runtime_fingerprint_covers_each_required_tracked_patch(
    tmp_path, tracked_patch
):
    checkout, wrapper = _runtime_fixture(tmp_path)
    before = contract.current_hunyuan_runtime_provenance(checkout, wrapper)

    (checkout / tracked_patch).write_bytes(b"changed tracked patch\n")
    after = contract.current_hunyuan_runtime_provenance(checkout, wrapper)

    assert before["git_head"] == after["git_head"] == "1" * 40
    assert before["fingerprint"] != after["fingerprint"]


def test_runtime_fingerprint_covers_the_spear_paint_wrapper(tmp_path):
    checkout, wrapper = _runtime_fixture(tmp_path)
    before = contract.current_hunyuan_runtime_provenance(checkout, wrapper)

    wrapper.write_bytes(b"changed paint wrapper\n")
    after = contract.current_hunyuan_runtime_provenance(checkout, wrapper)

    assert before["fingerprint"] != after["fingerprint"]


@pytest.mark.parametrize(
    "runtime_file",
    (
        "hy3dshape/hy3dshape/pipelines.py",
        "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml",
        "hy3dpaint/custom_rasterizer/custom_rasterizer_kernel.so",
    ),
)
def test_runtime_fingerprint_covers_shape_source_config_and_binary(
    tmp_path, runtime_file
):
    checkout, wrapper = _runtime_fixture(tmp_path)
    before = contract.current_hunyuan_runtime_provenance(checkout, wrapper)

    (checkout / runtime_file).write_bytes(b"changed runtime bytes\n")
    after = contract.current_hunyuan_runtime_provenance(checkout, wrapper)

    assert before["fingerprint"] != after["fingerprint"]


def test_runtime_fingerprint_covers_current_git_head(tmp_path):
    checkout, wrapper = _runtime_fixture(tmp_path)
    before = contract.current_hunyuan_runtime_provenance(checkout, wrapper)

    (checkout / ".git/HEAD").write_text("2" * 40 + "\n", encoding="ascii")
    after = contract.current_hunyuan_runtime_provenance(checkout, wrapper)

    assert before["git_head"] == "1" * 40
    assert after["git_head"] == "2" * 40
    assert before["fingerprint"] != after["fingerprint"]


def test_runtime_fingerprint_rejects_symlinked_runtime_source(tmp_path):
    checkout, wrapper = _runtime_fixture(tmp_path)
    runtime_file = checkout / "hy3dpaint/utils/multiview_utils.py"
    external = tmp_path / "external-runtime.py"
    external.write_bytes(runtime_file.read_bytes())
    runtime_file.unlink()
    runtime_file.symlink_to(external)

    with pytest.raises(Hy3DHumanNotReady, match="symlink|runtime"):
        contract.current_hunyuan_runtime_provenance(checkout, wrapper)


def test_generation_rejects_a_stale_reference_review(tmp_path):
    review_root = _approved_review_root(tmp_path / "reviews")
    candidate_dir = review_root / "rocketbox_male_adult_01"
    (candidate_dir / "candidate.png").write_bytes(b"changed")

    with pytest.raises(Hy3DHumanNotReady, match="stale|hash"):
        assert_generation_ready(review_root)


def test_generation_rejects_review_root_or_candidate_paths_that_use_symlinks(tmp_path):
    review_root = _approved_review_root(tmp_path / "reviews")
    linked_root = tmp_path / "linked-reviews"
    linked_root.symlink_to(review_root, target_is_directory=True)

    with pytest.raises(Hy3DHumanNotReady, match="symlink"):
        assert_generation_ready(linked_root)

    outside = tmp_path / "outside"
    outside.mkdir()
    target = review_root / "rocketbox_female_adult_01" / "candidate.png"
    outside_candidate = outside / "candidate.png"
    outside_candidate.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside_candidate)

    with pytest.raises(Hy3DHumanNotReady, match="regular file|symlink|asset root"):
        assert_generation_ready(review_root)


def test_asset_seeds_are_fixed_per_approved_identity():
    assert ASSET_SEEDS == {
        "rocketbox_male_adult_01": 4101,
        "rocketbox_female_adult_01": 7301,
    }


def test_canonical_paint_dependencies_are_inside_the_weight_root():
    assert contract.CANONICAL_REALESRGAN_CKPT == (
        CANONICAL_MODEL_ROOT
        / "dependencies"
        / "realesrgan"
        / "RealESRGAN_x4plus.pth"
    )
    assert contract.CANONICAL_DINOV2_ROOT == (
        CANONICAL_MODEL_ROOT / "dependencies" / "dinov2-giant"
    )
    assert DEPENDENCY_FILES <= contract.REQUIRED_MODEL_FILES


@pytest.mark.parametrize("missing_dependency", DEPENDENCY_FILES)
def test_weight_manifest_requires_every_canonical_paint_dependency(
    tmp_path, missing_dependency
):
    model_root, manifest_path = _weight_fixture(tmp_path / "weights")
    lines = [
        line
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if not line.endswith(f"./{missing_dependency}")
    ]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(Hy3DHumanNotReady, match="required|dependencies"):
        contract.verify_weight_manifest(model_root, manifest_path)


def test_weight_manifest_rejects_modified_dinov2_bytes(tmp_path):
    model_root, manifest_path = _weight_fixture(tmp_path / "weights")
    (model_root / "dependencies/dinov2-giant/model.safetensors").write_bytes(
        b"modified-dinov2"
    )

    with pytest.raises(Hy3DHumanNotReady, match="SHA-256|hash"):
        contract.verify_weight_manifest(model_root, manifest_path)


def test_weight_manifest_rejects_symlinked_realesrgan_dependency(tmp_path):
    model_root, manifest_path = _weight_fixture(tmp_path / "weights")
    dependency = model_root / "dependencies/realesrgan/RealESRGAN_x4plus.pth"
    external = tmp_path / "external-realesrgan.pth"
    external.write_bytes(dependency.read_bytes())
    dependency.unlink()
    dependency.symlink_to(external)

    with pytest.raises(Hy3DHumanNotReady, match="symlink|regular file"):
        contract.verify_weight_manifest(model_root, manifest_path)


def test_weight_manifest_rejects_dependency_path_escape(tmp_path):
    model_root, manifest_path = _weight_fixture(tmp_path / "weights")
    dependency = "dependencies/dinov2-giant/config.json"
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    escaped = [
        f"{line.split()[0]}  ../outside-config.json"
        if line.endswith(f"./{dependency}")
        else line
        for line in lines
    ]
    manifest_path.write_text("\n".join(escaped) + "\n", encoding="utf-8")

    with pytest.raises(Hy3DHumanNotReady, match="containment|traversal"):
        contract.verify_weight_manifest(model_root, manifest_path)


def test_weight_manifest_verifies_contained_nonempty_files_and_required_models(
    tmp_path,
):
    model_root, manifest_path = _weight_fixture(tmp_path / "weights")

    manifest_sha256 = contract.verify_weight_manifest(model_root, manifest_path)

    assert manifest_sha256 == _sha256(manifest_path)


def test_weight_manifest_rejects_a_hash_mismatch(tmp_path):
    model_root, manifest_path = _weight_fixture(tmp_path / "weights")
    (model_root / "hunyuan3d-dit-v2-1/model.fp16.ckpt").write_bytes(b"changed")

    with pytest.raises(Hy3DHumanNotReady, match="SHA-256|hash"):
        contract.verify_weight_manifest(model_root, manifest_path)


def test_weight_manifest_rejects_zero_length_files(tmp_path):
    model_root, manifest_path = _weight_fixture(tmp_path / "weights")
    target = model_root / "hunyuan3d-vae-v2-1/model.fp16.ckpt"
    target.write_bytes(b"")
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    lines = [
        f"{_sha256(target)}  ./hunyuan3d-vae-v2-1/model.fp16.ckpt"
        if line.endswith("./hunyuan3d-vae-v2-1/model.fp16.ckpt")
        else line
        for line in lines
    ]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(Hy3DHumanNotReady, match="empty|size"):
        contract.verify_weight_manifest(model_root, manifest_path)


def test_weight_manifest_rejects_parent_traversal(tmp_path):
    model_root, manifest_path = _weight_fixture(tmp_path / "weights")
    outside = model_root.parent / "outside.ckpt"
    outside.write_bytes(b"outside")
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8")
        + f"{_sha256(outside)}  ../outside.ckpt\n",
        encoding="utf-8",
    )

    with pytest.raises(Hy3DHumanNotReady, match="containment|relative|traversal"):
        contract.verify_weight_manifest(model_root, manifest_path)


def test_weight_manifest_rejects_symlinked_entries(tmp_path):
    model_root, manifest_path = _weight_fixture(tmp_path / "weights")
    target = model_root / "hunyuan3d-dit-v2-1/model.fp16.ckpt"
    external = tmp_path / "external.ckpt"
    external.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(external)

    with pytest.raises(Hy3DHumanNotReady, match="symlink|regular file"):
        contract.verify_weight_manifest(model_root, manifest_path)


def test_weight_manifest_rejects_unlisted_alternate_weight(tmp_path):
    model_root, manifest_path = _weight_fixture(tmp_path / "weights")
    alternate = model_root / "hunyuan3d-dit-v2-1/model.safetensors"
    alternate.write_bytes(b"unreviewed alternate weights")

    with pytest.raises(Hy3DHumanNotReady, match="unlisted|manifest|extra"):
        contract.verify_weight_manifest(model_root, manifest_path)


def test_weight_manifest_rejects_unlisted_symlink_in_model_tree(tmp_path):
    model_root, manifest_path = _weight_fixture(tmp_path / "weights")
    outside = tmp_path / "outside-alternate.ckpt"
    outside.write_bytes(b"outside")
    (model_root / "alternate.ckpt").symlink_to(outside)

    with pytest.raises(Hy3DHumanNotReady, match="symlink|non-regular"):
        contract.verify_weight_manifest(model_root, manifest_path)


def test_weight_manifest_rejects_unlisted_nonregular_entry(tmp_path):
    model_root, manifest_path = _weight_fixture(tmp_path / "weights")
    fifo = model_root / "alternate-weight.pipe"
    os.mkfifo(fifo)

    with pytest.raises(Hy3DHumanNotReady, match="non-regular|regular file"):
        contract.verify_weight_manifest(model_root, manifest_path)


def test_weight_manifest_requires_every_key_checkpoint_and_model_index(tmp_path):
    model_root, manifest_path = _weight_fixture(tmp_path / "weights")
    lines = [
        line
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if "model_index.json" not in line
    ]
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(Hy3DHumanNotReady, match="required|model_index"):
        contract.verify_weight_manifest(model_root, manifest_path)


def test_hy3d_manifest_is_atomic_and_records_only_the_current_staging_outputs(
    tmp_path, monkeypatch
):
    job, weight_manifest, staging_dir = _job_for_manifest(tmp_path, monkeypatch)
    outputs = _write_outputs(staging_dir)

    manifest_path = write_hy3d_manifest(staging_dir, job=job, outputs=outputs)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "hy3d_human_candidate_v1"
    assert manifest["asset_id"] == job["asset_id"]
    assert manifest["hunyuan_code_revision"] == job["hunyuan_runtime_git_head"]
    assert manifest["weight_root"] == str(CANONICAL_MODEL_ROOT)
    assert manifest["seed"] == ASSET_SEEDS[job["asset_id"]]
    assert manifest["steps"] == 50
    assert manifest["usage_scope"] == "technical_spike_only"
    assert manifest["reference_review_sha256"] == job["reference_review_sha256"]
    assert manifest["candidate_manifest_sha256"] == job["candidate_manifest_sha256"]
    assert manifest["weight_manifest_sha256"] == _sha256(weight_manifest)
    assert manifest["hunyuan_runtime_fingerprint"] == job[
        "hunyuan_runtime_fingerprint"
    ]
    assert manifest["hunyuan_runtime_git_head"] == job["hunyuan_runtime_git_head"]
    assert manifest["hunyuan_runtime_file_count"] == job[
        "hunyuan_runtime_file_count"
    ]
    assert manifest["outputs"]["shape"]["sha256"] == _sha256(outputs["shape"])
    assert manifest["outputs"]["paint_obj"]["size_bytes"] == outputs["paint_obj"].stat().st_size
    assert not list(staging_dir.glob(".hy3d_manifest.json.*.tmp"))


def test_hy3d_manifest_rejects_noncanonical_guidance(tmp_path, monkeypatch):
    job, _, staging_dir = _job_for_manifest(tmp_path, monkeypatch)
    job["guidance_scale"] = 4.0

    with pytest.raises(ValueError, match="guidance"):
        write_hy3d_manifest(staging_dir, job=job, outputs=_write_outputs(staging_dir))


def test_hy3d_manifest_rejects_a_changed_weight_manifest(tmp_path, monkeypatch):
    job, weight_manifest, staging_dir = _job_for_manifest(tmp_path, monkeypatch)
    weight_manifest.write_text(
        weight_manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="weight manifest.*changed|SHA"):
        write_hy3d_manifest(staging_dir, job=job, outputs=_write_outputs(staging_dir))


def test_hy3d_manifest_rejects_changed_runtime_code(tmp_path, monkeypatch):
    job, _, staging_dir = _job_for_manifest(tmp_path, monkeypatch)
    monkeypatch.setattr(
        contract,
        "current_hunyuan_runtime_provenance",
        lambda *args: {
            "git_head": job["hunyuan_runtime_git_head"],
            "fingerprint": "f" * 64,
            "file_count": job["hunyuan_runtime_file_count"],
        },
    )

    with pytest.raises(ValueError, match="runtime.*changed|fingerprint"):
        write_hy3d_manifest(staging_dir, job=job, outputs=_write_outputs(staging_dir))


@pytest.mark.parametrize(
    ("bad_name", "bad_target"),
    (("unexpected", "extra.bin"), ("shape", "../outside.glb")),
)
def test_hy3d_manifest_rejects_output_names_or_paths_outside_asset_dir(
    tmp_path, monkeypatch, bad_name, bad_target
):
    job, _, staging_dir = _job_for_manifest(tmp_path, monkeypatch)
    outputs = _write_outputs(staging_dir)
    if bad_name == "unexpected":
        bad_path = staging_dir / bad_target
        bad_path.write_bytes(b"not allowed")
        outputs[bad_name] = bad_path
    else:
        outside = tmp_path / "outside.glb"
        outside.write_bytes(b"outside")
        outputs[bad_name] = staging_dir / bad_target

    with pytest.raises(ValueError, match="allowlist|directly under|containment"):
        write_hy3d_manifest(staging_dir, job=job, outputs=outputs)


def test_hy3d_manifest_rejects_symlinked_outputs(tmp_path, monkeypatch):
    job, _, staging_dir = _job_for_manifest(tmp_path, monkeypatch)
    outputs = _write_outputs(staging_dir)
    external = tmp_path / "external-shape.glb"
    external.write_bytes(outputs["shape"].read_bytes())
    outputs["shape"].unlink()
    outputs["shape"].symlink_to(external)

    with pytest.raises(ValueError, match="regular file|symlink"):
        write_hy3d_manifest(staging_dir, job=job, outputs=outputs)
