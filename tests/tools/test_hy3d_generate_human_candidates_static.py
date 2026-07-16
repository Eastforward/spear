"""CPU-only tests for the approved Hunyuan human candidate runner."""

from __future__ import annotations

import ast
import hashlib
import builtins
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "hy3d_generate_human_candidates.py"
PAINT_SCRIPT = REPO / "tools" / "hy3d_bake_diffuse.py"
SHAPE_UTILS = (
    REPO.parent
    / "Hunyuan3D-2.1"
    / "hy3dshape"
    / "hy3dshape"
    / "utils"
    / "utils.py"
)
SHAPE_PIPELINES = (
    REPO.parent
    / "Hunyuan3D-2.1"
    / "hy3dshape"
    / "hy3dshape"
    / "pipelines.py"
)
CONTRACT_DIR = REPO / "tools" / "spike_rlr"
if str(CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTRACT_DIR))

from hy3d_human_candidate import (  # noqa: E402
    ASSET_SEEDS,
    CANONICAL_MODEL_PARENT,
    CANONICAL_MODEL_ROOT,
    GUIDANCE_SCALE,
    INFERENCE_STEPS,
    OUTPUT_FILENAMES,
    WEIGHT_ROOT_HASH_MANIFEST,
    Hy3DHumanNotReady,
)
import hy3d_human_candidate as contract  # noqa: E402


def _load_runner():
    spec = importlib.util.spec_from_file_location("hy3d_generate_human_candidates", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_paint_wrapper():
    old_cwd = Path.cwd()
    try:
        spec = importlib.util.spec_from_file_location("hy3d_bake_diffuse_task1", PAINT_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        os.chdir(old_cwd)


def _load_real_shape_utils(monkeypatch):
    fake_torch = ModuleType("torch")
    fake_hub = ModuleType("huggingface_hub")

    def forbid_download(*args, **kwargs):
        raise AssertionError("shape loader attempted a network fallback")

    fake_hub.snapshot_download = forbid_download
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    spec = importlib.util.spec_from_file_location("task1_real_shape_utils", SHAPE_UTILS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _shape_model_fixture(root: Path) -> tuple[Path, Path]:
    model_root = root / "hunyuan3d-2.1"
    submodel = model_root / "hunyuan3d-dit-v2-1"
    submodel.mkdir(parents=True)
    (submodel / "config.yaml").write_bytes(b"model: {}\n")
    (submodel / "model.fp16.ckpt").write_bytes(b"shape weights")
    return model_root, submodel


def _execute_real_pipeline_from_pretrained():
    tree = ast.parse(SHAPE_PIPELINES.read_text(encoding="utf-8"))
    pipeline_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Hunyuan3DDiTPipeline"
    )
    method = next(
        node
        for node in pipeline_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "from_pretrained"
    )
    harness = ast.ClassDef(
        name="PipelineHarness",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[harness], type_ignores=[]))
    load_calls = []

    def smart_load_model(model_path, **kwargs):
        load_calls.append((model_path, kwargs))
        return "/local/config.yaml", "/local/model.fp16.ckpt"

    namespace = {
        "smart_load_model": smart_load_model,
        "torch": SimpleNamespace(float16="float16"),
    }
    exec(compile(module, str(SHAPE_PIPELINES), "exec"), namespace)
    pipeline = namespace["PipelineHarness"]

    def from_single_file(cls, ckpt_path, config_path, **kwargs):
        return {
            "ckpt_path": ckpt_path,
            "config_path": config_path,
            "kwargs": kwargs,
        }

    pipeline.from_single_file = classmethod(from_single_file)
    return pipeline, load_calls


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _local_model_layout(root: Path) -> Path:
    model_root = root / "hunyuan3d-2.1"
    for relative in (
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
    ):
        path = model_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"{}\n" if path.suffix == ".json" else relative.encode())
    return model_root


def _job(tmp_path: Path, asset_id: str = "rocketbox_male_adult_01") -> dict:
    review_root = tmp_path / "reviews"
    candidate_dir = review_root / asset_id
    candidate_dir.mkdir(parents=True)
    candidate = candidate_dir / "candidate.png"
    candidate.write_bytes(b"approved candidate")
    asset_dir = tmp_path / "out" / asset_id
    asset_dir.mkdir(parents=True)
    runtime = getattr(
        contract,
        "current_hunyuan_runtime_provenance",
        lambda *args: {
            "git_head": "1" * 40,
            "fingerprint": "2" * 64,
            "file_count": 1,
        },
    )()
    return {
        "asset_id": asset_id,
        "review_root": review_root,
        "candidate_path": candidate,
        "candidate_sha256": _sha256(candidate),
        "candidate_manifest_sha256": "b" * 64,
        "source_sha256": "c" * 64,
        "source_approval_sha256": "d" * 64,
        "reference_review_sha256": "e" * 64,
        "seed": ASSET_SEEDS[asset_id],
        "steps": INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "model_root": CANONICAL_MODEL_ROOT,
        "weight_root_hash_manifest": WEIGHT_ROOT_HASH_MANIFEST,
        "weight_manifest_sha256": _sha256(WEIGHT_ROOT_HASH_MANIFEST),
        "hunyuan_runtime_git_head": runtime["git_head"],
        "hunyuan_runtime_fingerprint": runtime["fingerprint"],
        "hunyuan_runtime_file_count": runtime["file_count"],
        "asset_dir": asset_dir,
    }


def _current_approval_job(job: dict) -> dict:
    return {
        key: value
        for key, value in job.items()
        if key not in {"asset_dir", "weight_manifest_sha256"}
    }


def _write_old_asset(job: dict) -> dict[str, bytes]:
    asset_dir = job["asset_dir"]
    old = {}
    for label, filename in OUTPUT_FILENAMES.items():
        content = f"old-{label}".encode()
        (asset_dir / filename).write_bytes(content)
        old[label] = content
    (asset_dir / "hy3d_manifest.json").write_text('{"old": true}\n', encoding="utf-8")
    return old


def _install_approval_gate(monkeypatch, runner, job, mutate_second=None):
    calls = []

    def gate(review_root):
        calls.append(Path(review_root))
        current = _current_approval_job(job)
        if len(calls) == 2 and mutate_second is not None:
            current = dict(current)
            current[mutate_second] = _different_value(current[mutate_second])
        return {job["asset_id"]: current}

    monkeypatch.setattr(runner, "assert_generation_ready", gate)
    return calls


def _different_value(value):
    if isinstance(value, Path):
        return value.with_name(f"changed-{value.name}")
    if isinstance(value, str):
        return "f" * 64 if len(value) == 64 and value != "f" * 64 else value + "-changed"
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, int):
        return value + 1
    raise AssertionError(f"no mutation for {value!r}")


def _install_successful_generation(monkeypatch, runner, staging_observations):
    def remove_background(reference, destination):
        staging_observations.append(("rembg", Path(destination).parent))
        Path(destination).write_bytes(b"new-cutout")

    def generate_shape(pipeline, job, reference, shape):
        staging_observations.append(("shape", Path(shape).parent))
        Path(shape).write_bytes(b"new-shape")

    def run_paint(shape, reference, workdir):
        workdir = Path(workdir)
        staging_observations.append(("paint", workdir))
        for label in ("paint_obj", "diffuse", "metallic", "roughness"):
            (workdir / OUTPUT_FILENAMES[label]).write_bytes(f"new-{label}".encode())

    monkeypatch.setattr(runner, "remove_background", remove_background)
    monkeypatch.setattr(runner, "generate_shape", generate_shape)
    monkeypatch.setattr(runner, "run_paint", run_paint)


class _FakeGenerator:
    created = []

    def __init__(self, device):
        self.device = device
        self.seed = None
        self.__class__.created.append(self)

    def manual_seed(self, seed):
        self.seed = seed
        return self


class _FakeTorch(ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.Generator = _FakeGenerator


class _FakeShapePipeline:
    from_pretrained_calls = []
    load_observations = []

    def __init__(self):
        self.calls = []

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        cls.load_observations.append(
            {
                "cwd": Path.cwd(),
                "sys_path": tuple(sys.path),
                "offline": tuple(
                    os.environ.get(name)
                    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DIFFUSERS_OFFLINE")
                ),
                "model_env": os.environ.get("HY3DGEN_MODELS"),
            }
        )
        return cls()

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        mesh = SimpleNamespace(export=lambda path: Path(path).write_bytes(b"shape"))
        return SimpleNamespace(meshes=[mesh])


def test_load_shape_pipeline_uses_checkout_scope_and_absolute_local_weights(
    monkeypatch,
):
    runner = _load_runner()
    fake_package = ModuleType("hy3dshape")
    fake_pipelines = ModuleType("hy3dshape.pipelines")
    fake_pipelines.Hunyuan3DDiTFlowMatchingPipeline = _FakeShapePipeline
    monkeypatch.setitem(sys.modules, "hy3dshape", fake_package)
    monkeypatch.setitem(sys.modules, "hy3dshape.pipelines", fake_pipelines)
    verified = []
    monkeypatch.setattr(
        runner,
        "verify_canonical_weights",
        lambda: verified.append("weights") or "9" * 64,
    )
    monkeypatch.setattr(
        runner, "_validate_local_model_layout", lambda: CANONICAL_MODEL_ROOT
    )
    old_cwd = Path.cwd()
    old_sys_path = list(sys.path)
    monkeypatch.setenv("HY3DGEN_MODELS", "original-model-env")
    _FakeShapePipeline.from_pretrained_calls.clear()
    _FakeShapePipeline.load_observations.clear()

    runner.load_shape_pipeline()

    assert verified == ["weights"]
    assert _FakeShapePipeline.from_pretrained_calls == [
        ((str(CANONICAL_MODEL_ROOT),), {"local_files_only": True})
    ]
    observation = _FakeShapePipeline.load_observations[0]
    assert observation["cwd"] == runner.HUNYUAN_CHECKOUT
    assert str(runner.HUNYUAN_CHECKOUT / "hy3dshape") in observation["sys_path"]
    assert str(runner.HUNYUAN_CHECKOUT / "hy3dpaint") in observation["sys_path"]
    assert observation["offline"] == ("1", "1", "1")
    assert observation["model_env"] == str(CANONICAL_MODEL_PARENT)
    assert Path.cwd() == old_cwd
    assert sys.path == old_sys_path
    assert os.environ["HY3DGEN_MODELS"] == "original-model-env"


def test_real_checkout_shape_loader_contains_no_hub_or_repo_fallback():
    source = SHAPE_UTILS.read_text(encoding="utf-8")

    assert "huggingface_hub" not in source
    assert "snapshot_download" not in source
    assert "repo_id" not in source
    assert "HY3DGEN_MODELS" not in source


def test_real_checkout_shape_loader_accepts_absolute_root_or_submodel(
    tmp_path, monkeypatch
):
    shape_utils = _load_real_shape_utils(monkeypatch)
    model_root, submodel = _shape_model_fixture(tmp_path)

    from_root = shape_utils.smart_load_model(
        str(model_root),
        subfolder="hunyuan3d-dit-v2-1",
        use_safetensors=False,
        variant="fp16",
        local_files_only=True,
    )
    from_submodel = shape_utils.smart_load_model(
        str(submodel),
        subfolder="hunyuan3d-dit-v2-1",
        use_safetensors=False,
        variant="fp16",
        local_files_only=True,
    )

    expected = (
        str(submodel / "config.yaml"),
        str(submodel / "model.fp16.ckpt"),
    )
    assert from_root == from_submodel == expected


def test_real_checkout_shape_loader_rejects_relative_or_missing_paths(
    tmp_path, monkeypatch
):
    shape_utils = _load_real_shape_utils(monkeypatch)
    model_root, submodel = _shape_model_fixture(tmp_path)

    with pytest.raises(ValueError, match="absolute|local"):
        shape_utils.smart_load_model(
            model_root.name,
            subfolder="hunyuan3d-dit-v2-1",
            use_safetensors=False,
            variant="fp16",
            local_files_only=True,
        )

    (submodel / "model.fp16.ckpt").unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint|model|ckpt"):
        shape_utils.smart_load_model(
            str(model_root),
            subfolder="hunyuan3d-dit-v2-1",
            use_safetensors=False,
            variant="fp16",
            local_files_only=True,
        )


def test_real_checkout_shape_loader_rejects_symlinks_and_subfolder_escape(
    tmp_path, monkeypatch
):
    shape_utils = _load_real_shape_utils(monkeypatch)
    model_root, submodel = _shape_model_fixture(tmp_path / "canonical")
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(model_root, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|real local"):
        shape_utils.smart_load_model(
            str(linked_root),
            subfolder="hunyuan3d-dit-v2-1",
            use_safetensors=False,
            variant="fp16",
            local_files_only=True,
        )

    outside = model_root.parent / "outside-model"
    outside.mkdir()
    (outside / "config.yaml").write_bytes((submodel / "config.yaml").read_bytes())
    (outside / "model.fp16.ckpt").write_bytes(
        (submodel / "model.fp16.ckpt").read_bytes()
    )
    with pytest.raises(ValueError, match="escape|subfolder|relative"):
        shape_utils.smart_load_model(
            str(model_root),
            subfolder="../outside-model",
            use_safetensors=False,
            variant="fp16",
            local_files_only=True,
        )


def test_real_checkout_shape_loader_rejects_symlinked_checkpoint(
    tmp_path, monkeypatch
):
    shape_utils = _load_real_shape_utils(monkeypatch)
    model_root, submodel = _shape_model_fixture(tmp_path)
    checkpoint = submodel / "model.fp16.ckpt"
    outside = tmp_path / "outside.ckpt"
    outside.write_bytes(checkpoint.read_bytes())
    checkpoint.unlink()
    checkpoint.symlink_to(outside)

    with pytest.raises(ValueError, match="symlink|regular"):
        shape_utils.smart_load_model(
            str(model_root),
            subfolder="hunyuan3d-dit-v2-1",
            use_safetensors=False,
            variant="fp16",
            local_files_only=True,
        )


def test_real_pipeline_from_pretrained_propagates_and_enforces_local_only():
    pipeline, load_calls = _execute_real_pipeline_from_pretrained()

    result = pipeline.from_pretrained(
        str(CANONICAL_MODEL_ROOT), local_files_only=True
    )

    assert load_calls == [
        (
            str(CANONICAL_MODEL_ROOT),
            {
                "subfolder": "hunyuan3d-dit-v2-1",
                "use_safetensors": False,
                "variant": "fp16",
                "local_files_only": True,
            },
        )
    ]
    assert result["kwargs"]["from_pretrained_kwargs"]["local_files_only"] is True

    with pytest.raises(ValueError, match="local_files_only|local-only"):
        pipeline.from_pretrained(
            str(CANONICAL_MODEL_ROOT), local_files_only=False
        )
    assert len(load_calls) == 1


def test_generate_shape_uses_asset_seed_and_exactly_50_steps(tmp_path, monkeypatch):
    runner = _load_runner()
    job = _job(tmp_path, "rocketbox_female_adult_01")
    reference = tmp_path / "reference_rembg.png"
    reference.write_bytes(b"reference")
    shape = tmp_path / "shape.glb"
    pipeline = _FakeShapePipeline()
    _FakeGenerator.created.clear()
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())

    runner.generate_shape(pipeline, job, reference, shape)

    assert [(item.device, item.seed) for item in _FakeGenerator.created] == [
        ("cuda", 7301)
    ]
    assert pipeline.calls[0]["num_inference_steps"] == 50
    assert pipeline.calls[0]["guidance_scale"] == 5.0
    assert shape.read_bytes() == b"shape"


def test_build_generation_job_adds_verified_weight_manifest_sha(tmp_path, monkeypatch):
    runner = _load_runner()
    job = _job(tmp_path)
    approval = _current_approval_job(job)
    monkeypatch.setattr(
        runner, "assert_generation_ready", lambda root: {job["asset_id"]: approval}
    )
    monkeypatch.setattr(runner, "current_weight_manifest_sha256", lambda: "7" * 64)

    built = runner.build_generation_job(
        job["review_root"], job["asset_id"], tmp_path / "new-output"
    )

    assert built["weight_manifest_sha256"] == "7" * 64
    assert built["asset_dir"] == tmp_path / "new-output" / job["asset_id"]


def test_build_generation_job_confines_assets_to_a_real_output_root(tmp_path, monkeypatch):
    runner = _load_runner()
    job = _job(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    output_root = tmp_path / "linked-output"
    output_root.symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(
        runner,
        "assert_generation_ready",
        lambda root: {job["asset_id"]: _current_approval_job(job)},
    )

    with pytest.raises(ValueError, match="symlink|output_root"):
        runner.build_generation_job(job["review_root"], job["asset_id"], output_root)
    assert list(external.iterdir()) == []


def test_run_job_uses_empty_sibling_staging_and_replaces_all_old_outputs(
    tmp_path, monkeypatch
):
    runner = _load_runner()
    job = _job(tmp_path)
    old = _write_old_asset(job)
    approval_calls = _install_approval_gate(monkeypatch, runner, job)
    monkeypatch.setattr(
        runner, "verify_canonical_weights", lambda: job["weight_manifest_sha256"]
    )
    staging_observations = []
    _install_successful_generation(monkeypatch, runner, staging_observations)
    original_create_staging = runner._create_staging_dir

    def create_empty_staging(asset_dir):
        staging = original_create_staging(asset_dir)
        assert list(staging.iterdir()) == []
        assert staging.parent == Path(asset_dir).parent
        return staging

    monkeypatch.setattr(runner, "_create_staging_dir", create_empty_staging)

    manifest_path = runner.run_job(object(), job)

    assert len(approval_calls) == 2
    staging_dirs = {path for _, path in staging_observations}
    assert len(staging_dirs) == 1
    staging_dir = staging_dirs.pop()
    assert staging_dir.parent == job["asset_dir"].parent
    assert not staging_dir.exists()
    assert manifest_path == job["asset_dir"] / "hy3d_manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["seed"] == 4101
    for label, filename in OUTPUT_FILENAMES.items():
        assert (job["asset_dir"] / filename).read_bytes() != old[label]


@pytest.mark.parametrize(
    "changed_field",
    (
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
    ),
)
def test_run_job_rejects_every_approval_job_field_changed_before_publish(
    tmp_path, monkeypatch, changed_field
):
    runner = _load_runner()
    job = _job(tmp_path)
    old = _write_old_asset(job)
    _install_approval_gate(monkeypatch, runner, job, mutate_second=changed_field)
    monkeypatch.setattr(
        runner, "verify_canonical_weights", lambda: job["weight_manifest_sha256"]
    )
    _install_successful_generation(monkeypatch, runner, [])

    with pytest.raises(Hy3DHumanNotReady, match=changed_field):
        runner.run_job(object(), job)

    assert not (job["asset_dir"] / "hy3d_manifest.json").exists()
    for label, filename in OUTPUT_FILENAMES.items():
        assert (job["asset_dir"] / filename).read_bytes() == old[label]
    assert not list(job["asset_dir"].parent.glob(f".{job['asset_id']}.*.staging"))


def test_run_job_rejects_runtime_code_changed_during_generation(tmp_path, monkeypatch):
    runner = _load_runner()
    job = _job(tmp_path)
    old = _write_old_asset(job)
    _install_approval_gate(
        monkeypatch,
        runner,
        job,
        mutate_second="hunyuan_runtime_fingerprint",
    )
    monkeypatch.setattr(
        runner, "verify_canonical_weights", lambda: job["weight_manifest_sha256"]
    )
    _install_successful_generation(monkeypatch, runner, [])

    with pytest.raises(Hy3DHumanNotReady, match="hunyuan_runtime_fingerprint"):
        runner.run_job(object(), job)

    assert not (job["asset_dir"] / "hy3d_manifest.json").exists()
    for label, filename in OUTPUT_FILENAMES.items():
        assert (job["asset_dir"] / filename).read_bytes() == old[label]


def test_failed_generation_invalidates_old_manifest_and_cleans_staging(tmp_path, monkeypatch):
    runner = _load_runner()
    job = _job(tmp_path)
    _write_old_asset(job)
    _install_approval_gate(monkeypatch, runner, job)
    monkeypatch.setattr(
        runner, "verify_canonical_weights", lambda: job["weight_manifest_sha256"]
    )
    monkeypatch.setattr(
        runner,
        "remove_background",
        lambda reference, destination: (_ for _ in ()).throw(RuntimeError("injected failure")),
    )

    with pytest.raises(RuntimeError, match="injected failure"):
        runner.run_job(object(), job)

    assert not (job["asset_dir"] / "hy3d_manifest.json").exists()
    assert not list(job["asset_dir"].parent.glob(f".{job['asset_id']}.*.staging"))


def test_run_job_rejects_a_copied_candidate_hash_before_generation(tmp_path, monkeypatch):
    runner = _load_runner()
    job = _job(tmp_path)
    _write_old_asset(job)
    _install_approval_gate(monkeypatch, runner, job)
    monkeypatch.setattr(
        runner, "verify_canonical_weights", lambda: job["weight_manifest_sha256"]
    )
    original_copy = runner._copy_atomically

    def copy_then_tamper(source, destination):
        original_copy(source, destination)
        Path(destination).write_bytes(b"tampered copy")

    monkeypatch.setattr(runner, "_copy_atomically", copy_then_tamper)
    monkeypatch.setattr(
        runner,
        "remove_background",
        lambda *args: pytest.fail("generation must not see a hash-mismatched copy"),
    )

    with pytest.raises(Hy3DHumanNotReady, match="copied candidate|hash"):
        runner.run_job(object(), job)

    assert not (job["asset_dir"] / "hy3d_manifest.json").exists()


def test_missing_new_metallic_cannot_reuse_old_metallic(tmp_path, monkeypatch):
    runner = _load_runner()
    job = _job(tmp_path)
    old = _write_old_asset(job)
    _install_approval_gate(monkeypatch, runner, job)
    monkeypatch.setattr(
        runner, "verify_canonical_weights", lambda: job["weight_manifest_sha256"]
    )

    def remove_background(reference, destination):
        Path(destination).write_bytes(b"new-cutout")

    def generate_shape(pipeline, received_job, reference, shape):
        Path(shape).write_bytes(b"new-shape")

    def incomplete_paint(shape, reference, workdir):
        for label in ("paint_obj", "diffuse", "roughness"):
            (Path(workdir) / OUTPUT_FILENAMES[label]).write_bytes(b"new")

    monkeypatch.setattr(runner, "remove_background", remove_background)
    monkeypatch.setattr(runner, "generate_shape", generate_shape)
    monkeypatch.setattr(runner, "run_paint", incomplete_paint)

    with pytest.raises(ValueError, match="metallic|regular file"):
        runner.run_job(object(), job)

    assert (job["asset_dir"] / OUTPUT_FILENAMES["metallic"]).read_bytes() == old["metallic"]
    assert not (job["asset_dir"] / "hy3d_manifest.json").exists()


def test_publish_unlinks_output_and_manifest_symlinks_without_touching_targets(
    tmp_path, monkeypatch
):
    runner = _load_runner()
    job = _job(tmp_path)
    _write_old_asset(job)
    external_output = tmp_path / "external-metallic.jpg"
    external_output.write_bytes(b"external output")
    metallic = job["asset_dir"] / OUTPUT_FILENAMES["metallic"]
    metallic.unlink()
    metallic.symlink_to(external_output)
    external_manifest = tmp_path / "external-manifest.json"
    external_manifest.write_bytes(b"external manifest")
    manifest = job["asset_dir"] / "hy3d_manifest.json"
    manifest.unlink()
    manifest.symlink_to(external_manifest)
    _install_approval_gate(monkeypatch, runner, job)
    monkeypatch.setattr(
        runner, "verify_canonical_weights", lambda: job["weight_manifest_sha256"]
    )
    _install_successful_generation(monkeypatch, runner, [])

    runner.run_job(object(), job)

    assert external_output.read_bytes() == b"external output"
    assert external_manifest.read_bytes() == b"external manifest"
    assert metallic.is_file() and not metallic.is_symlink()
    assert manifest.is_file() and not manifest.is_symlink()


def test_publish_rejects_directory_at_canonical_output_without_a_manifest(
    tmp_path, monkeypatch
):
    runner = _load_runner()
    job = _job(tmp_path)
    _write_old_asset(job)
    shape = job["asset_dir"] / OUTPUT_FILENAMES["shape"]
    shape.unlink()
    shape.mkdir()
    _install_approval_gate(monkeypatch, runner, job)
    monkeypatch.setattr(
        runner, "verify_canonical_weights", lambda: job["weight_manifest_sha256"]
    )
    _install_successful_generation(monkeypatch, runner, [])

    with pytest.raises(ValueError, match="non-regular|canonical output"):
        runner.run_job(object(), job)

    assert shape.is_dir()
    assert not (job["asset_dir"] / "hy3d_manifest.json").exists()


def test_run_paint_uses_absolute_paths_checkout_and_offline_environment(
    tmp_path, monkeypatch
):
    runner = _load_runner()
    shape = tmp_path / "shape.glb"
    reference = tmp_path / "reference_rembg.png"
    workdir = tmp_path / "staging"
    shape.write_bytes(b"shape")
    reference.write_bytes(b"reference")
    workdir.mkdir()
    calls = []

    def capture_offline_call(*args, **kwargs):
        cache_paths = {
            Path(kwargs["env"][name])
            for name in (
                "HF_HOME",
                "HF_HUB_CACHE",
                "TRANSFORMERS_CACHE",
                "DIFFUSERS_CACHE",
            )
        }
        assert len(cache_paths) == 1
        cache_path = cache_paths.pop()
        assert cache_path.parent == workdir
        assert cache_path.is_dir()
        assert list(cache_path.iterdir()) == []
        calls.append((args, kwargs))

    monkeypatch.setattr(runner.subprocess, "run", capture_offline_call)
    monkeypatch.setattr(
        runner, "_validate_local_model_layout", lambda: CANONICAL_MODEL_ROOT
    )

    runner.run_paint(shape, reference, workdir)

    args, kwargs = calls[0]
    command = args[0]
    assert command[0] == sys.executable
    assert command[1] == str((REPO / "tools" / "hy3d_bake_diffuse.py").resolve())
    assert all(Path(value).is_absolute() for value in (command[3], command[5], command[7]))
    realesrgan_index = command.index("--realesrgan-ckpt")
    dinov2_index = command.index("--dinov2-root")
    weight_manifest_index = command.index("--weight-manifest")
    assert command[realesrgan_index + 1] == str(
        CANONICAL_MODEL_ROOT
        / "dependencies"
        / "realesrgan"
        / "RealESRGAN_x4plus.pth"
    )
    assert command[dinov2_index + 1] == str(
        CANONICAL_MODEL_ROOT / "dependencies" / "dinov2-giant"
    )
    assert command[weight_manifest_index + 1] == str(WEIGHT_ROOT_HASH_MANIFEST)
    assert kwargs["env"]["HY3D_ROOT"] == str(runner.HUNYUAN_CHECKOUT)
    assert kwargs["env"]["HY3DGEN_MODELS"] == str(CANONICAL_MODEL_PARENT)
    assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
    assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert kwargs["env"]["DIFFUSERS_OFFLINE"] == "1"
    assert kwargs["check"] is True
    assert not Path(kwargs["env"]["HF_HOME"]).exists()


def test_paint_wrapper_requires_explicit_dependency_arguments(tmp_path, monkeypatch):
    wrapper = _load_paint_wrapper()
    common = [
        "hy3d_bake_diffuse.py",
        "--input-glb",
        str(tmp_path / "shape.glb"),
        "--reference-image",
        str(tmp_path / "reference.png"),
        "--workdir",
        str(tmp_path / "work"),
        "--realesrgan-ckpt",
        str(CANONICAL_MODEL_ROOT / "dependencies/realesrgan/RealESRGAN_x4plus.pth"),
        "--dinov2-root",
        str(CANONICAL_MODEL_ROOT / "dependencies/dinov2-giant"),
    ]
    monkeypatch.setattr(sys, "argv", common)

    with pytest.raises(SystemExit):
        wrapper.parse_args()


def test_paint_wrapper_verifies_manifest_before_importing_pipeline(
    tmp_path, monkeypatch
):
    wrapper = _load_paint_wrapper()
    input_glb = tmp_path / "shape.glb"
    reference = tmp_path / "reference.png"
    input_glb.write_bytes(b"shape")
    reference.write_bytes(b"reference")
    manifest = tmp_path / "weights.sha256"
    manifest.write_text("placeholder\n", encoding="utf-8")
    events = []
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "textureGenPipeline":
            events.append("paint-import")
            raise RuntimeError("stop at paint import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(
        wrapper,
        "verify_canonical_weight_manifest",
        lambda path: events.append(("verify", Path(path))) or "1" * 64,
        raising=False,
    )
    monkeypatch.setattr(wrapper, "_canonical_directory", lambda path, *args: Path(path))
    monkeypatch.setattr(wrapper, "validate_paint_dependencies", lambda *args: None)
    monkeypatch.setattr(
        wrapper,
        "_resolve_multiview_pretrained_path",
        lambda: str(CANONICAL_MODEL_ROOT),
    )
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    args = SimpleNamespace(
        input_glb=input_glb,
        reference_image=reference,
        workdir=tmp_path / "work",
        realesrgan_ckpt=CANONICAL_MODEL_ROOT
        / "dependencies/realesrgan/RealESRGAN_x4plus.pth",
        dinov2_root=CANONICAL_MODEL_ROOT / "dependencies/dinov2-giant",
        weight_manifest=manifest,
        max_num_view=6,
        resolution=512,
        texture_size=4096,
    )

    with pytest.raises(RuntimeError, match="stop at paint import"):
        wrapper._run(args)

    assert events == [("verify", manifest), "paint-import"]


def test_paint_wrapper_sets_only_validated_canonical_dependency_paths(
    tmp_path, monkeypatch
):
    wrapper = _load_paint_wrapper()
    model_root = tmp_path / "hunyuan3d-2.1"
    realesrgan = model_root / "dependencies/realesrgan/RealESRGAN_x4plus.pth"
    dino_root = model_root / "dependencies/dinov2-giant"
    realesrgan.parent.mkdir(parents=True)
    realesrgan.write_bytes(b"realesrgan")
    dino_root.mkdir(parents=True)
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        (dino_root / name).write_bytes(name.encode())
    monkeypatch.setattr(wrapper, "CANONICAL_MODEL_ROOT", model_root)
    monkeypatch.setattr(wrapper, "CANONICAL_REALESRGAN_CKPT", realesrgan)
    monkeypatch.setattr(wrapper, "CANONICAL_DINOV2_ROOT", dino_root)
    config = SimpleNamespace()

    wrapper.configure_paint_dependencies(config, realesrgan, dino_root)

    assert config.realesrgan_ckpt_path == str(realesrgan)
    assert config.dino_ckpt_path == str(dino_root)
    assert "facebook/dinov2-giant" not in vars(config).values()


def test_paint_wrapper_rejects_noncanonical_or_symlinked_dependencies(
    tmp_path, monkeypatch
):
    wrapper = _load_paint_wrapper()
    model_root = tmp_path / "hunyuan3d-2.1"
    canonical_realesrgan = (
        model_root / "dependencies/realesrgan/RealESRGAN_x4plus.pth"
    )
    canonical_dino = model_root / "dependencies/dinov2-giant"
    canonical_realesrgan.parent.mkdir(parents=True)
    canonical_realesrgan.write_bytes(b"canonical")
    canonical_dino.mkdir(parents=True)
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        (canonical_dino / name).write_bytes(name.encode())
    monkeypatch.setattr(wrapper, "CANONICAL_MODEL_ROOT", model_root)
    monkeypatch.setattr(
        wrapper, "CANONICAL_REALESRGAN_CKPT", canonical_realesrgan
    )
    monkeypatch.setattr(wrapper, "CANONICAL_DINOV2_ROOT", canonical_dino)
    outside = tmp_path / "outside.pth"
    outside.write_bytes(b"outside")

    with pytest.raises(ValueError, match="canonical"):
        wrapper.configure_paint_dependencies(SimpleNamespace(), outside, canonical_dino)

    external_dino = tmp_path / "external-dino-model.safetensors"
    external_dino.write_bytes(b"external")
    dino_model = canonical_dino / "model.safetensors"
    dino_model.unlink()
    dino_model.symlink_to(external_dino)
    with pytest.raises(ValueError, match="symlink"):
        wrapper.configure_paint_dependencies(
            SimpleNamespace(), canonical_realesrgan, canonical_dino
        )


def test_paint_wrapper_rejects_relative_dependency_arguments(tmp_path, monkeypatch):
    wrapper = _load_paint_wrapper()
    model_root = tmp_path / "hunyuan3d-2.1"
    realesrgan = model_root / "dependencies/realesrgan/RealESRGAN_x4plus.pth"
    dino_root = model_root / "dependencies/dinov2-giant"
    realesrgan.parent.mkdir(parents=True)
    realesrgan.write_bytes(b"canonical")
    dino_root.mkdir(parents=True)
    for name in ("config.json", "preprocessor_config.json", "model.safetensors"):
        (dino_root / name).write_bytes(name.encode())
    monkeypatch.setattr(wrapper, "CANONICAL_MODEL_ROOT", model_root)
    monkeypatch.setattr(wrapper, "CANONICAL_REALESRGAN_CKPT", realesrgan)
    monkeypatch.setattr(wrapper, "CANONICAL_DINOV2_ROOT", dino_root)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="absolute"):
        wrapper.configure_paint_dependencies(
            SimpleNamespace(),
            realesrgan.relative_to(tmp_path),
            dino_root.relative_to(tmp_path),
        )


def test_paint_wrapper_and_runner_contain_no_remote_dinov2_identifier():
    wrapper_source = PAINT_SCRIPT.read_text(encoding="utf-8")
    runner_source = SCRIPT.read_text(encoding="utf-8")

    assert "facebook/dinov2-giant" not in wrapper_source
    assert "facebook/dinov2-giant" not in runner_source


def test_paint_wrapper_has_no_repo_id_or_missing_local_model_fallback(monkeypatch):
    wrapper = _load_paint_wrapper()
    monkeypatch.delenv("HY3DGEN_MODELS", raising=False)

    with pytest.raises((ValueError, FileNotFoundError), match="HY3DGEN_MODELS|local"):
        wrapper._resolve_multiview_pretrained_path()


def test_cli_requires_exactly_one_approved_asset_id(tmp_path):
    runner = _load_runner()
    common = ["--review-root", str(tmp_path), "--output-root", str(tmp_path / "out")]

    with pytest.raises(SystemExit):
        runner.parse_args(common)

    args = runner.parse_args(
        common + ["--asset-id", "rocketbox_female_adult_01"]
    )
    assert args.asset_id == "rocketbox_female_adult_01"


def test_main_runs_only_the_requested_asset(tmp_path, monkeypatch):
    runner = _load_runner()
    args = SimpleNamespace(
        review_root=tmp_path / "reviews",
        output_root=tmp_path / "out",
        asset_id="rocketbox_female_adult_01",
    )
    job = _job(tmp_path, args.asset_id)
    pipeline = object()
    built = []
    loaded = []
    run = []
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(
        runner,
        "build_generation_job",
        lambda review_root, asset_id, output_root: built.append(
            (review_root, asset_id, output_root)
        )
        or job,
    )
    monkeypatch.setattr(
        runner, "load_shape_pipeline", lambda: loaded.append(True) or pipeline
    )
    monkeypatch.setattr(
        runner, "run_job", lambda received_pipeline, received_job: run.append((received_pipeline, received_job))
    )

    runner.main()

    assert built == [(args.review_root, args.asset_id, args.output_root)]
    assert loaded == [True]
    assert run == [(pipeline, job)]


def test_main_invalidates_manifest_before_pipeline_load_failure(tmp_path, monkeypatch):
    runner = _load_runner()
    args = SimpleNamespace(
        review_root=tmp_path / "reviews",
        output_root=tmp_path / "out",
        asset_id="rocketbox_male_adult_01",
    )
    job = _job(tmp_path, args.asset_id)
    _write_old_asset(job)
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "build_generation_job", lambda *args: job)
    monkeypatch.setattr(
        runner,
        "load_shape_pipeline",
        lambda: (_ for _ in ()).throw(RuntimeError("injected load failure")),
    )

    with pytest.raises(RuntimeError, match="injected load failure"):
        runner.main()

    assert not (job["asset_dir"] / "hy3d_manifest.json").exists()


def test_real_checkout_and_local_model_config_subprocess_smoke(tmp_path):
    model_root = _local_model_layout(tmp_path / "models")
    code = f"""
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('hy3d_runner_smoke', {str(SCRIPT)!r})
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
info = runner.smoke_local_configuration(model_root=Path({str(model_root)!r}))
assert Path(info['checkout']) == runner.HUNYUAN_CHECKOUT
assert Path(info['model_root']) == Path({str(model_root)!r})
assert info['shape_import'] == 'hy3dshape'
assert info['paint_import'] == 'textureGenPipeline'
print('HY3D_LOCAL_SMOKE_OK')
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "HY3D_LOCAL_SMOKE_OK"


def test_runner_source_has_no_weight_migration_or_network_model_identifier():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "rename(" not in source
    assert "snapshot_download" not in source
    assert 'from_pretrained(\n        SHAPE_MODEL_NAME' not in source
