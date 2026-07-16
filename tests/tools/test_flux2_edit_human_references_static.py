import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "flux2_edit_human_references.py"
CONTRACT_DIR = REPO / "tools" / "spike_rlr"
if str(CONTRACT_DIR) not in sys.path:
    sys.path.insert(0, str(CONTRACT_DIR))

from human_reference_review import (  # noqa: E402
    HumanReferenceNotApproved,
    assert_reference_approved,
    record_review,
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("flux2_edit_human_references", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _current_snapshot(candidate_dir):
    return {
        "candidate_manifest_sha256": _sha256(candidate_dir / "candidate_manifest.json"),
        "source_sha256": _sha256(candidate_dir / "source.png"),
        "candidate_sha256": _sha256(candidate_dir / "candidate.png"),
    }


def _job(asset_id="rocketbox_male_adult_01", prompt="Keep this prompt exactly."):
    return {
        "asset_id": asset_id,
        "source_image": "/approved/front.png",
        "source_image_sha256": "a" * 64,
        "source_review": "/approved/source_review.json",
        "prompt": prompt,
        "seed": 1234,
        "width": 1152,
        "height": 1536,
        "steps": 28,
        "guidance_scale": 4.0,
    }


def _prepare_source_bundle(tmp_path, job, *, name=None, color="red", size=(12, 16)):
    source_dir = tmp_path / (name or f"{job['asset_id']}-approved")
    source_dir.mkdir()
    source = source_dir / "front.png"
    Image.new("RGB", size, color).save(source)
    review = source_dir / "source_review.json"
    review.write_text(f"approved review for {job['asset_id']}", encoding="utf-8")
    job["source_image"] = str(source)
    job["source_image_sha256"] = _sha256(source)
    job["source_review"] = str(review)
    return source, review


def _allow_source_reviews(monkeypatch, runner, review_asset_ids):
    monkeypatch.setattr(
        runner,
        "assert_source_review_approved",
        lambda path: {"asset_id": review_asset_ids[Path(path)]},
    )


def _create_snapshot(model_root, revision):
    snapshot = model_root / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    return snapshot


def _install_fake_pipeline_modules(monkeypatch):
    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.Flux2KleinPipeline = _FakePipeline
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)


class _FakeTorch:
    bfloat16 = object()

    class Generator:
        created = []

        def __init__(self, device):
            self.device = device
            self.seed = None
            self.__class__.created.append(self)

        def manual_seed(self, seed):
            self.seed = seed
            return self


class _FakePipeline:
    from_pretrained_calls = []

    def __init__(self, color="white"):
        self.device = None
        self.calls = []
        self.color = color

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        cls.from_pretrained_calls.append((args, kwargs))
        return cls()

    def to(self, device):
        self.device = device
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(images=[Image.new("RGB", (1152, 1536), self.color)])


def _run_one_valid_job(tmp_path, monkeypatch, runner, *, color="red"):
    job = _job()
    source, review = _prepare_source_bundle(tmp_path, job, color=color)
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    _allow_source_reviews(monkeypatch, runner, {review: job["asset_id"]})
    return job, source, review


def test_load_pipeline_uses_revision_snapshot_bf16_cuda_and_local_files_only(
    tmp_path, monkeypatch
):
    runner = _load_runner()
    model_root = tmp_path / "model-cache"
    snapshot = _create_snapshot(model_root, runner.MODEL_REVISION)
    _install_fake_pipeline_modules(monkeypatch)
    monkeypatch.setattr(runner, "PINNED_MODEL_ROOT", model_root)
    _FakePipeline.from_pretrained_calls.clear()

    pipeline = runner.load_pipeline(model_root, local_files_only=True)

    assert pipeline.device == "cuda"
    assert _FakePipeline.from_pretrained_calls == [
        (
            (str(snapshot),),
            {"torch_dtype": _FakeTorch.bfloat16, "local_files_only": True},
        )
    ]


def test_load_pipeline_rejects_non_pinned_or_network_enabled_loading(
    tmp_path, monkeypatch
):
    runner = _load_runner()
    model_root = tmp_path / "model-cache"
    _create_snapshot(model_root, runner.MODEL_REVISION)
    _install_fake_pipeline_modules(monkeypatch)
    monkeypatch.setattr(runner, "PINNED_MODEL_ROOT", model_root)

    with pytest.raises(ValueError, match="pinned local snapshot"):
        runner.load_pipeline(Path("/models/other"), local_files_only=True)
    with pytest.raises(ValueError, match="local_files_only"):
        runner.load_pipeline(model_root, local_files_only=False)


def test_load_pipeline_requires_revision_snapshot_and_model_index(tmp_path, monkeypatch):
    runner = _load_runner()
    model_root = tmp_path / "model-cache"
    model_root.mkdir()
    _install_fake_pipeline_modules(monkeypatch)
    monkeypatch.setattr(runner, "PINNED_MODEL_ROOT", model_root)

    with pytest.raises(ValueError, match="snapshot"):
        runner.load_pipeline(model_root, local_files_only=True)

    snapshot = model_root / "snapshots" / runner.MODEL_REVISION
    snapshot.mkdir(parents=True)
    with pytest.raises(ValueError, match="model_index.json"):
        runner.load_pipeline(model_root, local_files_only=True)


def test_load_pipeline_rejects_symlinked_model_cache_root(tmp_path, monkeypatch):
    runner = _load_runner()
    real_root = tmp_path / "real-model-cache"
    _create_snapshot(real_root, runner.MODEL_REVISION)
    linked_root = tmp_path / "linked-model-cache"
    linked_root.symlink_to(real_root, target_is_directory=True)
    _install_fake_pipeline_modules(monkeypatch)
    monkeypatch.setattr(runner, "PINNED_MODEL_ROOT", linked_root)

    with pytest.raises(ValueError, match="symlink"):
        runner.load_pipeline(linked_root, local_files_only=True)


def test_load_pipeline_rejects_symlinked_revision_snapshot(tmp_path, monkeypatch):
    runner = _load_runner()
    model_root = tmp_path / "model-cache"
    snapshots = model_root / "snapshots"
    snapshots.mkdir(parents=True)
    external_snapshot = tmp_path / "external-snapshot"
    external_snapshot.mkdir()
    (external_snapshot / "model_index.json").write_text("{}", encoding="utf-8")
    (snapshots / runner.MODEL_REVISION).symlink_to(
        external_snapshot, target_is_directory=True
    )
    _install_fake_pipeline_modules(monkeypatch)
    monkeypatch.setattr(runner, "PINNED_MODEL_ROOT", model_root)

    with pytest.raises(ValueError, match="snapshot.*symlink|symlink.*snapshot"):
        runner.load_pipeline(model_root, local_files_only=True)


def test_run_jobs_reuses_one_pipeline_and_preserves_generation_parameters(
    tmp_path, monkeypatch
):
    runner = _load_runner()
    _FakeTorch.Generator.created.clear()
    fake_pipeline = _FakePipeline()
    first = _job(prompt="No prompt rewriting, including punctuation: [] {}.")
    first_source, first_review = _prepare_source_bundle(
        tmp_path, first, name="male-approved"
    )
    second = _job("rocketbox_female_adult_01", "A second exact prompt.")
    second_source, second_review = _prepare_source_bundle(
        tmp_path, second, name="female-approved"
    )
    manifests = []
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    _allow_source_reviews(
        monkeypatch,
        runner,
        {first_review: first["asset_id"], second_review: second["asset_id"]},
    )
    monkeypatch.setattr(
        runner,
        "write_candidate_manifest",
        lambda candidate_dir, **kwargs: manifests.append((candidate_dir, kwargs)),
    )

    runner.run_jobs([first, second], tmp_path / "out", fake_pipeline)

    assert len(fake_pipeline.calls) == 2
    assert [call["prompt"] for call in fake_pipeline.calls] == [
        first["prompt"],
        second["prompt"],
    ]
    assert all(call["image"].size == (12, 16) for call in fake_pipeline.calls)
    assert all(call["width"] == 1152 and call["height"] == 1536 for call in fake_pipeline.calls)
    assert all(call["num_inference_steps"] == 28 for call in fake_pipeline.calls)
    assert all(call["guidance_scale"] == 4.0 for call in fake_pipeline.calls)
    assert all(call["max_sequence_length"] == 512 for call in fake_pipeline.calls)
    assert [(generator.device, generator.seed) for generator in _FakeTorch.Generator.created] == [
        ("cuda", 1234),
        ("cuda", 1234),
    ]
    assert [kwargs["prompt"] for _, kwargs in manifests] == [first["prompt"], second["prompt"]]
    assert [kwargs["source_approval_sha256"] for _, kwargs in manifests] == [
        _sha256(first_review),
        _sha256(second_review),
    ]
    assert _sha256(tmp_path / "out" / first["asset_id"] / "source.png") == _sha256(
        first_source
    )
    assert _sha256(tmp_path / "out" / second["asset_id"] / "source.png") == _sha256(
        second_source
    )


@pytest.mark.parametrize("field", ("source_review", "source_image_sha256"))
def test_validate_job_requires_explicit_source_approval_fields(field):
    runner = _load_runner()
    job = _job()
    del job[field]

    with pytest.raises(ValueError, match=field):
        runner.validate_job(job)


@pytest.mark.parametrize("bad_hash", ("a" * 63, "A" * 64, "g" * 64, 123))
def test_validate_job_rejects_noncanonical_source_image_sha256(bad_hash):
    runner = _load_runner()
    job = _job()
    job["source_image_sha256"] = bad_hash

    with pytest.raises(ValueError, match="source_image_sha256.*lowercase.*64|64.*lowercase"):
        runner.validate_job(job)


@pytest.mark.parametrize(
    "asset_id", ("unexpected", "../rocketbox_male_adult_01", "rocketbox_male_adult_01/child")
)
def test_validate_job_rejects_asset_ids_outside_runner_allowlist(asset_id):
    runner = _load_runner()
    job = _job(asset_id=asset_id)

    with pytest.raises(ValueError, match="asset_id"):
        runner.validate_job(job)


def test_run_jobs_rejects_unapproved_source_review(tmp_path, monkeypatch):
    runner = _load_runner()
    job = _job()
    _, review = _prepare_source_bundle(tmp_path, job)
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())

    def reject_source_review(path):
        assert Path(path) == review
        raise RuntimeError("source approval is rejected")

    monkeypatch.setattr(runner, "assert_source_review_approved", reject_source_review)

    with pytest.raises(RuntimeError, match="source approval is rejected"):
        runner.run_jobs([job], tmp_path / "out", _FakePipeline())


def test_run_jobs_rejects_source_review_asset_mismatch(tmp_path, monkeypatch):
    runner = _load_runner()
    job = _job()
    _, review = _prepare_source_bundle(tmp_path, job)
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    _allow_source_reviews(monkeypatch, runner, {review: "rocketbox_female_adult_01"})

    with pytest.raises(RuntimeError, match="does not match"):
        runner.run_jobs([job], tmp_path / "out", _FakePipeline())


def test_run_jobs_rejects_front_image_from_unrelated_directory(tmp_path, monkeypatch):
    runner = _load_runner()
    job = _job()
    source, review = _prepare_source_bundle(tmp_path, job)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    unrelated_source = unrelated / "front.png"
    unrelated_source.write_bytes(source.read_bytes())
    job["source_image"] = str(unrelated_source)
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    _allow_source_reviews(monkeypatch, runner, {review: job["asset_id"]})

    with pytest.raises(ValueError, match="same directory|directly under"):
        runner.run_jobs([job], tmp_path / "out", _FakePipeline())


def test_run_jobs_rejects_wrong_pinned_source_hash(tmp_path, monkeypatch):
    runner = _load_runner()
    job = _job()
    _, review = _prepare_source_bundle(tmp_path, job)
    job["source_image_sha256"] = "0" * 64
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    _allow_source_reviews(monkeypatch, runner, {review: job["asset_id"]})

    with pytest.raises(ValueError, match="source image.*hash"):
        runner.run_jobs([job], tmp_path / "out", _FakePipeline())

    assert not (tmp_path / "out" / job["asset_id"]).exists()


def test_run_jobs_rechecks_copied_source_hash_before_pipeline(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _run_one_valid_job(tmp_path, monkeypatch, runner)
    original_copy = runner.copy_source_image
    pipeline = _FakePipeline()

    def copy_then_tamper(source, destination):
        original_copy(source, destination)
        destination.write_bytes(b"tampered after copy")

    monkeypatch.setattr(runner, "copy_source_image", copy_then_tamper)

    with pytest.raises(ValueError, match="copied source.*hash"):
        runner.run_jobs([job], tmp_path / "out", pipeline)

    assert pipeline.calls == []


@pytest.mark.parametrize(
    ("field", "filename", "message"),
    (
        ("source_image", "pose.png", "front.png"),
        ("source_review", "approval.json", "source_review.json"),
    ),
)
def test_run_jobs_requires_fixed_source_contract_filenames(
    tmp_path, monkeypatch, field, filename, message
):
    runner = _load_runner()
    job = _job()
    source, review = _prepare_source_bundle(tmp_path, job)
    original = source if field == "source_image" else review
    renamed = original.with_name(filename)
    original.rename(renamed)
    job[field] = str(renamed)
    if field == "source_image":
        job["source_image_sha256"] = _sha256(renamed)
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setattr(
        runner,
        "assert_source_review_approved",
        lambda path: pytest.fail("invalid source contract must fail before approval"),
    )

    with pytest.raises(ValueError, match=message):
        runner.run_jobs([job], tmp_path / "out", _FakePipeline())


def test_run_jobs_rejects_symlinked_source_parent(tmp_path, monkeypatch):
    runner = _load_runner()
    job = _job()
    source, review = _prepare_source_bundle(tmp_path, job)
    linked_parent = tmp_path / "linked-approved"
    linked_parent.symlink_to(source.parent, target_is_directory=True)
    job["source_image"] = str(linked_parent / "front.png")
    job["source_review"] = str(linked_parent / "source_review.json")
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setattr(
        runner,
        "assert_source_review_approved",
        lambda path: pytest.fail("approval gate must not receive a symlinked path"),
    )

    with pytest.raises(ValueError, match="symlink"):
        runner.run_jobs([job], tmp_path / "out", _FakePipeline())


@pytest.mark.parametrize("filename", ("front.png", "source_review.json"))
def test_run_jobs_rejects_symlinked_source_contract_file(tmp_path, monkeypatch, filename):
    runner = _load_runner()
    job = _job()
    source, review = _prepare_source_bundle(tmp_path, job)
    contract_path = source if filename == "front.png" else review
    external = tmp_path / f"external-{filename}"
    external.write_bytes(contract_path.read_bytes())
    contract_path.unlink()
    contract_path.symlink_to(external)
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    monkeypatch.setattr(
        runner,
        "assert_source_review_approved",
        lambda path: pytest.fail("approval gate must not receive a symlinked file"),
    )

    with pytest.raises(ValueError, match="symlink|regular file"):
        runner.run_jobs([job], tmp_path / "out", _FakePipeline())


def test_symlinked_output_root_never_writes_outside(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, review = _run_one_valid_job(tmp_path, monkeypatch, runner)
    external = tmp_path / "external-output"
    external.mkdir()
    output_root = tmp_path / "output"
    output_root.symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="output_root.*symlink|symlink.*output_root"):
        runner.run_jobs([job], output_root, _FakePipeline())

    assert list(external.iterdir()) == []
    assert review.is_file()


def test_symlinked_output_parent_never_writes_outside(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _run_one_valid_job(tmp_path, monkeypatch, runner)
    external = tmp_path / "external-parent"
    external.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(external, target_is_directory=True)
    output_root = linked_parent / "new-output"

    with pytest.raises(ValueError, match="output_root.*symlink|symlink.*output_root"):
        runner.run_jobs([job], output_root, _FakePipeline())

    assert list(external.iterdir()) == []


def test_symlinked_candidate_directory_never_writes_outside(tmp_path, monkeypatch):
    runner = _load_runner()
    job, _, _ = _run_one_valid_job(tmp_path, monkeypatch, runner)
    output_root = tmp_path / "output"
    output_root.mkdir()
    external = tmp_path / "external-candidate"
    external.mkdir()
    (output_root / job["asset_id"]).symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="candidate.*symlink|symlink.*candidate"):
        runner.run_jobs([job], output_root, _FakePipeline())

    assert list(external.iterdir()) == []


def test_main_loads_one_pipeline_for_the_entire_job_batch(tmp_path, monkeypatch):
    runner = _load_runner()
    model_root = tmp_path / "pinned-model"
    model_root.mkdir()
    jobs = [_job(), _job("rocketbox_female_adult_01", "Second exact prompt.")]
    args = SimpleNamespace(
        jobs_json=tmp_path / "jobs.json",
        output_root=tmp_path / "out",
        model_root=model_root,
        local_files_only=True,
    )
    pipeline = object()
    loaded = []
    executed = []
    monkeypatch.setattr(runner, "PINNED_MODEL_ROOT", model_root)
    monkeypatch.setattr(runner, "parse_args", lambda: args)
    monkeypatch.setattr(runner, "load_jobs", lambda path: jobs)
    monkeypatch.setattr(
        runner,
        "load_pipeline",
        lambda path, *, local_files_only: loaded.append((path, local_files_only)) or pipeline,
    )
    monkeypatch.setattr(
        runner,
        "run_jobs",
        lambda received_jobs, output_root, received_pipeline: executed.append(
            (received_jobs, output_root, received_pipeline)
        ),
    )

    runner.main()

    assert loaded == [(model_root, True)]
    assert executed == [(jobs, args.output_root, pipeline)]


def test_run_jobs_writes_manifest_only_after_a_valid_candidate_png_exists(
    tmp_path, monkeypatch
):
    runner = _load_runner()
    job, _, _ = _run_one_valid_job(tmp_path, monkeypatch, runner)
    manifest_states = []
    monkeypatch.setattr(
        runner,
        "write_candidate_manifest",
        lambda candidate_dir, **kwargs: manifest_states.append(
            (candidate_dir / "candidate.png").is_file()
            and Image.open(candidate_dir / "candidate.png").format == "PNG"
        ),
    )

    runner.run_jobs([job], tmp_path / "out", _FakePipeline())

    candidate = tmp_path / "out" / job["asset_id"] / "candidate.png"
    assert manifest_states == [True]
    with Image.open(candidate) as image:
        assert image.format == "PNG"
        assert image.size == (1152, 1536)


def test_run_jobs_writes_the_human_reference_contract_manifest(tmp_path, monkeypatch):
    runner = _load_runner()
    job = _job()
    source, review = _prepare_source_bundle(
        tmp_path, job, color="red", size=(1200, 1600)
    )
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    _allow_source_reviews(monkeypatch, runner, {review: job["asset_id"]})

    runner.run_jobs([job], tmp_path / "out", _FakePipeline())

    candidate_dir = tmp_path / "out" / job["asset_id"]
    manifest = json.loads((candidate_dir / "candidate_manifest.json").read_text())
    assert manifest["prompt"] == job["prompt"]
    assert manifest["width"] == 1152
    assert manifest["height"] == 1536
    assert manifest["input_sha256"] == _sha256(source)
    assert manifest["source_approval_sha256"] == _sha256(review)


def test_failed_regeneration_preserves_old_records_but_locks_them_stale(
    tmp_path, monkeypatch
):
    runner = _load_runner()
    first_job = _job()
    _, first_review = _prepare_source_bundle(
        tmp_path, first_job, name="first-approved", color="red"
    )
    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
    _allow_source_reviews(monkeypatch, runner, {first_review: first_job["asset_id"]})
    output_root = tmp_path / "out"
    runner.run_jobs([first_job], output_root, _FakePipeline(color="white"))
    candidate_dir = output_root / first_job["asset_id"]
    record_review(
        candidate_dir,
        "approved",
        "reviewer",
        "old candidate approved",
        expected_snapshot=_current_snapshot(candidate_dir),
    )
    assert_reference_approved(candidate_dir)
    manifest_path = candidate_dir / "candidate_manifest.json"
    review_path = candidate_dir / "reference_review.json"
    old_manifest = manifest_path.read_bytes()
    old_review = review_path.read_bytes()

    second_job = _job()
    _, second_review = _prepare_source_bundle(
        tmp_path, second_job, name="second-approved", color="blue"
    )
    _allow_source_reviews(monkeypatch, runner, {second_review: second_job["asset_id"]})

    def fail_manifest(*args, **kwargs):
        raise OSError("injected manifest failure")

    monkeypatch.setattr(runner, "write_candidate_manifest", fail_manifest)
    with pytest.raises(OSError, match="injected manifest failure"):
        runner.run_jobs([second_job], output_root, _FakePipeline(color="black"))

    assert manifest_path.read_bytes() == old_manifest
    assert review_path.read_bytes() == old_review
    with pytest.raises((ValueError, HumanReferenceNotApproved)):
        assert_reference_approved(candidate_dir)


def test_atomic_png_replacement_leaves_no_partial_file(tmp_path):
    runner = _load_runner()
    target = tmp_path / "candidate.png"
    target.write_bytes(b"old image")

    runner.save_png_atomically(Image.new("RGB", (1152, 1536), "white"), target)

    with Image.open(target) as image:
        assert image.format == "PNG"
        assert image.size == (1152, 1536)
    assert not list(tmp_path.glob(".candidate.png.*.tmp"))


def test_validate_job_requires_1152_by_1536_output():
    runner = _load_runner()
    job = _job()
    job["width"] = 1151

    with pytest.raises(ValueError, match="1152x1536"):
        runner.validate_job(job)
