import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tools" / "i23d_human_bakeoff.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("i23d_human_bakeoff", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_inspect_rgba_input_requires_real_transparency(tmp_path):
    runner = _load_runner()
    opaque = tmp_path / "opaque.png"
    Image.new("RGBA", (8, 12), (255, 0, 0, 255)).save(opaque)

    with pytest.raises(ValueError, match="transparent RGBA"):
        runner.inspect_rgba_input(opaque)

    transparent = tmp_path / "transparent.png"
    image = Image.new("RGBA", (8, 12), (0, 0, 0, 0))
    for x in range(2, 6):
        for y in range(2, 10):
            image.putpixel((x, y), (255, 0, 0, 255))
    image.save(transparent)

    metadata = runner.inspect_rgba_input(transparent)

    assert metadata["mode"] == "RGBA"
    assert metadata["size"] == [8, 12]
    assert metadata["alpha_min"] == 0
    assert metadata["alpha_max"] == 255
    assert len(metadata["sha256"]) == 64


def test_resolve_snapshot_requires_pinned_complete_local_cache(tmp_path):
    runner = _load_runner()
    root = tmp_path / "model"
    revision = "a" * 40
    snapshot = root / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "pipeline.json").write_text("{}", encoding="utf-8")

    assert runner.resolve_snapshot(root, revision, ["pipeline.json"]) == snapshot

    (root / "blobs" / "partial.incomplete").parent.mkdir()
    (root / "blobs" / "partial.incomplete").write_bytes(b"partial")
    with pytest.raises(ValueError, match="incomplete"):
        runner.resolve_snapshot(root, revision, ["pipeline.json"])

    (root / "blobs" / "partial.incomplete").unlink()
    (snapshot / "pipeline.json").unlink()
    with pytest.raises(ValueError, match="required file"):
        runner.resolve_snapshot(root, revision, ["pipeline.json"])


def test_build_runtime_env_forces_offline_canonical_caches_and_backend():
    runner = _load_runner()

    trellis = runner.build_runtime_env("trellis2", gpu=2, base_env={"KEEP": "yes"})
    assert trellis["KEEP"] == "yes"
    assert trellis["CUDA_VISIBLE_DEVICES"] == "2"
    assert trellis["HF_HUB_CACHE"] == "/data/models/hub"
    assert trellis["TORCH_HOME"] == "/data/models/torch"
    assert trellis["HF_HUB_OFFLINE"] == "1"
    assert trellis["TRANSFORMERS_OFFLINE"] == "1"
    assert trellis["ATTN_BACKEND"] == "xformers"

    pixal = runner.build_runtime_env("pixal3d", gpu=3, base_env={})
    assert pixal["ATTN_BACKEND"] == "sdpa"

    with pytest.raises(ValueError, match="backend"):
        runner.build_runtime_env("hunyuan", gpu=0, base_env={})


def test_resolve_backend_assets_uses_only_pinned_model_and_dino(tmp_path, monkeypatch):
    runner = _load_runner()
    model_root = tmp_path / "model"
    dino_root = tmp_path / "dino"
    model_revision = "1" * 40
    dino_revision = "2" * 40
    model_snapshot = model_root / "snapshots" / model_revision
    dino_snapshot = dino_root / "snapshots" / dino_revision
    model_snapshot.mkdir(parents=True)
    dino_snapshot.mkdir(parents=True)
    (model_snapshot / "pipeline.json").write_text("{}", encoding="utf-8")
    (dino_snapshot / "config.json").write_text("{}", encoding="utf-8")
    (dino_snapshot / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(
        runner,
        "MODEL_SPECS",
        {
            "trellis2": {
                "root": model_root,
                "revision": model_revision,
                "required": ["pipeline.json"],
            }
        },
    )
    monkeypatch.setattr(
        runner,
        "DINO_SPEC",
        {
            "root": dino_root,
            "revision": dino_revision,
            "required": ["config.json", "model.safetensors"],
        },
    )

    assets = runner.resolve_backend_assets("trellis2")

    assert assets == {"model": model_snapshot, "dino": dino_snapshot}


def test_patch_trellis_conditioning_disables_rembg_and_pins_dino(tmp_path):
    runner = _load_runner()
    calls = []

    class FakeDino:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    extractors = SimpleNamespace(DinoV3FeatureExtractor=FakeDino)
    rembg = SimpleNamespace(BiRefNet=object())
    dino_snapshot = tmp_path / "dino"

    runner.patch_trellis_conditioning(extractors, rembg, dino_snapshot)
    extractors.DinoV3FeatureExtractor(model_name="facebook/gated")

    assert calls == [{"model_name": str(dino_snapshot)}]
    assert rembg.BiRefNet(model_name="briaai/RMBG-2.0") is None


def test_patch_pixal_conditioning_disables_rembg_and_pins_every_dino(tmp_path):
    runner = _load_runner()
    rembg = SimpleNamespace(BiRefNet=object())
    configs = {
        "ss": {"model_name": "remote-a", "image_size": 512},
        "shape": {"model_name": "remote-b", "image_size": 1024},
    }
    dino_snapshot = tmp_path / "dino"

    runner.patch_pixal_conditioning(rembg, configs, dino_snapshot)

    assert rembg.BiRefNet(model_name="briaai/RMBG-2.0") is None
    assert {config["model_name"] for config in configs.values()} == {
        str(dino_snapshot)
    }


def test_execute_job_dispatches_and_records_reproducible_manifest(tmp_path, monkeypatch):
    runner = _load_runner()
    source = tmp_path / "input.png"
    image = Image.new("RGBA", (8, 12), (0, 0, 0, 0))
    image.putpixel((4, 6), (10, 20, 30, 255))
    image.save(source)
    output = tmp_path / "result.glb"
    model_snapshot = tmp_path / "model-snapshot"
    dino_snapshot = tmp_path / "dino-snapshot"
    calls = []

    monkeypatch.setattr(
        runner,
        "resolve_backend_assets",
        lambda backend: {"model": model_snapshot, "dino": dino_snapshot},
    )

    def fake_generate(**kwargs):
        calls.append(kwargs)
        kwargs["output_path"].write_bytes(b"glb")

    monkeypatch.setattr(runner, "run_trellis2", fake_generate)

    manifest_path = runner.execute_job(
        backend="trellis2",
        image_path=source,
        output_path=output,
        seed=42,
        resolution=1024,
        manual_fov=0.2,
        low_vram=True,
    )

    assert calls == [
        {
            "image_path": source,
            "output_path": output,
            "model_snapshot": model_snapshot,
            "dino_snapshot": dino_snapshot,
            "seed": 42,
            "resolution": 1024,
            "low_vram": True,
        }
    ]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["backend"] == "trellis2"
    assert manifest["input"]["sha256"] == runner.inspect_rgba_input(source)["sha256"]
    assert manifest["output"] == {
        "bytes": 3,
        "path": str(output.resolve()),
        "sha256": "8bc31540bd104a36b11b1061925e19e4bd8be22607025a5bb131d389a3cdbc40",
    }
    assert manifest["parameters"] == {
        "low_vram": True,
        "manual_fov": 0.2,
        "resolution": 1024,
        "seed": 42,
    }


def test_run_trellis2_uses_pinned_conditioning_and_exports_glb(tmp_path, monkeypatch):
    runner = _load_runner()
    source = tmp_path / "input.png"
    Image.new("RGBA", (8, 12), (0, 0, 0, 0)).save(source)
    output = tmp_path / "result.glb"
    calls = []
    mesh = SimpleNamespace(
        vertices="vertices",
        faces="faces",
        attrs="attrs",
        coords="coords",
        layout="layout",
        voxel_size=1.0,
    )

    class FakePipeline:
        low_vram = None

        @classmethod
        def from_pretrained(cls, path):
            calls.append(("load", path))
            return cls()

        def cuda(self):
            calls.append(("cuda",))

        def run(self, image, **kwargs):
            calls.append(("run", image.mode, kwargs))
            return [mesh]

    class FakeGlb:
        def export(self, path, extension_webp):
            calls.append(("export", path, extension_webp))
            Path(path).write_bytes(b"glb")

    class FakePostprocess:
        @staticmethod
        def to_glb(**kwargs):
            calls.append(("to_glb", kwargs))
            return FakeGlb()

    runtime = SimpleNamespace(
        pipeline_class=FakePipeline,
        extractor_module=SimpleNamespace(DinoV3FeatureExtractor=type("Dino", (), {})),
        rembg_module=SimpleNamespace(BiRefNet=object()),
        o_voxel=SimpleNamespace(postprocess=FakePostprocess),
        torch=SimpleNamespace(device=lambda value: value),
    )
    monkeypatch.setattr(runner, "_import_trellis_runtime", lambda: runtime)

    runner.run_trellis2(
        image_path=source,
        output_path=output,
        model_snapshot=tmp_path / "model",
        dino_snapshot=tmp_path / "dino",
        seed=7,
        resolution=1024,
        low_vram=False,
    )

    assert calls[0] == ("load", str(tmp_path / "model"))
    assert ("cuda",) in calls
    run_call = next(call for call in calls if call[0] == "run")
    assert run_call[1:] == (
        "RGBA",
        {"pipeline_type": "1024_cascade", "seed": 7},
    )
    assert calls[-1] == ("export", str(output), True)


def test_run_pixal3d_uses_manual_fov_local_snapshot_and_no_rembg(
    tmp_path, monkeypatch
):
    runner = _load_runner()
    calls = []
    configs = {"ss": {"model_name": "remote"}}
    rembg = SimpleNamespace(BiRefNet=object())

    def fake_run_inference(**kwargs):
        calls.append(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"glb")

    runtime = SimpleNamespace(
        inference=SimpleNamespace(
            IMAGE_COND_CONFIGS=configs,
            run_inference=fake_run_inference,
        ),
        rembg_module=rembg,
    )
    monkeypatch.setattr(runner, "_import_pixal_runtime", lambda: runtime)
    output = tmp_path / "result.glb"

    runner.run_pixal3d(
        image_path=tmp_path / "input.png",
        output_path=output,
        model_snapshot=tmp_path / "model",
        dino_snapshot=tmp_path / "dino",
        seed=9,
        resolution=1536,
        manual_fov=0.2,
        low_vram=True,
    )

    assert configs["ss"]["model_name"] == str(tmp_path / "dino")
    assert rembg.BiRefNet(model_name="briaai/RMBG-2.0") is None
    assert calls == [
        {
            "image_path": str(tmp_path / "input.png"),
            "output_path": str(output),
            "seed": 9,
            "manual_fov": 0.2,
            "model_path": str(tmp_path / "model"),
            "low_vram": True,
            "resolution": 1536,
        }
    ]


def test_main_sets_runtime_environment_before_dispatch(tmp_path, monkeypatch):
    runner = _load_runner()
    source = tmp_path / "input.png"
    output = tmp_path / "output.glb"
    calls = []
    monkeypatch.setattr(
        runner,
        "build_runtime_env",
        lambda backend, gpu, base_env=None: {"I23D_TEST_ENV": f"{backend}-{gpu}"},
    )
    monkeypatch.setattr(
        runner,
        "execute_job",
        lambda **kwargs: calls.append(kwargs) or output.with_suffix(".manifest.json"),
    )
    monkeypatch.delenv("I23D_TEST_ENV", raising=False)

    result = runner.main(
        [
            "--backend",
            "pixal3d",
            "--image",
            str(source),
            "--output",
            str(output),
            "--gpu",
            "3",
            "--resolution",
            "1536",
            "--low-vram",
        ]
    )

    assert result == output.with_suffix(".manifest.json")
    assert runner.os.environ["I23D_TEST_ENV"] == "pixal3d-3"
    assert calls == [
        {
            "backend": "pixal3d",
            "image_path": source,
            "output_path": output,
            "seed": 42,
            "resolution": 1536,
            "manual_fov": 0.2,
            "low_vram": True,
        }
    ]


def test_verify_pinned_file_rejects_missing_or_wrong_hash(tmp_path):
    runner = _load_runner()
    path = tmp_path / "weights.bin"
    path.write_bytes(b"known")
    expected = "7117fff2d0fd294462b3c802b7cb8753579f23f3946b99cf55f38e873f013f10"

    assert runner.verify_pinned_file(path, expected) == path

    with pytest.raises(ValueError, match="SHA-256"):
        runner.verify_pinned_file(path, "0" * 64)
    with pytest.raises(ValueError, match="missing"):
        runner.verify_pinned_file(tmp_path / "missing.bin", expected)
