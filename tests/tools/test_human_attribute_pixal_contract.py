from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from tools import human_attribute_pixal_contract as contract


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _execution_guard_fixture(*, wrapper_ctime_ns: int = 100) -> dict:
    paths = {
        "python_executable": contract.route2_instance.PIXAL_PYTHON_EXECUTABLE,
        "wrapper": Path(contract.PIXAL_WRAPPER_PATH),
        "executor": Path(contract.EXECUTOR_PATH),
    }
    files = {}
    for index, (role, raw_path) in enumerate(paths.items(), start=1):
        path = Path(raw_path).absolute()
        ctime_ns = wrapper_ctime_ns if role == "wrapper" else 100
        files[role] = {
            "path": str(path),
            "sha256": hashlib.sha256(f"guard-file:{role}".encode()).hexdigest(),
            "size_bytes": index,
            "mode": "0444",
            "identity": {
                "device": 1,
                "inode": index,
                "mode": 0o100444,
                "size_bytes": index,
                "mtime_ns": 100,
                "ctime_ns": ctime_ns,
            },
            "parent": {
                "path": str(path.parent),
                "identity": {
                    "device": 1,
                    "inode": 100 + index,
                    "mode": 0o40555,
                    "size_bytes": 1,
                    "mtime_ns": 100,
                    "ctime_ns": 100,
                },
            },
        }
    models = {}
    for role, revision in {
        "pixal": contract.PIXAL3D_REVISION,
        "dino": contract.DINO_REVISION,
    }.items():
        model_contract = contract.route2_instance.MODEL_SNAPSHOT_CONTRACTS[revision]
        models[role] = {
            "path": str(
                (
                    contract.route2_instance.MODEL_ROOT
                    / model_contract["relative_path"]
                ).absolute()
            ),
            "revision": revision,
            "entry_count": 1,
            "metadata_sha256": hashlib.sha256(
                f"guard-model:{role}".encode()
            ).hexdigest(),
        }
    normalized = {
        "schema": "pixal3d_execution_guard_v1",
        "scope": [
            "python_executable",
            "pixal_wrapper",
            "atomic_executor",
            "pixal_model_snapshot_metadata",
            "dino_model_snapshot_metadata",
        ],
        "files": files,
        "models": models,
    }
    return {
        **normalized,
        "guard_sha256": hashlib.sha256(
            contract.common.canonical_json(normalized).encode("utf-8")
        ).hexdigest(),
    }


@pytest.fixture(autouse=True)
def _pin_execution_guard(monkeypatch):
    evidence = _execution_guard_fixture()
    monkeypatch.setattr(
        contract.route2_instance,
        "pixal_execution_guard_evidence",
        lambda: evidence,
    )


def _write_glb(path: Path, value: dict, binary: bytes | None = None) -> None:
    document = json.dumps(value, separators=(",", ":")).encode()
    document += b" " * ((-len(document)) % 4)
    chunks = struct.pack("<II", len(document), 0x4E4F534A) + document
    if binary is not None:
        binary += b"\x00" * ((-len(binary)) % 4)
        chunks += struct.pack("<II", len(binary), 0x004E4942) + binary
    length = 12 + len(chunks)
    path.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, length)
        + chunks
    )


def _encoded_image(mime_type: str, color: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    image_format = {"image/png": "PNG", "image/webp": "WEBP"}[mime_type]
    Image.new("RGB", (2, 2), color).save(buffer, format=image_format)
    return buffer.getvalue()


def _pbr_glb_payload(
    *, mime_type: str = "image/png", image_count: int = 1
) -> tuple[dict, bytes]:
    binary = bytearray()
    buffer_views: list[dict] = []

    def append_buffer_view(payload: bytes) -> int:
        binary.extend(b"\x00" * ((-len(binary)) % 4))
        index = len(buffer_views)
        buffer_views.append(
            {"buffer": 0, "byteOffset": len(binary), "byteLength": len(payload)}
        )
        binary.extend(payload)
        return index

    position_view = append_buffer_view(
        struct.pack(
            "<9f",
            -0.5,
            0.0,
            -0.5,
            0.5,
            0.0,
            -0.5,
            0.0,
            1.0,
            0.5,
        )
    )
    texcoord_view = append_buffer_view(
        struct.pack("<6f", 0.0, 0.0, 1.0, 0.0, 0.5, 1.0)
    )
    image_views = [
        append_buffer_view(
            _encoded_image(mime_type, (20 + index, 40 + index, 60 + index))
        )
        for index in range(image_count)
    ]
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "accessors": [
            {
                "bufferView": position_view,
                "componentType": 5126,
                "count": 3,
                "type": "VEC3",
                "min": [-0.5, 0.0, -0.5],
                "max": [0.5, 1.0, 0.5],
            },
            {
                "bufferView": texcoord_view,
                "componentType": 5126,
                "count": 3,
                "type": "VEC2",
                "min": [0.0, 0.0],
                "max": [1.0, 1.0],
            },
        ],
        "bufferViews": buffer_views,
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {"POSITION": 0, "TEXCOORD_0": 1},
                        "material": 0,
                    }
                ]
            }
        ],
        "materials": [
            {
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicRoughnessTexture": {
                        "index": 1 if image_count > 1 else 0
                    },
                }
            }
        ],
        "textures": [{"source": index} for index in range(image_count)],
        "images": [
            {"bufferView": view, "mimeType": mime_type} for view in image_views
        ],
    }
    return document, bytes(binary)


def _write_pbr_glb(path: Path) -> None:
    document, binary = _pbr_glb_payload()
    _write_glb(path, document, binary)


def _executor_fixture(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    """Build a canonical immutable job while replacing all expensive runtime work."""
    candidate_root = tmp_path / "tall_man"
    candidate_root.mkdir()
    source = candidate_root / "candidate_rgba.png"
    rgba = Image.new("RGBA", (2, 2), (20, 30, 40, 0))
    rgba.putpixel((0, 0), (20, 30, 40, 255))
    rgba.save(source)
    source.chmod(0o444)
    candidate = candidate_root / "candidate_manifest.json"
    contract.common.write_json_immutable_noreplace(
        candidate,
        {
            "schema": "flux2_human_attribute_candidate_v2",
            "case_id": "tall_man",
            "base_asset_id": "rocketbox_male_adult_01",
            "downstream_asset_id": "route2_tall_man_v1",
            "state_classification": "research_candidate",
            "artifacts": {
                "candidate_rgba.png": {
                    "path": str(source),
                    "sha256": _sha(source),
                    "size_bytes": source.stat().st_size,
                }
            },
        },
        RuntimeError,
        "fixture candidate manifest",
    )
    decision = candidate_root.with_name("tall_man.agent_2d_visual_qa.json")
    contract.common.write_json_immutable_noreplace(
        decision,
        {
            "schema": "human_attribute_agent_2d_visual_qa_v1",
            "case_id": "tall_man",
            "base_asset_id": "rocketbox_male_adult_01",
            "downstream_asset_id": "route2_tall_man_v1",
            "status": contract.PASS_STATUS,
            "reviewer_kind": "agent",
            "user_acceptance": "pending_user_review",
        },
        RuntimeError,
        "fixture agent decision",
    )
    output_root = tmp_path / "pixal"
    output_root.mkdir()
    asset_id = "route2_tall_man_v1"
    public_glb = output_root / asset_id / "canary_1024_seed42.glb"
    wrapper = Path(contract.PIXAL_WRAPPER_PATH).resolve()
    argv = [
        str(wrapper),
        "--backend",
        "pixal3d",
        "--image",
        str(source),
        "--output",
        str(public_glb),
        "--gpu",
        "3",
        "--seed",
        "42",
        "--resolution",
        "1024",
        "--manual-fov",
        "0.2",
        "--low-vram",
    ]
    job_payload = {
        "schema": "pixal3d_human_attribute_job_v1",
        "case_id": "tall_man",
        "asset_id": asset_id,
        "base_asset_id": "rocketbox_male_adult_01",
        "state_classification": "research_candidate",
        "input_rgba": {
            "path": str(source),
            "sha256": _sha(source),
            "size_bytes": source.stat().st_size,
            "mode": "RGBA",
            "size": [2, 2],
            "alpha_min": 0,
            "alpha_max": 255,
        },
        "candidate_manifest": {
            "path": str(candidate),
            "sha256": _sha(candidate),
            "size_bytes": candidate.stat().st_size,
        },
        "agent_2d_decision": {
            "path": str(decision),
            "sha256": _sha(decision),
            "size_bytes": decision.stat().st_size,
            "status": contract.PASS_STATUS,
        },
        "model_revision": contract.PIXAL3D_REVISION,
        "dino_revision": contract.DINO_REVISION,
        "parameters": {
            "seed": 42,
            "manual_fov": 0.2,
            "resolution": 1024,
            "low_vram": True,
        },
        "wrapper": {
            "path": str(wrapper),
            "sha256": _sha(wrapper),
            "size_bytes": wrapper.stat().st_size,
        },
        "output_glb": str(public_glb),
        "output_manifest": str(public_glb.with_suffix(".manifest.json")),
        "output_policy": "atomic_no_replace",
        "executor": {
            "kind": "atomic_pixal3d_executor_v1",
            "argv": argv,
            "execution_authorized": True,
            "atomic_no_replace": True,
            "path": str(contract.EXECUTOR_PATH),
            "sha256": _sha(contract.EXECUTOR_PATH),
            "size_bytes": contract.EXECUTOR_PATH.stat().st_size,
        },
    }
    job = output_root / f"{asset_id}.pixal_job.json"
    contract.common.write_json_immutable_noreplace(
        job, job_payload, RuntimeError, "fixture Pixal job"
    )
    snapshot = {
        "payload": job_payload,
        "job_record": {
            "path": str(job),
            "sha256": _sha(job),
            "size_bytes": job.stat().st_size,
        },
        "model_evidence": {
            "pixal": {
                "path": "/data/models/pixal",
                "revision": contract.PIXAL3D_REVISION,
                "file_count": 1,
                "inventory_sha256": "c" * 64,
                "license": {"sha256": "d" * 64},
            },
            "dino": {
                "path": "/data/models/dino",
                "revision": contract.DINO_REVISION,
                "file_count": 1,
                "inventory_sha256": "e" * 64,
                "license": {"sha256": "f" * 64},
            },
        },
    }
    environment = {
        "python_executable": str(contract.route2_instance.PIXAL_PYTHON_EXECUTABLE),
        "python_executable_record": {
            "path": str(contract.route2_instance.PIXAL_PYTHON_EXECUTABLE),
            "sha256": "1" * 64,
            "size_bytes": 1,
            "mode": "0775",
        },
        "python_version": "3.10.20",
        "torch_version": "2.6.0+cu124",
        "cuda_version": "12.4",
        "cuda_visible_devices": "3",
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_device_name": "simulated-gpu",
        "cuda_device_uuid": "simulated-uuid",
        "attention_backend": "sdpa",
        "hf_hub_cache": "/data/models/hub",
        "hf_hub_offline": "1",
        "transformers_offline": "1",
        "torch_home": "/data/models/torch",
        "opencv_io_enable_openexr": "1",
        "pytorch_cuda_alloc_conf": "expandable_segments:True",
    }
    return job, public_glb, snapshot, environment


def _candidate_bundle(tmp_path: Path, *, decision_status: str) -> tuple[Path, Path]:
    root = tmp_path / "tall_man"
    root.mkdir()
    rgba = root / "candidate_rgba.png"
    image = Image.new("RGBA", (16, 16), (30, 40, 50, 0))
    for x in range(5, 11):
        for y in range(2, 15):
            image.putpixel((x, y), (100, 110, 120, 255))
    image.save(rgba)
    manifest = root / "candidate_manifest.json"
    payload = {
        "schema": "flux2_human_attribute_candidate_v2",
        "case_id": "tall_man",
        "base_asset_id": "rocketbox_male_adult_01",
        "downstream_asset_id": "route2_tall_man_v1",
        "state_classification": "research_candidate",
        "artifacts": {
            "candidate_rgba.png": {
                "path": str(rgba),
                "sha256": _sha(rgba),
                "size_bytes": rgba.stat().st_size,
            }
        },
    }
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    decision = root.with_name("tall_man.agent_2d_visual_qa.json")
    decision.write_text(
        json.dumps(
            {
                "schema": "human_attribute_agent_2d_visual_qa_v1",
                "case_id": "tall_man",
                "status": decision_status,
                "reviewer_kind": "agent",
                "candidate_manifest_sha256": _sha(manifest),
                "user_acceptance": "pending_user_review",
            }
        ),
        encoding="utf-8",
    )
    decision.chmod(0o444)
    return manifest, decision


def _pin_wrapper(monkeypatch: pytest.MonkeyPatch, wrapper: Path) -> None:
    monkeypatch.setattr(contract, "PIXAL_WRAPPER_PATH", wrapper)
    monkeypatch.setattr(contract, "PIXAL_WRAPPER_SHA256", _sha(wrapper))


def _authorize(monkeypatch: pytest.MonkeyPatch, manifest: Path, decision: Path) -> None:
    monkeypatch.setattr(
        contract.attribute_review,
        "decision_path",
        lambda bundle: decision,
    )
    monkeypatch.setattr(
        contract.attribute_review,
        "assert_agent_2d_qa_passed",
        lambda bundle: {
            "status": "agent_qa_passed_pending_user_acceptance",
            "case_id": "tall_man",
            "base_asset_id": "rocketbox_male_adult_01",
            "downstream_asset_id": "route2_tall_man_v1",
            "snapshot": {"candidate_manifest_sha256": _sha(manifest)},
        },
    )


def test_passed_agent_decision_builds_hash_locked_pixal_job_without_running_pixal(tmp_path, monkeypatch):
    manifest, decision = _candidate_bundle(
        tmp_path, decision_status="agent_qa_passed_pending_user_acceptance"
    )
    wrapper = tmp_path / "i23d_human_bakeoff.py"
    wrapper.write_text("# pinned wrapper\n", encoding="utf-8")
    _pin_wrapper(monkeypatch, wrapper)
    _authorize(monkeypatch, manifest, decision)
    output_root = tmp_path / "pixal"
    output_root.mkdir()

    payload = contract.build_pixal_job(
        candidate_manifest=manifest,
        agent_decision=decision,
        output_root=output_root,
        wrapper_path=wrapper,
    )

    assert payload["schema"] == "pixal3d_human_attribute_job_v1"
    assert payload["asset_id"] == "route2_tall_man_v1"
    assert payload["base_asset_id"] == "rocketbox_male_adult_01"
    assert payload["model_revision"] == contract.PIXAL3D_REVISION
    assert payload["dino_revision"] == contract.DINO_REVISION
    assert payload["parameters"] == {
        "seed": 42,
        "manual_fov": 0.2,
        "resolution": 1024,
        "low_vram": True,
    }
    assert payload["input_rgba"]["sha256"] == _sha(
        manifest.parent / "candidate_rgba.png"
    )
    assert payload["wrapper"]["sha256"] == _sha(wrapper)
    assert payload["output_glb"].endswith("route2_tall_man_v1/canary_1024_seed42.glb")
    assert payload["executor"]["argv"][-1] == "--low-vram"
    assert payload["executor"]["argv"][payload["executor"]["argv"].index("--gpu") + 1] == "3"
    assert payload["executor"] == {
        "kind": "atomic_pixal3d_executor_v1",
        "argv": payload["executor"]["argv"],
        "execution_authorized": True,
        "atomic_no_replace": True,
        "path": str(contract.EXECUTOR_PATH),
        "sha256": _sha(contract.EXECUTOR_PATH),
        "size_bytes": contract.EXECUTOR_PATH.stat().st_size,
    }
    assert not Path(payload["output_glb"]).exists()


@pytest.mark.parametrize(
    "status", ["pending_agent_2d_visual_qa", "rejected", "approved", "user_approved"]
)
def test_pixal_contract_rejects_every_non_agent_pending_user_pass(tmp_path, status, monkeypatch):
    manifest, decision = _candidate_bundle(tmp_path, decision_status=status)
    wrapper = tmp_path / "i23d_human_bakeoff.py"
    wrapper.write_text("# wrapper\n", encoding="utf-8")
    _pin_wrapper(monkeypatch, wrapper)
    output_root = tmp_path / "pixal"
    output_root.mkdir()

    with pytest.raises(contract.PixalContractError, match="agent 2D QA"):
        contract.build_pixal_job(
            candidate_manifest=manifest,
            agent_decision=decision,
            output_root=output_root,
            wrapper_path=wrapper,
        )


def test_pixal_job_publication_is_atomic_no_replace(tmp_path, monkeypatch):
    manifest, decision = _candidate_bundle(
        tmp_path, decision_status="agent_qa_passed_pending_user_acceptance"
    )
    wrapper = tmp_path / "i23d_human_bakeoff.py"
    wrapper.write_text("# wrapper\n", encoding="utf-8")
    _pin_wrapper(monkeypatch, wrapper)
    _authorize(monkeypatch, manifest, decision)
    output_root = tmp_path / "pixal"
    output_root.mkdir()
    payload = contract.build_pixal_job(
        candidate_manifest=manifest,
        agent_decision=decision,
        output_root=output_root,
        wrapper_path=wrapper,
    )
    destination = tmp_path / "pixal_job.json"

    contract.publish_pixal_job(payload, destination)

    assert destination.stat().st_mode & 0o777 == 0o444
    with pytest.raises(FileExistsError):
        contract.publish_pixal_job(payload, destination)


def test_pixal_job_publication_rolls_back_postlink_directory_fsync_interrupt(
    tmp_path, monkeypatch
):
    destination = tmp_path / "pixal_job.json"
    real_fsync = contract.common.os.fsync
    interrupted = False

    def interrupt_first_directory_fsync(descriptor):
        nonlocal interrupted
        if stat.S_ISDIR(os.fstat(descriptor).st_mode) and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("synthetic Pixal job directory fsync interrupt")
        return real_fsync(descriptor)

    monkeypatch.setattr(
        contract.common.os,
        "fsync",
        interrupt_first_directory_fsync,
    )

    with pytest.raises(KeyboardInterrupt, match="directory fsync interrupt"):
        contract.publish_pixal_job({"schema": "fixture"}, destination)

    assert interrupted is True
    assert not destination.exists()
    assert not list(tmp_path.glob(f".{destination.name}.*.staging"))


def test_pixal_contract_rejects_traversal_asset_id_even_with_forged_agent_payload(tmp_path, monkeypatch):
    manifest, decision = _candidate_bundle(
        tmp_path, decision_status="agent_qa_passed_pending_user_acceptance"
    )
    payload = json.loads(manifest.read_text())
    payload["downstream_asset_id"] = "../escape"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    wrapper = tmp_path / "i23d_human_bakeoff.py"
    wrapper.write_text("# wrapper\n", encoding="utf-8")
    _pin_wrapper(monkeypatch, wrapper)
    _authorize(monkeypatch, manifest, decision)

    with pytest.raises(contract.PixalContractError, match="asset lineage"):
        contract.build_pixal_job(
            candidate_manifest=manifest,
            agent_decision=decision,
            output_root=tmp_path,
            wrapper_path=wrapper,
        )


def test_pixal_contract_rejects_any_wrapper_other_than_pinned_executor(tmp_path, monkeypatch):
    manifest, decision = _candidate_bundle(
        tmp_path, decision_status="agent_qa_passed_pending_user_acceptance"
    )
    pinned = tmp_path / "pinned.py"
    pinned.write_text("# pinned\n", encoding="utf-8")
    wrong = tmp_path / "wrong.py"
    wrong.write_text("# wrong\n", encoding="utf-8")
    _pin_wrapper(monkeypatch, pinned)
    _authorize(monkeypatch, manifest, decision)

    with pytest.raises(contract.PixalContractError, match="pinned Pixal wrapper"):
        contract.build_pixal_job(
            candidate_manifest=manifest,
            agent_decision=decision,
            output_root=tmp_path,
            wrapper_path=wrong,
        )


def test_atomic_pixal_executor_stages_and_publishes_complete_attempt_without_gpu(
    tmp_path, monkeypatch
):
    source = tmp_path / "candidate_rgba.png"
    Image.new("RGBA", (2, 2), (20, 30, 40, 0)).save(source)
    wrapper = tmp_path / "i23d_human_bakeoff.py"
    wrapper.write_text("# simulated pinned generator\n", encoding="utf-8")
    output_root = tmp_path / "pixal"
    output_root.mkdir()
    asset_id = "route2_tall_man_v1"
    public_glb = output_root / asset_id / "canary_1024_seed42.glb"
    argv = [
        str(wrapper),
        "--backend",
        "pixal3d",
        "--image",
        str(source),
        "--output",
        str(public_glb),
        "--gpu",
        "3",
        "--seed",
        "42",
        "--resolution",
        "1024",
        "--manual-fov",
        "0.2",
        "--low-vram",
    ]
    job_payload = {
        "schema": "pixal3d_human_attribute_job_v1",
        "case_id": "tall_man",
        "asset_id": asset_id,
        "base_asset_id": "rocketbox_male_adult_01",
        "state_classification": "research_candidate",
        "input_rgba": {"path": str(source), "sha256": _sha(source), "size_bytes": source.stat().st_size, "mode": "RGBA", "size": [2, 2], "alpha_min": 0, "alpha_max": 255},
        "candidate_manifest": {"path": str(tmp_path / "candidate.json"), "sha256": "a" * 64, "size_bytes": 1},
        "agent_2d_decision": {"path": str(tmp_path / "decision.json"), "sha256": "b" * 64, "size_bytes": 1, "status": contract.PASS_STATUS},
        "model_revision": contract.PIXAL3D_REVISION,
        "dino_revision": contract.DINO_REVISION,
        "parameters": {"seed": 42, "manual_fov": 0.2, "resolution": 1024, "low_vram": True},
        "wrapper": {"path": str(wrapper), "sha256": _sha(wrapper), "size_bytes": wrapper.stat().st_size},
        "output_glb": str(public_glb),
        "output_manifest": str(public_glb.with_suffix(".manifest.json")),
        "output_policy": "atomic_no_replace",
        "executor": {"kind": "atomic_pixal3d_executor_v1", "argv": argv, "execution_authorized": True, "atomic_no_replace": True, "path": str(contract.EXECUTOR_PATH), "sha256": _sha(contract.EXECUTOR_PATH), "size_bytes": contract.EXECUTOR_PATH.stat().st_size},
    }
    job = output_root / f"{asset_id}.pixal_job.json"
    job.write_text(json.dumps(job_payload), encoding="utf-8")
    job.chmod(0o444)
    model_evidence = {
        "pixal": {"path": "/data/models/pixal", "revision": contract.PIXAL3D_REVISION, "inventory_sha256": "c" * 64, "license": {"sha256": "d" * 64}},
        "dino": {"path": "/data/models/dino", "revision": contract.DINO_REVISION, "inventory_sha256": "e" * 64, "license": {"sha256": "f" * 64}},
    }
    snapshot = {
        "payload": job_payload,
        "job_record": {"path": str(job), "sha256": _sha(job), "size_bytes": job.stat().st_size},
        "model_evidence": model_evidence,
    }
    monkeypatch.setattr(contract, "reauthenticate_pixal_job", lambda path: snapshot)
    environment = {
        "python_executable": str(contract.route2_instance.PIXAL_PYTHON_EXECUTABLE),
        "python_executable_record": {"path": str(contract.route2_instance.PIXAL_PYTHON_EXECUTABLE), "sha256": "1" * 64, "size_bytes": 1, "mode": "0775"},
        "python_version": "3.10.20",
        "torch_version": "2.6.0+cu124",
        "cuda_version": "12.4",
        "cuda_visible_devices": "3",
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_device_name": "simulated-gpu",
        "cuda_device_uuid": "simulated-uuid",
        "attention_backend": "sdpa",
        "hf_hub_cache": "/data/models/hub",
        "hf_hub_offline": "1",
        "transformers_offline": "1",
        "torch_home": "/data/models/torch",
        "opencv_io_enable_openexr": "1",
        "pytorch_cuda_alloc_conf": "expandable_segments:True",
    }
    monkeypatch.setattr(contract, "probe_pixal_runtime", lambda: environment)
    commands = []

    def fake_subprocess(command, **kwargs):
        commands.append(command)
        starts = list(
            (output_root / ".attempts" / asset_id).glob("*.started.json")
        )
        assert len(starts) == 1
        assert starts[0].stat().st_mode & 0o777 == 0o444
        assert json.loads(starts[0].read_text())["status"] == "started"
        staged_glb = Path(command[command.index("--output") + 1])
        assert staged_glb != public_glb
        _write_pbr_glb(staged_glb)
        staged_manifest = staged_glb.with_suffix(".manifest.json")
        staged_manifest.write_text(
            json.dumps(
                {
                    "backend": "pixal3d",
                    "input": {"path": str(source), "sha256": _sha(source)},
                    "output": {"path": str(staged_glb), "sha256": _sha(staged_glb), "bytes": staged_glb.stat().st_size},
                    "model": {"revision": contract.PIXAL3D_REVISION},
                    "dino": {"revision": contract.DINO_REVISION},
                    "parameters": job_payload["parameters"],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            stdout=f"simulated inference\n{staged_manifest}\n",
            stderr="",
        )

    attempt_path = contract.execute_atomic_pixal_job(
        job,
        subprocess_runner=fake_subprocess,
    )

    assert commands[0][0] == str(contract.route2_instance.PIXAL_PYTHON_EXECUTABLE)
    assert attempt_path == public_glb.parent / "pixal_attempt.json"
    assert not list(output_root.glob("*.staging"))
    assert all(path.stat().st_mode & 0o777 == 0o444 for path in public_glb.parent.iterdir())
    attempt = json.loads(attempt_path.read_text())
    assert set(attempt) == set(contract.route2_instance.PIXAL_ATTRIBUTE_ATTEMPT_FIELDS)
    assert attempt["argv"] == argv
    assert attempt["preflight_reauthenticated"] is True
    assert attempt["postflight_reauthenticated"] is True
    assert attempt["execution_guard"] == {
        "before": _execution_guard_fixture(),
        "after": _execution_guard_fixture(),
        "unchanged": True,
    }
    assert attempt["model_inventory"]["pixal_snapshot_inventory_sha256"] == "c" * 64
    assert attempt["executor"] == {
        key: job_payload["executor"][key] for key in ("path", "sha256", "size_bytes")
    }
    assert attempt["start_ledger"]["path"].endswith(".started.json")
    execution_log = json.loads(Path(attempt["execution_log"]["path"]).read_text())
    assert execution_log["stdout"].strip().splitlines()[-1] == execution_log[
        "success_sentinel"
    ]


def test_atomic_executor_rejects_transient_execution_guard_swap_restore(
    tmp_path, monkeypatch
):
    job, public_glb, snapshot, environment = _executor_fixture(tmp_path)
    monkeypatch.setattr(contract, "reauthenticate_pixal_job", lambda path: snapshot)
    monkeypatch.setattr(contract, "probe_pixal_runtime", lambda: environment)
    guard_before = _execution_guard_fixture(wrapper_ctime_ns=100)
    guard_after = _execution_guard_fixture(wrapper_ctime_ns=200)
    assert (
        guard_before["files"]["wrapper"]["sha256"]
        == guard_after["files"]["wrapper"]["sha256"]
    )
    assert guard_before["guard_sha256"] != guard_after["guard_sha256"]
    guards = iter((guard_before, guard_after))
    guard_calls = []

    def next_guard():
        guard_calls.append(len(guard_calls) + 1)
        return next(guards)

    monkeypatch.setattr(
        contract.route2_instance,
        "pixal_execution_guard_evidence",
        next_guard,
    )

    def successful_subprocess(command, **kwargs):
        staged_glb = Path(command[command.index("--output") + 1])
        _write_pbr_glb(staged_glb)
        staged_manifest = staged_glb.with_suffix(".manifest.json")
        staged_manifest.write_text(
            json.dumps(
                {
                    "backend": "pixal3d",
                    "input": {
                        "path": snapshot["payload"]["input_rgba"]["path"],
                        "sha256": snapshot["payload"]["input_rgba"]["sha256"],
                    },
                    "output": {
                        "path": str(staged_glb),
                        "sha256": _sha(staged_glb),
                        "bytes": staged_glb.stat().st_size,
                    },
                    "model": {"revision": contract.PIXAL3D_REVISION},
                    "dino": {"revision": contract.DINO_REVISION},
                    "parameters": snapshot["payload"]["parameters"],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            stdout=f"successful generator\n{staged_manifest}\n",
            stderr="",
        )

    with pytest.raises(contract.PixalContractError, match="execution guard changed"):
        contract.execute_atomic_pixal_job(
            job,
            subprocess_runner=successful_subprocess,
        )

    assert guard_calls == [1, 2]
    assert not public_glb.parent.exists()
    bundle = next(
        (
            public_glb.parent.parent
            / ".failed_attempts"
            / "route2_tall_man_v1"
        ).iterdir()
    )
    failure = contract.route2_instance.validate_pixal_attribute_failure_bundle(bundle)
    assert failure["failure_stage"] == "execution_guard"
    start_path = next(
        (
            public_glb.parent.parent
            / ".attempts"
            / "route2_tall_man_v1"
        ).glob("*.started.json")
    )
    assert json.loads(start_path.read_text())["execution_guard_before"] == guard_before


def test_staged_pixal_readback_rejects_mesh_without_material_or_pbr(tmp_path):
    rgba = tmp_path / "input.png"
    Image.new("RGBA", (2, 2), (1, 2, 3, 255)).save(rgba)
    glb = tmp_path / "empty_material.glb"
    _write_glb(
        glb,
        {
            "asset": {"version": "2.0"},
            "accessors": [
                {
                    "componentType": 5126,
                    "count": 3,
                    "type": "VEC3",
                    "min": [0.0, 0.0, 0.0],
                    "max": [1.0, 1.0, 1.0],
                }
            ],
            "meshes": [
                {"primitives": [{"attributes": {"POSITION": 0}}]}
            ],
        },
    )

    with pytest.raises(contract.PixalContractError, match="material|PBR"):
        contract.validate_staged_pixal_glb(
            glb,
            staging=tmp_path,
            input_rgba=rgba,
        )


def test_staged_pixal_readback_rejects_missing_glb_bin(tmp_path):
    rgba = tmp_path / "input.png"
    Image.new("RGBA", (2, 2), (1, 2, 3, 255)).save(rgba)
    glb = tmp_path / "missing_bin.glb"
    document, _ = _pbr_glb_payload()
    _write_glb(glb, document)

    with pytest.raises(contract.PixalContractError, match="packed GLB BIN"):
        contract.validate_staged_pixal_glb(
            glb,
            staging=tmp_path,
            input_rgba=rgba,
        )


def test_staged_pixal_readback_rejects_out_of_range_buffer_view(tmp_path):
    rgba = tmp_path / "input.png"
    Image.new("RGBA", (2, 2), (1, 2, 3, 255)).save(rgba)
    glb = tmp_path / "out_of_range_buffer_view.glb"
    document, binary = _pbr_glb_payload()
    document["bufferViews"][2]["byteOffset"] = len(binary) + 4
    _write_glb(glb, document, binary)

    with pytest.raises(contract.PixalContractError, match="bufferView range"):
        contract.validate_staged_pixal_glb(
            glb,
            staging=tmp_path,
            input_rgba=rgba,
        )


def test_staged_pixal_readback_rejects_invalid_base64_image(tmp_path):
    rgba = tmp_path / "input.png"
    Image.new("RGBA", (2, 2), (1, 2, 3, 255)).save(rgba)
    glb = tmp_path / "invalid_base64.glb"
    document, binary = _pbr_glb_payload()
    document["images"][0] = {"uri": "data:image/png;base64,%%%"}
    _write_glb(glb, document, binary)

    with pytest.raises(contract.PixalContractError, match="invalid base64 data URI"):
        contract.validate_staged_pixal_glb(
            glb,
            staging=tmp_path,
            input_rgba=rgba,
        )


def test_failure_bundle_is_recursively_readonly_fsynced_and_inventory_bound(tmp_path):
    evidence = tmp_path / "attempt_001"
    nested = evidence / "runtime"
    nested.mkdir(parents=True)
    (evidence / "partial.glb").write_bytes(b"partial")
    (nested / "stderr.txt").write_text("failure", encoding="utf-8")

    manifest = contract._seal_failure_bundle(
        evidence,
        payload={
            "schema": "pixal3d_human_attribute_failure_bundle_v1",
            "attempt_id": "attempt_001",
            "status": "failed",
            "case_id": "tall_man",
            "asset_id": "route2_tall_man_v1",
            "base_avatar_id": "rocketbox_male_adult_01",
            "job": {"path": "/job", "sha256": "a" * 64, "size_bytes": 1},
            "start_ledger": {"path": "/start", "sha256": "b" * 64, "size_bytes": 1},
            "failure_stage": "output_readback",
            "error": {"type": "RuntimeError", "message": "synthetic"},
            "returncode": 1,
        },
    )

    payload = json.loads(manifest.read_text())
    assert [item["relative_path"] for item in payload["artifacts"]] == [
        "partial.glb",
        "runtime/stderr.txt",
    ]
    assert all(item["mode"] == "0444" for item in payload["artifacts"])
    assert all(
        path.stat().st_mode & 0o777 == 0o444
        for path in evidence.rglob("*")
        if path.is_file()
    )
    assert all(
        path.stat().st_mode & 0o777 == 0o555
        for path in [evidence, nested]
    )


def test_failure_bundle_owner_readback_and_started_ledger_tamper_rejection(tmp_path):
    job, output_glb, snapshot, _ = _executor_fixture(tmp_path)
    job_payload = snapshot["payload"]
    job_record = snapshot["job_record"]
    output_root = output_glb.parent.parent
    asset_id = job_payload["asset_id"]
    attempt_id = "attempt_001"
    executor_record = {
        key: job_payload["executor"][key]
        for key in ("path", "sha256", "size_bytes")
    }
    argv = job_payload["executor"]["argv"]
    staging = output_root / f".{asset_id}.{attempt_id}.fixture.staging"
    start_dir = output_root / ".attempts" / asset_id
    start_dir.mkdir(parents=True)
    start = start_dir / f"{attempt_id}.started.json"
    started_at = "2026-07-12T00:00:00Z"
    start_payload = {
        "schema": "pixal3d_human_attribute_attempt_start_v1",
        "attempt_id": attempt_id,
        "status": "started",
        "case_id": "tall_man",
        "asset_id": asset_id,
        "base_avatar_id": "rocketbox_male_adult_01",
        "job": job_record,
        "executor": executor_record,
        "execution_guard_before": _execution_guard_fixture(),
        "argv": argv,
        "started_at_utc": started_at,
        "staging": {"path": str(staging), "created": True},
        "publication_policy": "atomic_no_replace",
    }
    contract.common.write_json_immutable_noreplace(
        start, start_payload, RuntimeError, "fixture Pixal start"
    )
    start_record = {"path": str(start), "sha256": _sha(start), "size_bytes": start.stat().st_size}
    evidence = output_root / ".failed_attempts" / asset_id / attempt_id
    evidence.mkdir(parents=True)
    (evidence / "partial.glb").write_bytes(b"partial")
    contract._seal_failure_bundle(
        evidence,
        payload={
            "schema": "pixal3d_human_attribute_failure_bundle_v1",
            "attempt_id": attempt_id,
            "status": "failed",
            "case_id": "tall_man",
            "asset_id": asset_id,
            "base_avatar_id": "rocketbox_male_adult_01",
            "job": job_record,
            "start_ledger": start_record,
            "failure_stage": "output_readback",
            "error": {"type": "RuntimeError", "message": "synthetic"},
            "returncode": 1,
        },
    )

    validated = contract.route2_instance.validate_pixal_attribute_failure_bundle(
        evidence
    )
    assert validated["artifacts"][0]["relative_path"] == "partial.glb"

    start.chmod(0o644)
    start_payload["attempt_id"] = "attempt_tampered"
    start.write_text(json.dumps(start_payload), encoding="utf-8")
    start.chmod(0o444)
    with pytest.raises(
        contract.route2_instance.InstanceContractError,
        match="start ledger descriptor changed|inconsistent",
    ):
        contract.route2_instance.validate_pixal_attribute_failure_bundle(evidence)


def test_runtime_probe_interrupt_preserves_started_attempt_as_owner_valid_bundle(
    tmp_path, monkeypatch
):
    job, public_glb, snapshot, _ = _executor_fixture(tmp_path)
    monkeypatch.setattr(contract, "reauthenticate_pixal_job", lambda path: snapshot)

    def interrupt_runtime_probe():
        raise KeyboardInterrupt("synthetic runtime-probe interrupt")

    monkeypatch.setattr(contract, "probe_pixal_runtime", interrupt_runtime_probe)

    with pytest.raises(KeyboardInterrupt, match="runtime-probe interrupt"):
        contract.execute_atomic_pixal_job(job)

    output_root = public_glb.parent.parent
    assert not public_glb.parent.exists()
    assert not list(output_root.glob("*.staging"))
    bundles = list(
        (output_root / ".failed_attempts" / "route2_tall_man_v1").iterdir()
    )
    assert len(bundles) == 1
    assert bundles[0].stat().st_mode & 0o777 == 0o555
    failure = contract.route2_instance.validate_pixal_attribute_failure_bundle(
        bundles[0]
    )
    assert failure["failure_stage"] == "runtime_probe"
    assert failure["error"]["type"] == "KeyboardInterrupt"


def test_nonzero_subprocess_preserves_stdout_stderr_log_in_failure_inventory(
    tmp_path, monkeypatch
):
    job, public_glb, snapshot, environment = _executor_fixture(tmp_path)
    monkeypatch.setattr(contract, "reauthenticate_pixal_job", lambda path: snapshot)
    monkeypatch.setattr(contract, "probe_pixal_runtime", lambda: environment)

    def failed_subprocess(command, **kwargs):
        return SimpleNamespace(
            returncode=7,
            stdout="partial generator stdout\n",
            stderr="deterministic generator failure\n",
        )

    with pytest.raises(contract.PixalContractError, match="returncode 7"):
        contract.execute_atomic_pixal_job(job, subprocess_runner=failed_subprocess)

    output_root = public_glb.parent.parent
    bundle = next(
        (output_root / ".failed_attempts" / "route2_tall_man_v1").iterdir()
    )
    failure = contract.route2_instance.validate_pixal_attribute_failure_bundle(bundle)
    records = {item["relative_path"]: item for item in failure["artifacts"]}
    assert records["execution.log"]["mode"] == "0444"
    execution = json.loads((bundle / "execution.log").read_text())
    assert execution["returncode"] == 7
    assert execution["stdout"] == "partial generator stdout\n"
    assert execution["stderr"] == "deterministic generator failure\n"
    assert failure["failure_stage"] == "subprocess_returncode"


def test_subprocess_keyboard_interrupt_is_sealed_before_it_is_reraised(
    tmp_path, monkeypatch
):
    job, public_glb, snapshot, environment = _executor_fixture(tmp_path)
    monkeypatch.setattr(contract, "reauthenticate_pixal_job", lambda path: snapshot)
    monkeypatch.setattr(contract, "probe_pixal_runtime", lambda: environment)

    def interrupted_subprocess(command, **kwargs):
        raise KeyboardInterrupt("synthetic subprocess interrupt")

    with pytest.raises(KeyboardInterrupt, match="subprocess interrupt"):
        contract.execute_atomic_pixal_job(
            job, subprocess_runner=interrupted_subprocess
        )

    bundle = next(
        (
            public_glb.parent.parent
            / ".failed_attempts"
            / "route2_tall_man_v1"
        ).iterdir()
    )
    failure = contract.route2_instance.validate_pixal_attribute_failure_bundle(bundle)
    assert failure["failure_stage"] == "subprocess"
    assert failure["error"]["type"] == "KeyboardInterrupt"


def test_success_sentinel_must_be_an_exact_unique_final_nonblank_line(
    tmp_path, monkeypatch
):
    job, public_glb, snapshot, environment = _executor_fixture(tmp_path)
    monkeypatch.setattr(contract, "reauthenticate_pixal_job", lambda path: snapshot)
    monkeypatch.setattr(contract, "probe_pixal_runtime", lambda: environment)

    def whitespace_sentinel(command, **kwargs):
        staged_glb = Path(command[command.index("--output") + 1])
        _write_pbr_glb(staged_glb)
        staged_manifest = staged_glb.with_suffix(".manifest.json")
        staged_manifest.write_text("{}", encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout=f"generator output\n {staged_manifest} \n",
            stderr="",
        )

    with pytest.raises(contract.PixalContractError, match="unique.*sentinel"):
        contract.execute_atomic_pixal_job(job, subprocess_runner=whitespace_sentinel)

    bundle = next(
        (
            public_glb.parent.parent
            / ".failed_attempts"
            / "route2_tall_man_v1"
        ).iterdir()
    )
    failure = contract.route2_instance.validate_pixal_attribute_failure_bundle(bundle)
    assert failure["failure_stage"] == "success_sentinel"
    assert any(item["relative_path"] == "execution.log" for item in failure["artifacts"])


def test_publication_fsync_interrupt_moves_published_tree_to_failure_bundle(
    tmp_path, monkeypatch
):
    job, public_glb, snapshot, environment = _executor_fixture(tmp_path)
    monkeypatch.setattr(contract, "reauthenticate_pixal_job", lambda path: snapshot)
    monkeypatch.setattr(contract, "probe_pixal_runtime", lambda: environment)

    def successful_subprocess(command, **kwargs):
        staged_glb = Path(command[command.index("--output") + 1])
        _write_pbr_glb(staged_glb)
        staged_manifest = staged_glb.with_suffix(".manifest.json")
        staged_manifest.write_text(
            json.dumps(
                {
                    "backend": "pixal3d",
                    "input": {
                        "path": snapshot["payload"]["input_rgba"]["path"],
                        "sha256": snapshot["payload"]["input_rgba"]["sha256"],
                    },
                    "output": {
                        "path": str(staged_glb),
                        "sha256": _sha(staged_glb),
                        "bytes": staged_glb.stat().st_size,
                    },
                    "model": {"revision": contract.PIXAL3D_REVISION},
                    "dino": {"revision": contract.DINO_REVISION},
                    "parameters": snapshot["payload"]["parameters"],
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            returncode=0,
            stdout=f"successful generator\n{staged_manifest}\n",
            stderr="",
        )

    output_root = public_glb.parent.parent
    real_fsync_directory = contract.common.fsync_directory
    output_root_fsync_calls = 0

    def interrupt_first_publication_fsync(path):
        nonlocal output_root_fsync_calls
        if Path(path) == output_root:
            output_root_fsync_calls += 1
            if output_root_fsync_calls == 1 and public_glb.parent.exists():
                raise KeyboardInterrupt("synthetic publication fsync interrupt")
        return real_fsync_directory(path)

    monkeypatch.setattr(
        contract.common, "fsync_directory", interrupt_first_publication_fsync
    )
    with pytest.raises(KeyboardInterrupt, match="publication fsync interrupt"):
        contract.execute_atomic_pixal_job(
            job, subprocess_runner=successful_subprocess
        )

    assert not public_glb.parent.exists()
    assert output_root_fsync_calls >= 2
    bundle = next(
        (output_root / ".failed_attempts" / "route2_tall_man_v1").iterdir()
    )
    failure = contract.route2_instance.validate_pixal_attribute_failure_bundle(bundle)
    assert failure["failure_stage"] == "publication"
    assert any(item["relative_path"] == "pixal_attempt.json" for item in failure["artifacts"])


@pytest.mark.parametrize("mutation", ["input_path", "output_path", "output_bytes"])
def test_generator_manifest_must_bind_exact_staged_paths_and_size(
    tmp_path, monkeypatch, mutation
):
    job, public_glb, snapshot, environment = _executor_fixture(tmp_path)
    monkeypatch.setattr(contract, "reauthenticate_pixal_job", lambda path: snapshot)
    monkeypatch.setattr(contract, "probe_pixal_runtime", lambda: environment)

    def spliced_manifest(command, **kwargs):
        staged_glb = Path(command[command.index("--output") + 1])
        _write_pbr_glb(staged_glb)
        staged_manifest = staged_glb.with_suffix(".manifest.json")
        payload = {
            "backend": "pixal3d",
            "input": {
                "path": snapshot["payload"]["input_rgba"]["path"],
                "sha256": snapshot["payload"]["input_rgba"]["sha256"],
            },
            "output": {
                "path": str(staged_glb),
                "sha256": _sha(staged_glb),
                "bytes": staged_glb.stat().st_size,
            },
            "model": {"revision": contract.PIXAL3D_REVISION},
            "dino": {"revision": contract.DINO_REVISION},
            "parameters": snapshot["payload"]["parameters"],
        }
        if mutation == "input_path":
            payload["input"]["path"] = str(tmp_path / "spliced_input.png")
        elif mutation == "output_path":
            payload["output"]["path"] = str(public_glb)
        else:
            payload["output"]["bytes"] += 1
        staged_manifest.write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(
            returncode=0,
            stdout=f"generator\n{staged_manifest}\n",
            stderr="",
        )

    with pytest.raises(contract.PixalContractError, match="manifest readback"):
        contract.execute_atomic_pixal_job(job, subprocess_runner=spliced_manifest)

    bundle = next(
        (
            public_glb.parent.parent
            / ".failed_attempts"
            / "route2_tall_man_v1"
        ).iterdir()
    )
    failure = contract.route2_instance.validate_pixal_attribute_failure_bundle(bundle)
    assert failure["failure_stage"] == "manifest_readback"


def test_staged_pixal_readback_accepts_actual_ext_texture_webp_pbr_layout(tmp_path):
    rgba = tmp_path / "input.png"
    Image.new("RGBA", (2, 2), (1, 2, 3, 255)).save(rgba)
    glb = tmp_path / "pixal_webp_layout.glb"
    document, binary = _pbr_glb_payload(mime_type="image/webp", image_count=2)
    document["extensionsUsed"] = ["EXT_texture_webp"]
    document["extensionsRequired"] = ["EXT_texture_webp"]
    document["textures"] = [
        {"extensions": {"EXT_texture_webp": {"source": index}}}
        for index in range(2)
    ]
    _write_glb(glb, document, binary)

    document, record = contract.validate_staged_pixal_glb(
        glb, staging=tmp_path, input_rgba=rgba
    )

    assert document["extensionsRequired"] == ["EXT_texture_webp"]
    assert record["sha256"] == _sha(glb)


def test_model_evidence_persistent_projection_ignores_only_cache_hit():
    evidence = {
        "path": "/data/models/example",
        "revision": "a" * 40,
        "file_count": 12,
        "inventory_sha256": "b" * 64,
        "license": {"sha256": "c" * 64},
        "cache_hit": False,
    }
    first = contract._persistent_model_evidence(evidence)
    evidence["cache_hit"] = True
    second = contract._persistent_model_evidence(evidence)

    assert first == second
    assert "cache_hit" not in first
