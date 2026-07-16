"""Contract tests for the authenticated Pixal -> TokenRig male canary."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import socket
import struct
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import tokenrig_human_canary as runner


RUNTIME_INFO = {
    "python": "3.11.13",
    "packages": {
        "bpy": "5.0.1",
        "diffusers": "0.39.0",
        "flash_attn": "2.8.3.post1",
        "lightning": "2.6.5",
        "numpy": "1.26.4",
        "omegaconf": "2.3.1",
        "open3d": "0.19.0",
        "torch": "2.7.1+cu126",
        "transformers": "5.13.1",
        "trimesh": "4.12.2",
    },
    "cuda": "12.6",
    "gpu": {
        "available": True,
        "visible_devices": "3",
        "logical_index": 0,
        "name": "Test GPU",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _glb_bytes(*, with_skin: bool = True) -> bytes:
    document = {
        "asset": {"version": "2.0", "generator": "test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{}]}],
    }
    if with_skin:
        document["nodes"][0]["skin"] = 0
        document["skins"] = [{"joints": [0]}]
    json_chunk = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_chunk += b" " * ((-len(json_chunk)) % 4)
    total_length = 12 + 8 + len(json_chunk)
    return (
        struct.pack("<4sII", b"glTF", 2, total_length)
        + struct.pack("<II", len(json_chunk), 0x4E4F534A)
        + json_chunk
    )


def _write(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _source_hash(source: str, *names: str) -> str:
    node = ast.parse(source)
    current = node
    for name in names:
        children = current.body
        current = next(
            item
            for item in children
            if isinstance(item, (ast.ClassDef, ast.FunctionDef)) and item.name == name
        )
    segment = ast.get_source_segment(source, current)
    assert segment is not None
    return hashlib.sha256(segment.encode("utf-8")).hexdigest()


@pytest.fixture
def workspace(tmp_path):
    input_root = tmp_path / "approved-input"
    input_glb = _write(input_root / "canary_1024_seed42.glb", _glb_bytes(with_skin=False))
    input_manifest = input_root / "canary_1024_seed42.manifest.json"
    input_manifest.write_text(
        json.dumps(
            {
                "backend": "pixal3d",
                "model": {"revision": runner.PIXAL3D_REVISION},
                "output": {
                    "path": str(input_glb.resolve()),
                    "bytes": input_glb.stat().st_size,
                    "sha256": _sha256(input_glb),
                },
                "parameters": {
                    "low_vram": True,
                    "manual_fov": 0.2,
                    "resolution": 1024,
                    "seed": 42,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    skintokens_root = tmp_path / "SkinTokens"
    python = _write(skintokens_root / ".venv/bin/python", b"test-python\n")
    python.chmod(0o755)
    files = {
        "demo.py": b"# pinned demo\n",
        "bpy_server.py": b"# pinned bpy server\n",
        "README.md": b"SkinTokens trained on ArticulationXL 2.0 + VRoid Hub + ModelsResource\n",
        runner.TOKENRIG_CHECKPOINT_RELATIVE: b"tokenrig weights\n",
        runner.SKIN_VAE_CHECKPOINT_RELATIVE: b"skin vae weights\n",
        "LICENSE": b"MIT test snapshot\n",
    }
    for relative, content in files.items():
        _write(skintokens_root / relative, content)
    qwen_files = {
        "config.json": b'{"model_type":"qwen3-test"}\n',
        "generation_config.json": b'{"do_sample":true}\n',
        "README.md": b"# pinned Qwen config fixture\n",
    }
    for relative, content in qwen_files.items():
        _write(skintokens_root / "models/Qwen3-0.6B" / relative, content)

    output_root = tmp_path / "approved-output"
    output_root.mkdir()
    output_dir = output_root / runner.ASSET_ID
    contract = runner.CanaryContract(
        asset_id=runner.ASSET_ID,
        input_glb=input_glb,
        input_manifest=input_manifest,
        output_dir=output_dir,
        skintokens_root=skintokens_root,
        input_glb_sha256=_sha256(input_glb),
        input_manifest_sha256=_sha256(input_manifest),
        skintokens_commit="a" * 40,
        model_revision="b" * 40,
        code_hashes={name: hashlib.sha256(files[name]).hexdigest() for name in ("demo.py", "bpy_server.py")},
        checkpoint_hashes={
            relative: hashlib.sha256(files[relative]).hexdigest()
            for relative in (
                runner.TOKENRIG_CHECKPOINT_RELATIVE,
                runner.SKIN_VAE_CHECKPOINT_RELATIVE,
            )
        },
        checkpoint_sizes={relative: len(files[relative]) for relative in (
            runner.TOKENRIG_CHECKPOINT_RELATIVE,
            runner.SKIN_VAE_CHECKPOINT_RELATIVE,
        )},
        weight_cache_root=None,
        qwen_revision="c" * 40,
        qwen_cache_root=None,
        qwen_file_hashes={name: hashlib.sha256(content).hexdigest() for name, content in qwen_files.items()},
        qwen_file_sizes={name: len(content) for name, content in qwen_files.items()},
        license_hash=hashlib.sha256(files["LICENSE"]).hexdigest(),
    )
    return SimpleNamespace(
        contract=contract,
        input_glb=input_glb,
        input_manifest=input_manifest,
        output_dir=output_dir,
        skintokens_root=skintokens_root,
    )


@pytest.fixture
def recovery_workspace(tmp_path):
    route_root = tmp_path / "pixal_tokenrig_route2_v1"
    failed_dir = route_root / f"{runner.ASSET_ID}.tokenrig_failed_attempt"
    failed_dir.mkdir(parents=True)
    input_root = tmp_path / "approved-input"
    input_glb = _write(
        input_root / "canary_1024_seed42.glb", _glb_bytes(with_skin=False)
    )
    input_manifest = input_root / "canary_1024_seed42.manifest.json"
    input_manifest.write_text(
        json.dumps(
            {
                "backend": "pixal3d",
                "output": {
                    "path": str(input_glb.resolve()),
                    "sha256": _sha256(input_glb),
                    "bytes": input_glb.stat().st_size,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    failed_glb = _write(failed_dir / "tokenrig_transfer.glb", _glb_bytes())
    stderr = (
        "Error in sitecustomize; set PYTHONVERBOSE for traceback:\n"
        "ModuleNotFoundError: No module named 'src'\n"
        "Error in sitecustomize; set PYTHONVERBOSE for traceback:\n"
        "ModuleNotFoundError: No module named 'src'\n"
    )
    stdout = "inference completed\n"
    inference_log = _write(
        failed_dir / "inference.log",
        ("=== stdout ===\n" + stdout + "=== stderr ===\n" + stderr).encode(
            "utf-8"
        ),
    )
    patch_path = _write(
        failed_dir / "runtime_patch/sitecustomize.py", b"# failed hygiene patch\n"
    )
    markers_dir = failed_dir / "runtime_patch/markers"
    markers_dir.mkdir()

    parser_source = """\
class BpyParser:
    @classmethod
    def load(cls, filepath, **kwargs):
        clean_bpy()
        load(filepath=filepath, **kwargs)
        return filepath

def clean_bpy():
    bpy.ops.outliner.orphans_purge(do_recursive=True)
    for item in list(bpy.data.objects):
        bpy.data.objects.remove(item)

def load(filepath, **kwargs):
    return filepath
"""
    skintokens_root = tmp_path / "SkinTokens"
    parser_path = _write(
        skintokens_root / "src/rig_package/parser/bpy.py",
        parser_source.encode("utf-8"),
    )

    files = {}
    for path in sorted(failed_dir.rglob("*")):
        if path.is_file():
            files[path.relative_to(failed_dir).as_posix()] = _record(path)
    inventory_payload = {
        relative: {
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
        for relative, record in files.items()
    }
    inventory_sha256 = hashlib.sha256(
        (json.dumps(inventory_payload, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()
    failed_evidence = {
        "path": str(failed_dir.resolve()),
        "inventory_sha256": inventory_sha256,
        "files": files,
    }
    input_snapshot = {
        "glb": _record(input_glb),
        "manifest": _record(input_manifest),
    }
    attempt_id = "d" * 64
    attempt = {
        "schema": "pixal_tokenrig_attempt_v1",
        "attempt_id": attempt_id,
        "asset_id": runner.ASSET_ID,
        "status": "failed",
        "returncode": 0,
        "failure_stage": "output_validation",
        "error": {
            "type": "CanaryError",
            "message": "server hygiene patch did not run in both demo and bpy server processes",
        },
        "stdout": stdout,
        "stderr": stderr,
        "command": ["python", "-c", "seeded demo", "--use_transfer"],
        "authenticated_hashes": {
            "input": input_snapshot,
            "skintokens": {"commit": "a" * 40},
            "server_hygiene_patch": _record(patch_path),
        },
        "failed_evidence": failed_evidence,
    }
    attempt_ledger = route_root / f"{runner.ASSET_ID}.tokenrig_attempt.json"
    attempt_ledger.write_text(
        json.dumps(attempt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    recovery_output_dir = route_root / runner.ASSET_ID
    contract = runner.RecoveryContract(
        asset_id=runner.ASSET_ID,
        attempt_id=attempt_id,
        attempt_ledger=attempt_ledger,
        attempt_ledger_sha256=_sha256(attempt_ledger),
        attempt_ledger_size=attempt_ledger.stat().st_size,
        failed_evidence_dir=failed_dir,
        failed_inventory_sha256=inventory_sha256,
        failed_glb=failed_glb,
        failed_glb_sha256=_sha256(failed_glb),
        failed_glb_size=failed_glb.stat().st_size,
        inference_log=inference_log,
        inference_log_sha256=_sha256(inference_log),
        inference_log_size=inference_log.stat().st_size,
        sitecustomize_path=patch_path,
        sitecustomize_sha256=_sha256(patch_path),
        sitecustomize_size=patch_path.stat().st_size,
        input_glb=input_glb,
        input_glb_sha256=_sha256(input_glb),
        input_glb_size=input_glb.stat().st_size,
        input_manifest=input_manifest,
        input_manifest_sha256=_sha256(input_manifest),
        input_manifest_size=input_manifest.stat().st_size,
        skintokens_root=skintokens_root,
        skintokens_commit="a" * 40,
        parser_path=parser_path,
        parser_sha256=_sha256(parser_path),
        parser_size=parser_path.stat().st_size,
        bpyparser_load_sha256=_source_hash(parser_source, "BpyParser", "load"),
        clean_bpy_sha256=_source_hash(parser_source, "clean_bpy"),
        recovery_output_dir=recovery_output_dir,
    )
    return SimpleNamespace(
        contract=contract,
        route_root=route_root,
        failed_dir=failed_dir,
        failed_glb=failed_glb,
        inference_log=inference_log,
        attempt_ledger=attempt_ledger,
        recovery_output_dir=recovery_output_dir,
        stderr=stderr,
    )


class FakeSubprocess:
    def __init__(self, contract, *, commit=None, output="valid"):
        self.contract = contract
        self.commit = commit or contract.skintokens_commit
        self.output = output
        self.calls = []

    @property
    def inference_calls(self):
        return [call for call in self.calls if "--use_transfer" in call[0]]

    def __call__(self, command, **kwargs):
        command = [str(value) for value in command]
        self.calls.append((command, kwargs))
        if command[:2] == ["git", "-C"]:
            if "rev-parse" in command:
                return subprocess.CompletedProcess(command, 0, self.commit + "\n", "")
            if "status" in command:
                return subprocess.CompletedProcess(command, 0, "", "")
        marker_dir = Path(kwargs["env"]["TOKENRIG_HYGIENE_MARKER_DIR"])
        marker_dir.mkdir(parents=True, exist_ok=True)
        for pid, argv in ((101, ["-c"]), (102, ["bpy_server.py"])):
            (marker_dir / f"{pid}.json").write_text(
                json.dumps(
                    {
                        "argv": argv,
                        "patch_sha256": kwargs["env"]["TOKENRIG_SERVER_HYGIENE_SHA256"],
                        "pid": pid,
                        "seed": int(kwargs["env"]["TOKENRIG_CANARY_SEED"]),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        audit_value = kwargs["env"].get("TOKENRIG_LOAD_AUDIT_PATH")
        if audit_value:
            audit_path = Path(audit_value)
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            filepath = command[command.index("--input") + 1]
            events = []
            for sequence, before_objects in (
                (
                    1,
                    [
                        {"name": "Cube", "type": "MESH"},
                        {"name": "Camera", "type": "CAMERA"},
                        {"name": "Light", "type": "LIGHT"},
                    ],
                ),
                (2, [{"name": "geometry_0", "type": "MESH"}]),
            ):
                events.extend(
                    [
                        {
                            "sequence": sequence,
                            "phase": "before_clean",
                            "filepath": filepath,
                            "inventory": {
                                "objects": before_objects,
                                "mesh_count": sum(
                                    item["type"] == "MESH" for item in before_objects
                                ),
                                "material_count": 0,
                                "image_count": 0,
                            },
                        },
                        {
                            "sequence": sequence,
                            "phase": "after_clean",
                            "filepath": filepath,
                            "inventory": {
                                "objects": [],
                                "mesh_count": 0,
                                "material_count": 0,
                                "image_count": 0,
                            },
                        },
                        {
                            "sequence": sequence,
                            "phase": "after_import",
                            "filepath": filepath,
                            "inventory": {
                                "objects": [{"name": "geometry_0", "type": "MESH"}],
                                "mesh_count": 1,
                                "material_count": 1,
                                "image_count": 1,
                            },
                        },
                    ]
                )
            audit_path.write_text(
                "".join(
                    json.dumps(event, sort_keys=True) + "\n" for event in events
                ),
                encoding="utf-8",
            )
        if self.output == "failure":
            return subprocess.CompletedProcess(
                command, 7, "partial stdout\n", "inference failed\n"
            )
        output_path = Path(command[command.index("--output") + 1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output == "valid":
            output_path.write_bytes(_glb_bytes())
        elif self.output == "invalid":
            output_path.write_bytes(b"not a glb")
        elif self.output == "alias":
            os.link(self.contract.input_glb, output_path)
        return subprocess.CompletedProcess(command, 0, "inference ok\n", "")


def _run(workspace, fake, **overrides):
    arguments = {
        "input_glb": workspace.input_glb,
        "input_manifest": workspace.input_manifest,
        "output_dir": workspace.output_dir,
        "skintokens_root": workspace.skintokens_root,
        "model_revision": workspace.contract.model_revision,
        "seed": 42,
        "use_skeleton_input": False,
        "contract": workspace.contract,
        "subprocess_runner": fake,
        "runtime_probe": lambda **_kwargs: RUNTIME_INFO,
        "bpy_port_probe": lambda: None,
        "base_env": {"CUDA_VISIBLE_DEVICES": "3", "PATH": "/usr/bin"},
    }
    arguments.update(overrides)
    return runner.run_canary(**arguments)


def test_production_contract_pins_exact_male_and_skintokens_artifacts():
    contract = runner.PINNED_CONTRACT

    assert runner.EXPECTED_RUNTIME_PACKAGES["numpy"] == "1.26.4"
    assert contract.input_glb_sha256 == "1df2490d6b83e52fa3b7c4e9d6b69207fa59cad0deae80e3dc3f894dfc443c42"
    assert contract.input_manifest_sha256 == "f0658fbcf84d3505d5ea08fcf3011c9070de46933da50c7304d44875a3b038e3"
    assert contract.skintokens_commit == "273b691d35989d71cd17ff2895fdc735097b92d1"
    assert contract.model_revision == "79736cad0fd84de384d5eede659b4ebd24effe33"
    assert contract.checkpoint_hashes == {
        runner.TOKENRIG_CHECKPOINT_RELATIVE: "f4e4706a11cfb520cdde65156a0358545e4fbf8f36237aca01ea5e79d5cb5692",
        runner.SKIN_VAE_CHECKPOINT_RELATIVE: "4843f49e58afff88345806b94ca82e6cc9d8def6e7432e2853c677b154de0ed4",
    }
    assert contract.checkpoint_sizes == {
        runner.TOKENRIG_CHECKPOINT_RELATIVE: 1131603979,
        runner.SKIN_VAE_CHECKPOINT_RELATIVE: 487311745,
    }
    assert contract.weight_cache_root == Path("/data/models/hub/models--VAST-AI--SkinTokens")
    assert contract.qwen_revision == "c1899de289a04d12100db370d81485cdf75e47ca"
    assert contract.qwen_cache_root == Path("/data/models/hub/models--Qwen--Qwen3-0.6B")
    assert contract.qwen_file_hashes == {
        "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
        "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
        "README.md": "1ab64a26fcb3b461423b89a433a8c858f1bf8d4086f979cbb3ff878d47cf20e9",
    }
    assert contract.qwen_file_sizes == {"config.json": 726, "generation_config.json": 239, "README.md": 13965}
    assert contract.code_hashes == {
        "demo.py": "8e6d058225c39caad0fccf7c4d6942f8e7e32e3f57c5b14cdc60cf2d6cb5d316",
        "bpy_server.py": "0764aa1436130bdf32ffc2892a4497b77e9e22399c7ed1af5cf11a1f32500130",
    }
    assert contract.license_hash == "4f818b00ed33ed1772236c8b0acfd40e740e11e93f4d6f4f846b506b5b690789"


def test_run_uses_seeded_offline_command_and_publishes_complete_manifest(workspace):
    fake = FakeSubprocess(workspace.contract)

    manifest_path = _run(workspace, fake)

    assert manifest_path == workspace.output_dir / "tokenrig_manifest.json"
    assert (workspace.output_dir / "tokenrig_transfer.glb").is_file()
    assert not list(workspace.output_dir.parent.glob(".*.staging"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    command, kwargs = fake.inference_calls[0]
    bootstrap = command[2]
    for expression in (
        "random.seed(seed)",
        "np.random.seed(seed)",
        "torch.manual_seed(seed)",
        "torch.cuda.manual_seed_all(seed)",
    ):
        assert expression in bootstrap
    assert bootstrap.index("random.seed(seed)") < bootstrap.index("runpy.run_path")
    assert command[:2] == [str(workspace.skintokens_root / ".venv/bin/python"), "-c"]
    assert command[3:5] == ["42", str(workspace.skintokens_root / "demo.py")]
    assert "--use_transfer" in command
    assert "--use_skeleton" not in command
    assert kwargs["cwd"] == str(workspace.skintokens_root)
    for key in runner.OFFLINE_ENVIRONMENT:
        assert kwargs["env"][key] == "1"
    assert kwargs["env"]["TOKENRIG_CANARY_SEED"] == "42"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert kwargs["env"]["PYTHONHASHSEED"] == "42"
    assert kwargs["env"]["TOKENIZERS_PARALLELISM"] == "false"
    pythonpath_parts = kwargs["env"]["PYTHONPATH"].split(os.pathsep)
    injected_path = Path(pythonpath_parts[0])
    assert injected_path.name == "runtime_patch"
    assert pythonpath_parts[1] == str(workspace.skintokens_root)

    assert manifest["schema"] == "pixal_tokenrig_canary_v1"
    assert manifest["source_front"] == "positive-y"
    assert manifest["canonical_front"] == "negative-y"
    assert manifest["command"] == command
    assert manifest["sampling_parameters"] == runner.SAMPLING_PARAMETERS
    assert manifest["random_parameters"] == {
        "seed": 42,
        "seed_bootstrap_before_demo_import": True,
        "seeded_libraries": ["random", "numpy", "torch", "torch.cuda"],
    }
    assert manifest["environment"] == RUNTIME_INFO
    assert manifest["gpu"] == RUNTIME_INFO["gpu"]
    assert manifest["input"]["glb"]["sha256"] == workspace.contract.input_glb_sha256
    assert manifest["input"]["manifest"]["sha256"] == workspace.contract.input_manifest_sha256
    assert manifest["output"]["sha256"] == _sha256(workspace.output_dir / "tokenrig_transfer.glb")
    assert manifest["skintokens"]["commit"] == workspace.contract.skintokens_commit
    assert manifest["skintokens"]["model_revision"] == workspace.contract.model_revision
    assert manifest["skintokens"]["qwen"]["revision"] == workspace.contract.qwen_revision
    risks = manifest["skintokens"]["training_provenance_risks"]
    assert {risk["training_source"] for risk in risks} == {
        "ArticulationXL 2.0",
        "VRoid Hub",
        "ModelsResource",
    }
    for risk in risks:
        assert risk["status"] == "unresolved_for_formal_registration"
        reference = risk["upstream_reference"]
        assert reference["model_card_url"].startswith(
            "https://huggingface.co/VAST-AI/SkinTokens"
        )
        assert reference["readme"]["sha256"] == _sha256(
            workspace.skintokens_root / "README.md"
        )
        assert reference["readme"]["size_bytes"] == (
            workspace.skintokens_root / "README.md"
        ).stat().st_size
    assert set(manifest["skintokens"]["code_hashes"]) == {"demo.py", "bpy_server.py"}
    assert set(manifest["skintokens"]["weight_hashes"]) == {
        runner.TOKENRIG_CHECKPOINT_RELATIVE,
        runner.SKIN_VAE_CHECKPOINT_RELATIVE,
    }
    assert manifest["skintokens"]["license_hash"] == workspace.contract.license_hash
    runner_record = manifest["orchestrator"]["runner"]
    assert runner_record == {
        "path": str(runner.RUNNER_PATH),
        "sha256": _sha256(runner.RUNNER_PATH),
        "size_bytes": runner.RUNNER_PATH.stat().st_size,
    }
    inference_log = workspace.output_dir / "inference.log"
    assert manifest["inference_log"]["sha256"] == _sha256(inference_log)
    assert "inference ok" in inference_log.read_text(encoding="utf-8")
    hygiene = manifest["server_hygiene"]
    published_patch = workspace.output_dir / hygiene["relative_path"]
    assert hygiene["sha256"] == _sha256(published_patch)
    assert hygiene["cleans_before_every_bpyparser_load"] is True
    assert "preserves_transfer_target_pbr" not in hygiene
    assert hygiene["pbr_preservation_requested_by_use_transfer"] is True
    assert hygiene["pbr_validation_status"] == "pending_static_audit"
    assert hygiene["server_runtime"] == {
        "bpy": "5.0.1",
        "purpose": "SkinTokens embedded transfer server",
    }
    assert {record["role"] for record in hygiene["processes"]} == {"demo", "bpy_server"}
    assert [load["role"] for load in hygiene["loads"]] == [
        "source",
        "transfer_target",
    ]
    assert [load["sequence"] for load in hygiene["loads"]] == [1, 2]
    load_audit = workspace.output_dir / hygiene["load_audit"]["relative_path"]
    assert hygiene["load_audit"]["sha256"] == _sha256(load_audit)
    events = [
        json.loads(line)
        for line in load_audit.read_text(encoding="utf-8").splitlines()
    ]
    assert [(event["sequence"], event["phase"]) for event in events] == [
        (1, "before_clean"),
        (1, "after_clean"),
        (1, "after_import"),
        (2, "before_clean"),
        (2, "after_clean"),
        (2, "after_import"),
    ]
    assert all(
        not event["inventory"]["objects"]
        for event in events
        if event["phase"] == "after_clean"
    )
    assert all(
        event["inventory"]["mesh_count"] >= 1
        for event in events
        if event["phase"] == "after_import"
    )
    assert all(
        {
            item["name"] for item in event["inventory"]["objects"]
        }.isdisjoint({"Cube", "Camera", "Light"})
        for event in events
        if event["phase"] == "after_import"
    )

    attempt_path = (
        workspace.output_dir.parent
        / f"{workspace.output_dir.name}.tokenrig_attempt.json"
    )
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert attempt["status"] == "succeeded"
    assert attempt["command"] == command
    assert attempt["started_at_utc"]
    assert attempt["ended_at_utc"]
    assert attempt["returncode"] == 0
    assert attempt["stdout"] == "inference ok\n"
    assert attempt["stderr"] == ""
    assert attempt["failure_stage"] is None
    assert attempt["authenticated_hashes"]["orchestrator"] == runner_record
    assert manifest["attempt_ledger"]["path"] == str(attempt_path)
    assert manifest["attempt_ledger"]["sha256"] == _sha256(attempt_path)


def test_child_bpy_server_hygiene_patch_cleans_then_loads_and_seeds(workspace):
    fake = FakeSubprocess(workspace.contract)

    manifest_path = _run(workspace, fake)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    patch_path = workspace.output_dir / manifest["server_hygiene"]["relative_path"]
    source = patch_path.read_text(encoding="utf-8")
    assert "parser.clean_bpy()" in source
    assert "_record_load_event" in source
    assert '"before_clean"' in source
    assert '"after_clean"' in source
    assert '"after_import"' in source
    assert "_original_load(cls, filepath" in source
    assert source.index("parser.clean_bpy()") < source.index("_original_load(cls, filepath")
    assert "BpyParser.load = classmethod" in source
    for expression in (
        "random.seed(seed)",
        "np.random.seed(seed)",
        "torch.manual_seed(seed)",
        "torch.cuda.manual_seed_all(seed)",
    ):
        assert expression in source
    assert "materials.remove" not in source
    assert "images.remove" not in source
    command, kwargs = fake.inference_calls[0]
    assert kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0].endswith("/runtime_patch")
    assert kwargs["env"]["TOKENRIG_CANARY_SEED"] == "42"
    assert kwargs["env"]["TOKENRIG_SERVER_HYGIENE_SHA256"] == manifest["server_hygiene"]["sha256"]
    assert manifest["command"] == command


def test_runtime_pythonpath_preserves_prior_after_patch_and_skintokens_root(workspace):
    fake = FakeSubprocess(workspace.contract)

    _run(
        workspace,
        fake,
        base_env={
            "CUDA_VISIBLE_DEVICES": "3",
            "PATH": "/usr/bin",
            "PYTHONPATH": "/prior/one:/prior/two",
        },
    )

    pythonpath = fake.inference_calls[0][1]["env"]["PYTHONPATH"].split(os.pathsep)
    assert Path(pythonpath[0]).name == "runtime_patch"
    assert pythonpath[1:] == [
        str(workspace.skintokens_root),
        "/prior/one",
        "/prior/two",
    ]


def test_real_sitecustomize_smoke_imports_src_patches_load_and_writes_evidence(tmp_path):
    skintokens_root = runner.PINNED_CONTRACT.skintokens_root
    python = skintokens_root / ".venv/bin/python"
    assert python.is_file()
    patch_dir = tmp_path / "runtime_patch"
    patch_dir.mkdir()
    (patch_dir / "sitecustomize.py").write_text(
        runner.SERVER_HYGIENE_SOURCE, encoding="utf-8"
    )
    marker_dir = patch_dir / "markers"
    audit_path = patch_dir / "load_audit.jsonl"
    patch_hash = hashlib.sha256(
        runner.SERVER_HYGIENE_SOURCE.encode("utf-8")
    ).hexdigest()
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "DIFFUSERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "PYTHONHASHSEED": "42",
        "PYTHONPATH": os.pathsep.join((str(patch_dir), str(skintokens_root))),
        "TOKENIZERS_PARALLELISM": "false",
        "TOKENRIG_CANARY_SEED": "42",
        "TOKENRIG_HYGIENE_MARKER_DIR": str(marker_dir),
        "TOKENRIG_LOAD_AUDIT_PATH": str(audit_path),
        "TOKENRIG_SERVER_HYGIENE_SHA256": patch_hash,
        "TRANSFORMERS_OFFLINE": "1",
    }
    smoke = """
from src.rig_package.parser import bpy as parser
scope = parser.BpyParser.load.__func__.__globals__
assert '_record_load_event' in scope
def fake_original(cls, filepath, **kwargs):
    return 'patched-load'
scope['_original_load'] = fake_original
assert parser.BpyParser.load('/nonexistent/fake.glb') == 'patched-load'
print('SITECUSTOMIZE_SMOKE_OK')
"""

    result = subprocess.run(
        [str(python), "-c", smoke],
        cwd=str(tmp_path),
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "SITECUSTOMIZE_SMOKE_OK"
    markers = list(marker_dir.glob("*.json"))
    assert len(markers) == 1
    events = [
        json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["phase"] for event in events] == [
        "before_clean",
        "after_clean",
        "after_import",
    ]


def test_rejects_load_audit_without_prior_source_before_transfer(workspace):
    class MissingPriorSourceSubprocess(FakeSubprocess):
        def __call__(self, command, **kwargs):
            result = super().__call__(command, **kwargs)
            if "--use_transfer" in command:
                audit_path = Path(kwargs["env"]["TOKENRIG_LOAD_AUDIT_PATH"])
                events = [
                    json.loads(line)
                    for line in audit_path.read_text(encoding="utf-8").splitlines()
                ]
                event = next(
                    item
                    for item in events
                    if item["sequence"] == 2 and item["phase"] == "before_clean"
                )
                event["inventory"] = {
                    "objects": [],
                    "mesh_count": 0,
                    "material_count": 0,
                    "image_count": 0,
                }
                audit_path.write_text(
                    "".join(
                        json.dumps(item, sort_keys=True) + "\n" for item in events
                    ),
                    encoding="utf-8",
                )
            return result

    fake = MissingPriorSourceSubprocess(workspace.contract)

    with pytest.raises(runner.CanaryError, match="prior source scene"):
        _run(workspace, fake)

    assert not workspace.output_dir.exists()


def test_rejects_orchestrator_bytes_changed_during_inference(workspace, tmp_path):
    orchestrator = _write(tmp_path / "tokenrig_human_canary.py", b"pinned runner\n")

    class MutatingSubprocess(FakeSubprocess):
        def __call__(self, command, **kwargs):
            result = super().__call__(command, **kwargs)
            if "--use_transfer" in command:
                orchestrator.write_bytes(b"changed runner\n")
            return result

    fake = MutatingSubprocess(workspace.contract)

    with pytest.raises(runner.CanaryError, match="orchestrator.*changed"):
        _run(workspace, fake, orchestrator_path=orchestrator)
    assert not workspace.output_dir.exists()
    assert not list(workspace.output_dir.parent.glob(".*.staging"))


def test_socket_preflight_detects_an_existing_listener():
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    try:
        with pytest.raises(runner.CanaryError, match="BPY.*already listening"):
            runner.assert_bpy_port_available(port=port)
    finally:
        listener.close()


def test_run_rejects_stale_fixed_bpy_server_before_inference(workspace):
    calls = []

    def occupied_port():
        calls.append(runner.BPY_PORT)
        raise runner.CanaryError("BPY port 59876 is already listening")

    fake = FakeSubprocess(workspace.contract)

    with pytest.raises(runner.CanaryError, match="BPY port 59876"):
        _run(workspace, fake, bpy_port_probe=occupied_port)
    assert calls == [59876]
    assert not fake.inference_calls
    assert not workspace.output_dir.exists()
    attempt_path = (
        workspace.output_dir.parent
        / f"{workspace.output_dir.name}.tokenrig_attempt.json"
    )
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert attempt["status"] == "failed"
    assert attempt["failure_stage"] == "bpy_port_preflight"
    assert attempt["returncode"] is None
    assert not (
        workspace.output_dir.parent
        / f"{workspace.output_dir.name}.tokenrig_failed_attempt"
    ).exists()


def test_attempt_ledger_exists_before_child_launch(workspace):
    attempt_path = (
        workspace.output_dir.parent
        / f"{workspace.output_dir.name}.tokenrig_attempt.json"
    )
    observed = []

    class ObservingSubprocess(FakeSubprocess):
        def __call__(self, command, **kwargs):
            if "--use_transfer" in command:
                attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
                observed.append(attempt["status"])
            return super().__call__(command, **kwargs)

    fake = ObservingSubprocess(workspace.contract)

    _run(workspace, fake)

    assert observed == ["started"]


def test_stale_attempt_ledger_rejects_before_child_launch(workspace):
    attempt_path = (
        workspace.output_dir.parent
        / f"{workspace.output_dir.name}.tokenrig_attempt.json"
    )
    attempt_path.write_text("sealed prior attempt\n", encoding="utf-8")
    fake = FakeSubprocess(workspace.contract)

    with pytest.raises(runner.CanaryError, match="attempt ledger already exists"):
        _run(workspace, fake)

    assert attempt_path.read_text(encoding="utf-8") == "sealed prior attempt\n"
    assert not fake.inference_calls
    assert not workspace.output_dir.exists()


def test_failed_subprocess_is_preserved_in_attempt_ledger(workspace):
    fake = FakeSubprocess(workspace.contract, output="failure")

    with pytest.raises(runner.CanaryError, match="subprocess failed with exit 7"):
        _run(workspace, fake)

    attempt_path = (
        workspace.output_dir.parent
        / f"{workspace.output_dir.name}.tokenrig_attempt.json"
    )
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert attempt["status"] == "failed"
    assert attempt["failure_stage"] == "inference_subprocess"
    assert attempt["returncode"] == 7
    assert attempt["stdout"] == "partial stdout\n"
    assert attempt["stderr"] == "inference failed\n"
    assert attempt["ended_at_utc"]
    assert not workspace.output_dir.exists()
    failed_dir = (
        workspace.output_dir.parent
        / f"{workspace.output_dir.name}.tokenrig_failed_attempt"
    )
    assert failed_dir.is_dir()
    evidence = attempt["failed_evidence"]
    assert evidence["path"] == str(failed_dir)
    assert evidence["inventory_sha256"]
    assert "runtime_patch/sitecustomize.py" in evidence["files"]
    assert "runtime_patch/load_audit.jsonl" in evidence["files"]


def test_pinned_huggingface_weight_symlink_chain_is_allowed_and_recorded(workspace, tmp_path):
    source_bytes = {
        relative: (workspace.skintokens_root / relative).read_bytes()
        for relative in workspace.contract.checkpoint_hashes
    }
    shutil.rmtree(workspace.skintokens_root / "experiments")
    cache_root = tmp_path / "models--VAST-AI--SkinTokens"
    snapshot = cache_root / "snapshots" / workspace.contract.model_revision
    (snapshot / "experiments").mkdir(parents=True)
    (cache_root / "blobs").mkdir()
    for relative, content in source_bytes.items():
        digest = hashlib.sha256(content).hexdigest()
        blob = _write(cache_root / "blobs" / digest, content)
        logical_in_snapshot = snapshot / relative
        logical_in_snapshot.parent.mkdir(parents=True, exist_ok=True)
        logical_in_snapshot.symlink_to(Path("../../../../blobs") / blob.name)
    (workspace.skintokens_root / "experiments").symlink_to(snapshot / "experiments")
    contract = replace(
        workspace.contract,
        weight_cache_root=cache_root,
        checkpoint_sizes={relative: len(content) for relative, content in source_bytes.items()},
    )
    fake = FakeSubprocess(contract)

    manifest_path = _run(workspace, fake, contract=contract)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, record in manifest["skintokens"]["weight_hashes"].items():
        assert record["logical_path"] == str(workspace.skintokens_root / relative)
        assert Path(record["resolved_path"]).parent == cache_root / "blobs"
        assert len(record["link_chain"]) == 2
        assert record["size_bytes"] == len(source_bytes[relative])


def test_pinned_qwen_snapshot_symlink_and_config_files_are_recorded(workspace, tmp_path):
    logical_model = workspace.skintokens_root / "models/Qwen3-0.6B"
    source_bytes = {path.name: path.read_bytes() for path in logical_model.iterdir()}
    shutil.rmtree(logical_model)
    cache_root = tmp_path / "models--Qwen--Qwen3-0.6B"
    snapshot = cache_root / "snapshots" / workspace.contract.qwen_revision
    snapshot.mkdir(parents=True)
    (cache_root / "blobs").mkdir()
    for name, content in source_bytes.items():
        blob = _write(cache_root / "blobs" / hashlib.sha256(content).hexdigest(), content)
        (snapshot / name).symlink_to(Path("../../blobs") / blob.name)
    logical_model.symlink_to(snapshot)
    contract = replace(workspace.contract, qwen_cache_root=cache_root)
    fake = FakeSubprocess(contract)

    manifest_path = _run(workspace, fake, contract=contract)

    qwen = json.loads(manifest_path.read_text(encoding="utf-8"))["skintokens"]["qwen"]
    assert qwen["logical_path"] == str(logical_model)
    assert qwen["resolved_path"] == str(snapshot)
    assert qwen["revision"] == contract.qwen_revision
    assert set(qwen["files"]) == {"config.json", "generation_config.json", "README.md"}
    for name, record in qwen["files"].items():
        assert record["sha256"] == contract.qwen_file_hashes[name]
        assert record["size_bytes"] == contract.qwen_file_sizes[name]
        assert len(record["link_chain"]) == 2


@pytest.mark.parametrize(
    "runtime_update, message",
    [
        ({"gpu": {"available": False, "visible_devices": "3", "logical_index": None, "name": None}}, "GPU"),
        ({"gpu": {"available": True, "visible_devices": "0", "logical_index": 0, "name": "Test GPU"}}, "GPU"),
        ({"cuda": "12.5"}, "CUDA"),
        ({"packages": {**RUNTIME_INFO["packages"], "torch": "2.7.0"}}, "torch"),
        ({"python": "3.13.2"}, "Python"),
    ],
)
def test_rejects_unverified_runtime_or_wrong_gpu(workspace, runtime_update, message):
    runtime = {**RUNTIME_INFO, **runtime_update}
    fake = FakeSubprocess(workspace.contract)

    with pytest.raises(runner.CanaryError, match=message):
        _run(workspace, fake, runtime_probe=lambda **_kwargs: runtime)
    assert not fake.inference_calls


def test_fallback_records_and_passes_use_skeleton(workspace):
    fake = FakeSubprocess(workspace.contract)

    manifest_path = _run(workspace, fake, use_skeleton_input=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "--use_skeleton" in fake.inference_calls[0][0]
    assert manifest["inference_parameters"]["use_skeleton"] is True
    assert manifest["attempt"] == "fitted_skeleton_transfer"


def test_rejects_changed_input_and_manifest_hashes_before_subprocess(workspace):
    workspace.input_glb.write_bytes(workspace.input_glb.read_bytes() + b"tamper")
    fake = FakeSubprocess(workspace.contract)
    with pytest.raises(runner.CanaryError, match="input GLB SHA-256"):
        _run(workspace, fake)
    assert not fake.calls

    workspace.input_glb.write_bytes(_glb_bytes(with_skin=False))
    workspace.input_manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(runner.CanaryError, match="input manifest SHA-256"):
        _run(workspace, fake)
    assert not fake.calls


def test_rejects_cleaned_obj_even_if_contract_is_replaced(workspace):
    cleaned = _write(workspace.input_glb.parent / "cleaned.obj", b"mesh")
    fake = FakeSubprocess(workspace.contract)

    with pytest.raises(runner.CanaryError, match="cleaned.obj|original.*GLB"):
        _run(workspace, fake, input_glb=cleaned)
    assert not fake.calls


def test_rejects_symlink_that_resolves_outside_approved_input_root(workspace, tmp_path):
    outside = _write(tmp_path / "outside.glb", _glb_bytes())
    linked = workspace.input_glb.parent / "linked.glb"
    linked.symlink_to(outside)
    linked_manifest = workspace.input_glb.parent / "linked.manifest.json"
    linked_manifest.write_text(
        json.dumps(
            {
                "backend": "pixal3d",
                "model": {"revision": runner.PIXAL3D_REVISION},
                "output": {
                    "path": str(linked.absolute()),
                    "bytes": outside.stat().st_size,
                    "sha256": _sha256(outside),
                },
                "parameters": {"seed": 42, "resolution": 1024, "manual_fov": 0.2, "low_vram": True},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    contract = replace(
        workspace.contract,
        input_glb=linked,
        input_manifest=linked_manifest,
        input_glb_sha256=_sha256(outside),
        input_manifest_sha256=_sha256(linked_manifest),
    )
    fake = FakeSubprocess(contract)

    with pytest.raises(runner.CanaryError, match="symlink|approved input root"):
        _run(
            workspace,
            fake,
            contract=contract,
            input_glb=linked,
            input_manifest=linked_manifest,
        )
    assert not fake.calls


def test_rejects_unpinned_or_dirty_skintokens_checkout(workspace):
    wrong = FakeSubprocess(workspace.contract, commit="f" * 40)
    with pytest.raises(runner.CanaryError, match="SkinTokens commit"):
        _run(workspace, wrong)
    assert not wrong.inference_calls

    dirty = FakeSubprocess(workspace.contract)
    original = dirty.__call__

    def dirty_call(command, **kwargs):
        result = original(command, **kwargs)
        if "status" in command:
            return subprocess.CompletedProcess(command, 0, " M demo.py\n", "")
        return result

    with pytest.raises(runner.CanaryError, match="tracked changes"):
        _run(workspace, dirty_call)


@pytest.mark.parametrize("relative", [runner.TOKENRIG_CHECKPOINT_RELATIVE, runner.SKIN_VAE_CHECKPOINT_RELATIVE])
def test_rejects_missing_or_changed_checkpoints(workspace, relative):
    checkpoint = workspace.skintokens_root / relative
    checkpoint.unlink()
    fake = FakeSubprocess(workspace.contract)
    with pytest.raises(runner.CanaryError, match="checkpoint.*missing"):
        _run(workspace, fake)
    assert not fake.inference_calls

    checkpoint.write_bytes(b"changed")
    with pytest.raises(runner.CanaryError, match="checkpoint.*SHA-256"):
        _run(workspace, fake)
    assert not fake.inference_calls


def test_rejects_output_hardlink_aliasing_the_input(workspace):
    fake = FakeSubprocess(workspace.contract, output="alias")

    with pytest.raises(runner.CanaryError, match="alias.*input"):
        _run(workspace, fake)
    assert not workspace.output_dir.exists()
    assert not list(workspace.output_dir.parent.glob(".*.staging"))


def test_rejects_stale_existing_output_before_inference(workspace):
    workspace.output_dir.mkdir()
    (workspace.output_dir / "tokenrig_manifest.json").write_text("stale", encoding="utf-8")
    fake = FakeSubprocess(workspace.contract)

    with pytest.raises(runner.CanaryError, match="stale|already exists"):
        _run(workspace, fake)
    assert not fake.calls


def test_rejects_unrecorded_inference_argument(workspace, monkeypatch):
    real_builder = runner.build_inference_command

    def changed_builder(**kwargs):
        return real_builder(**kwargs) + ["--use_postprocess"]

    monkeypatch.setattr(runner, "build_inference_command", changed_builder)
    fake = FakeSubprocess(workspace.contract)

    with pytest.raises(runner.CanaryError, match="unrecorded inference parameter"):
        _run(workspace, fake)
    assert not fake.inference_calls


def test_rejects_unparseable_glb_without_publication(workspace):
    fake = FakeSubprocess(workspace.contract, output="invalid")

    with pytest.raises(runner.CanaryError, match="GLB"):
        _run(workspace, fake)
    assert not workspace.output_dir.exists()
    assert not list(workspace.output_dir.parent.glob(".*.staging"))
    attempt_path = (
        workspace.output_dir.parent
        / f"{workspace.output_dir.name}.tokenrig_attempt.json"
    )
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    assert attempt["status"] == "failed"
    assert attempt["failure_stage"] == "output_validation"
    assert attempt["returncode"] == 0
    assert attempt["stdout"] == "inference ok\n"
    failed_dir = (
        workspace.output_dir.parent
        / f"{workspace.output_dir.name}.tokenrig_failed_attempt"
    )
    assert failed_dir.is_dir()
    evidence = attempt["failed_evidence"]
    assert evidence["path"] == str(failed_dir)
    assert set(evidence["files"]) >= {
        "tokenrig_transfer.glb",
        "inference.log",
        "runtime_patch/sitecustomize.py",
        "runtime_patch/load_audit.jsonl",
    }
    for relative, record in evidence["files"].items():
        path = failed_dir / relative
        assert record["sha256"] == _sha256(path)
        assert record["size_bytes"] == path.stat().st_size


def test_atomic_publication_uses_no_replace_rename(workspace, monkeypatch):
    calls = []
    real_rename = runner._rename_noreplace

    def recording_rename(source, destination):
        calls.append((Path(source), Path(destination)))
        return real_rename(source, destination)

    monkeypatch.setattr(runner, "_rename_noreplace", recording_rename)
    fake = FakeSubprocess(workspace.contract)

    _run(workspace, fake)

    assert len(calls) == 1
    staging, destination = calls[0]
    assert staging.parent == workspace.output_dir.parent
    assert staging.name.startswith(f".{workspace.output_dir.name}.")
    assert staging.name.endswith(".staging")
    assert destination == workspace.output_dir


def test_rejects_wrong_model_revision_and_cli_defaults_seed(workspace):
    fake = FakeSubprocess(workspace.contract)
    with pytest.raises(runner.CanaryError, match="model revision"):
        _run(workspace, fake, model_revision="c" * 40)
    assert not fake.calls

    args = runner.parse_args(
        [
            "--input-glb", str(workspace.input_glb),
            "--input-manifest", str(workspace.input_manifest),
            "--output-dir", str(workspace.output_dir),
            "--skintokens-root", str(workspace.skintokens_root),
            "--model-revision", workspace.contract.model_revision,
        ]
    )
    assert args.seed == 42
    assert args.use_skeleton_input is False


def test_production_recovery_contract_pins_the_exact_direct_attempt():
    contract = runner.PINNED_RECOVERY_CONTRACT

    assert contract.attempt_id == "22ff2d5d4b1181eb728a74c38f95dadf43bf3e45c963a3d3b31beb472759e4ed"
    assert contract.attempt_ledger_sha256 == "b76e3e65733151e394fbfe59a219162deba8bed9ca85faa4718181fa32b15d29"
    assert contract.attempt_ledger_size == 17979
    assert contract.failed_inventory_sha256 == "ac97e4b5293ae6cb073d39fbe4af598c7d43854572ea6b77ce00a4c248b0a9a8"
    assert contract.failed_glb_sha256 == "8606c013fba02f722e1d5c65accddc4398eab1fa925467a9233aaf458d93f01c"
    assert contract.failed_glb_size == 50843552
    assert contract.inference_log_sha256 == "c78cc5d4665a2794fe662bbd6a5abbfe05fcfd4942489aaf030c840a19f7f29d"
    assert contract.inference_log_size == 3372
    assert contract.sitecustomize_sha256 == "d643aa31ee39a12c356802bf659864328fa7ce7b51df9d6a3e7b8d85acbf207d"
    assert contract.sitecustomize_size == 2309
    assert contract.parser_sha256 == "ac186556d424b2581d0127e579440e605909f2420ae9e48bf8f401db31114a39"
    assert contract.bpyparser_load_sha256 == "26bde20077ca21e34822e6fbab5f6395924f097a94e7b21d0b2e656bd648d4a0"
    assert contract.clean_bpy_sha256 == "af3707436fa42f3c2388b65f0522684bff0c28eee9ac5334a77639e8b035cb62"


def test_recovery_manifest_authenticates_failed_attempt_without_moving_bytes(
    recovery_workspace,
):
    protected = {
        path: _sha256(path)
        for path in [recovery_workspace.attempt_ledger]
        + [path for path in recovery_workspace.failed_dir.rglob("*") if path.is_file()]
    }

    manifest_path = runner.recover_failed_attempt(
        contract=recovery_workspace.contract
    )

    assert manifest_path == recovery_workspace.recovery_output_dir / "tokenrig_manifest.json"
    assert {path: _sha256(path) for path in protected} == protected
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "pixal_tokenrig_recovery_v1"
    assert manifest["asset_id"] == runner.ASSET_ID
    assert manifest["source_front"] == "positive-y"
    assert manifest["canonical_front"] == "negative-y"
    assert manifest["attempt"] == "direct_transfer_recovered_from_hygiene_assertion"
    assert manifest["state_classification"] == "research_candidate_recovered_from_hygiene_assertion"
    assert manifest["task3_gate_status"] == "failed"
    assert manifest["pbr_validation_status"] == "pending_static_audit"
    assert manifest["input"]["glb"] == _record(recovery_workspace.contract.input_glb)
    assert manifest["input"]["manifest"] == _record(
        recovery_workspace.contract.input_manifest
    )
    assert manifest["output"]["path"] == str(recovery_workspace.failed_glb.resolve())
    assert manifest["output"]["sha256"] == _sha256(recovery_workspace.failed_glb)
    assert manifest["output"]["size_bytes"] == recovery_workspace.failed_glb.stat().st_size
    assert manifest["output"]["readback"]["skin_count"] >= 1
    recovery = manifest["recovery"]
    assert recovery["returncode"] == 0
    assert recovery["failure_stage"] == "output_validation"
    assert recovery["error"] == {
        "type": "CanaryError",
        "message": "server hygiene patch did not run in both demo and bpy server processes",
    }
    assert recovery["sitecustomize_import_failure"] == {
        "exception": "ModuleNotFoundError: No module named 'src'",
        "occurrences": 2,
    }
    assert recovery["attempt_ledger"] == _record(recovery_workspace.attempt_ledger)
    assert recovery["inference_log"] == _record(recovery_workspace.inference_log)
    assert recovery["failed_evidence"]["inventory_sha256"] == (
        recovery_workspace.contract.failed_inventory_sha256
    )
    clean = recovery["upstream_clean_bpy"]
    assert clean["skintokens_commit"] == recovery_workspace.contract.skintokens_commit
    assert clean["parser"] == _record(recovery_workspace.contract.parser_path)
    assert clean["bpyparser_load_sha256"] == recovery_workspace.contract.bpyparser_load_sha256
    assert clean["clean_bpy_sha256"] == recovery_workspace.contract.clean_bpy_sha256
    assert clean["bpyparser_load_calls_clean_before_import"] is True
    assert set(recovery_workspace.recovery_output_dir.iterdir()) == {manifest_path}


def test_recovery_publication_is_no_replace(recovery_workspace):
    manifest_path = runner.recover_failed_attempt(contract=recovery_workspace.contract)
    before = manifest_path.read_bytes()

    with pytest.raises(runner.CanaryError, match="stale output already exists"):
        runner.recover_failed_attempt(contract=recovery_workspace.contract)

    assert manifest_path.read_bytes() == before


def test_recovery_rejects_changed_failed_glb(recovery_workspace):
    recovery_workspace.failed_glb.write_bytes(
        recovery_workspace.failed_glb.read_bytes() + b"tamper"
    )

    with pytest.raises(runner.CanaryError, match="failed TokenRig GLB SHA-256"):
        runner.recover_failed_attempt(contract=recovery_workspace.contract)

    assert not recovery_workspace.recovery_output_dir.exists()


def test_recovery_rejects_nonexact_sitecustomize_failure(recovery_workspace):
    attempt = json.loads(recovery_workspace.attempt_ledger.read_text(encoding="utf-8"))
    attempt["stderr"] = (
        "Error in sitecustomize; set PYTHONVERBOSE for traceback:\n"
        "ModuleNotFoundError: No module named 'src'\n"
    )
    recovery_workspace.attempt_ledger.write_text(
        json.dumps(attempt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    contract = replace(
        recovery_workspace.contract,
        attempt_ledger_sha256=_sha256(recovery_workspace.attempt_ledger),
        attempt_ledger_size=recovery_workspace.attempt_ledger.stat().st_size,
    )

    with pytest.raises(runner.CanaryError, match="exactly two.*sitecustomize"):
        runner.recover_failed_attempt(contract=contract)


def test_recovery_rejects_parser_load_without_builtin_clean_first(recovery_workspace):
    parser_path = recovery_workspace.contract.parser_path
    source = parser_path.read_text(encoding="utf-8").replace(
        "        clean_bpy()\n        load(filepath=filepath, **kwargs)",
        "        load(filepath=filepath, **kwargs)\n        clean_bpy()",
    )
    parser_path.write_text(source, encoding="utf-8")
    contract = replace(
        recovery_workspace.contract,
        parser_sha256=_sha256(parser_path),
        parser_size=parser_path.stat().st_size,
        bpyparser_load_sha256=_source_hash(source, "BpyParser", "load"),
    )

    with pytest.raises(runner.CanaryError, match="clean_bpy before import"):
        runner.recover_failed_attempt(contract=contract)
