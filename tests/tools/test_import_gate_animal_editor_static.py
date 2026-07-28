import ast
import copy
import hashlib
import importlib.util
import io
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from tools import transcode_glb_webp_to_png as transcode_tool
ROOT = Path(__file__).resolve().parents[2]
IMPORT_ONE = ROOT / "tools/import_gate_animal_editor.py"
BATCH_IMPORT = ROOT / "tools/import_pixal_animal_batch_editor.py"


def _source(path):
    return path.read_text(encoding="utf-8")


def _function_source(path, name):
    source = _source(path)
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"missing function {name}")


def _load_module(path, name, monkeypatch, *, include_spear=False):
    monkeypatch.setitem(
        sys.modules,
        "unreal",
        SimpleNamespace(EditorAssetLibrary=SimpleNamespace()),
    )
    if include_spear:
        monkeypatch.setitem(sys.modules, "spear", SimpleNamespace())
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_glb(path):
    document = {
        "asset": {"version": "2.0"},
        "images": [],
        "animations": [{"name": "Idle"}, {"name": "Walking"}],
    }
    _write_glb_json(path, json.dumps(document, separators=(",", ":")).encode())


def _write_glb_json(path, payload):
    payload += b" " * ((4 - len(payload) % 4) % 4)
    total_length = 12 + 8 + len(payload)
    path.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total_length)
        + struct.pack("<II", len(payload), 0x4E4F534A)
        + payload
    )


def _write_editor_transcode_fixture(tmp_path, batch):
    webp_stream = io.BytesIO()
    Image.new("RGBA", (2, 2), (40, 90, 130, 255)).save(
        webp_stream,
        format="WEBP",
        lossless=True,
    )
    webp = webp_stream.getvalue()
    source = tmp_path / "reviewed.glb"
    binary = b"\x01\x02\x03\x04" + webp
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 4},
            {"buffer": 0, "byteOffset": 4, "byteLength": len(webp)},
        ],
        "images": [{"bufferView": 1, "mimeType": "image/webp"}],
        "textures": [
            {
                "source": 0,
                "extensions": {"EXT_texture_webp": {"source": 0}},
            }
        ],
        "extensionsUsed": ["EXT_texture_webp"],
        "extensionsRequired": ["EXT_texture_webp"],
        "animations": [{"name": "Idle"}, {"name": "Walking"}],
    }
    source.write_bytes(transcode_tool.encode_glb(document, binary))
    readback, readback_binary = transcode_tool.read_glb(source)
    rewritten, rewritten_binary, records = transcode_tool.transcode(
        readback,
        readback_binary,
    )
    output = tmp_path / "compatible.glb"
    output.write_bytes(transcode_tool.encode_glb(rewritten, rewritten_binary))
    manifest_path = tmp_path / "texture_transcode_manifest.json"
    _write_json(
        manifest_path,
        {
            "schema": batch.TEXTURE_TRANSCODE_SCHEMA,
            "purpose": batch.TEXTURE_TRANSCODE_PURPOSE,
            "geometry_skin_animation_byte_graph_changed": False,
            "input": _record(source),
            "output": _record(output),
            "images": records,
        },
    )
    job = {
        "rigged_glb_sha256": _sha256(output),
        "upstream_rigged_glb": str(source.resolve()),
        "upstream_rigged_glb_sha256": _sha256(source),
        "texture_transcode_manifest": str(manifest_path.resolve()),
        "texture_transcode_manifest_sha256": _sha256(manifest_path),
        "texture_transcode_manifest_size_bytes": manifest_path.stat().st_size,
    }
    return source, output, manifest_path, job


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path):
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _stubbed_presentation_evidence(tmp_path, batch, animation_review):
    presentation_root = tmp_path / "owner_review_presentation"
    presentation_root.mkdir()
    output_video = presentation_root / "owner_review_six_view.mp4"
    output_video.write_bytes(b"authenticated-owner-review-video")
    output_record = {
        **_record(output_video),
        "codec": "h264",
        "width": 1536,
        "height": 768,
        "frame_count": 8,
        "frame_rate": "8/1",
        "duration_seconds": 1.0,
        "readback": {
            "stream_count": 1,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1536,
            "height": 768,
            "nb_frames": 8,
        },
        "full_decode_passed": True,
    }
    receipt_path = presentation_root / "presentation_receipt.json"
    receipt_payload = {
        "schema": "owner_review_presentation_fixture_v1",
        "expected_source_review_sha256": _sha256(animation_review),
        "source_review": _record(animation_review),
        "output": copy.deepcopy(output_record),
    }
    receipt_payload["receipt_sha256"] = batch._hash_without(
        receipt_payload, "receipt_sha256"
    )
    _write_json(receipt_path, receipt_payload)
    evidence = {
        "presentation_receipt": _record(receipt_path),
        "expected_presentation_receipt_file_sha256": _sha256(receipt_path),
        "presentation_receipt_sha256": receipt_payload["receipt_sha256"],
        "output_video": copy.deepcopy(output_record),
    }
    canonical_evidence = copy.deepcopy(evidence)
    loader_calls = []

    def load_presentation_receipt(
        path,
        expected_receipt_file_sha256,
        *,
        expected_source_review_sha256,
    ):
        loader_calls.append(
            (
                Path(path),
                expected_receipt_file_sha256,
                expected_source_review_sha256,
            )
        )
        if (
            Path(path) != receipt_path.resolve()
            or expected_receipt_file_sha256
            != canonical_evidence["expected_presentation_receipt_file_sha256"]
            or expected_source_review_sha256 != _sha256(animation_review)
            or _record(receipt_path) != canonical_evidence["presentation_receipt"]
            or _record(output_video)
            != {
                key: canonical_evidence["output_video"][key]
                for key in ("path", "sha256", "size_bytes")
            }
            or {path.name for path in presentation_root.iterdir()}
            != {"presentation_receipt.json", "owner_review_six_view.mp4"}
        ):
            raise batch.contracts.ContractError(
                "stubbed presentation bytes or binding changed"
            )
        return (
            copy.deepcopy(receipt_payload),
            copy.deepcopy(canonical_evidence["presentation_receipt"]),
        )

    batch.presentation = SimpleNamespace(
        PresentationContractError=batch.contracts.ContractError,
        load_presentation_receipt=load_presentation_receipt,
    )
    return evidence, output_video, loader_calls


def _build_v3_contract(tmp_path, batch):
    asset_id = "animal_british_shorthair_v1"
    tag = f"pixal_{asset_id}"
    source_asset = tmp_path / "source_asset.json"
    source_registry = tmp_path / "source_registry.json"
    animation_review = tmp_path / "animation_review.json"
    animation_decision = tmp_path / "animation_decision.json"
    freeze_receipt = tmp_path / "animation_decision_freeze_receipt.json"
    rigged_glb = tmp_path / "reviewed_animated.glb"
    _write_json(
        source_asset,
        {
            "schema": "controlled_source_asset_v2",
            "asset_id": asset_id,
        },
    )
    _write_json(
        source_registry,
        {
            "schema": "controlled_source_asset_registry_v1",
            "asset_ids": [asset_id],
        },
    )
    _write_json(
        animation_review,
        {
            "schema": "generated_animal_animation_review_v4",
            "asset_id": asset_id,
        },
    )
    decision_payload = {
        "schema": "target_native_generated_animal_animation_decision_v1",
        "decision": "approved_for_ue_apartment",
        "review_sha256": _sha256(animation_review),
    }
    decision_payload["decision_sha256"] = batch._hash_without(
        decision_payload, "decision_sha256"
    )
    _write_json(animation_decision, decision_payload)
    _write_glb(rigged_glb)
    presentation_evidence, presentation_video, presentation_loader_calls = (
        _stubbed_presentation_evidence(tmp_path, batch, animation_review)
    )

    authority = copy.deepcopy(batch.USER_INSTRUCTION_AUTHORITY)
    receipt_payload = {
        "schema": batch.DECISION_FREEZE_RECEIPT_SCHEMA,
        "status": "frozen",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "source_asset_registry": _record(source_registry),
        "expected_source_asset_registry_file_sha256": _sha256(source_registry),
        "source_asset_registry_validation_mode": "current_exact_rebuild",
        "source_asset": _record(source_asset),
        "animation_review": _record(animation_review),
        "expected_animation_review_file_sha256": _sha256(animation_review),
        "user_instruction_binding": {
            "decision": "approved_for_ue_apartment",
            "review_sha256": _sha256(animation_review),
            "all_six_checks_explicit": True,
            "presentation_receipt_file_sha256": presentation_evidence[
                "expected_presentation_receipt_file_sha256"
            ],
        },
        "user_instruction_authority": authority,
        "authenticated_review_artifact_count": 1,
        "animation_decision": {
            "path": animation_decision.name,
            "sha256": _sha256(animation_decision),
            "size_bytes": animation_decision.stat().st_size,
        },
        "decision_sha256": decision_payload["decision_sha256"],
        "presentation_evidence": copy.deepcopy(presentation_evidence),
    }
    receipt_payload["receipt_sha256"] = batch._hash_without(
        receipt_payload, "receipt_sha256"
    )
    _write_json(freeze_receipt, receipt_payload)

    sampled_attributes = {"coat_length": "short", "tail_count": 1}
    job = {
        "job_type": batch.JOB_TYPE,
        "asset_id": asset_id,
        "legacy_tag": asset_id,
        "tag": tag,
        "profile_schema_id": "animal_british_shorthair_v1",
        "sampled_attributes": sampled_attributes,
        "expected_actions": list(batch.EXPECTED_ACTIONS),
        "rigged_glb": str(rigged_glb.resolve()),
        "rigged_glb_sha256": _sha256(rigged_glb),
        "source_registry_sha256": _sha256(source_registry),
        "source_asset_sha256": _sha256(source_asset),
        "request_sha256": "1" * 64,
        "animation_decision_file_sha256": _sha256(animation_decision),
        "animation_decision_sha256": decision_payload["decision_sha256"],
    }
    policy = (
        f"new unique gate_{tag} content directories; never rewrite or "
        "reuse any historical generated-animal UE job/content directory"
    )
    manifest = {
        "schema": batch.BATCH_SCHEMA,
        "status": "ready_for_new_ue_import",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "job_type": batch.JOB_TYPE,
        "job_count": 1,
        "jobs": [job],
        "non_destructive_policy": policy,
    }
    manifest["batch_sha256"] = batch._hash_without(manifest, "batch_sha256")
    manifest_path = tmp_path / "ue_import_jobs.json"
    _write_json(manifest_path, manifest)

    canonical_identity = {
        "asset_id": asset_id,
        "legacy_tag": asset_id,
        "tag": tag,
        "profile_schema_id": job["profile_schema_id"],
        "profile_sha256": "3" * 64,
        "request_sha256": job["request_sha256"],
        "taxonomy": {"species": "cat"},
        "fixed_attributes": {"tail_count": 1},
        "sampled_attributes": sampled_attributes,
        "target_physical_profile": {"leg_style": "proportional"},
    }
    preparation = {
        "schema": batch.PREPARATION_SCHEMA,
        "status": "ready_for_new_ue_import",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "created_at": "2026-07-28T00:00:00+00:00",
        "canonical_identity": canonical_identity,
        "source_asset": _record(source_asset),
        "source_asset_registry": _record(source_registry),
        "source_asset_registry_sha256": "4" * 64,
        "expected_source_asset_registry_file_sha256": _sha256(source_registry),
        "source_asset_registry_validation_mode": "current_exact_rebuild",
        "source_artifact_roots": {"generated_animal": str(tmp_path)},
        "authenticated_source_artifact_count": 1,
        "animation_review": _record(animation_review),
        "animation_review_schema": "review_v1",
        "animation_decision": _record(animation_decision),
        "expected_animation_decision_file_sha256": _sha256(animation_decision),
        "animation_decision_sha256": job["animation_decision_sha256"],
        "animation_decision_freeze_receipt": _record(freeze_receipt),
        "expected_animation_decision_freeze_receipt_file_sha256": _sha256(
            freeze_receipt
        ),
        "animation_decision_freeze_receipt_sha256": receipt_payload["receipt_sha256"],
        "presentation_evidence": copy.deepcopy(presentation_evidence),
        "user_instruction_authority": authority,
        "reviewed_animated_glb": _record(rigged_glb),
        "authenticated_review_artifact_count": 1,
        "ue_import_jobs": {
            "path": manifest_path.name,
            "sha256": _sha256(manifest_path),
            "size_bytes": manifest_path.stat().st_size,
        },
        "automatic_checks": {
            field: ("passed" if field == "overall" else True)
            for field in batch.PREPARATION_AUTOMATIC_CHECK_FIELDS
        },
    }
    preparation["manifest_sha256"] = batch._hash_without(preparation, "manifest_sha256")
    preparation_path = tmp_path / "ue_import_preparation_manifest.json"
    _write_json(preparation_path, preparation)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "preparation": preparation,
        "preparation_path": preparation_path,
        "freeze_receipt": receipt_payload,
        "freeze_receipt_path": freeze_receipt,
        "presentation_evidence": presentation_evidence,
        "presentation_video_path": presentation_video,
        "presentation_loader_calls": presentation_loader_calls,
    }


def test_interchange_glb_import_joins_before_asset_readback():
    text = _source(IMPORT_ONE)

    import_call = text.index("asset_tools.import_asset_tasks")
    blocking_join = text.index("task.get_objects()", import_call)
    registry_join = text.index("wait_for_completion()", blocking_join)
    directory_readback = text.index(
        "unreal.EditorAssetLibrary.list_assets", registry_join
    )

    assert import_call < blocking_join < registry_join < directory_readback
    assert 'name="async_", value=True' in text
    assert "save_directory(" in text


def test_pixal_batch_preflights_ue_texture_compatibility():
    text = _source(BATCH_IMPORT)

    preflight = text.index("_validate_ue_compatible_glb(job, source)")
    ue_import = text.index("runpy.run_path", preflight)

    assert preflight < ue_import
    assert '"EXT_texture_webp" in required' in text
    assert 'image.get("mimeType") == "image/webp"' in text
    assert "geometry_skin_animation_byte_graph_changed" in text


def test_editor_revalidates_manifest_descriptor_and_complete_bin_graph(
    tmp_path,
    monkeypatch,
):
    batch = _load_module(
        BATCH_IMPORT,
        "_test_editor_transcode_complete_graph",
        monkeypatch,
    )
    reviewed, output, manifest_path, job = _write_editor_transcode_fixture(
        tmp_path,
        batch,
    )

    batch._validate_texture_transcode_lineage(
        job,
        source=output.resolve(),
        reviewed_source=reviewed.resolve(),
    )

    document, binary = transcode_tool.read_glb(output)
    tampered = bytearray(binary)
    tampered[0] ^= 1
    output.write_bytes(transcode_tool.encode_glb(document, bytes(tampered)))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output"] = _record(output)
    _write_json(manifest_path, manifest)
    job["rigged_glb_sha256"] = _sha256(output)
    job["texture_transcode_manifest_sha256"] = _sha256(manifest_path)
    job["texture_transcode_manifest_size_bytes"] = manifest_path.stat().st_size

    with pytest.raises(RuntimeError, match="original GLB byte graph"):
        batch._validate_texture_transcode_lineage(
            job,
            source=output.resolve(),
            reviewed_source=reviewed.resolve(),
        )


def test_editor_rejects_changed_transcode_manifest_descriptor(
    tmp_path,
    monkeypatch,
):
    batch = _load_module(
        BATCH_IMPORT,
        "_test_editor_transcode_manifest_descriptor",
        monkeypatch,
    )
    _reviewed, output, manifest_path, job = _write_editor_transcode_fixture(
        tmp_path,
        batch,
    )
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

    with pytest.raises(RuntimeError, match="does not authenticate"):
        batch._load_texture_transcode_manifest(job, output.resolve())


@pytest.mark.parametrize("residual_location", ["extensionsUsed", "texture"])
def test_editor_rejects_any_residual_webp_extension(
    tmp_path,
    monkeypatch,
    residual_location,
):
    batch = _load_module(
        BATCH_IMPORT,
        f"_test_editor_residual_webp_{residual_location}",
        monkeypatch,
    )
    document = {
        "asset": {"version": "2.0"},
        "images": [{"mimeType": "image/png"}],
        "textures": [{"source": 0}],
        "animations": [{"name": "Idle"}, {"name": "Walking"}],
    }
    if residual_location == "extensionsUsed":
        document["extensionsUsed"] = ["EXT_texture_webp"]
    else:
        document["textures"][0]["extensions"] = {
            "EXT_texture_webp": {"source": 0}
        }
    source = tmp_path / f"{residual_location}.glb"
    _write_glb_json(
        source,
        json.dumps(document, separators=(",", ":")).encode("utf-8"),
    )

    with pytest.raises(RuntimeError, match="still requires embedded WebP"):
        batch._validate_ue_compatible_glb({}, source)


def test_existing_gate_directory_is_unchanged_and_creation_fails():
    existing = "/Game/MyAssets/Audioset/Meshes/gate_pixal_existing"

    class FakeEditorAssetLibrary:
        contents = {existing: ["sentinel.uasset", "nested/keep.uasset"]}
        make_calls = []

        @classmethod
        def does_directory_exist(cls, *, directory_path):
            return directory_path in cls.contents

        @classmethod
        def make_directory(cls, *, directory_path):
            cls.make_calls.append(directory_path)
            cls.contents[directory_path] = []
            return True

    fake_unreal = type(
        "FakeUnreal",
        (),
        {"EditorAssetLibrary": FakeEditorAssetLibrary},
    )
    namespace = {"unreal": fake_unreal}
    exec(_function_source(IMPORT_ONE, "_make_or_clear_dir"), namespace)
    before = copy.deepcopy(FakeEditorAssetLibrary.contents)

    with pytest.raises(RuntimeError, match="refusing to replace"):
        namespace["_make_or_clear_dir"](existing)

    assert FakeEditorAssetLibrary.contents == before
    assert FakeEditorAssetLibrary.make_calls == []


def test_gate_directory_is_created_only_when_absent():
    new_path = "/Game/MyAssets/Audioset/Meshes/gate_pixal_new"

    class FakeEditorAssetLibrary:
        contents = {}

        @classmethod
        def does_directory_exist(cls, *, directory_path):
            return directory_path in cls.contents

        @classmethod
        def make_directory(cls, *, directory_path):
            if directory_path in cls.contents:
                return False
            cls.contents[directory_path] = []
            return True

    fake_unreal = type(
        "FakeUnreal",
        (),
        {"EditorAssetLibrary": FakeEditorAssetLibrary},
    )
    namespace = {"unreal": fake_unreal}
    exec(_function_source(IMPORT_ONE, "_make_or_clear_dir"), namespace)

    namespace["_make_or_clear_dir"](new_path)

    assert FakeEditorAssetLibrary.contents == {new_path: []}


def test_batch_rejects_any_existing_target_without_changing_ue_content():
    tag = "pixal_existing"
    mesh_dir = f"/Game/MyAssets/Audioset/Meshes/gate_{tag}"
    bp_dir = f"/Game/MyAssets/Audioset/Blueprints/gate_{tag}"
    bp_path = f"{bp_dir}/BP_gate_{tag}"

    class FakeEditorAssetLibrary:
        directories = {mesh_dir: ["sentinel.uasset"]}
        assets = {bp_path: {"sentinel": "keep"}}

        @classmethod
        def does_directory_exist(cls, *, directory_path):
            return directory_path in cls.directories

        @classmethod
        def does_asset_exist(cls, *, asset_path):
            return asset_path in cls.assets

    fake_unreal = type(
        "FakeUnreal",
        (),
        {"EditorAssetLibrary": FakeEditorAssetLibrary},
    )
    namespace = {"unreal": fake_unreal}
    exec(_function_source(BATCH_IMPORT, "_content_targets"), namespace)
    exec(
        _function_source(BATCH_IMPORT, "_assert_content_targets_absent"),
        namespace,
    )
    before = (
        copy.deepcopy(FakeEditorAssetLibrary.directories),
        copy.deepcopy(FakeEditorAssetLibrary.assets),
    )

    with pytest.raises(RuntimeError, match="refusing to replace"):
        namespace["_assert_content_targets_absent"]([tag, "pixal_new"])

    assert FakeEditorAssetLibrary.directories == before[0]
    assert FakeEditorAssetLibrary.assets == before[1]


def test_batch_result_writer_refuses_to_replace_existing_evidence(tmp_path):
    result = tmp_path / "ue_import_result.json"
    result.write_bytes(b"sentinel-result\n")
    namespace = {"Path": Path, "json": json}
    exec(_function_source(BATCH_IMPORT, "_write_result_no_replace"), namespace)

    with pytest.raises(RuntimeError, match="refusing to replace"):
        namespace["_write_result_no_replace"](result, {"status": "passed"})

    assert result.read_bytes() == b"sentinel-result\n"


def test_importers_have_no_existing_content_delete_or_replace_path():
    gate = _source(IMPORT_ONE)
    batch = _source(BATCH_IMPORT)
    batch_main = _function_source(BATCH_IMPORT, "main")

    assert "delete_directory" not in gate
    assert "delete_asset" not in gate
    assert 'name="replace_existing", value=False' in gate
    assert 'name="replace_existing_settings", value=False' in gate
    assert 'name="replace_existing", value=True' not in gate
    assert 'name="replace_existing_settings", value=True' not in gate
    assert "delete_directory" not in batch
    assert "delete_asset" not in batch

    batch_preflight = batch_main.index("_assert_content_targets_absent(tags)")
    first_import = batch_main.index("_run_one(")
    assert batch_preflight < first_import
    assert "runpy.run_path" in _function_source(BATCH_IMPORT, "_run_one")
    assert 'path.open("x", encoding="utf-8")' in batch
    assert ".write_text(" not in batch


def test_formal_pixal_import_contract_uses_v3_preparation_and_is_externally_anchored(
    tmp_path, monkeypatch
):
    batch = _load_module(
        BATCH_IMPORT,
        "_test_import_pixal_animal_batch_contract",
        monkeypatch,
    )
    fixture = _build_v3_contract(tmp_path, batch)
    preparation, preparation_descriptor, manifest_descriptor = (
        batch._validate_preparation_anchor(
            fixture["preparation_path"],
            _sha256(fixture["preparation_path"]),
            fixture["manifest_path"],
            _sha256(fixture["manifest_path"]),
        )
    )
    jobs, identity = batch._validate_batch_payload(fixture["manifest"], preparation)

    assert jobs[0]["expected_actions"] == ["Idle", "Walking"]
    assert identity == {
        "schema": "pixal_animal_ue_import_batch_v2",
        "job_type": "user_approved_generated_animal",
        "job_count": 1,
        "batch_sha256": fixture["manifest"]["batch_sha256"],
        "asset_ids": ["animal_british_shorthair_v1"],
        "tags": ["pixal_animal_british_shorthair_v1"],
        "job_identity_sha256s": [batch._json_sha256(jobs[0])],
    }
    assert preparation_descriptor["sha256"] == _sha256(fixture["preparation_path"])
    assert preparation_descriptor["manifest_sha256"] == preparation["manifest_sha256"]
    assert manifest_descriptor["sha256"] == _sha256(fixture["manifest_path"])
    assert preparation["presentation_evidence"] == fixture["presentation_evidence"]
    assert fixture["presentation_loader_calls"]

    result_path = tmp_path / "ue_import_result.json"
    monkeypatch.setenv(
        "PIXAL_ANIMAL_IMPORT_PREPARATION",
        str(fixture["preparation_path"]),
    )
    monkeypatch.setenv(
        "PIXAL_ANIMAL_IMPORT_PREPARATION_SHA256",
        _sha256(fixture["preparation_path"]),
    )
    monkeypatch.setenv("PIXAL_ANIMAL_IMPORT_MANIFEST", str(fixture["manifest_path"]))
    monkeypatch.setenv(
        "PIXAL_ANIMAL_IMPORT_MANIFEST_SHA256",
        _sha256(fixture["manifest_path"]),
    )
    monkeypatch.setenv("PIXAL_ANIMAL_IMPORT_RESULT", str(result_path))
    authenticated = batch._authenticate_import_contract()
    assert authenticated["batch_identity"] == identity
    assert authenticated["result_path"] == result_path
    assert not result_path.exists()


def test_presentation_video_and_cross_layer_binding_mutations_fail_closed(
    tmp_path, monkeypatch
):
    batch = _load_module(
        BATCH_IMPORT,
        "_test_import_pixal_presentation_mutations",
        monkeypatch,
    )
    fixture = _build_v3_contract(tmp_path, batch)

    fixture["presentation_video_path"].write_bytes(b"replaced-video")
    with pytest.raises(RuntimeError, match="presentation evidence is invalid"):
        batch._validate_preparation_anchor(
            fixture["preparation_path"],
            _sha256(fixture["preparation_path"]),
            fixture["manifest_path"],
            _sha256(fixture["manifest_path"]),
        )

    fresh_root = tmp_path / "fresh"
    fresh_root.mkdir()
    fixture = _build_v3_contract(fresh_root, batch)
    preparation = copy.deepcopy(fixture["preparation"])
    preparation["presentation_evidence"]["presentation_receipt_sha256"] = "0" * 64
    preparation["manifest_sha256"] = batch._hash_without(preparation, "manifest_sha256")
    changed_path = tmp_path / "changed_presentation_preparation.json"
    _write_json(changed_path, preparation)
    with pytest.raises(RuntimeError, match="presentation binding changed"):
        batch._validate_preparation_anchor(
            changed_path,
            _sha256(changed_path),
            fixture["manifest_path"],
            _sha256(fixture["manifest_path"]),
        )


def test_legacy_freeze_v1_and_resealed_wrong_user_presentation_sha_fail_closed(
    tmp_path, monkeypatch
):
    batch = _load_module(
        BATCH_IMPORT,
        "_test_import_pixal_freeze_v2_only",
        monkeypatch,
    )
    fixture = _build_v3_contract(tmp_path, batch)

    for index, mutate in enumerate(
        (
            lambda receipt: receipt.update(
                schema=(
                    "avengine_target_native_generated_animal_animation_"
                    "decision_freeze_receipt_v1"
                )
            ),
            lambda receipt: receipt["user_instruction_binding"].update(
                presentation_receipt_file_sha256="0" * 64
            ),
        )
    ):
        receipt = copy.deepcopy(fixture["freeze_receipt"])
        mutate(receipt)
        receipt["receipt_sha256"] = batch._hash_without(receipt, "receipt_sha256")
        receipt_path = tmp_path / f"mutated_freeze_{index}.json"
        _write_json(receipt_path, receipt)
        preparation = copy.deepcopy(fixture["preparation"])
        preparation["animation_decision_freeze_receipt"] = _record(receipt_path)
        preparation["expected_animation_decision_freeze_receipt_file_sha256"] = _sha256(
            receipt_path
        )
        preparation["animation_decision_freeze_receipt_sha256"] = receipt[
            "receipt_sha256"
        ]
        preparation["manifest_sha256"] = batch._hash_without(
            preparation, "manifest_sha256"
        )
        preparation_path = tmp_path / f"mutated_freeze_preparation_{index}.json"
        _write_json(preparation_path, preparation)
        with pytest.raises(RuntimeError):
            batch._validate_preparation_anchor(
                preparation_path,
                _sha256(preparation_path),
                fixture["manifest_path"],
                _sha256(fixture["manifest_path"]),
            )


def test_import_contract_is_reauthenticated_before_and_after_ue_writes(
    monkeypatch,
):
    source = _function_source(BATCH_IMPORT, "main")
    target_preflight = source.index("_assert_content_targets_absent(tags)")
    first_reauthentication = source.index(
        "_require_import_contract_unchanged(contract)",
        target_preflight,
    )
    first_write = source.index("_run_one(", first_reauthentication)
    success_result = source.index("success_result = _base_result", first_write)
    final_reauthentication = source.index(
        "_require_import_contract_unchanged(contract)",
        success_result,
    )
    success_receipt = source.index(
        '_write_result_no_replace(contract["result_path"], success_result)',
        final_reauthentication,
    )

    assert (
        target_preflight
        < first_reauthentication
        < first_write
        < success_result
        < final_reauthentication
        < success_receipt
    )

    batch = _load_module(
        BATCH_IMPORT,
        "_test_import_pixal_contract_reauthentication",
        monkeypatch,
    )
    monkeypatch.setattr(
        batch,
        "_authenticate_import_contract",
        lambda: {"authority": "changed"},
    )
    with pytest.raises(RuntimeError, match="authority graph changed"):
        batch._require_import_contract_unchanged({"authority": "original"})


def test_batch_contract_mutations_fail_closed_before_ue_write(tmp_path, monkeypatch):
    batch = _load_module(
        BATCH_IMPORT,
        "_test_import_pixal_animal_batch_mutations",
        monkeypatch,
    )
    fixture = _build_v3_contract(tmp_path, batch)
    preparation = fixture["preparation"]

    def rehash(payload):
        payload["batch_sha256"] = batch._hash_without(payload, "batch_sha256")

    mutations = [
        lambda payload: payload.update(schema=batch.LEGACY_BATCH_SCHEMA),
        lambda payload: payload.update(status="ready_for_ue_import"),
        lambda payload: payload.update(job_count=2),
        lambda payload: payload.update(job_type="generic_animal"),
        lambda payload: payload.update(extra_field=True),
        lambda payload: payload.update(non_destructive_policy="unique enough"),
        lambda payload: payload["jobs"][0].update(expected_actions=["Walking", "Idle"]),
        lambda payload: payload["jobs"][0].update(expected_actions=["Idle", "Idle"]),
        lambda payload: payload["jobs"][0].update(legacy_tag="old_cat"),
        lambda payload: payload["jobs"][0].update(tag="pixal_unbound_identity"),
        lambda payload: payload["jobs"][0].update(job_type="generic_animal"),
        lambda payload: payload["jobs"][0].update(extra_field=True),
    ]
    for mutate in mutations:
        payload = copy.deepcopy(fixture["manifest"])
        mutate(payload)
        rehash(payload)
        with pytest.raises(RuntimeError):
            batch._validate_batch_payload(payload, preparation)

    bad_hash = copy.deepcopy(fixture["manifest"])
    bad_hash["batch_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="top-level contract"):
        batch._validate_batch_payload(bad_hash, preparation)


def test_preparation_and_manifest_anchor_mutations_fail_closed(tmp_path, monkeypatch):
    batch = _load_module(
        BATCH_IMPORT,
        "_test_import_pixal_preparation_mutations",
        monkeypatch,
    )
    fixture = _build_v3_contract(tmp_path, batch)
    with pytest.raises(RuntimeError, match="external UE import preparation"):
        batch._validate_preparation_anchor(
            fixture["preparation_path"],
            "0" * 64,
            fixture["manifest_path"],
            _sha256(fixture["manifest_path"]),
        )

    mutations = [
        lambda payload: payload.update(extra_field=True),
        lambda payload: payload.update(schema="wrong_preparation"),
        lambda payload: payload.update(formal_dataset_registration_authorized=True),
        lambda payload: payload["automatic_checks"].update(overall="failed"),
        lambda payload: payload["automatic_checks"].update(
            raw_pixal_geometry_to_tokenrig_target_closure_reauthenticated=False
        ),
        lambda payload: payload["ue_import_jobs"].update(sha256="0" * 64),
    ]
    for index, mutate in enumerate(mutations):
        payload = copy.deepcopy(fixture["preparation"])
        mutate(payload)
        payload["manifest_sha256"] = batch._hash_without(payload, "manifest_sha256")
        path = tmp_path / f"mutated_preparation_{index}.json"
        _write_json(path, payload)
        with pytest.raises(RuntimeError):
            batch._validate_preparation_anchor(
                path,
                _sha256(path),
                fixture["manifest_path"],
                _sha256(fixture["manifest_path"]),
            )


@pytest.mark.parametrize(
    "legacy_schema",
    [
        "avengine_user_approved_generated_animal_ue_import_preparation_v1",
        "avengine_user_approved_generated_animal_ue_import_preparation_v2",
    ],
)
def test_legacy_preparation_v1_v2_is_explicitly_audit_only(
    tmp_path, monkeypatch, legacy_schema
):
    batch = _load_module(
        BATCH_IMPORT,
        "_test_import_pixal_legacy_preparation",
        monkeypatch,
    )
    fixture = _build_v3_contract(tmp_path, batch)
    preparation = copy.deepcopy(fixture["preparation"])
    preparation["schema"] = legacy_schema
    preparation["manifest_sha256"] = batch._hash_without(preparation, "manifest_sha256")
    path = tmp_path / "legacy_preparation.json"
    _write_json(path, preparation)

    with pytest.raises(RuntimeError, match="v1/v2 is audit-only"):
        batch._validate_preparation_anchor(
            path,
            _sha256(path),
            fixture["manifest_path"],
            _sha256(fixture["manifest_path"]),
        )


@pytest.mark.parametrize("invalid_value", ["duplicate_key", "nonfinite"])
def test_freeze_receipt_strict_json_failures_are_rejected(
    tmp_path, monkeypatch, invalid_value
):
    batch = _load_module(
        BATCH_IMPORT,
        f"_test_import_pixal_receipt_strict_{invalid_value}",
        monkeypatch,
    )
    fixture = _build_v3_contract(tmp_path, batch)
    receipt_path = fixture["freeze_receipt_path"]
    raw = receipt_path.read_text(encoding="utf-8")
    if invalid_value == "duplicate_key":
        raw = raw.replace(
            '  "schema":',
            '  "schema": "duplicate",\n  "schema":',
            1,
        )
    else:
        raw = raw.replace(
            '  "authenticated_review_artifact_count": 1,',
            '  "authenticated_review_artifact_count": NaN,',
            1,
        )
    receipt_path.write_text(raw, encoding="utf-8")

    preparation = copy.deepcopy(fixture["preparation"])
    preparation["animation_decision_freeze_receipt"] = _record(receipt_path)
    preparation["expected_animation_decision_freeze_receipt_file_sha256"] = _sha256(
        receipt_path
    )
    preparation["manifest_sha256"] = batch._hash_without(preparation, "manifest_sha256")
    preparation_path = tmp_path / f"preparation_{invalid_value}.json"
    _write_json(preparation_path, preparation)

    with pytest.raises(RuntimeError, match="not readable strict JSON"):
        batch._validate_preparation_anchor(
            preparation_path,
            _sha256(preparation_path),
            fixture["manifest_path"],
            _sha256(fixture["manifest_path"]),
        )


def test_freeze_receipt_cannot_upgrade_caller_assertion_to_crypto_identity(
    tmp_path, monkeypatch
):
    batch = _load_module(
        BATCH_IMPORT,
        "_test_import_pixal_receipt_authority_upgrade",
        monkeypatch,
    )
    fixture = _build_v3_contract(tmp_path, batch)
    receipt = copy.deepcopy(fixture["freeze_receipt"])
    upgraded_authority = copy.deepcopy(batch.USER_INSTRUCTION_AUTHORITY)
    upgraded_authority["cryptographic_user_identity_verified"] = True
    receipt["user_instruction_authority"] = upgraded_authority
    receipt["receipt_sha256"] = batch._hash_without(receipt, "receipt_sha256")
    _write_json(fixture["freeze_receipt_path"], receipt)

    preparation = copy.deepcopy(fixture["preparation"])
    preparation["animation_decision_freeze_receipt"] = _record(
        fixture["freeze_receipt_path"]
    )
    preparation["expected_animation_decision_freeze_receipt_file_sha256"] = _sha256(
        fixture["freeze_receipt_path"]
    )
    preparation["animation_decision_freeze_receipt_sha256"] = receipt["receipt_sha256"]
    preparation["user_instruction_authority"] = upgraded_authority
    preparation["manifest_sha256"] = batch._hash_without(preparation, "manifest_sha256")
    preparation_path = tmp_path / "preparation_crypto_upgrade.json"
    _write_json(preparation_path, preparation)

    with pytest.raises(RuntimeError, match="authority is invalid"):
        batch._validate_preparation_anchor(
            preparation_path,
            _sha256(preparation_path),
            fixture["manifest_path"],
            _sha256(fixture["manifest_path"]),
        )


def test_importer_rejects_arbitrary_symlink_parent(tmp_path, monkeypatch):
    batch = _load_module(
        BATCH_IMPORT,
        "_test_import_pixal_arbitrary_symlink_parent",
        monkeypatch,
    )
    real_parent = tmp_path / "real_parent"
    real_parent.mkdir()
    source = real_parent / "source.json"
    _write_json(source, {"schema": "fixture_v1"})
    linked_parent = tmp_path / "linked_parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink path component"):
        batch._direct_absolute_file(
            str(linked_parent / source.name),
            "arbitrary-parent fixture",
        )
    with pytest.raises(RuntimeError, match="symlink path component"):
        batch._validate_result_target(str(linked_parent / "result.json"))


def test_importer_allows_only_exact_spear_tmp_bridge(tmp_path, monkeypatch):
    batch = _load_module(
        BATCH_IMPORT,
        "_test_import_pixal_exact_tmp_bridge",
        monkeypatch,
    )
    physical_tmp = tmp_path / "physical_tmp"
    physical_tmp.mkdir()
    exact_bridge = tmp_path / "tmp"
    exact_bridge.symlink_to(physical_tmp, target_is_directory=True)
    monkeypatch.setattr(batch, "SPEAR_TMP_BRIDGE", exact_bridge.absolute())

    suffix = hashlib.sha256(str(tmp_path).encode()).hexdigest()[:16]
    source = batch.SPEAR_TMP_BRIDGE / f"importer_bridge_{suffix}.json"
    result = batch.SPEAR_TMP_BRIDGE / f"importer_result_{suffix}.json"
    nested_link = batch.SPEAR_TMP_BRIDGE / f"importer_nested_{suffix}"
    assert not source.exists()
    assert not result.exists()
    assert not nested_link.exists()
    try:
        _write_json(source, {"schema": "fixture_v1"})
        assert (
            batch._direct_absolute_file(str(source), "exact SPEAR/tmp bridge fixture")
            == source.resolve()
        )
        assert batch._validate_result_target(str(result)) == result

        nested_link.symlink_to(tmp_path, target_is_directory=True)
        nested_source = tmp_path / "nested_source.json"
        _write_json(nested_source, {"schema": "fixture_v1"})
        with pytest.raises(RuntimeError, match="symlink path component"):
            batch._direct_absolute_file(
                str(nested_link / nested_source.name),
                "nested bridge fixture",
            )
    finally:
        source.unlink(missing_ok=True)
        nested_link.unlink(missing_ok=True)


def test_pixal_per_asset_prewrite_requires_bound_identity(tmp_path, monkeypatch):
    source = tmp_path / "animal.glb"
    _write_glb(source)
    monkeypatch.setenv("GATE_TAG", "pixal_animal_british_shorthair_v1")
    monkeypatch.setenv("GATE_RIGGED_GLB", str(source.resolve()))
    monkeypatch.delenv("GATE_IMPORT_JOB_JSON", raising=False)
    gate = _load_module(
        IMPORT_ONE,
        "_test_import_gate_prewrite_contract",
        monkeypatch,
        include_spear=True,
    )

    with pytest.raises(RuntimeError, match="authenticated"):
        gate._validate_prewrite_contract()

    source_sha256 = _sha256(source)
    identity = {
        "schema": gate.IMPORT_JOB_IDENTITY_SCHEMA,
        "job_type": gate.PIXAL_JOB_TYPE,
        "asset_id": "animal_british_shorthair_v1",
        "legacy_tag": "animal_british_shorthair_v1",
        "tag": "pixal_animal_british_shorthair_v1",
        "expected_actions": ["Idle", "Walking"],
        "rigged_glb": str(source.resolve()),
        "rigged_glb_sha256": source_sha256,
        "input_manifest_sha256": "1" * 64,
        "batch_sha256": "2" * 64,
        "job_identity_sha256": "3" * 64,
    }
    monkeypatch.setenv("GATE_IMPORT_JOB_JSON", json.dumps(identity))
    validated_source, job_identity = gate._validate_prewrite_contract()
    assert validated_source == source.resolve()
    assert job_identity == "3" * 64

    identity["expected_actions"] = ["Walking", "Idle"]
    monkeypatch.setenv("GATE_IMPORT_JOB_JSON", json.dumps(identity))
    with pytest.raises(RuntimeError, match="identity mismatched"):
        gate._validate_prewrite_contract()


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        (
            '{"schema":"identity","schema":"shadow"}',
            "duplicate JSON object key",
        ),
        ('{"value":NaN}', "non-finite JSON number"),
        ('{"value":Infinity}', "non-finite JSON number"),
        ('{"value":-Infinity}', "non-finite JSON number"),
        ('{"value":1e999}', "non-finite JSON number"),
    ],
)
def test_per_asset_job_identity_uses_central_strict_json(
    tmp_path, monkeypatch, payload, expected_error
):
    source = tmp_path / "animal.glb"
    _write_glb(source)
    monkeypatch.setenv("GATE_TAG", "animal_fixture")
    monkeypatch.setenv("GATE_RIGGED_GLB", str(source.resolve()))
    monkeypatch.delenv("GATE_IMPORT_JOB_JSON", raising=False)
    gate = _load_module(
        IMPORT_ONE,
        f"_test_import_gate_strict_identity_{hashlib.sha256(payload.encode()).hexdigest()}",
        monkeypatch,
        include_spear=True,
    )

    with pytest.raises(RuntimeError, match="not strict JSON") as caught:
        gate._load_job_identity(payload)
    assert expected_error in str(caught.value.__cause__)


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        (
            b'{"asset":{"version":"2.0"},"animations":[],"animations":[]}',
            "duplicate JSON object key",
        ),
        (
            b'{"asset":{"version":"2.0"},"score":NaN}',
            "non-finite JSON number",
        ),
        (
            b'{"asset":{"version":"2.0"},"score":Infinity}',
            "non-finite JSON number",
        ),
        (
            b'{"asset":{"version":"2.0"},"score":-Infinity}',
            "non-finite JSON number",
        ),
        (
            b'{"asset":{"version":"2.0"},"score":1e999}',
            "non-finite JSON number",
        ),
    ],
)
def test_per_asset_glb_json_chunk_uses_central_strict_json(
    tmp_path, monkeypatch, payload, expected_error
):
    source = tmp_path / "animal.glb"
    _write_glb_json(source, payload)
    monkeypatch.setenv("GATE_TAG", "animal_fixture")
    monkeypatch.setenv("GATE_RIGGED_GLB", str(source.resolve()))
    monkeypatch.delenv("GATE_IMPORT_JOB_JSON", raising=False)
    gate = _load_module(
        IMPORT_ONE,
        f"_test_import_gate_strict_glb_{hashlib.sha256(payload).hexdigest()}",
        monkeypatch,
        include_spear=True,
    )

    with pytest.raises(RuntimeError, match="GLB JSON document is invalid") as caught:
        gate._read_glb_document(source, _sha256(source))
    assert expected_error in str(caught.value.__cause__)


def test_per_asset_glb_external_sha_is_checked_before_json_parse(tmp_path, monkeypatch):
    source = tmp_path / "animal.glb"
    _write_glb_json(
        source,
        b'{"asset":{"version":"2.0"},"animations":[],"animations":[]}',
    )
    monkeypatch.setenv("GATE_TAG", "animal_fixture")
    monkeypatch.setenv("GATE_RIGGED_GLB", str(source.resolve()))
    monkeypatch.delenv("GATE_IMPORT_JOB_JSON", raising=False)
    gate = _load_module(
        IMPORT_ONE,
        "_test_import_gate_sha_before_glb_parse",
        monkeypatch,
        include_spear=True,
    )
    parse_calls = []

    def record_parse(value):
        parse_calls.append(value)
        raise AssertionError("GLB JSON parsed before its external SHA check")

    monkeypatch.setattr(gate.contracts, "strict_json_loads", record_parse)

    with pytest.raises(RuntimeError, match="external rigged GLB SHA-256"):
        gate._read_glb_document(source, "0" * 64)
    assert parse_calls == []


def test_exact_import_and_blueprint_readback_is_fail_closed():
    gate = _source(IMPORT_ONE)
    batch = _source(BATCH_IMPORT)
    gate_main = _function_source(IMPORT_ONE, "main")

    assert not any(
        isinstance(node, ast.Assert)
        for path in (IMPORT_ONE, BATCH_IMPORT)
        for node in ast.walk(ast.parse(_source(path)))
    )
    prewrite = gate_main.index("_validate_prewrite_contract()")
    first_directory_create = gate_main.index("_make_or_clear_dir")
    assert prewrite < first_directory_create
    assert "len(skeletal_mesh_paths) != 1" in gate
    assert "set(animation_paths) != set(EXPECTED_ACTIONS)" in gate
    assert "readback_component.get_skeletal_mesh_asset()" in gate
    assert 'name="anim_to_play"' in gate
    assert "failed_write_residue_requires_manual_audit" in batch
    assert '"existing_content_was_deleted": False' in batch
    assert '"preparation_manifest": contract["preparation_descriptor"]' in batch
    assert '"input_manifest": contract["manifest_descriptor"]' in batch
    assert '"batch_identity": contract["batch_identity"]' in batch


def test_batch_exact_asset_class_readback_rejects_mutations(monkeypatch):
    batch = _load_module(
        BATCH_IMPORT,
        "_test_import_pixal_exact_asset_readback",
        monkeypatch,
    )
    tag = "pixal_animal_british_shorthair_v1"
    mesh_dir, _bp_dir, bp_path = batch._content_targets(tag)
    assets = ["mesh", "idle", "walking", "material"]
    object_details = {
        "mesh": ("SkeletalMesh", "Cat", "/Game/Test/Cat.Cat"),
        "idle": ("AnimSequence", "Idle", "/Game/Test/Idle.Idle"),
        "walking": (
            "AnimSequence",
            "Walking",
            "/Game/Test/Walking.Walking",
        ),
        "material": ("Material", "Fur", "/Game/Test/Fur.Fur"),
    }

    class FakeEditorAssetLibrary:
        @staticmethod
        def list_assets(*, directory_path, recursive):
            assert directory_path == mesh_dir
            assert recursive is True
            return list(assets)

        @staticmethod
        def does_asset_exist(*, asset_path):
            return asset_path == bp_path

    batch.unreal.EditorAssetLibrary = FakeEditorAssetLibrary
    batch._object_path = lambda asset: object_details[asset]
    validated_job = {
        "job": {
            "job_type": batch.JOB_TYPE,
            "asset_id": "animal_british_shorthair_v1",
            "tag": tag,
            "legacy_tag": "animal_british_shorthair_v1",
            "rigged_glb_sha256": "1" * 64,
        },
        "source": Path("/tmp/cat.glb"),
        "job_identity_sha256": "2" * 64,
    }
    receipt = {
        "skeletal_mesh": "/Game/Test/Cat.Cat",
        "walking_animation": "/Game/Test/Walking.Walking",
        "blueprint": bp_path,
    }

    result = batch._readback_import(validated_job, receipt)
    assert result["actions"] == ["Idle", "Walking"]
    assert result["skeletal_mesh"] == "/Game/Test/Cat.Cat"

    object_details["mesh_2"] = (
        "SkeletalMesh",
        "CatCopy",
        "/Game/Test/CatCopy.CatCopy",
    )
    assets.append("mesh_2")
    with pytest.raises(RuntimeError, match="exact readback"):
        batch._readback_import(validated_job, receipt)
    assets.pop()

    object_details["run"] = (
        "AnimSequence",
        "Running",
        "/Game/Test/Running.Running",
    )
    assets.append("run")
    with pytest.raises(RuntimeError, match="exact readback"):
        batch._readback_import(validated_job, receipt)


def test_mid_import_failure_writes_exclusive_residue_receipt(tmp_path, monkeypatch):
    batch = _load_module(
        BATCH_IMPORT,
        "_test_import_pixal_failure_receipt",
        monkeypatch,
    )
    tag = "pixal_animal_british_shorthair_v1"
    mesh_dir, bp_dir, bp_path = batch._content_targets(tag)

    class FakeEditorAssetLibrary:
        @staticmethod
        def does_directory_exist(*, directory_path):
            return directory_path in {mesh_dir, bp_dir}

        @staticmethod
        def list_assets(*, directory_path, recursive):
            assert recursive is True
            return [f"{directory_path}/residue"]

        @staticmethod
        def does_asset_exist(*, asset_path):
            return asset_path == bp_path

    batch.unreal.EditorAssetLibrary = FakeEditorAssetLibrary
    result_path = tmp_path / "failed_import_receipt.json"
    job = {
        "job_type": batch.JOB_TYPE,
        "asset_id": "animal_british_shorthair_v1",
        "tag": tag,
    }
    contract = {
        "preparation_descriptor": {
            "path": "/tmp/preparation.json",
            "sha256": "1" * 64,
            "size_bytes": 1,
            "manifest_sha256": "2" * 64,
        },
        "manifest_descriptor": {
            "path": "/tmp/jobs.json",
            "sha256": "3" * 64,
            "size_bytes": 1,
            "batch_sha256": "4" * 64,
        },
        "batch_identity": {
            "schema": batch.BATCH_SCHEMA,
            "job_count": 1,
        },
        "manifest": {"non_destructive_policy": "create-only"},
        "result_path": result_path,
        "validated_jobs": [
            {
                "job": job,
                "source": Path("/tmp/cat.glb"),
                "job_identity_sha256": "5" * 64,
            }
        ],
    }
    monkeypatch.setattr(batch, "_authenticate_import_contract", lambda: contract)
    monkeypatch.setattr(batch, "_assert_content_targets_absent", lambda tags: None)

    def fail_import(*_args):
        raise RuntimeError("synthetic import failure")

    monkeypatch.setattr(batch, "_run_one", fail_import)
    with pytest.raises(RuntimeError, match="synthetic import failure"):
        batch.main()

    receipt = json.loads(result_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "failed_write_residue_requires_manual_audit"
    assert receipt["failure"]["existing_content_was_deleted"] is False
    assert receipt["failure"]["residue"] == [
        {
            "tag": tag,
            "directories": [bp_dir, mesh_dir],
            "assets": [
                f"{bp_dir}/residue",
                f"{mesh_dir}/residue",
            ],
            "blueprint": bp_path,
            "blueprint_exists": True,
            "readback_errors": [],
            "manual_cleanup_required": True,
        }
    ]
