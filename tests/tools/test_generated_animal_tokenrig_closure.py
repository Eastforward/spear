from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import tempfile

import pytest

from tools import generated_animal_tokenrig_closure as closure
from tools import publish_generated_animal_geometry_closure as geometry_closures
from tools import register_controlled_animal_source_assets as source_registry


def test_closure_validation_remains_compatible_with_python_39() -> None:
    source = Path(closure.__file__).read_text(encoding="utf-8")

    assert "zip(extra_records, extra_upstream, strict=True)" not in source


def descriptor(path: Path) -> dict:
    return {
        "path": str(path),
        "sha256": closure.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_bytes(closure.canonical_bytes(payload) + b"\n")


def valid_readback(path: Path, source: Path, rig: Path) -> dict:
    payload = {
        "schema": closure.READBACK_SCHEMA,
        "tokenrig_input": descriptor(source),
        "tokenrig_output": descriptor(rig),
        "automatic_checks": {
            "overall": "passed",
            "world_geometry_unchanged": True,
            "logical_vertices_bijective": True,
            "triangles_bijective": True,
            "uvs_unchanged": True,
            "pbr_unchanged": True,
            "exactly_one_output_skin": True,
            "exactly_one_output_armature": True,
            "no_animation": True,
        },
        "formal_dataset_registration_authorized": False,
    }
    payload["manifest_sha256"] = closure.hash_without(payload, "manifest_sha256")
    write_json(path, payload)
    return payload


def test_same_workspace_same_bytes_cannot_replace_expected_glb(tmp_path):
    expected = tmp_path / "expected.glb"
    replacement = tmp_path / "replacement.glb"
    expected.write_bytes(b"same GLB bytes")
    replacement.write_bytes(expected.read_bytes())
    record = descriptor(replacement)

    with pytest.raises(closure.ClosureError, match="different file"):
        closure.require_descriptor_matches(record, expected, "expected target rig GLB")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("sha256", "0" * 64, "SHA-256 mismatch"),
        ("size_bytes", 1, "size mismatch"),
    ],
)
def test_input_output_descriptor_hash_and_size_mutations_fail(
    tmp_path, field, value, match
):
    artifact = tmp_path / "artifact.glb"
    artifact.write_bytes(b"authenticated GLB")
    record = descriptor(artifact)
    record[field] = value

    with pytest.raises(closure.ClosureError, match=match):
        closure.descriptor_path(record, "TokenRig input/output")


def execution_pair():
    argv = [
        "demo.py",
        "--input",
        "/asset/source.glb",
        "--output",
        "/asset/rig.glb",
    ]
    model = {
        "invocation_path": "/model/checkpoint",
        "symlink_chain": [],
        "snapshot_revision": "1" * 40,
        "resolved_payload": {
            "path": "/model/blob",
            "sha256": "2" * 64,
            "size_bytes": 100,
        },
    }
    reconstructed = {
        "exact_argv": argv,
        "exact_argv_sha256": closure.sha256_bytes(closure.canonical_bytes(argv)),
        "environment": {"TOKENRIG_CANARY_SEED": "42"},
        "patch_sha256": "3" * 64,
        "model_checkpoint": model,
    }
    execution = {
        "seed": 42,
        "exact_argv": deepcopy(argv),
        "exact_argv_sha256": reconstructed["exact_argv_sha256"],
        "environment": deepcopy(reconstructed["environment"]),
        "runtime_patch_sha256": reconstructed["patch_sha256"],
        "model_checkpoint": deepcopy(model),
    }
    return execution, reconstructed


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda value: value.__setitem__("seed", 43),
            "seed is not exactly 42",
        ),
        (
            lambda value: value["exact_argv"].__setitem__(-1, "/asset/other.glb"),
            "argv",
        ),
        (
            lambda value: value["model_checkpoint"]["resolved_payload"].__setitem__(
                "sha256", "4" * 64
            ),
            "model identity",
        ),
        (
            lambda value: value.__setitem__("runtime_patch_sha256", "5" * 64),
            "runtime patch",
        ),
    ],
)
def test_seed_argv_model_and_patch_mutations_fail_closed(mutate, match):
    execution, reconstructed = execution_pair()
    mutate(execution)

    with pytest.raises(closure.ClosureError, match=match):
        closure.validate_execution_identity_fields(execution, reconstructed)


def test_upstream_raw_pixal_link_cannot_be_redirected_to_same_bytes(tmp_path):
    raw = tmp_path / "pixal_raw.glb"
    raw.write_bytes(b"raw pixal")
    raw_clone = tmp_path / "pixal_raw_clone.glb"
    raw_clone.write_bytes(raw.read_bytes())
    tokenrig_input = tmp_path / "watertight.glb"
    tokenrig_input.write_bytes(b"watertight")
    raw_manifest = tmp_path / "raw.manifest.json"
    write_json(
        raw_manifest,
        {
            "backend": "pixal3d",
            "output": {
                **descriptor(raw),
                "bytes": raw.stat().st_size,
            },
        },
    )
    upstream_manifest = tmp_path / "watertight.manifest.json"
    upstream = {
        "schema": "avengine_watertight_textured_runtime_proxy_v1",
        "input": descriptor(raw),
        "output": descriptor(tokenrig_input),
        "authority_contract": {
            "approved_skeleton_or_animation_touched": False,
        },
    }
    write_json(upstream_manifest, upstream)
    observed_raw, kind, extra = closure.validate_upstream_lineage(
        raw_manifest, upstream_manifest, tokenrig_input
    )
    assert os.path.samefile(observed_raw, raw)
    assert kind == "watertight_runtime_proxy"
    assert extra == []

    upstream["input"] = descriptor(raw_clone)
    write_json(upstream_manifest, upstream)
    with pytest.raises(closure.ClosureError, match="different file"):
        closure.validate_upstream_lineage(
            raw_manifest, upstream_manifest, tokenrig_input
        )


def _bounded_upstream_fixture(tmp_path):
    raw = tmp_path / "pixal_raw.glb"
    raw.write_bytes(b"raw Pixel3D")
    tokenrig_input = tmp_path / "bounded.glb"
    tokenrig_input.write_bytes(b"bounded Pixel3D")
    raw_manifest = tmp_path / "raw.manifest.json"
    write_json(
        raw_manifest,
        {
            "backend": "pixal3d",
            "output": {
                **descriptor(raw),
                "bytes": raw.stat().st_size,
            },
        },
    )
    repair_path = tmp_path / "repair.json"
    write_json(repair_path, {"fixture": True})
    upstream_path = tmp_path / "geometry_closure.json"
    write_json(
        upstream_path,
        {
            "schema": "avengine_generated_animal_geometry_closure_v1",
            "status": "pass_geometry_only",
            "candidate": {"source_pixal_glb": descriptor(raw)},
            "output": {
                "glb": descriptor(tokenrig_input),
                "repair_manifest": descriptor(repair_path),
            },
            "downstream": {"tokenrig_entry_authorized": True},
        },
    )
    return raw, raw_manifest, tokenrig_input, repair_path, upstream_path


def _direct_copy_v2_fixture(tmp_path, monkeypatch):
    historical_raw = tmp_path / "original" / "pixal_raw.glb"
    historical_raw.parent.mkdir()
    historical_raw.write_bytes(b"direct-adopted Pixel3D")
    adopted_raw = tmp_path / "adopted" / "pixal_raw.glb"
    adopted_raw.parent.mkdir()
    adopted_raw.write_bytes(historical_raw.read_bytes())
    tokenrig_input = tmp_path / "bounded.glb"
    tokenrig_input.write_bytes(b"bounded Pixel3D")
    raw_manifest = tmp_path / "adopted" / "pixal_raw.manifest.json"
    write_json(
        raw_manifest,
        {
            "backend": "pixal3d",
            "output": {
                **descriptor(historical_raw),
                "bytes": historical_raw.stat().st_size,
            },
        },
    )
    upstream = tmp_path / "geometry_closure_v2.json"
    write_json(
        upstream,
        {"schema": "avengine_generated_animal_geometry_closure_v2"},
    )
    repair = tmp_path / "repair.json"
    repair.write_bytes(b"repair")
    audit = tmp_path / "audit.json"
    audit.write_bytes(b"audit")
    decision_batch = tmp_path / "raw_decision_batch.json"
    decision_batch.write_bytes(b"decision batch")
    observed: dict[str, object] = {}

    def load_geometry_closure(path, **expected):
        observed["path"] = path
        observed["expected"] = expected
        return {
            "paths": {
                "raw_pixal_glb": adopted_raw,
                "repair_manifest": repair,
                "geometry_audit": audit,
                "raw_static_decision_batch": decision_batch,
            }
        }

    monkeypatch.setattr(
        geometry_closures,
        "load_geometry_closure_v2",
        load_geometry_closure,
    )
    return {
        "historical_raw": historical_raw,
        "adopted_raw": adopted_raw,
        "raw_manifest": raw_manifest,
        "tokenrig_input": tokenrig_input,
        "upstream": upstream,
        "repair": repair,
        "audit": audit,
        "decision_batch": decision_batch,
        "observed": observed,
    }


def test_geometry_closure_v2_selects_hash_equivalent_direct_adopted_raw(
    tmp_path,
    monkeypatch,
):
    case = _direct_copy_v2_fixture(tmp_path, monkeypatch)
    case["historical_raw"].unlink()

    observed_raw, kind, extra = closure.validate_upstream_lineage(
        case["raw_manifest"],
        case["upstream"],
        case["tokenrig_input"],
    )

    assert os.path.samefile(observed_raw, case["adopted_raw"])
    assert kind == "bounded_geometry_closure"
    assert extra == [case["repair"], case["audit"], case["decision_batch"]]
    assert case["observed"]["path"] == case["upstream"]
    expected = case["observed"]["expected"]
    assert "expected_raw_pixal_glb" not in expected
    assert expected["expected_pixal_manifest"] == case["raw_manifest"]
    assert expected["expected_repaired_glb"] == case["tokenrig_input"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("sha256", "0" * 64, "SHA-256 mismatch"),
        ("bytes", 1, "size mismatch"),
    ],
)
def test_geometry_closure_v2_direct_copy_still_requires_raw_content_binding(
    tmp_path,
    monkeypatch,
    field,
    value,
    match,
):
    case = _direct_copy_v2_fixture(tmp_path, monkeypatch)
    manifest = closure.load_json(case["raw_manifest"], "raw Pixal manifest")
    manifest["output"][field] = value
    write_json(case["raw_manifest"], manifest)

    with pytest.raises(closure.ClosureError, match=match):
        closure.validate_upstream_lineage(
            case["raw_manifest"],
            case["upstream"],
            case["tokenrig_input"],
        )


def test_tokenrig_lineage_reader_keeps_strict_mirror_v2_path(
    tmp_path,
    monkeypatch,
):
    raw, raw_manifest, tokenrig_input, repair_path, upstream = (
        _bounded_upstream_fixture(tmp_path)
    )
    mirror = {
        "implementation_contract": (
            closure.derived_review_contract.REPAIR_IMPLEMENTATION_CONTRACT
        ),
        "lineage": {
            "pixal_source": descriptor(raw),
            "pixal_manifest": descriptor(raw_manifest),
        },
        "output": descriptor(tokenrig_input),
    }
    monkeypatch.setattr(
        closure.derived_review_contract,
        "validate_bounded_repair_manifest",
        lambda _payload: deepcopy(mirror),
    )

    observed_raw, kind, extra = closure.validate_upstream_lineage(
        raw_manifest,
        upstream,
        tokenrig_input,
    )

    assert observed_raw == raw
    assert kind == "bounded_geometry_closure"
    assert extra == [repair_path]


def test_tokenrig_build_mode_keeps_legacy_watertight_without_batch(tmp_path):
    raw = tmp_path / "pixal_raw.glb"
    raw.write_bytes(b"raw pixal")
    tokenrig_input = tmp_path / "watertight.glb"
    tokenrig_input.write_bytes(b"watertight")
    raw_manifest = tmp_path / "raw.manifest.json"
    write_json(
        raw_manifest,
        {
            "backend": "pixal3d",
            "output": {
                **descriptor(raw),
                "bytes": raw.stat().st_size,
            },
        },
    )
    upstream = tmp_path / "watertight.json"
    write_json(
        upstream,
        {
            "schema": "avengine_watertight_textured_runtime_proxy_v1",
            "input": descriptor(raw),
            "output": descriptor(tokenrig_input),
            "authority_contract": {
                "approved_skeleton_or_animation_touched": False
            },
        },
    )
    _raw, kind, extra = closure.validate_upstream_lineage(
        raw_manifest,
        upstream,
        tokenrig_input,
        require_oriented_batch_external_authority=True,
    )

    assert kind == "watertight_runtime_proxy"
    assert extra == []


def test_tokenrig_lineage_reader_rejects_oriented_batch_on_mirror_v2(
    tmp_path,
    monkeypatch,
):
    raw, raw_manifest, tokenrig_input, _repair_path, upstream = (
        _bounded_upstream_fixture(tmp_path)
    )
    batch = tmp_path / "raw_decision_batch.json"
    batch.write_bytes(b"oriented authority")
    mirror = {
        "implementation_contract": (
            closure.derived_review_contract.REPAIR_IMPLEMENTATION_CONTRACT
        ),
        "lineage": {
            "pixal_source": descriptor(raw),
            "pixal_manifest": descriptor(raw_manifest),
        },
        "output": descriptor(tokenrig_input),
    }
    monkeypatch.setattr(
        closure.derived_review_contract,
        "validate_bounded_repair_manifest",
        lambda _payload: deepcopy(mirror),
    )

    with pytest.raises(
        closure.ClosureError,
        match="mirror-v2 repair cannot consume oriented",
    ):
        closure.validate_upstream_lineage(
            raw_manifest,
            upstream,
            tokenrig_input,
            raw_static_decision_batch_path=batch,
            expected_raw_static_decision_batch_sha256=(
                closure.sha256_file(batch)
            ),
            require_oriented_batch_external_authority=True,
        )


def test_tokenrig_lineage_reader_accepts_oriented_canonical_batch(
    tmp_path,
    monkeypatch,
):
    raw, raw_manifest, tokenrig_input, repair_path, upstream = (
        _bounded_upstream_fixture(tmp_path)
    )
    decision = tmp_path / "raw_decision.json"
    write_json(decision, {"fixture": True})
    batch = tmp_path / "raw_decision_batch.json"
    write_json(batch, {"fixture": True})
    instance_id = "dog_fixture_oriented_sheet_v1"
    oriented = {
        "implementation_contract": (
            closure.derived_review_contract
            .ORIENTED_REPAIR_IMPLEMENTATION_CONTRACT
        ),
        "lineage": {
            "instance_id": instance_id,
            "pixal_source": descriptor(raw),
            "pixal_manifest": descriptor(raw_manifest),
            "static_decision_batch": descriptor(batch),
            "static_decision": descriptor(decision),
        },
        "output": descriptor(tokenrig_input),
    }
    approved = {
        "decision": "approved_for_lod_and_binding",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "next_gate": "lod_then_species_rig_binding",
        "checks": {
            name: True
            for name in closure.derived_review_contract.RAW_STATIC_CHECK_FIELDS
        },
    }
    monkeypatch.setattr(
        closure.derived_review_contract,
        "validate_bounded_repair_manifest",
        lambda _payload: deepcopy(oriented),
    )
    monkeypatch.setattr(
        source_registry,
        "load_decision_batch",
        lambda _path: (
            batch,
            {},
            {
                instance_id: {
                    "path": decision,
                    "payload": deepcopy(approved),
                }
            },
        ),
    )

    _raw, kind, extra = closure.validate_upstream_lineage(
        raw_manifest,
        upstream,
        tokenrig_input,
        raw_static_decision_batch_path=batch,
        expected_raw_static_decision_batch_sha256=closure.sha256_file(batch),
        require_oriented_batch_external_authority=True,
    )

    assert kind == "bounded_geometry_closure"
    assert extra == [repair_path, batch]


def test_tokenrig_lineage_reader_rejects_oriented_batch_rebinding(
    tmp_path,
    monkeypatch,
):
    raw, raw_manifest, tokenrig_input, _repair_path, upstream = (
        _bounded_upstream_fixture(tmp_path)
    )
    decision = tmp_path / "raw_decision.json"
    decision.write_bytes(b"decision")
    rebound = tmp_path / "rebound_decision.json"
    rebound.write_bytes(decision.read_bytes())
    batch = tmp_path / "raw_decision_batch.json"
    batch.write_bytes(b"batch")
    instance_id = "dog_fixture_oriented_sheet_v1"
    oriented = {
        "implementation_contract": (
            closure.derived_review_contract
            .ORIENTED_REPAIR_IMPLEMENTATION_CONTRACT
        ),
        "lineage": {
            "instance_id": instance_id,
            "pixal_source": descriptor(raw),
            "pixal_manifest": descriptor(raw_manifest),
            "static_decision_batch": descriptor(batch),
            "static_decision": descriptor(decision),
        },
        "output": descriptor(tokenrig_input),
    }
    monkeypatch.setattr(
        closure.derived_review_contract,
        "validate_bounded_repair_manifest",
        lambda _payload: deepcopy(oriented),
    )
    monkeypatch.setattr(
        source_registry,
        "load_decision_batch",
        lambda _path: (
            batch,
            {},
            {
                instance_id: {
                    "path": rebound,
                    "payload": {
                        "decision": "approved_for_lod_and_binding",
                    },
                }
            },
        ),
    )

    with pytest.raises(
        closure.ClosureError,
        match="canonical approved raw static decision",
    ):
        closure.validate_upstream_lineage(
            raw_manifest,
            upstream,
            tokenrig_input,
            raw_static_decision_batch_path=batch,
            expected_raw_static_decision_batch_sha256=(
                closure.sha256_file(batch)
            ),
            require_oriented_batch_external_authority=True,
        )


def test_tokenrig_lineage_reader_rejects_raw_fallback_with_derived_authority(
    tmp_path,
):
    raw = tmp_path / "pixal_raw.glb"
    raw.write_bytes(b"raw pixal")
    tokenrig_input = tmp_path / "watertight.glb"
    tokenrig_input.write_bytes(b"watertight")
    raw_manifest = tmp_path / "raw.manifest.json"
    write_json(
        raw_manifest,
        {
            "backend": "pixal3d",
            "output": {
                **descriptor(raw),
                "bytes": raw.stat().st_size,
            },
        },
    )
    upstream = tmp_path / "watertight.json"
    write_json(
        upstream,
        {
            "schema": "avengine_watertight_textured_runtime_proxy_v1",
            "input": descriptor(raw),
            "output": descriptor(tokenrig_input),
            "authority_contract": {
                "approved_skeleton_or_animation_touched": False
            },
        },
    )
    batch = tmp_path / "decision_batch.json"
    batch.write_bytes(b"batch")

    with pytest.raises(closure.ClosureError, match="raw/watertight fallback"):
        closure.validate_upstream_lineage(
            raw_manifest,
            upstream,
            tokenrig_input,
            raw_static_decision_batch_path=batch,
            expected_raw_static_decision_batch_sha256=(
                closure.sha256_file(batch)
            ),
            require_oriented_batch_external_authority=True,
        )


def test_geometry_readback_changed_gate_fails_even_after_self_rehash(tmp_path):
    source = tmp_path / "source.glb"
    rig = tmp_path / "rig.glb"
    source.write_bytes(b"source")
    rig.write_bytes(b"rig")
    readback = tmp_path / "readback.json"
    payload = valid_readback(readback, source, rig)
    closure.validate_readback(readback, source, rig)

    payload["automatic_checks"]["world_geometry_unchanged"] = False
    payload["manifest_sha256"] = closure.hash_without(payload, "manifest_sha256")
    write_json(readback, payload)
    with pytest.raises(closure.ClosureError, match="did not pass every gate"):
        closure.validate_readback(readback, source, rig)


def test_recomputed_embedded_hash_cannot_replace_external_manifest_authority(
    tmp_path,
):
    root = tmp_path / "closure"
    root.mkdir()
    manifest = root / "manifest.json"
    original = {
        "schema": closure.SCHEMA,
        "asset_id": "original",
        "status": "passed_source_to_tokenrig_closure",
        "formal_dataset_registration_authorized": False,
    }
    original["manifest_sha256"] = closure.hash_without(original, "manifest_sha256")
    write_json(manifest, original)
    external_hash = closure.sha256_file(manifest)

    mutated = deepcopy(original)
    mutated["asset_id"] = "mutated"
    mutated["manifest_sha256"] = closure.hash_without(mutated, "manifest_sha256")
    write_json(manifest, mutated)
    with pytest.raises(closure.ClosureError, match="external authority"):
        closure.validate_closure(root, expected_manifest_sha256=external_hash)


def test_unknown_evidence_mode_fails_before_any_downstream_use(tmp_path):
    root = tmp_path / "closure"
    root.mkdir()
    payload = {
        "schema": closure.SCHEMA,
        "asset_id": "asset",
        "status": "passed_source_to_tokenrig_closure",
        "formal_dataset_registration_authorized": False,
        "evidence_mode": {
            "mode": "invented_mode",
            "no_missing_evidence_was_fabricated": True,
        },
    }
    payload["manifest_sha256"] = closure.hash_without(payload, "manifest_sha256")
    manifest = root / "manifest.json"
    write_json(manifest, payload)

    with pytest.raises(closure.ClosureError, match="unsupported.*evidence mode"):
        closure.validate_closure(
            root,
            expected_manifest_sha256=closure.sha256_file(manifest),
        )


def test_leaf_and_arbitrary_parent_symlinks_are_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    artifact = real / "asset.glb"
    artifact.write_bytes(b"asset")
    leaf = tmp_path / "leaf.glb"
    leaf.symlink_to(artifact)
    parent = tmp_path / "parent"
    parent.symlink_to(real, target_is_directory=True)

    with pytest.raises(closure.ClosureError, match="unsafe symlink"):
        closure.require_regular_file(leaf, "leaf")
    with pytest.raises(closure.ClosureError, match="unsafe symlink"):
        closure.require_regular_file(parent / "asset.glb", "parent")


def test_known_logical_tmp_path_resolves_to_physical_identity():
    if not closure.LOGICAL_TMP_ROOT.is_symlink():
        pytest.skip("workspace compatibility tmp link is not configured")
    with tempfile.TemporaryDirectory(
        prefix="tokenrig-closure-path-test-",
        dir=closure.PHYSICAL_TMP_ROOT,
    ) as physical_directory:
        physical = Path(physical_directory) / "asset.glb"
        physical.write_bytes(b"identity")
        relative = physical.relative_to(closure.PHYSICAL_TMP_ROOT)
        logical = closure.LOGICAL_TMP_ROOT / relative

        logical_result = closure.require_regular_file(logical, "logical asset")
        physical_result = closure.require_regular_file(physical, "physical asset")
        assert os.path.samefile(logical_result, physical_result)


def test_nonfinite_json_is_rejected():
    with pytest.raises(closure.ClosureError, match="non-finite"):
        closure.recursive_finite({"value": float("nan")})
