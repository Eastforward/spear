from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools import blender_tokenrig_lower_limb_static_audit_v2 as adapter


def _write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    path.chmod(0o444)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, monkeypatch):
    source = _write(tmp_path / "source.glb", b"glTFsource")
    tokenrig = _write(tmp_path / "tokenrig_transfer.glb", b"glTFtokenrig")
    manifest_path = _write(tmp_path / "tokenrig_manifest.json", b"manifest")
    manifest = {
        "schema": adapter.sanitizer.SCHEMA,
        "asset_id": "person_female_01",
        "input_mode": "pre_static_repair",
        "input": {
            "source_glb": {
                "path": str(source.resolve()),
                "sha256": _sha(source),
                "size_bytes": source.stat().st_size,
            },
            "glb": {"sha256": "a" * 64},
            "manifest": {"sha256": "b" * 64},
            "prior_failures": [{"sha256": "c" * 64}],
        },
        "output": {
            "path": str(tokenrig.resolve()),
            "sha256": _sha(tokenrig),
            "size_bytes": tokenrig.stat().st_size,
        },
        "code": {"sanitizer_v2": {"sha256": "d" * 64}},
        "artifacts": {"weight_changes": {"sha256": "e" * 64}},
    }
    monkeypatch.setattr(
        adapter.sanitizer,
        "validate_published_manifest",
        lambda path: manifest,
    )
    return source, tokenrig, manifest_path, manifest


def test_adapter_authenticates_v2_without_extending_historical_static_authenticator(
    tmp_path, monkeypatch
):
    source, tokenrig, manifest_path, manifest = _fixture(tmp_path, monkeypatch)
    historical = adapter.static_audit.authenticate_task3_inputs

    result = adapter.authenticate_v2_static_inputs(
        asset_id="person_female_01",
        source_glb=source,
        tokenrig_glb=tokenrig,
        tokenrig_manifest=manifest_path,
    )

    assert result["manifest_schema"] == adapter.sanitizer.SCHEMA
    assert result["input_mode"] == "pre_static_repair"
    assert result["sanitized_candidate"] is True
    assert result["lower_limb_sanitized_candidate"] is True
    assert result["fresh_full_static_audit_required"] is True
    assert result["tokenrig_glb_sha256"] == _sha(tokenrig)
    assert result["prior_static_failure_sha256"] == ["c" * 64]
    assert adapter.static_audit.authenticate_task3_inputs is historical


def test_adapter_rejects_spliced_source_output_and_asset(tmp_path, monkeypatch):
    source, tokenrig, manifest_path, manifest = _fixture(tmp_path, monkeypatch)

    wrong_source = _write(tmp_path / "wrong.glb", b"glTFwrong")
    with pytest.raises(adapter.LowerLimbStaticAdapterError, match="source"):
        adapter.authenticate_v2_static_inputs(
            asset_id="person_female_01",
            source_glb=wrong_source,
            tokenrig_glb=tokenrig,
            tokenrig_manifest=manifest_path,
        )
    with pytest.raises(adapter.LowerLimbStaticAdapterError, match="asset_id"):
        adapter.authenticate_v2_static_inputs(
            asset_id="person_male_01",
            source_glb=source,
            tokenrig_glb=tokenrig,
            tokenrig_manifest=manifest_path,
        )
    wrong_output = _write(tmp_path / "wrong_output.glb", b"glTFwrong-output")
    with pytest.raises(adapter.LowerLimbStaticAdapterError, match="output"):
        adapter.authenticate_v2_static_inputs(
            asset_id="person_female_01",
            source_glb=source,
            tokenrig_glb=wrong_output,
            tokenrig_manifest=manifest_path,
        )


def test_wrapper_reauthenticates_inside_core_and_restores_global_function(
    tmp_path, monkeypatch
):
    source, tokenrig, manifest_path, manifest = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "static_audit_v1"
    historical = adapter.static_audit.authenticate_task3_inputs
    historical_bilateral = adapter.static_audit.validate_bilateral_contamination
    calls = []

    def fake_core(**kwargs):
        calls.append(kwargs)
        authenticated = adapter.static_audit.authenticate_task3_inputs(
            asset_id=kwargs["asset_id"],
            source_glb=kwargs["source_glb"],
            tokenrig_glb=kwargs["tokenrig_glb"],
            tokenrig_manifest=kwargs["tokenrig_manifest"],
        )
        assert authenticated["lower_limb_sanitized_candidate"] is True
        return kwargs["output_dir"]

    monkeypatch.setattr(adapter.static_audit, "run_static_audit", fake_core)

    result = adapter.run_v2_static_audit(
        asset_id="person_female_01",
        source_glb=source,
        tokenrig_glb=tokenrig,
        tokenrig_manifest=manifest_path,
        output_dir=output,
    )

    assert result == output
    assert len(calls) == 1
    assert adapter.static_audit.authenticate_task3_inputs is historical
    assert adapter.static_audit.validate_bilateral_contamination is historical_bilateral


def test_wrapper_restores_authenticator_when_core_raises(tmp_path, monkeypatch):
    source, tokenrig, manifest_path, manifest = _fixture(tmp_path, monkeypatch)
    historical = adapter.static_audit.authenticate_task3_inputs
    historical_bilateral = adapter.static_audit.validate_bilateral_contamination

    def fail_core(**kwargs):
        raise RuntimeError("injected core failure")

    monkeypatch.setattr(adapter.static_audit, "run_static_audit", fail_core)
    with pytest.raises(RuntimeError, match="injected"):
        adapter.run_v2_static_audit(
            asset_id="person_female_01",
            source_glb=source,
            tokenrig_glb=tokenrig,
            tokenrig_manifest=manifest_path,
            output_dir=tmp_path / "static_audit_v1",
        )
    assert adapter.static_audit.authenticate_task3_inputs is historical
    assert adapter.static_audit.validate_bilateral_contamination is historical_bilateral


def test_static_core_canonical_world_positions_are_not_flipped_a_second_time():
    chains = {
        "axial": ["root", "spine_a", "spine_b", "head"],
        "left_arm": ["l_clavicle", "l_upper_arm", "l_forearm", "l_hand"],
        "right_arm": ["r_clavicle", "r_upper_arm", "r_forearm", "r_hand"],
        "left_leg": ["l_thigh", "l_calf", "l_foot", "l_toe"],
        "right_leg": ["r_thigh", "r_calf", "r_foot", "r_toe"],
    }
    positions = ((0.12, 0.0, 0.0), (-0.12, 0.0, 0.0))
    weights = ({"l_toe": 1.0}, {"r_toe": 1.0})

    result = adapter.validate_static_canonical_lower_limb(
        positions, weights, chains
    )

    assert result["contaminated_vertex_count"] == 0
    assert result["considered_distal_vertex_count"] == 2


def test_cli_and_source_delegate_full_owner_audit_without_modifying_v1():
    args = adapter.parse_args(
        [
            "--asset-id",
            "person_female_01",
            "--source-glb",
            "/source.glb",
            "--tokenrig-glb",
            "/v2/tokenrig_transfer.glb",
            "--tokenrig-manifest",
            "/v2/tokenrig_manifest.json",
            "--output-dir",
            "/v2/static_audit_v1",
        ]
    )
    assert args.output_dir == Path("/v2/static_audit_v1")
    source = Path(adapter.__file__).read_text(encoding="utf-8")
    assert "static_audit.run_static_audit(" in source
    assert "sanitizer.validate_published_manifest(" in source
    assert "fresh_full_static_audit_required" in source
