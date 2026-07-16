from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "tokenrig_human_female_canary.py"
)
SPEC = importlib.util.spec_from_file_location("tokenrig_human_female_canary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
female = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(female)


def test_female_contract_pins_the_reviewed_original_pixal_pbr_glb():
    contract = female.PINNED_FEMALE_CONTRACT

    assert contract.asset_id == "rocketbox_female_adult_01"
    assert contract.input_glb.name == "canary_1024_seed42.glb"
    assert contract.input_glb.parent.name == "rocketbox_female_adult_01"
    assert contract.input_glb_sha256 == (
        "894e7f88d96d59510837bd4550a136a53fdd32e421910281351fdb20aedbb746"
    )
    assert contract.input_manifest_sha256 == (
        "bbcffc16a63ee2a2cb0f7bf063a620e40aff4dcf98103da4f19bc7eea82a954b"
    )
    assert contract.output_dir.name == contract.asset_id
    assert contract.output_dir.parent.name == "pixal_tokenrig_route2_v1"
    assert contract.model_revision == female.base.PINNED_CONTRACT.model_revision
    assert contract.code_hashes == female.base.PINNED_CONTRACT.code_hashes
    assert contract.checkpoint_hashes == female.base.PINNED_CONTRACT.checkpoint_hashes


def _male_gate(tmp_path: Path) -> tuple[Path, str]:
    review_dir = tmp_path / "dynamic_review_v1"
    review_dir.mkdir()
    decision = tmp_path / "dynamic_review_v1.agent_visual_qa.json"
    decision.write_text("{}", encoding="utf-8")
    decision.chmod(0o444)
    digest = hashlib.sha256(b"authenticated-male-bundle").hexdigest()
    return review_dir, digest


def test_female_wrapper_delegates_only_after_authenticated_male_gate_and_records_base_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    calls = []
    pointer = tmp_path / "qualified_candidate_v1.json"
    pointer.write_text("{}", encoding="utf-8")
    gate = {"qualified_candidate": {"path": str(pointer)}}

    monkeypatch.setattr(
        female,
        "authenticate_male_gate",
        lambda supplied: calls.append(("gate", Path(supplied))) or gate,
    )
    monkeypatch.setattr(female, "publish_female_gate_record", lambda **kwargs: tmp_path / "gate.json")
    monkeypatch.setattr(
        female,
        "seal_female_success_artifacts",
        lambda supplied: calls.append(("seal", Path(supplied))) or {},
    )
    authorization = tmp_path / "authorization.json"
    monkeypatch.setattr(
        female,
        "publish_female_authorization_manifest",
        lambda **kwargs: authorization,
    )
    monkeypatch.setattr(
        female,
        "validate_female_authorization_manifest",
        lambda path, **kwargs: calls.append(("validate", Path(path))) or {},
    )

    def fake_run_canary(**kwargs):
        calls.append(kwargs)
        return female.PINNED_FEMALE_CONTRACT.output_dir / "tokenrig_manifest.json"

    monkeypatch.setattr(female.base, "run_canary", fake_run_canary)

    result = female.run_female_canary(male_qualified_candidate=pointer)

    assert result == authorization
    assert calls[0] == ("gate", pointer)
    assert calls[-4:] == [
        ("seal", female.PINNED_FEMALE_CONTRACT.output_dir / "tokenrig_manifest.json"),
        ("gate", pointer),
        ("validate", authorization),
        ("gate", pointer),
    ]
    call = [value for value in calls if isinstance(value, dict)][0]
    assert call["contract"] is female.PINNED_FEMALE_CONTRACT
    assert call["input_glb"] == female.PINNED_FEMALE_CONTRACT.input_glb
    assert call["input_manifest"] == female.PINNED_FEMALE_CONTRACT.input_manifest
    assert call["output_dir"] == female.PINNED_FEMALE_CONTRACT.output_dir
    assert call["skintokens_root"] == female.PINNED_FEMALE_CONTRACT.skintokens_root
    assert call["model_revision"] == female.PINNED_FEMALE_CONTRACT.model_revision
    assert call["orchestrator_path"] == MODULE_PATH
    assert call["seed"] == 42
    assert call["use_skeleton_input"] is False
    assert female.base.sha256_file(female.base.RUNNER_PATH) == female.BASE_RUNNER_SHA256


def test_cli_requires_exact_male_gate_and_has_no_skeleton_override_surface():
    with pytest.raises(SystemExit):
        female.parse_args([])
    args = female.parse_args(
        [
            "--male-qualified-candidate",
            "/tmp/male/qualified_candidate_v1.json",
        ]
    )
    assert vars(args) == {
        "male_qualified_candidate": Path("/tmp/male/qualified_candidate_v1.json"),
    }
    with pytest.raises(SystemExit):
        female.parse_args(
            [
                "--male-qualified-candidate",
                "/tmp/male/qualified_candidate_v1.json",
                "--use-skeleton-input",
            ]
        )


def test_production_male_gate_fails_closed_for_noncanonical_pointer(tmp_path: Path):
    pointer = tmp_path / "qualified_candidate_v1.json"
    pointer.write_text("{}", encoding="utf-8")
    pointer.chmod(0o444)
    with pytest.raises(female.FemaleGateError, match="not canonical"):
        female.authenticate_male_gate(pointer)


def test_female_gate_authenticates_the_canonical_male_final_branch_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    pointer = tmp_path / "rocketbox_male_adult_01/qualified_candidate_v1.json"
    pointer.parent.mkdir()
    pointer.write_text("{}", encoding="utf-8")
    pointer.chmod(0o444)
    branch = pointer.parent / "fitted_skeleton_v1/sanitized_weights_v1"
    review_dir = branch / "dynamic_review_v1"
    monkeypatch.setattr(female, "CANONICAL_MALE_QUALIFIED_CANDIDATE", pointer)
    monkeypatch.setattr(
        female.qualified_candidate,
        "validate_qualified_candidate",
        lambda path: {
            "asset_id": "rocketbox_male_adult_01",
            "base_avatar_id": "rocketbox_male_adult_01",
            "status": "agent_qa_passed_pending_user_acceptance",
            "final_branch": {
                "branch_id": "sanitized_weights",
                "path": str(branch),
                "relative_root": "fitted_skeleton_v1/sanitized_weights_v1",
            },
            "dynamic": {"review_dir": str(review_dir)},
            "inventory_sha256": "b" * 64,
        },
    )

    gate = female.authenticate_male_gate(pointer)

    assert gate["qualified_candidate"]["path"] == str(pointer)
    assert gate["qualified_candidate"]["sha256"] == hashlib.sha256(
        pointer.read_bytes()
    ).hexdigest()
    assert gate["final_branch"]["branch_id"] == "sanitized_weights"
    assert gate["review_dir"] == str(review_dir)


def test_female_authorization_manifest_binds_gate_to_tokenrig_producer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    pointer = tmp_path / "male/qualified_candidate_v1.json"
    pointer.parent.mkdir()
    pointer.write_text("{}", encoding="utf-8")
    pointer.chmod(0o444)
    female_output = tmp_path / female.ASSET_ID
    gate = {
        "schema": "route2_male_qualified_gate_snapshot_v2",
        "asset_id": "rocketbox_male_adult_01",
        "status": "agent_qa_passed_pending_user_acceptance",
        "qualified_candidate": {
            "path": str(pointer),
            "sha256": hashlib.sha256(pointer.read_bytes()).hexdigest(),
            "size_bytes": pointer.stat().st_size,
        },
        "final_branch": {
            "branch_id": "sanitized_weights",
            "path": str(pointer.parent / "sanitized_weights_v1"),
            "relative_root": "fitted_skeleton_v1/sanitized_weights_v1",
        },
        "review_dir": str(pointer.parent / "sanitized_weights_v1/dynamic_review_v1"),
        "inventory_sha256": "b" * 64,
    }
    calls = []
    monkeypatch.setattr(female, "_OUTPUT_DIR", female_output)
    monkeypatch.setattr(female, "CANONICAL_MALE_QUALIFIED_CANDIDATE", pointer)
    monkeypatch.setattr(
        female,
        "authenticate_male_gate",
        lambda supplied: calls.append(Path(supplied)) or gate,
    )

    def fake_run_canary(**kwargs):
        female_output.mkdir()
        manifest = female_output / "tokenrig_manifest.json"
        manifest.write_text(
            json.dumps({"asset_id": female.ASSET_ID}), encoding="utf-8"
        )
        (female_output / "tokenrig_transfer.glb").write_bytes(b"glb")
        female_output.with_name(
            f"{female.ASSET_ID}.tokenrig_attempt.json"
        ).write_text("{}", encoding="utf-8")
        return manifest

    monkeypatch.setattr(female.base, "run_canary", fake_run_canary)

    authorization = female.run_female_canary(male_qualified_candidate=pointer)

    payload = json.loads(authorization.read_text())
    assert calls == [pointer, pointer, pointer, pointer]
    assert authorization.name == "tokenrig_female_authorization_v2.json"
    assert authorization.stat().st_mode & 0o777 == 0o444
    assert payload["male_gate"] == gate
    assert payload["female_gate_record"]["sha256"] == hashlib.sha256(
        Path(payload["female_gate_record"]["path"]).read_bytes()
    ).hexdigest()
    assert payload["tokenrig_manifest"]["sha256"] == hashlib.sha256(
        (female_output / "tokenrig_manifest.json").read_bytes()
    ).hexdigest()
    validated = female.validate_female_authorization_manifest(
        authorization,
        expected_tokenrig_manifest=female_output / "tokenrig_manifest.json",
    )
    assert validated["payload"] == payload
    assert set(validated["records"]) == {
        "authorization",
        "female_gate_record",
        "male_qualified_candidate",
        "tokenrig_manifest",
        "female_wrapper",
        "base_runner",
    }

    tokenrig_manifest = female_output / "tokenrig_manifest.json"
    tokenrig_manifest.chmod(0o644)
    with pytest.raises(female.FemaleGateError, match="mode 0444"):
        female.validate_female_authorization_manifest(
            authorization,
            expected_tokenrig_manifest=tokenrig_manifest,
        )
    tokenrig_manifest.chmod(0o444)

    gate_record = Path(payload["female_gate_record"]["path"])
    gate_payload = json.loads(gate_record.read_text())
    gate_payload["female_wrapper"]["sha256"] = "f" * 64
    gate_record.chmod(0o644)
    gate_record.write_text(
        json.dumps(gate_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    gate_record.chmod(0o444)
    payload["female_gate_record"]["sha256"] = hashlib.sha256(
        gate_record.read_bytes()
    ).hexdigest()
    payload["female_gate_record"]["size_bytes"] = gate_record.stat().st_size
    authorization.chmod(0o644)
    authorization.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    authorization.chmod(0o444)
    with pytest.raises(female.FemaleGateError, match="producer binding changed"):
        female.validate_female_authorization_manifest(
            authorization,
            expected_tokenrig_manifest=tokenrig_manifest,
        )


def test_female_success_seals_every_published_producer_file_but_keeps_branch_creatable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    output = tmp_path / female.ASSET_ID
    patch_dir = output / "runtime_patch/markers"
    patch_dir.mkdir(parents=True)
    files = [
        output / "tokenrig_manifest.json",
        output / "tokenrig_transfer.glb",
        output / "inference.log",
        output / "runtime_patch/sitecustomize.py",
        patch_dir / "demo.json",
    ]
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"evidence")
        path.chmod(0o600)
    attempt = output.with_name(f"{female.ASSET_ID}.tokenrig_attempt.json")
    attempt.write_bytes(b"attempt")
    attempt.chmod(0o600)
    monkeypatch.setattr(female, "_OUTPUT_DIR", output)

    records = female.seal_female_success_artifacts(
        output / "tokenrig_manifest.json"
    )

    assert set(records) == {str(path) for path in [*files, attempt]}
    assert all(path.stat().st_mode & 0o777 == 0o444 for path in [*files, attempt])
    assert output.stat().st_mode & 0o200
