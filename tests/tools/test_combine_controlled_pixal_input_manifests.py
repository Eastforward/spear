from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from tools import combine_controlled_pixal_input_manifests as combiner
from tools import controlled_animal_one_shot_policy as one_shot
from tools import controlled_source_asset_schema as contracts
from tools import prepare_controlled_animal_pixal_inputs as preparation
from tools import run_controlled_animal_pixal_jobs as runner


def _write_parent(
    root: Path,
    *,
    instance_id: str,
    request_digit: str,
    route: str = "flux2_pixal3d_static_v1",
    pixal_output_root: Path | None = None,
) -> Path:
    root.mkdir()
    static = route == "flux2_pixal3d_static_v1"
    asset_class = "static_object" if static else "animal"
    prefix = "static_" if static else "animal_"
    request_sha256 = request_digit * 64
    source = root / "candidate.png"
    rgba = root / "segmentation" / instance_id / "input_rgba_isnet.png"
    rgba.parent.mkdir(parents=True)
    source.write_bytes(f"source-{instance_id}".encode("utf-8"))
    Image.new("RGBA", (1024, 1024), (120, 140, 160, 255)).save(rgba)
    pixal_output_root = pixal_output_root or root.with_name(f"{root.name}_pixal")
    output = pixal_output_root / instance_id / "pixal_raw_1024.glb"
    controlled_request = {
        "execution_job_id": f"{prefix}{request_sha256[:16]}",
        "instance_id": instance_id,
        "request_sha256": request_sha256,
        "generation_seed": 42,
        "profile_schema_id": f"{instance_id}_profile_v1",
        "profile_sha256": request_digit.swapcase().lower() * 64
        if request_digit in "abcdef"
        else str((int(request_digit) + 4) % 10) * 64,
        "asset_class": asset_class,
        "route": route,
        "sampled_attributes": {"body_color": "white"},
        "target_physical_profile": {
            "measurement": "height_cm" if static else "shoulder_height_cm",
            "target_value_cm": 30.0,
            "tolerance_cm": 2.0,
        },
        "rig_profile": (
            None
            if static
            else {
                "profile_id": "quadruped_test_v1",
                "actions": ["Walking", "Idle"],
            }
        ),
    }
    job = {
        "legacy_tag": instance_id,
        "candidate_tag": f"{instance_id}_pixal_v1",
        "asset_class": asset_class,
        "route": route,
        "seed": 42,
        "attempt_ordinal": 0,
        "one_shot_execution": one_shot.stage_record("pixal3d"),
        "reference": {
            "source": {
                "path": str(source.resolve()),
                "sha256": combiner._sha256_file(source),
                "size_bytes": source.stat().st_size,
            },
            "pixal_input": {
                "path": str(rgba.resolve()),
                "sha256": combiner._sha256_file(rgba),
                "size_bytes": rgba.stat().st_size,
            },
            "normalization": "pinned_isnet_general_use_alpha_v1",
        },
        "output": str(output.resolve()),
        "manifest": str(output.with_suffix(".manifest.json").resolve()),
        "controlled_request": controlled_request,
        "model_revisions": {
            "pixal3d": preparation.PIXAL_MODEL_REVISION,
            "dino": preparation.DINO_REVISION,
        },
        "parameters": {
            "resolution": 1024,
            "manual_fov": 0.2,
            "low_vram": False,
        },
    }
    if not static:
        job["rig_mode"] = "animated_transfer"
    payload = {
        "schema": preparation.PIXAL_INPUT_SCHEMA,
        "status": "ready_for_pixal3d",
        "state_classification": "research_candidate",
        "formal_dataset_registration_authorized": False,
        "asset_class": asset_class,
        "route": route,
        "one_shot_execution": one_shot.stage_record("pixal3d"),
        "upstream_flux_one_shot_evidence": {
            "mode": "legacy_sealed_manifest_attestation",
            "policy": one_shot.policy_record(),
            "flux_batch_sha256": request_digit * 64,
            "recorded_flux_invocations_per_candidate": 1,
            "recorded_candidates_per_request": 1,
            "cross_batch_seed_lottery_exclusion_proven": False,
            "profile_qualification_authorized": False,
        },
        "pixal_output_root": str(pixal_output_root.resolve()),
        "job_count": 1,
        "jobs": [job],
        "automatic_checks": {
            "static_jobs_have_no_rig_or_animation_binding": True,
            "overall": "passed",
        },
    }
    payload["manifest_sha256"] = combiner._hash_without(
        payload, "manifest_sha256"
    )
    manifest = root / "pixal_inputs_manifest.json"
    contracts.write_json_no_replace(manifest, payload)
    return manifest


@pytest.fixture
def no_seal(monkeypatch):
    monkeypatch.setattr(
        combiner.material_execution.native, "_seal_readonly_tree", lambda _path: None
    )


def _combine_two(tmp_path: Path) -> tuple[Path, Path, Path]:
    parent_a = _write_parent(
        tmp_path / "parent_a",
        instance_id="microwave_test_a",
        request_digit="1",
    )
    parent_b = _write_parent(
        tmp_path / "parent_b",
        instance_id="alarm_clock_test_b",
        request_digit="2",
    )
    combined = combiner.combine_pixal_inputs(
        [parent_b, parent_a],
        tmp_path / "combined_inputs",
        tmp_path / "combined_pixal_outputs",
    )
    return parent_a, parent_b, combined


def test_combines_homogeneous_parents_with_private_rgba_copies_and_runner_support(
    tmp_path, no_seal
):
    parent_a, parent_b, combined = _combine_two(tmp_path)
    parent_hashes = {
        parent_a: combiner._sha256_file(parent_a),
        parent_b: combiner._sha256_file(parent_b),
    }

    loaded_path, payload = runner.load_pixal_inputs(combined)

    assert loaded_path == combined.resolve()
    assert payload["schema"] == combiner.COMBINED_PIXAL_INPUT_SCHEMA
    assert payload["asset_class"] == "static_object"
    assert payload["route"] == "flux2_pixal3d_static_v1"
    assert payload["parent_count"] == 2
    assert payload["job_count"] == 2
    assert payload["formal_dataset_registration_authorized"] is False
    assert [item["path"] for item in payload["parents"]] == sorted(
        [str(parent_a.resolve()), str(parent_b.resolve())]
    )
    for binding in payload["input_copies"]:
        original = binding["parent_pixal_input"]
        copied = binding["copied_pixal_input"]
        copied_path = Path(copied["path"])
        assert copied_path.is_file()
        assert not copied_path.is_symlink()
        assert copied["sha256"] == original["sha256"]
        assert copied["size_bytes"] == original["size_bytes"]
        assert copied_path.read_bytes() == Path(original["path"]).read_bytes()
    for job in payload["jobs"]:
        instance_id = job["controlled_request"]["instance_id"]
        assert Path(job["output"]) == (
            tmp_path
            / "combined_pixal_outputs"
            / instance_id
            / "pixal_raw_1024.glb"
        )
    assert {
        path: combiner._sha256_file(path) for path in parent_hashes
    } == parent_hashes


def test_combined_loader_fails_closed_when_a_parent_manifest_is_tampered(
    tmp_path, no_seal
):
    parent_a, _parent_b, combined = _combine_two(tmp_path)
    parent_a.write_text(parent_a.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(contracts.ContractError, match="parent Pixal manifest file changed"):
        runner.load_pixal_inputs(combined)


@pytest.mark.parametrize("tamper_target", ["source", "copied_rgba"])
def test_combined_loader_reauthenticates_source_and_copied_rgba_bytes(
    tmp_path, no_seal, tamper_target
):
    _parent_a, _parent_b, combined = _combine_two(tmp_path)
    payload = contracts.load_json(combined)
    binding = payload["input_copies"][0]
    path = Path(
        binding["source"]["path"]
        if tamper_target == "source"
        else binding["copied_pixal_input"]["path"]
    )
    path.write_bytes(path.read_bytes() + b"tamper")

    with pytest.raises(contracts.ContractError, match="(source|copied RGBA) file changed"):
        runner.load_pixal_inputs(combined)


@pytest.mark.parametrize("target", ["parent_manifest", "parent_pixal_plan"])
def test_combined_loader_rejects_rehashed_output_root_inside_parent_scope(
    tmp_path, no_seal, target
):
    parent_a, _parent_b, combined = _combine_two(tmp_path)
    payload = contracts.load_json(combined)
    if target == "parent_manifest":
        redirected_root = parent_a.parent / "unborn_combined_pixal_output"
    else:
        parent_payload = contracts.load_json(parent_a)
        redirected_root = (
            Path(parent_payload["pixal_output_root"])
            / "unborn_combined_pixal_output"
        )
    for job in payload["jobs"]:
        instance_id = job["controlled_request"]["instance_id"]
        output = redirected_root / instance_id / "pixal_raw_1024.glb"
        job["output"] = str(output)
        job["manifest"] = str(output.with_suffix(".manifest.json"))
    payload["pixal_output_root"] = str(redirected_root)
    payload["manifest_sha256"] = combiner._hash_without(payload, "manifest_sha256")
    combined.write_text(contracts.canonical_json(payload) + "\n", encoding="utf-8")

    with pytest.raises(contracts.ContractError, match="must not overlap|new output root"):
        runner.load_pixal_inputs(combined)


def test_rejects_mixed_animal_and_static_routes(tmp_path, no_seal):
    static_parent = _write_parent(
        tmp_path / "static_parent",
        instance_id="microwave_static",
        request_digit="3",
    )
    animal_parent = _write_parent(
        tmp_path / "animal_parent",
        instance_id="dog_animal",
        request_digit="4",
        route="flux2_pixal3d_animal_v1",
    )

    with pytest.raises(contracts.ContractError, match="mix routes"):
        combiner.combine_pixal_inputs(
            [static_parent, animal_parent],
            tmp_path / "combined",
            tmp_path / "pixal_outputs",
        )


@pytest.mark.parametrize("duplicate_kind", ["instance", "execution_job"])
def test_rejects_duplicate_global_instance_or_execution_job(
    tmp_path, no_seal, duplicate_kind
):
    instance_a = "telephone_a"
    instance_b = instance_a if duplicate_kind == "instance" else "telephone_b"
    request_a = "5"
    request_b = request_a if duplicate_kind == "execution_job" else "6"
    parent_a = _write_parent(
        tmp_path / "parent_a",
        instance_id=instance_a,
        request_digit=request_a,
    )
    parent_b = _write_parent(
        tmp_path / "parent_b",
        instance_id=instance_b,
        request_digit=request_b,
    )

    with pytest.raises(contracts.ContractError, match="duplicate Pixal"):
        combiner.combine_pixal_inputs(
            [parent_a, parent_b],
            tmp_path / "combined",
            tmp_path / "pixal_outputs",
        )


def test_reauthenticates_controlled_request_instead_of_only_trusting_parent_hash(
    tmp_path, no_seal
):
    parent_a = _write_parent(
        tmp_path / "parent_a",
        instance_id="telephone_a",
        request_digit="7",
    )
    parent_b = _write_parent(
        tmp_path / "parent_b",
        instance_id="telephone_b",
        request_digit="8",
    )
    payload = contracts.load_json(parent_b)
    payload["jobs"][0]["controlled_request"]["execution_job_id"] = "static_tampered"
    payload["manifest_sha256"] = combiner._hash_without(
        payload, "manifest_sha256"
    )
    parent_b.write_text(
        contracts.canonical_json(payload) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(contracts.ContractError, match="execution job changed"):
        combiner.combine_pixal_inputs(
            [parent_a, parent_b],
            tmp_path / "combined",
            tmp_path / "pixal_outputs",
        )


def test_refuses_to_overwrite_combined_or_pixal_output_roots(tmp_path, no_seal):
    parent_a = _write_parent(
        tmp_path / "parent_a",
        instance_id="telephone_a",
        request_digit="1",
    )
    parent_b = _write_parent(
        tmp_path / "parent_b",
        instance_id="telephone_b",
        request_digit="2",
    )
    combined_root = tmp_path / "combined"
    pixal_root = tmp_path / "pixal_outputs"
    combined_root.mkdir()

    with pytest.raises(contracts.ContractError, match="refusing to replace"):
        combiner.combine_pixal_inputs(
            [parent_a, parent_b], combined_root, pixal_root
        )

    combined_root.rmdir()
    pixal_root.mkdir()
    with pytest.raises(contracts.ContractError, match="existing Pixal output root"):
        combiner.combine_pixal_inputs(
            [parent_a, parent_b], combined_root, pixal_root
        )


def test_requires_a_new_pixal_output_root_instead_of_reusing_a_parent_plan(
    tmp_path, no_seal
):
    shared_parent_plan = tmp_path / "old_parent_plan"
    parent_a = _write_parent(
        tmp_path / "parent_a",
        instance_id="telephone_a",
        request_digit="3",
        pixal_output_root=shared_parent_plan,
    )
    parent_b = _write_parent(
        tmp_path / "parent_b",
        instance_id="telephone_b",
        request_digit="4",
    )

    with pytest.raises(contracts.ContractError, match="new output root"):
        combiner.combine_pixal_inputs(
            [parent_a, parent_b],
            tmp_path / "combined",
            shared_parent_plan,
        )


def test_atomic_no_replace_rejects_a_racing_empty_destination(
    tmp_path, no_seal, monkeypatch
):
    parent_a = _write_parent(
        tmp_path / "parent_a",
        instance_id="telephone_a",
        request_digit="5",
    )
    parent_b = _write_parent(
        tmp_path / "parent_b",
        instance_id="telephone_b",
        request_digit="6",
    )
    combined_root = tmp_path / "combined"
    real_rename = combiner._rename_noreplace

    def race_with_empty_directory(source, destination):
        destination.mkdir()
        real_rename(source, destination)

    monkeypatch.setattr(combiner, "_rename_noreplace", race_with_empty_directory)

    with pytest.raises(FileExistsError, match="concurrently-created"):
        combiner.combine_pixal_inputs(
            [parent_a, parent_b],
            combined_root,
            tmp_path / "pixal_outputs",
        )

    assert combined_root.is_dir()
    assert list(combined_root.iterdir()) == []
    assert list(tmp_path.glob(".combined.*.staging")) == []


def test_prepublish_validation_failure_leaves_no_final_root(
    tmp_path, no_seal, monkeypatch
):
    parent_a = _write_parent(
        tmp_path / "parent_a",
        instance_id="telephone_a",
        request_digit="7",
    )
    parent_b = _write_parent(
        tmp_path / "parent_b",
        instance_id="telephone_b",
        request_digit="8",
    )
    combined_root = tmp_path / "combined"

    def reject_staging(_manifest_path, _public_root):
        raise contracts.ContractError("injected prepublish validation failure")

    def forbid_publication(_source, _destination):
        raise AssertionError("publication must follow complete staging validation")

    monkeypatch.setattr(
        combiner, "_validate_staged_combined_manifest", reject_staging
    )
    monkeypatch.setattr(combiner, "_rename_noreplace", forbid_publication)

    with pytest.raises(contracts.ContractError, match="injected prepublish"):
        combiner.combine_pixal_inputs(
            [parent_a, parent_b],
            combined_root,
            tmp_path / "pixal_outputs",
        )

    assert not combined_root.exists()
    assert list(tmp_path.glob(".combined.*.staging")) == []
