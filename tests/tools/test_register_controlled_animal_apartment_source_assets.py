import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import wave

import numpy as np
import pytest
import soundfile as sf

from tools import controlled_source_asset_schema as schema
import tools.register_controlled_animal_apartment_source_assets as apartment_registration
from tests.tools import (
    test_build_user_approved_generated_animal_apartment_specs as apartment_builder_support,
)
from tools import (
    build_user_approved_generated_animal_apartment_specs as apartment_builder,
)
from tools import (
    measure_controlled_animal_physical_attributes as measurement_builder,
)
from tools.register_controlled_animal_apartment_source_assets import (
    _validate_audio,
    _validate_registration_audio,
    upgrade_source_asset,
)
from tools.spike_rlr.animal_audio import (
    pinned_animal_audio_contract,
    resolve_registered_animal_audio_artifacts,
    validate_animal_audio_evidence,
)
from tools.spike_rlr.acoustic_scene_contract import (
    approved_acoustic_scene_contract,
    approved_rlr_renderer_contract,
)
from tools.spike_rlr.active_frame_rir_evidence import (
    active_frame_indices,
    serialize_active_frame_rir_evidence,
)
from tools.spike_rlr.rlr_materials import build_rlr_materials_payload
from tools.spike_rlr.run_audio_pass_rlr import _load_dry_source


def _artifact():
    return {
        "root_id": "fixture_root",
        "path": "fixture.bin",
        "sha256": "a" * 64,
        "size_bytes": 1,
    }


def test_upgrade_preserves_absolute_identity_rights_and_passes_scene_qa():
    profile = schema.load_json(
        Path("data/controlled_source_attributes_v1/profiles/animal/cat_siamese_bindpose_v2.json")
    )
    request = schema.sample_instance_requests(profile, count=1, batch_seed=13)[0]
    asset = schema.build_source_asset_v2(
        request,
        artifacts={"static_mesh": _artifact()},
        physical_measurements={"status": "pending"},
        provenance={
            "attempt_id": "static_fixture_v1",
            "request_sha256": request["request_sha256"],
            "models": copy.deepcopy(request["generation_plan"]["model_revisions"]),
        },
        rights={
            "status": "review_required",
            "licenses": [_artifact()],
            "blockers": ["fixture_rights_review"],
        },
        qa={
            "reference_2d": "passed",
            "static_mesh": "passed",
            "binding": "pending",
            "walking": "pending",
            "idle": "pending",
            "ue_import_readback": "pending",
            "apartment_media": "pending",
            "audio": "pending",
        },
        state_classification="research_candidate",
    )
    measured = {
        "status": "measured",
        "method": "fixture_measurement_v1",
        "runtime": {"actor_scale": 0.1, "shoulder_height_cm": 30.0},
    }

    upgraded = upgrade_source_asset(
        asset,
        physical_measurements=measured,
        added_artifacts={"apartment_registry": _artifact()},
    )

    assert upgraded["asset_id"] == asset["asset_id"]
    assert upgraded["sampled_attributes"] == asset["sampled_attributes"]
    assert upgraded["physical_measurements"] == measured
    assert upgraded["rights"] == asset["rights"]
    assert upgraded["qa"] == {
        "reference_2d": "passed",
        "static_mesh": "passed",
        "binding": "passed",
        "walking": "passed",
        "idle": "passed",
        "ue_import_readback": "passed",
        "apartment_media": "passed",
        "audio": "pending",
    }
    assert upgraded["state_classification"] == "research_candidate"


def _apartment_v2_manifest(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    presentation_path = tmp_path / "presentation.json"
    presentation_path.write_text("{}")
    review_path = tmp_path / "review.mp4"
    review_path.write_bytes(b"v")
    input_path = tmp_path / "input.json"
    input_path.write_text("{}")
    presentation = {
        "presentation_receipt": _descriptor(presentation_path),
        "expected_presentation_receipt_file_sha256": _descriptor(
            presentation_path
        )["sha256"],
        "presentation_receipt_sha256": "b" * 64,
        "output_video": _descriptor(review_path),
    }
    descriptor = _descriptor(input_path)
    asset_id = "cat_british_shorthair_fixture"
    payload = {
        "schema": apartment_registration.APARTMENT_SCHEMA_V2,
        "generated_at": "2026-07-28T00:00:00+00:00",
        "usage_scope": "research_candidate",
        "formal_registration_authorized": False,
        "trajectory_policy": "fixture trajectory",
        "audio_policy": "fixture audio",
        "avatar_count": 1,
        "clip_count": 2,
        "presentation_evidence": presentation,
        "presentation_automatic_checks": copy.deepcopy(
            apartment_registration.APARTMENT_V2_PRESENTATION_CHECKS
        ),
        "inputs": {
            name: copy.deepcopy(descriptor)
            for name in (
                "config",
                "ue_import_jobs",
                "ue_import_result",
                "ue_import_preparation",
                "animation_decision",
                "animation_decision_freeze_receipt",
                "template",
            )
        },
        "records": [
            {
                "base_avatar_id": asset_id,
                "asset_id": asset_id,
                "tag": f"pixal_{asset_id}",
                "profile_schema_id": "cat_british_shorthair_v1",
                "species": "cat",
                "breed": "british_shorthair",
                "sampled_attributes": {"size": "medium"},
                "target_physical_profile": {"measurement": "shoulder_height_cm"},
                "source_glb": {
                    "path": str((tmp_path / "runtime.glb").resolve()),
                    "sha256": "e" * 64,
                },
                "actions": {
                    "Walking": {"clip_id": "walking"},
                    "Idle": {"clip_id": "idle"},
                },
            }
        ],
    }
    payload["manifest_sha256"] = apartment_registration._hash_without(
        payload,
        "manifest_sha256",
    )
    path = tmp_path / "spec_manifest.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_apartment_records_accepts_strict_builder_v2(tmp_path):
    inputs = apartment_builder_support._fixture(tmp_path)
    inputs.pop("semantic_evidence")
    manifest = apartment_builder.build_specs(
        **inputs,
        output_root=tmp_path / "apartment",
    )

    records = apartment_registration._load_apartment_records([manifest])

    assert set(records) == {"horse_candidate_001"}
    record = records["horse_candidate_001"]
    assert record["_authenticated_apartment_v2"]["runtime_lineage"] == record[
        "runtime_lineage"
    ]


def test_load_apartment_records_rejects_v2_authority_expansion(tmp_path):
    manifest = _apartment_v2_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["unreviewed_authority"] = True
    payload["manifest_sha256"] = apartment_registration._hash_without(
        payload,
        "manifest_sha256",
    )
    manifest.write_text(json.dumps(payload))

    with pytest.raises(schema.ContractError, match="v2 authority"):
        apartment_registration._load_apartment_records([manifest])


def test_load_apartment_records_rejects_v2_formal_promotion(tmp_path):
    manifest = _apartment_v2_manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["formal_registration_authorized"] = True
    payload["manifest_sha256"] = apartment_registration._hash_without(
        payload,
        "manifest_sha256",
    )
    manifest.write_text(json.dumps(payload))

    with pytest.raises(schema.ContractError, match="v2 authority"):
        apartment_registration._load_apartment_records([manifest])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": _sha(path),
        "size_bytes": len(payload),
    }


def _authenticated_measurement_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    shoulder_height_units: float = 2.5,
) -> tuple[Path, dict[str, dict]]:
    inputs = apartment_builder_support._fixture(tmp_path / "builder")
    inputs.pop("semantic_evidence")
    manifest_path = apartment_builder.build_specs(
        **inputs,
        output_root=tmp_path / "apartment",
    )
    apartment_records = apartment_registration._load_apartment_records(
        [manifest_path]
    )
    manifest_path.parent.chmod(0o755)
    record = apartment_records["horse_candidate_001"]
    visual_path = (
        Path(record["actions"]["Walking"]["output_dir"])
        / "videos"
        / "actor_visual_metadata.json"
    )
    visual_path.parent.mkdir(parents=True)
    frame = {
        "bounds_ue": {
            "minimum_cm": [0.0, 0.0, 20.0],
            "maximum_cm": [90.0, 30.0, 100.0],
        },
        "root_transform_ue": {"scale": [0.332, 0.332, 0.332]},
        "floor_contact": {"within_penetration_tolerance": True},
    }
    visual_path.write_text(
        json.dumps(
            {
                "automatic_checks": {"overall": "passed"},
                "sources": [
                    {
                        "tag": record["tag"],
                        "runtime_frames": [frame, frame],
                    }
                ],
            }
        )
        + "\n"
    )

    def fake_blender(command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        input_glb = Path(command[command.index("--input-glb") + 1])
        output.write_text(
            json.dumps(
                {
                    "schema": "weighted_quadruped_geometry_measurement_v1",
                    "input_glb": _descriptor(input_glb),
                    "mesh_name": "fixture_mesh",
                    "vertex_count": 1000,
                    "front_upper_groups": [
                        "generated_front_negative_upper",
                        "generated_front_positive_upper",
                    ],
                    "front_upper_group_authority": (
                        "hash_bound_rig_semantic_evidence_v1"
                    ),
                    "selected_shoulder_vertex_count": 200,
                    "quantiles": {
                        "floor": 0.001,
                        "top": 0.999,
                        "shoulder_surface": 0.95,
                        "length_min": 0.005,
                        "length_max": 0.995,
                    },
                    "bounds_height_units": 4.0,
                    "shoulder_height_units": shoulder_height_units,
                    "nose_to_tail_length_units": 5.0,
                    "shoulder_fraction_of_bounds_height": (
                        shoulder_height_units / 4.0
                    ),
                }
            )
            + "\n"
        )
        return subprocess.CompletedProcess(command, 0, "fixture blender\n")

    monkeypatch.setattr(measurement_builder.subprocess, "run", fake_blender)
    batch_path = measurement_builder.build_measurements(
        manifest_paths=[manifest_path],
        output_root=tmp_path / "measurements",
        blender=Path("/fixture/blender"),
        workers=1,
    )
    return batch_path, apartment_records


def test_measurement_consumer_recomputes_current_v2_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_path, apartment_records = _authenticated_measurement_fixture(
        tmp_path,
        monkeypatch,
    )

    measurements = apartment_registration._load_measurements(
        batch_path,
        apartment_records,
    )

    assert set(measurements) == {"horse_candidate_001"}
    runtime = measurements["horse_candidate_001"]["payload"][
        "physical_measurements"
    ]["runtime"]
    assert runtime["shoulder_height_cm"] == 50.0
    assert runtime["audio_source_height_offset_m"] == 1.3


def test_measurement_consumer_rejects_old_method_even_when_resealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_path, apartment_records = _authenticated_measurement_fixture(
        tmp_path,
        monkeypatch,
    )
    payload = json.loads(batch_path.read_text())
    payload["method"] = "ue_bounds_calibrated_weighted_foreleg_surface_v1"
    payload["batch_sha256"] = apartment_registration._hash_without(
        payload,
        "batch_sha256",
    )
    batch_path.chmod(0o644)
    batch_path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(schema.ContractError, match="invalid physical"):
        apartment_registration._load_measurements(
            batch_path,
            apartment_records,
        )


def test_measurement_consumer_rejects_outside_tolerance_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_path, apartment_records = _authenticated_measurement_fixture(
        tmp_path,
        monkeypatch,
        shoulder_height_units=2.0,
    )

    with pytest.raises(schema.ContractError, match="admission failed"):
        apartment_registration._load_measurements(
            batch_path,
            apartment_records,
        )


def _strict_source_spec(tag: str, lookup: str, contract: dict) -> dict:
    return {
        "tag": tag,
        "asset_class": "animal",
        "species": contract["species"],
        "audio_lookup": lookup,
        "audio_contract": contract,
        "audio_sha256": contract["sha256"],
        "audio_source_size_bytes": contract["size_bytes"],
        "audio_source_codec": contract["codec"],
        "audio_source_channels": contract["channels"],
        "audio_source_sample_width_bytes": contract["sample_width_bytes"],
        "audio_source_sample_rate_hz": contract["sample_rate_hz"],
        "audio_source_frame_count": contract["frame_count"],
        "audio_source_duration_s": contract["duration_s"],
        "audio_source_species": contract["species"],
        "audio_dry_source_policy": contract["dry_source_policy"],
        "audio_spatialization_status": contract["spatialization_status"],
        "audio_known_spatialized_derivative_sha256": contract[
            "known_spatialized_derivative_sha256"
        ],
        "audio_item_origin": contract["item_origin"],
        "audio_objective_content_qa_status": contract[
            "objective_audio_content_qa_status"
        ],
        "audio_item_level_license_status": contract[
            "item_level_license_status"
        ],
        "audio_item_level_license_snapshot": contract[
            "item_level_license_snapshot"
        ],
        "audio_formal_registration_authorized": contract[
            "formal_registration_authorized"
        ],
        "strict_audio": True,
    }


def _write_binaural(path: Path, *, near_silent: bool = False) -> None:
    rate = 16000
    duration_s = 18.0
    frame_count = int(rate * duration_s)
    mono = np.zeros(frame_count, dtype=np.int16)
    if not near_silent:
        for start, end in ((1600, 8000), (24000, 30400)):
            phase = np.arange(end - start, dtype=np.float64) / rate
            mono[start:end] = np.round(
                5000.0 * np.sin(2.0 * math.pi * 2500.0 * phase)
            ).astype(np.int16)
        mono = np.round(
            mono.astype(np.float64)
            * (0.9 * 32768.0 / np.max(np.abs(mono)))
        ).astype(np.int16)
    right = np.round(mono.astype(np.float64) * 0.65).astype(np.int16)
    stereo = np.column_stack((mono, right)).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(stereo.tobytes())


def _strict_audio_action(tmp_path: Path):
    tag = "pixal_dog_pembroke_welsh_corgi_candidate"
    lookup = "dog_bark"
    contract = pinned_animal_audio_contract(lookup)
    output = tmp_path / "audio"
    output.mkdir(parents=True)
    spec_path = tmp_path / "spec.json"
    source_spec = _strict_source_spec(tag, lookup, contract)
    n_frames = 270
    source_spec.update(
        {
            "trajectory_m": [[1.0, 1.0, 0.45]] * n_frames,
            "walking_forward_yaw_offset_deg": 0.0,
        }
    )
    spec_path.write_text(
        json.dumps(
            {
                "spec_version": "apartment_v1",
                "coordinate_frame": {
                    "system": "right-handed Z-up meters",
                },
                "mic": {"pos_m": [0.0, 0.0, 1.2], "yaw_deg": 0.0},
                "audio_config": {
                    "sample_rate_hz": 16000,
                    "duration_s": 18.0,
                },
                "render_config": {"n_frames": n_frames, "fps": 15},
                "sources": [source_spec],
            }
        )
    )
    source = {}
    dry = _load_dry_source(
        tag,
        sample_rate=16000,
        duration_s=18.0,
        source_spec=source_spec,
        schedule_metadata_out=source,
    )
    schedule_path = output / "binaural_source_schedule.json"
    schedule_path.write_text(
        json.dumps(
            {
                "schema": "rlr_audio_source_schedules_v1",
                "sources": {tag: source},
            }
        )
    )
    scheduled_dry = output / f"binaural_{tag}_scheduled_dry.wav"
    sf.write(str(scheduled_dry), dry, 16000, subtype="PCM_16")
    raw_wet = np.column_stack((dry, dry * 0.65)).astype(np.float32)
    pre_normalization_peak = float(np.max(np.abs(raw_wet)))
    normalized_wet = raw_wet * (0.9 / pre_normalization_peak)
    sf.write(
        str(output / "binaural.wav"),
        normalized_wet,
        16000,
        subtype="PCM_16",
    )
    solo = output / f"binaural_{tag}_binaural.wav"
    solo.write_bytes((output / "binaural.wav").read_bytes())
    staged = output / ".authenticated_inputs"
    staged.mkdir()
    mesh = staged / "acoustic_mesh.glb"
    materials = staged / "acoustic_materials.json"
    derived_materials = staged / "rlr_materials.json"
    mesh.write_bytes(Path("tmp/spike_rlr/apartment_v1_mesh.glb").read_bytes())
    materials.write_bytes(
        Path("tmp/spike_rlr/apartment_v1_materials.json").read_bytes()
    )
    derived_materials.write_bytes(
        build_rlr_materials_payload(
            json.loads(materials.read_text(encoding="utf-8"))
        )
    )
    for path in (mesh, materials, derived_materials):
        path.chmod(0o400)
    staged.chmod(0o500)
    frame_indices = active_frame_indices(
        dry,
        n_frames=n_frames,
        samples_per_frame=round(16000 / 15),
    )
    rir_path = output / f"{tag}_active_frame_binaural_rir.npz"
    rir_path.write_bytes(
        serialize_active_frame_rir_evidence(
            source_tag=tag,
            frame_indices=frame_indices,
            rirs=[
                np.asarray([[1.0], [0.65]], dtype=np.float32)
                for _ in frame_indices
            ],
            source_positions_scene_m=np.asarray(
                source_spec["trajectory_m"],
                dtype=np.float64,
            )[frame_indices],
            mic_position_scene_m=np.asarray([0.0, 0.0, 1.2]),
            mic_yaw_deg=0.0,
            sample_rate_hz=16000,
            n_samples_total=len(dry),
            n_frames=n_frames,
            fps=15.0,
            samples_per_frame=round(16000 / 15),
        )
    )
    spec_payload = json.loads(spec_path.read_text())
    (output / "binaural_audio_render_manifest.json").write_text(
        json.dumps(
            {
                "schema": "rlr_audio_render_manifest_v3",
                "backend": "habitat_sim_rlr_audio_sensor",
                "channel_layout": "binaural_native",
                "sample_rate_hz": 16000,
                "duration_s": 18.0,
                "n_frames": 270,
                "fps": 15.0,
                "quality_mode": "high",
                "indirect_ray_count": 500,
                "native_binaural_channel_order": [0, 1],
                "source_tags": [tag],
                "spec": _descriptor(spec_path),
                "acoustic_mesh": _descriptor(mesh),
                "acoustic_materials": _descriptor(materials),
                "derived_rlr_materials": _descriptor(derived_materials),
                "acoustic_scene_contract": approved_acoustic_scene_contract(
                    spec_payload
                ),
                "renderer_contract": approved_rlr_renderer_contract(
                    sample_rate_hz=16000,
                    quality_mode="high",
                ),
                "source_schedule": _descriptor(schedule_path),
                "output_wav": _descriptor(output / "binaural.wav"),
                "per_source_outputs": {
                    tag: {
                        "binaural": _descriptor(solo),
                        "scheduled_dry": _descriptor(scheduled_dry),
                        "active_frame_rir": _descriptor(rir_path),
                        "pre_normalization_peak": pre_normalization_peak,
                        "mic_local_azimuth_deg_per_frame": [45.0] * n_frames,
                    }
                },
                "mix_pre_normalization_peak": pre_normalization_peak,
                "behavior_gates": {
                    "per_source_event_alignment": "validate_on_readback",
                    "binaural_spatial_ild": "validate_on_readback",
                    "mixture_reconstruction": "validate_on_readback",
                    "active_frame_rir_replay": "validate_on_readback",
                },
                "technical_render_status": "passed",
                "formal_registration_authorized": False,
            }
        )
    )
    return (
        {"output_dir": str(output), "spec": str(spec_path)},
        tag,
        schedule_path,
    )


def _install_fake_independent_worker(
    monkeypatch,
    *,
    mutate=None,
    returncode=0,
    write_result=True,
    timeout=False,
):
    calls = []

    def fake_run(command, **kwargs):
        if timeout:
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        snapshot_fd = int(command[command.index("--snapshot-fd") + 1])
        request_name = command[command.index("--request-name") + 1]
        output_fd = int(command[command.index("--output-fd") + 1])
        request_fd = os.open(request_name, os.O_RDONLY, dir_fd=snapshot_fd)
        try:
            request_payload = b""
            while True:
                chunk = os.read(request_fd, 1024 * 1024)
                if not chunk:
                    break
                request_payload += chunk
        finally:
            os.close(request_fd)
        request = json.loads(request_payload)
        calls.append(
            {
                "command": list(command),
                "kwargs": dict(kwargs),
                "request": request,
            }
        )
        if write_result:
            subprocess_contract = request["subprocess_contract"]
            renderer_contract = request["renderer_contract"]
            arrays = {
                "schema": np.asarray(
                    apartment_registration.INDEPENDENT_RLR_RESULT_SCHEMA
                ),
                "request_sha256": np.asarray(
                    hashlib.sha256(request_payload).hexdigest()
                ),
                "subprocess_contract_sha256": np.asarray(
                    apartment_registration._canonical_sha256(
                        subprocess_contract
                    )
                ),
                "renderer_contract_sha256": np.asarray(
                    apartment_registration._canonical_sha256(
                        renderer_contract
                    )
                ),
                "environment_contract_sha256": np.asarray(
                    apartment_registration._canonical_sha256(
                        subprocess_contract["environment"]
                    )
                ),
                "runtime_packages_sha256": np.asarray(
                    apartment_registration._canonical_sha256(
                        subprocess_contract["runtime_packages"]
                    )
                ),
                "python_sha256": np.asarray(
                    subprocess_contract["python"]["sha256"]
                ),
                "worker_sha256": np.asarray(
                    subprocess_contract["worker"]["sha256"]
                ),
                "source_tag": np.asarray(request["source_tag"]),
                "frame_index": np.asarray(
                    request["frame_index"],
                    dtype=np.int64,
                ),
                "selection_seed_sha256": np.asarray(
                    request["selection_seed_sha256"]
                ),
                "source_position_coordinate_frame": np.asarray(
                    request["source_position_coordinate_frame"]
                ),
                "trajectory_prefix_sha256": np.asarray(
                    request["trajectory_prefix_sha256"]
                ),
                "trajectory_prefix_shape": np.asarray(
                    request["trajectory_prefix_shape"],
                    dtype=np.int64,
                ),
                "channel_order": np.asarray([0, 1], dtype=np.int64),
                "rir": np.asarray([[1.0], [0.65]], dtype=np.float32),
            }
            if mutate is not None:
                mutate(arrays, request)
            os.lseek(output_fd, 0, os.SEEK_SET)
            with os.fdopen(os.dup(output_fd), "wb") as stream:
                np.savez(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
        return subprocess.CompletedProcess(
            command,
            returncode,
            stdout=b'{"stage":"observation_ready"}\n',
            stderr=b"worker failed" if returncode else b"",
        )

    monkeypatch.setattr(
        apartment_registration,
        "_run_isolated_process_group",
        fake_run,
    )
    return calls


def test_isolated_process_group_kills_and_reaps_the_session_on_timeout(
    monkeypatch,
):
    events = []

    class FakeProcess:
        pid = 43210
        returncode = -signal.SIGKILL
        stdout = None
        stderr = None

        def communicate(self, timeout):
            events.append(("communicate", timeout))
            if len([event for event in events if event[0] == "communicate"]) == 1:
                raise subprocess.TimeoutExpired(["worker"], timeout)
            return b"partial stdout", b"partial stderr"

        def poll(self):
            return None

        def kill(self):
            events.append(("kill", self.pid))

        def wait(self, timeout):
            events.append(("wait", timeout))
            return self.returncode

    def fake_popen(command, **kwargs):
        events.append(("popen", list(command), dict(kwargs)))
        return FakeProcess()

    monkeypatch.setattr(apartment_registration.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        apartment_registration.os,
        "killpg",
        lambda pid, sig: events.append(("killpg", pid, sig)),
    )

    with pytest.raises(subprocess.TimeoutExpired) as raised:
        apartment_registration._run_isolated_process_group(
            ["approved-python", "worker.py"],
            executable="/proc/self/fd/17",
            cwd="/approved/repo",
            env={"LANG": "C.UTF-8"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            pass_fds=(17, 18),
            timeout=0.25,
        )

    popen = events[0]
    assert popen[0] == "popen"
    assert popen[2]["executable"] == "/proc/self/fd/17"
    assert popen[2]["pass_fds"] == (17, 18)
    assert popen[2]["start_new_session"] is True
    assert ("killpg", 43210, signal.SIGKILL) in events
    assert events[-1] == ("communicate", 5.0)
    assert raised.value.output == b"partial stdout"
    assert raised.value.stderr == b"partial stderr"


def test_registration_audio_requires_fixed_independent_subprocess_sample(
    tmp_path,
    monkeypatch,
):
    action, tag, _schedule = _strict_audio_action(tmp_path)
    calls = _install_fake_independent_worker(monkeypatch)
    evidence_base = tmp_path / "registration_evidence" / "walking"

    audio = _validate_registration_audio(
        action,
        tag=tag,
        action="Walking",
        evidence_output_base=evidence_base,
    )

    assert len(calls) == 1
    call = calls[0]
    contract = call["request"]["subprocess_contract"]
    assert call["command"][:4] == [
        contract["python"]["resolved_path"],
        "-I",
        "-B",
        "-S",
    ]
    assert call["command"][4].startswith("/proc/self/fd/")
    assert call["kwargs"]["cwd"] == str(
        apartment_registration.SPEAR_ROOT
    )
    assert call["kwargs"]["env"] == contract["environment"]
    assert "PYTHONPATH" not in call["kwargs"]["env"]
    assert "PYTHONHOME" not in call["kwargs"]["env"]
    assert "LD_LIBRARY_PATH" not in call["kwargs"]["env"]
    assert call["kwargs"]["timeout"] == 600.0
    executable = call["kwargs"]["executable"]
    assert executable.startswith("/proc/self/fd/")
    executable_fd = int(executable.rsplit("/", 1)[1])
    assert executable_fd in call["kwargs"]["pass_fds"]
    assert len(call["kwargs"]["pass_fds"]) == 4
    request = call["request"]
    assert request["trajectory_prefix_shape"] == [
        request["frame_index"] + 1,
        3,
    ]
    prefix = np.asarray(
        request["warmup_source_positions_scene_m"],
        dtype=np.dtype("<f8"),
    )
    assert hashlib.sha256(prefix.tobytes(order="C")).hexdigest() == request[
        "trajectory_prefix_sha256"
    ]
    assert not {
        "rir",
        "rir_left",
        "rir_right",
        "expected_rir",
        "active_frame_rir",
    } & set(request)
    report = audio["independent_rlr_sample"]["report"]
    assert report["status"] == "passed"
    assert report["frame_selection"]["seed_sha256"] == call["request"][
        "selection_seed_sha256"
    ]
    assert report["comparison"]["max_absolute_error"] == 0.0
    assert report["runtime_assurance_scope"]["classification"] == (
        "process_isolated_artifact_pinned_not_hermetic_v1"
    )
    assert Path(f"{evidence_base}.npz").is_file()
    assert Path(f"{evidence_base}.json").is_file()


def test_registration_audio_rejects_prefix_final_position_that_disagrees_with_rir(
    tmp_path,
):
    action, tag, _schedule = _strict_audio_action(tmp_path)
    audio = _validate_audio(action, tag)
    manifest = audio["render_manifest_value"]
    selected = apartment_registration._select_expected_active_frame_rir(
        descriptor=manifest["per_source_outputs"][tag]["active_frame_rir"],
        tag=tag,
        action="Walking",
        manifest=manifest,
    )
    trajectory = np.asarray(
        audio["validated_source_trajectory_scene_m"],
        dtype=np.float64,
    )
    trajectory[selected["frame_index"], 0] += 0.125

    with pytest.raises(schema.ContractError, match="trajectory changed"):
        apartment_registration._validate_independent_active_frame_rir_sample(
            render_manifest=manifest,
            validated_source_trajectory_scene_m=trajectory.tolist(),
            tag=tag,
            action="Walking",
            evidence_output_base=tmp_path / "position_mismatch",
        )


def test_registration_audio_independent_sample_timeout_is_fail_closed(
    tmp_path,
    monkeypatch,
):
    action, tag, _schedule = _strict_audio_action(tmp_path)
    calls = _install_fake_independent_worker(monkeypatch, timeout=True)

    with pytest.raises(schema.ContractError, match="timed out"):
        _validate_registration_audio(
            action,
            tag=tag,
            action="Walking",
            evidence_output_base=tmp_path / "timeout_evidence",
        )

    assert calls == []
    assert not (tmp_path / "timeout_evidence.npz").exists()


@pytest.mark.parametrize(
    ("returncode", "write_result", "message"),
    [
        (17, True, "crashed"),
        (0, False, "missing/oversized"),
    ],
)
def test_registration_audio_rejects_worker_crash_or_missing_result(
    tmp_path,
    monkeypatch,
    returncode,
    write_result,
    message,
):
    action, tag, _schedule = _strict_audio_action(tmp_path)
    _install_fake_independent_worker(
        monkeypatch,
        returncode=returncode,
        write_result=write_result,
    )

    with pytest.raises(schema.ContractError, match=message):
        _validate_registration_audio(
            action,
            tag=tag,
            action="Walking",
            evidence_output_base=tmp_path / "failed_evidence",
        )


def _mutate_independent_result(variant):
    def mutate(arrays, _request):
        if variant == "wrong_request_hash":
            arrays["request_sha256"] = np.asarray("0" * 64)
        elif variant == "wrong_channels":
            arrays["rir"] = np.asarray([[1.0]], dtype=np.float32)
        elif variant == "swapped_channels":
            arrays["rir"] = arrays["rir"][::-1].copy()
        elif variant == "tampered_values":
            arrays["rir"] = arrays["rir"] * np.float32(0.5)
        elif variant == "extra_array":
            arrays["forged"] = np.asarray(1, dtype=np.int64)
        elif variant == "nonfinite":
            arrays["rir"][0, 0] = np.float32(np.nan)
        elif variant == "wrong_trajectory_hash":
            arrays["trajectory_prefix_sha256"] = np.asarray("0" * 64)
        elif variant == "wrong_trajectory_shape":
            arrays["trajectory_prefix_shape"] = np.asarray(
                [1, 3],
                dtype=np.int64,
            )
        elif variant == "wrong_runtime_packages":
            arrays["runtime_packages_sha256"] = np.asarray("0" * 64)
        else:  # pragma: no cover - test construction guard
            raise AssertionError(variant)

    return mutate


@pytest.mark.parametrize(
    "variant",
    [
        "wrong_request_hash",
        "wrong_channels",
        "swapped_channels",
        "tampered_values",
        "extra_array",
        "nonfinite",
        "wrong_trajectory_hash",
        "wrong_trajectory_shape",
        "wrong_runtime_packages",
    ],
)
def test_registration_audio_rejects_tampered_independent_result(
    tmp_path,
    monkeypatch,
    variant,
):
    action, tag, _schedule = _strict_audio_action(tmp_path)
    _install_fake_independent_worker(
        monkeypatch,
        mutate=_mutate_independent_result(variant),
    )

    with pytest.raises(schema.ContractError):
        _validate_registration_audio(
            action,
            tag=tag,
            action="Walking",
            evidence_output_base=tmp_path / "tampered_evidence",
        )


@pytest.mark.parametrize("worker_state", ["missing", "wrong_hash"])
def test_registration_audio_rejects_missing_or_wrong_hash_worker(
    tmp_path,
    monkeypatch,
    worker_state,
):
    action, tag, _schedule = _strict_audio_action(tmp_path / "fixture")
    fake_root = tmp_path / "fake_spear"
    worker = fake_root / "tools/spike_rlr/sample_active_frame_rir_rlr.py"
    if worker_state == "wrong_hash":
        worker.parent.mkdir(parents=True)
        worker.write_text("#!/usr/bin/env python3\nraise SystemExit(0)\n")
    monkeypatch.setattr(apartment_registration, "SPEAR_ROOT", fake_root)
    _install_fake_independent_worker(monkeypatch)

    with pytest.raises(
        schema.ContractError,
        match="worker (is missing or unsafe|hash changed)",
    ):
        _validate_registration_audio(
            action,
            tag=tag,
            action="Walking",
            evidence_output_base=tmp_path / "worker_evidence",
        )


def test_apartment_registry_wires_independent_gate_for_walk_and_idle(
    tmp_path,
    monkeypatch,
):
    tag = "dog_registration_wiring"
    asset_id = "animal_registration_wiring_v1"
    walking_output = tmp_path / "actions" / "Walking"
    idle_output = tmp_path / "actions" / "Idle"
    walking_output.mkdir(parents=True)
    idle_output.mkdir(parents=True)
    registry_dir = walking_output.parent / "registry"
    registry_dir.mkdir()
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "asset_id": asset_id,
                "decision": "approved_for_ue_apartment",
                "checks": {"all": True},
            }
        )
    )
    import_path = tmp_path / "ue_import.json"
    import_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "tag": tag,
                        "status": "passed",
                        "actions": ["Walking", "Idle"],
                        "source_sha256": "b" * 64,
                    }
                ]
            }
        )
    )
    registry = {
        "schema_version": (
            apartment_registration.APARTMENT_REGISTRY_SCHEMA
        ),
        "usage_scope": "research_candidate",
        "formal_registry_promotion": False,
        "tag": tag,
        "asset_id": asset_id,
        "sampled_attributes": {"coat": "fixture"},
        "animation_decision": _descriptor(decision_path),
        "ue_import_result": _descriptor(import_path),
        "ue_source_sha256": "b" * 64,
        "clips": {
            "Walking": {"clip_id": "walk_clip"},
            "Idle": {"clip_id": "idle_clip"},
        },
    }
    (registry_dir / f"{tag}.json").write_text(json.dumps(registry))
    record = {
        "tag": tag,
        "base_avatar_id": asset_id,
        "sampled_attributes": {"coat": "fixture"},
        "actions": {
            "Walking": {
                "clip_id": "walk_clip",
                "output_dir": str(walking_output),
            },
            "Idle": {
                "clip_id": "idle_clip",
                "output_dir": str(idle_output),
            },
        },
    }
    placeholder = tmp_path / "authority-placeholder"
    placeholder.write_bytes(b"authority")
    placeholder_descriptor = _descriptor(placeholder)
    reviewed_descriptor = {
        "path": str(placeholder.resolve()),
        "sha256": "b" * 64,
        "size_bytes": placeholder.stat().st_size,
    }
    record["source_glb"] = {
        "path": reviewed_descriptor["path"],
        "sha256": reviewed_descriptor["sha256"],
    }
    record["_authenticated_apartment_v2"] = {
        "inputs": {
            "animation_decision": {
                **_descriptor(decision_path),
                "decision_sha256": "d" * 64,
            },
            "ue_import_result": _descriptor(import_path),
            "ue_import_jobs": placeholder_descriptor,
            "ue_import_preparation": placeholder_descriptor,
            "animation_decision_freeze_receipt": placeholder_descriptor,
        },
        "runtime_lineage": {
            "reviewed_animated_glb": reviewed_descriptor,
            "ue_import_glb": copy.deepcopy(reviewed_descriptor),
            "texture_transcode_manifest": None,
        },
        "presentation_evidence": {
            "presentation_receipt": placeholder_descriptor,
            "output_video": placeholder_descriptor,
        },
        "emitter_measurement": placeholder_descriptor,
    }
    for action_name, output_dir in (
        ("Walking", walking_output),
        ("Idle", idle_output),
    ):
        videos_dir = output_dir / "videos"
        videos_dir.mkdir()
        rendered_paths = {
            "spec": output_dir / "spec.json",
            "runtime_gate": output_dir / "runtime_gate.json",
            "actor_visual_metadata": (
                videos_dir / "actor_visual_metadata.json"
            ),
            "apartment_video": videos_dir / "apartment_v1_view0.mp4",
            "topdown_review_video": videos_dir / "topdown_review.mp4",
            "annotated_review_video": (
                videos_dir / "side_by_side_review_annotated.mp4"
            ),
        }
        for name, path in rendered_paths.items():
            path.write_bytes(f"{action_name}-{name}".encode())
        registry["clips"][action_name].update(
            {
                name: _descriptor(path)
                for name, path in rendered_paths.items()
            }
        )
        record["actions"][action_name]["spec_evidence"] = _descriptor(
            rendered_paths["spec"]
        )
    (registry_dir / f"{tag}.json").write_text(json.dumps(registry))
    calls = []

    def fake_registration_audio(
        _action_record,
        *,
        tag,
        action,
        evidence_output_base,
    ):
        calls.append((tag, action, Path(evidence_output_base)))
        result = Path(f"{evidence_output_base}.npz")
        report = Path(f"{evidence_output_base}.json")
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_bytes(b"independent result")
        report.write_bytes(b"independent report")
        common = tmp_path / "common"
        common.mkdir(exist_ok=True)
        paths = {}
        for role in (
            "audio",
            "schedule",
            "render_manifest",
            "render_spec_snapshot",
            "binaural_solo",
            "scheduled_dry",
            "active_frame_rir",
            "acoustic_mesh",
            "acoustic_materials",
            "derived_rlr_materials",
            "pinned_dry_source",
        ):
            path = common / role
            if not path.exists():
                path.write_bytes(role.encode())
            paths[role] = path
        paths["independent_rlr_sample"] = {
            "result_path": result,
            "report_path": report,
        }
        return paths

    monkeypatch.setattr(
        apartment_registration,
        "_validate_registration_audio",
        fake_registration_audio,
    )
    monkeypatch.setattr(
        apartment_registration,
        "_pinned_audio_source_artifact",
        lambda _path: {
            "root_id": "fixture_root",
            "path": "dry.wav",
            "sha256": "c" * 64,
            "size_bytes": 1,
        },
    )
    monkeypatch.setattr(
        apartment_registration,
        "_future_spear_artifact",
        lambda staged, final: {
            "root_id": "fixture_root",
            "path": Path(final).name,
            "sha256": hashlib.sha256(Path(staged).read_bytes()).hexdigest(),
            "size_bytes": Path(staged).stat().st_size,
        },
    )

    _path, _payload, extra = (
        apartment_registration._validate_apartment_registry(
            record,
            independent_evidence_staging_root=tmp_path / "staging_samples",
            independent_evidence_final_root=tmp_path / "final_samples",
        )
    )

    assert [item[1] for item in calls] == ["Walking", "Idle"]
    assert "apartment_walking_independent_rlr_sample" in extra
    assert "apartment_walking_independent_rlr_sample_report" in extra
    assert "apartment_idle_independent_rlr_sample" in extra
    assert "apartment_idle_independent_rlr_sample_report" in extra

    registry["clips"]["Walking"]["spec"] = _descriptor(decision_path)
    (registry_dir / f"{tag}.json").write_text(json.dumps(registry))
    with pytest.raises(schema.ContractError, match="rendered a different spec"):
        apartment_registration._validate_apartment_registry(
            record,
            independent_evidence_staging_root=tmp_path / "staging_samples_2",
            independent_evidence_final_root=tmp_path / "final_samples_2",
        )


def test_apartment_registry_rejects_render_b_for_reviewed_asset_a_without_transcode(
    tmp_path: Path,
) -> None:
    tag = "pixal_lineage_fixture"
    asset_id = "lineage_fixture"
    walking_output = tmp_path / "actions" / "Walking"
    idle_output = tmp_path / "actions" / "Idle"
    walking_output.mkdir(parents=True)
    idle_output.mkdir(parents=True)
    registry_dir = walking_output.parent / "registry"
    registry_dir.mkdir()
    reviewed = tmp_path / "reviewed.glb"
    reviewed.write_bytes(b"reviewed-a")
    ue_import = tmp_path / "ue-import.glb"
    ue_import.write_bytes(b"render-b")
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "asset_id": asset_id,
                "decision": "approved_for_ue_apartment",
                "checks": {"all": True},
            }
        )
    )
    import_path = tmp_path / "ue_import.json"
    import_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "tag": tag,
                        "status": "passed",
                        "actions": ["Walking", "Idle"],
                        "source_sha256": _sha(ue_import),
                    }
                ]
            }
        )
    )
    decision_descriptor = _descriptor(decision_path)
    result_descriptor = _descriptor(import_path)
    registry = {
        "schema_version": apartment_registration.APARTMENT_REGISTRY_SCHEMA,
        "usage_scope": "research_candidate",
        "formal_registry_promotion": False,
        "tag": tag,
        "asset_id": asset_id,
        "sampled_attributes": {"coat": "fixture"},
        "animation_decision": decision_descriptor,
        "ue_import_result": result_descriptor,
        "ue_source_sha256": _sha(ue_import),
        "clips": {
            "Walking": {"clip_id": "walk_clip"},
            "Idle": {"clip_id": "idle_clip"},
        },
    }
    (registry_dir / f"{tag}.json").write_text(json.dumps(registry))
    placeholder = tmp_path / "placeholder"
    placeholder.write_bytes(b"p")
    placeholder_descriptor = _descriptor(placeholder)
    record = {
        "tag": tag,
        "base_avatar_id": asset_id,
        "sampled_attributes": {"coat": "fixture"},
        "source_glb": {
            "path": str(reviewed.resolve()),
            "sha256": _sha(reviewed),
        },
        "_authenticated_apartment_v2": {
            "inputs": {
                "animation_decision": decision_descriptor,
                "ue_import_result": result_descriptor,
                "ue_import_jobs": placeholder_descriptor,
                "ue_import_preparation": placeholder_descriptor,
                "animation_decision_freeze_receipt": placeholder_descriptor,
            },
            "runtime_lineage": {
                "reviewed_animated_glb": _descriptor(reviewed),
                "ue_import_glb": _descriptor(ue_import),
                "texture_transcode_manifest": None,
            },
            "presentation_evidence": {
                "presentation_receipt": placeholder_descriptor,
                "output_video": placeholder_descriptor,
            },
            "emitter_measurement": placeholder_descriptor,
        },
        "actions": {
            "Walking": {
                "clip_id": "walk_clip",
                "output_dir": str(walking_output),
            },
            "Idle": {
                "clip_id": "idle_clip",
                "output_dir": str(idle_output),
            },
        },
    }

    with pytest.raises(schema.ContractError, match="runtime lineage changed"):
        apartment_registration._validate_apartment_registry(
            record,
            independent_evidence_staging_root=tmp_path / "staging",
            independent_evidence_final_root=tmp_path / "final",
        )


def test_validate_audio_reauthenticates_full_nonformal_dry_source_contract(
    tmp_path,
):
    action, tag, _schedule = _strict_audio_action(tmp_path)

    evidence = _validate_audio(action, tag)

    assert evidence["audio"].name == "binaural.wav"
    assert evidence["schedule"].name == "binaural_source_schedule.json"
    assert evidence["render_manifest"].name == (
        "binaural_audio_render_manifest.json"
    )
    assert evidence["binaural_solo"].name.endswith("_binaural.wav")
    assert evidence["scheduled_dry"].name.endswith("_scheduled_dry.wav")
    assert evidence["active_frame_rir"].name.endswith(
        "_active_frame_binaural_rir.npz"
    )
    assert evidence["acoustic_mesh"].name == "acoustic_mesh.glb"
    assert evidence["acoustic_materials"].name == "acoustic_materials.json"
    assert evidence["derived_rlr_materials"].name == "rlr_materials.json"


def _portable_audio_bundle(tmp_path, monkeypatch):
    import tools.spike_rlr.animal_audio as animal_audio

    action, tag, schedule_path = _strict_audio_action(tmp_path)
    output = Path(action["output_dir"])
    spec_path = Path(action["spec"])
    manifest_path = output / "binaural_audio_render_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    schedule = json.loads(schedule_path.read_text())
    spec = json.loads(spec_path.read_text())
    pinned_source = Path(
        schedule["sources"][tag]["source_contract"]["path"]
    )

    # Make the catalog's canonical source path deliberately nonexistent while
    # preserving the exact pinned bytes/hash in the portable root.
    missing_pinned = tmp_path / "removed_original" / pinned_source.name
    catalog = json.loads(
        animal_audio.AUDIO_LIBRARY_PATH.read_text(encoding="utf-8")
    )
    for item in catalog["samples"]:
        if item.get("category") == "dog_bark":
            item["path"] = str(missing_pinned)
    catalog_path = tmp_path / "portable_audio_library.json"
    catalog_path.write_text(json.dumps(catalog))
    monkeypatch.setattr(animal_audio, "AUDIO_LIBRARY_PATH", catalog_path)
    spec["sources"][0]["audio_contract"]["path"] = str(missing_pinned)
    schedule["sources"][tag]["source_contract"]["path"] = str(missing_pinned)
    schedule["sources"][tag]["audio_path"] = str(missing_pinned)
    spec_path.write_text(json.dumps(spec))
    schedule_path.write_text(json.dumps(schedule))
    manifest["spec"] = _descriptor(spec_path)
    manifest["source_schedule"] = _descriptor(schedule_path)
    manifest_path.write_text(json.dumps(manifest))

    repo_root = tmp_path / "portable_repo"
    action_root = repo_root / "walking"
    action_root.mkdir(parents=True)
    pinned_root = tmp_path / "portable_pinned"
    pinned_root.mkdir()

    source_paths = {
        "spec": spec_path,
        "acoustic_mesh": Path(manifest["acoustic_mesh"]["path"]),
        "acoustic_materials": Path(
            manifest["acoustic_materials"]["path"]
        ),
        "derived_rlr_materials": Path(
            manifest["derived_rlr_materials"]["path"]
        ),
        "source_schedule": schedule_path,
        "output_wav": Path(manifest["output_wav"]["path"]),
        "binaural": Path(
            manifest["per_source_outputs"][tag]["binaural"]["path"]
        ),
        "scheduled_dry": Path(
            manifest["per_source_outputs"][tag]["scheduled_dry"]["path"]
        ),
        "active_frame_rir": Path(
            manifest["per_source_outputs"][tag]["active_frame_rir"]["path"]
        ),
    }
    copied = {}
    for key, source in source_paths.items():
        destination = action_root / source.name
        shutil.copy2(source, destination)
        copied[key] = destination
    copied_pinned = pinned_root / pinned_source.name
    shutil.copy2(pinned_source, copied_pinned)

    def registered(root_id, root, path):
        payload = path.read_bytes()
        return {
            "root_id": root_id,
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    artifacts = {
        "apartment_walking_spec": registered(
            "portable_repo", repo_root, copied["spec"]
        ),
        "apartment_walking_acoustic_mesh": registered(
            "portable_repo", repo_root, copied["acoustic_mesh"]
        ),
        "apartment_walking_acoustic_materials": registered(
            "portable_repo", repo_root, copied["acoustic_materials"]
        ),
        "apartment_walking_derived_rlr_materials": registered(
            "portable_repo", repo_root, copied["derived_rlr_materials"]
        ),
        "apartment_walking_audio_schedule": registered(
            "portable_repo", repo_root, copied["source_schedule"]
        ),
        "apartment_walking_binaural_audio": registered(
            "portable_repo", repo_root, copied["output_wav"]
        ),
        "apartment_walking_binaural_solo": registered(
            "portable_repo", repo_root, copied["binaural"]
        ),
        "apartment_walking_scheduled_dry": registered(
            "portable_repo", repo_root, copied["scheduled_dry"]
        ),
        "apartment_walking_active_frame_rir": registered(
            "portable_repo", repo_root, copied["active_frame_rir"]
        ),
        "animal_audio_pinned_dry_source": registered(
            "portable_pinned", pinned_root, copied_pinned
        ),
    }
    root_paths = {
        "portable_repo": repo_root,
        "portable_pinned": pinned_root,
    }

    # Every absolute path embedded by the original renderer is now absent.
    spec_path.rename(tmp_path / "removed_original_spec.json")
    output.rename(tmp_path / "removed_original_audio")
    assert not Path(manifest["spec"]["path"]).exists()
    assert not Path(manifest["output_wav"]["path"]).exists()
    assert not missing_pinned.exists()
    return {
        "tag": tag,
        "spec": spec,
        "schedule": schedule,
        "manifest": manifest,
        "artifacts": artifacts,
        "root_paths": root_paths,
        "copied": copied,
    }


def test_portable_audio_artifact_overrides_revalidate_without_original_paths(
    tmp_path,
    monkeypatch,
):
    fixture = _portable_audio_bundle(tmp_path, monkeypatch)
    overrides = resolve_registered_animal_audio_artifacts(
        artifacts=fixture["artifacts"],
        root_paths=fixture["root_paths"],
        action="Walking",
        tag=fixture["tag"],
    )

    result = validate_animal_audio_evidence(
        spec=fixture["spec"],
        schedule=fixture["schedule"],
        audio_path=Path(fixture["manifest"]["output_wav"]["path"]),
        expected_tags={fixture["tag"]},
        audio_descriptor=fixture["manifest"]["output_wav"],
        render_manifest=fixture["manifest"],
        spec_path=Path(fixture["manifest"]["spec"]["path"]),
        schedule_path=Path(
            fixture["manifest"]["source_schedule"]["path"]
        ),
        artifact_overrides=overrides,
    )

    assert result["render_manifest"]["active_frame_rir_mix_replay"][
        "max_pcm16_error_lsb"
    ] <= 2.0


def test_portable_audio_override_rejects_self_consistent_wrong_rir(
    tmp_path,
    monkeypatch,
):
    fixture = _portable_audio_bundle(tmp_path, monkeypatch)
    rir = fixture["copied"]["active_frame_rir"]
    rir.write_bytes(rir.read_bytes() + b"forged")
    payload = rir.read_bytes()
    record = fixture["artifacts"]["apartment_walking_active_frame_rir"]
    record["sha256"] = hashlib.sha256(payload).hexdigest()
    record["size_bytes"] = len(payload)
    overrides = resolve_registered_animal_audio_artifacts(
        artifacts=fixture["artifacts"],
        root_paths=fixture["root_paths"],
        action="Walking",
        tag=fixture["tag"],
    )

    with pytest.raises(ValueError, match="disagrees with its manifest"):
        validate_animal_audio_evidence(
            spec=fixture["spec"],
            schedule=fixture["schedule"],
            audio_path=Path("missing.wav"),
            expected_tags={fixture["tag"]},
            render_manifest=fixture["manifest"],
            artifact_overrides=overrides,
        )


def test_portable_audio_resolver_rejects_symlink_and_root_escape(
    tmp_path,
    monkeypatch,
):
    fixture = _portable_audio_bundle(tmp_path, monkeypatch)
    rir = fixture["copied"]["active_frame_rir"]
    direct = rir.with_name("direct_rir.npz")
    rir.rename(direct)
    rir.symlink_to(direct)
    with pytest.raises((OSError, ValueError)):
        resolve_registered_animal_audio_artifacts(
            artifacts=fixture["artifacts"],
            root_paths=fixture["root_paths"],
            action="Walking",
            tag=fixture["tag"],
        )

    fixture["artifacts"]["apartment_walking_active_frame_rir"][
        "path"
    ] = "../escape.npz"
    with pytest.raises(ValueError, match="escaped"):
        resolve_registered_animal_audio_artifacts(
            artifacts=fixture["artifacts"],
            root_paths=fixture["root_paths"],
            action="Walking",
            tag=fixture["tag"],
        )


def test_portable_audio_validation_detects_post_resolution_replacement(
    tmp_path,
    monkeypatch,
):
    fixture = _portable_audio_bundle(tmp_path, monkeypatch)
    overrides = resolve_registered_animal_audio_artifacts(
        artifacts=fixture["artifacts"],
        root_paths=fixture["root_paths"],
        action="Walking",
        tag=fixture["tag"],
    )
    rir = fixture["copied"]["active_frame_rir"]
    rir.write_bytes(b"replaced after resolver")

    with pytest.raises(ValueError, match="bytes changed"):
        validate_animal_audio_evidence(
            spec=fixture["spec"],
            schedule=fixture["schedule"],
            audio_path=Path("missing.wav"),
            expected_tags={fixture["tag"]},
            render_manifest=fixture["manifest"],
            artifact_overrides=overrides,
        )


@pytest.mark.parametrize("adversary", ["colored_noise", "reverse_20ms"])
def test_validate_audio_rejects_self_consistent_unrelated_wet_audio(
    tmp_path,
    adversary,
):
    action, tag, _schedule = _strict_audio_action(tmp_path)
    output = Path(action["output_dir"])
    solo = output / f"binaural_{tag}_binaural.wav"
    final = output / "binaural.wav"
    dry, rate = sf.read(
        str(output / f"binaural_{tag}_scheduled_dry.wav"),
        dtype="float32",
    )
    if adversary == "colored_noise":
        rng = np.random.default_rng(9127)
        noise = rng.standard_normal(len(dry)).astype(np.float32)
        noise = np.convolve(
            noise,
            np.ones(3, dtype=np.float32) / 3.0,
            mode="same",
        ).astype(np.float32)
        block = round(0.020 * rate)
        for start in range(0, len(dry), block):
            end = min(len(dry), start + block)
            target_rms = float(np.sqrt(np.mean(dry[start:end] ** 2)))
            noise_rms = float(np.sqrt(np.mean(noise[start:end] ** 2)))
            noise[start:end] *= target_rms / (noise_rms + 1.0e-12)
        fake_raw = np.column_stack((noise, noise * 0.65)).astype(np.float32)
        fake_raw *= 0.8 / float(np.max(np.abs(fake_raw)))
        fake = fake_raw * (0.9 / 0.8)
    else:
        fake, rate = sf.read(str(solo), dtype="float32")
        block = round(0.020 * rate)
        for start in range(0, len(fake), block):
            fake[start : start + block] = fake[start : start + block][::-1]
    sf.write(str(solo), fake, rate, subtype="PCM_16")
    final.write_bytes(solo.read_bytes())

    manifest_path = output / "binaural_audio_render_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["per_source_outputs"][tag]["binaural"] = _descriptor(solo)
    manifest["per_source_outputs"][tag]["pre_normalization_peak"] = 0.8
    manifest["output_wav"] = _descriptor(final)
    manifest["mix_pre_normalization_peak"] = 0.8
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(schema.ContractError, match="source contract failed"):
        _validate_audio(action, tag)


def test_validate_audio_rejects_rehashed_rir_and_wrong_acoustic_scene(tmp_path):
    action, tag, _schedule = _strict_audio_action(tmp_path)
    output = Path(action["output_dir"])
    manifest_path = output / "binaural_audio_render_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rir_record = manifest["per_source_outputs"][tag]["active_frame_rir"]
    rir_path = Path(rir_record["path"])
    with np.load(rir_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    arrays["rir_left"][0] *= np.float32(0.5)
    with rir_path.open("wb") as stream:
        np.savez(stream, **arrays)
    manifest["per_source_outputs"][tag]["active_frame_rir"] = _descriptor(rir_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(schema.ContractError, match="source contract failed"):
        _validate_audio(action, tag)

    action, tag, _schedule = _strict_audio_action(tmp_path / "wrong_derived")
    output = Path(action["output_dir"])
    wrong_stage = output / ".wrong_derived"
    wrong_stage.mkdir()
    wrong_derived = wrong_stage / "rlr_materials.json"
    wrong_derived.write_text("{}\n")
    wrong_derived.chmod(0o400)
    wrong_stage.chmod(0o500)
    manifest_path = output / "binaural_audio_render_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["derived_rlr_materials"] = _descriptor(wrong_derived)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(schema.ContractError, match="source contract failed"):
        _validate_audio(action, tag)

    action, tag, _schedule = _strict_audio_action(tmp_path / "wrong_scene")
    output = Path(action["output_dir"])
    wrong_mesh = output / "wrong_but_self_consistent.glb"
    wrong_mesh.write_bytes(b"glb")
    manifest_path = output / "binaural_audio_render_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["acoustic_mesh"] = _descriptor(wrong_mesh)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(schema.ContractError, match="source contract failed"):
        _validate_audio(action, tag)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda source: source.__setitem__("source_sha256", "0" * 64),
        lambda source: source.__setitem__("audio_lookup", "cat_meow"),
        lambda source: source.__setitem__(
            "item_level_license_status",
            "verified",
        ),
        lambda source: source.__setitem__(
            "formal_registration_authorized",
            True,
        ),
        lambda source: source.pop("source_contract"),
        lambda source: source.__setitem__(
            "spatialization_status",
            "pre_spatialized_hrtf_binaural",
        ),
    ],
)
def test_validate_audio_rejects_weak_or_cross_bound_schedule(
    tmp_path,
    mutate,
):
    action, tag, schedule_path = _strict_audio_action(tmp_path)
    schedule = json.loads(schedule_path.read_text())
    mutate(schedule["sources"][tag])
    schedule_path.write_text(json.dumps(schedule))

    with pytest.raises(schema.ContractError, match="source contract failed"):
        _validate_audio(action, tag)


def test_validate_audio_rejects_zero_or_near_silent_binaural_output(tmp_path):
    action, tag, _schedule = _strict_audio_action(tmp_path)
    _write_binaural(
        Path(action["output_dir"]) / "binaural.wav",
        near_silent=True,
    )

    with pytest.raises(schema.ContractError, match="source contract failed"):
        _validate_audio(action, tag)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda source: source.__setitem__("render_sample_rate_hz", 12345),
        lambda source: source.__setitem__("events", []),
        lambda source: source.__setitem__("mode", "unknown_policy"),
        lambda source: source.__setitem__("minimum_silence_gap_s", float("nan")),
    ],
)
def test_validate_audio_rejects_forged_event_schedule_even_with_rehashed_manifest(
    tmp_path,
    mutate,
):
    action, tag, schedule_path = _strict_audio_action(tmp_path)
    schedule = json.loads(schedule_path.read_text())
    mutate(schedule["sources"][tag])
    schedule_path.write_text(json.dumps(schedule))
    manifest_path = (
        Path(action["output_dir"]) / "binaural_audio_render_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["source_schedule"] = _descriptor(schedule_path)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(schema.ContractError, match="source contract failed"):
        _validate_audio(action, tag)


def test_validate_audio_rejects_missing_manifest_or_missing_strict_spec(tmp_path):
    missing_action, missing_tag, _schedule = _strict_audio_action(
        tmp_path / "missing"
    )
    (
        Path(missing_action["output_dir"])
        / "binaural_audio_render_manifest.json"
    ).unlink()
    with pytest.raises(schema.ContractError, match="source contract failed"):
        _validate_audio(missing_action, missing_tag)

    weak_action, weak_tag, _schedule = _strict_audio_action(tmp_path / "weak")
    spec_path = Path(weak_action["spec"])
    spec = json.loads(spec_path.read_text())
    spec["sources"][0].pop("strict_audio")
    spec_path.write_text(json.dumps(spec))
    manifest_path = (
        Path(weak_action["output_dir"])
        / "binaural_audio_render_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["spec"] = _descriptor(spec_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(schema.ContractError, match="source contract failed"):
        _validate_audio(weak_action, weak_tag)


def test_validate_audio_rejects_left_right_swapped_per_source_ild(tmp_path):
    action, tag, _schedule = _strict_audio_action(tmp_path)
    output = Path(action["output_dir"])
    solo = output / f"binaural_{tag}_binaural.wav"
    with wave.open(str(solo), "rb") as stream:
        params = stream.getparams()
        samples = np.frombuffer(
            stream.readframes(stream.getnframes()),
            dtype="<i2",
        ).reshape(-1, 2)
    with wave.open(str(solo), "wb") as stream:
        stream.setparams(params)
        stream.writeframes(samples[:, ::-1].astype("<i2", copy=False).tobytes())
    manifest_path = output / "binaural_audio_render_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["per_source_outputs"][tag]["binaural"] = _descriptor(solo)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(schema.ContractError, match="source contract failed"):
        _validate_audio(action, tag)


def test_validate_audio_rejects_manifest_azimuth_not_derived_from_spec(
    tmp_path,
):
    action, tag, _schedule = _strict_audio_action(tmp_path)
    output = Path(action["output_dir"])
    manifest_path = output / "binaural_audio_render_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["per_source_outputs"][tag][
        "mic_local_azimuth_deg_per_frame"
    ] = [30.0] * 270
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(schema.ContractError, match="source contract failed"):
        _validate_audio(action, tag)


def test_validate_audio_rejects_event_shaped_tone_unrelated_to_pinned_dry(
    tmp_path,
):
    action, tag, schedule_path = _strict_audio_action(tmp_path)
    output = Path(action["output_dir"])
    schedule = json.loads(schedule_path.read_text())["sources"][tag]
    frame_count = int(
        round(
            schedule["render_sample_rate_hz"]
            * schedule["target_duration_s"]
        )
    )
    left = np.zeros(frame_count, dtype=np.float32)
    for event in schedule["events"]:
        start = event["start_sample"]
        end = event["end_sample"]
        phase = np.arange(end - start, dtype=np.float64) / 16000.0
        left[start:end] = (
            0.9 * np.sin(2.0 * math.pi * 2500.0 * phase)
        ).astype(np.float32)
    stereo = np.column_stack((left, left * 0.65)).astype(np.float32)
    mixed = output / "binaural.wav"
    solo = output / f"binaural_{tag}_binaural.wav"
    sf.write(str(mixed), stereo, 16000, subtype="PCM_16")
    solo.write_bytes(mixed.read_bytes())

    manifest_path = output / "binaural_audio_render_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_wav"] = _descriptor(mixed)
    manifest["per_source_outputs"][tag]["binaural"] = _descriptor(solo)
    manifest["per_source_outputs"][tag]["pre_normalization_peak"] = 1.0
    manifest["mix_pre_normalization_peak"] = 1.0
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(schema.ContractError, match="source contract failed"):
        _validate_audio(action, tag)
