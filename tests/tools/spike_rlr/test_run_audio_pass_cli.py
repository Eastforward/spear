"""Verify CLI args exist on the audio pass + topdown render scripts.

These are cheap smoke tests — they only run `--help` and check that the
new/existing CLI flags are advertised. Full end-to-end audio rendering
is verified in Task 9's live runs.
"""
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _strict_pinned_source_spec(lookup, *, include_lookup=True):
    from tools.spike_rlr.animal_audio import pinned_animal_audio_contract

    contract = pinned_animal_audio_contract(lookup)
    source = {
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
    if include_lookup:
        source["audio_lookup"] = lookup
    return source


def test_audio_pass_help_shows_spec_and_mesh_args():
    r = subprocess.run(
        ["/data/jzy/miniconda3/envs/ss2/bin/python",
         str(REPO / "tools" / "spike_rlr" / "run_audio_pass_rlr.py"),
         "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"help failed:\n{r.stderr}"
    assert "--spec" in r.stdout
    assert "--mesh" in r.stdout
    assert "--materials" in r.stdout


def test_topdown_help_shows_spec_arg():
    r = subprocess.run(
        ["/data/jzy/miniconda3/envs/spear-env/bin/python",
         str(REPO / "tools" / "spike_rlr" / "render_topdown_2d.py"),
         "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"help failed:\n{r.stderr}"
    assert "--spec" in r.stdout


def test_audio_pass_load_scene_dispatch_shoebox():
    """The dispatcher (imported directly) should return shoebox composer for v2."""
    import sys
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from run_audio_pass_rlr import _load_scene_and_scene_two_dogs
    fn = _load_scene_and_scene_two_dogs(REPO / "data" / "shoebox_v2_spec.json")
    assert fn.__name__ == "compose_two_dog_scene_v2"


def test_audio_pass_load_scene_dispatch_apartment():
    """The dispatcher should return apartment composer for apartment_v1."""
    import sys
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from run_audio_pass_rlr import _load_scene_and_scene_two_dogs
    fn = _load_scene_and_scene_two_dogs(REPO / "data" / "apartment_v1_spec.json")
    assert fn.__name__ == "compose_two_dog_scene_apartment"


def test_audio_pass_has_explicit_sources_for_review_animals():
    text = (REPO / "tools" / "spike_rlr" / "run_audio_pass_rlr.py").read_text()

    assert '"dog_beagle_v2":' in text
    assert '"cat_british_shorthair_v2":' in text
    assert "Growling and Barking Dog.wav" in text
    assert "Cat Meowing.wav" in text
    assert "Barking Aldi Dog_358.wav" not in text
    assert "Cat Meowing_293.wav" not in text


def test_audio_agent_yaw_tracks_scene_mic_yaw():
    import sys
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from run_audio_pass_rlr import _habitat_agent_yaw_deg_for_scene_yaw_deg

    assert _habitat_agent_yaw_deg_for_scene_yaw_deg(90.0) == 0.0
    assert _habitat_agent_yaw_deg_for_scene_yaw_deg(0.0) == 270.0
    assert _habitat_agent_yaw_deg_for_scene_yaw_deg(180.0) == 90.0
    assert _habitat_agent_yaw_deg_for_scene_yaw_deg(270.0) == 180.0


def test_audio_scene_to_habitat_matches_loaded_glb_axes():
    import sys
    import numpy as np
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from run_audio_pass_rlr import _habitat_from_scene

    np.testing.assert_allclose(
        _habitat_from_scene([1.0, 2.0, 3.0]),
        [1.0, 3.0, -2.0],
    )


def test_audio_sensor_position_is_explicitly_zeroed():
    text = (REPO / "tools" / "spike_rlr" / "run_audio_pass_rlr.py").read_text()

    assert "audio_spec.position = [0.0, 0.0, 0.0]" in text


def test_native_binaural_channel_order_is_not_swapped_after_coord_fix():
    import sys
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from run_audio_pass_rlr import RLR_NATIVE_BINAURAL_CHANNEL_ORDER

    assert RLR_NATIVE_BINAURAL_CHANNEL_ORDER == (0, 1)


def test_renderer_contract_pins_native_rlr_binary_fingerprint():
    import sys

    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from acoustic_scene_contract import (
        approved_rlr_renderer_contract,
        validate_approved_rlr_runtime_artifact,
    )

    contract = approved_rlr_renderer_contract(
        sample_rate_hz=16000,
        quality_mode="low",
    )
    native = contract["runtime_artifacts"][
        "rlr_audio_propagation_shared_library"
    ]
    assert native["basename"] == "libRLRAudioPropagation.so"
    assert len(native["sha256"]) == 64
    assert native["size_bytes"] == 7327344
    with pytest.raises(ValueError, match="not approved"):
        validate_approved_rlr_runtime_artifact(
            artifact_name="rlr_audio_propagation_shared_library",
            path="/fixture/libRLRAudioPropagation.so",
            payload=b"forged",
        )


def test_audio_render_manifest_is_published_as_final_hash_closure(tmp_path):
    import sys

    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    import run_audio_pass_rlr as rlr
    from acoustic_scene_contract import approved_rlr_renderer_contract

    def write(name, payload):
        path = tmp_path / name
        path.write_bytes(payload)
        return path, rlr._authenticated_descriptor(path)

    spec, spec_descriptor = write("spec.json", b"{}")
    mesh, mesh_descriptor = write("mesh.glb", b"glb")
    materials, materials_descriptor = write("materials.json", b"{}")
    derived, derived_descriptor = write("binaural_rlr_materials.json", b"{}")
    schedule, _ = write(
        "binaural_source_schedule.json",
        b'{"schema":"rlr_audio_source_schedules_v1","sources":{}}',
    )
    output, _ = write("binaural.wav", b"wav")

    manifest_path = rlr._write_audio_render_manifest(
        out_wav_path=output,
        schedule_path=schedule,
        spec_descriptor=spec_descriptor,
        mesh_descriptor=mesh_descriptor,
        materials_descriptor=materials_descriptor,
        derived_materials_descriptor=derived_descriptor,
        acoustic_scene_contract={
            "schema": "avengine_acoustic_scene_contract_v1",
        },
        renderer_contract=approved_rlr_renderer_contract(
            sample_rate_hz=16000,
            quality_mode="high",
        ),
        channel_layout="binaural_native",
        sample_rate_hz=16000,
        duration_s=3.0,
        n_frames=45,
        fps=15,
        quality_mode="high",
        indirect_ray_count=500,
        source_tags=[],
        per_source_outputs={},
        mix_pre_normalization_peak=1.0,
    )

    payload = __import__("json").loads(manifest_path.read_text())
    assert manifest_path.name == "binaural_audio_render_manifest.json"
    assert payload["schema"] == "rlr_audio_render_manifest_v3"
    assert payload["spec"] == spec_descriptor
    assert payload["acoustic_mesh"] == mesh_descriptor
    assert payload["acoustic_materials"] == materials_descriptor
    assert payload["derived_rlr_materials"] == derived_descriptor
    assert payload["acoustic_scene_contract"] == {
        "schema": "avengine_acoustic_scene_contract_v1",
    }
    assert payload["renderer_contract"] == approved_rlr_renderer_contract(
        sample_rate_hz=16000,
        quality_mode="high",
    )
    assert payload["source_schedule"]["path"] == str(schedule.resolve())
    assert payload["output_wav"]["path"] == str(output.resolve())
    assert payload["behavior_gates"]["binaural_spatial_ild"] == (
        "validate_on_readback"
    )
    assert payload["behavior_gates"]["active_frame_rir_replay"] == (
        "validate_on_readback"
    )
    assert payload["technical_render_status"] == "passed"
    assert payload["formal_registration_authorized"] is False
    assert spec.is_file() and mesh.is_file() and materials.is_file()
    assert derived.is_file()


def test_compute_binaural_consumes_immutable_staged_derived_materials(
    tmp_path,
    monkeypatch,
):
    import json
    import stat
    import sys
    from types import SimpleNamespace

    import numpy as np

    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    import run_audio_pass_rlr as rlr
    from rlr_materials import build_rlr_materials_payload

    tag = "dog_immutable_material_canary"
    trajectory = [[1.0, 1.0, 0.4]] * 3
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "spec_version": "apartment_v1",
                "audio_config": {"sample_rate_hz": 12, "duration_s": 1.0},
                "render_config": {"n_frames": 3, "fps": 3.0},
                "mic": {"pos_m": [0.0, 0.0, 1.2], "yaw_deg": 0.0},
                "sources": [
                    {
                        "tag": tag,
                        "asset_class": "animal",
                        "species": "dog",
                        "audio_lookup": "dog_bark",
                        "trajectory_m": trajectory,
                    }
                ],
            }
        )
    )
    mesh = REPO / "tmp/spike_rlr/apartment_v1_mesh.glb"
    materials = REPO / "tmp/spike_rlr/apartment_v1_materials.json"
    out = tmp_path / "binaural.wav"
    captured = {}

    class FakeSensor:
        def setAudioSourceTransform(self, _position):
            return None

    class FakeSim:
        def get_sensor_observations(self):
            return {
                "audio_sensor": np.asarray(
                    [[1.0, 0.0], [0.65, 0.0]],
                    dtype=np.float32,
                )
            }

    def fake_build(mesh_path, materials_path, **_kwargs):
        captured["mesh"] = Path(mesh_path)
        captured["materials"] = Path(materials_path)
        captured["uses_pinned_proc_root"] = str(materials_path).startswith(
            "/proc/self/fd/"
        )
        captured["materials_mode"] = stat.S_IMODE(
            Path(materials_path).stat().st_mode
        )
        captured["parent_mode"] = stat.S_IMODE(
            Path(materials_path).parent.stat().st_mode
        )
        approved_payload = Path(materials_path).read_bytes()
        original = Path(materials_path).parent.resolve()
        backup = original.with_name(f"{original.name}.rename_attack_backup")
        original.rename(backup)
        original.mkdir()
        attacker_materials = original / "rlr_materials.json"
        attacker_materials.write_bytes(b'{"attacker":true}')
        try:
            captured["payload"] = Path(materials_path).read_bytes()
        finally:
            attacker_materials.unlink()
            original.rmdir()
            backup.rename(original)
        assert captured["payload"] == approved_payload
        return FakeSim(), FakeSensor()

    scene = SimpleNamespace(
        animals=[SimpleNamespace(tag=tag, trajectory_m=trajectory)]
    )

    def fake_load(
        _tag,
        sample_rate,
        duration_s,
        *,
        source_spec,
        schedule_metadata_out,
        **_kwargs,
    ):
        schedule_metadata_out.update(
            {"schema": "fixture_schedule", "tag": source_spec["tag"]}
        )
        return np.asarray(
            [0.0, 0.2, -0.2, 0.0, 0.1, -0.1, 0.0, 0.3, -0.3, 0.0, 0.0, 0.0],
            dtype=np.float32,
        )

    monkeypatch.setattr(rlr, "build_rlr_sim", fake_build)
    monkeypatch.setattr(rlr, "_set_agent_pose", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        rlr,
        "_load_scene_and_scene_two_dogs",
        lambda _path: lambda _staged_path: scene,
    )
    monkeypatch.setattr(rlr, "_load_dry_source", fake_load)

    manifest_path = rlr.compute_binaural(
        spec_path,
        mesh,
        materials,
        out,
        quality_mode="low",
        verbose=False,
    )
    manifest = json.loads(manifest_path.read_text())

    assert captured["uses_pinned_proc_root"] is True
    assert captured["materials"].name == "rlr_materials.json"
    assert captured["materials_mode"] == 0o400
    assert captured["parent_mode"] == 0o500
    assert captured["payload"] == build_rlr_materials_payload(
        json.loads(materials.read_text())
    )
    rir = Path(
        manifest["per_source_outputs"][tag]["active_frame_rir"]["path"]
    )
    assert stat.S_IMODE(rir.stat().st_mode) == 0o400
    assert stat.S_IMODE(rir.parent.stat().st_mode) == 0o500


def test_rir_replay_reconstructs_implicit_composed_trajectory(
    tmp_path,
):
    import hashlib
    import json
    import sys

    import numpy as np
    import soundfile as sf

    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    import animal_audio
    import run_audio_pass_rlr as rlr
    from active_frame_rir_evidence import (
        active_frame_indices,
        serialize_active_frame_rir_evidence,
    )
    from source_trajectory import acoustic_trajectory

    tag = "dog_implicit_trajectory_canary"
    source_spec = {
        "tag": tag,
        "asset_class": "animal",
        "species": "dog",
        "audio_lookup": "dog_bark",
        "start_pos_m": [1.0, 1.0, 0.0],
        "end_pos_m": [1.0, 1.0, 0.0],
        "motion": "linear_uniform_raw",
        "walking_forward_yaw_offset_deg": 0.0,
        "audio_source_height_offset_m": 0.45,
        "strict_audio": True,
    }
    animal_audio.bind_pinned_animal_audio_contract(source_spec)
    spec = json.loads((REPO / "data/apartment_v1_spec.json").read_text())
    spec.update(
        {
            "mic": {"pos_m": [0.0, 0.0, 1.2], "yaw_deg": 0.0},
            "audio_config": {
                "sample_rate_hz": 16000,
                "duration_s": 3.0,
            },
            "render_config": {"n_frames": 45, "fps": 15.0},
            "sources": [source_spec],
        }
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    schedule_source = {}
    dry = rlr._load_dry_source(
        tag,
        16000,
        3.0,
        source_spec=source_spec,
        schedule_metadata_out=schedule_source,
    )
    placement = rlr._load_scene_and_scene_two_dogs(spec_path)(spec_path).animals[0]
    acoustic = acoustic_trajectory(placement.trajectory_m, source_spec)
    indices = active_frame_indices(
        dry,
        n_frames=45,
        samples_per_frame=round(16000 / 15),
    )
    rir_path = tmp_path / "implicit_active_rir.npz"
    rir_path.write_bytes(
        serialize_active_frame_rir_evidence(
            source_tag=tag,
            frame_indices=indices,
            rirs=[
                np.asarray([[1.0], [0.65]], dtype=np.float32)
                for _ in indices
            ],
            source_positions_scene_m=acoustic[indices],
            mic_position_scene_m=np.asarray([0.0, 0.0, 1.2]),
            mic_yaw_deg=0.0,
            sample_rate_hz=16000,
            n_samples_total=len(dry),
            n_frames=45,
            fps=15.0,
            samples_per_frame=round(16000 / 15),
        )
    )
    raw = np.stack((dry, dry * 0.65), axis=0).astype(np.float32)
    peak = float(np.max(np.abs(raw)))
    solo_path = tmp_path / "solo.wav"
    sf.write(
        str(solo_path),
        (raw * (0.9 / peak)).T,
        16000,
        subtype="PCM_16",
    )

    def descriptor(path):
        payload = path.read_bytes()
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    replayed, result = animal_audio._validate_active_frame_rir_source_replay(
        tag=tag,
        spec=spec,
        spec_path=spec_path,
        source_spec=source_spec,
        schedule_source=schedule_source,
        rir_descriptor=descriptor(rir_path),
        solo_path=solo_path,
        solo_descriptor=descriptor(solo_path),
        reported_peak=peak,
        sample_rate_hz=16000,
        duration_s=3.0,
        n_frames=45,
        fps=15.0,
    )

    np.testing.assert_array_equal(replayed, raw)
    assert result["status"] == "passed"


def test_audio_pass_reads_mic_yaw_from_spec_with_camera_fallback():
    import sys
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from run_audio_pass_rlr import _mic_yaw_deg_from_spec

    assert _mic_yaw_deg_from_spec({
        "mic": {"yaw_deg": 34.5},
        "camera_configs": [{"yaw_deg": 90.0}],
    }) == 34.5
    assert _mic_yaw_deg_from_spec({
        "mic": {},
        "camera_configs": [{"yaw_deg": 90.0}],
    }) == 90.0
    assert _mic_yaw_deg_from_spec({"mic": {}}) == 90.0


def test_load_dry_source_returns_silence_for_muted_source(tmp_path):
    import sys
    import numpy as np
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from animal_audio import bind_animal_silence_contract
    from run_audio_pass_rlr import _load_dry_source

    registry_path = tmp_path / "registry.json"
    import_path = tmp_path / "import.json"
    registry_path.write_text("{}")
    import_path.write_text("{}")

    def gate_descriptor(path):
        payload = path.read_bytes()
        return {
            "path": str(path.resolve()),
            "sha256": __import__("hashlib").sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    source = {
        "tag": "stable_alpaca_test",
        "asset_id": "alpaca_test",
        "template_id": "alpaca_test",
        "asset_class": "animal",
        "species": "alpaca",
        "audio_lookup": "silent",
        "stable_animal_gate": {
            "schema": "stable_animal_apartment_gate_v1",
            "status": "approved_for_automated_research_candidate_apartment",
            "asset_id": "alpaca_test",
            "template_id": "alpaca_test",
            "tag": "stable_alpaca_test",
            "species": "alpaca",
            "source_sha256": "a" * 64,
            "template_registry": gate_descriptor(registry_path),
            "ue_import_result": gate_descriptor(import_path),
            "formal_dataset_registration_authorized": False,
        },
    }
    bind_animal_silence_contract(source)
    y = _load_dry_source(
        "stable_alpaca_test",
        sample_rate=16000,
        duration_s=0.25,
        source_spec=source,
    )

    assert y.shape == (4000,)
    assert np.allclose(y, 0.0)


def test_load_dry_source_rejects_dog_silence_bypass():
    import sys

    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from run_audio_pass_rlr import _load_dry_source

    with pytest.raises(RuntimeError, match="cannot bypass"):
        _load_dry_source(
            "dog_generated_candidate",
            sample_rate=16000,
            duration_s=0.25,
            source_spec={
                "asset_class": "animal",
                "species": "dog",
                "audio_lookup": "silent",
                "mute_audio": True,
                "strict_audio": True,
            },
        )


def test_load_dry_source_repeats_requested_audio_clip():
    import sys
    import numpy as np
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from run_audio_pass_rlr import _load_dry_source

    y = _load_dry_source(
        "dog_beagle_v2",
        sample_rate=16000,
        duration_s=1.2,
        source_spec={
            "audio_lookup": "dog_sharp_bark",
            "audio_clip_start_s": 2.0,
            "audio_clip_duration_s": 0.2,
            "audio_repeat_interval_s": 0.5,
        },
    )

    first = y[0:3200]
    second = y[8000:11200]
    gap = y[3200:8000]
    assert np.max(np.abs(first)) > 0.01
    np.testing.assert_allclose(first, second, atol=1e-6)
    assert np.max(np.abs(gap)) < 1e-6


@pytest.mark.parametrize(
    ("tag", "lookup", "expected_sha256"),
    [
        (
            "dog_pembroke_welsh_corgi_candidate",
            "dog_bark",
            "5481218ef268b4df98b03e52c48a4973852d60ea5cae9cbf42d0f203a6b9a505",
        ),
        (
            "cat_british_shorthair_candidate",
            "cat_meow",
            "aa8736bc58a4cd8a35d8911e2cfa22fb2e93fa899b67b1c04a927322c074df3d",
        ),
    ],
)
def test_pinned_animal_source_is_resampled_scheduled_and_hash_bound(
    tag, lookup, expected_sha256
):
    import sys
    import numpy as np

    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from run_audio_pass_rlr import _load_dry_source

    schedule = {}
    y = _load_dry_source(
        tag,
        sample_rate=16000,
        duration_s=1.0,
        source_spec=_strict_pinned_source_spec(lookup),
        schedule_metadata_out=schedule,
    )

    assert y.shape == (16000,)
    assert np.max(np.abs(y)) > 0.01
    assert schedule["source_sha256"] == expected_sha256
    assert schedule["source_original_sample_rate_hz"] == 44100
    assert schedule["source_original_frame_count"] == 441000
    assert schedule["source_original_channels"] == 1
    assert schedule["render_sample_rate_hz"] == 16000
    assert schedule["source_contract"]["sha256"] == expected_sha256
    assert schedule["source_contract"]["channels"] == 1
    assert schedule["dry_source_policy"] == (
        "mono_no_hrtf_no_pre_spatialization"
    )
    assert schedule["spatialization_status"] == (
        "mono_unspatialized_dry_source"
    )
    assert schedule["objective_audio_content_qa_status"] == {
        "dog_bark": "clotho_five_caption_consensus_animal_only_pending_listening",
        "cat_meow": "pending_background_contamination_review",
    }[lookup]
    assert schedule["item_level_license_status"] == "missing"
    assert schedule["formal_registration_authorized"] is False


def test_pinned_animal_source_failure_never_falls_back_to_synthetic(
    monkeypatch,
):
    import sys

    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    import run_audio_pass_rlr as rlr

    def fail_resolve(*args, **kwargs):
        raise ValueError("stale pinned source")

    monkeypatch.setattr(rlr, "resolve_animal_audio_path", fail_resolve)

    with pytest.raises(RuntimeError, match="strict audio source"):
        rlr._load_dry_source(
            "dog_pembroke_welsh_corgi_candidate",
            sample_rate=16000,
            duration_s=0.25,
            source_spec={"audio_lookup": "dog_bark"},
        )


def test_unknown_explicit_animal_lookup_never_falls_back_to_synthetic():
    import sys

    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    import run_audio_pass_rlr as rlr

    with pytest.raises(RuntimeError, match="unknown animal audio_lookup"):
        rlr._load_dry_source(
            "cat_british_shorthair_candidate",
            sample_rate=16000,
            duration_s=0.25,
            source_spec={"audio_lookup": "cat_meow_stale_typo"},
        )


@pytest.mark.parametrize(
    "sentinel",
    ["__pink_noise__", "__click_train__", "__synth_piano_scale__"],
)
def test_strict_dog_cat_sources_reject_every_synthetic_sentinel(sentinel):
    import sys

    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    import run_audio_pass_rlr as rlr

    with pytest.raises(RuntimeError, match="synthetic sentinel"):
        rlr._load_dry_source(
            "cat_british_shorthair_candidate",
            sample_rate=16000,
            duration_s=0.25,
            source_spec={"audio_path": sentinel},
        )


def test_cat_without_lookup_uses_pinned_default_before_legacy_override(
    monkeypatch,
):
    import sys

    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    import run_audio_pass_rlr as rlr

    monkeypatch.setitem(
        rlr._TAG_AUDIO_OVERRIDES,
        "cat_british_shorthair_v2",
        "__pink_noise__",
    )
    schedule = {}
    y = rlr._load_dry_source(
        "cat_british_shorthair_v2",
        sample_rate=16000,
        duration_s=0.25,
        source_spec=_strict_pinned_source_spec(
            "cat_meow",
            include_lookup=False,
        ),
        schedule_metadata_out=schedule,
    )

    assert y.shape == (4000,)
    assert schedule["audio_lookup"] == "cat_meow"
    assert schedule["source_sha256"] == (
        "aa8736bc58a4cd8a35d8911e2cfa22fb2e93fa899b67b1c04a927322c074df3d"
    )


def test_runner_rejects_stale_propagated_pinned_contract():
    import copy
    import sys

    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    import run_audio_pass_rlr as rlr

    contract = rlr.pinned_animal_audio_contract("dog_bark")
    stale = copy.deepcopy(contract)
    stale["duration_s"] = 5.0
    source_spec = {
        "audio_lookup": "dog_bark",
        "audio_path": contract["path"],
        "audio_contract": stale,
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
    with pytest.raises(RuntimeError, match="propagated pinned audio"):
        rlr._load_dry_source(
            "dog_pembroke_welsh_corgi_candidate",
            sample_rate=16000,
            duration_s=0.25,
            source_spec=source_spec,
        )


def test_hash_and_decode_use_the_same_immutable_file_snapshot(
    tmp_path,
    monkeypatch,
):
    import hashlib
    import io
    import sys
    import wave

    import numpy as np

    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    import run_audio_pass_rlr as rlr

    def write_wav(path, value):
        samples = np.full(1600, value, dtype="<i2")
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(samples.tobytes())

    source = tmp_path / "source.wav"
    replacement = tmp_path / "replacement.wav"
    write_wav(source, 4000)
    write_wav(replacement, -12000)
    original_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    replacement_bytes = replacement.read_bytes()
    real_read = rlr.sf.read
    decoded_snapshot_sha = []

    def replace_path_then_decode(file, *args, **kwargs):
        assert isinstance(file, io.BytesIO)
        decoded_snapshot_sha.append(hashlib.sha256(file.getvalue()).hexdigest())
        source.write_bytes(replacement_bytes)
        return real_read(file, *args, **kwargs)

    monkeypatch.setattr(rlr.sf, "read", replace_path_then_decode)
    y = rlr._load_dry_source(
        "dog_golden",
        sample_rate=16000,
        duration_s=0.1,
        source_spec={
            "audio_lookup": "dog_growl",
            "audio_path": str(source),
            "audio_sha256": original_sha,
            "adaptive_repeat_short_calls": False,
        },
    )

    assert decoded_snapshot_sha == [original_sha]
    assert hashlib.sha256(source.read_bytes()).hexdigest() != original_sha
    assert np.max(np.abs(y)) > 0.1


def test_runner_rejects_pre_spatialized_stereo_file():
    import sys

    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    import run_audio_pass_rlr as rlr

    with pytest.raises(RuntimeError, match="does not match pinned"):
        rlr._load_dry_source(
            "dog_pembroke_welsh_corgi_candidate",
            sample_rate=16000,
            duration_s=0.25,
            source_spec={
                "audio_lookup": "dog_bark",
                "audio_path": (
                    "/data/datasets/omniaudio/train-data-az-360-large/"
                    "Barking Aldi Dog_358.wav"
                ),
            },
        )


def test_strict_unpinned_animal_never_downmixes_stereo_to_fake_dry_source():
    import sys

    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    import run_audio_pass_rlr as rlr

    with pytest.raises(RuntimeError, match="must be mono"):
        rlr._load_dry_source(
            "stable_horse_bay_native",
            sample_rate=16000,
            duration_s=0.25,
            source_spec={
                "audio_lookup": "horse_neigh",
                "strict_audio": True,
            },
        )


def test_strict_dog_rejects_unpinned_stereo_override_too():
    import sys

    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    import run_audio_pass_rlr as rlr

    with pytest.raises(RuntimeError, match="must be mono"):
        rlr._load_dry_source(
            "dog_golden",
            sample_rate=16000,
            duration_s=0.25,
            source_spec={
                "audio_lookup": "dog_growl",
                "audio_path": (
                    "/data/datasets/omniaudio/train-data-az-360-large/"
                    "Dog Growls_184.wav"
                ),
            },
        )


def test_package_runner_import_uses_relative_dependencies():
    import tools.spike_rlr.run_audio_pass_rlr as packaged

    assert packaged.pinned_audio_lookup_for_tag("dog_test") == "dog_bark"


def test_topdown_load_scene_dispatch_apartment():
    """render_topdown_2d._load_scene should return an apartment SceneSpec."""
    import sys
    sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
    from render_topdown_2d import _load_scene
    sc = _load_scene(REPO / "data" / "apartment_v1_spec.json")
    tags = {a.tag for a in sc.animals}
    assert tags == {"dog_golden", "dog_beagle_v2"}
