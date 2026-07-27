import hashlib
import json
from pathlib import Path
import wave

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "build_stable_animal_ofat_apartment_review.py"


def source():
    return SCRIPT.read_text(encoding="utf-8")


def test_review_requires_complete_walk_idle_batch_and_per_instance_registry():
    text = source()
    assert 'STATUS_SCHEMA = "stable_animal_apartment_render_status_v1"' in text
    assert 'status.get("passed_job_count") != spec["clip_count"]' in text
    assert 'status.get("failed_job_count") != 0' in text
    assert 'set(registry.get("clips", {})) != set(ACTIONS)' in text
    assert 'planned["spec_evidence"]["sha256"]' in text


def test_review_rehashes_all_ue_video_audio_and_runtime_artifacts():
    text = source()
    for field in (
        "actor_visual_metadata",
        "apartment_video",
        "topdown_review_video",
        "annotated_review_video",
        "binaural_audio",
        "audio_evidence",
        "binaural_source_schedule",
        "binaural_audio_render_manifest",
    ):
        assert f'"{field}"' in text
    assert "verify_descriptor" in text
    assert "sha256_file(path)" in text
    assert "validate_animal_audio_evidence" in text
    assert "validate_animal_silence_contract" in text
    assert "_validate_zero_pcm16_stereo" in text
    assert '"intentional_silence"' in text
    assert "audio_render_manifest = load_authenticated_json" in text
    assert 'expected_sha256=artifacts["spec"]["sha256"]' in text
    assert "expected_size_bytes=artifacts[" in text
    assert '"all_audio_source_contracts_and_waveforms_passed": True' in text


def test_review_preserves_absolute_paths_and_browser_urls_without_promotion():
    text = source()
    assert "resolved.relative_to(AVENGINE_ROOT.resolve())" in text
    assert '"formal_dataset_registration_authorized": False' in text
    assert '"human_visual_review": "pending"' in text
    assert 'output_manifest.open("x"' in text
    assert 'output_html.open("x"' in text


def _descriptor(path):
    payload = Path(path).read_bytes()
    return {
        "path": str(Path(path).resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _silent_clip_fixture(tmp_path):
    from tools.spike_rlr.animal_audio import bind_animal_silence_contract

    tag = "stable_alpaca_ofat"
    registry = tmp_path / "template_registry.json"
    imported = tmp_path / "ue_import_result.json"
    registry.write_text('{"status":"fixture"}')
    imported.write_text('{"status":"fixture"}')
    source = {
        "tag": tag,
        "asset_id": "alpaca_ofat",
        "template_id": "alpaca_ofat",
        "asset_class": "animal",
        "species": "alpaca",
        "audio_lookup": "silent",
        "stable_animal_gate": {
            "schema": "stable_animal_apartment_gate_v1",
            "status": "approved_for_automated_research_candidate_apartment",
            "asset_id": "alpaca_ofat",
            "template_id": "alpaca_ofat",
            "tag": tag,
            "species": "alpaca",
            "source_sha256": "a" * 64,
            "template_registry": _descriptor(registry),
            "ue_import_result": _descriptor(imported),
            "formal_dataset_registration_authorized": False,
        },
    }
    bind_animal_silence_contract(source)
    spec = {
        "audio_config": {"sample_rate_hz": 8000, "duration_s": 0.1},
        "render_config": {"duration_s": 0.1},
        "sources": [source],
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    audio_path = tmp_path / "binaural.wav"
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x00" * 800 * 4)
    audio_descriptor = _descriptor(audio_path)
    evidence_path = tmp_path / "audio_evidence.json"
    evidence = {
        "schema_version": "avengine_audio_artifact_evidence_v1",
        "status": "intentional_silence",
        "expects_audible_audio": False,
        "eligible_for_visual_review": True,
        "eligible_for_acoustic_training": False,
        "signal": {
            **audio_descriptor,
            "sample_rate_hz": 8000,
            "frame_count": 800,
            "channel_count": 2,
            "duration_s": 0.1,
            "peak_abs": 0.0,
            "is_effectively_silent": True,
        },
    }
    evidence_path.write_text(json.dumps(evidence))
    placeholder = _descriptor(spec_path)
    artifacts = {
        "spec": _descriptor(spec_path),
        "runtime_gate": placeholder,
        "actor_visual_metadata": placeholder,
        "apartment_video": placeholder,
        "topdown_review_video": placeholder,
        "annotated_review_video": placeholder,
        "binaural_audio": audio_descriptor,
        "audio_evidence": _descriptor(evidence_path),
    }
    return tag, spec, artifacts, audio_path, evidence_path


def test_silent_ofat_accepts_exact_contract_without_fake_rir(tmp_path):
    from tools.build_stable_animal_ofat_apartment_review import (
        _validate_clip_audio,
    )

    tag, spec, artifacts, _audio, _evidence = _silent_clip_fixture(tmp_path)

    assert _validate_clip_audio(
        artifacts=artifacts,
        final_spec=spec,
        tag=tag,
    ) == {"mode": "authenticated_intentional_silence"}


def test_silent_ofat_rejects_nonzero_wav_even_with_self_consistent_hashes(
    tmp_path,
):
    from tools.build_stable_animal_ofat_apartment_review import (
        _validate_clip_audio,
    )

    tag, spec, artifacts, audio_path, evidence_path = _silent_clip_fixture(
        tmp_path
    )
    with wave.open(str(audio_path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x01\x00\x00\x00" + b"\x00" * (799 * 4))
    artifacts["binaural_audio"] = _descriptor(audio_path)
    evidence = json.loads(evidence_path.read_text())
    evidence["signal"].update(artifacts["binaural_audio"])
    evidence_path.write_text(json.dumps(evidence))
    artifacts["audio_evidence"] = _descriptor(evidence_path)

    with pytest.raises(ValueError, match="exact PCM16 stereo zero"):
        _validate_clip_audio(
            artifacts=artifacts,
            final_spec=spec,
            tag=tag,
        )
