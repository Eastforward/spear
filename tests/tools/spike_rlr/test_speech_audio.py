import sys
from pathlib import Path

import numpy as np
import soundfile as sf


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def _write_wav(path: Path, duration_s: float = 0.5, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(duration_s * sample_rate), dtype=np.float32) / sample_rate
    y = 0.1 * np.sin(2 * np.pi * 220.0 * t)
    sf.write(path, y, sample_rate)


def test_pick_librispeech_sample_reads_transcript_and_metadata(tmp_path):
    from speech_audio import pick_speech_sample

    wav = tmp_path / "LibriTTS" / "train-clean-100" / "1234" / "5678" / "1234_5678_000001_000000.wav"
    _write_wav(wav, duration_s=1.25, sample_rate=24000)
    wav.with_suffix(".normalized.txt").write_text("Hello From A Test Speaker.\n", encoding="utf-8")

    sample = pick_speech_sample(
        root=tmp_path / "LibriTTS",
        rng=np.random.default_rng(0),
        duration_range_s=(0.5, 4.0),
    )

    assert sample.path == wav
    assert sample.category == "speech"
    assert sample.corpus == "LibriTTS"
    assert sample.speaker_id == "1234"
    assert sample.transcript == "Hello From A Test Speaker."
    assert sample.sample_rate == 24000
    assert 1.2 < sample.duration_s < 1.3


def test_pick_librispeech_sample_skips_out_of_range_audio(tmp_path):
    from speech_audio import pick_speech_sample

    short = tmp_path / "LibriTTS" / "train-clean-100" / "1000" / "1" / "1000_1_000001_000000.wav"
    good = tmp_path / "LibriTTS" / "train-clean-100" / "1001" / "1" / "1001_1_000001_000000.wav"
    _write_wav(short, duration_s=0.1)
    _write_wav(good, duration_s=1.0)

    sample = pick_speech_sample(
        root=tmp_path / "LibriTTS",
        rng=np.random.default_rng(1),
        duration_range_s=(0.5, 2.0),
    )

    assert sample.path == good
    assert sample.speaker_id == "1001"


def test_resolve_speech_audio_path_allows_explicit_real_file(tmp_path):
    from speech_audio import resolve_speech_audio_path

    wav = tmp_path / "custom.wav"
    _write_wav(wav)

    assert resolve_speech_audio_path("speech", explicit_path=wav) == str(wav)


def test_run_audio_pass_resolves_speech_lookup_without_animal_fallback(monkeypatch, tmp_path):
    import run_audio_pass_rlr as rlr

    wav = tmp_path / "speech.wav"
    _write_wav(wav, duration_s=0.5, sample_rate=16000)
    calls = []

    def fake_resolve_speech_audio_path(audio_lookup, explicit_path=None, root=None):
        calls.append((audio_lookup, explicit_path, root))
        return str(wav)

    monkeypatch.setattr(rlr, "resolve_speech_audio_path", fake_resolve_speech_audio_path, raising=False)

    y = rlr._load_dry_source(
        "human_casual_male_v1",
        sample_rate=16000,
        duration_s=0.25,
        source_spec={"audio_lookup": "speech"},
    )

    assert calls == [("speech", None, None)]
    assert y.shape == (4000,)
    assert np.max(np.abs(y)) > 0.1
