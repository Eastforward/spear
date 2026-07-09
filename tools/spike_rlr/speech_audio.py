"""Speech dry-source resolver for visible human sources.

The first human-source path uses local LibriTTS wav files because they are
single-speaker utterances with transcripts and do not need mp3/flac conversion.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


DEFAULT_LIBRITTS_ROOT = Path(
    os.environ.get("AVENGINE_LIBRITTS_ROOT", "/data/datasets/LibriTTS")
)
SPEECH_LOOKUPS = {"speech", "talking", "conversation"}
_PREFERRED_SPLITS = (
    "train-clean-100",
    "train-clean-360",
    "dev-clean",
    "test-clean",
    "train-other-500",
    "dev-other",
    "test-other",
)


@dataclass(frozen=True)
class SpeechSample:
    category: str
    path: Path
    corpus: str
    speaker_id: str | None
    transcript: str | None
    duration_s: float
    sample_rate: int
    is_synthetic: bool = False


def is_speech_lookup(audio_lookup: str | None) -> bool:
    return str(audio_lookup or "") in SPEECH_LOOKUPS


def transcript_path_for_audio(path: Path) -> Path | None:
    for suffix in (".normalized.txt", ".original.txt", ".txt"):
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _speaker_id_from_librispeech_path(path: Path) -> str | None:
    # LibriTTS layout: split/speaker/chapter/utterance.wav.
    try:
        return path.parent.parent.name
    except IndexError:
        return None


def _iter_librttts_wavs(root: Path) -> Iterable[Path]:
    split_dirs = [root / name for name in _PREFERRED_SPLITS if (root / name).is_dir()]
    if not split_dirs:
        split_dirs = [root]
    for split in split_dirs:
        for speaker_dir in sorted(p for p in split.iterdir() if p.is_dir()):
            for chapter_dir in sorted(p for p in speaker_dir.iterdir() if p.is_dir()):
                yield from sorted(chapter_dir.glob("*.wav"))


def _read_transcript(path: Path) -> str | None:
    transcript_path = transcript_path_for_audio(path)
    if transcript_path is None:
        return None
    text = transcript_path.read_text(encoding="utf-8").strip()
    return text or None


def _sample_from_wav(path: Path) -> SpeechSample:
    info = sf.info(str(path))
    return SpeechSample(
        category="speech",
        path=path,
        corpus="LibriTTS",
        speaker_id=_speaker_id_from_librispeech_path(path),
        transcript=_read_transcript(path),
        duration_s=float(info.duration),
        sample_rate=int(info.samplerate),
    )


def pick_speech_sample(
    *,
    root: Path | str | None = None,
    rng: np.random.Generator | None = None,
    duration_range_s: tuple[float, float] = (1.0, 8.0),
    max_candidates: int = 512,
) -> SpeechSample:
    root_path = Path(root) if root is not None else DEFAULT_LIBRITTS_ROOT
    if not root_path.exists():
        raise FileNotFoundError(f"LibriTTS root does not exist: {root_path}")

    lo, hi = duration_range_s
    candidates: list[SpeechSample] = []
    for wav in _iter_librttts_wavs(root_path):
        try:
            sample = _sample_from_wav(wav)
        except RuntimeError:
            continue
        if lo <= sample.duration_s <= hi:
            candidates.append(sample)
            if len(candidates) >= max_candidates:
                break
    if not candidates:
        raise RuntimeError(
            f"no LibriTTS wav in {root_path} within {duration_range_s} seconds"
        )
    rng = rng or np.random.default_rng(0)
    return candidates[int(rng.integers(0, len(candidates)))]


def resolve_speech_audio_path(
    audio_lookup: str | None = "speech",
    *,
    explicit_path: str | Path | None = None,
    root: Path | str | None = None,
) -> str:
    if not is_speech_lookup(audio_lookup):
        raise KeyError(f"not a speech audio lookup: {audio_lookup!r}")
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return str(path)
    return str(pick_speech_sample(root=root).path)
