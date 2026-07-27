"""Audio catalog for scene generation.

Plan 2 initial payload: reuse Plan 1's dog_bark (real, FSD50K-like) +
music_piano (synthetic, in-code sine synth). Plan 3 extends with 8 full
categories from FSD50K + SAO.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PINNED_FIELDS = frozenset(
    {
        "species",
        "sha256",
        "size_bytes",
        "codec",
        "channels",
        "sample_width_bytes",
        "frame_count",
        "item_level_license_status",
        "item_level_license_snapshot",
        "formal_registration_authorized",
    }
)


@dataclass(frozen=True)
class AudioSample:
    category: str
    path: Path
    is_synthetic: bool
    duration_s: float
    sample_rate: int
    source: str    # e.g. "FSD50K", "SAO", "in-code-synth"
    species: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    codec: Optional[str] = None
    channels: Optional[int] = None
    sample_width_bytes: Optional[int] = None
    frame_count: Optional[int] = None
    item_level_license_status: Optional[str] = None
    item_level_license_snapshot: Optional[str] = None
    formal_registration_authorized: bool = False


class AudioLibrary:
    def __init__(self, samples):
        self._samples = list(samples)
        self._by_category = {}
        for s in self._samples:
            self._by_category.setdefault(s.category, []).append(s)

    @property
    def categories(self):
        return sorted(self._by_category.keys())

    def sample(self, category: str, rng: np.random.Generator) -> AudioSample:
        if category not in self._by_category:
            raise KeyError(f"unknown category {category!r}; "
                            f"available: {self.categories}")
        pool = self._by_category[category]
        return pool[int(rng.integers(0, len(pool)))]

    def sample_random_source(self, rng: np.random.Generator) -> AudioSample:
        cat = self.categories[int(rng.integers(0, len(self.categories)))]
        return self.sample(cat, rng)

    def require_single(self, category: str) -> AudioSample:
        """Return the sole registered sample for a controlled lookup."""
        pool = self._by_category.get(category)
        if not pool:
            raise KeyError(
                f"unknown category {category!r}; available: {self.categories}"
            )
        if len(pool) != 1:
            raise ValueError(
                f"controlled category {category!r} must have exactly one "
                f"sample, found {len(pool)}"
            )
        return pool[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pinned_entry(entry: dict, *, index: int) -> None:
    """Fail closed when an entry opts into the exact dry-source contract."""
    present = _PINNED_FIELDS.intersection(entry)
    if not present:
        return
    missing = sorted(_PINNED_FIELDS.difference(entry))
    if missing:
        raise ValueError(
            f"samples[{index}] pinned contract missing fields: {missing}"
        )
    category = str(entry.get("category", ""))
    species = str(entry["species"])
    expected_sha256 = str(entry["sha256"])
    if not species:
        raise ValueError(f"samples[{index}].species must be non-empty")
    if not _SHA256_RE.fullmatch(expected_sha256):
        raise ValueError(f"samples[{index}].sha256 is not lowercase SHA-256")
    if bool(entry.get("is_synthetic")):
        raise ValueError(
            f"samples[{index}] pinned animal dry source cannot be synthetic"
        )
    if bool(entry["formal_registration_authorized"]):
        if (
            entry["item_level_license_status"] != "verified"
            or not entry["item_level_license_snapshot"]
        ):
            raise ValueError(
                f"samples[{index}] {category!r} cannot authorize formal "
                "registration without verified item-level license evidence"
            )
    elif entry["item_level_license_status"] == "verified":
        if not entry["item_level_license_snapshot"]:
            raise ValueError(
                f"samples[{index}] verified item-level license lacks snapshot"
            )

    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(entry["size_bytes"]):
        raise ValueError(f"samples[{index}] pinned source size changed: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"samples[{index}] pinned source SHA-256 changed: {path}"
        )
    if str(entry["codec"]) != "pcm_s16le":
        raise ValueError(
            f"samples[{index}] unsupported pinned codec {entry['codec']!r}"
        )
    try:
        with wave.open(str(path), "rb") as stream:
            channels = stream.getnchannels()
            sample_width_bytes = stream.getsampwidth()
            sample_rate = stream.getframerate()
            frame_count = stream.getnframes()
            compression = stream.getcomptype()
    except (wave.Error, EOFError) as error:
        raise ValueError(
            f"samples[{index}] pinned source is not readable PCM WAV: {path}"
        ) from error
    observed = {
        "channels": channels,
        "sample_width_bytes": sample_width_bytes,
        "sample_rate": sample_rate,
        "frame_count": frame_count,
    }
    expected = {
        "channels": int(entry["channels"]),
        "sample_width_bytes": int(entry["sample_width_bytes"]),
        "sample_rate": int(entry["sample_rate"]),
        "frame_count": int(entry["frame_count"]),
    }
    if observed != expected or compression != "NONE":
        raise ValueError(
            f"samples[{index}] pinned WAV metadata changed: "
            f"observed={observed}, expected={expected}, compression={compression}"
        )
    duration_s = frame_count / sample_rate
    if not math.isclose(
        duration_s,
        float(entry["duration_s"]),
        rel_tol=0.0,
        abs_tol=0.5 / sample_rate,
    ):
        raise ValueError(
            f"samples[{index}] stale duration_s for {path}: "
            f"observed={duration_s}, catalog={entry['duration_s']}"
        )


def load_library(catalog_json_path: Path) -> AudioLibrary:
    j = json.loads(Path(catalog_json_path).read_text())
    entries = j["samples"]
    for index, entry in enumerate(entries):
        _validate_pinned_entry(entry, index=index)

    pinned_species_by_hash = {}
    for index, entry in enumerate(entries):
        if not _PINNED_FIELDS.intersection(entry):
            continue
        source_hash = str(entry["sha256"])
        species = str(entry["species"])
        previous_species = pinned_species_by_hash.setdefault(source_hash, species)
        if previous_species != species:
            raise ValueError(
                "pinned animal dry source is duplicated across species: "
                f"sha256={source_hash}, species={previous_species!r}/{species!r}, "
                f"samples[{index}]"
            )

    samples = [
        AudioSample(
            category=e["category"],
            path=Path(e["path"]),
            is_synthetic=bool(e["is_synthetic"]),
            duration_s=float(e["duration_s"]),
            sample_rate=int(e["sample_rate"]),
            source=e["source"],
            species=e.get("species"),
            sha256=e.get("sha256"),
            size_bytes=(
                int(e["size_bytes"]) if e.get("size_bytes") is not None else None
            ),
            codec=e.get("codec"),
            channels=(
                int(e["channels"]) if e.get("channels") is not None else None
            ),
            sample_width_bytes=(
                int(e["sample_width_bytes"])
                if e.get("sample_width_bytes") is not None
                else None
            ),
            frame_count=(
                int(e["frame_count"])
                if e.get("frame_count") is not None
                else None
            ),
            item_level_license_status=e.get("item_level_license_status"),
            item_level_license_snapshot=e.get("item_level_license_snapshot"),
            formal_registration_authorized=bool(
                e.get("formal_registration_authorized", False)
            ),
        )
        for e in entries
    ]
    return AudioLibrary(samples)
