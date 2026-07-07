"""Audio catalog for scene generation.

Plan 2 initial payload: reuse Plan 1's dog_bark (real, FSD50K-like) +
music_piano (synthetic, in-code sine synth). Plan 3 extends with 8 full
categories from FSD50K + SAO.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class AudioSample:
    category: str
    path: Path
    is_synthetic: bool
    duration_s: float
    sample_rate: int
    source: str    # e.g. "FSD50K", "SAO", "in-code-synth"


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


def load_library(catalog_json_path: Path) -> AudioLibrary:
    j = json.loads(Path(catalog_json_path).read_text())
    samples = [
        AudioSample(
            category=e["category"],
            path=Path(e["path"]),
            is_synthetic=bool(e["is_synthetic"]),
            duration_s=float(e["duration_s"]),
            sample_rate=int(e["sample_rate"]),
            source=e["source"],
        )
        for e in j["samples"]
    ]
    return AudioLibrary(samples)
