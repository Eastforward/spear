import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))

from audio_library import AudioSample, AudioLibrary, load_library  # noqa: E402


def _write_catalog(tmp_path, entries):
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps({"samples": entries}))
    return p


def test_load_library_from_json(tmp_path):
    catalog = _write_catalog(tmp_path, [
        {"category": "dog_bark", "path": "sound_a.wav", "is_synthetic": False,
         "duration_s": 3.0, "sample_rate": 16000, "source": "FSD50K"},
        {"category": "music_piano", "path": "sound_b.wav", "is_synthetic": True,
         "duration_s": 5.0, "sample_rate": 16000, "source": "SAO"},
    ])
    lib = load_library(catalog)
    assert isinstance(lib, AudioLibrary)
    assert set(lib.categories) == {"dog_bark", "music_piano"}


def test_sample_by_category(tmp_path):
    import numpy as np
    catalog = _write_catalog(tmp_path, [
        {"category": "dog_bark", "path": "a.wav", "is_synthetic": False,
         "duration_s": 3.0, "sample_rate": 16000, "source": "FSD50K"},
        {"category": "dog_bark", "path": "b.wav", "is_synthetic": False,
         "duration_s": 4.0, "sample_rate": 16000, "source": "FSD50K"},
    ])
    lib = load_library(catalog)
    rng = np.random.default_rng(0)
    s = lib.sample("dog_bark", rng)
    assert isinstance(s, AudioSample)
    assert s.category == "dog_bark"
    assert s.path.name in ("a.wav", "b.wav")


def test_sample_random_category(tmp_path):
    import numpy as np
    catalog = _write_catalog(tmp_path, [
        {"category": "dog_bark", "path": "a.wav", "is_synthetic": False,
         "duration_s": 3.0, "sample_rate": 16000, "source": "FSD50K"},
        {"category": "music_piano", "path": "b.wav", "is_synthetic": True,
         "duration_s": 5.0, "sample_rate": 16000, "source": "SAO"},
    ])
    lib = load_library(catalog)
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = lib.sample_random_source(rng)
        assert s.category in {"dog_bark", "music_piano"}


def test_unknown_category_raises(tmp_path):
    import numpy as np
    catalog = _write_catalog(tmp_path, [
        {"category": "dog_bark", "path": "a.wav", "is_synthetic": False,
         "duration_s": 3.0, "sample_rate": 16000, "source": "FSD50K"},
    ])
    lib = load_library(catalog)
    with pytest.raises(KeyError, match="unknown"):
        lib.sample("cat_meow", np.random.default_rng(0))


def test_deterministic_sampling(tmp_path):
    import numpy as np
    catalog = _write_catalog(tmp_path, [
        {"category": "x", "path": "a.wav", "is_synthetic": False,
         "duration_s": 1.0, "sample_rate": 16000, "source": "T"},
        {"category": "x", "path": "b.wav", "is_synthetic": False,
         "duration_s": 1.0, "sample_rate": 16000, "source": "T"},
    ])
    lib = load_library(catalog)
    a = [lib.sample("x", np.random.default_rng(42)).path.name for _ in range(3)]
    b = [lib.sample("x", np.random.default_rng(42)).path.name for _ in range(3)]
    assert a == b
