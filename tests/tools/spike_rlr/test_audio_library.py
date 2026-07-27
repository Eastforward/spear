import copy
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


def test_pinned_dog_and_cat_sources_match_real_wav_and_remain_nonformal():
    lib = load_library(REPO / "data" / "audio_library_v1.json")

    dog = lib.require_single("dog_bark")
    cat = lib.require_single("cat_meow")

    assert (dog.species, dog.sample_rate, dog.duration_s, dog.channels) == (
        "dog",
        44100,
        10.0,
        1,
    )
    assert (cat.species, cat.sample_rate, cat.duration_s, cat.channels) == (
        "cat",
        44100,
        10.0,
        1,
    )
    assert dog.sha256 == (
        "5481218ef268b4df98b03e52c48a4973852d60ea5cae9cbf42d0f203a6b9a505"
    )
    assert cat.sha256 == (
        "aa8736bc58a4cd8a35d8911e2cfa22fb2e93fa899b67b1c04a927322c074df3d"
    )
    expected_content_status = {
        "dog_bark": "clotho_five_caption_consensus_animal_only_pending_listening",
        "cat_meow": "pending_background_contamination_review",
    }
    for sample in (dog, cat):
        assert sample.codec == "pcm_s16le"
        assert sample.sample_width_bytes == 2
        assert sample.frame_count == 441000
        assert sample.dry_source_policy == "mono_no_hrtf_no_pre_spatialization"
        assert sample.spatialization_status == "mono_unspatialized_dry_source"
        assert sample.objective_audio_content_qa_status == (
            expected_content_status[sample.category]
        )
        assert sample.item_level_license_status == "missing"
        assert sample.item_level_license_snapshot is None
        assert sample.formal_registration_authorized is False


def _real_pinned_entries():
    payload = json.loads((REPO / "data" / "audio_library_v1.json").read_text())
    return [
        copy.deepcopy(entry)
        for entry in payload["samples"]
        if entry["category"] in {"dog_bark", "cat_meow"}
    ]


def test_pinned_catalog_rejects_stale_wav_metadata(tmp_path):
    entries = _real_pinned_entries()
    entries[0]["sample_rate"] = 16000

    with pytest.raises(ValueError, match="WAV metadata changed"):
        load_library(_write_catalog(tmp_path, entries))


def test_pinned_catalog_rejects_cross_species_duplicate(tmp_path):
    entries = _real_pinned_entries()
    cat = entries[1]
    dog = entries[0]
    for field in (
        "path",
        "sha256",
        "size_bytes",
        "codec",
        "channels",
        "sample_width_bytes",
        "sample_rate",
        "frame_count",
        "duration_s",
    ):
        dog[field] = cat[field]

    with pytest.raises(ValueError, match="duplicated across species"):
        load_library(_write_catalog(tmp_path, entries))


def test_pinned_catalog_cannot_authorize_formal_without_item_license(tmp_path):
    entries = _real_pinned_entries()
    entries[0]["formal_registration_authorized"] = True

    with pytest.raises(ValueError, match="v1 pinned source"):
        load_library(_write_catalog(tmp_path, entries))


def test_pinned_catalog_rejects_known_pre_spatialized_stereo_derivative(
    tmp_path,
):
    entries = _real_pinned_entries()
    dog = entries[0]
    dog.update(
        {
            "path": (
                "/data/datasets/omniaudio/train-data-az-360-large/"
                "Growling and Barking Dog_184.wav"
            ),
            "sha256": (
                "c5e243e2293a85352184b279263911d5fe6ac2ba63517e0991e559e03e4d1f55"
            ),
            "size_bytes": 1764044,
            "channels": 2,
            "spatialization_status": "pre_spatialized_hrtf_binaural",
        }
    )

    with pytest.raises(ValueError, match="provenance/spatialization"):
        load_library(_write_catalog(tmp_path, entries))


def test_v1_rejects_unbound_verified_license_snapshot(tmp_path):
    entries = _real_pinned_entries()
    entries[0].update(
        {
            "item_level_license_status": "verified",
            "item_level_license_snapshot": {
                "path": "/tmp/unbound-license.html",
                "sha256": "b" * 64,
            },
        }
    )

    with pytest.raises(ValueError, match="v1 pinned source"):
        load_library(_write_catalog(tmp_path, entries))
