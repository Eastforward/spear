import hashlib
import math
import struct
import sys
from pathlib import Path
import wave

import numpy as np

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))


def test_dog_tags_resolve_to_real_dog_audio():
    from animal_audio import resolve_animal_audio_path

    path = Path(resolve_animal_audio_path("dog_beagle_v2", "dog_bark"))

    assert path.exists()
    assert "dog" in path.name.lower() or "bark" in path.name.lower()
    assert not path.name.startswith("__")


def test_cat_tags_resolve_to_real_cat_audio():
    from animal_audio import resolve_animal_audio_path

    path = Path(resolve_animal_audio_path("cat_british_shorthair_v2", "cat_meow"))

    assert path.exists()
    assert "cat" in path.name.lower() or "meow" in path.name.lower()


def test_pixal_namespace_preserves_animal_species_detection():
    from animal_audio import is_animal_tag, species_for_tag

    assert species_for_tag("pixal_dog_golden_retriever_example") == "dog"
    assert species_for_tag("pixal_cat_siamese_example") == "cat"
    assert species_for_tag("gate_pixal_dog_pug_example") == "dog"
    assert is_animal_tag("pixal_dog_golden_retriever_example")


def test_stable_namespace_preserves_animal_species_detection():
    from animal_audio import is_animal_tag, species_for_tag

    tag = "stable_dog_husky_quaternius_ultimate_husky_v1"
    assert species_for_tag(tag) == "dog"
    assert is_animal_tag(tag)


@pytest.mark.parametrize(
    ("tag", "lookup", "expected_species"),
    [
        ("stable_cattle_bovinae_cow_native", "cattle_moo", "cattle_bovinae"),
        ("stable_deer_stag_native", "deer_call", "deer"),
        ("stable_fox_red_fox_native", "fox_call", "fox"),
        ("stable_horse_bay_native", "horse_neigh", "horse"),
        ("stable_wolf_gray_wolf_native", "wolf_howl", "wolf"),
    ],
)
def test_stable_native_species_resolve_to_species_matched_real_audio(
    tag, lookup, expected_species
):
    from animal_audio import resolve_animal_audio_path, species_for_tag

    path = Path(resolve_animal_audio_path(tag, lookup))

    assert species_for_tag(tag) == expected_species
    assert path.is_file()
    assert not path.name.startswith("__")


def test_unpinned_explicit_real_audio_path_wins(tmp_path):
    from animal_audio import resolve_animal_audio_path

    explicit = tmp_path / "custom_dog.wav"
    explicit.write_bytes(b"custom")

    assert resolve_animal_audio_path(
        "dog_golden", "dog_growl", explicit_path=explicit
    ) == str(explicit)


def test_pinned_explicit_path_cannot_bypass_hash_contract(tmp_path):
    from animal_audio import resolve_animal_audio_path

    explicit = tmp_path / "wrong.wav"
    explicit.write_bytes(b"not the pinned dog bark")

    with pytest.raises(ValueError, match="does not match pinned"):
        resolve_animal_audio_path(
            "dog_pembroke_welsh_corgi_candidate",
            "dog_bark",
            explicit_path=explicit,
        )


def test_unknown_lookup_does_not_silently_fall_back_by_species():
    from animal_audio import resolve_animal_audio_path

    with pytest.raises(KeyError, match="unknown animal audio_lookup"):
        resolve_animal_audio_path(
            "dog_pembroke_welsh_corgi_candidate",
            "cat_meow_typo",
        )


def test_cross_species_lookup_is_rejected_for_new_candidates():
    from animal_audio import resolve_animal_audio_path

    with pytest.raises(ValueError, match="but source tag"):
        resolve_animal_audio_path(
            "cat_british_shorthair_candidate",
            "dog_bark",
        )


def test_resolver_rejects_synthetic_sentinel_as_explicit_real_source():
    from animal_audio import resolve_animal_audio_path

    with pytest.raises(ValueError, match="synthetic sentinel"):
        resolve_animal_audio_path(
            "dog_pembroke_welsh_corgi_candidate",
            "dog_bark",
            explicit_path="__synth_piano_scale__",
        )


@pytest.mark.parametrize(
    ("tag", "lookup", "expected_sha256"),
    [
        (
            "dog_pembroke_welsh_corgi_four_limb_rest_side_candidate",
            "dog_bark",
            "5481218ef268b4df98b03e52c48a4973852d60ea5cae9cbf42d0f203a6b9a505",
        ),
        (
            "cat_british_shorthair_four_limb_rest_side_candidate",
            "cat_meow",
            "aa8736bc58a4cd8a35d8911e2cfa22fb2e93fa899b67b1c04a927322c074df3d",
        ),
    ],
)
def test_new_candidate_lookups_resolve_exact_nonformal_contract(
    tag, lookup, expected_sha256
):
    from animal_audio import (
        pinned_animal_audio_contract,
        resolve_animal_audio_path,
    )

    contract = pinned_animal_audio_contract(lookup)
    path = Path(resolve_animal_audio_path(tag, lookup))

    assert path.is_file()
    assert contract["sha256"] == expected_sha256
    assert contract["sample_rate_hz"] == 44100
    assert contract["channels"] == 1
    assert contract["duration_s"] == 10.0
    assert contract["spatialization_status"] == "mono_unspatialized_dry_source"
    assert contract["objective_audio_content_qa_status"] == {
        "dog_bark": "clotho_five_caption_consensus_animal_only_pending_listening",
        "cat_meow": "pending_background_contamination_review",
    }[lookup]
    assert contract["item_level_license_status"] == "missing"
    assert contract["formal_registration_authorized"] is False


@pytest.mark.parametrize(
    ("tag", "expected_lookup", "expected_species"),
    [
        (
            "dog_pembroke_welsh_corgi_candidate",
            "dog_bark",
            "dog",
        ),
        (
            "cat_british_shorthair_candidate",
            "cat_meow",
            "cat",
        ),
    ],
)
def test_audio_audit_uses_pinned_source_without_synthetic_fallback(
    tag, expected_lookup, expected_species
):
    import numpy as np

    sys.path.insert(0, str(REPO / "tools"))
    from audit_animal_audio_events import _resolve_audit_source

    path, source_kind, lookup, contract = _resolve_audit_source(
        tag, np.random.default_rng(0)
    )

    assert path.is_file()
    assert source_kind == "pinned_local"
    assert lookup == expected_lookup
    assert contract["species"] == expected_species
    assert contract["formal_registration_authorized"] is False


def test_synthetic_audio_paths_are_detected():
    from animal_audio import is_synthetic_audio_path

    assert is_synthetic_audio_path("__synth_piano_scale__")
    assert is_synthetic_audio_path("__piano_scale__")
    assert not is_synthetic_audio_path(
        "/data/datasets/omniaudio/train-data-az-360-large/Barking Aldi Dog_358.wav"
    )


@pytest.mark.parametrize(
    ("tag", "lookup", "spatialized_path"),
    [
        (
            "dog_pembroke_welsh_corgi_candidate",
            "dog_bark",
            "/data/datasets/omniaudio/train-data-az-360-large/"
            "Barking Aldi Dog_358.wav",
        ),
        (
            "cat_british_shorthair_candidate",
            "cat_meow",
            "/data/datasets/omniaudio/train-data-az-360-large/"
            "Cat Meowing_293.wav",
        ),
    ],
)
def test_resolver_rejects_known_pre_spatialized_stereo_derivative(
    tag,
    lookup,
    spatialized_path,
):
    from animal_audio import resolve_animal_audio_path

    with pytest.raises(ValueError, match="does not match pinned"):
        resolve_animal_audio_path(
            tag,
            lookup,
            explicit_path=spatialized_path,
        )


def test_package_import_path_is_supported_without_flat_module_aliases():
    from tools.spike_rlr import animal_audio as packaged

    assert packaged.pinned_audio_lookup_for_tag(
        "cat_british_shorthair_candidate"
    ) == "cat_meow"


def _write_test_binaural(path, *, dual_mono=False, continuous=False):
    rate = 16000
    frame_count = 32000
    left = np.zeros(frame_count, dtype=np.int16)
    if continuous:
        windows = ((0, frame_count),)
    else:
        windows = ((1600, 4800), (20800, 24000))
    for start, end in windows:
        phase = np.arange(end - start, dtype=np.float64) / rate
        left[start:end] = np.round(
            5000.0 * np.sin(2.0 * math.pi * 440.0 * phase)
        ).astype(np.int16)
    right = (
        left.copy()
        if dual_mono
        else np.round(left.astype(np.float64) * 0.65).astype(np.int16)
    )
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes(
            np.column_stack((left, right)).astype("<i2", copy=False).tobytes()
        )


def test_binaural_validator_rejects_dual_mono_and_continuous_schedule_spoof(
    tmp_path,
):
    from animal_audio import validate_binaural_wav

    dual_mono = tmp_path / "dual_mono.wav"
    _write_test_binaural(dual_mono, dual_mono=True)
    with pytest.raises(ValueError, match="dual-mono"):
        validate_binaural_wav(
            dual_mono,
            sample_rate_hz=16000,
            duration_s=2.0,
        )

    continuous = tmp_path / "continuous.wav"
    _write_test_binaural(continuous, continuous=True)
    with pytest.raises(ValueError, match="does not follow"):
        validate_binaural_wav(
            continuous,
            sample_rate_hz=16000,
            duration_s=2.0,
            event_windows=[(1600, 4800), (20800, 24000)],
        )


def test_binaural_validator_rejects_truncated_pcm_payload(tmp_path):
    from animal_audio import validate_binaural_wav

    path = tmp_path / "truncated.wav"
    _write_test_binaural(path)
    payload = bytearray(path.read_bytes())
    claimed_data_bytes = 64000 * 2 * 2
    struct.pack_into("<I", payload, 4, 36 + claimed_data_bytes)
    struct.pack_into("<I", payload, 40, claimed_data_bytes)
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="truncated"):
        validate_binaural_wav(
            path,
            sample_rate_hz=16000,
            duration_s=4.0,
        )


def test_generated_tag_uses_spec_identity_and_uncontracted_animal_fails_closed(
    tmp_path,
):
    from animal_audio import (
        animal_species_for_source,
        pinned_animal_audio_contract,
        validate_animal_audio_evidence,
        validate_pinned_source_spec,
    )

    contract = pinned_animal_audio_contract("dog_bark")
    generated = {
        "asset_class": "animal",
        "species": "dog",
        "audio_lookup": "dog_bark",
        "strict_audio": True,
    }
    assert animal_species_for_source(
        "gate_pixal_generated_shiba_inu_red_v1",
        generated,
    ) == "dog"
    assert validate_pinned_source_spec(
        "gate_pixal_generated_shiba_inu_red_v1",
        generated,
        contract=contract,
    ) == contract

    with pytest.raises(ValueError, match="declarations conflict"):
        animal_species_for_source(
            "cat_generated_mislabeled",
            {"asset_class": "animal", "species": "dog"},
        )

    with pytest.raises(ValueError, match="lacks an authenticated"):
        validate_animal_audio_evidence(
            spec={
                "audio_config": {
                    "sample_rate_hz": 16000,
                    "duration_s": 2.0,
                },
                "sources": [
                    {
                        "tag": "stable_horse_bay_native",
                        "audio_lookup": "horse_neigh",
                        "strict_audio": True,
                    }
                ],
            },
            schedule={
                "schema": "rlr_audio_source_schedules_v1",
                "sources": {"stable_horse_bay_native": {}},
            },
            audio_path=tmp_path / "not_reached.wav",
        )


def test_pinned_source_spec_requires_strict_and_embedded_contract():
    from animal_audio import (
        pinned_animal_audio_contract,
        validate_pinned_source_spec,
    )

    contract = pinned_animal_audio_contract("cat_meow")
    tag = "cat_british_shorthair_candidate"
    with pytest.raises(ValueError, match="strict_audio"):
        validate_pinned_source_spec(
            tag,
            {"audio_lookup": "cat_meow"},
            contract=contract,
            require_embedded_contract=True,
        )
    with pytest.raises(ValueError, match="missing from source spec"):
        validate_pinned_source_spec(
            tag,
            {"audio_lookup": "cat_meow", "strict_audio": True},
            contract=contract,
            require_embedded_contract=True,
        )


def test_bind_pinned_contract_populates_every_field_and_rejects_conflict():
    from animal_audio import (
        bind_pinned_animal_audio_contract,
        pinned_animal_audio_contract,
        validate_pinned_source_spec,
    )

    tag = "pixal_generated_shiba_inu"
    source = {
        "tag": tag,
        "asset_class": "animal",
        "species": "dog",
        "audio_lookup": "dog_bark",
        "strict_audio": True,
    }
    bound = bind_pinned_animal_audio_contract(source)
    contract = pinned_animal_audio_contract("dog_bark")

    assert bound is source
    assert source["audio_contract"] == contract
    assert source["audio_path"] == contract["path"]
    assert source["audio_sha256"] == contract["sha256"]
    assert source["audio_source_size_bytes"] == contract["size_bytes"]
    assert source["audio_source_channels"] == 1
    assert source["audio_formal_registration_authorized"] is False
    assert validate_pinned_source_spec(
        tag,
        source,
        contract=contract,
        require_embedded_contract=True,
    ) == contract

    conflicting = dict(source)
    conflicting["audio_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="existing audio_sha256 conflicts"):
        bind_pinned_animal_audio_contract(conflicting)

    unsupported_silence_migration = {
        "tag": "stable_alpaca_quaternius",
        "asset_class": "animal",
        "species": "alpaca",
        "audio_lookup": "silent",
        "strict_audio": True,
    }
    assert (
        bind_pinned_animal_audio_contract(unsupported_silence_migration)
        is unsupported_silence_migration
    )
    assert "audio_contract" not in unsupported_silence_migration


def test_authenticated_reader_rejects_indirect_parent_directory(tmp_path):
    from animal_audio import load_authenticated_file_bytes

    real = tmp_path / "real"
    real.mkdir()
    (real / "evidence.bin").write_bytes(b"evidence")
    indirect = tmp_path / "indirect"
    indirect.symlink_to(real, target_is_directory=True)

    with pytest.raises(OSError):
        load_authenticated_file_bytes(indirect / "evidence.bin")


def test_binaural_mix_must_reconstruct_from_every_distinct_solo(tmp_path):
    from animal_audio import _validate_binaural_mixture_reconstruction

    rate = 16000
    frame_count = rate
    time_s = np.arange(frame_count, dtype=np.float64) / rate

    def write_stereo(path, samples):
        pcm = np.clip(
            np.floor(samples * 32768.0),
            -32768,
            32767,
        ).astype("<i2")
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(2)
            stream.setsampwidth(2)
            stream.setframerate(rate)
            stream.writeframes(pcm.tobytes())

    def descriptor(path):
        payload = path.read_bytes()
        return {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }

    raw_a = np.column_stack(
        (
            0.20 * np.sin(2.0 * np.pi * 440.0 * time_s),
            0.13 * np.sin(2.0 * np.pi * 440.0 * time_s),
        )
    )
    raw_b = np.column_stack(
        (
            0.07 * np.sin(2.0 * np.pi * 880.0 * time_s),
            0.11 * np.sin(2.0 * np.pi * 880.0 * time_s),
        )
    )
    peaks = {
        "dog_a": float(np.max(np.abs(raw_a))),
        "dog_b": float(np.max(np.abs(raw_b))),
    }
    solo_paths = {
        "dog_a": tmp_path / "dog_a.wav",
        "dog_b": tmp_path / "dog_b.wav",
    }
    write_stereo(solo_paths["dog_a"], raw_a * (0.9 / peaks["dog_a"]))
    write_stereo(solo_paths["dog_b"], raw_b * (0.9 / peaks["dog_b"]))

    def decoded(path):
        with wave.open(str(path), "rb") as stream:
            return (
                np.frombuffer(
                    stream.readframes(stream.getnframes()),
                    dtype="<i2",
                )
                .reshape(-1, 2)
                .astype(np.float64)
                / 32768.0
            )

    recovered = sum(
        decoded(solo_paths[tag]) * (peaks[tag] / 0.9)
        for tag in sorted(solo_paths)
    )
    mix_peak = float(np.max(np.abs(recovered)))
    mix_path = tmp_path / "mix.wav"
    write_stereo(mix_path, recovered * (0.9 / mix_peak))
    records = {
        tag: {"pre_normalization_peak": peaks[tag]}
        for tag in solo_paths
    }
    solo_descriptors = {
        tag: descriptor(path) for tag, path in solo_paths.items()
    }
    result = _validate_binaural_mixture_reconstruction(
        mix_path,
        output_descriptor=descriptor(mix_path),
        per_source_records=records,
        solo_descriptors=solo_descriptors,
        mix_pre_normalization_peak=mix_peak,
        sample_rate_hz=rate,
        duration_s=1.0,
    )
    assert result["rms_error"] < 2.0e-4

    write_stereo(mix_path, decoded(solo_paths["dog_a"]))
    with pytest.raises(ValueError, match="does not reconstruct"):
        _validate_binaural_mixture_reconstruction(
            mix_path,
            output_descriptor=descriptor(mix_path),
            per_source_records=records,
            solo_descriptors=solo_descriptors,
            mix_pre_normalization_peak=mix_peak,
            sample_rate_hz=rate,
            duration_s=1.0,
        )

    duplicated = {
        "dog_a": solo_descriptors["dog_a"],
        "dog_b": solo_descriptors["dog_a"],
    }
    with pytest.raises(ValueError, match="duplicate solo"):
        _validate_binaural_mixture_reconstruction(
            solo_paths["dog_a"],
            output_descriptor=solo_descriptors["dog_a"],
            per_source_records=records,
            solo_descriptors=duplicated,
            mix_pre_normalization_peak=mix_peak,
            sample_rate_hz=rate,
            duration_s=1.0,
        )
