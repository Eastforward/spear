import sys
from pathlib import Path

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
            "d244289ddde2d60065e258ef8f336776f2209f7b9240a706dbf8d42888854033",
        ),
        (
            "cat_british_shorthair_four_limb_rest_side_candidate",
            "cat_meow",
            "accd2babb3facabd1f140ce16da9a3986e457f49f175bb0b162c9fa2e070b158",
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
    assert contract["duration_s"] == 10.0
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
