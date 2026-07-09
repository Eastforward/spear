import sys
from pathlib import Path

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


def test_explicit_audio_path_wins():
    from animal_audio import resolve_animal_audio_path

    explicit = "/tmp/custom_dog.wav"

    assert resolve_animal_audio_path(
        "dog_golden", "dog_bark", explicit_path=explicit
    ) == explicit


def test_synthetic_audio_paths_are_detected():
    from animal_audio import is_synthetic_audio_path

    assert is_synthetic_audio_path("__synth_piano_scale__")
    assert is_synthetic_audio_path("__piano_scale__")
    assert not is_synthetic_audio_path(
        "/data/datasets/omniaudio/train-data-az-360-large/Barking Aldi Dog_358.wav"
    )
