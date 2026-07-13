"""Small, explicit animal-audio resolver for spike review scenes.

The RLR pass is allowed to synthesize debug tones when a spec asks for one
explicitly. Review/data-generation animal scenes should instead resolve from
animal species + audio_lookup to real files, so a dog tag does not silently
turn into a piano tone.
"""
from __future__ import annotations

from pathlib import Path


SYNTHETIC_AUDIO_SENTINELS = {
    "__pink_noise__",
    "__click_train__",
    "__hf_tone__",
    "__piano_scale__",
    "__synth_piano_scale__",
}


_AUDIO_BY_LOOKUP = {
    "dog_bark": {
        "species": "dog",
        "path": Path("/data/datasets/omniaudio/train-data-az-360-large/Barking Aldi Dog_358.wav"),
    },
    "dog_growl": {
        "species": "dog",
        "path": Path("/data/datasets/omniaudio/train-data-az-360-large/Dog Growls_184.wav"),
    },
    "dog_sharp_bark": {
        "species": "dog",
        "path": Path("/data/datasets/omniaudio/train-data-az-360-large/Tiny Dog Barking in Park_338.wav"),
    },
    "cat_meow": {
        "species": "cat",
        "path": Path("/data/datasets/omniaudio/train-data-az-360-large/Cat Meowing_293.wav"),
    },
    "cat_purring": {
        "species": "cat",
        "path": Path("/data/datasets/cy/omniloc/train/audio/cat purring/-A1eKkZVSRw_000070.mp3"),
    },
}


_FALLBACK_LOOKUP_BY_SPECIES = {
    "dog": "dog_bark",
    "cat": "cat_meow",
}

_ANIMAL_TAG_PREFIXES = (
    "dog_",
    "cat_",
    "chipmunk",
    "goat",
    "sheep",
    "pig",
    "horse",
    "cattle_bovinae",
    "yak",
    "donkey_ass",
)

_TECHNICAL_TAG_NAMESPACES = (
    "gate_pixal_",
    "pixal_",
    "stable_",
)


def is_synthetic_audio_path(path: str | Path | None) -> bool:
    if path is None:
        return False
    text = str(path)
    return text in SYNTHETIC_AUDIO_SENTINELS or (
        text.startswith("__") and text.endswith("__")
    )


def species_for_tag(tag: str) -> str | None:
    tag_l = tag.lower()
    for namespace in _TECHNICAL_TAG_NAMESPACES:
        if tag_l.startswith(namespace):
            tag_l = tag_l[len(namespace) :]
            break
    for prefix in _ANIMAL_TAG_PREFIXES:
        if tag_l == prefix.rstrip("_") or tag_l.startswith(prefix):
            return prefix.rstrip("_")
    return None


def is_animal_tag(tag: str) -> bool:
    return species_for_tag(tag) is not None


def _lookup_for_tag(tag: str, audio_lookup: str | None) -> str:
    species = species_for_tag(tag)
    if audio_lookup in _AUDIO_BY_LOOKUP:
        lookup_species = _AUDIO_BY_LOOKUP[audio_lookup]["species"]
        if species is not None and lookup_species != species:
            raise ValueError(
                f"audio_lookup {audio_lookup!r} is {lookup_species}, "
                f"but source tag {tag!r} is {species}"
            )
        return audio_lookup
    if species in _FALLBACK_LOOKUP_BY_SPECIES:
        return _FALLBACK_LOOKUP_BY_SPECIES[species]
    raise KeyError(f"no animal audio fallback for tag {tag!r}")


def resolve_animal_audio_path(
    tag: str,
    audio_lookup: str | None = None,
    explicit_path: str | Path | None = None,
) -> str:
    """Resolve a real dry-source file for an animal source.

    `explicit_path` wins when it points to a real file. Synthetic sentinel
    strings are intentionally not returned here; the caller should detect and
    route them to its synthesis code.
    """
    if explicit_path and not is_synthetic_audio_path(explicit_path):
        return str(explicit_path)

    lookup = _lookup_for_tag(tag, audio_lookup)
    path = Path(_AUDIO_BY_LOOKUP[lookup]["path"])
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path)


def audio_lookup_species(audio_lookup: str) -> str | None:
    item = _AUDIO_BY_LOOKUP.get(audio_lookup)
    return item["species"] if item else None
