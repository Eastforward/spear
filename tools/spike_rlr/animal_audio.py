"""Small, explicit animal-audio resolver for spike review scenes.

The RLR pass is allowed to synthesize debug tones when a spec asks for one
explicitly. Review/data-generation animal scenes should instead resolve from
animal species + audio_lookup to real files, so a dog tag does not silently
turn into a piano tone.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


SPEAR_ROOT = Path(__file__).resolve().parents[2]
AUDIO_LIBRARY_PATH = SPEAR_ROOT / "data/audio_library_v1.json"
PINNED_AUDIO_LOOKUPS = frozenset({"dog_bark", "cat_meow"})

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
        "catalog": True,
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
        "catalog": True,
    },
    "cat_purring": {
        "species": "cat",
        "path": Path("/data/datasets/cy/omniloc/train/audio/cat purring/-A1eKkZVSRw_000070.mp3"),
    },
    "cattle_moo": {
        "species": "cattle_bovinae",
        "path": Path(
            "/data/datasets/omniaudio/train-data-az-360-large/"
            "Herd Of Cows Mooing_315.wav"
        ),
    },
    "deer_call": {
        "species": "deer",
        "path": Path(
            "/data/datasets/omniaudio/train-data-az-360-large/"
            "rutting deer 1_72.wav"
        ),
    },
    "fox_call": {
        "species": "fox",
        "path": Path(
            "/data/datasets/omniaudio/train-data-az-360-large/"
            "Foxes, West Ham Cemetary 2.3.09 (high pass)_206.wav"
        ),
    },
    "horse_neigh": {
        "species": "horse",
        "path": Path(
            "/data/datasets/omniaudio/train-data-az-360-large/"
            "20090501.horse.neigh_168.wav"
        ),
    },
    "wolf_howl": {
        "species": "wolf",
        "path": Path(
            "/data/datasets/cy/omniloc/train/audio/dog howling/"
            "6WIPUATvzL4_000001.mp3"
        ),
    },
}


_FALLBACK_LOOKUP_BY_SPECIES = {
    "dog": "dog_bark",
    "cat": "cat_meow",
    "cattle_bovinae": "cattle_moo",
    "deer": "deer_call",
    "fox": "fox_call",
    "horse": "horse_neigh",
    "wolf": "wolf_howl",
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
    "alpaca",
    "deer",
    "fox",
    "wolf",
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
    if audio_lookup is not None:
        if audio_lookup not in _AUDIO_BY_LOOKUP:
            raise KeyError(f"unknown animal audio_lookup {audio_lookup!r}")
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


def is_pinned_animal_audio_lookup(audio_lookup: str | None) -> bool:
    return audio_lookup in PINNED_AUDIO_LOOKUPS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_animal_audio_contract(audio_lookup: str) -> dict | None:
    """Return a freshly authenticated exact-source record for a pinned lookup."""
    if audio_lookup not in PINNED_AUDIO_LOOKUPS:
        return None
    # Import lazily so the lightweight species helpers do not need NumPy until
    # a controlled dry source is actually resolved.
    from audio_library import load_library

    sample = load_library(AUDIO_LIBRARY_PATH).require_single(audio_lookup)
    expected_species = _AUDIO_BY_LOOKUP[audio_lookup]["species"]
    if sample.species != expected_species:
        raise ValueError(
            f"catalog species for {audio_lookup!r} is {sample.species!r}, "
            f"expected {expected_species!r}"
        )
    if (
        sample.sha256 is None
        or sample.size_bytes is None
        or sample.channels is None
        or sample.sample_width_bytes is None
        or sample.frame_count is None
        or sample.item_level_license_status is None
    ):
        raise ValueError(f"incomplete pinned animal audio contract: {audio_lookup}")
    return {
        "audio_lookup": audio_lookup,
        "species": sample.species,
        "path": sample.path,
        "sha256": sample.sha256,
        "size_bytes": sample.size_bytes,
        "codec": sample.codec,
        "channels": sample.channels,
        "sample_width_bytes": sample.sample_width_bytes,
        "sample_rate_hz": sample.sample_rate,
        "frame_count": sample.frame_count,
        "duration_s": sample.duration_s,
        "item_level_license_status": sample.item_level_license_status,
        "item_level_license_snapshot": sample.item_level_license_snapshot,
        "formal_registration_authorized": (
            sample.formal_registration_authorized
        ),
    }


def resolve_animal_audio_path(
    tag: str,
    audio_lookup: str | None = None,
    explicit_path: str | Path | None = None,
) -> str:
    """Resolve a real dry-source file for an animal source.

    Pinned lookups are resolved from ``data/audio_library_v1.json`` and
    authenticated on every call. An explicit path for such a lookup must have
    the same SHA-256; it cannot bypass the controlled source contract.
    Synthetic sentinel strings are intentionally rejected here; callers may
    route an explicitly requested debug sentinel to synthesis before invoking
    this resolver.
    """
    lookup = _lookup_for_tag(tag, audio_lookup)
    pinned = pinned_animal_audio_contract(lookup)
    if pinned is not None:
        path = Path(pinned["path"])
    else:
        path = Path(_AUDIO_BY_LOOKUP[lookup]["path"])

    if explicit_path:
        if is_synthetic_audio_path(explicit_path):
            raise ValueError(
                "synthetic sentinel is not a real animal dry-source path"
            )
        explicit = Path(explicit_path)
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        if pinned is not None and _sha256(explicit) != pinned["sha256"]:
            raise ValueError(
                f"explicit path does not match pinned {lookup!r} SHA-256"
            )
        path = explicit

    if not path.is_file():
        raise FileNotFoundError(path)
    if pinned is not None and _sha256(path) != pinned["sha256"]:
        raise ValueError(f"pinned {lookup!r} source changed after catalog load")
    return str(path)


def audio_lookup_species(audio_lookup: str) -> str | None:
    item = _AUDIO_BY_LOOKUP.get(audio_lookup)
    return item["species"] if item else None
