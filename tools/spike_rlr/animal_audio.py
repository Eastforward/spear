"""Small, explicit animal-audio resolver for spike review scenes.

The RLR pass is allowed to synthesize debug tones when a spec asks for one
explicitly. Review/data-generation animal scenes should instead resolve from
animal species + audio_lookup to real files, so a dog tag does not silently
turn into a piano tone.
"""
from __future__ import annotations

from array import array
import contextlib
from contextvars import ContextVar
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping
import wave


SPEAR_ROOT = Path(__file__).resolve().parents[2]
AUDIO_LIBRARY_PATH = SPEAR_ROOT / "data/audio_library_v1.json"
PINNED_AUDIO_LOOKUPS = frozenset({"dog_bark", "cat_meow"})
PINNED_AUDIO_CONTRACT_SCHEMA = "avengine_pinned_animal_dry_source_v1"
ANIMAL_SILENCE_CONTRACT_SCHEMA = "avengine_animal_silence_contract_v1"
ANIMAL_SILENCE_POLICY = "no_authenticated_species_source_available_v1"
APPROVED_SILENT_ANIMAL_SPECIES = frozenset({"alpaca", "donkey_ass"})
AUDIO_RENDER_MANIFEST_SCHEMA = "rlr_audio_render_manifest_v3"
WET_REPLAY_PCM16_TOLERANCE_LSB = 2.0
MAX_AUTHENTICATED_FILE_SIZE_BYTES = 1024 * 1024 * 1024
MAX_AUTHENTICATED_JSON_SIZE_BYTES = 16 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_AUTHENTICATED_PAYLOAD_SNAPSHOT: ContextVar[
    Mapping[str, bytes] | None
] = ContextVar("animal_audio_authenticated_payload_snapshot", default=None)
_PINNED_SOURCE_SPEC_FIELDS = {
    "audio_sha256": "sha256",
    "audio_source_size_bytes": "size_bytes",
    "audio_source_codec": "codec",
    "audio_source_channels": "channels",
    "audio_source_sample_width_bytes": "sample_width_bytes",
    "audio_source_sample_rate_hz": "sample_rate_hz",
    "audio_source_frame_count": "frame_count",
    "audio_source_duration_s": "duration_s",
    "audio_source_species": "species",
    "audio_dry_source_policy": "dry_source_policy",
    "audio_spatialization_status": "spatialization_status",
    "audio_known_spatialized_derivative_sha256": (
        "known_spatialized_derivative_sha256"
    ),
    "audio_item_origin": "item_origin",
    "audio_objective_content_qa_status": "objective_audio_content_qa_status",
    "audio_item_level_license_status": "item_level_license_status",
    "audio_item_level_license_snapshot": "item_level_license_snapshot",
    "audio_formal_registration_authorized": (
        "formal_registration_authorized"
    ),
}

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
        "path": Path(
            "/data/datasets/omniaudio/source_data/processed/clothov2/"
            "Dog Growls.wav"
        ),
    },
    "dog_sharp_bark": {
        "species": "dog",
        "path": Path(
            "/data/datasets/omniaudio/source_data/processed/clothov2/"
            "Tiny Dog Barking in Park.wav"
        ),
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
KNOWN_ANIMAL_SPECIES = frozenset(
    prefix.rstrip("_") for prefix in _ANIMAL_TAG_PREFIXES
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


def animal_species_for_source(
    tag: str,
    source: Mapping[str, Any],
) -> str | None:
    """Resolve animal identity from authenticated source fields, not tag shape.

    Tags remain a useful consistency check, but generated asset tags are not
    required to encode ``dog_`` or ``cat_``.  An explicit animal class,
    species, audio species, pinned contract, or known animal lookup therefore
    takes precedence over tag recognition.  Conflicting declarations fail
    closed instead of silently selecting one.
    """
    if not isinstance(tag, str) or not tag:
        raise ValueError("animal source tag is missing")
    if not isinstance(source, Mapping):
        raise ValueError(f"animal source spec is malformed for {tag!r}")

    asset_class = source.get("asset_class")
    if asset_class is not None and (
        not isinstance(asset_class, str) or not asset_class
    ):
        raise ValueError(f"asset_class is malformed for source {tag!r}")

    declarations: list[tuple[str, str]] = []

    def add(field: str, value: Any) -> None:
        if value in (None, ""):
            return
        if not isinstance(value, str):
            raise ValueError(
                f"animal species field {field!r} is malformed for {tag!r}"
            )
        declarations.append((field, value))

    add("tag", species_for_tag(tag))
    for field in ("species", "audio_source_species", "source_species"):
        add(field, source.get(field))
    for field in ("audio_contract", "source_contract"):
        contract = source.get(field)
        if contract is not None:
            if not isinstance(contract, Mapping):
                raise ValueError(
                    f"animal source contract {field!r} is malformed for {tag!r}"
                )
            add(f"{field}.species", contract.get("species"))

    lookup = source.get("audio_lookup")
    lookup_species = None
    if lookup not in (None, "", "silent"):
        if not isinstance(lookup, str):
            raise ValueError(f"audio_lookup is malformed for source {tag!r}")
        lookup_species = audio_lookup_species(lookup)
        add("audio_lookup", lookup_species)

    declares_animal = (
        asset_class == "animal"
        or bool(declarations)
        or lookup_species is not None
    )
    if not declares_animal:
        return None
    if asset_class not in (None, "animal"):
        raise ValueError(
            f"source {tag!r} declares animal identity but asset_class is "
            f"{asset_class!r}"
        )
    if not declarations:
        raise ValueError(
            f"animal source {tag!r} lacks an authenticated species declaration"
        )
    species_values = {value for _field, value in declarations}
    if len(species_values) != 1:
        rendered = ", ".join(
            f"{field}={value!r}" for field, value in declarations
        )
        raise ValueError(
            f"animal species declarations conflict for {tag!r}: {rendered}"
        )
    species = next(iter(species_values))
    if species not in KNOWN_ANIMAL_SPECIES:
        raise ValueError(
            f"animal source {tag!r} declares unsupported species {species!r}"
        )
    return species


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


def pinned_audio_lookup_for_tag(tag: str) -> str | None:
    """Return the mandatory default lookup for dog/cat tags."""
    species = species_for_tag(tag)
    if species == "dog":
        return "dog_bark"
    if species == "cat":
        return "cat_meow"
    return None


def pinned_audio_lookup_for_source(
    tag: str,
    source: Mapping[str, Any],
) -> str | None:
    """Return the canonical pinned lookup from the source-declared species."""
    species = animal_species_for_source(tag, source)
    if species == "dog":
        return "dog_bark"
    if species == "cat":
        return "cat_meow"
    return None


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
    if __package__:
        from .audio_library import load_library
    else:  # pragma: no cover - exercised by direct-script consumers
        from audio_library import load_library

    source_payload_overrides = _AUTHENTICATED_PAYLOAD_SNAPSHOT.get()
    sample = load_library(
        AUDIO_LIBRARY_PATH,
        source_payload_overrides=source_payload_overrides,
    ).require_single(audio_lookup)
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
        or sample.dry_source_policy is None
        or sample.spatialization_status is None
        or sample.known_spatialized_derivative_sha256 is None
        or sample.item_origin is None
        or sample.objective_audio_content_qa_status is None
        or sample.item_level_license_status is None
    ):
        raise ValueError(f"incomplete pinned animal audio contract: {audio_lookup}")
    if (
        sample.item_level_license_status != "missing"
        or sample.item_level_license_snapshot is not None
        or sample.formal_registration_authorized is not False
    ):
        raise ValueError(
            f"pinned v1 animal audio must remain nonformal without an "
            f"item-level license snapshot: {audio_lookup}"
        )
    return {
        "schema": PINNED_AUDIO_CONTRACT_SCHEMA,
        "audio_lookup": audio_lookup,
        "species": sample.species,
        "path": str(sample.path),
        "sha256": sample.sha256,
        "size_bytes": sample.size_bytes,
        "codec": sample.codec,
        "channels": sample.channels,
        "sample_width_bytes": sample.sample_width_bytes,
        "sample_rate_hz": sample.sample_rate,
        "frame_count": sample.frame_count,
        "duration_s": sample.duration_s,
        "dry_source_policy": sample.dry_source_policy,
        "spatialization_status": sample.spatialization_status,
        "known_spatialized_derivative_sha256": (
            sample.known_spatialized_derivative_sha256
        ),
        "item_origin": dict(sample.item_origin),
        "objective_audio_content_qa_status": (
            sample.objective_audio_content_qa_status
        ),
        "item_level_license_status": sample.item_level_license_status,
        "item_level_license_snapshot": sample.item_level_license_snapshot,
        "formal_registration_authorized": (
            sample.formal_registration_authorized
        ),
    }


def bind_pinned_animal_audio_contract(
    source_spec: dict[str, Any],
) -> dict[str, Any]:
    """Attach the complete authenticated dog/cat dry-source contract.

    Production spec builders must call this after assigning ``tag``,
    ``species``, ``audio_lookup`` and ``strict_audio``. Existing declarations
    are treated as assertions: a stale or conflicting template value is
    rejected instead of silently overwritten.
    """
    if not isinstance(source_spec, dict):
        raise TypeError("pinned animal source spec must be a mutable object")
    tag = source_spec.get("tag")
    if not isinstance(tag, str) or not tag:
        raise ValueError("pinned animal source tag is missing")
    lookup = source_spec.get("audio_lookup")
    declared_species = source_spec.get("species")
    if lookup in (None, ""):
        if declared_species == "dog":
            lookup = "dog_bark"
        elif declared_species == "cat":
            lookup = "cat_meow"
        else:
            lookup = pinned_audio_lookup_for_tag(tag)
    if lookup not in PINNED_AUDIO_LOOKUPS:
        if declared_species in {"dog", "cat"} or species_for_tag(tag) in {
            "dog",
            "cat",
        }:
            raise ValueError(
                f"{tag!r} does not select its mandatory pinned animal audio lookup"
            )
        return source_spec
    species = animal_species_for_source(tag, source_spec)
    contract = pinned_animal_audio_contract(lookup)
    if contract is None:
        raise ValueError(f"pinned source contract is unavailable for {lookup!r}")

    propagated = {
        "audio_lookup": lookup,
        "audio_path": contract["path"],
        "audio_contract": copy.deepcopy(contract),
        "is_synthetic": False,
        "strict_audio": True,
    }
    propagated.update(
        {
            source_field: copy.deepcopy(contract[contract_field])
            for source_field, contract_field in _PINNED_SOURCE_SPEC_FIELDS.items()
        }
    )
    for field, expected in propagated.items():
        if field in source_spec and source_spec[field] != expected:
            raise ValueError(
                f"existing {field} conflicts with pinned source {tag!r}"
            )
    source_spec.update(propagated)
    validate_pinned_source_spec(
        tag,
        source_spec,
        contract=contract,
        require_embedded_contract=True,
    )
    return source_spec


def _expected_animal_silence_contract(
    tag: str,
    source_spec: Mapping[str, Any],
) -> dict[str, Any]:
    species = animal_species_for_source(tag, source_spec)
    if species in {"dog", "cat"}:
        raise ValueError(
            f"controlled {species} source {tag!r} cannot bypass pinned dry audio"
        )
    if species not in APPROVED_SILENT_ANIMAL_SPECIES:
        raise ValueError(
            f"animal source {tag!r} is not approved for explicit silence"
        )
    gate = source_spec.get("stable_animal_gate")
    if not isinstance(gate, Mapping):
        raise ValueError(
            f"silent animal source {tag!r} lacks its authenticated stable gate"
        )
    for field in (
        "asset_id",
        "template_id",
        "tag",
        "species",
        "source_sha256",
        "template_registry",
        "ue_import_result",
    ):
        if gate.get(field) in (None, ""):
            raise ValueError(
                f"silent animal stable gate lacks {field!r} for {tag!r}"
            )
    if (
        gate.get("schema") != "stable_animal_apartment_gate_v1"
        or gate.get("status")
        != "approved_for_automated_research_candidate_apartment"
        or gate.get("tag") != tag
        or gate.get("species") != species
        or gate.get("asset_id") != source_spec.get("asset_id")
        or gate.get("template_id") != source_spec.get("template_id")
        or gate.get("formal_dataset_registration_authorized") is not False
        or not isinstance(gate.get("template_registry"), Mapping)
        or not isinstance(gate.get("ue_import_result"), Mapping)
    ):
        raise ValueError(f"silent animal stable gate identity changed for {tag!r}")
    source_sha256 = gate.get("source_sha256")
    if not isinstance(source_sha256, str) or not _SHA256_RE.fullmatch(
        source_sha256
    ):
        raise ValueError(f"silent animal source hash is malformed for {tag!r}")
    for label in ("template_registry", "ue_import_result"):
        descriptor = gate[label]
        if (
            not isinstance(descriptor.get("path"), str)
            or not Path(descriptor["path"]).is_absolute()
            or not isinstance(descriptor.get("sha256"), str)
            or not _SHA256_RE.fullmatch(descriptor["sha256"])
            or isinstance(descriptor.get("size_bytes"), bool)
            or not isinstance(descriptor.get("size_bytes"), int)
            or descriptor["size_bytes"] <= 0
        ):
            raise ValueError(
                f"silent animal {label} descriptor is malformed for {tag!r}"
            )
    return {
        "schema": ANIMAL_SILENCE_CONTRACT_SCHEMA,
        "policy_id": ANIMAL_SILENCE_POLICY,
        "tag": tag,
        "species": species,
        "asset_id": source_spec["asset_id"],
        "template_id": source_spec["template_id"],
        "source_sha256": gate["source_sha256"],
        "template_registry": copy.deepcopy(gate["template_registry"]),
        "ue_import_result": copy.deepcopy(gate["ue_import_result"]),
        "formal_registration_authorized": False,
    }


def bind_animal_silence_contract(
    source_spec: dict[str, Any],
) -> dict[str, Any]:
    """Attach the only approved explicit-silence contract for stable animals."""
    if not isinstance(source_spec, dict):
        raise TypeError("silent animal source spec must be a mutable object")
    tag = source_spec.get("tag")
    if not isinstance(tag, str) or not tag:
        raise ValueError("silent animal source tag is missing")
    if source_spec.get("audio_lookup") != "silent":
        raise ValueError(
            f"silent animal source {tag!r} must select audio_lookup='silent'"
        )
    expected = _expected_animal_silence_contract(tag, source_spec)
    for field in (
        "audio_contract",
        "audio_path",
        "dry_audio_path",
        *_PINNED_SOURCE_SPEC_FIELDS,
    ):
        source_spec.pop(field, None)
    source_spec.update(
        {
            "audio_lookup": "silent",
            "mute_audio": True,
            "strict_audio": True,
            "is_synthetic": False,
            "audio_silence_policy": ANIMAL_SILENCE_POLICY,
            "audio_silence_contract": expected,
        }
    )
    validate_animal_silence_contract(tag, source_spec)
    return source_spec


def validate_animal_silence_contract(
    tag: str,
    source_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed unless an animal is bound to the approved silence policy."""
    expected = _expected_animal_silence_contract(tag, source_spec)
    embedded = source_spec.get("audio_silence_contract")
    if (
        source_spec.get("audio_lookup") != "silent"
        or source_spec.get("mute_audio") is not True
        or source_spec.get("strict_audio") is not True
        or source_spec.get("is_synthetic") is not False
        or source_spec.get("audio_silence_policy") != ANIMAL_SILENCE_POLICY
        or not isinstance(embedded, Mapping)
        or dict(embedded) != expected
        or source_spec.get("audio_contract") is not None
        or source_spec.get("audio_path") is not None
        or source_spec.get("dry_audio_path") is not None
        or set(_PINNED_SOURCE_SPEC_FIELDS).intersection(source_spec)
    ):
        raise ValueError(f"silent animal source contract changed for {tag!r}")
    for label in ("template_registry", "ue_import_result"):
        _validate_file_descriptor(
            expected[label],
            label=f"silent animal {label}",
        )
    return expected


def validate_pinned_source_spec(
    tag: str,
    source_spec: Mapping[str, Any],
    *,
    contract: Mapping[str, Any] | None = None,
    require_embedded_contract: bool = False,
) -> dict[str, Any]:
    """Authenticate any pinned fields propagated in a scene source spec."""
    species = animal_species_for_source(tag, source_spec)
    lookup = source_spec.get("audio_lookup")
    if lookup in (None, ""):
        lookup = pinned_audio_lookup_for_source(tag, source_spec)
    if not isinstance(lookup, str) or lookup not in PINNED_AUDIO_LOOKUPS:
        raise ValueError(f"{tag!r} does not select a pinned animal audio lookup")
    expected = dict(contract or pinned_animal_audio_contract(lookup) or {})
    if expected.get("schema") != PINNED_AUDIO_CONTRACT_SCHEMA:
        raise ValueError(f"pinned source contract is unavailable for {lookup!r}")
    if species is None or species != expected["species"]:
        raise ValueError(
            f"source tag {tag!r} species {species!r} does not match "
            f"{lookup!r} species {expected['species']!r}"
        )
    if source_spec.get("is_synthetic") is True:
        raise ValueError(f"pinned animal source {tag!r} cannot be synthetic")
    if source_spec.get("strict_audio") is not True:
        raise ValueError(f"pinned animal source {tag!r} must enable strict_audio")
    explicit_path = source_spec.get("audio_path") or source_spec.get(
        "dry_audio_path"
    )
    if explicit_path is not None:
        if is_synthetic_audio_path(explicit_path):
            raise ValueError(f"pinned animal source {tag!r} uses a synthetic sentinel")
        if Path(str(explicit_path)).resolve() != Path(expected["path"]).resolve():
            raise ValueError(f"pinned animal source path changed for {tag!r}")

    embedded = source_spec.get("audio_contract")
    propagated_fields = set(_PINNED_SOURCE_SPEC_FIELDS).intersection(source_spec)
    if propagated_fields or embedded is not None:
        missing = sorted(set(_PINNED_SOURCE_SPEC_FIELDS).difference(source_spec))
        if missing:
            raise ValueError(
                f"propagated pinned audio fields are incomplete for {tag!r}: "
                f"{missing}"
            )
        if not isinstance(embedded, Mapping):
            raise ValueError(
                f"propagated pinned audio contract is missing for {tag!r}"
            )
        if source_spec.get("strict_audio") is not True:
            raise ValueError(
                f"propagated pinned source disabled strict_audio for {tag!r}"
            )
    if require_embedded_contract and not isinstance(embedded, Mapping):
        raise ValueError(f"pinned audio contract is missing from source spec {tag!r}")
    if embedded is not None and dict(embedded) != expected:
        raise ValueError(f"propagated pinned audio contract changed for {tag!r}")
    for source_field, contract_field in _PINNED_SOURCE_SPEC_FIELDS.items():
        if (
            source_field in source_spec
            and source_spec[source_field] != expected[contract_field]
        ):
            raise ValueError(
                f"propagated {source_field} changed for pinned source {tag!r}"
            )
    if (
        "strict_audio" in source_spec
        and source_spec.get("strict_audio") is not True
    ):
        raise ValueError(f"pinned animal source {tag!r} disabled strict_audio")
    return expected


def validate_pinned_schedule_source(
    tag: str,
    source: Mapping[str, Any],
    *,
    source_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject old, synthetic, cross-species, or weak pinned schedule evidence."""
    lookup = source.get("audio_lookup")
    if not isinstance(lookup, str) or lookup not in PINNED_AUDIO_LOOKUPS:
        raise ValueError(f"schedule for {tag!r} lacks a pinned audio_lookup")
    contract = pinned_animal_audio_contract(lookup)
    if contract is None:
        raise ValueError(f"schedule lookup is not pinned: {lookup!r}")
    identity_source = source_spec if source_spec is not None else source
    species = animal_species_for_source(tag, identity_source)
    if (
        source.get("schema") != "animal_audio_event_schedule_v1"
        or source.get("tag") != tag
        or species != contract["species"]
        or source.get("source_species") != species
        or source.get("source_sha256") != contract["sha256"]
        or source.get("source_codec") != contract["codec"]
        or source.get("source_channels") != contract["channels"]
        or source.get("source_sample_width_bytes")
        != contract["sample_width_bytes"]
        or source.get("source_original_sample_rate_hz")
        != contract["sample_rate_hz"]
        or source.get("source_original_frame_count") != contract["frame_count"]
        or source.get("source_original_channels") != contract["channels"]
        or source.get("source_catalog_frame_count") != contract["frame_count"]
        or source.get("source_catalog_duration_s") != contract["duration_s"]
        or source.get("dry_source_policy") != contract["dry_source_policy"]
        or source.get("spatialization_status")
        != contract["spatialization_status"]
        or source.get("known_spatialized_derivative_sha256")
        != contract["known_spatialized_derivative_sha256"]
        or source.get("item_origin") != contract["item_origin"]
        or source.get("objective_audio_content_qa_status")
        != contract["objective_audio_content_qa_status"]
        or source.get("item_level_license_status") != "missing"
        or source.get("item_level_license_snapshot") is not None
        or source.get("formal_registration_authorized") is not False
        or source.get("source_contract") != contract
    ):
        raise ValueError(f"pinned schedule source contract changed for {tag!r}")
    audio_path = source.get("audio_path")
    if (
        not isinstance(audio_path, str)
        or is_synthetic_audio_path(audio_path)
        or Path(audio_path).resolve() != Path(contract["path"]).resolve()
    ):
        raise ValueError(f"schedule source path changed for {tag!r}")
    if not isinstance(source.get("render_sample_rate_hz"), int):
        raise ValueError(f"schedule render sample rate is missing for {tag!r}")
    if source_spec is not None:
        expected_lookup = source_spec.get("audio_lookup")
        if expected_lookup in (None, ""):
            expected_lookup = pinned_audio_lookup_for_source(tag, source_spec)
        if expected_lookup != lookup:
            raise ValueError(f"spec/schedule audio_lookup changed for {tag!r}")
        validate_pinned_source_spec(
            tag,
            source_spec,
            contract=contract,
            require_embedded_contract=True,
        )
    return contract


@contextlib.contextmanager
def _authenticated_payload_snapshot(payloads: Mapping[str, bytes]):
    normalized = {
        str(Path(os.path.abspath(path))): bytes(payload)
        for path, payload in payloads.items()
    }
    token = _AUTHENTICATED_PAYLOAD_SNAPSHOT.set(normalized)
    try:
        yield
    finally:
        _AUTHENTICATED_PAYLOAD_SNAPSHOT.reset(token)


def load_authenticated_file_bytes(
    path: Path,
    *,
    max_size_bytes: int = MAX_AUTHENTICATED_FILE_SIZE_BYTES,
) -> bytes:
    original = Path(os.path.abspath(os.fspath(path)))
    snapshot = _AUTHENTICATED_PAYLOAD_SNAPSHOT.get()
    if snapshot is not None and str(original) in snapshot:
        payload = snapshot[str(original)]
        if not payload or len(payload) > int(max_size_bytes):
            raise ValueError(
                f"authenticated snapshot byte size is invalid: {path}"
            )
        return payload
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    parts = original.parts
    if not original.is_absolute() or len(parts) < 2:
        raise ValueError(f"audio evidence path is invalid: {path}")
    directory_descriptor = os.open(parts[0], directory_flags)
    try:
        for component in parts[1:-1]:
            next_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        descriptor = os.open(
            parts[-1],
            flags,
            dir_fd=directory_descriptor,
        )
    finally:
        os.close(directory_descriptor)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > int(max_size_bytes)
        ):
            raise ValueError(f"audio evidence is not a non-empty regular file: {path}")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or len(payload) != before.st_size:
            raise ValueError(f"audio evidence changed while being read: {path}")
        return payload
    finally:
        os.close(descriptor)


def load_authenticated_json(
    path: Path,
    *,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
) -> dict[str, Any]:
    """Parse one immutable regular-file snapshot as a JSON object."""
    payload = load_authenticated_file_bytes(
        path,
        max_size_bytes=MAX_AUTHENTICATED_JSON_SIZE_BYTES,
    )
    if expected_sha256 is not None:
        if not _SHA256_RE.fullmatch(str(expected_sha256)):
            raise ValueError("audio schedule expected SHA-256 is invalid")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError(f"audio schedule descriptor hash changed: {path}")
    if expected_size_bytes is not None and len(payload) != expected_size_bytes:
        raise ValueError(f"audio schedule descriptor size changed: {path}")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"audio schedule is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"audio schedule must be a JSON object: {path}")
    return value


def validate_binaural_wav(
    path: Path,
    *,
    sample_rate_hz: int,
    duration_s: float,
    expected_sha256: str | None = None,
    expected_size_bytes: int | None = None,
    event_windows: list[tuple[int, int]] | None = None,
) -> dict[str, float | int | str]:
    """Authenticate and inspect the same PCM bytes, rejecting near-silence."""
    payload = load_authenticated_file_bytes(path)
    digest = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None:
        if not _SHA256_RE.fullmatch(str(expected_sha256)):
            raise ValueError("binaural WAV expected SHA-256 is invalid")
        if digest != expected_sha256:
            raise ValueError(f"binaural WAV descriptor hash changed: {path}")
    if expected_size_bytes is not None and len(payload) != expected_size_bytes:
        raise ValueError(f"binaural WAV descriptor size changed: {path}")
    try:
        with wave.open(io.BytesIO(payload), "rb") as stream:
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            rate = stream.getframerate()
            frame_count = stream.getnframes()
            compression = stream.getcomptype()
            frames = stream.readframes(frame_count)
    except (EOFError, wave.Error) as error:
        raise ValueError(f"cannot decode binaural PCM WAV: {path}") from error
    observed_duration_s = frame_count / float(rate)
    if (
        channels != 2
        or sample_width != 2
        or compression != "NONE"
        or rate != int(sample_rate_hz)
        or not math.isclose(
            observed_duration_s,
            float(duration_s),
            rel_tol=0.0,
            abs_tol=1.0 / rate,
        )
    ):
        raise ValueError(f"binaural WAV format contract changed: {path}")
    expected_frame_bytes = frame_count * channels * sample_width
    if len(frames) != expected_frame_bytes:
        raise ValueError(
            f"binaural WAV PCM payload is truncated: expected "
            f"{expected_frame_bytes} bytes, observed {len(frames)}"
        )
    samples = array("h")
    samples.frombytes(frames)
    if sys.byteorder != "little":
        samples.byteswap()
    if not samples:
        raise ValueError(f"binaural WAV contains no samples: {path}")
    peak = max(abs(int(value)) for value in samples) / 32768.0
    rms = math.sqrt(
        sum(float(value) * float(value) for value in samples) / len(samples)
    ) / 32768.0
    if peak < 1.0e-3 or rms < 1.0e-4:
        raise ValueError(
            f"binaural WAV is zero or near-silent: peak={peak}, rms={rms}"
        )
    left_right_square_sum = 0.0
    for index in range(0, len(samples), 2):
        delta = float(samples[index]) - float(samples[index + 1])
        left_right_square_sum += delta * delta
    left_right_rms = math.sqrt(left_right_square_sum / frame_count) / 32768.0
    if left_right_rms <= max(1.0e-7, rms * 1.0e-5):
        raise ValueError(
            f"binaural WAV is dual-mono or lacks a measurable spatial cue: {path}"
        )

    event_rms = None
    gap_rms = None
    if event_windows:
        event_mask = bytearray(frame_count)
        # Keep a short RIR tail with each event. The remaining >=0.85 s gaps
        # still provide a conservative non-event comparison window.
        tail_frames = int(round(0.25 * rate))
        for start, end in event_windows:
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end <= start
                or end > frame_count
            ):
                raise ValueError("animal audio event window is invalid")
            expanded_end = min(frame_count, end + tail_frames)
            event_mask[start:expanded_end] = b"\x01" * (expanded_end - start)
        event_square_sum = 0.0
        gap_square_sum = 0.0
        event_sample_count = 0
        gap_sample_count = 0
        for frame_index in range(frame_count):
            left = float(samples[2 * frame_index])
            right = float(samples[2 * frame_index + 1])
            square = 0.5 * (left * left + right * right)
            if event_mask[frame_index]:
                event_square_sum += square
                event_sample_count += 1
            else:
                gap_square_sum += square
                gap_sample_count += 1
        if event_sample_count <= 0 or gap_sample_count < max(1, rate // 10):
            raise ValueError(
                "animal audio event schedule lacks auditable event/gap windows"
            )
        event_rms = (
            math.sqrt(event_square_sum / event_sample_count) / 32768.0
        )
        gap_rms = math.sqrt(gap_square_sum / gap_sample_count) / 32768.0
        if event_rms <= max(1.0e-4, gap_rms * 1.05):
            raise ValueError(
                "binaural WAV energy does not follow the animal event schedule"
            )
    return {
        "sha256": digest,
        "size_bytes": len(payload),
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate_hz": rate,
        "frame_count": frame_count,
        "duration_s": observed_duration_s,
        "peak": peak,
        "rms": rms,
        "left_right_rms": left_right_rms,
        "event_rms": event_rms,
        "gap_rms": gap_rms,
    }


def _validated_animal_event_windows(
    *,
    tag: str,
    source: Mapping[str, Any],
    sample_rate_hz: int,
    duration_s: float,
) -> list[tuple[int, int]]:
    """Validate self-consistency of one rendered animal event schedule."""
    render_rate = source.get("render_sample_rate_hz")
    schedule_rate = source.get("sample_rate_hz")
    target_duration_s = source.get("target_duration_s")
    if (
        isinstance(render_rate, bool)
        or not isinstance(render_rate, int)
        or render_rate != sample_rate_hz
        or isinstance(schedule_rate, bool)
        or not isinstance(schedule_rate, int)
        or schedule_rate != sample_rate_hz
        or not isinstance(target_duration_s, (int, float))
        or isinstance(target_duration_s, bool)
        or not math.isfinite(float(target_duration_s))
        or not math.isclose(
            float(target_duration_s),
            duration_s,
            rel_tol=0.0,
            abs_tol=1.0 / sample_rate_hz,
        )
    ):
        raise ValueError(f"animal audio render configuration changed for {tag!r}")

    events = source.get("events")
    event_count = source.get("event_count")
    source_events = source.get("source_events")
    source_event_count = source.get("source_event_count")
    if (
        source.get("mode") != "repeated_events_with_silence_gaps"
        or source.get("adaptive_repeat_short_calls") is not True
        or source.get("short_call_detected") is not True
        or not isinstance(events, list)
        or not events
        or isinstance(event_count, bool)
        or not isinstance(event_count, int)
        or event_count != len(events)
        or event_count <= 1
        or not isinstance(source_events, list)
        or not source_events
        or isinstance(source_event_count, bool)
        or not isinstance(source_event_count, int)
        or source_event_count != len(source_events)
    ):
        raise ValueError(f"animal audio event policy changed for {tag!r}")

    for expected_index, source_event in enumerate(source_events):
        if not isinstance(source_event, Mapping):
            raise ValueError(f"animal source event is malformed for {tag!r}")
        source_index = source_event.get("source_event_index")
        crop_start_s = source_event.get("crop_start_s")
        crop_end_s = source_event.get("crop_end_s")
        event_duration_s = source_event.get("duration_s")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index != expected_index
            or not isinstance(crop_start_s, (int, float))
            or isinstance(crop_start_s, bool)
            or not math.isfinite(float(crop_start_s))
            or float(crop_start_s) < 0.0
            or not isinstance(crop_end_s, (int, float))
            or isinstance(crop_end_s, bool)
            or not math.isfinite(float(crop_end_s))
            or float(crop_end_s) <= float(crop_start_s)
            or not isinstance(event_duration_s, (int, float))
            or isinstance(event_duration_s, bool)
            or not math.isfinite(float(event_duration_s))
            or not math.isclose(
                float(event_duration_s),
                float(crop_end_s) - float(crop_start_s),
                rel_tol=0.0,
                abs_tol=1.0 / sample_rate_hz,
            )
        ):
            raise ValueError(f"animal source event bounds changed for {tag!r}")

    frame_count = int(round(sample_rate_hz * duration_s))
    windows: list[tuple[int, int]] = []
    previous_end = None
    gaps = []
    for expected_index, event in enumerate(events):
        if not isinstance(event, Mapping):
            raise ValueError(f"animal audio event is malformed for {tag!r}")
        index = event.get("index")
        start = event.get("start_sample")
        end = event.get("end_sample")
        start_s = event.get("start_s")
        end_s = event.get("end_s")
        source_event_index = event.get("source_event_index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index != expected_index
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > frame_count
            or isinstance(source_event_index, bool)
            or not isinstance(source_event_index, int)
            or not 0 <= source_event_index < source_event_count
            or not isinstance(start_s, (int, float))
            or isinstance(start_s, bool)
            or not math.isfinite(float(start_s))
            or not isinstance(end_s, (int, float))
            or isinstance(end_s, bool)
            or not math.isfinite(float(end_s))
            or not math.isclose(
                float(start_s),
                start / sample_rate_hz,
                rel_tol=0.0,
                abs_tol=1.0 / sample_rate_hz,
            )
            or not math.isclose(
                float(end_s),
                end / sample_rate_hz,
                rel_tol=0.0,
                abs_tol=1.0 / sample_rate_hz,
            )
        ):
            raise ValueError(f"animal audio event bounds changed for {tag!r}")
        if previous_end is not None:
            if start < previous_end:
                raise ValueError(f"animal audio events overlap for {tag!r}")
            gaps.append((start - previous_end) / sample_rate_hz)
        windows.append((start, end))
        previous_end = end

    minimum_gap = source.get("minimum_silence_gap_s")
    if (
        not isinstance(minimum_gap, (int, float))
        or isinstance(minimum_gap, bool)
        or not math.isfinite(float(minimum_gap))
        or float(minimum_gap) < 0.0
    ):
        raise ValueError(f"animal audio silence gap changed for {tag!r}")
    if (
        float(minimum_gap) < 0.85
        or any(
            gap + 1.0 / sample_rate_hz < float(minimum_gap)
            for gap in gaps
        )
    ):
        raise ValueError(f"animal audio repeated-event policy changed for {tag!r}")
    return windows


def _validate_file_descriptor(
    descriptor: Mapping[str, Any],
    *,
    label: str,
    expected_path: Path | None = None,
) -> dict[str, Any]:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{label} descriptor is missing")
    raw_path = descriptor.get("path")
    sha256 = descriptor.get("sha256")
    size_bytes = descriptor.get("size_bytes")
    if (
        not isinstance(raw_path, str)
        or not raw_path
        or not isinstance(sha256, str)
        or not _SHA256_RE.fullmatch(sha256)
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
    ):
        raise ValueError(f"{label} descriptor is malformed")
    path = Path(os.path.abspath(raw_path))
    if expected_path is not None and path != Path(
        os.path.abspath(os.fspath(expected_path))
    ):
        raise ValueError(f"{label} descriptor path changed")
    payload = load_authenticated_file_bytes(path)
    if len(payload) != size_bytes or hashlib.sha256(payload).hexdigest() != sha256:
        raise ValueError(f"{label} descriptor bytes changed")
    return {
        "path": str(path),
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def _recompute_mic_local_azimuths(
    *,
    spec: Mapping[str, Any],
    spec_path: Path,
    source_tags: list[str],
    n_frames: int,
) -> dict[str, list[float]]:
    """Independently derive listener-local source azimuths from the spec."""
    import numpy as np

    raw_sources = spec.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ValueError("RLR spec sources are malformed")
    source_specs = {
        source.get("tag"): source
        for source in raw_sources
        if isinstance(source, Mapping)
    }
    if any(tag not in source_specs for tag in source_tags):
        raise ValueError("RLR azimuth source identity set changed")

    trajectories: dict[str, Any] = {}
    if all("trajectory_m" in source_specs[tag] for tag in source_tags):
        for tag in source_tags:
            trajectory = np.asarray(
                source_specs[tag]["trajectory_m"],
                dtype=np.float64,
            )
            if trajectory.shape != (n_frames, 3):
                raise ValueError(
                    f"RLR source trajectory shape changed for {tag!r}"
                )
            trajectories[tag] = trajectory
    else:
        # The authenticated spec is the input to the same deterministic scene
        # composer used by the renderer.  This invocation is independent of
        # producer-reported manifest azimuths.
        spike_root = SPEAR_ROOT / "tools" / "spike_rlr"
        if str(spike_root) not in sys.path:
            sys.path.insert(0, str(spike_root))
        version = spec.get("spec_version", "v2")
        if version == "apartment_v1":
            from scene_two_dogs_apartment import (
                compose_two_dog_scene_apartment as compose_scene,
            )
        elif version == "v2":
            from scene_two_dogs_v2 import (
                compose_two_dog_scene_v2 as compose_scene,
            )
        else:
            raise ValueError(
                f"unsupported RLR scene version for azimuth audit: {version!r}"
            )
        scene = compose_scene(spec_path)
        placements = {placement.tag: placement for placement in scene.animals}
        if any(tag not in placements for tag in source_tags):
            raise ValueError("RLR composed source identity set changed")
        for tag in source_tags:
            trajectory = np.asarray(
                placements[tag].trajectory_m,
                dtype=np.float64,
            )
            if trajectory.shape != (n_frames, 3):
                raise ValueError(
                    f"RLR composed trajectory shape changed for {tag!r}"
                )
            trajectories[tag] = trajectory

    mic = spec.get("mic")
    if not isinstance(mic, Mapping):
        raise ValueError("RLR mic specification is missing")
    mic_pos = np.asarray(mic.get("pos_m"), dtype=np.float64)
    if mic_pos.shape != (3,) or not np.all(np.isfinite(mic_pos)):
        raise ValueError("RLR mic position is malformed")
    mic_yaw = mic.get("yaw_deg")
    if mic_yaw is None:
        cameras = spec.get("camera_configs", [])
        if isinstance(cameras, list) and cameras and isinstance(cameras[0], Mapping):
            mic_yaw = cameras[0].get("yaw_deg")
    if (
        not isinstance(mic_yaw, (int, float))
        or isinstance(mic_yaw, bool)
        or not math.isfinite(float(mic_yaw))
    ):
        raise ValueError("RLR mic yaw is missing or malformed")
    yaw_radians = math.radians(float(mic_yaw))
    cosine = math.cos(yaw_radians)
    sine = math.sin(yaw_radians)

    result = {}
    for tag in source_tags:
        source = source_specs[tag]
        offset = source.get("audio_source_height_offset_m", 0.0)
        if (
            not isinstance(offset, (int, float))
            or isinstance(offset, bool)
            or not math.isfinite(float(offset))
        ):
            raise ValueError(
                f"RLR acoustic emitter offset changed for {tag!r}"
            )
        trajectory = trajectories[tag].copy()
        trajectory[:, 2] += float(offset)
        relative = trajectory[:, :2] - mic_pos[:2]
        forward = cosine * relative[:, 0] + sine * relative[:, 1]
        left = -sine * relative[:, 0] + cosine * relative[:, 1]
        if np.any(np.hypot(forward, left) <= 1.0e-9):
            raise ValueError(f"RLR source overlaps the mic for {tag!r}")
        azimuths = np.degrees(np.arctan2(left, forward))
        if not np.all(np.isfinite(azimuths)):
            raise ValueError(f"RLR source azimuth is non-finite for {tag!r}")
        result[tag] = azimuths.astype(float).tolist()
    return result


def _require_reported_azimuth_matches_spec(
    reported: Any,
    recomputed: list[float],
    *,
    tag: str,
) -> None:
    if (
        not isinstance(reported, list)
        or len(reported) != len(recomputed)
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in reported
        )
        or any(
            not math.isclose(
                float(observed),
                float(expected),
                rel_tol=0.0,
                abs_tol=1.0e-6,
            )
            for observed, expected in zip(reported, recomputed)
        )
    ):
        raise ValueError(
            f"RLR producer-reported azimuth disagrees with the spec for {tag!r}"
        )


def _validate_band_limited_binaural_ild(
    path: Path,
    *,
    descriptor: Mapping[str, Any],
    azimuth_deg_per_frame: Any,
    event_windows: list[tuple[int, int]],
    sample_rate_hz: int,
    duration_s: float,
    n_frames: int,
    fps: float,
) -> dict[str, Any]:
    """Verify listener-local L/R semantics in the prescribed high bands."""
    if (
        not isinstance(azimuth_deg_per_frame, list)
        or len(azimuth_deg_per_frame) != n_frames
        or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not -180.0 <= float(value) <= 180.0
            for value in azimuth_deg_per_frame
        )
    ):
        raise ValueError("RLR mic-local azimuth evidence is malformed")

    payload = load_authenticated_file_bytes(path)
    if (
        len(payload) != descriptor["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]
    ):
        raise ValueError("RLR binaural ILD input descriptor changed")
    try:
        with wave.open(io.BytesIO(payload), "rb") as stream:
            channels = stream.getnchannels()
            sample_width = stream.getsampwidth()
            rate = stream.getframerate()
            frame_count = stream.getnframes()
            frames = stream.readframes(frame_count)
    except (EOFError, wave.Error) as error:
        raise ValueError("cannot decode RLR binaural ILD input") from error
    if (
        channels != 2
        or sample_width != 2
        or rate != sample_rate_hz
        or frame_count != int(round(sample_rate_hz * duration_s))
        or len(frames) != frame_count * 4
    ):
        raise ValueError("RLR binaural ILD input format changed")

    # NumPy stays a lazy dependency so catalog/spec-only consumers remain
    # lightweight.
    import numpy as np

    signal = (
        np.frombuffer(frames, dtype="<i2")
        .reshape(frame_count, 2)
        .astype(np.float64)
        / 32768.0
    )
    samples_per_frame = int(round(sample_rate_hz / fps))
    bands = ((1500.0, 5000.0), (5000.0, 7500.0))
    observations = {band: [] for band in bands}

    def overlaps_event(start: int, end: int) -> bool:
        return any(start < event_end and end > event_start for event_start, event_end in event_windows)

    for frame_index, raw_azimuth in enumerate(azimuth_deg_per_frame):
        start = frame_index * samples_per_frame
        end = min(start + samples_per_frame, frame_count)
        if end - start < 32 or not overlaps_event(start, end):
            continue
        block = signal[start:end]
        if float(np.sqrt(np.mean(block * block))) < 1.0e-4:
            continue
        windowed = block * np.hanning(len(block))[:, None]
        spectrum = np.fft.rfft(windowed, axis=0)
        power = np.abs(spectrum) ** 2
        frequencies = np.fft.rfftfreq(len(block), d=1.0 / sample_rate_hz)
        broadband_power = float(np.sum(power[1:, :]))
        azimuth = float(raw_azimuth)
        for band in bands:
            selected = (frequencies >= band[0]) & (frequencies < band[1])
            left_power = float(np.sum(power[selected, 0]))
            right_power = float(np.sum(power[selected, 1]))
            total_power = left_power + right_power
            if (
                total_power <= 1.0e-8
                or total_power < broadband_power * 1.0e-4
            ):
                continue
            ild_db = 10.0 * math.log10(
                (left_power + 1.0e-12) / (right_power + 1.0e-12)
            )
            observations[band].append((azimuth, ild_db))

    diagnostics = {}
    eligible_but_failed = False
    for band, rows in observations.items():
        label = f"{int(band[0])}_{int(band[1])}_hz"
        left = [ild for azimuth, ild in rows if azimuth >= 30.0]
        right = [ild for azimuth, ild in rows if azimuth <= -30.0]
        if len(left) >= 3 and len(right) >= 3:
            left_median = float(np.median(left))
            right_median = float(np.median(right))
            separation = left_median - right_median
            passed = (
                left_median > 0.0
                and right_median < 0.0
                and separation >= 1.0
            )
            diagnostics[label] = {
                "mode": "cross_side",
                "left_frame_count": len(left),
                "right_frame_count": len(right),
                "left_median_ild_db": left_median,
                "right_median_ild_db": right_median,
                "separation_db": separation,
                "status": "passed" if passed else "failed",
            }
            if passed:
                return {
                    "status": "passed",
                    "accepted_band": label,
                    "bands": diagnostics,
                }
            eligible_but_failed = True
            continue

        signed = [
            (1.0 if azimuth > 0.0 else -1.0) * ild
            for azimuth, ild in rows
            if abs(azimuth) >= 15.0
        ]
        if len(signed) >= 3:
            median_signed = float(np.median(signed))
            passed = median_signed >= 0.25
            diagnostics[label] = {
                "mode": "single_side",
                "eligible_frame_count": len(signed),
                "median_signed_ild_db": median_signed,
                "status": "passed" if passed else "failed",
            }
            if passed:
                return {
                    "status": "passed",
                    "accepted_band": label,
                    "bands": diagnostics,
                }
            eligible_but_failed = True
        else:
            diagnostics[label] = {
                "mode": "not_applicable",
                "eligible_frame_count": len(signed),
                "status": "not_applicable",
            }
    if eligible_but_failed:
        raise ValueError("RLR band-limited binaural ILD has wrong L/R semantics")
    raise ValueError("RLR band-limited binaural ILD lacks eligible event frames")


def _validate_binaural_mixture_reconstruction(
    audio_path: Path,
    *,
    output_descriptor: Mapping[str, Any],
    per_source_records: Mapping[str, Mapping[str, Any]],
    solo_descriptors: Mapping[str, Mapping[str, Any]],
    mix_pre_normalization_peak: float,
    sample_rate_hz: int,
    duration_s: float,
) -> dict[str, float]:
    """Reconstruct the normalized mix from authenticated normalized solos."""
    import numpy as np

    def load_stereo(path: Path, descriptor: Mapping[str, Any]) -> Any:
        payload = load_authenticated_file_bytes(path)
        if (
            len(payload) != descriptor["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]
        ):
            raise ValueError("RLR mix-closure descriptor changed")
        try:
            with wave.open(io.BytesIO(payload), "rb") as stream:
                if (
                    stream.getnchannels() != 2
                    or stream.getsampwidth() != 2
                    or stream.getframerate() != sample_rate_hz
                    or stream.getnframes()
                    != int(round(sample_rate_hz * duration_s))
                ):
                    raise ValueError("RLR mix-closure WAV format changed")
                frames = stream.readframes(stream.getnframes())
        except (EOFError, wave.Error) as error:
            raise ValueError("cannot decode RLR mix-closure WAV") from error
        expected_bytes = int(round(sample_rate_hz * duration_s)) * 4
        if len(frames) != expected_bytes:
            raise ValueError("RLR mix-closure WAV is truncated")
        return (
            np.frombuffer(frames, dtype="<i2")
            .reshape(-1, 2)
            .astype(np.float64)
            / 32768.0
        )

    if (
        not math.isfinite(float(mix_pre_normalization_peak))
        or float(mix_pre_normalization_peak) <= 0.0
    ):
        raise ValueError("RLR mixed pre-normalization peak is invalid")
    mixture = load_stereo(audio_path, output_descriptor)
    recovered = np.zeros_like(mixture)
    solo_hashes = []
    for tag, record in per_source_records.items():
        descriptor = solo_descriptors[tag]
        solo_hashes.append(descriptor["sha256"])
        peak = float(record["pre_normalization_peak"])
        solo_path = Path(descriptor["path"])
        recovered += load_stereo(solo_path, descriptor) * (peak / 0.9)
    if len(solo_hashes) != len(set(solo_hashes)):
        raise ValueError("distinct RLR sources published duplicate solo WAVs")
    recovered_peak = float(np.max(np.abs(recovered)))
    if not math.isclose(
        recovered_peak,
        float(mix_pre_normalization_peak),
        rel_tol=5.0e-3,
        abs_tol=2.0e-5,
    ):
        raise ValueError("RLR per-source peaks do not reconstruct the mix peak")
    expected = recovered * (0.9 / float(mix_pre_normalization_peak))
    error = mixture - expected
    rms_error = float(np.sqrt(np.mean(error * error)))
    max_error = float(np.max(np.abs(error)))
    if rms_error > 2.0e-4 or max_error > 2.0e-3:
        raise ValueError("RLR mixed WAV does not reconstruct from per-source solos")
    return {
        "recovered_pre_normalization_peak": recovered_peak,
        "rms_error": rms_error,
        "max_error": max_error,
    }


def _validate_scheduled_dry_to_binaural_causality(
    *,
    dry_path: Path,
    dry_descriptor: Mapping[str, Any],
    wet_path: Path,
    wet_descriptor: Mapping[str, Any],
    event_windows: list[tuple[int, int]],
    sample_rate_hz: int,
    duration_s: float,
) -> dict[str, float]:
    """Reject event-shaped tones that are unrelated to the pinned dry call.

    Room convolution changes phase and colors the spectrum, so byte equality
    is neither expected nor useful here.  The gate instead requires both a
    matching coarse spectral fingerprint and a matching event envelope.  The
    exact scheduled dry bytes are separately reconstructed from the pinned
    source and schedule before this function is called.
    """
    import numpy as np

    frame_count = int(round(sample_rate_hz * duration_s))

    def decode(
        path: Path,
        descriptor: Mapping[str, Any],
        *,
        channels: int,
    ) -> Any:
        payload = load_authenticated_file_bytes(path)
        if (
            len(payload) != descriptor["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]
        ):
            raise ValueError("RLR dry/wet causality descriptor changed")
        try:
            with wave.open(io.BytesIO(payload), "rb") as stream:
                if (
                    stream.getnchannels() != channels
                    or stream.getsampwidth() != 2
                    or stream.getframerate() != sample_rate_hz
                    or stream.getnframes() != frame_count
                    or stream.getcomptype() != "NONE"
                ):
                    raise ValueError("RLR dry/wet causality WAV format changed")
                frames = stream.readframes(frame_count)
        except (EOFError, wave.Error) as error:
            raise ValueError("cannot decode RLR dry/wet causality WAV") from error
        if len(frames) != frame_count * channels * 2:
            raise ValueError("RLR dry/wet causality WAV is truncated")
        return (
            np.frombuffer(frames, dtype="<i2")
            .reshape(frame_count, channels)
            .astype(np.float64)
            / 32768.0
        )

    dry = decode(dry_path, dry_descriptor, channels=1)[:, 0]
    wet_stereo = decode(wet_path, wet_descriptor, channels=2)
    wet = np.sqrt(np.mean(np.square(wet_stereo), axis=1))
    if (
        float(np.max(np.abs(dry))) < 1.0e-3
        or float(np.sqrt(np.mean(dry * dry))) < 1.0e-4
    ):
        raise ValueError("scheduled pinned dry reference is near-silent")

    analysis_size = 1024
    hop = 512
    window = np.hanning(analysis_size)
    frequencies = np.fft.rfftfreq(analysis_size, d=1.0 / sample_rate_hz)
    band_edges = np.geomspace(100.0, min(7500.0, sample_rate_hz * 0.475), 25)
    dry_band_power = np.zeros(len(band_edges) - 1, dtype=np.float64)
    wet_band_power = np.zeros_like(dry_band_power)
    analyzed_frames = 0

    def overlaps_event(start: int, end: int) -> bool:
        return any(
            start < event_end and end > event_start
            for event_start, event_end in event_windows
        )

    for start in range(0, max(1, frame_count - analysis_size + 1), hop):
        end = start + analysis_size
        if end > frame_count or not overlaps_event(start, end):
            continue
        dry_block = dry[start:end]
        if float(np.sqrt(np.mean(dry_block * dry_block))) < 1.0e-4:
            continue
        wet_block = wet_stereo[start:end]
        dry_power = np.abs(np.fft.rfft(dry_block * window)) ** 2
        wet_power = np.mean(
            np.abs(np.fft.rfft(wet_block * window[:, None], axis=0)) ** 2,
            axis=1,
        )
        for index, (low, high) in enumerate(
            zip(band_edges[:-1], band_edges[1:])
        ):
            selected = (frequencies >= low) & (frequencies < high)
            dry_band_power[index] += float(np.sum(dry_power[selected]))
            wet_band_power[index] += float(np.sum(wet_power[selected]))
        analyzed_frames += 1
    if analyzed_frames < 3:
        raise ValueError("scheduled dry causality lacks analyzable event frames")
    if (
        float(np.sum(dry_band_power)) <= 1.0e-8
        or float(np.sum(wet_band_power)) <= 1.0e-8
    ):
        raise ValueError("scheduled dry causality lacks spectral energy")

    dry_distribution = dry_band_power / np.sum(dry_band_power)
    wet_distribution = wet_band_power / np.sum(wet_band_power)
    spectral_overlap = float(
        np.sum(np.sqrt(dry_distribution * wet_distribution))
    )
    dry_log = np.log10(dry_band_power + np.max(dry_band_power) * 1.0e-8)
    wet_log = np.log10(wet_band_power + np.max(wet_band_power) * 1.0e-8)
    spectral_correlation = float(np.corrcoef(dry_log, wet_log)[0, 1])

    envelope_block = max(1, int(round(0.020 * sample_rate_hz)))
    usable = frame_count - frame_count % envelope_block
    dry_envelope = np.sqrt(
        np.mean(
            np.square(dry[:usable].reshape(-1, envelope_block)),
            axis=1,
        )
    )
    wet_envelope = np.sqrt(
        np.mean(
            np.square(wet[:usable].reshape(-1, envelope_block)),
            axis=1,
        )
    )
    max_lag = max(1, int(round(0.25 / 0.020)))
    envelope_correlation = -1.0
    accepted_lag = 0
    for lag in range(max_lag + 1):
        left = dry_envelope[: len(dry_envelope) - lag or None]
        right = wet_envelope[lag:]
        if len(left) < 8 or np.std(left) <= 1.0e-8 or np.std(right) <= 1.0e-8:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        if correlation > envelope_correlation:
            envelope_correlation = correlation
            accepted_lag = lag
    if (
        not math.isfinite(spectral_correlation)
        or spectral_overlap < 0.55
        or spectral_correlation < 0.20
        or envelope_correlation < 0.15
    ):
        raise ValueError(
            "RLR binaural output is not causally consistent with the pinned "
            "scheduled dry source"
        )
    return {
        "spectral_overlap": spectral_overlap,
        "spectral_correlation": spectral_correlation,
        "envelope_correlation": envelope_correlation,
        "accepted_envelope_lag_s": accepted_lag * 0.020,
        "analyzed_stft_frames": float(analyzed_frames),
    }


def _decode_authenticated_stereo_pcm16(
    path: Path,
    descriptor: Mapping[str, Any],
    *,
    sample_rate_hz: int,
    frame_count: int,
    label: str,
) -> Any:
    """Decode one authenticated fixed-format stereo PCM16 WAV."""
    import numpy as np

    payload = load_authenticated_file_bytes(path)
    if (
        len(payload) != descriptor["size_bytes"]
        or hashlib.sha256(payload).hexdigest() != descriptor["sha256"]
    ):
        raise ValueError(f"{label} descriptor changed")
    try:
        with wave.open(io.BytesIO(payload), "rb") as stream:
            if (
                stream.getnchannels() != 2
                or stream.getsampwidth() != 2
                or stream.getframerate() != sample_rate_hz
                or stream.getnframes() != frame_count
                or stream.getcomptype() != "NONE"
            ):
                raise ValueError(f"{label} WAV format changed")
            frames = stream.readframes(frame_count)
    except (EOFError, wave.Error) as error:
        raise ValueError(f"cannot decode {label} WAV") from error
    if len(frames) != frame_count * 4:
        raise ValueError(f"{label} WAV is truncated")
    return (
        np.frombuffer(frames, dtype="<i2")
        .reshape(frame_count, 2)
        .astype(np.float64)
        / 32768.0
    )


def _validate_active_frame_rir_source_replay(
    *,
    tag: str,
    spec: Mapping[str, Any],
    spec_path: Path,
    source_spec: Mapping[str, Any],
    schedule_source: Mapping[str, Any],
    rir_descriptor: Mapping[str, Any],
    solo_path: Path,
    solo_descriptor: Mapping[str, Any],
    reported_peak: float,
    sample_rate_hz: int,
    duration_s: float,
    n_frames: int,
    fps: float,
) -> tuple[Any, dict[str, Any]]:
    """Rebuild one raw RLR solo from authenticated dry and active-frame RIRs."""
    import numpy as np

    if __package__:
        from .active_frame_rir_evidence import (
            load_active_frame_rir_evidence,
            replay_active_frame_rirs,
        )
        from .scheduled_dry_evidence import reconstruct_scheduled_dry
        from .source_trajectory import acoustic_trajectory
    else:  # pragma: no cover - direct-script consumers
        from active_frame_rir_evidence import (
            load_active_frame_rir_evidence,
            replay_active_frame_rirs,
        )
        from scheduled_dry_evidence import reconstruct_scheduled_dry
        from source_trajectory import acoustic_trajectory

    dry = reconstruct_scheduled_dry(
        source_contract=schedule_source["source_contract"],
        schedule=schedule_source,
    )
    frame_count = int(round(sample_rate_hz * duration_s))
    samples_per_frame = int(round(sample_rate_hz / fps))
    if dry.shape != (frame_count,):
        raise ValueError(f"scheduled dry length changed for RIR replay {tag!r}")
    trajectory = source_spec.get("trajectory_m")
    if trajectory is None:
        spike_root = SPEAR_ROOT / "tools" / "spike_rlr"
        if str(spike_root) not in sys.path:
            sys.path.insert(0, str(spike_root))
        version = spec.get("spec_version", "v2")
        if version == "apartment_v1":
            from scene_two_dogs_apartment import (
                compose_two_dog_scene_apartment as compose_scene,
            )
        elif version == "v2":
            from scene_two_dogs_v2 import (
                compose_two_dog_scene_v2 as compose_scene,
            )
        else:
            raise ValueError(
                f"unsupported RIR replay scene version: {version!r}"
            )
        scene = compose_scene(spec_path)
        placements = {
            placement.tag: placement for placement in scene.animals
        }
        if tag not in placements:
            raise ValueError(
                f"RIR replay composed trajectory is missing for {tag!r}"
            )
        trajectory = placements[tag].trajectory_m
    expected_trajectory = acoustic_trajectory(trajectory, dict(source_spec))
    if expected_trajectory.shape != (n_frames, 3):
        raise ValueError(f"RIR replay trajectory shape changed for {tag!r}")
    mic = spec.get("mic")
    if not isinstance(mic, Mapping):
        raise ValueError("RIR replay microphone specification is missing")
    mic_position = np.asarray(mic.get("pos_m"), dtype=np.float64)
    mic_yaw = mic.get("yaw_deg")
    if mic_yaw is None:
        cameras = spec.get("camera_configs", [])
        if isinstance(cameras, list) and cameras and isinstance(cameras[0], Mapping):
            mic_yaw = cameras[0].get("yaw_deg")
    if (
        mic_position.shape != (3,)
        or not np.all(np.isfinite(mic_position))
        or isinstance(mic_yaw, bool)
        or not isinstance(mic_yaw, (int, float))
        or not math.isfinite(float(mic_yaw))
    ):
        raise ValueError("RIR replay microphone transform is malformed")

    rir_payload = load_authenticated_file_bytes(Path(rir_descriptor["path"]))
    if (
        len(rir_payload) != rir_descriptor["size_bytes"]
        or hashlib.sha256(rir_payload).hexdigest() != rir_descriptor["sha256"]
    ):
        raise ValueError(f"active-frame RIR descriptor changed for {tag!r}")
    evidence = load_active_frame_rir_evidence(
        rir_payload,
        source_tag=tag,
        dry=dry,
        expected_source_trajectory_scene_m=expected_trajectory,
        mic_position_scene_m=mic_position,
        mic_yaw_deg=float(mic_yaw),
        sample_rate_hz=sample_rate_hz,
        n_samples_total=frame_count,
        n_frames=n_frames,
        fps=fps,
        samples_per_frame=samples_per_frame,
    )
    replayed_raw = replay_active_frame_rirs(
        dry,
        evidence,
        n_samples_total=frame_count,
        samples_per_frame=samples_per_frame,
    )
    replayed_peak = float(np.max(np.abs(replayed_raw)))
    if (
        not math.isfinite(replayed_peak)
        or replayed_peak <= 1.0e-9
        or not math.isclose(
            replayed_peak,
            float(reported_peak),
            rel_tol=1.0e-6,
            abs_tol=1.0e-8,
        )
    ):
        raise ValueError(f"RLR raw solo peak does not replay for {tag!r}")
    expected_solo = (
        replayed_raw.T.astype(np.float64)
        * (0.9 / replayed_peak)
    )
    observed_solo = _decode_authenticated_stereo_pcm16(
        solo_path,
        solo_descriptor,
        sample_rate_hz=sample_rate_hz,
        frame_count=frame_count,
        label=f"RLR solo {tag!r}",
    )
    error_lsb = np.abs(observed_solo - expected_solo) * 32768.0
    max_error_lsb = float(np.max(error_lsb))
    if max_error_lsb > WET_REPLAY_PCM16_TOLERANCE_LSB:
        raise ValueError(
            f"RLR solo does not replay from active-frame RIRs for {tag!r}: "
            f"{max_error_lsb:.3f} LSB"
        )
    return replayed_raw, {
        "status": "passed",
        "active_frame_count": int(len(evidence["frame_indices"])),
        "replayed_pre_normalization_peak": replayed_peak,
        "max_pcm16_error_lsb": max_error_lsb,
        # Registration reuses this exact in-memory trajectory after the
        # evidence loader has bound every active position to the authenticated
        # spec/composer result.  The independent worker receives only the
        # prefix required to warm temporal coherence; it never receives an
        # expected RIR.
        "validated_source_trajectory_scene_m": expected_trajectory.tolist(),
    }


def _validate_active_frame_rir_final_mix(
    *,
    replayed_sources: Mapping[str, Any],
    audio_path: Path,
    output_descriptor: Mapping[str, Any],
    reported_mix_peak: float,
    sample_rate_hz: int,
    duration_s: float,
) -> dict[str, float]:
    """Rebuild the final normalized mix from independently replayed raw solos."""
    import numpy as np

    frame_count = int(round(sample_rate_hz * duration_s))
    if not replayed_sources:
        raise ValueError("RIR replay has no audible sources")
    replayed_mix = np.zeros((2, frame_count), dtype=np.float32)
    for tag in sorted(replayed_sources):
        raw = np.asarray(replayed_sources[tag], dtype=np.float32)
        if raw.shape != replayed_mix.shape or not np.all(np.isfinite(raw)):
            raise ValueError(f"RIR replayed source is malformed for {tag!r}")
        replayed_mix += raw
    replayed_peak = float(np.max(np.abs(replayed_mix)))
    if (
        replayed_peak <= 1.0e-9
        or not math.isclose(
            replayed_peak,
            float(reported_mix_peak),
            rel_tol=1.0e-6,
            abs_tol=1.0e-8,
        )
    ):
        raise ValueError("RLR final raw mix peak does not replay")
    expected = replayed_mix.T.astype(np.float64) * (0.9 / replayed_peak)
    observed = _decode_authenticated_stereo_pcm16(
        audio_path,
        output_descriptor,
        sample_rate_hz=sample_rate_hz,
        frame_count=frame_count,
        label="RLR final mix",
    )
    error_lsb = np.abs(observed - expected) * 32768.0
    max_error_lsb = float(np.max(error_lsb))
    if max_error_lsb > WET_REPLAY_PCM16_TOLERANCE_LSB:
        raise ValueError(
            "RLR final mix does not replay from active-frame RIRs: "
            f"{max_error_lsb:.3f} LSB"
        )
    return {
        "replayed_pre_normalization_peak": replayed_peak,
        "max_pcm16_error_lsb": max_error_lsb,
    }


def validate_audio_render_manifest(
    manifest: Mapping[str, Any],
    *,
    spec: Mapping[str, Any],
    spec_path: Path,
    schedule: Mapping[str, Any],
    schedule_path: Path,
    audio_path: Path,
    portable_artifact_resolution: bool = False,
) -> dict[str, Any]:
    """Authenticate the RLR inputs, per-source solos, schedule, and mix."""
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema") != AUDIO_RENDER_MANIFEST_SCHEMA
        or manifest.get("backend") != "habitat_sim_rlr_audio_sensor"
        or manifest.get("channel_layout") != "binaural_native"
        or manifest.get("native_binaural_channel_order") != [0, 1]
        or manifest.get("technical_render_status") != "passed"
        or manifest.get("formal_registration_authorized") is not False
        or manifest.get("behavior_gates")
        != {
            "per_source_event_alignment": "validate_on_readback",
            "binaural_spatial_ild": "validate_on_readback",
            "mixture_reconstruction": "validate_on_readback",
            "active_frame_rir_replay": "validate_on_readback",
        }
    ):
        raise ValueError("RLR audio render manifest contract changed")

    config = spec.get("audio_config", {})
    render = spec.get("render_config", {})
    sample_rate_hz = int(config["sample_rate_hz"])
    duration_s = float(config["duration_s"])
    n_frames = int(render["n_frames"])
    fps = float(render["fps"])
    quality_mode = manifest.get("quality_mode")
    if __package__:
        from .acoustic_scene_contract import approved_rlr_renderer_contract
    else:  # pragma: no cover - direct-script consumers
        from acoustic_scene_contract import approved_rlr_renderer_contract
    expected_renderer = approved_rlr_renderer_contract(
        sample_rate_hz=sample_rate_hz,
        quality_mode=quality_mode,
    )
    expected_rays = expected_renderer["acoustics"]["indirect_ray_count"]
    if (
        manifest.get("sample_rate_hz") != sample_rate_hz
        or not isinstance(manifest.get("duration_s"), (int, float))
        or isinstance(manifest.get("duration_s"), bool)
        or not math.isclose(
            float(manifest["duration_s"]),
            duration_s,
            rel_tol=0.0,
            abs_tol=1.0 / sample_rate_hz,
        )
        or manifest.get("n_frames") != n_frames
        or not isinstance(manifest.get("fps"), (int, float))
        or isinstance(manifest.get("fps"), bool)
        or not math.isclose(
            float(manifest["fps"]),
            fps,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        or manifest.get("indirect_ray_count") != expected_rays
        or manifest.get("renderer_contract") != expected_renderer
    ):
        raise ValueError("RLR audio render configuration changed")

    schedule_sources = schedule.get("sources", {})
    if not isinstance(schedule_sources, Mapping):
        raise ValueError("RLR source schedule is malformed")
    source_tags = sorted(schedule_sources)
    if (
        manifest.get("source_tags") != source_tags
        or any(
            not isinstance(tag, str)
            or not re.fullmatch(r"[A-Za-z0-9_]+", tag)
            for tag in source_tags
        )
    ):
        raise ValueError("RLR manifest source identity set changed")
    raw_spec_sources = spec.get("sources", [])
    if not isinstance(raw_spec_sources, list) or not all(
        isinstance(source, Mapping) for source in raw_spec_sources
    ):
        raise ValueError("RLR spec sources are malformed")
    source_specs_by_tag = {
        source.get("tag"): source for source in raw_spec_sources
    }
    if (
        None in source_specs_by_tag
        or len(source_specs_by_tag) != len(raw_spec_sources)
        or any(tag not in source_specs_by_tag for tag in source_tags)
    ):
        raise ValueError("RLR spec source identity set changed")

    rendered_spec_descriptor = _validate_file_descriptor(
        manifest.get("spec", {}),
        label="RLR spec",
    )
    final_spec_payload = load_authenticated_file_bytes(Path(spec_path))
    if (
        len(final_spec_payload) != rendered_spec_descriptor["size_bytes"]
        or hashlib.sha256(final_spec_payload).hexdigest()
        != rendered_spec_descriptor["sha256"]
    ):
        raise ValueError("finalized RLR spec bytes changed")
    acoustic_mesh_descriptor = _validate_file_descriptor(
        manifest.get("acoustic_mesh", {}),
        label="RLR acoustic mesh",
    )
    acoustic_materials_descriptor = _validate_file_descriptor(
        manifest.get("acoustic_materials", {}),
        label="RLR acoustic materials",
    )
    derived_materials_descriptor = _validate_file_descriptor(
        manifest.get("derived_rlr_materials", {}),
        label="RLR derived materials",
    )
    if __package__:
        from .acoustic_scene_contract import approved_acoustic_scene_contract
        from .rlr_materials import build_rlr_materials_payload
    else:  # pragma: no cover - direct-script consumers
        from acoustic_scene_contract import approved_acoustic_scene_contract
        from rlr_materials import build_rlr_materials_payload
    approved_scene = approved_acoustic_scene_contract(spec)
    if manifest.get("acoustic_scene_contract") != approved_scene:
        raise ValueError("RLR acoustic scene approval contract changed")
    for label, descriptor in (
        ("acoustic_mesh", acoustic_mesh_descriptor),
        ("acoustic_materials", acoustic_materials_descriptor),
    ):
        expected = approved_scene[label]
        if (
            descriptor["sha256"] != expected["sha256"]
            or descriptor["size_bytes"] != expected["size_bytes"]
        ):
            raise ValueError(f"RLR {label} is not the approved scene input")
    materials_payload = load_authenticated_file_bytes(
        Path(acoustic_materials_descriptor["path"])
    )
    try:
        source_materials = json.loads(materials_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("RLR acoustic materials are not valid JSON") from error
    expected_derived_payload = build_rlr_materials_payload(source_materials)
    observed_derived_payload = load_authenticated_file_bytes(
        Path(derived_materials_descriptor["path"])
    )
    if observed_derived_payload != expected_derived_payload:
        raise ValueError(
            "RLR derived materials do not match the authenticated source sidecar"
        )
    derived_path = Path(derived_materials_descriptor["path"])
    derived_mode = stat.S_IMODE(derived_path.stat().st_mode)
    parent_mode = stat.S_IMODE(derived_path.parent.stat().st_mode)
    if (
        not portable_artifact_resolution
        and (derived_mode & 0o222 or parent_mode & 0o222)
    ):
        raise ValueError("RLR derived materials are not an immutable staged snapshot")
    schedule_descriptor = _validate_file_descriptor(
        manifest.get("source_schedule", {}),
        label="RLR source schedule",
        expected_path=schedule_path,
    )
    output_descriptor = _validate_file_descriptor(
        manifest.get("output_wav", {}),
        label="RLR output WAV",
        expected_path=audio_path,
    )

    per_source = manifest.get("per_source_outputs")
    if not isinstance(per_source, Mapping) or set(per_source) != set(source_tags):
        raise ValueError("RLR per-source output identity set changed")
    solo_results = {}
    solo_descriptors = {}
    dry_descriptors = {}
    replayed_raw_sources = {}
    recomputed_azimuths = _recompute_mic_local_azimuths(
        spec=spec,
        spec_path=Path(rendered_spec_descriptor["path"]),
        source_tags=source_tags,
        n_frames=n_frames,
    )
    for tag in source_tags:
        record = per_source[tag]
        if not isinstance(record, Mapping):
            raise ValueError(f"RLR per-source output is malformed for {tag!r}")
        peak = record.get("pre_normalization_peak")
        if (
            not isinstance(peak, (int, float))
            or isinstance(peak, bool)
            or not math.isfinite(float(peak))
            or float(peak) <= 0.0
        ):
            raise ValueError(f"RLR per-source output peak is invalid for {tag!r}")
        expected_solo = Path(audio_path).with_name(
            f"{Path(audio_path).stem}_{tag}_binaural.wav"
        )
        solo_descriptor = _validate_file_descriptor(
            record.get("binaural", {}),
            label=f"RLR per-source WAV {tag!r}",
            expected_path=expected_solo,
        )
        solo_descriptors[tag] = solo_descriptor
        expected_dry = Path(audio_path).with_name(
            f"{Path(audio_path).stem}_{tag}_scheduled_dry.wav"
        )
        dry_descriptor = _validate_file_descriptor(
            record.get("scheduled_dry", {}),
            label=f"RLR scheduled dry WAV {tag!r}",
            expected_path=expected_dry,
        )
        dry_descriptors[tag] = dry_descriptor
        rir_descriptor = _validate_file_descriptor(
            record.get("active_frame_rir", {}),
            label=f"RLR active-frame RIR evidence {tag!r}",
        )
        source = schedule_sources[tag]
        source_spec = source_specs_by_tag[tag]
        animal_species = animal_species_for_source(tag, source_spec)
        windows = None
        if animal_species is not None:
            windows = _validated_animal_event_windows(
                tag=tag,
                source=source,
                sample_rate_hz=sample_rate_hz,
                duration_s=duration_s,
            )
        solo_results[tag] = validate_binaural_wav(
            expected_solo,
            sample_rate_hz=sample_rate_hz,
            duration_s=duration_s,
            expected_sha256=solo_descriptor["sha256"],
            expected_size_bytes=solo_descriptor["size_bytes"],
            event_windows=windows,
        )
        if animal_species is not None:
            if __package__:
                from .scheduled_dry_evidence import (
                    validate_scheduled_dry_evidence,
                )
            else:  # pragma: no cover - direct-script consumers
                from scheduled_dry_evidence import (
                    validate_scheduled_dry_evidence,
                )
            scheduled_dry_validation = validate_scheduled_dry_evidence(
                scheduled_wav_descriptor=dry_descriptor,
                source_contract=source["source_contract"],
                schedule=source,
            )
            solo_results[tag]["scheduled_dry"] = scheduled_dry_validation
            replayed_raw, replay_result = (
                _validate_active_frame_rir_source_replay(
                    tag=tag,
                    spec=spec,
                    spec_path=Path(rendered_spec_descriptor["path"]),
                    source_spec=source_spec,
                    schedule_source=source,
                    rir_descriptor=rir_descriptor,
                    solo_path=expected_solo,
                    solo_descriptor=solo_descriptor,
                    reported_peak=float(peak),
                    sample_rate_hz=sample_rate_hz,
                    duration_s=duration_s,
                    n_frames=n_frames,
                    fps=fps,
                )
            )
            replayed_raw_sources[tag] = replayed_raw
            solo_results[tag]["active_frame_rir_replay"] = replay_result
            _require_reported_azimuth_matches_spec(
                record.get("mic_local_azimuth_deg_per_frame"),
                recomputed_azimuths[tag],
                tag=tag,
            )
            solo_results[tag]["spatial_ild"] = (
                _validate_band_limited_binaural_ild(
                    expected_solo,
                    descriptor=solo_descriptor,
                    azimuth_deg_per_frame=recomputed_azimuths[tag],
                    event_windows=windows,
                    sample_rate_hz=sample_rate_hz,
                    duration_s=duration_s,
                    n_frames=n_frames,
                    fps=fps,
                )
            )
    mix_peak = manifest.get("mix_pre_normalization_peak")
    if (
        not isinstance(mix_peak, (int, float))
        or isinstance(mix_peak, bool)
        or not math.isfinite(float(mix_peak))
        or float(mix_peak) <= 0.0
    ):
        raise ValueError("RLR mixed pre-normalization peak is invalid")
    mixture_reconstruction = _validate_binaural_mixture_reconstruction(
        audio_path,
        output_descriptor=output_descriptor,
        per_source_records=per_source,
        solo_descriptors=solo_descriptors,
        mix_pre_normalization_peak=float(mix_peak),
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
    )
    active_frame_rir_mix_replay = _validate_active_frame_rir_final_mix(
        replayed_sources=replayed_raw_sources,
        audio_path=audio_path,
        output_descriptor=output_descriptor,
        reported_mix_peak=float(mix_peak),
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
    )
    return {
        "spec": dict(manifest["spec"]),
        "schedule": schedule_descriptor,
        "output_wav": output_descriptor,
        "solo_wavs": solo_results,
        "mixture_reconstruction": mixture_reconstruction,
        "active_frame_rir_mix_replay": active_frame_rir_mix_replay,
    }


def _artifact_override_descriptor(
    *,
    key: str,
    override: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    if not isinstance(override, Mapping):
        raise ValueError(f"audio artifact override {key!r} must be an object")
    raw_path = override.get("path")
    sha256 = override.get("sha256")
    size_bytes = override.get("size_bytes")
    if (
        not isinstance(raw_path, str)
        or not Path(raw_path).is_absolute()
        or not isinstance(sha256, str)
        or not _SHA256_RE.fullmatch(sha256)
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
        or sha256 != expected.get("sha256")
        or size_bytes != expected.get("size_bytes")
    ):
        raise ValueError(
            f"audio artifact override {key!r} disagrees with its manifest"
        )
    path = Path(os.path.abspath(raw_path))
    payload = load_authenticated_file_bytes(path)
    if (
        len(payload) != size_bytes
        or hashlib.sha256(payload).hexdigest() != sha256
    ):
        raise ValueError(f"audio artifact override {key!r} bytes changed")
    return {
        "path": str(path),
        "sha256": sha256,
        "size_bytes": size_bytes,
    }, payload


def _resolve_animal_audio_artifact_overrides(
    *,
    manifest: Mapping[str, Any],
    schedule: Mapping[str, Any],
    overrides: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, bytes]]:
    """Resolve a complete portable artifact set into one immutable byte view."""
    if not isinstance(manifest, Mapping) or not isinstance(schedule, Mapping):
        raise ValueError("portable audio manifest/schedule must be objects")
    if not isinstance(overrides, Mapping):
        raise ValueError("audio artifact overrides must be an object")
    resolved_manifest = copy.deepcopy(dict(manifest))
    source_tags = resolved_manifest.get("source_tags")
    per_source = resolved_manifest.get("per_source_outputs")
    schedule_sources = schedule.get("sources")
    if (
        not isinstance(source_tags, list)
        or not all(isinstance(tag, str) for tag in source_tags)
        or not isinstance(per_source, dict)
        or set(per_source) != set(source_tags)
        or not isinstance(schedule_sources, Mapping)
        or set(schedule_sources) != set(source_tags)
    ):
        raise ValueError("portable audio source identities are malformed")

    descriptor_targets: dict[str, dict[str, Any]] = {
        "spec": resolved_manifest["spec"],
        "acoustic_mesh": resolved_manifest["acoustic_mesh"],
        "acoustic_materials": resolved_manifest["acoustic_materials"],
        "derived_rlr_materials": resolved_manifest[
            "derived_rlr_materials"
        ],
        "source_schedule": resolved_manifest["source_schedule"],
        "output_wav": resolved_manifest["output_wav"],
    }
    for tag in source_tags:
        record = per_source[tag]
        if not isinstance(record, dict):
            raise ValueError(
                f"portable per-source artifact record is malformed for {tag!r}"
            )
        for field in ("binaural", "scheduled_dry", "active_frame_rir"):
            descriptor_targets[f"source:{tag}:{field}"] = record[field]

    pinned_targets: dict[str, Mapping[str, Any]] = {}
    for tag in source_tags:
        source = schedule_sources[tag]
        if not isinstance(source, Mapping):
            raise ValueError(
                f"portable schedule source is malformed for {tag!r}"
            )
        source_contract = source.get("source_contract")
        if isinstance(source_contract, Mapping):
            pinned_targets[f"source:{tag}:pinned_dry"] = source_contract

    required_keys = set(descriptor_targets) | set(pinned_targets)
    if set(overrides) != required_keys:
        raise ValueError(
            "audio artifact override set changed: "
            f"missing={sorted(required_keys - set(overrides))}, "
            f"extra={sorted(set(overrides) - required_keys)}"
        )

    payload_snapshot: dict[str, bytes] = {}
    resolved_descriptors: dict[str, dict[str, Any]] = {}
    for key, target in descriptor_targets.items():
        old_path = str(Path(os.path.abspath(str(target["path"]))))
        descriptor, payload = _artifact_override_descriptor(
            key=key,
            override=overrides[key],
            expected=target,
        )
        target["path"] = descriptor["path"]
        resolved_descriptors[key] = descriptor
        payload_snapshot[old_path] = payload
        payload_snapshot[descriptor["path"]] = payload
    for key, target in pinned_targets.items():
        old_path = str(Path(os.path.abspath(str(target["path"]))))
        descriptor, payload = _artifact_override_descriptor(
            key=key,
            override=overrides[key],
            expected=target,
        )
        resolved_descriptors[key] = descriptor
        payload_snapshot[old_path] = payload
        payload_snapshot[descriptor["path"]] = payload

    try:
        resolved_spec = json.loads(
            payload_snapshot[resolved_descriptors["spec"]["path"]].decode(
                "utf-8"
            )
        )
        resolved_schedule = json.loads(
            payload_snapshot[
                resolved_descriptors["source_schedule"]["path"]
            ].decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("portable spec/schedule override is not valid JSON") from error
    if not isinstance(resolved_spec, dict) or not isinstance(
        resolved_schedule, dict
    ):
        raise ValueError("portable spec/schedule override must be an object")
    return (
        resolved_spec,
        resolved_schedule,
        resolved_manifest,
        payload_snapshot,
    )


def resolve_registered_animal_audio_artifacts(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    root_paths: Mapping[str, Path],
    action: str,
    tag: str,
) -> dict[str, dict[str, Any]]:
    """Map source_asset roles into portable validator overrides.

    ``root_paths`` is supplied by the registration consumer.  Artifact paths
    must be relative POSIX paths and every component is subsequently opened by
    the shared O_NOFOLLOW reader; root or child symlinks therefore fail closed.
    """
    if action not in {"Walking", "Idle"}:
        raise ValueError(f"unsupported registered audio action: {action!r}")
    if not isinstance(tag, str) or not re.fullmatch(r"[A-Za-z0-9_]+", tag):
        raise ValueError("registered audio tag is malformed")
    prefix = f"apartment_{action.lower()}_"
    role_by_key = {
        "spec": f"{prefix}spec",
        "acoustic_mesh": f"{prefix}acoustic_mesh",
        "acoustic_materials": f"{prefix}acoustic_materials",
        "derived_rlr_materials": f"{prefix}derived_rlr_materials",
        "source_schedule": f"{prefix}audio_schedule",
        "output_wav": f"{prefix}binaural_audio",
        f"source:{tag}:binaural": f"{prefix}binaural_solo",
        f"source:{tag}:scheduled_dry": f"{prefix}scheduled_dry",
        f"source:{tag}:active_frame_rir": f"{prefix}active_frame_rir",
        f"source:{tag}:pinned_dry": "animal_audio_pinned_dry_source",
    }
    overrides: dict[str, dict[str, Any]] = {}
    for key, role in role_by_key.items():
        record = artifacts.get(role)
        if not isinstance(record, Mapping):
            raise ValueError(f"registered audio artifact role is missing: {role}")
        root_id = record.get("root_id")
        relative_text = record.get("path")
        if (
            not isinstance(root_id, str)
            or root_id not in root_paths
            or not isinstance(relative_text, str)
            or not relative_text
        ):
            raise ValueError(f"registered audio artifact root changed: {role}")
        relative = Path(relative_text)
        if (
            relative.is_absolute()
            or "\\" in relative_text
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(f"registered audio artifact path escaped: {role}")
        root = Path(os.path.abspath(os.fspath(root_paths[root_id])))
        if not root.is_absolute():
            raise ValueError(f"registered audio root must be absolute: {root_id}")
        path = root.joinpath(*relative.parts)
        payload = load_authenticated_file_bytes(path)
        sha256 = record.get("sha256")
        size_bytes = record.get("size_bytes")
        if (
            not isinstance(sha256, str)
            or not _SHA256_RE.fullmatch(sha256)
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
            or len(payload) != size_bytes
            or hashlib.sha256(payload).hexdigest() != sha256
        ):
            raise ValueError(f"registered audio artifact bytes changed: {role}")
        overrides[key] = {
            "path": str(path),
            "sha256": sha256,
            "size_bytes": size_bytes,
        }
    return overrides


def _validate_animal_audio_evidence_impl(
    *,
    spec: Mapping[str, Any],
    schedule: Mapping[str, Any],
    audio_path: Path,
    expected_tags: set[str] | None = None,
    audio_descriptor: Mapping[str, Any] | None = None,
    render_manifest: Mapping[str, Any] | None = None,
    spec_path: Path | None = None,
    schedule_path: Path | None = None,
    portable_artifact_resolution: bool = False,
) -> dict[str, Any]:
    """Close controlled animal evidence over spec, schedules, and WAV bytes."""
    raw_sources = spec.get("sources", [])
    if not isinstance(raw_sources, list) or not all(
        isinstance(source, Mapping) for source in raw_sources
    ):
        raise ValueError("audio source specs must be a list of objects")
    sources = []
    for source in raw_sources:
        mute_audio = source.get("mute_audio", False)
        if not isinstance(mute_audio, bool):
            raise ValueError("mute_audio must be boolean")
        if not mute_audio and source.get("audio_lookup") != "silent":
            sources.append(source)
    source_specs = {source.get("tag"): source for source in sources}
    if None in source_specs or len(source_specs) != len(sources):
        raise ValueError("audio source spec tags are missing or duplicated")
    required_tags = set(source_specs) if expected_tags is None else set(expected_tags)
    if set(source_specs) != required_tags:
        raise ValueError("audio source spec identity set changed")
    if set(schedule.get("sources", {})) != required_tags:
        raise ValueError("audio schedule/spec source identity set changed")
    if schedule.get("schema") != "rlr_audio_source_schedules_v1":
        raise ValueError("audio schedule schema changed")
    config = spec.get("audio_config", {})
    sample_rate_hz = int(config["sample_rate_hz"])
    duration_s = float(config["duration_s"])
    if (
        sample_rate_hz <= 0
        or not math.isfinite(duration_s)
        or duration_s <= 0.0
    ):
        raise ValueError("animal audio output configuration is invalid")

    contracts = {}
    animal_species_by_tag = {
        tag: animal_species_for_source(tag, source_specs[tag])
        for tag in sorted(required_tags)
    }
    event_windows: list[tuple[int, int]] = []
    for tag in sorted(required_tags):
        source_spec = source_specs.get(tag)
        if source_spec is None:
            raise ValueError(f"audio source spec is missing for {tag!r}")
        schedule_source = schedule["sources"].get(tag)
        if not isinstance(schedule_source, Mapping):
            raise ValueError(f"audio schedule source is malformed for {tag!r}")
        animal_species = animal_species_by_tag[tag]
        lookup = source_spec.get("audio_lookup")
        pinned_default = pinned_audio_lookup_for_source(tag, source_spec)
        if pinned_default is not None or lookup in PINNED_AUDIO_LOOKUPS:
            contracts[tag] = validate_pinned_schedule_source(
                tag,
                schedule_source,
                source_spec=source_spec,
            )
        elif animal_species is not None:
            raise ValueError(
                f"controlled animal source {tag!r} lacks an authenticated "
                "mono dry-source contract"
            )
        if animal_species is not None:
            if source_spec.get("strict_audio") is not True:
                raise ValueError(
                    f"controlled animal source {tag!r} must enable strict_audio"
                )
            event_windows.extend(
                _validated_animal_event_windows(
                    tag=tag,
                    source=schedule_source,
                    sample_rate_hz=sample_rate_hz,
                    duration_s=duration_s,
                )
            )
    render_evidence = None
    expected_sha256 = None
    expected_size_bytes = None
    if render_manifest is not None:
        if spec_path is None or schedule_path is None:
            raise ValueError(
                "RLR render manifest validation requires spec and schedule paths"
            )
        render_evidence = validate_audio_render_manifest(
            render_manifest,
            spec=spec,
            spec_path=spec_path,
            schedule=schedule,
            schedule_path=schedule_path,
            audio_path=audio_path,
            portable_artifact_resolution=portable_artifact_resolution,
        )
        expected_sha256 = render_evidence["output_wav"]["sha256"]
        expected_size_bytes = render_evidence["output_wav"]["size_bytes"]
    if audio_descriptor is not None:
        descriptor_sha256 = audio_descriptor.get("sha256")
        descriptor_size_bytes = audio_descriptor.get("size_bytes")
        if (
            expected_sha256 is not None
            and (
                descriptor_sha256 != expected_sha256
                or descriptor_size_bytes != expected_size_bytes
            )
        ):
            raise ValueError("binaural WAV descriptors disagree")
        expected_sha256 = descriptor_sha256
        expected_size_bytes = descriptor_size_bytes
    wav = validate_binaural_wav(
        audio_path,
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        event_windows=event_windows or None,
    )
    return {
        "contracts": contracts,
        "wav": wav,
        "render_manifest": render_evidence,
    }


def validate_animal_audio_evidence(
    *,
    spec: Mapping[str, Any],
    schedule: Mapping[str, Any],
    audio_path: Path,
    expected_tags: set[str] | None = None,
    audio_descriptor: Mapping[str, Any] | None = None,
    render_manifest: Mapping[str, Any] | None = None,
    spec_path: Path | None = None,
    schedule_path: Path | None = None,
    artifact_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate local evidence or one complete relocated artifact closure."""
    if artifact_overrides is None:
        return _validate_animal_audio_evidence_impl(
            spec=spec,
            schedule=schedule,
            audio_path=audio_path,
            expected_tags=expected_tags,
            audio_descriptor=audio_descriptor,
            render_manifest=render_manifest,
            spec_path=spec_path,
            schedule_path=schedule_path,
            portable_artifact_resolution=False,
        )
    if render_manifest is None:
        raise ValueError("artifact overrides require an RLR render manifest")
    (
        resolved_spec,
        resolved_schedule,
        resolved_manifest,
        payload_snapshot,
    ) = _resolve_animal_audio_artifact_overrides(
        manifest=render_manifest,
        schedule=schedule,
        overrides=artifact_overrides,
    )
    resolved_spec_path = Path(resolved_manifest["spec"]["path"])
    resolved_schedule_path = Path(
        resolved_manifest["source_schedule"]["path"]
    )
    resolved_audio_path = Path(resolved_manifest["output_wav"]["path"])
    resolved_audio_descriptor = None
    if audio_descriptor is not None:
        resolved_audio_descriptor = {
            **dict(audio_descriptor),
            "path": str(resolved_audio_path),
        }
    with _authenticated_payload_snapshot(payload_snapshot):
        return _validate_animal_audio_evidence_impl(
            spec=resolved_spec,
            schedule=resolved_schedule,
            audio_path=resolved_audio_path,
            expected_tags=expected_tags,
            audio_descriptor=resolved_audio_descriptor,
            render_manifest=resolved_manifest,
            spec_path=resolved_spec_path,
            schedule_path=resolved_schedule_path,
            portable_artifact_resolution=True,
        )


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
