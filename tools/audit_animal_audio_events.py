"""Resolve and energy-schedule every registered animal vocalization."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


SPEAR_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = SPEAR_ROOT / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

from audio_event_schedule import prepare_animal_call  # noqa: E402
from gpurir_scenes.audio_registry import pick_audio  # noqa: E402
from species_rig_map import ANIMATED_RIG_MAP, STATIC_MESH_MAP  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_mono(path: Path, sample_rate: int) -> np.ndarray:
    signal, source_rate = sf.read(path, dtype="float32", always_2d=False)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if int(source_rate) != int(sample_rate):
        divisor = int(np.gcd(source_rate, sample_rate))
        signal = resample_poly(
            signal,
            sample_rate // divisor,
            source_rate // divisor,
        ).astype(np.float32)
    return np.asarray(signal, dtype=np.float32)


def _canonical_species(tag: str) -> str:
    if tag.startswith("cat_"):
        return "cat"
    if tag.startswith("dog_"):
        return "dog"
    return tag


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--duration-s", type=float, default=15.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=7301)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tags = list(ANIMATED_RIG_MAP) + list(STATIC_MESH_MAP)
    if "dog_pug_pixal_canary_v2_100k" not in tags:
        tags.append("dog_pug_pixal_canary_v2_100k")
    rows = []
    for index, tag in enumerate(tags):
        pick_rng = np.random.default_rng(args.seed + index)
        audio_path, source_kind, keyword = pick_audio(tag, pick_rng)
        source_path = Path(audio_path).resolve()
        source_signal = _load_mono(source_path, args.sample_rate)
        scheduled, schedule = prepare_animal_call(
            source_signal,
            sample_rate=args.sample_rate,
            duration_s=args.duration_s,
            rng=np.random.default_rng(args.seed + 1000 + index),
        )
        preview_path = output_dir / f"{tag}_scheduled_dry.wav"
        peak = float(np.max(np.abs(scheduled))) if len(scheduled) else 0.0
        if peak > 1e-9:
            scheduled = scheduled * (0.8 / peak)
        sf.write(preview_path, scheduled, args.sample_rate, subtype="PCM_16")
        schedule.update(
            {
                "tag": tag,
                "source_kind": source_kind,
                "lookup_keyword": str(keyword),
                "source_path": str(source_path),
                "source_sha256": _sha256(source_path),
                "preview_path": str(preview_path),
                "preview_sha256": _sha256(preview_path),
            }
        )
        rows.append(
            {
                "tag": tag,
                "semantic_match": "provisional_species_specific_lookup",
                "source_kind": source_kind,
                "source_path": str(source_path),
                "license_status": (
                    "Stable Audio Open local license snapshot required"
                    if source_kind.startswith("sao")
                    else "local corpus item-level provenance/license review required"
                ),
                "registration_status": "research_candidate",
                "schedule": schedule,
            }
        )

    hashes_to_tags = {}
    for row in rows:
        source_hash = row["schedule"]["source_sha256"]
        hashes_to_tags.setdefault(source_hash, []).append(row["tag"])
    duplicate_groups = [
        tags
        for tags in hashes_to_tags.values()
        if len({_canonical_species(tag) for tag in tags}) > 1
    ]
    for row in rows:
        source_hash = row["schedule"]["source_sha256"]
        if any(row["tag"] in group for group in duplicate_groups):
            row["semantic_match"] = "rejected_duplicate_audio_across_species"

    legacy_cache_paths = {
        tag: SPEAR_ROOT / f"tmp/gpurir_scenes_v1/sao_cache/{tag}.wav"
        for tag in ("chipmunk", "yak", "donkey_ass")
    }
    legacy_cache_hashes = {
        tag: _sha256(path)
        for tag, path in legacy_cache_paths.items()
        if path.is_file()
    }

    manifest_path = output_dir / "animal_audio_audit_manifest.json"
    manifest = {
        "schema": "animal_audio_audit_v1",
        "duration_s": float(args.duration_s),
        "sample_rate_hz": int(args.sample_rate),
        "seed": int(args.seed),
        "asset_count": len(rows),
        "all_source_hashes_unique_across_species": not duplicate_groups,
        "duplicate_source_hash_tag_groups": duplicate_groups,
        "rejected_legacy_sao_cache": {
            "status": "rejected_duplicate_audio_across_species",
            "paths": {tag: str(path) for tag, path in legacy_cache_paths.items()},
            "sha256": legacy_cache_hashes,
            "all_three_identical": len(set(legacy_cache_hashes.values())) == 1,
        },
        "semantic_status": "provisional_pending_listening_or_audio_text_classifier",
        "formal_registration_blocked_on_item_level_audio_license": True,
        "assets": rows,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"ANIMAL_AUDIO_AUDIT_OK count={len(rows)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
