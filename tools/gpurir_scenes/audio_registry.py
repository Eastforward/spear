"""Per-animal audio picker.

Two-tier lookup:
  1) `omniaudio/train-data-az-360-large` — 58k .wav files with descriptive
     filenames. We match by keyword substring on filename. This gives us dog,
     cat, horse, cow, pig, sheep, goat and their sound-type variants (bark,
     meow, neigh, moo, bleat) all locally.
  2) Stable Audio Open 1.0 fallback — for the classes with no filename match
     (donkey, chipmunk, yak). Generated once, cached under sao_cache/.
"""
from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np

import os as _os
AUDIO_CORPUS = _os.environ.get(
    "AVENGINE_AUDIO_CORPUS", "/data/datasets/omniaudio/train-data-az-360-large"
)
_SPEAR_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
DEFAULT_SAO_CACHE = _os.path.join(_SPEAR_ROOT, "tmp/gpurir_scenes_v1/sao_cache")
GENERATED_SPECIES_V2 = {
    "chipmunk": _os.path.join(
        _SPEAR_ROOT,
        "tmp/animal_audio_event_audit_v1/generated_species_v2/chipmunk_seed7311.wav",
    ),
    "yak": _os.path.join(
        _SPEAR_ROOT,
        "tmp/animal_audio_event_audit_v1/generated_species_v2/yak_seed7321.wav",
    ),
    "donkey_ass": _os.path.join(
        _SPEAR_ROOT,
        "tmp/animal_audio_event_audit_v1/generated_species_v2/donkey_ass_seed7331.wav",
    ),
}

# Ordered filename-keyword candidates per tag. Case-insensitive substring
# match on the wav filename. First keyword with hits wins.
# Keyword ordering matters: earlier entries are tried first. We put the most
# animal-specific tokens (meow, bark, neigh, bleat) BEFORE ambiguous ones
# (cat, dog, cow) that would also match unrelated sounds ("cat eating",
# "cow bells", "wind howl"). Nothing here is perfect - the corpus is noisy -
# but this ordering biases us toward vocalizations over ambient captures.
# Keyword ordering: earlier entries win. Prefer species-specific vocalizations
# (meow, bark, neigh, bleat, oink) over ambient captures ("cat eating",
# "cow bells"). Word-boundary at keyword START avoids "moo"->"smooth" but
# ALLOWS "meow"->"meowing" (letters after keyword are fine).
# Each entry is a list of PATTERNS. A pattern is either:
#   - str: a single keyword (matched with word-boundary at start, allow suffix)
#   - tuple: (kw, must_also_contain) - both must appear in the filename;
#     used to disambiguate polysemous words ("neigh" -> "horse.neigh",
#     "howl" -> "wolf howl", "moo" -> "cow moo").
TAG_TO_KEYWORDS = {
    "cat_persian":    ["meow", "purr"],
    "cat_tabby":      ["meow"],
    "cat_british_shorthair_v2": ["meow", "purr"],
    "cat_siamese_v1": ["meow", "purr"],
    "chipmunk":       ["chipmunk"],                        # SAO fallback
    "dog_golden":     [("bark", "dog"), "woof"],
    "dog_beagle_v2":  [("bark", "dog"), "woof"],
    "dog_pug_v1":     [("bark", "dog"), "woof"],
    "dog_pug_pixal_canary_v1": [("bark", "dog"), "woof"],
    "dog_pug_pixal_canary_v2_100k": [("bark", "dog"), "woof"],
    "goat":           [("bleat", "goat"), "goat bleating"],
    "sheep":          [("bleat", "sheep"), "sheep bleating"],
    "pig":            [("oink", "pig"), "pig snort", ("pig", "snort")],
    "horse":          [("neigh", "horse"), "whinny"],
    "cattle_bovinae": [("moo", "cow"), "mooing", "cow"],
    "yak":            ["yak"],                             # SAO fallback
    "donkey_ass":     ["donkey", "bray"],                  # SAO fallback
}

# Text prompt for SAO fallback per tag.
TAG_TO_SAO_PROMPT = {
    "chipmunk":   "a chipmunk chirping and squeaking, close mic recording, clean",
    "yak":        "a yak bellowing and grunting, close mic recording, clean",
    "donkey_ass": "a donkey braying loudly, close mic recording, clean",
}


def _list_matches(pattern):
    """Return sorted list of absolute wav paths whose filename matches `pattern`.

    `pattern` is either:
      - str keyword: word-boundary match at start, letters allowed after
        ("meow" hits "meowing", "moo" does NOT hit "smooth").
      - (kw, must_also_contain) tuple: BOTH must appear in the filename
        (as substrings, case-insensitive). Used to disambiguate polysemous
        keywords like "neigh"/"howl"/"moo".
    """
    if isinstance(pattern, tuple):
        kw, also = pattern
        pat = re.compile(r"(?<![a-z])" + re.escape(kw.lower()), re.IGNORECASE)
        also_lc = also.lower()
        req_second = True
    else:
        kw = pattern
        pat = re.compile(r"(?<![a-z])" + re.escape(kw.lower()), re.IGNORECASE)
        req_second = False

    hits = []
    if not os.path.isdir(AUDIO_CORPUS):
        return hits
    for name in os.listdir(AUDIO_CORPUS):
        if not name.lower().endswith(".wav"):
            continue
        if pat.search(name) and (not req_second or also_lc in name.lower()):
            hits.append(os.path.join(AUDIO_CORPUS, name))
    hits.sort()
    return hits


def _lookup_local(tag, rng):
    for pat in TAG_TO_KEYWORDS.get(tag, []):
        hits = _list_matches(pat)
        if hits:
            path = str(rng.choice(hits))
            return path, str(pat)
    return None, None


HUNYUAN_PY = "/data/jzy/miniconda3/envs/sao-env/bin/python"
SAO_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gen_sao_clip.py")


def _generate_sao(tag, out_wav, seed=0):
    """Delegate SAO generation to hunyuan3d env (has torch + diffusers).
    Keeps spear-env torch-free to avoid breaking compiled spear_ext.
    """
    import subprocess
    prompt = TAG_TO_SAO_PROMPT.get(tag) or f"a {tag.replace('_', ' ')} animal sound, close mic recording, clean"
    subprocess.run(
        [HUNYUAN_PY, SAO_SCRIPT, prompt, out_wav, "--seed", str(seed)],
        check=True,
    )
    return out_wav


def pick_audio(tag, rng, sao_cache_dir=DEFAULT_SAO_CACHE):
    path, kw = _lookup_local(tag, rng)
    if path is not None:
        return path, "local", kw
    generated_v2 = GENERATED_SPECIES_V2.get(tag)
    if generated_v2 and os.path.isfile(generated_v2):
        return generated_v2, "sao_v2", TAG_TO_SAO_PROMPT.get(tag) or "generic"
    os.makedirs(sao_cache_dir, exist_ok=True)
    cached = os.path.join(sao_cache_dir, f"{tag}.wav")
    if not os.path.exists(cached):
        _generate_sao(tag, cached)
    return cached, "sao", "<sao>"


def audit_all_tags(rng_seed=0):
    """Enumerate every animal tag and print how it resolves. Does not generate SAO clips."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from species_rig_map import ANIMATED_RIG_MAP, STATIC_MESH_MAP

    rng = np.random.default_rng(rng_seed)
    rows = []
    for tag in list(ANIMATED_RIG_MAP.keys()) + list(STATIC_MESH_MAP.keys()):
        path, kw = _lookup_local(tag, rng)
        if path is not None:
            rows.append((tag, "local", kw, path))
        else:
            rows.append((tag, "SAO (fallback)", TAG_TO_SAO_PROMPT.get(tag) or "generic", ""))
    return rows


if __name__ == "__main__":
    for row in audit_all_tags():
        tag, src, kw, path = row
        print(f"{tag:16s} src={src:16s} kw={kw!r:40s} path={path}")
