# GPURIR 10-Scene Animals + Audio Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render 10 randomized scenes each containing 1-2 of our 12 animals (5 animated, 7 static) into two rooms (apartment_0000 + GPURIR shoebox) with 4 fixed camera angles per scene, and a shared 4-channel GPURIR audio track built from real audioset sound clips (Stable Audio Open fallback for classes with no audioset match).

**Architecture:** A single scene-composer script generates 10 seeded scene specs (which animals, per-animal trajectory or static position). Each spec drives (a) two UE render passes — one in apartment, one in shoebox — each producing 4 view videos, and (b) a shared audio pass that generates per-source moving-source 4-channel GPURIR IRs, convolves them with per-species audioset clips (or SAO-generated ones), and mixes them into a single 4ch wav. Videos are mux'd with the audio (folding 4ch → stereo for playable mp4).

**Tech Stack:**
- SPEAR (`spear.Instance`, `render_in_apartment.py`, `render_animated_dog_gpurir.py`, `trajectory.py`)
- GPURIR (`gpuRIR.simulateRIR`, `gpuRIR.simulateTrajectory`)
- audioset via `/datasets/WavCaps/` manifests
- Stable Audio Open 1.0 via `diffusers` StableAudioPipeline for fallback classes
- ffmpeg for mux
- `/data/jzy/miniconda3/envs/spear-env/bin/python` interpreter

## Global Constraints

- Python interpreter: `/data/jzy/miniconda3/envs/spear-env/bin/python` (SPEAR env, py3.11). Do NOT use `thu` env.
- Room (both simulation + shoebox visual): fixed 5.2m × 4.4m × 2.8m. T60 = 0.45s. Do NOT randomize.
- Mic position: `(room_x/2, room_y/2, 1.2m)` = `(2.6, 2.2, 1.2)` in meters, world coords.
- World coord convention: **mic-forward is +Y (toward the window wall in shoebox)**. Camera main view (`yaw=0`) points at +Y. Camera right = +X = audio right. Camera left = -X = audio left. Behind = -Y = audio back. This is fixed; do NOT rotate the mic frame between scenes.
- 4 camera angles per scene per room: `yaw ∈ {0, 90, 180, 270}` degrees (values 0/90/180/270 exact; not randomized).
- Video: 640×480, 15 fps, 5 s → 75 frames.
- Audio: 16 kHz, 4 channel tetrahedral (v77 layout), mux'd to video after downmix to stereo for playability.
- Trajectory: `wall_margin_m = 0.5` (source stays ≥ 0.5 m from any wall).
- Static animal placement: random position with `wall_margin_m = 0.5` and `mic_margin_m = 1.0` (don't block camera), random yaw in [0, 360).
- Scene RNG: seeds 0-9. Each scene's RNG deterministically decides: how many animals (1 or 2), which tags, static-vs-animated is set by tag (5 animated tags fixed, 7 static tags fixed), each animal's trajectory or static pose, audio clip selection.
- Audio strategy per animal tag: (1) look up audioset class → find a random matching clip from `/datasets/WavCaps/`; (2) if no match, generate a 5s clip with Stable Audio Open 1.0. Print each match for user audit; abort if any lookup returns wrong class.
- Source audio duration: clip to 5s; if source is shorter, silence the tail (do NOT loop; user was explicit).
- Persistent test data lives at `/data/jzy/code/SPEAR/tools/gpurir_scenes/` (scripts), scenes output at `/data/jzy/code/SPEAR/tmp/gpurir_scenes_v1/scene_XX/`.
- Never commit HF token to git.

## Prerequisite state (verified 2026-07-06 during grilling)

- 5 animated animal BPs already cooked or will be by Task 6a of prior plan `2026-07-05-12-quadrupeds-species-rigs.md`:
  - `/Game/MyAssets/Audioset/Blueprints/gate_{cat_persian,cat_tabby,chipmunk,dog_golden,dog_husky}/BP_gate_{TAG}`
- 7 static ungulate meshes in Hunyuan audioset output dir; will need import → static mesh BP.
- GPURIR + gpuRIR wheels installed in spear-env (used by v77 pipeline).
- Stable Audio Open 1.0 requires diffusers ≥ 0.29 with StableAudioPipeline; will install in Task 3.

---

## File Structure

New files under `/data/jzy/code/SPEAR/tools/gpurir_scenes/`:

| File | Responsibility |
|------|----------------|
| `__init__.py` | package marker |
| `scene_spec.py` | pure-Python scene composer: seed → SceneSpec dataclass |
| `audio_registry.py` | tag → audioset class map + wavcaps lookup + SAO fallback wrapper |
| `run_audio_pass.py` | produce `audio.wav` (4ch) for one scene from its SceneSpec |
| `run_render_pass.py` | UE renderer: reads SceneSpec, spawns animals, captures 4 yaw views into apartment OR shoebox |
| `render_gate_animal_editor.py` | UE editor commandlet used to import the 7 static ungulate meshes as StaticMesh + BP wrapper |
| `mux_audio_video.py` | ffmpeg wrapper: 4 videos + 1 4ch wav → 4 videos with stereo audio track |
| `run_scene.py` | end-to-end for one scene: spec → audio → apartment renders → shoebox renders → mux |
| `run_all_scenes.py` | driver: run seeds 0-9 sequentially |

Modified files:
- Extends `species_rig_map.py` with `ANIMATED_RIG_MAP` and `STATIC_MESH_MAP` (already done in prior plan).

Test file:
- `SPEAR/tests/tools/test_gpurir_scenes.py` — unit tests for scene_spec, audio_registry lookup, and trajectory validation.

---

### Task 1: Scene spec composer (pure Python, no UE / no audio)

**Files:**
- Create: `/data/jzy/code/SPEAR/tools/gpurir_scenes/__init__.py` (empty)
- Create: `/data/jzy/code/SPEAR/tools/gpurir_scenes/scene_spec.py`
- Test: `/data/jzy/code/SPEAR/tests/tools/test_gpurir_scenes.py`

**Interfaces:**
- Consumes: `species_rig_map.ANIMATED_RIG_MAP`, `species_rig_map.STATIC_MESH_MAP`.
- Produces:
  - `SceneSpec` dataclass with fields:
    - `seed: int`
    - `room_size_m: tuple[float, float, float] = (5.2, 4.4, 2.8)`
    - `t60_s: float = 0.45`
    - `mic_pos_m: tuple[float, float, float] = (2.6, 2.2, 1.2)`
    - `animals: list[AnimalPlacement]` (length 1 or 2)
  - `AnimalPlacement` dataclass with fields:
    - `tag: str`
    - `is_animated: bool`
    - `trajectory_m: np.ndarray | None` (shape (75, 3), None if static)
    - `yaw_deg: np.ndarray | None` (shape (75,), None if static)
    - `static_pos_m: tuple[float, float, float] | None` (None if animated)
    - `static_yaw_deg: float | None` (None if animated)
  - `compose_scene(seed: int) -> SceneSpec` deterministic from seed.

- [ ] **Step 1: Write the failing tests**

Write `/data/jzy/code/SPEAR/tests/tools/test_gpurir_scenes.py`:

```python
"""Tests for gpurir_scenes.scene_spec."""
import numpy as np
import pytest

from SPEAR.tools.gpurir_scenes.scene_spec import (
    compose_scene, SceneSpec, AnimalPlacement,
    ANIMATED_TAGS, STATIC_TAGS,
)


def test_seed_reproducible():
    a = compose_scene(seed=0)
    b = compose_scene(seed=0)
    assert a.animals[0].tag == b.animals[0].tag
    assert len(a.animals) == len(b.animals)
    if a.animals[0].is_animated:
        np.testing.assert_allclose(a.animals[0].trajectory_m, b.animals[0].trajectory_m)


def test_animal_count_is_one_or_two():
    for seed in range(20):
        spec = compose_scene(seed=seed)
        assert 1 <= len(spec.animals) <= 2


def test_animated_static_tag_partition():
    assert set(ANIMATED_TAGS).isdisjoint(STATIC_TAGS)
    assert len(ANIMATED_TAGS) == 5
    assert len(STATIC_TAGS) == 7


def test_animated_has_trajectory_static_has_pos():
    for seed in range(10):
        spec = compose_scene(seed=seed)
        for a in spec.animals:
            if a.is_animated:
                assert a.trajectory_m is not None and a.trajectory_m.shape == (75, 3)
                assert a.yaw_deg is not None and a.yaw_deg.shape == (75,)
                assert a.static_pos_m is None
            else:
                assert a.static_pos_m is not None and len(a.static_pos_m) == 3
                assert a.static_yaw_deg is not None
                assert a.trajectory_m is None


def test_trajectory_within_wall_margin():
    for seed in range(20):
        spec = compose_scene(seed=seed)
        rx, ry, rz = spec.room_size_m
        for a in spec.animals:
            if a.trajectory_m is not None:
                xs = a.trajectory_m[:, 0]; ys = a.trajectory_m[:, 1]
                assert xs.min() >= 0.5 and xs.max() <= rx - 0.5
                assert ys.min() >= 0.5 and ys.max() <= ry - 0.5
            elif a.static_pos_m is not None:
                x, y, _z = a.static_pos_m
                assert 0.5 <= x <= rx - 0.5
                assert 0.5 <= y <= ry - 0.5
                # mic distance ≥ 1.0
                mx, my, _mz = spec.mic_pos_m
                assert ((x - mx) ** 2 + (y - my) ** 2) ** 0.5 >= 1.0


def test_uses_only_known_tags():
    known = set(ANIMATED_TAGS) | set(STATIC_TAGS)
    for seed in range(50):
        spec = compose_scene(seed=seed)
        for a in spec.animals:
            assert a.tag in known
```

- [ ] **Step 2: Verify tests fail**

Run: `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/tools/test_gpurir_scenes.py -v`
Expected: ImportError / collect error (module doesn't exist yet).

- [ ] **Step 3: Implement scene_spec.py**

Write `/data/jzy/code/SPEAR/tools/gpurir_scenes/__init__.py` (empty), then `/data/jzy/code/SPEAR/tools/gpurir_scenes/scene_spec.py`:

```python
"""Deterministic per-scene composition: seed -> SceneSpec.

Fixed conventions:
  - Room 5.2 x 4.4 x 2.8 m (constant across scenes).
  - Mic at room center, height 1.2 m. Mic-forward = +Y = window direction.
  - Each scene has 1 or 2 animals; distribution 50/50.
  - Animated animals get a smooth trajectory (10 anchor pts, cubic interp);
    static animals get a random position with wall + mic margins.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, "/data/jzy/code/SPEAR/tools")
from species_rig_map import ANIMATED_RIG_MAP, STATIC_MESH_MAP  # noqa: E402


ANIMATED_TAGS = list(ANIMATED_RIG_MAP.keys())
STATIC_TAGS = list(STATIC_MESH_MAP.keys())
ALL_TAGS = ANIMATED_TAGS + STATIC_TAGS

ROOM_SIZE_M = (5.2, 4.4, 2.8)
T60_S = 0.45
MIC_POS_M = (ROOM_SIZE_M[0] / 2.0, ROOM_SIZE_M[1] / 2.0, 1.2)
N_FRAMES = 75
FPS = 15
WALL_MARGIN_M = 0.5
MIC_MARGIN_M = 1.0
SOURCE_HEIGHT_M = 0.45  # dog-mouth-ish; audio source height
STATIC_ACTOR_Z_M = 0.0  # actor on floor

TRAJ_ANCHORS = 10


@dataclass
class AnimalPlacement:
    tag: str
    is_animated: bool
    trajectory_m: Optional[np.ndarray] = None
    yaw_deg: Optional[np.ndarray] = None
    static_pos_m: Optional[tuple] = None
    static_yaw_deg: Optional[float] = None


@dataclass
class SceneSpec:
    seed: int
    room_size_m: tuple = ROOM_SIZE_M
    t60_s: float = T60_S
    mic_pos_m: tuple = MIC_POS_M
    animals: list = field(default_factory=list)


def _sample_static_pos(rng, room_size_m, mic_pos_m):
    rx, ry, _ = room_size_m
    for _ in range(200):
        x = rng.uniform(WALL_MARGIN_M, rx - WALL_MARGIN_M)
        y = rng.uniform(WALL_MARGIN_M, ry - WALL_MARGIN_M)
        if ((x - mic_pos_m[0]) ** 2 + (y - mic_pos_m[1]) ** 2) ** 0.5 >= MIC_MARGIN_M:
            return (float(x), float(y), STATIC_ACTOR_Z_M)
    raise RuntimeError("could not sample a static pose within margins after 200 tries")


def _generate_trajectory(rng, room_size_m):
    """10 random anchors, cubic-spline to 75 frames."""
    rx, ry, _ = room_size_m
    anchors = np.empty((TRAJ_ANCHORS, 2))
    for i in range(TRAJ_ANCHORS):
        anchors[i, 0] = rng.uniform(WALL_MARGIN_M, rx - WALL_MARGIN_M)
        anchors[i, 1] = rng.uniform(WALL_MARGIN_M, ry - WALL_MARGIN_M)
    ts = np.linspace(0.0, 1.0, TRAJ_ANCHORS)
    tf = np.linspace(0.0, 1.0, N_FRAMES)
    cs_x = CubicSpline(ts, anchors[:, 0])
    cs_y = CubicSpline(ts, anchors[:, 1])
    xs = np.clip(cs_x(tf), WALL_MARGIN_M, rx - WALL_MARGIN_M)
    ys = np.clip(cs_y(tf), WALL_MARGIN_M, ry - WALL_MARGIN_M)
    zs = np.full(N_FRAMES, SOURCE_HEIGHT_M)
    traj = np.stack([xs, ys, zs], axis=1)
    # Yaw: tangent direction, forward-facing walk
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    yaw = np.degrees(np.arctan2(dy, dx))
    return traj, yaw


def compose_scene(seed: int) -> SceneSpec:
    rng = np.random.default_rng(seed)
    n_animals = int(rng.choice([1, 2]))
    tags = list(rng.choice(ALL_TAGS, size=n_animals, replace=False))
    animals = []
    for tag in tags:
        is_animated = tag in ANIMATED_TAGS
        if is_animated:
            traj, yaw = _generate_trajectory(rng, ROOM_SIZE_M)
            animals.append(AnimalPlacement(
                tag=str(tag), is_animated=True,
                trajectory_m=traj, yaw_deg=yaw,
            ))
        else:
            pos = _sample_static_pos(rng, ROOM_SIZE_M, MIC_POS_M)
            yaw = float(rng.uniform(0.0, 360.0))
            animals.append(AnimalPlacement(
                tag=str(tag), is_animated=False,
                static_pos_m=pos, static_yaw_deg=yaw,
            ))
    return SceneSpec(seed=seed, animals=animals)


if __name__ == "__main__":
    import json, sys as _s
    seed = int(_s.argv[1]) if len(_s.argv) > 1 else 0
    spec = compose_scene(seed=seed)
    out = {
        "seed": spec.seed,
        "room_size_m": list(spec.room_size_m),
        "t60_s": spec.t60_s,
        "mic_pos_m": list(spec.mic_pos_m),
        "animals": [
            {"tag": a.tag, "is_animated": a.is_animated,
             "static_pos_m": list(a.static_pos_m) if a.static_pos_m else None,
             "static_yaw_deg": a.static_yaw_deg,
             "trajectory_shape": list(a.trajectory_m.shape) if a.trajectory_m is not None else None}
            for a in spec.animals
        ],
    }
    print(json.dumps(out, indent=2))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `cd /data/jzy/code/SPEAR && /data/jzy/miniconda3/envs/spear-env/bin/python -m pytest tests/tools/test_gpurir_scenes.py -v`
Expected: 6 passed.

- [ ] **Step 5: Dry-run compose_scene**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python /data/jzy/code/SPEAR/tools/gpurir_scenes/scene_spec.py 0`
Expected: JSON with a valid animal tag and either trajectory or static_pos.

- [ ] **Step 6: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/gpurir_scenes/__init__.py tools/gpurir_scenes/scene_spec.py tests/tools/test_gpurir_scenes.py
git commit -m "gpurir_scenes: deterministic seeded scene composer"
```

---

### Task 2: audio_registry — tag → audioset class + wavcaps sampler

**Files:**
- Create: `/data/jzy/code/SPEAR/tools/gpurir_scenes/audio_registry.py`

**Interfaces:**
- Consumes: `/datasets/WavCaps/ontology.json`, `/datasets/WavCaps/audioset_train_strong.tsv`, `/datasets/WavCaps/json_files/AudioSet_SL/as_final.json` (verify existence in Step 2).
- Produces:
  - `TAG_TO_AUDIOSET_CLASS: dict[str, list[str]]` — for each animal tag, list of candidate audioset class names to search (first found wins).
  - `pick_audio(tag, rng) -> tuple[str, str]` returns `(audio_path, source_name)` where `source_name ∈ {"audioset", "sao"}`. Raises if neither works.

- [ ] **Step 1: Confirm wavcaps layout**

Run: `ls /datasets/WavCaps/ontology.json /datasets/WavCaps/audioset_train_strong.tsv /datasets/WavCaps/json_files/AudioSet_SL/as_final.json 2>&1 | head`

Expected: all three exist. If not, adjust paths in Step 3 to point at what's actually mounted; use `find /datasets/WavCaps -name "*.json" -maxdepth 3` to locate.

- [ ] **Step 2: Add tag→class mapping table**

Write `/data/jzy/code/SPEAR/tools/gpurir_scenes/audio_registry.py`:

```python
"""Per-animal audio picker: audioset first, SAO fallback."""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

WAVCAPS_ROOT = "/datasets/WavCaps"
ONTOLOGY_PATH = f"{WAVCAPS_ROOT}/ontology.json"
STRONG_TSV = f"{WAVCAPS_ROOT}/audioset_train_strong.tsv"
FLAC_ROOT = f"{WAVCAPS_ROOT}/Zip_files/AudioSet_SL/mnt/fast"

# Ordered candidate audioset class names for each animal tag.
# First one that hits a real clip wins.
TAG_TO_AUDIOSET_CLASS = {
    "cat_persian":    ["Cat", "Meow", "Purr"],
    "cat_tabby":      ["Cat", "Meow", "Purr"],
    "chipmunk":       ["Squeak", "Chirp, tweet", "Rodents, rats, mice"],
    "dog_golden":     ["Dog", "Bark", "Bow-wow"],
    "dog_husky":      ["Dog", "Howl", "Bark"],
    "goat":           ["Goat", "Bleat"],
    "sheep":          ["Sheep", "Bleat"],
    "pig":            ["Pig", "Oink"],
    "horse":          ["Horse", "Neigh, whinny"],
    "cattle_bovinae": ["Cattle, bovinae", "Moo"],
    "yak":            ["Cattle, bovinae", "Moo"],  # audioset has no yak
    "donkey_ass":     ["Donkey, ass", "Bray"],
}


def _load_ontology():
    with open(ONTOLOGY_PATH) as f:
        onto = json.load(f)
    return {entry["name"]: entry["id"] for entry in onto}


def _load_strong():
    """Return {audioset_class_id: [audio_id, ...]}."""
    idx = {}
    with open(STRONG_TSV) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cid = row.get("label")
            aid = row.get("segment_id") or row.get("audio_id")
            if not cid or not aid:
                continue
            idx.setdefault(cid, []).append(aid)
    return idx


_ONTOLOGY = None
_STRONG = None


def _ensure_loaded():
    global _ONTOLOGY, _STRONG
    if _ONTOLOGY is None:
        _ONTOLOGY = _load_ontology()
    if _STRONG is None:
        _STRONG = _load_strong()


def _find_flac(audio_id):
    """Locate a flac by its audioset id (walks FLAC_ROOT)."""
    # audio_ids look like "Y1a2b..." or "1a2b..."; try both, and either .flac
    aid = audio_id.lstrip("Y")
    for root, _dirs, files in os.walk(FLAC_ROOT):
        for f in files:
            if f.startswith(aid) and f.endswith(".flac"):
                return os.path.join(root, f)
    return None


def _lookup_audioset(tag, rng, max_attempts=8):
    _ensure_loaded()
    for cls_name in TAG_TO_AUDIOSET_CLASS.get(tag, []):
        if cls_name not in _ONTOLOGY:
            continue
        cid = _ONTOLOGY[cls_name]
        ids = _STRONG.get(cid, [])
        if not ids:
            continue
        # Sample up to max_attempts; return first that resolves to disk
        picks = rng.choice(ids, size=min(max_attempts, len(ids)), replace=False)
        for aid in picks:
            path = _find_flac(str(aid))
            if path:
                return path, cls_name
    return None, None


def _generate_sao(tag, out_wav, prompt=None):
    """Fallback: Stable Audio Open 1.0 generates a 5s clip."""
    import torch
    import torchaudio
    from diffusers import StableAudioPipeline
    prompt = prompt or f"a {tag.replace('_', ' ')} animal sound, clean recording, 5 seconds"
    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16,
    ).to("cuda")
    audio = pipe(prompt, num_inference_steps=100, audio_end_in_s=5.0).audios[0]
    # StableAudioPipeline returns (channels, samples) at 44100 Hz
    wav = audio.detach().cpu().float()
    torchaudio.save(out_wav, wav, sample_rate=44100)
    return out_wav


def pick_audio(tag, rng, sao_cache_dir="/data/jzy/code/SPEAR/tmp/gpurir_scenes_v1/sao_cache"):
    path, cls = _lookup_audioset(tag, rng)
    if path is not None:
        return path, "audioset", cls
    # SAO fallback (cached by tag to avoid re-generating)
    os.makedirs(sao_cache_dir, exist_ok=True)
    cached = os.path.join(sao_cache_dir, f"{tag}.wav")
    if not os.path.exists(cached):
        _generate_sao(tag, cached)
    return cached, "sao", "<sao>"


if __name__ == "__main__":
    from species_rig_map import ANIMATED_RIG_MAP, STATIC_MESH_MAP
    sys.path.insert(0, "/data/jzy/code/SPEAR/tools")
    rng = np.random.default_rng(0)
    for tag in list(ANIMATED_RIG_MAP.keys()) + list(STATIC_MESH_MAP.keys()):
        try:
            p, src, cls = pick_audio(tag, rng)
            print(f"{tag:20s} src={src:8s} cls={cls!r:30s} path={p}")
        except Exception as e:
            print(f"{tag:20s} FAIL {e}")
```

- [ ] **Step 3: Print the pick_audio audit table**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python /data/jzy/code/SPEAR/tools/gpurir_scenes/audio_registry.py`

Expected: table of 12 tags with either `src=audioset cls='Cat'` (etc.) or `src=sao`. Only chipmunk / yak likely fall through to SAO.

**STOP for user review** — show the table to the user for approval before proceeding to Task 3 (SAO install + generation).

- [ ] **Step 4: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/gpurir_scenes/audio_registry.py
git commit -m "gpurir_scenes: audio registry with audioset + SAO fallback"
```

---

### Task 3: Install Stable Audio Open 1.0 + pre-generate fallback clips

**Files:** none new; uses `audio_registry.py` from Task 2.

**Interfaces:**
- Consumes: `audio_registry.pick_audio` for tags that landed on SAO.
- Produces: cached wavs at `/data/jzy/code/SPEAR/tmp/gpurir_scenes_v1/sao_cache/{tag}.wav`.

- [ ] **Step 1: Install diffusers + torchaudio in spear-env**

Run:
```bash
/data/jzy/miniconda3/envs/spear-env/bin/pip install --upgrade "diffusers>=0.29" "transformers>=4.42" torchaudio soundfile
```

Expected: successful install. If diffusers already up, "Requirement already satisfied."

- [ ] **Step 2: Pre-download SAO checkpoint**

Run:
```bash
export HF_TOKEN=<REDACTED_HUGGINGFACE_TOKEN>
/data/jzy/miniconda3/envs/spear-env/bin/python -c "
from diffusers import StableAudioPipeline
import torch
p = StableAudioPipeline.from_pretrained('stabilityai/stable-audio-open-1.0', torch_dtype=torch.float16)
print('OK loaded')
"
```

Expected: model files download; print "OK loaded". If gated, browse to huggingface.co and accept license.

- [ ] **Step 3: Generate SAO clips only for tags without audioset match**

Run:
```bash
/data/jzy/miniconda3/envs/spear-env/bin/python /data/jzy/code/SPEAR/tools/gpurir_scenes/audio_registry.py 2>&1 | tee /tmp/audio_registry_audit.log
```

Expected: any tag showing `src=sao` triggers generation on first call and caches the wav.

- [ ] **Step 4: Verify SAO cache**

Run: `ls -la /data/jzy/code/SPEAR/tmp/gpurir_scenes_v1/sao_cache/*.wav`

Expected: at least chipmunk.wav (if audioset had no "Chipmunk"/"Squeak" match), each ≥ 400 KB.

Play one back locally to spot-check that it sounds like the intended class.

- [ ] **Step 5: Commit (no new source, just docs)**

```bash
cd /data/jzy/code/SPEAR
git commit --allow-empty -m "gpurir_scenes: SAO fallback verified for tags with no audioset match"
```

---

### Task 4: audio pass — trajectory → 4ch RIR → convolve → wav

**Files:**
- Create: `/data/jzy/code/SPEAR/tools/gpurir_scenes/run_audio_pass.py`

**Interfaces:**
- Consumes: `SceneSpec` (Task 1), `pick_audio` (Task 2), gpuRIR library.
- Produces:
  - `run_audio_pass(spec, out_wav_path, rng) -> dict` — writes 4-channel 16 kHz wav (shape samples×4). Returns metadata dict with per-source `audio_source_name`, `audioset_class`, `rir_shape`.

- [ ] **Step 1: Reference existing GPURIR wiring to make sure our API matches**

Run: `grep -nE "simulateRIR|simulateTrajectory|beta_Sabine" /data/jzy/code/Spatial/v77_4ch_S2L/data_gen/gen_rir_multiscene_v77.py | head`

Expected: lines using `beta_SabineEstimation(room_sz, t60)`, `simulateRIR(room_sz, beta, pos_src, pos_rcv, nb_img, Tmax, fs)`, and later `simulateTrajectory(waveform, RIRs)`. Copy these idioms in Step 2.

- [ ] **Step 2: Write run_audio_pass.py**

```python
"""Audio pass: SceneSpec -> 4-channel wav via GPURIR + audio_registry."""
from __future__ import annotations

import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, "/data/jzy/code/SPEAR/tools")
from gpurir_scenes.audio_registry import pick_audio  # noqa: E402

import gpuRIR  # noqa: E402

FS = 16000
MIC_RADIUS_M = 0.042
TETRA_UNIT_SPHERE = np.array([
    [0.5, 0.5, 0.5],
    [0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [-0.5, -0.5, 0.5],
], dtype=np.float64) / np.linalg.norm([0.5, 0.5, 0.5])


def tetra_mic_positions(center_m):
    return np.asarray(center_m, dtype=np.float64) + TETRA_UNIT_SPHERE * MIC_RADIUS_M


def _load_source_wav(path, fs=FS, duration_s=5.0):
    x, sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != fs:
        import scipy.signal
        n = int(len(x) * fs / sr)
        x = scipy.signal.resample(x, n)
    n_out = int(duration_s * fs)
    if len(x) >= n_out:
        return x[:n_out].astype(np.float32)
    # Pad with silence (do NOT loop; user was explicit)
    out = np.zeros(n_out, dtype=np.float32)
    out[:len(x)] = x
    return out


def _simulate_traj_rirs(pos_traj_m, mic_pos_m, room_size_m, t60_s, tmax_s=0.5):
    """Return RIRs shape (n_pts, 4, n_samples). One RIR per trajectory anchor."""
    beta = gpuRIR.beta_SabineEstimation(np.asarray(room_size_m), t60_s)
    mic_pts = tetra_mic_positions(mic_pos_m)
    nb_img = gpuRIR.t2n(tmax_s, np.asarray(room_size_m))
    rirs = gpuRIR.simulateRIR(
        room_sz=np.asarray(room_size_m),
        beta=beta,
        pos_src=np.asarray(pos_traj_m, dtype=np.float64),
        pos_rcv=mic_pts,
        nb_img=nb_img,
        Tmax=tmax_s,
        fs=FS,
    )
    return rirs  # (n_pos, n_mic=4, n_samples)


def _convolve_moving(source_wav, rirs, fs=FS):
    """Convolve mono source with moving-source RIR trajectory -> (samples, 4)."""
    return gpuRIR.simulateTrajectory(source_wav.astype(np.float32), rirs, fs=fs)


def _convolve_static(source_wav, rir):
    """rir shape (4, n_taps). Return (samples, 4)."""
    import scipy.signal
    out = np.stack([scipy.signal.fftconvolve(source_wav, rir[ch]) for ch in range(4)], axis=1)
    return out


def run_audio_pass(spec, out_wav_path, rng):
    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
    per_source_meta = []
    per_source_out = []
    n_frames = 75
    duration_s = 5.0
    n_samples = int(duration_s * FS)

    for placement in spec.animals:
        audio_path, audio_src, cls = pick_audio(placement.tag, rng)
        wav = _load_source_wav(audio_path, duration_s=duration_s)
        if placement.is_animated:
            rirs = _simulate_traj_rirs(
                placement.trajectory_m, spec.mic_pos_m, spec.room_size_m, spec.t60_s,
            )
            mix = _convolve_moving(wav, rirs)
        else:
            # Static: one anchor position, still call simulateRIR for a single point
            rir = _simulate_traj_rirs(
                np.array([placement.static_pos_m], dtype=np.float64),
                spec.mic_pos_m, spec.room_size_m, spec.t60_s,
            )[0]  # (4, n_taps)
            mix = _convolve_static(wav, rir)
        # Trim/pad to exact n_samples
        if mix.shape[0] < n_samples:
            pad = np.zeros((n_samples - mix.shape[0], 4), dtype=np.float32)
            mix = np.concatenate([mix, pad], axis=0)
        else:
            mix = mix[:n_samples]
        per_source_out.append(mix.astype(np.float32))
        per_source_meta.append({
            "tag": placement.tag, "audio_src": audio_src, "class": cls,
            "audio_path": audio_path, "is_animated": placement.is_animated,
        })

    total = np.sum(np.stack(per_source_out, axis=0), axis=0)
    # Peak-normalize to -1 dBFS
    peak = float(np.max(np.abs(total))) or 1.0
    total = (total / peak * 0.9).astype(np.float32)
    sf.write(out_wav_path, total, FS, subtype="PCM_16")
    return {"per_source": per_source_meta, "wav_path": out_wav_path, "shape": list(total.shape)}


if __name__ == "__main__":
    import sys as _s
    from gpurir_scenes.scene_spec import compose_scene
    seed = int(_s.argv[1]) if len(_s.argv) > 1 else 0
    out = f"/tmp/gpurir_scenes_v1/scene_{seed:02d}/audio.wav"
    spec = compose_scene(seed=seed)
    rng = np.random.default_rng(seed + 10000)
    meta = run_audio_pass(spec, out, rng)
    print("META:", meta)
```

- [ ] **Step 3: Dry-run seed 0 audio pass**

Run: `mkdir -p /tmp/gpurir_scenes_v1/scene_00 && /data/jzy/miniconda3/envs/spear-env/bin/python /data/jzy/code/SPEAR/tools/gpurir_scenes/run_audio_pass.py 0`

Expected: prints META including per_source list; wav written at `/tmp/gpurir_scenes_v1/scene_00/audio.wav`; `soxi` shows 4-channel 5s 16kHz PCM.

Verify:
```bash
sox --i /tmp/gpurir_scenes_v1/scene_00/audio.wav 2>&1 | grep -E "Channels|Sample Rate|Duration"
```

Expected: `Channels : 4`, `Sample Rate : 16000`, `Duration : 00:00:05.00`.

- [ ] **Step 4: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/gpurir_scenes/run_audio_pass.py
git commit -m "gpurir_scenes: audio pass using GPURIR + audioset/SAO clips"
```

---

### Task 5: Import 7 static ungulates into UE as static-mesh BPs

**Files:**
- Create: `/data/jzy/code/SPEAR/tools/gpurir_scenes/render_gate_animal_editor.py` (UE editor commandlet, handles STATIC MESH import)
- Create: `/data/jzy/code/SPEAR/tools/gpurir_scenes/build_static_meshes.sh` (driver)

**Interfaces:**
- Consumes: `STATIC_MESH_MAP` entries (mesh obj path + diffuse jpg path).
- Produces: `/Game/MyAssets/Audioset/Blueprints/gate_static_{tag}/BP_gate_static_{tag}.BP_gate_static_{tag}_C` for each of 7 tags.

- [ ] **Step 1: Adapt import_gate_animal_editor.py for StaticMesh flow**

Write `/data/jzy/code/SPEAR/tools/gpurir_scenes/render_gate_animal_editor.py`:

```python
"""Headless UE editor script: import a Hunyuan textured obj as StaticMesh + BP.

Reads env vars:
  STATIC_TAG   - species tag, e.g. "horse"
  STATIC_MESH  - absolute path to textured.obj (or .glb)
"""
import os, posixpath
import spear, unreal

TAG = os.environ["STATIC_TAG"]
MESH = os.environ["STATIC_MESH"]
MESH_DIR = f"/Game/MyAssets/Audioset/Meshes/gate_static_{TAG}"
BP_DIR = f"/Game/MyAssets/Audioset/Blueprints/gate_static_{TAG}"
BP_NAME = f"BP_gate_static_{TAG}"


def _make_or_clear(path):
    if unreal.EditorAssetLibrary.does_directory_exist(directory_path=path):
        assert unreal.EditorAssetLibrary.delete_directory(directory_path=path)
    assert unreal.EditorAssetLibrary.make_directory(directory_path=path)


def main():
    assert os.path.exists(MESH), MESH
    _make_or_clear(MESH_DIR)
    _make_or_clear(BP_DIR)

    task = unreal.AssetImportTask()
    task.set_editor_property("filename", MESH)
    task.set_editor_property("destination_path", MESH_DIR)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("automated", True)
    task.set_editor_property("save", True)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])

    static_mesh_path = None
    for ap in unreal.EditorAssetLibrary.list_assets(MESH_DIR, recursive=True):
        data = unreal.EditorAssetLibrary.find_asset_data(asset_path=ap)
        cls_name = str(data.get_editor_property("asset_class_path").get_editor_property("asset_name"))
        if cls_name == "StaticMesh":
            n = str(data.get_editor_property("asset_name"))
            pkg = str(data.get_editor_property("package_path"))
            static_mesh_path = posixpath.join(pkg, f"{n}.{n}")
            break
    assert static_mesh_path is not None, "no StaticMesh imported"

    bp_path = posixpath.join(BP_DIR, BP_NAME)
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path=bp_path):
        assert unreal.EditorAssetLibrary.delete_asset(asset_path_to_delete=bp_path)

    bp = spear.editor.create_blueprint_asset(
        asset_name=BP_NAME, package_dir=BP_DIR,
        parent_class=unreal.StaticMeshActor,
    )
    subobjs = spear.editor.get_subobject_descs_for_blueprint_asset(blueprint_asset=bp)
    sm_comp = None
    for so in subobjs:
        if isinstance(so["object"], unreal.StaticMeshComponent):
            sm_comp = so["object"]
            break
    assert sm_comp is not None
    sm_comp.set_static_mesh(new_mesh=unreal.load_asset(name=static_mesh_path))
    unreal.get_editor_subsystem(unreal.EditorAssetSubsystem).save_loaded_asset(asset_to_save=bp)
    spear.log(f"IMPORT_STATIC_OK tag={TAG} sm={static_mesh_path} bp={bp_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the driver shell script**

Write `/data/jzy/code/SPEAR/tools/gpurir_scenes/build_static_meshes.sh`:

```bash
#!/bin/bash
set -uo pipefail
SPEAR_DIR=/data/jzy/code/SPEAR
UE_DIR=/data/UE_5.5
PY=/data/jzy/miniconda3/envs/spear-env/bin/python

STATIC_TAGS="horse cattle_bovinae yak donkey_ass goat sheep pig"
for TAG in $STATIC_TAGS; do
    echo "########## $TAG"
    MESH=$($PY -c "
import sys; sys.path.insert(0, '$SPEAR_DIR/tools')
from species_rig_map import STATIC_MESH_MAP
print(STATIC_MESH_MAP['$TAG']['mesh'])
")
    MESH_UE_DIR="$SPEAR_DIR/cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Meshes/gate_static_${TAG}"
    BP_UE_DIR="$SPEAR_DIR/cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Blueprints/gate_static_${TAG}"
    BP_PATH="$BP_UE_DIR/BP_gate_static_${TAG}.uasset"
    rm -rf "$MESH_UE_DIR" "$BP_UE_DIR"
    STATIC_TAG="$TAG" STATIC_MESH="$MESH" \
        "$PY" "$SPEAR_DIR/tools/run_editor_script.py" \
        --script "$SPEAR_DIR/tools/gpurir_scenes/render_gate_animal_editor.py" \
        --unreal-engine-dir "$UE_DIR" \
        --launch-mode commandlet \
        || echo "(editor commandlet returned nonzero - checking BP presence)"
    if [ ! -f "$BP_PATH" ]; then
        echo "STATIC_FAIL $TAG: no BP at $BP_PATH"
        exit 1
    fi
    echo "STATIC_OK $TAG"
done

echo "=== UE cook ==="
"$PY" "$SPEAR_DIR/tools/run_uat.py" \
    --unreal-engine-dir $UE_DIR \
    --skip-cook-default-maps \
    -build -cook -stage -package -archive -pak

echo "BUILD_STATIC_MESHES_DONE"
```

Make executable:
```bash
chmod +x /data/jzy/code/SPEAR/tools/gpurir_scenes/build_static_meshes.sh
```

- [ ] **Step 3: Run the driver**

```bash
DISPLAY=:99 bash /data/jzy/code/SPEAR/tools/gpurir_scenes/build_static_meshes.sh 2>&1 | tee /tmp/build_static_meshes.log
```

Expected: 7 `STATIC_OK <tag>` lines and final `BUILD_STATIC_MESHES_DONE`. Cook may take 10-15 min.

- [ ] **Step 4: Verify all 7 BPs on disk**

Run:
```bash
for TAG in horse cattle_bovinae yak donkey_ass goat sheep pig; do
    P="/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Blueprints/gate_static_${TAG}/BP_gate_static_${TAG}.uasset"
    [ -f "$P" ] && echo "OK $TAG" || echo "MISS $TAG"
done
```

Expected: 7 `OK` lines.

- [ ] **Step 5: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/gpurir_scenes/render_gate_animal_editor.py tools/gpurir_scenes/build_static_meshes.sh
git commit -m "gpurir_scenes: import 7 static ungulate meshes as UE BPs"
```

---

### Task 6: render_pass — 4-yaw video capture in one room

**Files:**
- Create: `/data/jzy/code/SPEAR/tools/gpurir_scenes/run_render_pass.py`

**Interfaces:**
- Consumes: `SceneSpec`, existing SPEAR Instance / configure_instance / configure_gpurir_instance / camera & light helpers, animated BP path `/Game/.../gate_{tag}/BP_gate_{tag}_C` (from prior plan), static BP path `/Game/.../gate_static_{tag}/BP_gate_static_{tag}_C` (from Task 5).
- Produces: `run_render_pass(spec, room: str, out_dir: str)` → writes `view0.png/view1.png/view2.png/view3.png` sequences plus `view{0..3}.mp4`. `room ∈ {"apartment", "shoebox"}`.

- [ ] **Step 1: Sketch the renderer**

Write `/data/jzy/code/SPEAR/tools/gpurir_scenes/run_render_pass.py`:

```python
"""Render a scene into one of two rooms with 4 fixed camera yaws.

room="apartment"  -> apartment_0000 map via configure_instance()
room="shoebox"    -> shoebox 5.2x4.4x2.8 map via configure_gpurir_instance()

Camera is at world-space mic position (2.6, 2.2, 1.2) in shoebox coords, or
at a fixed viewpoint marker in apartment. Camera yaw sweeps {0, 90, 180, 270}.
Yaw=0 points +X in UE's coordinate system, which we align to world +Y via the
scene rotation applied when spawning actors (see Step 3).
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys

import cv2
import numpy as np
import spear

REPO = "/data/jzy/code/SPEAR"
sys.path.insert(0, os.path.join(REPO, "examples"))
sys.path.insert(0, os.path.join(REPO, "tools"))
from render_in_apartment import (  # noqa: E402
    APARTMENT_MAP, configure_instance, spawn_camera, read_frame,
)
from render_in_gpurir_room import (  # noqa: E402
    configure_gpurir_instance, spawn_sky, spawn_directional_light, spawn_point_light,
    compute_shoebox_room_layout, spawn_room_piece,
)
from gpurir_scenes.scene_spec import compose_scene, N_FRAMES, FPS  # noqa: E402


M2CM = 100.0
WIDTH = 640
HEIGHT = 480


def _bp_path(placement):
    if placement.is_animated:
        return f"/Game/MyAssets/Audioset/Blueprints/gate_{placement.tag}/BP_gate_{placement.tag}.BP_gate_{placement.tag}_C"
    return f"/Game/MyAssets/Audioset/Blueprints/gate_static_{placement.tag}/BP_gate_static_{placement.tag}.BP_gate_static_{placement.tag}_C"


def _spawn_room(game, room, room_size_m):
    """Build shoebox surfaces for the 'shoebox' room; no-op for apartment."""
    if room != "shoebox":
        return
    pieces = compute_shoebox_room_layout(room_size_m=room_size_m)
    for p in pieces:
        spawn_room_piece(game=game, piece=p)
    spawn_sky(game=game)
    spawn_directional_light(game=game, yaw_deg=-90.0, pitch_deg=-45.0, intensity_lux=8.0)


def _spawn_animal(game, placement, room_size_m):
    bp = game.unreal_service.load_class(uclass="AActor", name=_bp_path(placement))
    if placement.is_animated:
        p0 = placement.trajectory_m[0]
    else:
        p0 = placement.static_pos_m
    actor = game.unreal_service.spawn_actor(
        uclass=bp,
        location={"X": float(p0[0] * M2CM), "Y": float(p0[1] * M2CM), "Z": 0.0},
        spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
    )
    actor.SetActorScale3D(NewScale3D={"X": 0.3, "Y": 0.3, "Z": 0.3})
    if not placement.is_animated:
        actor.K2_SetActorLocationAndRotation(
            NewLocation={"X": float(p0[0] * M2CM), "Y": float(p0[1] * M2CM), "Z": 0.0},
            NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": float(placement.static_yaw_deg)},
            bSweep=False, bTeleport=True,
        )
    return actor


def _step_animal(actor, placement, frame_i):
    if not placement.is_animated:
        return
    p = placement.trajectory_m[frame_i]
    y = float(placement.yaw_deg[frame_i])
    actor.K2_SetActorLocationAndRotation(
        NewLocation={"X": float(p[0] * M2CM), "Y": float(p[1] * M2CM), "Z": 0.0},
        NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": y},
        bSweep=False, bTeleport=True,
    )


def run_render_pass(spec, room, out_dir):
    assert room in ("apartment", "shoebox")
    os.makedirs(out_dir, exist_ok=True)

    if room == "apartment":
        instance = configure_instance(rpc_port=39002)
    else:
        instance = configure_gpurir_instance(rpc_port=39002)
    game = instance.get_game()

    try:
        # Setup room (shoebox only) + camera + spawn actors
        with instance.begin_frame():
            _spawn_room(game, room, spec.room_size_m)
            cam, comp = spawn_camera(game=game, width=WIDTH, height=HEIGHT)
            actors = [_spawn_animal(game, a, spec.room_size_m) for a in spec.animals]
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(bPaused=False)
        with instance.end_frame():
            pass
        instance.step(num_frames=20)  # warm up anim state

        mic_x_cm = spec.mic_pos_m[0] * M2CM
        mic_y_cm = spec.mic_pos_m[1] * M2CM
        mic_z_cm = spec.mic_pos_m[2] * M2CM

        # We need one video per yaw. Simplest correct implementation: sweep
        # frames 4 times, once per yaw, resetting actor state each time.
        # (Actor animation state is regenerated by re-spawning; we simply
        # replay trajectory positions per pass — anim keeps looping.)
        for view_i, yaw in enumerate([0, 90, 180, 270]):
            for frame_i in range(N_FRAMES):
                with instance.begin_frame():
                    for actor, placement in zip(actors, spec.animals):
                        _step_animal(actor, placement, frame_i)
                    cam.K2_SetActorLocationAndRotation(
                        NewLocation={"X": mic_x_cm, "Y": mic_y_cm, "Z": mic_z_cm},
                        NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": float(yaw)},
                        bSweep=False, bTeleport=True,
                    )
                with instance.end_frame():
                    img = read_frame(comp)
                    cv2.imwrite(os.path.join(out_dir, f"view{view_i}_frame_{frame_i:04d}.png"), img)
            # Encode this view's video
            out_mp4 = os.path.join(out_dir, f"view{view_i}.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-framerate", str(FPS),
                "-i", os.path.join(out_dir, f"view{view_i}_frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_mp4,
            ], check=True, capture_output=True)
    finally:
        instance.close(force=True)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--room", required=True, choices=["apartment", "shoebox"])
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    spec = compose_scene(seed=args.seed)
    run_render_pass(spec, args.room, args.out_dir)
    print(f"RENDER_DONE {args.out_dir}")
```

- [ ] **Step 2: Dry-run one apartment pass for seed 0**

Run:
```bash
DISPLAY=:99 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/tools/gpurir_scenes/run_render_pass.py \
  --seed 0 --room apartment \
  --out-dir /tmp/gpurir_scenes_v1/scene_00/apartment
```

Expected: `RENDER_DONE ...`; 4 view mp4s each ~5s at 640x480; total 300 png frames on disk. Runtime ~2-3 min per view = 8-12 min.

- [ ] **Step 3: STOP for user review**

Show all 4 view mp4s to the user; confirm:
- Camera is at room center, not orbiting.
- Same actor(s) appear across views (only camera yaw differs).
- Animated actors visibly walking; static ones stationary.

Do NOT continue to Task 7 without approval.

- [ ] **Step 4: Dry-run shoebox pass**

Run:
```bash
DISPLAY=:99 VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  /data/jzy/miniconda3/envs/spear-env/bin/python \
  /data/jzy/code/SPEAR/tools/gpurir_scenes/run_render_pass.py \
  --seed 0 --room shoebox \
  --out-dir /tmp/gpurir_scenes_v1/scene_00/shoebox
```

Expected: same 4 view mp4s but inside a 5.2×4.4×2.8 empty room with visible walls.

- [ ] **Step 5: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/gpurir_scenes/run_render_pass.py
git commit -m "gpurir_scenes: 4-yaw render pass for apartment + shoebox"
```

---

### Task 7: mux_audio_video — attach shared audio to 4 view mp4s

**Files:**
- Create: `/data/jzy/code/SPEAR/tools/gpurir_scenes/mux_audio_video.py`

**Interfaces:**
- Consumes: `audio.wav` (4ch 16kHz PCM) from Task 4, `view{0..3}.mp4` from Task 6.
- Produces: `view{0..3}_with_audio.mp4` — video + downmixed stereo audio track.

- [ ] **Step 1: Implement mux**

Write:

```python
"""Downmix 4ch audio to stereo (ambisonic W±Y for L/R) and mux with videos."""
import argparse, os, subprocess


def _downmix_stereo(in_wav, out_wav):
    """Ambisonic-style downmix: L = W + Y, R = W - Y (channels 0 and 1)."""
    # ffmpeg pan filter: names channels FL/FR
    subprocess.run([
        "ffmpeg", "-y", "-i", in_wav,
        "-af", "pan=stereo|FL=0.5*c0+0.5*c1|FR=0.5*c0-0.5*c1",
        out_wav,
    ], check=True, capture_output=True)


def mux(view_mp4, stereo_wav, out_mp4):
    subprocess.run([
        "ffmpeg", "-y", "-i", view_mp4, "-i", stereo_wav,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest", out_mp4,
    ], check=True, capture_output=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-dir", required=True,
                   help="dir containing apartment/ and shoebox/ subdirs and audio.wav")
    args = p.parse_args()
    audio_wav = os.path.join(args.scene_dir, "audio.wav")
    stereo_wav = os.path.join(args.scene_dir, "audio_stereo.wav")
    _downmix_stereo(audio_wav, stereo_wav)
    for room in ("apartment", "shoebox"):
        room_dir = os.path.join(args.scene_dir, room)
        for i in range(4):
            v = os.path.join(room_dir, f"view{i}.mp4")
            out = os.path.join(room_dir, f"view{i}_with_audio.mp4")
            mux(v, stereo_wav, out)
    print(f"MUX_DONE {args.scene_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test mux for seed 0**

Run:
```bash
/data/jzy/miniconda3/envs/spear-env/bin/python /data/jzy/code/SPEAR/tools/gpurir_scenes/mux_audio_video.py \
  --scene-dir /tmp/gpurir_scenes_v1/scene_00
```

Expected: 8 `view{i}_with_audio.mp4` files (4 per room), each with an AAC stereo audio track. `ffprobe` reports both video and audio streams.

- [ ] **Step 3: STOP for user review — this is the first playable output**

Show user the 8 muxed mp4s. Confirm:
- `apartment/view0_with_audio.mp4` — the main-view audio-visual pairing.
- Audio L/R matches visual: when an animal walks toward +X (right of frame in view0), the sound should shift right in the stereo mix.

Do NOT continue to Task 8 without approval.

- [ ] **Step 4: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/gpurir_scenes/mux_audio_video.py
git commit -m "gpurir_scenes: 4ch->stereo downmix + video mux"
```

---

### Task 8: End-to-end per-scene driver

**Files:**
- Create: `/data/jzy/code/SPEAR/tools/gpurir_scenes/run_scene.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `/tmp/gpurir_scenes_v1/scene_{seed:02d}/` with `trajectory.json`, `audio.wav`, `audio_stereo.wav`, `apartment/view{0..3}[_with_audio].mp4`, `shoebox/view{0..3}[_with_audio].mp4`.

- [ ] **Step 1: Implement**

Write:

```python
"""End-to-end for one scene: spec -> audio -> renders -> mux."""
import argparse, json, os, subprocess, sys
import numpy as np

sys.path.insert(0, "/data/jzy/code/SPEAR/tools")
from gpurir_scenes.scene_spec import compose_scene
from gpurir_scenes.run_audio_pass import run_audio_pass


def _spec_to_json(spec):
    return {
        "seed": spec.seed, "room_size_m": list(spec.room_size_m), "t60_s": spec.t60_s,
        "mic_pos_m": list(spec.mic_pos_m),
        "animals": [
            {"tag": a.tag, "is_animated": a.is_animated,
             "static_pos_m": list(a.static_pos_m) if a.static_pos_m else None,
             "static_yaw_deg": a.static_yaw_deg,
             "trajectory_m": a.trajectory_m.tolist() if a.trajectory_m is not None else None,
             "yaw_deg": a.yaw_deg.tolist() if a.yaw_deg is not None else None}
            for a in spec.animals
        ],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-root", default="/tmp/gpurir_scenes_v1")
    args = p.parse_args()

    scene_dir = os.path.join(args.out_root, f"scene_{args.seed:02d}")
    os.makedirs(scene_dir, exist_ok=True)

    spec = compose_scene(seed=args.seed)
    with open(os.path.join(scene_dir, "trajectory.json"), "w") as f:
        json.dump(_spec_to_json(spec), f, indent=2)

    # Audio (fast, no UE)
    rng = np.random.default_rng(args.seed + 10000)
    audio_meta = run_audio_pass(spec, os.path.join(scene_dir, "audio.wav"), rng)
    with open(os.path.join(scene_dir, "audio_meta.json"), "w") as f:
        json.dump(audio_meta, f, indent=2)

    py = "/data/jzy/miniconda3/envs/spear-env/bin/python"
    env = dict(os.environ)
    env.setdefault("DISPLAY", ":99")
    env.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")

    # Render both rooms (sequential; single UE lock)
    for room in ("apartment", "shoebox"):
        subprocess.run([
            py, "/data/jzy/code/SPEAR/tools/gpurir_scenes/run_render_pass.py",
            "--seed", str(args.seed), "--room", room,
            "--out-dir", os.path.join(scene_dir, room),
        ], env=env, check=True)

    # Mux audio into every view mp4
    subprocess.run([
        py, "/data/jzy/code/SPEAR/tools/gpurir_scenes/mux_audio_video.py",
        "--scene-dir", scene_dir,
    ], check=True)

    print(f"SCENE_DONE {scene_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run seed 0 end-to-end**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python /data/jzy/code/SPEAR/tools/gpurir_scenes/run_scene.py --seed 0 2>&1 | tee /tmp/scene_00.log`

Expected: final line `SCENE_DONE /tmp/gpurir_scenes_v1/scene_00`; directory contains 8 muxed mp4s.

- [ ] **Step 3: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/gpurir_scenes/run_scene.py
git commit -m "gpurir_scenes: end-to-end per-scene driver"
```

---

### Task 9: Run all 10 scenes (seeds 0-9)

**Files:**
- Create: `/data/jzy/code/SPEAR/tools/gpurir_scenes/run_all_scenes.py`

**Interfaces:**
- Consumes: `run_scene.py` from Task 8.
- Produces: `/tmp/gpurir_scenes_v1/scene_{0..9}/`, aggregate `batch_summary.json`.

- [ ] **Step 1: Implement**

Write:

```python
"""Sequentially run scenes for seeds 0..9. Skips already-complete scenes."""
import argparse, json, os, subprocess, sys, time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--out-root", default="/tmp/gpurir_scenes_v1")
    args = p.parse_args()

    py = "/data/jzy/miniconda3/envs/spear-env/bin/python"
    summary = []
    for seed in args.seeds:
        sd = os.path.join(args.out_root, f"scene_{seed:02d}")
        marker = os.path.join(sd, "shoebox", "view3_with_audio.mp4")
        if os.path.exists(marker):
            summary.append({"seed": seed, "status": "cached", "path": sd})
            continue
        t0 = time.time()
        rc = subprocess.run([
            py, "/data/jzy/code/SPEAR/tools/gpurir_scenes/run_scene.py",
            "--seed", str(seed), "--out-root", args.out_root,
        ]).returncode
        summary.append({
            "seed": seed, "status": "ok" if rc == 0 else "fail",
            "seconds": round(time.time() - t0, 1), "path": sd,
        })
        with open(os.path.join(args.out_root, "batch_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        if rc != 0:
            print(f"[warn] seed {seed} failed, continuing")

    print("BATCH_DONE")
    for s in summary:
        print(f"  seed {s['seed']:2d} status={s['status']:6s} secs={s.get('seconds', 0)}  {s['path']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all 10 scenes**

Run: `/data/jzy/miniconda3/envs/spear-env/bin/python /data/jzy/code/SPEAR/tools/gpurir_scenes/run_all_scenes.py 2>&1 | tee /tmp/gpurir_scenes_batch.log`

Expected: 10 successful scenes; each takes ~20-25 min (audio 30s + apartment render 8-10 min + shoebox render 8-10 min + mux 30s). Total ~3.5 hours.

- [ ] **Step 3: Sanity spot check**

Run:
```bash
for i in 0 1 2 3 4 5 6 7 8 9; do
    sd=/tmp/gpurir_scenes_v1/scene_$(printf "%02d" $i)
    [ -f "$sd/apartment/view0_with_audio.mp4" ] && [ -f "$sd/shoebox/view0_with_audio.mp4" ] \
        && echo "OK  scene_$(printf %02d $i)" || echo "MIS scene_$(printf %02d $i)"
done
```

Expected: 10 OK lines.

- [ ] **Step 4: Commit**

```bash
cd /data/jzy/code/SPEAR
git add tools/gpurir_scenes/run_all_scenes.py
git commit -m "gpurir_scenes: batch driver for 10 seeded scenes"
```

- [ ] **Step 5: STOP for final user review**

Present the batch summary + example muxed videos (at minimum: scene_00 apartment/view0, shoebox/view0). Ask: "10 scenes complete. Please review a few videos and confirm audio-visual direction alignment is correct in the yaw=0 main view."

---

## Rollback plans

- **Audioset filesystem layout doesn't match** (Task 2 Step 1): update paths in `audio_registry.py`. If audioset is not mounted at all, force everything to SAO (all 12 tags will need a cache entry, adding ~10 min to Task 3).
- **GPURIR crashes on very short T60**: Increase Tmax in `_simulate_traj_rirs` from 0.5 to 1.0 s.
- **UE cook after Task 5 hangs**: use `pkill -9 -f UnrealEditor`; the plan's `rm -rf` prevents the stale-uasset lock.
- **SAO output too quiet / wrong class**: swap prompt in `_generate_sao` or accept-license on HF then re-run.
- **Ambisonic downmix sounds wrong**: replace `pan=stereo|FL=0.5*c0+0.5*c1|FR=0.5*c0-0.5*c1` with the equivalent B-format ACN->stereo mixing weights per convention.

## Self-review notes

- Rooms constant per user directive: Task 4 uses `spec.t60_s = 0.45`, `spec.room_size_m = (5.2, 4.4, 2.8)` from `scene_spec.py` constants. Scene-to-scene changes are only animals + trajectories + audio picks, not room acoustics.
- Camera yaw fixed to 0/90/180/270: Task 6 iterates `[0, 90, 180, 270]`. Main view yaw=0 is aligned to world +Y = mic-forward = window direction; audio L/R = world -X/+X; front/back = -Y/+Y.
- Same audio across 4 views + across 2 rooms: Task 8 runs audio pass once per scene; mux uses the same `audio_stereo.wav` for all 8 output mp4s.
- Static animals: Task 1 samples `static_pos_m` with wall+mic margins; Task 4 uses single-position convolution; Task 6 spawns without ticking a trajectory.
- Chipmunk / yak audioset fallback: Task 2 auto-routes to SAO when audioset lookup fails; Task 3 caches SAO wavs; Step 3 of Task 2 stops for user audit.
- User verification checkpoints: Task 2 (audio audit), Task 6 (first apartment render), Task 7 (first mux), Task 9 (final).
- Trajectory length: 5s@15fps = 75 frames, consistent across scene_spec, run_render_pass, and run_audio_pass.
