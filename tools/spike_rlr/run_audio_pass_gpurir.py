"""A group: GPURIR baseline audio pass on shoebox v2 scene.

Uses:
  - Same SceneSpec/trajectory as B/C (scene_two_dogs_v2.compose_two_dog_scene_v2)
  - Same audio overrides as B (golden = Barking Aldi Dog, husky = 2 kHz vibrato tone)
  - Same 4-ch output topology (tetrahedral capsules; NOT FOA -- see caveat)
  - Same peak normalization scheme

The point of this pass is Gate 1: show that GPURIR, having no notion of the
sofa's existence, produces NO energy drop during the husky occlusion window
(t=3-5s), while B group RLR (which does model occlusion) produced a large
drop. This is the core "why RLR" evidence for the spike.

CAVEAT: A group tetra-mic and B group FOA are not the same downmix. For a
fair "did occlusion happen" comparison we compare monaural / total-energy
metrics, not the stereo downmix directly. The stereo downmix is only for
casual A/B listening.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_scene_and_registry():
    """Load scene_two_dogs_v2 + existing GPURIR pipeline modules."""
    sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from scene_two_dogs_v2 import compose_two_dog_scene_v2
    from gpurir_scenes.run_audio_pass import (
        FS, tetra_mic_positions, _load_source_wav,
        _simulate_traj_rirs, _convolve_moving, _convolve_static,
    )
    return {
        "compose": compose_two_dog_scene_v2,
        "FS": FS,
        "tetra_mic_positions": tetra_mic_positions,
        "load_source_wav": _load_source_wav,
        "sim_traj_rirs": _simulate_traj_rirs,
        "conv_moving": _convolve_moving,
        "conv_static": _convolve_static,
    }


# ---------- audio overrides (mirror run_audio_pass_rlr.py) -----------------
# Same overrides as B group so audio content is identical -> any energy/spatial
# difference is 100% due to the acoustic backend.

_TAG_AUDIO_OVERRIDES = {
    "dog_golden": "/data/datasets/omniaudio/train-data-az-360-large/Nightpianola_152.wav",
    "dog_husky":  "/data/datasets/omniaudio/train-data-az-360-large/Nightpianola_183.wav",
}


def _synth_hf_tone(sample_rate, duration_s, base_hz=2000.0, vibrato_hz=6.0,
                    vibrato_cents=40.0, amp=0.5):
    """Identical to run_audio_pass_rlr._synth_hf_tone; duplicated here so this
    script has no cross-import into the RLR script's heavier dep set (habitat)."""
    n = int(round(sample_rate * duration_s))
    t = np.arange(n, dtype=np.float32) / sample_rate
    vib_ratio = np.exp(np.log(2) / 1200.0 * vibrato_cents *
                        np.sin(2 * np.pi * vibrato_hz * t))
    inst_hz = base_hz * vib_ratio
    phase = 2 * np.pi * np.cumsum(inst_hz) / sample_rate
    return (amp * np.sin(phase)).astype(np.float32)


def _load_dry_source_for_gpurir(tag, sample_rate, duration_s, seed=42):
    """Same override table as B group RLR pipeline for audio-content parity."""
    override = _TAG_AUDIO_OVERRIDES.get(tag)
    n = int(round(sample_rate * duration_s))
    if override == "__hf_tone__":
        print(f"[gpurir] {tag}: SYNTHETIC 2kHz vibrato tone")
        return _synth_hf_tone(sample_rate, duration_s)
    if override and os.path.exists(override):
        print(f"[gpurir] {tag}: OVERRIDE {os.path.basename(override)}")
        # reuse the existing helper
        from gpurir_scenes.run_audio_pass import _load_source_wav
        return _load_source_wav(override, fs=sample_rate, duration_s=duration_s)
    # Fallback to registry
    from gpurir_scenes.audio_registry import pick_audio
    picked = pick_audio(tag, np.random.default_rng(seed))
    wav_path = picked[0] if isinstance(picked, tuple) else picked
    print(f"[gpurir] {tag}: REGISTRY {os.path.basename(wav_path)}")
    from gpurir_scenes.run_audio_pass import _load_source_wav
    return _load_source_wav(wav_path, fs=sample_rate, duration_s=duration_s)


def run_gpurir_pass(out_wav_path, verbose=True):
    mods = _import_scene_and_registry()
    scene = mods["compose"]()
    FS = mods["FS"]
    duration_s = 5.0
    n_samples = int(duration_s * FS)

    per_source_out = {}  # tag -> (n_samples, 4) tetra-4ch
    t0 = time.time()

    for placement in scene.animals:
        tag = placement.tag
        wav = _load_dry_source_for_gpurir(tag, FS, duration_s)
        print(f"[gpurir] {tag}: dry rms={np.sqrt(np.mean(wav**2)):.4f}, "
              f"peak={np.abs(wav).max():.4f}")

        if placement.is_animated:
            rirs = mods["sim_traj_rirs"](
                placement.trajectory_m,
                scene.mic_pos_m,
                scene.room_size_m,
                scene.t60_s,
            )
            mix = mods["conv_moving"](wav, rirs)
        else:
            rir = mods["sim_traj_rirs"](
                np.array([placement.static_pos_m], dtype=np.float64),
                scene.mic_pos_m, scene.room_size_m, scene.t60_s,
            )[0]
            mix = mods["conv_static"](wav, rir)

        if mix.shape[0] < n_samples:
            pad = np.zeros((n_samples - mix.shape[0], 4), dtype=np.float32)
            mix = np.concatenate([mix, pad], axis=0)
        else:
            mix = mix[:n_samples]
        per_source_out[tag] = mix.astype(np.float32)

    out_wav_path = Path(out_wav_path)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- per-source solo tracks (4ch tetra + stereo downmix) ----
    for tag, mix in per_source_out.items():
        solo_peak = float(np.abs(mix).max()) or 1.0
        mix_norm = mix * (0.9 / solo_peak)
        solo_4ch = out_wav_path.parent / f"{out_wav_path.stem}_{tag}_4ch.wav"
        sf.write(str(solo_4ch), mix_norm, FS, subtype="PCM_16")
        # legacy tetra->stereo downmix (matches existing SPEAR mux_audio_video)
        L = 0.5 * mix_norm[:, 0] + 0.5 * mix_norm[:, 2]
        R = 0.5 * mix_norm[:, 1] + 0.5 * mix_norm[:, 3]
        stereo = np.stack([L, R], axis=1)
        peak = np.abs(stereo).max() or 1.0
        stereo = stereo * (0.9 / peak)
        solo_stereo = out_wav_path.parent / f"{out_wav_path.stem}_{tag}_stereo.wav"
        sf.write(str(solo_stereo), stereo, FS, subtype="PCM_16")
        print(f"[gpurir] SOLO {solo_4ch.name} + {solo_stereo.name}")

    # ---- mixed track ----
    total = sum(per_source_out.values())
    peak = float(np.abs(total).max()) or 1.0
    total = (total * (0.9 / peak)).astype(np.float32)
    sf.write(str(out_wav_path), total, FS, subtype="PCM_16")
    # mixed stereo downmix
    L = 0.5 * total[:, 0] + 0.5 * total[:, 2]
    R = 0.5 * total[:, 1] + 0.5 * total[:, 3]
    stereo = np.stack([L, R], axis=1)
    peak = np.abs(stereo).max() or 1.0
    stereo = stereo * (0.9 / peak)
    stereo_path = out_wav_path.parent / f"{out_wav_path.stem.replace('4ch','')}stereo.wav"
    if not stereo_path.name.endswith('stereo.wav'):
        stereo_path = out_wav_path.parent / f"{out_wav_path.stem}_stereo.wav"
    sf.write(str(stereo_path), stereo, FS, subtype="PCM_16")

    elapsed = time.time() - t0
    print(f"[gpurir] wrote {out_wav_path}  shape={total.shape}")
    print(f"[gpurir] wrote MIXED stereo {stereo_path}")
    print(f"[gpurir] TOTAL wall time: {elapsed:.1f}s")
    return {"wall_time_s": elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "tmp" / "spike_output" / "raw_audio" / "audio_A_gpurir_4ch.wav"))
    args = ap.parse_args()
    run_gpurir_pass(args.out)


if __name__ == "__main__":
    main()
