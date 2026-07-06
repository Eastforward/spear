"""Audio pass: SceneSpec -> 4-channel wav via GPURIR + audio_registry.

RUN WITH sao-env python (has gpuRIR + soundfile + scipy):
  /data/jzy/miniconda3/envs/sao-env/bin/python run_audio_pass.py <seed>
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpurir_scenes.audio_registry import pick_audio  # noqa: E402
from gpurir_scenes.scene_spec import compose_scene, N_FRAMES  # noqa: E402

import gpuRIR  # noqa: E402


FS = 16000
MIC_RADIUS_M = 0.042

# Tetrahedral mic layout: 4 omni capsules at corners of a tetrahedron on a
# sphere of radius MIC_RADIUS_M. Unit vectors from Spatial/v75 v77 layout.
_TETRA_RAW = np.array([
    [ 1.0,  1.0,  1.0],
    [ 1.0, -1.0, -1.0],
    [-1.0,  1.0, -1.0],
    [-1.0, -1.0,  1.0],
], dtype=np.float64)
TETRA_UNIT_SPHERE = _TETRA_RAW / np.linalg.norm(_TETRA_RAW[0])


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
    # Pad tail with silence (do NOT loop; user directive).
    out = np.zeros(n_out, dtype=np.float32)
    out[:len(x)] = x
    return out


def _simulate_traj_rirs(pos_traj_m, mic_pos_m, room_size_m, t60_s, tmax_s=0.5):
    """Return (n_pts, n_mic=4, n_samples) RIRs — one per trajectory anchor."""
    room = np.asarray(room_size_m, dtype=np.float64)
    beta = gpuRIR.beta_SabineEstimation(room, t60_s)
    mic_pts = tetra_mic_positions(mic_pos_m)
    nb_img = gpuRIR.t2n(tmax_s, room)
    rirs = gpuRIR.simulateRIR(
        room_sz=room,
        beta=beta,
        pos_src=np.asarray(pos_traj_m, dtype=np.float64),
        pos_rcv=mic_pts,
        nb_img=nb_img,
        Tmax=tmax_s,
        fs=FS,
    )
    return rirs


def _convolve_moving(source_wav, rirs, fs=FS):
    """Convolve mono source with (n_pts, 4, n_taps) trajectory -> (samples, 4)."""
    return gpuRIR.simulateTrajectory(source_wav.astype(np.float32), rirs, fs=fs)


def _convolve_static(source_wav, rir):
    """rir shape (4, n_taps) -> (samples, 4)."""
    import scipy.signal
    out = np.stack([scipy.signal.fftconvolve(source_wav, rir[ch]) for ch in range(4)], axis=1)
    return out


def run_audio_pass(spec, out_wav_path, rng):
    os.makedirs(os.path.dirname(out_wav_path) or ".", exist_ok=True)
    n_samples = int(5.0 * FS)
    per_source_out = []
    per_source_meta = []

    for placement in spec.animals:
        audio_path, audio_src, cls = pick_audio(placement.tag, rng)
        wav = _load_source_wav(audio_path, duration_s=5.0)
        if placement.is_animated:
            rirs = _simulate_traj_rirs(
                placement.trajectory_m, spec.mic_pos_m, spec.room_size_m, spec.t60_s,
            )
            mix = _convolve_moving(wav, rirs)
        else:
            rir = _simulate_traj_rirs(
                np.array([placement.static_pos_m], dtype=np.float64),
                spec.mic_pos_m, spec.room_size_m, spec.t60_s,
            )[0]  # (4, n_taps)
            mix = _convolve_static(wav, rir)
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
    peak = float(np.max(np.abs(total))) or 1.0
    total = (total / peak * 0.9).astype(np.float32)
    sf.write(out_wav_path, total, FS, subtype="PCM_16")
    return {"per_source": per_source_meta, "wav_path": out_wav_path, "shape": list(total.shape)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out-wav", default=None)
    args = p.parse_args()
    spec = compose_scene(seed=args.seed)
    _default_out_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tmp/gpurir_scenes_v1",
    )
    out_wav = args.out_wav or f"{_default_out_root}/scene_{args.seed:02d}/audio.wav"
    rng = np.random.default_rng(args.seed + 10000)
    meta = run_audio_pass(spec, out_wav, rng)
    print("META:")
    for src in meta["per_source"]:
        print(f"  {src['tag']:16s} src={src['audio_src']:16s} anim={src['is_animated']} path={src['audio_path']}")
    print(f"WROTE {out_wav}  shape={meta['shape']}")


if __name__ == "__main__":
    main()
