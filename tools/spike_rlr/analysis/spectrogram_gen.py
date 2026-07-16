"""Spectrogram comparison figure for the A/B (and later C) backends.

Produces a multi-row figure:
  Row 1: A group (GPURIR) - stereo L+R spectrogram
  Row 2: B group (RLR)    - stereo L+R spectrogram
  Row 3: (later) C group  - stereo L+R spectrogram

Vertical dashed lines mark t=3.0s (occlusion enters) and t=4.0s (fully
occluded). The eye should immediately see that B row has energy loss and
high-freq attenuation in the [3, 5] window that A row does not.

Focuses on HUSKY solo (that's where the occlusion event lives).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import stft


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "tmp" / "spike_output" / "analysis"


def _load_stereo(path):
    y, sr = sf.read(str(path))
    if y.ndim == 1:
        y = np.stack([y, y], axis=1)
    return y, sr


def _spec_db(y, sr, nperseg=1024, noverlap=768):
    f, t, Z = stft(y, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=None)
    S = 20 * np.log10(np.abs(Z) + 1e-9)
    return f, t, S


def _plot_row(ax_L, ax_R, y, sr, title_prefix, vmin=-90, vmax=-15):
    f, t, S_L = _spec_db(y[:, 0], sr)
    _, _, S_R = _spec_db(y[:, 1], sr)
    im = ax_L.pcolormesh(t, f, S_L, cmap='magma', vmin=vmin, vmax=vmax, shading='auto')
    ax_L.set_title(f'{title_prefix}  |  Left channel', fontsize=10)
    ax_L.set_ylabel('Hz')
    ax_R.pcolormesh(t, f, S_R, cmap='magma', vmin=vmin, vmax=vmax, shading='auto')
    ax_R.set_title(f'{title_prefix}  |  Right channel', fontsize=10)
    # Occlusion markers
    for ax in (ax_L, ax_R):
        ax.axvline(3.0, color='cyan', linestyle='--', linewidth=1.2, alpha=0.8)
        ax.axvline(4.0, color='red',  linestyle='--', linewidth=1.2, alpha=0.8)
        ax.set_ylim(0, 8000)
    return im


def build_figure(backends, out_path, source_label="husky"):
    """backends: dict {name -> stereo_wav_path}."""
    n_rows = len(backends)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 3.5 * n_rows), squeeze=False)
    im = None
    for i, (name, path) in enumerate(backends.items()):
        y, sr = _load_stereo(path)
        im = _plot_row(axes[i, 0], axes[i, 1], y, sr, f'{name}  [{source_label}]')

    axes[-1, 0].set_xlabel('time (s)')
    axes[-1, 1].set_xlabel('time (s)')

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='dB')

    fig.suptitle(
        f'Spectrogram comparison  |  source: {source_label}  |  '
        f'cyan=occlusion begin (t=3.0s), red=full occlusion (t=4.0s)',
        fontsize=12, fontweight='bold', y=0.995,
    )
    plt.subplots_adjust(right=0.9, top=0.93, hspace=0.4)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"[spectrogram] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    raw = REPO_ROOT / "tmp" / "spike_output" / "raw_audio"
    out_dir = Path(args.out_dir)

    # HUSKY solo: the key comparison. Occlusion event lives here.
    build_figure({
        "A. SPEAR + GPURIR": raw / "audio_A_gpurir_4ch_dog_husky_stereo.wav",
        "B. SPEAR + RLR":    raw / "audio_B_rlr_FOA_dog_husky_stereo.wav",
    }, out_dir / "spectrogram_husky.png", source_label="husky (2 kHz vibrato tone)")

    # GOLDEN solo: sanity check that L/R azimuth works in both
    build_figure({
        "A. SPEAR + GPURIR": raw / "audio_A_gpurir_4ch_dog_golden_stereo.wav",
        "B. SPEAR + RLR":    raw / "audio_B_rlr_FOA_dog_golden_stereo.wav",
    }, out_dir / "spectrogram_golden.png", source_label="golden (dog bark)")

    # MIXED: what you actually hear in the final video
    build_figure({
        "A. SPEAR + GPURIR": raw / "audio_A_gpurir_stereo.wav",
        "B. SPEAR + RLR":    raw / "audio_B_rlr_stereo.wav",
    }, out_dir / "spectrogram_mixed.png", source_label="mixed (both dogs)")


if __name__ == "__main__":
    main()
