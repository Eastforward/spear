"""Plot IPD & ILD spectrograms for the 3 binaural qualities side-by-side."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
BIN_DIR = REPO_ROOT / "tmp" / "spike_output" / "binaural"
OUT = REPO_ROOT / "tmp" / "spike_output" / "analysis" / "ipd_ild_binaural.png"


def ipd_ild(path):
    x, sr = sf.read(str(path), always_2d=True)
    L, R = x[:, 0], x[:, 1]
    n_fft, hop = 1024, 256
    fL = signal.stft(L, sr, nperseg=n_fft, noverlap=n_fft - hop)[2]
    fR = signal.stft(R, sr, nperseg=n_fft, noverlap=n_fft - hop)[2]
    ipd = np.angle(fL * np.conj(fR))
    ild = 20 * np.log10((np.abs(fL) + 1e-10) / (np.abs(fR) + 1e-10))
    freqs = np.linspace(0, sr / 2, ipd.shape[0])
    times = np.arange(ipd.shape[1]) * hop / sr
    return ipd, ild, freqs, times


def main():
    qualities = ["LOW", "HIGH", "MAX"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharey=True)
    fig.suptitle(
        "HRTF-decoded binaural: IPD (top) & ILD (bottom) time-frequency maps\n"
        "3 quality levels of the same B-group scene "
        "(same husky detour behind sofa, same golden L→R)",
        fontsize=11, fontweight='bold',
    )
    for col, q in enumerate(qualities):
        path = BIN_DIR / f"audio_B_rlr_{q}_binaural.wav"
        ipd, ild, freqs, times = ipd_ild(path)
        # focus on 200Hz-8kHz where binaural cues matter
        band = (freqs >= 200) & (freqs <= 8000)

        im1 = axes[0, col].pcolormesh(times, freqs[band] / 1000, ipd[band],
                                       cmap='hsv', vmin=-np.pi, vmax=np.pi,
                                       shading='auto')
        axes[0, col].set_title(f"{q} ({{100,500,5000}}[{col}]) rays  |  IPD (rad)", fontsize=9)
        axes[0, col].set_xlabel("time (s)")
        if col == 0:
            axes[0, col].set_ylabel("freq (kHz)")
        axes[0, col].set_yscale('log')
        axes[0, col].set_ylim(0.2, 8)

        im2 = axes[1, col].pcolormesh(times, freqs[band] / 1000, ild[band],
                                       cmap='RdBu_r', vmin=-15, vmax=15,
                                       shading='auto')
        axes[1, col].set_title(f"{q}  |  ILD (dB, red=L>R)", fontsize=9)
        axes[1, col].set_xlabel("time (s)")
        if col == 0:
            axes[1, col].set_ylabel("freq (kHz)")
        axes[1, col].set_yscale('log')
        axes[1, col].set_ylim(0.2, 8)

    fig.colorbar(im1, ax=axes[0, :].tolist(), fraction=0.02, pad=0.02, label='rad')
    fig.colorbar(im2, ax=axes[1, :].tolist(), fraction=0.02, pad=0.02, label='dB')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120, bbox_inches='tight')
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
