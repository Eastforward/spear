"""Plot running-RMS energy of the husky solo track vs time for each backend.

Shows two lines (A, B) on the same axes, sharing a t=3.0s (occlusion begins)
and t=4.0s (fully occluded) marker. The eye should see B curve dip
noticeably in [3, 5] while A stays flat (raw distance falloff only).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[3]


def _running_rms_db(y, sr, window_s=0.1):
    if y.ndim == 2:
        y = np.mean(y ** 2, axis=1)
    else:
        y = y ** 2
    win = int(sr * window_s)
    # Moving average via cumulative sum
    csum = np.cumsum(y)
    win_e = (csum[win:] - csum[:-win]) / win
    times = (np.arange(len(win_e)) + win / 2) / sr
    rms = np.sqrt(win_e)
    db = 20 * np.log10(rms + 1e-9)
    return times, db


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO_ROOT / "tmp" / "spike_output" / "analysis" / "ir_energy_curve.png"))
    args = ap.parse_args()

    raw = REPO_ROOT / "tmp" / "spike_output" / "raw_audio"

    fig, ax = plt.subplots(figsize=(12, 5))

    # Prefer howl-based tracks for Gate 1 comparison (broadband dramatic
    # signal); fall back to tone-based tracks if the howl re-run wasn't done.
    candidates = [
        ("A. SPEAR + GPURIR",  raw / "audio_A_gpurir_howl_4ch_dog_husky_4ch.wav",  "#c04040"),
        ("B. SPEAR + RLR",     raw / "audio_B_rlr_howl_FOA_dog_husky_FOA.wav",     "#2060c0"),
        ("A (tone). GPURIR",   raw / "audio_A_gpurir_4ch_dog_husky_4ch.wav",       "#e08888"),
        ("B (tone). RLR",      raw / "audio_B_rlr_FOA_dog_husky_FOA.wav",          "#6090e0"),
    ]
    for name, path, color in candidates:
        if not path.exists():
            continue
        y, sr = sf.read(str(path))
        t, db = _running_rms_db(y, sr, window_s=0.1)
        ax.plot(t, db, label=name, color=color, linewidth=1.5, alpha=0.85)

    # Occlusion markers
    ax.axvline(3.0, color='cyan', linestyle='--', linewidth=1.5,
               label='occlusion begin (t=3.0s)')
    ax.axvline(4.0, color='red', linestyle='--', linewidth=1.5,
               label='full occlusion (t=4.0s)')

    ax.set_xlabel('time (s)', fontsize=11)
    ax.set_ylabel('running RMS (dB)', fontsize=11)
    ax.set_title('Husky solo — energy vs time per backend '
                 '(occlusion window t=3-5s)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, 5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=9)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"[ir_energy] wrote {out}")


if __name__ == "__main__":
    main()
