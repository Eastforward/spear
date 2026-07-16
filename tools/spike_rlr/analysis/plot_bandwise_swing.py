"""Band-wise L-R analysis + HP-filtered demo.

Shows that the spatial cues ARE correct but concentrated in the 1-4kHz
band where the HRTF's head-shadow ILD is strongest. Piano's LF-heavy
spectrum dilutes the broadband L-R skew.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))
from scene_two_dogs_v2 import compose_two_dog_scene_v2

SPEC = REPO_ROOT / "data" / "shoebox_v2_spec.json"
OUT = REPO_ROOT / "tmp" / "spike_output" / "analysis" / "bandwise_swing.png"
BN = REPO_ROOT / "tmp" / "spike_output" / "binaural_native"


def lr_db_per_win(L, R, sr, win_ms=100):
    w = int(win_ms * sr / 1000)
    n_win = len(L) // w
    out = np.zeros(n_win)
    t = np.zeros(n_win)
    for i in range(n_win):
        s = i*w; e = s+w
        eL = np.sqrt(np.mean(L[s:e]**2))+1e-9
        eR = np.sqrt(np.mean(R[s:e]**2))+1e-9
        out[i] = 20*np.log10(eL/eR)
        t[i] = (s+e)/(2*sr)
    return t, out


def main():
    with open(SPEC) as f:
        spec = json.load(f)
    mic = np.asarray(spec["mic"]["pos_m"])
    sc = compose_two_dog_scene_v2(SPEC)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.55, wspace=0.15)

    tags = ["dog_golden", "dog_husky"]
    tag_descr = {
        "dog_golden": "Golden trajectory: azi −111°→+180°→+111° (L-back → back → R-back)",
        "dog_husky":  "Husky trajectory:  azi 0°→+76°→+37°→+7°→0° (front → R → FR → front)",
    }

    # Row 1: azimuth timeline
    for col, tag in enumerate(tags):
        ax = fig.add_subplot(gs[0, col])
        traj = None
        for pl in sc.animals:
            if pl.tag == tag:
                traj = np.asarray(pl.trajectory_m); break
        n_fr = len(traj)
        dur = 5.0
        t = np.arange(n_fr) / (n_fr / dur)
        rel = traj - mic
        azi_deg = np.degrees(np.arctan2(rel[:, 0], rel[:, 1]))
        ax.plot(t, azi_deg, 'k-', lw=1.8)
        ax.axhline(0, color='gray', lw=0.5)
        ax.axhline(+90, color='r', lw=0.5, ls='--', label='+90° (right)')
        ax.axhline(-90, color='b', lw=0.5, ls='--', label='−90° (left)')
        ax.set_title(f"[Row 1] {tag_descr[tag]}", fontsize=9, fontweight='bold')
        ax.set_ylabel("azimuth (°)")
        ax.set_xlim(0, dur); ax.set_ylim(-190, 190)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='lower right')

    # Row 2: Broadband L-R over time (native binaural + naive downmix)
    for col, tag in enumerate(tags):
        ax = fig.add_subplot(gs[1, col])
        bn_path = BN / f"audio_B_rlr_HIGH_binaural_native_{tag}_binaural.wav"
        b, sr = sf.read(str(bn_path), always_2d=True)
        t, lr_db = lr_db_per_win(b[:,0], b[:,1], sr)
        # colored regions: red where R louder, blue where L louder
        ax.axhline(0, color='gray', lw=0.6)
        ax.fill_between(t, 0, lr_db, where=(lr_db>0), color='b', alpha=0.35, label='L louder')
        ax.fill_between(t, 0, lr_db, where=(lr_db<0), color='r', alpha=0.35, label='R louder')
        ax.plot(t, lr_db, 'k-', lw=1)
        ax.set_title(f"[Row 2] {tag}: broadband L−R (native binaural, 100ms windows)",
                     fontsize=9, fontweight='bold')
        ax.set_ylabel("L−R (dB)")
        ax.set_xlim(0, 5); ax.set_ylim(-8, 8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

    # Row 3: BAND-WISE L-R vs time (spectrogram-style, positive=blue negative=red)
    for col, tag in enumerate(tags):
        ax = fig.add_subplot(gs[2, col])
        bn_path = BN / f"audio_B_rlr_HIGH_binaural_native_{tag}_binaural.wav"
        b, sr = sf.read(str(bn_path), always_2d=True)
        L, R = b[:,0], b[:,1]
        bands = [(125,250),(250,500),(500,1000),(1000,2000),(2000,4000),(4000,7000)]
        band_lr = np.zeros((len(bands), 50))  # 50 windows over 5s = 100ms
        for bi, (lo, hi) in enumerate(bands):
            hi = min(hi, sr//2 - 100)
            sos = signal.butter(4, [lo, hi], btype='bandpass', fs=sr, output='sos')
            Lb = signal.sosfilt(sos, L)
            Rb = signal.sosfilt(sos, R)
            t_, lr = lr_db_per_win(Lb, Rb, sr, win_ms=100)
            band_lr[bi, :len(lr)] = lr
        im = ax.imshow(band_lr, aspect='auto', origin='lower', cmap='RdBu_r',
                       vmin=-10, vmax=10, extent=[0, 5, 0, len(bands)])
        ax.set_yticks(np.arange(len(bands)) + 0.5)
        ax.set_yticklabels([f"{lo}-{hi}" for lo, hi in bands])
        ax.set_title(f"[Row 3] {tag}: BAND-WISE L−R (dB)  blue=L louder, red=R louder",
                     fontsize=9, fontweight='bold')
        ax.set_ylabel("band (Hz)")
        ax.set_xlim(0, 5)
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='L−R (dB)')

    # Row 4: HIGH-PASS FILTERED demo — same audio, HP>1kHz to remove LF piano
    for col, tag in enumerate(tags):
        ax = fig.add_subplot(gs[3, col])
        bn_path = BN / f"audio_B_rlr_HIGH_binaural_native_{tag}_binaural.wav"
        b, sr = sf.read(str(bn_path), always_2d=True)
        sos = signal.butter(4, 1000, btype='highpass', fs=sr, output='sos')
        Lh = signal.sosfilt(sos, b[:,0])
        Rh = signal.sosfilt(sos, b[:,1])
        t, lr_db_hp = lr_db_per_win(Lh, Rh, sr)
        ax.axhline(0, color='gray', lw=0.6)
        ax.fill_between(t, 0, lr_db_hp, where=(lr_db_hp>0), color='b', alpha=0.35, label='L louder')
        ax.fill_between(t, 0, lr_db_hp, where=(lr_db_hp<0), color='r', alpha=0.35, label='R louder')
        ax.plot(t, lr_db_hp, 'k-', lw=1.5)
        ax.set_title(f"[Row 4] {tag}: L−R after >1kHz high-pass filter (removes LF piano energy)",
                     fontsize=9, fontweight='bold')
        ax.set_ylabel("L−R (dB)")
        ax.set_xlabel("time (s)")
        ax.set_xlim(0, 5); ax.set_ylim(-12, 12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

    fig.suptitle(
        "L−R skew analysis: broadband looks weak because piano is LF-dominant.\n"
        "HRTF head-shadow ILD peaks at 1-4kHz — that band shows the real spatial cue.\n"
        "Row 3 shows band-wise skew; Row 4 shows HP-filtered broadband matches expected direction.",
        fontsize=11, fontweight='bold'
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches='tight')
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
