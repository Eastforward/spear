"""Waveform + energy comparison across representations.

Rows:
  1. Trajectory azimuth-vs-time (ground truth spatial timeline)
  2. FOA W channel (omni energy) per source
  3. FOA X, Y, Z overlays (RLR native channel semantics)
  4. FOA→stereo naive downmix (L/R RMS envelopes)
  5. Native binaural (L/R RMS envelopes) — the corrected one
  6. Native binaural L-R dB (per-100ms) — clearest side-swing indicator
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))
from scene_two_dogs_v2 import compose_two_dog_scene_v2

SPEC = REPO_ROOT / "data" / "shoebox_v2_spec.json"
OUT = REPO_ROOT / "tmp" / "spike_output" / "analysis" / "wave_compare.png"


def envelope(x, sr, win_ms=50):
    n = int(win_ms * sr / 1000)
    if x.ndim == 1:
        return np.sqrt(np.convolve(x**2, np.ones(n)/n, mode='same'))
    return np.stack([envelope(x[c], sr, win_ms) for c in range(x.shape[0])])


def load_solo_traj(sc, tag):
    for pl in sc.animals:
        if pl.tag == tag:
            return np.asarray(pl.trajectory_m)
    raise KeyError(tag)


def main():
    with open(SPEC) as f:
        spec = json.load(f)
    mic = np.asarray(spec["mic"]["pos_m"])
    sc = compose_two_dog_scene_v2(SPEC)

    # -- Load audio (HIGH quality, per-source solo) --
    hq = REPO_ROOT / "tmp" / "spike_output" / "raw_audio_hq"
    bn = REPO_ROOT / "tmp" / "spike_output" / "binaural_native"

    tags = ["dog_golden", "dog_husky"]
    audio = {}
    for tag in tags:
        foa, sr_a = sf.read(str(hq / f"audio_B_rlr_HIGH_FOA_{tag}_FOA.wav"), always_2d=True)
        st_dm, _ = sf.read(str(hq / f"audio_B_rlr_HIGH_FOA_{tag}_stereo.wav"), always_2d=True)
        bin_native, sr_b = sf.read(str(bn / f"audio_B_rlr_HIGH_binaural_native_{tag}_binaural.wav"), always_2d=True)
        audio[tag] = {
            "foa": foa.T.astype(np.float32),         # (4, N)
            "stereo_downmix": st_dm.T.astype(np.float32),  # (2, N)
            "bin_native": bin_native.T.astype(np.float32), # (2, N)
            "sr": sr_a,
        }
        assert sr_a == sr_b, (sr_a, sr_b)

    dur_s = audio[tags[0]]["foa"].shape[1] / audio[tags[0]]["sr"]

    # ---------- FIGURE ----------
    fig = plt.figure(figsize=(18, 14))
    n_rows = 6
    gs = fig.add_gridspec(n_rows, 2, hspace=0.65, wspace=0.15)

    # Row 1: azimuth-vs-time from trajectory (per source, listener frame)
    for col, tag in enumerate(tags):
        ax = fig.add_subplot(gs[0, col])
        traj = load_solo_traj(sc, tag)  # (n_frames, 3), 15 fps → n=75
        n_fr = len(traj)
        t = np.arange(n_fr) / (n_fr / dur_s)  # dur_s seconds
        rel = traj - mic
        # listener-frame: listener faces +Y_scene; right = +X_scene (SSOT)
        azi_deg = np.degrees(np.arctan2(rel[:, 0], rel[:, 1]))
        ax.plot(t, azi_deg, 'k-', lw=1.8)
        ax.axhline(0, color='gray', lw=0.5)
        ax.axhline(90, color='r', lw=0.5, alpha=0.5, label='RIGHT (+90°)')
        ax.axhline(-90, color='b', lw=0.5, alpha=0.5, label='LEFT (-90°)')
        ax.axhline(180, color='gray', lw=0.5, alpha=0.3)
        ax.axhline(-180, color='gray', lw=0.5, alpha=0.3)
        ax.set_title(f"[ROW 1] {tag}: source azimuth vs time (0=front, ±180=behind, +90=right)",
                     fontsize=9, fontweight='bold')
        ax.set_ylabel("azi (deg)")
        ax.set_ylim(-190, 190)
        ax.set_xlim(0, dur_s)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

    # Rows 2-6: audio per source
    sr = audio[tags[0]]["sr"]
    t_a = np.arange(audio[tags[0]]["foa"].shape[1]) / sr

    def norm_plot(a, b, gap=0.02):
        """Normalize two arrays together for plotting on same axes."""
        peak = max(np.abs(a).max(), np.abs(b).max(), 1e-9)
        return a / peak, b / peak

    for col, tag in enumerate(tags):
        d = audio[tag]

        # Row 2: FOA W (omni energy envelope)
        ax = fig.add_subplot(gs[1, col])
        env_W = envelope(d["foa"][0], sr, win_ms=30)
        ax.plot(t_a, env_W, 'k-', lw=1)
        ax.fill_between(t_a, 0, env_W, alpha=0.2, color='k')
        ax.set_title(f"[ROW 2] {tag}: FOA channel 0 (W = omni pressure) — 30ms RMS envelope",
                     fontsize=9, fontweight='bold')
        ax.set_ylabel("|W|")
        ax.set_xlim(0, dur_s)
        ax.grid(True, alpha=0.3)

        # Row 3: FOA ch1,2,3 envelopes overlaid
        ax = fig.add_subplot(gs[2, col])
        colors = ['#e41a1c', '#377eb8', '#4daf4a']
        labels = ['ch1 (RLR ~ up/vert?)', 'ch2 (RLR fwd/-back)', 'ch3 (RLR right/-left)']
        for c, color, lab in zip([1, 2, 3], colors, labels):
            env_c = envelope(d["foa"][c], sr, win_ms=30)
            ax.plot(t_a, env_c, color=color, lw=1, label=lab)
        ax.set_title(f"[ROW 3] {tag}: FOA channels 1/2/3 — 30ms RMS envelopes",
                     fontsize=9, fontweight='bold')
        ax.set_ylabel("|ch_i|")
        ax.set_xlim(0, dur_s)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

        # Row 4: FOA→stereo naive downmix (L,R envelopes) — this is what the
        #        old files audio_B_rlr_*_stereo.wav used
        ax = fig.add_subplot(gs[3, col])
        env_L = envelope(d["stereo_downmix"][0], sr, win_ms=30)
        env_R = envelope(d["stereo_downmix"][1], sr, win_ms=30)
        ax.plot(t_a, env_L, 'b-', lw=1, label='L (W - 0.707·ch3)')
        ax.plot(t_a, env_R, 'r-', lw=1, label='R (W + 0.707·ch3)')
        ax.set_title(f"[ROW 4] {tag}: FOA→stereo NAIVE DOWNMIX (L,R envelopes) — used in old topdown mp4s",
                     fontsize=9, fontweight='bold')
        ax.set_ylabel("|amp|")
        ax.set_xlim(0, dur_s)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

        # Row 5: RLR native binaural (L,R envelopes)
        ax = fig.add_subplot(gs[4, col])
        env_L = envelope(d["bin_native"][0], sr, win_ms=30)
        env_R = envelope(d["bin_native"][1], sr, win_ms=30)
        ax.plot(t_a, env_L, 'b-', lw=1, label='L (native binaural)')
        ax.plot(t_a, env_R, 'r-', lw=1, label='R (native binaural)')
        ax.set_title(f"[ROW 5] {tag}: RLR NATIVE BINAURAL (L,R envelopes) — corrected via L↔R swap",
                     fontsize=9, fontweight='bold')
        ax.set_ylabel("|amp|")
        ax.set_xlim(0, dur_s)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

        # Row 6: L-R dB per 100ms — clearest side-swing story
        ax = fig.add_subplot(gs[5, col])
        win_ms = 100
        w = int(win_ms * sr / 1000)
        n_win = d["bin_native"].shape[1] // w
        t_win = np.arange(n_win) * w / sr + w/(2*sr)
        lr_db_native = np.zeros(n_win)
        lr_db_naive = np.zeros(n_win)
        for i in range(n_win):
            s = i*w; e = s+w
            eL_n = np.sqrt(np.mean(d["bin_native"][0, s:e]**2))+1e-9
            eR_n = np.sqrt(np.mean(d["bin_native"][1, s:e]**2))+1e-9
            lr_db_native[i] = 20*np.log10(eL_n/eR_n)
            eL_dm = np.sqrt(np.mean(d["stereo_downmix"][0, s:e]**2))+1e-9
            eR_dm = np.sqrt(np.mean(d["stereo_downmix"][1, s:e]**2))+1e-9
            lr_db_naive[i] = 20*np.log10(eL_dm/eR_dm)
        ax.axhline(0, color='gray', lw=0.6)
        ax.plot(t_win, lr_db_native, 'g-', lw=1.8, marker='.', ms=4, label='Native binaural L−R (dB)')
        ax.plot(t_win, lr_db_naive, color='#888', lw=1, ls='--', label='FOA naive stereo L−R (dB)')
        ax.set_title(f"[ROW 6] {tag}: L−R energy (dB per 100ms) — POSITIVE = LEFT louder",
                     fontsize=9, fontweight='bold')
        ax.set_ylabel("L−R (dB)")
        ax.set_xlabel("time (s)")
        ax.set_xlim(0, dur_s)
        ax.set_ylim(-8, 8)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='upper right')

    fig.suptitle(
        "Waveform / envelope comparison: FOA vs FOA-naive-stereo vs RLR-native-binaural\n"
        "Golden trajectory: L-back → back → R-back  (should give left-then-right L−R swing)\n"
        "Husky trajectory:  front → R-front → R → front  (should give strong RIGHT skew mid-clip)",
        fontsize=12, fontweight='bold'
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=110, bbox_inches='tight')
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
