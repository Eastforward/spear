"""Render top-down 2D videos where only ONE dog is highlighted.

Produces three videos:
  - golden_only.mp4  (video: golden colored + husky grayed; audio: golden solo)
  - husky_only.mp4   (video: husky colored + golden grayed; audio: husky solo)
  - mixed.mp4        (video: both colored; audio: both mixed)

This lets the user listen to each dog in isolation for A/B/C spike audio
verification.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_scene():
    sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    from scene_two_dogs_v2 import compose_two_dog_scene_v2
    return compose_two_dog_scene_v2(REPO_ROOT / "data" / "shoebox_v2_spec.json")


def _is_occluded(pos, mic_pos, sofa_center, sofa_size):
    mic = np.array(mic_pos)
    tgt = np.array(pos)
    d = tgt - mic
    aabb_min = np.array(sofa_center) - np.array(sofa_size) / 2
    aabb_max = np.array(sofa_center) + np.array(sofa_size) / 2
    tmin, tmax = -np.inf, np.inf
    for i in range(3):
        if abs(d[i]) < 1e-9:
            if mic[i] < aabb_min[i] or mic[i] > aabb_max[i]:
                return False
            continue
        t1 = (aabb_min[i] - mic[i]) / d[i]
        t2 = (aabb_max[i] - mic[i]) / d[i]
        tmin = max(tmin, min(t1, t2))
        tmax = min(tmax, max(t1, t2))
    return (tmax >= tmin) and (tmin <= 1.0) and (tmax >= 0.0)


def render_frame(frame_idx, scene, spec, tmp_dir, highlight_tag=None):
    """highlight_tag=None -> both colored. Otherwise the other is grayed."""
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=100)
    rs = spec["room_size_m"]

    room_rect = mpatches.Rectangle((0, 0), rs[0], rs[1], linewidth=2,
                                    edgecolor='black', facecolor='#f8f8f8')
    ax.add_patch(room_rect)

    sofa = spec["furniture"][0]
    sc = sofa["center_m"]
    ss = sofa["size_m"]
    sofa_rect = mpatches.Rectangle(
        (sc[0] - ss[0]/2, sc[1] - ss[1]/2), ss[0], ss[1],
        linewidth=2, edgecolor='#a05a1a', facecolor='#f4c880',
    )
    ax.add_patch(sofa_rect)
    ax.text(sc[0], sc[1], 'SOFA', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#5a3610')

    mic_pos = spec["mic"]["pos_m"]
    ax.plot(mic_pos[0], mic_pos[1], 'v', markersize=14, color='#1e5ea8',
            markeredgecolor='black', markeredgewidth=1)
    ax.annotate('', xy=(mic_pos[0], mic_pos[1] + 0.4),
                xytext=(mic_pos[0], mic_pos[1]),
                arrowprops=dict(arrowstyle='->', color='#1e5ea8', lw=2))
    ax.text(mic_pos[0] + 0.15, mic_pos[1] - 0.1, 'MIC / view0', fontsize=8,
            color='#1e5ea8')

    fov_half_rad = np.radians(60)
    fov_reach = 3.5
    fov_verts = [
        mic_pos[:2],
        [mic_pos[0] - fov_reach * np.sin(fov_half_rad),
         mic_pos[1] + fov_reach * np.cos(fov_half_rad)],
        [mic_pos[0] + fov_reach * np.sin(fov_half_rad),
         mic_pos[1] + fov_reach * np.cos(fov_half_rad)],
    ]
    fov_patch = mpatches.Polygon(fov_verts, facecolor='#1e5ea8', alpha=0.06,
                                  edgecolor='#1e5ea8', linewidth=0.5, linestyle='--')
    ax.add_patch(fov_patch)

    active_occluded = False
    for a in scene.animals:
        pos = a.trajectory_m[frame_idx]
        is_highlighted = (highlight_tag is None) or (a.tag == highlight_tag)
        occluded = False
        if a.tag == "dog_husky":
            occluded = _is_occluded(pos, mic_pos, sc, ss)

        if a.tag == "dog_golden":
            base_color = '#e08820'
            label = 'GOLDEN'
        else:
            base_color = '#1a76c8'
            label = 'HUSKY' + (' (OCCLUDED)' if occluded else '')

        if not is_highlighted:
            color = '#c0c0c0'
            label += ' (muted)'
        else:
            color = '#909090' if occluded else base_color
            if a.tag == highlight_tag:
                label = '► ' + label + ' ◄'

        # Track whether the highlighted (active audio) source is currently occluded
        if is_highlighted and occluded:
            active_occluded = True

        ax.plot(pos[0], pos[1], 'o', markersize=14, color=color,
                markeredgecolor='black', markeredgewidth=1)
        ax.text(pos[0] + 0.2, pos[1] + 0.15, label, fontsize=8,
                fontweight='bold', color=color)

        trail_start = max(0, frame_idx - 20)
        trail = a.trajectory_m[trail_start:frame_idx + 1]
        if len(trail) > 1:
            ax.plot(trail[:, 0], trail[:, 1], '-', color=color, alpha=0.4, lw=1)

    # LOS lines for both, but color only the highlighted one strongly
    for a in scene.animals:
        pos = a.trajectory_m[frame_idx]
        if a.tag == "dog_husky":
            occluded = _is_occluded(pos, mic_pos, sc, ss)
        else:
            occluded = False
        is_highlighted = (highlight_tag is None) or (a.tag == highlight_tag)
        alpha = 0.6 if is_highlighted else 0.15
        los_color = '#d04040' if occluded else '#40b040'
        ax.plot([mic_pos[0], pos[0]], [mic_pos[1], pos[1]],
                color=los_color, alpha=alpha, lw=1.2, linestyle=':')

    t_s = frame_idx / spec["render_config"]["fps"]
    if highlight_tag is None:
        title_prefix = 'MIXED'
        title_color = 'black'
    else:
        title_prefix = f'ACTIVE: {highlight_tag.upper()}'
        title_color = '#d04040' if active_occluded else '#40b040'
    title = f'{title_prefix}  |  frame {frame_idx:02d} / 74  |  t = {t_s:.2f}s'
    if highlight_tag == "dog_husky":
        title += f'  |  occluded: {"YES" if active_occluded else "no"}'
    ax.set_title(title, fontsize=11, fontweight='bold', color=title_color)

    ax.set_xlim(-0.5, rs[0] + 0.5)
    ax.set_ylim(-0.5, rs[1] + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (world, meters) --- audio: L to R')
    ax.set_ylabel('Y (world, meters) --- audio: back to front')
    ax.grid(True, alpha=0.3)

    fpath = tmp_dir / f"frame_{frame_idx:03d}.png"
    fig.savefig(fpath, dpi=100, bbox_inches='tight')
    plt.close(fig)
    return fpath


def render_video(spec_path, audio_wav_path, out_mp4_path, highlight_tag=None,
                  tmp_frames_dir=None):
    with open(spec_path) as f:
        spec = json.load(f)
    scene = _load_scene()

    label = highlight_tag or "mixed"
    if tmp_frames_dir is None:
        tmp_frames_dir = REPO_ROOT / "tmp" / "spike_rlr" / f"topdown_{label}_frames"
    tmp_frames_dir = Path(tmp_frames_dir)
    tmp_frames_dir.mkdir(parents=True, exist_ok=True)

    n_frames = spec["render_config"]["n_frames"]
    fps = spec["render_config"]["fps"]

    print(f"[topdown-{label}] rendering {n_frames} frames -> {tmp_frames_dir}")
    for f in range(n_frames):
        render_frame(f, scene, spec, tmp_frames_dir, highlight_tag=highlight_tag)
        if f % 20 == 0:
            print(f"  frame {f}/{n_frames}")

    silent_mp4 = out_mp4_path.parent / f"{out_mp4_path.stem}_silent.mp4"
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(tmp_frames_dir / "frame_%03d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        str(silent_mp4),
    ], check=True)

    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(silent_mp4),
        "-i", str(audio_wav_path),
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v", "-map", "1:a",
        "-shortest",
        str(out_mp4_path),
    ], check=True)
    print(f"[topdown-{label}] muxed video -> {out_mp4_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(REPO_ROOT / "data" / "shoebox_v2_spec.json"))
    ap.add_argument("--audio-dir", default=str(REPO_ROOT / "tmp" / "spike_output" / "raw_audio"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "tmp" / "spike_output" / "videos"))
    args = ap.parse_args()

    audio_dir = Path(args.audio_dir)
    out_dir = Path(args.out_dir)

    # 3 videos: golden only, husky only, mixed
    render_video(args.spec,
                 audio_dir / "audio_B_rlr_FOA_dog_golden_stereo.wav",
                 out_dir / "B_rlr_topdown_golden_only.mp4",
                 highlight_tag="dog_golden")

    render_video(args.spec,
                 audio_dir / "audio_B_rlr_FOA_dog_husky_stereo.wav",
                 out_dir / "B_rlr_topdown_husky_only.mp4",
                 highlight_tag="dog_husky")

    render_video(args.spec,
                 audio_dir / "audio_B_rlr_stereo.wav",
                 out_dir / "B_rlr_topdown_mixed.mp4",
                 highlight_tag=None)


if __name__ == "__main__":
    main()
