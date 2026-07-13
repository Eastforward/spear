"""Render a top-down 2D bird's-eye visualization of the shoebox scene.

Produces a 75-frame video (matches audio duration) with:
  - Room outline (5.2 x 6.0 rectangle)
  - Sofa (yellow box at 2.6, 3.45)
  - Mic (blue triangle at 2.6, 2.2 pointing +Y)
  - Golden retriever (orange dot moving L->R behind camera)
  - Husky (blue dot moving in 4-segment detour, GREYED OUT when behind sofa)
  - Occlusion status ribbon (green/red)
  - Frame counter + time

Purpose: user can visually check dog positions each frame against the audio
they hear, verifying spatial audio direction.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))
from apartment_builtin_obstacles import apartment_builtin_visual_obstacles  # noqa: E402
from review_video_encode import encode_rgb_frames  # noqa: E402


def _style_for_tag(tag):
    styles = {
        "dog_golden": {"color": "#e08820", "label": "GOLDEN"},
        "dog_beagle_v2": {"color": "#c05a2b", "label": "BEAGLE"},
        "cat_british_shorthair_v2": {
            "color": "#6b5fb5",
            "label": "BRITISH SHORTHAIR",
        },
        "hy3d_rocketbox_male_adult_01_spike": {
            "color": "#1b7489",
            "label": "MALE",
        },
        "hy3d_rocketbox_female_adult_01_spike": {
            "color": "#ba484e",
            "label": "FEMALE",
        },
    }
    if tag in styles:
        return styles[tag]
    return {"color": "#2a9d8f", "label": tag.upper()}


def _load_scene(spec_path=None):
    """Load scene by dispatching on spec_version.

    - spec_version == "v2" (shoebox_v2): compose_two_dog_scene_v2
    - spec_version == "apartment_v1":    compose_two_dog_scene_apartment
    Defaults to shoebox_v2 when spec_path is None (legacy behavior).
    """
    sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    if spec_path is None:
        spec_path = REPO_ROOT / "data" / "shoebox_v2_spec.json"
    with open(spec_path) as f:
        version = json.load(f).get("spec_version", "v2")
    if version == "apartment_v1":
        from scene_two_dogs_apartment import compose_two_dog_scene_apartment
        return compose_two_dog_scene_apartment(spec_path)
    from scene_two_dogs_v2 import compose_two_dog_scene_v2
    return compose_two_dog_scene_v2(spec_path)


def _is_sofa_occluded(pos, mic_pos, sofa_center, sofa_size):
    """Check if line-of-sight from mic to pos is blocked by sofa AABB."""
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


def render_frame(frame_idx, scene, spec, tmp_dir):
    """Render a single frame to a PNG file. Dispatches by spec_version."""
    if spec.get("spec_version") == "apartment_v1":
        return _render_frame_apartment(frame_idx, scene, spec, tmp_dir)
    return _render_frame_shoebox(frame_idx, scene, spec, tmp_dir)


def render_frame_rgb(frame_idx, scene, spec):
    """Render directly to an even-sized RGB array for streamed video encoding."""
    if spec.get("spec_version") == "apartment_v1":
        return _render_frame_apartment(
            frame_idx, scene, spec, tmp_dir=None, return_rgb=True
        )
    return _render_frame_shoebox(
        frame_idx, scene, spec, tmp_dir=None, return_rgb=True
    )


def _finish_figure(fig, *, frame_idx, tmp_dir, return_rgb):
    if return_rgb:
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        height = rgba.shape[0] - (rgba.shape[0] % 2)
        width = rgba.shape[1] - (rgba.shape[1] % 2)
        rgb = np.ascontiguousarray(rgba[:height, :width, :3])
        plt.close(fig)
        return rgb
    fpath = Path(tmp_dir) / f"frame_{frame_idx:03d}.png"
    fig.savefig(fpath, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return fpath


def _render_frame_shoebox(
    frame_idx, scene, spec, tmp_dir, *, return_rgb=False
):
    """Shoebox v2 top-down: uses room_size_m + furniture[0]==sofa."""
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=100)
    rs = spec["room_size_m"]

    # Room outline
    room_rect = mpatches.Rectangle((0, 0), rs[0], rs[1], linewidth=2,
                                    edgecolor='black', facecolor='#f8f8f8')
    ax.add_patch(room_rect)

    # Sofa
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

    # Mic + camera at (2.6, 2.2), pointing +Y
    mic_pos = spec["mic"]["pos_m"]
    ax.plot(mic_pos[0], mic_pos[1], 'v', markersize=14, color='#1e5ea8',
            markeredgecolor='black', markeredgewidth=1)
    # forward arrow (+Y)
    ax.annotate('', xy=(mic_pos[0], mic_pos[1] + 0.4),
                xytext=(mic_pos[0], mic_pos[1]),
                arrowprops=dict(arrowstyle='->', color='#1e5ea8', lw=2))
    ax.text(mic_pos[0] + 0.15, mic_pos[1] - 0.1, 'MIC / view0', fontsize=8,
            color='#1e5ea8')

    # Camera field-of-view cone (view0 fov=120)
    fov_half_rad = np.radians(60)  # 120 total
    fov_reach = 3.0
    fov_verts = [
        mic_pos[:2],
        [mic_pos[0] - fov_reach * np.sin(fov_half_rad),
         mic_pos[1] + fov_reach * np.cos(fov_half_rad)],
        [mic_pos[0] + fov_reach * np.sin(fov_half_rad),
         mic_pos[1] + fov_reach * np.cos(fov_half_rad)],
    ]
    fov_patch = mpatches.Polygon(fov_verts, facecolor='#1e5ea8', alpha=0.08,
                                  edgecolor='#1e5ea8', linewidth=0.5, linestyle='--')
    ax.add_patch(fov_patch)

    # Animals
    for a in scene.animals:
        pos = a.trajectory_m[frame_idx]
        if a.tag == "dog_golden":
            color, label = '#e08820', 'GOLDEN'
        else:
            style = _style_for_tag(a.tag)
            occluded = _is_sofa_occluded(pos, mic_pos, sc, ss)
            color = '#909090' if occluded else style["color"]
            label = style["label"] + (' (OCCLUDED)' if occluded else '')
        ax.plot(pos[0], pos[1], 'o', markersize=14, color=color,
                markeredgecolor='black', markeredgewidth=1)
        ax.text(pos[0] + 0.2, pos[1] + 0.15, label, fontsize=8,
                fontweight='bold', color=color)

        # Draw trail (last 20 frames)
        trail_start = max(0, frame_idx - 20)
        trail = a.trajectory_m[trail_start:frame_idx + 1]
        if len(trail) > 1:
            ax.plot(trail[:, 0], trail[:, 1], '-', color=color, alpha=0.4, lw=1)

    # Line-of-sight from mic to the first non-golden source.
    highlighted = next((a for a in scene.animals if a.tag != "dog_golden"), None)
    if highlighted is None:
        highlighted = scene.animals[0]
    highlighted_pos = highlighted.trajectory_m[frame_idx]
    occluded = _is_sofa_occluded(highlighted_pos, mic_pos, sc, ss)
    los_color = '#d04040' if occluded else '#40b040'
    ax.plot([mic_pos[0], highlighted_pos[0]], [mic_pos[1], highlighted_pos[1]],
            '-', color=los_color, alpha=0.6, lw=1.5, linestyle=':')

    # Title with frame + time
    t_s = frame_idx / spec["render_config"]["fps"]
    ax.set_title(f'Frame {frame_idx:02d} / 74  |  t = {t_s:.2f}s  |  '
                 f'occlusion: {"YES" if occluded else "no"}',
                 fontsize=11, fontweight='bold',
                 color=los_color if occluded else 'black')

    # Axes: SPEAR world coords
    ax.set_xlim(-0.5, rs[0] + 0.5)
    ax.set_ylim(-0.5, rs[1] + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (world, meters) --- audio: L to R')
    ax.set_ylabel('Y (world, meters) --- audio: back to front')
    ax.grid(True, alpha=0.3)

    return _finish_figure(
        fig, frame_idx=frame_idx, tmp_dir=tmp_dir, return_rgb=return_rgb
    )


def _render_frame_apartment(
    frame_idx, scene, spec, tmp_dir, *, return_rgb=False
):
    """Apartment v1 top-down: uses apartment_shell_map.json for room outline
    and apartment_furniture_map.json for furniture rectangles."""
    fig, ax = plt.subplots(figsize=(7.5, 7.0), dpi=100)

    # Load shell + furniture from linked JSONs (both in UE cm; convert to SSOT)
    shell_path = REPO_ROOT / spec.get("apartment_shell_map",
                                       "data/apartment_shell_map.json")
    furn_path = REPO_ROOT / spec.get("apartment_furniture_map",
                                      "data/apartment_furniture_map.json")
    shell = json.loads(shell_path.read_text())
    furn = json.loads(furn_path.read_text())

    # UE-to-SSOT conversion (matches gen_mesh_apartment.py::ue_to_ssot)
    APARTMENT_MIC_ORIGIN_UE_CM = (-120.0, 80.0, 120.0)
    APARTMENT_FLOOR_Z_UE_CM = 27.1

    def _ue_to_ssot_xy(bmin_ue, bmax_ue):
        # returns (x0, y0, x1, y1) in SSOT meters; caller may sort lo<=hi
        x0 = (bmin_ue[0] - APARTMENT_MIC_ORIGIN_UE_CM[0]) / 100.0
        x1 = (bmax_ue[0] - APARTMENT_MIC_ORIGIN_UE_CM[0]) / 100.0
        # Y flip: SSOT y = -(UE y - origin_y)/100. Swap bmin/bmax as needed.
        y0 = -(bmin_ue[1] - APARTMENT_MIC_ORIGIN_UE_CM[1]) / 100.0
        y1 = -(bmax_ue[1] - APARTMENT_MIC_ORIGIN_UE_CM[1]) / 100.0
        lo_x, hi_x = min(x0, x1), max(x0, x1)
        lo_y, hi_y = min(y0, y1), max(y0, y1)
        return lo_x, lo_y, hi_x, hi_y

    # Compute overall XY bbox for axes
    xs, ys = [], []
    for a in shell["shell_actors"]:
        x0, y0, x1, y1 = _ue_to_ssot_xy(a["bbox_min_ue_cm"], a["bbox_max_ue_cm"])
        xs += [x0, x1]; ys += [y0, y1]
    room_x_min, room_x_max = min(xs), max(xs)
    room_y_min, room_y_max = min(ys), max(ys)

    # Draw shell (walls/floor/ceiling projected onto XY): walls as thick outlines,
    # doors/windows as colored rects, floor/ceiling skipped (they cover everything).
    color_by_label = {
        "shell_wall": ("#333333", "#dddddd"),
        "shell_door": ("#8b4513", "#d2b48c"),
        "shell_window": ("#1e5ea8", "#a8d0f0"),
        "shell_curtain": ("#800080", "#e0b0e0"),
        "shell_picture": ("#606060", "#d0d0d0"),
        "shell_mirror": ("#404080", "#c0c0f0"),
        "structural": ("#555555", "#e0e0e0"),
    }
    for a in shell["shell_actors"]:
        if a["shell_label"] in ("shell_floor", "shell_ceiling"):
            continue
        x0, y0, x1, y1 = _ue_to_ssot_xy(a["bbox_min_ue_cm"], a["bbox_max_ue_cm"])
        edge, face = color_by_label.get(a["shell_label"], ("black", "none"))
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=1.2, edgecolor=edge, facecolor=face, alpha=0.6,
        ))

    # Draw furniture (only those actually kept in this clip's furniture_mode)
    include_categories = set(spec.get("furniture_include_categories", []))
    cats_path = REPO_ROOT / "tools" / "spike_rlr" / "apartment_furniture_categories.json"
    if include_categories and cats_path.exists():
        cats = json.loads(cats_path.read_text())
        kept_actors = set()
        for cat in include_categories:
            kept_actors.update(cats.get(cat, []))
        kept_actors.update(spec.get("furniture_include_actors_extra", []))
        kept_actors.difference_update(spec.get("furniture_exclude_actors", []))
    else:
        kept_actors = None  # None means draw all furniture
    for f in furn["furniture"]:
        if kept_actors is not None and f["actor_name"] not in kept_actors:
            continue
        x0, y0, x1, y1 = _ue_to_ssot_xy(f["bbox_min_ue_cm"], f["bbox_max_ue_cm"])
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=0.8, edgecolor='#a05a1a', facecolor='#f4c880', alpha=0.5,
        ))

    for obs in apartment_builtin_visual_obstacles(spec):
        x0, y0, x1, y1 = obs.bbox_xy
        ax.add_patch(mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=1.0, edgecolor='#6f2c2c', facecolor='#d89090',
            alpha=0.35, hatch='//',
        ))

    # Mic + camera glued
    mic_pos = spec["mic"]["pos_m"]
    mic_yaw = spec["mic"]["yaw_deg"]
    ax.plot(mic_pos[0], mic_pos[1], 'v', markersize=14, color='#1e5ea8',
            markeredgecolor='black', markeredgewidth=1)
    # Forward arrow along mic_yaw.
    # apartment_v1 convention: at yaw=0, mic-forward = +X world (per
    # spec["mic"]["forward"] = [1, 0, 0]). yaw rotates CCW in XY plane.
    # Therefore: fwd = (cos(yaw), sin(yaw)).
    yaw_rad = np.deg2rad(mic_yaw)
    fwd_dx = 0.6 * np.cos(yaw_rad)
    fwd_dy = 0.6 * np.sin(yaw_rad)
    ax.annotate('', xy=(mic_pos[0] + fwd_dx, mic_pos[1] + fwd_dy),
                xytext=(mic_pos[0], mic_pos[1]),
                arrowprops=dict(arrowstyle='->', color='#1e5ea8', lw=2))
    ax.text(mic_pos[0] + 0.2, mic_pos[1] - 0.2, 'MIC / cam', fontsize=8,
            color='#1e5ea8')

    # FOV cone (from spec's camera_configs[0])
    fov_deg = float(spec["camera_configs"][0]["fov_deg"])
    fov_half = np.deg2rad(fov_deg / 2)
    fov_reach = 4.0
    # Left/right edges of the cone rotated by mic_yaw (same +X-forward convention)
    left_ang = yaw_rad - fov_half
    right_ang = yaw_rad + fov_half
    fov_verts = [
        mic_pos[:2],
        [mic_pos[0] + fov_reach * np.cos(left_ang),
         mic_pos[1] + fov_reach * np.sin(left_ang)],
        [mic_pos[0] + fov_reach * np.cos(right_ang),
         mic_pos[1] + fov_reach * np.sin(right_ang)],
    ]
    ax.add_patch(mpatches.Polygon(fov_verts, facecolor='#1e5ea8', alpha=0.08,
                                    edgecolor='#1e5ea8', linewidth=0.5, linestyle='--'))

    # Animals
    for a in scene.animals:
        pos = a.trajectory_m[frame_idx]
        style = _style_for_tag(a.tag)
        color = style["color"]
        label = style["label"]
        ax.plot(pos[0], pos[1], 'o', markersize=14, color=color,
                markeredgecolor='black', markeredgewidth=1)
        ax.text(pos[0] + 0.2, pos[1] + 0.15, label, fontsize=8,
                fontweight='bold', color=color)
        # Trail (last 20 frames)
        trail_start = max(0, frame_idx - 20)
        trail = a.trajectory_m[trail_start:frame_idx + 1]
        if len(trail) > 1:
            ax.plot(trail[:, 0], trail[:, 1], '-', color=color, alpha=0.4, lw=1)

    # Title
    t_s = frame_idx / spec["render_config"]["fps"]
    total_frames = spec["render_config"]["n_frames"]
    ax.set_title(f'Apartment_v1 top-down  |  Frame {frame_idx:02d}/{total_frames-1}  |  '
                 f't = {t_s:.2f}s', fontsize=11, fontweight='bold')
    # Axes
    pad = 1.0
    ax.set_xlim(room_x_min - pad, room_x_max + pad)
    ax.set_ylim(room_y_min - pad, room_y_max + pad)
    ax.set_aspect('equal')
    ax.set_xlabel('X (world, meters) --- audio: L to R')
    ax.set_ylabel('Y (world, meters) --- audio: back to front')
    ax.grid(True, alpha=0.3)

    return _finish_figure(
        fig, frame_idx=frame_idx, tmp_dir=tmp_dir, return_rgb=return_rgb
    )


def build_video(spec_path, audio_wav_path, out_mp4_path, tmp_frames_dir=None):
    with open(spec_path) as f:
        spec = json.load(f)
    scene = _load_scene(spec_path)

    if tmp_frames_dir is None:
        tmp_frames_dir = REPO_ROOT / "tmp" / "spike_rlr" / "topdown_frames"
    tmp_frames_dir = Path(tmp_frames_dir)
    if tmp_frames_dir.exists():
        shutil.rmtree(tmp_frames_dir)

    n_frames = spec["render_config"]["n_frames"]
    fps = spec["render_config"]["fps"]

    # Silent video first
    silent_mp4 = out_mp4_path.parent / f"{out_mp4_path.stem}_silent.mp4"
    out_mp4_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[topdown] streaming {n_frames} RGB frames -> {silent_mp4}")

    def iter_topdown_frames():
        for frame_index in range(n_frames):
            yield render_frame_rgb(frame_index, scene, spec)
            if frame_index % 15 == 0:
                print(f"  frame {frame_index}/{n_frames}")

    encoded_count = encode_rgb_frames(
        iter_topdown_frames(), silent_mp4, fps=int(fps)
    )
    if encoded_count != n_frames:
        raise RuntimeError(
            f"top-down frame count changed: expected {n_frames}, got {encoded_count}"
        )
    print(f"[topdown] silent video -> {silent_mp4}")

    # Mux audio
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(silent_mp4),
        "-i", str(audio_wav_path),
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v", "-map", "1:a",
        "-shortest",
        str(out_mp4_path),
    ], check=True)
    print(f"[topdown] muxed video -> {out_mp4_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(REPO_ROOT / "data" / "shoebox_v2_spec.json"))
    ap.add_argument("--audio", default=str(REPO_ROOT / "tmp" / "spike_output" / "raw_audio" / "audio_B_rlr_stereo.wav"))
    ap.add_argument("--out", default=str(REPO_ROOT / "tmp" / "spike_output" / "videos" / "B_rlr_topdown.mp4"))
    ap.add_argument("--tmp-frames-dir", type=Path)
    args = ap.parse_args()
    build_video(
        args.spec,
        args.audio,
        Path(args.out),
        tmp_frames_dir=args.tmp_frames_dir,
    )


if __name__ == "__main__":
    main()
