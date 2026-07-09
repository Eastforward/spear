"""Build per-clip human-review videos for apartment_v1 dataset clips."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]


_MARKER_STYLES = {
    "dog_golden": {"label": "GOLDEN", "color": (255, 181, 43)},
    "dog_beagle_v2": {"label": "BEAGLE", "color": (222, 92, 48)},
    "cat_british_shorthair_v2": {
        "label": "BRITISH SHORTHAIR",
        "color": (124, 88, 211),
    },
}

_SOURCE_SHORT_LABELS = {
    "dog_golden": "GOLDEN",
    "dog_beagle_v2": "BEAGLE",
    "cat_british_shorthair_v2": "BRITISH",
}

_FLAG_SHORT_LABELS = {
    "occluded_by_furniture": "occFurn",
    "occluded_by_wall": "occWall",
    "never_occluded": "noOcc",
    "leaves_camera_fov": "leaveFOV",
    "stays_in_camera_fov": "stayFOV",
    "crosses_azimuth_zero": "crossAz",
    "passes_close_to_mic": "closeMic",
    "far_from_mic_whole_clip": "farMic",
    "stationary": "stat",
    "steady_walk": "walk",
    "stop_and_go": "stopGo",
    "sources_pass_each_other": "passSrc",
}


def _format_flag_names(flag_names: list[str]) -> str:
    return ",".join(_FLAG_SHORT_LABELS.get(name, name) for name in flag_names)


def _true_ranges(values):
    ranges = []
    start = None
    for i, v in enumerate(values):
        if v and start is None:
            start = i
        elif not v and start is not None:
            ranges.append((start, i - 1))
            start = None
    if start is not None:
        ranges.append((start, len(values) - 1))
    if not ranges:
        return "none"
    parts = []
    for a, b in ranges:
        parts.append(str(a) if a == b else f"{a}-{b}")
    return ",".join(parts)


def _count(values):
    return sum(1 for v in values if bool(v))


def _effective_audio_values(src: dict) -> list[bool]:
    if "source_effective_audio_per_frame" in src:
        return [bool(v) for v in src.get("source_effective_audio_per_frame", [])]
    threshold = float(src.get("effective_audio_gain_threshold", 0.05))
    return [
        float(gain) >= threshold
        for gain in src.get("source_amp_gain_per_frame", [])
    ]


def _marker_style(tag: str) -> dict:
    return _MARKER_STYLES.get(tag, {"label": tag.upper(), "color": (64, 210, 255)})


def _overlay_source_label(tag: str) -> str:
    return _SOURCE_SHORT_LABELS.get(tag, tag)


def project_source_to_frame(src_xyz, mic_pos, mic_yaw_deg: float,
                            fov_h_deg: float, fov_v_deg: float,
                            width: int, height: int):
    """Project a source center into the UE camera image plane."""
    sx, sy, sz = [float(v) for v in src_xyz]
    mx, my, mz = [float(v) for v in mic_pos]
    vx, vy, vz = sx - mx, sy - my, sz - mz
    if math.sqrt(vx * vx + vy * vy + vz * vz) < 1e-9:
        return None

    yr = math.radians(float(mic_yaw_deg))
    c, s = math.cos(yr), math.sin(yr)
    x_local = c * vx + s * vy
    y_local = -s * vx + c * vy
    z_local = vz
    azi = math.degrees(math.atan2(y_local, x_local))
    ele = math.degrees(math.atan2(z_local, math.hypot(x_local, y_local)))
    if abs(azi) > fov_h_deg / 2.0 or abs(ele) > fov_v_deg / 2.0:
        return None

    # Positive mic-local azimuth is the listener/camera's left side. Pixel X
    # grows to image-right, so the horizontal image projection is mirrored
    # relative to the azimuth sign.
    px = round(((fov_h_deg / 2.0 - azi) / fov_h_deg) * (width - 1))
    py = round(((fov_v_deg / 2.0 - ele) / fov_v_deg) * (height - 1))
    return (int(max(0, min(width - 1, px))), int(max(0, min(height - 1, py))))


def _draw_marker(draw: ImageDraw.ImageDraw, xy, label: str, color,
                 width: int, height: int):
    x, y = xy
    radius = 9
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        outline=color,
        width=4,
    )
    draw.line((x - radius - 4, y, x + radius + 4, y), fill=color, width=2)
    draw.line((x, y - radius - 4, x, y + radius + 4), fill=color, width=2)

    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    tx = max(4, min(width - text_w - 8, x + 13))
    ty = max(4, min(height - text_h - 8, y - 18))
    draw.rectangle((tx - 3, ty - 2, tx + text_w + 3, ty + text_h + 2),
                   fill=(0, 0, 0))
    draw.text((tx, ty), label, font=font, fill=color)


def load_actor_visual_metadata(clip_dir: Path) -> dict:
    """Load UE-authored visual marker anchors keyed by source tag.

    These centers come from actual actor bounds during render.  They are
    distinct from source_world_xyz_per_frame, which is the acoustic point used
    by spatial metadata.
    """
    path = Path(clip_dir) / "videos" / "actor_visual_metadata.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    out = {}
    for src in payload.get("sources", []):
        tag = src.get("tag")
        if tag:
            out[tag] = src
    return out


def marker_xyz_for_source_frame(src: dict, frame_idx: int,
                                actor_visual_by_tag: dict) -> list[float]:
    """Return the 3D point to mark in the UE review frame."""
    tag = src.get("tag")
    visual = actor_visual_by_tag.get(tag, {})
    centers = visual.get("visual_center_world_xyz_per_frame") or []
    if frame_idx < len(centers):
        return [float(x) for x in centers[frame_idx]]
    return [float(x) for x in src["source_world_xyz_per_frame"][frame_idx]]


def write_ue_marker_video(clip_dir: Path, out_video: Path) -> Path:
    clip_dir = Path(clip_dir)
    spec = json.loads((clip_dir / "spec.json").read_text())
    metadata = json.loads((clip_dir / "apartment_v1_metadata.json").read_text())
    frames_dir = clip_dir / "videos" / "apartment_v1_view0"
    work_dir = REPO_ROOT / "tmp" / "spike_rlr" / "ue_marker_frames" / clip_dir.name
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    mic = metadata["mic_pose_6DoF"]
    mic_pos = mic["pos_m"]
    mic_yaw = float(mic["yaw_deg"])
    camera = spec["camera_configs"][0]
    fov_h = float(camera.get("fov_deg", 90.0))
    fov_v = float(camera.get("fov_v_deg", 60.0))
    fps = int(spec["render_config"]["fps"])
    n_frames = int(metadata["n_frames"])
    actor_visual_by_tag = load_actor_visual_metadata(clip_dir)

    for i in range(n_frames):
        frame_path = frames_dir / f"frame_{i:04d}.png"
        image = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for src in metadata.get("sources", []):
            in_fov = src.get("source_in_fov_per_frame", [])
            if i >= len(in_fov) or not bool(in_fov[i]):
                continue
            xyz = marker_xyz_for_source_frame(src, i, actor_visual_by_tag)
            xy = project_source_to_frame(
                xyz, mic_pos, mic_yaw, fov_h, fov_v, width, height
            )
            if xy is None:
                continue
            style = _marker_style(src["tag"])
            _draw_marker(draw, xy, style["label"], style["color"], width, height)
        image.save(work_dir / f"frame_{i:04d}.png")

    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", str(fps),
        "-i", str(work_dir / "frame_%04d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(out_video),
    ])
    return out_video


def build_overlay_lines(clip_dir: Path) -> list[str]:
    clip_dir = Path(clip_dir)
    spec = json.loads((clip_dir / "spec.json").read_text())
    flags_path = clip_dir / "flags.json"
    flags = json.loads(flags_path.read_text()) if flags_path.exists() else {}
    details_path = clip_dir / "flag_details.json"
    flag_details = json.loads(details_path.read_text()) if details_path.exists() else {}
    per_source_flags = flag_details.get("per_source", {})
    metadata = json.loads((clip_dir / "apartment_v1_metadata.json").read_text())
    n_frames = int(metadata["n_frames"])

    true_flags = [name for name, enabled in flags.items() if enabled]
    flag_text = _format_flag_names(true_flags) if true_flags else "none"
    lines = [
        f"{clip_dir.name} | n_src={len(metadata.get('sources', []))} | flags={flag_text}"
    ]

    spec_by_tag = {s.get("tag"): s for s in spec.get("sources", [])}
    for src in metadata.get("sources", []):
        tag = src["tag"]
        spec_src = spec_by_tag.get(tag, {})
        motion = spec_src.get("motion_style") or spec_src.get("motion") or "unknown"
        category = (
            "silent" if spec_src.get("mute_audio")
            else src.get("category") or spec_src.get("audio_lookup") or "unknown"
        )
        in_fov = [bool(v) for v in src.get("source_in_fov_per_frame", [])]
        visible = [bool(v) for v in src.get("source_visible_from_camera_per_frame", [])]
        occ = [bool(v) for v in src.get("source_occluded_by_furniture_per_frame", [])]
        sound = _effective_audio_values(src)
        line = (
            f"{_overlay_source_label(tag)} {category} {motion} | "
            f"sound {_count(sound)}/{n_frames} | "
            f"FOV {_count(in_fov)}/{n_frames} | "
            f"centerVis {_count(visible)}/{n_frames} ({_true_ranges(visible)}) | "
            f"occ {_count(occ)}/{n_frames}"
        )
        source_true_flags = [
            name for name, enabled in per_source_flags.get(tag, {}).items()
            if enabled
        ]
        if source_true_flags:
            line += f" | src={_format_flag_names(source_true_flags)}"
        lines.append(line)
    return lines


def write_overlay_file(clip_dir: Path) -> Path:
    videos_dir = Path(clip_dir) / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    overlay = videos_dir / "review_overlay.txt"
    overlay.write_text("\n".join(build_overlay_lines(clip_dir)) + "\n")
    return overlay


def _run(cmd):
    subprocess.run(cmd, check=True)


def build_review_videos(clip_dir: Path, python_exe: str = sys.executable) -> dict:
    clip_dir = Path(clip_dir)
    videos_dir = clip_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    spec = clip_dir / "spec.json"
    audio = clip_dir / "binaural.wav"
    ue_video = videos_dir / "apartment_v1_view0.mp4"
    ue_marked = videos_dir / "ue_source_markers.mp4"
    topdown = videos_dir / "topdown_review.mp4"
    topdown_silent = videos_dir / "topdown_review_silent.mp4"
    ue_with_audio = videos_dir / "ue_with_audio.mp4"
    side_by_side = videos_dir / "side_by_side_review.mp4"
    annotated = videos_dir / "side_by_side_review_annotated.mp4"
    overlay = write_overlay_file(clip_dir)

    write_ue_marker_video(clip_dir, ue_marked)
    _run([
        python_exe,
        str(REPO_ROOT / "tools/spike_rlr/render_topdown_2d.py"),
        "--spec", str(spec),
        "--audio", str(audio),
        "--out", str(topdown),
    ])
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(ue_marked),
        "-i", str(audio),
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(ue_with_audio),
    ])
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(ue_with_audio),
        "-i", str(topdown_silent),
        "-filter_complex",
        "[0:v]scale=640:480[a];[1:v]scale=640:480[b];[a][b]hstack=inputs=2[v]",
        "-map", "[v]", "-map", "0:a", "-c:a", "copy",
        str(side_by_side),
    ])
    _run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(side_by_side),
        "-vf",
        "drawtext="
        f"textfile={overlay}:fontcolor=white:fontsize=18:"
        "box=1:boxcolor=black@0.65:x=10:y=10:line_spacing=6",
        "-c:a", "copy",
        str(annotated),
    ])
    return {
        "overlay": overlay,
        "topdown": topdown,
        "ue_with_audio": ue_with_audio,
        "ue_marked": ue_marked,
        "side_by_side": side_by_side,
        "annotated": annotated,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-dir", required=True)
    args = ap.parse_args()
    out = build_review_videos(Path(args.clip_dir))
    print(f"[review] overlay -> {out['overlay']}")
    print(f"[review] ue markers -> {out['ue_marked']}")
    print(f"[review] annotated -> {out['annotated']}")


if __name__ == "__main__":
    main()
