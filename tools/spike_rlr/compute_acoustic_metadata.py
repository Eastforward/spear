"""Per-source acoustic + spatial metadata for apartment_v1 clip.

Computes:
  - per-frame source_world_xyz (from scene composer trajectory)
  - per-frame source_azi_ele_dist_mic_local (spherical, mic frame)
  - per-frame source_amp_gain in [0,1] (RMS from per-source binaural WAV
    divided by clip peak; is_silent[t] = gain < 0.05 is a downstream derivation)
  - per-frame drr_db (distance-driven proxy; real per-frame IR extraction
    is deferred to Plan 2)

Output: apartment_v1_metadata.json in the clip's out-dir (schema in this
module's docstring below).

This is Plan-1 metadata. Plan 2 adds M1/M2/M3-ready expansions:
  - real per-frame DRR from RLR IR export
  - occlusion booleans per frame (O-vis raycast)
  - room_metadata.json (RT60 measurement) written once per room
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))
from scene_two_dogs_apartment import (  # noqa: E402
    compose_two_dog_scene_apartment,
    _kept_furniture_bboxes, _shell_wall_bboxes,
)
from apartment_builtin_obstacles import (  # noqa: E402
    apartment_builtin_visual_obstacle_bboxes_xyz,
)
from profiling import StageTimer  # noqa: E402
from visibility import batch_frame_visibility  # noqa: E402
from source_trajectory import acoustic_trajectory  # noqa: E402


# Mapping from audio_lookup key in the spec -> semantic source category.
# Plan 3 will expand this via the 8-class audio library.
_LOOKUP_TO_CATEGORY = {
    "silent": "silent",
    "dog_bark": "dog_bark",
    "dog_growl": "dog_growl",
    "dog_sharp_bark": "dog_sharp_bark",
    "cat_meow": "cat_meow",
    "cat_purring": "cat_purring",
    "speech": "speech",
    "talking": "speech",
    "conversation": "conversation",
    "music_piano": "music_piano",
    "wolf_howl": "music_piano",   # legacy debug lookup; current animal specs
                                    # should use species-correct audio lookups.
}


EFFECTIVE_AUDIO_GAIN_THRESHOLD = 0.05
_SYNTHETIC_AUDIO_LOOKUPS = {"music_piano", "wolf_howl"}


def _source_is_synthetic(src_spec: dict) -> bool:
    if "is_synthetic" in src_spec:
        return bool(src_spec["is_synthetic"])
    return src_spec.get("audio_lookup") in _SYNTHETIC_AUDIO_LOOKUPS


def effective_audio_frames_from_gains(
    gains: list[float],
    threshold: float = EFFECTIVE_AUDIO_GAIN_THRESHOLD,
) -> list[bool]:
    """Return per-frame wet-audio activity from normalized binaural RMS gain."""
    return [float(gain) >= float(threshold) for gain in gains]


def azi_ele_dist_local(src_xyz, mic_xyz, mic_yaw_deg):
    """Return (azi_deg, ele_deg, dist_m) with source relative to mic-local frame.

    Convention:
      - mic-forward at yaw=0 (SSOT) is +X world (per apartment_v1_spec.json).
      - mic yaw rotates in the XY plane; positive yaw rotates mic-forward CCW
        (right-handed, +Z is up).
      - azi_deg = angle CCW from mic-forward, in [-180, 180]. Positive azi
        means source is on mic's LEFT if mic yaw's rotation matches audio
        convention (+X_local rot to face source's direction).
      - ele_deg = angle above/below XY-plane, in [-90, 90].
    """
    v = np.asarray(src_xyz) - np.asarray(mic_xyz)
    yaw_rad = np.deg2rad(mic_yaw_deg)
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    # Rotate world XY by -yaw so mic-forward aligns with +X_local.
    x_local = c * v[0] + s * v[1]
    y_local = -s * v[0] + c * v[1]
    z_local = v[2]
    dist = float(np.linalg.norm(v))
    # azi = atan2(y_local, x_local): 0 = directly ahead, +90 = mic-left
    azi_deg = float(np.degrees(np.arctan2(y_local, x_local)))
    ele_deg = float(np.degrees(np.arctan2(z_local, np.hypot(x_local, y_local))))
    return azi_deg, ele_deg, dist


def per_frame_amp_gain(bin_wav_path: Path, n_frames: int) -> list[float]:
    """RMS envelope per frame from a binaural WAV, normalized by peak."""
    if not bin_wav_path.exists():
        # Fallback: silent audio if per-source file wasn't written
        return [0.0] * n_frames
    x, sr = sf.read(str(bin_wav_path), always_2d=True)
    L = x.shape[0]
    win = max(1, L // n_frames)
    peak = float(np.abs(x).max()) + 1e-9
    gains = []
    for k in range(n_frames):
        s = k * win
        e = min(s + win, L)
        rms = float(np.sqrt(np.mean(x[s:e] ** 2)))
        gains.append(min(1.0, rms / peak))
    return gains


def drr_proxy_per_frame(src_traj, mic_pos):
    """Distance-driven DRR proxy in dB.

    Empirical curve tuned to give sensible ranges for a small apartment:
      d = 0.1 m  -> ~ +32 dB (very close)
      d = 1.0 m  -> ~ +12 dB
      d = 5.0 m  -> ~ +0 dB
      d = 10.0 m -> ~ -6 dB
    Real per-frame IR extraction (which RLR could give us) is Plan-2.
    """
    drrs = []
    for xyz in src_traj:
        d = float(np.linalg.norm(np.asarray(xyz) - np.asarray(mic_pos)))
        drrs.append(round(12.0 - 20.0 * np.log10(max(d, 0.1)), 2))
    return drrs


def compute(spec_path: Path, out_dir: Path, csv_path: Path,
            clip_id: str = "apartment_v1_000"):
    with StageTimer("metadata_extract", clip_id=clip_id, csv_path=csv_path):
        spec = json.loads(spec_path.read_text())
        n_frames = int(spec["render_config"]["n_frames"])
        fps = int(spec["render_config"]["fps"])
        mic_pos = np.asarray(spec["mic"]["pos_m"])
        mic_yaw = float(spec["mic"]["yaw_deg"])
        scene = compose_two_dog_scene_apartment(spec_path)

        # Build 3D obstacle bboxes for visibility check (Plan 1.5.C).
        # Reuse the same kept-furniture + shell-wall lists that the scene
        # composer used, and lift them into Z ranges: furniture 0-1.5m,
        # walls 0-2.8m. Windows/doors are already in shell walls.
        cats = json.loads(
            (REPO_ROOT / "tools/spike_rlr/apartment_furniture_categories.json").read_text()
        )
        furn_xy = _kept_furniture_bboxes(spec, cats)
        shell_xy = _shell_wall_bboxes(spec)
        obstacles_3d = []
        for x0, y0, x1, y1 in furn_xy:
            obstacles_3d.append(((x0, y0, 0.0), (x1, y1, 1.5)))
        obstacles_3d.extend(apartment_builtin_visual_obstacle_bboxes_xyz(spec))
        for x0, y0, x1, y1 in shell_xy:
            obstacles_3d.append(((x0, y0, 0.0), (x1, y1, 2.8)))
        fov_h = float(spec["camera_configs"][0]["fov_deg"])
        fov_v = 60.0  # conventional vertical FOV; can be added to spec later.

        # For each source, locate its per-source binaural WAV (LOW quality
        # is the default Plan-1 render).
        bin_dir = out_dir / "binaural_native"

        sources_out = []
        for pl in scene.animals:
            src_spec = [s for s in spec["sources"] if s["tag"] == pl.tag][0]
            source_traj = acoustic_trajectory(pl.trajectory_m, src_spec)
            audio_lookup = src_spec.get("audio_lookup", "unknown")
            bin_candidates = [
                out_dir / f"binaural_{pl.tag}_binaural.wav",
                bin_dir / f"audio_B_rlr_LOW_binaural_native_{pl.tag}_binaural.wav",
            ]
            bin_wav = next((p for p in bin_candidates if p.exists()), bin_candidates[0])

            muted = bool(src_spec.get("mute_audio") or audio_lookup == "silent")
            gains = [0.0] * n_frames if muted else per_frame_amp_gain(bin_wav, n_frames)
            effective_audio = effective_audio_frames_from_gains(gains)

            azi_ele_dist = [azi_ele_dist_local(xyz, mic_pos, mic_yaw)
                             for xyz in source_traj]
            drrs = drr_proxy_per_frame(source_traj, mic_pos)

            # Per-frame visibility (Plan 1.5.C)
            traj_xyz = source_traj
            vis = batch_frame_visibility(
                traj_xyz, mic_pos, mic_yaw,
                fov_h_deg=fov_h, fov_v_deg=fov_v,
                obstacles_xyz=obstacles_3d,
            )

            sources_out.append({
                "tag": pl.tag,
                "category": "silent" if muted else _LOOKUP_TO_CATEGORY.get(audio_lookup, "unknown"),
                "is_synthetic": _source_is_synthetic(src_spec),
                "audio_source_height_offset_m": float(
                    src_spec.get("audio_source_height_offset_m", 0.0)
                ),
                "drr_db_per_frame": drrs,
                "source_world_xyz_per_frame": [
                    [float(x) for x in xyz] for xyz in source_traj
                ],
                "source_azi_ele_dist_mic_local_per_frame": [list(t) for t in azi_ele_dist],
                "source_amp_gain_per_frame": gains,
                "source_effective_audio_per_frame": effective_audio,
                "effective_audio_frame_count": int(np.count_nonzero(effective_audio)),
                "effective_audio_gain_threshold": EFFECTIVE_AUDIO_GAIN_THRESHOLD,
                "source_in_fov_per_frame": [bool(x) for x in vis["in_fov"]],
                "source_occluded_by_furniture_per_frame":
                    [bool(x) for x in vis["occluded_by_furniture"]],
                "source_visible_from_camera_per_frame":
                    [bool(x) for x in vis["visible"]],
            })

        payload = {
            "clip_id": clip_id,
            "spec_path": str(spec_path),
            "duration_s": float(spec["render_config"]["duration_s"]),
            "n_frames": n_frames,
            "fps": fps,
            "mic_pose_6DoF": {
                "pos_m": [float(x) for x in mic_pos],
                "yaw_deg": mic_yaw, "pitch_deg": 0.0, "roll_deg": 0.0,
            },
            "sources": sources_out,
            "_note": "Plan-1 metadata: drr_db_per_frame is a distance proxy. "
                      "Plan-2 will replace with per-frame IR-derived DRR + "
                      "occlusion booleans + separate room_metadata.json.",
        }

        out_path = out_dir / "apartment_v1_metadata.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"[metadata] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(REPO_ROOT / "data" / "apartment_v1_spec.json"))
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "tmp" / "spike_output_apartment"))
    ap.add_argument("--clip-id", default="apartment_v1_000")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    csv_path = out_dir / "profile_per_clip.csv"
    compute(Path(args.spec), out_dir, csv_path, args.clip_id)


if __name__ == "__main__":
    main()
