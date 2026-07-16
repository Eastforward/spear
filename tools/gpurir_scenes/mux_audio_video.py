"""Downmix 4ch audio to stereo (front-facing binaural approximation) and mux
with each of the 4 view mp4s in a scene dir.

Downmix policy:
  channels 0..3 are the tetrahedral capsules from run_audio_pass.
  Approx binaural stereo: FL = 0.5*c0 + 0.5*c2  (front-left facing)
                          FR = 0.5*c1 + 0.5*c3  (front-right facing)
This is a simple omni-pair sum; produces useful L/R separation for
+X = right / -X = left mic geometry.
"""
import argparse
import os
import subprocess


def _downmix_stereo(in_wav, out_wav):
    subprocess.run([
        "ffmpeg", "-y", "-i", in_wav,
        "-af", "pan=stereo|FL=0.5*c0+0.5*c2|FR=0.5*c1+0.5*c3",
        out_wav,
    ], check=True, capture_output=True)


def _mux(video_mp4, stereo_wav, out_mp4):
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_mp4, "-i", stereo_wav,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", out_mp4,
    ], check=True, capture_output=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene-dir", required=True,
                   help="dir containing audio.wav and apartment/ + shoebox/ view mp4s")
    args = p.parse_args()

    audio_wav = os.path.join(args.scene_dir, "audio.wav")
    stereo_wav = os.path.join(args.scene_dir, "audio_stereo.wav")
    assert os.path.isfile(audio_wav), f"missing {audio_wav}"
    _downmix_stereo(audio_wav, stereo_wav)

    for room in ("apartment", "shoebox"):
        room_dir = os.path.join(args.scene_dir, room)
        if not os.path.isdir(room_dir):
            print(f"[mux] skip {room}: no dir")
            continue
        for i in range(4):
            v = os.path.join(room_dir, f"view{i}.mp4")
            if not os.path.isfile(v):
                print(f"[mux] skip {room}/view{i}: no mp4")
                continue
            out = os.path.join(room_dir, f"view{i}_with_audio.mp4")
            _mux(v, stereo_wav, out)
            print(f"[mux] wrote {out}")
    print(f"MUX_DONE {args.scene_dir}")


if __name__ == "__main__":
    main()
