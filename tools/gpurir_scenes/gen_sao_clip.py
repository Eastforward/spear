"""Stable Audio Open 1.0 generator using LOCAL checkpoint (no HF fetch).

Uses stable-audio-tools' generate_diffusion_cond directly against the raw
checkpoint at /data/datasets/omniaudio/stable-audio-open/, avoiding gated HF
downloads.

Run with hunyuan3d python (has torch + stable-audio-tools):
  /data/jzy/miniconda3/envs/hunyuan3d/bin/python gen_sao_clip.py <prompt> <out_wav>
"""
import argparse
import json
import os
import sys


SAO_CKPT_DIR = "/data/datasets/omniaudio/stable-audio-open"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("prompt")
    p.add_argument("out_wav")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_wav) or ".", exist_ok=True)

    import torch
    import torchaudio
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    from stable_audio_tools.models.factory import create_model_from_config

    with open(os.path.join(SAO_CKPT_DIR, "base_model_config.json")) as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)
    ckpt_path = os.path.join(SAO_CKPT_DIR, "base_model.ckpt")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model = model.to("cuda").eval()

    sample_rate = model_config["sample_rate"]
    sample_size = int(args.seconds * sample_rate)

    conditioning = [{
        "prompt": args.prompt,
        "seconds_start": 0,
        "seconds_total": args.seconds,
    }]

    # SAO uses rectified_flow objective; use its native sampler.
    with torch.cuda.amp.autocast():
        audio = generate_diffusion_cond(
            model,
            steps=args.steps,
            cfg_scale=7,
            conditioning=conditioning,
            sample_size=sample_size,
            sampler_type="pingpong",  # rectified_flow-compatible
            device="cuda",
            seed=args.seed,
        )
    # audio shape is (batch=1, channels, samples). Trim to sample_size and save.
    audio = audio.squeeze(0).cpu().float()
    audio = audio[:, :sample_size]
    torchaudio.save(args.out_wav, audio, sample_rate=sample_rate)
    print(f"SAO_OK wrote {args.out_wav}  shape={list(audio.shape)}", flush=True)


if __name__ == "__main__":
    main()
