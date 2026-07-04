"""Task 7: text-to-image reference generator for the Hunyuan pipeline.

Uses diffusers' Flux pipeline (or SDXL fallback) to produce a clean
white-background dog photo suitable for Hunyuan3D-Paint's multiview
diffusion. Applies a fixed prompt template that biases toward Hunyuan-
friendly output (standing side view, plain background).

Runs in the hunyuan3d env; the same GPU that Hunyuan later uses. Prints
FLUX_GEN_OK on success.
"""
import argparse
import os
import sys

# Force real HF endpoint for the model resolution / snapshot check —
# same reason as tools/prefetch_t2i_models.py. Must happen before any
# huggingface_hub import.
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

import torch


TEMPLATE = ("{prompt}, full body, standing side view, "
            "plain white background, studio photo, photorealistic, "
            "detailed fur, professional pet photography")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="auto",
                   choices=["auto", "flux_dev", "sdxl_base"],
                   help="auto = use flux_dev if cached else sdxl_base; explicit choice forces one")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--no-rembg", action="store_true",
                   help="Skip background removal (Hunyuan pipeline actually adds white bg itself)")
    return p.parse_args()


def _is_cached(repo_id):
    """Return True if the HF snapshot is already fully downloaded."""
    import huggingface_hub
    try:
        # If the snapshot dir exists and has a config.json, treat as cached
        # (huggingface_hub.snapshot_download will be a no-op then).
        path = huggingface_hub.snapshot_download(repo_id, local_files_only=True)
        return os.path.isdir(path)
    except Exception:
        return False


def resolve_model(model):
    """Turn 'auto' into a concrete choice based on what's cached locally."""
    if model != "auto":
        return model
    if _is_cached("black-forest-labs/FLUX.1-dev"):
        return "flux_dev"
    if _is_cached("stabilityai/stable-diffusion-xl-base-1.0"):
        return "sdxl_base"
    raise SystemExit("FLUX_GEN_FAIL neither flux_dev nor sdxl_base cached; run Task 0 first")


def load_pipeline(model):
    from diffusers import FluxPipeline, StableDiffusionXLPipeline
    if model == "flux_dev":
        return FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        )
    if model == "sdxl_base":
        return StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
        )
    raise SystemExit(f"unknown model {model}")


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    model = resolve_model(args.model)
    prompt = TEMPLATE.format(prompt=args.prompt)
    print(f"[flux_gen] model={model} prompt={prompt!r}", flush=True)

    pipe = load_pipeline(model).to("cuda")
    pipe.set_progress_bar_config(disable=False)

    steps = args.steps or (28 if model == "flux_dev" else 30)
    gen = None
    if args.seed is not None:
        gen = torch.Generator("cuda").manual_seed(args.seed)

    image = pipe(
        prompt=prompt,
        width=args.width, height=args.height,
        num_inference_steps=steps,
        generator=gen,
    ).images[0]

    if not args.no_rembg:
        try:
            from rembg import remove
            image = remove(image)
        except ImportError:
            print("[flux_gen] rembg not installed; skipping bg removal", flush=True)

    image.save(args.output)
    print(f"FLUX_GEN_OK output={args.output} model={model} steps={steps} "
          f"seed={args.seed if args.seed is not None else 'random'}")


if __name__ == "__main__":
    main()
