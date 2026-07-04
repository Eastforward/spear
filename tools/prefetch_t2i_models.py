"""Task 0: prefetch Flux.1 dev + SDXL base 1.0 weights into HF cache.

Two important quirks (see plan's env-prep appendix):
  1. Gated repo Flux.1-dev requires real HF endpoint. The globally-set
     HF_ENDPOINT=https://hf-mirror.com strips Authorization headers and
     will return 403 for gated repos. We override to https://huggingface.co.
  2. Flux.1-dev requires license acceptance at huggingface.co UI. If not
     accepted, downloads return GatedRepoError — surface a friendly message.

Runs in the hunyuan3d env. Designed to be launched in the background
BEFORE Task 1 so weights download while the early tasks run. Idempotent.

Usage (in background):
  nohup /data/jzy/miniconda3/envs/hunyuan3d/bin/python \\
    /data/jzy/code/SPEAR/tools/prefetch_t2i_models.py \\
    > /tmp/prefetch_t2i.log 2>&1 &
"""
import os
import sys

# CRITICAL: huggingface_hub reads HF_ENDPOINT into a module-level constant
# at import time. Setting os.environ AFTER `from huggingface_hub import ...`
# has NO effect. We must set it here, before ANY hf import happens.
# Both target models live on real HF (mirror strips Authorization on gated
# Flux; SDXL happens to work fine on real HF too — verified 2026-07-05).
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

# (repo_id, description)
MODELS = [
    ("black-forest-labs/FLUX.1-dev",
        "Flux.1 dev (gated; requires license accepted at huggingface.co)"),
    ("stabilityai/stable-diffusion-xl-base-1.0",
        "SDXL base 1.0 (open)"),
]


def main():
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError

    print(f"[prefetch] HF_ENDPOINT={os.environ['HF_ENDPOINT']}", flush=True)
    for repo_id, desc in MODELS:
        try:
            print(f"[prefetch] {desc}", flush=True)
            path = snapshot_download(
                repo_id=repo_id,
                ignore_patterns=["*.bin", "*.msgpack", "*.h5", "*.onnx"],
                max_workers=4,
            )
            print(f"PREFETCH_OK {repo_id} -> {path}", flush=True)
        except GatedRepoError:
            print(f"PREFETCH_FAIL {repo_id}: gated repo denied.", flush=True)
            print(f"  Accept license at https://huggingface.co/{repo_id} while logged in.", flush=True)
        except Exception as e:
            print(f"PREFETCH_FAIL {repo_id}: {type(e).__name__}: {e}", flush=True)

    print("PREFETCH_DONE", flush=True)


if __name__ == "__main__":
    main()
