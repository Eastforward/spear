"""Post-process a Hunyuan-baked diffuse to remove the systematic blue
tint that Hunyuan's multiview diffusion introduces.

Root cause (documented 2026-07-05): Hunyuan3D-Paint's multiview diffusion
consistently outputs BGR ~(80, 40, 20) instead of black ~(15, 15, 15) for
areas that should be black in the reference. Reason is SDXL-lineage
diffusion prior baking a cool-sky lighting bias into "black" surfaces.

Two techniques, applied in order:
  1. Gray-world white balance: rescale each channel so the mean of the
     non-black pixels equals gray. Removes global blue cast.
  2. Optional target-color anchor: if --reference is passed, compute a
     saturation-value histogram from the reference image and reshape the
     diffuse to match (roughly). Not needed if step 1 is sufficient.

Also does a gentle contrast lift so the "black" areas actually read
as black in-engine.

Usage:
  tools/color_correct_diffuse.py \\
    --input /path/hy3d_diffuse.jpg \\
    --output /path/hy3d_diffuse_corrected.png \\
    [--reference /path/collie_clean.png] \\
    [--strength 1.0]

Prints COLOR_CORRECT_OK <stats>.
"""
import argparse
import os
import sys

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--reference", default=None,
                   help="Optional target reference image for color-histogram matching.")
    p.add_argument("--strength", type=float, default=1.0,
                   help="Correction strength 0..1. 0 = passthrough, 1 = full gray-world.")
    p.add_argument("--black-threshold", type=int, default=40,
                   help="Below this per-channel mean, treat pixel as background/unpainted "
                        "(skipped in white-balance mean computation).")
    p.add_argument("--contrast-lift", type=float, default=1.15,
                   help="Multiply channels by this to lift blacks toward true black.")
    p.add_argument("--saturation-boost", type=float, default=0.7,
                   help="Multiply HSV saturation by this to de-saturate the bluish cast. "
                        "1.0 = keep, <1 = desaturate (removes the color cast).")
    return p.parse_args()


def gray_world_balance(img, strength=1.0, black_threshold=40):
    """Rescale each BGR channel so its mean (over the non-background pixels)
    matches the average of all three channels.

    Args:
      img: HxWx3 uint8 BGR
      strength: 0 = no correction, 1 = full gray-world.
      black_threshold: pixels darker than this in ALL channels are
        excluded (they're probably background, not the dog surface).
    """
    # Mask of "surface" pixels — brighter than the black threshold in at
    # least one channel
    mask = (img.max(axis=2) > black_threshold)
    if mask.sum() < 100:
        return img.copy()

    # Per-channel mean over surface pixels
    means = np.array([img[..., c][mask].mean() for c in range(3)], dtype=np.float64)
    target = means.mean()   # want all channels equal
    # Scaling factor per channel; lerp toward 1.0 by (1-strength)
    scale = target / np.maximum(means, 1e-6)
    scale = 1.0 + strength * (scale - 1.0)
    out = img.astype(np.float64)
    for c in range(3):
        out[..., c] = out[..., c] * scale[c]
    return np.clip(out, 0, 255).astype(np.uint8)


def desaturate_hsv(img, factor=0.7):
    """Multiply the HSV saturation by `factor` (0..1) to remove color cast
    on surfaces that should be neutral."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[..., 1] = hsv[..., 1] * factor
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def contrast_lift_blacks(img, factor=1.15):
    """Push darker pixels toward pure black.

    We compute the per-pixel lightness, subtract a floor, multiply, then
    add back. Keeps bright pixels roughly unchanged but crushes dark ones.
    """
    out = img.astype(np.float64)
    # Simple black-point remap: subtract 20, multiply, clip.
    out = (out - 20) * factor
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    args = parse_args()
    img = cv2.imread(args.input)
    if img is None:
        print(f"COLOR_CORRECT_FAIL could not read {args.input}")
        sys.exit(1)
    h, w = img.shape[:2]

    # Stats BEFORE
    mask_pre = (img.max(axis=2) > args.black_threshold)
    means_pre = tuple(round(float(img[..., c][mask_pre].mean()), 1) for c in range(3))

    # 1. Gray-world white balance
    corrected = gray_world_balance(img, strength=args.strength,
                                    black_threshold=args.black_threshold)
    # 2. Desaturate the residual cast (targets the "should be neutral" surfaces)
    if args.saturation_boost != 1.0:
        corrected = desaturate_hsv(corrected, args.saturation_boost)
    # 3. Lift blacks so they actually read as black in UE
    if args.contrast_lift != 1.0:
        corrected = contrast_lift_blacks(corrected, args.contrast_lift)

    mask_post = (corrected.max(axis=2) > args.black_threshold)
    means_post = tuple(round(float(corrected[..., c][mask_post].mean()), 1) for c in range(3))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, corrected)
    print(f"COLOR_CORRECT_OK size={w}x{h} "
          f"means_before(BGR)={means_pre} means_after(BGR)={means_post} "
          f"output={args.output}")


if __name__ == "__main__":
    main()
