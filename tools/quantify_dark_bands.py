"""Quantify the alleged dark bands with ROI mean/std, per frame, per version.

We want to answer: are the 'dark bands' actually darker than surrounding pixels,
and by how much? Or are they perceptual illusion next to a bright sunlit patch?

Approach:
  - Use frame 0027 (camera facing window, symmetric, both wall-foot strips visible)
  - Define 4 ROIs in pixel coords:
      * wall_foot_strip_left  : narrow horizontal band on left wall just above floor line
      * wall_normal_left      : same width band ~40cm above wall_foot_strip_left
      * floor_near_wall       : floor strip 5-15cm from left wall, mid-frame
      * floor_center          : floor strip near frame center (reference)
  - Compute mean, std, min per ROI in RGB and in luminance (0.299R+0.587G+0.114B)
  - Do it for 3 versions: codex baseline, wall_overlap_fix, ao_off_diag
  - Write a table + a side-by-side annotated PNG showing the ROIs

Also compute frame 0000 numbers where dark patch is in front-left floor.
"""
import os
import cv2
import numpy as np


ROOT = "/data/jzy/code/SPEAR/tmp/render_gpurir_room"
OUT = "/tmp/quantify_dark_bands"
os.makedirs(OUT, exist_ok=True)

VERSIONS = [
    ("baseline_6000_p30", "codex_tune_light6000_pitch_m30"),
    ("wall_overlap_fix",  "claude_wall_overlap_fix"),
    ("ao_off",            "claude_ao_off_diag"),
]

# ROIs on frame 0027 (1280x720, camera facing window from the south, walls left/right)
# Frame layout:
#   y=0 top of image
#   window occupies upper-center
#   floor occupies lower half
#   left wall visible at x<180
#   right wall visible at x>1100
#
# Values chosen by inspecting frame 0027; we'll draw them for visual QA.
ROIS_F27 = {
    # Left wall foot strip: on left wall right at floor line.
    # After looking at f0027, floor line ~y=360; left wall x range ~0..180.
    "wall_foot_strip_left":  ((30, 355), (170, 375)),   # (x0,y0)-(x1,y1)
    "wall_normal_left":      ((30, 220), (170, 240)),
    "wall_foot_strip_right": ((1110, 355), (1250, 375)),
    "wall_normal_right":     ((1110, 220), (1250, 240)),
    # Floor near wall (left) vs center
    "floor_near_wall_left":  ((30, 400), (170, 440)),
    "floor_center":          ((560, 500), (720, 560)),
}

# ROIs on frame 0000 (dog + sun patch on right, dark patch bottom-left)
# f0000 layout: dog at ~x=580, sun patch spans ~x=800-1100 y=250-380,
# dark patch mostly ~x=0-300 y=450-680.
ROIS_F00 = {
    "floor_dark_bottom_left": ((30, 490), (300, 660)),
    "floor_sun_patch":        ((820, 300), (1080, 380)),
    "floor_neutral_center":   ((450, 500), (620, 640)),
    "wall_foot_strip_left":   ((10, 335), (180, 355)),
    "wall_normal_left":       ((10, 220), (180, 240)),
}


def load(rel):
    path = os.path.join(ROOT, rel)
    if not os.path.exists(path):
        return None, path
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR uint8
    return img, path


def luma(img):
    # BGR -> luminance (ITU-R BT.601). cv2 loads BGR so channel order is [B,G,R].
    b, g, r = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def roi_stats(img, x0y0, x1y1):
    (x0, y0), (x1, y1) = x0y0, x1y1
    patch = img[y0:y1, x0:x1]
    lum = luma(patch)
    return {
        "mean": float(lum.mean()),
        "std": float(lum.std()),
        "min": float(lum.min()),
        "max": float(lum.max()),
        "n": int(lum.size),
    }


def analyze_frame(frame_name, rois):
    print(f"\n=== FRAME {frame_name} ===")
    header = f"{'version':<24}" + "  ".join(f"{k:>26}" for k in rois.keys())
    print(header)
    rows = []
    imgs_for_annot = {}
    for label, subdir in VERSIONS:
        img, path = load(f"{subdir}/{frame_name}")
        if img is None:
            print(f"{label:<24}  MISSING: {path}")
            continue
        imgs_for_annot[label] = img
        vals = []
        cells = []
        for k, (a, b) in rois.items():
            s = roi_stats(img, a, b)
            vals.append((k, s))
            cells.append(f"{s['mean']:6.1f}±{s['std']:4.1f}")
        row_str = f"{label:<24}" + "  ".join(f"{c:>26}" for c in cells)
        print(row_str)
        rows.append((label, vals))

    # Compute deltas: wall_foot vs wall_normal (both sides), floor_near_wall vs floor_center
    print("\nDELTAS (wall_foot - wall_normal), positive means foot IS darker:")
    for label, vals in rows:
        d = {k: s for k, s in vals}
        deltas = []
        for foot_key in ("wall_foot_strip_left", "wall_foot_strip_right"):
            norm_key = foot_key.replace("foot_strip", "normal").replace("strip_", "")
            # actual key is 'wall_normal_left' / 'wall_normal_right'
            norm_key2 = "wall_normal_left" if "left" in foot_key else "wall_normal_right"
            if foot_key in d and norm_key2 in d:
                deltas.append((foot_key, d[foot_key]["mean"] - d[norm_key2]["mean"]))
        for fkey in ("floor_dark_bottom_left", "floor_near_wall_left"):
            ref = "floor_neutral_center" if "dark_bottom_left" in fkey else "floor_center"
            if fkey in d and ref in d:
                deltas.append((fkey, d[fkey]["mean"] - d[ref]["mean"]))
        for k, v in deltas:
            arrow = "DARKER" if v < 0 else "brighter"
            print(f"  {label:<24} {k:<32} delta={v:+7.2f}  {arrow}")

    # Emit an annotated PNG (baseline only) showing where the ROIs are.
    base_label, base_subdir = VERSIONS[0]
    base_img, _ = load(f"{base_subdir}/{frame_name}")
    if base_img is not None:
        annot = base_img.copy()
        for name, ((x0, y0), (x1, y1)) in rois.items():
            color = (0, 255, 0)
            if "foot" in name or "dark" in name or "near_wall" in name:
                color = (0, 0, 255)  # red for the ROIs we suspect are darker
            cv2.rectangle(annot, (x0, y0), (x1, y1), color, 2)
            cv2.putText(annot, name, (x0, max(15, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1, cv2.LINE_AA)
        out = os.path.join(OUT, f"annot_{frame_name}")
        cv2.imwrite(out, annot)
        print(f"annotated ROIs -> {out}")


analyze_frame("frame_0027.png", ROIS_F27)
analyze_frame("frame_0000.png", ROIS_F00)
print(f"\nAll outputs in {OUT}")
