"""Side-by-side figure: A group (SPEAR/UE) vs C group (Habitat).

Shows 3 timestamps (early / middle / occluded) x 2 backends. The gap in
visual quality (PBR textures + lighting vs Habitat's headlight raster) is
exactly the argument for 档 ① swap-in: keep SPEAR/UE visuals, swap in
RLR audio, avoid the full-stack Habitat migration.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = REPO_ROOT / "tmp" / "spike_output" / "analysis" / "visual_gap.png"


def _first_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    a_ue_dir = REPO_ROOT / "tmp" / "spike_output" / "videos" / "A_gpurir_ue"
    c_hab_dir = REPO_ROOT / "tmp" / "spike_rlr" / "habitat_frames" / "view0"

    frame_ids = [0, 20, 60]
    labels = [
        f"t=0.00s (start)\nhusky in front, unoccluded",
        f"t=1.33s (mid detour)\nhusky at right, edge of FoV",
        f"t=4.00s (fully occluded)\nhusky behind sofa",
    ]

    fig, axes = plt.subplots(2, len(frame_ids), figsize=(15, 8))
    fig.suptitle(
        "Visual quality: A. SPEAR/UE (top row) vs C. Habitat (bottom row) "
        "on the same shoebox v2 scene\n"
        "Same geometry, same dog positions, same lighting spec\n"
        "→ SPEAR/UE ships polished PBR + Quaternius dog fur; "
        "Habitat is bare rasterizer + T-pose dog. Argument for keeping SPEAR "
        "visuals + swapping only the audio backend (档 ①).",
        fontsize=11, fontweight='bold',
    )

    for col, (frame_id, label) in enumerate(zip(frame_ids, labels)):
        a_path = a_ue_dir / f"view0_frame_{frame_id:04d}.png"
        c_path = c_hab_dir / f"frame_{frame_id:03d}.png"

        if a_path.exists():
            axes[0, col].imshow(mpimg.imread(str(a_path)))
        axes[0, col].set_title(f"A. SPEAR/UE  |  frame {frame_id}", fontsize=10)
        axes[0, col].axis('off')

        if c_path.exists():
            axes[1, col].imshow(mpimg.imread(str(c_path)))
        axes[1, col].set_title(f"C. Habitat  |  frame {frame_id}", fontsize=10)
        axes[1, col].axis('off')

        axes[1, col].text(0.5, -0.08, label, transform=axes[1, col].transAxes,
                          ha='center', fontsize=9, color='#555')

    plt.subplots_adjust(top=0.85, hspace=0.15, wspace=0.05)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches='tight')
    plt.close(fig)
    print(f"[visual_gap] wrote {out}")


if __name__ == "__main__":
    main()
