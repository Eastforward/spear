"""Render an obstacle-aware top-down trajectory preflight from an apartment spec."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


SPEAR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPEAR_ROOT / "tools"))
sys.path.insert(0, str(SPEAR_ROOT / "tools/spike_rlr"))

from scene_two_dogs_apartment import _static_obstacle_bboxes  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    spec_path = Path(args.spec).resolve()
    output_path = Path(args.output).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    categories = json.loads(
        (SPEAR_ROOT / "tools/spike_rlr/apartment_furniture_categories.json").read_text(
            encoding="utf-8"
        )
    )
    obstacles = _static_obstacle_bboxes(spec, categories)
    trajectory = np.asarray(spec["sources"][0]["trajectory_m"], dtype=np.float64)
    contract = spec.get("camera_pass_table_loop_contract", {})
    camera = np.asarray(spec["camera_configs"][0]["pos_m"][:2], dtype=np.float64)
    forward = np.asarray(contract.get("camera_forward_xy", (1.0, 0.0)))
    right = np.asarray(contract.get("camera_right_xy", (0.0, -1.0)))

    figure, axis = plt.subplots(figsize=(10, 9), dpi=160)
    for x0, y0, x1, y1 in obstacles:
        axis.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="#c9c9c9",
                edgecolor="#777777",
                linewidth=0.5,
                alpha=0.45,
            )
        )
    table_bbox = contract.get("target_bbox_ssot_m")
    if table_bbox:
        x0, y0, x1, y1 = table_bbox
        axis.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="#f4a261",
                edgecolor="#9c4f00",
                linewidth=1.5,
                alpha=0.65,
                label="target round table",
            )
        )
    colors = np.linspace(0.0, 1.0, len(trajectory) - 1)
    for index in range(len(trajectory) - 1):
        axis.plot(
            trajectory[index : index + 2, 0],
            trajectory[index : index + 2, 1],
            color=plt.cm.viridis(colors[index]),
            linewidth=2.0,
        )
    axis.scatter(*trajectory[0, :2], color="#d00000", s=70, marker="o", label="start right/rear")
    axis.scatter(*trajectory[-1, :2], color="#005f73", s=70, marker="s", label="end / loop entry")
    left_front_frame = int(contract.get("left_front_nearest_frame", 0))
    axis.scatter(
        *trajectory[left_front_frame, :2],
        color="#9b5de5",
        s=70,
        marker="D",
        label=f"left/front pass frame {left_front_frame}",
    )
    axis.scatter(*camera, color="black", s=90, marker="*", label="camera/mic")
    axis.arrow(*camera, *(forward * 0.9), width=0.025, color="black", length_includes_head=True)
    axis.arrow(*camera, *(right * 0.6), width=0.018, color="#264653", length_includes_head=True)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("SSOT X (m)")
    axis.set_ylabel("SSOT Y (m)")
    axis.set_title(spec.get("trajectory_profile", "apartment trajectory"))
    axis.grid(True, linewidth=0.3, alpha=0.35)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    print(f"TRAJECTORY_PREFLIGHT_OK {output_path}")


if __name__ == "__main__":
    main()
