"""Headless matplotlib 4-view preview renderer for mesh direction audit.

Renders a 2×2 grid PNG:
  Top-left:  +HEAD view (looking down detected head direction)
  Top-right: -HEAD view (looking down opposite direction)
  Bottom-left:  Top-down (birds-eye) with red arrow pointing head
  Bottom-right: Side view with red arrow pointing head + confidence text

Zero GUI dependency — matplotlib 'Agg' backend writes PNG to disk.
Human reviewer opens the PNG in Cursor/VSCode remote / web UI.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — no GUI required
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def _load_mesh(mesh_path: Path):
    """Load mesh via trimesh; concatenate if scene."""
    scene = trimesh.load(str(mesh_path))
    if isinstance(scene, trimesh.Scene):
        geoms = list(scene.geometry.values())
        if not geoms:
            raise ValueError(f"empty scene {mesh_path}")
        m = trimesh.util.concatenate(geoms)
    else:
        m = scene
    return m


def _draw_mesh_view(ax, mesh, elev, azim, title, arrow_start=None, arrow_end=None):
    coll = Poly3DCollection(
        mesh.vertices[mesh.faces],
        alpha=0.35, edgecolor="k", linewidth=0.1, facecolor="#87ceeb",
    )
    ax.add_collection3d(coll)
    ax.set_xlim(mesh.bounds[:, 0])
    ax.set_ylim(mesh.bounds[:, 1])
    ax.set_zlim(mesh.bounds[:, 2])
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    if arrow_start is not None and arrow_end is not None:
        ax.plot(
            [arrow_start[0], arrow_end[0]],
            [arrow_start[1], arrow_end[1]],
            [arrow_start[2], arrow_end[2]],
            color="red", linewidth=3.0,
        )
        # arrowhead
        ax.scatter(
            [arrow_end[0]], [arrow_end[1]], [arrow_end[2]],
            color="red", s=100, marker="^",
        )


def render_direction_preview(mesh_path, detection_result, out_png_path) -> None:
    """Write a 4-view PNG preview.

    Args:
      mesh_path: path to .glb / .obj (trimesh loadable)
      detection_result: HeadDetectionResult from detect_head_axis()
      out_png_path: where to write .png
    """
    mesh_path = Path(mesh_path)
    out_png_path = Path(out_png_path)
    m = _load_mesh(mesh_path)

    head_dir = detection_result.head_direction
    body_center = m.vertices.mean(axis=0)
    # Arrow: from center to +30% along head direction (bbox-scaled)
    bbox_size = m.bounds[1] - m.bounds[0]
    scale = 0.4 * bbox_size.max()
    arrow_start = body_center
    arrow_end = body_center + head_dir * scale

    fig = plt.figure(figsize=(10, 10))

    # Compute azimuth angles for the two "along-head" views
    # matplotlib 3d convention: azim=0 -> looking from +X; azim=90 -> from +Y
    head_azim = np.degrees(np.arctan2(head_dir[1], head_dir[0]))
    head_elev = np.degrees(np.arctan2(head_dir[2],
                                        np.hypot(head_dir[0], head_dir[1])))

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    _draw_mesh_view(ax1, m,
                     elev=head_elev + 20, azim=head_azim,
                     title="+HEAD view (looking WITH head arrow)",
                     arrow_start=arrow_start, arrow_end=arrow_end)

    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    _draw_mesh_view(ax2, m,
                     elev=head_elev + 20, azim=head_azim + 180,
                     title="-HEAD view (looking AGAINST head arrow)",
                     arrow_start=arrow_start, arrow_end=arrow_end)

    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    _draw_mesh_view(ax3, m, elev=90, azim=0,
                     title="Top-down (red arrow = detected head)",
                     arrow_start=arrow_start, arrow_end=arrow_end)

    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    _draw_mesh_view(ax4, m, elev=5, azim=90,
                     title="Side view",
                     arrow_start=arrow_start, arrow_end=arrow_end)

    # Suptitle: detection summary
    signals_str = ", ".join(f"{k}={v:+d}" for k, v in detection_result.signals.items())
    fig.suptitle(
        f"[{mesh_path.name}]\n"
        f"Detected head direction: [{head_dir[0]:+.2f}, {head_dir[1]:+.2f}, {head_dir[2]:+.2f}]  |  "
        f"Confidence: {detection_result.confidence:.0%}  |  "
        f"Unanimous: {detection_result.unanimous}\n"
        f"Signals: {signals_str} (total votes: {detection_result.total_votes:+d})\n"
        f"↳ Does the red arrow point at the dog's HEAD? (approve if yes)",
        fontsize=10, y=0.995,
    )

    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png_path), dpi=80, bbox_inches="tight")
    plt.close(fig)


def _draw_review_view(ax, mesh, elev, azim, title, bbox_max_extent,
                        target_head_axis=(1, 0, 0), up_axis=(0, 1, 0)):
    """Render mesh with world axes overlaid. Big green arrow along +X
    (target head direction) and blue arrow along +Y (target up)."""
    coll = Poly3DCollection(
        mesh.vertices[mesh.faces],
        alpha=0.4, edgecolor="k", linewidth=0.15, facecolor="#87ceeb",
    )
    ax.add_collection3d(coll)
    # Fix world-axis-centered limits so the giant reference arrows are
    # visible even if the mesh is off-center
    lim = 1.3 * bbox_max_extent
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("X (head→)", color="#0a0", fontweight="bold")
    ax.set_ylabel("Y (up)",   color="#00a", fontweight="bold")
    ax.set_zlabel("Z (side)", color="#666")

    # World reference arrows drawn from origin
    L = lim * 0.8
    # +X = target head direction (GREEN, big)
    tx = np.array(target_head_axis) * L
    ax.plot([0, tx[0]], [0, tx[1]], [0, tx[2]],
            color="#0a0", linewidth=4.0)
    ax.scatter([tx[0]], [tx[1]], [tx[2]], color="#0a0", s=250, marker=">")
    ax.text(tx[0] * 1.05, tx[1] * 1.05, tx[2] * 1.05, "HEAD →",
            color="#0a0", fontsize=12, fontweight="bold")

    # +Y = up (BLUE)
    uy = np.array(up_axis) * L
    ax.plot([0, uy[0]], [0, uy[1]], [0, uy[2]],
            color="#00a", linewidth=3.0)
    ax.scatter([uy[0]], [uy[1]], [uy[2]], color="#00a", s=150, marker="^")
    ax.text(uy[0] * 1.05, uy[1] * 1.05, uy[2] * 1.05, "UP",
            color="#00a", fontsize=11, fontweight="bold")


def render_review_preview(mesh_path, out_png_path,
                           note: str = "") -> None:
    """Render a review-oriented preview: mesh in current world orientation,
    with GIANT green +X (head direction target) and blue +Y (up target)
    reference arrows overlaid.

    The reviewer's job is:
      "rotate the mesh until its head points along the green arrow AND
       the mesh is standing up (dorsal side up along blue arrow)".

    Args:
      mesh_path: path to .glb / .obj
      out_png_path: where to write .png
      note: optional annotation text under the title
    """
    mesh_path = Path(mesh_path)
    out_png_path = Path(out_png_path)
    m = _load_mesh(mesh_path)

    bbox_size = m.bounds[1] - m.bounds[0]
    bbox_max_extent = 0.5 * bbox_size.max()

    fig = plt.figure(figsize=(12, 10))

    # 4 views: front (looking down -X), side (looking along -Z), top (looking
    # down -Y), isometric.  All show the same world axes so the user can
    # see how the mesh sits relative to the target head direction.
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    _draw_review_view(ax1, m, elev=15, azim=-70,
                       title="Isometric (should look natural)",
                       bbox_max_extent=bbox_max_extent)
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    _draw_review_view(ax2, m, elev=0, azim=90,
                       title="Side view (head should be RIGHT →)",
                       bbox_max_extent=bbox_max_extent)
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    _draw_review_view(ax3, m, elev=90, azim=-90,
                       title="Top-down (head should point RIGHT →)",
                       bbox_max_extent=bbox_max_extent)
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    _draw_review_view(ax4, m, elev=0, azim=0,
                       title="Front view (head coming AT you)",
                       bbox_max_extent=bbox_max_extent)

    title = f"[{mesh_path.name}]  Rotate until: head points → (GREEN) & animal stands up ↑ (BLUE)"
    if note:
        title += f"\n{note}"
    fig.suptitle(title, fontsize=11, y=0.995)

    out_png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png_path), dpi=80, bbox_inches="tight")
    plt.close(fig)
