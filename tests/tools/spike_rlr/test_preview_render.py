"""Tests for tools/spike_rlr/preview_render.py."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "spike_rlr"))
sys.path.insert(0, str(REPO / "tests" / "tools" / "spike_rlr"))

from preview_render import render_direction_preview  # noqa: E402
from detect_head_axis import detect_head_axis  # noqa: E402


def _write_synth_glb(tmp_path, head_axis="+X"):
    """Write a tiny synthesized dog-mesh GLB to disk for testing."""
    import trimesh
    from test_detect_head_axis import _synth_dog

    verts = _synth_dog(head_axis=head_axis)
    # Build a simple convex-hull surface for visualization
    hull = trimesh.convex.convex_hull(verts)
    out = tmp_path / f"synth_{head_axis.replace('+', 'p').replace('-', 'm')}.glb"
    hull.export(str(out))
    return out


def _load_verts(glb_path):
    import trimesh
    m = trimesh.load(str(glb_path))
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate(list(m.geometry.values()))
    return np.array(m.vertices)


def test_preview_png_written(tmp_path):
    glb_path = _write_synth_glb(tmp_path, head_axis="+X")
    verts = _load_verts(glb_path)
    result = detect_head_axis(verts)
    out_png = tmp_path / "preview.png"
    render_direction_preview(glb_path, result, out_png)
    assert out_png.exists()
    assert out_png.stat().st_size > 5000  # not an empty PNG


def test_preview_png_readable_as_image(tmp_path):
    glb_path = _write_synth_glb(tmp_path, head_axis="+X")
    verts = _load_verts(glb_path)
    result = detect_head_axis(verts)
    out_png = tmp_path / "preview.png"
    render_direction_preview(glb_path, result, out_png)
    # PNG signature
    with out_png.open("rb") as f:
        header = f.read(8)
    assert header == b"\x89PNG\r\n\x1a\n", f"not a valid PNG: {header!r}"


def test_preview_handles_lowconfidence(tmp_path):
    """Preview must render even when detection is low-confidence."""
    # A single point is a degenerate mesh, but shouldn't crash preview
    rng = np.random.default_rng(seed=1)
    verts = rng.uniform(-0.5, 0.5, (300, 3))
    # skip actual mesh save — just pass an in-memory pseudo path
    import trimesh
    hull = trimesh.convex.convex_hull(verts)
    glb_path = tmp_path / "amorphous.glb"
    hull.export(str(glb_path))
    result = detect_head_axis(verts)
    assert result.confidence < 0.6  # sanity: this mesh IS low confidence
    out_png = tmp_path / "amorphous_preview.png"
    render_direction_preview(glb_path, result, out_png)
    assert out_png.exists()


def test_review_preview_written(tmp_path):
    """render_review_preview produces PNG regardless of head axis (no detection needed)."""
    from preview_render import render_review_preview
    glb_path = _write_synth_glb(tmp_path, head_axis="+X")
    out_png = tmp_path / "review.png"
    render_review_preview(glb_path, out_png, note="test note")
    assert out_png.exists()
    assert out_png.stat().st_size > 5000
    with out_png.open("rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"
