"""Flask web UI for Hunyuan mesh direction audit — v2.

v2 changes vs v1:
  - Single-card centered layout (one tag at a time, auto-jumps to next
    after each decision, no more scrolling through a long list).
  - Rotation buttons: Rot X/Y/Z ±90° flip the mesh live and regenerate
    the preview. Accumulated rotation is persisted per-tag under
    pending/{tag}/rotation.json.
  - New preview shows a GIANT green "HEAD →" reference arrow along
    world +X plus a blue "UP ↑" arrow along +Y. The reviewer's task is
    to rotate the mesh until it aligns with those arrows.
  - Approve bakes the accumulated rotation into mesh_oriented.glb.

Usage:
  /data/jzy/miniconda3/envs/ss2/bin/python \\
      tools/spike_rlr/review_ui_server.py --port 8080
Then SSH -N -L 8080:localhost:8080 <server> and open http://localhost:8080/

Routes:
  GET  /                     -- redirect to next pending tag (or /done)
  GET  /tag/<tag>            -- single-card review page for one tag
  GET  /preview/<tag>.png    -- current preview PNG (regenerated on rotate)
  POST /rotate/<tag>         -- apply 90° rotation about form.axis (x/y/z)
                                 with form.deg (±90); regenerate preview,
                                 redirect back to /tag/<tag>
  POST /approve/<tag>        -- bake rotation, move to approved/, jump to next
  POST /reject/<tag>         -- move to rejected/, jump to next
  POST /skip/<tag>           -- leave in pending, jump to next
"""
from __future__ import annotations

import argparse
import datetime
import getpass
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh
from flask import Flask, abort, redirect, request, send_file, url_for

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))
from preview_render import render_review_preview  # noqa: E402

PREVIEW_RENDER_SOURCE = REPO_ROOT / "tools" / "spike_rlr" / "preview_render.py"


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Mesh Review — {{tag}}</title>
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <style>
        html, body { margin: 0; padding: 0; background: #f2f4f7;
                     font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
        .wrap { max-width: 820px; margin: 0 auto; padding: 20px; }
        h1 { text-align: center; color: #222; margin: 0 0 6px; font-size: 20px; }
        .progress { text-align: center; color: #667; margin-bottom: 16px; font-size: 13px; }
        .card { background: #fff; border-radius: 12px;
                 box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                 padding: 20px; }
        .tag-name { text-align: center; font-size: 22px; font-weight: 700;
                     color: #223; margin-bottom: 2px; }
        .conf { text-align: center; color: #667; font-size: 12px; margin-bottom: 10px; }
        .conf-high { color: #060; }
        .conf-mid  { color: #a60; }
        .conf-low  { color: #c00; }
        .rot-applied { color: #049; font-family: monospace; font-size: 12px;
                        margin-top: 3px; }
        .instructions { background: #fffbe6; border-left: 4px solid #f5c518;
                         padding: 8px 12px; border-radius: 4px;
                         color: #443; font-size: 12px; margin: 6px 0 12px;
                         line-height: 1.5; text-align: center; }

        /* ---- Spatial rotate cross around preview ---- */
        .stage {
            display: grid;
            grid-template-columns: 70px 1fr 70px;
            grid-template-rows: 60px 1fr 60px;
            align-items: center; justify-items: center;
            gap: 6px;
            max-width: 640px;
            margin: 0 auto;
        }
        .stage .preview { grid-column: 2; grid-row: 2;
                            display: flex; justify-content: center; }
        .stage .preview img { max-width: 100%; max-height: 480px;
                               border: 1px solid #ddd; border-radius: 6px;
                               background: #fff; display: block; }

        .rot-btn { background: #e5e7eb; color: #223; border: none;
                    border-radius: 8px; padding: 6px 10px; font-size: 22px;
                    cursor: pointer; font-weight: 700; line-height: 1;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
                    transition: background 0.1s, transform 0.05s;
                    min-width: 54px; min-height: 42px; }
        .rot-btn:hover { background: #d1d5db; }
        .rot-btn:active { transform: translateY(1px); }
        .rot-btn small { display: block; font-size: 9px; font-weight: 500;
                          color: #667; margin-top: 2px; }

        /* Position within grid */
        .rot-top    { grid-column: 2; grid-row: 1; }
        .rot-bottom { grid-column: 2; grid-row: 3; }
        .rot-left   { grid-column: 1; grid-row: 2; }
        .rot-right  { grid-column: 3; grid-row: 2; }
        .rot-tl     { grid-column: 1; grid-row: 1; }
        .rot-tr     { grid-column: 3; grid-row: 1; }
        .rot-bl     { grid-column: 1; grid-row: 3; }
        .rot-br     { grid-column: 3; grid-row: 3; }

        .btn-row { display: flex; gap: 12px; justify-content: center;
                    margin-top: 18px; flex-wrap: wrap; }
        .btn { padding: 11px 18px; font-size: 14px; font-weight: 600;
                cursor: pointer; border: none; border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .btn:hover { filter: brightness(0.95); }
        .btn:active { transform: translateY(1px); }
        .approve { background: #16a34a; color: white; }
        .reject  { background: #dc2626; color: white; }
        .skip    { background: #64748b; color: white; }
        .flip    { background: #f59e0b; color: white; }
        .reset   { background: #e5e7eb; color: #334; }

        .footer { text-align: center; color: #999; font-size: 11px; margin-top: 16px; }
        form { display: inline; margin: 0; padding: 0; }

        /* Corner buttons (Y-axis yaw = in-plane rotation) — smaller & round */
        .rot-tl button, .rot-tr button, .rot-bl button, .rot-br button {
            font-size: 20px;
            background: #dbeafe;
        }
        .rot-tl button:hover, .rot-tr button:hover,
        .rot-bl button:hover, .rot-br button:hover { background: #bfdbfe; }
    </style>
</head>
<body>
    <div class="wrap">
        <h1>Mesh Direction Review</h1>
        <div class="progress">
            Pending: <b>{{n_pending}}</b> &nbsp;|&nbsp;
            Approved: <b>{{n_approved}}</b> &nbsp;|&nbsp;
            Rejected: <b>{{n_rejected}}</b><br>
            <span style="font-size:11px;">
              📂 <code>{{pending_dir}}</code>
            </span>
        </div>

        <div class="card">
            <div class="tag-name">{{tag}}</div>
            <div class="conf">
                Auto-detected head: {{head_direction}} &nbsp;|&nbsp;
                Confidence: <span class="{{conf_class}}">{{confidence_pct}}%</span>
                {% if rotation_applied %}
                <div class="rot-applied">Applied: {{rotation_applied}}</div>
                {% endif %}
                <div class="rot-applied" style="color:#555; font-size:11px;">
                  📁 <code>{{mesh_abs_path}}</code>
                </div>
            </div>

            <div class="instructions">
                <b>Goal:</b> rotate so head points to the <b style="color:#0a0">green → HEAD</b> arrow
                on the right. Then click <b style="color:#16a34a">Approve</b>.
            </div>

            <!-- SPATIAL ROTATE CROSS: arrows around the image -->
            <div class="stage">

                <!-- Top-left corner: yaw CCW (in-plane, about Y) -->
                <div class="rot-tl">
                    <form action="/rotate/{{tag}}" method="post">
                        <input type="hidden" name="axis" value="y">
                        <input type="hidden" name="deg" value="-90">
                        <button type="submit" class="rot-btn" title="Rotate mesh counter-clockwise in this view (yaw −90°)">↺<small>yaw −90°</small></button>
                    </form>
                </div>

                <!-- Top center: tilt back-away (roll about X, +90) -->
                <div class="rot-top">
                    <form action="/rotate/{{tag}}" method="post">
                        <input type="hidden" name="axis" value="x">
                        <input type="hidden" name="deg" value="90">
                        <button type="submit" class="rot-btn" title="Tilt back — flip the mesh away from viewer (roll +90°)">↑<small>roll +90°</small></button>
                    </form>
                </div>

                <!-- Top-right corner: yaw CW -->
                <div class="rot-tr">
                    <form action="/rotate/{{tag}}" method="post">
                        <input type="hidden" name="axis" value="y">
                        <input type="hidden" name="deg" value="90">
                        <button type="submit" class="rot-btn" title="Rotate mesh clockwise in this view (yaw +90°)">↻<small>yaw +90°</small></button>
                    </form>
                </div>

                <!-- Left: pitch −90 (topple to left in this view) -->
                <div class="rot-left">
                    <form action="/rotate/{{tag}}" method="post">
                        <input type="hidden" name="axis" value="z">
                        <input type="hidden" name="deg" value="-90">
                        <button type="submit" class="rot-btn" title="Pitch −90° (about Z axis)">←<small>pitch −</small></button>
                    </form>
                </div>

                <!-- Center: preview image -->
                <div class="preview">
                    <img src="/preview/{{tag}}.png?t={{cache_bust}}" alt="preview">
                </div>

                <!-- Right: pitch +90 -->
                <div class="rot-right">
                    <form action="/rotate/{{tag}}" method="post">
                        <input type="hidden" name="axis" value="z">
                        <input type="hidden" name="deg" value="90">
                        <button type="submit" class="rot-btn" title="Pitch +90° (about Z axis)">→<small>pitch +</small></button>
                    </form>
                </div>

                <!-- Bottom-left corner: flip head↔tail (180 yaw) -->
                <div class="rot-bl">
                    <form action="/rotate/{{tag}}" method="post">
                        <input type="hidden" name="axis" value="y">
                        <input type="hidden" name="deg" value="180">
                        <button type="submit" class="rot-btn" title="180° yaw — swap head and tail" style="background:#fef3c7;">⇄<small>flip 180°</small></button>
                    </form>
                </div>

                <!-- Bottom center: tilt forward-toward (roll about X, −90) -->
                <div class="rot-bottom">
                    <form action="/rotate/{{tag}}" method="post">
                        <input type="hidden" name="axis" value="x">
                        <input type="hidden" name="deg" value="-90">
                        <button type="submit" class="rot-btn" title="Tilt forward — flip the mesh toward viewer (roll −90°)">↓<small>roll −90°</small></button>
                    </form>
                </div>

                <!-- Bottom-right corner: reset -->
                <div class="rot-br">
                    <form action="/reset/{{tag}}" method="post">
                        <button type="submit" class="rot-btn" title="Reset all rotations" style="background:#fee2e2;">⟲<small>reset</small></button>
                    </form>
                </div>

            </div>

            <div class="btn-row">
                <form action="/approve/{{tag}}" method="post">
                    <button class="btn approve" type="submit">✅ Approve</button>
                </form>
                <form action="/reject/{{tag}}" method="post">
                    <input type="hidden" name="reason" value="rejected via UI">
                    <button class="btn reject" type="submit">❌ Reject</button>
                </form>
                <form action="/skip/{{tag}}" method="post">
                    <button class="btn skip" type="submit">⏭ Skip</button>
                </form>
            </div>
        </div>

        <div class="footer">
            {{host}}:{{port}} &nbsp;|&nbsp; <code>{{pending_dir}}</code>
        </div>
    </div>
</body>
</html>
"""


DONE_TEMPLATE = """<!DOCTYPE html>
<html><head>
    <title>All done — Mesh Direction Review</title>
    <meta http-equiv="refresh" content="10">
    <style>
        body { margin: 0; padding: 60px 24px; text-align: center;
                font-family: -apple-system, sans-serif; background: #f2f4f7; color: #223; }
        h1 { color: #16a34a; }
        .stats { color: #667; margin: 20px 0; }
        code { background: #e5e7eb; padding: 2px 6px; border-radius: 3px; }
    </style>
</head><body>
    <h1>🎉 All pending meshes reviewed</h1>
    <div class="stats">
        Approved: <b>{{n_approved}}</b> &nbsp;|&nbsp;
        Rejected: <b>{{n_rejected}}</b>
    </div>
    <p>Auto-refreshes every 10 s. To review new meshes, drop them into
        <code>{{pending_dir}}</code> and run
        <code>auto_orient_ingest.py</code>.
    </p>
</body></html>
"""


def _axis_rotation_matrix(axis: str, deg: float) -> np.ndarray:
    """Return 3x3 rotation matrix for a rotation about world axis by deg."""
    rad = np.deg2rad(deg)
    c, s = np.cos(rad), np.sin(rad)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    raise ValueError(f"unknown axis {axis!r}")


def _load_and_concat(mesh_path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(str(mesh_path))
    if isinstance(scene, trimesh.Scene):
        geoms = list(scene.geometry.values())
        return trimesh.util.concatenate(geoms)
    return scene


def _rotated_mesh_preserving_visuals(mesh: trimesh.Trimesh, R: np.ndarray):
    rotated = mesh.copy()
    rotated.vertices = np.asarray(mesh.vertices) @ R.T
    return rotated


def _rotation_json_path(tag_dir: Path) -> Path:
    return tag_dir / "rotation.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_rotation(tag_dir: Path) -> np.ndarray:
    """Return the accumulated rotation matrix (3x3). Identity if file
    missing, empty, corrupted, or mid-write (see _write_rotation atomicity
    note)."""
    return _read_rotation_safe(tag_dir)


def _write_rotation(tag_dir: Path, R: np.ndarray, history: list):
    """Atomic write: write to a temp sibling then rename over the target.
    Prevents readers from seeing a truncated / empty file mid-write when a
    reload races the rotate button."""
    p = _rotation_json_path(tag_dir)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "matrix": R.tolist(),
        "history": history,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }, indent=2))
    tmp.replace(p)


def _read_rotation_safe(tag_dir: Path) -> np.ndarray:
    """Read with fallback: empty / corrupted file -> identity."""
    p = _rotation_json_path(tag_dir)
    if not p.exists():
        return np.eye(3)
    try:
        raw = p.read_text()
        if not raw.strip():
            return np.eye(3)
        j = json.loads(raw)
        return np.array(j["matrix"], dtype=np.float64)
    except (json.JSONDecodeError, KeyError, ValueError):
        return np.eye(3)


def _read_rotation_history(tag_dir: Path) -> list:
    p = _rotation_json_path(tag_dir)
    if not p.exists():
        return []
    try:
        raw = p.read_text()
        if not raw.strip():
            return []
        return json.loads(raw).get("history", [])
    except (json.JSONDecodeError, ValueError):
        return []


def _apply_rotation_and_regen_preview(tag_dir: Path):
    """Regenerate mesh_current.glb from mesh.glb rotated by the stored rotation,
    then re-render preview PNG."""
    src = tag_dir / "mesh.glb"
    if not src.exists():
        src = tag_dir / "mesh.obj"
    R = _read_rotation(tag_dir)
    m = _load_and_concat(src)
    rotated = _rotated_mesh_preserving_visuals(m, R)
    current_path = tag_dir / "mesh_current.glb"
    rotated.export(str(current_path))
    preview_path = tag_dir / "direction_preview_review.png"
    hist = _read_rotation_history(tag_dir)
    hist_str = " + ".join(hist) if hist else "identity"
    render_review_preview(current_path, preview_path,
                           note=f"Accumulated rotation: {hist_str}")


def _review_preview_stale(tag_dir: Path) -> bool:
    current_glb = tag_dir / "mesh_current.glb"
    preview_png = tag_dir / "direction_preview_review.png"
    if not current_glb.exists() or not preview_png.exists():
        return True
    if PREVIEW_RENDER_SOURCE.exists():
        return preview_png.stat().st_mtime < PREVIEW_RENDER_SOURCE.stat().st_mtime
    return False


def _classify_conf(pct: float) -> str:
    if pct >= 70:
        return "conf-high"
    if pct >= 40:
        return "conf-mid"
    return "conf-low"


def create_app(pending_dir, approved_dir, rejected_dir, host="127.0.0.1", port=8080):
    from flask import render_template_string
    app = Flask(__name__)
    pending_dir = Path(pending_dir)
    approved_dir = Path(approved_dir)
    rejected_dir = Path(rejected_dir)
    for d in (pending_dir, approved_dir, rejected_dir):
        d.mkdir(parents=True, exist_ok=True)

    def _pending_tags():
        tags = []
        for tag_dir in sorted(pending_dir.iterdir()):
            if not tag_dir.is_dir() or tag_dir.name.startswith("."):
                continue
            if (tag_dir / "direction.json").exists():
                tags.append(tag_dir.name)
        return tags

    def _count(d):
        return sum(1 for x in d.iterdir() if x.is_dir() and not x.name.startswith("."))

    @app.route("/")
    def index():
        tags = _pending_tags()
        if not tags:
            return render_template_string(
                DONE_TEMPLATE,
                n_approved=_count(approved_dir), n_rejected=_count(rejected_dir),
                pending_dir=str(pending_dir),
            )
        return redirect(url_for("tag_view", tag=tags[0]))

    @app.route("/tag/<tag>")
    def tag_view(tag):
        tag_dir = pending_dir / tag
        if not tag_dir.exists() or not (tag_dir / "direction.json").exists():
            return redirect(url_for("index"))

        # If we haven't yet generated the review preview for this tag, do so
        # (also picks up any current-rotation state).
        if _review_preview_stale(tag_dir):
            _apply_rotation_and_regen_preview(tag_dir)

        dj = json.loads((tag_dir / "direction.json").read_text())
        det = dj["detection"]
        head = det["head_direction_original_mesh_frame"]
        conf_pct = int(det["confidence"] * 100)
        history = _read_rotation_history(tag_dir)

        # Resolve absolute mesh path so the user can see exactly which file
        # we're reviewing (feature request 2026-07-08).
        mesh_source = tag_dir / "mesh.glb"
        if not mesh_source.exists():
            mesh_source = tag_dir / "mesh.obj"
        mesh_abs_path = str(mesh_source.resolve())
        print(f"[review_ui] GET /tag/{tag}  mesh={mesh_abs_path}", flush=True)

        import time
        return render_template_string(
            HTML_TEMPLATE, tag=tag,
            head_direction=f"[{head[0]:+.2f}, {head[1]:+.2f}, {head[2]:+.2f}]",
            confidence_pct=conf_pct,
            conf_class=_classify_conf(conf_pct),
            rotation_applied=" + ".join(history) if history else "",
            mesh_abs_path=mesh_abs_path,
            n_pending=len(_pending_tags()),
            n_approved=_count(approved_dir), n_rejected=_count(rejected_dir),
            host=host, port=port,
            pending_dir=str(pending_dir.resolve()),
            cache_bust=int(time.time() * 1000),
        )

    @app.route("/preview/<tag>.png")
    def preview(tag):
        tag_dir = pending_dir / tag
        p = tag_dir / "direction_preview_review.png"
        # Regenerate on demand if missing (first visit)
        if _review_preview_stale(tag_dir):
            _apply_rotation_and_regen_preview(tag_dir)
        if not p.exists():
            abort(404)
        return send_file(str(p), mimetype="image/png")

    @app.route("/rotate/<tag>", methods=["POST"])
    def rotate(tag):
        axis = request.form.get("axis", "y").lower()
        try:
            deg = float(request.form.get("deg", "90"))
        except ValueError:
            abort(400, "deg must be a number")
        if axis not in ("x", "y", "z"):
            abort(400, "axis must be x, y, or z")
        tag_dir = pending_dir / tag
        if not tag_dir.exists():
            abort(404)
        # Compose new rotation
        R_prev = _read_rotation(tag_dir)
        R_new = _axis_rotation_matrix(axis, deg) @ R_prev
        history = _read_rotation_history(tag_dir)
        history.append(f"{axis}{'+' if deg >= 0 else ''}{int(deg)}")
        _write_rotation(tag_dir, R_new, history)
        print(f"[review_ui] ROTATE {tag}  axis={axis} deg={deg:+.0f}  "
              f"history={' + '.join(history)}", flush=True)
        _apply_rotation_and_regen_preview(tag_dir)
        return redirect(url_for("tag_view", tag=tag))

    @app.route("/reset/<tag>", methods=["POST"])
    def reset(tag):
        tag_dir = pending_dir / tag
        if not tag_dir.exists():
            abort(404)
        _write_rotation(tag_dir, np.eye(3), [])
        _apply_rotation_and_regen_preview(tag_dir)
        return redirect(url_for("tag_view", tag=tag))

    def _bake_and_move(tag, dest_dir, updates):
        src = pending_dir / tag
        if not src.exists():
            abort(404, f"tag {tag} not in pending")
        dst = dest_dir / tag
        src_mesh_name = "mesh.glb" if (src / "mesh.glb").exists() else "mesh.obj"

        # Bake accumulated user rotation on top of the auto-orient step.
        # The direction.json's rotation_applied_to_align_to_plus_x rotates
        # mesh.glb -> mesh_oriented.glb.  We now compose the user rotation
        # against the ORIGINAL mesh (not the mesh_oriented one, since our
        # UI works on original coordinates).
        R_user = _read_rotation(src)
        m = _load_and_concat(src / src_mesh_name)
        final = _rotated_mesh_preserving_visuals(m, R_user)
        oriented_src = src / "mesh_oriented.glb"
        final.export(str(oriented_src))

        dj_path = src / "direction.json"
        dj = json.loads(dj_path.read_text())
        dj.update(updates)
        dj["human_approved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            dj["human_approved_by"] = getpass.getuser()
        except Exception:
            dj["human_approved_by"] = "unknown"
        # Record final user-applied rotation for audit trail
        dj["human_applied_rotation_matrix"] = R_user.tolist()
        dj["human_applied_rotation_history"] = _read_rotation_history(src)
        dj["mesh_source"] = str((dst / src_mesh_name).resolve())
        dj["mesh_oriented"] = str((dst / "mesh_oriented.glb").resolve())
        dj["mesh_sha256"] = _sha256_file(oriented_src)
        dj_path.write_text(json.dumps(dj, indent=2))

        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        print(f"[review_ui] MOVED {tag}  ->  {dst.resolve()}", flush=True)
        print(f"[review_ui]   direction.json.human_approved = "
              f"{updates.get('human_approved')}", flush=True)
        if updates.get("human_notes"):
            print(f"[review_ui]   notes: {updates['human_notes']}", flush=True)

    @app.route("/approve/<tag>", methods=["POST"])
    def approve(tag):
        _bake_and_move(tag, approved_dir, {"human_approved": True})
        return redirect(url_for("index"))

    @app.route("/reject/<tag>", methods=["POST"])
    def reject(tag):
        reason = request.form.get("reason", "rejected via UI")
        _bake_and_move(tag, rejected_dir,
                        {"human_approved": False, "human_notes": reason})
        return redirect(url_for("index"))

    @app.route("/skip/<tag>", methods=["POST"])
    def skip(tag):
        # Leave in pending; move on
        tags = _pending_tags()
        try:
            idx = tags.index(tag)
        except ValueError:
            return redirect(url_for("index"))
        next_tag = tags[(idx + 1) % len(tags)] if tags else None
        if next_tag and next_tag != tag:
            return redirect(url_for("tag_view", tag=next_tag))
        return redirect(url_for("index"))

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pending-dir",  default=str(REPO_ROOT / "tmp/hy3d_batch/pending"))
    ap.add_argument("--approved-dir", default=str(REPO_ROOT / "tmp/hy3d_batch/approved"))
    ap.add_argument("--rejected-dir", default=str(REPO_ROOT / "tmp/hy3d_batch/rejected"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1",
                     help="Bind host (default 127.0.0.1; SSH-forward from local)")
    args = ap.parse_args()

    pending_abs = Path(args.pending_dir).resolve()
    approved_abs = Path(args.approved_dir).resolve()
    rejected_abs = Path(args.rejected_dir).resolve()

    app = create_app(pending_abs, approved_abs, rejected_abs,
                      host=args.host, port=args.port)
    print("=" * 72, flush=True)
    print(f"Review UI serving  http://{args.host}:{args.port}/", flush=True)
    print(f"  pending  dir:  {pending_abs}", flush=True)
    print(f"  approved dir:  {approved_abs}", flush=True)
    print(f"  rejected dir:  {rejected_abs}", flush=True)
    # Enumerate pending tags + their mesh files
    if pending_abs.exists():
        pending_tags = sorted(d for d in pending_abs.iterdir()
                               if d.is_dir() and (d / "direction.json").exists())
        print(f"  {len(pending_tags)} tag(s) awaiting review:", flush=True)
        for td in pending_tags:
            mesh = td / "mesh.glb"
            if not mesh.exists():
                mesh = td / "mesh.obj"
            print(f"     - {td.name:<28s}  {mesh.resolve()}", flush=True)
    print("SSH port-forward from your local machine:", flush=True)
    print(f"  ssh -N -L {args.port}:localhost:{args.port} <this-server>", flush=True)
    print("=" * 72, flush=True)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
