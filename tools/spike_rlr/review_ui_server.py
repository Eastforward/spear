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


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Mesh Direction Review — {{tag}}</title>
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <style>
        html, body { margin: 0; padding: 0; background: #f2f4f7;
                     font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
        .wrap { max-width: 900px; margin: 0 auto; padding: 24px; }
        h1 { text-align: center; color: #222; margin: 0 0 8px; font-size: 22px; }
        .progress { text-align: center; color: #667; margin-bottom: 20px; font-size: 14px; }
        .card { background: #fff; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                 padding: 24px; margin-bottom: 16px; }
        .tag-name { text-align: center; font-size: 24px; font-weight: 700; color: #223;
                     margin-bottom: 4px; }
        .conf { text-align: center; color: #667; font-size: 13px; margin-bottom: 14px; }
        .conf-high { color: #060; }
        .conf-mid  { color: #a60; }
        .conf-low  { color: #c00; }
        .preview-wrap { text-align: center; margin: 12px 0; }
        .preview-wrap img { max-width: 100%; border: 1px solid #ddd;
                             border-radius: 6px; background: #fff; }
        .instructions { background: #fffbe6; border-left: 4px solid #f5c518;
                         padding: 10px 14px; border-radius: 4px;
                         color: #443; font-size: 13px; margin: 10px 0 18px; line-height: 1.5; }
        .actions { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap;
                    margin: 14px 0 6px; }
        button { padding: 10px 16px; font-size: 14px; font-weight: 600;
                  cursor: pointer; border: none; border-radius: 6px;
                  transition: transform 0.05s, box-shadow 0.15s; }
        button:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
        button:active { transform: translateY(1px); }
        .approve   { background: #16a34a; color: white; }
        .reject    { background: #dc2626; color: white; }
        .skip      { background: #64748b; color: white; }
        .rot       { background: #e5e7eb; color: #223; font-weight: 500; padding: 8px 12px;
                      font-size: 13px; }
        .rot:hover { background: #d1d5db; }
        .rot-x { border-left: 3px solid #dc2626; }
        .rot-y { border-left: 3px solid #16a34a; }
        .rot-z { border-left: 3px solid #2563eb; }
        .section-label { text-align: center; font-size: 12px; color: #667;
                          text-transform: uppercase; letter-spacing: 0.06em;
                          margin: 18px 0 8px; font-weight: 600; }
        .reset-form   { display: inline-block; margin-left: 10px; }
        .footer { text-align: center; color: #999; font-size: 11px; margin-top: 20px; }
        form { display: inline; margin: 0; padding: 0; }
    </style>
</head>
<body>
    <div class="wrap">
        <h1>Hunyuan Mesh Direction Review</h1>
        <div class="progress">
            Pending: <b>{{n_pending}}</b> &nbsp;|&nbsp;
            Approved: <b>{{n_approved}}</b> &nbsp;|&nbsp;
            Rejected: <b>{{n_rejected}}</b>
        </div>

        <div class="card">
            <div class="tag-name">{{tag}}</div>
            <div class="conf">
                Auto-detected head: {{head_direction}}
                &nbsp;|&nbsp; Confidence:
                <span class="{{conf_class}}">{{confidence_pct}}%</span>
                {% if rotation_applied %}
                &nbsp;|&nbsp; Applied rotation: {{rotation_applied}}
                {% endif %}
            </div>

            <div class="instructions">
                <b>Target:</b> rotate the mesh so it stands up ↑ (BLUE arrow)
                AND its head points RIGHT → (GREEN arrow). Then click
                <b style="color:#16a34a">Approve</b>.
                <br>Wrong direction? Use the rotate buttons.
                Unusable mesh? <b style="color:#dc2626">Reject</b>.
                Come back later? <b>Skip</b>.
            </div>

            <div class="preview-wrap">
                <img src="/preview/{{tag}}.png?t={{cache_bust}}" alt="preview">
            </div>

            <div class="section-label">Rotate mesh (each click = 90°)</div>
            <div class="actions">
                <form action="/rotate/{{tag}}" method="post">
                    <input type="hidden" name="axis" value="x">
                    <input type="hidden" name="deg" value="90">
                    <button class="rot rot-x" type="submit">↻ Roll +90° (about X)</button>
                </form>
                <form action="/rotate/{{tag}}" method="post">
                    <input type="hidden" name="axis" value="x">
                    <input type="hidden" name="deg" value="-90">
                    <button class="rot rot-x" type="submit">↺ Roll −90° (about X)</button>
                </form>
                <form action="/rotate/{{tag}}" method="post">
                    <input type="hidden" name="axis" value="y">
                    <input type="hidden" name="deg" value="90">
                    <button class="rot rot-y" type="submit">↻ Yaw +90° (about Y = up)</button>
                </form>
                <form action="/rotate/{{tag}}" method="post">
                    <input type="hidden" name="axis" value="y">
                    <input type="hidden" name="deg" value="-90">
                    <button class="rot rot-y" type="submit">↺ Yaw −90° (about Y = up)</button>
                </form>
                <form action="/rotate/{{tag}}" method="post">
                    <input type="hidden" name="axis" value="z">
                    <input type="hidden" name="deg" value="90">
                    <button class="rot rot-z" type="submit">↻ Pitch +90° (about Z)</button>
                </form>
                <form action="/rotate/{{tag}}" method="post">
                    <input type="hidden" name="axis" value="z">
                    <input type="hidden" name="deg" value="-90">
                    <button class="rot rot-z" type="submit">↺ Pitch −90° (about Z)</button>
                </form>
                <form action="/rotate/{{tag}}" method="post">
                    <input type="hidden" name="axis" value="y">
                    <input type="hidden" name="deg" value="180">
                    <button class="rot" type="submit">Flip head ↔ tail (180° yaw)</button>
                </form>
                <form action="/reset/{{tag}}" method="post" class="reset-form">
                    <button class="rot" type="submit">Reset all rotations</button>
                </form>
            </div>

            <div class="section-label">Decision</div>
            <div class="actions">
                <form action="/approve/{{tag}}" method="post">
                    <button class="approve" type="submit">✅ Approve (head points right)</button>
                </form>
                <form action="/reject/{{tag}}" method="post">
                    <input type="hidden" name="reason" value="rejected via UI">
                    <button class="reject" type="submit">❌ Reject (bad mesh)</button>
                </form>
                <form action="/skip/{{tag}}" method="post">
                    <button class="skip" type="submit">⏭ Skip (come back later)</button>
                </form>
            </div>
        </div>

        <div class="footer">
            Server: {{host}}:{{port}} &nbsp;|&nbsp;
            pending dir: <code>{{pending_dir}}</code>
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


def _rotation_json_path(tag_dir: Path) -> Path:
    return tag_dir / "rotation.json"


def _read_rotation(tag_dir: Path) -> np.ndarray:
    """Return the accumulated rotation matrix (3x3). Identity by default."""
    p = _rotation_json_path(tag_dir)
    if not p.exists():
        return np.eye(3)
    j = json.loads(p.read_text())
    return np.array(j["matrix"], dtype=np.float64)


def _write_rotation(tag_dir: Path, R: np.ndarray, history: list):
    p = _rotation_json_path(tag_dir)
    p.write_text(json.dumps({
        "matrix": R.tolist(),
        "history": history,
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }, indent=2))


def _read_rotation_history(tag_dir: Path) -> list:
    p = _rotation_json_path(tag_dir)
    if not p.exists():
        return []
    j = json.loads(p.read_text())
    return j.get("history", [])


def _apply_rotation_and_regen_preview(tag_dir: Path):
    """Regenerate mesh_current.glb from mesh.glb rotated by the stored rotation,
    then re-render preview PNG."""
    src = tag_dir / "mesh.glb"
    if not src.exists():
        src = tag_dir / "mesh.obj"
    R = _read_rotation(tag_dir)
    m = _load_and_concat(src)
    verts_rot = np.asarray(m.vertices) @ R.T
    rotated = trimesh.Trimesh(vertices=verts_rot, faces=m.faces, process=False)
    current_path = tag_dir / "mesh_current.glb"
    rotated.export(str(current_path))
    preview_path = tag_dir / "direction_preview_review.png"
    hist = _read_rotation_history(tag_dir)
    hist_str = " + ".join(hist) if hist else "identity"
    render_review_preview(current_path, preview_path,
                           note=f"Accumulated rotation: {hist_str}")


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
        current_glb = tag_dir / "mesh_current.glb"
        preview_png = tag_dir / "direction_preview_review.png"
        if not current_glb.exists() or not preview_png.exists():
            _apply_rotation_and_regen_preview(tag_dir)

        dj = json.loads((tag_dir / "direction.json").read_text())
        det = dj["detection"]
        head = det["head_direction_original_mesh_frame"]
        conf_pct = int(det["confidence"] * 100)
        history = _read_rotation_history(tag_dir)

        import time
        return render_template_string(
            HTML_TEMPLATE, tag=tag,
            head_direction=f"[{head[0]:+.2f}, {head[1]:+.2f}, {head[2]:+.2f}]",
            confidence_pct=conf_pct,
            conf_class=_classify_conf(conf_pct),
            rotation_applied=" + ".join(history) if history else "",
            n_pending=len(_pending_tags()),
            n_approved=_count(approved_dir), n_rejected=_count(rejected_dir),
            host=host, port=port, pending_dir=str(pending_dir),
            cache_bust=int(time.time() * 1000),
        )

    @app.route("/preview/<tag>.png")
    def preview(tag):
        tag_dir = pending_dir / tag
        p = tag_dir / "direction_preview_review.png"
        # Regenerate on demand if missing (first visit)
        if not p.exists():
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

        # Bake accumulated user rotation on top of the auto-orient step.
        # The direction.json's rotation_applied_to_align_to_plus_x rotates
        # mesh.glb -> mesh_oriented.glb.  We now compose the user rotation
        # against the ORIGINAL mesh (not the mesh_oriented one, since our
        # UI works on original coordinates).
        R_user = _read_rotation(src)
        m = _load_and_concat(src / "mesh.glb"
                              if (src / "mesh.glb").exists() else src / "mesh.obj")
        verts_final = np.asarray(m.vertices) @ R_user.T
        final = trimesh.Trimesh(vertices=verts_final, faces=m.faces, process=False)
        final.export(str(src / "mesh_oriented.glb"))

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
        dj_path.write_text(json.dumps(dj, indent=2))

        dst = dest_dir / tag
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))

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

    app = create_app(args.pending_dir, args.approved_dir, args.rejected_dir,
                      host=args.host, port=args.port)
    print(f"Review UI serving http://{args.host}:{args.port}/")
    print(f"  pending: {args.pending_dir}")
    print(f"  approved: {args.approved_dir}")
    print(f"  rejected: {args.rejected_dir}")
    print("SSH port-forward from your local machine:")
    print(f"  ssh -N -L {args.port}:localhost:{args.port} <this-server>")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
