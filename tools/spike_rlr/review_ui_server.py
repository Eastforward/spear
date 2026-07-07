"""Flask web UI for Hunyuan mesh direction audit.

Usage (headless server):
  /data/jzy/miniconda3/envs/ss2/bin/python \\
      tools/spike_rlr/review_ui_server.py \\
      --port 8080

Then locally:
  ssh -L 8080:localhost:8080 <server>
  open http://localhost:8080/

Routes:
  GET  /                 -- list pending tags with previews
  GET  /preview/<tag>.png -- serve preview PNG
  POST /approve/<tag>    -- mv pending/{tag}/ -> approved/{tag}/, set human_approved=True
  POST /reject/<tag>     -- mv pending/{tag}/ -> rejected/{tag}/, keep human_approved=False
  POST /override/<tag>   -- mv pending -> rejected + record human_override so
                             re-ingest can use the correct head direction
"""
from __future__ import annotations

import argparse
import datetime
import getpass
import json
import shutil
from pathlib import Path

from flask import Flask, abort, redirect, request, send_file, url_for


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Hunyuan Mesh Direction Review</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        h1 { border-bottom: 2px solid #333; padding-bottom: 8px; }
        .tag-card { border: 1px solid #ccc; padding: 15px; margin: 15px 0;
                     border-radius: 8px; background: #fafafa; }
        .tag-title { font-size: 18px; font-weight: bold; }
        .confidence-high { color: #060; }
        .confidence-low  { color: #c60; }
        .meta { color: #666; font-size: 13px; margin: 5px 0; }
        img { max-width: 600px; border: 1px solid #999; }
        button { margin-right: 8px; padding: 8px 16px; font-size: 14px;
                  cursor: pointer; border: none; border-radius: 4px; }
        .approve { background: #0a0; color: white; }
        .reject  { background: #a00; color: white; }
        .override { background: #a60; color: white; }
        form { display: inline; }
        .stats { background: #eef; padding: 10px; border-radius: 4px; margin: 10px 0; }
    </style>
</head>
<body>
    <h1>Hunyuan Mesh Direction Review</h1>
    <div class="stats">
      Pending: {{n_pending}} | Approved: {{n_approved}} | Rejected: {{n_rejected}}
    </div>
    {% if tags %}
    {% for tag in tags %}
    <div class="tag-card">
        <div class="tag-title">{{tag.name}}</div>
        <div class="meta">
            Detected head direction: {{tag.head_direction}} |
            Confidence: <span class="{% if tag.confidence >= 0.7 %}confidence-high{% else %}confidence-low{% endif %}">{{tag.confidence_pct}}%</span> |
            Unanimous: {{tag.unanimous}} |
            Votes: {{tag.total_votes}}
        </div>
        <div class="meta">Signals: {{tag.signals_str}}</div>
        <img src="/preview/{{tag.name}}.png" alt="preview">
        <div>
            <form action="/approve/{{tag.name}}" method="post">
              <button type="submit" class="approve">Approve (head is at red arrow)</button>
            </form>
            <form action="/reject/{{tag.name}}" method="post">
              <input type="hidden" name="reason" value="rejected via UI">
              <button type="submit" class="reject">Reject (bad mesh)</button>
            </form>
            <form action="/override/{{tag.name}}" method="post">
              <input type="hidden" name="correct_direction_x" value="{{ -tag.raw_head[0] }}">
              <input type="hidden" name="correct_direction_y" value="{{ -tag.raw_head[1] }}">
              <input type="hidden" name="correct_direction_z" value="{{ -tag.raw_head[2] }}">
              <input type="hidden" name="reason" value="head is at OPPOSITE end">
              <button type="submit" class="override">Head is at opposite end</button>
            </form>
        </div>
    </div>
    {% endfor %}
    {% else %}
    <p><em>No pending tags. All caught up!</em></p>
    {% endif %}
</body>
</html>
"""


def create_app(pending_dir, approved_dir, rejected_dir):
    from flask import render_template_string
    app = Flask(__name__)
    pending_dir = Path(pending_dir)
    approved_dir = Path(approved_dir)
    rejected_dir = Path(rejected_dir)
    for d in (pending_dir, approved_dir, rejected_dir):
        d.mkdir(parents=True, exist_ok=True)

    def _load_pending_tags():
        result = []
        for tag_dir in sorted(pending_dir.iterdir()):
            if not tag_dir.is_dir() or tag_dir.name.startswith("."):
                continue
            dj_path = tag_dir / "direction.json"
            if not dj_path.exists():
                continue
            dj = json.loads(dj_path.read_text())
            det = dj["detection"]
            head = det["head_direction_original_mesh_frame"]
            result.append({
                "name": tag_dir.name,
                "head_direction": f"[{head[0]:+.2f}, {head[1]:+.2f}, {head[2]:+.2f}]",
                "raw_head": head,
                "confidence": det["confidence"],
                "confidence_pct": int(det["confidence"] * 100),
                "unanimous": det["unanimous"],
                "total_votes": det["total_votes"],
                "signals_str": ", ".join(f"{k}={v:+d}" for k, v in det["signals"].items()),
            })
        return result

    @app.route("/")
    def index():
        tags = _load_pending_tags()
        n_approved = sum(1 for d in approved_dir.iterdir()
                          if d.is_dir() and not d.name.startswith("."))
        n_rejected = sum(1 for d in rejected_dir.iterdir()
                          if d.is_dir() and not d.name.startswith("."))
        return render_template_string(HTML_TEMPLATE, tags=tags,
                                       n_pending=len(tags), n_approved=n_approved,
                                       n_rejected=n_rejected)

    @app.route("/preview/<tag>.png")
    def preview(tag):
        p = pending_dir / tag / "direction_preview.png"
        if not p.exists():
            abort(404)
        return send_file(str(p), mimetype="image/png")

    def _move_tag(tag, dest_dir, updates):
        src = pending_dir / tag
        if not src.exists():
            abort(404, f"tag {tag} not in pending")
        dj_path = src / "direction.json"
        dj = json.loads(dj_path.read_text())
        dj.update(updates)
        dj["human_approved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        try:
            dj["human_approved_by"] = getpass.getuser()
        except Exception:
            dj["human_approved_by"] = "unknown"
        dj_path.write_text(json.dumps(dj, indent=2))
        dst = dest_dir / tag
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))

    @app.route("/approve/<tag>", methods=["POST"])
    def approve(tag):
        _move_tag(tag, approved_dir, {"human_approved": True})
        return redirect(url_for("index"))

    @app.route("/reject/<tag>", methods=["POST"])
    def reject(tag):
        reason = request.form.get("reason", "rejected via UI")
        _move_tag(tag, rejected_dir,
                   {"human_approved": False, "human_notes": reason})
        return redirect(url_for("index"))

    @app.route("/override/<tag>", methods=["POST"])
    def override(tag):
        try:
            cx = float(request.form.get("correct_direction_x", "0"))
            cy = float(request.form.get("correct_direction_y", "0"))
            cz = float(request.form.get("correct_direction_z", "0"))
        except ValueError:
            abort(400, "invalid override direction vector")
        reason = request.form.get("reason", "human override")
        _move_tag(tag, rejected_dir, {
            "human_approved": False,
            "human_notes": reason,
            "human_override": {
                "correct_head_direction_in_original_mesh": [cx, cy, cz],
                "reason": reason,
            },
        })
        return redirect(url_for("index"))

    return app


def main():
    REPO = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser()
    ap.add_argument("--pending-dir", default=str(REPO / "tmp/hy3d_batch/pending"))
    ap.add_argument("--approved-dir", default=str(REPO / "tmp/hy3d_batch/approved"))
    ap.add_argument("--rejected-dir", default=str(REPO / "tmp/hy3d_batch/rejected"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1",
                     help="Bind host (default 127.0.0.1; SSH-forward from local)")
    args = ap.parse_args()

    app = create_app(args.pending_dir, args.approved_dir, args.rejected_dir)
    print(f"Review UI serving http://{args.host}:{args.port}/")
    print(f"  pending: {args.pending_dir}")
    print(f"  approved: {args.approved_dir}")
    print(f"  rejected: {args.rejected_dir}")
    print("SSH port-forward from your local machine:")
    print(f"  ssh -L {args.port}:localhost:{args.port} <this-server>")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
