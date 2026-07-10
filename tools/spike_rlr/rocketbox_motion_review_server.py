"""Flask UI for reviewing the two Rocketbox retarget motion packages."""

from __future__ import annotations

import argparse
import getpass
import hmac
import json
import secrets
from pathlib import Path

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_file,
    session,
    url_for,
)

from rocketbox_motion_review import (
    EXPECTED_ASSET_IDS,
    REQUIRED_MEDIA,
    MotionReviewNotApproved,
    assert_pair_approved,
    ensure_pending_review,
    record_decision,
    validate_ready_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[2]

MEDIA_TABS = (
    ("front", "正面 Front"),
    ("side", "侧面 Side"),
    ("top", "俯视 Top"),
    ("joints", "关节 Joints"),
    ("feet", "脚部 Feet"),
    ("source_target", "源骨架+人物 Source + Target"),
    ("contact_sheet", "接触表 Contact"),
)


PAGE_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rocketbox 动作审核 / Motion Review</title>
  <style>
    :root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f5f7f8; color: #17212b; }
    a { color: inherit; }
    .shell { max-width: 1200px; margin: 0 auto; padding: 16px; }
    .masthead { display: flex; justify-content: space-between; gap: 16px; align-items: baseline; border-bottom: 1px solid #cbd3d9; padding-bottom: 12px; }
    h1 { margin: 0; font-size: 20px; font-weight: 700; letter-spacing: 0; }
    .gate { font-size: 13px; color: #43515c; text-align: right; }
    .gate strong { color: {% if gate_state == 'approved' %}#17643c{% else %}#8a3d12{% endif %}; }
    .layout { display: grid; grid-template-columns: 204px minmax(0, 1fr); gap: 20px; padding-top: 16px; }
    .rail { border-right: 1px solid #cbd3d9; padding-right: 12px; }
    .rail h2, .panel-title { margin: 0 0 9px; font-size: 12px; color: #5b6873; font-weight: 700; letter-spacing: 0; text-transform: uppercase; }
    .rail-list { display: grid; gap: 7px; }
    .rail-item { display: block; text-decoration: none; border: 1px solid #c7d0d7; border-radius: 6px; background: #fff; padding: 9px; }
    .rail-item[aria-current="page"] { border-color: #176f8a; box-shadow: inset 3px 0 #176f8a; }
    .rail-name { display: block; font-size: 13px; font-weight: 650; overflow-wrap: anywhere; }
    .state { display: block; margin-top: 4px; color: #52616d; font-size: 12px; text-transform: capitalize; }
    .main { min-width: 0; }
    .asset-head { display: flex; gap: 12px; justify-content: space-between; align-items: baseline; margin-bottom: 12px; }
    .asset-head h2 { margin: 0; font-size: 18px; overflow-wrap: anywhere; }
    .decision { font-size: 13px; color: #43515c; white-space: nowrap; }
    .decision strong { color: #17643c; text-transform: capitalize; }
    .tabs { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 9px; }
    .tab, .command { border: 1px solid #aebbc4; border-radius: 6px; background: #fff; color: #24313b; min-height: 34px; padding: 6px 10px; font: inherit; font-size: 13px; cursor: pointer; }
    .tab[aria-selected="true"] { background: #d9edf3; border-color: #176f8a; }
    .stage { width: 100%; aspect-ratio: 16 / 9; background: #11181e; display: grid; place-items: center; overflow: hidden; }
    .stage video, .stage img { width: 100%; height: 100%; object-fit: contain; }
    .stage img[hidden], .stage video[hidden] { display: none; }
    .review-form { display: grid; grid-template-columns: minmax(0, 1fr) 190px auto auto; gap: 8px; align-items: end; padding-top: 12px; border-top: 1px solid #cbd3d9; margin-top: 12px; }
    label { display: grid; gap: 5px; color: #52616d; font-size: 12px; }
    input, textarea { width: 100%; border: 1px solid #aebbc4; border-radius: 6px; background: #fff; color: #17212b; font: inherit; font-size: 13px; padding: 7px; }
    textarea { min-height: 62px; resize: vertical; }
    .command { font-weight: 700; align-self: stretch; min-width: 82px; }
    .approve { border-color: #17643c; background: #e5f3eb; color: #124a2d; }
    .reject { border-color: #9a3b34; background: #f9e8e6; color: #7a2b25; }
    details { margin-top: 13px; border-top: 1px solid #cbd3d9; padding-top: 10px; color: #52616d; font-size: 12px; }
    summary { cursor: pointer; }
    pre { margin: 8px 0 0; max-height: 220px; overflow: auto; padding: 8px; border-radius: 6px; background: #edf1f3; color: #24313b; white-space: pre-wrap; overflow-wrap: anywhere; }
    @media (max-width: 720px) {
      .shell { padding: 12px; }
      .masthead { align-items: flex-start; flex-direction: column; gap: 5px; }
      .gate { text-align: left; }
      .layout { grid-template-columns: 1fr; gap: 12px; }
      .rail { border-right: 0; border-bottom: 1px solid #cbd3d9; padding: 0 0 12px; }
      .rail-list { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .review-form { grid-template-columns: 1fr 1fr; }
      .review-form label:first-child { grid-column: 1 / -1; }
      .command { min-height: 40px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="masthead">
      <h1>Rocketbox 动作审核 / Motion Review</h1>
      <div class="gate">配对闸门 Pair gate: <strong>{{ gate_state }}</strong>{% if gate_reason %}<br>{{ gate_reason }}{% endif %}</div>
    </header>
    <div class="layout">
      <nav class="rail" aria-label="角色状态 Asset status">
        <h2>角色状态 Assets</h2>
        <div class="rail-list">
          {% for asset in assets %}
          <a class="rail-item" href="{{ url_for('asset_view', asset_id=asset.asset_id) }}" {% if asset.asset_id == asset_id %}aria-current="page"{% endif %}>
            <span class="rail-name">{{ asset.label }}</span>
            <span class="state">{{ asset.decision }}</span>
          </a>
          {% endfor %}
        </div>
      </nav>
      <section class="main" aria-labelledby="asset-title">
        <div class="asset-head">
          <h2 id="asset-title">{{ asset_id }}</h2>
          <div class="decision">当前决定 Current decision: <strong>{{ review.decision }}</strong></div>
        </div>
        <div class="tabs" role="tablist" aria-label="审核视图 Review views">
          {% for kind, label in media_tabs %}
          <button class="tab" type="button" role="tab" data-kind="{{ kind }}" data-url="{{ media_urls[kind] }}" aria-selected="{{ 'true' if kind == 'front' else 'false' }}">{{ label }}</button>
          {% endfor %}
        </div>
        <div class="stage">
          <video id="review-video" src="{{ media_urls['front'] }}" controls loop muted playsinline></video>
          <img id="contact-sheet" src="{{ media_urls['contact_sheet'] }}" alt="Contact sheet" hidden>
        </div>
        <form class="review-form" method="post" action="{{ url_for('decision', asset_id=asset_id) }}">
          <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
          <label>备注 Notes<textarea name="notes" placeholder="观察、帧号或返工原因">{{ review.notes }}</textarea></label>
          <label>审核人 Reviewer<input name="reviewer" value="{{ reviewer }}" placeholder="当前用户"></label>
          <button class="command approve" type="submit" name="decision" value="approved">批准 Approve</button>
          <button class="command reject" type="submit" name="decision" value="rejected">驳回 Reject</button>
        </form>
        <details>
          <summary>诊断 Diagnostics</summary>
          <pre>{{ diagnostics }}</pre>
        </details>
      </section>
    </div>
  </main>
  <script>
    const video = document.getElementById("review-video");
    const contactSheet = document.getElementById("contact-sheet");
    document.querySelectorAll(".tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        const isContact = tab.dataset.kind === "contact_sheet";
        document.querySelectorAll(".tab").forEach((button) => button.setAttribute("aria-selected", String(button === tab)));
        video.hidden = isContact;
        contactSheet.hidden = !isContact;
        if (!isContact) {
          video.src = tab.dataset.url;
          video.load();
          video.play().catch(() => {});
        }
      });
    });
  </script>
</body>
</html>"""


def create_app(review_root: Path | str) -> Flask:
    """Create the review app over a root containing exactly the expected assets."""
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=secrets.token_urlsafe(32),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
    )
    root = Path(review_root).absolute()

    def csrf_token() -> str:
        token = session.get("rocketbox_motion_review_csrf")
        if not isinstance(token, str):
            token = secrets.token_urlsafe(32)
            session["rocketbox_motion_review_csrf"] = token
        return token

    def csrf_is_valid() -> bool:
        submitted = request.form.get("csrf_token")
        expected = session.get("rocketbox_motion_review_csrf")
        return (
            isinstance(submitted, str)
            and isinstance(expected, str)
            and hmac.compare_digest(submitted, expected)
        )

    def form_reviewer(review: dict) -> str:
        reviewer = review.get("reviewer")
        if isinstance(reviewer, str) and reviewer.strip():
            return reviewer.strip()
        return getpass.getuser()

    def asset_dir(asset_id: str) -> Path:
        if asset_id not in EXPECTED_ASSET_IDS:
            abort(404)
        return root / asset_id

    def asset_review(asset_id: str) -> tuple[dict, dict[str, Path]]:
        review_dir = asset_dir(asset_id)
        try:
            manifest, media_paths = validate_ready_manifest(review_dir)
        except ValueError as error:
            abort(409, description=str(error))
        if manifest["asset_id"] != asset_id:
            abort(409, description="review directory asset_id does not match the URL asset")
        try:
            review = ensure_pending_review(review_dir)
        except ValueError as error:
            abort(409, description=str(error))
        return review, media_paths

    def gate_status() -> tuple[str, str]:
        try:
            assert_pair_approved(root)
        except (MotionReviewNotApproved, ValueError) as error:
            return "locked", str(error)
        return "approved", ""

    def rail_assets() -> list[dict[str, str]]:
        labels = ("男性 Male", "女性 Female")
        entries = []
        for asset_id, label in zip(EXPECTED_ASSET_IDS, labels):
            try:
                review, _ = asset_review(asset_id)
                decision = review.get("decision", "pending")
            except Exception:
                decision = "unavailable"
            entries.append({"asset_id": asset_id, "label": label, "decision": decision})
        return entries

    @app.get("/")
    def index():
        return redirect(url_for("asset_view", asset_id=EXPECTED_ASSET_IDS[0]))

    @app.get("/asset/<asset_id>")
    def asset_view(asset_id: str):
        review, media_paths = asset_review(asset_id)
        gate_state, gate_reason = gate_status()
        media_urls = {
            kind: url_for("media", asset_id=asset_id, kind=kind)
            for kind in REQUIRED_MEDIA
        }
        diagnostics = json.dumps(
            {
                "asset_id": asset_id,
                "review": review,
                "media": {kind: path.name for kind, path in media_paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
        return render_template_string(
            PAGE_TEMPLATE,
            asset_id=asset_id,
            assets=rail_assets(),
            review=review,
            gate_state=gate_state,
            gate_reason=gate_reason,
            media_tabs=MEDIA_TABS,
            media_urls=media_urls,
            diagnostics=diagnostics,
            csrf_token=csrf_token(),
            reviewer=form_reviewer(review),
        )

    @app.get("/media/<asset_id>/<kind>")
    def media(asset_id: str, kind: str):
        if kind not in REQUIRED_MEDIA:
            abort(404)
        _, media_paths = asset_review(asset_id)
        return send_file(media_paths[kind], conditional=True)

    @app.post("/decision/<asset_id>")
    def decision(asset_id: str):
        review_dir = asset_dir(asset_id)
        if not csrf_is_valid():
            abort(400, description="invalid CSRF token")
        selected = request.form.get("decision", "")
        if selected not in {"approved", "rejected"}:
            abort(400, description="decision must be approved or rejected")
        reviewer = request.form.get("reviewer", "").strip()
        if not reviewer:
            abort(400, description="reviewer must be non-empty")
        try:
            record_decision(review_dir, selected, reviewer, request.form.get("notes", ""))
        except ValueError as error:
            abort(409, description=str(error))
        return redirect(url_for("asset_view", asset_id=asset_id))

    @app.get("/gate")
    def gate():
        state, reason = gate_status()
        return jsonify(state=state, reason=reason)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-root",
        default=str(REPO_ROOT / "tmp" / "rocketbox_motion_review"),
        help="directory containing the two Rocketbox review asset directories",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    create_app(args.review_root).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
