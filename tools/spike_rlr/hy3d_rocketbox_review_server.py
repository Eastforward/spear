"""Flask UI for hash-locked Hunyuan/Rocketbox Walk/Idle review."""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import re
import secrets
from io import BytesIO
from pathlib import Path

from flask import Flask, abort, jsonify, redirect, render_template_string, request, send_file, session, url_for

from hy3d_rocketbox_review import (
    EXPECTED_ASSET_IDS,
    REQUIRED_MOTIONS,
    REQUIRED_VIEWS,
    SNAPSHOT_FIELDS,
    Hy3DRocketboxNotApproved,
    assert_pair_approved,
    assert_snapshot_current,
    read_review_state_for_snapshot,
    record_decision,
    validated_review_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

PAGE_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><link rel="icon" href="data:,">
<title>Walk / Idle Review</title><style>
:root{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#17212b;background:#f5f7f8}*{box-sizing:border-box}body{margin:0}.shell{max-width:1280px;margin:auto;padding:16px}.masthead,.asset-head,.identity{display:flex;justify-content:space-between;gap:16px;align-items:center;border-bottom:1px solid #cbd3d9;padding-bottom:12px}h1{font-size:20px;margin:0}h2{font-size:18px;margin:0}.gate,.decision{font-size:13px;color:#43515c}.gate strong,.decision strong{color:#17643c;text-transform:capitalize}.layout{display:grid;grid-template-columns:190px minmax(0,1fr);gap:20px;padding-top:16px}.rail{border-right:1px solid #cbd3d9;padding-right:12px}.rail-list{display:grid;gap:7px}.rail-item{display:block;padding:9px;border:1px solid #c7d0d7;border-radius:6px;background:#fff;text-decoration:none;color:inherit;font-size:13px;font-weight:650}.rail-item[aria-current="page"]{border-color:#176f8a;box-shadow:inset 3px 0 #176f8a}.state{display:block;margin-top:4px;color:#52616d;font-size:12px;text-transform:capitalize}.main{min-width:0}.asset-head{margin-bottom:12px}.identity{justify-content:flex-start;border:0;padding:0;margin:0 0 12px;color:#43515c;font-size:13px;font-weight:650}.identity img{width:42px;height:42px;object-fit:cover;border:1px solid #aebbc4;border-radius:4px}.tabs{display:flex;gap:7px;border-bottom:1px solid #cbd3d9;margin-bottom:12px}.tab,.command{min-height:36px;border:1px solid #aebbc4;border-radius:6px;background:#fff;color:#24313b;cursor:pointer;font:inherit;font-size:13px;font-weight:700;padding:7px 11px}.tab[aria-selected="true"]{border-color:#176f8a;background:#e5f1f4;color:#124a5c}.video-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.video-grid figure{min-width:0;margin:0}.video-grid figcaption{margin-bottom:5px;color:#43515c;font-size:13px;font-weight:700}.video-grid video{display:block;width:100%;aspect-ratio:16/9;background:#dde3e6}.review-form{display:grid;grid-template-columns:minmax(0,1fr) 180px auto auto;gap:8px;align-items:end;margin-top:14px;border-top:1px solid #cbd3d9;padding-top:12px}label{display:grid;gap:5px;color:#52616d;font-size:12px}input,textarea{width:100%;border:1px solid #aebbc4;border-radius:6px;background:#fff;color:#17212b;font:inherit;font-size:13px;padding:7px}textarea{min-height:62px;resize:vertical}.approve{border-color:#17643c;background:#e5f3eb;color:#124a2d}.reject{border-color:#9a3b34;background:#f9e8e6;color:#7a2b25}@media(max-width:720px){.shell{padding:12px}.masthead{align-items:flex-start;flex-direction:column;gap:5px}.layout{grid-template-columns:1fr;gap:12px}.rail{border-right:0;border-bottom:1px solid #cbd3d9;padding:0 0 12px}.rail-list{grid-template-columns:repeat(2,minmax(0,1fr))}.video-grid{grid-template-columns:1fr}.review-form{grid-template-columns:1fr 1fr}.review-form label:first-child{grid-column:1/-1}.command{min-height:40px}}</style></head>
<body><main class="shell"><header class="masthead"><h1>Walk / Idle Review</h1><div class="gate">Pair gate: <strong>{{ gate_state }}</strong>{% if gate_reason %}<br>{{ gate_reason }}{% endif %}</div></header>
<div class="layout"><nav class="rail" aria-label="Asset status"><div class="rail-list">{% for asset in assets %}<a class="rail-item" href="{{ url_for('asset_view', asset_id=asset.asset_id) }}" {% if asset.asset_id == asset_id %}aria-current="page"{% endif %}>{{ asset.label }}<span class="state">{{ asset.decision }}</span></a>{% endfor %}</div></nav>
<section class="main"><div class="asset-head"><h2>{{ asset_id }}</h2><div class="decision">Current decision: <strong>{{ review.decision }}</strong></div></div><div class="identity"><img src="{{ reference_data_url }}" alt="Approved FLUX reference"><span>Approved FLUX reference</span></div>
<div class="tabs" role="tablist" aria-label="Motion"><button class="tab" type="button" data-motion="walk" aria-selected="true">Walk</button><button class="tab" type="button" data-motion="idle" aria-selected="false">Idle</button></div><div class="video-grid">{% for view in views %}<figure><figcaption>{{ view|capitalize }}</figcaption><video data-view="{{ view }}" src="{{ media_urls.walk[view] }}" controls loop muted playsinline></video></figure>{% endfor %}</div>
<form class="review-form" method="post" action="{{ url_for('decision', asset_id=asset_id) }}"><input type="hidden" name="csrf_token" value="{{ csrf_token }}">{% for field in snapshot_fields %}<input type="hidden" name="{{ field }}" value="{{ snapshot[field] }}">{% endfor %}<label>Notes<textarea name="notes">{{ review.notes }}</textarea></label><label>Reviewer<input name="reviewer" value="{{ reviewer }}"></label><button class="command approve" type="submit" name="decision" value="approved">Approve</button><button class="command reject" type="submit" name="decision" value="rejected">Reject</button></form></section></div></main>
<script>const urls={{ media_urls|tojson }};document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>{const motion=tab.dataset.motion;document.querySelectorAll('.tab').forEach(button=>button.setAttribute('aria-selected',String(button===tab)));document.querySelectorAll('video[data-view]').forEach(video=>{video.src=urls[motion][video.dataset.view];video.load();video.play().catch(()=>{});});}));</script></body></html>"""


def create_app(review_root: Path | str) -> Flask:
    """Create a review app over exactly the two expected asset directories."""
    app = Flask(__name__)
    app.config.update(SECRET_KEY=secrets.token_urlsafe(32), SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict")
    root = Path(review_root).absolute()

    def csrf_token() -> str:
        token = session.get("hy3d_rocketbox_review_csrf")
        if not isinstance(token, str):
            token = secrets.token_urlsafe(32)
            session["hy3d_rocketbox_review_csrf"] = token
        return token

    def csrf_is_valid() -> bool:
        submitted = request.form.get("csrf_token")
        expected = session.get("hy3d_rocketbox_review_csrf")
        return isinstance(submitted, str) and isinstance(expected, str) and hmac.compare_digest(submitted, expected)

    def asset_dir(asset_id: str) -> Path:
        if asset_id not in EXPECTED_ASSET_IDS:
            abort(404)
        return root / asset_id

    def asset_review(asset_id: str):
        directory = asset_dir(asset_id)
        try:
            bind_manifest, _, captured, snapshot = validated_review_snapshot(directory)
            if bind_manifest["asset_id"] != asset_id:
                abort(409, description="review directory asset_id does not match the URL asset")
            review = read_review_state_for_snapshot(
                directory, bind_manifest, snapshot
            )
            assert_snapshot_current(directory, snapshot)
        except ValueError as error:
            abort(409, description=str(error))
        return review, captured, snapshot

    def gate_status() -> tuple[str, str]:
        try:
            assert_pair_approved(root)
        except (Hy3DRocketboxNotApproved, ValueError) as error:
            return "locked", str(error)
        return "approved", ""

    def rail_assets(
        selected_asset_id: str | None = None,
        selected_review: dict | None = None,
    ) -> list[dict[str, str]]:
        labels = ("Male", "Female")
        entries = []
        for asset_id, label in zip(EXPECTED_ASSET_IDS, labels):
            if asset_id == selected_asset_id and selected_review is not None:
                decision = selected_review.get("decision", "pending")
            else:
                try:
                    decision = asset_review(asset_id)[0].get("decision", "pending")
                except Exception:
                    decision = "unavailable"
            entries.append({"asset_id": asset_id, "label": label, "decision": decision})
        return entries

    @app.get("/")
    def index():
        return redirect(url_for("asset_view", asset_id=EXPECTED_ASSET_IDS[0]))

    @app.get("/asset/<asset_id>")
    def asset_view(asset_id: str):
        review, captured, snapshot = asset_review(asset_id)
        media_urls = {
            motion: {
                view: url_for("media", asset_id=asset_id, motion=motion, view=view, expected_sha256=snapshot[f"{motion}_{view}_sha256"])
                for view in REQUIRED_VIEWS
            }
            for motion in REQUIRED_MOTIONS
        }
        reference_bytes = captured["reference"]
        if not hmac.compare_digest(
            snapshot["reference_sha256"],
            hashlib.sha256(reference_bytes).hexdigest(),
        ):
            abort(409, description="captured reference hash does not match snapshot")
        reference_data_url = "data:image/png;base64," + base64.b64encode(
            reference_bytes
        ).decode("ascii")
        gate_state, gate_reason = gate_status()
        reviewer = review.get("reviewer") if isinstance(review.get("reviewer"), str) else ""
        return render_template_string(PAGE_TEMPLATE, asset_id=asset_id, assets=rail_assets(asset_id, review), review=review, gate_state=gate_state, gate_reason=gate_reason, media_urls=media_urls, views=REQUIRED_VIEWS, snapshot=snapshot, snapshot_fields=SNAPSHOT_FIELDS, csrf_token=csrf_token(), reviewer=reviewer.strip() or getpass.getuser(), reference_data_url=reference_data_url)

    @app.get("/media/<asset_id>/<motion>/<view>")
    def media(asset_id: str, motion: str, view: str):
        if motion not in REQUIRED_MOTIONS or view not in REQUIRED_VIEWS:
            abort(404)
        asset_dir(asset_id)
        expected = request.args.get("expected_sha256", "")
        if _SHA256_RE.fullmatch(expected) is None:
            abort(400, description="expected_sha256 must be a 64-character lowercase hex value")
        _, captured, snapshot = asset_review(asset_id)
        if not hmac.compare_digest(expected, snapshot[f"{motion}_{view}_sha256"]):
            abort(409, description="review snapshot changed; reload before reviewing")
        media_bytes = captured[f"{motion}_{view}"]
        if not hmac.compare_digest(
            expected, hashlib.sha256(media_bytes).hexdigest()
        ):
            abort(409, description="captured media hash does not match expected SHA")
        response = send_file(
            BytesIO(media_bytes),
            download_name=f"{motion}_{view}.mp4",
            mimetype="video/mp4",
            conditional=True,
            max_age=0,
        )
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @app.post("/decision/<asset_id>")
    def decision(asset_id: str):
        directory = asset_dir(asset_id)
        if not csrf_is_valid():
            abort(400, description="invalid CSRF token")
        selected = request.form.get("decision", "")
        if selected not in {"approved", "rejected"}:
            abort(400, description="decision must be approved or rejected")
        reviewer = request.form.get("reviewer", "").strip()
        if not reviewer:
            abort(400, description="reviewer must be non-empty")
        submitted = {field: request.form.get(field, "") for field in SNAPSHOT_FIELDS}
        if any(_SHA256_RE.fullmatch(value) is None for value in submitted.values()):
            abort(400, description="snapshot hashes must be 64-character lowercase hex")
        try:
            _, _, current = asset_review(asset_id)
            if any(not hmac.compare_digest(submitted[field], current[field]) for field in SNAPSHOT_FIELDS):
                abort(409, description="review snapshot changed; reload before reviewing")
            record_decision(directory, selected, reviewer, request.form.get("notes", ""), expected_snapshot=submitted)
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
    parser.add_argument("--review-root", default=str(REPO_ROOT / "tmp" / "hy3d_rocketbox_review"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8093)
    args = parser.parse_args()
    create_app(args.review_root).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
