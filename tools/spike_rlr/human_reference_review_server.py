"""Flask UI for reviewing paired FLUX human reference images."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import re
import secrets
from io import BytesIO
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

from human_reference_review import (
    EXPECTED_ASSET_IDS,
    HumanReferenceNotApproved,
    assert_pair_approved,
    read_review_state,
    record_review,
    validated_candidate_snapshot,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE_KINDS = ("source", "candidate")
SNAPSHOT_FIELDS = (
    "candidate_manifest_sha256",
    "source_sha256",
    "candidate_sha256",
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>Human Reference Review</title>
  <style>
    :root { color-scheme: light; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #f5f7f8; color: #17212b; }
    a { color: inherit; }
    .shell { max-width: 1240px; margin: 0 auto; padding: 16px; }
    .masthead { display: flex; align-items: baseline; justify-content: space-between; gap: 16px; border-bottom: 1px solid #cbd3d9; padding-bottom: 12px; }
    h1 { margin: 0; font-size: 20px; font-weight: 700; letter-spacing: 0; }
    .gate { color: #43515c; font-size: 13px; text-align: right; }
    .gate strong { color: {% if gate_state == 'approved' %}#17643c{% else %}#8a3d12{% endif %}; text-transform: capitalize; }
    .layout { display: grid; grid-template-columns: 204px minmax(0, 1fr); gap: 20px; padding-top: 16px; }
    .rail { border-right: 1px solid #cbd3d9; padding-right: 12px; }
    .rail h2, .section-label { margin: 0 0 9px; color: #5b6873; font-size: 12px; font-weight: 700; letter-spacing: 0; text-transform: uppercase; }
    .rail-list { display: grid; gap: 7px; }
    .rail-item { display: block; border: 1px solid #c7d0d7; border-radius: 6px; background: #fff; padding: 9px; text-decoration: none; }
    .rail-item[aria-current="page"] { border-color: #176f8a; box-shadow: inset 3px 0 #176f8a; }
    .rail-name { display: block; font-size: 13px; font-weight: 650; overflow-wrap: anywhere; }
    .state { display: block; margin-top: 4px; color: #52616d; font-size: 12px; text-transform: capitalize; }
    .main { min-width: 0; }
    .asset-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 12px; }
    .asset-head h2 { margin: 0; font-size: 18px; overflow-wrap: anywhere; }
    .decision { color: #43515c; font-size: 13px; white-space: nowrap; }
    .decision strong { color: #17643c; text-transform: capitalize; }
    .comparison { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    figure { min-width: 0; margin: 0; }
    figcaption { margin-bottom: 6px; color: #43515c; font-size: 13px; font-weight: 700; }
    .image-stage { display: grid; min-height: 360px; place-items: center; overflow: hidden; background: #dde3e6; }
    .image-stage img { display: block; width: 100%; height: 100%; max-height: 72vh; object-fit: contain; }
    .prompt { margin-top: 14px; border-top: 1px solid #cbd3d9; padding-top: 11px; }
    .prompt p { margin: 0; color: #24313b; font-size: 14px; line-height: 1.45; overflow-wrap: anywhere; white-space: pre-wrap; }
    .review-form { display: grid; grid-template-columns: minmax(0, 1fr) 190px auto auto; gap: 8px; align-items: end; margin-top: 14px; border-top: 1px solid #cbd3d9; padding-top: 12px; }
    label { display: grid; gap: 5px; color: #52616d; font-size: 12px; }
    input, textarea { width: 100%; border: 1px solid #aebbc4; border-radius: 6px; background: #fff; color: #17212b; font: inherit; font-size: 13px; padding: 7px; }
    textarea { min-height: 62px; resize: vertical; }
    .command { min-width: 82px; min-height: 36px; align-self: stretch; border: 1px solid #aebbc4; border-radius: 6px; background: #fff; color: #24313b; cursor: pointer; font: inherit; font-size: 13px; font-weight: 700; }
    .approve { border-color: #17643c; background: #e5f3eb; color: #124a2d; }
    .reject { border-color: #9a3b34; background: #f9e8e6; color: #7a2b25; }
    @media (max-width: 720px) {
      .shell { padding: 12px; }
      .masthead { align-items: flex-start; flex-direction: column; gap: 5px; }
      .gate { text-align: left; }
      .layout { grid-template-columns: 1fr; gap: 12px; }
      .rail { border-right: 0; border-bottom: 1px solid #cbd3d9; padding: 0 0 12px; }
      .rail-list { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .comparison { grid-template-columns: 1fr; }
      .image-stage { min-height: 280px; }
      .review-form { grid-template-columns: 1fr 1fr; }
      .review-form label:first-child { grid-column: 1 / -1; }
      .command { min-height: 40px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="masthead">
      <h1>Human Reference Review</h1>
      <div class="gate">Pair gate: <strong>{{ gate_state }}</strong>{% if gate_reason %}<br>{{ gate_reason }}{% endif %}</div>
    </header>
    <div class="layout">
      <nav class="rail" aria-label="Asset status">
        <h2>Assets</h2>
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
          <div class="decision">Current decision: <strong>{{ review.decision }}</strong></div>
        </div>
        <div class="comparison">
          <figure>
            <figcaption>Source image</figcaption>
            <div class="image-stage"><img src="{{ media_urls.source }}" alt="Source image"></div>
          </figure>
          <figure>
            <figcaption>Candidate image</figcaption>
            <div class="image-stage"><img src="{{ media_urls.candidate }}" alt="Candidate image"></div>
          </figure>
        </div>
        <section class="prompt" aria-labelledby="prompt-label">
          <h3 class="section-label" id="prompt-label">Prompt</h3>
          <p>{{ manifest.prompt }}</p>
        </section>
        <form class="review-form" method="post" action="{{ url_for('review', asset_id=asset_id) }}">
          <input type="hidden" name="csrf_token" value="{{ csrf_token }}">
          <input type="hidden" name="candidate_manifest_sha256" value="{{ snapshot.candidate_manifest_sha256 }}">
          <input type="hidden" name="source_sha256" value="{{ snapshot.source_sha256 }}">
          <input type="hidden" name="candidate_sha256" value="{{ snapshot.candidate_sha256 }}">
          <label>Notes<textarea name="notes" placeholder="Observations or regeneration reason">{{ review.notes }}</textarea></label>
          <label>Reviewer<input name="reviewer" value="{{ reviewer }}" placeholder="Current user"></label>
          <button class="command approve" type="submit" name="decision" value="approved">Approve</button>
          <button class="command reject" type="submit" name="decision" value="rejected">Reject</button>
        </form>
      </section>
    </div>
  </main>
</body>
</html>"""


def create_app(review_root: Path | str) -> Flask:
    """Create the review app over the two expected human reference directories."""
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=secrets.token_urlsafe(32),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
    )
    root = Path(review_root).absolute()

    def csrf_token() -> str:
        token = session.get("human_reference_review_csrf")
        if not isinstance(token, str):
            token = secrets.token_urlsafe(32)
            session["human_reference_review_csrf"] = token
        return token

    def csrf_is_valid() -> bool:
        submitted = request.form.get("csrf_token")
        expected = session.get("human_reference_review_csrf")
        return (
            isinstance(submitted, str)
            and isinstance(expected, str)
            and hmac.compare_digest(submitted, expected)
        )

    def asset_dir(asset_id: str) -> Path:
        if asset_id not in EXPECTED_ASSET_IDS:
            abort(404)
        return root / asset_id

    def asset_review(
        asset_id: str,
    ) -> tuple[dict, dict, dict[str, Path], dict[str, str]]:
        candidate_dir = asset_dir(asset_id)
        try:
            manifest, images, snapshot = validated_candidate_snapshot(candidate_dir)
            review = read_review_state(candidate_dir)
        except ValueError as error:
            abort(409, description=str(error))
        if manifest["asset_id"] != asset_id:
            abort(409, description="candidate directory asset_id does not match the URL asset")
        return manifest, review, images, snapshot

    def gate_status() -> tuple[str, str]:
        try:
            assert_pair_approved(root)
        except (HumanReferenceNotApproved, ValueError) as error:
            return "locked", str(error)
        return "approved", ""

    def form_reviewer(review: dict) -> str:
        reviewer = review.get("reviewer")
        if isinstance(reviewer, str) and reviewer.strip():
            return reviewer.strip()
        return getpass.getuser()

    def rail_assets() -> list[dict[str, str]]:
        labels = ("Male", "Female")
        entries = []
        for asset_id, label in zip(EXPECTED_ASSET_IDS, labels):
            try:
                _, review, _, _ = asset_review(asset_id)
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
        manifest, review, _, snapshot = asset_review(asset_id)
        gate_state, gate_reason = gate_status()
        media_urls = {
            kind: url_for(
                "media",
                asset_id=asset_id,
                kind=kind,
                expected_sha256=snapshot[f"{kind}_sha256"],
            )
            for kind in IMAGE_KINDS
        }
        return render_template_string(
            PAGE_TEMPLATE,
            asset_id=asset_id,
            assets=rail_assets(),
            manifest=manifest,
            review=review,
            gate_state=gate_state,
            gate_reason=gate_reason,
            media_urls=media_urls,
            csrf_token=csrf_token(),
            reviewer=form_reviewer(review),
            snapshot=snapshot,
        )

    @app.get("/media/<asset_id>/<kind>")
    def media(asset_id: str, kind: str):
        if kind not in IMAGE_KINDS:
            abort(404)
        asset_dir(asset_id)
        expected_sha256 = request.args.get("expected_sha256", "")
        if _SHA256_RE.fullmatch(expected_sha256) is None:
            abort(400, description="expected media hash must be 64-character lowercase hex")
        _, _, images, snapshot = asset_review(asset_id)
        if not hmac.compare_digest(
            expected_sha256, snapshot[f"{kind}_sha256"]
        ):
            abort(409, description="candidate snapshot changed; reload before reviewing")
        try:
            media_bytes = images[kind].read_bytes()
        except OSError as error:
            abort(409, description=f"could not read expected {kind} image: {error}")
        if not hmac.compare_digest(
            expected_sha256, hashlib.sha256(media_bytes).hexdigest()
        ):
            abort(409, description="candidate image changed while it was being read")
        response = send_file(
            BytesIO(media_bytes),
            download_name=f"{kind}.png",
            mimetype="image/png",
            conditional=True,
            max_age=0,
        )
        response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    @app.post("/review/<asset_id>")
    def review(asset_id: str):
        candidate_dir = asset_dir(asset_id)
        if not csrf_is_valid():
            abort(400, description="invalid CSRF token")
        decision = request.form.get("decision", "")
        if decision not in {"approved", "rejected"}:
            abort(400, description="decision must be approved or rejected")
        reviewer = request.form.get("reviewer", "").strip()
        if not reviewer:
            abort(400, description="reviewer must be non-empty")
        submitted_snapshot = {
            field: request.form.get(field, "") for field in SNAPSHOT_FIELDS
        }
        if any(
            _SHA256_RE.fullmatch(value) is None
            for value in submitted_snapshot.values()
        ):
            abort(400, description="snapshot hashes must be 64-character lowercase hex")
        try:
            _, _, _, snapshot = asset_review(asset_id)
            if any(
                not hmac.compare_digest(submitted_snapshot[field], snapshot[field])
                for field in SNAPSHOT_FIELDS
            ):
                abort(409, description="candidate snapshot changed; reload before reviewing")
            record_review(
                candidate_dir,
                decision,
                reviewer,
                request.form.get("notes", ""),
                expected_snapshot=submitted_snapshot,
            )
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
        default=str(REPO_ROOT / "tmp" / "human_reference_review"),
        help="directory containing the two human reference candidate directories",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    create_app(args.review_root).run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
