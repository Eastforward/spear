#!/usr/bin/env python3
"""Build a self-contained local HTML review page for a generated animal.

One page per asset workspace, three panels:

1. Forward decision: draggable turntable of the raw TokenRig mesh, the
   deterministic estimator's axis/candidates/votes, and one click on the
   correct head-end candidate.  The page live-renders the head-end decision
   JSON (download button) plus the exact CLI command that turns it into the
   authoritative forward declaration.
2. Gait and preview: the fail-closed gait-direction verdict and the
   ``--preview-only`` walking renders.
3. Final animation review: the six-view videos with the six human boolean
   checks; approving requires all six, mirroring the CLI decision tool.

Everything is embedded or referenced by relative path, so the page works
from the local filesystem with no server and no external dependencies.  The
page never mutates pipeline state: it only produces decision JSON for the
existing fail-closed tools.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "avengine_forward_review_page_v1"
CHECK_FIELDS = (
    "walking_direction",
    "walking_limb_deformation",
    "walking_ground_contact",
    "idle_ground_contact",
    "body_stability",
    "detached_geometry_absent",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-workspace", required=True)
    parser.add_argument("--turntable-dir", type=Path, required=True)
    parser.add_argument("--estimate-json", type=Path, required=True)
    parser.add_argument("--gait-audit", type=Path)
    parser.add_argument(
        "--preview-video",
        type=Path,
        action="append",
        default=[],
        help="Preview MP4s (repeatable), e.g. walking_side.mp4",
    )
    parser.add_argument(
        "--review-video",
        type=Path,
        action="append",
        default=[],
        help="Final six-view MP4s (repeatable)",
    )
    parser.add_argument("--output-html", type=Path, required=True)
    return parser.parse_args(argv)


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative_to_page(path: Path, page: Path) -> str:
    return os.path.relpath(Path(path).resolve(), page.resolve().parent)


def main(argv=None):
    args = parse_args(argv)
    output = args.output_html.resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace review page: {output}")

    turntable = load_json(args.turntable_dir / "turntable_manifest.json")
    estimate_run = load_json(args.estimate_json)
    estimate = estimate_run["estimate"]
    gait = load_json(args.gait_audit) if args.gait_audit else None

    frames = [
        relative_to_page(args.turntable_dir / frame["path"], output)
        for frame in turntable["frames"]
    ]
    candidates = [
        {
            "yaw": float(candidate["candidate_yaw_deg"]),
            "src": relative_to_page(args.turntable_dir / candidate["path"], output),
        }
        for candidate in turntable["candidates"]
    ]
    previews = [
        {"label": video.stem, "src": relative_to_page(video, output)}
        for video in args.preview_video
    ]
    reviews = [
        {"label": video.stem, "src": relative_to_page(video, output)}
        for video in args.review_video
    ]

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "asset_workspace": args.asset_workspace,
        "input_glb": estimate_run.get("input"),
        "estimate": estimate,
        "gait": gait,
        "frames": frames,
        "candidates": candidates,
        "previews": previews,
        "reviews": reviews,
        "check_fields": list(CHECK_FIELDS),
    }

    title = html.escape(f"Forward review — {args.asset_workspace}")
    page = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.5 system-ui, sans-serif; background: #16161a;
       color: #e6e6ea; }
main { max-width: 1200px; margin: 0 auto; padding: 24px; }
h1 { font-size: 20px; } h2 { font-size: 16px; margin-top: 0; }
section { background: #1f1f26; border: 1px solid #2c2c36; border-radius: 10px;
          padding: 18px; margin-bottom: 20px; }
.row { display: flex; gap: 18px; flex-wrap: wrap; }
.col { flex: 1 1 320px; min-width: 300px; }
img.turntable { width: 100%; border-radius: 8px; background: #000;
                cursor: ew-resize; user-select: none; }
input[type=range] { width: 100%; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
td, th { border-bottom: 1px solid #2c2c36; padding: 4px 8px; text-align: left; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px;
         font-size: 12px; font-weight: 600; }
.badge.pass { background: #143d2b; color: #5be49b; }
.badge.fail { background: #46181c; color: #ff8089; }
.badge.warn { background: #423512; color: #ffd166; }
.cand { flex: 1 1 260px; border: 2px solid #2c2c36; border-radius: 10px;
        padding: 10px; cursor: pointer; text-align: center; }
.cand img { width: 100%; border-radius: 6px; }
.cand.selected { border-color: #5be49b; background: #143d2b22; }
textarea, pre.cli { width: 100%; background: #131318; color: #cde3d3;
  border: 1px solid #2c2c36; border-radius: 8px; padding: 10px;
  font: 12px/1.5 ui-monospace, monospace; white-space: pre-wrap; }
textarea { min-height: 180px; }
button, a.btn { background: #2b5f46; color: #eafff3; border: 0;
  border-radius: 8px; padding: 8px 16px; font-weight: 600; cursor: pointer;
  text-decoration: none; display: inline-block; }
button:disabled { background: #333; color: #777; cursor: not-allowed; }
video { width: 100%; border-radius: 8px; background: #000; }
label.check { display: block; padding: 4px 0; }
.muted { color: #9a9aa5; font-size: 12px; }
</style>
</head>
<body>
<main>
<h1>__TITLE__</h1>
<p class="muted">此页面只产出决策 JSON，喂给 fail-closed 的 CLI 工具；它本身不修改任何管线状态。</p>

<section id="panel-forward">
  <h2>① 前向判定 — 哪端是头？</h2>
  <div class="row">
    <div class="col">
      <img id="turntable" class="turntable" alt="turntable">
      <input id="orbit" type="range" min="0" value="0">
      <p class="muted">拖动滑条或在图上左右拖拽旋转模型。</p>
    </div>
    <div class="col">
      <table id="estimate-table"></table>
      <div class="row" id="candidates"></div>
      <p><button id="dl-forward" disabled>下载头端决策 JSON</button></p>
      <pre class="cli" id="cli-forward">（选择头端候选后生成 CLI 命令）</pre>
    </div>
  </div>
</section>

<section id="panel-gait">
  <h2>② 步态审计与预览</h2>
  <div id="gait-summary"></div>
  <div class="row" id="previews"></div>
</section>

<section id="panel-final">
  <h2>③ 六视图终审</h2>
  <div class="row" id="reviews"></div>
  <div class="row">
    <div class="col">
      <div id="checks"></div>
      <p>
        <label><input type="radio" name="verdict" value="approved_for_ue_apartment"> 接受</label>
        <label><input type="radio" name="verdict" value="rejected"> 拒绝</label>
      </p>
      <p><button id="dl-final" disabled>下载终审决策 JSON</button></p>
      <p class="muted">接受要求全部六项勾选；拒绝要求至少一项未勾选（与 CLI 决策工具同规则）。</p>
    </div>
    <div class="col"><textarea id="final-json" readonly></textarea></div>
  </div>
</section>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);

// -- turntable --
const img = document.getElementById("turntable");
const orbit = document.getElementById("orbit");
orbit.max = D.frames.length - 1;
function show(i) { img.src = D.frames[((i % D.frames.length) + D.frames.length) % D.frames.length]; }
orbit.addEventListener("input", () => show(+orbit.value));
let dragging = null;
img.addEventListener("pointerdown", e => { dragging = { x: e.clientX, f: +orbit.value }; img.setPointerCapture(e.pointerId); });
img.addEventListener("pointermove", e => {
  if (!dragging) return;
  const delta = Math.round((e.clientX - dragging.x)/12);
  const f = ((dragging.f + delta) % D.frames.length + D.frames.length) % D.frames.length;
  orbit.value = f; show(f);
});
img.addEventListener("pointerup", () => dragging = null);
show(0);

// -- estimate table --
const est = D.estimate;
const votes = est.head_end_vote.signals;
document.getElementById("estimate-table").innerHTML = `
  <tr><th>PCA 无符号轴</th><td>${est.unsigned_axis_yaw_deg.toFixed(2)}°</td></tr>
  <tr><th>估计前向</th><td>${est.estimated_front_yaw_deg.toFixed(2)}°</td></tr>
  <tr><th>头尾置信度</th><td>${est.head_end_vote.confidence.toFixed(2)}
      ${est.head_end_vote.confidence < 0.5 ? '<span class="badge warn">低置信度，需人工裁决</span>' : ''}</td></tr>
  <tr><th>信号投票</th><td>${Object.entries(votes).map(([k, v]) => `${k}=${v}`).join("&nbsp; ")}</td></tr>`;

// -- candidates --
let chosen = null;
const candBox = document.getElementById("candidates");
D.candidates.forEach(c => {
  const div = document.createElement("div");
  div.className = "cand";
  div.innerHTML = `<img src="${c.src}"><div>前向 = ${c.yaw.toFixed(2)}°<br>
    <span class="muted">此视角正对该候选端；若看到脸，这端就是头</span></div>`;
  div.addEventListener("click", () => {
    chosen = c;
    document.querySelectorAll(".cand").forEach(el => el.classList.remove("selected"));
    div.classList.add("selected");
    renderForward();
  });
  candBox.appendChild(div);
});

function download(name, obj) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(obj, null, 2) + "\\n"],
    { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

function renderForward() {
  const btn = document.getElementById("dl-forward");
  btn.disabled = !chosen;
  if (!chosen) return;
  const decision = {
    schema: "avengine_head_end_review_decision_v1",
    asset_workspace: D.asset_workspace,
    input_glb: D.input_glb,
    reviewed_by: "project_owner",
    confirmed_front_yaw_deg: chosen.yaw,
    estimator_confidence: est.head_end_vote.confidence,
    decided_from: "forward_review_page_turntable_and_candidates",
  };
  btn.onclick = () => download("head_end_review.json", decision);
  document.getElementById("cli-forward").textContent =
    "python tools/build_generated_animal_forward_declaration.py \\\\\\n" +
    `  --asset-workspace ${D.asset_workspace} \\\\\\n` +
    `  --input-glb ${D.input_glb ? D.input_glb.path : "<rig.glb>"} \\\\\\n` +
    "  --estimate-json <estimate_run.json> \\\\\\n" +
    `  --confirmed-front-yaw-deg ${chosen.yaw.toFixed(6)} \\\\\\n` +
    "  --head-end-evidence head_end_review.json \\\\\\n" +
    "  --head-end-decision-source human_confirming_estimator \\\\\\n" +
    "  --output forward_declaration.json";
}

// -- gait --
const gaitBox = document.getElementById("gait-summary");
if (D.gait) {
  const r = D.gait.result;
  const ok = D.gait.status === "pass";
  gaitBox.innerHTML = `<p><span class="badge ${ok ? "pass" : "fail"}">${D.gait.status}</span>
    分类 <b>${r.classification}</b>，站立足漂移方位 ${r.stance_drift_yaw_deg.toFixed(1)}°，
    漂移量/对角线 ${r.stance_drift.mean_drift_ratio_of_diagonal.toExponential(2)}</p>`;
} else {
  gaitBox.innerHTML = '<p class="muted">尚无步态审计结果。</p>';
}
const prevBox = document.getElementById("previews");
D.previews.forEach(v => {
  const div = document.createElement("div");
  div.className = "col";
  div.innerHTML = `<video src="${v.src}" controls loop muted></video>
    <p class="muted">${v.label}</p>`;
  prevBox.appendChild(div);
});

// -- final review --
const revBox = document.getElementById("reviews");
D.reviews.forEach(v => {
  const div = document.createElement("div");
  div.className = "col";
  div.innerHTML = `<video src="${v.src}" controls loop muted></video>
    <p class="muted">${v.label}</p>`;
  revBox.appendChild(div);
});
const checksBox = document.getElementById("checks");
D.check_fields.forEach(field => {
  const label = document.createElement("label");
  label.className = "check";
  label.innerHTML = `<input type="checkbox" data-field="${field}"> ${field}`;
  checksBox.appendChild(label);
});
function renderFinal() {
  const checks = {};
  document.querySelectorAll("#checks input").forEach(el => {
    checks[el.dataset.field] = el.checked;
  });
  const verdictEl = document.querySelector('input[name="verdict"]:checked');
  const allPass = Object.values(checks).every(Boolean);
  const decision = {
    schema: "avengine_controlled_animal_animation_decision_input_v1",
    asset_id: D.asset_workspace,
    decision: verdictEl ? verdictEl.value : null,
    checks: checks,
  };
  const consistent = verdictEl &&
    ((verdictEl.value === "approved_for_ue_apartment" && allPass) ||
     (verdictEl.value === "rejected" && !allPass));
  document.getElementById("final-json").value =
    JSON.stringify(decision, null, 2) +
    (consistent ? "" : "\\n// 不一致：接受须六项全过，拒绝须至少一项未过");
  const btn = document.getElementById("dl-final");
  btn.disabled = !consistent;
  btn.onclick = () => download("animation_decision_input.json", decision);
}
document.querySelectorAll("#checks input, input[name=verdict]").forEach(el =>
  el.addEventListener("change", renderFinal));
renderFinal();
</script>
</body>
</html>
"""
    page = page.replace("__TITLE__", title)
    page = page.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(page)
        stream.flush()
        os.fsync(stream.fileno())
    print(f"FORWARD_REVIEW_PAGE_OK output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
