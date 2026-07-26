#!/usr/bin/env python3
"""Build a self-contained local HTML review page for a generated animal.

Confirm-first contract: the pipeline has already auto-applied the estimated
heading, so the reviewer sees the *adjusted* asset with a green ground arrow
marking the claimed forward direction and only chooses between three
outcomes — confirm, flip 180 degrees, or reject to manual review.  The
reviewer never has to judge the raw oblique mesh unless they reject.

Panels:

1. Auto-normalized confirmation: preloaded draggable turntable of the
   heading-normalized mesh (green +X arrow), estimator evidence, three
   decision buttons, live decision JSON and the exact follow-up CLI command.
   An optional collapsed section shows the raw un-normalized turntable for
   the reject/manual path.
2. Gait and preview: the fail-closed gait-direction verdict and preview
   walking renders.
3. Final animation review: six-view videos plus the six human boolean
   checks; approving requires all six, mirroring the CLI decision tool.

Everything is referenced by relative path, so the page works from the local
filesystem or any static file server with no external dependencies.  The
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


SCHEMA = "avengine_forward_review_page_v2"
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
    parser.add_argument(
        "--normalized-turntable-dir",
        type=Path,
        required=True,
        help="Turntable of the heading-normalized mesh (forward-axis arrow).",
    )
    parser.add_argument(
        "--raw-turntable-dir",
        type=Path,
        help="Optional turntable of the raw mesh for the manual-review path.",
    )
    parser.add_argument("--estimate-json", type=Path, required=True)
    parser.add_argument("--gait-audit", type=Path)
    parser.add_argument(
        "--preview-video", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--review-video", type=Path, action="append", default=[]
    )
    parser.add_argument("--output-html", type=Path, required=True)
    return parser.parse_args(argv)


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative_to_page(path: Path, page: Path) -> str:
    return os.path.relpath(Path(path).resolve(), page.resolve().parent)


def turntable_frames(directory: Path, page: Path) -> list[str]:
    manifest = load_json(directory / "turntable_manifest.json")
    return [
        relative_to_page(directory / frame["path"], page)
        for frame in manifest["frames"]
    ]


def main(argv=None):
    args = parse_args(argv)
    output = args.output_html.resolve()
    if output.exists() or output.is_symlink():
        raise SystemExit(f"refusing to replace review page: {output}")

    estimate_run = load_json(args.estimate_json)
    estimate = estimate_run["estimate"]
    gait = load_json(args.gait_audit) if args.gait_audit else None

    payload = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "asset_workspace": args.asset_workspace,
        "input_glb": estimate_run.get("input"),
        "estimate": estimate,
        "gait": gait,
        "normalized_frames": turntable_frames(
            args.normalized_turntable_dir, output
        ),
        "raw_frames": (
            turntable_frames(args.raw_turntable_dir, output)
            if args.raw_turntable_dir
            else []
        ),
        "previews": [
            {"label": video.stem, "src": relative_to_page(video, output)}
            for video in args.preview_video
        ],
        "reviews": [
            {"label": video.stem, "src": relative_to_page(video, output)}
            for video in args.review_video
        ],
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
                cursor: ew-resize; user-select: none; touch-action: none; }
input[type=range] { width: 100%; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
td, th { border-bottom: 1px solid #2c2c36; padding: 4px 8px; text-align: left; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px;
         font-size: 12px; font-weight: 600; }
.badge.pass { background: #143d2b; color: #5be49b; }
.badge.fail { background: #46181c; color: #ff8089; }
.badge.warn { background: #423512; color: #ffd166; }
.decide { display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0; }
.decide button { flex: 1 1 150px; padding: 12px; border-radius: 10px;
  border: 2px solid #2c2c36; background: #131318; color: #e6e6ea;
  font-weight: 600; cursor: pointer; }
.decide button.selected { border-color: #5be49b; background: #143d2b33; }
.decide button.flip.selected { border-color: #ffd166; background: #42351233; }
.decide button.reject.selected { border-color: #ff8089; background: #46181c33; }
textarea, pre.cli { width: 100%; background: #131318; color: #cde3d3;
  border: 1px solid #2c2c36; border-radius: 8px; padding: 10px;
  font: 12px/1.5 ui-monospace, monospace; white-space: pre-wrap; }
textarea { min-height: 160px; }
button.dl, a.btn { background: #2b5f46; color: #eafff3; border: 0;
  border-radius: 8px; padding: 8px 16px; font-weight: 600; cursor: pointer; }
button.dl:disabled { background: #333; color: #777; cursor: not-allowed; }
video { width: 100%; border-radius: 8px; background: #000; }
label.check { display: block; padding: 4px 0; }
.muted { color: #9a9aa5; font-size: 12px; }
details { margin-top: 14px; }
#load-progress { font-size: 12px; color: #9a9aa5; }
</style>
</head>
<body>
<main>
<h1>__TITLE__</h1>
<p class="muted">管线已按估计值自动转正；绿色箭头 = 声明的前进方向。你只需确认。此页面只产出决策 JSON，不修改任何管线状态。</p>

<section id="panel-forward">
  <h2>① 自动转正确认</h2>
  <div class="row">
    <div class="col">
      <img id="turntable" class="turntable" alt="normalized turntable">
      <input id="orbit" type="range" min="0" value="0" disabled>
      <div id="load-progress">帧预加载中…</div>
      <p class="muted">拖动滑条或在图上左右拖拽旋转。绿色箭头指向声明的"前"。</p>
    </div>
    <div class="col">
      <table id="estimate-table"></table>
      <div class="decide">
        <button id="btn-confirm">✅ 确认<br><span class="muted">动物面朝箭头方向</span></button>
        <button id="btn-flip" class="flip">🔄 头尾反了<br><span class="muted">翻转 180°</span></button>
        <button id="btn-reject" class="reject">❌ 轴不对<br><span class="muted">转人工细审</span></button>
      </div>
      <p><button id="dl-forward" class="dl" disabled>下载头端决策 JSON</button></p>
      <pre class="cli" id="cli-forward">（做出选择后生成后续 CLI 命令）</pre>
      <details id="raw-details" hidden>
        <summary>原始未转正网格（人工细审用）</summary>
        <img id="raw-turntable" class="turntable" alt="raw turntable">
        <input id="raw-orbit" type="range" min="0" value="0">
      </details>
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
      <p><button id="dl-final" class="dl" disabled>下载终审决策 JSON</button></p>
      <p class="muted">接受要求全部六项勾选；拒绝要求至少一项未勾选（与 CLI 决策工具同规则）。</p>
    </div>
    <div class="col"><textarea id="final-json" readonly></textarea></div>
  </div>
</section>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);

// -- preloaded turntable widget --
function turntableWidget(imgEl, rangeEl, frames, onReady) {
  if (!frames.length) return;
  rangeEl.max = frames.length - 1;
  const cache = new Array(frames.length);
  let loaded = 0;
  frames.forEach((src, i) => {
    const im = new Image();
    im.onload = im.onerror = () => {
      loaded += 1;
      if (onReady) onReady(loaded, frames.length);
      if (loaded === frames.length) rangeEl.disabled = false;
    };
    im.src = src;
    cache[i] = im;
  });
  function show(i) {
    const n = frames.length;
    imgEl.src = cache[((i % n) + n) % n].src;
  }
  rangeEl.addEventListener("input", () => show(+rangeEl.value));
  let drag = null;
  imgEl.addEventListener("pointerdown", e => {
    drag = { x: e.clientX, f: +rangeEl.value };
    imgEl.setPointerCapture(e.pointerId);
  });
  imgEl.addEventListener("pointermove", e => {
    if (!drag) return;
    const n = frames.length;
    const f = (((drag.f + Math.round((e.clientX - drag.x)/8)) % n) + n) % n;
    rangeEl.value = f; show(f);
  });
  imgEl.addEventListener("pointerup", () => drag = null);
  show(0);
}

const progress = document.getElementById("load-progress");
turntableWidget(
  document.getElementById("turntable"),
  document.getElementById("orbit"),
  D.normalized_frames,
  (done, total) => {
    progress.textContent = done === total ? "已就绪" : `帧预加载中 ${done}/${total}`;
  });

// -- estimate table --
const est = D.estimate;
const votes = est.head_end_vote.signals;
document.getElementById("estimate-table").innerHTML = `
  <tr><th>估计源前向</th><td>${est.estimated_front_yaw_deg.toFixed(2)}°（已自动应用）</td></tr>
  <tr><th>头尾置信度</th><td>${est.head_end_vote.confidence.toFixed(2)}
      ${est.head_end_vote.confidence < 0.5 ? '<span class="badge warn">低置信度，请仔细看箭头</span>' : ''}</td></tr>
  <tr><th>信号投票</th><td>${Object.entries(votes).map(([k, v]) => `${k}=${v}`).join("&nbsp; ")}</td></tr>`;

// -- three-way decision --
let choice = null;
const wrap = yaw => { let y = yaw % 360; if (y > 180) y -= 360; if (y <= -180) y += 360; return y; };
const buttons = {
  confirm: document.getElementById("btn-confirm"),
  flip: document.getElementById("btn-flip"),
  reject: document.getElementById("btn-reject"),
};
Object.entries(buttons).forEach(([kind, el]) =>
  el.addEventListener("click", () => { choice = kind; renderForward(); }));

function renderForward() {
  Object.entries(buttons).forEach(([kind, el]) =>
    el.classList.toggle("selected", choice === kind));
  const btn = document.getElementById("dl-forward");
  const cli = document.getElementById("cli-forward");
  document.getElementById("raw-details").hidden =
    choice !== "reject" || !D.raw_frames.length;
  if (!choice) { btn.disabled = true; return; }
  let decision, command;
  if (choice === "reject") {
    decision = {
      schema: "avengine_head_end_review_decision_v1",
      asset_workspace: D.asset_workspace,
      input_glb: D.input_glb,
      reviewed_by: "project_owner",
      outcome: "manual_review_required",
      reason: "estimated axis rejected by reviewer",
    };
    command = "# 轴被拒：使用原始转台人工审出前向后，走 human_review 路线\\n" +
      "python tools/build_generated_animal_forward_declaration.py \\\\\\n" +
      `  --asset-workspace ${D.asset_workspace} \\\\\\n` +
      "  --input-glb <rig.glb> --confirmed-front-yaw-deg <人工审定> \\\\\\n" +
      "  --head-end-evidence head_end_review.json \\\\\\n" +
      "  --head-end-decision-source human_review --output forward_declaration.json";
  } else {
    const yaw = choice === "confirm"
      ? est.estimated_front_yaw_deg
      : wrap(est.estimated_front_yaw_deg + 180.0);
    decision = {
      schema: "avengine_head_end_review_decision_v1",
      asset_workspace: D.asset_workspace,
      input_glb: D.input_glb,
      reviewed_by: "project_owner",
      outcome: choice === "confirm" ? "estimate_confirmed" : "estimate_flipped_180",
      confirmed_front_yaw_deg: yaw,
      estimator_confidence: est.head_end_vote.confidence,
    };
    command =
      "python tools/build_generated_animal_forward_declaration.py \\\\\\n" +
      `  --asset-workspace ${D.asset_workspace} \\\\\\n` +
      `  --input-glb ${D.input_glb ? D.input_glb.path : "<rig.glb>"} \\\\\\n` +
      "  --estimate-json <estimate_run.json> \\\\\\n" +
      `  --confirmed-front-yaw-deg ${yaw.toFixed(6)} \\\\\\n` +
      "  --head-end-evidence head_end_review.json \\\\\\n" +
      "  --head-end-decision-source human_confirming_estimator \\\\\\n" +
      "  --output forward_declaration.json";
  }
  btn.disabled = false;
  btn.onclick = () => {
    const url = URL.createObjectURL(new Blob(
      [JSON.stringify(decision, null, 2) + "\\n"], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url; a.download = "head_end_review.json"; a.click();
    URL.revokeObjectURL(url);
  };
  cli.textContent = command;
}

turntableWidget(
  document.getElementById("raw-turntable"),
  document.getElementById("raw-orbit"),
  D.raw_frames);

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
  btn.onclick = () => {
    const url = URL.createObjectURL(new Blob(
      [JSON.stringify(decision, null, 2) + "\\n"], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url; a.download = "animation_decision_input.json"; a.click();
    URL.revokeObjectURL(url);
  };
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
