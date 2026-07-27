# Shiba Inu hardened-route shipping test (2026-07-26)

Branch `cc-asset-pipeline-hardening`.  First new asset generated from zero
through the hardened route to validate the owner's shipping question: "can a
new asset ship stably through this pipeline?"  Workspace:
`tmp/new_animal_assets/shiba_inu_20260726_01/`.

## Chain executed (single day, one candidate, no seed retries)

FLUX.2 canonical (one-shot, profile
`dog_shiba_inu_four_limb_rest_side_clay_v1`) -> owner 2D accept ->
ISNet/Pixal inputs -> Pixal3D (1024 cascade, GPU 3) -> geometry audit ->
watertight r200 proxy (42,380 faces, 0 boundary, 0 nonmanifold) -> owner
static accept -> TokenRig (seed 42; the run used a prestarted bpy experiment)
-> forward estimate
(170.60 deg, confidence 0.67, leg-spacing signal fired) -> owner one-click
confirm -> forward declaration -> heading -> rig audit -> support-plane
leveling -> retarget (donor-constant yaw0/matched) -> gait audit -> 
deformation audit -> deterministic weight repair -> re-audits -> six-view
renders -> owner final approval.

## Shipping criteria scorecard

| criterion | result |
|---|---|
| gait passes first retarget, zero orientation parameters | PASS (three audits forward; final stance drift -177.2 deg) |
| deformation under 0.07/0.08 gates without acceptance exception | PASS at full amplitude 1.0 (Walking 0.0222 / Idle 0.0108) via the deterministic repair stage; as-generated was 0.1089 |
| at most three designed human touchpoints | PASS (2D accept, head-end confirm, final review) |
| no cross-limb membrane class defect | PASS (largest low-slice component 3,347 faces vs legacy Collie's 17,784 welded blob; verified visually front/rear) |
| UE import/readback | PASS-except-live-bundle (2026-07-27): headless import 8 assets + BP, full-flag UAT cook, orbit z readback -1.181 cm at scale 1.0 (mesh-foot leveling holds; frame delta 0), frames personally inspected 4 views, runtime registry entry committed (habitat-native fbeab98) with instance-measured basis bones, canary binding resolution exercised. The live in-episode readback render (forward <=25 deg / floor <=5 cm) rides the asset-bound UE bundle batched with Corgi. Evidence: `shiba_inu_20260726_01/ue_readback_supervision_20260727.json` |

## Defects found by the run and fixed as reusable pipeline improvements

1. `9f940c34` — Pixal runner failed any symlinked invocation at the final
   manifest-path validation after burning a full generation; output root is
   now resolved.
2. The prestarted TokenRig experiment exposed a false optimization: upstream
   `demo.py` still started a second child, whose bind failure was masked by
   the old server's `pong`; the old service also listened on `0.0.0.0`.
   That resident was stopped.  The runtime patch now forces loopback and
   generation-scoped audit records, but resident reuse is not admitted until
   an authenticated consumer explicitly skips child creation.
3. Deterministic weight repair promoted to a standard chain stage: TokenRig
   native weights exceeded the Walking gate at full amplitude on both
   measured generated assets (Collie 0.0884@0.40, Shiba 0.1089@1.0); the
   gentle recipe (threshold 0.02, component-parent-lock, rings 4) passes
   both at full amplitude with no visual regression.
4. `4e4fbf34` — mesh-foot-bottoms support-plane source: TokenRig extends
   foot bone tails unevenly below the mesh, so bone-tip leveling left the
   asset floating 4.4 cm with a +3.66 cm front-high pitch (owner-caught);
   leveling on the visible mesh contact bands zeroes both (rest diff
   -0.12 cm) while remaining one rigid transform.

## Deterministic runner replay (2026-07-27)

The integrated conditional-repair runner was first replayed from the frozen
TokenRig GLB into
`review_run_v3_runner_auto_20260727/`; no image or rig inference was rerun.
The initial retargeted GLB was byte-identical to the earlier candidate
(`30cfbcac...`) and correctly failed deformation
(Walking `0.108901891`, Idle `0.029886422`).  `auto` then selected the
gentle repair, converged in five passes with zero remaining seed edges, and
produced the byte-identical previously approved repaired GLB
`6cd1a2f709dff0834d563d6508cfb5a30849d01040bc62fa779ea39641ca96c0`.
The final gait passed and final rotation-invariant deformation maxima were
Walking `0.022163509` and Idle `0.010814239`.

All six H.264 views passed media readback at 512x384, 8 frames; representative
Walking/Idle side, front and rear frames were visually inspected without a
new poke/tear regression.  The subsequently hardened v3 verifier was also
run over every frozen stage manifest and accepted heading, rig, mesh-foot
leveling, authenticated donor retarget, initial gates, exact repair recipe,
and both final gates.

The current v3 runner was then executed as a second, complete end-to-end run
into `review_run_v3_runner_auto_replay_20260727_codex1/`.  Its
`review_run.json` records schema
`avengine_target_native_generated_quadruped_review_run_v3`; the original
target, authenticated declaration and donor remained hash-identical before
and after their consuming stages.  It reproduced the byte-identical initial
and repaired GLBs (`30cfbcac...` and `6cd1a2f7...`), the same initial
deformation rejection, five-pass repair, final gait/deformation pass and six
H.264 readbacks.  A new six-view contact sheet plus all eight Walking side
frames were inspected without a poke, tear or direction regression.  This
evidence remains a research candidate and does not bypass the recorded human
final-review or live UE bundle gate.

## Owner review trail (verbatim, recorded per gate)

2D "柴犬2d的图可以" -> static "这个狗还可以" -> head-end "我觉得没有问题"
-> tilt finding "狗狗其实有一点点倾斜，并且前脚高于后脚" (fixed) -> final
"我觉得确实没有问题了".  Full decisions in the workspace JSON records.

## Standing observations

- The strict-side-view 2D gate correlated with clean 3D topology (first
  positive data point: near-axis yaw 8.6 deg, separated legs, preserved
  tail-curl hole).  Corgi (medium-risk) and British Shorthair (revival)
  will test whether the correlation holds.
- Fur-heavy breeds reconstruct as thousands of floating fur shells; the
  watertight stage absorbed them cleanly here, and the resulting proxy kept
  breed silhouette.
- This test is research-route evidence only; no dataset admission claim.
  UE import/readback and runtime registration remain before the asset can
  replace the deprecated legacy Collie in any pair template.
