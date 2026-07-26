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
static accept -> TokenRig (seed 42, resident bpy server) -> forward estimate
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
| UE import/readback | PENDING (next gate) |

## Defects found by the run and fixed as reusable pipeline improvements

1. `9f940c34` — Pixal runner failed any symlinked invocation at the final
   manifest-path validation after burning a full generation; output root is
   now resolved.
2. `tools/dev_warm_services.sh` + reuse-mode TokenRig script — the ~390 s
   CPU-bound bpy import tax is paid once per resident server instead of per
   asset; the server serves consecutive assets (historical precedent).
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
