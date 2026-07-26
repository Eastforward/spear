# Forward-contract backtest against retained human decisions (2026-07-26)

Branch `feature/asset-pipeline-hardening`.  Zero-risk validation of the new
single-point forward contract tooling: run the deterministic estimator and
the gait-direction audit on retained real asset files whose human review
truth is already recorded, without touching any frozen artifact.  Evidence
root: `tmp/new_animal_assets/forward_contract_backtest_20260726_01/`.

## Results: 4/4 agree with the retained human decisions

| Test | Input (retained file) | Tool output | Human truth | Verdict |
|---|---|---|---|---|
| Border Collie forward estimate | `border_collie_target_native_v2_20260722_01/.../tokenrig_seed42_v2_prestarted/tokenrig_native.glb` | 51.024 deg, confidence 0.17 | reviewed 49.325 deg | axis error 1.70 deg, head end correct; low confidence correctly routes to human confirmation |
| Labrador forward estimate | `animal_generated_mesh_rig_v2_20260722_01/flux_base_labrador/.../tokenrig_native.glb` | 173.663 deg, confidence 0.65 | reviewed 180.0 deg | axis error 6.34 deg, head end correct |
| Border Collie gait audit | accepted leveled retarget `.../support_plane_level_v1/retarget_v5_spike_yaw0_matched_amp0p40/border_collie_animated.glb` | forward, stance-drift yaw 172.61 deg, drift ratio 1.28e-3 | owner-accepted motion | consistent |
| Labrador gait audit | accepted bound `.../flux_base_labrador/binding/animated_walk_idle_bound.glb` | forward, stance-drift yaw 179.50 deg, drift ratio 2.85e-2 | owner-accepted motion | consistent |

Visual confirmation: the candidate hero render at 51 deg shows the Border
Collie face-on (head, ears, tongue); the 231 deg render shows the tail end.
The turntable + candidate review flow gives an unambiguous one-click
head-end decision (`collie_forward_review.html` in the evidence root).

## Boundaries and known gaps

- This validates the deterministic core (PCA axis, head-end voting, stance
  drift) on two accepted assets.  It is not dataset admission and does not
  qualify any asset.
- The leg-spacing vote returned 0 on both real meshes (the deterministic
  quadrant split is conservative on real leg geometry); head-end confidence
  currently rests on the high-verts and mass-end signals.  Improving the
  leg-band clustering is an open enhancement.
- No known-rejected (backward/sideways) animated GLB was located in this
  pass; a true-positive gait-audit backtest on a retained rejected asset is
  still worth adding when one is found.
- The full chain (declaration -> heading -> donor-constant retarget -> gait
  -> preview) has not yet been exercised end-to-end on a *new* asset; that
  is the next planned test and requires an owner-declared breed.
