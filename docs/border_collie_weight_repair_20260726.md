# Border Collie Walking weight repair (2026-07-26)

Branch `cc-asset-pipeline-hardening`.  Closes the standing Walking
deformation exception (`0.0884` > `0.08` reject gate, accepted-with-exception
on 2026-07-23) on the generated Border Collie.  Evidence roots:
`tmp/new_animal_assets/border_collie_weight_repair_20260726_01/` (aggressive
v1, superseded) and `..._02_gentle/` (accepted candidate v2).

## Root cause

Diagnostic tracing (top-64 worst Walking edges) showed 64/64 cross-limb
dominant edges, 60 of them between the two hind legs: the single-view
Pixal3D reconstruction welded the hind legs into one connected membrane
below ~40% rest height (one low-slice component), whose vertices carried
weights from both hind-leg chains and tore at two gait phases half a period
apart.  Idle was always clean (0.0113) — a pose-specific, extremely local
defect (13 of 115,019 edges over gate).

## Repair runs (tools/blender_repair_animated_quadruped_weight_stretch.py)

| run | parameters | Walking worst (rot-inv) | changed vertices | visual |
|---|---|---|---|---|
| v1 aggressive | component-parent-lock, rings 3, threshold 0.006 (historical default), passes 12 | 0.0283 pass | 17,124 | REJECT-class regression: new dark poke/speckle artifacts on chest and forelegs, caught by project owner |
| v2 gentle | component-parent-lock, rings 4, threshold 0.02, passes 6 | 0.0256 pass | 6,779 | clean at both peak-tear phases; visually equivalent to pre-repair |

Both runs: geometry/topology/action-curve fingerprints unchanged (hard
gates), gait-direction audit `forward` (stance drift ~172.6 deg), Idle
0.006–0.007.

## Lessons

- Numeric gates certify that tearing is gone; only visual review certifies
  that nothing new appeared.  v1 passed every automatic gate while
  introducing visible skin pokes from over-broad weight reassignment.
- Prefer the smallest seed set that clears the acceptance gate
  (`--extension-threshold` at ~0.02).  The tool and hardened runner now use
  this owner-reviewed gentle default; the historical 0.006 setting broadly
  reseeded an order of magnitude more vertices for no acceptance benefit.
- The hind-leg membrane still exists geometrically; this repair stops the
  tearing but true removal needs topology separation or better multi-view
  generation.  Recorded as an open item for the generation route.

## Status

v2 is `research_candidate_pending_owner_visual_signoff` for replacing the
Walking-exception GLB.  Formal replacement additionally requires re-running
the downstream UE import/readback chain on the repaired GLB before any
registry revision change.
