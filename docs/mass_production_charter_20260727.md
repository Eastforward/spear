# Mass-production charter (owner mandate, 2026-07-27)

Owner directive (verbatim intent): generate EVERY sound-source asset and
EVERY instance-level variant enumerated in
`AVEngine-habitat-native/docs/planning/INDOOR_SOUND_SOURCE_ASSET_CANDIDATES_20260727.md`;
the assistant personally inspects every product; every method must
generalize (registry/contract-driven, zero per-asset special cases).
Goal-mode sessions execute against this charter.

## Scope

1. **Animals (hardened route)** — T1 roster: 14 dog breeds + 7 cat breeds
   (Shiba shipped; Corgi, British Shorthair queued first).  T2 small
   mammals (rabbit/ferret/rodent) only after a motion-family decision.
2. **Static objects** — T1-B ~14 items + T2 ~13 items.  Default 3D route:
   FLUX->Pixal3D->watertight WITHOUT rigging (skip TokenRig/retarget/
   gait), then emitter-anchor measurement and registry.  Mesh libraries
   only with per-item rights records if the generated route fails an item.
3. **Instance variants** —
   - Animals: per-breed OFAT axes via deterministic realizers
     (size 0.85/1.0/1.15, build 0.84/1.0/1.16, life_stage young/adult/
     senior, coat = registered 3-level luminance).  True recolors stay
     BLOCKED until the projection rework ships (owner rejection on
     blue-merle stands).
   - Statics: appearance variants generated directly at FLUX stage
     (per-color/material instances are cheap without rigging); same-class
     different-appearance PAIRS are a priority product (benchmark hard
     negatives).
4. **Dry sounds** — per approved class: AudioSet curation (local 22k
   subset first, full crawl via owner's script), per-clip rights
   provenance, registry entries with acoustic_profile independence.

## Execution order (waves, batch-amortized)

W1: Corgi + British Shorthair (in-flight validators) + first 5 static
    objects (phone, alarm clock, doorbell, microwave, kettle).
W2: remaining low-risk dogs/cats (Husky, GSD, Chihuahua, Siamese, Sphynx,
    Russian Blue, American Shorthair, Beagle-gen, Jack Russell) +
    remaining T1 statics.
W3: medium-risk coats (Golden, Poodle, Pomeranian, Ragdoll, Maine Coon,
    French Bulldog) + hardened remakes of Labrador/Border Collie + T2
    statics.
W4: Dachshund (stress), OFAT variant batches over every shipped animal,
    static appearance-pair batches, dry-sound registry completion.

Batch FLUX/Pixal jobs per wave (one model load per batch); TokenRig via
the resident warm server; UE imports batched per wave with one cook.

## Supervision protocol (assistant, per artifact, before any owner gate)

- 2D canonical: 8-hard-gate pre-check with written verdict per image.
- Raw mesh: geometry audit numbers + low-slice membrane test + turntable
  visual inspection (legs/tail/holes) — my eyes on frames, not just gates.
- Watertight: boundary/nonmanifold zeros + turntable inspection.
- Post-retarget (animals): gait audit + deformation numbers + my visual
  check of peak-phase frames front/side/rear BEFORE the owner sees pages.
- Weight repair: verify no new poke/speckle artifacts vs pre-repair
  renders (the v1-aggressive lesson).
- Foot-level probe on every animal (front/hind mesh-bottom differential;
  the tilt lesson).
- Statics: silhouette fidelity vs 2D, emitter anchor sanity (grille/bell
  location), scale sanity vs physical profile.
- Every inspection verdict recorded in the workspace as JSON/notes; any
  fail stops the item (fail-closed), never silently reworked.

## Owner-gate batching (goal mode, owner intermittently present)

The three contract gates stay owner-only.  In goal mode: never skip or
self-approve a gate; instead QUEUE — build combined review pages
(all pending 2D decisions on one page; all pending final reviews on one
page), post the URLs, and continue pipeline work on items that are past
their gates.  Statics have no animation gates: 2D accept + final static
review only.  If the queue blocks all remaining work, park cleanly and
report.

## Generalization rules (hard)

New breed/object = data only: attribute profile JSON + registry rows +
coat-contract registration.  Any need to touch production Python for a
specific asset is a defect to fix generically first.  Per-asset UE Z
deltas must come from measured leveling evidence.  All thresholds live in
tools/contracts, not in run scripts.

## Definition of done (per asset)

Animal: shipped scorecard 5/5 (incl. UE readback) + registry row + owner
final approval recorded.  Static: watertight + emitter + registry row +
owner static approval.  Variant batch: OFAT verifier pass + my visual
contact-sheet check + owner attribute review.  Dry sound: registry entry
with rights provenance + audition spot-check.
