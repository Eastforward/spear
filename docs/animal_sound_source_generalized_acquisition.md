# Generalized acquisition for animal sound sources

> Status: `research_only`. The routing and bounded-exploration contracts are
> tested, but the wave runner and resident warm-service consumer are not part
> of the formal production route. Do not use this document as an admission
> claim or enable Qwen-based experiments from it.

How a new animal sound source gets onto the hardened route without new
production code, and how the seed search that finds it is bounded and
recorded instead of hidden.

Read with `generated_animal_hardened_route_runbook.md` (the per-stage
operating procedure, unchanged by this document) and
`generated_animal_morphotype_readiness_checklist.md` (the human-facing risk
reasoning this contract mechanizes).

## Why this exists

Three animals shipped on three prompt templates named after the breed:

| asset | prompt_template_id |
|---|---|
| Shiba | `quadruped_four_limb_rest_side_i2i_v1_shiba_inu` |
| British Shorthair | `..._i2i_v2_reconstruction_safe_british_shorthair` |
| Corgi | `quadruped_authoritative_image_local_tail_stump_edit_v1` |

Each template carried knowledge that had been paid for: the reconstruction-safe
limb block, the tail-singularity wording, the localized anatomical edit.  None
of it reached the next breed, because it was keyed to a breed name.  Measured
against the trait library below, the Shiba profile carries 2 of its 7 relevant
guards, the Corgi 3 of 14, the British Shorthair 4 of 11 — and the Corgi's
negative vocabulary is disjoint from the other two, because its route was an
image edit rather than a clay-guide transform.

The Corgi's limb-separation fix survived only inside workspace directory
names (`_11_imagegen_four_limb_separation_v1`,
`_13_imagegen_upper_limb_corridors_v1`) across thirteen workspaces and roughly
forty-six hours.

## The two contracts

**`data/.../contracts/animal_morphotype_traits_v1.json`** — what a morphotype
costs and what it needs.  Each trait value carries its risk weight, the pose
guards and negative terms it requires, the gates it must pass, and the
evidence it was scored from.  Adding knowledge means editing one trait value;
every animal declaring that trait inherits it.

**`data/.../contracts/animal_acquisition_roster_v1.json`** — every animal
sound source, declared as traits plus the AudioSet classes and event classes
it exists to voice.  A roster row is the whole per-animal authoring step.

Traits are declarative body facts, not routes: `leg_length_class`,
`coat_class`, `tail_class`, `muzzle_class`, `rest_pose_family`,
`gait_family`, `size_class`.

## Adding an animal sound source

1. Add a roster row: taxonomy, wave, shoulder height, traits, AudioSet classes
   and event classes.
2. `tools/controlled_animal_morphotype_routing.py --asset-key <key>` — read the
   risk tier, the exploration budget, the required gates and the composed
   prompt blocks.
3. Author the candidate profile JSON using those blocks.  The 17-field profile
   schema is unchanged; the composer supplies content, not structure.
4. Run the wave (below).

There is no step where production Python is edited for a specific animal.  If
one appears, that is the defect to fix, per the standing generalization rule.

## What the traits buy you

A morphotype that has never been generated starts from everything the last
animal with that trait learned.  The Dachshund has never been attempted, and
already inherits the Corgi's operative fix — which is a **pose** change, not a
word list:

- stagger the near/far limbs longitudinally rather than widening the stance;
- keep negative-space corridors through the **upper attachments** (armpit and
  groin), which is where the under-chest membrane forms;
- a 10–15 degree off-lateral view is permitted and preferred when that is what
  reveals all four limbs, with the yaw declared so heading estimation can
  cross-check it.

It also inherits the British Shorthair's tail-singularity guards from its
`long_thin` tail, and lands at tier `high` (10 exploration seeds, mandatory
`--preview-only` triage) before any GPU is committed.

## Risk tiers drive budget, never admission

The checklist is explicit that it is a readiness list, not a ban list, and the
contract keeps that: a high tier means a **bigger** exploration batch and
mandatory triage, not a refusal.  Two things escalate to the owner *before*
GPU spend rather than after — a motion-donor risk at or above the declared
threshold, and a non-walking gait or non-planted rest pose
(`motion_family_decision`).  Both are declared budget decisions, not
rejections.

Tier weights are a first calibration from the retained failure history.  They
are data: when an outcome disagrees with a score, correct the trait value and
every animal declaring it moves with it.  Over-scoring is the cheap direction —
exploration images are batched, a failed downstream round is not.

## Exploration then freeze

`data/.../contracts/animal_exploration_then_freeze_v1.json` and
`tools/controlled_animal_exploration_freeze.py`.

The no-seed-lottery policy is untouched and still governs the shipped asset.
Exploration selects a **seed** and produces a **declaration**; it never
produces shippable bytes.  Those come afterwards from the same one-shot route
as before: one invocation, one image, one declared seed, no retry.

```
plan    --asset-key <key> --plan-id <id>
        # candidate set declared up front; seeds are a sha256 ladder over the
        # plan id, so the operator picks the plan and the count, never a seed
record  --plan <plan.json> --candidate ORDINAL=PATH ...
        # every declared ordinal, failures included
freeze  --plan <plan.json> --batch <batch.json> --selected-index N \
        --selection-criterion <gate> --unselected-reason ORDINAL=REASON ...
verify  --declaration <frozen.json> --regenerated-image <candidate.png>
        # the frozen seed must reproduce the selected candidate bitwise
```

This is net *more* provenance than the previous regime, not less.  The search
happened either way; before, its size and stopping criterion lived in directory
suffixes.  Now the frozen declaration records how many candidates were
considered, which gate selected the winner, why each of the others was
rejected, and whether the seed reproduces bitwise.  Fail-closed throughout:
a missing candidate, a missing rejection reason, a criterion outside the
declared vocabulary, a tampered ladder, or a non-reproducing seed all stop the
run.  A non-deterministic regeneration can be recorded as
`seed_recorded_bitwise_unverified` only behind an explicit flag.

Exhausting a batch without a winner is recorded as `exploration_exhausted` and
escalates to the owner; retrying needs a new plan id, so the previous batch
stays in the ledger.

## Waves

`tools/run_controlled_animal_acquisition_wave.py --wave T2 --plan-id-prefix
<prefix> --write-exploration-plans`

Plans every acquirable entry in a wave as one batch, cheapest tier first, and
reports the owner decisions that block before GPU spend.  Wave T2 as rostered:
9 animals, 49 exploration images, **one** FLUX load and one Pixal3D load
instead of nine of each.  Against a roughly twenty-minute Pixal cold load
versus roughly one minute of inference, that is the single largest scheduling
win available on this route.

The driver plans and emits the runbook's own commands; it does not replace the
execution path.
