# Generated-animal morphotype readiness checklist

Purpose: predict, before any FLUX/Pixal3D/TokenRig spend, whether a requested
species or breed is likely to survive the target-native generation pipeline,
and force the risky decisions to be made explicitly instead of being
discovered four retries later.  This is a readiness checklist, not a ban
list: any morphotype may be attempted, but a high-risk attempt must say so in
its request record.

The risk factors below are synthesized from the retained failure history
(static-QA batch rejections: Persian cat, cattle, chipmunk, goat, pig, sheep,
yak; retarget/skinning rejections: horse r5–r7, donkey, British Shorthair;
accepted assets: Beagle lineage, Border Collie, Labrador, Abyssinian).

## 1. Motion-donor distance (strongest predictor)

The only production motion donor is the Quaternius universal quadruped, a
medium dog-like body plan.  Score each factor; more checks = more risk:

- [ ] Torso length-to-leg-length ratio differs strongly from a medium dog
      (horse, dachshund-like, chipmunk).
- [ ] Leg thickness is far from the donor's (thin ungulate legs failed
      repeatedly: horse, donkey, cattle).
- [ ] Neutral rest pose is not a four-feet-planted stand (seated chipmunk was
      rejected as incompatible with the walking rig).
- [ ] Gait pattern differs from a walking dog (amble, hop, trot-only).

Any checked box here predicts retarget/deformation trouble even when the
static mesh passes.  Plan for the deformation gate (`0.07` review / `0.08`
reject) to be the binding constraint.

## 2. Silhouette and surface reconstructability

Single-view image-to-3D fails on ambiguous or high-frequency surfaces:

- [ ] Long or fluffy fur that hides the limb silhouette (Persian, sheep):
      expect planar/spiky fur geometry and fragmented silhouettes.
- [ ] Body color or texture likely to fuse with ground/background in the
      canonical image (pig, cattle lying).
- [ ] Thin protruding features that the reconstruction may duplicate or
      detach (tails have produced two-tail and floating-tail rejections in
      three separate batches; ears and horns carry the same risk).
- [ ] Near-side/far-side limb ambiguity in the intended canonical view; if
      the view must be angled to expose all limbs, record the declared view
      yaw so heading estimation can cross-check it.

## 3. Required explicit decisions before generation

- [ ] Canonical-image view angle declared (affects the mesh's source yaw;
      see the forward declaration contract).
- [ ] Motion donor tag declared and present in
      `tools/generated_animal_forward_contract.py::MOTION_DONOR_BASIS`
      (an unknown donor fails at build time by design).
- [ ] Breed-scoped coat profile planned: three reviewed levels and a
      preserved pattern, registered in the appearance contract before the
      asset's runtime registration (an unregistered coat now fails registry
      validation).
- [ ] Emitter anchor measurement planned from the concrete final mesh (never
      reused from another species or size).
- [ ] UE component vertical correction: plan to derive it from support-plane
      leveling evidence; if it must be measured manually, record how.

## 4. Attempt budgeting

- Low risk (0 checks in sections 1–2): budget one generation round plus one
  contingency round.
- Medium risk (1–2 checks): budget two rounds and schedule the
  `--preview-only` triage after the first retarget before any full render.
- High risk (3+ checks): expect the historical horse/donkey pattern
  (multiple rejected rounds, possible deterministic limb repair).  Get an
  explicit owner decision that the species is worth that budget before
  generating, and record the decision with the attempt ledger.

## 5. Standing rule

A failed attempt is recorded with its original request and failure reason;
it is never silently replaced by a different seed, another breed's mesh or a
template silhouette.  Four visible limbs do not excuse a wrong-breed
silhouette, and an easier template mesh is never a substitute for the
requested morphotype.
