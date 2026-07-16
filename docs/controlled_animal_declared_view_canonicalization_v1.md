# Controlled animal declared-view canonicalization v1

## Frozen decision

For generated quadrupeds, a three-quarter source image is useful because it
keeps all four lower limbs and paws visibly separated.  That camera azimuth is
not the runtime forward direction.  The stable route therefore separates two
operations:

1. Code applies the source-view yaw declared before inference.  It never
   estimates the applied yaw from the generated mesh.
2. A human reviewer only decides whether the animal's head/tail direction must
   be flipped by 180 degrees.

For the accepted light Beagle canary, the source-view contract declares 30
degrees.  The human-selected head/tail value is 180 degrees, so the composed
binding yaw is normalized to -150 degrees.

## Executable contract

`tools/build_generated_animal_direction_canary.py` accepts:

```text
--declared-view-canonicalization-yaw-deg 30
--maximum-postcanonical-residual-deg 3
```

The generated review manifest records
`deterministic_declared_camera_view_canonicalization_v1`.  The geometry audit
fits the unsigned torso axis over three central-percentile spans.  Those
measurements are a pass/reject gate only: they cannot change the declared 30
degree transform.  Every residual must be at most 3 degrees.  A failure rejects
the single attempt and must be fixed by a new pose-guide/prompt/profile
revision; changing only the seed is forbidden.

`tools/spike_rlr/controlled_animal_direction_review_server.py` initializes the
preview with the authenticated fixed yaw.  In declared-view mode:

- axis-delta and manual two-point adjustments return HTTP 409;
- 90 and -90 degree cardinal choices return HTTP 409;
- only 0 and 180 degree head/tail choices are accepted;
- the decision records `axis_alignment_authority=declared_source_view_contract`;
- `automatic_orientation_inference_used` remains false.

`tools/run_controlled_animal_lod_binding.py` authenticates the fixed-view audit,
the exact declared yaw, the 0/180 human choice, the composed yaw, and the
rotation matrix before using the decision.  Legacy human-axis decisions remain
readable and are not rewritten.

## Real Beagle execution evidence

The exact Pixal source is:

```text
/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/dog_beagle_open_tricolor_pixal_outputs_v2_20260716/dog_beagle_open_tricolor_photorealistic_recolor_canary_169a0af3c610/pixal_raw_1024.glb
```

- Pixal GLB SHA-256:
  `5c623a4f5ef01f5073f39f62bd5b9fbfcfcec0a87e0675afc0f045c58233ba0b`
- FLUX.2 reference SHA-256:
  `acc3725f33c7d2be8bab99612cf42cccd2b814ed274327b18fec15598828d5a4`
- Reviewed 100k LOD SHA-256:
  `150923ea84d361558daed5ea4b622b6ecad2a105fb8fb4156d7231570d98814a`

The new immutable canary manifest is:

```text
/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/dog_beagle_open_tricolor_direction_canary_v3_declared_axis_20260716/review_manifest.json
```

Its internal canonical `manifest_sha256` is
`da8146c9171b5fabd5b9df04a8c7ffb33c00e6fcec978bbe9aa8fef1a7cc8081`;
the JSON file SHA-256 is
`5d9b9a9f284ea9c39084ed7895a84724fbea926d5ebd78b121d72f22bd02cb56`.
The three measured raw torso axes were 31.25755, 32.09411, and 31.20601
degrees.  After the fixed 30 degree canonicalization, the residuals were
1.25755, 2.09411, and 1.20601 degrees, so the maximum 2.09411 degree residual
passed the 3 degree gate.

The real Flask smoke state is under:

```text
/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/dog_beagle_open_tricolor_direction_state_v4_declared_axis_smoke_20260716
```

It starts at axis 30 / head-tail 0 / total 30.  Axis +1 and cardinal 90 were
rejected; cardinal 180 produced total -150.  No new human approval was
fabricated and the prior approved decision was not modified.

The preserved human decision is
`dog_beagle_open_tricolor_direction_state_v2_two_stage_20260716/decisions/dog_beagle_open_tricolor_photorealistic_recolor_canary_169a0af3c610.json`.
Its file SHA-256 is
`aec7d903802f60d1aa0f378bbcaad6c672b62d5f3acedeaef44e72872d1ed2dd`
and its canonical `decision_sha256` is
`dc223ffcf7df58c52eff0841d7fb81d0f1c7d27397f589bca4168e5b5fe84d44`.

## Accepted locked-paw visual result

The matching 20-bone Walk/Idle diagnostic output is:

```text
/data/jzy/code/AVEngine/external/SPEAR/tmp/controlled_source_asset_execution_v1/dog_beagle_open_tricolor_locked_paw_skeleton_binding_spike_v1_20260716/animated_100000_double_sided.glb
```

- SHA-256: `6bfaaae754c2435dbdb7cfd8c6c86c1594932777d1b4ac560134428201773e85`
- 99,992 triangles, one skin, 20 joints, `Idle` and `Walking`
- four paw lateral-excursion ratios: 0.0005372, 0.0003497, 0.0008826,
  and 0.0010501
- all four terminal-paw yaw excursions below 0.00037 degrees

The user visually accepted this result as sufficiently good.  Review media:

- `review_v1/walking_side.mp4`
- `review_v1/walking_front.mp4`
- `review_v1/idle_side.mp4`
- `review_v1/index.html`

This result remains a `research_candidate`, not a formal dataset asset.  The
strict deformation audit still reports isolated maximum stretch outliers even
though the user accepted the rendered motion; preserve that warning.  Direct
weight transfer from the 106,902-vertex motion carrier took 5:52 wall time and
peaked at 16,858,436 KiB RSS.  This document freezes the accepted functional
route, not a claim that its binding stage is throughput-optimized.

## Verification

The focused regression command is:

```bash
/data/jzy/miniconda3/envs/ss2/bin/python -m pytest -q \
  tests/tools/spike_rlr/test_controlled_animal_direction_review_server.py \
  tests/tools/test_run_controlled_animal_lod_binding.py \
  tests/tools/test_build_generated_animal_direction_canary_static.py
```

At freeze time, all 43 focused tests passed.
