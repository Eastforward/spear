# Plan 2 M1 Dataset Results

Generated: 2026-07-08 via `tools/spike_rlr/dataset_runner.py`
Branch: `feature/plan2-flag-generator-m1`
Seed: 20260708

## Delivery summary
- **40 clips sampled** (spec + flags + trajectory saved)
- **37 clips with full UE video + RLR binaural + FOA audio**
- **27 clips also have `apartment_v1_metadata.json`** (all 4 stages OK)
- **13 clips marked `render_failed.txt`** — 2 due to UE-side planner failure
  (fixed post-run: sampler now always validates plannability with the same
  furniture+walls set the UE renderer uses), 11 due to a `t0` UnboundLocalError
  in `run_audio_pass_rlr.compute_rir_and_render` when trying to record wall
  time after 0-source clips (also fixed post-run).

## Where the actual data lives

All 40 clips are under **`/data/jzy/code/AVEngine/external/SPEAR/tmp/spike_output_apartment_v2_m1/clips/clip_XXXX/`** (git-ignored; regenerate via `dataset_runner.py`). Per clip:

| File | What it is |
|---|---|
| `videos/apartment_v1_view0.mp4` | UE forward-camera 5 s clip (640×480 @ 15 fps) |
| `binaural.wav` | RLR 2-ch HRTF binaural mix (per-source solos too) |
| `foa.wav` / `foa_stereo.wav` | RLR 4-ch first-order Ambisonics + stereo decode |
| `spec.json` | Auto-generated per-clip spec (mic pose, sources, motion) |
| `flags.json` | 12 boolean flags (occluded/FOV/spatial/motion/multi-source) |
| `apartment_v1_metadata.json` | per-frame DoA / DRR / visibility metadata |
| `trajectories.npz` | numpy arrays of source XYZ per frame |
| `profile_per_clip.csv` | Per-stage timing (StageTimer aggregate) |

Example concrete paths:
- `/data/jzy/code/AVEngine/external/SPEAR/tmp/spike_output_apartment_v2_m1/clips/clip_0000/videos/apartment_v1_view0.mp4`
- `/data/jzy/code/AVEngine/external/SPEAR/tmp/spike_output_apartment_v2_m1/clips/clip_0000/binaural.wav`
- `/data/jzy/code/AVEngine/external/SPEAR/tmp/spike_output_apartment_v2_m1/clips/clip_0000/foa_stereo.wav`
- `/data/jzy/code/AVEngine/external/SPEAR/tmp/spike_output_apartment_v2_m1/clips/clip_0000/apartment_v1_metadata.json`

## What's committed here

Because clip data itself is git-ignored, this directory only carries the small
aggregate artifacts you use for reporting:

- `dataset_stats.json` — full coverage + timing
- `coverage_bar.png` — per-flag histogram
- `stage_pie.png` — pipeline stage timing pie (dominated by UE)

## Flag coverage (40 clips)
See `dataset_stats.json`. Notable coverage gaps:
- `stationary`: 0 clips — motion style weights (70/10/20 steady/stationary/stop_and_go)
  gave the planner enough moving-source volume that stationary was rarely selected.
- `sources_pass_each_other`: 1 clip — trajectories rarely cross when sampled independently.

Plan 3 will introduce I-in flag mode to boost these gaps.

## Timing (from stage_seconds)
- ue_render: ~1125 s over 27 rendered clips ≈ 41.6 s/clip
- metadata_extract: negligible (<0.4 s total)

## Post-run fixes (should give ~100% success on re-run)

1. **`rejection_sampler`** now runs `plan_path_2d('steady')` for every source as
   a validation step regardless of motion_style, so unreachable endpoints are
   rejected BEFORE the UE render tries them. `inflate_m` also aligned with the
   UE-side planner (0.15 both).
2. **`compute_rir_and_render`** initializes `t0` at function scope so 0-source
   clips no longer crash on return.
3. **`SPEAR_RIG_ASSERT`** is re-enabled in `dataset_runner._render_one_clip`;
   the per-clip bone-query check now happens INSIDE the render loop's
   begin_frame windows (no more post-loop `engine_service.begin_frame:157
   assert False`).
4. **`review_gate.assert_mesh_approved`** is now called in
   `run_render_pass_apartment` BEFORE UE spawn — unapproved rigs are refused
   with an actionable error. Bypass with `SPEAR_SKIP_REVIEW_GATE=1` for legacy
   specs.
