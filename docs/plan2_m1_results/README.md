# Plan 2 M1 Dataset Results

Generated: 2026-07-08 via `tools/spike_rlr/dataset_runner.py`
Branch: `feature/plan2-flag-generator-m1`
Seed: 20260708

## Delivery summary
- **40 clips sampled** (spec + flags + trajectory saved)
- **27 clips fully rendered** end-to-end (UE video + RLR binaural + FOA + metadata)
- **13 clips render-failed** — path-planning inside UE-side scene composer failed for
  some randomly-sampled endpoints; skipped via `render_failed.txt` marker (dataset_runner
  continues to next clip). Sampler + planner endpoints validated against
  furniture+walls; remaining failures are due to shell fragmentation
  (e.g. mic placed in an outside-shell pocket not fully captured by our shell_actor bboxes).

## Flag coverage (40 clips)
See `dataset_stats.json`. Notable coverage gaps:
- `stationary`: 0 clips — motion style weights (70/10/20 steady/stationary/stop_and_go)
  gave the planner enough moving-source volume that stationary was rarely selected.
- `sources_pass_each_other`: 1 clip — trajectories rarely cross when sampled independently.

Plan 3 will introduce I-in flag mode to boost these gaps.

## Timing (from stage_seconds)
- ue_render: ~1125 s over 27 rendered clips ≈ 41.6 s/clip
- metadata_extract: negligible (<0.4 s total)

## Files
- `dataset_stats.json` — full coverage + timing
- `coverage_bar.png` — per-flag histogram
- `stage_pie.png` — pipeline stage timing pie (dominated by UE)
- Raw per-clip data lives in `tmp/spike_output_apartment_v2_m1/` (gitignored)
