# Reporting Conventions

Rules for how the agent reports rendered/generated clip results back to the user. These live outside `agents.style_guide.md` because they cover human-facing communication, not code shape.

## Per-clip source count must separate in-FOV vs out-of-FOV

When reporting a clip that contains N audio sources (e.g., animals), do **not** collapse to "N sources". Split by camera FOV membership so the user knows which sources should be visible in the video and which are audio-only:

    clip_XXXX — N sources total (K in camera FOV, N-K out of FOV)

Where "in FOV" is the boolean `is_stays_in_camera_fov` from `tools/spike_rlr/flag_verifier.py` (a source is "in FOV" iff every frame of its trajectory falls inside the camera frustum). A source that starts in and drifts out counts as **out** — because when the user opens the video they can't confirm the source without watching every frame.

**Why:** without the split the user can't validate what they see. "clip_0001 — 1 dog" and a video with no visible dog looks like a rendering bug, when it's actually a legitimate `leaves_camera_fov=true` sample. The distinction removes that ambiguity.

Rationale traced from Plan 2 M1 smoke incident on 2026-07-08 where a report of "1 dog" for a video with the dog outside the camera frustum was flagged as a suspected bug (see `tmp/spike_output_apartment_v2_smoke/clips/clip_0001/`).

## Absolute paths for artifacts the user needs to open

When pointing to a video, image, log, or generated file the user is expected to open, always give the **absolute** path. Relative paths ambiguate against the terminal's cwd and force the user to guess the base.
