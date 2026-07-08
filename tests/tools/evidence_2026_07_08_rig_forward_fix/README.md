# 2026-07-08 rig-forward-yaw fix — visual evidence

Bug: `QUATERNIUS_FORWARD_YAW_OFFSET_DEG = 180.0` had been the assumption
since the pipeline was first authored, based on rig-blueprint inspection
(someone reading the Blueprint asset and concluding the walking anim's
local-forward was -X_local). Running full pipeline visually showed
**every Quaternius dog walked head-first BACKWARDS**.

## Reproduction

1. Rendered `data/shoebox_v2_spec.json` with the old 180.0 offset. Golden
   walks `(0.8, 0.9) → (4.4, 0.9)` — motion +X_world. Mic in front of
   dog at yaw=270° means +X_world appears as image-right.
2. Frames 20/40 (see `shoebox_offset180_frame{20,40}_BROKEN.png`) show
   the dog's **tail on the right, head on the left**. Head opposite of
   motion = walking backwards.
3. Set the constant to 0.0 in `tools/species_rig_map.py`, re-rendered.
   Same scene, same mic pose, same trajectory.
4. Frames 20/40/60 (see `shoebox_offset0_frame*.png`) now show
   **head on the right, tail on the left**. Head = motion direction.
   Fixed.
5. Same fix verified on `data/apartment_v1_spec.json` (mic yaw=180°,
   golden walking +Y_world). See `apartment_offset0_frame{60,70}.png` —
   golden appears in image-right (which is +Y_world for mic-yaw=180)
   with head to the right = motion direction. Fixed.

## Why the assumption was wrong

Reading a Quaternius `.glb` Blueprint in an editor makes the local-forward
look ambiguous — the arrow gizmo Blender/UE draws depends on which axis
convention you assume. The only reliable check is: **render the animal
and look at whether the head or the tail is at the leading edge of
motion**. Rig-inspection is not evidence; render-and-look is.
