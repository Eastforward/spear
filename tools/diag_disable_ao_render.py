"""Diagnostic: render one turntable with SSAO / DFAO / Lumen screen-trace AO all off,
to test the hypothesis that the mysterious wall-foot dark strips + adjacent floor
darkening are ambient-occlusion artifacts (contact darkening at the wall/floor seam).

Approach: monkey-patch spawn_camera to fire r.* console commands on the game
instance's world right after the scene capture is initialized, then invoke the
existing render_gpurir_room pipeline with the same 6000lm + pitch -30 args used
in the last baseline (codex_tune_light6000_pitch_m30 / claude_wall_overlap_fix).

Output: /data/jzy/code/SPEAR/tmp/render_gpurir_room/claude_ao_off_diag/turntable.mp4
"""
import os, sys

_EX = "/data/jzy/code/SPEAR/examples"
if _EX not in sys.path:
    sys.path.insert(0, _EX)

import render_in_apartment as ria
import render_in_gpurir_room as rgr


_original_spawn_camera = ria.spawn_camera


def spawn_camera_with_ao_off(game, width, height):
    cam, comp = _original_spawn_camera(game=game, width=width, height=height)
    # Send AO-related console commands. execute_console_command is executed by
    # the running game world. These are RENDERING cvars, not per-scene-capture,
    # so they apply globally to every draw from here on out.
    ao_off_cmds = [
        # Classic screen-space AO
        "r.AmbientOcclusionLevels 0",
        "r.AmbientOcclusion.Method 0",
        # Distance Field AO (used by Lumen when mesh SDFs are on)
        "r.DistanceFieldAO 0",
        # Lumen short-range AO (screen probe contact shadow)
        "r.Lumen.ScreenProbeGather.ShortRangeAO 0",
        # SSGI could also cause contact darkening
        "r.SSGI.Enable 0",
    ]
    for cmd in ao_off_cmds:
        try:
            game.unreal_service.execute_console_command(command=cmd)
            print(f"[AO_OFF] sent: {cmd}", flush=True)
        except Exception as e:
            print(f"[AO_OFF] FAILED {cmd}: {e}", flush=True)
    return cam, comp


# Monkey-patch BOTH names: render_gpurir_room imports spawn_camera by name
ria.spawn_camera = spawn_camera_with_ao_off
rgr.spawn_camera = spawn_camera_with_ao_off


if __name__ == "__main__":
    argv = [
        "--animal", "dog",
        "--run-name", "claude_ao_off_diag",
        "--ceiling-light-lumens", "6000",
        "--directional-light-pitch-deg", "-30",
    ]
    args = rgr.parse_args(argv)
    rgr.render_gpurir_room(args)
    print("DIAG_DONE", flush=True)
