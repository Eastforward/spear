"""SPEAR/UE render pass for apartment_v1_spec.

Flow:
  1. Load apartment_0000 map via configure_instance (SPEAR RPC).
  2. Enumerate all StaticMeshActors; classify each; destroy furniture
     actors NOT in the spec's furniture_mode keep-set.
     - furniture_mode == 'shell'   -> keep no furniture (only structural
       shell stays)
     - furniture_mode == 'subset'  -> keep actors in
       furniture_include_categories (core + decoration by default)
     - furniture_mode == 'full'    -> keep all furniture
     Shell actors are always kept.
  3. Spawn dogs via scene_two_dogs_apartment.
  4. Spawn 1 forward camera glued to mic pose, FOV from spec.
  5. Per-frame: advance dog animation, capture, save png.
  6. ffmpeg png sequence -> mp4.

Must be run under spear-env (has spear_ext + SPEAR RPC bindings) with
DISPLAY=:99 + VK_ICD_FILENAMES set. See tools/spike_rlr/run_apartment.sh
for env setup.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "data" / "apartment_v1_spec.json"
DEFAULT_OUT = REPO_ROOT / "tmp" / "spike_output_apartment"

sys.path.insert(0, str(REPO_ROOT / "examples"))
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "gpurir_scenes"))
sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))

from render_in_apartment import (  # noqa: E402
    APARTMENT_MAP, configure_instance, spawn_camera, read_frame,
)
from gpurir_scenes.run_render_pass import (  # noqa: E402
    APARTMENT_MIC_ORIGIN_CM, APARTMENT_FLOOR_Z_CM, M2CM,
    _world_from_scene, _yaw_world_to_ue, _spawn_animal, _step_animated,
)
from apartment_actor_classifier import classify_actor, SHELL_LABELS  # noqa: E402
from scene_two_dogs_apartment import (  # noqa: E402
    compose_two_dog_scene_apartment,
    _kept_furniture_bboxes,
    _shell_wall_bboxes,
)
from profiling import StageTimer  # noqa: E402


def _rig_assert_enabled() -> bool:
    """Check if per-clip rig direction assertion should run.

    Enabled by: SPEAR_RIG_ASSERT=1 env var OR --rig-assert CLI flag.
    Opt-in to preserve fast Plan 1 iteration; will be on-by-default in Plan 2.
    """
    return os.environ.get("SPEAR_RIG_ASSERT", "0") == "1" or \
           any("--rig-assert" in a for a in sys.argv)


def _check_no_clipping_apartment(spec_dict, scene, cats):
    """Safety net: raise AssertionError if any animal trajectory point enters
    any kept furniture or shell wall bbox. Called before UE render so we
    catch bad specs cheaply rather than staring at a 30 s render of a dog
    walking through a sofa."""
    obstacles = _kept_furniture_bboxes(spec_dict, cats) + _shell_wall_bboxes(spec_dict)
    for a in scene.animals:
        for k, xyz in enumerate(a.trajectory_m):
            x, y = float(xyz[0]), float(xyz[1])
            for x0, y0, x1, y1 in obstacles:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    raise AssertionError(
                        f"{a.tag} clips obstacle at frame {k}: "
                        f"pos=({x:.2f}, {y:.2f}) inside bbox "
                        f"[{x0:.2f},{y0:.2f}]-[{x1:.2f},{y1:.2f}]. "
                        f"Fix spec start/end so planner can route around, "
                        f"or set spec 'furniture_exclude_actors' to remove it."
                    )


def _load_categories():
    return json.loads((REPO_ROOT / "tools" / "spike_rlr"
                       / "apartment_furniture_categories.json").read_text())


def _compute_keep_set(spec, cats):
    """Given the loaded spec and categories JSON, return the set of actor names
    (from apartment_furniture_map.json) that should be KEPT this clip."""
    mode = spec["furniture_mode"]
    include_cats = set(spec.get("furniture_include_categories", []))
    keep = set()
    if mode == "shell":
        pass  # keep no furniture
    elif mode == "full":
        for c in ("core", "decoration", "misc"):
            keep.update(cats.get(c, []))
    else:  # subset
        for c in include_cats:
            keep.update(cats.get(c, []))
    keep.update(spec.get("furniture_include_actors_extra", []))
    keep.difference_update(spec.get("furniture_exclude_actors", []))
    return keep


def _destroy_unwanted_actors(instance, game, spec, cats):
    """Enumerate all StaticMeshActors; destroy furniture actors NOT in keep.

    Shell actors are always kept. Enumeration + classification + destruction
    all happen inside ONE begin_frame..end_frame pair (SPEAR RPC forbids
    nested begin_frame calls).
    """
    keep_actors = _compute_keep_set(spec, cats)
    n_shell = 0
    n_kept = 0
    n_removed = 0
    with instance.begin_frame():
        actors = game.unreal_service.find_actors_by_class(uclass="AStaticMeshActor")
        print(f"[apt_render] enumerated {len(actors)} StaticMeshActors")

        for actor in actors:
            try:
                name = game.unreal_service.get_stable_name_for_actor(
                    actor=actor, include_unreal_name=True) or "<empty>"
            except Exception:
                name = "<no-stable-name>"
            try:
                b = actor.GetActorBounds(bOnlyCollidingComponents=False, as_dict=True)
                origin = b["Origin"]; ext = b["BoxExtent"]
                bmin_z = origin["z"] - ext["z"]
                bmax_z = origin["z"] + ext["z"]
                x_ext = ext["x"] * 2.0
                y_ext = ext["y"] * 2.0
            except Exception:
                # No bounds -> skip (can't classify safely)
                continue
            label = classify_actor(name, bmin_z, bmax_z, x_ext, y_ext)
            if label in SHELL_LABELS:
                n_shell += 1
                continue  # always keep shell
            # furniture
            if name in keep_actors:
                n_kept += 1
                continue
            # Destroy unwanted furniture actor in place
            try:
                game.unreal_service.destroy_actor(actor=actor)
                n_removed += 1
            except Exception as e:
                print(f"[apt_render] WARN failed to destroy {name!r}: {e}")
    with instance.end_frame():
        pass

    print(f"[apt_render] shell kept: {n_shell}, furniture kept: {n_kept}, "
          f"furniture removed: {n_removed}")


def render_apartment(spec_path: Path, out_dir: Path, csv_path: Path,
                     clip_id: str = "apartment_v1_000"):
    spec = json.loads(spec_path.read_text())
    n_frames = int(spec["render_config"]["n_frames"])
    fps = int(spec["render_config"]["fps"])
    cam_cfg = spec["camera_configs"][0]
    fov_deg = float(cam_cfg["fov_deg"])
    yaw_world_deg = float(cam_cfg["yaw_deg"])
    yaw_ue_deg = _yaw_world_to_ue(yaw_world_deg, "apartment")

    scene = compose_two_dog_scene_apartment(spec_path)
    cats = _load_categories()
    # Safety net: fail fast if any planned trajectory clips a furniture / wall
    # bbox. Cheap (a few hundred us) and saves ~30 s of UE render on bad specs.
    _check_no_clipping_apartment(spec, scene, cats)

    frames_dir = out_dir / "videos" / "apartment_v1_view0"
    frames_dir.mkdir(parents=True, exist_ok=True)

    with StageTimer("ue_render", clip_id=clip_id, csv_path=csv_path):
        instance = configure_instance(rpc_port=39004)
        game = instance.get_game()
        try:
            # 1. Destroy unwanted furniture actors
            _destroy_unwanted_actors(instance, game, spec, cats)

            # 2. Spawn camera glued to mic pose
            with instance.begin_frame():
                cam, comp = spawn_camera(
                    game=game,
                    width=int(spec["render_config"]["width"]),
                    height=int(spec["render_config"]["height"]),
                )
                # Set FOV
                try:
                    comp.set_property_value(property_name="FOVAngle",
                                             property_value=fov_deg)
                except Exception as e:
                    print(f"[apt_render] WARN failed to set FOV: {e}")

                # 3. Spawn dogs
                actors = [_spawn_animal(game, a, "apartment", scene)
                          for a in scene.animals]

                # Unpause the game
                game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(
                    bPaused=False)
            with instance.end_frame():
                pass

            # Warmup for virtual texture streaming (matches shoebox pipeline)
            instance.step(num_frames=120)

            # Mic UE cm position (from APARTMENT_MIC_ORIGIN_CM + spec offset)
            mic_pos_m = np.asarray(spec["mic"]["pos_m"])
            mic_x_cm = APARTMENT_MIC_ORIGIN_CM[0] + mic_pos_m[0] * M2CM
            # apartment Y-flip: world +Y -> UE -Y from origin
            mic_y_cm = APARTMENT_MIC_ORIGIN_CM[1] - mic_pos_m[1] * M2CM
            mic_z_cm = APARTMENT_FLOOR_Z_CM + mic_pos_m[2] * M2CM

            # Set camera to mic pose, one yaw only (Plan-1 single view)
            with instance.begin_frame():
                cam.K2_SetActorLocationAndRotation(
                    NewLocation={"X": mic_x_cm, "Y": mic_y_cm, "Z": mic_z_cm},
                    NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": float(yaw_ue_deg)},
                    bSweep=False, bTeleport=True,
                )
            with instance.end_frame():
                pass
            instance.step(num_frames=40)

            # Per-frame render
            import cv2
            for frame_i in range(n_frames):
                with instance.begin_frame():
                    for actor, placement in zip(actors, scene.animals):
                        if placement.is_animated:
                            _step_animated(actor, placement, frame_i,
                                            "apartment", scene)
                    cam.K2_SetActorLocationAndRotation(
                        NewLocation={"X": mic_x_cm, "Y": mic_y_cm, "Z": mic_z_cm},
                        NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": float(yaw_ue_deg)},
                        bSweep=False, bTeleport=True,
                    )
                with instance.end_frame():
                    img = read_frame(comp)
                    cv2.imwrite(str(frames_dir / f"frame_{frame_i:04d}.png"), img)
                if frame_i % 15 == 0:
                    print(f"[apt_render] frame {frame_i}/{n_frames}")

            # ---- Plan 1.5.B: per-clip rig direction sanity check ----
            # Opt-in via env var (SPEAR_RIG_ASSERT=1) or --rig-assert flag.
            # Verifies actor's body was actually walking in the expected
            # world-frame direction. Catches yaw-formula regressions.
            if _rig_assert_enabled():
                from rig_direction_check import assert_body_forward
                for actor, placement in zip(actors, scene.animals):
                    if not placement.is_animated:
                        continue
                    # Expected world yaw derived from mid-clip velocity in
                    # the trajectory (motion_yaw before rig offset). We
                    # convert traj[frame_i+1] - traj[frame_i] on the SSOT
                    # frame for a mid-clip window.
                    mid = n_frames // 2
                    t = np.asarray(placement.trajectory_m)
                    if mid + 5 >= len(t):
                        continue
                    dxy = t[mid + 5, :2] - t[mid, :2]
                    if np.linalg.norm(dxy) < 1e-3:
                        continue
                    expected_motion_yaw_ssot = np.degrees(np.arctan2(dxy[1], dxy[0]))
                    # World<->UE apartment convention: yaw_ue = -yaw_world.
                    # Observed bone velocity is in UE cm frame.
                    expected_yaw_ue = -expected_motion_yaw_ssot
                    try:
                        assert_body_forward(
                            actor, instance,
                            expected_yaw_world_deg=expected_yaw_ue,
                            tolerance_deg=25.0,
                            context=f"apartment_v1/{placement.tag}",
                        )
                        print(f"[apt_render] rig direction OK for {placement.tag}")
                    except AssertionError as e:
                        print(f"[apt_render] {e}")
                        # Re-raise so CI can catch, but log first for humans
                        raise

            # ffmpeg -> mp4
            mp4_path = out_dir / "videos" / "apartment_v1_view0.mp4"
            subprocess.run([
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(fps),
                "-i", str(frames_dir / "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                str(mp4_path),
            ], check=True)
            print(f"[apt_render] wrote {mp4_path}")

        finally:
            instance.close(force=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(DEFAULT_SPEC))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--clip-id", default="apartment_v1_000")
    ap.add_argument("--rig-assert", action="store_true",
                    help="Enable Plan 1.5.B rig direction assertion per clip "
                         "(also enabled by SPEAR_RIG_ASSERT=1 env var).")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    csv_path = out_dir / "profile_per_clip.csv"
    render_apartment(Path(args.spec), out_dir, csv_path, args.clip_id)


if __name__ == "__main__":
    main()
