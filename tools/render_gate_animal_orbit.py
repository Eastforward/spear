"""Render a cooked ``gate_<tag>`` animal from every side in SPEAR.

This is the repository-owned replacement for the historical
``/tmp/orbit_animal.py`` gate helper.  It keeps the old rendering contract,
but resolves imports and the packaged executable from this SPEAR checkout and
lets callers select Walking or Idle explicitly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import spear


SPEAR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPEAR_ROOT / "examples"))

from render_in_apartment import clean_frames, read_frame, spawn_camera  # noqa: E402
from render_in_gpurir_room import (  # noqa: E402
    FLOOR_MATERIAL,
    configure_gpurir_instance,
    spawn_directional_light,
    spawn_point_light,
    spawn_room_piece,
    spawn_sky,
)


class _GpuSampler:
    """Best-effort whole-GPU telemetry without adding Python dependencies."""

    def __init__(self, interval_seconds=0.25):
        self.interval_seconds = float(interval_seconds)
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _sample(self):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except (OSError, subprocess.SubprocessError):
            return
        timestamp = time.time()
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                continue
            try:
                self.samples.append(
                    {
                        "timestamp": timestamp,
                        "gpu_index": int(fields[0]),
                        "memory_used_mib": int(fields[1]),
                        "utilization_pct": int(fields[2]),
                    }
                )
            except ValueError:
                continue

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self):
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=3.0)
        self._sample()
        self._thread = None

    def summary(self):
        by_gpu = {}
        for sample in self.samples:
            entry = by_gpu.setdefault(
                str(sample["gpu_index"]),
                {"peak_memory_used_mib": 0, "peak_utilization_pct": 0, "samples": 0},
            )
            entry["peak_memory_used_mib"] = max(
                entry["peak_memory_used_mib"], sample["memory_used_mib"]
            )
            entry["peak_utilization_pct"] = max(
                entry["peak_utilization_pct"], sample["utilization_pct"]
            )
            entry["samples"] += 1
        return {"sample_interval_seconds": self.interval_seconds, "by_gpu": by_gpu}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--action", choices=("Walking", "Idle"), default="Walking")
    parser.add_argument("--n-frames", type=int, default=72)
    parser.add_argument("--per-frame-warmup", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=60)
    parser.add_argument("--rpc-port", type=int, default=39002)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--scale", type=float, default=0.15)
    parser.add_argument("--radius-cm", type=float, default=160.0)
    parser.add_argument("--camera-height-cm", type=float, default=50.0)
    parser.add_argument("--look-height-cm", type=float, default=28.0)
    parser.add_argument("--full-turns", type=float, default=1.0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    if args.n_frames < 2:
        parser.error("--n-frames must be at least 2")
    if args.scale <= 0.0 or args.radius_cm <= 0.0:
        parser.error("--scale and --radius-cm must be positive")
    return args


def _dict_components(value, names):
    current = value
    expected = [name.lower() for name in names]
    for _ in range(3):
        if not isinstance(current, dict):
            break
        lowered = {str(key).lower(): item for key, item in current.items()}
        if all(name in lowered for name in expected):
            return [float(lowered[name]) for name in expected]
        if "returnvalue" in lowered and isinstance(lowered["returnvalue"], dict):
            current = lowered["returnvalue"]
            continue
        if len(current) == 1 and isinstance(next(iter(current.values())), dict):
            current = next(iter(current.values()))
            continue
        break
    raise KeyError(f"expected Unreal components {expected}, got {current!r}")


def _destroy_default_actors(game):
    for cls in (
        "APlayerStart",
        "ADefaultPawn",
        "ASpectatorPawn",
        "AStaticMeshActor",
        "ASkeletalMeshActor",
        "ABrush",
        "ADecalActor",
        "AInstancedFoliageActor",
    ):
        try:
            actors = game.unreal_service.find_actors_by_class(uclass=cls)
        except Exception:
            actors = []
        for actor in actors:
            try:
                game.unreal_service.destroy_actor(actor=actor)
            except Exception:
                pass


def run(args):
    total_start = time.perf_counter()
    gpu_sampler = _GpuSampler()
    gpu_sampler.start()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_frames(str(output_dir))

    bp_path = (
        f"/Game/MyAssets/Audioset/Blueprints/gate_{args.tag}/"
        f"BP_gate_{args.tag}.BP_gate_{args.tag}_C"
    )
    anim_path = f"/Game/MyAssets/Audioset/Meshes/gate_{args.tag}/{args.action}"
    instance = configure_gpurir_instance(rpc_port=args.rpc_port)
    launch_seconds = time.perf_counter() - total_start
    game = instance.get_game()
    manifest = {
        "tag": args.tag,
        "action": args.action,
        "blueprint": bp_path,
        "animation": anim_path,
        "scale": float(args.scale),
        "frame_count": int(args.n_frames),
        "width": int(args.width),
        "height": int(args.height),
        "radius_cm": float(args.radius_cm),
        "timings": {"launch_seconds": launch_seconds},
        "status": "running",
    }
    try:
        with instance.begin_frame():
            _destroy_default_actors(game)
            spawn_sky(game=game)
            # A visible floor is required for judging foot contact.  The cube
            # is 10 cm thick and its top surface is exactly world Z=0.
            spawn_room_piece(
                game=game,
                piece={
                    "name": "review_floor",
                    "location_cm": (0.0, 0.0, -5.0),
                    "scale": (6.0, 6.0, 0.1),
                },
                material_path=FLOOR_MATERIAL,
                cast_shadow=True,
            )
            spawn_directional_light(
                game=game,
                yaw_deg=-30.0,
                pitch_deg=-45.0,
                intensity_lux=8.0,
            )
            spawn_point_light(
                game=game,
                x_cm=0.0,
                y_cm=0.0,
                z_cm=300.0,
                intensity_lumens=4000.0,
                attenuation_cm=800.0,
            )
            camera, capture = spawn_camera(
                game=game,
                width=args.width,
                height=args.height,
            )
            bp = game.unreal_service.load_class(uclass="AActor", name=bp_path)
            actor = game.unreal_service.spawn_actor(
                uclass=bp,
                location={"X": 0.0, "Y": 0.0, "Z": 0.0},
                spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
            )
            actor.SetActorScale3D(
                NewScale3D={"X": args.scale, "Y": args.scale, "Z": args.scale}
            )
            actor.SetActorTickEnabled(bEnabled=True)
            skeletal_mesh = game.unreal_service.get_component_by_class(
                actor=actor,
                uclass="USkeletalMeshComponent",
            )
            skeletal_mesh.SetComponentTickEnabled(bEnabled=True)
            animation = game.unreal_service.load_object(
                uclass="UAnimationAsset",
                name=anim_path,
            )
            skeletal_mesh.PlayAnimation(NewAnimToPlay=animation, bLooping=True)
            game.get_unreal_object(uclass="UGameplayStatics").SetGamePaused(
                bPaused=False
            )
        with instance.end_frame():
            pass

        instance.step(num_frames=args.warmup)
        # Both calls execute on Unreal's game thread, so SPEAR requires them to
        # live inside one explicit frame transaction.
        with instance.begin_frame():
            bounds = actor.GetActorBounds(
                bOnlyCollidingComponents=False,
                as_dict=True,
            )
            origin = _dict_components(bounds["Origin"], ("x", "y", "z"))
            extent = _dict_components(bounds["BoxExtent"], ("x", "y", "z"))
            bottom_cm = origin[2] - extent[2]
            correction_cm = -bottom_cm
            actor.K2_SetActorLocationAndRotation(
                NewLocation={"X": 0.0, "Y": 0.0, "Z": correction_cm},
                NewRotation={"Roll": 0.0, "Pitch": 0.0, "Yaw": 0.0},
                bSweep=False,
                bTeleport=True,
            )
        with instance.end_frame():
            pass
        manifest["ground_snap"] = {
            "bounds_origin_before_cm": origin,
            "bounds_extent_before_cm": extent,
            "bottom_before_cm": bottom_cm,
            "z_correction_cm": correction_cm,
        }

        frame_seconds = []
        render_start = time.perf_counter()
        for index in range(args.n_frames):
            frame_start = time.perf_counter()
            theta = 2.0 * math.pi * args.full_turns * index / args.n_frames
            camera_x = args.radius_cm * math.cos(theta)
            camera_y = args.radius_cm * math.sin(theta)
            camera_z = args.camera_height_cm
            yaw = math.degrees(math.atan2(-camera_y, -camera_x))
            pitch = -math.degrees(
                math.atan2(
                    camera_z - args.look_height_cm,
                    math.hypot(camera_x, camera_y),
                )
            )
            instance.step(num_frames=args.per_frame_warmup)
            with instance.begin_frame():
                camera.K2_SetActorLocationAndRotation(
                    NewLocation={"X": camera_x, "Y": camera_y, "Z": camera_z},
                    NewRotation={"Roll": 0.0, "Pitch": pitch, "Yaw": yaw},
                    bSweep=False,
                    bTeleport=True,
                )
            with instance.end_frame():
                image = read_frame(capture)
                cv2.imwrite(str(output_dir / f"frame_{index:04d}.png"), image)
            frame_seconds.append(time.perf_counter() - frame_start)

        render_seconds = time.perf_counter() - render_start
        ordered_frame_seconds = sorted(frame_seconds)
        percentile_index = min(
            len(ordered_frame_seconds) - 1,
            math.ceil(0.95 * len(ordered_frame_seconds)) - 1,
        )
        manifest["timings"].update(
            {
                "spawn_warmup_and_ground_seconds": render_start
                - total_start
                - launch_seconds,
                "frame_loop_seconds": render_seconds,
                "seconds_per_frame_mean": sum(frame_seconds) / len(frame_seconds),
                "seconds_per_frame_p95": ordered_frame_seconds[percentile_index],
                "captured_frames_per_second": len(frame_seconds) / render_seconds,
                "total_before_close_seconds": time.perf_counter() - total_start,
            }
        )
        gpu_sampler.stop()
        manifest["gpu_telemetry"] = gpu_sampler.summary()
        manifest["status"] = "passed"
        (output_dir / "runtime_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"GATE_ANIMAL_ORBIT_OK {output_dir}", flush=True)
    finally:
        gpu_sampler.stop()
        instance.close(force=True)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
