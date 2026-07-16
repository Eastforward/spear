"""Render imported Hunyuan3D assets directly inside SPEAR apartment_0000.

Modes:
  reference  - capture one living-room frame for visual diagnosis.
  turntable  - place one imported Blueprint in the apartment and capture 360 deg.

Use spear-env:
  DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    /data/jzy/miniconda3/envs/spear-env/bin/python examples/render_in_apartment.py --mode reference

  DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    /data/jzy/miniconda3/envs/spear-env/bin/python examples/render_in_apartment.py
"""

import argparse
import glob
import json
import math
import os
import subprocess


# SpearSim.sh lives inside this SPEAR checkout — resolve relative to this file
# to survive being moved (e.g. into AVEngine/external/SPEAR/).
_SPEAR_ROOT_FOR_EXEC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXECUTABLE = os.path.join(_SPEAR_ROOT_FOR_EXEC, "cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh")
APARTMENT_MAP = "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"
CAMERA_BP = "/SpContent/Blueprints/BP_CameraSensor.BP_CameraSensor_C"
DEFAULT_ASSET_BP = "/Game/MyAssets/Blueprints/BP_Clock.BP_Clock_C"
DEFAULT_NAME = "Clock"
DEFAULT_TMP_ROOT = "/data/jzy/code/SPEAR/tmp"
DEFAULT_META_DIR = "/data/jzy/code/SPEAR/tmp/asset_meta"
DEFAULT_REFERENCE_OUTPUT = "/data/jzy/code/SPEAR/tmp/apartment_reference/frame_0000.png"

SUPPORTED_ANIMALS = ("cat", "dog", "goose", "yak")
ANIMAL_BP_TEMPLATE = "/Game/MyAssets/Audioset/Blueprints/{name}/BP_{name}.BP_{name}_C"
ANIMALS_OUTPUT_SUBDIR = "render_animals_apartment"


def animal_bp_path(name):
    if name not in SUPPORTED_ANIMALS:
        raise ValueError(
            f"Unsupported animal {name!r}; supported = {SUPPORTED_ANIMALS}"
        )
    return ANIMAL_BP_TEMPLATE.format(name=name)


def animal_meta_path(meta_dir, name):
    return os.path.join(meta_dir, f"{name}.json")

REMOVABLE_PREFIXES = (
    "Meshes/05_chair",
    "Meshes/06_sofa",
    "Meshes/07_table",
    "Meshes/10_bookshelf",
    "Meshes/18_pillow",
    "Meshes/35_lamp",
)

REFERENCE_LOCATION = {"X": 300.0, "Y": 0.0, "Z": 200.0}
REFERENCE_ROTATION = {"Roll": 0.0, "Pitch": -25.0, "Yaw": 180.0}


def build_output_dir(output_root, name):
    return os.path.join(output_root, f"render_apartment_{name}")


def compute_asset_fit(meta, target_cm, floor_z):
    ext = float(meta["ext"])
    bmin_z = float(meta["bmin_z"])
    height = float(meta["height"])
    scale = float(target_cm) / max(ext, 1e-3)
    actor_z = float(floor_z) - bmin_z * scale
    center_z = float(floor_z) + (height * scale) / 2.0
    return {"scale": scale, "actor_z": actor_z, "center_z": center_z}


def compute_bounds_lift(bounds_bottom_z, ground_z, clearance_cm, tolerance_cm):
    target_bottom_z = float(ground_z) + float(clearance_cm)
    penetration_cm = target_bottom_z - float(bounds_bottom_z)
    if penetration_cm > float(tolerance_cm):
        return penetration_cm
    return 0.0


def compute_orbit_pose(
    frame_index,
    total_frames,
    center_x,
    center_y,
    center_z,
    target_cm,
    r_factor,
    max_radius_cm,
    cam_z_offset_cm,
):
    radius = min(float(r_factor) * float(target_cm), float(max_radius_cm))
    theta = 2.0 * math.pi * int(frame_index) / int(total_frames)
    cam_x = float(center_x) + radius * math.cos(theta)
    cam_y = float(center_y) + radius * math.sin(theta)
    cam_z = float(center_z) + float(cam_z_offset_cm)
    yaw = math.degrees(math.atan2(float(center_y) - cam_y, float(center_x) - cam_x))
    pitch = -math.degrees(math.atan2(float(cam_z_offset_cm), radius))
    return {
        "location": {"X": cam_x, "Y": cam_y, "Z": cam_z},
        "rotation": {"Roll": 0.0, "Pitch": pitch, "Yaw": yaw},
        "radius": radius,
    }


def compute_lineup_positions(
    *,
    animals,
    metas,
    target_cm,
    gap_cm,
    center_x,
    center_y,
):
    names = list(animals)
    for name in names:
        if name not in metas:
            raise KeyError(f"Missing meta for animal {name!r}")
    half_extent = float(target_cm) / 2.0
    spacing = float(target_cm) + float(gap_cm)
    n = len(names)
    start_offset = -spacing * (n - 1) / 2.0
    positions = []
    for i, name in enumerate(names):
        positions.append(
            {
                "name": name,
                "x": float(center_x) + start_offset + i * spacing,
                "y": float(center_y),
                "half_extent_cm": half_extent,
            }
        )
    return positions


def compute_group_orbit_radius(
    *,
    positions,
    target_cm,
    base_r_factor,
    max_radius_cm,
):
    if not positions:
        return min(float(base_r_factor) * float(target_cm), float(max_radius_cm))
    xs = [float(p["x"]) for p in positions]
    half_span = (max(xs) - min(xs)) / 2.0
    ideal = float(base_r_factor) * float(target_cm) + half_span
    return min(ideal, float(max_radius_cm))


def write_group_layout(output_dir, positions, radius_cm, center_x, center_y):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    theta = [2.0 * math.pi * i / 128 for i in range(129)]
    ax.plot(
        [float(center_x) + float(radius_cm) * math.cos(t) for t in theta],
        [float(center_y) + float(radius_cm) * math.sin(t) for t in theta],
        linestyle="--",
        color="tab:blue",
        label=f"camera orbit r={float(radius_cm):.0f}cm",
    )
    for pos in positions:
        ax.add_patch(
            plt.Circle(
                (float(pos["x"]), float(pos["y"])),
                float(pos["half_extent_cm"]),
                fill=True,
                alpha=0.4,
                color="tab:orange",
            )
        )
        ax.text(
            float(pos["x"]),
            float(pos["y"]),
            pos["name"],
            ha="center",
            va="center",
            fontsize=9,
        )
    ax.plot([float(center_x)], [float(center_y)], marker="+", color="k", label="orbit center")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (cm, UE world)")
    ax.set_ylabel("Y (cm, UE world)")
    ax.set_title("apartment_0000 line-up layout (top-down)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "layout.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def should_remove_actor(actor_name, prefixes=REMOVABLE_PREFIXES):
    stable_name = str(actor_name or "").split(":", maxsplit=1)[0]
    return any(stable_name.startswith(prefix) for prefix in prefixes)


def find_meta_path(meta_dir, name):
    candidates = [
        os.path.join(meta_dir, f"{name}.json"),
        os.path.join(meta_dir, f"{name.lower()}.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No asset meta found for {name!r} in {meta_dir}")


def clean_frames(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for path in glob.glob(os.path.join(output_dir, "frame_*.png")):
        os.remove(path)


def parallel_instance_settings(rpc_port, graphics_adapter=None):
    """Return collision-free SPEAR/UE process settings for one render worker."""
    rpc_port = int(rpc_port)
    if not 1024 <= rpc_port <= 65535:
        raise ValueError(f"rpc_port must be in [1024, 65535], got {rpc_port}")
    if graphics_adapter is not None:
        graphics_adapter = int(graphics_adapter)
        if graphics_adapter < 0:
            raise ValueError(
                f"graphics_adapter must be non-negative, got {graphics_adapter}"
            )
    return {
        "rpc_port": rpc_port,
        "graphics_adapter": graphics_adapter,
        "temp_dir": f"tmp/spear_instance_{rpc_port}",
        "log": f"SpearSim_rpc_{rpc_port}.log",
        "shared_memory_initial_unique_id": rpc_port * 10000,
    }


def configure_instance(rpc_port, fixed_delta_time=None):
    import spear

    graphics_adapter_env = os.environ.get("SPEAR_GRAPHICS_ADAPTER")
    settings = parallel_instance_settings(
        rpc_port,
        graphics_adapter=(
            int(graphics_adapter_env) if graphics_adapter_env not in (None, "") else None
        ),
    )
    config = spear.get_config(user_config_files=[])
    config.defrost()
    config.SPEAR.LAUNCH_MODE = "game"
    config.SPEAR.INSTANCE.GAME_EXECUTABLE = EXECUTABLE
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP = True
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = APARTMENT_MAP
    config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT = settings["rpc_port"]
    config.SPEAR.INSTANCE.TEMP_DIR = settings["temp_dir"]
    config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.log = settings["log"]
    config.SP_CORE.SHARED_MEMORY_INITIAL_UNIQUE_ID = settings[
        "shared_memory_initial_unique_id"
    ]
    if settings["graphics_adapter"] is not None:
        config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.graphicsadapter = settings[
            "graphics_adapter"
        ]
    if os.environ.get("SPEAR_RENDER_OFFSCREEN", "0") == "1":
        config.SPEAR.INSTANCE.COMMAND_LINE_ARGS.renderoffscreen = None
    if fixed_delta_time is not None:
        config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.FIXED_DELTA_TIME = float(fixed_delta_time)
    config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES = "/etc/vulkan/icd.d/nvidia_icd.json"
    config.freeze()
    spear.configure_system(config=config)
    return spear.Instance(config=config)


def spawn_camera(game, width, height):
    cam_uclass = game.unreal_service.load_class(uclass="AActor", name=CAMERA_BP)
    cam = game.unreal_service.spawn_actor(uclass=cam_uclass)
    comp = game.unreal_service.get_component_by_name(
        actor=cam,
        component_name="DefaultSceneRoot.final_tone_curve_hdr_",
        uclass="USpSceneCaptureComponent2D",
    )
    viewport_desc = game.rendering_service.get_current_viewport_desc()
    game.rendering_service.align_camera_with_viewport(
        camera_sensor=cam,
        camera_components=[comp],
        viewport_desc=viewport_desc,
        widths=int(width),
        heights=int(height),
    )
    comp.Initialize()
    comp.initialize_sp_funcs()
    try:
        disable_level_model_component_shadows(game=game)
    except Exception as e:
        print(f"[spawn_camera] warn: could not disable level model shadows ({e})", flush=True)
    # /Engine/Maps/Entry ships with a leftover starter Brush whose BSP geometry
    # is baked into ULevel::ModelComponents. It survives destroy_actor(ABrush)
    # because UModelComponent is owned by the level (not any AActor), so
    # find_actors_by_class can never reach it. The cooked Entry.umap has no
    # material bound on that BSP, so UE renders it with WorldGridMaterial —
    # the checkerboard "mystery cube" that keeps appearing near the east wall
    # inside corner of every render_in_gpurir_room shot. The SceneCapture show
    # flag hides it from the main pass, but those model components can still
    # cast shadows, so disable their shadow contribution separately. Use the
    # UFUNCTION setter, not a raw UPROPERTY write, so UE rebuilds the component's
    # internal FEngineShowFlags via UpdateShowFlags().
    try:
        comp.SetShowFlagSettings(
            InShowFlagSettings=[
                {"ShowFlagName": "BSP", "Enabled": False},
                {"ShowFlagName": "BSPTriangles", "Enabled": False},
            ],
        )
    except Exception as e:
        print(f"[spawn_camera] warn: could not disable BSP show flag ({e})", flush=True)
    return cam, comp


def disable_level_model_component_shadows(game):
    world = game.get_unreal_object(uobject=game.get_world(), with_sp_funcs=False)
    raw_level = world.get_property_value("PersistentLevel", as_raw_value=True)
    if not raw_level.value:
        return 0
    level = game.get_unreal_object(uobject=int(raw_level.value, 16), with_sp_funcs=False)
    raw_components = level.get_property_value("ModelComponents", as_raw_value=True)
    if not isinstance(raw_components.value, list):
        return 0
    count = 0
    for handle in raw_components.value:
        if not handle:
            continue
        model_component = game.get_unreal_object(
            uobject=int(handle, 16),
            with_sp_funcs=False,
        )
        model_component.SetCastShadow(NewCastShadow=False)
        model_component.SetCastHiddenShadow(NewCastHiddenShadow=False)
        model_component.bCastDynamicShadow = False
        model_component.bCastStaticShadow = False
        count += 1
    return count


def read_frame(component):
    bundle = component.read_pixels()
    return bundle["arrays"]["data"][:, :, [0, 1, 2]]


def get_actor_stable_name(game, actor):
    try:
        return game.unreal_service.get_stable_name_for_actor(
            actor=actor, include_unreal_name=True
        )
    except Exception:
        return ""


def clear_removable_furniture(game):
    actors = game.unreal_service.find_actors_by_class(uclass="AStaticMeshActor")
    removed = []
    for actor in actors:
        name = get_actor_stable_name(game=game, actor=actor)
        if should_remove_actor(name):
            game.unreal_service.destroy_actor(actor=actor)
            removed.append(name)
    return removed


def spawn_fill_light(game, x, y, z, intensity_lumens):
    if float(intensity_lumens) <= 0.0:
        return None
    light = game.unreal_service.spawn_actor(
        uclass="APointLight",
        location={"X": float(x), "Y": float(y), "Z": float(z)},
    )
    root = light.K2_GetRootComponent()
    root.SetMobility(NewMobility="Movable")
    comp = game.unreal_service.get_component_by_class(
        actor=light, uclass="UPointLightComponent"
    )
    comp.SetIntensity(NewIntensity=float(intensity_lumens))
    comp.SetAttenuationRadius(NewRadius=350.0)
    return light


def vector_z(value):
    return float(value.get("z", value.get("Z")))


def sample_ground_z(game, x, y, fallback_z, trace_start_z, trace_end_z):
    kismet = game.get_unreal_object(uclass="UKismetSystemLibrary")
    result = kismet.LineTraceSingleByProfile(
        Start={"X": float(x), "Y": float(y), "Z": float(trace_start_z)},
        End={"X": float(x), "Y": float(y), "Z": float(trace_end_z)},
        ProfileName="BlockAll",
        bTraceComplex=True,
        ActorsToIgnore=[],
        DrawDebugType="None",
        bIgnoreSelf=True,
        TraceColor={"R": 1.0, "G": 0.0, "B": 0.0, "A": 1.0},
        TraceHitColor={"R": 0.0, "G": 1.0, "B": 0.0, "A": 1.0},
        DrawTime=0.0,
        as_dict=True,
    )
    if result.get("ReturnValue"):
        return vector_z(result["OutHit"]["location"]), result["OutHit"]
    return float(fallback_z), None


def get_actor_bounds_bottom_z(actor):
    bounds = actor.GetActorBounds(bOnlyCollidingComponents=False, as_dict=True)
    origin = bounds["Origin"]
    extent = bounds["BoxExtent"]
    return vector_z(origin) - vector_z(extent), bounds


def build_solo_checklist(
    *,
    name,
    ground_z,
    bounds_bottom_z,
    lift_cm,
    penetration_after_lift,
    scale,
    target_cm,
    radius,
    frames,
    clearance_cm,
    tolerance_cm,
):
    return {
        "name": name,
        "frames": int(frames),
        "target_cm": float(target_cm),
        "scale": float(scale),
        "radius_cm": float(radius),
        "ground_z_cm": float(ground_z),
        "bounds_bottom_z_cm": float(bounds_bottom_z),
        "lift_applied_cm": float(lift_cm),
        "penetration_after_lift_cm": float(penetration_after_lift),
        "clearance_cm": float(clearance_cm),
        "tolerance_cm": float(tolerance_cm),
        "ground_ok": abs(float(penetration_after_lift)) <= float(tolerance_cm),
    }


def write_checklist(output_dir, checklist):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "checklist.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checklist, f, indent=2, sort_keys=True)
    return path


def render_reference(args):
    import cv2

    os.makedirs(os.path.dirname(args.reference_output), exist_ok=True)
    instance = configure_instance(rpc_port=args.rpc_port)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            cam, comp = spawn_camera(game=game, width=args.width, height=args.height)
            cam.K2_SetActorLocationAndRotation(
                NewLocation=REFERENCE_LOCATION,
                NewRotation=REFERENCE_ROTATION,
                bSweep=False,
                bTeleport=True,
            )
        with instance.end_frame():
            pass

        instance.step(num_frames=args.warmup_frames)
        with instance.begin_frame():
            pass
        with instance.end_frame():
            cv2.imwrite(args.reference_output, read_frame(component=comp))
        print(f"REFERENCE_DONE {args.reference_output}", flush=True)
    finally:
        instance.close(force=True)


def render_turntable(args):
    import cv2

    output_dir = build_output_dir(args.output_root, args.name)
    clean_frames(output_dir)
    meta_path = find_meta_path(meta_dir=args.meta_dir, name=args.name)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    instance = configure_instance(rpc_port=args.rpc_port)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            removed = clear_removable_furniture(game=game) if args.clear_furniture else []
            ground_z, ground_hit = sample_ground_z(
                game=game,
                x=args.spawn_x,
                y=args.spawn_y,
                fallback_z=args.floor_z,
                trace_start_z=args.ground_trace_start_z,
                trace_end_z=args.ground_trace_end_z,
            )
            placement_floor_z = ground_z + args.ground_clearance_cm
            fit = compute_asset_fit(
                meta=meta, target_cm=args.target_cm, floor_z=placement_floor_z
            )

            bp_asset = game.unreal_service.load_class(uclass="AActor", name=args.asset_bp)
            asset = game.unreal_service.spawn_actor(
                uclass=bp_asset,
                location={"X": args.spawn_x, "Y": args.spawn_y, "Z": args.floor_z},
            )
            try:
                asset.K2_GetRootComponent().SetMobility(NewMobility="Movable")
            except Exception:
                pass
            game.unreal_service.set_stable_name_for_actor(
                actor=asset, stable_name=f"MyAssets/{args.name}"
            )

            cam, comp = spawn_camera(game=game, width=args.width, height=args.height)
            spawn_fill_light(
                game=game,
                x=args.spawn_x,
                y=args.spawn_y - 90.0,
                z=fit["center_z"] + 140.0,
                intensity_lumens=args.fill_light_lumens,
            )
        with instance.end_frame():
            pass

        instance.step(num_frames=4)
        with instance.begin_frame():
            asset.SetActorScale3D(
                NewScale3D={"X": fit["scale"], "Y": fit["scale"], "Z": fit["scale"]}
            )
            asset.K2_SetActorLocation(
                NewLocation={"X": args.spawn_x, "Y": args.spawn_y, "Z": fit["actor_z"]},
                bSweep=False,
                bTeleport=True,
            )
            bounds_bottom_z, bounds = get_actor_bounds_bottom_z(actor=asset)
            lift_cm = compute_bounds_lift(
                bounds_bottom_z=bounds_bottom_z,
                ground_z=ground_z,
                clearance_cm=args.ground_clearance_cm,
                tolerance_cm=args.ground_tolerance_cm,
            )
            if lift_cm > 0.0:
                fit["actor_z"] += lift_cm
                fit["center_z"] += lift_cm
                asset.K2_SetActorLocation(
                    NewLocation={
                        "X": args.spawn_x,
                        "Y": args.spawn_y,
                        "Z": fit["actor_z"],
                    },
                    bSweep=False,
                    bTeleport=True,
                )
        with instance.end_frame():
            pass

        center_x = float(args.spawn_x)
        center_y = float(args.spawn_y)
        center_z = float(fit["center_z"])
        hit_component = ""
        if ground_hit is not None:
            hit_component = str(ground_hit.get("component", ""))
        print(
            "[apartment] "
            f"removed_furniture={len(removed)} meta={meta_path} "
            f"ground_z={ground_z:.2f}cm clearance={args.ground_clearance_cm:.2f}cm "
            f"bounds_bottom_z={bounds_bottom_z:.2f}cm bounds_lift={lift_cm:.2f}cm "
            f"ground_component={hit_component} "
            f"scale={fit['scale']:.3f} target={args.target_cm:.1f}cm "
            f"center=({center_x:.1f},{center_y:.1f},{center_z:.1f})",
            flush=True,
        )

        instance.step(num_frames=args.warmup_frames)
        for i in range(args.frames):
            # First orbit frame is the largest camera jump (spawn pose ->
            # orbit start), so give VT streaming a full warmup budget on i=0
            # to match the sharpness of later frames.
            frame_warmup = args.warmup_frames if i == 0 else args.per_frame_warmup_frames
            pose = compute_orbit_pose(
                frame_index=i,
                total_frames=args.frames,
                center_x=center_x,
                center_y=center_y,
                center_z=center_z,
                target_cm=args.target_cm,
                r_factor=args.r_factor,
                max_radius_cm=args.max_radius_cm,
                cam_z_offset_cm=args.cam_z_offset_cm,
            )
            with instance.begin_frame():
                cam.K2_SetActorLocationAndRotation(
                    NewLocation=pose["location"],
                    NewRotation=pose["rotation"],
                    bSweep=False,
                    bTeleport=True,
                )
            with instance.end_frame():
                pass
            # Let virtual-texture streaming catch up with the new camera pose
            # before we sample pixels. Without this, the first ~half of the
            # orbit reads unstreamed low-mip floor/wall tiles ("blurry ground").
            if frame_warmup > 0:
                instance.step(num_frames=frame_warmup)
            with instance.begin_frame():
                pass
            with instance.end_frame():
                cv2.imwrite(os.path.join(output_dir, f"frame_{i:04d}.png"), read_frame(comp))

        video_path = os.path.join(output_dir, "turntable.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(args.framerate),
                "-i",
                os.path.join(output_dir, "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "23",
                video_path,
            ],
            check=True,
            capture_output=True,
        )
        print(f"VIDEO_DONE {video_path}", flush=True)

        with instance.begin_frame():
            final_bounds_bottom_z, _ = get_actor_bounds_bottom_z(actor=asset)
        with instance.end_frame():
            pass
        penetration_after_lift = (ground_z + args.ground_clearance_cm) - final_bounds_bottom_z
        checklist = build_solo_checklist(
            name=args.name,
            ground_z=ground_z,
            bounds_bottom_z=final_bounds_bottom_z,
            lift_cm=lift_cm,
            penetration_after_lift=penetration_after_lift,
            scale=fit["scale"],
            target_cm=args.target_cm,
            radius=min(args.r_factor * args.target_cm, args.max_radius_cm),
            frames=args.frames,
            clearance_cm=args.ground_clearance_cm,
            tolerance_cm=args.ground_tolerance_cm,
        )
        checklist_path = write_checklist(output_dir, checklist)
        print(f"CHECKLIST_DONE {checklist_path}", flush=True)
    finally:
        instance.close(force=True)


def render_group(args):
    import cv2

    output_dir = os.path.join(args.output_root, ANIMALS_OUTPUT_SUBDIR, args.group_name)
    clean_frames(output_dir)

    metas = {}
    for name in args.animals:
        with open(animal_meta_path(args.meta_dir, name), "r", encoding="utf-8") as f:
            metas[name] = json.load(f)

    positions = compute_lineup_positions(
        animals=args.animals,
        metas=metas,
        target_cm=args.target_cm,
        gap_cm=args.gap_cm,
        center_x=args.spawn_x,
        center_y=args.spawn_y,
    )
    radius = compute_group_orbit_radius(
        positions=positions,
        target_cm=args.target_cm,
        base_r_factor=args.r_factor,
        max_radius_cm=args.max_radius_cm,
    )

    instance = configure_instance(rpc_port=args.rpc_port)
    game = instance.get_game()
    spawned_records = []
    try:
        with instance.begin_frame():
            removed = clear_removable_furniture(game=game) if args.clear_furniture else []
        with instance.end_frame():
            pass

        with instance.begin_frame():
            ground_z_by_name = {}
            for pos in positions:
                ground_z, _ = sample_ground_z(
                    game=game,
                    x=pos["x"],
                    y=pos["y"],
                    fallback_z=args.floor_z,
                    trace_start_z=args.ground_trace_start_z,
                    trace_end_z=args.ground_trace_end_z,
                )
                ground_z_by_name[pos["name"]] = ground_z
                fit = compute_asset_fit(
                    meta=metas[pos["name"]],
                    target_cm=args.target_cm,
                    floor_z=ground_z + args.ground_clearance_cm,
                )
                bp = game.unreal_service.load_class(
                    uclass="AActor", name=animal_bp_path(pos["name"])
                )
                actor = game.unreal_service.spawn_actor(
                    uclass=bp,
                    location={"X": pos["x"], "Y": pos["y"], "Z": args.floor_z},
                )
                try:
                    actor.K2_GetRootComponent().SetMobility(NewMobility="Movable")
                except Exception:
                    pass
                game.unreal_service.set_stable_name_for_actor(
                    actor=actor, stable_name=f"MyAssets/group/{pos['name']}"
                )
                spawned_records.append({"pos": pos, "actor": actor, "fit": fit})

            cam, comp = spawn_camera(game=game, width=args.width, height=args.height)
            group_center_z = max(rec["fit"]["center_z"] for rec in spawned_records)
            spawn_fill_light(
                game=game,
                x=args.spawn_x,
                y=args.spawn_y - 90.0,
                z=group_center_z + 140.0,
                intensity_lumens=args.fill_light_lumens,
            )
        with instance.end_frame():
            pass

        instance.step(num_frames=4)
        with instance.begin_frame():
            for rec in spawned_records:
                fit = rec["fit"]
                pos = rec["pos"]
                actor = rec["actor"]
                actor.SetActorScale3D(
                    NewScale3D={"X": fit["scale"], "Y": fit["scale"], "Z": fit["scale"]}
                )
                actor.K2_SetActorLocation(
                    NewLocation={"X": pos["x"], "Y": pos["y"], "Z": fit["actor_z"]},
                    bSweep=False,
                    bTeleport=True,
                )
                bounds_bottom_z, _ = get_actor_bounds_bottom_z(actor=actor)
                lift_cm = compute_bounds_lift(
                    bounds_bottom_z=bounds_bottom_z,
                    ground_z=ground_z_by_name[pos["name"]],
                    clearance_cm=args.ground_clearance_cm,
                    tolerance_cm=args.ground_tolerance_cm,
                )
                if lift_cm > 0.0:
                    fit["actor_z"] += lift_cm
                    fit["center_z"] += lift_cm
                    actor.K2_SetActorLocation(
                        NewLocation={"X": pos["x"], "Y": pos["y"], "Z": fit["actor_z"]},
                        bSweep=False,
                        bTeleport=True,
                    )
                rec["bounds_bottom_z"] = bounds_bottom_z
                rec["lift_cm"] = lift_cm
        with instance.end_frame():
            pass

        center_x = float(args.spawn_x)
        center_y = float(args.spawn_y)
        center_z = max(rec["fit"]["center_z"] for rec in spawned_records)

        print(
            "[apartment-group] "
            f"animals={args.animals} radius={radius:.1f}cm "
            f"center=({center_x:.1f},{center_y:.1f},{center_z:.1f})",
            flush=True,
        )

        instance.step(num_frames=args.warmup_frames)
        for i in range(args.frames):
            frame_warmup = args.warmup_frames if i == 0 else args.per_frame_warmup_frames
            theta = 2.0 * math.pi * i / args.frames
            cam_x = center_x + radius * math.cos(theta)
            cam_y = center_y + radius * math.sin(theta)
            cam_z = center_z + args.cam_z_offset_cm
            yaw = math.degrees(math.atan2(center_y - cam_y, center_x - cam_x))
            pitch = -math.degrees(math.atan2(args.cam_z_offset_cm, radius))
            with instance.begin_frame():
                cam.K2_SetActorLocationAndRotation(
                    NewLocation={"X": cam_x, "Y": cam_y, "Z": cam_z},
                    NewRotation={"Roll": 0.0, "Pitch": pitch, "Yaw": yaw},
                    bSweep=False,
                    bTeleport=True,
                )
            with instance.end_frame():
                pass
            if frame_warmup > 0:
                instance.step(num_frames=frame_warmup)
            with instance.begin_frame():
                pass
            with instance.end_frame():
                cv2.imwrite(
                    os.path.join(output_dir, f"frame_{i:04d}.png"),
                    read_frame(comp),
                )

        video_path = os.path.join(output_dir, "turntable.mp4")
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(args.framerate),
                "-i", os.path.join(output_dir, "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                video_path,
            ],
            check=True,
            capture_output=True,
        )
        print(f"VIDEO_DONE {video_path}", flush=True)

        layout_path = write_group_layout(output_dir, positions, radius, center_x, center_y)
        print(f"LAYOUT_DONE {layout_path}", flush=True)

        checklist = {
            "name": args.group_name,
            "animals": list(args.animals),
            "gap_cm": float(args.gap_cm),
            "target_cm": float(args.target_cm),
            "radius_cm": float(radius),
            "center": {"x": center_x, "y": center_y, "z": center_z},
            "per_animal": [
                {
                    "name": rec["pos"]["name"],
                    "x": rec["pos"]["x"],
                    "y": rec["pos"]["y"],
                    "scale": rec["fit"]["scale"],
                    "ground_z_cm": ground_z_by_name[rec["pos"]["name"]],
                    "bounds_bottom_z_cm": rec["bounds_bottom_z"],
                    "lift_applied_cm": rec["lift_cm"],
                }
                for rec in spawned_records
            ],
            "removed_furniture_count": len(removed),
        }
        checklist_path = write_checklist(output_dir, checklist)
        print(f"CHECKLIST_DONE {checklist_path}", flush=True)
    finally:
        instance.close(force=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("reference", "turntable", "group"),
        default="turntable",
    )
    parser.add_argument("--asset-bp", default=DEFAULT_ASSET_BP)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--target-cm", type=float, default=80.0)
    parser.add_argument("--spawn-x", type=float, default=-120.0)
    parser.add_argument("--spawn-y", type=float, default=80.0)
    parser.add_argument("--floor-z", type=float, default=0.0)
    parser.add_argument("--ground-clearance-cm", type=float, default=0.5)
    parser.add_argument("--ground-tolerance-cm", type=float, default=0.5)
    parser.add_argument("--ground-trace-start-z", type=float, default=300.0)
    parser.add_argument("--ground-trace-end-z", type=float, default=-200.0)
    parser.add_argument("--meta-dir", default=DEFAULT_META_DIR)
    parser.add_argument("--output-root", default=DEFAULT_TMP_ROOT)
    parser.add_argument("--reference-output", default=DEFAULT_REFERENCE_OUTPUT)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--framerate", type=int, default=12)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument(
        "--per-frame-warmup-frames",
        type=int,
        default=6,
        help=(
            "Idle steps to run after each camera move so virtual-texture "
            "streaming can catch up before pixels are sampled. Fixes the "
            "'blurry floor first, sharp later' artifact."
        ),
    )
    parser.add_argument("--r-factor", type=float, default=4.0)
    parser.add_argument("--max-radius-cm", type=float, default=130.0)
    parser.add_argument("--cam-z-offset-cm", type=float, default=40.0)
    parser.add_argument("--fill-light-lumens", type=float, default=350.0)
    parser.add_argument("--rpc-port", type=int, default=39002)
    parser.add_argument(
        "--clear-furniture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove chair/sofa/table/bookshelf actors at runtime before orbiting.",
    )
    def _animals_list(value):
        names = [n.strip() for n in str(value).split(",") if n.strip()]
        for n in names:
            if n not in SUPPORTED_ANIMALS:
                raise argparse.ArgumentTypeError(
                    f"Unsupported animal {n!r}; supported = {SUPPORTED_ANIMALS}"
                )
        return names

    parser.add_argument(
        "--animals",
        type=_animals_list,
        default=list(SUPPORTED_ANIMALS),
        help="Comma-separated animal names for --mode group (default: all four).",
    )
    parser.add_argument("--gap-cm", type=float, default=30.0)
    parser.add_argument("--group-name", default="group")
    parser.add_argument(
        "--animal",
        choices=SUPPORTED_ANIMALS,
        default=None,
        help=(
            "Shortcut for --asset-bp/--name when rendering an imported Hunyuan3D "
            "animal from AudioSet. Fills --asset-bp always; fills --name only if "
            "--name was not explicitly passed."
        ),
    )
    args = parser.parse_args(argv)
    if args.animal is not None:
        args.asset_bp = animal_bp_path(args.animal)
        if args.name == DEFAULT_NAME:
            args.name = args.animal
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.mode == "reference":
        render_reference(args)
    elif args.mode == "group":
        render_group(args)
    else:
        render_turntable(args)


if __name__ == "__main__":
    main()
