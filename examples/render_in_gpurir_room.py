"""Render an imported Hunyuan3D asset inside a GPURIR-parameter-aligned
shoebox room built from Cube meshes with apartment_0000 materials.

Use spear-env:
  DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    /data/jzy/miniconda3/envs/spear-env/bin/python examples/render_in_gpurir_room.py \
    --animal dog

See HANDOFF_GPURIR_ROOM.md for the full spec.
"""

import argparse
import json
import math
import os
import subprocess
import sys


# Allow importing sibling module render_in_apartment
_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from render_in_apartment import (  # noqa: E402
    animal_bp_path,
    animal_meta_path,
    build_solo_checklist,
    clean_frames,
    compute_asset_fit,
    compute_bounds_lift,
    get_actor_bounds_bottom_z,
    parallel_instance_settings,
    read_frame,
    sample_ground_z,
    spawn_camera,
    SUPPORTED_ANIMALS,
    write_checklist,
)


EMPTY_MAP = "/Engine/Maps/Entry"
LIGHT_STUDIO_BP = "/Engine/EngineSky/BP_LightStudio.BP_LightStudio_C"
CUBE_MESH = "/Engine/BasicShapes/Cube.Cube"
# 4m x 4m starter-content floor mesh; its UVs are authored for the 4m tile size,
# so wood-plank materials render at realistic plank width when we scale to
# room_x/4 x room_y/4. Using Cube.Cube instead stretched a single plank across
# the whole 5.2m room (visually wrong: room-wide "boards").
FLOOR_MESH = "/Game/StarterContent/Architecture/Floor_400x400.Floor_400x400"
FLOOR_MESH_TILE_M = 4.0                # the mesh's designed footprint (per side)
FLOOR_MESH_THICKNESS_CM = 20.0         # local bounds: z in [-20, 0]

# Floor-material pool. Each render picks one (default: dataset-random). The pool
# excludes MI_Floor (mesh-specific) and M_Basic_Floor (flat gray, unrealistic).
# Every entry is a plain wood material shipped in StarterContent, so cook is a
# no-op — the assets are already packaged.
FLOOR_MATERIAL_POOL = [
    "/Game/StarterContent/Materials/M_Wood_Oak.M_Wood_Oak",
    "/Game/StarterContent/Materials/M_Wood_Pine.M_Wood_Pine",
    "/Game/StarterContent/Materials/M_Wood_Walnut.M_Wood_Walnut",
    "/Game/StarterContent/Materials/M_Wood_Floor_Walnut_Polished.M_Wood_Floor_Walnut_Polished",
    "/Game/StarterContent/Materials/M_Wood_Floor_Walnut_Worn.M_Wood_Floor_Walnut_Worn",
]
# Kept for backward compat (build_room_checklist still records a floor_material
# field for humans reviewing the run). Not used as an actual render default.
FLOOR_MATERIAL = FLOOR_MATERIAL_POOL[0]

# Wall material pool. Only materials whose textures survive the 5.2 x 2.8 m Cube
# stretch without visible tiling artifacts are included. Verified by tools/
# probe_wall_materials.py -- concrete_grime / rock_slate / concrete_panels showed
# heavy horizontal smearing and were excluded.
WALL_MATERIAL_POOL = [
    "/Game/StarterContent/Materials/M_Basic_Wall.M_Basic_Wall",
    "/Game/StarterContent/Materials/M_Brick_Clay_New.M_Brick_Clay_New",
    "/Game/StarterContent/Materials/M_Brick_Clay_Beveled.M_Brick_Clay_Beveled",
    "/Game/StarterContent/Materials/M_Concrete_Poured.M_Concrete_Poured",
    "/Game/StarterContent/Materials/M_Wood_Pine.M_Wood_Pine",
]
WALL_MATERIAL = WALL_MATERIAL_POOL[0]  # kept for legacy callers/checklist field
GLASS_MATERIAL = (
    "/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/"
    "MI_Casement_Glass.MI_Casement_Glass"
)
WINDOW_FRAME_MATERIAL = (
    "/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Casement.MI_Casement"
)
WINDOW_FRAME_THICKNESS_M = 0.06  # 6cm dark trim around the window opening
GLASS_THICKNESS_M = 0.03  # 3cm — thick enough to catch reflection
OUTDOOR_GROUND_MATERIAL = (
    "/Game/StarterContent/Materials/M_Ground_Grass.M_Ground_Grass"
)
_SPEAR_ROOT_FOR_EXEC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEARSIM_EXECUTABLE = os.path.join(
    _SPEAR_ROOT_FOR_EXEC,
    "cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh",
)
DEFAULT_TMP_ROOT = os.path.join(_SPEAR_ROOT_FOR_EXEC, "tmp/render_gpurir_room")
DEFAULT_META_DIR = os.path.join(_SPEAR_ROOT_FOR_EXEC, "tmp/asset_meta")


M2CM = 100.0
WALL_THICKNESS_M = 0.1
# Small vertical overlap so walls sink INTO the floor and stick UP INTO the ceiling
# by this amount at both ends. Turns the wall/floor and wall/ceiling contact from
# a coplanar T-junction (where UE's Lumen shadow trace produces dark stripes right
# at the seam) into a proper solid overlap. 3cm is well below what any camera can
# resolve at the room-scale distances we render at.
WALL_JOINT_OVERLAP_M = 0.03
MIC_HEIGHT_M = 1.2  # v77 gen_rir_multiscene_v77.py compute_mic_pos


def compute_mic_position_cm(*, room_size_m):
    rx, ry, _rz = (float(v) for v in room_size_m)
    return (rx * M2CM / 2.0, ry * M2CM / 2.0, MIC_HEIGHT_M * M2CM)


def compute_source_position_cm(*, room_size_m, source_offset_m=(0.0, 1.7, 0.0)):
    mic_x_cm, mic_y_cm, mic_z_cm = compute_mic_position_cm(room_size_m=room_size_m)
    ox, oy, oz = (float(v) for v in source_offset_m)
    return (mic_x_cm + ox * M2CM, mic_y_cm + oy * M2CM, mic_z_cm + oz * M2CM)


def build_room_checklist(
    *,
    solo_checklist,
    room_size_m,
    mic_pos_cm,
    source_pos_cm,
    window_bounds_cm,
    directional_light_intensity_lux,
    ceiling_casts_shadow,
    window_top_wall_casts_shadow,
    window_wall_casts_shadow,
    wall_material,
    floor_material,
):
    extended = dict(solo_checklist)
    extended["room_size_m"] = tuple(float(v) for v in room_size_m)
    extended["mic_pos_cm"] = tuple(float(v) for v in mic_pos_cm)
    extended["source_pos_cm"] = tuple(float(v) for v in source_pos_cm)
    extended["window_bounds_cm"] = {k: float(v) for k, v in window_bounds_cm.items()}
    extended["directional_light_intensity_lux"] = float(directional_light_intensity_lux)
    extended["ceiling_casts_shadow"] = bool(ceiling_casts_shadow)
    extended["window_top_wall_casts_shadow"] = bool(window_top_wall_casts_shadow)
    extended["window_wall_casts_shadow"] = bool(window_wall_casts_shadow)
    extended["wall_material"] = str(wall_material)
    extended["floor_material"] = str(floor_material)
    extended["human_review"] = [
        "All 4 walls carry a visible apartment wall texture (not gray/untextured)",
        "Ceiling has a visible texture (falls back to MI_Walls if no MI_Ceiling)",
        "Window is a real hole with sky/light visible through it (not a painted decal)",
        "Directional light casts a clear shadow FROM the window direction (window -> interior)",
    ]
    return extended


def write_gpurir_layout(
    output_dir,
    *,
    room_size_m,
    mic_pos_cm,
    source_pos_cm,
    window_bounds_cm,
    orbit_radius_cm,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    rx_cm = float(room_size_m[0]) * M2CM
    ry_cm = float(room_size_m[1]) * M2CM

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.add_patch(plt.Rectangle((0.0, 0.0), rx_cm, ry_cm, fill=False, edgecolor="black"))

    ax.plot(
        [float(window_bounds_cm["left_x"]), float(window_bounds_cm["right_x"])],
        [float(window_bounds_cm["y"]), float(window_bounds_cm["y"])],
        color="tab:green",
        linewidth=4,
        label="window",
    )

    ax.plot([mic_pos_cm[0]], [mic_pos_cm[1]], marker="o", color="tab:blue",
            markersize=8, label="mic")
    ax.plot([source_pos_cm[0]], [source_pos_cm[1]], marker="o", color="tab:orange",
            markersize=10, label="source (dog)")

    theta = [2.0 * math.pi * i / 128 for i in range(129)]
    ax.plot(
        [float(source_pos_cm[0]) + float(orbit_radius_cm) * math.cos(t) for t in theta],
        [float(source_pos_cm[1]) + float(orbit_radius_cm) * math.sin(t) for t in theta],
        linestyle="--",
        color="tab:blue",
        alpha=0.5,
        label=f"orbit r={float(orbit_radius_cm):.0f}cm",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_title(
        f"GPURIR shoebox {room_size_m[0]:.2f}x{room_size_m[1]:.2f}x{room_size_m[2]:.2f} m (top-down)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "layout.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


def compute_shoebox_room_layout(*, room_size_m, wall_thickness_m=WALL_THICKNESS_M):
    rx, ry, rz = (float(v) for v in room_size_m)
    t = float(wall_thickness_m)
    rx_cm = rx * M2CM
    ry_cm = ry * M2CM
    rz_cm = rz * M2CM
    t_cm = t * M2CM

    pieces = []

    # Floor uses the starter-content Floor_400x400 mesh (400x400x0 cm local, pivot
    # in the -X-Y-Z corner, thickness 20cm extending DOWN from y=0). Its UVs are
    # designed for the 4m tile size, so wood-plank materials render at realistic
    # plank width when we scale by room_x/4 x room_y/4. Top face still sits at z=0.
    pieces.append({
        "name": "floor",
        "mesh": FLOOR_MESH,
        "location_cm": (0.0, 0.0, 0.0),
        "scale": (rx / FLOOR_MESH_TILE_M, ry / FLOOR_MESH_TILE_M, 1.0),
    })

    # Ceiling stays a stretched Cube — walls hide any UV-stretch on its underside
    # from the camera unless you point straight up, and even then a plain plaster
    # material has no directional pattern to give the stretch away.
    pieces.append({
        "name": "ceiling",
        "location_cm": (rx_cm / 2.0, ry_cm / 2.0, rz_cm + t_cm / 2.0),
        "scale": (rx, ry, t),
    })

    # Walls: interior kept clear; thickness pushed OUTWARD from room bounds.
    # Vertical span is extended DOWN into the floor and UP into the ceiling by
    # WALL_JOINT_OVERLAP_M at each end, so wall/floor and wall/ceiling contacts
    # are proper solid overlaps instead of coplanar T-junctions (the coplanar
    # case makes Lumen produce a dark shadow-leak stripe right along the seam).
    o = float(WALL_JOINT_OVERLAP_M)
    o_cm = o * M2CM
    wall_rz = rz + 2.0 * o          # vertical scale of every wall Cube
    wall_z_center = rz_cm / 2.0     # unchanged: still centered on room mid-height
    pieces.append({
        "name": "wall_x0",
        "location_cm": (-t_cm / 2.0, ry_cm / 2.0, wall_z_center),
        "scale": (t, ry, wall_rz),
    })
    pieces.append({
        "name": "wall_x1",
        "location_cm": (rx_cm + t_cm / 2.0, ry_cm / 2.0, wall_z_center),
        "scale": (t, ry, wall_rz),
    })
    pieces.append({
        "name": "wall_y0",
        "location_cm": (rx_cm / 2.0, -t_cm / 2.0, wall_z_center),
        "scale": (rx, t, wall_rz),
    })
    # wall_y1 is REPLACED by 4 window-pieces in render_gpurir_room; kept here
    # so callers who want a closed shoebox still get 6 surfaces.
    pieces.append({
        "name": "wall_y1",
        "location_cm": (rx_cm / 2.0, ry_cm + t_cm / 2.0, wall_z_center),
        "scale": (rx, t, wall_rz),
    })

    return pieces


def compute_window_wall_layout(
    *,
    room_size_m,
    window_w_m,
    window_h_m,
    window_cx_m,
    window_z_bottom_m,
    wall_thickness_m=WALL_THICKNESS_M,
):
    rx, ry, rz = (float(v) for v in room_size_m)
    ww = float(window_w_m)
    wh = float(window_h_m)
    wcx = float(window_cx_m)
    wzb = float(window_z_bottom_m)
    t = float(wall_thickness_m)

    rx_cm = rx * M2CM
    rz_cm = rz * M2CM
    ry_cm = ry * M2CM
    t_cm = t * M2CM

    window_left_x_cm = (wcx - ww / 2.0) * M2CM
    window_right_x_cm = (wcx + ww / 2.0) * M2CM
    window_bottom_z_cm = wzb * M2CM
    window_top_z_cm = (wzb + wh) * M2CM

    wall_y_center_cm = ry_cm + t_cm / 2.0

    o = float(WALL_JOINT_OVERLAP_M)
    o_cm = o * M2CM

    pieces = []

    # Bottom sill: full room width, spans z=[-o, window_bottom].
    # Sunk into floor by o to kill wall/floor coplanar seam.
    sill_h_cm = window_bottom_z_cm + o_cm
    sill_center_z_cm = (window_bottom_z_cm - o_cm) / 2.0
    pieces.append({
        "name": "wall_y1_bottom",
        "location_cm": (rx_cm / 2.0, wall_y_center_cm, sill_center_z_cm),
        "scale": (rx, t, sill_h_cm / M2CM),
    })

    # Top lintel: full room width, spans z=[window_top, room_z+o].
    # Pushed up into ceiling by o to kill wall/ceiling coplanar seam.
    lintel_h_cm = rz_cm - window_top_z_cm + o_cm
    lintel_center_z_cm = (window_top_z_cm + rz_cm + o_cm) / 2.0
    pieces.append({
        "name": "wall_y1_top",
        "location_cm": (rx_cm / 2.0, wall_y_center_cm, lintel_center_z_cm),
        "scale": (rx, t, lintel_h_cm / M2CM),
    })

    # Left jamb: from x=0 to x=window_left, spans z=[window_bottom, window_top]
    jamb_h_cm = window_top_z_cm - window_bottom_z_cm
    left_w_cm = window_left_x_cm
    pieces.append({
        "name": "wall_y1_left",
        "location_cm": (
            left_w_cm / 2.0,
            wall_y_center_cm,
            window_bottom_z_cm + jamb_h_cm / 2.0,
        ),
        "scale": (left_w_cm / M2CM, t, jamb_h_cm / M2CM),
    })

    # Right jamb: from x=window_right to x=room_x
    right_w_cm = rx_cm - window_right_x_cm
    pieces.append({
        "name": "wall_y1_right",
        "location_cm": (
            window_right_x_cm + right_w_cm / 2.0,
            wall_y_center_cm,
            window_bottom_z_cm + jamb_h_cm / 2.0,
        ),
        "scale": (right_w_cm / M2CM, t, jamb_h_cm / M2CM),
    })

    return pieces


def compute_window_frame_layout(
    *,
    room_size_m,
    window_w_m,
    window_h_m,
    window_cx_m,
    window_z_bottom_m,
    wall_thickness_m=WALL_THICKNESS_M,
    frame_thickness_m=WINDOW_FRAME_THICKNESS_M,
):
    """Return 4 thin dark trim bars (top/bottom/left/right) that outline the
    window opening on the interior side of the wall_y1 wall. Each bar sits at
    y = ry_cm - frame_thickness_cm/2 so it sticks INTO the room by the frame
    thickness and hides the cube-wall seams."""
    ry, _rz = float(room_size_m[1]), float(room_size_m[2])
    ww = float(window_w_m)
    wh = float(window_h_m)
    wcx = float(window_cx_m)
    wzb = float(window_z_bottom_m)
    fw = float(frame_thickness_m)

    ry_cm = ry * M2CM
    fw_cm = fw * M2CM
    # Frame sits flush with the interior face of the wall (y = ry_cm)
    # protruding INTO the room by frame_thickness (so y_center = ry_cm - fw/2)
    frame_y_cm = ry_cm - fw_cm / 2.0

    window_left_x_cm = (wcx - ww / 2.0) * M2CM
    window_right_x_cm = (wcx + ww / 2.0) * M2CM
    window_bottom_z_cm = wzb * M2CM
    window_top_z_cm = (wzb + wh) * M2CM

    # bar width along the window edge (overlap 2*fw so corners look clean)
    bar_len_x_m = ww + 2 * fw
    bar_len_z_m = wh + 2 * fw

    pieces = []
    # Top bar
    pieces.append({
        "name": "window_frame_top",
        "location_cm": (
            wcx * M2CM,
            frame_y_cm,
            window_top_z_cm + fw_cm / 2.0,
        ),
        "scale": (bar_len_x_m, fw, fw),
    })
    # Bottom bar (sill trim)
    pieces.append({
        "name": "window_frame_bottom",
        "location_cm": (
            wcx * M2CM,
            frame_y_cm,
            window_bottom_z_cm - fw_cm / 2.0,
        ),
        "scale": (bar_len_x_m, fw, fw),
    })
    # Left jamb trim
    pieces.append({
        "name": "window_frame_left",
        "location_cm": (
            window_left_x_cm - fw_cm / 2.0,
            frame_y_cm,
            (window_bottom_z_cm + window_top_z_cm) / 2.0,
        ),
        "scale": (fw, fw, bar_len_z_m),
    })
    # Right jamb trim
    pieces.append({
        "name": "window_frame_right",
        "location_cm": (
            window_right_x_cm + fw_cm / 2.0,
            frame_y_cm,
            (window_bottom_z_cm + window_top_z_cm) / 2.0,
        ),
        "scale": (fw, fw, bar_len_z_m),
    })
    return pieces


def configure_gpurir_instance(*, rpc_port, fixed_delta_time=None):
    import spear

    graphics_adapter_env = os.environ.get("SPEAR_GRAPHICS_ADAPTER")
    settings = parallel_instance_settings(
        rpc_port,
        graphics_adapter=(
            int(graphics_adapter_env)
            if graphics_adapter_env not in (None, "")
            else None
        ),
    )
    config = spear.get_config(user_config_files=[])
    config.defrost()
    config.SPEAR.LAUNCH_MODE = "game"
    config.SPEAR.INSTANCE.GAME_EXECUTABLE = SPEARSIM_EXECUTABLE
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP = True
    config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = EMPTY_MAP
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
        config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.FIXED_DELTA_TIME = float(
            fixed_delta_time
        )
    config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES = "/etc/vulkan/icd.d/nvidia_icd.json"
    config.freeze()
    spear.configure_system(config=config)
    return spear.Instance(config=config)


def piece_casts_shadow(
    name,
    *,
    ceiling_casts_shadow=True,
    window_top_wall_casts_shadow=True,
    window_wall_casts_shadow=True,
):
    if name == "ceiling":
        return bool(ceiling_casts_shadow)
    elif name == "wall_y1_top":
        return bool(window_wall_casts_shadow and window_top_wall_casts_shadow)
    elif name.startswith("wall_y1_"):
        return bool(window_wall_casts_shadow)
    else:
        return True


def spawn_room_piece(game, *, piece, material_path, cast_shadow=None):
    # Cube by default; some pieces (floor) override to a purpose-built mesh
    # whose UVs match a realistic tile size instead of stretching one plank
    # across the whole room.
    mesh_path = piece.get("mesh", CUBE_MESH)
    mesh = game.unreal_service.load_object(uclass="UStaticMesh", name=mesh_path)
    material = game.unreal_service.load_object(uclass="UMaterialInterface", name=material_path)
    actor = game.unreal_service.spawn_actor(
        uclass="AStaticMeshActor",
        location={
            "X": float(piece["location_cm"][0]),
            "Y": float(piece["location_cm"][1]),
            "Z": float(piece["location_cm"][2]),
        },
        spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
    )
    smc = game.unreal_service.get_component_by_class(
        actor=actor, uclass="UStaticMeshComponent"
    )
    smc.SetMobility(NewMobility="Movable")
    smc.SetStaticMesh(NewMesh=mesh)
    smc.SetMaterial(ElementIndex=0, Material=material)
    if cast_shadow is None:
        cast_shadow = piece_casts_shadow(name=piece["name"])
    smc.SetCastShadow(NewCastShadow=bool(cast_shadow))
    if not cast_shadow:
        try:
            smc.SetCastHiddenShadow(NewCastHiddenShadow=False)
        except Exception:
            pass
        try:
            smc.bCastDynamicShadow = False
        except Exception:
            pass
        try:
            smc.bCastStaticShadow = False
        except Exception:
            pass
    actor.SetActorScale3D(
        NewScale3D={
            "X": float(piece["scale"][0]),
            "Y": float(piece["scale"][1]),
            "Z": float(piece["scale"][2]),
        }
    )
    game.unreal_service.set_stable_name_for_actor(
        actor=actor, stable_name=f"GpurirRoom/{piece['name']}"
    )
    return actor


def spawn_directional_light(game, *, yaw_deg, pitch_deg, intensity_lux):
    light = game.unreal_service.spawn_actor(
        uclass="ADirectionalLight",
        location={"X": 0.0, "Y": 0.0, "Z": 500.0},
    )
    root = light.K2_GetRootComponent()
    root.SetMobility(NewMobility="Movable")
    light.K2_SetActorLocationAndRotation(
        NewLocation={"X": 0.0, "Y": 0.0, "Z": 500.0},
        NewRotation={"Roll": 0.0, "Pitch": float(pitch_deg), "Yaw": float(yaw_deg)},
        bSweep=False,
        bTeleport=True,
    )
    comp = game.unreal_service.get_component_by_class(
        actor=light, uclass="UDirectionalLightComponent"
    )
    comp.SetIntensity(NewIntensity=float(intensity_lux))
    return light


def spawn_point_light(game, *, x_cm, y_cm, z_cm, intensity_lumens, attenuation_cm):
    """Spawn a movable point light (ceiling lamp)."""
    light = game.unreal_service.spawn_actor(
        uclass="APointLight",
        location={"X": float(x_cm), "Y": float(y_cm), "Z": float(z_cm)},
    )
    root = light.K2_GetRootComponent()
    root.SetMobility(NewMobility="Movable")
    comp = game.unreal_service.get_component_by_class(
        actor=light, uclass="UPointLightComponent"
    )
    comp.SetIntensity(NewIntensity=float(intensity_lumens))
    comp.SetAttenuationRadius(NewRadius=float(attenuation_cm))
    comp.SetCastShadows(bNewValue=True)
    return light


def spawn_sphere_reflection_capture(game, *, x_cm, y_cm, z_cm, influence_radius_cm):
    """Spawn a reflection capture so glass and shiny surfaces get environment reflections."""
    actor = game.unreal_service.spawn_actor(
        uclass="ASphereReflectionCapture",
        location={"X": float(x_cm), "Y": float(y_cm), "Z": float(z_cm)},
    )
    comp = game.unreal_service.get_component_by_class(
        actor=actor, uclass="USphereReflectionCaptureComponent"
    )
    try:
        comp.InfluenceRadius = float(influence_radius_cm)
    except Exception:
        pass
    # Hide the editor preview sphere while keeping the reflection data active.
    # This is safe: reflection capture data lives in the light rendering path,
    # NOT in the visible mesh — unlike DirLight/Camera where hiding the actor
    # kills the underlying function.
    try:
        actor.SetActorHiddenInGame(bNewHidden=True)
    except Exception:
        pass
    return actor


def spawn_sky(game):
    """Spawn native SkyAtmosphere + SkyLight for outside-the-window ambient.

    We deliberately do NOT try to spawn BP_LightStudio because that BP is not
    packaged in the standalone build (only its /Game/... referrer is). Native
    ASkyAtmosphere + ASkyLight compile into every UE build without asset deps.
    """
    actors = {}
    for uclass in ("ASkyAtmosphere", "ASkyLight", "AExponentialHeightFog"):
        try:
            actor = game.unreal_service.spawn_actor(
                uclass=uclass,
                location={"X": 0.0, "Y": 0.0, "Z": 100.0},
                spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
            )
            actors[uclass] = actor
        except Exception as e:
            print(f"[gpurir-room] spawn_sky: skip {uclass} ({e})", flush=True)
    return actors


def _material_for_piece(name, wall_material=WALL_MATERIAL):
    if name == "floor":
        return FLOOR_MATERIAL
    if name == "outdoor_ground":
        return OUTDOOR_GROUND_MATERIAL
    if name == "window_glass":
        return GLASS_MATERIAL
    if name.startswith("window_frame_"):
        return WINDOW_FRAME_MATERIAL
    return wall_material


def resolve_floor_material(*, floor_material=None, floor_material_seed=0):
    """Pick which floor material to render with.

    Precedence: explicit --floor-material wins; otherwise deterministic index
    into FLOOR_MATERIAL_POOL from --floor-material-seed. This gives dataset
    generation callers a single scalar knob (seed) to vary floor appearance
    across runs while keeping any single run reproducible.
    """
    if floor_material:
        return floor_material
    idx = int(floor_material_seed) % len(FLOOR_MATERIAL_POOL)
    return FLOOR_MATERIAL_POOL[idx]


def resolve_wall_material(*, wall_material=None, wall_material_seed=0):
    """Same idea as resolve_floor_material, independent seed so floor x wall
    combinations multiply out for dataset variety."""
    if wall_material:
        return wall_material
    idx = int(wall_material_seed) % len(WALL_MATERIAL_POOL)
    return WALL_MATERIAL_POOL[idx]


def _try_hide(actor):
    try:
        actor.SetActorHiddenInGame(bNewHidden=True)
    except Exception:
        pass


def render_gpurir_room(args):
    import cv2

    output_dir = os.path.join(args.output_root, args.run_name)
    clean_frames(output_dir)

    room_pieces = compute_shoebox_room_layout(
        room_size_m=args.room_size_m,
        wall_thickness_m=args.wall_thickness_m,
    )
    window_cx = args.window_cx_m if args.window_cx_m is not None else args.room_size_m[0] / 2.0
    window_pieces = compute_window_wall_layout(
        room_size_m=args.room_size_m,
        window_w_m=args.window_w_m,
        window_h_m=args.window_h_m,
        window_cx_m=window_cx,
        window_z_bottom_m=args.window_z_bottom_m,
        wall_thickness_m=args.wall_thickness_m,
    )
    window_frame_pieces = compute_window_frame_layout(
        room_size_m=args.room_size_m,
        window_w_m=args.window_w_m,
        window_h_m=args.window_h_m,
        window_cx_m=window_cx,
        window_z_bottom_m=args.window_z_bottom_m,
        wall_thickness_m=args.wall_thickness_m,
        frame_thickness_m=args.window_frame_thickness_m,
    )
    non_y1 = [p for p in room_pieces if p["name"] != "wall_y1"]

    # Window glass: 3cm thick Cube filling the window opening, aligned with wall_y1 center
    ry_cm = args.room_size_m[1] * M2CM
    t_cm = args.wall_thickness_m * M2CM
    glass_piece = {
        "name": "window_glass",
        "location_cm": (
            window_cx * M2CM,
            ry_cm + t_cm / 2.0,
            (args.window_z_bottom_m + args.window_h_m / 2.0) * M2CM,
        ),
        "scale": (args.window_w_m, GLASS_THICKNESS_M, args.window_h_m),
    }

    outdoor_ground_piece = {
        "name": "outdoor_ground",
        "location_cm": (
            args.room_size_m[0] * M2CM / 2.0,
            args.room_size_m[1] * M2CM / 2.0,
            -args.wall_thickness_m * M2CM - 5.0,
        ),
        "scale": (80.0, 80.0, 0.1),
    }

    all_pieces = non_y1 + window_pieces + window_frame_pieces + [glass_piece, outdoor_ground_piece]

    mic_pos_cm = compute_mic_position_cm(room_size_m=args.room_size_m)
    source_pos_cm = compute_source_position_cm(
        room_size_m=args.room_size_m,
        source_offset_m=args.source_offset_m,
    )
    window_bounds_cm = {
        "left_x": (window_cx - args.window_w_m / 2.0) * M2CM,
        "right_x": (window_cx + args.window_w_m / 2.0) * M2CM,
        "bottom_z": args.window_z_bottom_m * M2CM,
        "top_z": (args.window_z_bottom_m + args.window_h_m) * M2CM,
        "y": args.room_size_m[1] * M2CM,
    }

    # Assemble animal specs: primary at index 0, extras follow.
    # Each extra is (name, dx_m, dy_m) offset from source_pos_cm.
    animal_specs = [{"name": args.animal, "dx_m": 0.0, "dy_m": 0.0}]
    for name, dx, dy in (args.extra_animal or []):
        animal_specs.append({"name": name, "dx_m": float(dx), "dy_m": float(dy)})

    # Preload meta for each animal
    for spec in animal_specs:
        with open(animal_meta_path(args.meta_dir, spec["name"]), "r", encoding="utf-8") as f:
            spec["meta"] = json.load(f)

    instance = configure_gpurir_instance(rpc_port=args.rpc_port)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            # /Engine/Maps/Entry auto-spawns a PlayerStart + SpectatorPawn at
            # (55, 79, 208) whose gizmo/mesh renders as a gray checkerboard
            # 80cm cube in the corner of the shot. We already spawn our own
            # camera actor, so these two are safe to destroy.
            #
            # NOTE: destroying AStaticMeshActor / SkeletalMeshActor / Brush /
            # Decal here is a defensive sweep for future maps but does NOT
            # remove the PlayerStart cube by itself.
            for cls in ("APlayerStart", "ADefaultPawn", "ASpectatorPawn",
                        "AStaticMeshActor", "ASkeletalMeshActor", "ABrush",
                        "ADecalActor", "AInstancedFoliageActor",
                        "AGameplayDebuggerCategoryReplicator",
                        "AGameplayDebuggerPlayerManager"):
                try:
                    victims = game.unreal_service.find_actors_by_class(uclass=cls)
                except Exception:
                    victims = []
                for existing in victims:
                    try:
                        game.unreal_service.destroy_actor(actor=existing)
                    except Exception:
                        pass

            resolved_floor_material = resolve_floor_material(
                floor_material=args.floor_material,
                floor_material_seed=args.floor_material_seed,
            )
            resolved_wall_material = resolve_wall_material(
                wall_material=args.wall_material,
                wall_material_seed=args.wall_material_seed,
            )
            print(f"[gpurir-room] floor material = {resolved_floor_material}", flush=True)
            print(f"[gpurir-room] wall material  = {resolved_wall_material}", flush=True)
            for piece in all_pieces:
                material_path = _material_for_piece(
                    piece["name"], wall_material=resolved_wall_material
                )
                if piece["name"] == "floor":
                    material_path = resolved_floor_material
                spawn_room_piece(
                    game=game,
                    piece=piece,
                    material_path=material_path,
                    cast_shadow=piece_casts_shadow(
                        name=piece["name"],
                        ceiling_casts_shadow=args.ceiling_casts_shadow,
                        window_top_wall_casts_shadow=args.window_top_wall_casts_shadow,
                        window_wall_casts_shadow=args.window_wall_casts_shadow,
                    ),
                )

            spawn_sky(game=game)
            dir_light = spawn_directional_light(
                game=game,
                yaw_deg=args.directional_light_yaw_deg,
                pitch_deg=args.directional_light_pitch_deg,
                intensity_lux=args.directional_light_intensity_lux,
            )
            _ = dir_light  # do NOT hide light actors — SetActorHiddenInGame kills them

            # Ceiling light: mounted just below ceiling at room center
            room_x_cm = args.room_size_m[0] * M2CM
            room_y_cm = args.room_size_m[1] * M2CM
            room_z_cm = args.room_size_m[2] * M2CM
            spawn_point_light(
                game=game,
                x_cm=room_x_cm / 2.0,
                y_cm=room_y_cm / 2.0,
                z_cm=room_z_cm - args.ceiling_light_drop_cm,
                intensity_lumens=args.ceiling_light_lumens,
                attenuation_cm=args.ceiling_light_attenuation_cm,
            )

            # Reflection capture at room center so glass + walls get proper reflections
            if args.reflection_capture:
                spawn_sphere_reflection_capture(
                    game=game,
                    x_cm=room_x_cm / 2.0,
                    y_cm=room_y_cm / 2.0,
                    z_cm=room_z_cm / 2.0,
                    influence_radius_cm=max(room_x_cm, room_y_cm, room_z_cm),
                )

            # Spawn each animal + record fit for phase 2
            for spec in animal_specs:
                pos_cm = (
                    source_pos_cm[0] + spec["dx_m"] * M2CM,
                    source_pos_cm[1] + spec["dy_m"] * M2CM,
                )
                gz, _ = sample_ground_z(
                    game=game,
                    x=pos_cm[0],
                    y=pos_cm[1],
                    fallback_z=args.floor_z,
                    trace_start_z=args.ground_trace_start_z,
                    trace_end_z=args.ground_trace_end_z,
                )
                fit = compute_asset_fit(
                    meta=spec["meta"],
                    target_cm=args.target_cm,
                    floor_z=gz + args.ground_clearance_cm,
                )
                bp_asset = game.unreal_service.load_class(
                    uclass="AActor", name=animal_bp_path(spec["name"])
                )
                actor = game.unreal_service.spawn_actor(
                    uclass=bp_asset,
                    location={"X": pos_cm[0], "Y": pos_cm[1], "Z": args.floor_z},
                    spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
                )
                try:
                    actor.K2_GetRootComponent().SetMobility(NewMobility="Movable")
                except Exception:
                    pass
                game.unreal_service.set_stable_name_for_actor(
                    actor=actor, stable_name=f"GpurirRoom/{spec['name']}"
                )
                spec["pos_cm"] = pos_cm
                spec["ground_z"] = gz
                spec["fit"] = fit
                spec["actor"] = actor

            cam, comp = spawn_camera(game=game, width=args.width, height=args.height)
            # NOTE: do NOT SetActorHiddenInGame on cam — it kills the capture.
        with instance.end_frame():
            pass

        instance.step(num_frames=4)

        with instance.begin_frame():
            for spec in animal_specs:
                actor = spec["actor"]
                fit = spec["fit"]
                pos_cm = spec["pos_cm"]
                actor.SetActorScale3D(
                    NewScale3D={"X": fit["scale"], "Y": fit["scale"], "Z": fit["scale"]}
                )
                actor.K2_SetActorLocation(
                    NewLocation={"X": pos_cm[0], "Y": pos_cm[1], "Z": fit["actor_z"]},
                    bSweep=False,
                    bTeleport=True,
                )
                bounds_bottom_z, _ = get_actor_bounds_bottom_z(actor=actor)
                lift_cm = compute_bounds_lift(
                    bounds_bottom_z=bounds_bottom_z,
                    ground_z=spec["ground_z"],
                    clearance_cm=args.ground_clearance_cm,
                    tolerance_cm=args.ground_tolerance_cm,
                )
                if lift_cm > 0.0:
                    fit["actor_z"] += lift_cm
                    fit["center_z"] += lift_cm
                    actor.K2_SetActorLocation(
                        NewLocation={"X": pos_cm[0], "Y": pos_cm[1], "Z": fit["actor_z"]},
                        bSweep=False,
                        bTeleport=True,
                    )
                spec["bounds_bottom_z"] = bounds_bottom_z
                spec["lift_cm"] = lift_cm
        with instance.end_frame():
            pass

        # Bind primary animal for downstream logic (orbit center + checklist)
        primary = animal_specs[0]
        asset = primary["actor"]
        fit = primary["fit"]
        ground_z = primary["ground_z"]
        lift_cm = primary["lift_cm"]

        center_x = float(source_pos_cm[0])
        center_y = float(source_pos_cm[1])
        center_z = float(fit["center_z"])

        print(
            "[gpurir-room] "
            f"animal={args.animal} room={args.room_size_m}m "
            f"mic={mic_pos_cm} source={source_pos_cm} "
            f"ground_z={ground_z:.2f}cm bounds_bottom={bounds_bottom_z:.2f}cm "
            f"lift={lift_cm:.2f}cm scale={fit['scale']:.3f} "
            f"radius={args.orbit_radius_cm:.0f}cm center=({center_x:.1f},{center_y:.1f},{center_z:.1f})",
            flush=True,
        )

        instance.step(num_frames=args.warmup_frames)
        for i in range(args.frames):
            frame_warmup = args.warmup_frames if i == 0 else args.per_frame_warmup_frames
            theta = 2.0 * math.pi * i / args.frames
            cam_x = center_x + args.orbit_radius_cm * math.cos(theta)
            cam_y = center_y + args.orbit_radius_cm * math.sin(theta)
            cam_z = center_z + args.cam_z_offset_cm
            yaw = math.degrees(math.atan2(center_y - cam_y, center_x - cam_x))
            pitch = -math.degrees(math.atan2(args.cam_z_offset_cm, args.orbit_radius_cm))
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

        with instance.begin_frame():
            final_bounds_bottom_z, _ = get_actor_bounds_bottom_z(actor=asset)
        with instance.end_frame():
            pass

        penetration = (ground_z + args.ground_clearance_cm) - final_bounds_bottom_z
        solo = build_solo_checklist(
            name=args.animal,
            ground_z=ground_z,
            bounds_bottom_z=final_bounds_bottom_z,
            lift_cm=lift_cm,
            penetration_after_lift=penetration,
            scale=fit["scale"],
            target_cm=args.target_cm,
            radius=args.orbit_radius_cm,
            frames=args.frames,
            clearance_cm=args.ground_clearance_cm,
            tolerance_cm=args.ground_tolerance_cm,
        )
        checklist = build_room_checklist(
            solo_checklist=solo,
            room_size_m=args.room_size_m,
            mic_pos_cm=mic_pos_cm,
            source_pos_cm=source_pos_cm,
            window_bounds_cm=window_bounds_cm,
            directional_light_intensity_lux=args.directional_light_intensity_lux,
            ceiling_casts_shadow=args.ceiling_casts_shadow,
            window_top_wall_casts_shadow=args.window_top_wall_casts_shadow,
            window_wall_casts_shadow=args.window_wall_casts_shadow,
            wall_material=resolved_wall_material,
            floor_material=resolved_floor_material,
        )
        checklist["extras"] = [
            {
                "name": s["name"],
                "offset_m": [s["dx_m"], s["dy_m"]],
                "pos_cm": list(s["pos_cm"]),
                "scale": s["fit"]["scale"],
                "ground_z_cm": s["ground_z"],
                "bounds_bottom_z_cm": s["bounds_bottom_z"],
                "lift_applied_cm": s["lift_cm"],
            }
            for s in animal_specs[1:]
        ]
        checklist["window_frame_material"] = WINDOW_FRAME_MATERIAL
        checklist["glass_material"] = GLASS_MATERIAL
        checklist_path = write_checklist(output_dir, checklist)
        print(f"CHECKLIST_DONE {checklist_path}", flush=True)

        layout_path = write_gpurir_layout(
            output_dir,
            room_size_m=args.room_size_m,
            mic_pos_cm=mic_pos_cm,
            source_pos_cm=source_pos_cm,
            window_bounds_cm=window_bounds_cm,
            orbit_radius_cm=args.orbit_radius_cm,
        )
        print(f"LAYOUT_DONE {layout_path}", flush=True)
    finally:
        instance.close(force=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--animal", choices=SUPPORTED_ANIMALS, default="dog")
    parser.add_argument(
        "--room-size-m",
        type=float,
        nargs=3,
        default=[5.2, 4.4, 2.8],
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument("--wall-thickness-m", type=float, default=WALL_THICKNESS_M)
    parser.add_argument("--window-w-m", type=float, default=1.4)
    parser.add_argument("--window-h-m", type=float, default=1.4)
    parser.add_argument("--window-frame-thickness-m", type=float, default=WINDOW_FRAME_THICKNESS_M)
    parser.add_argument(
        "--window-cx-m",
        type=float,
        default=None,
        help="Window center X in meters. Default = room_x/2.",
    )
    parser.add_argument("--window-z-bottom-m", type=float, default=0.9)
    parser.add_argument(
        "--source-offset-m",
        type=float,
        nargs=3,
        default=[0.0, 1.7, 0.0],
        metavar=("DX", "DY", "DZ"),
        help="Source (animal) position offset from mic, in meters.",
    )
    parser.add_argument("--target-cm", type=float, default=80.0)
    parser.add_argument("--orbit-radius-cm", type=float, default=200.0)
    parser.add_argument("--cam-z-offset-cm", type=float, default=40.0)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--framerate", type=int, default=12)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--per-frame-warmup-frames", type=int, default=6)
    parser.add_argument("--ground-clearance-cm", type=float, default=0.5)
    parser.add_argument("--ground-tolerance-cm", type=float, default=0.5)
    # Default trace starts JUST BELOW the ceiling of the default room (2.8m = 280cm).
    # Starting above the ceiling would hit the ceiling's outside face at z=290cm
    # and place the animal on the ROOF (the room is closed). Users overriding
    # --room-size-m with a taller room should also override --ground-trace-start-z.
    parser.add_argument("--ground-trace-start-z", type=float, default=270.0)
    parser.add_argument("--ground-trace-end-z", type=float, default=-200.0)
    parser.add_argument("--floor-z", type=float, default=0.0)
    parser.add_argument("--directional-light-intensity-lux", type=float, default=10.0)
    # Floor material control. Two mutually-compatible knobs:
    #   --floor-material PATH  : force a specific material (overrides pool)
    #   --floor-material-seed N: pick from FLOOR_MATERIAL_POOL deterministically.
    # When both --floor-material is unset AND no seed given, we still pick from
    # the pool but with seed=0 for reproducibility unless the user opts in.
    parser.add_argument(
        "--floor-material",
        default=None,
        help="Override floor material. If unset, we sample from FLOOR_MATERIAL_POOL "
             "using --floor-material-seed for dataset multiplicity.",
    )
    parser.add_argument(
        "--floor-material-seed", type=int, default=0,
        help="Deterministic index into FLOOR_MATERIAL_POOL when --floor-material is not set. "
             "Use `--floor-material-seed N` where N is any int; picks pool[N %% len(pool)].",
    )
    parser.add_argument(
        "--wall-material",
        default=None,
        help="Override wall material. If unset, samples from WALL_MATERIAL_POOL "
             "using --wall-material-seed. Independent from floor for combinatorial variety.",
    )
    parser.add_argument(
        "--wall-material-seed", type=int, default=0,
        help="Deterministic index into WALL_MATERIAL_POOL. Independent from floor seed.",
    )
    parser.add_argument("--ceiling-casts-shadow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--window-top-wall-casts-shadow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--window-wall-casts-shadow", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ceiling-light-lumens", type=float, default=2200.0,
                        help="Point light intensity for the ceiling lamp (~100W incandescent).")
    parser.add_argument("--ceiling-light-drop-cm", type=float, default=15.0,
                        help="Distance ceiling light hangs below the ceiling mesh.")
    parser.add_argument("--ceiling-light-attenuation-cm", type=float, default=600.0)
    parser.add_argument(
        "--reflection-capture",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Spawn SphereReflectionCapture. Off by default because its preview "
             "mesh sometimes leaks into game view as a gray checkerboard cube.",
    )
    parser.add_argument(
        "--directional-light-yaw-deg", type=float, default=-90.0,
        help="Yaw pointing INTO the room from the +Y window (UE -90 = light travels in -Y).",
    )
    parser.add_argument("--directional-light-pitch-deg", type=float, default=-40.0)
    parser.add_argument("--rpc-port", type=int, default=39002)
    parser.add_argument("--meta-dir", default=DEFAULT_META_DIR)
    parser.add_argument("--output-root", default=DEFAULT_TMP_ROOT)
    parser.add_argument("--run-name", default="dog_default")
    parser.add_argument(
        "--extra-animal",
        action="append",
        nargs=3,
        default=[],
        metavar=("NAME", "DX_M", "DY_M"),
        help=(
            "Additional animal placed at (source + (dx_m, dy_m)). "
            "Repeat to add multiple. NAME must be one of SUPPORTED_ANIMALS."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    render_gpurir_room(args)


if __name__ == "__main__":
    main()
