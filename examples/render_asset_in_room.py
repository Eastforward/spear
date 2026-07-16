"""Per-asset turntable render in the clean MI_Walls/MI_Floor room.

Concretely:
- Spawns the 6 BP_Room surfaces + 1 BP_RoomLight at high above the map
- Spawns BP_<NAME> on the room floor, auto-fit by tmp/asset_meta/<NAME>.json
- Orbits a camera at R = 3.0 * TARGET (cm), CAM_Z_OFF = 40 cm above asset center,
  pitch = -atan2(40, R), yaw = pointed at asset center
- Reads 36 frames into PNGs + ffmpeg -> turntable.mp4

Usage:
  python examples/render_asset_in_room.py [<BP>] [<NAME>] [<TARGET_cm>]
  defaults to Clock in this project.

Tunable constants near top:
  R_FACTOR, CAM_Z_OFF_CM, N_FRAMES, ROOM_LIGHT_LUMENS_AT_RUNTIME
"""
import os, math, subprocess, sys, json, spear, cv2

ASSET_BP = sys.argv[1] if len(sys.argv) > 1 else "/Game/MyAssets/Blueprints/BP_Clock.BP_Clock_C"
NAME = sys.argv[2] if len(sys.argv) > 2 else "Clock"
TARGET = float(sys.argv[3]) if len(sys.argv) > 3 else 35.0  # cm, realistic per-asset
OUT = f"/data/jzy/code/SPEAR/tmp/render_{NAME}"
os.makedirs(OUT, exist_ok=True)
EXEC = "/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh"
MAP = "/Game/StarterContent/Maps/Minimal_Default"
FLOOR_Z = 1000.0
RES_W, RES_H = 1280, 720

# Tunables
R_FACTOR = 3.0
CAM_Z_OFF_CM = 40.0
N_FRAMES = 36
ROOM_LIGHT_LUMENS_AT_RUNTIME = 800.0  # try 800 to avoid blow-out

ROOM = [
    ("/Game/MyAssets/Room/BP_RoomFloor.BP_RoomFloor_C",     0,    0,    -5),
    ("/Game/MyAssets/Room/BP_RoomCeiling.BP_RoomCeiling_C", 0,    0,    285),
    ("/Game/MyAssets/Room/BP_RoomWallXP.BP_RoomWallXP_C",   260,  0,    140),
    ("/Game/MyAssets/Room/BP_RoomWallXN.BP_RoomWallXN_C",  -260,  0,    140),
    ("/Game/MyAssets/Room/BP_RoomWallYP.BP_RoomWallYP_C",   0,    220,  140),
    ("/Game/MyAssets/Room/BP_RoomWallYN.BP_RoomWallYN_C",   0,   -220,  140),
]
ROOM_LIGHT = ("/Game/MyAssets/Room/BP_RoomLight.BP_RoomLight_C", 0, 0, 250)

config = spear.get_config(user_config_files=[]); config.defrost()
config.SPEAR.LAUNCH_MODE = "game"; config.SPEAR.INSTANCE.GAME_EXECUTABLE = EXEC
config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP = True
config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = MAP
config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES = "/etc/vulkan/icd.d/nvidia_icd.json"
config.freeze(); spear.configure_system(config=config)
instance = spear.Instance(config=config); game = instance.get_game()

with instance.begin_frame():
    for bp, x, y, z in ROOM:
        c = game.unreal_service.load_class(uclass="AActor", name=bp)
        game.unreal_service.spawn_actor(uclass=c, location={"X": float(x), "Y": float(y), "Z": FLOOR_Z + float(z)})
    lc = game.unreal_service.load_class(uclass="AActor", name=ROOM_LIGHT[0])
    light_actor = game.unreal_service.spawn_actor(uclass=lc, location={"X": float(ROOM_LIGHT[1]),
                                                                      "Y": float(ROOM_LIGHT[2]),
                                                                      "Z": FLOOR_Z + float(ROOM_LIGHT[3])})
    try:
        lcomp = game.unreal_service.get_component_by_class(actor=light_actor, uclass="UPointLightComponent")
        lcomp.set_editor_property("intensity", ROOM_LIGHT_LUMENS_AT_RUNTIME)
        print(f"[light] runtime intensity -> {ROOM_LIGHT_LUMENS_AT_RUNTIME} lm", flush=True)
    except Exception as e:
        print(f"[light] runtime set failed ({e}); using BP default", flush=True)

    bp_asset = game.unreal_service.load_class(uclass="AActor", name=ASSET_BP)
    asset = game.unreal_service.spawn_actor(uclass=bp_asset, location={"X": 0.0, "Y": 0.0, "Z": FLOOR_Z})

    cam_uclass = game.unreal_service.load_class(uclass="AActor", name="/SpContent/Blueprints/BP_CameraSensor.BP_CameraSensor_C")
    cam = game.unreal_service.spawn_actor(uclass=cam_uclass)
    comp = game.unreal_service.get_component_by_name(actor=cam, component_name="DefaultSceneRoot.final_tone_curve_hdr_", uclass="USpSceneCaptureComponent2D")
    vd = game.rendering_service.get_current_viewport_desc()
    game.rendering_service.align_camera_with_viewport(camera_sensor=cam, camera_components=[comp], viewport_desc=vd, widths=RES_W, heights=RES_H)
    comp.Initialize(); comp.initialize_sp_funcs()
with instance.end_frame():
    pass

instance.step(num_frames=4)
meta = json.load(open(f"/data/jzy/code/SPEAR/tmp/asset_meta/{NAME}.json"))
ext, bmin_z, height = meta["ext"], meta["bmin_z"], meta["height"]
scale = TARGET / max(ext, 1e-3)
with instance.begin_frame():
    asset.SetActorScale3D(NewScale3D={"X": scale, "Y": scale, "Z": scale})
    print(f"[autofit] ext={ext:.1f} -> scale={scale:.3f} (target {TARGET:.0f}cm)", flush=True)
with instance.end_frame():
    pass

instance.step(num_frames=4)
with instance.begin_frame():
    asset.K2_SetActorLocation(NewLocation={"X": 0.0, "Y": 0.0, "Z": FLOOR_Z - bmin_z * scale})
    cx, cy, cz = 0.0, 0.0, FLOOR_Z + (height / 2.0) * scale
    # camera radius: ideal R_FACTOR*TARGET, but clamp so camera stays INSIDE the room.
    # Room is 5.2x4.4m centered at origin -> X walls at +-260, Y walls at +-220. Use 200cm margin.
    R = min(R_FACTOR * TARGET, 200.0)
    print(f"[autofit] grounded center=({cx:.0f},{cy:.0f},{cz:.0f}) R={R:.0f} (clamped to room) cam_z_off={CAM_Z_OFF_CM}", flush=True)
with instance.end_frame():
    pass

instance.step(num_frames=30)

camz = cz + CAM_Z_OFF_CM
pitch = -math.degrees(math.atan2(CAM_Z_OFF_CM, R))
for i in range(N_FRAMES):
    th = 2.0 * math.pi * i / N_FRAMES
    camx, camy = cx + R * math.cos(th), cy + R * math.sin(th)
    yaw = math.degrees(math.atan2(cy - camy, cx - camx))
    with instance.begin_frame():
        cam.K2_SetActorLocationAndRotation(NewLocation={"X": camx, "Y": camy, "Z": camz},
                                           NewRotation={"Roll": 0.0, "Pitch": pitch, "Yaw": yaw})
    with instance.end_frame():
        b = comp.read_pixels()
        img = b["arrays"]["data"][:, :, [0, 1, 2]]
        cv2.imwrite(f"{OUT}/frame_{i:04d}.png", img)
print(f"[render] captured {N_FRAMES}", flush=True)
instance.close(force=True)
subprocess.run(["ffmpeg", "-y", "-framerate", "12", "-i", f"{OUT}/frame_%04d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                f"{OUT}/turntable.mp4"], capture_output=True)
print(f"VIDEO DONE {OUT}/turntable.mp4", flush=True)
