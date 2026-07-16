"""T1 preview: empty room, 4 camera angles -> 4 frames.

Why this exists:
  Before running a full 36-frame turntable for any asset, we want to verify the
  newly-applied MI_Walls / MI_Floor materials look correct on ALL 6 surfaces
  (walls, floor, ceiling). Run this after my_build_room.py has been re-run with
  the apartment MIs swapped in.

Output:
  /data/jzy/code/SPEAR/tmp/preview_room/wallN.png  (N, S, E, W)
  /data/jzy/code/SPEAR/tmp/preview_room/ceiling.png  (looking up)
  /data/jzy/code/SPEAR/tmp/preview_room/floor.png    (looking down)

Each frame the camera is placed near the wall surface (offset slightly inward to
avoid clipping), looking at the center of the room, with full 640x480 framing.

Run:
  python examples/preview_room.py
"""
import os, spear, cv2

OUT = "/data/jzy/code/SPEAR/tmp/preview_room"
os.makedirs(OUT, exist_ok=True)

EXEC = "/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh"
MAP = "/Game/StarterContent/Maps/Minimal_Default"
FLOOR_Z = 1000.0  # keep in sync with my_build_room.py + my_render_in_room.py

# Reuse the same ROOM list from my_render_in_room.py
ROOM = [
    ("/Game/MyAssets/Room/BP_RoomFloor.BP_RoomFloor_C",     0,    0,    -5),
    ("/Game/MyAssets/Room/BP_RoomCeiling.BP_RoomCeiling_C", 0,    0,    285),
    ("/Game/MyAssets/Room/BP_RoomWallXP.BP_RoomWallXP_C",   260,  0,    140),
    ("/Game/MyAssets/Room/BP_RoomWallXN.BP_RoomWallXN_C",  -260,  0,    140),
    ("/Game/MyAssets/Room/BP_RoomWallYP.BP_RoomWallYP_C",   0,    220,  140),
    ("/Game/MyAssets/Room/BP_RoomWallYN.BP_RoomWallYN_C",   0,   -220,  140),
]
ROOM_LIGHT = ("/Game/MyAssets/Room/BP_RoomLight.BP_RoomLight_C", 0, 0, 250)

# Camera shots.  Each shot: (label, location (X, Y, Z above FLOOR_Z), yaw_deg)
#  Room center horizontally is (0, 0); center vertically is FLOOR_Z + 140 (mid-height).
#  Walls live at X = ±260 / Y = ±220.  Place camera 40 cm inside each wall.
ROOM_CZ = FLOOR_Z + 140.0
SHOTS = [
    ("wallN", (   0.0,  -180.0, ROOM_CZ + 60.0),   0.0),   # YN wall to the south -> look +Y
    ("wallS", (   0.0,   180.0, ROOM_CZ + 60.0), 180.0),   # YP wall to the north -> look -Y
    ("wallE", (  220.0,    0.0, ROOM_CZ + 60.0), 270.0),   # XP wall to the east  -> look -X
    ("wallW", ( -220.0,    0.0, ROOM_CZ + 60.0),  90.0),   # XN wall to the west  -> look +X
    ("ceiling", (   0.0,    0.0,  280.0),          0.0),   # high & centered, looking down (Pitch -90 set inline)
    ("floor",   (   0.0,    0.0,  ROOM_CZ + 60.0), 0.0),   # looking down
]

config = spear.get_config(user_config_files=[]); config.defrost()
config.SPEAR.LAUNCH_MODE = "game"; config.SPEAR.INSTANCE.GAME_EXECUTABLE = EXEC
config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP = True
config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = MAP
config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES = "/etc/vulkan/icd.d/nvidia_icd.json"
# use a non-default RPC port to dodge stale binding races between attempts
config.SP_SERVICES.RPC_SERVICE.RPC_SERVER_PORT = 39001
config.freeze(); spear.configure_system(config=config)
instance = spear.Instance(config=config); game = instance.get_game()

with instance.begin_frame():
    # spawn room
    for bp, x, y, z in ROOM:
        c = game.unreal_service.load_class(uclass="AActor", name=bp)
        game.unreal_service.spawn_actor(uclass=c, location={"X": float(x), "Y": float(y), "Z": FLOOR_Z + float(z)})
    lc = game.unreal_service.load_class(uclass="AActor", name=ROOM_LIGHT[0])
    game.unreal_service.spawn_actor(uclass=lc, location={"X": float(ROOM_LIGHT[1]),
                                                          "Y": float(ROOM_LIGHT[2]),
                                                          "Z": FLOOR_Z + float(ROOM_LIGHT[3])})
    # camera
    cam_uclass = game.unreal_service.load_class(uclass="AActor",
                                                name="/SpContent/Blueprints/BP_CameraSensor.BP_CameraSensor_C")
    cam = game.unreal_service.spawn_actor(uclass=cam_uclass)
    comp = game.unreal_service.get_component_by_name(actor=cam,
                                                     component_name="DefaultSceneRoot.final_tone_curve_hdr_",
                                                     uclass="USpSceneCaptureComponent2D")
    vd = game.rendering_service.get_current_viewport_desc()
    game.rendering_service.align_camera_with_viewport(camera_sensor=cam,
                                                     camera_components=[comp],
                                                     viewport_desc=vd, widths=1280, heights=720)
    comp.Initialize(); comp.initialize_sp_funcs()
with instance.end_frame():
    pass

instance.step(num_frames=30)  # warm up auto-exposure

for label, (cx, cy, cz), yaw in SHOTS:
    pitch = -90.0 if label in ("ceiling", "floor") else 0.0  # floor + ceiling frame: look straight down
    # override center for the 'floor' shot — we want a slight downward tilt to see all walls + floor
    if label == "floor":
        pitch = -45.0
    with instance.begin_frame():
        cam.K2_SetActorLocationAndRotation(
            NewLocation={"X": cx, "Y": cy, "Z": cz},
            NewRotation={"Roll": 0.0, "Pitch": pitch, "Yaw": yaw},
        )
    with instance.end_frame():
        b = comp.read_pixels()
        img = b["arrays"]["data"][:, :, [0, 1, 2]]
        path = f"{OUT}/{label}.png"
        cv2.imwrite(path, img)
        print(f"[preview] {label} -> {path}", flush=True)

instance.close(force=True)
print(f"PREVIEW DONE ({len(SHOTS)} frames) -> {OUT}", flush=True)
