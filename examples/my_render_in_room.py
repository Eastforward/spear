"""Render an asset inside a self-contained gpuRIR-style room (5.2x4.4x2.8m).
Spawns 6 Cube surfaces (floor/4 walls/ceiling) + a point light (built as BP_Room* by
my_build_room.py) high at z=1000 to avoid map geometry, then the auto-fit asset on the
room floor, and orbits the camera. Clean, lit, fully INDOOR — no scene clutter.

Usage: python my_render_in_room.py [<BP>] [<name>] [<target_cm>]
"""
import os, math, subprocess, sys, json, spear, cv2

ASSET_BP = sys.argv[1] if len(sys.argv) > 1 else "/Game/MyAssets/Blueprints/BP_Clock.BP_Clock_C"
NAME = sys.argv[2] if len(sys.argv) > 2 else "Clock"
TARGET = float(sys.argv[3]) if len(sys.argv) > 3 else 35.0
OUT = f"/data/jzy/code/SPEAR/tmp/render_{NAME}"
os.makedirs(OUT, exist_ok=True)
EXEC = "/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh"
MAP = "/Game/StarterContent/Maps/Minimal_Default"
FLOOR_Z = 1000.0   # room spawned high -> no map geometry inside
PEDESTAL_H = 70.0  # cm — elevate small assets to viewing height (their pivot is at center)
RES_W, RES_H = 1280, 720

# room surfaces: (BP, x, y, z_offset_from_FLOOR_Z) in cm
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
    # room (enclosed, lit, high above map)
    for bp, x, y, z in ROOM:
        c = game.unreal_service.load_class(uclass="AActor", name=bp)
        game.unreal_service.spawn_actor(uclass=c, location={"X": float(x), "Y": float(y), "Z": FLOOR_Z + float(z)})
    lc = game.unreal_service.load_class(uclass="AActor", name=ROOM_LIGHT[0])
    light_actor = game.unreal_service.spawn_actor(uclass=lc, location={"X": float(ROOM_LIGHT[1]), "Y": float(ROOM_LIGHT[2]), "Z": FLOOR_Z + float(ROOM_LIGHT[3])})
    try:  # dim the light at runtime (4k lumens blew out; proxy may or may not allow this)
        lcomp = game.unreal_service.get_component_by_class(actor=light_actor, uclass="UPointLightComponent")
        lcomp.set_editor_property("intensity", 800.0)
        print("[light] runtime intensity -> 800 lm", flush=True)
    except Exception as e:
        print(f"[light] runtime set failed ({e}); using BP default", flush=True)
    # asset on the room floor
    bp_asset = game.unreal_service.load_class(uclass="AActor", name=ASSET_BP)
    asset = game.unreal_service.spawn_actor(uclass=bp_asset, location={"X": 0.0, "Y": 0.0, "Z": FLOOR_Z})
    # camera
    # (pedestal dropped — it hid the clock; floor placement works)
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
    R = 2.5 * TARGET
    print(f"[autofit] grounded center=({cx:.0f},{cy:.0f},{cz:.0f}) R={R:.0f}", flush=True)
with instance.end_frame():
    pass

instance.step(num_frames=30)
N = 36
camz = cz + R * 0.20  # gentle elevation above center
pitch = -math.degrees(math.atan2(R * 0.20, R))
# orbit the camera around the asset (proven: clock face visible at front angles, no motion blur)
for i in range(N):
    th = 2.0 * math.pi * i / N
    camx, camy = cx + R * math.cos(th), cy + R * math.sin(th)
    yaw = math.degrees(math.atan2(cy - camy, cx - camx))
    with instance.begin_frame():
        cam.K2_SetActorLocationAndRotation(NewLocation={"X": camx, "Y": camy, "Z": camz}, NewRotation={"Roll": 0.0, "Pitch": pitch, "Yaw": yaw})
    with instance.end_frame():
        b = comp.read_pixels()
        img = b["arrays"]["data"][:, :, [0, 1, 2]]
        cv2.imwrite(f"{OUT}/frame_{i:04d}.png", img)
print(f"[render] captured {N}", flush=True)
instance.close(force=True)
subprocess.run(["ffmpeg", "-y", "-framerate", "12", "-i", f"{OUT}/frame_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", f"{OUT}/turntable.mp4"], capture_output=True)
print(f"VIDEO DONE {OUT}/turntable.mp4", flush=True)
