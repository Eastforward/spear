"""Auto-fit turntable: spawn an imported BP, scale to a REALISTIC per-asset size, ground it on
the floor, orbit a camera sized to the asset, capture 36 frames -> mp4. Indoor scene (apartment).

Usage: python my_render_asset.py [<BP_path>] [<name>] [<target_cm>] [<spawn_x>] [<spawn_y>]
  target_cm = realistic max-extent for THIS asset (clock ~35, chair ~100, dog ~80, guitar ~110...).
  spawn_x/y = floor location in the apartment (default 0,0 — move if occluded by furniture).
"""
import os, math, subprocess, sys, spear, cv2

ASSET_BP = sys.argv[1] if len(sys.argv) > 1 else "/Game/MyAssets/Blueprints/BP_Clock.BP_Clock_C"
NAME = sys.argv[2] if len(sys.argv) > 2 else "Clock"
TARGET = float(sys.argv[3]) if len(sys.argv) > 3 else 35.0     # cm — REALISTIC per-asset (clock~35)
SPAWN_X = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
SPAWN_Y = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
OUT = f"/data/jzy/code/SPEAR/tmp/render_{NAME}"
os.makedirs(OUT, exist_ok=True)
EXEC = "/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh"
MAP = "/Game/StarterContent/Maps/Minimal_Default"   # clean lit indoor room (apartment is too cluttered/dark)
RES_W, RES_H = 1280, 720

config = spear.get_config(user_config_files=[])
config.defrost()
config.SPEAR.LAUNCH_MODE = "game"
config.SPEAR.INSTANCE.GAME_EXECUTABLE = EXEC
config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP = True
config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = MAP
config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES = "/etc/vulkan/icd.d/nvidia_icd.json"
config.freeze()
spear.configure_system(config=config)
instance = spear.Instance(config=config)
game = instance.get_game()

# spawn asset + camera
with instance.begin_frame():
    bp = game.unreal_service.load_class(uclass="AActor", name=ASSET_BP)
    asset = game.unreal_service.spawn_actor(uclass=bp, location={"X": SPAWN_X, "Y": SPAWN_Y, "Z": 0.0})
    cam_uclass = game.unreal_service.load_class(uclass="AActor", name="/SpContent/Blueprints/BP_CameraSensor.BP_CameraSensor_C")
    cam = game.unreal_service.spawn_actor(uclass=cam_uclass)
    comp = game.unreal_service.get_component_by_name(actor=cam, component_name="DefaultSceneRoot.final_tone_curve_hdr_", uclass="USpSceneCaptureComponent2D")
    vd = game.rendering_service.get_current_viewport_desc()
    game.rendering_service.align_camera_with_viewport(camera_sensor=cam, camera_components=[comp], viewport_desc=vd, widths=RES_W, heights=RES_H)
    comp.Initialize(); comp.initialize_sp_funcs()
with instance.end_frame():
    pass

instance.step(num_frames=2)

# --- auto-fit from import-time metadata (game-session proxy can't call get_local_bounds) ---
import json as _json
meta = _json.load(open(f"/data/jzy/code/SPEAR/tmp/asset_meta/{NAME}.json"))
ext = meta["ext"]; bmin_z = meta["bmin_z"]; height = meta["height"]
scale = TARGET / max(ext, 1e-3)
with instance.begin_frame():
    asset.SetActorScale3D(NewScale3D={"X": scale, "Y": scale, "Z": scale})
    print(f"[autofit] meta ext={ext:.1f} height={height:.1f} bmin_z={bmin_z:.1f} -> scale={scale:.3f}", flush=True)
with instance.end_frame():
    pass

instance.step(num_frames=2)

with instance.begin_frame():
    # ground: lift so mesh bottom (bmin_z*scale below smc origin) sits at z=0; keep spawn X,Y
    asset.K2_SetActorLocation(NewLocation={"X": SPAWN_X, "Y": SPAWN_Y, "Z": -bmin_z * scale})
    cx, cy, cz = SPAWN_X, SPAWN_Y, (height / 2.0) * scale
    R = 4.0 * TARGET
    print(f"[autofit] grounded center=({cx:.0f},{cy:.0f},{cz:.0f}) R={R:.0f}", flush=True)
with instance.end_frame():
    pass

instance.step(num_frames=30)  # warm up auto-exposure

# --- orbit + capture ---
N = 36
camz = cz + 40.0  # 40cm above center — gentle 3/4 (steep angles cut the top of tall thin objects)
pitch = -math.degrees(math.atan2(40.0, R))
for i in range(N):
    th = 2.0 * math.pi * i / N
    camx, camy = cx + R * math.cos(th), cy + R * math.sin(th)
    yaw = math.degrees(math.atan2(cy - camy, cx - camx))  # point at asset center
    with instance.begin_frame():
        cam.K2_SetActorLocationAndRotation(NewLocation={"X": camx, "Y": camy, "Z": camz},
                                           NewRotation={"Roll": 0.0, "Pitch": pitch, "Yaw": yaw})
    with instance.end_frame():
        b = comp.read_pixels()
        img = b["arrays"]["data"][:, :, [0, 1, 2]]
        cv2.imwrite(f"{OUT}/frame_{i:04d}.png", img)
print(f"[render] captured {N} frames", flush=True)

instance.close(force=True)
subprocess.run(["ffmpeg", "-y", "-framerate", "12", "-i", f"{OUT}/frame_%04d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", f"{OUT}/turntable.mp4"],
               capture_output=True)
print(f"VIDEO DONE {OUT}/turntable.mp4", flush=True)
