"""Batch turntable: render ALL imported AudioSet assets inside the room, ONE game session.
Spawns the room once, then per asset: load BP, auto-fit to a realistic per-category size,
orbit + capture 36 frames -> mp4, destroy. Efficient (one game launch for all ~117).

Usage: python batch_render.py
"""
import os, math, json, subprocess, spear, cv2

MANIFEST = json.load(open("/data/jzy/code/Hunyuan3D-2.1/demos/audioset/manifest.json"))
META_DIR = "/data/jzy/code/SPEAR/tmp/asset_meta"
OUT_BASE = "/data/jzy/code/SPEAR/tmp/audioset_turntables"
BP_BASE = "/Game/MyAssets/Audioset/Blueprints"
os.makedirs(OUT_BASE, exist_ok=True)

# realistic per-category max-extent (cm) — clock<chair as the user asked
SIZE_BY_CAT = {"animal": 80, "instrument": 110, "household_appliance": 80,
               "electronics": 40, "furniture": 100, "tool": 50, "other": 60, "vehicle": 60}

EXEC = "/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh"
MAP = "/Game/StarterContent/Maps/Minimal_Default"
FLOOR_Z = 1000.0
RES_W, RES_H = 1280, 720
ROOM = [
    ("/Game/MyAssets/Room/BP_RoomFloor.BP_RoomFloor_C", 0, 0, -5),
    ("/Game/MyAssets/Room/BP_RoomCeiling.BP_RoomCeiling_C", 0, 0, 285),
    ("/Game/MyAssets/Room/BP_RoomWallXP.BP_RoomWallXP_C", 260, 0, 140),
    ("/Game/MyAssets/Room/BP_RoomWallXN.BP_RoomWallXN_C", -260, 0, 140),
    ("/Game/MyAssets/Room/BP_RoomWallYP.BP_RoomWallYP_C", 0, 220, 140),
    ("/Game/MyAssets/Room/BP_RoomWallYN.BP_RoomWallYN_C", 0, -220, 140),
]
ROOM_LIGHT = ("/Game/MyAssets/Room/BP_RoomLight.BP_RoomLight_C", 0, 0, 250)

config = spear.get_config(user_config_files=[]); config.defrost()
config.SPEAR.LAUNCH_MODE = "game"; config.SPEAR.INSTANCE.GAME_EXECUTABLE = EXEC
config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP = True
config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = MAP
config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES = "/etc/vulkan/icd.d/nvidia_icd.json"
config.freeze(); spear.configure_system(config=config)
instance = spear.Instance(config=config); game = instance.get_game()

# spawn room + camera once
with instance.begin_frame():
    for bp, x, y, z in ROOM:
        c = game.unreal_service.load_class(uclass="AActor", name=bp)
        game.unreal_service.spawn_actor(uclass=c, location={"X": float(x), "Y": float(y), "Z": FLOOR_Z + float(z)})
    lc = game.unreal_service.load_class(uclass="AActor", name=ROOM_LIGHT[0])
    game.unreal_service.spawn_actor(uclass=lc, location={"X": float(ROOM_LIGHT[1]), "Y": float(ROOM_LIGHT[2]), "Z": FLOOR_Z + float(ROOM_LIGHT[3])})
    cam_uclass = game.unreal_service.load_class(uclass="AActor", name="/SpContent/Blueprints/BP_CameraSensor.BP_CameraSensor_C")
    cam = game.unreal_service.spawn_actor(uclass=cam_uclass)
    comp = game.unreal_service.get_component_by_name(actor=cam, component_name="DefaultSceneRoot.final_tone_curve_hdr_", uclass="USpSceneCaptureComponent2D")
    vd = game.rendering_service.get_current_viewport_desc()
    game.rendering_service.align_camera_with_viewport(camera_sensor=cam, camera_components=[comp], viewport_desc=vd, widths=RES_W, heights=RES_H)
    comp.Initialize(); comp.initialize_sp_funcs()
with instance.end_frame():
    pass
instance.step(num_frames=20)

done, failed = 0, 0
for ci, c in enumerate(MANIFEST["classes"]):
    name = c["name"]; cat = c.get("category", "other"); target = SIZE_BY_CAT.get(cat, 60)
    meta_path = os.path.join(META_DIR, f"{name}.json")
    bp_path = f"{BP_BASE}/{name}/BP_{name}.BP_{name}_C"
    if not os.path.exists(meta_path):
        print(f"[batch] {ci+1} {name}: no meta, skip", flush=True); failed += 1; continue
    meta = json.load(open(meta_path))
    ext, bmin_z, height = meta["ext"], meta["bmin_z"], meta["height"]
    scale = target / max(ext, 1e-3)
    out_dir = os.path.join(OUT_BASE, name); os.makedirs(out_dir, exist_ok=True)
    try:
        with instance.begin_frame():
            bp = game.unreal_service.load_class(uclass="AActor", name=bp_path)
            asset = game.unreal_service.spawn_actor(uclass=bp, location={"X": 0.0, "Y": 0.0, "Z": FLOOR_Z})
        with instance.end_frame():
            pass
        instance.step(num_frames=2)
        with instance.begin_frame():
            asset.SetActorScale3D(NewScale3D={"X": scale, "Y": scale, "Z": scale})
        with instance.end_frame():
            pass
        instance.step(num_frames=2)
        with instance.begin_frame():
            asset.K2_SetActorLocation(NewLocation={"X": 0.0, "Y": 0.0, "Z": FLOOR_Z - bmin_z * scale})
            cx, cy, cz = 0.0, 0.0, FLOOR_Z + (height / 2.0) * scale
            R = 2.5 * target
        with instance.end_frame():
            pass
        instance.step(num_frames=20)
        camz = cz + R * 0.20; pitch = -math.degrees(math.atan2(R * 0.20, R))
        for i in range(36):
            th = 2.0 * math.pi * i / 36
            camx, camy = cx + R * math.cos(th), cy + R * math.sin(th)
            yaw = math.degrees(math.atan2(cy - camy, cx - camx))
            with instance.begin_frame():
                cam.K2_SetActorLocationAndRotation(NewLocation={"X": camx, "Y": camy, "Z": camz}, NewRotation={"Roll": 0.0, "Pitch": pitch, "Yaw": yaw})
            with instance.end_frame():
                b = comp.read_pixels(); img = b["arrays"]["data"][:, :, [0, 1, 2]]
                cv2.imwrite(f"{out_dir}/frame_{i:04d}.png", img)
        with instance.begin_frame():
            game.unreal_service.destroy_actor(actor=asset)
        with instance.end_frame():
            pass
        subprocess.run(["ffmpeg", "-y", "-framerate", "12", "-i", f"{out_dir}/frame_%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", f"{out_dir}/turntable.mp4"], capture_output=True)
        done += 1
        print(f"[batch] {ci+1}/{len(MANIFEST['classes'])} {name} ({cat}, {target}cm) -> {out_dir}/turntable.mp4", flush=True)
    except Exception as e:
        print(f"[batch] {name}: FAIL {type(e).__name__}: {e}", flush=True); failed += 1
        try:
            with instance.begin_frame():
                game.unreal_service.destroy_actor(actor=asset)
            with instance.end_frame():
                pass
        except Exception:
            pass

instance.close(force=True)
print(f"[batch] ALL DONE: {done} rendered, {failed} failed -> {OUT_BASE}/", flush=True)
