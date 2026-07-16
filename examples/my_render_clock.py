"""Pilot render: spawn the imported BP_Clock in apartment_0000, orbit a camera, capture frames -> mp4.
Run in a game session via the spear-env python (NOT the editor). Under Xvfb."""
import os, math, subprocess, spear, cv2

OUT = "/data/jzy/code/SPEAR/tmp/render_clock"
os.makedirs(OUT, exist_ok=True)
ASSET_BP = "/Game/MyAssets/Blueprints/BP_Clock.BP_Clock_C"
EXEC = "/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh"
MAP = "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"

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

with instance.begin_frame():
    bp_asset = game.unreal_service.load_class(uclass="AActor", name=ASSET_BP)
    asset = game.unreal_service.spawn_actor(uclass=bp_asset, location={"X": 0.0, "Y": 0.0, "Z": 0.0})
    cam_uclass = game.unreal_service.load_class(uclass="AActor", name="/SpContent/Blueprints/BP_CameraSensor.BP_CameraSensor_C")
    cam = game.unreal_service.spawn_actor(uclass=cam_uclass)
    comp = game.unreal_service.get_component_by_name(actor=cam, component_name="DefaultSceneRoot.final_tone_curve_hdr_", uclass="USpSceneCaptureComponent2D")
    vd = game.rendering_service.get_current_viewport_desc()
    game.rendering_service.align_camera_with_viewport(camera_sensor=cam, camera_components=[comp], viewport_desc=vd, widths=640, heights=480)
    comp.Initialize()
    comp.initialize_sp_funcs()
with instance.end_frame():
    pass

instance.step(num_frames=30)  # warm up (auto-exposure)

N, R, H, PITCH = 36, 300.0, 200.0, -25.0
for i in range(N):
    th = 2.0 * math.pi * i / N
    cx, cy = R * math.cos(th), R * math.sin(th)
    yaw = math.degrees(math.atan2(0.0 - cy, 0.0 - cx))  # point at origin
    loc = {"X": cx, "Y": cy, "Z": H}
    rot = {"Roll": 0.0, "Pitch": PITCH, "Yaw": yaw}
    with instance.begin_frame():
        cam.K2_SetActorLocationAndRotation(NewLocation=loc, NewRotation=rot)
    with instance.end_frame():
        bundle = comp.read_pixels()
        img = bundle["arrays"]["data"][:, :, [0, 1, 2]]  # BGR, drop alpha
        cv2.imwrite(f"{OUT}/frame_{i:04d}.png", img)
        print(f"frame {i+1}/{N} yaw={yaw:.0f}", flush=True)

instance.close(force=True)
subprocess.run(["ffmpeg", "-y", "-framerate", "12", "-i", f"{OUT}/frame_%04d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", f"{OUT}/turntable.mp4"],
               capture_output=True)
print(f"VIDEO DONE {OUT}/turntable.mp4", flush=True)
