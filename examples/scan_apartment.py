"""One-frame high oblique view of apartment_0000 to find an open floor spot for asset placement."""
import os, spear, cv2
EXEC = "/data/jzy/code/SPEAR/cpp/unreal_projects/SpearSim/Standalone-Development/Linux/SpearSim.sh"
config = spear.get_config(user_config_files=[]); config.defrost()
config.SPEAR.LAUNCH_MODE = "game"; config.SPEAR.INSTANCE.GAME_EXECUTABLE = EXEC
config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.OVERRIDE_GAME_DEFAULT_MAP = True
config.SP_SERVICES.INITIALIZE_ENGINE_SERVICE.GAME_DEFAULT_MAP = "/Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000"
config.SPEAR.ENVIRONMENT_VARS.VK_ICD_FILENAMES = "/etc/vulkan/icd.d/nvidia_icd.json"
config.freeze(); spear.configure_system(config=config)
instance = spear.Instance(config=config); game = instance.get_game()
with instance.begin_frame():
    cam_uclass = game.unreal_service.load_class(uclass="AActor", name="/SpContent/Blueprints/BP_CameraSensor.BP_CameraSensor_C")
    cam = game.unreal_service.spawn_actor(uclass=cam_uclass)
    comp = game.unreal_service.get_component_by_name(actor=cam, component_name="DefaultSceneRoot.final_tone_curve_hdr_", uclass="USpSceneCaptureComponent2D")
    game.rendering_service.align_camera_with_viewport(camera_sensor=cam, camera_components=[comp], viewport_desc=game.rendering_service.get_current_viewport_desc(), widths=1280, heights=720)
    comp.Initialize(); comp.initialize_sp_funcs()
    cam.K2_SetActorLocationAndRotation(NewLocation={"X": 0.0, "Y": 0.0, "Z": 800.0}, NewRotation={"Roll": 0.0, "Pitch": -90.0, "Yaw": 0.0})
with instance.end_frame(): pass
instance.step(num_frames=20)
with instance.begin_frame(): pass
with instance.end_frame():
    b = comp.read_pixels(); img = b["arrays"]["data"][:, :, [0, 1, 2]]
    cv2.imwrite("/data/jzy/code/SPEAR/tmp/apartment_layout.png", img)
instance.close(force=True)
print("SCAN_DONE /data/jzy/code/SPEAR/tmp/apartment_layout.png", flush=True)
