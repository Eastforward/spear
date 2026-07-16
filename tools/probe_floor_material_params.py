"""Probe the exposed scalar/vector parameters on candidate floor materials.

For each candidate:
  - load_object(UMaterialInterface)
  - call GetScalarParameterNames / GetVectorParameterNames / GetTextureParameterNames
  - print the names
  - if a 'Tiling' / 'UVScale' / 'UV_Scale' / 'TilingScale' scalar exists, try setting a large value on an actual spawned Cube in the scene and render one frame to prove we can tile the wood planks smaller.

Output: printed param names + one rendered frame per candidate under /tmp/probe_tiling/.
"""
import os, sys
sys.path.insert(0, '/data/jzy/code/SPEAR/examples')

from render_in_gpurir_room import (
    configure_gpurir_instance, compute_shoebox_room_layout,
    spawn_directional_light, spawn_sky, spawn_point_light,
    WALL_MATERIAL, M2CM, WALL_THICKNESS_M,
)
from render_in_apartment import spawn_camera, read_frame

OUT = '/tmp/probe_tiling'
os.makedirs(OUT, exist_ok=True)

CANDIDATES = [
    ("mi_floor",        "/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Floor.MI_Floor"),
    ("walnut_polished", "/Game/StarterContent/Materials/M_Wood_Floor_Walnut_Polished.M_Wood_Floor_Walnut_Polished"),
    ("oak",             "/Game/StarterContent/Materials/M_Wood_Oak.M_Wood_Oak"),
    ("pine",            "/Game/StarterContent/Materials/M_Wood_Pine.M_Wood_Pine"),
    ("walnut_worn",     "/Game/StarterContent/Materials/M_Wood_Floor_Walnut_Worn.M_Wood_Floor_Walnut_Worn"),
    ("walnut",          "/Game/StarterContent/Materials/M_Wood_Walnut.M_Wood_Walnut"),
]


def probe(mat_short, mat_path, tiling_test_value):
    import cv2
    instance = configure_gpurir_instance(rpc_port=39002)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            # kill spawn cube
            for cls in ("APlayerStart","ADefaultPawn","ASpectatorPawn",
                        "AGameplayDebuggerCategoryReplicator",
                        "AGameplayDebuggerPlayerManager"):
                try:
                    for a in game.unreal_service.find_actors_by_class(uclass=cls):
                        game.unreal_service.destroy_actor(actor=a)
                except Exception:
                    pass

            # spawn just floor+walls with the material under test on floor
            for p in compute_shoebox_room_layout(room_size_m=(5.2, 4.4, 2.8)):
                cube_mesh = game.unreal_service.load_object(uclass="UStaticMesh", name="/Engine/BasicShapes/Cube.Cube")
                material = game.unreal_service.load_object(uclass="UMaterialInterface", name=(mat_path if p["name"]=="floor" else WALL_MATERIAL))
                actor = game.unreal_service.spawn_actor(
                    uclass="AStaticMeshActor",
                    location={"X": float(p["location_cm"][0]),
                              "Y": float(p["location_cm"][1]),
                              "Z": float(p["location_cm"][2])},
                    spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
                )
                smc = game.unreal_service.get_component_by_class(actor=actor, uclass="UStaticMeshComponent")
                smc.SetMobility(NewMobility="Movable")
                smc.SetStaticMesh(NewMesh=cube_mesh)
                actor.SetActorScale3D(
                    NewScale3D={"X": float(p["scale"][0]),
                                "Y": float(p["scale"][1]),
                                "Z": float(p["scale"][2])},
                )
                if p["name"] == "floor":
                    # Create MID on element 0 so we can override params without
                    # touching the shared material asset.
                    mid = smc.CreateAndSetMaterialInstanceDynamic(ElementIndex=0)
                    # Probe parameter names (both float and vector, both API styles)
                    for meth in ("GetScalarParameterNames", "GetVectorParameterNames",
                                 "GetTextureParameterNames", "GetStaticSwitchParameterNames"):
                        try:
                            names = getattr(mid, meth)()
                            print(f"    {mat_short:20s} {meth} -> {names}", flush=True)
                        except Exception as e:
                            print(f"    {mat_short:20s} {meth} FAILED: {e}", flush=True)
                    # Try setting various probable scalar tiling names
                    for pname in ("Tiling", "UVScale", "UV_Scale", "TilingScale",
                                  "TextureScale", "UTiling", "VTiling", "UV Scale"):
                        try:
                            mid.SetScalarParameterValue(ParameterName=pname, Value=float(tiling_test_value))
                            print(f"    {mat_short:20s} SetScalar({pname!r}) OK", flush=True)
                        except Exception as e:
                            pass  # too noisy if we log every miss
                else:
                    smc.SetMaterial(ElementIndex=0, Material=material)

            spawn_sky(game=game)
            spawn_directional_light(game=game, yaw_deg=-90.0, pitch_deg=-30.0, intensity_lux=10.0)
            spawn_point_light(game=game, x_cm=260.0, y_cm=220.0, z_cm=265.0, intensity_lumens=6000.0, attenuation_cm=600.0)
            cam, comp = spawn_camera(game=game, width=1280, height=720)
        with instance.end_frame(): pass

        instance.step(num_frames=30)

        with instance.begin_frame():
            # Point camera at frame_0027 pose (facing window from south)
            cam.K2_SetActorLocationAndRotation(
                NewLocation={"X": 260.0, "Y": 240.0, "Z": 73.4},
                NewRotation={"Roll": 0.0, "Pitch": -11.3, "Yaw": 90.0},
                bSweep=False, bTeleport=True,
            )
        with instance.end_frame(): pass
        instance.step(num_frames=25)
        with instance.begin_frame(): pass
        with instance.end_frame():
            out = os.path.join(OUT, f"{mat_short}_tiling{int(tiling_test_value)}.png")
            cv2.imwrite(out, read_frame(comp))
            print(f"    -> {out}", flush=True)
    finally:
        instance.close(force=True)


if __name__ == "__main__":
    for short, path in CANDIDATES:
        print(f"\n=== {short} ===", flush=True)
        probe(short, path, tiling_test_value=8.0)
    print("PROBE_DONE", flush=True)
