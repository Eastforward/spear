"""Test whether Floor_400x400 mesh (starter content) gives correct wood-plank
scale when we scale it to the shoebox room footprint (5.2 x 4.4 m -> scale 1.3 x 1.1).

Also compare against Cube.Cube (current, wrong tiling) for each candidate material.
"""
import os, sys
sys.path.insert(0, '/data/jzy/code/SPEAR/examples')

from render_in_gpurir_room import (
    configure_gpurir_instance, compute_shoebox_room_layout,
    spawn_directional_light, spawn_sky, spawn_point_light,
    WALL_MATERIAL, M2CM,
)
from render_in_apartment import spawn_camera, read_frame

OUT = '/tmp/probe_floor_mesh'
os.makedirs(OUT, exist_ok=True)

FLOOR_MESHES = [
    ("cube",         "/Engine/BasicShapes/Cube.Cube",              5.2, 4.4, 0.1),  # current
    ("floor400",     "/Game/StarterContent/Architecture/Floor_400x400.Floor_400x400", 5.2/4.0, 4.4/4.0, 1.0),
]

MATERIALS = [
    ("walnut_polished", "/Game/StarterContent/Materials/M_Wood_Floor_Walnut_Polished.M_Wood_Floor_Walnut_Polished"),
    ("oak",             "/Game/StarterContent/Materials/M_Wood_Oak.M_Wood_Oak"),
    ("pine",            "/Game/StarterContent/Materials/M_Wood_Pine.M_Wood_Pine"),
    ("walnut_worn",     "/Game/StarterContent/Materials/M_Wood_Floor_Walnut_Worn.M_Wood_Floor_Walnut_Worn"),
]


def render_one(mesh_short, mesh_path, sx, sy, sz, mat_short, mat_path):
    import cv2
    instance = configure_gpurir_instance(rpc_port=39002)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            for cls in ("APlayerStart","ADefaultPawn","ASpectatorPawn",
                        "AGameplayDebuggerCategoryReplicator",
                        "AGameplayDebuggerPlayerManager"):
                try:
                    for a in game.unreal_service.find_actors_by_class(uclass=cls):
                        game.unreal_service.destroy_actor(actor=a)
                except Exception: pass

            # Spawn floor with selected mesh + material
            floor_mesh = game.unreal_service.load_object(uclass="UStaticMesh", name=mesh_path)
            floor_mat  = game.unreal_service.load_object(uclass="UMaterialInterface", name=mat_path)
            floor_actor = game.unreal_service.spawn_actor(
                uclass="AStaticMeshActor",
                location={"X": 260.0, "Y": 220.0, "Z": -0.05*M2CM if mesh_short=="cube" else 0.0},
                spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
            )
            smc = game.unreal_service.get_component_by_class(actor=floor_actor, uclass="UStaticMeshComponent")
            smc.SetMobility(NewMobility="Movable")
            smc.SetStaticMesh(NewMesh=floor_mesh)
            smc.SetMaterial(ElementIndex=0, Material=floor_mat)
            floor_actor.SetActorScale3D(NewScale3D={"X": float(sx), "Y": float(sy), "Z": float(sz)})

            # Spawn 4 walls + ceiling with plain wall material (from the layout, minus floor)
            for p in compute_shoebox_room_layout(room_size_m=(5.2, 4.4, 2.8)):
                if p["name"] == "floor": continue
                cube_mesh = game.unreal_service.load_object(uclass="UStaticMesh", name="/Engine/BasicShapes/Cube.Cube")
                wall_mat  = game.unreal_service.load_object(uclass="UMaterialInterface", name=WALL_MATERIAL)
                a = game.unreal_service.spawn_actor(
                    uclass="AStaticMeshActor",
                    location={"X": float(p["location_cm"][0]),
                              "Y": float(p["location_cm"][1]),
                              "Z": float(p["location_cm"][2])},
                    spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
                )
                smc2 = game.unreal_service.get_component_by_class(actor=a, uclass="UStaticMeshComponent")
                smc2.SetMobility(NewMobility="Movable")
                smc2.SetStaticMesh(NewMesh=cube_mesh)
                smc2.SetMaterial(ElementIndex=0, Material=wall_mat)
                a.SetActorScale3D(NewScale3D={"X": float(p["scale"][0]), "Y": float(p["scale"][1]), "Z": float(p["scale"][2])})

            spawn_sky(game=game)
            spawn_directional_light(game=game, yaw_deg=-90.0, pitch_deg=-30.0, intensity_lux=10.0)
            spawn_point_light(game=game, x_cm=260.0, y_cm=220.0, z_cm=265.0, intensity_lumens=6000.0, attenuation_cm=600.0)
            cam, comp = spawn_camera(game=game, width=1280, height=720)
        with instance.end_frame(): pass
        instance.step(num_frames=30)
        with instance.begin_frame():
            cam.K2_SetActorLocationAndRotation(
                NewLocation={"X": 260.0, "Y": 240.0, "Z": 73.4},
                NewRotation={"Roll": 0.0, "Pitch": -11.3, "Yaw": 90.0},
                bSweep=False, bTeleport=True,
            )
        with instance.end_frame(): pass
        instance.step(num_frames=25)
        with instance.begin_frame(): pass
        with instance.end_frame():
            import cv2
            out = os.path.join(OUT, f"{mesh_short}_{mat_short}.png")
            cv2.imwrite(out, read_frame(comp))
            print(f"    -> {out}", flush=True)
    finally:
        instance.close(force=True)


if __name__ == "__main__":
    for mesh_short, mesh_path, sx, sy, sz in FLOOR_MESHES:
        for mat_short, mat_path in MATERIALS:
            print(f"\n=== {mesh_short} + {mat_short} ===", flush=True)
            render_one(mesh_short, mesh_path, sx, sy, sz, mat_short, mat_path)
    print("PROBE_DONE", flush=True)
