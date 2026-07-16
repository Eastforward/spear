"""Build a self-contained gpuRIR-style shoebox room in SpearSim (editor script).
Creates 6 StaticMeshActor BPs (floor, 4 walls, ceiling — Engine Cube scaled) with
StarterContent materials + 1 PointLight BP for interior lighting.
Spawn all 7 at runtime to get a clean, lit, enclosed indoor room (blocks scene clutter).

Room: 5.2 x 4.4 x 2.8 m (realistic living room, matches v77 compose_scene / gpuRIR).
"""
import os, posixpath, spear, unreal

ROOM_W, ROOM_D, ROOM_H, THICK = 5.2, 4.4, 2.8, 0.1   # meters
CUBE = "/Engine/BasicShapes/Cube.Cube"
# Use SPEAR apartment_0000's actual structural materials (MI_Walls / MI_Floor) — same look as the
# default SPEAR room, minus furniture. See README in /Game/SPEAR/Scenes/apartment_0000/Materials.
FLOOR_MAT = "/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Floor.MI_Floor"
WALL_MAT  = "/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Walls.MI_Walls"
BP_DIR = "/Game/MyAssets/Room"

eas = unreal.get_editor_subsystem(unreal.EditorAssetSubsystem)
eal = unreal.EditorAssetLibrary
cube_mesh = unreal.load_asset(CUBE)
floor_m = unreal.load_asset(FLOOR_MAT)
wall_m = unreal.load_asset(WALL_MAT)

# (name, scale_xyz_m, spawn_xyz_m, material)
SURFACES = [
    ("Floor",   (ROOM_W, ROOM_D, THICK), (0.0, 0.0, -THICK/2),            floor_m),
    ("Ceiling", (ROOM_W, ROOM_D, THICK), (0.0, 0.0, ROOM_H + THICK/2),   wall_m),
    ("WallXP",  (THICK, ROOM_D, ROOM_H), (ROOM_W/2, 0.0, ROOM_H/2),      wall_m),
    ("WallXN",  (THICK, ROOM_D, ROOM_H), (-ROOM_W/2, 0.0, ROOM_H/2),     wall_m),
    ("WallYP",  (ROOM_W, THICK, ROOM_H), (0.0, ROOM_D/2, ROOM_H/2),      wall_m),
    ("WallYN",  (ROOM_W, THICK, ROOM_H), (0.0, -ROOM_D/2, ROOM_H/2),     wall_m),
]

if __name__ == "__main__":
    for name, scale, loc, mat in SURFACES:
        bp_name = f"BP_Room{name}"; bp_path = posixpath.join(BP_DIR, bp_name)
        if eal.does_asset_exist(bp_path):
            eal.delete_asset(bp_path)
        bp = spear.editor.create_blueprint_asset(asset_name=bp_name, package_dir=BP_DIR, parent_class=unreal.StaticMeshActor)
        descs = spear.editor.get_subobject_descs_for_blueprint_asset(blueprint_asset=bp)
        smc = descs[1]["object"]
        smc.set_editor_property("static_mesh", cube_mesh)
        smc.set_editor_property("relative_scale3d", unreal.Vector(x=scale[0], y=scale[1], z=scale[2]))
        smc.set_material(element_index=0, material=mat)
        eas.save_loaded_asset(asset_to_save=bp)
        spear.log(f"[room] {bp_name} scale={scale}")

    # interior point light (enclosed room needs its own light)
    light_path = posixpath.join(BP_DIR, "BP_RoomLight")
    if eal.does_asset_exist(light_path):
        eal.delete_asset(light_path)
    light_bp = spear.editor.create_blueprint_asset(asset_name="BP_RoomLight", package_dir=BP_DIR, parent_class=unreal.PointLight)
    ldescs = spear.editor.get_subobject_descs_for_blueprint_asset(blueprint_asset=light_bp)
    light_comp = ldescs[1]["object"]
    light_comp.set_editor_property("intensity_units", unreal.LightUnits.LUMENS)
    light_comp.set_editor_property("intensity", 200.0)  # lumens (600 still bloomed; small enclosed room needs little)
    eas.save_loaded_asset(asset_to_save=light_bp)
    spear.log("[room] DONE: 6 surfaces + BP_RoomLight")
