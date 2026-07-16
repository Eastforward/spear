"""Editor-time scan of apartment_0000 for window-facing surfaces.

Goal: figure out which world-axis the apartment windows live on, so my_build_room.py
can mark that side specially (or skip modeling the window and just keep a closed box).

Approach: enumerate every actor in the level via EditorLevelLibrary.get_all_level_actors(),
filter for StaticMeshActor whose StaticMesh materials include a window-y MI (Casement /
Glass / VinylFrame / Window / DoorGlass).  Print world location + forward yaw for each.

Run:
  python tools/run_editor_script.py --unreal-engine-dir /data/UE_5.5 \\
       --script "$(pwd)/examples/inspect_apartment_windows.py" \\
       --launch-mode full --render-offscreen \\
       --map /Game/SPEAR/Scenes/apartment_0000/Maps/apartment_0000
"""
import math, spear, unreal


WINDOW_MAT_HINTS = ("Casement", "Glass", "VinylFrame", "Window", "DoorGlass", "Door_Glass")


def forward_from_yaw(yaw_deg):
    r = math.radians(yaw_deg)
    return (math.cos(r), math.sin(r))


def main():
    spear.log("[inspect] enumerating level actors ...")
    actors = unreal.EditorLevelLibrary.get_all_level_actors()
    spear.log(f"[inspect] total level actors: {len(actors)}")

    window_actors = []
    sma_count = 0
    for actor in actors:
        if not isinstance(actor, unreal.StaticMeshActor):
            continue
        sma_count += 1
        smc = actor.static_mesh_component
        if smc is None:
            continue
        mats = smc.get_editor_property("materials")
        if not mats:
            continue
        for entry in mats:
            mat = entry.get_editor_property("material_interface")
            if mat is None:
                continue
            path = mat.get_path_name()
            if any(hint in path for hint in WINDOW_MAT_HINTS):
                loc = actor.get_actor_location()
                rot = actor.get_actor_rotation()
                yaw = rot.get("Yaw", 0.0)
                fwd = forward_from_yaw(yaw)
                window_actors.append((actor.get_name(), loc, rot, yaw, fwd, path))
                break

    spear.log(f"[inspect] StaticMeshActors visited: {sma_count}")
    spear.log(f"[inspect] window-y actors: {len(window_actors)}")
    for name, loc, rot, yaw, fwd, path in window_actors:
        spear.log(f"  {name:35s} loc=({loc.x:.0f},{loc.y:.0f},{loc.z:.0f})  "
                  f"yaw={yaw:6.1f}  fwd=({fwd[0]:+.2f},{fwd[1]:+.2f})  mat={path.split('/')[-1]}")

    if window_actors:
        ax = sum(1 for _, _, _, _, fwd, _ in window_actors if abs(fwd[0]) > abs(fwd[1]))
        ay = len(window_actors) - ax
        spear.log(f"[inspect] majority axis: {'X' if ax > ay else 'Y'} "
                  f"(X-facing: {ax}, Y-facing: {ay})")


if __name__ == "__main__":
    main()
