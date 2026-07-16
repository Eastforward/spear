"""Render a gallery of 8 candidate wall materials as a single-shot comparison.

Each material is applied to a 5 m x 2.8 m Cube "sample wall". All 8 walls are
lined up in a row (2 m spacing) in an empty map, lit by SkyAtmosphere +
DirectionalLight + SkyLight. The camera captures ONE close-up frame per wall
plus one overview shot, all inside a single SpearSim session (~2 min total).

Use spear-env:
  DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
    /data/jzy/miniconda3/envs/spear-env/bin/python examples/material_sample_gallery.py

Output: /data/jzy/code/SPEAR/tmp/material_gallery/{sample_<i>_<name>.png, overview.png, gallery.png}
"""

import math
import os
import sys


_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from render_in_gpurir_room import (  # noqa: E402
    CUBE_MESH,
    EMPTY_MAP,
    SPEARSIM_EXECUTABLE,
    configure_gpurir_instance,
    spawn_directional_light,
    spawn_sky,
)
from render_in_apartment import read_frame, spawn_camera  # noqa: E402


OUTPUT_DIR = "/data/jzy/code/SPEAR/tmp/material_gallery"

# 8 candidates: (label, material_path, source_note)
CANDIDATES = [
    ("00_MI_Walls",             "/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Walls.MI_Walls",             "apartment default (baseline)"),
    ("01_MI_LivingRoom_Table",  "/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_LivingRoom_Table.MI_LivingRoom_Table", "apartment furniture"),
    ("02_MI_Cabinet_Vase_Mir",  "/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Cabinet_Vase_Mirror.MI_Cabinet_Vase_Mirror", "apartment cabinet"),
    ("03_MI_Fireplace",         "/Game/SPEAR/Scenes/apartment_0000/Materials/Material_Instances/MI_Fireplace.MI_Fireplace",     "apartment fireplace"),
    ("04_M_Basic_Wall",         "/Game/StarterContent/Materials/M_Basic_Wall.M_Basic_Wall",                                     "engine starter (recommend)"),
    ("05_M_Concrete_Panels",    "/Game/StarterContent/Materials/M_Concrete_Panels.M_Concrete_Panels",                           "engine starter (loft)"),
    ("06_M_Brick_Clay_Beveled", "/Game/StarterContent/Materials/M_Brick_Clay_Beveled.M_Brick_Clay_Beveled",                     "engine starter (brick)"),
    ("07_M_Wood_Pine",          "/Game/StarterContent/Materials/M_Wood_Pine.M_Wood_Pine",                                       "engine starter (wood)"),
]

# Sample-wall geometry (meters)
WALL_W_M = 3.0        # wall x-extent
WALL_H_M = 2.4        # wall z-extent
WALL_THICK_M = 0.1    # wall y-extent
WALL_SPACING_M = 4.0  # x-distance between adjacent wall centers
WALL_ROW_Y_M = 0.0

WIDTH = 512
HEIGHT = 512
WARMUP_FRAMES = 40
PER_SAMPLE_WARMUP = 12
RPC_PORT = 39002


def spawn_sample_wall(game, *, label, material_path, center_x_m):
    """Spawn a single flat wall (Cube scaled) at the given x, tagged with label."""
    cube_mesh = game.unreal_service.load_object(uclass="UStaticMesh", name=CUBE_MESH)
    material = game.unreal_service.load_object(uclass="UMaterialInterface", name=material_path)
    actor = game.unreal_service.spawn_actor(
        uclass="AStaticMeshActor",
        location={
            "X": center_x_m * 100.0,
            "Y": WALL_ROW_Y_M * 100.0,
            "Z": WALL_H_M * 100.0 / 2.0,
        },
        spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
    )
    smc = game.unreal_service.get_component_by_class(actor=actor, uclass="UStaticMeshComponent")
    smc.SetMobility(NewMobility="Movable")
    smc.SetStaticMesh(NewMesh=cube_mesh)
    smc.SetMaterial(ElementIndex=0, Material=material)
    actor.SetActorScale3D(NewScale3D={"X": WALL_W_M, "Y": WALL_THICK_M, "Z": WALL_H_M})
    game.unreal_service.set_stable_name_for_actor(actor=actor, stable_name=f"MatGallery/{label}")
    return actor


def spawn_ground_plane(game):
    """Big grey ground so shadows look grounded and lighting balances."""
    cube_mesh = game.unreal_service.load_object(uclass="UStaticMesh", name=CUBE_MESH)
    material = game.unreal_service.load_object(
        uclass="UMaterialInterface", name="/Game/StarterContent/Materials/M_Concrete_Poured.M_Concrete_Poured"
    )
    actor = game.unreal_service.spawn_actor(
        uclass="AStaticMeshActor",
        location={"X": 0.0, "Y": 0.0, "Z": -5.0},  # top face at z=0
        spawn_parameters={"SpawnCollisionHandlingOverride": "AlwaysSpawn"},
    )
    smc = game.unreal_service.get_component_by_class(actor=actor, uclass="UStaticMeshComponent")
    smc.SetMobility(NewMobility="Movable")
    smc.SetStaticMesh(NewMesh=cube_mesh)
    smc.SetMaterial(ElementIndex=0, Material=material)
    # 80m x 80m x 0.1m
    actor.SetActorScale3D(NewScale3D={"X": 80.0, "Y": 80.0, "Z": 0.1})
    return actor


def sample_wall_close_up_pose(*, center_x_m):
    """Camera pose that centers on one wall from ~3.5m away, y=+3m offset."""
    cam_x = center_x_m * 100.0
    cam_y = -350.0
    cam_z = WALL_H_M * 100.0 / 2.0
    yaw = 90.0   # facing +y
    pitch = 0.0
    return (
        {"X": cam_x, "Y": cam_y, "Z": cam_z},
        {"Roll": 0.0, "Pitch": pitch, "Yaw": yaw},
    )


def overview_pose(*, n_walls, spacing_m):
    """Camera pose showing all N walls in one frame from further back."""
    center_x_m = (n_walls - 1) * spacing_m / 2.0
    cam_x = center_x_m * 100.0
    cam_y = -1200.0
    cam_z = WALL_H_M * 100.0 * 1.3
    yaw = 90.0
    pitch = -8.0
    return (
        {"X": cam_x, "Y": cam_y, "Z": cam_z},
        {"Roll": 0.0, "Pitch": pitch, "Yaw": yaw},
    )


def _make_gallery_grid(sample_paths, gallery_path):
    """Assemble a 2x4 grid from 8 sample PNGs using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.image import imread

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, entry in zip(axes.flatten(), sample_paths):
        label, path, note = entry
        img = imread(path)
        ax.imshow(img)
        ax.set_title(f"{label}\n{note}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Wall material candidates (2x4 grid) — pick by label number", fontsize=13)
    fig.tight_layout()
    fig.savefig(gallery_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def main():
    import cv2

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    n = len(CANDIDATES)
    x0 = -(n - 1) * WALL_SPACING_M / 2.0
    wall_x_centers = [x0 + i * WALL_SPACING_M for i in range(n)]

    instance = configure_gpurir_instance(rpc_port=RPC_PORT)
    game = instance.get_game()
    sample_records = []  # (label, close_up_path, note)

    try:
        with instance.begin_frame():
            spawn_ground_plane(game)
            spawn_sky(game)
            spawn_directional_light(
                game=game,
                yaw_deg=-100.0,
                pitch_deg=-40.0,
                intensity_lux=10.0,
            )
            for (label, mat_path, _note), x_m in zip(CANDIDATES, wall_x_centers):
                spawn_sample_wall(game=game, label=label, material_path=mat_path, center_x_m=x_m)
            cam, comp = spawn_camera(game=game, width=WIDTH, height=HEIGHT)
        with instance.end_frame():
            pass

        instance.step(num_frames=WARMUP_FRAMES)

        # 1) Close-up per wall
        for (label, mat_path, note), x_m in zip(CANDIDATES, wall_x_centers):
            loc, rot = sample_wall_close_up_pose(center_x_m=x_m)
            with instance.begin_frame():
                cam.K2_SetActorLocationAndRotation(
                    NewLocation=loc, NewRotation=rot, bSweep=False, bTeleport=True
                )
            with instance.end_frame():
                pass
            instance.step(num_frames=PER_SAMPLE_WARMUP)
            with instance.begin_frame():
                pass
            with instance.end_frame():
                path = os.path.join(OUTPUT_DIR, f"sample_{label}.png")
                cv2.imwrite(path, read_frame(comp))
                sample_records.append((label, path, note))
                print(f"SAMPLE_DONE {path}", flush=True)

        # 2) Overview shot
        loc, rot = overview_pose(n_walls=n, spacing_m=WALL_SPACING_M)
        with instance.begin_frame():
            cam.K2_SetActorLocationAndRotation(
                NewLocation=loc, NewRotation=rot, bSweep=False, bTeleport=True
            )
        with instance.end_frame():
            pass
        instance.step(num_frames=PER_SAMPLE_WARMUP)
        with instance.begin_frame():
            pass
        with instance.end_frame():
            overview_path = os.path.join(OUTPUT_DIR, "overview.png")
            cv2.imwrite(overview_path, read_frame(comp))
            print(f"OVERVIEW_DONE {overview_path}", flush=True)
    finally:
        instance.close(force=True)

    # 3) Gallery grid PNG for at-a-glance comparison
    gallery_path = os.path.join(OUTPUT_DIR, "gallery.png")
    _make_gallery_grid(sample_records, gallery_path)
    print(f"GALLERY_DONE {gallery_path}", flush=True)


if __name__ == "__main__":
    main()
