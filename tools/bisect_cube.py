"""Bisect: render one frame with progressively fewer things spawned.
After each render, save frame. We then eyeball which configuration makes the
cube disappear. Binary search by elimination."""

import sys, os, cv2
sys.path.insert(0, '/data/jzy/code/SPEAR/examples')
from render_in_gpurir_room import (configure_gpurir_instance, spawn_room_piece, spawn_point_light,
    spawn_directional_light, spawn_sky, compute_shoebox_room_layout,
    FLOOR_MATERIAL, WALL_MATERIAL)
from render_in_apartment import spawn_camera, read_frame

OUT = '/tmp/diag_bisect'
os.makedirs(OUT, exist_ok=True)

CONFIGS = [
    # (label, spawn_floor, spawn_ceiling, spawn_walls_x, spawn_walls_y, sky, dir_light, point_light)
    # Keep ALL lights on; only remove geometry piece by piece.
    ('A_all',            True,  True,  True,  True,  True, True, True),
    ('B_no_ceiling',     True,  False, True,  True,  True, True, True),
    ('C_no_walls_y',     True,  False, True,  False, True, True, True),
    ('D_no_walls_x',     True,  False, False, False, True, True, True),
    ('E_floor_only',     True,  False, False, False, True, True, True),
    ('F_no_floor',       False, False, False, False, True, True, True),
]


def run_one(label, floor, ceiling, walls_x, walls_y, sky, dir_light, point_light):
    instance = configure_gpurir_instance(rpc_port=39002)
    game = instance.get_game()
    try:
        with instance.begin_frame():
            for cls in ('APlayerStart','ADefaultPawn','ASpectatorPawn','AInstancedFoliageActor','AGameplayDebuggerCategoryReplicator','AGameplayDebuggerPlayerManager'):
                try:
                    for a in game.unreal_service.find_actors_by_class(uclass=cls):
                        game.unreal_service.destroy_actor(actor=a)
                except: pass
            for p in compute_shoebox_room_layout(room_size_m=(5.2,4.4,2.8)):
                n = p['name']
                if n == 'floor' and not floor: continue
                if n == 'ceiling' and not ceiling: continue
                if n in ('wall_x0','wall_x1') and not walls_x: continue
                if n in ('wall_y0','wall_y1') and not walls_y: continue
                spawn_room_piece(game=game, piece=p, material_path=(FLOOR_MATERIAL if n=='floor' else WALL_MATERIAL))
            if sky: spawn_sky(game=game)
            if dir_light: spawn_directional_light(game=game, yaw_deg=-90.0, pitch_deg=-40.0, intensity_lux=10.0)
            if point_light: spawn_point_light(game=game, x_cm=260.0, y_cm=220.0, z_cm=265.0, intensity_lumens=2200.0, attenuation_cm=600.0)
            cam, comp = spawn_camera(game=game, width=1280, height=720)
        with instance.end_frame(): pass
        instance.step(num_frames=40)
        with instance.begin_frame():
            cam.K2_SetActorLocationAndRotation(NewLocation={'X':460.0,'Y':390.0,'Z':73.4}, NewRotation={'Roll':0.0,'Pitch':-11.3,'Yaw':180.0}, bSweep=False, bTeleport=True)
        with instance.end_frame(): pass
        instance.step(num_frames=25)
        with instance.begin_frame(): pass
        with instance.end_frame():
            cv2.imwrite(os.path.join(OUT, label + '.png'), read_frame(comp))
            print('RENDERED ' + label, flush=True)
    finally:
        instance.close(force=True)


for cfg in CONFIGS:
    run_one(*cfg)
print('ALL_DONE', flush=True)
