"""Raycast grid toward the east wall to find what the mystery cube is."""
import sys, os
sys.path.insert(0, '/data/jzy/code/SPEAR/examples')
from render_in_gpurir_room import (configure_gpurir_instance, spawn_room_piece, spawn_point_light,
    spawn_directional_light, spawn_sky, compute_shoebox_room_layout, FLOOR_MATERIAL, WALL_MATERIAL)
from render_in_apartment import spawn_camera, read_frame
import cv2

os.makedirs('/tmp/diag_raycast', exist_ok=True)
instance = configure_gpurir_instance(rpc_port=39002)
game = instance.get_game()
try:
    with instance.begin_frame():
        for cls in ('APlayerStart','ADefaultPawn','ASpectatorPawn','AInstancedFoliageActor',
                    'AGameplayDebuggerCategoryReplicator','AGameplayDebuggerPlayerManager'):
            try:
                for a in game.unreal_service.find_actors_by_class(uclass=cls):
                    game.unreal_service.destroy_actor(actor=a)
            except: pass
        for p in compute_shoebox_room_layout(room_size_m=(5.2,4.4,2.8)):
            spawn_room_piece(game=game, piece=p, material_path=(FLOOR_MATERIAL if p['name']=='floor' else WALL_MATERIAL))
        spawn_sky(game=game)
        spawn_directional_light(game=game, yaw_deg=-90.0, pitch_deg=-40.0, intensity_lux=10.0)
        spawn_point_light(game=game, x_cm=260.0, y_cm=220.0, z_cm=265.0, intensity_lumens=2200.0, attenuation_cm=600.0)
        cam, comp = spawn_camera(game=game, width=1280, height=720)
    with instance.end_frame(): pass
    instance.step(num_frames=40)

    cam_x, cam_y, cam_z = 460.0, 390.0, 73.4
    with instance.begin_frame():
        cam.K2_SetActorLocationAndRotation(NewLocation={'X':cam_x,'Y':cam_y,'Z':cam_z}, NewRotation={'Roll':0.0,'Pitch':-11.3,'Yaw':180.0}, bSweep=False, bTeleport=True)
    with instance.end_frame(): pass
    instance.step(num_frames=30)
    with instance.begin_frame():
        cv2.imwrite('/tmp/diag_raycast/frame_baseline.png', read_frame(comp))
    with instance.end_frame(): pass

    with instance.begin_frame():
        kismet = game.get_unreal_object(uclass='UKismetSystemLibrary')
        hits = []
        # Dense grid of rays toward east wall region
        for tx in [400, 450, 500]:
            for ty in [250, 300, 350, 400, 430]:
                for tz in [30, 80, 130, 180, 230]:
                    start = {'X': cam_x, 'Y': cam_y, 'Z': cam_z}
                    end = {'X': float(tx), 'Y': float(ty), 'Z': float(tz)}
                    for profile in ('BlockAll', 'Visibility'):
                        try:
                            r = kismet.LineTraceSingleByProfile(
                                Start=start, End=end, ProfileName=profile, bTraceComplex=True,
                                ActorsToIgnore=[], DrawDebugType='None', bIgnoreSelf=True,
                                TraceColor={'R':1,'G':0,'B':0,'A':1}, TraceHitColor={'R':0,'G':1,'B':0,'A':1},
                                DrawTime=0.0, as_dict=True)
                        except Exception as e:
                            print(f'ERR profile={profile} {e}', flush=True); continue
                        if r.get('ReturnValue'):
                            h = r['OutHit']
                            comp_name = str(h.get('component',''))
                            actor_name = str(h.get('actor',''))
                            loc = h['location']
                            lx = loc['x']; ly = loc['y']; lz = loc['z']
                            tag = ''
                            if 'wall_x1' in actor_name or 'wall_x1' in comp_name: tag='wall_x1'
                            elif 'wall_y1' in actor_name or 'wall_y1' in comp_name: tag='wall_y1'
                            elif 'floor' in actor_name: tag='floor'
                            else: tag='MYSTERY'
                            if tag == 'MYSTERY':
                                line = 'HIT profile=%s target=(%s,%s,%s) -> actor=%r comp=%r loc=(%.0f,%.0f,%.0f) [%s]' % (profile, tx, ty, tz, actor_name, comp_name, lx, ly, lz, tag)
                                print(line, flush=True)
                                hits.append(line)
        if not hits:
            print('NO_MYSTERY_HITS — all rays hit known walls/floor only', flush=True)
        else:
            print(f'TOTAL_MYSTERY_HITS={len(hits)}', flush=True)
    with instance.end_frame(): pass
finally:
    instance.close(force=True)
print('DONE', flush=True)
