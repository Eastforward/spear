"""Enumerate ALL UPrimitiveComponents in the world, not just actors.
This catches components that live on non-actor owners (e.g. ULevel::ModelComponents,
debug draw components, HUD canvas components). For each primitive, print:
  - owner actor stable name (or '<no-actor-owner>')
  - component class
  - static mesh asset path (if UStaticMeshComponent)
  - material[0] asset path
  - world bounds
Then we look for any primitive whose bounds match the mystery cube (~1m box
near east wall, world ~(400-500, 250-440, 50-200)).
"""
import sys, os, json
sys.path.insert(0, '/data/jzy/code/SPEAR/examples')
from render_in_gpurir_room import (configure_gpurir_instance, spawn_room_piece, spawn_point_light,
    spawn_directional_light, spawn_sky, compute_shoebox_room_layout, FLOOR_MATERIAL, WALL_MATERIAL)
from render_in_apartment import spawn_camera

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
            spawn_room_piece(game=game, piece=p, material_path=(FLOOR_MATERIAL if p['name']=='floor' else WALL_MATERIAL))
        spawn_sky(game=game)
        spawn_directional_light(game=game, yaw_deg=-90.0, pitch_deg=-40.0, intensity_lux=10.0)
        spawn_point_light(game=game, x_cm=260.0, y_cm=220.0, z_cm=265.0, intensity_lumens=2200.0, attenuation_cm=600.0)
        cam, comp = spawn_camera(game=game, width=1280, height=720)
    with instance.end_frame(): pass
    instance.step(num_frames=10)

    with instance.begin_frame():
        # Get ALL actors, then for each, enumerate its components of class UPrimitiveComponent
        actors = game.unreal_service.find_actors_by_class(uclass='AActor')
        print(f'TOTAL_ACTORS={len(actors)}', flush=True)
        prim_count = 0
        suspicious = []
        for actor in actors:
            try:
                actor_name = game.unreal_service.get_stable_name_for_actor(actor=actor, include_unreal_name=True)
            except Exception:
                actor_name = '<unknown>'
            # Get all primitive components on this actor
            try:
                prims = game.unreal_service.get_components_by_class_as_dict(
                    actor=actor, uclass='UPrimitiveComponent',
                    include_actor_stable_name=True, include_actor_unreal_name=True,
                )
            except Exception as e:
                prims = {}
            for prim in prims:
                prim_count += 1
                try:
                    cls_name = type(prim).__name__ if hasattr(prim, '__class__') else str(prim)
                except Exception:
                    cls_name = '?'
                # World bounds of this component
                try:
                    bounds = prim.GetLocalBounds(as_dict=True) if hasattr(prim, 'GetLocalBounds') else None
                except Exception:
                    bounds = None
                # Try to get the static mesh this component renders
                mesh_path = ''
                try:
                    if hasattr(prim, 'GetStaticMesh'):
                        mesh = prim.GetStaticMesh()
                        if mesh is not None:
                            mesh_path = str(mesh)
                except Exception:
                    pass
                line = f'PRIM actor={actor_name!r} comp_class={cls_name} mesh={mesh_path!r} bounds={bounds}'
                print(line, flush=True)
                # Heuristic: flag primitives that could be a ~1m cube (extent 30-150cm all axes)
                if bounds:
                    be = bounds.get('BoxExtent', {})
                    bx, by, bz = be.get('x',0), be.get('y',0), be.get('z',0)
                    if 30 < bx < 200 and 30 < by < 200 and 30 < bz < 200:
                        suspicious.append(line)
        print(f'TOTAL_PRIMS={prim_count}', flush=True)
        print(f'SUSPICIOUS (cube-sized prims):', flush=True)
        for s in suspicious:
            print('  ' + s, flush=True)
    with instance.end_frame(): pass
finally:
    instance.close(force=True)
print('SCRIPT_DONE', flush=True)
