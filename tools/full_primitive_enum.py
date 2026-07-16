"""Diagnostic: after loading Entry map, spawn the 6 shoebox walls + 1 point
light + 1 camera (nothing else). Then iterate every actor AND every
UPrimitiveComponent under each actor. For each primitive record:

  - outer actor stable name + unreal name
  - component class (fully qualified)
  - component's static mesh asset path (if UStaticMeshComponent)
  - component world location
  - component world bounds (origin + box extent)
  - component bIsVisible / bHiddenInGame / bVisible flags where accessible

Save the report to /tmp/full_primitive_enum.txt and print it. The mystery
"checkerboard cube" is visible from the camera in the northeast interior
corner of the room and stands ~1.5-2m tall / ~1m wide, so we're hunting for
a primitive with world-space bounds Origin near (~450-500, ~350-440, ~50-100)
and BoxExtent around (~50, ~50, ~80-100). Anything unexpected in the output
is the culprit.

Run with:

  DISPLAY=:99 VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
      /data/jzy/miniconda3/envs/spear-env/bin/python \
      /data/jzy/code/SPEAR/tools/full_primitive_enum.py
"""

import os
import sys
import traceback


_EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__) + "/../examples")
sys.path.insert(0, _EXAMPLES_DIR)

from render_in_gpurir_room import (  # noqa: E402
    configure_gpurir_instance,
    spawn_room_piece,
    spawn_point_light,
    compute_shoebox_room_layout,
    FLOOR_MATERIAL,
    WALL_MATERIAL,
)
from render_in_apartment import spawn_camera  # noqa: E402


OUT_PATH = "/tmp/full_primitive_enum.txt"


def _safe(fn, default="<err>"):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return f"{default}({type(e).__name__}: {e})"


def _xyz(d):
    """Read a dict with either lowercase (x,y,z) or uppercase (X,Y,Z) keys.
    Also unwraps {'ReturnValue': {...}} which UE reflection produces for
    functions whose signature returns a struct."""
    if d is None:
        return None
    if isinstance(d, dict) and "ReturnValue" in d and isinstance(d["ReturnValue"], dict):
        d = d["ReturnValue"]
    try:
        return (float(d["x"]), float(d["y"]), float(d["z"]))
    except (KeyError, TypeError):
        pass
    try:
        return (float(d["X"]), float(d["Y"]), float(d["Z"]))
    except (KeyError, TypeError):
        pass
    return None


def _unwrap_return(d):
    if isinstance(d, dict) and "ReturnValue" in d and len(d) == 1:
        return d["ReturnValue"]
    return d


def _actor_stable(game, actor):
    return _safe(
        lambda: game.unreal_service.get_stable_name_for_actor(
            actor=actor, include_unreal_name=True
        ),
        default="<no-stable>",
    )


def _class_name(game, uobj):
    return _safe(
        lambda: game.unreal_service.get_type_for_class_as_string(
            uclass=game.unreal_service.get_class(uobject=uobj)
        ),
        default="<no-class>",
    )


def _comp_world_loc(comp):
    def _do():
        loc = comp.K2_GetComponentLocation(as_dict=True)
        v = _xyz(loc)
        if v is None:
            raise ValueError(f"bad-loc-shape={loc!r}")
        return v
    return _safe(_do, default=None)


def _comp_world_scale(comp):
    """Return the component's world-space (x,y,z) scale, or None."""
    def _try_relative():
        v = _xyz(comp.GetRelativeScale3D(as_dict=True))
        if v is None:
            raise ValueError("bad")
        return v
    v = _safe(_try_relative, default=None)
    if isinstance(v, tuple):
        return v

    def _try_vector_getter():
        v = _xyz(comp.GetComponentScale(as_dict=True))
        if v is None:
            raise ValueError("bad")
        return v
    v = _safe(_try_vector_getter, default=None)
    if isinstance(v, tuple):
        return v
    return None


def _sm_asset(game, comp):
    """If component is a UStaticMeshComponent, return the mesh asset path
    via the "StaticMesh" property (not the getter, which the SPEAR wrapper
    exposes as an UnrealObject descriptor, not a callable)."""
    try:
        if not comp.is_a(uclass="UStaticMeshComponent"):
            return "<not-static-mesh-component>"
    except Exception:
        pass
    try:
        val = game.unreal_service.get_property_value_for_object(
            uobject=comp, property_name="StaticMesh"
        )
    except Exception as e:  # noqa: BLE001
        return f"<StaticMesh property err: {type(e).__name__}: {e}>"
    return repr(val)


def _bounds_from_local(comp):
    """UPrimitiveComponent::GetLocalBounds returns (min, max) in local space.
    Approximate world bounds = component_loc + component_scale * local_center,
    world_extent = |scale * local_extent|. Ignores rotation for speed."""
    def _do():
        b = comp.GetLocalBounds(
            Min={"X": 0.0, "Y": 0.0, "Z": 0.0},
            Max={"X": 0.0, "Y": 0.0, "Z": 0.0},
            as_dict=True,
        )
        b = _unwrap_return(b) if isinstance(b, dict) else b
        # UE reflection encodes OUT params under the arg name, may keep
        # ReturnValue too. Peel off "ReturnValue" first, then look for Min/Max.
        if isinstance(b, dict) and "ReturnValue" in b:
            b = b["ReturnValue"] if isinstance(b["ReturnValue"], dict) else b
        # dict returned may use either "Min"/"Max" or "min"/"max"
        mn = b.get("Min", b.get("min"))
        mx = b.get("Max", b.get("max"))
        mnv, mxv = _xyz(mn), _xyz(mx)
        if mnv is None or mxv is None:
            raise ValueError(f"bad-bounds-shape={b!r}")
        cx = (mnv[0] + mxv[0]) / 2.0
        cy = (mnv[1] + mxv[1]) / 2.0
        cz = (mnv[2] + mxv[2]) / 2.0
        ex = (mxv[0] - mnv[0]) / 2.0
        ey = (mxv[1] - mnv[1]) / 2.0
        ez = (mxv[2] - mnv[2]) / 2.0
        return {"center_local": (cx, cy, cz), "extent_local": (ex, ey, ez)}
    return _safe(_do, default=None)


def _visibility(comp):
    """Try both bVisible / bHiddenInGame via IsVisible/K2_IsVisible-style getters."""
    out = {}
    for name, fn_name in (
        ("visible", "IsVisible"),
        ("hidden_in_game", "IsHiddenInGame"),
    ):
        try:
            out[name] = getattr(comp, fn_name)()
        except Exception as e:  # noqa: BLE001
            out[name] = f"<{type(e).__name__}>"
    return out


def _is_primitive(game, comp):
    """Return True if comp is a UPrimitiveComponent (renders geometry)."""
    try:
        return comp.is_a(uclass="UPrimitiveComponent")
    except Exception:
        return False


def _iter_actor_components(game, actor):
    """Return raw list of every UActorComponent on the actor."""
    try:
        return list(game.unreal_service.get_components(actor=actor))
    except Exception as e:  # noqa: BLE001
        print(f"    <get_components failed: {e}>", flush=True)
        return []


def enumerate_and_dump(game, fp):
    actors = game.unreal_service.find_actors_by_class(uclass="AActor")
    fp.write(f"==== TOTAL ACTORS: {len(actors)} ====\n\n")

    prims_total = 0
    for i, actor in enumerate(actors):
        stable = _actor_stable(game, actor)
        acls = _class_name(game, actor)
        try:
            loc = actor.K2_GetActorLocation(as_dict=True)
            v = _xyz(loc)
            if v is None:
                loc_s = f"<bad-loc-shape={loc!r}>"
            else:
                loc_s = f"({v[0]:.1f},{v[1]:.1f},{v[2]:.1f})"
        except Exception as e:  # noqa: BLE001
            loc_s = f"<no-loc: {e}>"

        fp.write(f"[A{i:03d}] {stable}\n")
        fp.write(f"       actor_class = {acls}\n")
        fp.write(f"       actor_loc   = {loc_s}\n")

        components = _iter_actor_components(game, actor)
        fp.write(f"       n_components = {len(components)}\n")

        for j, comp in enumerate(components):
            cname = _class_name(game, comp)
            is_prim = _is_primitive(game, comp)
            cloc = _comp_world_loc(comp)
            cscale = _comp_world_scale(comp)
            try:
                cstable = game.unreal_service.get_stable_name_for_component(
                    component=comp
                )
            except Exception:
                cstable = "<no-stable>"

            fp.write(
                f"  [C{j:02d}] {cstable} :: {cname}"
                f" is_prim={is_prim}\n"
            )
            fp.write(f"         world_loc   = {cloc}\n")
            fp.write(f"         world_scale = {cscale}\n")

            if is_prim:
                prims_total += 1
                lb = _bounds_from_local(comp)
                fp.write(f"         local_bounds = {lb}\n")

                # Reliable world bounds from UPrimitiveComponent::Bounds (updated after tick).
                try:
                    wb = game.unreal_service.get_property_value_for_object(
                        uobject=comp, property_name="Bounds"
                    )
                    fp.write(f"         world_bounds_prop = {wb}\n")
                except Exception as e:  # noqa: BLE001
                    fp.write(f"         world_bounds_prop = <err {type(e).__name__}: {e}>\n")

                # Approx world bounds (ignore rotation; sufficient for hunting cubes)
                if (
                    isinstance(lb, dict)
                    and isinstance(cloc, tuple)
                    and isinstance(cscale, tuple)
                ):
                    try:
                        cx = cloc[0] + cscale[0] * lb["center_local"][0]
                        cy = cloc[1] + cscale[1] * lb["center_local"][1]
                        cz = cloc[2] + cscale[2] * lb["center_local"][2]
                        ex = abs(cscale[0] * lb["extent_local"][0])
                        ey = abs(cscale[1] * lb["extent_local"][1])
                        ez = abs(cscale[2] * lb["extent_local"][2])
                        fp.write(
                            f"         approx_world_bounds = orig=({cx:.1f},{cy:.1f},{cz:.1f})"
                            f" ext=({ex:.1f},{ey:.1f},{ez:.1f})\n"
                        )
                    except Exception as e:  # noqa: BLE001
                        fp.write(f"         approx_world_bounds = <err {e}>\n")

                # Static mesh asset (only meaningful for UStaticMeshComponent, harmless otherwise)
                asset = _sm_asset(game, comp)
                fp.write(f"         static_mesh = {asset}\n")

                vis = _visibility(comp)
                fp.write(f"         visibility  = {vis}\n")

        fp.write("\n")

    fp.write(f"==== TOTAL PRIMITIVE COMPONENTS: {prims_total} ====\n")


def main():
    instance = configure_gpurir_instance(rpc_port=39002)
    game = instance.get_game()
    try:
        # 1) Purge every renderable / spawn-happy actor class that could hide a mesh.
        with instance.begin_frame():
            for cls in (
                "APlayerStart", "ADefaultPawn", "ASpectatorPawn",
                "AStaticMeshActor", "ASkeletalMeshActor", "ABrush", "ADecalActor",
                "AInstancedFoliageActor",
                "AGameplayDebuggerCategoryReplicator",
                "AGameplayDebuggerPlayerManager",
            ):
                try:
                    victims = game.unreal_service.find_actors_by_class(uclass=cls)
                except Exception:
                    victims = []
                for existing in victims:
                    try:
                        game.unreal_service.destroy_actor(actor=existing)
                    except Exception:
                        pass
        with instance.end_frame():
            pass

        # 2) Spawn only the 6 shoebox walls + 1 point light + 1 camera.
        with instance.begin_frame():
            for p in compute_shoebox_room_layout(room_size_m=(5.2, 4.4, 2.8)):
                spawn_room_piece(
                    game=game,
                    piece=p,
                    material_path=(FLOOR_MATERIAL if p["name"] == "floor" else WALL_MATERIAL),
                )
            spawn_point_light(
                game=game, x_cm=260.0, y_cm=220.0, z_cm=265.0,
                intensity_lumens=2200.0, attenuation_cm=600.0,
            )
            cam, comp = spawn_camera(game=game, width=1280, height=720)
            _ = (cam, comp)
        with instance.end_frame():
            pass

        # 3) Let the world tick a few frames so late-spawned components (debug drawers,
        # capture proxies, etc.) attach.
        instance.step(num_frames=4)

        # 4) Enumerate.
        with instance.begin_frame():
            with open(OUT_PATH, "w", encoding="utf-8") as fp:
                enumerate_and_dump(game, fp)
        with instance.end_frame():
            pass

        print(f"WROTE {OUT_PATH}", flush=True)
    finally:
        instance.close(force=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
