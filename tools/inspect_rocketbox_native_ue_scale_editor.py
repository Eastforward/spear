"""Read-only UE scale diagnostics for one already-imported native Rocketbox tag."""

import json
import os
from pathlib import Path

import spear
import unreal


manifest_path = Path(os.environ["ROCKETBOX_NATIVE_UE_MANIFEST"]).resolve()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
content = manifest["content"]
mesh = unreal.load_asset(content["skeletal_mesh"])
blueprint = unreal.load_asset(content["blueprint"])
if mesh is None or blueprint is None:
    raise RuntimeError("native Rocketbox UE assets could not be loaded")


def xyz(value):
    return [float(value.x), float(value.y), float(value.z)]


result = {
    "tag": manifest["tag"],
    "mesh_path": mesh.get_path_name(),
    "mesh_bound_methods": [name for name in dir(mesh) if "bound" in name.lower()],
}
for method_name in ("get_imported_bounds", "get_bounds"):
    try:
        value = getattr(mesh, method_name)()
    except Exception as error:
        result[method_name] = {"error": str(error)}
        continue
    result[method_name] = {
        "origin_cm": xyz(value.origin),
        "box_extent_cm": xyz(value.box_extent),
        "sphere_radius_cm": float(value.sphere_radius),
        "height_cm": 2.0 * float(value.box_extent.z),
    }
for property_name in (
    "imported_bounds",
    "positive_bounds_extension",
    "negative_bounds_extension",
):
    try:
        value = mesh.get_editor_property(property_name)
    except Exception as error:
        result[property_name] = {"error": str(error)}
        continue
    if hasattr(value, "origin") and hasattr(value, "box_extent"):
        result[property_name] = {
            "origin_cm": xyz(value.origin),
            "box_extent_cm": xyz(value.box_extent),
            "sphere_radius_cm": float(value.sphere_radius),
        }
    elif hasattr(value, "x"):
        result[property_name] = xyz(value)
    else:
        result[property_name] = str(value)

components = [
    item["object"]
    for item in spear.editor.get_subobject_descs_for_blueprint_asset(
        blueprint_asset=blueprint
    )
    if isinstance(item["object"], unreal.SkeletalMeshComponent)
]
if len(components) != 1:
    raise RuntimeError(f"expected one skeletal component, got {len(components)}")
component = components[0]
result["component_bound_methods"] = [
    name for name in dir(component) if "bound" in name.lower()
]
for property_name in ("bounds_scale", "component_scale"):
    try:
        value = component.get_editor_property(property_name)
    except Exception as error:
        result[property_name] = {"error": str(error)}
        continue
    result[property_name] = xyz(value) if hasattr(value, "x") else float(value)

print("ROCKETBOX_NATIVE_UE_SCALE=" + json.dumps(result, sort_keys=True))
