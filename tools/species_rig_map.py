"""Species -> source rig mapping for 12-quadruped batch pipeline.

Cat rig (Quaternius) -- small felids and small mammals:   cat, chipmunk
Dog rig -- medium canid/pig/goat/sheep:                   dog, goat, sheep, pig
Wolf rig -- large ungulates:                              horse, cattle, yak, donkey
"""

import os

# --- path resolution -------------------------------------------------------
# SPEAR_ROOT = this file's grandparent (tools/species_rig_map.py → SPEAR/).
SPEAR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# AVENGINE_ROOT: parent of external/SPEAR/ if this checkout lives inside an
# AVEngine monorepo (default post-2026-07-07). Env var overrides; else infer.
AVENGINE_ROOT = os.environ.get(
    "AVENGINE_ROOT",
    os.path.dirname(os.path.dirname(SPEAR_ROOT)),   # ../.. from SPEAR
)

# Quaternius rigs live in AVEngine assets/ (copied from Spatial by AVEngine setup).
QUATERNIUS_DIR = os.path.join(AVENGINE_ROOT, "assets/mesh_library/quaternius_animalpack")
QUATERNIUS_FARM = os.path.join(AVENGINE_ROOT, "assets/mesh_library/quaternius_farm")

# Hunyuan-generated intermediate mesh outputs. Historically lived under
# SPEAR/tmp/hy3d_batch and Hunyuan3D-2.1/outputs/audioset_assets. At runtime
# these are only referenced for filesystem paths already baked into UE .uasset
# at cook time — the demo pipeline doesn't actually load .obj at runtime.
HY3D_BATCH_DIR = os.environ.get(
    "HY3D_BATCH_DIR", os.path.join(SPEAR_ROOT, "tmp/hy3d_batch")
)
HY3D_AUDIOSET_DIR = os.environ.get(
    "HY3D_AUDIOSET_DIR",
    os.path.join(AVENGINE_ROOT, "external/Hunyuan3D-2.1/outputs/audioset_assets"),
)


def _batch_mesh(tag):
    return {
        "mesh": f"{HY3D_BATCH_DIR}/{tag}/hy3d_textured.obj",
        "diffuse": f"{HY3D_BATCH_DIR}/{tag}/hy3d_diffuse.jpg",
    }


def _audioset_mesh(dirname):
    """Prefer GLB (has embedded texture; UE Interchange glTF importer is much
    faster/more robust than the OBJ importer on Hunyuan meshes).
    """
    base = f"{HY3D_AUDIOSET_DIR}/{dirname}/{dirname}_textured"
    return {"mesh": f"{base}.glb", "diffuse": f"{base}.jpg"}


CAT_RIG = f"{QUATERNIUS_DIR}/Cat.glb"
DOG_RIG = f"{QUATERNIUS_DIR}/Dog.glb"

# Per-rig-family: how many degrees to add to motion direction to get body yaw
# so the animal walks head-first.
#
# 2026-07-08 correction: was 180.0 (assumption: rig-local-forward = -X_local).
# Visual verification on shoebox_v2 view2 (mic looking -Y, golden walking +X)
# showed the animation ran head-first-BACKWARDS: head pointed at -X while
# motion was +X. That means Quaternius Dog/Cat Walk rig-local-forward is
# actually +X_local, NOT -X_local — no offset needed. See tests/tools/
# spike_rlr/test_rig_forward_offset.py for the regression check.
QUATERNIUS_FORWARD_YAW_OFFSET_DEG = 0.0


# 2026-07-06: quaternius_farm rigs (Horse/Cow/Zebra) have real Walk anims
# but use SEMANTIC bone names (FrontFoot.R, Tail4, Head...) instead of the
# Bone.NNN template shared by animalpack Cat/Dog/Wolf. robust_skin_transfer
# and blender_robust_swap's dampening args are hardcoded to Bone.NNN names,
# so farm rigs would produce broken animations. For this pass we ship large
# ungulates as STATIC meshes and revisit animated ungulates as a follow-up.
ANIMATED_RIG_MAP = {
    # 2026-07-06: Cat.glb gate check on cat_persian looked natural once the
    # orbit camera was pulled back and actor scaled to 0.3 -- the "spikes"
    # I initially saw were the persian's long tail fur, not corrupt geometry.
    "cat_persian":     {"rig": CAT_RIG, "walking_forward_yaw_offset_deg": QUATERNIUS_FORWARD_YAW_OFFSET_DEG, **_batch_mesh("cat_persian")},
    "cat_tabby":       {"rig": CAT_RIG, "walking_forward_yaw_offset_deg": QUATERNIUS_FORWARD_YAW_OFFSET_DEG, **_batch_mesh("cat_tabby")},
    "chipmunk":        {"rig": CAT_RIG, "walking_forward_yaw_offset_deg": QUATERNIUS_FORWARD_YAW_OFFSET_DEG, **_batch_mesh("chipmunk")},
    "dog_golden":      {"rig": DOG_RIG, "walking_forward_yaw_offset_deg": QUATERNIUS_FORWARD_YAW_OFFSET_DEG, **_batch_mesh("dog_golden")},
    "dog_husky":       {"rig": DOG_RIG, "walking_forward_yaw_offset_deg": QUATERNIUS_FORWARD_YAW_OFFSET_DEG, **_batch_mesh("dog_husky")},
}

# Import-time guard: every animated tag MUST declare its walking yaw offset.
# Adding a new rig without this field will fail loudly the first time
# species_rig_map is imported, preventing silent backward-walk regressions.
for _tag, _meta in ANIMATED_RIG_MAP.items():
    assert "walking_forward_yaw_offset_deg" in _meta, (
        f"animated tag {_tag} missing 'walking_forward_yaw_offset_deg'. "
        f"Set it based on your rig's Walking anim local-forward direction. "
        f"For Quaternius Dog/Cat use QUATERNIUS_FORWARD_YAW_OFFSET_DEG (180.0)."
    )

STATIC_MESH_MAP = {
    # 7 ungulates -- imported as static meshes for apartment integration.
    "goat":            _audioset_mesh("goat"),
    "sheep":           _audioset_mesh("sheep"),
    "pig":             _audioset_mesh("pig"),
    "horse":           _audioset_mesh("horse"),
    "cattle_bovinae":  _audioset_mesh("cattle_bovinae"),
    "yak":             _audioset_mesh("yak"),
    "donkey_ass":      _audioset_mesh("donkey_ass"),
}

# Legacy alias -- some callers still want a merged view.
RIG_MAP = dict(ANIMATED_RIG_MAP)


def assert_inputs_exist(tag):
    if tag in ANIMATED_RIG_MAP:
        entry = ANIMATED_RIG_MAP[tag]
        keys = ("rig", "mesh", "diffuse")
    elif tag in STATIC_MESH_MAP:
        entry = STATIC_MESH_MAP[tag]
        keys = ("mesh", "diffuse")
    else:
        raise SystemExit(f"[species_rig_map] unknown tag: {tag}")
    missing = [k for k in keys if not os.path.exists(entry[k])]
    if missing:
        raise SystemExit(
            f"[species_rig_map] {tag} missing: {[(k, entry[k]) for k in missing]}"
        )
    return entry


if __name__ == "__main__":
    import sys

    tag = sys.argv[1] if len(sys.argv) > 1 else None
    if tag:
        entry = assert_inputs_exist(tag)
        print(f"[species_rig_map] {tag}: OK -> {entry}")
    else:
        print("=== ANIMATED ===")
        for t, e in ANIMATED_RIG_MAP.items():
            marks = "".join("Y" if os.path.exists(e[k]) else "-" for k in ("rig", "mesh", "diffuse"))
            print(f"{marks}  {t}")
        print("=== STATIC ===")
        for t, e in STATIC_MESH_MAP.items():
            marks = "".join("Y" if os.path.exists(e[k]) else "-" for k in ("mesh", "diffuse"))
            print(f"{marks}   {t}")
