#!/bin/bash
# Batch-import 7 static ungulate meshes into UE as StaticMesh + BP wrappers,
# then run one project-wide cook that includes all of them.
set -uo pipefail

SPEAR_DIR=/data/jzy/code/SPEAR
UE_DIR=/data/UE_5.5
PY=/data/jzy/miniconda3/envs/spear-env/bin/python
STATIC_TAGS="horse cattle_bovinae yak donkey_ass goat sheep pig"

for TAG in $STATIC_TAGS; do
    echo ""
    echo "########## $TAG"
    MESH=$($PY -c "
import sys; sys.path.insert(0, '$SPEAR_DIR/tools')
from species_rig_map import STATIC_MESH_MAP
print(STATIC_MESH_MAP['$TAG']['mesh'])
")
    if [ ! -f "$MESH" ]; then
        echo "STATIC_FAIL $TAG: mesh not found $MESH"
        exit 1
    fi
    MESH_UE_DIR="$SPEAR_DIR/cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Meshes/gate_static_${TAG}"
    BP_UE_DIR="$SPEAR_DIR/cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Blueprints/gate_static_${TAG}"
    BP_PATH="$BP_UE_DIR/BP_gate_static_${TAG}.uasset"

    if [ -f "$BP_PATH" ]; then
        echo "STATIC_OK $TAG (already exists, skip)"
        continue
    fi
    rm -rf "$MESH_UE_DIR" "$BP_UE_DIR"

    STATIC_TAG="$TAG" STATIC_MESH="$MESH" \
        timeout 180 "$PY" "$SPEAR_DIR/tools/run_editor_script.py" \
        --script "$SPEAR_DIR/tools/gpurir_scenes/render_gate_animal_editor.py" \
        --unreal-engine-dir "$UE_DIR" \
        --launch-mode commandlet \
        || echo "(commandlet returned nonzero or hit 120s timeout - verifying BP presence)"

    # If timeout killed it mid-way, ensure no zombie UE hangs around before
    # the next iteration launches its own UE.
    pkill -9 -f "render_gate_animal_editor" 2>/dev/null || true
    pkill -9 -f "UnrealEditor.*render_gate" 2>/dev/null || true
    sleep 2

    if [ ! -f "$BP_PATH" ]; then
        echo "STATIC_FAIL $TAG: BP not written $BP_PATH -- skipping (may be a stubborn glb)"
        continue
    fi
    echo "STATIC_OK $TAG"
done

echo ""
echo "=== UE cook (all 7 in one go) ==="
"$PY" "$SPEAR_DIR/tools/run_uat.py" \
    --unreal-engine-dir "$UE_DIR" \
    --skip-cook-default-maps \
    -build -cook -stage -package -archive -pak

echo "BUILD_STATIC_MESHES_DONE"
