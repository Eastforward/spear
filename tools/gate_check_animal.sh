#!/bin/bash
# Gate check runner for one species tag:
#   1. Rig-swap Hunyuan mesh onto species-matched Quaternius source rig
#   2. UE headless import + cook
#   3. Render 72-frame side-view orbit video
#
# Output: /tmp/gate_check_v4/<tag>_side.mp4
set -uo pipefail

if [ -z "${1:-}" ]; then
    echo "usage: gate_check_animal.sh <tag>"
    exit 1
fi
TAG="$1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPEAR_DIR="${SPEAR_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
UE_DIR="${UE_DIR:-/data/UE_5.5}"
PY="${PY:-/data/jzy/miniconda3/envs/spear-env/bin/python}"
BLENDER="${BLENDER:-/data/jzy/blender/blender-4.2.1-linux-x64/blender}"
GATE_RUNTIME_TARGET_FACES="${GATE_RUNTIME_TARGET_FACES:-40000}"
export SPEAR_DIR

mkdir -p /tmp/gate_check_v4

# --- ensure approved high-poly meshes have a lighter runtime proxy
"$PY" "$SPEAR_DIR/tools/ensure_runtime_proxy_mesh.py" \
    --tag "$TAG" \
    --approved-dir "$SPEAR_DIR/tmp/hy3d_batch/approved" \
    --target-faces "$GATE_RUNTIME_TARGET_FACES" \
    --blender "$BLENDER"

# --- verify inputs, dump paths as json for shell parsing
INPUT_JSON="/tmp/gate_check_v4/${TAG}_inputs.json"
$PY - "$TAG" > "$INPUT_JSON" <<'PY'
import json, sys, os
sys.path.insert(0, os.path.join(os.environ["SPEAR_DIR"], "tools"))
import species_rig_map as m
tag = sys.argv[1]
entry = m.assert_inputs_exist(tag)
print(json.dumps(entry))
PY

RIG=$($PY -c "import json; print(json.load(open('$INPUT_JSON'))['rig'])")
MESH=$($PY -c "import json; print(json.load(open('$INPUT_JSON'))['mesh'])")
DIFF=$($PY -c "import json; print(json.load(open('$INPUT_JSON')).get('diffuse',''))")
echo "[gate_check] tag=$TAG rig=$RIG mesh=$MESH diff=$DIFF"

# --- rig swap (Blender)
RIGGED_GLB=/tmp/gate_check_v4/${TAG}_rigged.glb
BLENDER_LOG=/tmp/gate_check_v4/${TAG}_blender.log
BLENDER_ARGS=(
    "$BLENDER" --background --python "$SPEAR_DIR/tools/blender_robust_swap_mesh_keep_rig.py" --
    --rig-glb "$RIG"
    --new-mesh "$MESH"
    --output "$RIGGED_GLB"
    --reverse-actions no
)
if [ -n "$DIFF" ]; then
    BLENDER_ARGS+=(--new-diffuse "$DIFF")
fi
"${BLENDER_ARGS[@]}" > "$BLENDER_LOG" 2>&1
BLENDER_RC=$?
tail -30 "$BLENDER_LOG"

if [ ! -f "$RIGGED_GLB" ]; then
    echo "GATE_CHECK_FAIL rig swap did not produce $RIGGED_GLB (blender rc=$BLENDER_RC)"
    exit 1
fi
echo "GATE_CHECK_RIGSWAP_OK $RIGGED_GLB"

# --- UE cook
MESH_UE_DIR="$SPEAR_DIR/cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Meshes/gate_${TAG}"
BP_UE_DIR="$SPEAR_DIR/cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Blueprints/gate_${TAG}"
BP_PATH="$BP_UE_DIR/BP_gate_${TAG}.uasset"

echo "=== wipe stale gate_${TAG} uassets ==="
rm -rf "$MESH_UE_DIR" "$BP_UE_DIR"

echo "=== UE headless import ==="
GATE_TAG="$TAG" GATE_RIGGED_GLB="$RIGGED_GLB" \
    "$PY" "$SPEAR_DIR/tools/run_editor_script.py" \
    --script "$SPEAR_DIR/tools/import_gate_animal_editor.py" \
    --unreal-engine-dir "$UE_DIR" \
    --launch-mode commandlet \
    || echo "(editor commandlet returned nonzero -- checking BP presence)"

if [ ! -f "$BP_PATH" ]; then
    echo "GATE_CHECK_FAIL BP_uasset missing at $BP_PATH"
    exit 1
fi
echo "GATE_CHECK_IMPORT_OK $BP_PATH"

echo "=== UE cook ==="
"$PY" "$SPEAR_DIR/tools/run_uat.py" \
    --unreal-engine-dir "$UE_DIR" \
    --skip-cook-default-maps \
    -build -cook -stage -package -archive -pak \
    || { echo "GATE_CHECK_FAIL cook failed"; exit 1; }

# --- orbit render
ORBIT_DIR=/tmp/gate_check_v4/orbit_${TAG}
rm -rf "$ORBIT_DIR"
DISPLAY="${DISPLAY:-:99}" \
VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}" \
PYTHONPATH="$SPEAR_DIR/examples:$SPEAR_DIR/tools:${PYTHONPATH:-}" \
    "$PY" /tmp/orbit_animal.py \
    --tag "$TAG" \
    --n-frames 72 \
    --output-dir "$ORBIT_DIR"

# --- encode
OUTMP4=/tmp/gate_check_v4/${TAG}_side.mp4
ffmpeg -y -framerate 15 -i "$ORBIT_DIR/frame_%04d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 "$OUTMP4" > /tmp/gate_check_v4/${TAG}_ffmpeg.log 2>&1

if [ ! -s "$OUTMP4" ]; then
    echo "GATE_CHECK_FAIL video empty at $OUTMP4"
    exit 1
fi
echo "GATE_CHECK_DONE $OUTMP4"
