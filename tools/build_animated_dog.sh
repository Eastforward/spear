#!/bin/bash
# End-to-end Stage 2 automation for the animated dog:
#   1. Headless UE Editor commandlet imports Dog_textured.glb + creates BP
#   2. run_uat.py cooks the new content into the SpearSim pak
#
# No GUI required. Replaces the manual docs/animated_dog_ue_import.md steps.
#
# NOTE: the editor commandlet emits a non-fatal Interchange glTF ensure()
# during import (harmless — assets still land correctly) but the commandlet
# nonetheless returns exit 1 whenever any warnings/errors were logged. We
# therefore don't use `set -e` on the editor step: we ignore its exit code
# and verify success by checking that the BP uasset actually exists on disk.

set -uo pipefail

SPEAR_DIR=/data/jzy/code/SPEAR
UE_DIR=/data/UE_5.5
PY=/data/jzy/miniconda3/envs/spear-env/bin/python
BP_PATH="$SPEAR_DIR/cpp/unreal_projects/SpearSim/Content/MyAssets/Audioset/Blueprints/animated_dog/BP_dog_animated.uasset"

echo "=== Stage 2a: headless import (UE commandlet) ==="
"$PY" "$SPEAR_DIR/tools/run_editor_script.py" \
    --script "$SPEAR_DIR/tools/import_animated_dog_editor.py" \
    --unreal-engine-dir "$UE_DIR" \
    --launch-mode commandlet || echo "(editor commandlet returned nonzero — checking on-disk state before treating as failure)"

if [ ! -f "$BP_PATH" ]; then
    echo "STAGE2A_FAILED BP_uasset missing at $BP_PATH"
    exit 1
fi
echo "STAGE2A_OK BP uasset present: $BP_PATH"

echo "=== Stage 2b: cook (Audioset already in DirectoriesToAlwaysCook) ==="
# DefaultGame.ini already registers /Game/MyAssets/Audioset in
# +DirectoriesToAlwaysCook=, so a normal cook picks up our new
# animated_dog/ subdirectories automatically. Do NOT pass --cook-dirs
# here — run_uat.py prepends unreal_project_dir to each arg, so passing
# /Game/MyAssets/... results in a bogus filesystem path and cook silently
# no-ops in ~2s.
# -build -cook -stage -package -archive -pak: the RunUAT-side phase flags
# that actually cook and archive. Without them BuildCookRun is a 2-second
# no-op. See docs/getting_started.md § "Build SpearSim in Standalone".
"$PY" "$SPEAR_DIR/tools/run_uat.py" \
    --unreal-engine-dir "$UE_DIR" \
    --skip-cook-default-maps \
    -build -cook -stage -package -archive -pak

echo "BUILD_ANIMATED_DOG_DONE"
