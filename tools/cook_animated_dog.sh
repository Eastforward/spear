#!/bin/bash
# Cook the animated_dog uassets into the SpearSim pak so SPEAR RPC can
# load_class(BP_dog_animated) at runtime.
#
# Prereqs: docs/animated_dog_ue_import.md fully executed. The two directories
# below MUST exist (with the uassets in them) before running.

set -euo pipefail

/data/jzy/miniconda3/envs/spear-env/bin/python /data/jzy/code/SPEAR/tools/run_uat.py \
  --unreal-engine-dir /data/UE_5.5 \
  --cook-dirs /Game/MyAssets/Audioset/Meshes/animated_dog \
              /Game/MyAssets/Audioset/Blueprints/animated_dog \
  --skip-cook-default-maps

echo "COOK_DONE"
