#!/bin/bash
# RLR vs GPURIR spike — one-shot runner.
#
# Runs the pieces of the spike in the correct env with the correct
# LD_PRELOAD to work around the ss2 env's libEGL/libGLdispatch conflict
# with the nvidia driver.
#
# Assumes:
#   - /data/jzy/miniconda3/envs/sao-env      -- GPURIR + SPEAR audio pipeline
#   - /data/jzy/miniconda3/envs/ss2          -- habitat-sim 0.2.2 + RLR + Ambisonics
#   - /data/jzy/miniconda3/envs/spear-env    -- SPEAR RPC (only needed if you also
#                                               regenerate the UE Level; not
#                                               required for the audio-only spike)
#
# Outputs land under:
#   external/SPEAR/tmp/spike_rlr/          -- intermediate mesh + materials JSON
#   external/SPEAR/tmp/spike_output/       -- final deliverables (videos, wav, analysis)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"  # external/SPEAR

SAO=/data/jzy/miniconda3/envs/sao-env/bin/python
SS2=/data/jzy/miniconda3/envs/ss2/bin/python
SPEAR_PY=/data/jzy/miniconda3/envs/spear-env/bin/python

SPIKE_OUT="$REPO_ROOT/tmp/spike_output"

# The ss2 env carries its own libEGL/libGLdispatch that fail to create an
# EGL context on this box's nvidia driver. Forcing the system loaders
# resolves it (see docs in run_audio_pass_rlr.py comments).
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libEGL.so.1:/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0"

echo "== [1/8] mesh + materials from shoebox_v2_spec.json =="
"$SAO" "$SCRIPT_DIR/gen_mesh.py"

echo
echo "== [2/8] A group -- SPEAR GPURIR baseline (bark tone; audible videos) =="
"$SAO" "$SCRIPT_DIR/run_audio_pass_gpurir.py"

echo
echo "== [3/8] B group -- Habitat RLR (bark tone; audible videos) =="
"$SS2" "$SCRIPT_DIR/run_audio_pass_rlr.py" --quality low

echo
echo "== [4/8] top-down 2D videos (A + B, 3 tracks each) =="
"$SS2" "$SCRIPT_DIR/render_topdown_solo.py"
# Also render A group top-down videos
"$SS2" - <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("$SCRIPT_DIR")))
sys.path.insert(0, "$SCRIPT_DIR")
sys.path.insert(0, "/data/jzy/code/AVEngine/external/SPEAR/tools/spike_rlr")
from render_topdown_solo import render_video
audio_dir = Path("/data/jzy/code/AVEngine/external/SPEAR/tmp/spike_output/raw_audio")
out_dir   = Path("/data/jzy/code/AVEngine/external/SPEAR/tmp/spike_output/videos")
render_video("/data/jzy/code/AVEngine/external/SPEAR/data/shoebox_v2_spec.json",
             audio_dir / "audio_A_gpurir_4ch_dog_golden_stereo.wav",
             out_dir / "A_gpurir_topdown_golden_only.mp4",
             highlight_tag="dog_golden")
render_video("/data/jzy/code/AVEngine/external/SPEAR/data/shoebox_v2_spec.json",
             audio_dir / "audio_A_gpurir_4ch_dog_husky_stereo.wav",
             out_dir / "A_gpurir_topdown_husky_only.mp4",
             highlight_tag="dog_husky")
render_video("/data/jzy/code/AVEngine/external/SPEAR/data/shoebox_v2_spec.json",
             audio_dir / "audio_A_gpurir_stereo.wav",
             out_dir / "A_gpurir_topdown_mixed.mp4",
             highlight_tag=None)
PYEOF

echo
echo "== [5/8] measurement runs -- howl source for Gate 1 broadband delta =="
"$SS2" - <<'PYEOF'
import sys
sys.path.insert(0, "/data/jzy/code/AVEngine/external/SPEAR/tools/spike_rlr")
sys.path.insert(0, "/data/jzy/code/AVEngine/external/SPEAR/tools")
import run_audio_pass_rlr as R
R._TAG_AUDIO_OVERRIDES["dog_husky"] = "/data/datasets/omniaudio/train-data-az-360-large/bareboneshowling_312.wav"
R.compute_rir_and_render(
    spec_path="/data/jzy/code/AVEngine/external/SPEAR/data/shoebox_v2_spec.json",
    glb_path="/data/jzy/code/AVEngine/external/SPEAR/tmp/spike_rlr/shoebox_v2_mesh.glb",
    materials_sidecar_path="/data/jzy/code/AVEngine/external/SPEAR/tmp/spike_rlr/shoebox_v2_materials.json",
    out_wav_path="/data/jzy/code/AVEngine/external/SPEAR/tmp/spike_output/raw_audio/audio_B_rlr_howl_FOA.wav",
    downmix_stereo_path="/data/jzy/code/AVEngine/external/SPEAR/tmp/spike_output/raw_audio/audio_B_rlr_howl_stereo.wav",
    quality_mode="low", verbose=False,
)
PYEOF

"$SAO" - <<'PYEOF'
import sys
sys.path.insert(0, "/data/jzy/code/AVEngine/external/SPEAR/tools/spike_rlr")
import run_audio_pass_gpurir as G
G._TAG_AUDIO_OVERRIDES["dog_husky"] = "/data/datasets/omniaudio/train-data-az-360-large/bareboneshowling_312.wav"
G.run_gpurir_pass("/data/jzy/code/AVEngine/external/SPEAR/tmp/spike_output/raw_audio/audio_A_gpurir_howl_4ch.wav")
PYEOF

echo
echo "== [6/8] spectrogram figures =="
"$SS2" "$SCRIPT_DIR/analysis/spectrogram_gen.py"

echo
echo "== [7/8] IR energy curve + metrics.json =="
"$SS2" "$SCRIPT_DIR/analysis/ir_energy_curve.py"
"$SS2" "$SCRIPT_DIR/analysis/metrics.py"

echo
echo "== [8/9] SPEAR/UE render of shoebox_v2 (Task 3) =="
# LD_PRELOAD must NOT leak into spear-env's UE launcher (nvidia driver only
# accepts the system libs when python is ss2's; spear-env python is fine
# without the preload). Unset for this one step.
unset LD_PRELOAD
export DISPLAY=:99
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
"$SPEAR_PY" "$SCRIPT_DIR/run_render_pass_shoebox_v2.py" \
    --out "$SPIKE_OUT/videos/A_gpurir_ue"
# Mux A group UE videos with the A group mixed stereo audio
for view in view0 view1 view2 view3; do
    ffmpeg -y -loglevel error \
        -i "$SPIKE_OUT/videos/A_gpurir_ue/$view.mp4" \
        -i "$SPIKE_OUT/raw_audio/audio_A_gpurir_stereo.wav" \
        -c:v copy -c:a aac -map 0:v -map 1:a -shortest \
        "$SPIKE_OUT/videos/A_gpurir_ue_${view}_with_audio.mp4"
    echo "muxed A_gpurir_ue_${view}_with_audio.mp4"
done
# Restore LD_PRELOAD for downstream steps
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libEGL.so.1:/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0"

echo
echo "== [9/9] DECISION_TABLE.md =="
"$SS2" "$SCRIPT_DIR/analysis/build_decision_table.py"

echo
echo "== DONE =="
echo "Read: $SPIKE_OUT/DECISION_TABLE.md"
echo "Videos: $SPIKE_OUT/videos/"
echo "Analysis: $SPIKE_OUT/analysis/"
