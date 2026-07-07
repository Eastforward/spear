#!/usr/bin/env bash
# End-to-end pipeline for apartment_v1: single hand-tuned Plan-1 demo clip.
#
# Produces (all under tmp/spike_output_apartment/):
#   mesh/apartment_v1_mesh.glb + apartment_v1_materials.json (RLR mesh)
#   videos/apartment_v1_view0.mp4                            (UE render, silent)
#   videos/apartment_v1_view0_with_audio.mp4                 (muxed binaural)
#   videos/topdown_apartment_v1.mp4                          (topdown, muxed)
#   videos/apartment_v1_side_by_side_view0.mp4               (final deliverable)
#   binaural_native/audio_B_rlr_LOW_binaural_native*.wav     (native binaural)
#   raw_audio_hq/audio_B_rlr_HIGH_FOA*.wav                   (4-ch FOA)
#   apartment_v1_metadata.json                               (per-frame metadata)
#   profile_per_clip.csv                                     (Level-2 profile)
#   profile_stage_summary.txt                                (Level-1 summary)

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export DISPLAY=:99
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json

SPEAR_PY=/data/jzy/miniconda3/envs/spear-env/bin/python
SS2_PY=/data/jzy/miniconda3/envs/ss2/bin/python
LD_PRE=/usr/lib/x86_64-linux-gnu/libEGL.so.1:/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0

OUT=tmp/spike_output_apartment
MESH_DIR=tmp/spike_rlr

echo "=== [1/8] Generate apartment shell mesh (from data/apartment_shell_map.json) ==="
$SS2_PY tools/spike_rlr/gen_mesh_apartment.py

echo "=== [2/8] UE render pass (apartment, 1 forward camera, 90 deg FOV) ==="
$SPEAR_PY tools/spike_rlr/run_render_pass_apartment.py

echo "=== [3/8] RLR audio: binaural LOW quality ==="
LD_PRELOAD=$LD_PRE $SS2_PY tools/spike_rlr/run_audio_pass_rlr.py \
    --spec data/apartment_v1_spec.json \
    --mesh $MESH_DIR/apartment_v1_mesh.glb \
    --materials $MESH_DIR/apartment_v1_materials.json \
    --out $OUT/binaural_native/audio_B_rlr_LOW_binaural_native.wav \
    --channel-layout binaural --quality low

echo "=== [4/8] RLR audio: FOA HIGH quality ==="
LD_PRELOAD=$LD_PRE $SS2_PY tools/spike_rlr/run_audio_pass_rlr.py \
    --spec data/apartment_v1_spec.json \
    --mesh $MESH_DIR/apartment_v1_mesh.glb \
    --materials $MESH_DIR/apartment_v1_materials.json \
    --out $OUT/raw_audio_hq/audio_B_rlr_HIGH_FOA.wav \
    --stereo-out $OUT/raw_audio_hq/audio_B_rlr_HIGH_stereo.wav \
    --channel-layout ambisonics --quality high

echo "=== [5/8] Compute per-clip metadata (DRR / azi-ele / amp gain) ==="
$SS2_PY tools/spike_rlr/compute_acoustic_metadata.py

echo "=== [6/8] Topdown 2D render (with binaural audio muxed) ==="
$SPEAR_PY tools/spike_rlr/render_topdown_2d.py \
    --spec data/apartment_v1_spec.json \
    --audio $OUT/binaural_native/audio_B_rlr_LOW_binaural_native.wav \
    --out $OUT/videos/topdown_apartment_v1.mp4

echo "=== [7/8] Mux UE video + binaural + side-by-side ==="
ffmpeg -y -loglevel error \
       -i $OUT/videos/apartment_v1_view0.mp4 \
       -i $OUT/binaural_native/audio_B_rlr_LOW_binaural_native.wav \
       -c:v copy -c:a aac -shortest \
       $OUT/videos/apartment_v1_view0_with_audio.mp4
ffmpeg -y -loglevel error \
       -i $OUT/videos/apartment_v1_view0_with_audio.mp4 \
       -i $OUT/videos/topdown_apartment_v1_silent.mp4 \
       -filter_complex "[0:v]scale=640:480[a];[1:v]scale=640:480[b];[a][b]hstack=inputs=2[v]" \
       -map "[v]" -map 0:a -c:a copy \
       $OUT/videos/apartment_v1_side_by_side_view0.mp4

echo "=== [8/8] Print stage summary ==="
$SS2_PY -c "
import sys, csv
from pathlib import Path
sys.path.insert(0, 'tools/spike_rlr')
from profiling import StageTimer, print_stage_summary
csv_path = Path('$OUT/profile_per_clip.csv')
if csv_path.exists():
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            StageTimer.aggregate[row['stage']] = \
                StageTimer.aggregate.get(row['stage'], 0.0) + float(row['seconds'])
print(print_stage_summary(total_clips=1, out_path=Path('$OUT/profile_stage_summary.txt')))
"

echo "=== DONE. Final deliverable: $OUT/videos/apartment_v1_side_by_side_view0.mp4 ==="
