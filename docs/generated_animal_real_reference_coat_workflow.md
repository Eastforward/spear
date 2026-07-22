# Real-reference coat editing for a rigged animal mesh

This workflow changes breed-scoped coat appearance on an already accepted,
rigged animal GLB. It does not generate a replacement topology, infer a new
skeleton, or alter animation keyframes. The current implementation is a
research path; a successful coat review does not admit an asset to a formal
dataset registry.

## Authority split

- The input GLB owns geometry, UVs, silhouette, skin, skeleton, rest pose,
  orientation and `Idle`/`Walking` actions.
- A fixed four-view montage of that GLB is image 1 and remains the geometry,
  camera and layout authority during editing.
- A board of real, rights-reviewed examples is image 2 and supplies only the
  breed/coat colour, ticking and facial-coat evidence.
- Undistilled FLUX.2 Klein Base edits coat pixels once with a declared seed.
  Qwen is not required by the default route.
- Blender projects the accepted edited chroma onto the original UV texture,
  preserves its luminance/detail and exports the unchanged rig and actions.

Do not use this route to turn one body shape into another breed silhouette. A
new silhouette or morphotype needs a separately reviewed generated mesh and
binding route. Size, body build and life stage also remain independent instance
attributes; a coat edit is not evidence that those attributes are correct.

## Inputs

1. One rigged GLB with one primary skinned mesh, an active UV layer, one Base
   Color image and exactly `Idle` plus `Walking` actions.
2. Four to nine curated real photographs representing one exact
   species/breed/coat profile. Reject ambiguous breed identity, literal colour
   naming errors, anatomy defects and incompatible rights before model use.
3. The undistilled `black-forest-labs/FLUX.2-klein-base-4B` snapshot. Set
   `AVENGINE_FLUX2_KLEIN_BASE_SNAPSHOT` or pass `--snapshot` explicitly.
4. Blender 4.2.x, the `avengine-imagegen` Python environment and `ffmpeg` for
   the review video.

Generated photos, boards, textures, GLBs and videos stay under `tmp/`; only the
tools, compact manifests, tests and this workflow belong in Git.

## Reusable commands

The paths below are placeholders, not repository configuration:

```bash
SPEAR_ROOT=/path/to/SPEAR
RUN_ROOT=/path/to/ignored/run
INPUT_GLB=/path/to/accepted/animated_walk_idle.glb
BLENDER=/path/to/blender
IMAGEGEN_PY=/path/to/avengine-imagegen/bin/python
```

Build a deterministic 1024-pixel appearance board from already curated photos:

```bash
cd "$SPEAR_ROOT"
"$IMAGEGEN_PY" tools/build_animal_coat_reference_board.py \
  --input photo_01.jpg photo_02.jpg photo_03.jpg \
          photo_04.jpg photo_05.jpg photo_06.jpg \
  --output "$RUN_ROOT/appearance_board.png"
```

Render geometry-locked rest-pose views. `--front-axis` must come from the
accepted asset contract; do not infer it from the generated edit:

```bash
"$BLENDER" --background \
  --python tools/blender_render_generated_animal_coat_views.py -- \
  --input-glb "$INPUT_GLB" \
  --output-dir "$RUN_ROOT/source_views" \
  --front-axis positive-x
```

Run one FLUX edit. The target description must use the breed's real coat
terminology and observable regional distribution, not a generic RGB name:

```bash
"$IMAGEGEN_PY" tools/flux2_edit_animal_multiview_coat.py \
  --input-view-dir "$RUN_ROOT/source_views" \
  --appearance-reference-board "$RUN_ROOT/appearance_board.png" \
  --output-root "$RUN_ROOT/flux_edit" \
  --target-description "<breed-scoped coat description>" \
  --seed 20260722 --steps 40 --guidance-scale 4.0
```

Review `flux_edit/edited_montage.png` before any UV work. Reject changed panel
layout, silhouette, limb/tail count, pose, camera, background, literal dye
colours or continuous stripes when the profile requires ticking.

Project an accepted edit back to the original GLB:

```bash
"$BLENDER" --background \
  --python tools/blender_project_animal_multiview_coat.py -- \
  --input-glb "$INPUT_GLB" \
  --source-view-dir "$RUN_ROOT/source_views" \
  --edited-view-dir "$RUN_ROOT/flux_edit/edited_views" \
  --output-root "$RUN_ROOT/projected" \
  --output-stem <species_breed_coat> \
  --texture-size 2048 \
  --minimum-direct-coverage 0.40 \
  --colour-transfer-mode absolute_edited_chroma \
  --absolute-chroma-strength 0.65
```

`absolute_chroma_strength=0.65` is the reviewed Blue Abyssinian canary setting,
not a universal truth. A new breed/coat profile must declare its value and
review the result. The relative mode is retained for small corrections, but it
was too weak when the source texture was materially more saturated than the
neutral source render.

Render both `Walking` and `Idle` before promotion. The existing
`tools/blender_render_glb_animation.py` is the review renderer; the exported
GLB must read back with one skin, one skinned primitive and exactly the two
actions.

## Required checks

- Board: exact breed/coat, multiple real examples, no single lighting condition
  treated as colour truth, and sources/rights reviewed separately.
- FLUX: one declared invocation; image 1 geometry unchanged; image 2 used only
  for appearance; no seed lottery.
- Projection: explicit sRGB-to-linear input decode and sRGB bake output. A
  source-as-edited identity run must not brighten or recolour the original.
- Geometry/runtime: unchanged bounds within exporter precision, one skin, one
  skinned primitive, `Idle` and `Walking` retained.
- Visual: front/back/left/right coat consistency plus moving side/quarter views;
  reject UV seams, white/grey washout, residual generic recolour and animation
  regressions.
- Status: keep `technical_spike_only` or `research_candidate` until the normal
  geometry, deformation, contact, collision, rights and human-review gates pass.

## Blue Abyssinian canary record

The successful technical canary used a user-accepted generated Abyssinian mesh
and its existing accepted animation binding. Six real Blue Abyssinian photos
formed the appearance board. The FLUX edit correctly changed the target to
blue-grey ticking over a warm mushroom/beige ground without changing the
four-view layout. The final reviewed projection used absolute edited chroma at
strength `0.65`.

Two failures are intentionally retained as lessons:

- Relative chroma preserved too much of the original ruddy/orange texture.
- Absolute chroma at `1.0` over-corrected the head and legs toward grey-white.

The identity bake also exposed and fixed a double-sRGB conversion: an original
mean RGB near `(179, 118, 53)` incorrectly became `(217, 180, 122)` before the
fix and returned to approximately `(179, 118, 53)` afterward. The canary's
four-view direct vertex coverage was `0.454771`; uncovered vertices used the
nearest covered 3D vertex and therefore still require moving visual review.
