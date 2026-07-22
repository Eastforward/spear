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
  --front-axis positive-x \
  --write-neutral-shading-pass
```

Run one FLUX edit. The target description must use the breed's real coat
terminology and observable regional distribution, not a generic RGB name:

```bash
"$IMAGEGEN_PY" tools/flux2_edit_animal_multiview_coat.py \
  --input-view-dir "$RUN_ROOT/source_views" \
  --appearance-reference-board "$RUN_ROOT/appearance_board.png" \
  --output-root "$RUN_ROOT/flux_edit" \
  --target-description "<breed-scoped coat description>" \
  --coat-detail-instruction "<regional pattern semantics visible in the real references>" \
  --negative-description "<breed/coat-specific forbidden markings or literal dye colours>" \
  --seed 20260722 --steps 40 --guidance-scale 4.0
```

The editor's default prompt is species-neutral. Ticking, merling, masks,
blazes, socks, points and their forbidden alternatives belong in the two
explicit coat-specific arguments above; they must not be hard-coded from a
previous cat or dog experiment.

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
  --edited-mask-dir "$RUN_ROOT/edited_foreground_masks" \
  --neutral-shading-view-dir "$RUN_ROOT/source_views/neutral_shading" \
  --output-root "$RUN_ROOT/projected" \
  --output-stem <species_breed_coat> \
  --texture-size 2048 \
  --minimum-direct-coverage 0.40 \
  --colour-transfer-mode neutral_shading_division \
  --absolute-rgb-strength 1.0 \
  --view-fusion-mode dominant_facing_view
```

The command above is the reviewed final Border Collie path. The foreground
mask directory is produced by the repository's pinned ISNet worker; it must
contain `front.png`, `back.png`, `left.png` and `right.png` from the accepted
FLUX edit. `absolute_edited_chroma` with `absolute_chroma_strength=0.65` remains
the reviewed Blue Abyssinian canary setting, not a universal truth. A new
breed/coat profile must declare its parameters and review the result.

`pattern_luminance_strength` is independently reviewed and defaults to `0.0`.
Leave it at zero for colour-only changes. A spatial coat such as blue merle may
use a non-zero value to transfer the edited/source dark-light pattern onto the
original UV while still keeping the original mesh, rig and texture detail. It
is not a global brightness or RGB multiplier. `1.0` follows the measured
edited/source difference; a reviewed value up to `2.0` may amplify that spatial
difference when the source Base Color is so dark that the pattern otherwise
collapses in a neutral render.

If both ratio modes still collapse a strongly patterned coat into an almost
black source texture, `absolute_edited_rgb` is the explicit fallback. It
projects FLUX's four-view, per-surface RGB field and blends it with the original
UV using `absolute_rgb_strength`; it never applies one global material colour
or RGB multiplier. Because it can also transfer neutral review lighting, it
requires front/back/left/right and moving visual review rather than automatic
promotion.

For a dark source texture and a strongly different coat, prefer
`relative_rgb`. It applies the spatial per-channel ratio between the FLUX edit
and the geometry-locked source render to the original UV. Because both views
share the same camera and neutral illumination, most preview lighting divides
out while the local coat change remains. This is distinct from the older
`relative_chroma` mode, which intentionally restores original luminance and
can therefore keep a near-black source near-black.

FLUX is allowed to refine fur outlines, so an edited silhouette can differ by
a few pixels from the geometry-locked source render. The projector therefore
fits the neutral background in every edited view and rejects background-like
samples before UV baking. Missing surface vertices are filled only through
mesh edges plus spatially coincident vertices split by export-time UV/normal
seams, not by an arbitrary nearest point in 3D. This prevents a nearby leg,
tail or opposite body side from donating an unrelated coat patch. A truly
separate component with no valid view evidence keeps its original texture.
The foreground decision combines chromatic contrast for coloured markings with
Rec.709 luminance contrast for neutral black or white fur. A plain mid-grey
background or cast shadow must not become an albedo patch.
For final projection, generate four foreground masks with the repository's
pinned ISNet worker and pass their directory as `--edited-mask-dir`; mask alpha
then replaces the heuristic as foreground authority. The heuristic remains a
fast diagnostic fallback, not final coat evidence.

Also render `--write-neutral-shading-pass` from the geometry-locked source and
use `--colour-transfer-mode neutral_shading_division` together with
`--neutral-shading-view-dir`. The neutral-grey pass estimates illumination at
the exact surface/camera location; dividing it out prevents preview highlights
and shadows from becoming permanent Base Color. `absolute_edited_rgb` remains
a diagnostic fallback and must not be treated as final albedo evidence.

The baked PNG is inserted into the accepted GLB by replacing only its embedded
Base Color bufferView. The projector does not re-export mesh, UV, skin,
skeleton or animation data through Blender. Every non-target bufferView and the
protected node/mesh/skin/accessor/animation JSON sections must read back
unchanged.

For a discrete blaze, sock, mask or merle boundary, use
`--view-fusion-mode dominant_facing_view`. Each surface vertex then takes the
valid view whose camera most directly faces its normal. This avoids averaging
the front-view blaze with a side-view ear or cheek into an artificial grey
patch. `weighted_average` remains suitable for diffuse, view-consistent coats.

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
four-view direct vertex coverage was `0.454771`. That historical run used a
nearest covered 3D vertex for uncovered areas; the production projector now
uses mesh-edge and coincident-export-seam propagation and leaves disconnected
components on the original texture.

## Border Collie cross-validation and instance attributes

The Border Collie cross-validation used an independently generated FLUX →
Pixel3D mesh, not a reshaped Beagle or Labrador template. After topology/PBR
repair, TokenRig, cardinal heading normalization, four-foot support-plane
leveling and reviewed motion retarget, the accepted mesh remains the authority
for all instance variants.

`tools/blender_build_generated_animal_instance_ofat.py` emits one baseline and
eight one-factor-at-a-time variants:

- size: `small / medium / large` through one uniform instance root;
- body build: `slim / standard / stocky` through bounded semantic torso
  deformation (`0.84 / 1.00 / 1.16`);
- life stage: `young / adult / senior` through bounded head proportions
  (`1.12 / 1.00 / 0.97`); senior additionally receives a local semantic-UV
  muzzle greying cue that preserves already-light fur and never uses a global
  RGB material factor;
- breed-scoped coat: `black_white / blue_merle / red_white`, where both
  nonbaseline coats must provide the real-reference, undistilled-FLUX
  multiview projection evidence described above.

The reviewed Border Collie batch measured torso RMS ratios of approximately
`0.911 / 1.000 / 1.092` and head RMS ratios of approximately
`1.095 / 1.000 / 0.976`. All variants retain one skin, one skinned primitive,
and exact `Idle` and `Walking` action names. Coat GLBs must additionally match
the baseline topology, UV, skin weights, rest skeleton and action-keyframe
signatures exactly; only the embedded Base Color PNG payload may differ.

Every emitted instance, including FLUX coat variants, is re-imported and sent
through the same runtime preparation. The accepted rest mesh is translated as
one rigid asset so its lowest sole vertex is at local `Z=0`; geometry, skin and
animation data are not baked or rewritten by grounding. A fixed, asset-specific
mouth emitter is then derived from the forward part of that concrete mesh's
semantic head region and recorded in AVEngine local coordinates (`X` forward,
`Y` up, `Z` left). This deliberately simple anchor is recomputed after size,
body-build and life-stage realization, so Apartment RIR binding uses the actual
instance height rather than a dog- or cat-wide template height.
