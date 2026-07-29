# Generated-animal hardened route — operational runbook

First validated end-to-end on the Shiba Inu shipping test (2026-07-26, see
`shiba_hardened_route_shipping_test_20260726.md`).  This is the
copy-pasteable procedure for taking a NEW breed from zero to a
UE-registered research asset, with every known pitfall inlined at the step
where it bites.  Conventions: `SPEAR=/data/jzy/code/AVEngine/external/SPEAR`,
workspace `WS=$SPEAR/tmp/new_animal_assets/<breed>_<yyyymmdd>_01`.

Human touchpoints are exactly three: 2D canonical accept, head-end one-click
confirm, final animation review.  Everything else is automated and
fail-closed.

## Python environments (wrong env fails silently or late)

| stage | interpreter |
|---|---|
| FLUX / ISNet / Pixal runner | `/data/jzy/miniconda3/envs/avengine-imagegen/bin/python` |
| UAT cook (`tools/run_uat.py` imports `spear`) | `/data/jzy/miniconda3/envs/spear-env/bin/python` |
| TokenRig (SkinTokens) | `/data/jzy/code/AVEngine/external/SkinTokens/.venv/bin/python` (pinned; do NOT migrate to conda without a same-seed reproducibility canary) |
| geometry audit (trimesh) | `/data/jzy/miniconda3/envs/avengine-3dgen/bin/python` |
| SPEAR RPC / registry tests / review pages | `/data/jzy/miniconda3/envs/spear-env/bin/python` |
| Blender stages | `/data/jzy/.local/bin/blender` (4.2.1) |

General pitfalls that recur everywhere:
- Published output trees are sealed read-only (atomic staging publish);
  write inspection artifacts to a sibling dir, never into them.
- Every fail-closed tool refuses to overwrite; failed attempts keep their
  dir and the retry uses a new suffix (`retry1`, `_v2`), never deletion.
- Long Blender/UE steps exceed the default 2-minute shell timeout; run
  them in background tasks.
- zsh does not word-split `$var` — loops with packed specs need bash.
- Python output is fully buffered when redirected; an empty log does not
  mean a dead process (check the pid and CPU%).
- When the owner's training jobs saturate `/dev/sdb` (check `iostat -x`:
  90%+ util, 100+ MB/s), FLUX/Pixal workers sit in D-state for many
  minutes during model load with near-zero RSS growth.  That is disk
  starvation, not a hang — do not kill and retry (the retry queues behind
  the same disk).
- A first defect is fixed in the existing stage.  It becomes a permanent
  generic gate only after the same defect is reproduced on two different
  assets.  Do not add a breed-specific tool or route for a single failure.

## 1. Attribute profile (one JSON, no new reference photos)

`data/controlled_source_attributes_v1/candidate_profiles/animal/`
`<species>_<breed>_four_limb_rest_side_clay_v1.json` — copy the Labrador or
Shiba profile and adapt: taxonomy, fixed_attributes (ear/tail/coat pattern),
singleton sampled domains (one-shot policy), value_labels for every
placeholder, physical shoulder height, qa_contract questions.  The schema
validator requires EXACTLY 17 top-level fields; `base_acquisition_policy`
must equal `tools/controlled_animal_one_shot_policy.py` verbatim; article
grammar in `positive_template` is hand-written ("a curled tail" vs "an
otter tail").  Consult `generated_animal_morphotype_readiness_checklist.md`
for risk scoring before spending GPU.

For a morphology that needs shared reconstruction guards, declare
`generation_contract.morphotype_traits` in the same profile (use `{}` when
only the long-coat guard is needed).  The current evidence-backed traits are
short legs and a brachycephalic muzzle.  A
short-leg profile must declare a 10--15 degree `source_view_contract` yaw;
the compiler then adds the longitudinal limb staggering and continuous
armpit/groin background-corridor guard.  Long-coat protection is derived
from the existing `coat_length`, so it is not declared a second time.

## 2. FLUX canonical image (one shot, owner 2D gate)

```
PY=/data/jzy/miniconda3/envs/avengine-imagegen/bin/python
$PY tools/build_controlled_source_asset_inputs.py --profile <profile.json> \
  --count-per-profile 1 --seed <yyyymmddNN> --plan-id <breed>_base_<date>_v1 \
  --split-salt avengine_<breed>_<date> --output-dir $WS/inputs
$PY tools/prepare_controlled_source_asset_execution.py \
  --input-dir $WS/inputs --output-dir $WS/preflight
$PY tools/run_controlled_animal_flux2_jobs.py \
  --preflight $WS/preflight/execution_preflight.json \
  --output-root $WS/flux --gpu <free-gpu>
```
No seed retries ever; a rejected image is recorded and a re-attempt changes
the declared request.  Owner reviews `candidates/*/candidate.png`; decision
file `review_inputs/flux2_2d_decisions.json` (v2 schema: exactly 4 top keys,
10 fields per decision incl. 8 hard gates), then:
```
$PY tools/review_controlled_animal_flux2_candidates.py \
  --flux-batch $WS/flux/flux2_batch_manifest.json \
  --decisions $WS/review_inputs/flux2_2d_decisions.json \
  --output-root $WS/flux_2d_review
```

For an unproven morphology, use one predeclared bounded exploration batch
instead of serial workspaces:
```
$PY tools/build_controlled_source_asset_inputs.py --profile <profile.json> \
  --count-per-profile <N> --seed <declared-seed> --plan-id <plan-id> \
  --split-salt <salt> --output-dir $WS/inputs
$PY tools/run_controlled_animal_flux2_jobs.py \
  --preflight $WS/preflight/execution_preflight.json \
  --output-root $WS/flux --gpu <free-gpu> --bounded-exploration
```
All candidates must receive the normal exact-hash hard-gate review.  Each
profile must have exactly one approval (frozen) or zero approvals
(`exploration_exhausted`); multiple approvals fail closed.  The frozen
candidate's existing bytes go directly to Pixal3D.  Do not regenerate the
selected seed.  Put multiple profiles in one input/preflight batch to pay
one model load per active GPU worker.

## 3. Pixal3D

```
$PY tools/prepare_controlled_animal_pixal_inputs.py \
  --review-batch $WS/flux_2d_review/review_batch_manifest.json \
  --output-root $WS/pixal_inputs --pixal-output-root $WS/pixal   # ISNet, ~3 min
$PY tools/run_controlled_animal_pixal_jobs.py \
  --pixal-inputs $WS/pixal_inputs/pixal_inputs_manifest.json \
  --output-root $WS/pixal --gpu <free-gpu>
```
Cold model load ~20 min before inference (~1 min); batch multiple assets to
amortize.  Since `9f940c34` symlinked invocation paths validate correctly.

## 4. Geometry audit + watertight proxy

```
/data/jzy/miniconda3/envs/avengine-3dgen/bin/python tools/audit_quadruped_i23d_geometry.py \
  --mesh <breed>_raw=$WS/pixal/*/pixal_raw_1024.glb --output $WS/inspection/i23d_geometry_audit.json
```
Membrane check (the legacy-Collie defect): solid connected components below
0.4 rest height — one giant blob = welded legs, stop and reassess.  Then
the watertight proxy (Collie v3 recipe):
```
blender -b --python-exit-code 2 --python tools/blender_create_watertight_textured_proxy_mesh.py -- \
  --source <raw.glb> --output $WS/watertight_r200_v1/<breed>_watertight.glb \
  --manifest $WS/watertight_r200_v1/manifest.json \
  --voxel-resolution 200 --target-faces 100000 --smooth-iterations 1 \
  --shrinkwrap-strength 0.0 --post-shrinkwrap-smooth-iterations 2 \
  --torso-fold-repair-iterations 6 --attribute-transfer-backend bake \
  --bake-resolution 2048 --base-color-encoding-policy preserve-bake
```
Render a turntable (`tools/blender_render_forward_turntable.py`), encode
mp4s, owner takes a static look.

## 5. TokenRig (isolated per invocation)

Use the scripted Shiba precedent with seed 42, a dedicated free port, GPU
pinned, hygiene `sitecustomize` on `PYTHONPATH`, `--use_transfer` WITHOUT
`--use_skeleton`, and checkpoint `grpo_1400.ckpt`.  The runtime patch forces
the child service to `127.0.0.1` and writes a per-process generation into its
marker/load audit.  Success = `[OK] Exported` in `run.log`, a non-empty
`tokenrig_native.glb`, and matching runtime markers.

Do not use or advertise the experimental resident helper as batch reuse.
The tracked upstream `demo.py` still starts its own bpy child unconditionally,
and the pickle-based local API has no client authentication.  A resident
process would extend that exposure without eliminating the child start.
Warm reuse is deferred until there is an explicit authenticated consumer
launcher with per-job audit binding.

## 6. Forward contract chain (single-point forward truth)

```
blender -b --python-exit-code 2 --python tools/blender_estimate_generated_animal_forward.py -- \
  --input <tokenrig_native.glb> --output $WS/forward_contract_v1/forward_estimate.json
# auto-normalize by the ESTIMATE, render arrow turntable + 2 candidate heros,
# build the confirm page:
tools/build_forward_review_page.py --asset-workspace <ws> \
  --normalized-turntable-dir ... --estimate-json ... --output-html ...
```
Owner one-click (confirm / flip-180 / reject-to-manual).  Then:
```
tools/build_generated_animal_forward_declaration.py --asset-workspace <ws> \
  --input-glb <tokenrig_native.glb> --estimate-json ... \
  --confirmed-front-yaw-deg <value> --head-end-evidence <decision.json> \
  --head-end-decision-source human_confirming_estimator --output forward_declaration.json
```
Motion basis is a DONOR CONSTANT (`quaternius_universal_quadruped_v1` =
yaw0/matched).  Needing anything else means the declaration is wrong; fix
it there, never at retarget.  Declaration-mode execution authenticates both
the declared target GLB (path, size and SHA-256) and the donor tag's registered
GLB bytes before creating an output directory.  The old free-parameter mode
is retained only for `--validate-only` plan inspection and cannot execute or
publish a hardened-route result.

## 7. Review runner (heading -> leveling -> retarget -> gait -> deformation -> repair/re-audit -> renders)

```
/data/jzy/miniconda3/envs/spear-env/bin/python tools/run_target_native_generated_quadruped_review.py \
  --target-rig-glb <tokenrig_native.glb> \
  --forward-declaration $WS/forward_contract_v1/forward_declaration.json \
  --source-motion-glb /data/jzy/code/AVEngine/assets/mesh_library/quaternius_animalpack/Dog.glb \
  --output-root $WS/review_run_v1 --blender /data/jzy/.local/bin/blender
```
`mesh-foot-bottoms` (commit `4e4fbf34`) is the correct plane source for
generated rigs: bone tails protrude unevenly below the mesh and bone-tip
leveling leaves the asset floating and pitched.  It is now the runner
default, and a missing mesh contact band fails closed instead of silently
falling back to a bone endpoint.  `--preview-only` exists for cheap triage.
Gait audit fails closed on backward/sideways walks.

The v3 runner authenticates every stage boundary, not only the subprocess
exit code: heading evidence and rigid preservation; rig coordinate/skin
invariants; support-plane evidence, exact limits and four non-trivial mesh
contact bands; retarget target/donor/export identities, fixed solver and +X
Walk/Idle export; gait semantics; deformation thresholds, both required
actions and sample counts; and the exact gentle-repair recipe plus protected
topology/animation fingerprints.  A manifest that merely writes
`status=pass` or changes a threshold no longer reaches rendering.

## 8. Deterministic weight repair (standard stage, not an exception)

TokenRig native weights exceed the Walking gate at full amplitude on every
measured generated asset.  The review runner now defaults to
`--weight-repair-policy auto`: a non-passing initial deformation audit
triggers the reviewed gentle recipe, authenticates its authority manifest,
and re-runs gait and deformation before any render.  Repair incomplete,
post-repair manual review and post-repair rejection all stop the run.
`--weight-repair-policy never` is fail-closed; `always` is available for
controlled revalidation.

The standalone tool now also defaults to the same gentle recipe (verified
Collie + Shiba at amp 1.0).  An explicit equivalent invocation is:
```
blender -b --python-exit-code 2 --python tools/blender_repair_animated_quadruped_weight_stretch.py -- \
  --input <review_run>/04_motion/target_animated.glb --output <repaired.glb> \
  --manifest <manifest.json> --front-axis positive-x \
  --repair-mode component-parent-lock --component-rings 4 \
  --extension-threshold 0.02 --maximum-passes 6 --inner-iterations 4
```
The runner performs the required repaired-GLB gait/deformation audits and
six-view re-render automatically.  Owner final review remains the third and
last touchpoint.

## 9. UE import (headless commandlet, no GUI)

Pitfalls first: embedded WebP textures MUST be transcoded
(`tools/transcode_glb_webp_to_png.py`) or Interchange silently imports
nothing; `tag` must start with `pixal_`; trust `ue_import_result.json`
not the editor exit code.
```
# ue_import_jobs.json: schema pixal_animal_ue_import_batch_v1 (copy Shiba's)
PIXAL_ANIMAL_IMPORT_MANIFEST=$WS/ue_import_jobs.json \
PIXAL_ANIMAL_IMPORT_RESULT=$WS/ue_import_result.json \
/data/UE_5.5/Engine/Binaries/Linux/UnrealEditor-Cmd \
  $SPEAR/cpp/unreal_projects/SpearSim/SpearSim.uproject \
  -RenderOffscreen -graphicsadapter=<free-gpu> -unattended -nop4 -nosplash -NoSound \
  -run=pythonscript -script=$SPEAR/tools/import_pixal_animal_batch_editor.py \
  -AbsLog=$WS/ue_import.log
```
Cook needs the FULL flag set or it is a 2-second no-op:
```
$PY tools/run_uat.py --unreal-engine-dir /data/UE_5.5 --skip-cook-default-maps \
  -build -cook -stage -package -archive -pak
```

## 9b. Orbit render + runtime z evidence (after the cook)

`tools/render_gate_animal_orbit.py` needs a live X display or UE SIGSEGVs
after "Could not detect DISPLAY": keep a resident Xvfb
(`nohup setsid Xvfb :99 -screen 0 1280x720x24 &`, then `DISPLAY=:99` +
`VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json`).  NEVER pipe the
command through `| tail` — the pipeline exit code becomes tail's 0 and the
failure log is swallowed; redirect the full output to a file instead.
The legacy default `--scale 0.15` was tuned for ~5 m template meshes and
renders a real-scale generated asset too small to inspect; run
`--scale 1.0 --radius-cm 160 --camera-height-cm 60 --look-height-cm 35`
for both the supervision frames and the component-space z evidence
(`ground_snap.z_correction_cm` scales linearly with `--scale`; verified
Shiba -0.177 cm @0.15 = -1.181 cm @1.0 exactly).

## 10. Registry (measurements only, never copied between breeds)

- Muzzle emitter: `tools/blender_measure_generated_animal_emitter.py`
  (asset-specific; Shiba `[0.3727, 0.4866, 0]`, Labrador `[0.454, 0.585, 0]`).
- UE Z correction: with mesh-foot leveling the theoretical value is ~0;
  verify against the orbit render's `ground_snap.z_correction_cm` and
  record how it was measured.
- `ue_anatomical_basis_bones` (five roles, per-instance, never index-copied):
  derive from the shipped GLB rest hierarchy (measured world positions).
  rear/body = root bone; front = the axial-head-chain bone nearest the
  measured muzzle emitter (Shiba: bone_4, 0.07 deg horizontal yaw — the
  chain LEAF may be skull-top with several degrees of lateral offset);
  left/right front paws by +Z = anatomical RIGHT in the GLB frame
  (forward +X, up +Y; right = forward x up).  CAUTION: the retarget
  manifest's `front_side_positive/negative` naming is the OPPOSITE of the
  GLB Z sign (side sign is assigned in the Blender frame where glTF +Z
  maps to -Y) — measure positions, never trust the side names.  The
  emitter measurement v2 records
  `avengine_local_x_forward_y_up_z_right_m`.  Historical v1 measurements
  used an incorrect `...z_left_m` label; do not rewrite those immutable
  artifacts or propagate that label.  Their sagittal-plane offsets have
  `z=0`, so the correction changes provenance semantics, not the numbers.
- `walk_phase_period_frames`: measure the shipped GLB Walking clip
  duration and multiply by the Timeline 15 fps (Shiba: 1.666667 s -> 25).
- Coat profile MUST be registered in the appearance contract
  (`avengine/appearance/contracts.py` COAT_PROFILE_DOMAINS +
  REALIZATION_RULES) before the runtime registry entry: the registry
  validator cross-checks and fails closed on unregistered coats.
- Entry goes to `examples/runtime/source_asset_runtime_profiles.json`
  (habitat-native repo); run `tests/unit/test_runtime_profiles.py`.
- Runtime readback gates (anatomical forward <=25 deg, floor <=5 cm,
  bounds) run via `tools/m6y/run_spear_apartment_canary.py` against a UE
  input bundle; dry-run first.

## Review infrastructure

Local review server: `python3 -m http.server 8765 --bind 127.0.0.1` from
`tmp/new_animal_assets/` (VSCode auto-forwards the port).  Confirm/final
pages are self-contained HTML from `tools/build_forward_review_page.py`;
owner decisions are recorded verbatim as JSON review records next to the
asset workspace.
