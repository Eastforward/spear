# Hunyuan Mesh Direction Audit Pipeline

Trust-but-verify pipeline for Hunyuan 3D meshes.

## Directory convention

```
tmp/hy3d_batch/
  pending/{tag}/    ← Auto-orient run, awaiting human review
  approved/{tag}/   ← Human-approved (safe for downstream)
  rejected/{tag}/   ← Human rejected (with override record if applicable)
```

## Workflow

1. **Generate meshes** (Hunyuan3D pipeline drops them in `pending/{tag}/mesh.glb`)
2. **Auto-orient**: `python tools/spike_rlr/auto_orient_ingest.py --pending-dir tmp/hy3d_batch/pending`
3. **Human audit**: Start web UI, review, click Approve or Reject.
4. **Downstream pipelines** only read `approved/` (enforced by
   `tools/spike_rlr/review_gate.py::assert_mesh_approved`).

## Gate integration

Any pipeline that reads a Hunyuan tag must call:
```python
from review_gate import assert_mesh_approved, resolve_approved_mesh_path
assert_mesh_approved(tag)   # raises if not human-approved
mesh_path = resolve_approved_mesh_path(tag)  # returns mesh_oriented.glb path
```

## Sidecar `direction.json` schema

Written by `auto_orient_ingest.py`; updated by `review_ui_server.py`.

```json
{
  "mesh_source": "tmp/hy3d_batch/pending/dog_golden/mesh.glb",
  "mesh_oriented": "tmp/hy3d_batch/pending/dog_golden/mesh_oriented.glb",
  "algorithm_version": "auto_orient_v1",
  "detected_at": "2026-07-08T...Z",
  "detection": {
    "head_direction_original_mesh_frame": [0.98, 0.05, -0.19],
    "rotation_applied_to_align_to_plus_x": [[...], [...], [...]],
    "signals": {"leg_spacing_vote": 3, "high_verts_vote": 2, "mass_end_vote": 1},
    "total_votes": 6,
    "unanimous": true,
    "confidence": 0.95
  },
  "human_approved": true,
  "human_approved_by": "jzy",
  "human_approved_at": "2026-07-08T...Z",
  "human_notes": null,
  "human_override": null,
  "quarantined": false
}
```
