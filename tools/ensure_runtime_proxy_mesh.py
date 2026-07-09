"""Ensure an approved Hunyuan tag has a current runtime proxy mesh.

This script has no heavy mesh dependencies.  It validates sidecar hashes and
only launches Blender when mesh_runtime.glb is missing or stale.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SPIKE_RLR_DIR = REPO_ROOT / "tools" / "spike_rlr"
if str(SPIKE_RLR_DIR) not in sys.path:
    sys.path.insert(0, str(SPIKE_RLR_DIR))

from review_gate import MeshNotApprovedError, approved_mesh_record  # noqa: E402
from runtime_proxy_mesh import (  # noqa: E402
    DEFAULT_TARGET_FACES,
    RUNTIME_PROXY_MESH_NAME,
    RUNTIME_PROXY_META_NAME,
    load_current_runtime_proxy_record,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--approved-dir", default=str(REPO_ROOT / "tmp/hy3d_batch/approved"))
    p.add_argument("--target-faces", type=int, default=DEFAULT_TARGET_FACES)
    p.add_argument("--blender", default=os.environ.get(
        "BLENDER", "/data/jzy/blender/blender-4.2.1-linux-x64/blender"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    approved_dir = Path(args.approved_dir)

    try:
        rec = approved_mesh_record(args.tag, approved_dir=approved_dir)
    except MeshNotApprovedError as exc:
        print(f"RUNTIME_PROXY_SKIP tag={args.tag} reason={exc}", flush=True)
        return 0

    source = Path(rec["mesh_path"])
    tag_dir = source.parent
    output = tag_dir / RUNTIME_PROXY_MESH_NAME
    metadata = tag_dir / RUNTIME_PROXY_META_NAME
    current = load_current_runtime_proxy_record(
        tag_dir,
        source_mesh_sha256=rec["mesh_sha256"],
        target_faces=args.target_faces,
    )
    if current is not None and not args.force:
        print(f"RUNTIME_PROXY_OK {output}", flush=True)
        return 0

    cmd = [
        args.blender,
        "--background",
        "--python",
        str(REPO_ROOT / "tools/blender_create_runtime_proxy_mesh.py"),
        "--",
        "--source",
        str(source),
        "--output",
        str(output),
        "--metadata",
        str(metadata),
        "--target-faces",
        str(args.target_faces),
    ]
    print(f"RUNTIME_PROXY_BUILD tag={args.tag} target_faces={args.target_faces}", flush=True)
    subprocess.run(cmd, check=True)

    refreshed = load_current_runtime_proxy_record(
        tag_dir,
        source_mesh_sha256=rec["mesh_sha256"],
        target_faces=args.target_faces,
    )
    if refreshed is None:
        raise SystemExit(f"runtime proxy did not validate after Blender run: {output}")
    print(f"RUNTIME_PROXY_DONE {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
