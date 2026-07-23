#!/usr/bin/env python3
"""Measure one generated quadruped's fixed mouth emitter in canonical space.

The accepted generated-animal pipeline normalizes every runtime GLB to +X
forward, +Y up and +Z left before this step.  This tool measures the concrete
asset instead of copying an offset from a species template.  It does not
require or infer mouth animation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys


TOOLS_DIR = Path(__file__).resolve().parent
SPEAR_ROOT = TOOLS_DIR.parent
if str(SPEAR_ROOT) not in sys.path:
    sys.path.insert(0, str(SPEAR_ROOT))

from tools import blender_build_generated_animal_instance_ofat as generated  # noqa: E402


def _arguments() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _arguments()
    source = generated.require_file(args.input_glb, "generated animal GLB")
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    mesh, armature = generated.import_asset(source)
    emitter = generated.derive_muzzle_emitter(mesh, armature)
    result = {
        "schema": "avengine_generated_animal_emitter_measurement_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input": {
            "path": str(source),
            "sha256": _sha256(source),
            "size_bytes": source.stat().st_size,
        },
        "canonical_front_axis": "positive-x",
        "emitter_anchor": emitter,
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "GENERATED_ANIMAL_EMITTER_OK "
        f"offset={emitter['emitter_offset_m']} output={output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
