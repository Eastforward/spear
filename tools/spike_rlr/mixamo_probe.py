"""Probe locally available Mixamo FBX animation assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from external_data_paths import dataset_root, dataset_spec


def discover_mixamo_fbx(root: Path) -> list[Path]:
    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".fbx"),
        key=lambda p: p.relative_to(root).as_posix().lower(),
    )


def write_mixamo_probe_status(out_path: Path) -> dict:
    spec = dataset_spec("mixamo")
    root = dataset_root("mixamo")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        status = {
            "state": "missing_data",
            "dataset": spec.name,
            "root": str(root),
            "manual_action": spec.acquisition_hint,
        }
    else:
        files = discover_mixamo_fbx(root)
        state = "ready" if files else "no_assets"
        status = {
            "state": state,
            "dataset": spec.name,
            "root": str(root),
            "fbx_count": len(files),
            "fbx_files": [p.relative_to(root).as_posix() for p in files[:50]],
        }

    out_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("tmp/mixamo_probe/status.json"))
    args = parser.parse_args(argv)
    status = write_mixamo_probe_status(args.out)
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
