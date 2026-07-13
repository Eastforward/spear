"""Launch one artifact-gated humanoid clip in packaged SPEAR/apartment_0000."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


os.environ.setdefault("DISPLAY", ":99")
os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/avengine-matplotlib")

from run_render_pass_apartment import render_apartment  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "tmp" / "hy3d_rocketbox_template_fit_v1" / "ue_apartment_smoke"


def _append_command_log(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def execute_stage(
    *,
    stage: str,
    spec_path: Path,
    out_dir: Path,
    clip_id: str,
    render_function=render_apartment,
    finalize_function=None,
) -> None:
    """Run one explicit pipeline stage without tying CPU work to a GPU slot."""
    if stage not in {"render", "finalize", "all"}:
        raise ValueError(f"unsupported human Apartment stage: {stage!r}")
    spec_path = Path(spec_path).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if stage in {"render", "all"}:
        render_function(
            spec_path,
            out_dir,
            out_dir / "profile_per_clip.csv",
            clip_id,
        )
    if stage in {"finalize", "all"}:
        if finalize_function is None:
            from human_apartment_evidence import finalize_human_apartment_clip

            finalize_function = finalize_human_apartment_clip
        finalize_function(
            spec_path=spec_path,
            out_dir=out_dir,
            clip_id=clip_id,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec",
        default=str(DEFAULT_ROOT / "male_walk_canary_spec.json"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_ROOT / "male_walk_canary"),
    )
    parser.add_argument("--clip-id", default="male_walk_canary")
    parser.add_argument(
        "--finalize-evidence",
        action="store_true",
        help="Legacy alias for --stage all.",
    )
    parser.add_argument(
        "--stage",
        choices=("render", "finalize", "all"),
        help=(
            "Run only UE render, only CPU evidence finalization, or both. "
            "The default preserves legacy behavior."
        ),
    )
    args = parser.parse_args()
    if args.stage is not None and args.finalize_evidence:
        parser.error("--stage and --finalize-evidence are mutually exclusive")
    stage = args.stage or ("all" if args.finalize_evidence else "render")

    spec_path = Path(args.spec).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    command_log = out_dir / "command.log"
    started_at = datetime.now(timezone.utc).isoformat()
    _append_command_log(command_log, {
        "event": "start",
        "started_at": started_at,
        "argv": [sys.executable, *sys.argv],
        "spec": str(spec_path),
        "out_dir": str(out_dir),
        "clip_id": args.clip_id,
        "stage": stage,
        "finalize_evidence": bool(args.finalize_evidence),
    })
    try:
        execute_stage(
            stage=stage,
            spec_path=spec_path,
            out_dir=out_dir,
            clip_id=args.clip_id,
        )
    except BaseException as error:
        _append_command_log(command_log, {
            "event": "finish",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        })
        raise
    else:
        _append_command_log(command_log, {
            "event": "finish",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "status": "passed",
        })


if __name__ == "__main__":
    main()
