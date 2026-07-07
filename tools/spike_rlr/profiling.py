"""Pipeline profiling utilities: Level-1 aggregate + Level-2 per-clip CSV.

Usage:
    from profiling import StageTimer, print_stage_summary

    with StageTimer("scene_gen", clip_id="apartment_v1_000",
                    csv_path=Path("tmp/spike_output_apartment/profile_per_clip.csv"),
                    flags=["occluded_by_furniture"]):
        ...do work...

    # at pipeline end
    print(print_stage_summary(total_clips=1,
                              out_path=Path("tmp/.../profile_stage_summary.txt")))

Level 1 = per-stage aggregate accumulated in StageTimer.aggregate class dict.
Level 2 = per-clip per-stage rows appended to CSV (columns:
clip_id, stage, seconds, retry_count, flags_json).
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Optional


class StageTimer:
    aggregate: dict[str, float] = {}

    def __init__(self, stage_name: str, clip_id: str,
                 csv_path: Optional[Path] = None,
                 flags: Optional[list[str]] = None,
                 retry_count: int = 0):
        self.stage = stage_name
        self.clip_id = clip_id
        self.csv_path = Path(csv_path) if csv_path else None
        self.flags = flags or []
        self.retry_count = int(retry_count)
        self._t0: Optional[float] = None

    def __enter__(self):
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - (self._t0 or time.time())
        self.__class__.aggregate[self.stage] = \
            self.__class__.aggregate.get(self.stage, 0.0) + elapsed
        if self.csv_path is not None:
            self._append_csv(elapsed)
        return False  # never suppress exceptions

    def _append_csv(self, elapsed: float):
        header = ["clip_id", "stage", "seconds", "retry_count", "flags_json"]
        row = [self.clip_id, self.stage, f"{elapsed:.4f}",
               str(self.retry_count), json.dumps(self.flags)]
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.csv_path.exists()
        with self.csv_path.open("a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(header)
            w.writerow(row)


def reset_aggregate():
    """Clear the class-level aggregate. Used by tests for isolation."""
    StageTimer.aggregate.clear()


def print_stage_summary(total_clips: int, out_path: Optional[Path] = None) -> str:
    """Format the aggregate into a text table; optionally write to a file.

    Returns the formatted string (also written to out_path when provided).
    """
    total = sum(StageTimer.aggregate.values()) or 1e-9
    lines = [
        "=" * 65,
        f"Pipeline stage summary ({total_clips} clip(s))",
        "=" * 65,
    ]
    for stage, sec in sorted(StageTimer.aggregate.items(),
                             key=lambda kv: kv[1], reverse=True):
        pct = 100.0 * sec / total
        per_clip_ms = 1000.0 * sec / max(total_clips, 1)
        lines.append(f"[  {stage:<20} ]  {sec:8.2f}s   ({pct:5.1f}%)   "
                     f"{per_clip_ms:8.1f} ms/clip")
    lines.append("=" * 65)
    lines.append(f"TOTAL                    {total:8.2f}s")
    txt = "\n".join(lines)
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(txt)
    return txt
