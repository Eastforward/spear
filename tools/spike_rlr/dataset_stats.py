"""Aggregate per-clip metadata + optional matplotlib charts."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "spike_rlr"))

from flag_definitions import ALL_FLAGS


def aggregate(out_dir: Path) -> dict:
    clips_dir = out_dir / "clips"
    clip_dirs = sorted(d for d in clips_dir.iterdir() if d.is_dir())
    coverage = {f: 0 for f in ALL_FLAGS}
    for cd in clip_dirs:
        f = cd / "flags.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        for name, v in d.items():
            if v:
                coverage[name] = coverage.get(name, 0) + 1
    # Stage timing from any profile_per_clip.csv files
    csv_paths = list(clips_dir.glob("*/profile_per_clip.csv"))
    stage_seconds = {}
    for p in csv_paths:
        with p.open() as f:
            for row in csv.DictReader(f):
                stage_seconds[row["stage"]] = \
                    stage_seconds.get(row["stage"], 0.0) + float(row["seconds"])
    return {
        "n_clips": len(clip_dirs),
        "flag_coverage": coverage,
        "stage_seconds": stage_seconds,
    }


def generate_charts(stats: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Bar chart of flag coverage
    fig, ax = plt.subplots(figsize=(10, 5))
    names = list(stats["flag_coverage"].keys())
    counts = [stats["flag_coverage"][n] for n in names]
    ax.bar(range(len(names)), counts)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("clip count")
    ax.set_title(f"Flag coverage across {stats['n_clips']} clips")
    fig.tight_layout()
    fig.savefig(out_dir / "analysis" / "coverage_bar.png", dpi=100)
    plt.close(fig)
    # Pie chart of stage timing
    if stats["stage_seconds"]:
        fig, ax = plt.subplots(figsize=(6, 6))
        labels = list(stats["stage_seconds"].keys())
        sizes = list(stats["stage_seconds"].values())
        ax.pie(sizes, labels=labels, autopct="%1.1f%%")
        ax.set_title(f"Total pipeline time = {sum(sizes):.1f}s")
        fig.savefig(out_dir / "analysis" / "stage_pie.png", dpi=100)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--generate-charts", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    stats = aggregate(out)
    (out / "analysis" / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    if args.generate_charts:
        generate_charts(stats, out)
        print(f"charts -> {out}/analysis/coverage_bar.png (+ stage_pie.png)")


if __name__ == "__main__":
    main()
