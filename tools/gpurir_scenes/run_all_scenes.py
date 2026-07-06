"""Sequentially run scenes for seeds 0..9 (or arbitrary list)."""
import argparse
import json
import os
import subprocess
import sys
import time


SPEAR_PY = "/data/jzy/miniconda3/envs/spear-env/bin/python"
TOOLS = os.path.dirname(os.path.abspath(__file__))


def main():
    _default_out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "tmp/gpurir_scenes_v1",
    )
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--out-root", default=_default_out)
    args = p.parse_args()

    summary = []
    for seed in args.seeds:
        sd = os.path.join(args.out_root, f"scene_{seed:02d}")
        marker = os.path.join(sd, "shoebox", "view3_with_audio.mp4")
        if os.path.exists(marker):
            summary.append({"seed": seed, "status": "cached", "path": sd})
            print(f"[skip cached] seed {seed}")
            continue
        t0 = time.time()
        rc = subprocess.run([
            SPEAR_PY, os.path.join(TOOLS, "run_scene.py"),
            "--seed", str(seed), "--out-root", args.out_root,
        ]).returncode
        summary.append({
            "seed": seed, "status": "ok" if rc == 0 else "fail",
            "seconds": round(time.time() - t0, 1), "path": sd,
        })
        with open(os.path.join(args.out_root, "batch_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        if rc != 0:
            print(f"[warn] seed {seed} failed rc={rc}")

    print("\nBATCH_DONE")
    for s in summary:
        print(f"  seed {s['seed']:2d} status={s['status']:6s} secs={s.get('seconds', 0):>6}  {s['path']}")


if __name__ == "__main__":
    main()
