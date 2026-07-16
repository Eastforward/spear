"""Run technical-only human apartment examples through UE, RLR, and review."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


os.environ.setdefault("DISPLAY", ":99")
os.environ.setdefault("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/avengine-matplotlib")

from human_apartment_evidence import finalize_human_apartment_clip  # noqa: E402
from run_render_pass_apartment import render_apartment  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = (
    REPO_ROOT
    / "tmp"
    / "hy3d_rocketbox_template_fit_v1"
    / "human_apartment_examples_v1"
    / "scenario_bundle.json"
)
DEFAULT_MESH = REPO_ROOT / "tmp" / "spike_rlr" / "apartment_v1_mesh.glb"
DEFAULT_MATERIALS = (
    REPO_ROOT / "tmp" / "spike_rlr" / "apartment_v1_materials.json"
)
DEFAULT_SS2_PYTHON = Path("/data/jzy/miniconda3/envs/ss2/bin/python")


def _append_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_file(path: Path, label: str) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _audio_command(
    *,
    spec_path: Path,
    out_path: Path,
    mesh_path: Path,
    materials_path: Path,
    ss2_python: Path,
    quality: str,
) -> list[str]:
    return [
        str(ss2_python),
        str(REPO_ROOT / "tools/spike_rlr/run_audio_pass_rlr.py"),
        "--spec",
        str(spec_path),
        "--mesh",
        str(mesh_path),
        "--materials",
        str(materials_path),
        "--out",
        str(out_path),
        "--channel-layout",
        "binaural",
        "--quality",
        str(quality),
    ]


def run_human_apartment_example(
    *,
    spec_path: Path | str,
    out_dir: Path | str,
    clip_id: str,
    mesh_path: Path | str = DEFAULT_MESH,
    materials_path: Path | str = DEFAULT_MATERIALS,
    ss2_python: Path | str = DEFAULT_SS2_PYTHON,
    quality: str = "low",
    skip_ue_render: bool = False,
    skip_audio_render: bool = False,
    skip_finalize: bool = False,
) -> dict:
    spec_path = _require_file(Path(spec_path), "scenario spec")
    mesh_path = _require_file(Path(mesh_path), "RLR apartment mesh")
    materials_path = _require_file(Path(materials_path), "RLR materials sidecar")
    ss2_python = Path(ss2_python)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    command_log = out_dir / "command.log"
    _append_log(command_log, {
        "event": "example_start",
        "timestamp": _timestamp(),
        "clip_id": str(clip_id),
        "spec": str(spec_path),
        "skip_ue_render": bool(skip_ue_render),
        "skip_audio_render": bool(skip_audio_render),
        "skip_finalize": bool(skip_finalize),
    })

    if not skip_ue_render:
        _append_log(command_log, {"event": "ue_start", "timestamp": _timestamp()})
        render_apartment(
            spec_path,
            out_dir,
            out_dir / "profile_per_clip.csv",
            str(clip_id),
        )
        _append_log(command_log, {"event": "ue_passed", "timestamp": _timestamp()})
    else:
        _require_file(
            out_dir / "videos" / "apartment_v1_view0.mp4",
            "existing UE video",
        )

    audio_path = out_dir / "binaural.wav"
    if not skip_audio_render:
        command = _audio_command(
            spec_path=spec_path,
            out_path=audio_path,
            mesh_path=mesh_path,
            materials_path=materials_path,
            ss2_python=ss2_python,
            quality=quality,
        )
        _append_log(command_log, {
            "event": "rlr_start",
            "timestamp": _timestamp(),
            "argv": command,
        })
        rlr_env = dict(os.environ)
        rlr_env["LD_PRELOAD"] = (
            "/usr/lib/x86_64-linux-gnu/libEGL.so.1:"
            "/usr/lib/x86_64-linux-gnu/libGLdispatch.so.0"
        )
        subprocess.run(
            command,
            check=True,
            cwd=str(REPO_ROOT),
            env=rlr_env,
        )
        _append_log(command_log, {"event": "rlr_passed", "timestamp": _timestamp()})
    _require_file(audio_path, "binaural audio")

    review_outputs = {}
    if not skip_finalize:
        review_outputs = finalize_human_apartment_clip(
            spec_path=spec_path,
            out_dir=out_dir,
            clip_id=str(clip_id),
            publish_registry=False,
        )
        _append_log(command_log, {
            "event": "finalize_passed",
            "timestamp": _timestamp(),
        })

    result = {
        "clip_id": str(clip_id),
        "spec": spec_path,
        "out_dir": out_dir,
        "video": out_dir / "videos" / "apartment_v1_view0.mp4",
        "audio": audio_path,
        **review_outputs,
    }
    _append_log(command_log, {
        "event": "example_passed",
        "timestamp": _timestamp(),
        "clip_id": str(clip_id),
    })
    return result


def run_human_apartment_bundle(
    *,
    bundle_path: Path | str = DEFAULT_BUNDLE,
    scenario_ids: list[str] | None = None,
    mesh_path: Path | str = DEFAULT_MESH,
    materials_path: Path | str = DEFAULT_MATERIALS,
    ss2_python: Path | str = DEFAULT_SS2_PYTHON,
    quality: str = "low",
    skip_ue_render: bool = False,
    skip_audio_render: bool = False,
    skip_finalize: bool = False,
) -> list[dict]:
    bundle_path = _require_file(Path(bundle_path), "scenario bundle")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if bundle.get("schema_version") != "human_apartment_scenario_bundle_v1":
        raise ValueError("unsupported human apartment scenario bundle schema")
    scenarios = bundle.get("scenarios", {})
    selected = list(scenario_ids) if scenario_ids else list(scenarios)
    unknown = [scenario_id for scenario_id in selected if scenario_id not in scenarios]
    if unknown:
        raise KeyError(f"unknown human apartment scenarios: {unknown}")

    results = []
    for scenario_id in selected:
        descriptor = scenarios[scenario_id]
        results.append(run_human_apartment_example(
            spec_path=Path(descriptor["spec_path"]),
            out_dir=Path(descriptor["output_dir"]),
            clip_id=str(descriptor.get("clip_id") or scenario_id),
            mesh_path=mesh_path,
            materials_path=materials_path,
            ss2_python=ss2_python,
            quality=quality,
            skip_ue_render=skip_ue_render,
            skip_audio_render=skip_audio_render,
            skip_finalize=skip_finalize,
        ))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenario_ids",
        help="Scenario id to run; repeat to control order. Default: all.",
    )
    parser.add_argument("--mesh", default=str(DEFAULT_MESH))
    parser.add_argument("--materials", default=str(DEFAULT_MATERIALS))
    parser.add_argument("--ss2-python", default=str(DEFAULT_SS2_PYTHON))
    parser.add_argument("--quality", choices=("low", "high", "max"), default="low")
    parser.add_argument("--skip-ue-render", action="store_true")
    parser.add_argument("--skip-audio-render", action="store_true")
    parser.add_argument("--skip-finalize", action="store_true")
    parser.add_argument("--rig-assert", action="store_true")
    args = parser.parse_args()
    if args.rig_assert:
        os.environ["SPEAR_RIG_ASSERT"] = "1"
    results = run_human_apartment_bundle(
        bundle_path=args.bundle,
        scenario_ids=args.scenario_ids,
        mesh_path=args.mesh,
        materials_path=args.materials,
        ss2_python=args.ss2_python,
        quality=args.quality,
        skip_ue_render=args.skip_ue_render,
        skip_audio_render=args.skip_audio_render,
        skip_finalize=args.skip_finalize,
    )
    for result in results:
        print(f"{result['clip_id']}: {result['out_dir']}")


if __name__ == "__main__":
    main()
