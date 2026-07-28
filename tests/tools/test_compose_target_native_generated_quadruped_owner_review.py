from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
from types import SimpleNamespace

import pytest

from tools import (
    compose_target_native_generated_quadruped_owner_review as subject,
)
from tools.run_target_native_generated_quadruped_review import (
    ENCODE_MANIFEST_SCHEMA,
    RENDER_MANIFEST_SCHEMA,
    REVIEW_RENDER_CONFIG,
    expected_review_ffmpeg_config,
    file_record,
    render_frame_set,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def unseal_output(path: Path) -> None:
    if not path.exists():
        return
    path.chmod(0o755)
    for child in path.iterdir():
        child.chmod(0o644)


def video_record(path: Path, *, width=512, height=384) -> dict:
    return {
        **file_record(path),
        "codec": "h264",
        "width": width,
        "height": height,
        "frame_count": 8,
        "frame_rate": "8/1",
        "duration_seconds": 1.0,
    }


def fake_probe(
    path: Path,
    *,
    ffprobe: Path,
    ffmpeg: Path,
    expected_width: int,
    expected_height: int,
    expected_frames: int,
    expected_fps: int,
) -> dict:
    assert expected_frames == 8
    assert expected_fps == 8
    path = Path(path).resolve()
    video = video_record(path, width=expected_width, height=expected_height)
    return {
        "video": video,
        "readback": {
            "stream_count": 1,
            "stream_index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "pix_fmt": "yuv420p",
            "sample_aspect_ratio": "1:1",
            "width": expected_width,
            "height": expected_height,
            "r_frame_rate": "8/1",
            "avg_frame_rate": "8/1",
            "nb_frames": 8,
            "nb_read_frames": 8,
            "duration_seconds": 1.0,
        },
        "ffprobe_argv": subject.build_ffprobe_argv(ffprobe, path),
        "full_decode": {
            "argv": subject.build_decode_argv(ffmpeg, path),
            "passed": True,
        },
        "file_guard": subject.file_guard(path, "fake video"),
    }


def render_payload(
    *,
    animated_glb: Path,
    frame_dir: Path,
    action: str,
    view: str,
    yaw: float,
) -> dict:
    frame_dir.mkdir()
    frame_range = [0.0, 40.0]
    frames = []
    for index in range(8):
        fraction = index / 7
        frame = float(
            int(round(frame_range[0] + (frame_range[1] - frame_range[0]) * fraction))
        )
        artifact = frame_dir / f"frame_{index:04d}.png"
        artifact.write_bytes(f"png-{index}".encode("utf-8"))
        frames.append(
            {
                "index": index,
                "sample_fraction": fraction,
                "source_action_frame": frame,
                "artifact": file_record(artifact),
            }
        )
    return {
        "schema": RENDER_MANIFEST_SCHEMA,
        "status": "frames_rendered",
        "formal_dataset_registration_authorized": False,
        "input_glb": file_record(animated_glb),
        "request": {
            "action": action,
            "resolved_action": f"{action}_Armature",
            "rest_pose": False,
            "view": view,
            "asset_yaw_deg": float(yaw),
            "n_frames": 8,
            "resolution": {"width": 512, "height": 384},
            "fps": 8,
            "output_dir": str(frame_dir.resolve()),
        },
        "render_config": deepcopy(REVIEW_RENDER_CONFIG),
        "action_frame_range": frame_range,
        "frames": frames,
    }


def review_gates(branch: str) -> dict:
    return {
        "heading": "passed",
        "rig": "passed",
        "support_plane": "passed",
        "retarget_export_front_axis": "positive-x",
        "gait_initial": "passed",
        "deformation_initial": "passed",
        "weight_repair_policy": "auto",
        "weight_repair_triggered": branch != "not_needed",
        "weight_repair": "passed",
        "weight_repair_strategy": branch,
        "weight_repair_branch": branch,
        "weight_repair_attempts": [],
        "weight_repair_final_artifact": {"branch": branch},
        "gait_final": "passed",
        "deformation_final": "passed",
        "all_automatic_gates_passed": True,
    }


def build_review_fixture(tmp_path: Path, *, branch: str = "primary") -> dict:
    review_root = tmp_path / f"review_{branch}"
    media_root = review_root / "05_review"
    media_root.mkdir(parents=True)
    animated_glb = review_root / "04_motion" / "target_animated.glb"
    animated_glb.parent.mkdir()
    animated_glb.write_bytes(b"animated-generated-target")

    media = {}
    media_lineage = {}
    for (
        label,
        action,
        view,
        yaw,
        _title,
        _row,
        _column,
    ) in subject.MEDIA_LAYOUT:
        frame_dir = media_root / f"{label}_frames"
        render = render_payload(
            animated_glb=animated_glb,
            frame_dir=frame_dir,
            action=action,
            view=view,
            yaw=yaw,
        )
        render_manifest = media_root / f"{label}_render_manifest.json"
        write_json(render_manifest, render)
        video = media_root / f"{label}.mp4"
        video.write_bytes(f"h264-{label}".encode("utf-8"))
        encoded_video = video_record(video)
        encode = {
            "schema": ENCODE_MANIFEST_SCHEMA,
            "status": "video_encoded_and_probed",
            "formal_dataset_registration_authorized": False,
            "media_identity": {
                "label": label,
                "action": action,
                "view": view,
                "asset_yaw_deg": float(yaw),
            },
            "render_manifest": file_record(render_manifest),
            "frame_set": render_frame_set(render),
            "ffmpeg": expected_review_ffmpeg_config(frame_dir, 8),
            "video": encoded_video,
        }
        encode_manifest = media_root / f"{label}_encode_manifest.json"
        write_json(encode_manifest, encode)
        media[label] = encoded_video
        media_lineage[label] = {
            "render_manifest": file_record(render_manifest),
            "encode_manifest": file_record(encode_manifest),
        }

    outputs = {
        "animated_glb": file_record(animated_glb),
        "media": media,
        "media_lineage": media_lineage,
    }
    if branch == "primary":
        outputs.update(
            {
                "weight_repair_primary_glb": {"ignored": True},
                "weight_repair_primary_manifest": {"ignored": True},
                "weight_repair_manifest": {"ignored": True},
            }
        )
    elif branch == "fallback_a_b":
        outputs.update(
            {
                "weight_repair_primary_glb": {"ignored": True},
                "weight_repair_primary_manifest": {"ignored": True},
                "weight_repair_fallback_a_glb": {"ignored": True},
                "weight_repair_fallback_a_manifest": {"ignored": True},
                "weight_repair_fallback_b_glb": {"ignored": True},
                "weight_repair_fallback_b_manifest": {"ignored": True},
                "weight_repair_manifest": {"ignored": True},
            }
        )
    review = {
        "schema": subject.REVIEW_SCHEMA,
        "created_at": "2026-07-28T00:00:00+00:00",
        "status": "research_candidate_pending_human_review",
        "formal_dataset_registration_authorized": False,
        "forward_contract": {"target_species": "cat"},
        "inputs": {},
        "pipeline_order": [],
        "automatic_admission_gates": review_gates(branch),
        "outputs": outputs,
        "timings_seconds": {},
    }
    review_path = review_root / "review_run.json"
    write_json(review_path, review)
    return {
        "root": review_root,
        "media_root": media_root,
        "review": review_path,
        "sha256": sha256_file(review_path),
        "animated_glb": animated_glb,
    }


def rebind_fixture_video(fixture: dict, label: str) -> None:
    video = fixture["media_root"] / f"{label}.mp4"
    encoded_video = video_record(video)
    encode_path = fixture["media_root"] / f"{label}_encode_manifest.json"
    encode = json.loads(encode_path.read_text(encoding="utf-8"))
    encode["video"] = encoded_video
    write_json(encode_path, encode)
    review = json.loads(fixture["review"].read_text(encoding="utf-8"))
    review["outputs"]["media"][label] = encoded_video
    review["outputs"]["media_lineage"][label]["encode_manifest"] = file_record(
        encode_path
    )
    write_json(fixture["review"], review)
    fixture["sha256"] = sha256_file(fixture["review"])


def materialize_real_video_fixture(fixture: dict, tmp_path: Path) -> None:
    colors = (
        "0x17202a",
        "0xd35400",
        "0xf4d03f",
        "0x1e8449",
        "0x2471a3",
        "0x7d3c98",
    )
    for label, color in zip(subject.MEDIA_LABELS, colors):
        video = fixture["media_root"] / f"{label}.mp4"
        subprocess.run(
            [
                shutil.which("ffmpeg") or "ffmpeg",
                "-nostdin",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"color=c={color}:s=512x384:r=8:d=1",
                "-frames:v",
                "8",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video),
            ],
            check=True,
        )
        rebind_fixture_video(fixture, label)


def install_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Path]:
    executables = {}
    for name in ("ffmpeg", "ffprobe"):
        path = tmp_path / f"fake_{name}"
        path.write_bytes(name.encode("utf-8"))
        path.chmod(0o755)
        executables[name] = path.resolve()
    font = tmp_path / "font.ttf"
    font.write_bytes(b"font")
    monkeypatch.setattr(
        subject,
        "resolve_named_executable",
        lambda name: executables[name],
    )
    monkeypatch.setattr(subject, "FONT_PATH", font)
    monkeypatch.setattr(subject, "probe_video", fake_probe)
    tool_path = Path(subject.__file__).resolve()
    runtime_identity = {
        "presentation_tool": {
            "version": subject.PRESENTATION_TOOL_VERSION,
            "file": file_record(tool_path),
            "file_guard": subject.file_guard(tool_path, "test tool"),
        },
        "ffmpeg": {
            "executable": file_record(executables["ffmpeg"]),
            "file_guard": subject.file_guard(executables["ffmpeg"], "test ffmpeg"),
            "version_argv": [str(executables["ffmpeg"]), "-version"],
            "version_first_line": "ffmpeg version test",
            "version_output_sha256": "5" * 64,
        },
        "ffprobe": {
            "executable": file_record(executables["ffprobe"]),
            "file_guard": subject.file_guard(executables["ffprobe"], "test ffprobe"),
            "version_argv": [str(executables["ffprobe"]), "-version"],
            "version_first_line": "ffprobe version test",
            "version_output_sha256": "6" * 64,
        },
        "font": {
            "file": file_record(font.resolve()),
            "file_guard": subject.file_guard(font.resolve(), "test font"),
            "sfnt_version_hex": "00010000",
            "family": "Test Sans",
            "subfamily": "Bold",
            "version": "Version 1.0",
            "postscript_name": "TestSans-Bold",
        },
        "python": {"implementation": "CPython", "version": "3.9.test"},
    }
    monkeypatch.setattr(
        subject,
        "capture_runtime_identity",
        lambda **_kwargs: deepcopy(runtime_identity),
    )
    content_readback = {
        "schema": subject.CONTENT_READBACK_SCHEMA,
        "method": subject.content_readback_method(),
        "reference_argv": ["fake-reference"],
        "observed_argv": ["fake-observed"],
        "reference_gray_frames_sha256": "1" * 64,
        "observed_gray_frames_sha256": "2" * 64,
        "cells": [
            {
                "label": label,
                "title": title,
                "row": row,
                "column": column,
                "frames": [
                    {
                        "frame_index": index,
                        "reference_gray_sha256": "3" * 64,
                        "observed_gray_sha256": "4" * 64,
                        "mean_absolute_error": 0.0,
                        "root_mean_square_error": 0.0,
                        "max_absolute_error": 0,
                        "passed": True,
                    }
                    for index in range(8)
                ],
                "all_frames_passed": True,
            }
            for (
                label,
                _action,
                _view,
                _yaw,
                title,
                row,
                column,
            ) in subject.MEDIA_LAYOUT
        ],
        "all_cells_all_frames_passed": True,
        "content_readback_sha256": None,
    }

    def fake_content_audit(**kwargs):
        result = deepcopy(content_readback)
        result["reference_argv"] = subject.build_reference_rawvideo_argv(
            ffmpeg=kwargs["ffmpeg"],
            font=kwargs["font"],
            videos=kwargs["snapshots"],
        )
        result["observed_argv"] = subject.build_observed_rawvideo_argv(
            ffmpeg=kwargs["ffmpeg"],
            video=kwargs["output_video"],
        )
        result["content_readback_sha256"] = subject.hash_without(
            result, "content_readback_sha256"
        )
        return result

    monkeypatch.setattr(
        subject,
        "audit_frame_cell_content",
        fake_content_audit,
    )
    return {**executables, "font": font.resolve()}


@pytest.mark.parametrize("branch", ["primary", "fallback_a_b"])
def test_authenticator_accepts_shiba_and_british_review_output_shapes(
    tmp_path,
    monkeypatch,
    branch,
):
    fixture = build_review_fixture(tmp_path, branch=branch)
    runtime = install_fake_runtime(monkeypatch, tmp_path)

    authenticated = subject.authenticate_review(
        fixture["review"],
        fixture["sha256"],
        ffprobe=runtime["ffprobe"],
        ffmpeg=runtime["ffmpeg"],
    )

    assert authenticated["review_run"]["sha256"] == fixture["sha256"]
    assert authenticated["animated_glb"] == file_record(fixture["animated_glb"])
    assert tuple(authenticated["media"]) == subject.MEDIA_LABELS
    assert {
        authenticated["media"][label]["frame_set"]["count"]
        for label in subject.MEDIA_LABELS
    } == {8}


def test_external_review_hash_is_checked_before_strict_json_parse(tmp_path):
    review = tmp_path / "review_run.json"
    review.write_text(
        '{"schema":"first","schema":"second"}',
        encoding="utf-8",
    )

    with pytest.raises(
        subject.PresentationContractError,
        match="external SHA-256",
    ) as caught:
        subject.read_exact_review(review, "0" * 64)
    assert caught.value.__cause__ is None

    with pytest.raises(
        subject.PresentationContractError,
        match="not strict JSON",
    ) as caught:
        subject.read_exact_review(review, sha256_file(review))
    assert "duplicate JSON object key" in str(caught.value.__cause__)


def test_authenticator_rejects_missing_or_extra_media_labels(
    tmp_path,
    monkeypatch,
):
    fixture = build_review_fixture(tmp_path)
    runtime = install_fake_runtime(monkeypatch, tmp_path)
    review = json.loads(fixture["review"].read_text(encoding="utf-8"))
    review["outputs"]["media"].pop("idle_rear")
    review["outputs"]["media"]["copied_side"] = review["outputs"]["media"][
        "walking_side"
    ]
    write_json(fixture["review"], review)

    with pytest.raises(
        subject.PresentationContractError,
        match="exactly six",
    ):
        subject.authenticate_review(
            fixture["review"],
            sha256_file(fixture["review"]),
            ffprobe=runtime["ffprobe"],
            ffmpeg=runtime["ffmpeg"],
        )


def test_authenticator_rejects_video_mutation(
    tmp_path,
    monkeypatch,
):
    fixture = build_review_fixture(tmp_path)
    runtime = install_fake_runtime(monkeypatch, tmp_path)
    video = fixture["media_root"] / "walking_side.mp4"
    video.write_bytes(b"mutated-copied-black-video")

    with pytest.raises(
        subject.PresentationContractError,
        match="contradicts FFprobe",
    ):
        subject.authenticate_review(
            fixture["review"],
            fixture["sha256"],
            ffprobe=runtime["ffprobe"],
            ffmpeg=runtime["ffmpeg"],
        )


def test_authenticator_rejects_semantically_rebound_render_receipt(
    tmp_path,
    monkeypatch,
):
    fixture = build_review_fixture(tmp_path)
    runtime = install_fake_runtime(monkeypatch, tmp_path)
    render_path = fixture["media_root"] / "walking_side_render_manifest.json"
    render = json.loads(render_path.read_text(encoding="utf-8"))
    render["request"]["action"] = "Idle"
    render["request"]["resolved_action"] = "Idle_Armature"
    write_json(render_path, render)
    review = json.loads(fixture["review"].read_text(encoding="utf-8"))
    review["outputs"]["media_lineage"]["walking_side"]["render_manifest"] = file_record(
        render_path
    )
    write_json(fixture["review"], review)

    with pytest.raises(RuntimeError, match="request identity"):
        subject.authenticate_review(
            fixture["review"],
            sha256_file(fixture["review"]),
            ffprobe=runtime["ffprobe"],
            ffmpeg=runtime["ffmpeg"],
        )


def test_authenticator_rejects_swapped_encode_identity(
    tmp_path,
    monkeypatch,
):
    fixture = build_review_fixture(tmp_path)
    runtime = install_fake_runtime(monkeypatch, tmp_path)
    encode_path = fixture["media_root"] / "walking_side_encode_manifest.json"
    encode = json.loads(encode_path.read_text(encoding="utf-8"))
    encode["media_identity"]["label"] = "idle_side"
    encode["media_identity"]["action"] = "Idle"
    write_json(encode_path, encode)
    review = json.loads(fixture["review"].read_text(encoding="utf-8"))
    review["outputs"]["media_lineage"]["walking_side"]["encode_manifest"] = file_record(
        encode_path
    )
    write_json(fixture["review"], review)

    with pytest.raises(RuntimeError, match="identity/status"):
        subject.authenticate_review(
            fixture["review"],
            sha256_file(fixture["review"]),
            ffprobe=runtime["ffprobe"],
            ffmpeg=runtime["ffmpeg"],
        )


@pytest.mark.parametrize(
    "payload",
    [
        '{"programs":[],"streams":[],"streams":[],"format":{"duration":"1"}}',
        (
            '{"programs":[],"streams":[{"codec_type":"video",'
            '"codec_name":"h264","width":512,"height":384,'
            '"r_frame_rate":"8/1","avg_frame_rate":"8/1",'
            '"nb_frames":"8"}],"format":{"duration":NaN}}'
        ),
    ],
)
def test_ffprobe_readback_rejects_ambiguous_json(
    tmp_path,
    monkeypatch,
    payload,
):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    ffprobe = tmp_path / "ffprobe"
    ffprobe.write_bytes(b"probe")
    monkeypatch.setattr(
        subject.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=payload),
    )

    with pytest.raises(
        subject.PresentationContractError,
        match="ambiguous JSON",
    ):
        subject.probe_video(
            video,
            ffprobe=ffprobe,
            ffmpeg=ffprobe,
            expected_width=512,
            expected_height=384,
            expected_frames=8,
            expected_fps=8,
        )


def test_ffmpeg_command_has_fixed_order_labels_and_xstack(tmp_path):
    ffmpeg = tmp_path / "ffmpeg"
    font = tmp_path / "font.ttf"
    videos = [tmp_path / f"{label}.mp4" for label in subject.MEDIA_LABELS]
    output = tmp_path / "owner_review_six_view.mp4"

    argv = subject.build_ffmpeg_argv(
        ffmpeg=ffmpeg,
        font=font,
        videos=videos,
        output=output,
    )
    inputs = [argv[index + 1] for index, item in enumerate(argv[:-1]) if item == "-i"]
    filter_graph = argv[argv.index("-filter_complex") + 1]

    assert inputs == [str(path) for path in videos]
    assert "-n" in argv
    assert argv[-1] == str(output)
    assert subject.XSTACK_LAYOUT in filter_graph
    for item in subject.MEDIA_LAYOUT:
        assert item[4] in filter_graph
    assert "[v0][v1][v2][v3][v4][v5]xstack=inputs=6" in filter_graph


def test_main_writes_fail_closed_presentation_receipt_without_decision(
    tmp_path,
    monkeypatch,
):
    fixture = build_review_fixture(tmp_path)
    runtime = install_fake_runtime(monkeypatch, tmp_path)
    review_before = fixture["review"].read_bytes()
    probe_calls = []

    def recording_probe(path, **kwargs):
        probe_calls.append(Path(path).name)
        return fake_probe(path, **kwargs)

    def fake_run_ffmpeg(argv):
        assert str(runtime["ffmpeg"]) == argv[0]
        assert "-n" in argv
        inputs = [
            Path(argv[index + 1])
            for index, value in enumerate(argv[:-1])
            if value == "-i"
        ]
        assert len(inputs) == 6
        assert all(path.parent.name == ".private_input_snapshots" for path in inputs)
        assert all(fixture["media_root"] not in path.parents for path in inputs)
        Path(argv[-1]).write_bytes(b"composed-h264-six-view")

    monkeypatch.setattr(subject, "probe_video", recording_probe)
    monkeypatch.setattr(subject, "run_ffmpeg", fake_run_ffmpeg)
    output_root = tmp_path / "presentation"

    assert (
        subject.main(
            [
                "--review-run",
                str(fixture["review"]),
                "--expected-review-run-sha256",
                fixture["sha256"],
                "--output-root",
                str(output_root),
            ]
        )
        == 0
    )

    video = output_root / subject.OUTPUT_VIDEO_NAME
    receipt_path = output_root / subject.RECEIPT_NAME
    receipt = subject.strict_json_loads(receipt_path.read_bytes())
    assert video.is_file()
    assert set(receipt) == subject.RECEIPT_TOP_LEVEL_FIELDS
    assert (
        subject.validate_presentation_receipt(
            receipt,
            expected_source_review_sha256=fixture["sha256"],
        )
        == receipt
    )
    loaded, receipt_record = subject.load_presentation_receipt(
        receipt_path,
        sha256_file(receipt_path),
        expected_source_review_sha256=fixture["sha256"],
    )
    assert loaded == receipt
    assert receipt_record == file_record(receipt_path)
    assert receipt["schema"] == subject.PRESENTATION_SCHEMA
    assert receipt["status"] == subject.PRESENTATION_STATUS
    assert receipt["expected_source_review_sha256"] == fixture["sha256"]
    assert receipt["source_review"]["sha256"] == fixture["sha256"]
    assert receipt["authority"] == {
        "purpose": "owner_animation_review_presentation_only",
        "decision_authority": "none",
        "user_decision_recorded": False,
        "source_review_modified": False,
        "formal_dataset_registration_authorized": False,
    }
    assert receipt["output"]["sha256"] == sha256_file(video)
    assert receipt["output"]["width"] == 1536
    assert receipt["output"]["height"] == 768
    assert set(receipt["authenticated_inputs"]) == set(subject.MEDIA_LABELS)
    assert len(probe_calls) == 31
    assert receipt["source_order_sha256"] == subject.canonical_json_sha256(
        receipt["source_order"]
    )
    assert receipt["source_set_sha256"] == subject.canonical_json_sha256(
        receipt["source_set"]
    )
    assert receipt["receipt_sha256"] == subject.hash_without(receipt, "receipt_sha256")
    assert all(receipt["automatic_checks"].values())
    assert fixture["review"].read_bytes() == review_before
    assert output_root.stat().st_mode & 0o777 == 0o555
    assert video.stat().st_mode & 0o777 == 0o444
    assert receipt_path.stat().st_mode & 0o777 == 0o444
    assert {path.name for path in output_root.iterdir()} == {
        subject.OUTPUT_VIDEO_NAME,
        subject.RECEIPT_NAME,
    }
    extra = deepcopy(receipt)
    extra["owner_decision"] = "approved"
    extra["receipt_sha256"] = subject.hash_without(extra, "receipt_sha256")
    with pytest.raises(
        subject.PresentationContractError,
        match="receipt fields changed",
    ):
        subject.validate_presentation_receipt(extra)

    failed_content = deepcopy(receipt)
    failed_content["frame_cell_content_readback"]["cells"][0]["frames"][0]["passed"] = (
        False
    )
    failed_content["frame_cell_content_readback"]["content_readback_sha256"] = (
        subject.hash_without(
            failed_content["frame_cell_content_readback"],
            "content_readback_sha256",
        )
    )
    failed_content["receipt_sha256"] = subject.hash_without(
        failed_content, "receipt_sha256"
    )
    with pytest.raises(
        subject.PresentationContractError,
        match="content evidence changed",
    ):
        subject.validate_presentation_receipt(failed_content)

    relative_output = deepcopy(receipt)
    relative_output["output"]["path"] = subject.OUTPUT_VIDEO_NAME
    relative_output["receipt_sha256"] = subject.hash_without(
        relative_output,
        "receipt_sha256",
    )
    with pytest.raises(
        subject.PresentationContractError,
        match="presentation output is invalid",
    ):
        subject.validate_presentation_receipt(relative_output)

    with pytest.raises(
        subject.PresentationContractError,
        match="external SHA-256",
    ):
        subject.load_presentation_receipt(receipt_path, "0" * 64)

    video_before = video.read_bytes()
    with pytest.raises(
        subject.PresentationContractError,
        match="refusing to replace output root",
    ):
        subject.main(
            [
                "--review-run",
                str(fixture["review"]),
                "--expected-review-run-sha256",
                fixture["sha256"],
                "--output-root",
                str(output_root),
            ]
        )
    assert video.read_bytes() == video_before
    unseal_output(output_root)


def test_main_reauthenticates_after_ffmpeg_and_removes_partial_output(
    tmp_path,
    monkeypatch,
):
    fixture = build_review_fixture(tmp_path)
    install_fake_runtime(monkeypatch, tmp_path)
    review_before = fixture["review"].read_bytes()

    def mutate_after_read(argv):
        Path(argv[-1]).write_bytes(b"composed-h264-six-view")
        (fixture["media_root"] / "walking_side.mp4").write_bytes(
            b"mutated-during-composition"
        )

    monkeypatch.setattr(subject, "run_ffmpeg", mutate_after_read)
    output_root = tmp_path / "failed_presentation"

    with pytest.raises(
        subject.PresentationContractError,
        match="contradicts FFprobe",
    ):
        subject.main(
            [
                "--review-run",
                str(fixture["review"]),
                "--expected-review-run-sha256",
                fixture["sha256"],
                "--output-root",
                str(output_root),
            ]
        )
    assert not output_root.exists()
    assert fixture["review"].read_bytes() == review_before


def test_output_root_cannot_modify_the_immutable_review_tree(
    tmp_path,
    monkeypatch,
):
    fixture = build_review_fixture(tmp_path)
    install_fake_runtime(monkeypatch, tmp_path)
    forbidden_parent = fixture["root"] / "new_presentation_parent"
    output_root = forbidden_parent / "presentation"

    with pytest.raises(
        subject.PresentationContractError,
        match="outside the immutable review run",
    ):
        subject.main(
            [
                "--review-run",
                str(fixture["review"]),
                "--expected-review-run-sha256",
                fixture["sha256"],
                "--output-root",
                str(output_root),
            ]
        )
    assert not output_root.exists()
    assert not forbidden_parent.exists()


def test_all_path_components_reject_non_bridge_symlinks(tmp_path):
    fixture = build_review_fixture(tmp_path)
    alias = tmp_path / "review_alias"
    alias.symlink_to(fixture["root"], target_is_directory=True)

    with pytest.raises(
        subject.PresentationContractError,
        match="unsafe symlink component",
    ):
        subject.resolve_regular_file(alias / "review_run.json", "aliased review")

    video = fixture["media_root"] / "walking_side.mp4"
    direct = fixture["media_root"] / "walking_side.direct.mp4"
    video.rename(direct)
    video.symlink_to(direct.name)
    with pytest.raises(
        subject.PresentationContractError,
        match="unsafe symlink component",
    ):
        subject.resolve_regular_file(video, "symlinked video")


def test_main_rejects_restore_race_even_when_bytes_and_mtime_are_restored(
    tmp_path,
    monkeypatch,
):
    fixture = build_review_fixture(tmp_path)
    install_fake_runtime(monkeypatch, tmp_path)
    source = fixture["media_root"] / "walking_side.mp4"
    original = source.read_bytes()
    original_stat = source.stat()

    def restore_race(argv):
        Path(argv[-1]).write_bytes(b"composed-h264-six-view")
        source.write_bytes(b"temporary-race-mutation")
        source.write_bytes(original)
        os.utime(
            source,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )

    monkeypatch.setattr(subject, "run_ffmpeg", restore_race)
    output_root = tmp_path / "restore_race_presentation"

    with pytest.raises(
        subject.PresentationContractError,
        match="authority graph changed during composition",
    ):
        subject.main(
            [
                "--review-run",
                str(fixture["review"]),
                "--expected-review-run-sha256",
                fixture["sha256"],
                "--output-root",
                str(output_root),
            ]
        )
    assert source.read_bytes() == original
    assert source.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert source.stat().st_ctime_ns != original_stat.st_ctime_ns
    assert not output_root.exists()
    assert not list(tmp_path.glob(".restore_race_presentation.*.staging"))


def test_main_rejects_private_snapshot_restore_race(
    tmp_path,
    monkeypatch,
):
    fixture = build_review_fixture(tmp_path)
    install_fake_runtime(monkeypatch, tmp_path)

    def restore_snapshot_race(argv):
        Path(argv[-1]).write_bytes(b"composed-h264-six-view")
        inputs = [
            Path(argv[index + 1])
            for index, value in enumerate(argv[:-1])
            if value == "-i"
        ]
        snapshot = inputs[0]
        original = snapshot.read_bytes()
        original_stat = snapshot.stat()
        snapshot.chmod(0o600)
        snapshot.write_bytes(b"temporary-snapshot-race")
        snapshot.write_bytes(original)
        os.utime(
            snapshot,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        snapshot.chmod(0o400)

    monkeypatch.setattr(subject, "run_ffmpeg", restore_snapshot_race)
    output_root = tmp_path / "snapshot_restore_race_presentation"

    with pytest.raises(
        subject.PresentationContractError,
        match="private snapshot changed during composition",
    ):
        subject.main(
            [
                "--review-run",
                str(fixture["review"]),
                "--expected-review-run-sha256",
                fixture["sha256"],
                "--output-root",
                str(output_root),
            ]
        )
    assert not output_root.exists()
    assert not list(tmp_path.glob(".snapshot_restore_race_presentation.*.staging"))


def test_real_probe_rejects_an_authenticated_hash_with_an_extra_stream(
    tmp_path,
):
    fixture = build_review_fixture(tmp_path)
    materialize_real_video_fixture(fixture, tmp_path)
    label = "walking_side"
    video = fixture["media_root"] / f"{label}.mp4"
    subprocess.run(
        [
            shutil.which("ffmpeg") or "ffmpeg",
            "-nostdin",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=512x384:r=8:d=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=8000:duration=1",
            "-frames:v",
            "8",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(video),
        ],
        check=True,
    )
    rebind_fixture_video(fixture, label)
    ffmpeg = subject.resolve_named_executable("ffmpeg")
    ffprobe = subject.resolve_named_executable("ffprobe")

    with pytest.raises(
        subject.PresentationContractError,
        match="stream coverage is not exact",
    ):
        subject.authenticate_review(
            fixture["review"],
            fixture["sha256"],
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )


def test_runtime_identity_records_tool_versions_and_font_version():
    ffmpeg = subject.resolve_named_executable("ffmpeg")
    ffprobe = subject.resolve_named_executable("ffprobe")
    font = subject.resolve_regular_file(subject.FONT_PATH, "font")

    identity = subject.capture_runtime_identity(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        font=font,
    )

    assert identity["presentation_tool"]["version"] == "1"
    assert identity["presentation_tool"]["file"]["sha256"] == sha256_file(
        Path(subject.__file__)
    )
    assert identity["ffmpeg"]["version_first_line"].startswith("ffmpeg version")
    assert identity["ffprobe"]["version_first_line"].startswith("ffprobe version")
    assert identity["font"]["family"] == "DejaVu Sans"
    assert identity["font"]["subfamily"] == "Bold"
    assert identity["font"]["version"].startswith("Version ")
    assert identity["font"]["postscript_name"] == "DejaVuSans-Bold"


def test_atomic_publish_never_replaces_a_concurrent_output(tmp_path):
    review_root = tmp_path / "review"
    review_root.mkdir()
    review_path = review_root / "review_run.json"
    review_path.write_bytes(b"review")
    output = tmp_path / "output"
    location = subject.prepare_output_location(output, review_path)
    staging = subject.create_private_staging(location)
    try:
        artifact = staging.path / "artifact"
        artifact.write_bytes(b"private")
        os.fchmod(staging.descriptor, 0o555)
        staging.mode = 0o555
        output.mkdir()
        (output / "owner").write_bytes(b"concurrent")

        with pytest.raises(
            subject.PresentationContractError,
            match="refusing to replace output root",
        ):
            subject.atomic_publish_no_replace(location, staging)
        assert artifact.read_bytes() == b"private"
        assert (output / "owner").read_bytes() == b"concurrent"
    finally:
        os.fchmod(staging.descriptor, 0o700)
        staging.close()
        location.parent.close()


def test_output_parent_policy_rejects_world_writable_directory(tmp_path):
    review_root = tmp_path / "review"
    review_root.mkdir()
    review_path = review_root / "review_run.json"
    review_path.write_bytes(b"review")
    output_parent = tmp_path / "world_writable"
    output_parent.mkdir()
    output_parent.chmod(0o707)
    try:
        with pytest.raises(
            subject.PresentationContractError,
            match="owner/mode policy",
        ):
            subject.prepare_output_location(
                output_parent / "presentation",
                review_path,
            )
    finally:
        output_parent.chmod(0o700)


def test_final_parent_resolution_rejects_ancestor_symlink_swap_into_review(
    tmp_path,
    monkeypatch,
):
    fixture = build_review_fixture(tmp_path)
    install_fake_runtime(monkeypatch, tmp_path)
    output_parent = tmp_path / "safe_output_parent"
    output_parent.mkdir()
    displaced_parent = tmp_path / "displaced_output_parent"
    output_root = output_parent / "presentation"
    original_reject = subject._reject_unsafe_symlinks
    swapped = False

    def swap_after_component_check(path, label, *, file_leaf):
        nonlocal swapped
        original_reject(path, label, file_leaf=file_leaf)
        if label == "output parent" and not swapped:
            output_parent.rename(displaced_parent)
            output_parent.symlink_to(
                fixture["root"],
                target_is_directory=True,
            )
            swapped = True

    monkeypatch.setattr(
        subject,
        "_reject_unsafe_symlinks",
        swap_after_component_check,
    )
    with pytest.raises(
        subject.PresentationContractError,
        match="outside the immutable review run",
    ):
        subject.main(
            [
                "--review-run",
                str(fixture["review"]),
                "--expected-review-run-sha256",
                fixture["sha256"],
                "--output-root",
                str(output_root),
            ]
        )
    assert swapped is True
    assert not output_root.exists()
    assert not list(fixture["root"].glob(".presentation.*.staging"))


def test_main_dirfd_publish_never_replaces_concurrent_output(
    tmp_path,
    monkeypatch,
):
    fixture = build_review_fixture(tmp_path)
    install_fake_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        subject,
        "run_ffmpeg",
        lambda argv: Path(argv[-1]).write_bytes(b"composed-h264-six-view"),
    )
    output_parent = tmp_path / "concurrent_parent"
    output_parent.mkdir()
    output_root = output_parent / "presentation"
    original_publish = subject.atomic_publish_no_replace

    def publish_after_concurrent_creator(location, staging):
        output_root.mkdir()
        (output_root / "owner").write_bytes(b"concurrent")
        original_publish(location, staging)

    monkeypatch.setattr(
        subject,
        "atomic_publish_no_replace",
        publish_after_concurrent_creator,
    )
    with pytest.raises(
        subject.PresentationContractError,
        match="refusing to replace output root",
    ):
        subject.main(
            [
                "--review-run",
                str(fixture["review"]),
                "--expected-review-run-sha256",
                fixture["sha256"],
                "--output-root",
                str(output_root),
            ]
        )
    assert (output_root / "owner").read_bytes() == b"concurrent"
    assert not list(output_parent.glob(".presentation.*.staging"))
    quarantines = list(output_parent.glob(".quarantine.presentation.*"))
    assert len(quarantines) == 1
    assert {entry.name for entry in quarantines[0].iterdir()} == {
        subject.OUTPUT_VIDEO_NAME,
        subject.RECEIPT_NAME,
    }
    retained = list(output_parent.glob(".retained.presentation.*.composition-inputs"))
    assert len(retained) == 1
    assert {entry.name for entry in retained[0].iterdir()} == {
        f"{index:02d}_{label}.mp4" for index, label in enumerate(subject.MEDIA_LABELS)
    }


def test_parent_swap_and_same_inode_move_fail_without_deleting_published_final(
    tmp_path,
    monkeypatch,
):
    fixture = build_review_fixture(tmp_path)
    install_fake_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        subject,
        "run_ffmpeg",
        lambda argv: Path(argv[-1]).write_bytes(b"composed-h264-six-view"),
    )
    output_parent = tmp_path / "published_parent"
    output_parent.mkdir()
    displaced_parent = tmp_path / "published_parent.displaced"
    output_root = output_parent / "presentation"
    original_publish = subject.atomic_publish_no_replace
    published_inode = None

    def publish_then_swap_parent(location, staging):
        nonlocal published_inode
        original_publish(location, staging)
        published_inode = staging.inode
        output_parent.rename(displaced_parent)
        output_parent.mkdir(mode=location.parent.mode)
        displaced_parent.chmod(0o700)
        output_parent.chmod(0o700)
        os.fchmod(staging.descriptor, 0o700)
        (displaced_parent / location.output_name).rename(
            output_parent / location.output_name
        )
        os.fchmod(staging.descriptor, 0o555)
        displaced_parent.chmod(location.parent.mode)
        output_parent.chmod(location.parent.mode)

    monkeypatch.setattr(
        subject,
        "atomic_publish_no_replace",
        publish_then_swap_parent,
    )
    try:
        with pytest.raises(
            subject.PresentationContractError,
            match="published output parent entry identity changed",
        ):
            subject.main(
                [
                    "--review-run",
                    str(fixture["review"]),
                    "--expected-review-run-sha256",
                    fixture["sha256"],
                    "--output-root",
                    str(output_root),
                ]
            )
        assert output_root.is_dir()
        assert output_root.stat().st_ino == published_inode
        assert (output_root / subject.OUTPUT_VIDEO_NAME).is_file()
        assert (output_root / subject.RECEIPT_NAME).is_file()
        assert not list(displaced_parent.glob(".presentation.*.staging"))
    finally:
        unseal_output(output_root)


def test_cleanup_substitution_quarantines_owned_inode_without_deleting_replacement(
    tmp_path,
    monkeypatch,
):
    fixture = build_review_fixture(tmp_path)
    install_fake_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        subject,
        "run_ffmpeg",
        lambda argv: Path(argv[-1]).write_bytes(b"composed-h264-six-view"),
    )
    output_parent = tmp_path / "cleanup_parent"
    output_parent.mkdir()
    output_root = output_parent / "presentation"
    original_seal = subject.seal_readonly_tree
    replacement_sentinel = None
    owned_inode = None

    def substitute_staging_before_failure(staging, inventory):
        nonlocal replacement_sentinel, owned_inode
        owned_inode = staging.inode
        displaced = staging.path.with_name(f"{staging.name}.displaced")
        staging.path.rename(displaced)
        staging.path.mkdir()
        replacement_sentinel = staging.path / "must_survive"
        replacement_sentinel.write_bytes(b"replacement")
        raise subject.PresentationContractError("forced post-substitution failure")

    monkeypatch.setattr(
        subject,
        "seal_readonly_tree",
        substitute_staging_before_failure,
    )
    try:
        with pytest.raises(
            subject.PresentationContractError,
            match="forced post-substitution failure",
        ):
            subject.main(
                [
                    "--review-run",
                    str(fixture["review"]),
                    "--expected-review-run-sha256",
                    fixture["sha256"],
                    "--output-root",
                    str(output_root),
                ]
            )
        assert replacement_sentinel.read_bytes() == b"replacement"
        quarantines = list(output_parent.glob(".quarantine.presentation.*"))
        assert len(quarantines) == 1
        assert quarantines[0].stat().st_ino == owned_inode
        assert not output_root.exists()
    finally:
        monkeypatch.setattr(subject, "seal_readonly_tree", original_seal)


def test_cleanup_midrename_root_substitution_retains_victim_and_owned_tree(
    tmp_path,
    monkeypatch,
):
    fixture = build_review_fixture(tmp_path)
    install_fake_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        subject,
        "run_ffmpeg",
        lambda argv: Path(argv[-1]).write_bytes(b"composed-h264-six-view"),
    )
    output_parent = tmp_path / "cleanup_midrename_parent"
    output_parent.mkdir()
    output_root = output_parent / "presentation"
    original_rename = subject._renameat2_no_replace
    detached_owned = output_parent / "detached-owned-staging"
    injected = False

    def fail_before_publication(_staging, _inventory):
        raise subject.PresentationContractError("forced cleanup race")

    def replace_root_at_quarantine_boundary(
        source_directory_descriptor,
        source_name,
        target_directory_descriptor,
        target_name,
        *,
        target_display,
    ):
        nonlocal injected
        if target_name.startswith(".quarantine.presentation.") and not injected:
            injected = True
            source = output_parent / source_name
            source.rename(detached_owned)
            source.mkdir()
            (source / "external_victim").write_bytes(b"must survive cleanup")
        return original_rename(
            source_directory_descriptor,
            source_name,
            target_directory_descriptor,
            target_name,
            target_display=target_display,
        )

    monkeypatch.setattr(subject, "seal_readonly_tree", fail_before_publication)
    monkeypatch.setattr(
        subject, "_renameat2_no_replace", replace_root_at_quarantine_boundary
    )
    with pytest.raises(
        subject.PresentationContractError,
        match="forced cleanup race",
    ):
        subject.main(
            [
                "--review-run",
                str(fixture["review"]),
                "--expected-review-run-sha256",
                fixture["sha256"],
                "--output-root",
                str(output_root),
            ]
        )

    assert injected is True
    quarantines = list(output_parent.glob(".quarantine.presentation.*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "external_victim").read_bytes() == (
        b"must survive cleanup"
    )
    assert (detached_owned / subject.OUTPUT_VIDEO_NAME).is_file()
    assert (detached_owned / subject.RECEIPT_NAME).is_file()


def test_staging_restore_race_is_quarantined_before_publication(
    tmp_path,
    monkeypatch,
):
    fixture = build_review_fixture(tmp_path)
    install_fake_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        subject,
        "run_ffmpeg",
        lambda argv: Path(argv[-1]).write_bytes(b"composed-h264-six-view"),
    )
    output_parent = tmp_path / "byte_swap_parent"
    output_parent.mkdir()
    output_root = output_parent / "presentation"
    original_seal = subject.seal_readonly_tree

    def restore_video_bytes_before_seal(staging, inventory):
        video = staging.path / subject.OUTPUT_VIDEO_NAME
        original = video.read_bytes()
        original_stat = video.stat()
        video.write_bytes(b"x" * len(original))
        video.write_bytes(original)
        os.utime(
            video,
            ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns),
        )
        original_seal(staging, inventory)

    monkeypatch.setattr(
        subject,
        "seal_readonly_tree",
        restore_video_bytes_before_seal,
    )
    with pytest.raises(
        subject.PresentationContractError,
        match="publication artifact identity changed",
    ):
        subject.main(
            [
                "--review-run",
                str(fixture["review"]),
                "--expected-review-run-sha256",
                fixture["sha256"],
                "--output-root",
                str(output_root),
            ]
        )
    quarantines = list(output_parent.glob(".quarantine.presentation.*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / subject.OUTPUT_VIDEO_NAME).is_file()
    assert not output_root.exists()
    assert not list(output_parent.glob(".presentation.*.staging"))


def test_unknown_staging_entry_forces_quarantine_without_recursive_cleanup(
    tmp_path,
    monkeypatch,
):
    fixture = build_review_fixture(tmp_path)
    install_fake_runtime(monkeypatch, tmp_path)
    monkeypatch.setattr(
        subject,
        "run_ffmpeg",
        lambda argv: Path(argv[-1]).write_bytes(b"composed-h264-six-view"),
    )
    output_parent = tmp_path / "unknown_entry_parent"
    output_parent.mkdir()
    output_root = output_parent / "presentation"
    original_seal = subject.seal_readonly_tree

    def add_unknown_entry(staging, inventory):
        (staging.path / "unknown_must_survive").write_bytes(b"unknown")
        raise subject.PresentationContractError("forced unknown-entry failure")

    monkeypatch.setattr(subject, "seal_readonly_tree", add_unknown_entry)
    try:
        with pytest.raises(
            subject.PresentationContractError,
            match="forced unknown-entry failure",
        ):
            subject.main(
                [
                    "--review-run",
                    str(fixture["review"]),
                    "--expected-review-run-sha256",
                    fixture["sha256"],
                    "--output-root",
                    str(output_root),
                ]
            )
        quarantines = list(output_parent.glob(".quarantine.presentation.*"))
        assert len(quarantines) == 1
        assert (quarantines[0] / "unknown_must_survive").read_bytes() == b"unknown"
        assert not output_root.exists()
    finally:
        monkeypatch.setattr(subject, "seal_readonly_tree", original_seal)


def test_real_ffmpeg_end_to_end_publishes_only_sealed_video_and_receipt(
    tmp_path,
):
    fixture = build_review_fixture(tmp_path)
    materialize_real_video_fixture(fixture, tmp_path)
    output_root = tmp_path / "real_presentation"
    copied_root = tmp_path / "copied_presentation"
    review_before = fixture["review"].read_bytes()

    try:
        assert (
            subject.main(
                [
                    "--review-run",
                    str(fixture["review"]),
                    "--expected-review-run-sha256",
                    fixture["sha256"],
                    "--output-root",
                    str(output_root),
                ]
            )
            == 0
        )
        video = output_root / subject.OUTPUT_VIDEO_NAME
        receipt_path = output_root / subject.RECEIPT_NAME
        receipt = subject.strict_json_loads(receipt_path.read_bytes())

        assert {path.name for path in output_root.iterdir()} == {
            subject.OUTPUT_VIDEO_NAME,
            subject.RECEIPT_NAME,
        }
        assert stat.S_IMODE(output_root.stat().st_mode) == 0o555
        assert stat.S_IMODE(video.stat().st_mode) == 0o444
        assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
        assert receipt["output"]["readback"] == {
            "stream_count": 1,
            "stream_index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "pix_fmt": "yuv420p",
            "sample_aspect_ratio": "1:1",
            "width": 1536,
            "height": 768,
            "r_frame_rate": "8/1",
            "avg_frame_rate": "8/1",
            "nb_frames": 8,
            "nb_read_frames": 8,
            "duration_seconds": 1.0,
        }
        assert receipt["output"]["full_decode_passed"] is True
        content = receipt["frame_cell_content_readback"]
        assert content["schema"] == subject.CONTENT_READBACK_SCHEMA
        assert content["all_cells_all_frames_passed"] is True
        assert content["content_readback_sha256"] == subject.hash_without(
            content, "content_readback_sha256"
        )
        assert [cell["label"] for cell in content["cells"]] == list(
            subject.MEDIA_LABELS
        )
        assert {len(cell["frames"]) for cell in content["cells"]} == {8}
        assert all(
            frame["passed"] for cell in content["cells"] for frame in cell["frames"]
        )
        assert receipt["receipt_sha256"] == subject.hash_without(
            receipt, "receipt_sha256"
        )
        assert all(receipt["automatic_checks"].values())
        assert fixture["review"].read_bytes() == review_before
        ffmpeg_inputs = [
            Path(receipt["command"]["ffmpeg_argv"][index + 1])
            for index, value in enumerate(receipt["command"]["ffmpeg_argv"][:-1])
            if value == "-i"
        ]
        assert len(ffmpeg_inputs) == 6
        assert all(
            path.parent.name == ".private_input_snapshots" for path in ffmpeg_inputs
        )
        assert all(not path.exists() for path in ffmpeg_inputs)
        loaded, receipt_record = subject.load_presentation_receipt(
            receipt_path,
            sha256_file(receipt_path),
            expected_source_review_sha256=fixture["sha256"],
        )
        assert loaded == receipt
        assert receipt_record == file_record(receipt_path)

        copied_root.mkdir()
        copied_video = copied_root / subject.OUTPUT_VIDEO_NAME
        copied_receipt = copied_root / subject.RECEIPT_NAME
        shutil.copy2(video, copied_video)
        shutil.copy2(receipt_path, copied_receipt)
        copied_video.chmod(0o444)
        copied_receipt.chmod(0o444)
        copied_root.chmod(0o555)
        with pytest.raises(
            subject.PresentationContractError,
            match="must be next to its receipt",
        ):
            subject.load_presentation_receipt(
                copied_receipt,
                sha256_file(copied_receipt),
                expected_source_review_sha256=fixture["sha256"],
            )

        unseal_output(output_root)
        video.write_bytes(b"replaced presentation video")
        video.chmod(0o444)
        receipt_path.chmod(0o444)
        output_root.chmod(0o555)
        with pytest.raises(
            subject.PresentationContractError,
            match="file-record authentication",
        ):
            subject.load_presentation_receipt(
                receipt_path,
                sha256_file(receipt_path),
                expected_source_review_sha256=fixture["sha256"],
            )
    finally:
        unseal_output(output_root)
        unseal_output(copied_root)


def test_real_content_readback_rejects_swapped_cells(
    tmp_path,
    monkeypatch,
):
    fixture = build_review_fixture(tmp_path)
    materialize_real_video_fixture(fixture, tmp_path)
    output_root = tmp_path / "swapped_cell_presentation"

    def compose_with_swapped_inputs(argv):
        changed = list(argv)
        input_value_indices = [
            index + 1 for index, value in enumerate(changed[:-1]) if value == "-i"
        ]
        changed[input_value_indices[0]], changed[input_value_indices[1]] = (
            changed[input_value_indices[1]],
            changed[input_value_indices[0]],
        )
        subprocess.run(changed, cwd=subject.SPEAR_ROOT, check=True)

    monkeypatch.setattr(subject, "run_ffmpeg", compose_with_swapped_inputs)
    with pytest.raises(
        subject.PresentationContractError,
        match="frame/cell content readback failed",
    ):
        subject.main(
            [
                "--review-run",
                str(fixture["review"]),
                "--expected-review-run-sha256",
                fixture["sha256"],
                "--output-root",
                str(output_root),
            ]
        )
    assert not output_root.exists()
    assert not list(tmp_path.glob(".swapped_cell_presentation.*.staging"))
