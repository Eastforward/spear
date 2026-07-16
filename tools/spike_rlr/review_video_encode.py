"""Low-overhead review-video encoders shared by marker and top-down passes."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def _rgb_frame(frame, expected_shape: tuple[int, int, int] | None = None) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("review frame must be uint8 RGB with shape HxWx3")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(
            f"review frame shape changed: expected {expected_shape}, got {array.shape}"
        )
    return np.ascontiguousarray(array)


def encode_rgb_frames(
    frames: Iterable[np.ndarray],
    output_path: Path,
    *,
    fps: int,
    preset: str = "veryfast",
) -> int:
    """Stream RGB frames to FFmpeg without staging per-frame PNG files."""
    if int(fps) <= 0:
        raise ValueError("review video fps must be positive")
    iterator = iter(frames)
    try:
        first = _rgb_frame(next(iterator))
    except StopIteration as error:
        raise ValueError("review video requires at least one frame") from error

    height, width, _channels = first.shape
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.stem}.",
        suffix=output_path.suffix,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)

    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(int(fps)),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        str(temporary),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    count = 0
    try:
        assert process.stdin is not None
        process.stdin.write(first.tobytes())
        count = 1
        for frame in iterator:
            process.stdin.write(_rgb_frame(frame, first.shape).tobytes())
            count += 1
        process.stdin.close()
        return_code = process.wait()
        assert process.stderr is not None
        stderr = process.stderr.read().decode("utf-8", errors="replace")
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg raw RGB encoder returned {return_code}: {stderr.strip()}"
            )
        os.replace(temporary, output_path)
        return count
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        temporary.unlink(missing_ok=True)
        raise
