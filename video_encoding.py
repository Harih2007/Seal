"""Browser-playable annotated video encoding, preferring NVIDIA NVENC when available."""
from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent


def find_ffmpeg() -> str | None:
    configured = os.environ.get("FFMPEG_PATH")
    candidates = [configured, str(APP_DIR / ".tools" / "ffmpeg" / "bin" / "ffmpeg.exe"), shutil.which("ffmpeg")]
    return next((str(path) for path in candidates if path and Path(path).is_file()), None)


@lru_cache(maxsize=1)
def nvenc_available() -> bool:
    """Check that FFmpeg can perform a real H.264 NVENC encode on this machine."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return False
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
             "color=c=black:s=128x128:r=24:d=0.1", "-frames:v", "1", "-c:v", "h264_nvenc",
             "-preset", "p4", "-f", "null", "NUL"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def preferred_video_suffix() -> str:
    return ".mp4" if nvenc_available() else ".webm"


class NvencVideoWriter:
    """Stream OpenCV BGR frames to FFmpeg's NVIDIA H.264 hardware encoder."""
    encoder = "NVIDIA NVENC H.264"

    def __init__(self, path: Path, fps: float, width: int, height: int):
        ffmpeg = find_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("FFmpeg was not found for NVENC video output.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.process = subprocess.Popen(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
             "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
             "-framerate", f"{fps:.6f}", "-i", "pipe:0", "-an", "-fps_mode", "passthrough",
             "-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq", "-rc", "vbr",
             "-cq", "21", "-b:v", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def isOpened(self) -> bool:
        return self.process.poll() is None

    def write(self, frame):
        if self.process.poll() is not None or self.process.stdin is None:
            raise RuntimeError("FFmpeg stopped while writing the NVENC annotated video.")
        try:
            self.process.stdin.write(memoryview(np.ascontiguousarray(frame)))
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("FFmpeg could not write an annotated frame through NVENC.") from exc

    def release(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            return_code = self.process.wait(timeout=60)
        except subprocess.TimeoutExpired as exc:
            self.process.kill()
            self.process.wait()
            raise RuntimeError("FFmpeg did not finish the NVENC video export in time.") from exc
        if return_code != 0:
            self.path.unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg NVENC encoding failed (exit code {return_code}).")


class OpenCvVideoWriter:
    """Small adapter because OpenCV's Python VideoWriter cannot hold extra attributes."""
    def __init__(self, writer, encoder: str):
        self.writer = writer
        self.encoder = encoder

    def isOpened(self) -> bool:
        return self.writer.isOpened()

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()


def create_video_writer(path: Path, fps: float, width: int, height: int):
    """Use NVENC for MP4, falling back to the existing OpenCV encoders."""
    if path.suffix.lower() == ".mp4" and nvenc_available():
        return NvencVideoWriter(path, fps, width, height)
    codec = "VP80" if path.suffix.lower() == ".webm" else "mp4v"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create annotated video with the {codec} encoder.")
    return OpenCvVideoWriter(writer, "OpenCV " + ("VP8" if codec == "VP80" else "MPEG-4 Part 2"))
