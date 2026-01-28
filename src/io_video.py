from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    frame_count: int


def open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {path}")
    return capture


def get_video_meta(capture: cv2.VideoCapture) -> VideoMeta:
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return VideoMeta(fps=fps, width=width, height=height, frame_count=frame_count)


def make_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for {path}")
    return writer
