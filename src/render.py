from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


COLOR_L2R = (0, 200, 0)
COLOR_R2L = (0, 0, 220)
COLOR_OTHER = (40, 40, 40)


def draw_boxes(
    frame: np.ndarray,
    boxes: Iterable[tuple[tuple[float, float, float, float], str, int, int]],
    font_scale: float,
    box_thickness: int,
    thin_box_thickness: int,
) -> None:
    for bbox, direction, tracker_id, vehicle_uid in boxes:
        color = COLOR_OTHER
        thickness = thin_box_thickness
        if direction == "L2R":
            color = COLOR_L2R
            thickness = box_thickness
        elif direction == "R2L":
            color = COLOR_R2L
            thickness = box_thickness

        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"T{tracker_id} V{vehicle_uid}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, thickness - 1),
            cv2.LINE_AA,
        )


def draw_overlay(
    frame: np.ndarray,
    lines: list[str],
    font_scale: float,
    thickness: int,
) -> None:
    height, width = frame.shape[:2]
    y = int(height * 0.05)
    x = int(width * 0.02)
    line_height = int(height * 0.05)
    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height
