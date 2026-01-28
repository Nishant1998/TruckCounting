from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionEstimate:
    cumulative: np.ndarray
    valid: bool


class MotionCompensator:
    def __init__(self, max_features: int = 600, match_ratio: float = 0.75, min_matches: int = 10) -> None:
        self.orb = cv2.ORB_create(max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.match_ratio = match_ratio
        self.min_matches = min_matches
        self.cumulative = np.eye(3, dtype=np.float32)
        self.prev_gray: np.ndarray | None = None

    def update(self, frame_gray: np.ndarray) -> MotionEstimate:
        if self.prev_gray is None:
            self.prev_gray = frame_gray
            return MotionEstimate(cumulative=self.cumulative.copy(), valid=False)

        kp1, des1 = self.orb.detectAndCompute(self.prev_gray, None)
        kp2, des2 = self.orb.detectAndCompute(frame_gray, None)
        if des1 is None or des2 is None or len(kp1) < self.min_matches or len(kp2) < self.min_matches:
            self.prev_gray = frame_gray
            return MotionEstimate(cumulative=self.cumulative.copy(), valid=False)

        matches = self.matcher.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < self.match_ratio * n.distance:
                good.append(m)

        if len(good) < self.min_matches:
            self.prev_gray = frame_gray
            return MotionEstimate(cumulative=self.cumulative.copy(), valid=False)

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        affine, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3)
        if affine is None:
            self.prev_gray = frame_gray
            return MotionEstimate(cumulative=self.cumulative.copy(), valid=False)

        transform = np.eye(3, dtype=np.float32)
        transform[:2, :] = affine
        self.cumulative = transform @ self.cumulative
        self.prev_gray = frame_gray
        return MotionEstimate(cumulative=self.cumulative.copy(), valid=True)

    @staticmethod
    def stabilize_point(point: tuple[float, float], cumulative: np.ndarray) -> tuple[float, float]:
        matrix = np.linalg.inv(cumulative)
        vec = np.array([point[0], point[1], 1.0], dtype=np.float32)
        warped = matrix @ vec
        return float(warped[0]), float(warped[1])
