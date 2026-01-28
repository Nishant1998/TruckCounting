from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass
class TrackletMetrics:
    dx_total: float
    dy_total: float
    median_dx: float
    median_speed: float
    area_ratio: float
    corridor_ratio: float


def compute_metrics(stable_points: list[tuple[float, float]], areas: list[float], in_corridor: list[bool]) -> TrackletMetrics:
    dxs = []
    dys = []
    speeds = []
    for (x0, y0), (x1, y1) in zip(stable_points[:-1], stable_points[1:]):
        dx = x1 - x0
        dy = y1 - y0
        dxs.append(dx)
        dys.append(dy)
        speeds.append((dx**2 + dy**2) ** 0.5)
    dx_total = stable_points[-1][0] - stable_points[0][0]
    dy_total = stable_points[-1][1] - stable_points[0][1]
    median_dx = median([abs(dx) for dx in dxs]) if dxs else 0.0
    median_speed = median(speeds) if speeds else 0.0
    area_min = min(areas) if areas else 1.0
    area_max = max(areas) if areas else area_min
    area_ratio = area_max / area_min if area_min > 0 else 1.0
    corridor_ratio = sum(in_corridor) / len(in_corridor) if in_corridor else 0.0
    return TrackletMetrics(
        dx_total=dx_total,
        dy_total=dy_total,
        median_dx=median_dx,
        median_speed=median_speed,
        area_ratio=area_ratio,
        corridor_ratio=corridor_ratio,
    )


def classify_direction(
    metrics: TrackletMetrics,
    stationary_thresh: float,
    approach_area_ratio: float,
    approach_dx_thresh: float,
    direction_dx_thresh: float,
    direction_dominance: float,
    corridor_min_ratio: float,
) -> str:
    if metrics.corridor_ratio < corridor_min_ratio:
        return "OTHER"
    if metrics.median_dx < stationary_thresh:
        return "IGNORE_STATIONARY"
    if metrics.area_ratio > approach_area_ratio and abs(metrics.dx_total) < approach_dx_thresh:
        return "IGNORE_APPROACH"
    if abs(metrics.dx_total) >= direction_dx_thresh and abs(metrics.dx_total) > direction_dominance * abs(metrics.dy_total):
        return "L2R" if metrics.dx_total > 0 else "R2L"
    return "OTHER"
