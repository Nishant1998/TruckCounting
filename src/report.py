from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from .tracklets import VehicleRecord


def write_events_csv(path: Path, records: Iterable[VehicleRecord]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "vehicle_uid",
                "direction",
                "start_time_sec",
                "end_time_sec",
                "start_bbox",
                "end_bbox",
                "tracker_ids_merged",
                "dx_total_stable",
                "median_speed_stable",
                "area_ratio",
            ]
        )
        for record in records:
            metrics = record.metrics
            writer.writerow(
                [
                    record.vehicle_uid,
                    record.direction,
                    round(record.start_time, 3),
                    round(record.end_time, 3),
                    record.start_bbox,
                    record.end_bbox,
                    record.tracker_ids,
                    round(metrics.dx_total if metrics else 0.0, 3),
                    round(metrics.median_speed if metrics else 0.0, 3),
                    round(metrics.area_ratio if metrics else 1.0, 3),
                ]
            )


def write_events_json(path: Path, records: Iterable[VehicleRecord]) -> None:
    payload = []
    for record in records:
        metrics = record.metrics
        payload.append(
            {
                "vehicle_uid": record.vehicle_uid,
                "direction": record.direction,
                "start_time_sec": record.start_time,
                "end_time_sec": record.end_time,
                "start_bbox": record.start_bbox,
                "end_bbox": record.end_bbox,
                "tracker_ids_merged": record.tracker_ids,
                "dx_total_stable": metrics.dx_total if metrics else 0.0,
                "median_speed_stable": metrics.median_speed if metrics else 0.0,
                "area_ratio": metrics.area_ratio if metrics else 1.0,
            }
        )
    path.write_text(json.dumps(payload, indent=2))


def write_tracklets_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_per_second_counts(path: Path, records: Iterable[VehicleRecord]) -> None:
    counts: dict[int, dict[str, int]] = {}
    for record in records:
        sec = int(record.start_time)
        counts.setdefault(sec, {"L2R": 0, "R2L": 0})
        if record.direction == "L2R":
            counts[sec]["L2R"] += 1
        elif record.direction == "R2L":
            counts[sec]["R2L"] += 1

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sec", "count_L2R", "count_R2L"])
        for sec in sorted(counts.keys()):
            writer.writerow([sec, counts[sec]["L2R"], counts[sec]["R2L"]])
