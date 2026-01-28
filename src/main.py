from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .config import DEFAULT_CONFIG, Config
from .io_video import get_video_meta, make_writer, open_video
from .motion import MotionCompensator
from .render import draw_boxes, draw_overlay
from .report import write_events_csv, write_events_json, write_per_second_counts, write_tracklets_csv
from .tracking import TruckTracker
from .tracklets import TrackletManager


def _corridor_flag(center: tuple[float, float], width: int, height: int, config: Config) -> bool:
    x, y = center
    return (
        config.corridor_x_min * width <= x <= config.corridor_x_max * width
        and config.corridor_y_min * height <= y <= config.corridor_y_max * height
    )


def run(config: Config = DEFAULT_CONFIG) -> None:
    config.ensure_output_dirs()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.device = device

    tracker = TruckTracker(str(config.weights_path), device=config.device, class_name=config.class_name, conf=config.conf_thresh)
    capture = open_video(config.input_video)
    meta = get_video_meta(capture)
    writer = make_writer(config.output_video, meta.fps, meta.width, meta.height)

    motion = MotionCompensator(
        max_features=config.orb_features,
        match_ratio=config.match_ratio,
        min_matches=config.motion_min_matches,
    )

    tracklet_manager = TrackletManager(
        max_disappeared=config.max_disappeared,
        merge_window_sec=config.merge_window_sec,
        merge_distance_px=config.merge_distance_px,
        merge_area_ratio_tol=config.merge_area_ratio_tol,
        stationary_dx_thresh=config.stationary_dx_thresh,
        approach_area_ratio=config.approach_area_ratio,
        approach_dx_thresh=config.approach_dx_thresh,
        direction_dx_thresh=config.direction_dx_thresh,
        direction_dominance=config.direction_dominance,
        corridor_min_ratio=config.corridor_min_ratio,
    )

    start_time = time.time()

    for frame_idx in tqdm(range(meta.frame_count), desc="Processing frames"):
        ret, frame = capture.read()
        if not ret:
            break
        time_sec = frame_idx / meta.fps

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_est = motion.update(frame_gray)

        detections = tracker.track(frame)
        detection_rows: list[tuple[int, tuple[float, float, float, float], float]] = []
        stable_centers: dict[int, tuple[float, float]] = {}
        corridor_flags: dict[int, bool] = {}

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            area = max(1.0, (x2 - x1) * (y2 - y1))
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            stable_center = motion.stabilize_point(center, motion_est.cumulative)
            detection_rows.append((det.tracker_id, det.bbox, area))
            stable_centers[det.tracker_id] = stable_center
            corridor_flags[det.tracker_id] = _corridor_flag(center, meta.width, meta.height, config)

        tracklet_manager.update_tracklets(
            frame_idx=frame_idx,
            time_sec=time_sec,
            detections=detection_rows,
            stable_centers=stable_centers,
            corridor_flags=corridor_flags,
        )

        vehicle_records = tracklet_manager.get_vehicle_records()
        active_tracklets = tracklet_manager.get_active_tracklets()

        boxes_to_draw = []
        for tracker_id, tracklet in active_tracklets.items():
            record = vehicle_records[tracklet.vehicle_uid]
            direction = record.direction or "OTHER"
            last_obs = tracklet.observations[-1]
            boxes_to_draw.append((last_obs.bbox, direction, tracker_id, tracklet.vehicle_uid))

        finalized = tracklet_manager.get_finalized()
        l2r_count = sum(1 for rec in finalized if rec.direction == "L2R")
        r2l_count = sum(1 for rec in finalized if rec.direction == "R2L")
        elapsed = max(1e-6, time.time() - start_time)
        fps_proc = (frame_idx + 1) / elapsed

        draw_boxes(
            frame,
            boxes=boxes_to_draw,
            font_scale=max(0.4, meta.height * config.font_scale_ratio / 20.0),
            box_thickness=config.box_thickness,
            thin_box_thickness=config.thin_box_thickness,
        )
        overlay_lines = [
            f"L2R: {l2r_count}  R2L: {r2l_count}",
            f"FPS: {fps_proc:.1f}  Frame: {frame_idx}  Time: {time_sec:.1f}s",
        ]
        draw_overlay(
            frame,
            lines=overlay_lines,
            font_scale=max(0.4, meta.height * config.font_scale_ratio / 20.0),
            thickness=config.font_thickness,
        )

        writer.write(frame)

    capture.release()
    writer.release()

    tracklet_manager.finalize_all()
    finalized = tracklet_manager.get_finalized()
    tracklet_rows: list[dict] = []
    for record in tracklet_manager.get_vehicle_records().values():
        if not record.observations:
            continue
        first_stable_x = record.observations[0].stable_center[0]
        for obs in record.observations:
            dx_rel = obs.stable_center[0] - first_stable_x
            tracklet_rows.append(
                {
                    "vehicle_uid": record.vehicle_uid,
                    "tracker_id": obs.tracker_id,
                    "frame_idx": obs.frame_idx,
                    "time_sec": round(obs.time_sec, 3),
                    "bbox": obs.bbox,
                    "cx_stable": round(obs.stable_center[0], 2),
                    "cy_stable": round(obs.stable_center[1], 2),
                    "dx_rel": round(dx_rel, 2),
                    "area": round(obs.area, 2),
                    "in_corridor": obs.in_corridor,
                }
            )

    write_events_csv(config.events_csv, finalized)
    write_events_json(config.events_json, finalized)
    write_tracklets_csv(config.tracklets_csv, tracklet_rows)
    write_per_second_counts(config.per_second_csv, finalized)


if __name__ == "__main__":
    run(DEFAULT_CONFIG)
