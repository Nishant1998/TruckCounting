from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .classify import TrackletMetrics, classify_direction, compute_metrics


@dataclass
class FrameObservation:
    tracker_id: int
    frame_idx: int
    time_sec: float
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    stable_center: tuple[float, float]
    area: float
    in_corridor: bool


@dataclass
class Tracklet:
    tracker_id: int
    vehicle_uid: int
    observations: List[FrameObservation] = field(default_factory=list)
    last_seen_frame: int = 0
    disappeared: int = 0


@dataclass
class VehicleRecord:
    vehicle_uid: int
    tracker_ids: List[int] = field(default_factory=list)
    observations: List[FrameObservation] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    start_bbox: tuple[float, float, float, float] | None = None
    end_bbox: tuple[float, float, float, float] | None = None
    metrics: Optional[TrackletMetrics] = None
    direction: Optional[str] = None


@dataclass
class PendingFinalize:
    vehicle_uid: int
    due_time: float


@dataclass
class RecentEnding:
    vehicle_uid: int
    end_time: float
    end_point: tuple[float, float]
    end_area: float


def should_merge_tracklets(
    start_time: float,
    start_point: tuple[float, float],
    start_area: float,
    ending: RecentEnding,
    merge_window: float,
    merge_distance: float,
    area_ratio_tol: float,
) -> bool:
    if start_time - ending.end_time > merge_window:
        return False
    dx = start_point[0] - ending.end_point[0]
    dy = start_point[1] - ending.end_point[1]
    if (dx * dx + dy * dy) ** 0.5 > merge_distance:
        return False
    if ending.end_area <= 0 or start_area <= 0:
        return False
    ratio = start_area / ending.end_area
    return area_ratio_tol <= ratio <= (1 / area_ratio_tol)


class TrackletManager:
    def __init__(
        self,
        max_disappeared: int,
        merge_window_sec: float,
        merge_distance_px: float,
        merge_area_ratio_tol: float,
        stationary_dx_thresh: float,
        approach_area_ratio: float,
        approach_dx_thresh: float,
        direction_dx_thresh: float,
        direction_dominance: float,
        corridor_min_ratio: float,
    ) -> None:
        self.max_disappeared = max_disappeared
        self.merge_window_sec = merge_window_sec
        self.merge_distance_px = merge_distance_px
        self.merge_area_ratio_tol = merge_area_ratio_tol
        self.stationary_dx_thresh = stationary_dx_thresh
        self.approach_area_ratio = approach_area_ratio
        self.approach_dx_thresh = approach_dx_thresh
        self.direction_dx_thresh = direction_dx_thresh
        self.direction_dominance = direction_dominance
        self.corridor_min_ratio = corridor_min_ratio

        self.next_vehicle_uid = 1
        self.active_tracklets: Dict[int, Tracklet] = {}
        self.vehicles: Dict[int, VehicleRecord] = {}
        self.pending_finalize: Dict[int, PendingFinalize] = {}
        self.recent_endings: List[RecentEnding] = []
        self.finalized: List[VehicleRecord] = []

    def _create_vehicle(self, tracker_id: int, observation: FrameObservation) -> Tracklet:
        vehicle_uid = self.next_vehicle_uid
        self.next_vehicle_uid += 1
        record = VehicleRecord(vehicle_uid=vehicle_uid, tracker_ids=[tracker_id])
        record.start_time = observation.time_sec
        record.end_time = observation.time_sec
        record.start_bbox = observation.bbox
        record.end_bbox = observation.bbox
        record.observations.append(observation)
        self.vehicles[vehicle_uid] = record
        tracklet = Tracklet(tracker_id=tracker_id, vehicle_uid=vehicle_uid)
        tracklet.observations.append(observation)
        tracklet.last_seen_frame = observation.frame_idx
        return tracklet

    def _attach_vehicle(self, vehicle_uid: int, tracker_id: int, observation: FrameObservation) -> Tracklet:
        record = self.vehicles[vehicle_uid]
        if tracker_id not in record.tracker_ids:
            record.tracker_ids.append(tracker_id)
        record.observations.append(observation)
        record.end_time = observation.time_sec
        record.end_bbox = observation.bbox
        tracklet = Tracklet(tracker_id=tracker_id, vehicle_uid=vehicle_uid)
        tracklet.observations.append(observation)
        tracklet.last_seen_frame = observation.frame_idx
        return tracklet

    def _find_merge_candidate(self, observation: FrameObservation) -> Optional[int]:
        for ending in list(self.recent_endings):
            if should_merge_tracklets(
                observation.time_sec,
                observation.stable_center,
                observation.area,
                ending,
                self.merge_window_sec,
                self.merge_distance_px,
                self.merge_area_ratio_tol,
            ):
                self.recent_endings.remove(ending)
                if ending.vehicle_uid in self.pending_finalize:
                    self.pending_finalize.pop(ending.vehicle_uid, None)
                return ending.vehicle_uid
        return None

    def update_tracklets(
        self,
        frame_idx: int,
        time_sec: float,
        detections: list[tuple[int, tuple[float, float, float, float], float]],
        stable_centers: dict[int, tuple[float, float]],
        corridor_flags: dict[int, bool],
    ) -> None:
        updated_ids: set[int] = set()
        for tracker_id, bbox, area in detections:
            center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
            stable_center = stable_centers[tracker_id]
            in_corridor = corridor_flags[tracker_id]
            observation = FrameObservation(
                tracker_id=tracker_id,
                frame_idx=frame_idx,
                time_sec=time_sec,
                bbox=bbox,
                center=center,
                stable_center=stable_center,
                area=area,
                in_corridor=in_corridor,
            )

            if tracker_id in self.active_tracklets:
                tracklet = self.active_tracklets[tracker_id]
                tracklet.observations.append(observation)
                tracklet.last_seen_frame = frame_idx
                tracklet.disappeared = 0
                record = self.vehicles[tracklet.vehicle_uid]
                record.observations.append(observation)
                record.end_time = time_sec
                record.end_bbox = bbox
            else:
                merge_vehicle = self._find_merge_candidate(observation)
                if merge_vehicle is not None:
                    tracklet = self._attach_vehicle(merge_vehicle, tracker_id, observation)
                else:
                    tracklet = self._create_vehicle(tracker_id, observation)
                self.active_tracklets[tracker_id] = tracklet
            updated_ids.add(tracker_id)

        for tracker_id in list(self.active_tracklets.keys()):
            if tracker_id not in updated_ids:
                tracklet = self.active_tracklets[tracker_id]
                tracklet.disappeared += 1
                if tracklet.disappeared > self.max_disappeared:
                    self._mark_tracklet_ended(tracklet)
                    self.active_tracklets.pop(tracker_id, None)

        self._finalize_due(time_sec)

    def _mark_tracklet_ended(self, tracklet: Tracklet) -> None:
        record = self.vehicles[tracklet.vehicle_uid]
        if record.observations:
            end_obs = record.observations[-1]
            self.recent_endings.append(
                RecentEnding(
                    vehicle_uid=record.vehicle_uid,
                    end_time=end_obs.time_sec,
                    end_point=end_obs.stable_center,
                    end_area=end_obs.area,
                )
            )
            self.pending_finalize[record.vehicle_uid] = PendingFinalize(
                vehicle_uid=record.vehicle_uid,
                due_time=end_obs.time_sec + self.merge_window_sec,
            )

    def _finalize_due(self, current_time: float) -> None:
        for vehicle_uid, pending in list(self.pending_finalize.items()):
            if current_time >= pending.due_time:
                record = self.vehicles.get(vehicle_uid)
                if record:
                    self._finalize_vehicle(record)
                self.pending_finalize.pop(vehicle_uid, None)

    def finalize_all(self) -> None:
        for tracklet in list(self.active_tracklets.values()):
            self._mark_tracklet_ended(tracklet)
        self.active_tracklets.clear()
        for vehicle_uid in list(self.pending_finalize.keys()):
            record = self.vehicles.get(vehicle_uid)
            if record:
                self._finalize_vehicle(record)
            self.pending_finalize.pop(vehicle_uid, None)

    def _finalize_vehicle(self, record: VehicleRecord) -> None:
        if record.direction is not None:
            return
        stable_points = [obs.stable_center for obs in record.observations]
        areas = [obs.area for obs in record.observations]
        in_corridor = [obs.in_corridor for obs in record.observations]
        if len(stable_points) < 2:
            record.direction = "OTHER"
            record.metrics = TrackletMetrics(0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        else:
            metrics = compute_metrics(stable_points, areas, in_corridor)
            record.metrics = metrics
            record.direction = classify_direction(
                metrics,
                stationary_thresh=self.stationary_dx_thresh,
                approach_area_ratio=self.approach_area_ratio,
                approach_dx_thresh=self.approach_dx_thresh,
                direction_dx_thresh=self.direction_dx_thresh,
                direction_dominance=self.direction_dominance,
                corridor_min_ratio=self.corridor_min_ratio,
            )
        self.finalized.append(record)

    def get_finalized(self) -> List[VehicleRecord]:
        return self.finalized

    def get_vehicle_records(self) -> Dict[int, VehicleRecord]:
        return self.vehicles

    def get_active_tracklets(self) -> Dict[int, Tracklet]:
        return self.active_tracklets
