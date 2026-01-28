from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    tracker_id: int
    bbox: tuple[float, float, float, float]
    conf: float


class TruckTracker:
    def __init__(self, weights_path: str, device: str, class_name: str = "truck", conf: float = 0.25) -> None:
        self.model = YOLO(weights_path)
        self.device = device
        self.conf = conf
        self.class_name = class_name
        self.truck_class_id = self._resolve_truck_class_id()

    def _resolve_truck_class_id(self) -> int:
        for idx, name in self.model.names.items():
            if str(name).lower() == self.class_name.lower():
                return int(idx)
        raise ValueError(f"Class '{self.class_name}' not found in model names: {self.model.names}")

    def track(self, frame: np.ndarray) -> Iterable[Detection]:
        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.conf,
            classes=[self.truck_class_id],
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return detections

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        for tracker_id, bbox, conf in zip(ids, xyxy, confs):
            detections.append(
                Detection(
                    tracker_id=int(tracker_id),
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    conf=float(conf),
                )
            )
        return detections
