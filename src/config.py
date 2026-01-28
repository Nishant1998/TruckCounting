from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    weights_path: Path = Path("best.pt")
    input_video: Path = Path("video.mp4")
    output_dir: Path = Path("outputs")
    output_video: Path = Path("outputs/annotated.mp4")
    events_csv: Path = Path("outputs/events.csv")
    events_json: Path = Path("outputs/events.json")
    tracklets_csv: Path = Path("outputs/tracklets.csv")
    per_second_csv: Path = Path("outputs/per_second_counts.csv")

    device: str = "cuda:0"
    conf_thresh: float = 0.25

    max_disappeared: int = 15
    merge_window_sec: float = 1.0
    merge_distance_px: float = 60.0
    merge_area_ratio_tol: float = 0.6

    stationary_dx_thresh: float = 3.0
    direction_dx_thresh: float = 60.0
    direction_dominance: float = 1.2
    approach_area_ratio: float = 2.0
    approach_dx_thresh: float = 40.0
    corridor_min_ratio: float = 0.4

    corridor_x_min: float = 0.25
    corridor_x_max: float = 0.75
    corridor_y_min: float = 0.20
    corridor_y_max: float = 0.95

    orb_features: int = 600
    match_ratio: float = 0.75
    motion_min_matches: int = 10

    tracklet_history: int = 2000

    class_name: str = "truck"

    font_scale_ratio: float = 0.03
    font_thickness: int = 2
    box_thickness: int = 3
    thin_box_thickness: int = 1

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for path in [self.output_video, self.events_csv, self.events_json, self.tracklets_csv, self.per_second_csv]:
            path.parent.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = Config()
