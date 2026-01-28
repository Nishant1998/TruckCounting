from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Input/output paths
    weights_path: Path = Path("best.pt")  # YOLOv8 weights with a truck class
    input_video: Path = Path("video.mp4")  # handheld portrait MP4 input
    output_dir: Path = Path("outputs")  # base output directory
    output_video: Path = Path("outputs/annotated.mp4")  # annotated video output
    events_csv: Path = Path("outputs/events.csv")  # per-vehicle event CSV
    events_json: Path = Path("outputs/events.json")  # per-vehicle event JSON
    tracklets_csv: Path = Path("outputs/tracklets.csv")  # per-frame tracklet CSV
    per_second_csv: Path = Path("outputs/per_second_counts.csv")  # per-second counts CSV

    # Runtime options
    device: str = "cuda:0"  # preferred device, auto-overridden if CUDA unavailable
    conf_thresh: float = 0.25  # YOLOv8 confidence threshold
    enable_motion_compensation: bool = True  # toggle ego-motion compensation on/off
    enable_tracklet_merging: bool = True  # toggle tracklet fragmentation merging on/off

    # Tracklet lifecycle + merging
    max_disappeared: int = 15  # frames before a tracklet is considered ended
    merge_window_sec: float = 1.0  # max time gap to merge fragmented tracklets
    merge_distance_px: float = 60.0  # max stabilized distance to merge
    merge_area_ratio_tol: float = 0.6  # area ratio tolerance for merging

    # Classification thresholds
    stationary_dx_thresh: float = 3.0  # median |dx| below this => stationary
    direction_dx_thresh: float = 60.0  # min stabilized dx for L2R/R2L
    direction_dominance: float = 1.2  # |dx| must dominate |dy| by this ratio
    approach_area_ratio: float = 2.0  # large area change => approach gate
    approach_dx_thresh: float = 40.0  # small dx with big area => IGNORE_APPROACH
    corridor_min_ratio: float = 0.4  # min ratio of frames inside corridor

    # Corridor ROI (in normalized image coordinates)
    corridor_x_min: float = 0.25  # left bound (fraction of width)
    corridor_x_max: float = 0.75  # right bound (fraction of width)
    corridor_y_min: float = 0.20  # top bound (fraction of height)
    corridor_y_max: float = 0.95  # bottom bound (fraction of height)

    # Motion estimation settings
    orb_features: int = 600  # ORB feature count for ego-motion
    match_ratio: float = 0.75  # Lowe ratio for BFMatcher
    motion_min_matches: int = 10  # minimum matches to accept motion estimate

    # Model metadata
    class_name: str = "truck"  # detection class name to keep

    # Rendering controls
    font_scale_ratio: float = 0.03  # text size relative to frame height
    font_thickness: int = 2  # text stroke thickness
    box_thickness: int = 3  # thick box for L2R/R2L
    thin_box_thickness: int = 1  # thin box for OTHER/IGNORE

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for path in [self.output_video, self.events_csv, self.events_json, self.tracklets_csv, self.per_second_csv]:
            path.parent.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = Config()
