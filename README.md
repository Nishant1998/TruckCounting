# TruckCounting (YOLOv8 + ByteTrack)

Lightweight pipeline to count haulage trucks moving **LEFT→RIGHT** vs **RIGHT→LEFT** in handheld portrait videos using custom YOLOv8 weights.

## Project Tree
```
./
├── requirements.txt
├── README.md
├── src/
│   ├── classify.py
│   ├── config.py
│   ├── io_video.py
│   ├── motion.py
│   ├── render.py
│   ├── report.py
│   ├── tracking.py
│   └── tracklets.py
├── main.py
└── tests/
    └── test_merge.py
```

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start (Smoke Test)
```bash
python main.py
```

By default, the pipeline expects:
- `best.pt` in the repo root
- `video.mp4` in the repo root

It writes outputs to `outputs/`:
- `annotated.mp4`
- `events.csv`, `events.json`
- `tracklets.csv`
- `per_second_counts.csv`

## Configuration
All knobs live in `src/config.py` (dataclass `Config`). Edit paths and thresholds there.
Key parameters to tune:
- `merge_window_sec`, `merge_distance_px`, `merge_area_ratio_tol`
- `stationary_dx_thresh`, `direction_dx_thresh`, `direction_dominance`
- `corridor_min_ratio`, `corridor_*` bounds
- `approach_area_ratio`, `approach_dx_thresh`

## Notes on Direction
Direction is computed **only when a vehicle tracklet is finalized**, using stabilized centers from the motion compensation module. This avoids per-frame overcounting and camera shake.

## Optional Merge Logic Check
```bash
python -m tests.test_merge
```
