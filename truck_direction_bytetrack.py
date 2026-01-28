from __future__ import annotations

import os
from collections import defaultdict, deque

import cv2
import numpy as np
from tqdm import tqdm
# (11, 14)
# ---------- CONFIG (EDIT THESE) ----------
VIDEO_PATH = r"video.mp4"                 # input video path
WEIGHTS = r"best.pt"                      # your YOLOv8 model .pt
CLASS_NAME = "truck"                      # class name exactly as in your model.names
OUTPUT_VIDEO = r"annotated.mp4"           # output annotated video
OUTPUT_CSV = r"counts.csv"                # per-frame counts log (optional)

TRACKER_YAML = "bytetrack.yaml"           # Ultralytics tracker config
CONF = 0.5                               # detection confidence
IOU = 0.5                                 # NMS IoU
IMGSZ = 960                               # inference size (increase for accuracy, reduce for speed)

MAX_DISAPPEARED = 30                      # frames to wait before finalizing a track
MIN_TOTAL_DX = 30                         # px: minimum net x movement to count direction (filters tiny jitter)

DRAW_BOXES = True
DRAW_IDS = True
FONT_SCALE = 0.8
THICKNESS = 2
# ----------------------------------------


def _get_device() -> str:
    # Ultralytics will import torch internally, but we keep this robust.
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _names_map(model_names) -> dict[int, str]:
    # model.names can be dict or list
    if isinstance(model_names, dict):
        return {int(k): str(v) for k, v in model_names.items()}
    return {i: str(v) for i, v in enumerate(model_names)}


def main() -> None:
    # Lazy import YOLO (ensures ultralytics is installed)
    from ultralytics import YOLO

    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"VIDEO_PATH not found: {VIDEO_PATH}")
    if not os.path.exists(WEIGHTS):
        raise FileNotFoundError(f"WEIGHTS not found: {WEIGHTS}")

    device = _get_device()
    print(f"[INFO] device = {device}  (GPU used only if CUDA PyTorch is installed and available)")

    model = YOLO(WEIGHTS)
    id2name = _names_map(model.names)

    # Find class id from CLASS_NAME (case-insensitive)
    class_id = None
    for cid, cname in id2name.items():
        if cname.strip().lower() == CLASS_NAME.strip().lower():
            class_id = int(cid)
            break
    if class_id is None:
        raise ValueError(
            f"CLASS_NAME='{CLASS_NAME}' not found in model.names.\n"
            f"Available classes: {list(id2name.values())}"
        )

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None

    print("[INFO] starting video streams...")
    print(f"(W, H) = {(w, h)}")
    print(f"Total Frame = {total}")
    print(f"FPS = {fps}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {OUTPUT_VIDEO}")

    # Track bookkeeping
    first_x: dict[int, float] = {}
    last_x: dict[int, float] = {}
    last_seen: dict[int, int] = {}
    counted: set[int] = set()

    # Optional trail history (nice visuals / debug)
    track_hist: dict[int, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=30))

    left_to_right = 0
    right_to_left = 0

    csv_rows = []
    frame_idx = 0

    pbar = tqdm(total=total, desc="Processing", unit="frame") if total else tqdm(desc="Processing", unit="frame")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # YOLOv8 tracking with ByteTrack, persist tracks across frames
            # We filter by classes=[class_id] so ONLY the desired class is tracked.
            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_YAML,
                conf=CONF,
                iou=IOU,
                imgsz=IMGSZ,
                classes=[class_id],
                device=device,
                verbose=False,
            )

            res = results[0]
            annotated = frame.copy()

            # Extract tracked boxes
            # res.boxes.id exists when tracker returns IDs
            if res.boxes is not None and getattr(res.boxes, "is_track", False) and res.boxes.id is not None:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                track_ids = res.boxes.id.int().cpu().tolist()

                for (x1, y1, x2, y2), tid in zip(boxes_xyxy, track_ids):
                    cx = float((x1 + x2) / 2.0)
                    cy = float((y1 + y2) / 2.0)

                    if tid not in first_x:
                        first_x[tid] = cx
                    last_x[tid] = cx
                    last_seen[tid] = frame_idx
                    track_hist[tid].append((cx, cy))

                    if DRAW_BOXES:
                        cv2.rectangle(
                            annotated,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            THICKNESS,
                        )

                    if DRAW_IDS:
                        cv2.putText(
                            annotated,
                            f"ID {tid}",
                            (int(x1), max(0, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            FONT_SCALE,
                            (0, 255, 0),
                            THICKNESS,
                            cv2.LINE_AA,
                        )

                    # Optional: draw trajectory
                    pts = np.array(track_hist[tid], dtype=np.int32)
                    if len(pts) >= 2:
                        cv2.polylines(annotated, [pts.reshape(-1, 1, 2)], False, (255, 255, 255), 2)

            # Finalize tracks that disappeared
            to_finalize = []
            for tid, seen_at in list(last_seen.items()):
                if tid in counted:
                    continue
                if frame_idx - seen_at > MAX_DISAPPEARED:
                    to_finalize.append(tid)

            for tid in to_finalize:
                dx = float(last_x.get(tid, 0.0) - first_x.get(tid, 0.0))
                if abs(dx) >= MIN_TOTAL_DX:
                    if dx > 0:
                        left_to_right += 1
                    else:
                        right_to_left += 1
                counted.add(tid)

            # Overlay counts
            cv2.putText(
                annotated,
                f"{CLASS_NAME}  L->R: {left_to_right}   R->L: {right_to_left}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

            writer.write(annotated)

            # CSV per-frame log (optional)
            csv_rows.append((frame_idx, left_to_right, right_to_left))

            frame_idx += 1
            pbar.update(1)

    finally:
        pbar.close()
        cap.release()
        writer.release()

    # Finalize any remaining uncounted tracks at end of video
    for tid in list(last_seen.keys()):
        if tid in counted:
            continue
        dx = float(last_x.get(tid, 0.0) - first_x.get(tid, 0.0))
        if abs(dx) >= MIN_TOTAL_DX:
            if dx > 0:
                left_to_right += 1
            else:
                right_to_left += 1
        counted.add(tid)

    # Rewrite last line counts to reflect final totals (optional)
    if csv_rows:
        csv_rows[-1] = (csv_rows[-1][0], left_to_right, right_to_left)

    # Save CSV
    try:
        import csv

        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["frame_idx", "left_to_right", "right_to_left"])
            wcsv.writerows(csv_rows)
        print(f"[OK] CSV saved: {OUTPUT_CSV}")
    except Exception as e:
        print(f"[WARN] CSV not saved ({e})")

    print(f"[OK] Video saved: {OUTPUT_VIDEO}")
    print(f"[DONE] Final counts for '{CLASS_NAME}':  L->R={left_to_right}  R->L={right_to_left}")


if __name__ == "__main__":
    main()
