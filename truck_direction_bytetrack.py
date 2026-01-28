from __future__ import annotations

import os
import time
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
CONF = 0.25                               # detection confidence
IOU = 0.45                                # NMS IoU
IMGSZ = 960                               # inference size (increase for accuracy, reduce for speed)
DEDUP_IOU = 0.7                           # IoU threshold for per-frame dedup before drawing/counting

MAX_DISAPPEARED = 30                      # frames to wait before finalizing a track
MIN_TOTAL_DX = 30                         # px: minimum net x movement to count direction (filters tiny jitter)
FILL_THRESH = 0.35                        # fraction of frame area to force finalize (close truck exit)

DEBUG_OVERLAY = True                      # show per-frame debug overlay
DEBUG_PRINT_EVERY = 30                    # print debug line every N frames
CLASS_FILTER_PATIENCE = 10                # frames of zero detections before disabling class filter
MAX_CENTROID_DIST = 80.0                  # max centroid distance for fallback tracker

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


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0.0, (box_a[2] - box_a[0])) * max(0.0, (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0])) * max(0.0, (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dedup_boxes(
    boxes: np.ndarray, confs: np.ndarray, ids: list[int] | None, iou_thresh: float
) -> tuple[np.ndarray, np.ndarray, list[int] | None]:
    if len(boxes) == 0:
        return boxes, confs, ids
    order = np.argsort(-confs)
    keep = []
    for idx in order:
        candidate = boxes[idx]
        if any(_iou(candidate, boxes[k]) > iou_thresh for k in keep):
            continue
        keep.append(int(idx))
    kept_boxes = boxes[keep]
    kept_confs = confs[keep]
    kept_ids = [ids[i] for i in keep] if ids is not None else None
    return kept_boxes, kept_confs, kept_ids


class CentroidTracker:
    def __init__(self, max_distance: float, max_disappeared: int) -> None:
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.next_id = 1
        self.objects: dict[int, tuple[float, float]] = {}
        self.disappeared: dict[int, int] = {}

    def update(self, boxes: np.ndarray) -> dict[int, np.ndarray]:
        if len(boxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.objects.pop(object_id, None)
                    self.disappeared.pop(object_id, None)
            return {}

        centroids = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0))
        if not self.objects:
            results = {}
            for centroid, box in zip(centroids, boxes):
                object_id = self.next_id
                self.next_id += 1
                self.objects[object_id] = tuple(centroid)
                self.disappeared[object_id] = 0
                results[object_id] = box
            return results

        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[oid] for oid in object_ids])

        distances = np.linalg.norm(object_centroids[:, None, :] - centroids[None, :, :], axis=2)
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()
        results: dict[int, np.ndarray] = {}

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if distances[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = tuple(centroids[col])
            self.disappeared[object_id] = 0
            results[object_id] = boxes[col]
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_ids))) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.objects.pop(object_id, None)
                self.disappeared.pop(object_id, None)

        unused_cols = set(range(len(boxes))) - used_cols
        for col in unused_cols:
            object_id = self.next_id
            self.next_id += 1
            self.objects[object_id] = tuple(centroids[col])
            self.disappeared[object_id] = 0
            results[object_id] = boxes[col]

        return results


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
    print(f"[INFO] class_id={class_id} class_name='{id2name[class_id]}' model.names={id2name}")

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
    first_seen: dict[int, int] = {}
    counted: set[int] = set()
    display_id_map: dict[int, int] = {}
    next_display_id = 0

    # Optional trail history (nice visuals / debug)
    track_hist: dict[int, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=30))
    centroid_tracker = CentroidTracker(max_distance=MAX_CENTROID_DIST, max_disappeared=MAX_DISAPPEARED)
    use_class_filter = True
    empty_class_filter_frames = 0

    left_to_right = 0
    right_to_left = 0

    csv_rows = []
    frame_idx = 0
    start_time = time.time()

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
                classes=[class_id] if use_class_filter else None,
                device=device,
                verbose=False,
            )

            res = results[0]
            annotated = frame.copy()
            raw_det_count = int(len(res.boxes)) if res.boxes is not None else 0
            raw_ids_present = bool(res.boxes is not None and res.boxes.id is not None)

            boxes_xyxy = np.empty((0, 4), dtype=np.float32)
            confs = np.empty((0,), dtype=np.float32)
            track_ids: list[int] | None = None

            if res.boxes is not None and len(res.boxes) > 0:
                boxes_xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                classes = res.boxes.cls.cpu().numpy().astype(int) if res.boxes.cls is not None else None
                mask = None
                if not use_class_filter and classes is not None:
                    mask = classes == class_id
                    boxes_xyxy = boxes_xyxy[mask]
                    confs = confs[mask]
                if len(boxes_xyxy) == 0:
                    empty_class_filter_frames += 1
                else:
                    empty_class_filter_frames = 0
                if empty_class_filter_frames >= CLASS_FILTER_PATIENCE and use_class_filter:
                    use_class_filter = False
                    print("[WARN] class filter yielded zero detections; disabling class filter and filtering manually.")
                if raw_ids_present:
                    track_ids = res.boxes.id.int().cpu().tolist()
                    if mask is not None:
                        track_ids = list(np.array(track_ids)[mask])
            else:
                empty_class_filter_frames += 1

            if track_ids is not None and len(track_ids) > 0 and len(boxes_xyxy) > 0:
                boxes_xyxy, confs, track_ids = _dedup_boxes(boxes_xyxy, confs, track_ids, DEDUP_IOU)
                detections = list(zip(boxes_xyxy, track_ids, confs))
            else:
                pred = model.predict(
                    frame,
                    conf=CONF,
                    iou=IOU,
                    imgsz=IMGSZ,
                    classes=[class_id] if use_class_filter else None,
                    device=device,
                    verbose=False,
                )
                pred_boxes = pred[0].boxes
                if pred_boxes is not None and len(pred_boxes) > 0:
                    boxes_xyxy = pred_boxes.xyxy.cpu().numpy()
                    confs = pred_boxes.conf.cpu().numpy()
                    classes = pred_boxes.cls.cpu().numpy().astype(int) if pred_boxes.cls is not None else None
                    if not use_class_filter and classes is not None:
                        mask = classes == class_id
                        boxes_xyxy = boxes_xyxy[mask]
                        confs = confs[mask]
                boxes_xyxy, confs, _ = _dedup_boxes(boxes_xyxy, confs, None, DEDUP_IOU)
                fallback_tracks = centroid_tracker.update(boxes_xyxy)
                detections = [(box, tid, 1.0) for tid, box in fallback_tracks.items()]

            det_count = len(detections)
            id_count = len({tid for _, tid, _ in detections})

            if DEBUG_PRINT_EVERY and frame_idx % DEBUG_PRINT_EVERY == 0:
                print(
                    "[DEBUG] "
                    f"frame={frame_idx} raw_det={raw_det_count} raw_ids_present={raw_ids_present} "
                    f"det={det_count} id_count={id_count}"
                )

            for (x1, y1, x2, y2), tid, conf in detections:
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)

                if tid not in first_x:
                    first_x[tid] = cx
                    first_seen[tid] = frame_idx
                last_x[tid] = cx
                last_seen[tid] = frame_idx
                track_hist[tid].append((cx, cy))

                if tid not in display_id_map:
                    display_id_map[tid] = next_display_id
                    next_display_id += 1
                display_id = display_id_map[tid]
                track_len = frame_idx - first_seen.get(tid, frame_idx) + 1
                dx = cx - first_x.get(tid, cx)

                color = (40, 40, 40)
                thickness = 1
                if track_len >= 5 and abs(dx) > (MIN_TOTAL_DX / 2):
                    if dx > 0:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                    thickness = THICKNESS

                if DRAW_BOXES:
                    cv2.rectangle(
                        annotated,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        thickness,
                    )

                if DRAW_IDS:
                    cv2.putText(
                        annotated,
                        f"ID{display_id:02d}",
                        (int(x1), max(0, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SCALE,
                        color,
                        max(1, thickness),
                        cv2.LINE_AA,
                    )

                # Optional: draw trajectory
                pts = np.array(track_hist[tid], dtype=np.int32)
                if len(pts) >= 2:
                    cv2.polylines(annotated, [pts.reshape(-1, 1, 2)], False, (255, 255, 255), 2)

                if tid not in counted:
                    box_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                    if (box_area / float(w * h)) >= FILL_THRESH:
                        dx_total = float(last_x.get(tid, cx) - first_x.get(tid, cx))
                        if abs(dx_total) >= MIN_TOTAL_DX:
                            if dx_total > 0:
                                left_to_right += 1
                            else:
                                right_to_left += 1
                            counted.add(tid)

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

            elapsed = max(1e-6, time.time() - start_time)
            fps_proc = frame_idx / elapsed if frame_idx > 0 else 0.0

            # Overlay counts
            cv2.putText(
                annotated,
                f"L2R: {left_to_right}  R2L: {right_to_left}  FPS: {fps_proc:.1f}  Frame: {frame_idx}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if DEBUG_OVERLAY:
                cv2.putText(
                    annotated,
                    f"det={det_count} id_count={id_count} raw_det={raw_det_count} ids_present={raw_ids_present}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    1,
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
