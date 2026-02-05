#!/usr/bin/env python3
"""
Batch process locust videos under Locust/all_vids/<condition>/<date>/.
For each video file directly inside a date folder, creates an output folder
named after the video (stem) and writes an annotated video and CSV there.

Output folder example:
  /.../Locust/all_vids/Off_1s_LOOL/12.02.2025/
    Off_1s_LOOL_12022025_Testing_2-OCT_L1_Trial_1_Recording/
      Off_1s_LOOL_12022025_Testing_2-OCT_L1_Trial_1_Recording_palps_annotated_30fps.mp4
      Off_1s_LOOL_12022025_Testing_2-OCT_L1_Trial_1_Recording_palps_tracks.csv
"""

import argparse
import logging
import os
import re
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

# -----------------------------
# Performance / CUDA settings
# -----------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# -----------------------------
# User defaults (override via CLI)
# -----------------------------
DEFAULT_ALL_VIDS_DIR = Path(
    "/home/ramanlab/Documents/cole/VSCode/RamanLab-Locust-Behavior/Locust/all_vids"
)
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "YOLO_MODEL_PATH",
        "/home/ramanlab/Documents/cole/VSCode/RamanLab-Locust-Behavior/model/best.pt",
    )
)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}
DATE_DIR_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")

# -----------------------------
# Odor mapping (from filename tokens)
# -----------------------------
ODOR_MAP = {
    "LOOL": "Linalool",
    "2-OCT": "2-octanol",
    "2OCT": "2-octanol",
    "2-OCTANOL": "2-octanol",
    "BZA": "Benzaldehyde",
    "HEX": "Hexanol",
    "IAA": "Isoamyl Acetate",
    "GER": "Geraniol",
    "GERMIOL": "Geraniol",
    "CIT": "Citral",
}

# -----------------------------
# Tracking / output config
# -----------------------------
TARGET_CLASS_ID = 1
NUM_TARGETS = 2
OUTPUT_FPS = 30.0

# Odor on window (30 fps): frames 300-420 inclusive => 10s to 14s
ODOR_ON_START_FRAME = int(10 * OUTPUT_FPS)  # 300
ODOR_ON_END_FRAME = int(14 * OUTPUT_FPS)    # 420

# Temporal / spatial enhancement
CONF_THRES = 0.40
IOU_MATCH_THRES = 0.70
MAX_AGE = 15
EMA_ALPHA = 0.20

FLOW_ENABLE = True
FLOW_SKIP_EDGE = 10
FLOW_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)


# -----------------------------
# Video writer helpers
# -----------------------------
def create_video_writer(output_path: str, width: int, height: int, fps: float):
    """
    Try to create an OpenCV VideoWriter for MP4.
    If fails, fallback to AVI + XVID.
    Returns: writer object, final output path
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if writer.isOpened():
        return writer, output_path

    avi_path = output_path.replace(".mp4", ".avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {output_path} or fallback {avi_path}")

    print(f"[INFO] OpenCV MP4 writer failed, using AVI fallback: {avi_path}")
    return writer, avi_path


def convert_avi_to_mp4(avi_path: str, mp4_path: str):
    """Convert AVI to MP4 using ffmpeg"""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        avi_path,
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "fast",
        mp4_path,
    ]
    print(f"[INFO] Converting {avi_path} -> {mp4_path} via ffmpeg")
    subprocess.run(cmd, check=True)


# -----------------------------
# Geometry helpers
# -----------------------------
def order_corners(corners):
    """Return 4 corners in clockwise order."""
    pts = np.array(corners, dtype=np.float32)
    cx, cy = np.mean(pts, axis=0)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    pts_sorted = pts[np.argsort(angles)]
    return pts_sorted.tolist()


def xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = b
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return np.array([cx, cy, w, h], dtype=np.float32)


def cxcywh_to_xyxy(s):
    cx, cy, w, h = s
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def iou(a, b):
    N = a.shape[0]
    M = b.shape[0]
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)

    ax1, ay1, ax2, ay2 = a[:, 0][:, None], a[:, 1][:, None], a[:, 2][:, None], a[:, 3][:, None]
    bx1, by1, bx2, by2 = b[:, 0][None, :], b[:, 1][None, :], b[:, 2][None, :], b[:, 3][None, :]

    inter_w = np.maximum(0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    inter_h = np.maximum(0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return (inter / union).astype(np.float32)


# -----------------------------
# Kalman tracker
# -----------------------------
class KalmanBBox:
    def __init__(self):
        self.x = np.zeros((8, 1), dtype=np.float32)
        self.P = np.eye(8, dtype=np.float32) * 10.0

        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        self.H = np.zeros((4, 8), dtype=np.float32)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        self.Q = np.eye(8, dtype=np.float32) * 0.02
        self.R = np.eye(4, dtype=np.float32) * 1.0

    def init(self, cxcywh):
        self.x[:4, 0] = cxcywh
        self.x[4:, 0] = 0.0
        self.P = np.eye(8, dtype=np.float32) * 10.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4, 0].copy()

    def update(self, z):
        z = z.reshape(4, 1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(8, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P


class Track:
    _next_id = 1

    def __init__(self, cxcywh, score, corners: Optional[List[List[float]]] = None):
        self.id = Track._next_id
        Track._next_id += 1

        self.kf = KalmanBBox()
        self.kf.init(cxcywh)

        self.score = float(score)
        self.box_xyxy = cxcywh_to_xyxy(cxcywh)

        self.time_since_update = 0
        self.hits = 1
        self.history = deque(maxlen=30)

        self.corners = corners

    def predict(self):
        pred = self.kf.predict()
        box = cxcywh_to_xyxy(pred)
        self.box_xyxy = box
        self.history.append(box.copy())
        self.time_since_update += 1
        return box

    def correct(self, cxcywh, score, corners: Optional[List[List[float]]] = None):
        self.kf.update(cxcywh)
        box = cxcywh_to_xyxy(self.kf.x[:4, 0])

        self.box_xyxy = (1 - EMA_ALPHA) * box + EMA_ALPHA * self.box_xyxy

        self.score = float(score)
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.box_xyxy.copy())

        if corners is not None:
            self.corners = corners


class MultiObjectSingleClassTracker:
    def __init__(self, iou_thres=0.25, max_age=15):
        self.iou_thres = iou_thres
        self.max_age = max_age
        self.tracks: List[Track] = []

    def step(self, det_xyxy: np.ndarray, det_scores: np.ndarray, det_corners: Optional[List[Optional[List[List[float]]]]] = None) -> List[Track]:
        if det_corners is None:
            det_corners = [None] * len(det_xyxy)

        preds = [t.predict() for t in self.tracks]

        assigned_tr, assigned_det = set(), set()
        if len(self.tracks) and len(det_xyxy):
            M = iou(np.stack(preds), det_xyxy)
            while True:
                i, j = np.unravel_index(np.argmax(M), M.shape)
                if M[i, j] < self.iou_thres:
                    break
                if i in assigned_tr or j in assigned_det:
                    M[i, j] = -1
                    continue

                self.tracks[i].correct(
                    xyxy_to_cxcywh(det_xyxy[j]),
                    det_scores[j],
                    corners=det_corners[j],
                )
                assigned_tr.add(i)
                assigned_det.add(j)
                M[i, :] = -1
                M[:, j] = -1

        for j in range(len(det_xyxy)):
            if j in assigned_det:
                continue
            self.tracks.append(
                Track(xyxy_to_cxcywh(det_xyxy[j]), det_scores[j], corners=det_corners[j])
            )

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        self.tracks.sort(key=lambda t: (t.time_since_update, -t.hits, -t.score, t.id))
        return self.tracks


def flow_nudge(prev_gray, gray, box_xyxy):
    if prev_gray is None:
        return box_xyxy

    x1, y1, x2, y2 = box_xyxy.astype(int)
    x1 = max(FLOW_SKIP_EDGE, x1)
    y1 = max(FLOW_SKIP_EDGE, y1)
    x2 = min(gray.shape[1] - FLOW_SKIP_EDGE, x2)
    y2 = min(gray.shape[0] - FLOW_SKIP_EDGE, y2)
    if x2 <= x1 or y2 <= y1:
        return box_xyxy

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray[y1:y2, x1:x2],
        gray[y1:y2, x1:x2],
        None,
        **FLOW_PARAMS,
    )
    dx = np.median(flow[..., 0])
    dy = np.median(flow[..., 1])

    nudged = box_xyxy.copy().astype(np.float32)
    nudged[0::2] += dx
    nudged[1::2] += dy
    return nudged


# -----------------------------
# Odor parsing helpers
# -----------------------------
def _normalize_odor_token(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    return token.strip().upper()


def _find_odor_in_tokens(tokens: List[str]) -> Optional[str]:
    for t in reversed(tokens):
        code = _normalize_odor_token(t)
        if code in ODOR_MAP:
            return code
    return None


def infer_odor_sent_from_filename(stem: str) -> Optional[str]:
    """
    Returns the odor sent (full name), based on filename tokens.

    For Testing videos, the token after "Testing" is the odor sent.
    For Training videos, odor sent is inferred from the condition tokens.
    """
    tokens = stem.split("_")
    date_idx = next((i for i, t in enumerate(tokens) if t.isdigit() and len(t) == 8), None)

    # Fallback: just find any known odor token
    if date_idx is None:
        code = _find_odor_in_tokens(tokens)
        return ODOR_MAP.get(code, code)

    condition_tokens = tokens[:date_idx]
    phase = tokens[date_idx + 1] if date_idx + 1 < len(tokens) else ""
    trained_code = _find_odor_in_tokens(condition_tokens)

    sent_code = None
    if phase.lower() == "testing":
        candidate = tokens[date_idx + 2] if date_idx + 2 < len(tokens) else None
        code = _normalize_odor_token(candidate)
        if code in ODOR_MAP:
            sent_code = code

    if sent_code is None:
        sent_code = trained_code

    return ODOR_MAP.get(sent_code, sent_code)


# -----------------------------
# Frame processing
# -----------------------------
logging.getLogger("ultralytics").setLevel(logging.WARNING)


def process_frame(frame, frame_number, current_timestamp, tracker, prev_gray, model):
    r = model.predict(source=frame, conf=CONF_THRES, verbose=False)[0]

    det_boxes = []
    det_scores = []
    det_corners = []

    if hasattr(r, "obb") and r.obb is not None:
        xyxyxyxy = r.obb.xyxyxyxy.cpu().numpy()
        cls_arr = r.obb.cls.cpu().numpy().astype(int)
        conf_arr = (
            r.obb.conf.cpu().numpy().astype(np.float32)
            if hasattr(r.obb, "conf") and r.obb.conf is not None
            else np.ones_like(cls_arr, dtype=np.float32)
        )

        for i, (c, s) in enumerate(zip(cls_arr, conf_arr)):
            if c != TARGET_CLASS_ID:
                continue
            corners = xyxyxyxy[i].reshape(4, 2)
            x1, y1 = float(corners[:, 0].min()), float(corners[:, 1].min())
            x2, y2 = float(corners[:, 0].max()), float(corners[:, 1].max())
            det_boxes.append([x1, y1, x2, y2])
            det_scores.append(float(s))
            det_corners.append(order_corners(corners))
    else:
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls_arr = r.boxes.cls.cpu().numpy().astype(int)
            conf_arr = r.boxes.conf.cpu().numpy().astype(np.float32)
            for b, c, s in zip(xyxy, cls_arr, conf_arr):
                if c != TARGET_CLASS_ID:
                    continue
                det_boxes.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                det_scores.append(float(s))
                det_corners.append(None)

    det_xyxy = (
        np.array(det_boxes, dtype=np.float32)
        if len(det_boxes)
        else np.zeros((0, 4), dtype=np.float32)
    )
    det_scores = (
        np.array(det_scores, dtype=np.float32)
        if len(det_scores)
        else np.zeros((0,), dtype=np.float32)
    )

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    tracks = tracker.step(det_xyxy, det_scores, det_corners)
    selected = tracks[:NUM_TARGETS]

    if FLOW_ENABLE and prev_gray is not None:
        for t in selected:
            if t.time_since_update > 0:
                nudged = flow_nudge(prev_gray, gray, t.box_xyxy)
                t.box_xyxy = nudged

    palps = []
    for idx in range(NUM_TARGETS):
        if idx < len(selected):
            t = selected[idx]
            box = t.box_xyxy.astype(np.float32)
            cx, cy, bw, bh = xyxy_to_cxcywh(box)
            palps.append(
                {
                    "track_id": t.id,
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                    "cx": float(cx),
                    "cy": float(cy),
                    "corners": (t.corners if t.corners is not None else np.nan),
                    "age": int(t.time_since_update),
                    "score": float(t.score),
                }
            )

            if t.corners is not None and isinstance(t.corners, list) and len(t.corners) == 4:
                pts = np.array(t.corners, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
            else:
                cv2.rectangle(
                    frame,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 255, 255),
                    2,
                )

            cv2.circle(frame, (int(cx), int(cy)), 1, (0, 255, 255), -1)

            cv2.putText(
                frame,
                f"palp#{idx + 1} id={t.id} age={t.time_since_update}",
                (max(0, int(box[0])), max(20, int(box[1]) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 255, 255),
                1,
            )
        else:
            palps.append(
                {
                    "track_id": np.nan,
                    "x1": np.nan,
                    "y1": np.nan,
                    "x2": np.nan,
                    "y2": np.nan,
                    "cx": np.nan,
                    "cy": np.nan,
                    "corners": np.nan,
                    "age": np.nan,
                    "score": np.nan,
                }
            )

    distance = np.nan
    if not (np.isnan(palps[0]["cx"]) or np.isnan(palps[1]["cx"])):
        p0 = (int(palps[0]["cx"]), int(palps[0]["cy"]))
        p1 = (int(palps[1]["cx"]), int(palps[1]["cy"]))
        cv2.line(frame, p0, p1, (0, 255, 0), 2)
        distance = float(
            np.hypot(
                palps[0]["cx"] - palps[1]["cx"],
                palps[0]["cy"] - palps[1]["cy"],
            )
        )
        cv2.putText(
            frame,
            f"dist={distance:.2f}px",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    row = {
        "frame": frame_number,
        "timestamp": current_timestamp,
        "track_id_palp1": palps[0]["track_id"],
        "x1_palp1": palps[0]["x1"],
        "y1_palp1": palps[0]["y1"],
        "x2_palp1": palps[0]["x2"],
        "y2_palp1": palps[0]["y2"],
        "cx_palp1": palps[0]["cx"],
        "cy_palp1": palps[0]["cy"],
        "corners_palp1": str(palps[0]["corners"]),
        "age_palp1": palps[0]["age"],
        "score_palp1": palps[0]["score"],
        "track_id_palp2": palps[1]["track_id"],
        "x1_palp2": palps[1]["x1"],
        "y1_palp2": palps[1]["y1"],
        "x2_palp2": palps[1]["x2"],
        "y2_palp2": palps[1]["y2"],
        "cx_palp2": palps[1]["cx"],
        "cy_palp2": palps[1]["cy"],
        "corners_palp2": str(palps[1]["corners"]),
        "age_palp2": palps[1]["age"],
        "score_palp2": palps[1]["score"],
        "distance_palp1_palp2_px": distance,
    }

    return frame, row, gray


# -----------------------------
# Main processing
# -----------------------------
def find_date_dirs(all_vids_dir: Path) -> List[Path]:
    date_dirs: List[Path] = []
    for condition_dir in sorted(all_vids_dir.iterdir()):
        if not condition_dir.is_dir():
            continue
        for date_dir in sorted(condition_dir.iterdir()):
            if not date_dir.is_dir():
                continue
            if DATE_DIR_RE.match(date_dir.name):
                date_dirs.append(date_dir)
    return date_dirs


def is_odor_on(frame_idx: int) -> bool:
    return ODOR_ON_START_FRAME <= frame_idx <= ODOR_ON_END_FRAME


def process_video(video_path: Path, model, overwrite: bool) -> None:
    video_name = video_path.name
    video_stem = video_path.stem

    output_dir = video_path.parent / video_stem
    output_video_path = output_dir / f"{video_stem}_palps_annotated_30fps.mp4"
    out_csv_path = output_dir / f"{video_stem}_palps_tracks.csv"

    if output_dir.exists() and not overwrite:
        print(f"[SKIP] {video_path} (output folder exists)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    odor_sent = infer_odor_sent_from_filename(video_stem)
    if odor_sent is None:
        odor_sent = "Unknown"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    writer = None
    writer_path = None

    all_rows = []
    tracker = MultiObjectSingleClassTracker(iou_thres=IOU_MATCH_THRES, max_age=MAX_AGE)

    start_time = time.time()
    frame_count = 0
    prev_gray = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if writer is None:
            h, w = frame.shape[:2]
            writer, writer_path = create_video_writer(str(output_video_path), w, h, OUTPUT_FPS)

        current_timestamp = frame_count / OUTPUT_FPS

        frame, row, prev_gray = process_frame(
            frame=frame,
            frame_number=frame_count,
            current_timestamp=current_timestamp,
            tracker=tracker,
            prev_gray=prev_gray,
            model=model,
        )

        row["odor_sent"] = odor_sent
        row["odor_on"] = is_odor_on(frame_count)

        writer.write(frame)
        all_rows.append(row)
        frame_count += 1

    cap.release()
    if writer is not None:
        writer.release()

    if writer_path and writer_path.endswith(".avi"):
        try:
            convert_avi_to_mp4(writer_path, str(output_video_path))
            os.remove(writer_path)
        except Exception as exc:
            print(f"[WARN] AVI conversion failed: {exc}")

    pd.DataFrame(all_rows).to_csv(out_csv_path, index=False)

    elapsed_time = time.time() - start_time
    print(f"[DONE] {video_name} -> {output_dir} ({frame_count} frames, {elapsed_time:.2f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch process locust palps videos.")
    parser.add_argument(
        "--all-vids-dir",
        type=Path,
        default=DEFAULT_ALL_VIDS_DIR,
        help="Base directory containing condition/date folders (default: Locust/all_vids).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to YOLO model weights.",
    )
    parser.add_argument(
        "--date-dir",
        type=Path,
        action="append",
        default=[],
        help="Process only this date directory (can be used multiple times).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs if the output folder already exists.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU execution if CUDA is unavailable.",
    )

    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model path not found: {args.model}. Use --model or set YOLO_MODEL_PATH."
        )

    if torch.cuda.is_available():
        device = "cuda"
        print("CUDA is available; running on GPU.")
    else:
        if not args.allow_cpu:
            raise RuntimeError("CUDA is not available. Use --allow-cpu to run on CPU.")
        device = "cpu"
        print("CUDA not available; running on CPU.")

    model = YOLO(str(args.model))
    model.to(device)

    if args.date_dir:
        date_dirs = [d for d in args.date_dir if d.exists()]
        missing = [d for d in args.date_dir if not d.exists()]
        for d in missing:
            print(f"[WARN] Date dir not found, skipping: {d}")
    else:
        if not args.all_vids_dir.exists():
            raise FileNotFoundError(f"all_vids dir not found: {args.all_vids_dir}")
        date_dirs = find_date_dirs(args.all_vids_dir)

    if not date_dirs:
        print("No date directories found. Nothing to do.")
        return

    total_videos = 0
    for date_dir in date_dirs:
        for file_path in sorted(date_dir.iterdir()):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            if file_path.stem.endswith("_palps_annotated_30fps"):
                continue
            total_videos += 1
            process_video(file_path, model=model, overwrite=args.overwrite)

    print(f"Total videos processed: {total_videos}")


if __name__ == "__main__":
    main()
