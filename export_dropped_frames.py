#!/usr/bin/env python3
"""
Export frames where tracking is invalid or below the minimum distance threshold.

Criteria:
  - distance_palp1_palp2_px_raw is NaN, OR
  - distance_palp1_palp2_px_raw < --min-px

Testing-only:
  - Only files with "testing" in the name and NOT "training".
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


DEFAULT_ALL_VIDS_DIR = Path(
    "/home/ramanlab/Documents/cole/VSCode/RamanLab-Locust-Behavior/Locust/all_vids"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/ramanlab/Documents/cole/VSCode/RamanLab-Locust-Behavior/Locust/analysis_outputs/dropped_frames"
)

DIST_RAW_COL = "distance_palp1_palp2_px_raw"
DIST_COL = "distance_palp1_palp2_px"

DATE_DIR_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")
VIDEO_EXTS = [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]


def _parse_video_stem(csv_path: Path) -> str:
    stem = csv_path.stem
    return stem[:-len("_palps_tracks")] if stem.endswith("_palps_tracks") else stem


def _infer_dataset(csv_path: Path) -> str:
    parts = csv_path.parts
    if "all_vids" in parts:
        i = parts.index("all_vids")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "UNKNOWN"


def _find_video_for_csv(csv_path: Path, video_stem: str) -> Optional[Path]:
    candidates = []
    # common: date_dir/<video_stem>.mp4
    date_dir = csv_path.parent.parent
    candidates.append(date_dir)
    # sometimes video is next to CSV folder
    candidates.append(csv_path.parent)

    for base in candidates:
        for ext in VIDEO_EXTS:
            p = base / f"{video_stem}{ext}"
            if p.exists():
                return p
    return None


def _invalid_frames(df: pd.DataFrame, min_px: float) -> pd.DataFrame:
    if DIST_RAW_COL in df.columns:
        raw = pd.to_numeric(df[DIST_RAW_COL], errors="coerce")
    else:
        raw = pd.to_numeric(df[DIST_COL], errors="coerce")

    frame = (
        pd.to_numeric(df["frame"], errors="coerce").fillna(0).astype(int)
        if "frame" in df.columns
        else pd.Series(np.arange(len(df), dtype=int))
    )

    is_nan = raw.isna()
    is_low = raw < min_px
    mask = is_nan | is_low

    out = pd.DataFrame(
        {
            "frame": frame[mask].to_numpy(),
            "distance_raw": raw[mask].to_numpy(),
            "reason": np.where(is_nan[mask], "nan", "below_min"),
        }
    )
    return out.sort_values("frame")


def export_frames(video_path: Path, invalid_df: pd.DataFrame, out_dir: Path) -> int:
    if invalid_df.empty:
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    invalid_df.to_csv(out_dir / "invalid_frames.csv", index=False)

    frames = invalid_df["frame"].to_numpy(dtype=int)
    frame_set = set(frames.tolist())
    if not frame_set:
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video: {video_path}")
        return 0

    saved = 0
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_set:
            out_path = out_dir / f"frame_{idx:06d}.png"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        idx += 1

    cap.release()
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Export dropped/invalid frames for locust tracking.")
    parser.add_argument("--all-vids-dir", type=Path, default=DEFAULT_ALL_VIDS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-px", type=float, default=9.5)
    args = parser.parse_args()

    all_vids_dir = args.all_vids_dir.expanduser().resolve()
    out_root = args.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(all_vids_dir.rglob("*_palps_tracks.csv"))
    if not csv_paths:
        raise SystemExit(f"No *_palps_tracks.csv found under {all_vids_dir}")

    total_saved = 0
    total_videos = 0
    for csv_path in csv_paths:
        video_stem = _parse_video_stem(csv_path)
        stem_lower = video_stem.lower()
        if "testing" not in stem_lower or "training" in stem_lower:
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[WARN] Failed reading {csv_path}: {exc}")
            continue

        invalid_df = _invalid_frames(df, args.min_px)
        if invalid_df.empty:
            continue

        video_path = _find_video_for_csv(csv_path, video_stem)
        if video_path is None:
            print(f"[WARN] Video not found for {csv_path}")
            continue

        dataset = _infer_dataset(csv_path)
        out_dir = out_root / dataset / video_stem
        saved = export_frames(video_path, invalid_df, out_dir)
        if saved:
            total_saved += saved
            total_videos += 1
            print(f"[OK] {video_stem}: saved {saved} frames -> {out_dir}")

    print(f"[DONE] Videos processed: {total_videos}")
    print(f"[DONE] Frames saved: {total_saved}")


if __name__ == "__main__":
    main()
