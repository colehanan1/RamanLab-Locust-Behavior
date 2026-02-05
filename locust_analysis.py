#!/usr/bin/env python3
"""
Locust palp distance analysis pipeline.

This script scans all *_palps_tracks.csv files under Locust/all_vids,
adds per-locust distance percentage columns, builds combined CSVs,
and generates trace plots + reaction matrices (testing-only).

Key assumptions:
  - Videos are 30 FPS.
  - Odor ON at 10s, OFF at 14s.
  - Threshold = mean(before) + k * std(before), k=4 by default.
  - Reaction = >= min_samples_over frames above threshold during odor window.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


DEFAULT_ALL_VIDS_DIR = Path(
    "/home/ramanlab/Documents/cole/VSCode/RamanLab-Locust-Behavior/Locust/all_vids"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/ramanlab/Documents/cole/VSCode/RamanLab-Locust-Behavior/Locust/analysis_outputs"
)

DIST_COL = "distance_palp1_palp2_px"
DIST_RAW_COL = "distance_palp1_palp2_px_raw"
PCT_COL = "distance_palp1_palp2_pct"
MIN_COL = "min_distance_palp1_palp2_px"
MAX_COL = "max_distance_palp1_palp2_px"
RAW_MIN_COL = "raw_min_distance_palp1_palp2_px"
EFF_MIN_COL = "effective_min_distance_palp1_palp2_px"

DATE_DIR_RE = re.compile(r"^\d{2}\.\d{2}\.\d{4}$")
LOCUST_RE = re.compile(r"(?:^|_)L(\d+)(?:_|$)", re.IGNORECASE)
TRIAL_RE = re.compile(r"Trial[_-]?(\d+)", re.IGNORECASE)

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


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "UNKNOWN"


def _find_palps_csvs(root: Path) -> List[Path]:
    return sorted(root.rglob("*_palps_tracks.csv"))


def _parse_video_stem(csv_path: Path) -> str:
    stem = csv_path.stem
    return stem[:-len("_palps_tracks")] if stem.endswith("_palps_tracks") else stem


def _infer_dataset_and_date(csv_path: Path) -> Tuple[str, str]:
    parts = csv_path.parts
    dataset = "UNKNOWN"
    date = "UNKNOWN"
    if "all_vids" in parts:
        idx = parts.index("all_vids")
        if idx + 1 < len(parts):
            dataset = parts[idx + 1]
        for part in parts[idx + 2 :]:
            if DATE_DIR_RE.match(part):
                date = part
                break
    return dataset, date


def _infer_trial_type(video_stem: str) -> str:
    if re.search(r"testing", video_stem, re.IGNORECASE):
        return "testing"
    if re.search(r"training", video_stem, re.IGNORECASE):
        return "training"
    return "unknown"


def _infer_trial_num(video_stem: str) -> Optional[int]:
    m = TRIAL_RE.search(video_stem)
    return int(m.group(1)) if m else None


def _infer_locust_id(video_stem: str) -> str:
    m = LOCUST_RE.search(video_stem)
    return f"L{m.group(1)}" if m else "UNKNOWN"


def _normalize_odor_token(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    return token.strip().upper()


def _find_odor_token(tokens: Iterable[str]) -> Optional[str]:
    for tok in reversed(list(tokens)):
        code = _normalize_odor_token(tok)
        if code in ODOR_MAP:
            return code
    return None


def infer_odor_sent_from_stem(video_stem: str) -> str:
    tokens = video_stem.split("_")
    date_idx = next((i for i, t in enumerate(tokens) if t.isdigit() and len(t) == 8), None)

    if date_idx is None:
        code = _find_odor_token(tokens)
        return ODOR_MAP.get(code, code or "Unknown")

    condition_tokens = tokens[:date_idx]
    phase = tokens[date_idx + 1] if date_idx + 1 < len(tokens) else ""
    trained_code = _find_odor_token(condition_tokens)

    sent_code = None
    if phase.lower() == "testing":
        candidate = tokens[date_idx + 2] if date_idx + 2 < len(tokens) else None
        code = _normalize_odor_token(candidate)
        if code in ODOR_MAP:
            sent_code = code

    if sent_code is None:
        sent_code = trained_code

    return ODOR_MAP.get(sent_code, sent_code or "Unknown")


def compute_distance_pct(values: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    pct = np.full_like(values, np.nan, dtype=float)
    if not np.isfinite(min_val) or not np.isfinite(max_val):
        return pct
    if max_val == min_val:
        pct[np.isfinite(values)] = 0.0
        return pct
    mask = np.isfinite(values)
    pct[mask] = 100.0 * (values[mask] - min_val) / (max_val - min_val)
    return pct


def apply_min_floor(raw_min: float, raw_max: float, floor_px: Optional[float]) -> float:
    if not np.isfinite(raw_min) or not np.isfinite(raw_max):
        return raw_min
    if floor_px is None or not np.isfinite(floor_px):
        return raw_min
    eff_min = max(raw_min, float(floor_px))
    if eff_min > raw_max:
        return raw_max
    return eff_min


def compute_threshold(
    dist_pct: np.ndarray,
    fps: float,
    odor_on_s: float,
    k: float,
) -> float:
    if dist_pct.size == 0 or fps <= 0:
        return math.nan
    before_end = int(round(odor_on_s * fps))
    before_end = max(0, min(before_end, dist_pct.size))
    if before_end == 0:
        return math.nan
    baseline = dist_pct[:before_end]
    finite = baseline[np.isfinite(baseline)]
    if finite.size == 0:
        return math.nan
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=0))
    return float(mean + k * std)


def compute_reaction(
    dist_pct: np.ndarray,
    fps: float,
    odor_on_s: float,
    odor_off_s: float,
    threshold: float,
    min_samples_over: int,
) -> Tuple[int, int]:
    if dist_pct.size == 0 or fps <= 0 or not np.isfinite(threshold):
        return 0, 0
    start = int(round(odor_on_s * fps))
    end = int(round(odor_off_s * fps))
    start = max(0, min(start, dist_pct.size))
    end = max(start, min(end, dist_pct.size))
    segment = dist_pct[start:end]
    if segment.size == 0:
        return 0, 0
    count_over = int(np.sum(segment > threshold))
    reacted = 1 if count_over >= min_samples_over else 0
    return reacted, count_over


def write_combined_long(
    out_path: Path,
    df: pd.DataFrame,
    meta: Dict[str, object],
    time_s: np.ndarray,
    dist_pct: np.ndarray,
    threshold: float,
    reacted: int,
    fps: float,
) -> None:
    out_df = pd.DataFrame(
        {
            "dataset": meta["dataset"],
            "date": meta["date"],
            "locust_id": meta["locust_id"],
            "locust_key": meta["locust_key"],
            "trial_type": meta["trial_type"],
            "trial_num": meta["trial_num"],
            "odor_sent": meta["odor_sent"],
            "video_stem": meta["video_stem"],
            "csv_path": meta["csv_path"],
            "frame": df["frame"] if "frame" in df.columns else np.arange(len(df)),
            "time_s": time_s,
            "distance_px": df[DIST_COL],
            "distance_pct": dist_pct,
            "odor_on": df["odor_on"] if "odor_on" in df.columns else False,
            "threshold": threshold,
            "reacted": reacted,
            "fps": fps,
        }
    )
    header = not out_path.exists()
    out_df.to_csv(out_path, mode="a", header=header, index=False)


def plot_locust_traces(
    out_dir: Path,
    dataset: str,
    locust_label: str,
    trials: List[Dict[str, object]],
    odor_on_s: float,
    odor_off_s: float,
) -> None:
    if not trials:
        return
    trials = sorted(trials, key=lambda t: (t.get("trial_num") or 0, t.get("video_stem", "")))
    n = len(trials)
    fig_h = max(3.0, n * 1.6 + 1.2)
    fig, axes = plt.subplots(n, 1, figsize=(10, fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    y_max = 0.0
    for trial in trials:
        dist_pct = trial["dist_pct"]
        if dist_pct.size and np.isfinite(dist_pct).any():
            y_max = max(y_max, float(np.nanmax(dist_pct)))
        theta = trial.get("threshold", math.nan)
        if np.isfinite(theta):
            y_max = max(y_max, float(theta))
    if y_max <= 0:
        y_max = 1.0

    for ax, trial in zip(axes, trials):
        t = trial["time_s"]
        dist_pct = trial["dist_pct"]
        theta = trial.get("threshold", math.nan)
        label = trial.get("label", "trial")

        ax.plot(t, dist_pct, linewidth=1.2, color="black")
        ax.axvline(odor_on_s, linestyle="--", linewidth=1.0, color="black")
        ax.axvline(odor_off_s, linestyle="--", linewidth=1.0, color="black")
        ax.axvspan(odor_on_s, odor_off_s, alpha=0.15, color="gray")
        if np.isfinite(theta):
            ax.axhline(theta, linestyle="-", linewidth=1.0, color="tab:red", alpha=0.9)

        ax.set_ylabel("Distance %")
        ax.set_title(label, loc="left", fontsize=10, weight="bold")
        ax.set_ylim(0, y_max * 1.05 if y_max > 0 else 1.0)
        ax.margins(x=0, y=0.02)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"{dataset} — {locust_label} — Testing Trials",
        fontsize=13,
        weight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_name(locust_label)}_testing_traces.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_reaction_matrix(
    out_dir: Path,
    dataset: str,
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
) -> None:
    if matrix.size == 0:
        return

    cmap = ListedColormap(["white", "black"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig_h = max(3.5, 0.25 * len(row_labels) + 2.5)
    fig_w = max(6.5, 0.5 * len(col_labels) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_title(f"{dataset} — Reaction Matrix (Testing)", fontsize=13, weight="bold")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Odor Sent")
    ax.set_ylabel("Locust")

    legend_handles = [
        plt.Line2D([0], [0], marker="s", color="white", label="No reaction", markerfacecolor="white", markersize=10, markeredgecolor="black"),
        plt.Line2D([0], [0], marker="s", color="black", label="Reaction", markerfacecolor="black", markersize=10),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_name(dataset)}_reaction_matrix.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dataset_mean_sem(
    out_dir: Path,
    dataset: str,
    odor_stats: Dict[str, Dict[str, np.ndarray]],
    fps: float,
    odor_on_s: float,
    odor_off_s: float,
) -> None:
    if not odor_stats:
        return

    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.get_cmap("tab10")
    y_max = 0.0

    for idx, (odor, stats) in enumerate(sorted(odor_stats.items())):
        mean = stats["mean"]
        sem = stats["sem"]
        n_trials = int(stats["n"])
        if mean.size == 0:
            continue
        t = np.arange(len(mean), dtype=float) / fps
        color = cmap(idx % cmap.N)
        label = f"{odor} (n={n_trials})"
        ax.plot(t, mean, linewidth=1.6, color=color, label=label)
        ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.2)
        if np.isfinite(mean).any():
            y_max = max(y_max, float(np.nanmax(mean + sem)))

    ax.axvline(odor_on_s, linestyle="--", linewidth=1.0, color="black")
    ax.axvline(odor_off_s, linestyle="--", linewidth=1.0, color="black")
    ax.axvspan(odor_on_s, odor_off_s, alpha=0.15, color="gray")

    ax.set_title(f"{dataset} — Testing Odor Means ± SEM", fontsize=13, weight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance %")
    ax.margins(x=0, y=0.02)
    if y_max > 0:
        ax.set_ylim(0, y_max * 1.05)

    ax.legend(loc="upper right", frameon=True, fontsize=9)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_safe_name(dataset)}_testing_odors_mean_sem.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Locust palp distance analysis.")
    parser.add_argument("--all-vids-dir", type=Path, default=DEFAULT_ALL_VIDS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--odor-on-s", type=float, default=10.0)
    parser.add_argument("--odor-off-s", type=float, default=14.0)
    parser.add_argument("--threshold-k", type=float, default=4.0)
    parser.add_argument("--min-samples-over", type=int, default=20)
    parser.add_argument(
        "--min-floor-px",
        type=float,
        default=9.5,
        help=(
            "Minimum pixel floor for 0%% normalization (default: 9.5). "
            "effective_min = max(raw_min, min_floor_px) per locust. "
            "Values below min_floor_px are treated as invalid and set to NaN in distance_pct."
        ),
    )
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot generation.")
    parser.add_argument("--skip-combined", action="store_true", help="Skip combined CSV outputs.")
    args = parser.parse_args()

    all_vids_dir = args.all_vids_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_plots:
        for sub in ("traces", "reaction_matrix", "dataset_means"):
            subdir = output_dir / sub
            if subdir.exists():
                shutil.rmtree(subdir)

    csv_paths = _find_palps_csvs(all_vids_dir)
    if not csv_paths:
        raise SystemExit(f"No *_palps_tracks.csv files found under {all_vids_dir}")

    entries: List[Dict[str, object]] = []
    for path in csv_paths:
        dataset, date = _infer_dataset_and_date(path)
        video_stem = _parse_video_stem(path)
        trial_type = _infer_trial_type(video_stem)
        if trial_type != "testing" or "training" in video_stem.lower():
            continue
        trial_num = _infer_trial_num(video_stem)
        locust_id = _infer_locust_id(video_stem)
        locust_key = f"{dataset}/{date}/{locust_id}"
        locust_label = f"{locust_id}_{date}" if date != "UNKNOWN" else locust_id
        entries.append(
            {
                "csv_path": path,
                "dataset": dataset,
                "date": date,
                "video_stem": video_stem,
                "trial_type": trial_type,
                "trial_num": trial_num,
                "locust_id": locust_id,
                "locust_key": locust_key,
                "locust_label": locust_label,
            }
        )

    # Pass 1: compute per-locust min/max
    locust_stats: Dict[str, Tuple[float, float]] = {}
    for entry in entries:
        path = entry["csv_path"]
        try:
            df = pd.read_csv(path, usecols=[DIST_COL])
        except Exception as exc:
            print(f"[WARN] Failed reading {path}: {exc}")
            continue
        if DIST_COL not in df.columns:
            print(f"[WARN] Missing {DIST_COL} in {path}")
            continue
        dist = pd.to_numeric(df[DIST_COL], errors="coerce")
        if not dist.notna().any():
            continue
        key = entry["locust_key"]
        current = locust_stats.get(key, (math.inf, -math.inf))
        loc_min = float(np.nanmin(dist.to_numpy()))
        loc_max = float(np.nanmax(dist.to_numpy()))
        locust_stats[key] = (min(current[0], loc_min), max(current[1], loc_max))

    if not locust_stats:
        raise SystemExit("No valid distances found to compute min/max.")

    combined_long_path = output_dir / "locust_combined_long.csv"
    if combined_long_path.exists():
        combined_long_path.unlink()

    combined_trials: List[Dict[str, object]] = []
    avg_traces: Dict[str, Dict[str, Dict[str, List[np.ndarray]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    traces_by_locust: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for entry in entries:
        path = entry["csv_path"]
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[WARN] Failed reading {path}: {exc}")
            continue

        if DIST_COL not in df.columns:
            print(f"[WARN] Missing {DIST_COL} in {path}")
            continue

        dist = pd.to_numeric(df[DIST_COL], errors="coerce").to_numpy(dtype=float)
        df[DIST_RAW_COL] = dist
        raw_min, max_val = locust_stats.get(entry["locust_key"], (math.nan, math.nan))
        min_val = apply_min_floor(raw_min, max_val, args.min_floor_px)

        dist_masked = dist.copy()
        if args.min_floor_px is not None and np.isfinite(args.min_floor_px):
            mask_low = dist_masked < args.min_floor_px
            dist_masked[mask_low] = np.nan
            df.loc[mask_low, DIST_COL] = np.nan

        dist_pct = compute_distance_pct(dist_masked, min_val, max_val)

        df[PCT_COL] = dist_pct
        df[MIN_COL] = min_val
        df[MAX_COL] = max_val
        df[RAW_MIN_COL] = raw_min
        df[EFF_MIN_COL] = min_val
        df["min_floor_px_used"] = args.min_floor_px if args.min_floor_px is not None else np.nan

        if "frame" in df.columns:
            frames = pd.to_numeric(df["frame"], errors="coerce").fillna(0).to_numpy(dtype=int)
        else:
            frames = np.arange(len(df), dtype=int)
            df["frame"] = frames

        time_s = frames.astype(float) / args.fps

        if "odor_on" not in df.columns:
            df["odor_on"] = (time_s >= args.odor_on_s) & (time_s <= args.odor_off_s)

        odor_sent = None
        if "odor_sent" in df.columns:
            values = df["odor_sent"].dropna().astype(str)
            if not values.empty:
                odor_sent = values.iloc[0]
        if not odor_sent:
            odor_sent = infer_odor_sent_from_stem(entry["video_stem"])
            df["odor_sent"] = odor_sent
        entry["odor_sent"] = odor_sent

        threshold = compute_threshold(dist_pct, args.fps, args.odor_on_s, args.threshold_k)
        reacted, count_over = compute_reaction(
            dist_pct,
            args.fps,
            args.odor_on_s,
            args.odor_off_s,
            threshold,
            args.min_samples_over,
        )

        df.to_csv(path, index=False)

        if not args.skip_combined:
            write_combined_long(
                combined_long_path,
                df,
                entry,
                time_s,
                dist_pct,
                threshold,
                reacted,
                args.fps,
            )

        combined_trials.append(
            {
                "dataset": entry["dataset"],
                "date": entry["date"],
                "locust_id": entry["locust_id"],
                "locust_key": entry["locust_key"],
                "trial_type": entry["trial_type"],
                "trial_num": entry["trial_num"],
                "odor_sent": odor_sent,
                "video_stem": entry["video_stem"],
                "csv_path": str(path),
                "n_frames": len(df),
                "min_distance_px": min_val,
                "raw_min_distance_px": raw_min,
                "max_distance_px": max_val,
                "threshold": threshold,
                "count_over": count_over,
                "reacted": reacted,
                "fps": args.fps,
            }
        )

        if entry["trial_type"] == "testing" and "training" not in entry["video_stem"].lower():
            label = f"{entry['video_stem']} ({odor_sent})"
            traces_by_locust[entry["locust_key"]].append(
                {
                    "time_s": time_s,
                    "dist_pct": dist_pct,
                    "threshold": threshold,
                    "label": label,
                    "trial_num": entry["trial_num"],
                    "video_stem": entry["video_stem"],
                    "dataset": entry["dataset"],
                    "locust_label": entry["locust_label"],
                }
            )
            avg_traces[entry["dataset"]][odor_sent][entry["locust_label"]].append(dist_pct)

    combined_trials_path = output_dir / "locust_combined_trials.csv"
    pd.DataFrame(combined_trials).to_csv(combined_trials_path, index=False)

    # Reaction matrix data (testing only)
    trials_df = pd.DataFrame(combined_trials)
    testing_df = trials_df[trials_df["trial_type"] == "testing"].copy()

    if not args.skip_plots:
        traces_root = output_dir / "traces"
        for locust_key, trials in traces_by_locust.items():
            dataset = trials[0]["dataset"] if trials else "UNKNOWN"
            locust_label = trials[0].get("locust_label", locust_key)
            plot_locust_traces(
                traces_root / _safe_name(dataset),
                dataset,
                locust_label,
                trials,
                args.odor_on_s,
                args.odor_off_s,
            )

        matrix_root = output_dir / "reaction_matrix"
        for dataset, subset in testing_df.groupby("dataset"):
            if subset.empty:
                continue
            odor_list = sorted(subset["odor_sent"].dropna().unique().tolist())
            locust_list = sorted(subset["locust_key"].dropna().unique().tolist())

            matrix = np.full((len(locust_list), len(odor_list)), 0, dtype=int)
            for i, locust in enumerate(locust_list):
                for j, odor in enumerate(odor_list):
                    mask = (subset["locust_key"] == locust) & (subset["odor_sent"] == odor)
                    if mask.any():
                        matrix[i, j] = int(subset.loc[mask, "reacted"].max())

            csv_out = matrix_root / f"{_safe_name(dataset)}_reaction_matrix.csv"
            df_matrix = pd.DataFrame(matrix, index=locust_list, columns=odor_list)
            csv_out.parent.mkdir(parents=True, exist_ok=True)
            df_matrix.to_csv(csv_out)

            plot_reaction_matrix(matrix_root, dataset, matrix, locust_list, odor_list)

        dataset_mean_root = output_dir / "dataset_means"
        for dataset, odors in avg_traces.items():
            odor_stats: Dict[str, Dict[str, np.ndarray]] = {}
            for odor, trials in odors.items():
                if not trials:
                    continue

                locust_means: List[np.ndarray] = []
                for locust_label, locust_trials in trials.items():
                    if not locust_trials:
                        continue
                    max_len = max(len(t) for t in locust_trials)
                    stack = np.full((len(locust_trials), max_len), np.nan, dtype=float)
                    for idx, arr in enumerate(locust_trials):
                        stack[idx, : len(arr)] = arr
                    if not np.isfinite(stack).any():
                        continue
                    locust_means.append(np.nanmean(stack, axis=0))

                if not locust_means:
                    continue

                max_len = max(len(t) for t in locust_means)
                stack = np.full((len(locust_means), max_len), np.nan, dtype=float)
                for idx, arr in enumerate(locust_means):
                    stack[idx, : len(arr)] = arr
                if not np.isfinite(stack).any():
                    continue

                mean = np.nanmean(stack, axis=0)
                n = np.sum(np.isfinite(stack), axis=0).astype(float)
                std = np.nanstd(stack, axis=0, ddof=0)
                sem = np.where(n > 1, std / np.sqrt(n), 0.0)
                odor_stats[odor] = {"mean": mean, "sem": sem, "n": len(locust_means)}

            plot_dataset_mean_sem(
                dataset_mean_root / _safe_name(dataset),
                dataset,
                odor_stats,
                args.fps,
                args.odor_on_s,
                args.odor_off_s,
            )

    print(f"[OK] Updated {len(combined_trials)} trial CSVs.")
    if args.min_floor_px is not None:
        print(f"[OK] Applied min floor for normalization: {args.min_floor_px:.4f}px")
    if not args.skip_combined:
        print(f"[OK] Combined long CSV: {combined_long_path}")
    print(f"[OK] Combined trials CSV: {combined_trials_path}")
    if not args.skip_plots:
        print(f"[OK] Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
