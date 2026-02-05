#!/usr/bin/env python3
"""
Move renamed video files up to their date parent folder under Locust/all_vids.

Example:
  From: Locust/all_vids/2-OCT_ON/08.26.2025/Training/L1/2-OCT_ON_08262025_Training_L1_Trial_1_Recording.mp4
  To:   Locust/all_vids/2-OCT_ON/08.26.2025/2-OCT_ON_08262025_Training_L1_Trial_1_Recording.mp4

Run in dry-run (preview) mode by default. Use --execute to actually move files.
"""

import sys
import os
import shutil
from pathlib import Path

BASE_DIR = Path("/home/ramanlab/Documents/cole/VSCode/RamanLab-Locust-Behavior/Locust/all_vids")
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v', '.MP4', '.AVI', '.MOV'}


def find_all_videos(base_dir):
    for p in base_dir.rglob("*"):
        if p.is_file() and p.suffix in VIDEO_EXTENSIONS:
            yield p


def target_date_dir(path: Path):
    parts = path.parts
    try:
        idx = parts.index('all_vids')
    except ValueError:
        return None
    # expect at least: all_vids / condition / date / ... / file
    if len(parts) <= idx + 2:
        return None
    condition = parts[idx + 1]
    date = parts[idx + 2]
    return Path(*parts[:idx + 3])


def unique_target_path(target_dir: Path, name: str) -> Path:
    target = target_dir / name
    if not target.exists():
        return target
    stem = target.stem
    suf = target.suffix
    i = 1
    while True:
        new_name = f"{stem}_{i}{suf}"
        new_target = target_dir / new_name
        if not new_target.exists():
            return new_target
        i += 1


def move_files(dry_run=True):
    moved = 0
    skipped = 0
    for file in find_all_videos(BASE_DIR):
        tdir = target_date_dir(file)
        if tdir is None:
            print(f"Skipping (can't determine date dir): {file}")
            skipped += 1
            continue
        # ensure tdir exists
        if not tdir.exists():
            print(f"Target date dir does not exist (skipping): {tdir} for file {file}")
            skipped += 1
            continue
        # if file already in target date dir, skip
        if file.parent.resolve() == tdir.resolve():
            # already in place
            continue
        # compute target path
        target_path = tdir / file.name
        if target_path.exists():
            target_path = unique_target_path(tdir, file.name)
        if dry_run:
            print(f"Would move:\n  FROM: {file}\n  TO:   {target_path}\n")
        else:
            try:
                shutil.move(str(file), str(target_path))
                print(f"Moved: {file} -> {target_path}")
                moved += 1
            except Exception as e:
                print(f"Error moving {file}: {e}")
                skipped += 1
    print("\n" + "="*60)
    if dry_run:
        print("DRY RUN complete - no files moved")
    print(f"Moved: {moved}")
    print(f"Skipped: {skipped}")
    print("="*60)


if __name__ == '__main__':
    execute = '--execute' in sys.argv
    if execute:
        print("EXECUTING moves (not a dry run)")
        move_files(dry_run=False)
    else:
        print("DRY RUN - preview moves. Use --execute to perform them.")
        move_files(dry_run=True)
