#!/usr/bin/env python3
"""
Move any videos under each date folder up to the date folder, then delete all other
subdirectories and non-video files under each date folder.

Dry-run by default. Use --execute to actually delete.

Keeps: <base>/condition/date/*.mp4 (and other video extensions)
Removes: any subfolders under a date folder and any non-video files inside date folder.
"""

import sys
import shutil
from pathlib import Path

BASE_DIR = Path("/home/ramanlab/Documents/cole/VSCode/RamanLab-Locust-Behavior/Locust/all_vids")
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v', '.MP4', '.AVI', '.MOV'}


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


def is_video(p: Path) -> bool:
    return p.is_file() and p.suffix in VIDEO_EXTENSIONS


def process(dry_run=True):
    actions = []
    # iterate condition folders
    if not BASE_DIR.exists():
        print(f"Base directory not found: {BASE_DIR}")
        return

    for cond in sorted(BASE_DIR.iterdir()):
        if not cond.is_dir():
            continue
        for date_dir in sorted(cond.iterdir()):
            if not date_dir.is_dir():
                continue
            # 1) Move any videos from subfolders into date_dir
            for f in date_dir.rglob("*"):
                if is_video(f):
                    if f.parent.resolve() != date_dir.resolve():
                        target = date_dir / f.name
                        if target.exists():
                            target = unique_target_path(date_dir, f.name)
                        actions.append(("move", f, target))
            # 2) Delete subdirectories under date_dir
            for child in sorted(date_dir.iterdir()):
                if child.is_dir():
                    actions.append(("rmdir", child, None))
                elif child.is_file():
                    # if not a video file, schedule delete
                    if not is_video(child):
                        actions.append(("unlink", child, None))
            # end date_dir
    # summarize and optionally execute
    if not actions:
        print("No actions found. Nothing to do.")
        return

    # Print preview
    for act, src, dst in actions:
        if act == 'move':
            print(f"Would move:\n  FROM: {src}\n  TO:   {dst}\n")
        elif act == 'rmdir':
            print(f"Would remove directory and contents:\n  {src}\n")
        elif act == 'unlink':
            print(f"Would remove file:\n  {src}\n")

    if dry_run:
        print('\n' + '='*60)
        print('DRY RUN complete - no changes made')
        print('='*60)
        return

    # Execute actions
    moved = 0
    removed_dirs = 0
    removed_files = 0
    for act, src, dst in actions:
        try:
            if act == 'move':
                shutil.move(str(src), str(dst))
                moved += 1
            elif act == 'rmdir':
                shutil.rmtree(str(src))
                removed_dirs += 1
            elif act == 'unlink':
                src.unlink()
                removed_files += 1
        except Exception as e:
            print(f"Error on {act} {src}: {e}")
    print('\n' + '='*60)
    print(f"Moved files: {moved}")
    print(f"Removed directories: {removed_dirs}")
    print(f"Removed files: {removed_files}")
    print('='*60)


if __name__ == '__main__':
    execute = '--execute' in sys.argv
    if execute:
        print('EXECUTING cleanup (not a dry run)')
        process(dry_run=False)
    else:
        print('DRY RUN - preview deletions. Use --execute to perform them.')
        process(dry_run=True)
