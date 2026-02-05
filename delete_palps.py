#!/usr/bin/env python3
"""
Delete video files whose names end with
    Recording_palps_annotated_30fps.mp4

Dry-run by default. Use --execute to actually delete files.
"""

import sys
from pathlib import Path

BASE_DIR = Path("/home/ramanlab/Documents/cole/VSCode/RamanLab-Locust-Behavior/Locust/all_vids")
TARGET_SUFFIX = 'recording_palps_annotated_30fps.mp4'


def find_targets(base_dir: Path):
    for p in base_dir.rglob('*'):
        if p.is_file():
            if p.name.lower().endswith(TARGET_SUFFIX):
                yield p


def run(dry_run=True):
    targets = list(find_targets(BASE_DIR))
    if not targets:
        print('No matching files found.')
        return

    for p in targets:
        if dry_run:
            print(f"Would delete: {p}")
        else:
            try:
                p.unlink()
                print(f"Deleted: {p}")
            except Exception as e:
                print(f"Error deleting {p}: {e}")

    if dry_run:
        print('\nDRY RUN complete - no files were deleted')


if __name__ == '__main__':
    execute = '--execute' in sys.argv
    if execute:
        print('EXECUTING deletions (not a dry run)')
        run(dry_run=False)
    else:
        print('DRY RUN - preview deletions. Use --execute to perform them.')
        run(dry_run=True)
