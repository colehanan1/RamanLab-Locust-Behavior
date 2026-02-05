#!/usr/bin/env python3
"""
Fix files with duplicated trailing segments like:
  2-OCT_ON_08262025_Training_L1_Trial_1_Recording.mp4_Trial_1_Recording.mp4
to:
  2-OCT_ON_08262025_Training_L1_Trial_1_Recording.mp4

Dry-run by default; use --execute to actually rename files.
"""

import re
import sys
import shutil
from pathlib import Path

BASE_DIR = Path("/home/ramanlab/Documents/cole/VSCode/RamanLab-Locust-Behavior/Locust/all_vids")
VIDEO_EXTENSIONS = ['mp4','avi','mov','mkv','wmv','flv','m4v']

# Build regex to match duplicated trailing segment (case-insensitive)
ext_group = '|'.join(VIDEO_EXTENSIONS)
pattern = re.compile(rf'^(?P<prefix>.*?)(?P<dup>[^/]+\.(?:{ext_group}))_(?P=dup)$'.replace('{ext_group}', ext_group), re.IGNORECASE)


def find_files(base_dir: Path):
    return base_dir.rglob("*")


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


def fix_duplicates(dry_run=True):
    fixed = 0
    skipped = 0
    for p in find_files(BASE_DIR):
        if not p.is_file():
            continue
        name = p.name
        m = pattern.match(name)
        if not m:
            continue
        prefix = m.group('prefix')
        dup = m.group('dup')
        new_name = f"{prefix}{dup}"
        target = p.with_name(new_name)
        if target.exists():
            target = unique_target_path(p.parent, new_name)
        if dry_run:
            print(f"Would rename:\n  FROM: {p}\n  TO:   {target}\n")
        else:
            try:
                shutil.move(str(p), str(target))
                print(f"Renamed: {p} -> {target}")
                fixed += 1
            except Exception as e:
                print(f"Error renaming {p}: {e}")
                skipped += 1
    print("\n" + "="*60)
    if dry_run:
        print("DRY RUN complete - no files were changed")
    print(f"Fixed: {fixed}")
    print(f"Skipped (errors): {skipped}")
    print("="*60)


if __name__ == '__main__':
    execute = '--execute' in sys.argv
    if execute:
        print("EXECUTING fixes (not a dry run)")
        fix_duplicates(dry_run=False)
    else:
        print("DRY RUN - preview fixes. Use --execute to perform them.")
        fix_duplicates(dry_run=True)
