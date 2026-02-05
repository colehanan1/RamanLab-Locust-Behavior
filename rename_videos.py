#!/usr/bin/env python3
"""
Script to rename video files in Locust/all_vids directory structure.
Renames files from nested folder paths to a flat naming convention.

Example:
  Input:  Locust/all_vids/2-OCT_ON/08.26.2025/Testing/2-OCT/L1/Trial_1_Recording.mp4
  Output: 2-OCT_ON_08262025_Testing_2-OCT_L1_Trial_1_Recording.mp4
"""

import os
import re
from pathlib import Path

# Define the base directory and video extensions
BASE_DIR = Path("/home/ramanlab/Documents/cole/VSCode/RamanLab-Locust-Behavior/Locust/all_vids")
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.m4v', '.MOV', '.MP4', '.AVI'}

def extract_path_components(file_path):
    """Extract relevant path components from the full file path."""
    parts = file_path.parts
    
    # Find the index of 'all_vids'
    try:
        all_vids_idx = parts.index('all_vids')
    except ValueError:
        return None
    
    # Extract components after all_vids
    relative_parts = parts[all_vids_idx + 1:]
    
    if len(relative_parts) < 5:
        return None
    
    # Expected structure: condition/date/phase/odor/locust_id/filename
    condition = relative_parts[0]
    date = relative_parts[1].replace('.', '')  # Remove dots from date
    phase = relative_parts[2]  # Testing or Training
    odor = relative_parts[3]
    locust_id = relative_parts[4]
    filename = relative_parts[-1]
    
    return {
        'condition': condition,
        'date': date,
        'phase': phase,
        'odor': odor,
        'locust_id': locust_id,
        'filename': filename,
        'name_only': os.path.splitext(filename)[0],
        'extension': os.path.splitext(filename)[1]
    }

def create_new_filename(components):
    """Create new filename based on path components."""
    new_name = f"{components['condition']}_{components['date']}_{components['phase']}_{components['odor']}_{components['locust_id']}_{components['name_only']}{components['extension']}"
    return new_name

def rename_videos(dry_run=True):
    """
    Rename all video files in the directory structure.
    
    Args:
        dry_run (bool): If True, only print what would be renamed without actually renaming.
    """
    if not BASE_DIR.exists():
        print(f"Error: Base directory not found: {BASE_DIR}")
        return
    
    renamed_count = 0
    skipped_count = 0
    
    # Walk through all directories
    for root, dirs, files in os.walk(BASE_DIR):
        for filename in files:
            file_ext = os.path.splitext(filename)[1]
            
            # Check if it's a video file
            if file_ext not in VIDEO_EXTENSIONS:
                continue
            
            full_path = os.path.join(root, filename)
            file_path = Path(full_path)
            
            # Extract path components
            components = extract_path_components(file_path)
            
            if components is None:
                print(f"⚠️  Skipped (invalid path structure): {full_path}")
                skipped_count += 1
                continue
            
            # Create new filename
            new_filename = create_new_filename(components)
            new_path = os.path.join(root, new_filename)
            
            if dry_run:
                print(f"✓ Would rename:")
                print(f"  FROM: {full_path}")
                print(f"  TO:   {new_path}")
                print()
            else:
                try:
                    os.rename(full_path, new_path)
                    print(f"✓ Renamed: {filename} → {new_filename}")
                    renamed_count += 1
                except Exception as e:
                    print(f"✗ Error renaming {filename}: {e}")
                    skipped_count += 1
    
    print("\n" + "="*80)
    if dry_run:
        print(f"DRY RUN COMPLETE - No files were actually renamed")
    print(f"Total videos processed: {renamed_count + skipped_count}")
    print(f"Ready to rename: {renamed_count}")
    print(f"Skipped: {skipped_count}")
    print("="*80)

if __name__ == "__main__":
    import sys
    
    # Check for --execute flag to actually rename files
    execute = "--execute" in sys.argv
    
    if execute:
        print("⚠️  EXECUTING FILE RENAMES (not a dry run)")
        print("="*80)
        rename_videos(dry_run=False)
    else:
        print("DRY RUN MODE - No files will be renamed")
        print("Run with --execute flag to actually rename files")
        print("="*80)
        print()
        rename_videos(dry_run=True)
