RamanLab-Locust-Behavior

This repository contains utilities to standardize and clean video files under `Locust/all_vids`.

Scripts

- `rename_videos.py`
  - Walks `Locust/all_vids` and creates standardized filenames from path components.
  - Dry-run (preview): `python3 rename_videos.py`
  - Execute: `python3 rename_videos.py --execute`

- `move_videos_to_date.py`
  - Moves renamed videos up into their parent date folder (e.g. `.../08.26.2025/filename.mp4`).
  - Dry-run: `python3 move_videos_to_date.py`
  - Execute: `python3 move_videos_to_date.py --execute`

- `fix_duplicate_names.py`
  - Fixes filenames that accidentally contain duplicated trailing segments (e.g. `...mp4_Trial_1_Recording.mp4`).
  - Dry-run: `python3 fix_duplicate_names.py`
  - Execute: `python3 fix_duplicate_names.py --execute`

- `cleanup_keep_dates.py`
  - Moves any videos inside nested subfolders up to the date folder and removes all other subfolders and non-video files under each date.
  - Dry-run: `python3 cleanup_keep_dates.py`
  - Execute: `python3 cleanup_keep_dates.py --execute`

- `delete_palps.py`
  - Finds and deletes files that end with `Recording_palps_annotated_30fps.mp4`.
  - Dry-run: `python3 delete_palps.py`
  - Execute: `python3 delete_palps.py --execute`

Notes & Safety

- All scripts default to a dry-run mode that prints planned actions without making changes. Always run dry-runs first and verify output before using `--execute`.
- The scripts assume the repository root path; adjust `BASE_DIR` at the top of each script if you move them.
- A `.gitignore` has been added to ignore the `Locust/` folder to avoid checking large video data into git.
- Back up your data before running destructive operations (`--execute`).

If you want, I can:
- Run a specific dry-run or execute one of the scripts now.
- Add a single command `make` target or small `requirements.txt` if we add dependencies.
