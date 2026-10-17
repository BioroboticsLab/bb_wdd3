"""
Merge the fully-expanded new-batch annotations
(data/raw/annotations_fps_expanded.csv) into the main training file
(data/annotations/fps_multires_clean.csv).

Backs up the existing clean CSV first (mirrors the .bak convention used by
prepare_training_data.py) and refuses to merge if any video_name already
exists in the clean CSV, to avoid silently duplicating or corrupting
existing rows.

Usage:
    python scripts/merge_new_annotations.py
"""
import os
import shutil

import pandas as pd

NEW_PATH = "./data/raw/annotations_fps_expanded.csv"
CLEAN_PATH = "./data/annotations/fps_multires_clean.csv"
BACKUP_PATH = "./data/annotations/fps_multires_clean.csv.bak"


def main():
    new_df = pd.read_csv(NEW_PATH)
    clean_df = pd.read_csv(CLEAN_PATH)

    overlap = set(new_df["video_name"]) & set(clean_df["video_name"])
    if overlap:
        raise SystemExit(
            f"Refusing to merge: {len(overlap)} video_name(s) already present "
            f"in {CLEAN_PATH}, e.g. {sorted(overlap)[:5]}"
        )

    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(CLEAN_PATH, BACKUP_PATH)
        print(f"Backed up {CLEAN_PATH} -> {BACKUP_PATH}")
    else:
        print(f"Backup already exists at {BACKUP_PATH}, not overwriting")

    merged = pd.concat([clean_df, new_df], ignore_index=True)
    merged.to_csv(CLEAN_PATH, index=False)

    print(f"Merged {len(new_df):,} new rows into {CLEAN_PATH}")
    print(f"  Before: {len(clean_df):,} rows, {clean_df['video_name'].nunique()} videos")
    print(f"  After:  {len(merged):,} rows, {merged['video_name'].nunique()} videos")


if __name__ == "__main__":
    main()
