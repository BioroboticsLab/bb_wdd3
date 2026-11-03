"""
Merge the fully-expanded new-batch annotations
(data/raw/<BATCH>_annotations_fps_expanded.csv) into the main training
file (data/annotations/fps_multires_clean.csv).

BATCH is imported from convert_raw_annotations.py (the single source of
truth for which batch is currently being processed).

Backs up the existing clean CSV first (mirrors the .bak convention used by
prepare_training_data.py). Since convert_raw_annotations.py and the
expansion scripts always reprocess every clean raw file in the batch from
scratch (not just newly-fixed ones), the fps-expanded CSV will often
contain a mix of videos already merged in a previous run alongside
genuinely new ones (e.g. after fixing one more bad CSV in
data/raw/annotations/<BATCH>/). Rows for video_names already present in
the clean CSV are skipped (not re-merged, not treated as an error); only
rows for video_names not yet in the clean CSV get appended.

Usage:
    python scripts/merge_new_annotations.py
"""
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from convert_raw_annotations import BATCH

NEW_PATH = f"./data/raw/{BATCH}_annotations_fps_expanded.csv"
CLEAN_PATH = "./data/annotations/fps_multires_clean.csv"
BACKUP_PATH = "./data/annotations/fps_multires_clean.csv.bak"


def main():
    new_df = pd.read_csv(NEW_PATH)
    clean_df = pd.read_csv(CLEAN_PATH)

    already_merged = set(new_df["video_name"]) & set(clean_df["video_name"])
    to_add = new_df[~new_df["video_name"].isin(already_merged)]

    if already_merged:
        print(f"Skipping {len(already_merged)} video(s) already merged previously, "
              f"e.g. {sorted(already_merged)[:5]}")

    if to_add.empty:
        print("Nothing new to merge -- every video in "
              f"{NEW_PATH} is already in {CLEAN_PATH}.")
        return

    if not os.path.exists(BACKUP_PATH):
        shutil.copy2(CLEAN_PATH, BACKUP_PATH)
        print(f"Backed up {CLEAN_PATH} -> {BACKUP_PATH}")
    else:
        print(f"Backup already exists at {BACKUP_PATH}, not overwriting")

    merged = pd.concat([clean_df, to_add], ignore_index=True)
    merged.to_csv(CLEAN_PATH, index=False)

    print(f"Merged {len(to_add):,} new rows ({to_add['video_name'].nunique()} videos) "
          f"into {CLEAN_PATH}")
    print(f"  Before: {len(clean_df):,} rows, {clean_df['video_name'].nunique()} videos")
    print(f"  After:  {len(merged):,} rows, {merged['video_name'].nunique()} videos")


if __name__ == "__main__":
    main()
