#!/usr/bin/env python3
"""Analyze waggle run distribution by location in train/val splits."""
import sys, yaml
sys.path.insert(0, '/home/landgraf/workspaces/waggle_net/bb_wdd3')
import pandas as pd
from src.utils.video_utils import train_val_split_videos, get_base_video_stem

df = pd.read_csv('/home/landgraf/workspaces/waggle_net/bb_wdd3/data/annotations/fps_multires_clean.csv')
with open('/home/landgraf/workspaces/waggle_net/bb_wdd3/configs/config.yaml') as f:
    config = yaml.safe_load(f)

train_vids, val_vids = train_val_split_videos(df['video_name'].unique(), config['data']['train_ratio'], 42)
df['split'] = df['video_name'].apply(lambda v: 'train' if v in train_vids else 'val')
df['stem'] = df['video_name'].apply(get_base_video_stem)

def get_location(stem):
    if stem[0].isdigit(): return 'Berlin'
    elif stem.startswith('T'): return 'San Diego'
    elif stem.startswith('C') or stem.startswith('Capture'): return 'Jerusalem'
    return 'Unknown'

df['location'] = df['stem'].apply(get_location)

# Count unique waggle runs per location per split
# A waggle run is unique per (video_name, waggle_run_id) but we want per base stem
runs = df.groupby(['video_name', 'waggle_run_id', 'split', 'location', 'stem']).first().reset_index()

print("=== WAGGLE RUNS BY LOCATION AND SPLIT ===\n")
for split in ['train', 'val']:
    print(f"--- {split.upper()} ---")
    s = runs[runs['split'] == split]
    for loc in ['Berlin', 'San Diego', 'Jerusalem']:
        loc_runs = s[s['location'] == loc]
        n_runs = len(loc_runs)
        n_stems = loc_runs['stem'].nunique()
        n_videos = loc_runs['video_name'].nunique()
        print(f"  {loc:12s}: {n_runs:4d} runs across {n_stems:3d} base recordings ({n_videos} video variants)")
    print(f"  {'TOTAL':12s}: {len(s):4d} runs")
    print()

# Per-stem breakdown showing runs/recording
print("=== RUNS PER BASE RECORDING (train only) ===\n")
train_runs = runs[runs['split'] == 'train']
for loc in ['Berlin', 'San Diego', 'Jerusalem']:
    lr = train_runs[train_runs['location'] == loc]
    by_stem = lr.groupby('stem').size().describe()
    print(f"{loc}: {lr['stem'].nunique()} recordings, {len(lr)} total runs")
    print(f"  runs/recording: mean={by_stem['mean']:.1f}, min={by_stem['min']:.0f}, max={by_stem['max']:.0f}")
    # Are runs duplicated across resolution variants?
    by_video = lr.groupby('video_name').size()
    variants_per_stem = lr.groupby('stem')['video_name'].nunique()
    print(f"  avg variants/recording: {variants_per_stem.mean():.1f}")
    print()

# Show the imbalance
print("=== IMBALANCE SUMMARY (train split, all variants) ===\n")
for loc in ['Berlin', 'San Diego', 'Jerusalem']:
    n = len(train_runs[train_runs['location'] == loc])
    pct = n / len(train_runs) * 100
    print(f"  {loc:12s}: {n:5d} runs ({pct:5.1f}%)")
print(f"  {'TOTAL':12s}: {len(train_runs):5d}")
