#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from src.utils.video_utils import train_val_split_videos
df = pd.read_csv("data/annotations/fps_multires_clean.csv")
for i in range(5):
    t, v = train_val_split_videos(df["video_name"].unique(), 0.9, 42)
    print(f"Run {i+1}: {len(t)} train / {len(v)} val")
