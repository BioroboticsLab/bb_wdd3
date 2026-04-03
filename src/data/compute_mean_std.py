"""
Compute per-channel mean and std over the full dataset for normalization.

Applies greyscale (prob=1.0) to match training distribution — the model will
always see greyscaled frames, so stats should reflect that. All other
augmentations (flips, rotation, blur, etc.) are disabled since they are
stochastic and dont affect the underlying pixel distribution meaningfully.

Run before training.
Usage:
    python -m compute_norm_stats --config_path ./configs/config.yaml
"""

import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor, Grayscale
from tqdm import tqdm
import torchvision.transforms as T
from src.data.dataset import TemporalWaggleCollator, VideoYoloDataset
from src.utils.data_utils import load_config, balance_sample
from src.utils.video_utils import get_video_category
import numpy as np 
import random 

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_args():
    parser = argparse.ArgumentParser(description="Compute dataset mean and std for normalization.")
    parser.add_argument("--config_path", type=str, default="./configs/config.yaml")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


def main():
    args = get_args()
    config = load_config(args.config_path)

    # Load data — same logic as main.py
    data = pd.read_csv(config['data']['annotations'])
    full_data_size = len(data)

    # Stem-based split to match training (prevents info-leak between variants)
    from src.utils.video_utils import train_val_split_videos
    train_videos, _ = train_val_split_videos(
        data['video_name'].unique(),
        train_ratio=config['data']['train_ratio'],
        seed=SEED
    )
    
    train_df = data[data['video_name'].isin(train_videos)].reset_index(drop=True)

    # Balance/subsample data
    if config['data']['data_fraction_divisor'] > 1:
        train_df = balance_sample(train_df, config['data']['data_fraction_divisor'])

    print(f"Computing stats over {len(train_df)} / {full_data_size} train vs. total samples.")

    # Raw resize + to tensor, no normalization
    raw_transform = Compose([
        ToPILImage(),
        Resize((config['augmentations']['width'], config['augmentations']['height'])),
        Grayscale(num_output_channels=3),
        ToTensor(),
    ])

    dataset = VideoYoloDataset(
        dataframe=train_df,
        video_dir=config['data']['data_dir'],
        transform=raw_transform,
        width=config['data']['width'],
        height=config['data']['height'],
        window_size=config['data']['window_size'],
        grid_size=config['model']['grid_size'],
        max_detections_per_cell=config['model']['max_detections_per_cell'],
        n_classes=config['model']['n_classes'],
        augment=None,
        is_training=False, 
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=TemporalWaggleCollator(),
        pin_memory=False,
        drop_last=False,
    )

    mean_accum = torch.zeros(3)
    sq_accum   = torch.zeros(3)
    n_pixels   = 0

    for batch in tqdm(loader, total=len(loader), desc="Computing stats"):
        video = batch['video'].float()  # (B, C, T, H, W)
        B, C, T, H, W = video.shape
        pixels = video.permute(1, 0, 2, 3, 4).reshape(C, -1)  # (C, N)
        mean_accum += pixels.sum(dim=1)
        sq_accum   += (pixels ** 2).sum(dim=1)
        n_pixels   += pixels.shape[1]

    mean = mean_accum / n_pixels
    std  = (sq_accum / n_pixels - mean ** 2).sqrt()

    print("\n" + "="*60)
    print("Summary stats — paste these into your config:")
    print(f"  mean: {[round(v, 4) for v in mean.tolist()]}")
    print(f"  std:  {[round(v, 4) for v in std.tolist()]}")
    print("="*60)

    # Sanity check: after greyscale all 3 channels should be identical
    channel_diff_mean = mean.max() - mean.min()
    channel_diff_std  = std.max()  - std.min()
    print(f"\nSanity check — channel spread (should be approx 0.0 after greyscale):")
    print(f"  mean channel spread: {channel_diff_mean:.4f}")
    print(f"  std  channel spread: {channel_diff_std:.4f}")


if __name__ == "__main__":
    main()