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
from src.data.augmentation import WaggleAugmentations
from src.utils.data_utils import load_config, balance_sample


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

    if config['data']['data_fraction_divisor'] == 1:
        data = data.iloc[:len(data) // config['data']['data_fraction_divisor']].reset_index(drop=True)
    elif config['data']['data_fraction_divisor'] > 1:
        data = balance_sample(data, config['data']['data_fraction_divisor'])

    print(f"Computing stats over {len(data)} / {full_data_size} samples.")

    # Raw resize + to tensor, no normalization
    raw_transform = Compose([
        ToPILImage(),
        Resize((config['augmentations']['width'], config['augmentations']['height'])),
        Grayscale(num_output_channels=3),
        ToTensor(),
    ])

    # Greyscale only — all stochastic augmentations off.
    # prob_greyscale=1.0 because the model always sees greyscaled frames at train time,
    # so the normalization stats should reflect that distribution.
    # normalize=False because we are computing the stats to use for normalization.
    greyscale_only_aug = WaggleAugmentations(
        width=config['augmentations']['width'],
        height=config['augmentations']['height'],
        prob_flip_h=0.0,
        prob_flip_v=0.0,
        prob_rotate=0.0,
        prob_scale=0.0,
        prob_translate=0.0,
        prob_hsv=0.0,
        prob_brightness=0.0,
        prob_contrast=0.0,
        prob_gamma=0.0,
        prob_blur=0.0,
        prob_clahe=0.0,
        prob_color_shuffle=0.0,
        prob_posterize=0.0,
        prob_greyscale=1.0,
        normalize=False,
    )

    dataset = VideoYoloDataset(
        dataframe=data,
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
    std_accum  = torch.zeros(3)
    n_batches  = len(loader)

    for batch in tqdm(loader, total=n_batches, desc="Computing stats"):
        video = batch['video'].float()  # (B, C, T, H, W)
        B, C, T, H, W = video.shape
        pixels = video.permute(1, 0, 2, 3, 4).reshape(C, -1)  # (C, N)
        mean_accum += pixels.mean(dim=1)
        std_accum  += pixels.std(dim=1)

    mean = mean_accum / n_batches
    std  = std_accum  / n_batches

    print("\n" + "="*60)
    print("Summary stats — paste these into your config:")
    print(f"  mean: {[round(v, 4) for v in mean.tolist()]}")
    print(f"  std:  {[round(v, 4) for v in std.tolist()]}")
    print("="*60)

    # Sanity check: after greyscale all 3 channels should be identical
    channel_diff_mean = mean.max() - mean.min()
    channel_diff_std  = std.max()  - std.min()
    print(f"\nSanity check — channel spread (should be ~0.0 after greyscale):")
    print(f"  mean channel spread: {channel_diff_mean:.4f}")
    print(f"  std  channel spread: {channel_diff_std:.4f}")


if __name__ == "__main__":
    main()