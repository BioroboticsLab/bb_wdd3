import argparse
import os
import torchvision.transforms as T
import pandas as pd

from src.data import VideoYoloDataset, WaggleAugmentations
from src.utils.visualisation import (
    visualize_gt_video_snippets,
    save_middle_frame_grid,
    create_dataset_heatmap,
    visualize_single_sample_side_by_side,
    visualize_preds_and_gt
)


def parse_args():
    parser = argparse.ArgumentParser(description="Flexible Visualization Tool")

    # Dataset + base config
    parser.add_argument("--annotations", type=str,
                        default="./data/annotations/fps_multires_full_data.csv")
    parser.add_argument("--data-dir", type=str, default="./data/resvideos")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--num-augs", type=int, default=3)
    parser.add_argument("--aug-prob", type=float, default=0.5)
    parser.add_argument("--output-dir", type=str, default="./visualizations")

    # Visualization toggles
    parser.add_argument("--vis-single", action="store_true",
                        help="Visualize a single dataset sample side-by-side.")
    parser.add_argument("--vis-snippets", action="store_true",
                        help="Visualize GT video snippets + augmentations.")
    parser.add_argument("--vis-heatmap", action="store_true",
                        help="Generate dataset heatmap.")
    parser.add_argument("--vis-pred-gt", action="store_true",
                        help="Visualize predictions + ground truth overlay on video.")

    # Extra args for prediction/GT visualization
    parser.add_argument("--pred-csv", type=str, default=None,
                        help="Path to predictions CSV file.")
    parser.add_argument("--gt-csv", type=str, default=None,
                        help="Path to ground truth CSV file.")
    parser.add_argument("--video-path", type=str, default=None,
                        help="Path to the input video.")
    parser.add_argument("--vis-output", type=str, default=None,
                        help="Output path for pred+GT visualization video.")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.vis_pred_gt:
        print("→ Visualizing predictions & GT on video...")

        if not args.pred_csv or not args.gt_csv or not args.video_path:
            raise ValueError(
                "For --vis-pred-gt you must provide --pred-csv, --gt-csv and --video-path."
            )

        pred_df = pd.read_csv(args.pred_csv)
        gt_df = pd.read_csv(args.gt_csv)

        out_path = args.vis_output or os.path.join(
            args.output_dir, "pred_gt_visualisation.mp4"
        )

        visualize_preds_and_gt(
            args.video_path,
            preds_csv=pred_df,
            gt_csv=gt_df,
            output_path=out_path
        )

        print(f" Pred+GT visualization saved to {out_path}")
    else:

        # Load annotations
        df = pd.read_csv(args.annotations)

        # Base transforms
        transforms = T.Compose([
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Augmentations if needed
        augmentation = WaggleAugmentations(
            width=224, height=224,
            prob=args.aug_prob,
            scale_range=(0.7, 1.3),
            mutual_exclusive=True,
            max_jitter_ratio=0.4
        )

        # Dataset
        dataset = VideoYoloDataset(
            df, args.data_dir, transforms,
            width=224, height=224,
            clip_len=16,
            grid_size=28,
            augment=None
        )

        # --------------------------------------------------
        # VISUALISATION OPTIONS
        # --------------------------------------------------

        if args.vis_single:
            print("→ Visualizing single sample side-by-side...")
            visualize_single_sample_side_by_side(dataset, args.sample_idx)
            print("✓ Saved single-sample visualisation")

        if args.vis_snippets:
            print(f"→ Visualizing sample {args.sample_idx} with {args.num_augs} augmentations...")
            visualize_gt_video_snippets(
                dataset=dataset,
                idx=args.sample_idx,
                grid_size=28,
                save_dir=args.output_dir,
                num_augs=args.num_augs
            )
            save_middle_frame_grid(args.output_dir)
            print(f"✓ Snippet visualizations saved to {args.output_dir}")

        if args.vis_heatmap:
            heatmap_path = os.path.join(args.output_dir, "dataset_heatmap.png")
            print("→ Creating dataset heatmap...")
            create_dataset_heatmap(dataset, output_path=heatmap_path)
            print(f"✓ Heatmap saved to {heatmap_path}")

           

    print("\nAll selected visualizations completed!")


if __name__ == "__main__":
    main()
