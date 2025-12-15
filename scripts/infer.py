import os
import pandas as pd
import json
import torch
import numpy as np 
import time
import argparse

import glob
from src.models.r2plus1_yolo import R2Plus1D_YOLO
from src.evaluation import (
    process_video_with_detections, 
    batch_postprocess_predictions,
    get_metrics, 
    print_clean_comparison,
    process_video_with_detections_tiles
)
from src.utils import detections_to_df, df_to_detections,visualize_preds_and_gt

def append_metrics_to_csv(csv_path, row_dict):
    """
    Append a single row of metrics to CSV.
    Creates file with header if it does not exist.
    """
    df = pd.DataFrame([row_dict])

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, mode="w", header=True, index=False)

def parse_args():
    """Parse command line arguments for model evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate bee waggle dance detection model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model checkpoint file (.pth)')
    parser.add_argument('--epoch', type=int, required=True,
                       help='Epoch number of the model')
    parser.add_argument('--video_folder', type=str, required=True,
                       help='Directory containing evaluation videos')
    parser.add_argument('--gt_folder', type=str, required=True,
                       help='Directory containing ground truth annotations')
    parser.add_argument('--output_folder', type=str, required=True,
                       help='Directory to save evaluation results and predictions')
    
    # Optional arguments
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections')
    parser.add_argument('--stride', type=int, default=4,
                       help='Stride for sliding window evaluation')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference')
    
    # Evaluation thresholds
    parser.add_argument('--pos-thresholds', nargs='+', type=float, default=[30],
                       help='List of positional thresholds')
    parser.add_argument('--iou-thresholds', nargs='+', type=float, default=[0.5],
                       help='List of IoU thresholds')
    parser.add_argument('--angular-thresholds', nargs='+', type=float, default=[20],
                       help='List of angular thresholds')
    parser.add_argument('--consolidation-strategy', type=str, 
                       default='cluster_consolidate_v2',
                       help='Post-processing strategy')
    parser.add_argument('--resize', action='store_true',default=False,
                       help='Enable evaluation by resizing the input window to (224,224)')
    
    return parser.parse_args()


def load_model(model_path, device):
    """Load model from checkpoint with proper state dict handling."""
    model = R2Plus1D_YOLO()
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Clean state dict keys
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k
        # Remove "_orig_mod." prefix if present
        if k.startswith("_orig_mod."):
            new_key = k[len("_orig_mod."):]
        # Remove "module." prefix if not using DataParallel
        if not isinstance(model, torch.nn.DataParallel) and new_key.startswith("module."):
            new_key = new_key[len("module."):]
        cleaned[new_key] = v

    model.load_state_dict(cleaned)
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_single_video(model, device, video_path, video_name, epoch, 
                         gt_folder, output_folder, eval_config,
                         confidence=0.5, stride=4, resize=False):
    """
    Evaluate model on a single video.
    
    KEY DIFFERENCES FROM DATALOADER:
    - Uses process_video_with_detections (frame-by-frame)
    - Dataloader uses pre-windowed clips
    - Post-processing parameters must match exactly
    """
    
    epoch_folder = os.path.join(output_folder, f"epoch_{epoch}")
    os.makedirs(epoch_folder, exist_ok=True)
    output_video_path = os.path.join(epoch_folder, video_name)

    summary_csv = os.path.join(epoch_folder, "summary_metrics.csv")
    
    print(f"Processing video: {video_name}")
    print(f"  Confidence threshold: {confidence}")
    print(f"  Stride: {stride}")
    print(f"  Strategy: {eval_config['strategy']}")
    
    # Step 1: Get raw detections from video
    all_detections, fps = process_video_with_detections(
        model, video_path, device,
        window_size=16, 
        stride=stride, 
        confidence=confidence,
        resize=resize
    )
    
    # Save raw predictions before post-processing
    raw_df = detections_to_df(all_detections)
    raw_df.to_csv(
        output_video_path + "_preds_before.csv", index=False
    )

    if fps == 15:
        temporal_threshold=8
    elif fps==30:
        temporal_threshold=10
    elif fps==60:
        temporal_threshold=30
    else:
        temporal_threshold=20
    
    # Step 2: Post-process predictions
    processed = batch_postprocess_predictions(
        [all_detections],
        strategy=eval_config['strategy'],
        spatial_threshold=30,  
        temporal_threshold=temporal_threshold,
        confidence_threshold=confidence,
        mode='median'
    )
    
    # Save processed predictions
    processed_preds = processed[0]
    preds_df = detections_to_df(processed_preds)
    preds_df.to_csv(
        os.path.join(epoch_folder, f"{video_name}_preds_after.csv"),
        index=False
    )

    raw_df.to_csv(os.path.join(epoch_folder, f"{video_name}_preds_before.csv"),
        index=False
    )
    
    # Step 3: Load ground truth and evaluate
    pattern = os.path.join(gt_folder, f"{video_name}*.csv")
    matches = glob.glob(pattern)

    if len(matches) == 0:
        print(f"No ground truth found matching: {pattern}")
        print(f"Total detections: {len(preds_df)}")
        return None

    # Use the first matching file
    gt_path = matches[0]
    
    gt_df = pd.read_csv(gt_path)

    # results = get_metrics(preds_df, gt_df,max_spatial_dist=eval_config['pos_thresholds'][0])
    before_metrics = get_metrics(
        raw_df, gt_df,
        max_spatial_dist=eval_config["pos_thresholds"][0]
    )

    after_metrics = get_metrics(
        preds_df, gt_df,
        max_spatial_dist=eval_config["pos_thresholds"][0]
    )

    print("\n" + "="*60)
    print(f"EVALUATION RESULTS before postprocessing - {video_name}")
    print("="*60)
    print(f"True Positives: {before_metrics['True Positives']}")
    print(f"False Positives: {before_metrics['False Positives']}")
    print(f"False Negatives: {before_metrics['False Negatives']}")
    print(f"Precision: {before_metrics['precision']:.4f}")
    print(f"Recall: {before_metrics['recall']:.4f}")
    print(f"F1 Score: {before_metrics['f1']:.4f}")
    print(f"Total Detections: {len(all_detections)}")
    
    if before_metrics['spatial_error_mean'] is not None:
        print(f"\nSpatial Error: {before_metrics['spatial_error_mean']:.2f} ± {before_metrics['spatial_error_std']:.2f} pixels")
        print(f"Angular Error: {before_metrics['angular_error_mean']:.2f} ± {before_metrics['angular_error_std']:.2f} degrees")
        print(f"Temporal IoU: {before_metrics['temporal_iou_mean']:.4f}")
    print("="*60)

    print(f"EVALUATION RESULTS  - {video_name}")
    print("="*60)
    print(f"True Positives: {after_metrics['True Positives']}")
    print(f"False Positives: {after_metrics['False Positives']}")
    print(f"False Negatives: {after_metrics['False Negatives']}")
    print(f"Precision: {after_metrics['precision']:.4f}")
    print(f"Recall: {after_metrics['recall']:.4f}")
    print(f"F1 Score: {after_metrics['f1']:.4f}")
    
    if after_metrics['spatial_error_mean'] is not None:
        print(f"\nSpatial Error: {after_metrics['spatial_error_mean']:.2f} ± {after_metrics['spatial_error_std']:.2f} pixels")
        print(f"Angular Error: {after_metrics['angular_error_mean']:.2f} ± {after_metrics['angular_error_std']:.2f} degrees")
        print(f"Temporal IoU: {after_metrics['temporal_iou_mean']:.4f}")
    print("="*60)

    row = {
        "video_name": video_name,
        "fps": fps,

        "num_gt": len(gt_df),
        "num_preds_before": len(raw_df),
        "num_preds_after": len(preds_df),

        # BEFORE
        "TP_before": before_metrics["True Positives"],
        "FP_before": before_metrics["False Positives"],
        "FN_before": before_metrics["False Negatives"],
        "precision_before": before_metrics["precision"],
        "recall_before": before_metrics["recall"],
        "f1_before": before_metrics["f1"],
        "spatial_err_before": before_metrics.get("spatial_error_mean"),
        "angular_err_before": before_metrics.get("angular_error_mean"),
        "temporal_iou_before": before_metrics.get("temporal_iou_mean"),

        # AFTER
        "TP_after": after_metrics["True Positives"],
        "FP_after": after_metrics["False Positives"],
        "FN_after": after_metrics["False Negatives"],
        "precision_after": after_metrics["precision"],
        "recall_after": after_metrics["recall"],
        "f1_after": after_metrics["f1"],
        "spatial_err_after": after_metrics.get("spatial_error_mean"),
        "angular_err_after": after_metrics.get("angular_error_mean"),
        "temporal_iou_after": after_metrics.get("temporal_iou_mean"),

        # Config
        "confidence": confidence,
        "spatial_threshold": eval_config["pos_thresholds"][0],
        "temporal_threshold": temporal_threshold,
        "strategy": eval_config["strategy"],
    }

    append_metrics_to_csv(summary_csv, row)
    print(f"Metrics appended → {summary_csv}")

    visualize_preds_and_gt(
            video_path,
            preds_csv=preds_df,
            gt_csv=gt_df,
            output_path=output_video_path+"_visualised.mp4"
        )
    
    with open(output_video_path + "_detailed_metrics.json", "w") as f:
        json.dump(after_metrics, f, indent=4)
    
    print(f"Evaluation complete")

    return {
        "before": before_metrics,
        "after": after_metrics
    }


def evaluate(model_path, epoch, video_folder, gt_folder, output_folder, 
            eval_config, confidence=0.5, stride=4, batch_size=16, resize=False):
    """
    Main evaluation function.
    
    POTENTIAL ISSUES CAUSING DIFFERENT RESULTS:
    1. Window alignment: Video processing uses sliding windows, dataloader uses pre-extracted clips
    2. Frame indexing: Ensure start_frame/end_frame match between methods
    3. Post-processing: Must use identical parameters (spatial_threshold=30, temporal_threshold=8)
    4. Confidence filtering: Applied at different stages
    5. Resolution handling: Coordinate transformations must match
    """
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = load_model(model_path, device)
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Get video files
    video_files = [f for f in os.listdir(video_folder) 
                   if f.endswith(('.mp4', '.MP4', '.avi', '.mov'))]
    
    if not video_files:
        print(f"No video files found in {video_folder}")
        return {}
    
    print(f"\n{'='*70}")
    print(f"Found {len(video_files)} video(s) to process")
    print(f"{'='*70}")
    # print(f"Configuration:")
    # print(f"  - Confidence threshold: {confidence}")
    # print(f"  - Stride: {stride}")
    # print(f"  - Batch size: {batch_size}")
    # print(f"  - Strategy: {eval_config['strategy']}")
    # print(f"  - Spatial threshold: {eval_config['pos_thresholds'][0]}")
    # print(f"{'='*70}\n")
    
    # Process each video
    all_results = {}
    
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        start_time = time.time()
        
        result = evaluate_single_video(
            model, device, video_path, video_name, epoch, 
            gt_folder, output_folder, eval_config, 
            confidence=confidence, stride=stride, resize=resize
        )
        
        elapsed = time.time() - start_time
        print(f"Time: {elapsed:.2f}s\n")
        
        all_results[video_name] = result
    
    print(f"\n{'='*70}")
    print(f"Evaluation Complete")
    print(f"{'='*70}")
    print(f"Processed {len(video_files)} videos")
    print(f"Results saved to: {output_folder}")
    
    return all_results


if __name__ == "__main__":
    args = parse_args()
    
    # Build evaluation config
    # CRITICAL: These must match dataloader evaluation
    eval_config = {
        'pos_thresholds': args.pos_thresholds,
        'iou_threshold_range': tuple(args.iou_thresholds),
        'angular_thresholds': args.angular_thresholds,
        'confidence_threshold': args.confidence,
        'strategy': args.consolidation_strategy
    }
    
    print("\n" + "="*70)
    print("EVALUATION CONFIGURATION")
    print("="*70)
    for key, value in eval_config.items():
        print(f"{key:25s}: {value}")
    print("="*70 + "\n")
    
    # Run evaluation
    evaluate(
        model_path=args.model_path,
        epoch=args.epoch,
        video_folder=args.video_folder,
        gt_folder=args.gt_folder,
        output_folder=args.output_folder,
        eval_config=eval_config,
        confidence=args.confidence,
        stride=args.stride,
        batch_size=args.batch_size,
        resize=args.resize
    )