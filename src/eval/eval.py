import cv2
import torch
import numpy as np
import gc
from src.models.model import R2Plus1D_YOLO
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
import os
import json
from scipy.spatial.distance import euclidean
# from utils.nms import nms_spatiotemporal
from src.utils.frames_utils import sample_frames, preprocess_frames
from src.utils.video_utils import process_video_with_detections, visualize_preds_and_gt
from src.utils.data_utils import detections_to_df
from src.utils.metrics import match_detections_to_gt, get_metrics
from tqdm import tqdm
import wandb

def eval_single_video(model, device, video_path, video_name, epoch, gt_folder,
                         output_folder="eval_results", confidence=0.5,
                         max_spatial_dist=50.0, max_angular_error=45.0, min_temporal_iou=0.3):

    #print('Evaluate Single Vid - Start.')

    #print('Evaluate Single Vid - video_path:', video_path)
    #print('Evaluate Single Vid - video_name:', video_name)
    #print('Evaluate Single Vid - gt_folder:', gt_folder)

    epoch_folder = os.path.join(output_folder, f"epoch_{epoch}")
    os.makedirs(epoch_folder, exist_ok=True)
    #print('Evaluate Single Vid - epoch_folder:', epoch_folder)
    output_video_path = os.path.join(epoch_folder, video_name)
    #print('Evaluate Single Vid - output_video_path:', output_video_path)
    
    all_detections = process_video_with_detections(
        model, video_path, device, 
        window_size=16, stride=1, top_k=7, 
        output_path=output_video_path + "_before_post_process.mp4", 
        confidence=confidence
    )
    #print(f'Evaluate Single Vid - Preview all_detections: type {type(all_detections)} : {len(all_detections)}: {all_detections[:3]}')
    #print(f'Evaluate SIngle VId - All Detections: {all_detections}')
    clean_preds_df = detections_to_df(all_detections)
    #print(f'Evaluate Single Vid - Preview clean_preds_df: type {type(clean_preds_df)}')
    #if clean_preds_df is not None and not clean_preds_df.empty:
    #    print("Evaluate Single Vid - Preview clean_preds_df:")
    #    print(clean_preds_df.head())
    #else:
        #print("Evaluate Single Vid - clean_preds_df is empty or None")


    pred_save_path = output_video_path + "_preds.csv"
    clean_preds_df.to_csv(pred_save_path, index=False)
    #print(f"Predictions saved to {pred_save_path}")
    #print('Evaluate Single Vid - pred_save_path:', pred_save_path)


    gt_path = os.path.join(gt_folder, f"{video_name}.csv")
    #print('Evaluate Single Vid - gt_path:', gt_path)
    num_detections = len(clean_preds_df)
    avg_confidence = float(clean_preds_df['confidence'].mean()) if num_detections > 0 else 0.0
    
    if os.path.exists(gt_path):
        gt_df = pd.read_csv(gt_path)

        vis_output_path = output_video_path + "_with_gt.mp4"
        #print(f"'Evaluate Single Vid - Creating visualization with ground truth: {vis_output_path}")
        visualize_preds_and_gt(video_path, clean_preds_df, gt_df, vis_output_path)
        
        results = get_metrics(
            clean_preds_df, gt_df,
            max_spatial_dist=max_spatial_dist,
            max_angular_error=max_angular_error,
            min_temporal_iou=min_temporal_iou
        )

        with open(output_video_path + "_detailed_metrics.json", "w") as f:
            json.dump(results, f, indent=4)
        
        results['num_detections'] = num_detections
        results['avg_confidence'] = avg_confidence
        results['confidence_threshold'] = confidence

        print(f"Evaluating Single Video: {video_name}")
        print(f"True Positives: {results['True Positives']}")
        print(f"False Positives: {results['False Positives']}")
        print(f"False Negatives: {results['False Negatives']}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1']:.4f}")
        print(f"Total Detections: {num_detections}")
        print(f"Average Confidence: {avg_confidence:.4f}")
        
        if results['spatial_error_mean'] is not None:
            print(f"\nSpatial Error: {results['spatial_error_mean']:.2f} ± {results['spatial_error_std']:.2f} pixels")
            print(f"Angular Error: {results['angular_error_mean']:.2f} ± {results['angular_error_std']:.2f} degrees")
            print(f"Temporal IoU: {results['temporal_iou_mean']:.4f}")
        
        return results
    else:
        print(f"\nNo ground truth found for {video_name}. Only visualization generated.")
        print(f"Total detections found: {num_detections}")
        print(f"Average Confidence: {avg_confidence:.4f}")
        return None


#def evaluate(model_path, epoch, video_folder="/home/prajna/multiscale_wdd/data/eval_videos", gt_folder="/home/prajna/multiscale_wdd/data/eval_gt",
def evaluate(model_path, epoch, video_folder="/home/prajna/multiscale_wdd/data/videos", gt_folder="/home/deniz/waggle_dance_detector/dataset/eval_gt",
             output_folder="eval_results", confidence=0.5,
             max_spatial_dist=50.0, max_angular_error=45.0, min_temporal_iou=0.3, writer=None):
    #print('Eval.py - eval starting.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.8)
        print(f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    model = R2Plus1D_YOLO()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model = model.to(device)

    os.makedirs(output_folder, exist_ok=True)

    # video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.MP4', '.avi', '.mov'))]
    vid_name = '029_vid_downsample.mp4' 
    #vid_name = 'Capture_2025-04-29_14-59-11-1-00.00.01.000-00.02.00.688_downsampled_95.mp4'
    video_files = [
    f for f in os.listdir(video_folder)
    if f == vid_name
]
    #print('Eval.py - video_files.', video_files)
    #print('Eval.py - gt_folder.', gt_folder)


    if not video_files:
        print(f"No video files found in {video_folder}")
        return {}
    
    print(f"\nFound {len(video_files)} video(s) to process")
    
    all_results = {}
    
    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        #print('Eval.py - video_name.', video_name)

        #if video_name != "C1_11_03_25_my_video-2_d_output_40s":
        #    print('TRUE')
        #    continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {video_name}")
        print(f"{'='*60}")
        
        result = eval_single_video(
            model, device, video_path, video_name, epoch, gt_folder,
            output_folder, confidence, max_spatial_dist, max_angular_error, min_temporal_iou
        )
        
        all_results[video_name] = result
        
        if writer is not None and result is not None:
            writer.add_scalar(f'Eval/{video_name}/Precision', result['precision'], epoch)
            writer.add_scalar(f'Eval/{video_name}/Recall', result['recall'], epoch)
            writer.add_scalar(f'Eval/{video_name}/F1', result['f1'], epoch)
            writer.add_scalar(f'Eval/{video_name}/True_Positives', result['True Positives'], epoch)
            writer.add_scalar(f'Eval/{video_name}/False_Positives', result['False Positives'], epoch)
            writer.add_scalar(f'Eval/{video_name}/False_Negatives', result['False Negatives'], epoch)
            writer.add_scalar(f'Eval/{video_name}/Num_Detections', result['num_detections'], epoch)
            writer.add_scalar(f'Eval/{video_name}/Avg_Confidence', result['avg_confidence'], epoch)
            writer.add_scalar(f'Eval/{video_name}/Confidence_Threshold', result['confidence_threshold'], epoch)
            
            if result['spatial_error_mean'] is not None:
                writer.add_scalar(f'Eval/{video_name}/Spatial_Error_Mean', result['spatial_error_mean'], epoch)
                writer.add_scalar(f'Eval/{video_name}/Angular_Error_Mean', result['angular_error_mean'], epoch)
                writer.add_scalar(f'Eval/{video_name}/Temporal_IoU_Mean', result['temporal_iou_mean'], epoch)
    
    print("\n" + "="*60)
    print("SUMMARY OF ALL VIDEOS")
    print("="*60)
    
    evaluated_videos = {k: v for k, v in all_results.items() if v is not None}
    
    if evaluated_videos:
        avg_precision = np.mean([v['precision'] for v in evaluated_videos.values()])
        avg_recall = np.mean([v['recall'] for v in evaluated_videos.values()])
        avg_f1 = np.mean([v['f1'] for v in evaluated_videos.values()])
        avg_num_detections = np.mean([v['num_detections'] for v in evaluated_videos.values()])
        avg_confidence = np.mean([v['avg_confidence'] for v in evaluated_videos.values()])
        
        print(f"Videos evaluated with GT: {len(evaluated_videos)}")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1: {avg_f1:.4f}")
        print(f"Average Num Detections: {avg_num_detections:.2f}")
        print(f"Average Confidence: {avg_confidence:.4f}")
        
        if writer is not None:
            writer.add_scalar('Eval/Avg_Precision', avg_precision, epoch)
            writer.add_scalar('Eval/Avg_Recall', avg_recall, epoch)
            writer.add_scalar('Eval/Avg_F1', avg_f1, epoch)
            writer.add_scalar('Eval/Avg_Num_Detections', avg_num_detections, epoch)
            writer.add_scalar('Eval/Avg_Confidence', avg_confidence, epoch)
    
    print(f"Videos without GT: {len(all_results) - len(evaluated_videos)}")
    print("="*60)
    
    return all_results

def eval(model, device, yolocriterion, val_loader, epoch, ema):
    model.eval()
    total_loss = 0.0
    total_obj_loss = 0.0
    total_no_obj_loss = 0.0
    total_position_loss = 0.0
    total_direction_loss = 0.0
    total_temporal_loss = 0.0
    num_batches = 0

    # Apply EMA parameters for testing
    ema.apply_shadow()
    
    progress_bar = tqdm(val_loader, desc=f'Val Epoch {epoch+1}', leave=True, position=int(epoch))
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            inputs = batch["video"].to(device)
            targets = batch['targets'].to(device)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs)
                total_loss_batch, obj_loss, no_obj_loss, position_loss, direction_loss, temporal_loss = yolocriterion(outputs, targets)
            
            total_loss += total_loss_batch.item()
            total_obj_loss += obj_loss.item()
            total_no_obj_loss += no_obj_loss.item()
            total_position_loss += position_loss.item()
            total_direction_loss += direction_loss.item()
            total_temporal_loss += temporal_loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'Loss': f'{total_loss / num_batches:.4f}',
                'Obj': f'{total_obj_loss / num_batches:.4f}',
                'NoObj': f'{total_no_obj_loss / num_batches:.4f}',
                'Pos': f'{total_position_loss / num_batches:.4f}',
                'Dir': f'{total_direction_loss / num_batches:.4f}',
                'Temp': f'{total_temporal_loss / num_batches:.4f}',
            })
    
    # Calculate averages
    avg_val_loss = total_loss / num_batches
    avg_obj_loss = total_obj_loss / num_batches
    avg_no_obj_loss = total_no_obj_loss / num_batches
    avg_position_loss = total_position_loss / num_batches
    avg_direction_loss = total_direction_loss / num_batches
    avg_temporal_loss = total_temporal_loss / num_batches
    
    # Log to wandb (once per epoch, after all batches)
    wandb.log({
        'epoch': epoch,
        'val/total_loss': avg_val_loss,
        'val/object_loss': avg_obj_loss,
        'val/no_object_loss': avg_no_obj_loss,
        'val/position_loss': avg_position_loss,
        'val/direction_loss': avg_direction_loss,
        'val/temporal_loss': avg_temporal_loss
    })
    
    return avg_val_loss