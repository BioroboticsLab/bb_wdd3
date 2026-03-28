import os 
from src.utils.data_utils import create_video_frames_df
import random
import numpy as np
import torch
import torchvision.transforms as T
import pandas as pd
from src.train.train import train, train
from src.eval.eval import eval
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.data.dataset import VideoYoloDataset, TemporalWaggleCollator
from src.data.dataset_tempaug import VideoYoloDatasetTemporalJitter
from src.models.model import R2Plus1D_YOLO_MultiHead
from src.loss.loss import WaggleDetectionLoss
from src.loss.loss_new import WaggleDetectionLoss_New
from src.data.augmentation import WaggleAugmentations
from src.tests.aug_vis import demo_visualization
import torch.nn  as nn
from src.utils.data_utils import fix_dataframe_with_video_lengths, find_overlapping_rows, preds_to_df, save_preds_to_csv, load_config, balance_sample
import datetime
from torch.utils.tensorboard import SummaryWriter
from src.utils.eval_utils import get_preds_gt, yolo_to_img_space, yolo_to_img_space_gt, get_eval_metrics, print_evaluation_results
import argparse
from src.utils.vis_utils import reverse_transform_batch, save_frames
from src.utils.postprocess import batch_postprocess_predictions
from src.utils.video_utils import frames_to_video
from src.utils.draw_utils import  draw_waggle, draw_waggle_batch, draw_waggle_batch_union
from src.utils.model_utils import load_pretrained_model, EMA
from src.utils.video_utils import get_video_category
import wandb
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(args):
    config = load_config(args.config_path)

    num_epochs = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        use_multi_gpu = True
    else:
        print("Using single GPU")
        use_multi_gpu = False

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", current_time)
    writer = SummaryWriter(log_dir=log_dir)

    data = pd.read_csv(config['data']['annotations'])
    full_data_size = len(data)

    # Stratified video-level split:
    # all windows from a given video go entirely to train or test never both
    # split is done per category so each group contributes proportionally to both splits 
    # regardless of how many videos it has
    video_df = pd.DataFrame({'video_name': data['video_name'].unique()})
    video_df['category'] = video_df['video_name'].apply(get_video_category)

    train_videos = set()
    for category, group in video_df.groupby('category'):
        vids = np.random.RandomState(SEED).permutation(group['video_name'].values)
        n_train = int(config['data']['train_ratio'] * len(vids))
        train_videos.update(vids[:n_train])
    
    test_df  = data[~data['video_name'].isin(train_videos)].reset_index(drop=True)

    test_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((config['augmentations']['width'], 
                  config['augmentations']['height'])),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize(mean=config['augmentations']['mean'], 
                    std=config['augmentations']['std'])])
    
    test_dataset = VideoYoloDataset(
        test_df,
        config['data']['data_dir'],
        test_transform,
        width=config['data']['width'],
        height=config['data']['height'],
        window_size=config['data']['window_size'],
        grid_size=config['model']['grid_size'],
        max_detections_per_cell=config['model']['max_detections_per_cell'],
        n_classes=config['model']['n_classes'],
        augment=None,
        is_training=False 
        )
        
    collator = TemporalWaggleCollator()

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['eval']['batch_size'],
        collate_fn=collator, 
        shuffle=False, 
        num_workers=config['train']['num_workers'],
        persistent_workers=(config['train']['num_workers'] > 0),
        pin_memory=True
    )

    model, checkpoint = load_pretrained_model(args.ckpt_path, config, device)
    ema = EMA(model, decay=config['train']['ema_decay'], device=device)

    # for ema
    if 'ema_state_dict' in checkpoint:
        ema.load_state_dict(checkpoint['ema_state_dict'])

    yolocriteria = WaggleDetectionLoss(lambda_obj=config["loss"]["lambda_obj"], 
                                       lambda_coord=config["loss"]["lambda_coord"], 
                                       lambda_noobj=config["loss"]["lambda_noobj"], 
                                       lambda_direction=config["loss"]["lambda_direction"], 
                                       lambda_temporal=config["loss"]["lambda_temporal"],
                                       use_varifocal=config["loss"]["use_varifocal"],
                                       gamma=config["loss"]["varifocal_gamma"],
                                       quality_decay=config["loss"]["quality_decay"],
                                       grid_size=config["loss"]["grid_size"],
                                       input_size=config["loss"]["input_size"])
    
    scaler = torch.amp.GradScaler()
    
    wandb.init(mode='disabled')

    # swap in ema weights
    ema.apply_shadow() 

    for epoch in range(num_epochs):
        #train_loss =  eval(model, device, yolocriteria, train_loader, epoch, writer)
        test_loss = eval(model, device, yolocriteria, test_loader, epoch)
        # Its not possible to fit all training or test frames onto cpu for visualisations
        # batch_idx_for_frames is set to 0 indicating that it will index into the first batch of the entire data loader and store the frames in there
        # if batch_size is set to 16, that means we have 16*window_size frames in our case 16 * 16, each individual batch represents a single waggle dance event of 16 frames
        # with 16 batches that is 16 * 16
        test_preds_raw, test_gt_raw, test_all_starts, test_all_ends, _ , test_frames, _ = get_preds_gt(model, 
                                                                                                       test_loader, 
                                                                                                       device, 
                                                                                                       return_frames=True, 
                                                                                                batch_idx_for_frames=0)
        # denorms imgs
        test_frames = reverse_transform_batch(test_frames, 
                                              original_size=(224,224))

        # visualize fetched test frames as a video if you want
        #frames_to_video(test_frames, output_name= 'data_loader_batches_0')
        
        # Transform yolo coordinates onto image domain for both gt and predicted values
        test_gts = yolo_to_img_space_gt(test_gt_raw, 
                                        all_starts=test_all_starts, 
                                        all_ends=test_all_ends,
                                        window_size = config['data']['window_size'],
                                        original_size=(config['data']['width'],
                                                       config['data']['height']))
        
        test_preds  = yolo_to_img_space(test_preds_raw, all_starts=test_all_starts, all_ends=test_all_ends, 
                                        confidence_threshold=config['eval']['confidence_threshold'], 
                                        window_size = config['data']['window_size'],
                                        original_size=(config['data']['width'],
                                                       config['data']['height']),
                                                       max_dets=config['eval']['max_dets'])
        
        # use raw tensor BEFORE max_dets and BEFORE sigmoid — this shows true distribution
        raw_confs = torch.sigmoid(test_preds_raw[..., 0]).flatten().cpu().numpy()

        # also get post-filter confidences from the dict list
        filtered_confs = np.array([
            pred['confidence'] 
            for sample in test_preds 
            for pred in sample
        ])

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # full raw distribution — all 784 cells per clip
        axes[0].hist(raw_confs, bins=50, edgecolor='black')
        axes[0].set_title('Confidence across all cells per clip')
        axes[0].set_xlabel('Confidence')
        axes[0].set_ylabel('Count')
        axes[0].axvline(raw_confs.mean(), color='red', linestyle='--', 
                        label=f'mean={raw_confs.mean():.3f}')
        axes[0].legend()

        # zoomed into high confidence region
        axes[1].hist(raw_confs, bins=50, range=(0.9, 1.0), edgecolor='black')
        axes[1].set_title('Confidences at 0.9 to 1.0 region')
        axes[1].set_xlabel('Confidence')
        axes[1].set_ylabel('Count')

        # post-filter distribution (top max_dets per clip)
        axes[2].hist(filtered_confs, bins=50, edgecolor='black')
        axes[2].set_title(f'After max_dets={config["eval"]["max_dets"]} filter')
        axes[2].set_xlabel('Confidence')
        axes[2].set_ylabel('Count')
        axes[2].axvline(filtered_confs.mean(), color='red', linestyle='--',
                        label=f'mean={filtered_confs.mean():.3f}')
        axes[2].legend()

        n_raw = len(raw_confs)
        n_filtered = len(filtered_confs)
        n_gt = sum(len(g) for g in test_gts)

        plt.suptitle(
            f'Predictions: {n_raw} | After filter: {n_filtered} | GTs: {n_gt} | '
            f'Ratio GT vs. predictions: {100*n_gt/n_raw:.3f}% | After filter: {100*n_gt//n_filtered:.3f}%'
        )

        plt.tight_layout()
        plt.savefig('./outputs/confidence_histogram.png', dpi=150, bbox_inches='tight')
        plt.show()

        n_empty_gts = sum(1 for g in test_gts if len(g) == 0)
        n_nonempty_gts = sum(1 for g in test_gts if len(g) > 0)
        print(f"Clips with GT:    {n_nonempty_gts}")
        print(f"Clips without GT: {n_empty_gts}")
        print(f"Total GTs:        {sum(len(g) for g in test_gts)}")

        print(f"Raw cells:        {n_raw}")
        print(f"Filtered cells:   {n_filtered}")
        print(f"Ground truths:    {n_gt}")
        print(f"Positive rate:    {100*n_gt/n_raw:.3f}%")
        print(f"\nRaw confidence:")
        print(f"  mean:    {raw_confs.mean():.4f}")
        print(f"  std:     {raw_confs.std():.4f}")
        print(f"  >0.99:   {(raw_confs > 0.99).mean():.4f}")
        print(f"  >0.5:    {(raw_confs > 0.5).mean():.4f}")
        print(f"  >0.1:    {(raw_confs > 0.1).mean():.4f}")
        print(f"\nFiltered confidence:")
        print(f"  mean:    {filtered_confs.mean():.4f}")
        print(f"  std:     {filtered_confs.std():.4f}")
        print(f"  >0.99:   {(filtered_confs > 0.99).mean():.4f}")

        # check how many clips pass each threshold independently
        n_spatial_pass = 0
        n_temporal_pass = 0
        n_angular_pass = 0
        n_all_pass = 0
        n_clips_with_preds = 0

        for sample_preds, sample_gts in zip(test_preds, test_gts):
            if not sample_preds or not sample_gts:
                continue
            
            n_clips_with_preds += 1
            pred = sample_preds[0]  # top prediction
            gt = sample_gts[0]      # only GT
            
            pos_dist = np.linalg.norm(np.array(gt['position']) - np.array(pred['position']))
            
            gt_s, gt_e = gt['temporal_offsets']
            pr_s, pr_e = pred['temporal_offsets']
            inter = max(0, min(gt_e, pr_e) - max(gt_s, pr_s))
            union = max(gt_e, pr_e) - min(gt_s, pr_s)
            tiou = inter / union if union > 0 else 0
            
            gt_d = np.array(gt['direction'])
            pr_d = np.array(pred['direction'])
            gt_d = gt_d / (np.linalg.norm(gt_d) + 1e-8)
            pr_d = pr_d / (np.linalg.norm(pr_d) + 1e-8)
            ang_err = np.degrees(np.arccos(np.clip(np.dot(gt_d, pr_d), -1, 1)))
            
            if pos_dist <= 20:   n_spatial_pass += 1
            if tiou >= 0.5:      n_temporal_pass += 1
            if ang_err <= 20:    n_angular_pass += 1
            if pos_dist <= 20 and tiou >= 0.5 and ang_err <= 15:
                n_all_pass += 1

        print(f"Clips with predictions: {n_clips_with_preds}")
        print(f"Pass spatial  (<=20px): {n_spatial_pass} ({100*n_spatial_pass/n_clips_with_preds:.1f}%)")
        print(f"Pass temporal (>=0.5):  {n_temporal_pass} ({100*n_temporal_pass/n_clips_with_preds:.1f}%)")
        print(f"Pass angular  (<=15°):  {n_angular_pass} ({100*n_angular_pass/n_clips_with_preds:.1f}%)")
        print(f"Pass ALL three:         {n_all_pass} ({100*n_all_pass/n_clips_with_preds:.1f}%)")
        # Draw gt and predictions onto frames and saves as video
        # Note: This shows each 16-frame window independently, so frames repeat at window intersections
        #draw_waggle_batch(
        #   all_frames=test_frames,
        #    all_detections=test_preds, 
        #    all_ground_truths=test_gts,
        #    all_start_frame_idxs=test_all_starts,
        #   output_dir='./outputs/vids/pre',
        #    draw_gt=False, draw_pred=True,
        #    save_clean_frames=True, save_annotated_frames=True)

        # This creates a continuous timeline without repeating frames
        # We use [:16] because test_frames only contains the first 16 sequences (batch #0),
        # while test_preds/test_gts contain predictions for all 964 sequences in the test set
        #draw_waggle_batch_union(
        #    all_frames=test_frames,
        #    all_detections=test_preds[:16],  # Only first 16 sequences (matches our saved frames)
        #    all_ground_truths=test_gts[:16], # Only first 16 sequences  
        #    all_start_frame_idxs=test_all_starts[:16],  # Only first 16 start indices
        #    output_dir='./outputs/vids'
        #)

        # Post Process all predictions
        # You can try different strategies if you want but, only cluster_consolidate is important for our purpose. 
        # 'cluster_consolidate' is DBSCAN across spatial and temporal aspect. 4 strategies are supported you can have a look if you want. 
        # mode supporst mean or median, meaning it will compute a point that summarises a cluster based on median or mean summary. 
        post_test_preds = batch_postprocess_predictions(test_preds, 
                                                        spatial_threshold=config['post_process']['spatial_threshold'], 
                                                        temporal_threshold=config['post_process']['temporal_threshold'], 
                                                        confidence_threshold=config['post_process']['confidence_threshold'], 
                                                        strategy=config['post_process']['strategy'], 
                                                        mode=config['post_process']['mode'],
                                                        remove_outliers=config['post_process']['outlier_detection'],
                                                        outlier_method='isolation_forest')

        # Can postprocess entire predictions no need for limit to 16 sequences, its only needed when we visualise
        #save_preds_to_csv(test_preds, f'raw_predictions_epoch_{epoch}.csv', 'raw', './outputs/preds_csv')
        #save_preds_to_csv(post_test_preds, f'postprocessed_predictions_epoch_{epoch}.csv', 'postprocessed', './outputs/preds_csv')

        # visualise and store postprocessed results
        #draw_waggle_batch(
        #    all_frames=test_frames,
        #    all_detections=post_test_preds, 
        #    all_ground_truths=test_gts,
        #    all_start_frame_idxs=test_all_starts,
        #    output_dir='./outputs/vids/post',
        #    draw_gt=False, draw_pred=True,
        #    save_clean_frames=True, save_annotated_frames=True)

        # Print unique clusters identifies
        #unique_clusters_all = len({det['cluster_id'] for seq in post_test_preds for det in seq})
        #unique_clusters_frames = len({det['cluster_id'] for seq in post_test_preds[:16] for det in seq})

        #print('Number of clusterns - Entire data loader:', unique_clusters_all)
        #print('Number of clusterns - Frames used for visualisation:', unique_clusters_frames)

        # Visualise post processed results
        #draw_waggle_batch_union(
        #    all_frames=test_frames,
        #    all_detections=post_test_preds[:16], 
        #    all_ground_truths=test_gts[:16],
        #    all_start_frame_idxs=test_all_starts,
        #    file_name='dbscan',
        #    output_dir='./outputs/vids'
        #)

        # Compute eval metrics
        test_metrics = get_eval_metrics(test_preds, test_gts, 
                                        pos_thresholds=config['eval']['pos_thresholds'],
                                        iou_threshold_range=config['eval']['iou_thresholds'],
                                        angular_thresholds=config['eval']['angular_thresholds'],
                                        match_pairs=config['eval']['match_pairs'])
        
        post_test_metrics = get_eval_metrics(post_test_preds, test_gts, 
                                        pos_thresholds=config['eval']['pos_thresholds'],
                                        iou_threshold_range=config['eval']['iou_thresholds'],
                                        angular_thresholds=config['eval']['angular_thresholds'],
                                        match_pairs=config['eval']['match_pairs'])
        
        print_evaluation_results(test_metrics, post_test_metrics)

        # Restore original parameters after metrics
        ema.restore()

    print('Eval on single checkpoint complete.')
    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description="Waggle detection training")

    parser.add_argument("--config_path", type=str, default='./configs/config.yaml', help="Path to the config file.")
    parser.add_argument("--ckpt_path", type=str, default='./ckpt/best_model.pth', help="Path to the ckpt file.")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)