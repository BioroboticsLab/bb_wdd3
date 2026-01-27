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
from src.models.model import R2Plus1D_YOLO
#from src.models.model_multihead import R2Plus1D_YOLO_MultiHead
from src.models.model_multihead_deeper_heads import R2Plus1D_YOLO_MultiHead

from src.loss.loss import WaggleDetectionLoss
from src.loss.loss_new import WaggleDetectionLoss_New
from src.data.augmentation import WaggleAugmentations
from src.tests.aug_vis import demo_visualization
import torch.nn  as nn
from src.utils.data_utils import fix_dataframe_with_video_lengths, find_overlapping_rows, preds_to_df, save_preds_to_csv, load_config
import datetime
from torch.utils.tensorboard import SummaryWriter
from src.utils.eval_utils import get_preds_gt, yolo_to_img_space, yolo_to_img_space_gt, get_eval_metrics, print_evaluation_results
import argparse
from src.utils.vis_utils import reverse_transform_batch, save_frames
from src.utils.nms import batch_postprocess_predictions
from src.utils.video_utils import frames_to_video
from src.utils.draw_utils import  draw_waggle, draw_waggle_batch, draw_waggle_batch_union
from src.utils.model_utils import load_pretrained_model
import wandb

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

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", current_time)
    writer = SummaryWriter(log_dir=log_dir)

    data = pd.read_csv(config['data']['annotations'])
    print(f"Original dataset length: {len(data)}")
    # 1/8 of original data for fine-tuning
    data = data.iloc[:len(data)//8].reset_index(drop=True)
    #data = data.iloc[:100].reset_index(drop=True)
    print(f"After subsetting dataset: {len(data)} samples")

    transforms = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        ])
    
    test_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])])
    
    train_augmentation = WaggleAugmentations(
        width=224, height=224, 
        prob_flip_h=0.5, prob_flip_v=0.0, 
        prob_rotate=0.3, rotate_range=(-25, 25), 
        prob_scale=1.0, scale_range=(0.9, 1.1),
        prob_translate=0.3, translate_range=0.1,
        prob_hsv=0.0, hsv_hue=0.1, hsv_saturation=0.9, hsv_value=0.9,
        prob_brightness=1.0, brightness_range=0.4, 
        prob_contrast=1.0, contrast_range=0.4,
        prob_gamma=0.0, gamma_range=(0.8, 1.2),
        prob_blur=0.1, blur_range=(0.5, 2.0),
        prob_clahe=0.1, clahe_clip_limit=2.0, clahe_tile_grid_size=(8, 8),
        prob_color_shuffle=0.0,
        prob_posterize=0.0, posterize_bits=(4, 7),
        prob_greyscale=0.0,
        normalize=True,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        debug=False)

    total_len = len(data)
    train_len = int(0.8 * total_len)

    train_indices = list(range(train_len))
    test_indices = list(range(train_len, total_len))

    train_df = data.iloc[train_indices].reset_index(drop=True)
    test_df = data.iloc[test_indices].reset_index(drop=True)
    # sort by video name and start frames to recover chronological/sequential order
    train_df = train_df.sort_values(['video_name', 'start_frame']).reset_index(drop=True)
    test_df = test_df.sort_values(['video_name', 'start_frame']).reset_index(drop=True)


    # filter out non-overlapping windows
    #test_df = find_overlapping_rows(test_df)
    #print('Find Overlapping')
    

    train_dataset = VideoYoloDataset(
        train_df,
        config['data']['data_dir'],
        transforms,
        width=224,
        height=224,
        clip_len=16,
        grid_size=28,
        max_detections_per_cell=1,
        num_classes=1,
        augment=train_augmentation,
        is_training=True
    )

    test_dataset = VideoYoloDataset(
        test_df,
        config['data']['data_dir'], 
        test_transform,
        width=224,
        height=224,
        clip_len=16,
        grid_size=28,
        max_detections_per_cell=1,
        num_classes=1,
        augment=False, 
        )
        
    collator = TemporalWaggleCollator()

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train']['batch_size'],
        collate_fn=collator, 
        shuffle=True,
        num_workers=config['train']['num_workers'],
        persistent_workers=(config['train']['num_workers'] > 0),
        pin_memory=True,  
        drop_last=True   
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['eval']['batch_size'],
        collate_fn=collator, 
        shuffle=False, 
        num_workers=config['train']['num_workers'],
        persistent_workers=(config['train']['num_workers'] > 0),
        pin_memory=True
    )


    model = load_pretrained_model(args.ckpt_path, device)
    yolocriteria = WaggleDetectionLoss_New(lambda_obj=config["loss"]["lambda_obj"], 
                                       lambda_coord=config["loss"]["lambda_coord"], 
                                       lambda_noobj=config["loss"]["lambda_noobj"], 
                                       lambda_direction=config["loss"]["lambda_direction"], 
                                       lambda_temporal=config["loss"]["lambda_temporal"],
                                       use_varifocal=config["loss"]["use_varifocal"],
                                       gamma=config["loss"]["varifocal_gamma"],
                                       quality_scale=config["loss"]["varifocal_quality_scale"])
    
    wandb.init(mode='disabled')

    for epoch in range(num_epochs):
        #train_loss =  eval(model, device, yolocriteria, train_loader, epoch, writer)
        test_loss = eval(model, device, yolocriteria, test_loader, epoch)
        # Its not possible to fit all training or test frames onto cpu for visualisations
        # batch_idx_for_frames is set to 0 indicating that it will index into the first batch of the entire data loader and store the frames in there
        # if batch_size is set to 16, that means we have 16*window_size frames in our case 16 * 16, each individual batch represents a single waggle dance event of 16 frames
        # with 16 batches that is 16 * 16
        test_preds_raw, test_gt_raw, test_all_starts, test_all_ends, _ , test_frames, _ = get_preds_gt(model, test_loader, device, return_frames=True, batch_idx_for_frames=0)
        # denorms imgs
        test_frames = reverse_transform_batch(test_frames, original_size=(224,224))

        # visualize fetched test frames as a video if you want
        frames_to_video(test_frames, output_name= 'data_loader_batches_0')
        
        # Transform yolo coordinates onto image domain for both gt and predicted values
        test_gts = yolo_to_img_space_gt(test_gt_raw, all_starts=test_all_starts, all_ends=test_all_ends, window_size = 16, original_size=(224,224))
        test_preds  = yolo_to_img_space(test_preds_raw, all_starts=test_all_starts, all_ends=test_all_ends, 
                                        confidence_threshold=config['eval']['confidence_threshold'], 
                                        window_size = 16, original_size=(224,224))

        # Draw gt and predictions onto frames and saves as video
        # Note: This shows each 16-frame window independently, so frames repeat at window intersections
        draw_waggle_batch(
            all_frames=test_frames,
            all_detections=test_preds, 
            all_ground_truths=test_gts,
            all_start_frame_idxs=test_all_starts,
            output_dir='./outputs/vids'
        )

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
                                                        mode=config['post_process']['mode'])

        # Can postprocess entire predictions no need for limit to 16 sequences, its only needed when we visualise
        save_preds_to_csv(test_preds, f'raw_predictions_epoch_{epoch}.csv', 'raw', './outputs/preds_csv')
        save_preds_to_csv(post_test_preds, f'postprocessed_predictions_epoch_{epoch}.csv', 'postprocessed', './outputs/preds_csv')

        # visualise and store postprocessed results
        draw_waggle_batch(
            all_frames=test_frames,
            all_detections=post_test_preds, 
            all_ground_truths=test_gts,
            all_start_frame_idxs=test_all_starts,
            output_dir='./outputs/vids'
        )

        # Print unique clusters identifies
        unique_clusters_all = len({det['cluster_id'] for seq in post_test_preds for det in seq})
        unique_clusters_frames = len({det['cluster_id'] for seq in post_test_preds[:16] for det in seq})

        print('Number of clusterns - Entire data loader:', unique_clusters_all)
        print('Number of clusterns - Frames used for visualisation:', unique_clusters_frames)

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
                                        angular_thresholds=config['eval']['angular_thresholds'])
        post_test_metrics = get_eval_metrics(post_test_preds, test_gts, 
                                        pos_thresholds=config['eval']['pos_thresholds'],
                                        iou_threshold_range=config['eval']['iou_thresholds'],
                                        angular_thresholds=config['eval']['angular_thresholds'])
        
        print_evaluation_results(test_metrics, post_test_metrics)

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