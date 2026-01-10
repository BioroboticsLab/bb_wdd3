import os 
#from prepare_data import create_video_frames_df
from utils.data_utils import create_video_frames_df
import random
import numpy as np
import torch
import torchvision.transforms as T
import pandas as pd
from src.train.train import train, train_v2
from src.eval.eval import eval
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import VideoYoloDataset, TemporalWaggleCollator
from src.models.model import R2Plus1D_YOLO
from src.loss.loss import WaggleDetectionLoss
from src.data.augmentation import WaggleAugmentations
from src.tests.aug_vis import demo_visualization
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn  as nn
from utils.data_utils import fix_dataframe_with_video_lengths, load_config
import datetime
from torch.utils.tensorboard import SummaryWriter
from src.utils.eval_utils import get_preds_gt, yolo_to_img_space, yolo_to_img_space_gt, get_eval_metrics
from src.utils.nms import batch_postprocess_predictions
from src.utils.vis_utils import reverse_transform, save_frames
import argparse


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main(args):
    config = load_config(args.config_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        use_multi_gpu = True
    else:
        print("Using single GPU")
        use_multi_gpu = False

    data = pd.read_csv(config['data']['annotations'])
    print(f"Original dataset length: {len(data)}")
    # 1/8 of original data for fine-tuning
    data = data.iloc[:len(data)//16].reset_index(drop=True)
    #data = data.iloc[:100].reset_index(drop=True)
    print(f"After subsetting dataset: {len(data)} samples")
    
    video_frames_dict = create_video_frames_df(data["video_name"].unique())

    data = fix_dataframe_with_video_lengths(data, video_frames_dict)

    data = data.sample(frac=1).reset_index(drop=True)

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
    
    # Create augmentation
    train_augmentation = WaggleAugmentations(
        width=224, height=224, 
        prob_flip_h=0.5, prob_flip_v=0.0, 
        prob_rotate=0.3, rotate_range=(-45, 45), 
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
    test_len = total_len - train_len

    train_indices = list(range(train_len))
    test_indices = list(range(train_len, total_len))

    train_df = data.iloc[train_indices].reset_index(drop=True)
    test_df = data.iloc[test_indices].reset_index(drop=True)

    # Create datasets
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
        augment=None,
        is_training=False 
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
        batch_size=config['train']['batch_size'],
        collate_fn=collator, 
        shuffle=False, 
        num_workers=config['train']['num_workers'],
        persistent_workers=(config['train']['num_workers'] > 0),
        pin_memory=True
    )

    print('Len train loader:', len(train_loader))

    model = R2Plus1D_YOLO(max_detections_per_cell=1, grid_size=28)

    os.makedirs('./ckpt', exist_ok=True)

    model = model.to(device)

    if use_multi_gpu:
        model = nn.DataParallel(model)
        print("Using Distributed Data Parallel (DDP).")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0005, betas=(0.937, 0.999))
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['train']['lr'],
        steps_per_epoch=len(train_loader),
        epochs=config['train']['epochs'],
        anneal_strategy='cos'
    )
    yolocriteria = WaggleDetectionLoss(lambda_obj=config["loss"]["lambda_obj"], 
                                       lambda_coord=config["loss"]["lambda_coord"], 
                                       lambda_noobj=config["loss"]["lambda_noobj"], 
                                       lambda_direction=config["loss"]["lambda_direction"], 
                                       lambda_temporal=config["loss"]["lambda_temporal"],
                                       use_varifocal=config["loss"]["use_varifocal"],
                                       gamma=config["loss"]["varifocal_gamma"],
                                       quality_scale=config["loss"]["varifocal_quality_scale"])
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", current_time)
    writer = SummaryWriter(log_dir=log_dir)
    scaler = torch.amp.GradScaler()
    
    best_val_loss = float('inf')
    
    for epoch in range(config['train']['epochs']):
        print(f'\nEpoch {epoch+1}/{config["train"]["epochs"]}')
        # Train one epoch
        train_loss = train_v2(
            model, device, optimizer, yolocriteria, scheduler, train_loader, epoch, scaler, writer
        )
        
        # Validate
        val_loss = eval(model, device, yolocriteria, test_loader, epoch, writer)

        print(f'Train loss: {train_loss}')
        print(f'Test/Val loss: {val_loss}')

        # Save best model
        # periodicly checkpoint lastest and best model
        # perodicly compute eval metrics and postprocessing to save compute
        if epoch % config['train']['val_freq']  == 0 or epoch == config['train']['epochs'] - 1:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), f'./ckpt/best_model_{epoch}.pth')
                else:
                    torch.save(model.state_dict(), f'./ckpt/best_model_{epoch}.pth')
                print(f"New best model saved with val_loss: {val_loss:.4f}")
            
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), f'./ckpt/lastest_{epoch}.pth')
            else:
                torch.save(model.state_dict(), f'./ckpt/latest_{epoch}.pth')
            print(f'Saved and evaluated model at Epoch {epoch}/{config["train"]["epochs"]}')
            
            '''
            # fetch all raw logits
            train_preds_raw, train_gt_raw, train_all_starts, train_all_ends, _ = get_preds_gt(model, train_loader, device,) 
            test_preds_raw, test_gt_raw, test_all_starts, test_all_ends, _, test_frames = get_preds_gt(model, test_loader, device, return_frames=True)
            # transform yolo gt annotations to image domain
            train_gts = yolo_to_img_space_gt(train_gt_raw, all_starts=train_all_starts, all_ends=train_all_ends)
            test_gts = yolo_to_img_space_gt(test_gt_raw, all_starts=test_all_starts, all_ends=test_all_ends)
            # transform raw logits to img space and filter by confidence
            train_preds  = yolo_to_img_space(train_preds_raw, all_starts=train_all_starts, all_ends=train_all_ends, confidence_threshold=0.8) 
            test_preds = yolo_to_img_space(test_preds_raw, all_starts=test_all_starts, all_ends=test_all_ends, confidence_threshold=0.8)
            # get eval metrics
            train_metrics = get_eval_metrics(train_preds, train_gts)
            test_metrics = get_eval_metrics(test_preds, test_gts)
            print("Before Post-Processing - Train Metrics:", train_metrics)
            print("Before Post-Processing - Test Metrics:", test_metrics)
            # overwrite train and test preds with postprocessed ones
            train_preds = batch_postprocess_predictions(train_preds)
            test_preds = batch_postprocess_predictions(test_preds)
            # metrics after postprocessing
            train_metrics = get_eval_metrics(train_preds, train_gts)
            test_metrics = get_eval_metrics(test_preds, test_gts)
            print("After Post-Processing - Train Metrics:", train_metrics)
            print("After Post-Processing - Test Metrics:", test_metrics)
            '''
    print('Training complete.')
    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description="Waggle detection training")

    parser.add_argument("--config_path", type=str, default='./configs/config.yaml', help="Path to the config file.")

    #parser.add_argument("--data_dir", type=str, default='./data/videos/', help="Path to directory containing video files")
    #parser.add_argument("--csv_path", type=str, default="./data/annotations/fps_multires_full_data.csv", help="Path to annotations csv")

    #parser.add_argument("--batch_size", type=int, default=16)

    #parser.add_argument("--n_epochs", type=int, default=100)

    #parser.add_argument("--num_workers", type=int, default=8)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)