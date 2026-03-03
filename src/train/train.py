from tqdm import tqdm
import torch
import wandb
import datetime
import os
import torch.nn as nn

def train(model, device, optimizer, yolocriterion, scheduler, train_loader, epoch, scaler, ema):
    model.train()
    #torch.cuda.empty_cache()
    total_loss = 0.0
    total_obj_loss = 0.0
    total_no_obj_loss = 0.0
    total_position_loss = 0.0
    total_direction_loss = 0.0
    total_temporal_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc=f'Train Epoch {epoch+1}', leave=True, position=int(epoch))
    
    for batch_idx, batch in enumerate(progress_bar):
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        inputs = batch["video"].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs)
            total_loss_batch, obj_loss, no_obj_loss, position_loss, direction_loss, temporal_loss = yolocriterion(outputs, targets)
        
        scaler.scale(total_loss_batch).backward()
        scaler.step(optimizer)
        ema.update(model)

        scaler.update()
        
        # Scheduler step per batch for OneCycleLR
        scheduler.step()
        
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
        'train/total_loss': avg_val_loss,
        'train/object_loss': avg_obj_loss,
        'train/no_object_loss': avg_no_obj_loss,
        'train/position_loss': avg_position_loss,
        'train/direction_loss': avg_direction_loss,
        'train/temporal_loss': avg_temporal_loss,
        'learning_rate': optimizer.param_groups[0]['lr']
    })
    
    return avg_val_loss