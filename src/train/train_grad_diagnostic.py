from tqdm import tqdm
import torch
import wandb
import datetime
import os
import torch.nn as nn

def train(model, device, optimizer, yolocriterion, scheduler, train_loader, epoch, scaler):
    model.train()
    
    total_loss = 0.0
    total_obj_loss = 0.0
    total_no_obj_loss = 0.0
    total_position_loss = 0.0
    total_direction_loss = 0.0
    total_temporal_loss = 0.0
    num_batches = 0
    
    # Gradient diagnostics checks every nth epoch
    check_gradients_every = 10
    total_spatial_grad = 0.0
    total_direction_grad = 0.0
    total_temporal_grad = 0.0
    total_shared_grad = 0.0
    gradient_checks = 0
    
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
        
       # Gradient diagnostic 
        if batch_idx % check_gradients_every == 0:
            # Get the scale factor to unscale gradients for fair comparison
            scale = scaler.get_scale()
            
            spatial_grad = 0.0
            direction_grad = 0.0
            temporal_grad = 0.0
            shared_grad = 0.0
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Unscale gradient for true magnitude
                    true_grad_norm = param.grad.norm().item() / scale
                    
                    if 'spatial_head' in name:
                        spatial_grad += true_grad_norm
                    elif 'direction_head' in name:
                        direction_grad += true_grad_norm
                    elif 'temporal_head' in name:
                        temporal_grad += true_grad_norm
                    elif 'shared_features' in name or 'feature_projection' in name:
                        shared_grad += true_grad_norm
            
            total_spatial_grad += spatial_grad
            total_direction_grad += direction_grad
            total_temporal_grad += temporal_grad
            total_shared_grad += shared_grad
            gradient_checks += 1
            
            # Print every check_gradients_every batches
            print(f"\n[Batch {batch_idx}] Gradient Norms:")
            print(f"  Spatial: {spatial_grad:.4f}, Direction: {direction_grad:.4f}, "
                  f"Temporal: {temporal_grad:.4f}, Shared: {shared_grad:.4f}")
            print(f"  Ratio (Dir/Spatial): {direction_grad/spatial_grad if spatial_grad > 0 else 0:.3f}")
            print(f"  Weighted Loss Contributions - "
                  f"Pos: {yolocriterion.lambda_coord * position_loss.item():.4f}, "
                  f"Dir: {yolocriterion.lambda_direction * direction_loss.item():.4f}, "
                  f"Temp: {yolocriterion.lambda_temporal * temporal_loss.item():.4f}")
        
        scaler.step(optimizer)
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
    
    # Log gradient diagnostics to wandb (epoch averages)
    if gradient_checks > 0:
        avg_spatial_grad = total_spatial_grad / gradient_checks
        avg_direction_grad = total_direction_grad / gradient_checks
        avg_temporal_grad = total_temporal_grad / gradient_checks
        avg_shared_grad = total_shared_grad / gradient_checks
        
        wandb.log({
            'epoch': epoch,
            'gradients/spatial_head': avg_spatial_grad,
            'gradients/direction_head': avg_direction_grad,
            'gradients/temporal_head': avg_temporal_grad,
            'gradients/shared_features': avg_shared_grad,
            'gradients/direction_to_spatial_ratio': avg_direction_grad / avg_spatial_grad if avg_spatial_grad > 0 else 0,
        })
        
        print(f"\nEpoch {epoch+1} Gradient Summary")
        print(f"Avg Gradient Norms - Spatial: {avg_spatial_grad:.4f}, "
              f"Direction: {avg_direction_grad:.4f}, Temporal: {avg_temporal_grad:.4f}")
        print(f"Direction/Spatial Ratio: {avg_direction_grad/avg_spatial_grad if avg_spatial_grad > 0 else 0:.3f}")
    
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