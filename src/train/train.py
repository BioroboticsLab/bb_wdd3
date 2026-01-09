from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter 
import datetime
import os
import torch.nn as nn
from eval.eval import evaluate


def train(model, device, optimizer, yolocriterion, scheduler, train_loader, num_epochs=10):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("logs", current_time)
    writer = SummaryWriter(log_dir=log_dir)

    scaler = torch.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        torch.cuda.empty_cache()
        running_loss = 0.0
        batch_count = 0
        pos_loss, ob_loss, nob_loss, temp_loss, d_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            inputs = batch["video"].to(device)
            targets = batch['targets'].to(device)
            
            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(inputs, True)
                total_loss, obj_loss, no_obj_loss, position_loss, direction_loss, temporal_loss = yolocriterion(outputs, targets)
            
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            running_loss += total_loss.item()
            pos_loss += position_loss
            d_loss += direction_loss
            ob_loss += obj_loss
            nob_loss += no_obj_loss
            temp_loss += temporal_loss
            
            batch_count += 1
            global_step = epoch * len(train_loader) + batch_idx

            # writer.add_scalar('Loss/Train/Total_Batch', total_loss.item(), global_step)

            progress_bar.set_postfix({'Loss:': running_loss / batch_count})

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/batch_count:.4f}\n')
        epoch_avg_loss = running_loss / batch_count
        avg_position_loss = pos_loss / batch_count
        avg_direction_loss = d_loss / batch_count
        avg_temporal_loss = temp_loss / batch_count
        avg_obj_loss = ob_loss / batch_count
        avg_no_obj_loss = nob_loss / batch_count
        writer.add_scalar('Loss/Train/Total_Epoch_Avg', epoch_avg_loss, epoch)
        writer.add_scalar('Loss/Train/Position', avg_position_loss, global_step)
        writer.add_scalar('Loss/Train/Direction', avg_direction_loss, global_step)
        writer.add_scalar('Loss/Train/Temporal', avg_temporal_loss, global_step)
        writer.add_scalar('Loss/Train/Object', avg_obj_loss, global_step)
        writer.add_scalar('Loss/Train/No_obj_loss', avg_no_obj_loss, global_step)

    print('Finished Training')
    writer.close() 
    return model

def train_v2(model, device, optimizer, yolocriterion, scheduler, train_loader, epoch, scaler, writer):
    model.train()
    #torch.cuda.empty_cache()
    total_loss = 0.0
    total_obj_loss = 0.0
    total_no_obj_loss = 0.0
    total_position_loss = 0.0
    total_direction_loss = 0.0
    total_temporal_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch+1}', leave=False)
    for batch_idx, batch in enumerate(progress_bar):
        #print(f'Iteration: {batch_idx} / {len(train_loader)}')
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        inputs = batch["video"].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs)
            #print(f'Train Model out shape: {outputs.shape}')
            total_loss_batch, obj_loss, no_obj_loss, position_loss, direction_loss, temporal_loss = yolocriterion(outputs, targets)
        
        scaler.scale(total_loss_batch).backward()
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
            'Train Loss': f'{total_loss / num_batches:.4f}',
            'Obj': f'{total_obj_loss / num_batches:.4f}',
            'NoObj': f'{total_no_obj_loss / num_batches:.4f}'
        })
    
    # Calculate averages
    avg_val_loss = total_loss / num_batches
    avg_obj_loss = total_obj_loss / num_batches
    avg_no_obj_loss = total_no_obj_loss / num_batches
    avg_position_loss = total_position_loss / num_batches
    avg_direction_loss = total_direction_loss / num_batches
    avg_temporal_loss = total_temporal_loss / num_batches
    
    # Log to tensorboard
    writer.add_scalar('Loss/Training/Total', avg_val_loss, epoch)
    writer.add_scalar('Loss/Training/Object', avg_obj_loss, epoch)
    writer.add_scalar('Loss/Training/No_Object', avg_no_obj_loss, epoch)
    writer.add_scalar('Loss/Training/Position', avg_position_loss, epoch)
    writer.add_scalar('Loss/Training/Direction', avg_direction_loss, epoch)
    writer.add_scalar('Loss/Training/Temporal', avg_temporal_loss, epoch)
    
    print(f"\nTraining Results - Epoch {epoch+1}:")
    print(f"Total Loss: {avg_val_loss:.4f}")
    print(f"Object Loss: {avg_obj_loss:.4f}")
    print(f"No Object Loss: {avg_no_obj_loss:.4f}")
    print(f"Position Loss: {avg_position_loss:.4f}")
    print(f"Direction Loss: {avg_direction_loss:.4f}")
    print(f"Temporal Loss: {avg_temporal_loss:.4f}")
    
    return avg_val_loss

def validate_model(model, val_loader, yolocriterion, device, epoch, writer):
    model.eval()
    total_loss = total_obj_loss = total_no_obj_loss = total_temporal_loss= 0
    total_position_loss = total_direction_loss = 0
    num_batches = 0
    val_loss = 0
    
    with torch.no_grad():
        loop = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Validation Epoch {epoch}")
        for batch_idx, batch in loop:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            inputs = batch['video'].to(device)
            # inputs = {k: v.to(device) for k, v in batch["video"].items()}
            targets = batch['targets'].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs= model(inputs, True)
                total_loss, obj_loss, no_obj_loss, position_loss, direction_loss, temporal_loss = yolocriterion(outputs, targets)

            val_loss += total_loss.item()
            total_obj_loss += obj_loss.item()
            total_no_obj_loss += no_obj_loss.item()
            total_position_loss += position_loss.item()
            total_direction_loss += direction_loss.item()
            total_temporal_loss += temporal_loss.item()
            num_batches += 1
            
            loop.set_postfix({
                'Loss': f'{val_loss / num_batches:.4f}',
                'Obj': f'{total_obj_loss / num_batches:.4f}',
                'NoObj': f'{total_no_obj_loss / num_batches:.4f}'
            })


    avg_val_loss = val_loss / num_batches
    avg_obj_loss = total_obj_loss / num_batches
    avg_no_obj_loss = total_no_obj_loss / num_batches
    avg_position_loss = total_position_loss / num_batches
    avg_direction_loss = total_direction_loss / num_batches
    avg_temporal_loss = total_temporal_loss / num_batches

    writer.add_scalar('Loss/Validation/Total_Loss', avg_val_loss, epoch)
    writer.add_scalar('Loss/Validation/Object', avg_obj_loss, epoch)
    writer.add_scalar('Loss/Validation/No_obj_loss', avg_no_obj_loss, epoch)
    writer.add_scalar('Loss/Validation/Position', avg_position_loss, epoch)
    writer.add_scalar('Loss/Validation/Direction', avg_direction_loss, epoch)
    writer.add_scalar('Loss/Validation/Temporal', avg_temporal_loss, epoch)

    print(f"Validation Results:")
    print(f"Loss: {avg_val_loss:.4f}")
    print(f"Object Loss: {avg_obj_loss:.4f}")
    print(f"No Object Loss: {avg_no_obj_loss:.4f}")
    print(f"Position Loss: {avg_position_loss:.4f}")
    print(f"Direction Loss: {avg_direction_loss:.4f}")
    print(f"Temporal Loss: {avg_temporal_loss:.4f}")

    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), f'./models/model_{epoch}.pth')
    else:
        torch.save(model.state_dict(), f'./models/model_{epoch}.pth')

    if epoch == 0 or epoch == 1 or epoch % 10 == 0:
        print('Evaluating.')
        evaluate(f'./models/model_{epoch}.pth', epoch, confidence=0.5, writer=writer)
    
    return avg_val_loss
