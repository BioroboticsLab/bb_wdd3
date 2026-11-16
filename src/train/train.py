from tqdm import tqdm
import torch
import datetime
import os
import wandb
import torch.nn as nn

TQDM_BAR_FORMAT = '{l_bar}{bar:10}| {n_fmt}/{total_fmt} {elapsed}'  # fixed-width bar, no rate/ETA noise
TRAIN_COLS = ('Epoch', 'GPU_mem', 'Loss', 'Obj', 'NoObj', 'Pos', 'Dir', 'Temp', 'Instances', 'Size')


class GradAccumState:
    """Remembers the last real optimizer step across epochs, for gradient
    accumulation. Make one before the epoch loop, reuse it every epoch."""
    def __init__(self):
        self.last_opt_step = -1


def train(model, device, optimizer, yolocriterion, scheduler, train_loader, epoch, scaler, ema,
          grad_clip_norm=10.0, accumulate=1, accum_state=None):
    model.train()

    if accum_state is None:
        accum_state = GradAccumState()
    nb = len(train_loader)  # batches per epoch

    running_loss = torch.zeros(6, device=device)  # running MEAN, not raw per-batch loss -- less jumpy

    print(('\n' + '%11s' * len(TRAIN_COLS)) % TRAIN_COLS)  # header row, printed once per epoch

    pbar = enumerate(train_loader)
    pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # new bar each epoch -- lets the
                                                              # previous epoch's bar stay put
                                                              # and a new one start below it

    for batch_idx, batch in pbar:
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        inputs = batch["video"].to(device)
        targets = batch['targets'].to(device)

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs)
            total_loss_batch, obj_loss, no_obj_loss, position_loss, direction_loss, temporal_loss = yolocriterion(outputs, targets)

        # backward every batch, gradients just pile up in .grad since we
        # don't zero them yet. Only step the optimizer every `accumulate` batches.
        scaler.scale(total_loss_batch).backward()

        ni = epoch * nb + batch_idx  # step count since training started
        if ni - accum_state.last_opt_step >= accumulate:
            scaler.unscale_(optimizer)  # undo fp16 scaling before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            scaler.step(optimizer)
            ema.update(model)

            scaler.update()
            optimizer.zero_grad()
            accum_state.last_opt_step = ni

        # scheduler steps every batch no matter what, so the LR curve
        # still matches real epochs/time
        scheduler.step()

        loss_items = torch.stack([
            total_loss_batch, obj_loss, no_obj_loss,
            position_loss, direction_loss, temporal_loss,
        ]).detach()
        running_loss = (running_loss * batch_idx + loss_items) / (batch_idx + 1)

        mem = f'{torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0:.3g}G'
        n_instances = (targets[..., 0] > 0).sum().item()
        pbar.set_description(
            ('%11s' * 2 + '%11.4g' * 8) %
            (f'{epoch + 1}', mem, *running_loss.tolist(), n_instances, inputs.shape[-1])
        )

        # wandb gets the same running-mean numbers, logged every 100 batches
        if (batch_idx + 1) % 100 == 0:
            wandb.log({
                'batch/total_loss':     running_loss[0].item(),
                'batch/object_loss':    running_loss[1].item(),
                'batch/no_object_loss': running_loss[2].item(),
                'batch/position_loss':  running_loss[3].item(),
                'batch/direction_loss': running_loss[4].item(),
                'batch/temporal_loss':  running_loss[5].item(),
                'batch/lr':            optimizer.param_groups[0]['lr'],
                'batch/step':          ni,
            })

    return running_loss[0].item(), {
        'train/total_loss':     running_loss[0].item(),
        'train/object_loss':    running_loss[1].item(),
        'train/no_object_loss': running_loss[2].item(),
        'train/position_loss':  running_loss[3].item(),
        'train/direction_loss': running_loss[4].item(),
        'train/temporal_loss':  running_loss[5].item(),
        'learning_rate':        optimizer.param_groups[0]['lr']
    }
