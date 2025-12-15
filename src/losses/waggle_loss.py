import torch
import torch.nn as nn
import torch.nn.functional as F

class WaggleDetectionLoss(nn.Module):
    """
        Loss function for waggle dance detection using a YOLO-style prediction head.

        Combines multiple components to supervise:
            • Objectness confidence
            • Bee position within grid cell (x, y)
            • Direction vector (normalized orientation)
            • Temporal embedding (start/end of waggle)

        Each component is weighted by a configurable lambda.

        Parameters
        ----------
        lambda_obj : float
            Weight for the objectness loss on positive cells.
        lambda_coord : float
            Weight for the coordinate (x, y) regression loss.
        lambda_noobj : float
            Weight for the objectness loss on cells without objects.
        lambda_direction : float
            Weight for direction-vector supervision using cosine similarity.
        lambda_temporal : float
            Weight for the temporal start/end embedding regression.
        
        Returns
        -------
        tuple
            A tuple containing:
            (total_loss, obj_conf_loss, noobj_conf_loss, pos_loss, dir_loss, temporal_loss)
    """
    def __init__(self, lambda_obj=3.0, lambda_coord=7.5, lambda_noobj=5.0, lambda_direction=7.0, lambda_temporal=7.0):
        super().__init__()
        self.lambda_obj = lambda_obj
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_direction = lambda_direction
        self.lambda_temporal = lambda_temporal

    def forward(self, predictions, targets):

        pred_conf = predictions[..., 0]        # confidence
        pred_pos  = predictions[..., 1:3]     # x, y
        pred_dir  = predictions[..., 3:5]     # direction x,y
        pred_temporal = predictions[..., 5:7] # temporal embeddings

        target_conf = targets[..., 0]
        target_pos  = targets[..., 1:3]
        target_dir  = targets[..., 3:5]
        target_temporal = targets[..., 5:7]

        # Masks
        obj_mask = target_conf > 0
        noobj_mask = target_conf == 0

        obj_conf_loss = F.binary_cross_entropy_with_logits(
            pred_conf[obj_mask], target_conf[obj_mask]
        ) if obj_mask.sum() > 0 else torch.tensor(0.0, device=predictions.device)

        noobj_conf_loss = F.binary_cross_entropy_with_logits(
            pred_conf[noobj_mask], target_conf[noobj_mask]
        ) if noobj_mask.sum() > 0 else torch.tensor(0.0, device=predictions.device)

        pos_loss = F.mse_loss(pred_pos[obj_mask], target_pos[obj_mask], reduction='sum')
        pos_loss = pos_loss / obj_mask.sum() if obj_mask.sum() > 0 else torch.tensor(0.0, device=predictions.device)

        if obj_mask.sum() > 0:
            pred_dir_obj = pred_dir[obj_mask]
            target_dir_obj = target_dir[obj_mask]
            cosine_sim = F.cosine_similarity(pred_dir_obj, target_dir_obj, dim=-1)
            dir_loss = (1 - cosine_sim).mean()

            temporal_loss = F.mse_loss(pred_temporal[obj_mask], target_temporal[obj_mask], reduction='sum')
            temporal_loss = temporal_loss / obj_mask.sum()
        else:
            dir_loss = torch.tensor(0.0, device=predictions.device)
            temporal_loss = torch.tensor(0.0, device=predictions.device)

        # Total loss with weights
        total_loss = (
            self.lambda_obj * obj_conf_loss +
            self.lambda_noobj * noobj_conf_loss +
            self.lambda_coord * pos_loss +
            self.lambda_direction * dir_loss +
            self.lambda_temporal * temporal_loss
        )

        return (
            total_loss,
            obj_conf_loss,
            noobj_conf_loss,
            pos_loss,
            dir_loss,
            temporal_loss
        )