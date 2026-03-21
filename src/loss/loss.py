import torch
import torch.nn as nn
import torch.nn.functional as F

class WaggleDetectionLoss(nn.Module):
    def __init__(self, lambda_obj=2.0, lambda_coord=7.5, lambda_noobj=2.0, 
                 lambda_direction=7.0, lambda_temporal=7.0, use_varifocal=False,
                 gamma=2.0, grid_size=28, input_size=224, quality_decay='exponential'):
        super().__init__()
        self.lambda_obj = lambda_obj
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_direction = lambda_direction
        self.lambda_temporal = lambda_temporal
        self.use_varifocal = use_varifocal
        self.gamma = gamma
        self.cell_size = input_size / grid_size  # 8px for 224/28
        self.quality_decay = quality_decay

        if quality_decay not in ('exponential', 'gaussian', 'linear'):
            raise ValueError(f"Unknown quality_decay: '{quality_decay}'. Use 'exponential', 'gaussian' or 'linear'.")
        
    def compute_localization_quality(self, pred_pos, target_pos, obj_mask):
        """
        Compute soft quality score based on spatial distance between
        predicted and target position. Returns 1.0 for perfect prediction,
        decays with distance according to quality_decay mode:
          - exponential : exp(-dist / cell_size)         — immediate decay, never 0
          - gaussian    : exp(-dist^2 / 2*cell_size^2)     — flat near GT, sharp falloff beyond
          - linear      : clamp(1 - dist / 2*cell_size)  — constant decay, hard cutoff at 2 cells
        """
        quality = torch.zeros(obj_mask.shape, dtype=torch.float32, device=obj_mask.device)
        if obj_mask.sum() == 0:
            return quality

        dist = torch.norm(pred_pos[obj_mask] - target_pos[obj_mask], dim=-1)

        if self.quality_decay == 'exponential':
            quality[obj_mask] = torch.exp(-dist / self.cell_size)
        elif self.quality_decay == 'gaussian':
            quality[obj_mask] = torch.exp(-dist**2 / (2 * self.cell_size**2))
        elif self.quality_decay == 'linear':
            quality[obj_mask] = torch.clamp(1.0 - dist / (2 * self.cell_size), min=0.0)

        return quality
    
    def varifocal_loss(self, pred_conf, target_conf, target_quality):
        """
        Varifocal-style loss with soft targets.
        Quality is used as the target value for positives not as a loss weight,
        so gradients are never suppressed early in training. This ensures in 
        theory that the model always keeps learning.
        Focal weighting is only applied to negatives to handle class imbalance.
        """
        pred_prob = torch.sigmoid(pred_conf)
        pos_mask = target_conf > 0
        neg_mask = target_conf == 0

        # soft targets: quality score for positives, 0 for negatives
        soft_targets = torch.zeros_like(target_conf)
        soft_targets[pos_mask] = target_quality[pos_mask]

        # focal weighting only on negatives to suppress easy background
        focal_weight = torch.ones_like(target_conf)
        focal_weight[neg_mask] = (1 - pred_prob[neg_mask]).pow(self.gamma)

        bce = F.binary_cross_entropy_with_logits(pred_conf, soft_targets, reduction='none')
        weighted_loss = focal_weight * bce

        obj_loss   = weighted_loss[pos_mask].mean() if pos_mask.sum() > 0 else torch.tensor(0.0, device=pred_conf.device)
        noobj_loss = weighted_loss[neg_mask].mean() if neg_mask.sum() > 0 else torch.tensor(0.0, device=pred_conf.device)
        total_loss = weighted_loss.mean()

        return total_loss, obj_loss, noobj_loss
        
    def forward(self, predictions, targets):
        pred_conf     = predictions[..., 0]    # confidence
        pred_pos      = predictions[..., 1:3]  # x, y
        pred_dir      = predictions[..., 3:5]  # direction x, y
        pred_temporal = predictions[..., 5:7]  # temporal offsets
        
        target_conf     = targets[..., 0]
        target_pos      = targets[..., 1:3]
        target_dir      = targets[..., 3:5]
        target_temporal = targets[..., 5:7]
        
        obj_mask   = target_conf > 0
        noobj_mask = target_conf == 0
        
        # Position loss
        pos_loss = F.mse_loss(pred_pos[obj_mask], target_pos[obj_mask], reduction='sum')
        pos_loss = pos_loss / obj_mask.sum() if obj_mask.sum() > 0 else torch.tensor(0.0, device=predictions.device)
        
        # Direction and temporal loss
        if obj_mask.sum() > 0:
            pred_dir_obj   = pred_dir[obj_mask]
            target_dir_obj = target_dir[obj_mask]
            
            with torch.no_grad():
                pred_dir_norms    = torch.norm(pred_dir_obj, dim=-1)
                target_dir_norms  = torch.norm(target_dir_obj, dim=-1)
                pred_not_normed   = torch.abs(pred_dir_norms - 1.0) > 0.1
                target_not_normed = torch.abs(target_dir_norms - 1.0) > 0.1
                if pred_not_normed.any():
                    print(f"WARNING: {pred_not_normed.sum().item()} predicted direction vectors are not normalized!")
                    print(f"  Min norm: {pred_dir_norms.min().item():.4f}, Max norm: {pred_dir_norms.max().item():.4f}, Mean: {pred_dir_norms.mean().item():.4f}")
                if target_not_normed.any():
                    print(f"WARNING: {target_not_normed.sum().item()} target direction vectors are not normalized!")
                    print(f"  Min norm: {target_dir_norms.min().item():.4f}, Max norm: {target_dir_norms.max().item():.4f}, Mean: {target_dir_norms.mean().item():.4f}")
            
            cosine_sim = F.cosine_similarity(pred_dir_obj, target_dir_obj, dim=-1)
            dir_loss = (1 - cosine_sim).mean()
            
            temporal_loss = F.mse_loss(pred_temporal[obj_mask], target_temporal[obj_mask], reduction='sum')
            temporal_loss = temporal_loss / obj_mask.sum()
        else:
            dir_loss      = torch.tensor(0.0, device=predictions.device)
            temporal_loss = torch.tensor(0.0, device=predictions.device)
        
        # Confidence loss
        if self.use_varifocal:
            with torch.no_grad():
                loc_quality = self.compute_localization_quality(pred_pos, target_pos, obj_mask)
            conf_loss, obj_conf_loss, noobj_conf_loss = self.varifocal_loss(
                pred_conf, target_conf, loc_quality
            )
        else:
            obj_conf_loss = F.binary_cross_entropy_with_logits(
                pred_conf[obj_mask], target_conf[obj_mask]
            ) if obj_mask.sum() > 0 else torch.tensor(0.0, device=predictions.device)
            
            noobj_conf_loss = F.binary_cross_entropy_with_logits(
                pred_conf[noobj_mask], target_conf[noobj_mask]
            ) if noobj_mask.sum() > 0 else torch.tensor(0.0, device=predictions.device)
            
            conf_loss = obj_conf_loss + noobj_conf_loss

        # Total loss
        if self.use_varifocal:
            total_loss = (
                self.lambda_obj       * conf_loss +
                self.lambda_coord     * pos_loss +
                self.lambda_direction * dir_loss +
                self.lambda_temporal  * temporal_loss
            )
        else:
            total_loss = (
                self.lambda_obj       * obj_conf_loss +
                self.lambda_noobj     * noobj_conf_loss +
                self.lambda_coord     * pos_loss +
                self.lambda_direction * dir_loss +
                self.lambda_temporal  * temporal_loss
            )
        
        return total_loss, obj_conf_loss, noobj_conf_loss, pos_loss, dir_loss, temporal_loss