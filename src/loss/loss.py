import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class WaggleDetectionLoss(nn.Module):
    def __init__(self, lambda_obj=2.0, lambda_coord=7.5, lambda_noobj=2.0, 
                 lambda_direction=7.0, lambda_temporal=7.0, use_varifocal=False,
                 gamma=2.0, quality_scale=0.1):
        super().__init__()
        self.lambda_obj = lambda_obj
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_direction = lambda_direction
        self.lambda_temporal = lambda_temporal
        self.use_varifocal = use_varifocal
        self.gamma = gamma
        self.quality_scale = quality_scale
    def compute_localization_quality(self, pred_pos, target_pos, obj_mask):
        """Compute quality score based on localization accuracy"""
        if obj_mask.sum() == 0:
            return torch.zeros(obj_mask.shape, dtype=torch.float32, device=obj_mask.device)
        
        # Create quality tensor with correct shape and device
        quality = torch.zeros(obj_mask.shape, dtype=torch.float32, device=obj_mask.device)
        
        # Compute distance for positive samples
        dist = torch.norm(pred_pos[obj_mask] - target_pos[obj_mask], dim=-1)
        
        # Convert to quality scores
        quality_scores = torch.exp(-dist / self.quality_scale)
        
        # Assign quality scores to positive positions
        quality[obj_mask] = quality_scores
        
        return quality
    
    def varifocal_loss(self, pred_conf, target_conf, target_quality):
        """Varifocal loss for confidence prediction"""
        pred_prob = torch.sigmoid(pred_conf)
        
        pos_mask = target_conf > 0
        neg_mask = target_conf == 0
        
        # Positive samples: weighted by quality
        # Negative samples: focal loss weighting
        focal_weight = torch.where(
            target_conf > 0,
            target_quality,
            (1 - pred_prob).pow(self.gamma)
        )
        
        bce = F.binary_cross_entropy_with_logits(
            pred_conf, target_conf, reduction='none'
        )
        
        weighted_loss = focal_weight * bce
        
        # Calculate separate losses for logging
        obj_loss = weighted_loss[pos_mask].mean() if pos_mask.sum() > 0 else torch.tensor(0.0, device=pred_conf.device)
        noobj_loss = weighted_loss[neg_mask].mean() if neg_mask.sum() > 0 else torch.tensor(0.0, device=pred_conf.device)
        total_loss = weighted_loss.mean()
    
        return total_loss, obj_loss, noobj_loss
        
    def forward(self, predictions, targets):
        pred_conf = predictions[..., 0]        # confidence
        pred_pos  = predictions[..., 1:3]      # x, y
        pred_dir  = predictions[..., 3:5]      # direction x,y
        pred_temporal = predictions[..., 5:7]  # temporal embeddings
        
        target_conf = targets[..., 0]
        target_pos  = targets[..., 1:3]
        target_dir  = targets[..., 3:5]
        target_temporal = targets[..., 5:7]
        
        # Masks
        obj_mask = target_conf > 0
        noobj_mask = target_conf == 0
        
        # Position loss
        pos_loss = F.mse_loss(pred_pos[obj_mask], target_pos[obj_mask], reduction='sum')
        pos_loss = pos_loss / obj_mask.sum() if obj_mask.sum() > 0 else torch.tensor(0.0, device=predictions.device)
        
        # Direction and temporal loss (MOVED UP)
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
        
        # Confidence loss
        if self.use_varifocal:
            # Compute localization quality for varifocal loss
            with torch.no_grad():
                loc_quality = self.compute_localization_quality(pred_pos, target_pos, obj_mask)
            
            # Combined varifocal loss (handles both obj and noobj)
            conf_loss, obj_conf_loss, noobj_conf_loss = self.varifocal_loss(pred_conf, target_conf, loc_quality)
        else:
            # Original BCE loss
            obj_conf_loss = F.binary_cross_entropy_with_logits(
                pred_conf[obj_mask], target_conf[obj_mask]
            ) if obj_mask.sum() > 0 else torch.tensor(0.0, device=predictions.device)
            
            noobj_conf_loss = F.binary_cross_entropy_with_logits(
                pred_conf[noobj_mask], target_conf[noobj_mask]
            ) if noobj_mask.sum() > 0 else torch.tensor(0.0, device=predictions.device)
            
            conf_loss = obj_conf_loss + noobj_conf_loss  # For consistency
        
        # Total loss with weights
        if self.use_varifocal:
            total_loss = (
                self.lambda_obj * conf_loss +
                self.lambda_coord * pos_loss +
                self.lambda_direction * dir_loss +
                self.lambda_temporal * temporal_loss
            )
        else:
            total_loss = (
                self.lambda_obj * obj_conf_loss +
                self.lambda_noobj * noobj_conf_loss +
                self.lambda_coord * pos_loss +
                self.lambda_direction * dir_loss +
                self.lambda_temporal * temporal_loss
            )
        
        return total_loss, obj_conf_loss, noobj_conf_loss, pos_loss, dir_loss, temporal_loss