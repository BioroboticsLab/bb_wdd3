import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18
from typing import Dict, List, Tuple

class R2Plus1D_YOLO(nn.Module):
    def __init__(self, num_classes=1, grid_size=25, pretrained=True, 
                 max_detections_per_cell=1, dropout_rate=0.3):
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.max_detections_per_cell = max_detections_per_cell
        
        self.detection_params = 7
        self.total_outputs_per_cell = self.max_detections_per_cell * self.detection_params
        
        base_r2p1d = r2plus1d_18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(base_r2p1d.children())[:-2])
        
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        
        self.feature_projection = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate * 0.5)
        )
        
        self.detection_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.total_outputs_per_cell, kernel_size=1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        final_conv = self.detection_head[-1]
        nn.init.normal_(final_conv.weight, 0, 0.01)
        if final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, 0)
    
    def forward(self, x, return_activations=False):
        backbone_feat = self.backbone(x)  
        
        pooled_feat = self.temporal_pool(backbone_feat).squeeze(2) 
        
        projected_feat = self.feature_projection(pooled_feat)  
        
        detection_output = self.detection_head(projected_feat)  
        
        if detection_output.shape[-1] != self.grid_size or detection_output.shape[-2] != self.grid_size:
            detection_output = F.adaptive_avg_pool2d(detection_output, (self.grid_size, self.grid_size))
        
        batch_size = detection_output.shape[0]
        detections = detection_output.view(
            batch_size, 
            self.total_outputs_per_cell, 
            self.grid_size, 
            self.grid_size
        ).permute(0, 2, 3, 1)  # (B, grid, grid, total_outputs)
        
        detections = detections.view(
            batch_size,
            self.grid_size,
            self.grid_size,
            self.max_detections_per_cell,
            self.detection_params
        )
        
        detections = self.apply_output_activations(detections)
        
        return detections
    
    def apply_output_activations(self, detections):
        confidence = detections[..., 0:1] 
        position = detections[..., 1:3]    
        direction = detections[..., 3:5]   
        offsets = detections[..., 5:7]
        
        position = torch.sigmoid(position)   
        direction = F.normalize(direction, p=2, dim=-1)
        offsets = torch.sigmoid(offsets)
        
        activated_detections = torch.cat([confidence, position, direction, offsets], dim=-1)
        
        return activated_detections
