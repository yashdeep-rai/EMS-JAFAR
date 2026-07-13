import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearUpsampler(nn.Module):
    def __init__(self, feature_dim=384, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.name = "bilinear"

    def forward(self, image, lr_feats, output_size):
        # Bilinearly interpolate the low-resolution features to the target output size
        return F.interpolate(lr_feats, size=output_size, mode="bilinear", align_corners=False)
