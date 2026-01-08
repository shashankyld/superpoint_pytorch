import torch
import torch.nn as nn
import torch.nn.functional as F
# Import your existing classes here
from .magicpoint import VGGEncoder, DetectorHead 

class SuperPoint(nn.Module):
    def __init__(self):
        super().__init__()
        # EXACT SAME structures as MagicPoint for seamless loading
        self.encoder = VGGEncoder(using_bn=True)
        self.detector = DetectorHead(in_ch=128)
        
        # NEW: Descriptor Head
        self.descriptor = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1)
        )

    def forward(self, x):
        # Support both raw tensor and dict inputs
        img = x['img'] if isinstance(x, dict) else x
        
        # Shared Encoding
        feat_map = self.encoder(img)
        
        # Head 1: Detection
        logits = self.detector(feat_map)
        
        # Head 2: Description
        desc = self.descriptor(feat_map)
        desc = F.normalize(desc, p=2, dim=1) # L2 Normalization is mandatory
        
        return {'logits': logits, 'desc': desc}