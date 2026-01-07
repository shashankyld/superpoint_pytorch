import torch
import torch.nn as nn

class VGGEncoder(nn.Module):
    """
    Standard VGG-style backbone. 
    Reduces spatial resolution by 8x (240x320 -> 30x40).
    """
    def __init__(self, using_bn=True):
        super().__init__()
        
        def conv_block(in_ch, out_ch):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ]
            if using_bn:
                layers.insert(1, nn.BatchNorm2d(out_ch))
            return nn.Sequential(*layers)

        # Stage 1: 1 -> 64 | Output: 120x160
        self.block1 = nn.Sequential(conv_block(1, 64), conv_block(64, 64), nn.MaxPool2d(2, 2))
        # Stage 2: 64 -> 64 | Output: 60x80
        self.block2 = nn.Sequential(conv_block(64, 64), conv_block(64, 64), nn.MaxPool2d(2, 2))
        # Stage 3: 64 -> 128 | Output: 30x40
        self.block3 = nn.Sequential(conv_block(64, 128), conv_block(128, 128), nn.MaxPool2d(2, 2))
        # Stage 4: 128 -> 128 | Output: 30x40 (No pooling here)
        self.block4 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

class DetectorHead(nn.Module):
    """Maps the 128 feature channels to 65 channels for the 8x8 grid."""
    def __init__(self, in_ch=128):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.out = nn.Conv2d(256, 65, kernel_size=1)

    def forward(self, x):
        x = self.relu(self.conv(x))
        return self.out(x)

class MagicPoint(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        # We hardcode BN=True for scratch training success
        self.encoder = VGGEncoder(using_bn=True)
        self.detector = DetectorHead(in_ch=128)
        
    def forward(self, x):
        # Support both raw tensor and dict inputs from DataLoader
        img = x['img'] if isinstance(x, dict) else x
        
        feat_map = self.encoder(img)
        logits = self.detector(feat_map)
        
        return logits # Returns (B, 65, H/8, W/8)