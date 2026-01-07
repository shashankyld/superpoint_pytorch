# Random Noise Pattern
# Random image Crops/Distortions
# Random global color
# Random blobs - diffused circles 
# Random homographic transformations
# Random Salt and Pepper
# ...

import torch
import torch.nn as nn
import kornia.augmentation as K
import kornia.filters as F
import torch.nn.functional as F_func

class SyntheticAugmentor(nn.Module):
    def __init__(self, h=240, w=320):
        super().__init__()
        self.h, self.w = h, w
        
        # 1. GEOMETRIC PIPELINE: Synchronized Image + Keypoint movement
        self.geometric_pipeline = K.AugmentationSequential(
            K.RandomPerspective(distortion_scale=0.1, p=0.5),
            K.RandomRotation(degrees=45.0, p=0.3),
            K.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), p=0.3),
            # THE VETO: Removes boundary artifacts and updates keypoint coordinates
            K.RandomResizedCrop(size=(self.h, self.w), scale=(0.5, 0.5), p=0.5),
            K.RandomResizedCrop(size=(self.h, self.w), scale=(0.9, 0.9), p=0.5),
            # K.CenterCrop(size=(self.h, self.w), keepdim=True, p=1.0),
            data_keys=["input", "keypoints"],
            keepdim=True
        )

        # 2. PHOTOMETRIC PIPELINE: Pixel-only intensity changes
        self.photometric_pipeline = K.AugmentationSequential(
            K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),
            K.ColorJitter(brightness=0.3, contrast=0.3, p=0.7),
            # FIXED: Added required 'direction' argument
            K.RandomMotionBlur(kernel_size=3, angle=35.0, direction=0.5, p=0.3),
            # Salt Pepper Noise
            K.RandomSaltAndPepperNoise(amount=0.02, salt_vs_pepper=0.5, p=1),
            K.RandomAutoContrast(p=0.3),
            K.RandomBrightness(brightness=0.2, p=0.5),
            K.RandomInvert(p=1),
            
            data_keys=["input"],
            keepdim=True
        )



    def generate_blobs(self, x):
        """Generates organic diffused occlusions (light leaks/blobs)."""
        low_res = torch.randn(x.shape[0], 1, self.h // 30, self.w // 30, device=x.device)
        blobs = F_func.interpolate(low_res, size=(self.h, self.w), mode='bicubic', align_corners=False)
        blobs = F.gaussian_blur2d(blobs, (21, 21), (8.0, 8.0))
        return torch.sigmoid(blobs * 2.0)

    def forward(self, x, pts):
        """
        x: (B, 1, H, W)
        pts: (B, N, 2)
        """
        # STEP 1: Apply Geometry
        # Coordinates and pixels are transformed as a single mathematical unit.
        x_aug, pts_aug = self.geometric_pipeline(x, pts)

        # STEP 2: Apply Photometry (Pixels only)
        x_aug = self.photometric_pipeline(x_aug)
        
        # STEP 3: Apply Blobs (Occlusion)
        # These change pixel intensity but do not move the corners.
        # if torch.rand(1) > 0.5:
        if True:
            blob_map = self.generate_blobs(x_aug)
            if x_aug.shape != blob_map.shape:
             raise RuntimeError(f"Shape Mismatch! x_aug: {x_aug.shape}, blob_map: {blob_map.shape}. "
                                f"Init H/W: {self.h}/{self.w}")
            if torch.rand(1) > 0.5:
                x_aug = torch.clamp(x_aug * (1 - 0.4 * blob_map), 0, 1) # Darken
            else:
                x_aug = torch.clamp(x_aug + 0.3 * blob_map, 0, 1) # Lighten
            
        return x_aug, pts_aug