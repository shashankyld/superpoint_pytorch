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
import numpy as np

class SyntheticAugmentor(nn.Module):
    def __init__(self):
        super().__init__()
        # We define the geometric part separately to handle coordinate updates
        self.geometric_pipeline = K.AugmentationSequential(
            K.RandomPerspective(distortion_scale=0.15, p=0.5),
            K.RandomRotation(degrees=15.0, p=0.3),
            K.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), p=0.3),
            data_keys=["input", "keypoints"],
            keepdim=True
        )

        # Photometric pipeline only affects the image pixels
        self.photometric_pipeline = K.AugmentationSequential(
            K.RandomGaussianNoise(mean=0.0, std=0.05, p=0.5),
            K.ColorJitter(brightness=0.3, contrast=0.3, p=0.7),
            K.RandomMotionBlur(kernel_size=3, angle=35.0, direction=0.5, p=0.2),
            data_keys=["input"],
            keepdim=True
        )

    def add_salt_and_pepper(self, x, amount=0.01):
        """Adds salt and pepper noise to the image tensor."""
        mask = torch.rand_like(x)
        x[mask < amount/2] = 0.0
        x[mask > 1 - amount/2] = 1.0
        return x

    def forward(self, x, pts):
        """
        Args:
            x: Image tensor of shape (B, 1, H, W).
            pts: Keypoint tensor of shape (B, N, 2).
        Returns:
            aug_x: Augmented image.
            aug_pts: Updated coordinates.
        """
        # 1. Apply Geometric Transforms (Image + Points move together)
        # Kornia re-calculates pts based on the homography of the warp.
        out = self.geometric_pipeline(x, pts)
        aug_x, aug_pts = out[0], out[1]

        # 2. Apply Photometric Transforms (Image pixels only)
        aug_x = self.photometric_pipeline(aug_x)
        
        # 3. Apply Manual Effects
        aug_x = self.add_salt_and_pepper(aug_x)
        
        # Random Blobs logic
        if torch.rand(1) > 0.5:
            # Diffused blobs simulate light leaks or occlusions.
            blobs = F.gaussian_blur2d(torch.randn_like(aug_x), (11, 11), (4.0, 4.0))
            aug_x = torch.clamp(aug_x + 0.15 * blobs, 0, 1)
            
        return aug_x, aug_pts