import torch
import torch.nn as nn
import kornia.geometry.transform as K_geo
import kornia.augmentation as K
import torch.nn.functional as F
import torchvision.utils as vutils

class HomographicAdaptation(nn.Module):
    def __init__(self, model, device, num_iter=100):
        super().__init__()
        self.model = model
        self.device = device
        self.num_iter = num_iter
        
        # Using a single geometric sampler that provides the 3x3 matrices
        self.sampler = K.AugmentationSequential(
            K.RandomPerspective(distortion_scale=0.5, p=1.0),
            K.RandomRotation(degrees=90.0, p=1.0),
            K.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), p=1.0),
            data_keys=["input"],
            keepdim=True,
            transformation_matrix_mode="silent"
        )

    def forward(self, x):
        """
        x: (B, 1, H, W) -> e.g., (1, 1, 240, 320)
        Returns: 
            aggregated_heatmap: (B, 1, H, W)
            diversity_grid: np.array (for visualization)
        """
        self.model.eval()
        B, C, H, W = x.shape
        
        # 1. Create the Super-Batch: [B, C, H, W] -> [B * N, C, H, W]
        # This allows the GPU to process all 100 warps at once
        x_repeated = x.repeat(self.num_iter, 1, 1, 1)
        
        with torch.no_grad():
            # 2. Warp the entire super-batch in one go
            x_warped = self.sampler(x_repeated)
            homo = self.sampler.transform_matrix # (B*N, 3, 3)

            # 3. Forward pass through MagicPoint
            logits = self.model(x_warped)
            
            # 4. Space-to-Depth Reversal
            prob = F.softmax(logits, dim=1) # Softmax across 65 channels
            # Shape: (B*N, 64, 30, 40) -> (B*N, 1, 240, 320)
            heatmaps = prob[:, :-1, :, :].view(B * self.num_iter, 8, 8, 30, 40)
            heatmaps = heatmaps.permute(0, 3, 1, 4, 2).reshape(B * self.num_iter, 1, H, W)

            # 5. Invert and un-warp back to the original frame
            inv_homo = torch.inverse(homo)
            unwarped_heatmaps = K_geo.warp_perspective(
                heatmaps, inv_homo, dsize=(H, W), align_corners=True
            )

            # 6. Combine: Split the Super-Batch back into individual images and average
            # Returns a tuple of N tensors, each of size (B, 1, H, W)
            split_heatmaps = torch.split(unwarped_heatmaps, B, dim=0)
            aggregated_heatmap = sum(split_heatmaps) / self.num_iter

        # 7. Diversity Visualization Logic
        # Grab first 16 warps of the first image in the batch
        diversity_sample = x_warped[:16]
        grid = vutils.make_grid(diversity_sample, nrow=4, normalize=True, pad_value=1)
        diversity_grid = grid.permute(1, 2, 0).cpu().numpy()

        return aggregated_heatmap, diversity_grid