import torch
import torch.nn as nn
import kornia.geometry.transform as K_geo
import kornia.augmentation as K
import torch.nn.functional as F
import torchvision.utils as vutils

class HomographicAdaptation(nn.Module):
    def __init__(self, model, device, num_iter=100, sub_batch_size=10):
        super().__init__()
        self.model = model
        self.device = device
        self.num_iter = num_iter
        self.sub_batch_size = sub_batch_size
        
        self.sampler = K.AugmentationSequential(
            K.RandomPerspective(distortion_scale=0.2, p=1.0),
            K.RandomRotation(degrees=45.0, p=1.0),
            K.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), p=1.0),
            data_keys=["input"],
            keepdim=True,
            transformation_matrix_mode="silent"
        )

    def forward(self, x, return_diversity=False):
        self.model.eval()
        B, C, H, W = x.shape
        aggregated_heatmap = torch.zeros((B, 1, H, W), device=self.device)
        diversity_grid = None
        
        num_chunks = self.num_iter // self.sub_batch_size
        
        with torch.no_grad():
            for i in range(num_chunks):
                x_sub = x.repeat(self.sub_batch_size, 1, 1, 1)
                x_warped = self.sampler(x_sub)
                homo = self.sampler.transform_matrix
                
                # Logic to capture the grid for visualization
                if return_diversity and i == 0:
                    # Capture 16 samples for the montage
                    grid = vutils.make_grid(x_warped[:16], nrow=4, normalize=True)
                    diversity_grid = grid.permute(1, 2, 0).cpu().numpy()

                logits = self.model(x_warped)
                prob = F.softmax(logits, dim=1)
                
                # Reshape 65-channel logits back to HxW heatmap
                h_small, w_small = H // 8, W // 8
                heatmaps = prob[:, :-1, :, :].view(-1, 8, 8, h_small, w_small)
                heatmaps = heatmaps.permute(0, 3, 1, 4, 2).reshape(-1, 1, H, W)
                
                # Un-warp to original frame
                inv_homo = torch.inverse(homo)
                unwarped = K_geo.warp_perspective(heatmaps, inv_homo, dsize=(H, W), align_corners=True)
                
                # Sum the whole chunk into the aggregator
                aggregated_heatmap += unwarped.sum(dim=0, keepdim=True)

        return aggregated_heatmap / self.num_iter, diversity_grid