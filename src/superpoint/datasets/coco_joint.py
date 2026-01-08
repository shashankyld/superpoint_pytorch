import torch
import numpy as np
import cv2
import os
from torch.utils.data import Dataset
import kornia.geometry.transform as K_geo
import kornia.geometry.linalg as K_linalg
import kornia.augmentation as K

class COCOJointDataset(Dataset):
    def __init__(self, img_dir, label_dir):
        self.img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.label_dir = label_dir
        self.h, self.w = 240, 320

        # Pixel-only changes
        self.photo_aug = K.AugmentationSequential(
            K.RandomGaussianNoise(std=0.05, p=0.5),
            K.ColorJitter(brightness=0.3, contrast=0.3, p=0.5),
            data_keys=["input"], keepdim=True
        )

    def __len__(self):
        return len(self.img_paths)

    def _generate_homo(self):
        src = torch.tensor([[0,0], [self.w-1, 0], [self.w-1, self.h-1], [0, self.h-1]]).float()
        dst = src + torch.randn_like(src) * 20.0 
        return K_geo.get_perspective_transform(src.unsqueeze(0), dst.unsqueeze(0))

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_a = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_a = cv2.resize(img_a, (self.w, self.h))
        img_a_t = torch.from_numpy(img_a).float().unsqueeze(0).unsqueeze(0) / 255.0

        # Load labels: [N, 2] -> (y, x)
        label_name = os.path.basename(img_path).replace('.jpg', '.npy')
        pts_a_yx = np.load(os.path.join(self.label_dir, label_name))
        
        # Convert to (x, y) for Kornia: [1, N, 2]
        pts_a_xy = torch.from_numpy(pts_a_yx).float().flip(-1).unsqueeze(0) 

        # 1. Warp
        H = self._generate_homo()
        img_b_t = K_geo.warp_perspective(img_a_t, H, dsize=(self.h, self.w), align_corners=True)
        pts_b_xy = K_linalg.transform_points(H, pts_a_xy)
        
        # 2. Validity
        valid_mask = K_geo.warp_perspective(torch.ones_like(img_a_t), H, dsize=(self.h, self.w), align_corners=True)

        # 3. Photo Aug
        img_b_aug = self.photo_aug(img_b_t)

        # 4. Generate Heatmaps
        label_a = self._coords_to_heatmap(pts_a_xy.squeeze(0))
        label_b = self._coords_to_heatmap(pts_b_xy.squeeze(0), mask=valid_mask.squeeze())

        return {
            'image_a': img_a_t.squeeze(0), 'label_a': label_a.unsqueeze(0),
            'image_b': img_b_aug.squeeze(0), 'label_b': label_b.unsqueeze(0),
            'valid_mask': valid_mask.squeeze(0), 'homography': H.squeeze(0)
        }

    def _coords_to_heatmap(self, coords, mask=None):
        heatmap = torch.zeros((self.h, self.w))
        # Round and convert to long for indexing
        c = torch.round(coords).long()
        
        # Bound check
        valid = (c[:, 0] >= 0) & (c[:, 0] < self.w) & (c[:, 1] >= 0) & (c[:, 1] < self.h)
        c = c[valid]
        
        if mask is not None and c.shape[0] > 0:
            # Only keep points where valid_mask is white
            m_vals = mask[c[:, 1], c[:, 0]]
            c = c[m_vals > 0.5]
            
        if c.shape[0] > 0:
            heatmap[c[:, 1], c[:, 0]] = 1.0
        return heatmap