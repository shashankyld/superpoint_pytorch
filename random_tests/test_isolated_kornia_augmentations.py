import torch
import matplotlib.pyplot as plt
import numpy as np
import kornia.augmentation as K

from superpoint.datasets.synthetic import SyntheticDataset
from superpoint.datasets.synthetic.utils import SyntheticAugmentor

def test_isolated_kornia_augmentations():
    class Config:
        height = 240
        width = 320
        n_shapes = 1 
        quality_level = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticDataset(Config())
    augmentor = SyntheticAugmentor().to(device)

    # 1. FIXED: Standard indexing and clean tensor conversion
    img_cpu, pts_cpu = dataset[0] 
    img_gpu = img_cpu.unsqueeze(0).to(device)
    # Use clone().detach() to satisfy the PyTorch warning
    pts_gpu = pts_cpu.clone().detach().float().unsqueeze(0).to(device)

    # 2. FIXED: Wrapping geometric ops in AugmentationSequential to use data_keys
    # This ensures Kornia knows to move the points with the image pixels.
    def wrap_geo(op):
        return K.AugmentationSequential(op, data_keys=["input", "keypoints"])

    tests = {
        "Base (Clean)": lambda x, p: (x, p),
        
        # Geometric (Coordinates move)
        "Perspective": lambda x, p: wrap_geo(K.RandomPerspective(distortion_scale=0.2, p=1.0))(x, p),
        # Add random rotation test, random affine test
        "Rotation": lambda x, p: wrap_geo(K.RandomRotation(degrees=30.0, p=1.0))(x, p),
        "Affine": lambda x, p: wrap_geo(K.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), p=1.0))(x, p),
        
        # Photometric (Coordinates stay fixed)
        "Gauss Noise": lambda x, p: (K.RandomGaussianNoise(std=0.08, p=1.0)(x), p),
        "Motion Blur": lambda x, p: (K.RandomMotionBlur(kernel_size=5, angle=45., direction=0.5, p=1.0)(x), p),
        "Salt & Pepper": lambda x, p: (augmentor.add_salt_and_pepper(x.clone(), amount=0.03), p),
        "Diffused Blobs": lambda x, p: (augmentor.forward(x, p)[0], p) 
    }

    # 3. Visualization logic
    fig, axes = plt.subplots(1, len(tests), figsize=(22, 4))
    
    for i, (name, aug_fn) in enumerate(tests.items()):
        with torch.no_grad():
            aug_img, aug_pts = aug_fn(img_gpu.clone(), pts_gpu.clone())
        
        img_np = aug_img.squeeze().cpu().numpy()
        pts_np = aug_pts.squeeze().cpu().numpy()
        
        # Filter for points inside the image boundary
        mask = (pts_np[:, 0] >= 0) & (pts_np[:, 0] < Config.width) & \
               (pts_np[:, 1] >= 0) & (pts_np[:, 1] < Config.height)
        valid_pts = pts_np[mask]

        axes[i].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        if len(valid_pts) > 0:
            axes[i].scatter(valid_pts[:, 0], valid_pts[:, 1], s=40, 
                           edgecolors='lime', facecolors='none', linewidths=2)
        
        axes[i].set_title(name)
        axes[i].axis('off')

    plt.suptitle("Isolated Kornia Augmentation Audit - Fix Applied", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_isolated_kornia_augmentations()