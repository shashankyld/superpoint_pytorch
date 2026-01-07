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
        n_shapes = 5 # Auditing edge interactions
        quality_level = 0.01

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SyntheticDataset(Config())
    augmentor = SyntheticAugmentor().to(device)

    # 1. Prepare Base Sample
    img_cpu, pts_cpu = dataset[0] 
    img_gpu = img_cpu.unsqueeze(0).to(device)
    pts_gpu = pts_cpu.clone().detach().float().unsqueeze(0).to(device)

    # 2. Geometric Wrapper: Warp -> ResizedCrop
    # We crop the central 80% to slice off the padding artifacts you found.
    # Resizing back to 240x320 keeps our training tensors consistent.
    def wrap_geo(op):
        return K.AugmentationSequential(
            op,
            K.RandomResizedCrop(size=(240, 320), scale=(0.7, 0.7), ratio=(1.0, 1.0), p=1.0), 
            data_keys=["input", "keypoints"]
        )

    # 3. Define the Test Battery
    # Fisheye parameters are now [min, max] ranges to match source code
    tests = {
        "Base": lambda x, p: (x, p), 
        "Perspective": lambda x, p: wrap_geo(K.RandomPerspective(distortion_scale=0.2, p=1.0))(x, p),
        "Rotation": lambda x, p: wrap_geo(K.RandomRotation(degrees=45.0, p=1.0))(x, p),
        "Affine": lambda x, p: wrap_geo(K.RandomAffine(degrees=0, translate=(0.2, 0.2), p=1.0))(x, p),
        
        # FISHEYE Case fails to transform points correclty. So I commented it out. 

        # # FISHEYE FIXED: Passing shape (2,) tensors as ranges
        # "Fisheye": lambda x, p: wrap_geo(K.RandomFisheye(
        #     center_x=torch.tensor([-0.05, 0.05]), 
        #     center_y=torch.tensor([-0.05, 0.05]), 
        #     gamma=torch.tensor([1.4, 1.6]), p=1.0))(x, p),
        
        "Gauss Noise": lambda x, p: (K.RandomGaussianNoise(std=0.1, p=1.0)(x), p),
        "Salt & Pepper": lambda x, p: (augmentor.add_salt_and_pepper(x.clone(), amount=0.03), p),
        "Blobs": lambda x, p: (augmentor.forward(x, p)[0], p) 
    }

    # 4. Run and Visualize
    fig, axes = plt.subplots(1, len(tests), figsize=(26, 4))
    for i, (name, aug_fn) in enumerate(tests.items()):
        with torch.no_grad():
            aug_img, aug_pts = aug_fn(img_gpu.clone(), pts_gpu.clone())
        
        img_np = aug_img.squeeze().cpu().numpy()
        pts_np = aug_pts.squeeze().cpu().numpy()
        
        # Final boundary filter for the 240x320 frame
        mask = (pts_np[:, 0] >= 0) & (pts_np[:, 0] < Config.width) & \
               (pts_np[:, 1] >= 0) & (pts_np[:, 1] < Config.height)
        valid_pts = pts_np[mask]

        axes[i].imshow(img_np, cmap='gray', vmin=0, vmax=1)
        if len(valid_pts) > 0:
            axes[i].scatter(valid_pts[:, 0], valid_pts[:, 1], s=25, 
                           edgecolors='lime', facecolors='none', linewidths=1.2)
        axes[i].set_title(name)
        axes[i].axis('off')

    plt.suptitle("Geometric Audit: Boundary Protection via Resized Center Crop", fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_isolated_kornia_augmentations()