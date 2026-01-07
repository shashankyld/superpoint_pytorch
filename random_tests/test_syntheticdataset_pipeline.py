import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# These work because of your 'uv pip install -e .' structure
from superpoint.datasets.synthetic import SyntheticDataset
from superpoint.datasets.synthetic.utils import SyntheticAugmentor

def collate_fn(batch):
    """
    Standardizes variable point counts by padding with -1.0.
    """
    images, points_list = zip(*batch)
    images = torch.stack(images, 0) # (B, 1, H, W)
    
    # Pad to the maximum number of points present in this batch
    max_pts = max([len(pts) for pts in points_list])
    padded_pts = torch.full((len(batch), max_pts, 2), -1.0)
    
    for i, pts in enumerate(points_list):
        if len(pts) > 0:
            padded_pts[i, :len(pts)] = torch.tensor(pts)
            
    return images, padded_pts

def visualize_batch(images, points_batch):
    batch_size = images.shape[0]
    cols = 4
    rows = (batch_size + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()

    for i in range(batch_size):
        img = images[i].cpu().squeeze().numpy()
        pts = points_batch[i].cpu().numpy()

        # Filter out the -1.0 padding before displaying
        mask = pts[:, 0] >= 0
        valid_pts = pts[mask]

        axes[i].imshow(img, cmap='gray')
        if len(valid_pts) > 0:
            # We use 'none' facecolors to see the actual corner pixels clearly
            axes[i].scatter(valid_pts[:, 0], valid_pts[:, 1], s=30, 
                           edgecolors='lime', facecolors='none', linewidths=1.5)
        
        axes[i].set_title(f"Sample {i+1} | Pts: {len(valid_pts)}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def test_pipeline():
    # MODIFIED LOCAL CONFIG: Added missing attributes for the Dataset class
    class Config:
        height = 240              # Height
        width = 320              # Width
        n_shapes = 8         # Number of shapes per image
        quality_level = 0.01 # Used for the optional OpenCV snap
        batch_size = 8       # Batch size for DataLoader
        num_workers = 4      # Parallel loading threads

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running Test on {device} ---")

    # 1. Instantiate Dataset with the updated config
    dataset = SyntheticDataset(Config())
    
    # 2. Use our proper collate_fn for padded tensors
    loader = DataLoader(
        dataset, 
        batch_size=Config.batch_size, 
        num_workers=Config.num_workers,
        collate_fn=collate_fn
    )
    
    # 3. Instantiate GPU Augmentor
    augmentor = SyntheticAugmentor().to(device)

    # Fetch one batch from the CPU generator
    images, points = next(iter(loader))
    print(f"✓ CPU Generation successful. Padded Points Shape: {points.shape}")

    # 4. Move to GPU and Augment together
    images_gpu = images.to(device)
    points_gpu = points.to(device)
    
    with torch.no_grad():
        # augmented_points now reflect the geometric warps (deformations)
        augmented_images, augmented_points = augmentor(images_gpu, points_gpu)
    
    print(f"✓ GPU Augmentation successful. Points deformed correctly.")

    # 5. Visualize to audit the results
    visualize_batch(augmented_images, augmented_points)

if __name__ == "__main__":
    test_pipeline()