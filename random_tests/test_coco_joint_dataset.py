import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from superpoint.datasets.coco_joint import COCOJointDataset

def test_coco_joint_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    img_dir = os.path.join(root_dir, "data/coco/val2017")
    label_dir = os.path.join(root_dir, "data/coco/labels/val2017")

    dataset = COCOJointDataset(img_dir, label_dir)
    print(f"[*] Dataset size: {len(dataset)}")

    data = dataset[np.random.randint(len(dataset))]
    
    # Extract data
    img_a = data['image_a'].squeeze().numpy()
    img_b = data['image_b'].squeeze().numpy()
    mask_a = data['label_a'].squeeze().numpy()
    mask_b = data['label_b'].squeeze().numpy()
    
    # DEBUG: Check if points exist in the binary mask
    pts_a_y, pts_a_x = np.where(mask_a > 0.5)
    pts_b_y, pts_b_x = np.where(mask_b > 0.5)
    
    print(f"[*] DEBUG: Points in A: {len(pts_a_x)} | Points in B: {len(pts_b_x)}")

    if len(pts_a_x) == 0:
        print("[!] ERROR: No points found in Label A! Check your .npy files.")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Plot Image A
    axes[0].imshow(img_a, cmap='gray')
    axes[0].scatter(pts_a_x, pts_a_y, s=15, c='lime', edgecolors='black')
    axes[0].set_title(f"Image A ({len(pts_a_x)} pts)")
    
    # Plot Image B
    axes[1].imshow(img_b, cmap='gray')
    axes[1].scatter(pts_b_x, pts_b_y, s=15, c='cyan', edgecolors='black')
    axes[1].set_title(f"Image B ({len(pts_b_x)} pts)")

    save_path = os.path.join(root_dir, "data/test_results/joint_debug_viz.png")
    plt.savefig(save_path)
    print(f"[*] Viz saved to {save_path}")

if __name__ == "__main__":
    test_coco_joint_dataset()