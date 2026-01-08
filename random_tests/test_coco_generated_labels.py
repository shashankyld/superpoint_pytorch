import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def test_coco_generated_labels():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    img_dir = os.path.join(root_dir, "data/coco/val2017")
    label_dir = os.path.join(root_dir, "data/coco/labels/val2017")
    
    if not os.path.exists(label_dir):
        print(f"Error: {label_dir} not found. Run generation script first.")
        return

    label_files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
    if not label_files:
        print("No labels found yet.")
        return

    samples = random.sample(label_files, min(4, len(label_files)))
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, label_name in enumerate(samples):
        img_name = label_name.replace(".npy", ".jpg")
        img_path = os.path.join(img_dir, img_name)
        pts = np.load(os.path.join(label_dir, label_name)) 

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Scaling labels (240x320) to original image size
        h_orig, w_orig = img.shape[:2]
        y_scale, x_scale = h_orig / 240.0, w_orig / 320.0

        axes[i].imshow(img)
        axes[i].scatter(pts[:, 1] * x_scale, pts[:, 0] * y_scale, s=20, c='lime', edgecolors='black')
        axes[i].set_title(f"ID: {img_name}\nPoints: {len(pts)}")
        axes[i].axis('off')

    plt.tight_layout()
    save_path = os.path.join(root_dir, "data/test_results/dataset_verify_coco.png")
    plt.savefig(save_path)
    print(f"[*] Verification saved to {save_path}")

if __name__ == "__main__":
    test_coco_generated_labels()