import cv2
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from superpoint.models.magicpoint import MagicPoint
from superpoint.training.homographic_adaptation import HomographicAdaptation

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    
    # 1. Load Model
    model = MagicPoint().to(device)
    ckpt_path = os.path.join(root_dir, "checkpoints/magicpoint_best.pth")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    # 2. Load Image
    img_path = os.path.join(root_dir, "data/test_images/indoor.jpg")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not find image at {img_path}")
        return
    img_resized = cv2.resize(img, (320, 240))
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

    # 3. Run HA - CRITICAL: Added return_diversity=True
    ha = HomographicAdaptation(model, device, num_iter=50, sub_batch_size=10)
    print(f"[*] Running Vectorized HA (50 iterations in chunks of 10)...")
    adapted_heatmap_tensor, diversity_grid = ha(img_tensor, return_diversity=True)
    
    heatmap = adapted_heatmap_tensor.squeeze()
    
    # 4. Post-Processing (NMS)
    heatmap_batch = heatmap.unsqueeze(0).unsqueeze(0)
    max_h = torch.nn.functional.max_pool2d(heatmap_batch, kernel_size=3, stride=1, padding=1)
    
    # Lower threshold to 0.005 if you still see 0 points
    thresh = 0.015 
    keep = (heatmap == max_h.squeeze()) & (heatmap > thresh)
    
    pred_coords = torch.where(keep)
    ys, xs = pred_coords[0].cpu().numpy(), pred_coords[1].cpu().numpy()
    
    img_out = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    for y, x in zip(ys, xs):
        cv2.circle(img_out, (int(x), int(y)), 3, (0, 255, 0), -1)

    # 5. Visualize
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes[0, 0].set_title("1. Original Image")
    axes[0, 0].imshow(img_resized, cmap='gray')
    
    axes[0, 1].set_title("2. Homographic Variations (Sample 16)")
    if diversity_grid is not None:
        axes[0, 1].imshow(diversity_grid)
    else:
        axes[0, 1].text(0.5, 0.5, "Diversity Grid Not Generated", ha='center')
    axes[0, 1].axis('off')
    
    axes[1, 0].set_title("3. Aggregated Heatmap")
    im = axes[1, 0].imshow(heatmap.cpu().numpy(), cmap='jet')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    axes[1, 1].set_title(f"4. Final Pseudo-GT (Found {len(ys)} points)")
    axes[1, 1].imshow(img_out)

    plt.tight_layout()
    save_path = os.path.join(root_dir, "data/test_results/vectorized_ha_complete_test.png")
    plt.savefig(save_path)
    print(f"[*] Success! Found {len(ys)} points. Saved to {save_path}")

if __name__ == "__main__":
    main()