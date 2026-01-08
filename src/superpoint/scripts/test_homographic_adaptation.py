import cv2
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from superpoint.models.magicpoint import MagicPoint
from superpoint.training.homographic_adaptation import HomographicAdaptation

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Pathing relative to your specific Legion folder structure
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    
    # 1. Load Model with "Smart Loader"
    model = MagicPoint().to(device)
    ckpt_path = os.path.join(root_dir, "checkpoints/magicpoint_best.pth")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # 2. Load Real Image
    img_path = os.path.join(root_dir, "data/test_images/Kolkata.jpg")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not find image at {img_path}")
        return
    img_resized = cv2.resize(img, (320, 240))
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

    # 3. Run Vectorized Homographic Adaptation
    ha = HomographicAdaptation(model, device, num_iter=50)
    print(f"[*] Running Vectorized HA (50 warps parallel)...")
    adapted_heatmap_tensor, diversity_grid = ha(img_tensor)
    
    # 4. Post-Processing (NMS)
    # Move heatmap to CPU for processing
    heatmap = adapted_heatmap_tensor.squeeze() # (240, 320)
    
    # Apply simple NMS on GPU
    heatmap_batch = heatmap.unsqueeze(0).unsqueeze(0)
    max_h = torch.nn.functional.max_pool2d(heatmap_batch, kernel_size=3, stride=1, padding=1)
    
    # Thresholding (You can tune 0.015 based on results)
    thresh = 0.015 
    keep = (heatmap == max_h.squeeze()) & (heatmap > thresh)
    
    # Extract coordinates
    pred_coords = torch.where(keep)
    ys, xs = pred_coords[0].cpu().numpy(), pred_coords[1].cpu().numpy()
    
    # Prepare image for drawing
    img_out = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    for y, x in zip(ys, xs):
        cv2.circle(img_out, (int(x), int(y)), 3, (0, 255, 0), -1)

    # 5. Visualize (2x2 Grid)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Top-Left: Original
    axes[0, 0].set_title("1. Original Image")
    axes[0, 0].imshow(img_resized, cmap='gray')
    
    # Top-Right: Diversity montage
    axes[0, 1].set_title("2. Homographic Variations Seen (Sample 16)")
    axes[0, 1].imshow(diversity_grid)
    axes[0, 1].axis('off')
    
    # Bottom-Left: Heatmap
    axes[1, 0].set_title("3. Aggregated Heatmap (Teacher Votes)")
    im = axes[1, 0].imshow(heatmap.cpu().numpy(), cmap='jet')
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Bottom-Right: Final Points on Image
    axes[1, 1].set_title(f"4. Final Pseudo-Ground Truth (NMS, Thresh {thresh})")
    axes[1, 1].imshow(img_out)

    # Save results
    results_dir = os.path.join(root_dir, "data/test_results")
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, "vectorized_ha_complete_test.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[*] Success! Found {len(ys)} points. Results saved to {save_path}")

if __name__ == "__main__":
    main()