import cv2
import torch
import numpy as np
import os
import glob
from superpoint.models.magicpoint import MagicPoint

def preprocess_image(img_path, device):
    # Load as grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    
    # Resize to the model's training resolution
    img_resized = cv2.resize(img, (320, 240))
    
    # Normalize and convert to tensor
    img_tensor = torch.from_numpy(img_resized).float().unsqueeze(0).unsqueeze(0) / 255.0
    return img_tensor.to(device), img_resized

def run_inference(model, img_tensor, thresh=0.015):
    with torch.no_grad():
        logits = model(img_tensor)
        
        # 1. Space-to-Depth Reversal
        prob = torch.nn.functional.softmax(logits[0], dim=0)
        heatmap = prob[:-1, :, :].view(8, 8, 30, 40).permute(2, 0, 3, 1).reshape(240, 320)
        
        # 2. NMS
        heatmap_batch = heatmap.unsqueeze(0).unsqueeze(0)
        max_h = torch.nn.functional.max_pool2d(heatmap_batch, kernel_size=3, stride=1, padding=1)
        keep = (heatmap == max_h.squeeze()) & (heatmap > thresh)
        
        pred_coords = torch.where(keep)
        return pred_coords[0].cpu().numpy(), pred_coords[1].cpu().numpy()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = MagicPoint().to(device)
    best_path = "checkpoints/magicpoint_best.pth"
    if not os.path.exists(best_path):
        print("Error: magicpoint_best.pth not found!")
        return
    
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    model.eval()
    
    # 2. Setup Image Paths (Create a folder called 'test_images')
    os.makedirs("data/test_results", exist_ok=True)
    image_paths = glob.glob("data/test_images/*.jpg") + glob.glob("data/test_images/*.png")
    
    for path in image_paths:
        img_tensor, img_orig = preprocess_image(path, device)
        if img_tensor is None: continue
        
        # 3. Detect Corners
        ys, xs = run_inference(model, img_tensor)
        
        # 4. Visualize
        img_out = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2RGB)
        for y, x in zip(ys, xs):
            cv2.circle(img_out, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        # 5. Save
        fname = os.path.basename(path)
        cv2.imwrite(f"data/test_results/pred_magicpoint_{fname}", img_out)
        print(f"Processed {fname}: Found {len(ys)} points.")

if __name__ == "__main__":
    main()