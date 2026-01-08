import os
import cv2
import torch
import numpy as np
import time
from tqdm import tqdm
from glob import glob
from superpoint.models.magicpoint import MagicPoint
from superpoint.training.homographic_adaptation import HomographicAdaptation

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    
    # 1. Setup Paths
    img_dir = os.path.join(root_dir, "data/coco/val2017")
    label_dir = os.path.join(root_dir, "data/coco/labels/val2017")
    os.makedirs(label_dir, exist_ok=True)
    
    img_paths = sorted(glob(os.path.join(img_dir, "*.jpg")))
    print(f"[*] Found {len(img_paths)} images. Starting Label Generation...")

    # 2. Load MagicPoint (Teacher)
    model = MagicPoint().to(device)
    ckpt_path = os.path.join(root_dir, "checkpoints/magicpoint_best.pth")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    # 3. Setup HA Engine (Chunked for Memory)
    ha = HomographicAdaptation(model, device, num_iter=100, sub_batch_size=10)
    conf_thresh = 0.015
    start_time = time.time()

    # 4. Processing Loop
    with torch.no_grad():
        for path in tqdm(img_paths):
            npy_name = os.path.basename(path).replace(".jpg", ".npy")
            save_path = os.path.join(label_dir, npy_name)
            
            if os.path.exists(save_path):
                continue

            # Load and Preprocess
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            img_res = cv2.resize(img, (320, 240))
            img_tensor = torch.from_numpy(img_res).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
            
            # Generate Heatmap
            heatmap_tensor, _ = ha(img_tensor)
            heatmap = heatmap_tensor.squeeze()

            # GPU NMS
            max_h = torch.nn.functional.max_pool2d(heatmap.unsqueeze(0).unsqueeze(0), 3, 1, 1)
            keep = (heatmap == max_h.squeeze()) & (heatmap > conf_thresh)
            
            # Save Coordinates [N, 2] -> (y, x)
            coords = torch.where(keep)
            pts = torch.stack(coords, dim=1).cpu().numpy()
            np.save(save_path, pts)

    avg_time = (time.time() - start_time) / len(img_paths)
    print(f"[*] Done! Avg time per image: {avg_time:.2f}s")

if __name__ == "__main__":
    main()