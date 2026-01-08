import shutil
import cv2
import torch
import numpy as np
import wandb  
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

# Internal imports from your package structure
from superpoint.datasets.synthetic import SyntheticDataset
from superpoint.datasets.synthetic.utils import SyntheticAugmentor
from superpoint.models.magicpoint import MagicPoint
from superpoint.training.losses import SuperPointLoss

def superpoint_collate(batch):
    """
    Pads keypoints with -1 so they can be stacked into a single batch.
    Required because each synthetic image has a different number of corners.
    """
    images = torch.stack([item[0] for item in batch], 0)
    keypoints = [item[1] for item in batch]
    
    # Find the maximum number of keypoints in this specific batch
    max_pts = max([p.shape[0] for p in keypoints])
    
    padded_keypoints = []
    for pts in keypoints:
        num_pts = pts.shape[0]
        # Create a padded tensor of shape (max_pts, 2) filled with -1
        pad = torch.full((max_pts, 2), -1.0, dtype=torch.float32)
        pad[:num_pts, :] = pts
        padded_keypoints.append(pad)
        
    return images, torch.stack(padded_keypoints, 0)



def get_visual_logs(img_aug, pts_aug, logits, thresh, epoch, step):
    """
    Advanced visualizer: 
    1. Truth Red: The "Perfect Truth"
    2. Confidence Heatmap: The model's raw belief (grayscale)
    3. Preds Green: Raw detections (the "strings" on edges)
    4. Final Coords NMS: Cleaned detections (the isolated dots)
    """
    # 1. Image preparation
    img = img_aug[0].squeeze().detach().cpu().numpy()
    img_u8 = (img * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    
    # Create separate copies for different overlays
    img_gt = img_rgb.copy()
    img_raw_pred = img_rgb.copy()
    img_nms_pred = img_rgb.copy()
    
    # 2. Ground Truth Overlay (Red)
    gt_pts = pts_aug[0].detach().cpu().numpy()
    for pt in gt_pts:
        if pt[0] >= 0:
            cv2.circle(img_gt, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)

    # 3. Prediction Heatmap Processing (Space-to-Depth Reversal)
    prob = torch.nn.functional.softmax(logits[0], dim=0)
    heatmap = prob[:-1, :, :].view(8, 8, 30, 40).permute(2, 0, 3, 1).reshape(240, 320)
    heatmap_np = heatmap.detach().cpu().numpy()
    
    # 4. Raw Predicted Coordinates (The "Strings")
    ys_raw, xs_raw = np.where(heatmap_np > thresh)
    for y, x in zip(ys_raw, xs_raw):
        cv2.circle(img_raw_pred, (int(x), int(y)), 2, (0, 255, 0), -1)
    
    # 5. Non-Maximum Suppression (The "Stars")
    heatmap_batch = heatmap.unsqueeze(0).unsqueeze(0)
    max_h = torch.nn.functional.max_pool2d(heatmap_batch, kernel_size=3, stride=1, padding=1)
    keep = (heatmap == max_h.squeeze()) & (heatmap > thresh)
    
    pred_coords_nms = torch.where(keep)
    ys_nms, xs_nms = pred_coords_nms[0].cpu().numpy(), pred_coords_nms[1].cpu().numpy()
    
    for y, x in zip(ys_nms, xs_nms):
        cv2.circle(img_nms_pred, (int(x), int(y)), 2, (0, 255, 0), -1)

    return {
        "visuals/gt_truth_overlay": wandb.Image(img_gt, caption=f"Truth Red (Ep{epoch} S{step})"),
        "visuals/confidence_heatmap": wandb.Image(heatmap_np, caption="Raw Confidence Map"),
        "visuals/prediction_overlay_raw": wandb.Image(img_raw_pred, caption=f"Raw Preds (Threshold Only)"),
        "visuals/final_coords_nms": wandb.Image(img_nms_pred, caption=f"Final Points (NMS + Thresh {thresh})")
    }

def train():
    # 1. INITIALIZE W&B
    wandb.init(
        project="superpoint-pytorch",
        name="magicpoint-synthetic-weighted",
        config={
            "lr": 1e-6,
            "batch_size": 32,
            "epochs": 20,
            "height": 240,
            "width": 320,
            "n_shapes": 8,
            "quality_level": 0.01,
            "det_thresh": 0.015,
            "architecture": "MagicPoint-VGG"
        }
    )
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. DATASET & DATALOADER
    dataset = SyntheticDataset(config)
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=superpoint_collate
    )
    
    total_batches = len(dataloader)
    save_freq = max(1, total_batches // 10)
    
    # 3. MODEL & RESUME LOGIC
    model = MagicPoint().to(device)
    
    # --- RESUME CHECK ---
    # Calculate the project root (3 levels up from src/superpoint/scripts/)
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    checkpoint_dir = os.path.join(root_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "magicpoint_best.pth")

    start_epoch = 0
    if os.path.exists(best_path):
        print(f"[*] Found checkpoint at {best_path}. Resuming...")
        # Load everything: weights, optimizer momentum, and epoch count
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
    else:
        print("[!] Starting fresh.")

    augmentor = SyntheticAugmentor(config.height, config.width).to(device)
    criterion = SuperPointLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    print(f"Ready. Total batches: {total_batches}. Saving every {save_freq} batches.")

    # 4. TRAINING LOOP
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for i, batch in enumerate(pbar):
            img, pts = batch
            img, pts = img.to(device), pts.to(device)
            img_aug, pts_aug = augmentor(img, pts)

            logits = model(img_aug)
            loss, det_loss, _ = criterion(
                outputs={'logits': logits}, 
                targets={'keypoints': pts_aug}
            )      
                  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
                        
            # --- REPLACE YOUR SAVING LOGIC WITH THIS ---
            if (i + 1) % save_freq == 0:
                pct = int(((i + 1) / total_batches) * 100)
                path = os.path.join(checkpoint_dir, f"magicpoint_ep{epoch}_p{pct}.pth")
                
                # Save the full state dictionary
                ckpt_data = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(ckpt_data, path)
                shutil.copyfile(path, best_path)
            
            # --- LOGGING ---
            if i % 100 == 0:
                metrics = {
                    "batch_loss": loss.item(),
                    "det_loss": det_loss.item(),
                    "epoch": epoch,
                    "percent_of_epoch": (i / total_batches) * 100
                }
                visuals = get_visual_logs(img_aug, pts_aug, logits, config.det_thresh, epoch, i)
                metrics.update(visuals)
                wandb.log(metrics)
            
            pbar.set_postfix(loss=loss.item())

        # Save at end of epoch and update best
        final_epoch_path = os.path.join(checkpoint_dir, f"magicpoint_final_ep{epoch}.pth")
        torch.save(model.state_dict(), final_epoch_path)
        shutil.copyfile(final_epoch_path, best_path)

if __name__ == "__main__":
    train()