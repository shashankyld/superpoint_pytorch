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
    Advanced visualizer: Converts 65-ch logits back to (x, y) coordinates.
    Shows Truth (Red) vs. Model Detections (Green).
    """
    # 1. Image preparation (B, 1, H, W) -> (H, W, 3)
    img = img_aug[0].squeeze().detach().cpu().numpy()
    img_u8 = (img * 255).astype(np.uint8)
    img_rgb_gt = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)
    img_rgb_pred = img_rgb_gt.copy()
    
    # 2. Ground Truth Overlay (Red)
    gt_pts = pts_aug[0].detach().cpu().numpy()
    for pt in gt_pts:
        if pt[0] >= 0: # Only draw real points, not -1 padding
            cv2.circle(img_rgb_gt, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)

    # 3. Prediction Heatmap Processing (Space-to-Depth Reversal)
    # Shape: (65, 30, 40) -> Softmax -> Remove Dustbin -> Reshape to (240, 320)
    prob = torch.nn.functional.softmax(logits[0], dim=0)
    heatmap = prob[:-1, :, :].view(8, 8, 30, 40).permute(2, 0, 3, 1).reshape(240, 320)
    heatmap_np = heatmap.detach().cpu().numpy()
    
    # 4. Extract Predicted Coordinates (Green)
    ys, xs = np.where(heatmap_np > thresh)
    for y, x in zip(ys, xs):
        cv2.circle(img_rgb_pred, (int(x), int(y)), 2, (0, 255, 0), -1)

    return {
        "visuals/gt_truth_overlay": wandb.Image(img_rgb_gt, caption=f"Truth Red (Ep{epoch} S{step})"),
        "visuals/confidence_heatmap": wandb.Image(heatmap_np, caption="Model Confidence"),
        "visuals/prediction_overlay": wandb.Image(img_rgb_pred, caption=f"Preds Green (Thresh {thresh})")
    }
def train():
    # 1. INITIALIZE W&B
    wandb.init(
        project="superpoint-pytorch",
        name="magicpoint-synthetic-weighted",
        config={
            "lr": 1e-3,
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
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_path = os.path.join(checkpoint_dir, "magicpoint_best.pth")
    
    if os.path.exists(best_path):
        print(f"[*] Found existing checkpoint at {best_path}. Resuming...")
        state_dict = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        print("[!] No checkpoint found. Starting training from scratch.")
    # --------------------

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
            
            # --- 10% CHECKPOINTING & UPDATING BEST ---
            if (i + 1) % save_freq == 0:
                pct = int(((i + 1) / total_batches) * 100)
                current_ckpt_name = f"magicpoint_ep{epoch}_p{pct}.pth"
                path = os.path.join(checkpoint_dir, current_ckpt_name)
                
                # Save the specific checkpoint
                torch.save(model.state_dict(), path)
                
                # Replace magicpoint_best.pth with this latest save
                shutil.copyfile(path, best_path)
                # print(f" [Checkpoint] Saved {current_ckpt_name} and updated magicpoint_best.pth")
            
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