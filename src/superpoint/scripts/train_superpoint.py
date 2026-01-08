import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import numpy as np
import cv2
import os
from superpoint.models.superpoint import SuperPoint
from superpoint.datasets.coco_joint import COCOJointDataset
from superpoint.training.joint_losses import SuperPointJointLoss

def train():
    # Setup device and stability fixes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on {device}")
    
    # Initialize WandB
    wandb.init(project="SuperPoint-Joint", name="Descriptor-FineTune-v1")

    # 1. Init Model & Weights
    model = SuperPoint().to(device)
    ckpt_path = "checkpoints/magicpoint_best.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
        print(f"[*] Loaded pretrained detector from {ckpt_path}")

    # 2. Data - Using num_workers=0 for stability during debug
    dataset = COCOJointDataset(img_dir="data/coco/val2017", label_dir="data/coco/labels/val2017")
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    
    criterion = SuperPointJointLoss(device=device)

    # 3. Differential Learning Rates
    # Split parameters into pretrained (Slow) and new (Fast)
    desc_params = list(model.descriptor.parameters())
    base_params = [p for n, p in model.named_parameters() if 'descriptor' not in n]
    
    optimizer = optim.Adam([
        {'params': base_params, 'lr': 1e-5}, 
        {'params': desc_params, 'lr': 1e-3}
    ])

    model.train()
    for epoch in range(10):
        for i, batch in enumerate(loader):
            # Move to device
            img_a = batch['image_a'].to(device)
            img_b = batch['image_b'].to(device)
            
            optimizer.zero_grad()
            out_a, out_b = model(img_a), model(img_b)
            
            # Loss Calculation
            targets = {
                'label_a': batch['label_a'].to(device),
                'label_b': batch['label_b'].to(device),
                'homography': batch['homography'].to(device),
                'valid_mask': batch['valid_mask'].to(device)
            }
            
            loss, det_l, desc_l = criterion(out_a, out_b, targets)
            
            loss.backward()
            optimizer.step()

            # Periodic Logging
            if i % 20 == 0:
                print(f"E{epoch} B{i} | Total: {loss.item():.4f} | Det: {det_l:.4f} | Desc: {desc_l:.4f}")
                
                with torch.no_grad():
                    # Generate the visual debug image with actual matching
                    vis_img = log_visuals(img_a[0], img_b[0], out_a, out_b, batch['label_a'][0])
                    
                    wandb.log({
                        "losses/total": loss.item(), 
                        "losses/detector": det_l, 
                        "losses/descriptor": desc_l,
                        "visuals/matching_debug": wandb.Image(vis_img)
                    })

        # Save Checkpoint
        save_path = f"checkpoints/superpoint_e{epoch}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, save_path)
        print(f"[*] Saved checkpoint: {save_path}")
def log_visuals(img_a, img_b, out_a, out_b, gt_a):
    """Refined Visualizer: Color-coded coordinates for both sides."""
    # Convert to numpy
    img_a_np = (img_a.squeeze().cpu().numpy() * 255).astype(np.uint8)
    img_b_np = (img_b.squeeze().cpu().numpy() * 255).astype(np.uint8)
    
    def extract_features(logits, desc):
        prob = torch.nn.functional.softmax(logits, dim=0)[:-1, :, :]
        prob = prob.permute(1, 2, 0).reshape(30, 40, 8, 8).permute(0, 2, 1, 3).reshape(240, 320)
        
        # Get coordinates of top points
        mask = prob > 0.015
        pts_yx = torch.stack(torch.where(mask), dim=-1)
        
        # Grid sample descriptors
        grid = pts_yx.float()
        grid[:, 0] = (grid[:, 0] / (240/2)) - 1
        grid[:, 1] = (grid[:, 1] / (320/2)) - 1
        grid = grid.flip(-1).unsqueeze(0).unsqueeze(0)
        
        feat = torch.nn.functional.grid_sample(desc.unsqueeze(0), grid, align_corners=True)
        feat = feat.squeeze().transpose(0, 1)
        
        return pts_yx.cpu().numpy(), feat.cpu().numpy()

    pts_a, feat_a = extract_features(out_a['logits'][0], out_a['desc'][0])
    pts_b, feat_b = extract_features(out_b['logits'][0], out_b['desc'][0])

    # Side-by-side canvas
    canvas = np.hstack([cv2.cvtColor(img_a_np, cv2.COLOR_GRAY2RGB), 
                        cv2.cvtColor(img_b_np, cv2.COLOR_GRAY2RGB)])
    
    # 1. Left Side: Ground Truth (Green)
    gt_pts = torch.where(gt_a.squeeze() > 0.5)
    for y, x in zip(gt_pts[0], gt_pts[1]):
        cv2.circle(canvas, (int(x), int(y)), 2, (0, 255, 0), -1)

    # 2. Right Side: Predicted Points (Yellow)
    for y, x in pts_b:
        cv2.circle(canvas, (int(x + 320), int(y)), 2, (0, 255, 255), -1)

    # 3. Matching Lines
    if len(pts_a) > 0 and len(pts_b) > 0:
        # Descriptor matching (Nearest Neighbor)
        distances = 1.0 - np.dot(feat_a, feat_b.T)
        matches = np.argmin(distances, axis=1)
        
        # Draw top 40 matches
        num_to_draw = min(40, len(pts_a))
        draw_indices = np.random.choice(len(pts_a), num_to_draw, replace=False)
        
        for idx in draw_indices:
            p1 = (int(pts_a[idx][1]), int(pts_a[idx][0]))
            p2 = (int(pts_b[matches[idx]][1] + 320), int(pts_b[matches[idx]][0]))
            
            # Use distance to color the line (Magenta to Red)
            d = distances[idx, matches[idx]]
            color = (255, 0, 255) if d < 0.15 else (0, 0, 255)
            cv2.line(canvas, p1, p2, color, 1)

    return canvas


if __name__ == "__main__":
    train()