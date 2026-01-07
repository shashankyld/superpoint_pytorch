import torch
import wandb  
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from tqdm import tqdm
import os

from superpoint.datasets.synthetic import SyntheticDataset
from superpoint.datasets.synthetic.utils import SyntheticAugmentor
from superpoint.models.magicpoint import MagicPoint
from superpoint.training.losses import SuperPointLoss


def superpoint_collate(batch):
    """
    Pads keypoints with -1 so they can be stacked into a single batch.
    """
    images = [item[0] for item in batch]
    keypoints = [item[1] for item in batch]
    
    # 1. Stack images normally (they are all 240x320)
    images = torch.stack(images, 0)
    
    # 2. Find the maximum number of keypoints in this batch
    max_pts = max([p.shape[0] for p in keypoints])
    
    # 3. Pad each keypoint tensor with -1
    padded_keypoints = []
    for pts in keypoints:
        num_pts = pts.shape[0]
        # Create a padded tensor of shape (max_pts, 2)
        pad = torch.full((max_pts, 2), -1.0, dtype=torch.float32)
        pad[:num_pts, :] = pts
        padded_keypoints.append(pad)
        
    return images, torch.stack(padded_keypoints, 0)


def train():
    # 1. INITIALIZE W&B with all required dataset params
    wandb.init(
        project="superpoint-pytorch",
        name="magicpoint-synthetic-initial",
        config={
            "lr": 1e-3,
            "batch_size": 32,
            "epochs": 20,
            "height": 240,
            "width": 320,
            "n_shapes": 8,            # Added this
            "quality_level": 0.01,    # Fixed the missing attribute
            "architecture": "MagicPoint-VGG"
        }
    )
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. DATA & AUGMENTATION
    dataset = SyntheticDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, collate_fn=superpoint_collate)
    augmentor = SyntheticAugmentor(config.height, config.width).to(device)

    # 3. MODEL & LOSS
    model = MagicPoint().to(device)
    criterion = SuperPointLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # 4. TRAINING LOOP
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for i, batch in enumerate(pbar):
            img, pts = batch
            img, pts = img.to(device), pts.to(device)

            # Synchronized GPU Augmentation
            img_aug, pts_aug = augmentor(img, pts)

            # Forward & Loss
            logits = model(img_aug)
            loss, det_loss, desc_loss = criterion(
                            outputs={'logits': logits}, 
                            targets={'keypoints': pts_aug}
                        )      
                  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            # Log metrics to W&B every step
            wandb.log({"batch_loss": loss.item()})
            pbar.set_postfix(loss=loss.item())

            # 5. VISUAL DEBUGGING (Log images every 100 steps)
            if i % 100 == 0:
                log_visuals(img_aug, pts_aug, logits, epoch, i)

        avg_loss = epoch_loss / len(dataloader)
        wandb.log({"epoch": epoch, "avg_loss": avg_loss})
        
        torch.save(model.state_dict(), f"checkpoints/magicpoint_ep{epoch}.pth")

def log_visuals(img_aug, pts_aug, logits, epoch, step):
    """Helper to log Ground Truth vs Model Predictions to W&B."""
    # Grab the first image in the batch
    img_np = img_aug[0].squeeze().detach().cpu().numpy()
    
    # Process Logits to get confidence map (Softmax + Reshape)
    # We take the 65-ch output and reshape it back to HxW
    prob = torch.nn.functional.softmax(logits[0], dim=0)[:-1] # Remove dustbin
    prob = prob.view(8, 8, 30, 40).permute(2, 0, 3, 1).reshape(240, 320)
    prob_np = prob.detach().cpu().numpy()

    # Log to W&B dashboard
    wandb.log({
        "debug_viz": [
            wandb.Image(img_np, caption=f"Augmented Input (Ep {epoch})"),
            wandb.Image(prob_np, caption=f"Model Confidence Map (Step {step})")
        ]
    })

if __name__ == "__main__":
    train()