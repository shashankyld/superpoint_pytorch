import torch
import torch.nn as nn
import torch.nn.functional as F

class JointDetectorLoss(nn.Module):
    def __init__(self, cell_size=8):
        super().__init__()
        self.cell_size = cell_size
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, heatmap, valid_mask):
        B, C, Hc, Wc = logits.shape
        # Space-to-Depth
        labels = heatmap.view(B, 1, Hc, 8, Wc, 8).permute(0, 2, 4, 1, 3, 5).reshape(B, Hc, Wc, 64)
        dustbin = 1.0 - torch.max(labels, dim=-1, keepdim=True)[0]
        targets = torch.argmax(torch.cat([labels, dustbin], dim=-1), dim=-1)
        
        loss = self.cross_entropy(logits, targets)
        mask_down = F.max_pool2d(valid_mask, kernel_size=8, stride=8).squeeze(1)
        return (loss * mask_down).mean()

class JointDescriptorLoss(nn.Module):
    def __init__(self, margin_pos=1.0, margin_neg=0.2, lambda_d=0.0001, threshold=8):
        super().__init__()
        self.m_pos = margin_pos
        self.m_neg = margin_neg
        self.lambda_d = lambda_d
        self.threshold = threshold # pixel distance to consider a "match"

    def forward(self, desc_a, desc_b, H, valid_mask):
        B, C, Hc, Wc = desc_a.shape
        device = desc_a.device

        # 1. Create Grid of centers for Image A
        grid = torch.stack(torch.meshgrid(torch.arange(Hc, device=device), 
                                         torch.arange(Wc, device=device), indexing='ij'), dim=-1).float()
        centers_a = (grid * 8 + 4).view(-1, 2).flip(-1).repeat(B, 1, 1) # (B, 1200, 2)

        # 2. Warp centers from A to B
        ones = torch.ones(B, centers_a.shape[1], 1, device=device)
        centers_a_h = torch.cat([centers_a, ones], dim=-1)
        warped_centers = torch.matmul(H, centers_a_h.transpose(1, 2)).transpose(1, 2)
        warped_centers = warped_centers[:, :, :2] / warped_centers[:, :, 2:] # (B, 1200, 2)

        # 3. Compute Distance between every cell in A and every cell in B
        da = desc_a.view(B, C, -1).transpose(1, 2) # (B, 1200, 256)
        db = desc_b.view(B, C, -1).transpose(1, 2) # (B, 1200, 256)
        
        # Cosine similarity matrix (1.0 = same, 0.0 = different)
        sim = torch.matmul(da, db.transpose(1, 2))
        
        # 4. Define Matches (S_hw matrix from paper)
        # centers_b is just the same grid we used for centers_a
        centers_b = centers_a[0].unsqueeze(0) # (1, 1200, 2)
        # Distance in pixels between warped A centers and all B centers
        dist_pixels = torch.norm(warped_centers.unsqueeze(2) - centers_b.unsqueeze(1), dim=-1)
        s_mask = (dist_pixels <= self.threshold).float() # (B, 1200, 1200)

        # 5. Hinge Loss
        # Positive: minimize distance (maximize similarity) for matches
        # Negative: ensure distance > margin for non-matches
        loss_pos = s_mask * F.relu(self.m_pos - sim)
        loss_neg = (1 - s_mask) * F.relu(sim - self.m_neg)
        
        return (loss_pos + loss_neg).mean()

class SuperPointJointLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.detector_loss = JointDetectorLoss()
        self.descriptor_loss = JointDescriptorLoss()

    def forward(self, out_a, out_b, targets):
        det_a = self.detector_loss(out_a['logits'], targets['label_a'], torch.ones_like(targets['valid_mask']))
        det_b = self.detector_loss(out_b['logits'], targets['label_b'], targets['valid_mask'])
        
        desc_l = self.descriptor_loss(out_a['desc'], out_b['desc'], targets['homography'], targets['valid_mask'])
        
        total = (det_a + det_b)/2 + (0.0001 * desc_l)
        return total, (det_a + det_b)/2, desc_l