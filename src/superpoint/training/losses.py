import torch
import torch.nn as nn

class DetectorLoss(nn.Module):
    """65-channel classification loss for the MagicPoint head."""
    def __init__(self, cell_size=8):
        super().__init__()
        self.cell_size = cell_size
        weights = torch.ones(65)
        weights[64] = 1.0  # Background weight
        weights[:64] = 100.0 # Corner weight
        # This makes the tensor "visible" to the .to(device) call
        self.register_buffer('loss_weights', weights)
        self.cross_entropy = nn.CrossEntropyLoss(weight=self.loss_weights, reduction='mean')

    def points_to_labels(self, points, h, w):
            B = points.shape[0]
            hc, wc = h // self.cell_size, w // self.cell_size
            
            labels = torch.full((B, hc, wc), 64, dtype=torch.long, device=points.device)
            
            for b in range(B):
                pts = points[b]
                mask = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
                valid_pts = pts[mask]
                
                if valid_pts.shape[0] == 0: continue

                # Ensure these are long tensors for indexing
                x, y = valid_pts[:, 0], valid_pts[:, 1]
                gx, gy = (x // self.cell_size).long(), (y // self.cell_size).long()
                lx, ly = (x % self.cell_size).long(), (y % self.cell_size).long()
                
                local_idx = ly * self.cell_size + lx
                
                # The "Perfect Truth" Fix: Indexing with long tensors directly
                labels[b, gy, gx] = local_idx
                
            return labels

    def forward(self, logits, target_points):
        B, C, Hc, Wc = logits.shape
        labels = self.points_to_labels(target_points, Hc * 8, Wc * 8)
        return self.cross_entropy(logits, labels)


class DescriptorLoss(nn.Module):
    """Placeholder for Descriptor Loss (Phase 3)."""
    def __init__(self):
        super().__init__()
        # Usually a Hinge loss or Triplet loss based on homographic warping
        pass

    def forward(self, descriptors, warped_descriptors, homography):
        # We will implement this once we start Phase 3
        return torch.tensor(0.0, device=descriptors.device)


class SuperPointLoss(nn.Module):
    """Joint Loss that manages both heads."""
    def __init__(self, det_weight=1.0, desc_weight=1.0):
        super().__init__()
        self.detector_loss = DetectorLoss()
        self.descriptor_loss = DescriptorLoss()
        self.det_weight = det_weight
        self.desc_weight = desc_weight

    def forward(self, outputs, targets):
        """
        Args:
            outputs: dict containing 'logits' (and later 'descriptors')
            targets: dict containing 'keypoints' (and later 'warped_keypoints')
        """
        # 1. Detector Loss
        det_loss = self.detector_loss(outputs['logits'], targets['keypoints'])
        
        # 2. Descriptor Loss (Phase 3)
        desc_loss = torch.tensor(0.0, device=det_loss.device)
        if 'descriptors' in outputs:
            desc_loss = self.descriptor_loss(outputs['descriptors'], 
                                            outputs['warped_descriptors'], 
                                            targets['homography'])

        # 3. Weighted Sum
        total_loss = (self.det_weight * det_loss) + (self.desc_weight * desc_loss)
        
        return total_loss, det_loss, desc_loss