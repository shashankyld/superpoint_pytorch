# Wrapper for the synthetic shapes - pytorch dataset 
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from .utils import shapes 

class SyntheticDataset(Dataset):
    def __init__(self, config):
        self.h, self.w = config.height, config.width
        self.q_level = config.quality_level
        self.n_shapes = config.n_shapes

    def __len__(self):
        return 100000 # Infinite generator



    def _verify_points(self, img, candidates):
        """
        Hybrid Verification Pipeline:
        1. Snaps to high-quality OpenCV corners only if very close (drift prevention).
        2. Fallback to analytical math only if the area shows gradient energy (occlusion check).
        3. Final NMS at 5px to prevent label collisions in the 8x8 grid.
        """
        if not candidates: return np.array([]).reshape(0, 2)
        candidates = np.array(candidates)

        # 1. Boundary Clip (2px safety margin)
        mask = (candidates[:, 0] >= 2) & (candidates[:, 0] < self.w - 2) & \
            (candidates[:, 1] >= 2) & (candidates[:, 1] < self.h - 2)
        candidates = candidates[mask]
        if len(candidates) == 0: return np.array([]).reshape(0, 2)
        
        # 2. High-Sensitivity Detection (Suggestion Pool)
        # Increased qualityLevel to 0.01 to ensure we only snap to 'real' peaks
        corners = cv2.goodFeaturesToTrack(
            img, maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=3
        )
        
        final_candidates_list = []
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        for cand in candidates:
            snapped = False
            # A. Attempt Snap: Only if a strong corner is within 2.0 pixels
            if corners is not None:
                corners_2d = corners.reshape(-1, 2)
                dists = np.linalg.norm(corners_2d - cand, axis=1)
                
                if np.min(dists) < 2.0:
                    match = corners_2d[np.argmin(dists)].reshape(1, 1, 2)
                    refined = cv2.cornerSubPix(img, match, (5, 5), (-1, -1), criteria)
                    final_candidates_list.append(refined.reshape(2))
                    snapped = True
            
            # B. Analytical Fallback with Occlusion/Energy Check
            if not snapped:
                ix, iy = int(round(cand[0])), int(round(cand[1]))
                # 5x5 patch check
                patch = img[iy-2:iy+3, ix-2:ix+3].astype(np.float32)
                grad_x = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
                
                # Must have energy in both directions to be a visible junction
                if np.mean(grad_x**2) > 15.0 and np.mean(grad_y**2) > 15.0:
                    final_candidates_list.append(cand)
                    
        if not final_candidates_list: return np.array([]).reshape(0, 2)
        
        # 3. Final NMS (5.0px) 
        final_candidates = np.array(final_candidates_list)
        final_candidates = final_candidates[np.lexsort((final_candidates[:, 1], final_candidates[:, 0]))]
        
        final_verified = []
        for p in final_candidates:
            if not final_verified:
                final_verified.append(p)
            else:
                dists = np.linalg.norm(np.array(final_verified) - p, axis=1)
                if np.min(dists) >= 5.0: 
                    final_verified.append(p)
                    
        return np.array(final_verified)

    def __getitem__(self, idx):
        img = np.zeros((self.h, self.w), dtype=np.uint8) 
        # Set random color background
        img[:] = np.random.randint(0, 50)
        pts = []
        
        funcs = [shapes.draw_triangle, shapes.draw_cube, shapes.draw_star, 
                 shapes.draw_checkerboard, shapes.draw_lines, shapes.draw_polygon, shapes.draw_ellipse]
        
        # Draw 2-5 random shapes per frame
        for func in np.random.choice(funcs, size=self.n_shapes, replace=True):
            func(img, pts)
            
        final_pts = self._verify_points(img, pts)
        img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        
        return img_tensor, torch.from_numpy(final_pts)