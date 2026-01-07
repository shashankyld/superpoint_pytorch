import cv2
import numpy as np
import matplotlib.pyplot as plt
from superpoint.datasets.synthetic.utils import shapes
from superpoint.datasets.synthetic.dataset import SyntheticDataset

def test_individual_shapes():
    class Config:
        height, width = 240, 320
        quality_level = 0.01
    
    ds = SyntheticDataset(Config())
    
    # MATCHING NAMES TO shapes.py
    # shape_tests = [
    #     ("Triangle", shapes.draw_triangle)
    #     ("Cube", shapes.draw_cube),
    #     ("Star", shapes.draw_star),
    #     ("Lines", shapes.draw_lines),
    #     ("Polygon", shapes.draw_polygon),
    #     ("Checkerboard", shapes.draw_checkerboard)
    #     ("Ellipse", shapes.draw_ellipse)
    # ]
    # 6 checkerboard tests
    shape_tests = [("Checkerboard", shapes.draw_checkerboard)] * 6

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (name, func) in enumerate(shape_tests):
        img = np.zeros((240, 320), dtype=np.uint8)
        pts = []
        func(img, pts)
        
        verified_pts = ds._verify_points(img, pts)
        axes[i].imshow(img, cmap='gray')
        
        raw_pts = np.array(pts)
        if len(raw_pts) > 0:
            # Analytical candidates in RED
            axes[i].scatter(raw_pts[:, 0], raw_pts[:, 1], s=40, color='red', marker='x', alpha=0.6)
        
        if len(verified_pts) > 0:
            # Veto-passed points in LIME
            axes[i].scatter(verified_pts[:, 0], verified_pts[:, 1], s=80, edgecolors='lime', facecolors='none', linewidths=2)
            
        axes[i].set_title(f"{name}\nRaw: {len(raw_pts)} | Verified: {len(verified_pts)}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_individual_shapes()