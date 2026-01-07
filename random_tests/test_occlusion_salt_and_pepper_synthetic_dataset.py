import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_salt_and_pepper(image, amount=0.01):
    output = image.copy()
    num_salt = np.ceil(amount * image.size * 0.5).astype(int)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    output[tuple(coords)] = 255
    num_pepper = np.ceil(amount * image.size * 0.5).astype(int)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    output[tuple(coords)] = 0
    return output

def get_line_intersection(p1, p2, p3, p4):
    """Analytical intersection for simple segments."""
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = p3; x4, y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        return np.array([x1 + ua*(x2-x1), y1 + ua*(y2-y1)])
    return None

def generate_audited_sample(h=240, w=320):
    img = np.zeros((h, w), dtype=np.uint8)
    candidates = []
    lines = []

    # 1. Simple Shapes (Triangles/Quads)
    for _ in range(np.random.randint(2, 4)):
        num_v = np.random.randint(3, 5)
        pts = np.random.randint(30, [w-30, h-30], size=(num_v, 2))
        cv2.fillPoly(img, [pts], 255)
        candidates.extend(pts.astype(float))
        for i in range(num_v):
            lines.append((pts[i], pts[(i+1)%num_v]))

    # 2. Simple Lines
    for _ in range(np.random.randint(2, 5)):
        p1 = np.random.randint(10, [w-10, h-10])
        p2 = np.random.randint(10, [w-10, h-10])
        cv2.line(img, tuple(p1), tuple(p2), 255, 1)
        candidates.append(p1.astype(float))
        candidates.append(p2.astype(float))
        lines.append((p1, p2))

    # 3. Intersection Check
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            inter = get_line_intersection(lines[i][0], lines[i][1], lines[j][0], lines[j][1])
            if inter is not None:
                candidates.append(inter)

    # 4. The Verification Filter (Audited)
    # We use a high quality level to be strict about what is a "corner"
    cv_corners = cv2.goodFeaturesToTrack(img, maxCorners=100, qualityLevel=0.1, minDistance=10)
    cv_corners = cv_corners.reshape(-1, 2) if cv_corners is not None else np.array([])

    final_verified = []
    rejected = []

    if len(candidates) > 0:
        # Remove duplicates from candidates first
        candidates = np.unique(np.array(candidates), axis=0)
        
        if len(cv_corners) == 0:
             # If OpenCV sees nothing, rejected everything
            rejected = candidates.tolist()
        else:
            for cand in candidates:
                dists = np.linalg.norm(cv_corners - cand, axis=1)
                # Check 1: Is it mathematically stable? (close to OpenCV point)
                if np.min(dists) < 5.0: 
                    # Check 2: Visibility (is it occluded?)
                    ix, iy = int(round(cand[0])), int(round(cand[1]))
                    if 0 <= ix < w and 0 <= iy < h and img[iy, ix] > 0:
                        final_verified.append(cand)
                    else:
                        rejected.append(cand) # mathematically good, but occluded
                else:
                    rejected.append(cand) # mathematically weak

    # Format outputs
    verified_np = np.array(final_verified) if final_verified else np.array([]).reshape(0, 2)
    rejected_np = np.array(rejected) if rejected else np.array([]).reshape(0, 2)

    return add_salt_and_pepper(img, 0.01), verified_np, rejected_np

if __name__ == "__main__":
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i in range(3):
        image, verified_pts, rejected_pts = generate_audited_sample()
        
        axes[i].imshow(image, cmap='gray')
        
        # Plot Rejected first (Red, smaller, thinner)
        if len(rejected_pts) > 0:
            axes[i].scatter(rejected_pts[:, 0], rejected_pts[:, 1], 
                            s=50, edgecolors='red', facecolors='none', linewidths=1, label='Rejected')
            
        # Plot Verified second (Green, larger, thicker)
        if len(verified_pts) > 0:
            axes[i].scatter(verified_pts[:, 0], verified_pts[:, 1], 
                            s=100, edgecolors='lime', facecolors='none', linewidths=2, label='Verified')
            
        axes[i].set_title(f"Sample {i+1}: {len(verified_pts)} kept, {len(rejected_pts)} rejected")
        if i == 0: axes[i].legend() # Only show legend on first plot

    plt.tight_layout()
    plt.show()