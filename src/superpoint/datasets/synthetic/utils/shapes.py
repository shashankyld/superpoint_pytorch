import cv2
import numpy as np

'''
# triangle shape - 3 corners as keypoints
# ellipse shape - (2 corners on Major Axis, 2 corners on Minor Axis -- only if eccentricity is greater than 0.9), centre
# cube reprojected to 2D with hidden occluded edges
# checkerboard pattern
# stars - one point as fixed corner and [1 - 10] new points sampled at random distance 
# Lines - Multiple lines and their intersections and end points
# Random polygon [4-7] sides and their corners
'''

import cv2
import numpy as np

def get_line_intersection(p1, p2, p3, p4):
    """Analytical intersection of two segments."""
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = p3; x4, y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        return np.array([x1 + ua*(x2-x1), y1 + ua*(y2-y1)])
    return None

def _is_visible(pts, w, h):
    return np.any((pts[:, 0] >= 0) & (pts[:, 0] < w) & 
                  (pts[:, 1] >= 0) & (pts[:, 1] < h))

def draw_triangle(img, points_list):
    h, w = img.shape[:2]
    while True:
        anchor = np.random.randint(0, [w, h])
        pts = anchor + np.random.randint(-100, 100, size=(3, 2))
        if _is_visible(pts, w, h):
            cv2.fillPoly(img, [pts.astype(np.int32)], 255)
            points_list.extend(pts.astype(float).tolist())
            break


def draw_cube(img, points_list):
    h_img, w_img = img.shape[:2]
    
    # 1. Randomization of Position and Dimensions
    anchor_x = np.random.randint(40, w_img - 40)
    anchor_y = np.random.randint(40, h_img - 40)
    focal = 300 
    l, b, h_c = np.random.randint(60, 100, size=3)
    
    # 2. Geometry: Defined with consistent winding for outward normals
    vertices = np.array([
        [0,0,0], [l,0,0], [l,b,0], [0,b,0],         # Bottom indices 0-3
        [0,0,h_c], [l,0,h_c], [l,b,h_c], [0,b,h_c]   # Top indices 4-7
    ]) - [l/2, b/2, h_c/2]

    face_indices_list = [
        [0, 3, 2, 1], [4, 5, 6, 7], # Bottom, Top
        [0, 1, 5, 4], [1, 2, 6, 5], # Front, Right
        [2, 3, 7, 6], [3, 0, 4, 7]  # Back, Left
    ]

    # 3. Apply Random Rotation
    R, _ = cv2.Rodrigues(np.random.randn(3) * 0.5)
    rotated = vertices @ R.T
    rotated[:, 2] += focal 
    
    # 4. Perspective Projection to 2D
    proj = np.zeros((8, 2))
    proj[:, 0] = (rotated[:, 0] * focal / rotated[:, 2]) + anchor_x
    proj[:, 1] = (rotated[:, 1] * focal / rotated[:, 2]) + anchor_y

    # 5. Shading with Diversity: High-contrast random grays
    colors = np.linspace(60, 220, 6)
    np.random.shuffle(colors)
    
    # 6. Back-face Culling & Visibility Mapping
    face_data = []
    vertex_visibility_count = np.zeros(8) # Track how many visible faces share a vertex

    for i, face_idx in enumerate(face_indices_list):
        # Calculate normal to determine camera-facing orientation
        v0, v1, v2 = rotated[face_idx[0]], rotated[face_idx[1]], rotated[face_idx[2]]
        normal = np.cross(v1 - v0, v2 - v0)
        
        if normal[2] < 0: # Face is pointing toward the camera
            avg_z = np.mean(rotated[face_idx, 2])
            face_data.append((avg_z, face_idx, int(colors[i])))
            # Increment visibility count for all vertices in this face
            for v_idx in face_idx:
                vertex_visibility_count[v_idx] += 1
    
    # 7. Painter's Algorithm: Sort and Draw visible faces
    face_data.sort(key=lambda x: x[0], reverse=True)

    for _, face_indices, color in face_data:
        poly_pts = proj[face_indices].astype(np.int32)
        cv2.fillPoly(img, [poly_pts], color)

    # 8. CRITICAL FIX: Only export vertices that are NOT occluded
    # A vertex is only a 'true' corner if it is part of at least two visible faces
    # or it is an outer corner on a visible face with sufficient contrast
    for v_idx in range(8):
        if vertex_visibility_count[v_idx] >= 1:
            # We add it as a candidate; _verify_points will then check patch variance
            # and gradient energy to finalize the decision.
            points_list.append(proj[v_idx].tolist())


def draw_star(img, points_list):
    h, w = img.shape[:2]
    while True:
        center = np.random.randint(-5, [w+5, h+5])
        tips = []
        num_tips = np.random.randint(3, 10)
        for _ in range(num_tips):
            ang = np.random.uniform(0, 2*np.pi)
            dist = np.random.randint(30, 100)
            tips.append([center[0] + dist*np.cos(ang), center[1] + dist*np.sin(ang)])
        
        if _is_visible(np.array([center] + tips), w, h):
            # Randomize the brightness of the star's lines for each line drawn
            for tip in tips:
                star_brightness = np.random.randint(100, 240)
                cv2.line(img, tuple(center.astype(int)), (int(tip[0]), int(tip[1])), star_brightness, np.random.randint(1, 3))
            
            points_list.append(center.tolist())
            points_list.extend(tips)
            break

def draw_lines(img, points_list):
    h, w = img.shape[:2]
    l_list = []
    for _ in range(np.random.randint(2, 6)):
        p1, p2 = np.random.randint(-50, [w+50, h+50], size=(2, 2))
        if _is_visible(np.array([p1, p2]), w, h):
            cv2.line(img, tuple(p1), tuple(p2), np.random.randint(100, 255), 1)
            points_list.extend([p1.astype(float).tolist(), p2.astype(float).tolist()])
            l_list.append((p1, p2))
    for i in range(len(l_list)):
        for j in range(i+1, len(l_list)):
            inter = get_line_intersection(l_list[i][0], l_list[i][1], l_list[j][0], l_list[j][1])
            if inter is not None: points_list.append(inter.tolist())


def draw_polygon(img, points_list):
    h, w = img.shape[:2]
    while True:
        num_v = np.random.randint(3, 8)
        anchor = np.random.randint(40, [w-40, h-40])
        pts = anchor + np.random.randint(-60, 60, size=(num_v, 2))
        
        # 1. Angular Sort: Ensures the polygon is simple and non-self-intersecting
        center = np.mean(pts, axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        pts = pts[np.argsort(angles)]
        
        # 2. Min Distance Check (5.0px): Prevents labels from being too crowded
        diffs = np.roll(pts, -1, axis=0) - pts
        dists = np.linalg.norm(diffs, axis=1)
        if np.any(dists < 5.0):
            continue 
        
        if _is_visible(pts, w, h):
            # DRAWING: Always draw the full polygon with all original vertices
            color = np.random.randint(60, 220)
            cv2.fillPoly(img, [pts.astype(np.int32)], color)
            
            # LABELING: Only return coordinates for "sharp" corners
            valid_corners = []
            for i in range(len(pts)):
                p_prev = pts[i - 1]
                p_curr = pts[i]
                p_next = pts[(i + 1) % len(pts)]
                
                # Compute normalized vectors for the two meeting edges
                v1 = p_prev - p_curr
                v2 = p_next - p_curr
                v1_u = v1 / np.linalg.norm(v1)
                v2_u = v2 / np.linalg.norm(v2)
                
                # Angle calculation in degrees
                angle = np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
                
                # Reject labels near 180 degrees (flat edges)
                if not (177.0 <= angle <= 183.0):
                    valid_corners.append(p_curr.tolist())
            
            # Export the filtered coordinates to the global list
            points_list.extend(valid_corners)
            break

def is_convex(pts):
    """Checks if the 4 points form a convex quad to prevent 'twisting'."""
    # pts: (4, 2)
    # Uses cross product to check if all turns are in the same direction
    def cross_product(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    cp1 = cross_product(pts[0], pts[1], pts[2])
    cp2 = cross_product(pts[1], pts[2], pts[3])
    cp3 = cross_product(pts[2], pts[3], pts[0])
    cp4 = cross_product(pts[3], pts[0], pts[1])
    
    # All must have the same sign for convexity
    return (cp1 > 0 and cp2 > 0 and cp3 > 0 and cp4 > 0) or \
           (cp1 < 0 and cp2 < 0 and cp3 < 0 and cp4 < 0)

def draw_checkerboard(img, points_list):
    h_img, w_img = img.shape[:2]
    
    # 1. Define Non-Uniform Grid
    rows, cols = np.random.randint(3, 6), np.random.randint(3, 6)
    col_widths = np.random.randint(20, 60, size=cols)
    row_heights = np.random.randint(20, 60, size=rows)
    grid_w, grid_h = np.sum(col_widths), np.sum(row_heights)
    
    checker_canvas = np.zeros((grid_h, grid_w), dtype=np.uint8)
    x_coords = np.concatenate(([0], np.cumsum(col_widths)))
    y_coords = np.concatenate(([0], np.cumsum(row_heights)))
    
    canonical_pts = []
    for i in range(rows + 1):
        for j in range(cols + 1):
            canonical_pts.append([float(x_coords[j]), float(y_coords[i])])
            if i < rows and j < cols and (i + j) % 2 == 0:
                cv2.rectangle(checker_canvas, (x_coords[j], y_coords[i]), 
                              (x_coords[j+1], y_coords[i+1]), np.random.randint(60, 220), -1)

    # 2. Convex-Only Homography
    src_pts = np.array([[0, 0], [grid_w, 0], [grid_w, grid_h], [0, grid_h]], dtype=np.float32)
    
    for _ in range(10): # Try to get a non-twisted quad
        scale = np.random.uniform(0.15, 0.8)
        out_w, out_h = w_img * scale, h_img * scale
        cx, cy = np.random.randint(0, w_img), np.random.randint(0, h_img)
        dst_pts = np.array([[cx-out_w/2, cy-out_h/2], [cx+out_w/2, cy-out_h/2],
                            [cx+out_w/2, cy+out_h/2], [cx-out_w/2, cy+out_h/2]], dtype=np.float32)
        dst_pts += np.random.uniform(-out_w*0.4, out_w*0.4, size=dst_pts.shape)
        
        if is_convex(dst_pts): # is_convex is the helper function from earlier
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(checker_canvas, M, (w_img, h_img), borderValue=0)
            img[:] = np.maximum(img, warped)
            
            # Project points and pass them to points_list raw
            proj_pts = cv2.perspectiveTransform(np.array([canonical_pts], dtype=np.float32), M)[0]
            points_list.extend(proj_pts.tolist())
            return
 



def draw_ellipse(img, points_list):
    h, w = img.shape[:2]
    
    while True:
        # 1. Randomization of dimensions
        # 'a' will be major semi-axis, 'b' will be minor semi-axis
        center = np.random.randint(50, [w - 50, h - 50])
        axes = np.random.randint(15, 80, size=2)
        major_idx = np.argmax(axes)
        minor_idx = 1 - major_idx
        a, b = axes[major_idx], axes[minor_idx]
        
        # 2. Geometric Calculation for Eccentricity
        # e = sqrt(1 - (b^2 / a^2))
        eccentricity = np.sqrt(1 - (b**2 / a**2))
        
        # 3. Only proceed if e > 0.9 (Ensures high curvature at major tips)
        if eccentricity > 0.9:
            angle = np.random.uniform(0, 360) 
            angle_rad = np.deg2rad(angle)
            
            # Canonical points for ONLY the Major Axis Vertices (+-a, 0)
            # These are the sharpest points on the ellipse
            canonical_pts = np.array([
                [a, 0], [-a, 0]
            ], dtype=np.float32)
            
            # 4. Transform to Image Space (Rotation + Translation)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            transformed_pts = (canonical_pts @ R.T) + center
            
            # 5. Visibility and Occlusion Safety
            if _is_visible(transformed_pts, w, h):
                color = np.random.randint(60, 220)
                # OpenCV uses half-axis lengths (a, b) for the axes parameter
                cv2.ellipse(img, tuple(center.astype(int)), (a, b), angle, 0, 360, color, -1)
                
                # Export ONLY the 2 major axis vertices
                points_list.extend(transformed_pts.tolist())
                break
        else:
            # Re-sample to ensure we get a sharp ellipse for the dataset
            continue