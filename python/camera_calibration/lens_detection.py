"""
Lens detection functions for dual-fisheye camera calibration.
Includes boundary detection, circle fitting, and center estimation.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict


def detect_lens_boundary_points(gray_img: np.ndarray, initial_center: Tuple[int, int], 
                                 num_rays: int = 360) -> np.ndarray:
    """Detect boundary points of fisheye lens using radial ray casting.
    
    Looks for the transition from image content to dark/black border.
    The boundary is where intensity drops to near-black (< 15).
    """
    h, w = gray_img.shape
    cx, cy = initial_center
    
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 1.0)
    
    boundary_points = []
    
    for i in range(num_rays):
        angle = 2 * np.pi * i / num_rays
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        max_r = int(min(w, h) * 0.5)
        
        boundary_r = None
        for r in range(max_r - 1, int(max_r * 0.3), -1):
            px = int(cx + r * cos_a)
            py = int(cy + r * sin_a)
            
            if 0 <= px < w and 0 <= py < h:
                intensity = blurred[py, px]
                
                if intensity > 15:
                    boundary_r = r
                    break
        
        if boundary_r is not None:
            px = cx + boundary_r * cos_a
            py = cy + boundary_r * sin_a
            boundary_points.append([px, py])
    
    return np.array(boundary_points, dtype=np.float32) if boundary_points else np.array([], dtype=np.float32)


def fit_circle_ransac(points: np.ndarray, iterations: int = 100) -> Tuple[float, float, float]:
    """Fit circle to points using RANSAC for robustness."""
    best_cx, best_cy, best_r = 0, 0, 0
    best_inliers = 0
    
    n_points = len(points)
    if n_points < 3:
        return (np.mean(points[:, 0]), np.mean(points[:, 1]), 100.0)
    
    for _ in range(iterations):
        idx = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[idx]
        
        ax, ay = p1
        bx, by = p2
        cx, cy = p3
        
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            continue
        
        ux = ((ax*ax + ay*ay) * (by - cy) + (bx*bx + by*by) * (cy - ay) + (cx*cx + cy*cy) * (ay - by)) / d
        uy = ((ax*ax + ay*ay) * (cx - bx) + (bx*bx + by*by) * (ax - cx) + (cx*cx + cy*cy) * (bx - ax)) / d
        
        r = np.sqrt((ax - ux)**2 + (ay - uy)**2)
        
        distances = np.sqrt((points[:, 0] - ux)**2 + (points[:, 1] - uy)**2)
        inliers = np.sum(np.abs(distances - r) < r * 0.05)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_cx, best_cy, best_r = ux, uy, r
    
    return (best_cx, best_cy, best_r)


def detect_lens_center_advanced(frame: np.ndarray, lens_region: Tuple[int, int, int, int]) -> Optional[Dict]:
    """Advanced lens center detection with multiple methods and refinement."""
    x, y, w, h = lens_region
    lens_img = frame[y:y+h, x:x+w].copy()
    gray = cv2.cvtColor(lens_img, cv2.COLOR_BGR2GRAY)
    
    initial_cx, initial_cy = w // 2, h // 2
    
    boundary_points = detect_lens_boundary_points(gray, (initial_cx, initial_cy), num_rays=360)
    
    if len(boundary_points) < 10:
        return None
    
    cx, cy, radius = fit_circle_ransac(boundary_points)
    
    distances = np.sqrt((boundary_points[:, 0] - cx)**2 + (boundary_points[:, 1] - cy)**2)
    inlier_mask = np.abs(distances - radius) < radius * 0.1
    inlier_points = boundary_points[inlier_mask]
    
    if len(inlier_points) >= 10:
        cx = np.mean(inlier_points[:, 0])
        cy = np.mean(inlier_points[:, 1])
        radius = np.median(np.sqrt((inlier_points[:, 0] - cx)**2 + (inlier_points[:, 1] - cy)**2))
    
    radius_values = np.sqrt((inlier_points[:, 0] - cx)**2 + (inlier_points[:, 1] - cy)**2) if len(inlier_points) > 0 else distances
    radius_std = np.std(radius_values)
    circularity = 1.0 - (radius_std / radius) if radius > 0 else 0
    
    return {
        'center_x': cx / w,
        'center_y': cy / h,
        'radius': float(radius),
        'circularity': float(circularity),
        'boundary_points': boundary_points,
        'inlier_ratio': len(inlier_points) / len(boundary_points) if len(boundary_points) > 0 else 0
    }


def detect_lens_center(frame: np.ndarray, lens_region: Tuple[int, int, int, int],
                       method: str = 'advanced') -> Optional[Tuple[float, float, float]]:
    """Detect lens center and radius from a fisheye image region."""
    result = detect_lens_center_advanced(frame, lens_region)
    if result is not None:
        return (result['center_x'], result['center_y'], result['radius'])
    return None
