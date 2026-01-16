"""
FOV and distortion optimization functions for dual-fisheye camera calibration.
"""

import cv2
import numpy as np
from typing import Tuple, Dict


def estimate_fov_from_coverage(frame: np.ndarray, lens_region: Tuple[int, int, int, int],
                                center: Tuple[float, float], radius: float) -> float:
    """Estimate lens FOV based on coverage pattern.
    
    The FOV scaling factor relates to degrees:
    - fov=1.0 means the fisheye circle maps to exactly 180° (hemisphere)
    - fov>1.0 means the lens sees more than 180° (e.g., fov=1.055 for 190°)
    - fov<1.0 means the lens sees less than 180°
    
    Returns the fov scaling factor (not degrees).
    """
    x, y, w, h = lens_region
    lens_img = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(lens_img, cv2.COLOR_BGR2GRAY)
    
    cx = center[0] * w
    cy = center[1] * h
    
    num_rays = 72
    coverage_radii = []
    
    for i in range(num_rays):
        angle = 2 * np.pi * i / num_rays
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        for r in range(int(radius), int(radius * 0.3), -1):
            px = int(cx + r * cos_a)
            py = int(cy + r * sin_a)
            
            if 0 <= px < w and 0 <= py < h:
                if gray[py, px] > 15:
                    coverage_radii.append(r)
                    break
                
    if coverage_radii:
        avg_coverage = np.mean(coverage_radii)
        coverage_ratio = avg_coverage / radius
        fov_scale = coverage_ratio
        return float(np.clip(fov_scale, 0.9, 1.2))
    
    return 1.0


def optimize_fov(frame: np.ndarray, params: Dict, 
                 project_dual_func,
                 output_size: Tuple[int, int] = (640, 320)) -> Tuple[float, float]:
    """Optimize FOV scaling to minimize black pixels and maximize coverage.
    
    Starts at 1.0 (180°) and increases to find optimal coverage.
    
    Args:
        frame: Input dual-fisheye frame
        params: Calibration parameters
        project_dual_func: Function to project dual fisheye to equirect
        output_size: Output resolution
    
    Returns:
        (fov1, fov2): Optimal FOV scaling for each lens
    """
    output_width, output_height = output_size
    height, width = frame.shape[:2]
    is_horizontal = width > height
    
    center1_x = params.get('lens1CenterX', 0.5)
    center1_y = params.get('lens1CenterY', 0.5)
    center2_x = params.get('lens2CenterX', 0.5)
    center2_y = params.get('lens2CenterY', 0.5)
    
    def evaluate_fov(fov_pair):
        fov1, fov2 = fov_pair
        try:
            if is_horizontal:
                equirect = project_dual_func(
                    frame, fov=(fov1, fov2), offset=np.pi/2,
                    center1_x=center1_x, center1_y=center1_y,
                    center2_x=center2_x, center2_y=center2_y
                )
            else:
                frame_rotated = np.rot90(frame)
                equirect = project_dual_func(
                    frame_rotated, fov=(fov1, fov2), offset=np.pi/2,
                    center1_x=center1_x, center1_y=center1_y,
                    center2_x=center2_x, center2_y=center2_y
                )
            
            if equirect.shape[0] != output_height or equirect.shape[1] != output_width:
                equirect = cv2.resize(equirect, (output_width, output_height))
            
            if len(equirect.shape) == 3:
                gray = cv2.cvtColor(equirect, cv2.COLOR_BGR2GRAY)
            else:
                gray = equirect
            
            black_pixels = np.sum(gray < 5)
            total_pixels = gray.size
            black_ratio = black_pixels / total_pixels
            
            edge_penalty = 0
            top_row = gray[0, :]
            bottom_row = gray[-1, :]
            edge_penalty += np.sum(top_row < 10) / len(top_row)
            edge_penalty += np.sum(bottom_row < 10) / len(bottom_row)
            
            fov1, fov2 = fov_pair
            avg_fov = (fov1 + fov2) / 2
            
            if avg_fov > 1.05:
                regularization = (avg_fov - 1.05) ** 2 * 10.0
            elif avg_fov < 0.98:
                regularization = (0.98 - avg_fov) ** 2 * 2.0
            else:
                regularization = 0
            
            return black_ratio + edge_penalty * 0.5 + regularization
        except:
            return 1.0
    
    # Coarse search
    coarse_fovs = np.arange(0.96, 1.13, 0.02)
    best_fov = 1.0
    best_score = evaluate_fov((1.0, 1.0))
    
    for fov in coarse_fovs:
        score = evaluate_fov((fov, fov))
        if score < best_score:
            best_score = score
            best_fov = fov
    
    # Medium search
    medium_fovs = np.arange(best_fov - 0.02, best_fov + 0.025, 0.005)
    for fov in medium_fovs:
        score = evaluate_fov((fov, fov))
        if score < best_score:
            best_score = score
            best_fov = fov
    
    # Fine search
    fine_fovs = np.arange(best_fov - 0.005, best_fov + 0.006, 0.001)
    for fov in fine_fovs:
        score = evaluate_fov((fov, fov))
        if score < best_score:
            best_score = score
            best_fov = fov
    
    # Per-lens refinement
    best_fov_pair = (best_fov, best_fov)
    for fov1_delta in [-0.002, 0, 0.002]:
        for fov2_delta in [-0.002, 0, 0.002]:
            fov1 = best_fov + fov1_delta
            fov2 = best_fov + fov2_delta
            score = evaluate_fov((fov1, fov2))
            if score < best_score:
                best_score = score
                best_fov_pair = (fov1, fov2)
    
    final_fov = (
        float(np.clip(best_fov_pair[0], 0.95, 1.15)),
        float(np.clip(best_fov_pair[1], 0.95, 1.15))
    )
    
    return final_fov


def optimize_distortion(frame: np.ndarray, params: Dict, 
                        project_dual_func,
                        output_size: Tuple[int, int] = (640, 320)) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """Optimize polynomial distortion coefficients to improve projection quality.
    
    Uses the f-theta model with polynomial correction:
    r = f * θ * (1 + p1*θ + p2*θ² + p3*θ³ + p4*θ⁴)
    
    Returns:
        (distortion1, distortion2): Tuple of (p1, p2, p3, p4) for each lens
    """
    output_width, output_height = output_size
    height, width = frame.shape[:2]
    is_horizontal = width > height
    
    center1_x = params.get('lens1CenterX', 0.5)
    center1_y = params.get('lens1CenterY', 0.5)
    center2_x = params.get('lens2CenterX', 0.5)
    center2_y = params.get('lens2CenterY', 0.5)
    fov1 = params.get('lens1FOV', 1.0)
    fov2 = params.get('lens2FOV', 1.0)
    
    def evaluate_distortion(dist_params):
        p1, p2 = dist_params[0], dist_params[1]
        p3, p4 = 0.0, 0.0
        
        try:
            if is_horizontal:
                equirect = project_dual_func(
                    frame, fov=(fov1, fov2), offset=np.pi/2,
                    center1_x=center1_x, center1_y=center1_y,
                    center2_x=center2_x, center2_y=center2_y,
                    distortion1=(p1, p2, p3, p4),
                    distortion2=(p1, p2, p3, p4)
                )
            else:
                frame_rotated = np.rot90(frame)
                equirect = project_dual_func(
                    frame_rotated, fov=(fov1, fov2), offset=np.pi/2,
                    center1_x=center1_x, center1_y=center1_y,
                    center2_x=center2_x, center2_y=center2_y,
                    distortion1=(p1, p2, p3, p4),
                    distortion2=(p1, p2, p3, p4)
                )
            
            if equirect.shape[0] != output_height or equirect.shape[1] != output_width:
                equirect = cv2.resize(equirect, (output_width, output_height))
            
            if len(equirect.shape) == 3:
                gray = cv2.cvtColor(equirect, cv2.COLOR_BGR2GRAY)
            else:
                gray = equirect
            
            seam_x = output_width // 2
            band_width = output_width // 8
            seam_region = gray[:, seam_x - band_width:seam_x + band_width]
            
            laplacian = cv2.Laplacian(seam_region, cv2.CV_64F)
            sharpness = laplacian.var()
            
            black_pixels = np.sum(gray < 5)
            black_ratio = black_pixels / gray.size
            
            return -sharpness * 0.0001 + black_ratio + abs(p1) * 5.0 + abs(p2) * 5.0
        except:
            return 1.0
    
    best_dist = (0.0, 0.0)
    best_score = evaluate_distortion(best_dist)
    
    # Coarse search
    for p1 in np.arange(-0.1, 0.11, 0.02):
        for p2 in np.arange(-0.05, 0.06, 0.01):
            score = evaluate_distortion((p1, p2))
            if score < best_score:
                best_score = score
                best_dist = (p1, p2)
    
    # Fine search
    p1_best, p2_best = best_dist
    for p1 in np.arange(p1_best - 0.02, p1_best + 0.025, 0.005):
        for p2 in np.arange(p2_best - 0.01, p2_best + 0.015, 0.002):
            score = evaluate_distortion((p1, p2))
            if score < best_score:
                best_score = score
                best_dist = (p1, p2)
    
    p1, p2 = best_dist
    distortion = (float(p1), float(p2), 0.0, 0.0)
    
    return distortion, distortion
