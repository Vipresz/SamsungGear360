"""
Alignment functions for dual-fisheye camera calibration.
Includes feature extraction, seam error computation, and alignment optimization.
"""

import cv2
import numpy as np
from typing import Tuple, Dict

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def extract_overlap_features(equirect_img: np.ndarray, seam_x: int, band_width: int) -> Tuple[list, np.ndarray]:
    """Extract features from overlap region around seam."""
    h, w = equirect_img.shape[:2]
    x1 = max(0, seam_x - band_width)
    x2 = min(w, seam_x + band_width)
    
    band = equirect_img[:, x1:x2]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY) if len(band.shape) == 3 else band
    
    sift = cv2.SIFT_create(nfeatures=500)
    kp, des = sift.detectAndCompute(gray, None)
    
    if des is None:
        orb = cv2.ORB_create(nfeatures=500)
        kp, des = orb.detectAndCompute(gray, None)
    
    return kp, des, x1


def compute_seam_alignment_error(left_img: np.ndarray, right_img: np.ndarray, 
                                  seam_x: int, band_width: int = 50) -> float:
    """Compute alignment error at seam using intensity difference."""
    h, w = left_img.shape[:2]
    x1 = max(0, seam_x - band_width)
    x2 = min(w, seam_x + band_width)
    
    left_band = left_img[:, x1:x2]
    right_band = right_img[:, x1:x2]
    
    if len(left_band.shape) == 3:
        left_gray = cv2.cvtColor(left_band, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_band, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_band
        right_gray = right_band
    
    diff = np.abs(left_gray.astype(np.float32) - right_gray.astype(np.float32))
    
    center_weight = np.exp(-0.5 * ((np.arange(x2 - x1) - band_width) / (band_width * 0.5))**2)
    weighted_diff = diff * center_weight[np.newaxis, :]
    
    return float(np.mean(weighted_diff))


def optimize_alignment_parameters(frame: np.ndarray, initial_params: Dict,
                                   output_size: Tuple[int, int] = (640, 320),
                                   project_func=None) -> Dict:
    """Optimize alignment parameters to minimize seam error.
    
    Args:
        frame: Input dual-fisheye frame
        initial_params: Initial calibration parameters
        output_size: Output resolution (width, height)
        project_func: Projection function (project_fisheye_to_equirectangular)
    """
    if project_func is None:
        # Import here to avoid circular dependency
        from projection import project_fisheye_to_equirectangular
        project_func = project_fisheye_to_equirectangular
    
    output_width, output_height = output_size
    
    def objective(x):
        params = initial_params.copy()
        params['alignmentOffset1X'] = x[0]
        params['alignmentOffset1Y'] = x[1]
        params['alignmentOffset2X'] = x[2]
        params['alignmentOffset2Y'] = x[3]
        
        try:
            equirect = project_func(
                frame, params, output_width, output_height, 
                apply_calibration=True
            )
            
            seam_x = output_width // 2
            band_width = output_width // 16
            
            left_half = equirect[:, :seam_x + band_width]
            right_half = equirect[:, seam_x - band_width:]
            
            error = compute_seam_alignment_error(left_half, right_half, seam_x, band_width)
            return error
        except:
            return 1e6
    
    x0 = [
        initial_params.get('alignmentOffset1X', 0.0),
        initial_params.get('alignmentOffset1Y', 0.0),
        initial_params.get('alignmentOffset2X', 0.0),
        initial_params.get('alignmentOffset2Y', 0.0)
    ]
    
    if HAS_SCIPY:
        result = minimize(objective, x0, method='Powell',
                         options={'maxiter': 50, 'ftol': 1e-4})
        best_x = result.x
    else:
        best_x = x0
        best_error = objective(x0)
        
        for _ in range(20):
            for i in range(4):
                for delta in [-0.01, 0.01, -0.005, 0.005]:
                    test_x = list(best_x)
                    test_x[i] += delta
                    error = objective(test_x)
                    if error < best_error:
                        best_error = error
                        best_x = test_x
    
    optimized_params = initial_params.copy()
    optimized_params['alignmentOffset1X'] = float(np.clip(best_x[0], -0.05, 0.05))
    optimized_params['alignmentOffset1Y'] = float(np.clip(best_x[1], -0.05, 0.05))
    optimized_params['alignmentOffset2X'] = float(np.clip(best_x[2], -0.05, 0.05))
    optimized_params['alignmentOffset2Y'] = float(np.clip(best_x[3], -0.05, 0.05))
    
    return optimized_params
