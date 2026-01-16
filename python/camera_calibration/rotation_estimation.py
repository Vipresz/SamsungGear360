"""
Rotation estimation functions for dual-fisheye camera calibration.
Includes ring extraction, feature matching, and rotation calculation.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional


def extract_ring_region(fisheye_img: np.ndarray, center_x: float, center_y: float, 
                        inner_ratio: float = 0.7, outer_ratio: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the ring region at the edge of a fisheye image where overlap occurs.
    
    Args:
        fisheye_img: Input fisheye image
        center_x, center_y: Lens center (0-1 normalized)
        inner_ratio: Inner radius as fraction of image (0.7 = 70%)
        outer_ratio: Outer radius as fraction of image (0.95 = 95%)
    
    Returns:
        (ring_image, ring_mask): Ring region with mask applied
    """
    h, w = fisheye_img.shape[:2]
    cx = int(center_x * w)
    cy = int(center_y * h)
    radius = min(w, h) // 2
    
    inner_r = int(radius * inner_ratio)
    outer_r = int(radius * outer_ratio)
    
    # Create ring mask
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    ring_mask = (dist_from_center >= inner_r) & (dist_from_center <= outer_r)
    
    # Apply mask
    result = fisheye_img.copy()
    result[~ring_mask] = 0
    
    return result, ring_mask


def estimate_lens_rotation_from_rings(left_fisheye: np.ndarray, right_fisheye: np.ndarray,
                                      center1: Tuple[float, float], center2: Tuple[float, float],
                                      inner_ratio: float = 0.65, outer_ratio: float = 0.95) -> Tuple[float, float, float]:
    """Estimate relative rotation between lenses using feature matching on the ring edges.
    
    The overlap between dual fisheye lenses occurs at the edges (ring region) of each image.
    This function extracts features from these ring regions and matches them.
    
    Args:
        left_fisheye: Left fisheye image
        right_fisheye: Right fisheye image (will be flipped for matching)
        center1, center2: Lens centers (x, y) normalized 0-1
        inner_ratio: Inner radius of ring (fraction of radius)
        outer_ratio: Outer radius of ring (fraction of radius)
    
    Returns:
        (yaw, pitch, roll) rotation estimates
    """
    # Extract ring regions
    left_ring, left_mask = extract_ring_region(left_fisheye, center1[0], center1[1], 
                                                inner_ratio, outer_ratio)
    
    # Flip right fisheye horizontally (lenses face opposite directions)
    right_flipped = np.fliplr(right_fisheye)
    right_ring, right_mask = extract_ring_region(right_flipped, 1.0 - center2[0], center2[1],
                                                  inner_ratio, outer_ratio)
    
    # Convert to grayscale
    if len(left_ring.shape) == 3:
        left_gray = cv2.cvtColor(left_ring, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_ring
    
    if len(right_ring.shape) == 3:
        right_gray = cv2.cvtColor(right_ring, cv2.COLOR_BGR2GRAY)
    else:
        right_gray = right_ring
    
    # Feature detection with SIFT
    sift = cv2.SIFT_create(nfeatures=1000)
    kp1, des1 = sift.detectAndCompute(left_gray, None)
    kp2, des2 = sift.detectAndCompute(right_gray, None)
    
    if des1 is None or des2 is None:
        print(f"    No features found in ring regions")
        return (0.0, 0.0, 0.0)
    
    print(f"    Ring features: left={len(kp1)}, right={len(kp2)}")
    
    if len(des1) < 4 or len(des2) < 4:
        return (0.0, 0.0, 0.0)
    
    # Match features
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good_matches = []
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    print(f"    Good matches: {len(good_matches)}")
    
    if len(good_matches) < 6:
        return (0.0, 0.0, 0.0)
    
    # Get matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Estimate transform using RANSAC
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    
    if H is None:
        return (0.0, 0.0, 0.0)
    
    # Extract rotation from homography
    inliers = np.sum(mask) if mask is not None else 0
    print(f"    Homography inliers: {inliers}")
    
    h_img = left_gray.shape[0]
    w_img = left_gray.shape[1]
    
    # Translation (in pixels) -> normalized
    tx = H[0, 2] / w_img
    ty = H[1, 2] / h_img
    
    # Rotation angle from homography
    raw_roll = np.arctan2(H[1, 0], H[0, 0])
    
    # Normalize roll to small angle
    if abs(raw_roll) > np.pi / 2:
        if raw_roll > 0:
            roll = raw_roll - np.pi
        else:
            roll = raw_roll + np.pi
    else:
        roll = raw_roll
    
    # Scale translation to angular values
    yaw = float(tx * 0.1)
    pitch = float(ty * 0.1)
    roll = float(roll * 0.5)
    
    print(f"    Raw roll: {raw_roll:.4f} rad, corrected: {roll:.4f} rad")
    
    return (yaw, pitch, roll)


def estimate_lens_rotation(left_equirect: np.ndarray, right_equirect: np.ndarray,
                           overlap_band: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
    """Estimate relative rotation between lenses using feature matching (legacy equirect method)."""
    x, y, w, h = overlap_band
    left_band = left_equirect[y:y+h, x:x+w]
    right_band = right_equirect[y:y+h, x:x+w]
    
    left_gray = cv2.cvtColor(left_band, cv2.COLOR_BGR2GRAY) if len(left_band.shape) == 3 else left_band
    right_gray = cv2.cvtColor(right_band, cv2.COLOR_BGR2GRAY) if len(right_band.shape) == 3 else right_band
    
    sift = cv2.SIFT_create(nfeatures=500)
    kp1, des1 = sift.detectAndCompute(left_gray, None)
    kp2, des2 = sift.detectAndCompute(right_gray, None)
    
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return (0.0, 0.0, 0.0)
    
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 4:
        return (0.0, 0.0, 0.0)
    
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
    
    if H is None:
        return (0.0, 0.0, 0.0)
    
    dx = H[0, 2]
    dy = H[1, 2]
    rotation = np.arctan2(H[1, 0], H[0, 0])
    
    yaw = dx / w * 0.1
    pitch = dy / h * 0.1
    roll = rotation * 0.1
    
    return (float(yaw), float(pitch), float(roll))


def compute_alignment_offsets_equirect(frame: np.ndarray, params: Dict,
                                       project_func,
                                       temp_width: int = 640, temp_height: int = 320) -> Optional[Tuple[float, float, float, float, float, float, float]]:
    """Compute alignment offsets and rotation in equirectangular space using phase correlation."""
    temp_params = params.copy()
    
    left_params = temp_params.copy()
    left_params['lens1CenterX'] = temp_params.get('lens1CenterX', 0.5)
    left_params['lens1CenterY'] = temp_params.get('lens1CenterY', 0.5)
    left_params['lens2CenterX'] = 0.5
    left_params['lens2CenterY'] = 0.5
    left_params['alignmentOffset1X'] = 0.0
    left_params['alignmentOffset1Y'] = 0.0
    left_params['alignmentOffset2X'] = 0.0
    left_params['alignmentOffset2Y'] = 0.0
    
    left_equirect = project_func(frame, left_params, temp_width, temp_height, 180.0, use_both_lenses=False)
    
    right_params = temp_params.copy()
    right_params['lens1CenterX'] = 0.5
    right_params['lens1CenterY'] = 0.5
    right_params['lens2CenterX'] = temp_params.get('lens2CenterX', 0.5)
    right_params['lens2CenterY'] = temp_params.get('lens2CenterY', 0.5)
    right_params['alignmentOffset1X'] = 0.0
    right_params['alignmentOffset1Y'] = 0.0
    right_params['alignmentOffset2X'] = 0.0
    right_params['alignmentOffset2Y'] = 0.0
    
    frame_swapped = frame.copy()
    if temp_params['isHorizontal']:
        h, w = frame.shape[:2]
        left_half = frame[:, :w//2].copy()
        right_half = frame[:, w//2:].copy()
        frame_swapped[:, :w//2] = right_half
        frame_swapped[:, w//2:] = left_half
    else:
        h, w = frame.shape[:2]
        top_half = frame[:h//2, :].copy()
        bottom_half = frame[h//2:, :].copy()
        frame_swapped[:h//2, :] = bottom_half
        frame_swapped[h//2:, :] = top_half
    
    temp_params_right = temp_params.copy()
    temp_params_right['lens1CenterX'] = temp_params.get('lens2CenterX', 0.5)
    temp_params_right['lens1CenterY'] = temp_params.get('lens2CenterY', 0.5)
    temp_params_right['lens2CenterX'] = 0.5
    temp_params_right['lens2CenterY'] = 0.5
    right_equirect = project_func(frame_swapped, temp_params_right, temp_width, temp_height, 180.0, use_both_lenses=False)
    
    left_gray = cv2.cvtColor(left_equirect, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_equirect, cv2.COLOR_BGR2GRAY)
    
    seam1_x = temp_width // 4
    seam2_x = 3 * temp_width // 4
    band_width = temp_width // 8
    
    band1_left = left_gray[:, seam1_x - band_width:seam1_x + band_width]
    band1_right = right_gray[:, seam1_x - band_width:seam1_x + band_width]
    
    band2_left = left_gray[:, seam2_x - band_width:seam2_x + band_width]
    band2_right = right_gray[:, seam2_x - band_width:seam2_x + band_width]
    
    offsets = []
    
    if band1_left.shape[1] > 0 and band1_right.shape[1] > 0:
        try:
            result = cv2.phaseCorrelate(band1_left.astype(np.float32), band1_right.astype(np.float32))
            dx, dy = result[0]
            
            if params['isHorizontal']:
                offset2X = dx / (temp_width * 0.5)
                offset2Y = dy / temp_height
                offsets.append((offset2X, offset2Y))
        except:
            pass
    
    if band2_left.shape[1] > 0 and band2_right.shape[1] > 0:
        try:
            result = cv2.phaseCorrelate(band2_left.astype(np.float32), band2_right.astype(np.float32))
            dx, dy = result[0]
            
            if params['isHorizontal']:
                offset2X = dx / (temp_width * 0.5)
                offset2Y = dy / temp_height
                offsets.append((offset2X, offset2Y))
        except:
            pass
    
    if offsets:
        avg_offset2X = np.mean([o[0] for o in offsets])
        avg_offset2Y = np.mean([o[1] for o in offsets])
        
        avg_offset2X = np.clip(avg_offset2X, -0.1, 0.1)
        avg_offset2Y = np.clip(avg_offset2Y, -0.1, 0.1)
        
        return (0.0, 0.0, avg_offset2X, avg_offset2Y, 0.0, 0.0, 0.0)
    
    return None
