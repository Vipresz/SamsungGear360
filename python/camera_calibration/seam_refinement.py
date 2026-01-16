"""
Seam-based refinement functions for dual-fisheye camera calibration.
Refines FOV, Y-offset, and roll by comparing seam regions.
"""

import cv2
import numpy as np
from typing import Tuple

LENS_FOV_DEG = 195


def compute_unified_seam_score(equirect: np.ndarray) -> float:
    """Compute seam score using consistent rectangular overlap regions.
    
    Evaluates both center seam and edge seams (left/right edges that wrap around).
    
    Region parameters:
    - Height: 50% of image height (centered vertically)
    - Width: min(5% of image width, actual overlap width)
    - Comparison uses 80% of overlap width from each side
    
    Returns:
        float: Combined seam error score (lower is better)
    """
    h, w = equirect.shape[:2]
    
    # Vertical bounds: middle 50% of image
    y_start = h // 4
    y_end = 3 * h // 4
    region_height = y_end - y_start
    
    # Maximum overlap width: 5% of image width
    max_overlap_width = max(2, int(w * 0.05))
    
    # Use 80% of overlap for comparison (to avoid boundary artifacts)
    compare_fraction = 0.8
    
    total_score = 0.0
    num_seams = 0
    
    # === CENTER SEAM (between left and right lens at x = w/2) ===
    center_x = w // 2
    half_overlap = max_overlap_width // 2
    
    # Left side of center seam (from left lens)
    left_x_start = max(0, center_x - half_overlap)
    left_x_end = center_x
    # Right side of center seam (from right lens)
    right_x_start = center_x
    right_x_end = min(w, center_x + half_overlap)
    
    # Calculate actual overlap width at center
    left_width = left_x_end - left_x_start
    right_width = right_x_end - right_x_start
    
    # Use 80% of the smaller overlap
    compare_width = max(1, int(min(left_width, right_width) * compare_fraction))
    
    # Extract comparison strips (innermost pixels near the seam)
    left_strip = equirect[y_start:y_end, center_x - compare_width:center_x, :]
    right_strip = equirect[y_start:y_end, center_x:center_x + compare_width, :]
    
    if left_strip.size > 0 and right_strip.size > 0:
        left_gray = cv2.cvtColor(left_strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
        right_gray = cv2.cvtColor(right_strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Compare the edge columns (rightmost of left vs leftmost of right)
        left_edge = left_gray[:, -1]
        right_edge = right_gray[:, 0]
        
        # Mask out black/invalid pixels
        valid = (left_edge > 5) & (right_edge > 5)
        
        if np.sum(valid) > region_height * 0.3:  # At least 30% valid
            pixel_diff = np.abs(left_edge[valid] - right_edge[valid])
            center_mae = np.mean(pixel_diff)
            total_score += center_mae
            num_seams += 1
    
    # === EDGE SEAMS (left edge at x=0 and right edge at x=w-1 wrap around) ===
    # In equirectangular, left edge connects to right edge
    
    # Left edge strip (first few columns - from right lens wrapped)
    edge_left = equirect[y_start:y_end, :half_overlap, :]
    # Right edge strip (last few columns - from left lens wrapped)
    edge_right = equirect[y_start:y_end, -half_overlap:, :]
    
    if edge_left.size > 0 and edge_right.size > 0:
        # Calculate actual widths
        left_edge_width = edge_left.shape[1]
        right_edge_width = edge_right.shape[1]
        
        compare_width_edge = max(1, int(min(left_edge_width, right_edge_width) * compare_fraction))
        
        # Compare: rightmost columns of right edge strip vs leftmost of left edge strip
        # (They should match when wrapped around)
        left_edge_strip = edge_left[:, :compare_width_edge, :]
        right_edge_strip = edge_right[:, -compare_width_edge:, :]
        
        left_edge_gray = cv2.cvtColor(left_edge_strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
        right_edge_gray = cv2.cvtColor(right_edge_strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Compare leftmost column of left edge with rightmost column of right edge
        left_edge_col = left_edge_gray[:, 0]
        right_edge_col = right_edge_gray[:, -1]
        
        valid_edge = (left_edge_col > 5) & (right_edge_col > 5)
        
        if np.sum(valid_edge) > region_height * 0.3:
            pixel_diff_edge = np.abs(left_edge_col[valid_edge] - right_edge_col[valid_edge])
            edge_mae = np.mean(pixel_diff_edge)
            total_score += edge_mae
            num_seams += 1
    
    if num_seams == 0:
        return float('inf')
    
    return total_score / num_seams


def refine_fov_from_seam(frame: np.ndarray, 
                          center1_x: float, center1_y: float,
                          center2_x: float, center2_y: float,
                          p2_1: float, p2_2: float,
                          is_horizontal: bool,
                          project_func,
                          initial_fov1: float = 1.0,
                          initial_fov2: float = 1.0,
                          yaw: float = 0.0,
                          pitch: float = 0.0,
                          roll: float = 0.0,
                          strip_width: int = 40) -> Tuple[float, float]:
    """Refine FOV by comparing vertical strips at the seam.
    
    Projects the image with different FOV values and compares the vertical strips
    where the two lens projections meet. The optimal FOV minimizes the difference
    between the left edge of the right projection and right edge of the left projection.
    
    Now also applies rotation parameters for consistency with rotation optimization.
    """
    h, w = frame.shape[:2]
    
    if is_horizontal:
        lens_w = w // 2
        lens_h = h
    else:
        lens_w = w
        lens_h = h // 2
    
    test_width = lens_h * 2
    test_height = lens_h
    
    # Determine if we should apply rotation
    has_rotation = (yaw != 0.0 or pitch != 0.0 or roll != 0.0)
    
    def project_with_fov(fov1: float, fov2: float) -> np.ndarray:
        """Project with given FOV values."""
        params = {
            'lens1CenterX': center1_x,
            'lens1CenterY': center1_y,
            'lens2CenterX': center2_x,
            'lens2CenterY': center2_y,
            'lens1FOV': fov1,
            'lens2FOV': fov2,
            'lens1P2': p2_1,
            'lens2P2': p2_2,
            'lens1P1': 0.0, 'lens1P3': 0.0, 'lens1P4': 0.0,
            'lens2P1': 0.0, 'lens2P3': 0.0, 'lens2P4': 0.0,
            'lensRotationYaw': yaw,
            'lensRotationPitch': pitch,
            'lensRotationRoll': roll,
            'applyRotation': has_rotation,
            'isHorizontal': is_horizontal,
        }
        return project_func(
            frame, params, test_width, test_height, 
            apply_calibration=True
        )
    
    def compute_seam_difference(fov1: float, fov2: float) -> float:
        """Compute difference using unified seam scoring."""
        try:
            equirect = project_with_fov(fov1, fov2)
        except:
            return float('inf')
        
        # Use unified seam score (center + edge seams, 50% height, 5% width)
        seam_score = compute_unified_seam_score(equirect)
        
        # Add penalty for black pixels (indicates insufficient coverage)
        all_gray = cv2.cvtColor(equirect, cv2.COLOR_BGR2GRAY)
        black_ratio = np.sum(all_gray < 5) / all_gray.size
        black_penalty = black_ratio * 50
        
        return seam_score + black_penalty
    
    # Start with provided initial FOV
    best_fov1, best_fov2 = initial_fov1, initial_fov2
    best_score = compute_seam_difference(best_fov1, best_fov2)
    
    print(f"    Initial seam score: {best_score:.2f} (FOV={best_fov1:.3f}, {best_fov2:.3f})")
    
    # FOV search range: search around initial value ±0.02
    # For 195° lens (initial=1.0833), search from ~1.06 to ~1.10
    search_start = max(1.0, min(initial_fov1, initial_fov2) - 0.02)
    search_end = max(initial_fov1, initial_fov2) + 0.02
    
    # Step 1: Coarse search in 0.005 steps
    for fov in np.arange(search_start, search_end + 0.001, 0.005):
        score = compute_seam_difference(fov, fov)
        if score < best_score:
            best_score = score
            best_fov1, best_fov2 = fov, fov
            print(f"    FOV {fov:.3f}: score {score:.2f} (better)")
    
    # Step 2: Fine search around best ±0.003
    fine_best = best_fov1
    for fov in np.arange(fine_best - 0.003, fine_best + 0.004, 0.001):
        score = compute_seam_difference(fov, fov)
        if score < best_score:
            best_score = score
            best_fov1, best_fov2 = fov, fov
    
    # Step 3: Per-lens fine tuning
    symmetric_best = (best_fov1 + best_fov2) / 2
    for d1 in np.arange(-0.003, 0.004, 0.001):
        for d2 in np.arange(-0.003, 0.004, 0.001):
            fov1 = max(1.0, symmetric_best + d1)
            fov2 = max(1.0, symmetric_best + d2)
            score = compute_seam_difference(fov1, fov2)
            if score < best_score:
                best_score = score
                best_fov1, best_fov2 = fov1, fov2
    
    print(f"    Final FOV seam score: {best_score:.2f}")
    
    # FOV scale is a multiplier on 180° (equirectangular base)
    # Lens FOV is 195°, so scale of 195/180 ≈ 1.083 = full lens coverage
    fov1_deg = best_fov1 * 180.0
    fov2_deg = best_fov2 * 180.0
    print(f"    Lens 1: FOV scale {best_fov1:.4f} = {fov1_deg:.1f}° effective")
    print(f"    Lens 2: FOV scale {best_fov2:.4f} = {fov2_deg:.1f}° effective")
    
    return float(best_fov1), float(best_fov2)


def optimize_offsets(frame: np.ndarray,
                      center1_x: float, center1_y: float,
                      center2_x: float, center2_y: float,
                      fov1: float, fov2: float,
                      p2_1: float, p2_2: float,
                      roll1: float, roll2: float,
                      is_horizontal: bool,
                      project_func) -> tuple:
    """Optimize X and Y offsets for BOTH lenses to align the seam.
    
    Returns:
        (x_off1, y_off1, x_off2, y_off2) - offsets for each lens
    """
    h, w = frame.shape[:2]
    
    if is_horizontal:
        lens_w = w // 2
        lens_h = h
    else:
        lens_w = w
        lens_h = h // 2
    
    test_width = lens_h * 2
    test_height = lens_h
    
    def project_with_offsets(x_off1: float, y_off1: float, x_off2: float, y_off2: float) -> np.ndarray:
        params = {
            'lens1CenterX': center1_x,
            'lens1CenterY': center1_y,
            'lens2CenterX': center2_x,
            'lens2CenterY': center2_y,
            'lens1FOV': fov1,
            'lens2FOV': fov2,
            'lens1P2': p2_1,
            'lens2P2': p2_2,
            'lens1P1': 0.0, 'lens1P3': 0.0, 'lens1P4': 0.0,
            'lens2P1': 0.0, 'lens2P3': 0.0, 'lens2P4': 0.0,
            # Per-lens rotations
            'lens1RotationYaw': 0.0,
            'lens1RotationPitch': 0.0,
            'lens1RotationRoll': roll1,
            'lens2RotationYaw': 0.0,
            'lens2RotationPitch': 0.0,
            'lens2RotationRoll': roll2,
            # Per-lens offsets (in pixels)
            'alignmentOffset1X': x_off1,
            'alignmentOffset1Y': y_off1,
            'alignmentOffset2X': x_off2,
            'alignmentOffset2Y': y_off2,
            'isHorizontal': is_horizontal,
            'applyAlignment': True,
        }
        return project_func(
            frame, params, test_width, test_height,
            apply_calibration=True
        )
    
    def compute_alignment_score(x_off1: float, y_off1: float, x_off2: float, y_off2: float) -> float:
        try:
            equirect = project_with_offsets(x_off1, y_off1, x_off2, y_off2)
        except:
            return float('inf')
        
        return compute_unified_seam_score(equirect)
    
    # Start with no offsets
    best_x1, best_y1, best_x2, best_y2 = 0.0, 0.0, 0.0, 0.0
    best_score = compute_alignment_score(0.0, 0.0, 0.0, 0.0)
    
    print(f"    Initial alignment score: {best_score:.2f}")
    
    # Normalize offsets to pixel units based on lens dimensions
    max_offset = 0.02  # 2% of lens dimension
    
    # Coarse search: Y offsets first (most common misalignment)
    print("    Optimizing Y-offsets...")
    for y1 in np.arange(-max_offset, max_offset + 0.005, 0.005):
        for y2 in np.arange(-max_offset, max_offset + 0.005, 0.005):
            score = compute_alignment_score(best_x1, y1, best_x2, y2)
            if score < best_score:
                best_score = score
                best_y1, best_y2 = y1, y2
    
    # Fine search Y
    for y1 in np.arange(best_y1 - 0.005, best_y1 + 0.006, 0.001):
        for y2 in np.arange(best_y2 - 0.005, best_y2 + 0.006, 0.001):
            score = compute_alignment_score(best_x1, y1, best_x2, y2)
            if score < best_score:
                best_score = score
                best_y1, best_y2 = y1, y2
    
    # Coarse search: X offsets
    print("    Optimizing X-offsets...")
    for x1 in np.arange(-max_offset, max_offset + 0.005, 0.005):
        for x2 in np.arange(-max_offset, max_offset + 0.005, 0.005):
            score = compute_alignment_score(x1, best_y1, x2, best_y2)
            if score < best_score:
                best_score = score
                best_x1, best_x2 = x1, x2
    
    # Fine search X
    for x1 in np.arange(best_x1 - 0.005, best_x1 + 0.006, 0.001):
        for x2 in np.arange(best_x2 - 0.005, best_x2 + 0.006, 0.001):
            score = compute_alignment_score(x1, best_y1, x2, best_y2)
            if score < best_score:
                best_score = score
                best_x1, best_x2 = x1, x2
    
    print(f"    Final alignment score: {best_score:.2f}")
    print(f"    Lens 1: X={best_x1:.4f}, Y={best_y1:.4f}")
    print(f"    Lens 2: X={best_x2:.4f}, Y={best_y2:.4f}")
    
    return float(best_x1), float(best_y1), float(best_x2), float(best_y2)


def optimize_y_offset(frame: np.ndarray,
                       center1_x: float, center1_y: float,
                       center2_x: float, center2_y: float,
                       fov1: float, fov2: float,
                       p2_1: float, p2_2: float,
                       roll: float,
                       is_horizontal: bool,
                       project_func) -> float:
    """Optimize Y-offset to align vertical misalignment at the seam (legacy function).
    
    For new code, use optimize_offsets() which handles both lenses.
    """
    h, w = frame.shape[:2]
    
    if is_horizontal:
        lens_w = w // 2
        lens_h = h
    else:
        lens_w = w
        lens_h = h // 2
    
    test_width = lens_h * 2
    test_height = lens_h
    
    def project_with_y_offset(y_off: float) -> np.ndarray:
        params = {
            'lens1CenterX': center1_x,
            'lens1CenterY': center1_y,
            'lens2CenterX': center2_x,
            'lens2CenterY': center2_y + y_off,
            'lens1FOV': fov1,
            'lens2FOV': fov2,
            'lens1P2': p2_1,
            'lens2P2': p2_2,
            'lens1P1': 0.0, 'lens1P3': 0.0, 'lens1P4': 0.0,
            'lens2P1': 0.0, 'lens2P3': 0.0, 'lens2P4': 0.0,
            'lensRotationYaw': 0.0,
            'lensRotationPitch': 0.0,
            'lensRotationRoll': roll,
            'isHorizontal': is_horizontal,
        }
        return project_func(
            frame, params, test_width, test_height,
            apply_calibration=True
        )
    
    def compute_vertical_alignment(y_off: float) -> float:
        try:
            equirect = project_with_y_offset(y_off)
        except:
            return float('inf')
        
        center_x = test_width // 2
        strip_width = 20
        
        left_strip = equirect[:, center_x - strip_width : center_x, :]
        right_strip = equirect[:, center_x : center_x + strip_width, :]
        
        left_gray = cv2.cvtColor(left_strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
        right_gray = cv2.cvtColor(right_strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        left_col = left_gray[:, -1].reshape(-1, 1)
        right_col = right_gray[:, 0].reshape(-1, 1)
        
        valid = (left_col.flatten() > 5) & (right_col.flatten() > 5)
        if np.sum(valid) < 100:
            return float('inf')
        
        left_valid = left_col.flatten()[valid]
        right_valid = right_col.flatten()[valid]
        
        left_norm = (left_valid - np.mean(left_valid)) / (np.std(left_valid) + 1e-6)
        right_norm = (right_valid - np.mean(right_valid)) / (np.std(right_valid) + 1e-6)
        
        best_corr = -1
        best_shift = 0
        max_shift = 20
        
        for shift in range(-max_shift, max_shift + 1):
            if shift < 0:
                l = left_norm[-shift:]
                r = right_norm[:shift]
            elif shift > 0:
                l = left_norm[:-shift]
                r = right_norm[shift:]
            else:
                l = left_norm
                r = right_norm
            
            if len(l) > 50:
                corr = np.corrcoef(l, r)[0, 1]
                if corr > best_corr:
                    best_corr = corr
                    best_shift = shift
        
        return abs(best_shift) + (1 - best_corr) * 10
    
    best_y_off = 0.0
    best_score = compute_vertical_alignment(0.0)
    
    print(f"    Initial Y-alignment score: {best_score:.2f}")
    
    # Coarse search
    for y_off in np.arange(-0.02, 0.021, 0.005):
        score = compute_vertical_alignment(y_off)
        if score < best_score:
            best_score = score
            best_y_off = y_off
    
    # Fine search
    for y_off in np.arange(best_y_off - 0.005, best_y_off + 0.006, 0.001):
        score = compute_vertical_alignment(y_off)
        if score < best_score:
            best_score = score
            best_y_off = y_off
    
    print(f"    Final Y-alignment score: {best_score:.2f}, offset: {best_y_off:.4f}")
    
    return float(best_y_off)


def refine_rotation_from_seam(frame: np.ndarray,
                               center1_x: float, center1_y: float,
                               center2_x: float, center2_y: float,
                               fov1: float, fov2: float,
                               p2_1: float, p2_2: float,
                               initial_yaw: float, initial_pitch: float, initial_roll: float,
                               is_horizontal: bool,
                               project_func) -> tuple:
    """Refine yaw, pitch, and roll by minimizing seam error at center.
    
    Uses a 30px wide × 400px tall region at the center seam for error computation.
    
    Returns:
        (yaw, pitch, roll) - optimized rotation angles in radians
    """
    h, w = frame.shape[:2]
    
    if is_horizontal:
        lens_w = w // 2
        lens_h = h
    else:
        lens_w = w
        lens_h = h // 2
    
    test_width = lens_h * 2
    test_height = lens_h
    
    def project_with_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
        params = {
            'lens1CenterX': center1_x,
            'lens1CenterY': center1_y,
            'lens2CenterX': center2_x,
            'lens2CenterY': center2_y,
            'lens1FOV': fov1,
            'lens2FOV': fov2,
            'lens1P2': p2_1,
            'lens2P2': p2_2,
            'lens1P1': 0.0, 'lens1P3': 0.0, 'lens1P4': 0.0,
            'lens2P1': 0.0, 'lens2P3': 0.0, 'lens2P4': 0.0,
            'lensRotationYaw': yaw,
            'lensRotationPitch': pitch,
            'lensRotationRoll': roll,
            'applyRotation': True,  # Enable rotation application
            'isHorizontal': is_horizontal,
        }
        return project_func(
            frame, params, test_width, test_height,
            apply_calibration=True
        )
    
    def compute_seam_error(yaw: float, pitch: float, roll: float) -> float:
        """Compute seam error using unified scoring (center + edge seams)."""
        try:
            equirect = project_with_rotation(yaw, pitch, roll)
        except:
            return float('inf')
        
        # Use unified seam score (50% height, 5% width, center + edge seams)
        return compute_unified_seam_score(equirect)
    
    # Initialize with provided values
    best_yaw = initial_yaw
    best_pitch = initial_pitch
    best_roll = initial_roll
    best_score = compute_seam_error(best_yaw, best_pitch, best_roll)
    
    print(f"    Initial seam error: {best_score:.2f} (yaw={np.degrees(best_yaw):.2f}°, "
          f"pitch={np.degrees(best_pitch):.2f}°, roll={np.degrees(best_roll):.2f}°)")
    
    # Use coordinate descent: optimize one axis at a time (much faster than 3D grid)
    # This is O(n) per pass instead of O(n³)
    
    def optimize_single_axis(axis_name: str, current_yaw: float, current_pitch: float, 
                             current_roll: float, step_deg: float, range_deg: float) -> tuple:
        """Optimize one axis while keeping others fixed."""
        best_y, best_p, best_r = current_yaw, current_pitch, current_roll
        best_s = compute_seam_error(best_y, best_p, best_r)
        
        for delta in np.arange(-range_deg, range_deg + step_deg/2, step_deg):
            if axis_name == 'yaw':
                y, p, r = current_yaw + np.radians(delta), current_pitch, current_roll
            elif axis_name == 'pitch':
                y, p, r = current_yaw, current_pitch + np.radians(delta), current_roll
            else:  # roll
                y, p, r = current_yaw, current_pitch, current_roll + np.radians(delta)
            
            score = compute_seam_error(y, p, r)
            if score < best_s:
                best_s = score
                best_y, best_p, best_r = y, p, r
        
        return best_y, best_p, best_r, best_s
    
    # === PASS 1: Coarse coordinate descent (±5°, 1° steps) ===
    print(f"    Coarse pass (±5°, 1° steps)...")
    
    for iteration in range(3):  # Multiple passes to let axes interact
        prev_score = best_score
        
        # Optimize each axis in turn
        best_yaw, best_pitch, best_roll, best_score = optimize_single_axis(
            'roll', best_yaw, best_pitch, best_roll, 1.0, 5.0)
        best_yaw, best_pitch, best_roll, best_score = optimize_single_axis(
            'pitch', best_yaw, best_pitch, best_roll, 1.0, 3.0)
        best_yaw, best_pitch, best_roll, best_score = optimize_single_axis(
            'yaw', best_yaw, best_pitch, best_roll, 1.0, 3.0)
        
        # Stop if no improvement
        if abs(prev_score - best_score) < 0.01:
            break
    
    print(f"    After coarse: {best_score:.2f} (yaw={np.degrees(best_yaw):.2f}°, "
          f"pitch={np.degrees(best_pitch):.2f}°, roll={np.degrees(best_roll):.2f}°)")
    
    # === PASS 2: Medium coordinate descent (±1°, 0.2° steps) ===
    print(f"    Medium pass (±1°, 0.2° steps)...")
    
    for iteration in range(3):
        prev_score = best_score
        
        best_yaw, best_pitch, best_roll, best_score = optimize_single_axis(
            'roll', best_yaw, best_pitch, best_roll, 0.2, 1.0)
        best_yaw, best_pitch, best_roll, best_score = optimize_single_axis(
            'pitch', best_yaw, best_pitch, best_roll, 0.2, 1.0)
        best_yaw, best_pitch, best_roll, best_score = optimize_single_axis(
            'yaw', best_yaw, best_pitch, best_roll, 0.2, 1.0)
        
        if abs(prev_score - best_score) < 0.01:
            break
    
    print(f"    After medium: {best_score:.2f} (yaw={np.degrees(best_yaw):.2f}°, "
          f"pitch={np.degrees(best_pitch):.2f}°, roll={np.degrees(best_roll):.2f}°)")
    
    # === PASS 3: Fine coordinate descent (±0.2°, 0.05° steps) ===
    print(f"    Fine pass (±0.2°, 0.05° steps)...")
    
    for iteration in range(2):
        prev_score = best_score
        
        best_yaw, best_pitch, best_roll, best_score = optimize_single_axis(
            'roll', best_yaw, best_pitch, best_roll, 0.05, 0.2)
        best_yaw, best_pitch, best_roll, best_score = optimize_single_axis(
            'pitch', best_yaw, best_pitch, best_roll, 0.05, 0.2)
        best_yaw, best_pitch, best_roll, best_score = optimize_single_axis(
            'yaw', best_yaw, best_pitch, best_roll, 0.05, 0.2)
        
        if abs(prev_score - best_score) < 0.001:
            break
    
    print(f"    Final: {best_score:.2f} (yaw={np.degrees(best_yaw):.2f}°, "
          f"pitch={np.degrees(best_pitch):.2f}°, roll={np.degrees(best_roll):.2f}°)")
    
    return float(best_yaw), float(best_pitch), float(best_roll)


def refine_rotation_both_lenses(frame: np.ndarray,
                                 center1_x: float, center1_y: float,
                                 center2_x: float, center2_y: float,
                                 fov1: float, fov2: float,
                                 p2_1: float, p2_2: float,
                                 is_horizontal: bool,
                                 project_func) -> tuple:
    """Optimize rotation (yaw, pitch, roll) for BOTH lenses.
    
    Returns:
        ((yaw1, pitch1, roll1), (yaw2, pitch2, roll2)) - rotation angles in radians for each lens
    """
    h, w = frame.shape[:2]
    
    if is_horizontal:
        lens_h = h
    else:
        lens_h = h // 2
    
    test_width = lens_h * 2
    test_height = lens_h
    
    def project_with_dual_rotation(y1, p1, r1, y2, p2, r2) -> np.ndarray:
        params = {
            'lens1CenterX': center1_x,
            'lens1CenterY': center1_y,
            'lens2CenterX': center2_x,
            'lens2CenterY': center2_y,
            'lens1FOV': fov1,
            'lens2FOV': fov2,
            'lens1P2': p2_1,
            'lens2P2': p2_2,
            'lens1P1': 0.0, 'lens1P3': 0.0, 'lens1P4': 0.0,
            'lens2P1': 0.0, 'lens2P3': 0.0, 'lens2P4': 0.0,
            # Per-lens rotations
            'lens1RotationYaw': y1,
            'lens1RotationPitch': p1,
            'lens1RotationRoll': r1,
            'lens2RotationYaw': y2,
            'lens2RotationPitch': p2,
            'lens2RotationRoll': r2,
            'applyRotation': True,
            'isHorizontal': is_horizontal,
        }
        return project_func(
            frame, params, test_width, test_height,
            apply_calibration=True
        )
    
    def compute_error(y1, p1, r1, y2, p2, r2) -> float:
        try:
            equirect = project_with_dual_rotation(y1, p1, r1, y2, p2, r2)
        except:
            return float('inf')
        return compute_unified_seam_score(equirect)
    
    # Initialize
    best = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # y1, p1, r1, y2, p2, r2
    best_score = compute_error(*best)
    
    print(f"    Initial seam error: {best_score:.2f}")
    
    # Coordinate descent: optimize one parameter at a time
    # Order: roll (usually largest), then pitch, then yaw
    param_names = ['roll1', 'roll2', 'pitch1', 'pitch2', 'yaw1', 'yaw2']
    param_idx = [2, 5, 1, 4, 0, 3]  # Indices into best array
    
    def optimize_single_param(idx: int, step_deg: float, range_deg: float):
        nonlocal best, best_score
        current_val = best[idx]
        for delta in np.arange(-range_deg, range_deg + step_deg/2, step_deg):
            test = best.copy()
            test[idx] = current_val + np.radians(delta)
            score = compute_error(*test)
            if score < best_score:
                best_score = score
                best[idx] = test[idx]
    
    # Coarse pass (±5°, 1° steps)
    print(f"    Coarse pass (±5°, 1° steps)...")
    for iteration in range(2):
        prev_score = best_score
        for idx in param_idx:
            optimize_single_param(idx, 1.0, 5.0)
        if abs(prev_score - best_score) < 0.01:
            break
    
    print(f"    After coarse: {best_score:.2f}")
    
    # Medium pass (±1°, 0.2° steps)
    print(f"    Medium pass (±1°, 0.2° steps)...")
    for iteration in range(2):
        prev_score = best_score
        for idx in param_idx:
            optimize_single_param(idx, 0.2, 1.0)
        if abs(prev_score - best_score) < 0.01:
            break
    
    print(f"    After medium: {best_score:.2f}")
    
    # Fine pass (±0.2°, 0.05° steps)
    print(f"    Fine pass (±0.2°, 0.05° steps)...")
    for iteration in range(2):
        prev_score = best_score
        for idx in param_idx:
            optimize_single_param(idx, 0.05, 0.2)
        if abs(prev_score - best_score) < 0.01:
            break
    
    print(f"    Final: {best_score:.2f}")
    print(f"    Lens 1: yaw={np.degrees(best[0]):.2f}°, pitch={np.degrees(best[1]):.2f}°, roll={np.degrees(best[2]):.2f}°")
    print(f"    Lens 2: yaw={np.degrees(best[3]):.2f}°, pitch={np.degrees(best[4]):.2f}°, roll={np.degrees(best[5]):.2f}°")
    
    return ((float(best[0]), float(best[1]), float(best[2])), 
            (float(best[3]), float(best[4]), float(best[5])))


def refine_roll_from_seam(frame: np.ndarray,
                           center1_x: float, center1_y: float,
                           center2_x: float, center2_y: float,
                           fov1: float, fov2: float,
                           p2_1: float, p2_2: float,
                           initial_roll: float,
                           is_horizontal: bool,
                           project_func) -> float:
    """Refine roll only (legacy function, use refine_rotation_from_seam for full optimization)."""
    yaw, pitch, roll = refine_rotation_from_seam(
        frame, center1_x, center1_y, center2_x, center2_y,
        fov1, fov2, p2_1, p2_2,
        0.0, 0.0, initial_roll,  # Start with yaw=0, pitch=0
        is_horizontal, project_func
    )
    return roll
