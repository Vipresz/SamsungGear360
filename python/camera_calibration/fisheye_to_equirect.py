#!/usr/bin/env python3
"""Fisheye to equirectangular conversion."""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from calibration_config import CameraCalibration, LensCalibration


def mask_fisheye_circle(img, margin=0):
    """
    Mask the circular fisheye region, zeroing everything outside.
    
    Args:
        img: Input fisheye image
        margin: Pixels to shrink the circle by (to remove edge artifacts)
    
    Returns:
        Masked image with only the circular fisheye region
    """
    h, w = img.shape[:2]
    radius = min(w, h) // 2 - margin
    cx, cy = w // 2, h // 2
    
    # Create circular mask
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    mask = dist <= radius
    
    # Apply mask
    result = img.copy()
    if len(img.shape) == 3:
        result[~mask] = 0
    else:
        result[~mask] = 0
    
    return result, mask


def blend_dual_patches(left_patch, left_mask, right_patch, right_mask, overlap_px):
    """
    Blend two patches for dual-lens 360° camera using linear feathering at seams.
    
    Each patch is placed in its half of the canvas. At the seams (center and edges),
    we use linear blending to create smooth transitions.
    
    This avoids duplication artifacts from the previous accumulator approach.
    """
    h, half_w = left_patch.shape[:2]
    out_w = half_w * 2
    
    # Start with simple adjacent placement
    if len(left_patch.shape) == 3:
        result = np.zeros((h, out_w, left_patch.shape[2]), dtype=np.uint8)
    else:
        result = np.zeros((h, out_w), dtype=np.uint8)
    
    result[:, :half_w] = left_patch
    result[:, half_w:] = right_patch
    
    # Linear blend at center seam (around column half_w)
    if overlap_px > 0:
        blend_width = min(overlap_px, half_w // 4)  # Don't blend too wide
        
        # Center seam: blend left_patch's right edge with right_patch's left edge
        for i in range(blend_width):
            alpha = i / blend_width  # 0 at left edge of blend zone, 1 at right
            col_left = half_w - blend_width + i  # Column in left patch region
            col_right = half_w + i  # Column in right patch region
            
            # Blend: use more of left at start, more of right at end
            # At center seam, left's right edge and right's left edge should match
            left_col = half_w - blend_width + i
            right_col = i
            
            if left_col >= 0 and left_col < half_w and right_col >= 0 and right_col < half_w:
                blended = (1 - alpha) * left_patch[:, -blend_width + i].astype(np.float64) + \
                          alpha * right_patch[:, i].astype(np.float64)
                result[:, half_w - blend_width + i] = blended.astype(np.uint8)
        
        # Side seam: blend at column 0 (wrap-around)
        for i in range(blend_width):
            alpha = i / blend_width
            left_col = i  # Left patch's left edge
            right_col = half_w - blend_width + i  # Right patch's right edge
            
            if left_col < half_w and right_col < half_w:
                blended = (1 - alpha) * right_patch[:, -blend_width + i].astype(np.float64) + \
                          alpha * left_patch[:, i].astype(np.float64)
                result[:, i] = blended.astype(np.uint8)
        
        # Right edge (wrap to complete the 360°)
        for i in range(blend_width):
            alpha = i / blend_width
            blended = (1 - alpha) * left_patch[:, i].astype(np.float64) + \
                      alpha * right_patch[:, -blend_width + i].astype(np.float64)
            result[:, out_w - blend_width + i] = blended.astype(np.uint8)
    
    print(f"  Linear blend at seams (width={overlap_px}px)")
    return result


def fisheye_to_equirect_single(img, output_width, output_height, fov_degrees=180.0):
    """Convert single fisheye to equirectangular."""
    h, w = img.shape[:2]
    radius = min(w, h) // 2
    cx = w / 2.0
    cy = h / 2.0
    
    v, u = np.mgrid[0:output_height, 0:output_width]
    
    # Map to longitude/latitude
    # Use actual lens FOV for longitude range (e.g., 195° covers -97.5° to +97.5°)
    lon_range = np.radians(fov_degrees)
    longitude = (u / output_width) * lon_range - lon_range / 2
    latitude = (0.5 - v / output_height) * np.pi
    
    # Equirectangular to 3D
    Xworld = np.cos(latitude) * np.cos(longitude)
    Yworld = np.cos(latitude) * np.sin(longitude)
    Zworld = np.sin(latitude)
    
    # World to fisheye camera
    Px = -Zworld
    Py = Yworld
    Pz = Xworld
    
    # Fisheye projection
    phi = np.arccos(np.clip(Pz, -1, 1))
    theta = np.arctan2(Py, Px) - np.pi / 2
    
    phi_max_half = np.radians(fov_degrees / 2)
    r = phi / phi_max_half
    
    u_fish = r * np.cos(theta)
    v_fish = -r * np.sin(theta)
    
    x_fish = (cx + u_fish * radius).astype(int)
    y_fish = (cy + v_fish * radius).astype(int)
    
    valid = (x_fish >= 0) & (x_fish < w) & (y_fish >= 0) & (y_fish < h) & (r <= 1)
    
    # Create output with correct dimensions (output_height, output_width, channels)
    if len(img.shape) == 3:
        output = np.zeros((output_height, output_width, img.shape[2]), dtype=img.dtype)
    else:
        output = np.zeros((output_height, output_width), dtype=img.dtype)
    
    output[valid] = img[y_fish[valid], x_fish[valid]]
    
    return output


def fisheye_to_equirect_calibrated(img, output_width, output_height, lens_calib: LensCalibration, base_fov=180.0):
    """
    Convert single fisheye to equirectangular with lens calibration.
    
    Args:
        img: Input fisheye image
        output_width: Output width (resolution for the patch)
        output_height: Output height (spans 180° latitude)
        lens_calib: LensCalibration parameters
            - fov: FOV scale (>1.0 = wider lens, <1.0 = narrower lens than base_fov)
            - k1, k2, k3: Radial distortion (r_distorted = r * (1 + k1*r² + k2*r⁴ + k3*r⁶))
            - center_x, center_y: Optical center (normalized 0-1)
            - rotation_*: 3D rotation (yaw, pitch, roll in radians)
        base_fov: Physical lens FOV in degrees (e.g., 195° for Samsung Gear 360)
    
    Returns:
        tuple: (output_image, valid_mask)
            - output_image: Projected patch spanning base_fov degrees horizontally
            - valid_mask: Boolean array indicating geometrically valid pixels
    
    Notes:
        - Output is a PATCH spanning base_fov degrees (NOT full 360°!)
        - For dual-lens 360° cameras, two patches are stitched together
        - Distortion applied to normalized radius (matches optimizer model)
        - Effective lens FOV = base_fov * fov_scale
        - WARNING: Variable names cx0,cy0 are lens centers; rotation uses cyaw,syaw etc.
    """
    h, w = img.shape[:2]
    
    # Validate input
    if img is None or h == 0 or w == 0:
        raise ValueError(f"Invalid input image: shape={img.shape if img is not None else None}")
    if not (0.0 <= lens_calib.center_x <= 1.0 and 0.0 <= lens_calib.center_y <= 1.0):
        raise ValueError(f"Invalid lens center: ({lens_calib.center_x}, {lens_calib.center_y}) must be in [0,1]")
    
    # Lens parameters
    cx0 = lens_calib.center_x * w  # Optical center X in pixels
    cy0 = lens_calib.center_y * h  # Optical center Y in pixels
    radius = min(w, h) // 2
    fov_scale = lens_calib.fov
    
    v, u = np.mgrid[0:output_height, 0:output_width]
    
    # Map to longitude/latitude
    # Output is a PATCH spanning base_fov degrees (not full 360°)
    # Each lens produces a patch, and patches are later stitched together
    lon_range = np.radians(base_fov)
    longitude = (u / output_width) * lon_range - lon_range / 2  # e.g., -97.5° to +97.5° for 195°
    latitude = (0.5 - v / output_height) * np.pi  # -π/2 to π/2
    
    # Equirectangular to 3D
    Xworld = np.cos(latitude) * np.cos(longitude)
    Yworld = np.cos(latitude) * np.sin(longitude)
    Zworld = np.sin(latitude)
    
    # Apply rotation (if any)
    if abs(lens_calib.rotation_yaw) > 1e-6 or abs(lens_calib.rotation_pitch) > 1e-6 or abs(lens_calib.rotation_roll) > 1e-6:
        # Rotation matrix from yaw, pitch, roll (ZYX Euler angles)
        cyaw, syaw = np.cos(lens_calib.rotation_yaw), np.sin(lens_calib.rotation_yaw)
        cpitch, spitch = np.cos(lens_calib.rotation_pitch), np.sin(lens_calib.rotation_pitch)
        croll, sroll = np.cos(lens_calib.rotation_roll), np.sin(lens_calib.rotation_roll)
        
        # ZYX Euler rotation matrices
        Rz = np.array([[cyaw, -syaw, 0], [syaw, cyaw, 0], [0, 0, 1]], dtype=np.float64)
        Ry = np.array([[cpitch, 0, spitch], [0, 1, 0], [-spitch, 0, cpitch]], dtype=np.float64)
        Rx = np.array([[1, 0, 0], [0, croll, -sroll], [0, sroll, croll]], dtype=np.float64)
        R = Rz @ Ry @ Rx
        
        # Apply rotation
        world_coords = np.stack([Xworld, Yworld, Zworld], axis=-1)
        rotated = world_coords @ R.T
        Xworld, Yworld, Zworld = rotated[..., 0], rotated[..., 1], rotated[..., 2]
        
        # Validate rotation didn't produce invalid values
        if np.any(~np.isfinite(Xworld)) or np.any(~np.isfinite(Yworld)) or np.any(~np.isfinite(Zworld)):
            raise ValueError(f"Rotation produced invalid coordinates (yaw={np.degrees(lens_calib.rotation_yaw):.1f}°)")
    
    # World to fisheye camera
    Px = -Zworld
    Py = Yworld
    Pz = Xworld
    
    # Fisheye projection
    phi = np.arccos(np.clip(Pz, -1, 1))  # Angle from optical axis (radians)
    theta = np.arctan2(Py, Px) - np.pi / 2  # Azimuthal angle
    
    # Calculate effective lens FOV
    # base_fov: Expected lens FOV (e.g., 195° for Samsung Gear 360)
    # fov_scale: Adjustment factor (1.0833 for 195° lens if base_fov=180°)
    phi_max_half = np.radians(base_fov / 2)
    effective_phi_max = phi_max_half * fov_scale  # Actual lens max angle
    
    # Normalize angle by effective FOV (establishes correct coordinate space for distortion)
    r = phi / effective_phi_max  # Normalized radius (0 at center, 1 at edge)
    
    # Apply radial distortion in the properly normalized space
    # Distortion: r_distorted = r * (1 + k1*r² + k2*r⁴ + k3*r⁶)
    # Now k coefficients are independent of fov_scale (no degeneracy)
    if abs(lens_calib.k1) > 1e-6 or abs(lens_calib.k2) > 1e-6 or abs(lens_calib.k3) > 1e-6:
        distortion_factor = 1.0 + lens_calib.k1 * r**2 + lens_calib.k2 * r**4 + lens_calib.k3 * r**6
        r = r * distortion_factor
    
    u_fish = r * np.cos(theta)
    v_fish = -r * np.sin(theta)
    
    # Apply offsets and convert to pixel coordinates
    x_fish = (cx0 + u_fish * radius + lens_calib.offset_x).astype(np.float32)
    y_fish = (cy0 + v_fish * radius + lens_calib.offset_y).astype(np.float32)
    
    # Validate coordinates before remap
    if np.any(~np.isfinite(x_fish)) or np.any(~np.isfinite(y_fish)):
        raise ValueError(f"Fisheye coordinates contain NaN/Inf - check rotation and projection parameters")
    
    # Compute valid mask - check bounds and fisheye circle
    # Use small tolerance for floating point at boundaries
    x_valid = (x_fish >= 0) & (x_fish <= w - 1)
    y_valid = (y_fish >= 0) & (y_fish <= h - 1)
    r_valid = (r <= 1.001)  # Small tolerance for floating point at edge
    
    valid = x_valid & y_valid & r_valid
    
    # Debug: print stats - enable temporarily to diagnose
    import os
    if os.environ.get('DEBUG_PROJECTION'):
        print(f"  Projection debug: w={w}, h={h}, cx0={cx0:.1f}, cy0={cy0:.1f}, radius={radius:.1f}, fov_scale={fov_scale:.4f}")
        print(f"  x_valid={np.sum(x_valid)}, y_valid={np.sum(y_valid)}, r_valid={np.sum(r_valid)}, combined={np.sum(valid)}")
        print(f"  x_fish range: [{np.min(x_fish):.1f}, {np.max(x_fish):.1f}]")
        print(f"  y_fish range: [{np.min(y_fish):.1f}, {np.max(y_fish):.1f}]")
        print(f"  r range: [{np.min(r):.3f}, {np.max(r):.3f}]")

    # Ensure arrays are contiguous and properly typed for cv2.remap
    x_fish = np.ascontiguousarray(x_fish, dtype=np.float32)
    y_fish = np.ascontiguousarray(y_fish, dtype=np.float32)
    
    # Safety: clip coordinates AGGRESSIVELY to prevent segfault
    # OpenCV can crash if coordinates are too far out of bounds
    x_fish = np.clip(x_fish, -10, w+10)
    y_fish = np.clip(y_fish, -10, h+10)
    
    try:
        output = cv2.remap(
            img,
            x_fish,
            y_fish,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
    except Exception as e:
        raise RuntimeError(f"cv2.remap failed: {e}. Coords shape={x_fish.shape}, img shape={img.shape}")
    
    output[~valid] = 0
    return output, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--fov', type=float, default=195.0,
                       help='Lens FOV in degrees (Samsung Gear 360 = 195°)')
    parser.add_argument('--calibration', '-c', type=str, default=None, 
                       help='Calibration JSON file')
    parser.add_argument('--use-calibrated-path', action='store_true',
                       help='Force use of calibrated path with null calibration (for testing)')
    parser.add_argument('--extract-seams', type=str, default=None,
                       help='Output seam overlap visualization to this file')
    args = parser.parse_args()
    
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Could not load {args.input}", file=sys.stderr)
        sys.exit(1)
    
    h, w = img.shape[:2]
    is_dual = abs(w/h - 2.0) < 0.1
    
    # Load calibration if provided
    calibration = None
    use_calibrated = False
    
    if args.calibration:
        try:
            calibration = CameraCalibration.load_json(args.calibration)
            print(f"Loaded calibration from {args.calibration}")
            use_calibrated = True
        except Exception as e:
            print(f"Warning: Could not load calibration: {e}")
    
    # Create null calibration for testing if flag is set
    if args.use_calibrated_path:
        print("Using calibrated path with null/default calibration values:")
        print(f"  center=(0.5, 0.5), fov=1.0, k1=k2=k3=0.0, rotations=0.0, offsets=0.0")
        null_lens = LensCalibration(
            center_x=0.5,
            center_y=0.5,
            fov=1.0,
            k1=0.0,
            k2=0.0,
            k3=0.0,
            rotation_yaw=0.0,
            rotation_pitch=0.0,
            rotation_roll=0.0,
            offset_x=0.0,
            offset_y=0.0
        )
        calibration = CameraCalibration(lens1=null_lens, lens2=null_lens)
        use_calibrated = True
        print(f"  This should produce identical output to uncalibrated path")
    
    if is_dual:
        # Split into left and right
        left_img = img[:, :w//2]
        right_img = img[:, w//2:]
        
        # Mask fisheye circles (remove data outside the lens)
        # margin=0 to use full circle, increase if there are edge artifacts
        left_img, _ = mask_fisheye_circle(left_img, margin=0)
        right_img, _ = mask_fisheye_circle(right_img, margin=0)
        
        # Mirror right image horizontally (it's mounted backward)
        right_img = np.fliplr(right_img)
        
        out_w = args.width or w
        out_h = args.height or (out_w // 2)
        half_w = out_w // 2
        
        print(f"Converting {w}x{h} dual-lens (FOV: {args.fov}°)...")
        
        # Project each lens to its own patch (spanning base_fov degrees)
        if use_calibrated:
            left_patch, left_mask = fisheye_to_equirect_calibrated(left_img, half_w, out_h, calibration.lens1, args.fov)
            right_patch, right_mask = fisheye_to_equirect_calibrated(right_img, half_w, out_h, calibration.lens2, args.fov)
        else:
            left_patch = fisheye_to_equirect_single(left_img, half_w, out_h, args.fov)
            right_patch = fisheye_to_equirect_single(right_img, half_w, out_h, args.fov)
            left_mask = np.any(left_patch > 0, axis=2) if len(left_patch.shape) == 3 else left_patch > 0
            right_mask = np.any(right_patch > 0, axis=2) if len(right_patch.shape) == 3 else right_patch > 0
        
        # Mirror right back to correct orientation
        right_patch = np.fliplr(right_patch)
        right_mask = np.fliplr(right_mask)
        
        # Calculate overlap width from EFFECTIVE FOV
        # If calibration has fov_scale > 1.0, apply it to base_fov
        effective_fov = args.fov
        if use_calibrated and calibration.lens1.fov != 1.0:
            effective_fov = args.fov * calibration.lens1.fov
            print(f"  Effective FOV: {args.fov}° × {calibration.lens1.fov:.3f} = {effective_fov:.1f}°")
        
        # For FOV > 180°, each lens sees (FOV-180)/2 degrees into the other hemisphere
        overlap_deg = (effective_fov - 180.0) / 2.0
        overlap_px = 0
        if overlap_deg > 0:
            pixels_per_degree = half_w / effective_fov
            overlap_px = int(overlap_deg * pixels_per_degree)
            print(f"  Overlap region: {overlap_deg:.1f}° = {overlap_px}px on each side")
            
            # Blend overlapping regions
            result = blend_dual_patches(left_patch, left_mask, right_patch, right_mask, overlap_px)
        else:
            # No overlap, just stitch
            result = np.hstack([left_patch, right_patch])
        
        # Extract seams if requested
        if args.extract_seams and overlap_px > 0:
            # CENTER SEAM: right edge of left vs left edge of right
            center_L = left_patch[:, -overlap_px:]
            center_R = right_patch[:, :overlap_px]
            
            # SIDE SEAM: left edge of left vs right edge of right (wrap-around)
            side_L = left_patch[:, :overlap_px]
            side_R = right_patch[:, -overlap_px:]
            
            # Scale up for visibility
            scale = max(1, 80 // overlap_px)
            if scale > 1:
                center_L = cv2.resize(center_L, (overlap_px * scale, out_h), cv2.INTER_NEAREST)
                center_R = cv2.resize(center_R, (overlap_px * scale, out_h), cv2.INTER_NEAREST)
                side_L = cv2.resize(side_L, (overlap_px * scale, out_h), cv2.INTER_NEAREST)
                side_R = cv2.resize(side_R, (overlap_px * scale, out_h), cv2.INTER_NEAREST)
            
            # Build visualization: [center_L | center_R] / [side_L | side_R]
            gap = np.ones((out_h, 3, 3), dtype=np.uint8) * 128
            row1 = np.hstack([center_L, gap, center_R])
            row2 = np.hstack([side_L, gap, side_R])
            gap_h = np.ones((3, row1.shape[1], 3), dtype=np.uint8) * 128
            seam_vis = np.vstack([row1, gap_h, row2])
            
            # Labels
            cv2.putText(seam_vis, "CENTER: L | R", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(seam_vis, "SIDE: L | R", (5, out_h + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imwrite(args.extract_seams, seam_vis)
            print(f"  Seams saved to: {args.extract_seams}")
    else:
        out_w = args.width or w
        out_h = args.height or w
        
        print(f"Converting {w}x{h} single-lens (FOV: {args.fov}°)...")
        if use_calibrated:
            result, _ = fisheye_to_equirect_calibrated(img, out_w, out_h, calibration.lens1, args.fov)
        else:
            result = fisheye_to_equirect_single(img, out_w, out_h, args.fov)
    
    cv2.imwrite(args.output, result)
    print(f"Saved {result.shape[1]}x{result.shape[0]} to {args.output}")


if __name__ == '__main__':
    main()
