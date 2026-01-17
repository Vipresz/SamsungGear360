#!/usr/bin/env python3
"""Fisheye to equirectangular conversion."""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

try:
    from camera_calibration.calib.calibration_config import CameraCalibration, LensCalibration
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from camera_calibration.calib.calibration_config import CameraCalibration, LensCalibration


def mask_fisheye_circle(img, margin=0):
    """Mask circular fisheye region, zeroing outside."""
    h, w = img.shape[:2]
    radius = min(w, h) // 2 - margin
    cx, cy = w // 2, h // 2
    
    y, x = np.ogrid[:h, :w]
    mask = np.sqrt((x - cx)**2 + (y - cy)**2) <= radius
    
    result = img.copy()
    result[~mask] = 0
    return result, mask


def blend_dual_patches(left_patch, left_mask, right_patch, right_mask, overlap_px):
    """Blend two patches using linear feathering at seams."""
    h, half_w = left_patch.shape[:2]
    out_w = half_w * 2
    
    if len(left_patch.shape) == 3:
        result = np.zeros((h, out_w, left_patch.shape[2]), dtype=np.uint8)
    else:
        result = np.zeros((h, out_w), dtype=np.uint8)
    
    result[:, :half_w] = left_patch
    result[:, half_w:] = right_patch
    
    if overlap_px > 0:
        blend_width = min(overlap_px, half_w // 4)
        
        # Center seam
        for i in range(blend_width):
            alpha = i / blend_width
            left_col = half_w - blend_width + i
            if left_col >= 0 and i < half_w:
                blended = (1 - alpha) * left_patch[:, -blend_width + i].astype(np.float64) + \
                          alpha * right_patch[:, i].astype(np.float64)
                result[:, half_w - blend_width + i] = blended.astype(np.uint8)
        
        # Side seam (left edge)
        for i in range(blend_width):
            alpha = i / blend_width
            if i < half_w:
                blended = (1 - alpha) * right_patch[:, -blend_width + i].astype(np.float64) + \
                          alpha * left_patch[:, i].astype(np.float64)
                result[:, i] = blended.astype(np.uint8)
        
        # Right edge (wrap)
        for i in range(blend_width):
            alpha = i / blend_width
            blended = (1 - alpha) * left_patch[:, i].astype(np.float64) + \
                      alpha * right_patch[:, -blend_width + i].astype(np.float64)
            result[:, out_w - blend_width + i] = blended.astype(np.uint8)
    
    return result


def fisheye_to_equirect_single(img, output_width, output_height, fov_degrees=180.0):
    """Convert single fisheye to equirectangular."""
    h, w = img.shape[:2]
    radius = min(w, h) // 2
    cx, cy = w / 2.0, h / 2.0
    
    v, u = np.mgrid[0:output_height, 0:output_width]
    
    lon_range = np.radians(fov_degrees)
    longitude = (u / output_width) * lon_range - lon_range / 2
    latitude = (0.5 - v / output_height) * np.pi
    
    Xworld = np.cos(latitude) * np.cos(longitude)
    Yworld = np.cos(latitude) * np.sin(longitude)
    Zworld = np.sin(latitude)
    
    Px, Py, Pz = -Zworld, Yworld, Xworld
    
    phi = np.arccos(np.clip(Pz, -1, 1))
    theta = np.arctan2(Py, Px) - np.pi / 2
    
    phi_max_half = np.radians(fov_degrees / 2)
    r = phi / phi_max_half
    
    u_fish = r * np.cos(theta)
    v_fish = -r * np.sin(theta)
    
    x_fish = (cx + u_fish * radius).astype(int)
    y_fish = (cy + v_fish * radius).astype(int)
    
    valid = (x_fish >= 0) & (x_fish < w) & (y_fish >= 0) & (y_fish < h) & (r <= 1)
    
    if len(img.shape) == 3:
        output = np.zeros((output_height, output_width, img.shape[2]), dtype=img.dtype)
    else:
        output = np.zeros((output_height, output_width), dtype=img.dtype)
    
    output[valid] = img[y_fish[valid], x_fish[valid]]
    return output


def fisheye_to_equirect_calibrated(img, output_width, output_height, lens_calib: LensCalibration, base_fov=180.0):
    """Convert single fisheye to equirectangular with lens calibration."""
    h, w = img.shape[:2]
    
    if img is None or h == 0 or w == 0:
        raise ValueError(f"Invalid input image: shape={img.shape if img is not None else None}")
    
    cx0 = lens_calib.center_x * w
    cy0 = lens_calib.center_y * h
    radius = min(w, h) // 2
    fov_scale = lens_calib.fov
    
    v, u = np.mgrid[0:output_height, 0:output_width]
    
    lon_range = np.radians(base_fov)
    longitude = (u / output_width) * lon_range - lon_range / 2
    latitude = (0.5 - v / output_height) * np.pi
    
    Xworld = np.cos(latitude) * np.cos(longitude)
    Yworld = np.cos(latitude) * np.sin(longitude)
    Zworld = np.sin(latitude)
    
    # Apply rotation if any
    if abs(lens_calib.rotation_yaw) > 1e-6 or abs(lens_calib.rotation_pitch) > 1e-6 or abs(lens_calib.rotation_roll) > 1e-6:
        cyaw, syaw = np.cos(lens_calib.rotation_yaw), np.sin(lens_calib.rotation_yaw)
        cpitch, spitch = np.cos(lens_calib.rotation_pitch), np.sin(lens_calib.rotation_pitch)
        croll, sroll = np.cos(lens_calib.rotation_roll), np.sin(lens_calib.rotation_roll)
        
        Rz = np.array([[cyaw, -syaw, 0], [syaw, cyaw, 0], [0, 0, 1]], dtype=np.float64)
        Ry = np.array([[cpitch, 0, spitch], [0, 1, 0], [-spitch, 0, cpitch]], dtype=np.float64)
        Rx = np.array([[1, 0, 0], [0, croll, -sroll], [0, sroll, croll]], dtype=np.float64)
        R = Rz @ Ry @ Rx
        
        world_coords = np.stack([Xworld, Yworld, Zworld], axis=-1)
        rotated = world_coords @ R.T
        Xworld, Yworld, Zworld = rotated[..., 0], rotated[..., 1], rotated[..., 2]
    
    Px, Py, Pz = -Zworld, Yworld, Xworld
    
    phi = np.arccos(np.clip(Pz, -1, 1))
    theta = np.arctan2(Py, Px) - np.pi / 2
    
    phi_max_half = np.radians(base_fov / 2)
    effective_phi_max = phi_max_half * fov_scale
    r = phi / effective_phi_max
    
    # Apply radial distortion
    if abs(lens_calib.k1) > 1e-6 or abs(lens_calib.k2) > 1e-6 or abs(lens_calib.k3) > 1e-6:
        distortion_factor = 1.0 + lens_calib.k1 * r**2 + lens_calib.k2 * r**4 + lens_calib.k3 * r**6
        r = r * distortion_factor
    
    u_fish = r * np.cos(theta)
    v_fish = -r * np.sin(theta)
    
    x_fish = (cx0 + u_fish * radius + lens_calib.offset_x).astype(np.float32)
    y_fish = (cy0 + v_fish * radius + lens_calib.offset_y).astype(np.float32)
    
    x_valid = (x_fish >= 0) & (x_fish <= w - 1)
    y_valid = (y_fish >= 0) & (y_fish <= h - 1)
    r_valid = (r <= 1.001)
    valid = x_valid & y_valid & r_valid
    
    x_fish = np.clip(np.ascontiguousarray(x_fish, dtype=np.float32), -10, w+10)
    y_fish = np.clip(np.ascontiguousarray(y_fish, dtype=np.float32), -10, h+10)
    
    output = cv2.remap(img, x_fish, y_fish, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    output[~valid] = 0
    return output, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--width', type=int, default=None)
    parser.add_argument('--height', type=int, default=None)
    parser.add_argument('--fov', type=float, default=195.0, help='Lens FOV in degrees')
    parser.add_argument('--calibration', '-c', type=str, default=None, help='Calibration JSON file')
    parser.add_argument('--extract-seams', type=str, default=None, help='Output seam visualization')
    args = parser.parse_args()
    
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Could not load {args.input}", file=sys.stderr)
        sys.exit(1)
    
    h, w = img.shape[:2]
    is_dual = abs(w/h - 2.0) < 0.1
    
    calibration = None
    if args.calibration:
        try:
            calibration = CameraCalibration.load_json(args.calibration)
            print(f"Loaded calibration from {args.calibration}")
        except Exception as e:
            print(f"Warning: Could not load calibration: {e}")
    
    if is_dual:
        left_img, _ = mask_fisheye_circle(img[:, :w//2], margin=0)
        right_img, _ = mask_fisheye_circle(img[:, w//2:], margin=0)
        right_img = np.fliplr(right_img)
        
        out_w = args.width or w
        out_h = args.height or (out_w // 2)
        half_w = out_w // 2
        
        print(f"Converting {w}x{h} dual-lens (FOV: {args.fov}°)...")
        
        if calibration:
            left_patch, left_mask = fisheye_to_equirect_calibrated(left_img, half_w, out_h, calibration.lens1, args.fov)
            right_patch, right_mask = fisheye_to_equirect_calibrated(right_img, half_w, out_h, calibration.lens2, args.fov)
        else:
            left_patch = fisheye_to_equirect_single(left_img, half_w, out_h, args.fov)
            right_patch = fisheye_to_equirect_single(right_img, half_w, out_h, args.fov)
            left_mask = np.any(left_patch > 0, axis=2) if len(left_patch.shape) == 3 else left_patch > 0
            right_mask = np.any(right_patch > 0, axis=2) if len(right_patch.shape) == 3 else right_patch > 0
        
        right_patch = np.fliplr(right_patch)
        right_mask = np.fliplr(right_mask)
        
        effective_fov = args.fov * (calibration.lens1.fov if calibration else 1.0)
        overlap_deg = (effective_fov - 180.0) / 2.0
        
        if overlap_deg > 0:
            pixels_per_degree = half_w / effective_fov
            overlap_px = int(overlap_deg * pixels_per_degree)
            print(f"  Overlap: {overlap_deg:.1f}° = {overlap_px}px")
            result = blend_dual_patches(left_patch, left_mask, right_patch, right_mask, overlap_px)
        else:
            result = np.hstack([left_patch, right_patch])
        
        if args.extract_seams and overlap_deg > 0:
            scale = max(1, 80 // overlap_px)
            center_L = cv2.resize(left_patch[:, -overlap_px:], (overlap_px * scale, out_h), cv2.INTER_NEAREST)
            center_R = cv2.resize(right_patch[:, :overlap_px], (overlap_px * scale, out_h), cv2.INTER_NEAREST)
            side_L = cv2.resize(left_patch[:, :overlap_px], (overlap_px * scale, out_h), cv2.INTER_NEAREST)
            side_R = cv2.resize(right_patch[:, -overlap_px:], (overlap_px * scale, out_h), cv2.INTER_NEAREST)
            
            gap = np.ones((out_h, 3, 3), dtype=np.uint8) * 128
            row1 = np.hstack([center_L, gap, center_R])
            row2 = np.hstack([side_L, gap, side_R])
            gap_h = np.ones((3, row1.shape[1], 3), dtype=np.uint8) * 128
            seam_vis = np.vstack([row1, gap_h, row2])
            cv2.imwrite(args.extract_seams, seam_vis)
            print(f"  Seams saved to: {args.extract_seams}")
    else:
        out_w = args.width or w
        out_h = args.height or w
        print(f"Converting {w}x{h} single-lens (FOV: {args.fov}°)...")
        
        if calibration:
            result, _ = fisheye_to_equirect_calibrated(img, out_w, out_h, calibration.lens1, args.fov)
        else:
            result = fisheye_to_equirect_single(img, out_w, out_h, args.fov)
    
    cv2.imwrite(args.output, result)
    print(f"Saved {result.shape[1]}x{result.shape[0]} to {args.output}")


if __name__ == '__main__':
    main()
