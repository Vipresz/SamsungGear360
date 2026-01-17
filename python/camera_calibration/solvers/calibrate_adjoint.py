#!/usr/bin/env python3
"""Joint optimization calibration for dual-fisheye 360° cameras."""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import minimize

try:
    from camera_calibration.calib.calibration_config import CameraCalibration, LensCalibration
    from camera_calibration.projections.fisheye_to_equirect import (
        fisheye_to_equirect_calibrated, mask_fisheye_circle, blend_dual_patches
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from camera_calibration.calib.calibration_config import CameraCalibration, LensCalibration
    from camera_calibration.projections.fisheye_to_equirect import (
        fisheye_to_equirect_calibrated, mask_fisheye_circle, blend_dual_patches
    )


def downsample_frames(frames_data, scale):
    """Downsample frames for faster optimization."""
    if scale >= 1.0:
        return frames_data
    downsampled = []
    for left_img, right_img in frames_data:
        h, w = left_img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        left_small = cv2.resize(left_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        right_small = cv2.resize(right_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        downsampled.append((left_small, right_small))
    return downsampled


def compute_seam_error(left_img, right_img, lens1, lens2, base_fov):
    """Compute seam alignment error."""
    h, w = left_img.shape[:2]
    
    left_masked, _ = mask_fisheye_circle(left_img, margin=0)
    right_masked, _ = mask_fisheye_circle(right_img, margin=0)
    right_flipped = np.fliplr(right_masked)
    
    half_w, out_h = w, h
    
    try:
        left_patch, left_mask = fisheye_to_equirect_calibrated(left_masked, half_w, out_h, lens1, base_fov)
        right_patch, right_mask = fisheye_to_equirect_calibrated(right_flipped, half_w, out_h, lens2, base_fov)
    except Exception:
        return 1.0
    
    right_patch = np.fliplr(right_patch)
    right_mask = np.fliplr(right_mask)
    
    fov_eff = base_fov * lens1.fov
    overlap_deg = (fov_eff - 180.0) / 2.0
    if overlap_deg <= 0:
        return 1.0
    
    pixels_per_degree = half_w / base_fov
    overlap_px = min(max(1, int(overlap_deg * pixels_per_degree * 1.5)), half_w // 4)
    
    center_L, center_R = left_patch[:, -overlap_px:], right_patch[:, :overlap_px]
    center_mask = left_mask[:, -overlap_px:] & right_mask[:, :overlap_px]
    
    side_L, side_R = left_patch[:, :overlap_px], right_patch[:, -overlap_px:]
    side_mask = left_mask[:, :overlap_px] & right_mask[:, -overlap_px:]
    
    if np.sum(center_mask) < 100 and np.sum(side_mask) < 100:
        return 1.0
    
    def seam_error(img1, img2, mask):
        if np.sum(mask) < 50:
            return 0.5
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
        a, b = g1[mask], g2[mask]
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        return float(np.mean(np.abs(a - b)))
    
    return (seam_error(center_L, center_R, center_mask) + seam_error(side_L, side_R, side_mask)) / 2.0


def params_to_lenses(params, fix_lens1=False):
    """Convert parameter vector to lens calibrations."""
    if fix_lens1:
        lens1 = LensCalibration(
            center_x=params[0], center_y=params[1], fov=params[4],
            k1=params[5], k2=params[6], k3=params[7],
            rotation_yaw=0.0, rotation_pitch=0.0, rotation_roll=0.0
        )
        lens2 = LensCalibration(
            center_x=params[2], center_y=params[3], fov=params[4],
            k1=params[8], k2=params[9], k3=params[10],
            rotation_yaw=params[11], rotation_pitch=params[12], rotation_roll=params[13]
        )
    else:
        lens1 = LensCalibration(
            center_x=params[0], center_y=params[1], fov=params[4],
            k1=params[5], k2=params[6], k3=params[7],
            rotation_yaw=params[11], rotation_pitch=params[12], rotation_roll=params[13]
        )
        lens2 = LensCalibration(
            center_x=params[2], center_y=params[3], fov=params[4],
            k1=params[8], k2=params[9], k3=params[10],
            rotation_yaw=params[14], rotation_pitch=params[15], rotation_roll=params[16]
        )
    return lens1, lens2


def objective(params, frames_data, base_fov, fix_lens1=False):
    """Joint objective function."""
    lens1, lens2 = params_to_lenses(params, fix_lens1)
    total_error = sum(compute_seam_error(l, r, lens1, lens2, base_fov) for l, r in frames_data)
    return total_error / len(frames_data)


def _get_bounds_and_n_params(fix_lens1):
    """Get parameter bounds."""
    if fix_lens1:
        bounds = [
            (0.45, 0.55), (0.45, 0.55), (0.45, 0.55), (0.45, 0.55),  # centers
            (1.01, 1.15),  # fov
            (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3),  # k1,k2,k3 lens1
            (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3),  # k1,k2,k3 lens2
            (-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15),  # rot lens2
        ]
        return bounds, 14
    else:
        bounds = [
            (0.45, 0.55), (0.45, 0.55), (0.45, 0.55), (0.45, 0.55),
            (1.01, 1.15),
            (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3),
            (-0.3, 0.3), (-0.3, 0.3), (-0.3, 0.3),
            (-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15),
            (-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15),
        ]
        return bounds, 17


def _generate_initial_params(fix_lens1, restart_idx):
    """Generate initial parameters."""
    if fix_lens1:
        if restart_idx == 0:
            return np.array([0.5, 0.5, 0.5, 0.5, 1.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return np.array([
            0.5 + np.random.uniform(-0.03, 0.03), 0.5 + np.random.uniform(-0.03, 0.03),
            0.5 + np.random.uniform(-0.03, 0.03), 0.5 + np.random.uniform(-0.03, 0.03),
            1.02 + np.random.uniform(-0.01, 0.05),
            *np.random.uniform(-0.1, 0.1, 6), *np.random.uniform(-0.05, 0.05, 3)
        ])
    else:
        if restart_idx == 0:
            return np.array([0.5, 0.5, 0.5, 0.5, 1.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        return np.array([
            0.5 + np.random.uniform(-0.03, 0.03), 0.5 + np.random.uniform(-0.03, 0.03),
            0.5 + np.random.uniform(-0.03, 0.03), 0.5 + np.random.uniform(-0.03, 0.03),
            1.02 + np.random.uniform(-0.01, 0.05),
            *np.random.uniform(-0.1, 0.1, 6), *np.random.uniform(-0.05, 0.05, 6)
        ])


def calibrate_joint(frames_data, base_fov, fix_lens1=False, n_restarts=1):
    """Joint optimization of all parameters with multi-start."""
    print(f"\n{'='*60}")
    print(f"JOINT CALIBRATION{' (lens1 rotation fixed)' if fix_lens1 else ''}")
    print(f"{'='*60}\n")
    
    bounds, n_params = _get_bounds_and_n_params(fix_lens1)
    print(f"Optimizing {n_params} parameters")
    
    x0_default = _generate_initial_params(fix_lens1, 0)
    lens1, lens2 = params_to_lenses(x0_default, fix_lens1)
    init_error = compute_seam_error(frames_data[0][0], frames_data[0][1], lens1, lens2, base_fov)
    print(f"Initial error: {init_error:.4f}\n")
    
    best_params, best_error = None, float('inf')
    
    for restart in range(n_restarts):
        x0 = _generate_initial_params(fix_lens1, restart)
        if n_restarts > 1:
            print(f"Restart {restart + 1}/{n_restarts}:")
        
        result = minimize(
            objective, x0, args=(frames_data, base_fov, fix_lens1),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 50, 'ftol': 1e-5, 'eps': 0.005}
        )
        
        if result.fun < best_error:
            best_error = result.fun
            best_params = result.x
    
    print(f"\nBest error: {best_error:.4f}")
    
    lens1, lens2 = params_to_lenses(best_params, fix_lens1)
    print(f"\nLens1: center=({lens1.center_x:.4f}, {lens1.center_y:.4f}), k=({lens1.k1:.4f}, {lens1.k2:.4f}, {lens1.k3:.4f})")
    print(f"Lens2: center=({lens2.center_x:.4f}, {lens2.center_y:.4f}), k=({lens2.k1:.4f}, {lens2.k2:.4f}, {lens2.k3:.4f})")
    print(f"FOV scale: {lens1.fov:.4f}")
    
    return CameraCalibration(lens1=lens1, lens2=lens2, is_horizontal=True)


def main():
    parser = argparse.ArgumentParser(description='Joint dual-fisheye calibration')
    parser.add_argument('--image', '-i', help='Dual-fisheye image')
    parser.add_argument('--video', help='Video for multi-frame calibration')
    parser.add_argument('--output', '-o', help='Output calibration JSON')
    parser.add_argument('--output_image', help='Output equirectangular image')
    parser.add_argument('--fov', type=float, default=195.0, help='Base FOV in degrees')
    parser.add_argument('--frames', type=int, default=5, help='Number of video frames')
    parser.add_argument('--scale', type=float, default=0.25, help='Downsample scale')
    parser.add_argument('--fix-lens1', action='store_true', help='Fix lens 1 rotation at 0')
    parser.add_argument('--restarts', type=int, default=1, help='Random restarts')
    args = parser.parse_args()
    
    if not args.image and not args.video:
        parser.error("Provide --image or --video")
    
    frames_data = []
    
    if args.image:
        print(f"Loading: {args.image}")
        img = cv2.imread(args.image)
        if img is None:
            print(f"Error: Cannot load {args.image}")
            sys.exit(1)
        h, w = img.shape[:2]
        frames_data = [(img[:, :w//2], img[:, w//2:])]
    else:
        print(f"Loading video: {args.video}")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: Cannot open {args.video}")
            sys.exit(1)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in np.linspace(0, total_frames-1, min(args.frames, total_frames), dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                h, w = frame.shape[:2]
                frames_data.append((frame[:, :w//2], frame[:, w//2:]))
        cap.release()
    
    print(f"Using {len(frames_data)} frames, FOV={args.fov}°")
    
    if args.scale < 1.0:
        h, w = frames_data[0][0].shape[:2]
        print(f"Downsampling to {int(w*args.scale)}x{int(h*args.scale)}")
        frames_small = downsample_frames(frames_data, args.scale)
    else:
        frames_small = frames_data
    
    final_calib = calibrate_joint(frames_small, args.fov, fix_lens1=args.fix_lens1, n_restarts=args.restarts)
    
    if args.output:
        final_calib.save_json(args.output)
        print(f"\nSaved: {args.output}")
    
    if args.output_image:
        left_img, right_img = frames_data[0]
        h, w = left_img.shape[:2]
        left_masked, _ = mask_fisheye_circle(left_img)
        right_masked, _ = mask_fisheye_circle(right_img)
        right_flipped = np.fliplr(right_masked)
        
        half_w = w
        left_patch, left_mask = fisheye_to_equirect_calibrated(left_masked, half_w, h, final_calib.lens1, args.fov)
        right_patch, right_mask = fisheye_to_equirect_calibrated(right_flipped, half_w, h, final_calib.lens2, args.fov)
        right_patch, right_mask = np.fliplr(right_patch), np.fliplr(right_mask)
        
        fov_eff = args.fov * final_calib.lens1.fov
        overlap_px = max(1, int((fov_eff - 180.0) / 2.0 * half_w / args.fov))
        result = blend_dual_patches(left_patch, left_mask, right_patch, right_mask, overlap_px)
        cv2.imwrite(args.output_image, result)
        print(f"Saved: {args.output_image}")


if __name__ == '__main__':
    main()
