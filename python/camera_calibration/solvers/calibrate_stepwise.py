#!/usr/bin/env python3
"""Stepwise calibration for dual-fisheye 360° cameras."""

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
    left_patch, left_mask = fisheye_to_equirect_calibrated(left_masked, half_w, out_h, lens1, base_fov)
    right_patch, right_mask = fisheye_to_equirect_calibrated(right_flipped, half_w, out_h, lens2, base_fov)
    
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


def optimize_for_fov(frames_data, base_fov, fov_scale, n_restarts=3, fix_lens1=False):
    """Optimize centers, distortion, and rotation for a fixed FOV scale."""
    best_error = float('inf')
    best_centers, best_distortion, best_rotations = None, None, None
    n_rot_params = 3 if fix_lens1 else 6
    
    for restart in range(n_restarts):
        if restart == 0:
            init_centers = np.array([0.5, 0.5, 0.5, 0.5])
            init_distortion = np.zeros(6)
            init_rotations = np.zeros(n_rot_params)
        else:
            init_centers = 0.5 + np.random.uniform(-0.03, 0.03, 4)
            init_distortion = np.random.uniform(-0.1, 0.1, 6)
            init_rotations = np.random.uniform(-0.05, 0.05, n_rot_params)
        
        centers, distortion, rotations, error = _optimize_single_start(
            frames_data, base_fov, fov_scale, init_centers, init_distortion, init_rotations, fix_lens1
        )
        
        if error < best_error:
            best_error = error
            best_centers, best_distortion, best_rotations = centers, distortion, rotations
    
    return best_centers, best_distortion, best_rotations, best_error


def _optimize_single_start(frames_data, base_fov, fov_scale, init_centers, init_distortion, init_rotations, fix_lens1):
    """Run optimization from a single starting point."""
    
    def obj_centers(params):
        cx1, cy1, cx2, cy2 = params
        lens1 = LensCalibration(center_x=cx1, center_y=cy1, fov=fov_scale)
        lens2 = LensCalibration(center_x=cx2, center_y=cy2, fov=fov_scale)
        return sum(compute_seam_error(l, r, lens1, lens2, base_fov) for l, r in frames_data) / len(frames_data)
    
    def obj_distortion(params, centers):
        k1_1, k2_1, k3_1, k1_2, k2_2, k3_2 = params
        cx1, cy1, cx2, cy2 = centers
        lens1 = LensCalibration(center_x=cx1, center_y=cy1, fov=fov_scale, k1=k1_1, k2=k2_1, k3=k3_1)
        lens2 = LensCalibration(center_x=cx2, center_y=cy2, fov=fov_scale, k1=k1_2, k2=k2_2, k3=k3_2)
        return sum(compute_seam_error(l, r, lens1, lens2, base_fov) for l, r in frames_data) / len(frames_data)
    
    def obj_rotation(params, centers, distortion):
        cx1, cy1, cx2, cy2 = centers
        k1_1, k2_1, k3_1, k1_2, k2_2, k3_2 = distortion
        if fix_lens1:
            yaw2, pitch2, roll2 = params
            yaw1, pitch1, roll1 = 0.0, 0.0, 0.0
        else:
            yaw1, pitch1, roll1, yaw2, pitch2, roll2 = params
        
        lens1 = LensCalibration(center_x=cx1, center_y=cy1, fov=fov_scale,
                                k1=k1_1, k2=k2_1, k3=k3_1,
                                rotation_yaw=yaw1, rotation_pitch=pitch1, rotation_roll=roll1)
        lens2 = LensCalibration(center_x=cx2, center_y=cy2, fov=fov_scale,
                                k1=k1_2, k2=k2_2, k3=k3_2,
                                rotation_yaw=yaw2, rotation_pitch=pitch2, rotation_roll=roll2)
        return sum(compute_seam_error(l, r, lens1, lens2, base_fov) for l, r in frames_data) / len(frames_data)
    
    result_c = minimize(obj_centers, init_centers, method='L-BFGS-B',
                        bounds=[(0.45, 0.55)] * 4, options={'maxiter': 20, 'ftol': 1e-5, 'eps': 0.001})
    centers = result_c.x
    
    result_d = minimize(obj_distortion, init_distortion, args=(centers,), method='L-BFGS-B',
                        bounds=[(-0.3, 0.3)] * 6, options={'maxiter': 20, 'ftol': 1e-5, 'eps': 0.005})
    distortion = result_d.x
    
    n_rot_params = 3 if fix_lens1 else 6
    result_r = minimize(obj_rotation, init_rotations, args=(centers, distortion), method='L-BFGS-B',
                        bounds=[(-0.15, 0.15)] * n_rot_params, options={'maxiter': 20, 'ftol': 1e-5, 'eps': 0.002})
    rotations = result_r.x
    
    if fix_lens1:
        rotations = np.array([0.0, 0.0, 0.0, rotations[0], rotations[1], rotations[2]])
    
    return centers, distortion, rotations, result_r.fun


def calibrate_stepwise(frames_data, base_fov, n_restarts=3, fix_lens1=False):
    """Stepwise calibration with FOV grid search."""
    print(f"\n{'='*60}")
    print(f"STEPWISE CALIBRATION{' (lens1 rotation fixed)' if fix_lens1 else ''}")
    print(f"{'='*60}\n")
    
    left_img, right_img = frames_data[0]
    init_lens = LensCalibration()
    init_error = compute_seam_error(left_img, right_img, init_lens, init_lens, base_fov)
    print(f"Initial error: {init_error:.4f}\n")
    
    fov_scales = np.linspace(1.0, 1.1, 6)
    print("Stage 1: FOV Grid Search")
    
    best_fov, best_error = 1.0, float('inf')
    best_centers, best_distortion, best_rotations = None, None, None
    
    for fov_scale in fov_scales:
        centers, distortion, rotations, error = optimize_for_fov(
            frames_data, base_fov, fov_scale, n_restarts=n_restarts, fix_lens1=fix_lens1
        )
        effective_fov = base_fov * fov_scale
        overlap_deg = (effective_fov - 180.0) / 2.0
        print(f"  FOV={fov_scale:.3f} ({effective_fov:.1f}°, overlap={overlap_deg:.1f}°): error={error:.4f}")
        
        if error < best_error:
            best_error, best_fov = error, fov_scale
            best_centers, best_distortion, best_rotations = centers, distortion, rotations
    
    print(f"\nBest FOV: {best_fov:.3f}, error: {best_error:.4f}\n")
    
    print("Stage 2: Fine-tuning...")
    centers, distortion, rotations, final_error = optimize_for_fov(
        frames_data, base_fov, best_fov, n_restarts=n_restarts * 2, fix_lens1=fix_lens1
    )
    print(f"Final error: {final_error:.4f}")
    
    lens1 = LensCalibration(center_x=centers[0], center_y=centers[1], fov=best_fov,
                            k1=distortion[0], k2=distortion[1], k3=distortion[2],
                            rotation_yaw=rotations[0], rotation_pitch=rotations[1], rotation_roll=rotations[2])
    lens2 = LensCalibration(center_x=centers[2], center_y=centers[3], fov=best_fov,
                            k1=distortion[3], k2=distortion[4], k3=distortion[5],
                            rotation_yaw=rotations[3], rotation_pitch=rotations[4], rotation_roll=rotations[5])
    
    return CameraCalibration(lens1=lens1, lens2=lens2, is_horizontal=True)


def main():
    parser = argparse.ArgumentParser(description='Stepwise dual-fisheye calibration')
    parser.add_argument('--image', '-i', help='Dual-fisheye image')
    parser.add_argument('--video', help='Video for multi-frame calibration')
    parser.add_argument('--output', '-o', help='Output calibration JSON')
    parser.add_argument('--output_image', help='Output equirectangular image')
    parser.add_argument('--fov', type=float, default=195.0, help='Base FOV in degrees')
    parser.add_argument('--frames', type=int, default=5, help='Number of video frames')
    parser.add_argument('--scale', type=float, default=0.25, help='Downsample scale')
    parser.add_argument('--restarts', type=int, default=3, help='Random restarts')
    parser.add_argument('--fix-lens1', action='store_true', help='Fix lens 1 rotation at 0')
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
    
    final_calib = calibrate_stepwise(frames_small, args.fov, n_restarts=args.restarts, fix_lens1=args.fix_lens1)
    
    print(f"\n{'='*60}")
    print("FINAL CALIBRATION")
    print(f"{'='*60}")
    print(f"Lens 1: center=({final_calib.lens1.center_x:.4f}, {final_calib.lens1.center_y:.4f}), fov={final_calib.lens1.fov:.4f}")
    print(f"Lens 2: center=({final_calib.lens2.center_x:.4f}, {final_calib.lens2.center_y:.4f}), fov={final_calib.lens2.fov:.4f}")
    
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
