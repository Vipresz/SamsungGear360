#!/usr/bin/env python3
"""
Joint optimization calibration for dual-fisheye 360° cameras.

Optimizes all parameters simultaneously using seam alignment error.
Uses the same projection as fisheye_to_equirect.py.
"""
import argparse
import sys
import cv2
import numpy as np
from scipy.optimize import minimize
from calibration_config import CameraCalibration, LensCalibration
from fisheye_to_equirect import (
    fisheye_to_equirect_calibrated,
    mask_fisheye_circle,
    blend_dual_patches
)

LENS_FOV_DEG = 195.0


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
    """
    Compute seam alignment error using same projection as fisheye_to_equirect.py.
    
    Returns average error across center and side seams.
    """
    h, w = left_img.shape[:2]
    
    # Mask fisheye circles
    left_masked, _ = mask_fisheye_circle(left_img, margin=0)
    right_masked, _ = mask_fisheye_circle(right_img, margin=0)
    
    # Flip right (mounted backward)
    right_flipped = np.fliplr(right_masked)
    
    # Project each lens
    half_w = w
    out_h = h
    
    try:
        left_patch, left_mask = fisheye_to_equirect_calibrated(left_masked, half_w, out_h, lens1, base_fov)
        right_patch, right_mask = fisheye_to_equirect_calibrated(right_flipped, half_w, out_h, lens2, base_fov)
    except Exception:
        return 1.0  # Return high error on projection failure
    
    # Flip right back
    right_patch = np.fliplr(right_patch)
    right_mask = np.fliplr(right_mask)
    
    # Calculate overlap width from FOV (as upper bound)
    fov_eff = base_fov * lens1.fov
    overlap_deg = (fov_eff - 180.0) / 2.0
    if overlap_deg <= 0:
        return 1.0  # No overlap
    
    pixels_per_degree = half_w / base_fov
    # Use 1.5x the expected overlap to ensure we capture actual valid region
    # (center/distortion/rotation may shift the valid area)
    overlap_px = max(1, int(overlap_deg * pixels_per_degree * 1.5))
    overlap_px = min(overlap_px, half_w // 4)  # Don't exceed 1/4 of patch width
    
    # Extract seams - the mask intersection will define actual valid overlap
    # CENTER: right edge of left vs left edge of right
    center_L = left_patch[:, -overlap_px:]
    center_R = right_patch[:, :overlap_px]
    center_mask = left_mask[:, -overlap_px:] & right_mask[:, :overlap_px]
    
    # SIDE: left edge of left vs right edge of right
    side_L = left_patch[:, :overlap_px]
    side_R = right_patch[:, -overlap_px:]
    side_mask = left_mask[:, :overlap_px] & right_mask[:, -overlap_px:]
    
    # Check if we have enough valid overlap
    if np.sum(center_mask) < 100 and np.sum(side_mask) < 100:
        return 1.0  # Not enough valid overlap
    
    # Compute normalized seam error (robust to exposure/gain differences)
    def seam_error(img1, img2, mask):
        if np.sum(mask) < 50:
            return 0.5
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # Normalize per-strip: subtract mean, divide by std
        # This makes error robust to exposure/gain/vignetting differences
        a = g1[mask]
        b = g2[mask]
        a = (a - a.mean()) / (a.std() + 1e-6)
        b = (b - b.mean()) / (b.std() + 1e-6)
        return float(np.mean(np.abs(a - b)))
    
    center_err = seam_error(center_L, center_R, center_mask)
    side_err = seam_error(side_L, side_R, side_mask)
    
    return (center_err + side_err) / 2.0


def params_to_lenses(params, fix_lens1=False):
    """Convert parameter vector to lens calibrations.
    
    If fix_lens1=False (17 params):
        [cx1, cy1, cx2, cy2, fov, 
         k1_1, k2_1, k3_1, k1_2, k2_2, k3_2,
         yaw1, pitch1, roll1, yaw2, pitch2, roll2]
    
    If fix_lens1=True (14 params, lens 1 rotation fixed at 0):
        [cx1, cy1, cx2, cy2, fov, 
         k1_1, k2_1, k3_1, k1_2, k2_2, k3_2,
         yaw2, pitch2, roll2]
    """
    if fix_lens1:
        lens1 = LensCalibration(
            center_x=params[0],
            center_y=params[1],
            fov=params[4],
            k1=params[5],
            k2=params[6],
            k3=params[7],
            rotation_yaw=0.0,
            rotation_pitch=0.0,
            rotation_roll=0.0
        )
        lens2 = LensCalibration(
            center_x=params[2],
            center_y=params[3],
            fov=params[4],  # Shared FOV
            k1=params[8],
            k2=params[9],
            k3=params[10],
            rotation_yaw=params[11],
            rotation_pitch=params[12],
            rotation_roll=params[13]
        )
    else:
        lens1 = LensCalibration(
            center_x=params[0],
            center_y=params[1],
            fov=params[4],
            k1=params[5],
            k2=params[6],
            k3=params[7],
            rotation_yaw=params[11],
            rotation_pitch=params[12],
            rotation_roll=params[13]
        )
        lens2 = LensCalibration(
            center_x=params[2],
            center_y=params[3],
            fov=params[4],  # Shared FOV
            k1=params[8],
            k2=params[9],
            k3=params[10],
            rotation_yaw=params[14],
            rotation_pitch=params[15],
            rotation_roll=params[16]
        )
    return lens1, lens2


def objective(params, frames_data, base_fov, fix_lens1=False):
    """Joint objective function."""
    lens1, lens2 = params_to_lenses(params, fix_lens1)
    
    total_error = 0.0
    for left_img, right_img in frames_data:
        error = compute_seam_error(left_img, right_img, lens1, lens2, base_fov)
        total_error += error
    
    return total_error / len(frames_data)


def _get_bounds_and_n_params(fix_lens1):
    """Get parameter bounds based on fix_lens1 setting."""
    if fix_lens1:
        # 14 parameters: centers, fov, distortion, lens2 rotation only
        bounds = [
            (0.45, 0.55),  # cx1
            (0.45, 0.55),  # cy1
            (0.45, 0.55),  # cx2
            (0.45, 0.55),  # cy2
            (1.01, 1.15),  # fov
            (-0.3, 0.3),   # k1_1
            (-0.3, 0.3),   # k2_1
            (-0.3, 0.3),   # k3_1
            (-0.3, 0.3),   # k1_2
            (-0.3, 0.3),   # k2_2
            (-0.3, 0.3),   # k3_2
            (-0.15, 0.15), # yaw2 (±8.6°)
            (-0.15, 0.15), # pitch2
            (-0.15, 0.15), # roll2
        ]
        n_params = 14
    else:
        # Full 17 parameters
        bounds = [
            (0.45, 0.55),  # cx1
            (0.45, 0.55),  # cy1
            (0.45, 0.55),  # cx2
            (0.45, 0.55),  # cy2
            (1.01, 1.15),  # fov - must be > 1.0 for overlap!
            (-0.3, 0.3),   # k1_1
            (-0.3, 0.3),   # k2_1
            (-0.3, 0.3),   # k3_1
            (-0.3, 0.3),   # k1_2
            (-0.3, 0.3),   # k2_2
            (-0.3, 0.3),   # k3_2
            (-0.15, 0.15), # yaw1 (±8.6°)
            (-0.15, 0.15), # pitch1
            (-0.15, 0.15), # roll1
            (-0.15, 0.15), # yaw2
            (-0.15, 0.15), # pitch2
            (-0.15, 0.15), # roll2
        ]
        n_params = 17
    return bounds, n_params


def _generate_initial_params(fix_lens1, restart_idx):
    """Generate initial parameters for optimization.
    
    restart_idx=0: use default centered values
    restart_idx>0: random perturbation within bounds
    """
    if fix_lens1:
        # 14 parameters
        if restart_idx == 0:
            return np.array([0.5, 0.5, 0.5, 0.5, 1.02, 
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0])
        else:
            return np.array([
                0.5 + np.random.uniform(-0.03, 0.03),  # cx1
                0.5 + np.random.uniform(-0.03, 0.03),  # cy1
                0.5 + np.random.uniform(-0.03, 0.03),  # cx2
                0.5 + np.random.uniform(-0.03, 0.03),  # cy2
                1.02 + np.random.uniform(-0.01, 0.05), # fov
                *np.random.uniform(-0.1, 0.1, 6),      # distortion
                *np.random.uniform(-0.05, 0.05, 3),    # lens2 rotation
            ])
    else:
        # 17 parameters
        if restart_idx == 0:
            return np.array([0.5, 0.5, 0.5, 0.5, 1.02, 
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                           0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            return np.array([
                0.5 + np.random.uniform(-0.03, 0.03),  # cx1
                0.5 + np.random.uniform(-0.03, 0.03),  # cy1
                0.5 + np.random.uniform(-0.03, 0.03),  # cx2
                0.5 + np.random.uniform(-0.03, 0.03),  # cy2
                1.02 + np.random.uniform(-0.01, 0.05), # fov
                *np.random.uniform(-0.1, 0.1, 6),      # distortion
                *np.random.uniform(-0.05, 0.05, 6),    # rotations
            ])


def _run_single_optimization(frames_data, base_fov, fix_lens1, x0, bounds, verbose=True):
    """Run a single optimization from given starting point."""
    iter_count = [0]
    best_error = [float('inf')]
    
    def callback(xk):
        iter_count[0] += 1
        if verbose:
            err = objective(xk, frames_data, base_fov, fix_lens1)
            if err < best_error[0]:
                best_error[0] = err
                print(f"    iter {iter_count[0]:3d}: error={err:.6f} (improved)")
            elif iter_count[0] % 10 == 0:
                print(f"    iter {iter_count[0]:3d}: error={err:.6f}")
    
    result = minimize(
        objective, x0,
        args=(frames_data, base_fov, fix_lens1),
        method='L-BFGS-B',
        bounds=bounds,
        callback=callback if verbose else None,
        options={'maxiter': 50, 'ftol': 1e-5, 'eps': 0.005}
    )
    
    return result.x, result.fun, result.message


def calibrate_joint(frames_data, base_fov, fix_lens1=False, n_restarts=1):
    """
    Joint optimization of all parameters with multi-start.
    
    If fix_lens1=True, lens 1 rotation is fixed at 0 (only lens 2 rotation is optimized).
    This reduces ambiguity since only relative rotation matters.
    
    n_restarts: Number of random restarts (1 = single run from default start)
    """
    mode = "JOINT CALIBRATION"
    if fix_lens1:
        mode += " (lens1 rotation fixed)"
    if n_restarts > 1:
        mode += f" ({n_restarts} restarts)"
    
    print(f"\n{'='*60}")
    print(mode)
    print(f"{'='*60}\n")
    
    bounds, n_params = _get_bounds_and_n_params(fix_lens1)
    print(f"Optimizing {n_params} parameters")
    
    # Show initial error
    x0_default = _generate_initial_params(fix_lens1, 0)
    lens1, lens2 = params_to_lenses(x0_default, fix_lens1)
    left_img, right_img = frames_data[0]
    init_error = compute_seam_error(left_img, right_img, lens1, lens2, base_fov)
    fov_eff = base_fov * x0_default[4]
    overlap_deg = (fov_eff - 180.0) / 2.0
    print(f"Initial: FOV={x0_default[4]:.3f} ({fov_eff:.1f}°, overlap={overlap_deg:.1f}°), error={init_error:.4f}\n")
    
    # Multi-start optimization
    best_params = None
    best_error = float('inf')
    best_message = ""
    
    for restart in range(n_restarts):
        x0 = _generate_initial_params(fix_lens1, restart)
        
        if n_restarts > 1:
            print(f"Restart {restart + 1}/{n_restarts}:")
            verbose = (restart == 0)  # Only verbose on first restart
        else:
            print("Optimizing...")
            verbose = True
        
        params, error, message = _run_single_optimization(
            frames_data, base_fov, fix_lens1, x0, bounds, verbose=verbose
        )
        
        if n_restarts > 1:
            improved = " (new best!)" if error < best_error else ""
            print(f"  → error={error:.6f}{improved}")
        
        if error < best_error:
            best_error = error
            best_params = params
            best_message = message
    
    print(f"\nBest error: {best_error:.4f} ({best_message})")
    
    # Extract final calibration
    lens1, lens2 = params_to_lenses(best_params, fix_lens1)
    
    print(f"\nOptimized parameters:")
    print(f"  Lens1: center=({lens1.center_x:.4f}, {lens1.center_y:.4f})")
    print(f"         k1={lens1.k1:.4f}, k2={lens1.k2:.4f}, k3={lens1.k3:.4f}")
    rot1_note = " (fixed)" if fix_lens1 else ""
    print(f"         rot=(yaw={np.degrees(lens1.rotation_yaw):.2f}°, pitch={np.degrees(lens1.rotation_pitch):.2f}°, roll={np.degrees(lens1.rotation_roll):.2f}°){rot1_note}")
    print(f"  Lens2: center=({lens2.center_x:.4f}, {lens2.center_y:.4f})")
    print(f"         k1={lens2.k1:.4f}, k2={lens2.k2:.4f}, k3={lens2.k3:.4f}")
    print(f"         rot=(yaw={np.degrees(lens2.rotation_yaw):.2f}°, pitch={np.degrees(lens2.rotation_pitch):.2f}°, roll={np.degrees(lens2.rotation_roll):.2f}°)")
    print(f"  FOV scale: {lens1.fov:.4f}")
    
    return CameraCalibration(lens1=lens1, lens2=lens2, is_horizontal=True)


def main():
    parser = argparse.ArgumentParser(description='Joint dual-fisheye calibration')
    parser.add_argument('--image', '-i', help='Dual-fisheye image')
    parser.add_argument('--video', help='Video for multi-frame calibration')
    parser.add_argument('--output', '-o', help='Output calibration JSON')
    parser.add_argument('--output_image', help='Output equirectangular image')
    parser.add_argument('--fov', type=float, default=195.0, help='Base FOV in degrees')
    parser.add_argument('--frames', type=int, default=5, help='Number of video frames')
    parser.add_argument('--scale', type=float, default=0.25, help='Downsample scale for optimization (0.25=4x faster)')
    parser.add_argument('--fix-lens1', action='store_true', help='Fix lens 1 rotation at 0 (only optimize lens 2 rotation)')
    parser.add_argument('--restarts', type=int, default=1, help='Number of random restarts to escape local minima')
    args = parser.parse_args()
    
    if not args.image and not args.video:
        parser.error("Provide --image or --video")
    
    frames_data = []
    source_frame = None
    
    if args.image:
        print(f"Loading: {args.image}")
        img = cv2.imread(args.image)
        if img is None:
            print(f"Error: Cannot load {args.image}")
            sys.exit(1)
        
        source_frame = img
        h, w = img.shape[:2]
        left_img = img[:, :w//2]
        right_img = img[:, w//2:]
        frames_data = [(left_img, right_img)]
    else:
        print(f"Loading video: {args.video}")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Error: Cannot open {args.video}")
            sys.exit(1)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, min(args.frames, total_frames), dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                if source_frame is None:
                    source_frame = frame
                h, w = frame.shape[:2]
                left_img = frame[:, :w//2]
                right_img = frame[:, w//2:]
                frames_data.append((left_img, right_img))
        cap.release()
    
    print(f"Using {len(frames_data)} frames, FOV={args.fov}°")
    
    # Downsample for faster optimization
    if args.scale < 1.0:
        h, w = frames_data[0][0].shape[:2]
        print(f"Downsampling {w}x{h} → {int(w*args.scale)}x{int(h*args.scale)} for optimization ({args.scale:.0%} scale)")
        frames_small = downsample_frames(frames_data, args.scale)
    else:
        frames_small = frames_data
    
    # Run calibration on downsampled frames
    final_calib = calibrate_joint(frames_small, args.fov, fix_lens1=args.fix_lens1, n_restarts=args.restarts)
    
    # Compute final error on full-res frames for reporting
    left_img, right_img = frames_data[0]
    final_error = compute_seam_error(left_img, right_img, final_calib.lens1, final_calib.lens2, args.fov)
    
    # Print results
    print(f"\n{'='*60}")
    print("FINAL CALIBRATION")
    print(f"{'='*60}")
    print(f">>> BEST ERROR: {final_error:.6f} <<<\n")
    print(f"Lens 1: center=({final_calib.lens1.center_x:.4f}, {final_calib.lens1.center_y:.4f})")
    print(f"        k1={final_calib.lens1.k1:.4f}, k2={final_calib.lens1.k2:.4f}, k3={final_calib.lens1.k3:.4f}")
    print(f"        rot=(yaw={np.degrees(final_calib.lens1.rotation_yaw):.2f}°, pitch={np.degrees(final_calib.lens1.rotation_pitch):.2f}°, roll={np.degrees(final_calib.lens1.rotation_roll):.2f}°)")
    print(f"        fov={final_calib.lens1.fov:.4f}")
    print(f"Lens 2: center=({final_calib.lens2.center_x:.4f}, {final_calib.lens2.center_y:.4f})")
    print(f"        k1={final_calib.lens2.k1:.4f}, k2={final_calib.lens2.k2:.4f}, k3={final_calib.lens2.k3:.4f}")
    print(f"        rot=(yaw={np.degrees(final_calib.lens2.rotation_yaw):.2f}°, pitch={np.degrees(final_calib.lens2.rotation_pitch):.2f}°, roll={np.degrees(final_calib.lens2.rotation_roll):.2f}°)")
    print(f"        fov={final_calib.lens2.fov:.4f}")
    
    # Save calibration
    if args.output:
        final_calib.save_json(args.output)
        print(f"\nSaved calibration to: {args.output}")
    
    # Generate output image
    if args.output_image:
        print(f"\nGenerating output image...")
        left_img, right_img = frames_data[0]
        h, w = left_img.shape[:2]
        
        left_masked, _ = mask_fisheye_circle(left_img, margin=0)
        right_masked, _ = mask_fisheye_circle(right_img, margin=0)
        right_flipped = np.fliplr(right_masked)
        
        out_w = w * 2
        out_h = h
        half_w = out_w // 2
        
        left_patch, left_mask = fisheye_to_equirect_calibrated(left_masked, half_w, out_h, final_calib.lens1, args.fov)
        right_patch, right_mask = fisheye_to_equirect_calibrated(right_flipped, half_w, out_h, final_calib.lens2, args.fov)
        right_patch = np.fliplr(right_patch)
        right_mask = np.fliplr(right_mask)
        
        fov_eff = args.fov * final_calib.lens1.fov
        overlap_deg = (fov_eff - 180.0) / 2.0
        overlap_px = max(1, int(overlap_deg * half_w / args.fov))
        
        result = blend_dual_patches(left_patch, left_mask, right_patch, right_mask, overlap_px)
        cv2.imwrite(args.output_image, result)
        print(f"Saved to: {args.output_image}")


if __name__ == '__main__':
    main()
