#!/usr/bin/env python3
"""
Stepwise calibration for dual-fisheye 360° cameras.

Optimizes lens parameters to minimize seam alignment error.
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


def compute_seam_error(left_img, right_img, lens1, lens2, base_fov, verbose=False):
    """
    Compute seam alignment error using same projection as fisheye_to_equirect.py.
    
    The overlap region is RECOMPUTED each call based on current FOV scale:
    - fov_effective = base_fov * lens.fov
    - overlap_deg = (fov_effective - 180) / 2
    - overlap_px = overlap_deg * (patch_width / base_fov)
    
    This ensures the optimizer always compares the correct overlap regions
    as parameters change.
    
    Returns average error across center and side seams.
    """
    h, w = left_img.shape[:2]
    
    # Mask fisheye circles
    left_masked, _ = mask_fisheye_circle(left_img, margin=0)
    right_masked, _ = mask_fisheye_circle(right_img, margin=0)
    
    # Flip right (mounted backward)
    right_flipped = np.fliplr(right_masked)
    
    # Project each lens with CURRENT calibration parameters
    half_w = w
    out_h = h
    
    left_patch, left_mask = fisheye_to_equirect_calibrated(left_masked, half_w, out_h, lens1, base_fov)
    right_patch, right_mask = fisheye_to_equirect_calibrated(right_flipped, half_w, out_h, lens2, base_fov)
    
    # Flip right back
    right_patch = np.fliplr(right_patch)
    right_mask = np.fliplr(right_mask)
    
    # Calculate overlap width from CURRENT FOV scale (as upper bound)
    fov_eff = base_fov * lens1.fov
    overlap_deg = (fov_eff - 180.0) / 2.0
    if overlap_deg <= 0:
        return 1.0  # No overlap possible
    
    pixels_per_degree = half_w / base_fov
    # Use 1.5x the expected overlap to ensure we capture actual valid region
    # (center/distortion/rotation may shift the valid area)
    overlap_px = max(1, int(overlap_deg * pixels_per_degree * 1.5))
    overlap_px = min(overlap_px, half_w // 4)  # Don't exceed 1/4 of patch width
    
    if verbose:
        print(f"    [compute_seam_error] fov_eff={fov_eff:.1f}°, overlap={overlap_deg:.2f}° ({overlap_px}px)")
    
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


def optimize_for_fov(frames_data, base_fov, fov_scale, verbose=False, n_restarts=3, fix_lens1=False):
    """
    Optimize centers, distortion, and rotation for a fixed FOV scale.
    
    Uses multi-start optimization to escape local minima:
    - Runs n_restarts times with random initial conditions
    - Keeps the best result across all restarts
    
    If fix_lens1=True, lens 1 rotation is fixed at 0.
    """
    best_error = float('inf')
    best_centers = None
    best_distortion = None
    best_rotations = None
    
    # Number of rotation parameters: 3 if fix_lens1, 6 otherwise
    n_rot_params = 3 if fix_lens1 else 6
    
    for restart in range(n_restarts):
        # Random perturbation for restarts > 0
        if restart == 0:
            init_centers = np.array([0.5, 0.5, 0.5, 0.5])
            init_distortion = np.zeros(6)
            init_rotations = np.zeros(n_rot_params)
        else:
            # Random starting point within bounds
            init_centers = 0.5 + np.random.uniform(-0.03, 0.03, 4)
            init_distortion = np.random.uniform(-0.1, 0.1, 6)
            init_rotations = np.random.uniform(-0.05, 0.05, n_rot_params)
        
        centers, distortion, rotations, error = _optimize_single_start(
            frames_data, base_fov, fov_scale, 
            init_centers, init_distortion, init_rotations,
            verbose=(verbose and restart == 0),
            fix_lens1=fix_lens1
        )
        
        if error < best_error:
            best_error = error
            best_centers = centers
            best_distortion = distortion
            best_rotations = rotations
            if verbose and restart > 0:
                print(f"    [restart {restart}] found better: {error:.6f}")
    
    return best_centers, best_distortion, best_rotations, best_error


def _optimize_single_start(frames_data, base_fov, fov_scale, 
                           init_centers, init_distortion, init_rotations, 
                           verbose=False, fix_lens1=False):
    """Run optimization from a single starting point.
    
    If fix_lens1=True, lens 1 rotation is fixed at 0 (only 3 rotation params).
    """
    call_count = [0]
    errors_history = []
    
    def obj_centers(params):
        cx1, cy1, cx2, cy2 = params
        lens1 = LensCalibration(center_x=cx1, center_y=cy1, fov=fov_scale)
        lens2 = LensCalibration(center_x=cx2, center_y=cy2, fov=fov_scale)
        
        total = 0.0
        for left_img, right_img in frames_data:
            # Each call re-projects with NEW centers and computes NEW overlap content
            total += compute_seam_error(left_img, right_img, lens1, lens2, base_fov)
        
        error = total / len(frames_data)
        
        # Track for verbose output
        if verbose and call_count[0] < 10:
            errors_history.append((call_count[0], cx1, cy1, cx2, cy2, error))
        call_count[0] += 1
        
        return error
    
    distortion_call_count = [0]
    
    def obj_distortion(params, centers):
        # 6 parameters: k1, k2, k3 for each lens
        k1_1, k2_1, k3_1, k1_2, k2_2, k3_2 = params
        cx1, cy1, cx2, cy2 = centers
        lens1 = LensCalibration(center_x=cx1, center_y=cy1, fov=fov_scale, k1=k1_1, k2=k2_1, k3=k3_1)
        lens2 = LensCalibration(center_x=cx2, center_y=cy2, fov=fov_scale, k1=k1_2, k2=k2_2, k3=k3_2)
        
        total = 0.0
        for left_img, right_img in frames_data:
            total += compute_seam_error(left_img, right_img, lens1, lens2, base_fov)
        
        error = total / len(frames_data)
        
        if verbose and distortion_call_count[0] < 3:
            print(f"      [distortion {distortion_call_count[0]}] k1=({k1_1:.4f},{k1_2:.4f}) → err={error:.6f}")
        distortion_call_count[0] += 1
        
        return error
    
    def obj_rotation(params, centers, distortion):
        cx1, cy1, cx2, cy2 = centers
        k1_1, k2_1, k3_1, k1_2, k2_2, k3_2 = distortion
        
        if fix_lens1:
            # 3 parameters: yaw, pitch, roll for lens 2 only (lens 1 fixed at 0)
            yaw2, pitch2, roll2 = params
            yaw1, pitch1, roll1 = 0.0, 0.0, 0.0
        else:
            # 6 parameters: yaw, pitch, roll for each lens
            yaw1, pitch1, roll1, yaw2, pitch2, roll2 = params
        
        lens1 = LensCalibration(
            center_x=cx1, center_y=cy1, fov=fov_scale,
            k1=k1_1, k2=k2_1, k3=k3_1,
            rotation_yaw=yaw1, rotation_pitch=pitch1, rotation_roll=roll1
        )
        lens2 = LensCalibration(
            center_x=cx2, center_y=cy2, fov=fov_scale,
            k1=k1_2, k2=k2_2, k3=k3_2,
            rotation_yaw=yaw2, rotation_pitch=pitch2, rotation_roll=roll2
        )
        
        total = 0.0
        for left_img, right_img in frames_data:
            total += compute_seam_error(left_img, right_img, lens1, lens2, base_fov)
        
        return total / len(frames_data)
    
    # Stage 1: Optimize centers (from init_centers)
    result_c = minimize(
        obj_centers, init_centers,
        method='L-BFGS-B',
        bounds=[(0.45, 0.55)] * 4,
        options={'maxiter': 20, 'ftol': 1e-5, 'eps': 0.001}
    )
    centers = result_c.x
    
    if verbose and errors_history:
        print(f"    Center optimization: {len(errors_history)} evaluations")
        for i, cx1, cy1, cx2, cy2, err in errors_history[:3]:
            print(f"      [{i}] c1=({cx1:.6f},{cy1:.6f}) c2=({cx2:.6f},{cy2:.6f}) → err={err:.6f}")
    
    # Stage 2: Optimize distortion (from init_distortion)
    result_d = minimize(
        obj_distortion, init_distortion,
        args=(centers,),
        method='L-BFGS-B',
        bounds=[(-0.3, 0.3)] * 6,
        options={'maxiter': 20, 'ftol': 1e-5, 'eps': 0.005}
    )
    distortion = result_d.x
    
    # Stage 3: Optimize rotations (from init_rotations)
    # Number of params: 3 if fix_lens1, 6 otherwise
    n_rot_params = 3 if fix_lens1 else 6
    result_r = minimize(
        obj_rotation, init_rotations,
        args=(centers, distortion),
        method='L-BFGS-B',
        bounds=[(-0.15, 0.15)] * n_rot_params,  # ±8.6° rotation range
        options={'maxiter': 20, 'ftol': 1e-5, 'eps': 0.002}
    )
    rotations = result_r.x
    
    # If fix_lens1, expand rotations to 6 values for consistency (lens1 = 0)
    if fix_lens1:
        rotations = np.array([0.0, 0.0, 0.0, rotations[0], rotations[1], rotations[2]])
    
    return centers, distortion, rotations, result_r.fun


def calibrate_stepwise(frames_data, initial_calib, base_fov, verbose=False, n_restarts=3, fix_lens1=False):
    """
    Stepwise calibration with FOV grid search and multi-start optimization.
    
    Steps through FOV scales from 1.0 to 1.1 (assuming base_fov=180),
    optimizing centers, distortion, and rotation for each, then picks the best.
    
    Uses n_restarts random starting points to escape local minima.
    
    If fix_lens1=True, lens 1 rotation is fixed at 0 (only lens 2 rotation is optimized).
    """
    mode = f"STEPWISE CALIBRATION (FOV Grid + {n_restarts} restarts)"
    if fix_lens1:
        mode += " [lens1 rotation fixed]"
    print(f"\n{'='*60}")
    print(mode)
    print(f"{'='*60}\n")
    
    # Evaluate initial error
    left_img, right_img = frames_data[0]
    init_error = compute_seam_error(
        left_img, right_img, 
        initial_calib.lens1, initial_calib.lens2, 
        base_fov
    )
    print(f"Initial seam error: {init_error:.4f}\n")
    
    # FOV scale grid search (coarse first, then fine-tune)
    # For base_fov=180: fov_scale=1.0 → 180°, fov_scale=1.0833 → 195°
    fov_scales = np.linspace(1.0, 1.1, 6)  # 1.0, 1.02, 1.04, 1.06, 1.08, 1.1
    
    print("Stage 1: FOV Grid Search")
    print("-" * 40)
    
    best_fov = 1.0
    best_error = float('inf')
    best_centers = None
    best_distortion = None
    best_rotations = None
    
    for i, fov_scale in enumerate(fov_scales):
        show_verbose = verbose and (i == 0)
        centers, distortion, rotations, error = optimize_for_fov(
            frames_data, base_fov, fov_scale, verbose=show_verbose, n_restarts=n_restarts, fix_lens1=fix_lens1
        )
        effective_fov = base_fov * fov_scale
        overlap_deg = (effective_fov - 180.0) / 2.0
        print(f"  FOV={fov_scale:.3f} ({effective_fov:.1f}°, overlap={overlap_deg:.1f}°): error={error:.4f}")
        
        if error < best_error:
            best_error = error
            best_fov = fov_scale
            best_centers = centers
            best_distortion = distortion
            best_rotations = rotations
    
    print(f"\nBest FOV scale: {best_fov:.3f} ({base_fov * best_fov:.1f}°)")
    print(f"Best error: {best_error:.4f}\n")
    
    # Stage 2: Fine-tune with best FOV (more restarts for final refinement)
    print(f"Stage 2: Fine-tuning with best FOV ({n_restarts * 2} restarts)...")
    centers, distortion, rotations, final_error = optimize_for_fov(
        frames_data, base_fov, best_fov, n_restarts=n_restarts * 2, fix_lens1=fix_lens1
    )
    
    rot1_note = " (fixed)" if fix_lens1 else ""
    print(f"  Lens1: center=({centers[0]:.4f}, {centers[1]:.4f})")
    print(f"         k1={distortion[0]:.4f}, k2={distortion[1]:.4f}, k3={distortion[2]:.4f}")
    print(f"         rot=({np.degrees(rotations[0]):.2f}°, {np.degrees(rotations[1]):.2f}°, {np.degrees(rotations[2]):.2f}°){rot1_note}")
    print(f"  Lens2: center=({centers[2]:.4f}, {centers[3]:.4f})")
    print(f"         k1={distortion[3]:.4f}, k2={distortion[4]:.4f}, k3={distortion[5]:.4f}")
    print(f"         rot=({np.degrees(rotations[3]):.2f}°, {np.degrees(rotations[4]):.2f}°, {np.degrees(rotations[5]):.2f}°)")
    print(f"  Final error: {final_error:.4f}\n")
    
    # Build final calibration with all parameters
    lens1 = LensCalibration(
        center_x=centers[0], center_y=centers[1],
        fov=best_fov, k1=distortion[0], k2=distortion[1], k3=distortion[2],
        rotation_yaw=rotations[0], rotation_pitch=rotations[1], rotation_roll=rotations[2]
    )
    lens2 = LensCalibration(
        center_x=centers[2], center_y=centers[3],
        fov=best_fov, k1=distortion[3], k2=distortion[4], k3=distortion[5],
        rotation_yaw=rotations[3], rotation_pitch=rotations[4], rotation_roll=rotations[5]
    )
    
    return CameraCalibration(lens1=lens1, lens2=lens2, is_horizontal=True)


def main():
    parser = argparse.ArgumentParser(description='Stepwise dual-fisheye calibration')
    parser.add_argument('--image', '-i', help='Single dual-fisheye image')
    parser.add_argument('--video', help='Video file for multi-frame calibration')
    parser.add_argument('--output', '-o', help='Output calibration JSON')
    parser.add_argument('--output_image', help='Output equirectangular image')
    parser.add_argument('--fov', type=float, default=195.0, help='Base FOV in degrees')
    parser.add_argument('--frames', type=int, default=5, help='Number of video frames')
    parser.add_argument('--scale', type=float, default=0.25, help='Downsample scale for optimization (0.25=4x faster)')
    parser.add_argument('--restarts', type=int, default=3, help='Number of random restarts to escape local minima')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show overlap computation details')
    parser.add_argument('--fix-lens1', action='store_true', help='Fix lens 1 rotation at 0 (only optimize lens 2 rotation)')
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
    
    # Initial calibration
    initial_calib = CameraCalibration(
        lens1=LensCalibration(),
        lens2=LensCalibration(),
        is_horizontal=True
    )
    
    # Run calibration on downsampled frames
    final_calib = calibrate_stepwise(
        frames_small, initial_calib, args.fov, 
        verbose=args.verbose, n_restarts=args.restarts, fix_lens1=args.fix_lens1
    )
    
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
    
    # Generate output image with seams
    if args.output_image:
        print(f"\nGenerating output image...")
        left_img, right_img = frames_data[0]
        h, w = left_img.shape[:2]
        
        # Use fisheye_to_equirect pipeline
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
        
        # Calculate overlap
        fov_eff = args.fov * final_calib.lens1.fov
        overlap_deg = (fov_eff - 180.0) / 2.0
        overlap_px = max(1, int(overlap_deg * half_w / args.fov))
        
        result = blend_dual_patches(left_patch, left_mask, right_patch, right_mask, overlap_px)
        cv2.imwrite(args.output_image, result)
        print(f"Saved to: {args.output_image}")


if __name__ == '__main__':
    main()
