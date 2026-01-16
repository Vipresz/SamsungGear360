#!/usr/bin/env python3
"""
Calibrate stitching parameters for dual-fisheye 360° camera from MP4 video file.
Projects dual fisheye lenses (195° FOV each) to equirectangular format.

This is the main entry point that imports from modular components:
- lens_detection: Lens boundary/center detection
- alignment: Alignment and seam error computation
- fov_optimization: FOV and distortion optimization
- rotation_estimation: Rotation estimation from feature matching
- projection: Fisheye to equirectangular projection
"""

import argparse
import cv2
import numpy as np
import json
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict

# Import from modular components
from lens_detection import (
    detect_lens_boundary_points,
    fit_circle_ransac,
    detect_lens_center_advanced,
    detect_lens_center
)

from alignment import (
    extract_overlap_features,
    compute_seam_alignment_error,
    optimize_alignment_parameters
)

from fov_optimization import (
    estimate_fov_from_coverage,
    optimize_fov,
    optimize_distortion
)

from rotation_estimation import (
    extract_ring_region,
    estimate_lens_rotation_from_rings,
    estimate_lens_rotation,
    compute_alignment_offsets_equirect
)

from projection import (
    fisheye_to_equirect_half,
    fisheye_to_equirect_dual,
    project_fisheye_to_equirectangular,
    apply_calibration_to_video
)

from tracking import (
    FeatureTracker,
    estimate_rotation_from_tracks,
    estimate_distortion_from_tracks,
    estimate_lens_center_from_tracks
)

from seam_refinement import (
    refine_fov_from_seam,
    optimize_y_offset,
    optimize_offsets,
    refine_roll_from_seam,
    refine_rotation_from_seam,
    refine_rotation_both_lenses
)

try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available. Some optimization features will be limited.", file=sys.stderr)


LENS_FOV_DEG = 195


def analyze_frame(frame: np.ndarray, is_horizontal: bool, optimize: bool = True) -> Optional[Dict]:
    """Analyze a single frame to extract stitching parameters with sophisticated detection."""
    height, width = frame.shape[:2]
    
    if is_horizontal:
        lens_w = width // 2
        lens_h = height
        lens1_region = (0, 0, lens_w, lens_h)
        lens2_region = (lens_w, 0, lens_w, lens_h)
    else:
        lens_w = width
        lens_h = height // 2
        lens1_region = (0, 0, lens_w, lens_h)
        lens2_region = (0, lens_h, lens_w, lens_h)
    
    min_dim = min(lens_w, lens_h)
    default_radius = min_dim * 0.48
    
    detection1 = detect_lens_center_advanced(frame, lens1_region)
    detection2 = detect_lens_center_advanced(frame, lens2_region)
    
    # Process lens 1 detection
    if detection1 is None:
        center1_x, center1_y, radius1 = 0.5, 0.5, default_radius
        circularity1 = 0.0
    else:
        center1_x = detection1['center_x']
        center1_y = detection1['center_y']
        radius1 = detection1['radius']
        circularity1 = detection1.get('circularity', 0.0)
        
        expected_radius = min_dim * 0.5
        radius_error = abs(radius1 - expected_radius) / expected_radius
        center_error = abs(center1_x - 0.5) + abs(center1_y - 0.5)
        
        if circularity1 < 0.55 or radius_error > 0.25 or center_error > 0.2:
            print(f"  Lens 1 detection unreliable (circularity={circularity1:.2f})")
            center1_x, center1_y, radius1 = 0.5, 0.5, default_radius
            circularity1 = 0.0
        else:
            print(f"  Lens 1: center=({center1_x:.4f}, {center1_y:.4f}), "
                  f"radius={radius1:.1f}, circularity={circularity1:.3f}")
    
    # Process lens 2 detection
    if detection2 is None:
        center2_x, center2_y, radius2 = 0.5, 0.5, default_radius
        circularity2 = 0.0
    else:
        center2_x = detection2['center_x']
        center2_y = detection2['center_y']
        radius2 = detection2['radius']
        circularity2 = detection2.get('circularity', 0.0)
        
        expected_radius = min_dim * 0.5
        radius_error = abs(radius2 - expected_radius) / expected_radius
        center_error = abs(center2_x - 0.5) + abs(center2_y - 0.5)
        
        if circularity2 < 0.55 or radius_error > 0.25 or center_error > 0.2:
            print(f"  Lens 2 detection unreliable (circularity={circularity2:.2f})")
            center2_x, center2_y, radius2 = 0.5, 0.5, default_radius
            circularity2 = 0.0
        else:
            print(f"  Lens 2: center=({center2_x:.4f}, {center2_y:.4f}), "
                  f"radius={radius2:.1f}, circularity={circularity2:.3f}")
    
    avg_radius = (radius1 + radius2) / 2
    
    params = {
        'lens1CenterX': float(center1_x),
        'lens1CenterY': float(center1_y),
        'lens2CenterX': float(center2_x),
        'lens2CenterY': float(center2_y),
        'lensRadius': float(avg_radius),
        'lens1Radius': float(radius1),
        'lens2Radius': float(radius2),
        'lens1Circularity': float(circularity1),
        'lens2Circularity': float(circularity2),
        'innerRadiusRatio': 0.65,
        'alignmentOffset1X': 0.0,
        'alignmentOffset1Y': 0.0,
        'alignmentOffset2X': 0.0,
        'alignmentOffset2Y': 0.0,
        'lensRotationYaw': 0.0,
        'lensRotationPitch': 0.0,
        'lensRotationRoll': 0.0,
        'lens1FOV': 1.0,
        'lens2FOV': 1.0,
        'lensFOV': 1.0,
        'lens1P1': 0.0, 'lens1P2': 0.0, 'lens1P3': 0.0, 'lens1P4': 0.0,
        'lens2P1': 0.0, 'lens2P2': 0.0, 'lens2P3': 0.0, 'lens2P4': 0.0,
        'isHorizontal': is_horizontal,
        'lensWidth': lens_w,
        'lensHeight': lens_h
    }
    
    if optimize:
        print("  Optimizing FOV...")
        fov1, fov2 = optimize_fov(frame, params, fisheye_to_equirect_dual)
        params['lens1FOV'] = fov1
        params['lens2FOV'] = fov2
        params['lensFOV'] = (fov1 + fov2) / 2
        print(f"  Optimized FOV: lens1={fov1:.4f}, lens2={fov2:.4f}")
    
    return params


def calibrate_from_video_tracking(video_path: str, num_frames: Optional[int] = None,
                                   skip_frames: int = 2) -> Dict:
    """Calibrate stitching parameters using feature tracking across video frames.
    
    PROCEDURAL CALIBRATION STEPS:
    =============================
    Step 0: Detect lens centers (geometric)
    Step 1: Adjust ABERRATION by tracking feature motion patterns
    Step 2: Adjust FOV (first pass)
    Step 3: Adjust ROLL
    Step 4: Adjust FOV (second pass, starting from step 2 result)
    
    This sequence ensures:
    - Aberration is corrected first (affects all subsequent measurements)
    - FOV is roughly set before roll correction
    - Roll is corrected with good FOV baseline
    - FOV is fine-tuned after roll correction
    """
    print(f"\n{'='*60}")
    print("PROCEDURAL VIDEO CALIBRATION")
    print(f"{'='*60}")
    print("\nCalibration sequence:")
    print("  Step 0: Detect lens centers")
    print("  Step 1: Estimate aberration from tracking")
    print("  Step 2: Adjust FOV (first pass)")
    print("  Step 3: Adjust roll")
    print("  Step 4: Adjust FOV (final pass)")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    is_horizontal = width > height
    lens_w = width // 2 if is_horizontal else width
    lens_h = height if is_horizontal else height // 2
    
    print(f"\nVideo: {width}x{height}, {total_frames} frames @ {fps:.1f} fps")
    print(f"Lens dimensions: {lens_w}x{lens_h}")
    
    if num_frames is None:
        num_frames = min(total_frames, 300)
    
    # =========================================================================
    # STEP 0: DETECT LENS CENTERS (Geometric)
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 0: DETECT LENS CENTERS")
    print(f"{'='*60}")
    
    cap_temp = cv2.VideoCapture(video_path)
    ret, first_frame = cap_temp.read()
    cap_temp.release()
    
    if not ret:
        raise ValueError("Failed to read first frame from video")
    
    if is_horizontal:
        lens1_region = (0, 0, lens_w, lens_h)
        lens2_region = (lens_w, 0, lens_w, lens_h)
    else:
        lens1_region = (0, 0, lens_w, lens_h)
        lens2_region = (0, lens_h, lens_w, lens_h)

    detection1 = detect_lens_center_advanced(first_frame, lens1_region)
    detection2 = detect_lens_center_advanced(first_frame, lens2_region)
    
    if detection1 is not None:
        center1_x = detection1['center_x']
        center1_y = detection1['center_y']
        radius1 = detection1['radius']
        print(f"  Lens 1: center=({center1_x:.4f}, {center1_y:.4f}), radius={radius1:.1f}px")
    else:
        center1_x, center1_y = 0.5, 0.5
        radius1 = min(lens_w, lens_h) * 0.48
        print(f"  Lens 1: using defaults center=(0.5, 0.5)")
        
    if detection2 is not None:
        center2_x = detection2['center_x']
        center2_y = detection2['center_y']
        radius2 = detection2['radius']
        print(f"  Lens 2: center=({center2_x:.4f}, {center2_y:.4f}), radius={radius2:.1f}px")
    else:
        center2_x, center2_y = 0.5, 0.5
        radius2 = min(lens_w, lens_h) * 0.48
        print(f"  Lens 2: using defaults center=(0.5, 0.5)")
    
    # =========================================================================
    # STEP 1: ESTIMATE ABERRATION FROM TRACKING
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 1: ESTIMATE ABERRATION FROM TRACKING")
    print(f"{'='*60}")
    
    # Initialize tracker
    tracker = FeatureTracker(max_features=500, quality_level=0.01, min_distance=15)
    
    print(f"\nTracking features across {num_frames} frames...")
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    processed = 0
    
    while processed < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % skip_frames == 0:
            stats = tracker.process_frame(frame, processed)
            processed += 1
            
            if processed % 50 == 0:
                print(f"  Frame {processed}/{num_frames}: {stats['num_tracks']} active tracks")
        
        frame_idx += 1
    
    cap.release()
    
    print(f"\nTracking complete: {len(tracker.tracks)} total tracks")
    
    # Get long tracks for analysis
    long_tracks = tracker.get_long_tracks(min_length=10)
    print(f"Long tracks (10+ frames): {len(long_tracks)}")
    
    # Estimate distortion from track curvature
    print("\nEstimating distortion coefficients from track motion patterns...")
    print("  (Comparing inner vs outer feature radial motion)")
    
    p1_1, p2_1 = estimate_distortion_from_tracks(long_tracks, lens_w, lens_h, center1_x, center1_y, 0)
    p1_2, p2_2 = estimate_distortion_from_tracks(long_tracks, lens_w, lens_h, center2_x, center2_y, 1)
    
    print(f"\n  Lens 1 aberration: p1={p1_1:.4f}, p2={p2_1:.4f}")
    print(f"  Lens 2 aberration: p1={p1_2:.4f}, p2={p2_2:.4f}")
    
    # Get initial roll estimate from tracking
    print("\nEstimating initial rotation from motion patterns...")
    yaw, pitch, initial_roll = estimate_rotation_from_tracks(long_tracks, lens_w, lens_h)
    print(f"  Initial roll estimate: {np.degrees(initial_roll):.2f}°")
    
    # =========================================================================
    # STEP 2: ADJUST FOV (First Pass)
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 2: ADJUST FOV (First Pass)")
    print(f"{'='*60}")
    
    # Start with nominal FOV based on lens specification (195°)
    # FOV scale = actual_fov / 180° = 195 / 180 = 1.0833
    initial_fov_scale = LENS_FOV_DEG / 180.0  # 1.0833 for 195° lens
    fov1, fov2 = initial_fov_scale, initial_fov_scale
    roll = 0.0  # No roll applied yet
    y_offset = 0.0
    
    print(f"Starting FOV scale: {initial_fov_scale:.4f} (lens has {LENS_FOV_DEG}°, scale 1.0 = 180°)")
    
    print(f"\nRefining FOV from seam alignment (starting at {initial_fov_scale:.4f})...")
    print(f"  (No rotation applied in first pass)")
    fov1, fov2 = refine_fov_from_seam(
        first_frame, 
        center1_x, center1_y, 
        center2_x, center2_y,
        p2_1, p2_2,
        is_horizontal,
        project_fisheye_to_equirectangular,
        initial_fov1=fov1,
        initial_fov2=fov2,
        yaw=0.0, pitch=0.0, roll=0.0  # No rotation in first pass
    )
    
    print(f"\n  FOV after first pass: L1={fov1:.4f}, L2={fov2:.4f}")
    print(f"  (Equivalent to: L1={fov1 * 180:.1f}°, L2={fov2 * 180:.1f}°)")
    
    # =========================================================================
    # STEP 3: ADJUST ROTATION AND OFFSETS (BOTH LENSES)
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 3: ADJUST ROTATION AND OFFSETS (BOTH LENSES)")
    print(f"{'='*60}")
    
    # Optimize rotation for both lenses
    print(f"\nRefining rotation for both lenses to minimize seam error...")
    print(f"  (Tracking suggested roll={np.degrees(initial_roll):.2f}°)")
    
    (yaw1, pitch1, roll1), (yaw2, pitch2, roll2) = refine_rotation_both_lenses(
        first_frame,
        center1_x, center1_y,
        center2_x, center2_y,
        fov1, fov2,
        p2_1, p2_2,
        is_horizontal,
        project_fisheye_to_equirectangular
    )
    
    print(f"\n  Rotation after refinement:")
    print(f"    Lens 1: yaw={np.degrees(yaw1):.2f}°, pitch={np.degrees(pitch1):.2f}°, roll={np.degrees(roll1):.2f}°")
    print(f"    Lens 2: yaw={np.degrees(yaw2):.2f}°, pitch={np.degrees(pitch2):.2f}°, roll={np.degrees(roll2):.2f}°")
    
    # Optimize X and Y offsets for both lenses
    print("\nOptimizing X/Y offsets for both lenses...")
    x_off1, y_off1, x_off2, y_off2 = optimize_offsets(
        first_frame,
        center1_x, center1_y,
        center2_x, center2_y,
        fov1, fov2,
        p2_1, p2_2,
        roll1, roll2,
        is_horizontal,
        project_fisheye_to_equirectangular
    )
    
    print(f"  Lens 1 offsets: X={x_off1:.4f}, Y={y_off1:.4f}")
    print(f"  Lens 2 offsets: X={x_off2:.4f}, Y={y_off2:.4f}")
    
    # Legacy y_offset for backward compatibility
    y_offset = y_off2 - y_off1  # Relative offset between lenses
    
    # =========================================================================
    # STEP 4: ADJUST FOV (Final Pass)
    # =========================================================================
    print(f"\n{'='*60}")
    print("STEP 4: ADJUST FOV (Final Pass)")
    print(f"{'='*60}")
    
    print(f"\nRefining FOV with rotation and Y-offset applied (starting from {fov1:.4f}, {fov2:.4f})...")
    print(f"  Using rotation: yaw={np.degrees(yaw):.2f}°, pitch={np.degrees(pitch):.2f}°, roll={np.degrees(roll):.2f}°")
    
    # Note: Pass center2_y with y_offset applied for accurate seam comparison
    # Also apply the optimized rotation from STEP 3
    fov1_final, fov2_final = refine_fov_from_seam(
        first_frame, 
        center1_x, center1_y, 
        center2_x, center2_y + y_offset,  # Apply Y-offset
        p2_1, p2_2,
        is_horizontal,
        project_fisheye_to_equirectangular,
        initial_fov1=fov1,  # Start from step 2 result
        initial_fov2=fov2,
        yaw=yaw,    # Apply rotation from step 3
        pitch=pitch,
        roll=roll
    )
    
    fov1, fov2 = fov1_final, fov2_final
    
    print(f"\n  Final FOV: L1={fov1:.4f}, L2={fov2:.4f}")
    print(f"  (Equivalent to: L1={fov1 * 180:.1f}°, L2={fov2 * 180:.1f}°)")
    
    # =========================================================================
    # FINAL RESULTS
    # =========================================================================
    print(f"\n{'='*60}")
    print("CALIBRATION COMPLETE")
    print(f"{'='*60}")
    
    avg_radius = (radius1 + radius2) / 2 if detection1 and detection2 else min(lens_w, lens_h) * 0.48
    
    params = {
        # Lens centers
        'lens1CenterX': float(center1_x),
        'lens1CenterY': float(center1_y),
        'lens2CenterX': float(center2_x),
        'lens2CenterY': float(center2_y),
        
        # Lens radii
        'lensRadius': float(avg_radius),
        'lens1Radius': float(radius1),
        'lens2Radius': float(radius2),
        'innerRadiusRatio': 0.85,
        
        # Layout
        'isHorizontal': is_horizontal,
        'lensWidth': int(lens_w),
        'lensHeight': int(lens_h),
        
        # FOV (from steps 2 & 4)
        'lensFOV': float((fov1 + fov2) / 2),
        'lens1FOV': float(fov1),
        'lens2FOV': float(fov2),
        
        # Aberration/distortion (from step 1)
        'lens1P1': float(p1_1),
        'lens1P2': float(p2_1),
        'lens1P3': 0.0,
        'lens1P4': 0.0,
        'lens2P1': float(p1_2),
        'lens2P2': float(p2_2),
        'lens2P3': 0.0,
        'lens2P4': 0.0,
        
        # Per-lens alignment offsets (from step 3)
        'alignmentOffset1X': float(x_off1),
        'alignmentOffset1Y': float(y_off1),
        'alignmentOffset2X': float(x_off2),
        'alignmentOffset2Y': float(y_off2),
        
        # Per-lens rotation (from step 3)
        'lens1RotationYaw': float(yaw1),
        'lens1RotationPitch': float(pitch1),
        'lens1RotationRoll': float(roll1),
        'lens2RotationYaw': float(yaw2),
        'lens2RotationPitch': float(pitch2),
        'lens2RotationRoll': float(roll2),
        
        # Legacy single rotation (average for backward compatibility)
        'lensRotationYaw': float((yaw1 + yaw2) / 2),
        'lensRotationPitch': float((pitch1 + pitch2) / 2),
        'lensRotationRoll': float((roll1 + roll2) / 2),
        
        # Metadata
        'calibration_method': 'procedural_tracking_v2',
        'frames_processed': processed,
        'total_tracks': len(tracker.tracks),
        'long_tracks': len(long_tracks),
    }
    
    print(f"\n  SUMMARY:")
    print(f"  --------")
    print(f"  Lens Centers: L1=({center1_x:.4f}, {center1_y:.4f}), L2=({center2_x:.4f}, {center2_y:.4f})")
    print(f"  Aberration:   L1=(p1={p1_1:.4f}, p2={p2_1:.4f}), L2=(p1={p1_2:.4f}, p2={p2_2:.4f})")
    print(f"  FOV:          L1={fov1:.4f} ({fov1*180:.1f}°), L2={fov2:.4f} ({fov2*180:.1f}°)")
    print(f"  Rotation L1:  yaw={np.degrees(yaw1):.2f}°, pitch={np.degrees(pitch1):.2f}°, roll={np.degrees(roll1):.2f}°")
    print(f"  Rotation L2:  yaw={np.degrees(yaw2):.2f}°, pitch={np.degrees(pitch2):.2f}°, roll={np.degrees(roll2):.2f}°")
    print(f"  Offsets L1:   X={x_off1:.4f}, Y={y_off1:.4f}")
    print(f"  Offsets L2:   X={x_off2:.4f}, Y={y_off2:.4f}")
    print(f"\n  Tracking stats: {processed} frames, {len(long_tracks)} long tracks")
    
    return params


def calibrate_from_video(video_path: str, num_frames: Optional[int] = None) -> Dict:
    """Calibrate stitching parameters from video by analyzing multiple frames."""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    is_horizontal = width > height
    
    if num_frames is None:
        num_frames = min(10, total_frames)
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    all_params = []
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        print(f"\nAnalyzing frame {i+1}/{len(frame_indices)} (frame #{frame_idx})...")
        params = analyze_frame(frame, is_horizontal, optimize=(i == 0))
        
        if params:
            all_params.append(params)
    
    cap.release()
    
    if not all_params:
        raise ValueError("Failed to analyze any frames")
    
    # Average parameters
    avg_params = {}
    stats = {}
    
    for key in all_params[0].keys():
        if isinstance(all_params[0][key], (int, float)) and not isinstance(all_params[0][key], bool):
            values = [p[key] for p in all_params]
            avg_params[key] = float(np.mean(values))
            stats[f'{key}_std'] = float(np.std(values))
        else:
            avg_params[key] = all_params[0][key]
    
    return {
        'parameters': avg_params,
        'statistics': stats,
        'frames_analyzed': len(all_params),
        'total_frames': total_frames
    }


def main():
    parser = argparse.ArgumentParser(
        description='Calibrate stitching parameters from dual-fisheye video/image (195° FOV per lens)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Video input
  python calibrate_stitching.py --video video.mp4
  python calibrate_stitching.py --video video.mp4 --output params.json
  python calibrate_stitching.py --video video.mp4 --output_image preview.png --apply_calibration
  
  # Image input
  python calibrate_stitching.py --image fisheye.jpg --output_image equirect.png
  python calibrate_stitching.py --image fisheye.jpg --output_image equirect.png --apply_calibration
        """
    )
    
    parser.add_argument('--video', type=str, default=None, help='Path to MP4 video file')
    parser.add_argument('--image', type=str, default=None, help='Path to input image file')
    parser.add_argument('--tracking', action='store_true',
                       help='Use feature tracking calibration (requires video, more accurate)')
    parser.add_argument('--frames', type=int, default=None, help='Number of frames to analyze')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')
    parser.add_argument('--output_video', type=str, default=None, help='Output equirectangular video path')
    parser.add_argument('--output_width', type=int, default=0, help='Output width')
    parser.add_argument('--output_height', type=int, default=0, help='Output height')
    parser.add_argument('--fov', type=float, default=180.0, help='Vertical FOV in degrees')
    parser.add_argument('--calibration_file', type=str, default=None, help='Load calibration from JSON')
    parser.add_argument('--output_image', type=str, default=None, help='Output image file')
    parser.add_argument('--apply_calibration', action='store_true', help='Apply calibration parameters')
    
    args = parser.parse_args()
    
    if args.video is None and args.image is None:
        print("Error: Must specify either --video or --image", file=sys.stderr)
        parser.print_help()
        sys.exit(1)
    
    if args.video is not None and args.image is not None:
        print("Error: Cannot specify both --video and --image", file=sys.stderr)
        sys.exit(1)
    
    is_image_input = args.image is not None
    input_path = Path(args.image if is_image_input else args.video)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    
    try:
        if is_image_input:
            input_frame = cv2.imread(str(input_path))
            if input_frame is None:
                raise ValueError(f"Failed to read image: {input_path}")
            input_height, input_width = input_frame.shape[:2]
            print(f"Input image: {input_path.name} ({input_width}x{input_height})")
        else:
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {input_path}")
            input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            print(f"Input video: {input_path.name} ({input_width}x{input_height})")
        
        if args.output_width == 0 or args.output_height == 0:
            args.output_width = input_width
            args.output_height = input_height
        
        params = {}
        apply_calibration = args.apply_calibration
        
        if apply_calibration:
            if args.calibration_file:
                with open(args.calibration_file, 'r') as f:
                    calib_data = json.load(f)
                params = calib_data['parameters']
                print(f"Loaded calibration from: {args.calibration_file}")
            elif is_image_input:
                print(f"\nCalibrating from image...")
                print("=" * 60)
                is_horizontal = input_width > input_height
                params = analyze_frame(input_frame, is_horizontal, optimize=True)
                if params:
                    print("\n" + "=" * 60)
                    print("CALIBRATION RESULTS")
                    print("=" * 60)
                    print(f"\nLens Centers: L1=({params['lens1CenterX']:.4f}, {params['lens1CenterY']:.4f}), "
                          f"L2=({params['lens2CenterX']:.4f}, {params['lens2CenterY']:.4f})")
                    print(f"Lens FOV: L1={params['lens1FOV']:.4f}, L2={params['lens2FOV']:.4f}")
                else:
                    print("Warning: Could not detect lens parameters. Using defaults.")
                    apply_calibration = False
            else:
                print(f"\nCalibrating from video...")
                print("=" * 60)
                num_frames = args.frames if args.frames else (100 if args.tracking else 10)
                
                if args.tracking:
                    # Use feature tracking calibration (more accurate)
                    params = calibrate_from_video_tracking(str(input_path), num_frames)
                    result = {'parameters': params, 'statistics': {
                        'lens1CenterX_std': 0.0, 'lens1CenterY_std': 0.0,
                        'lens2CenterX_std': 0.0, 'lens2CenterY_std': 0.0,
                        'lensRadius_std': 0.0
                    }, 'frames_analyzed': params.get('frames_processed', 0),
                       'total_frames': params.get('frames_processed', 0)}
                else:
                    # Use frame-by-frame analysis
                    result = calibrate_from_video(str(input_path), num_frames)
                    params = result['parameters']
                
                print("\n" + "=" * 60)
                print("CALIBRATION RESULTS")
                print("=" * 60)
                print(f"\nLens Centers: L1=({params['lens1CenterX']:.4f}, {params['lens1CenterY']:.4f}), "
                      f"L2=({params['lens2CenterX']:.4f}, {params['lens2CenterY']:.4f})")
                print(f"Lens FOV: L1={params.get('lens1FOV', 1.0):.4f}, L2={params.get('lens2FOV', 1.0):.4f}")
                
                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(result, f, indent=2)
                    print(f"\nCalibration saved to: {args.output}")
        
        if args.output_image:
            print(f"\nGenerating equirectangular image...")
            
            if is_image_input:
                source_frame = input_frame
            else:
                cap = cv2.VideoCapture(str(input_path))
                ret, source_frame = cap.read()
                cap.release()
                if not ret:
                    raise ValueError("Could not read video frame")
            
            equirect_frame = project_fisheye_to_equirectangular(
                source_frame, params,
                args.output_width, args.output_height, args.fov,
                apply_calibration=apply_calibration
            )
            cv2.imwrite(args.output_image, equirect_frame)
            print(f"Saved to: {args.output_image}")
        
        if args.output_video:
            if is_image_input:
                print("Error: Cannot generate video from image input", file=sys.stderr)
            else:
                apply_calibration_to_video(
                    str(input_path), args.output_video,
                    params, args.output_width, args.output_height, args.fov,
                    apply_calibration=apply_calibration
                )
    
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
