"""
Fisheye to equirectangular projection functions.
Includes single-lens and dual-lens projection with calibration support.
"""

import cv2
import numpy as np
from typing import Tuple, Dict


def fisheye_to_equirect_half(fisheye_img: np.ndarray, fov: float = 1.0, 
                              offset: float = np.pi/2, x_offset: float = 0, y_offset: float = 0,
                              center_x: float = 0.5, center_y: float = 0.5,
                              p1: float = 0.0, p2: float = 0.0, p3: float = 0.0, p4: float = 0.0,
                              rotation: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """Convert one fisheye image to equirectangular half.
    Based on reference implementation fisheye_to_eq_rect_FO with polynomial distortion.
    
    Applies calibration in order: rotation -> offsets -> distortion -> FOV
    
    Uses the f-theta model with polynomial correction:
    r = f * θ * (1 + p1*θ + p2*θ² + p3*θ³ + p4*θ⁴)
    
    Args:
        fisheye_img: Input fisheye image
        fov: Field of view scaling (1.0 = full circle)
        offset: Longitude offset for this lens
        x_offset: X alignment offset in pixels
        y_offset: Y alignment offset in pixels
        center_x: Lens center X (0-1 normalized)
        center_y: Lens center Y (0-1 normalized)
        p1, p2, p3, p4: Polynomial distortion coefficients
        rotation: (yaw, pitch, roll) rotation correction in radians
    """
    h, w = fisheye_img.shape[:2]
    Rad = h // 2
    Dia = Rad * 2
    
    cx = center_x * w
    cy = center_y * h
    
    R, C = np.mgrid[0:Dia, 0:Dia]
    
    Y = 2.0 * (R / Dia - 0.5)
    X = 2.0 * (0.5 - C / Dia)
    
    lon = X * np.pi / 2 + offset
    lat = Y * np.pi / 2
    
    # Convert to 3D direction vectors on unit sphere
    x_sphere = np.cos(lat) * np.cos(lon)
    y_sphere = np.cos(lat) * np.sin(lon)
    z_sphere = np.sin(lat)
    
    # Step 1: Apply rotation correction (yaw, pitch, roll)
    yaw, pitch, roll = rotation
    if yaw != 0 or pitch != 0 or roll != 0:
        # Build rotation matrices
        # Roll (rotation around Y axis - the lens forward direction)
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        # Pitch (rotation around X axis)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        # Yaw (rotation around Z axis)
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        
        # Combined rotation: R = Rz(yaw) * Rx(pitch) * Ry(roll)
        # Apply to each point
        # First roll (around Y)
        x_rot = cos_r * x_sphere + sin_r * z_sphere
        z_rot = -sin_r * x_sphere + cos_r * z_sphere
        x_sphere, z_sphere = x_rot, z_rot
        
        # Then pitch (around X)
        y_rot = cos_p * y_sphere - sin_p * z_sphere
        z_rot = sin_p * y_sphere + cos_p * z_sphere
        y_sphere, z_sphere = y_rot, z_rot
        
        # Then yaw (around Z)
        x_rot = cos_y * x_sphere - sin_y * y_sphere
        y_rot = sin_y * x_sphere + cos_y * y_sphere
        x_sphere, y_sphere = x_rot, y_rot
    
    # Convert rotated sphere coords to theta/phi for fisheye sampling
    theta = np.arctan2(np.sqrt(x_sphere**2 + z_sphere**2), y_sphere)
    phi = np.arctan2(z_sphere, x_sphere)
    
    # Normalized theta (0 at center, 1 at edge for 180° lens)
    theta_norm = theta / (np.pi / 2)
    
    # Step 2: Apply polynomial distortion: r = f*θ*(1 + p1*θ + p2*θ² + p3*θ³ + p4*θ⁴)
    distortion = 1.0 + p1*theta_norm + p2*(theta_norm**2) + p3*(theta_norm**3) + p4*(theta_norm**4)
    r_f = theta * 2.0 / np.pi * distortion
    
    # Step 3: Apply FOV scaling
    u = r_f * np.cos(phi) / fov
    v = r_f * np.sin(phi) / fov
    
    # Step 4: Apply alignment offsets (in pixels)
    x_fish = (cx + u * Rad + x_offset).astype(np.int32)
    y_fish = (cy + v * Rad + y_offset).astype(np.int32)
    
    x_fish = np.clip(x_fish, 0, w - 1)
    y_fish = np.clip(y_fish, 0, h - 1)
    
    if len(fisheye_img.shape) > 2:
        img_out = np.zeros((Dia, Dia, 3), dtype=fisheye_img.dtype)
        img_out[R, C, :] = fisheye_img[y_fish, x_fish, :]
    else:
        img_out = np.zeros((Dia, Dia), dtype=fisheye_img.dtype)
        img_out[R, C] = fisheye_img[y_fish, x_fish]
    
    return img_out


def fisheye_to_equirect_dual(dual_fisheye_img: np.ndarray, fov: Tuple[float, float] = (1.0, 1.0),
                              offset: float = np.pi/2, 
                              x_offset1: float = 0, y_offset1: float = 0,
                              x_offset2: float = 0, y_offset2: float = 0,
                              center1_x: float = 0.5, center1_y: float = 0.5,
                              center2_x: float = 0.5, center2_y: float = 0.5,
                              distortion1: Tuple[float, float, float, float] = (0, 0, 0, 0),
                              distortion2: Tuple[float, float, float, float] = (0, 0, 0, 0),
                              rotation1: Tuple[float, float, float] = (0, 0, 0),
                              rotation2: Tuple[float, float, float] = (0, 0, 0),
                              rotation: Tuple[float, float, float] = None) -> np.ndarray:
    """Convert dual fisheye (horizontal layout) to equirectangular.
    Based on reference implementation fisheye_to_eq_rect_2lens_FO with polynomial distortion.
    
    Applies calibration in order: rotation -> offsets -> distortion -> FOV
    
    Args:
        dual_fisheye_img: Dual fisheye input (horizontal layout)
        fov: FOV scaling for each lens (1.0 = full circle)
        offset: Longitude offset
        x_offset1/y_offset1: Alignment offsets for left lens in pixels
        x_offset2/y_offset2: Alignment offsets for right lens in pixels
        center1_x/center1_y: Left lens center (0-1 normalized)
        center2_x/center2_y: Right lens center (0-1 normalized)
        distortion1/distortion2: Polynomial coefficients (p1, p2, p3, p4) for each lens
        rotation1: (yaw, pitch, roll) rotation for left lens in radians
        rotation2: (yaw, pitch, roll) rotation for right lens in radians
        rotation: (deprecated) single rotation for right lens (for backward compatibility)
    """
    # Backward compatibility: if rotation is provided but rotation2 is default, use rotation
    if rotation is not None and rotation2 == (0, 0, 0):
        rotation2 = rotation
    
    h, w = dual_fisheye_img.shape[:2]
    Rad = h // 2
    Dia = Rad * 2
    
    if len(dual_fisheye_img.shape) > 2:
        img_out = np.zeros((Dia, Dia * 2, 3), dtype=dual_fisheye_img.dtype)
    else:
        img_out = np.zeros((Dia, Dia * 2), dtype=dual_fisheye_img.dtype)
    
    left_fisheye = dual_fisheye_img[:, :Dia, ...]
    right_fisheye = dual_fisheye_img[:, Dia:, ...]
    
    p1_1, p2_1, p3_1, p4_1 = distortion1
    p1_2, p2_2, p3_2, p4_2 = distortion2
    
    # Left lens: apply per-lens rotation
    left_half = fisheye_to_equirect_half(left_fisheye, fov=fov[0], offset=offset,
                                          x_offset=x_offset1, y_offset=y_offset1,
                                          center_x=center1_x, center_y=center1_y,
                                          p1=p1_1, p2=p2_1, p3=p3_1, p4=p4_1,
                                          rotation=rotation1)
    
    # Right lens: apply per-lens rotation
    right_fisheye_flipped = np.fliplr(right_fisheye)
    right_half = fisheye_to_equirect_half(right_fisheye_flipped, fov=fov[1], offset=offset,
                                           x_offset=-x_offset2, y_offset=y_offset2,
                                           center_x=1.0 - center2_x, center_y=center2_y,
                                           p1=p1_2, p2=p2_2, p3=p3_2, p4=p4_2,
                                           rotation=rotation2)
    right_half_flipped = np.fliplr(right_half)
    
    img_out[:, :Dia, ...] = left_half
    img_out[:, Dia:, ...] = right_half_flipped
    
    return img_out


def project_fisheye_to_equirectangular(frame: np.ndarray, params: Dict,
                                       output_width: int = 1920, output_height: int = 960,
                                       fov: float = 180.0, use_both_lenses: bool = True,
                                       apply_calibration: bool = False) -> np.ndarray:
    """Project dual-fisheye frame (195° FOV each) to equirectangular.
    Uses reference implementation for accurate projection.
    
    Args:
        frame: Dual fisheye input frame
        params: Calibration parameters dict
        output_width/output_height: Output resolution
        fov: Field of view for output
        use_both_lenses: Use both lenses (ignored, always True)
        apply_calibration: Whether to apply calibration parameters
    """
    height, width = frame.shape[:2]
    is_horizontal = width > height
    lens_w = width // 2 if is_horizontal else width
    lens_h = height if is_horizontal else height // 2
    
    if apply_calibration and params:
        center1_x = params.get('lens1CenterX', 0.5)
        center1_y = params.get('lens1CenterY', 0.5)
        center2_x = params.get('lens2CenterX', 0.5)
        center2_y = params.get('lens2CenterY', 0.5)
        
        # Alignment offsets: stored but not applied by default
        apply_alignment = params.get('applyAlignment', False)
        if apply_alignment:
            x_offset1 = params.get('alignmentOffset1X', 0.0) * lens_w
            y_offset1 = params.get('alignmentOffset1Y', 0.0) * lens_h
            x_offset2 = params.get('alignmentOffset2X', 0.0) * lens_w
            y_offset2 = params.get('alignmentOffset2Y', 0.0) * lens_h
        else:
            x_offset1 = y_offset1 = x_offset2 = y_offset2 = 0.0
        
        fov1 = params.get('lens1FOV', 1.0)
        fov2 = params.get('lens2FOV', 1.0)
        
        # Polynomial distortion coefficients
        distortion1 = (
            params.get('lens1P1', 0.0),
            params.get('lens1P2', 0.0),
            params.get('lens1P3', 0.0),
            params.get('lens1P4', 0.0)
        )
        distortion2 = (
            params.get('lens2P1', 0.0),
            params.get('lens2P2', 0.0),
            params.get('lens2P3', 0.0),
            params.get('lens2P4', 0.0)
        )
        
        # Rotation correction - per-lens or single (backward compatible)
        apply_rotation = params.get('applyRotation', False)
        max_rotation_deg = 10.0  # Allow larger rotations for per-lens optimization
        max_rotation_rad = np.radians(max_rotation_deg)
        
        # Check for per-lens rotations first
        yaw1 = params.get('lens1RotationYaw', 0.0)
        pitch1 = params.get('lens1RotationPitch', 0.0)
        roll1 = params.get('lens1RotationRoll', 0.0)
        yaw2 = params.get('lens2RotationYaw', 0.0)
        pitch2 = params.get('lens2RotationPitch', 0.0)
        roll2 = params.get('lens2RotationRoll', 0.0)
        
        # Fall back to single rotation for backward compatibility
        if yaw1 == 0 and pitch1 == 0 and roll1 == 0 and yaw2 == 0 and pitch2 == 0 and roll2 == 0:
            yaw = params.get('lensRotationYaw', 0.0)
            pitch = params.get('lensRotationPitch', 0.0)
            roll = params.get('lensRotationRoll', 0.0)
            yaw1, pitch1, roll1 = 0.0, 0.0, 0.0  # Left lens no rotation in legacy mode
            yaw2, pitch2, roll2 = yaw, pitch, roll  # Right lens gets rotation
        
        if apply_rotation:
            rotation1 = (
                yaw1 if abs(yaw1) < max_rotation_rad else 0.0,
                pitch1 if abs(pitch1) < max_rotation_rad else 0.0,
                roll1 if abs(roll1) < max_rotation_rad else 0.0
            )
            rotation2 = (
                yaw2 if abs(yaw2) < max_rotation_rad else 0.0,
                pitch2 if abs(pitch2) < max_rotation_rad else 0.0,
                roll2 if abs(roll2) < max_rotation_rad else 0.0
            )
        else:
            rotation1 = (0.0, 0.0, 0.0)
            rotation2 = (0.0, 0.0, 0.0)
    else:
        center1_x = center1_y = center2_x = center2_y = 0.5
        x_offset1 = y_offset1 = x_offset2 = y_offset2 = 0.0
        fov1 = fov2 = 1.0
        distortion1 = distortion2 = (0.0, 0.0, 0.0, 0.0)
        rotation1 = rotation2 = (0.0, 0.0, 0.0)
    
    if is_horizontal:
        equirect = fisheye_to_equirect_dual(
            frame, fov=(fov1, fov2), offset=np.pi/2,
            x_offset1=x_offset1, y_offset1=y_offset1,
            x_offset2=x_offset2, y_offset2=y_offset2,
            center1_x=center1_x, center1_y=center1_y,
            center2_x=center2_x, center2_y=center2_y,
            distortion1=distortion1, distortion2=distortion2,
            rotation1=rotation1, rotation2=rotation2
        )
    else:
        frame_rotated = np.rot90(frame)
        equirect = fisheye_to_equirect_dual(
            frame_rotated, fov=(fov1, fov2), offset=np.pi/2,
            x_offset1=x_offset1, y_offset1=y_offset1,
            x_offset2=x_offset2, y_offset2=y_offset2,
            center1_x=center1_x, center1_y=center1_y,
            center2_x=center2_x, center2_y=center2_y,
            distortion1=distortion1, distortion2=distortion2,
            rotation1=rotation1, rotation2=rotation2
        )
    
    if equirect.shape[0] != output_height or equirect.shape[1] != output_width:
        equirect = cv2.resize(equirect, (output_width, output_height), interpolation=cv2.INTER_LINEAR)
    
    return equirect


def apply_calibration_to_video(input_video: str, output_video: str, params: Dict,
                               output_width: int = 1920, output_height: int = 960,
                               fov: float = 180.0, apply_calibration: bool = False):
    """Apply calibration and generate equirectangular video."""
    cap = cv2.VideoCapture(input_video)
    
    if not cap.isOpened():
        raise ValueError(f"Failed to open input video: {input_video}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (output_width, output_height))
    
    if not out.isOpened():
        raise ValueError(f"Failed to create output video: {output_video}")
    
    print(f"\nProcessing video:")
    print(f"  Input: {input_video}")
    print(f"  Output: {output_video}")
    print(f"  Resolution: {output_width}x{output_height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        equirect = project_fisheye_to_equirectangular(
            frame, params, output_width, output_height, fov,
            apply_calibration=apply_calibration
        )
        
        out.write(equirect)
        
        frame_count += 1
        if frame_count % 100 == 0:
            progress = frame_count / total_frames * 100
            print(f"  Progress: {frame_count}/{total_frames} ({progress:.1f}%)")
    
    cap.release()
    out.release()
    
    print(f"  Complete! Processed {frame_count} frames")
