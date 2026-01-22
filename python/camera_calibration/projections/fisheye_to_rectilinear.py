#!/usr/bin/env python3
"""Fisheye to rectilinear (perspective) projection."""

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


def fisheye_to_rectilinear(fisheye_img, out_width, out_height,
                            out_fov_deg=90.0, yaw_deg=0.0, pitch_deg=0.0,
                            fisheye_fov_deg=180.0, cx=None, cy=None, radius=None):
    """Extract a rectilinear view from an equidistant fisheye image."""
    h, w = fisheye_img.shape[:2]
    cx = cx if cx is not None else w / 2.0
    cy = cy if cy is not None else h / 2.0
    radius = radius if radius is not None else min(w, h) / 2.0
    
    out_fov_deg = min(out_fov_deg, 150.0)
    half_fov_h = np.radians(out_fov_deg / 2.0)
    half_fov_v = half_fov_h * out_height / out_width
    
    v_idx, u_idx = np.mgrid[0:out_height, 0:out_width]
    u_norm = (u_idx - (out_width - 1) / 2.0) / ((out_width - 1) / 2.0)
    v_norm = (v_idx - (out_height - 1) / 2.0) / ((out_height - 1) / 2.0)
    
    # Rectilinear rays: Z=forward, X=right, Y=down
    ray_x = u_norm * np.tan(half_fov_h)
    ray_y = v_norm * np.tan(half_fov_v)
    ray_z = np.ones_like(u_idx, dtype=np.float64)
    ray_len = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
    ray_x, ray_y, ray_z = ray_x / ray_len, ray_y / ray_len, ray_z / ray_len
    
    # Apply yaw (Y axis) and pitch (X axis)
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
    ray_x, ray_z = ray_x * c_yaw + ray_z * s_yaw, -ray_x * s_yaw + ray_z * c_yaw
    c_pitch, s_pitch = np.cos(pitch), np.sin(pitch)
    ray_y, ray_z = ray_y * c_pitch - ray_z * s_pitch, ray_y * s_pitch + ray_z * c_pitch
    
    # Equidistant fisheye projection
    phi = np.arccos(np.clip(ray_z, -1.0, 1.0))
    r_norm = phi / np.radians(fisheye_fov_deg / 2.0)
    theta = np.arctan2(ray_y, ray_x)
    
    x_fish = cx + r_norm * np.cos(theta) * radius
    y_fish = cy + r_norm * np.sin(theta) * radius
    
    valid = (r_norm <= 1.0) & (x_fish >= 0) & (x_fish < w) & (y_fish >= 0) & (y_fish < h)
    result = cv2.remap(fisheye_img, x_fish.astype(np.float32), y_fish.astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    result[~valid] = 0
    return result, valid


def _remap_fisheye(fisheye_img, ray_x, ray_y, ray_z, fisheye_fov_deg, cx, cy, radius,
                   lens_calib=None, is_lens2=False):
    """Remap world rays to fisheye pixels with optional calibration."""
    h, w = fisheye_img.shape[:2]
    
    if lens_calib is not None:
        cx = (1.0 - lens_calib.center_x) * w if is_lens2 else lens_calib.center_x * w
        cy = lens_calib.center_y * h
        fisheye_fov_deg = lens_calib.fov
        radius = min(w, h) / 2.0
    
    # Convert to equirect world coords: X=forward, Y=right, Z=up
    Xworld, Yworld, Zworld = ray_z, ray_x, -ray_y
    
    # Apply rotation correction
    if lens_calib is not None:
        rot_yaw = -lens_calib.rotation_yaw if is_lens2 else lens_calib.rotation_yaw
        rot_pitch, rot_roll = lens_calib.rotation_pitch, lens_calib.rotation_roll
        
        if abs(rot_yaw) > 1e-6 or abs(rot_pitch) > 1e-6 or abs(rot_roll) > 1e-6:
            cyaw, syaw = np.cos(rot_yaw), np.sin(rot_yaw)
            cpitch, spitch = np.cos(rot_pitch), np.sin(rot_pitch)
            croll, sroll = np.cos(rot_roll), np.sin(rot_roll)
            
            Rz = np.array([[cyaw, -syaw, 0], [syaw, cyaw, 0], [0, 0, 1]], dtype=np.float64)
            Ry = np.array([[cpitch, 0, spitch], [0, 1, 0], [-spitch, 0, cpitch]], dtype=np.float64)
            Rx = np.array([[1, 0, 0], [0, croll, -sroll], [0, sroll, croll]], dtype=np.float64)
            
            world_coords = np.stack([Xworld, Yworld, Zworld], axis=-1)
            rotated = world_coords @ (Rz @ Ry @ Rx).T
            Xworld, Yworld, Zworld = rotated[..., 0], rotated[..., 1], rotated[..., 2]
    
    # Fisheye projection
    Px, Py, Pz = -Zworld, Yworld, Xworld
    phi = np.arccos(np.clip(Pz, -1.0, 1.0))
    phi_max = np.radians(fisheye_fov_deg / 2.0)
    r_norm = phi / phi_max
    theta = np.arctan2(Py, Px) - np.pi / 2
    
    # Apply radial distortion
    if lens_calib is not None:
        k1, k2, k3 = lens_calib.k1, getattr(lens_calib, 'k2', 0.0), getattr(lens_calib, 'k3', 0.0)
        if abs(k1) > 1e-6 or abs(k2) > 1e-6 or abs(k3) > 1e-6:
            r_norm = r_norm * (1.0 + k1 * r_norm**2 + k2 * r_norm**4 + k3 * r_norm**6)
    
    x_fish = cx + r_norm * np.cos(theta) * radius
    y_fish = cy - r_norm * np.sin(theta) * radius
    
    valid = (r_norm <= 1.001) & (x_fish >= 0) & (x_fish < w) & (y_fish >= 0) & (y_fish < h)
    x_fish = np.clip(x_fish.astype(np.float32), -10, w + 10)
    y_fish = np.clip(y_fish.astype(np.float32), -10, h + 10)
    
    result = cv2.remap(fisheye_img, x_fish, y_fish, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    result[~valid] = 0
    return result, valid


def dual_fisheye_to_rectilinear(dual_img, out_width, out_height,
                                 out_fov_deg=90.0, yaw_deg=0.0, pitch_deg=0.0,
                                 fisheye_fov_deg=180.0, calibration=None, blend=True):
    """Extract a rectilinear view from dual-fisheye, blending both lenses."""
    h, w = dual_img.shape[:2]
    half_w = w // 2
    lens1, lens2 = dual_img[:, :half_w], dual_img[:, half_w:]
    
    lens_h, lens_w = lens1.shape[:2]
    cx, cy = lens_w / 2.0, lens_h / 2.0
    radius = min(lens_w, lens_h) / 2.0
    
    lens1_fov = calibration.lens1.fov if calibration else fisheye_fov_deg
    lens2_fov = calibration.lens2.fov if calibration else fisheye_fov_deg
    blend_fov = min(lens1_fov, lens2_fov)
    
    # Compute world rays
    out_fov_clamped = min(out_fov_deg, 150.0)
    half_fov_h = np.radians(out_fov_clamped / 2.0)
    half_fov_v = half_fov_h * out_height / out_width
    
    v_idx, u_idx = np.mgrid[0:out_height, 0:out_width]
    u_norm = (u_idx - (out_width - 1) / 2.0) / ((out_width - 1) / 2.0)
    v_norm = (v_idx - (out_height - 1) / 2.0) / ((out_height - 1) / 2.0)
    
    ray_x = u_norm * np.tan(half_fov_h)
    ray_y = v_norm * np.tan(half_fov_v)
    ray_z = np.ones_like(u_idx, dtype=np.float64)
    ray_len = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
    ray_x, ray_y, ray_z = ray_x / ray_len, ray_y / ray_len, ray_z / ray_len
    
    # Apply yaw and pitch
    yaw, pitch = np.radians(yaw_deg), np.radians(pitch_deg)
    c_yaw, s_yaw = np.cos(yaw), np.sin(yaw)
    ray_x, ray_z = ray_x * c_yaw + ray_z * s_yaw, -ray_x * s_yaw + ray_z * c_yaw
    c_pitch, s_pitch = np.cos(pitch), np.sin(pitch)
    ray_y, ray_z = ray_y * c_pitch - ray_z * s_pitch, ray_y * s_pitch + ray_z * c_pitch
    
    # Transform rays for each lens
    lens1_calib = calibration.lens1 if calibration else None
    lens2_calib = calibration.lens2 if calibration else None
    
    result1, valid1 = _remap_fisheye(lens1, ray_x, ray_y, ray_z,
                                      fisheye_fov_deg, cx, cy, radius, lens1_calib, is_lens2=False)
    result2, valid2 = _remap_fisheye(lens2, -ray_x, ray_y, -ray_z,
                                      fisheye_fov_deg, cx, cy, radius, lens2_calib, is_lens2=True)
    
    # Compute blending weights
    angle1_deg = np.degrees(np.arccos(np.clip(ray_z, -1, 1)))
    angle2_deg = np.degrees(np.arccos(np.clip(-ray_z, -1, 1)))
    max_angle = blend_fov / 2.0
    
    if blend:
        blend_width = 10.0
        weight1 = np.clip((max_angle - angle1_deg) / blend_width, 0, 1)
        weight2 = np.clip((max_angle - angle2_deg) / blend_width, 0, 1)
        total = np.maximum(weight1 + weight2, 1e-6)
        weight1, weight2 = weight1 / total, weight2 / total
    else:
        weight1 = (ray_z > 0).astype(np.float32)
        weight2 = 1.0 - weight1
    
    # Composite results
    both_valid = valid1 & valid2
    result = np.zeros_like(result1)
    
    if len(result.shape) == 3:
        w1, w2 = weight1[:, :, np.newaxis], weight2[:, :, np.newaxis]
        result[both_valid] = (w1[both_valid] * result1[both_valid].astype(np.float32) +
                              w2[both_valid] * result2[both_valid].astype(np.float32)).astype(np.uint8)
    else:
        result[both_valid] = (weight1[both_valid] * result1[both_valid].astype(np.float32) +
                              weight2[both_valid] * result2[both_valid].astype(np.float32)).astype(np.uint8)
    
    result[valid1 & ~valid2] = result1[valid1 & ~valid2]
    result[valid2 & ~valid1] = result2[valid2 & ~valid1]
    
    return result, valid1 | valid2


def extract_rectilinear_view(img, out_width, out_height, out_fov_deg=90.0,
                              yaw_deg=0.0, pitch_deg=0.0, fisheye_fov_deg=180.0,
                              calibration=None, blend=True):
    """Extract rectilinear view, auto-detecting single vs dual-lens."""
    h, w = img.shape[:2]
    is_dual = abs(w / h - 2.0) < 0.1
    
    if is_dual:
        return dual_fisheye_to_rectilinear(
            img, out_width, out_height, out_fov_deg, yaw_deg, pitch_deg,
            fisheye_fov_deg, calibration, blend)
    
    cx, cy, radius = w / 2.0, h / 2.0, min(w, h) / 2.0
    lens_calib = calibration.lens1 if calibration else None
    if lens_calib:
        cx, cy = lens_calib.center_x * w, lens_calib.center_y * h
        fisheye_fov_deg = lens_calib.fov
    return fisheye_to_rectilinear(img, out_width, out_height, out_fov_deg,
                                   yaw_deg, pitch_deg, fisheye_fov_deg, cx, cy, radius)


def generate_cubemap(img, face_size=512, fisheye_fov_deg=180.0, layout='horizontal',
                     calibration=None, blend=True):
    """Generate cubemap from fisheye. Faces: Front, Right, Back, Left, Up, Down."""
    faces_spec = [(0, 0), (90, 0), (180, 0), (-90, 0), (0, 90), (0, -90)]
    
    face_images = [
        extract_rectilinear_view(img, face_size, face_size, 90.0, yaw, pitch,
                                  fisheye_fov_deg, calibration, blend)[0]
        for yaw, pitch in faces_spec
    ]
    
    return _arrange_cubemap(face_images, face_size, layout)


def _arrange_cubemap(face_images, face_size, layout):
    """Arrange 6 cube faces into the specified layout."""
    front, right, back, left, up, down = face_images
    
    if layout == 'horizontal':
        return np.hstack(face_images), face_images
    elif layout == 'vertical':
        return np.vstack(face_images), face_images
    elif layout == 'cross':
        h = w = face_size
        channels = face_images[0].shape[2] if len(face_images[0].shape) == 3 else 1
        shape = (3 * h, 4 * w, channels) if channels > 1 else (3 * h, 4 * w)
        result = np.zeros(shape, dtype=np.uint8)
        result[0:h, w:2*w] = up
        result[h:2*h, 0:w] = left
        result[h:2*h, w:2*w] = front
        result[h:2*h, 2*w:3*w] = right
        result[h:2*h, 3*w:4*w] = back
        result[2*h:3*h, w:2*w] = down
        return result, face_images
    raise ValueError(f"Unknown layout: {layout}")


def main():
    parser = argparse.ArgumentParser(description='Extract rectilinear view from fisheye')
    parser.add_argument('--input', '-i', required=True, help='Input fisheye image')
    parser.add_argument('--output', '-o', required=True, help='Output image')
    parser.add_argument('--width', type=int, default=800, help='Output width (or face size for cubemap)')
    parser.add_argument('--height', type=int, default=600, help='Output height')
    parser.add_argument('--out-fov', type=float, default=90.0, help='Output FOV (max 150)')
    parser.add_argument('--fisheye-fov', type=float, default=180.0, help='Fisheye FOV')
    parser.add_argument('--yaw', type=float, default=0.0, help='Yaw: +right, -left')
    parser.add_argument('--pitch', type=float, default=0.0, help='Pitch: +up, -down')
    parser.add_argument('--cubemap', nargs='?', const='horizontal',
                        choices=['horizontal', 'vertical', 'cross'],
                        help='Output cubemap with specified layout')
    parser.add_argument('--calibration', '-c', type=str, help='Calibration JSON file')
    parser.add_argument('--no-blend', action='store_true', help='Disable seam blending')
    
    args = parser.parse_args()
    
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Could not load {args.input}", file=sys.stderr)
        sys.exit(1)
    
    h, w = img.shape[:2]
    is_dual = abs(w / h - 2.0) < 0.1
    
    calibration = None
    fisheye_fov = args.fisheye_fov
    if args.calibration:
        try:
            calibration = CameraCalibration.load_json(args.calibration)
            fisheye_fov = calibration.lens1.fov
            print(f"Loaded calibration: lens1 fov={fisheye_fov:.1f}°")
        except Exception as e:
            print(f"Warning: Could not load calibration: {e}", file=sys.stderr)
    
    print(f"Input: {w}x{h}, {'dual-lens' if is_dual else 'single-lens'}")
    
    if args.cubemap:
        result, _ = generate_cubemap(img, args.width, fisheye_fov, args.cubemap,
                                      calibration, not args.no_blend)
        print(f"Cubemap: {args.cubemap} layout, face size {args.width}")
    else:
        result, valid = extract_rectilinear_view(img, args.width, args.height,
                                                  args.out_fov, args.yaw, args.pitch,
                                                  fisheye_fov, calibration, not args.no_blend)
        print(f"Output: {args.width}x{args.height}, FOV {args.out_fov}°, "
              f"yaw {args.yaw}°, pitch {args.pitch}°")
    
    cv2.imwrite(args.output, result)
    print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
