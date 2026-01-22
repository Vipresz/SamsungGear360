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

PENALIZE_BOTH_BLACK = False
ENABLE_BLENDING = False


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


def extract_vignette_from_frames(frames, half_w, n_bins=50):
    """Extract averaged vignette falloff from multiple frames."""
    left_sum, right_sum = np.zeros(n_bins), np.zeros(n_bins)
    left_count, right_count = np.zeros(n_bins), np.zeros(n_bins)
    
    for left_img, right_img in frames:
        for img, b_sum, count in [(left_img, left_sum, left_count), (right_img, right_sum, right_count)]:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if len(img.shape) == 3 else img.astype(np.float32)
            h_img, w_img = gray.shape
            cx, cy = w_img // 2, h_img // 2
            max_r = min(w_img, h_img) // 2
            y, x = np.ogrid[:h_img, :w_img]
            r_map = np.sqrt((x - cx)**2 + (y - cy)**2) / max_r
            
            for j in range(n_bins - 1):
                r_min, r_max = j / n_bins, (j + 1) / n_bins
                mask = (r_map >= r_min) & (r_map < r_max) & (gray > 10) & (gray < 245)
                if np.sum(mask) > 50:
                    b_sum[j] += np.median(gray[mask])
                    count[j] += 1
    
    def create_falloff(brightness, count):
        count = np.maximum(count, 1)
        brightness = brightness / count
        brightness[-1] = brightness[-2] if brightness[-2] > 0 else brightness[-1]
        center = brightness[0] if brightness[0] > 0 else 1.0
        normalized = np.convolve(brightness / center, np.ones(5) / 5, mode='same')
        normalized = np.clip(normalized, 0.1, 2.0)
        return lambda r: 1.0 / normalized[np.clip((r * (n_bins - 1)).astype(int), 0, n_bins - 1)], normalized
    
    left_func, left_smooth = create_falloff(left_sum, left_count)
    right_func, right_smooth = create_falloff(right_sum, right_count)
    print(f"    Left:  edge={100*left_smooth[-1]:.1f}% of center, correction={1.0/left_smooth[-1]:.2f}x")
    print(f"    Right: edge={100*right_smooth[-1]:.1f}% of center, correction={1.0/right_smooth[-1]:.2f}x")
    return left_func, right_func


def apply_vignette_correction(img, falloff_func, strength=0.5):
    """Apply vignette correction to a fisheye image."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    max_r = min(w, h) // 2
    y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    r_map = np.clip(np.sqrt((x - cx)**2 + (y - cy)**2) / max_r, 0, 1)
    correction = np.clip(1.0 + strength * (falloff_func(r_map) - 1.0), 1.0, 1.5)
    
    if len(img.shape) == 3:
        corrected = img.astype(np.float32) * correction[:, :, np.newaxis]
    else:
        corrected = img.astype(np.float32) * correction
    return np.clip(corrected, 0, 255).astype(np.uint8)


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
    r = phi / np.radians(fov_degrees / 2)
    
    x_fish = (cx + r * np.cos(theta) * radius).astype(np.float32)
    y_fish = (cy - r * np.sin(theta) * radius).astype(np.float32)
    valid = (x_fish >= 0) & (x_fish < w) & (y_fish >= 0) & (y_fish < h) & (r <= 1)
    
    x_fish = np.clip(np.ascontiguousarray(x_fish), -10, w + 10)
    y_fish = np.clip(np.ascontiguousarray(y_fish), -10, h + 10)
    output = cv2.remap(img, x_fish, y_fish, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    output[~valid] = 0
    return output


def fisheye_to_equirect_calibrated(img, output_width, output_height, lens_calib: LensCalibration, base_fov=180.0):
    """Convert single fisheye to equirectangular with lens calibration."""
    h, w = img.shape[:2]
    if img is None or h == 0 or w == 0:
        raise ValueError(f"Invalid input image")
    
    cx0, cy0 = lens_calib.center_x * w, lens_calib.center_y * h
    radius = min(w, h) // 2
    lens_fov = lens_calib.fov
    
    v, u = np.mgrid[0:output_height, 0:output_width]
    lon_range = np.radians(base_fov)
    longitude = (u / output_width) * lon_range - lon_range / 2
    latitude = (0.5 - v / output_height) * np.pi
    
    Xworld = np.cos(latitude) * np.cos(longitude)
    Yworld = np.cos(latitude) * np.sin(longitude)
    Zworld = np.sin(latitude)
    
    # Apply rotation
    if abs(lens_calib.rotation_yaw) > 1e-6 or abs(lens_calib.rotation_pitch) > 1e-6 or abs(lens_calib.rotation_roll) > 1e-6:
        cyaw, syaw = np.cos(lens_calib.rotation_yaw), np.sin(lens_calib.rotation_yaw)
        cpitch, spitch = np.cos(lens_calib.rotation_pitch), np.sin(lens_calib.rotation_pitch)
        croll, sroll = np.cos(lens_calib.rotation_roll), np.sin(lens_calib.rotation_roll)
        Rz = np.array([[cyaw, -syaw, 0], [syaw, cyaw, 0], [0, 0, 1]], dtype=np.float64)
        Ry = np.array([[cpitch, 0, spitch], [0, 1, 0], [-spitch, 0, cpitch]], dtype=np.float64)
        Rx = np.array([[1, 0, 0], [0, croll, -sroll], [0, sroll, croll]], dtype=np.float64)
        world_coords = np.stack([Xworld, Yworld, Zworld], axis=-1)
        rotated = world_coords @ (Rz @ Ry @ Rx).T
        Xworld, Yworld, Zworld = rotated[..., 0], rotated[..., 1], rotated[..., 2]
    
    Px, Py, Pz = -Zworld, Yworld, Xworld
    phi = np.arccos(np.clip(Pz, -1, 1))
    theta = np.arctan2(Py, Px) - np.pi / 2
    r = phi / np.radians(lens_fov / 2)
    
    # Apply distortion
    if abs(lens_calib.k1) > 1e-6 or abs(lens_calib.k2) > 1e-6 or abs(lens_calib.k3) > 1e-6:
        r = r * (1.0 + lens_calib.k1 * r**2 + lens_calib.k2 * r**4 + lens_calib.k3 * r**6)
    
    x_fish = (cx0 + r * np.cos(theta) * radius).astype(np.float32)
    y_fish = (cy0 - r * np.sin(theta) * radius).astype(np.float32)
    valid = (x_fish >= 0) & (x_fish <= w - 1) & (y_fish >= 0) & (y_fish <= h - 1) & (r <= 1.001)
    
    x_fish = np.clip(np.ascontiguousarray(x_fish), -10, w + 10)
    y_fish = np.clip(np.ascontiguousarray(y_fish), -10, h + 10)
    output = cv2.remap(img, x_fish, y_fish, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    output[~valid] = 0
    return output, valid


def compute_seam_score(region1, region2, penalize_both_black=PENALIZE_BOTH_BLACK):
    """Compute RMS difference between two overlap regions."""
    r1, r2 = region1.astype(np.float32), region2.astype(np.float32)
    n_ch = region1.shape[2] if len(region1.shape) == 3 else 1
    
    has_data1 = np.any(region1 > 0, axis=2) if len(region1.shape) == 3 else region1 > 0
    has_data2 = np.any(region2 > 0, axis=2) if len(region2.shape) == 3 else region2 > 0
    both_black = ~has_data1 & ~has_data2
    
    diff_sq = np.sum((r1 - r2) ** 2, axis=2) if len(region1.shape) == 3 else (r1 - r2) ** 2
    
    if penalize_both_black:
        diff_sq[both_black] = 255.0 ** 2 * n_ch
        return np.sqrt(np.sum(diff_sq) / (diff_sq.size * n_ch)), diff_sq.size
    else:
        valid_mask = has_data1 | has_data2
        valid_count = np.sum(valid_mask)
        if valid_count == 0:
            return 0.0, 0
        return np.sqrt(np.sum(diff_sq[valid_mask]) / (valid_count * n_ch)), valid_count


def extract_seam_visualization(left_patch, right_patch, overlap_px):
    """Extract and visualize seam overlap regions."""
    out_h = left_patch.shape[0]
    if overlap_px < 1:
        return np.zeros((out_h, 100, 3), dtype=np.uint8), {'center': 0, 'side': 0, 'combined': 0, 'center_pixels': 0, 'side_pixels': 0, 'total_pixels': 0}
    
    center_left, center_right = left_patch[:, -overlap_px:], right_patch[:, :overlap_px]
    side_left, side_right = left_patch[:, :overlap_px], right_patch[:, -overlap_px:]
    
    center_score, center_count = compute_seam_score(center_left, center_right, True)
    side_score, side_count = compute_seam_score(side_left, side_right, True)
    total = center_count + side_count
    combined = (center_score * center_count + side_score * side_count) / total if total > 0 else 0
    
    center_avg = ((center_left.astype(np.float32) + center_right.astype(np.float32)) / 2).astype(np.uint8)
    side_avg = ((side_left.astype(np.float32) + side_right.astype(np.float32)) / 2).astype(np.uint8)
    
    scale = max(1, 120 // overlap_px)
    upscale = lambda img: cv2.resize(img, (overlap_px * scale, out_h), cv2.INTER_NEAREST)
    gap = np.ones((out_h, 4, 3), dtype=np.uint8) * 40
    
    row1 = np.hstack([upscale(center_left), gap, upscale(center_avg), gap, upscale(center_right)])
    row2 = np.hstack([upscale(side_right), gap, upscale(side_avg), gap, upscale(side_left)])
    
    header = np.zeros((40, row1.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, f"SEAM SCORE: {combined:.2f} RMS (center: {center_score:.2f}, side: {side_score:.2f})", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    
    label1 = np.zeros((30, row1.shape[1], 3), dtype=np.uint8)
    label2 = np.zeros((30, row2.shape[1], 3), dtype=np.uint8)
    cv2.putText(label1, f"CENTER: Left|Avg|Right ({center_score:.2f})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(label2, f"SIDE: Right|Avg|Left ({side_score:.2f})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    gap_h = np.ones((8, row1.shape[1], 3), dtype=np.uint8) * 40
    seam_vis = np.vstack([header, label1, row1, gap_h, label2, row2])
    
    return seam_vis, {'center': center_score, 'side': side_score, 'combined': combined,
                      'center_pixels': center_count, 'side_pixels': side_count, 'total_pixels': total}


def stitch_dual_fisheye(left_img, right_img, calibration, base_fov, output_width=None, output_height=None, blend=None):
    """Stitch dual fisheye images to equirectangular."""
    if blend is None:
        blend = ENABLE_BLENDING
    
    h, w = left_img.shape[:2]
    out_w = output_width or (w * 2)
    out_h = output_height or out_w // 2
    half_w = out_w // 2
    
    lens_fov = min(calibration.lens1.fov, calibration.lens2.fov) if calibration else base_fov
    proj_w = int(half_w * lens_fov / base_fov)
    overlap_px = proj_w - half_w
    
    right_flipped = np.fliplr(right_img)
    
    if calibration:
        left_patch, left_mask = fisheye_to_equirect_calibrated(left_img, proj_w, out_h, calibration.lens1, lens_fov)
        right_patch, right_mask = fisheye_to_equirect_calibrated(right_flipped, proj_w, out_h, calibration.lens2, lens_fov)
    else:
        left_patch = fisheye_to_equirect_single(left_img, proj_w, out_h, lens_fov)
        right_patch = fisheye_to_equirect_single(right_flipped, proj_w, out_h, lens_fov)
        left_mask = np.any(left_patch > 0, axis=2) if len(left_patch.shape) == 3 else left_patch > 0
        right_mask = np.any(right_patch > 0, axis=2) if len(right_patch.shape) == 3 else right_patch > 0
    
    right_patch, right_mask = np.fliplr(right_patch), np.fliplr(right_mask)
    
    if overlap_px > 0 and blend:
        result = _blend_with_overlap(left_patch, left_mask, right_patch, right_mask, half_w, overlap_px)
    else:
        # No blending: take center half_w columns from each patch
        margin = max(0, overlap_px // 2)
        left_crop = left_patch[:, margin:margin + half_w]
        right_crop = right_patch[:, margin:margin + half_w]
        # Ensure we have exactly half_w columns
        if left_crop.shape[1] < half_w:
            left_crop = left_patch[:, :half_w]
        if right_crop.shape[1] < half_w:
            right_crop = right_patch[:, :half_w]
        result = np.hstack([left_crop, right_crop])
    
    return result, left_patch, right_patch, overlap_px


def _blend_with_overlap(left_patch, left_mask, right_patch, right_mask, half_w, overlap_px):
    """Blend two patches with overlapping content."""
    out_h, proj_w = left_patch.shape[0], left_patch.shape[1]
    n_channels = left_patch.shape[2] if len(left_patch.shape) == 3 else 1
    if n_channels > 1:
        result = np.zeros((out_h, half_w * 2, n_channels), dtype=np.uint8)
    else:
        result = np.zeros((out_h, half_w * 2), dtype=np.uint8)
    margin = overlap_px // 2
    
    result[:, margin:half_w - margin] = left_patch[:, 2*margin:half_w]
    result[:, half_w + margin:2*half_w - margin] = right_patch[:, 2*margin:half_w]
    
    # Center seam blend
    for i in range(2 * margin):
        alpha = i / max(1, 2 * margin - 1)
        out_col, left_col, right_col = half_w - margin + i, half_w + i, i
        if left_col < proj_w and right_col < proj_w:
            both = left_mask[:, left_col] & right_mask[:, right_col]
            if np.any(both):
                result[both, out_col] = ((1 - alpha) * left_patch[both, left_col].astype(np.float32) +
                                         alpha * right_patch[both, right_col].astype(np.float32)).astype(np.uint8)
            result[left_mask[:, left_col] & ~right_mask[:, right_col], out_col] = left_patch[left_mask[:, left_col] & ~right_mask[:, right_col], left_col]
            result[~left_mask[:, left_col] & right_mask[:, right_col], out_col] = right_patch[~left_mask[:, left_col] & right_mask[:, right_col], right_col]
    
    # Side seam blend
    for i in range(margin):
        alpha = i / max(1, margin - 1)
        left_col, right_col = margin + i, proj_w - margin + i
        if left_col < proj_w and right_col < proj_w:
            both = left_mask[:, left_col] & right_mask[:, right_col]
            if np.any(both):
                result[both, i] = (alpha * left_patch[both, left_col].astype(np.float32) +
                                   (1 - alpha) * right_patch[both, right_col].astype(np.float32)).astype(np.uint8)
            result[left_mask[:, left_col] & ~right_mask[:, right_col], i] = left_patch[left_mask[:, left_col] & ~right_mask[:, right_col], left_col]
            result[~left_mask[:, left_col] & right_mask[:, right_col], i] = right_patch[~left_mask[:, left_col] & right_mask[:, right_col], right_col]
        
        out_col = 2*half_w - margin + i
        if out_col < 2*half_w:
            left_col, right_col = i, half_w + margin + i
            if left_col < proj_w and right_col < proj_w:
                both = left_mask[:, left_col] & right_mask[:, right_col]
                if np.any(both):
                    result[both, out_col] = ((1 - alpha) * left_patch[both, left_col].astype(np.float32) +
                                             alpha * right_patch[both, right_col].astype(np.float32)).astype(np.uint8)
                result[left_mask[:, left_col] & ~right_mask[:, right_col], out_col] = left_patch[left_mask[:, left_col] & ~right_mask[:, right_col], left_col]
                result[~left_mask[:, left_col] & right_mask[:, right_col], out_col] = right_patch[~left_mask[:, left_col] & right_mask[:, right_col], right_col]
    
    return result


def process_video(input_path, output_path, calibration, base_fov=180.0, blend=True,
                  output_width=None, output_height=None, frame_rate=1.0, max_frames=None,
                  vignette_correction=False, vignette_strength=0.5):
    """Process a dual-fisheye video to equirectangular."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise ValueError("Cannot read first frame")
    
    h, w = first_frame.shape[:2]
    half_w = w // 2
    out_w, out_h = output_width or w, output_height or (output_width or w) // 2
    
    step = max(1, int(1.0 / frame_rate)) if frame_rate > 0 else 1
    frame_limit = min(max_frames, total_frames) if max_frames else total_frames
    frame_indices = list(range(0, frame_limit, step))
    
    left_falloff, right_falloff = None, None
    if vignette_correction:
        print(f"  Extracting vignette from {min(len(frame_indices), 20)} frames...")
        sample_step = max(1, len(frame_indices) // 20)
        sample_frames = []
        for idx in frame_indices[::sample_step][:20]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                sample_frames.append((frame[:, :half_w], frame[:, half_w:]))
        if sample_frames:
            left_falloff, right_falloff = extract_vignette_from_frames(sample_frames, half_w)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    output_fps = fps / step
    print(f"Processing video: {input_path}")
    print(f"  Input: {w}x{h}, {total_frames} frames @ {fps:.1f} fps")
    print(f"  Output: {out_w}x{out_h}, {len(frame_indices)} frames @ {output_fps:.1f} fps")
    print(f"  Sampling: every {step} frame(s), max {frame_limit}")
    if vignette_correction:
        print(f"  Vignette correction: enabled (strength={vignette_strength})")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Cannot create output video: {output_path}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    processed = 0
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        left_raw, right_raw = frame[:, :half_w], frame[:, half_w:]
        if vignette_correction and left_falloff and right_falloff:
            left_raw = apply_vignette_correction(left_raw, left_falloff, vignette_strength)
            right_raw = apply_vignette_correction(right_raw, right_falloff, vignette_strength)
        
        left_img, _ = mask_fisheye_circle(left_raw)
        right_img, _ = mask_fisheye_circle(right_raw)
        result, _, _, _ = stitch_dual_fisheye(left_img, right_img, calibration, base_fov, out_w, out_h, blend)
        
        writer.write(result)
        processed += 1
        if processed % 50 == 0 or processed == len(frame_indices):
            print(f"  Processed {processed}/{len(frame_indices)} frames ({100*processed/len(frame_indices):.1f}%)")
    
    writer.release()
    cap.release()
    print(f"Saved: {output_path}")
    return processed


def main():
    parser = argparse.ArgumentParser(description='Convert fisheye to equirectangular')
    parser.add_argument('--input', '-i', required=True, help='Input image/video')
    parser.add_argument('--output', '-o', default=None, help='Output path')
    parser.add_argument('--width', type=int, default=None, help='Output width')
    parser.add_argument('--height', type=int, default=None, help='Output height')
    parser.add_argument('--fov', type=float, default=180.0, help='Base FOV')
    parser.add_argument('--calibration', '-c', type=str, help='Calibration JSON')
    parser.add_argument('--no-blend', action='store_true', help='Disable blending')
    parser.add_argument('--extract-seams', action='store_true', help='Visualize seams')
    parser.add_argument('--plot', action='store_true', help='Plot output')
    parser.add_argument('--frame-rate', type=float, default=1.0, help='Video sampling rate')
    parser.add_argument('--max-frames', type=int, help='Max frames')
    parser.add_argument('--vignette-correction', action='store_true', help='Apply vignette correction')
    parser.add_argument('--vignette-strength', type=float, default=0.5, help='Vignette strength')
    args = parser.parse_args()
    
    is_video = Path(args.input).suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    
    calibration = None
    if args.calibration:
        try:
            calibration = CameraCalibration.load_json(args.calibration)
            print(f"Loaded calibration from {args.calibration}")
            for name, lens in [('Lens1', calibration.lens1), ('Lens2', calibration.lens2)]:
                print(f"  {name}: center=({lens.center_x:.4f}, {lens.center_y:.4f}), fov={lens.fov:.4f}, k1={lens.k1:.4f}")
                print(f"         yaw={lens.rotation_yaw:.4f}, pitch={lens.rotation_pitch:.4f}, roll={lens.rotation_roll:.4f}")
        except Exception as e:
            print(f"Warning: Could not load calibration: {e}")
    
    if is_video:
        if not args.output:
            args.output = str(Path(args.input).with_suffix('_equirect.mp4'))
        process_video(args.input, args.output, calibration, args.fov, not args.no_blend,
                      args.width, args.height, args.frame_rate, args.max_frames,
                      args.vignette_correction, args.vignette_strength)
        return
    
    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Could not load {args.input}", file=sys.stderr)
        sys.exit(1)
    
    h, w = img.shape[:2]
    is_dual = abs(w / h - 2.0) < 0.1
    result, seam_vis = None, None
    
    if is_dual:
        left_img, _ = mask_fisheye_circle(img[:, :w // 2])
        right_img, _ = mask_fisheye_circle(img[:, w // 2:])
        out_w, out_h = args.width or w, args.height or ((args.width or w) // 2)
        
        print(f"Converting {w}x{h} dual-lens (FOV: {args.fov}°)...")
        result, left_patch, right_patch, overlap_px = stitch_dual_fisheye(
            left_img, right_img, calibration, args.fov, out_w, out_h, False if args.no_blend else None)
        
        print(f"  Overlap: {overlap_px}px, Blending: {'off' if args.no_blend else 'on'}")
        if overlap_px > 0 and args.extract_seams:
            seam_vis, scores = extract_seam_visualization(left_patch, right_patch, overlap_px)
            print(f"  Seam scores: center={scores['center']:.2f}, side={scores['side']:.2f}, combined={scores['combined']:.2f}")
    else:
        out_w, out_h = args.width or w, args.height or w
        print(f"Converting {w}x{h} single-lens (FOV: {args.fov}°)...")
        result = fisheye_to_equirect_calibrated(img, out_w, out_h, calibration.lens1, args.fov)[0] if calibration else fisheye_to_equirect_single(img, out_w, out_h, args.fov)
    
    if args.output:
        output_img = seam_vis if args.extract_seams and seam_vis is not None else result
        if output_img is not None:
            cv2.imwrite(args.output, output_img)
            print(f"Saved {output_img.shape[1]}x{output_img.shape[0]} to {args.output}")
    
    if args.plot and result is not None:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(14, 8))
            plt.imshow(cv2.cvtColor(seam_vis if args.extract_seams and seam_vis is not None else result, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        except ImportError:
            print("matplotlib not installed")


if __name__ == '__main__':
    main()
