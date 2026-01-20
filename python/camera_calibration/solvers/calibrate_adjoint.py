#!/usr/bin/env python3
"""Joint optimization calibration for dual-fisheye 360° cameras."""

import argparse
import sys
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import differential_evolution

try:
    import tomllib
except ImportError:
    try:
        import toml as tomllib
    except ImportError:
        tomllib = None

try:
    from camera_calibration.calib.calibration_config import CameraCalibration, LensCalibration
    from camera_calibration.projections.fisheye_to_equirectangular import (
        fisheye_to_equirect_calibrated, mask_fisheye_circle, stitch_dual_fisheye
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from camera_calibration.calib.calibration_config import CameraCalibration, LensCalibration
    from camera_calibration.projections.fisheye_to_equirectangular import (
        fisheye_to_equirect_calibrated, mask_fisheye_circle, stitch_dual_fisheye
    )

DEFAULT_CONFIG_PATH = Path(__file__).parent / "calibration_config.toml"


class OptimizationConfig:
    """Manages optimization parameters from TOML config."""
    
    PARAM_NAMES = [
        'lens1.center_x', 'lens1.center_y', 'lens1.fov',
        'lens1.k1', 'lens1.k2', 'lens1.k3',
        'lens1.rotation_yaw', 'lens1.rotation_pitch', 'lens1.rotation_roll',
        'lens2.center_x', 'lens2.center_y', 'lens2.fov',
        'lens2.k1', 'lens2.k2', 'lens2.k3',
        'lens2.rotation_yaw', 'lens2.rotation_pitch', 'lens2.rotation_roll',
    ]
    
    def __init__(self, config_path=None):
        self.params = {}
        self.optimizer = {}
        self.regularization = {}
        
        if config_path and Path(config_path).exists():
            self._load_toml(config_path)
        else:
            self._use_defaults()
        self._build_param_mapping()
    
    def _load_toml(self, config_path):
        if tomllib is None:
            print("Warning: TOML not available, using defaults")
            self._use_defaults()
            return
        
        with open(config_path, 'rb') as f:
            try:
                config = tomllib.load(f)
            except Exception:
                with open(config_path, 'r') as ft:
                    config = tomllib.load(ft)
        
        for lens in ['lens1', 'lens2']:
            lens_config = config.get(lens, {})
            for param in ['center_x', 'center_y', 'fov', 'k1', 'k2', 'k3',
                          'rotation_yaw', 'rotation_pitch', 'rotation_roll']:
                p = lens_config.get(param, {})
                default = 180.0 if param == 'fov' else (0.5 if 'center' in param else 0.0)
                self.params[f"{lens}.{param}"] = {
                    'optimize': p.get('optimize', False),
                    'nominal': p.get('nominal', default),
                    'min': p.get('min', -1.0),
                    'max': p.get('max', 1.0),
                    'use_lens1': p.get('use_lens1', False),
                }
        
        self.optimizer = config.get('optimizer', {})
        
        prep = config.get('preprocessing', {})
        self.preprocessing = {
            'normalize_lighting': prep.get('normalize_lighting', True),
        }
        
        reg = config.get('regularization', {})
        self.regularization = {
            'enabled': reg.get('enabled', True),
            'lambda_center': reg.get('lambda_center', 100.0),
            'lambda_fov': reg.get('lambda_fov', 0.1),
            'lambda_distortion': reg.get('lambda_distortion', 10.0),
            'lambda_rotation': reg.get('lambda_rotation', 50.0),
        }
        print(f"Loaded config: {config_path}")
    
    def _use_defaults(self):
        defaults = {
            'lens1.center_x': {'optimize': True, 'nominal': 0.5, 'min': 0.45, 'max': 0.55},
            'lens1.center_y': {'optimize': True, 'nominal': 0.5, 'min': 0.45, 'max': 0.55},
            'lens1.fov': {'optimize': True, 'nominal': 195.0, 'min': 180.0, 'max': 220.0},
            'lens1.k1': {'optimize': True, 'nominal': 0.0, 'min': -0.3, 'max': 0.3},
            'lens1.k2': {'optimize': True, 'nominal': 0.0, 'min': -0.3, 'max': 0.3},
            'lens1.k3': {'optimize': True, 'nominal': 0.0, 'min': -0.3, 'max': 0.3},
            'lens1.rotation_yaw': {'optimize': False, 'nominal': 0.0, 'min': -0.15, 'max': 0.15},
            'lens1.rotation_pitch': {'optimize': False, 'nominal': 0.0, 'min': -0.15, 'max': 0.15},
            'lens1.rotation_roll': {'optimize': False, 'nominal': 0.0, 'min': -0.15, 'max': 0.15},
            'lens2.center_x': {'optimize': True, 'nominal': 0.5, 'min': 0.45, 'max': 0.55},
            'lens2.center_y': {'optimize': True, 'nominal': 0.5, 'min': 0.45, 'max': 0.55},
            'lens2.fov': {'optimize': False, 'nominal': 195.0, 'min': 180.0, 'max': 220.0, 'use_lens1': True},
            'lens2.k1': {'optimize': True, 'nominal': 0.0, 'min': -0.3, 'max': 0.3},
            'lens2.k2': {'optimize': True, 'nominal': 0.0, 'min': -0.3, 'max': 0.3},
            'lens2.k3': {'optimize': True, 'nominal': 0.0, 'min': -0.3, 'max': 0.3},
            'lens2.rotation_yaw': {'optimize': True, 'nominal': 0.0, 'min': -0.15, 'max': 0.15},
            'lens2.rotation_pitch': {'optimize': True, 'nominal': 0.0, 'min': -0.15, 'max': 0.15},
            'lens2.rotation_roll': {'optimize': True, 'nominal': 0.0, 'min': -0.15, 'max': 0.15},
        }
        for name, vals in defaults.items():
            vals.setdefault('use_lens1', False)
            self.params[name] = vals
        
        self.optimizer = {'maxiter': 100, 'popsize': 15, 'strategy': 'best1bin'}
        self.preprocessing = {'normalize_lighting': True}
        self.regularization = {
            'enabled': True,
            'lambda_center': 100.0, 'lambda_fov': 0.1,
            'lambda_distortion': 10.0, 'lambda_rotation': 50.0,
        }
    
    def _build_param_mapping(self):
        self.opt_params = []
        self.opt_indices = {}
        for name in self.PARAM_NAMES:
            p = self.params[name]
            if p['optimize']:
                self.opt_indices[name] = len(self.opt_params)
                self.opt_params.append((name, (p['min'], p['max'])))
    
    def get_bounds(self):
        return [bounds for _, bounds in self.opt_params]
    
    def get_n_params(self):
        return len(self.opt_params)
    
    def get_initial_vector(self):
        return [self.params[name]['nominal'] for name, _ in self.opt_params]
    
    def vector_to_lenses(self, opt_vector):
        values = {name: self.params[name]['nominal'] for name in self.PARAM_NAMES}
        for i, (name, _) in enumerate(self.opt_params):
            values[name] = opt_vector[i]
        for name, p in self.params.items():
            if p.get('use_lens1') and not p['optimize']:
                values[name] = values[name.replace('lens2.', 'lens1.')]
        
        lens1 = LensCalibration(
            center_x=values['lens1.center_x'], center_y=values['lens1.center_y'],
            fov=values['lens1.fov'], k1=values['lens1.k1'], k2=values['lens1.k2'], k3=values['lens1.k3'],
            rotation_yaw=values['lens1.rotation_yaw'], rotation_pitch=values['lens1.rotation_pitch'],
            rotation_roll=values['lens1.rotation_roll'],
        )
        lens2 = LensCalibration(
            center_x=values['lens2.center_x'], center_y=values['lens2.center_y'],
            fov=values['lens2.fov'], k1=values['lens2.k1'], k2=values['lens2.k2'], k3=values['lens2.k3'],
            rotation_yaw=values['lens2.rotation_yaw'], rotation_pitch=values['lens2.rotation_pitch'],
            rotation_roll=values['lens2.rotation_roll'],
        )
        return lens1, lens2
    
    def compute_regularization(self, lens1, lens2):
        """Compute regularization: E = λ_c||c-c0||² + λ_f(fov-fov0)² + λ_k||k||² + λ_r||rot||²"""
        reg = self.regularization
        breakdown = {'center': 0.0, 'fov': 0.0, 'distortion': 0.0, 'rotation': 0.0}
        
        if not reg.get('enabled', True):
            return 0.0, breakdown
        
        c0_x, c0_y = self.params['lens1.center_x']['nominal'], self.params['lens1.center_y']['nominal']
        fov0 = self.params['lens1.fov']['nominal']
        
        breakdown['center'] = reg['lambda_center'] * (
            (lens1.center_x - c0_x)**2 + (lens1.center_y - c0_y)**2 +
            (lens2.center_x - c0_x)**2 + (lens2.center_y - c0_y)**2
        )
        breakdown['fov'] = reg['lambda_fov'] * ((lens1.fov - fov0)**2 + (lens2.fov - fov0)**2)
        breakdown['distortion'] = reg['lambda_distortion'] * (
            lens1.k1**2 + lens1.k2**2 + lens1.k3**2 + lens2.k1**2 + lens2.k2**2 + lens2.k3**2
        )
        breakdown['rotation'] = reg['lambda_rotation'] * (
            lens1.rotation_yaw**2 + lens1.rotation_pitch**2 + lens1.rotation_roll**2 +
            lens2.rotation_yaw**2 + lens2.rotation_pitch**2 + lens2.rotation_roll**2
        )
        
        return sum(breakdown.values()), breakdown
    
    def print_summary(self):
        print("\nOptimization Configuration:")
        print("-" * 50)
        opt_names = [name for name, _ in self.opt_params]
        print(f"Optimizing {len(opt_names)} parameters:")
        for name in opt_names:
            p = self.params[name]
            print(f"  {name}: [{p['min']:.4f}, {p['max']:.4f}] (nominal: {p['nominal']:.4f})")
        
        fixed = [n for n in self.PARAM_NAMES if n not in opt_names]
        if fixed:
            print(f"\nFixed ({len(fixed)}):")
            for name in fixed:
                p = self.params[name]
                link = " (=lens1)" if p.get('use_lens1') else ""
                print(f"  {name} = {p['nominal']:.4f}{link}")
        
        prep = self.preprocessing
        print(f"\nPreprocessing: normalize_lighting={'on' if prep.get('normalize_lighting', True) else 'off'}")
        
        reg = self.regularization
        if reg.get('enabled', True):
            print(f"Regularization: enabled")
            print(f"  λ_center={reg['lambda_center']}, λ_fov={reg['lambda_fov']}, λ_dist={reg['lambda_distortion']}, λ_rot={reg['lambda_rotation']}")
        else:
            print(f"Regularization: disabled")
        print("-" * 50)


def downsample_frames(frames_data, scale):
    if scale >= 1.0:
        return frames_data
    return [
        (cv2.resize(l, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
         cv2.resize(r, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        for l, r in frames_data
    ]


def normalize_lighting(img):
    """Normalize lighting to focus on features rather than brightness."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) if len(img.shape) == 3 else img.astype(np.float32)
    valid = gray > 0
    if np.sum(valid) < 100:
        return img
    
    mean_val, std_val = np.mean(gray[valid]), max(np.std(gray[valid]), 1e-6)
    normalized = np.zeros_like(gray)
    normalized[valid] = np.clip(((gray[valid] - mean_val) / std_val) * 50 + 128, 0, 255)
    
    return cv2.cvtColor(normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR) if len(img.shape) == 3 else normalized.astype(np.uint8)


def compute_seam_error(left_img, right_img, lens1, lens2, base_fov, normalize=True):
    """Compute seam alignment error using RMS difference on TRUE overlap regions.
    
    Projects each lens to its full FOV (lens_fov) so we get actual overlapping data
    where both lenses can see the same 3D scene.
    
    Args:
        normalize: If True, normalize lighting (grayscale) before comparison.
    """
    h, w = left_img.shape[:2]
    left_masked, _ = mask_fisheye_circle(left_img, margin=0)
    right_masked, _ = mask_fisheye_circle(right_img, margin=0)
    right_flipped = np.fliplr(right_masked)
    
    # Use the smaller of the two FOVs to determine overlap
    lens_fov = min(lens1.fov, lens2.fov)
    overlap_deg = max(0, lens_fov - base_fov)  # Total overlap in degrees
    
    if overlap_deg < 1:
        return 1000.0  # No overlap
    
    # Project each lens to its FULL FOV to get actual overlap data
    # half_w covers base_fov (180°), so lens_fov requires more pixels
    half_w = w
    out_h = h
    proj_w = int(half_w * lens_fov / base_fov)  # Wider projection
    overlap_px = proj_w - half_w  # Extra pixels = overlap
    
    try:
        left_patch, left_mask = fisheye_to_equirect_calibrated(left_masked, proj_w, out_h, lens1, lens_fov)
        right_patch, right_mask = fisheye_to_equirect_calibrated(right_flipped, proj_w, out_h, lens2, lens_fov)
    except Exception:
        return 1000.0
    
    right_patch, right_mask = np.fliplr(right_patch), np.fliplr(right_mask)
    
    if overlap_px < 1:
        return 1000.0
    
    # Maximum penalty for black pixels (as if they differ by 255)
    BLACK_PENALTY = 255.0
    
    total_sq_sum, total_count = 0.0, 0
    
    def compute_region_error(r1, r2, mask1, mask2):
        """Compute error with heavy penalty for both-black pixels."""
        nonlocal total_sq_sum, total_count
        
        # Identify pixel categories
        both_valid = mask1 & mask2
        both_black = ~mask1 & ~mask2
        one_black = (mask1 & ~mask2) | (~mask1 & mask2)
        
        total_pixels = mask1.size
        
        # 1. Both valid: compute actual RMS
        if np.any(both_valid):
            r1_proc = normalize_lighting(r1) if normalize else r1
            r2_proc = normalize_lighting(r2) if normalize else r2
            
            if len(r1.shape) == 3:
                valid_3d = np.stack([both_valid] * 3, axis=2)
                diff_sq = np.where(valid_3d, (r1_proc.astype(np.float32) - r2_proc.astype(np.float32))**2, 0)
                total_sq_sum += np.sum(diff_sq)
                total_count += np.sum(both_valid) * 3
            else:
                diff_sq = np.where(both_valid, (r1_proc.astype(np.float32) - r2_proc.astype(np.float32))**2, 0)
                total_sq_sum += np.sum(diff_sq)
                total_count += np.sum(both_valid)
        
        # 2. Both black: HEAVY penalty (max difference squared)
        n_both_black = np.sum(both_black)
        if n_both_black > 0:
            n_channels = 3 if len(r1.shape) == 3 else 1
            total_sq_sum += n_both_black * n_channels * (BLACK_PENALTY ** 2)
            total_count += n_both_black * n_channels
        
        # 3. One black (only one lens has data): moderate penalty
        n_one_black = np.sum(one_black)
        if n_one_black > 0:
            n_channels = 3 if len(r1.shape) == 3 else 1
            # Penalty of ~128 (half of max) for one-sided data
            total_sq_sum += n_one_black * n_channels * (128.0 ** 2)
            total_count += n_one_black * n_channels
    
    # Center seam: both patches have data for the SAME longitude range
    r1 = left_patch[:, -overlap_px:]
    r2 = right_patch[:, :overlap_px]
    m1 = left_mask[:, -overlap_px:]
    m2 = right_mask[:, :overlap_px]
    compute_region_error(r1, r2, m1, m2)
    
    # Side seam: Left's first overlap_px columns = Right's last overlap_px columns
    r1 = left_patch[:, :overlap_px]
    r2 = right_patch[:, -overlap_px:]
    m1 = left_mask[:, :overlap_px]
    m2 = right_mask[:, -overlap_px:]
    compute_region_error(r1, r2, m1, m2)
    
    return np.sqrt(total_sq_sum / total_count) if total_count > 0 else 1000.0


def objective_with_config(params, frames_data, base_fov, config):
    """Objective: E = seam_error + regularization."""
    lens1, lens2 = config.vector_to_lenses(params)
    normalize = config.preprocessing.get('normalize_lighting', True)
    seam_error = sum(compute_seam_error(l, r, lens1, lens2, base_fov, normalize) for l, r in frames_data) / len(frames_data)
    reg_penalty, _ = config.compute_regularization(lens1, lens2)
    return seam_error + reg_penalty


def calibrate_with_config(frames_data, base_fov, config, workers=1):
    """Optimization using differential evolution with TOML config."""
    print(f"\n{'='*60}")
    print("CALIBRATION - Differential Evolution")
    print(f"{'='*60}")
    config.print_summary()
    
    bounds = config.get_bounds()
    n_params = config.get_n_params()
    maxiter = config.optimizer.get('maxiter', 100)
    popsize = config.optimizer.get('popsize', 15)
    strategy = config.optimizer.get('strategy', 'best1bin')
    tol = config.optimizer.get('tol', 0.01)
    mutation = config.optimizer.get('mutation', [0.5, 1.0])
    recombination = config.optimizer.get('recombination', 0.7)
    if isinstance(mutation, list):
        mutation = tuple(mutation)
    
    print(f"\nPopulation: {popsize * n_params}, Max generations: {maxiter}\n")
    
    init_vector = config.get_initial_vector()
    lens1, lens2 = config.vector_to_lenses(init_vector)
    init_rms = compute_seam_error(frames_data[0][0], frames_data[0][1], lens1, lens2, base_fov)
    print(f"Initial RMS: {init_rms:.2f}\n")
    
    best_so_far, gen_count = [float('inf')], [0]
    
    def callback(xk, convergence):
        gen_count[0] += 1
        err = objective_with_config(xk, frames_data, base_fov, config)
        marker = " (new best)" if err < best_so_far[0] else ""
        if err < best_so_far[0]:
            best_so_far[0] = err
        print(f"  gen {gen_count[0]:3d}: {err:.2f}{marker}")
        return False
    
    result = differential_evolution(
        objective_with_config, bounds=bounds, args=(frames_data, base_fov, config),
        strategy=strategy, maxiter=maxiter, popsize=popsize, tol=tol,
        mutation=mutation, recombination=recombination, seed=42,
        callback=callback, polish=True, updating='deferred', workers=workers
    )
    
    print(f"\n  final: {result.fun:.2f}")
    print(f"  {result.message}, {result.nfev} evaluations")
    
    lens1, lens2 = config.vector_to_lenses(result.x)
    seam_error = sum(compute_seam_error(l, r, lens1, lens2, base_fov) for l, r in frames_data) / len(frames_data)
    reg_penalty, reg = config.compute_regularization(lens1, lens2)
    
    print(f"\n{'='*60}")
    print("OBJECTIVE BREAKDOWN")
    print(f"{'='*60}")
    print(f"  Seam RMS:     {seam_error:.2f}")
    print(f"  Regularization: {reg_penalty:.2f}")
    print(f"    center={reg['center']:.4f}, fov={reg['fov']:.4f}, dist={reg['distortion']:.4f}, rot={reg['rotation']:.4f}")
    print(f"  Total:        {result.fun:.2f}")
    
    print(f"\n{'='*60}")
    print("OPTIMIZED PARAMETERS")
    print(f"{'='*60}")
    for i, lens in enumerate([lens1, lens2], 1):
        print(f"\nLens {i}:")
        print(f"  center: ({lens.center_x:.6f}, {lens.center_y:.6f})")
        print(f"  fov: {lens.fov:.2f}°")
        print(f"  k: [{lens.k1:.6f}, {lens.k2:.6f}, {lens.k3:.6f}]")
        print(f"  rot: [{lens.rotation_yaw:.6f}, {lens.rotation_pitch:.6f}, {lens.rotation_roll:.6f}]")
    
    return CameraCalibration(lens1=lens1, lens2=lens2, is_horizontal=True)


def project_equirectangular(left_img, right_img, calib, base_fov, blend=False):
    """Project dual-fisheye to equirectangular."""
    left_masked, _ = mask_fisheye_circle(left_img)
    right_masked, _ = mask_fisheye_circle(right_img)
    result, _, _, _ = stitch_dual_fisheye(left_masked, right_masked, calib, base_fov, blend=blend)
    return result


def main():
    parser = argparse.ArgumentParser(description='Dual-fisheye calibration')
    parser.add_argument('--image', '-i', help='Dual-fisheye image')
    parser.add_argument('--video', help='Video for multi-frame calibration')
    parser.add_argument('--output', '-o', help='Output image path')
    parser.add_argument('--config', '-c', help='TOML config file')
    parser.add_argument('--fov', type=float, default=180.0, help='Base FOV (degrees)')
    parser.add_argument('--frames', type=int, default=None, help='Upper limit of first N frames to consider (default: all)')
    parser.add_argument('--frame-rate', type=float, default=1.0, help='Frame sampling rate (0.2 = every 5th frame, 0.5 = every 2nd)')
    parser.add_argument('--scale', type=float, default=0.25, help='Downsample scale')
    parser.add_argument('--workers', type=int, default=1, help='Parallel workers')
    parser.add_argument('--maxiter', type=int, help='Override max generations')
    parser.add_argument('--popsize', type=int, help='Override population size')
    parser.add_argument('--no-regularization', action='store_true', help='Disable regularization (pure seam error)')
    parser.add_argument('--no-normalize', action='store_true', help='Disable lighting normalization (use raw RGB)')
    parser.add_argument('--output-video', help='Output equirectangular video path (requires --video)')
    parser.add_argument('--output-frame-rate', type=float, default=1.0, help='Output video frame rate (0.5 = every 2nd frame)')
    args = parser.parse_args()
    
    if not args.image and not args.video:
        parser.error("Provide --image or --video")
    
    config_path = args.config or (str(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else None)
    config = OptimizationConfig(config_path)
    
    if args.maxiter:
        config.optimizer['maxiter'] = args.maxiter
    if args.popsize:
        config.optimizer['popsize'] = args.popsize
    if args.no_regularization:
        config.regularization['enabled'] = False
    if args.no_normalize:
        config.preprocessing['normalize_lighting'] = False
    
    frames_data, source_frame = [], None
    
    if args.image:
        print(f"Loading: {args.image}")
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"Error: Cannot load {args.image}")
        h, w = img.shape[:2]
        left, right = img[:, :w//2], img[:, w//2:]
        frames_data, source_frame = [(left, right)], (left, right)
    else:
        print(f"Loading video: {args.video}")
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            sys.exit(f"Error: Cannot open {args.video}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ret, first = cap.read()
        if ret:
            h, w = first.shape[:2]
            source_frame = (first[:, :w//2], first[:, w//2:])
        
        # --frames: upper limit of first N frames to consider (default: all)
        # --frame-rate: sampling rate (0.2 = every 5th frame)
        max_frame = min(args.frames, total) if args.frames else total
        step = max(1, int(1.0 / args.frame_rate)) if args.frame_rate > 0 else 1
        
        frame_indices = list(range(0, max_frame, step))
        print(f"  Total frames: {total}, considering first {max_frame}, sampling every {step} (rate={args.frame_rate})")
        print(f"  Using {len(frame_indices)} frames: {frame_indices[:5]}{'...' if len(frame_indices) > 5 else ''}")
        
        for idx in frame_indices:
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
        frames_data = downsample_frames(frames_data, args.scale)
    
    calib = calibrate_with_config(frames_data, args.fov, config, workers=args.workers)
    
    calib.save_json("calibration.json")
    print(f"\nSaved: calibration.json")
    
    if args.output and source_frame:
        result = project_equirectangular(source_frame[0], source_frame[1], calib, args.fov, blend=False)
        cv2.imwrite(args.output, result)
        print(f"Saved: {args.output}")
    
    # Generate output video if requested
    if args.output_video and args.video:
        from projections.fisheye_to_equirect import process_video
        process_video(
            args.video, args.output_video, calib, args.fov,
            blend=True, frame_rate=args.output_frame_rate
        )


if __name__ == '__main__':
    main()
