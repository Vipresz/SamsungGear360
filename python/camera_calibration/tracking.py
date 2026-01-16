"""
Feature tracking functions for dual-fisheye camera calibration.
Uses optical flow to track features across video frames and estimate
calibration parameters from motion patterns.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Optional


class FeatureTracker:
    """Track features across video frames to estimate calibration parameters."""
    
    def __init__(self, max_features: int = 500, quality_level: float = 0.01, min_distance: int = 10):
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        
        # Feature detection
        self.feature_params = dict(
            maxCorners=max_features,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7
        )
        
        # Optical flow params (Lucas-Kanade)
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Track storage
        self.tracks = {}  # track_id -> [(frame_idx, x, y, lens_id), ...]
        self.next_track_id = 0
        self.prev_gray = None
        self.prev_pts = None
        self.prev_ids = None
        
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Dict:
        """Process a single frame, tracking existing features and detecting new ones."""
        height, width = frame.shape[:2]
        is_horizontal = width > height
        lens_w = width // 2 if is_horizontal else width
        lens_h = height if is_horizontal else height // 2
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        tracked_pts = []
        tracked_ids = []
        
        # Track existing features using optical flow
        if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts) > 0:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None, **self.lk_params
            )
            
            # Filter good tracks
            if next_pts is not None:
                for i, (pt, st) in enumerate(zip(next_pts, status)):
                    if st[0] == 1:
                        x, y = pt.ravel()
                        # Check bounds
                        if 0 <= x < width and 0 <= y < height:
                            track_id = self.prev_ids[i]
                            # Determine which lens
                            if is_horizontal:
                                lens_id = 0 if x < lens_w else 1
                                local_x = x if lens_id == 0 else x - lens_w
                            else:
                                lens_id = 0 if y < lens_h else 1
                                local_x = x
                            
                            self.tracks[track_id].append((frame_idx, float(x), float(y), lens_id))
                            tracked_pts.append([x, y])
                            tracked_ids.append(track_id)
        
        # Detect new features in areas with few tracks
        mask = np.ones(gray.shape, dtype=np.uint8) * 255
        for pt in tracked_pts:
            cv2.circle(mask, (int(pt[0]), int(pt[1])), self.min_distance * 2, 0, -1)
        
        new_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        
        if new_pts is not None:
            for pt in new_pts:
                x, y = pt.ravel()
                track_id = self.next_track_id
                self.next_track_id += 1
                
                # Determine which lens
                if is_horizontal:
                    lens_id = 0 if x < lens_w else 1
                else:
                    lens_id = 0 if y < lens_h else 1
                
                self.tracks[track_id] = [(frame_idx, float(x), float(y), lens_id)]
                tracked_pts.append([x, y])
                tracked_ids.append(track_id)
        
        # Update state
        self.prev_gray = gray
        self.prev_pts = np.array(tracked_pts, dtype=np.float32).reshape(-1, 1, 2) if tracked_pts else None
        self.prev_ids = tracked_ids
        
        return {
            'num_tracks': len(tracked_pts),
            'total_tracks': len(self.tracks)
        }
    
    def get_long_tracks(self, min_length: int = 10) -> Dict:
        """Get tracks that span at least min_length frames."""
        return {tid: pts for tid, pts in self.tracks.items() if len(pts) >= min_length}
    
    def get_overlap_tracks(self, lens_w: int, overlap_ratio: float = 0.15) -> Dict:
        """Get tracks that are in the overlap region between lenses."""
        overlap_tracks = {}
        overlap_x_min = lens_w * (1 - overlap_ratio)
        overlap_x_max = lens_w * overlap_ratio
        
        for tid, pts in self.tracks.items():
            if len(pts) < 5:
                continue
            in_overlap = False
            for _, x, y, lens_id in pts:
                local_x = x if lens_id == 0 else x - lens_w
                if local_x > overlap_x_min or local_x < overlap_x_max:
                    in_overlap = True
                    break
            if in_overlap:
                overlap_tracks[tid] = pts
        
        return overlap_tracks


def compute_motion_consistency(tracks: Dict, lens_w: int, lens_h: int, lens_id: int,
                               candidate_cx: float, candidate_cy: float) -> float:
    """Compute how consistently features move relative to a candidate center."""
    # Group tracks by radius from candidate center
    radius_bins = {}
    bin_size = min(lens_w, lens_h) / 10
    
    for tid, pts in tracks.items():
        lens_pts = [(f, x, y) for f, x, y, lid in pts if lid == lens_id]
        if len(lens_pts) < 3:
            continue
        
        positions = []
        for f, x, y in lens_pts:
            local_x = x if lens_id == 0 else x - lens_w
            positions.append((local_x, y))
        
        positions = np.array(positions)
        
        dx = positions[-1, 0] - positions[0, 0]
        dy = positions[-1, 1] - positions[0, 1]
        motion_mag = np.sqrt(dx*dx + dy*dy)
        
        if motion_mag < 2:
            continue
        
        mid_x = np.mean(positions[:, 0]) - candidate_cx
        mid_y = np.mean(positions[:, 1]) - candidate_cy
        radius = np.sqrt(mid_x*mid_x + mid_y*mid_y)
        
        motion_angle = np.arctan2(dy, dx)
        radial_angle = np.arctan2(mid_y, mid_x)
        angle_diff = motion_angle - radial_angle
        
        bin_idx = int(radius / bin_size)
        if bin_idx not in radius_bins:
            radius_bins[bin_idx] = []
        radius_bins[bin_idx].append((radius, motion_mag, angle_diff))
    
    total_variance = 0
    total_count = 0
    
    for bin_idx, motions in radius_bins.items():
        if len(motions) < 5:
            continue
        
        angle_diffs = [m[2] for m in motions]
        
        sin_sum = sum(np.sin(a) for a in angle_diffs)
        cos_sum = sum(np.cos(a) for a in angle_diffs)
        R = np.sqrt(sin_sum**2 + cos_sum**2) / len(angle_diffs)
        variance = 1 - R
        
        total_variance += variance * len(motions)
        total_count += len(motions)
    
    if total_count == 0:
        return float('inf')
    
    return total_variance / total_count


def estimate_lens_center_from_motion_deviation(tracks: Dict, lens_w: int, lens_h: int, 
                                                lens_id: int) -> Tuple[float, float]:
    """Estimate lens center by comparing center vs. outer feature motion."""
    best_center = (0.5, 0.5)
    best_score = float('inf')
    
    for cx_ratio in np.arange(0.4, 0.6, 0.02):
        for cy_ratio in np.arange(0.4, 0.6, 0.02):
            candidate_cx = cx_ratio * lens_w
            candidate_cy = cy_ratio * lens_h
            
            score = compute_motion_consistency(tracks, lens_w, lens_h, lens_id, 
                                               candidate_cx, candidate_cy)
            
            if score < best_score:
                best_score = score
                best_center = (cx_ratio, cy_ratio)
    
    return best_center


def estimate_lens_center_from_tracks(tracks: Dict, lens_w: int, lens_h: int, lens_id: int) -> Tuple[float, float]:
    """Estimate lens center using motion consistency across all directions."""
    center_x, center_y = estimate_lens_center_from_motion_deviation(tracks, lens_w, lens_h, lens_id)
    print(f"    Lens {lens_id}: motion-consistency center = ({center_x:.4f}, {center_y:.4f})")
    return float(center_x), float(center_y)


def estimate_rotation_from_tracks(tracks: Dict, lens_w: int, lens_h: int) -> Tuple[float, float, float]:
    """Estimate relative rotation between lenses from overlapping feature tracks."""
    lens0_motions = []
    lens1_motions = []
    
    for tid, pts in tracks.items():
        if len(pts) < 5:
            continue
        
        for i in range(1, len(pts)):
            f0, x0, y0, lid0 = pts[i-1]
            f1, x1, y1, lid1 = pts[i]
            
            if f1 - f0 == 1 and lid0 == lid1:
                dx, dy = x1 - x0, y1 - y0
                mag = np.sqrt(dx*dx + dy*dy)
                if mag > 0.5:
                    angle = np.arctan2(dy, dx)
                    if lid0 == 0:
                        lens0_motions.append(angle)
                    else:
                        lens1_motions.append(angle)
    
    if len(lens0_motions) < 20 or len(lens1_motions) < 20:
        return 0.0, 0.0, 0.0
    
    avg_angle0 = np.median(lens0_motions)
    avg_angle1 = np.median(lens1_motions)
    
    roll_estimate = (avg_angle1 - avg_angle0) * 0.1
    
    return 0.0, 0.0, float(roll_estimate)


def estimate_distortion_from_tracks(tracks: Dict, lens_w: int, lens_h: int, 
                                    center_x: float, center_y: float, lens_id: int) -> Tuple[float, float]:
    """Estimate radial distortion by comparing center vs. outer feature motion."""
    cx = center_x * lens_w
    cy = center_y * lens_h
    max_radius = min(lens_w, lens_h) / 2
    
    center_motions = []
    for tid, pts in tracks.items():
        lens_pts = [(f, x, y) for f, x, y, lid in pts if lid == lens_id]
        if len(lens_pts) < 3:
            continue
        
        positions = []
        for f, x, y in lens_pts:
            local_x = x if lens_id == 0 else x - lens_w
            positions.append((local_x, y))
        
        positions = np.array(positions)
        mid_x = np.mean(positions[:, 0])
        mid_y = np.mean(positions[:, 1])
        radius = np.sqrt((mid_x - cx)**2 + (mid_y - cy)**2)
        
        if radius < max_radius * 0.3:
            dx = positions[-1, 0] - positions[0, 0]
            dy = positions[-1, 1] - positions[0, 1]
            mag = np.sqrt(dx*dx + dy*dy)
            if mag > 1:
                center_motions.append((dx, dy, len(lens_pts)))
    
    if len(center_motions) < 20:
        return 0.0, 0.0
    
    total_weight = sum(m[2] for m in center_motions)
    avg_dx = sum(m[0] * m[2] for m in center_motions) / total_weight
    avg_dy = sum(m[1] * m[2] for m in center_motions) / total_weight
    avg_mag = np.sqrt(avg_dx**2 + avg_dy**2)
    
    if avg_mag < 3:
        return 0.0, 0.0
    
    deviations_by_radius = []
    
    for tid, pts in tracks.items():
        lens_pts = [(f, x, y) for f, x, y, lid in pts if lid == lens_id]
        if len(lens_pts) < 3:
            continue
        
        positions = []
        for f, x, y in lens_pts:
            local_x = x if lens_id == 0 else x - lens_w
            positions.append((local_x, y))
        
        positions = np.array(positions)
        mid_x = np.mean(positions[:, 0])
        mid_y = np.mean(positions[:, 1])
        radius = np.sqrt((mid_x - cx)**2 + (mid_y - cy)**2)
        norm_radius = radius / max_radius
        
        if norm_radius < 0.3 or norm_radius > 0.95:
            continue
        
        dx = positions[-1, 0] - positions[0, 0]
        dy = positions[-1, 1] - positions[0, 1]
        mag = np.sqrt(dx*dx + dy*dy)
        
        if mag < 1:
            continue
        
        expected_mag = avg_mag
        mag_ratio = mag / expected_mag
        
        deviations_by_radius.append((norm_radius, mag_ratio))
    
    if len(deviations_by_radius) < 50:
        return 0.0, 0.0
    
    deviations_by_radius = np.array(deviations_by_radius)
    
    inner_mask = (deviations_by_radius[:, 0] >= 0.3) & (deviations_by_radius[:, 0] < 0.5)
    outer_mask = (deviations_by_radius[:, 0] >= 0.6) & (deviations_by_radius[:, 0] < 0.9)
    
    if np.sum(inner_mask) < 10 or np.sum(outer_mask) < 10:
        return 0.0, 0.0
    
    inner_ratio = np.median(deviations_by_radius[inner_mask, 1])
    outer_ratio = np.median(deviations_by_radius[outer_mask, 1])
    
    ratio_diff = outer_ratio - inner_ratio
    p2_estimate = ratio_diff * 0.03
    p2_estimate = float(np.clip(p2_estimate, -0.05, 0.05))
    
    print(f"    Lens {lens_id}: inner_ratio={inner_ratio:.3f}, outer_ratio={outer_ratio:.3f}, p2={p2_estimate:.4f}")
    
    return 0.0, p2_estimate
