"""
Feature tracking for video-based dual-fisheye calibration.
Tracks features across frames to estimate lens parameters from motion patterns.
"""
import cv2
import numpy as np


class FeatureTracker:
    """Track features across video frames using optical flow."""
    
    def __init__(self, max_features=500, quality=0.01, min_distance=10):
        self.feature_params = dict(maxCorners=max_features, qualityLevel=quality,
                                   minDistance=min_distance, blockSize=7)
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.tracks = {}  # track_id -> [(frame_idx, x, y, lens_id), ...]
        self.next_id = 0
        self.prev_gray = None
        self.prev_pts = None
        self.prev_ids = None
    
    def process_frame(self, frame, frame_idx):
        """Process frame, returns stats dict."""
        h, w = frame.shape[:2]
        is_horizontal = w > h
        lens_w = w // 2 if is_horizontal else w
        lens_h = h if is_horizontal else h // 2
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        tracked_pts, tracked_ids = [], []
        
        # Track existing features
        if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts) > 0:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, self.prev_pts, None, **self.lk_params)
            
            if next_pts is not None:
                for i, (pt, st) in enumerate(zip(next_pts, status)):
                    if st[0] == 1:
                        x, y = pt.ravel()
                        if 0 <= x < w and 0 <= y < h:
                            tid = self.prev_ids[i]
                            lens_id = (0 if x < lens_w else 1) if is_horizontal else (0 if y < lens_h else 1)
                            self.tracks[tid].append((frame_idx, float(x), float(y), lens_id))
                            tracked_pts.append([x, y])
                            tracked_ids.append(tid)
        
        # Detect new features (excluding areas near existing tracks)
        from masks import make_feature_tracking_mask
        mask = make_feature_tracking_mask(gray, tracked_pts, min_distance=20)
        
        new_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        if new_pts is not None:
            for pt in new_pts:
                x, y = pt.ravel()
                tid = self.next_id
                self.next_id += 1
                lens_id = (0 if x < lens_w else 1) if is_horizontal else (0 if y < lens_h else 1)
                self.tracks[tid] = [(frame_idx, float(x), float(y), lens_id)]
                tracked_pts.append([x, y])
                tracked_ids.append(tid)
        
        self.prev_gray = gray
        self.prev_pts = np.array(tracked_pts, dtype=np.float32).reshape(-1, 1, 2) if tracked_pts else None
        self.prev_ids = tracked_ids
        
        return {'num_tracks': len(tracked_pts), 'total_tracks': len(self.tracks)}
    
    def get_long_tracks(self, min_length=10):
        """Get tracks spanning at least min_length frames."""
        return {tid: pts for tid, pts in self.tracks.items() if len(pts) >= min_length}


def estimate_distortion_from_tracks(tracks, lens_w, lens_h, center_x, center_y, lens_id):
    """Estimate radial distortion from track motion patterns. Returns (p1, p2)."""
    cx, cy = center_x * lens_w, center_y * lens_h
    max_r = min(lens_w, lens_h) / 2
    
    # Get center motion reference
    center_motions = []
    for pts in tracks.values():
        lens_pts = [(f, x, y) for f, x, y, lid in pts if lid == lens_id]
        if len(lens_pts) < 3:
            continue
        
        positions = np.array([(x if lens_id == 0 else x - lens_w, y) for f, x, y in lens_pts])
        mid_x, mid_y = np.mean(positions, axis=0)
        r = np.sqrt((mid_x - cx)**2 + (mid_y - cy)**2)
        
        if r < max_r * 0.3:
            dx = positions[-1, 0] - positions[0, 0]
            dy = positions[-1, 1] - positions[0, 1]
            if np.sqrt(dx**2 + dy**2) > 1:
                center_motions.append((dx, dy, len(lens_pts)))
    
    if len(center_motions) < 20:
        return 0.0, 0.0
    
    total_w = sum(m[2] for m in center_motions)
    avg_dx = sum(m[0] * m[2] for m in center_motions) / total_w
    avg_dy = sum(m[1] * m[2] for m in center_motions) / total_w
    avg_mag = np.sqrt(avg_dx**2 + avg_dy**2)
    
    if avg_mag < 3:
        return 0.0, 0.0
    
    # Compare inner vs outer motion
    deviations = []
    for pts in tracks.values():
        lens_pts = [(f, x, y) for f, x, y, lid in pts if lid == lens_id]
        if len(lens_pts) < 3:
            continue
        
        positions = np.array([(x if lens_id == 0 else x - lens_w, y) for f, x, y in lens_pts])
        mid_x, mid_y = np.mean(positions, axis=0)
        r = np.sqrt((mid_x - cx)**2 + (mid_y - cy)**2)
        norm_r = r / max_r
        
        if 0.3 <= norm_r <= 0.95:
            dx = positions[-1, 0] - positions[0, 0]
            dy = positions[-1, 1] - positions[0, 1]
            mag = np.sqrt(dx**2 + dy**2)
            if mag > 1:
                deviations.append((norm_r, mag / avg_mag))
    
    if len(deviations) < 50:
        return 0.0, 0.0
    
    devs = np.array(deviations)
    inner = devs[(devs[:, 0] >= 0.3) & (devs[:, 0] < 0.5)]
    outer = devs[(devs[:, 0] >= 0.6) & (devs[:, 0] < 0.9)]
    
    if len(inner) < 10 or len(outer) < 10:
        return 0.0, 0.0
    
    ratio_diff = np.median(outer[:, 1]) - np.median(inner[:, 1])
    p2 = float(np.clip(ratio_diff * 0.03, -0.05, 0.05))
    
    return 0.0, p2


def estimate_rotation_from_tracks(tracks, lens_w, lens_h):
    """Estimate relative rotation from track motion. Returns (yaw, pitch, roll)."""
    motions = {0: [], 1: []}
    
    for pts in tracks.values():
        if len(pts) < 5:
            continue
        for i in range(1, len(pts)):
            f0, x0, y0, lid0 = pts[i - 1]
            f1, x1, y1, lid1 = pts[i]
            if f1 - f0 == 1 and lid0 == lid1:
                dx, dy = x1 - x0, y1 - y0
                if np.sqrt(dx**2 + dy**2) > 0.5:
                    motions[lid0].append(np.arctan2(dy, dx))
    
    if len(motions[0]) < 20 or len(motions[1]) < 20:
        return 0.0, 0.0, 0.0
    
    angle0, angle1 = np.median(motions[0]), np.median(motions[1])
    roll = (angle1 - angle0) * 0.1
    
    return 0.0, 0.0, float(roll)
