"""
Camera Calibration Package for Dual-Fisheye 360° Cameras

Modules:
- projection: Fisheye to equirectangular projection
- lens_detection: Lens boundary and center detection
- masks: Overlap region masking utilities
- tracking: Video-based feature tracking for calibration
- least_squares_stitch: Spherical least-squares calibration

Usage:
    from camera_calibration import project_fisheye_to_equirectangular
    from camera_calibration.lens_detection import detect_lens_center
"""

from .projection import (
    fisheye_to_equirect_half,
    fisheye_to_equirect_dual,
    project_fisheye_to_equirectangular,
    apply_calibration_to_video,
    LENS_FOV_DEG,
    THETA_MAX
)

from .lens_detection import (
    detect_lens_center_advanced,
    detect_lens_center,
    detect_boundary_points,
    fit_circle_ransac
)

from .masks import (
    make_ring_mask,
    make_overlap_ring_mask,
    extract_ring_region,
    make_feature_tracking_mask
)

from .tracking import (
    FeatureTracker,
    estimate_rotation_from_tracks,
    estimate_distortion_from_tracks
)

from .least_squares_stitch import (
    FisheyeParams,
    collect_matches,
    calibrate,
    compute_angular_errors,
    filter_outliers,
    effective_fov
)

__version__ = '1.0.0'
__all__ = [
    # Projection
    'fisheye_to_equirect_half', 'fisheye_to_equirect_dual',
    'project_fisheye_to_equirectangular', 'apply_calibration_to_video',
    'LENS_FOV_DEG', 'THETA_MAX',
    # Lens detection
    'detect_lens_center_advanced', 'detect_lens_center',
    'detect_boundary_points', 'fit_circle_ransac',
    # Masks
    'make_ring_mask', 'make_overlap_ring_mask', 
    'extract_ring_region', 'make_feature_tracking_mask',
    # Tracking
    'FeatureTracker', 'estimate_rotation_from_tracks', 'estimate_distortion_from_tracks',
    # Least-squares
    'FisheyeParams', 'collect_matches', 'calibrate', 
    'compute_angular_errors', 'filter_outliers', 'effective_fov'
]
