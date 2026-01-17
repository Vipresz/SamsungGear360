"""
Camera Calibration Package for Dual-Fisheye 360° Cameras

Subpackages:
- calib: Calibration configuration and creation
- projections: Fisheye to equirectangular projection
- solvers: Calibration optimization solvers

Usage:
    from camera_calibration.calib import CameraCalibration, LensCalibration
    from camera_calibration.projections import fisheye_to_equirect_calibrated
    from camera_calibration.solvers import calibrate_joint
"""

from .calib.calibration_config import CameraCalibration, LensCalibration
from .calib.masks import (
    make_ring_mask,
    make_overlap_ring_mask,
    extract_ring_region,
    make_feature_tracking_mask
)
from .projections.fisheye_to_equirect import (
    fisheye_to_equirect_calibrated,
    mask_fisheye_circle,
    blend_dual_patches
)
from .solvers.tracking import (
    FeatureTracker,
    estimate_rotation_from_tracks,
    estimate_distortion_from_tracks
)

__version__ = '1.0.0'
__all__ = [
    # Calibration config
    'CameraCalibration', 'LensCalibration',
    # Masks
    'make_ring_mask', 'make_overlap_ring_mask', 
    'extract_ring_region', 'make_feature_tracking_mask',
    # Projection
    'fisheye_to_equirect_calibrated', 'mask_fisheye_circle', 'blend_dual_patches',
    # Tracking
    'FeatureTracker', 'estimate_rotation_from_tracks', 'estimate_distortion_from_tracks',
]
