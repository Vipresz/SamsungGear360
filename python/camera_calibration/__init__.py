"""
Camera Calibration Package for Dual-Fisheye 360° Cameras

This package provides tools for calibrating and stitching dual-fisheye images
into equirectangular projections.

Modules:
- lens_detection: Lens boundary and center detection
- alignment: Seam alignment and optimization
- fov_optimization: FOV and distortion optimization
- rotation_estimation: Rotation estimation from feature matching
- projection: Fisheye to equirectangular projection

Usage:
    from camera_calibration import project_fisheye_to_equirectangular
    from camera_calibration.lens_detection import detect_lens_center_advanced
"""

from .projection import (
    fisheye_to_equirect_half,
    fisheye_to_equirect_dual,
    project_fisheye_to_equirectangular,
    apply_calibration_to_video
)

from .lens_detection import (
    detect_lens_center_advanced,
    detect_lens_center,
    detect_lens_boundary_points,
    fit_circle_ransac
)

from .alignment import (
    compute_seam_alignment_error,
    extract_overlap_features,
    optimize_alignment_parameters
)

from .fov_optimization import (
    estimate_fov_from_coverage,
    optimize_fov,
    optimize_distortion
)

from .rotation_estimation import (
    extract_ring_region,
    estimate_lens_rotation_from_rings,
    estimate_lens_rotation,
    compute_alignment_offsets_equirect
)

from .tracking import (
    FeatureTracker,
    estimate_rotation_from_tracks,
    estimate_distortion_from_tracks,
    estimate_lens_center_from_tracks,
    compute_motion_consistency
)

from .seam_refinement import (
    refine_fov_from_seam,
    optimize_y_offset,
    refine_roll_from_seam,
    refine_rotation_from_seam
)

__version__ = '1.0.0'
__all__ = [
    # Projection
    'fisheye_to_equirect_half',
    'fisheye_to_equirect_dual',
    'project_fisheye_to_equirectangular',
    'apply_calibration_to_video',
    # Lens detection
    'detect_lens_center_advanced',
    'detect_lens_center',
    'detect_lens_boundary_points',
    'fit_circle_ransac',
    # Alignment
    'compute_seam_alignment_error',
    'extract_overlap_features',
    'optimize_alignment_parameters',
    # FOV optimization
    'estimate_fov_from_coverage',
    'optimize_fov',
    'optimize_distortion',
    # Rotation estimation
    'extract_ring_region',
    'estimate_lens_rotation_from_rings',
    'estimate_lens_rotation',
    'compute_alignment_offsets_equirect',
    # Feature tracking
    'FeatureTracker',
    'estimate_rotation_from_tracks',
    'estimate_distortion_from_tracks',
    'estimate_lens_center_from_tracks',
    'compute_motion_consistency',
    # Seam refinement
    'refine_fov_from_seam',
    'optimize_y_offset',
    'refine_roll_from_seam',
    'refine_rotation_from_seam'
]
