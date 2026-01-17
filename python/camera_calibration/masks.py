"""Masking utilities for dual-fisheye overlap region detection."""
import cv2
import numpy as np
from typing import List, Tuple, Optional

# Angles normalized to [0°, 360°]: 0°=right, 90°=bottom, 180°=left, 270°=top
EXCLUDE_TOP_BOTTOM_90 = [(225, 315), (45, 135)]  # top (270°±45°) and bottom (90°±45°)
EXCLUDE_TOP_BOTTOM_60 = [(240, 300), (60, 120)]  # top (270°±30°) and bottom (90°±30°)


def make_ring_mask(h: int, w: int, cx: float, cy: float,
                   inner_r: float, outer_r: float,
                   exclude_angles: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
    """
    Create ring mask with optional angular exclusions.
    
    Args:
        h, w: Image dimensions
        cx, cy: Center in pixels
        inner_r, outer_r: Ring radii in pixels
        exclude_angles: Angle ranges to exclude [(start, end), ...]
    
    Returns:
        uint8 mask (255=valid, 0=excluded)
    """
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dist = np.sqrt((X.copy() - cx)**2 + (Y.copy() - cy)**2)
    # Negate Y because image coords have Y increasing downward
    phi_rad = np.arctan2(-(Y.copy() - cy), X.copy() - cx)  # [-π, π]
    phi_deg = np.degrees(phi_rad) % 360  # [0, 360)
    mask = (dist >= inner_r) & (dist <= outer_r)
    if exclude_angles:
        for start, end in exclude_angles:
                excl = (phi_deg >= start) & (phi_deg <= end)
                mask &= ~excl
    return mask.astype(np.uint8) * 255


def make_overlap_ring_mask(h: int, w: int, cx: float, cy: float, radius: float,
                           inner_ratio: float = 0.90, outer_ratio: float = 0.98,
                           exclude_angles: Optional[List[Tuple[float, float]]] = None) -> np.ndarray:
    """
    Create overlap ring mask for dual-fisheye calibration.
    
    This is the mask used during optimization to find feature matches
    in the overlap region between two fisheye lenses.
    
    Args:
        h, w: Image dimensions
        cx, cy: Lens center in pixels
        radius: Lens radius in pixels
        inner_ratio: Inner ring boundary as fraction of radius (default 0.75)
        outer_ratio: Outer ring boundary as fraction of radius (default 0.98)
        exclude_angles: Angle ranges to exclude [(start, end), ...] in degrees
                       Defaults to EXCLUDE_TOP_BOTTOM_90 to avoid tripod/handle
    
    Returns:
        uint8 mask (255=valid, 0=excluded)
    """
    if exclude_angles is None:
        exclude_angles = EXCLUDE_TOP_BOTTOM_90
    
    inner_r = inner_ratio * radius
    outer_r = outer_ratio * radius
    
    return make_ring_mask(h, w, cx, cy, inner_r, outer_r, exclude_angles)


def make_rectangular_overlap_mask(h: int, w: int, 
                                   width_ratio: float = 0.08,
                                   height_ratio: float = 1.0) -> np.ndarray:
    """
    Create rectangular overlap mask for dual-fisheye equirectangular images.
    
    Creates vertical bands at ALL four overlap regions where opposing fisheye 
    lenses overlap in equirectangular space:
    1. Left edge (wrap-around seam)
    2. Middle seam (where left/right halves meet)
    3. Right edge (wrap-around seam)
    
    For dual-fisheye 360° cameras projecting to width w:
    - Left lens (0 to w/2): overlaps at its right edge (middle) and left edge (wrap)
    - Right lens (w/2 to w): overlaps at its left edge (middle) and right edge (wrap)
    
    Args:
        h, w: Image dimensions (equirectangular space, full 360° width)
        width_ratio: Width of overlap band as fraction of HALF width (default 0.08 = 8%)
        height_ratio: Height of overlap band as fraction of image height (default 1.0 = 100%)
    
    Returns:
        uint8 mask (255=valid, 0=excluded)
        
    Example:
        For a 2000x1000 equirect image with width_ratio=0.08:
        - Left wrap band: columns 0-80 (8% of half-width = 8% of 1000)
        - Middle seam band: columns 920-1080 (±8% around center at 1000)
        - Right wrap band: columns 1920-2000 (last 8% of half-width)
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Calculate band dimensions based on half-width (each lens covers half)
    half_w = w // 2
    overlap_width = int(half_w * width_ratio)
    overlap_height = int(h * height_ratio)
    h_start = (h - overlap_height) // 2
    h_end = h_start + overlap_height
    
    # Left edge band (wrap-around: left of left image, right of right image)
    mask[h_start:h_end, :overlap_width] = 255
    
    # Middle seam band (right of left image, left of right image)
    middle = w // 2
    mask[h_start:h_end, middle - overlap_width:middle + overlap_width] = 255
    
    # Right edge band (wrap-around: right of right image, left of left image)
    mask[h_start:h_end, -overlap_width:] = 255
    
    return mask

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Test mask visualizations')
    parser.add_argument('--rect', action='store_true', help='Test rectangular overlap mask (for equirect images)')
    parser.add_argument('--image', type=str, help='Input image file path')
    args = parser.parse_args()
    
    if args.rect:
        # Test rectangular overlap mask
        print("Testing rectangular overlap mask for equirectangular images")
        
        # Check if an image file is provided
        if args.image:
            # Use provided image
            test_img = cv2.imread(args.image)
            if test_img is None:
                print(f"Error: Could not load image: {args.image}")
                sys.exit(1)
            print(f"Loaded image: {args.image}")
        else:
            # Create test image (simulating 360° equirect)
            h, w = 1000, 2000
            test_img = np.zeros((h, w, 3), dtype=np.uint8)
            test_img[:, :w//2] = (100, 100, 255)  # Left half (blue)
            test_img[:, w//2:] = (255, 100, 100)  # Right half (red)
            print("Using synthetic test image")
        
        h, w = test_img.shape[:2]
        
        # Create rectangular overlap mask
        mask = make_rectangular_overlap_mask(h, w, width_ratio=0.08, height_ratio=.5)
        
        print(f"Image: {w}x{h}")
        print(f"Overlap width: {int((w/2) * 0.08)} pixels per band ({0.08*100}% of half-width)")
        print(f"Overlap height: {int(h * 0.5)} pixels ({0.5*100}% of height)")
        print(f"Mask pixels included: {np.sum(mask > 0)}")
        print(f"Three bands: left edge, middle seam, right edge")
        
        # Overlay mask on image for visualization
        overlay = test_img.copy()
        overlay[mask > 0] = (0, 255, 0)  # Green = overlap region
        blended = cv2.addWeighted(test_img, 0.5, overlay, 0.5, 0)
        
        cv2.imshow("rectangular mask (white=overlap)", mask)
        cv2.imshow("overlay (green=overlap bands)", blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # Test ring mask on fisheye image
        if args.image:
            img_path = args.image
        else:
            img_path = "fisheye.png"
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load image: {img_path}")
            print("\nUsage:")
            print("  python masks.py --image image.png     # Test ring mask on fisheye image")
            print("  python masks.py --rect --image image.png  # Test rectangular overlap mask on equirect image")
            sys.exit(1)
        else:
            print(f"Loaded image: {img_path}")
            h, w = img.shape[:2]
            # For dual-fisheye side-by-side: each lens is h×h, left lens center is at (h/2, h/2)
            lens_size = min(h, w)
            cx, cy = lens_size // 2, lens_size // 2
            radius = lens_size // 2
            
            print(f"Image: {w}x{h}, center=({cx}, {cy}), radius={radius}")
            print(f"Excluding angles: {EXCLUDE_TOP_BOTTOM_90}")
            
            # Use same mask as calibrate_stitching.py optimization
            mask = make_overlap_ring_mask(h, w, cx, cy, radius)
            
            print(f"Mask pixels included: {np.sum(mask > 0)}")
            
            # Overlay mask on image for visualization
            overlay = img.copy()
            overlay[mask > 0] = (0, 255, 0)  # Green = included region
            blended = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)
            
            cv2.imshow("ring mask (white=included)", mask)
            cv2.imshow("overlay (green=included)", blended)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
