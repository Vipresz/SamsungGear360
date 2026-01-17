#!/usr/bin/env python3
"""
Create calibration configuration files for dual fisheye cameras.
"""

import argparse
import json
import sys


def create_calibration(
    lens_fov_deg=195.0,
    lens1_center=(0.5, 0.5),
    lens2_center=(0.5, 0.5),
    lens1_k=(0.0, 0.0, 0.0),
    lens2_k=(0.0, 0.0, 0.0),
    lens2_yaw_deg=180.0,
    lens1_offset=(0.0, 0.0),
    lens2_offset=(0.0, 0.0),
    is_horizontal=True
):
    """Create calibration dictionary with specified parameters."""
    import math
    
    fov_scale = lens_fov_deg / 180.0
    lens2_yaw_rad = math.radians(lens2_yaw_deg)
    
    return {
        "parameters": {
            "lens1CenterX": lens1_center[0],
            "lens1CenterY": lens1_center[1],
            "lens1FOV": fov_scale,
            "lens1K1": lens1_k[0],
            "lens1K2": lens1_k[1],
            "lens1K3": lens1_k[2],
            "lens1RotationYaw": 0.0,
            "lens1RotationPitch": 0.0,
            "lens1RotationRoll": 0.0,
            "lens1OffsetX": lens1_offset[0],
            "lens1OffsetY": lens1_offset[1],
            
            "lens2CenterX": lens2_center[0],
            "lens2CenterY": lens2_center[1],
            "lens2FOV": fov_scale,
            "lens2K1": lens2_k[0],
            "lens2K2": lens2_k[1],
            "lens2K3": lens2_k[2],
            "lens2RotationYaw": lens2_yaw_rad,
            "lens2RotationPitch": 0.0,
            "lens2RotationRoll": 0.0,
            "lens2OffsetX": lens2_offset[0],
            "lens2OffsetY": lens2_offset[1],
            
            "isHorizontal": is_horizontal
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Create calibration JSON file for dual fisheye camera',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Samsung Gear 360 default
  %(prog)s -o gear360.json --fov 195
  
  # Ricoh Theta default  
  %(prog)s -o theta.json --fov 180
  
  # Custom with distortion correction
  %(prog)s -o custom.json --fov 195 --k1 -0.1 --k2 0.05
  
  # Offset lens centers
  %(prog)s -o offset.json --lens1-center 0.48 0.52 --lens2-center 0.51 0.49
        """
    )
    
    parser.add_argument('-o', '--output', required=True,
                       help='Output JSON file path')
    parser.add_argument('--fov', type=float, default=195.0,
                       help='Lens field of view in degrees (default: 195 for Gear 360)')
    parser.add_argument('--lens1-center', type=float, nargs=2, default=[0.5, 0.5],
                       metavar=('X', 'Y'),
                       help='Lens 1 center (normalized 0-1, default: 0.5 0.5)')
    parser.add_argument('--lens2-center', type=float, nargs=2, default=[0.5, 0.5],
                       metavar=('X', 'Y'),
                       help='Lens 2 center (normalized 0-1, default: 0.5 0.5)')
    parser.add_argument('--k1', type=float, default=0.0,
                       help='Radial distortion k1 coefficient (default: 0.0)')
    parser.add_argument('--k2', type=float, default=0.0,
                       help='Radial distortion k2 coefficient (default: 0.0)')
    parser.add_argument('--k3', type=float, default=0.0,
                       help='Radial distortion k3 coefficient (default: 0.0)')
    parser.add_argument('--lens2-yaw', type=float, default=180.0,
                       help='Lens 2 yaw rotation in degrees (default: 180)')
    parser.add_argument('--lens1-offset', type=float, nargs=2, default=[0.0, 0.0],
                       metavar=('X', 'Y'),
                       help='Lens 1 alignment offset in pixels (default: 0 0)')
    parser.add_argument('--lens2-offset', type=float, nargs=2, default=[0.0, 0.0],
                       metavar=('X', 'Y'),
                       help='Lens 2 alignment offset in pixels (default: 0 0)')
    parser.add_argument('--vertical', action='store_true',
                       help='Vertical lens layout (default: horizontal)')
    parser.add_argument('--preset', choices=['gear360', 'theta', 'generic'],
                       help='Use preset configuration')
    
    args = parser.parse_args()
    
    # Apply presets if specified
    if args.preset == 'gear360':
        args.fov = 195.0
        print("Using Samsung Gear 360 preset (195° FOV)")
    elif args.preset == 'theta':
        args.fov = 180.0
        print("Using Ricoh Theta preset (180° FOV)")
    elif args.preset == 'generic':
        args.fov = 180.0
        print("Using generic preset (180° FOV, no corrections)")
    
    # Create calibration
    calibration = create_calibration(
        lens_fov_deg=args.fov,
        lens1_center=tuple(args.lens1_center),
        lens2_center=tuple(args.lens2_center),
        lens1_k=(args.k1, args.k2, args.k3),
        lens2_k=(args.k1, args.k2, args.k3),
        lens2_yaw_deg=args.lens2_yaw,
        lens1_offset=tuple(args.lens1_offset),
        lens2_offset=tuple(args.lens2_offset),
        is_horizontal=not args.vertical
    )
    
    # Save to file
    with open(args.output, 'w') as f:
        json.dump(calibration, f, indent=2)
    
    print(f"✓ Created calibration file: {args.output}")
    print(f"  Lens FOV: {args.fov}° (scale: {args.fov/180.0:.4f})")
    print(f"  Lens 1 center: ({args.lens1_center[0]:.3f}, {args.lens1_center[1]:.3f})")
    print(f"  Lens 2 center: ({args.lens2_center[0]:.3f}, {args.lens2_center[1]:.3f})")
    print(f"  Distortion: k1={args.k1:.4f}, k2={args.k2:.4f}, k3={args.k3:.4f}")
    print(f"  Lens 2 yaw: {args.lens2_yaw}°")
    print(f"  Layout: {'horizontal' if not args.vertical else 'vertical'}")
    
    print(f"\nTest with:")
    print(f"  python3 fisheye_to_equirect.py -i input.jpg -o output.jpg -c {args.output}")


if __name__ == '__main__':
    main()
