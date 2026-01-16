# Camera Calibration for Dual-Fisheye Stitching

Tools for calibrating optimal stitching parameters from dual-fisheye video/image files (e.g., Samsung Gear 360).

## Requirements

```bash
pip install opencv-python numpy scipy
```

## Quick Start

```bash
# Calibrate from video using feature tracking (recommended)
python calibrate_stitching.py --video input.mp4 --tracking --apply_calibration --output_image result.png

# Calibrate from a single image
python calibrate_stitching.py --image fisheye.png --apply_calibration --output_image result.png
```

## Commands

### Video Calibration (Feature Tracking - Recommended)

The `--tracking` option uses optical flow to track features across frames, providing more accurate calibration:

```bash
# Full calibration pipeline with tracking
python calibrate_stitching.py --video video.mp4 --tracking --apply_calibration --output_image output.png

# Specify number of frames to analyze (default: 100)
python calibrate_stitching.py --video video.mp4 --tracking --frames 200 --apply_calibration --output_image output.png

# Save calibration to JSON
python calibrate_stitching.py --video video.mp4 --tracking --output params.json
```

### Video Calibration (Frame-by-Frame)

```bash
# Analyze multiple frames without tracking
python calibrate_stitching.py --video video.mp4 --frames 20 --apply_calibration --output_image output.png
```

### Image Calibration

```bash
# Calibrate and project from a single image
python calibrate_stitching.py --image fisheye.png --apply_calibration --output_image equirect.png

# Use existing calibration file
python calibrate_stitching.py --image fisheye.png --calibration_file params.json --apply_calibration --output_image output.png
```

### Generate Output Video

```bash
# Generate equirectangular video
python calibrate_stitching.py --video input.mp4 --tracking --apply_calibration --output_video output.mp4

# Custom resolution and FOV
python calibrate_stitching.py --video input.mp4 --tracking --apply_calibration --output_video output.mp4 \
    --output_width 3840 --output_height 1920 --fov 180
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--video <file>` | Input video file (MP4, etc.) |
| `--image <file>` | Input image file (PNG, JPG, etc.) |
| `--tracking` | Use feature tracking for calibration (recommended for video) |
| `--frames <n>` | Number of frames to analyze (default: 100 with tracking, 10 without) |
| `--apply_calibration` | Apply calibration parameters to output |
| `--calibration_file <file>` | Load calibration from JSON file |
| `--output <file>` | Save calibration parameters to JSON |
| `--output_image <file>` | Output equirectangular image |
| `--output_video <file>` | Output equirectangular video |
| `--output_width <n>` | Output width (default: same as input) |
| `--output_height <n>` | Output height (default: same as input) |
| `--fov <degrees>` | Vertical FOV for output (default: 180) |

## Calibration Pipeline (with --tracking)

The tracking-based calibration performs these steps:

1. **STEP 1: Detect Lens Parameters**
   - Lens centers and radii
   - Polynomial distortion coefficients (p1, p2) from feature motion

2. **STEP 2: Adjust FOV (First Pass)**
   - Optimizes FOV to minimize seam error
   - Starting from nominal 195° lens FOV

3. **STEP 3: Adjust Rotation and Offsets**
   - Per-lens yaw, pitch, roll optimization
   - Per-lens X/Y offset optimization

4. **STEP 4: Adjust FOV (Final Pass)**
   - Re-optimizes FOV with rotation/offsets applied

## Output Parameters

The calibration produces these per-lens parameters:

| Parameter | Description |
|-----------|-------------|
| `lens1CenterX/Y` | Lens 1 optical center (normalized 0-1) |
| `lens2CenterX/Y` | Lens 2 optical center (normalized 0-1) |
| `lens1FOV`, `lens2FOV` | FOV scale (1.0 = 180°) |
| `lens1P1`, `lens1P2` | Lens 1 distortion coefficients |
| `lens2P1`, `lens2P2` | Lens 2 distortion coefficients |
| `lens1RotationYaw/Pitch/Roll` | Lens 1 rotation (radians) |
| `lens2RotationYaw/Pitch/Roll` | Lens 2 rotation (radians) |
| `alignmentOffset1X/Y` | Lens 1 alignment offset |
| `alignmentOffset2X/Y` | Lens 2 alignment offset |

## Example Output

```
============================================================
CALIBRATION COMPLETE
============================================================
  SUMMARY:
  --------
  Lens Centers: L1=(0.5000, 0.5001), L2=(0.5010, 0.4984)
  Aberration:   L1=(p1=0.0000, p2=-0.0048), L2=(p1=0.0000, p2=-0.0129)
  FOV:          L1=1.0653 (191.8°), L2=1.0653 (191.8°)
  Rotation L1:  yaw=0.00°, pitch=0.00°, roll=0.00°
  Rotation L2:  yaw=-1.00°, pitch=0.80°, roll=-0.40°
  Offsets L1:   X=0.0000, Y=0.0000
  Offsets L2:   X=0.0000, Y=0.0140
```

## Using with C++ Viewer

Export calibration to TOML format for the C++ viewer:

```bash
# The Python calibration output can be converted to calibration.toml format
# See ../cpp/calibration.toml for the expected format
```

Then run the C++ viewer with:

```bash
./gear360_viewer --calibration calibration.toml --equirectangular video.mp4
```
