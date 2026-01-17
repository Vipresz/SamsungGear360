# Samsung Gear 360 Camera Calibration

Tools for calibrating dual-fisheye 360° cameras and converting to equirectangular projection.

## Directory Structure

```
camera_calibration/
├── calib/                  # Calibration configuration
│   ├── calibration_config.py   # Data classes (CameraCalibration, LensCalibration)
│   ├── create_calibration.py   # Create calibration JSON files
│   └── masks.py                # Fisheye circle masking utilities
├── projections/            # Image projection
│   └── fisheye_to_equirect.py  # Dual-fisheye → equirectangular
├── solvers/                # Calibration optimizers
│   ├── calibrate_adjoint.py    # Joint optimization (all params at once)
│   ├── calibrate_stepwise.py   # Stepwise optimization (FOV grid search)
│   └── tracking.py             # Feature tracking between frames
├── images/                 # Sample images
└── __init__.py
```

## Usage

**Run all commands from the `python/` directory:**

```bash
cd /path/to/SamsungGear360/python
```

### Calibration (Joint Optimizer)

Optimizes all lens parameters simultaneously:

```bash
# Basic usage
python -m camera_calibration.solvers.calibrate_adjoint \
    --video input.mp4 --output_image output.png --fov 180

# With restarts to escape local minima
python -m camera_calibration.solvers.calibrate_adjoint \
    --video input.mp4 --output_image output.png --fov 180 --restarts 5

# Fix lens 1 rotation (reduces ambiguity)
python -m camera_calibration.solvers.calibrate_adjoint \
    --video input.mp4 --output_image output.png --fov 180 --fix-lens1 --restarts 3
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--image` / `--video` | Input source | required |
| `--output` | Save calibration JSON | - |
| `--output_image` | Save stitched result | - |
| `--fov` | Base FOV in degrees | 195 |
| `--scale` | Downsample scale (0.25=4x faster) | 0.25 |
| `--frames` | Video frames to use | 5 |
| `--fix-lens1` | Fix lens 1 rotation at 0 | false |
| `--restarts` | Random restarts | 1 |

### Calibration (Stepwise Optimizer)

Grid search over FOV with stepwise parameter optimization:

```bash
python -m camera_calibration.solvers.calibrate_stepwise \
    --video input.mp4 --output_image output.png --fov 180 --restarts 5 -v
```

**Additional options:** `--verbose` / `-v` for detailed progress.

### Projection

Convert dual-fisheye to equirectangular:

```bash
# Using calibration file
python -m camera_calibration.projections.fisheye_to_equirect \
    --input dual_fisheye.png --output equirect.png --calibration calib.json

# Using default projection
python -m camera_calibration.projections.fisheye_to_equirect \
    --input dual_fisheye.png --output equirect.png --use-calibrated-path --fov 180
```

### Create Calibration File

```bash
python -m camera_calibration.calib.create_calibration --output my_calibration.json
```

## Calibration Parameters

Each lens has the following optimizable parameters:

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `center_x`, `center_y` | Optical center (normalized 0-1) | 0.45 - 0.55 |
| `fov` | FOV scale factor | 1.01 - 1.15 |
| `k1`, `k2`, `k3` | Radial distortion coefficients | -0.3 to 0.3 |
| `rotation_yaw` | Yaw rotation (radians) | ±0.15 |
| `rotation_pitch` | Pitch rotation (radians) | ±0.15 |
| `rotation_roll` | Roll rotation (radians) | ±0.15 |

## Python API

```python
from camera_calibration.calib import CameraCalibration, LensCalibration
from camera_calibration.projections import fisheye_to_equirect_calibrated, mask_fisheye_circle
from camera_calibration.solvers import FeatureTracker

# Load calibration
calib = CameraCalibration.load_json('calibration.json')

# Project fisheye to equirectangular
patch, mask = fisheye_to_equirect_calibrated(fisheye_img, width, height, calib.lens1, base_fov)
```

## Tips

- Use `--fix-lens1` to reduce optimization ambiguity (only relative rotation matters)
- Use `--restarts` to escape local minima
- Use `--scale 0.25` (default) for faster optimization, `--scale 1.0` for full resolution
- Start with `--fov 180` for Gear 360 cameras
