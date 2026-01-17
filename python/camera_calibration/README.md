# Samsung Gear 360 Camera Calibration

Tools for calibrating dual-fisheye 360° cameras and converting to equirectangular projection.

## Files

### Core Calibration

**`calibrate_adjoint.py`** - Joint optimization of all lens parameters simultaneously.
```bash
# Basic usage
python3 calibrate_adjoint.py --video input.mp4 --output_image output.png --fov 180

# With restarts to escape local minima
python3 calibrate_adjoint.py --video input.mp4 --output_image output.png --fov 180 --restarts 5

# Fix lens 1 rotation (reduces parameter ambiguity)
python3 calibrate_adjoint.py --video input.mp4 --output_image output.png --fov 180 --fix-lens1 --restarts 3

# Options:
#   --image / --video    Input source
#   --output             Save calibration JSON
#   --output_image       Save stitched result
#   --fov                Base FOV in degrees (default: 195)
#   --scale              Downsample for faster optimization (default: 0.25)
#   --frames             Number of video frames to use (default: 5)
#   --fix-lens1          Fix lens 1 rotation at 0
#   --restarts           Number of random restarts (default: 1)
```

**`calibrate_stepwise.py`** - Stepwise calibration with FOV grid search.
```bash
# Basic usage
python3 calibrate_stepwise.py --video input.mp4 --output_image output.png --fov 180

# With more restarts and verbose output
python3 calibrate_stepwise.py --video input.mp4 --output_image output.png --fov 180 --restarts 5 -v

# Options:
#   --image / --video    Input source
#   --output             Save calibration JSON
#   --output_image       Save stitched result
#   --fov                Base FOV in degrees (default: 195)
#   --scale              Downsample for faster optimization (default: 0.25)
#   --restarts           Number of random restarts per FOV (default: 3)
#   --fix-lens1          Fix lens 1 rotation at 0
#   -v, --verbose        Show detailed optimization progress
```

### Projection

**`fisheye_to_equirect.py`** - Convert dual-fisheye images to equirectangular.
```bash
# Using calibration file
python3 fisheye_to_equirect.py --input dual_fisheye.png --output equirect.png --calibration calib.json

# Using calibrated projection path
python3 fisheye_to_equirect.py --input dual_fisheye.png --output equirect.png --use-calibrated-path --fov 180
```

### Configuration

**`calibration_config.py`** - Data classes for calibration parameters.
- `LensCalibration`: Per-lens parameters (center, FOV, distortion k1/k2/k3, rotation)
- `CameraCalibration`: Combined calibration for both lenses

**`create_calibration.py`** - Create calibration JSON files manually.
```bash
python3 create_calibration.py --output my_calibration.json
```

### Utilities

**`masks.py`** - Fisheye circle masking utilities.

**`tracking.py`** - Feature tracking between frames.

**`__init__.py`** - Package exports.

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

## Tips

- Use `--fix-lens1` to reduce optimization ambiguity (only relative rotation matters)
- Use `--restarts` to escape local minima
- Use `--scale 0.25` (default) for faster optimization, `--scale 1.0` for full resolution
- Start with `--fov 180` for Gear 360 cameras
