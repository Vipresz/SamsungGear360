# Samsung Gear360 Viewer & Calibration

A comprehensive toolset for streaming and viewing 360° video from Samsung Gear360 cameras, with advanced lens calibration capabilities for high-quality dual-fisheye stitching.

## Overview

This project provides tools to:
- **Stream live 360° video** from Samsung Gear360 cameras over WiFi
- **Calibrate dual-fisheye lenses** to optimize stitching quality
- **Project and view** dual-fisheye footage as equirectangular 360° video
- **Real-time rendering** with OpenGL for smooth playback

## Components

### Python Tools (`python/`)
- **Simple viewer**: Quick video stream viewer using OpenCV
- **Calibration tools**: Advanced lens calibration with joint and stepwise optimizers
- **Projection utilities**: Convert dual-fisheye images to equirectangular format

### C++ Viewer (`cpp/`)
- **High-performance viewer**: OpenGL-based real-time video rendering
- **Multiple projection modes**: Raw, rectilinear, and equirectangular views
- **Calibration support**: Load and hot-reload lens calibration parameters
- **Real-time stitching**: Automatic dual-lens alignment and blending

## Quick Start

### Python Viewer (Easiest)
```bash
cd python
pip install opencv-python numpy
python gear360_viewer.py
```

### C++ Viewer (Best Performance)
```bash
cd cpp
# See cpp/README.md for platform-specific build instructions
```

### Calibration
```bash
cd python
python -m camera_calibration.solvers.calibrate_adjoint \
    --video input.mp4 --output_image output.png --fov 180
```

## Default Stream URL

`http://192.168.43.1:7679/livestream_high.avi`

## Documentation

- **C++ Viewer**: See [`cpp/README.md`](cpp/README.md) for build instructions, features, and usage
- **Python Tools**: See [`python/README.md`](python/README.md) for calibration and projection tools
