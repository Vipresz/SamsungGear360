# Gear360 C++ Viewer

High-performance OpenGL-based viewer for streaming and viewing 360° video from Samsung Gear360 cameras with real-time projection and stitching capabilities.

## Features

- **Live Video Streaming**: Low-latency streaming from Gear360 camera via WiFi
- **Multiple Projection Modes**:
  - Raw equirectangular (default)
  - Rectilinear conversion (spherical to flat view)
  - Equirectangular projection (dual-fisheye to 360° view)
- **Lens Calibration**: Load calibration parameters from TOML files
- **Real-time Stitching**: Automatic dual-lens alignment and blending
- **Hot Reload**: Press `R` to reload calibration without restarting
- **Customizable FOV**: Adjustable field of view for projections

## Building

### Prerequisites

- **CMake** 3.22 or higher
- **C++17** compatible compiler
- **Visual Studio 2019+** (Windows) or **GCC/Clang** (Linux/macOS)

### Windows

**Using vcpkg (Recommended):**
```powershell
# Install vcpkg (if not already installed)
cd C:\
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install dependencies
.\vcpkg install ffmpeg:x64-windows glfw3:x64-windows glew:x64-windows

# Build
cd cpp
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake ..
cmake --build .
```

### macOS

```bash
# Install dependencies
brew install ffmpeg glfw glew cmake

# Build
cd cpp
mkdir build
cd build
cmake ..
cmake --build .
```

### Linux (Debian/Ubuntu)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libavformat-dev libavcodec-dev libswscale-dev libavutil-dev
sudo apt-get install libglfw3-dev libglew-dev libgl1-mesa-dev

# Build
cd cpp
mkdir build
cd build
cmake ..
cmake --build .
```

### Linux (Fedora/RHEL)

```bash
# Install dependencies
sudo dnf install gcc-c++ cmake
sudo dnf install ffmpeg-devel glfw-devel glew-devel mesa-libGL-devel

# Build
cd cpp
mkdir build
cd build
cmake ..
cmake --build .
```

## Running

### Basic Usage

```bash
# Raw video (default)
./build/bin/gear360_viewer

# With custom URL
./build/bin/gear360_viewer http://192.168.43.1:7679/livestream_high.avi
```

### Projection Modes

```bash
# Rectilinear conversion
./build/bin/gear360_viewer --rectilinear

# Rectilinear with custom FOV
./build/bin/gear360_viewer --rectilinear --fov 120

# Equirectangular projection
./build/bin/gear360_viewer --equirectangular

# Equirectangular with custom FOV
./build/bin/gear360_viewer --equirectangular --fov 180

# Equirectangular with automatic stitching
./build/bin/gear360_viewer --equirectangular --stitch
```

### Using Calibration Files

```bash
# Load calibration from TOML file
./build/bin/gear360_viewer --equirectangular --calibration calibration.toml
```

### Controls

- **ESC**: Quit application
- **R**: Reload calibration file (hot reload)

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--rectilinear` | Enable rectilinear conversion (spherical to flat view) |
| `--equirectangular` | Enable equirectangular projection (dual-lens to 360° view) |
| `--fov <degrees>` | Field of view for conversions (default: 195°) |
| `--stitch` | Enable stitch mode for automatic dual-lens alignment |
| `--calibration <file>` | Load calibration parameters from TOML file |
| `--help` | Show help message |

## Calibration Files

Calibration files are TOML format and contain lens parameters:
- Lens center positions
- FOV scaling factors
- Distortion coefficients
- Rotation corrections
- Alignment offsets

Example calibration file structure:
```toml
[lens1]
center_x = 0.5
center_y = 0.5
fov = 1.08
# ... additional parameters

[lens2]
# ... lens2 parameters
```

## Troubleshooting

### Dependencies Not Found

**Windows:**
- Ensure vcpkg toolchain is specified: `-DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake`
- Verify packages: `C:\vcpkg\vcpkg list`

**macOS:**
- Verify Homebrew installation: `brew list ffmpeg glfw glew`
- Check paths: `ls /opt/homebrew/opt/ffmpeg/include/libavformat/avformat.h`

**Linux:**
- Install development packages (with `-dev` suffix)
- Verify: `pkg-config --libs glfw3`

### Stream Connection Issues

- Verify camera is streaming and URL is correct
- Check network connectivity
- Default URL: `http://192.168.43.1:7679/livestream_high.avi`

### Build Errors

- Clean build directory: `rm -rf build && mkdir build`
- Check CMake output for specific error messages
- Verify all dependencies are installed (see Prerequisites)

## Output Location

- **macOS/Linux**: `build/bin/gear360_viewer`
- **Windows**: `build/bin/gear360_viewer.exe` (Release) or `build/bin/Debug/gear360_viewer.exe` (Debug)
