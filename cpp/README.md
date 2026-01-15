# Gear360 Viewer - C++ Build Guide

Build instructions and troubleshooting for Windows, Linux, and macOS.

## Prerequisites

### macOS
```bash
brew install ffmpeg glfw glew cmake
```

### Linux (Debian/Ubuntu)
```bash
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libavformat-dev libavcodec-dev libswscale-dev libavutil-dev
sudo apt-get install libglfw3-dev libglew-dev
sudo apt-get install libgl1-mesa-dev
```

### Linux (Fedora/RHEL)
```bash
sudo dnf install gcc-c++ cmake
sudo dnf install ffmpeg-devel glfw-devel glew-devel
sudo dnf install mesa-libGL-devel
```

### Windows

**Option 1: vcpkg (Recommended)**
```powershell
# Install vcpkg (if not already installed)
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install dependencies
.\vcpkg install ffmpeg:x64-windows glfw3:x64-windows glew:x64-windows

# Build with vcpkg toolchain
cmake -DCMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmake ..
```

**Option 2: Manual Installation**
- Download FFmpeg from https://ffmpeg.org/download.html
- Extract to `C:/ffmpeg/` (should have `include/` and `lib/` subdirectories)
- Download GLFW and GLEW pre-built binaries or build from source
- Set environment variables or use CMake GUI to specify paths

## Building

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

**Windows with vcpkg:**
```powershell
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=C:/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake ..
cmake --build .
```

## Verifying Dependencies

CMake will report found libraries during configuration. Look for:
```
Found FFmpeg (Homebrew): /opt/homebrew/opt/ffmpeg
Found FFmpeg via pkg-config
Found FFmpeg (manual search)
```

If you see `FATAL_ERROR`, dependencies are missing.

## Common Issues

### FFmpeg Not Found

**macOS:**
- Verify: `brew list ffmpeg`
- Check path: `ls /opt/homebrew/opt/ffmpeg/include/libavformat/avformat.h`
- Reinstall: `brew reinstall ffmpeg`

**Linux:**
- Verify packages: `dpkg -l | grep ffmpeg` (Debian/Ubuntu)
- Check headers: `ls /usr/include/libavformat/avformat.h`
- Install dev packages (not just runtime): `libavformat-dev` not `libavformat`

**Windows:**
- Verify vcpkg: `vcpkg list | findstr ffmpeg`
- Check manual install: `dir C:\ffmpeg\include\libavformat\avformat.h`
- Use vcpkg toolchain: `-DCMAKE_TOOLCHAIN_FILE=...`

### GLFW/GLEW Not Found

**macOS:**
- Install: `brew install glfw glew`
- Verify: `ls /opt/homebrew/opt/glfw/lib/libglfw.dylib`

**Linux:**
- Install dev packages: `libglfw3-dev libglew-dev`
- Verify: `pkg-config --libs glfw3`

**Windows:**
- Use vcpkg: `vcpkg install glfw3 glew`
- Or download pre-built binaries and set paths manually

### OpenGL Not Found

**Linux:**
- Install: `sudo apt-get install libgl1-mesa-dev` (Debian/Ubuntu)
- Or: `sudo dnf install mesa-libGL-devel` (Fedora)

**Windows:**
- Usually included with graphics drivers
- May need to install development headers separately

### Compilation Errors

**"Cannot find -lavformat" (Linux):**
- Install development packages: `-dev` suffix packages
- Verify: `pkg-config --libs libavformat`

**"Undefined reference" errors:**
- Libraries found but not linked correctly
- Check CMake output for library paths
- Verify library files exist at reported paths

**Windows DLL errors:**
- Copy DLLs to executable directory
- Or add library directories to PATH
- vcpkg handles this automatically

## Debugging CMake Configuration

Enable verbose output:
```bash
cmake .. --debug-output
```

Check what CMake found:
```bash
cmake .. -DCMAKE_FIND_DEBUG_MODE=ON
```

View cached variables:
```bash
cmake .. -L
# or
cat CMakeCache.txt | grep FFMPEG
```

## Platform-Specific Notes

### macOS
- Uses Homebrew paths: `/opt/homebrew/` (Apple Silicon) or `/usr/local/` (Intel)
- Frameworks automatically linked (Cocoa, IOKit, CoreVideo)
- May need Xcode Command Line Tools: `xcode-select --install`

### Linux
- Prefers pkg-config for dependency detection
- May need to set `PKG_CONFIG_PATH` if libraries in non-standard locations
- OpenGL via Mesa on most distributions

### Windows
- vcpkg is the easiest dependency management solution
- Visual Studio 2019+ recommended
- MinGW/MSYS2 also supported
- DLLs must be in PATH or executable directory

## Build Output

Executable location:
- **macOS/Linux:** `build/bin/gear360_viewer`
- **Windows:** `build/bin/gear360_viewer.exe` (Release) or `build/bin/Debug/gear360_viewer.exe` (Debug)

## Testing

Run the viewer:
```bash
./build/bin/gear360_viewer [stream_url]
```

Default URL: `http://192.168.43.1:7680/livestream_high.avi`

## Getting Help

If build fails:
1. Check CMake output for specific error messages
2. Verify all dependencies are installed (see Prerequisites)
3. Try cleaning build directory: `rm -rf build && mkdir build`
4. Check platform-specific notes above
5. Review error messages - they include platform-specific install instructions
