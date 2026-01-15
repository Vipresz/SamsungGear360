# Gear360 Viewer

View live video streams from Gear360 camera.

**Default Stream URL:** `http://192.168.43.1:7679/livestream_high.avi`

## Quick Start

### Python (Recommended - Easiest)

```bash
cd python
pip install opencv-python numpy
python gear360_viewer.py
```

### C++

```bash
cd cpp
mkdir build && cd build
cmake ..
cmake --build .
./bin/gear360_viewer
```

### Command Line (FFmpeg)

```bash
ffplay -hide_banner -fflags nobuffer -flags low_delay -framedrop \
  -i "http://192.168.43.1:7679/livestream_high.avi"
```

## Python Viewer

**Location:** `python/gear360_viewer.py`

**Install:**
```bash
pip install -r python/requirements.txt
```

**Usage:**
```bash
python python/gear360_viewer.py [url] [--fps 30] [--scale 0.5]
```

**Controls:** Press `q` or `ESC` to quit

## C++ Viewer

**Location:** `cpp/`

**Dependencies (macOS):**
```bash
brew install ffmpeg glfw glew
```

**Build:**
```bash
cd cpp
mkdir build && cd build
cmake ..
cmake --build .
```

**Run:**
```bash
./bin/gear360_viewer [url]
```

**Controls:** Press `ESC` to quit

## Troubleshooting

- **Stream won't connect:** Verify camera is streaming and URL is correct
- **Python errors:** Ensure OpenCV is installed: `pip install opencv-python`
- **C++ build fails:** Install dependencies via Homebrew (see above)
