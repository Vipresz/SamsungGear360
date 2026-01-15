cmake_minimum_required(VERSION 3.22)

project(Gear360Viewer VERSION 1.0.0 LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ============================================================================
# Platform Detection and Dependency Setup
# ============================================================================

if(APPLE)
    # macOS: Use Homebrew paths
    if(DEFINED HOMEBREW_PREFIX)
        set(BREW_PREFIX "${HOMEBREW_PREFIX}")
    elseif(EXISTS "/opt/homebrew")
        set(BREW_PREFIX "/opt/homebrew")  # Apple Silicon
    else()
        set(BREW_PREFIX "/usr/local")  # Intel Mac
    endif()
    set(USE_HOMEBREW TRUE)
    message(STATUS "Using Homebrew prefix: ${BREW_PREFIX}")
else()
    set(USE_HOMEBREW FALSE)
endif()

# ============================================================================
# Find FFmpeg libraries
# ============================================================================

set(FFMPEG_FOUND FALSE)

if(USE_HOMEBREW)
    # macOS: Use Homebrew FFmpeg
    set(FFMPEG_PREFIX "${BREW_PREFIX}/opt/ffmpeg")
    
    # Check if FFmpeg include directory exists
    if(EXISTS "${FFMPEG_PREFIX}/include" AND (EXISTS "${FFMPEG_PREFIX}/include/libavformat/avformat.h" OR EXISTS "${FFMPEG_PREFIX}/include/libavformat"))
        set(FFMPEG_INCLUDE_DIR "${FFMPEG_PREFIX}/include")
        
        find_library(AVFORMAT_LIB
            NAMES avformat
            PATHS "${FFMPEG_PREFIX}/lib"
            NO_DEFAULT_PATH
        )
        
        find_library(AVCODEC_LIB
            NAMES avcodec
            PATHS "${FFMPEG_PREFIX}/lib"
            NO_DEFAULT_PATH
        )
        
        find_library(SWSCALE_LIB
            NAMES swscale
            PATHS "${FFMPEG_PREFIX}/lib"
            NO_DEFAULT_PATH
        )
        
        find_library(AVUTIL_LIB
            NAMES avutil
            PATHS "${FFMPEG_PREFIX}/lib"
            NO_DEFAULT_PATH
        )
        
        if(AVFORMAT_LIB AND AVCODEC_LIB AND SWSCALE_LIB AND AVUTIL_LIB)
            set(FFMPEG_FOUND TRUE)
            message(STATUS "Found FFmpeg (Homebrew): ${FFMPEG_PREFIX}")
        endif()
    endif()
endif()

# Try pkg-config for FFmpeg (Linux/Unix)
if(NOT FFMPEG_FOUND)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(FFMPEG libavformat libavcodec libswscale libavutil)
        if(FFMPEG_FOUND)
            set(FFMPEG_INCLUDE_DIR ${FFMPEG_INCLUDE_DIRS})
            set(AVFORMAT_LIB ${FFMPEG_LIBRARIES})
            set(AVCODEC_LIB ${FFMPEG_LIBRARIES})
            set(SWSCALE_LIB ${FFMPEG_LIBRARIES})
            set(AVUTIL_LIB ${FFMPEG_LIBRARIES})
            set(FFMPEG_FOUND TRUE)
            message(STATUS "Found FFmpeg via pkg-config")
        endif()
    endif()
endif()

# Try manual finding for Windows or if pkg-config fails
if(NOT FFMPEG_FOUND)
    find_path(FFMPEG_INCLUDE_DIR libavformat/avformat.h
        PATHS
        "C:/ffmpeg/include"
        "C:/vcpkg/installed/x64-windows/include"
        "C:/vcpkg/installed/x86-windows/include"
        "/usr/include"
        "/usr/local/include"
    )
    
    if(FFMPEG_INCLUDE_DIR)
        find_library(AVFORMAT_LIB
            NAMES avformat
            PATHS
            "C:/ffmpeg/lib"
            "C:/vcpkg/installed/x64-windows/lib"
            "C:/vcpkg/installed/x86-windows/lib"
            "/usr/lib"
            "/usr/local/lib"
        )
        
        find_library(AVCODEC_LIB
            NAMES avcodec
            PATHS
            "C:/ffmpeg/lib"
            "C:/vcpkg/installed/x64-windows/lib"
            "C:/vcpkg/installed/x86-windows/lib"
            "/usr/lib"
            "/usr/local/lib"
        )
        
        find_library(SWSCALE_LIB
            NAMES swscale
            PATHS
            "C:/ffmpeg/lib"
            "C:/vcpkg/installed/x64-windows/lib"
            "C:/vcpkg/installed/x86-windows/lib"
            "/usr/lib"
            "/usr/local/lib"
        )
        
        find_library(AVUTIL_LIB
            NAMES avutil
            PATHS
            "C:/ffmpeg/lib"
            "C:/vcpkg/installed/x64-windows/lib"
            "C:/vcpkg/installed/x86-windows/lib"
            "/usr/lib"
            "/usr/local/lib"
        )
        
        if(AVFORMAT_LIB AND AVCODEC_LIB AND SWSCALE_LIB AND AVUTIL_LIB)
            set(FFMPEG_FOUND TRUE)
            message(STATUS "Found FFmpeg (manual search)")
        endif()
    endif()
endif()

if(NOT FFMPEG_FOUND)
    if(APPLE)
        message(FATAL_ERROR "FFmpeg is required but not found. Install via: brew install ffmpeg")
    elseif(UNIX)
        message(FATAL_ERROR "FFmpeg is required but not found. Install via: sudo apt-get install libavformat-dev libavcodec-dev libswscale-dev libavutil-dev (Debian/Ubuntu) or equivalent for your distribution")
    else()
        message(FATAL_ERROR "FFmpeg is required but not found. Install FFmpeg and set FFMPEG_DIR, or use vcpkg: vcpkg install ffmpeg")
    endif()
endif()

# ============================================================================
# Find GLFW and GLEW
# ============================================================================

if(USE_HOMEBREW)
    find_library(GLFW_LIB
        NAMES glfw
        PATHS "${BREW_PREFIX}/opt/glfw/lib"
        NO_DEFAULT_PATH
    )
    
    find_library(GLEW_LIB
        NAMES GLEW glew
        PATHS "${BREW_PREFIX}/opt/glew/lib"
        NO_DEFAULT_PATH
    )
    
    if(GLFW_LIB)
        set(GLFW_INCLUDE_DIR "${BREW_PREFIX}/opt/glfw/include")
    endif()
    
    if(GLEW_LIB)
        set(GLEW_INCLUDE_DIR "${BREW_PREFIX}/opt/glew/include")
    endif()
else()
    # Try pkg-config first (Linux)
    find_package(PkgConfig QUIET)
    if(PKG_CONFIG_FOUND)
        pkg_check_modules(GLFW_PKG glfw3)
        if(GLFW_PKG_FOUND)
            set(GLFW_INCLUDE_DIR ${GLFW_PKG_INCLUDE_DIRS})
            set(GLFW_LIB ${GLFW_PKG_LIBRARIES})
        endif()
    endif()
    
    # Manual search if pkg-config didn't find it
    if(NOT GLFW_LIB)
        find_library(GLFW_LIB 
            NAMES glfw3 glfw
            PATHS
            "C:/vcpkg/installed/x64-windows/lib"
            "C:/vcpkg/installed/x86-windows/lib"
            "/usr/lib"
            "/usr/local/lib"
        )
        find_path(GLFW_INCLUDE_DIR GLFW/glfw3.h
            PATHS
            "C:/vcpkg/installed/x64-windows/include"
            "C:/vcpkg/installed/x86-windows/include"
            "/usr/include"
            "/usr/local/include"
        )
    endif()
    
    if(NOT GLEW_LIB)
        find_library(GLEW_LIB 
            NAMES GLEW glew
            PATHS
            "C:/vcpkg/installed/x64-windows/lib"
            "C:/vcpkg/installed/x86-windows/lib"
            "/usr/lib"
            "/usr/local/lib"
        )
        find_path(GLEW_INCLUDE_DIR GL/glew.h
            PATHS
            "C:/vcpkg/installed/x64-windows/include"
            "C:/vcpkg/installed/x86-windows/include"
            "/usr/include"
            "/usr/local/include"
        )
    endif()
endif()

if(NOT GLFW_LIB)
    if(APPLE)
        message(FATAL_ERROR "GLFW not found. Install via: brew install glfw")
    elseif(UNIX)
        message(FATAL_ERROR "GLFW not found. Install via: sudo apt-get install libglfw3-dev (Debian/Ubuntu) or equivalent")
    else()
        message(FATAL_ERROR "GLFW not found. Install GLFW or use vcpkg: vcpkg install glfw3")
    endif()
endif()

if(NOT GLEW_LIB)
    if(APPLE)
        message(FATAL_ERROR "GLEW not found. Install via: brew install glew")
    elseif(UNIX)
        message(FATAL_ERROR "GLEW not found. Install via: sudo apt-get install libglew-dev (Debian/Ubuntu) or equivalent")
    else()
        message(FATAL_ERROR "GLEW not found. Install GLEW or use vcpkg: vcpkg install glew")
    endif()
endif()

# ============================================================================
# OpenGL
# ============================================================================

find_package(OpenGL REQUIRED)

# ============================================================================
# macOS-specific frameworks
# ============================================================================

if(APPLE)
    find_library(COCOA_FRAMEWORK Cocoa REQUIRED)
    find_library(IOKIT_FRAMEWORK IOKit REQUIRED)
    find_library(COREVIDEO_FRAMEWORK CoreVideo REQUIRED)
endif()

# ============================================================================
# Gear360 Viewer Executable
# ============================================================================

add_executable(gear360_viewer
    Gear360Viewer.cpp
)

# Include directories
target_include_directories(gear360_viewer PRIVATE
    ${FFMPEG_INCLUDE_DIR}
    ${GLFW_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIR}
)

# Link libraries
target_link_libraries(gear360_viewer
    ${AVFORMAT_LIB}
    ${AVCODEC_LIB}
    ${SWSCALE_LIB}
    ${AVUTIL_LIB}
    ${GLFW_LIB}
    ${GLEW_LIB}
    OpenGL::GL
)

# macOS-specific frameworks
if(APPLE)
    target_link_libraries(gear360_viewer
        ${COCOA_FRAMEWORK}
        ${IOKIT_FRAMEWORK}
        ${COREVIDEO_FRAMEWORK}
    )
endif()

# Compiler options
if(MSVC)
    target_compile_options(gear360_viewer PRIVATE /W4 /permissive- /Zc:__cplusplus /EHsc)
    target_compile_definitions(gear360_viewer PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(gear360_viewer PRIVATE -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# C++ standard
set_target_properties(gear360_viewer PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

message(STATUS "")
message(STATUS "Gear360 Viewer Build Configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Platform: ${CMAKE_SYSTEM_NAME}")
if(FFMPEG_PREFIX)
    message(STATUS "  FFmpeg: ${FFMPEG_PREFIX}")
else()
    message(STATUS "  FFmpeg: ${FFMPEG_INCLUDE_DIR}")
endif()
message(STATUS "  GLFW: ${GLFW_LIB}")
message(STATUS "  GLEW: ${GLEW_LIB}")
message(STATUS "")
message(STATUS "To build:")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  cmake --build .")
message(STATUS "")
