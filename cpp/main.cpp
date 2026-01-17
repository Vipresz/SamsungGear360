#include "config.h"
#include "shaders/shaders.h"
#include "renderer.h"
#include "video_decoder.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <thread>
#include <csignal>
#include <cstring>
#include <string>
#include <mutex>
#include <atomic>

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options] [url]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --rectilinear       Enable rectilinear conversion (spherical to flat view)" << std::endl;
    std::cout << "  --equirectangular   Enable equirectangular projection (dual-lens to 360° view)" << std::endl;
    std::cout << "  --fov <degrees>     Field of view for conversions (default: 195 when --rectilinear or --equirectangular used)" << std::endl;
    std::cout << "  --stitch            Enable stitch mode for equirectangular projection (analyzes frames to configure dual-lens stitching)" << std::endl;
    std::cout << "  --calibration <file>  Load calibration parameters from TOML file" << std::endl;
    std::cout << "  --help              Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Default: Displays raw equirectangular video (no conversion)" << std::endl;
    std::cout << "Default URL: http://192.168.43.1:7679/livestream_high.avi" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << programName << "                              # Raw video" << std::endl;
    std::cout << "  " << programName << " --rectilinear                 # Rectilinear with default 195° FOV" << std::endl;
    std::cout << "  " << programName << " --rectilinear --fov 120        # Rectilinear with 120° FOV" << std::endl;
    std::cout << "  " << programName << " --equirectangular              # Equirectangular with default 195° FOV" << std::endl;
    std::cout << "  " << programName << " --equirectangular --fov 180    # Equirectangular with 180° FOV" << std::endl;
    std::cout << "  " << programName << " --equirectangular --stitch      # Equirectangular with stitching" << std::endl;
    std::cout << "  " << programName << " --equirectangular --calibration calibration.toml  # Use calibration file" << std::endl;
}

int main(int argc, char* argv[]) {
    // Set up signal handlers for graceful shutdown
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // Default URL (port 7679 based on ffplay working example)
    g_options.url = "http://192.168.43.1:7679/livestream_high.avi";
    bool fovSpecified = false;
    std::string calibrationFile = "";
    
    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "--rectilinear") == 0) {
            g_options.rectilinearMode = true;
        } else if (strcmp(argv[i], "--equirectangular") == 0) {
            g_options.equirectangularMode = true;
        } else if (strcmp(argv[i], "--fov") == 0) {
            if (i + 1 < argc) {
                g_options.fov = std::stof(argv[++i]);
                if (g_options.fov <= 0 || g_options.fov > 360) {
                    std::cerr << "Error: FOV must be between 0 and 360 degrees" << std::endl;
                    return 1;
                }
                fovSpecified = true;
            } else {
                std::cerr << "Error: --fov requires a value" << std::endl;
                return 1;
            }
        } else if (strcmp(argv[i], "--stitch") == 0) {
            g_options.stitchMode = true;
        } else if (strcmp(argv[i], "--calibration") == 0) {
            if (i + 1 < argc) {
                calibrationFile = argv[++i];
            } else {
                std::cerr << "Error: --calibration requires a file path" << std::endl;
                return 1;
            }
        } else if (argv[i][0] != '-') {
            g_options.url = argv[i];
        } else {
            std::cerr << "Unknown option: " << argv[i] << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Load calibration file if specified
    if (!calibrationFile.empty()) {
        setCalibrationFilePath(calibrationFile);
        if (loadCalibrationFromFile(calibrationFile)) {
            // If calibration is loaded, enable equirectangular mode automatically
            if (!g_options.equirectangularMode && !g_options.rectilinearMode) {
                g_options.equirectangularMode = true;
                std::cout << "Equirectangular mode auto-enabled due to calibration file" << std::endl;
            }
        } else {
            std::cerr << "Warning: Failed to load calibration file, using defaults" << std::endl;
        }
    } else {
        // Default calibration file path for reload functionality
        setCalibrationFilePath("calibration.toml");
    }
    
    // If rectilinear or equirectangular is enabled but FOV not specified, use default 195
    if ((g_options.rectilinearMode || g_options.equirectangularMode) && !fovSpecified) {
        g_options.fov = 195.0f;
    }
    
    std::cout << "Gear360 Viewer - Connecting to: " << g_options.url << std::endl;
    std::cout << "Press 'R' to reload calibration from: " << getCalibrationFilePath() << std::endl;
    if (g_options.rectilinearMode) {
        std::cout << "Rectilinear mode: ENABLED (FOV: " << g_options.fov << " degrees)" << std::endl;
    } else if (g_options.equirectangularMode) {
        std::cout << "Equirectangular mode: ENABLED (FOV: " << g_options.fov << " degrees)";
        if (g_options.stitchMode) {
            std::cout << " with stitching";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Display mode: Raw equirectangular video" << std::endl;
    }
    if (g_options.stitchMode && g_options.equirectangularMode) {
        std::cout << "Stitch mode: ENABLED (analyzing first " << g_options.stitchFrames << " frames)" << std::endl;
    }
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Gear360 Viewer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    
    while (glGetError() != GL_NO_ERROR) {}
    
    if (!createShaders()) {
        std::cerr << "Failed to create shaders" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    
    setupQuad();
    
    // Start video decode thread
    std::thread decodeThread(videoDecodeThread, g_options.url);
    
    // Main render loop
    while (!glfwWindowShouldClose(window) && g_running) {
        glfwPollEvents();
        
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        
        // Reload calibration on 'R' key press
        static bool rKeyWasPressed = false;
        if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS) {
            if (!rKeyWasPressed) {
                rKeyWasPressed = true;
                reloadCalibration();
            }
        } else {
            rKeyWasPressed = false;
        }
        
        // Update texture if new frame available
        {
            std::lock_guard<std::mutex> lock(g_frameData.mutex);
            if (g_frameData.updated && g_frameData.width > 0 && g_frameData.height > 0) {
                updateTexture(g_frameData.data, g_frameData.width, g_frameData.height);
                g_frameData.updated = false;
            }
        }
        
        // Render
        renderFrame();
        
        glfwSwapBuffers(window);
    }
    
    // Cleanup - stop running first to signal decode thread
    std::cout << "Shutting down..." << std::endl;
    g_running = false;
    
    // Wait for decode thread to finish and close stream
    if (decodeThread.joinable()) {
        decodeThread.join();
    }
    
    // Ensure format context is closed
    AVFormatContext* fmt = g_formatContext.load();
    if (fmt) {
        avformat_close_input(&fmt);
        g_formatContext.store(nullptr);
    }
    
    // Cleanup OpenGL resources
    if (g_texture != 0) {
        glDeleteTextures(1, &g_texture);
    }
    if (g_shaderProgram != 0) {
        glDeleteProgram(g_shaderProgram);
    }
    if (g_VAO != 0) {
        glDeleteVertexArrays(1, &g_VAO);
    }
    if (g_VBO != 0) {
        glDeleteBuffers(1, &g_VBO);
    }
    if (g_EBO != 0) {
        glDeleteBuffers(1, &g_EBO);
    }
    
    glfwDestroyWindow(window);
    glfwTerminate();
    
    // Network deinit is handled in video_decoder.cpp
    
    std::cout << "Application closed, stream disconnected" << std::endl;
    
    return 0;
}
