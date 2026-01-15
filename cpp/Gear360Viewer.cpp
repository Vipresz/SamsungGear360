// Gear360Viewer.cpp - Simple viewer application for Gear360 video stream
// Displays decoded video frames in a GLFW/OpenGL window
//
// Usage:
//   ./gear360_viewer [stream_url]
//
//   Default URL: http://192.168.43.1:7679/livestream_high.avi
//
// Controls:
//   ESC - Exit application
//
// Requirements:
//   - FFmpeg libraries (libavformat, libavcodec, libswscale, libavutil)
//   - GLFW and GLEW for windowing and OpenGL
//   - OpenGL 3.3+
//
// Build:
//   The viewer is configured in camera_system/CMakeLists.txt
//   Build with: cmake --build <build_dir> --target gear360_viewer

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libswscale/swscale.h>
    #include <libavutil/imgutils.h>
}

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>

// Global state
struct FrameData {
    std::vector<uint8_t> data;
    int width;
    int height;
    std::mutex mutex;
    bool updated = false;
};

FrameData g_frameData;
std::atomic<bool> g_running(true);

// OpenGL resources
GLuint g_texture = 0;
GLuint g_shaderProgram = 0;
GLuint g_VAO = 0;
GLuint g_VBO = 0;
GLuint g_EBO = 0;

// Simple vertex shader for displaying a texture
const char* vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = aTexCoord;
}
)";

// Simple fragment shader for displaying a texture
const char* fragmentShaderSource = R"(
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D ourTexture;

void main() {
    FragColor = texture(ourTexture, TexCoord);
}
)";

GLuint compileShader(GLenum type, const char* source) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);
    
    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(shader, 512, nullptr, infoLog);
        std::cerr << "Shader compilation error: " << infoLog << std::endl;
        return 0;
    }
    return shader;
}

bool createShaders() {
    GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    
    if (vertexShader == 0 || fragmentShader == 0) {
        return false;
    }
    
    g_shaderProgram = glCreateProgram();
    glAttachShader(g_shaderProgram, vertexShader);
    glAttachShader(g_shaderProgram, fragmentShader);
    glLinkProgram(g_shaderProgram);
    
    GLint success;
    glGetProgramiv(g_shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetProgramInfoLog(g_shaderProgram, 512, nullptr, infoLog);
        std::cerr << "Shader linking error: " << infoLog << std::endl;
        return false;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    return true;
}

void setupQuad() {
    // Fullscreen quad vertices (position + texture coordinates)
    float vertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 0.0f,  // top-left
         1.0f,  1.0f,  1.0f, 0.0f,  // top-right
         1.0f, -1.0f,  1.0f, 1.0f,  // bottom-right
        -1.0f, -1.0f,  0.0f, 1.0f   // bottom-left
    };
    
    unsigned int indices[] = {
        0, 1, 2,
        2, 3, 0
    };
    
    glGenVertexArrays(1, &g_VAO);
    glGenBuffers(1, &g_VBO);
    glGenBuffers(1, &g_EBO);
    
    glBindVertexArray(g_VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, g_VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
}

void updateTexture(const std::vector<uint8_t>& data, int width, int height) {
    if (g_texture == 0) {
        glGenTextures(1, &g_texture);
        glBindTexture(GL_TEXTURE_2D, g_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }
    
    glBindTexture(GL_TEXTURE_2D, g_texture);
    // Note: BGR24 format, but OpenGL expects RGB, so we'll use GL_BGR
    // However, GL_BGR might not be available, so we'll convert to RGB
    // For simplicity, we'll use GL_RGB and swap B and R channels in shader or convert
    // Actually, let's just use GL_BGR_EXT if available, or convert to RGB
    
    // Convert BGR to RGB
    std::vector<uint8_t> rgbData(data.size());
    for (size_t i = 0; i < data.size(); i += 3) {
        rgbData[i] = data[i + 2];     // R
        rgbData[i + 1] = data[i + 1]; // G
        rgbData[i + 2] = data[i];     // B
    }
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, rgbData.data());
}

void videoDecodeThread(const char* url) {
    avformat_network_init();
    
    AVFormatContext* fmt = nullptr;
    AVDictionary* opts = nullptr;
    
    // Lower latency options
    av_dict_set(&opts, "fflags", "nobuffer", 0);
    av_dict_set(&opts, "flags", "low_delay", 0);
    av_dict_set(&opts, "analyzeduration", "0", 0);
    av_dict_set(&opts, "probesize", "32", 0);
    
    if (avformat_open_input(&fmt, url, nullptr, &opts) < 0) {
        std::cerr << "Failed to open input: " << url << std::endl;
        g_running = false;
        return;
    }
    av_dict_free(&opts);
    
    if (avformat_find_stream_info(fmt, nullptr) < 0) {
        std::cerr << "Failed to find stream info" << std::endl;
        g_running = false;
        avformat_close_input(&fmt);
        return;
    }
    
    int videoStream = -1;
    for (unsigned i = 0; i < fmt->nb_streams; i++) {
        if (fmt->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            videoStream = (int)i;
            break;
        }
    }
    
    if (videoStream < 0) {
        std::cerr << "No video stream found" << std::endl;
        g_running = false;
        avformat_close_input(&fmt);
        return;
    }
    
    const AVCodecParameters* par = fmt->streams[videoStream]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(par->codec_id);
    if (!codec) {
        std::cerr << "Decoder not found" << std::endl;
        g_running = false;
        avformat_close_input(&fmt);
        return;
    }
    
    AVCodecContext* dec = avcodec_alloc_context3(codec);
    avcodec_parameters_to_context(dec, par);
    
    if (avcodec_open2(dec, codec, nullptr) < 0) {
        std::cerr << "Failed to open decoder" << std::endl;
        g_running = false;
        avcodec_free_context(&dec);
        avformat_close_input(&fmt);
        return;
    }
    
    AVPacket* pkt = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    
    // Convert to BGR24
    SwsContext* sws = sws_getContext(
        dec->width, dec->height, dec->pix_fmt,
        dec->width, dec->height, AV_PIX_FMT_BGR24,
        SWS_BILINEAR, nullptr, nullptr, nullptr
    );
    
    int bgrStride[4] = { dec->width * 3, 0, 0, 0 };
    std::vector<uint8_t> bgr(dec->width * dec->height * 3);
    uint8_t* bgrData[4] = { bgr.data(), nullptr, nullptr, nullptr };
    
    std::cout << "Video stream opened: " << dec->width << "x" << dec->height << std::endl;
    
    long long frameCount = 0;
    
    while (g_running && av_read_frame(fmt, pkt) >= 0) {
        if (pkt->stream_index != videoStream) {
            av_packet_unref(pkt);
            continue;
        }
        
        if (avcodec_send_packet(dec, pkt) == 0) {
            while (avcodec_receive_frame(dec, frame) == 0) {
                sws_scale(sws, frame->data, frame->linesize, 0, dec->height, bgrData, bgrStride);
                
                // Update frame data
                {
                    std::lock_guard<std::mutex> lock(g_frameData.mutex);
                    g_frameData.data = bgr;
                    g_frameData.width = dec->width;
                    g_frameData.height = dec->height;
                    g_frameData.updated = true;
                }
                
                frameCount++;
                if (frameCount % 30 == 0) {
                    std::cout << "Decoded frames: " << frameCount << std::endl;
                }
            }
        }
        
        av_packet_unref(pkt);
    }
    
    sws_freeContext(sws);
    av_frame_free(&frame);
    av_packet_free(&pkt);
    avcodec_free_context(&dec);
    avformat_close_input(&fmt);
    
    std::cout << "Video decode thread finished" << std::endl;
}

int main(int argc, char* argv[]) {
    const char* url = "http://192.168.43.1:7679/livestream_high.avi";
    
    if (argc > 1) {
        url = argv[1];
    }
    
    std::cout << "Gear360 Viewer - Connecting to: " << url << std::endl;
    
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return 1;
    }
    
    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    
    // Create window
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Gear360 Viewer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return 1;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    
    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    
    // Clear any GL errors from GLEW
    while (glGetError() != GL_NO_ERROR) {}
    
    // Setup OpenGL
    if (!createShaders()) {
        std::cerr << "Failed to create shaders" << std::endl;
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }
    
    setupQuad();
    
    // Start video decode thread
    std::thread decodeThread(videoDecodeThread, url);
    
    // Main render loop
    while (!glfwWindowShouldClose(window) && g_running) {
        glfwPollEvents();
        
        // Check for ESC key
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
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
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        
        if (g_texture != 0) {
            glUseProgram(g_shaderProgram);
            glBindTexture(GL_TEXTURE_2D, g_texture);
            glBindVertexArray(g_VAO);
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        }
        
        glfwSwapBuffers(window);
    }
    
    // Cleanup
    g_running = false;
    decodeThread.join();
    
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
    
    return 0;
}
