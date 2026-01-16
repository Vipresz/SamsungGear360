#ifndef STITCHING_H
#define STITCHING_H

#include "config.h"
#include <vector>

// Forward declarations
bool computeFrameAlignment(const std::vector<uint8_t>& data, int width, int height,
                          float& offset1X, float& offset1Y, float& offset2X, float& offset2Y);

// Analyze a single frame to extract stitching parameters (simplified, immediate analysis)
void analyzeFrameForStitchingImmediate(const std::vector<uint8_t>& data, int width, int height);

// Analyze all collected frames to extract optimal stitching parameters
void analyzeCollectedFramesForStitching();

// Refine alignment by computing photometric alignment in overlap region
void refineAlignmentFromCollectedFrames();

// Analyze single frame to detect lens parameters (legacy function, kept for initial detection)
void analyzeFrameForStitching(const std::vector<uint8_t>& data, int width, int height);

#endif // STITCHING_H
