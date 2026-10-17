#pragma once

#include <memory>
#include <vector>
#include "esp_camera.h"

class HumanFaceDetect;

// A detected face in image pixel coordinates
struct Face {
    float cx;      // box center
    float cy;
    int width;
    int height;
    float score;
};

/**
 * Espressif's human_face_detect model (ESP-DL). The default MSR+MNP model runs
 * in ~40 ms per frame on the ESP32-S3.
 */
class FaceDetector {

private:
    std::unique_ptr<HumanFaceDetect> model_;

public:
    FaceDetector();
    ~FaceDetector();
    FaceDetector(const FaceDetector &) = delete;
    FaceDetector &operator=(const FaceDetector &) = delete;

    // Run the model on an RGB565 frame and return every face found
    [[nodiscard]] std::vector<Face> detect(const camera_fb_t &frame);
};
