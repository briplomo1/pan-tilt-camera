#pragma once

#include <utility>
#include "esp_camera.h"

// One captured frame. Hands the buffer back to the driver when it goes out of scope.
class Frame {

private:
    camera_fb_t *fb_;

public:
    explicit Frame(camera_fb_t *fb) : fb_(fb) {}
    ~Frame() { if (fb_) esp_camera_fb_return(fb_); }
    Frame(const Frame &) = delete;
    Frame &operator=(const Frame &) = delete;
    Frame(Frame &&other) noexcept : fb_(std::exchange(other.fb_, nullptr)) {}
    Frame &operator=(Frame &&) = delete;

    explicit operator bool() const { return fb_ != nullptr; }
    const camera_fb_t &operator*() const { return *fb_; }
    const camera_fb_t *operator->() const { return fb_; }
};

/**
 * OV2640/OV3660 on the XIAO ESP32S3 Sense board, set up for detection:
 * 320x240 RGB565 in PSRAM, always returning the newest frame.
 */
class Camera {
public:
    static void init();
    [[nodiscard]] static Frame capture();
};
