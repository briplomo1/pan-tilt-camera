#pragma once

#include <utility>
#include "esp_camera.h"
#include "vision/ImageView.h"

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

    // The pixels as an image any detector can take; empty unless the frame is RGB565.
    // Only valid while this Frame is alive.
    [[nodiscard]] ImageView view() const {
        if (!fb_ || fb_->format != PIXFORMAT_RGB565) return {};
        return ImageView{fb_->buf, static_cast<uint16_t>(fb_->width), static_cast<uint16_t>(fb_->height),
                         PixelFormat::Rgb565Be};
    }
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
