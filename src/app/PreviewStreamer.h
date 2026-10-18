#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <span>
#include <string_view>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "link/Link.h"
#include "link/Protocol.h"
#include "memory/HeapBuffer.h"
#include "vision/IDetector.h"

struct PreviewSettings {
    bool enabled = false;
    proto::ImageFormat format = proto::ImageFormat::Jpeg;   // Jpeg, Rgb565Be or Gray8
    uint8_t scaleShift = 1;                                 // 0 full, 1 half, 2 quarter
    uint8_t everyN = 1;
    uint8_t jpegQuality = 60;
};

// One frame offered to the live view, with what was detected in it
struct PreviewFrame {
    ImageView image;
    uint32_t frameId = 0;
    int64_t timeUs = 0;
    uint8_t trackerState = 0;
    proto::Mode mode = proto::Mode::Track;
    float pan = 0.0f;
    float tilt = 0.0f;
    std::span<const Face> faces;
    int targetIndex = -1;
    DetectTiming timing;
    std::string_view modelId;
};

/**
 * Live view: streams camera frames plus detections to the host (tools/view/live_view.py),
 * which draws the boxes.
 *
 * offer() never blocks the vision task. If the previous frame is still being sent the new one
 * is dropped; otherwise it is copied (optionally downscaled) and a low-priority task on core 0
 * JPEG-encodes and sends it. Buffers exist only while the preview is enabled. Measured runs
 * should keep it off (see docs/EXPERIMENT_PLAN.md, observer effect).
 */
class PreviewStreamer {

private:
    static constexpr size_t kMaxFaces = 16;

    Link &link_;
    PreviewSettings settings_;
    HeapBuffer pixels_;            // copy of the frame, possibly downscaled
    HeapBuffer encoded_;           // JPEG output
    size_t pixelLength_ = 0;
    proto::PreviewMsg meta_{};
    std::array<proto::WireFace, kMaxFaces> faces_{};
    std::atomic<bool> busy_{false};   // set by offer(), cleared by the task after sending
    uint32_t offered_ = 0;
    uint32_t dropped_ = 0;
    TaskHandle_t task_ = nullptr;

    static void taskEntry(void *arg);
    [[noreturn]] void run();
    void sendPending();
    void waitIdle() const;

public:
    explicit PreviewStreamer(Link &link);
    ~PreviewStreamer();
    PreviewStreamer(const PreviewStreamer &) = delete;
    PreviewStreamer &operator=(const PreviewStreamer &) = delete;
    PreviewStreamer(PreviewStreamer &&) = delete;
    PreviewStreamer &operator=(PreviewStreamer &&) = delete;

    // Apply new settings (vision task only). False if they're invalid or memory ran out.
    bool configure(const PreviewSettings &settings);

    void offer(const PreviewFrame &frame);

    [[nodiscard]] bool enabled() const { return settings_.enabled; }
};
