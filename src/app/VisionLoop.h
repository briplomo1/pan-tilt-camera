#pragma once

#include <string>
#include <string_view>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "PanTilt.h"
#include "Tracker.h"
#include "app/BenchService.h"
#include "app/DetectorSlot.h"
#include "app/PreviewStreamer.h"
#include "link/Link.h"

/**
 * The vision task (core 1). It owns the loaded detector, the tracker and command handling, so
 * the detector is only ever touched from this one task and needs no locking:
 *
 *   loop: handle any packets from the host, then (Track mode) run one tracker step.
 *
 * Track mode tracks faces and can stream telemetry and the live view. Bench mode pauses the
 * tracker, holds the mount still and lets the host drive inference on its own images.
 */
class VisionLoop final : private ITrackObserver {

private:
    Link &link_;
    DetectorSlot &slot_;
    BenchService &bench_;
    PreviewStreamer &preview_;
    Tracker tracker_;
    proto::Mode mode_ = proto::Mode::Track;
    bool telemetry_ = false;
    std::string initialModel_;
    TaskHandle_t task_ = nullptr;

    static void taskEntry(void *arg);
    [[noreturn]] void run();
    void dispatch(const RxPacket &packet);
    void hello();
    void setMode(const RxPacket &packet);
    void subscribe(const RxPacket &packet);
    void sync(const RxPacket &packet);
    void setPreview(const RxPacket &packet);
    void onTrackFrame(const TrackSample &sample) override;

public:
    VisionLoop(PanTilt &mount, Link &link, DetectorSlot &slot, BenchService &bench, PreviewStreamer &preview);
    ~VisionLoop() override;
    VisionLoop(const VisionLoop &) = delete;
    VisionLoop &operator=(const VisionLoop &) = delete;
    VisionLoop(VisionLoop &&) = delete;
    VisionLoop &operator=(VisionLoop &&) = delete;

    // Start the task on core 1; it loads `modelId` first
    void start(std::string_view modelId);
};
