#pragma once

#include <cstdint>
#include "PanTilt.h"
#include "app/DetectorSlot.h"
#include "app/PreviewStreamer.h"
#include "link/Link.h"
#include "models/ModelStore.h"
#include "vision/DetectorRegistry.h"

/**
 * The host's benchmark commands: list and load models, run inference on images streamed over
 * USB, capture camera frames for the dataset, and inspect the model store.
 * Runs on the vision task, so the detector is never used from two tasks at once.
 */
class BenchService {

private:
    Link &link_;
    DetectorSlot &slot_;
    const DetectorRegistry &registry_;
    const ModelStore &store_;
    PreviewStreamer &preview_;
    const PanTilt &mount_;
    uint32_t captureId_ = 0;

    void listModels();
    void loadModel(const RxPacket &packet);
    void setThreshold(const RxPacket &packet);
    void infer(const RxPacket &packet, proto::Mode mode);
    void capture(const RxPacket &packet);
    void listStore();
    void loadBlob(const RxPacket &packet);

public:
    BenchService(Link &link, DetectorSlot &slot, const DetectorRegistry &registry, const ModelStore &store,
                 PreviewStreamer &preview, const PanTilt &mount);
    BenchService(const BenchService &) = delete;
    BenchService &operator=(const BenchService &) = delete;
    BenchService(BenchService &&) = delete;
    BenchService &operator=(BenchService &&) = delete;

    // Handle one request; false if it isn't a benchmark command
    bool handle(const RxPacket &packet, proto::Mode mode);

    // Describe the loaded model and what loading it cost
    void sendModelInfo(const LoadReport &report);
};
