#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include "vision/Face.h"
#include "vision/ImageView.h"

// Where the time in one detect() call went. Backends that can't split the stages only fill totalUs.
struct DetectTiming {
    uint32_t preprocessUs = 0;
    uint32_t inferenceUs = 0;
    uint32_t postprocessUs = 0;
    uint32_t totalUs = 0;
};

// Description of a deployed model, reported to the host with benchmark results
struct ModelInfo {
    std::string id;             // registry id, e.g. "espdl-msrmnp-s8"
    std::string backend;        // "esp-dl", "tflite-micro", ...
    std::string quantization;   // e.g. "w8a8 pot per-tensor ptq"
    uint16_t inputWidth = 0;    // model input size; the backend resizes frames to it
    uint16_t inputHeight = 0;
    size_t modelBytes = 0;      // model file size in flash, 0 if unknown
};

/**
 * A face detection model. Each backend (ESP-DL, later TFLite Micro, ...) implements this,
 * so the tracker and the benchmark never depend on which one is loaded.
 */
class IDetector {

public:
    virtual ~IDetector() = default;
    IDetector(const IDetector &) = delete;
    IDetector &operator=(const IDetector &) = delete;
    IDetector(IDetector &&) = delete;
    IDetector &operator=(IDetector &&) = delete;

    [[nodiscard]] virtual const ModelInfo &info() const = 0;

    // Minimum score for a face to be reported
    virtual void setScoreThreshold(float threshold) = 0;

    // Find faces in an image. The result stays valid until the next detect() call.
    [[nodiscard]] virtual std::span<const Face> detect(const ImageView &image, DetectTiming &timing) = 0;

protected:
    IDetector() = default;
};
