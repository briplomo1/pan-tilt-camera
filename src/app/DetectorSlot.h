#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include "memory/MemoryProbe.h"
#include "vision/DetectorRegistry.h"

// What loading a model cost
struct LoadReport {
    bool ok = false;
    uint32_t loadUs = 0;
    MemoryUsage memory;
};

/**
 * Holds the one detector that is resident at a time. Loading a new model destroys the old one
 * first, so two models never compete for the S3's RAM, and measures what the new one takes.
 * Only used from the vision task.
 */
class DetectorSlot {

private:
    const DetectorRegistry &registry_;
    std::unique_ptr<IDetector> detector_;
    LoadReport lastLoad_;
    std::optional<float> scoreThreshold_;   // applied to every model loaded after it's set

public:
    explicit DetectorSlot(const DetectorRegistry &registry) : registry_(registry) {}
    DetectorSlot(const DetectorSlot &) = delete;
    DetectorSlot &operator=(const DetectorSlot &) = delete;
    DetectorSlot(DetectorSlot &&) = delete;
    DetectorSlot &operator=(DetectorSlot &&) = delete;

    // Swap to another model. An unknown id leaves the current model loaded.
    LoadReport load(std::string_view id);

    void setScoreThreshold(float threshold);

    [[nodiscard]] IDetector *get() const { return detector_.get(); }
    [[nodiscard]] const LoadReport &lastLoad() const { return lastLoad_; }
};
