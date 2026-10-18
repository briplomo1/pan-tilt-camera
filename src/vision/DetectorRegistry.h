#pragma once

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include "vision/IDetector.h"

class ModelStore;
struct DetectorEntry;

// Builds a detector. `store` gives backends access to model files in the `model` partition.
using DetectorFactory = std::unique_ptr<IDetector> (*)(const DetectorEntry &entry, const ModelStore &store);

struct DetectorEntry {
    std::string id;           // what the host asks for, e.g. "espdl-msrmnp-s8"
    std::string backend;      // "esp-dl", "tflite-micro", ...
    std::string modelName;    // file in the model partition; empty for models built into the app
    DetectorFactory create = nullptr;
};

/**
 * Every detector the firmware can run. Models built into the app are always listed. Model
 * files flashed into the `model` partition appear for each backend registered for their format,
 * so deploying a new model is: pack it, flash it, and it shows up here by name.
 */
class DetectorRegistry {

private:
    const ModelStore &store_;
    std::vector<DetectorEntry> entries_;

public:
    explicit DetectorRegistry(const ModelStore &store);
    DetectorRegistry(const DetectorRegistry &) = delete;
    DetectorRegistry &operator=(const DetectorRegistry &) = delete;
    DetectorRegistry(DetectorRegistry &&) = delete;
    DetectorRegistry &operator=(DetectorRegistry &&) = delete;

    void add(DetectorEntry entry);

    // One entry per model file of `format` in the store, with id "<backend>/<file name>"
    void addStoreBackend(std::string_view format, std::string_view backend, DetectorFactory factory);

    [[nodiscard]] std::span<const DetectorEntry> entries() const { return entries_; }
    [[nodiscard]] const DetectorEntry *find(std::string_view id) const;

    // Build a detector by id; nullptr if unknown. Destroy the current detector first so only
    // one model is ever resident.
    [[nodiscard]] std::unique_ptr<IDetector> create(std::string_view id) const;
};
