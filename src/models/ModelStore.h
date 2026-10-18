#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include "Storage.h"
#include "memory/HeapBuffer.h"

// Where a model's bytes live while it runs
enum class Placement : uint8_t {
    FlashXip = 0,   // mapped flash, read through the cache (no copy)
    Psram = 1,      // copied into PSRAM
    Internal = 2,   // copied into internal SRAM (fastest, smallest)
};

[[nodiscard]] const char *placementName(Placement placement);

// One file in the model pack
struct ModelFile {
    std::string name;
    std::string format;     // "tflite", ...
    uint32_t offset = 0;    // from the start of the partition
    uint32_t size = 0;
    uint32_t crc32 = 0;
};

/**
 * A model file ready to use, wherever it was placed. Owns its mapping or copy and releases it
 * when destroyed; move-only.
 */
class ModelBlob {

private:
    MappedRegion region_;   // FlashXip
    HeapBuffer copy_;       // Psram / Internal
    Placement placement_;

public:
    explicit ModelBlob(MappedRegion region) : region_(std::move(region)), placement_(Placement::FlashXip) {}
    ModelBlob(HeapBuffer copy, Placement placement) : copy_(std::move(copy)), placement_(placement) {}
    ~ModelBlob() = default;

    ModelBlob(const ModelBlob &) = delete;
    ModelBlob &operator=(const ModelBlob &) = delete;
    ModelBlob(ModelBlob &&) noexcept = default;
    ModelBlob &operator=(ModelBlob &&) noexcept = default;

    [[nodiscard]] std::span<const uint8_t> data() const {
        return placement_ == Placement::FlashXip ? region_.data() : copy_.span();
    }
    [[nodiscard]] Placement placement() const { return placement_; }
};

/**
 * Model files deployed to the `model` flash partition (see ModelPack.h). Flashing new models
 * doesn't need a firmware rebuild: tools/models/pack_models.py + flash_models.py.
 */
class ModelStore {

private:
    Partition partition_;
    std::vector<ModelFile> files_;

    void readTable();

public:
    // Reads the pack's table. A partition that was never flashed gives an empty store.
    explicit ModelStore(const std::string &label = "model");
    ModelStore(const ModelStore &) = delete;
    ModelStore &operator=(const ModelStore &) = delete;
    ModelStore(ModelStore &&) = delete;
    ModelStore &operator=(ModelStore &&) = delete;

    [[nodiscard]] std::span<const ModelFile> files() const { return files_; }
    [[nodiscard]] const ModelFile *find(std::string_view name) const;

    // Map or copy a model and check its CRC. Empty if it's missing, doesn't fit or is corrupt.
    [[nodiscard]] std::optional<ModelBlob> load(std::string_view name, Placement placement) const;
};
