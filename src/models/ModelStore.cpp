#include "models/ModelStore.h"
#include <cstring>
#include "esp_log.h"
#include "esp_rom_crc.h"
#include "models/ModelPack.h"

namespace {
constexpr const char *TAG = "models";

std::string fixedString(const char *text, size_t capacity) {
    return std::string(text, strnlen(text, capacity));
}

uint32_t crc32(std::span<const uint8_t> data) {
    return esp_rom_crc32_le(0, data.data(), data.size());
}
} // namespace

const char *placementName(Placement placement) {
    switch (placement) {
    case Placement::FlashXip: return "flash-xip";
    case Placement::Psram: return "psram";
    case Placement::Internal: return "internal";
    }
    return "?";
}

ModelStore::ModelStore(const std::string &label) : partition_(Storage::getPartition(label)) {
    readTable();
}

void ModelStore::readTable() {
    modelpack::Header header;
    partition_.read(0, {reinterpret_cast<uint8_t *>(&header), sizeof(header)});

    if (std::memcmp(header.magic, modelpack::kMagic, sizeof(header.magic)) != 0) {
        ESP_LOGI(TAG, "No model pack in partition '%.*s' (flash one with tools/models/flash_models.py)",
                 static_cast<int>(partition_.label().size()), partition_.label().data());
        return;
    }
    if (header.version != modelpack::kVersion || header.count > modelpack::kMaxEntries) {
        ESP_LOGE(TAG, "Unsupported model pack (version %u, %u entries)", header.version, header.count);
        return;
    }

    std::vector<modelpack::Entry> table(header.count);
    std::span<uint8_t> tableBytes(reinterpret_cast<uint8_t *>(table.data()), table.size() * sizeof(modelpack::Entry));
    partition_.read(sizeof(header), tableBytes);
    if (crc32(tableBytes) != header.tableCrc) {
        ESP_LOGE(TAG, "Model pack table is corrupt - reflash it");
        return;
    }

    for (const modelpack::Entry &entry : table) {
        bool inside = entry.offset <= partition_.size() && entry.size <= partition_.size() - entry.offset;
        if (!inside || entry.size == 0 || entry.offset % modelpack::kAlignment != 0) {
            ESP_LOGE(TAG, "Skipping bad entry '%.32s'", entry.name);
            continue;
        }
        files_.push_back(ModelFile{
            .name = fixedString(entry.name, sizeof(entry.name)),
            .format = fixedString(entry.format, sizeof(entry.format)),
            .offset = entry.offset,
            .size = entry.size,
            .crc32 = entry.crc32,
        });
        ESP_LOGI(TAG, "Model file %s (%s, %lu bytes)", files_.back().name.c_str(),
                 files_.back().format.c_str(), static_cast<unsigned long>(entry.size));
    }
}

const ModelFile *ModelStore::find(std::string_view name) const {
    for (const ModelFile &file : files_) {
        if (file.name == name) return &file;
    }
    return nullptr;
}

std::optional<ModelBlob> ModelStore::load(std::string_view name, Placement placement) const {
    const ModelFile *file = find(name);
    if (!file) {
        ESP_LOGE(TAG, "No model file '%.*s'", static_cast<int>(name.size()), name.data());
        return std::nullopt;
    }

    std::optional<ModelBlob> blob;
    if (placement == Placement::FlashXip) {
        blob.emplace(partition_.map(file->offset, file->size));
    } else {
        HeapBuffer copy = HeapBuffer::allocate(file->size, placement == Placement::Psram ? kPsramCaps : kInternalCaps);
        if (!copy) {
            ESP_LOGE(TAG, "No room for %s (%lu bytes) in %s", file->name.c_str(),
                     static_cast<unsigned long>(file->size), placementName(placement));
            return std::nullopt;
        }
        partition_.read(file->offset, copy.span());
        blob.emplace(std::move(copy), placement);
    }

    if (crc32(blob->data()) != file->crc32) {
        ESP_LOGE(TAG, "Model file %s failed its CRC check - reflash the pack", file->name.c_str());
        return std::nullopt;
    }
    return blob;
}
