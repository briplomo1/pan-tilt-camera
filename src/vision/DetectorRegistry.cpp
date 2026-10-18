#include "vision/DetectorRegistry.h"
#include "sdkconfig.h"
#include "esp_log.h"
#include "models/ModelStore.h"
#include "vision/EspDlDetector.h"

namespace {
constexpr const char *TAG = "registry";

template <EspDlDetector::Model M>
std::unique_ptr<IDetector> makeEspDl(const DetectorEntry &, const ModelStore &) {
    return std::make_unique<EspDlDetector>(M);
}
} // namespace

DetectorRegistry::DetectorRegistry(const ModelStore &store) : store_(store) {
    // ESP-DL models compiled into the app. The Pico models are only there when their
    // CONFIG_FLASH_* option is set, which needs its own build (app slot is 3 MB).
    add({"espdl-msrmnp-s8", "esp-dl", "", &makeEspDl<EspDlDetector::Model::MsrMnp>});
#ifdef CONFIG_FLASH_ESPDET_PICO_224_224_FACE
    add({"espdl-espdet-pico-224-s8", "esp-dl", "", &makeEspDl<EspDlDetector::Model::EspDetPico224>});
#endif
#ifdef CONFIG_FLASH_ESPDET_PICO_416_416_FACE
    add({"espdl-espdet-pico-416-s8", "esp-dl", "", &makeEspDl<EspDlDetector::Model::EspDetPico416>});
#endif
}

void DetectorRegistry::add(DetectorEntry entry) {
    if (find(entry.id)) {
        ESP_LOGW(TAG, "Detector '%s' already registered", entry.id.c_str());
        return;
    }
    ESP_LOGI(TAG, "Detector available: %s (%s)", entry.id.c_str(), entry.backend.c_str());
    entries_.push_back(std::move(entry));
}

void DetectorRegistry::addStoreBackend(std::string_view format, std::string_view backend, DetectorFactory factory) {
    for (const ModelFile &file : store_.files()) {
        if (file.format != format) continue;
        std::string id = std::string(backend) + "/" + file.name;
        add({std::move(id), std::string(backend), file.name, factory});
    }
}

const DetectorEntry *DetectorRegistry::find(std::string_view id) const {
    for (const DetectorEntry &entry : entries_) {
        if (entry.id == id) return &entry;
    }
    return nullptr;
}

std::unique_ptr<IDetector> DetectorRegistry::create(std::string_view id) const {
    const DetectorEntry *entry = find(id);
    if (!entry || !entry->create) {
        ESP_LOGE(TAG, "No detector named '%.*s'", static_cast<int>(id.size()), id.data());
        return nullptr;
    }
    return entry->create(*entry, store_);
}
