#include "app/DetectorSlot.h"
#include "esp_log.h"
#include "esp_timer.h"

namespace {
constexpr const char *TAG = "detector";
} // namespace

LoadReport DetectorSlot::load(std::string_view id) {
    if (!registry_.find(id)) {
        ESP_LOGE(TAG, "Unknown model '%.*s'", static_cast<int>(id.size()), id.data());
        return LoadReport{};
    }

    // Free the old model before measuring, so the probe only sees the new one
    detector_.reset();

    LoadReport report;
    {
        MemoryProbe probe;
        int64_t start = esp_timer_get_time();
        detector_ = registry_.create(id);
        report.loadUs = static_cast<uint32_t>(esp_timer_get_time() - start);
        report.memory = probe.usage();
    }
    report.ok = detector_ != nullptr;
    if (detector_ && scoreThreshold_) detector_->setScoreThreshold(*scoreThreshold_);

    ESP_LOGI(TAG, "Model %.*s %s in %lu ms: internal %+ld B, psram %+ld B", static_cast<int>(id.size()), id.data(),
             report.ok ? "loaded" : "FAILED", static_cast<unsigned long>(report.loadUs / 1000),
             static_cast<long>(report.memory.internalBytes), static_cast<long>(report.memory.psramBytes));
    lastLoad_ = report;
    return report;
}

void DetectorSlot::setScoreThreshold(float threshold) {
    scoreThreshold_ = threshold;
    if (detector_) detector_->setScoreThreshold(threshold);
}
