#include "memory/MemoryProbe.h"
#include "esp_heap_caps.h"
#include "esp_log.h"

namespace {
constexpr const char *TAG = "memprobe";
} // namespace

MemoryProbe::MemoryProbe() : start_(snapshot()) {
    monitoring_ = heap_caps_monitor_local_minimum_free_size_start() == ESP_OK;
    if (!monitoring_) {
        ESP_LOGW(TAG, "Another probe is running - peak memory won't be measured");
    }
}

MemoryProbe::~MemoryProbe() {
    if (monitoring_) heap_caps_monitor_local_minimum_free_size_stop();
}

HeapSnapshot MemoryProbe::snapshot() {
    return HeapSnapshot{
        .freeInternal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
        .freePsram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
    };
}

MemoryUsage MemoryProbe::usage() const {
    HeapSnapshot now = snapshot();
    MemoryUsage usage;
    usage.internalBytes = static_cast<int32_t>(start_.freeInternal) - static_cast<int32_t>(now.freeInternal);
    usage.psramBytes = static_cast<int32_t>(start_.freePsram) - static_cast<int32_t>(now.freePsram);
    // While monitoring, the minimum is counted from when this probe started instead of from boot
    usage.minFreeInternal = monitoring_ ? heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL) : now.freeInternal;
    usage.minFreePsram = monitoring_ ? heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM) : now.freePsram;
    return usage;
}
