#include "PanTilt.h"
#include <cmath>
#include "freertos/task.h"
#include "esp_log.h"

namespace {
constexpr const char *TAG = "pantilt";
constexpr int kUpdateMs = 20;          // one servo PWM period
constexpr int kTaskPriority = 5;       // above the tracker so motion never stutters
constexpr int kTaskStack = 3072;

// Move 'pos' toward 'target' by at most 'step'
float approach(float pos, float target, float step) {
    float diff = target - pos;
    if (std::fabs(diff) <= step) return target;
    return pos + (diff > 0 ? step : -step);
}
} // namespace

void PanTilt::start(float pan, float tilt) {
    pan = panRange_.clamp(pan);
    tilt = tiltRange_.clamp(tilt);
    panPos_ = panTarget_ = pan;
    tiltPos_ = tiltTarget_ = tilt;
    pan_.writeAngle(pan);
    tilt_.writeAngle(tilt);

    // Core 0, so the detector on core 1 can't starve it
    xTaskCreatePinnedToCore(taskEntry, "pantilt", kTaskStack, this, kTaskPriority, nullptr, 0);
    ESP_LOGI(TAG, "Motion task started at pan %.1f tilt %.1f", pan, tilt);
}

void PanTilt::setTarget(float pan, float tilt) {
    pan = panRange_.clamp(pan);
    tilt = tiltRange_.clamp(tilt);
    taskENTER_CRITICAL(&lock_);
    panTarget_ = pan;
    tiltTarget_ = tilt;
    taskEXIT_CRITICAL(&lock_);
}

void PanTilt::setMaxSpeed(float degPerSec) {
    taskENTER_CRITICAL(&lock_);
    maxSpeed_ = degPerSec;
    taskEXIT_CRITICAL(&lock_);
}

float PanTilt::pan() const {
    taskENTER_CRITICAL(&lock_);
    float v = panPos_;
    taskEXIT_CRITICAL(&lock_);
    return v;
}

float PanTilt::tilt() const {
    taskENTER_CRITICAL(&lock_);
    float v = tiltPos_;
    taskEXIT_CRITICAL(&lock_);
    return v;
}

void PanTilt::taskEntry(void *arg) {
    static_cast<PanTilt *>(arg)->run();
}

void PanTilt::run() {
    TickType_t wake = xTaskGetTickCount();
    float lastPan = -1.0f, lastTilt = -1.0f;

    while (true) {
        taskENTER_CRITICAL(&lock_);
        float step = maxSpeed_ * kUpdateMs / 1000.0f;
        panPos_ = approach(panPos_, panTarget_, step);
        tiltPos_ = approach(tiltPos_, tiltTarget_, step);
        float pan = panPos_, tilt = tiltPos_;
        taskEXIT_CRITICAL(&lock_);

        // Only touch the hardware when something changed
        if (pan != lastPan) { pan_.writeAngle(pan); lastPan = pan; }
        if (tilt != lastTilt) { tilt_.writeAngle(tilt); lastTilt = tilt; }

        vTaskDelayUntil(&wake, pdMS_TO_TICKS(kUpdateMs));
    }
}
