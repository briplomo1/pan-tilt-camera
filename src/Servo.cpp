#include "Servo.h"
#include <algorithm>
#include "esp_log.h"

namespace {
    constexpr const char *TAG = "servo";

    constexpr uint32_t freqHz = 50;
    constexpr uint32_t periodUs = 20000;
    constexpr ledc_timer_bit_t resolution = LEDC_TIMER_14_BIT;
    constexpr uint32_t dutySteps = uint32_t{1} << 14;
    constexpr ledc_timer_t timer = LEDC_TIMER_0;
    constexpr ledc_mode_t mode = LEDC_LOW_SPEED_MODE;

} // namespace

Servo::Servo(gpio_num_t pin) : pin_(pin) {
    int freeIndex = -1;
    for (int i = 0; i < LEDC_CHANNEL_MAX; ++i) {
        if (!(usedChannels_ & (1u << i))) {
            freeIndex = i;
            break;
        }
    }

    if (freeIndex < 0) {
        ESP_LOGE(TAG, "No free ledc channel for servo on pin %d", pin_);
        ESP_ERROR_CHECK(ESP_ERR_NO_MEM);
    }
    channel_ = static_cast<ledc_channel_t>(freeIndex);
    usedChannels_ |= 1u << freeIndex;

    if (!timerReady_) {
        ledc_timer_config_t config = {};

        config.speed_mode = mode;
        config.timer_num = timer;
        config.duty_resolution = resolution;
        config.freq_hz = freqHz;
        config.clk_cfg = LEDC_AUTO_CLK;

        ESP_ERROR_CHECK(ledc_timer_config(&config));
        timerReady_ = true;
    }

    ledc_channel_config_t ch = {};
    ch.gpio_num = pin_;
    ch.speed_mode = mode;
    ch.channel = channel_;
    ch.timer_sel = timer;
    ch.duty = 0;
    ch.hpoint = 0;
    ESP_ERROR_CHECK(ledc_channel_config(&ch));
}

Servo::~Servo() {
    stop();
    usedChannels_ &= ~(1u << channel_);
}

void Servo::setRange(int low_us, int high_us) {
    lowUs_ = std::clamp(low_us, minUs_, maxUs_);
    highUs_ = std::clamp(high_us, minUs_, maxUs_);
}

void Servo::writeUS(int us) const {
    us = std::clamp(us, minUs_, maxUs_);
    uint32_t duty = static_cast<uint32_t>(us) * dutySteps / periodUs;
    ledc_set_duty(mode, channel_, duty);
    ledc_update_duty(mode, channel_);
}

void Servo::writeAngle(float angle) const {
    angle = std::clamp(angle, 0.0f, 180.0f);
    writeUS(lowUs_ + static_cast<int>(angle * (highUs_ - lowUs_) / 180.0f + 0.5f));
}

void Servo::stop() const {
    ledc_stop(mode, channel_, 0);
}
