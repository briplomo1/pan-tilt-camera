#pragma once

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/ledc.h"
#include "esp_log.h"
#include "driver/gpio.h"

/**
 * Hobby servo driven by one LEDC channel. All servos share LEDC_TIMER_0 at 50 Hz.
 * Channels are handed out from 0 upward; the camera's XCLK uses LEDC_TIMER_1 /
 * LEDC_CHANNEL_7 so keep the servo count below 7.
 */
class Servo {

    private:
    gpio_num_t pin_;
    ledc_channel_t channel_;
    int lowUs_ = 500;     // pulse for 0 degrees
    int highUs_ = 2500;   // pulse for 180 degrees
    static inline bool timerReady_ = false;
    static constexpr int minUs_ = 400;
    static constexpr int maxUs_ = 2600;
    static inline uint32_t usedChannels_ = 0;

    public:

    explicit Servo(gpio_num_t pin);
    ~Servo();
    Servo(const Servo &) = delete;
    Servo &operator=(const Servo &) = delete;
    Servo(Servo &&) = delete;
    Servo &operator=(Servo &&) = delete;

    // Pulse widths (us) that map to 0 and 180 degrees
    void setRange(int low_us, int high_us);
    // Write a raw pulse width, clamped to the safe raw range
    void writeUS(int us) const;
    // Write an angle in degrees (0..180) using the calibrated range
    void writeAngle(float angle) const;
    // Stop sending pulses (servo goes limp)
    void stop() const;

    [[nodiscard]] gpio_num_t pin() const { return pin_; }
    [[nodiscard]] int minUs() const { return minUs_; }
    [[nodiscard]] int maxUs() const { return maxUs_; }

};
