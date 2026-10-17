#pragma once

#include <algorithm>
#include "freertos/FreeRTOS.h"
#include "Servo.h"

// Allowed angle range for one axis, in degrees
struct Range {
    float min;
    float max;

    [[nodiscard]] float clamp(float v) const { return std::clamp(v, min, max); }
    [[nodiscard]] float center() const { return (min + max) / 2.0f; }
};

/**
 * Two-axis mount. Callers set a target position; a background task moves the
 * servos toward it at no more than the configured speed, so motion stays smooth
 * no matter how often (or how jumpily) targets are updated.
 */
class PanTilt {

private:
    Servo &pan_;
    Servo &tilt_;
    Range panRange_;
    Range tiltRange_;

    mutable portMUX_TYPE lock_ = portMUX_INITIALIZER_UNLOCKED;
    float panPos_ = 90.0f;
    float tiltPos_ = 90.0f;
    float panTarget_ = 90.0f;
    float tiltTarget_ = 90.0f;
    float maxSpeed_ = 60.0f;   // degrees per second

    static void taskEntry(void *arg);
    [[noreturn]] void run();

public:
    PanTilt(Servo &pan, Servo &tilt, Range panRange, Range tiltRange)
        : pan_(pan), tilt_(tilt), panRange_(panRange), tiltRange_(tiltRange) {}

    PanTilt(const PanTilt &) = delete;
    PanTilt &operator=(const PanTilt &) = delete;

    // Jump to a starting position and start the motion task
    void start(float pan, float tilt);

    // New position to move toward (clamped to the axis ranges)
    void setTarget(float pan, float tilt);

    void setMaxSpeed(float degPerSec);

    // Where the servos have been commanded to right now
    [[nodiscard]] float pan() const;
    [[nodiscard]] float tilt() const;

    [[nodiscard]] const Range &panRange() const { return panRange_; }
    [[nodiscard]] const Range &tiltRange() const { return tiltRange_; }
};
