#pragma once

/**
 * Constant-velocity Kalman filter for one axis of the face's direction, in mount degrees.
 *
 * State: angle (deg) and angular velocity (deg/s). Measurements are the face angle worked out
 * from each detection. The filter smooths detection jitter and, through the velocity, predicts
 * where the face will be once the servos get there, which makes up for the frame delay.
 */
class AxisFilter {

private:
    float accelNoise_;      // how quickly the face may change speed, deg/s^2 (1 sigma)
    float measNoise_;       // detection noise, deg (1 sigma)
    float gate_;            // reject measurements further than this many sigmas from the prediction

    bool ready_ = false;
    float angle_ = 0.0f;
    float velocity_ = 0.0f;
    // Covariance [[p00, p01], [p01, p11]]
    float p00_ = 0.0f;
    float p01_ = 0.0f;
    float p11_ = 0.0f;

public:
    static constexpr float kInitialVelocitySigma = 60.0f;   // deg/s: unknown at first sight

    AxisFilter(float accelNoise, float measNoise, float gate)
        : accelNoise_(accelNoise), measNoise_(measNoise), gate_(gate) {}

    void reset() { ready_ = false; }

    // Start from a first measurement, velocity unknown
    void init(float angle) {
        ready_ = true;
        angle_ = angle;
        velocity_ = 0.0f;
        p00_ = measNoise_ * measNoise_;
        p01_ = 0.0f;
        p11_ = kInitialVelocitySigma * kInitialVelocitySigma;
    }

    // Move the estimate forward by dt seconds
    void predict(float dt) {
        if (!ready_ || dt <= 0.0f) return;
        angle_ += velocity_ * dt;
        // P = F P F' + Q, F = [[1, dt], [0, 1]], Q from white-noise acceleration
        float q = accelNoise_ * accelNoise_;
        float dt2 = dt * dt;
        p00_ += dt * (2.0f * p01_ + dt * p11_) + q * dt2 * dt2 / 4.0f;
        p01_ += dt * p11_ + q * dt2 * dt / 2.0f;
        p11_ += q * dt2;
    }

    // Correct with a measured angle. False if it was too far from the prediction to trust.
    bool update(float measured) {
        if (!ready_) {
            init(measured);
            return true;
        }
        float innovation = measured - angle_;
        float s = p00_ + measNoise_ * measNoise_;
        if (innovation * innovation > gate_ * gate_ * s) return false;

        float k0 = p00_ / s;
        float k1 = p01_ / s;
        angle_ += k0 * innovation;
        velocity_ += k1 * innovation;
        // P = (I - K H) P
        float p00 = p00_, p01 = p01_;
        p00_ -= k0 * p00;
        p01_ -= k0 * p01;
        p11_ -= k1 * p01;
        return true;
    }

    // Where the face is expected to be `ahead` seconds from the current estimate
    [[nodiscard]] float predicted(float ahead) const { return angle_ + velocity_ * ahead; }

    [[nodiscard]] bool ready() const { return ready_; }
    [[nodiscard]] float angle() const { return angle_; }
    [[nodiscard]] float velocity() const { return velocity_; }
};
