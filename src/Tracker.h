#pragma once

#include <optional>
#include <vector>
#include "FaceDetector.h"
#include "PanTilt.h"

/**
 * Scan / track loop.
 *
 *  SCAN:  sweep pan back and forth, stepping tilt to a new row at each end,
 *         running face detection on every frame.
 *  TRACK: steer so the face sits in the middle of the frame. If no face is seen
 *         for kLostTimeoutMs, go back to SCAN from wherever the mount is.
 */
class Tracker {

private:
    enum class State { Scan, Track };

    PanTilt &mount_;
    FaceDetector &detector_;

    State state_ = State::Scan;
    int scanDir_ = 1;              // +1 sweeping toward pan max, -1 toward pan min
    size_t scanRow_ = 0;
    int64_t lastSeenUs_ = 0;
    std::optional<Face> lastFace_;

    void enterScan();
    void enterTrack();
    void scanStep();
    void steerToward(const Face &face, int frameW, int frameH, float panAtCapture, float tiltAtCapture);
    [[nodiscard]] std::optional<Face> pickFace(const std::vector<Face> &faces) const;
    [[nodiscard]] float scanRowTilt() const;

public:
    Tracker(PanTilt &mount, FaceDetector &detector) : mount_(mount), detector_(detector) {}

    [[noreturn]] void run();
};
