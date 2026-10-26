#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include "FaceFilter.h"
#include "PanTilt.h"
#include "vision/IDetector.h"

enum class TrackState : uint8_t { Scan = 0, Track = 1, Paused = 2 };

// Controller settings the host may change at run time (SetTracker), for A/B runs without a reflash.
// Defaults are the frozen values in Tracker.cpp (docs/EXPERIMENT_PLAN.md §8).
struct TrackerOptions {
    bool filter;           // Kalman-filter the face and aim ahead; false: plain proportional steering
    float leadS;           // aim this far past the frame's processing time
    float coastS;          // keep following the prediction this long after a missed detection
};

// What happened in one tracker step, for telemetry and the live view
struct TrackSample {
    uint32_t frameId = 0;
    int64_t captureUs = 0;
    TrackState state = TrackState::Scan;
    float pan = 0.0f;                // mount angles when the frame was captured
    float tilt = 0.0f;
    float panTarget = 0.0f;          // target set this step
    float tiltTarget = 0.0f;
    std::span<const Face> faces;
    int targetIndex = -1;            // face being followed, -1 for none
    DetectTiming timing;
    ImageView image;                 // camera pixels; only valid during the callback
    // Kalman estimate of the face direction (mount degrees), when the filter has one
    bool filterOn = false;
    bool filterReady = false;
    bool coasting = false;           // steering on the prediction through a missed detection
    float facePan = 0.0f;
    float faceTilt = 0.0f;
    float facePanVelocity = 0.0f;    // deg/s
    float faceTiltVelocity = 0.0f;
};

class ITrackObserver {

public:
    virtual ~ITrackObserver() = default;
    ITrackObserver(const ITrackObserver &) = delete;
    ITrackObserver &operator=(const ITrackObserver &) = delete;
    ITrackObserver(ITrackObserver &&) = delete;
    ITrackObserver &operator=(ITrackObserver &&) = delete;

    // Called once per frame while the camera buffer is still held
    virtual void onTrackFrame(const TrackSample &sample) = 0;

protected:
    ITrackObserver() = default;
};

/**
 * Scan / track logic, one camera frame per step().
 *
 *  SCAN:   step pan across its range, pausing at each stop so the camera gets a
 *          sharp frame, and move tilt to a new row at each end. Face detection
 *          runs on every frame.
 *  TRACK:  steer so the face sits in the middle of the frame. With the filter on, the face
 *          direction is Kalman-filtered and the mount aims where the face will be; a short miss
 *          (coastS) keeps following the prediction. If no face is seen for kLostTimeoutUs, go
 *          back to SCAN from wherever the mount is.
 *  PAUSED: mount held still (benchmark mode); step() does nothing.
 */
class Tracker {

private:
    PanTilt &mount_;
    ITrackObserver *observer_ = nullptr;

    TrackState state_ = TrackState::Scan;
    int scanDir_ = 1;              // +1 sweeping toward pan max, -1 toward pan min
    size_t scanRow_ = 0;
    int64_t scanArrivedUs_ = 0;    // when the mount reached the current scan stop, 0 while moving
    int64_t lastSeenUs_ = 0;
    std::optional<Face> lastFace_;
    float panTarget_ = 0.0f;
    float tiltTarget_ = 0.0f;

    // Face direction in mount angles, smoothed and with velocity (see Tracker.cpp tuning)
    AxisFilter panFilter_;
    AxisFilter tiltFilter_;
    int64_t filterUs_ = 0;         // capture time the filters were last updated for
    int rejects_ = 0;              // outlier detections in a row
    bool coasting_ = false;
    TrackerOptions options_;

    uint32_t frameId_ = 0;
    int64_t lastStatsUs_ = 0;
    int frames_ = 0;

    void enterScan();
    void enterTrack();
    void scanStep(int64_t now);
    void steerToward(const Face &face, int frameW, int frameH, float panAtCapture, float tiltAtCapture,
                     int64_t captureUs, int64_t now);
    void filterFace(float facePan, float faceTilt, int64_t captureUs);
    void coast(float panAtCapture, float tiltAtCapture, int64_t now);
    [[nodiscard]] float aim(const AxisFilter &filter, float atCapture, float ahead, float halfFovDeg) const;
    void setMountTarget(float pan, float tilt);
    [[nodiscard]] int pickFace(std::span<const Face> faces) const;
    [[nodiscard]] float scanRowTilt() const;
    void logStats(int64_t now);

public:
    explicit Tracker(PanTilt &mount);
    Tracker(const Tracker &) = delete;
    Tracker &operator=(const Tracker &) = delete;
    Tracker(Tracker &&) = delete;
    Tracker &operator=(Tracker &&) = delete;

    void setObserver(ITrackObserver *observer) { observer_ = observer; }

    // Capture one frame, find faces with `detector` and steer. Call repeatedly.
    void step(IDetector &detector);

    // Hold the mount where it is until resume()
    void pause();

    // Start scanning again from the current position
    void resume();

    [[nodiscard]] TrackState state() const { return state_; }

    [[nodiscard]] static TrackerOptions defaultOptions();
    [[nodiscard]] const TrackerOptions &options() const { return options_; }
    // Takes effect from the next frame; the filter restarts from the next detection
    void setOptions(const TrackerOptions &options);
};
