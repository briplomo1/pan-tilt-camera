#include "Tracker.h"
#include <algorithm>
#include <array>
#include <cmath>
#include "esp_log.h"
#include "esp_timer.h"
#include "Camera.h"
#include "Delay.h"

using namespace std::chrono_literals;

namespace {
constexpr const char *TAG = "tracker";

// --- Tuning -----------------------------------------------------------------
// Fixed for every model from the B0 baseline on (docs/EXPERIMENT_PLAN.md §8), so
// models are compared under the same controller. Don't retune per model.

// Camera field of view (OV2640 stock lens at 4:3). Used to turn a pixel offset
// into degrees; only needs to be roughly right.
constexpr float kHFovDeg = 54.0f;
constexpr float kVFovDeg = 41.0f;

// Flip to -1 if the mount moves away from the face instead of toward it.
// +1 pan: face right of center -> pan angle increases.
// +1 tilt: face below center -> tilt angle increases.
constexpr float kPanDir = -1.0f;   // reversed on our mount: +1 panned away from the face
constexpr float kTiltDir = 1.0f;

// Smooth the face position with a Kalman filter and aim where it will be, instead of steering
// straight from each raw detection. false restores the plain proportional controller below.
// This and kLeadS / kCoastS are only defaults: the host can change them at run time (SetTracker).
constexpr bool kFilterFace = true;
constexpr float kFaceAccelNoise = 150.0f;   // deg/s^2: how sharply a head changes speed
constexpr float kFaceMeasNoise = 0.6f;      // deg: detection jitter (~3-4 px at 320 wide)
constexpr float kFaceGate = 4.0f;           // sigmas; further detections are treated as outliers
constexpr int kMaxRejects = 3;              // this many outliers in a row means the face really moved
// Aim this far past the frame's processing time: roughly how long the servos take to respond
constexpr float kLeadS = 0.05f;
// After a missed detection keep following the prediction this long, then hold still. The stock
// model's misses while tracking are mostly single frames (< 0.2 s).
constexpr float kCoastS = 0.3f;
// A miss longer than this leaves the velocity estimate meaningless: restart the filter
constexpr int64_t kFilterStaleUs = 500 * 1000;

// Without the filter: fraction of the measured error corrected per frame. <1 avoids overshoot,
// since each frame is already ~50-80 ms old by the time it's processed.
constexpr float kGain = 0.6f;

// Ignore errors smaller than this fraction of half the frame (stops jitter).
// tools/view/live_view.py draws the same box; update it there too.
constexpr float kDeadband = 0.06f;

constexpr float kTrackSpeedDegS = 90.0f;   // faster blurred the face and the detector missed it

// Scan sweeps slowly and pauses every kScanStepDeg so the camera gets sharp frames: a
// non-stop 30°/s sweep blurred faces past detection, and fast hops between stops were jerky.
constexpr float kScanStepDeg = 20.0f;          // well under the 54° FOV, so neighbouring views overlap
constexpr float kScanMoveSpeedDegS = 20.0f;
constexpr int64_t kScanDwellUs = 500 * 1000;   // servo settles (~250 ms), then a few frames to detect on

// Tilt rows the scan visits, as offsets from the calibrated level tilt
constexpr std::array<float, 3> kScanRowOffsets = {0.0f, -20.0f, 20.0f};

// Hold position this long without a detection before scanning again. The stock model
// misses runs of frames (measured up to ~1.3 s) with the face still in view.
constexpr int64_t kLostTimeoutUs = 3000 * 1000;

// ----------------------------------------------------------------------------

float applyDeadband(float err) {
    return std::fabs(err) < kDeadband ? 0.0f : err;
}
} // namespace

Tracker::Tracker(PanTilt &mount)
    : mount_(mount),
      panFilter_(kFaceAccelNoise, kFaceMeasNoise, kFaceGate),
      tiltFilter_(kFaceAccelNoise, kFaceMeasNoise, kFaceGate),
      options_(defaultOptions()) {}

TrackerOptions Tracker::defaultOptions() {
    return TrackerOptions{.filter = kFilterFace, .leadS = kLeadS, .coastS = kCoastS};
}

void Tracker::setOptions(const TrackerOptions &options) {
    options_ = options;
    panFilter_.reset();
    tiltFilter_.reset();
    coasting_ = false;
    ESP_LOGI(TAG, "Options: filter %s, lead %.0f ms, coast %.0f ms", options_.filter ? "on" : "off",
             options_.leadS * 1000.0f, options_.coastS * 1000.0f);
}

void Tracker::step(IDetector &detector) {
    if (state_ == TrackState::Paused) return;
    if (lastStatsUs_ == 0) {
        // First frame: start scanning from wherever the mount is
        enterScan();
        lastStatsUs_ = esp_timer_get_time();
    }

    Frame frame = Camera::capture();
    ImageView image = frame.view();
    if (!image) {
        ESP_LOGW(TAG, "Camera capture failed");
        delay(100ms);
        return;
    }
    // Where the mount was pointing when this frame was taken
    int64_t captureUs = esp_timer_get_time();
    float panAtCapture = mount_.pan();
    float tiltAtCapture = mount_.tilt();

    DetectTiming timing;
    std::span<const Face> faces = detector.detect(image, timing);

    int64_t now = esp_timer_get_time();
    int target = pickFace(faces);

    if (target >= 0) {
        const Face &face = faces[target];
        if (state_ == TrackState::Scan) {
            ESP_LOGI(TAG, "Face found (score %.2f) at pan %.1f tilt %.1f - tracking",
                     face.score, panAtCapture, tiltAtCapture);
            enterTrack();
        }
        lastSeenUs_ = now;
        lastFace_ = face;
        coasting_ = false;
        steerToward(face, image.width, image.height, panAtCapture, tiltAtCapture, captureUs, now);
    } else if (state_ == TrackState::Track && now - lastSeenUs_ > kLostTimeoutUs) {
        ESP_LOGI(TAG, "Face lost - scanning");
        enterScan();
    } else if (state_ == TrackState::Track) {
        // No face this frame: follow the prediction for a moment, then hold until the timeout
        coast(panAtCapture, tiltAtCapture, now);
    }

    if (state_ == TrackState::Scan) {
        scanStep(now);
    }

    if (observer_) {
        observer_->onTrackFrame(TrackSample{
            .frameId = frameId_,
            .captureUs = captureUs,
            .state = state_,
            .pan = panAtCapture,
            .tilt = tiltAtCapture,
            .panTarget = panTarget_,
            .tiltTarget = tiltTarget_,
            .faces = faces,
            .targetIndex = target,
            .timing = timing,
            .image = image,
            .filterOn = options_.filter,
            .filterReady = panFilter_.ready(),
            .coasting = coasting_,
            .facePan = panFilter_.angle(),
            .faceTilt = tiltFilter_.angle(),
            .facePanVelocity = panFilter_.velocity(),
            .faceTiltVelocity = tiltFilter_.velocity(),
        });
    }
    ++frameId_;
    logStats(now);
}   // frame buffer goes back to the driver here

void Tracker::pause() {
    if (state_ == TrackState::Paused) return;
    state_ = TrackState::Paused;
    lastFace_.reset();
    panTarget_ = mount_.pan();
    tiltTarget_ = mount_.tilt();
    mount_.setTarget(panTarget_, tiltTarget_);
    ESP_LOGI(TAG, "Paused at pan %.1f tilt %.1f", panTarget_, tiltTarget_);
}

void Tracker::resume() {
    if (state_ != TrackState::Paused) return;
    ESP_LOGI(TAG, "Resumed - scanning");
    enterScan();
    lastStatsUs_ = esp_timer_get_time();
    frames_ = 0;
}

void Tracker::logStats(int64_t now) {
    ++frames_;
    if (now - lastStatsUs_ >= 5000 * 1000) {
        ESP_LOGI(TAG, "%s, %.1f fps", state_ == TrackState::Scan ? "scanning" : "tracking",
                 frames_ * 1e6f / (now - lastStatsUs_));
        frames_ = 0;
        lastStatsUs_ = now;
    }
}

void Tracker::enterScan() {
    state_ = TrackState::Scan;
    lastFace_.reset();
    panFilter_.reset();
    tiltFilter_.reset();
    coasting_ = false;
    mount_.setMaxSpeed(kScanMoveSpeedDegS);
    // Carry on in whichever direction has more room, looking first where the mount already is
    scanDir_ = mount_.pan() < mount_.panRange().center() ? 1 : -1;
    panTarget_ = mount_.pan();
    tiltTarget_ = scanRowTilt();
    scanArrivedUs_ = 0;
    mount_.setTarget(panTarget_, tiltTarget_);
}

void Tracker::enterTrack() {
    state_ = TrackState::Track;
    mount_.setMaxSpeed(kTrackSpeedDegS);
}

void Tracker::scanStep(int64_t now) {
    bool arrived = std::fabs(mount_.pan() - panTarget_) < 0.5f && std::fabs(mount_.tilt() - tiltTarget_) < 0.5f;
    if (!arrived) {
        scanArrivedUs_ = 0;
        return;
    }
    if (scanArrivedUs_ == 0) scanArrivedUs_ = now;
    if (now - scanArrivedUs_ < kScanDwellUs) return;   // still looking at this stop

    const Range &pan = mount_.panRange();
    float end = scanDir_ > 0 ? pan.max : pan.min;
    if (std::fabs(panTarget_ - end) < 0.5f) {
        // Looked at the last stop of this sweep: turn around and move to the next row
        scanDir_ = -scanDir_;
        scanRow_ = (scanRow_ + 1) % kScanRowOffsets.size();
    } else {
        panTarget_ = pan.clamp(panTarget_ + scanDir_ * kScanStepDeg);
    }
    tiltTarget_ = scanRowTilt();
    scanArrivedUs_ = 0;
    mount_.setTarget(panTarget_, tiltTarget_);
}

float Tracker::scanRowTilt() const {
    const Range &tilt = mount_.tiltRange();
    return tilt.clamp(mount_.tiltLevel() + kScanRowOffsets[scanRow_]);
}

void Tracker::steerToward(const Face &face, int frameW, int frameH, float panAtCapture, float tiltAtCapture,
                          int64_t captureUs, int64_t now) {
    // Offset of the face from the frame center, -1..1 on each axis
    float rawX = (face.cx - frameW / 2.0f) / (frameW / 2.0f);
    float rawY = (face.cy - frameH / 2.0f) / (frameH / 2.0f);
    float errX = applyDeadband(rawX);
    float errY = applyDeadband(rawY);

    float pan;
    float tilt;
    if (options_.filter) {
        // Direction of the face in mount angles, from where the mount was when the frame was taken
        float facePan = panAtCapture + kPanDir * rawX * (kHFovDeg / 2.0f);
        float faceTilt = tiltAtCapture + kTiltDir * rawY * (kVFovDeg / 2.0f);
        filterFace(facePan, faceTilt, captureUs);

        // Aim where the face will be once the servos respond
        float ahead = (now - captureUs) / 1e6f + options_.leadS;
        pan = aim(panFilter_, panAtCapture, ahead, kHFovDeg / 2.0f);
        tilt = aim(tiltFilter_, tiltAtCapture, ahead, kVFovDeg / 2.0f);
    } else {
        pan = panAtCapture + kPanDir * kGain * errX * (kHFovDeg / 2.0f);
        tilt = tiltAtCapture + kTiltDir * kGain * errY * (kVFovDeg / 2.0f);
    }
    setMountTarget(pan, tilt);

    ESP_LOGD(TAG, "face (%.0f,%.0f) err (%.2f,%.2f) -> pan %.1f tilt %.1f",
             face.cx, face.cy, errX, errY, pan, tilt);
}

void Tracker::setMountTarget(float pan, float tilt) {
    // Record what the mount will actually aim for
    panTarget_ = mount_.panRange().clamp(pan);
    tiltTarget_ = mount_.tiltRange().clamp(tilt);
    mount_.setTarget(panTarget_, tiltTarget_);
}

float Tracker::aim(const AxisFilter &filter, float atCapture, float ahead, float halfFovDeg) const {
    // Face already inside the deadband: stay put
    if (std::fabs(filter.angle() - atCapture) < kDeadband * halfFovDeg) return atCapture;
    // Lead by the estimated velocity, but no faster than the mount may move, so a noisy
    // velocity estimate can't fling it
    float velocity = std::clamp(filter.velocity(), -kTrackSpeedDegS, kTrackSpeedDegS);
    return filter.angle() + velocity * ahead;
}

void Tracker::coast(float panAtCapture, float tiltAtCapture, int64_t now) {
    coasting_ = options_.filter && panFilter_.ready() &&
                now - lastSeenUs_ <= static_cast<int64_t>(options_.coastS * 1e6f);
    if (!coasting_) return;
    float ahead = (now - filterUs_) / 1e6f + options_.leadS;
    setMountTarget(aim(panFilter_, panAtCapture, ahead, kHFovDeg / 2.0f),
                   aim(tiltFilter_, tiltAtCapture, ahead, kVFovDeg / 2.0f));
}

void Tracker::filterFace(float facePan, float faceTilt, int64_t captureUs) {
    // First sighting, or the estimate went stale over a long miss: start again from here
    if (!panFilter_.ready() || captureUs - filterUs_ > kFilterStaleUs) {
        panFilter_.init(facePan);
        tiltFilter_.init(faceTilt);
        filterUs_ = captureUs;
        rejects_ = 0;
        return;
    }
    float dt = (captureUs - filterUs_) / 1e6f;
    filterUs_ = captureUs;
    panFilter_.predict(dt);
    tiltFilter_.predict(dt);
    bool panOk = panFilter_.update(facePan);
    bool tiltOk = tiltFilter_.update(faceTilt);
    if (panOk && tiltOk) {
        rejects_ = 0;
    } else if (++rejects_ >= kMaxRejects) {
        // Consistently somewhere else: the face really jumped, so start again from here
        panFilter_.init(facePan);
        tiltFilter_.init(faceTilt);
        rejects_ = 0;
    }
}

int Tracker::pickFace(std::span<const Face> faces) const {
    if (faces.empty()) return -1;

    size_t best = 0;
    for (size_t i = 1; i < faces.size(); ++i) {
        const Face &f = faces[i];
        if (lastFace_) {
            // Already following someone: stick with whoever is closest to them
            auto dist = [&](const Face &a) {
                float dx = a.cx - lastFace_->cx, dy = a.cy - lastFace_->cy;
                return dx * dx + dy * dy;
            };
            if (dist(f) < dist(faces[best])) best = i;
        } else if (f.width * f.height > faces[best].width * faces[best].height) {
            // New target: take the biggest (closest) face
            best = i;
        }
    }
    return static_cast<int>(best);
}
