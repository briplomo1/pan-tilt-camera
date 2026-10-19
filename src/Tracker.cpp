#include "Tracker.h"
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

// Fraction of the measured error corrected per frame. <1 avoids overshoot,
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
        steerToward(face, image.width, image.height, panAtCapture, tiltAtCapture);
    } else if (state_ == TrackState::Track && now - lastSeenUs_ > kLostTimeoutUs) {
        ESP_LOGI(TAG, "Face lost - scanning");
        enterScan();
    }
    // While tracking with no face this frame: hold position until the timeout

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

void Tracker::steerToward(const Face &face, int frameW, int frameH, float panAtCapture, float tiltAtCapture) {
    // Offset of the face from the frame center, -1..1 on each axis
    float errX = applyDeadband((face.cx - frameW / 2.0f) / (frameW / 2.0f));
    float errY = applyDeadband((face.cy - frameH / 2.0f) / (frameH / 2.0f));

    float pan = panAtCapture + kPanDir * kGain * errX * (kHFovDeg / 2.0f);
    float tilt = tiltAtCapture + kTiltDir * kGain * errY * (kVFovDeg / 2.0f);
    // Record what the mount will actually aim for
    panTarget_ = mount_.panRange().clamp(pan);
    tiltTarget_ = mount_.tiltRange().clamp(tilt);
    mount_.setTarget(panTarget_, tiltTarget_);

    ESP_LOGD(TAG, "face (%.0f,%.0f) err (%.2f,%.2f) -> pan %.1f tilt %.1f",
             face.cx, face.cy, errX, errY, pan, tilt);
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
