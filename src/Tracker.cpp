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

// Camera field of view (OV2640 stock lens at 4:3). Used to turn a pixel offset
// into degrees; only needs to be roughly right.
constexpr float kHFovDeg = 54.0f;
constexpr float kVFovDeg = 41.0f;

// Flip to -1 if the mount moves away from the face instead of toward it.
// +1 pan: face right of center -> pan angle increases.
// +1 tilt: face below center -> tilt angle increases.
constexpr float kPanDir = 1.0f;
constexpr float kTiltDir = 1.0f;

// Fraction of the measured error corrected per frame. <1 avoids overshoot,
// since each frame is already ~50-80 ms old by the time it's processed.
constexpr float kGain = 0.6f;

// Ignore errors smaller than this fraction of half the frame (stops jitter)
constexpr float kDeadband = 0.06f;

constexpr float kScanSpeedDegS = 30.0f;    // slow enough that frames aren't blurred
constexpr float kTrackSpeedDegS = 240.0f;

// Tilt rows the scan visits, as offsets from the tilt range center
constexpr std::array<float, 3> kScanRowOffsets = {0.0f, -20.0f, 20.0f};

constexpr int64_t kLostTimeoutUs = 1500 * 1000;

// ----------------------------------------------------------------------------

float applyDeadband(float err) {
    return std::fabs(err) < kDeadband ? 0.0f : err;
}
} // namespace

void Tracker::run() {
    enterScan();
    int64_t lastStatsUs = esp_timer_get_time();
    int frames = 0;

    while (true) {
        float panAtCapture, tiltAtCapture;
        int frameW, frameH;
        std::vector<Face> faces;
        {
            Frame frame = Camera::capture();
            if (!frame) {
                ESP_LOGW(TAG, "Camera capture failed");
                delay(100ms);
                continue;
            }
            // Where the mount was pointing when this frame was taken
            panAtCapture = mount_.pan();
            tiltAtCapture = mount_.tilt();
            frameW = frame->width;
            frameH = frame->height;
            faces = detector_.detect(*frame);
        } // frame buffer goes back to the driver here

        int64_t now = esp_timer_get_time();
        std::optional<Face> face = pickFace(faces);

        if (face) {
            if (state_ == State::Scan) {
                ESP_LOGI(TAG, "Face found (score %.2f) at pan %.1f tilt %.1f - tracking",
                         face->score, panAtCapture, tiltAtCapture);
                enterTrack();
            }
            lastSeenUs_ = now;
            lastFace_ = face;
            steerToward(*face, frameW, frameH, panAtCapture, tiltAtCapture);
        } else if (state_ == State::Track && now - lastSeenUs_ > kLostTimeoutUs) {
            ESP_LOGI(TAG, "Face lost - scanning");
            enterScan();
        }
        // While tracking with no face this frame: hold position until the timeout

        if (state_ == State::Scan) {
            scanStep();
        }

        ++frames;
        if (now - lastStatsUs >= 5000 * 1000) {
            ESP_LOGI(TAG, "%s, %.1f fps", state_ == State::Scan ? "scanning" : "tracking",
                     frames * 1e6f / (now - lastStatsUs));
            frames = 0;
            lastStatsUs = now;
        }
    }
}

void Tracker::enterScan() {
    state_ = State::Scan;
    lastFace_.reset();
    mount_.setMaxSpeed(kScanSpeedDegS);
    // Carry on in whichever direction has more room
    scanDir_ = mount_.pan() < mount_.panRange().center() ? 1 : -1;
}

void Tracker::enterTrack() {
    state_ = State::Track;
    mount_.setMaxSpeed(kTrackSpeedDegS);
}

void Tracker::scanStep() {
    const Range &pan = mount_.panRange();
    float end = scanDir_ > 0 ? pan.max : pan.min;

    if (std::fabs(mount_.pan() - end) < 0.5f) {
        // Reached the end of this sweep: turn around and move to the next row
        scanDir_ = -scanDir_;
        scanRow_ = (scanRow_ + 1) % kScanRowOffsets.size();
        end = scanDir_ > 0 ? pan.max : pan.min;
    }
    mount_.setTarget(end, scanRowTilt());
}

float Tracker::scanRowTilt() const {
    const Range &tilt = mount_.tiltRange();
    return tilt.clamp(tilt.center() + kScanRowOffsets[scanRow_]);
}

void Tracker::steerToward(const Face &face, int frameW, int frameH, float panAtCapture, float tiltAtCapture) {
    // Offset of the face from the frame center, -1..1 on each axis
    float errX = applyDeadband((face.cx - frameW / 2.0f) / (frameW / 2.0f));
    float errY = applyDeadband((face.cy - frameH / 2.0f) / (frameH / 2.0f));

    float pan = panAtCapture + kPanDir * kGain * errX * (kHFovDeg / 2.0f);
    float tilt = tiltAtCapture + kTiltDir * kGain * errY * (kVFovDeg / 2.0f);
    mount_.setTarget(pan, tilt);

    ESP_LOGD(TAG, "face (%.0f,%.0f) err (%.2f,%.2f) -> pan %.1f tilt %.1f",
             face.cx, face.cy, errX, errY, pan, tilt);
}

std::optional<Face> Tracker::pickFace(const std::vector<Face> &faces) const {
    if (faces.empty()) return std::nullopt;

    const Face *best = &faces.front();
    for (const Face &f : faces) {
        if (lastFace_) {
            // Already following someone: stick with whoever is closest to them
            auto dist = [&](const Face &a) {
                float dx = a.cx - lastFace_->cx, dy = a.cy - lastFace_->cy;
                return dx * dx + dy * dy;
            };
            if (dist(f) < dist(*best)) best = &f;
        } else if (f.width * f.height > best->width * best->height) {
            // New target: take the biggest (closest) face
            best = &f;
        }
    }
    return *best;
}
