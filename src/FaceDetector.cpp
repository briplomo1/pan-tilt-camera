#include "FaceDetector.h"
#include "human_face_detect.hpp"
#include "esp_log.h"

namespace {
constexpr const char *TAG = "faces";
} // namespace

// lazy_load = false: load the model now instead of stalling the first frame
FaceDetector::FaceDetector() : model_(std::make_unique<HumanFaceDetect>(
    static_cast<HumanFaceDetect::model_type_t>(CONFIG_DEFAULT_HUMAN_FACE_DETECT_MODEL), false)) {
    ESP_LOGI(TAG, "Face detection model loaded");
}

FaceDetector::~FaceDetector() = default;

std::vector<Face> FaceDetector::detect(const camera_fb_t &frame) {
    std::vector<Face> faces;
    if (frame.format != PIXFORMAT_RGB565) {
        ESP_LOGE(TAG, "Expected an RGB565 frame, got format %d", frame.format);
        return faces;
    }

    dl::image::img_t img = {};
    img.data = frame.buf;
    img.width = static_cast<uint16_t>(frame.width);
    img.height = static_cast<uint16_t>(frame.height);
    img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565BE;   // esp32-camera outputs big-endian RGB565

    auto &results = model_->run(img);
    for (auto &r : results) {
        r.limit_box(img.width, img.height);
        faces.push_back(Face{
            .cx = (r.box[0] + r.box[2]) / 2.0f,
            .cy = (r.box[1] + r.box[3]) / 2.0f,
            .width = r.box[2] - r.box[0],
            .height = r.box[3] - r.box[1],
            .score = r.score,
        });
    }
    return faces;
}
