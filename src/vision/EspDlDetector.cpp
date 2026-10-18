#include "vision/EspDlDetector.h"
#include "human_face_detect.hpp"
#include "esp_log.h"
#include "esp_timer.h"

namespace {
constexpr const char *TAG = "espdl";

// More faces than this in one frame are ignored, so detect() never reallocates
constexpr size_t kMaxFaces = 32;

ModelInfo describe(EspDlDetector::Model model) {
    switch (model) {
    case EspDlDetector::Model::EspDetPico224:
        return {"espdl-espdet-pico-224-s8", "esp-dl", "w8a8 pot per-tensor (vendor ptq)", 224, 224, 0};
    case EspDlDetector::Model::EspDetPico416:
        return {"espdl-espdet-pico-416-s8", "esp-dl", "w8a8 pot per-tensor (vendor ptq)", 416, 416, 0};
    case EspDlDetector::Model::MsrMnp:
    default:
        return {"espdl-msrmnp-s8", "esp-dl", "w8a8 pot per-tensor (vendor ptq)", 160, 120, 0};
    }
}

HumanFaceDetect::model_type_t toModelType(EspDlDetector::Model model) {
    switch (model) {
    case EspDlDetector::Model::EspDetPico224: return HumanFaceDetect::ESPDET_PICO_224_224_FACE;
    case EspDlDetector::Model::EspDetPico416: return HumanFaceDetect::ESPDET_PICO_416_416_FACE;
    case EspDlDetector::Model::MsrMnp:
    default: return HumanFaceDetect::MSRMNP_S8_V1;
    }
}

bool toPixType(PixelFormat format, dl::image::pix_type_t &out) {
    switch (format) {
    case PixelFormat::Rgb565Be: out = dl::image::DL_IMAGE_PIX_TYPE_RGB565BE; return true;
    case PixelFormat::Rgb888: out = dl::image::DL_IMAGE_PIX_TYPE_RGB888; return true;
    case PixelFormat::Gray8: out = dl::image::DL_IMAGE_PIX_TYPE_GRAY; return true;
    }
    return false;
}
} // namespace

// lazy_load = false: load the model now instead of stalling the first frame
EspDlDetector::EspDlDetector(Model model)
    : model_(std::make_unique<HumanFaceDetect>(toModelType(model), false)),
      info_(describe(model)),
      stages_(model == Model::MsrMnp ? 2 : 1) {
    faces_.reserve(kMaxFaces);

    // Size of the weights the model reads from flash, summed over its stages
    for (int i = 0; i < stages_; ++i) {
        dl::Model *raw = model_->get_raw_model(i);
        if (raw && raw->get_fbs_model()) {
            size_t internal = 0, psram = 0, psramRodata = 0, flash = 0;
            raw->get_fbs_model()->get_model_size(&internal, &psram, &psramRodata, &flash);
            info_.modelBytes += flash + psramRodata;
        }
    }
    ESP_LOGI(TAG, "Loaded %s (%zu bytes of weights)", info_.id.c_str(), info_.modelBytes);
}

EspDlDetector::~EspDlDetector() = default;

void EspDlDetector::setScoreThreshold(float threshold) {
    // Only the final stage: lowering MSR's proposal threshold would make MNP run on far more
    // candidates and change the model's speed, not just what it reports
    model_->set_score_thr(threshold, stages_ - 1);
}

std::span<const Face> EspDlDetector::detect(const ImageView &image, DetectTiming &timing) {
    faces_.clear();
    timing = {};

    dl::image::pix_type_t pixType;
    if (!image || !toPixType(image.format, pixType)) {
        ESP_LOGE(TAG, "Unsupported image %ux%u format %d", image.width, image.height,
                 static_cast<int>(image.format));
        return {};
    }

    dl::image::img_t img = {};
    img.data = const_cast<uint8_t *>(image.data);   // ESP-DL only reads it
    img.width = image.width;
    img.height = image.height;
    img.pix_type = pixType;

    // ESP-DL resizes, runs and decodes in one call, so only the total time is known
    int64_t start = esp_timer_get_time();
    auto &results = model_->run(img);
    timing.totalUs = static_cast<uint32_t>(esp_timer_get_time() - start);

    for (auto &r : results) {
        if (faces_.size() == kMaxFaces) break;
        r.limit_box(img.width, img.height);
        faces_.push_back(Face{
            .cx = (r.box[0] + r.box[2]) / 2.0f,
            .cy = (r.box[1] + r.box[3]) / 2.0f,
            .width = r.box[2] - r.box[0],
            .height = r.box[3] - r.box[1],
            .score = r.score,
        });
    }
    return faces_;
}
