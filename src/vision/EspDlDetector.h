#pragma once

#include <memory>
#include <vector>
#include "vision/IDetector.h"

class HumanFaceDetect;

/**
 * Espressif's human_face_detect models on the ESP-DL runtime (vendor int8 PTQ, power-of-two
 * per-tensor scales). The model files are linked into the app (flash rodata), so only models
 * enabled with CONFIG_FLASH_* in sdkconfig can be created.
 */
class EspDlDetector final : public IDetector {

public:
    enum class Model : uint8_t {
        MsrMnp,          // two-stage MSR (160x120) + MNP (48x48 crops) - the baseline
        EspDetPico224,
        EspDetPico416,
    };

    explicit EspDlDetector(Model model);
    ~EspDlDetector() override;

    [[nodiscard]] const ModelInfo &info() const override { return info_; }
    void setScoreThreshold(float threshold) override;
    [[nodiscard]] std::span<const Face> detect(const ImageView &image, DetectTiming &timing) override;

private:
    std::unique_ptr<HumanFaceDetect> model_;
    ModelInfo info_;
    int stages_;              // 2 for MSR+MNP, 1 for the single-stage models
    std::vector<Face> faces_; // reused every frame
};
