#include "app/PreviewStreamer.h"
#include <algorithm>
#include <cstring>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "app/Wire.h"
#include "vision/Jpeg.h"

namespace {
constexpr const char *TAG = "preview";

constexpr int kTaskPriority = 1;        // lowest: never delays servo motion or USB receive
constexpr int kTaskStack = 8192;        // the JPEG encoder needs a few KB
constexpr uint16_t kMaxWidth = 320;
constexpr uint16_t kMaxHeight = 240;
constexpr size_t kPixelBytes = size_t{kMaxWidth} * kMaxHeight * 2;
constexpr size_t kJpegBytes = 64 * 1024;
constexpr TickType_t kSendTimeout = pdMS_TO_TICKS(200);

// Copy an RGB565 image keeping every (1 << shift)-th pixel, optionally converting to gray
void copyScaled(const ImageView &src, uint8_t shift, bool gray, uint8_t *out) {
    const size_t step = size_t{1} << shift;
    const uint16_t width = src.width >> shift;
    const uint16_t height = src.height >> shift;

    if (shift == 0 && !gray) {
        std::memcpy(out, src.data, src.bytes());
        return;
    }
    for (uint16_t y = 0; y < height; ++y) {
        const uint8_t *row = src.data + y * step * src.width * 2;
        for (uint16_t x = 0; x < width; ++x) {
            const uint8_t *p = row + x * step * 2;
            if (gray) {
                // RGB565 big-endian: RRRRRGGG GGGBBBBB
                int r = p[0] & 0xF8;
                int g = ((p[0] & 0x07) << 5) | ((p[1] & 0xE0) >> 3);
                int b = (p[1] & 0x1F) << 3;
                *out++ = static_cast<uint8_t>((77 * r + 150 * g + 29 * b) >> 8);
            } else {
                *out++ = p[0];
                *out++ = p[1];
            }
        }
    }
}
} // namespace

PreviewStreamer::PreviewStreamer(Link &link) : link_(link) {
    xTaskCreatePinnedToCore(taskEntry, "preview", kTaskStack, this, kTaskPriority, &task_, 0);
}

PreviewStreamer::~PreviewStreamer() {
    configure(PreviewSettings{});
    if (task_) vTaskDelete(task_);
}

void PreviewStreamer::waitIdle() const {
    while (busy_.load(std::memory_order_acquire)) vTaskDelay(pdMS_TO_TICKS(5));
}

bool PreviewStreamer::configure(const PreviewSettings &settings) {
    // The task may still be sending from the buffers we're about to change
    waitIdle();

    if (!settings.enabled) {
        settings_ = settings;
        pixels_ = HeapBuffer();
        encoded_ = HeapBuffer();
        ESP_LOGI(TAG, "Preview off (%lu frames dropped while busy)", static_cast<unsigned long>(dropped_));
        return true;
    }

    bool validFormat = settings.format == proto::ImageFormat::Jpeg || settings.format == proto::ImageFormat::Rgb565Be ||
                       settings.format == proto::ImageFormat::Gray8;
    if (!validFormat || settings.scaleShift > 2 || settings.everyN == 0 || settings.jpegQuality == 0 ||
        settings.jpegQuality > 100) {
        return false;
    }

    if (!pixels_) pixels_ = HeapBuffer::allocate(kPixelBytes, kPsramCaps);
    if (settings.format == proto::ImageFormat::Jpeg && !encoded_) encoded_ = HeapBuffer::allocate(kJpegBytes, kPsramCaps);
    if (!pixels_ || (settings.format == proto::ImageFormat::Jpeg && !encoded_)) {
        pixels_ = HeapBuffer();
        encoded_ = HeapBuffer();
        settings_.enabled = false;
        ESP_LOGE(TAG, "Not enough PSRAM for the preview");
        return false;
    }

    settings_ = settings;
    offered_ = 0;
    dropped_ = 0;
    ESP_LOGI(TAG, "Preview on: format %u, 1/%u scale, every %u frame(s), quality %u",
             static_cast<unsigned>(settings.format), 1u << settings.scaleShift, settings.everyN, settings.jpegQuality);
    return true;
}

void PreviewStreamer::offer(const PreviewFrame &frame) {
    if (!settings_.enabled) return;
    if (offered_++ % settings_.everyN != 0) return;
    if (busy_.load(std::memory_order_acquire)) {
        ++dropped_;
        return;
    }

    const ImageView &src = frame.image;
    if (!src || src.format != PixelFormat::Rgb565Be || src.width > kMaxWidth || src.height > kMaxHeight) return;

    const bool gray = settings_.format == proto::ImageFormat::Gray8;
    const uint16_t width = src.width >> settings_.scaleShift;
    const uint16_t height = src.height >> settings_.scaleShift;
    if (width == 0 || height == 0) return;

    copyScaled(src, settings_.scaleShift, gray, pixels_.data());
    pixelLength_ = size_t{width} * height * (gray ? 1 : 2);

    const size_t faceCount = std::min(frame.faces.size(), kMaxFaces);
    for (size_t i = 0; i < faceCount; ++i) faces_[i] = toWire(frame.faces[i]);

    meta_ = {};
    meta_.timeUs = frame.timeUs;
    meta_.frameId = frame.frameId;
    meta_.trackerState = frame.trackerState;
    meta_.faceCount = static_cast<uint8_t>(faceCount);
    meta_.targetIndex = frame.targetIndex >= 0 && static_cast<size_t>(frame.targetIndex) < faceCount
                            ? static_cast<uint8_t>(frame.targetIndex) : 0xFF;
    meta_.mode = frame.mode;
    meta_.pan = frame.pan;
    meta_.tilt = frame.tilt;
    meta_.timing = toWire(frame.timing);
    meta_.freeInternal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    meta_.sourceWidth = src.width;
    meta_.sourceHeight = src.height;
    copyString(meta_.modelId, frame.modelId);
    meta_.image = proto::ImageHeader{frame.frameId, width, height, settings_.format, {}};

    busy_.store(true, std::memory_order_release);
    xTaskNotifyGive(task_);
}

void PreviewStreamer::taskEntry(void *arg) {
    static_cast<PreviewStreamer *>(arg)->run();
}

void PreviewStreamer::run() {
    while (true) {
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
        sendPending();
        busy_.store(false, std::memory_order_release);
    }
}

void PreviewStreamer::sendPending() {
    std::span<const uint8_t> image(pixels_.data(), pixelLength_);

    if (settings_.format == proto::ImageFormat::Jpeg) {
        ImageView raw{pixels_.data(), meta_.image.width, meta_.image.height, PixelFormat::Rgb565Be};
        size_t length = encodeJpeg(raw, settings_.jpegQuality, encoded_.span());
        if (length == 0) {
            ESP_LOGW(TAG, "JPEG encode failed");
            return;
        }
        image = {encoded_.data(), length};
    }

    meta_.imageBytes = static_cast<uint32_t>(image.size());
    std::span<const proto::WireFace> faces(faces_.data(), meta_.faceCount);
    link_.send(proto::MsgType::Preview, {proto::bytesOf(meta_), proto::bytesOf(faces), image}, kSendTimeout);
}
