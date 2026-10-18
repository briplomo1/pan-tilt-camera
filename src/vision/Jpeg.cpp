#include "vision/Jpeg.h"
#include <cstring>
#include "img_converters.h"

namespace {
struct Output {
    std::span<uint8_t> buffer;
    size_t length = 0;
    bool overflow = false;
};

// Called by the encoder with consecutive chunks; returning 0 aborts it
size_t writeChunk(void *arg, size_t index, const void *data, size_t len) {
    auto *out = static_cast<Output *>(arg);
    if (index > out->buffer.size() || len > out->buffer.size() - index) {
        out->overflow = true;
        return 0;
    }
    std::memcpy(out->buffer.data() + index, data, len);
    out->length = index + len;
    return len;
}
} // namespace

size_t encodeJpeg(const ImageView &image, uint8_t quality, std::span<uint8_t> out) {
    pixformat_t format;
    switch (image.format) {
    case PixelFormat::Rgb565Be: format = PIXFORMAT_RGB565; break;
    case PixelFormat::Gray8: format = PIXFORMAT_GRAYSCALE; break;
    case PixelFormat::Rgb888: format = PIXFORMAT_RGB888; break;
    default: return 0;
    }

    Output output{out};
    bool ok = fmt2jpg_cb(const_cast<uint8_t *>(image.data), image.bytes(), image.width, image.height,
                         format, quality, &writeChunk, &output);
    return ok && !output.overflow ? output.length : 0;
}
