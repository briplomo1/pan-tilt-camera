#pragma once

#include <cstddef>
#include <cstdint>

// Values match the image format byte of the USB protocol (proto::ImageFormat)
enum class PixelFormat : uint8_t {
    Rgb565Be = 0,   // what the camera produces
    Rgb888 = 1,
    Gray8 = 2,
};

[[nodiscard]] constexpr size_t bytesPerPixel(PixelFormat format) {
    switch (format) {
    case PixelFormat::Rgb565Be: return 2;
    case PixelFormat::Rgb888: return 3;
    case PixelFormat::Gray8: return 1;
    }
    return 0;
}

// Non-owning view of an image somewhere in memory (camera buffer, USB packet, ...)
struct ImageView {
    const uint8_t *data = nullptr;
    uint16_t width = 0;
    uint16_t height = 0;
    PixelFormat format = PixelFormat::Rgb565Be;

    [[nodiscard]] size_t bytes() const { return size_t{width} * height * bytesPerPixel(format); }
    explicit operator bool() const { return data != nullptr && width > 0 && height > 0; }
};
