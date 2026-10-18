#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include "vision/ImageView.h"

/**
 * JPEG-encode an RGB565 or gray image into `out` (no allocation). Returns the encoded size,
 * 0 if encoding failed or the result didn't fit.
 */
[[nodiscard]] size_t encodeJpeg(const ImageView &image, uint8_t quality, std::span<uint8_t> out);
