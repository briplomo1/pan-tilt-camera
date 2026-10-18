#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string_view>
#include "link/Protocol.h"
#include "vision/IDetector.h"

// Conversions between firmware types and their wire form (see link/Protocol.h)

[[nodiscard]] inline proto::WireFace toWire(const Face &face) {
    float halfW = face.width / 2.0f;
    float halfH = face.height / 2.0f;
    return proto::WireFace{
        .x0 = static_cast<int16_t>(std::lround(face.cx - halfW)),
        .y0 = static_cast<int16_t>(std::lround(face.cy - halfH)),
        .x1 = static_cast<int16_t>(std::lround(face.cx + halfW)),
        .y1 = static_cast<int16_t>(std::lround(face.cy + halfH)),
        .score = face.score,
    };
}

[[nodiscard]] inline proto::WireTiming toWire(const DetectTiming &timing) {
    return proto::WireTiming{timing.preprocessUs, timing.inferenceUs, timing.postprocessUs, timing.totalUs};
}

// Copy into a fixed-size, NUL-terminated wire string, truncating if needed
template <size_t N>
void copyString(char (&out)[N], std::string_view text) {
    size_t n = std::min(text.size(), N - 1);
    std::memcpy(out, text.data(), n);
    std::memset(out + n, 0, N - n);
}

// Read a fixed-size wire string that may not be NUL-terminated
template <size_t N>
[[nodiscard]] std::string_view readString(const char (&in)[N]) {
    return std::string_view(in, strnlen(in, N));
}
