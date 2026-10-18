#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include "esp_heap_caps.h"

// Memory regions a buffer can come from
inline constexpr uint32_t kPsramCaps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT;
inline constexpr uint32_t kInternalCaps = MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT;

/**
 * Block of heap memory from a chosen region (PSRAM or internal RAM). Freed when it goes out
 * of scope; moving hands the memory over and leaves the source empty.
 */
class HeapBuffer {

private:
    uint8_t *data_ = nullptr;
    size_t size_ = 0;

    HeapBuffer(uint8_t *data, size_t size) : data_(data), size_(size) {}

public:
    HeapBuffer() = default;
    ~HeapBuffer() { heap_caps_free(data_); }

    HeapBuffer(const HeapBuffer &) = delete;
    HeapBuffer &operator=(const HeapBuffer &) = delete;
    HeapBuffer(HeapBuffer &&other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}
    HeapBuffer &operator=(HeapBuffer &&other) noexcept {
        if (this != &other) {
            heap_caps_free(data_);
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }

    // Empty buffer if the region doesn't have a free block that big
    [[nodiscard]] static HeapBuffer allocate(size_t size, uint32_t caps, size_t alignment = 16) {
        auto *data = static_cast<uint8_t *>(heap_caps_aligned_alloc(alignment, size, caps));
        return data ? HeapBuffer(data, size) : HeapBuffer();
    }

    [[nodiscard]] uint8_t *data() { return data_; }
    [[nodiscard]] const uint8_t *data() const { return data_; }
    [[nodiscard]] size_t size() const { return size_; }
    [[nodiscard]] std::span<uint8_t> span() { return {data_, size_}; }
    [[nodiscard]] std::span<const uint8_t> span() const { return {data_, size_}; }
    explicit operator bool() const { return data_ != nullptr; }
};
