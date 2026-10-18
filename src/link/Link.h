#pragma once

#include <array>
#include <cstdarg>
#include <cstring>
#include <initializer_list>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/ringbuf.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "link/ITransport.h"
#include "link/Protocol.h"
#include "memory/HeapBuffer.h"

class Link;

/**
 * A packet from the host. Holds one of the link's receive buffers until it is destroyed, the
 * same way a camera Frame holds a frame buffer; move-only.
 */
class RxPacket {

private:
    Link *link_ = nullptr;
    int slot_ = -1;
    proto::Header header_{};
    std::span<const uint8_t> payload_;

    friend class Link;
    RxPacket(Link *link, int slot, const proto::Header &header, std::span<const uint8_t> payload)
        : link_(link), slot_(slot), header_(header), payload_(payload) {}
    void release();

public:
    RxPacket() = default;
    ~RxPacket() { release(); }
    RxPacket(const RxPacket &) = delete;
    RxPacket &operator=(const RxPacket &) = delete;
    RxPacket(RxPacket &&other) noexcept
        : link_(std::exchange(other.link_, nullptr)), slot_(std::exchange(other.slot_, -1)),
          header_(other.header_), payload_(std::exchange(other.payload_, {})) {}
    RxPacket &operator=(RxPacket &&other) noexcept {
        if (this != &other) {
            release();
            link_ = std::exchange(other.link_, nullptr);
            slot_ = std::exchange(other.slot_, -1);
            header_ = other.header_;
            payload_ = std::exchange(other.payload_, {});
        }
        return *this;
    }

    [[nodiscard]] proto::MsgType type() const { return header_.type; }
    [[nodiscard]] std::span<const uint8_t> payload() const { return payload_; }

    // The fixed-size struct at the start of the payload; empty if the payload is too short
    template <typename T>
    [[nodiscard]] std::optional<T> as() const {
        static_assert(std::is_trivially_copyable_v<T>);
        if (payload_.size() < sizeof(T)) return std::nullopt;
        T value;
        std::memcpy(&value, payload_.data(), sizeof(T));
        return value;
    }
};

/**
 * Packet link to the host tools over any ITransport (see Protocol.h for the format).
 *
 * A receive task on core 0 decodes incoming packets into a small pool of PSRAM buffers and
 * queues them; the vision task takes them with receive(). Any task can send(); sends are
 * serialized so packets never interleave on the wire.
 */
class Link {

private:
    static constexpr int kSlots = 2;
    static constexpr size_t kSlotSize = sizeof(proto::Header) + proto::kMaxPayload + sizeof(uint32_t);

    struct Ready {
        int slot;
        size_t size;
    };

    ITransport &transport_;
    std::array<HeapBuffer, kSlots> slots_;
    QueueHandle_t free_ = nullptr;      // slot indices the receive task may fill
    QueueHandle_t ready_ = nullptr;     // whole packets waiting for receive()
    SemaphoreHandle_t txLock_ = nullptr;
    TaskHandle_t rxTask_ = nullptr;
    RingbufHandle_t logRing_ = nullptr; // formatted log lines waiting for the log task
    TaskHandle_t logTask_ = nullptr;
    uint16_t txSeq_ = 0;
    TickType_t txTimeout_ = 0;
    uint32_t rxErrors_ = 0;

    static inline Link *logTarget_ = nullptr;
    static inline vprintf_like_t previousLog_ = nullptr;

    friend class RxPacket;
    void release(int slot);
    [[nodiscard]] bool valid(int slot, size_t size) const;
    static bool sink(void *context, std::span<const uint8_t> bytes);
    static void rxEntry(void *arg);
    [[noreturn]] void rxLoop();
    static void logEntry(void *arg);
    [[noreturn]] void logLoop();
    static int logVprintf(const char *format, va_list args);

public:
    static constexpr TickType_t kSendTimeout = pdMS_TO_TICKS(500);

    explicit Link(ITransport &transport);
    ~Link();
    Link(const Link &) = delete;
    Link &operator=(const Link &) = delete;
    Link(Link &&) = delete;
    Link &operator=(Link &&) = delete;

    // Next packet from the host, or empty after `timeout`
    [[nodiscard]] std::optional<RxPacket> receive(TickType_t timeout);

    // Send one packet built from several pieces (struct, face array, pixels) without copying
    // them. False if no host is connected or the write timed out.
    bool send(proto::MsgType type, std::initializer_list<std::span<const uint8_t>> parts,
              TickType_t timeout = kSendTimeout);

    bool sendAck(proto::MsgType forType, bool ok);
    bool sendError(proto::MsgType forType, proto::ErrorCode code, std::string_view message);

    // From now on ESP_LOG output travels as Log packets, so it can't corrupt binary packets.
    // Logging only queues the line; a low-priority task sends it, so a log call never blocks
    // and costs the calling task little stack.
    void captureLogs();

    [[nodiscard]] bool connected() const { return transport_.connected(); }
    [[nodiscard]] uint32_t rxErrors() const { return rxErrors_; }
};
