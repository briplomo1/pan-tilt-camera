#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include "freertos/FreeRTOS.h"

/**
 * Byte pipe to the host. The link protocol runs on top of any implementation (USB today; UART
 * or WiFi could be added without touching the protocol code).
 */
class ITransport {

public:
    virtual ~ITransport() = default;
    ITransport(const ITransport &) = delete;
    ITransport &operator=(const ITransport &) = delete;
    ITransport(ITransport &&) = delete;
    ITransport &operator=(ITransport &&) = delete;

    // Read up to buffer.size() bytes, waiting at most `timeout` for data. Returns bytes read.
    [[nodiscard]] virtual size_t read(std::span<uint8_t> buffer, TickType_t timeout) = 0;

    // Write all of `data`; false if it couldn't be queued within `timeout`
    [[nodiscard]] virtual bool write(std::span<const uint8_t> data, TickType_t timeout) = 0;

    // False when no host is listening, so senders can skip work instead of blocking
    [[nodiscard]] virtual bool connected() const = 0;

protected:
    ITransport() = default;
};
