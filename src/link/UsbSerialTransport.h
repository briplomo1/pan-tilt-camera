#pragma once

#include "link/ITransport.h"

/**
 * The XIAO's USB-C port (the ESP32-S3's built-in USB-Serial-JTAG). Installs the driver for its
 * lifetime and routes the console through it, so printf and the link don't fight over the port.
 */
class UsbSerialTransport final : public ITransport {

public:
    UsbSerialTransport();
    ~UsbSerialTransport() override;

    [[nodiscard]] size_t read(std::span<uint8_t> buffer, TickType_t timeout) override;
    [[nodiscard]] bool write(std::span<const uint8_t> data, TickType_t timeout) override;
    [[nodiscard]] bool connected() const override;
};
