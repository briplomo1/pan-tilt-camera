#include "link/UsbSerialTransport.h"
#include <algorithm>
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#include "esp_err.h"

namespace {
constexpr uint32_t kRxBufferSize = 8192;
constexpr uint32_t kTxBufferSize = 4096;

// The driver rejects a single write bigger than its ring buffer, so large writes are split
constexpr size_t kMaxWriteChunk = kTxBufferSize / 2;
} // namespace

UsbSerialTransport::UsbSerialTransport() {
    usb_serial_jtag_driver_config_t config = {};
    config.rx_buffer_size = kRxBufferSize;
    config.tx_buffer_size = kTxBufferSize;
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&config));
    usb_serial_jtag_vfs_use_driver();
}

UsbSerialTransport::~UsbSerialTransport() {
    usb_serial_jtag_vfs_use_nonblocking();
    usb_serial_jtag_driver_uninstall();
}

size_t UsbSerialTransport::read(std::span<uint8_t> buffer, TickType_t timeout) {
    int n = usb_serial_jtag_read_bytes(buffer.data(), buffer.size(), timeout);
    return n > 0 ? static_cast<size_t>(n) : 0;
}

bool UsbSerialTransport::write(std::span<const uint8_t> data, TickType_t timeout) {
    while (!data.empty()) {
        size_t chunk = std::min(data.size(), kMaxWriteChunk);
        int written = usb_serial_jtag_write_bytes(data.data(), chunk, timeout);
        if (written <= 0) return false;
        data = data.subspan(static_cast<size_t>(written));
    }
    return true;
}

bool UsbSerialTransport::connected() const {
    return usb_serial_jtag_is_connected();
}
