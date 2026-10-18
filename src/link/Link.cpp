#include "link/Link.h"
#include <algorithm>
#include <cstdio>
#include "esp_rom_crc.h"
#include "link/Cobs.h"

namespace {
constexpr const char *TAG = "link";

constexpr int kRxTaskPriority = 4;      // below the servo motion task
constexpr int kRxTaskStack = 4096;
constexpr size_t kRxChunk = 512;
constexpr size_t kMaxLogLine = 160;
constexpr size_t kLogRingSize = 4096;
constexpr int kLogTaskPriority = 1;
constexpr int kLogTaskStack = 3072;
constexpr TickType_t kLogTimeout = pdMS_TO_TICKS(50);

uint32_t crc32(uint32_t crc, std::span<const uint8_t> data) {
    return esp_rom_crc32_le(crc, data.data(), data.size());
}
} // namespace

void RxPacket::release() {
    if (link_) {
        link_->release(slot_);
        link_ = nullptr;
    }
}

Link::Link(ITransport &transport) : transport_(transport) {
    // Same check as tools/tests/test_protocol.py: the host computes CRCs with zlib.crc32
    static constexpr char kCheck[] = "123456789";
    if (crc32(0, {reinterpret_cast<const uint8_t *>(kCheck), 9}) != 0xCBF43926u) {
        ESP_LOGE(TAG, "CRC32 self-test failed - the host will reject every packet");
    }

    free_ = xQueueCreate(kSlots, sizeof(int));
    ready_ = xQueueCreate(kSlots, sizeof(Ready));
    txLock_ = xSemaphoreCreateMutex();
    if (!free_ || !ready_ || !txLock_) ESP_ERROR_CHECK(ESP_ERR_NO_MEM);

    for (int i = 0; i < kSlots; ++i) {
        slots_[i] = HeapBuffer::allocate(kSlotSize, kPsramCaps);
        if (!slots_[i]) ESP_ERROR_CHECK(ESP_ERR_NO_MEM);
        xQueueSend(free_, &i, 0);
    }

    xTaskCreatePinnedToCore(rxEntry, "link_rx", kRxTaskStack, this, kRxTaskPriority, &rxTask_, 0);
    ESP_LOGI(TAG, "USB link ready (protocol v%u, %d x %zu byte receive buffers)", proto::kVersion, kSlots, kSlotSize);
}

Link::~Link() {
    if (logTarget_ == this) {
        esp_log_set_vprintf(previousLog_);
        logTarget_ = nullptr;
    }
    if (logTask_) vTaskDelete(logTask_);
    if (logRing_) vRingbufferDelete(logRing_);
    if (rxTask_) vTaskDelete(rxTask_);
    vQueueDelete(ready_);
    vQueueDelete(free_);
    vSemaphoreDelete(txLock_);
}

void Link::release(int slot) {
    xQueueSend(free_, &slot, 0);
}

std::optional<RxPacket> Link::receive(TickType_t timeout) {
    Ready ready;
    if (xQueueReceive(ready_, &ready, timeout) != pdTRUE) return std::nullopt;

    const uint8_t *data = slots_[ready.slot].data();
    proto::Header header;
    std::memcpy(&header, data, sizeof(header));
    return RxPacket(this, ready.slot, header, {data + sizeof(header), header.length});
}

bool Link::valid(int slot, size_t size) const {
    constexpr size_t kOverhead = sizeof(proto::Header) + sizeof(uint32_t);
    if (size < kOverhead) return false;

    const uint8_t *data = slots_[slot].data();
    proto::Header header;
    std::memcpy(&header, data, sizeof(header));
    if (header.version != proto::kVersion || header.length != size - kOverhead) return false;

    uint32_t expected;
    std::memcpy(&expected, data + size - sizeof(uint32_t), sizeof(expected));
    return crc32(0, {data, size - sizeof(uint32_t)}) == expected;
}

void Link::rxEntry(void *arg) {
    static_cast<Link *>(arg)->rxLoop();
}

void Link::rxLoop() {
    std::array<uint8_t, kRxChunk> chunk;
    CobsDecoder decoder;
    int slot;
    xQueueReceive(free_, &slot, portMAX_DELAY);
    decoder.reset(slots_[slot].span());

    while (true) {
        size_t n = transport_.read(chunk, pdMS_TO_TICKS(100));
        for (size_t i = 0; i < n; ++i) {
            switch (decoder.push(chunk[i])) {
            case CobsDecoder::Result::Packet:
                if (valid(slot, decoder.size())) {
                    Ready ready{slot, decoder.size()};
                    xQueueSend(ready_, &ready, portMAX_DELAY);
                    // Waits here if the vision task still holds both buffers; USB flow control
                    // then makes the host wait too
                    xQueueReceive(free_, &slot, portMAX_DELAY);
                } else {
                    ++rxErrors_;
                }
                decoder.reset(slots_[slot].span());
                break;
            case CobsDecoder::Result::Error:
                ++rxErrors_;
                break;
            case CobsDecoder::Result::Pending:
                break;
            }
        }
    }
}

bool Link::sink(void *context, std::span<const uint8_t> bytes) {
    auto *link = static_cast<Link *>(context);
    return link->transport_.write(bytes, link->txTimeout_);
}

bool Link::send(proto::MsgType type, std::initializer_list<std::span<const uint8_t>> parts, TickType_t timeout) {
    if (!transport_.connected()) return false;
    // A task logging from inside send() must not wait for the lock it already holds
    if (xSemaphoreGetMutexHolder(txLock_) == xTaskGetCurrentTaskHandle()) return false;
    if (xSemaphoreTake(txLock_, timeout) != pdTRUE) return false;

    size_t length = 0;
    for (const auto &part : parts) length += part.size();

    proto::Header header{proto::kVersion, type, txSeq_++, static_cast<uint32_t>(length)};
    txTimeout_ = timeout;

    CobsEncoder encoder(&Link::sink, this);
    uint32_t crc = crc32(0, proto::bytesOf(header));
    bool ok = encoder.write(proto::bytesOf(header));
    for (const auto &part : parts) {
        if (!ok) break;
        crc = crc32(crc, part);
        ok = encoder.write(part);
    }
    ok = ok && encoder.write(proto::bytesOf(crc));
    ok = encoder.finish() && ok;

    if (!ok) {
        // Close the broken packet so the host resyncs on the next one
        static constexpr uint8_t kDelimiter = 0x00;
        (void)transport_.write({&kDelimiter, 1}, timeout);
    }
    xSemaphoreGive(txLock_);
    return ok;
}

bool Link::sendAck(proto::MsgType forType, bool ok) {
    proto::Ack ack{forType, static_cast<uint8_t>(ok), {}};
    return send(proto::MsgType::Ack, {proto::bytesOf(ack)});
}

bool Link::sendError(proto::MsgType forType, proto::ErrorCode code, std::string_view message) {
    proto::ErrorMsg error{forType, code, {}};
    ESP_LOGW(TAG, "Request 0x%02x failed: %.*s", static_cast<unsigned>(forType),
             static_cast<int>(message.size()), message.data());
    return send(proto::MsgType::Error,
                {proto::bytesOf(error), {reinterpret_cast<const uint8_t *>(message.data()), message.size()}});
}

void Link::captureLogs() {
    if (logTarget_ == this) return;
    logRing_ = xRingbufferCreate(kLogRingSize, RINGBUF_TYPE_NOSPLIT);
    if (!logRing_) ESP_ERROR_CHECK(ESP_ERR_NO_MEM);
    xTaskCreatePinnedToCore(logEntry, "link_log", kLogTaskStack, this, kLogTaskPriority, &logTask_, 0);
    logTarget_ = this;
    previousLog_ = esp_log_set_vprintf(&Link::logVprintf);
}

int Link::logVprintf(const char *format, va_list args) {
    Link *link = logTarget_;
    if (!link || xPortInIsrContext()) return 0;

    char line[kMaxLogLine];
    int n = vsnprintf(line, sizeof(line), format, args);
    if (n <= 0) return n;
    size_t length = std::min(static_cast<size_t>(n), sizeof(line) - 1);
    // Queue only, never wait: if the ring is full the line is dropped
    xRingbufferSend(link->logRing_, line, length, 0);
    return n;
}

void Link::logEntry(void *arg) {
    static_cast<Link *>(arg)->logLoop();
}

void Link::logLoop() {
    while (true) {
        size_t length = 0;
        auto *line = static_cast<const uint8_t *>(xRingbufferReceive(logRing_, &length, portMAX_DELAY));
        if (!line) continue;
        // Dropped when no host is listening
        send(proto::MsgType::Log, {{line, length}}, kLogTimeout);
        vRingbufferReturnItem(logRing_, const_cast<uint8_t *>(line));
    }
}
