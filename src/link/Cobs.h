#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

/**
 * COBS (Consistent Overhead Byte Stuffing) framing. Encoded data never contains 0x00, so a 0x00
 * marks a packet boundary and the receiver can always resync after garbage or a lost byte.
 * Overhead is 1 byte per 254, plus the delimiters.
 */

// Encodes a packet piece by piece without buffering it all, so multi-part packets (header,
// metadata, pixels) go straight from where they live to the transport.
class CobsEncoder {

public:
    // Receives encoded bytes in chunks; return false to abort the packet
    using Sink = bool (*)(void *context, std::span<const uint8_t> bytes);

    // Starts the packet with a delimiter so stray text before it can't merge into it
    CobsEncoder(Sink sink, void *context);
    CobsEncoder(const CobsEncoder &) = delete;
    CobsEncoder &operator=(const CobsEncoder &) = delete;
    CobsEncoder(CobsEncoder &&) = delete;
    CobsEncoder &operator=(CobsEncoder &&) = delete;

    [[nodiscard]] bool write(std::span<const uint8_t> data);

    // Flush the last block and the closing delimiter
    [[nodiscard]] bool finish();

private:
    void emitBlock(uint8_t code);
    void put(uint8_t byte);
    void flush();

    Sink sink_;
    void *context_;
    std::array<uint8_t, 254> block_{};   // pending non-zero bytes
    size_t blockLength_ = 0;
    bool lastBlockFull_ = false;         // a 254-byte block has no implied zero after it
    std::array<uint8_t, 512> out_{};     // encoded bytes waiting for the sink
    size_t outLength_ = 0;
    bool ok_ = true;
};

// Decodes one byte at a time into a caller-provided buffer.
class CobsDecoder {

public:
    enum class Result : uint8_t {
        Pending,    // need more bytes
        Packet,     // size() bytes of the destination hold a whole packet
        Error,      // packet corrupt or too big for the destination; it was dropped
    };

    void reset(std::span<uint8_t> destination);

    [[nodiscard]] Result push(uint8_t byte);

    [[nodiscard]] size_t size() const { return length_; }

private:
    bool append(uint8_t byte);

    std::span<uint8_t> destination_;
    size_t length_ = 0;
    uint8_t remaining_ = 0;    // data bytes left in the current block
    uint8_t code_ = 0;         // code byte of the current block
    bool started_ = false;
    bool discarding_ = false;  // skip to the next delimiter
};
