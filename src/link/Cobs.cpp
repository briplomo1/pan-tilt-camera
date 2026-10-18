#include "link/Cobs.h"

CobsEncoder::CobsEncoder(Sink sink, void *context) : sink_(sink), context_(context) {
    put(0x00);
}

void CobsEncoder::put(uint8_t byte) {
    out_[outLength_++] = byte;
    if (outLength_ == out_.size()) flush();
}

void CobsEncoder::flush() {
    if (outLength_ == 0) return;
    if (ok_) ok_ = sink_(context_, {out_.data(), outLength_});
    outLength_ = 0;
}

void CobsEncoder::emitBlock(uint8_t code) {
    put(code);
    for (size_t i = 0; i < blockLength_; ++i) put(block_[i]);
    blockLength_ = 0;
}

bool CobsEncoder::write(std::span<const uint8_t> data) {
    for (uint8_t byte : data) {
        if (byte == 0) {
            // The code byte says how far the next zero is
            emitBlock(static_cast<uint8_t>(blockLength_ + 1));
            lastBlockFull_ = false;
        } else {
            block_[blockLength_++] = byte;
            lastBlockFull_ = false;
            if (blockLength_ == block_.size()) {
                emitBlock(0xFF);
                lastBlockFull_ = true;
            }
        }
    }
    return ok_;
}

bool CobsEncoder::finish() {
    if (blockLength_ > 0 || !lastBlockFull_) {
        emitBlock(static_cast<uint8_t>(blockLength_ + 1));
    }
    put(0x00);
    flush();
    return ok_;
}

void CobsDecoder::reset(std::span<uint8_t> destination) {
    destination_ = destination;
    length_ = 0;
    remaining_ = 0;
    code_ = 0;
    started_ = false;
    discarding_ = false;
}

bool CobsDecoder::append(uint8_t byte) {
    if (length_ >= destination_.size()) {
        discarding_ = true;
        return false;
    }
    destination_[length_++] = byte;
    return true;
}

CobsDecoder::Result CobsDecoder::push(uint8_t byte) {
    if (byte == 0x00) {
        bool hadData = started_ || discarding_;
        bool complete = started_ && !discarding_ && remaining_ == 0;
        started_ = false;
        discarding_ = false;
        remaining_ = 0;
        if (!hadData) {
            length_ = 0;       // back-to-back delimiters: nothing between them
            return Result::Pending;
        }
        if (!complete) {
            length_ = 0;
            return Result::Error;
        }
        return Result::Packet;
    }

    if (discarding_) return Result::Pending;

    if (!started_) {
        // First code byte of a new packet
        started_ = true;
        length_ = 0;
        code_ = byte;
        remaining_ = byte - 1;
        return Result::Pending;
    }

    if (remaining_ == 0) {
        // Next code byte: the previous block ended with a zero unless it was a full 254-byte block
        if (code_ != 0xFF && !append(0x00)) return Result::Pending;
        code_ = byte;
        remaining_ = byte - 1;
        return Result::Pending;
    }

    if (append(byte)) --remaining_;
    return Result::Pending;
}
