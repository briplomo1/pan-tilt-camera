#pragma once

#include <cstddef>
#include <cstdint>

/**
 * Layout of the model pack written into the `model` partition by tools/models/pack_models.py.
 * Keep the two in sync. All fields are little-endian.
 *
 *   Header | Entry[count] | payloads (each 16-byte aligned, offsets from partition start)
 */
namespace modelpack {

inline constexpr char kMagic[4] = {'M', 'P', 'K', '1'};
inline constexpr uint16_t kVersion = 1;
inline constexpr size_t kAlignment = 16;
inline constexpr size_t kMaxEntries = 64;

#pragma pack(push, 1)
struct Header {
    char magic[4];
    uint16_t version;
    uint16_t count;
    uint32_t tableCrc;      // CRC32 of the entry table
    uint32_t reserved;
};

struct Entry {
    char name[32];          // NUL-terminated
    char format[8];         // "tflite", "espdl", "bin", ...
    uint32_t offset;
    uint32_t size;
    uint32_t crc32;         // CRC32 of the payload
    uint32_t reserved;
};
#pragma pack(pop)

static_assert(sizeof(Header) == 16);
static_assert(sizeof(Entry) == 56);

} // namespace modelpack
