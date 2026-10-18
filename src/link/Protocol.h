#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>

/**
 * USB protocol between the firmware and the host tools (tools/link/protocol.py mirrors every
 * struct here; keep them in sync and bump kVersion on any change).
 *
 * Packet on the wire:  0x00 | COBS( Header | payload | CRC32(Header + payload) ) | 0x00
 * All fields are little-endian. Requests are stop-and-wait: the host waits for the reply
 * before sending the next request. Preview, Telemetry and Log packets arrive at any time.
 */
namespace proto {

inline constexpr uint8_t kVersion = 1;
inline constexpr size_t kMaxPayload = 160 * 1024;   // a QVGA RGB565 frame plus metadata
inline constexpr size_t kIdLength = 32;

enum class MsgType : uint8_t {
    // Host -> device
    Hello = 0x01,
    ListModels = 0x02,
    LoadModel = 0x03,
    SetThreshold = 0x04,
    Infer = 0x05,
    Capture = 0x06,
    SetMode = 0x07,
    Subscribe = 0x08,
    Sync = 0x09,
    Echo = 0x0A,
    SetPreview = 0x0B,
    ListStore = 0x0C,
    LoadBlob = 0x0D,

    // Device -> host
    HelloAck = 0x81,
    ModelList = 0x82,
    ModelInfo = 0x83,
    Ack = 0x84,
    InferResult = 0x85,
    Frame = 0x86,
    Preview = 0x87,
    Telemetry = 0x88,
    SyncAck = 0x89,
    EchoReply = 0x8A,
    StoreList = 0x8C,
    BlobInfo = 0x8D,
    Log = 0xF0,
    Error = 0xFF,
};

// Values for raw formats match PixelFormat
enum class ImageFormat : uint8_t {
    Rgb565Be = 0,
    Rgb888 = 1,
    Gray8 = 2,
    Jpeg = 3,
};

enum class Mode : uint8_t {
    Track = 0,   // tracker runs, host may watch
    Bench = 1,   // tracker paused and mount held; host drives inference
};

enum class ErrorCode : uint8_t {
    BadRequest = 1,
    UnknownModel = 2,
    NoModel = 3,
    NoMemory = 4,
    CameraFailed = 5,
    Unsupported = 6,
};

#pragma pack(push, 1)

struct Header {
    uint8_t version;
    MsgType type;
    uint16_t seq;
    uint32_t length;    // payload bytes
};

struct WireFace {
    int16_t x0, y0, x1, y1;   // box corners in source-image pixels
    float score;
};

struct WireTiming {
    uint32_t preprocessUs, inferenceUs, postprocessUs, totalUs;
};

// Precedes raw or JPEG image bytes
struct ImageHeader {
    uint32_t frameId;
    uint16_t width;
    uint16_t height;
    ImageFormat format;
    uint8_t reserved[3];
};

struct HelloAck {
    uint8_t protocol;
    Mode mode;
    uint8_t reserved[2];
    uint32_t freeInternal;
    uint32_t freePsram;
    uint32_t totalInternal;
    uint32_t totalPsram;
    char firmware[32];
    char activeModel[kIdLength];
};

struct ModelListHeader {
    uint8_t count;
    uint8_t reserved[3];
};

struct ModelListEntry {
    char id[kIdLength];
    char backend[16];
};

struct LoadModelRequest {
    char id[kIdLength];
};

struct ModelInfoMsg {
    uint8_t ok;
    uint8_t reserved[3];
    char id[kIdLength];
    char backend[16];
    char quantization[32];
    uint16_t inputWidth;
    uint16_t inputHeight;
    uint32_t modelBytes;
    uint32_t loadUs;
    int32_t internalBytes;      // heap taken by loading the model
    int32_t psramBytes;
    uint32_t minFreeInternal;   // lowest free heap while loading
    uint32_t minFreePsram;
};

struct SetThresholdRequest {
    float threshold;
};

struct Ack {
    MsgType forType;
    uint8_t ok;
    uint8_t reserved[2];
};

// Infer request: ImageHeader + pixels. Reply: InferResultMsg + WireFace[faceCount]
struct InferResultMsg {
    uint32_t frameId;
    WireTiming timing;
    uint32_t minFreeInternal;   // lowest free heap during this inference
    uint32_t minFreePsram;
    uint8_t faceCount;
    uint8_t reserved[3];
};

struct CaptureRequest {
    uint16_t count;
    ImageFormat format;         // Rgb565Be or Jpeg
    uint8_t jpegQuality;
};

// Reply to Capture (one per frame): FrameMsg + image bytes
struct FrameMsg {
    int64_t timeUs;
    float pan;
    float tilt;
    ImageHeader image;
};

struct SetModeRequest {
    Mode mode;
    uint8_t reserved[3];
};

struct SubscribeRequest {
    uint8_t telemetry;
    uint8_t reserved[3];
};

// Sent once per tracker frame while subscribed
struct TelemetryMsg {
    int64_t timeUs;
    uint32_t frameId;
    uint8_t state;              // 0 scan, 1 track, 2 paused
    uint8_t faceCount;
    uint8_t hasTarget;
    uint8_t reserved;
    float pan;                  // mount angles when the frame was captured
    float tilt;
    float panTarget;            // target set this frame
    float tiltTarget;
    WireFace target;
    WireTiming timing;
};

struct SyncRequest {
    uint32_t marker;
};

struct SyncAckMsg {
    uint32_t marker;
    uint32_t reserved;
    int64_t timeUs;
};

struct SetPreviewRequest {
    uint8_t enabled;
    ImageFormat format;         // Jpeg, Rgb565Be or Gray8
    uint8_t scaleShift;         // 0 full size, 1 half, 2 quarter
    uint8_t everyN;             // send every Nth frame
    uint8_t jpegQuality;        // 1..100
    uint8_t reserved[3];
};

// Live view frame: PreviewMsg + WireFace[faceCount] + image bytes
struct PreviewMsg {
    int64_t timeUs;
    uint32_t frameId;
    uint8_t trackerState;
    uint8_t faceCount;
    uint8_t targetIndex;        // 0xFF: no target
    Mode mode;
    float pan;
    float tilt;
    WireTiming timing;
    uint32_t freeInternal;
    uint16_t sourceWidth;       // face boxes are in these coordinates
    uint16_t sourceHeight;
    char modelId[kIdLength];
    ImageHeader image;
    uint32_t imageBytes;
};

struct StoreListHeader {
    uint8_t count;
    uint8_t reserved[3];
};

struct StoreListEntry {
    char name[32];
    char format[8];
    uint32_t size;
    uint32_t crc32;
};

struct LoadBlobRequest {
    char name[32];
    uint8_t placement;          // Placement
    uint8_t reserved[3];
};

struct BlobInfoMsg {
    uint8_t ok;
    uint8_t placement;
    uint8_t reserved[2];
    char name[32];
    uint32_t size;
    uint32_t crc32;
    uint32_t loadUs;
    int32_t internalBytes;
    int32_t psramBytes;
    uint32_t address;           // where the bytes were, to confirm flash vs RAM
};

// Followed by a message string
struct ErrorMsg {
    MsgType forType;
    ErrorCode code;
    uint8_t reserved[2];
};

#pragma pack(pop)

static_assert(sizeof(Header) == 8);
static_assert(sizeof(WireFace) == 12);
static_assert(sizeof(WireTiming) == 16);
static_assert(sizeof(ImageHeader) == 12);
static_assert(sizeof(HelloAck) == 84);
static_assert(sizeof(ModelListEntry) == 48);
static_assert(sizeof(ModelInfoMsg) == 112);
static_assert(sizeof(InferResultMsg) == 32);
static_assert(sizeof(CaptureRequest) == 4);
static_assert(sizeof(FrameMsg) == 28);
static_assert(sizeof(TelemetryMsg) == 60);
static_assert(sizeof(SyncAckMsg) == 16);
static_assert(sizeof(SetPreviewRequest) == 8);
static_assert(sizeof(PreviewMsg) == 96);
static_assert(sizeof(StoreListEntry) == 48);
static_assert(sizeof(LoadBlobRequest) == 36);
static_assert(sizeof(BlobInfoMsg) == 60);
static_assert(sizeof(ErrorMsg) == 4);

// View any plain struct or array as the bytes to send
template <typename T>
[[nodiscard]] std::span<const uint8_t> bytesOf(const T &value) {
    static_assert(std::is_trivially_copyable_v<T>);
    return {reinterpret_cast<const uint8_t *>(&value), sizeof(T)};
}

template <typename T>
[[nodiscard]] std::span<const uint8_t> bytesOf(std::span<const T> values) {
    static_assert(std::is_trivially_copyable_v<T>);
    return {reinterpret_cast<const uint8_t *>(values.data()), values.size_bytes()};
}

} // namespace proto
