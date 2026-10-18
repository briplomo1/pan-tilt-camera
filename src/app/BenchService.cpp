#include "app/BenchService.h"
#include <algorithm>
#include <array>
#include <vector>
#include "esp_log.h"
#include "esp_timer.h"
#include "Camera.h"
#include "Tracker.h"
#include "app/Wire.h"
#include "memory/MemoryProbe.h"
#include "vision/Jpeg.h"

namespace {
constexpr const char *TAG = "bench";

constexpr size_t kMaxWireFaces = 64;
constexpr uint16_t kMaxCaptures = 100;
constexpr size_t kJpegBytes = 64 * 1024;
constexpr TickType_t kResultTimeout = pdMS_TO_TICKS(1000);
constexpr TickType_t kFrameTimeout = pdMS_TO_TICKS(2000);

bool isRawFormat(proto::ImageFormat format) {
    return format == proto::ImageFormat::Rgb565Be || format == proto::ImageFormat::Rgb888 ||
           format == proto::ImageFormat::Gray8;
}
} // namespace

BenchService::BenchService(Link &link, DetectorSlot &slot, const DetectorRegistry &registry, const ModelStore &store,
                           PreviewStreamer &preview, const PanTilt &mount)
    : link_(link), slot_(slot), registry_(registry), store_(store), preview_(preview), mount_(mount) {}

bool BenchService::handle(const RxPacket &packet, proto::Mode mode) {
    switch (packet.type()) {
    case proto::MsgType::ListModels: listModels(); return true;
    case proto::MsgType::LoadModel: loadModel(packet); return true;
    case proto::MsgType::SetThreshold: setThreshold(packet); return true;
    case proto::MsgType::Infer: infer(packet, mode); return true;
    case proto::MsgType::Capture: capture(packet); return true;
    case proto::MsgType::Echo: link_.send(proto::MsgType::EchoReply, {packet.payload()}); return true;
    case proto::MsgType::ListStore: listStore(); return true;
    case proto::MsgType::LoadBlob: loadBlob(packet); return true;
    default: return false;
    }
}

void BenchService::listModels() {
    std::vector<proto::ModelListEntry> entries;
    for (const DetectorEntry &entry : registry_.entries()) {
        proto::ModelListEntry wire{};
        copyString(wire.id, entry.id);
        copyString(wire.backend, entry.backend);
        entries.push_back(wire);
    }
    proto::ModelListHeader header{static_cast<uint8_t>(std::min<size_t>(entries.size(), 255)), {}};
    std::span<const proto::ModelListEntry> list(entries.data(), header.count);
    link_.send(proto::MsgType::ModelList, {proto::bytesOf(header), proto::bytesOf(list)});
}

void BenchService::loadModel(const RxPacket &packet) {
    auto request = packet.as<proto::LoadModelRequest>();
    if (!request) {
        link_.sendError(packet.type(), proto::ErrorCode::BadRequest, "LoadModel needs a model id");
        return;
    }
    std::string_view id = readString(request->id);
    if (!registry_.find(id)) {
        link_.sendError(packet.type(), proto::ErrorCode::UnknownModel, id);
        return;
    }
    sendModelInfo(slot_.load(id));
}

void BenchService::sendModelInfo(const LoadReport &report) {
    proto::ModelInfoMsg msg{};
    msg.ok = report.ok && slot_.get();
    if (const IDetector *detector = slot_.get()) {
        const ModelInfo &info = detector->info();
        copyString(msg.id, info.id);
        copyString(msg.backend, info.backend);
        copyString(msg.quantization, info.quantization);
        msg.inputWidth = info.inputWidth;
        msg.inputHeight = info.inputHeight;
        msg.modelBytes = static_cast<uint32_t>(info.modelBytes);
    }
    msg.loadUs = report.loadUs;
    msg.internalBytes = report.memory.internalBytes;
    msg.psramBytes = report.memory.psramBytes;
    msg.minFreeInternal = report.memory.minFreeInternal;
    msg.minFreePsram = report.memory.minFreePsram;
    link_.send(proto::MsgType::ModelInfo, {proto::bytesOf(msg)});
}

void BenchService::setThreshold(const RxPacket &packet) {
    auto request = packet.as<proto::SetThresholdRequest>();
    if (!request || !(request->threshold >= 0.0f && request->threshold <= 1.0f)) {
        link_.sendError(packet.type(), proto::ErrorCode::BadRequest, "Threshold must be 0..1");
        return;
    }
    slot_.setScoreThreshold(request->threshold);
    link_.sendAck(packet.type(), true);
}

void BenchService::infer(const RxPacket &packet, proto::Mode mode) {
    auto header = packet.as<proto::ImageHeader>();
    if (!header || !isRawFormat(header->format)) {
        link_.sendError(packet.type(), proto::ErrorCode::Unsupported, "Infer takes raw RGB565/RGB888/gray images");
        return;
    }
    ImageView image{packet.payload().data() + sizeof(proto::ImageHeader), header->width, header->height,
                    static_cast<PixelFormat>(header->format)};
    if (!image || packet.payload().size() != sizeof(proto::ImageHeader) + image.bytes()) {
        link_.sendError(packet.type(), proto::ErrorCode::BadRequest, "Image size doesn't match its header");
        return;
    }
    IDetector *detector = slot_.get();
    if (!detector) {
        link_.sendError(packet.type(), proto::ErrorCode::NoModel, "No model loaded");
        return;
    }

    DetectTiming timing;
    std::span<const Face> faces;
    MemoryUsage memory;
    {
        MemoryProbe probe;
        faces = detector->detect(image, timing);
        memory = probe.usage();
    }

    std::array<proto::WireFace, kMaxWireFaces> wire;
    size_t count = std::min(faces.size(), kMaxWireFaces);
    for (size_t i = 0; i < count; ++i) wire[i] = toWire(faces[i]);

    proto::InferResultMsg result{};
    result.frameId = header->frameId;
    result.timing = toWire(timing);
    result.minFreeInternal = memory.minFreeInternal;
    result.minFreePsram = memory.minFreePsram;
    result.faceCount = static_cast<uint8_t>(count);
    link_.send(proto::MsgType::InferResult,
               {proto::bytesOf(result), proto::bytesOf(std::span<const proto::WireFace>(wire.data(), count))},
               kResultTimeout);

    preview_.offer(PreviewFrame{
        .image = image,
        .frameId = header->frameId,
        .timeUs = esp_timer_get_time(),
        .trackerState = static_cast<uint8_t>(TrackState::Paused),
        .mode = mode,
        .pan = mount_.pan(),
        .tilt = mount_.tilt(),
        .faces = faces,
        .targetIndex = -1,
        .timing = timing,
        .modelId = detector->info().id,
    });
}

void BenchService::capture(const RxPacket &packet) {
    auto request = packet.as<proto::CaptureRequest>();
    bool jpeg = request && request->format == proto::ImageFormat::Jpeg;
    if (!request || request->count == 0 || request->count > kMaxCaptures ||
        !(jpeg || request->format == proto::ImageFormat::Rgb565Be)) {
        link_.sendError(packet.type(), proto::ErrorCode::BadRequest, "Capture 1..100 frames as RGB565 or JPEG");
        return;
    }

    HeapBuffer encoded;
    if (jpeg) {
        encoded = HeapBuffer::allocate(kJpegBytes, kPsramCaps);
        if (!encoded) {
            link_.sendError(packet.type(), proto::ErrorCode::NoMemory, "No PSRAM for JPEG output");
            return;
        }
    }

    for (uint16_t i = 0; i < request->count; ++i) {
        Frame frame = Camera::capture();
        ImageView image = frame.view();
        if (!image) {
            link_.sendError(packet.type(), proto::ErrorCode::CameraFailed, "Camera capture failed");
            return;
        }

        std::span<const uint8_t> bytes(image.data, image.bytes());
        if (jpeg) {
            size_t length = encodeJpeg(image, request->jpegQuality ? request->jpegQuality : 90, encoded.span());
            if (length == 0) {
                link_.sendError(packet.type(), proto::ErrorCode::NoMemory, "JPEG didn't fit");
                return;
            }
            bytes = {encoded.data(), length};
        }

        proto::FrameMsg msg{};
        msg.timeUs = esp_timer_get_time();
        msg.pan = mount_.pan();
        msg.tilt = mount_.tilt();
        msg.image = proto::ImageHeader{captureId_++, image.width, image.height, request->format, {}};
        if (!link_.send(proto::MsgType::Frame, {proto::bytesOf(msg), bytes}, kFrameTimeout)) {
            ESP_LOGW(TAG, "Capture aborted: host stopped reading");
            return;
        }
    }
}

void BenchService::listStore() {
    std::vector<proto::StoreListEntry> entries;
    for (const ModelFile &file : store_.files()) {
        proto::StoreListEntry wire{};
        copyString(wire.name, file.name);
        copyString(wire.format, file.format);
        wire.size = file.size;
        wire.crc32 = file.crc32;
        entries.push_back(wire);
    }
    proto::StoreListHeader header{static_cast<uint8_t>(std::min<size_t>(entries.size(), 255)), {}};
    std::span<const proto::StoreListEntry> list(entries.data(), header.count);
    link_.send(proto::MsgType::StoreList, {proto::bytesOf(header), proto::bytesOf(list)});
}

void BenchService::loadBlob(const RxPacket &packet) {
    auto request = packet.as<proto::LoadBlobRequest>();
    if (!request || request->placement > static_cast<uint8_t>(Placement::Internal)) {
        link_.sendError(packet.type(), proto::ErrorCode::BadRequest, "LoadBlob needs a name and placement 0..2");
        return;
    }
    std::string_view name = readString(request->name);
    auto placement = static_cast<Placement>(request->placement);

    proto::BlobInfoMsg msg{};
    copyString(msg.name, name);
    msg.placement = request->placement;
    {
        MemoryProbe probe;
        int64_t start = esp_timer_get_time();
        std::optional<ModelBlob> blob = store_.load(name, placement);
        msg.loadUs = static_cast<uint32_t>(esp_timer_get_time() - start);
        MemoryUsage memory = probe.usage();
        msg.internalBytes = memory.internalBytes;
        msg.psramBytes = memory.psramBytes;
        if (blob) {
            const ModelFile *file = store_.find(name);
            msg.ok = 1;
            msg.size = static_cast<uint32_t>(blob->data().size());
            msg.crc32 = file ? file->crc32 : 0;
            msg.address = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(blob->data().data()));
        }
    }   // blob released here: this command only proves the model can be placed and read back
    link_.send(proto::MsgType::BlobInfo, {proto::bytesOf(msg)});
}
