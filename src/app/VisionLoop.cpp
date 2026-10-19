#include "app/VisionLoop.h"
#include <cstdio>
#include "esp_app_desc.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "app/Wire.h"

namespace {
constexpr const char *TAG = "vision";

// Detection runs here, on core 1, so it never competes with the servo task on core 0
constexpr int kTaskPriority = 3;
constexpr int kTaskStack = 16384;
constexpr TickType_t kIdleWait = pdMS_TO_TICKS(50);
// Telemetry is queued and sent from its own task, so a live-view frame holding the link
// never costs a telemetry packet. 32 packets is ~1.8 s of frames at 18 fps.
constexpr UBaseType_t kTelemetryDepth = 32;
constexpr int kTelemetryTaskPriority = 2;   // above the preview task, below the vision task
constexpr int kTelemetryTaskStack = 3072;
} // namespace

VisionLoop::VisionLoop(PanTilt &mount, Link &link, DetectorSlot &slot, BenchService &bench, PreviewStreamer &preview)
    : link_(link), slot_(slot), bench_(bench), preview_(preview), tracker_(mount) {
    tracker_.setObserver(this);
}

VisionLoop::~VisionLoop() {
    if (task_) vTaskDelete(task_);
    if (telemetryTask_) vTaskDelete(telemetryTask_);
    if (telemetryQueue_) vQueueDelete(telemetryQueue_);
}

void VisionLoop::start(std::string_view modelId) {
    initialModel_ = modelId;
    telemetryQueue_ = xQueueCreate(kTelemetryDepth, sizeof(proto::TelemetryMsg));
    if (!telemetryQueue_) ESP_ERROR_CHECK(ESP_ERR_NO_MEM);
    xTaskCreatePinnedToCore(telemetryEntry, "telemetry", kTelemetryTaskStack, this, kTelemetryTaskPriority,
                            &telemetryTask_, 0);
    xTaskCreatePinnedToCore(taskEntry, "vision", kTaskStack, this, kTaskPriority, &task_, 1);
}

void VisionLoop::taskEntry(void *arg) {
    static_cast<VisionLoop *>(arg)->run();
}

void VisionLoop::telemetryEntry(void *arg) {
    static_cast<VisionLoop *>(arg)->telemetryLoop();
}

void VisionLoop::telemetryLoop() {
    proto::TelemetryMsg msg{};
    while (true) {
        if (xQueueReceive(telemetryQueue_, &msg, portMAX_DELAY) != pdTRUE) continue;
        link_.send(proto::MsgType::Telemetry, {proto::bytesOf(msg)});
    }
}

void VisionLoop::run() {
    if (!slot_.load(initialModel_).ok) {
        ESP_LOGE(TAG, "Default model '%s' failed to load - waiting for the host to load one", initialModel_.c_str());
    }

    while (true) {
        // Don't wait for packets while tracking: the camera sets the pace
        bool tracking = mode_ == proto::Mode::Track && slot_.get();
        TickType_t wait = tracking ? 0 : kIdleWait;
        while (std::optional<RxPacket> packet = link_.receive(wait)) {
            dispatch(*packet);
            wait = 0;
        }

        if (mode_ == proto::Mode::Track) {
            if (IDetector *detector = slot_.get()) tracker_.step(*detector);
        }
    }
}

void VisionLoop::dispatch(const RxPacket &packet) {
    switch (packet.type()) {
    case proto::MsgType::Hello: hello(); return;
    case proto::MsgType::SetMode: setMode(packet); return;
    case proto::MsgType::Subscribe: subscribe(packet); return;
    case proto::MsgType::Sync: sync(packet); return;
    case proto::MsgType::SetPreview: setPreview(packet); return;
    default: break;
    }
    if (!bench_.handle(packet, mode_)) {
        link_.sendError(packet.type(), proto::ErrorCode::Unsupported, "Unknown request");
    }
}

void VisionLoop::hello() {
    proto::HelloAck ack{};
    ack.protocol = proto::kVersion;
    ack.mode = mode_;
    ack.freeInternal = heap_caps_get_free_size(MALLOC_CAP_INTERNAL);
    ack.freePsram = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ack.totalInternal = heap_caps_get_total_size(MALLOC_CAP_INTERNAL);
    ack.totalPsram = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    const esp_app_desc_t *app = esp_app_get_description();
    snprintf(ack.firmware, sizeof(ack.firmware), "%s %s", app->date, app->time);
    if (const IDetector *detector = slot_.get()) copyString(ack.activeModel, detector->info().id);
    link_.send(proto::MsgType::HelloAck, {proto::bytesOf(ack)});
}

void VisionLoop::setMode(const RxPacket &packet) {
    auto request = packet.as<proto::SetModeRequest>();
    if (!request || (request->mode != proto::Mode::Track && request->mode != proto::Mode::Bench)) {
        link_.sendError(packet.type(), proto::ErrorCode::BadRequest, "Mode must be 0 (track) or 1 (bench)");
        return;
    }
    mode_ = request->mode;
    if (mode_ == proto::Mode::Bench) {
        tracker_.pause();
    } else {
        tracker_.resume();
    }
    link_.sendAck(packet.type(), true);
}

void VisionLoop::subscribe(const RxPacket &packet) {
    auto request = packet.as<proto::SubscribeRequest>();
    telemetry_ = request && request->telemetry != 0;
    link_.sendAck(packet.type(), request.has_value());
}

void VisionLoop::sync(const RxPacket &packet) {
    auto request = packet.as<proto::SyncRequest>();
    proto::SyncAckMsg ack{};
    ack.marker = request ? request->marker : 0;
    ack.timeUs = esp_timer_get_time();
    link_.send(proto::MsgType::SyncAck, {proto::bytesOf(ack)});
}

void VisionLoop::setPreview(const RxPacket &packet) {
    auto request = packet.as<proto::SetPreviewRequest>();
    if (!request) {
        link_.sendError(packet.type(), proto::ErrorCode::BadRequest, "SetPreview payload too short");
        return;
    }
    bool ok = preview_.configure(PreviewSettings{
        .enabled = request->enabled != 0,
        .format = request->format,
        .scaleShift = request->scaleShift,
        .everyN = request->everyN,
        .jpegQuality = request->jpegQuality,
    });
    link_.sendAck(packet.type(), ok);
}

void VisionLoop::onTrackFrame(const TrackSample &sample) {
    std::string_view modelId;
    if (const IDetector *detector = slot_.get()) modelId = detector->info().id;

    if (telemetry_) {
        proto::TelemetryMsg msg{};
        msg.timeUs = sample.captureUs;
        msg.frameId = sample.frameId;
        msg.state = static_cast<uint8_t>(sample.state);
        msg.faceCount = static_cast<uint8_t>(std::min<size_t>(sample.faces.size(), 255));
        msg.hasTarget = sample.targetIndex >= 0;
        msg.pan = sample.pan;
        msg.tilt = sample.tilt;
        msg.panTarget = sample.panTarget;
        msg.tiltTarget = sample.tiltTarget;
        if (sample.targetIndex >= 0) msg.target = toWire(sample.faces[sample.targetIndex]);
        msg.timing = toWire(sample.timing);
        // Never wait: if the queue is full the packet is dropped rather than slowing the tracker
        if (xQueueSend(telemetryQueue_, &msg, 0) != pdTRUE && ++telemetryDropped_ % 100 == 1) {
            ESP_LOGW(TAG, "Telemetry queue full (%lu dropped)", static_cast<unsigned long>(telemetryDropped_));
        }
    }

    preview_.offer(PreviewFrame{
        .image = sample.image,
        .frameId = sample.frameId,
        .timeUs = sample.captureUs,
        .trackerState = static_cast<uint8_t>(sample.state),
        .mode = mode_,
        .pan = sample.pan,
        .tilt = sample.tilt,
        .faces = sample.faces,
        .targetIndex = sample.targetIndex,
        .timing = sample.timing,
        .modelId = modelId,
    });
}
