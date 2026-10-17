#include "Camera.h"
#include "esp_log.h"

namespace {
constexpr const char *TAG = "camera";

// XIAO ESP32S3 Sense camera connector
constexpr int kPinPwdn  = -1;
constexpr int kPinReset = -1;
constexpr int kPinXclk  = 10;
constexpr int kPinSda   = 40;
constexpr int kPinScl   = 39;
constexpr int kPinD7    = 48;
constexpr int kPinD6    = 11;
constexpr int kPinD5    = 12;
constexpr int kPinD4    = 14;
constexpr int kPinD3    = 16;
constexpr int kPinD2    = 18;
constexpr int kPinD1    = 17;
constexpr int kPinD0    = 15;
constexpr int kPinVsync = 38;
constexpr int kPinHref  = 47;
constexpr int kPinPclk  = 13;
} // namespace

void Camera::init() {
    camera_config_t config = {};
    config.pin_pwdn = kPinPwdn;
    config.pin_reset = kPinReset;
    config.pin_xclk = kPinXclk;
    config.pin_sccb_sda = kPinSda;
    config.pin_sccb_scl = kPinScl;
    config.pin_d7 = kPinD7;
    config.pin_d6 = kPinD6;
    config.pin_d5 = kPinD5;
    config.pin_d4 = kPinD4;
    config.pin_d3 = kPinD3;
    config.pin_d2 = kPinD2;
    config.pin_d1 = kPinD1;
    config.pin_d0 = kPinD0;
    config.pin_vsync = kPinVsync;
    config.pin_href = kPinHref;
    config.pin_pclk = kPinPclk;

    config.xclk_freq_hz = 20000000;
    // Servos own LEDC_TIMER_0 and the low channels
    config.ledc_timer = LEDC_TIMER_1;
    config.ledc_channel = LEDC_CHANNEL_7;

    config.pixel_format = PIXFORMAT_RGB565;   // detector takes raw pixels, no JPEG decode
    config.frame_size = FRAMESIZE_QVGA;       // 320x240; the face model works at 160x120 anyway
    config.jpeg_quality = 12;
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
    config.grab_mode = CAMERA_GRAB_LATEST;    // stale frames would make tracking lag

    ESP_ERROR_CHECK(esp_camera_init(&config));

    sensor_t *sensor = esp_camera_sensor_get();
    ESP_LOGI(TAG, "Camera ready, sensor PID 0x%02x", sensor ? sensor->id.PID : 0);
}

Frame Camera::capture() {
    return Frame(esp_camera_fb_get());
}
