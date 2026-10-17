#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "servo_config.h"
#include "driver/gpio.h"
#include <stdio.h>
#include <algorithm>
#include "Servo.h"
#include "PanTilt.h"
#include "Camera.h"
#include "FaceDetector.h"
#include "Tracker.h"

#define TAG "servos"

#define RAW_MIN_US 400
#define RAW_MAX_US 2600
#define MARGIN_US       20

// GPIO mappings
#define BTN_CAL   GPIO_NUM_4   // D3
#define BTN_UP    GPIO_NUM_5   // D4
#define BTN_DOWN  GPIO_NUM_6   // D5

// Buttons settings
#define US_PER_STEP     6

// Tracker runs on core 1 so detection never competes with the servo task on core 0
#define TRACKER_STACK   16384
#define TRACKER_PRIO    3

static servo_config config;

// Init button gpios
static void buttons_init(void) {
    gpio_config_t io = {
        .pin_bit_mask = (1ULL << BTN_CAL) | (1ULL << BTN_UP) | (1ULL << BTN_DOWN),
        .mode = GPIO_MODE_INPUT,
        .pull_up_en = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&io));
}

// Check button pressed
static bool pressed(gpio_num_t pin) {
    if (gpio_get_level(pin) == 0) {
        //ESP_LOGI(TAG, "%d button pressed", pin);
        return true;
    }
    return false;
}

// Move to one end: hold Up/Down to move, press Cal to confirm this spot.
static int nudge_to_end(Servo &servo, const char *name, const char *which) {
    int us = 1500;
    servo.writeUS(us);
    ESP_LOGI(TAG, "%s servo - %s end: hold Up/Down to move, press Calibrate to confirm", name, which);

    // Wait for the calibrate button to be released first, so the press that
    // entered calibration doesn't immediately confirm this end.
    while (pressed(BTN_CAL)) vTaskDelay(pdMS_TO_TICKS(10));
    vTaskDelay(pdMS_TO_TICKS(150));

    while (1) {
        if (pressed(BTN_CAL)) break;            // confirm setting
        if (pressed(BTN_UP))   {
            us = std::clamp(us + US_PER_STEP, RAW_MIN_US, RAW_MAX_US);
        }
        if (pressed(BTN_DOWN)) {
            us = std::clamp(us - US_PER_STEP, RAW_MIN_US, RAW_MAX_US);
        }
        servo.writeUS(us);
        //ESP_LOGI(TAG, "  %s %s: %d us   \r", name, which, us);
        fflush(stdout);
        vTaskDelay(pdMS_TO_TICKS(50));
    }

    while (pressed(BTN_CAL)) vTaskDelay(pdMS_TO_TICKS(10));
    vTaskDelay(pdMS_TO_TICKS(150));

    ESP_LOGI(TAG, "%s %s end set at %d us", name, which, us);
    return us;
}

static void calibrate_servo(Servo &servo, const char *name, int *out_min, int *out_max) {
    ESP_LOGI(TAG, "=== Calibrating %s servo (remove horn if it may jam) ===", name);
    int hi = nudge_to_end(servo, name, "HIGH");
    int lo = nudge_to_end(servo, name, "LOW");

    *out_max = hi - MARGIN_US;
    *out_min = lo + MARGIN_US;
    if (*out_min > *out_max) { int t = *out_min; *out_min = *out_max; *out_max = t; }

    servo.writeUS(1500);
    ESP_LOGI(TAG, "%s done: min %d us, max %d us", name, *out_min, *out_max);
}

static void run_calibration(Servo &pan, Servo &tilt) {
    int pan_min, pan_max, tilt_min, tilt_max;
    calibrate_servo(tilt, "tilt", &tilt_min, &tilt_max);
    calibrate_servo(pan,  "pan",  &pan_min,  &pan_max);

    config.version = CONFIG_VERSION;
    config.min_us[PAN_SERVO]  = pan_min;  config.max_us[PAN_SERVO]  = pan_max;
    config.min_us[TILT_SERVO] = tilt_min; config.max_us[TILT_SERVO] = tilt_max;

    // Save new servos config
    bool ok = save_config(&config);
    ESP_LOGI(TAG, "Calibration %s: pan %d..%d, tilt %d..%d", ok ? "saved" : "SAVE FAILED", pan_min, pan_max, tilt_min, tilt_max);

}

static void tracker_task(void *arg) {
    static_cast<Tracker *>(arg)->run();
}

extern "C" void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(2000));   // give the USB monitor time to reconnect
    ESP_LOGI(TAG, "Program start");

    init_storage();
    ESP_LOGI(TAG, "Flash storage initialized");
    // Static so they outlive app_main; the motion and tracker tasks use them forever
    static Servo pan(static_cast<gpio_num_t>(SERVO_PINS[PAN_SERVO]));
    static Servo tilt(static_cast<gpio_num_t>(SERVO_PINS[TILT_SERVO]));
    ESP_LOGI(TAG, "Servos initialized");
    buttons_init();
    ESP_LOGI(TAG, "Buttons initialized");

    ESP_LOGI(TAG, "Press CAL within 3 s to calibrate...");
    bool want_cal = false;
    for (int i = 0; i < 150; ++i) {          // 150 * 20 ms = 3 s
        if (pressed(BTN_CAL)) { want_cal = true; break; }
        vTaskDelay(pdMS_TO_TICKS(20));
    }

    // Load saved calibration (or defaults if none)
    if (load_config(&config)) {
        ESP_LOGI(TAG, "Loaded saved calibration");
    } else {
        ESP_LOGW(TAG, "No saved calibration - using defaults until you calibrate");
    }

    // Calibrate if the button is held at startup
    if (want_cal) {
        ESP_LOGI(TAG, "Calibrate button held - entering calibration");
        run_calibration(pan, tilt);
    } else {
        ESP_LOGI(TAG, "Skipping calibration - hold Calibrate at reset to run it");
    }

    pan.setRange(config.min_us[PAN_SERVO], config.max_us[PAN_SERVO]);
    tilt.setRange(config.min_us[TILT_SERVO], config.max_us[TILT_SERVO]);

    // Move all servos to the center and start smooth motion control
    static PanTilt mount(pan, tilt,
                         Range{config.pan_min, config.pan_max},
                         Range{config.tilt_min, config.tilt_max});
    mount.start(mount.panRange().center(), mount.tiltRange().center());
    vTaskDelay(pdMS_TO_TICKS(1000));
    ESP_LOGI(TAG, "Reset Motors to center");

    Camera::init();
    static FaceDetector detector;
    static Tracker tracker(mount, detector);

    xTaskCreatePinnedToCore(tracker_task, "tracker", TRACKER_STACK, &tracker, TRACKER_PRIO, nullptr, 1);
}
