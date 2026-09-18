#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/ledc.h"
#include "esp_log.h"
#include "servo_config.h"
#include "driver/gpio.h"
#include <stdio.h>

#define TAG "servos"

#define SERVO_FREQ_HZ 50
#define SERVO_RES LEDC_TIMER_13_BIT
#define SERVO_MAX_DUTY ((1<<14) - 1)
#define PERIOD_US 20000

#define RAW_MIN_US 400
#define RAW_MAX_US 2600
#define US_PER_KEY      6                 // pulse change per key repeat
#define MARGIN_US       15

// GPIO mappings
#define BTN_CAL   GPIO_NUM_4   // D3
#define BTN_UP    GPIO_NUM_5   // D4
#define BTN_DOWN  GPIO_NUM_6    // index in SERVO_PINS (D1) - top servo, tilts up/over

// Pan and tilt
#define PAN_MIN    0
#define PAN_MAX    180
#define TILT_MIN   15
#define TILT_MAX   165
#define EL_MIN  ((TILT_MIN) > (180 - TILT_MAX) ? (TILT_MIN) : (180 - TILT_MAX))

// Servo speed settings
#define STEP_DEG       1     // max degrees per update (speed); 6 = 300°/s
#define UPDATE_MS      20    // one servo update period

// Buttons settings
#define US_PER_STEP     6

static servo_config config;

static void servo_init(void) {
    // Define pwm timer
    ledc_timer_config_t timer = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .timer_num = LEDC_TIMER_0,
        .duty_resolution = SERVO_RES,
        .freq_hz = SERVO_FREQ_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    // Set pwm timer
    ESP_ERROR_CHECK(ledc_timer_config(&timer));

    // Configure each led channel
    for(int i=0;i<NUM_SERVOS;++i) {
        // Define gpio channel config
        ledc_channel_config_t ch = {
            .gpio_num = SERVO_PINS[i],
            .speed_mode = LEDC_LOW_SPEED_MODE,
            .channel = (ledc_channel_t)i,
            .timer_sel = LEDC_TIMER_0,
            .duty = 0,
            .hpoint = 0,
        };
        // Set channel config
        ESP_ERROR_CHECK(ledc_channel_config(&ch));
    }
}

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

// Keeps voltage to a certain range given a minimmum 
// and maximum values of that range
static int clamp(int v, int min_v, int max_v) {
    return v < min_v ? min_v : (v > max_v ? max_v: v);
}

static int clampi(int v, int lo, int hi) { 
    return v < lo ? lo : (v > hi ? hi : v); 
}

// Move servo
static void servo_write_us(int servo, int pwidth) {
    pwidth = clamp(pwidth, RAW_MIN_US, RAW_MAX_US);
    uint32_t duty = (uint32_t)pwidth * SERVO_MAX_DUTY / PERIOD_US;
    ledc_set_duty(LEDC_LOW_SPEED_MODE, (ledc_channel_t)servo, duty);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, (ledc_channel_t)servo);
    //ESP_LOGI(TAG, "Wrote to servo %d", servo);
}

// Move servo to angle
static void servo_write_angle(int servo, int angle) {
    angle = clamp(angle, 0, 180);
    int lo = config.min_us[servo];
    int hi = config.max_us[servo];
    int us = lo + angle * (hi - lo) / 180; 
    //ESP_LOGI(TAG, "s%d ang%d -> %d us (lo%d hi%d)", servo, angle, us, lo, hi);
    servo_write_us(servo, us);
}

// Move to one end: hold Up/Down to move, press Cal to confirm this spot.
static int nudge_to_end(int servo, const char *name, const char *which) {
    int us = 1500;
    servo_write_us(servo, us);
    ESP_LOGI(TAG, "%s servo - %s end: hold Up/Down to move, press Calibrate to confirm", name, which);

    // Wait for the calibrate button to be released first, so the press that
    // entered calibration doesn't immediately confirm this end.
    while (pressed(BTN_CAL)) vTaskDelay(pdMS_TO_TICKS(10));
    vTaskDelay(pdMS_TO_TICKS(150));

    while (1) {
        if (pressed(BTN_CAL)) break;            // confirm setting
        if (pressed(BTN_UP))   {
            us = clampi(us + US_PER_STEP, RAW_MIN_US, RAW_MAX_US);
        }
        if (pressed(BTN_DOWN)) {
            us = clampi(us - US_PER_STEP, RAW_MIN_US, RAW_MAX_US);
        }
        servo_write_us(servo, us);
        ESP_LOGI(TAG, "  %s %s: %d us   \r", name, which, us);
        fflush(stdout);
        vTaskDelay(pdMS_TO_TICKS(50));
    }

    while (pressed(BTN_CAL)) vTaskDelay(pdMS_TO_TICKS(10));
    vTaskDelay(pdMS_TO_TICKS(150));

    ESP_LOGI(TAG, "%s %s end set at %d us", name, which, us);
    return us;
}

static void calibrate_servo(int servo, const char *name, int *out_min, int *out_max) {
    ESP_LOGI(TAG, "=== Calibrating %s servo (remove horn if it may jam) ===", name);
    int hi = nudge_to_end(servo, name, "HIGH");
    int lo = nudge_to_end(servo, name, "LOW");

    *out_max = hi - MARGIN_US;
    *out_min = lo + MARGIN_US;
    if (*out_min > *out_max) { int t = *out_min; *out_min = *out_max; *out_max = t; }

    servo_write_us(servo, 1500);
    ESP_LOGI(TAG, "%s done: min %d us, max %d us", name, *out_min, *out_max);
}

static void run_calibration(void) {
    int pan_min, pan_max, tilt_min, tilt_max;
    calibrate_servo(TILT_SERVO, "tilt", &tilt_min, &tilt_max);
    calibrate_servo(PAN_SERVO,  "pan",  &pan_min,  &pan_max);

    config.version = CONFIG_VERSION;
    config.min_us[PAN_SERVO]  = pan_min;  config.max_us[PAN_SERVO]  = pan_max;
    config.min_us[TILT_SERVO] = tilt_min; config.max_us[TILT_SERVO] = tilt_max;

    // Save new servos config
    bool ok = save_config(&config);
    ESP_LOGI(TAG, "Calibration %s: pan %d..%d, tilt %d..%d",
             ok ? "saved" : "SAVE FAILED", pan_min, pan_max, tilt_min, tilt_max);

}

static void scan_routine(void) {
    // Start routine
    ESP_LOGI(TAG, "Doing scan routine");
    while (1) {
        for (int angle = 0; angle <= 180; angle += STEP_DEG) {
            for (int s = 0; s < NUM_SERVOS; ++s) {
                servo_write_angle(s, angle);
            }
            vTaskDelay(pdMS_TO_TICKS(UPDATE_MS));
        }
        vTaskDelay(pdMS_TO_TICKS(500));

        for (int angle = 180; angle >= 0; angle -= STEP_DEG) {
            for (int s = 0; s < NUM_SERVOS; ++s) {
                servo_write_angle(s, angle);
            }
            vTaskDelay(pdMS_TO_TICKS(UPDATE_MS));
        }
        vTaskDelay(pdMS_TO_TICKS(500));
    }
}

void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(2000));   // give the USB monitor time to reconnect
    ESP_LOGI(TAG, "Program start");

    init_storage();
    ESP_LOGI(TAG, "Flash storage initialized");
    servo_init();
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
        run_calibration();
    } else {
        ESP_LOGI(TAG, "Skipping calibration - hold Calibrate at reset to run it");
    }


    // Move all servos to the center
    for (int i=0; i<NUM_SERVOS; ++i) {
        servo_write_angle(i, 90);
    }
    vTaskDelay(pdMS_TO_TICKS(1000));
    ESP_LOGI(TAG, "Reset Motors to center");

    scan_routine();
    
}

// void app_main(void)
// {
//     vTaskDelay(pdMS_TO_TICKS(5000)); 
//     servo_init();
//     ESP_LOGI(TAG, "Servos initialized");
//     vTaskDelay(pdMS_TO_TICKS(2000)); 

//     for (int i = 0; i < NUM_SERVOS; i++) servo_write_angle(i, 90);  // center
//     vTaskDelay(pdMS_TO_TICKS(1000));

//     while (1) {
//         for (int a = 0; a <= 180; a += 2) {
//             for (int i = 0; i < NUM_SERVOS; i++) servo_write_angle(i, a);
//             vTaskDelay(pdMS_TO_TICKS(15));
//         }
//         vTaskDelay(pdMS_TO_TICKS(500));

//         for (int a = 180; a >= 0; a -= 2) {
//             for (int i = 0; i < NUM_SERVOS; i++) servo_write_angle(i, a);
//             vTaskDelay(pdMS_TO_TICKS(15));
//         }
//         vTaskDelay(pdMS_TO_TICKS(500));
//     }
// }