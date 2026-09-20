#ifndef SERVO_CONFIG_H
#define SERVO_CONFIG_H

#include <stdbool.h>
#include "nvs_flash.h"
#include "nvs.h"
#include "esp_log.h"

#define CONFIG_NAMESPACE "servo"
#define CONFIG_KEY "cal"
static const int SERVO_PINS[] = {1, 2};
#define NUM_SERVOS     (sizeof(SERVO_PINS) / sizeof(SERVO_PINS[0]))
#define PAN_SERVO      0
#define TILT_SERVO     1

#define CONFIG_VERSION 1

// Everything we want to persist. Bump 'version' if you change the fields.
typedef struct {
    uint32_t version;
    int32_t  min_us[NUM_SERVOS];   // pulse for 0°   {pan, tilt}
    int32_t  max_us[NUM_SERVOS];   // pulse for 180° {pan, tilt}
    float    pan_min,  pan_max;    // safe scan angles
    float    tilt_min, tilt_max;
} servo_config;

static const servo_config DEFAULTS = {
    .version = 1,
    .min_us = {400, 400},
    .max_us = {2500, 2500},
    .pan_min = 0.0f,
    .pan_max = 180.0f,
    .tilt_min = 15.0f,
    .tilt_max = 165.0f,
};

// Initialize flash storage
static inline void init_storage(void) {
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        nvs_flash_erase();
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
}

// Load saved servo config
static inline bool load_config(servo_config *out) {
    *out = DEFAULTS;
    nvs_handle_t handle;
    // Try open flash storage return flase if error
    if (nvs_open(CONFIG_NAMESPACE, NVS_READONLY, &handle) != ESP_OK) {
        return false;
    }

    servo_config tmp;
    size_t len = sizeof(tmp);
    // Load config from flash checking size and version are correct
    bool ok = nvs_get_blob(handle, CONFIG_KEY, &tmp, &len) == ESP_OK 
        && len == sizeof(tmp) 
        && tmp.version == CONFIG_VERSION;
    nvs_close(handle);
    if (ok) {
        *out = tmp;
    }
    return ok;
}

// Save new config
static inline bool save_config(const servo_config *config) {
    nvs_handle_t handle;
    if (nvs_open(CONFIG_NAMESPACE, NVS_READWRITE, &handle) != ESP_OK) {
        return false;
    }
    bool ok = nvs_set_blob(handle, CONFIG_KEY, config, sizeof(*config)) == ESP_OK
        && nvs_commit(handle) == ESP_OK;
    nvs_close(handle);
    return ok;
}

#endif