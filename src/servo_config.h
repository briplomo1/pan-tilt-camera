#ifndef SERVO_CONFIG_H
#define SERVO_CONFIG_H

#include <stdint.h>

// Key the calibration is stored under (see Storage)
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
    .tilt_min = 0.0f,
    .tilt_max = 180.0f,
};

#endif
