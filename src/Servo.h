#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/ledc.h"
#include "esp_log.h"
#include "driver/gpio.h"
#include <string>

class Servo {
    private:
        std::string name_;
        gpio_num_t pin_;
        ledc_channel_t channel_;
        int minUS_, maxUS_;
        int angle_ = -1;
        static inline int nextChannel_;

    public:

    Servo(gpio_num_t pin, std::string name) : name_(name), pin_(pin), minUS_(500), maxUS_(2500) {

    }
    esp_err_t attach(gpio_num_t pin, std::string *name);
    void setRange(int min_ud, int max_us);
    void writeUS(int us) const;
    void writeAngle(short angle);
    void stop() const;
    bool loadRange();
    bool saveRange() const;

    
};