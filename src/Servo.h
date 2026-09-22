#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/ledc.h"
#include "esp_log.h"
#include "driver/gpio.h"
#include <string>

class Servo {

    private:
    gpio_num_t pin_;
    ledc_channel_t channel_;
    static inline bool timerReady_ = false;
    static constexpr int minUs_ = 400;
    static constexpr int maxUs_ = 2600;
    static inline uint32_t usedChannels_ = 0;

    public:

    explicit Servo(gpio_num_t pin) : pin_(pin) {}
    ~Servo();
    Servo(const Servo &) = delete;
    Servo &operator=(const Servo &) = delete;
    Servo(Servo &&) = delete;
    Servo &operator(Servo &&) = delete;

    void setRange(int min_ud, int max_us);
    void writeUS(int us) const;
    void writeAngle(short angle);
    void stop() const;

    [[nodiscard]] gpio_num_t pin() const { return pin_; }
    [[nodiscard]] int minUs() const { return minUs_; }
    [[nodiscard]] int maxUs() const { return maxUs_; }
    
};