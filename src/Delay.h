#include <chrono>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

/***
 * Delay execution of this thread by milliseconds.
 * 
 * delay_ms The amount of miiliseconds to delay
 */
inline void delay(std::chrono::milliseconds delay_ms) {
    vTaskDelay(pdMS_TO_TICKS(delay_ms.count()));
}