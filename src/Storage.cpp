#include "Storage.h"



namespace {
constexpr const char *TAG = "storage";
constexpr const char *kNamespace = "app"; // All app settings share one NVS namespace
constexpr size_t kSectorSize = 4096; // Flash is in 4kb sectors

class NvsHandle {
    private:
    nvs_handle_t h_ = 0;
    esp_err_t status_;

    public:
    explicit NvsHandle(nvs_open_mode_t mode) : status_(nvs_open(kNamespace, mode, &h_)){}

    // Close handle on destruction
    ~NvsHandle() { if(status_ == ESP_OK) nvs_close(h_); }

    // Delete copy assignment and constructor
    NvsHandle(const NvsHandle &) = delete;
    NvsHandle &operator=(const NvsHandle &) = delete;

    esp_err_t status() const { return status_; }
    nvs_handle_t get() const { return h_; }
};

void Storage::init() {
    esp_err_t err = nvs_flash_init()
}

}