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

} // namespace

Storage &Storage::instance() {
    static Storage storage;
    return storage;
}

Storage::Storage() {
        esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        // NVS partition cont be used. Wipe and retry.
        ESP_LOGW(TAG, "Settings partition failed: %s. Erasing partition", esp_err_to_name(err));
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err); // Stil failing...something broken
}

Storage::~Storage() {
    nvs_flash_deinit();
}

bool Storage::loadRaw(const std::string &key, void *out, size_t len) {
    NvsHandle handle(NVS_READONLY);
    if (handle.status() == ESP_ERR_NVS_NOT_FOUND) return false; // Not found
    ESP_ERROR_CHECK(handle.status());

    size_t saved = 0;
    esp_err_t err = nvs_get_blob(handle.get(), key.c_str(), nullptr, &saved);
    if (err == ESP_ERR_NVS_NOT_FOUND) {
        return false;
    }
    ESP_ERROR_CHECK(err);

    if (saved != len) {
        ESP_LOGW(TAG, "Setting '%s' has a incompatible type, ignoring setting", key.c_str());
        return false;
    }
    ESP_ERROR_CHECK(nvs_get_blob(handle.get(), key.c_str(), out, &saved));
    return true;
}

void Storage::saveRaw(const std::string &key, const void *out, size_t len) {
    NvsHandle handle(NVS_READWRITE);
    ESP_ERROR_CHECK(handle.status());
    ESP_ERROR_CHECK(nvs_set_blob(handle.get(), key.c_str(), out, len));
    ESP_ERROR_CHECK(nvs_commit(handle.get()));
}

void Storage::erase(const std::string &key) {
    NvsHandle handle(NVS_READWRITE);
    ESP_ERROR_CHECK(handle.status());
    esp_err_t err = nvs_erase_key(handle.get(), key.c_str());
    if (err == ESP_ERR_NVS_NOT_FOUND) return;
    ESP_ERROR_CHECK(err);
    ESP_ERROR_CHECK(nvs_commit(handle.get()));
}

Partition Storage::getPartition(const std::string &label) {
    const esp_partition_t *part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, ESP_PARTITION_SUBTYPE_ANY, label.c_str());
 
    if (!part) {
        ESP_LOGE(TAG, "No data partition named '%s' in partitions.csv", label.c_str());
        ESP_ERROR_CHECK(ESP_ERR_NOT_FOUND);
    }
    if (part->subtype == ESP_PARTITION_SUBTYPE_DATA_NVS) {
        // Raw writes would corrupt the settings; use Storage::save/load for nvs
        ESP_LOGE(TAG, "'%s' is the settings partition, use save/load instead", label.c_str());
        ESP_ERROR_CHECK(ESP_ERR_INVALID_ARG);
    }
    return Partition(part);
}

std::string_view Partition::label() const { return part_->label; }

size_t Partition::size() const { return part_->size; }
 
void Partition::read(size_t offset, std::span<uint8_t> out) const {
    // Out-of-range reads return an error and stop here
    ESP_ERROR_CHECK(esp_partition_read(part_, offset, out.data(), out.size()));
}
 
/**
 * It makes the partition's flash contents look like ordinary 
 * read-only memory, and hands you a std::span covering the whole partition
 */
std::span<const uint8_t> Partition::map() {
    if (mapped_.empty()) {
        const void *ptr = nullptr;
        esp_partition_mmap_handle_t handle;
        ESP_ERROR_CHECK(esp_partition_mmap(part_, 0, part_->size,
                                           ESP_PARTITION_MMAP_DATA, &ptr, &handle));
        mapped_ = std::span<const uint8_t>(static_cast<const uint8_t *>(ptr),
                                   static_cast<size_t>(part_->size));
    }
    return mapped_;
}

void Partition::replace(std::span<const uint8_t> data) const {
    // Error if trying to save data bigger than partition
    if (data.size() > part_->size) {
        ESP_LOGE(TAG, "%zu bytes wont fit in partition %s of size %zu bytes", 
            data.size(), part_->label, static_cast<size_t>(part_->size));
        ESP_ERROR_CHECK(ESP_ERR_INVALID_SIZE);
    }
    size_t eraseLen = (data.size() + kSectorSize - 1) / kSectorSize * kSectorSize;
    ESP_ERROR_CHECK(esp_partition_erase_range(part_, 0, eraseLen));
    ESP_ERROR_CHECK(esp_partition_write(part_, 0, data.data(), data.size()));
}


