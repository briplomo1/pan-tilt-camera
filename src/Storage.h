#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include "esp_err.h"
#include "nvs_flash.h"
#include "esp_log.h"
#include "esp_partition.h"


/**
 * Part of a flash partition mapped into the address space, read through the cache like
 * ordinary read-only memory. Unmapped when it goes out of scope.
 */
class MappedRegion {

private:
    esp_partition_mmap_handle_t handle_ = 0;
    std::span<const uint8_t> data_;     // empty when nothing is mapped
    friend class Partition;
    MappedRegion(esp_partition_mmap_handle_t handle, std::span<const uint8_t> data)
        : handle_(handle), data_(data) {}

public:
    MappedRegion() = default;
    ~MappedRegion() { reset(); }

    MappedRegion(const MappedRegion &) = delete;
    MappedRegion &operator=(const MappedRegion &) = delete;
    MappedRegion(MappedRegion &&other) noexcept
        : handle_(std::exchange(other.handle_, 0)), data_(std::exchange(other.data_, {})) {}
    MappedRegion &operator=(MappedRegion &&other) noexcept {
        if (this != &other) {
            reset();
            handle_ = std::exchange(other.handle_, 0);
            data_ = std::exchange(other.data_, {});
        }
        return *this;
    }

    // Unmap now instead of at destruction
    void reset();

    [[nodiscard]] std::span<const uint8_t> data() const { return data_; }
    explicit operator bool() const { return !data_.empty(); }
};


class Partition {

private:
    const esp_partition_t *part_; // Non owning
    friend class Storage;
    explicit Partition(const esp_partition_t *part) : part_(part) {}

public:
    [[nodiscard]] std::string_view label() const;

    [[nodiscard]] size_t size() const;

    void read(size_t offset, std::span<uint8_t> out) const;

    // Map [offset, offset + size) into memory. Stops if the range is outside the partition.
    [[nodiscard]] MappedRegion map(size_t offset, size_t size) const;

    // Map the whole partition
    [[nodiscard]] MappedRegion map() const { return map(0, size()); }

    void replace(std::span<const uint8_t> data) const;
};


class Storage {

private:

    [[nodiscard]] static bool loadRaw(const std::string &key, void *out, size_t len);

    static void saveRaw(const std::string &key, const void *out, size_t len);

    Storage();

    ~Storage();

public:

    // NVS is initialized on first use; every static call below goes through here
    [[nodiscard]] static Storage& instance();

    Storage(const Storage &) = delete;
    Storage &operator=(const Storage &) = delete;
    Storage(Storage &&) = delete;
    Storage &operator=(Storage&&) = delete;

    /***
     *  Get an item from storage by its key. Returns optional. Will be emoty if key not found.
     */
    template<typename T>
    [[nodiscard]] static std::optional<T> load(const std::string &key) {
        static_assert(std::is_trivially_copyable_v<T>, "Store and read plain structs only");
        T value{};
        if (!loadRaw(key, &value, sizeof(T))) return std::nullopt;
        return value;
    }

    /**
     * Save an item by its key. Can only read and write plain structs
     */
    template<typename T>
    static void save(const std::string &key, const T &value) {
        static_assert(std::is_trivially_copyable_v<T>, "Store and read plain structs only");
        saveRaw(key, &value, sizeof(T));
    }

    // Erase an item by its key from storage
    static void erase(const std::string &key);

    /**
     * Find a data partition by its label and return a handle to it.
     * Will stop if partition doesnt exist or is the NVS partition
     */
    [[nodiscard]] static Partition getPartition(const std::string &label);

};
