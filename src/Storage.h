#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <optional>
#include <span>
#include "esp_err.h"
#include "nvs_flash.h"
#include "esp_log.h"
#include <string>
#include "esp_partition.h"


class Partition {

private:
    const esp_partition_t *part_; // Non owning
    std::span<const uint8_t> mapped_;
    friend class Storage;
    explicit Partition(const esp_partition_t *part) : part_(part) {}

public:
    [[nodiscard]] std::string_view label() const;

    [[nodiscard]] size_t size() const;

    void read(size_t offset, std::span<uint8_t> out) const;

    [[nodiscard]] std::span<const uint8_t> map();

    void replace(std::span<const uint8_t> data) const;
};


class Storage {

private:

    static void init();

    [[nodiscard]] static bool loadRaw(const std::string &key, void *out, size_t len);

    static void saveRaw(const std::string &key, const void *out, size_t len);

    Storage();

    ~Storage();

public:

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
