#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <optional>
#include <span>
#include "esp_err.h"
#include "nvs_flash.h"
#include "esp_log.h"
#include <string>


class Partition {
    private:
    const esp_partition_t *part_; // Non owning
    std::span<const uint8_t> mapped_;
    friend class Storage;
    explicit Partition(const esp_partition_t *part) : part_(part) {}

    public:
    [[nodiscard]] std::string_view label() const;
    [[nodiscard]] size_t size() const;
};


class Storage {
    private:
    static void init();

    public:
    /***
     *  Get an item from storage by its key. Returns optional. Will be emoty if key not found.
     */
    template<typename T>
    [[nodiscard]] static std::optional<T> load(const std::string &key) {
        static_assert(std::is_trivially_copyable_v<T>, "Store and read plain structs only");
        T value{};
        if (!loadBlob(key, &value, sizeof(T))) return std::nullopt;
        return value;
    }

    /**
     * Save an item by its key. Can only read and write plain structs
     */
    template<typename T>
    static void save(const std::string &key, const T &value) {
        static_assert(std::is_trivially_copyable_v<T>, "Store and read plain structs only");
        return
    }

    // Erase a item by its key
    static void erase(const std::string &key);

    /**
     * Find a data partition by its label
     * Will stop if partition doesnt exist or is the NVS partition
     */
    [[nodiscard]] static Partition partition(const std::string &label);

};
