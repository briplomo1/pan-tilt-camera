#pragma once

#include <cstddef>
#include <cstdint>

// Free heap in each region at one moment
struct HeapSnapshot {
    size_t freeInternal = 0;
    size_t freePsram = 0;
};

struct MemoryUsage {
    int32_t internalBytes = 0;      // taken since the probe started (negative: freed)
    int32_t psramBytes = 0;
    uint32_t minFreeInternal = 0;   // lowest free memory while the probe was running
    uint32_t minFreePsram = 0;
};

/**
 * Measures how much heap a piece of code takes, including its peak, e.g.
 *     MemoryProbe probe;
 *     detector = registry.create(id);
 *     MemoryUsage used = probe.usage();
 * The heap's low-water-mark monitor is global, so only one probe can run at a time; a second
 * one still reports usage but not peaks.
 */
class MemoryProbe {

private:
    HeapSnapshot start_;
    bool monitoring_ = false;

public:
    MemoryProbe();
    ~MemoryProbe();
    MemoryProbe(const MemoryProbe &) = delete;
    MemoryProbe &operator=(const MemoryProbe &) = delete;
    MemoryProbe(MemoryProbe &&) = delete;
    MemoryProbe &operator=(MemoryProbe &&) = delete;

    [[nodiscard]] MemoryUsage usage() const;

    [[nodiscard]] static HeapSnapshot snapshot();
};
