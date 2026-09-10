// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sl2cfoam::gpu::production {

// -----------------------------------------------------------------------------
// Common configuration shared by the coherent-amplitude and full-tensor routes.
// -----------------------------------------------------------------------------
//
// This header contains only settings that remain meaningful for the fixed
// production algorithms:
//
//   * which GPUs may be used;
//   * how much VRAM the planner may consume;
//   * whether profiling counters should be collected.
//
// Backend selection is deliberately absent.  Each public production entry point
// below represents exactly one validated algorithm and applies only memory-driven
// fallbacks when required.

struct DeviceMemoryPolicy {
    // Bit i selects CUDA device i.  A zero mask means "all visible devices".
    std::uint64_t device_mask = 0;

    // Fraction of currently free VRAM that may be reserved by one contraction.
    // A value below one leaves room for the CUDA context, library workspaces and
    // unrelated allocations owned by the application.
    double usable_vram_fraction = 0.90;

    // Bytes held back on every selected device after applying the fraction.
    std::uint64_t reserve_bytes = 512ull * 1024ull * 1024ull;

    // Optional absolute per-device cap.  Zero disables the cap.
    std::uint64_t maximum_device_bytes = 0;
};

struct ProfilingPolicy {
    // CUDA-event profiling introduces synchronizations in a few code paths.
    // Keep it disabled for production timings and enable it only for diagnostics.
    bool enabled = false;

    // NVTX ranges are useful with Nsight Systems and have negligible overhead,
    // but they are optional because not every build links NVTX.
    bool nvtx = false;
};

// A compact record of a failed fallback attempt.  It is returned to callers so
// an out-of-memory fallback never silently changes the execution strategy.
struct FallbackFailure {
    std::string configuration;
    std::string reason;
};

class FallbackLog {
public:
    void add(std::string configuration, std::string reason)
    {
        failures_.push_back(
            FallbackFailure{std::move(configuration), std::move(reason)});
    }

    [[nodiscard]] bool empty() const noexcept { return failures_.empty(); }

    [[nodiscard]] const std::vector<FallbackFailure>& failures() const noexcept
    {
        return failures_;
    }

    [[nodiscard]] std::string message(const char* heading) const
    {
        std::ostringstream out;
        out << heading;
        for (const FallbackFailure& failure : failures_) {
            out << "\n  - " << failure.configuration << ": " << failure.reason;
        }
        return out.str();
    }

private:
    std::vector<FallbackFailure> failures_;
};

// Checked integer helpers are kept in the public implementation headers because
// memory planning uses products of six or more tensor extents.  Overflow here is
// not an academic concern: it turns into a small allocation followed by a large
// kernel write, which is an unusually efficient way to ruin an evening.
[[nodiscard]] inline std::uint64_t checked_add(
    std::uint64_t a,
    std::uint64_t b,
    const char* what)
{
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw std::overflow_error(std::string("overflow while computing ") + what);
    }
    return a + b;
}

[[nodiscard]] inline std::uint64_t checked_multiply(
    std::uint64_t a,
    std::uint64_t b,
    const char* what)
{
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::overflow_error(std::string("overflow while computing ") + what);
    }
    return a * b;
}

template <class... Values>
[[nodiscard]] inline std::uint64_t checked_product(
    const char* what,
    Values... values)
{
    std::uint64_t product = 1;
    ((product = checked_multiply(
          product, static_cast<std::uint64_t>(values), what)), ...);
    return product;
}

[[nodiscard]] inline std::uint32_t ceil_divide_u32(
    std::uint32_t numerator,
    std::uint32_t denominator)
{
    if (denominator == 0) {
        throw std::invalid_argument("ceil_divide_u32 received a zero denominator");
    }
    return numerator / denominator + (numerator % denominator != 0 ? 1u : 0u);
}

[[nodiscard]] inline bool valid_memory_policy(
    const DeviceMemoryPolicy& policy) noexcept
{
    return policy.usable_vram_fraction > 0.0 &&
           policy.usable_vram_fraction <= 1.0;
}

inline void validate_memory_policy(const DeviceMemoryPolicy& policy)
{
    if (!valid_memory_policy(policy)) {
        throw std::invalid_argument(
            "usable_vram_fraction must lie in the interval (0, 1]");
    }
}

} // namespace sl2cfoam::gpu::production
