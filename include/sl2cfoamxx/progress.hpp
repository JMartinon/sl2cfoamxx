// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "sl2cfoamxx/types.hpp"

#include <cstdint>
#include <functional>
#include <string_view>

namespace sl2cfoamxx {

// Machine-readable progress events. These are independent of terminal
// rendering so language bindings can drive their own native progress bars.
enum class ProgressOperation : std::uint8_t {
    coherent_amplitude,
    full_tensor
};

enum class ProgressPhase : std::uint8_t {
    contraction,
    // A low-memory full-tensor pass that alternates 6j generation with the
    // contraction of each streamed slab, or performs a fully streamed sequence
    // of support, booster, 6j, contraction, and output work.
    streamed_computation
};

enum class ProgressStatus : std::uint8_t {
    started,
    running,
    restarted,
    completed,
    stopped
};

struct ProgressEvent {
    // Status is the lifecycle of this phase, not necessarily of the entire
    // public call. In particular, full-tensor publication may follow a
    // completed contraction phase.
    ProgressOperation operation = ProgressOperation::full_tensor;
    ProgressPhase phase = ProgressPhase::contraction;
    Backend backend = Backend::CPU;
    ProgressStatus status = ProgressStatus::running;
    std::uint32_t attempt = 1;
    std::uint64_t completed = 0;
    std::uint64_t total = 0;
    double elapsed_seconds = 0.0;
    double eta_seconds = 0.0;
    bool eta_available = false;
};

// Callbacks are serialized for a computation, but may run on either the
// calling thread or the library's progress-reporting thread. They should be
// short and thread-safe. Exceptions are ignored because progress reporting
// must never change a numerical result.
using ProgressCallback = std::function<void(const ProgressEvent&)>;

[[nodiscard]] constexpr std::string_view to_string(
    ProgressOperation operation) noexcept
{
    switch (operation) {
    case ProgressOperation::coherent_amplitude:
        return "coherent_amplitude";
    case ProgressOperation::full_tensor:
        return "full_tensor";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view to_string(
    ProgressPhase phase) noexcept
{
    switch (phase) {
    case ProgressPhase::contraction:
        return "contraction";
    case ProgressPhase::streamed_computation:
        return "streamed_computation";
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view to_string(
    ProgressStatus status) noexcept
{
    switch (status) {
    case ProgressStatus::started:
        return "started";
    case ProgressStatus::running:
        return "running";
    case ProgressStatus::restarted:
        return "restarted";
    case ProgressStatus::completed:
        return "completed";
    case ProgressStatus::stopped:
        return "stopped";
    }
    return "unknown";
}

} // namespace sl2cfoamxx
