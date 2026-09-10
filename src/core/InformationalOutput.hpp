// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <atomic>
#include <mutex>

namespace sl2cfoamxx::detail {

// Diagnostics and live progress displays share one output lock. This keeps a
// caller-supplied stream safe when independent API computations run on
// different application threads.
inline std::mutex& informational_output_mutex() noexcept
{
    static std::mutex mutex;
    return mutex;
}

// The default terminal can host only one carriage-return-updated line. A
// concurrent progress reporter falls back to newline-delimited records.
inline std::atomic<bool>& interactive_progress_active_flag() noexcept
{
    static std::atomic<bool> active{false};
    return active;
}

inline bool try_acquire_interactive_progress() noexcept
{
    bool expected = false;
    return interactive_progress_active_flag().compare_exchange_strong(
        expected,
        true,
        std::memory_order_acq_rel,
        std::memory_order_relaxed);
}

inline void release_interactive_progress() noexcept
{
    interactive_progress_active_flag().store(
        false, std::memory_order_release);
}

inline bool interactive_progress_is_active() noexcept
{
    return interactive_progress_active_flag().load(std::memory_order_acquire);
}

} // namespace sl2cfoamxx::detail
