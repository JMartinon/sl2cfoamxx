// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <limits>

namespace b4support::host_memory {

struct Snapshot {
    std::uint64_t physical_total_bytes = 0;
    std::uint64_t system_available_bytes = 0;
    std::uint64_t process_rss_bytes = 0;
    std::uint64_t cgroup_limit_bytes = 0;
    std::uint64_t cgroup_headroom_bytes = 0;
    std::uint64_t slurm_limit_bytes = 0;
    std::uint64_t numa_affinity_capacity_bytes = 0;
    std::uint64_t numa_affinity_available_bytes = 0;
    bool has_system_available = false;
    bool has_cgroup_limit = false;
    bool has_cgroup_headroom = false;
    bool has_slurm_limit = false;
    bool has_numa_affinity_capacity = false;
    bool has_numa_affinity_available = false;
};

inline std::uint64_t sat_add(std::uint64_t a, std::uint64_t b) noexcept
{
    const std::uint64_t m = std::numeric_limits<std::uint64_t>::max();
    return m - a < b ? m : a + b;
}

inline std::uint64_t sat_mul(std::uint64_t a, std::uint64_t b) noexcept
{
    const std::uint64_t m = std::numeric_limits<std::uint64_t>::max();
    return a != 0 && b > m / a ? m : a * b;
}

Snapshot detect_snapshot() noexcept;

std::uint64_t hard_capacity_bytes(const Snapshot& snapshot) noexcept;

std::uint64_t additional_headroom_bytes(const Snapshot& snapshot) noexcept;

std::uint64_t reachable_process_peak_bytes(const Snapshot& snapshot) noexcept;

inline bool has_hard_capacity(const Snapshot& snapshot) noexcept
{
    return snapshot.physical_total_bytes != 0 ||
           snapshot.has_cgroup_limit || snapshot.has_slurm_limit ||
           snapshot.has_numa_affinity_capacity;
}

inline bool has_headroom_constraint(const Snapshot& snapshot) noexcept
{
    return snapshot.has_system_available || snapshot.has_cgroup_headroom ||
           snapshot.has_slurm_limit || snapshot.has_numa_affinity_available;
}

inline bool has_any_probe(const Snapshot& snapshot) noexcept
{
    return has_hard_capacity(snapshot) || has_headroom_constraint(snapshot);
}

std::uint64_t automatic_budget_bytes(const Snapshot& snapshot,
                                            double ram_fraction) noexcept;

} // namespace b4support::host_memory
