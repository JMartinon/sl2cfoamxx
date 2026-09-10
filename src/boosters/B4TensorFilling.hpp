// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later


#pragma once

#include <vector>
#include <array>
#include <complex>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include "core/OpenMp.hpp"
#include <limits>
#include <cstdlib>
#include <thread>
#include <fstream>
#include <sstream>
#include <set>
#include <string>
#include <utility>
#include <span>
#include <stdexcept>

#include "sl2cfoamxx/types.hpp"


#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#include "boosters/B4_function.hpp"
#include "boosters/B4Quadrature.hpp"
#include "core/Failure.hpp"
#include "core/LowLevelTuning.hpp"
#include "core/tensors.hpp"
#include "core/utils.hpp"


//===================
// Simplicity constraint
//===================

inline double simplicity_constraint(
    const double gamma,
    const int Dspin,
    const sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j)
{
    switch (prescription) {
    case sl2cfoamxx::SimplicityConstraint::gamma_j:
        return gamma * (Dspin * 0.5);
    case sl2cfoamxx::SimplicityConstraint::gamma_j_p1:
        return gamma * (Dspin * 0.5 + 1.0);
    }
    throw std::invalid_argument("unknown simplicity-constraint prescription");
}


// ============================================================================
// OpenMP runtime policy
// ============================================================================
//
// Goal:
//   Do not force users to manually discover that "all logical CPUs" is often
//   slower than using a sane number of physical/performance cores.
//
// Logic:
//   1. If B4_NUM_THREADS is set, use it as an upper bound.
//   2. Else if OMP_NUM_THREADS is set, use the current OpenMP runtime ceiling.
//      Configuration or omp_set_num_threads may have replaced its initial value.
//   3. Else choose a conservative automatic thread count based on:
//        - estimated physical/performance cores
//        - number of independent jobs
//
// This avoids oversubscribing simultaneous-multithreading contexts when doing
// so would reduce cache locality and throughput.

inline int b4_parse_positive_env_int(const char* name) {
    const char* s = std::getenv(name);

    if (s == nullptr || *s == '\0') {
        return 0;
    }

    char* end = nullptr;
    long v = std::strtol(s, &end, 10);

    if (end == s || v <= 0 || v > 4096) {
        return 0;
    }

    return static_cast<int>(v);
}

#if defined(__linux__)

inline int b4_detect_physical_cores_linux() {
    std::ifstream in("/proc/cpuinfo");

    if (!in) {
        return 0;
    }

    std::set<std::pair<int, int>> cores;

    std::string line;
    int physical_id = 0;
    int core_id = -1;
    bool inside_cpu_block = false;

    auto flush_cpu = [&]() {
        if (inside_cpu_block && core_id >= 0) {
            cores.insert({physical_id, core_id});
        }

        physical_id = 0;
        core_id = -1;
        inside_cpu_block = false;
    };

    while (std::getline(in, line)) {
        if (line.rfind("processor", 0) == 0) {
            flush_cpu();
            inside_cpu_block = true;
        } else if (line.rfind("physical id", 0) == 0) {
            const std::size_t pos = line.find(':');
            if (pos != std::string::npos) {
                physical_id = std::atoi(line.c_str() + pos + 1);
            }
        } else if (line.rfind("core id", 0) == 0) {
            const std::size_t pos = line.find(':');
            if (pos != std::string::npos) {
                core_id = std::atoi(line.c_str() + pos + 1);
            }
        }
    }

    flush_cpu();

    return static_cast<int>(cores.size());
}

#else

inline int b4_detect_physical_cores_linux() {
    return 0;
}

#endif

#if defined(__APPLE__)

inline int b4_sysctl_int(const char* name) {
    int value = 0;
    std::size_t size = sizeof(value);

    if (sysctlbyname(name, &value, &size, nullptr, 0) == 0 && value > 0) {
        return value;
    }

    return 0;
}

inline int b4_detect_physical_cores_apple() {
    int physical_cores = b4_sysctl_int("hw.physicalcpu");
    if (physical_cores > 0) return physical_cores;
    return 0;
}

#else

inline int b4_detect_physical_cores_apple() {
    return 0;
}

#endif

inline int b4_detect_reasonable_core_count() {
#if defined(__APPLE__)
    const int apple_cores = b4_detect_physical_cores_apple();

    if (apple_cores > 0) {
        return apple_cores;
    }
#endif

#if defined(__linux__)
    const int linux_cores = b4_detect_physical_cores_linux();

    if (linux_cores > 0) {
        return linux_cores;
    }
#endif

    // Last fallback. This usually returns logical CPUs, not physical cores.
    // Better than returning zero, worse than civilization.
    const unsigned hc = std::thread::hardware_concurrency();

    if (hc > 0) {
        return static_cast<int>(hc);
    }

    return 1;
}

inline int b4_choose_num_threads_for_group_work(
    int group_count,
    std::size_t total_group_work)
{
    const int runtime_cap = std::max(1, omp_get_max_threads());

    // A user request is a cap, not permission to create threads that can never
    // receive a nonempty group.  Each worker owns a sizeable recurrence
    // workspace, so idle workers are particularly expensive here.
    if (const int t = b4_parse_positive_env_int("B4_NUM_THREADS"); t > 0) {
        return std::max(
            1, std::min({t, runtime_cap, std::max(1, group_count)}));
    }

    if (b4_parse_positive_env_int("OMP_NUM_THREADS") > 0) {
        // The environment initializes OpenMP; it is not a permanent cap after
        // the public configuration or the caller updates the runtime ceiling.
        return std::max(
            1, std::min(runtime_cap, std::max(1, group_count)));
    }

    const int cores =
        std::max(1, b4_detect_reasonable_core_count());

    // Conservative upper bound: no point using more threads than groups.
    //
    // Cap the thread count at the number of groups so every launched worker has
    // at least one schedulable unit.
    int tmax = std::min(
        {cores, runtime_cap, std::max(1, group_count)});

    // Work-based heuristic:
    // Require enough inner work per thread to amortize OpenMP scheduling,
    // DP workspace pressure, and cache disruption.
    //
    // This threshold is intentionally conservative. Tune this constant later
    // if a wider benchmark says otherwise.
    constexpr std::size_t work_per_thread_target = 15000;

    int t_work =
        static_cast<int>(
            (total_group_work + work_per_thread_target - 1) /
            work_per_thread_target
        );

    t_work = std::max(1, t_work);

    return std::max(1, std::min(tmax, t_work));
}

// -----------------------------------------------------------------------------
// Diagnostics switches
// -----------------------------------------------------------------------------
//
// Keep these as compile-time constants, not runtime booleans.
//
// If B4_ENABLE_TIMING == 0, the compiler completely removes the omp_get_wtime()
// calls and timing accumulation, avoiding measurement overhead in hot loops.
//
// Enable with:
//   -DB4_ENABLE_TIMING=1
//
// Disable, default:
//   -DB4_ENABLE_TIMING=0

#ifndef B4_ENABLE_TIMING
#define B4_ENABLE_TIMING 0
#endif

#ifndef B4_ENABLE_DIAGNOSTICS
#define B4_ENABLE_DIAGNOSTICS 0
#endif

// Diagnostic timing container.
// One instance is used per OpenMP thread.
//
// alignas(64) avoids false sharing: without it, different threads could update
// timing counters that live on the same cache line, which would poison the
// measurement with cache-coherency traffic.
struct ThreadTimingZ {
    double kappa12_seconds  = 0.0;
    double kappa34_seconds  = 0.0;
    double contract_seconds = 0.0;
    double clear_seconds    = 0.0;

    std::uint64_t tiles = 0;
};

#ifndef B4_ENABLE_WORK_DIAGNOSTICS
#define B4_ENABLE_WORK_DIAGNOSTICS B4_ENABLE_TIMING
#endif

inline std::uint64_t b4_fnv1a_hash_int_span(std::span<const int> vals) noexcept {
    std::uint64_t h = 1469598103934665603ull;
    for (int v : vals) {
        h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(v));
        h *= 1099511628211ull;
    }
    return h;
}

inline void b4_print_int_span_stats(const char* name, std::span<const int> vals) {
    std::cerr << "  " << name << ": size=" << vals.size();

    if (vals.empty()) {
        std::cerr << ", empty\n";
        return;
    }

    const auto [mn_it, mx_it] = std::minmax_element(vals.begin(), vals.end());

    std::cerr << ", first=" << vals.front()
              << ", last=" << vals.back()
              << ", min=" << *mn_it
              << ", max=" << *mx_it
              << ", hash=" << b4_fnv1a_hash_int_span(vals)
              << "\n";
}
using C = std::complex<double>;

constexpr double B4_PI  = 3.141592653589793238462643383279502884;
constexpr double B4_LN2 = 0.693147180559945309417232121458176568;

constexpr int V_LEN = GL64_SPLIT_N;

// Rho-lane tile used by the vectorized/tiled DP executor.
// Try 4, 8, 16. On many x86 CPUs, 8 is a good first choice.
static_assert(V_LEN % 4 == 0 && V_LEN % 8 == 0 && V_LEN % 16 == 0,
              "supported rho tiles must divide the quadrature length");

// Scratch values reused by both rank-4 and rank-5 harvesters. Keeping each
// component cache-line aligned gives the vectorized loops independent streams.
template <int RhoTile>
struct RhoComplexTileZ {
    alignas(64) double R[RhoTile];
    alignas(64) double I[RhoTile];
};

inline int b4_resolve_rho_tile(std::uint32_t requested)
{
    if (!sl2cfoamxx::detail::is_supported_booster_rho_tile(requested)) {
        throw std::invalid_argument("booster rho tile must be 0, 4, 8, or 16");
    }
    return static_cast<int>(
        sl2cfoamxx::detail::effective_booster_rho_tile(requested));
}

#if defined(__GNUC__) || defined(__clang__)
#define B4_NOINLINE __attribute__((noinline))
#else
#define B4_NOINLINE
#endif

// ----------------------------------------------------------------------------
// SMALL HELPERS
// ----------------------------------------------------------------------------



template <int Rank>
inline std::size_t flat_tensor_index(
    std::array<std::size_t,3> Dim_ls,
    std::size_t a_size,
    int d1,
    int d2,
    int d3,
    std::size_t a_idx,
    std::size_t b_idx)
{
    if constexpr (Rank == 5) {
        return
            d1 +
        Dim_ls[0] *
            (
                d2 +
             Dim_ls[1] *
                (
                    d3 +
                 Dim_ls[2] *
                    (
                        a_idx +
                        a_size * b_idx
                    )
                )
            );
    } else {
        return
            d1 +
        Dim_ls[0] *
            (
                d2 +
             Dim_ls[1] *
                (
                    d3 +
                 Dim_ls[2] * b_idx
                )
            );
    }
}

inline double rup_val(int Dnu, double rho, int DJ) {
    if (DJ == 0) return 0.0;

    double J  = 0.5 * DJ;
    double nu = 0.5 * Dnu;

    double val = J * J - nu * nu;

    return (val > 0.0)
        ? (2.0 / J) * std::sqrt(val * (J * J + rho * rho))
        : 0.0;
}

inline int ws_flat_idx_precomputed(
    int DJ1,
    int DJ2,
    int DJ,
    int DJ1_min,
    int DJ2_min,
    int DJ_min,
    int stride1,
    int stride2) noexcept
{
    return
        ((DJ1 - DJ1_min) / 2 + 2) * stride1 +
        ((DJ2 - DJ2_min) / 2 + 2) * stride2 +
        ((DJ  - DJ_min ) / 2 + 2);
}

// ----------------------------------------------------------------------------
// GEOMETRY CACHE
// ----------------------------------------------------------------------------

struct PureGeomEntryZ {
    uint8_t branch;
    double inv_lhs;
    double g_t1;
    double g_t2;
    double g_t3;
    double g_t4;
};

struct PureGeomCacheZ {
    std::vector<PureGeomEntryZ> entries;

    int dim1 = 0;
    int dim2 = 0;
    int dim3 = 0;

    int DJ1_min = 0;
    int DJ2_min = 0;
    int DJ_min  = 0;

    void init(int Dl1_max, int Dl2_max, int DmaxJ, int Dnu1, int Dnu2) {
        // J1 and J2 are physical only for J1 >= nu1 and J2 >= nu2.  Starting
        // these dense axes at parity (zero or one) used to retain a very large
        // unreachable prefix for high spins.  The recurrence workspace keeps
        // its own two-cell lower halo; the geometry table never needs entries
        // below the physical minima.
        DJ1_min = Dnu1;
        DJ2_min = Dnu2;
        DJ_min  = DmaxJ & 1;

        dim1 = (Dl1_max - DJ1_min) / 2 + 1;
        dim2 = (Dl2_max - DJ2_min) / 2 + 1;
        dim3 = (DmaxJ   - DJ_min ) / 2 + 1;

        entries.assign(
            static_cast<std::size_t>(dim1) *
            static_cast<std::size_t>(dim2) *
            static_cast<std::size_t>(dim3),
            PureGeomEntryZ{0, 0.0, 0.0, 0.0, 0.0, 0.0}
        );

#pragma omp parallel for schedule(static, 1)
        for (int DJ1 = DJ1_min; DJ1 <= Dl1_max; DJ1 += 2) {
            double J1  = 0.5 * DJ1;
            double jj1 = J1 - 1.0;

            bool canJ1 = DJ1 > Dnu1;

            for (int DJ2 = DJ2_min; DJ2 <= Dl2_max; DJ2 += 2) {
                double J2  = 0.5 * DJ2;
                double jj2 = J2 - 1.0;

                bool canJ2 = DJ2 > Dnu2;

                int ow_idx =
                    ((DJ1 - DJ1_min) / 2) * dim2 +
                    ((DJ2 - DJ2_min) / 2);

                int start_DJ = std::max(0, iabs_int(DJ1 - DJ2));
                if ((start_DJ & 1) != DJ_min) ++start_DJ;

                for (int DJ = start_DJ; DJ <= std::min(DmaxJ, DJ1 + DJ2); DJ += 2) {
                    double J  = 0.5 * DJ;
                    double jj = J - 1.0;

                    int idx = ow_idx * dim3 + (DJ - DJ_min) / 2;

                    double common = (J + J1 + J2 + 1.0) * (J + J1 + J2);

                    double radJ1 =
                        common *
                        (-J + J1 + J2) *
                        ( J + J1 - J2);

                    double radJ2 =
                        common *
                        ( J - J1 + J2) *
                        (-J + J1 + J2);

                    double radJ =
                        common *
                        (J + J1 - J2) *
                        (J - J1 + J2);

                    PureGeomEntryZ e{0, 0.0, 0.0, 0.0, 0.0, 0.0};

                    if (canJ1 && radJ1 > 0.0) {
                        e.branch = 1;

                        e.inv_lhs =
                            std::sqrt((2.0 * jj1 + 1.0) * (2.0 * jj1 + 3.0)) /
                            std::sqrt(radJ1);

                        e.g_t1 =
                            std::sqrt((J - jj1 + J2) * (J + jj1 + J2 + 1.0));

                        e.g_t2 =
                            (DJ >= 2)
                                ? ((jj1 + 1.0) / J) *
                                  std::sqrt((J + jj1 - J2) * (-J + jj1 + J2 + 1.0))
                                : 0.0;

                        e.g_t3 =
                            (DJ1 >= 4)
                                ? ((jj1 + 1.0) / jj1) *
                                  std::sqrt(
                                      (-J + jj1 + J2) *
                                      ( J + jj1 - J2) *
                                      ( J - jj1 + J2) *
                                      ( J - jj1 + J2 + 1.0)
                                  ) /
                                  std::sqrt((2.0 * jj1 + 1.0) * (2.0 * jj1 - 1.0))
                                : 0.0;

                        e.g_t4 =
                            (DJ2 >= 2)
                                ? -((jj1 + 1.0) / J2) *
                                  std::sqrt((2.0 * J2 + 1.0) / (2.0 * J2 - 1.0)) *
                                  std::sqrt((J + jj1 - J2 + 1.0) * (-J + jj1 + J2))
                                : 0.0;
                    } else if (canJ2 && radJ2 > 0.0) {
                        e.branch = 2;

                        e.inv_lhs =
                            std::sqrt((2.0 * jj2 + 1.0) * (2.0 * jj2 + 3.0)) /
                            std::sqrt(radJ2);

                        e.g_t1 =
                            std::sqrt((J + J1 - jj2) * (J + J1 + jj2 + 1.0));

                        e.g_t2 =
                            (DJ >= 2)
                                ? ((jj2 + 1.0) / J) *
                                  std::sqrt((J - J1 + jj2) * (-J + J1 + jj2 + 1.0))
                                : 0.0;

                        e.g_t3 =
                            (DJ1 >= 2)
                                ? -((jj2 + 1.0) / J1) *
                                  std::sqrt((2.0 * J1 + 1.0) / (2.0 * J1 - 1.0)) *
                                  std::sqrt((J - J1 + jj2 + 1.0) * (-J + J1 + jj2))
                                : 0.0;

                        e.g_t4 =
                            (DJ2 >= 4)
                                ? ((jj2 + 1.0) / jj2) *
                                  std::sqrt(
                                      (-J + J1 + jj2) *
                                      ( J + J1 - jj2) *
                                      ( J + J1 - jj2 + 1.0) *
                                      ( J - J1 + jj2)
                                  ) /
                                  std::sqrt((2.0 * jj2 + 1.0) * (2.0 * jj2 - 1.0))
                                : 0.0;
                    } else if (radJ > 0.0) {
                        e.branch = 3;

                        e.inv_lhs =
                            (2.0 * jj + 1.0) /
                            std::sqrt(radJ);

                        e.g_t1 =
                            std::sqrt((jj + J1 + J2 + 1.0) * (-jj + J1 + J2));

                        e.g_t2 =
                            (DJ >= 4)
                                ? ((jj + 1.0) / (jj * (2.0 * jj + 1.0))) *
                                  std::sqrt(
                                      ( jj + J1 - J2) *
                                      ( jj - J1 + J2) *
                                      (-jj + J1 + J2) *
                                      (1.0 - jj + J1 + J2)
                                  )
                                : 0.0;

                        e.g_t3 =
                            (DJ1 >= 2)
                                ? ((jj + 1.0) / J1) *
                                  std::sqrt((2.0 * J1 + 1.0) / (2.0 * J1 - 1.0)) *
                                  std::sqrt((jj + J1 - J2) * (1.0 + jj - J1 + J2))
                                : 0.0;

                        e.g_t4 =
                            (DJ2 >= 2)
                                ? ((jj + 1.0) / J2) *
                                  std::sqrt((2.0 * J2 + 1.0) / (2.0 * J2 - 1.0)) *
                                  std::sqrt((1.0 + jj + J1 - J2) * (jj - J1 + J2))
                                : 0.0;
                    }

                    entries[idx] = e;
                }
            }
        }
    }
};

// ----------------------------------------------------------------------------
// SEED 3J CACHE
// ----------------------------------------------------------------------------

struct KappaSeed3jRowZ {
    bool valid = false;

    int Dnu    = 0;
    int seedJ1 = 0;
    int seedJ2 = 0;
    int seedJ  = 0;

    std::vector<double> vec3j;
};

inline bool compute_kappa_seed_geometry(
    int Dnu1,
    int Dnu2,
    int Dnu,
    int DmaxJ,
    int& seedJ1,
    int& seedJ2,
    int& seedJ)
{
    seedJ1 = Dnu1;
    seedJ2 = Dnu2;

    if (iabs_int(Dnu1 - Dnu2) <= Dnu && Dnu <= Dnu1 + Dnu2) {
        seedJ = Dnu;
    } else if (Dnu1 > Dnu2 + Dnu) {
        seedJ = Dnu1 - Dnu2;
    } else if (Dnu2 > Dnu1 + Dnu) {
        seedJ = Dnu2 - Dnu1;
    } else {
        return false;
    }

    return
        seedJ <= DmaxJ &&
        seedJ >= Dnu &&
        seedJ >= iabs_int(seedJ1 - seedJ2) &&
        seedJ <= seedJ1 + seedJ2;
}

struct KappaSeed3jCacheZ {
    int Dnu1  = 0;
    int Dnu2  = 0;
    int DmaxJ = 0;

    std::vector<KappaSeed3jRowZ> rows;

    void init(int Dnu1_in, int Dnu2_in, int DmaxJ_in) {
        Dnu1  = Dnu1_in;
        Dnu2  = Dnu2_in;
        DmaxJ = DmaxJ_in;

        rows.assign(DmaxJ + 1, KappaSeed3jRowZ{});

        for (int Dnu = 0; Dnu <= DmaxJ; ++Dnu) {
            int seedJ1;
            int seedJ2;
            int seedJ;

            if (!compute_kappa_seed_geometry(Dnu1, Dnu2, Dnu, DmaxJ, seedJ1, seedJ2, seedJ)) {
                continue;
            }

            KappaSeed3jRowZ row;

            row.Dnu    = Dnu;
            row.seedJ1 = seedJ1;
            row.seedJ2 = seedJ2;
            row.seedJ  = seedJ;

            row.vec3j = GetVec3j(seedJ1, seedJ2, Dnu, seedJ);
            row.valid = !row.vec3j.empty();

            rows[Dnu] = std::move(row);
        }
    }

    inline const KappaSeed3jRowZ* get(int Dnu) const noexcept {
        return
            (Dnu >= 0 && Dnu <= DmaxJ && rows[Dnu].valid)
                ? &rows[Dnu]
                : nullptr;
    }
};

// ----------------------------------------------------------------------------
// SCALAR WORKSPACE USED ONLY DURING TAPE COMPILATION
// ----------------------------------------------------------------------------

struct ThreadWorkspaceZ {
    std::vector<uint8_t> protected_flag;

    int stride1 = 0;
    int stride2 = 0;

    int DJ1_min = 0;
    int DJ2_min = 0;
    int DJ_min  = 0;

    std::size_t active_size = 0;

    void init_dims(int min1, int min2, int min3) {
        DJ1_min = min1;
        DJ2_min = min2;
        DJ_min  = min3;
    }

    void resize(int max_dim1, int max_dim2, int max_dim3) {
        stride2 = max_dim3 + 2;
        stride1 = (max_dim2 + 2) * stride2;

        active_size =
            static_cast<std::size_t>(max_dim1 + 2) *
            static_cast<std::size_t>(stride1);

        if (protected_flag.size() < active_size) {
            protected_flag.resize(active_size, 0);
        }
    }

    inline int get_idx(int DJ1, int DJ2, int DJ) const noexcept {
        return
            ((DJ1 - DJ1_min) / 2 + 2) * stride1 +
            ((DJ2 - DJ2_min) / 2 + 2) * stride2 +
            ((DJ  - DJ_min ) / 2 + 2);
    }
};

// ----------------------------------------------------------------------------
// TILED WORKSPACE USED DURING EXECUTION
// ----------------------------------------------------------------------------

template <int RhoTile>
struct ThreadWorkspaceTileZ {
    static_assert(RhoTile == 4 || RhoTile == 8 || RhoTile == 16);
    std::vector<double> z;

    // Per-DJ factors used by the taped recurrence.  Keep them in the
    // thread-local workspace so storage is allocated only when the required
    // DJ range grows, rather than on every quadrature tile.  Separate
    // cache-line-aligned buffers preserve the alignment assumptions of the
    // SIMD loops below without imposing a fixed spin cutoff.
    b4utils::AlignedBuffer<double> rho_sq_nrjj;
    b4utils::AlignedBuffer<double> rho_inv_sq_nrjj;
    b4utils::AlignedBuffer<double> rho_sq_nrj_inv;
    
    int stride1 = 0;
    int stride2 = 0;
    int DJ1_min = 0;
    int DJ2_min = 0;
    int DJ_min  = 0;
    std::size_t active_size = 0;
    
    void init_dims(int min1, int min2, int min3) {
        DJ1_min = min1;
        DJ2_min = min2;
        DJ_min  = min3;
    }
    
    void resize(int max_dim1, int max_dim2, int max_dim3) {
        stride2 = max_dim3 + 2;
        stride1 = (max_dim2 + 2) * stride2;
        active_size = static_cast<std::size_t>(max_dim1 + 2) * static_cast<std::size_t>(stride1);
        
        // Interleave R and I: Size is active_size * RhoTile * 2.
        const std::size_t n = active_size * RhoTile * 2;
        if (z.size() < n) {
            z.resize(n, 0.0);
        }
    }

    void resize_rho_lookups(int max_DJ) {
        if (max_DJ < 0) {
            throw std::invalid_argument(
                "negative maximum doubled spin for rho lookup storage");
        }

        const std::size_t rows = static_cast<std::size_t>(max_DJ) + 1;
        if (rows > std::numeric_limits<std::size_t>::max() / RhoTile) {
            throw std::overflow_error("rho lookup storage size overflows size_t");
        }

        const std::size_t count = rows * RhoTile;
        rho_sq_nrjj.resize_discard(count);
        rho_inv_sq_nrjj.resize_discard(count);
        rho_sq_nrj_inv.resize_discard(count);
    }
    
    inline int get_idx(int DJ1, int DJ2, int DJ) const noexcept {
        return ((DJ1 - DJ1_min) / 2 + 2) * stride1 + ((DJ2 - DJ2_min) / 2 + 2) * stride2 + ((DJ  - DJ_min ) / 2 + 2);
    }
    
    // R and I are now perfectly adjacent in memory!
    inline double* cell_R(int idx) noexcept {
        return z.data() + static_cast<std::size_t>(idx) * (RhoTile * 2);
    }
    
    inline double* cell_I(int idx) noexcept {
        return z.data() + static_cast<std::size_t>(idx) * (RhoTile * 2) + RhoTile;
    }
    
    inline const double* cell_R(int idx) const noexcept {
        return z.data() + static_cast<std::size_t>(idx) * (RhoTile * 2);
    }
    
    inline const double* cell_I(int idx) const noexcept {
        return z.data() + static_cast<std::size_t>(idx) * (RhoTile * 2) + RhoTile;
    }
};

// ----------------------------------------------------------------------------
// TAPE STRUCTURES
// ----------------------------------------------------------------------------

struct BranchOpZ {
    uint32_t ic;
    uint16_t dj;
    double t1A;
    double t1B;
    double t2C;
    double t3C;
    double t4C;
};

struct BranchTapeZ {
    std::vector<BranchOpZ> ops;
    void clear() { ops.clear(); }
};

struct TapeChunkZ {
    uint8_t kind;
    uint32_t begin;
    uint32_t end;
};

inline C kappaMin_seed_cached(
    double rho1,
    int Dnu1,
    double rho2,
    int Dnu2,
    double rho,
    int Dnu,
    const KappaSeed3jCacheZ* cache,
    const std::vector<double>& LF,
    double logDen)
{
    if (cache != nullptr) {
        if (const KappaSeed3jRowZ* row = cache->get(Dnu)) {
            return kappaMinVec(
                rho1,
                row->seedJ1,
                rho2,
                row->seedJ2,
                rho,
                Dnu,
                row->seedJ,
                row->vec3j,
                LF,
                logDen
            );
        }
    }

    int seedJ1;
    int seedJ2;
    int seedJ;

    if (!compute_kappa_seed_geometry(
            Dnu1,
            Dnu2,
            Dnu,
            std::numeric_limits<int>::max(),
            seedJ1,
            seedJ2,
            seedJ))
    {
        return C(0.0, 0.0);
    }

    return kappaMin(
        rho1,
        seedJ1,
        rho2,
        seedJ2,
        rho,
        Dnu,
        seedJ,
        LF,
        logDen
    );
}

struct KappaTapeZ {
    enum : uint8_t { CH_B1 = 1, CH_B2 = 2, CH_B3 = 3 };
    
    BranchTapeZ branch1;
    BranchTapeZ branch2;
    BranchTapeZ branch3;
    
    std::vector<TapeChunkZ> chunks;
    
    int max_DJ_needed = 0;
    
    void clear_for_compile() {
        branch1.clear(); branch2.clear(); branch3.clear();
        chunks.clear();
        max_DJ_needed = 0;
    }

    inline void note_chunk(uint8_t kind, uint32_t begin, uint32_t end) {
        if (!chunks.empty() && chunks.back().kind == kind && chunks.back().end == begin) {
            chunks.back().end = end;
        } else {
            chunks.push_back(TapeChunkZ{kind, begin, end});
        }
    }


    inline void push_op(BranchTapeZ& tape, uint8_t kind, uint32_t idx_curr, uint32_t rad_DJ, double t1A, double t1B, double t2C, double t3C, double t4C) {
        const uint32_t pos = static_cast<uint32_t>(tape.ops.size());
        tape.ops.push_back({idx_curr, static_cast<uint16_t>(rad_DJ), t1A, t1B, t2C, t3C, t4C});
        note_chunk(kind, pos, pos + 1);
    }

    void compile(
        double rho1,
        int Dnu1,
        int Dl1_max,
        double rho2,
        int Dnu2,
        int Dl2_max,
        int Dnu,
        int DmaxJ,
        const PureGeomCacheZ& geom)
    {
        clear_for_compile();

        // Prevent heavy reallocations during compilation
        branch1.ops.reserve(8192);
        branch2.ops.reserve(8192);
        branch3.ops.reserve(8192);
        chunks.reserve(8192);

        const int a = Dnu1;
        const int b = Dnu2;
        const int c = Dnu;

        const double nu1 = 0.5 * Dnu1;
        const double nu2 = 0.5 * Dnu2;
        const double nu  = 0.5 * Dnu;

        max_DJ_needed = DmaxJ;

        ThreadWorkspaceZ ws;

        // The +2 addressing halo in ThreadWorkspaceZ makes predecessors at
        // Dnu1-2/Dnu1-4 and Dnu2-2/Dnu2-4 safe even though those states are
        // nonphysical and remain zero.
        ws.init_dims(Dnu1, Dnu2, DmaxJ & 1);

        ws.resize(
            (Dl1_max - ws.DJ1_min) / 2 + 1,
            (Dl2_max - ws.DJ2_min) / 2 + 1,
            (DmaxJ   - ws.DJ_min ) / 2 + 1
        );

        std::fill(ws.protected_flag.begin(), ws.protected_flag.end(), 0);

        auto validQ = [&](int J1x, int J2x, int Jx) {
            return
                J1x >= a &&
                J2x >= b &&
                Jx  >= c &&
                J1x <= Dl1_max &&
                J2x <= Dl2_max &&
                Jx  <= DmaxJ &&
                Jx  <= J1x + J2x &&
                Jx  >= iabs_int(J1x - J2x);
        };

        auto protect = [&](int DJ1, int DJ2, int DJ) {
            const int idx = ws.get_idx(DJ1, DJ2, DJ);
            ws.protected_flag[idx] = 1;
        };

        int seedJ1;
        int seedJ2;
        int seedJ;

        if (iabs_int(a - b) <= c && c <= a + b) {
            seedJ1 = a;
            seedJ2 = b;
            seedJ  = c;
        } else if (a > b + c) {
            seedJ1 = a;
            seedJ2 = b;
            seedJ  = a - b;
        } else if (b > a + c) {
            seedJ1 = a;
            seedJ2 = b;
            seedJ  = b - a;
        } else {
            return;
        }

        if (!validQ(seedJ1, seedJ2, seedJ)) {
            return;
        }

        protect(seedJ1, seedJ2, seedJ);

        if (a > b + c && c <= b) {
            for (int q = 0; q < (a - b - c) / 2; ++q) {
                const int n2 = b + 2 * q + 2;
                const int n3 = a - b - 2 * q - 2;

                if (validQ(a, n2, n3)) {
                    protect(a, n2, n3);
                }
            }
        } else if (a > b + c && b < c) {
            for (int q = (a - b - c) / 2 - 1; q >= 0; --q) {
                const int o2 = a - c - 2 * q;
                const int o3 = c + 2 * q;

                if (validQ(a, o2, o3)) {
                    protect(a, o2, o3);
                }
            }
        } else if (b > a + c && c <= a) {
            for (int q = 0; q < (b - a - c) / 2; ++q) {
                const int n1 = a + 2 * q + 2;
                const int n3 = b - a - 2 * q - 2;

                if (validQ(n1, b, n3)) {
                    protect(n1, b, n3);
                }
            }
        } else if (b > a + c && a < c) {
            for (int q = (b - a - c) / 2 - 1; q >= 0; --q) {
                const int o1 = b - c - 2 * q;
                const int o3 = c + 2 * q;

                if (validQ(o1, b, o3)) {
                    protect(o1, b, o3);
                }
            }
        }

        if (c == 0) {
            for (int K = std::max(a, b); K < std::min(Dl1_max, Dl2_max); K += 2) {
                if (validQ(K + 2, K + 2, 0)) {
                    protect(K + 2, K + 2, 0);
                }
            }
        }

        if (a == 0) {
            for (int K = std::max(b, c); K < std::min(Dl2_max, DmaxJ); K += 2) {
                if (validQ(0, K + 2, K + 2)) {
                    protect(0, K + 2, K + 2);
                }
            }
        }

        if (b == 0) {
            for (int K = std::max(a, c); K < std::min(Dl1_max, DmaxJ); K += 2) {
                if (validQ(K + 2, 0, K + 2)) {
                    protect(K + 2, 0, K + 2);
                }
            }
        }

        for (int DJ1 = a; DJ1 <= Dl1_max; DJ1 += 2) {
            const int curr = (DJ1 - ws.DJ1_min) / 2 + 2;
            const double J1  = 0.5 * DJ1;
            const double jj1 = J1 - 1.0;

            const double sqNRJ1 =
                (J1 >= nu1)
                    ? std::sqrt((J1 * J1 - nu1 * nu1) * (J1 * J1 + rho1 * rho1))
                    : 0.0;

            const double inv_sqNRJ1 =
                (sqNRJ1 > 0.0)
                    ? 1.0 / sqNRJ1
                    : 0.0;

            const double sqNR1 =
                (jj1 >= nu1)
                    ? std::sqrt((jj1 * jj1 - nu1 * nu1) * (jj1 * jj1 + rho1 * rho1))
                    : 0.0;

            const double C2_1 =
                (jj1 == 0.0)
                    ? 0.0
                    : (nu1 * rho1) / jj1;

            for (int DJ2 = b; DJ2 <= Dl2_max; DJ2 += 2) {
                const int idx2 =
                    (DJ2 - ws.DJ2_min) / 2 + 2;

                const int ow_idx =
                    ((DJ1 - ws.DJ1_min) / 2) * geom.dim2 +
                    ((DJ2 - ws.DJ2_min) / 2);

                const double J2  = 0.5 * DJ2;
                const double jj2 = J2 - 1.0;

                const double sqNRJ2 =
                    (J2 >= nu2)
                        ? std::sqrt((J2 * J2 - nu2 * nu2) * (J2 * J2 + rho2 * rho2))
                        : 0.0;

                const double sqNR2 =
                    (jj2 >= nu2)
                        ? std::sqrt((jj2 * jj2 - nu2 * nu2) * (jj2 * jj2 + rho2 * rho2))
                        : 0.0;

                const double C3_2 =
                    (jj2 == 0.0)
                        ? 0.0
                        : (nu2 * rho2) / jj2;

                int start_DJ = std::max(c, iabs_int(DJ1 - DJ2));

                if ((start_DJ & 1) != ws.DJ_min) {
                    ++start_DJ;
                }

                const double f3_1 =
                    (DJ2 == 0)
                        ? 0.0
                        : (J1 * nu2 * rho2) / J2;

                const double f2_2 =
                    (DJ1 == 0)
                        ? 0.0
                        : (J2 * nu1 * rho1) / J1;

                const double C1_3_coeff =
                    nu * (J1 - J2);

                const double C23_3 =
                    ((DJ2 == 0) ? 0.0 : (nu2 * rho2) / J2) -
                    ((DJ1 == 0) ? 0.0 : (nu1 * rho1) / J1);

                for (int DJ = start_DJ; DJ <= std::min(DmaxJ, DJ1 + DJ2); DJ += 2) {
                    const int idx3 =
                        (DJ - ws.DJ_min) / 2 + 2;

                    const int i_curr =
                        curr * ws.stride1 +
                        idx2 * ws.stride2 +
                        idx3;

                    if (ws.protected_flag[i_curr]) {
                        continue;
                    }

                    const PureGeomEntryZ& e =
                        geom.entries[
                            ow_idx * geom.dim3 +
                            (DJ - geom.DJ_min) / 2
                        ];

                    if (e.branch == 1) {
                        const double J = 0.5 * DJ;

                        const double inv_J =
                            (DJ == 0)
                                ? 0.0
                                : 1.0 / J;

                        push_op(
                            branch1,
                            CH_B1,
                            static_cast<uint32_t>(i_curr),
                            static_cast<uint32_t>(DJ),

                            (e.g_t1 * inv_sqNRJ1) *
                                (-J1 * nu * inv_J) *
                                e.inv_lhs,

                            (e.g_t1 * inv_sqNRJ1) *
                                (J * C2_1 + f3_1 - J2 * C2_1) *
                                e.inv_lhs,

                            e.g_t2 * inv_sqNRJ1 * e.inv_lhs,
                            e.g_t3 * sqNR1 * inv_sqNRJ1 * e.inv_lhs,
                            e.g_t4 * sqNRJ2 * inv_sqNRJ1 * e.inv_lhs
                        );
                    } else if (e.branch == 2) {
                        const double J = 0.5 * DJ;

                        const double inv_J =
                            (DJ == 0)
                                ? 0.0
                                : 1.0 / J;

                        const double inv_sqNRJ2 =
                            (sqNRJ2 > 0.0)
                                ? 1.0 / sqNRJ2
                                : 0.0;

                        push_op(
                            branch2,
                            CH_B2,
                            static_cast<uint32_t>(i_curr),
                            static_cast<uint32_t>(DJ),

                            -(e.g_t1 * inv_sqNRJ2) *
                                (-J2 * nu * inv_J) *
                                e.inv_lhs,

                            -(e.g_t1 * inv_sqNRJ2) *
                                (J * C3_2 + f2_2 - J1 * C3_2) *
                                e.inv_lhs,

                            e.g_t2 * inv_sqNRJ2 * e.inv_lhs,
                            e.g_t3 * sqNRJ1 * inv_sqNRJ2 * e.inv_lhs,
                            e.g_t4 * sqNR2 * inv_sqNRJ2 * e.inv_lhs
                        );
                    } else if (e.branch == 3 && DJ > c) {
                        const double J  = 0.5 * DJ;
                        const double jj = J - 1.0;

                        const double inv_jj =
                            (jj == 0.0)
                                ? 0.0
                                : 1.0 / jj;

                        push_op(
                            branch3,
                            CH_B3,
                            static_cast<uint32_t>(i_curr),
                            static_cast<uint32_t>(DJ),

                            e.g_t1 *
                                (C1_3_coeff * inv_jj) *
                                e.inv_lhs,

                            e.g_t1 *
                                (J * C23_3) *
                                e.inv_lhs,

                            ((DJ - 4 >= c) ? e.g_t2 : 0.0) *
                                e.inv_lhs,

                            ((DJ1 - 2 >= a) ? e.g_t3 * sqNRJ1 : 0.0) *
                                e.inv_lhs,

                            ((DJ2 - 2 >= b) ? e.g_t4 * sqNRJ2 : 0.0) *
                                e.inv_lhs
                        );
                    }
                }

            }
        }

    }
};

struct OutputZeroBlockZ {
    std::size_t begin = 0;
    std::size_t end   = 0;
};

// ----------------------------------------------------------------------------
// TILE EXECUTOR BRANCH MICROKERNELS
// ----------------------------------------------------------------------------

template <int RhoTile>
inline void execute_branch1_tile(
                                 const BranchTapeZ& t, uint32_t begin, uint32_t end,
                                 const double* __restrict rho, int npt,
                                 ThreadWorkspaceTileZ<RhoTile>& ws, const double* __restrict RHO_sqNRJJ)
{
    double* __restrict ZR = ws.z.data(); // Note: ws.z.data() is used now!
    
    const int stride1 = ws.stride1;
    const int stride2 = ws.stride2;
    
    for (uint32_t i = begin; i < end; ++i) {
        const BranchOpZ& op = t.ops[i];
        
        const uint32_t ic = op.ic;
        const uint32_t i1 = ic - stride1;
        const uint32_t i2 = ic - stride1 - 1;
        const uint32_t i3 = ic - 2 * stride1;
        const uint32_t i4 = ic - stride1 - stride2;
        
        const double* __restrict R1 = ZR + static_cast<std::size_t>(i1) * (RhoTile * 2);
        const double* __restrict I1 = R1 + RhoTile;
        
        const double* __restrict R2 = ZR + static_cast<std::size_t>(i2) * (RhoTile * 2);
        const double* __restrict I2 = R2 + RhoTile;
        
        const double* __restrict R3 = ZR + static_cast<std::size_t>(i3) * (RhoTile * 2);
        const double* __restrict I3 = R3 + RhoTile;
        
        const double* __restrict R4 = ZR + static_cast<std::size_t>(i4) * (RhoTile * 2);
        const double* __restrict I4 = R4 + RhoTile;
        
        double* __restrict CR = ZR + static_cast<std::size_t>(ic) * (RhoTile * 2);
        double* __restrict CI = CR + RhoTile;
        
        const double* __restrict sq = RHO_sqNRJJ + static_cast<std::size_t>(op.dj) * RhoTile;
        
        const double t1A = op.t1A; const double t1B = op.t1B;
        const double t2C = op.t2C; const double t3 = op.t3C; const double t4 = op.t4C;
        
#pragma omp simd
        for (int lane = 0; lane < npt; ++lane) {
            const double tt1 = t1A * rho[lane] + t1B;
            const double tt2 = t2C * sq[lane];
            CR[lane] = -I1[lane] * tt1 + R2[lane] * tt2 + R3[lane] * t3 + R4[lane] * t4;
            CI[lane] = R1[lane] * tt1 + I2[lane] * tt2 + I3[lane] * t3 + I4[lane] * t4;
        }
    }
}

template <int RhoTile>
inline void execute_branch2_tile(
                                 const BranchTapeZ& t, uint32_t begin, uint32_t end,
                                 const double* __restrict rho, int npt,
                                 ThreadWorkspaceTileZ<RhoTile>& ws, const double* __restrict RHO_sqNRJJ)
{
    double* __restrict ZR = ws.z.data();
    
    const int stride1 = ws.stride1;
    const int stride2 = ws.stride2;
    
    for (uint32_t i = begin; i < end; ++i) {
        const BranchOpZ& op = t.ops[i];
        
        const uint32_t ic = op.ic;
        const uint32_t i1 = ic - stride2;
        const uint32_t i2 = ic - stride2 - 1;
        const uint32_t i3 = ic - stride1 - stride2;
        const uint32_t i4 = ic - 2 * stride2;
        
        const double* __restrict R1 = ZR + static_cast<std::size_t>(i1) * (RhoTile * 2);
        const double* __restrict I1 = R1 + RhoTile;
        
        const double* __restrict R2 = ZR + static_cast<std::size_t>(i2) * (RhoTile * 2);
        const double* __restrict I2 = R2 + RhoTile;
        
        const double* __restrict R3 = ZR + static_cast<std::size_t>(i3) * (RhoTile * 2);
        const double* __restrict I3 = R3 + RhoTile;
        
        const double* __restrict R4 = ZR + static_cast<std::size_t>(i4) * (RhoTile * 2);
        const double* __restrict I4 = R4 + RhoTile;
        
        double* __restrict CR = ZR + static_cast<std::size_t>(ic) * (RhoTile * 2);
        double* __restrict CI = CR + RhoTile;
        
        const double* __restrict sq = RHO_sqNRJJ + static_cast<std::size_t>(op.dj) * RhoTile;
        
        const double t1A = op.t1A; const double t1B = op.t1B;
        const double t2C = op.t2C; const double t3 = op.t3C; const double t4 = op.t4C;
        
#pragma omp simd
        for (int lane = 0; lane < npt; ++lane) {
            const double tt1 = t1A * rho[lane] + t1B;
            const double tt2 = t2C * sq[lane];
            CR[lane] = -I1[lane] * tt1 + R2[lane] * tt2 + R3[lane] * t3  + R4[lane] * t4;
            CI[lane] = R1[lane] * tt1 + I2[lane] * tt2 + I3[lane] * t3  + I4[lane] * t4;
        }
    }
}

template <int RhoTile>
inline void execute_branch3_tile(
                                 const BranchTapeZ& t, uint32_t begin, uint32_t end,
                                 const double* __restrict rho, int npt,
                                 ThreadWorkspaceTileZ<RhoTile>& ws, const double* __restrict RHO_inv_sqNRJJ, const double* __restrict RHO_sqNRJ_inv)
{
    double* __restrict ZR = ws.z.data();
    
    const int stride1 = ws.stride1;
    const int stride2 = ws.stride2;
    
    for (uint32_t i = begin; i < end; ++i) {
        const BranchOpZ& op = t.ops[i];
        
        const uint32_t ic = op.ic;
        const uint32_t i1 = ic - 1;
        const uint32_t i2 = ic - 2;
        const uint32_t i3 = ic - stride1 - 1;
        const uint32_t i4 = ic - stride2 - 1;
        
        const double* __restrict R1 = ZR + static_cast<std::size_t>(i1) * (RhoTile * 2);
        const double* __restrict I1 = R1 + RhoTile;
        
        const double* __restrict R2 = ZR + static_cast<std::size_t>(i2) * (RhoTile * 2);
        const double* __restrict I2 = R2 + RhoTile;
        
        const double* __restrict R3 = ZR + static_cast<std::size_t>(i3) * (RhoTile * 2);
        const double* __restrict I3 = R3 + RhoTile;
        
        const double* __restrict R4 = ZR + static_cast<std::size_t>(i4) * (RhoTile * 2);
        const double* __restrict I4 = R4 + RhoTile;
        
        double* __restrict CR = ZR + static_cast<std::size_t>(ic) * (RhoTile * 2);
        double* __restrict CI = CR + RhoTile;
        
        const double* __restrict inv = RHO_inv_sqNRJJ + static_cast<std::size_t>(op.dj) * RhoTile;
        const double* __restrict sqinv = RHO_sqNRJ_inv + static_cast<std::size_t>(op.dj) * RhoTile;
        
        const double t1A = op.t1A; const double t1B = op.t1B;
        const double t2C = op.t2C; const double t3C = op.t3C; const double t4C = op.t4C;
        
#pragma omp simd
        for (int lane = 0; lane < npt; ++lane) {
            const double ii = inv[lane];
            const double tt1 = (t1A * rho[lane] + t1B) * ii;
            const double tt2 = t2C * sqinv[lane];
            const double tt3 = t3C * ii;
            const double tt4 = t4C * ii;
            
            CR[lane] = -I1[lane] * tt1 + R2[lane] * tt2 + R3[lane] * tt3 + R4[lane] * tt4;
            CI[lane] = R1[lane] * tt1 + I2[lane] * tt2 + I3[lane] * tt3 + I4[lane] * tt4;
        }
    }
}

// ----------------------------------------------------------------------------
// TILED DP EXECUTOR
// ----------------------------------------------------------------------------

template <int RhoTile>
inline void KappaFastZ_ExecuteTapeTile(
    double rho1,
    int Dnu1,
    int Dl1_max,
    double rho2,
    int Dnu2,
    int Dl2_max,
    const double* __restrict rho_vec,
    int npt,
    int Dnu,
    int DmaxJ,
    const std::vector<double>& LF,
    ThreadWorkspaceTileZ<RhoTile>& ws,
    const KappaTapeZ& tape,
    const KappaSeed3jCacheZ* seed_cache)
{
    const int a = Dnu1;
    const int b = Dnu2;
    const int c = Dnu;

    const double nu = 0.5 * Dnu;

    auto validQ = [&](int J1x, int J2x, int Jx) {
        return
            J1x >= a &&
            J2x >= b &&
            Jx  >= c &&
            J1x <= Dl1_max &&
            J2x <= Dl2_max &&
            Jx  <= DmaxJ &&
            Jx  <= J1x + J2x &&
            Jx  >= iabs_int(J1x - J2x);
    };

    int seedJ1;
    int seedJ2;
    int seedJ;

    if (iabs_int(a - b) <= c && c <= a + b) {
        seedJ1 = a;
        seedJ2 = b;
        seedJ  = c;
    } else if (a > b + c) {
        seedJ1 = a;
        seedJ2 = b;
        seedJ  = a - b;
    } else if (b > a + c) {
        seedJ1 = a;
        seedJ2 = b;
        seedJ  = b - a;
    } else {
        return;
    }

    if (!validQ(seedJ1, seedJ2, seedJ)) {
        return;
    }

    const int signCos =
        (((Dnu1 + Dnu2 + Dnu) / 2) & 1)
            ? -1
            : 1;

    const double seed_scale =
        zsign_as_double(Dnu1, Dnu2, seedJ1, seedJ2) *
        std::sqrt((seedJ1 + 1.0) * (seedJ2 + 1.0) * (seedJ + 1.0));

    {
        const int seed_idx = ws.get_idx(seedJ1, seedJ2, seedJ);

        double* __restrict SR = ws.cell_R(seed_idx);
        double* __restrict SI = ws.cell_I(seed_idx);

        for (int lane = 0; lane < npt; ++lane) {
            const double rho = rho_vec[lane];

            const double hx =
                0.5 * std::abs(B4_PI * (rho1 + rho2 + rho));

            const double logDen =
                (signCos == +1)
                    ? (B4_LN2 + 2.0 * log_cosh(hx))
                    : (B4_LN2 + 2.0 * log_sinh_pos(hx));

            const C kMin =
                kappaMin_seed_cached(
                    rho1,
                    Dnu1,
                    rho2,
                    Dnu2,
                    rho,
                    Dnu,
                    seed_cache,
                    LF,
                    logDen
                );

            SR[lane] = kMin.real() * seed_scale;
            SI[lane] = kMin.imag() * seed_scale;
        }
    }

    // ------------------------------------------------------------------------
    // Dynamic boundary mapping, vectorized over rho lanes.
    // ------------------------------------------------------------------------

    if (a > b + c && c <= b) {
        for (int q = 0; q < (a - b - c) / 2; ++q) {
            const int o2 = b + 2 * q;
            const int o3 = a - b - 2 * q;

            const int n2 = o2 + 2;
            const int n3 = o3 - 2;

            if (!validQ(a, n2, n3)) continue;

            const int old_idx = ws.get_idx(a, o2, o3);
            const int new_idx = ws.get_idx(a, n2, n3);

            const double denom = rup_val(b, rho2, n2);

            if (std::abs(denom) <= 1e-300) continue;

            const double geom =
                std::sqrt(
                    (b + 2.0 * q + 3.0) *
                    (a - b - 2.0 * q) /
                    (
                        (a - b - 2.0 * q - 1.0) *
                        (b + 2.0 * q + 2.0)
                    )
                ) / denom;

            const double* __restrict OR = ws.cell_R(old_idx);
            const double* __restrict OI = ws.cell_I(old_idx);

            double* __restrict NR = ws.cell_R(new_idx);
            double* __restrict NI = ws.cell_I(new_idx);

#pragma omp simd
            for (int lane = 0; lane < npt; ++lane) {
                const double factor =
                    rup_val(c, rho_vec[lane], o3) * geom;

                NR[lane] = factor * OR[lane];
                NI[lane] = factor * OI[lane];
            }
        }
    } else if (a > b + c && b < c) {
        for (int q = (a - b - c) / 2 - 1; q >= 0; --q) {
            const int o2 = a - c - 2 * q;
            const int o3 = c + 2 * q;

            const int n2 = o2 - 2;
            const int n3 = o3 + 2;

            if (!validQ(a, o2, o3)) continue;

            const int old_idx = ws.get_idx(a, o2, o3);
            const int new_idx = ws.get_idx(a, n2, n3);

            const double rup_b = rup_val(b, rho2, o2);

            const double geom =
                std::sqrt(
                    (a - c - 2.0 * q) *
                    (c + 2.0 * q + 1.0) /
                    (
                        (a - c - 2.0 * q + 1.0) *
                        (c + 2.0 * q + 2.0)
                    )
                );

            const double* __restrict NR0 = ws.cell_R(new_idx);
            const double* __restrict NI0 = ws.cell_I(new_idx);

            double* __restrict OR0 = ws.cell_R(old_idx);
            double* __restrict OI0 = ws.cell_I(old_idx);

#pragma omp simd
            for (int lane = 0; lane < npt; ++lane) {
                const double denom = rup_val(c, rho_vec[lane], n3);

                double FX = 0.0;

                if (std::abs(denom) > 1e-300) {
                    FX = (rup_b / denom) * geom;
                }

                if (std::abs(FX) > 1e-14) {
                    OR0[lane] = NR0[lane] / FX;
                    OI0[lane] = NI0[lane] / FX;
                }
            }
        }
    } else if (b > a + c && c <= a) {
        for (int q = 0; q < (b - a - c) / 2; ++q) {
            const int o1 = a + 2 * q;
            const int o3 = b - a - 2 * q;

            const int n1 = o1 + 2;
            const int n3 = o3 - 2;

            if (!validQ(n1, b, n3)) continue;

            const int old_idx = ws.get_idx(o1, b, o3);
            const int new_idx = ws.get_idx(n1, b, n3);

            const double denom = rup_val(a, rho1, n1);

            if (std::abs(denom) <= 1e-300) continue;

            const double geom =
                std::sqrt(
                    (b - a - 2.0 * q) *
                    (a + 2.0 * q + 3.0) /
                    (
                        (b - a - 2.0 * q - 1.0) *
                        (a + 2.0 * q + 2.0)
                    )
                ) / denom;

            const double* __restrict OR = ws.cell_R(old_idx);
            const double* __restrict OI = ws.cell_I(old_idx);

            double* __restrict NR = ws.cell_R(new_idx);
            double* __restrict NI = ws.cell_I(new_idx);

#pragma omp simd
            for (int lane = 0; lane < npt; ++lane) {
                const double factor =
                    rup_val(c, rho_vec[lane], o3) * geom;

                NR[lane] = factor * OR[lane];
                NI[lane] = factor * OI[lane];
            }
        }
    } else if (b > a + c && a < c) {
        for (int q = (b - a - c) / 2 - 1; q >= 0; --q) {
            const int o1 = b - c - 2 * q;
            const int o3 = c + 2 * q;

            const int n1 = o1 - 2;
            const int n3 = o3 + 2;

            if (!validQ(o1, b, o3)) continue;

            const int old_idx = ws.get_idx(o1, b, o3);
            const int new_idx = ws.get_idx(n1, b, n3);

            const double rup_a = rup_val(a, rho1, o1);

            const double geom =
                std::sqrt(
                    (b - c - 2.0 * q) *
                    (c + 2.0 * q + 1.0) /
                    (
                        (b - c - 2.0 * q + 1.0) *
                        (c + 2.0 * q + 2.0)
                    )
                );

            const double* __restrict NR0 = ws.cell_R(new_idx);
            const double* __restrict NI0 = ws.cell_I(new_idx);

            double* __restrict OR0 = ws.cell_R(old_idx);
            double* __restrict OI0 = ws.cell_I(old_idx);

#pragma omp simd
            for (int lane = 0; lane < npt; ++lane) {
                const double denom = rup_val(c, rho_vec[lane], n3);

                double FX = 0.0;

                if (std::abs(denom) > 1e-300) {
                    FX = (rup_a / denom) * geom;
                }

                if (std::abs(FX) > 1e-14) {
                    OR0[lane] = NR0[lane] / FX;
                    OI0[lane] = NI0[lane] / FX;
                }
            }
        }
    }

    if (c == 0) {
        for (int K = std::max(a, b); K < std::min(Dl1_max, Dl2_max); K += 2) {
            if (!validQ(K + 2, K + 2, 0)) continue;

            const int out_idx = ws.get_idx(K + 2, K + 2, 0);
            const int cur_idx = ws.get_idx(K, K, 0);

            const int prev_idx =
                (K >= std::max(a, b) + 2)
                    ? ws.get_idx(K - 2, K - 2, 0)
                    : -1;

            double* __restrict OR = ws.cell_R(out_idx);
            double* __restrict OI = ws.cell_I(out_idx);

            const double* __restrict CR = ws.cell_R(cur_idx);
            const double* __restrict CI = ws.cell_I(cur_idx);

            const double* __restrict PR =
                (prev_idx >= 0)
                    ? ws.cell_R(prev_idx)
                    : nullptr;

            const double* __restrict PI =
                (prev_idx >= 0)
                    ? ws.cell_I(prev_idx)
                    : nullptr;

            const double L_pref =
                -((K + 2.0) / std::sqrt((K + 1.0) * (K + 3.0))) *
                rup_val(a, rho1, K + 2) *
                rup_val(b, rho2, K + 2);

            // Anderson Eq. (25a) has no predecessor on the first admissible
            // diagonal cell.  In particular, for half-integer input K=1 the
            // formal coefficient contains sqrt(K-1) and must not be evaluated:
            // that term multiplies the nonphysical X(K-2,K-2,0), not zero data
            // stored at a physical predecessor.
            const double prev_coeff =
                (prev_idx < 0)
                    ? 0.0
                    : -K / std::sqrt((K - 1.0) * (K + 1.0)) *
                      rup_val(a, rho1, K) *
                      rup_val(b, rho2, K);

#pragma omp simd
            for (int lane = 0; lane < npt; ++lane) {
                if (std::abs(L_pref) <= 1e-14) continue;

                const double rho = rho_vec[lane];

                const double center =
                    2.0 * K * K +
                    4.0 * K +
                    4.0 -
                    a * a -
                    b * b +
                    4.0 * rho1 * rho1 +
                    4.0 * rho2 * rho2 -
                    4.0 * rho * rho -
                    (
                        K != 0
                            ? (8.0 * a * rho1 * b * rho2) / (K * (K + 2.0))
                            : 0.0
                    );

                const double pR = (PR != nullptr) ? PR[lane] : 0.0;
                const double pI = (PI != nullptr) ? PI[lane] : 0.0;

                OR[lane] =
                    (center * CR[lane] - prev_coeff * pR) /
                    L_pref;

                OI[lane] =
                    (center * CI[lane] - prev_coeff * pI) /
                    L_pref;
            }
        }
    }

    if (a == 0) {
        for (int K = std::max(b, c); K < std::min(Dl2_max, DmaxJ); K += 2) {
            if (!validQ(0, K + 2, K + 2)) continue;

            const int out_idx = ws.get_idx(0, K + 2, K + 2);
            const int cur_idx = ws.get_idx(0, K, K);

            const int prev_idx =
                (K >= std::max(b, c) + 2)
                    ? ws.get_idx(0, K - 2, K - 2)
                    : -1;

            double* __restrict OR = ws.cell_R(out_idx);
            double* __restrict OI = ws.cell_I(out_idx);

            const double* __restrict CR = ws.cell_R(cur_idx);
            const double* __restrict CI = ws.cell_I(cur_idx);

            const double* __restrict PR =
                (prev_idx >= 0)
                    ? ws.cell_R(prev_idx)
                    : nullptr;

            const double* __restrict PI =
                (prev_idx >= 0)
                    ? ws.cell_I(prev_idx)
                    : nullptr;

#pragma omp simd
            for (int lane = 0; lane < npt; ++lane) {
                const double rho = rho_vec[lane];

                const double L =
                    -((K + 2.0) / (K + 1.0)) *
                    rup_val(b, rho2, K + 2) *
                    rup_val(c, rho, K + 2);

                if (std::abs(L) <= 1e-14) continue;

                const double center =
                    -(
                        2.0 * K * K +
                        4.0 * K +
                        4.0 -
                        c * c -
                        b * b -
                        4.0 * rho1 * rho1 +
                        4.0 * rho2 * rho2 +
                        4.0 * rho * rho -
                        (
                            K != 0
                                ? (8.0 * c * rho * b * rho2) / (K * (K + 2.0))
                                : 0.0
                        )
                    );

                const double prev_coeff =
                    (prev_idx < 0)
                        ? 0.0
                        : -K / (K + 1.0) *
                          rup_val(b, rho2, K) *
                          rup_val(c, rho, K);

                const double pR = (PR != nullptr) ? PR[lane] : 0.0;
                const double pI = (PI != nullptr) ? PI[lane] : 0.0;

                OR[lane] =
                    (center * CR[lane] - prev_coeff * pR) /
                    L;

                OI[lane] =
                    (center * CI[lane] - prev_coeff * pI) /
                    L;
            }
        }
    }

    if (b == 0) {
        for (int K = std::max(a, c); K < std::min(Dl1_max, DmaxJ); K += 2) {
            if (!validQ(K + 2, 0, K + 2)) continue;

            const int out_idx = ws.get_idx(K + 2, 0, K + 2);
            const int cur_idx = ws.get_idx(K, 0, K);

            const int prev_idx =
                (K >= std::max(a, c) + 2)
                    ? ws.get_idx(K - 2, 0, K - 2)
                    : -1;

            double* __restrict OR = ws.cell_R(out_idx);
            double* __restrict OI = ws.cell_I(out_idx);

            const double* __restrict CR = ws.cell_R(cur_idx);
            const double* __restrict CI = ws.cell_I(cur_idx);

            const double* __restrict PR =
                (prev_idx >= 0)
                    ? ws.cell_R(prev_idx)
                    : nullptr;

            const double* __restrict PI =
                (prev_idx >= 0)
                    ? ws.cell_I(prev_idx)
                    : nullptr;

#pragma omp simd
            for (int lane = 0; lane < npt; ++lane) {
                const double rho = rho_vec[lane];

                const double L =
                    -((K + 2.0) / (K + 1.0)) *
                    rup_val(a, rho1, K + 2) *
                    rup_val(c, rho, K + 2);

                if (std::abs(L) <= 1e-14) continue;

                const double center =
                    -(
                        2.0 * K * K +
                        4.0 * K +
                        4.0 -
                        c * c -
                        a * a +
                        4.0 * rho1 * rho1 -
                        4.0 * rho2 * rho2 +
                        4.0 * rho * rho -
                        (
                            K != 0
                                ? (8.0 * a * rho1 * c * rho) / (K * (K + 2.0))
                                : 0.0
                        )
                    );

                const double prev_coeff =
                    (prev_idx < 0)
                        ? 0.0
                        : -K / (K + 1.0) *
                          rup_val(a, rho1, K) *
                          rup_val(c, rho, K);

                const double pR = (PR != nullptr) ? PR[lane] : 0.0;
                const double pI = (PI != nullptr) ? PI[lane] : 0.0;

                OR[lane] =
                    (center * CR[lane] - prev_coeff * pR) /
                    L;

                OI[lane] =
                    (center * CI[lane] - prev_coeff * pI) /
                    L;
            }
        }
    }

    // ------------------------------------------------------------------------
    // Rho-dependent lookup tables.
    // ------------------------------------------------------------------------

    const int maxDJ = tape.max_DJ_needed;
    ws.resize_rho_lookups(maxDJ);

    double* __restrict RHO_sqNRJJ = ws.rho_sq_nrjj.aligned_data();
    double* __restrict RHO_inv_sqNRJJ =
        ws.rho_inv_sq_nrjj.aligned_data();
    double* __restrict RHO_sqNRJ_inv =
        ws.rho_sq_nrj_inv.aligned_data();

    for (int DJ = 0; DJ <= maxDJ; ++DJ) {
        double* __restrict A =
            RHO_sqNRJJ + static_cast<std::size_t>(DJ) * RhoTile;

        double* __restrict B =
            RHO_inv_sqNRJJ + static_cast<std::size_t>(DJ) * RhoTile;

        double* __restrict Cc =
            RHO_sqNRJ_inv + static_cast<std::size_t>(DJ) * RhoTile;

#pragma omp simd
        for (int lane = 0; lane < npt; ++lane) {
            A[lane]  = 0.0;
            B[lane]  = 0.0;
            Cc[lane] = 0.0;
        }
    }

    for (int DJ = c + 1; DJ <= maxDJ; ++DJ) {
        const double J  = 0.5 * DJ;
        const double jj = J - 1.0;

        double* __restrict A =
            RHO_sqNRJJ + static_cast<std::size_t>(DJ) * RhoTile;

        double* __restrict B =
            RHO_inv_sqNRJJ + static_cast<std::size_t>(DJ) * RhoTile;

        double* __restrict Cc =
            RHO_sqNRJ_inv + static_cast<std::size_t>(DJ) * RhoTile;

#pragma omp simd
        for (int lane = 0; lane < npt; ++lane) {
            const double rho = rho_vec[lane];

            const double sqNRJJ =
                std::sqrt((J * J - nu * nu) * (J * J + rho * rho));

            const double sqNRJ =
                (jj >= nu)
                    ? std::sqrt((jj * jj - nu * nu) * (jj * jj + rho * rho))
                    : 0.0;

            const double inv =
                (sqNRJJ > 0.0)
                    ? 1.0 / sqNRJJ
                    : 0.0;

            A[lane]  = sqNRJJ;
            B[lane]  = inv;
            Cc[lane] = sqNRJ * inv;
        }
    }

    for (const TapeChunkZ& ch : tape.chunks) {
        if (ch.kind == KappaTapeZ::CH_B1) {
            execute_branch1_tile(
                tape.branch1,
                ch.begin,
                ch.end,
                rho_vec,
                npt,
                ws,
                RHO_sqNRJJ
            );
        } else if (ch.kind == KappaTapeZ::CH_B2) {
            execute_branch2_tile(
                tape.branch2,
                ch.begin,
                ch.end,
                rho_vec,
                npt,
                ws,
                RHO_sqNRJJ
            );
        } else {
            execute_branch3_tile(
                tape.branch3,
                ch.begin,
                ch.end,
                rho_vec,
                npt,
                ws,
                RHO_inv_sqNRJJ,
                RHO_sqNRJ_inv
            );
        }
    }

}

// ----------------------------------------------------------------------------
// HARVESTER TASKS
// ----------------------------------------------------------------------------

struct HalfTaskZ {
    uint32_t dp_idx;
    int flat_offset;
    int dsum;
    double fact;
};

struct ABTaskZ {
    int a;
    int b;
    int a_pos;
    int b_pos;
    double base_pref;
    std::size_t flat_out_base;
    std::vector<HalfTaskZ> t12;
    std::vector<HalfTaskZ> t34;
};

struct KGroupZ {
    int k12_max = 0;

    std::vector<int> unique_a;
    std::vector<int> unique_b;

    std::vector<ABTaskZ> ab_tasks;

    std::vector<std::vector<int>> ab_indices_for_dk_slot;
};

struct JobHybridZ {
    int group_idx = 0;
    int dk_slot   = 0;
    int Dk12      = 0;

    // Rho quadrature nodes for this k12/job.
    double rhos[V_LEN];

    // Full quadrature weights, ideally already including:
    //   quadrature weight,
    //   rho mapping Jacobian,
    //   measure rho^2 + k^2,
    //   and the 1/2 factor for k12 == 0.
    double quad_weights[V_LEN];
};

struct GroupHarvestIndexZ {
    // Minimal-channel indices for each unique a.
    std::vector<int> min12_idx;
    std::vector<int> min34_idx;
};

template <int RhoTile>
struct EvalBuffersHybridZ {
    std::vector<double> min_R;
    std::vector<double> min_I;

    void resize(int n_a) {
        const std::size_t need =
            static_cast<std::size_t>(n_a) * RhoTile;

        if (min_R.size() < need) {
            min_R.resize(need);
            min_I.resize(need);
        }
    }
};

// ----------------------------------------------------------------------------
// MAIN HARVESTER
// ----------------------------------------------------------------------------

// Optional caller-owned observations, collected by the primary worker inside
// existing parallel regions. Harvest calls sharing this record are sequential.
struct B4HarvestThreadStats {
    std::uint64_t calls = 0;
    int minimum = 0;
    int maximum = 0;
};

template <int FixedIdx, int Rank, int RhoTile>
void harvest_b4_impl(
                int Dj1,
                int Dj2,
                int Dj3,
                int Dj4,
                const std::array<std::size_t,3> Dim_ls,
                std::span<const int> a_vals,
                std::span<const int> b_vals,
                double gamma,
                const std::vector<double>& LF,
                Tensor<double, Rank>& out_tensor,
                const std::array<std::size_t,3> ell0,
                bool reduce_exchange12_harvest,
                sl2cfoamxx::SimplicityConstraint prescription,
                B4HarvestThreadStats* thread_stats)
{
    static_assert(RhoTile == 4 || RhoTile == 8 || RhoTile == 16);
    const std::size_t a_size = a_vals.size();
    const std::size_t b_size = b_vals.size();
    
    const std::size_t out_size =
    (Rank == 5)
    ? Dim_ls[0] *
    Dim_ls[1] *
    Dim_ls[2] *
    a_size *
    b_size
    : Dim_ls[0] *
    Dim_ls[1] *
    Dim_ls[2] *
    b_size;
    
    if (out_size == 0) return;
    
    const double rho1 = simplicity_constraint(gamma, Dj1, prescription);
    const double rho2 = simplicity_constraint(gamma, Dj2, prescription);
    const double rho3 = simplicity_constraint(gamma, Dj3, prescription);
    const double rho4 = simplicity_constraint(gamma, Dj4, prescription);
    
    int Dl1_max;
    int Dl2_max;
    int Dl3_max;
    int Dl4_max;
    
    static_assert(FixedIdx >= 1 && FixedIdx <= 4,
                  "FixedIdx must be in {1,2,3,4}");

    const int off1 = (FixedIdx == 1) ? 0 : 2 * static_cast<int>(ell0[0]);
    const int off2 = (FixedIdx == 1) ? 2 * static_cast<int>(ell0[0])
                   : (FixedIdx == 2) ? 0 : 2 * static_cast<int>(ell0[1]);
    const int off3 = (FixedIdx <= 2) ? 2 * static_cast<int>(ell0[1])
                   : (FixedIdx == 3) ? 0 : 2 * static_cast<int>(ell0[2]);
    const int off4 = (FixedIdx == 4) ? 0 : 2 * static_cast<int>(ell0[2]);

    const int dsum01 = static_cast<int>(ell0[0]);
    const int dsum12 = static_cast<int>(ell0[1] + ell0[2]);
    const int dsum01_pair = static_cast<int>(ell0[0] + ell0[1]);
    const int dsum2 = static_cast<int>(ell0[2]);

    // Optional harvest reduction using the 1<->2 booster symmetry.
    // For FixedIdx=3/4 the first two booster legs are both tensor axes
    // (d1,d2). When they have identical ranges and spins, we only build
    // the upper triangular half d1 <= d2, then complete the tensor by a
    // signed copy after the harvest. This reduces output contraction tasks.
    const bool do_reduce_exchange12 =
        reduce_exchange12_harvest &&
        (FixedIdx == 3 || FixedIdx == 4) &&
        Dj1 == Dj2 &&
        Dim_ls[0] == Dim_ls[1] &&
        ell0[0] == ell0[1];
    
    if constexpr (FixedIdx == 1) {
        Dl1_max = Dj1;
        Dl2_max = Dj2 + off2 + 2 * static_cast<int>(Dim_ls[0] - 1);
        Dl3_max = Dj3 + off3 + 2 * static_cast<int>(Dim_ls[1] - 1);
        Dl4_max = Dj4 + off4 + 2 * static_cast<int>(Dim_ls[2] - 1);
    }
    else if constexpr (FixedIdx == 2) {
        Dl1_max = Dj1 + off1 + 2 * static_cast<int>(Dim_ls[0] - 1);
        Dl2_max = Dj2;
        Dl3_max = Dj3 + off3 + 2 * static_cast<int>(Dim_ls[1] - 1);
        Dl4_max = Dj4 + off4 + 2 * static_cast<int>(Dim_ls[2] - 1);
    }
    else if constexpr (FixedIdx == 3) {
        Dl1_max = Dj1 + off1 + 2 * static_cast<int>(Dim_ls[0] - 1);
        Dl2_max = Dj2 + off2 + 2 * static_cast<int>(Dim_ls[1] - 1);
        Dl3_max = Dj3;
        Dl4_max = Dj4 + off4 + 2 * static_cast<int>(Dim_ls[2] - 1);
    }
    else if constexpr (FixedIdx == 4) {
        Dl1_max = Dj1 + off1 + 2 * static_cast<int>(Dim_ls[0] - 1);
        Dl2_max = Dj2 + off2 + 2 * static_cast<int>(Dim_ls[1] - 1);
        Dl3_max = Dj3 + off3 + 2 * static_cast<int>(Dim_ls[2] - 1);
        Dl4_max = Dj4;
    }

    
    const int DmaxJ12_global = Dl1_max + Dl2_max;
    const int DmaxJ34_global = Dl3_max + Dl4_max;
    
    PureGeomCacheZ geom12;
    geom12.init(Dl1_max, Dl2_max, DmaxJ12_global, Dj1, Dj2);
    
    PureGeomCacheZ geom34;
    geom34.init(Dl3_max, Dl4_max, DmaxJ34_global, Dj3, Dj4);

    KappaSeed3jCacheZ seed_cache12;
    seed_cache12.init(Dj1, Dj2, DmaxJ12_global);
    
    KappaSeed3jCacheZ seed_cache34;
    seed_cache34.init(Dj3, Dj4, DmaxJ34_global);
    
    
    const int ws12_DJ1_min = Dj1;
    const int ws12_DJ2_min = Dj2;
    const int ws12_DJ_min  = DmaxJ12_global & 1;
    const int ws12_dim2 = (Dl2_max - ws12_DJ2_min) / 2 + 1;
    const int ws12_dim3 = (DmaxJ12_global - ws12_DJ_min) / 2 + 1;
    const int ws12_stride2 = ws12_dim3 + 2;
    const int ws12_stride1 = (ws12_dim2 + 2) * ws12_stride2;
    
    const int ws34_DJ1_min = Dj3;
    const int ws34_DJ2_min = Dj4;
    const int ws34_DJ_min  = DmaxJ34_global & 1;
    const int ws34_dim2 = (Dl4_max - ws34_DJ2_min) / 2 + 1;
    const int ws34_dim3 = (DmaxJ34_global - ws34_DJ_min) / 2 + 1;
    const int ws34_stride2 = ws34_dim3 + 2;
    const int ws34_stride1 = (ws34_dim2 + 2) * ws34_stride2;
    
    std::vector<KGroupZ> groups;
    groups.reserve(static_cast<std::size_t>(std::min(a_vals.empty() ? 0 : a_vals.back(), b_vals.empty() ? 0 : b_vals.back()) / 2 + 2));
    
    auto get_group_index = [&](int k12_max) -> int {
        for (int i = 0; i < static_cast<int>(groups.size()); ++i) {
            if (groups[i].k12_max == k12_max) return i;
        }
        groups.push_back(KGroupZ{k12_max, {}, {}, {}, {}});
        return static_cast<int>(groups.size()) - 1;
    };
    
    for (std::size_t a_idx = 0; a_idx < a_size; ++a_idx) {
        for (std::size_t b_idx = 0; b_idx < b_size; ++b_idx) {
            const int a = a_vals[a_idx];
            const int b = b_vals[b_idx];
            
            KGroupZ& g = groups[get_group_index(std::min(a, b))];
            
            ABTaskZ ab;
            ab.a = a;
            ab.b = b;
            ab.a_pos = find_or_push_int(g.unique_a, a);
            ab.b_pos = find_or_push_int(g.unique_b, b);
            ab.flat_out_base = flat_tensor_index<Rank>(Dim_ls, a_size, 0, 0, 0, a_idx, b_idx);
            
            ab.base_pref =
            (((a - b) / 2) & 1 ? -1.0 : 1.0) *
            B4_PI * B4_PI / 32.0 *
            (a + 1.0) * (b + 1.0) *
            std::sqrt((a + 1.0) * (b + 1.0)) *
            std::sqrt((Dj1 + 1.0) * (Dj2 + 1.0) * (Dj3 + 1.0) * (Dj4 + 1.0)) *
            z_to_k_factor(Dj1, Dj2, a, Dj1, Dj2) *
            z_to_k_factor(Dj3, Dj4, a, Dj3, Dj4);
            
            if constexpr (FixedIdx == 1 || FixedIdx == 2) {
                for (int d1 = 0; d1 < static_cast<int>(Dim_ls[0]); ++d1) {
                    int Dl1 = (FixedIdx == 1) ? Dj1 : Dj1 + off1 + 2 * d1;
                    int Dl2 = (FixedIdx == 1) ? Dj2 + off2 + 2 * d1 : Dj2;
                    if (b < iabs_int(Dl1 - Dl2) || b > Dl1 + Dl2) continue;
                    if (Dj1 == Dj2 && Dl1 == Dl2 && (((b - a) / 2) & 1)) continue;
                    
                    HalfTaskZ t;
                    t.dp_idx = ws_flat_idx_precomputed(Dl1, Dl2, b, ws12_DJ1_min, ws12_DJ2_min, ws12_DJ_min, ws12_stride1, ws12_stride2);
                    t.flat_offset = d1;
                    t.dsum = dsum01 + d1;
                    t.fact = std::sqrt((Dl1 + 1.0) * (Dl2 + 1.0)) * z_to_k_factor(Dl1, Dl2, b, Dj1, Dj2);
                    ab.t12.push_back(t);
                }
                for (int d2 = 0; d2 < static_cast<int>(Dim_ls[1]); ++d2) {
                    for (int d3 = 0; d3 < static_cast<int>(Dim_ls[2]); ++d3) {
                        int Dl3 = Dj3 + off3 + 2 * d2;
                        int Dl4 = Dj4 + off4 + 2 * d3;
                        if (b < iabs_int(Dl3 - Dl4) || b > Dl3 + Dl4) continue;
                        if (Dj3 == Dj4 && Dl3 == Dl4 && (((b - a) / 2) & 1)) continue;
                        
                        HalfTaskZ t;
                        t.dp_idx = ws_flat_idx_precomputed(Dl3, Dl4, b, ws34_DJ1_min, ws34_DJ2_min, ws34_DJ_min, ws34_stride1, ws34_stride2);
                        t.flat_offset = Dim_ls[0] * d2 + Dim_ls[0] * Dim_ls[1] * d3;
                        t.dsum = dsum12 + d2 + d3;
                        t.fact = std::sqrt((Dl3 + 1.0) * (Dl4 + 1.0)) * z_to_k_factor(Dl3, Dl4, b, Dj3, Dj4);
                        ab.t34.push_back(t);
                    }
                }
            } else {
                for (int d1 = 0; d1 < static_cast<int>(Dim_ls[0]); ++d1) {
                    for (int d2 = 0; d2 < static_cast<int>(Dim_ls[1]); ++d2) {
                        if (do_reduce_exchange12 && d2 < d1) continue;

                        int Dl1 = Dj1 + off1 + 2 * d1;
                        int Dl2 = Dj2 + off2 + 2 * d2;
                        if (b < iabs_int(Dl1 - Dl2) || b > Dl1 + Dl2) continue;
                        if (Dj1 == Dj2 && Dl1 == Dl2 && (((b - a) / 2) & 1)) continue;
                        
                        HalfTaskZ t;
                        t.dp_idx = ws_flat_idx_precomputed(Dl1, Dl2, b, ws12_DJ1_min, ws12_DJ2_min, ws12_DJ_min, ws12_stride1, ws12_stride2);
                        t.flat_offset = d1 + Dim_ls[0] * d2;
                        t.dsum = dsum01_pair + d1 + d2;
                        t.fact = std::sqrt((Dl1 + 1.0) * (Dl2 + 1.0)) * z_to_k_factor(Dl1, Dl2, b, Dj1, Dj2);
                        ab.t12.push_back(t);
                    }
                }
                for (int d3 = 0; d3 < static_cast<int>(Dim_ls[2]); ++d3) {
                    int Dl3 = (FixedIdx == 3) ? Dj3 : Dj3 + off3 + 2 * d3;
                    int Dl4 = (FixedIdx == 3) ? Dj4 + off4 + 2 * d3 : Dj4;
                    if (b < iabs_int(Dl3 - Dl4) || b > Dl3 + Dl4) continue;
                    if (Dj3 == Dj4 && Dl3 == Dl4 && (((b - a) / 2) & 1)) continue;
                    
                    HalfTaskZ t;
                    t.dp_idx = ws_flat_idx_precomputed(Dl3, Dl4, b, ws34_DJ1_min, ws34_DJ2_min, ws34_DJ_min, ws34_stride1, ws34_stride2);
                    t.flat_offset = Dim_ls[0] * Dim_ls[1] * d3;
                    t.dsum = dsum2 + d3;
                    t.fact = std::sqrt((Dl3 + 1.0) * (Dl4 + 1.0)) * z_to_k_factor(Dl3, Dl4, b, Dj3, Dj4);
                    ab.t34.push_back(t);
                }
            }
            
            if (!ab.t12.empty() && !ab.t34.empty()) {
                g.ab_tasks.push_back(std::move(ab));
            }
        }
    }
    
    const bool symmetric_pair = (Dj1 == Dj2 || Dj3 == Dj4);
    int max_Dk12 = 0;
    
    for (KGroupZ& g : groups) {
        max_Dk12 = std::max(max_Dk12, g.k12_max);
        const int start_Dk = g.k12_max & 1;
        const int nslots = (g.k12_max - start_Dk) / 2 + 1;
        g.ab_indices_for_dk_slot.assign(nslots, {});
        for (int i = 0; i < static_cast<int>(g.ab_tasks.size()); ++i) {
            for (int slot = 0; slot < nslots; ++slot) {
                const int Dk = start_Dk + 2 * slot;
                if (symmetric_pair && (((g.ab_tasks[i].a + Dk) / 2) & 1)) continue;
                g.ab_indices_for_dk_slot[slot].push_back(i);
            }
        }
    }
    
    std::vector<KappaTapeZ> tapes12(max_Dk12 + 1);
    std::vector<KappaTapeZ> tapes34(max_Dk12 + 1);
#pragma omp parallel for schedule(dynamic, 1)
    for (int Dk = 0; Dk <= max_Dk12; ++Dk) {
        tapes12[Dk].compile(rho1, Dj1, Dl1_max, rho2, Dj2, Dl2_max, Dk, DmaxJ12_global, geom12);
        tapes34[Dk].compile(rho3, Dj3, Dl3_max, rho4, Dj4, Dl4_max, Dk, DmaxJ34_global, geom34);
    }
    
    std::vector<JobHybridZ> jobs;
    const auto& rule = get_split_gl64_rule();
    const double Jtot = 0.5 * (Dj1 + Dj2 + Dj3 + Dj4);
    
    for (int gidx = 0; gidx < static_cast<int>(groups.size()); ++gidx) {
        KGroupZ& g = groups[gidx];
        const double W = std::sqrt((gamma > 1.0) ? (Jtot + 4.0) * gamma : (Jtot + 4.0));
        for (int slot = 0; slot < static_cast<int>(g.ab_indices_for_dk_slot.size()); ++slot) {
            if (g.ab_indices_for_dk_slot[slot].empty()) continue;
            JobHybridZ job{};
            job.group_idx = gidx;
            job.dk_slot = slot;
            job.Dk12 = (g.k12_max & 1) + 2 * slot;
            const double k = 0.5 * job.Dk12;
            const double k_symmetry_factor = (job.Dk12 == 0) ? 0.5 : 1.0;
            for (int pt = 0; pt < V_LEN; ++pt) {
                const double xi = rule.x[pt];
                const double inv_comp = 1.0 / (1.0 - xi * xi);
                const double inv_sqrt = std::sqrt(inv_comp);
                job.rhos[pt] = simplicity_constraint(
                    gamma, g.k12_max, prescription) + W * xi * inv_sqrt;
                const double measure = job.rhos[pt] * job.rhos[pt] + k * k;
                job.quad_weights[pt] = rule.w[pt] * W * inv_comp * inv_sqrt * measure * k_symmetry_factor;
            }
            jobs.push_back(job);
        }
    }
    
    if (jobs.empty()) return;
    
    std::vector<GroupHarvestIndexZ> group_indices(groups.size());
    for (std::size_t gidx = 0; gidx < groups.size(); ++gidx) {
        const KGroupZ& g = groups[gidx];
        GroupHarvestIndexZ& gi = group_indices[gidx];
        gi.min12_idx.resize(g.unique_a.size());
        gi.min34_idx.resize(g.unique_a.size());
        for (std::size_t ia = 0; ia < g.unique_a.size(); ++ia) {
            const int a = g.unique_a[ia];
            gi.min12_idx[ia] = ws_flat_idx_precomputed(Dj1, Dj2, a, ws12_DJ1_min, ws12_DJ2_min, ws12_DJ_min, ws12_stride1, ws12_stride2);
            gi.min34_idx[ia] = ws_flat_idx_precomputed(Dj3, Dj4, a, ws34_DJ1_min, ws34_DJ2_min, ws34_DJ_min, ws34_stride1, ws34_stride2);
        }
    }
    
    std::vector<std::vector<int>> jobs_by_group(groups.size());
    for (int j_idx = 0; j_idx < static_cast<int>(jobs.size()); ++j_idx) {
        jobs_by_group[jobs[j_idx].group_idx].push_back(j_idx);
    }
    
    std::vector<std::vector<OutputZeroBlockZ>> zero_blocks_by_group(groups.size());
    auto find_group_index_for_kmax = [&](int k12_max) -> int {
        for (int gidx = 0; gidx < static_cast<int>(groups.size()); ++gidx) {
            if (groups[gidx].k12_max == k12_max) return gidx;
        }
        return -1;
    };
    
    const std::size_t d_block = Dim_ls[0] * Dim_ls[1] * Dim_ls[2];
    
    if constexpr (Rank == 5) {
        for (std::size_t a_idx = 0; a_idx < a_size; ++a_idx) {
            for (std::size_t b_idx = 0; b_idx < b_size; ++b_idx) {
                const int gidx = find_group_index_for_kmax(std::min(a_vals[a_idx], b_vals[b_idx]));
                if (gidx < 0) continue;
                const std::size_t block_begin = d_block * (a_idx + a_size * b_idx);
                zero_blocks_by_group[gidx].push_back({block_begin, block_begin + d_block});
            }
        }
    } else {
        const int a_fixed = a_vals.empty() ? 0 : a_vals[0];
        for (std::size_t b_idx = 0; b_idx < b_size; ++b_idx) {
            const int gidx = find_group_index_for_kmax(std::min(a_fixed, b_vals[b_idx]));
            if (gidx < 0) continue;
            const std::size_t block_begin = d_block * b_idx;
            zero_blocks_by_group[gidx].push_back({block_begin, block_begin + d_block});
        }
    }
    
    double* __restrict dst = out_tensor.data();
    std::size_t total_group_work = 0;
    std::vector<std::size_t> group_work(groups.size(), 0);

#if B4_ENABLE_WORK_DIAGNOSTICS
    std::size_t total_ab_tasks = 0;
    std::size_t total_unique_a_refs = 0;
    std::size_t total_unique_b_refs = 0;
    std::size_t total_t12_entries = 0;
    std::size_t total_t34_entries = 0;
    std::size_t total_t12_t34_pairs_once = 0;
    std::size_t max_ab_t12 = 0;
    std::size_t max_ab_t34 = 0;
    std::size_t nonempty_groups = 0;
    std::size_t max_group_ab_tasks = 0;
#endif

    for (int gidx = 0; gidx < static_cast<int>(groups.size()); ++gidx) {
        const KGroupZ& g = groups[gidx];

#if B4_ENABLE_WORK_DIAGNOSTICS
        total_ab_tasks += g.ab_tasks.size();
        total_unique_a_refs += g.unique_a.size();
        total_unique_b_refs += g.unique_b.size();
        if (!g.ab_tasks.empty()) ++nonempty_groups;
        max_group_ab_tasks = std::max(max_group_ab_tasks, g.ab_tasks.size());

        for (const ABTaskZ& ab : g.ab_tasks) {
            total_t12_entries += ab.t12.size();
            total_t34_entries += ab.t34.size();
            total_t12_t34_pairs_once += ab.t12.size() * ab.t34.size();
            max_ab_t12 = std::max(max_ab_t12, ab.t12.size());
            max_ab_t34 = std::max(max_ab_t34, ab.t34.size());
        }
#endif

        std::size_t work = 0;
        for (int j_idx : jobs_by_group[gidx]) {
            const JobHybridZ& job = jobs[j_idx];
            for (int ab_idx : g.ab_indices_for_dk_slot[job.dk_slot]) {
                work += g.ab_tasks[ab_idx].t12.size() * g.ab_tasks[ab_idx].t34.size();
            }
        }
        group_work[gidx] = work;
    }
    
    int useful_group_count = 0;
    for (std::size_t w : group_work) {
        total_group_work += w;
        useful_group_count += (w != 0);
    }

#if B4_ENABLE_WORK_DIAGNOSTICS
    std::size_t nonzero_work_groups = 0;
    std::size_t max_group_work = 0;
    std::size_t min_nonzero_group_work = std::numeric_limits<std::size_t>::max();

    for (std::size_t w : group_work) {
        if (w == 0) continue;
        ++nonzero_work_groups;
        max_group_work = std::max(max_group_work, w);
        min_nonzero_group_work = std::min(min_nonzero_group_work, w);
    }
    if (nonzero_work_groups == 0) {
        min_nonzero_group_work = 0;
    }

    std::size_t total_scheduled_ab_refs = 0;
    std::size_t max_scheduled_ab_refs_per_job = 0;
    std::size_t nonempty_jobs = 0;

    for (const JobHybridZ& job : jobs) {
        const KGroupZ& g = groups[job.group_idx];
        const std::size_t refs = g.ab_indices_for_dk_slot[job.dk_slot].size();
        if (refs != 0) ++nonempty_jobs;
        total_scheduled_ab_refs += refs;
        max_scheduled_ab_refs_per_job =
            std::max(max_scheduled_ab_refs_per_job, refs);
    }
#endif
    
    std::vector<int> group_order(groups.size());
    for (int gidx = 0; gidx < static_cast<int>(groups.size()); ++gidx) group_order[gidx] = gidx;
    
    std::sort(group_order.begin(), group_order.end(),
              [&](int lhs, int rhs) {
        const std::size_t wl = group_work[lhs];
        const std::size_t wr = group_work[rhs];
        if (wl != wr) return wl > wr;
        return lhs < rhs;
    }
              );
    
    const int nthreads = b4_choose_num_threads_for_group_work(useful_group_count, total_group_work);
    
#if B4_ENABLE_DIAGNOSTICS
    std::cerr << "Nb threads: " << nthreads << "\n";
#endif
    
#if B4_ENABLE_TIMING
    std::vector<ThreadTimingZ> timings(nthreads);
#endif
    
    omp_set_dynamic(0);
    
    const std::size_t max_t12 =
    (FixedIdx == 1 || FixedIdx == 2)
    ? Dim_ls[0]
    : Dim_ls[0] * Dim_ls[1];
    
#pragma omp parallel num_threads(nthreads)
    {
        if (thread_stats != nullptr && omp_get_thread_num() == 0) {
            const int observed = omp_get_num_threads();
            thread_stats->minimum = thread_stats->calls == 0
                ? observed : std::min(thread_stats->minimum, observed);
            thread_stats->maximum = std::max(thread_stats->maximum, observed);
            ++thread_stats->calls;
        }
#if B4_ENABLE_TIMING
        const int tid = omp_get_thread_num();
#endif

        std::vector<RhoComplexTileZ<RhoTile>> pre_t12_buffer(max_t12);
        
        ThreadWorkspaceTileZ<RhoTile> ws12;
        ws12.init_dims(ws12_DJ1_min, ws12_DJ2_min, ws12_DJ_min);
        ws12.resize((Dl1_max - ws12_DJ1_min) / 2 + 1, (Dl2_max - ws12_DJ2_min) / 2 + 1, (DmaxJ12_global - ws12_DJ_min) / 2 + 1);
        
        ThreadWorkspaceTileZ<RhoTile> ws34;
        ws34.init_dims(ws34_DJ1_min, ws34_DJ2_min, ws34_DJ_min);
        ws34.resize((Dl3_max - ws34_DJ1_min) / 2 + 1, (Dl4_max - ws34_DJ2_min) / 2 + 1, (DmaxJ34_global - ws34_DJ_min) / 2 + 1);
        
        EvalBuffersHybridZ<RhoTile> buf;
        
#pragma omp for schedule(dynamic, 1)
        for (int ord = 0; ord < static_cast<int>(group_order.size()); ++ord) {
            const int gidx = group_order[ord];
            const KGroupZ& g = groups[gidx];
            const GroupHarvestIndexZ& gi = group_indices[gidx];
            
            for (const OutputZeroBlockZ& zb : zero_blocks_by_group[gidx]) {
                std::fill(dst + zb.begin, dst + zb.end, 0.0);
            }
            
            buf.resize(static_cast<int>(g.unique_a.size()));
            
            for (int j_idx : jobs_by_group[gidx]) {
                const JobHybridZ& job = jobs[j_idx];
                const KappaTapeZ& tape12 = tapes12[job.Dk12];
                const KappaTapeZ& tape34 = tapes34[job.Dk12];
                
#if B4_ENABLE_TIMING
                double job_kappa12_time = 0.0;
                double job_kappa34_time = 0.0;
                double job_contract_time = 0.0;
#endif
                
                for (int pt0 = 0; pt0 < V_LEN; pt0 += RhoTile) {
                    constexpr int npt = RhoTile;
                    const double* __restrict rho_tile = job.rhos + pt0;
                    
                    alignas(64) double W_ARR[RhoTile];
                    for (int lane = 0; lane < npt; ++lane) W_ARR[lane] = job.quad_weights[pt0 + lane];
                    
#if B4_ENABLE_TIMING
                    const double t0 = omp_get_wtime();
#endif
                    
                    KappaFastZ_ExecuteTapeTile(rho1, Dj1, Dl1_max, rho2, Dj2, Dl2_max, rho_tile, npt, job.Dk12, DmaxJ12_global, LF, ws12, tape12, &seed_cache12);
                    
#if B4_ENABLE_TIMING
                    const double t1 = omp_get_wtime();
#endif
                    
                    KappaFastZ_ExecuteTapeTile(rho3, Dj3, Dl3_max, rho4, Dj4, Dl4_max, rho_tile, npt, job.Dk12, DmaxJ34_global, LF, ws34, tape34, &seed_cache34);
                    
#if B4_ENABLE_TIMING
                    const double t2 = omp_get_wtime();
#endif
                    
                    for (int ia = 0; ia < static_cast<int>(g.unique_a.size()); ++ia) {
                        const int idx12 = gi.min12_idx[ia];
                        const int idx34 = gi.min34_idx[ia];
                        const double* __restrict z12r = ws12.cell_R(idx12);
                        const double* __restrict z12i = ws12.cell_I(idx12);
                        const double* __restrict z34r = ws34.cell_R(idx34);
                        const double* __restrict z34i = ws34.cell_I(idx34);
                        double* __restrict mR = buf.min_R.data() + ia * RhoTile;
                        double* __restrict mI = buf.min_I.data() + ia * RhoTile;
#pragma omp simd
                        for (int lane = 0; lane < npt; ++lane) {
                            mR[lane] = z12r[lane] * z34r[lane] - z12i[lane] * z34i[lane];
                            mI[lane] = -(z12r[lane] * z34i[lane] + z12i[lane] * z34r[lane]);
                        }
                    }
                    
                    for (int ab_idx : g.ab_indices_for_dk_slot[job.dk_slot]) {
                        const ABTaskZ& ab = g.ab_tasks[ab_idx];
                        
                        const double* __restrict mR = buf.min_R.data() + ab.a_pos * RhoTile;
                        const double* __restrict mI = buf.min_I.data() + ab.a_pos * RhoTile;
                        
                        alignas(64) double w_mR[RhoTile];
                        alignas(64) double w_mI[RhoTile];
                        
                        #pragma omp simd
                        for (int lane = 0; lane < RhoTile; ++lane) {
                            double w_base = W_ARR[lane] * ab.base_pref;
                            w_mR[lane] = w_base * mR[lane];
                            w_mI[lane] = w_base * mI[lane];
                        }
                        
                        if (pre_t12_buffer.size() < ab.t12.size()) {
                            pre_t12_buffer.resize(ab.t12.size());
                        }
                        
                        for (size_t i = 0; i < ab.t12.size(); ++i) {
                            const HalfTaskZ& t12 = ab.t12[i];
                            const double* r12 = ws12.cell_R(t12.dp_idx);
                            const double* i12 = ws12.cell_I(t12.dp_idx);
                            const double f12 = t12.fact;
                            
                            switch (t12.dsum & 3) {
                                case 0:
                                    #pragma omp simd
                                    for (int lane = 0; lane < RhoTile; ++lane) { pre_t12_buffer[i].R[lane] =  r12[lane] * f12; pre_t12_buffer[i].I[lane] = -i12[lane] * f12; }
                                    break;
                                case 1:
                                    #pragma omp simd
                                    for (int lane = 0; lane < RhoTile; ++lane) { pre_t12_buffer[i].R[lane] =  i12[lane] * f12; pre_t12_buffer[i].I[lane] =  r12[lane] * f12; }
                                    break;
                                case 2:
                                    #pragma omp simd
                                    for (int lane = 0; lane < RhoTile; ++lane) { pre_t12_buffer[i].R[lane] = -r12[lane] * f12; pre_t12_buffer[i].I[lane] =  i12[lane] * f12; }
                                    break;
                                default:
                                    #pragma omp simd
                                    for (int lane = 0; lane < RhoTile; ++lane) { pre_t12_buffer[i].R[lane] = -i12[lane] * f12; pre_t12_buffer[i].I[lane] = -r12[lane] * f12; }
                                    break;
                            }
                        }
                        
                        alignas(64) double P_R[RhoTile];
                        alignas(64) double P_I[RhoTile];
                        
                        for (const HalfTaskZ& t34 : ab.t34) {
                            const double* __restrict r34 = ws34.cell_R(t34.dp_idx);
                            const double* __restrict i34 = ws34.cell_I(t34.dp_idx);
                            const double f34 = t34.fact;
                            
                            switch (t34.dsum & 3) {
                                case 0:
#pragma omp simd
                                    for (int lane = 0; lane < RhoTile; ++lane) {
                                        double wfR = w_mR[lane] * f34, wfI = w_mI[lane] * f34;
                                        P_R[lane] = r34[lane] * wfR - i34[lane] * wfI;
                                        P_I[lane] = r34[lane] * wfI + i34[lane] * wfR;
                                    }
                                    break;
                                case 1:
#pragma omp simd
                                    for (int lane = 0; lane < RhoTile; ++lane) {
                                        double wfR = w_mR[lane] * f34, wfI = w_mI[lane] * f34;
                                        P_R[lane] = i34[lane] * wfR + r34[lane] * wfI;
                                        P_I[lane] = i34[lane] * wfI - r34[lane] * wfR;
                                    }
                                    break;
                                case 2:
#pragma omp simd
                                    for (int lane = 0; lane < RhoTile; ++lane) {
                                        double n_wfR = -w_mR[lane] * f34, n_wfI = -w_mI[lane] * f34;
                                        P_R[lane] = r34[lane] * n_wfR - i34[lane] * n_wfI;
                                        P_I[lane] = r34[lane] * n_wfI + i34[lane] * n_wfR;
                                    }
                                    break;
                                default:
#pragma omp simd
                                    for (int lane = 0; lane < RhoTile; ++lane) {
                                        double wfR = w_mR[lane] * f34, wfI = w_mI[lane] * f34;
                                        P_R[lane] = -i34[lane] * wfR - r34[lane] * wfI;
                                        P_I[lane] = -i34[lane] * wfI + r34[lane] * wfR;
                                    }
                                    break;
                            }
                            
                            const std::size_t out_base34 = ab.flat_out_base + t34.flat_offset;
                            
                            for (size_t i = 0; i < ab.t12.size(); ++i) {
                                double sum = 0.0;
#pragma omp simd reduction(+:sum)
                                for (int lane = 0; lane < RhoTile; ++lane) {
                                    sum += P_R[lane] * pre_t12_buffer[i].R[lane] + P_I[lane] * pre_t12_buffer[i].I[lane];
                                }
                                dst[out_base34 + ab.t12[i].flat_offset] += sum;
                            }
                        }
                    }
                    
#if B4_ENABLE_TIMING
                    const double t3 = omp_get_wtime();
                    job_kappa12_time += t1 - t0;
                    job_kappa34_time += t2 - t1;
                    job_contract_time += t3 - t2;
                    timings[tid].tiles += 1;
#endif
                }
#if B4_ENABLE_TIMING
                const double t_clear_start = omp_get_wtime();
#endif
                
                
#if B4_ENABLE_TIMING
                ThreadTimingZ& tm = timings[tid];
                tm.kappa12_seconds += job_kappa12_time;
                tm.kappa34_seconds += job_kappa34_time;
                tm.contract_seconds += job_contract_time;
                tm.clear_seconds += omp_get_wtime() - t_clear_start;
#endif
            }
        }
    }
    
#if B4_ENABLE_TIMING
    double sum_kappa12 = 0, sum_kappa34 = 0, sum_contract = 0, sum_clear = 0;
    double max_kappa12 = 0, max_kappa34 = 0, max_contract = 0, max_clear = 0;
    std::uint64_t total_tiles = 0;
    for (const ThreadTimingZ& tm : timings) {
        sum_kappa12 += tm.kappa12_seconds; sum_kappa34 += tm.kappa34_seconds;
        sum_contract += tm.contract_seconds; sum_clear += tm.clear_seconds;
        max_kappa12 = std::max(max_kappa12, tm.kappa12_seconds);
        max_kappa34 = std::max(max_kappa34, tm.kappa34_seconds);
        max_contract = std::max(max_contract, tm.contract_seconds);
        max_clear = std::max(max_clear, tm.clear_seconds);
        total_tiles += tm.tiles;
    }
    std::cerr << "Timing diagnostics for harvest_b4<" << FixedIdx << ", " << Rank << ">\n"
    << "  tiles:              " << total_tiles << "\n";

#if B4_ENABLE_WORK_DIAGNOSTICS
    b4_print_int_span_stats("a_vals", a_vals);
    b4_print_int_span_stats("b_vals", b_vals);
    std::cerr
    << "  groups:             " << groups.size() << "\n"
    << "  nonempty groups:    " << nonempty_groups << "\n"
    << "  jobs:               " << jobs.size() << "\n"
    << "  nonempty jobs:      " << nonempty_jobs << "\n"
    << "  ab tasks:           " << total_ab_tasks << "\n"
    << "  scheduled ab refs:  " << total_scheduled_ab_refs << "\n"
    << "  max ab refs/job:    " << max_scheduled_ab_refs_per_job << "\n"
    << "  unique a refs sum:  " << total_unique_a_refs << "\n"
    << "  unique b refs sum:  " << total_unique_b_refs << "\n"
    << "  total t12 entries:  " << total_t12_entries << "\n"
    << "  total t34 entries:  " << total_t34_entries << "\n"
    << "  ab t12*t34 once:    " << total_t12_t34_pairs_once << "\n"
    << "  scheduled work:     " << total_group_work << "\n"
    << "  min group work:     " << min_nonzero_group_work << "\n"
    << "  max group work:     " << max_group_work << "\n"
    << "  max group ab tasks: " << max_group_ab_tasks << "\n"
    << "  max |t12| per ab:   " << max_ab_t12 << "\n"
    << "  max |t34| per ab:   " << max_ab_t34 << "\n";
#endif

    std::cerr
    << "  kappa12 CPU-sec:    " << sum_kappa12 << " | max-thread-sec: " << max_kappa12 << "\n"
    << "  kappa34 CPU-sec:    " << sum_kappa34 << " | max-thread-sec: " << max_kappa34 << "\n"
    << "  contraction CPU-sec:" << sum_contract << " | max-thread-sec: " << max_contract << "\n"
    << "  clearing CPU-sec:   " << sum_clear << " | max-thread-sec: " << max_clear << "\n";
#endif
}

template <int FixedIdx, int Rank>
void harvest_b4(
    int Dj1,
    int Dj2,
    int Dj3,
    int Dj4,
    const std::array<std::size_t,3> Dim_ls,
    std::span<const int> a_vals,
    std::span<const int> b_vals,
    double gamma,
    const std::vector<double>& LF,
    Tensor<double, Rank>& out_tensor,
    const std::array<std::size_t,3> ell0 = {0,0,0},
    bool reduce_exchange12_harvest = false,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0,
    B4HarvestThreadStats* thread_stats = nullptr)
{
    const int resolved = b4_resolve_rho_tile(rho_tile);
    if (resolved == 4) {
        return harvest_b4_impl<FixedIdx, Rank, 4>(
            Dj1, Dj2, Dj3, Dj4, Dim_ls, a_vals, b_vals, gamma, LF,
            out_tensor, ell0, reduce_exchange12_harvest, prescription, thread_stats);
    }
    if (resolved == 8) {
        return harvest_b4_impl<FixedIdx, Rank, 8>(
            Dj1, Dj2, Dj3, Dj4, Dim_ls, a_vals, b_vals, gamma, LF,
            out_tensor, ell0, reduce_exchange12_harvest, prescription, thread_stats);
    }
    return harvest_b4_impl<FixedIdx, Rank, 16>(
        Dj1, Dj2, Dj3, Dj4, Dim_ls, a_vals, b_vals, gamma, LF,
        out_tensor, ell0, reduce_exchange12_harvest, prescription, thread_stats);
}






// ----------------------------------------------------------------------------
// SPECIALIZED RANK 4 HARVESTER
// ----------------------------------------------------------------------------

struct ABTaskZ_Rank4 {
    int b;
    double base_pref;
    std::size_t flat_out_base;
    std::vector<HalfTaskZ> t12;
    std::vector<HalfTaskZ> t34;
};

struct KGroupZ_Rank4 {
    int k12_max = 0;
    std::vector<ABTaskZ_Rank4> ab_tasks;
    std::vector<std::vector<int>> ab_indices_for_dk_slot;
};

struct GroupHarvestIndexZ_Rank4 {
    int min12_idx;
    int min34_idx;
};

template <int FixedIdx, int RhoTile>
void harvest_b4_rank4_impl(
    int Dj1, int Dj2, int Dj3, int Dj4,
                      const std::array<std::size_t,3> Dim_ls, int a, std::span<const int> b_vals,
    double gamma, const std::vector<double>& LF,
    Tensor<double, 4>& out_tensor,
    const std::array<std::size_t,3> ell0,
    bool reduce_exchange12_harvest,
    sl2cfoamxx::SimplicityConstraint prescription)
{
    static_assert(RhoTile == 4 || RhoTile == 8 || RhoTile == 16,
                  "RhoTile must be one of the compiled booster tile widths");
    //[PROFILER] Start Timer
    #if B4_ENABLE_TIMING
    double t_start = omp_get_wtime();
    #endif

    const std::size_t b_size = b_vals.size();
    
    const std::size_t out_size =
    Dim_ls[0] * Dim_ls[1] * Dim_ls[2] * b_size;
    
    if (out_size == 0) return;

    // Fast zeroing (OpenMP thread spin-up takes longer than zeroing 416KB, do it sequentially)
    double* __restrict dst = out_tensor.data();
    std::fill(dst, dst + out_size, 0.0);
    
    const double rho1 = simplicity_constraint(gamma, Dj1, prescription);
    const double rho2 = simplicity_constraint(gamma, Dj2, prescription);
    const double rho3 = simplicity_constraint(gamma, Dj3, prescription);
    const double rho4 = simplicity_constraint(gamma, Dj4, prescription);
    
    int Dl1_max;
    int Dl2_max;
    int Dl3_max;
    int Dl4_max;
    
    static_assert(FixedIdx >= 1 && FixedIdx <= 4,
                  "FixedIdx must be in {1,2,3,4}");

    const int off1 = (FixedIdx == 1) ? 0 : 2 * static_cast<int>(ell0[0]);
    const int off2 = (FixedIdx == 1) ? 2 * static_cast<int>(ell0[0])
                   : (FixedIdx == 2) ? 0 : 2 * static_cast<int>(ell0[1]);
    const int off3 = (FixedIdx <= 2) ? 2 * static_cast<int>(ell0[1])
                   : (FixedIdx == 3) ? 0 : 2 * static_cast<int>(ell0[2]);
    const int off4 = (FixedIdx == 4) ? 0 : 2 * static_cast<int>(ell0[2]);

    const int dsum01 = static_cast<int>(ell0[0]);
    const int dsum12 = static_cast<int>(ell0[1] + ell0[2]);
    const int dsum01_pair = static_cast<int>(ell0[0] + ell0[1]);
    const int dsum2 = static_cast<int>(ell0[2]);

    // Optional harvest reduction using the 1<->2 booster symmetry.
    // For FixedIdx=3/4 the first two booster legs are both tensor axes
    // (d1,d2). When they have identical ranges and spins, we only build
    // the upper triangular half d1 <= d2, then complete the tensor by a
    // signed copy after the harvest. This reduces output contraction tasks.
    const bool do_reduce_exchange12 =
        reduce_exchange12_harvest &&
        (FixedIdx == 3 || FixedIdx == 4) &&
        Dj1 == Dj2 &&
        Dim_ls[0] == Dim_ls[1] &&
        ell0[0] == ell0[1];
    
    if constexpr (FixedIdx == 1) {
        Dl1_max = Dj1;
        Dl2_max = Dj2 + off2 + 2 * static_cast<int>(Dim_ls[0] - 1);
        Dl3_max = Dj3 + off3 + 2 * static_cast<int>(Dim_ls[1] - 1);
        Dl4_max = Dj4 + off4 + 2 * static_cast<int>(Dim_ls[2] - 1);
    }
    else if constexpr (FixedIdx == 2) {
        Dl1_max = Dj1 + off1 + 2 * static_cast<int>(Dim_ls[0] - 1);
        Dl2_max = Dj2;
        Dl3_max = Dj3 + off3 + 2 * static_cast<int>(Dim_ls[1] - 1);
        Dl4_max = Dj4 + off4 + 2 * static_cast<int>(Dim_ls[2] - 1);
    }
    else if constexpr (FixedIdx == 3) {
        Dl1_max = Dj1 + off1 + 2 * static_cast<int>(Dim_ls[0] - 1);
        Dl2_max = Dj2 + off2 + 2 * static_cast<int>(Dim_ls[1] - 1);
        Dl3_max = Dj3;
        Dl4_max = Dj4 + off4 + 2 * static_cast<int>(Dim_ls[2] - 1);
    }
    else if constexpr (FixedIdx == 4) {
        Dl1_max = Dj1 + off1 + 2 * static_cast<int>(Dim_ls[0] - 1);
        Dl2_max = Dj2 + off2 + 2 * static_cast<int>(Dim_ls[1] - 1);
        Dl3_max = Dj3 + off3 + 2 * static_cast<int>(Dim_ls[2] - 1);
        Dl4_max = Dj4;
    }
    
    const int DmaxJ12_global = Dl1_max + Dl2_max;
    const int DmaxJ34_global = Dl3_max + Dl4_max;
    
    PureGeomCacheZ geom12;
    geom12.init(Dl1_max, Dl2_max, DmaxJ12_global, Dj1, Dj2);
    PureGeomCacheZ geom34;
    geom34.init(Dl3_max, Dl4_max, DmaxJ34_global, Dj3, Dj4);

    KappaSeed3jCacheZ seed_cache12;
    seed_cache12.init(Dj1, Dj2, DmaxJ12_global);
    KappaSeed3jCacheZ seed_cache34;
    seed_cache34.init(Dj3, Dj4, DmaxJ34_global);
    
    const int ws12_DJ1_min = Dj1;
    const int ws12_DJ2_min = Dj2;
    const int ws12_DJ_min  = DmaxJ12_global & 1;
    const int ws12_dim2 = (Dl2_max - ws12_DJ2_min) / 2 + 1;
    const int ws12_dim3 = (DmaxJ12_global - ws12_DJ_min) / 2 + 1;
    const int ws12_stride2 = ws12_dim3 + 2;
    const int ws12_stride1 = (ws12_dim2 + 2) * ws12_stride2;
    
    const int ws34_DJ1_min = Dj3;
    const int ws34_DJ2_min = Dj4;
    const int ws34_DJ_min  = DmaxJ34_global & 1;
    const int ws34_dim2 = (Dl4_max - ws34_DJ2_min) / 2 + 1;
    const int ws34_dim3 = (DmaxJ34_global - ws34_DJ_min) / 2 + 1;
    const int ws34_stride2 = ws34_dim3 + 2;
    const int ws34_stride1 = (ws34_dim2 + 2) * ws34_stride2;
    
    std::vector<KGroupZ_Rank4> groups;
    groups.reserve(b_vals.empty() ? 0 : (b_vals.back() / 2 + 2));
    
    auto get_group_index = [&](int k12_max) -> int {
        for (int i = 0; i < static_cast<int>(groups.size()); ++i) {
            if (groups[i].k12_max == k12_max) return i;
        }
        groups.push_back(KGroupZ_Rank4{k12_max, {}, {}});
        return static_cast<int>(groups.size()) - 1;
    };
    
    for (std::size_t b_idx = 0; b_idx < b_size; ++b_idx) {
        const int b = b_vals[b_idx];
        KGroupZ_Rank4& g = groups[get_group_index(std::min(a, b))];
        
        ABTaskZ_Rank4 ab;
        ab.b = b;
        ab.flat_out_base = Dim_ls[0] * Dim_ls[1] * Dim_ls[2] * b_idx;
        
        ab.base_pref =
        (((a - b) / 2) & 1 ? -1.0 : 1.0) *
        B4_PI * B4_PI / 32.0 *
        (a + 1.0) * (b + 1.0) *
        std::sqrt((a + 1.0) * (b + 1.0)) *
        std::sqrt((Dj1 + 1.0) * (Dj2 + 1.0) * (Dj3 + 1.0) * (Dj4 + 1.0)) *
        z_to_k_factor(Dj1, Dj2, a, Dj1, Dj2) *
        z_to_k_factor(Dj3, Dj4, a, Dj3, Dj4);
        
        if constexpr (FixedIdx == 1 || FixedIdx == 2) {
            for (int d1 = 0; d1 < static_cast<int>(Dim_ls[0]); ++d1) {
                int Dl1 = (FixedIdx == 1) ? Dj1 : Dj1 + off1 + 2 * d1;
                int Dl2 = (FixedIdx == 1) ? Dj2 + off2 + 2 * d1 : Dj2;
                if (b < iabs_int(Dl1 - Dl2) || b > Dl1 + Dl2) continue;
                if (Dj1 == Dj2 && Dl1 == Dl2 && (((b - a) / 2) & 1)) continue;
                
                HalfTaskZ t;
                t.dp_idx = ws_flat_idx_precomputed(Dl1, Dl2, b, ws12_DJ1_min, ws12_DJ2_min, ws12_DJ_min, ws12_stride1, ws12_stride2);
                t.flat_offset = d1;
                t.dsum = dsum01 + d1;
                t.fact = std::sqrt((Dl1 + 1.0) * (Dl2 + 1.0)) * z_to_k_factor(Dl1, Dl2, b, Dj1, Dj2);
                ab.t12.push_back(t);
            }
            for (int d2 = 0; d2 < static_cast<int>(Dim_ls[1]); ++d2) {
                for (int d3 = 0; d3 < static_cast<int>(Dim_ls[2]); ++d3) {
                    int Dl3 = Dj3 + off3 + 2 * d2;
                    int Dl4 = Dj4 + off4 + 2 * d3;
                    if (b < iabs_int(Dl3 - Dl4) || b > Dl3 + Dl4) continue;
                    if (Dj3 == Dj4 && Dl3 == Dl4 && (((b - a) / 2) & 1)) continue;
                    
                    HalfTaskZ t;
                    t.dp_idx = ws_flat_idx_precomputed(Dl3, Dl4, b, ws34_DJ1_min, ws34_DJ2_min, ws34_DJ_min, ws34_stride1, ws34_stride2);
                    t.flat_offset = Dim_ls[0] * d2 + Dim_ls[0] * Dim_ls[1] * d3;
                    t.dsum = dsum12 + d2 + d3;
                    t.fact = std::sqrt((Dl3 + 1.0) * (Dl4 + 1.0)) * z_to_k_factor(Dl3, Dl4, b, Dj3, Dj4);
                    ab.t34.push_back(t);
                }
            }
        } else {
            for (int d1 = 0; d1 < static_cast<int>(Dim_ls[0]); ++d1) {
                for (int d2 = 0; d2 < static_cast<int>(Dim_ls[1]); ++d2) {
                    if (do_reduce_exchange12 && d2 < d1) continue;

                    int Dl1 = Dj1 + off1 + 2 * d1;
                    int Dl2 = Dj2 + off2 + 2 * d2;
                    if (b < iabs_int(Dl1 - Dl2) || b > Dl1 + Dl2) continue;
                    if (Dj1 == Dj2 && Dl1 == Dl2 && (((b - a) / 2) & 1)) continue;
                    
                    HalfTaskZ t;
                    t.dp_idx = ws_flat_idx_precomputed(Dl1, Dl2, b, ws12_DJ1_min, ws12_DJ2_min, ws12_DJ_min, ws12_stride1, ws12_stride2);
                    t.flat_offset = d1 + Dim_ls[0] * d2;
                    t.dsum = dsum01_pair + d1 + d2;
                    t.fact = std::sqrt((Dl1 + 1.0) * (Dl2 + 1.0)) * z_to_k_factor(Dl1, Dl2, b, Dj1, Dj2);
                    ab.t12.push_back(t);
                }
            }
            for (int d3 = 0; d3 < static_cast<int>(Dim_ls[2]); ++d3) {
                int Dl3 = (FixedIdx == 3) ? Dj3 : Dj3 + off3 + 2 * d3;
                int Dl4 = (FixedIdx == 3) ? Dj4 + off4 + 2 * d3 : Dj4;
                if (b < iabs_int(Dl3 - Dl4) || b > Dl3 + Dl4) continue;
                if (Dj3 == Dj4 && Dl3 == Dl4 && (((b - a) / 2) & 1)) continue;
                
                HalfTaskZ t;
                t.dp_idx = ws_flat_idx_precomputed(Dl3, Dl4, b, ws34_DJ1_min, ws34_DJ2_min, ws34_DJ_min, ws34_stride1, ws34_stride2);
                t.flat_offset =  Dim_ls[0] *  Dim_ls[1] * d3;
                t.dsum = dsum2 + d3;
                t.fact = std::sqrt((Dl3 + 1.0) * (Dl4 + 1.0)) * z_to_k_factor(Dl3, Dl4, b, Dj3, Dj4);
                ab.t34.push_back(t);
            }
        }
        
        if (!ab.t12.empty() && !ab.t34.empty()) {
            g.ab_tasks.push_back(std::move(ab));
        }
    }
    
    const bool symmetric_pair = (Dj1 == Dj2 || Dj3 == Dj4);
    int max_Dk12 = 0;
    
    for (KGroupZ_Rank4& g : groups) {
        max_Dk12 = std::max(max_Dk12, g.k12_max);
        const int start_Dk = g.k12_max & 1;
        const int nslots = (g.k12_max - start_Dk) / 2 + 1;
        g.ab_indices_for_dk_slot.assign(nslots, {});
        for (int i = 0; i < static_cast<int>(g.ab_tasks.size()); ++i) {
            for (int slot = 0; slot < nslots; ++slot) {
                const int Dk = start_Dk + 2 * slot;
                if (symmetric_pair && (((a + Dk) / 2) & 1)) continue;
                g.ab_indices_for_dk_slot[slot].push_back(i);
            }
        }
    }

    std::vector<bool> active_Dk(max_Dk12 + 1, false);
    for (const KGroupZ_Rank4& g : groups) {
        const int start_Dk = g.k12_max & 1;
        const int nslots = (g.k12_max - start_Dk) / 2 + 1;
        for (int slot = 0; slot < nslots; ++slot) {
            if (!g.ab_indices_for_dk_slot[slot].empty()) {
                active_Dk[start_Dk + 2 * slot] = true;
            }
        }
    }
    
    std::vector<KappaTapeZ> tapes12(max_Dk12 + 1);
    std::vector<KappaTapeZ> tapes34(max_Dk12 + 1);

    std::vector<JobHybridZ> jobs;
    const auto& rule = get_split_gl64_rule();
    const double Jtot = 0.5 * (Dj1 + Dj2 + Dj3 + Dj4);
    
    std::size_t total_work = 0;
    for (int gidx = 0; gidx < static_cast<int>(groups.size()); ++gidx) {
        KGroupZ_Rank4& g = groups[gidx];
        const double W = std::sqrt((gamma > 1.0) ? (Jtot + 4.0) * gamma : (Jtot + 4.0));
        for (int slot = 0; slot < static_cast<int>(g.ab_indices_for_dk_slot.size()); ++slot) {
            if (g.ab_indices_for_dk_slot[slot].empty()) continue;
            JobHybridZ job{};
            job.group_idx = gidx;
            job.dk_slot = slot;
            job.Dk12 = (g.k12_max & 1) + 2 * slot;
            const double k = 0.5 * job.Dk12;
            const double k_symmetry_factor = (job.Dk12 == 0) ? 0.5 : 1.0;
            for (int pt = 0; pt < V_LEN; ++pt) {
                const double xi = rule.x[pt];
                const double inv_comp = 1.0 / (1.0 - xi * xi);
                const double inv_sqrt = std::sqrt(inv_comp);
                job.rhos[pt] = simplicity_constraint(
                    gamma, g.k12_max, prescription) + W * xi * inv_sqrt;
                const double measure = job.rhos[pt] * job.rhos[pt] + k * k;
                job.quad_weights[pt] = rule.w[pt] * W * inv_comp * inv_sqrt * measure * k_symmetry_factor;
            }
            jobs.push_back(job);
            for (int ab_idx : g.ab_indices_for_dk_slot[slot]) {
                total_work += g.ab_tasks[ab_idx].t12.size() * g.ab_tasks[ab_idx].t34.size();
            }
        }
    }
    
    if (jobs.empty()) return;
    
    // Explicitly sort jobs to guarantee perfect static distribution.
    std::vector<int> job_order(jobs.size());
    for(int i = 0; i < static_cast<int>(jobs.size()); ++i) job_order[i] = i;
    std::sort(job_order.begin(), job_order.end(), [&](int lhs, int rhs) {
        int work_lhs = groups[jobs[lhs].group_idx].ab_indices_for_dk_slot[jobs[lhs].dk_slot].size();
        int work_rhs = groups[jobs[rhs].group_idx].ab_indices_for_dk_slot[jobs[rhs].dk_slot].size();
        return work_lhs > work_rhs;
    });
    
    std::vector<GroupHarvestIndexZ_Rank4> group_indices(groups.size());
    for (std::size_t gidx = 0; gidx < groups.size(); ++gidx) {
        group_indices[gidx].min12_idx = ws_flat_idx_precomputed(Dj1, Dj2, a, ws12_DJ1_min, ws12_DJ2_min, ws12_DJ_min, ws12_stride1, ws12_stride2);
        group_indices[gidx].min34_idx = ws_flat_idx_precomputed(Dj3, Dj4, a, ws34_DJ1_min, ws34_DJ2_min, ws34_DJ_min, ws34_stride1, ws34_stride2);
    }
    
    const int nthreads = b4_choose_num_threads_for_group_work(static_cast<int>(jobs.size()), total_work);
    
    #if B4_ENABLE_TIMING
    std::vector<ThreadTimingZ> timings(nthreads);
    #endif

    omp_set_dynamic(0);
    
    const std::size_t max_t12 =
    (FixedIdx == 1 || FixedIdx == 2)
    ? Dim_ls[0]
    : Dim_ls[0] * Dim_ls[1];
    
    // Compile both tape families and execute their jobs within one parallel
    // region. Static schedules avoid lock-based work queues in both phases.
    #pragma omp parallel num_threads(nthreads)
    {
        #if B4_ENABLE_TIMING
        const int tid = omp_get_thread_num();
        #endif

        std::vector<RhoComplexTileZ<RhoTile>> pre_t12_buffer(max_t12);
        
        ThreadWorkspaceTileZ<RhoTile> ws12;
        ws12.init_dims(ws12_DJ1_min, ws12_DJ2_min, ws12_DJ_min);
        ws12.resize((Dl1_max - ws12_DJ1_min) / 2 + 1, (Dl2_max - ws12_DJ2_min) / 2 + 1, (DmaxJ12_global - ws12_DJ_min) / 2 + 1);
        
        ThreadWorkspaceTileZ<RhoTile> ws34;
        ws34.init_dims(ws34_DJ1_min, ws34_DJ2_min, ws34_DJ_min);
        ws34.resize((Dl3_max - ws34_DJ1_min) / 2 + 1, (Dl4_max - ws34_DJ2_min) / 2 + 1, (DmaxJ34_global - ws34_DJ_min) / 2 + 1);
        
        std::vector<double> thread_local_dst(out_size, 0.0);

        // Compile tapes with a static, lock-free assignment.
        #pragma omp for schedule(static, 1) nowait
        for (int Dk = 0; Dk <= max_Dk12; ++Dk) {
            if (active_Dk[Dk]) {
                tapes12[Dk].compile(rho1, Dj1, Dl1_max, rho2, Dj2, Dl2_max, Dk, DmaxJ12_global, geom12);
                tapes34[Dk].compile(rho3, Dj3, Dl3_max, rho4, Dj4, Dl4_max, Dk, DmaxJ34_global, geom34);
            }
        }

        // Wait until every tape is ready.
        #pragma omp barrier

        // Execute jobs with a static, lock-free assignment.
        #pragma omp for schedule(static, 1)
        for (int ord = 0; ord < static_cast<int>(job_order.size()); ++ord) {
            const int j_idx = job_order[ord];
            const JobHybridZ& job = jobs[j_idx];
            const KGroupZ_Rank4& g = groups[job.group_idx];
            const GroupHarvestIndexZ_Rank4& gi = group_indices[job.group_idx];
            
            const KappaTapeZ& tape12 = tapes12[job.Dk12];
            const KappaTapeZ& tape34 = tapes34[job.Dk12];
            
            for (int pt0 = 0; pt0 < V_LEN; pt0 += RhoTile) {
                constexpr int npt = RhoTile;
                const double* __restrict rho_tile = job.rhos + pt0;
                
                alignas(64) double W_ARR[RhoTile];
                for (int lane = 0; lane < npt; ++lane) W_ARR[lane] = job.quad_weights[pt0 + lane];
                
                #if B4_ENABLE_TIMING
                const double t0 = omp_get_wtime();
                #endif

                KappaFastZ_ExecuteTapeTile(rho1, Dj1, Dl1_max, rho2, Dj2, Dl2_max, rho_tile, npt, job.Dk12, DmaxJ12_global, LF, ws12, tape12, &seed_cache12);

                #if B4_ENABLE_TIMING
                const double t1 = omp_get_wtime();
                #endif

                KappaFastZ_ExecuteTapeTile(rho3, Dj3, Dl3_max, rho4, Dj4, Dl4_max, rho_tile, npt, job.Dk12, DmaxJ34_global, LF, ws34, tape34, &seed_cache34);
                
                #if B4_ENABLE_TIMING
                const double t2 = omp_get_wtime();
                #endif
                
                alignas(64) double mR[RhoTile];
                alignas(64) double mI[RhoTile];
                
                const double* __restrict z12r = ws12.cell_R(gi.min12_idx);
                const double* __restrict z12i = ws12.cell_I(gi.min12_idx);
                const double* __restrict z34r = ws34.cell_R(gi.min34_idx);
                const double* __restrict z34i = ws34.cell_I(gi.min34_idx);

                #pragma omp simd
                for (int lane = 0; lane < npt; ++lane) {
                    mR[lane] = z12r[lane] * z34r[lane] - z12i[lane] * z34i[lane];
                    mI[lane] = -(z12r[lane] * z34i[lane] + z12i[lane] * z34r[lane]);
                }
                
                for (int ab_idx : g.ab_indices_for_dk_slot[job.dk_slot]) {
                    const ABTaskZ_Rank4& ab = g.ab_tasks[ab_idx];
                    
                    alignas(64) double w_mR[RhoTile];
                    alignas(64) double w_mI[RhoTile];
                    
                    #pragma omp simd
                    for (int lane = 0; lane < RhoTile; ++lane) {
                        double w_base = W_ARR[lane] * ab.base_pref;
                        w_mR[lane] = w_base * mR[lane];
                        w_mI[lane] = w_base * mI[lane];
                    }
                    
                    for (size_t i = 0; i < ab.t12.size(); ++i) {
                        const HalfTaskZ& t12 = ab.t12[i];
                        const double* r12 = ws12.cell_R(t12.dp_idx);
                        const double* i12 = ws12.cell_I(t12.dp_idx);
                        const double f12 = t12.fact;
                        
                        switch (t12.dsum & 3) {
                            case 0:
                                #pragma omp simd
                                for (int lane = 0; lane < RhoTile; ++lane) { pre_t12_buffer[i].R[lane] =  r12[lane] * f12; pre_t12_buffer[i].I[lane] = -i12[lane] * f12; }
                                break;
                            case 1:
                                #pragma omp simd
                                for (int lane = 0; lane < RhoTile; ++lane) { pre_t12_buffer[i].R[lane] =  i12[lane] * f12; pre_t12_buffer[i].I[lane] =  r12[lane] * f12; }
                                break;
                            case 2:
                                #pragma omp simd
                                for (int lane = 0; lane < RhoTile; ++lane) { pre_t12_buffer[i].R[lane] = -r12[lane] * f12; pre_t12_buffer[i].I[lane] =  i12[lane] * f12; }
                                break;
                            default:
                                #pragma omp simd
                                for (int lane = 0; lane < RhoTile; ++lane) { pre_t12_buffer[i].R[lane] = -i12[lane] * f12; pre_t12_buffer[i].I[lane] = -r12[lane] * f12; }
                                break;
                        }
                    }
                    
                    alignas(64) double P_R[RhoTile];
                    alignas(64) double P_I[RhoTile];
                    
                    for (const HalfTaskZ& t34 : ab.t34) {
                        const double* __restrict r34 = ws34.cell_R(t34.dp_idx);
                        const double* __restrict i34 = ws34.cell_I(t34.dp_idx);
                        const double f34 = t34.fact;
                        
                        switch (t34.dsum & 3) {
                            case 0:
                                #pragma omp simd
                                for (int lane = 0; lane < RhoTile; ++lane) {
                                    double wfR = w_mR[lane] * f34, wfI = w_mI[lane] * f34;
                                    P_R[lane] = r34[lane] * wfR - i34[lane] * wfI;
                                    P_I[lane] = r34[lane] * wfI + i34[lane] * wfR;
                                } break;
                            case 1:
                                #pragma omp simd
                                for (int lane = 0; lane < RhoTile; ++lane) {
                                    double wfR = w_mR[lane] * f34, wfI = w_mI[lane] * f34;
                                    P_R[lane] = i34[lane] * wfR + r34[lane] * wfI;
                                    P_I[lane] = i34[lane] * wfI - r34[lane] * wfR;
                                } break;
                            case 2:
                                #pragma omp simd
                                for (int lane = 0; lane < RhoTile; ++lane) {
                                    double n_wfR = -w_mR[lane] * f34, n_wfI = -w_mI[lane] * f34;
                                    P_R[lane] = r34[lane] * n_wfR - i34[lane] * n_wfI;
                                    P_I[lane] = r34[lane] * n_wfI + i34[lane] * n_wfR;
                                } break;
                            default:
                                #pragma omp simd
                                for (int lane = 0; lane < RhoTile; ++lane) {
                                    double wfR = w_mR[lane] * f34, wfI = w_mI[lane] * f34;
                                    P_R[lane] = -i34[lane] * wfR - r34[lane] * wfI;
                                    P_I[lane] = -i34[lane] * wfI + r34[lane] * wfR;
                                } break;
                        }
                        
                        const std::size_t out_base34 = ab.flat_out_base + t34.flat_offset;
                        
                        for (size_t i = 0; i < ab.t12.size(); ++i) {
                            double sum = 0.0;
                            #pragma omp simd reduction(+:sum)
                            for (int lane = 0; lane < RhoTile; ++lane) {
                                sum += P_R[lane] * pre_t12_buffer[i].R[lane] + P_I[lane] * pre_t12_buffer[i].I[lane];
                            }
                            
                            if (sum != 0.0) {
                                thread_local_dst[out_base34 + ab.t12[i].flat_offset] += sum;
                            }
                        }
                    }
                }

                #if B4_ENABLE_TIMING
                const double t3 = omp_get_wtime();
                ThreadTimingZ& tm = timings[tid];
                tm.kappa12_seconds += t1 - t0;
                tm.kappa34_seconds += t2 - t1;
                tm.contract_seconds += t3 - t2;
                tm.tiles += 1;
                #endif
            } 
        }

        // Phase 3: Final Merge
        #pragma omp critical
        {
            const double* __restrict local_ptr = thread_local_dst.data();
            #pragma omp simd
            for (std::size_t i = 0; i < out_size; ++i) {
                dst[i] += local_ptr[i];
            }
        }
    }
    
    #if B4_ENABLE_TIMING
    double sum_kappa12 = 0.0, sum_kappa34 = 0.0, sum_contract = 0.0;
    double max_kappa12 = 0.0, max_kappa34 = 0.0, max_contract = 0.0;
    std::uint64_t total_tiles = 0;

    for (const ThreadTimingZ& tm : timings) {
        sum_kappa12 += tm.kappa12_seconds; sum_kappa34 += tm.kappa34_seconds; sum_contract += tm.contract_seconds;
        max_kappa12 = std::max(max_kappa12, tm.kappa12_seconds); max_kappa34 = std::max(max_kappa34, tm.kappa34_seconds); max_contract = std::max(max_contract, tm.contract_seconds);
        total_tiles += tm.tiles;
    }

    std::cerr
        << "---[Rank4 detailed profile - FixedIdx " << FixedIdx << "] ---\n"
        << "  tiles:              " << total_tiles << "\n"
        << "  kappa12 CPU-sec:    " << sum_kappa12 << " | max-thread-sec: " << max_kappa12 << "\n"
        << "  kappa34 CPU-sec:    " << sum_kappa34 << " | max-thread-sec: " << max_kappa34 << "\n"
        << "  contraction CPU-sec:" << sum_contract << " | max-thread-sec: " << max_contract << "\n\n";
    #endif
}

template <int FixedIdx>
void harvest_b4_rank4(
    int Dj1, int Dj2, int Dj3, int Dj4,
    const std::array<std::size_t,3> Dim_ls,
    int a,
    std::span<const int> b_vals,
    double gamma,
    const std::vector<double>& LF,
    Tensor<double, 4>& out_tensor,
    const std::array<std::size_t,3> ell0 = {0,0,0},
    bool reduce_exchange12_harvest = false,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0)
{
    const int resolved_rho_tile = b4_resolve_rho_tile(rho_tile);
    if (resolved_rho_tile == 4) {
        return harvest_b4_rank4_impl<FixedIdx, 4>(
            Dj1, Dj2, Dj3, Dj4, Dim_ls, a, b_vals, gamma, LF,
            out_tensor, ell0, reduce_exchange12_harvest, prescription);
    }
    if (resolved_rho_tile == 8) {
        return harvest_b4_rank4_impl<FixedIdx, 8>(
            Dj1, Dj2, Dj3, Dj4, Dim_ls, a, b_vals, gamma, LF,
            out_tensor, ell0, reduce_exchange12_harvest, prescription);
    }
    return harvest_b4_rank4_impl<FixedIdx, 16>(
        Dj1, Dj2, Dj3, Dj4, Dim_ls, a, b_vals, gamma, LF,
        out_tensor, ell0, reduce_exchange12_harvest, prescription);
}

// Symmetry-aware replacement snippets for B4TensorFilling.hpp
// Insert the helper functions above fill_rank4_tensors_offset / fill_rank5_tensors_offset,
// then replace those two fill_*_offset functions and their wrappers with the versions below.

// -----------------------------------------------------------------------------
// Booster symmetry copy helpers
// -----------------------------------------------------------------------------

inline int b4_phase_from_even_Dsum_local(int Dsum) noexcept {
#ifndef NDEBUG
    if (Dsum & 1) {
        std::cerr << "Odd doubled-spin phase sum in booster symmetry: "
                  << Dsum << "\n";
        std::abort();
    }
#endif
    const int n = Dsum / 2;
    return (std::abs(n) & 1) ? -1 : 1;
}

// For B4 pair exchange (12)<->(34): (-1)^(2 i + 2 k).
// With doubled labels Da=2 i and Db=2 k, this is (-1)^(Da+Db).
inline int b4_phase_pair_exchange_local(int Da, int Db) noexcept {
    return ((Da + Db) & 1) ? -1 : 1;
}

inline bool b4_equal_int_vectors(const std::vector<int>& a,
                                 const std::vector<int>& b) {
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
}

// dst(i0,i1,i2,b) = phase_34(i2,b) * src(i0,i1,i2,b)
// This implements the 3<->4 exchange used to derive B3 from B2.
inline void copy_rank4_exchange34_same_axes_to_dst(
    const Tensor<double, 4>& src,
    Tensor<double, 4>& dst,
    int Dj3,
    int Dj4,
    int Dl3_base,
    int Dl4_fixed,
    int Da,
    std::span<const int> b_vals)
{
    const std::size_t D0 = dst.dims()[0];
    const std::size_t D1 = dst.dims()[1];
    const std::size_t D2 = dst.dims()[2];
    const std::size_t DB = dst.dims()[3];

#ifndef NDEBUG
    assert(src.dims()[0] == D0);
    assert(src.dims()[1] == D1);
    assert(src.dims()[2] == D2);
    assert(src.dims()[3] == DB);
    assert(b_vals.size() == DB);
#endif

    const double* __restrict s = src.data();
    double* __restrict d = dst.data();

    const std::size_t s0 = src.strides()[0];
    const std::size_t s1 = src.strides()[1];
    const std::size_t s2 = src.strides()[2];
    const std::size_t s3 = src.strides()[3];

    const std::size_t d0 = dst.strides()[0];
    const std::size_t d1 = dst.strides()[1];
    const std::size_t d2 = dst.strides()[2];
    const std::size_t d3 = dst.strides()[3];

    const std::size_t total = D0 * D1 * D2 * DB;

    #pragma omp parallel for collapse(2) schedule(static) if(total > (1ull << 20))
    for (std::ptrdiff_t b_s = 0; b_s < static_cast<std::ptrdiff_t>(DB); ++b_s) {
        for (std::ptrdiff_t i2_s = 0; i2_s < static_cast<std::ptrdiff_t>(D2); ++i2_s) {
            const std::size_t ib = static_cast<std::size_t>(b_s);
            const std::size_t i2 = static_cast<std::size_t>(i2_s);

            const int Db  = b_vals[ib];
            const int Dl3 = Dl3_base + 2 * static_cast<int>(i2);
            const double phase = static_cast<double>(
                b4_phase_from_even_Dsum_local(Dj3 + Dj4 + Da + Dl3 + Dl4_fixed + Db)
            );

            for (std::size_t i1 = 0; i1 < D1; ++i1) {
                for (std::size_t i0 = 0; i0 < D0; ++i0) {
                    const std::size_t src_idx =
                        i0 * s0 + i1 * s1 + i2 * s2 + ib * s3;
                    const std::size_t dst_idx =
                        i0 * d0 + i1 * d1 + i2 * d2 + ib * d3;
                    d[dst_idx] = phase * s[src_idx];
                }
            }
        }
    }
}

// dst(i0,i1,i2,b) = phase_pair(b) * src(i1,i2,i0,b)
// This implements the (12)<->(34) pair exchange, used for B4 from B2 and B5 from B3.
inline void copy_rank4_pair_exchange_120_to_dst(
    const Tensor<double, 4>& src,
    Tensor<double, 4>& dst,
    int Da,
    std::span<const int> b_vals)
{
    const std::size_t D0 = dst.dims()[0];
    const std::size_t D1 = dst.dims()[1];
    const std::size_t D2 = dst.dims()[2];
    const std::size_t DB = dst.dims()[3];

#ifndef NDEBUG
    // dst(i0,i1,i2,b) = src(i1,i2,i0,b)
    assert(src.dims()[0] == D1);
    assert(src.dims()[1] == D2);
    assert(src.dims()[2] == D0);
    assert(src.dims()[3] == DB);
    assert(b_vals.size() == DB);
#endif

    const double* __restrict s = src.data();
    double* __restrict d = dst.data();

    const std::size_t s0 = src.strides()[0];
    const std::size_t s1 = src.strides()[1];
    const std::size_t s2 = src.strides()[2];
    const std::size_t s3 = src.strides()[3];

    const std::size_t d0 = dst.strides()[0];
    const std::size_t d1 = dst.strides()[1];
    const std::size_t d2 = dst.strides()[2];
    const std::size_t d3 = dst.strides()[3];

    const std::size_t total = D0 * D1 * D2 * DB;

    #pragma omp parallel for collapse(2) schedule(static) if(total > (1ull << 20))
    for (std::ptrdiff_t b_s = 0; b_s < static_cast<std::ptrdiff_t>(DB); ++b_s) {
        for (std::ptrdiff_t i2_s = 0; i2_s < static_cast<std::ptrdiff_t>(D2); ++i2_s) {
            const std::size_t ib = static_cast<std::size_t>(b_s);
            const std::size_t i2 = static_cast<std::size_t>(i2_s);
            const double phase = static_cast<double>(
                b4_phase_pair_exchange_local(Da, b_vals[ib])
            );

            for (std::size_t i1 = 0; i1 < D1; ++i1) {
                for (std::size_t i0 = 0; i0 < D0; ++i0) {
                    const std::size_t dst_idx =
                        i0 * d0 + i1 * d1 + i2 * d2 + ib * d3;
                    const std::size_t src_idx =
                        i1 * s0 + i2 * s1 + i0 * s2 + ib * s3;
                    d[dst_idx] = phase * s[src_idx];
                }
            }
        }
    }
}

// Rank-5 version: dst(i0,i1,i2,a,b) = phase_34(i2,a,b) * src(i0,i1,i2,a,b)
inline void copy_rank5_exchange34_same_axes_to_dst(
    const Tensor<double, 5>& src,
    Tensor<double, 5>& dst,
    int Dj3,
    int Dj4,
    int Dl3_base,
    int Dl4_fixed,
    std::span<const int> a_vals,
    std::span<const int> b_vals)
{
    const std::size_t D0 = dst.dims()[0];
    const std::size_t D1 = dst.dims()[1];
    const std::size_t D2 = dst.dims()[2];
    const std::size_t DA = dst.dims()[3];
    const std::size_t DB = dst.dims()[4];

#ifndef NDEBUG
    assert(src.dims()[0] == D0);
    assert(src.dims()[1] == D1);
    assert(src.dims()[2] == D2);
    assert(src.dims()[3] == DA);
    assert(src.dims()[4] == DB);
    assert(a_vals.size() == DA);
    assert(b_vals.size() == DB);
#endif

    const double* __restrict s = src.data();
    double* __restrict d = dst.data();

    const std::size_t s0 = src.strides()[0];
    const std::size_t s1 = src.strides()[1];
    const std::size_t s2 = src.strides()[2];
    const std::size_t s3 = src.strides()[3];
    const std::size_t s4 = src.strides()[4];

    const std::size_t d0 = dst.strides()[0];
    const std::size_t d1 = dst.strides()[1];
    const std::size_t d2 = dst.strides()[2];
    const std::size_t d3 = dst.strides()[3];
    const std::size_t d4 = dst.strides()[4];

    const std::size_t total = D0 * D1 * D2 * DA * DB;

    #pragma omp parallel for collapse(3) schedule(static) if(total > (1ull << 20))
    for (std::ptrdiff_t ib_s = 0; ib_s < static_cast<std::ptrdiff_t>(DB); ++ib_s) {
        for (std::ptrdiff_t ia_s = 0; ia_s < static_cast<std::ptrdiff_t>(DA); ++ia_s) {
            for (std::ptrdiff_t i2_s = 0; i2_s < static_cast<std::ptrdiff_t>(D2); ++i2_s) {
                const std::size_t ib = static_cast<std::size_t>(ib_s);
                const std::size_t ia = static_cast<std::size_t>(ia_s);
                const std::size_t i2 = static_cast<std::size_t>(i2_s);

                const int Da  = a_vals[ia];
                const int Db  = b_vals[ib];
                const int Dl3 = Dl3_base + 2 * static_cast<int>(i2);
                const double phase = static_cast<double>(
                    b4_phase_from_even_Dsum_local(Dj3 + Dj4 + Da + Dl3 + Dl4_fixed + Db)
                );

                for (std::size_t i1 = 0; i1 < D1; ++i1) {
                    for (std::size_t i0 = 0; i0 < D0; ++i0) {
                        const std::size_t src_idx =
                            i0 * s0 + i1 * s1 + i2 * s2 + ia * s3 + ib * s4;
                        const std::size_t dst_idx =
                            i0 * d0 + i1 * d1 + i2 * d2 + ia * d3 + ib * d4;
                        d[dst_idx] = phase * s[src_idx];
                    }
                }
            }
        }
    }
}

// Rank-5 version: dst(i0,i1,i2,a,b) = phase_pair(a,b) * src(i1,i2,i0,a,b)
inline void copy_rank5_pair_exchange_120_to_dst(
    const Tensor<double, 5>& src,
    Tensor<double, 5>& dst,
    std::span<const int> a_vals,
    std::span<const int> b_vals)
{
    const std::size_t D0 = dst.dims()[0];
    const std::size_t D1 = dst.dims()[1];
    const std::size_t D2 = dst.dims()[2];
    const std::size_t DA = dst.dims()[3];
    const std::size_t DB = dst.dims()[4];

#ifndef NDEBUG
    assert(src.dims()[0] == D1);
    assert(src.dims()[1] == D2);
    assert(src.dims()[2] == D0);
    assert(src.dims()[3] == DA);
    assert(src.dims()[4] == DB);
    assert(a_vals.size() == DA);
    assert(b_vals.size() == DB);
#endif

    const double* __restrict s = src.data();
    double* __restrict d = dst.data();

    const std::size_t s0 = src.strides()[0];
    const std::size_t s1 = src.strides()[1];
    const std::size_t s2 = src.strides()[2];
    const std::size_t s3 = src.strides()[3];
    const std::size_t s4 = src.strides()[4];

    const std::size_t d0 = dst.strides()[0];
    const std::size_t d1 = dst.strides()[1];
    const std::size_t d2 = dst.strides()[2];
    const std::size_t d3 = dst.strides()[3];
    const std::size_t d4 = dst.strides()[4];

    const std::size_t total = D0 * D1 * D2 * DA * DB;

    #pragma omp parallel for collapse(3) schedule(static) if(total > (1ull << 20))
    for (std::ptrdiff_t ib_s = 0; ib_s < static_cast<std::ptrdiff_t>(DB); ++ib_s) {
        for (std::ptrdiff_t ia_s = 0; ia_s < static_cast<std::ptrdiff_t>(DA); ++ia_s) {
            for (std::ptrdiff_t i2_s = 0; i2_s < static_cast<std::ptrdiff_t>(D2); ++i2_s) {
                const std::size_t ib = static_cast<std::size_t>(ib_s);
                const std::size_t ia = static_cast<std::size_t>(ia_s);
                const std::size_t i2 = static_cast<std::size_t>(i2_s);
                const double phase = static_cast<double>(
                    b4_phase_pair_exchange_local(a_vals[ia], b_vals[ib])
                );

                for (std::size_t i1 = 0; i1 < D1; ++i1) {
                    for (std::size_t i0 = 0; i0 < D0; ++i0) {
                        const std::size_t dst_idx =
                            i0 * d0 + i1 * d1 + i2 * d2 + ia * d3 + ib * d4;
                        const std::size_t src_idx =
                            i1 * s0 + i2 * s1 + i0 * s2 + ia * s3 + ib * s4;
                        d[dst_idx] = phase * s[src_idx];
                    }
                }
            }
        }
    }
}


// Complete a triangular harvest using the 1<->2 exchange symmetry.
// The harvester computed only entries with i0 <= i1 for the first two
// variable legs. This fills the missing lower triangle i1 < i0 by
//
//   B(l1,l2,l3,l4; a,b) = phase12(l1,l2,a,b) * B(l2,l1,l3,l4; a,b)
//
// where phase12 = (-1)^(j1+j2+a+l1+l2+b).
inline void complete_rank4_exchange12_lower_from_upper(
    Tensor<double, 4>& tensor,
    int Dj1,
    int Dj2,
    int Dl1_base,
    int Dl2_base,
    int Da,
    std::span<const int> b_vals)
{
    const std::size_t D0 = tensor.dims()[0];
    const std::size_t D1 = tensor.dims()[1];
    const std::size_t D2 = tensor.dims()[2];
    const std::size_t DB = tensor.dims()[3];

#ifndef NDEBUG
    assert(D0 == D1);
    assert(b_vals.size() == DB);
#endif

    double* __restrict d = tensor.data();

    const std::size_t s0 = tensor.strides()[0];
    const std::size_t s1 = tensor.strides()[1];
    const std::size_t s2 = tensor.strides()[2];
    const std::size_t s3 = tensor.strides()[3];

    const std::size_t total = D0 * D1 * D2 * DB;

    #pragma omp parallel for collapse(2) schedule(static) if(total > (1ull << 20))
    for (std::ptrdiff_t ib_s = 0; ib_s < static_cast<std::ptrdiff_t>(DB); ++ib_s) {
        for (std::ptrdiff_t i2_s = 0; i2_s < static_cast<std::ptrdiff_t>(D2); ++i2_s) {
            const std::size_t ib = static_cast<std::size_t>(ib_s);
            const std::size_t i2 = static_cast<std::size_t>(i2_s);
            const int Db = b_vals[ib];

            for (std::size_t i0 = 1; i0 < D0; ++i0) {
                const int Dl1 = Dl1_base + 2 * static_cast<int>(i0);
                for (std::size_t i1 = 0; i1 < i0; ++i1) {
                    const int Dl2 = Dl2_base + 2 * static_cast<int>(i1);
                    const double phase = static_cast<double>(
                        b4_phase_from_even_Dsum_local(Dj1 + Dj2 + Da + Dl1 + Dl2 + Db)
                    );

                    const std::size_t dst_idx = i0 * s0 + i1 * s1 + i2 * s2 + ib * s3;
                    const std::size_t src_idx = i1 * s0 + i0 * s1 + i2 * s2 + ib * s3;
                    d[dst_idx] = phase * d[src_idx];
                }
            }
        }
    }
}

inline void complete_rank5_exchange12_lower_from_upper(
    Tensor<double, 5>& tensor,
    int Dj1,
    int Dj2,
    int Dl1_base,
    int Dl2_base,
    std::span<const int> a_vals,
    std::span<const int> b_vals)
{
    const std::size_t D0 = tensor.dims()[0];
    const std::size_t D1 = tensor.dims()[1];
    const std::size_t D2 = tensor.dims()[2];
    const std::size_t DA = tensor.dims()[3];
    const std::size_t DB = tensor.dims()[4];

#ifndef NDEBUG
    assert(D0 == D1);
    assert(a_vals.size() == DA);
    assert(b_vals.size() == DB);
#endif

    double* __restrict d = tensor.data();

    const std::size_t s0 = tensor.strides()[0];
    const std::size_t s1 = tensor.strides()[1];
    const std::size_t s2 = tensor.strides()[2];
    const std::size_t s3 = tensor.strides()[3];
    const std::size_t s4 = tensor.strides()[4];

    const std::size_t total = D0 * D1 * D2 * DA * DB;

    #pragma omp parallel for collapse(3) schedule(static) if(total > (1ull << 20))
    for (std::ptrdiff_t ib_s = 0; ib_s < static_cast<std::ptrdiff_t>(DB); ++ib_s) {
        for (std::ptrdiff_t ia_s = 0; ia_s < static_cast<std::ptrdiff_t>(DA); ++ia_s) {
            for (std::ptrdiff_t i2_s = 0; i2_s < static_cast<std::ptrdiff_t>(D2); ++i2_s) {
                const std::size_t ib = static_cast<std::size_t>(ib_s);
                const std::size_t ia = static_cast<std::size_t>(ia_s);
                const std::size_t i2 = static_cast<std::size_t>(i2_s);
                const int Da = a_vals[ia];
                const int Db = b_vals[ib];

                for (std::size_t i0 = 1; i0 < D0; ++i0) {
                    const int Dl1 = Dl1_base + 2 * static_cast<int>(i0);
                    for (std::size_t i1 = 0; i1 < i0; ++i1) {
                        const int Dl2 = Dl2_base + 2 * static_cast<int>(i1);
                        const double phase = static_cast<double>(
                            b4_phase_from_even_Dsum_local(Dj1 + Dj2 + Da + Dl1 + Dl2 + Db)
                        );

                        const std::size_t dst_idx =
                            i0 * s0 + i1 * s1 + i2 * s2 + ia * s3 + ib * s4;
                        const std::size_t src_idx =
                            i1 * s0 + i0 * s1 + i2 * s2 + ia * s3 + ib * s4;
                        d[dst_idx] = phase * d[src_idx];
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Symmetry guards: conservative exact tensor-layout checks.
// -----------------------------------------------------------------------------


inline bool can_reduce_B2_exchange12_harvest_amp(
    const B4Spins& s,
    const std::array<std::size_t,6>& Dim_ls,
    const std::array<std::size_t,6>& ell0)
{
    return
        s.Dj23 == s.Dj24 &&
        Dim_ls[0] == Dim_ls[1] &&
        ell0[0] == ell0[1];
}

inline bool can_reduce_B2_exchange12_harvest_full(
    const B4Spins& s,
    const std::array<std::size_t,6>& Dim_ls,
    const std::array<std::size_t,6>& ell0)
{
    return
        s.Dj23 == s.Dj24 &&
        Dim_ls[0] == Dim_ls[1] &&
        ell0[0] == ell0[1];
}

inline bool can_derive_B3_from_B2_exchange34_amp(
    const B4Spins& s,
    const std::array<std::size_t,6>& Dim_ls,
    const std::array<std::size_t,6>& ell0,
    const AmplitudeChannels& ch)
{
    return
        s.Dj34 == s.Dj23 &&
        s.Dj35 == s.Dj24 &&
        s.Dj13 == s.Dj12 &&
        s.Dj23 == s.Dj25 &&
        ch.i[2] == ch.i[1] &&
        ch.k3.size() == ch.k2.size() &&
        std::equal(ch.k3.begin(), ch.k3.end(), ch.k2.begin()) &&
        Dim_ls[3] == Dim_ls[0] && ell0[3] == ell0[0] &&
        Dim_ls[4] == Dim_ls[1] && ell0[4] == ell0[1] &&
        Dim_ls[0] == Dim_ls[2] && ell0[0] == ell0[2];
}

inline bool can_derive_B4_from_B2_pair_exchange_amp(
    const B4Spins& s,
    const std::array<std::size_t,6>& Dim_ls,
    const std::array<std::size_t,6>& ell0,
    const AmplitudeChannels& ch)
{
    return
        s.Dj45 == s.Dj25 &&
        s.Dj14 == s.Dj12 &&
        s.Dj24 == s.Dj23 &&
        s.Dj34 == s.Dj24 &&
        ch.i[3] == ch.i[1] &&
        ch.k4.size() == ch.k2.size() &&
        std::equal(ch.k4.begin(), ch.k4.end(), ch.k2.begin()) &&
        Dim_ls[5] == Dim_ls[2] && ell0[5] == ell0[2] &&
        Dim_ls[1] == Dim_ls[0] && ell0[1] == ell0[0] &&
        Dim_ls[3] == Dim_ls[1] && ell0[3] == ell0[1];
}

inline bool can_derive_B5_from_B3_pair_exchange_amp(
    const B4Spins& s,
    const std::array<std::size_t,6>& Dim_ls,
    const std::array<std::size_t,6>& ell0,
    const AmplitudeChannels& ch)
{
    return
        s.Dj15 == s.Dj13 &&
        s.Dj25 == s.Dj23 &&
        s.Dj35 == s.Dj34 &&
        s.Dj45 == s.Dj35 &&
        ch.i[4] == ch.i[2] &&
        ch.k5.size() == ch.k3.size() &&
        std::equal(ch.k5.begin(), ch.k5.end(), ch.k3.begin()) &&
        Dim_ls[2] == Dim_ls[0] && ell0[2] == ell0[0] &&
        Dim_ls[4] == Dim_ls[3] && ell0[4] == ell0[3] &&
        Dim_ls[5] == Dim_ls[4] && ell0[5] == ell0[4];
}


inline bool can_derive_B3_from_B2_exchange34_coherent(
    const B4Spins& s,
    const std::array<std::size_t,6>& Dim_ls,
    const std::array<std::size_t,6>& ell0,
    const AmplitudeChannels& ch)
{
    return
        s.Dj34 == s.Dj23 &&
        s.Dj35 == s.Dj24 &&
        s.Dj13 == s.Dj12 &&
        s.Dj23 == s.Dj25 &&
        ch.k3.size() == ch.k2.size() &&
        std::equal(ch.k3.begin(), ch.k3.end(), ch.k2.begin()) &&
        Dim_ls[3] == Dim_ls[0] && ell0[3] == ell0[0] &&
        Dim_ls[4] == Dim_ls[1] && ell0[4] == ell0[1] &&
        Dim_ls[0] == Dim_ls[2] && ell0[0] == ell0[2];
}

inline bool can_derive_B4_from_B2_pair_exchange_coherent(
    const B4Spins& s,
    const std::array<std::size_t,6>& Dim_ls,
    const std::array<std::size_t,6>& ell0,
    const AmplitudeChannels& ch)
{
    return
        s.Dj45 == s.Dj25 &&
        s.Dj14 == s.Dj12 &&
        s.Dj24 == s.Dj23 &&
        s.Dj34 == s.Dj24 &&
        ch.k4.size() == ch.k2.size() &&
        std::equal(ch.k4.begin(), ch.k4.end(), ch.k2.begin()) &&
        Dim_ls[5] == Dim_ls[2] && ell0[5] == ell0[2] &&
        Dim_ls[1] == Dim_ls[0] && ell0[1] == ell0[0] &&
        Dim_ls[3] == Dim_ls[1] && ell0[3] == ell0[1];
}

inline bool can_derive_B5_from_B3_pair_exchange_coherent(
    const B4Spins& s,
    const std::array<std::size_t,6>& Dim_ls,
    const std::array<std::size_t,6>& ell0,
    const AmplitudeChannels& ch)
{
    return
        s.Dj15 == s.Dj13 &&
        s.Dj25 == s.Dj23 &&
        s.Dj35 == s.Dj34 &&
        s.Dj45 == s.Dj35 &&
        ch.k5.size() == ch.k3.size() &&
        std::equal(ch.k5.begin(), ch.k5.end(), ch.k3.begin()) &&
        Dim_ls[2] == Dim_ls[0] && ell0[2] == ell0[0] &&
        Dim_ls[4] == Dim_ls[3] && ell0[4] == ell0[3] &&
        Dim_ls[5] == Dim_ls[4] && ell0[5] == ell0[4];
}

inline bool can_derive_B3_from_B2_exchange34_full(
    const B4Spins& s,
    const std::array<std::size_t,6>& Dim_ls,
    const std::array<std::size_t,6>& ell0,
    const FullTensorChannels& ch)
{
    return
        s.Dj34 == s.Dj23 &&
        s.Dj35 == s.Dj24 &&
        s.Dj13 == s.Dj12 &&
        s.Dj23 == s.Dj25 &&
        b4_equal_int_vectors(ch.i3, ch.i2) &&
        b4_equal_int_vectors(ch.k3, ch.k2) &&
        Dim_ls[3] == Dim_ls[0] && ell0[3] == ell0[0] &&
        Dim_ls[4] == Dim_ls[1] && ell0[4] == ell0[1] &&
        Dim_ls[0] == Dim_ls[2] && ell0[0] == ell0[2];
}

inline bool can_derive_B4_from_B2_pair_exchange_full(
    const B4Spins& s,
    const std::array<std::size_t,6>& Dim_ls,
    const std::array<std::size_t,6>& ell0,
    const FullTensorChannels& ch)
{
    return
        s.Dj45 == s.Dj25 &&
        s.Dj14 == s.Dj12 &&
        s.Dj24 == s.Dj23 &&
        s.Dj34 == s.Dj24 &&
        b4_equal_int_vectors(ch.i4, ch.i2) &&
        b4_equal_int_vectors(ch.k4, ch.k2) &&
        Dim_ls[5] == Dim_ls[2] && ell0[5] == ell0[2] &&
        Dim_ls[1] == Dim_ls[0] && ell0[1] == ell0[0] &&
        Dim_ls[3] == Dim_ls[1] && ell0[3] == ell0[1];
}

inline bool can_derive_B5_from_B3_pair_exchange_full(
    const B4Spins& s,
    const std::array<std::size_t,6>& Dim_ls,
    const std::array<std::size_t,6>& ell0,
    const FullTensorChannels& ch)
{
    return
        s.Dj15 == s.Dj13 &&
        s.Dj25 == s.Dj23 &&
        s.Dj35 == s.Dj34 &&
        s.Dj45 == s.Dj35 &&
        b4_equal_int_vectors(ch.i5, ch.i3) &&
        b4_equal_int_vectors(ch.k5, ch.k3) &&
        Dim_ls[2] == Dim_ls[0] && ell0[2] == ell0[0] &&
        Dim_ls[4] == Dim_ls[3] && ell0[4] == ell0[3] &&
        Dim_ls[5] == Dim_ls[4] && ell0[5] == ell0[4];
}

// -----------------------------------------------------------------------------
// Symmetry-aware single-amplitude rank-4 booster filling.
// -----------------------------------------------------------------------------

inline void fill_rank4_tensors_offset(
    const B4Spins& s,
    const std::array<std::size_t,6> Dim_ls,
    const std::array<std::size_t,6> ell0,
    double gamma,
    const std::vector<double>& LF,
    const AmplitudeChannels& ch,
    Tensor<double, 4>& T1,
    Tensor<double, 4>& T2,
    Tensor<double, 4>& T3,
    Tensor<double, 4>& T4,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0)
{
    std::array<std::size_t,3> Dim_lsLocal{Dim_ls[0], Dim_ls[1], Dim_ls[2]};
    std::array<std::size_t,3> ell0Local{ell0[0], ell0[1], ell0[2]};

    const bool reduce_B2_exchange12 =
        can_reduce_B2_exchange12_harvest_amp(s, Dim_ls, ell0);

    // B2 is the representative. If legs 1 and 2 are exchange-symmetric,
    // harvest only the upper triangular (l23,l24) half and complete it by
    // the analytic 1<->2 phase below.
    harvest_b4_rank4<4>(
        s.Dj23, s.Dj24, s.Dj25, s.Dj12,
        Dim_lsLocal,
        ch.i[1], ch.k2,
        gamma, LF,
        T1,
        ell0Local,
        reduce_B2_exchange12,
        prescription,
        rho_tile
    );

    if (reduce_B2_exchange12) {
        const int Dl23_base = s.Dj23 + 2 * static_cast<int>(ell0[0]);
        const int Dl24_base = s.Dj24 + 2 * static_cast<int>(ell0[1]);
        complete_rank4_exchange12_lower_from_upper(
            T1,
            s.Dj23, s.Dj24,
            Dl23_base, Dl24_base,
            ch.i[1], ch.k2
        );
    }

    const bool make_B3_from_B2 =
        can_derive_B3_from_B2_exchange34_amp(s, Dim_ls, ell0, ch);
    const bool make_B4_from_B2 =
        can_derive_B4_from_B2_pair_exchange_amp(s, Dim_ls, ell0, ch);
    const bool make_B5_from_B3 =
        can_derive_B5_from_B3_pair_exchange_amp(s, Dim_ls, ell0, ch);

    // B3: exchange legs 3<->4 of B2, if allowed.
    if (make_B3_from_B2) {
        const int Dl25_base = s.Dj25 + 2 * static_cast<int>(ell0[2]);
        copy_rank4_exchange34_same_axes_to_dst(
            T1, T2,
            s.Dj25, s.Dj12,
            Dl25_base,
            s.Dj12,
            ch.i[1], ch.k2
        );
    } else {
        Dim_lsLocal = {Dim_ls[3], Dim_ls[4], Dim_ls[0]};
        ell0Local   = {ell0[3],   ell0[4],   ell0[0]};
        harvest_b4_rank4<3>(
            s.Dj34, s.Dj35, s.Dj13, s.Dj23,
            Dim_lsLocal,
            ch.i[2], ch.k3,
            gamma, LF,
            T2,
            ell0Local,
            false,
            prescription,
            rho_tile
        );
    }

    // B4: exchange pairs (12)<->(34) of B2, if allowed.
    if (make_B4_from_B2) {
        copy_rank4_pair_exchange_120_to_dst(T1, T3, ch.i[1], ch.k2);
    } else {
        Dim_lsLocal = {Dim_ls[5], Dim_ls[1], Dim_ls[3]};
        ell0Local   = {ell0[5],   ell0[1],   ell0[3]};
        harvest_b4_rank4<2>(
            s.Dj45, s.Dj14, s.Dj24, s.Dj34,
            Dim_lsLocal,
            ch.i[3], ch.k4,
            gamma, LF,
            T3,
            ell0Local,
            false,
            prescription,
            rho_tile
        );
    }

    // B5: exchange pairs (12)<->(34) of B3, if allowed.
    // If B3 was derived from B2, this still avoids the B5 harvest.
    if (make_B5_from_B3) {
        copy_rank4_pair_exchange_120_to_dst(T2, T4, ch.i[2], ch.k3);
    } else {
        Dim_lsLocal = {Dim_ls[2], Dim_ls[4], Dim_ls[5]};
        ell0Local   = {ell0[2],   ell0[4],   ell0[5]};
        harvest_b4_rank4<1>(
            s.Dj15, s.Dj25, s.Dj35, s.Dj45,
            Dim_lsLocal,
            ch.i[4], ch.k5,
            gamma, LF,
            T4,
            ell0Local,
            false,
            prescription,
            rho_tile
        );
    }

}

inline void fill_rank4_tensors(
    const B4Spins& s,
    const std::array<std::size_t,6> Dim_ls,
    double gamma,
    const std::vector<double>& LF,
    const AmplitudeChannels& ch,
    Tensor<double, 4>& T1,
    Tensor<double, 4>& T2,
    Tensor<double, 4>& T3,
    Tensor<double, 4>& T4,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0)
{
    fill_rank4_tensors_offset(
        s, Dim_ls, {0,0,0,0,0,0},
        gamma, LF, ch,
        T1, T2, T3, T4,
        prescription,
        rho_tile
    );
}

// -----------------------------------------------------------------------------
// Symmetry-aware full-tensor rank-5 booster filling.
// -----------------------------------------------------------------------------

inline void fill_rank5_tensors_offset(
    const B4Spins& s,
    const std::array<std::size_t,6> Dim_ls,
    const std::array<std::size_t,6> ell0,
    double gamma,
    const std::vector<double>& LF,
    const FullTensorChannels& ch,
    Tensor<double, 5>& T1,
    Tensor<double, 5>& T2,
    Tensor<double, 5>& T3,
    Tensor<double, 5>& T4,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0)
{
    std::array<std::size_t,3> Dim_lsLocal{Dim_ls[0], Dim_ls[1], Dim_ls[2]};
    std::array<std::size_t,3> ell0Local{ell0[0], ell0[1], ell0[2]};

    const bool reduce_B2_exchange12 =
        can_reduce_B2_exchange12_harvest_full(s, Dim_ls, ell0);

    // B2 is the representative. If legs 1 and 2 are exchange-symmetric,
    // harvest only the upper triangular (l23,l24) half and complete it by
    // the analytic 1<->2 phase below.
    harvest_b4<4, 5>(
        s.Dj23, s.Dj24, s.Dj25, s.Dj12,
        Dim_lsLocal,
        ch.i2, ch.k2,
        gamma, LF,
        T1,
        ell0Local,
        reduce_B2_exchange12,
        prescription,
        rho_tile
    );

    if (reduce_B2_exchange12) {
        const int Dl23_base = s.Dj23 + 2 * static_cast<int>(ell0[0]);
        const int Dl24_base = s.Dj24 + 2 * static_cast<int>(ell0[1]);
        complete_rank5_exchange12_lower_from_upper(
            T1,
            s.Dj23, s.Dj24,
            Dl23_base, Dl24_base,
            ch.i2, ch.k2
        );
    }

    const bool make_B3_from_B2 =
        can_derive_B3_from_B2_exchange34_full(s, Dim_ls, ell0, ch);
    const bool make_B4_from_B2 =
        can_derive_B4_from_B2_pair_exchange_full(s, Dim_ls, ell0, ch);
    const bool make_B5_from_B3 =
        can_derive_B5_from_B3_pair_exchange_full(s, Dim_ls, ell0, ch);

    // B3: exchange legs 3<->4 of B2, if allowed.
    if (make_B3_from_B2) {
        const int Dl25_base = s.Dj25 + 2 * static_cast<int>(ell0[2]);
        copy_rank5_exchange34_same_axes_to_dst(
            T1, T2,
            s.Dj25, s.Dj12,
            Dl25_base,
            s.Dj12,
            ch.i2, ch.k2
        );
    } else {
        Dim_lsLocal = {Dim_ls[3], Dim_ls[4], Dim_ls[0]};
        ell0Local   = {ell0[3],   ell0[4],   ell0[0]};
        harvest_b4<3, 5>(
            s.Dj34, s.Dj35, s.Dj13, s.Dj23,
            Dim_lsLocal,
            ch.i3, ch.k3,
            gamma, LF,
            T2,
            ell0Local,
            false,
            prescription,
            rho_tile
        );
    }

    // B4: exchange pairs (12)<->(34) of B2, if allowed.
    if (make_B4_from_B2) {
        copy_rank5_pair_exchange_120_to_dst(T1, T3, ch.i2, ch.k2);
    } else {
        Dim_lsLocal = {Dim_ls[5], Dim_ls[1], Dim_ls[3]};
        ell0Local   = {ell0[5],   ell0[1],   ell0[3]};
        harvest_b4<2, 5>(
            s.Dj45, s.Dj14, s.Dj24, s.Dj34,
            Dim_lsLocal,
            ch.i4, ch.k4,
            gamma, LF,
            T3,
            ell0Local,
            false,
            prescription,
            rho_tile
        );
    }

    // B5: exchange pairs (12)<->(34) of B3, if allowed.
    if (make_B5_from_B3) {
        copy_rank5_pair_exchange_120_to_dst(T2, T4, ch.i3, ch.k3);
    } else {
        Dim_lsLocal = {Dim_ls[2], Dim_ls[4], Dim_ls[5]};
        ell0Local   = {ell0[2],   ell0[4],   ell0[5]};
        harvest_b4<1, 5>(
            s.Dj15, s.Dj25, s.Dj35, s.Dj45,
            Dim_lsLocal,
            ch.i5, ch.k5,
            gamma, LF,
            T4,
            ell0Local,
            false,
            prescription,
            rho_tile
        );
    }

}

inline void fill_rank5_tensors(
    const B4Spins& s,
    const std::array<std::size_t,6> Dim_ls,
    double gamma,
    const std::vector<double>& LF,
    const FullTensorChannels& ch,
    Tensor<double, 5>& T1,
    Tensor<double, 5>& T2,
    Tensor<double, 5>& T3,
    Tensor<double, 5>& T4,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0)
{
    fill_rank5_tensors_offset(
        s, Dim_ls, {0,0,0,0,0,0},
        gamma, LF, ch,
        T1, T2, T3, T4,
        prescription,
        rho_tile
    );
}




// -----------------------------------------------------------------------------
// Coherent-state rank-4 booster filling, direct no-atomic b-tile variant.
// -----------------------------------------------------------------------------
// This path keeps the boundary intertwiner index explicit inside the B4 recurrence
// contraction and accumulates directly into complex rank-4 native-layout outputs:
//
//   B2: (l23,l25,l24,k2)
//   B3: (l23,l34,l35,k3)
//   B4: (l45,l24,l34,k4)
//   B5: (l45,l25,l35,k5)
//
// No rank-5 real slab is materialized.  To avoid atomics, each OpenMP thread
// accumulates into a private canonical b-tile, then the thread-local tiles are
// reduced once into the native output layout.

struct B4CoherentCoefficientsView {
    std::span<const int> doubled_iotas;
    std::span<const b4utils::Complex64> coefficients;

    std::size_t size() const noexcept { return doubled_iotas.size(); }
    bool empty() const noexcept { return doubled_iotas.empty(); }
};

inline B4CoherentCoefficientsView make_b4_coherent_coefficients_view(
    const std::vector<int>& doubled_iotas,
    const std::vector<b4utils::Complex64>& coefficients) noexcept
{
    return B4CoherentCoefficientsView{
        std::span<const int>(doubled_iotas.data(), doubled_iotas.size()),
        std::span<const b4utils::Complex64>(coefficients.data(), coefficients.size())};
}

inline void b4_assert_valid_coherent_coefficients(
    const B4CoherentCoefficientsView& C)
{
#ifndef NDEBUG
    assert(C.doubled_iotas.size() == C.coefficients.size());
#endif
    if (C.doubled_iotas.size() != C.coefficients.size()) {
        throw b4utils::InternalFailure(
            "invalid coherent coefficient view: iota/coeff size mismatch");
    }
}

inline bool b4_complex_is_exact_zero(b4utils::Complex64 z) noexcept {
    return z.re == 0.0 && z.im == 0.0;
}

template <int FixedIdx>
inline std::size_t b4_native_rank4_offset_from_canonical_l(
    std::size_t i0,
    std::size_t i1,
    std::size_t i2,
    std::size_t ib,
    const Tensor<b4utils::Complex64, 4>& out) noexcept;

template <int FixedIdx>
inline void b4_check_native_coherent_output_dims(
    const std::array<std::size_t, 3>& dimensions,
    std::size_t b_size,
    const Tensor<b4utils::Complex64, 4>& output);

#ifndef B4_COHERENT_RANK5_SLAB_TARGET_BYTES
#define B4_COHERENT_RANK5_SLAB_TARGET_BYTES (1024ull * 1024ull * 1024ull)
#endif

struct B4ActiveCoherentCoefficients {
    std::vector<int> a;
    std::vector<b4utils::Complex64> c;

    std::size_t size() const noexcept { return a.size(); }
    bool empty() const noexcept { return a.empty(); }
};

inline B4ActiveCoherentCoefficients b4_make_active_coherent_coefficients(
    const B4CoherentCoefficientsView& C)
{
    b4_assert_valid_coherent_coefficients(C);

    B4ActiveCoherentCoefficients active;
    active.a.reserve(C.size());
    active.c.reserve(C.size());

    for (std::size_t i = 0; i < C.size(); ++i) {
        const b4utils::Complex64 z = C.coefficients[i];
        if (b4_complex_is_exact_zero(z)) continue;
        active.a.push_back(C.doubled_iotas[i]);
        active.c.push_back(z);
    }

    return active;
}


struct B4CoherentCoeffUnion4 {
    std::vector<int> a;
    std::vector<b4utils::Complex64> c2;
    std::vector<b4utils::Complex64> c3;
    std::vector<b4utils::Complex64> c4;
    std::vector<b4utils::Complex64> c5;

    std::size_t size() const noexcept { return a.size(); }
    bool empty() const noexcept { return a.empty(); }
};

inline void b4_add_active_iotas_unique(
    std::vector<int>& a_union,
    const B4CoherentCoefficientsView& C)
{
    b4_assert_valid_coherent_coefficients(C);
    for (std::size_t i = 0; i < C.size(); ++i) {
        if (b4_complex_is_exact_zero(C.coefficients[i])) continue;
        const int a = C.doubled_iotas[i];
        if (std::find(a_union.begin(), a_union.end(), a) == a_union.end()) {
            a_union.push_back(a);
        }
    }
}

inline b4utils::Complex64 b4_find_coherent_coeff_for_iota(
    const B4CoherentCoefficientsView& C,
    int doubled_iota) noexcept
{
    for (std::size_t i = 0; i < C.size(); ++i) {
        if (C.doubled_iotas[i] == doubled_iota) return C.coefficients[i];
    }
    return b4utils::Complex64{0.0, 0.0};
}

inline B4CoherentCoeffUnion4 b4_make_coherent_coeff_union4(
    bool use_C2,
    const B4CoherentCoefficientsView& C2,
    bool use_C3,
    const B4CoherentCoefficientsView& C3,
    bool use_C4,
    const B4CoherentCoefficientsView& C4,
    bool use_C5,
    const B4CoherentCoefficientsView& C5)
{
    std::vector<int> a_union;
    if (use_C2) b4_add_active_iotas_unique(a_union, C2);
    if (use_C3) b4_add_active_iotas_unique(a_union, C3);
    if (use_C4) b4_add_active_iotas_unique(a_union, C4);
    if (use_C5) b4_add_active_iotas_unique(a_union, C5);

    std::sort(a_union.begin(), a_union.end());
    a_union.erase(std::unique(a_union.begin(), a_union.end()), a_union.end());

    B4CoherentCoeffUnion4 out;
    out.a = std::move(a_union);
    out.c2.resize(out.a.size(), b4utils::Complex64{0.0, 0.0});
    out.c3.resize(out.a.size(), b4utils::Complex64{0.0, 0.0});
    out.c4.resize(out.a.size(), b4utils::Complex64{0.0, 0.0});
    out.c5.resize(out.a.size(), b4utils::Complex64{0.0, 0.0});

    for (std::size_t i = 0; i < out.a.size(); ++i) {
        const int a = out.a[i];
        if (use_C2) out.c2[i] = b4_find_coherent_coeff_for_iota(C2, a);
        if (use_C3) out.c3[i] = b4_find_coherent_coeff_for_iota(C3, a);
        if (use_C4) out.c4[i] = b4_find_coherent_coeff_for_iota(C4, a);
        if (use_C5) out.c5[i] = b4_find_coherent_coeff_for_iota(C5, a);
    }
    return out;
}

inline bool b4_any_nonzero_coeff(const std::vector<b4utils::Complex64>& c) noexcept {
    for (const b4utils::Complex64 z : c) {
        if (!b4_complex_is_exact_zero(z)) return true;
    }
    return false;
}

inline std::size_t b4_choose_coherent_rank5_b_slab_size(
    std::size_t d_block,
    std::size_t a_count,
    std::size_t b_size,
    std::uint64_t target_bytes)
{
    if (d_block == 0 || a_count == 0 || b_size == 0) return 0;

    const unsigned long long denom =
        static_cast<unsigned long long>(d_block) *
        static_cast<unsigned long long>(a_count) *
        static_cast<unsigned long long>(sizeof(double));

    if (denom == 0ull) return 1;

    std::size_t nb = static_cast<std::size_t>(target_bytes / denom);
    nb = std::max<std::size_t>(1, nb);
    nb = std::min<std::size_t>(nb, b_size);
    return nb;
}

template <int FixedIdx>
inline void b4_contract_rank5_slab_to_native_complex(
    const std::array<std::size_t,3>& Dim_ls,
    std::size_t b_global_begin,
    const std::vector<b4utils::Complex64>& coeff,
    const Tensor<double,5>& real_slab,
    Tensor<b4utils::Complex64,4>& out)
{
    static_assert(FixedIdx >= 1 && FixedIdx <= 4,
                  "FixedIdx must be in {1,2,3,4}");

    const auto& sd = real_slab.dims();
    const auto st = real_slab.strides();

    const std::size_t a_count = sd[3];
    const std::size_t nb = sd[4];

#ifndef NDEBUG
    assert(sd[0] == Dim_ls[0]);
    assert(sd[1] == Dim_ls[1]);
    assert(sd[2] == Dim_ls[2]);
    assert(coeff.size() == a_count);
#endif

    if (a_count == 0 || nb == 0) return;

    const double* __restrict x = real_slab.data();
    b4utils::Complex64* __restrict y = out.data();

    // Each (local b, canonical i2) tile maps to a disjoint region of the
    // native rank-4 output.  Therefore this pass needs no atomics and no
    // thread-local reduction.  The loop order keeps i0 stride-one for the slab.
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(nb * Dim_ls[2] >= 32)
#endif
    for (std::ptrdiff_t ib_s = 0;
         ib_s < static_cast<std::ptrdiff_t>(nb);
         ++ib_s)
    {
        for (std::ptrdiff_t i2_s = 0;
             i2_s < static_cast<std::ptrdiff_t>(Dim_ls[2]);
             ++i2_s)
        {
            const std::size_t ib_local = static_cast<std::size_t>(ib_s);
            const std::size_t ib_global = b_global_begin + ib_local;
            const std::size_t i2 = static_cast<std::size_t>(i2_s);

            for (std::size_t ia = 0; ia < a_count; ++ia) {
                const double cr = coeff[ia].re;
                const double ci = coeff[ia].im;

                const std::size_t ab_i2_base =
                    ia * st[3] + ib_local * st[4] + i2 * st[2];

                for (std::size_t i1 = 0; i1 < Dim_ls[1]; ++i1) {
                    const std::size_t slab_base = ab_i2_base + i1 * st[1];

                    for (std::size_t i0 = 0; i0 < Dim_ls[0]; ++i0) {
                        const double v = x[slab_base + i0 * st[0]];
                        if (v == 0.0) continue;

                        const std::size_t dst =
                            b4_native_rank4_offset_from_canonical_l<FixedIdx>(
                                i0, i1, i2, ib_global, out);

                        b4utils::Complex64& z = y[dst];
                        z.re += cr * v;
                        z.im += ci * v;
                    }
                }
            }
        }
    }
}



template <int FixedIdx>
inline void b4_contract_rank5_slab_exchange34_same_axes_to_native_complex(
    const std::array<std::size_t,3>& Dim_ls,
    std::size_t b_global_begin,
    int Dj3,
    int Dj4,
    int Dl3_base,
    int Dl4_fixed,
    std::span<const int> a_vals,
    std::span<const int> b_vals,
    const std::vector<b4utils::Complex64>& coeff,
    const Tensor<double,5>& real_slab,
    Tensor<b4utils::Complex64,4>& out)
{
    static_assert(FixedIdx >= 1 && FixedIdx <= 4,
                  "FixedIdx must be in {1,2,3,4}");

    const auto& sd = real_slab.dims();
    const auto st = real_slab.strides();

#ifndef NDEBUG
    assert(sd[0] == Dim_ls[0]);
    assert(sd[1] == Dim_ls[1]);
    assert(sd[2] == Dim_ls[2]);
    assert(a_vals.size() == sd[3]);
    assert(coeff.size() == sd[3]);
    assert(b_vals.size() == sd[4]);
#endif

    const std::size_t a_count = sd[3];
    const std::size_t nb = sd[4];
    if (a_count == 0 || nb == 0) return;

    const double* __restrict x = real_slab.data();
    b4utils::Complex64* __restrict y = out.data();

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(nb * Dim_ls[2] >= 32)
#endif
    for (std::ptrdiff_t ib_s = 0; ib_s < static_cast<std::ptrdiff_t>(nb); ++ib_s) {
        for (std::ptrdiff_t i2_s = 0; i2_s < static_cast<std::ptrdiff_t>(Dim_ls[2]); ++i2_s) {
            const std::size_t ib_local = static_cast<std::size_t>(ib_s);
            const std::size_t ib_global = b_global_begin + ib_local;
            const std::size_t i2 = static_cast<std::size_t>(i2_s);
            const int Db = b_vals[ib_local];
            const int Dl3 = Dl3_base + 2 * static_cast<int>(i2);

            for (std::size_t ia = 0; ia < a_count; ++ia) {
                const b4utils::Complex64 c = coeff[ia];
                if (b4_complex_is_exact_zero(c)) continue;

                const int Da = a_vals[ia];
                const double phase = static_cast<double>(
                    b4_phase_from_even_Dsum_local(Dj3 + Dj4 + Da + Dl3 + Dl4_fixed + Db));
                const double cr = phase * c.re;
                const double ci = phase * c.im;

                const std::size_t ab_i2_base =
                    ia * st[3] + ib_local * st[4] + i2 * st[2];

                for (std::size_t i1 = 0; i1 < Dim_ls[1]; ++i1) {
                    const std::size_t slab_base = ab_i2_base + i1 * st[1];
                    for (std::size_t i0 = 0; i0 < Dim_ls[0]; ++i0) {
                        const double v = x[slab_base + i0 * st[0]];
                        if (v == 0.0) continue;

                        const std::size_t dst =
                            b4_native_rank4_offset_from_canonical_l<FixedIdx>(
                                i0, i1, i2, ib_global, out);
                        b4utils::Complex64& z = y[dst];
                        z.re += cr * v;
                        z.im += ci * v;
                    }
                }
            }
        }
    }
}

template <int FixedIdx>
inline void b4_contract_rank5_slab_pair_exchange_120_to_native_complex(
    const std::array<std::size_t,3>& DstDim_ls,
    std::size_t b_global_begin,
    std::span<const int> a_vals,
    std::span<const int> b_vals,
    const std::vector<b4utils::Complex64>& coeff,
    const Tensor<double,5>& src_slab,
    Tensor<b4utils::Complex64,4>& out)
{
    static_assert(FixedIdx >= 1 && FixedIdx <= 4,
                  "FixedIdx must be in {1,2,3,4}");

    const auto& sd = src_slab.dims();
    const auto st = src_slab.strides();

#ifndef NDEBUG
    // dst(i0,i1,i2,a,b) = phase(a,b) * src(i1,i2,i0,a,b)
    assert(sd[0] == DstDim_ls[1]);
    assert(sd[1] == DstDim_ls[2]);
    assert(sd[2] == DstDim_ls[0]);
    assert(a_vals.size() == sd[3]);
    assert(coeff.size() == sd[3]);
    assert(b_vals.size() == sd[4]);
#endif

    const std::size_t a_count = sd[3];
    const std::size_t nb = sd[4];
    if (a_count == 0 || nb == 0) return;

    const double* __restrict x = src_slab.data();
    b4utils::Complex64* __restrict y = out.data();

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(nb * DstDim_ls[2] >= 32)
#endif
    for (std::ptrdiff_t ib_s = 0; ib_s < static_cast<std::ptrdiff_t>(nb); ++ib_s) {
        for (std::ptrdiff_t i2_s = 0; i2_s < static_cast<std::ptrdiff_t>(DstDim_ls[2]); ++i2_s) {
            const std::size_t ib_local = static_cast<std::size_t>(ib_s);
            const std::size_t ib_global = b_global_begin + ib_local;
            const std::size_t i2 = static_cast<std::size_t>(i2_s);
            const int Db = b_vals[ib_local];

            for (std::size_t ia = 0; ia < a_count; ++ia) {
                const b4utils::Complex64 c = coeff[ia];
                if (b4_complex_is_exact_zero(c)) continue;

                const double phase = static_cast<double>(
                    b4_phase_pair_exchange_local(a_vals[ia], Db));
                const double cr = phase * c.re;
                const double ci = phase * c.im;

                for (std::size_t i1 = 0; i1 < DstDim_ls[1]; ++i1) {
                    for (std::size_t i0 = 0; i0 < DstDim_ls[0]; ++i0) {
                        const std::size_t src_idx =
                            i1 * st[0] + i2 * st[1] + i0 * st[2] +
                            ia * st[3] + ib_local * st[4];
                        const double v = x[src_idx];
                        if (v == 0.0) continue;

                        const std::size_t dst =
                            b4_native_rank4_offset_from_canonical_l<FixedIdx>(
                                i0, i1, i2, ib_global, out);
                        b4utils::Complex64& z = y[dst];
                        z.re += cr * v;
                        z.im += ci * v;
                    }
                }
            }
        }
    }
}

template <int FixedIdx>
inline void b4_contract_rank5_slab_exchange34_then_pair120_to_native_complex(
    const std::array<std::size_t,3>& DstDim_ls,
    std::size_t b_global_begin,
    int Dj3,
    int Dj4,
    int Dl3_base,
    int Dl4_fixed,
    std::span<const int> a_vals,
    std::span<const int> b_vals,
    const std::vector<b4utils::Complex64>& coeff,
    const Tensor<double,5>& src_slab,
    Tensor<b4utils::Complex64,4>& out)
{
    static_assert(FixedIdx >= 1 && FixedIdx <= 4,
                  "FixedIdx must be in {1,2,3,4}");

    const auto& sd = src_slab.dims();
    const auto st = src_slab.strides();

#ifndef NDEBUG
    // B3_from_B2(j0,j1,j2) = phase34(j2) * B2(j0,j1,j2)
    // B5(i0,i1,i2)         = phase_pair * B3(i1,i2,i0)
    // hence source B2 index is (i1,i2,i0).
    assert(sd[0] == DstDim_ls[1]);
    assert(sd[1] == DstDim_ls[2]);
    assert(sd[2] == DstDim_ls[0]);
    assert(a_vals.size() == sd[3]);
    assert(coeff.size() == sd[3]);
    assert(b_vals.size() == sd[4]);
#endif

    const std::size_t a_count = sd[3];
    const std::size_t nb = sd[4];
    if (a_count == 0 || nb == 0) return;

    const double* __restrict x = src_slab.data();
    b4utils::Complex64* __restrict y = out.data();

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if(nb * DstDim_ls[2] >= 32)
#endif
    for (std::ptrdiff_t ib_s = 0; ib_s < static_cast<std::ptrdiff_t>(nb); ++ib_s) {
        for (std::ptrdiff_t i2_s = 0; i2_s < static_cast<std::ptrdiff_t>(DstDim_ls[2]); ++i2_s) {
            const std::size_t ib_local = static_cast<std::size_t>(ib_s);
            const std::size_t ib_global = b_global_begin + ib_local;
            const std::size_t i2 = static_cast<std::size_t>(i2_s);
            const int Db = b_vals[ib_local];

            for (std::size_t ia = 0; ia < a_count; ++ia) {
                const b4utils::Complex64 c = coeff[ia];
                if (b4_complex_is_exact_zero(c)) continue;
                const int Da = a_vals[ia];
                const double phase_pair = static_cast<double>(
                    b4_phase_pair_exchange_local(Da, Db));

                for (std::size_t i1 = 0; i1 < DstDim_ls[1]; ++i1) {
                    for (std::size_t i0 = 0; i0 < DstDim_ls[0]; ++i0) {
                        const int Dl3 = Dl3_base + 2 * static_cast<int>(i0);
                        const double phase34 = static_cast<double>(
                            b4_phase_from_even_Dsum_local(Dj3 + Dj4 + Da + Dl3 + Dl4_fixed + Db));
                        const double phase = phase_pair * phase34;
                        const double cr = phase * c.re;
                        const double ci = phase * c.im;

                        const std::size_t src_idx =
                            i1 * st[0] + i2 * st[1] + i0 * st[2] +
                            ia * st[3] + ib_local * st[4];
                        const double v = x[src_idx];
                        if (v == 0.0) continue;

                        const std::size_t dst =
                            b4_native_rank4_offset_from_canonical_l<FixedIdx>(
                                i0, i1, i2, ib_global, out);
                        b4utils::Complex64& z = y[dst];
                        z.re += cr * v;
                        z.im += ci * v;
                    }
                }
            }
        }
    }
}

template <int FixedIdx>
void harvest_b4_rank4_coherent_rank5_slabbed_native(
    int Dj1, int Dj2, int Dj3, int Dj4,
    const std::array<std::size_t,3> Dim_ls,
    const B4CoherentCoefficientsView& C,
    std::span<const int> b_vals,
    double gamma,
    const std::vector<double>& LF,
    Tensor<b4utils::Complex64, 4>& out,
    const std::array<std::size_t,3> ell0 = {0,0,0},
    std::uint64_t slab_target_bytes = B4_COHERENT_RANK5_SLAB_TARGET_BYTES,
    bool reduce_exchange12_harvest = false,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0)
{
    static_assert(FixedIdx >= 1 && FixedIdx <= 4,
                  "FixedIdx must be in {1,2,3,4}");

    b4_assert_valid_coherent_coefficients(C);

    const std::size_t b_size = b_vals.size();
    b4_check_native_coherent_output_dims<FixedIdx>(Dim_ls, b_size, out);

    b4utils::Complex64* __restrict out_ptr = out.data();
    const std::size_t out_size = out.size();
    std::fill(out_ptr, out_ptr + out_size, b4utils::Complex64{0.0, 0.0});

    if (out_size == 0 || b_size == 0 || C.empty()) return;

    B4ActiveCoherentCoefficients active = b4_make_active_coherent_coefficients(C);
    if (active.empty()) return;

    const std::size_t d_block = Dim_ls[0] * Dim_ls[1] * Dim_ls[2];
    const std::size_t b_slab = b4_choose_coherent_rank5_b_slab_size(
        d_block, active.size(), b_size, slab_target_bytes);
    if (b_slab == 0) return;

#if B4_ENABLE_DIAGNOSTICS
    {
        const double mib = 1024.0 * 1024.0;
        const std::uint64_t slab_bytes = static_cast<std::uint64_t>(d_block) *
            static_cast<std::uint64_t>(active.size()) *
            static_cast<std::uint64_t>(b_slab) *
            static_cast<std::uint64_t>(sizeof(double));
        std::cerr << "coherent rank5 slab harvester<" << FixedIdx << ">\n"
                  << "  active boundary intertwiner = " << active.size() << "\n"
                  << "  b_size       = " << b_size << "\n"
                  << "  b_slab       = " << b_slab << "\n"
                  << "  slab bytes   = " << (static_cast<double>(slab_bytes) / mib) << " MiB\n";
    }
#endif

    const std::span<const int> a_span(active.a.data(), active.a.size());

    const bool do_reduce_exchange12 =
        reduce_exchange12_harvest &&
        (FixedIdx == 3 || FixedIdx == 4) &&
        Dj1 == Dj2 &&
        Dim_ls[0] == Dim_ls[1] &&
        ell0[0] == ell0[1];

    // Allocate once at the maximum chosen slab size and reuse it.  Only the
    // last partial slab, when present, gets its own smaller tensor because
    // harvest_b4 requires dimensions matching b_vals.size().  This avoids the
    // per-boundary intertwiner rank-4 disaster while keeping peak memory bounded.
    Tensor<double, 5> real_slab_full(
        {Dim_ls[0], Dim_ls[1], Dim_ls[2], active.size(), b_slab},
        false);

    for (std::size_t b_begin = 0; b_begin < b_size; b_begin += b_slab) {
        const std::size_t nb = std::min(b_slab, b_size - b_begin);
        const std::span<const int> b_sub(b_vals.data() + b_begin, nb);

        if (nb == b_slab) {
            harvest_b4<FixedIdx, 5>(
                Dj1, Dj2, Dj3, Dj4,
                Dim_ls,
                a_span,
                b_sub,
                gamma,
                LF,
                real_slab_full,
                ell0,
                do_reduce_exchange12,
                prescription,
                rho_tile);

            if (do_reduce_exchange12) {
                const int Dl1_base = Dj1 + 2 * static_cast<int>(ell0[0]);
                const int Dl2_base = Dj2 + 2 * static_cast<int>(ell0[1]);
                complete_rank5_exchange12_lower_from_upper(
                    real_slab_full, Dj1, Dj2, Dl1_base, Dl2_base, a_span, b_sub);
            }

            b4_contract_rank5_slab_to_native_complex<FixedIdx>(
                Dim_ls, b_begin, active.c, real_slab_full, out);
        } else {
            Tensor<double, 5> real_slab_tail(
                {Dim_ls[0], Dim_ls[1], Dim_ls[2], active.size(), nb},
                false);

            harvest_b4<FixedIdx, 5>(
                Dj1, Dj2, Dj3, Dj4,
                Dim_ls,
                a_span,
                b_sub,
                gamma,
                LF,
                real_slab_tail,
                ell0,
                do_reduce_exchange12,
                prescription,
                rho_tile);

            if (do_reduce_exchange12) {
                const int Dl1_base = Dj1 + 2 * static_cast<int>(ell0[0]);
                const int Dl2_base = Dj2 + 2 * static_cast<int>(ell0[1]);
                complete_rank5_exchange12_lower_from_upper(
                    real_slab_tail, Dj1, Dj2, Dl1_base, Dl2_base, a_span, b_sub);
            }

            b4_contract_rank5_slab_to_native_complex<FixedIdx>(
                Dim_ls, b_begin, active.c, real_slab_tail, out);
        }
    }
}

template <int FixedIdx>
inline std::size_t b4_native_rank4_offset_from_canonical_l(
    std::size_t i0,
    std::size_t i1,
    std::size_t i2,
    std::size_t ib,
    const Tensor<b4utils::Complex64, 4>& out) noexcept
{
    const auto st = out.strides();

    if constexpr (FixedIdx == 4) {
        // B2 recurrence canonical: (l23,l24,l25,k2)
        // B2 native output:        (l23,l25,l24,k2)
        return i0 * st[0] + i2 * st[1] + i1 * st[2] + ib * st[3];
    } else if constexpr (FixedIdx == 3) {
        // B3 recurrence canonical: (l34,l35,l23,k3)
        // B3 native output:        (l23,l34,l35,k3)
        return i2 * st[0] + i0 * st[1] + i1 * st[2] + ib * st[3];
    } else if constexpr (FixedIdx == 2) {
        // B4 recurrence canonical: (l45,l24,l34,k4)
        // B4 native output:        (l45,l24,l34,k4)
        return i0 * st[0] + i1 * st[1] + i2 * st[2] + ib * st[3];
    } else {
        static_assert(FixedIdx == 1, "FixedIdx must be in {1,2,3,4}");
        // B5 recurrence canonical: (l25,l35,l45,k5)
        // B5 native output:        (l45,l25,l35,k5)
        return i2 * st[0] + i0 * st[1] + i1 * st[2] + ib * st[3];
    }
}

template <int FixedIdx>
inline void b4_check_native_coherent_output_dims(
    const std::array<std::size_t,3>& Dim_ls,
    std::size_t b_size,
    const Tensor<b4utils::Complex64,4>& out)
{
    const auto& dr = out.dims();

    std::array<std::size_t,4> expected{};

    if constexpr (FixedIdx == 4) {
        expected = {Dim_ls[0], Dim_ls[2], Dim_ls[1], b_size};
    } else if constexpr (FixedIdx == 3) {
        expected = {Dim_ls[2], Dim_ls[0], Dim_ls[1], b_size};
    } else if constexpr (FixedIdx == 2) {
        expected = {Dim_ls[0], Dim_ls[1], Dim_ls[2], b_size};
    } else {
        static_assert(FixedIdx == 1, "FixedIdx must be in {1,2,3,4}");
        expected = {Dim_ls[2], Dim_ls[0], Dim_ls[1], b_size};
    }

    if (dr != expected) {
        throw b4utils::InternalFailure(
            "coherent native output tensor has wrong dimensions");
    }
}

inline void fill_rank4_tensors_coherent_contraction_native_offset(
    const B4Spins& s,
    const std::array<std::size_t,6> Dim_ls,
    const std::array<std::size_t,6> ell0,
    double gamma,
    const std::vector<double>& LF,
    const AmplitudeChannels& ch,
    const B4CoherentCoefficientsView& C2,
    const B4CoherentCoefficientsView& C3,
    const B4CoherentCoefficientsView& C4,
    const B4CoherentCoefficientsView& C5,
    Tensor<b4utils::Complex64, 4>& B2,
    Tensor<b4utils::Complex64, 4>& B3,
    Tensor<b4utils::Complex64, 4>& B4,
    Tensor<b4utils::Complex64, 4>& B5,
    std::uint64_t slab_target_bytes = B4_COHERENT_RANK5_SLAB_TARGET_BYTES,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0)
{
    b4_assert_valid_coherent_coefficients(C2);
    b4_assert_valid_coherent_coefficients(C3);
    b4_assert_valid_coherent_coefficients(C4);
    b4_assert_valid_coherent_coefficients(C5);

    // Public coherent outputs are always native rank-4 complex tensors.  Clear
    // them here because representative fanout may skip individual independent
    // fillers entirely.
    std::fill(B2.data(), B2.data() + B2.size(), b4utils::Complex64{0.0, 0.0});
    std::fill(B3.data(), B3.data() + B3.size(), b4utils::Complex64{0.0, 0.0});
    std::fill(B4.data(), B4.data() + B4.size(), b4utils::Complex64{0.0, 0.0});
    std::fill(B5.data(), B5.data() + B5.size(), b4utils::Complex64{0.0, 0.0});

    const bool make_B3_from_B2 =
        can_derive_B3_from_B2_exchange34_coherent(s, Dim_ls, ell0, ch);
    const bool make_B4_from_B2 =
        can_derive_B4_from_B2_pair_exchange_coherent(s, Dim_ls, ell0, ch);
    const bool make_B5_from_B3 =
        can_derive_B5_from_B3_pair_exchange_coherent(s, Dim_ls, ell0, ch);
    const bool make_B5_from_B2 = make_B3_from_B2 && make_B5_from_B3;

#if B4_ENABLE_DIAGNOSTICS
    std::cerr << "coherent booster symmetry plan\n"
              << "  B3 from B2 = " << (make_B3_from_B2 ? "yes" : "no") << "\n"
              << "  B4 from B2 = " << (make_B4_from_B2 ? "yes" : "no") << "\n"
              << "  B5 from B3 = " << (make_B5_from_B3 ? "yes" : "no") << "\n"
              << "  B5 from B2 = " << (make_B5_from_B2 ? "yes" : "no") << "\n";
#endif

    // ------------------------------------------------------------------
    // Representative B2 harvest.  Its real rank-5 slab can feed B2 itself
    // and, when the exact symmetry guards pass, B3/B4/B5.  The active boundary intertwiner
    // set is the union of all coherent maps needed by these targets; each
    // target gets its own coefficient vector aligned with that union.
    // ------------------------------------------------------------------
    {
        const B4CoherentCoeffUnion4 U = b4_make_coherent_coeff_union4(
            true, C2,
            make_B3_from_B2, C3,
            make_B4_from_B2, C4,
            make_B5_from_B2, C5);

        if (!U.empty()) {
            const std::array<std::size_t,3> D2loc{Dim_ls[0], Dim_ls[1], Dim_ls[2]};
            const std::array<std::size_t,3> E2loc{ell0[0], ell0[1], ell0[2]};
            const std::size_t d_block = D2loc[0] * D2loc[1] * D2loc[2];
            const std::size_t b_size = ch.k2.size();
            const std::size_t b_slab = b4_choose_coherent_rank5_b_slab_size(
                d_block, U.size(), b_size, slab_target_bytes);
            const bool reduce_B2_exchange12 = can_reduce_B2_exchange12_harvest_amp(s, Dim_ls, ell0);
            const std::span<const int> a_span(U.a.data(), U.a.size());

#if B4_ENABLE_DIAGNOSTICS
            {
                const double mib = 1024.0 * 1024.0;
                const std::uint64_t slab_bytes = static_cast<std::uint64_t>(d_block) *
                    static_cast<std::uint64_t>(U.size()) *
                    static_cast<std::uint64_t>(std::max<std::size_t>(std::size_t{1}, b_slab)) *
                    static_cast<std::uint64_t>(sizeof(double));
                std::cerr << "coherent rank5 representative B2 fanout\n"
                          << "  active boundary intertwiner = " << U.size() << "\n"
                          << "  b_size       = " << b_size << "\n"
                          << "  b_slab       = " << b_slab << "\n"
                          << "  slab bytes   = " << (static_cast<double>(slab_bytes) / mib) << " MiB\n"
                          << "  reduce 12    = " << (reduce_B2_exchange12 ? "yes" : "no") << "\n";
            }
#endif

            if (b_slab != 0) {
                Tensor<double,5> slab_full({D2loc[0], D2loc[1], D2loc[2], U.size(), b_slab}, false);

                for (std::size_t b_begin = 0; b_begin < b_size; b_begin += b_slab) {
                    const std::size_t nb = std::min(b_slab, b_size - b_begin);
                    const std::span<const int> b_sub(ch.k2.data() + b_begin, nb);

                    Tensor<double,5>* slab_ptr = &slab_full;
                    std::unique_ptr<Tensor<double,5>> slab_tail;
                    if (nb != b_slab) {
                        slab_tail = std::make_unique<Tensor<double,5>>(
                            std::array<std::size_t,5>{D2loc[0], D2loc[1], D2loc[2], U.size(), nb},
                            false);
                        slab_ptr = slab_tail.get();
                    }
                    Tensor<double,5>& slab = *slab_ptr;

                    harvest_b4<4,5>(
                        s.Dj23, s.Dj24, s.Dj25, s.Dj12,
                        D2loc, a_span, b_sub,
                        gamma, LF, slab, E2loc, reduce_B2_exchange12,
                        prescription,
                        rho_tile);

                    if (reduce_B2_exchange12) {
                        const int Dl23_base = s.Dj23 + 2 * static_cast<int>(ell0[0]);
                        const int Dl24_base = s.Dj24 + 2 * static_cast<int>(ell0[1]);
                        complete_rank5_exchange12_lower_from_upper(
                            slab, s.Dj23, s.Dj24, Dl23_base, Dl24_base, a_span, b_sub);
                    }

                    if (b4_any_nonzero_coeff(U.c2)) {
                        b4_contract_rank5_slab_to_native_complex<4>(
                            D2loc, b_begin, U.c2, slab, B2);
                    }
                    if (make_B3_from_B2 && b4_any_nonzero_coeff(U.c3)) {
                        const int Dl25_base = s.Dj25 + 2 * static_cast<int>(ell0[2]);
                        b4_contract_rank5_slab_exchange34_same_axes_to_native_complex<3>(
                            D2loc, b_begin,
                            s.Dj25, s.Dj12,
                            Dl25_base, s.Dj12,
                            a_span, b_sub, U.c3, slab, B3);
                    }
                    if (make_B4_from_B2 && b4_any_nonzero_coeff(U.c4)) {
                        const std::array<std::size_t,3> D4loc{Dim_ls[5], Dim_ls[1], Dim_ls[3]};
                        b4_contract_rank5_slab_pair_exchange_120_to_native_complex<2>(
                            D4loc, b_begin, a_span, b_sub, U.c4, slab, B4);
                    }
                    if (make_B5_from_B2 && b4_any_nonzero_coeff(U.c5)) {
                        const std::array<std::size_t,3> D5loc{Dim_ls[2], Dim_ls[4], Dim_ls[5]};
                        const int Dl25_base = s.Dj25 + 2 * static_cast<int>(ell0[2]);
                        b4_contract_rank5_slab_exchange34_then_pair120_to_native_complex<1>(
                            D5loc, b_begin,
                            s.Dj25, s.Dj12,
                            Dl25_base, s.Dj12,
                            a_span, b_sub, U.c5, slab, B5);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Fallback B3 representative.  If B3 could not be derived from B2, harvest
    // it.  It may still feed B5 through the pair-exchange symmetry.
    // ------------------------------------------------------------------
    if (!make_B3_from_B2) {
        const bool derive_B5_from_this_B3 = make_B5_from_B3;
        const B4CoherentCoeffUnion4 U = b4_make_coherent_coeff_union4(
            false, C2,
            true, C3,
            false, C4,
            derive_B5_from_this_B3, C5);

        if (!U.empty()) {
            const std::array<std::size_t,3> D3loc{Dim_ls[3], Dim_ls[4], Dim_ls[0]};
            const std::array<std::size_t,3> E3loc{ell0[3], ell0[4], ell0[0]};
            const std::size_t d_block = D3loc[0] * D3loc[1] * D3loc[2];
            const std::size_t b_size = ch.k3.size();
            const std::size_t b_slab = b4_choose_coherent_rank5_b_slab_size(
                d_block, U.size(), b_size, slab_target_bytes);
            const bool reduce_B3_exchange12 =
                s.Dj34 == s.Dj35 && Dim_ls[3] == Dim_ls[4] && ell0[3] == ell0[4];
            const std::span<const int> a_span(U.a.data(), U.a.size());

#if B4_ENABLE_DIAGNOSTICS
            std::cerr << "coherent rank5 representative B3 fanout\n"
                      << "  active boundary intertwiner = " << U.size() << "\n"
                      << "  b_size       = " << b_size << "\n"
                      << "  b_slab       = " << b_slab << "\n"
                      << "  reduce 12    = " << (reduce_B3_exchange12 ? "yes" : "no") << "\n";
#endif

            if (b_slab != 0) {
                Tensor<double,5> slab_full({D3loc[0], D3loc[1], D3loc[2], U.size(), b_slab}, false);

                for (std::size_t b_begin = 0; b_begin < b_size; b_begin += b_slab) {
                    const std::size_t nb = std::min(b_slab, b_size - b_begin);
                    const std::span<const int> b_sub(ch.k3.data() + b_begin, nb);

                    Tensor<double,5>* slab_ptr = &slab_full;
                    std::unique_ptr<Tensor<double,5>> slab_tail;
                    if (nb != b_slab) {
                        slab_tail = std::make_unique<Tensor<double,5>>(
                            std::array<std::size_t,5>{D3loc[0], D3loc[1], D3loc[2], U.size(), nb},
                            false);
                        slab_ptr = slab_tail.get();
                    }
                    Tensor<double,5>& slab = *slab_ptr;

                    harvest_b4<3,5>(
                        s.Dj34, s.Dj35, s.Dj13, s.Dj23,
                        D3loc, a_span, b_sub,
                        gamma, LF, slab, E3loc, reduce_B3_exchange12,
                        prescription,
                        rho_tile);

                    if (reduce_B3_exchange12) {
                        const int Dl34_base = s.Dj34 + 2 * static_cast<int>(ell0[3]);
                        const int Dl35_base = s.Dj35 + 2 * static_cast<int>(ell0[4]);
                        complete_rank5_exchange12_lower_from_upper(
                            slab, s.Dj34, s.Dj35, Dl34_base, Dl35_base, a_span, b_sub);
                    }

                    if (b4_any_nonzero_coeff(U.c3)) {
                        b4_contract_rank5_slab_to_native_complex<3>(
                            D3loc, b_begin, U.c3, slab, B3);
                    }
                    if (derive_B5_from_this_B3 && b4_any_nonzero_coeff(U.c5)) {
                        const std::array<std::size_t,3> D5loc{Dim_ls[2], Dim_ls[4], Dim_ls[5]};
                        b4_contract_rank5_slab_pair_exchange_120_to_native_complex<1>(
                            D5loc, b_begin, a_span, b_sub, U.c5, slab, B5);
                    }
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Fallback independent B4/B5 harvests.
    // ------------------------------------------------------------------
    if (!make_B4_from_B2) {
        harvest_b4_rank4_coherent_rank5_slabbed_native<2>(
            s.Dj45, s.Dj14, s.Dj24, s.Dj34,
            {Dim_ls[5], Dim_ls[1], Dim_ls[3]},
            C4, ch.k4, gamma, LF, B4,
            {ell0[5], ell0[1], ell0[3]}, slab_target_bytes, false,
            prescription,
            rho_tile);
    }

    if (!make_B5_from_B2 && !(make_B5_from_B3 && !make_B3_from_B2)) {
        harvest_b4_rank4_coherent_rank5_slabbed_native<1>(
            s.Dj15, s.Dj25, s.Dj35, s.Dj45,
            {Dim_ls[2], Dim_ls[4], Dim_ls[5]},
            C5, ch.k5, gamma, LF, B5,
            {ell0[2], ell0[4], ell0[5]}, slab_target_bytes, false,
            prescription,
            rho_tile);
    }

}

inline void fill_rank4_tensors_coherent_contraction_native(
    const B4Spins& s,
    const std::array<std::size_t,6> Dim_ls,
    double gamma,
    const std::vector<double>& LF,
    const AmplitudeChannels& ch,
    const B4CoherentCoefficientsView& C2,
    const B4CoherentCoefficientsView& C3,
    const B4CoherentCoefficientsView& C4,
    const B4CoherentCoefficientsView& C5,
    Tensor<b4utils::Complex64, 4>& B2,
    Tensor<b4utils::Complex64, 4>& B3,
    Tensor<b4utils::Complex64, 4>& B4,
    Tensor<b4utils::Complex64, 4>& B5,
    std::uint64_t slab_target_bytes = B4_COHERENT_RANK5_SLAB_TARGET_BYTES,
    sl2cfoamxx::SimplicityConstraint prescription =
        sl2cfoamxx::SimplicityConstraint::gamma_j,
    std::uint32_t rho_tile = 0)
{
    fill_rank4_tensors_coherent_contraction_native_offset(
        s, Dim_ls, {0,0,0,0,0,0}, gamma, LF, ch,
        C2, C3, C4, C5,
        B2, B3, B4, B5,
        slab_target_bytes,
        prescription,
        rho_tile);
}


#undef B4_NOINLINE
