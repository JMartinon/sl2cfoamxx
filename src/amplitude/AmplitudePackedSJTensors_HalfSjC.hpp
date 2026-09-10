// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "amplitude/AmplitudeInputSupport_HalfSjC.hpp"
#include "amplitude/AmplitudePackedSJTensors.hpp"
#include "wigner/SixJTensors.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#ifdef _OPENMP
#include "core/OpenMp.hpp"
#endif

namespace b4support {


#ifndef B4PACK_SJ_FILL_TIMING
#define B4PACK_SJ_FILL_TIMING 0
#endif

// ============================================================================
// Local runtime chunk selection.
//
// This header intentionally does not depend on AmplitudePackedSJFillChunks.hpp.
// The chunk heuristic is kept here so the packed sjX filler is self-contained.
// ============================================================================

struct PackedAmplitudeSJFillChunks {
    int sjA = 16;
    int sjB = 256;
    int sjC = 512;
    int sjD = 256;
    int sjE = 16;
    
    void print(std::ostream& os) const {
        os << "Packed amplitude sj fill OpenMP chunks\n";
        os << "======================================\n";
        os << "sjA chunk = " << sjA << "\n";
        os << "sjB chunk = " << sjB << "\n";
        os << "sjC chunk = " << sjC << "\n";
        os << "sjD chunk = " << sjD << "\n";
        os << "sjE chunk = " << sjE << "\n";
    }
};

inline int packed_sj_fill_num_threads(int requested_threads = 0) noexcept
{
    if (requested_threads > 0) {
        return requested_threads;
    }
#ifdef _OPENMP
    const int nt = omp_get_max_threads();
    return nt > 0 ? nt : 1;
#else
    return 1;
#endif
}

inline std::size_t packed_sj_mul_sat(std::size_t a, std::size_t b) noexcept
{
    if (a == 0u || b == 0u) {
        return 0u;
    }
    
    const std::size_t maxv = static_cast<std::size_t>(-1);
    if (a > maxv / b) {
        return maxv;
    }
    
    return a * b;
}

inline std::size_t packed_sj_ceil_div(std::size_t a, std::size_t b) noexcept
{
    return b == 0u ? a : (a + b - 1u) / b;
}

inline int packed_sj_nearest_power_of_two(std::size_t x) noexcept
{
    if (x <= 1u) {
        return 1;
    }
    
    std::size_t lo = 1u;
    while ((lo << 1u) > lo && (lo << 1u) <= x) {
        lo <<= 1u;
    }
    
    const std::size_t hi = lo << 1u;
    if (hi <= lo) {
        return static_cast<int>(std::min<std::size_t>(lo, 2147483647u));
    }
    
    const std::size_t chosen = (x - lo <= hi - x) ? lo : hi;
    return static_cast<int>(std::min<std::size_t>(chosen, 2147483647u));
}

inline int packed_sj_clamp_int(int x, int lo, int hi) noexcept
{
    return std::max(lo, std::min(x, hi));
}

inline int choose_dynamic_chunk(std::size_t total_iterations,
                                int num_threads,
                                int min_chunk,
                                int max_chunk,
                                int target_chunks_per_thread) noexcept
{
    if (total_iterations == 0u) {
        return 1;
    }
    
    num_threads = std::max(1, num_threads);
    target_chunks_per_thread = std::max(1, target_chunks_per_thread);
    
    const std::size_t denom =
    packed_sj_mul_sat(static_cast<std::size_t>(num_threads),
                      static_cast<std::size_t>(target_chunks_per_thread));
    
    std::size_t raw = packed_sj_ceil_div(total_iterations, denom);
    raw = std::max<std::size_t>(raw, static_cast<std::size_t>(min_chunk));
    raw = std::min<std::size_t>(raw, static_cast<std::size_t>(max_chunk));
    
    const int rounded = packed_sj_nearest_power_of_two(raw);
    return packed_sj_clamp_int(rounded, min_chunk, max_chunk);
}

PackedAmplitudeSJFillChunks choose_packed_amplitude_sj_fill_chunks_local(
                                                                                const AmplitudeInputSupportPlan_HalfSjC& support,
                                                                                int requested_threads = 0) noexcept;

// ============================================================================
// Hot-loop helpers.
// ============================================================================

inline Range1D axis_range_from_interval_regular2(const std::vector<int>& axis,
                                                 int Dmin,
                                                 int Dmax,
                                                 int parity) noexcept
{
    if (axis.empty()) {
        return Range1D{};
    }
    
    const int first = axis.front();
    const int last  = axis.back();
    
    int lo = std::max(Dmin, first);
    int hi = std::min(Dmax, last);
    
    if (lo > hi) {
        return Range1D{};
    }
    
    if ((lo & 1) != parity) {
        ++lo;
    }
    
    if ((hi & 1) != parity) {
        --hi;
    }
    
    if (lo > hi) {
        return Range1D{};
    }
    
    Range1D r;
    r.begin = static_cast<std::uint32_t>((lo - first) / 2);
    r.end   = static_cast<std::uint32_t>((hi - first) / 2 + 1);
    r.offset = 0;
    return r;
}

#ifndef NDEBUG
inline void assert_regular_step2_axis(const std::vector<int>& axis,
                                      const char* name)
{
    for (std::size_t i = 1; i < axis.size(); ++i) {
        if (axis[i] != axis[i - 1] + 2) {
            std::cerr << "Axis " << name << " is not a regular doubled-spin axis.\n";
            std::abort();
        }
    }
}
#endif

struct SjCStoredBlockTask {
    std::size_t ix = 0;
    std::size_t il24 = 0;
    std::size_t il34 = 0;
    std::size_t il35 = 0;
    
    std::uint32_t i_begin = 0;
    std::uint32_t i_end = 0;
    std::uint32_t j_begin = 0;
    std::uint32_t j_end = 0;
    std::uint64_t offset = 0;
};

std::vector<SjCStoredBlockTask>
make_sjC_stored_block_tasks(const AmplitudeInputSupportPlan_HalfSjC& support);

// Packed single-amplitude 6j tensors with half-stored sjC.
//
// This derives from PackedAmplitudeSJTensors so existing q-local T1/T3 helpers
// can keep accepting the packed object as a base reference.
struct PackedAmplitudeSJTensors_HalfSjC : public PackedAmplitudeSJTensors {
    void allocate_from_support(const AmplitudeInputSupportPlan_HalfSjC& support) {
        sjA.resize(static_cast<std::size_t>(support.sjA.packed_values));
        sjB.resize(static_cast<std::size_t>(support.sjB.packed_values));
        sjC.resize(static_cast<std::size_t>(support.sjC.packed_values));
        sjD.resize(static_cast<std::size_t>(support.sjD.packed_values));
        sjE.resize(static_cast<std::size_t>(support.sjE.packed_values));
    }
    
    void print(std::ostream& os) const {
        const double mib = 1024.0 * 1024.0;
        
        os << "\nPacked amplitude sj tensors, half-stored sjC\n";
        os << "==========================================\n";
        os << "sjA values = " << sjA.size()
        << " | " << static_cast<double>(sjA.size() * sizeof(double)) / mib
        << " MiB\n";
        os << "sjB values = " << sjB.size()
        << " | " << static_cast<double>(sjB.size() * sizeof(double)) / mib
        << " MiB\n";
        os << "sjC values = " << sjC.size()
        << " | " << static_cast<double>(sjC.size() * sizeof(double)) / mib
        << " MiB (half-stored logical sjC)\n";
        os << "sjD values = " << sjD.size()
        << " | " << static_cast<double>(sjD.size() * sizeof(double)) / mib
        << " MiB\n";
        os << "sjE values = " << sjE.size()
        << " | " << static_cast<double>(sjE.size() * sizeof(double)) / mib
        << " MiB\n";
        os << "total      = "
        << static_cast<double>(payload_bytes()) / mib
        << " MiB\n";
    }
};

void fill_packed_amplitude_sj_tensors_rec_half_sjC(
                                                          const B4Spins& s,
                                                          const AmplitudeInputSupportPlan_HalfSjC& support,
                                                          PackedAmplitudeSJTensors_HalfSjC& out);

} // namespace b4support
