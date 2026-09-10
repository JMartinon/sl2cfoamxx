// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#  include <omp.h>
#endif

#if defined(__APPLE__)
#  include <sys/types.h>
#  include <sys/sysctl.h>
#elif defined(__linux__)
#  include <sys/sysinfo.h>
#  include <unistd.h>
#endif

#include "amplitude/AmplitudeInputSupport_HalfSjC.hpp"
#include "amplitude/AmplitudePackedSJTensors_HalfSjC.hpp"
#include "amplitude/AmplitudePackedBoosters.hpp"
#include "amplitude/AmplitudePackedContractionGeneral.hpp"
#include "amplitude/AmplitudePackedContractionQLocal_ReflectionQOrbit.hpp"
#include "core/Failure.hpp"
#include "core/HostMemoryBudget.hpp"

namespace b4support {

// ============================================================================
// x-slice + q-slab streamed fallback, half-stored sjC
// ============================================================================
// This is the memory-emergency path used when even X=1 cannot store all sjC.
// It keeps the wing tensors for one x-slice, builds T1/T3 once, then streams
// sjC over q=(l24,l34,l35) slabs.  The q-slab support still uses the half-sjC
// storage map, including storage representatives outside the logical q-slab
// when the reflected partner owns the payload.
// ============================================================================

#ifndef B4PACK_XQSLAB_DEFAULT_RAM_FRACTION
#define B4PACK_XQSLAB_DEFAULT_RAM_FRACTION 0.80
#endif

#ifndef B4PACK_XQSLAB_DEFAULT_VARIABLE_OVERHEAD_FRACTION
#define B4PACK_XQSLAB_DEFAULT_VARIABLE_OVERHEAD_FRACTION 0.10
#endif

inline std::uint64_t xqslab_sat_add_u64(std::uint64_t a, std::uint64_t b) noexcept
{
    const std::uint64_t m = std::numeric_limits<std::uint64_t>::max();
    return (m - a < b) ? m : a + b;
}

inline std::uint64_t xqslab_sat_mul_u64(std::uint64_t a, std::uint64_t b) noexcept
{
    const std::uint64_t m = std::numeric_limits<std::uint64_t>::max();
    return (a != 0 && b > m / a) ? m : a * b;
}

inline std::uint64_t xqslab_bytes_for_values(std::uint64_t n) noexcept
{
    return xqslab_sat_mul_u64(n, static_cast<std::uint64_t>(sizeof(double)));
}

inline std::uint64_t xqslab_add_fractional_overhead(std::uint64_t bytes, double frac) noexcept
{
    if (!(frac > 0.0)) return bytes;
    const long double x = static_cast<long double>(bytes) *
                          (1.0L + static_cast<long double>(frac));
    const long double m = static_cast<long double>(std::numeric_limits<std::uint64_t>::max());
    return x >= m ? std::numeric_limits<std::uint64_t>::max()
                  : static_cast<std::uint64_t>(x);
}

inline std::uint64_t xqslab_hard_memory_capacity_bytes() noexcept
{
    const host_memory::Snapshot snapshot = host_memory::detect_snapshot();
    const std::uint64_t hard = host_memory::hard_capacity_bytes(snapshot);
    return hard != 0 ? hard : snapshot.physical_total_bytes;
}

inline std::uint64_t xqslab_default_safety_margin_bytes(std::uint64_t ram) noexcept
{
    const std::uint64_t GiB = 1024ull * 1024ull * 1024ull;
    if (ram == 0) return 2ull * GiB;
    std::uint64_t margin = ram / 20ull;
    margin = std::max<std::uint64_t>(margin, 1ull * GiB);
    margin = std::min<std::uint64_t>(margin, 32ull * GiB);
    return margin;
}

inline double xqslab_seconds_since(std::chrono::high_resolution_clock::time_point t0)
{
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clock::now() - t0).count();
}

inline void xqslab_print_bytes(std::ostream& os, std::uint64_t bytes)
{
    constexpr long double KiB = 1024.0L;
    constexpr long double MiB = 1024.0L * KiB;
    constexpr long double GiB = 1024.0L * MiB;
    constexpr long double TiB = 1024.0L * GiB;
    const long double b = static_cast<long double>(bytes);
    os << std::fixed << std::setprecision(4);
    if (b >= TiB) os << static_cast<double>(b / TiB) << " TiB";
    else if (b >= GiB) os << static_cast<double>(b / GiB) << " GiB";
    else if (b >= MiB) os << static_cast<double>(b / MiB) << " MiB";
    else if (b >= KiB) os << static_cast<double>(b / KiB) << " KiB";
    else os << bytes << " B";
}

struct PackedAmplitudeXQSlabOptions {
    // An explicit nonzero budget is authoritative. Automatic mode applies this
    // fraction to the physical/cgroup/Slurm hard capacity and also caps the
    // result by current system/cgroup allocation headroom.
    std::uint64_t memory_budget_bytes = 0;
    double ram_fraction = B4PACK_XQSLAB_DEFAULT_RAM_FRACTION;

    std::uint64_t output_bytes = sizeof(double);
    std::uint64_t extra_fixed_bytes = 0;
    std::uint64_t safety_margin_bytes = 0;
    std::uint64_t qlocal_thread_scratch_bytes = 0;

    std::size_t max_q_per_slab = std::numeric_limits<std::size_t>::max();
    double variable_overhead_fraction = B4PACK_XQSLAB_DEFAULT_VARIABLE_OVERHEAD_FRACTION;

    bool require_all_q_slabs_fit = true;
    bool print_plan = false;
    bool print_x_timing = false;
    bool print_q_timing = false;
    bool print_contraction_stats = false;
};

struct PackedAmplitudeQSlab {
    std::size_t q_begin = 0;
    std::size_t q_end = 0;
    std::uint64_t sjC_payload_bytes = 0;
    std::uint64_t estimated_variable_bytes = 0;
    std::uint64_t estimated_peak_bytes = 0;
    bool fits_budget = true;

    std::size_t size() const noexcept { return q_end > q_begin ? q_end - q_begin : 0u; }
};

struct PackedAmplitudeXQTimings {
    double planning = 0.0;
    double xsupport = 0.0;
    double wing_fill = 0.0;
    double T1T3 = 0.0;
    double qsupport = 0.0;
    double sjC_fill = 0.0;
    double qcontract = 0.0;
    double release = 0.0;
    double total = 0.0;

    double accounted() const noexcept {
        return planning + xsupport + wing_fill + T1T3 + qsupport + sjC_fill + qcontract + release;
    }
    double unaccounted() const noexcept { return total - accounted(); }
};

struct PackedAmplitudeXQResult {
    double amplitude = 0.0;
    std::size_t x_slices = 0;
    std::size_t q_slabs_total = 0;
    std::size_t unfitting_q_slabs = 0;
    std::uint64_t max_estimated_peak_bytes = 0;
    PackedAmplitudeXQTimings timings;
    PackedQLocalContractionStats stats;

    void print(std::ostream& os) const {
        os << "\nAmplitude x+q-slab run summary\n";
        os << "================================\n";
        os << std::scientific << std::setprecision(17)
           << "amplitude                  = " << amplitude << "\n";
        os << std::fixed << std::setprecision(6);
        os << "x slices                   = " << x_slices << "\n";
        os << "q slabs total              = " << q_slabs_total << "\n";
        os << "unfitting q slabs          = " << unfitting_q_slabs << "\n";
        os << "max estimated peak         = "; xqslab_print_bytes(os, max_estimated_peak_bytes); os << "\n";
        os << "time planning              = " << timings.planning << " s\n";
        os << "time x support             = " << timings.xsupport << " s\n";
        os << "time wing sj fill          = " << timings.wing_fill << " s\n";
        os << "time T1/T3 once per x      = " << timings.T1T3 << " s\n";
        os << "time q support             = " << timings.qsupport << " s\n";
        os << "time sjC q-slab fill       = " << timings.sjC_fill << " s\n";
        os << "time q-slab contraction    = " << timings.qcontract << " s\n";
        os << "time workspace release     = " << timings.release << " s\n";
        os << "time accounted             = " << timings.accounted() << " s\n";
        os << "time unaccounted           = " << timings.unaccounted() << " s\n";
        os << "time total xq path         = " << timings.total << " s\n";
    }
};

inline std::uint64_t xqslab_wing_payload_bytes(const AmplitudeInputSupportPlan_HalfSjC& support)
{
    return xqslab_sat_add_u64(
        xqslab_sat_add_u64(support.sjA.payload_bytes(), support.sjB.payload_bytes()),
        xqslab_sat_add_u64(support.sjD.payload_bytes(), support.sjE.payload_bytes())
    );
}

inline std::uint64_t xqslab_T1T3_payload_bytes(const RangeBlockSupport1D& T1,
                                               const RangeBlockSupport1D& T3)
{
    return xqslab_sat_add_u64(T1.payload_bytes(), T3.payload_bytes());
}

inline std::uint64_t xqslab_memory_budget(const PackedAmplitudeXQSlabOptions& opt)
{
    if (opt.memory_budget_bytes != 0) return opt.memory_budget_bytes;
    const host_memory::Snapshot snapshot = host_memory::detect_snapshot();
    if (host_memory::has_any_probe(snapshot)) {
        return host_memory::automatic_budget_bytes(snapshot, opt.ram_fraction);
    }
    return 8ull * 1024ull * 1024ull * 1024ull;
}

inline std::uint64_t xqslab_fixed_bytes(const AmplitudeInputSupportPlan_HalfSjC& static_support,
                                        const PackedAmplitudeBoosters& packed_B,
                                        const PackedAmplitudeXQSlabOptions& opt)
{
    const std::uint64_t ram = xqslab_hard_memory_capacity_bytes();
    const std::uint64_t safety = opt.safety_margin_bytes != 0
        ? opt.safety_margin_bytes
        : xqslab_default_safety_margin_bytes(ram);

    std::uint64_t fixed = 0;
    fixed = xqslab_sat_add_u64(fixed, packed_B.payload_bytes());
    fixed = xqslab_sat_add_u64(fixed, opt.output_bytes);
    fixed = xqslab_sat_add_u64(fixed, static_support.total_metadata_bytes());
    fixed = xqslab_sat_add_u64(fixed, safety);
    fixed = xqslab_sat_add_u64(fixed, opt.extra_fixed_bytes);
    fixed = xqslab_sat_add_u64(fixed, opt.qlocal_thread_scratch_bytes);
    return fixed;
}

std::vector<PackedAmplitudeQSlab> choose_q_slabs_for_xslice_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& xslice_full_support,
    const AmplitudeInputSupportPlan_HalfSjC& wing_support,
    const RangeBlockSupport1D& T1,
    const RangeBlockSupport1D& T3,
    const PackedAmplitudeXQSlabOptions& opt,
    std::uint64_t fixed_bytes);

// Fill only the half-stored sjC payload for q-slab support.  sjA/sjB/sjD/sjE
// are left untouched so a single wing workspace can be reused for all q-slabs.
void fill_packed_amplitude_sjC_only_half_sjC(
    const B4Spins& s,
    const AmplitudeInputSupportPlan_HalfSjC& support,
    PackedAmplitudeSJTensors_HalfSjC& out);

PackedAmplitudeXQResult contract_packed_amplitude_xqslab_streamed_generated_support(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const PackedAmplitudeBoosters& packed_B,
    const std::array<int, 10>& spins,
    const PackedAmplitudeXQSlabOptions& opt = {},
    WingExecutionOrder order = WingExecutionOrder::LeftThenRight,
    PackedContractionOptions contraction_options = {},
    std::ostream* log = nullptr);

} // namespace b4support
