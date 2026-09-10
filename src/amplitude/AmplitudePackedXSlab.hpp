// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
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
#include "core/Failure.hpp"
#include "core/HostMemoryBudget.hpp"

namespace b4support {

// ============================================================================
// Memory-bounded x-slab driver for scalar packed amplitude, half-stored sjC.
//
// First-stage design:
//   * keep boosters global, because they do not depend on x;
//   * make a slab-local support by slicing only sjA/sjB/sjC/sjD/sjE;
//   * keep B2/B3/B4/B5 supports and offsets identical to the global support;
//   * fill slab-local sj values;
//   * call the general half-sjC contraction dispatcher on the slab support.
//     Reflection-compatible data still uses the fast q-orbit backend; arbitrary
//     valid data uses the half-sjC-aware general q-local backend.
//
// This orchestration header keeps the existing full-x path as the numerical
// reference while providing the lower-memory execution mode.
// ============================================================================

#ifndef B4PACK_XSLAB_DEFAULT_RAM_FRACTION
#define B4PACK_XSLAB_DEFAULT_RAM_FRACTION 0.80
#endif

#ifndef B4PACK_XSLAB_DEFAULT_VARIABLE_OVERHEAD_FRACTION
#define B4PACK_XSLAB_DEFAULT_VARIABLE_OVERHEAD_FRACTION 0.10
#endif

#ifndef B4PACK_XSLAB_DEFAULT_MAX_X_PER_SLAB
// By default, do not impose an artificial x-tile cap.  The planner should be
// memory-driven: it will choose the full x-range when it fits, and split only
// when the estimated slab peak exceeds the selected memory budget.
#define B4PACK_XSLAB_DEFAULT_MAX_X_PER_SLAB (std::numeric_limits<std::size_t>::max())
#endif

#ifndef B4PACK_XSLAB_TIMING
#define B4PACK_XSLAB_TIMING 0
#endif

// ----------------------------------------------------------------------------
// Small arithmetic helpers.
// ----------------------------------------------------------------------------

inline std::uint64_t xslab_sat_add_u64(std::uint64_t a,
                                       std::uint64_t b) noexcept
{
    const std::uint64_t maxv = std::numeric_limits<std::uint64_t>::max();
    if (maxv - a < b) return maxv;
    return a + b;
}

inline std::uint64_t xslab_sat_mul_u64(std::uint64_t a,
                                       std::uint64_t b) noexcept
{
    const std::uint64_t maxv = std::numeric_limits<std::uint64_t>::max();
    if (a != 0 && b > maxv / a) return maxv;
    return a * b;
}

inline std::uint64_t xslab_bytes_for_values(std::uint64_t n_values) noexcept
{
    return xslab_sat_mul_u64(n_values, static_cast<std::uint64_t>(sizeof(double)));
}

inline std::uint64_t xslab_add_fractional_overhead(std::uint64_t bytes,
                                                   double frac) noexcept
{
    if (!(frac > 0.0)) return bytes;

    const long double scaled =
        static_cast<long double>(bytes) *
        (1.0L + static_cast<long double>(frac));

    const long double maxv =
        static_cast<long double>(std::numeric_limits<std::uint64_t>::max());

    if (scaled >= maxv) {
        return std::numeric_limits<std::uint64_t>::max();
    }

    return static_cast<std::uint64_t>(scaled);
}

inline std::uint64_t xslab_physical_ram_bytes() noexcept
{
    return host_memory::detect_snapshot().physical_total_bytes;
}

inline std::uint64_t xslab_hard_memory_capacity_bytes() noexcept
{
    const host_memory::Snapshot snapshot = host_memory::detect_snapshot();
    const std::uint64_t hard = host_memory::hard_capacity_bytes(snapshot);
    return hard != 0 ? hard : snapshot.physical_total_bytes;
}

inline std::uint64_t xslab_automatic_memory_budget_bytes(double ram_fraction) noexcept
{
    const host_memory::Snapshot snapshot = host_memory::detect_snapshot();
    if (host_memory::has_any_probe(snapshot)) {
        return host_memory::automatic_budget_bytes(snapshot, ram_fraction);
    }
    return 8ull * 1024ull * 1024ull * 1024ull;
}

inline std::uint64_t xslab_default_safety_margin_bytes(std::uint64_t ram) noexcept
{
    const std::uint64_t GiB = 1024ull * 1024ull * 1024ull;

    if (ram == 0) {
        return 2ull * GiB;
    }

    std::uint64_t margin = ram / 20ull; // 5%
    margin = std::max<std::uint64_t>(margin, 1ull * GiB);
    margin = std::min<std::uint64_t>(margin, 32ull * GiB);
    return margin;
}

inline double xslab_seconds_since(std::chrono::high_resolution_clock::time_point t0)
{
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clock::now() - t0).count();
}

inline void xslab_print_bytes(std::ostream& os, std::uint64_t bytes)
{
    constexpr long double KiB = 1024.0L;
    constexpr long double MiB = 1024.0L * KiB;
    constexpr long double GiB = 1024.0L * MiB;
    constexpr long double TiB = 1024.0L * GiB;

    const long double b = static_cast<long double>(bytes);
    os << std::fixed << std::setprecision(4);
    if (b >= TiB) {
        os << static_cast<double>(b / TiB) << " TiB";
    } else if (b >= GiB) {
        os << static_cast<double>(b / GiB) << " GiB";
    } else if (b >= MiB) {
        os << static_cast<double>(b / MiB) << " MiB";
    } else if (b >= KiB) {
        os << static_cast<double>(b / KiB) << " KiB";
    } else {
        os << bytes << " B";
    }
}

// ----------------------------------------------------------------------------
// Options, costs, plan.
// ----------------------------------------------------------------------------

struct PackedAmplitudeXSlabOptions {
    // If nonzero, this is the total memory budget used by the planner.
    // On SLURM/cgroup jobs this should be set explicitly from the job limit;
    // reported physical RAM can exceed the allocation headroom available to
    // the current process or cgroup.
    std::uint64_t memory_budget_bytes = 0;

    // Used only when memory_budget_bytes == 0. The fractional hard-cap result
    // is additionally limited by current system/cgroup allocation headroom.
    double ram_fraction = B4PACK_XSLAB_DEFAULT_RAM_FRACTION;

    std::uint64_t output_bytes = sizeof(double);
    std::uint64_t extra_fixed_bytes = 0;
    std::uint64_t safety_margin_bytes = 0;

    // Current first-stage prototype keeps global input_support alive, so this
    // should normally be true.  Once we build only slab-local supports, set it
    // false or use a static-plan variant.
    bool include_existing_input_support_metadata = true;

    // True while slab support owns copied metadata. A future view-only support
    // representation could set this to false.
    bool include_slab_sj_metadata = true;

    // The current q-local contraction builds and reuses T1/T3 per x-slice.
    bool reuse_T1T3_per_x = true;

    // Dense upper bound for T1/T3 per x.  Conservative and cheap.
    bool dense_T1T3_estimate = true;

    // If zero, estimate a small per-thread q-local scratch reserve.
    std::uint64_t qlocal_thread_scratch_bytes = 0;

    std::size_t min_x_per_slab = 1;
    std::size_t max_x_per_slab = B4PACK_XSLAB_DEFAULT_MAX_X_PER_SLAB;

    double variable_overhead_fraction =
        B4PACK_XSLAB_DEFAULT_VARIABLE_OVERHEAD_FRACTION;

    bool skip_unfitting_slabs = false;
};

struct PackedAmplitudeXCost {
    std::size_t ix = 0;
    std::uint64_t sj_payload_bytes = 0;
    std::uint64_t sj_metadata_bytes = 0;
    std::uint64_t T1T3_bytes = 0;
};

struct PackedAmplitudeXSlab {
    std::size_t ix_begin = 0;
    std::size_t ix_end = 0; // exclusive, global ix range

    std::uint64_t sj_payload_bytes = 0;
    std::uint64_t sj_metadata_bytes = 0;
    std::uint64_t T1T3_bytes = 0;
    std::uint64_t estimated_variable_bytes = 0;
    std::uint64_t estimated_peak_bytes = 0;

    bool fits_budget = true;

    std::size_t size() const noexcept {
        return ix_end > ix_begin ? ix_end - ix_begin : 0u;
    }
};

struct PackedAmplitudeXSlabPlan {
    std::uint64_t physical_ram = 0;
    std::uint64_t memory_budget_bytes = 0;

    std::uint64_t fixed_bytes = 0;
    std::uint64_t available_for_variable_bytes = 0;

    std::uint64_t booster_payload_bytes = 0;
    std::uint64_t output_bytes = 0;
    std::uint64_t existing_support_metadata_bytes = 0;
    std::uint64_t qlocal_thread_scratch_bytes = 0;
    std::uint64_t safety_margin_bytes = 0;
    std::uint64_t extra_fixed_bytes = 0;

    std::vector<PackedAmplitudeXCost> xcosts;
    std::vector<PackedAmplitudeXSlab> slabs;

    bool all_slabs_fit() const noexcept {
        for (const PackedAmplitudeXSlab& s : slabs) {
            if (!s.fits_budget) return false;
        }
        return true;
    }

    std::uint64_t max_peak_bytes() const noexcept {
        std::uint64_t m = 0;
        for (const PackedAmplitudeXSlab& s : slabs) {
            m = std::max(m, s.estimated_peak_bytes);
        }
        return m;
    }

    std::size_t num_unfitting_slabs() const noexcept {
        std::size_t n = 0;
        for (const PackedAmplitudeXSlab& s : slabs) {
            n += s.fits_budget ? 0u : 1u;
        }
        return n;
    }

    const PackedAmplitudeXCost* largest_x_slice() const noexcept {
        if (xcosts.empty()) return nullptr;
        return &*std::max_element(
            xcosts.begin(), xcosts.end(),
            [](const PackedAmplitudeXCost& a, const PackedAmplitudeXCost& b) {
                return a.sj_payload_bytes < b.sj_payload_bytes;
            });
    }

    void print(std::ostream& os) const {
        constexpr long double GiB = 1024.0L * 1024.0L * 1024.0L;

        os << "\nAmplitude x-slab memory plan\n";
        os << "============================\n";

        os << "physical RAM                  = ";
        xslab_print_bytes(os, physical_ram); os << "\n";
        os << "memory budget                 = ";
        xslab_print_bytes(os, memory_budget_bytes); os << "\n";
        os << "fixed bytes                   = ";
        xslab_print_bytes(os, fixed_bytes); os << "\n";
        os << "  booster payload             = ";
        xslab_print_bytes(os, booster_payload_bytes); os << "\n";
        os << "  output                      = ";
        xslab_print_bytes(os, output_bytes); os << "\n";
        os << "  existing support metadata   = ";
        xslab_print_bytes(os, existing_support_metadata_bytes); os << "\n";
        os << "  qlocal thread scratch       = ";
        xslab_print_bytes(os, qlocal_thread_scratch_bytes); os << "\n";
        os << "  safety margin               = ";
        xslab_print_bytes(os, safety_margin_bytes); os << "\n";
        os << "  extra fixed                 = ";
        xslab_print_bytes(os, extra_fixed_bytes); os << "\n";
        os << "available for slab variable   = ";
        xslab_print_bytes(os, available_for_variable_bytes); os << "\n";

        os << "number of x slices            = " << xcosts.size() << "\n";
        os << "number of slabs               = " << slabs.size() << "\n";
        os << "unfitting slabs               = " << num_unfitting_slabs() << "\n";
        os << "max estimated peak            = ";
        xslab_print_bytes(os, max_peak_bytes()); os << "\n";

        if (const PackedAmplitudeXCost* c = largest_x_slice()) {
            os << "largest single-x sj payload   = ix " << c->ix << ", ";
            xslab_print_bytes(os, c->sj_payload_bytes); os << "\n";
        }

        os << "\nSlabs:\n";
        os << "  #     ix range        X     sj payload     sj meta"
              "       T1/T3       variable      peak      fits\n";

        for (std::size_t i = 0; i < slabs.size(); ++i) {
            const PackedAmplitudeXSlab& s = slabs[i];

            os << std::setw(3) << i << "   ["
               << std::setw(4) << s.ix_begin << ","
               << std::setw(4) << s.ix_end << ")"
               << "   " << std::setw(4) << s.size()
               << "   " << std::setw(10)
               << static_cast<double>(s.sj_payload_bytes) / static_cast<double>(GiB)
               << "   " << std::setw(10)
               << static_cast<double>(s.sj_metadata_bytes) / static_cast<double>(GiB)
               << "   " << std::setw(10)
               << static_cast<double>(s.T1T3_bytes) / static_cast<double>(GiB)
               << "   " << std::setw(10)
               << static_cast<double>(s.estimated_variable_bytes) / static_cast<double>(GiB)
               << "   " << std::setw(10)
               << static_cast<double>(s.estimated_peak_bytes) / static_cast<double>(GiB)
               << "   " << (s.fits_budget ? "yes" : "NO")
               << "\n";
        }
    }
};

inline bool is_single_full_xslab_plan(const PackedAmplitudeXSlabPlan& plan,
                                      std::size_t X) noexcept
{
    return plan.slabs.size() == 1 &&
           plan.slabs[0].ix_begin == 0 &&
           plan.slabs[0].ix_end == X &&
           plan.slabs[0].fits_budget;
}

inline std::uint64_t xslab_default_qlocal_thread_scratch_bytes(
    const AmplitudeInputSupportPlan_HalfSjC& support);

std::uint64_t xslab_dense_T1T3_bytes_per_x(
    const AmplitudeInputSupportPlan_HalfSjC& support);


// ----------------------------------------------------------------------------
// Stats helpers.
// ----------------------------------------------------------------------------

void add_packed_qlocal_stats(PackedQLocalContractionStats& dst,
                                    const PackedQLocalContractionStats& src) noexcept;

// ----------------------------------------------------------------------------
// Cost estimation from an existing global half-sjC support.
// ----------------------------------------------------------------------------

inline std::uint64_t xslab_default_qlocal_thread_scratch_bytes(
    const AmplitudeInputSupportPlan_HalfSjC& support)
{
    const std::uint64_t MiB = 1024ull * 1024ull;

    const std::uint64_t row_values =
        static_cast<std::uint64_t>(
            std::max(support.channels.k3.size(), support.channels.k4.size())) *
        static_cast<std::uint64_t>(support.l25.size());

#ifdef _OPENMP
    const std::uint64_t nthreads =
        static_cast<std::uint64_t>(std::max(1, omp_get_max_threads()));
#else
    const std::uint64_t nthreads = 1;
#endif

    const std::uint64_t per_thread =
        xslab_sat_mul_u64(8ull, xslab_bytes_for_values(row_values));

    return xslab_sat_add_u64(64ull * MiB,
                             xslab_sat_mul_u64(nthreads, per_thread));
}

std::uint64_t xslab_dense_T1T3_bytes_per_x(
    const AmplitudeInputSupportPlan_HalfSjC& support);

PackedAmplitudeXCost estimate_amplitude_x_cost_from_global_support(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    std::size_t ix,
    bool include_slab_sj_metadata,
    bool dense_T1T3_estimate);

// ----------------------------------------------------------------------------
// Cost planning from static/global support without global sjX metadata.
// ----------------------------------------------------------------------------

inline PackedAmplitudeXCost estimate_amplitude_x_cost_from_generated_support(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const B4Spins& spins,
    std::size_t ix,
    bool include_slab_sj_metadata,
    bool dense_T1T3_estimate)
{
    const std::size_t X = static_support.channels.xvals.size();
    if (ix >= X) {
        throw std::out_of_range("estimate_amplitude_x_cost_from_generated_support: ix out of range");
    }

    const AmplitudeInputSupportPlan_HalfSjC one_x_support =
        build_amplitude_sj_xslab_support_half_sjC(static_support, spins, ix, ix + 1);

    PackedAmplitudeXCost c = estimate_amplitude_x_cost_from_global_support(
        one_x_support,
        0,
        include_slab_sj_metadata,
        dense_T1T3_estimate);
    c.ix = ix;
    return c;
}

PackedAmplitudeXSlabPlan choose_packed_amplitude_x_slabs_from_memory_generated_support(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const B4Spins& spins,
    const PackedAmplitudeXSlabOptions& opt = {});


// ----------------------------------------------------------------------------
// Slab execution.
// ----------------------------------------------------------------------------

struct PackedAmplitudeXSlabRunOptions {
    PackedAmplitudeXSlabOptions planner;
    WingExecutionOrder order = WingExecutionOrder::LeftThenRight;
    PackedContractionOptions contraction_options = {};

    bool require_all_slabs_fit = true;

    // If the planner returns exactly one slab [0,X), use the ordinary full-x
    // path instead of slicing/copying support metadata and then doing the same
    // numerical work with extra ceremony. This is the fast path for small cases
    // where all sjX tensors fit comfortably.
    bool enable_full_x_bypass = true;

    // Generated-support mode only: before doing the detailed per-x generated
    // planner, use a cheap dense upper-bound estimate to decide whether the
    // ordinary full-support/full-x path is safely within budget. This avoids
    // generating per-x planning supports before building the same full support
    // for small cases.
    //
    // This bypass is intentionally disabled when the user imposes a real
    // max_x_per_slab cap smaller than X, since that usually means they are
    // debugging or forcing a slab-decomposition test.
    bool enable_support_full_x_bypass = true;

    // Require the conservative dense full-sj estimate to fit within this
    // fraction of the selected memory budget before taking the support-level
    // bypass. The normal safety margin is already part of the budget model;
    // this extra fraction leaves headroom for allocations outside the model.
    double support_full_x_bypass_budget_fraction = 0.80;

    bool print_plan = false;
    bool print_slab_timing = false;
    bool print_slab_sj_size = false;
    bool print_contraction_stats = false;
};

struct PackedAmplitudeXSlabTimings {
    // Time spent choosing the x-slab execution plan.
    double planning = 0.0;

    // Time spent slicing/copying support metadata for genuine slab runs.
    double support_slice = 0.0;

    // Time spent filling sj tensors and contracting them.
    double sj_fill = 0.0;
    double contraction = 0.0;

    // Time spent releasing slab/full-sj workspaces inside this wrapper.
    // This matters for benchmarking: the old hand-written full-x main often
    // measures before packed_sj is destroyed, while this wrapper owns its
    // temporary workspace and must destroy it before returning, so destruction
    // belongs in the end-to-end timing.
    double workspace_release = 0.0;

    // Total time from function entry to just before return, including workspace
    // release when measured explicitly.
    double total = 0.0;

    double accounted() const noexcept {
        return planning + support_slice + sj_fill
             + contraction + workspace_release;
    }

    double unaccounted() const noexcept {
        return total - accounted();
    }
};

struct PackedAmplitudeXSlabResult {
    double amplitude = 0.0;
    bool used_full_x_bypass = false;
    PackedAmplitudeXSlabPlan plan;
    PackedAmplitudeXSlabTimings timings;
    PackedQLocalContractionStats stats;

    void print(std::ostream& os) const {
        os << "\nAmplitude x-slab run summary\n";
        os << "============================\n";
        os << "amplitude                  = " << std::scientific
           << std::setprecision(17) << amplitude << "\n";
        os << std::fixed << std::setprecision(6);
        os << "used full-x bypass          = " << (used_full_x_bypass ? "yes" : "no") << "\n";
        os << "time detailed planning      = " << timings.planning << " s\n";
        os << "time support slicing        = " << timings.support_slice << " s\n";
        os << "time sj slab filling        = " << timings.sj_fill << " s\n";
        os << "time slab contraction       = " << timings.contraction << " s\n";
        os << "time workspace release      = " << timings.workspace_release << " s\n";
        os << "time accounted              = " << timings.accounted() << " s\n";
        os << "time unaccounted            = " << timings.unaccounted() << " s\n";
        os << "time total x-slab path      = " << timings.total << " s\n";
    }
};

// ----------------------------------------------------------------------------
// Support-level full-x bypass for generated-support mode.
// ----------------------------------------------------------------------------

inline std::uint64_t xslab_sat_mul_many_u64(std::initializer_list<std::uint64_t> vals) noexcept
{
    std::uint64_t r = 1;
    for (const std::uint64_t v : vals) {
        r = xslab_sat_mul_u64(r, v);
    }
    return r;
}

inline std::uint64_t xslab_memory_budget_from_options(
    const PackedAmplitudeXSlabOptions& opt,
    std::uint64_t physical_ram)
{
    if (opt.memory_budget_bytes != 0) {
        return opt.memory_budget_bytes;
    }
    (void)physical_ram;
    return xslab_automatic_memory_budget_bytes(opt.ram_fraction);
}

std::uint64_t xslab_full_dense_sj_payload_upper_bound_bytes(
    const AmplitudeInputSupportPlan_HalfSjC& support);

inline std::uint64_t xslab_full_sj_metadata_upper_bound_bytes(
    const AmplitudeInputSupportPlan_HalfSjC& support)
{
    const std::uint64_t X   = static_cast<std::uint64_t>(support.channels.xvals.size());
    const std::uint64_t K3d = static_cast<std::uint64_t>(support.channels.k3.size());
    const std::uint64_t K4d = static_cast<std::uint64_t>(support.channels.k4.size());

    const std::uint64_t L23 = static_cast<std::uint64_t>(support.l23.size());
    const std::uint64_t L24 = static_cast<std::uint64_t>(support.l24.size());
    const std::uint64_t L25 = static_cast<std::uint64_t>(support.l25.size());
    const std::uint64_t L34 = static_cast<std::uint64_t>(support.l34.size());
    const std::uint64_t L35 = static_cast<std::uint64_t>(support.l35.size());
    const std::uint64_t L45 = static_cast<std::uint64_t>(support.l45.size());

    std::uint64_t bytes = 0;

    bytes = xslab_sat_add_u64(bytes,
        xslab_sat_mul_u64(xslab_sat_mul_many_u64({X, L25}), sizeof(Range1D)));
    bytes = xslab_sat_add_u64(bytes,
        xslab_sat_mul_u64(xslab_sat_mul_many_u64({X, K4d, L35, L45}), sizeof(Range1D)));
    bytes = xslab_sat_add_u64(bytes,
        xslab_sat_mul_u64(xslab_sat_mul_many_u64({X, L24, L34, L35}), sizeof(Range2D)));
    bytes = xslab_sat_add_u64(bytes,
        xslab_sat_mul_u64(xslab_sat_mul_many_u64({X, L24, L34, L35}), sizeof(AmplitudeSjCHalfBlockMap)));
    bytes = xslab_sat_add_u64(bytes,
        xslab_sat_mul_u64(xslab_sat_mul_many_u64({X, K3d, L23, L24}), sizeof(Range1D)));
    bytes = xslab_sat_add_u64(bytes,
        xslab_sat_mul_u64(xslab_sat_mul_many_u64({X, L25}), sizeof(Range1D)));

    return bytes;
}

struct PackedAmplitudeSupportFullXBypassEstimate {
    std::uint64_t physical_ram = 0;
    std::uint64_t memory_budget_bytes = 0;
    std::uint64_t allowed_peak_bytes = 0;
    std::uint64_t fixed_bytes = 0;
    std::uint64_t dense_sj_payload_upper_bound_bytes = 0;
    std::uint64_t sj_metadata_upper_bound_bytes = 0;
    std::uint64_t T1T3_bytes = 0;
    std::uint64_t estimated_variable_bytes = 0;
    std::uint64_t estimated_peak_bytes = 0;
    bool fits = false;
};

PackedAmplitudeSupportFullXBypassEstimate estimate_support_full_x_bypass_peak(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const PackedAmplitudeXSlabOptions& opt,
    double budget_fraction);

inline int xslab_infer_Dl_from_static_support(
    const AmplitudeInputSupportPlan_HalfSjC& support)
{
    const std::size_t n = std::max({support.l23.size(), support.l24.size(), support.l25.size(),
                                    support.l34.size(), support.l35.size(), support.l45.size()});
    if (n == 0) {
        throw std::invalid_argument("xslab_infer_Dl_from_static_support: empty l axes");
    }
    return static_cast<int>(n - 1u);
}

// Production x-slab entry point: global/static support only, with sjX support
// generated per slab. This avoids ever storing global sjA/sjB/sjC/sjD/sjE
// metadata while retaining the full-x bypass for cases that fit comfortably.
PackedAmplitudeXSlabResult contract_packed_amplitude_xslab_streamed_generated_support(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const PackedAmplitudeBoosters& packed_B,
    const std::array<int, 10>& spins,
    const std::array<int, 5>& fixed_intertwiners,
    const PackedAmplitudeXSlabRunOptions& run_opt = {},
    std::ostream* log = &std::cout);

} // namespace b4support
