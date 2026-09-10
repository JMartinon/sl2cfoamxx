// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "core/ProgressEtaEstimator.hpp"

// CPU full-tensor contraction for half-stored sjC support.  The retained path
// covers reflection-compact symmetric outputs and rectangular general-spin
// outputs, with q-orbit reduction, adaptive final BLAS/tiled accumulation, and
// bounded thread-local panel reduction.

#include "full_tensor/FullTensorInputSupport_HalfSjC.hpp"
#include "full_tensor/FullTensorQReflectionOrbitPlan.hpp"
#include "full_tensor/FullTensorPackedSJTensors_HalfSjC.hpp"
#include "full_tensor/FullTensorPackedBoosters_HalfSjC.hpp"
#include "core/Failure.hpp"
#include "core/InformationalOutput.hpp"
#include "core/OutputPresentation.hpp"
#include "gpu/FullTensorGpuNativeApi.hpp"
#include "sl2cfoamxx/progress.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include "core/OpenMp.hpp"
#endif


// L25-stacked full-tensor contraction with phase/work profiling:
//
//   A += sum_l25 T4(l25)^T * sjC * T2(l25)
//
// batched over l25 with detailed phase counters.

#ifndef B4FULL_RESTRICT
#  if defined(__GNUC__) || defined(__clang__)
#    define B4FULL_RESTRICT __restrict__
#  elif defined(_MSC_VER)
#    define B4FULL_RESTRICT __restrict
#  else
#    define B4FULL_RESTRICT
#  endif
#endif

#ifndef B4FULL_FORCE_INLINE
#  if defined(__GNUC__) || defined(__clang__)
#    define B4FULL_FORCE_INLINE inline __attribute__((always_inline))
#  elif defined(_MSC_VER)
#    define B4FULL_FORCE_INLINE __forceinline
#  else
#    define B4FULL_FORCE_INLINE inline
#  endif
#endif

namespace b4support {

// ============================================================================
// Small profiling utilities
// ============================================================================

struct FullTensorScalarPhaseProfile {
    const char* name = "";
    double seconds = 0.0;
    std::uint64_t calls = 0;
    long double approx_flops = 0.0L;

    void add(const FullTensorScalarPhaseProfile& other) noexcept {
        seconds      += other.seconds;
        calls        += other.calls;
        approx_flops += other.approx_flops;
    }

    double gflops_per_s() const noexcept {
        return seconds > 0.0
             ? static_cast<double>(approx_flops) / seconds / 1.0e9
             : 0.0;
    }
};

struct FullTensorScalarSupportProfile {
    std::uint64_t q_blocks_seen = 0;
    std::uint64_t q_blocks_nonempty_sjC = 0;
    std::uint64_t q_blocks_with_T2 = 0;
    std::uint64_t q_blocks_with_T4 = 0;
    std::uint64_t q_blocks_accumulated = 0;

    std::uint64_t sum_nk3 = 0;
    std::uint64_t sum_nk4 = 0;
    std::uint64_t max_nk3 = 0;
    std::uint64_t max_nk4 = 0;

    std::uint64_t T1_seed_values_total = 0;
    std::uint64_t T1_seed_values_nnz = 0;
    std::uint64_t T3_seed_values_total = 0;
    std::uint64_t T3_seed_values_nnz = 0;

    std::uint64_t T2_values_total = 0;
    std::uint64_t T2_values_nnz = 0;
    std::uint64_t T4_values_total = 0;
    std::uint64_t T4_values_nnz = 0;
    std::uint64_t CT2_values_total = 0;
    std::uint64_t CT2_values_nnz = 0;

    std::uint64_t T1_triple_terms = 0;
    std::uint64_t T3_triple_terms = 0;
    std::uint64_t T2_dot_terms = 0;
    std::uint64_t T4_dot_terms = 0;
    std::uint64_t CT2_rank1_updates = 0;
    std::uint64_t final_dot_terms = 0;

    void add(const FullTensorScalarSupportProfile& other) noexcept;
};

struct FullTensorPackedContractionL25ScalarProfile {
    FullTensorScalarPhaseProfile zero_output     {"zero_output"};
    FullTensorScalarPhaseProfile T1_seed         {"T1_seed"};
    FullTensorScalarPhaseProfile T3_seed         {"T3_seed"};
    FullTensorScalarPhaseProfile T2_zero         {"T2_zero"};
    FullTensorScalarPhaseProfile T2_stack        {"T2_stack"};
    FullTensorScalarPhaseProfile T4_zero         {"T4_zero"};
    FullTensorScalarPhaseProfile T4_stack        {"T4_stack"};
    FullTensorScalarPhaseProfile CT2_zero        {"CT2_zero"};
    FullTensorScalarPhaseProfile sjC_access      {"sjC_access"};
    FullTensorScalarPhaseProfile CT2_scalar      {"CT2_scalar"};
    FullTensorScalarPhaseProfile final_scalar    {"final_scalar"};
    FullTensorScalarPhaseProfile total_wall      {"total_wall"};

    FullTensorScalarSupportProfile support;
    bool used_cpu_q_reflection_orbits = false;
    std::uint64_t cpu_q_logical_members_covered = 0;
    std::uint64_t cpu_q_orbits_executed = 0;
    std::uint64_t cpu_q_paired_orbits_executed = 0;
    std::uint64_t cpu_q_self_orbits_executed = 0;
    std::uint64_t cpu_q_unpaired_blocks_executed = 0;
    std::uint64_t cpu_vendor_blas_final_calls = 0;
    std::uint64_t cpu_tiled_final_calls = 0;
    void add(const FullTensorPackedContractionL25ScalarProfile& other) noexcept;

    static double pct(double part, double whole) noexcept {
        return whole > 0.0 ? 100.0 * part / whole : 0.0;
    }

    static double density(std::uint64_t nnz, std::uint64_t total) noexcept {
        return total > 0 ? static_cast<double>(nnz) / static_cast<double>(total) : 0.0;
    }

    void print(std::ostream& os) const;
};

struct FullTensorPackedContractionL25ScalarOptions {
    bool use_openmp = true;
    bool zero_output = true;

    // Reflection-compatible compact CPU outputs can execute one q member per
    // reflection orbit.  Disable only for diagnostics/A-B comparisons; an
    // incompatible support plan automatically retains the logical-q route.
    bool use_cpu_q_reflection_orbits = true;

    // Profiling these densities adds a little overhead, but this is exactly the
    // information needed before designing the sparse backend.
    bool count_seed_nnz = true;
    bool count_stack_nnz = true;
    bool count_CT2_nnz = true;

    // Large dense support rectangles are better delegated to the vendor BLAS
    // than to the small-support microkernel.  Tiny matrices stay on the latter
    // to avoid call overhead.  The linked CBLAS implementation may be
    // Accelerate, OpenBLAS, BLIS, or MKL on production systems.
    bool use_vendor_blas_CT2 = true;
    std::uint64_t vendor_blas_CT2_min_flops = 131072;
    // The rectangular final update is exactly A(M,N) += L(P,M)^T * R(P,N)
    // in column-major storage.  Vendor DGEMM wins decisively for sufficiently
    // large panels, while the local 4x4 kernel avoids BLAS dispatch/packing
    // overhead for small or very skinny shapes.  These conservative defaults
    // are intentionally overridable for provider-specific tuning and A/B tests.
    bool use_vendor_blas_final_rectangular = true;
    std::size_t vendor_blas_final_rectangular_min_dimension = 8;
    std::uint64_t vendor_blas_final_rectangular_min_flops = 32768;

};

class FullTensorScalarScopedTimer {
public:
    explicit FullTensorScalarScopedTimer(FullTensorScalarPhaseProfile& p)
        : phase_(p), start_(clock::now()) {}

    ~FullTensorScalarScopedTimer() {
        const auto end = clock::now();
        phase_.seconds += std::chrono::duration<double>(end - start_).count();
        phase_.calls += 1;
    }

private:
    using clock = std::chrono::high_resolution_clock;
    FullTensorScalarPhaseProfile& phase_;
    clock::time_point start_;
};

// ============================================================================
// Local helpers, deliberately named with a scalar_profile prefix to avoid
// colliding with the earlier full-tensor contraction headers.
// ============================================================================

B4FULL_FORCE_INLINE void full_scalar_profile_zero(double* B4FULL_RESTRICT p,
                                                  std::size_t n) noexcept
{
#ifdef _OPENMP
#pragma omp simd
#endif
    for (std::size_t i = 0; i < n; ++i) {
        p[i] = 0.0;
    }
}

B4FULL_FORCE_INLINE Range1D full_scalar_profile_intersect_3_ranges(
    const Range1D& a,
    const Range1D& b,
    const Range1D& c) noexcept
{
    const std::uint32_t begin = std::max({a.begin, b.begin, c.begin});
    const std::uint32_t end   = std::min({a.end,   b.end,   c.end});
    return begin < end ? Range1D{begin, end, 0ull} : Range1D{0u, 0u, 0ull};
}

B4FULL_FORCE_INLINE double full_scalar_profile_triple_dot_intersection(
    const Range1D& ra,
    const double* B4FULL_RESTRICT a_base,
    const Range1D& rb,
    const double* B4FULL_RESTRICT b_base,
    const Range1D& rc,
    const double* B4FULL_RESTRICT c_base,
    std::uint64_t* terms) noexcept
{
    const Range1D r = full_scalar_profile_intersect_3_ranges(ra, rb, rc);
    if (r.empty()) {
        return 0.0;
    }

    const std::size_t len = static_cast<std::size_t>(r.end - r.begin);
    if (terms != nullptr) {
        *terms += static_cast<std::uint64_t>(len);
    }

    const double* B4FULL_RESTRICT a = a_base + (r.begin - ra.begin);
    const double* B4FULL_RESTRICT b = b_base + (r.begin - rb.begin);
    const double* B4FULL_RESTRICT c = c_base + (r.begin - rc.begin);

    double s = 0.0;
#ifdef _OPENMP
#pragma omp simd reduction(+:s)
#endif
    for (std::size_t k = 0; k < len; ++k) {
        s += a[k] * b[k] * c[k];
    }
    return s;
}

// ============================================================================
// Workspace
// ============================================================================

struct FullTensorPackedContractionL25ScalarWorkspace {
    std::vector<double> T1_cache;
    std::vector<double> T3_cache;
    std::vector<double> T2_stack;
    std::vector<double> T4_stack;
    std::vector<double> C_block; // Materialized logical sjC block when stored transposed.

    // Scratch for the reassociated triangular backend.
    // T2_B4_panel stores B4(ii4,l45) as [il45*I4 + ii4].
    // T4_B3_panel stores B3(ii3,l23) as [il23*I3 + ii3].
    // T4_T3_col stores one temporary T3 column over i2.
    // D4_stack stores D4(k4,l25,m) = sum_k3 C(k3,k4) T4(k3,l25,m)
    // in row-major layout [(l25*nk4+jb)*M + m].
    std::vector<double> T2_B4_panel;
    std::vector<double> T4_B3_panel;
    std::vector<double> T4_T3_col;
    std::vector<double> D4_stack;

    void ensure_seed_caches(const FullTensorInputSupportPlan& support);

    void ensure_stacks(std::size_t nk3,
                       std::size_t nk4,
                       std::size_t L25,
                       std::size_t M,
                       std::size_t N) {
        T2_stack.resize(nk4 * L25 * N);
        T4_stack.resize(nk3 * L25 * M);
        C_block.resize(nk3 * nk4);
    }
};

const double* full_tensor_sjC_logical_block_base_l25_scalar(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const FullTensorSjCHalfBlockMap& map,
    std::vector<double>& C_block);


void build_full_T1_seed_panel_scalar_profiled(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    std::size_t ii1,
    std::size_t ix,
    std::size_t il25,
    std::size_t il35,
    double* B4FULL_RESTRICT seed,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt);

void build_full_T3_seed_panel_scalar_profiled(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    std::size_t ii1,
    std::size_t ix,
    std::size_t il24,
    std::size_t il25,
    double* B4FULL_RESTRICT seed,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt);

void build_full_seed_caches_l25_scalar_profiled(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    std::size_t ii1,
    std::size_t ix,
    FullTensorPackedContractionL25ScalarWorkspace& ws,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt);

// A whole (i1,x) slice is identically zero if either seed wing has no
// structural support for every l25. Detecting that case costs O(L25), avoids
// building zero seed/T2/T4 panels, and keeps CPU progress from assigning late
// work to the inexpensive edge slices which the scheduler deliberately leaves
// until the tail.
inline bool full_tensor_cpu_i1_x_has_seed_wings(
    const FullTensorInputSupportPlan& support,
    std::size_t ii1,
    std::size_t ix) noexcept
{
    const std::size_t X = support.channels.xvals.size();
    const std::size_t L25 = support.l25.size();
    bool has_sjA = false;
    bool has_sjE = false;
    for (std::size_t il25 = 0; il25 < L25; ++il25) {
        const std::size_t block = idx3(ii1, ix, il25, X, L25);
        has_sjA = has_sjA || !support.sjA.blocks[block].empty();
        has_sjE = has_sjE || !support.sjE.blocks[block].empty();
        if (has_sjA && has_sjE) return true;
    }
    return false;
}

// ============================================================================
// L25-stacked T2/T4 builders, restricted to sjC support rectangle
// ============================================================================

void scatter_full_output_by_i1_to_canonical_scalar_profiled(
    const FullTensorInputSupportPlan& support,
    const double* B4FULL_RESTRICT A_by_i1,
    double* B4FULL_RESTRICT A_canonical);

} // namespace b4support


// Reflection-fast path for the half-stored-sjC, scalar-only L25-stacked full
// tensor contraction.
//
// This header exploits the component symmetry, valid for the current gauge-fixed
// contraction whenever the boundary data and boundary intertwiner channels are invariant under
//
//     1 fixed, 2 <-> 5, 3 <-> 4,
//
// i.e.
//
//     A(i1,i2,i3,i4,i5) = A(i1,i5,i4,i3,i2).
//
// The contraction computes only one representative of each output orbit in the
// fast by-i1 layout, then helper functions can fill/scatter the reflected half.
//
// Include this instead of calling the ordinary scalar-profile contraction when
// the reflection fast path is enabled.


namespace b4support {

// ============================================================================
// Reflection compatibility helpers
// ============================================================================

// SpinArray convention:
//   (j12,j13,j14,j15,j23,j24,j25,j34,j35,j45)
// Reflection:
//   1 fixed, 2 <-> 5, 3 <-> 4.
// Required equalities:
//   j12 = j15, j13 = j14, j23 = j45, j24 = j35.
// j25 and j34 are fixed by the reflection.
inline bool full_tensor_reflection_15432_preserves_spins(
    const std::array<int,10>& J) noexcept
{
    return J[0] == J[3]   // j12 == j15
        && J[1] == J[2]   // j13 == j14
        && J[4] == J[9]   // j23 == j45
        && J[5] == J[8];  // j24 == j35
}

inline int full_tensor_reflection_find_value_index(
    const std::vector<int>& values,
    int value) noexcept
{
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] == value) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

struct FullTensorReflectionPair {
    std::size_t m = 0;          // representative row in fast panel
    std::size_t n = 0;          // representative col in fast panel
    std::size_t partner_m = 0;  // reflected row
    std::size_t partner_n = 0;  // reflected col
};

struct FullTensorReflectionOutputMap {
    std::size_t I2 = 0;
    std::size_t I3 = 0;
    std::size_t I4 = 0;
    std::size_t I5 = 0;
    std::size_t M = 0;
    std::size_t N = 0;

    // One pair per representative output entry.  For fixed points,
    // (m,n)==(partner_m,partner_n).
    std::vector<FullTensorReflectionPair> representatives;

    // rep_index[flat] gives the representative flat index in fast panel layout
    // for any full output entry flat = m + M*n.
    std::vector<std::size_t> rep_index;

    bool compatible = false;

    std::size_t full_entries() const noexcept { return M * N; }
    std::size_t independent_entries() const noexcept { return representatives.size(); }

    double independent_fraction() const noexcept {
        return full_entries() > 0
            ? static_cast<double>(independent_entries()) / static_cast<double>(full_entries())
            : 0.0;
    }
};

FullTensorReflectionOutputMap build_full_tensor_reflection_output_map(
    const FullTensorInputSupportPlan& support);

} // namespace b4support


// Compact-output reflection-fast path for the half-stored-sjC, scalar-only
// L25-stacked full tensor contraction.
//
// This header keeps the trusted reflection-fast mathematics:
//
//     A(i1,i2,i3,i4,i5) = A(i1,i5,i4,i3,i2)
//
// but stores only one representative per output orbit.  The branch tensors T2
// and T4 are still built normally.  This is deliberate: branch-level reflection
// was tested and is not valid in the current packed basis/convention.  The safe
// win here is output storage, cache size, and avoiding the reflected-half fill
// inside the fast by-i1 buffer.
//
// Compact layout:
//
//     A_rep_by_i1[rep + nrep * ii1]
//
// where rep enumerates the representatives of
//
//     (i2,i3,i4,i5) ~ (i5,i4,i3,i2).
//
// Include this header after the half-sjC packed tensor/booster support headers.


namespace b4support {

// ============================================================================
// Compact reflection output map
// ============================================================================

struct FullTensorReflectionCompactOutputMap {
    FullTensorReflectionOutputMap full_map;

    // For every full panel entry flat = m + M*n, this gives the compact
    // representative ordinal in [0,nrep).  This is useful for diagnostics and
    // for optional full-layout expansion.
    std::vector<std::size_t> compact_rep_of_full;

    bool compatible = false;

    std::size_t I2() const noexcept { return full_map.I2; }
    std::size_t I3() const noexcept { return full_map.I3; }
    std::size_t I4() const noexcept { return full_map.I4; }
    std::size_t I5() const noexcept { return full_map.I5; }
    std::size_t M()  const noexcept { return full_map.M;  }
    std::size_t N()  const noexcept { return full_map.N;  }

    std::size_t full_entries() const noexcept {
        return full_map.full_entries();
    }

    std::size_t independent_entries() const noexcept {
        return full_map.independent_entries();
    }

    double independent_fraction() const noexcept {
        return full_map.independent_fraction();
    }

    std::size_t compact_values_total(std::size_t I1) const noexcept {
        return I1 * independent_entries();
    }
};

FullTensorReflectionCompactOutputMap
build_full_tensor_reflection_compact_output_map(
    const FullTensorInputSupportPlan& support);

inline std::size_t full_tensor_reflection_compact_values_count(
    const FullTensorInputSupportPlan& support)
{
    const FullTensorReflectionCompactOutputMap cmap =
        build_full_tensor_reflection_compact_output_map(support);

    if (!cmap.compatible) {
        throw b4utils::InternalFailure(
            "Cannot compute compact reflected output size: channels are not reflection-compatible.");
    }

    return support.channels.i1.size() * cmap.independent_entries();
}

void scatter_full_output_reflection_compact_by_i1_to_canonical(
    const FullTensorInputSupportPlan& support,
    const double* B4FULL_RESTRICT A_tri_by_i1,
    double* B4FULL_RESTRICT A_canonical);

} // namespace b4support


// Per-i1 private-panel parallelization for the compact-reflection full tensor
// contraction.
//
// This header keeps the mathematically trusted compact-reflection contraction,
// but changes the OpenMP work decomposition:
//
//   old: parallel over i1 panels only
//   new: for each i1, keep x serial, parallelize q=(l35,l24,l34)
//        with one private compact i1-panel per thread, then reduce.
//
// This gives many more OpenMP tasks on large-core nodes while preserving a
// serial x loop.  It avoids atomics in the hot final accumulation.


namespace b4support {

// ============================================================================
// Weighted progress reporting
// ============================================================================

struct FullTensorReflectionCompactWeightedProgressOptions {
    bool enabled = false;

    // Terminal rendering and structured callbacks are independent sinks. The
    // internal default preserves the behavior of direct reporter tests;
    // public API call sites disable it for callback-only reporting.
    bool render_output = true;
    sl2cfoamxx::ProgressCallback callback{};
    sl2cfoamxx::ProgressOperation operation =
        sl2cfoamxx::ProgressOperation::full_tensor;
    sl2cfoamxx::ProgressPhase phase =
        sl2cfoamxx::ProgressPhase::contraction;
    sl2cfoamxx::Backend backend_kind = sl2cfoamxx::Backend::CPU;

    // Add to the atomic progress counter only after this many estimated work
    // units have accumulated locally.  Larger values reduce atomic overhead.
    std::uint64_t update_granularity = 1u << 20;

    double refresh_seconds = 0.25;
    std::string label = "Full tensor compact reflection";
    std::string unit_label = "work";
    std::ostream* stream = &std::cerr;
    sl2cfoamxx::detail::OutputPresentation presentation{};
    std::string operation_label = "full tensor";
    std::string backend;
    std::string phase_label;
    std::size_t bar_width = 22;

    // A silent public call may use progress as a transient live indicator.
    // On an actual interactive terminal, successful completion then erases
    // the single live row instead of leaving a 100% record behind. Durable
    // non-TTY/custom-stream records and failed/stopped displays are retained.
    bool clear_on_success = false;

    // A cumulative-rate prediction is converted into a completion deadline.
    // The deadline is updated only when new work completes, and smoothed before
    // the live display counts down from it. This avoids the sawtooth produced
    // by polling bursty scheduled work.
    double eta_smoothing = 0.20;
    double eta_min_elapsed_seconds = 30.0;
    double eta_min_fraction = 0.01;

    // Guard against accidentally formatting effectively-infinite ETAs as giant
    // hour counts. If the ETA is larger than this, leave it unavailable.
    double eta_max_display_seconds = 30.0 * 24.0 * 3600.0;
};

bool full_tensor_progress_advance_milestones(
    std::uint64_t completed,
    std::uint64_t total,
    std::size_t& next_milestone) noexcept;

class FullTensorReflectionCompactWeightedProgressBar {
public:
    FullTensorReflectionCompactWeightedProgressBar(
        std::uint64_t total_work,
        FullTensorReflectionCompactWeightedProgressOptions options);

    ~FullTensorReflectionCompactWeightedProgressBar() { stop(false); }

    FullTensorReflectionCompactWeightedProgressBar(
        const FullTensorReflectionCompactWeightedProgressBar&) = delete;
    FullTensorReflectionCompactWeightedProgressBar& operator=(
        const FullTensorReflectionCompactWeightedProgressBar&) = delete;

    bool enabled() const noexcept { return options_.enabled; }

    std::uint64_t completed_work() const noexcept {
        return done_.load(std::memory_order_relaxed);
    }

    std::uint64_t update_granularity() const noexcept {
        return options_.update_granularity;
    }

    void start() noexcept;

    void add(std::uint64_t n) noexcept {
        if (n != 0) {
            done_.fetch_add(n, std::memory_order_relaxed);
        }
    }

    // A backend may display one work model while using a better-correlated
    // completion model for ETA. Zero leaves ETA coupled to the visible
    // completed/total counters.
    void set_eta_total(std::uint64_t total_work) noexcept {
        std::lock_guard render_lock(render_mutex_);
        eta_total_work_.store(total_work, std::memory_order_relaxed);
    }

    void add_eta(std::uint64_t n) noexcept {
        if (n != 0 &&
            eta_total_work_.load(std::memory_order_relaxed) != 0) {
            eta_done_.fetch_add(n, std::memory_order_relaxed);
        }
    }

    // Render an immediate snapshot when a caller has an external update
    // cadence. The ordinary backends rely on the monitor thread; this hook is
    // also useful for deterministic diagnostics tests without wall-clock
    // sleeps. Rendering remains serialized with the monitor.
    void refresh() noexcept {
        std::lock_guard lifecycle_lock(lifecycle_mutex_);
        if (started_) {
            print_update();
        }
    }

    // Backends whose work size is only known after planning may publish the
    // exact total immediately before start(). A retry may publish it again;
    // serialization with the renderer keeps the monitor free of data races.
    void set_total(std::uint64_t total_work) noexcept {
        std::lock_guard render_lock(render_mutex_);
        total_work_ = std::max<std::uint64_t>(1, total_work);
    }

    // Begin a fresh accounting epoch after an internal backend retry. The
    // failed attempt has finished before this is called, so no producer can
    // still be advancing the counter. Serializing with rendering keeps the
    // counter and ETA state internally consistent with the monitor thread.
    void reset() noexcept;

    void stop(bool completed = true) noexcept;

private:
    using clock = std::chrono::steady_clock;

    struct Snapshot {
        std::uint64_t done = 0;
        double elapsed = 0.0;
        double eta = 0.0;
        bool eta_available = false;
    };

    std::uint64_t total_work_;
    FullTensorReflectionCompactWeightedProgressOptions options_;
    std::atomic<std::uint64_t> done_;
    std::atomic<std::uint64_t> eta_done_;
    std::atomic<std::uint64_t> eta_total_work_{0};
    bool started_;
    bool interactive_;
    bool owns_interactive_line_;
    clock::time_point start_time_;
    clock::time_point rate_start_time_;
    sl2cfoamxx::detail::ProgressEtaEstimator eta_estimator_;
    std::size_t next_plain_milestone_ = 1;
    std::uint64_t last_plain_reported_done_ = 0;
    std::uint64_t last_callback_reported_done_ = 0;
    std::uint32_t attempt_ = 1;
    std::size_t last_interactive_width_ = 0;
    std::thread monitor_;
    std::mutex lifecycle_mutex_;
    std::mutex render_mutex_;
    std::mutex wake_mutex_;
    std::condition_variable wake_;
    bool stop_requested_ = false;

    void release_interactive_line() noexcept
    {
        if (!owns_interactive_line_) return;
        sl2cfoamxx::detail::release_interactive_progress();
        owns_interactive_line_ = false;
    }

    void monitor_loop() noexcept;

    [[nodiscard]] Snapshot snapshot(bool force_done) noexcept;

    [[nodiscard]] std::string_view public_unit_label() const noexcept {
        if (options_.unit_label == "sched-work") return "estimated work";
        if (options_.unit_label == "i1-pass") return "passes";
        if (options_.unit_label == "boundary intertwiner-tile") return "tiles";
        return options_.unit_label;
    }

    [[nodiscard]] std::string_view phase_label() const noexcept {
        if (!options_.phase_label.empty()) return options_.phase_label;
        if (options_.label.find("sj-stream") != std::string::npos) {
            return "6j stream";
        }
        return "contract";
    }

    [[nodiscard]] static std::size_t display_width(
        const std::string& text) noexcept
    {
        std::size_t width = 0;
        for (const unsigned char character : text) {
            if ((character & 0xc0u) != 0x80u) ++width;
        }
        return width;
    }

    [[nodiscard]] bool uses_transient_live_line() const noexcept
    {
        return options_.clear_on_success &&
               options_.presentation.terminal_control;
    }

    [[nodiscard]] std::size_t transient_terminal_columns() const noexcept
    {
        // TIOCGWINSZ may fail even for a usable TTY. Match the minimum width
        // accepted by detect_output_presentation() instead of treating an
        // unknown width as unlimited: a wrapped transient row cannot be
        // erased safely with carriage returns alone.
        constexpr std::size_t conservative_unknown_width = 48;
        return options_.presentation.columns != 0
            ? options_.presentation.columns
            : conservative_unknown_width;
    }

    [[nodiscard]] std::string transient_prefix() const;

    [[nodiscard]] std::string render_interactive_line_text(
        const Snapshot& current,
        bool completed_successfully,
        bool stopped) const;

    [[nodiscard]] sl2cfoamxx::detail::ProgressRenderState render_state(
        const Snapshot& current,
        bool completed_successfully,
        bool stopped) const noexcept;

    void emit_callback(
        const Snapshot& current,
        sl2cfoamxx::ProgressStatus status) noexcept;

    void print_initial() noexcept;

    void print_update() noexcept;

    void print_final(bool completed_successfully) noexcept;

    void print_interactive_line(
        const Snapshot& current,
        bool completed_successfully,
        bool stopped,
        bool terminate_line = false);

    void clear_interactive_line()
    {
        std::lock_guard output_lock(
            sl2cfoamxx::detail::informational_output_mutex());
        (*options_.stream) << '\r'
                           << std::string(last_interactive_width_, ' ')
                           << '\r' << std::flush;
        last_interactive_width_ = 0;
    }

    void print_plain_line(
        const Snapshot& current,
        bool completed_successfully,
        bool stopped);
};

// ============================================================================
// Cheap weighted-progress estimator
// ============================================================================

inline std::size_t full_tensor_reflection_compact_q_weight_index(
    std::size_t ix,
    std::size_t il35,
    std::size_t il24,
    std::size_t il34,
    std::size_t L35,
    std::size_t L24,
    std::size_t L34)
{
    return il34 + L34 * (il24 + L24 * (il35 + L35 * ix));
}

} // namespace b4support


// Blocked triangular compact-reflection backend for the half-stored-sjC,
// L25-stacked full tensor contraction.
//
// This backend keeps the safe output reflection only:
//
//     A(i1,i2,i3,i4,i5) = A(i1,i5,i4,i3,i2)
//
// and rewrites each fixed-i1 output panel as a symmetric matrix
//
//     B[m,s] = A[m, reverse(s)]
//
// where
//
//     m = (i2,i3),   n = (i4,i5),   s = reverse(n) = (i5,i4)
//
// The final update therefore becomes an upper-triangular product
//
//     upper(B) += upper(L_batch^T * Rperm_batch)
//
// with
//
//     L_batch[p,m]     = T4_q(k3,l25,i2,i3)
//     Rperm_batch[p,s] = CT2_q(k3,l25,i4,i5) with (i4,i5)=reverse(s)
//
// Public output layout remains the existing compact representative layout:
//
//     A_rep_by_i1[rep + nrep * ii1]
//
// so existing scatter helpers still work.  The triangular matrix is only an
// internal execution layout.


namespace b4support {

// ============================================================================
// Cache detection and options
// ============================================================================

struct FullTensorReflectionCompactTriangularCacheInfo {
    std::size_t l1d_bytes = 64u * 1024u;
    std::size_t l2_bytes  = 4u * 1024u * 1024u;
    std::size_t l3_bytes  = 0;

    // On Apple Silicon, these are often more useful than the generic hw.l2cachesize.
    std::size_t apple_perf_l2_bytes = 0;
    std::size_t apple_eff_l2_bytes  = 0;

    bool detected_l1d = false;
    bool detected_l2  = false;
    bool detected_l3  = false;
    bool detected_apple_perf_l2 = false;
    bool detected_apple_eff_l2  = false;
};

#if defined(__APPLE__)
bool full_tensor_triangular_read_sysctl_size(
    const char* name,
    std::size_t& value) noexcept;
#endif

#if defined(__linux__)
bool full_tensor_triangular_read_first_line(
    const std::string& path,
    std::string& out);

std::size_t full_tensor_triangular_parse_cache_size(std::string s);
#endif

FullTensorReflectionCompactTriangularCacheInfo
full_tensor_detect_cache_info_triangular_batched();

enum class FullTensorTriangularFinalTileShape {
    Tile4x4,
    Tile4x8,
    Tile8x4
};

inline const char* full_tensor_triangular_final_tile_shape_name(
    FullTensorTriangularFinalTileShape shape) noexcept
{
    switch (shape) {
        case FullTensorTriangularFinalTileShape::Tile4x4: return "4x4";
        case FullTensorTriangularFinalTileShape::Tile4x8: return "4x8";
        case FullTensorTriangularFinalTileShape::Tile8x4: return "8x4";
    }
    return "4x4";
}

struct FullTensorReflectionCompactTriangularBatchedOptions {
    // p_batch == 0 means auto-select using detected cache sizes and matrix shape.
    // For Apple M-series, 256 or 512 is usually the first manual test range.
    std::size_t p_batch = 512;

    // Outer parallelism is over i1 panels.  0 means use min(I1, omp_get_max_threads()).
    // On an M3 Pro, try 5 first to mostly target P-cores, then compare with 11.
    std::size_t max_outer_workers = 0;

    // Auto p_batch uses the largest detected L2-like cache.  This matters more
    // than a fictional generic L3 on Apple Silicon.
    double l2_fraction = 0.35;
    std::size_t min_worker_batch_bytes = 2u * 1024u * 1024u;
    std::size_t max_worker_batch_bytes = 8u * 1024u * 1024u;
    std::size_t min_p_batch = 128;
    std::size_t max_p_batch = 2048;
    std::size_t p_batch_round_multiple = 64;

    bool print_cache_info = true;

    // Final compact triangular update tile.  All three variants use padded
    // row/column storage so the hot p-loop has no boundary conditionals.
    // Benchmark this: the best shape depends on compiler/register pressure.
    FullTensorTriangularFinalTileShape final_tile_shape =
        FullTensorTriangularFinalTileShape::Tile4x4;
};

inline std::size_t full_tensor_triangular_upper_size(std::size_t M) noexcept {
    return M * (M + 1u) / 2u;
}

inline std::size_t full_tensor_triangular_upper_index(
    std::size_t i,
    std::size_t j) noexcept
{
    if (i > j) {
        std::swap(i, j);
    }
    return j * (j + 1u) / 2u + i;
}

inline std::size_t full_tensor_triangular_round_down_multiple(
    std::size_t x,
    std::size_t m) noexcept
{
    if (m == 0) return x;
    return (x / m) * m;
}

inline std::size_t full_tensor_triangular_round_up_multiple(
    std::size_t x,
    std::size_t m) noexcept
{
    if (m == 0) return x;
    const std::size_t r = x % m;
    return r == 0 ? x : x + (m - r);
}

std::size_t choose_full_tensor_triangular_p_batch(
    std::size_t M,
    std::size_t active_outer_workers,
    const FullTensorReflectionCompactTriangularCacheInfo& cache,
    const FullTensorReflectionCompactTriangularBatchedOptions& topt);

// ============================================================================
// Reverse-column map: original n=(i4,i5) -> symmetric column s=(i5,i4)
// ============================================================================

struct FullTensorTriangularReverseColumnMap {
    std::size_t M = 0;
    std::size_t N = 0;
    std::vector<std::size_t> n_to_s;
    std::vector<std::size_t> s_to_n;
    bool compatible = false;
};

FullTensorTriangularReverseColumnMap
build_full_tensor_triangular_reverse_column_map(
    const FullTensorInputSupportPlan& support);

// ============================================================================
// Batched triangular workspace and final kernel
// ============================================================================

inline std::uint64_t full_tensor_cpu_saturating_mul_u64(
    std::uint64_t a,
    std::uint64_t b) noexcept;

struct FullTensorTriangularBatchedLRWorkspace {
    std::size_t p_cap = 0;
    std::size_t p_used = 0;
    std::size_t M = 0;
    std::size_t M_pad = 0; // Padded dimension for branch-free final tiles.
    FullTensorTriangularFinalTileShape final_tile_shape =
        FullTensorTriangularFinalTileShape::Tile4x4;
    FullTensorReflectionCompactWeightedProgressBar* progress = nullptr;
    std::uint64_t skipped_rows_pending = 0;
    std::uint64_t skipped_rows_total = 0;
    std::vector<double> L;
    std::vector<double> Rperm;

    void reset(std::size_t p_capacity, std::size_t M_);

    void clear() noexcept {
        p_used = 0;
    }

    void begin_task() noexcept {
        p_used = 0;
        skipped_rows_pending = 0;
        skipped_rows_total = 0;
    }

    void skip_planned_rows(std::uint64_t rows) noexcept;
};

void full_tensor_triangular_batched_update_upper_tile4x4(
                                                                const double* B4FULL_RESTRICT L_batch,
                                                                const double* B4FULL_RESTRICT Rperm_batch,
                                                                double* B4FULL_RESTRICT B_upper,
                                                                std::size_t P,
                                                                std::size_t ldP,
                                                                std::size_t M,
                                                                std::size_t M_pad); // Accept the padded dimension

void full_tensor_triangular_batched_update_upper_tile4x8(
    const double* B4FULL_RESTRICT L_batch,
    const double* B4FULL_RESTRICT Rperm_batch,
    double* B4FULL_RESTRICT B_upper,
    std::size_t P,
    std::size_t ldP,
    std::size_t M,
    std::size_t M_pad);

void full_tensor_triangular_batched_update_upper_tile8x4(
    const double* B4FULL_RESTRICT L_batch,
    const double* B4FULL_RESTRICT Rperm_batch,
    double* B4FULL_RESTRICT B_upper,
    std::size_t P,
    std::size_t ldP,
    std::size_t M,
    std::size_t M_pad);

void flush_full_tensor_triangular_batched_lr(
                                                    FullTensorTriangularBatchedLRWorkspace& batch,
                                                    double* B4FULL_RESTRICT B_upper,
                                                    FullTensorPackedContractionL25ScalarProfile& profile);

// ============================================================================
// Reassociated triangular-batched backend
// ============================================================================
// This backend uses the identity
//
//     T4^T * C * T2 = (C^T * T4)^T * T2
//
// per q-block.  The goal is to let T2 and T4 be built in their natural
// row-major panel layouts, then form D4 = C^T*T4 and feed (D4,T2) into the
// same padded triangular final accumulator.  The old CT2 phase is replaced by
// D4 construction; for compatibility with the existing profile object, the
// printed CT2_zero/CT2_scalar phase names are changed to D4_zero/D4_stack in
// the public reassociated driver.

bool build_full_T2_l25_stack_reassociated_profiled(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorBoosters& B,
    const FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    const Range2D& rC,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* B4FULL_RESTRICT T2_row,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt);

bool build_full_T4_l25_stack_reassociated_profiled(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorBoosters& B,
    const FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    const Range2D& rC,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* B4FULL_RESTRICT T4_row,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt);

void build_full_D4_l25_stack_reassociated_profiled(
    const Range2D& rC,
    const double* B4FULL_RESTRICT C_base,
    const double* B4FULL_RESTRICT T4_row,
    double* B4FULL_RESTRICT D4_row,
    std::size_t L25,
    std::size_t M,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt);

void append_q_block_to_full_tensor_triangular_batched_lr_reassociated(
    const double* B4FULL_RESTRICT D4_row,
    const double* B4FULL_RESTRICT T2_row,
    std::size_t K,
    std::size_t M,
    const FullTensorTriangularReverseColumnMap& reverse_map,
    FullTensorTriangularBatchedLRWorkspace& batch,
    double* B4FULL_RESTRICT B_upper,
    FullTensorPackedContractionL25ScalarProfile& profile);

// A canonical reflection orbit contributes Z(m,s) + Z(s,m) to the public
// formula-indexed upper triangle, where
//
//     Z = D4^T * T2_reflected_columns.
//
// Packing (D4,T2) followed by (T2,D4) lets the existing triangular final
// kernel form both terms without materializing a dense MxM q result.  A
// self-reflected q contributes only once, so both packed orientations receive
// the same factor 1/2.  Unpaired logical fallback blocks retain the historical
// single orientation exactly.
void append_q_orbit_to_full_tensor_triangular_batched_lr_reassociated(
    const double* B4FULL_RESTRICT D4_row,
    const double* B4FULL_RESTRICT T2_row,
    std::size_t K,
    std::size_t M,
    const FullTensorTriangularReverseColumnMap& reverse_map,
    FullTensorQOrbitKind orbit_kind,
    FullTensorTriangularBatchedLRWorkspace& batch,
    double* B4FULL_RESTRICT B_upper,
    FullTensorPackedContractionL25ScalarProfile& profile);

void process_full_tensor_one_q_block_triangular_batched_reassociated(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    const FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    FullTensorTriangularBatchedLRWorkspace& batch,
    const FullTensorTriangularReverseColumnMap& reverse_map,
    std::size_t ix,
    std::size_t il35,
    std::size_t il24,
    std::size_t il34,
    double* B4FULL_RESTRICT B_upper,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt,
    FullTensorQOrbitKind orbit_kind = FullTensorQOrbitKind::Unpaired);

inline const FullTensorQOrbitMember&
full_tensor_select_left_associated_cpu_q_member(
    const FullTensorQOrbitDescriptor& orbit) noexcept
{
    // Both orbit members produce reflected-transpose output matrices.  The
    // left-associated CPU tree has K=L25*nk4, so selecting the member with the
    // narrower logical k4 support minimizes both aggregate rows and final-dot
    // work.  The half-sjC accessor transparently normalizes storage when this
    // selects the reflected member.
    if (orbit.has_distinct_partner() &&
        orbit.reflected.nk4() < orbit.representative.nk4()) {
        return orbit.reflected;
    }
    return orbit.representative;
}

void process_full_tensor_one_q_orbit_triangular_batched_reassociated(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    const FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    FullTensorTriangularBatchedLRWorkspace& batch,
    const FullTensorTriangularReverseColumnMap& reverse_map,
    const FullTensorQOrbitDescriptor& orbit,
    bool canonical_orbit,
    double* B4FULL_RESTRICT B_upper,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt);

std::uint64_t full_tensor_reflection_compact_triangular_batched_reassociated_q_weight_units(
    const FullTensorInputSupportPlan& support,
    const FullTensorReflectionCompactOutputMap& compact_map,
    std::size_t ix,
    std::size_t il35,
    std::size_t il24,
    std::size_t il34);

std::uint64_t build_full_tensor_reflection_compact_triangular_batched_reassociated_q_weights(
    const FullTensorInputSupportPlan& support,
    const FullTensorReflectionCompactOutputMap& compact_map,
    std::vector<std::uint64_t>& q_weights);

std::uint64_t
build_full_tensor_cpu_q_orbit_triangular_batched_reassociated_q_weights(
    const FullTensorInputSupportPlan& support,
    const FullTensorReflectionCompactOutputMap& compact_map,
    const FullTensorQReflectionOrbitPlan& orbit_plan,
    std::vector<std::uint64_t>& q_weights);

// ============================================================================
// General-spin reassociated rectangular-batched backend
// ============================================================================
// This backend uses the same algebraic reassociation as the compact triangular
// backend,
//
//     T4^T * C * T2 = (C^T * T4)^T * T2,
//
// but it does NOT use reflection symmetry.  It therefore accumulates the full
// rectangular output panel A[(i2,i3),(i4,i5)] and is valid for generic spin
// configurations.  For reflection-compatible data the compact triangular
// backend remains faster and smaller; this one is the correctness/general case.

struct FullTensorReassociatedRectangularBatchedLRWorkspace {
    std::size_t p_cap = 0;
    std::size_t p_used = 0;
    std::size_t M = 0;
    std::size_t N = 0;
    std::size_t M_pad = 0;
    std::size_t N_pad = 0;
    std::vector<double> L; // column-major: L[p + p_cap*m]
    std::vector<double> R; // column-major: R[p + p_cap*n]
    bool use_vendor_blas = false;
    std::size_t vendor_blas_min_dimension = 0;
    std::uint64_t vendor_blas_min_flops = 0;
    FullTensorReflectionCompactWeightedProgressBar* progress = nullptr;
    std::uint64_t skipped_rows_pending = 0;
    std::uint64_t skipped_rows_total = 0;

    void configure_cpu_final_accumulator(
        const FullTensorPackedContractionL25ScalarOptions& opt) noexcept
    {
        use_vendor_blas = opt.use_vendor_blas_final_rectangular;
        vendor_blas_min_dimension =
            opt.vendor_blas_final_rectangular_min_dimension;
        vendor_blas_min_flops =
            opt.vendor_blas_final_rectangular_min_flops;
    }

    void reset(std::size_t p_capacity, std::size_t M_, std::size_t N_);

    void clear() noexcept {
        p_used = 0;
    }

    void begin_task() noexcept {
        p_used = 0;
        skipped_rows_pending = 0;
        skipped_rows_total = 0;
    }

    void skip_planned_rows(std::uint64_t rows) noexcept;
};

bool full_tensor_reassociated_rectangular_use_vendor_blas(
    const FullTensorReassociatedRectangularBatchedLRWorkspace& batch) noexcept;

void full_tensor_reassociated_rectangular_batched_update_tile4x4(
    const double* B4FULL_RESTRICT L_batch,
    const double* B4FULL_RESTRICT R_batch,
    double* B4FULL_RESTRICT A_panel,
    std::size_t P,
    std::size_t ldP,
    std::size_t M,
    std::size_t N,
    std::size_t M_pad,
    std::size_t N_pad);

void flush_full_tensor_reassociated_rectangular_batched_lr(
    FullTensorReassociatedRectangularBatchedLRWorkspace& batch,
    double* B4FULL_RESTRICT A_panel,
    FullTensorPackedContractionL25ScalarProfile& profile);

void append_q_block_to_full_tensor_reassociated_rectangular_batched_lr(
    const double* B4FULL_RESTRICT D4_row,
    const double* B4FULL_RESTRICT T2_row,
    std::size_t K,
    std::size_t M,
    std::size_t N,
    FullTensorReassociatedRectangularBatchedLRWorkspace& batch,
    double* B4FULL_RESTRICT A_panel,
    FullTensorPackedContractionL25ScalarProfile& profile);

void process_full_tensor_one_q_block_reassociated_rectangular_batched(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    const FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    FullTensorReassociatedRectangularBatchedLRWorkspace& batch,
    std::size_t ix,
    std::size_t il35,
    std::size_t il24,
    std::size_t il34,
    double* B4FULL_RESTRICT A_panel,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt);

std::uint64_t full_tensor_reassociated_rectangular_batched_q_weight_units(
    const FullTensorInputSupportPlan& support,
    std::size_t ix,
    std::size_t il35,
    std::size_t il24,
    std::size_t il34);

std::uint64_t build_full_tensor_reassociated_rectangular_batched_q_weights(
    const FullTensorInputSupportPlan& support,
    std::vector<std::uint64_t>& q_weights);

std::size_t choose_full_tensor_reassociated_rectangular_p_batch(
    std::size_t M,
    std::size_t N,
    std::size_t active_outer_workers,
    const FullTensorReflectionCompactTriangularCacheInfo& cache,
    const FullTensorReflectionCompactTriangularBatchedOptions& topt);

// Convenience size helpers for the general-spin rectangular reassociated backend.
// The returned layout is by-i1 panels:
//   A_by_i1[(i2 + I2*i3) + M*((i4 + I4*i5) + N*i1)]
// with M=I2*I3 and N=I4*I5.  Use
// scatter_full_output_by_i1_to_canonical_scalar_profiled(...) to convert this
// to canonical Tensor order (i1,i2,i3,i4,i5).
inline std::size_t full_tensor_reassociated_rectangular_panel_values_count(
    const FullTensorInputSupportPlan& support) noexcept
{
    return support.channels.i2.size() * support.channels.i3.size() *
           support.channels.i4.size() * support.channels.i5.size();
}

inline std::size_t full_tensor_reassociated_rectangular_by_i1_values_count(
    const FullTensorInputSupportPlan& support) noexcept
{
    return support.channels.i1.size() *
           full_tensor_reassociated_rectangular_panel_values_count(support);
}


// ============================================================================
// Auto-input reassociated backend selector
// ============================================================================
// This small public layer chooses the fastest valid output layout from the
// input channels:
//   * reflection-compatible -> compact triangular reassociated backend
//   * generic/asymmetric    -> rectangular reassociated backend
//
// The algebraic reassociation is always used.  Only the final output layout
// changes.  In other words, symmetry is exploited when it is actually present,
// and ignored when it is not.  What a concept.

enum class FullTensorReassociatedAutoOutputLayout {
    ReflectionCompactTriangular,
    RectangularFull
};

struct FullTensorReassociatedAutoOutputPlan {
    FullTensorReassociatedAutoOutputLayout layout =
        FullTensorReassociatedAutoOutputLayout::RectangularFull;

    // Number of doubles required for the by-i1 output buffer used by the
    // selected backend.
    std::size_t values_count = 0;

    // Useful diagnostics for logging/debugging.
    std::size_t I1 = 0;
    std::size_t M = 0;
    std::size_t N = 0;
    std::size_t compact_representatives = 0;
    bool reflection_compatible = false;
    bool reverse_column_compatible = false;

    bool uses_reflection_compact() const noexcept {
        return layout == FullTensorReassociatedAutoOutputLayout::ReflectionCompactTriangular;
    }

};

inline const char* full_tensor_reassociated_auto_layout_name(
    FullTensorReassociatedAutoOutputLayout layout) noexcept
{
    switch (layout) {
        case FullTensorReassociatedAutoOutputLayout::ReflectionCompactTriangular:
            return "reflection-compact triangular";
        case FullTensorReassociatedAutoOutputLayout::RectangularFull:
            return "rectangular full";
    }
    return "unknown";
}

FullTensorReassociatedAutoOutputPlan
make_full_tensor_reassociated_auto_output_plan(
    const FullTensorInputSupportPlan& support);

// Conservative admission estimate for the per-worker L/R batches allocated by
// the scheduled CPU contraction.  The scheduler may later reduce its worker
// count when it produces fewer tasks than this ceiling, but it never allocates
// a larger batch or starts more workers.  Keeping the estimate beside the
// allocator's dimension rules prevents a public P_batch override from silently
// bypassing the host-memory planner.
struct FullTensorCpuBatchedWorkspaceAdmission {
    std::size_t worker_ceiling = 1;
    std::size_t p_batch = 1;
    std::size_t M_pad = 0;
    std::size_t N_pad = 0;
    std::uint64_t per_worker_bytes = 0;
    std::uint64_t total_bytes = 0;
};

inline std::uint64_t full_tensor_cpu_saturating_mul_u64(
    std::uint64_t a,
    std::uint64_t b) noexcept
{
    const std::uint64_t maxv = std::numeric_limits<std::uint64_t>::max();
    if (a != 0 && b > maxv / a) {
        return maxv;
    }
    return a * b;
}

inline std::uint64_t full_tensor_cpu_saturating_add_u64(
    std::uint64_t a,
    std::uint64_t b) noexcept
{
    const std::uint64_t maxv = std::numeric_limits<std::uint64_t>::max();
    return maxv - a < b ? maxv : a + b;
}

inline std::uint64_t full_tensor_cpu_progress_residual_rows(
    std::uint64_t planned_rows,
    std::uint64_t executed_rows,
    std::uint64_t skipped_rows) noexcept
{
    const std::uint64_t accounted_rows =
        full_tensor_cpu_saturating_add_u64(executed_rows, skipped_rows);
    return planned_rows > accounted_rows
        ? planned_rows - accounted_rows
        : 0;
}

FullTensorCpuBatchedWorkspaceAdmission
make_full_tensor_cpu_batched_workspace_admission(
    const FullTensorInputSupportPlan& support,
    const FullTensorPackedContractionL25ScalarOptions& opt,
    const FullTensorReflectionCompactTriangularBatchedOptions& topt,
    const FullTensorReflectionCompactTriangularCacheInfo& cache);

// Scatter the auto-selected by-i1 output layout into canonical full tensor order.
// This recomputes the layout plan from the support, so callers must pass the
// same support object that was used for the contraction.
inline void scatter_full_tensor_reassociated_auto_by_i1_to_canonical(
    const FullTensorInputSupportPlan& support,
    const double* B4FULL_RESTRICT A_auto_by_i1,
    double* B4FULL_RESTRICT A_canonical)
{
    const FullTensorReassociatedAutoOutputPlan plan =
        make_full_tensor_reassociated_auto_output_plan(support);

    if (plan.uses_reflection_compact()) {
        scatter_full_output_reflection_compact_by_i1_to_canonical(
            support, A_auto_by_i1, A_canonical);
    } else {
        scatter_full_output_by_i1_to_canonical_scalar_profiled(
            support, A_auto_by_i1, A_canonical);
    }
}

// Scatter using an already-computed plan.  The layout is trusted from the plan;
// this avoids recomputing the compatibility map if the caller saved the return
// value from the contraction.
inline void scatter_full_tensor_reassociated_auto_by_i1_to_canonical(
    const FullTensorInputSupportPlan& support,
    const FullTensorReassociatedAutoOutputPlan& plan,
    const double* B4FULL_RESTRICT A_auto_by_i1,
    double* B4FULL_RESTRICT A_canonical)
{
    if (plan.uses_reflection_compact()) {
        scatter_full_output_reflection_compact_by_i1_to_canonical(
            support, A_auto_by_i1, A_canonical);
    } else {
        scatter_full_output_by_i1_to_canonical_scalar_profiled(
            support, A_auto_by_i1, A_canonical);
    }
}


// ============================================================================
// Shared deterministic TLS panel reduction
// ============================================================================

inline void full_tensor_i1x_add_vector_inplace(
    double* B4FULL_RESTRICT dst,
    const double* B4FULL_RESTRICT src,
    std::size_t n)
{
#ifdef _OPENMP
#pragma omp simd
#endif
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] += src[i];
    }
}

struct FullTensorTlsPanelReductionStats {
    std::size_t panels_touched = 0;
    std::uint64_t panel_bytes = 0;
};

// Merge lazy (thread,i1) panels in a deterministic order.  The old reduction
// walked every complete panel serially, so a large output left all workers idle
// and read/wrote the destination once per contributing thread.  Here output
// blocks are independent OpenMP work items.  Each item visits source panels in
// increasing thread-id order, accumulates one destination value locally, and
// stores that value once.  The small-output cutoff avoids paying for another
// parallel region when the serial loop is already cheaper.
FullTensorTlsPanelReductionStats
full_tensor_reduce_tls_i1_panels_deterministic(
    const std::vector<std::vector<double>>& tls_output,
    std::size_t active_workers,
    std::size_t I1,
    std::size_t panel_elems,
    double* B4FULL_RESTRICT output_by_i1,
    bool use_openmp,
    std::size_t parallel_min_output_values = 256u * 1024u);

// ============================================================================
// Adaptive weighted (i1, x-range) scheduled backend
// ============================================================================
// This scheduler sits between the two previous extremes:
//   * full-i1 tasks: excellent locality, too few/imbalanced tasks;
//   * i1*x tasks: excellent balance, too fragmented.
//
// It builds weighted tasks (ii1, ix_begin, ix_end).  Low-cost i1 panels remain
// unsplit; expensive panels are split into x-ranges by predicted work.  Tasks
// are optionally sorted by decreasing predicted work (LPT-style scheduling).
// The progress bar tracks predicted scheduled work rather than raw task count.

enum class FullTensorAdaptiveScheduleMode {
    // Choose full-i1 weighted scheduling when I1 already gives enough tasks,
    // otherwise use weighted i1/x-range chunks with TLS panels.
    Auto,

    // One task per i1 panel, sorted by predicted work. Best locality, limited
    // scalability when I1 is small compared with the machine.
    WeightedI1Only,

    // Weighted (i1, x-range) tasks with thread-local per-i1 output panels.
    // Better scalability on large machines, more fragmented locality.
    WeightedXRangeTLS
};

inline const char* full_tensor_adaptive_schedule_mode_name(FullTensorAdaptiveScheduleMode mode) noexcept {
    switch (mode) {
        case FullTensorAdaptiveScheduleMode::Auto: return "Auto";
        case FullTensorAdaptiveScheduleMode::WeightedI1Only: return "WeightedI1Only";
        case FullTensorAdaptiveScheduleMode::WeightedXRangeTLS: return "WeightedXRangeTLS";
    }
    return "Unknown";
}

struct FullTensorAdaptiveScheduleOptions {
    // High-level scheduling strategy. Auto is the recommended default.
    FullTensorAdaptiveScheduleMode mode = FullTensorAdaptiveScheduleMode::Auto;

    // In Auto mode, if I1 >= i1_only_min_tasks_per_worker * active_workers,
    // do not split x. This preserves full-panel locality on small/medium CPUs.
    // For the M3 Pro example I1=21, workers=11 gives 1.91 tasks/worker, so
    // the default 1.50 selects weighted full-i1 scheduling. On 64 cores it
    // selects weighted x-range chunks instead.
    double i1_only_min_tasks_per_worker = 1.50;

    // If the worst i1 panel is much larger than the mean, Auto may still split
    // even when I1 is near the threshold. Use infinity-like values to disable.
    double i1_only_max_imbalance = 3.0;

    // Heavy-task guard for Auto mode.  I1-only scheduling is allowed only when
    // enough i1 panels exist AND no single i1 panel is predicted to be enormous.
    // The absolute threshold is in the same support-derived predictor units as
    // target_predicted_work.  The default keeps small Same=20,Dl=4 full-i1, but
    // forces weighted x-range splitting for much larger Same=30,Dl=10-like runs.
    bool split_heavy_i1_by_predicted_work = true;
    double heavy_i1_predicted_work_threshold = 5.0e12;
    double i1_only_max_work_over_target = 1.15;

    // Cap the weighted task target when the problem is huge, so progress and
    // load balance do not wait for monster full-i1 panels to finish.  0 disables
    // the cap.  This is deliberately separate from oversubscription: it limits
    // absolute task size, while oversubscription controls tasks/worker.
    double max_predicted_work_per_task = 5.0e12;

    // Number of target tasks per worker when x-range splitting is active, before
    // low-cost panel coalescing. 3 is a decent compromise: enough tasks to avoid
    // a tail, not 735 crumbs.
    double oversubscription = 3.0;

    // An i1 panel is kept whole if its predicted cost is below
    // split_threshold_factor * target_task_work.
    double split_threshold_factor = 1.15;

    // Avoid pathological one-x fragments unless an individual x is already
    // heavier than the target.
    std::size_t min_x_per_task = 1;

    // Optional cap. 0 means no explicit cap beyond the weighted target.
    std::size_t max_tasks_per_i1 = 0;

    // Merge a small trailing range into the previous task if its predicted work
    // is below this fraction of the target.
    double tail_merge_fraction = 0.35;

    // Process heavier predicted tasks first.  This is the classic LPT heuristic.
    bool sort_descending_predicted_work = true;

    // The seed-support term gives i1-dependence to the predictor.  Turning it
    // off leaves only q/x weights, which is usually too blind for edge i1 panels.
    bool use_seed_support_predictor = true;

    // Scale on the seed part of the work predictor.  Keep at 1 unless profiling
    // shows the predictor overreacts to edge/middle seed support.
    double seed_support_power = 1.0;

    // Use thread-local per-i1 output panels. This avoids a global lock/reduction
    // after every scheduled task. Each worker accumulates all chunks it receives
    // for a given i1 into a private panel, then all private panels are reduced
    // once after the task loop.
    bool use_thread_local_i1_panels = true;

    // Soft memory budget for thread-local output panels. 0 means unlimited.
    // If the estimated worst-case TLS panel memory exceeds this budget, the
    // implementation falls back to per-task locked reduction.
    std::uint64_t max_tls_output_scratch_bytes = 0;

    // Print the task-building summary.
    bool print_schedule_summary = true;
};

struct FullTensorScheduledTask {
    std::size_t ii1 = 0;
    std::size_t ix_begin = 0;
    std::size_t ix_end = 0; // exclusive
    double predicted_work = 0.0;
    std::uint64_t predicted_units = 1;
    std::uint64_t predicted_p_rows = 0;
    std::uint64_t planned_final_rows = 0;
    bool whole_i1 = false;

};

struct FullTensorScheduledTaskRecord {
    std::size_t ii1 = 0;
    std::size_t ix_begin = 0;
    std::size_t ix_end = 0;
    int thread_id = 0;
    bool whole_i1 = false;
    double predicted_work = 0.0;
    std::uint64_t predicted_units = 1;
    std::uint64_t predicted_p_rows = 0;
    double seconds = 0.0;
    std::uint64_t q_blocks_accumulated = 0;
    std::uint64_t final_dot_terms = 0;
    std::uint64_t T2_dot_terms = 0;
    std::uint64_t T4_dot_terms = 0;
    std::uint64_t middle_updates = 0;
    long double approx_flops = 0.0L;
};

struct FullTensorScheduledBalanceProfile {
    std::size_t I1 = 0;
    std::size_t X = 0;
    std::size_t workers = 0;
    std::size_t selected_p_batch = 0;
    FullTensorTriangularFinalTileShape selected_final_tile_shape =
        FullTensorTriangularFinalTileShape::Tile4x4;
    bool selected_final_tile_applicable = false;
    double target_task_work = 0.0;
    double total_predicted_work = 0.0;
    std::size_t whole_i1_tasks = 0;
    std::size_t split_i1_panels = 0;
    FullTensorAdaptiveScheduleMode selected_mode = FullTensorAdaptiveScheduleMode::Auto;
    bool auto_selected_i1_only = false;
    double i1_tasks_per_worker = 0.0;
    double predicted_i1_max_over_mean = 0.0;
    double predicted_i1_max_over_target = 0.0;
    double predicted_i1_max_work = 0.0;
    bool heavy_i1_forced_split = false;
    bool inner_tile_parallel_requested = false;
    bool inner_tile_parallel_enabled = false;
    bool used_thread_local_i1_panels = false;
    std::size_t tls_panels_touched = 0;
    std::uint64_t tls_output_bytes = 0;
    double tls_reduction_seconds = 0.0;
    std::vector<FullTensorScheduledTaskRecord> tasks;

    void reset(std::size_t I1_, std::size_t X_, std::size_t workers_, std::size_t n_tasks);

    void print(std::ostream& os, std::size_t top_n = 20) const;
};

// Count the final matrix-product rows which may be executed for each x slice.
// Within one selected output route every row has the same output-panel cost,
// so this is equivalent to final-dot terms without risking an unnecessary
// multiplication by the (potentially very large) panel extent.  The traversal
// uses only support metadata already visited by scheduling.
std::vector<std::uint64_t>
build_full_tensor_cpu_planned_final_rows_by_x(
    const FullTensorInputSupportPlan& support,
    const FullTensorReassociatedAutoOutputPlan& plan,
    const FullTensorQReflectionOrbitPlan& orbit_plan,
    bool use_canonical_q_orbits);

double full_tensor_scheduled_seed_support_units(
    const FullTensorInputSupportPlan& support,
    std::size_t ii1,
    std::size_t ix,
    const FullTensorAdaptiveScheduleOptions& sopt);

std::vector<double> full_tensor_scheduled_x_q_work(
    const FullTensorInputSupportPlan& support,
    const std::vector<std::uint64_t>& q_weights);

std::vector<FullTensorScheduledTask> build_full_tensor_adaptive_scheduled_tasks(
    const FullTensorInputSupportPlan& support,
    const FullTensorReassociatedAutoOutputPlan& plan,
    const std::vector<std::uint64_t>& q_weights,
    std::size_t active_workers,
    std::size_t p_batch,
    const FullTensorAdaptiveScheduleOptions& sopt,
    double& total_predicted_work,
    double& target_task_work,
    std::size_t& whole_i1_tasks,
    std::size_t& split_i1_panels,
    FullTensorAdaptiveScheduleMode selected_mode,
    double& i1_tasks_per_worker,
    double& predicted_i1_max_over_mean,
    double& predicted_i1_max_over_target,
    double& predicted_i1_max_work);

void print_full_tensor_adaptive_schedule_summary(
    const FullTensorReassociatedAutoOutputPlan& plan,
    const FullTensorReflectionCompactTriangularCacheInfo& cache,
    const std::vector<FullTensorScheduledTask>& tasks,
    std::size_t I1,
    std::size_t X,
    std::size_t M,
    std::size_t N,
    std::size_t p_batch,
    std::size_t active_workers,
    double total_predicted_work,
    double target_task_work,
    std::size_t whole_i1_tasks,
    std::size_t split_i1_panels,
    FullTensorAdaptiveScheduleMode selected_mode,
    double i1_tasks_per_worker,
    double predicted_i1_max_over_mean,
    double predicted_i1_max_over_target,
    double predicted_i1_max_work,
    bool heavy_i1_forced_split,
    bool inner_tile_parallel_requested,
    bool inner_tile_parallel_enabled,
    const FullTensorReflectionCompactTriangularBatchedOptions& topt,
    const FullTensorAdaptiveScheduleOptions& sopt,
    std::ostream& os);

void process_full_tensor_xrange_task_triangular_batched_reassociated(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    std::size_t ii1,
    std::size_t ix_begin,
    std::size_t ix_end,
    double* B4FULL_RESTRICT B_upper_task,
    const FullTensorReflectionCompactOutputMap& compact_map,
    const FullTensorTriangularReverseColumnMap& reverse_map,
    const FullTensorQReflectionOrbitPlan& q_orbit_plan,
    FullTensorPackedContractionL25ScalarOptions opt,
    const FullTensorReflectionCompactTriangularBatchedOptions& topt,
    FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    FullTensorTriangularBatchedLRWorkspace& batch,
    FullTensorPackedContractionL25ScalarProfile& profile);

void process_full_tensor_xrange_task_reassociated_rectangular_batched(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    std::size_t ii1,
    std::size_t ix_begin,
    std::size_t ix_end,
    double* B4FULL_RESTRICT A_panel_task,
    FullTensorPackedContractionL25ScalarOptions opt,
    const FullTensorReflectionCompactTriangularBatchedOptions& topt,
    FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    FullTensorReassociatedRectangularBatchedLRWorkspace& batch,
    FullTensorPackedContractionL25ScalarProfile& profile);


// Same as process_full_tensor_xrange_task_triangular_batched_reassociated(),
// but it does NOT zero the output panel. Used by the adaptive TLS scheduler:
// each worker owns a private (thread,i1) panel and accumulates several
// scheduled x-ranges into it before the final cross-thread reduction.
void process_full_tensor_xrange_task_triangular_batched_reassociated_add(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    std::size_t ii1,
    std::size_t ix_begin,
    std::size_t ix_end,
    double* B4FULL_RESTRICT B_upper_task,
    const FullTensorReflectionCompactOutputMap& compact_map,
    const FullTensorTriangularReverseColumnMap& reverse_map,
    const FullTensorQReflectionOrbitPlan& q_orbit_plan,
    FullTensorPackedContractionL25ScalarOptions opt,
    const FullTensorReflectionCompactTriangularBatchedOptions& topt,
    FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    FullTensorTriangularBatchedLRWorkspace& batch,
    FullTensorPackedContractionL25ScalarProfile& profile);

// Same as process_full_tensor_xrange_task_reassociated_rectangular_batched(),
// but without zeroing A_panel_task. Used by the adaptive TLS scheduler.
void process_full_tensor_xrange_task_reassociated_rectangular_batched_add(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    std::size_t ii1,
    std::size_t ix_begin,
    std::size_t ix_end,
    double* B4FULL_RESTRICT A_panel_task,
    FullTensorPackedContractionL25ScalarOptions opt,
    const FullTensorReflectionCompactTriangularBatchedOptions& topt,
    FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    FullTensorReassociatedRectangularBatchedLRWorkspace& batch,
    FullTensorPackedContractionL25ScalarProfile& profile);

b4gpu::FullTensorGpuNativeOutputLayout
make_full_tensor_gpu_native_output_layout(
    const FullTensorInputSupportPlan& support,
    const FullTensorReassociatedAutoOutputPlan& plan);


FullTensorReassociatedAutoOutputPlan
contract_full_tensor_packed_fused_l25_stacked_scalar_profiled_reassociated_auto_scheduled_by_i1_layout(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    double* B4FULL_RESTRICT A_auto_by_i1,
    FullTensorPackedContractionL25ScalarOptions opt,
    FullTensorPackedContractionL25ScalarProfile& profile,
    FullTensorReflectionCompactTriangularBatchedOptions topt = {},
    FullTensorAdaptiveScheduleOptions sopt = {},
    FullTensorReflectionCompactWeightedProgressOptions progress_options = {},
    FullTensorScheduledBalanceProfile* balance_profile = nullptr);

} // namespace b4support
