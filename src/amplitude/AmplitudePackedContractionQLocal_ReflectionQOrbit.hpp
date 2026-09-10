// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>
#include <array>
#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>

// Q-local fused/projected contraction support. This route never materializes
// global T2/T4 intermediates.
#include "amplitude/AmplitudePackedContraction.hpp"
#include "core/Failure.hpp"

#ifndef B4PACK_RESTRICT
#  if defined(__GNUC__) || defined(__clang__)
#    define B4PACK_RESTRICT __restrict__
#  elif defined(_MSC_VER)
#    define B4PACK_RESTRICT __restrict
#  else
#    define B4PACK_RESTRICT
#  endif
#endif



namespace b4support {

static constexpr std::size_t QLOCAL_K_HIST_MAX = 128;

inline void qlocal_add_k_histogram_count(
    std::array<std::uint64_t, QLOCAL_K_HIST_MAX + 1>& hist,
    std::uint64_t& overflow,
    std::size_t k) noexcept
{
    if (k <= QLOCAL_K_HIST_MAX) {
        ++hist[k];
    } else {
        ++overflow;
    }
}


// ============================================================================
// q-local streamed/projected contraction stats and phase profile
// ============================================================================

struct PackedQLocalContractionStats {
    std::uint64_t T1_values_total = 0;
    std::uint64_t T3_values_total = 0;

    std::uint64_t active_q_blocks = 0;
    std::uint64_t T2_rows_built   = 0;
    std::uint64_t T4_rows_built   = 0;
    std::uint64_t final_pairs     = 0;

    std::uint64_t project_C_times_T2_blocks  = 0;
    std::uint64_t project_Ct_times_T4_blocks = 0;
    std::uint64_t pairwise_blocks            = 0;

    // Small-K kernel usage inside q-local projection/dot kernels.
    std::uint64_t axpy_K1 = 0;
    std::uint64_t axpy_K2 = 0;
    std::uint64_t axpy_K3 = 0;
    std::uint64_t axpy_K4 = 0;
    std::uint64_t axpy_generic = 0;

    std::uint64_t dot_K1 = 0;
    std::uint64_t dot_K2 = 0;
    std::uint64_t dot_K3 = 0;
    std::uint64_t dot_K4 = 0;
    std::uint64_t dot_generic = 0;

    // Histograms for the kernels that actually matter in the q-loop.
    // project_axpy_K_hist counts only C*T2 / C^T*T4 projection AXPYs,
    // not the row-building AXPYs used to construct T2_row/T4_row.
    std::array<std::uint64_t, QLOCAL_K_HIST_MAX + 1> project_axpy_K_hist{};
    std::array<std::uint64_t, QLOCAL_K_HIST_MAX + 1> final_dot_K_hist{};
    std::uint64_t project_axpy_K_overflow = 0;
    std::uint64_t final_dot_K_overflow = 0;
    std::uint64_t project_axpy_full_range = 0;
    std::uint64_t final_dot_full_range = 0;

    // Coarse phase timings. These are intentionally not gathered inside the
    // innermost q-block kernels; timing millions of tiny pieces is how one
    // accidentally benchmarks the clock.
    double time_T1_support = 0.0;
    double time_T1_values  = 0.0;
    double time_T3_support = 0.0;
    double time_T3_values  = 0.0;
    double time_q_loop     = 0.0;
    double time_total      = 0.0;

    void clear() noexcept {
        T1_values_total = 0;
        T3_values_total = 0;
        active_q_blocks = 0;
        T2_rows_built = 0;
        T4_rows_built = 0;
        final_pairs = 0;
        project_C_times_T2_blocks = 0;
        project_Ct_times_T4_blocks = 0;
        pairwise_blocks = 0;
        axpy_K1 = axpy_K2 = axpy_K3 = axpy_K4 = axpy_generic = 0;
        dot_K1 = dot_K2 = dot_K3 = dot_K4 = dot_generic = 0;
        project_axpy_K_hist.fill(0);
        final_dot_K_hist.fill(0);
        project_axpy_K_overflow = 0;
        final_dot_K_overflow = 0;
        project_axpy_full_range = 0;
        final_dot_full_range = 0;
        time_T1_support = 0.0;
        time_T1_values = 0.0;
        time_T3_support = 0.0;
        time_T3_values = 0.0;
        time_q_loop = 0.0;
        time_total = 0.0;
    }

    void print(std::ostream& os) const {
        const double mib = 1024.0 * 1024.0;

        os << "\nPacked q-local streamed/projected contraction statistics\n";
        os << "=====================================================\n";
        os << "T1 values across slices        = " << T1_values_total
           << " | " << std::fixed << std::setprecision(6)
           << static_cast<double>(T1_values_total * sizeof(double)) / mib
           << " MiB\n";

        os << "T3 values across slices        = " << T3_values_total
           << " | " << std::fixed << std::setprecision(6)
           << static_cast<double>(T3_values_total * sizeof(double)) / mib
           << " MiB\n";

        os << "active q blocks                = " << active_q_blocks << "\n";
        os << "T2 rows built                  = " << T2_rows_built << "\n";
        os << "T4 rows built                  = " << T4_rows_built << "\n";
        os << "final (k3,k4) pairs visited    = " << final_pairs << "\n";
        os << "project C*T2 q-blocks          = " << project_C_times_T2_blocks << "\n";
        os << "project C^T*T4 q-blocks        = " << project_Ct_times_T4_blocks << "\n";
        os << "pairwise q-blocks              = " << pairwise_blocks << "\n";

        os << "\nSmall-K q-local kernel usage\n";
        os << "----------------------------\n";
        os << "axpy K=1                      = " << axpy_K1 << "\n";
        os << "axpy K=2                      = " << axpy_K2 << "\n";
        os << "axpy K=3                      = " << axpy_K3 << "\n";
        os << "axpy K=4                      = " << axpy_K4 << "\n";
        os << "axpy generic                  = " << axpy_generic << "\n";
        os << "dot  K=1                      = " << dot_K1 << "\n";
        os << "dot  K=2                      = " << dot_K2 << "\n";
        os << "dot  K=3                      = " << dot_K3 << "\n";
        os << "dot  K=4                      = " << dot_K4 << "\n";
        os << "dot  generic                  = " << dot_generic << "\n";

        os << "\nProjection AXPY K histogram\n";
        os << "---------------------------\n";
        for (std::size_t k = 1; k <= QLOCAL_K_HIST_MAX; ++k) {
            if (project_axpy_K_hist[k] != 0) {
                os << "K=" << std::setw(3) << k
                   << " : " << project_axpy_K_hist[k] << "\n";
            }
        }
        if (project_axpy_K_overflow != 0) {
            os << "K>" << QLOCAL_K_HIST_MAX
               << " : " << project_axpy_K_overflow << "\n";
        }
        os << "full-range projection AXPYs   = "
           << project_axpy_full_range << "\n";

        os << "\nFinal dot K histogram\n";
        os << "---------------------\n";
        for (std::size_t k = 1; k <= QLOCAL_K_HIST_MAX; ++k) {
            if (final_dot_K_hist[k] != 0) {
                os << "K=" << std::setw(3) << k
                   << " : " << final_dot_K_hist[k] << "\n";
            }
        }
        if (final_dot_K_overflow != 0) {
            os << "K>" << QLOCAL_K_HIST_MAX
               << " : " << final_dot_K_overflow << "\n";
        }
        os << "full-range final dots          = "
           << final_dot_full_range << "\n";

        os << "\nq-local phase profile\n";
        os << "---------------------\n";
        os << "T1 support time               = " << std::fixed << std::setprecision(6)
           << time_T1_support << " s\n";
        os << "T1 value time                 = " << time_T1_values << " s\n";
        os << "T3 support time               = " << time_T3_support << " s\n";
        os << "T3 value time                 = " << time_T3_values << " s\n";
        os << "q-loop streamed final time     = " << time_q_loop << " s\n";
        os << "total measured contraction     = " << time_total << " s\n";
    }
};

struct PackedQLocalThreadCounters {
    std::uint64_t active_q_blocks = 0;
    std::uint64_t T2_rows_built = 0;
    std::uint64_t T4_rows_built = 0;
    std::uint64_t final_pairs = 0;
    std::uint64_t project_C_times_T2_blocks = 0;
    std::uint64_t project_Ct_times_T4_blocks = 0;
    std::uint64_t pairwise_blocks = 0;
    std::uint64_t axpy_K1 = 0, axpy_K2 = 0, axpy_K3 = 0, axpy_K4 = 0, axpy_generic = 0;
    std::uint64_t dot_K1 = 0, dot_K2 = 0, dot_K3 = 0, dot_K4 = 0, dot_generic = 0;

    std::array<std::uint64_t, QLOCAL_K_HIST_MAX + 1> project_axpy_K_hist{};
    std::array<std::uint64_t, QLOCAL_K_HIST_MAX + 1> final_dot_K_hist{};
    std::uint64_t project_axpy_K_overflow = 0;
    std::uint64_t final_dot_K_overflow = 0;
    std::uint64_t project_axpy_full_range = 0;
    std::uint64_t final_dot_full_range = 0;

    void record_project_axpy_k(std::size_t k, std::size_t full_k) noexcept {
        qlocal_add_k_histogram_count(project_axpy_K_hist,
                                     project_axpy_K_overflow,
                                     k);
        if (full_k != 0 && k == full_k) {
            ++project_axpy_full_range;
        }
    }

    void record_final_dot_k(std::size_t k, std::size_t full_k) noexcept {
        qlocal_add_k_histogram_count(final_dot_K_hist,
                                     final_dot_K_overflow,
                                     k);
        if (full_k != 0 && k == full_k) {
            ++final_dot_full_range;
        }
    }

    void add_to(PackedQLocalContractionStats& s) const noexcept {
        s.active_q_blocks += active_q_blocks;
        s.T2_rows_built += T2_rows_built;
        s.T4_rows_built += T4_rows_built;
        s.final_pairs += final_pairs;
        s.project_C_times_T2_blocks += project_C_times_T2_blocks;
        s.project_Ct_times_T4_blocks += project_Ct_times_T4_blocks;
        s.pairwise_blocks += pairwise_blocks;
        s.axpy_K1 += axpy_K1; s.axpy_K2 += axpy_K2; s.axpy_K3 += axpy_K3;
        s.axpy_K4 += axpy_K4; s.axpy_generic += axpy_generic;
        s.dot_K1 += dot_K1; s.dot_K2 += dot_K2; s.dot_K3 += dot_K3;
        s.dot_K4 += dot_K4; s.dot_generic += dot_generic;

        for (std::size_t k = 0; k <= QLOCAL_K_HIST_MAX; ++k) {
            s.project_axpy_K_hist[k] += project_axpy_K_hist[k];
            s.final_dot_K_hist[k] += final_dot_K_hist[k];
        }
        s.project_axpy_K_overflow += project_axpy_K_overflow;
        s.final_dot_K_overflow += final_dot_K_overflow;
        s.project_axpy_full_range += project_axpy_full_range;
        s.final_dot_full_range += final_dot_full_range;
    }
};

inline double qlocal_seconds_since(
    const std::chrono::high_resolution_clock::time_point& t0) noexcept
{
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clock::now() - t0).count();
}

// ============================================================================
// Small local helpers
// ============================================================================

inline void qlocal_cover_range_with_range(Range1D& cover,
                                          const Range1D& r) noexcept
{
    if (r.empty()) {
        return;
    }

    if (cover.empty()) {
        cover.begin = r.begin;
        cover.end   = r.end;
        cover.offset = 0ull;
        return;
    }

    cover.begin = std::min(cover.begin, r.begin);
    cover.end   = std::max(cover.end,   r.end);
}

inline void qlocal_zero_row(double* B4PACK_RESTRICT row,
                            std::size_t n) noexcept
{
#ifdef _OPENMP
#pragma omp simd
#endif
    for (std::size_t i = 0; i < n; ++i) {
        row[i] = 0.0;
    }
}

enum class QLocalAxpyKind : std::uint8_t {
    RowBuild,
    Projection
};

inline void qlocal_axpy_full_range(double alpha,
                                   const double* B4PACK_RESTRICT x,
                                   double* B4PACK_RESTRICT y,
                                   std::size_t n) noexcept
{
#ifdef _OPENMP
#pragma omp simd
#endif
    for (std::size_t i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

inline double qlocal_dot_full_range(const double* B4PACK_RESTRICT a,
                                    const double* B4PACK_RESTRICT b,
                                    std::size_t n) noexcept
{
    double s = 0.0;
#ifdef _OPENMP
#pragma omp simd reduction(+:s)
#endif
    for (std::size_t i = 0; i < n; ++i) {
        s += a[i] * b[i];
    }
    return s;
}

inline void qlocal_axpy_adaptive(double alpha,
                                 const double* B4PACK_RESTRICT x,
                                 double* B4PACK_RESTRICT y,
                                 std::size_t n,
                                 PackedQLocalThreadCounters* counters = nullptr,
                                 QLocalAxpyKind kind = QLocalAxpyKind::RowBuild,
                                 std::size_t full_k = 0) noexcept
{
    if (n == 0) {
        return;
    }

    if (counters != nullptr && kind == QLocalAxpyKind::Projection) {
        counters->record_project_axpy_k(n, full_k);
    }

    // Full-range medium/large K path. This keeps the compiler-facing loop simple
    // when the active l25 interval is exactly the whole local vector. We do not
    // unroll K=5..21 here because that already proved slower than the SIMD loop.
    if (full_k != 0 && n == full_k && n > 4) {
        if (counters) {
            ++counters->axpy_generic;
        }
        qlocal_axpy_full_range(alpha, x, y, n);
        return;
    }

    switch (n) {
        case 1:
            y[0] += alpha * x[0];
            if (counters) ++counters->axpy_K1;
            return;
        case 2:
            y[0] += alpha * x[0];
            y[1] += alpha * x[1];
            if (counters) ++counters->axpy_K2;
            return;
        case 3:
            y[0] += alpha * x[0];
            y[1] += alpha * x[1];
            y[2] += alpha * x[2];
            if (counters) ++counters->axpy_K3;
            return;
        case 4:
            y[0] += alpha * x[0];
            y[1] += alpha * x[1];
            y[2] += alpha * x[2];
            y[3] += alpha * x[3];
            if (counters) ++counters->axpy_K4;
            return;
        default:
            qlocal_axpy_full_range(alpha, x, y, n);
            if (counters) ++counters->axpy_generic;
            return;
    }
}
inline double qlocal_dot_adaptive(const double* B4PACK_RESTRICT a,
                                  const double* B4PACK_RESTRICT b,
                                  std::size_t n,
                                  PackedQLocalThreadCounters* counters = nullptr,
                                  std::size_t full_k = 0) noexcept
{
    if (n == 0) {
        return 0.0;
    }

    if (counters != nullptr) {
        counters->record_final_dot_k(n, full_k);
    }

    // Full-range medium/large K path. Same arithmetic as the generic path, but
    // expressed as a clean local loop for the compiler when the whole l25 row is active.
    if (full_k != 0 && n == full_k && n > 4) {
        if (counters) ++counters->dot_generic;
        return qlocal_dot_full_range(a, b, n);
    }

    switch (n) {
        case 1:
            if (counters) ++counters->dot_K1;
            return a[0] * b[0];
        case 2:
            if (counters) ++counters->dot_K2;
            return a[0] * b[0] + a[1] * b[1];
        case 3:
            if (counters) ++counters->dot_K3;
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        case 4:
            if (counters) ++counters->dot_K4;
            return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        default:
            if (counters) ++counters->dot_generic;
            return qlocal_dot_full_range(a, b, n);
    }
}
struct PackedQLocalSupportWorkspace {
    RangeBlockSupport1D T1; // block (l45,k4,l35) -> l25
    RangeBlockSupport1D T3; // block (l23,k3,l24) -> l25

    void initialize(const AmplitudeInputSupportPlan& in) {
        const std::size_t K3d = in.channels.k3.size();
        const std::size_t K4d = in.channels.k4.size();

        const std::size_t L23 = in.l23.size();
        const std::size_t L24 = in.l24.size();
        const std::size_t L25 = in.l25.size();
        const std::size_t L35 = in.l35.size();
        const std::size_t L45 = in.l45.size();

        (void)L25;

        T1 = make_support_1d(
            "q-local T1 slice support: block (l45,k4,l35) -> l25",
            {L45, K4d, L35},
            in.l25.size()
        );

        T3 = make_support_1d(
            "q-local T3 slice support: block (l23,k3,l24) -> l25",
            {L23, K3d, L24},
            in.l25.size()
        );
    }
};

struct PackedQLocalValueWorkspace {
    ReusableDoubleBuffer T1;
    ReusableDoubleBuffer T3;
};

struct PackedQLocalThreadWorkspace {
    // Legacy single panel used by the tiny-block pairwise fallback. Kept because
    // deleting working fallback code is how benchmarks become archaeology.
    std::vector<double> panel;

    // One streamed row used by legacy fallback paths and by diagnostic builds.
    std::vector<double> row;

    // Dedicated q-block panels used by the panelized projected kernels:
    //   T2_panel[jb,l25], T4_panel[ib,l25], work_panel[min-side,l25].
    // These are per-thread scratch buffers, not global intermediates.
    std::vector<double> T2_panel;
    std::vector<double> T4_panel;
    std::vector<double> work_panel;

    // Ranges for rows currently stored in each panel.
    std::vector<Range1D> panel_ranges;
    std::vector<Range1D> T2_ranges;
    std::vector<Range1D> T4_ranges;

    std::size_t capacity_rows = 0;
    std::size_t capacity_l25 = 0;
    std::size_t capacity_ranges = 0;

    void ensure(std::size_t K3d, std::size_t K4d, std::size_t L25) {
        const std::size_t max_rows = std::max<std::size_t>(1, std::max(K3d, K4d));
        const std::size_t elems = max_rows * L25;

        if (panel.size() < elems)     panel.resize(elems);
        if (T2_panel.size() < elems)  T2_panel.resize(elems);
        if (T4_panel.size() < elems)  T4_panel.resize(elems);
        if (work_panel.size() < elems) work_panel.resize(elems);

        if (row.size() < L25) row.resize(L25);

        if (panel_ranges.size() < max_rows) panel_ranges.resize(max_rows);
        if (T2_ranges.size() < max_rows)    T2_ranges.resize(max_rows);
        if (T4_ranges.size() < max_rows)    T4_ranges.resize(max_rows);

        capacity_rows = std::max(capacity_rows, max_rows);
        capacity_l25 = std::max(capacity_l25, L25);
        capacity_ranges = std::max(capacity_ranges, max_rows);
    }
};

// ============================================================================
// T1/T3 support construction only

//
// The q-local backend does not build T2/T4 supports. Those intermediates are
// local panels, created and consumed immediately for each q=(l24,l34,l35).
// ============================================================================

void build_T1_support_qlocal_slice_into(
    const AmplitudeInputSupportPlan& in,
    std::size_t ix,
    RangeBlockSupport1D& T1,
    PackedContractionOptions opt = {});

void build_T3_support_qlocal_slice_into(
    const AmplitudeInputSupportPlan& in,
    std::size_t ix,
    RangeBlockSupport1D& T3,
    PackedContractionOptions opt = {});


// ============================================================================
// q-local streamed row construction and final contraction
// ============================================================================

Range1D build_T2_row_for_q_k4(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T1_support,
    const double* B4PACK_RESTRICT T1_values,
    const PackedAmplitudeBoosters& B,
    std::size_t ik4,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* B4PACK_RESTRICT row,
    PackedQLocalThreadCounters* counters = nullptr);

Range1D build_T4_row_for_q_k3(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T3_support,
    const double* B4PACK_RESTRICT T3_values,
    const PackedAmplitudeBoosters& B,
    std::size_t ik3,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* B4PACK_RESTRICT row,
    PackedQLocalThreadCounters* counters = nullptr);


// ----------------------------------------------------------------------------
// Dedicated q-block panel builders
//
// These build dense T2/T4 row panels for one q=(l24,l34,l35) block from the
// already materialized T1/T3 slice intermediates.  They are intentionally local
// to one q-block and one thread: no global T2/T4 storage is created.  The row
// ranges are preserved so nonsymmetric / partially valid cases still avoid
// multiplying over meaningless l25 entries.
// ----------------------------------------------------------------------------

Range1D build_T2_panel_for_q(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T1_support,
    const double* B4PACK_RESTRICT T1_values,
    const PackedAmplitudeBoosters& B,
    std::size_t ik4_begin,
    std::size_t nk4,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* B4PACK_RESTRICT T2_panel,
    Range1D* B4PACK_RESTRICT T2_ranges,
    PackedQLocalThreadCounters* counters = nullptr);

Range1D build_T4_panel_for_q(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T3_support,
    const double* B4PACK_RESTRICT T3_values,
    const PackedAmplitudeBoosters& B,
    std::size_t ik3_begin,
    std::size_t nk3,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* B4PACK_RESTRICT T4_panel,
    Range1D* B4PACK_RESTRICT T4_ranges,
    PackedQLocalThreadCounters* counters = nullptr);

enum class QLocalFinalStrategy : std::uint8_t {
    ProjectCTimesT2,
    ProjectCtTimesT4,
    PairwiseStoreT4,
    PairwiseStoreT2
};

inline QLocalFinalStrategy choose_qlocal_strategy(std::size_t nk3,
                                                  std::size_t nk4,
                                                  std::size_t L25) noexcept
{
    const std::size_t pairs = nk3 * nk4;
    const std::size_t small_side = std::min(nk3, nk4);

    // For tiny blocks, the projection setup can cost more than it saves.
    // This is intentionally conservative: large q-blocks should use projection.
    if (pairs <= 16 || (small_side <= 2 && pairs <= 64 && L25 <= 8)) {
        return (nk3 <= nk4)
            ? QLocalFinalStrategy::PairwiseStoreT4
            : QLocalFinalStrategy::PairwiseStoreT2;
    }

    // Otherwise project onto the smaller output side. This still considers more
    // than just nk3 <= nk4 because tiny blocks can use the pairwise fallback.
    return (nk3 <= nk4)
        ? QLocalFinalStrategy::ProjectCTimesT2
        : QLocalFinalStrategy::ProjectCtTimesT4;
}

double contract_q_block_project_C_times_T2_streamed(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* B4PACK_RESTRICT T1_values,
    const RangeBlockSupport1D& T3_support,
    const double* B4PACK_RESTRICT T3_values,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    PackedQLocalThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters = nullptr);

double contract_q_block_project_Ct_times_T4_streamed(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* B4PACK_RESTRICT T1_values,
    const RangeBlockSupport1D& T3_support,
    const double* B4PACK_RESTRICT T3_values,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    PackedQLocalThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters = nullptr);

double contract_q_block_pairwise_store_T4(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* B4PACK_RESTRICT T1_values,
    const RangeBlockSupport1D& T3_support,
    const double* B4PACK_RESTRICT T3_values,
    std::size_t ix,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    PackedQLocalThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters = nullptr);

double contract_q_block_pairwise_store_T2(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* B4PACK_RESTRICT T1_values,
    const RangeBlockSupport1D& T3_support,
    const double* B4PACK_RESTRICT T3_values,
    std::size_t ix,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    PackedQLocalThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters = nullptr);

double contract_q_block_streamed_adaptive(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* B4PACK_RESTRICT T1_values,
    const RangeBlockSupport1D& T3_support,
    const double* B4PACK_RESTRICT T3_values,
    std::size_t ix,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    PackedQLocalThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters = nullptr);

// ============================================================================
// Reflection-compatible q-orbit scalar amplitude contraction
// ============================================================================
//
// This is the safe scalar analogue of the output-reflection trick used for the
// full tensor. It does NOT try to identify the left and right branch rows.
// We already tested that stronger branch-level shortcut in the full-tensor case;
// it was wrong, because apparently phase conventions enjoy booby traps.
//
// Instead, this backend assumes only the fully contracted q-block scalar
// contribution is invariant under the reflection
//
//     (l24,l35) -> (l35,l24),
//
// for reflection-compatible boundary spins and selected intertwiners. It loops
// over representatives l24 <= l35 and multiplies off-diagonal contributions by 2.
//
// Conditions, using SpinsArray convention
//   {j12,j13,j14,j15,j23,j24,j25,j34,j35,j45}
// are:
//   j12 == j15, j13 == j14, j23 == j45, j24 == j35,
//   i2  == i5,  i3  == i4.
// ============================================================================

inline bool amplitude_reflection_spins_compatible(
    const std::array<int, 10>& spins) noexcept
{
    return spins[0] == spins[3]   // j12 == j15
        && spins[1] == spins[2]   // j13 == j14
        && spins[4] == spins[9]   // j23 == j45
        && spins[5] == spins[8];  // j24 == j35
}

inline bool amplitude_reflection_fixed_intertwiners_compatible(
    const std::array<int, 5>& fixed_intertwiners) noexcept
{
    return fixed_intertwiners[1] == fixed_intertwiners[4]  // i2 == i5
        && fixed_intertwiners[2] == fixed_intertwiners[3]; // i3 == i4
}

inline bool amplitude_reflection_support_axes_compatible(
    const AmplitudeInputSupportPlan& support) noexcept
{
    // These vector equalities are stronger than just equal dimensions. They make
    // sure that an index reflection really means the same doubled-spin value.
    return support.l23 == support.l45
        && support.l24 == support.l35
        && support.channels.k2 == support.channels.k5
        && support.channels.k3 == support.channels.k4;
}

inline bool amplitude_reflection_qorbit_compatible(
    const AmplitudeInputSupportPlan& support,
    const std::array<int, 10>& spins,
    const std::array<int, 5>& fixed_intertwiners) noexcept
{
    return amplitude_reflection_spins_compatible(spins)
        && amplitude_reflection_fixed_intertwiners_compatible(fixed_intertwiners)
        && amplitude_reflection_support_axes_compatible(support);
}

inline void require_amplitude_reflection_qorbit_compatible(
    const AmplitudeInputSupportPlan& support,
    const std::array<int, 10>& spins,
    const std::array<int, 5>& fixed_intertwiners)
{
    if (amplitude_reflection_qorbit_compatible(support, spins, fixed_intertwiners)) {
        return;
    }

    std::ostringstream oss;
    oss << "Amplitude q-orbit reflection backend requested for incompatible data.\n"
        << "Required conditions:\n"
        << "  spins: j12=j15, j13=j14, j23=j45, j24=j35\n"
        << "  fixed_intertwiners: i2=i5, i3=i4\n"
        << "  generated axes: l23=l45, l24=l35, k2=k5, k3=k4\n"
        << "Detected:\n"
        << "  spins ok   = " << (amplitude_reflection_spins_compatible(spins) ? "YES" : "NO") << "\n"
        << "  boundary intertwiner ok   = " << (amplitude_reflection_fixed_intertwiners_compatible(fixed_intertwiners) ? "YES" : "NO") << "\n"
        << "  axes ok    = " << (amplitude_reflection_support_axes_compatible(support) ? "YES" : "NO") << "\n";
    throw b4utils::InternalFailure(oss.str());
}

} // namespace b4support
