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
#include <mutex>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

#if defined(__APPLE__)
#  include <sys/types.h>
#  include <sys/sysctl.h>
#endif
#if defined(__unix__) || defined(__APPLE__)
#  include <unistd.h>
#endif

#ifdef _OPENMP
#  include <omp.h>
#endif

#include "amplitude/AmplitudeInputSupport_HalfSjC.hpp"
#include "amplitude/AmplitudePackedSJTensors_HalfSjC.hpp"
#include "amplitude/AmplitudePackedBoosters.hpp"
#include "amplitude/AmplitudePackedContractionQLocal_ReflectionQOrbitCostSelected_HalfSjC.hpp"
#include "amplitude/AmplitudePackedXQSlab.hpp"
#include "core/Failure.hpp"
#include "core/utils.hpp"

namespace b4support {


// ============================================================================
// Lightweight Linux RSS diagnostics for the coherent C1 memory model.
// These are intentionally compiled into the debug header rather than hidden in
// the allocator.  The point is to catch the exact phase where modeled bytes and
// real process memory stop agreeing, because apparently RAM accounting now needs
// a private investigator.
// ============================================================================

struct CoherentC1MemorySnapshot {
    std::uint64_t vm_size = 0;
    std::uint64_t vm_rss = 0;
    std::uint64_t vm_hwm = 0;
    std::uint64_t rss_anon = 0;
    std::uint64_t rss_file = 0;
    std::uint64_t rss_shmem = 0;
};

inline std::uint64_t coherent_C1_parse_status_kb_line(const std::string& line,
                                                       const char* key)
{
    const std::string prefix(key);
    if (line.rfind(prefix, 0) != 0) return 0;

    std::istringstream is(line.substr(prefix.size()));
    std::uint64_t kb = 0;
    is >> kb;
    return kb * 1024ull;
}

inline CoherentC1MemorySnapshot coherent_C1_current_memory_snapshot()
{
    CoherentC1MemorySnapshot m;
#if defined(__linux__)
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmSize:", 0) == 0) {
            m.vm_size = coherent_C1_parse_status_kb_line(line, "VmSize:");
        } else if (line.rfind("VmRSS:", 0) == 0) {
            m.vm_rss = coherent_C1_parse_status_kb_line(line, "VmRSS:");
        } else if (line.rfind("VmHWM:", 0) == 0) {
            m.vm_hwm = coherent_C1_parse_status_kb_line(line, "VmHWM:");
        } else if (line.rfind("RssAnon:", 0) == 0) {
            m.rss_anon = coherent_C1_parse_status_kb_line(line, "RssAnon:");
        } else if (line.rfind("RssFile:", 0) == 0) {
            m.rss_file = coherent_C1_parse_status_kb_line(line, "RssFile:");
        } else if (line.rfind("RssShmem:", 0) == 0) {
            m.rss_shmem = coherent_C1_parse_status_kb_line(line, "RssShmem:");
        }
    }
#endif
    return m;
}

inline void coherent_C1_print_memory_snapshot(std::ostream* log,
                                               const char* tag)
{
    if (!log) return;
    const CoherentC1MemorySnapshot m = coherent_C1_current_memory_snapshot();
    *log << "[coherent C1 rss] " << tag
         << " VmRSS="; xqslab_print_bytes(*log, m.vm_rss);
    *log << " VmHWM="; xqslab_print_bytes(*log, m.vm_hwm);
    *log << " VmSize="; xqslab_print_bytes(*log, m.vm_size);
    if (m.rss_anon || m.rss_file || m.rss_shmem) {
        *log << " RssAnon="; xqslab_print_bytes(*log, m.rss_anon);
        *log << " RssFile="; xqslab_print_bytes(*log, m.rss_file);
        *log << " RssShmem="; xqslab_print_bytes(*log, m.rss_shmem);
    }
    *log << "\n";
}

inline void coherent_C1_print_model_vs_rss(std::ostream* log,
                                           const char* tag,
                                           std::uint64_t fixed_bytes,
                                           std::uint64_t incremental_bytes,
                                           std::uint64_t peak_bytes,
                                           std::uint64_t budget_bytes)
{
    if (!log) return;
    *log << "[coherent C1 model] " << tag
         << " fixed="; xqslab_print_bytes(*log, fixed_bytes);
    *log << " incremental="; xqslab_print_bytes(*log, incremental_bytes);
    *log << " peak="; xqslab_print_bytes(*log, peak_bytes);
    *log << " budget="; xqslab_print_bytes(*log, budget_bytes);
    *log << "\n";
    coherent_C1_print_memory_snapshot(log, tag);
}

// ============================================================================
// Coherent q-local half-sjC contraction, fixed i1
// ============================================================================
// This is deliberately a sibling of the real q-local backend.  The support and
// sjX tensors are still the fixed-i1 real ones; only the already projected
// boosters B2..B5 are complex.  T1/T3 and the local q-panels are stored as SoA
// complex arrays so the CPU sees independent contiguous real and imaginary
// streams that it can vectorize.
// ============================================================================

struct PackedQLocalComplexValueWorkspace {
    b4utils::ComplexBuffer<> T1;
    b4utils::ComplexBuffer<> T3;
};

struct PackedQLocalComplexThreadWorkspace {
    std::vector<double> T2_re, T2_im;
    std::vector<double> T4_re, T4_im;
    std::vector<double> work_re, work_im;
    std::vector<Range1D> T2_ranges, T4_ranges;

    void ensure(std::size_t K3d, std::size_t K4d, std::size_t L25) {
        const std::size_t rows = std::max<std::size_t>(1, std::max(K3d, K4d));
        const std::size_t elems = rows * L25;
        if (T2_re.size() < elems) T2_re.resize(elems);
        if (T2_im.size() < elems) T2_im.resize(elems);
        if (T4_re.size() < elems) T4_re.resize(elems);
        if (T4_im.size() < elems) T4_im.resize(elems);
        if (work_re.size() < elems) work_re.resize(elems);
        if (work_im.size() < elems) work_im.resize(elems);
        if (T2_ranges.size() < rows) T2_ranges.resize(rows);
        if (T4_ranges.size() < rows) T4_ranges.resize(rows);
    }
};

inline void qlocal_zero_complex_row(double* B4PACK_RESTRICT re,
                                    double* B4PACK_RESTRICT im,
                                    std::size_t n) noexcept
{
#ifdef _OPENMP
#pragma omp simd
#endif
    for (std::size_t i = 0; i < n; ++i) {
        re[i] = 0.0;
        im[i] = 0.0;
    }
}

inline void qlocal_axpy_real_to_complex(double alpha,
                                        const double* B4PACK_RESTRICT x_re,
                                        const double* B4PACK_RESTRICT x_im,
                                        double* B4PACK_RESTRICT y_re,
                                        double* B4PACK_RESTRICT y_im,
                                        std::size_t n) noexcept
{
#ifdef _OPENMP
#pragma omp simd
#endif
    for (std::size_t i = 0; i < n; ++i) {
        y_re[i] += alpha * x_re[i];
        y_im[i] += alpha * x_im[i];
    }
}

inline void qlocal_axpy_complex_to_complex(double ar,
                                           double ai,
                                           const double* B4PACK_RESTRICT x_re,
                                           const double* B4PACK_RESTRICT x_im,
                                           double* B4PACK_RESTRICT y_re,
                                           double* B4PACK_RESTRICT y_im,
                                           std::size_t n) noexcept
{
#ifdef _OPENMP
#pragma omp simd
#endif
    for (std::size_t i = 0; i < n; ++i) {
        const double xr = x_re[i];
        const double xi = x_im[i];
        y_re[i] += ar * xr - ai * xi;
        y_im[i] += ar * xi + ai * xr;
    }
}

inline b4utils::Complex64 qlocal_dot_complex_no_conj(
    const double* B4PACK_RESTRICT a_re,
    const double* B4PACK_RESTRICT a_im,
    const double* B4PACK_RESTRICT b_re,
    const double* B4PACK_RESTRICT b_im,
    std::size_t n) noexcept
{
    double sr = 0.0;
    double si = 0.0;
#ifdef _OPENMP
#pragma omp simd reduction(+:sr,si)
#endif
    for (std::size_t i = 0; i < n; ++i) {
        const double ar = a_re[i];
        const double ai = a_im[i];
        const double br = b_re[i];
        const double bi = b_im[i];
        sr += ar * br - ai * bi;
        si += ar * bi + ai * br;
    }
    return {sr, si};
}

void fill_T1_values_packed_slice_coherent(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T1_support,
    std::size_t ix,
    b4utils::ComplexBuffer<>& T1_values,
    PackedContractionOptions opt = {});

void fill_T3_values_packed_slice_coherent(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T3_support,
    std::size_t ix,
    b4utils::ComplexBuffer<>& T3_values,
    PackedContractionOptions opt = {});

void build_T1_support_qlocal_slice_coherent_lane_into(
    const AmplitudeInputSupportPlan& common,
    const RangeBlockSupport1D& sjA,
    std::size_t ix,
    RangeBlockSupport1D& T1,
    PackedContractionOptions opt = {});

void build_T3_support_qlocal_slice_coherent_lane_into(
    const AmplitudeInputSupportPlan& common,
    const RangeBlockSupport1D& sjE,
    std::size_t ix,
    RangeBlockSupport1D& T3,
    PackedContractionOptions opt = {});

void fill_T1_values_packed_slice_coherent_lane(
    const AmplitudeInputSupportPlan& common,
    const RangeBlockSupport1D& sjA,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T1_support,
    std::size_t ix,
    b4utils::ComplexBuffer<>& T1_values,
    PackedContractionOptions opt = {});

void fill_T3_values_packed_slice_coherent_lane(
    const AmplitudeInputSupportPlan& common,
    const RangeBlockSupport1D& sjE,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T3_support,
    std::size_t ix,
    b4utils::ComplexBuffer<>& T3_values,
    PackedContractionOptions opt = {});

Range1D build_T2_panel_for_q_coherent(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T1_support,
    const b4utils::ComplexBuffer<>& T1_values,
    const PackedAmplitudeCoherentBoosters& B,
    std::size_t ik4_begin,
    std::size_t nk4,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* B4PACK_RESTRICT T2_re,
    double* B4PACK_RESTRICT T2_im,
    Range1D* B4PACK_RESTRICT T2_ranges,
    PackedQLocalThreadCounters* counters = nullptr);

Range1D build_T4_panel_for_q_coherent(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T3_support,
    const b4utils::ComplexBuffer<>& T3_values,
    const PackedAmplitudeCoherentBoosters& B,
    std::size_t ik3_begin,
    std::size_t nk3,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* B4PACK_RESTRICT T4_re,
    double* B4PACK_RESTRICT T4_im,
    Range1D* B4PACK_RESTRICT T4_ranges,
    PackedQLocalThreadCounters* counters = nullptr);

b4utils::Complex64 contract_q_block_project_C_times_T2_streamed_coherent(
    const AmplitudeInputSupportPlan& support,
    const double* C_base,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const b4utils::ComplexBuffer<>& T1_values,
    const RangeBlockSupport1D& T3_support,
    const b4utils::ComplexBuffer<>& T3_values,
    std::size_t ix,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    PackedQLocalComplexThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters = nullptr);

b4utils::Complex64 contract_q_block_project_Ct_times_T4_streamed_coherent(
    const AmplitudeInputSupportPlan& support,
    const double* C_base,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const b4utils::ComplexBuffer<>& T1_values,
    const RangeBlockSupport1D& T3_support,
    const b4utils::ComplexBuffer<>& T3_values,
    std::size_t ix,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    PackedQLocalComplexThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters = nullptr);

inline b4utils::Complex64 contract_q_block_streamed_adaptive_coherent(
    const AmplitudeInputSupportPlan& support,
    const double* C_base,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const b4utils::ComplexBuffer<>& T1_values,
    const RangeBlockSupport1D& T3_support,
    const b4utils::ComplexBuffer<>& T3_values,
    std::size_t ix,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    PackedQLocalComplexThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters = nullptr)
{
    const std::size_t L25 = support.l25.size();
    const std::size_t nk3 = static_cast<std::size_t>(rC.i_end - rC.i_begin);
    const std::size_t nk4 = static_cast<std::size_t>(rC.j_end - rC.j_begin);
    if (counters) {
        ++counters->active_q_blocks;
        counters->final_pairs += static_cast<std::uint64_t>(nk3) * static_cast<std::uint64_t>(nk4);
    }
    const QLocalFinalStrategy strategy = choose_qlocal_strategy(nk3, nk4, L25);
    if (strategy == QLocalFinalStrategy::ProjectCtTimesT4 || strategy == QLocalFinalStrategy::PairwiseStoreT2) {
        if (counters) ++counters->project_Ct_times_T4_blocks;
        return contract_q_block_project_Ct_times_T4_streamed_coherent(
            support, C_base, B, T1_support, T1_values, T3_support, T3_values,
            ix, il24, il34, il35, rC, ws, counters);
    }
    if (counters) ++counters->project_C_times_T2_blocks;
    return contract_q_block_project_C_times_T2_streamed_coherent(
        support, C_base, B, T1_support, T1_values, T3_support, T3_values,
        ix, il24, il34, il35, rC, ws, counters);
}

inline void materialize_logical_sjC_panel_half(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    std::size_t logical_block,
    const Range2D& logical,
    std::vector<double>& tmp)
{
    const std::size_t nk3 = static_cast<std::size_t>(logical.i_end - logical.i_begin);
    const std::size_t nk4 = static_cast<std::size_t>(logical.j_end - logical.j_begin);
    tmp.resize(nk3 * nk4);
    for (std::size_t jb = 0; jb < nk4; ++jb) {
        const std::size_t ik4 = static_cast<std::size_t>(logical.j_begin) + jb;
        for (std::size_t ib = 0; ib < nk3; ++ib) {
            const std::size_t ik3 = static_cast<std::size_t>(logical.i_begin) + ib;
            tmp[ib + nk3 * jb] = packed_load_sjC_half(support, sj.sjC.data(), logical_block, ik3, ik4);
        }
    }
}

b4utils::Complex64 contract_qlocal_streamed_slice_allq_half_sjC_coherent(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const b4utils::ComplexBuffer<>& T1_values,
    const RangeBlockSupport1D& T3_support,
    const b4utils::ComplexBuffer<>& T3_values,
    std::size_t ix,
    PackedContractionOptions opt = {},
    PackedQLocalContractionStats* stats = nullptr);

inline std::uint64_t xqslab_fixed_bytes_coherent(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const PackedAmplitudeCoherentBoosters& packed_B,
    const PackedAmplitudeXQSlabOptions& opt,
    const PackedContractionOptions& contraction_options)
{
    const std::uint64_t ram = xqslab_hard_memory_capacity_bytes();
    const std::uint64_t safety = opt.safety_margin_bytes != 0
        ? opt.safety_margin_bytes
        : xqslab_default_safety_margin_bytes(ram);
    std::uint64_t fixed = 0;
    fixed = xqslab_sat_add_u64(fixed, packed_B.payload_bytes());
    fixed = xqslab_sat_add_u64(fixed, opt.output_bytes == sizeof(double) ? sizeof(b4utils::Complex64) : opt.output_bytes);
    fixed = xqslab_sat_add_u64(fixed, static_support.total_metadata_bytes());
    fixed = xqslab_sat_add_u64(fixed, safety);
    fixed = xqslab_sat_add_u64(fixed, opt.extra_fixed_bytes);
    fixed = xqslab_sat_add_u64(fixed, opt.qlocal_thread_scratch_bytes);
    std::uint64_t workers = 1;
#ifdef _OPENMP
    if (contraction_options.use_openmp) {
        workers = static_cast<std::uint64_t>(std::max(1, omp_get_max_threads()));
    }
#endif
    const std::uint64_t logical_panel_values = xqslab_sat_mul_u64(
        static_cast<std::uint64_t>(static_support.channels.k3.size()),
        static_cast<std::uint64_t>(static_support.channels.k4.size()));
    fixed = xqslab_sat_add_u64(fixed, xqslab_sat_mul_u64(
        workers, xqslab_bytes_for_values(logical_panel_values)));
    return fixed;
}

std::vector<PackedAmplitudeQSlab> choose_q_slabs_for_xslice_half_sjC_coherent(
    const AmplitudeInputSupportPlan_HalfSjC& xslice_full_support,
    const AmplitudeInputSupportPlan_HalfSjC& wing_support,
    const RangeBlockSupport1D& T1,
    const RangeBlockSupport1D& T3,
    const PackedAmplitudeXQSlabOptions& opt,
    std::uint64_t fixed_bytes);

struct PackedAmplitudeCoherentXQResult {
    b4utils::Complex64 amplitude{0.0, 0.0};
    PackedAmplitudeXQTimings timings;
    PackedQLocalContractionStats stats;
    std::size_t x_slices = 0;
    std::size_t q_slabs_total = 0;
    std::size_t unfitting_q_slabs = 0;
    std::uint64_t max_estimated_peak_bytes = 0;

    void print(std::ostream& os) const {
        os << "\nPacked coherent x+q-slab result\n";
        os << "================================\n";
        os << "amplitude = (" << std::scientific << std::setprecision(17)
           << amplitude.re << ", " << amplitude.im << ")\n";
        os << std::fixed << std::setprecision(6);
        os << "x slices                = " << x_slices << "\n";
        os << "q slabs total           = " << q_slabs_total << "\n";
        os << "unfitting q slabs       = " << unfitting_q_slabs << "\n";
        os << "max estimated peak      = "; xqslab_print_bytes(os, max_estimated_peak_bytes); os << "\n";
        os << "total time              = " << timings.total << " s\n";
        os << "accounted time          = " << timings.accounted() << " s\n";
        os << "unaccounted time        = " << timings.unaccounted() << " s\n";
    }
};

PackedAmplitudeCoherentXQResult contract_packed_amplitude_coherent_xqslab_streamed_generated_support(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const PackedAmplitudeCoherentBoosters& packed_B,
    const std::array<int, 10>& spins,
    const PackedAmplitudeXQSlabOptions& opt = {},
    WingExecutionOrder order = WingExecutionOrder::LeftThenRight,
    PackedContractionOptions contraction_options = {},
    std::ostream* log = nullptr);

// ============================================================================
// C1/i1 driver
// ============================================================================


inline bool b4_contains_int_sorted_local(const std::vector<int>& v, int x) noexcept {
    return std::binary_search(v.begin(), v.end(), x);
}

inline int b4_pick_first_valid_or_throw_local(const std::vector<int>& v,
                                              const char* name)
{
    if (v.empty()) {
        throw b4utils::InternalFailure(
            std::string("coherent C1 contraction: no valid ") + name +
            " intertwiner");
    }
    return v.front();
}

inline int b4_pick_common_or_first_local(const std::vector<int>& lhs,
                                         const std::vector<int>& rhs,
                                         const char* lhs_name,
                                         const char* rhs_name)
{
    if (lhs.empty()) {
        throw b4utils::InternalFailure(
            std::string("coherent C1 contraction: no valid ") + lhs_name +
            " intertwiner");
    }
    if (rhs.empty()) {
        throw b4utils::InternalFailure(
            std::string("coherent C1 contraction: no valid ") + rhs_name +
            " intertwiner");
    }
    for (int x : lhs) {
        if (b4_contains_int_sorted_local(rhs, x)) return x;
    }
    return lhs.front();
}

// The coherent contraction sums over i1 explicitly.  The remaining i2..i5 are
// not physical fixed boundary labels anymore because B2..B5 have already been
// projected to the coherent basis.  Unfortunately the legacy fixed-a support
// builder still wants a 5-tuple of boundarys, mostly to validate channels and to
// feed the reflection-compatibility guard.  Pick harmless valid placeholders
// internally instead of asking the caller for fake fixed intertwiners.
std::array<int, 5> make_coherent_i1_support_template_boundarys(
    const B4Spins& spins,
    int Di1);

struct PackedCoherentI1BatchOptions {
    PackedAmplitudeXQSlabOptions xq;
    bool print_plan = true;
    bool print_i1_timing = true;
    bool skip_zero_c1 = true;
    bool require_same_booster_support = true;
    void* progress_context = nullptr;
    void (*progress_advance)(void*, std::uint64_t) noexcept = nullptr;

    // The shared dispatcher supplies exact completed-work units keyed by the
    // physical i1 value.  The pointers are borrowed for this synchronous call.
    // Standalone users may omit them and fall back to that i1's x extent.
    const std::vector<int>* progress_i1_values = nullptr;
    const std::vector<std::uint64_t>* progress_units_by_i1 = nullptr;
};

struct PackedCoherentI1BatchResult {
    b4utils::Complex64 amplitude{0.0, 0.0};
    std::size_t active_i1 = 0;
    std::uint64_t max_estimated_peak_bytes = 0;
    double total_seconds = 0.0;
    double support_seconds = 0.0;
    double sj_fill_seconds = 0.0;
    double sj_contract_seconds = 0.0;

    void print(std::ostream& os) const {
        os << "\nPacked coherent C1/i1 streamed result\n";
        os << "====================================\n";
        os << "active i1        = " << active_i1 << "\n";
        os << "amplitude        = (" << std::scientific << std::setprecision(17)
           << amplitude.re << ", " << amplitude.im << ")\n";
        os << std::fixed << std::setprecision(6)
           << "total time       = " << total_seconds << " s\n"
           << "support time     = " << support_seconds << " s\n"
           << "sj fill time     = " << sj_fill_seconds << " s\n"
           << "sj/contract time = " << sj_contract_seconds << " s\n"
           << "max peak estimate= ";
        xqslab_print_bytes(os, max_estimated_peak_bytes);
        os << "\n";
    }
};

PackedCoherentI1BatchResult contract_packed_amplitude_coherent_C1_streamed_xqslab(
    const B4Spins& spins,
    int Dl,
    const std::array<int, 10>& spins_array,
    const PackedAmplitudeCoherentBoosters& packed_B,
    const B4CoherentCoefficientsView& C1,
    const PackedCoherentI1BatchOptions& opt = {},
    WingExecutionOrder order = WingExecutionOrder::LeftThenRight,
    PackedContractionOptions contraction_options = {},
    std::ostream* log = nullptr);

// Backward-compatible overload: the fixed_intertwiners argument is ignored on purpose.
inline PackedCoherentI1BatchResult contract_packed_amplitude_coherent_C1_streamed_xqslab(
    const B4Spins& spins,
    int Dl,
    const std::array<int, 10>& spins_array,
    std::array<int, 5>,
    const PackedAmplitudeCoherentBoosters& packed_B,
    const B4CoherentCoefficientsView& C1,
    const PackedCoherentI1BatchOptions& opt = {},
    WingExecutionOrder order = WingExecutionOrder::LeftThenRight,
    PackedContractionOptions contraction_options = {},
    std::ostream* log = nullptr)
{
    return contract_packed_amplitude_coherent_C1_streamed_xqslab(
        spins, Dl, spins_array, packed_B, C1, opt, order, contraction_options, log);
}

inline PackedCoherentI1BatchResult contract_packed_amplitude_coherent_C1_streamed_xqslab(
    const B4Spins& spins,
    int Dl,
    const std::array<int, 10>& spins_array,
    const PackedAmplitudeCoherentBoosters& packed_B,
    const std::vector<int>& C1_iotas,
    const std::vector<b4utils::Complex64>& C1_coeffs,
    const PackedCoherentI1BatchOptions& opt = {},
    WingExecutionOrder order = WingExecutionOrder::LeftThenRight,
    PackedContractionOptions contraction_options = {},
    std::ostream* log = nullptr)
{
    return contract_packed_amplitude_coherent_C1_streamed_xqslab(
        spins, Dl, spins_array, packed_B,
        make_b4_coherent_coefficients_view(C1_iotas, C1_coeffs),
        opt, order, contraction_options, log);
}

// Backward-compatible vector overload: the fixed_intertwiners argument is ignored on purpose.
inline PackedCoherentI1BatchResult contract_packed_amplitude_coherent_C1_streamed_xqslab(
    const B4Spins& spins,
    int Dl,
    const std::array<int, 10>& spins_array,
    std::array<int, 5>,
    const PackedAmplitudeCoherentBoosters& packed_B,
    const std::vector<int>& C1_iotas,
    const std::vector<b4utils::Complex64>& C1_coeffs,
    const PackedCoherentI1BatchOptions& opt = {},
    WingExecutionOrder order = WingExecutionOrder::LeftThenRight,
    PackedContractionOptions contraction_options = {},
    std::ostream* log = nullptr)
{
    return contract_packed_amplitude_coherent_C1_streamed_xqslab(
        spins, Dl, spins_array, packed_B, C1_iotas, C1_coeffs,
        opt, order, contraction_options, log);
}

struct CoherentC1InputSupportPlan_HalfSjC {
    CoherentAmplitudeChannels coherent_channels;
    AmplitudeInputSupportPlan_HalfSjC common;
    RangeBlockSupport1D sjA_all; // block (x,l25,i1) -> k5
    RangeBlockSupport1D sjE_all; // block (x,l25,i1) -> k2

    // Dense bit mask: i1_x_valid[ia * X + ix] tells whether both A and E
    // wings have non-empty support for this (i1,x).  The global x list is a
    // union over i1; this mask keeps the contraction from dutifully doing
    // empty work.  A touching display of machines requiring kindergarten-level
    // reminders.
    std::vector<std::uint8_t> i1_x_valid;
    std::size_t i1_x_valid_pairs = 0;

    std::uint64_t total_payload_bytes() const noexcept {
        return common.sjB.payload_bytes()
             + common.sjC.payload_bytes()
             + common.sjD.payload_bytes()
             + sjA_all.payload_bytes()
             + sjE_all.payload_bytes()
             + common.B2.payload_bytes()
             + common.B3.payload_bytes()
             + common.B4.payload_bytes()
             + common.B5.payload_bytes();
    }

    std::uint64_t total_metadata_bytes() const noexcept {
        return common.total_metadata_bytes()
             + sjA_all.metadata_bytes()
             + sjE_all.metadata_bytes();
    }
};

inline AmplitudeChannels make_base_channels_from_coherent_C1_channels(
    const CoherentAmplitudeChannels& ch)
{
    AmplitudeChannels out;
    out.i = { ch.i1.empty() ? 0 : ch.i1.front(), 0, 0, 0, 0 };
    out.k2 = ch.k2;
    out.k3 = ch.k3;
    out.k4 = ch.k4;
    out.k5 = ch.k5;
    out.xvals = ch.xvals;
    return out;
}

CoherentC1InputSupportPlan_HalfSjC build_coherent_C1_input_support_plan_half_sjC(
    const B4Spins& s,
    int Dl,
    const CoherentAmplitudeChannels& ch);

inline bool coherent_C1_i1_x_valid(
    const CoherentC1InputSupportPlan_HalfSjC& support,
    std::size_t ia,
    std::size_t ix) noexcept
{
    const std::size_t X = support.coherent_channels.xvals.size();
    return ia < support.coherent_channels.i1.size() &&
           ix < X &&
           support.i1_x_valid.size() == support.coherent_channels.i1.size() * X &&
           support.i1_x_valid[ia * X + ix] != 0;
}

inline std::uint64_t coherent_C1_valid_i1_x_pairs_for_x(
    const CoherentC1InputSupportPlan_HalfSjC& support,
    std::size_t ix) noexcept
{
    const std::size_t I1 = support.coherent_channels.i1.size();
    std::uint64_t pairs = 0;
    for (std::size_t ia = 0; ia < I1; ++ia) {
        pairs += coherent_C1_i1_x_valid(support, ia, ix) ? 1u : 0u;
    }
    return pairs;
}

inline void set_coherent_C1_fixed_i1_wing_support(
    const CoherentC1InputSupportPlan_HalfSjC& src,
    std::size_t ii1,
    RangeBlockSupport1D& sjA,
    RangeBlockSupport1D& sjE)
{
    const std::size_t X   = src.coherent_channels.xvals.size();
    const std::size_t L25 = src.common.l25.size();
    const std::size_t I1  = src.coherent_channels.i1.size();

    sjA.name = "sjA fixed-i1 shallow view into coherent all-i1 payload";
    sjA.outer_dims = {X, L25};
    sjA.inner_dim = src.coherent_channels.k5.size();
    sjA.blocks.assign(X * L25, Range1D{});

    sjE.name = "sjE fixed-i1 shallow view into coherent all-i1 payload";
    sjE.outer_dims = {X, L25};
    sjE.inner_dim = src.coherent_channels.k2.size();
    sjE.blocks.assign(X * L25, Range1D{});

    for (std::size_t ix = 0; ix < X; ++ix) {
        for (std::size_t il25 = 0; il25 < L25; ++il25) {
            const std::size_t dst = idx2(ix, il25, L25);
            const std::size_t all = idx3(ix, il25, ii1, L25, I1);
            sjA.blocks[dst] = src.sjA_all.blocks[all];
            sjE.blocks[dst] = src.sjE_all.blocks[all];
        }
    }

    sjA.packed_values = src.sjA_all.packed_values;
    sjE.packed_values = src.sjE_all.packed_values;
}

inline void initialize_qlocal_support_workspace_from_common(
    const AmplitudeInputSupportPlan& common,
    PackedQLocalSupportWorkspace& sws)
{
    sws.initialize(common);
}

void fill_packed_coherent_C1_sjA_sjE_all_half_sjC(
    const B4Spins& s,
    const CoherentC1InputSupportPlan_HalfSjC& support,
    PackedAmplitudeSJTensors_HalfSjC& out);

inline void fill_packed_coherent_C1_sj_tensors_rec_half_sjC(
    const B4Spins& s,
    const CoherentC1InputSupportPlan_HalfSjC& support,
    PackedAmplitudeSJTensors_HalfSjC& out)
{
    // Fill shared sjB/sjC/sjD once using the existing filler.  Its dummy sjA/E
    // work is tiny and will be overwritten by all-i1 payloads below.
    fill_packed_amplitude_sj_tensors_rec_half_sjC(s, support.common, out);
    fill_packed_coherent_C1_sjA_sjE_all_half_sjC(s, support, out);
}

struct PackedCoherentC1SharedOptions {
    PackedAmplitudeXQSlabOptions xq;
    bool print_plan = true;
    bool print_i1_timing = true;
    bool print_memory = true;

    // Extra RSS/procfs diagnostics. Only printed when a log stream is supplied.
    // rss_debug_first_x limits the noisy per-x T/q-loop probes inside each slab.
    bool print_rss = true;
    std::size_t rss_debug_first_x = 3;
    bool skip_zero_c1 = true;

    // Approximate coherent C1 lane pruning.  Exact default is disabled.
    // A lane is dropped before support construction if
    //   |C1[i1]| < C1_prune_absolute_tol
    // or
    //   |C1[i1]| < C1_prune_relative_tol * max_a |C1[a]|.
    // This is coefficient-only pruning, not a rigorous amplitude error bound,
    // because A(i1) is not bounded here.  Use it for controlled scans and
    // validate thresholds against an exact run.
    double C1_prune_relative_tol = 0.0;
    double C1_prune_absolute_tol = 0.0;

    // If pruning would remove every nonzero coefficient, keep the largest one
    // instead of returning a silently zero amplitude.  Set false only if you
    // deliberately want thresholding to be allowed to erase everything.
    bool C1_prune_keep_largest_if_empty = true;

    bool print_C1_prune_summary = true;

    bool enable_full_x = true;
    bool enable_x_slab = true;

    // Emergency memory fallback for the shared coherent backend.  This is only
    // entered when full-x is too large and even one x-slice still exceeds the
    // memory budget.  Small cases therefore keep the fast resident/shared path.
    bool enable_xq_slab = true;

    // Memory-safe shared backend for large cases: keep sjB/sjC/sjD and the
    // x-local q skeleton shared, but stream i1 tiles instead of storing every
    // lane's T1/T3 RangeBlockSupport metadata at once.  Small cases still use
    // the faster resident all-lane path whenever the resident lane-workspace
    // estimate fits the memory budget.  In other words, speed when safe,
    // survival when not. Astonishing that this needed spelling out.
    bool enable_streamed_i1_tiles = true;

    // Use this fraction of the remaining memory budget as an additional soft
    // guard for resident all-lane T-support metadata.  1.0 means only reject
    // when the explicit budget is exceeded.  Values below 1 leave headroom for
    // allocator arenas, OpenMP stacks, and other tiny gremlins.
    double resident_lane_workspace_budget_fraction = 0.90;

    bool print_streamed_i1_summary = true;

    // Print one-line diagnostics for the emergency shared x+q fallback.
    bool print_xq_slab_summary = true;

    // Number of active i1 values contracted per q-local traversal.
    // If auto_i1_tile_size is true, this is only a user hint/diagnostic and the
    // contraction chooses a data-driven value from the support dimensions,
    // active C1 lanes, and per-thread scratch estimate.  Set
    // auto_i1_tile_size=false to force this value manually.  A value of 0 also
    // means auto, because making users tune this by hand is how CPUs become
    // resentful.
    std::size_t i1_tile_size = 0;
    bool auto_i1_tile_size = true;

    // Upper bound for the automatically selected tile.  0 means no explicit
    // cap beyond active i1 count and scratch memory.  Useful when validating
    // numerical drift against smaller tiles.
    std::size_t i1_tile_hard_max = 0;

    // Soft total scratch budget for q-loop/T-builder tile temporaries, across
    // OpenMP threads.  This is not the full algorithm memory budget; it is only
    // a hard guard against pathological tile choices.
    std::uint64_t i1_tile_scratch_budget_bytes = 64ull * 1024ull * 1024ull;

    // Soft per-thread cache budget for the auto tile selector.  0 lets the
    // selector choose a conservative cache-sized budget.  This matters because
    // the scalar q path reuses one scratch workspace per thread; the limiting
    // temporary is now mostly the tiled T1/T3 builder scratch rather than
    // capacity-limited main memory.
    std::uint64_t i1_tile_cache_budget_bytes = 0;

    // Preferred scalar backend for coherent C1: after T1/T3 values are built
    // for all tiles at a fixed x, flatten the independent (active i1 lane, q)
    // work into one OpenMP region.  This preserves the scalar build-and-consume
    // locality of tile=1 while avoiding one parallel region per i1 tile.
    // This avoids multi-lane T2/T4 buffers and repeated OpenMP regions.
    // Dynamic chunk for the flattened (lane,q) loop.  0 means automatic.
    // Small values improve balance; too small values just donate time to the
    // OpenMP runtime, which already has enough of our money.
    std::size_t flat_q_task_chunk = 0;

    // Accumulate the coherent C1 weight inside the q-loop when per-i1 timing is
    // not requested.  This avoids carrying per-lane final amplitudes and makes
    // the contraction structurally compute the coherent amplitude directly.
    // The heavy T2/T4 work is still exact per lane; SIMD may have its turn
    // later, if it behaves.
    bool accumulate_C1_inside_q = true;

    // Print density of the exact (i1,x) support mask.
    bool print_x_mask = true;

    std::size_t max_x_per_slab = std::numeric_limits<std::size_t>::max();

    // Automatic x-slab planner.  When max_x_per_slab is left at its default,
    // the dispatcher first probes one x-slice at a time, estimates whether the
    // full-X resident path is safe without allocating full-X support, then
    // greedily packs the largest x-slabs that fit the memory budget.
    bool auto_x_slab_size = true;

    // Use only this fraction of the remaining variable memory budget when
    // packing x-slabs from one-x estimates.  The actual slab is still checked
    // after construction; this is just the first guess.  Keep below 1 because
    // metadata and T workspaces enjoy lying by omission.
    double auto_x_slab_budget_fraction = 0.85;

    // If true, do not build the full-X support until a safe one-x preflight
    // estimate says it should fit.  This is the point of the whole circus.
    bool preflight_full_x_without_full_support = true;

    // Optional progress bridge owned by the public API layer.  The numerical
    // dispatcher discovers the exact number of active (i1,x) pairs only after
    // pruning C1 and building the coherent channels, so it reports that total
    // through begin before any advances.  Callbacks are noexcept: progress is
    // informational and must never change a numerical result.
    void* progress_context = nullptr;
    void (*progress_begin)(void*, std::uint64_t) noexcept = nullptr;
    void (*progress_advance)(void*, std::uint64_t) noexcept = nullptr;
};

inline bool coherent_C1_progress_enabled(
    const PackedCoherentC1SharedOptions& opt) noexcept
{
    return opt.progress_begin != nullptr && opt.progress_advance != nullptr;
}

inline void coherent_C1_progress_advance(
    const PackedCoherentC1SharedOptions& opt,
    std::uint64_t completed_units) noexcept
{
    if (completed_units != 0 && coherent_C1_progress_enabled(opt)) {
        opt.progress_advance(opt.progress_context, completed_units);
    }
}

struct PackedCoherentC1SharedResult {
    b4utils::Complex64 amplitude{0.0, 0.0};
    std::size_t active_i1 = 0;
    std::size_t C1_original_count = 0;
    std::size_t C1_kept_count = 0;
    std::size_t C1_pruned_count = 0;
    double C1_max_abs = 0.0;
    double C1_l1_total = 0.0;
    double C1_l1_dropped = 0.0;
    double C1_l1_dropped_fraction = 0.0;
    double C1_prune_relative_tol_used = 0.0;
    double C1_prune_absolute_tol_used = 0.0;
    bool C1_pruning_enabled = false;
    bool C1_prune_kept_largest_fallback = false;
    std::size_t full_x_slabs = 0;
    std::size_t x_slabs = 0;
    std::size_t xq_slices = 0;
    std::size_t xq_q_slabs_total = 0;
    std::size_t xq_unfitting_q_slabs = 0;
    std::uint64_t xq_max_estimated_peak_bytes = 0;
    std::size_t generic_allq_slabs = 0;
    std::size_t i1_tiles = 0;
    std::size_t selected_i1_tile_size = 0;
    std::size_t requested_i1_tile_size = 0;
    bool i1_tile_auto_selected = false;
    bool c1_accumulated_inside_q = false;
    bool used_flat_scalar_q_loop = false;
    std::uint64_t estimated_peak_bytes = 0;
    std::uint64_t incremental_bytes = 0;
    double total_seconds = 0.0;
    double support_seconds = 0.0;
    double sj_fill_seconds = 0.0;
    double contract_seconds = 0.0;

    // Detailed contraction phase timing.  These are coarse enough not to turn
    // the profiler into the benchmark, unlike timing every microscopic dot.
    double lane_setup_seconds = 0.0;
    double T_build_seconds = 0.0;
    double q_loop_seconds = 0.0;
    double q_skeleton_seconds = 0.0;
    double c1_accum_seconds = 0.0;
    std::uint64_t q_block_kernel_calls = 0;
    std::uint64_t q_block_lane_calls = 0;
    std::uint64_t q_skeletons_built = 0;
    std::uint64_t q_B3_row_descs = 0;
    std::uint64_t q_B4_row_descs = 0;
    std::uint64_t i1_x_pairs_total = 0;
    std::uint64_t i1_x_pairs_valid = 0;
    std::uint64_t i1_x_pairs_skipped = 0;
    std::uint64_t i1_x_pairs_processed = 0;

    void print(std::ostream& os) const {
        os << "\nPacked coherent C1 shared-sj result\n";
        os << "===================================\n";
        os << "active i1        = " << active_i1 << "\n";
        if (C1_pruning_enabled || C1_original_count != C1_kept_count) {
            os << "C1 prune         = kept " << C1_kept_count << "/" << C1_original_count
               << " dropped " << C1_pruned_count
               << " rel_tol=" << std::scientific << std::setprecision(3) << C1_prune_relative_tol_used
               << " abs_tol=" << C1_prune_absolute_tol_used
               << " dropped_l1=" << C1_l1_dropped_fraction
               << (C1_prune_kept_largest_fallback ? " (kept largest fallback)" : "")
               << std::fixed << std::setprecision(6) << "\n";
        }
        os << "full-x slabs     = " << full_x_slabs << "\n";
        os << "x slabs          = " << x_slabs << "\n";
        os << "xq slices        = " << xq_slices << "\n";
        os << "xq q slabs       = " << xq_q_slabs_total << "\n";
        os << "xq unfitting     = " << xq_unfitting_q_slabs << "\n";
        if (xq_slices != 0) {
            os << "xq max peak      = "; xqslab_print_bytes(os, xq_max_estimated_peak_bytes); os << "\n";
        }
        os << "i1 tiles         = " << i1_tiles << "\n";
        os << "selected i1 tile = " << selected_i1_tile_size
           << (i1_tile_auto_selected ? " (auto)" : " (manual)") << "\n";
        if (requested_i1_tile_size != 0 && requested_i1_tile_size != selected_i1_tile_size) {
            os << "requested i1 tile= " << requested_i1_tile_size << "\n";
        }
        os << "C1 q-accumulate  = " << (c1_accumulated_inside_q ? "inside q-loop" : "final lane sum") << "\n";
        os << "flat scalar q    = " << (used_flat_scalar_q_loop ? "enabled" : "disabled") << "\n";
        os << "dense C project  = enabled (no c==0 branch)\n";
        os << "generic slabs    = " << generic_allq_slabs << "\n";
        os << "amplitude        = (" << std::scientific << std::setprecision(17)
           << amplitude.re << ", " << amplitude.im << ")\n";
        os << std::fixed << std::setprecision(6)
           << "total time       = " << total_seconds << " s\n"
           << "support time     = " << support_seconds << " s\n"
           << "sj fill time     = " << sj_fill_seconds << " s\n"
           << "contract time    = " << contract_seconds << " s\n"
           << "  lane setup      = " << lane_setup_seconds << " s\n"
           << "  T1/T3 build     = " << T_build_seconds << " s\n"
           << "  q-loop kernels  = " << q_loop_seconds << " s\n"
           << "  q skeletons     = " << q_skeleton_seconds << " s\n"
           << "  C1 accumulate   = " << c1_accum_seconds << " s\n"
           << "q-block calls     = " << q_block_kernel_calls << "\n"
           << "q-block lane calls= " << q_block_lane_calls << "\n"
           << "q skeletons built = " << q_skeletons_built << "\n"
           << "q B3 row descs    = " << q_B3_row_descs << "\n"
           << "q B4 row descs    = " << q_B4_row_descs << "\n"
           << "i1-x pairs total  = " << i1_x_pairs_total << "\n"
           << "i1-x pairs valid  = " << i1_x_pairs_valid << "\n"
           << "i1-x pairs skipped= " << i1_x_pairs_skipped << "\n"
           << "i1-x pairs processed= " << i1_x_pairs_processed << "\n";
        os << "incremental mem  = "; xqslab_print_bytes(os, incremental_bytes); os << "\n";
        os << "peak estimate    = "; xqslab_print_bytes(os, estimated_peak_bytes); os << "\n";
    }
};

inline std::uint64_t coherent_C1_shared_fixed_bytes(
    const PackedAmplitudeCoherentBoosters& packed_B,
    const PackedAmplitudeXQSlabOptions& opt)
{
    const std::uint64_t ram = xqslab_hard_memory_capacity_bytes();
    const std::uint64_t safety = opt.safety_margin_bytes != 0
        ? opt.safety_margin_bytes
        : xqslab_default_safety_margin_bytes(ram);
    std::uint64_t fixed = 0;
    fixed = xqslab_sat_add_u64(fixed, packed_B.payload_bytes());
    fixed = xqslab_sat_add_u64(fixed, opt.output_bytes == 0 ? sizeof(b4utils::Complex64) : opt.output_bytes);
    fixed = xqslab_sat_add_u64(fixed, opt.extra_fixed_bytes);
    fixed = xqslab_sat_add_u64(fixed, opt.qlocal_thread_scratch_bytes);
    fixed = xqslab_sat_add_u64(fixed, safety);
    return fixed;
}

inline std::uint64_t coherent_C1_shared_incremental_bytes(
    const CoherentC1InputSupportPlan_HalfSjC& support)
{
    std::uint64_t inc = 0;
    inc = xqslab_sat_add_u64(inc, support.common.sjB.payload_bytes());
    inc = xqslab_sat_add_u64(inc, support.common.sjC.payload_bytes());
    inc = xqslab_sat_add_u64(inc, support.common.sjD.payload_bytes());
    inc = xqslab_sat_add_u64(inc, support.sjA_all.payload_bytes());
    inc = xqslab_sat_add_u64(inc, support.sjE_all.payload_bytes());
    inc = xqslab_sat_add_u64(inc, support.total_metadata_bytes());
    return inc;
}


inline void coherent_C1_print_support_memory_breakdown(
    std::ostream* log,
    const char* tag,
    const CoherentC1InputSupportPlan_HalfSjC& support)
{
    if (!log) return;
    *log << "[coherent C1 support memory] " << tag << "\n";
    *log << "  common.sjA payload = "; xqslab_print_bytes(*log, support.common.sjA.payload_bytes());
    *log << " metadata = "; xqslab_print_bytes(*log, support.common.sjA.metadata_bytes()); *log << "\n";
    *log << "  common.sjB payload = "; xqslab_print_bytes(*log, support.common.sjB.payload_bytes());
    *log << " metadata = "; xqslab_print_bytes(*log, support.common.sjB.metadata_bytes()); *log << "\n";
    *log << "  common.sjC payload = "; xqslab_print_bytes(*log, support.common.sjC.payload_bytes());
    *log << " metadata = "; xqslab_print_bytes(*log, support.common.sjC.metadata_bytes()); *log << "\n";
    *log << "  common.sjD payload = "; xqslab_print_bytes(*log, support.common.sjD.payload_bytes());
    *log << " metadata = "; xqslab_print_bytes(*log, support.common.sjD.metadata_bytes()); *log << "\n";
    *log << "  common.sjE payload = "; xqslab_print_bytes(*log, support.common.sjE.payload_bytes());
    *log << " metadata = "; xqslab_print_bytes(*log, support.common.sjE.metadata_bytes()); *log << "\n";
    *log << "  sjA_all    payload = "; xqslab_print_bytes(*log, support.sjA_all.payload_bytes());
    *log << " metadata = "; xqslab_print_bytes(*log, support.sjA_all.metadata_bytes()); *log << "\n";
    *log << "  sjE_all    payload = "; xqslab_print_bytes(*log, support.sjE_all.payload_bytes());
    *log << " metadata = "; xqslab_print_bytes(*log, support.sjE_all.metadata_bytes()); *log << "\n";
    *log << "  total metadata     = "; xqslab_print_bytes(*log, support.total_metadata_bytes()); *log << "\n";
    *log << "  model incremental  = "; xqslab_print_bytes(*log, coherent_C1_shared_incremental_bytes(support)); *log << "\n";
}


inline CoherentAmplitudeChannels coherent_C1_make_x_subchannels(
    const CoherentAmplitudeChannels& ch,
    std::size_t x_begin,
    std::size_t x_end)
{
    CoherentAmplitudeChannels sub = ch;
    if (x_begin > ch.xvals.size()) x_begin = ch.xvals.size();
    if (x_end > ch.xvals.size()) x_end = ch.xvals.size();
    if (x_begin > x_end) x_begin = x_end;
    sub.xvals.assign(ch.xvals.begin() + static_cast<std::ptrdiff_t>(x_begin),
                     ch.xvals.begin() + static_cast<std::ptrdiff_t>(x_end));
    return sub;
}


struct CoherentC1TileLaneState {
    std::size_t ia = 0;
    std::size_t cpos = 0;
    int Di1 = 0;
    b4utils::Complex64 coeff{0.0, 0.0};
    b4utils::Complex64 amplitude{0.0, 0.0};
    bool active_for_x = true;

    // High-performance coherent-C1 fixed-i1 lane view.
    //
    // Older versions stored an AmplitudeInputSupportPlan_HalfSjC by value here
    // and initialized it with
    //
    //     lane.view = support.common;
    //
    // That deep-copied all common RangeBlockSupport metadata for every resident
    // i1 lane.  In the lambda=44,Dl=12 stress case this was ~16 GiB of
    // duplicated metadata before the first T-build.  A fixed-i1 lane only needs
    // the two i1-dependent wings below; all other support is read from the
    // shared support.common passed to the kernels.
    RangeBlockSupport1D sjA;
    RangeBlockSupport1D sjE;

    PackedQLocalSupportWorkspace sws;
    PackedQLocalComplexValueWorkspace vws;
};



// --------------------------------------------------------------------------
// q-block skeletons shared across i1 lanes.
//
// These are x-local descriptors.  They move the q-grid discovery, C-block
// lookup, projection strategy choice, and B3/B4 row discovery out of the
// lane loop.  The numerical work is still exact per lane; the useful part is
// that all lanes consume the same skeleton instead of rediscovering the same
// q-block metadata like caffeinated amnesiacs.
// --------------------------------------------------------------------------

struct CoherentC1BoosterRowDesc {
    std::size_t local = 0;   // local row in the C/T panel
    std::size_t global = 0;  // global k3/k4 index
    Range1D range{};         // nonzero lifted-spin range in B3/B4
};

struct CoherentC1QBlockSkeleton {
    std::size_t ix = 0;
    std::size_t il24 = 0;
    std::size_t il34 = 0;
    std::size_t il35 = 0;
    std::size_t C_block = 0;
    Range2D rC{};
    std::size_t C_offset = 0;
    bool direct_C = true;
    QLocalFinalStrategy strategy = QLocalFinalStrategy::ProjectCTimesT2;
    std::vector<CoherentC1BoosterRowDesc> B4_rows;
    std::vector<CoherentC1BoosterRowDesc> B3_rows;
};

inline void coherent_C1_populate_qblock_booster_rows(
    const AmplitudeInputSupportPlan_HalfSjC& common,
    CoherentC1QBlockSkeleton& q)
{
    const std::size_t L24 = common.l24.size();
    const std::size_t L34 = common.l34.size();
    const std::size_t L35 = common.l35.size();

    q.B4_rows.clear();
    q.B3_rows.clear();

    const std::size_t nk4 = static_cast<std::size_t>(q.rC.j_end - q.rC.j_begin);
    const std::size_t nk3 = static_cast<std::size_t>(q.rC.i_end - q.rC.i_begin);
    q.B4_rows.reserve(nk4);
    q.B3_rows.reserve(nk3);

    for (std::size_t jb = 0; jb < nk4; ++jb) {
        const std::size_t ik4 = q.rC.j_begin + jb;
        const std::size_t B4_block = idx3(ik4, q.il24, q.il34, L24, L34);
        const Range1D& r = common.B4.blocks[B4_block];
        if (!r.empty()) q.B4_rows.push_back({jb, ik4, r});
    }

    for (std::size_t ib = 0; ib < nk3; ++ib) {
        const std::size_t ik3 = q.rC.i_begin + ib;
        const std::size_t B3_block = idx3(ik3, q.il34, q.il35, L34, L35);
        const Range1D& r = common.B3.blocks[B3_block];
        if (!r.empty()) q.B3_rows.push_back({ib, ik3, r});
    }
}

Range1D build_T2_panel_for_q_coherent_skeleton(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T1_support,
    const b4utils::ComplexBuffer<>& T1_values,
    const PackedAmplitudeCoherentBoosters& B,
    const CoherentC1QBlockSkeleton& q,
    double* B4PACK_RESTRICT T2_re,
    double* B4PACK_RESTRICT T2_im,
    Range1D* B4PACK_RESTRICT T2_ranges,
    PackedQLocalThreadCounters* counters = nullptr);

Range1D build_T4_panel_for_q_coherent_skeleton(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T3_support,
    const b4utils::ComplexBuffer<>& T3_values,
    const PackedAmplitudeCoherentBoosters& B,
    const CoherentC1QBlockSkeleton& q,
    double* B4PACK_RESTRICT T4_re,
    double* B4PACK_RESTRICT T4_im,
    Range1D* B4PACK_RESTRICT T4_ranges,
    PackedQLocalThreadCounters* counters = nullptr);

b4utils::Complex64 contract_q_block_C_times_T2_dense_coherent_Cbase_skeleton(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const b4utils::ComplexBuffer<>& T1_values,
    const RangeBlockSupport1D& T3_support,
    const b4utils::ComplexBuffer<>& T3_values,
    const CoherentC1QBlockSkeleton& q,
    const double* B4PACK_RESTRICT C_base,
    PackedQLocalComplexThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters = nullptr);

b4utils::Complex64 contract_q_block_Ct_times_T4_generic_coherent_Cbase_skeleton(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const b4utils::ComplexBuffer<>& T1_values,
    const RangeBlockSupport1D& T3_support,
    const b4utils::ComplexBuffer<>& T3_values,
    const CoherentC1QBlockSkeleton& q,
    const double* B4PACK_RESTRICT C_base,
    PackedQLocalComplexThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters = nullptr);

inline b4utils::Complex64 contract_q_block_streamed_adaptive_coherent_Cbase_skeleton(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeCoherentBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const b4utils::ComplexBuffer<>& T1_values,
    const RangeBlockSupport1D& T3_support,
    const b4utils::ComplexBuffer<>& T3_values,
    const CoherentC1QBlockSkeleton& q,
    const double* B4PACK_RESTRICT C_base,
    PackedQLocalComplexThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters = nullptr)
{
    (void)sj;

    const bool project_Ct_times_T4 =
        (q.strategy == QLocalFinalStrategy::ProjectCtTimesT4 ||
         q.strategy == QLocalFinalStrategy::PairwiseStoreT2);

    if (!project_Ct_times_T4) {
        return contract_q_block_C_times_T2_dense_coherent_Cbase_skeleton(
            support, B, T1_support, T1_values, T3_support, T3_values,
            q, C_base, ws, counters);
    }

    return contract_q_block_Ct_times_T4_generic_coherent_Cbase_skeleton(
        support, B, T1_support, T1_values, T3_support, T3_values,
        q, C_base, ws, counters);
}

inline void coherent_C1_build_tile_lanes(
    const CoherentC1InputSupportPlan_HalfSjC& support,
    const B4CoherentCoefficientsView& C1,
    std::size_t ia_begin,
    std::size_t ia_end,
    bool skip_zero_c1,
    std::vector<CoherentC1TileLaneState>& lanes,
    bool initialize_support_workspaces = true)
{
    lanes.clear();
    ia_end = std::min<std::size_t>(ia_end, support.coherent_channels.i1.size());
    if (ia_begin >= ia_end) return;
    lanes.reserve(ia_end - ia_begin);

    for (std::size_t ia = ia_begin; ia < ia_end; ++ia) {
        const std::size_t cpos = support.coherent_channels.c1_positions[ia];
        if (cpos >= C1.size()) continue;
        const b4utils::Complex64 c = C1.coefficients[cpos];
        if (skip_zero_c1 && c.re == 0.0 && c.im == 0.0) continue;

        CoherentC1TileLaneState lane;
        lane.ia = ia;
        lane.cpos = cpos;
        lane.Di1 = support.coherent_channels.i1[ia];
        lane.coeff = c;
        set_coherent_C1_fixed_i1_wing_support(support, ia, lane.sjA, lane.sjE);
        if (initialize_support_workspaces) {
            initialize_qlocal_support_workspace_from_common(support.common, lane.sws);
        }
        lanes.emplace_back(std::move(lane));
    }
}

void fill_T1_values_packed_slice_coherent_i1tile(
    const AmplitudeInputSupportPlan& common,
    std::vector<CoherentC1TileLaneState>& lanes,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeCoherentBoosters& B,
    std::size_t ix,
    PackedContractionOptions opt = {});

void fill_T3_values_packed_slice_coherent_i1tile(
    const AmplitudeInputSupportPlan& common,
    std::vector<CoherentC1TileLaneState>& lanes,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeCoherentBoosters& B,
    std::size_t ix,
    PackedContractionOptions opt = {});

std::size_t coherent_C1_tile_build_T_values_for_x(
    const CoherentC1InputSupportPlan_HalfSjC& support,
    std::vector<CoherentC1TileLaneState>& lanes,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeCoherentBoosters& B,
    std::size_t ix,
    WingExecutionOrder order,
    PackedContractionOptions opt);

inline std::uint64_t coherent_C1_per_lane_T_support_metadata_bytes(
    const CoherentC1InputSupportPlan_HalfSjC& support)
{
    const auto& ch = support.common.channels;
    const std::uint64_t L23 = static_cast<std::uint64_t>(support.common.l23.size());
    const std::uint64_t L24 = static_cast<std::uint64_t>(support.common.l24.size());
    const std::uint64_t L35 = static_cast<std::uint64_t>(support.common.l35.size());
    const std::uint64_t L45 = static_cast<std::uint64_t>(support.common.l45.size());
    const std::uint64_t K3d = static_cast<std::uint64_t>(ch.k3.size());
    const std::uint64_t K4d = static_cast<std::uint64_t>(ch.k4.size());

    const std::uint64_t T1_blocks = xqslab_sat_mul_u64(xqslab_sat_mul_u64(L45, K4d), L35);
    const std::uint64_t T3_blocks = xqslab_sat_mul_u64(xqslab_sat_mul_u64(L23, K3d), L24);
    return xqslab_sat_mul_u64(xqslab_sat_add_u64(T1_blocks, T3_blocks),
                              static_cast<std::uint64_t>(sizeof(Range1D)));
}

std::uint64_t coherent_C1_resident_lane_workspace_estimate_bytes(
    const CoherentC1InputSupportPlan_HalfSjC& support,
    std::size_t active_i1);

inline bool coherent_C1_resident_lane_workspace_fits(
    std::uint64_t base_peak,
    std::uint64_t resident_lane_bytes,
    std::uint64_t budget,
    double soft_fraction)
{
    const std::uint64_t resident_peak = xqslab_sat_add_u64(base_peak, resident_lane_bytes);
    if (resident_peak > budget) return false;
    if (!std::isfinite(soft_fraction) || soft_fraction <= 0.0 || soft_fraction >= 1.0) {
        return true;
    }
    const long double soft = static_cast<long double>(budget) * static_cast<long double>(soft_fraction);
    return static_cast<long double>(resident_peak) <= soft;
}

void coherent_C1_build_allq_skeletons_for_x(
    const AmplitudeInputSupportPlan_HalfSjC& common,
    std::size_t ix,
    std::vector<CoherentC1QBlockSkeleton>& qs);

void coherent_C1_contract_qlocal_skeletons_tile(
    const AmplitudeInputSupportPlan_HalfSjC& common,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeCoherentBoosters& B,
    std::vector<CoherentC1TileLaneState>& lanes,
    const std::vector<CoherentC1QBlockSkeleton>& qs,
    PackedContractionOptions opt,
    std::uint64_t* q_block_calls = nullptr,
    std::uint64_t* q_lane_calls = nullptr);




inline std::uint64_t coherent_C1_sysctl_u64(const char* name) noexcept
{
#if defined(__APPLE__)
    std::uint64_t value = 0;
    std::size_t size = sizeof(value);
    if (::sysctlbyname(name, &value, &size, nullptr, 0) == 0 && value != 0) return value;
#else
    (void)name;
#endif
    return 0;
}

inline std::uint64_t coherent_C1_sysconf_cache_bytes(int name) noexcept
{
#if defined(_SC_LEVEL1_DCACHE_SIZE)
    const long v = ::sysconf(name);
    if (v > 0) return static_cast<std::uint64_t>(v);
#else
    (void)name;
#endif
    return 0;
}

inline std::uint64_t coherent_C1_detect_L1D_cache_bytes() noexcept
{
#if defined(__APPLE__)
    if (const std::uint64_t v = coherent_C1_sysctl_u64("hw.l1dcachesize")) return v;
#endif
#if defined(_SC_LEVEL1_DCACHE_SIZE)
    if (const std::uint64_t v = coherent_C1_sysconf_cache_bytes(_SC_LEVEL1_DCACHE_SIZE)) return v;
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
    return 64ull * 1024ull;
#else
    return 48ull * 1024ull;
#endif
}

inline std::uint64_t coherent_C1_auto_i1_tile_cache_budget_bytes() noexcept
{
    const std::uint64_t l1d = coherent_C1_detect_L1D_cache_bytes();
    // The q-loop scalar path consumes one lane at a time, but the T1/T3 tiled
    // builders and metadata still scale with the tile.  A budget of about three
    // L1D caches usually lands on tile=4 for the 41x41x11 case while preventing
    // the tile=31 L2-squatting disaster.  Clamp the fallback so x86 and ARM do
    // not need separate hard-coded rituals.
    std::uint64_t budget = 3ull * std::max<std::uint64_t>(l1d, 32ull * 1024ull);
    budget = std::max<std::uint64_t>(budget, 128ull * 1024ull);
    budget = std::min<std::uint64_t>(budget, 256ull * 1024ull);
    return budget;
}

inline std::size_t coherent_C1_effective_threads_for_scratch(PackedContractionOptions opt) noexcept
{
#ifdef _OPENMP
    if (opt.use_openmp) {
        const int n = omp_get_max_threads();
        return n > 0 ? static_cast<std::size_t>(n) : 1u;
    }
#endif
    (void)opt;
    return 1u;
}

inline std::size_t coherent_C1_count_candidate_i1_lanes(
    const CoherentC1InputSupportPlan_HalfSjC& support,
    const B4CoherentCoefficientsView& C1,
    bool skip_zero_c1) noexcept
{
    std::size_t n = 0;
    const std::size_t I1 = support.coherent_channels.i1.size();
    for (std::size_t ia = 0; ia < I1; ++ia) {
        const std::size_t cpos = support.coherent_channels.c1_positions[ia];
        if (cpos >= C1.size()) continue;
        const b4utils::Complex64 c = C1.coefficients[cpos];
        if (skip_zero_c1 && c.re == 0.0 && c.im == 0.0) continue;
        ++n;
    }
    return n;
}

std::uint64_t coherent_C1_estimate_i1_tile_per_thread_scratch_bytes(
    const CoherentC1InputSupportPlan_HalfSjC& support,
    std::size_t tile) noexcept;
std::size_t coherent_C1_select_i1_tile_size(
    const CoherentC1InputSupportPlan_HalfSjC& support,
    const B4CoherentCoefficientsView& C1,
    const PackedCoherentC1SharedOptions& opt,
    PackedContractionOptions contraction_options);


b4utils::Complex64 coherent_C1_contract_qlocal_skeletons_flat_scalar_weighted(
    const AmplitudeInputSupportPlan_HalfSjC& common,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeCoherentBoosters& B,
    std::vector<std::vector<CoherentC1TileLaneState>>& tile_groups,
    const std::vector<CoherentC1QBlockSkeleton>& qs,
    PackedContractionOptions opt,
    std::size_t requested_chunk,
    std::uint64_t* q_block_calls = nullptr,
    std::uint64_t* q_lane_calls = nullptr);

b4utils::Complex64 coherent_C1_contract_qlocal_skeletons_tile_weighted(
    const AmplitudeInputSupportPlan_HalfSjC& common,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeCoherentBoosters& B,
    std::vector<CoherentC1TileLaneState>& lanes,
    const std::vector<CoherentC1QBlockSkeleton>& qs,
    PackedContractionOptions opt,
    std::uint64_t* q_block_calls = nullptr,
    std::uint64_t* q_lane_calls = nullptr);

inline void coherent_C1_print_sj_actual_buffers(std::ostream* log,
                                                 const char* tag,
                                                 const PackedAmplitudeSJTensors_HalfSjC& sj)
{
    if (!log) return;
    const auto print_one = [&](const char* name, const auto& b) {
        *log << "  " << name
             << " size=" << b.size()
             << " cap=" << b.capacity()
             << " bytes="; xqslab_print_bytes(*log, static_cast<std::uint64_t>(b.size() * sizeof(double)));
        *log << " cap_bytes="; xqslab_print_bytes(*log, static_cast<std::uint64_t>(b.capacity() * sizeof(double)));
        *log << "\n";
    };

    *log << "[coherent C1 actual sj buffers] " << tag << "\n";
    print_one("sjA", sj.sjA);
    print_one("sjB", sj.sjB);
    print_one("sjC", sj.sjC);
    print_one("sjD", sj.sjD);
    print_one("sjE", sj.sjE);
    *log << "  payload_bytes = "; xqslab_print_bytes(*log, sj.payload_bytes()); *log << "\n";
}

void coherent_C1_print_tile_group_memory(std::ostream* log,
                                                const char* tag,
                                                const std::vector<std::vector<CoherentC1TileLaneState>>& tile_groups);

inline std::uint64_t coherent_C1_q_skeleton_memory_model_bytes(
    const std::vector<CoherentC1QBlockSkeleton>& qs)
{
    std::uint64_t bytes = static_cast<std::uint64_t>(qs.capacity() * sizeof(CoherentC1QBlockSkeleton));
    for (const CoherentC1QBlockSkeleton& q : qs) {
        bytes = xqslab_sat_add_u64(bytes, static_cast<std::uint64_t>(q.B3_rows.capacity() * sizeof(CoherentC1BoosterRowDesc)));
        bytes = xqslab_sat_add_u64(bytes, static_cast<std::uint64_t>(q.B4_rows.capacity() * sizeof(CoherentC1BoosterRowDesc)));
    }
    return bytes;
}

inline void coherent_C1_print_q_skeleton_memory(std::ostream* log,
                                                const char* tag,
                                                const std::vector<CoherentC1QBlockSkeleton>& qs)
{
    if (!log) return;
    std::size_t k3_rows = 0;
    std::size_t k4_rows = 0;
    std::size_t k3_cap = 0;
    std::size_t k4_cap = 0;
    for (const CoherentC1QBlockSkeleton& q : qs) {
        k3_rows += q.B3_rows.size();
        k4_rows += q.B4_rows.size();
        k3_cap += q.B3_rows.capacity();
        k4_cap += q.B4_rows.capacity();
    }
    *log << "[coherent C1 q skeleton memory] " << tag
         << " q=" << qs.size() << " q_cap=" << qs.capacity()
         << " B3_rows=" << k3_rows << " B3_cap=" << k3_cap
         << " B4_rows=" << k4_rows << " B4_cap=" << k4_cap
         << " model="; xqslab_print_bytes(*log, coherent_C1_q_skeleton_memory_model_bytes(qs));
    *log << "\n";
}


void coherent_C1_accumulate_shared_support_contraction_streamed_i1_tiles(
    const B4Spins& spins,
    const PackedAmplitudeCoherentBoosters& packed_B,
    const B4CoherentCoefficientsView& C1,
    const CoherentC1InputSupportPlan_HalfSjC& support,
    const PackedCoherentC1SharedOptions& opt,
    WingExecutionOrder order,
    PackedContractionOptions contraction_options,
    PackedCoherentC1SharedResult& result,
    std::ostream* log);

void coherent_C1_accumulate_shared_support_contraction(
    const B4Spins& spins,
    const PackedAmplitudeCoherentBoosters& packed_B,
    const B4CoherentCoefficientsView& C1,
    const CoherentC1InputSupportPlan_HalfSjC& support,
    const PackedCoherentC1SharedOptions& opt,
    WingExecutionOrder order,
    PackedContractionOptions contraction_options,
    PackedCoherentC1SharedResult& result,
    std::ostream* log);

inline double coherent_C1_abs_coeff(const b4utils::Complex64& z) noexcept
{
    return std::hypot(z.re, z.im);
}

B4CoherentCoefficientsView coherent_C1_make_pruned_view(
    const B4CoherentCoefficientsView& C1,
    const PackedCoherentC1SharedOptions& opt,
    PackedCoherentC1SharedResult& result,
    std::vector<int>& kept_iotas,
    std::vector<b4utils::Complex64>& kept_coeffs);


struct CoherentC1SharedXPreflight {
    std::vector<std::uint64_t> incremental_by_x;
    std::vector<int> i1_values;
    std::vector<std::uint64_t> valid_x_pairs_by_i1;
    std::uint64_t valid_i1_x_pairs_total = 0;
    std::uint64_t full_incremental_upper = 0;
    std::uint64_t max_single_incremental = 0;
    std::uint64_t max_single_peak = 0;
    std::uint64_t fixed_bytes = 0;
    std::uint64_t budget_bytes = 0;

    bool empty() const noexcept { return incremental_by_x.empty(); }

    std::uint64_t full_peak_upper() const noexcept {
        return xqslab_sat_add_u64(fixed_bytes, full_incremental_upper);
    }
};

inline double coherent_C1_clamp_budget_fraction(double f) noexcept
{
    if (!std::isfinite(f)) return 0.85;
    if (f <= 0.0) return 0.10;
    if (f > 1.0) return 1.0;
    return f;
}

CoherentC1SharedXPreflight coherent_C1_shared_preflight_one_x_slices(
    const B4Spins& spins,
    int Dl,
    const CoherentAmplitudeChannels& ch,
    const std::uint64_t fixed_bytes,
    const std::uint64_t budget,
    PackedCoherentC1SharedResult& result,
    double& support_seconds_accum);

inline std::size_t coherent_C1_shared_choose_auto_x_end_from_preflight(
    const CoherentC1SharedXPreflight& pf,
    std::size_t x_begin,
    std::size_t x_cap,
    double budget_fraction)
{
    const std::size_t X = pf.incremental_by_x.size();
    x_begin = std::min(x_begin, X);
    x_cap = std::min(x_cap, X);
    if (x_cap <= x_begin) return std::min<std::size_t>(X, x_begin + 1);

    if (pf.budget_bytes <= pf.fixed_bytes) {
        return std::min<std::size_t>(x_cap, x_begin + 1);
    }

    const double frac = coherent_C1_clamp_budget_fraction(budget_fraction);
    const long double avail_ld =
        static_cast<long double>(pf.budget_bytes - pf.fixed_bytes) * static_cast<long double>(frac);
    const std::uint64_t target = static_cast<std::uint64_t>(std::max<long double>(1.0L, avail_ld));

    std::uint64_t acc = 0;
    std::size_t x_end = x_begin;
    while (x_end < x_cap) {
        const std::uint64_t next = xqslab_sat_add_u64(acc, pf.incremental_by_x[x_end]);
        if (x_end > x_begin && next > target) break;
        acc = next;
        ++x_end;
        if (acc >= target) break;
    }

    return std::max<std::size_t>(x_begin + 1, x_end);
}

void coherent_C1_shared_accumulate_xq_fallback_for_all_i1(
    const B4Spins& spins,
    int Dl,
    const std::array<int,10>& spins_array,
    const PackedAmplitudeCoherentBoosters& packed_B,
    const B4CoherentCoefficientsView& C1,
    const PackedCoherentC1SharedOptions& opt,
    const CoherentC1SharedXPreflight& progress_plan,
    WingExecutionOrder order,
    PackedContractionOptions contraction_options,
    PackedCoherentC1SharedResult& result,
    std::ostream* log);

PackedCoherentC1SharedResult contract_packed_amplitude_coherent_C1_shared_auto(
    const B4Spins& spins,
    int Dl,
    const std::array<int,10>& spins_array,
    const PackedAmplitudeCoherentBoosters& packed_B,
    const B4CoherentCoefficientsView& C1,
    const PackedCoherentC1SharedOptions& opt = {},
    WingExecutionOrder order = WingExecutionOrder::LeftThenRight,
    PackedContractionOptions contraction_options = {},
    std::ostream* log = nullptr);



inline PackedCoherentC1SharedResult contract_packed_amplitude_coherent_C1_shared_auto(
    const B4Spins& spins,
    int Dl,
    const std::array<int,10>& spins_array,
    const PackedAmplitudeCoherentBoosters& packed_B,
    const std::vector<int>& C1_iotas,
    const std::vector<b4utils::Complex64>& C1_coeffs,
    const PackedCoherentC1SharedOptions& opt = {},
    WingExecutionOrder order = WingExecutionOrder::LeftThenRight,
    PackedContractionOptions contraction_options = {},
    std::ostream* log = nullptr)
{
    return contract_packed_amplitude_coherent_C1_shared_auto(
        spins, Dl, spins_array, packed_B,
        make_b4_coherent_coefficients_view(C1_iotas, C1_coeffs),
        opt, order, contraction_options, log);
}

} // namespace b4support
