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
#include <stdexcept>
#include <string>

#include "amplitude/AmplitudePackedXSlab.hpp"
#include "amplitude/AmplitudePackedXQSlab.hpp"
#include "core/Failure.hpp"

namespace b4support {

// ============================================================================
// Automatic memory-driven amplitude dispatcher.
//
// Policy layer only.  It chooses the cheapest slicing mode allowed by the
// selected memory budget:
//   1. full support + full-x path when safely small;
//   2. generated x-slab path when every x-slab fits;
//   3. x-slice + q-slab sjC streaming fallback otherwise.
//
// This header intentionally does not change the lower-level entry points.  They
// remain useful for validation and forced-regime tests.  This function is the
// production policy layer so callers do not need to select a slicing strategy.
// ============================================================================

enum class PackedAmplitudeAutoMode {
    FullX,
    XSlab,
    XQSlab
};

inline const char* packed_amplitude_auto_mode_name(PackedAmplitudeAutoMode m) noexcept
{
    switch (m) {
        case PackedAmplitudeAutoMode::FullX:  return "full-x";
        case PackedAmplitudeAutoMode::XSlab:  return "x-slab";
        case PackedAmplitudeAutoMode::XQSlab: return "x+q-slab";
    }
    return "unknown";
}

struct PackedAmplitudeAutoOptions {
    PackedAmplitudeXSlabRunOptions xslab;
    PackedAmplitudeXQSlabOptions xqslab;

    bool enable_full_x = true;
    bool enable_xslab = true;
    bool enable_xqslab = true;

    // When true, q-slab fallback inherits the memory/accounting fields from the
    // x-slab planner unless explicitly set in xqslab.  This keeps one memory
    // constraint controlling all levels by default.
    bool inherit_memory_options_to_xqslab = true;

    bool print_decision = false;
    bool print_result_summary = false;
};

struct PackedAmplitudeAutoTimings {
    double full_x_estimate = 0.0;
    double xslab_planning = 0.0;
    double execution = 0.0;
    double total = 0.0;

    double accounted() const noexcept {
        return full_x_estimate + xslab_planning + execution;
    }

    double unaccounted() const noexcept {
        return total - accounted();
    }
};

struct PackedAmplitudeAutoResult {
    double amplitude = 0.0;
    PackedAmplitudeAutoMode mode = PackedAmplitudeAutoMode::FullX;

    PackedAmplitudeSupportFullXBypassEstimate full_x_estimate;
    PackedAmplitudeXSlabPlan xslab_plan;

    PackedAmplitudeXSlabResult xslab_result;
    PackedAmplitudeXQResult xqslab_result;

    PackedAmplitudeAutoTimings timings;

    void print(std::ostream& os) const {
        os << "\nAmplitude auto-slab run summary\n";
        os << "===============================\n";
        os << std::scientific << std::setprecision(17)
           << "amplitude                  = " << amplitude << "\n";
        os << std::fixed << std::setprecision(6);
        os << "selected mode              = " << packed_amplitude_auto_mode_name(mode) << "\n";
        os << "full-x estimated peak      = ";
        xslab_print_bytes(os, full_x_estimate.estimated_peak_bytes); os << "\n";
        os << "full-x allowed peak        = ";
        xslab_print_bytes(os, full_x_estimate.allowed_peak_bytes); os << "\n";
        os << "x-slab planned slabs       = " << xslab_plan.slabs.size() << "\n";
        os << "x-slab unfitting slabs     = " << xslab_plan.num_unfitting_slabs() << "\n";
        os << "x-slab max estimated peak  = ";
        xslab_print_bytes(os, xslab_plan.max_peak_bytes()); os << "\n";
        os << "time full-x estimate       = " << timings.full_x_estimate << " s\n";
        os << "time x-slab planning       = " << timings.xslab_planning << " s\n";
        os << "time selected execution    = " << timings.execution << " s\n";
        os << "time accounted             = " << timings.accounted() << " s\n";
        os << "time unaccounted           = " << timings.unaccounted() << " s\n";
        os << "time total auto path       = " << timings.total << " s\n";
    }
};

inline double packed_auto_seconds_since(std::chrono::high_resolution_clock::time_point t0)
{
    using clock = std::chrono::high_resolution_clock;
    return std::chrono::duration<double>(clock::now() - t0).count();
}

PackedAmplitudeXQSlabOptions packed_auto_make_xq_options(
    const PackedAmplitudeAutoOptions& opt,
    const AmplitudeInputSupportPlan_HalfSjC& static_support);


// Local generated-support slab builder used by the auto dispatcher.
//
// Do not call the x-slab header's internal helper here.  Different development
// snapshots of AmplitudePackedXSlab.hpp have exposed different helper names,
// and relying on them makes this policy header annoyingly fragile.  This helper
// depends only on the stable public half-sjC support builder, then restores the
// global/static booster supports so packed_B offsets remain valid.
inline int packed_auto_infer_Dl_from_static_support_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& static_support) noexcept
{
    if (!static_support.l23.empty()) return static_cast<int>(static_support.l23.size()) - 1;
    if (!static_support.l24.empty()) return static_cast<int>(static_support.l24.size()) - 1;
    if (!static_support.l25.empty()) return static_cast<int>(static_support.l25.size()) - 1;
    if (!static_support.l34.empty()) return static_cast<int>(static_support.l34.size()) - 1;
    if (!static_support.l35.empty()) return static_cast<int>(static_support.l35.size()) - 1;
    if (!static_support.l45.empty()) return static_cast<int>(static_support.l45.size()) - 1;
    return 0;
}

inline AmplitudeInputSupportPlan_HalfSjC packed_auto_build_generated_sj_support_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const B4Spins& spins,
    std::size_t ix_begin,
    std::size_t ix_end)
{
    const std::size_t X = static_support.channels.xvals.size();
    ix_begin = std::min(ix_begin, X);
    ix_end = std::min(ix_end, X);
    if (ix_end < ix_begin) {
        ix_end = ix_begin;
    }

    AmplitudeChannels slab_channels = static_support.channels;
    slab_channels.xvals.assign(
        static_support.channels.xvals.begin() + static_cast<std::ptrdiff_t>(ix_begin),
        static_support.channels.xvals.begin() + static_cast<std::ptrdiff_t>(ix_end));

    const int Dl = packed_auto_infer_Dl_from_static_support_half_sjC(static_support);

    AmplitudeInputSupportPlan_HalfSjC out =
        build_amplitude_input_support_plan_half_sjC(spins, Dl, slab_channels);

    // sj support is slab-local, but booster tensors are global/static.
    out.B2 = static_support.B2;
    out.B3 = static_support.B3;
    out.B4 = static_support.B4;
    out.B5 = static_support.B5;

    // These maps depend only on virtual intertwiner axes, not on x.
    out.sjC_k3_to_k4 = static_support.sjC_k3_to_k4;
    out.sjC_k4_to_k3 = static_support.sjC_k4_to_k3;

    return out;
}

// Execute a previously computed generated x-slab plan without replanning.  This
// avoids paying the generated-support planner twice in the automatic route.
PackedAmplitudeXSlabResult contract_packed_amplitude_xslab_streamed_generated_support_with_plan(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const PackedAmplitudeBoosters& packed_B,
    const std::array<int, 10>& spins,
    const std::array<int, 5>& fixed_intertwiners,
    const PackedAmplitudeXSlabPlan& plan,
    const PackedAmplitudeXSlabRunOptions& run_opt = {},
    std::ostream* log = nullptr);

PackedAmplitudeAutoResult contract_packed_amplitude_auto_streamed_generated_support(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const PackedAmplitudeBoosters& packed_B,
    const std::array<int, 10>& spins,
    const std::array<int, 5>& fixed_intertwiners,
    PackedAmplitudeAutoOptions opt = {},
    std::ostream* log = nullptr);

} // namespace b4support
