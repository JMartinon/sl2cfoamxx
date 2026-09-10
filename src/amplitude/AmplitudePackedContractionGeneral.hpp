// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "amplitude/AmplitudeInputSupport.hpp"
#include "amplitude/AmplitudeInputSupport_HalfSjC.hpp"
#include "amplitude/AmplitudePackedSJTensors_HalfSjC.hpp"
#include "amplitude/AmplitudePackedBoosters.hpp"
#include "amplitude/AmplitudePackedContractionQLocal_ReflectionQOrbitCostSelected_HalfSjC.hpp"

namespace b4support {

// ============================================================================
// General packed scalar-amplitude contraction
// ============================================================================
//
// This header is meant to be the safe front door:
//
//   * reflection-compatible data uses the existing fastest q-orbit backend;
//   * arbitrary valid data uses a general q-local streamed backend;
//   * half-stored sjC is supported by a half-aware sjC reader, so reflected
//     logical sjC blocks do not require physical payload.
//
// In other words: no more asking a reflection backend to process non-reflection
// data and then acting surprised when C++ throws a chair through the window.
// ============================================================================

inline bool amplitude_general_reflection_qorbit_compatible(
    const AmplitudeInputSupportPlan& support,
    const std::array<int, 10>& spins,
    const std::array<int, 5>& fixed_intertwiners) noexcept
{
    return amplitude_reflection_spins_compatible(spins)
        && amplitude_reflection_fixed_intertwiners_compatible(fixed_intertwiners)
        && amplitude_reflection_support_axes_compatible(support);
}

inline double amplitude_general_load_1d(
    const RangeBlockSupport1D& support,
    const double* values,
    std::size_t block,
    std::size_t inner_index) noexcept
{
    const Range1D& r = support.blocks[block];
    return values[r.offset + (inner_index - r.begin)];
}

inline void amplitude_general_cover_range(Range1D& dst, const Range1D& src) noexcept
{
    if (src.empty()) {
        return;
    }

    if (dst.empty()) {
        dst.begin = src.begin;
        dst.end = src.end;
        dst.offset = 0ull;
        return;
    }

    dst.begin = std::min(dst.begin, src.begin);
    dst.end = std::max(dst.end, src.end);
}

struct AmplitudeGeneralThreadWorkspace {
    std::vector<double> row;
    std::vector<double> projected;

    void ensure(std::size_t L25, std::size_t max_rows) {
        row.resize(L25);
        projected.resize(std::max<std::size_t>(1u, max_rows) * L25);
    }
};

Range1D amplitude_general_build_T2_row(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* T1_values,
    std::size_t ik4,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* row);

Range1D amplitude_general_build_T4_row(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T3_support,
    const double* T3_values,
    std::size_t ik3,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* row);

// Half-sjC-aware q-block contraction for arbitrary data.
//
// This is the general fallback.  It does not use q-orbit reflection at the
// amplitude level.  It still exploits the half-stored sjC payload through
// packed_load_sjC_half(), so it is correct when the logical C block is stored
// under the swapped l24/l35 representative.
double amplitude_general_contract_q_block_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* T1_values,
    const RangeBlockSupport1D& T3_support,
    const double* T3_values,
    std::size_t ix,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    AmplitudeGeneralThreadWorkspace& ws);

double amplitude_general_contract_slice_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* T1_values,
    const RangeBlockSupport1D& T3_support,
    const double* T3_values,
    std::size_t ix,
    PackedContractionOptions opt = {},
    PackedQLocalContractionStats* stats = nullptr);

double contract_packed_amplitude_general(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeBoosters& B,
    const std::array<int, 10>& spins,
    const std::array<int, 5>& fixed_intertwiners,
    WingExecutionOrder order,
    PackedContractionOptions opt = {},
    PackedQLocalContractionStats* stats = nullptr);

} // namespace b4support

