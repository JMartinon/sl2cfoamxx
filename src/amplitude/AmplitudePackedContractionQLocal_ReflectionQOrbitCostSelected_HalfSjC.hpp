// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "amplitude/AmplitudeInputSupport_HalfSjC.hpp"
#include "amplitude/AmplitudePackedSJTensors_HalfSjC.hpp"
#include "amplitude/AmplitudePackedBoosters.hpp"

// Reuse the q-local machinery, T1/T3 builders, statistics, workspace, and cost
// estimator. The half-sjC wrapper changes q-orbit traversal so that reflected
// logical sjC blocks never require physical payload.
#include "amplitude/AmplitudePackedContractionQLocal_ReflectionQOrbitCostSelected.hpp"

namespace b4support {

double contract_qlocal_streamed_slice_reflection_qorbit_cost_selected_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* B4PACK_RESTRICT T1_values,
    const RangeBlockSupport1D& T3_support,
    const double* B4PACK_RESTRICT T3_values,
    std::size_t ix,
    PackedContractionOptions opt = {},
    PackedQLocalContractionStats* stats = nullptr);

double contract_packed_amplitude_planAB_qlocal_reflection_qorbit_cost_selected_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeBoosters& B,
    const std::array<int, 10>& spins,
    const std::array<int, 5>& fixed_intertwiners,
    WingExecutionOrder order,
    PackedContractionOptions opt = {},
    PackedQLocalContractionStats* stats = nullptr,
    bool require_compatible = true);

} // namespace b4support
