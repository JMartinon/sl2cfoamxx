// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "gpu/CoherentGpuRecurrence.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace b4gpu::cuda_detail {

// Occupancy-selected recurrence configurations for the production full-tensor
// route.  Boundary wings feed the two binary cuTENSOR contractions, fused operand
// kernels form sjB*B5 and sjD*B2 directly, and C is generated as a bounded slab.
struct FullTensorDeviceSixJLaunchPlan {
    RecurrenceLaunchConfig wings{};
    RecurrenceLaunchConfig fused_operands{};
    RecurrenceLaunchConfig c_slab{};
    std::uint32_t wing_blocks_per_sm = 0;
    std::uint32_t fused_operand_blocks_per_sm = 0;
    std::uint32_t c_blocks_per_sm = 0;
};

// Select recurrence launch geometry using the doubled-spin bounds and lifted
// axis extents. Calling this after cudaSetDevice makes the plan device-specific.
[[nodiscard]] cudaError_t select_full_tensor_device_6j_launch_plan(
    const DeviceCoherentRecurrenceProblem& problem,
    int max_x,
    FullTensorDeviceSixJLaunchPlan* plan) noexcept;

// Generate A/E for one X value and one i1 tile.  B/D are generated directly
// into their pointwise cuTENSOR operands by the two routines below.
[[nodiscard]] cudaError_t launch_generate_full_tensor_6j_boundary_wings(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    std::uint64_t i1_begin,
    std::uint64_t i1_count,
    double* sjA,
    double* sjE,
    std::uint64_t* device_failure_count,
    const FullTensorDeviceSixJLaunchPlan& plan,
    cudaStream_t stream) noexcept;

// W1 layout: [k5,l45,i5,k4,l25,l35], k5 contiguous.
[[nodiscard]] cudaError_t launch_generate_full_tensor_6j_t1_operand(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    const double* dense_B5,
    std::uint64_t i5_count,
    double* W1,
    std::uint64_t* device_failure_count,
    const FullTensorDeviceSixJLaunchPlan& plan,
    cudaStream_t stream) noexcept;

// W3 layout: [k2,l23,i2,k3,l25,l24], k2 contiguous.
[[nodiscard]] cudaError_t launch_generate_full_tensor_6j_t3_operand(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    const double* dense_B2,
    std::uint64_t i2_count,
    double* W3,
    std::uint64_t* device_failure_count,
    const FullTensorDeviceSixJLaunchPlan& plan,
    cudaStream_t stream) noexcept;

// Generate one fixed-l24, l34-tiled C slab in block-dense
// [k4,k3,l34_count,l35] layout.
[[nodiscard]] cudaError_t launch_generate_full_tensor_6j_c_slab(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    std::uint64_t l24_index,
    std::uint64_t l34_begin,
    std::uint64_t l34_count,
    double* sjC,
    std::uint64_t* device_failure_count,
    const FullTensorDeviceSixJLaunchPlan& plan,
    cudaStream_t stream) noexcept;

static_assert(std::is_trivially_copyable_v<FullTensorDeviceSixJLaunchPlan> &&
              std::is_standard_layout_v<FullTensorDeviceSixJLaunchPlan>);

} // namespace b4gpu::cuda_detail

