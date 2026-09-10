// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "gpu/FullTensorGpuDeviceSixJ.cuh"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace b4gpu::cuda_detail {
namespace {

bool lifted_max(int origin, std::uint64_t extent, int& value) noexcept
{
    if (origin < 0 || extent == 0) return false;
    const std::uint64_t delta = 2u * (extent - 1u);
    if (delta > static_cast<std::uint64_t>(
                    std::numeric_limits<int>::max() - origin)) {
        return false;
    }
    value = origin + static_cast<int>(delta);
    return true;
}

bool recurrence_capacity(std::initializer_list<int> legs,
                         std::uint32_t& capacity) noexcept
{
    const int minimum = *std::min_element(legs.begin(), legs.end());
    if (minimum < 0) return false;
    const std::uint64_t capacity64 =
        static_cast<std::uint64_t>(minimum) + 1u;
    if (capacity64 > std::numeric_limits<std::uint32_t>::max()) return false;
    capacity = static_cast<std::uint32_t>(capacity64);
    return true;
}

} // namespace

cudaError_t select_full_tensor_device_6j_launch_plan(
    const DeviceCoherentRecurrenceProblem& problem,
    int max_x,
    FullTensorDeviceSixJLaunchPlan* plan) noexcept
{
    if (plan == nullptr || max_x < 0 || problem.dimensions.a == 0 ||
        problem.dimensions.k2 == 0 || problem.dimensions.k3 == 0 ||
        problem.dimensions.k4 == 0 || problem.dimensions.k5 == 0) {
        return cudaErrorInvalidValue;
    }
    const DenseCoherentDimensions& d = problem.dimensions;
    int max_l23 = 0, max_l24 = 0, max_l25 = 0;
    int max_l34 = 0, max_l35 = 0, max_l45 = 0;
    if (!lifted_max(problem.Dj23, d.l23, max_l23) ||
        !lifted_max(problem.Dj24, d.l24, max_l24) ||
        !lifted_max(problem.Dj25, d.l25, max_l25) ||
        !lifted_max(problem.Dj34, d.l34, max_l34) ||
        !lifted_max(problem.Dj35, d.l35, max_l35) ||
        !lifted_max(problem.Dj45, d.l45, max_l45)) {
        return cudaErrorInvalidValue;
    }

    std::uint32_t cap_a = 0, cap_b = 0, cap_d = 0, cap_e = 0, cap_c = 0;
    if (!recurrence_capacity(
            {problem.Dj15, max_l25, max_x, problem.Dj14}, cap_a) ||
        !recurrence_capacity(
            {max_x, problem.Dj14, max_l45, max_l35}, cap_b) ||
        !recurrence_capacity(
            {max_l23, max_l24, max_x, problem.Dj13}, cap_d) ||
        !recurrence_capacity(
            {problem.Dj12, max_l25, max_x, problem.Dj13}, cap_e) ||
        !recurrence_capacity(
            {max_l34, max_l35, max_x, max_l24}, cap_c)) {
        return cudaErrorInvalidValue;
    }

    FullTensorDeviceSixJLaunchPlan out{};
    int wing_blocks = 0;
    int fused_operand_blocks = 0;
    int c_blocks = 0;
    const std::uint32_t wing_lanes = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(32, std::max({d.a, d.k3, d.k4})));
    const std::uint32_t c_lanes = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(32, d.k4));
    cudaError_t status = select_coherent_recurrence_launch_config(
        std::max({cap_a, cap_b, cap_d, cap_e}), wing_lanes,
        RecurrenceKernelFamily::wings, &out.wings, &wing_blocks);
    if (status != cudaSuccess) return status;
    status = select_coherent_recurrence_launch_config(
        std::max(cap_b, cap_d), wing_lanes,
        RecurrenceKernelFamily::fused_full_tensor_operands,
        &out.fused_operands, &fused_operand_blocks);
    if (status != cudaSuccess) return status;
    status = select_coherent_recurrence_launch_config(
        cap_c, c_lanes, RecurrenceKernelFamily::c_tile,
        &out.c_slab, &c_blocks);
    if (status != cudaSuccess) return status;
    out.wing_blocks_per_sm = static_cast<std::uint32_t>(wing_blocks);
    out.fused_operand_blocks_per_sm =
        static_cast<std::uint32_t>(fused_operand_blocks);
    out.c_blocks_per_sm = static_cast<std::uint32_t>(c_blocks);
    *plan = out;
    return cudaSuccess;
}

cudaError_t launch_generate_full_tensor_6j_boundary_wings(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    std::uint64_t i1_begin,
    std::uint64_t i1_count,
    double* sjA,
    double* sjE,
    std::uint64_t* device_failure_count,
    const FullTensorDeviceSixJLaunchPlan& plan,
    cudaStream_t stream) noexcept
{
    const std::uint64_t A = problem.dimensions.a;
    if (i1_count == 0 || i1_begin > A || i1_count > A - i1_begin ||
        problem.i1 == nullptr) {
        return cudaErrorInvalidValue;
    }
    DeviceCoherentRecurrenceProblem tile = problem;
    tile.dimensions.a = i1_count;
    tile.i1 = problem.i1 + i1_begin;
    return launch_generate_coherent_boundary_wings(
        tile, x_index, sjA, sjE, device_failure_count,
        plan.wings, stream);
}

cudaError_t launch_generate_full_tensor_6j_t1_operand(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    const double* dense_B5,
    std::uint64_t i5_count,
    double* W1,
    std::uint64_t* device_failure_count,
    const FullTensorDeviceSixJLaunchPlan& plan,
    cudaStream_t stream) noexcept
{
    return launch_generate_coherent_sjB_times_B5_operand(
        problem, x_index, dense_B5, i5_count, W1,
        device_failure_count, plan.fused_operands, stream);
}

cudaError_t launch_generate_full_tensor_6j_t3_operand(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    const double* dense_B2,
    std::uint64_t i2_count,
    double* W3,
    std::uint64_t* device_failure_count,
    const FullTensorDeviceSixJLaunchPlan& plan,
    cudaStream_t stream) noexcept
{
    return launch_generate_coherent_sjD_times_B2_operand(
        problem, x_index, dense_B2, i2_count, W3,
        device_failure_count, plan.fused_operands, stream);
}

cudaError_t launch_generate_full_tensor_6j_c_slab(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    std::uint64_t l24_index,
    std::uint64_t l34_begin,
    std::uint64_t l34_count,
    double* sjC,
    std::uint64_t* device_failure_count,
    const FullTensorDeviceSixJLaunchPlan& plan,
    cudaStream_t stream) noexcept
{
    return launch_generate_coherent_c_qr_slab(
        problem, x_index, l24_index, l34_begin, l34_count, sjC,
        device_failure_count, plan.c_slab, stream);
}

} // namespace b4gpu::cuda_detail

