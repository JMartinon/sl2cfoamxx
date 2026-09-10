// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cuda_runtime.h>

#include "CoherentGpuApi.hpp"

#include <cstdint>
#include <type_traits>

namespace b4gpu::cuda_detail {

// All spin labels use the project's doubled-spin convention.  The channel
// arrays reside on the selected CUDA device.  The lifted-spin axes are regular
// step-two ranges and are represented by their boundary-spin origins plus the
// extents in dimensions.
struct DeviceCoherentRecurrenceProblem {
    DenseCoherentDimensions dimensions;

    const int* x;
    std::uint64_t x_count;
    const int* i1;
    const int* k2;
    const int* k3;
    const int* k4;
    const int* k5;

    int Dj12;
    int Dj13;
    int Dj14;
    int Dj15;
    int Dj23;
    int Dj24;
    int Dj25;
    int Dj34;
    int Dj35;
    int Dj45;
};

// One block is one warp.  `capacity` is the maximum complete recurrence
// string stored in shared memory, and `lanes` is the number of simultaneous
// variable-j4 strings in that warp.  The dynamic shared-memory requirement is
//
//     capacity * (3 + lanes) * sizeof(double).
//
// The three shared coefficient arrays are the common off-diagonal coupling,
// the j4-independent diagonal term, and its j4 slope.  X and Z are derived
// from adjacent coupling entries in registers.
//
// The recurrence-fused W1/W3 family reserves one additional double per lane
// for validity flags after the recurrence finishes.  Occupancy selection
// accounts for that small extension.
//
// Keeping this explicit lets the backend choose the best legal occupancy for
// the selected production device without changing the kernels.
struct RecurrenceLaunchConfig {
    std::uint32_t capacity;
    std::uint32_t lanes;
};

enum class RecurrenceKernelFamily : std::uint8_t {
    wings,
    fused_full_tensor_operands,
    c_tile
};

// Select the lane count that maximizes resident recurrence series according to
// CUDA's occupancy model.  Ties prefer more resident warps to hide the latency
// of the dependency-chained FP64 divisions.  When the device exposes opt-in
// dynamic shared memory, the helper includes those legal configurations in the
// same occupancy search.
[[nodiscard]] cudaError_t select_coherent_recurrence_launch_config(
    std::uint32_t capacity,
    std::uint32_t maximum_lanes,
    RecurrenceKernelFamily family,
    RecurrenceLaunchConfig* config,
    int* resident_blocks_per_sm = nullptr) noexcept;

// Generate all four one-X 6j wings in the first-axis-contiguous layouts used
// by DeviceContraction:
//
//   sjA [i1,k5,l25]       sjB [k5,k4,l35,l45]
//   sjD [k3,k2,l23,l24]   sjE [i1,k2,l25]
//
// Outputs are cleared asynchronously before their kernels are launched.
// `device_failure_count` is a required accumulator in device memory.  It is
// incremented for capacity or numerical failures and must be checked after
// stream completion; otherwise a failed string would be indistinguishable
// from a mathematically zero one in the pre-cleared output.
[[nodiscard]] cudaError_t launch_generate_coherent_wings(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    double* sjA,
    double* sjB,
    double* sjD,
    double* sjE,
    std::uint64_t* device_failure_count,
    const RecurrenceLaunchConfig& config,
    cudaStream_t stream) noexcept;

// Component launchers used by the full-tensor backend.  sjB/sjD are common to
// every i1 tile for a fixed X value, whereas sjA/sjE change with the i1
// tile.  Splitting the launch avoids regenerating the potentially large common
// wings when I1 itself must be tiled to a small VRAM budget.
[[nodiscard]] cudaError_t launch_generate_coherent_boundary_wings(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    double* sjA,
    double* sjE,
    std::uint64_t* device_failure_count,
    const RecurrenceLaunchConfig& config,
    cudaStream_t stream) noexcept;

[[nodiscard]] cudaError_t launch_generate_coherent_common_wings(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    double* sjB,
    double* sjD,
    std::uint64_t* device_failure_count,
    const RecurrenceLaunchConfig& config,
    cudaStream_t stream) noexcept;

// Full-tensor cuTENSOR operand generators.  They fuse the virtual intertwiner-recurring 6j
// wing with the corresponding dense booster and write the final virtual intertwiner-fast
// binary-contraction operand directly.  No intermediate sjB/sjD tensor is
// materialized.
//
// dense_B5 is laid out as
//
//   B5[k5,l45,l35,l25,i5]
//
// with k5 contiguous, equivalently
//
//   B5[((i5*l25 + p)*l35 + s)*l45 + t][k5].
//
// The output is
//
//   W1[k5,l45,i5,k4,l25,l35]
//
// with k5 contiguous, exactly as consumed by the cuTENSOR T1 plan.
[[nodiscard]] cudaError_t
launch_generate_coherent_sjB_times_B5_operand(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    const double* dense_B5,
    std::uint64_t i5_count,
    double* W1,
    std::uint64_t* device_failure_count,
    const RecurrenceLaunchConfig& config,
    cudaStream_t stream) noexcept;

// dense_B2 is laid out as
//
//   B2[k2,l25,l24,l23,i2]
//
// with k2 contiguous.  The output is
//
//   W3[k2,l23,i2,k3,l25,l24]
//
// with k2 contiguous, exactly as consumed by the cuTENSOR T3 plan.
[[nodiscard]] cudaError_t
launch_generate_coherent_sjD_times_B2_operand(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    const double* dense_B2,
    std::uint64_t i2_count,
    double* W3,
    std::uint64_t* device_failure_count,
    const RecurrenceLaunchConfig& config,
    cudaStream_t stream) noexcept;

// Generate a true fixed-l24 slab of sjC.  The output is [k4,k3,Z] with
//
//     Z = r_count * l35_count,
//     z = (r-r_begin) + r_count*s.
// This is the recurrence-native layout consumed directly by both retained GPU
// contractions.
[[nodiscard]] cudaError_t launch_generate_coherent_c_qr_slab(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    std::uint64_t q_index,
    std::uint64_t r_begin,
    std::uint64_t r_count,
    double* sjC,
    std::uint64_t* device_failure_count,
    const RecurrenceLaunchConfig& config,
    cudaStream_t stream) noexcept;

static_assert(std::is_trivial_v<DeviceCoherentRecurrenceProblem> &&
              std::is_standard_layout_v<DeviceCoherentRecurrenceProblem>);
static_assert(std::is_trivial_v<RecurrenceLaunchConfig> &&
              std::is_standard_layout_v<RecurrenceLaunchConfig>);

} // namespace b4gpu::cuda_detail

