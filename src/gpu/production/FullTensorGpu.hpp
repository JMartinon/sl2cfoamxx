// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "gpu/production/GpuProductionCommon.hpp"

#include <cstddef>
#include <cstdint>

// Project-owned input structures are forward-declared so this public header does
// not inherit the large historical GPU API.  Their definitions remain in the
// existing tensor/support headers used by the CPU and CUDA implementations.
struct B4Spins;

namespace b4support {
struct FullTensorInputSupportPlan;
struct PackedFullTensorBoosters;
} // namespace b4support

namespace b4gpu {
struct FullTensorGpuNativeOutputLayout;
} // namespace b4gpu

namespace sl2cfoam::gpu::production {

// Announces the exact row-work total for one backend attempt, then advances it
// only after the corresponding GPU output work has completed.  Both callbacks
// must be noexcept and cheap; CUDA may invoke progress_advance from an internal
// host-callback thread.
using FullTensorProgressBegin =
    void (*)(void* context, std::uint64_t total_rows) noexcept;
using FullTensorProgressAdvance =
    void (*)(void* context, std::uint64_t completed_rows) noexcept;
using FullTensorProgressReset = void (*)(void* context) noexcept;

// -----------------------------------------------------------------------------
// Final full-tensor GPU algorithm
// -----------------------------------------------------------------------------
//
//   recurrence-fused W1 = sjB*B5      recurrence-fused W3 = sjD*B2
//                 |                                  |
//          binary cuTENSOR T1                  binary cuTENSOR T3
//                 |                                  |
//       custom C^T*V producer              custom C*U producer
//                 |                                  |
//           materialized R slab                materialized L slab
//                 +---------------+------------------+
//                                 |
//                     cublasDgemmStridedBatched
//                                 |
//                   compact reflection fold/slab
//                                 |
//                         rank-five output tensor
//
// The public API exposes this single fixed contraction route; only its bounded
// memory and profiling controls are configurable.

struct FullTensorGpuOptions {
    DeviceMemoryPolicy memory{};
    ProfilingPolicy profiling{};

    // Zero means every visible CUDA device admitted by the backend.
    std::uint64_t device_mask = 0;

    // Bounded aggregate K dimension for materialized L/R slabs.  The A100
    // default is 8192; a memory-only fallback halves it down to 2048.
    std::uint64_t row_capacity = 8192;

    // Zero keeps the complete local i1 range.  Smaller i1 tiles repeated output
    // passes and were consistently slower.
    std::uint32_t i1_tile = 0;

    // Zero selects ceil(X/2) when x_extent is supplied.  This corresponds to
    // approximately two x slabs, the broad optimum of the measured matrix.
    std::uint32_t x_tile = 0;
    std::uint32_t x_extent = 0;

    // Zero requests the complete l34 extent.  The backend may stream C when a
    // resident full-r slab cannot be admitted to the device budget.
    std::uint32_t l34_tile = 0;

    // Per-SM cap on each producer grid. The actual block count is also limited
    // by the number of producer tasks. Equal left/right caps gave the best
    // overlap on A100.
    std::uint32_t producer_waves_per_sm = 8;

    std::uint64_t cutensor_workspace_bytes =
        128ull * 1024ull * 1024ull;
    bool cutensor_jit = false;

    bool allow_row_capacity_fallback = true;

    FullTensorProgressBegin progress_begin = nullptr;
    FullTensorProgressAdvance progress_advance = nullptr;
    FullTensorProgressReset progress_reset = nullptr;
    void* progress_context = nullptr;
};

enum class FullTensorStorageMode : std::uint8_t {
    resident_c,
    streamed_c
};

struct FullTensorGpuProfile {
    bool enabled = false;
    std::uint32_t devices_used = 0;
    std::uint64_t peak_device_bytes = 0;
    std::uint64_t row_capacity = 0;
    std::uint32_t i1_tile = 0;
    std::uint32_t x_tile = 0;
    std::uint32_t l34_tile = 0;
    // Resolved per-SM producer-grid ceiling, not achieved occupancy.
    std::uint32_t producer_waves_per_sm = 0;
    std::uint64_t final_gemm_calls = 0;
    std::uint64_t final_gemm_matrices = 0;

    double initialization_seconds = 0.0;
    double t1_seconds = 0.0;
    double t3_seconds = 0.0;
    double left_producer_seconds = 0.0;
    double right_producer_seconds = 0.0;
    double final_gemm_seconds = 0.0;
    double compact_fold_seconds = 0.0;
    double contraction_seconds = 0.0;
    double download_seconds = 0.0;
    double total_seconds = 0.0;
};

struct FullTensorGpuResult {
    FullTensorStorageMode storage = FullTensorStorageMode::resident_c;
    FullTensorGpuProfile profile{};
    FallbackLog fallback_log{};
};

// Compute the complete compact reflection-orbit tensor.
//
// host_output must contain exactly the number of doubles required by
// output_layout.  The implementation retries smaller row capacities only when
// the preferred materialized slab cannot be admitted or allocated.
[[nodiscard]] FullTensorGpuResult compute_full_tensor_gpu(
    const B4Spins& spins,
    const b4support::FullTensorInputSupportPlan& support,
    const b4support::PackedFullTensorBoosters& boosters,
    const b4gpu::FullTensorGpuNativeOutputLayout& output_layout,
    double* host_output,
    std::size_t host_output_values,
    bool zero_output,
    const FullTensorGpuOptions& options = {});

[[nodiscard]] bool full_tensor_gpu_available() noexcept;

} // namespace sl2cfoam::gpu::production
