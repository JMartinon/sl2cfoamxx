// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "gpu/FullTensorGpuNativeApi.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

struct B4Spins;
namespace b4support {
struct FullTensorInputSupportPlan;
struct PackedFullTensorBoosters;
}

namespace b4gpu {

using FullTensorSemiFusedProgressBegin =
    void (*)(void* context, std::uint64_t total_rows) noexcept;
using FullTensorSemiFusedProgressAdvance =
    void (*)(void* context, std::uint64_t completed_rows) noexcept;

enum class FullTensorSemiFusedMemoryPolicy : std::uint8_t {
    Automatic,
    ResidentC,
    StreamedC
};

enum class FullTensorSemiFusedFailureKind : std::uint8_t {
    None,
    MemoryPressure,
    Numerical,
    Internal,
    UnsupportedOperation,
    Fatal
};

[[nodiscard]] constexpr const char* full_tensor_semifused_memory_policy_name(
    FullTensorSemiFusedMemoryPolicy policy) noexcept
{
    switch (policy) {
    case FullTensorSemiFusedMemoryPolicy::Automatic: return "auto";
    case FullTensorSemiFusedMemoryPolicy::ResidentC: return "resident-c";
    case FullTensorSemiFusedMemoryPolicy::StreamedC: return "streamed-c";
    }
    return "unknown";
}

struct FullTensorSemiFusedOptions {
    std::uint64_t device_mask = 0;
    double vram_fraction = 0.90;
    std::uint64_t reserve_device_bytes = 512ull * 1024ull * 1024ull;
    std::uint64_t max_device_bytes = 0;
    FullTensorSemiFusedMemoryPolicy memory_policy =
        FullTensorSemiFusedMemoryPolicy::Automatic;
    FullTensorGpuNativeContractionTree tree =
        FullTensorGpuNativeContractionTree::Automatic;

    bool cutensor_jit = false;
    std::uint64_t cutensor_workspace_bytes = 128ull * 1024ull * 1024ull;
    std::uint32_t i1_tile = 0;
    std::uint32_t x_tile = 0;
    std::uint32_t l34_tile = 0;
    std::uint64_t row_capacity = 0;
    // Per-SM producer-grid ceiling; each launch is additionally task-limited.
    std::uint32_t producer_waves_per_sm = 0;
    bool enable_nvtx = false;
    bool enable_profiling = false;
    FullTensorSemiFusedProgressBegin progress_begin = nullptr;
    FullTensorSemiFusedProgressAdvance progress_advance = nullptr;
    void* progress_context = nullptr;
};

struct FullTensorSemiFusedPlan {
    FullTensorSemiFusedMemoryPolicy memory_policy =
        FullTensorSemiFusedMemoryPolicy::Automatic;
    std::uint32_t i1_tile = 0;
    std::uint32_t x_tile = 0;
    std::uint32_t l34_tile = 0;
    std::uint64_t row_capacity = 0;
    // Resolved per-SM producer-grid ceiling, not achieved occupancy.
    std::uint32_t producer_waves_per_sm = 0;
    std::uint64_t estimated_peak_device_bytes = 0;
    FullTensorGpuNativeContractionTree tree =
        FullTensorGpuNativeContractionTree::LogicalQ;
};

// Only timings surfaced by the installed production API are retained.
struct FullTensorSemiFusedStageProfile {
    double t1_seconds = 0.0;
    double t3_seconds = 0.0;
    double left_producer_seconds = 0.0;
    double right_producer_seconds = 0.0;
    double final_gemm_seconds = 0.0;
    double compact_fold_seconds = 0.0;
};

struct FullTensorSemiFusedDeviceStats {
    std::uint64_t final_gemm_calls = 0;
    std::uint64_t final_gemm_matrices = 0;
    std::uint64_t peak_device_bytes = 0;
    double initialization_seconds = 0.0;
    double contraction_seconds = 0.0;
    double download_seconds = 0.0;
    FullTensorSemiFusedStageProfile profile{};
};

struct FullTensorSemiFusedStats {
    FullTensorSemiFusedPlan plan{};
    std::vector<FullTensorSemiFusedDeviceStats> devices;
    double total_seconds = 0.0;
    FullTensorSemiFusedFailureKind failure_kind =
        FullTensorSemiFusedFailureKind::None;
    // Preserve standard and typed private exceptions across the worker/API
    // boundary. The string remains available for fallback diagnostics.
    std::exception_ptr failure{};
};

#if defined(HEAVY_COMPUTE_USE_CUDA)
[[nodiscard]] bool full_tensor_semifused_cuda_available() noexcept;
[[nodiscard]] bool contract_full_tensor_gpu_semifused_generated(
    const B4Spins& spins,
    const b4support::FullTensorInputSupportPlan& support,
    const b4support::PackedFullTensorBoosters& boosters,
    const FullTensorGpuNativeOutputLayout& output_layout,
    const FullTensorSemiFusedOptions& options,
    bool zero_output,
    double* host_output,
    std::size_t host_output_values,
    FullTensorSemiFusedStats& stats,
    std::string& error) noexcept;
#else
inline bool full_tensor_semifused_cuda_available() noexcept { return false; }
inline bool contract_full_tensor_gpu_semifused_generated(
    const B4Spins&,
    const b4support::FullTensorInputSupportPlan&,
    const b4support::PackedFullTensorBoosters&,
    const FullTensorGpuNativeOutputLayout&,
    const FullTensorSemiFusedOptions&,
    bool,
    double*,
    std::size_t,
    FullTensorSemiFusedStats& stats,
    std::string& error) noexcept
{
    stats = {};
    error = "CUDA production full-tensor backend was not compiled";
    return false;
}
#endif

} // namespace b4gpu
