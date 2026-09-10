// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "gpu/production/FullTensorGpu.hpp"

#include "core/Failure.hpp"
#include "gpu/FullTensorSemiFusedGpuApi.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace sl2cfoam::gpu::production {
namespace {

[[nodiscard]] b4gpu::FullTensorSemiFusedOptions make_options(
    const FullTensorGpuOptions& user,
    std::uint64_t rows)
{
    validate_memory_policy(user.memory);

    b4gpu::FullTensorSemiFusedOptions options{};
    options.device_mask = user.device_mask != 0
        ? user.device_mask
        : user.memory.device_mask;
    options.vram_fraction = user.memory.usable_vram_fraction;
    options.reserve_device_bytes = user.memory.reserve_bytes;
    options.max_device_bytes = user.memory.maximum_device_bytes;

    options.memory_policy =
        b4gpu::FullTensorSemiFusedMemoryPolicy::Automatic;
    // Use canonical q-reflection orbits whenever the output metadata proves
    // them valid, and the complete logical-q tree for general-spin inputs.
    options.tree = b4gpu::FullTensorGpuNativeContractionTree::Automatic;
    options.cutensor_jit = user.cutensor_jit;
    options.cutensor_workspace_bytes = user.cutensor_workspace_bytes;

    options.i1_tile = user.i1_tile;
    options.x_tile = user.x_tile != 0
        ? user.x_tile
        : (user.x_extent == 0 ? 0 : ceil_divide_u32(user.x_extent, 2));
    options.l34_tile = user.l34_tile;
    options.row_capacity = rows;
    options.producer_waves_per_sm = user.producer_waves_per_sm;
    options.enable_nvtx = user.profiling.nvtx;
    options.enable_profiling = user.profiling.enabled;
    options.progress_begin = user.progress_begin;
    options.progress_advance = user.progress_advance;
    options.progress_context = user.progress_context;

    return options;
}

[[nodiscard]] std::vector<std::uint64_t> row_capacities(
    const FullTensorGpuOptions& user)
{
    const std::uint64_t preferred =
        std::max<std::uint64_t>(1024, user.row_capacity);
    std::vector<std::uint64_t> values{preferred};
    if (!user.allow_row_capacity_fallback) return values;

    std::uint64_t current = preferred;
    while (current > 2048) {
        current = std::max<std::uint64_t>(2048, current / 2);
        if (values.back() != current) values.push_back(current);
    }
    return values;
}

[[nodiscard]] FullTensorGpuProfile convert_profile(
    const b4gpu::FullTensorSemiFusedStats& stats,
    bool enabled)
{
    FullTensorGpuProfile output{};
    output.enabled = enabled;
    output.devices_used = static_cast<std::uint32_t>(stats.devices.size());
    output.row_capacity = stats.plan.row_capacity;
    output.i1_tile = stats.plan.i1_tile;
    output.x_tile = stats.plan.x_tile;
    output.l34_tile = stats.plan.l34_tile;
    output.producer_waves_per_sm = stats.plan.producer_waves_per_sm;
    output.total_seconds = stats.total_seconds;

    for (const b4gpu::FullTensorSemiFusedDeviceStats& device : stats.devices) {
        output.peak_device_bytes = std::max(
            output.peak_device_bytes, device.peak_device_bytes);
        output.final_gemm_calls += device.final_gemm_calls;
        output.final_gemm_matrices += device.final_gemm_matrices;
        output.initialization_seconds = std::max(
            output.initialization_seconds, device.initialization_seconds);
        output.contraction_seconds = std::max(
            output.contraction_seconds, device.contraction_seconds);
        output.download_seconds = std::max(
            output.download_seconds, device.download_seconds);
        output.t1_seconds = std::max(
            output.t1_seconds, device.profile.t1_seconds);
        output.t3_seconds = std::max(
            output.t3_seconds, device.profile.t3_seconds);
        output.left_producer_seconds = std::max(
            output.left_producer_seconds,
            device.profile.left_producer_seconds);
        output.right_producer_seconds = std::max(
            output.right_producer_seconds,
            device.profile.right_producer_seconds);
        output.final_gemm_seconds = std::max(
            output.final_gemm_seconds,
            device.profile.final_gemm_seconds);
        output.compact_fold_seconds = std::max(
            output.compact_fold_seconds,
            device.profile.compact_fold_seconds);
    }
    return output;
}

} // namespace

bool full_tensor_gpu_available() noexcept
{
    return b4gpu::full_tensor_semifused_cuda_available();
}

FullTensorGpuResult compute_full_tensor_gpu(
    const B4Spins& spins,
    const b4support::FullTensorInputSupportPlan& support,
    const b4support::PackedFullTensorBoosters& boosters,
    const b4gpu::FullTensorGpuNativeOutputLayout& output_layout,
    double* host_output,
    std::size_t host_output_values,
    bool zero_output,
    const FullTensorGpuOptions& user)
{
    if (host_output == nullptr && host_output_values != 0) {
        throw std::invalid_argument(
            "host_output is null although host_output_values is nonzero");
    }
    if (!full_tensor_gpu_available()) {
        throw std::runtime_error(
            "the production full-tensor CUDA backend is unavailable");
    }

    FullTensorGpuResult answer{};
    b4gpu::FullTensorSemiFusedFailureKind terminal_failure_kind =
        b4gpu::FullTensorSemiFusedFailureKind::None;
    std::exception_ptr terminal_failure{};
    bool retrying = false;
    for (const std::uint64_t rows : row_capacities(user)) {
        if (retrying && user.progress_reset != nullptr) {
            user.progress_reset(user.progress_context);
        }
        b4gpu::FullTensorSemiFusedStats raw{};
        std::string error;
        const b4gpu::FullTensorSemiFusedOptions options =
            make_options(user, rows);

        if (b4gpu::contract_full_tensor_gpu_semifused_generated(
                spins,
                support,
                boosters,
                output_layout,
                options,
                zero_output,
                host_output,
                host_output_values,
                raw,
                error)) {
            answer.storage =
                raw.plan.memory_policy ==
                        b4gpu::FullTensorSemiFusedMemoryPolicy::StreamedC
                    ? FullTensorStorageMode::streamed_c
                    : FullTensorStorageMode::resident_c;
            answer.profile = convert_profile(raw, user.profiling.enabled);
            return answer;
        }

        answer.fallback_log.add(
            "materialized row capacity " + std::to_string(rows),
            error.empty() ? "backend returned failure without a message"
                          : error);
        terminal_failure_kind = raw.failure_kind;
        terminal_failure = raw.failure;
        if (!user.allow_row_capacity_fallback ||
            raw.failure_kind !=
                b4gpu::FullTensorSemiFusedFailureKind::MemoryPressure) {
            break;
        }
        retrying = true;
    }

    const std::string message = answer.fallback_log.message(
        "no full-tensor GPU production configuration succeeded");
    if (terminal_failure != nullptr) {
        try {
            std::rethrow_exception(terminal_failure);
        } catch (const std::bad_alloc&) {
            throw;
        } catch (const std::invalid_argument&) {
            throw;
        } catch (const std::out_of_range&) {
            throw;
        } catch (const std::length_error&) {
            throw;
        } catch (const std::overflow_error&) {
            throw;
        } catch (const b4utils::ResourceLimitFailure&) {
            throw;
        } catch (const b4utils::NumericalFailure&) {
            throw;
        } catch (const b4utils::InternalFailure&) {
            throw;
        } catch (const b4utils::UnsupportedOperationFailure&) {
            throw;
        } catch (...) {
            // Preserve the retry history for unclassified backend failures.
        }
    }
    if (terminal_failure_kind ==
        b4gpu::FullTensorSemiFusedFailureKind::MemoryPressure) {
        throw b4utils::ResourceLimitFailure(message);
    }
    if (terminal_failure_kind ==
        b4gpu::FullTensorSemiFusedFailureKind::Numerical) {
        throw b4utils::NumericalFailure(message);
    }
    if (terminal_failure_kind ==
        b4gpu::FullTensorSemiFusedFailureKind::Internal) {
        throw b4utils::InternalFailure(message);
    }
    if (terminal_failure_kind ==
        b4gpu::FullTensorSemiFusedFailureKind::UnsupportedOperation) {
        throw b4utils::UnsupportedOperationFailure(message);
    }
    throw std::runtime_error(message);
}

} // namespace sl2cfoam::gpu::production
