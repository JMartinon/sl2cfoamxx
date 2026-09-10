// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "amplitude/AmplitudePackedCoherentI1Batch.hpp"
#include "core/Failure.hpp"
#include "gpu/CoherentGpuApi.hpp"
#include "gpu/ProductionGpuFallback.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <vector>

namespace b4support {

struct CoherentProductionGpuResult {
    b4utils::Complex64 amplitude{};
    std::uint32_t pipeline_slots = 0;
    std::size_t c1_original_count = 0;
    std::size_t c1_kept_count = 0;
    b4gpu::DenseContractionResult diagnostics{};
};

// Adapter from the project-owned coherent preparation objects to the fixed
// production CUDA contraction.  It has no environment-variable selector or
// host-generated 6j fallback.
inline CoherentProductionGpuResult
contract_packed_coherent_production_on_gpus(
    const B4Spins& spins,
    int Dl,
    const AmplitudeInputSupportPlan& booster_support,
    const CoherentQSlabNativeBoosters& boosters,
    const B4CoherentCoefficientsView& C1,
    double c1_prune_relative_tolerance,
    std::uint32_t pipeline_slots,
    b4gpu::DenseContractionProgressBegin progress_begin = nullptr,
    b4gpu::DenseContractionProgressAdvance progress_advance = nullptr,
    b4gpu::DenseContractionProgressReset progress_reset = nullptr,
    void* progress_context = nullptr,
    b4gpu::DenseContractionWorkerDiagnosticCallback worker_diagnostic = nullptr,
    const void* diagnostic_context = nullptr)
{
#if !defined(HEAVY_COMPUTE_USE_CUDA)
    (void)spins;
    (void)Dl;
    (void)booster_support;
    (void)boosters;
    (void)C1;
    (void)c1_prune_relative_tolerance;
    (void)pipeline_slots;
    (void)progress_begin;
    (void)progress_advance;
    (void)progress_reset;
    (void)progress_context;
    (void)worker_diagnostic;
    (void)diagnostic_context;
    throw std::runtime_error(
        "GPU coherent amplitude requires a build configured with ENABLE_CUDA=ON");
#else
    if (!b4gpu::coherent_cuda_backend_available()) {
        throw std::runtime_error("no CUDA device is visible");
    }

    PackedCoherentC1SharedOptions prune_options;
    prune_options.skip_zero_c1 = true;
    prune_options.C1_prune_relative_tol =
        std::max(0.0, c1_prune_relative_tolerance);
    prune_options.C1_prune_absolute_tol = 0.0;
    prune_options.C1_prune_keep_largest_if_empty = true;

    PackedCoherentC1SharedResult prune_result;
    std::vector<int> kept_iotas;
    std::vector<b4utils::Complex64> kept_coefficients;
    const B4CoherentCoefficientsView kept = coherent_C1_make_pruned_view(
        C1,
        prune_options,
        prune_result,
        kept_iotas,
        kept_coefficients);

    const CoherentAmplitudeChannels channels =
        make_CoherentAmplitudeChannels(spins, Dl, kept_iotas);
    if (channels.empty()) {
        return {{0.0, 0.0}, 0,
                prune_result.C1_original_count,
                prune_result.C1_kept_count};
    }
    if (booster_support.channels.k2 != channels.k2 ||
        booster_support.channels.k3 != channels.k3 ||
        booster_support.channels.k4 != channels.k4 ||
        booster_support.channels.k5 != channels.k5) {
        throw b4utils::InternalFailure(
            "GPU booster support does not match the active coherent channels");
    }

    const auto lifted_axis_matches = [Dl](
                                         const std::vector<int>& axis,
                                         int base) {
        if (axis.size() != static_cast<std::size_t>(Dl + 1)) return false;
        for (std::size_t i = 0; i < axis.size(); ++i) {
            if (axis[i] != base + 2 * static_cast<int>(i)) return false;
        }
        return true;
    };
    if (!lifted_axis_matches(booster_support.l23, spins.Dj23) ||
        !lifted_axis_matches(booster_support.l24, spins.Dj24) ||
        !lifted_axis_matches(booster_support.l25, spins.Dj25) ||
        !lifted_axis_matches(booster_support.l34, spins.Dj34) ||
        !lifted_axis_matches(booster_support.l35, spins.Dj35) ||
        !lifted_axis_matches(booster_support.l45, spins.Dj45)) {
        throw b4utils::InternalFailure(
            "GPU booster lifted-spin axes do not match spins and Dl");
    }

    const b4gpu::DenseCoherentBoosters dense_boosters =
        boosters.view(channels.i1.size());
    const b4gpu::DenseCoherentDimensions dimensions =
        dense_boosters.dimensions;

    std::vector<b4gpu::Complex64> dense_c1(channels.i1.size());
    for (std::size_t ia = 0; ia < channels.i1.size(); ++ia) {
        const std::size_t position = channels.c1_positions[ia];
        if (position >= kept.size()) {
            throw b4utils::InternalFailure(
                "GPU C1 channel mapping is out of range");
        }
        dense_c1[ia] = {
            kept.coefficients[position].re,
            kept.coefficients[position].im};
    }

    const b4gpu::DenseComplexVectorView c1_view =
        b4gpu::make_first_axis_contiguous_view<b4gpu::Complex64, 1>(
            dense_c1.data(),
            {static_cast<std::uint64_t>(dense_c1.size())});
    const auto int_view = [](const std::vector<int>& values) {
        return b4gpu::make_first_axis_contiguous_view<int, 1>(
            values.data(),
            {static_cast<std::uint64_t>(values.size())});
    };
    const b4gpu::DenseCoherentRecurrenceProblem recurrence{
        dimensions,
        spins.Dj12,
        spins.Dj13,
        spins.Dj14,
        spins.Dj15,
        spins.Dj23,
        spins.Dj24,
        spins.Dj25,
        spins.Dj34,
        spins.Dj35,
        spins.Dj45,
        int_view(channels.xvals),
        int_view(channels.i1),
        int_view(channels.k2),
        int_view(channels.k3),
        int_view(channels.k4),
        int_view(channels.k5)};

    b4gpu::DenseContractionOptions options =
        b4gpu::default_dense_contraction_options();
    const std::uint32_t initial_pipeline_slots =
        pipeline_slots == 0 ? options.pipeline_slots : pipeline_slots;
    if (initial_pipeline_slots < 1 || initial_pipeline_slots > 3) {
        throw std::invalid_argument(
            "coherent GPU pipeline slots must be zero or fit in [1,3]");
    }
    options.progress_begin = progress_begin;
    options.progress_advance = progress_advance;
    options.progress_reset = progress_reset;
    options.progress_context = progress_context;
    options.worker_diagnostic = worker_diagnostic;
    options.diagnostic_context = diagnostic_context;
    std::exception_ptr last_memory_failure;
    for (std::uint32_t slots = initial_pipeline_slots; slots >= 1; --slots) {
        options.pipeline_slots = slots;
        try {
            const b4gpu::DenseContractionResult raw =
                b4gpu::contract_device_generated_coherent_multi_gpu(
                    dense_boosters, c1_view, recurrence, options);
            return {{raw.amplitude.re, raw.amplitude.im}, raw.pipeline_slots,
                    prune_result.C1_original_count,
                    prune_result.C1_kept_count, raw};
        } catch (const std::exception& error) {
            if (!sl2cfoam::gpu::production::detail::
                    is_memory_fallback_failure(error)) {
                throw;
            }
            last_memory_failure = std::current_exception();
        }
        if (slots == 1) break;
        if (options.progress_reset != nullptr) {
            options.progress_reset(options.progress_context);
        }
    }
    if (last_memory_failure) std::rethrow_exception(last_memory_failure);
    throw b4utils::InternalFailure(
        "coherent GPU production route did not run");
#endif
}

} // namespace b4support
