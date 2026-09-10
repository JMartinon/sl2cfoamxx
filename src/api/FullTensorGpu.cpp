// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "api/FullTensorInternal.hpp"

#include "api/ConfigurationInternal.hpp"
#include "api/InputValidation.hpp"

#include <stdexcept>

#if defined(HEAVY_COMPUTE_USE_CUDA)
#include "full_tensor/FullTensorInputSupport_HalfSjC.hpp"
#include "full_tensor/FullTensorPackedBoosters_HalfSjC.hpp"
#include "full_tensor/CpuContraction.hpp"
#include "gpu/production/FullTensorGpu.hpp"
#include "core/LowLevelTuning.hpp"
#include "core/tensors.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>
#endif

namespace sl2cfoamxx::detail {

#if defined(HEAVY_COMPUTE_USE_CUDA)
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double seconds_since(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

[[nodiscard]] constexpr bool is_disk_storage(
    TensorStorage storage) noexcept
{
    return storage == TensorStorage::on_disk_sym ||
           storage == TensorStorage::on_disk_dense;
}

[[nodiscard]] constexpr bool requests_symmetry_storage(
    TensorStorage storage) noexcept
{
    return storage == TensorStorage::in_memory_sym ||
           storage == TensorStorage::on_disk_sym;
}

[[nodiscard]] constexpr TensorStorage effective_storage(
    TensorStorage requested,
    bool symmetry_is_available) noexcept
{
    if (!requests_symmetry_storage(requested) || symmetry_is_available) {
        return requested;
    }
    return is_disk_storage(requested)
        ? TensorStorage::on_disk_dense
        : TensorStorage::in_memory_dense;
}

[[nodiscard]] FullTensorReassociatedStorageOrder reassociated_storage_order(
    const b4support::FullTensorReassociatedAutoOutputPlan& plan) noexcept
{
    return plan.uses_reflection_compact()
        ? FullTensorReassociatedStorageOrder::ReflectionTriangularByI1
        : FullTensorReassociatedStorageOrder::RectangularByI1;
}

[[nodiscard]] const char* gpu_storage_name(
    sl2cfoam::gpu::production::FullTensorStorageMode storage) noexcept
{
    using Storage = sl2cfoam::gpu::production::FullTensorStorageMode;
    switch (storage) {
    case Storage::resident_c:
        return "resident_c";
    case Storage::streamed_c:
        return "streamed_c";
    }
    return "unknown";
}

class TemporaryTensorFile {
public:
    explicit TemporaryTensorFile(const std::filesystem::path& directory)
    {
        static std::atomic<std::uint64_t> sequence{0};
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        if (error) {
            throw std::filesystem::filesystem_error(
                "cannot create the full-tensor scratch directory",
                directory,
                error);
        }
        for (std::filesystem::directory_iterator iterator(directory, error), end;
             !error && iterator != end;
             iterator.increment(error)) {
            const auto& entry = *iterator;
            if (entry.is_regular_file(error) &&
                entry.path().filename().string().starts_with(
                    "gpu_reassociated_")) {
                std::filesystem::remove(entry.path(), error);
            }
            if (error) break;
        }
        if (error) {
            throw std::filesystem::filesystem_error(
                "cannot clean stale GPU full-tensor scratch files",
                directory,
                error);
        }
        const auto stamp = std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count();
        path_ = directory /
                ("gpu_reassociated_" + std::to_string(::getpid()) + "_" +
                 std::to_string(stamp) + "_" +
                 std::to_string(sequence.fetch_add(1)) + ".sl2cxx");
    }

    ~TemporaryTensorFile()
    {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept
    {
        return path_;
    }

private:
    std::filesystem::path path_;
};

} // namespace
#endif

FullTensor compute_full_tensor_gpu(
    const Spins& spins_array,
    std::size_t Dl,
    double gamma,
    const FullTensorConfigSnapshot& resolved)
{
#if !defined(HEAVY_COMPUTE_USE_CUDA)
    (void)spins_array;
    (void)Dl;
    (void)gamma;
    (void)resolved;
    throw Error(
        ErrorCode::backend_unavailable,
        "GPU full tensor requires a build configured with ENABLE_CUDA=ON");
#else
    const auto backend_start = Clock::now();
    const Diagnostics diagnostics(
        resolved.config, DiagnosticOperation::full_tensor, Backend::GPU);
    if (!sl2cfoam::gpu::production::full_tensor_gpu_available()) {
        throw Error(
            ErrorCode::backend_unavailable,
            "the production full-tensor CUDA backend is unavailable");
    }
    const B4Spins spins(spins_array);
    const FullTensorChannels channels = make_FullTensorChannels(spins, Dl);
    const FullTensor::shape_type dimensions{{
        channels.i1.size(), channels.i2.size(), channels.i3.size(),
        channels.i4.size(), channels.i5.size()}};
    const double channels_seconds = seconds_since(backend_start);

    std::uint64_t output_values = 1;
    for (const std::size_t extent : dimensions) {
        if (extent != 0 &&
            output_values > std::numeric_limits<std::uint64_t>::max() /
                                extent) {
            throw std::overflow_error("full-tensor output size overflows uint64_t");
        }
        output_values *= static_cast<std::uint64_t>(extent);
    }
    if (output_values > std::numeric_limits<std::uint64_t>::max() /
                            sizeof(double)) {
        throw std::overflow_error("full-tensor output bytes overflow uint64_t");
    }
    const std::uint64_t output_bytes = output_values * sizeof(double);

    const auto support_start = Clock::now();
    const b4support::FullTensorInputSupportPlan support =
        b4support::build_full_tensor_input_support_plan(
            spins,
            static_cast<int>(Dl),
            channels);
    const b4support::FullTensorReassociatedAutoOutputPlan output_plan =
        b4support::make_full_tensor_reassociated_auto_output_plan(support);
    if (output_plan.values_count >
        std::numeric_limits<std::uint64_t>::max() / sizeof(double)) {
        throw std::overflow_error(
            "GPU reassociated output size overflows uint64_t");
    }
    const std::uint64_t reassociated_values =
        static_cast<std::uint64_t>(output_plan.values_count);
    const std::uint64_t reassociated_bytes =
        reassociated_values * sizeof(double);
    const double support_seconds = seconds_since(support_start);

    const bool symmetry_is_available = output_plan.uses_reflection_compact();
    const TensorStorage storage =
        effective_storage(resolved.storage, symmetry_is_available);
    const bool use_reassociated = requests_symmetry_storage(storage);
    const bool output_on_disk = is_disk_storage(storage);
    const std::uint64_t stored_values = use_reassociated
        ? reassociated_values
        : output_values;
    const std::uint64_t stored_bytes = use_reassociated
        ? reassociated_bytes
        : output_bytes;
    if (resolved.maximum_output_bytes != 0 &&
        stored_bytes > resolved.maximum_output_bytes) {
        throw Error(
            ErrorCode::resource_limit,
            "full-tensor output exceeds "
            "LibraryConfig::tensors.maximum_payload_size_bytes");
    }
    diagnostics.log(
        Verbosity::detailed,
        "output dimensions=[",
        dimensions[0], ',', dimensions[1], ',', dimensions[2], ',',
        dimensions[3], ',', dimensions[4], "] logical_values=", output_values,
        " logical_bytes=", output_bytes,
        " stored_values=", stored_values,
        " stored_bytes=", stored_bytes,
        " symmetry_available=", symmetry_is_available ? "yes" : "no",
        " requested_storage=", to_string(resolved.storage),
        " effective_storage=", to_string(storage),
        " path=",
        resolved.output_path.empty() ? std::string("<memory>")
                                     : resolved.output_path.string());

    double cache_lookup_seconds = 0.0;
    const char* cache_status = "disabled";
    if (output_on_disk) {
        const auto cache_start = Clock::now();
        std::error_code error;
        if (resolved.reuse_existing) cache_status = "miss";
        const bool exists = resolved.reuse_existing &&
            std::filesystem::exists(resolved.output_path, error);
        if (error) {
            throw std::filesystem::filesystem_error(
                "cannot inspect the existing full tensor",
                resolved.output_path,
                error);
        }
        if (exists) {
            try {
                FullTensor existing = FullTensor::map(
                    resolved.output_path, TensorAccess::read_write);
                const auto expected_tag = use_reassociated
                    ? make_full_tensor_binary_tag_reassociated_with_storage_order(
                        spins_array,
                        Dl,
                        gamma,
                        dimensions,
                        stored_values,
                        reassociated_storage_order(output_plan),
                        resolved.config.simplicity_constraint)
                    : make_full_tensor_binary_tag_canonical(
                        spins_array, Dl, gamma, dimensions, output_values,
                        resolved.config.simplicity_constraint);
                if (existing.shape() == dimensions &&
                    existing.storage() == storage &&
                    existing.layout() == (use_reassociated
                        ? TensorLayout::FullTensorReassociatedReflectionCompactByI1
                        : TensorLayout::CanonicalDenseColumnMajor) &&
                    existing.stored_size() == stored_values &&
                    existing.tag() == expected_tag) {
                    cache_lookup_seconds = seconds_since(cache_start);
                    diagnostics.log(
                        Verbosity::detailed,
                        "cache status=hit path=",
                        resolved.output_path.string(),
                        " lookup_seconds=", cache_lookup_seconds);
                    diagnostics.timing(
                        {
                            {"cache lookup", "cache_lookup_seconds",
                             cache_lookup_seconds},
                            {"total", "backend_total_seconds",
                             seconds_since(backend_start)},
                        },
                        "hit");
                    return existing;
                }
                throw Error(
                    ErrorCode::cache_failure,
                    "shape, layout, or physics tag does not match");
            } catch (const std::filesystem::filesystem_error&) {
                throw;
            } catch (const std::bad_alloc&) {
                throw;
            } catch (const std::exception& exception) {
                throw Error(
                    ErrorCode::cache_failure,
                    "existing full-tensor cache is not reusable: " +
                    std::string(exception.what()) +
                    ". Set LibraryConfig::tensors.reuse_existing=false "
                    "to replace it");
            }
        }
        cache_lookup_seconds = seconds_since(cache_start);
        diagnostics.log(
            Verbosity::detailed,
            "cache status=", cache_status,
            " path=", resolved.output_path.string(),
            " lookup_seconds=", cache_lookup_seconds);
    } else {
        diagnostics.log(
            Verbosity::detailed,
            "cache status=disabled path=<memory>");
    }

    std::unique_ptr<AtomicTensorOutput> publication;
    if (output_on_disk) {
        publication = std::make_unique<AtomicTensorOutput>(
            resolved.output_path);
    }

    const auto preflight_start = Clock::now();
    std::uint64_t estimated_host_peak_bytes = 0;
    if (resolved.memory_budget_bytes != 0) {
        const auto add_saturating = [](std::uint64_t left,
                                       std::uint64_t right) {
            return right > std::numeric_limits<std::uint64_t>::max() - left
                ? std::numeric_limits<std::uint64_t>::max()
                : left + right;
        };
        const auto multiply_saturating = [](std::uint64_t left,
                                            std::uint64_t right) {
            return left != 0 &&
                   right > std::numeric_limits<std::uint64_t>::max() / left
                ? std::numeric_limits<std::uint64_t>::max()
                : left * right;
        };
        const auto dense_bytes = [&](std::initializer_list<std::uint64_t> dims) {
            std::uint64_t values = 1;
            for (const std::uint64_t extent : dims) {
                values = multiply_saturating(values, extent);
            }
            return multiply_saturating(values, sizeof(double));
        };

        std::uint64_t base = 0;
        base = add_saturating(base, support.total_metadata_bytes());
        base = add_saturating(base, support.B2.payload_bytes());
        base = add_saturating(base, support.B3.payload_bytes());
        base = add_saturating(base, support.B4.payload_bytes());
        base = add_saturating(base, support.B5.payload_bytes());

        std::uint64_t dense_booster_transient = 0;
        dense_booster_transient = add_saturating(
            dense_booster_transient,
            dense_bytes({support.l23.size(), support.l24.size(),
                         support.l25.size(), channels.i2.size(),
                         channels.k2.size()}));
        dense_booster_transient = add_saturating(
            dense_booster_transient,
            dense_bytes({support.l34.size(), support.l35.size(),
                         support.l23.size(), channels.i3.size(),
                         channels.k3.size()}));
        dense_booster_transient = add_saturating(
            dense_booster_transient,
            dense_bytes({support.l45.size(), support.l24.size(),
                         support.l34.size(), channels.i4.size(),
                         channels.k4.size()}));
        dense_booster_transient = add_saturating(
            dense_booster_transient,
            dense_bytes({support.l25.size(), support.l35.size(),
                         support.l45.size(), channels.i5.size(),
                         channels.k5.size()}));

        const std::uint64_t booster_peak = add_saturating(
            base, dense_booster_transient);
        std::uint64_t contraction_peak = base;
        if (!output_on_disk) {
            contraction_peak = add_saturating(
                contraction_peak, reassociated_bytes);
            if (!use_reassociated) {
                contraction_peak = add_saturating(
                    contraction_peak, output_bytes);
            }
        }
        const std::uint64_t required =
            std::max(booster_peak, contraction_peak);
        estimated_host_peak_bytes = required;
        if (required > resolved.memory_budget_bytes) {
            throw Error(
                ErrorCode::resource_limit,
                "GPU full-tensor host preparation exceeds the configured "
                "memory budget; use an on-disk storage mode or increase "
                "LibraryConfig::memory_budget_bytes");
        }
    }
    const double preflight_seconds = seconds_since(preflight_start);
    diagnostics.log(
        Verbosity::detailed,
        "GPU host_plan memory_budget_bytes=",
        resolved.memory_budget_bytes,
        " estimated_peak_bytes=", estimated_host_peak_bytes,
        " support_metadata_bytes=", support.total_metadata_bytes(),
        " booster_payload_bytes=",
        support.B2.payload_bytes() + support.B3.payload_bytes() +
            support.B4.payload_bytes() + support.B5.payload_bytes(),
        " reassociated_output_bytes=", reassociated_bytes,
        " canonical_output_bytes=", output_bytes);

    const auto booster_start = Clock::now();
    const std::vector<double> factorials = LogFactTable(
        log_factorial_maximum(spins_array, Dl));

    b4support::PackedFullTensorBoosters boosters;
    const std::array<std::size_t, 6> lifted_extents{{
        support.l23.size(),
        support.l24.size(),
        support.l25.size(),
        support.l34.size(),
        support.l35.size(),
        support.l45.size()}};
    b4support::fill_packed_full_tensor_boosters_via_canonical(
        spins,
        lifted_extents,
        gamma,
        factorials,
        support,
        boosters,
        resolved.config.simplicity_constraint,
        resolved.config.low_level.booster_rho_tile);
    const double booster_fill_seconds = seconds_since(booster_start);

    const b4gpu::FullTensorGpuNativeOutputLayout output_layout =
        b4support::make_full_tensor_gpu_native_output_layout(
            support,
            output_plan);

    const auto output_allocate_start = Clock::now();
    const auto dense_output_tag = make_full_tensor_binary_tag_canonical(
        spins_array, Dl, gamma, dimensions, output_values,
        resolved.config.simplicity_constraint);
    const auto reassociated_output_tag =
        make_full_tensor_binary_tag_reassociated_with_storage_order(
            spins_array,
            Dl,
            gamma,
            dimensions,
            reassociated_values,
            reassociated_storage_order(output_plan),
            resolved.config.simplicity_constraint);

    std::unique_ptr<TensorOutputLease> scratch_lease;
    std::unique_ptr<TemporaryTensorFile> scratch_file;
    Tensor<double, 1> reassociated;
    if (output_on_disk && use_reassociated) {
        reassociated = Tensor<double, 1>::create_mapped(
            publication->temporary_path(),
            {{output_plan.values_count}},
            TensorInit::zero,
            TensorLayout::FullTensorReassociatedReflectionCompactByI1,
            reassociated_output_tag);
    } else if (output_on_disk) {
        const std::filesystem::path scratch_directory =
            resolved.config.tensors.directory / ".scratch";
        scratch_lease = std::make_unique<TensorOutputLease>(
            scratch_directory / "gpu_reassociated");
        scratch_file = std::make_unique<TemporaryTensorFile>(
            scratch_directory);
        reassociated = Tensor<double, 1>::create_mapped(
            scratch_file->path(),
            {{output_plan.values_count}},
            TensorInit::zero,
            TensorLayout::FullTensorReassociatedReflectionCompactByI1,
            reassociated_output_tag);
    } else {
        reassociated = Tensor<double, 1>(
            {{output_plan.values_count}},
            TensorInit::zero,
            TensorLayout::FullTensorReassociatedReflectionCompactByI1);
        reassociated.set_tag(reassociated_output_tag);
    }

    FullTensor result;
    if (!use_reassociated) {
        result = output_on_disk
            ? FullTensor::create_mapped(
                  publication->temporary_path(),
                  dimensions,
                  TensorInit::zero,
                  TensorLayout::CanonicalDenseColumnMajor,
                  dense_output_tag)
            : FullTensor(
                  dimensions,
                  TensorInit::zero,
                  TensorLayout::CanonicalDenseColumnMajor);
    }
    const double output_allocate_seconds =
        seconds_since(output_allocate_start);

    sl2cfoam::gpu::production::FullTensorGpuOptions options;
    if (resolved.config.low_level.full_tensor_gpu_row_capacity != 0) {
        options.row_capacity =
            resolved.config.low_level.full_tensor_gpu_row_capacity;
    }
    options.producer_waves_per_sm =
        effective_full_tensor_gpu_producer_waves_per_sm(
            resolved.config.low_level
                .full_tensor_gpu_producer_waves_per_sm);
    if (channels.xvals.size() >
        std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("GPU x extent overflows uint32_t");
    }
    options.x_extent = static_cast<std::uint32_t>(channels.xvals.size());
    options.profiling.enabled = diagnostics.enabled(Verbosity::detailed);

    b4support::FullTensorReflectionCompactWeightedProgressOptions
        progress_options;
    progress_options.enabled =
        resolved.config.progress_bar ||
        static_cast<bool>(resolved.config.progress_callback);
    progress_options.render_output = resolved.config.progress_bar;
    progress_options.callback = resolved.config.progress_callback;
    progress_options.operation = ProgressOperation::full_tensor;
    progress_options.phase = ProgressPhase::contraction;
    progress_options.backend_kind = Backend::GPU;
    progress_options.update_granularity = 1;
    progress_options.label = "Full tensor GPU";
    progress_options.unit_label = "row work";
    progress_options.stream = resolved.config.log_stream != nullptr
        ? resolved.config.log_stream
        : &std::clog;
    if (progress_options.render_output) {
        progress_options.presentation = detect_output_presentation(
            resolved.config.log_stream != nullptr,
            resolved.config.printing_style);
        progress_options.backend = "GPU";
        progress_options.clear_on_success =
            resolved.config.verbosity == Verbosity::silent;
    }
    progress_options.eta_min_elapsed_seconds = 2.0;
    progress_options.eta_min_fraction = 0.01;
    b4support::FullTensorReflectionCompactWeightedProgressBar progress_bar(
        1, std::move(progress_options));
    if (progress_bar.enabled()) {
        options.progress_context = &progress_bar;
        options.progress_begin = [](void* context,
                                    std::uint64_t total_units) noexcept {
            if (total_units == 0) return;
            auto* progress = static_cast<
                b4support::FullTensorReflectionCompactWeightedProgressBar*>(
                    context);
            progress->set_total(total_units);
            progress->start();
        };
        options.progress_advance = [](void* context,
                                      std::uint64_t completed_units) noexcept {
            static_cast<
                b4support::FullTensorReflectionCompactWeightedProgressBar*>(
                    context)->add(completed_units);
        };
        options.progress_reset = [](void* context) noexcept {
            static_cast<
                b4support::FullTensorReflectionCompactWeightedProgressBar*>(
                    context)->reset();
        };
    }

    const auto contraction_start = Clock::now();
    const sl2cfoam::gpu::production::FullTensorGpuResult gpu_result =
        sl2cfoam::gpu::production::compute_full_tensor_gpu(
            spins,
            support,
            boosters,
            output_layout,
            reassociated.data(),
            reassociated.size(),
            true,
            options);
    progress_bar.stop();
    const double contraction_seconds = seconds_since(contraction_start);

    double canonical_scatter_seconds = 0.0;
    if (!use_reassociated) {
        const auto scatter_start = Clock::now();
        b4support::scatter_full_tensor_reassociated_auto_by_i1_to_canonical(
            support,
            output_plan,
            reassociated.data(),
            result.data());
        canonical_scatter_seconds = seconds_since(scatter_start);
    }

    if (gpu_result.profile.enabled) {
        const auto& profile = gpu_result.profile;
        diagnostics.log(
            Verbosity::detailed,
            "GPU profile devices=", profile.devices_used,
            " storage=", gpu_storage_name(gpu_result.storage),
            " peak_device_bytes=", profile.peak_device_bytes,
            " row_capacity=", profile.row_capacity,
            " i1_tile=", profile.i1_tile,
            " x_tile=", profile.x_tile,
            " l34_tile=", profile.l34_tile,
            " full_tensor_gpu_producer_waves_per_sm_request=",
            resolved.config.low_level
                .full_tensor_gpu_producer_waves_per_sm,
            " full_tensor_gpu_producer_waves_per_sm_effective=",
            profile.producer_waves_per_sm,
            " final_gemm_calls=", profile.final_gemm_calls,
            " initialization_seconds=", profile.initialization_seconds,
            " t1_seconds=", profile.t1_seconds,
            " t3_seconds=", profile.t3_seconds,
            " left_producer_seconds=", profile.left_producer_seconds,
            " right_producer_seconds=", profile.right_producer_seconds,
            " final_gemm_seconds=", profile.final_gemm_seconds,
            " compact_fold_seconds=", profile.compact_fold_seconds,
            " download_seconds=", profile.download_seconds,
            " gpu_total_seconds=", profile.total_seconds,
            " fallbacks=", gpu_result.fallback_log.failures().size());
    }

    double publication_seconds = 0.0;
    if (!output_on_disk) {
        if (use_reassociated) {
            result = FullTensor::from_reassociated(
                std::move(reassociated),
                dimensions,
                FullTensorReassociatedLayout::reflection_triangular_by_i1);
        } else {
            result.set_tag(dense_output_tag);
        }
    } else {
        const auto publication_start = Clock::now();
        if (use_reassociated) {
            reassociated.flush();
            reassociated.clear();
        } else {
            result.flush();
            result.clear();
        }
        publication->publish();
        result = FullTensor::map(
            resolved.output_path, TensorAccess::read_write);
        publication_seconds = seconds_since(publication_start);
    }

    diagnostics.timing(
        {
            {"cache lookup", "cache_lookup_seconds", cache_lookup_seconds},
            {"channels", "channels_seconds", channels_seconds},
            {"preflight", "preflight_seconds", preflight_seconds},
            {"support", "support_seconds", support_seconds},
            {"booster fill", "booster_fill_seconds", booster_fill_seconds},
            {"output allocation", "output_allocate_seconds",
             output_allocate_seconds},
            {"contraction", "contraction_seconds", contraction_seconds},
            {"canonical scatter", "canonical_scatter_seconds",
             canonical_scatter_seconds},
            {"publication", "publication_seconds", publication_seconds},
            {"total", "backend_total_seconds", seconds_since(backend_start)},
        },
        cache_status);
    return result;
#endif
}

} // namespace sl2cfoamxx::detail
