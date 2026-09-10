// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "api/FullTensorInternal.hpp"

#include "api/ConfigurationInternal.hpp"
#include "api/InputValidation.hpp"
#include "core/LowLevelTuning.hpp"
#include "full_tensor/FullTensorExecutionPlanner.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sl2cfoamxx::detail {
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

[[nodiscard]] constexpr b4support::FullTensorTriangularFinalTileShape
internal_full_tensor_cpu_final_tile(
    FullTensorCpuFinalTile requested) noexcept
{
    switch (effective_full_tensor_cpu_final_tile(requested)) {
    case FullTensorCpuFinalTile::tile_4x4:
        return b4support::FullTensorTriangularFinalTileShape::Tile4x4;
    case FullTensorCpuFinalTile::tile_4x8:
        return b4support::FullTensorTriangularFinalTileShape::Tile4x8;
    case FullTensorCpuFinalTile::tile_8x4:
        return b4support::FullTensorTriangularFinalTileShape::Tile8x4;
    case FullTensorCpuFinalTile::automatic:
        break;
    }
    return b4support::FullTensorTriangularFinalTileShape::Tile4x4;
}

} // namespace

FullTensor compute_full_tensor_cpu(
    const Spins& spins_array,
    std::size_t Dl,
    double gamma,
    const FullTensorConfigSnapshot& resolved)
{
    const auto backend_start = Clock::now();
    const Diagnostics diagnostics(
        resolved.config, DiagnosticOperation::full_tensor, Backend::CPU);

    const B4Spins spins(spins_array);
    const FullTensorChannels channels = make_FullTensorChannels(spins, Dl);
    const FullTensor::shape_type dimensions{{
        channels.i1.size(), channels.i2.size(), channels.i3.size(),
        channels.i4.size(), channels.i5.size()}};

    // The low-level contractor always uses a reassociated work buffer.  It is
    // a genuine space-saving public representation only when the reflection
    // plan is triangular; asymmetric inputs keep the requested backing but
    // fall back to canonical dense output.
    const b4support::FullTensorReassociatedAutoOutputPlan output_plan =
        b4support::full_tensor_expected_reassociated_output_plan_from_channels(
            spins_array, channels);
    const bool symmetry_is_available = output_plan.uses_reflection_compact();
    const TensorStorage storage =
        effective_storage(resolved.storage, symmetry_is_available);
    const bool use_reassociated = requests_symmetry_storage(storage);
    const bool output_on_disk = is_disk_storage(storage);

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
    const std::uint64_t stored_values = use_reassociated
        ? static_cast<std::uint64_t>(output_plan.values_count)
        : output_values;
    if (stored_values > std::numeric_limits<std::uint64_t>::max() /
                            sizeof(double)) {
        throw std::overflow_error(
            "full-tensor stored payload bytes overflow uint64_t");
    }
    const std::uint64_t stored_bytes = stored_values * sizeof(double);
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
        if (resolved.reuse_existing) {
            cache_status = "miss";
            const bool exists =
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
                            b4support::full_tensor_reassociated_storage_order_for_output_plan(
                                output_plan),
                            resolved.config.simplicity_constraint)
                        : make_full_tensor_binary_tag_canonical(
                            spins_array,
                            Dl,
                            gamma,
                            dimensions,
                            output_values,
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

    b4support::FullTensorAutoOptions options;
    options.simplicity_constraint =
        resolved.config.simplicity_constraint;
    options.booster_rho_tile =
        resolved.config.low_level.booster_rho_tile;
    options.triangular_options.p_batch =
        effective_full_tensor_cpu_p_batch(
            resolved.config.low_level.full_tensor_cpu_p_batch);
    options.triangular_options.final_tile_shape =
        internal_full_tensor_cpu_final_tile(
            resolved.config.low_level.full_tensor_cpu_final_tile);
    if (use_reassociated) {
        options.output_mode = output_on_disk
            ? b4support::FullTensorOutputMode::ReassociatedOnDisk
            : b4support::FullTensorOutputMode::ReassociatedOnly;
    } else {
        options.output_mode = output_on_disk
            ? b4support::FullTensorOutputMode::CanonicalDenseOnDisk
            : b4support::FullTensorOutputMode::CanonicalDenseInMemory;
    }
    options.save_reassociated_auto_to_disk = false;
    options.reuse_existing_disk_output = false;
    options.force_recompute_existing_output =
        output_on_disk;
    options.print_existing_output_status = false;
    options.print_preflight = false;
    options.print_execution_plan = false;
    options.print_streaming_plan = false;
    options.print_timing = false;
    options.print_result_summary = false;
    options.progress_options.enabled =
        resolved.config.progress_bar ||
        static_cast<bool>(resolved.config.progress_callback);
    options.progress_options.render_output = resolved.config.progress_bar;
    options.progress_options.callback = resolved.config.progress_callback;
    options.progress_options.operation = ProgressOperation::full_tensor;
    options.progress_options.phase = ProgressPhase::contraction;
    options.progress_options.backend_kind = Backend::CPU;
    options.progress_options.stream = resolved.config.log_stream != nullptr
        ? resolved.config.log_stream
        : &std::clog;
    if (options.progress_options.render_output) {
        options.progress_options.presentation = detect_output_presentation(
            resolved.config.log_stream != nullptr,
            resolved.config.printing_style);
        options.progress_options.backend = "CPU";
        options.progress_options.clear_on_success =
            resolved.config.verbosity == Verbosity::silent;
    }
    options.keep_reassociated_output = false;
    options.memory_budget_bytes = resolved.memory_budget_bytes;
    options.planner.max_canonical_output_bytes =
        resolved.maximum_output_bytes == 0
            ? std::numeric_limits<std::uint64_t>::max()
            : resolved.maximum_output_bytes;
    options.max_canonical_output_disk_bytes =
        resolved.maximum_output_bytes;
    options.max_reassociated_output_disk_bytes =
        resolved.maximum_output_bytes;
    if (output_on_disk) {
        options.output_root = publication->temporary_path();
        options.canonical_output_suffix.clear();
    }

    const std::vector<double> factorials = LogFactTable(
        log_factorial_maximum(spins_array, Dl));
    b4support::FullTensorAutoResult internal =
        b4support::compute_full_tensor_auto(
            spins_array,
            Dl,
            gamma,
            factorials,
            options);

    diagnostics.log(
        Verbosity::detailed,
        "CPU contraction tuning p_batch_request=",
        resolved.config.low_level.full_tensor_cpu_p_batch,
        " p_batch_effective=",
        internal.schedule_balance.selected_p_batch,
        " final_tile_request=",
        full_tensor_cpu_final_tile_name(
            resolved.config.low_level.full_tensor_cpu_final_tile),
        " final_tile_effective=",
        internal.schedule_balance.selected_final_tile_applicable
            ? b4support::full_tensor_triangular_final_tile_shape_name(
                  internal.schedule_balance.selected_final_tile_shape)
            : "not_applicable");

    diagnostics.log(
        Verbosity::detailed,
        "planner mode=",
        b4support::full_tensor_planner_mode_name(
            internal.execution_plan.mode),
        " memory_budget_bytes=",
        internal.execution_plan.memory_budget_bytes,
        " resident_fast_peak_bytes=",
        internal.execution_plan.resident_fast_peak_bytes,
        " resident_boosters_only_peak_bytes=",
        internal.execution_plan.resident_boosters_only_peak_bytes,
        " support_metadata_bytes=",
        internal.execution_plan.support_metadata_bytes,
        " sj_payload_bytes=",
        internal.execution_plan.sj_payload_bytes,
        " booster_payload_bytes=",
        internal.execution_plan.booster_payload_bytes,
        " cpu_batched_workspace_bytes=",
        internal.execution_plan.cpu_batched_workspace_bytes,
        " boundary_tiles=", internal.boundary_tiles_completed,
        " sj_slab_builds=", internal.sj_slab_builds,
        " booster_leg_builds=", internal.booster_leg_builds);

    double publication_seconds = 0.0;
    FullTensor result;
    if (output_on_disk) {
        const bool materialized = use_reassociated
            ? internal.reassociated_materialized_on_disk
            : internal.canonical_materialized_on_disk;
        const std::filesystem::path& materialized_path = use_reassociated
            ? internal.reassociated_disk_root
            : internal.canonical_disk_root;
        if (!materialized || materialized_path.empty()) {
            throw Error(
                ErrorCode::internal_failure,
                "CPU full-tensor backend did not materialize mapped output");
        }
        const auto publication_start = Clock::now();
        publication->publish();
        result = FullTensor::map(
            resolved.output_path, TensorAccess::read_write);
        publication_seconds = seconds_since(publication_start);
    } else if (use_reassociated) {
        if (!internal.reassociated_materialized ||
            internal.reassociated_output.empty()) {
            throw Error(
                ErrorCode::internal_failure,
                "CPU full-tensor backend did not materialize compact output");
        }
        // The CPU backend already owns an aligned, parallel-first-touched
        // tensor. Transfer it directly instead of allocating and serially
        // copying another output-sized payload during API publication.
        Tensor<double, 1> reassociated =
            std::move(internal.reassociated_output);
        reassociated.set_layout(
            TensorLayout::FullTensorReassociatedReflectionCompactByI1);
        reassociated.set_tag(
            make_full_tensor_binary_tag_reassociated_with_storage_order(
                spins_array,
                Dl,
                gamma,
                dimensions,
                reassociated.size(),
                b4support::full_tensor_reassociated_storage_order_for_output_plan(
                    internal.output_plan),
                resolved.config.simplicity_constraint));
        result = FullTensor::from_reassociated(
            std::move(reassociated),
            dimensions,
            FullTensorReassociatedLayout::reflection_triangular_by_i1);
    } else {
        if (!internal.canonical_materialized ||
            internal.canonical_output.empty()) {
            throw Error(
                ErrorCode::internal_failure,
                "CPU full-tensor backend did not materialize canonical output");
        }

        internal.canonical_output.set_layout(
            TensorLayout::CanonicalDenseColumnMajor);
        internal.canonical_output.set_tag(
            make_full_tensor_binary_tag_canonical(
                spins_array,
                Dl,
                gamma,
                internal.canonical_output.shape(),
                internal.canonical_output.size(),
                resolved.config.simplicity_constraint));
        result = std::move(internal.canonical_output);
    }

    const b4support::FullTensorAutoTimings& timings = internal.timings;
    diagnostics.timing(
        {
            {"cache lookup", "cache_lookup_seconds", cache_lookup_seconds},
            {"channels", "channels_seconds", timings.channels_seconds},
            {"preflight", "preflight_seconds", timings.preflight_seconds},
            {"support", "support_seconds", timings.support_seconds},
            {"streamed support", "streamed_support_seconds",
             timings.streamed_support_seconds},
            {"planning", "planning_seconds", timings.planning_seconds},
            {"6j fill", "sj_fill_seconds", timings.sj_fill_seconds},
            {"booster fill", "booster_fill_seconds",
             timings.booster_fill_seconds},
            {"output allocation", "output_allocate_seconds",
             timings.output_allocate_seconds},
            {"contraction", "contraction_seconds",
             timings.contraction_seconds},
            {"canonical scatter", "canonical_scatter_seconds",
             timings.canonical_scatter_seconds},
            {"publication", "publication_seconds", publication_seconds},
            {"total", "backend_total_seconds", seconds_since(backend_start)},
        },
        cache_status);
    return result;
}

} // namespace sl2cfoamxx::detail
