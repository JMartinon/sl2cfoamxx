// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "sl2cfoamxx/sl2cfoamxx.hpp"

#include "api/ConfigurationInternal.hpp"
#include "api/ErrorInternal.hpp"
#include "api/FullTensorCachePath.hpp"
#include "api/FullTensorInternal.hpp"
#include "api/InputValidation.hpp"
#include "core/LowLevelTuning.hpp"

#include <chrono>
#include <exception>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace sl2cfoamxx {
namespace {

using Clock = std::chrono::steady_clock;

[[nodiscard]] double seconds_since(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

[[nodiscard]] const char* storage_name(TensorStorage storage) noexcept
{
    return to_string(storage).data();
}

[[nodiscard]] constexpr bool is_disk_storage(
    TensorStorage storage) noexcept
{
    return storage == TensorStorage::on_disk_sym ||
           storage == TensorStorage::on_disk_dense;
}

[[nodiscard]] constexpr bool is_symmetry_storage(
    TensorStorage storage) noexcept
{
    return storage == TensorStorage::in_memory_sym ||
           storage == TensorStorage::on_disk_sym;
}

[[nodiscard]] const char* simplicity_name(
    SimplicityConstraint prescription) noexcept
{
    return prescription == SimplicityConstraint::gamma_j_p1
        ? "gamma_j_p1"
        : "gamma_j";
}

[[nodiscard]] detail::FullTensorConfigSnapshot resolve_config(
    const LibraryConfig& config,
    const Spins& spins,
    std::size_t Dl,
    double gamma,
    Backend backend)
{
    detail::FullTensorConfigSnapshot options;
    options.config = config;
    options.backend = backend;
    options.storage = options.config.tensors.storage;
    options.reuse_existing = options.config.tensors.reuse_existing;
    options.memory_budget_bytes = options.config.memory_budget_bytes;
    options.maximum_output_bytes =
        options.config.tensors.maximum_payload_size_bytes;

    switch (options.storage) {
    case TensorStorage::in_memory_sym:
    case TensorStorage::in_memory_dense:
        break;
    case TensorStorage::on_disk_sym:
    case TensorStorage::on_disk_dense:
        options.output_path = options.config.tensors.output_path.empty()
            ? detail::automatic_full_tensor_cache_path(
                  options.config,
                  spins,
                  Dl,
                  gamma,
                  is_symmetry_storage(options.storage))
            : options.config.tensors.output_path;
        options.output_path = std::filesystem::weakly_canonical(
            options.output_path);
        break;
    default:
        throw std::invalid_argument(
            std::string(detail::product_display_name) +
            ": compute_full_tensor: unknown storage mode");
    }
    return options;
}

[[nodiscard]] FullTensor compute_full_tensor_impl(
    const Spins& spins,
    std::size_t Dl,
    double gamma,
    Backend backend,
    const LibraryConfig& config);

} // namespace

FullTensor compute_full_tensor(
    const Spins& spins,
    std::size_t Dl,
    double gamma)
{
    detail::validate_full_tensor_inputs(spins, Dl, gamma);
    const LibraryConfig config = detail::configuration_snapshot();
    return compute_full_tensor_impl(
        spins, Dl, gamma, config.backend, config);
}

FullTensor compute_full_tensor(
    const Spins& spins,
    std::size_t Dl,
    double gamma,
    Backend backend)
{
    detail::validate_full_tensor_inputs(spins, Dl, gamma, backend);
    const LibraryConfig config = detail::configuration_snapshot();
    return compute_full_tensor_impl(spins, Dl, gamma, backend, config);
}

FullTensor compute_full_tensor(
    const Spins& spins,
    std::size_t Dl,
    double gamma,
    Backend backend,
    ProgressCallback progress_callback)
{
    detail::validate_full_tensor_inputs(spins, Dl, gamma, backend);
    LibraryConfig config = detail::configuration_snapshot();
    config.progress_callback = std::move(progress_callback);
    return compute_full_tensor_impl(spins, Dl, gamma, backend, config);
}

namespace {

FullTensor compute_full_tensor_impl(
    const Spins& spins,
    std::size_t Dl,
    double gamma,
    Backend backend,
    const LibraryConfig& config)
{
    const auto total_start = Clock::now();
    const detail::FullTensorConfigSnapshot options =
        resolve_config(config, spins, Dl, gamma, backend);
    if (options.backend != Backend::CPU && options.backend != Backend::GPU) {
        throw std::invalid_argument(
            std::string(detail::product_display_name) +
            ": compute_full_tensor: unknown backend");
    }
    const detail::Diagnostics diagnostics(
        options.config,
        detail::DiagnosticOperation::full_tensor,
        options.backend);
    diagnostics.started(
        " storage=", storage_name(options.storage),
        " Dl=", Dl,
        " gamma=", gamma,
        " simplicity=",
        simplicity_name(options.config.simplicity_constraint));
    diagnostics.log(
        Verbosity::detailed,
        "input spins=[",
        spins[0], ',', spins[1], ',', spins[2], ',', spins[3], ',', spins[4],
        ',', spins[5], ',', spins[6], ',', spins[7], ',', spins[8], ',',
        spins[9], "] cpu_threads=", options.config.cpu_threads,
        " simplicity_constraint=",
        simplicity_name(options.config.simplicity_constraint),
        " memory_budget_bytes=", options.memory_budget_bytes,
        " booster_rho_tile_request=",
        options.config.low_level.booster_rho_tile,
        " booster_rho_tile_effective=",
        detail::effective_booster_rho_tile(
            options.config.low_level.booster_rho_tile),
        " gpu_row_capacity_request=",
        options.config.low_level.full_tensor_gpu_row_capacity,
        " full_tensor_gpu_producer_waves_per_sm_request=",
        options.config.low_level
            .full_tensor_gpu_producer_waves_per_sm,
        " full_tensor_gpu_producer_waves_per_sm_effective=",
        detail::effective_full_tensor_gpu_producer_waves_per_sm(
            options.config.low_level
                .full_tensor_gpu_producer_waves_per_sm),
        " output_path=",
        options.output_path.empty() ? std::string("<memory>")
                                    : options.output_path.string());

    std::unique_ptr<detail::TensorOutputLease> output_lease;
    try {
        if (is_disk_storage(options.storage)) {
            output_lease = std::make_unique<detail::TensorOutputLease>(
                options.output_path);
        }

        FullTensor result;
        switch (options.backend) {
        case Backend::CPU:
            result = detail::compute_full_tensor_cpu(
                spins, Dl, gamma, options);
            break;
        case Backend::GPU:
            result = detail::compute_full_tensor_gpu(
                spins, Dl, gamma, options);
            break;
        default:
            throw std::invalid_argument(
                std::string(detail::product_display_name) +
                ": compute_full_tensor: unknown backend");
        }

        diagnostics.completed(
            seconds_since(total_start),
            " requested_storage=", storage_name(options.storage),
            " actual_storage=", storage_name(result.storage()),
            " stored_values=", result.stored_size(),
            " logical_values=", result.logical_size());
        return result;
    } catch (const std::exception& exception) {
        diagnostics.failed(seconds_since(total_start), exception.what());
        detail::rethrow_as_public_error(
            std::current_exception(),
            backend == Backend::GPU
                ? ErrorCode::backend_failure
                : ErrorCode::numerical_failure);
    } catch (...) {
        diagnostics.failed(
            seconds_since(total_start), "unknown non-standard exception");
        detail::rethrow_as_public_error(
            std::current_exception(), ErrorCode::internal_failure);
    }
}

} // namespace

} // namespace sl2cfoamxx
