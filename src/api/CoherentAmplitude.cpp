// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "sl2cfoamxx/sl2cfoamxx.hpp"

#include "api/CoherentBoundaryInternal.hpp"
#include "api/ConfigurationInternal.hpp"
#include "api/ErrorInternal.hpp"
#include "api/InputValidation.hpp"
#include "core/LowLevelTuning.hpp"
#include "amplitude/AmplitudeInputSupport_HalfSjC.hpp"
#include "amplitude/AmplitudePackedBoosters.hpp"
#include "amplitude/AmplitudePackedCoherentI1Batch.hpp"
#include "core/ProgressReporter.hpp"
#include "gpu/CoherentProductionGpuContraction.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <complex>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sl2cfoamxx {
namespace {

using DiagnosticsClock = std::chrono::steady_clock;

[[nodiscard]] double seconds_since(DiagnosticsClock::time_point start)
{
    return std::chrono::duration<double>(DiagnosticsClock::now() - start)
        .count();
}

template <std::size_t Size>
[[nodiscard]] std::string format_integer_array(
    const std::array<int, Size>& values)
{
    std::ostringstream output;
    output << '[';
    for (std::size_t index = 0; index < Size; ++index) {
        if (index != 0) output << ',';
        output << values[index];
    }
    return output.str() + ']';
}

[[nodiscard]] const char* backend_name(Backend backend) noexcept
{
    return backend == Backend::GPU ? "GPU" : "CPU";
}

[[nodiscard]] const char* simplicity_name(
    SimplicityConstraint prescription) noexcept
{
    return prescription == SimplicityConstraint::gamma_j_p1
        ? "gamma_j_p1"
        : "gamma_j";
}

[[nodiscard]] std::string failed_closure_nodes(
    const ClosureCheckResult& closure)
{
    std::ostringstream output;
    output << '[';
    bool first = true;
    for (std::size_t node = 0; node < closure.nodes.size(); ++node) {
        if (closure.nodes[node].closed) continue;
        if (!first) output << ',';
        output << node + 1;
        first = false;
    }
    return output.str() + ']';
}

[[nodiscard]] std::complex<double> compute_coherent_amplitude_impl(
    const Spins& spins_array,
    std::size_t Dl,
    double gamma,
    const BoundaryAngles* angles,
    const std::array<const CoherentCoefficients*, 5>& supplied_coefficients,
    Backend backend,
    const LibraryConfig& config);

} // namespace

std::complex<double> compute_coherent_amplitude(
    const Spins& spins_array,
    std::size_t Dl,
    double gamma,
    const BoundaryAngles& angles)
{
    detail::validate_coherent_amplitude_inputs(
        spins_array, Dl, gamma, angles);
    const LibraryConfig config = detail::configuration_snapshot();
    return compute_coherent_amplitude_impl(
        spins_array, Dl, gamma, &angles, {}, config.backend, config);
}

std::complex<double> compute_coherent_amplitude(
    const Spins& spins_array,
    std::size_t Dl,
    double gamma,
    const BoundaryAngles& angles,
    Backend backend)
{
    detail::validate_coherent_amplitude_inputs(
        spins_array, Dl, gamma, angles, backend);
    const LibraryConfig config = detail::configuration_snapshot();
    return compute_coherent_amplitude_impl(
        spins_array, Dl, gamma, &angles, {}, backend, config);
}

std::complex<double> compute_coherent_amplitude(
    const Spins& spins_array,
    std::size_t Dl,
    double gamma,
    const BoundaryAngles& angles,
    Backend backend,
    ProgressCallback progress_callback)
{
    detail::validate_coherent_amplitude_inputs(
        spins_array, Dl, gamma, angles, backend);
    LibraryConfig config = detail::configuration_snapshot();
    config.progress_callback = std::move(progress_callback);
    return compute_coherent_amplitude_impl(
        spins_array, Dl, gamma, &angles, {}, backend, config);
}

std::complex<double> compute_coherent_amplitude(
    const DSpins& spins,
    std::size_t Dl,
    double gamma,
    const CoherentCoefficients& C1,
    const CoherentCoefficients& C2,
    const CoherentCoefficients& C3,
    const CoherentCoefficients& C4,
    const CoherentCoefficients& C5)
{
    const std::array<const CoherentCoefficients*, 5> coefficients{
        &C1, &C2, &C3, &C4, &C5};
    detail::validate_coherent_amplitude_inputs(spins, Dl, gamma, coefficients);
    const LibraryConfig config = detail::configuration_snapshot();
    return compute_coherent_amplitude_impl(
        spins, Dl, gamma, nullptr, coefficients, config.backend, config);
}

std::complex<double> compute_coherent_amplitude(
    const DSpins& spins,
    std::size_t Dl,
    double gamma,
    const CoherentCoefficients& C1,
    const CoherentCoefficients& C2,
    const CoherentCoefficients& C3,
    const CoherentCoefficients& C4,
    const CoherentCoefficients& C5,
    Backend backend)
{
    const std::array<const CoherentCoefficients*, 5> coefficients{
        &C1, &C2, &C3, &C4, &C5};
    detail::validate_coherent_amplitude_inputs(
        spins, Dl, gamma, coefficients, backend);
    const LibraryConfig config = detail::configuration_snapshot();
    return compute_coherent_amplitude_impl(
        spins, Dl, gamma, nullptr, coefficients, backend, config);
}

std::complex<double> compute_coherent_amplitude(
    const DSpins& spins,
    std::size_t Dl,
    double gamma,
    const CoherentCoefficients& C1,
    const CoherentCoefficients& C2,
    const CoherentCoefficients& C3,
    const CoherentCoefficients& C4,
    const CoherentCoefficients& C5,
    Backend backend,
    ProgressCallback progress_callback)
{
    const std::array<const CoherentCoefficients*, 5> coefficients{
        &C1, &C2, &C3, &C4, &C5};
    detail::validate_coherent_amplitude_inputs(
        spins, Dl, gamma, coefficients, backend);
    LibraryConfig config = detail::configuration_snapshot();
    config.progress_callback = std::move(progress_callback);
    return compute_coherent_amplitude_impl(
        spins, Dl, gamma, nullptr, coefficients, backend, config);
}

namespace {

std::complex<double> compute_coherent_amplitude_impl(
    const Spins& spins_array,
    std::size_t Dl,
    double gamma,
    const BoundaryAngles* angles,
    const std::array<const CoherentCoefficients*, 5>& supplied_coefficients,
    Backend backend,
    const LibraryConfig& config)
{
    const auto total_start = DiagnosticsClock::now();

    const detail::Diagnostics diagnostics(
        config, detail::DiagnosticOperation::coherent_amplitude, backend);
    diagnostics.started(" Dl=", Dl,
                        " gamma=", std::setprecision(17), gamma,
                        " simplicity=",
                        simplicity_name(config.simplicity_constraint));
    try {
        if (backend == Backend::GPU) {
#if !defined(HEAVY_COMPUTE_USE_CUDA)
            throw Error(
                ErrorCode::backend_unavailable,
                std::string(detail::product_display_name) +
                ": compute_coherent_amplitude: GPU backend requires a build "
                "configured with ENABLE_CUDA=ON");
#else
            if (!b4gpu::coherent_cuda_backend_available()) {
                throw Error(
                    ErrorCode::backend_unavailable,
                    std::string(detail::product_display_name) +
                    ": compute_coherent_amplitude: no CUDA device is visible");
            }
#endif
        }

        diagnostics.log(Verbosity::detailed,
                        "inputs spins=", format_integer_array(spins_array),
                        " Dl=", Dl, " gamma=", std::setprecision(17), gamma,
                        " backend=", backend_name(backend),
                        " simplicity_constraint=",
                        simplicity_name(config.simplicity_constraint),
                        " cpu_threads=", config.cpu_threads,
                        " booster_rho_tile_request=",
                        config.low_level.booster_rho_tile,
                        " booster_rho_tile_effective=",
                        detail::effective_booster_rho_tile(
                            config.low_level.booster_rho_tile),
                        " coherent_gpu_pipeline_slots_request=",
                        config.low_level.coherent_gpu_pipeline_slots,
                        " closure_check=",
                        angles == nullptr ? "not-applicable" :
                            (config.coherent_amplitude.check_closure
                                ? "enabled" : "disabled"),
                        " coefficient_source=",
                        angles == nullptr ? "caller" : "angles",
                        " pruning_ratio=", std::scientific,
                        std::setprecision(17),
                        config.coherent_amplitude.pruning_ratio);

        if (angles != nullptr && config.coherent_amplitude.check_closure) {
            const ClosureCheckResult closure = check_closure(
                spins_array, *angles);
            if (!closure.all_closed) {
                double maximum_residual = 0.0;
                double maximum_relative_residual = 0.0;
                for (const ClosureNodeResult& node : closure.nodes) {
                    if (node.closed) continue;
                    maximum_residual = std::max(
                        maximum_residual, node.residual_norm);
                    maximum_relative_residual = std::max(
                        maximum_relative_residual,
                        node.relative_residual);
                }
                diagnostics.warning(
                    "closure",
                    " failed_nodes=", failed_closure_nodes(closure),
                    " max_residual=", std::scientific,
                    std::setprecision(6), maximum_residual,
                    " max_relative_residual=",
                    maximum_relative_residual,
                    " action=continuing");
            } else {
                diagnostics.log(
                    Verbosity::detailed,
                    "closure check=passed nodes=5 absolute_tolerance=1e-12 "
                    "relative_tolerance=1e-12");
            }
        }

        const auto setup_start = DiagnosticsClock::now();
        const B4Spins spins(spins_array);
        const FullTensorChannels channels =
            make_FullTensorChannels(spins, static_cast<int>(Dl));
        if (channels.i1.empty() || channels.i2.empty() ||
            channels.i3.empty() || channels.i4.empty() ||
            channels.i5.empty() || channels.k2.empty() ||
            channels.k3.empty() || channels.k4.empty() ||
            channels.k5.empty() || channels.xvals.empty()) {
            const double setup_seconds = seconds_since(setup_start);
            const double total_seconds = seconds_since(total_start);
            diagnostics.completed(
                total_seconds, " result=", std::scientific,
                std::setprecision(17), '(', 0.0, ',', 0.0, ')');
            diagnostics.timing({
                {"setup", "setup", setup_seconds},
                {"coherent coefficients", "coherent coefficients", 0.0},
                {"support", "support", 0.0},
                {"boosters", "boosters", 0.0},
                {"6j tensors", "6j tensors", 0.0},
                {"contraction", "contraction", 0.0},
                {"total", "total", total_seconds},
            });
            diagnostics.log(
                Verbosity::detailed,
                "execution mode=selection-rule-zero; the boundary has no "
                "common internal recoupling support");
            return {0.0, 0.0};
        }

        const std::size_t lifted_dimension = Dl + 1;
        const std::array<std::size_t, 6> lifted_extents{
            {lifted_dimension, lifted_dimension, lifted_dimension,
             lifted_dimension, lifted_dimension, lifted_dimension}};

        const std::vector<double> factorials = LogFactTable(
            detail::log_factorial_maximum(spins_array, Dl));

        const double setup_seconds = seconds_since(setup_start);

        const auto support_start = DiagnosticsClock::now();
        const auto support =
            b4support::build_amplitude_static_booster_support_plan_half_sjC(
                spins, static_cast<int>(Dl), channels);
        const double support_seconds = seconds_since(support_start);

        std::array<double, 5> coefficient_seconds{};
        const std::array<const std::vector<int>*, 5> intertwiner_labels{
            &channels.i1, &channels.i2, &channels.i3, &channels.i4, &channels.i5};
        const auto compute_coefficients =
            [&](std::size_t index) {
                const auto start = DiagnosticsClock::now();
                coherent_intertwiner::Result result{};
                if (angles != nullptr) {
                    result = detail::compute_coherent_coefficients_internal(
                        detail::coherent_node_spins(spins_array, index),
                        detail::coherent_node_angles(*angles, index),
                        config);
                } else {
                    result.doubled_iotas = *intertwiner_labels[index];
                    result.coefficients.reserve(
                        supplied_coefficients[index]->size());
                    for (const auto value : *supplied_coefficients[index]) {
                        result.coefficients.push_back(
                            {value.real(), value.imag()});
                    }
                }
                coefficient_seconds[index] = seconds_since(start);
                return result;
            };
        const auto coefficients_start = DiagnosticsClock::now();
        const coherent_intertwiner::Result C1 =
            compute_coefficients(0);
        const coherent_intertwiner::Result C2 =
            compute_coefficients(1);
        const coherent_intertwiner::Result C3 =
            compute_coefficients(2);
        const coherent_intertwiner::Result C4 =
            compute_coefficients(3);
        const coherent_intertwiner::Result C5 =
            compute_coefficients(4);
        const double coefficients_seconds = seconds_since(coefficients_start);

        const B4CoherentCoefficientsView C1_view =
            make_b4_coherent_coefficients_view(C1.doubled_iotas,
                                               C1.coefficients);
        const B4CoherentCoefficientsView C2_view =
            make_b4_coherent_coefficients_view(C2.doubled_iotas,
                                               C2.coefficients);
        const B4CoherentCoefficientsView C3_view =
            make_b4_coherent_coefficients_view(C3.doubled_iotas,
                                               C3.coefficients);
        const B4CoherentCoefficientsView C4_view =
            make_b4_coherent_coefficients_view(C4.doubled_iotas,
                                               C4.coefficients);
        const B4CoherentCoefficientsView C5_view =
            make_b4_coherent_coefficients_view(C5.doubled_iotas,
                                               C5.coefficients);

        std::unique_ptr<detail::ProgressReporter> progress;
        if (config.progress_bar || config.progress_callback) {
            detail::ProgressReporterOptions progress_options;
            progress_options.enabled = true;
            progress_options.render_output = config.progress_bar;
            progress_options.callback = config.progress_callback;
            progress_options.operation = ProgressOperation::coherent_amplitude;
            progress_options.phase = ProgressPhase::contraction;
            progress_options.backend_kind = backend;
            progress_options.update_granularity = 1;
            progress_options.label = backend == Backend::GPU
                ? "Coherent amplitude GPU"
                : "Coherent amplitude CPU";
            progress_options.unit_label = backend == Backend::GPU
                ? "x slices"
                : "active pairs";
            progress_options.stream = config.log_stream != nullptr
                ? config.log_stream
                : &std::clog;
            if (progress_options.render_output) {
                progress_options.presentation =
                    detail::detect_output_presentation(
                        config.log_stream != nullptr,
                        config.printing_style);
            }
            progress_options.backend = backend_name(backend);
            progress_options.operation_label = "coherent amplitude";
            progress_options.phase_label = "contract";
            progress_options.clear_on_success =
                config.verbosity == Verbosity::silent;
            progress_options.eta_min_elapsed_seconds = 2.0;
            progress_options.eta_min_fraction = 0.01;
            progress = std::make_unique<detail::ProgressReporter>(
                1, std::move(progress_options));
        }
        const auto progress_begin = [](void* context,
                                       std::uint64_t total) noexcept {
            if (total == 0) return;
            auto* reporter = static_cast<detail::ProgressReporter*>(context);
            reporter->set_total(total);
            reporter->start();
        };
        const auto progress_advance = [](void* context,
                                         std::uint64_t units) noexcept {
            static_cast<detail::ProgressReporter*>(context)->add(units);
        };
        const auto progress_reset = [](void* context) noexcept {
            static_cast<detail::ProgressReporter*>(context)->reset();
        };

        b4utils::Complex64 amplitude{0.0, 0.0};
        double booster_seconds = 0.0;
        double sj_tensor_seconds = 0.0;
        double contraction_seconds = 0.0;
        std::string execution_details;
        b4support::CoherentBoosterProductionStats booster_stats;
        auto* measured_boosters = diagnostics.enabled(Verbosity::detailed)
            ? &booster_stats : nullptr;
        switch (backend) {
        case Backend::CPU: {
            const auto booster_start = DiagnosticsClock::now();
            b4support::PackedAmplitudeCoherentBoosters boosters;
            b4support::fill_packed_amplitude_coherent_boosters_direct(
                spins, lifted_extents, gamma, factorials, support,
                C2_view, C3_view, C4_view, C5_view, boosters, 0, measured_boosters,
                config.simplicity_constraint,
                config.low_level.booster_rho_tile);
            booster_seconds = seconds_since(booster_start);

            b4support::PackedCoherentC1SharedOptions options;
            options.xq.memory_budget_bytes = config.memory_budget_bytes;
            options.xq.output_bytes = sizeof(b4utils::Complex64);
            options.C1_prune_relative_tol =
                config.coherent_amplitude.pruning_ratio;

            // Historical defaults were tuned for standalone debugging. Keep the
            // library path deterministic and quiet, then report selected fields
            // through the thread-safe public diagnostics layer below.
            options.print_plan = false;
            options.print_i1_timing = false;
            options.print_memory = false;
            options.print_rss = false;
            options.print_C1_prune_summary = false;
            options.print_streamed_i1_summary = false;
            options.print_xq_slab_summary = false;
            options.print_x_mask = false;
            options.xq.print_plan = false;
            options.xq.print_x_timing = false;
            options.xq.print_q_timing = false;
            options.xq.print_contraction_stats = false;
            if (progress) {
                options.progress_context = progress.get();
                options.progress_begin = progress_begin;
                options.progress_advance = progress_advance;
            }

            b4support::PackedContractionOptions contraction_options;
            const auto contraction_start = DiagnosticsClock::now();
            const b4support::PackedCoherentC1SharedResult result =
                b4support::contract_packed_amplitude_coherent_C1_shared_auto(
                    spins, static_cast<int>(Dl), spins_array, boosters,
                    C1.doubled_iotas, C1.coefficients, options,
                    b4support::WingExecutionOrder::LeftThenRight,
                    contraction_options, nullptr);
            contraction_seconds = seconds_since(contraction_start);
            sj_tensor_seconds = result.sj_fill_seconds;
            contraction_seconds = std::max(
                0.0, contraction_seconds - sj_tensor_seconds);
            amplitude = result.amplitude;

            std::ostringstream details;
            const char* mode =
                result.full_x_slabs != 0
                    ? "full-x"
                    : (result.x_slabs != 0 ? "x-slab" : "x+q-slab");
            details << "execution backend=CPU mode=" << mode
                    << " active_i1=" << result.active_i1
                    << " pruning_ratio=" << std::scientific
                    << std::setprecision(6)
                    << config.coherent_amplitude.pruning_ratio
                    << " C1_kept=" << result.C1_kept_count << '/'
                    << result.C1_original_count
                    << " full_x_slabs=" << result.full_x_slabs
                    << " x_slabs=" << result.x_slabs
                    << " xq_slices=" << result.xq_slices
                    << " q_slabs=" << result.xq_q_slabs_total
                    << " peak_bytes=" << result.estimated_peak_bytes
                    << " internal_support=" << std::fixed
                    << std::setprecision(6) << result.support_seconds << " s"
                    << " internal_sj_fill=" << result.sj_fill_seconds << " s"
                    << " internal_contract=" << result.contract_seconds << " s";
            execution_details = details.str();
            break;
        }
        case Backend::GPU: {
            const auto worker_diagnostic = [](
                const void* context,
                const b4gpu::DenseContractionWorkerDiagnostics& worker) noexcept {
                const auto& log = *static_cast<const detail::Diagnostics*>(context);
                log.log(
                    Verbosity::detailed,
                    "GPU worker device=", worker.device,
                    " thread_id=", worker.thread_id,
                    " observed_cpu=", worker.cpu,
                    " cpus_allowed=", worker.cpus_allowed_list,
                    " mems_allowed=", worker.mems_allowed_list,
                    " x_slices=", worker.x_slices,
                    " q_tiles=", worker.q_tiles,
                    " r_tile=", worker.r_tile,
                    " peak_device_bytes=", worker.peak_device_bytes,
                    " upload_prepare_seconds=", worker.upload_seconds,
                    " contraction_seconds=", worker.contraction_seconds);
            };
            const auto booster_start = DiagnosticsClock::now();
            b4support::CoherentQSlabNativeBoosters boosters;
            b4support::fill_q_slab_native_amplitude_coherent_boosters_direct(
                spins, lifted_extents, gamma, factorials, support,
                C2_view, C3_view, C4_view, C5_view, boosters, 0, measured_boosters,
                config.simplicity_constraint,
                config.low_level.booster_rho_tile);
            booster_seconds = seconds_since(booster_start);
            const auto contraction_start = DiagnosticsClock::now();
            const b4support::CoherentProductionGpuResult gpu_result =
                b4support::contract_packed_coherent_production_on_gpus(
                    spins, static_cast<int>(Dl), support, boosters, C1_view,
                    config.coherent_amplitude.pruning_ratio,
                    config.low_level.coherent_gpu_pipeline_slots,
                    progress ? +progress_begin : nullptr,
                    progress ? +progress_advance : nullptr,
                    progress ? +progress_reset : nullptr,
                    progress.get(),
                    diagnostics.enabled(Verbosity::detailed)
                        ? +worker_diagnostic : nullptr,
                    &diagnostics);
            amplitude = gpu_result.amplitude;
            contraction_seconds = seconds_since(contraction_start);
            if (diagnostics.enabled(Verbosity::detailed)) {
                const auto& measured = gpu_result.diagnostics;
                // These existing host counters do not enable CUDA-event
                // profiling. Per-device maxima can overlap on several GPUs;
                // they are not additive phases of the total wall time.
                diagnostics.log(
                    Verbosity::detailed,
                    "GPU diagnostics devices=", measured.devices_used,
                    " x_slices=", measured.x_slices_completed,
                    " q_tiles=", measured.q_tiles_completed,
                    " r_tile_min=", measured.r_tile_min,
                    " r_tile_max=", measured.r_tile_max,
                    " peak_device_bytes=", measured.peak_device_bytes,
                    " upload_prepare_max_seconds=", measured.upload_seconds,
                    " contraction_max_seconds=", measured.contraction_seconds,
                    " gpu_total_seconds=", measured.total_seconds,
                    " worker_records=", measured.worker_records);
            }
            std::ostringstream details;
            details
                << "execution backend=GPU route=production-multi-GPU "
                   "profiling=disabled"
                << " coherent_gpu_pipeline_slots_request="
                << config.low_level.coherent_gpu_pipeline_slots
                << " coherent_gpu_pipeline_slots_effective="
                << gpu_result.pipeline_slots
                << " pruning_ratio=" << std::scientific
                << std::setprecision(6)
                << config.coherent_amplitude.pruning_ratio
                << " C1_kept=" << gpu_result.c1_kept_count << '/'
                << gpu_result.c1_original_count;
            execution_details = details.str();
            break;
        }
        default:
            throw std::invalid_argument(
                std::string(detail::product_display_name) +
                ": compute_coherent_amplitude: unknown backend");
        }
        if (progress) progress->stop();

        if (measured_boosters != nullptr) {
            diagnostics.log(
                Verbosity::detailed, "booster diagnostics harvest_calls=",
                booster_stats.harvest_threads.calls,
                " threads_min=", booster_stats.harvest_threads.minimum,
                " threads_max=", booster_stats.harvest_threads.maximum);
        }

        const double total_seconds = seconds_since(total_start);
        diagnostics.completed(
            total_seconds, " result=", std::scientific,
            std::setprecision(17), '(', amplitude.re, ',', amplitude.im, ')');
        if (backend == Backend::CPU) {
            diagnostics.timing({
                {"setup", "setup", setup_seconds},
                {"coherent coefficients", "coherent coefficients",
                 coefficients_seconds},
                {"support", "support", support_seconds},
                {"boosters", "boosters", booster_seconds},
                {"6j tensors", "6j tensors", sj_tensor_seconds},
                {"contraction", "contraction", contraction_seconds},
                {"total", "total", total_seconds},
            });
        } else {
            // Device-side recurrence is interleaved with the CUDA contraction;
            // without profiling events it cannot be reported as an isolated
            // phase without perturbing the production path.
            diagnostics.timing({
                {"setup", "setup", setup_seconds},
                {"coherent coefficients", "coherent coefficients",
                 coefficients_seconds},
                {"support", "support", support_seconds},
                {"boosters", "boosters", booster_seconds},
                {"contraction", "contraction", contraction_seconds},
                {"total", "total", total_seconds},
            });
        }
        diagnostics.log(
            Verbosity::detailed, "coherent coefficient timing C1=", std::fixed,
            std::setprecision(6), coefficient_seconds[0],
            " s C2=", coefficient_seconds[1], " s C3=", coefficient_seconds[2],
            " s C4=", coefficient_seconds[3], " s C5=", coefficient_seconds[4],
            " s sizes=", C1.coefficients.size(), ',', C2.coefficients.size(),
            ',', C3.coefficients.size(), ',', C4.coefficients.size(), ',',
            C5.coefficients.size());
        diagnostics.log(Verbosity::detailed, execution_details);

        return {amplitude.re, amplitude.im};
    } catch (const std::exception& exception) {
        diagnostics.failed(seconds_since(total_start), exception.what());
        detail::rethrow_as_public_error(
            std::current_exception(),
            backend == Backend::GPU
                ? ErrorCode::backend_failure
                : ErrorCode::numerical_failure);
    } catch (...) {
        diagnostics.failed(seconds_since(total_start), "unknown exception");
        detail::rethrow_as_public_error(
            std::current_exception(), ErrorCode::internal_failure);
    }
}

} // namespace
} // namespace sl2cfoamxx
