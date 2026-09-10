// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "sl2cfoamxx/sl2cfoamxx.hpp"

#include "api/ConfigurationInternal.hpp"
#include "api/ErrorInternal.hpp"
#include "api/InputValidation.hpp"
#include "amplitude/AmplitudeInputSupport_HalfSjC.hpp"
#include "amplitude/AmplitudePackedAutoSlab.hpp"
#include "amplitude/AmplitudePackedBoosters.hpp"
#include "core/LowLevelTuning.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace sl2cfoamxx {
namespace {

using DiagnosticsClock = std::chrono::steady_clock;

[[nodiscard]] double seconds_since(DiagnosticsClock::time_point start)
{
    return std::chrono::duration<double>(DiagnosticsClock::now() - start)
        .count();
}

[[nodiscard]] const char* simplicity_name(
    SimplicityConstraint prescription) noexcept
{
    return prescription == SimplicityConstraint::gamma_j_p1
        ? "gamma_j_p1"
        : "gamma_j";
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

[[nodiscard]] double sixj_tensor_seconds(
    const b4support::PackedAmplitudeAutoResult& result) noexcept
{
    switch (result.mode) {
    case b4support::PackedAmplitudeAutoMode::FullX:
    case b4support::PackedAmplitudeAutoMode::XSlab:
        return result.xslab_result.timings.sj_fill;
    case b4support::PackedAmplitudeAutoMode::XQSlab:
        return result.xqslab_result.timings.wing_fill +
               result.xqslab_result.timings.sjC_fill;
    }
    return 0.0;
}

} // namespace

double compute_amplitude(
    const Spins& spins_array,
    const Intertwiners& intertwiners,
    std::size_t Dl,
    double gamma)
{
    detail::validate_amplitude_inputs(
        spins_array, intertwiners, Dl, gamma);
    const auto total_start = DiagnosticsClock::now();
    const LibraryConfig config = detail::configuration_snapshot();
    const detail::Diagnostics diagnostics(
        config, detail::DiagnosticOperation::amplitude, Backend::CPU);
    diagnostics.started(" Dl=", Dl,
                        " gamma=", std::setprecision(17), gamma,
                        " simplicity=",
                        simplicity_name(config.simplicity_constraint));
    diagnostics.log(Verbosity::detailed,
                    "inputs spins=", format_integer_array(spins_array),
                    " intertwiners=", format_integer_array(intertwiners),
                    " Dl=", Dl, " gamma=", std::setprecision(17), gamma,
                    " cpu_threads=", config.cpu_threads,
                    " booster_rho_tile_request=",
                    config.low_level.booster_rho_tile,
                    " booster_rho_tile_effective=",
                    detail::effective_booster_rho_tile(
                        config.low_level.booster_rho_tile),
                    " simplicity_constraint=",
                    simplicity_name(config.simplicity_constraint));

    try {
        const auto setup_start = DiagnosticsClock::now();
        const B4Spins spins(spins_array);
        const AmplitudeChannels channels =
            make_AmplitudeChannels(spins, static_cast<int>(Dl), intertwiners);
        if (channels.k2.empty() || channels.k3.empty() ||
            channels.k4.empty() || channels.k5.empty() ||
            channels.xvals.empty()) {
            const double setup_seconds = seconds_since(setup_start);
            const double total_seconds = seconds_since(total_start);
            diagnostics.completed(
                total_seconds, " result=", std::scientific,
                std::setprecision(17), 0.0);
            diagnostics.timing({
                {"setup/support", "setup/support", setup_seconds},
                {"boosters", "boosters", 0.0},
                {"6j tensors", "6j tensors", 0.0},
                {"contraction", "contraction", 0.0},
                {"total", "total", total_seconds},
            });
            diagnostics.log(
                Verbosity::detailed,
                "execution mode=selection-rule-zero; locally admissible "
                "boundary labels have no common internal recoupling support");
            return 0.0;
        }

        const std::size_t lifted_dimension = Dl + 1;
        const std::array<std::size_t, 6> lifted_extents{
            {lifted_dimension, lifted_dimension, lifted_dimension,
             lifted_dimension, lifted_dimension, lifted_dimension}};

        const std::vector<double> factorials = LogFactTable(
            detail::log_factorial_maximum(spins_array, Dl));

        const auto support =
            b4support::build_amplitude_static_booster_support_plan_half_sjC(
                spins, static_cast<int>(Dl), channels);
        const double setup_seconds = seconds_since(setup_start);

        const auto booster_start = DiagnosticsClock::now();
        b4support::PackedAmplitudeBoosters boosters;
        b4support::fill_packed_amplitude_boosters_via_canonical(
            spins, lifted_extents, gamma, factorials, support,
            boosters, config.simplicity_constraint,
            config.low_level.booster_rho_tile);
        const double booster_seconds = seconds_since(booster_start);

        b4support::PackedAmplitudeAutoOptions options;
        options.xslab.planner.memory_budget_bytes = config.memory_budget_bytes;
        options.xslab.planner.output_bytes = sizeof(double);

        const auto contraction_start = DiagnosticsClock::now();
        const b4support::PackedAmplitudeAutoResult contraction =
            b4support::contract_packed_amplitude_auto_streamed_generated_support(
                support, boosters, spins_array, intertwiners, options, nullptr);
        const double contraction_execution_seconds =
            seconds_since(contraction_start);
        const double sj_tensor_seconds = sixj_tensor_seconds(contraction);
        const double contraction_seconds = std::max(
            0.0, contraction_execution_seconds - sj_tensor_seconds);
        const double total_seconds = seconds_since(total_start);

        diagnostics.completed(
            total_seconds, " result=", std::scientific,
            std::setprecision(17), contraction.amplitude);
        diagnostics.timing({
            {"setup/support", "setup/support", setup_seconds},
            {"boosters", "boosters", booster_seconds},
            {"6j tensors", "6j tensors", sj_tensor_seconds},
            {"contraction", "contraction", contraction_seconds},
            {"total", "total", total_seconds},
        });
        diagnostics.log(
            Verbosity::detailed, "execution mode=",
            b4support::packed_amplitude_auto_mode_name(contraction.mode),
            " full_x_peak_bytes=",
            contraction.full_x_estimate.estimated_peak_bytes,
            " allowed_peak_bytes=",
            contraction.full_x_estimate.allowed_peak_bytes,
            " x_slabs=", contraction.xslab_plan.slabs.size(),
            " unfitting_x_slabs=", contraction.xslab_plan.num_unfitting_slabs(),
            " xq_slices=", contraction.xqslab_result.x_slices,
            " q_slabs=", contraction.xqslab_result.q_slabs_total,
            " xq_peak_bytes=",
            contraction.xqslab_result.max_estimated_peak_bytes);

        return contraction.amplitude;
    } catch (const std::exception& exception) {
        diagnostics.failed(seconds_since(total_start), exception.what());
        detail::rethrow_as_public_error(
            std::current_exception(), ErrorCode::numerical_failure);
    } catch (...) {
        diagnostics.failed(seconds_since(total_start), "unknown exception");
        detail::rethrow_as_public_error(
            std::current_exception(), ErrorCode::internal_failure);
    }
}

} // namespace sl2cfoamxx
