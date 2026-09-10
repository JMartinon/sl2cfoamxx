// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "sl2cfoamxx/sl2cfoamxx.hpp"

#include "api/ConfigurationInternal.hpp"
#include "api/FullTensorCachePath.hpp"
#include "api/InputValidation.hpp"
#include "core/tensors.hpp"

#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <optional>
#include <string>
#include <utility>

namespace sl2cfoamxx {
namespace {

using Clock = std::chrono::steady_clock;

struct CachedAmplitude {
    double value = 0.0;
    std::filesystem::path path{};
};

struct FullTensorProbe {
    std::optional<CachedAmplitude> hit{};
    std::string miss_reason{"no candidate cache file exists"};
};

[[nodiscard]] double seconds_since(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

[[nodiscard]] const char* simplicity_name(
    SimplicityConstraint prescription) noexcept
{
    return prescription == SimplicityConstraint::gamma_j_p1
        ? "gamma_j_p1"
        : "gamma_j";
}

[[nodiscard]] bool exact_gamma_match(double left, double right) noexcept
{
    return std::bit_cast<std::uint64_t>(left) ==
           std::bit_cast<std::uint64_t>(right);
}

[[nodiscard]] std::string physics_mismatch_reason(
    const FullTensorBinaryTagV1& tag,
    const Spins& spins,
    std::size_t Dl,
    double gamma,
    SimplicityConstraint simplicity,
    const FullTensor::shape_type& expected_shape)
{
    if (tag.Dl != static_cast<std::uint64_t>(Dl)) {
        return "Dl metadata does not match";
    }
    if (!exact_gamma_match(tag.gamma, gamma)) {
        return "gamma metadata does not match exactly";
    }
    if (!full_tensor_simplicity_constraint_matches(tag, simplicity)) {
        return "simplicity-constraint metadata does not match";
    }
    for (std::size_t index = 0; index < spins.size(); ++index) {
        if (tag.spins[index] != static_cast<std::int32_t>(spins[index])) {
            return "boundary-spin metadata does not match";
        }
    }
    for (std::size_t axis = 0; axis < expected_shape.size(); ++axis) {
        if (tag.canonical_dims[axis] !=
            static_cast<std::uint64_t>(expected_shape[axis])) {
            return "canonical tensor shape metadata does not match";
        }
    }
    return {};
}

[[nodiscard]] FullTensorProbe probe_full_tensor_cache(
    const LibraryConfig& config,
    const Spins& spins,
    const Intertwiners& intertwiners,
    std::size_t Dl,
    double gamma)
{
    FullTensorProbe result;
    const FullTensor::shape_type expected_shape =
        detail::canonical_full_tensor_shape(spins);
    const FullTensor::shape_type indices =
        indices_from_intertwiners(spins, intertwiners);

    for (const std::filesystem::path& path :
         detail::full_tensor_cache_candidate_paths(
             config, spins, Dl, gamma)) {
        std::error_code error;
        const bool exists = std::filesystem::exists(path, error);
        if (error) {
            result.miss_reason =
                "cannot inspect candidate " + path.string() + ": " +
                error.message();
            continue;
        }
        if (!exists) continue;

        const bool regular = std::filesystem::is_regular_file(path, error);
        if (error || !regular) {
            result.miss_reason =
                "candidate is not a readable regular file: " + path.string();
            continue;
        }

        try {
            const FullTensor tensor =
                FullTensor::map(path, TensorAccess::read_only);
            FullTensorBinaryTagV1 tag{};
            if (!read_full_tensor_binary_tag(tensor.tag(), tag)) {
                result.miss_reason =
                    "candidate has no valid full-tensor physics metadata: " +
                    path.string();
                continue;
            }
            const std::string mismatch = physics_mismatch_reason(
                tag, spins, Dl, gamma, config.simplicity_constraint,
                expected_shape);
            if (!mismatch.empty() || tensor.shape() != expected_shape) {
                result.miss_reason =
                    (mismatch.empty()
                         ? "logical tensor shape does not match"
                         : mismatch) +
                    ": " + path.string();
                continue;
            }

            // tensor is const so at() uses the read-only-safe access path even
            // though FullTensor also provides a mutable overload.
            result.hit = CachedAmplitude{tensor.at(indices), path};
            return result;
        } catch (const std::exception& exception) {
            result.miss_reason =
                "candidate is not reusable: " + path.string() + ": " +
                exception.what();
        }
    }
    return result;
}

} // namespace

double compute_amplitude(
    const Spins& spins,
    const Intertwiners& intertwiners,
    std::size_t Dl,
    double gamma,
    AmplitudeCachePolicy cache_policy)
{
    if (cache_policy == AmplitudeCachePolicy::disabled) {
        return compute_amplitude(spins, intertwiners, Dl, gamma);
    }

    detail::validate_amplitude_inputs(spins, intertwiners, Dl, gamma);
    if (cache_policy != AmplitudeCachePolicy::prefer_full_tensor) {
        throw std::invalid_argument(
            std::string(detail::product_display_name) +
            ": compute_amplitude: unknown amplitude cache policy");
    }

    const auto lookup_start = Clock::now();
    const LibraryConfig config = detail::configuration_snapshot();
    const detail::Diagnostics diagnostics(
        config, detail::DiagnosticOperation::amplitude, Backend::CPU);
    const FullTensorProbe probe = probe_full_tensor_cache(
        config, spins, intertwiners, Dl, gamma);
    const double lookup_seconds = seconds_since(lookup_start);

    if (!probe.hit.has_value()) {
        diagnostics.log(
            Verbosity::detailed,
            "full-tensor cache status=miss policy=prefer_full_tensor reason=",
            probe.miss_reason,
            " lookup_seconds=", lookup_seconds);
        return compute_amplitude(spins, intertwiners, Dl, gamma);
    }

    const double total_seconds = seconds_since(lookup_start);
    diagnostics.started(
        " Dl=", Dl,
        " gamma=", std::setprecision(17), gamma,
        " simplicity=", simplicity_name(config.simplicity_constraint),
        " cache_policy=prefer_full_tensor");
    diagnostics.log(
        Verbosity::detailed,
        "full-tensor cache status=hit path=", probe.hit->path.string(),
        " lookup_seconds=", lookup_seconds);
    diagnostics.completed(
        total_seconds, " result=", std::scientific,
        std::setprecision(17), probe.hit->value,
        " source=full_tensor_cache");
    diagnostics.timing({
        {"full tensor lookup", "full tensor lookup", lookup_seconds},
        {"total", "total", total_seconds},
    });
    return probe.hit->value;
}

} // namespace sl2cfoamxx
