// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "api/InputValidation.hpp"

#include "core/OutputPresentation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sl2cfoamxx::detail {
namespace {

// This bounds legacy signed-int arithmetic and recurrence workspace sizes. It
// is far above the range for which materializing these tensors is practical.
constexpr int maximum_supported_lifted_doubled_spin = 10'000;

constexpr std::array<std::string_view, 10> spin_names{{
    "j12", "j13", "j14", "j15", "j23",
    "j24", "j25", "j34", "j35", "j45"}};

struct NodeDefinition {
    std::string_view intertwiner;
    std::array<std::size_t, 4> paired_spins;
};

// The first two and final two entries are the canonical recoupling pairs used
// by FullTensorChannels and AmplitudeChannels.
constexpr std::array<NodeDefinition, 5> nodes{{
    {"i1", {{2, 3, 1, 0}}},
    {"i2", {{4, 5, 6, 0}}},
    {"i3", {{7, 8, 1, 4}}},
    {"i4", {{9, 2, 5, 7}}},
    {"i5", {{3, 6, 8, 9}}},
}};

[[nodiscard]] std::string display_context(std::string_view operation)
{
    std::string context(product_display_name);
    context += ": ";
    context += operation;
    return context;
}

struct IntertwinerRange {
    std::int64_t lower = 0;
    std::int64_t upper = -1;
    int parity = 0;
    bool pair_parities_match = false;

    [[nodiscard]] bool empty() const noexcept
    {
        return !pair_parities_match || lower > upper;
    }
};

[[nodiscard]] IntertwinerRange admissible_range(
    const Spins& spins,
    const NodeDefinition& node) noexcept
{
    const std::int64_t p = spins[node.paired_spins[0]];
    const std::int64_t q = spins[node.paired_spins[1]];
    const std::int64_t r = spins[node.paired_spins[2]];
    const std::int64_t s = spins[node.paired_spins[3]];

    IntertwinerRange range;
    const int first_parity = static_cast<int>((p + q) & 1);
    const int second_parity = static_cast<int>((r + s) & 1);
    range.pair_parities_match = first_parity == second_parity;
    range.parity = first_parity;
    range.lower = std::max(std::abs(p - q), std::abs(r - s));
    range.upper = std::min(p + q, r + s);
    if ((range.lower & 1) != range.parity) ++range.lower;
    return range;
}

[[nodiscard]] std::string format_node_spins(
    const Spins& spins,
    const NodeDefinition& node)
{
    std::ostringstream message;
    const auto& index = node.paired_spins;
    message << '(' << spin_names[index[0]] << '=' << spins[index[0]]
            << ',' << spin_names[index[1]] << '=' << spins[index[1]]
            << ") and (" << spin_names[index[2]] << '=' << spins[index[2]]
            << ',' << spin_names[index[3]] << '=' << spins[index[3]] << ')';
    return message.str();
}

[[nodiscard]] FullTensor::shape_type validated_intertwiner_indices(
    const Spins& spins,
    const Intertwiners& intertwiners,
    std::string_view function_name)
{
    FullTensor::shape_type indices{};
    for (std::size_t index = 0; index < intertwiners.size(); ++index) {
        const int intertwiner = intertwiners[index];
        const NodeDefinition& node = nodes[index];
        const IntertwinerRange range = admissible_range(spins, node);
        const bool valid = intertwiner >= 0 &&
                           intertwiner >= range.lower &&
                           intertwiner <= range.upper &&
                           ((intertwiner & 1) == range.parity);
        if (!valid) {
            std::ostringstream message;
            message << function_name << ": unphysical intertwiner "
                    << node.intertwiner << '='
                    << intertwiner << " for " << format_node_spins(spins, node)
                    << "; allowed doubled values are [" << range.lower << ','
                    << range.upper << "] in steps of 2";
            throw std::invalid_argument(message.str());
        }
        indices[index] = static_cast<std::size_t>(
            (static_cast<std::int64_t>(intertwiner) - range.lower) / 2);
    }
    return indices;
}

void validate_boundary_spins(
    const Spins& spins,
    std::string_view function_name)
{
    for (std::size_t index = 0; index < spins.size(); ++index) {
        if (spins[index] < 0) {
            throw std::invalid_argument(
                std::string(function_name) +
                ": unphysical boundary spin: doubled " +
                std::string(spin_names[index]) + '=' +
                std::to_string(spins[index]) + " is negative");
        }
        if (spins[index] > maximum_supported_lifted_doubled_spin) {
            throw std::invalid_argument(
                std::string(function_name) +
                ": unsupported angular-momentum range: doubled spins[" +
                std::to_string(index) + "]=" +
                std::to_string(spins[index]) + " exceeds " +
                std::to_string(maximum_supported_lifted_doubled_spin));
        }
    }

    for (const NodeDefinition& node : nodes) {
        if (admissible_range(spins, node).empty()) {
            throw std::invalid_argument(
                std::string(function_name) +
                ": unphysical boundary spins at node " +
                std::string(node.intertwiner) + ": " +
                format_node_spins(spins, node) +
                " admit no common doubled intertwiner; the four doubled "
                "spins must have even total parity and satisfy the polygon "
                "inequality");
        }
    }
}

void validate_tetrahedron_spins(
    const TetrahedronSpins& spins,
    std::string_view function_name)
{
    for (std::size_t index = 0; index < spins.size(); ++index) {
        if (spins[index] < 0) {
            throw std::invalid_argument(
                std::string(function_name) +
                ": unphysical tetrahedron spin: doubled spins[" +
                std::to_string(index) + "]=" +
                std::to_string(spins[index]) + " is negative");
        }
        if (spins[index] > maximum_supported_lifted_doubled_spin) {
            throw std::invalid_argument(
                std::string(function_name) +
                ": unsupported angular-momentum range: doubled spins[" +
                std::to_string(index) + "]=" +
                std::to_string(spins[index]) + " exceeds " +
                std::to_string(maximum_supported_lifted_doubled_spin));
        }
    }

    const std::int64_t p = spins[0];
    const std::int64_t q = spins[1];
    const std::int64_t r = spins[2];
    const std::int64_t s = spins[3];
    const bool parities_match = ((p + q) & 1) == ((r + s) & 1);
    const std::int64_t lower = std::max(std::abs(p - q), std::abs(r - s));
    const std::int64_t upper = std::min(p + q, r + s);
    if (!parities_match || lower > upper) {
        throw std::invalid_argument(
            std::string(function_name) +
            ": the four doubled spins admit no common intertwiner in the "
            "(0,1)|(2,3) coupling basis");
    }
}

[[nodiscard]] int checked_lifted_maximum(
    const Spins& spins,
    std::size_t Dl,
    std::string_view function_name)
{
    const int maximum_spin = *std::max_element(spins.begin(), spins.end());
    if (maximum_spin < 0) {
        throw std::invalid_argument(
            std::string(function_name) +
            ": cannot determine a lifted-spin range from negative spins");
    }
    if (maximum_spin > maximum_supported_lifted_doubled_spin ||
        Dl > static_cast<std::size_t>(
                 (maximum_supported_lifted_doubled_spin - maximum_spin) / 2)) {
        std::ostringstream message;
        message << function_name
                << ": unsupported angular-momentum range: max(spins) + 2*Dl "
                << "must not exceed "
                << maximum_supported_lifted_doubled_spin
                << " (max(spins)=" << maximum_spin << ", Dl=" << Dl << ')';
        throw std::invalid_argument(message.str());
    }
    return maximum_spin + 2 * static_cast<int>(Dl);
}

void validate_common_inputs(
    const Spins& spins,
    std::size_t Dl,
    double gamma,
    std::string_view function_name)
{
    validate_boundary_spins(spins, function_name);
    (void)checked_lifted_maximum(spins, Dl, function_name);
    if (!std::isfinite(gamma)) {
        throw std::invalid_argument(
            std::string(function_name) + ": gamma must be finite");
    }
    if (gamma < 0.0) {
        throw std::invalid_argument(
            std::string(function_name) + ": gamma must be nonnegative");
    }
}

template <std::size_t Rows>
void validate_angles(
    const std::array<NormalAngles, Rows>& angles,
    std::string_view name,
    std::string_view function_name)
{
    static constexpr std::array<std::string_view, 2> coordinate_names{
        "theta", "phi"};
    for (std::size_t row = 0; row < angles.size(); ++row) {
        for (std::size_t coordinate = 0;
             coordinate < angles[row].size(); ++coordinate) {
            if (!std::isfinite(angles[row][coordinate])) {
                std::ostringstream message;
                message << function_name << ": " << name
                        << "[row=" << row << ","
                        << coordinate_names[coordinate] << "] must be finite";
                throw std::invalid_argument(message.str());
            }
        }
    }
}

} // namespace

void validate_amplitude_inputs(
    const Spins& spins,
    const Intertwiners& intertwiners,
    std::size_t Dl,
    double gamma)
{
    const std::string function_name = display_context("compute_amplitude");
    validate_common_inputs(spins, Dl, gamma, function_name);
    (void)validated_intertwiner_indices(
        spins, intertwiners, function_name);
}

void validate_coherent_amplitude_inputs(
    const Spins& spins,
    std::size_t Dl,
    double gamma,
    const BoundaryAngles& angles)
{
    const std::string function_name =
        display_context("compute_coherent_amplitude");
    validate_common_inputs(spins, Dl, gamma, function_name);
    validate_angles(angles, "angles", function_name);
}

void validate_coherent_amplitude_inputs(
    const Spins& spins,
    std::size_t Dl,
    double gamma,
    const BoundaryAngles& angles,
    Backend backend)
{
    const std::string function_name =
        display_context("compute_coherent_amplitude");
    if (backend != Backend::CPU && backend != Backend::GPU) {
        throw std::invalid_argument(
            std::string(function_name) + ": unknown backend");
    }
    validate_coherent_amplitude_inputs(spins, Dl, gamma, angles);
}

void validate_coherent_coefficients_inputs(
    const TetrahedronSpins& spins,
    const TetrahedronAngles& angles)
{
    const std::string function_name =
        display_context("compute_coherent_coefficients");
    validate_tetrahedron_spins(spins, function_name);
    validate_angles(angles, "angles", function_name);
}

void validate_coherent_amplitude_inputs(
    const Spins& spins,
    std::size_t Dl,
    double gamma,
    const std::array<const CoherentCoefficients*, 5>& coefficients)
{
    const std::string function_name =
        display_context("compute_coherent_amplitude");
    validate_common_inputs(spins, Dl, gamma, function_name);
    for (std::size_t node = 0; node < coefficients.size(); ++node) {
        const IntertwinerRange range = admissible_range(spins, nodes[node]);
        const auto expected = static_cast<std::size_t>(
            (range.upper - range.lower) / 2 + 1);
        const auto& values = *coefficients[node];
        const std::string name = "C" + std::to_string(node + 1);
        if (values.size() != expected) {
            throw std::invalid_argument(
                function_name + ": " + name + " must contain exactly " +
                std::to_string(expected) + " coefficients in ascending "
                "admissible doubled intertwiner order; received " +
                std::to_string(values.size()));
        }
        for (std::size_t index = 0; index < values.size(); ++index) {
            if (!std::isfinite(values[index].real()) ||
                !std::isfinite(values[index].imag())) {
                throw std::invalid_argument(
                    function_name + ": " + name + '[' +
                    std::to_string(index) + "] must have finite real and "
                    "imaginary parts");
            }
        }
    }
}

void validate_coherent_amplitude_inputs(
    const Spins& spins,
    std::size_t Dl,
    double gamma,
    const std::array<const CoherentCoefficients*, 5>& coefficients,
    Backend backend)
{
    if (backend != Backend::CPU && backend != Backend::GPU) {
        throw std::invalid_argument(
            display_context("compute_coherent_amplitude") +
            ": unknown backend");
    }
    validate_coherent_amplitude_inputs(spins, Dl, gamma, coefficients);
}

void validate_closure_inputs(
    const Spins& spins,
    const BoundaryAngles& angles,
    double absolute_tolerance,
    double relative_tolerance)
{
    const std::string function_name = display_context("check_closure");
    for (std::size_t index = 0; index < spins.size(); ++index) {
        if (spins[index] < 0) {
            throw std::invalid_argument(
                function_name + ": doubled " + std::string(spin_names[index]) +
                '=' + std::to_string(spins[index]) + " is negative");
        }
    }
    validate_angles(angles, "angles", function_name);
    if (!std::isfinite(absolute_tolerance) || absolute_tolerance < 0.0) {
        throw std::invalid_argument(
            function_name +
            ": absolute_tolerance must be finite and non-negative");
    }
    if (!std::isfinite(relative_tolerance) || relative_tolerance < 0.0) {
        throw std::invalid_argument(
            function_name +
            ": relative_tolerance must be finite and non-negative");
    }
}

void validate_full_tensor_inputs(
    const Spins& spins,
    std::size_t Dl,
    double gamma)
{
    validate_common_inputs(
        spins, Dl, gamma, display_context("compute_full_tensor"));
}

void validate_full_tensor_inputs(
    const Spins& spins,
    std::size_t Dl,
    double gamma,
    Backend backend)
{
    const std::string function_name = display_context("compute_full_tensor");
    if (backend != Backend::CPU && backend != Backend::GPU) {
        throw std::invalid_argument(function_name + ": unknown backend");
    }
    validate_full_tensor_inputs(spins, Dl, gamma);
}

int log_factorial_maximum(
    const Spins& spins,
    std::size_t Dl)
{
    return checked_lifted_maximum(
        spins, Dl, display_context("internal factorial table"));
}

FullTensor::shape_type canonical_full_tensor_shape(const Spins& spins)
{
    const std::string function_name = display_context("full-tensor shape");
    validate_boundary_spins(spins, function_name);

    FullTensor::shape_type shape{};
    for (std::size_t axis = 0; axis < nodes.size(); ++axis) {
        const IntertwinerRange range = admissible_range(spins, nodes[axis]);
        shape[axis] = static_cast<std::size_t>(
            (range.upper - range.lower) / 2 + 1);
    }
    return shape;
}

} // namespace sl2cfoamxx::detail

namespace sl2cfoamxx {

FullTensor::shape_type indices_from_intertwiners(
    const Spins& spins,
    const Intertwiners& intertwiners)
{
    const std::string function_name =
        detail::display_context("indices_from_intertwiners");
    detail::validate_boundary_spins(spins, function_name);
    return detail::validated_intertwiner_indices(
        spins, intertwiners, function_name);
}

} // namespace sl2cfoamxx
