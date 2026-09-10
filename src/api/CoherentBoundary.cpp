// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "sl2cfoamxx/sl2cfoamxx.hpp"

#include "api/CoherentBoundaryInternal.hpp"
#include "api/ConfigurationInternal.hpp"
#include "api/ErrorInternal.hpp"
#include "api/InputValidation.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <stdexcept>

namespace sl2cfoamxx::detail {
namespace {

struct CoherentNodeMap {
    std::array<std::size_t, 4> spin_rows;
    std::array<std::size_t, 4> angle_rows;
};

// Public spin order:
//   j12,j13,j14,j15,j23,j24,j25,j34,j35,j45
// Public angle order:
//   n12,n13,n14,n15,n21,n23,n24,n25,n31,n32,n34,n35,
//   n41,n42,n43,n45,n51,n52,n53,n54
//
// Every row below is also the local (0,1)|(2,3) recoupling order of the
// corresponding full-tensor axis. In particular, C2 ends with (j12,n21), and
// C5 starts with (j15,n51); no later within-row permutation is required.
constexpr std::array<CoherentNodeMap, 5> coherent_nodes{{
    {{{0, 1, 2, 3}}, {{0, 1, 2, 3}}},
    {{{4, 5, 6, 0}}, {{5, 6, 7, 4}}},
    {{{7, 8, 1, 4}}, {{10, 11, 8, 9}}},
    {{{9, 2, 5, 7}}, {{15, 12, 13, 14}}},
    {{{3, 6, 8, 9}}, {{16, 17, 18, 19}}},
}};

[[nodiscard]] const CoherentNodeMap& node_map(std::size_t node)
{
    if (node >= coherent_nodes.size()) {
        throw std::out_of_range("coherent boundary node must be in [0,4]");
    }
    return coherent_nodes[node];
}

} // namespace

TetrahedronSpins coherent_node_spins(
    const Spins& spins,
    std::size_t node)
{
    const auto& rows = node_map(node).spin_rows;
    return {
        spins[rows[0]], spins[rows[1]], spins[rows[2]], spins[rows[3]]};
}

TetrahedronAngles coherent_node_angles(
    const BoundaryAngles& angles,
    std::size_t node)
{
    const auto& rows = node_map(node).angle_rows;
    return {
        angles[rows[0]], angles[rows[1]], angles[rows[2]], angles[rows[3]]};
}

coherent_intertwiner::Result compute_coherent_coefficients_internal(
    const TetrahedronSpins& spins,
    const TetrahedronAngles& angles,
    const LibraryConfig& config)
{
    coherent_intertwiner::WorkspaceBuildOptions workspace_options;
    workspace_options.use_openmp = true;
    workspace_options.num_threads = static_cast<int>(config.cpu_threads);
    workspace_options.parallel_min_mu_blocks = 16;

    coherent_intertwiner::EvalOptions evaluation_options;
    evaluation_options.compensated = false;

    return coherent_intertwiner::compute_coherent_coefficients_for_intertwiner(
        spins, angles, workspace_options, evaluation_options);
}

} // namespace sl2cfoamxx::detail

namespace sl2cfoamxx {

CoherentCoefficients compute_coherent_coefficients(
    const TetrahedronSpins& spins,
    const TetrahedronAngles& angles)
{
    detail::validate_coherent_coefficients_inputs(spins, angles);
    const LibraryConfig config = detail::configuration_snapshot();
    coherent_intertwiner::Result result;
    try {
        result = detail::compute_coherent_coefficients_internal(
            spins, angles, config);
    } catch (...) {
        detail::rethrow_as_public_error(
            std::current_exception(), ErrorCode::numerical_failure);
    }

    CoherentCoefficients coefficients;
    coefficients.reserve(result.coefficients.size());
    for (const b4utils::Complex64 coefficient : result.coefficients) {
        coefficients.emplace_back(coefficient.re, coefficient.im);
    }
    return coefficients;
}

ClosureCheckResult check_closure(
    const Spins& spins,
    const BoundaryAngles& angles,
    double absolute_tolerance,
    double relative_tolerance)
{
    detail::validate_closure_inputs(
        spins, angles, absolute_tolerance, relative_tolerance);

    ClosureCheckResult result;
    result.all_closed = true;
    for (std::size_t node = 0; node < result.nodes.size(); ++node) {
        const TetrahedronSpins node_spins =
            detail::coherent_node_spins(spins, node);
        const TetrahedronAngles node_angles =
            detail::coherent_node_angles(angles, node);

        ClosureNodeResult& node_result = result.nodes[node];
        for (std::size_t leg = 0; leg < node_spins.size(); ++leg) {
            const double physical_spin =
                0.5 * static_cast<double>(node_spins[leg]);
            const double theta = node_angles[leg][0];
            const double phi = node_angles[leg][1];
            const double sin_theta = std::sin(theta);
            node_result.vector[0] +=
                physical_spin * sin_theta * std::cos(phi);
            node_result.vector[1] +=
                physical_spin * sin_theta * std::sin(phi);
            node_result.vector[2] += physical_spin * std::cos(theta);
            node_result.scale += std::abs(physical_spin);
        }

        node_result.residual_norm = std::sqrt(
            node_result.vector[0] * node_result.vector[0] +
            node_result.vector[1] * node_result.vector[1] +
            node_result.vector[2] * node_result.vector[2]);
        node_result.relative_residual = node_result.scale == 0.0
            ? node_result.residual_norm
            : node_result.residual_norm / node_result.scale;
        node_result.closed = node_result.residual_norm <=
            absolute_tolerance + relative_tolerance * node_result.scale;
        result.all_closed = result.all_closed && node_result.closed;
    }
    return result;
}

} // namespace sl2cfoamxx
