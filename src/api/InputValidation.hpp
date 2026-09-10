// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "sl2cfoamxx/sl2cfoamxx.hpp"

namespace sl2cfoamxx::detail {

// Validate public scientific inputs before any cache, channel, or numerical
// workspace is touched. All failures are reported as std::invalid_argument.
void validate_amplitude_inputs(
    const Spins& spins,
    const Intertwiners& intertwiners,
    std::size_t Dl,
    double gamma);

void validate_coherent_amplitude_inputs(
    const Spins& spins,
    std::size_t Dl,
    double gamma,
    const BoundaryAngles& angles);

void validate_coherent_amplitude_inputs(
    const Spins& spins,
    std::size_t Dl,
    double gamma,
    const BoundaryAngles& angles,
    Backend backend);

void validate_coherent_coefficients_inputs(
    const TetrahedronSpins& spins,
    const TetrahedronAngles& angles);

void validate_closure_inputs(
    const Spins& spins,
    const BoundaryAngles& angles,
    double absolute_tolerance,
    double relative_tolerance);

void validate_full_tensor_inputs(
    const Spins& spins,
    std::size_t Dl,
    double gamma);

void validate_full_tensor_inputs(
    const Spins& spins,
    std::size_t Dl,
    double gamma,
    Backend backend);

// Return the largest lifted doubled spin after the corresponding public-input
// validator has succeeded. This is the required upper index of LogFactTable.
[[nodiscard]] int log_factorial_maximum(
    const Spins& spins,
    std::size_t Dl);

// Canonical full-tensor extents derived from the five admissible intertwiner
// ranges. Boundary spins are validated before a shape is returned.
[[nodiscard]] FullTensor::shape_type canonical_full_tensor_shape(
    const Spins& spins);

} // namespace sl2cfoamxx::detail
