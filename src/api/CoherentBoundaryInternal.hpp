// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "sl2cfoamxx/configuration.hpp"
#include "sl2cfoamxx/sl2cfoamxx.hpp"

#include "amplitude/coherent_intertwiner.hpp"

#include <cstddef>

namespace sl2cfoamxx::detail {

// These two gather functions are the single source of truth for matching the
// ten unoriented boundary spins to the twenty oriented normals and to the five
// canonical full-tensor axes.
[[nodiscard]] TetrahedronSpins coherent_node_spins(
    const Spins& spins,
    std::size_t node);

[[nodiscard]] TetrahedronAngles coherent_node_angles(
    const BoundaryAngles& angles,
    std::size_t node);

[[nodiscard]] coherent_intertwiner::Result
compute_coherent_coefficients_internal(
    const TetrahedronSpins& spins,
    const TetrahedronAngles& angles,
    const LibraryConfig& config);

} // namespace sl2cfoamxx::detail
