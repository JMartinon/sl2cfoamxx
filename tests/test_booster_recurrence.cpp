// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "boosters/B4TensorFilling.hpp"

#include <array>
#include <cmath>
#include <iostream>

int main()
{
    // Doubled boundary spins and intertwiner channels; physical lift offsets.
    // The independent rapidity integral gives the same signed coefficient at
    // two quadrature densities. The old first-leg route gives +3.52e-4.
    const std::array<int, 1> boundary_i{48};
    const std::array<int, 1> lifted_k{176};
    Tensor<double, 5> output({1, 1, 1, 1, 1}, true);
    harvest_b4_impl<4, 5, 4>(
        48, 48, 47, 1, {1, 1, 1}, boundary_i, lifted_k, 0.1,
        LogFactTable(180), output, {64, 63, 64}, false,
        sl2cfoamxx::SimplicityConstraint::gamma_j, nullptr);

    constexpr double reference = -4.414242963715578e-8;
    const double value = output.data()[0];
    if (!std::isfinite(value) || std::abs(value - reference) > 1e-8 * std::abs(reference)) {
        std::cerr << "Unexpected signed booster coefficient: " << value
                  << " (expected " << reference << ")\n";
        return 1;
    }
    return 0;
}
