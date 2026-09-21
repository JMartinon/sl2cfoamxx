// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include <sl2cfoamxx.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using sl2cfoamxx::BoundaryAngles;
using sl2cfoamxx::DSpins;
using Complex = std::complex<double>;

constexpr auto lorentzian_angles = sl2cfoamxx::lorentzian_boundary_angles;

constexpr auto euclidean_angles = sl2cfoamxx::euclidean_boundary_angles;

void test_case(
    std::string_view name,
    const DSpins& spins,
    const BoundaryAngles& angles,
    Complex expected)
{
    using namespace sl2cfoamxx;

    const ClosureCheckResult closure = check_closure(spins, angles);
    if (!closure.all_closed) {
        throw std::runtime_error(std::string(name) + " data do not close");
    }

    // Build the five vectors in the canonical full-tensor recoupling bases.
    // The one-node API matches spins and angle pairs position by position.
    const CoherentCoefficients C1 = compute_coherent_coefficients(
        {{spins[0], spins[1], spins[2], spins[3]}},
        {{angles[0], angles[1], angles[2], angles[3]}});
    const CoherentCoefficients C2 = compute_coherent_coefficients(
        {{spins[4], spins[5], spins[6], spins[0]}},
        {{angles[5], angles[6], angles[7], angles[4]}});
    const CoherentCoefficients C3 = compute_coherent_coefficients(
        {{spins[7], spins[8], spins[1], spins[4]}},
        {{angles[10], angles[11], angles[8], angles[9]}});
    const CoherentCoefficients C4 = compute_coherent_coefficients(
        {{spins[9], spins[2], spins[5], spins[7]}},
        {{angles[15], angles[12], angles[13], angles[14]}});
    const CoherentCoefficients C5 = compute_coherent_coefficients(
        {{spins[3], spins[6], spins[8], spins[9]}},
        {{angles[16], angles[17], angles[18], angles[19]}});

    constexpr std::size_t Dl = 2;
    constexpr double gamma = 1.2;
    const Complex amplitude = compute_coherent_amplitude(
        spins, Dl, gamma, angles);
    // These vectors can also come from the caller's own boundary state.
    // Their complex phases and normalization are preserved.
    const Complex supplied_amplitude = compute_coherent_amplitude(
        spins, Dl, gamma, C1, C2, C3, C4, C5);
    const double allowed = 5.0e-26 +
        5.0e-9 * std::max(std::abs(expected), 1.0e-300);
    if (std::abs(amplitude - expected) > allowed) {
        throw std::runtime_error(std::string(name) + " amplitude mismatch");
    }
    if (std::abs(supplied_amplitude - amplitude) > allowed) {
        throw std::runtime_error(std::string(name) + " coefficient amplitude mismatch");
    }

    std::cout << name << ": closure=passed, C1 coefficients=" << C1.size()
              << ", W=" << amplitude << '\n';
}

} // namespace

int main()
{
    using namespace sl2cfoamxx;

    LibraryConfig config = default_config();
    config.backend = Backend::CPU;
    config.simplicity_constraint = SimplicityConstraint::gamma_j_p1;
    config.coherent_amplitude.check_closure = true;
    config.coherent_amplitude.pruning_ratio = 1.0e-8;
    config.verbosity = Verbosity::silent;
    configure(config);

    const DSpins lorentzian_spins{{
        10, 10, 10, 10, 4, 4, 4, 4, 4, 4,
    }};
    DSpins euclidean_spins{};
    euclidean_spins.fill(4);

    test_case(
        "Lorentzian", lorentzian_spins, lorentzian_angles,
        {-1.2613917379855847e-17, -1.039173505859162e-16});
    test_case(
        "Euclidean", euclidean_spins, euclidean_angles,
        {-1.961220849529498e-15, -1.0851108126375121e-15});
}
