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

const BoundaryAngles lorentzian_angles{{
    {{1.570796326794897,  0.000000000000000}}, // n12
    {{1.570796326794897,  1.910633236249019}}, // n13
    {{0.615479708670387, -2.186276035465284}}, // n14
    {{2.526112944919406, -2.186276035465284}}, // n15
    {{1.570796326794897,  3.141592653589793}}, // n21
    {{1.570796326794897,  0.585685543457151}}, // n23
    {{1.071607449443858, -0.320246032512208}}, // n24
    {{2.069985204145935, -0.320246032512208}}, // n25
    {{1.570796326794897, -1.230959417340775}}, // n31
    {{1.570796326794897,  1.324947692791868}}, // n32
    {{1.071607449443858,  2.230879268761226}}, // n34
    {{2.069985204145935,  2.230879268761226}}, // n35
    {{2.526112944919406,  0.955316618124509}}, // n41
    {{0.573540745391100, -1.105670539592056}}, // n42
    {{0.573540745391100,  3.016303775841075}}, // n43
    {{1.201165252127538, -2.186276035465284}}, // n45
    {{0.615479708670387,  0.955316618124509}}, // n51
    {{2.568051908198693, -1.105670539592056}}, // n52
    {{2.568051908198693,  3.016303775841075}}, // n53
    {{1.940427401462255, -2.186276035465284}}, // n54
}};

const BoundaryAngles euclidean_angles{{
    {{0.000000000000000,  0.000000000000000}}, // n12
    {{1.910633236249019,  0.000000000000000}}, // n13
    {{1.910633236249019,  2.094395102393196}}, // n14
    {{1.910633236249019, -2.094395102393196}}, // n15
    {{3.141592653589793,  0.000000000000000}}, // n21
    {{1.230959417340775, -1.823476581936975}}, // n23
    {{1.230959417340775,  0.270918520456220}}, // n24
    {{1.230959417340775,  2.365313622849416}}, // n25
    {{1.230959417340775,  3.141592653589793}}, // n31
    {{1.910633236249019,  1.318116071652818}}, // n32
    {{2.411864997362827, -1.435337066566786}}, // n34
    {{0.729727656226966, -0.388139515370189}}, // n35
    {{1.230959417340775, -1.047197551196598}}, // n41
    {{1.910633236249019, -2.870674133133573}}, // n42
    {{0.729727656226966,  1.706255587023007}}, // n43
    {{2.411864997362827,  0.659058035826409}}, // n45
    {{1.230959417340775,  1.047197551196598}}, // n51
    {{1.910633236249019, -0.776279030740378}}, // n52
    {{2.411864997362827,  2.753453138219605}}, // n53
    {{0.729727656226966, -2.482534617763384}}, // n54
}};

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
