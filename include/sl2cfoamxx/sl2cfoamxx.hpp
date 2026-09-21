// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "sl2cfoamxx/capabilities.hpp"
#include "sl2cfoamxx/configuration.hpp"
#include "sl2cfoamxx/error.hpp"
#include "sl2cfoamxx/export.hpp"
#include "sl2cfoamxx/full_tensor.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace sl2cfoamxx {

// Historical source-compatible aliases. New code should prefer DSpins,
// DIntertwiners, and TetrahedronDSpins so doubled-label units stay explicit.
using Spins = DSpins;
using Intertwiners = DIntertwiners;
using TetrahedronSpins = TetrahedronDSpins;

// Spherical coordinates {theta,phi}, in radians, of one unit normal.
using NormalAngles = std::array<double, 2>;

// Doubled spins and oriented normals at one four-valent boundary tetrahedron.
// Entries have a one-to-one positional correspondence. The coherent
// coefficient basis couples entries (0,1) with entries (2,3).
using TetrahedronAngles = std::array<NormalAngles, 4>;

// Spherical coordinates for all twenty oriented boundary normals, in the
// canonical row order
//   {n12,n13,n14,n15, n21,n23,n24,n25, n31,n32,n34,n35,
//    n41,n42,n43,n45, n51,n52,n53,n54}.
// Each row is {theta_ab,phi_ab}. A face has two distinct entries n_ab and
// n_ba because they belong to the local frames of different tetrahedra.
using BoundaryAngles = std::array<NormalAngles, 20>;

// Reference boundary geometries in radians and canonical oriented-normal order.
// These fixed angle sets do not select a model or a backend. The Euclidean
// boundary closes for equal face spins; the Lorentzian boundary closes for
// face-spin ratios (5,5,5,5,2,2,2,2,2,2). Copy a preset to modify its angles.
inline constexpr BoundaryAngles lorentzian_boundary_angles{{
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

inline constexpr BoundaryAngles euclidean_boundary_angles{{
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

// Alternate spellings for the same immutable presets.
inline constexpr const BoundaryAngles& lorentzian_bounday_angles =
    lorentzian_boundary_angles;
inline constexpr const BoundaryAngles& euclidean_bounday_angles =
    euclidean_boundary_angles;

// Coefficients are ordered by ascending admissible doubled intertwiner,
// starting at max(|J1-J2|,|J3-J4|) (adjusted for parity), ending at
// min(J1+J2,J3+J4), and advancing in steps of two.
using CoherentCoefficients = std::vector<std::complex<double>>;

struct ClosureNodeResult {
    std::array<double, 3> vector{};
    double residual_norm = 0.0;
    double scale = 0.0;
    double relative_residual = 0.0;
    bool closed = false;
};

struct ClosureCheckResult {
    std::array<ClosureNodeResult, 5> nodes{};
    bool all_closed = false;

    [[nodiscard]] explicit operator bool() const noexcept
    {
        return all_closed;
    }
};

// Convert the five doubled intertwiners to zero-based canonical full-tensor
// indices. Invalid boundary spins, parity, or out-of-range intertwiners throw
// std::invalid_argument.
[[nodiscard]] SL2CFOAMXX_API FullTensor::shape_type indices_from_intertwiners(
    const DSpins& spins,
    const DIntertwiners& intertwiners);

// Invalid scientific inputs are rejected with std::invalid_argument before a
// cache or numerical workspace is touched. Catch the exception around an
// individual call when the application should skip that computation and
// continue; as usual in C++, an uncaught exception terminates the process.
// Operational failures use Error with a stable ErrorCode; standard bounds,
// overflow, allocation, and filesystem exceptions retain their usual types.

// Compute the Livine-Speziale coherent-intertwiner coefficient vector for one
// four-valent tetrahedron. Spins are doubled integers and each angle row
// {theta,phi} corresponds to the spin in the same array position.
[[nodiscard]] SL2CFOAMXX_API CoherentCoefficients
compute_coherent_coefficients(
    const TetrahedronDSpins& spins,
    const TetrahedronAngles& angles);

// Evaluate sum_{b!=a} j_ab*n_ab at every boundary tetrahedron. Public spins
// are doubled, but vectors and scales are reported with physical j=J/2 so
// they match the mathematical closure equation. This function performs no
// printing; callers can inspect the five node results or all_closed.
[[nodiscard]] SL2CFOAMXX_API ClosureCheckResult check_closure(
    const DSpins& spins,
    const BoundaryAngles& angles,
    double absolute_tolerance = 1.0e-12,
    double relative_tolerance = 1.0e-12);

// The three primary computation APIs below use
// LibraryConfig::simplicity_constraint.

// Compute one fixed-intertwiner amplitude on the CPU. Each doubled
// intertwiner label must belong to the admissible range determined by the four incident doubled spins.
[[nodiscard]] SL2CFOAMXX_API double compute_amplitude(
    const DSpins& spins,
    const DIntertwiners& intertwiners,
    std::size_t Dl,
    double gamma);

// Explicit best-effort cache overload. prefer_full_tensor checks a configured
// exact output path, or both automatic dense/symmetry cache names, for a
// complete tensor with exactly matching spins, Dl, gamma bits, and simplicity
// constraint. A miss or invalid cache file falls back to direct computation.
// In-memory FullTensor objects are intentionally not retained globally; access
// those with tensor.at(indices_from_intertwiners(spins, intertwiners)).
[[nodiscard]] SL2CFOAMXX_API double compute_amplitude(
    const DSpins& spins,
    const DIntertwiners& intertwiners,
    std::size_t Dl,
    double gamma,
    AmplitudeCachePolicy cache_policy);

// Compute one coherent-boundary amplitude. Omitting the backend selects
// LibraryConfig::backend. The explicit overload always honors the caller's
// selection and never silently falls back.
[[nodiscard]] SL2CFOAMXX_API std::complex<double>
compute_coherent_amplitude(
    const DSpins& spins,
    std::size_t Dl,
    double gamma,
    const BoundaryAngles& angles);

[[nodiscard]] SL2CFOAMXX_API std::complex<double>
compute_coherent_amplitude(
    const DSpins& spins,
    std::size_t Dl,
    double gamma,
    const BoundaryAngles& angles,
    Backend backend);

// Per-call progress overload for concurrent applications and language
// bindings. The supplied callback replaces LibraryConfig::progress_callback
// for this computation only; pass an empty callback to suppress the configured
// callback. Terminal rendering still follows LibraryConfig::progress_bar.
[[nodiscard]] SL2CFOAMXX_API std::complex<double>
compute_coherent_amplitude(
    const DSpins& spins,
    std::size_t Dl,
    double gamma,
    const BoundaryAngles& angles,
    Backend backend,
    ProgressCallback progress_callback);

// Contract caller-supplied coefficients in the canonical bases for nodes
// 1--5: (j12,j13)|(j14,j15), (j23,j24)|(j25,j12),
// (j34,j35)|(j13,j23), (j45,j14)|(j24,j34), (j15,j25)|(j35,j45).
// Each vector contains every admissible intertwiner in ascending doubled
// label order and must have exactly the corresponding full-tensor extent.
// Entries must be finite; no conjugation or normalization is applied. Zero
// vectors are valid. C1 pruning still follows the configured pruning_ratio.
// Closure checking is skipped because no boundary normals are supplied.
// Backend selection and per-call callbacks follow the angle overloads above.
[[nodiscard]] SL2CFOAMXX_API std::complex<double>
compute_coherent_amplitude(
    const DSpins& spins,
    std::size_t Dl,
    double gamma,
    const CoherentCoefficients& C1,
    const CoherentCoefficients& C2,
    const CoherentCoefficients& C3,
    const CoherentCoefficients& C4,
    const CoherentCoefficients& C5);

[[nodiscard]] SL2CFOAMXX_API std::complex<double>
compute_coherent_amplitude(
    const DSpins& spins,
    std::size_t Dl,
    double gamma,
    const CoherentCoefficients& C1,
    const CoherentCoefficients& C2,
    const CoherentCoefficients& C3,
    const CoherentCoefficients& C4,
    const CoherentCoefficients& C5,
    Backend backend);

[[nodiscard]] SL2CFOAMXX_API std::complex<double>
compute_coherent_amplitude(
    const DSpins& spins,
    std::size_t Dl,
    double gamma,
    const CoherentCoefficients& C1,
    const CoherentCoefficients& C2,
    const CoherentCoefficients& C3,
    const CoherentCoefficients& C4,
    const CoherentCoefficients& C5,
    Backend backend,
    ProgressCallback progress_callback);

// Compute the complete tensor with canonical (i1,i2,i3,i4,i5) indexed access.
// Omitting the backend selects LibraryConfig::backend. The explicit overload
// always honors the caller's selection and never silently falls back. Both
// overloads use the physical representation, backing, reuse, path, and limits
// in LibraryConfig::tensors.
[[nodiscard]] SL2CFOAMXX_API FullTensor compute_full_tensor(
    const DSpins& spins,
    std::size_t Dl,
    double gamma);

[[nodiscard]] SL2CFOAMXX_API FullTensor compute_full_tensor(
    const DSpins& spins,
    std::size_t Dl,
    double gamma,
    Backend backend);

// Per-call counterpart of the overload above. The callback is owned for the
// duration of this synchronous call and replaces the configured callback
// without changing global configuration or another active computation.
[[nodiscard]] SL2CFOAMXX_API FullTensor compute_full_tensor(
    const DSpins& spins,
    std::size_t Dl,
    double gamma,
    Backend backend,
    ProgressCallback progress_callback);

} // namespace sl2cfoamxx
