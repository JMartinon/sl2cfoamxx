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
