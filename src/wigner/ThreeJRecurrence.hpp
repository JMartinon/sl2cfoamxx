// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "core/Ieee754.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

// Seed-free, complete-row Wigner 3j recurrences.
//
// All angular momenta and magnetic quantum numbers use doubled integers.  The
// implementation follows the stable two-sided Schulten--Gordon /
// Luscombe--Luban construction: minimal-solution ratios are propagated from
// both endpoints, the classical region is bridged by the ordinary three-term
// recurrence, orthogonality fixes the magnitude, and the analytic upper-end
// phase fixes the sign.  No scalar Wigner evaluator or symbol table is needed.

namespace threejrec {

struct Workspace {
    // Off-diagonal and diagonal coefficients for the current complete row.
    // Keeping them in a caller-owned or thread-local workspace makes repeated
    // row generation allocation-free after the largest row seen so far.
    std::vector<double> coupling;
    std::vector<double> diagonal;
};

namespace detail {

using b4utils::ieee754::finite_double;
using b4utils::ieee754::negative_double;
using b4utils::ieee754::preserve_zero_sign;
using b4utils::ieee754::zero_double;

inline double multiply_preserving_zero_sign(double lhs, double rhs) noexcept {
    const double product = lhs * rhs;
    // The overwhelmingly common path needs no bit inspection.  Only an exact
    // underflowed zero requires recovering the mathematical sign explicitly.
    if (product != 0.0) {
        return product;
    }
    return preserve_zero_sign(product,
                              negative_double(lhs) != negative_double(rhs));
}

inline bool valid_jm(int Dj, int Dm) noexcept {
    if (Dj < 0) {
        return false;
    }
    const std::int64_t j = Dj;
    const std::int64_t m = Dm;
    return -j <= m && m <= j && ((j - m) & 1) == 0;
}

inline bool valid_triangle(int Dj1, int Dj2, int Dj3) noexcept {
    if (Dj1 < 0 || Dj2 < 0 || Dj3 < 0) {
        return false;
    }
    const std::int64_t j1 = Dj1;
    const std::int64_t j2 = Dj2;
    const std::int64_t j3 = Dj3;
    return ((j1 + j2 + j3) & 1) == 0 &&
           (j1 >= j2 ? j1 - j2 : j2 - j1) <= j3 &&
           j3 <= j1 + j2;
}

inline int phase_from_even_integer(std::int64_t twice_exponent) noexcept {
    // twice_exponent is guaranteed even by the admissibility checks.  Taking
    // the magnitude also makes the parity operation explicit for negative
    // exponents.
    std::int64_t exponent = twice_exponent / 2;
    if (exponent < 0) {
        exponent = -exponent;
    }
    return (exponent & 1) != 0 ? -1 : 1;
}

inline void invalidate(int& Dmin, int& Dmax,
                       std::vector<double>& out) noexcept {
    Dmin = 0;
    Dmax = -2;
    out.clear();
}

inline bool resize_workspace(std::size_t n, Workspace& workspace) {
    workspace.coupling.resize(n);
    workspace.diagonal.resize(n);
    return n != 0;
}

// Reconstruct and normalize a complete row satisfying
//
//     X_i f_{i+1} + Y_i f_i + Z_i f_{i-1} = 0.
//
// forward_ratio_zero is f_1/f_0.  Passing it explicitly handles the j=0
// endpoint, where the ordinary j recurrence degenerates to 0/0 but its
// limiting boundary equation remains well defined.
template <typename XFn, typename ZFn, typename WeightFn>
inline bool reconstruct_and_normalize(std::size_t n,
                                      std::vector<double>& out,
                                      const Workspace& workspace,
                                      double forward_ratio_zero,
                                      int upper_endpoint_phase,
                                      XFn&& X,
                                      ZFn&& Z,
                                      WeightFn&& weight)
{
    if (n == 0 || workspace.diagonal.size() < n ||
        workspace.coupling.size() < n ||
        !finite_double(forward_ratio_zero)) {
        return false;
    }

    out.resize(n);
    if (n == 1) {
        const double w = weight(0);
        if (!(w > 0.0) || !finite_double(w)) {
            return false;
        }
        out[0] = static_cast<double>(upper_endpoint_phase) / std::sqrt(w);
        return finite_double(out[0]);
    }

    constexpr double one = 1.0;
    const double tiny =
        std::sqrt(std::sqrt(std::numeric_limits<double>::min()));

    // Lower minimal-solution tail.  out[i] temporarily stores f_i/f_{i+1}.
    std::size_t lower_seed = 0;
    if (!zero_double(forward_ratio_zero)) {
        double sn = one / forward_ratio_zero;
        std::size_t i = 0;
        while (finite_double(sn) && std::fabs(sn) <= one) {
            out[i] = sn;
            lower_seed = i + 1;
            if (lower_seed >= n - 1) {
                break;
            }
            i = lower_seed;
            const double den = std::fma(Z(i), sn, workspace.diagonal[i]);
            if (std::fabs(den) < tiny || !finite_double(den)) {
                break;
            }
            sn = -X(i) / den;
        }
    }

    // Upper minimal-solution tail.  out[i] temporarily stores f_i/f_{i-1}.
    std::size_t upper_ratio_begin = n;
    const std::size_t last = n - 1;
    const double y_last = workspace.diagonal[last];
    if (std::fabs(y_last) >= tiny) {
        double rn = -Z(last) / y_last;
        std::size_t i = last;
        while (finite_double(rn) && std::fabs(rn) <= one) {
            out[i] = rn;
            upper_ratio_begin = i;
            if (i == 0) {
                break;
            }
            --i;
            if (i < lower_seed || i == 0) {
                break;
            }
            const double den =
                std::fma(X(i), rn, workspace.diagonal[i]);
            if (std::fabs(den) < tiny || !finite_double(den)) {
                break;
            }
            rn = -Z(i) / den;
        }
    }

    // Anchor the row at the lower turning point, reconstruct the lower tail,
    // bridge the oscillatory region, then attach the upper ratios.  Accumulate
    // the orthogonality norm during reconstruction so no extra read pass is
    // needed before scaling.
    out[lower_seed] = one;
    double w = weight(lower_seed);
    if (!(w > 0.0) || !finite_double(w)) {
        return false;
    }
    double norm = w;

    for (std::size_t k = lower_seed; k > 0; --k) {
        out[k - 1] = multiply_preserving_zero_sign(out[k - 1], out[k]);
        w = weight(k - 1);
        norm = std::fma(w, out[k - 1] * out[k - 1], norm);
    }

    std::size_t k = lower_seed;
    if (k == 0) {
        out[1] = multiply_preserving_zero_sign(forward_ratio_zero, out[0]);
        w = weight(1);
        norm = std::fma(w, out[1] * out[1], norm);
        k = 1;
    }

    for (; k + 1 < upper_ratio_begin && k + 1 < n; ++k) {
        const double x = X(k);
        if (std::fabs(x) < tiny || !finite_double(x)) {
            return false;
        }
        const double numerator = std::fma(
            workspace.diagonal[k], out[k], Z(k) * out[k - 1]);
        out[k + 1] = -numerator / x;
        w = weight(k + 1);
        norm = std::fma(w, out[k + 1] * out[k + 1], norm);
    }

    for (; k + 1 < n; ++k) {
        out[k + 1] =
            multiply_preserving_zero_sign(out[k + 1], out[k]);
        w = weight(k + 1);
        norm = std::fma(w, out[k + 1] * out[k + 1], norm);
    }

    if (!(norm > 0.0) || !finite_double(norm)) {
        return false;
    }

    // The raw upper endpoint may underflow to signed zero; its sign still
    // carries the phase selected by the upper minimal-solution ratios.
    const bool desired_negative = upper_endpoint_phase < 0;
    const bool scale_negative =
        desired_negative != negative_double(out[last]);
    const double scale = (scale_negative ? -1.0 : 1.0) / std::sqrt(norm);
    if (!finite_double(scale)) {
        return false;
    }

    for (std::size_t i = 0; i < n; ++i) {
        out[i] *= scale;
    }
    return true;
}

inline bool prepare_m2_coefficients(int Dj1, int Dj2, int Dj3,
                                    int Dm1, int Dm2_min,
                                    std::size_t n, Workspace& workspace)
{
    resize_workspace(n, workspace);

    const long double j1 = Dj1;
    const long double j2 = Dj2;
    const long double j3 = Dj3;
    const long double m1 = Dm1;
    const long double linear_diagonal =
        j2 * (j2 + 2.0L) + j3 * (j3 + 2.0L) - j1 * (j1 + 2.0L);

    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t Dm2_i =
            static_cast<std::int64_t>(Dm2_min) +
            2 * static_cast<std::int64_t>(i);
        const long double m = static_cast<long double>(Dm2_i);
        const long double radicand =
            (j2 + 2.0L - m) * (j2 + m) *
            (j3 - m1 + 2.0L - m) * (j3 + m1 + m);
        if (radicand < 0.0L) {
            return false;
        }

        const double x = static_cast<double>(std::sqrt(radicand));
        const double y = static_cast<double>(
            linear_diagonal - 2.0L * m * (m + m1));
        workspace.coupling[i] = x;
        workspace.diagonal[i] = y;
    }
    return true;
}

inline bool prepare_j1_coefficients(int Dj2, int Dj3, int Dm2, int Dm3,
                                    int Dj1_min, std::size_t n,
                                    Workspace& workspace)
{
    resize_workspace(n, workspace);

    const long double j2 = Dj2;
    const long double j3 = Dj3;
    const long double m2 = Dm2;
    const long double m3 = Dm3;
    const long double difference = j2 - j3;
    const long double sum_plus_two = j2 + j3 + 2.0L;
    const long double magnetic_sum = m2 + m3;
    const long double j2_quadratic = j2 * (j2 + 2.0L);
    const long double j3_quadratic = j3 * (j3 + 2.0L);
    const long double fixed_diagonal =
        magnetic_sum * (j2_quadratic - j3_quadratic);
    const long double linear_diagonal = m2 - m3;

    for (std::size_t i = 0; i < n; ++i) {
        const std::int64_t Dj1_i =
            static_cast<std::int64_t>(Dj1_min) +
            2 * static_cast<std::int64_t>(i);
        const long double j = static_cast<long double>(Dj1_i);
        const long double j_squared = j * j;
        const long double j_quadratic = j * (j + 2.0L);
        const long double radicand =
            (j_squared - difference * difference) *
            (sum_plus_two * sum_plus_two - j_squared) *
            (j_squared - magnetic_sum * magnetic_sum);
        if (radicand < 0.0L) {
            return false;
        }

        const double x = static_cast<double>(std::sqrt(radicand));
        const double y = static_cast<double>(
            (2.0L * j + 2.0L) *
            (fixed_diagonal - linear_diagonal * j_quadratic));
        workspace.coupling[i] = x;
        workspace.diagonal[i] = y;
    }
    return true;
}

} // namespace detail

// Complete row
//
//     ( j1  j2  j3 )
//     ( m1   x -m1-x )
//
// with x=m2 swept in unit steps.  Dm2_min/max and out are invalidated on
// failure.  This is also the primitive used by the production magnetic-row
// convenience function below.
inline bool fill_wigner3j_m2_series(int Dj1, int Dj2, int Dj3, int Dm1,
                                    int& Dm2_min, int& Dm2_max,
                                    std::vector<double>& out,
                                    Workspace& workspace)
{
    if (!detail::valid_triangle(Dj1, Dj2, Dj3) ||
        !detail::valid_jm(Dj1, Dm1)) {
        detail::invalidate(Dm2_min, Dm2_max, out);
        return false;
    }

    const std::int64_t lower = std::max<std::int64_t>(
        -static_cast<std::int64_t>(Dj2),
        -static_cast<std::int64_t>(Dj3) - Dm1);
    const std::int64_t upper = std::min<std::int64_t>(
         static_cast<std::int64_t>(Dj2),
         static_cast<std::int64_t>(Dj3) - Dm1);
    if (lower > upper || lower < std::numeric_limits<int>::min() ||
        upper > std::numeric_limits<int>::max() || ((upper - lower) & 1)) {
        detail::invalidate(Dm2_min, Dm2_max, out);
        return false;
    }

    Dm2_min = static_cast<int>(lower);
    Dm2_max = static_cast<int>(upper);
    const std::size_t n = static_cast<std::size_t>((upper - lower) / 2 + 1);
    if (!detail::prepare_m2_coefficients(
            Dj1, Dj2, Dj3, Dm1, Dm2_min, n, workspace)) {
        detail::invalidate(Dm2_min, Dm2_max, out);
        return false;
    }

    if (n == 1) {
        out.resize(1);
        const int phase = detail::phase_from_even_integer(
            static_cast<std::int64_t>(Dj2) - Dj3 - Dm1);
        out[0] = static_cast<double>(phase) /
                 std::sqrt(static_cast<double>(Dj1) + 1.0);
        return detail::finite_double(out[0]);
    }

    const double x1 = workspace.coupling[1];
    if (!(x1 > 0.0) || !detail::finite_double(x1)) {
        detail::invalidate(Dm2_min, Dm2_max, out);
        return false;
    }
    const double forward_ratio_zero = -workspace.diagonal[0] / x1;
    const int phase = detail::phase_from_even_integer(
        static_cast<std::int64_t>(Dj2) - Dj3 - Dm1);

    auto X = [&workspace](std::size_t i) noexcept {
        return workspace.coupling[i + 1];
    };
    auto Z = [&workspace](std::size_t i) noexcept {
        return workspace.coupling[i];
    };
    const double weight = static_cast<double>(Dj1) + 1.0;
    auto Weight = [weight](std::size_t) noexcept { return weight; };

    if (!detail::reconstruct_and_normalize(
            n, out, workspace, forward_ratio_zero, phase, X, Z, Weight)) {
        detail::invalidate(Dm2_min, Dm2_max, out);
        return false;
    }
    return true;
}

inline bool fill_wigner3j_m2_series(int Dj1, int Dj2, int Dj3, int Dm1,
                                    int& Dm2_min, int& Dm2_max,
                                    std::vector<double>& out)
{
    static thread_local Workspace workspace;
    return fill_wigner3j_m2_series(
        Dj1, Dj2, Dj3, Dm1, Dm2_min, Dm2_max, out, workspace);
}

// Production sequence
//
//     ( j1  j2   J )
//     (  n  k-n  -k )
//
// with n swept in unit steps.  The implementation evaluates the cyclically
// permuted row (J,j1,j2;-k,n,k-n); a cyclic column permutation carries no 3j
// phase.  Arguments and returned endpoints are doubled integers.
inline bool fill_wigner3j_magnetic_row(int Dj1, int Dj2, int DJ, int Dk,
                                       int& Dn_min, int& Dn_max,
                                       std::vector<double>& out,
                                       Workspace& workspace)
{
    return fill_wigner3j_m2_series(
        DJ, Dj1, Dj2, -Dk, Dn_min, Dn_max, out, workspace);
}

inline bool fill_wigner3j_magnetic_row(int Dj1, int Dj2, int DJ, int Dk,
                                       int& Dn_min, int& Dn_max,
                                       std::vector<double>& out)
{
    static thread_local Workspace workspace;
    return fill_wigner3j_magnetic_row(
        Dj1, Dj2, DJ, Dk, Dn_min, Dn_max, out, workspace);
}

// Complete row
//
//     (  x  j2  j3 )
//     (-m2-m3 m2  m3 )
//
// with x=j1 swept in unit steps.  This sequence is useful whenever many
// coupling channels share fixed magnetic quantum numbers.
inline bool fill_wigner3j_j1_series(int Dj2, int Dj3, int Dm2, int Dm3,
                                    int& Dj1_min, int& Dj1_max,
                                    std::vector<double>& out,
                                    Workspace& workspace)
{
    if (!detail::valid_jm(Dj2, Dm2) || !detail::valid_jm(Dj3, Dm3)) {
        detail::invalidate(Dj1_min, Dj1_max, out);
        return false;
    }

    const std::int64_t difference =
        static_cast<std::int64_t>(Dj2) - Dj3;
    const std::int64_t magnetic_sum =
        static_cast<std::int64_t>(Dm2) + Dm3;
    const std::int64_t lower = std::max(
        difference >= 0 ? difference : -difference,
        magnetic_sum >= 0 ? magnetic_sum : -magnetic_sum);
    const std::int64_t upper =
        static_cast<std::int64_t>(Dj2) + Dj3;
    if (lower > upper || upper > std::numeric_limits<int>::max() ||
        ((upper - lower) & 1)) {
        detail::invalidate(Dj1_min, Dj1_max, out);
        return false;
    }

    Dj1_min = static_cast<int>(lower);
    Dj1_max = static_cast<int>(upper);
    const std::size_t n = static_cast<std::size_t>((upper - lower) / 2 + 1);
    if (!detail::prepare_j1_coefficients(
            Dj2, Dj3, Dm2, Dm3, Dj1_min, n, workspace)) {
        detail::invalidate(Dj1_min, Dj1_max, out);
        return false;
    }

    const int phase = detail::phase_from_even_integer(
        static_cast<std::int64_t>(Dj2) - Dj3 + Dm2 + Dm3);
    if (n == 1) {
        out.resize(1);
        out[0] = static_cast<double>(phase) /
                 std::sqrt(static_cast<double>(Dj1_min) + 1.0);
        return detail::finite_double(out[0]);
    }

    const double x1 = workspace.coupling[1];
    if (!(x1 > 0.0) || !detail::finite_double(x1)) {
        detail::invalidate(Dj1_min, Dj1_max, out);
        return false;
    }

    double forward_ratio_zero = 0.0;
    if (Dj1_min == 0) {
        // At j1=0, j2=j3 and m2=-m3.  The ordinary recurrence has a
        // removable 0/0 endpoint; its limiting boundary equation gives this
        // finite reciprocal ratio directly (including the exact zero at m2=0).
        forward_ratio_zero =
            (8.0 * static_cast<double>(Dm2)) / x1;
    } else {
        forward_ratio_zero =
            -workspace.diagonal[0] /
            (static_cast<double>(Dj1_min) * x1);
    }

    auto X = [Dj1_min, &workspace](std::size_t i) noexcept {
        const double Dj = static_cast<double>(
            static_cast<std::int64_t>(Dj1_min) +
            2 * static_cast<std::int64_t>(i));
        return Dj * workspace.coupling[i + 1];
    };
    auto Z = [Dj1_min, &workspace](std::size_t i) noexcept {
        const double Dj = static_cast<double>(
            static_cast<std::int64_t>(Dj1_min) +
            2 * static_cast<std::int64_t>(i));
        return (Dj + 2.0) * workspace.coupling[i];
    };
    auto Weight = [Dj1_min](std::size_t i) noexcept {
        return static_cast<double>(
            static_cast<std::int64_t>(Dj1_min) +
            2 * static_cast<std::int64_t>(i)) + 1.0;
    };

    if (!detail::reconstruct_and_normalize(
            n, out, workspace, forward_ratio_zero, phase, X, Z, Weight)) {
        detail::invalidate(Dj1_min, Dj1_max, out);
        return false;
    }
    return true;
}

inline bool fill_wigner3j_j1_series(int Dj2, int Dj3, int Dm2, int Dm3,
                                    int& Dj1_min, int& Dj1_max,
                                    std::vector<double>& out)
{
    static thread_local Workspace workspace;
    return fill_wigner3j_j1_series(
        Dj2, Dj3, Dm2, Dm3, Dj1_min, Dj1_max, out, workspace);
}

} // namespace threejrec
