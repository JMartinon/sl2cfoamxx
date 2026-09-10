// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <boosters/B4_function.hpp>
#include "core/Ieee754.hpp"

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <initializer_list>
#include <limits>
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <algorithm>

inline int pm1_from_even_Dsum(int Dsum) noexcept {
#ifndef NDEBUG
    if (Dsum & 1) {
        std::cerr << "Odd doubled-spin phase sum: " << Dsum << "\n";
        std::abort();
    }
#endif
    
    const int n = Dsum / 2;
    return (std::abs(n) & 1) ? -1 : 1;
}

// The five real 6j factors used below share the real phase of the first-kind
// 15j symbol.  For integer boundary spins that phase can be split between the
// five factors directly.  With half-integer spins, however, the phase assigned
// to an individual factor can be (-1)^(odd/2) = +/-i even though the product of
// all five phases is real.  Evaluating each odd doubled exponent with integer
// division loses those paired imaginary units and can flip the 15j symbol.
//
// Keep every stored tensor real by removing the fixed parity of the A, D and E
// exponents, then put the (necessarily real) product of the removed phases in
// E.  Triangle admissibility makes these parities independent of the summed
// intertwiner and lifted-spin values:
//
//   pA = parity(j14 + j15)
//   pD = parity(j23 + j24)
//   pE = parity(j12 - j35 - j34 - j24).
//
// SU(2) admissibility at the five four-valent nodes guarantees
// pA+pD+pE is even.
struct Wigner15jRealPhaseSplit {
    int a_parity = 0;
    int d_parity = 0;
    int e_parity = 0;
    int e_correction = 1;
};

inline int doubled_sum_parity(int Dsum) noexcept {
    const int remainder = Dsum % 2;
    return remainder < 0 ? -remainder : remainder;
}

inline Wigner15jRealPhaseSplit wigner15j_real_phase_split(
    const B4Spins& s) noexcept
{
    Wigner15jRealPhaseSplit split;
    split.a_parity = doubled_sum_parity(s.Dj14 + s.Dj15);
    split.d_parity = doubled_sum_parity(s.Dj23 + s.Dj24);
    split.e_parity = doubled_sum_parity(
        s.Dj12 - s.Dj35 - s.Dj34 - s.Dj24);
    split.e_correction = pm1_from_even_Dsum(
        split.a_parity + split.d_parity + split.e_parity);
    return split;
}



namespace sjrec_detail {

using b4utils::ieee754::double_bits;
using b4utils::ieee754::double_from_bits;
using b4utils::ieee754::finite_double;
using b4utils::ieee754::negative_double;
using b4utils::ieee754::preserve_zero_sign;
using b4utils::ieee754::zero_double;

// ============================================================================
// Stable recurrence helpers for
//
//      h(j) = { j  j2 j3
//               j4 j5 j6 }
//
// with j swept in unit steps. External code uses doubled spins Dj = 2*j.
//
// Complete 6j strings are not cached: every hit would still copy a vector.
// Instead, cache the reusable pair factors entering the Luscombe-Luban
// recurrence coefficients:
//
//      E(j) = G_{j2,j3}(j) * G_{j5,j6}(j)
//
// where
//
//      G_{a,b}(j) = sqrt(|(j^2 - (a-b)^2) (j^2 - (a+b+1)^2)|).
//
// For the packed amplitude fillers this is precisely the reusable structure:
// sjA, for example, sweeps k5 while looping over (l25,x), so many recurrence
// calls share G_{j15,l25}(k5) or G_{x,j14}(k5). Each cached factor is retained
// as a compensated high+correction pair. Two FMAs then form E(j) with the
// accuracy of a combined square root while preserving pair reuse.
// ============================================================================

#ifndef SJREC_VALIDATE_RECURRENCE
#define SJREC_VALIDATE_RECURRENCE 0
#endif

inline constexpr std::size_t pair_factor_cache_reserve = 4096u;

inline double half_spin(int Dj) noexcept {
    return 0.5 * static_cast<double>(Dj);
}

inline double multiply_preserving_zero_sign(double lhs, double rhs) noexcept {
    const double product = lhs * rhs;
    if (!zero_double(product)) {
        return product;
    }
    const bool negative = negative_double(lhs) != negative_double(rhs);
    return preserve_zero_sign(product, negative);
}

inline double t_of_j(double j) noexcept {
    return j * (j + 1.0);
}

inline int doubled_axis_value(int origin, std::size_t index) noexcept {
    return static_cast<int>(static_cast<std::int64_t>(origin) +
                            2 * static_cast<std::int64_t>(index));
}

inline bool valid_complete_wigner6j_x_range(int Dx_min,
                                            int Dx_max,
                                            int Dj2,
                                            int Dj3,
                                            int Dj5,
                                            int Dj6) noexcept
{
    if (Dx_min < 0 || Dx_max < Dx_min ||
        Dj2 < 0 || Dj3 < 0 || Dj5 < 0 || Dj6 < 0) {
        return false;
    }

    const std::int64_t j2 = Dj2;
    const std::int64_t j3 = Dj3;
    const std::int64_t j5 = Dj5;
    const std::int64_t j6 = Dj6;
    const std::int64_t sum23 = j2 + j3;
    const std::int64_t sum56 = j5 + j6;
    if ((sum23 & 1) != (sum56 & 1)) {
        return false;
    }

    const std::int64_t diff23 = j2 >= j3 ? j2 - j3 : j3 - j2;
    const std::int64_t diff56 = j5 >= j6 ? j5 - j6 : j6 - j5;
    const std::int64_t expected_min = std::max(diff23, diff56);
    const std::int64_t expected_max = std::min(sum23, sum56);
    return expected_min <= expected_max &&
           static_cast<std::int64_t>(Dx_min) == expected_min &&
           static_cast<std::int64_t>(Dx_max) == expected_max;
}

inline double sqrt_abs_long_double_product(long double p) noexcept {
    if (p == 0.0L) {
        return 0.0;
    }
    if (p < 0.0L) {
        p = -p;
    }
    return static_cast<double>(std::sqrt(p));
}

struct PairFactorValue {
    double high = 0.0;
    float correction = 0.0F;
};

struct PairFactorSeries {
    int Da = -1;
    int Db = -1;
    int Dmin = 0;
    int cached_Dmax = -2;
    std::vector<double> values;
    std::vector<float> corrections;
};

inline std::uint64_t pair_factor_key(int Da, int Db) noexcept {
    if (Db < Da) {
        const int tmp = Da;
        Da = Db;
        Db = tmp;
    }
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(Da)) << 32)
         |  static_cast<std::uint64_t>(static_cast<std::uint32_t>(Db));
}

inline std::unordered_map<std::uint64_t, PairFactorSeries>& pair_factor_cache_tls() {
    static thread_local std::unordered_map<std::uint64_t, PairFactorSeries> cache = [] {
        std::unordered_map<std::uint64_t, PairFactorSeries> m;
        m.reserve(pair_factor_cache_reserve);
        return m;
    }();
    return cache;
}

inline PairFactorValue compute_pair_factor_value(int Da,
                                                 int Db,
                                                 int Dvar) noexcept
{
    const long double D = static_cast<long double>(Dvar);
    const long double difference =
        static_cast<long double>(Da) - static_cast<long double>(Db);
    const long double sum =
        static_cast<long double>(Da) + static_cast<long double>(Db) + 2.0L;
    const long double low = (D - difference) * (D + difference);
    const long double high = (sum - D) * (sum + D);
    long double radicand = low * high;
    if (radicand < 0.0L) {
        radicand = -radicand;
    }

    const long double refined = 0.25L * std::sqrt(radicand);
    const double value = static_cast<double>(refined);
    double correction =
        static_cast<double>(refined - static_cast<long double>(value));

    // On platforms where long double has only double precision, recover the
    // unavailable low part from the square-root residual with one Newton step.
    if constexpr (std::numeric_limits<long double>::digits ==
                  std::numeric_limits<double>::digits) {
        if (!zero_double(value)) {
            const double scaled_radicand =
                static_cast<double>(0.0625L * radicand);
            correction = std::fma(-value, value, scaled_radicand) /
                         (2.0 * value);
        }
    }

    return {value, static_cast<float>(correction)};
}

inline double multiply_pair_factors(PairFactorValue lhs,
                                    PairFactorValue rhs) noexcept
{
    const double correction = std::fma(
        lhs.high, static_cast<double>(rhs.correction),
        static_cast<double>(lhs.correction) * rhs.high);
    return std::fma(lhs.high, rhs.high, correction);
}

inline double compute_zero_branch_coupling_value(int Dj2,
                                                 int Dj3,
                                                 int Dj5,
                                                 int Dj6,
                                                 int Dvar) noexcept
{
    const long double D = static_cast<long double>(Dvar);
    const long double sum23 =
        static_cast<long double>(Dj2) + static_cast<long double>(Dj3) + 2.0L;
    const long double sum56 =
        static_cast<long double>(Dj5) + static_cast<long double>(Dj6) + 2.0L;
    const long double first = (sum23 - D) * (sum23 + D);
    const long double second = (sum56 - D) * (sum56 + D);
    return 0.125 * static_cast<double>(D) *
           sqrt_abs_long_double_product(first * second);
}

inline void extend_pair_factor_series(PairFactorSeries& entry,
                                      int Da,
                                      int Db,
                                      int required_Dmax)
{
    if (Db < Da) {
        const int tmp = Da;
        Da = Db;
        Db = tmp;
    }

    const int Dmin = Db - Da;
    if (required_Dmax < Dmin) {
        required_Dmax = Dmin;
    }
    if ((required_Dmax & 1) != (Dmin & 1)) {
        --required_Dmax;
    }

    if (entry.Da != Da || entry.Db != Db || entry.Dmin != Dmin) {
        entry.Da = Da;
        entry.Db = Db;
        entry.Dmin = Dmin;
        entry.cached_Dmax = Dmin - 2;
        entry.values.clear();
        entry.corrections.clear();
    }

    const int old_max = entry.cached_Dmax;
    if (old_max >= required_Dmax) {
        return;
    }

    const std::size_t new_size =
        static_cast<std::size_t>((required_Dmax - Dmin) / 2 + 1);
    entry.values.resize(new_size);
    entry.corrections.resize(new_size);

    const int first = old_max + 2;
    for (int D = first; D <= required_Dmax; D += 2) {
        const std::size_t index =
            static_cast<std::size_t>((D - Dmin) / 2);
        const PairFactorValue value =
            compute_pair_factor_value(Da, Db, D);
        entry.values[index] = value.high;
        entry.corrections[index] = value.correction;
    }

    entry.cached_Dmax = required_Dmax;
}

inline const PairFactorSeries& get_pair_factor_values(int Da,
                                                      int Db,
                                                      int required_Dmax) {
    const std::uint64_t key = pair_factor_key(Da, Db);
    auto& cache = pair_factor_cache_tls();
    auto it = cache.find(key);

    if (it == cache.end()) {
        PairFactorSeries entry;
        extend_pair_factor_series(entry, Da, Db, required_Dmax);
        auto inserted = cache.emplace(key, std::move(entry));
        return inserted.first->second;
    }

    if (it->second.cached_Dmax < required_Dmax) {
        extend_pair_factor_series(it->second, Da, Db, required_Dmax);
    }
    return it->second;
}

struct Wigner6jXRecurrence {
    int Dj2 = 0;
    int Dj3 = 0;
    int Dj4 = 0;
    int Dj5 = 0;
    int Dj6 = 0;

    double j2 = 0.0;
    double j3 = 0.0;
    double j4 = 0.0;
    double j5 = 0.0;
    double j6 = 0.0;

    double f2 = 0.0;
    double f3 = 0.0;
    double f4 = 0.0;
    double f5 = 0.0;
    double f6 = 0.0;

    double Y1 = 0.0;
    double Y0 = 0.0;
    double Y0_zero = 0.0;

    bool jmin_is_zero = false;

    Wigner6jXRecurrence(int in_Dj2,
                        int in_Dj3,
                        int in_Dj4,
                        int in_Dj5,
                        int in_Dj6,
                        bool use_jmin_zero_branch)
    : Dj2(in_Dj2),
      Dj3(in_Dj3),
      Dj4(in_Dj4),
      Dj5(in_Dj5),
      Dj6(in_Dj6),
      j2(half_spin(in_Dj2)),
      j3(half_spin(in_Dj3)),
      j4(half_spin(in_Dj4)),
      j5(half_spin(in_Dj5)),
      j6(half_spin(in_Dj6)),
      f2(t_of_j(j2)),
      f3(t_of_j(j3)),
      f4(t_of_j(j4)),
      f5(t_of_j(j5)),
      f6(t_of_j(j6)),
      Y1(f2 + f3 - 2.0 * f4 + f5 + f6),
      Y0((f5 - f6) * (f2 - f3)),
      Y0_zero(2.0 * (f2 - f4 + f5)),
      jmin_is_zero(use_jmin_zero_branch)
    {}

    inline double Y_generic(double j) const noexcept {
        const double q = j * (j + 1.0);
        return (2.0 * j + 1.0) * std::fma(q, Y1 - q, Y0);
    }

    inline double Y_zero(double j) const noexcept {
        const double q = j * (j + 1.0);
        return (2.0 * j + 1.0) * (-q + Y0_zero);
    }

    inline double Z_zero_branch_from_D(int Dvar) const noexcept {
        // Keep the exact doubled-spin triangle boundaries until the product is
        // formed. This is the Dmin==0 analogue of the compact off-diagonal
        // coefficient used by WIGSGLL and by the CUDA recurrence.
        return compute_zero_branch_coupling_value(
            Dj2, Dj3, Dj5, Dj6, Dvar);
    }
};

struct Wigner6jRecurrenceWorkspace {
    std::vector<double> coupling;
    std::vector<double> Y;
};

inline void prepare_recurrence_coefficients(int Dj_min,
                                            int Dj_max,
                                            const Wigner6jXRecurrence& xyz,
                                            Wigner6jRecurrenceWorkspace& ws)
{
    const std::int64_t span =
        static_cast<std::int64_t>(Dj_max) - static_cast<std::int64_t>(Dj_min);
    const std::size_t n = static_cast<std::size_t>(span / 2 + 1);

    ws.coupling.resize(n);
    ws.Y.resize(n);

    if (n == 0) {
        return;
    }

    if (xyz.jmin_is_zero) {
        for (std::size_t i = 0; i < n; ++i) {
            const int D = doubled_axis_value(Dj_min, i);
            const double j = half_spin(D);
            ws.Y[i] = xyz.Y_zero(j);
            ws.coupling[i] = xyz.Z_zero_branch_from_D(D);
        }
        return;
    }

    // X(j_i) uses E(j_{i+1}); the last X is never consumed, so coupling only
    // needs to be prepared through the actual recurrence endpoint.
    const int required_Dmax = Dj_max;
    const PairFactorSeries& g23 =
        get_pair_factor_values(xyz.Dj2, xyz.Dj3, required_Dmax);
    const PairFactorSeries& g56 =
        get_pair_factor_values(xyz.Dj5, xyz.Dj6, required_Dmax);
    const std::size_t offset23 = static_cast<std::size_t>(
        (Dj_min - g23.Dmin) / 2);
    const std::size_t offset56 = static_cast<std::size_t>(
        (Dj_min - g56.Dmin) / 2);

#pragma omp simd
    for (std::size_t i = 0; i < n; ++i) {
        const int D = doubled_axis_value(Dj_min, i);
        const double j = half_spin(D);

        ws.Y[i] = xyz.Y_generic(j);
        ws.coupling[i] = multiply_pair_factors(
            {g23.values[offset23 + i], g23.corrections[offset23 + i]},
            {g56.values[offset56 + i], g56.corrections[offset56 + i]});
    }
}

template <typename XFn, typename YFn, typename ZFn>
inline bool reconstruct_luscombe_luban_6j(std::size_t n,
                                          std::vector<double>& out,
                                          XFn&& X,
                                          YFn&& Y,
                                          ZFn&& Z)
{
    if (n == 0) {
        return false;
    }
    if (n == 1) {
        out[0] = 1.0;
        return true;
    }

    constexpr double one = 1.0;
    const double tiny = std::sqrt(std::sqrt(std::numeric_limits<double>::min()));

    // Lower minimal-solution tail. Ratios are retained only while their
    // magnitude is at most one; lower_seed is anchored once at unity.
    std::size_t lower_seed = 0;
    const double ymin = Y(0);
    if (std::fabs(ymin) >= tiny) {
        double sn = -X(0) / ymin;
        std::size_t i = 0;
        while (finite_double(sn) && std::fabs(sn) <= one) {
            out[i] = sn;
            lower_seed = i + 1;
            if (lower_seed >= n - 1) {
                break;
            }
            i = lower_seed;
            const double den = std::fma(Z(i), sn, Y(i));
            if (std::fabs(den) < tiny) {
                break;
            }
            sn = -X(i) / den;
        }
    }

    // Upper minimal-solution tail. n is a sentinel meaning that the ordinary
    // recurrence will reach the endpoint without an attached ratio tail.
    std::size_t upper_ratio_begin = n;
    const std::size_t last = n - 1;
    const double ymax = Y(last);
    if (std::fabs(ymax) >= tiny) {
        double rn = -Z(last) / ymax;
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
            const double den = std::fma(X(i), rn, Y(i));
            if (std::fabs(den) < tiny) {
                break;
            }
            rn = -Z(i) / den;
        }
    }

    // Reconstruct the lower tail, advance once through the classical region,
    // and attach the stored upper ratios directly. Unlike the former two-anchor
    // bridge, this has no potentially tiny bridge division and no rescale pass.
    out[lower_seed] = one;
    for (std::size_t k = lower_seed; k > 0; --k) {
        out[k - 1] *= out[k];
    }

    std::size_t k = lower_seed;
    if (k == 0) {
        const double X0 = X(0);
        if (std::fabs(X0) < tiny) {
            return false;
        }
        out[1] = (-Y(0) * out[0]) / X0;
        k = 1;
    }

    for (; k + 1 < upper_ratio_begin && k + 1 < n; ++k) {
        const double Xk = X(k);
        if (std::fabs(Xk) < tiny) {
            return false;
        }
        const double numerator = std::fma(Y(k), out[k], Z(k) * out[k - 1]);
        out[k + 1] = -numerator / Xk;
    }
    for (; k + 1 < n; ++k) {
        // This tail can underflow after many individually well-scaled ratios.
        // Preserve the mathematical zero sign because the upper endpoint fixes
        // the phase of the normalized 6j row, even under -ffast-math.
        out[k + 1] = multiply_preserving_zero_sign(out[k + 1], out[k]);
    }

#if SJREC_VALIDATE_RECURRENCE
    for (double v : out) {
        if (!finite_double(v)) {
            return false;
        }
    }
#endif

    return true;
}

inline bool rec_luscombe_luban_6j_fast(int Dj_min,
                                       int Dj_max,
                                       const Wigner6jXRecurrence& xyz,
                                       std::vector<double>& out,
                                       Wigner6jRecurrenceWorkspace& ws)
{
    if (!valid_complete_wigner6j_x_range(
            Dj_min, Dj_max, xyz.Dj2, xyz.Dj3, xyz.Dj5, xyz.Dj6)) {
        out.clear();
        return false;
    }

    const std::int64_t span =
        static_cast<std::int64_t>(Dj_max) - static_cast<std::int64_t>(Dj_min);
    const std::size_t n = static_cast<std::size_t>(span / 2 + 1);

    out.resize(n);

    if (n == 0) {
        return false;
    }

    if (n == 1) {
        out[0] = 1.0;
        return true;
    }

    prepare_recurrence_coefficients(Dj_min, Dj_max, xyz, ws);

    auto X = [Dj_min, &xyz, &ws](std::size_t i) noexcept -> double {
        if (xyz.jmin_is_zero) {
            return ws.coupling[i + 1];
        }
        const double j = half_spin(doubled_axis_value(Dj_min, i));
        return j * ws.coupling[i + 1];
    };
    auto Y = [&ws](std::size_t i) noexcept -> double { return ws.Y[i]; };
    auto Z = [Dj_min, &xyz, &ws](std::size_t i) noexcept -> double {
        if (xyz.jmin_is_zero) {
            return ws.coupling[i];
        }
        const double j = half_spin(doubled_axis_value(Dj_min, i));
        return (j + 1.0) * ws.coupling[i];
    };

    return reconstruct_luscombe_luban_6j(n, out, X, Y, Z);
}

inline bool rec_luscombe_luban_6j(int Dj_min,
                                  int Dj_max,
                                  const Wigner6jXRecurrence& xyz,
                                  std::vector<double>& out)
{
    static thread_local Wigner6jRecurrenceWorkspace ws;
    return rec_luscombe_luban_6j_fast(Dj_min, Dj_max, xyz, out, ws);
}

inline bool wigner6j_normalization_scale(int Dx_min,
                                         std::size_t n,
                                         int Dj2,
                                         int Dj3,
                                         int Dj4,
                                         int Dj5,
                                         int Dj6,
                                         const std::vector<double>& values,
                                         double& scale) noexcept
{
    if (n == 0 || values.size() < n) {
        return false;
    }

    double norm = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double value = values[i];
        if (!finite_double(value)) {
            return false;
        }
        const double weight =
            static_cast<double>(Dx_min) + 2.0 * static_cast<double>(i) + 1.0;
        norm = std::fma(weight, value * value, norm);
    }

    const double j4_dimension = static_cast<double>(Dj4) + 1.0;
    norm *= j4_dimension;
    if (!(norm > 0.0) || !finite_double(norm)) {
        return false;
    }

    const std::int64_t Dsum =
        static_cast<std::int64_t>(Dj2) + Dj3 + Dj5 + Dj6;
    if (Dsum & 1) {
        return false;
    }

    int parity = static_cast<int>((Dsum / 2) & 1);
    // A forbidden endpoint may underflow to -0; its sign still fixes the row.
    if (negative_double(values[n - 1])) {
        parity ^= 1;
    }

    scale = (parity != 0 ? -1.0 : 1.0) / std::sqrt(norm);
    return finite_double(scale);
}

inline bool compute_wigner6j_x_series(int Dx_min,
                                      int Dx_max,
                                      int Dj2,
                                      int Dj3,
                                      int Dj4,
                                      int Dj5,
                                      int Dj6,
                                      std::vector<double>& out)
{
    if (!valid_complete_wigner6j_x_range(
            Dx_min, Dx_max, Dj2, Dj3, Dj5, Dj6)) {
        out.clear();
        return false;
    }

    const std::int64_t span =
        static_cast<std::int64_t>(Dx_max) - static_cast<std::int64_t>(Dx_min);
    if (span < 0 || (span & 1)) {
        out.clear();
        return false;
    }

    const std::size_t n = static_cast<std::size_t>(span / 2 + 1);
    out.resize(n);

    if (n == 0) {
        return false;
    }

    if (n == 1) {
        out[0] = 1.0;
    } else {
        const bool use_zero_branch = (Dx_min == 0);
        Wigner6jXRecurrence xyz(Dj2, Dj3, Dj4, Dj5, Dj6, use_zero_branch);
        if (!rec_luscombe_luban_6j(Dx_min, Dx_max, xyz, out)) {
            return false;
        }
    }

    double scale = 0.0;
    if (!wigner6j_normalization_scale(
            Dx_min, n, Dj2, Dj3, Dj4, Dj5, Dj6, out, scale)) {
        return false;
    }

#pragma omp simd
    for (std::size_t i = 0; i < n; ++i) {
        out[i] *= scale;
    }

    return true;
}



// ============================================================================
// Prepared variable-j4 recurrence for repeated 6j sweeps.
//
// This is intended for hot loops such as sjC, where the recurrence variable and
// four of the six labels are fixed across an inner loop and only j4 changes:
//
//     { x  j2  j3
//       j4 j5  j6 }
//
// with recurrence over x.  The common off-diagonal coupling and the non-j4
// part of Y(x) are prepared once.  For each j4 we derive X/Z from adjacent
// coupling entries and form Y(x) = Y_base(x) + f4 * Y_f4(x), then run the
// same Luscombe-Luban ratio recurrence and normalization. This avoids repeated
// pair-factor cache lookups and coefficient-array construction in loops that
// sweep over j4 many times.
// ============================================================================

struct Wigner6jPreparedVariableJ4 {
    int Dx_min = 0;
    int Dx_max = -2;
    int Dj2 = 0;
    int Dj3 = 0;
    int Dj5 = 0;
    int Dj6 = 0;

    std::size_t n = 0;
    bool valid = false;
    bool jmin_is_zero = false;

    std::vector<double> coupling;
    std::vector<double> Y_base;
    std::vector<double> Y_f4;
};

struct Wigner6jPreparedVariableJ4Workspace {
    // Reused output buffer for direct-store prepared calls.
    std::vector<double> values;
};

inline bool prepare_wigner6j_x_series_variable_j4(int Dx_min,
                                                  int Dx_max,
                                                  int Dj2,
                                                  int Dj3,
                                                  int Dj5,
                                                  int Dj6,
                                                  Wigner6jPreparedVariableJ4& prep)
{
    prep.Dx_min = Dx_min;
    prep.Dx_max = Dx_max;
    prep.Dj2 = Dj2;
    prep.Dj3 = Dj3;
    prep.Dj5 = Dj5;
    prep.Dj6 = Dj6;
    prep.n = 0;
    prep.valid = false;
    prep.jmin_is_zero = false;

    if (!valid_complete_wigner6j_x_range(
            Dx_min, Dx_max, Dj2, Dj3, Dj5, Dj6)) {
        prep.coupling.clear();
        prep.Y_base.clear();
        prep.Y_f4.clear();
        return false;
    }

    const std::int64_t span =
        static_cast<std::int64_t>(Dx_max) - static_cast<std::int64_t>(Dx_min);
    if (span < 0 || (span & 1)) {
        prep.coupling.clear();
        prep.Y_base.clear();
        prep.Y_f4.clear();
        return false;
    }

    const std::size_t n = static_cast<std::size_t>(span / 2 + 1);
    prep.n = n;
    prep.coupling.resize(n);
    prep.Y_base.resize(n);
    prep.Y_f4.resize(n);

    if (n == 0) {
        return false;
    }

    // Single-value recurrences are handled directly in the compute call.  The
    // prepared arrays are intentionally left valid but not meaningful.
    if (n == 1) {
        prep.valid = true;
        prep.jmin_is_zero = (Dx_min == 0);
        return true;
    }

    const double j2 = half_spin(Dj2);
    const double j3 = half_spin(Dj3);
    const double j5 = half_spin(Dj5);
    const double j6 = half_spin(Dj6);

    const double f2 = t_of_j(j2);
    const double f3 = t_of_j(j3);
    const double f5 = t_of_j(j5);
    const double f6 = t_of_j(j6);

    prep.jmin_is_zero = (Dx_min == 0);

    if (prep.jmin_is_zero) {
        for (std::size_t i = 0; i < n; ++i) {
            const int D = doubled_axis_value(Dx_min, i);
            const double j = half_spin(D);
            const double q = j * (j + 1.0);
            const double twoj1 = 2.0 * j + 1.0;

            prep.Y_base[i] = twoj1 * (-q + 2.0 * (f2 + f5));
            prep.Y_f4[i]  = twoj1 * (-2.0);
            prep.coupling[i] = compute_zero_branch_coupling_value(
                Dj2, Dj3, Dj5, Dj6, D);
        }

        prep.valid = true;
        return true;
    }

    const double Y1_base = f2 + f3 + f5 + f6;
    const double Y0_base = (f5 - f6) * (f2 - f3);

    const int required_Dmax = Dx_max;
    const PairFactorSeries& g23 =
        get_pair_factor_values(Dj2, Dj3, required_Dmax);
    const PairFactorSeries& g56 =
        get_pair_factor_values(Dj5, Dj6, required_Dmax);
    const std::size_t offset23 = static_cast<std::size_t>(
        (Dx_min - g23.Dmin) / 2);
    const std::size_t offset56 = static_cast<std::size_t>(
        (Dx_min - g56.Dmin) / 2);

#pragma omp simd
    for (std::size_t i = 0; i < n; ++i) {
        const int D = doubled_axis_value(Dx_min, i);
        const double j = half_spin(D);
        const double q = j * (j + 1.0);
        const double twoj1 = 2.0 * j + 1.0;

        prep.Y_base[i] = twoj1 * std::fma(q, Y1_base - q, Y0_base);
        prep.Y_f4[i]  = twoj1 * (-2.0 * q);
        prep.coupling[i] = multiply_pair_factors(
            {g23.values[offset23 + i], g23.corrections[offset23 + i]},
            {g56.values[offset56 + i], g56.corrections[offset56 + i]});
    }

    prep.valid = true;
    return true;
}

inline bool rec_luscombe_luban_6j_prepared_j4(const Wigner6jPreparedVariableJ4& prep,
                                              int Dj4,
                                              std::vector<double>& out,
                                              Wigner6jPreparedVariableJ4Workspace& /*ws*/)
{
    if (!prep.valid || prep.n == 0) {
        out.clear();
        return false;
    }

    const std::size_t n = prep.n;
    out.resize(n);

    if (n == 1) {
        out[0] = 1.0;
        return true;
    }

    const double j4 = half_spin(Dj4);
    const double f4 = t_of_j(j4);

    auto X = [&prep](std::size_t i) noexcept -> double {
        if (prep.jmin_is_zero) {
            return prep.coupling[i + 1];
        }
        const double j = half_spin(doubled_axis_value(prep.Dx_min, i));
        return j * prep.coupling[i + 1];
    };
    auto Y = [&prep, f4](std::size_t i) noexcept -> double {
        return std::fma(f4, prep.Y_f4[i], prep.Y_base[i]);
    };
    auto Z = [&prep](std::size_t i) noexcept -> double {
        if (prep.jmin_is_zero) {
            return prep.coupling[i];
        }
        const double j = half_spin(doubled_axis_value(prep.Dx_min, i));
        return (j + 1.0) * prep.coupling[i];
    };

    return reconstruct_luscombe_luban_6j(n, out, X, Y, Z);
}

inline bool compute_wigner6j_x_series_prepared_j4_raw(
    const Wigner6jPreparedVariableJ4& prep,
    int Dj4,
    std::vector<double>& out,
    Wigner6jPreparedVariableJ4Workspace& ws,
    double& scale)
{
    if (!prep.valid || prep.n == 0) {
        out.clear();
        return false;
    }

    if (prep.n == 1) {
        out.resize(1);
        out[0] = 1.0;
    } else if (!rec_luscombe_luban_6j_prepared_j4(prep, Dj4, out, ws)) {
        return false;
    }

    return wigner6j_normalization_scale(
        prep.Dx_min, prep.n, prep.Dj2, prep.Dj3, Dj4,
        prep.Dj5, prep.Dj6, out, scale);
}

inline bool compute_wigner6j_x_series_prepared_j4(
    const Wigner6jPreparedVariableJ4& prep,
    int Dj4,
    std::vector<double>& out,
    Wigner6jPreparedVariableJ4Workspace& ws)
{
    double scale = 0.0;
    if (!compute_wigner6j_x_series_prepared_j4_raw(
            prep, Dj4, out, ws, scale)) {
        return false;
    }

#pragma omp simd
    for (std::size_t i = 0; i < prep.n; ++i) {
        out[i] *= scale;
    }

    return true;
}


inline bool compute_wigner6j_x_series_prepared_j4_store_scaled(
    const Wigner6jPreparedVariableJ4& prep,
    int Dj4,
    double pref,
    std::size_t rec_off0,
    std::size_t n_store,
    double* __restrict dst,
    Wigner6jPreparedVariableJ4Workspace& ws)
{
    if (n_store == 0u) {
        return true;
    }

    double scale = 0.0;
    if (!compute_wigner6j_x_series_prepared_j4_raw(
            prep, Dj4, ws.values, ws, scale)) {
        return false;
    }

    if (rec_off0 > ws.values.size() ||
        n_store > ws.values.size() - rec_off0) {
        return false;
    }

#pragma omp simd
    for (std::size_t t = 0; t < n_store; ++t) {
        dst[t] = (pref * scale) * ws.values[rec_off0 + t];
    }

    return true;
}

inline std::size_t x_offset_from_Dx(int Dx, int Dx_min) noexcept {
    return static_cast<std::size_t>((Dx - Dx_min) / 2);
}

} // namespace sjrec_detail

inline void fill_all_sj_tensors_rec(
                             const B4Spins& s,
                             int Delta_l,
                             const std::vector<int>& xvals,
                             const std::vector<int>& i1,
                             const std::vector<int>& k2,
                             const std::vector<int>& k3,
                             const std::vector<int>& k4,
                             const std::vector<int>& k5,
                             double* __restrict__ sjA,
                             double* __restrict__ sjB,
                             double* __restrict__ sjC,
                             double* __restrict__ sjD,
                             double* __restrict__ sjE)
{
    // Seed-free Luscombe-Luban recurrence: forward/backward ratios determine
    // the shape, orthogonality fixes the norm, and the analytic endpoint phase
    // fixes the sign.  No on-disk 6j table is involved.
    const std::size_t Dim_x  = xvals.size();
    const std::size_t Dim_i1 = i1.size();
    const std::size_t Dim_k2 = k2.size();
    const std::size_t Dim_k3 = k3.size();
    const std::size_t Dim_k4 = k4.size();
    const std::size_t Dim_k5 = k5.size();
    const std::size_t Dim_l  = static_cast<std::size_t>(Delta_l + 1);
    
    if (Dim_x == 0) {
        return;
    }
    
    const int Dj12 = s.Dj12, Dj13 = s.Dj13, Dj14 = s.Dj14, Dj15 = s.Dj15;
    const int Dj23 = s.Dj23, Dj24 = s.Dj24, Dj25 = s.Dj25;
    const int Dj34 = s.Dj34, Dj35 = s.Dj35, Dj45 = s.Dj45;
    
    const int wig_max_two_j = std::max({
        s.max() + 2 * Delta_l,
        xvals.back(), i1.back(), k2.back(), k3.back(), k4.back(), k5.back()
    });
    
    const int Dx0 = xvals.front();
    const int x_parity = Dx0 & 1;
    
    const int* __restrict__ i1_data = i1.data();
    const int* __restrict__ k2_data = k2.data();
    const int* __restrict__ k3_data = k3.data();
    const int* __restrict__ k4_data = k4.data();
    const int* __restrict__ k5_data = k5.data();
    const int* __restrict__ x_data  = xvals.data();
    
    // Column-major old layout: x is stride-1.
    const std::size_t A_sa   = Dim_x;
    const std::size_t A_sk5  = Dim_x * Dim_i1;
    const std::size_t A_sl25 = Dim_x * Dim_i1 * Dim_k5;
    
    const std::size_t B_sk5  = Dim_x;
    const std::size_t B_sk4  = Dim_x * Dim_k5;
    const std::size_t B_sl35 = Dim_x * Dim_k5 * Dim_k4;
    const std::size_t B_sl45 = Dim_x * Dim_k5 * Dim_k4 * Dim_l;
    
    const std::size_t C_sk4  = Dim_x;
    const std::size_t C_sk3  = Dim_x * Dim_k4;
    const std::size_t C_sl24 = Dim_x * Dim_k4 * Dim_k3;
    const std::size_t C_sl34 = Dim_x * Dim_k4 * Dim_k3 * Dim_l;
    const std::size_t C_sl35 = Dim_x * Dim_k4 * Dim_k3 * Dim_l * Dim_l;
    
    const std::size_t D_sk3  = Dim_x;
    const std::size_t D_sk2  = Dim_x * Dim_k3;
    const std::size_t D_sl23 = Dim_x * Dim_k3 * Dim_k2;
    const std::size_t D_sl24 = Dim_x * Dim_k3 * Dim_k2 * Dim_l;
    
    const std::size_t E_sa   = Dim_x;
    const std::size_t E_sk2  = Dim_x * Dim_i1;
    const std::size_t E_sl25 = Dim_x * Dim_i1 * Dim_k2;
    
    const int chunk_size = 16;
    
    // =========================================================================
    // Precompute prefactor pieces.
    // =========================================================================
    std::vector<double> sqrt_i1(Dim_i1);
    std::vector<double> sqrt_k2(Dim_k2);
    std::vector<double> sqrt_k3(Dim_k3);
    std::vector<double> sqrt_k4(Dim_k4);
    std::vector<double> sqrt_k5(Dim_k5);
    
    for (std::size_t i = 0; i < Dim_i1; ++i) {
        sqrt_i1[i] = std::sqrt(static_cast<double>(i1_data[i] + 1));
    }
    
    for (std::size_t i = 0; i < Dim_k2; ++i) {
        sqrt_k2[i] = std::sqrt(static_cast<double>(k2_data[i] + 1));
    }
    
    for (std::size_t i = 0; i < Dim_k3; ++i) {
        sqrt_k3[i] = std::sqrt(static_cast<double>(k3_data[i] + 1));
    }
    
    for (std::size_t i = 0; i < Dim_k4; ++i) {
        sqrt_k4[i] = std::sqrt(static_cast<double>(k4_data[i] + 1));
    }
    
    for (std::size_t i = 0; i < Dim_k5; ++i) {
        sqrt_k5[i] = std::sqrt(static_cast<double>(k5_data[i] + 1));
    }
    
    const Wigner15jRealPhaseSplit phase_split =
        wigner15j_real_phase_split(s);
    const int PrefE = phase_split.e_correction * pm1_from_even_Dsum(
        Dj12 - Dj35 - Dj34 - Dj24 - phase_split.e_parity);
    
    std::vector<double> x_weight(Dim_x);
    
    for (std::size_t ix = 0; ix < Dim_x; ++ix) {
        x_weight[ix] = static_cast<double>(x_data[ix] + 1);
    }
    
    // =========================================================================
    // O(1) symmetry mappings for sjC.
    // =========================================================================
    const int max_two_j = wig_max_two_j + 10;
    
    std::vector<int> val_to_ik3(max_two_j, -1);
    for (std::size_t i = 0; i < Dim_k3; ++i) {
        val_to_ik3[k3_data[i]] = static_cast<int>(i);
    }
    
    std::vector<int> val_to_ik4(max_two_j, -1);
    for (std::size_t i = 0; i < Dim_k4; ++i) {
        val_to_ik4[k4_data[i]] = static_cast<int>(i);
    }
    
    std::vector<int> val_to_il24(max_two_j, -1);
    for (std::size_t i = 0; i < Dim_l; ++i) {
        val_to_il24[Dj24 + 2 * static_cast<int>(i)] = static_cast<int>(i);
    }
    
    std::vector<int> val_to_il35(max_two_j, -1);
    for (std::size_t i = 0; i < Dim_l; ++i) {
        val_to_il35[Dj35 + 2 * static_cast<int>(i)] = static_cast<int>(i);
    }
    
#pragma omp parallel
    {
        // Reused per-thread buffer for one full x-string.
        std::vector<double> rec_values;
        
        // ================================================================
        // sjA(x, i1, k5, ell_25)
        //
        // Old symbol:
        //      { i1  l25  x
        //        k5  j14  j15 }
        //
        // Rotate columns to make x the recurrence variable:
        //      { x   i1   l25
        //        j15 k5   j14 }
        //
        // Recurrence args:
        //      j2=i1, j3=l25, j4=j15, j5=k5, j6=j14.
        // ================================================================
#pragma omp for collapse(3) schedule(dynamic, chunk_size)
        for (std::size_t il25 = 0; il25 < Dim_l; ++il25) {
            for (std::size_t ik5 = 0; ik5 < Dim_k5; ++ik5) {
                for (std::size_t ia = 0; ia < Dim_i1; ++ia) {
                    
                    const int Dl25 = Dj25 + 2 * static_cast<int>(il25);
                    const int Dk5  = k5_data[ik5];
                    const int Di1  = i1_data[ia];
                    
                    if (!triangle_condition(Dk5, Dl25, Dj15) ||
                        !triangle_condition(Di1, Dj14, Dj15)) {
                        continue;
                    }
                    
                    if (((Di1 ^ Dl25 ^ x_parity) |
                         (Dk5 ^ Dj14 ^ x_parity)) & 1) {
                        continue;
                    }
                    
                    const int Dx_min = std::max(iabs_int(Di1 - Dl25),
                                                iabs_int(Dk5 - Dj14));
                    const int Dx_max = std::min(Di1 + Dl25,
                                                Dk5 + Dj14);
                    
                    if (Dx_min > Dx_max || Dx_max < Dx0) {
                        continue;
                    }
                    
                    const auto it_begin =
                    std::lower_bound(xvals.begin(), xvals.end(), Dx_min);
                    
                    const auto it_end =
                    std::upper_bound(xvals.begin(), xvals.end(), Dx_max);
                    
                    const std::size_t ix_begin =
                    static_cast<std::size_t>(it_begin - xvals.begin());
                    
                    const std::size_t ix_end =
                    static_cast<std::size_t>(it_end - xvals.begin());
                    
                    if (ix_begin >= ix_end) {
                        continue;
                    }
                    
                    const bool ok =
                    sjrec_detail::compute_wigner6j_x_series(
                                                            Dx_min, Dx_max,
                                                            Di1, Dl25, Dj15, Dk5, Dj14,
                                                            rec_values);
                    
                    if (!ok) {
                        continue;
                    }
                    
                    const std::size_t base =
                    il25 * A_sl25 + ik5 * A_sk5 + ia * A_sa;
                    
                    const double PrefA =
                    static_cast<double>(
                                        pm1_from_even_Dsum(
                                            Di1 + Dk5 + Dl25 + Dj15 -
                                            phase_split.a_parity))
                    * sqrt_i1[ia] * sqrt_k5[ik5];
                    
                    for (std::size_t ix = ix_begin; ix < ix_end; ++ix) {
                        const int Dx = x_data[ix];
                        const std::size_t off =
                        sjrec_detail::x_offset_from_Dx(Dx, Dx_min);
                        
                        sjA[base + ix] =
                        PrefA * x_weight[ix] * rec_values[off];
                    }
                }
            }
        }
        
        // ================================================================
        // sjB(x, k5, k4, ell_35, ell_45)
        //
        // Old:
        //      { k5  j14  x
        //        k4  l35  l45 }
        //
        // Rotated:
        //      { x   k5   j14
        //        l45 k4   l35 }
        //
        // Recurrence args:
        //      j2=k5, j3=j14, j4=l45, j5=k4, j6=l35.
        // ================================================================
#pragma omp for collapse(4) schedule(dynamic, chunk_size)
        for (std::size_t il45 = 0; il45 < Dim_l; ++il45) {
            for (std::size_t il35 = 0; il35 < Dim_l; ++il35) {
                for (std::size_t ik4 = 0; ik4 < Dim_k4; ++ik4) {
                    for (std::size_t ik5 = 0; ik5 < Dim_k5; ++ik5) {
                        
                        const int Dl45 = Dj45 + 2 * static_cast<int>(il45);
                        const int Dl35 = Dj35 + 2 * static_cast<int>(il35);
                        const int Dk4  = k4_data[ik4];
                        const int Dk5  = k5_data[ik5];
                        
                        if (!triangle_condition(Dk5, Dl35, Dl45) ||
                            !triangle_condition(Dk4, Dj14, Dl45)) {
                            continue;
                        }
                        
                        if (((Dk5 ^ Dj14 ^ x_parity) |
                             (Dk4 ^ Dl35 ^ x_parity)) & 1) {
                            continue;
                        }
                        
                        const int Dx_min = std::max(iabs_int(Dk4 - Dl35),
                                                    iabs_int(Dk5 - Dj14));
                        const int Dx_max = std::min(Dk4 + Dl35,
                                                    Dk5 + Dj14);
                        
                        if (Dx_min > Dx_max || Dx_max < Dx0) {
                            continue;
                        }
                        
                        const auto it_begin =
                        std::lower_bound(xvals.begin(), xvals.end(), Dx_min);
                        
                        const auto it_end =
                        std::upper_bound(xvals.begin(), xvals.end(), Dx_max);
                        
                        const std::size_t ix_begin =
                        static_cast<std::size_t>(it_begin - xvals.begin());
                        
                        const std::size_t ix_end =
                        static_cast<std::size_t>(it_end - xvals.begin());
                        
                        if (ix_begin >= ix_end) {
                            continue;
                        }
                        
                        const bool ok =
                        sjrec_detail::compute_wigner6j_x_series(
                                                                Dx_min, Dx_max,
                                                                Dk5, Dj14, Dl45, Dk4, Dl35,
                                                                rec_values);
                        
                        if (!ok) {
                            continue;
                        }
                        
                        const std::size_t base =
                        il45 * B_sl45
                        + il35 * B_sl35
                        + ik4  * B_sk4
                        + ik5  * B_sk5;
                        
                        const double PrefB =
                        static_cast<double>(
                                            pm1_from_even_Dsum(Dk4 + Dl45 + Dj14 +
                                                               Dl35 + Dj35))
                        * sqrt_k4[ik4];
                        
                        for (std::size_t ix = ix_begin; ix < ix_end; ++ix) {
                            const int Dx = x_data[ix];
                            const std::size_t off =
                            sjrec_detail::x_offset_from_Dx(Dx, Dx_min);
                            
                            sjB[base + ix] = PrefB * rec_values[off];
                        }
                    }
                }
            }
        }
        
        // ================================================================
        // sjC(x, k4, k3, ell_24, ell_34, ell_35)
        //
        // Old:
        //      { k4  l35  x
        //        k3  l24  l34 }
        //
        // Rotated:
        //      { x   k4   l35
        //        l34 k3   l24 }
        //
        // Recurrence args:
        //      j2=k4, j3=l35, j4=l34, j5=k3, j6=l24.
        //
        // This is the big one. If this does not move the profiling needle,
        // the profiler has been replaced by a decorative houseplant.
        // ================================================================
        if (sjC != nullptr) {
#pragma omp for collapse(5) schedule(dynamic, chunk_size)
            for (std::size_t il35 = 0; il35 < Dim_l; ++il35) {
                for (std::size_t il34 = 0; il34 < Dim_l; ++il34) {
                    for (std::size_t il24 = 0; il24 < Dim_l; ++il24) {
                        for (std::size_t ik3 = 0; ik3 < Dim_k3; ++ik3) {
                            for (std::size_t ik4 = 0; ik4 < Dim_k4; ++ik4) {
                            
                            const int Dl35 = Dj35 + 2 * static_cast<int>(il35);
                            const int Dl34 = Dj34 + 2 * static_cast<int>(il34);
                            const int Dl24 = Dj24 + 2 * static_cast<int>(il24);
                            const int Dk3  = k3_data[ik3];
                            const int Dk4  = k4_data[ik4];
                            
                            // Symmetry:
                            //
                            // { k4 l35 x ; k3 l24 l34 }
                            //
                            // equals the mirrored entry used below. No phase.
                            int sym_ik4  = val_to_ik4[Dk3];
                            int sym_il35 = val_to_il35[Dl24];
                            int sym_ik3  = val_to_ik3[Dk4];
                            int sym_il24 = val_to_il24[Dl35];
                            
                            bool sym_exists =
                            (sym_ik4  != -1 &&
                             sym_il35 != -1 &&
                             sym_ik3  != -1 &&
                             sym_il24 != -1);
                            
                            bool is_diagonal = false;
                            
                            if (sym_exists) {
                                if (Dk4 < Dk3) {
                                    continue;
                                }
                                
                                if (Dk4 == Dk3 && Dl35 < Dl24) {
                                    continue;
                                }
                                
                                if (Dk4 == Dk3 && Dl35 == Dl24) {
                                    is_diagonal = true;
                                }
                            }
                            
                            if (!triangle_condition(Dk3, Dl35, Dl34) ||
                                !triangle_condition(Dk4, Dl24, Dl34)) {
                                continue;
                            }
                            
                            if (((Dk4 ^ Dl35 ^ x_parity) |
                                 (Dk3 ^ Dl24 ^ x_parity)) & 1) {
                                continue;
                            }
                            
                            const int Dx_min = std::max(iabs_int(Dk4 - Dl35),
                                                        iabs_int(Dk3 - Dl24));
                            const int Dx_max = std::min(Dk4 + Dl35,
                                                        Dk3 + Dl24);
                            
                            if (Dx_min > Dx_max || Dx_max < Dx0) {
                                continue;
                            }
                            
                            const auto it_begin =
                            std::lower_bound(xvals.begin(), xvals.end(), Dx_min);
                            
                            const auto it_end =
                            std::upper_bound(xvals.begin(), xvals.end(), Dx_max);
                            
                            const std::size_t ix_begin =
                            static_cast<std::size_t>(it_begin - xvals.begin());
                            
                            const std::size_t ix_end =
                            static_cast<std::size_t>(it_end - xvals.begin());
                            
                            if (ix_begin >= ix_end) {
                                continue;
                            }
                            
                            const bool ok =
                            sjrec_detail::compute_wigner6j_x_series(
                                                                    Dx_min, Dx_max,
                                                                    Dk4, Dl35, Dl34, Dk3, Dl24,
                                                                    rec_values);
                            
                            if (!ok) {
                                continue;
                            }
                            
                            const std::size_t base =
                            il35 * C_sl35
                            + il34 * C_sl34
                            + il24 * C_sl24
                            + ik3  * C_sk3
                            + ik4  * C_sk4;
                            
                            std::size_t sym_base = 0;
                            
                            if (sym_exists && !is_diagonal) {
                                sym_base =
                                static_cast<std::size_t>(sym_il35) * C_sl35
                                + il34 * C_sl34
                                + static_cast<std::size_t>(sym_il24) * C_sl24
                                + static_cast<std::size_t>(sym_ik3)  * C_sk3
                                + static_cast<std::size_t>(sym_ik4)  * C_sk4;
                            }
                            
                            const int PrefC =
                            pm1_from_even_Dsum(Dl34 + Dj34);
                            
                            for (std::size_t ix = ix_begin; ix < ix_end; ++ix) {
                                const int Dx = x_data[ix];
                                const std::size_t off =
                                sjrec_detail::x_offset_from_Dx(Dx, Dx_min);
                                
                                const double out_val =
                                static_cast<double>(PrefC) * rec_values[off];
                                
                                sjC[base + ix] = out_val;
                                
                                if (sym_exists && !is_diagonal) {
                                    sjC[sym_base + ix] = out_val;
                                }
                            }
                            }
                        }
                    }
                }
            }
        }
        
        // ================================================================
        // sjD(x, k3, k2, ell_23, ell_24)
        //
        // Old:
        //      { k3  l24  x
        //        k2  j13  l23 }
        //
        // Rotated:
        //      { x   k3   l24
        //        l23 k2   j13 }
        //
        // Recurrence args:
        //      j2=k3, j3=l24, j4=l23, j5=k2, j6=j13.
        // ================================================================
#pragma omp for collapse(4) schedule(dynamic, chunk_size)
        for (std::size_t il24 = 0; il24 < Dim_l; ++il24) {
            for (std::size_t il23 = 0; il23 < Dim_l; ++il23) {
                for (std::size_t ik2 = 0; ik2 < Dim_k2; ++ik2) {
                    for (std::size_t ik3 = 0; ik3 < Dim_k3; ++ik3) {
                        
                        const int Dl24 = Dj24 + 2 * static_cast<int>(il24);
                        const int Dl23 = Dj23 + 2 * static_cast<int>(il23);
                        const int Dk2  = k2_data[ik2];
                        const int Dk3  = k3_data[ik3];
                        
                        if (!triangle_condition(Dk2, Dl24, Dl23) ||
                            !triangle_condition(Dk3, Dj13, Dl23)) {
                            continue;
                        }
                        
                        if (((Dk3 ^ Dl24 ^ x_parity) |
                             (Dk2 ^ Dj13 ^ x_parity)) & 1) {
                            continue;
                        }
                        
                        const int Dx_min = std::max(iabs_int(Dk3 - Dl24),
                                                    iabs_int(Dk2 - Dj13));
                        const int Dx_max = std::min(Dk3 + Dl24,
                                                    Dk2 + Dj13);
                        
                        if (Dx_min > Dx_max || Dx_max < Dx0) {
                            continue;
                        }
                        
                        const auto it_begin =
                        std::lower_bound(xvals.begin(), xvals.end(), Dx_min);
                        
                        const auto it_end =
                        std::upper_bound(xvals.begin(), xvals.end(), Dx_max);
                        
                        const std::size_t ix_begin =
                        static_cast<std::size_t>(it_begin - xvals.begin());
                        
                        const std::size_t ix_end =
                        static_cast<std::size_t>(it_end - xvals.begin());
                        
                        if (ix_begin >= ix_end) {
                            continue;
                        }
                        
                        const bool ok =
                        sjrec_detail::compute_wigner6j_x_series(
                                                                Dx_min, Dx_max,
                                                                Dk3, Dl24, Dl23, Dk2, Dj13,
                                                                rec_values);
                        
                        if (!ok) {
                            continue;
                        }
                        
                        const std::size_t base =
                        il24 * D_sl24
                        + il23 * D_sl23
                        + ik2  * D_sk2
                        + ik3  * D_sk3;
                        
                        const double PrefD =
                        static_cast<double>(
                                            pm1_from_even_Dsum(Dk2 + Dk3 + Dl23 +
                                                               Dj13 + Dl24 + Dj24 -
                                                               phase_split.d_parity))
                        * sqrt_k2[ik2] * sqrt_k3[ik3];
                        
                        for (std::size_t ix = ix_begin; ix < ix_end; ++ix) {
                            const int Dx = x_data[ix];
                            const std::size_t off =
                            sjrec_detail::x_offset_from_Dx(Dx, Dx_min);
                            
                            sjD[base + ix] = PrefD * rec_values[off];
                        }
                    }
                }
            }
        }
        
        // ================================================================
        // sjE(x, i1, k2, ell_25)
        //
        // Old:
        //      { i1  l25  x
        //        k2  j13  j12 }
        //
        // Rotated:
        //      { x   i1   l25
        //        j12 k2   j13 }
        //
        // Recurrence args:
        //      j2=i1, j3=l25, j4=j12, j5=k2, j6=j13.
        // ================================================================
#pragma omp for collapse(3) schedule(dynamic, chunk_size)
        for (std::size_t il25 = 0; il25 < Dim_l; ++il25) {
            for (std::size_t ik2 = 0; ik2 < Dim_k2; ++ik2) {
                for (std::size_t ia = 0; ia < Dim_i1; ++ia) {
                    
                    const int Dl25 = Dj25 + 2 * static_cast<int>(il25);
                    const int Dk2  = k2_data[ik2];
                    const int Di1  = i1_data[ia];
                    
                    if (!triangle_condition(Dk2, Dl25, Dj12) ||
                        !triangle_condition(Di1, Dj13, Dj12)) {
                        continue;
                    }
                    
                    if (((Di1 ^ Dl25 ^ x_parity) |
                         (Dk2 ^ Dj13 ^ x_parity)) & 1) {
                        continue;
                    }
                    
                    const int Dx_min = std::max(iabs_int(Di1 - Dl25),
                                                iabs_int(Dk2 - Dj13));
                    const int Dx_max = std::min(Di1 + Dl25,
                                                Dk2 + Dj13);
                    
                    if (Dx_min > Dx_max || Dx_max < Dx0) {
                        continue;
                    }
                    
                    const auto it_begin =
                    std::lower_bound(xvals.begin(), xvals.end(), Dx_min);
                    
                    const auto it_end =
                    std::upper_bound(xvals.begin(), xvals.end(), Dx_max);
                    
                    const std::size_t ix_begin =
                    static_cast<std::size_t>(it_begin - xvals.begin());
                    
                    const std::size_t ix_end =
                    static_cast<std::size_t>(it_end - xvals.begin());
                    
                    if (ix_begin >= ix_end) {
                        continue;
                    }
                    
                    const bool ok =
                    sjrec_detail::compute_wigner6j_x_series(
                                                            Dx_min, Dx_max,
                                                            Di1, Dl25, Dj12, Dk2, Dj13,
                                                            rec_values);
                    
                    if (!ok) {
                        continue;
                    }
                    
                    const std::size_t base =
                    il25 * E_sl25 + ik2 * E_sk2 + ia * E_sa;
                    
                    for (std::size_t ix = ix_begin; ix < ix_end; ++ix) {
                        const int Dx = x_data[ix];
                        const std::size_t off =
                        sjrec_detail::x_offset_from_Dx(Dx, Dx_min);
                        
                        sjE[base + ix] =
                        static_cast<double>(PrefE) * rec_values[off];
                    }
                }
            }
        }
    }
}

// Fill one flattened (ell_24, ell_34, ell_35) interval of sjC using the
// recurrence in x.  The flattened global id is
//
//     q + Dim_q * (r + Dim_r * s),
//
// and the output stores only the requested interval, with local z starting at
// zero and layout [x, k4, k3, z]:
//
//     x + X * (ik4 + B4 * (ik3 + B3 * z)).
//
// When an exact reflection partner also resides in the requested interval,
// this routine computes the same canonical half as fill_all_sj_tensors_rec and
// mirrors it locally.  A partner outside the interval cannot be written, so
// that boundary entry is computed independently instead.
inline void fill_sjC_tensor_rec_qrs_tile(
    const B4Spins& s,
    const std::vector<int>& xvals,
    const std::vector<int>& k3,
    const std::vector<int>& k4,
    std::size_t qrs_begin,
    std::size_t qrs_count,
    std::size_t Dim_q,
    std::size_t Dim_r,
    std::size_t Dim_s,
    double* __restrict__ sjC_qrs)
{
    const auto checked_mul_size = [](std::size_t a,
                                     std::size_t b,
                                     const char* what) -> std::size_t {
        if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
            throw std::overflow_error(std::string("size overflow for ") + what);
        }
        return a * b;
    };

    if (Dim_q == 0 || Dim_r == 0 || Dim_s == 0) {
        throw std::invalid_argument("sjC q/r/s dimensions must be nonzero");
    }

    const std::size_t Dim_qr = checked_mul_size(Dim_q, Dim_r, "sjC q/r extent");
    const std::size_t Dim_qrs = checked_mul_size(Dim_qr, Dim_s, "sjC q/r/s extent");
    if (qrs_begin > Dim_qrs || qrs_count > Dim_qrs - qrs_begin) {
        throw std::out_of_range("sjC q/r/s interval is outside the supplied axes");
    }

    const auto validate_lifted_axis = [](int base,
                                         std::size_t extent,
                                         const char* name) {
        if (base < 0) {
            throw std::invalid_argument(std::string(name) + " has a negative base spin");
        }
        const std::size_t max_index = extent - 1;
        const std::size_t max_safe_index = static_cast<std::size_t>(
            (std::numeric_limits<int>::max() - base) / 2);
        if (max_index > max_safe_index) {
            throw std::overflow_error(std::string(name) + " lifted spin exceeds int");
        }
    };
    validate_lifted_axis(s.Dj24, Dim_q, "ell_24");
    validate_lifted_axis(s.Dj34, Dim_r, "ell_34");
    validate_lifted_axis(s.Dj35, Dim_s, "ell_35");

    if (!xvals.empty()) {
        if (xvals.front() < 0) {
            throw std::invalid_argument("sjC x values must be nonnegative");
        }
        const int parity = xvals.front() & 1;
        for (std::size_t ix = 1; ix < xvals.size(); ++ix) {
            if (xvals[ix] <= xvals[ix - 1]) {
                throw std::invalid_argument(
                    "sjC x values must be strictly increasing");
            }
            if ((xvals[ix] & 1) != parity) {
                throw std::invalid_argument(
                    "sjC x values must have a common parity");
            }
        }
    }

    if (qrs_count == 0 || xvals.empty() || k3.empty() || k4.empty()) {
        return;
    }
    if (sjC_qrs == nullptr) {
        throw std::invalid_argument("sjC q/r/s output is null for a nonempty interval");
    }

    const std::size_t Dim_x = xvals.size();
    const std::size_t Dim_k3 = k3.size();
    const std::size_t Dim_k4 = k4.size();
    std::size_t output_count = checked_mul_size(Dim_x, Dim_k4, "sjC tile output");
    output_count = checked_mul_size(output_count, Dim_k3, "sjC tile output");
    output_count = checked_mul_size(output_count, qrs_count, "sjC tile output");
    std::fill_n(sjC_qrs, output_count, 0.0);

    std::vector<int> Dl24_by_z(qrs_count);
    std::vector<int> Dl34_by_z(qrs_count);
    std::vector<int> Dl35_by_z(qrs_count);
    std::vector<std::size_t> r_by_z(qrs_count);
    for (std::size_t z = 0; z < qrs_count; ++z) {
        std::size_t qrs = qrs_begin + z;
        const std::size_t q = qrs % Dim_q;
        qrs /= Dim_q;
        const std::size_t r = qrs % Dim_r;
        const std::size_t ss = qrs / Dim_r;
        r_by_z[z] = r;
        Dl24_by_z[z] = s.Dj24 + 2 * static_cast<int>(q);
        Dl34_by_z[z] = s.Dj34 + 2 * static_cast<int>(r);
        Dl35_by_z[z] = s.Dj35 + 2 * static_cast<int>(ss);
    }

    // Build one-to-one virtual intertwiner reflection maps.  Coherent channel axes are
    // normally sorted and unique.  If a custom caller supplies duplicates,
    // conservatively disable mirroring: computing every duplicate separately
    // preserves the pre-optimization behavior and avoids ambiguous writes.
    constexpr std::size_t no_symmetry =
        std::numeric_limits<std::size_t>::max();
    bool unique_virtual_axes = true;
    std::unordered_map<int, std::size_t> ik3_by_value;
    std::unordered_map<int, std::size_t> ik4_by_value;
    ik3_by_value.reserve(Dim_k3);
    ik4_by_value.reserve(Dim_k4);
    for (std::size_t ik3 = 0; ik3 < Dim_k3; ++ik3) {
        unique_virtual_axes =
            ik3_by_value.emplace(k3[ik3], ik3).second && unique_virtual_axes;
    }
    for (std::size_t ik4 = 0; ik4 < Dim_k4; ++ik4) {
        unique_virtual_axes =
            ik4_by_value.emplace(k4[ik4], ik4).second && unique_virtual_axes;
    }

    std::vector<std::size_t> reflected_ik4(Dim_k3, no_symmetry);
    std::vector<std::size_t> reflected_ik3(Dim_k4, no_symmetry);
    if (unique_virtual_axes) {
        for (std::size_t ik3 = 0; ik3 < Dim_k3; ++ik3) {
            const auto found = ik4_by_value.find(k3[ik3]);
            if (found != ik4_by_value.end()) {
                reflected_ik4[ik3] = found->second;
            }
        }
        for (std::size_t ik4 = 0; ik4 < Dim_k4; ++ik4) {
            const auto found = ik3_by_value.find(k4[ik4]);
            if (found != ik3_by_value.end()) {
                reflected_ik3[ik4] = found->second;
            }
        }
    }

    const auto lifted_index = [no_symmetry](
                                  const int value,
                                  const int base,
                                  const std::size_t extent) {
        const std::int64_t delta =
            static_cast<std::int64_t>(value) -
            static_cast<std::int64_t>(base);
        if (delta < 0 || (delta & 1) != 0) {
            return no_symmetry;
        }
        const std::uint64_t index = static_cast<std::uint64_t>(delta / 2);
        return index < extent ? static_cast<std::size_t>(index) : no_symmetry;
    };

    // The reflection keeps r=l34 fixed and swaps
    // (k4,l35) <-> (k3,l24).  Record only partners resident in this tile.
    const std::size_t qrs_end = qrs_begin + qrs_count;
    std::vector<std::size_t> reflected_z(qrs_count, no_symmetry);
    for (std::size_t z = 0; z < qrs_count; ++z) {
        const std::size_t reflected_q = lifted_index(
            Dl35_by_z[z], s.Dj24, Dim_q);
        const std::size_t reflected_s = lifted_index(
            Dl24_by_z[z], s.Dj35, Dim_s);
        if (reflected_q == no_symmetry || reflected_s == no_symmetry) {
            continue;
        }
        const std::size_t reflected_qrs =
            reflected_q +
            Dim_q * (r_by_z[z] + Dim_r * reflected_s);
        if (reflected_qrs >= qrs_begin && reflected_qrs < qrs_end) {
            reflected_z[z] = reflected_qrs - qrs_begin;
        }
    }

    const int Dx0 = xvals.front();
    const int x_parity = Dx0 & 1;
    constexpr int chunk_size = 16;

#pragma omp parallel
    {
        std::vector<double> rec_values;

#pragma omp for collapse(3) schedule(dynamic, chunk_size)
        for (std::size_t z = 0; z < qrs_count; ++z) {
            for (std::size_t ik3 = 0; ik3 < Dim_k3; ++ik3) {
                for (std::size_t ik4 = 0; ik4 < Dim_k4; ++ik4) {
                    const int Dl24 = Dl24_by_z[z];
                    const int Dl34 = Dl34_by_z[z];
                    const int Dl35 = Dl35_by_z[z];
                    const int Dk3 = k3[ik3];
                    const int Dk4 = k4[ik4];

                    const std::size_t sym_z = reflected_z[z];
                    const std::size_t sym_ik4 = reflected_ik4[ik3];
                    const std::size_t sym_ik3 = reflected_ik3[ik4];
                    const bool tile_local_symmetry =
                        sym_z != no_symmetry &&
                        sym_ik4 != no_symmetry &&
                        sym_ik3 != no_symmetry;

                    // Exactly the canonical ordering used by the full sjC
                    // filler.  The reflected loop iteration observes the
                    // opposite ordering and performs no write, so the two
                    // output locations have a single OpenMP writer.
                    if (tile_local_symmetry &&
                        (Dk4 < Dk3 ||
                         (Dk4 == Dk3 && Dl35 < Dl24))) {
                        continue;
                    }

                    if (!triangle_condition(Dk3, Dl35, Dl34) ||
                        !triangle_condition(Dk4, Dl24, Dl34)) {
                        continue;
                    }
                    if (((Dk4 ^ Dl35 ^ x_parity) |
                         (Dk3 ^ Dl24 ^ x_parity)) & 1) {
                        continue;
                    }

                    const int Dx_min = std::max(iabs_int(Dk4 - Dl35),
                                                iabs_int(Dk3 - Dl24));
                    const int Dx_max = std::min(Dk4 + Dl35,
                                                Dk3 + Dl24);
                    if (Dx_min > Dx_max || Dx_max < Dx0) {
                        continue;
                    }

                    const auto it_begin =
                        std::lower_bound(xvals.begin(), xvals.end(), Dx_min);
                    const auto it_end =
                        std::upper_bound(xvals.begin(), xvals.end(), Dx_max);
                    const std::size_t ix_begin =
                        static_cast<std::size_t>(it_begin - xvals.begin());
                    const std::size_t ix_end =
                        static_cast<std::size_t>(it_end - xvals.begin());
                    if (ix_begin >= ix_end) {
                        continue;
                    }

                    const bool ok = sjrec_detail::compute_wigner6j_x_series(
                        Dx_min, Dx_max,
                        Dk4, Dl35, Dl34, Dk3, Dl24,
                        rec_values);
                    if (!ok) {
                        continue;
                    }

                    const std::size_t base = Dim_x *
                        (ik4 + Dim_k4 * (ik3 + Dim_k3 * z));
                    std::size_t sym_base = 0;
                    const bool is_diagonal =
                        tile_local_symmetry && sym_z == z &&
                        sym_ik4 == ik4 && sym_ik3 == ik3;
                    if (tile_local_symmetry && !is_diagonal) {
                        sym_base = Dim_x *
                            (sym_ik4 +
                             Dim_k4 * (sym_ik3 + Dim_k3 * sym_z));
                    }
                    const double pref = static_cast<double>(
                        pm1_from_even_Dsum(Dl34 + s.Dj34));
                    for (std::size_t ix = ix_begin; ix < ix_end; ++ix) {
                        const std::size_t rec_offset =
                            sjrec_detail::x_offset_from_Dx(xvals[ix], Dx_min);
                        const double value = pref * rec_values[rec_offset];
                        sjC_qrs[base + ix] = value;
                        if (tile_local_symmetry && !is_diagonal) {
                            sjC_qrs[sym_base + ix] = value;
                        }
                    }
                }
            }
        }
    }
}
