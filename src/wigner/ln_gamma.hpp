// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <cassert>

#include <complex>

struct ln_gamma { double re; double im; };

namespace detail {
constexpr double g  = 7.0;                                   // Lanczos g
constexpr double ln_sqrt2pi = 0.9189385332046727417803297364;// ½ ln(2π)

// 9‑term coefficient set (Spouge/Lanczos, used by Boost & CPython)
constexpr std::array<double,9> c = {
    0.99999999999980993227684700473478,
    676.520368121885098567009190444019,
    -1259.13921672240287047156078755283,
    771.3234287776530788486528258894,
    -176.61502916214059906584551354,
    12.507343278686904814458936853,
    -0.13857109526572011689554707,
    9.984369578019570859563e-6,
    1.50563273514931155834e-7
};

inline void sincos_fast(double x, double& s, double& c) {
#if defined(__GNUC__) && !defined(__clang__)
    __builtin_sincos(x, &s, &c);         // GCC built-in
#elif defined(__APPLE__)
    __sincos(x, &s, &c);                 // macOS libc
#else
    s = std::sin(x); c = std::cos(x);    // portable fallback
#endif
}
}//end details

// cheap branch, costs ~2 cycles if almost always taken
inline double wrap_phase(double phi) noexcept {
    if (phi <=  M_PI && phi > -M_PI)   // already principal
        return phi;
    
    return std::remainder(phi, 2.0 * M_PI);      // fallback
    
}

static inline void log_sin_pi(double xr, double yi,
                              double& logabs, double& arg) noexcept
{
    double s, c;
    detail::sincos_fast(M_PI * xr, s, c);
    
    const double a  = M_PI * yi;
    const double aa = std::abs(a);
    
    if (aa < 20.0) {
        const double sh = std::sinh(a);
        const double ch = std::cosh(a);
        const double re = s * ch;
        const double im = c * sh;
        logabs = std::log(std::hypot(re, im));
        arg    = std::atan2(im, re);
        return;
    }
    
    // scaled: cosh(a)=0.5*e^{aa}(1+e^{-2aa}), sinh(a)=0.5*e^{aa}*sgn(a)(1-e^{-2aa})
    const double e2   = std::exp(-2.0 * aa);       // safe (may underflow to 0)
    const double sgnA = (a >= 0.0) ? 1.0 : -1.0;
    
    const double re_s = s * (1.0 + e2);
    const double im_s = c * sgnA * (1.0 - e2);
    
    const double h = std::hypot(re_s, im_s);       // O(1)
    logabs = aa - M_LN2 + std::log(h);
    arg    = std::atan2(im_s, re_s);               // factor is positive
}


// ---------------------------------------------------------------------
// Forward declaration (needed for recursion)
// ---------------------------------------------------------------------
inline ln_gamma ln_gamma_one(double zr, double zi) noexcept;


// ---------------------------------------------------------------------
// 1) Reflection for Re(z) < 0.5  (uses recursion once)
// ---------------------------------------------------------------------
inline ln_gamma ln_gamma_reflect(double zr, double zi) noexcept {
    double logabs_sin, arg_sin;
    log_sin_pi(zr, zi, logabs_sin, arg_sin);
    
    ln_gamma g1 = ln_gamma_one(1.0 - zr, -zi); // Re >= 0.5
    
    ln_gamma res;
    res.re = std::log(M_PI) - logabs_sin - g1.re;
    res.im = -arg_sin       - g1.im;
    res.im = wrap_phase(res.im);
    return res;
}


// ---------------------------------------------------------------------
// 2) Core Lanczos for Re(z) ≥ 0.5
// ---------------------------------------------------------------------
inline ln_gamma ln_gamma_core(double zr, double zi) noexcept {
    using namespace detail;
    
    // Shift so the first denominator is (z) rather than (z+1)
    double x = zr - 1.0;        // real part of (z-1)
    double y = zi;              // imag part stays
    
    // A(z) = c0 + Σ_{k=1}^{8} c[k] / ( (z-1) + k )
    double sum_r = c[0];
    double sum_i = 0.0;
#pragma omp simd reduction(+:sum_r,sum_i)
    for (int k = 1; k < 9; ++k) {
        double denom_r = x + static_cast<double>(k); // = zr + k - 1
        double denom_i = y;
        double mag2    = denom_r * denom_r + denom_i * denom_i;
        double inv     = c[k] / mag2;
        sum_r += inv * denom_r;
        sum_i -= inv * denom_i;
    }
    
    // t = (z-1) + g + 0.5  = z + g - 0.5
    double tr    = x + g + 0.5;
    double ti    = y;
    double log_t = std::log(std::hypot(tr, ti));    // == 0.5*log(tr^2+ti^2)
    double theta = std::atan2(ti, tr);
    
    // (z - 0.5) = (zr - 0.5) + i zi
    double coeff_r = zr - 0.5;
    double coeff_i = zi;
    
    ln_gamma out;
    out.re = ln_sqrt2pi
    + coeff_r * log_t - coeff_i * theta   // Re((z-0.5) ln t)
    - tr                                  // -Re(t)
    + std::log(std::hypot(sum_r, sum_i)); // + ln|A|
    
    out.im = coeff_r * theta + coeff_i * log_t   // Im((z-0.5) ln t)
    - ti                                  // -Im(t)
    + std::atan2(sum_i, sum_r);           // + arg(A)
    
    // Wrap into (−π, π]
    out.im = wrap_phase(out.im);
    return out;
}


// ---------------------------------------------------------------------
// Master entry
// ---------------------------------------------------------------------
inline ln_gamma ln_gamma_one(double zr, double zi) noexcept {
    return (zr < 0.5) ? ln_gamma_reflect(zr, zi)
    : ln_gamma_core   (zr, zi);
}


// Convenience wrappers --------------------------------------------------

inline std::complex<double> ln_gamma(double zr, double zi) noexcept
{
    struct ln_gamma v = ln_gamma_one(zr, zi);
    return {v.re, v.im};
}


// log |sin(pi*(x + i y))| without overflow
static inline double log_sin_pi_abs(double xr, double yi) noexcept {
    double s, c;
    detail::sincos_fast(M_PI * xr, s, c);
    
    const double a  = M_PI * yi;
    const double aa = std::abs(a);
    
    if (aa < 20.0) {
        const double sh = std::sinh(a);
        const double ch = std::cosh(a);
        const double re = s * ch;
        const double im = c * sh;
        return std::log(std::hypot(re, im));
    }
    
    // scaled form (never overflows)
    const double e2   = std::exp(-2.0 * aa);        // may underflow to 0 (fine)
    const double sgnA = (a >= 0.0) ? 1.0 : -1.0;
    
    const double re_s = s * (1.0 + e2);
    const double im_s = c * sgnA * (1.0 - e2);
    
    return aa - M_LN2 + std::log(std::hypot(re_s, im_s));
}

// core Lanczos: log|Gamma(z)| for Re(z) >= 0.5
static inline double ln_gamma_logabs_core(double zr, double zi) noexcept {
    using namespace detail;
    
    const double x = zr - 1.0;    // z-1
    const double y = zi;
    
    // A(z) = c0 + Σ c[k]/(z-1+k)
    double sum_r = c[0];
    double sum_i = 0.0;
#pragma omp simd reduction(+:sum_r,sum_i)
    for (int k = 1; k < 9; ++k) {
        const double denom_r = x + double(k);
        const double denom_i = y;
        const double mag2    = denom_r*denom_r + denom_i*denom_i;
        const double inv     = c[k] / mag2;
        sum_r += inv * denom_r;
        sum_i -= inv * denom_i;
    }
    
    // t = z + g - 0.5
    const double tr = x + g + 0.5;
    const double ti = y;
    
    const double log_t = std::log(std::hypot(tr, ti));
    const double theta = std::atan2(ti, tr);
    
    const double coeff_r = zr - 0.5;
    const double coeff_i = zi;
    
    // Re((z-0.5) ln t) = (zr-0.5)*log|t| - zi*arg(t)
    return ln_sqrt2pi
    + coeff_r * log_t - coeff_i * theta
    - tr
    + std::log(std::hypot(sum_r, sum_i));  // ln|A|
}

// master: log|Gamma(z)| for any z (uses reflection if Re<0.5)
static inline double ln_gamma_logabs(double zr, double zi) noexcept {
    if (zr < 0.5) {
        // log|Γ(z)| = log(pi) - log|sin(pi z)| - log|Γ(1-z)|
        const double ls = log_sin_pi_abs(zr, zi);
        return std::log(M_PI) - ls - ln_gamma_logabs(1.0 - zr, -zi);
    }
    return ln_gamma_logabs_core(zr, zi);
}

