// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#ifndef COHERENT_INTERTWINER_HPP
#define COHERENT_INTERTWINER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <limits>
#include <vector>

#include "wigner/ThreeJRecurrence.hpp"
#include "core/utils.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

// All angular momenta and magnetic numbers in this header use doubled integers:
//   J  = 2j,
//   I  = 2 iota,
//   nu = 2m,
//   mu = 2M.



namespace coherent_intertwiner {

using Complex = b4utils::Complex64;
using Angles = std::array<std::array<double, 2>, 4>;

struct Spinor {
    Complex z0;
    Complex z1;
};

struct EvalOptions {
    // Treat a spinor component as zero below this threshold.
    double zero_tol = 1.0e-14;

    // Normalize input spinors before rotating and before generating coherent vectors.
    // For spinors built from angles this is harmless and protects against roundoff.
    bool normalize_spinors = true;

    // Use Kahan-style compensated summation for the W34 dot products and final mu sum.
    // Slower, but useful for validation and cancellation-heavy configurations.
    bool compensated = false;

    // Parallelize the read-only evaluation over mu-blocks with OpenMP.
    // Compile with -fopenmp. If OpenMP is unavailable, this falls back to serial.
    bool use_openmp = false;

    // Number of OpenMP threads. 0 means use omp_get_max_threads().
    int num_threads = 0;

    // Avoid parallel overhead for tiny workspaces.
    int parallel_min_blocks = 16;
};

struct WorkspaceBuildOptions {
    // Parallelize make_workspace over mu-blocks with OpenMP.
    // Compile with -fopenmp. If OpenMP is unavailable, this falls back to serial.
    bool use_openmp = true;

    // Number of OpenMP threads. 0 means use omp_get_max_threads().
    int num_threads = 0;

    // Avoid OpenMP overhead for tiny numbers of mu-blocks.
    int parallel_min_mu_blocks = 16;

};

struct Result {
    std::vector<int> doubled_iotas;        // I = 2 iota
    std::vector<double> iotas;             // iota
    std::vector<Complex> coefficients;     // C_iota, same order as doubled_iotas
    std::array<Spinor, 4> rotated_spinors; // after rotating leg 1 to north
};

struct MuBlock {
    int mu = 0;              // mu = 2M
    int k2 = -1;             // k2 index for psi2, already validated

    std::vector<int> rows;   // indices into Workspace::doubled_iotas
    std::vector<int> k3s;    // k3 indices into psi3
    std::vector<int> k4s;    // k4 indices into psi4, paired with k3s

    // signed_w12[r] = (-1)^((I + mu)/2) * ( j1 j2 I ; j1, mu-J1, -mu )
    std::vector<double> signed_w12;

    // Row-major W34 matrix:
    //   W34[r * ncols + c]
    // = ( j3 j4 I ; nu3, -mu-nu3, mu )
    // where row r corresponds to rows[r], and col c corresponds to k3s[c].
    std::vector<double> W34;

    int nrows() const { return static_cast<int>(rows.size()); }
    int ncols() const { return static_cast<int>(k3s.size()); }
};

struct Workspace {
    std::array<int, 4> J{{0, 0, 0, 0}}; // doubled spins J1,J2,J3,J4

    std::vector<int> doubled_iotas;     // admissible I = 2 iota
    std::vector<double> iotas;          // same values as doubles
    std::vector<double> sqrt_I_plus_1;  // sqrt(2 iota + 1) = sqrt(I + 1)

    std::vector<MuBlock> blocks;

    int max_block_rows = 0;
    int max_block_cols = 0;
};

namespace detail {

inline int parity_mod2(int a) {
    int r = a % 2;
    return r < 0 ? r + 2 : r;
}

inline Complex pow_int(Complex z, int n) {
    if (n < 0) {
        throw std::invalid_argument("pow_int received a negative exponent");
    }

    Complex result{1.0, 0.0};
    while (n > 0) {
        if (n & 1) result *= z;
        z *= z;
        n >>= 1;
    }
    return result;
}

inline double norm2(const Spinor& z) {
    return b4utils::norm2(z.z0) + b4utils::norm2(z.z1);
}

inline Complex polar_complex(double r, double theta) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    double s, c;
    __builtin_sincos(theta, &s, &c);
    return Complex{r * c, r * s};
#else
    return Complex{r * std::cos(theta), r * std::sin(theta)};
#endif
}

inline Spinor normalize_spinor(const Spinor& z) {
    const double n2 = norm2(z);
    if (!(n2 > 0.0) || !std::isfinite(n2)) {
        throw std::invalid_argument("zero or non-finite spinor encountered");
    }
    const double inv_n = 1.0 / std::sqrt(n2);
    return Spinor{z.z0 * inv_n, z.z1 * inv_n};
}

struct KahanComplex {
    Complex sum{0.0, 0.0};
    Complex c{0.0, 0.0};

    inline void add(const Complex& x) {
        const Complex y = x - c;
        const Complex t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
};

inline void kahan_add(std::vector<Complex>& sums,
                      std::vector<Complex>& comps,
                      int idx,
                      const Complex& x) {
    const Complex y = x - comps[idx];
    const Complex t = sums[idx] + y;
    comps[idx] = (t - sums[idx]) - y;
    sums[idx] = t;
}

} // namespace detail

inline Spinor spinor_from_angles(double theta, double phi) {
    const double half_theta = 0.5 * theta;
    const double s = std::sin(half_theta);
    const double c = std::cos(half_theta);
    const double ss = std::sin(phi);
    const double cc = std::cos(phi);
    return Spinor{Complex{c, 0.0}, Complex{s * cc, s * ss}};
}

inline std::array<Spinor, 4>
spinors_from_angles(const Angles& angles) {
    return std::array<Spinor, 4>{{
        spinor_from_angles(angles[0][0], angles[0][1]),
        spinor_from_angles(angles[1][0], angles[1][1]),
        spinor_from_angles(angles[2][0], angles[2][1]),
        spinor_from_angles(angles[3][0], angles[3][1])
    }};
}

// Build the SU(2) matrix that sends z to the north spinor (1,0):
//
//     g = [ conj(a)  conj(b) ]
//         [   -b        a    ]
//
// for normalized z = (a,b).
//
// Instead of storing the full matrix, we store the normalized pivot spinor.
inline std::array<Spinor, 4>
rotate_leg1_to_north(const std::array<Spinor, 4>& zin,
                     bool normalize_inputs = true) {
    std::array<Spinor, 4> z = zin;

    if (normalize_inputs) {
        for (int a = 0; a < 4; ++a) {
            z[a] = detail::normalize_spinor(z[a]);
        }
    }

    const Spinor p = detail::normalize_spinor(z[0]);
    const Complex a = p.z0;
    const Complex b = p.z1;
    const Complex ca = b4utils::cconj(a);
    const Complex cb = b4utils::cconj(b);

    std::array<Spinor, 4> zout;
    for (int i = 0; i < 4; ++i) {
        zout[i].z0 = ca * z[i].z0 + cb * z[i].z1;
        zout[i].z1 = -b * z[i].z0 + a * z[i].z1;
    }

    // Avoid leaving tiny garbage on the pivot. This is mostly cosmetic, but useful
    // when debugging the future C++ swamp.
    zout[0].z0 = Complex{1.0, 0.0};
    zout[0].z1 = Complex{0.0, 0.0};

    return zout;
}

// psi[k] = sqrt(Binomial[J,k]) z0^k z1^(J-k), k=0..J.
// This corresponds to m = k - J/2.
std::vector<Complex>
coherent_vector_recurrence(int J,
                           Spinor z,
                           double zero_tol = 1.0e-14,
                           bool normalize_input = true);

inline std::vector<int>
admissible_doubled_iotas(int J1, int J2, int J3, int J4) {
    if (J1 < 0 || J2 < 0 || J3 < 0 || J4 < 0) {
        throw std::invalid_argument("negative doubled spin");
    }

    const int p12 = detail::parity_mod2(J1 + J2);
    const int p34 = detail::parity_mod2(J3 + J4);

    if (p12 != p34) {
        return {};
    }

    const int Imin_raw = std::max(std::abs(J1 - J2), std::abs(J3 - J4));
    const int Imax = std::min(J1 + J2, J3 + J4);

    if (Imin_raw > Imax) {
        return {};
    }

    int Imin = Imin_raw;
    if (detail::parity_mod2(Imin) != p12) {
        ++Imin;
    }

    std::vector<int> vals;
    for (int I = Imin; I <= Imax; I += 2) {
        vals.push_back(I);
    }
    return vals;
}



namespace detail {

bool build_mu_block(const Workspace& ws,
                           int J1, int J2, int J3, int J4,
                           int mu,
                           threejrec::Workspace& recurrence_workspace,
                           MuBlock& block_out);

inline Workspace initialize_workspace(int J1, int J2, int J3, int J4) {
    Workspace ws;
    ws.J = std::array<int, 4>{{J1, J2, J3, J4}};
    ws.doubled_iotas = admissible_doubled_iotas(J1, J2, J3, J4);
    ws.iotas.reserve(ws.doubled_iotas.size());
    ws.sqrt_I_plus_1.reserve(ws.doubled_iotas.size());

    for (int I : ws.doubled_iotas) {
        ws.iotas.push_back(0.5 * static_cast<double>(I));
        ws.sqrt_I_plus_1.push_back(std::sqrt(static_cast<double>(I + 1)));
    }
    return ws;
}

} // namespace detail

Workspace make_workspace_parallel(
    int J1, int J2, int J3, int J4,
    const WorkspaceBuildOptions& opt = WorkspaceBuildOptions{});

inline Workspace make_workspace_parallel(
    const std::array<int, 4>& J,
    const WorkspaceBuildOptions& opt = WorkspaceBuildOptions{}) {
    return make_workspace_parallel(J[0], J[1], J[2], J[3], opt);
}

inline Workspace make_workspace(int J1, int J2, int J3, int J4) {
    WorkspaceBuildOptions opt;
    opt.use_openmp = false;
    return make_workspace_parallel(J1, J2, J3, J4, opt);
}

inline Workspace make_workspace(const std::array<int, 4>& J) {
    return make_workspace(J[0], J[1], J[2], J[3]);
}
Result evaluate_spinors(const Workspace& ws,
                               const std::array<Spinor, 4>& input_spinors,
                               const EvalOptions& opt = EvalOptions{});

inline Result evaluate_angles(const Workspace& ws,
                              const Angles& angles,
                              const EvalOptions& opt = EvalOptions{}) {
    return evaluate_spinors(ws, spinors_from_angles(angles), opt);
}

inline Result
compute_coherent_coefficients_for_intertwiner(
    const std::array<int, 4>& spins,
    const Angles& angles,
    const coherent_intertwiner::WorkspaceBuildOptions& wopt,
    const coherent_intertwiner::EvalOptions& eopt)
{
    auto ws =
        coherent_intertwiner::make_workspace_parallel(
            spins,
            wopt
        );

    return coherent_intertwiner::evaluate_angles(
        ws,
        angles,
        eopt
    );
}

} // namespace coherent_intertwiner

#endif // COHERENT_INTERTWINER_HPP
