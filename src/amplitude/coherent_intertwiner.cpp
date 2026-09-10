// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "amplitude/coherent_intertwiner.hpp"

namespace coherent_intertwiner {

std::vector<Complex>
coherent_vector_recurrence(int J,
                           Spinor z,
                           double zero_tol,
                           bool normalize_input)
{
    if (J < 0) {
        throw std::invalid_argument("negative doubled spin J");
    }

    if (normalize_input) {
        z = detail::normalize_spinor(z);
    }

    std::vector<Complex> psi(static_cast<std::size_t>(J) + 1, Complex{0.0, 0.0});

    if (J == 0) {
        psi[0] = Complex{1.0, 0.0};
        return psi;
    }

    const double abs0 = b4utils::abs(z.z0);
    const double abs1 = b4utils::abs(z.z1);

    // South pole: only k=0 survives.
    if (abs0 <= zero_tol) {
        psi[0] = detail::pow_int(z.z1, J);
        return psi;
    }

    // North pole: only k=J survives.
    if (abs1 <= zero_tol) {
        psi[static_cast<std::size_t>(J)] = detail::pow_int(z.z0, J);
        return psi;
    }

    int k0 = static_cast<int>(std::llround(static_cast<double>(J) * b4utils::norm2(z.z0)));
    if (k0 < 0) k0 = 0;
    if (k0 > J) k0 = J;

    const double log_abs =
        0.5 * (std::lgamma(static_cast<double>(J) + 1.0)
             - std::lgamma(static_cast<double>(k0) + 1.0)
             - std::lgamma(static_cast<double>(J - k0) + 1.0))
        + static_cast<double>(k0) * std::log(abs0)
        + static_cast<double>(J - k0) * std::log(abs1);

    const double phase =
        static_cast<double>(k0) * b4utils::arg(z.z0)
      + static_cast<double>(J - k0) * b4utils::arg(z.z1);

    psi[static_cast<std::size_t>(k0)] = detail::polar_complex(std::exp(log_abs), phase);

    // Fill upward: psi[k+1] from psi[k].
    const Complex up_ratio_base = z.z0 / z.z1;
    for (int k = k0; k < J; ++k) {
        const double r = std::sqrt(static_cast<double>(J - k) /
                                   static_cast<double>(k + 1));
        psi[static_cast<std::size_t>(k + 1)] =
            psi[static_cast<std::size_t>(k)] * r * up_ratio_base;
    }

    // Fill downward: psi[k-1] from psi[k].
    const Complex down_ratio_base = z.z1 / z.z0;
    for (int k = k0; k >= 1; --k) {
        const double r = std::sqrt(static_cast<double>(k) /
                                   static_cast<double>(J - k + 1));
        psi[static_cast<std::size_t>(k - 1)] =
            psi[static_cast<std::size_t>(k)] * r * down_ratio_base;
    }

    return psi;
}


} // namespace coherent_intertwiner

namespace coherent_intertwiner::detail {

bool build_mu_block(const Workspace& ws,
                           int J1, int J2, int J3, int J4,
                           int mu,
                           threejrec::Workspace& recurrence_workspace,
                           MuBlock& block_out)
{
    const int k2_num = J2 + mu - J1;
    if ((k2_num & 1) != 0) {
        return false;
    }

    const int k2 = k2_num / 2;
    if (k2 < 0 || k2 > J2) {
        return false;
    }

    MuBlock block;
    block.mu = mu;
    block.k2 = k2;
    block.k3s.reserve(static_cast<std::size_t>(J3) + 1);
    block.k4s.reserve(static_cast<std::size_t>(J3) + 1);

    for (int k3 = 0; k3 <= J3; ++k3) {
        const int nu3 = 2 * k3 - J3;
        const int k4_num = J4 - mu - nu3;
        if ((k4_num & 1) != 0) {
            continue;
        }
        const int k4 = k4_num / 2;
        if (0 <= k4 && k4 <= J4) {
            block.k3s.push_back(k3);
            block.k4s.push_back(k4);
        }
    }

    if (block.k3s.empty()) {
        return false;
    }

    // Cyclic column permutation has positive phase:
    // (J1 J2 I; J1,mu-J1,-mu) = (I J1 J2; -mu,J1,mu-J1).
    int I12_min = 0;
    int I12_max = -2;
    std::vector<double> w12_series;
    if (!threejrec::fill_wigner3j_j1_series(
            J1, J2, J1, mu - J1,
            I12_min, I12_max, w12_series, recurrence_workspace)) {
        return false;
    }

    for (int row = 0; row < static_cast<int>(ws.doubled_iotas.size()); ++row) {
        const int I = ws.doubled_iotas[static_cast<std::size_t>(row)];
        if (std::abs(mu) > I || I < I12_min || I > I12_max ||
            ((I - I12_min) & 1) != 0) {
            continue;
        }

        const std::size_t i12 =
            static_cast<std::size_t>((I - I12_min) / 2);
        if (i12 >= w12_series.size()) {
            return false;
        }
        const double w12 = w12_series[i12];
        if (w12 == 0.0) {
            continue;
        }

        const int phase_int = (I + mu) / 2;
        const double sign = (phase_int & 1) ? -1.0 : 1.0;
        block.rows.push_back(row);
        block.signed_w12.push_back(sign * w12);
    }

    if (block.rows.empty()) {
        return false;
    }

    const int nr = block.nrows();
    const int nc = block.ncols();
    block.W34.assign(
        static_cast<std::size_t>(nr) * static_cast<std::size_t>(nc), 0.0);

    std::vector<double> row_values;
    for (int r = 0; r < nr; ++r) {
        const int I = ws.doubled_iotas[
            static_cast<std::size_t>(block.rows[static_cast<std::size_t>(r)])];
        int Dn_min = 0;
        int Dn_max = -2;
        if (!threejrec::fill_wigner3j_magnetic_row(
                J3, J4, I, -mu, Dn_min, Dn_max,
                row_values, recurrence_workspace)) {
            return false;
        }

        double* destination =
            block.W34.data() + static_cast<std::size_t>(r) *
                                   static_cast<std::size_t>(nc);
        if (static_cast<int>(row_values.size()) == nc &&
            Dn_min == 2 * block.k3s.front() - J3) {
            std::copy(row_values.begin(), row_values.end(), destination);
            continue;
        }

        // Keep the indexing explicit if a caller later supplies a sparse
        // magnetic range. The native recurrence still supplies every value;
        // no scalar evaluator or lookup-table fallback is needed.
        for (int c = 0; c < nc; ++c) {
            const int Dn = 2 * block.k3s[static_cast<std::size_t>(c)] - J3;
            if (Dn < Dn_min || Dn > Dn_max || ((Dn - Dn_min) & 1) != 0) {
                return false;
            }
            const std::size_t source =
                static_cast<std::size_t>((Dn - Dn_min) / 2);
            if (source >= row_values.size()) {
                return false;
            }
            destination[c] = row_values[source];
        }
    }

    block_out = std::move(block);
    return true;
}


} // namespace coherent_intertwiner::detail

namespace coherent_intertwiner {

Workspace make_workspace_parallel(
    int J1, int J2, int J3, int J4,
    const WorkspaceBuildOptions& opt)
{
    Workspace ws = detail::initialize_workspace(J1, J2, J3, J4);
    if (ws.doubled_iotas.empty()) {
        return ws;
    }

    const int Imax = ws.doubled_iotas.back();
    std::vector<int> mus;
    mus.reserve(static_cast<std::size_t>(Imax) + 1);
    for (int mu = -Imax; mu <= Imax; mu += 2) {
        mus.push_back(mu);
    }

    const std::size_t nmus = mus.size();
    std::vector<MuBlock> tmp_blocks(nmus);
    std::vector<unsigned char> keep(nmus, static_cast<unsigned char>(0));

#ifdef _OPENMP
    const bool do_parallel =
        opt.use_openmp &&
        nmus >= static_cast<std::size_t>(
                    std::max(1, opt.parallel_min_mu_blocks));
#else
    const bool do_parallel = false;
#endif

    if (!do_parallel) {
        threejrec::Workspace recurrence_workspace;
        for (std::size_t i = 0; i < nmus; ++i) {
            MuBlock block;
            if (detail::build_mu_block(
                    ws, J1, J2, J3, J4, mus[i],
                    recurrence_workspace, block)) {
                tmp_blocks[i] = std::move(block);
                keep[i] = static_cast<unsigned char>(1);
            }
        }
    }
#ifdef _OPENMP
    else {
        const int requested_threads =
            opt.num_threads > 0 ? opt.num_threads : omp_get_max_threads();
#pragma omp parallel num_threads(requested_threads)
        {
            threejrec::Workspace recurrence_workspace;
#pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < static_cast<int>(nmus); ++i) {
                MuBlock block;
                if (detail::build_mu_block(
                        ws, J1, J2, J3, J4,
                        mus[static_cast<std::size_t>(i)],
                        recurrence_workspace, block)) {
                    tmp_blocks[static_cast<std::size_t>(i)] = std::move(block);
                    keep[static_cast<std::size_t>(i)] =
                        static_cast<unsigned char>(1);
                }
            }
        }
    }
#endif

    ws.blocks.reserve(nmus);
    for (std::size_t i = 0; i < nmus; ++i) {
        if (!keep[i]) {
            continue;
        }
        ws.max_block_rows =
            std::max(ws.max_block_rows, tmp_blocks[i].nrows());
        ws.max_block_cols =
            std::max(ws.max_block_cols, tmp_blocks[i].ncols());
        ws.blocks.push_back(std::move(tmp_blocks[i]));
    }
    return ws;
}

Result evaluate_spinors(const Workspace& ws,
                               const std::array<Spinor, 4>& input_spinors,
                               const EvalOptions& opt)
{
    Result res;
    res.doubled_iotas = ws.doubled_iotas;
    res.iotas = ws.iotas;
    res.coefficients.assign(ws.doubled_iotas.size(), Complex{0.0, 0.0});

    if (ws.doubled_iotas.empty()) {
        res.rotated_spinors = input_spinors;
        return res;
    }

    const int J2 = ws.J[1];
    const int J3 = ws.J[2];
    const int J4 = ws.J[3];

    const std::array<Spinor, 4> zr =
        rotate_leg1_to_north(input_spinors, opt.normalize_spinors);

    res.rotated_spinors = zr;

    const std::vector<Complex> psi2 =
        coherent_vector_recurrence(J2, zr[1], opt.zero_tol, true);
    const std::vector<Complex> psi3 =
        coherent_vector_recurrence(J3, zr[2], opt.zero_tol, true);
    const std::vector<Complex> psi4 =
        coherent_vector_recurrence(J4, zr[3], opt.zero_tol, true);

    const std::size_t nI = ws.doubled_iotas.size();
    const std::size_t nb = ws.blocks.size();

#ifdef _OPENMP
    const bool do_parallel =
        opt.use_openmp &&
        nb >= static_cast<std::size_t>(std::max(1, opt.parallel_min_blocks));
#else
    const bool do_parallel = false;
#endif

    if (!do_parallel) {
        std::vector<Complex> u;
        std::vector<Complex> y;
        u.reserve(static_cast<std::size_t>(ws.max_block_cols));
        y.reserve(static_cast<std::size_t>(ws.max_block_rows));

        std::vector<Complex> coeff_comp;
        if (opt.compensated) {
            coeff_comp.assign(nI, Complex{0.0, 0.0});
        }

        for (const MuBlock& block : ws.blocks) {
            const int nr = block.nrows();
            const int nc = block.ncols();

            const Complex p2 = psi2[static_cast<std::size_t>(block.k2)];

            u.resize(static_cast<std::size_t>(nc));
            for (int c = 0; c < nc; ++c) {
                const int k3 = block.k3s[static_cast<std::size_t>(c)];
                const int k4 = block.k4s[static_cast<std::size_t>(c)];
                u[static_cast<std::size_t>(c)] =
                    psi3[static_cast<std::size_t>(k3)] * psi4[static_cast<std::size_t>(k4)];
            }

            y.assign(static_cast<std::size_t>(nr), Complex{0.0, 0.0});

            if (opt.compensated) {
                for (int r = 0; r < nr; ++r) {
                    detail::KahanComplex acc;
                    const double* rowptr = &block.W34[static_cast<std::size_t>(r) * static_cast<std::size_t>(nc)];

                    for (int c = 0; c < nc; ++c) {
                        acc.add(rowptr[c] * u[static_cast<std::size_t>(c)]);
                    }
                    y[static_cast<std::size_t>(r)] = acc.sum;
                }
            } else {
                for (int r = 0; r < nr; ++r) {
                    Complex acc{0.0, 0.0};
                    const double* rowptr = &block.W34[static_cast<std::size_t>(r) * static_cast<std::size_t>(nc)];

                    for (int c = 0; c < nc; ++c) {
                        acc += rowptr[c] * u[static_cast<std::size_t>(c)];
                    }
                    y[static_cast<std::size_t>(r)] = acc;
                }
            }

            for (int r = 0; r < nr; ++r) {
                const int row = block.rows[static_cast<std::size_t>(r)];
                const Complex contrib = block.signed_w12[static_cast<std::size_t>(r)]
                                      * p2
                                      * y[static_cast<std::size_t>(r)];

                if (opt.compensated) {
                    detail::kahan_add(res.coefficients, coeff_comp, row, contrib);
                } else {
                    res.coefficients[static_cast<std::size_t>(row)] += contrib;
                }
            }
        }
    }
#ifdef _OPENMP
    else {
        const int requested_threads =
            opt.num_threads > 0 ? opt.num_threads : omp_get_max_threads();

        std::vector<std::vector<Complex>> thread_coeffs(
            static_cast<std::size_t>(requested_threads),
            std::vector<Complex>(nI, Complex{0.0, 0.0}));

        std::vector<std::vector<Complex>> thread_comps;
        if (opt.compensated) {
            thread_comps.assign(
                static_cast<std::size_t>(requested_threads),
                std::vector<Complex>(nI, Complex{0.0, 0.0}));
        }

#pragma omp parallel num_threads(requested_threads)
        {
            const int tid = omp_get_thread_num();

            std::vector<Complex> u;
            std::vector<Complex> y;
            u.reserve(static_cast<std::size_t>(ws.max_block_cols));
            y.reserve(static_cast<std::size_t>(ws.max_block_rows));

            std::vector<Complex>& coeff = thread_coeffs[static_cast<std::size_t>(tid)];
            std::vector<Complex>* comp = nullptr;
            if (opt.compensated) {
                comp = &thread_comps[static_cast<std::size_t>(tid)];
            }

#pragma omp for schedule(dynamic, 1)
            for (int bi = 0; bi < static_cast<int>(nb); ++bi) {
                const MuBlock& block = ws.blocks[static_cast<std::size_t>(bi)];
                const int nr = block.nrows();
                const int nc = block.ncols();

                const Complex p2 = psi2[static_cast<std::size_t>(block.k2)];

                u.resize(static_cast<std::size_t>(nc));
                for (int c = 0; c < nc; ++c) {
                    const int k3 = block.k3s[static_cast<std::size_t>(c)];
                    const int k4 = block.k4s[static_cast<std::size_t>(c)];
                    u[static_cast<std::size_t>(c)] =
                        psi3[static_cast<std::size_t>(k3)] * psi4[static_cast<std::size_t>(k4)];
                }

                y.assign(static_cast<std::size_t>(nr), Complex{0.0, 0.0});

                if (opt.compensated) {
                    for (int r = 0; r < nr; ++r) {
                        detail::KahanComplex acc;
                        const double* rowptr = &block.W34[static_cast<std::size_t>(r) * static_cast<std::size_t>(nc)];

                        for (int c = 0; c < nc; ++c) {
                            acc.add(rowptr[c] * u[static_cast<std::size_t>(c)]);
                        }
                        y[static_cast<std::size_t>(r)] = acc.sum;
                    }
                } else {
                    for (int r = 0; r < nr; ++r) {
                        Complex acc{0.0, 0.0};
                        const double* rowptr = &block.W34[static_cast<std::size_t>(r) * static_cast<std::size_t>(nc)];

                        for (int c = 0; c < nc; ++c) {
                            acc += rowptr[c] * u[static_cast<std::size_t>(c)];
                        }
                        y[static_cast<std::size_t>(r)] = acc;
                    }
                }

                for (int r = 0; r < nr; ++r) {
                    const int row = block.rows[static_cast<std::size_t>(r)];
                    const Complex contrib = block.signed_w12[static_cast<std::size_t>(r)]
                                          * p2
                                          * y[static_cast<std::size_t>(r)];

                    if (opt.compensated) {
                        detail::kahan_add(coeff, *comp, row, contrib);
                    } else {
                        coeff[static_cast<std::size_t>(row)] += contrib;
                    }
                }
            }
        }

        if (opt.compensated) {
            std::vector<Complex> reduce_comp(nI, Complex{0.0, 0.0});
            for (int t = 0; t < requested_threads; ++t) {
                for (std::size_t r = 0; r < nI; ++r) {
                    detail::kahan_add(
                        res.coefficients,
                        reduce_comp,
                        static_cast<int>(r),
                        thread_coeffs[static_cast<std::size_t>(t)][r]);
                }
            }
        } else {
            for (int t = 0; t < requested_threads; ++t) {
                for (std::size_t r = 0; r < nI; ++r) {
                    res.coefficients[r] += thread_coeffs[static_cast<std::size_t>(t)][r];
                }
            }
        }
    }
#endif

    for (std::size_t r = 0; r < res.coefficients.size(); ++r) {
        res.coefficients[r] *= ws.sqrt_I_plus_1[r];
    }

    return res;
}

} // namespace coherent_intertwiner
