// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "amplitude/AmplitudePackedContractionGeneral.hpp"

namespace b4support {

Range1D amplitude_general_build_T2_row(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* T1_values,
    std::size_t ik4,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* row)
{
    const std::size_t K4d = support.channels.k4.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L25 = support.l25.size();

    std::fill(row, row + L25, 0.0);

    const std::size_t B4_block = idx3(ik4, il24, il34, L24, L34);
    const Range1D& r_B4 = support.B4.blocks[B4_block];
    if (r_B4.empty()) {
        return empty_range_1d();
    }

    Range1D cover = empty_range_1d();

    for (std::size_t il45 = r_B4.begin; il45 < r_B4.end; ++il45) {
        const std::size_t T1_block = idx3(il45, ik4, il35, K4d, L35);
        const Range1D& r_T1 = T1_support.blocks[T1_block];
        if (r_T1.empty()) {
            continue;
        }

        const double k4_val = amplitude_general_load_1d(
            support.B4, B.B4.data(), B4_block, il45
        );

        const double* t1 = T1_values + r_T1.offset;
        for (std::uint32_t il25 = r_T1.begin; il25 < r_T1.end; ++il25) {
            row[il25] += k4_val * t1[il25 - r_T1.begin];
        }

        amplitude_general_cover_range(cover, r_T1);
    }

    return cover;
}

Range1D amplitude_general_build_T4_row(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T3_support,
    const double* T3_values,
    std::size_t ik3,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* row)
{
    const std::size_t K3d = support.channels.k3.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L25 = support.l25.size();

    std::fill(row, row + L25, 0.0);

    const std::size_t B3_block = idx3(ik3, il34, il35, L34, L35);
    const Range1D& r_B3 = support.B3.blocks[B3_block];
    if (r_B3.empty()) {
        return empty_range_1d();
    }

    Range1D cover = empty_range_1d();

    for (std::size_t il23 = r_B3.begin; il23 < r_B3.end; ++il23) {
        const std::size_t T3_block = idx3(il23, ik3, il24, K3d, L24);
        const Range1D& r_T3 = T3_support.blocks[T3_block];
        if (r_T3.empty()) {
            continue;
        }

        const double k3_val = amplitude_general_load_1d(
            support.B3, B.B3.data(), B3_block, il23
        );

        const double* t3 = T3_values + r_T3.offset;
        for (std::uint32_t il25 = r_T3.begin; il25 < r_T3.end; ++il25) {
            row[il25] += k3_val * t3[il25 - r_T3.begin];
        }

        amplitude_general_cover_range(cover, r_T3);
    }

    return cover;
}

double amplitude_general_contract_q_block_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* T1_values,
    const RangeBlockSupport1D& T3_support,
    const double* T3_values,
    std::size_t ix,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    AmplitudeGeneralThreadWorkspace& ws)
{
    if (rC.empty()) {
        return 0.0;
    }

    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L25 = support.l25.size();

    const std::size_t C_block = idx4(ix, il24, il34, il35, L24, L34, L35);
    const std::size_t nk3 = static_cast<std::size_t>(rC.i_end - rC.i_begin);
    const std::size_t nk4 = static_cast<std::size_t>(rC.j_end - rC.j_begin);

    if (nk3 == 0 || nk4 == 0 || L25 == 0) {
        return 0.0;
    }

    // Project along the side that avoids building the larger number of rows.
    // Existing q-local kernels make a more refined decision; this keeps the
    // fallback fast without baking in reflection assumptions.
    const bool project_C_times_T2 = (nk3 <= nk4);
    const std::size_t projected_rows = project_C_times_T2 ? nk3 : nk4;
    ws.ensure(L25, projected_rows);

    double* row = ws.row.data();
    double* projected = ws.projected.data();
    std::fill(projected, projected + projected_rows * L25, 0.0);

    double total = 0.0;

    if (project_C_times_T2) {
        // P[k3,l25] = sum_k4 C[k3,k4] T2[k4,l25]
        for (std::size_t jb = 0; jb < nk4; ++jb) {
            const std::size_t ik4 = static_cast<std::size_t>(rC.j_begin) + jb;
            const Range1D rT2 = amplitude_general_build_T2_row(
                support, B, T1_support, T1_values,
                ik4, il24, il34, il35, row
            );

            if (rT2.empty()) {
                continue;
            }

            for (std::size_t ib = 0; ib < nk3; ++ib) {
                const std::size_t ik3 = static_cast<std::size_t>(rC.i_begin) + ib;
                const double c = packed_load_sjC_half(
                    support, sj.sjC.data(), C_block, ik3, ik4
                );

                double* prow = projected + ib * L25;
                for (std::uint32_t il25 = rT2.begin; il25 < rT2.end; ++il25) {
                    prow[il25] += c * row[il25];
                }
            }
        }

        for (std::size_t ib = 0; ib < nk3; ++ib) {
            const std::size_t ik3 = static_cast<std::size_t>(rC.i_begin) + ib;
            const Range1D rT4 = amplitude_general_build_T4_row(
                support, B, T3_support, T3_values,
                ik3, il24, il34, il35, row
            );

            if (rT4.empty()) {
                continue;
            }

            const double* prow = projected + ib * L25;
            for (std::uint32_t il25 = rT4.begin; il25 < rT4.end; ++il25) {
                total += prow[il25] * row[il25];
            }
        }
    } else {
        // P[k4,l25] = sum_k3 C[k3,k4] T4[k3,l25]
        for (std::size_t ib = 0; ib < nk3; ++ib) {
            const std::size_t ik3 = static_cast<std::size_t>(rC.i_begin) + ib;
            const Range1D rT4 = amplitude_general_build_T4_row(
                support, B, T3_support, T3_values,
                ik3, il24, il34, il35, row
            );

            if (rT4.empty()) {
                continue;
            }

            for (std::size_t jb = 0; jb < nk4; ++jb) {
                const std::size_t ik4 = static_cast<std::size_t>(rC.j_begin) + jb;
                const double c = packed_load_sjC_half(
                    support, sj.sjC.data(), C_block, ik3, ik4
                );

                double* prow = projected + jb * L25;
                for (std::uint32_t il25 = rT4.begin; il25 < rT4.end; ++il25) {
                    prow[il25] += c * row[il25];
                }
            }
        }

        for (std::size_t jb = 0; jb < nk4; ++jb) {
            const std::size_t ik4 = static_cast<std::size_t>(rC.j_begin) + jb;
            const Range1D rT2 = amplitude_general_build_T2_row(
                support, B, T1_support, T1_values,
                ik4, il24, il34, il35, row
            );

            if (rT2.empty()) {
                continue;
            }

            const double* prow = projected + jb * L25;
            for (std::uint32_t il25 = rT2.begin; il25 < rT2.end; ++il25) {
                total += prow[il25] * row[il25];
            }
        }
    }

    return total;
}

double amplitude_general_contract_slice_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* T1_values,
    const RangeBlockSupport1D& T3_support,
    const double* T3_values,
    std::size_t ix,
    PackedContractionOptions opt,
    PackedQLocalContractionStats* stats)
{
    (void)stats; // The generic half-aware kernel keeps wall times but not the old q-block histograms.

    const std::size_t K3d = support.channels.k3.size();
    const std::size_t K4d = support.channels.k4.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();

    assert(ix < support.channels.xvals.size());
    assert(T1_values != nullptr);
    assert(T3_values != nullptr);

    const std::size_t nq = L24 * L34 * L35;
    double slice_total = 0.0;

#ifdef _OPENMP
#pragma omp parallel reduction(+:slice_total) \
    if(opt.use_openmp && nq * K3d * K4d >= opt.parallel_threshold)
#endif
    {
        AmplitudeGeneralThreadWorkspace ws;

#ifdef _OPENMP
#pragma omp for collapse(2) schedule(dynamic, 1)
#endif
        for (std::ptrdiff_t il24_s = 0;
             il24_s < static_cast<std::ptrdiff_t>(L24);
             ++il24_s)
        {
            for (std::size_t il34 = 0; il34 < L34; ++il34) {
                const std::size_t il24 = static_cast<std::size_t>(il24_s);

                for (std::size_t il35 = 0; il35 < L35; ++il35) {
                    const std::size_t C_block = idx4(ix, il24, il34, il35,
                                                     L24, L34, L35);
                    const Range2D& rC = amplitude_half_sjC_logical_range(support, C_block);
                    if (rC.empty()) {
                        continue;
                    }

                    slice_total += amplitude_general_contract_q_block_half_sjC(
                        support,
                        sj,
                        B,
                        T1_support,
                        T1_values,
                        T3_support,
                        T3_values,
                        ix,
                        il24,
                        il34,
                        il35,
                        rC,
                        ws
                    );
                }
            }
        }
    }

    return slice_total;
}

double contract_packed_amplitude_general(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeBoosters& B,
    const std::array<int, 10>& spins,
    const std::array<int, 5>& fixed_intertwiners,
    WingExecutionOrder order,
    PackedContractionOptions opt,
    PackedQLocalContractionStats* stats)
{
    using clock = std::chrono::high_resolution_clock;

    if (amplitude_general_reflection_qorbit_compatible(support, spins, fixed_intertwiners)) {
        return contract_packed_amplitude_planAB_qlocal_reflection_qorbit_cost_selected_half_sjC(
            support,
            sj,
            B,
            spins,
            fixed_intertwiners,
            order,
            opt,
            stats,
            true
        );
    }

    PackedQLocalSupportWorkspace sws;
    sws.initialize(support);

    PackedQLocalValueWorkspace vws;

    const std::size_t X = support.channels.xvals.size();

    if (stats != nullptr) {
        stats->clear();
    }

    const auto total_t0 = clock::now();
    double amplitude = 0.0;

    for (std::size_t ix = 0; ix < X; ++ix) {
        if (order == WingExecutionOrder::LeftThenRight) {
            auto t0 = clock::now();
            build_T3_support_qlocal_slice_into(support, ix, sws.T3, opt);
            if (stats) stats->time_T3_support += qlocal_seconds_since(t0);

            vws.T3.ensure(std::max<std::size_t>(1, static_cast<std::size_t>(sws.T3.packed_values)));
            t0 = clock::now();
            fill_T3_values_packed_slice(support, sj, B, sws.T3, ix, vws.T3.ptr(), opt);
            if (stats) stats->time_T3_values += qlocal_seconds_since(t0);

            t0 = clock::now();
            build_T1_support_qlocal_slice_into(support, ix, sws.T1, opt);
            if (stats) stats->time_T1_support += qlocal_seconds_since(t0);

            vws.T1.ensure(std::max<std::size_t>(1, static_cast<std::size_t>(sws.T1.packed_values)));
            t0 = clock::now();
            fill_T1_values_packed_slice(support, sj, B, sws.T1, ix, vws.T1.ptr(), opt);
            if (stats) stats->time_T1_values += qlocal_seconds_since(t0);
        } else {
            auto t0 = clock::now();
            build_T1_support_qlocal_slice_into(support, ix, sws.T1, opt);
            if (stats) stats->time_T1_support += qlocal_seconds_since(t0);

            vws.T1.ensure(std::max<std::size_t>(1, static_cast<std::size_t>(sws.T1.packed_values)));
            t0 = clock::now();
            fill_T1_values_packed_slice(support, sj, B, sws.T1, ix, vws.T1.ptr(), opt);
            if (stats) stats->time_T1_values += qlocal_seconds_since(t0);

            t0 = clock::now();
            build_T3_support_qlocal_slice_into(support, ix, sws.T3, opt);
            if (stats) stats->time_T3_support += qlocal_seconds_since(t0);

            vws.T3.ensure(std::max<std::size_t>(1, static_cast<std::size_t>(sws.T3.packed_values)));
            t0 = clock::now();
            fill_T3_values_packed_slice(support, sj, B, sws.T3, ix, vws.T3.ptr(), opt);
            if (stats) stats->time_T3_values += qlocal_seconds_since(t0);
        }

        if (stats != nullptr) {
            stats->T1_values_total += sws.T1.packed_values;
            stats->T3_values_total += sws.T3.packed_values;
        }

        auto tq = clock::now();
        amplitude += amplitude_general_contract_slice_half_sjC(
            support,
            sj,
            B,
            sws.T1,
            vws.T1.ptr(),
            sws.T3,
            vws.T3.ptr(),
            ix,
            opt,
            stats
        );
        if (stats) stats->time_q_loop += qlocal_seconds_since(tq);
    }

    if (stats != nullptr) {
        stats->time_total = qlocal_seconds_since(total_t0);
    }

    return amplitude;
}

} // namespace b4support
