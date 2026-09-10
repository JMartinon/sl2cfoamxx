// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "amplitude/AmplitudePackedContractionQLocal_ReflectionQOrbit.hpp"

namespace b4support {

void build_T1_support_qlocal_slice_into(
    const AmplitudeInputSupportPlan& in,
    std::size_t ix,
    RangeBlockSupport1D& T1,
    PackedContractionOptions opt)
{
    const std::size_t K4d = in.channels.k4.size();

    const std::size_t L25 = in.l25.size();
    const std::size_t L35 = in.l35.size();
    const std::size_t L45 = in.l45.size();

    assert(ix < in.channels.xvals.size());
    assert(T1.blocks.size() == L45 * K4d * L35);

    const std::size_t nblocks = L45 * K4d * L35;

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static) \
    if(opt.use_openmp && nblocks >= opt.parallel_threshold)
#endif
    for (std::ptrdiff_t il45_s = 0;
         il45_s < static_cast<std::ptrdiff_t>(L45);
         ++il45_s)
    {
        for (std::size_t ik4 = 0; ik4 < K4d; ++ik4) {
            for (std::size_t il35 = 0; il35 < L35; ++il35) {
                const std::size_t il45 = static_cast<std::size_t>(il45_s);

                const std::size_t T1_block =
                    idx3(il45, ik4, il35, K4d, L35);

                const std::size_t sjB_block =
                    idx4(ix, ik4, il35, il45, K4d, L35, L45);

                const Range1D& r_sjB = in.sjB.blocks[sjB_block];

                if (r_sjB.empty()) {
                    T1.blocks[T1_block] = empty_range_1d();
                    continue;
                }

                CoverRangeBuilder cover;

                for (std::size_t il25 = 0; il25 < L25; ++il25) {
                    const std::size_t sjA_block = idx2(ix, il25, L25);
                    const std::size_t B5_block  = idx3(il25, il35, il45,
                                                       L35, L45);

                    const Range1D& r_sjA = in.sjA.blocks[sjA_block];
                    const Range1D& r_B5  = in.B5.blocks[B5_block];

                    if (ranges_intersect(r_sjA, r_sjB, r_B5)) {
                        cover.add(il25);
                    }
                }

                T1.blocks[T1_block] = cover.finish();
            }
        }
    }

    finalize_offsets(T1);
}

void build_T3_support_qlocal_slice_into(
    const AmplitudeInputSupportPlan& in,
    std::size_t ix,
    RangeBlockSupport1D& T3,
    PackedContractionOptions opt)
{
    const std::size_t K3d = in.channels.k3.size();

    const std::size_t L23 = in.l23.size();
    const std::size_t L24 = in.l24.size();
    const std::size_t L25 = in.l25.size();

    assert(ix < in.channels.xvals.size());
    assert(T3.blocks.size() == L23 * K3d * L24);

    const std::size_t nblocks = L23 * K3d * L24;

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static) \
    if(opt.use_openmp && nblocks >= opt.parallel_threshold)
#endif
    for (std::ptrdiff_t il23_s = 0;
         il23_s < static_cast<std::ptrdiff_t>(L23);
         ++il23_s)
    {
        for (std::size_t ik3 = 0; ik3 < K3d; ++ik3) {
            for (std::size_t il24 = 0; il24 < L24; ++il24) {
                const std::size_t il23 = static_cast<std::size_t>(il23_s);

                const std::size_t T3_block =
                    idx3(il23, ik3, il24, K3d, L24);

                const std::size_t sjD_block =
                    idx4(ix, ik3, il23, il24, K3d, L23, L24);

                const Range1D& r_sjD = in.sjD.blocks[sjD_block];

                if (r_sjD.empty()) {
                    T3.blocks[T3_block] = empty_range_1d();
                    continue;
                }

                CoverRangeBuilder cover;

                for (std::size_t il25 = 0; il25 < L25; ++il25) {
                    const std::size_t sjE_block = idx2(ix, il25, L25);
                    const std::size_t B2_block  = idx3(il23, il24, il25,
                                                       L24, L25);

                    const Range1D& r_sjE = in.sjE.blocks[sjE_block];
                    const Range1D& r_B2  = in.B2.blocks[B2_block];

                    if (ranges_intersect(r_sjE, r_sjD, r_B2)) {
                        cover.add(il25);
                    }
                }

                T3.blocks[T3_block] = cover.finish();
            }
        }
    }

    finalize_offsets(T3);
}

Range1D build_T2_row_for_q_k4(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T1_support,
    const double* B4PACK_RESTRICT T1_values,
    const PackedAmplitudeBoosters& B,
    std::size_t ik4,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* B4PACK_RESTRICT row,
    PackedQLocalThreadCounters* counters)
{
    const std::size_t K4d = support.channels.k4.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L25 = support.l25.size();

    qlocal_zero_row(row, L25);

    Range1D row_range = empty_range_1d();

    const std::size_t B4_block = idx3(ik4, il24, il34, L24, L34);
    const Range1D& r_B4 = support.B4.blocks[B4_block];

    if (r_B4.empty()) {
        return row_range;
    }

    const double* B4PACK_RESTRICT B4_base = B.B4.data() + r_B4.offset;

    for (std::size_t il45 = r_B4.begin; il45 < r_B4.end; ++il45) {
        const std::size_t T1_block = idx3(il45, ik4, il35, K4d, L35);
        const Range1D& r_T1 = T1_support.blocks[T1_block];

        if (r_T1.empty()) {
            continue;
        }

        const double coeff = B4_base[il45 - r_B4.begin];
        const double* B4PACK_RESTRICT src = T1_values + r_T1.offset;

        qlocal_axpy_adaptive(coeff,
                             src,
                             row + r_T1.begin,
                             r_T1.count(),
                             counters,
                             QLocalAxpyKind::RowBuild,
                             L25);
        qlocal_cover_range_with_range(row_range, r_T1);
    }

    return row_range;
}

Range1D build_T4_row_for_q_k3(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T3_support,
    const double* B4PACK_RESTRICT T3_values,
    const PackedAmplitudeBoosters& B,
    std::size_t ik3,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* B4PACK_RESTRICT row,
    PackedQLocalThreadCounters* counters)
{
    const std::size_t K3d = support.channels.k3.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L25 = support.l25.size();

    qlocal_zero_row(row, L25);

    Range1D row_range = empty_range_1d();

    const std::size_t B3_block = idx3(ik3, il34, il35, L34, L35);
    const Range1D& r_B3 = support.B3.blocks[B3_block];

    if (r_B3.empty()) {
        return row_range;
    }

    const double* B4PACK_RESTRICT B3_base = B.B3.data() + r_B3.offset;

    for (std::size_t il23 = r_B3.begin; il23 < r_B3.end; ++il23) {
        const std::size_t T3_block = idx3(il23, ik3, il24, K3d, L24);
        const Range1D& r_T3 = T3_support.blocks[T3_block];

        if (r_T3.empty()) {
            continue;
        }

        const double coeff = B3_base[il23 - r_B3.begin];
        const double* B4PACK_RESTRICT src = T3_values + r_T3.offset;

        qlocal_axpy_adaptive(coeff,
                             src,
                             row + r_T3.begin,
                             r_T3.count(),
                             counters,
                             QLocalAxpyKind::RowBuild,
                             L25);
        qlocal_cover_range_with_range(row_range, r_T3);
    }

    return row_range;
}

Range1D build_T2_panel_for_q(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T1_support,
    const double* B4PACK_RESTRICT T1_values,
    const PackedAmplitudeBoosters& B,
    std::size_t ik4_begin,
    std::size_t nk4,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* B4PACK_RESTRICT T2_panel,
    Range1D* B4PACK_RESTRICT T2_ranges,
    PackedQLocalThreadCounters* counters)
{
    const std::size_t K4d = support.channels.k4.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L25 = support.l25.size();

    Range1D panel_cover = empty_range_1d();

    for (std::size_t jb = 0; jb < nk4; ++jb) {
        const std::size_t ik4 = ik4_begin + jb;
        double* B4PACK_RESTRICT row = T2_panel + jb * L25;
        std::memset(row, 0, L25 * sizeof(double));

        Range1D row_range = empty_range_1d();

        const std::size_t B4_block = idx3(ik4, il24, il34, L24, L34);
        const Range1D& r_B4 = support.B4.blocks[B4_block];

        if (!r_B4.empty()) {
            const double* B4PACK_RESTRICT B4_base = B.B4.data() + r_B4.offset;

            for (std::size_t il45 = r_B4.begin; il45 < r_B4.end; ++il45) {
                const std::size_t T1_block = idx3(il45, ik4, il35, K4d, L35);
                const Range1D& r_T1 = T1_support.blocks[T1_block];

                if (r_T1.empty()) {
                    continue;
                }

                const double coeff = B4_base[il45 - r_B4.begin];
                const double* B4PACK_RESTRICT src = T1_values + r_T1.offset;

                qlocal_axpy_adaptive(coeff,
                                     src,
                                     row + r_T1.begin,
                                     r_T1.count(),
                                     counters,
                                     QLocalAxpyKind::RowBuild,
                                     L25);
                qlocal_cover_range_with_range(row_range, r_T1);
            }
        }

        T2_ranges[jb] = row_range;
        if (!row_range.empty()) {
            qlocal_cover_range_with_range(panel_cover, row_range);
            if (counters) {
                ++counters->T2_rows_built;
            }
        }
    }

    return panel_cover;
}

Range1D build_T4_panel_for_q(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T3_support,
    const double* B4PACK_RESTRICT T3_values,
    const PackedAmplitudeBoosters& B,
    std::size_t ik3_begin,
    std::size_t nk3,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* B4PACK_RESTRICT T4_panel,
    Range1D* B4PACK_RESTRICT T4_ranges,
    PackedQLocalThreadCounters* counters)
{
    const std::size_t K3d = support.channels.k3.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();

    Range1D panel_cover = empty_range_1d();

    for (std::size_t ib = 0; ib < nk3; ++ib) {
        const std::size_t ik3 = ik3_begin + ib;
        double* B4PACK_RESTRICT row = T4_panel + ib * L25;
        std::memset(row, 0, L25 * sizeof(double));

        Range1D row_range = empty_range_1d();

        const std::size_t B3_block = idx3(ik3, il34, il35, L34, L35);
        const Range1D& r_B3 = support.B3.blocks[B3_block];

        if (!r_B3.empty()) {
            const double* B4PACK_RESTRICT B3_base = B.B3.data() + r_B3.offset;

            for (std::size_t il23 = r_B3.begin; il23 < r_B3.end; ++il23) {
                const std::size_t T3_block = idx3(il23, ik3, il24, K3d, L24);
                const Range1D& r_T3 = T3_support.blocks[T3_block];

                if (r_T3.empty()) {
                    continue;
                }

                const double coeff = B3_base[il23 - r_B3.begin];
                const double* B4PACK_RESTRICT src = T3_values + r_T3.offset;

                qlocal_axpy_adaptive(coeff,
                                     src,
                                     row + r_T3.begin,
                                     r_T3.count(),
                                     counters,
                                     QLocalAxpyKind::RowBuild,
                                     L25);
                qlocal_cover_range_with_range(row_range, r_T3);
            }
        }

        T4_ranges[ib] = row_range;
        if (!row_range.empty()) {
            qlocal_cover_range_with_range(panel_cover, row_range);
            if (counters) {
                ++counters->T4_rows_built;
            }
        }
    }

    return panel_cover;
}

double contract_q_block_project_C_times_T2_streamed(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* B4PACK_RESTRICT T1_values,
    const RangeBlockSupport1D& T3_support,
    const double* B4PACK_RESTRICT T3_values,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    PackedQLocalThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters)
{
    const std::size_t L25 = support.l25.size();

    const std::size_t nk3 = static_cast<std::size_t>(rC.i_end - rC.i_begin);
    const std::size_t nk4 = static_cast<std::size_t>(rC.j_end - rC.j_begin);

    const double* B4PACK_RESTRICT C_base = sj.sjC.data() + rC.offset;

    double* B4PACK_RESTRICT T2_panel = ws.T2_panel.data(); // [local k4, l25]
    double* B4PACK_RESTRICT T4_panel = ws.T4_panel.data(); // [local k3, l25]
    double* B4PACK_RESTRICT W        = ws.work_panel.data(); // [local k3, l25]

    Range1D* B4PACK_RESTRICT T2_ranges = ws.T2_ranges.data();
    Range1D* B4PACK_RESTRICT T4_ranges = ws.T4_ranges.data();

    const Range1D T2_cover = build_T2_panel_for_q(
        support, T1_support, T1_values, B,
        rC.j_begin, nk4,
        il24, il34, il35,
        T2_panel,
        T2_ranges,
        counters
    );

    if (T2_cover.empty()) {
        return 0.0;
    }

    const Range1D T4_cover = build_T4_panel_for_q(
        support, T3_support, T3_values, B,
        rC.i_begin, nk3,
        il24, il34, il35,
        T4_panel,
        T4_ranges,
        counters
    );

    if (T4_cover.empty()) {
        return 0.0;
    }

    std::memset(W, 0, nk3 * L25 * sizeof(double));

    for (std::size_t jb = 0; jb < nk4; ++jb) {
        const Range1D& r_T2 = T2_ranges[jb];
        if (r_T2.empty()) {
            continue;
        }

        const double* B4PACK_RESTRICT T2_row = T2_panel + jb * L25;
        const double* B4PACK_RESTRICT C_col = C_base + nk3 * jb;

        for (std::size_t ib = 0; ib < nk3; ++ib) {
            const double c = C_col[ib];
            double* B4PACK_RESTRICT W_row = W + ib * L25;

            qlocal_axpy_adaptive(c,
                                 T2_row + r_T2.begin,
                                 W_row  + r_T2.begin,
                                 r_T2.count(),
                                 counters,
                                 QLocalAxpyKind::Projection,
                                 L25);
        }
    }

    const Range1D active_cover = intersect_2_ranges(T2_cover, T4_cover);
    if (active_cover.empty()) {
        return 0.0;
    }

    double local = 0.0;

    for (std::size_t ib = 0; ib < nk3; ++ib) {
        const Range1D& r_T4 = T4_ranges[ib];
        if (r_T4.empty()) {
            continue;
        }

        const Range1D r = intersect_2_ranges(active_cover, r_T4);
        if (r.empty()) {
            continue;
        }

        const double* B4PACK_RESTRICT T4_row = T4_panel + ib * L25;
        const double* B4PACK_RESTRICT W_row  = W        + ib * L25;

        local += qlocal_dot_adaptive(T4_row + r.begin,
                                     W_row  + r.begin,
                                     r.count(),
                                     counters,
                                     L25);
    }

    return local;
}

double contract_q_block_project_Ct_times_T4_streamed(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* B4PACK_RESTRICT T1_values,
    const RangeBlockSupport1D& T3_support,
    const double* B4PACK_RESTRICT T3_values,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    PackedQLocalThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters)
{
    const std::size_t L25 = support.l25.size();

    const std::size_t nk3 = static_cast<std::size_t>(rC.i_end - rC.i_begin);
    const std::size_t nk4 = static_cast<std::size_t>(rC.j_end - rC.j_begin);

    const double* B4PACK_RESTRICT C_base = sj.sjC.data() + rC.offset;

    double* B4PACK_RESTRICT T2_panel = ws.T2_panel.data(); // [local k4, l25]
    double* B4PACK_RESTRICT T4_panel = ws.T4_panel.data(); // [local k3, l25]
    double* B4PACK_RESTRICT V        = ws.work_panel.data(); // [local k4, l25]

    Range1D* B4PACK_RESTRICT T2_ranges = ws.T2_ranges.data();
    Range1D* B4PACK_RESTRICT T4_ranges = ws.T4_ranges.data();

    const Range1D T4_cover = build_T4_panel_for_q(
        support, T3_support, T3_values, B,
        rC.i_begin, nk3,
        il24, il34, il35,
        T4_panel,
        T4_ranges,
        counters
    );

    if (T4_cover.empty()) {
        return 0.0;
    }

    const Range1D T2_cover = build_T2_panel_for_q(
        support, T1_support, T1_values, B,
        rC.j_begin, nk4,
        il24, il34, il35,
        T2_panel,
        T2_ranges,
        counters
    );

    if (T2_cover.empty()) {
        return 0.0;
    }

    std::memset(V, 0, nk4 * L25 * sizeof(double));

    for (std::size_t ib = 0; ib < nk3; ++ib) {
        const Range1D& r_T4 = T4_ranges[ib];
        if (r_T4.empty()) {
            continue;
        }

        const double* B4PACK_RESTRICT T4_row = T4_panel + ib * L25;

        for (std::size_t jb = 0; jb < nk4; ++jb) {
            const double c = C_base[ib + nk3 * jb];
            double* B4PACK_RESTRICT V_row = V + jb * L25;

            qlocal_axpy_adaptive(c,
                                 T4_row + r_T4.begin,
                                 V_row  + r_T4.begin,
                                 r_T4.count(),
                                 counters,
                                 QLocalAxpyKind::Projection,
                                 L25);
        }
    }

    const Range1D active_cover = intersect_2_ranges(T4_cover, T2_cover);
    if (active_cover.empty()) {
        return 0.0;
    }

    double local = 0.0;

    for (std::size_t jb = 0; jb < nk4; ++jb) {
        const Range1D& r_T2 = T2_ranges[jb];
        if (r_T2.empty()) {
            continue;
        }

        const Range1D r = intersect_2_ranges(active_cover, r_T2);
        if (r.empty()) {
            continue;
        }

        const double* B4PACK_RESTRICT T2_row = T2_panel + jb * L25;
        const double* B4PACK_RESTRICT V_row  = V        + jb * L25;

        local += qlocal_dot_adaptive(T2_row + r.begin,
                                     V_row  + r.begin,
                                     r.count(),
                                     counters,
                                     L25);
    }

    return local;
}

double contract_q_block_pairwise_store_T4(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* B4PACK_RESTRICT T1_values,
    const RangeBlockSupport1D& T3_support,
    const double* B4PACK_RESTRICT T3_values,
    std::size_t ix,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    PackedQLocalThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters)
{
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L25 = support.l25.size();

    const std::size_t nk3 = static_cast<std::size_t>(rC.i_end - rC.i_begin);
    const std::size_t nk4 = static_cast<std::size_t>(rC.j_end - rC.j_begin);
    const double* B4PACK_RESTRICT C_base = sj.sjC.data() + rC.offset;
    (void)idx4(ix, il24, il34, il35, L24, L34, L35);

    double* B4PACK_RESTRICT T4_panel = ws.panel.data();
    double* B4PACK_RESTRICT T2_row = ws.row.data();

    for (std::size_t ib = 0; ib < nk3; ++ib) {
        const std::size_t ik3 = rC.i_begin + ib;
        double* B4PACK_RESTRICT row = T4_panel + ib * L25;
        ws.panel_ranges[ib] = build_T4_row_for_q_k3(
            support, T3_support, T3_values, B,
            ik3, il24, il34, il35,
            row,
            counters
        );
        if (!ws.panel_ranges[ib].empty() && counters) {
            ++counters->T4_rows_built;
        }
    }

    double local = 0.0;

    for (std::size_t jb = 0; jb < nk4; ++jb) {
        const std::size_t ik4 = rC.j_begin + jb;
        const Range1D r_T2 = build_T2_row_for_q_k4(
            support, T1_support, T1_values, B,
            ik4, il24, il34, il35,
            T2_row,
            counters
        );

        if (r_T2.empty()) {
            continue;
        }

        if (counters) {
            ++counters->T2_rows_built;
        }

        const double* B4PACK_RESTRICT C_col = C_base + nk3 * jb;

        for (std::size_t ib = 0; ib < nk3; ++ib) {
            const Range1D& r_T4 = ws.panel_ranges[ib];
            if (r_T4.empty()) {
                continue;
            }

            const Range1D r = intersect_2_ranges(r_T2, r_T4);
            if (r.empty()) {
                continue;
            }

            const double c = C_col[ib];

            const double* B4PACK_RESTRICT T4_row = T4_panel + ib * L25;
            local += c * qlocal_dot_adaptive(T2_row + r.begin,
                                             T4_row + r.begin,
                                             r.count(),
                                             counters,
                                             L25);
        }
    }

    return local;
}

double contract_q_block_pairwise_store_T2(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* B4PACK_RESTRICT T1_values,
    const RangeBlockSupport1D& T3_support,
    const double* B4PACK_RESTRICT T3_values,
    std::size_t ix,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    PackedQLocalThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters)
{
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L25 = support.l25.size();

    const std::size_t nk3 = static_cast<std::size_t>(rC.i_end - rC.i_begin);
    const std::size_t nk4 = static_cast<std::size_t>(rC.j_end - rC.j_begin);
    const double* B4PACK_RESTRICT C_base = sj.sjC.data() + rC.offset;
    (void)idx4(ix, il24, il34, il35, L24, L34, L35);

    double* B4PACK_RESTRICT T2_panel = ws.panel.data();
    double* B4PACK_RESTRICT T4_row = ws.row.data();

    for (std::size_t jb = 0; jb < nk4; ++jb) {
        const std::size_t ik4 = rC.j_begin + jb;
        double* B4PACK_RESTRICT row = T2_panel + jb * L25;
        ws.panel_ranges[jb] = build_T2_row_for_q_k4(
            support, T1_support, T1_values, B,
            ik4, il24, il34, il35,
            row,
            counters
        );
        if (!ws.panel_ranges[jb].empty() && counters) {
            ++counters->T2_rows_built;
        }
    }

    double local = 0.0;

    for (std::size_t ib = 0; ib < nk3; ++ib) {
        const std::size_t ik3 = rC.i_begin + ib;
        const Range1D r_T4 = build_T4_row_for_q_k3(
            support, T3_support, T3_values, B,
            ik3, il24, il34, il35,
            T4_row,
            counters
        );

        if (r_T4.empty()) {
            continue;
        }

        if (counters) {
            ++counters->T4_rows_built;
        }

        for (std::size_t jb = 0; jb < nk4; ++jb) {
            const Range1D& r_T2 = ws.panel_ranges[jb];
            if (r_T2.empty()) {
                continue;
            }

            const Range1D r = intersect_2_ranges(r_T2, r_T4);
            if (r.empty()) {
                continue;
            }

            const double c = C_base[ib + nk3 * jb];

            const double* B4PACK_RESTRICT T2_row = T2_panel + jb * L25;
            local += c * qlocal_dot_adaptive(T2_row + r.begin,
                                             T4_row + r.begin,
                                             r.count(),
                                             counters,
                                             L25);
        }
    }

    return local;
}

double contract_q_block_streamed_adaptive(
    const AmplitudeInputSupportPlan& support,
    const PackedAmplitudeSJTensors& sj,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* B4PACK_RESTRICT T1_values,
    const RangeBlockSupport1D& T3_support,
    const double* B4PACK_RESTRICT T3_values,
    std::size_t ix,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    PackedQLocalThreadWorkspace& ws,
    PackedQLocalThreadCounters* counters)
{
    const std::size_t L25 = support.l25.size();
    const std::size_t nk3 = static_cast<std::size_t>(rC.i_end - rC.i_begin);
    const std::size_t nk4 = static_cast<std::size_t>(rC.j_end - rC.j_begin);

    if (counters) {
        ++counters->active_q_blocks;
        counters->final_pairs += static_cast<std::uint64_t>(nk3) * static_cast<std::uint64_t>(nk4);
    }

    const QLocalFinalStrategy strategy = choose_qlocal_strategy(nk3, nk4, L25);

    switch (strategy) {
        case QLocalFinalStrategy::PairwiseStoreT4:
            if (counters) ++counters->pairwise_blocks;
            return contract_q_block_pairwise_store_T4(
                support, sj, B,
                T1_support, T1_values,
                T3_support, T3_values,
                ix, il24, il34, il35, rC,
                ws, counters
            );

        case QLocalFinalStrategy::PairwiseStoreT2:
            if (counters) ++counters->pairwise_blocks;
            return contract_q_block_pairwise_store_T2(
                support, sj, B,
                T1_support, T1_values,
                T3_support, T3_values,
                ix, il24, il34, il35, rC,
                ws, counters
            );

        case QLocalFinalStrategy::ProjectCtTimesT4:
            if (counters) ++counters->project_Ct_times_T4_blocks;
            return contract_q_block_project_Ct_times_T4_streamed(
                support, sj, B,
                T1_support, T1_values,
                T3_support, T3_values,
                il24, il34, il35, rC,
                ws, counters
            );

        case QLocalFinalStrategy::ProjectCTimesT2:
        default:
            if (counters) ++counters->project_C_times_T2_blocks;
            return contract_q_block_project_C_times_T2_streamed(
                support, sj, B,
                T1_support, T1_values,
                T3_support, T3_values,
                il24, il34, il35, rC,
                ws, counters
            );
    }
}

} // namespace b4support
