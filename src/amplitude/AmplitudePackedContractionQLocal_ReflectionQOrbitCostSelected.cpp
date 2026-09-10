// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "amplitude/AmplitudePackedContractionQLocal_ReflectionQOrbitCostSelected.hpp"

namespace b4support {

QLocalEstimatedQCost estimate_qlocal_qblock_cost(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T1_support,
    const RangeBlockSupport1D& T3_support,
    std::size_t ix,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC)
{
    (void)ix;

    QLocalEstimatedQCost out;
    if (rC.empty()) {
        out.score = std::numeric_limits<std::uint64_t>::max();
        return out;
    }

    const std::size_t L25 = support.l25.size();
    const std::size_t nk3 = static_cast<std::size_t>(rC.i_end - rC.i_begin);
    const std::size_t nk4 = static_cast<std::size_t>(rC.j_end - rC.j_begin);

    out.strategy = choose_qlocal_strategy(nk3, nk4, L25);
    out.c_pairs = static_cast<std::uint64_t>(nk3) * static_cast<std::uint64_t>(nk4);

    // Estimate all T2 rows and their combined l25 cover.
    Range1D T2_cover = empty_range_1d();
    std::uint64_t T2_range_sum = 0ull;

    for (std::size_t jb = 0; jb < nk4; ++jb) {
        const std::size_t ik4 = rC.j_begin + jb;
        std::uint64_t row_work = 0ull;
        const Range1D r_T2 = estimate_T2_row_range_for_q_k4(
            support, T1_support, ik4, il24, il34, il35, row_work
        );

        if (r_T2.empty()) {
            continue;
        }

        ++out.t2_rows;
        out.t2_row_work += row_work;
        T2_range_sum += qlocal_range_count_u64(r_T2);
        qlocal_cover_range_with_range(T2_cover, r_T2);
    }

    // Estimate all T4 rows and their combined l25 cover.
    Range1D T4_cover = empty_range_1d();
    std::uint64_t T4_range_sum = 0ull;

    for (std::size_t ib = 0; ib < nk3; ++ib) {
        const std::size_t ik3 = rC.i_begin + ib;
        std::uint64_t row_work = 0ull;
        const Range1D r_T4 = estimate_T4_row_range_for_q_k3(
            support, T3_support, ik3, il24, il34, il35, row_work
        );

        if (r_T4.empty()) {
            continue;
        }

        ++out.t4_rows;
        out.t4_row_work += row_work;
        T4_range_sum += qlocal_range_count_u64(r_T4);
        qlocal_cover_range_with_range(T4_cover, r_T4);
    }

    if (out.t2_rows == 0 || out.t4_rows == 0 ||
        T2_cover.empty() || T4_cover.empty())
    {
        out.score = std::numeric_limits<std::uint64_t>::max() / 4ull;
        return out;
    }

    out.empty = false;

    // This is intentionally a cheap estimator. It does not inspect C values and
    // does not reconstruct exact row intersections for every pair. Its job is
    // only to choose the cheaper member of a reflected pair, not to print a PhD
    // thesis in the inner loop.
    const std::uint64_t T2_avg = T2_range_sum / std::max<std::uint64_t>(1ull, out.t2_rows);
    const std::uint64_t T4_avg = T4_range_sum / std::max<std::uint64_t>(1ull, out.t4_rows);
    const std::uint64_t cover_intersection = qlocal_intersection_count_u64(T2_cover, T4_cover);

    switch (out.strategy) {
        case QLocalFinalStrategy::ProjectCTimesT2:
            // Build T2 rows, project through C onto k3 rows, then build T4 rows
            // and take final dots against the projected panel.
            out.projected_work = out.c_pairs * std::max<std::uint64_t>(1ull, T2_avg);
            out.final_work = out.t4_rows * std::max<std::uint64_t>(1ull, cover_intersection);
            break;

        case QLocalFinalStrategy::ProjectCtTimesT4:
            out.projected_work = out.c_pairs * std::max<std::uint64_t>(1ull, T4_avg);
            out.final_work = out.t2_rows * std::max<std::uint64_t>(1ull, cover_intersection);
            break;

        case QLocalFinalStrategy::PairwiseStoreT4:
        case QLocalFinalStrategy::PairwiseStoreT2:
        default:
            out.projected_work = 0ull;
            out.final_work = out.c_pairs * std::max<std::uint64_t>(1ull, cover_intersection);
            break;
    }

    // Row-zeroing and row-building matter at this scale, so include a small
    // constant model for them. Exact constants are not sacred; the ordering is.
    const std::uint64_t row_zero_work = (out.t2_rows + out.t4_rows)
        * static_cast<std::uint64_t>(L25);

    out.score = row_zero_work
              + out.t2_row_work
              + out.t4_row_work
              + out.projected_work
              + out.final_work;

    return out;
}

} // namespace b4support
