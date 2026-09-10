// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

// Cost-selected q-orbit backend for reflection-compatible single amplitudes.
// This header intentionally includes the trusted q-local/q-orbit backend and
// only adds a new representative selector. The scalar contribution symmetry is
// still used only after the full local q-block contraction, not at branch level.
#include "amplitude/AmplitudePackedContractionQLocal_ReflectionQOrbit.hpp"

namespace b4support {

// ============================================================================
// Cheap local q-block cost estimator
// ============================================================================

struct QLocalEstimatedQCost {
    std::uint64_t score = 0;
    std::uint64_t t2_rows = 0;
    std::uint64_t t4_rows = 0;
    std::uint64_t t2_row_work = 0;
    std::uint64_t t4_row_work = 0;
    std::uint64_t c_pairs = 0;
    std::uint64_t projected_work = 0;
    std::uint64_t final_work = 0;
    QLocalFinalStrategy strategy = QLocalFinalStrategy::ProjectCTimesT2;
    bool empty = true;
};

inline std::uint64_t qlocal_range_count_u64(const Range1D& r) noexcept
{
    return static_cast<std::uint64_t>(r.count());
}

inline std::uint64_t qlocal_intersection_count_u64(const Range1D& a,
                                                   const Range1D& b) noexcept
{
    if (a.empty() || b.empty()) {
        return 0ull;
    }

    const std::uint32_t begin = std::max(a.begin, b.begin);
    const std::uint32_t end   = std::min(a.end,   b.end);
    return end > begin ? static_cast<std::uint64_t>(end - begin) : 0ull;
}

inline Range1D estimate_T2_row_range_for_q_k4(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T1_support,
    std::size_t ik4,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    std::uint64_t& row_work)
{
    const std::size_t K4d = support.channels.k4.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();

    Range1D row_range = empty_range_1d();
    row_work = 0ull;

    const std::size_t B4_block = idx3(ik4, il24, il34, L24, L34);
    const Range1D& r_B4 = support.B4.blocks[B4_block];
    if (r_B4.empty()) {
        return row_range;
    }

    for (std::size_t il45 = r_B4.begin; il45 < r_B4.end; ++il45) {
        const std::size_t T1_block = idx3(il45, ik4, il35, K4d, L35);
        const Range1D& r_T1 = T1_support.blocks[T1_block];
        if (r_T1.empty()) {
            continue;
        }

        row_work += qlocal_range_count_u64(r_T1);
        qlocal_cover_range_with_range(row_range, r_T1);
    }

    return row_range;
}

inline Range1D estimate_T4_row_range_for_q_k3(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T3_support,
    std::size_t ik3,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    std::uint64_t& row_work)
{
    const std::size_t K3d = support.channels.k3.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();

    Range1D row_range = empty_range_1d();
    row_work = 0ull;

    const std::size_t B3_block = idx3(ik3, il34, il35, L34, L35);
    const Range1D& r_B3 = support.B3.blocks[B3_block];
    if (r_B3.empty()) {
        return row_range;
    }

    for (std::size_t il23 = r_B3.begin; il23 < r_B3.end; ++il23) {
        const std::size_t T3_block = idx3(il23, ik3, il24, K3d, L24);
        const Range1D& r_T3 = T3_support.blocks[T3_block];
        if (r_T3.empty()) {
            continue;
        }

        row_work += qlocal_range_count_u64(r_T3);
        qlocal_cover_range_with_range(row_range, r_T3);
    }

    return row_range;
}

QLocalEstimatedQCost estimate_qlocal_qblock_cost(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T1_support,
    const RangeBlockSupport1D& T3_support,
    std::size_t ix,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC);

inline bool qlocal_cost_select_reflected_partner(
    const AmplitudeInputSupportPlan& support,
    const RangeBlockSupport1D& T1_support,
    const RangeBlockSupport1D& T3_support,
    std::size_t ix,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    const Range2D& rC,
    const Range2D& rC_reflected)
{
    const QLocalEstimatedQCost c0 = estimate_qlocal_qblock_cost(
        support, T1_support, T3_support, ix, il24, il34, il35, rC
    );

    const QLocalEstimatedQCost c1 = estimate_qlocal_qblock_cost(
        support, T1_support, T3_support, ix, il35, il34, il24, rC_reflected
    );

    // Return true when the reflected member should be computed instead of the
    // current representative. Ties keep the deterministic l24<=l35 member.
    return c1.score < c0.score;
}

} // namespace b4support


