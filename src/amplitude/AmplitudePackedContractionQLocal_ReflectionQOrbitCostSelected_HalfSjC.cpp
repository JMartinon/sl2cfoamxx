// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "amplitude/AmplitudePackedContractionQLocal_ReflectionQOrbitCostSelected_HalfSjC.hpp"

namespace b4support {

double contract_qlocal_streamed_slice_reflection_qorbit_cost_selected_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeBoosters& B,
    const RangeBlockSupport1D& T1_support,
    const double* B4PACK_RESTRICT T1_values,
    const RangeBlockSupport1D& T3_support,
    const double* B4PACK_RESTRICT T3_values,
    std::size_t ix,
    PackedContractionOptions opt,
    PackedQLocalContractionStats* stats)
{
    const std::size_t K3d = support.channels.k3.size();
    const std::size_t K4d = support.channels.k4.size();

    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L25 = support.l25.size();

    assert(ix < support.channels.xvals.size());
    assert(T1_values != nullptr);
    assert(T3_values != nullptr);
    assert(L24 == L35);

    const std::size_t nq_rep = (L24 * (L24 + 1) / 2) * L34;

    double slice_total = 0.0;
    std::mutex stats_mutex;

#ifdef _OPENMP
#pragma omp parallel reduction(+:slice_total) \
    if(opt.use_openmp && nq_rep * K3d * K4d >= opt.parallel_threshold)
#endif
    {
        PackedQLocalThreadWorkspace ws;
        ws.ensure(K3d, K4d, L25);

        PackedQLocalThreadCounters local_counters;

#ifdef _OPENMP
#pragma omp for collapse(2) schedule(dynamic, 1)
#endif
        for (std::ptrdiff_t il24_s = 0;
             il24_s < static_cast<std::ptrdiff_t>(L24);
             ++il24_s)
        {
            for (std::size_t il34 = 0; il34 < L34; ++il34) {
                const std::size_t il24 = static_cast<std::size_t>(il24_s);
                // With half storage and reflection-compatible axes, il24 <= il35
                // is precisely the stored l-pair representative.  The orbit
                // weight accounts for the reflected q block.
                for (std::size_t il35 = il24; il35 < L35; ++il35) {

                    const bool diagonal = (il24 == il35);

                    const std::size_t C_block = idx4(ix, il24, il34, il35,
                                                     L24, L34, L35);

                    if (!amplitude_half_sjC_block_owns_storage(support, C_block)) {
                        // In a reflection-compatible run this should only be
                        // empty support.  If the axes are not actually symmetric,
                        // require_amplitude_reflection_qorbit_compatible() below
                        // should have rejected the call before we get here.
                        continue;
                    }

                    std::size_t use_il24 = il24;
                    std::size_t use_il35 = il35;
                    std::size_t use_C_block = C_block;
                    const Range2D* use_rC = &support.sjC.blocks[C_block];

                    // If both logical members physically own storage, preserve
                    // the old cost-selected behavior.  Under true half-sjC this
                    // is normally false for the reflected member, so we keep the
                    // stored canonical representative instead of materializing a
                    // temporary reflected C panel in the hot loop.
                    if (!diagonal) {
                        const std::size_t C_block_reflected = idx4(ix, il35, il34, il24,
                                                                    L24, L34, L35);
                        const bool reflected_owns_storage =
                            amplitude_half_sjC_block_owns_storage(support, C_block_reflected);

                        if (reflected_owns_storage) {
                            const bool use_reflected = qlocal_cost_select_reflected_partner(
                                support,
                                T1_support,
                                T3_support,
                                ix,
                                il24,
                                il34,
                                il35,
                                support.sjC.blocks[C_block],
                                support.sjC.blocks[C_block_reflected]
                            );

                            if (use_reflected) {
                                use_il24 = il35;
                                use_il35 = il24;
                                use_C_block = C_block_reflected;
                                use_rC = &support.sjC.blocks[C_block_reflected];
                            }
                        }
                    }

                    if (use_rC->empty()) {
                        continue;
                    }

                    (void)use_C_block;

                    const double orbit_weight = diagonal ? 1.0 : 2.0;
                    const double q_value = contract_q_block_streamed_adaptive(
                        support,
                        sj,
                        B,
                        T1_support,
                        T1_values,
                        T3_support,
                        T3_values,
                        ix,
                        use_il24,
                        il34,
                        use_il35,
                        *use_rC,
                        ws,
                        stats ? &local_counters : nullptr
                    );

                    slice_total += orbit_weight * q_value;
                }
            }
        }

        if (stats != nullptr) {
            {
                const std::lock_guard lock(stats_mutex);
                local_counters.add_to(*stats);
            }
        }
    }

    return slice_total;
}

double contract_packed_amplitude_planAB_qlocal_reflection_qorbit_cost_selected_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    const PackedAmplitudeSJTensors_HalfSjC& sj,
    const PackedAmplitudeBoosters& B,
    const std::array<int, 10>& spins,
    const std::array<int, 5>& fixed_intertwiners,
    WingExecutionOrder order,
    PackedContractionOptions opt,
    PackedQLocalContractionStats* stats,
    bool require_compatible)
{
    using clock = std::chrono::high_resolution_clock;

    if (require_compatible) {
        require_amplitude_reflection_qorbit_compatible(support, spins, fixed_intertwiners);
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
        amplitude += contract_qlocal_streamed_slice_reflection_qorbit_cost_selected_half_sjC(
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
