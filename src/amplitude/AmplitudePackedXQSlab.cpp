// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "amplitude/AmplitudePackedXQSlab.hpp"

namespace b4support {

std::vector<PackedAmplitudeQSlab> choose_q_slabs_for_xslice_half_sjC(
    const AmplitudeInputSupportPlan_HalfSjC& xslice_full_support,
    const AmplitudeInputSupportPlan_HalfSjC& wing_support,
    const RangeBlockSupport1D& T1,
    const RangeBlockSupport1D& T3,
    const PackedAmplitudeXQSlabOptions& opt,
    std::uint64_t fixed_bytes)
{
    const std::size_t Q = amplitude_q_count_half_sjC(xslice_full_support);
    const std::uint64_t budget = xqslab_memory_budget(opt);

    const std::uint64_t wing_bytes = xqslab_wing_payload_bytes(wing_support);
    const std::uint64_t T_bytes = xqslab_T1T3_payload_bytes(T1, T3);
    const std::uint64_t q_metadata = xslice_full_support.sjC.metadata_bytes()
        + static_cast<std::uint64_t>(xslice_full_support.sjC_map.size() * sizeof(AmplitudeSjCHalfBlockMap));

    std::uint64_t base_variable = 0;
    base_variable = xqslab_sat_add_u64(base_variable, wing_bytes);
    base_variable = xqslab_sat_add_u64(base_variable, T_bytes);
    base_variable = xqslab_sat_add_u64(base_variable, q_metadata);

    std::vector<PackedAmplitudeQSlab> slabs;
    slabs.reserve(Q == 0 ? 0 : std::min<std::size_t>(Q, 1024));

    std::vector<int> seen_epoch(Q, 0);
    int epoch = 1;

    std::size_t q = 0;
    while (q < Q) {
        if (epoch == std::numeric_limits<int>::max()) {
            std::fill(seen_epoch.begin(), seen_epoch.end(), 0);
            epoch = 1;
        }

        std::size_t q_end = q;
        std::uint64_t c_values = 0;
        bool added_any = false;

        while (q_end < Q && (q_end - q) < opt.max_q_per_slab) {
            std::size_t il24 = 0, il34 = 0, il35 = 0;
            amplitude_decode_q_half_sjC(xslice_full_support, q_end, il24, il34, il35);
            const std::size_t logical_block = idx4(0u, il24, il34, il35,
                                                   xslice_full_support.l24.size(),
                                                   xslice_full_support.l34.size(),
                                                   xslice_full_support.l35.size());
            const AmplitudeSjCHalfBlockMap& map = xslice_full_support.sjC_map[logical_block];

            std::uint64_t add_values = 0;
            if (!map.empty()) {
                const std::size_t storage = static_cast<std::size_t>(map.storage_block);
                if (storage < Q && seen_epoch[storage] != epoch) {
                    add_values = xslice_full_support.sjC.blocks[storage].count();
                }
            }

            std::uint64_t trial_values = xqslab_sat_add_u64(c_values, add_values);
            std::uint64_t trial_var = xqslab_sat_add_u64(base_variable, xqslab_bytes_for_values(trial_values));
            trial_var = xqslab_add_fractional_overhead(trial_var, opt.variable_overhead_fraction);
            const std::uint64_t trial_peak = xqslab_sat_add_u64(fixed_bytes, trial_var);

            if (added_any && trial_peak > budget) {
                break;
            }

            if (!map.empty()) {
                const std::size_t storage = static_cast<std::size_t>(map.storage_block);
                if (storage < Q && seen_epoch[storage] != epoch) {
                    seen_epoch[storage] = epoch;
                    c_values = trial_values;
                }
            }

            ++q_end;
            added_any = true;

            if (trial_peak > budget) {
                break;
            }
        }

        if (!added_any) {
            q_end = q + 1;
        }

        std::uint64_t var = xqslab_sat_add_u64(base_variable, xqslab_bytes_for_values(c_values));
        var = xqslab_add_fractional_overhead(var, opt.variable_overhead_fraction);
        const std::uint64_t peak = xqslab_sat_add_u64(fixed_bytes, var);

        PackedAmplitudeQSlab slab;
        slab.q_begin = q;
        slab.q_end = q_end;
        slab.sjC_payload_bytes = xqslab_bytes_for_values(c_values);
        slab.estimated_variable_bytes = var;
        slab.estimated_peak_bytes = peak;
        slab.fits_budget = peak <= budget;
        slabs.push_back(slab);

        q = q_end;
        ++epoch;
    }

    return slabs;
}

void fill_packed_amplitude_sjC_only_half_sjC(
    const B4Spins& s,
    const AmplitudeInputSupportPlan_HalfSjC& support,
    PackedAmplitudeSJTensors_HalfSjC& out)
{
    const AmplitudeChannels& AmC = support.channels;
    const std::vector<int>& xvals = AmC.xvals;
    const std::vector<int>& k3 = AmC.k3;
    const std::vector<int>& k4 = AmC.k4;
    const std::vector<int>& l24 = support.l24;
    const std::vector<int>& l34 = support.l34;
    const std::vector<int>& l35 = support.l35;

    out.sjC.resize(static_cast<std::size_t>(support.sjC.packed_values));
    if (support.sjC.packed_values == 0 || xvals.empty()) {
        return;
    }

    const std::vector<SjCStoredBlockTask> sjC_tasks = make_sjC_stored_block_tasks(support);
    if (sjC_tasks.empty()) {
        return;
    }

    const int Dj34 = s.Dj34;

    const std::size_t nthreads = packed_sj_fill_num_threads();
    const std::size_t chunk = choose_dynamic_chunk(sjC_tasks.size(), nthreads, 16, 256, 64);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<double> rec_values;
        rec_values.reserve(std::max<std::size_t>(k3.size(), xvals.size()) + 8u);

#ifdef _OPENMP
#pragma omp for schedule(dynamic, chunk)
#endif
        for (std::size_t itask = 0; itask < sjC_tasks.size(); ++itask) {
            const SjCStoredBlockTask& task = sjC_tasks[itask];

            const std::size_t ix   = task.ix;
            const std::size_t il24 = task.il24;
            const std::size_t il34 = task.il34;
            const std::size_t il35 = task.il35;

            const int Dx   = xvals[ix];
            const int Dl24 = l24[il24];
            const int Dl34 = l34[il34];
            const int Dl35 = l35[il35];
            const int PrefC = pm1_from_even_Dsum(Dl34 + Dj34);

            const std::size_t nk3 = static_cast<std::size_t>(task.i_end - task.i_begin);

            for (std::uint32_t ik4_u = task.j_begin; ik4_u < task.j_end; ++ik4_u) {
                const std::size_t ik4 = static_cast<std::size_t>(ik4_u);
                const int Dk4 = k4[ik4];

                if (!triangle_condition(Dk4, Dl34, Dl24) ||
                    !triangle_condition(Dk4, Dx,   Dl35))
                {
                    continue;
                }

                const int k3_parity = (Dl34 + Dl35) & 1;
                if (((Dx + Dl24) & 1) != k3_parity) {
                    continue;
                }

                const int Dk3_min = std::max(iabs_int(Dl34 - Dl35),
                                             iabs_int(Dx   - Dl24));
                const int Dk3_max = std::min(Dl34 + Dl35,
                                             Dx   + Dl24);
                if (Dk3_min > Dk3_max) {
                    continue;
                }

                const Range1D rk3 =
                    axis_range_from_interval_regular2(k3, Dk3_min, Dk3_max, k3_parity);
                if (rk3.empty()) {
                    continue;
                }

                const std::size_t ik3_begin =
                    std::max<std::size_t>(static_cast<std::size_t>(rk3.begin),
                                          static_cast<std::size_t>(task.i_begin));
                const std::size_t ik3_end =
                    std::min<std::size_t>(static_cast<std::size_t>(rk3.end),
                                          static_cast<std::size_t>(task.i_end));
                if (ik3_begin >= ik3_end) {
                    continue;
                }

                const bool ok = sjrec_detail::compute_wigner6j_x_series(
                    Dk3_min, Dk3_max, Dl34, Dl35, Dk4, Dx, Dl24, rec_values);
                if (!ok) {
                    continue;
                }

                double* const col = out.sjC.data()
                    + static_cast<std::size_t>(task.offset)
                    + static_cast<std::size_t>(ik4_u - task.j_begin) * nk3;

                const std::size_t local0 = ik3_begin - static_cast<std::size_t>(task.i_begin);
                const std::size_t n_store = ik3_end - ik3_begin;
                const std::size_t rec_off0 = sjrec_detail::x_offset_from_Dx(k3[ik3_begin], Dk3_min);

#pragma omp simd
                for (std::size_t t = 0; t < n_store; ++t) {
                    col[local0 + t] = static_cast<double>(PrefC) * rec_values[rec_off0 + t];
                }
            }
        }
    }
}

PackedAmplitudeXQResult contract_packed_amplitude_xqslab_streamed_generated_support(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const PackedAmplitudeBoosters& packed_B,
    const std::array<int, 10>& spins,
    const PackedAmplitudeXQSlabOptions& opt,
    WingExecutionOrder order,
    PackedContractionOptions contraction_options,
    std::ostream* log)
{
    using clock = std::chrono::high_resolution_clock;
    const auto total_t0 = clock::now();

    PackedAmplitudeXQResult result;
    result.stats.clear();
    result.x_slices = static_support.channels.xvals.size();

    const B4Spins b4spins(spins);
    const std::uint64_t fixed_bytes = xqslab_fixed_bytes(static_support, packed_B, opt);

    double amplitude = 0.0;

    for (std::size_t ix = 0; ix < static_support.channels.xvals.size(); ++ix) {
        if (log != nullptr && opt.print_x_timing) {
            *log << "\n[xqslab] x slice " << ix
                 << " (Dx=" << static_support.channels.xvals[ix] << ")\n";
        }

        double x_amp = 0.0;
        std::size_t x_q_slabs = 0;
        std::size_t x_unfit = 0;

        auto release_t0 = clock::now();
        {
            auto t0 = clock::now();
            AmplitudeInputSupportPlan_HalfSjC xslice_full_support =
                build_amplitude_xslice_full_support_half_sjC(static_support, b4spins, ix);
            AmplitudeInputSupportPlan_HalfSjC wing_support = xslice_full_support;
            amplitude_half_sjC_clear_physical_storage(wing_support);
            result.timings.xsupport += xqslab_seconds_since(t0);

            t0 = clock::now();
            PackedAmplitudeSJTensors_HalfSjC sj;
            fill_packed_amplitude_sj_tensors_rec_half_sjC(b4spins, wing_support, sj);
            result.timings.wing_fill += xqslab_seconds_since(t0);

            PackedQLocalSupportWorkspace sws;
            sws.initialize(wing_support);
            PackedQLocalValueWorkspace vws;

            t0 = clock::now();
            if (order == WingExecutionOrder::LeftThenRight) {
                build_T3_support_qlocal_slice_into(wing_support, 0u, sws.T3, contraction_options);
                vws.T3.ensure(std::max<std::size_t>(1u, static_cast<std::size_t>(sws.T3.packed_values)));
                fill_T3_values_packed_slice(wing_support, sj, packed_B, sws.T3, 0u, vws.T3.ptr(), contraction_options);

                build_T1_support_qlocal_slice_into(wing_support, 0u, sws.T1, contraction_options);
                vws.T1.ensure(std::max<std::size_t>(1u, static_cast<std::size_t>(sws.T1.packed_values)));
                fill_T1_values_packed_slice(wing_support, sj, packed_B, sws.T1, 0u, vws.T1.ptr(), contraction_options);
            } else {
                build_T1_support_qlocal_slice_into(wing_support, 0u, sws.T1, contraction_options);
                vws.T1.ensure(std::max<std::size_t>(1u, static_cast<std::size_t>(sws.T1.packed_values)));
                fill_T1_values_packed_slice(wing_support, sj, packed_B, sws.T1, 0u, vws.T1.ptr(), contraction_options);

                build_T3_support_qlocal_slice_into(wing_support, 0u, sws.T3, contraction_options);
                vws.T3.ensure(std::max<std::size_t>(1u, static_cast<std::size_t>(sws.T3.packed_values)));
                fill_T3_values_packed_slice(wing_support, sj, packed_B, sws.T3, 0u, vws.T3.ptr(), contraction_options);
            }
            result.timings.T1T3 += xqslab_seconds_since(t0);
            result.stats.T1_values_total += sws.T1.packed_values;
            result.stats.T3_values_total += sws.T3.packed_values;

            t0 = clock::now();
            const std::vector<PackedAmplitudeQSlab> qslabs =
                choose_q_slabs_for_xslice_half_sjC(xslice_full_support,
                                                   wing_support,
                                                   sws.T1,
                                                   sws.T3,
                                                   opt,
                                                   fixed_bytes);
            result.timings.planning += xqslab_seconds_since(t0);

            if (log != nullptr && opt.print_plan) {
                *log << "[xqslab] q slabs for x=" << ix
                     << ": " << qslabs.size() << "\n";
            }

            for (std::size_t iq = 0; iq < qslabs.size(); ++iq) {
                const PackedAmplitudeQSlab& qslab = qslabs[iq];
                ++x_q_slabs;
                ++result.q_slabs_total;
                result.max_estimated_peak_bytes = std::max(result.max_estimated_peak_bytes,
                                                           qslab.estimated_peak_bytes);
                if (!qslab.fits_budget) {
                    ++x_unfit;
                    ++result.unfitting_q_slabs;
                    if (opt.require_all_q_slabs_fit) {
                        throw b4utils::ResourceLimitFailure(
                            "x+q-slab fallback: one q-slab still exceeds the memory budget. "
                            "Next fallback would be k3/k4 tiling inside q; not implemented in this header."
                        );
                    }
                }

                if (log != nullptr && opt.print_q_timing) {
                    *log << "  q slab " << iq
                         << " q=[" << qslab.q_begin << "," << qslab.q_end << ") peak=";
                    xqslab_print_bytes(*log, qslab.estimated_peak_bytes);
                    *log << (qslab.fits_budget ? "\n" : "  NO\n");
                }

                t0 = clock::now();
                AmplitudeInputSupportPlan_HalfSjC q_support =
                    make_amplitude_xslice_qslab_support_half_sjC(xslice_full_support,
                                                                 qslab.q_begin,
                                                                 qslab.q_end);
                result.timings.qsupport += xqslab_seconds_since(t0);

                t0 = clock::now();
                fill_packed_amplitude_sjC_only_half_sjC(b4spins, q_support, sj);
                result.timings.sjC_fill += xqslab_seconds_since(t0);

                t0 = clock::now();
                x_amp += amplitude_general_contract_slice_half_sjC(
                    q_support,
                    sj,
                    packed_B,
                    sws.T1,
                    vws.T1.ptr(),
                    sws.T3,
                    vws.T3.ptr(),
                    0u,
                    contraction_options,
                    opt.print_contraction_stats ? &result.stats : nullptr
                );
                result.timings.qcontract += xqslab_seconds_since(t0);
            }

            release_t0 = clock::now();
        }
        result.timings.release += xqslab_seconds_since(release_t0);

        amplitude += x_amp;

        if (log != nullptr && opt.print_x_timing) {
            *log << "[xqslab] x=" << ix
                 << " q_slabs=" << x_q_slabs
                 << " unfitting=" << x_unfit
                 << " x_amp=" << std::scientific << std::setprecision(17) << x_amp
                 << " cumulative=" << amplitude
                 << std::fixed << std::setprecision(6) << "\n";
        }
    }

    result.amplitude = amplitude;
    result.timings.total = xqslab_seconds_since(total_t0);

    if (log != nullptr && opt.print_contraction_stats) {
        result.stats.print(*log);
    }

    return result;
}

} // namespace b4support
