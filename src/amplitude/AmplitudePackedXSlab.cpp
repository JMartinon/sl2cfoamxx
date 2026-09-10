// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "amplitude/AmplitudePackedXSlab.hpp"

namespace b4support {

void add_packed_qlocal_stats(PackedQLocalContractionStats& dst,
                                    const PackedQLocalContractionStats& src) noexcept
{
    dst.T1_values_total += src.T1_values_total;
    dst.T3_values_total += src.T3_values_total;

    dst.active_q_blocks += src.active_q_blocks;
    dst.T2_rows_built += src.T2_rows_built;
    dst.T4_rows_built += src.T4_rows_built;
    dst.final_pairs += src.final_pairs;

    dst.project_C_times_T2_blocks += src.project_C_times_T2_blocks;
    dst.project_Ct_times_T4_blocks += src.project_Ct_times_T4_blocks;
    dst.pairwise_blocks += src.pairwise_blocks;

    dst.axpy_K1 += src.axpy_K1;
    dst.axpy_K2 += src.axpy_K2;
    dst.axpy_K3 += src.axpy_K3;
    dst.axpy_K4 += src.axpy_K4;
    dst.axpy_generic += src.axpy_generic;

    dst.dot_K1 += src.dot_K1;
    dst.dot_K2 += src.dot_K2;
    dst.dot_K3 += src.dot_K3;
    dst.dot_K4 += src.dot_K4;
    dst.dot_generic += src.dot_generic;

    for (std::size_t i = 0; i < dst.project_axpy_K_hist.size(); ++i) {
        dst.project_axpy_K_hist[i] += src.project_axpy_K_hist[i];
    }
    for (std::size_t i = 0; i < dst.final_dot_K_hist.size(); ++i) {
        dst.final_dot_K_hist[i] += src.final_dot_K_hist[i];
    }

    dst.project_axpy_K_overflow += src.project_axpy_K_overflow;
    dst.final_dot_K_overflow += src.final_dot_K_overflow;
    dst.project_axpy_full_range += src.project_axpy_full_range;
    dst.final_dot_full_range += src.final_dot_full_range;

    dst.time_T1_support += src.time_T1_support;
    dst.time_T1_values += src.time_T1_values;
    dst.time_T3_support += src.time_T3_support;
    dst.time_T3_values += src.time_T3_values;
    dst.time_q_loop += src.time_q_loop;
    dst.time_total += src.time_total;
}

std::uint64_t xslab_dense_T1T3_bytes_per_x(
    const AmplitudeInputSupportPlan_HalfSjC& support)
{
    const std::uint64_t K3d = static_cast<std::uint64_t>(support.channels.k3.size());
    const std::uint64_t K4d = static_cast<std::uint64_t>(support.channels.k4.size());
    const std::uint64_t L23 = static_cast<std::uint64_t>(support.l23.size());
    const std::uint64_t L24 = static_cast<std::uint64_t>(support.l24.size());
    const std::uint64_t L25 = static_cast<std::uint64_t>(support.l25.size());
    const std::uint64_t L35 = static_cast<std::uint64_t>(support.l35.size());
    const std::uint64_t L45 = static_cast<std::uint64_t>(support.l45.size());

    const std::uint64_t T1_values =
        xslab_sat_mul_u64(L45,
        xslab_sat_mul_u64(K4d,
        xslab_sat_mul_u64(L35, L25)));

    const std::uint64_t T3_values =
        xslab_sat_mul_u64(L23,
        xslab_sat_mul_u64(K3d,
        xslab_sat_mul_u64(L24, L25)));

    std::uint64_t bytes =
        xslab_bytes_for_values(xslab_sat_add_u64(T1_values, T3_values));

    const std::uint64_t T1_blocks =
        xslab_sat_mul_u64(L45, xslab_sat_mul_u64(K4d, L35));
    const std::uint64_t T3_blocks =
        xslab_sat_mul_u64(L23, xslab_sat_mul_u64(K3d, L24));

    bytes = xslab_sat_add_u64(
        bytes,
        xslab_sat_mul_u64(xslab_sat_add_u64(T1_blocks, T3_blocks),
                          static_cast<std::uint64_t>(sizeof(Range1D)))
    );

    return bytes;
}

PackedAmplitudeXCost estimate_amplitude_x_cost_from_global_support(
    const AmplitudeInputSupportPlan_HalfSjC& support,
    std::size_t ix,
    bool include_slab_sj_metadata,
    bool dense_T1T3_estimate)
{
    const std::size_t X   = support.channels.xvals.size();
    const std::size_t K3d = support.channels.k3.size();
    const std::size_t K4d = support.channels.k4.size();
    const std::size_t L23 = support.l23.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L45 = support.l45.size();

    if (ix >= X) {
        throw std::out_of_range("estimate_amplitude_x_cost_from_global_support: ix out of range");
    }

    PackedAmplitudeXCost c;
    c.ix = ix;

    std::uint64_t values = 0;

    for (std::size_t il25 = 0; il25 < L25; ++il25) {
        values = xslab_sat_add_u64(
            values,
            support.sjA.blocks[idx2(ix, il25, L25)].count());
        values = xslab_sat_add_u64(
            values,
            support.sjE.blocks[idx2(ix, il25, L25)].count());
    }

    for (std::size_t ik4 = 0; ik4 < K4d; ++ik4) {
        for (std::size_t il35 = 0; il35 < L35; ++il35) {
            for (std::size_t il45 = 0; il45 < L45; ++il45) {
                values = xslab_sat_add_u64(
                    values,
                    support.sjB.blocks[idx4(ix, ik4, il35, il45,
                                             K4d, L35, L45)].count());
            }
        }
    }

    for (std::size_t il24 = 0; il24 < L24; ++il24) {
        for (std::size_t il34 = 0; il34 < L34; ++il34) {
            for (std::size_t il35 = 0; il35 < L35; ++il35) {
                values = xslab_sat_add_u64(
                    values,
                    support.sjC.blocks[idx4(ix, il24, il34, il35,
                                             L24, L34, L35)].count());
            }
        }
    }

    for (std::size_t ik3 = 0; ik3 < K3d; ++ik3) {
        for (std::size_t il23 = 0; il23 < L23; ++il23) {
            for (std::size_t il24 = 0; il24 < L24; ++il24) {
                values = xslab_sat_add_u64(
                    values,
                    support.sjD.blocks[idx4(ix, ik3, il23, il24,
                                             K3d, L23, L24)].count());
            }
        }
    }

    c.sj_payload_bytes = xslab_bytes_for_values(values);

    if (include_slab_sj_metadata) {
        std::uint64_t meta = 0;
        meta = xslab_sat_add_u64(meta,
            xslab_sat_mul_u64(static_cast<std::uint64_t>(L25),
                              static_cast<std::uint64_t>(sizeof(Range1D)))); // sjA
        meta = xslab_sat_add_u64(meta,
            xslab_sat_mul_u64(static_cast<std::uint64_t>(L25),
                              static_cast<std::uint64_t>(sizeof(Range1D)))); // sjE
        meta = xslab_sat_add_u64(meta,
            xslab_sat_mul_u64(
                xslab_sat_mul_u64(static_cast<std::uint64_t>(K4d),
                xslab_sat_mul_u64(static_cast<std::uint64_t>(L35),
                                  static_cast<std::uint64_t>(L45))),
                static_cast<std::uint64_t>(sizeof(Range1D)))); // sjB
        meta = xslab_sat_add_u64(meta,
            xslab_sat_mul_u64(
                xslab_sat_mul_u64(static_cast<std::uint64_t>(K3d),
                xslab_sat_mul_u64(static_cast<std::uint64_t>(L23),
                                  static_cast<std::uint64_t>(L24))),
                static_cast<std::uint64_t>(sizeof(Range1D)))); // sjD
        meta = xslab_sat_add_u64(meta,
            xslab_sat_mul_u64(
                xslab_sat_mul_u64(static_cast<std::uint64_t>(L24),
                xslab_sat_mul_u64(static_cast<std::uint64_t>(L34),
                                  static_cast<std::uint64_t>(L35))),
                static_cast<std::uint64_t>(sizeof(Range2D)))); // sjC physical blocks
        meta = xslab_sat_add_u64(meta,
            xslab_sat_mul_u64(
                xslab_sat_mul_u64(static_cast<std::uint64_t>(L24),
                xslab_sat_mul_u64(static_cast<std::uint64_t>(L34),
                                  static_cast<std::uint64_t>(L35))),
                static_cast<std::uint64_t>(sizeof(AmplitudeSjCHalfBlockMap)))); // sjC map

        c.sj_metadata_bytes = meta;
    }

    if (dense_T1T3_estimate) {
        c.T1T3_bytes = xslab_dense_T1T3_bytes_per_x(support);
    }

    return c;
}

PackedAmplitudeXSlabPlan choose_packed_amplitude_x_slabs_from_memory_generated_support(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const B4Spins& spins,
    const PackedAmplitudeXSlabOptions& opt)
{
    PackedAmplitudeXSlabPlan plan;
    plan.physical_ram = xslab_physical_ram_bytes();

    if (opt.memory_budget_bytes != 0) {
        plan.memory_budget_bytes = opt.memory_budget_bytes;
    } else {
        plan.memory_budget_bytes =
            xslab_automatic_memory_budget_bytes(opt.ram_fraction);
    }

    plan.booster_payload_bytes =
        static_support.B2.payload_bytes() + static_support.B3.payload_bytes() +
        static_support.B4.payload_bytes() + static_support.B5.payload_bytes();

    plan.output_bytes = opt.output_bytes;
    plan.extra_fixed_bytes = opt.extra_fixed_bytes;

    // In generated-support mode this is intentionally only the static metadata:
    // B2/B3/B4/B5 ranges, axes, channels, and half-sjC axis maps.  No global
    // sjA/sjB/sjC/sjD/sjE block metadata exists here.  Tiny miracles still count.
    plan.existing_support_metadata_bytes =
        opt.include_existing_input_support_metadata ? static_support.total_metadata_bytes() : 0ull;

    plan.qlocal_thread_scratch_bytes =
        opt.qlocal_thread_scratch_bytes != 0
        ? opt.qlocal_thread_scratch_bytes
        : xslab_default_qlocal_thread_scratch_bytes(static_support);
    plan.safety_margin_bytes =
        opt.safety_margin_bytes != 0
        ? opt.safety_margin_bytes
        : xslab_default_safety_margin_bytes(
              xslab_hard_memory_capacity_bytes());

    plan.fixed_bytes = 0;
    plan.fixed_bytes = xslab_sat_add_u64(plan.fixed_bytes, plan.booster_payload_bytes);
    plan.fixed_bytes = xslab_sat_add_u64(plan.fixed_bytes, plan.output_bytes);
    plan.fixed_bytes = xslab_sat_add_u64(plan.fixed_bytes, plan.existing_support_metadata_bytes);
    plan.fixed_bytes = xslab_sat_add_u64(plan.fixed_bytes, plan.qlocal_thread_scratch_bytes);
    plan.fixed_bytes = xslab_sat_add_u64(plan.fixed_bytes, plan.safety_margin_bytes);
    plan.fixed_bytes = xslab_sat_add_u64(plan.fixed_bytes, plan.extra_fixed_bytes);

    plan.available_for_variable_bytes =
        plan.fixed_bytes >= plan.memory_budget_bytes
        ? 0ull
        : plan.memory_budget_bytes - plan.fixed_bytes;

    const std::size_t X = static_support.channels.xvals.size();
    plan.xcosts.reserve(X);
    for (std::size_t ix = 0; ix < X; ++ix) {
        plan.xcosts.push_back(
            estimate_amplitude_x_cost_from_generated_support(
                static_support,
                spins,
                ix,
                opt.include_slab_sj_metadata,
                opt.dense_T1T3_estimate));
    }

    const std::size_t max_tile = opt.max_x_per_slab == 0 ? 1 : opt.max_x_per_slab;
    const std::size_t min_tile = std::max<std::size_t>(1, opt.min_x_per_slab);

    std::size_t begin = 0;
    while (begin < X) {
        PackedAmplitudeXSlab slab;
        slab.ix_begin = begin;
        slab.ix_end = begin;

        std::uint64_t sj_payload = 0;
        std::uint64_t sj_meta = 0;
        std::uint64_t T1T3_sum = 0;
        std::uint64_t T1T3_max = 0;

        while (slab.ix_end < X && slab.size() < max_tile) {
            const PackedAmplitudeXCost& cx = plan.xcosts[slab.ix_end];

            const std::uint64_t next_sj_payload = xslab_sat_add_u64(sj_payload, cx.sj_payload_bytes);
            const std::uint64_t next_sj_meta = xslab_sat_add_u64(sj_meta, cx.sj_metadata_bytes);
            const std::uint64_t next_T1T3_sum = xslab_sat_add_u64(T1T3_sum, cx.T1T3_bytes);
            const std::uint64_t next_T1T3_max = std::max(T1T3_max, cx.T1T3_bytes);

            const std::uint64_t T1T3_for_slab =
                opt.reuse_T1T3_per_x ? next_T1T3_max : next_T1T3_sum;

            std::uint64_t variable = xslab_sat_add_u64(next_sj_payload, next_sj_meta);
            variable = xslab_sat_add_u64(variable, T1T3_for_slab);
            variable = xslab_add_fractional_overhead(variable, opt.variable_overhead_fraction);

            const std::uint64_t peak = xslab_sat_add_u64(plan.fixed_bytes, variable);
            const bool would_fit = peak <= plan.memory_budget_bytes;
            const bool must_take_one = slab.ix_end == begin;
            const bool below_min = slab.size() < min_tile;

            if (!would_fit && !must_take_one && !below_min) {
                break;
            }

            sj_payload = next_sj_payload;
            sj_meta = next_sj_meta;
            T1T3_sum = next_T1T3_sum;
            T1T3_max = next_T1T3_max;
            ++slab.ix_end;

            slab.sj_payload_bytes = sj_payload;
            slab.sj_metadata_bytes = sj_meta;
            slab.T1T3_bytes = opt.reuse_T1T3_per_x ? T1T3_max : T1T3_sum;
            slab.estimated_variable_bytes = xslab_add_fractional_overhead(
                xslab_sat_add_u64(
                    xslab_sat_add_u64(slab.sj_payload_bytes, slab.sj_metadata_bytes),
                    slab.T1T3_bytes),
                opt.variable_overhead_fraction);
            slab.estimated_peak_bytes = xslab_sat_add_u64(plan.fixed_bytes, slab.estimated_variable_bytes);
            slab.fits_budget = slab.estimated_peak_bytes <= plan.memory_budget_bytes;

            if (!slab.fits_budget && must_take_one) {
                break;
            }
        }

        if (slab.ix_end == begin) {
            slab.ix_end = begin + 1;
            slab.fits_budget = false;
        }

        if (!opt.skip_unfitting_slabs || slab.fits_budget) {
            plan.slabs.push_back(slab);
        }
        begin = slab.ix_end;
    }

    return plan;
}

std::uint64_t xslab_full_dense_sj_payload_upper_bound_bytes(
    const AmplitudeInputSupportPlan_HalfSjC& support)
{
    const std::uint64_t X   = static_cast<std::uint64_t>(support.channels.xvals.size());
    const std::uint64_t K2d = static_cast<std::uint64_t>(support.channels.k2.size());
    const std::uint64_t K3d = static_cast<std::uint64_t>(support.channels.k3.size());
    const std::uint64_t K4d = static_cast<std::uint64_t>(support.channels.k4.size());
    const std::uint64_t K5d = static_cast<std::uint64_t>(support.channels.k5.size());

    const std::uint64_t L23 = static_cast<std::uint64_t>(support.l23.size());
    const std::uint64_t L24 = static_cast<std::uint64_t>(support.l24.size());
    const std::uint64_t L25 = static_cast<std::uint64_t>(support.l25.size());
    const std::uint64_t L34 = static_cast<std::uint64_t>(support.l34.size());
    const std::uint64_t L35 = static_cast<std::uint64_t>(support.l35.size());
    const std::uint64_t L45 = static_cast<std::uint64_t>(support.l45.size());

    std::uint64_t values = 0;

    // sjA(x,l25) -> k5, fixed i1.
    values = xslab_sat_add_u64(values, xslab_sat_mul_many_u64({X, L25, K5d}));

    // sjB(x,k4,l35,l45) -> k5.
    values = xslab_sat_add_u64(values, xslab_sat_mul_many_u64({X, K4d, L35, L45, K5d}));

    // sjC(x,l24,l34,l35) -> k3 x k4.  Use the logical dense upper bound;
    // half-storage and triangle sparsity only make the real packed payload smaller.
    values = xslab_sat_add_u64(values, xslab_sat_mul_many_u64({X, L24, L34, L35, K3d, K4d}));

    // sjD(x,k3,l23,l24) -> k2.
    values = xslab_sat_add_u64(values, xslab_sat_mul_many_u64({X, K3d, L23, L24, K2d}));

    // sjE(x,l25) -> k2, fixed i1.
    values = xslab_sat_add_u64(values, xslab_sat_mul_many_u64({X, L25, K2d}));

    return xslab_bytes_for_values(values);
}

PackedAmplitudeSupportFullXBypassEstimate estimate_support_full_x_bypass_peak(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const PackedAmplitudeXSlabOptions& opt,
    double budget_fraction)
{
    PackedAmplitudeSupportFullXBypassEstimate e;
    e.physical_ram = xslab_physical_ram_bytes();
    e.memory_budget_bytes = xslab_memory_budget_from_options(opt, e.physical_ram);

    long double allowed = static_cast<long double>(e.memory_budget_bytes) *
                          static_cast<long double>(budget_fraction);
    const long double maxv =
        static_cast<long double>(std::numeric_limits<std::uint64_t>::max());
    e.allowed_peak_bytes = allowed >= maxv
        ? std::numeric_limits<std::uint64_t>::max()
        : static_cast<std::uint64_t>(std::max<long double>(0.0L, allowed));

    const std::uint64_t booster_payload_bytes =
        static_support.B2.payload_bytes() + static_support.B3.payload_bytes() +
        static_support.B4.payload_bytes() + static_support.B5.payload_bytes();

    const std::uint64_t static_metadata_bytes =
        opt.include_existing_input_support_metadata ? static_support.total_metadata_bytes() : 0ull;
    const std::uint64_t qlocal_scratch =
        opt.qlocal_thread_scratch_bytes != 0
        ? opt.qlocal_thread_scratch_bytes
        : xslab_default_qlocal_thread_scratch_bytes(static_support);
    const std::uint64_t safety =
        opt.safety_margin_bytes != 0
        ? opt.safety_margin_bytes
        : xslab_default_safety_margin_bytes(
              xslab_hard_memory_capacity_bytes());

    e.fixed_bytes = 0;
    e.fixed_bytes = xslab_sat_add_u64(e.fixed_bytes, booster_payload_bytes);
    e.fixed_bytes = xslab_sat_add_u64(e.fixed_bytes, opt.output_bytes);
    e.fixed_bytes = xslab_sat_add_u64(e.fixed_bytes, static_metadata_bytes);
    e.fixed_bytes = xslab_sat_add_u64(e.fixed_bytes, qlocal_scratch);
    e.fixed_bytes = xslab_sat_add_u64(e.fixed_bytes, safety);
    e.fixed_bytes = xslab_sat_add_u64(e.fixed_bytes, opt.extra_fixed_bytes);

    e.dense_sj_payload_upper_bound_bytes =
        xslab_full_dense_sj_payload_upper_bound_bytes(static_support);
    e.sj_metadata_upper_bound_bytes =
        xslab_full_sj_metadata_upper_bound_bytes(static_support);
    e.T1T3_bytes = opt.dense_T1T3_estimate
        ? xslab_dense_T1T3_bytes_per_x(static_support)
        : 0ull;

    std::uint64_t variable = 0;
    variable = xslab_sat_add_u64(variable, e.dense_sj_payload_upper_bound_bytes);
    variable = xslab_sat_add_u64(variable, e.sj_metadata_upper_bound_bytes);
    variable = xslab_sat_add_u64(variable, e.T1T3_bytes);
    e.estimated_variable_bytes =
        xslab_add_fractional_overhead(variable, opt.variable_overhead_fraction);
    e.estimated_peak_bytes = xslab_sat_add_u64(e.fixed_bytes, e.estimated_variable_bytes);
    e.fits = e.estimated_peak_bytes <= e.allowed_peak_bytes;

    return e;
}

PackedAmplitudeXSlabResult contract_packed_amplitude_xslab_streamed_generated_support(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const PackedAmplitudeBoosters& packed_B,
    const std::array<int, 10>& spins,
    const std::array<int, 5>& fixed_intertwiners,
    const PackedAmplitudeXSlabRunOptions& run_opt,
    std::ostream* log)
{
    using clock = std::chrono::high_resolution_clock;

    const B4Spins b4spins(spins);

    PackedAmplitudeXSlabResult result;
    result.stats.clear();

    const auto total_t0 = clock::now();
    const std::size_t X = static_support.channels.xvals.size();

    const bool user_allows_full_range =
        run_opt.planner.max_x_per_slab == 0
        ? (X <= 1u)
        : (run_opt.planner.max_x_per_slab >= X);

    // Support-level full-x bypass.  This happens before the detailed generated
    // per-x planner.  It is meant for the small/medium regime where the ordinary
    // full support is plainly safe; in that case generating one-x supports just
    // to discover that [0,X) fits is a bureaucratic tax on the CPU.
    if (run_opt.enable_full_x_bypass &&
        run_opt.enable_support_full_x_bypass &&
        user_allows_full_range)
    {
        const auto estimate_t0 = clock::now();
        const PackedAmplitudeSupportFullXBypassEstimate bypass_est =
            estimate_support_full_x_bypass_peak(
                static_support,
                run_opt.planner,
                run_opt.support_full_x_bypass_budget_fraction);
        result.timings.planning += xslab_seconds_since(estimate_t0);

        if (bypass_est.fits) {
            result.used_full_x_bypass = true;

            if (log != nullptr && run_opt.print_slab_timing) {
                *log << "\n[xslab] support-level full-x bypass: dense upper-bound peak ";
                xslab_print_bytes(*log, bypass_est.estimated_peak_bytes);
                *log << " <= allowed ";
                xslab_print_bytes(*log, bypass_est.allowed_peak_bytes);
                *log << ", building full sj support once\n";
            }

            double amp = 0.0;
            auto release_t0 = clock::now();
            {
                auto t0 = clock::now();
                const int Dl = xslab_infer_Dl_from_static_support(static_support);
                AmplitudeInputSupportPlan_HalfSjC full_support =
                    build_amplitude_input_support_plan_half_sjC(
                        b4spins,
                        Dl,
                        static_support.channels);

                // packed_B was produced from static_support.  These supports
                // should match the full builder, but assign them explicitly so
                // offsets remain aligned even if future support builders learn
                // new tricks and ruin our day.
                full_support.B2 = static_support.B2;
                full_support.B3 = static_support.B3;
                full_support.B4 = static_support.B4;
                full_support.B5 = static_support.B5;
                full_support.l23 = static_support.l23;
                full_support.l24 = static_support.l24;
                full_support.l25 = static_support.l25;
                full_support.l34 = static_support.l34;
                full_support.l35 = static_support.l35;
                full_support.l45 = static_support.l45;
                result.timings.support_slice += xslab_seconds_since(t0);

                t0 = clock::now();
                PackedAmplitudeSJTensors_HalfSjC full_sj;
                fill_packed_amplitude_sj_tensors_rec_half_sjC(
                    b4spins,
                    full_support,
                    full_sj);
                result.timings.sj_fill += xslab_seconds_since(t0);

                if (log != nullptr && run_opt.print_slab_sj_size) {
                    full_sj.print(*log);
                }

                t0 = clock::now();
                PackedQLocalContractionStats stats;
                PackedQLocalContractionStats* stats_ptr =
                    run_opt.print_contraction_stats ? &stats : nullptr;

                amp = contract_packed_amplitude_general(
                    full_support,
                    full_sj,
                    packed_B,
                    spins,
                    fixed_intertwiners,
                    run_opt.order,
                    run_opt.contraction_options,
                    stats_ptr);
                result.timings.contraction += xslab_seconds_since(t0);

                if (stats_ptr != nullptr) {
                    result.stats = stats;
                }

                release_t0 = clock::now();
            }
            result.timings.workspace_release += xslab_seconds_since(release_t0);
            result.amplitude = amp;
            result.timings.total = xslab_seconds_since(total_t0);

            if (log != nullptr && run_opt.print_contraction_stats) {
                result.stats.print(*log);
            }

            return result;
        }
    }

    {
        const auto plan_t0 = clock::now();
        result.plan = choose_packed_amplitude_x_slabs_from_memory_generated_support(
            static_support,
            b4spins,
            run_opt.planner);
        result.timings.planning += xslab_seconds_since(plan_t0);
    }

    if (log != nullptr && run_opt.print_plan) {
        result.plan.print(*log);
    }

    if (run_opt.require_all_slabs_fit && !result.plan.all_slabs_fit()) {
        throw b4utils::ResourceLimitFailure(
            "contract_packed_amplitude_xslab_streamed_generated_support: at least one x-slab does not fit the selected memory budget. "
            "Use a larger memory budget, disable require_all_slabs_fit, or implement q-slab sjC streaming.");
    }

    const bool use_full_x_bypass =
        run_opt.enable_full_x_bypass &&
        is_single_full_xslab_plan(result.plan, X);

    double amp = 0.0;

    auto release_t0 = clock::now();
    {
        PackedAmplitudeSJTensors_HalfSjC sj_workspace;

        const std::size_t nslabs = use_full_x_bypass ? 1u : result.plan.slabs.size();

        for (std::size_t islab = 0; islab < nslabs; ++islab) {
            const std::size_t ix_begin = use_full_x_bypass ? 0u : result.plan.slabs[islab].ix_begin;
            const std::size_t ix_end   = use_full_x_bypass ? X  : result.plan.slabs[islab].ix_end;

            if (!use_full_x_bypass && !result.plan.slabs[islab].fits_budget && run_opt.require_all_slabs_fit) {
                continue;
            }

            if (use_full_x_bypass) {
                result.used_full_x_bypass = true;
                if (log != nullptr && run_opt.print_slab_timing) {
                    *log << "\n[xslab] generated-support full-x bypass: planner returned one slab [0,"
                         << X << "), building full sj support once\n";
                }
            } else if (log != nullptr && run_opt.print_slab_timing) {
                const PackedAmplitudeXSlab& slab = result.plan.slabs[islab];
                *log << "\n[xslab] generated slab " << islab
                     << " ix=[" << slab.ix_begin << "," << slab.ix_end << ")"
                     << " X=" << slab.size()
                     << " estimated peak=";
                xslab_print_bytes(*log, slab.estimated_peak_bytes);
                *log << "\n";
            }

            auto t0 = clock::now();
            AmplitudeInputSupportPlan_HalfSjC slab_support =
                build_amplitude_sj_xslab_support_half_sjC(
                    static_support,
                    b4spins,
                    ix_begin,
                    ix_end);
            result.timings.support_slice += xslab_seconds_since(t0);

            t0 = clock::now();
            fill_packed_amplitude_sj_tensors_rec_half_sjC(
                b4spins,
                slab_support,
                sj_workspace);
            result.timings.sj_fill += xslab_seconds_since(t0);

            if (log != nullptr && run_opt.print_slab_sj_size) {
                sj_workspace.print(*log);
            }

            t0 = clock::now();
            PackedQLocalContractionStats slab_stats;
            PackedQLocalContractionStats* stats_ptr =
                run_opt.print_contraction_stats ? &slab_stats : nullptr;

            const double slab_amp = contract_packed_amplitude_general(
                slab_support,
                sj_workspace,
                packed_B,
                spins,
                fixed_intertwiners,
                run_opt.order,
                run_opt.contraction_options,
                stats_ptr);

            result.timings.contraction += xslab_seconds_since(t0);
            amp += slab_amp;

            if (stats_ptr != nullptr) {
                add_packed_qlocal_stats(result.stats, slab_stats);
            }

            if (log != nullptr && run_opt.print_slab_timing) {
                *log << "[xslab] generated slab " << islab
                     << " amplitude=" << std::scientific << std::setprecision(17)
                     << slab_amp << std::fixed << std::setprecision(6)
                     << " cumulative=" << std::scientific << std::setprecision(17)
                     << amp << std::fixed << std::setprecision(6)
                     << "\n";
            }
        }

        release_t0 = clock::now();
    }

    result.timings.workspace_release += xslab_seconds_since(release_t0);
    result.amplitude = amp;
    result.timings.total = xslab_seconds_since(total_t0);

    if (log != nullptr && run_opt.print_contraction_stats) {
        result.stats.print(*log);
    }

    return result;
}

} // namespace b4support
