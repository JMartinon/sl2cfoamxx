// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "amplitude/AmplitudePackedAutoSlab.hpp"

namespace b4support {

PackedAmplitudeXQSlabOptions packed_auto_make_xq_options(
    const PackedAmplitudeAutoOptions& opt,
    const AmplitudeInputSupportPlan_HalfSjC& static_support)
{
    PackedAmplitudeXQSlabOptions xq = opt.xqslab;

    if (!opt.inherit_memory_options_to_xqslab) {
        return xq;
    }

    const PackedAmplitudeXSlabOptions& xs = opt.xslab.planner;

    if (xq.memory_budget_bytes == 0) {
        xq.memory_budget_bytes = xs.memory_budget_bytes;
    }

    if (xq.ram_fraction == B4PACK_XQSLAB_DEFAULT_RAM_FRACTION &&
        xs.ram_fraction != B4PACK_XSLAB_DEFAULT_RAM_FRACTION)
    {
        xq.ram_fraction = xs.ram_fraction;
    }

    if (xq.output_bytes == sizeof(double)) xq.output_bytes = xs.output_bytes;
    if (xq.extra_fixed_bytes == 0) xq.extra_fixed_bytes = xs.extra_fixed_bytes;
    if (xq.safety_margin_bytes == 0) xq.safety_margin_bytes = xs.safety_margin_bytes;
    if (xq.qlocal_thread_scratch_bytes == 0) {
        xq.qlocal_thread_scratch_bytes = xs.qlocal_thread_scratch_bytes;
    }
    if (xq.qlocal_thread_scratch_bytes == 0) {
        xq.qlocal_thread_scratch_bytes = xslab_default_qlocal_thread_scratch_bytes(static_support);
    }

    if (xq.variable_overhead_fraction == B4PACK_XQSLAB_DEFAULT_VARIABLE_OVERHEAD_FRACTION &&
        xs.variable_overhead_fraction != B4PACK_XSLAB_DEFAULT_VARIABLE_OVERHEAD_FRACTION)
    {
        xq.variable_overhead_fraction = xs.variable_overhead_fraction;
    }

    return xq;
}

PackedAmplitudeXSlabResult contract_packed_amplitude_xslab_streamed_generated_support_with_plan(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const PackedAmplitudeBoosters& packed_B,
    const std::array<int, 10>& spins,
    const std::array<int, 5>& fixed_intertwiners,
    const PackedAmplitudeXSlabPlan& plan,
    const PackedAmplitudeXSlabRunOptions& run_opt,
    std::ostream* log)
{
    using clock = std::chrono::high_resolution_clock;
    const auto total_t0 = clock::now();

    PackedAmplitudeXSlabResult result;
    result.stats.clear();
    result.plan = plan;

    const B4Spins b4spins(spins);
    const std::size_t X = static_support.channels.xvals.size();

    if (run_opt.require_all_slabs_fit && !result.plan.all_slabs_fit()) {
        throw b4utils::ResourceLimitFailure(
            "contract_packed_amplitude_xslab_streamed_generated_support_with_plan: "
            "at least one x-slab does not fit the selected memory budget."
        );
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

            if (!use_full_x_bypass && !result.plan.slabs[islab].fits_budget) {
                if (run_opt.require_all_slabs_fit) {
                    continue;
                }
                // If the caller intentionally allowed unfitting slabs, do not
                // execute them here.  The automatic dispatcher should have
                // selected x+q before reaching this function.
                continue;
            }

            if (use_full_x_bypass) {
                result.used_full_x_bypass = true;
                if (log != nullptr && run_opt.print_slab_timing) {
                    *log << "\n[xslab:auto] generated-support full-x bypass from precomputed plan [0,"
                         << X << ")\n";
                }
            } else if (log != nullptr && run_opt.print_slab_timing) {
                const PackedAmplitudeXSlab& slab = result.plan.slabs[islab];
                *log << "\n[xslab:auto] generated slab " << islab
                     << " ix=[" << slab.ix_begin << "," << slab.ix_end << ")"
                     << " X=" << slab.size()
                     << " estimated peak=";
                xslab_print_bytes(*log, slab.estimated_peak_bytes);
                *log << "\n";
            }

            auto t0 = clock::now();
            AmplitudeInputSupportPlan_HalfSjC slab_support =
                packed_auto_build_generated_sj_support_half_sjC(
                    static_support,
                    b4spins,
                    ix_begin,
                    ix_end);
            result.timings.support_slice += packed_auto_seconds_since(t0);

            t0 = clock::now();
            fill_packed_amplitude_sj_tensors_rec_half_sjC(
                b4spins,
                slab_support,
                sj_workspace);
            result.timings.sj_fill += packed_auto_seconds_since(t0);

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

            result.timings.contraction += packed_auto_seconds_since(t0);
            amp += slab_amp;

            if (stats_ptr != nullptr) {
                add_packed_qlocal_stats(result.stats, slab_stats);
            }

            if (log != nullptr && run_opt.print_slab_timing) {
                *log << "[xslab:auto] slab " << islab
                     << " amplitude=" << std::scientific << std::setprecision(17)
                     << slab_amp << " cumulative=" << amp
                     << std::fixed << std::setprecision(6) << "\n";
            }
        }

        release_t0 = clock::now();
    }
    result.timings.workspace_release += packed_auto_seconds_since(release_t0);
    result.amplitude = amp;
    result.timings.total = packed_auto_seconds_since(total_t0);

    if (log != nullptr && run_opt.print_contraction_stats) {
        result.stats.print(*log);
    }

    return result;
}

PackedAmplitudeAutoResult contract_packed_amplitude_auto_streamed_generated_support(
    const AmplitudeInputSupportPlan_HalfSjC& static_support,
    const PackedAmplitudeBoosters& packed_B,
    const std::array<int, 10>& spins,
    const std::array<int, 5>& fixed_intertwiners,
    PackedAmplitudeAutoOptions opt,
    std::ostream* log)
{
    using clock = std::chrono::high_resolution_clock;
    const auto total_t0 = clock::now();

    PackedAmplitudeAutoResult result;
    const B4Spins b4spins(spins);
    const std::size_t X = static_support.channels.xvals.size();

    // Auto mode must retain every slab: dropping an unfitting slab would make an
    // incomplete plan appear valid and silently omit numerical contributions.
    opt.xslab.planner.skip_unfitting_slabs = false;

    // ---------------------------------------------------------------------
    // 1. Support-level full-x test.
    // ---------------------------------------------------------------------
    {
        const auto t0 = clock::now();
        result.full_x_estimate = estimate_support_full_x_bypass_peak(
            static_support,
            opt.xslab.planner,
            opt.xslab.support_full_x_bypass_budget_fraction);
        result.timings.full_x_estimate += packed_auto_seconds_since(t0);
    }

    const bool user_forced_x_cap =
        opt.xslab.planner.max_x_per_slab != 0 &&
        opt.xslab.planner.max_x_per_slab < X;

    const bool can_use_full_x =
        opt.enable_full_x &&
        opt.xslab.enable_full_x_bypass &&
        opt.xslab.enable_support_full_x_bypass &&
        !user_forced_x_cap &&
        result.full_x_estimate.fits;

    if (can_use_full_x) {
        if (log != nullptr && opt.print_decision) {
            *log << "\n[auto] selected full-x: dense/support upper-bound peak ";
            xslab_print_bytes(*log, result.full_x_estimate.estimated_peak_bytes);
            *log << " <= allowed ";
            xslab_print_bytes(*log, result.full_x_estimate.allowed_peak_bytes);
            *log << "\n";
        }

        auto run_opt = opt.xslab;
        run_opt.enable_full_x_bypass = true;
        run_opt.enable_support_full_x_bypass = true;
        run_opt.require_all_slabs_fit = true;

        const auto t0 = clock::now();
        result.xslab_result = contract_packed_amplitude_xslab_streamed_generated_support(
            static_support,
            packed_B,
            spins,
            fixed_intertwiners,
            run_opt,
            log);
        result.timings.execution += packed_auto_seconds_since(t0);

        result.mode = PackedAmplitudeAutoMode::FullX;
        result.amplitude = result.xslab_result.amplitude;
        result.xslab_plan = result.xslab_result.plan;
        result.timings.total = packed_auto_seconds_since(total_t0);
        if (log != nullptr && opt.print_result_summary) result.print(*log);
        return result;
    }

    // ---------------------------------------------------------------------
    // 2. Generated x-slab planner.
    // ---------------------------------------------------------------------
    if (opt.enable_xslab) {
        const auto t0 = clock::now();
        result.xslab_plan = choose_packed_amplitude_x_slabs_from_memory_generated_support(
            static_support,
            b4spins,
            opt.xslab.planner);
        result.timings.xslab_planning += packed_auto_seconds_since(t0);

        if (log != nullptr && opt.xslab.print_plan) {
            result.xslab_plan.print(*log);
        }

        if (result.xslab_plan.all_slabs_fit()) {
            if (log != nullptr && opt.print_decision) {
                *log << "\n[auto] selected x-slab: all planned x-slabs fit the memory budget";
                if (is_single_full_xslab_plan(result.xslab_plan, X)) {
                    *log << " (single full-x slab by sparse planner)";
                }
                *log << "\n";
            }

            auto run_opt = opt.xslab;
            run_opt.enable_support_full_x_bypass = false; // already tested above
            run_opt.require_all_slabs_fit = true;

            const auto t1 = clock::now();
            result.xslab_result = contract_packed_amplitude_xslab_streamed_generated_support_with_plan(
                static_support,
                packed_B,
                spins,
                fixed_intertwiners,
                result.xslab_plan,
                run_opt,
                log);
            result.timings.execution += packed_auto_seconds_since(t1);

            result.mode = is_single_full_xslab_plan(result.xslab_plan, X)
                ? PackedAmplitudeAutoMode::FullX
                : PackedAmplitudeAutoMode::XSlab;
            result.amplitude = result.xslab_result.amplitude;
            result.timings.total = packed_auto_seconds_since(total_t0);
            if (log != nullptr && opt.print_result_summary) result.print(*log);
            return result;
        }

        if (log != nullptr && opt.print_decision) {
            *log << "\n[auto] x-slab rejected: "
                 << result.xslab_plan.num_unfitting_slabs()
                 << " slab(s) exceed the memory budget; falling back to x+q-slab\n";
        }
    }

    // ---------------------------------------------------------------------
    // 3. x-slice + q-slab fallback.
    // ---------------------------------------------------------------------
    if (!opt.enable_xqslab) {
        throw b4utils::ResourceLimitFailure(
            "contract_packed_amplitude_auto_streamed_generated_support: "
            "full-x/x-slab did not fit, and x+q-slab fallback is disabled."
        );
    }

    PackedAmplitudeXQSlabOptions xq = packed_auto_make_xq_options(opt, static_support);
    xq.require_all_q_slabs_fit = opt.xqslab.require_all_q_slabs_fit;

    if (log != nullptr && opt.print_decision) {
        *log << "\n[auto] selected x+q-slab fallback\n";
    }

    const auto t0 = clock::now();
    result.xqslab_result = contract_packed_amplitude_xqslab_streamed_generated_support(
        static_support,
        packed_B,
        spins,
        xq,
        opt.xslab.order,
        opt.xslab.contraction_options,
        log);
    result.timings.execution += packed_auto_seconds_since(t0);

    result.mode = PackedAmplitudeAutoMode::XQSlab;
    result.amplitude = result.xqslab_result.amplitude;
    result.timings.total = packed_auto_seconds_since(total_t0);
    if (log != nullptr && opt.print_result_summary) result.print(*log);
    return result;
}

} // namespace b4support
