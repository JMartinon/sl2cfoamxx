// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "full_tensor/FullTensorExecutionPlanner.hpp"

#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace b4support {

std::uint64_t full_tensor_planner_default_safety_margin_bytes(std::uint64_t ram) noexcept
{
    constexpr std::uint64_t GiB = 1024ull * 1024ull * 1024ull;
    if (ram == 0) return GiB;

    std::uint64_t margin = ram / 20u; // 5% of RAM.
    margin = std::max<std::uint64_t>(margin, GiB);
    margin = std::min<std::uint64_t>(margin, 32ull * GiB);
    return margin;
}

std::uint64_t full_tensor_planner_output_values_from_channels(const FullTensorChannels& ch,
                                                                     bool* overflow) noexcept
{
    bool ov = false;
    std::uint64_t n = static_cast<std::uint64_t>(ch.i1.size());
    n = full_tensor_planner_sat_mul_u64(n, static_cast<std::uint64_t>(ch.i2.size()), &ov);
    n = full_tensor_planner_sat_mul_u64(n, static_cast<std::uint64_t>(ch.i3.size()), &ov);
    n = full_tensor_planner_sat_mul_u64(n, static_cast<std::uint64_t>(ch.i4.size()), &ov);
    n = full_tensor_planner_sat_mul_u64(n, static_cast<std::uint64_t>(ch.i5.size()), &ov);
    if (overflow) *overflow = ov;
    return n;
}

void FullTensorPreflightOutputPlan::print(std::ostream& os) const
{
    os << "Full-tensor output preflight\n"
       << "============================\n"
       << "A dims              = " << I1 << " x " << I2 << " x " << I3
       << " x " << I4 << " x " << I5 << "\n"
       << "canonical values    = " << canonical_values
       << (canonical_output_overflows_u64 ? " (saturated/overflow)" : "") << "\n"
       << "canonical output    = ";
    full_tensor_planner_print_bytes(os, canonical_output_bytes);
    os << "\n"
       << "output guard        = "
       << (canonical_output_exceeds_guard ? "EXCEEDED" : "ok") << "\n";
}

const char* full_tensor_output_mode_name(FullTensorOutputMode mode) noexcept
{
    switch (mode) {
        case FullTensorOutputMode::ReassociatedAuto: return "ReassociatedAuto";
        case FullTensorOutputMode::ReassociatedOnly: return "ReassociatedOnly";
        case FullTensorOutputMode::ReassociatedOnDisk: return "ReassociatedOnDisk";
        case FullTensorOutputMode::CanonicalDenseInMemory: return "CanonicalDenseInMemory";
        case FullTensorOutputMode::CanonicalDenseOnDisk: return "CanonicalDenseOnDisk";
        case FullTensorOutputMode::BenchmarkOnly: return "BenchmarkOnly";
    }
    return "Unknown";
}

std::size_t full_tensor_checked_size_t_from_u64(std::uint64_t n,
                                                       const char* what)
{
    if (n > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        std::ostringstream oss;
        oss << what << " does not fit in std::size_t (" << n << ")";
        throw std::overflow_error(oss.str());
    }
    return static_cast<std::size_t>(n);
}

std::string full_tensor_format_bytes_string(std::uint64_t bytes)
{
    std::ostringstream oss;
    full_tensor_planner_print_bytes(oss, bytes);
    return oss.str();
}

[[noreturn]] void full_tensor_throw_unavailable_backend(
    const FullTensorExecutionPlan& plan)
{
    std::ostringstream oss;
    oss << "Full-tensor resident backend is not admissible. Planner selected "
        << full_tensor_planner_mode_name(plan.mode) << ".\n"
        << "resident fast peak = " << full_tensor_format_bytes_string(plan.resident_fast_peak_bytes)
        << ", budget = " << full_tensor_format_bytes_string(plan.memory_budget_bytes) << ".\n";

    if (plan.mode == FullTensorPlannerMode::NeedsSjStreaming) {
        oss << "This case requires the SJ-streamed x/q backend with resident boosters, "
               "but that backend is disabled by the current options.";
    } else if (plan.mode == FullTensorPlannerMode::NeedsBoosterStreamingOrImplicitSupport) {
        oss << "This case requires the boundary intertwiner/booster-streamed full-tensor backend, "
               "but that backend is disabled by the current options.";
    } else if (plan.mode == FullTensorPlannerMode::BudgetBelowFixedCost) {
        oss << "The memory budget is below the fixed external+safety cost: "
            << full_tensor_format_bytes_string(plan.external_fixed_bytes)
            << " > " << full_tensor_format_bytes_string(plan.memory_budget_bytes)
            << ". Streaming sj/boosters cannot fix that. Increase memory_budget_bytes "
               "or reduce planner.safety_margin_bytes for artificial tests.";
    } else if (plan.mode == FullTensorPlannerMode::RefuseOutputTooLarge) {
        oss << "Canonical dense output exceeds the configured guard: "
            << full_tensor_format_bytes_string(plan.canonical_output_bytes) << ".";
    } else {
        oss << "Resident execution was disabled by options.";
    }
    if (plan.mode == FullTensorPlannerMode::BudgetBelowFixedCost ||
        plan.mode == FullTensorPlannerMode::RefuseOutputTooLarge) {
        throw b4utils::ResourceLimitFailure(oss.str());
    }
    throw b4utils::UnsupportedOperationFailure(oss.str());
}

std::vector<int> full_tensor_slice_axis(const std::vector<int>& v,
                                               std::size_t begin,
                                               std::size_t end)
{
    begin = std::min(begin, v.size());
    end = std::min(end, v.size());
    if (end < begin) end = begin;
    return std::vector<int>(v.begin() + static_cast<std::ptrdiff_t>(begin),
                            v.begin() + static_cast<std::ptrdiff_t>(end));
}

FullTensorChannels full_tensor_channels_with_x_slice(const FullTensorChannels& ch,
                                                            std::size_t x_begin,
                                                            std::size_t x_end)
{
    FullTensorChannels out = ch;
    out.xvals = full_tensor_slice_axis(ch.xvals, x_begin, x_end);
    return out;
}

FullTensorChannels full_tensor_channels_with_boundary_slices(
    const FullTensorChannels& ch,
    std::size_t i1_begin, std::size_t i1_end,
    std::size_t i2_begin, std::size_t i2_end,
    std::size_t i3_begin, std::size_t i3_end,
    std::size_t i4_begin, std::size_t i4_end,
    std::size_t i5_begin, std::size_t i5_end)
{
    FullTensorChannels out = ch;
    out.i1 = full_tensor_slice_axis(ch.i1, i1_begin, i1_end);
    out.i2 = full_tensor_slice_axis(ch.i2, i2_begin, i2_end);
    out.i3 = full_tensor_slice_axis(ch.i3, i3_begin, i3_end);
    out.i4 = full_tensor_slice_axis(ch.i4, i4_begin, i4_end);
    out.i5 = full_tensor_slice_axis(ch.i5, i5_begin, i5_end);
    return out;
}

std::uint64_t full_tensor_output_peak_bytes_for_mode(
    const FullTensorExecutionPlan& plan,
    const FullTensorAutoOptions& options) noexcept
{
    // Only count outputs that are explicitly required to be resident in RAM.
    // Auto and disk-backed modes are planned as disk-backed; after the backend
    // has been selected, Auto may still choose RAM if the extra buffer fits.
    std::uint64_t out = 0;
    if (full_tensor_output_mode_forces_reassoc_memory(options.output_mode)) {
        out = full_tensor_planner_sat_add_u64(out, plan.reassociated_output_bytes);
    }
    if (full_tensor_output_mode_needs_canonical_memory(options.output_mode)) {
        out = full_tensor_planner_sat_add_u64(out, plan.canonical_output_bytes);
    }
    return out;
}

std::uint64_t full_tensor_estimate_sj_streamed_peak_bytes(
    const FullTensorExecutionPlan& plan,
    const FullTensorInputSupportPlan& slab_support,
    const FullTensorAutoOptions& options) noexcept
{
    std::uint64_t peak = plan.external_fixed_bytes;
    peak = full_tensor_planner_sat_add_u64(peak, plan.support_metadata_bytes); // full support still alive
    peak = full_tensor_planner_sat_add_u64(peak, slab_support.total_metadata_bytes()); // slab support metadata
    peak = full_tensor_planner_sat_add_u64(peak, plan.booster_payload_bytes);
    peak = full_tensor_planner_sat_add_u64(peak, full_tensor_support_sj_payload_bytes(slab_support));
    peak = full_tensor_planner_sat_add_u64(peak, full_tensor_output_peak_bytes_for_mode(plan, options));
    peak = full_tensor_planner_sat_add_u64(
        peak, full_tensor_planner_total_contraction_scratch_bytes(options.planner));
    return peak;
}

std::string full_tensor_gamma_path_component(double gamma)
{
    std::ostringstream oss;
    oss.imbue(std::locale::classic());
    oss << "gamma_" << std::fixed << std::setprecision(3) << gamma;
    std::string out = oss.str();
    for (char& c : out) {
        if (c == '+') c = 'p';
        if (c == '-') c = 'm';
    }
    return out;
}

std::string full_tensor_dspins_path_component(const std::array<int, 10>& spins)
{
    std::ostringstream oss;
    oss << "Dspins";
    for (int s : spins) oss << '_' << s;
    return oss.str();
}

std::string full_tensor_Dl_path_component(std::size_t Dl)
{
    std::ostringstream oss;
    oss << "Dl_" << Dl;
    return oss.str();
}

const char* full_tensor_simplicity_path_component(
    sl2cfoamxx::SimplicityConstraint prescription) noexcept
{
    return prescription == sl2cfoamxx::SimplicityConstraint::gamma_j_p1
        ? "gamma_j_p1"
        : "gamma_j";
}

std::filesystem::path full_tensor_run_output_directory(const FullTensorAutoOptions& options,
                                                              const std::array<int, 10>& spins,
                                                              double gamma)
{
    return options.output_base_dir /
           options.full_tensor_output_subdir /
           full_tensor_simplicity_path_component(
               options.simplicity_constraint) /
           full_tensor_gamma_path_component(gamma) /
           full_tensor_dspins_path_component(spins);
}

std::filesystem::path full_tensor_reassociated_disk_root(const FullTensorAutoOptions& options,
                                                                const std::array<int, 10>& spins,
                                                                std::size_t Dl,
                                                                double gamma)
{
    if (!options.output_root.empty()) {
        return options.output_root;
    }
    return full_tensor_run_output_directory(options, spins, gamma) /
           (full_tensor_Dl_path_component(Dl) + options.output_extension);
}

std::filesystem::path full_tensor_canonical_disk_root(const FullTensorAutoOptions& options,
                                                             const std::array<int, 10>& spins,
                                                             std::size_t Dl,
                                                             double gamma)
{
    if (!options.output_root.empty()) {
        const auto stem = options.output_root.stem().string();
        const auto ext = options.output_root.extension().empty()
            ? options.output_extension
            : options.output_root.extension().string();
        return options.output_root.parent_path() /
               (stem + options.canonical_output_suffix + ext);
    }
    return full_tensor_run_output_directory(options, spins, gamma) /
           (full_tensor_Dl_path_component(Dl) + options.canonical_output_suffix + options.output_extension);
}

std::filesystem::path full_tensor_canonical_reassociated_scratch_root(
    const FullTensorAutoOptions& options,
    const std::array<int, 10>& spins,
    std::size_t Dl,
    double gamma)
{
    const std::filesystem::path canonical = full_tensor_canonical_disk_root(
        options, spins, Dl, gamma);
    return canonical.parent_path() /
           (canonical.filename().string() + ".reassociated-scratch");
}

FullTensorExistingDiskCheck full_tensor_check_existing_reassociated_file(
    const std::filesystem::path& file,
    const std::array<int, 10>& spins,
    std::size_t Dl,
    double gamma,
    const std::array<std::size_t, 5>& canonical_dims,
    std::uint64_t values_count_or_zero,
    sl2cfoamxx::SimplicityConstraint simplicity)
{
    return full_tensor_check_existing_disk_file(
        file,
        tensor_disk::LayoutCode::FullTensorReassociatedReflectionCompactByI1,
        1,
        spins,
        Dl,
        gamma,
        canonical_dims,
        values_count_or_zero,
        simplicity);
}

FullTensorExistingDiskCheck full_tensor_check_existing_canonical_file(
    const std::filesystem::path& file,
    const std::array<int, 10>& spins,
    std::size_t Dl,
    double gamma,
    const std::array<std::size_t, 5>& canonical_dims,
    std::uint64_t values_count_or_zero,
    sl2cfoamxx::SimplicityConstraint simplicity)
{
    return full_tensor_check_existing_disk_file(
        file,
        tensor_disk::LayoutCode::CanonicalDenseColumnMajor,
        5,
        spins,
        Dl,
        gamma,
        canonical_dims,
        values_count_or_zero,
        simplicity);
}

void full_tensor_throw_if_disk_output_exceeds_guard(std::uint64_t bytes,
                                                           std::uint64_t guard,
                                                           const char* what)
{
    if (guard != 0 && bytes > guard) {
        std::ostringstream oss;
        oss << what << " disk output would require " << full_tensor_format_bytes_string(bytes)
            << ", exceeding the configured disk-output guard of "
            << full_tensor_format_bytes_string(guard) << ". Refusing to create a storage incident.";
        throw b4utils::ResourceLimitFailure(oss.str());
    }
}

FullTensorAutoResult compute_full_tensor_auto(
    const std::array<int, 10>& SpinsArray,
    std::size_t Dl,
    double gamma,
    FullTensorAutoOptions options,
    std::size_t log_fact_max)
{
    const std::vector<double> FactTable =
        LogFactTable(static_cast<int>(log_fact_max));
    return compute_full_tensor_auto(
        SpinsArray, Dl, gamma, FactTable, std::move(options));
}

void FullTensorAutoTimings::print(std::ostream& os) const
{
    os << "Full-tensor auto timings\n"
       << "========================\n"
       << std::fixed << std::setprecision(6)
       << "channels          = " << channels_seconds << " s\n"
       << "preflight         = " << preflight_seconds << " s\n"
       << "support           = " << support_seconds << " s\n"
       << "streamed support  = " << streamed_support_seconds << " s\n"
       << "planning          = " << planning_seconds << " s\n"
       << "sj fill           = " << sj_fill_seconds << " s\n"
       << "booster fill      = " << booster_fill_seconds << " s\n"
       << "output allocate   = " << output_allocate_seconds << " s\n"
       << "contraction       = " << contraction_seconds << " s\n"
       << "canonical scatter = " << canonical_scatter_seconds << " s\n"
       << "total             = " << total_seconds << " s\n";
}

FullTensorAutoOptions::FullTensorAutoOptions()
{
    // The low-level structs historically defaulted to a tuning/debug mode.
    // The public dispatcher should default to quiet production behavior.
    triangular_options.print_cache_info = false;
    schedule_options.print_schedule_summary = false;
    progress_options.enabled = true;
    progress_options.label = "Full tensor resident";
    progress_options.unit_label = "sched-work";

    // The weighted scheduler completes chunks in bursts. Build a
    // cumulative completion prediction only on new work, then smooth its
    // deadline so the live ETA counts down between chunk completions.
    progress_options.eta_min_elapsed_seconds = 2.0;
    progress_options.eta_min_fraction = 0.01;
    progress_options.eta_max_display_seconds = 30.0 * 24.0 * 3600.0;

    // Density counters are useful when designing kernels, but they are not
    // part of the value computation.  Keep the default fast/silent here.
    contraction_options.count_seed_nnz = false;
    contraction_options.count_stack_nnz = false;
    contraction_options.count_CT2_nnz = false;
}

void FullTensorAutoResult::print_summary(std::ostream& os) const
{
    os << "Full-tensor auto result\n"
       << "=======================\n"
       << "output mode          = " << full_tensor_output_mode_name(output_mode) << "\n"
       << "planner mode         = " << full_tensor_planner_mode_name(execution_plan.mode) << "\n"
       << "value checksum       = " << std::scientific << std::setprecision(17) << value_checksum << "\n"
       << "boundary intertwiner tiles          = " << boundary_tiles_completed << "\n"
       << "booster leg builds   = " << booster_leg_builds << "\n"
       << "sj slab builds       = " << sj_slab_builds << "\n"
       << "sj common cache      = " << sj_common_cache_slabs << " slabs, "
       << sj_common_cache_bytes << " bytes\n"
       << "sj common cache hits = " << sj_common_cache_hits << "\n"
       << "resident backend     = " << (ran_resident_backend ? "run" : "not run") << "\n"
       << "cache reused         = " << (reused_existing_disk_output ? "yes" : "no") << "\n"
       << "cache path           = " << (reused_existing_disk_path.empty() ? std::string("<none>") : reused_existing_disk_path.string()) << "\n"
       << "cache layout         = " << (reused_existing_disk_layout.empty() ? std::string("<none>") : reused_existing_disk_layout) << "\n"
       << "reassociated values  = " << reassociated_output.size() << "\n"
       << "reassoc disk root    = " << (reassociated_disk_root.empty() ? std::string("<none>") : reassociated_disk_root.string()) << "\n"
       << "canonical values     = " << canonical_output.size() << "\n"
       << "canonical disk root  = " << (canonical_disk_root.empty() ? std::string("<none>") : canonical_disk_root.string()) << "\n"
       << "canonical dims       = " << canonical_dims[0] << " x " << canonical_dims[1]
       << " x " << canonical_dims[2] << " x " << canonical_dims[3]
       << " x " << canonical_dims[4] << "\n";
}

void FullTensorScratchFileCleanup::remove_now()
{
    if (file_.empty()) return;
    std::error_code error;
    std::filesystem::remove(file_, error);
    if (error) {
        throw std::filesystem::filesystem_error(
            "cannot remove reassociated full-tensor scratch file",
            file_,
            error);
    }
    file_.clear();
}

const char* full_tensor_planner_mode_name(FullTensorPlannerMode mode) noexcept
{
    switch (mode) {
        case FullTensorPlannerMode::ResidentFast:
            return "ResidentFast";
        case FullTensorPlannerMode::ResidentFastNoCanonicalScatter:
            return "ResidentFastNoCanonicalScatter";
        case FullTensorPlannerMode::NeedsSjStreaming:
            return "NeedsSjStreaming";
        case FullTensorPlannerMode::NeedsBoosterStreamingOrImplicitSupport:
            return "NeedsBoosterStreamingOrImplicitSupport";
        case FullTensorPlannerMode::BudgetBelowFixedCost:
            return "BudgetBelowFixedCost";
        case FullTensorPlannerMode::RefuseOutputTooLarge:
            return "RefuseOutputTooLarge";
    }
    return "Unknown";
}

std::uint64_t full_tensor_planner_physical_ram_bytes() noexcept
{
#if defined(__APPLE__)
    std::uint64_t mem = 0;
    std::size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) return mem;
    return 0;
#elif defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) return static_cast<std::uint64_t>(status.ullTotalPhys);
    return 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || page_size <= 0) return 0;
    return static_cast<std::uint64_t>(pages) * static_cast<std::uint64_t>(page_size);
#endif
}

std::uint64_t full_tensor_planner_memory_budget_bytes(const FullTensorPlannerOptions& opt,
                                                             std::uint64_t physical_ram) noexcept
{
    if (opt.memory_budget_bytes != 0) return opt.memory_budget_bytes;
    const host_memory::Snapshot snapshot = host_memory::detect_snapshot();
    if (host_memory::has_any_probe(snapshot)) {
        return host_memory::automatic_budget_bytes(snapshot, opt.ram_fraction);
    }
    if (physical_ram != 0) {
        const long double b = static_cast<long double>(physical_ram) *
                              static_cast<long double>(opt.ram_fraction);
        const long double maxv = static_cast<long double>(std::numeric_limits<std::uint64_t>::max());
        return b >= maxv ? std::numeric_limits<std::uint64_t>::max()
                         : static_cast<std::uint64_t>(b);
    }
    return 8ull * 1024ull * 1024ull * 1024ull;
}

FullTensorPreflightOutputPlan
preflight_full_tensor_output_from_channels(const FullTensorChannels& ch,
                                           const FullTensorPlannerOptions& opt)
{
    FullTensorPreflightOutputPlan p;
    p.I1 = ch.i1.size();
    p.I2 = ch.i2.size();
    p.I3 = ch.i3.size();
    p.I4 = ch.i4.size();
    p.I5 = ch.i5.size();

    p.canonical_values = full_tensor_planner_output_values_from_channels(
        ch, &p.canonical_output_overflows_u64);
    p.canonical_output_bytes = full_tensor_planner_sat_mul_u64(
        p.canonical_values, static_cast<std::uint64_t>(sizeof(double)),
        &p.canonical_output_overflows_u64);

    p.canonical_output_exceeds_guard =
        p.canonical_output_overflows_u64 ||
        p.canonical_output_bytes > opt.max_canonical_output_bytes;

    return p;
}

FullTensorExecutionPlan
make_full_tensor_execution_plan_from_support(const FullTensorInputSupportPlan& support,
                                             const FullTensorPlannerOptions& opt)
{
    FullTensorExecutionPlan p;

    p.physical_ram_bytes = full_tensor_planner_physical_ram_bytes();
    p.memory_budget_bytes = full_tensor_planner_memory_budget_bytes(opt, p.physical_ram_bytes);
    p.safety_margin_bytes = opt.safety_margin_bytes != 0
        ? opt.safety_margin_bytes
        : full_tensor_planner_default_safety_margin_bytes([&]() {
              const host_memory::Snapshot snapshot = host_memory::detect_snapshot();
              const std::uint64_t hard = host_memory::hard_capacity_bytes(snapshot);
              return hard != 0 ? hard : p.physical_ram_bytes;
          }());

    p.external_fixed_bytes = 0;
    p.external_fixed_bytes = full_tensor_planner_sat_add_u64(p.external_fixed_bytes, opt.extra_fixed_bytes);
    p.external_fixed_bytes = full_tensor_planner_sat_add_u64(p.external_fixed_bytes, p.safety_margin_bytes);
    p.budget_below_fixed_cost = p.external_fixed_bytes > p.memory_budget_bytes;

    p.sj_payload_bytes = full_tensor_support_sj_payload_bytes(support);
    p.booster_payload_bytes = full_tensor_support_booster_payload_bytes(support);
    p.support_metadata_bytes = support.total_metadata_bytes();
    p.support_total_bytes = full_tensor_planner_sat_add_u64(support.total_payload_bytes(),
                                                            p.support_metadata_bytes);

    p.output_plan = make_full_tensor_reassociated_auto_output_plan(support);
    p.reassociated_values = static_cast<std::uint64_t>(p.output_plan.values_count);
    p.reassociated_output_bytes = full_tensor_planner_sat_mul_u64(
        p.reassociated_values, static_cast<std::uint64_t>(sizeof(double)));

    p.canonical_values = full_tensor_planner_output_values_from_channels(
        support.channels, &p.canonical_output_overflows_u64);
    p.canonical_output_bytes = full_tensor_planner_sat_mul_u64(
        p.canonical_values, static_cast<std::uint64_t>(sizeof(double)),
        &p.canonical_output_overflows_u64);
    p.canonical_output_exceeds_guard =
        p.canonical_output_overflows_u64 ||
        p.canonical_output_bytes > opt.max_canonical_output_bytes;
    p.contraction_scratch_bytes = opt.contraction_scratch_bytes;
    p.cpu_batched_workspace_bytes = opt.cpu_batched_workspace_bytes;
    const std::uint64_t total_contraction_scratch_bytes =
        full_tensor_planner_total_contraction_scratch_bytes(opt);

    std::uint64_t output_peak = 0;
    if (opt.include_reassociated_output) {
        output_peak = full_tensor_planner_sat_add_u64(output_peak, p.reassociated_output_bytes);
    }
    if (opt.include_canonical_output_in_peak) {
        output_peak = full_tensor_planner_sat_add_u64(output_peak, p.canonical_output_bytes);
    }

    p.resident_fast_peak_bytes = p.external_fixed_bytes;
    p.resident_fast_peak_bytes = full_tensor_planner_sat_add_u64(p.resident_fast_peak_bytes,
                                                                 p.support_metadata_bytes);
    p.resident_fast_peak_bytes = full_tensor_planner_sat_add_u64(
        p.resident_fast_peak_bytes, p.sj_payload_bytes);
    p.resident_fast_peak_bytes = full_tensor_planner_sat_add_u64(p.resident_fast_peak_bytes,
                                                                 p.booster_payload_bytes);
    p.resident_fast_peak_bytes = full_tensor_planner_sat_add_u64(p.resident_fast_peak_bytes,
                                                                 output_peak);
    p.resident_fast_peak_bytes = full_tensor_planner_sat_add_u64(p.resident_fast_peak_bytes,
                                                                 total_contraction_scratch_bytes);

    p.resident_boosters_only_peak_bytes = p.external_fixed_bytes;
    p.resident_boosters_only_peak_bytes = full_tensor_planner_sat_add_u64(
        p.resident_boosters_only_peak_bytes, p.support_metadata_bytes);
    p.resident_boosters_only_peak_bytes = full_tensor_planner_sat_add_u64(
        p.resident_boosters_only_peak_bytes, p.booster_payload_bytes);
    p.resident_boosters_only_peak_bytes = full_tensor_planner_sat_add_u64(
        p.resident_boosters_only_peak_bytes, output_peak);
    p.resident_boosters_only_peak_bytes = full_tensor_planner_sat_add_u64(
        p.resident_boosters_only_peak_bytes, total_contraction_scratch_bytes);

    p.resident_fast_fits = p.resident_fast_peak_bytes <= p.memory_budget_bytes;
    p.resident_boosters_fit = p.resident_boosters_only_peak_bytes <= p.memory_budget_bytes;

    if (opt.refuse_large_canonical_output && p.canonical_output_exceeds_guard) {
        p.mode = FullTensorPlannerMode::RefuseOutputTooLarge;
    } else if (p.budget_below_fixed_cost) {
        p.mode = FullTensorPlannerMode::BudgetBelowFixedCost;
    } else if (p.resident_fast_fits) {
        p.mode = opt.include_canonical_output_in_peak
            ? FullTensorPlannerMode::ResidentFast
            : FullTensorPlannerMode::ResidentFastNoCanonicalScatter;
    } else if (p.resident_boosters_fit) {
        p.mode = FullTensorPlannerMode::NeedsSjStreaming;
    } else {
        p.mode = FullTensorPlannerMode::NeedsBoosterStreamingOrImplicitSupport;
    }

    return p;
}

void full_tensor_planner_print_bytes(std::ostream& os, std::uint64_t bytes)
{
    constexpr long double KiB = 1024.0L;
    constexpr long double MiB = 1024.0L * KiB;
    constexpr long double GiB = 1024.0L * MiB;
    constexpr long double TiB = 1024.0L * GiB;

    const long double b = static_cast<long double>(bytes);
    os << std::fixed << std::setprecision(4);
    if (b >= TiB) os << static_cast<double>(b / TiB) << " TiB";
    else if (b >= GiB) os << static_cast<double>(b / GiB) << " GiB";
    else if (b >= MiB) os << static_cast<double>(b / MiB) << " MiB";
    else if (b >= KiB) os << static_cast<double>(b / KiB) << " KiB";
    else os << bytes << " B";
}

void FullTensorExecutionPlan::print(std::ostream& os) const
{
    os << "Full-tensor execution plan\n"
       << "==========================\n"
       << "mode                  = " << full_tensor_planner_mode_name(mode) << "\n"
       << "physical RAM          = ";
    full_tensor_planner_print_bytes(os, physical_ram_bytes); os << "\n";
    os << "memory budget         = ";
    full_tensor_planner_print_bytes(os, memory_budget_bytes); os << "\n";
    os << "external+safety       = ";
    full_tensor_planner_print_bytes(os, external_fixed_bytes);
    os << (budget_below_fixed_cost ? "  [exceeds budget]" : "") << "\n";
    os << "support metadata      = ";
    full_tensor_planner_print_bytes(os, support_metadata_bytes); os << "\n";
    os << "sj payload            = ";
    full_tensor_planner_print_bytes(os, sj_payload_bytes);
    os << "  [included in host budget]\n";
    os << "booster payload       = ";
    full_tensor_planner_print_bytes(os, booster_payload_bytes); os << "\n";
    os << "contraction scratch   = ";
    full_tensor_planner_print_bytes(os, contraction_scratch_bytes); os << "\n";
    os << "CPU batched workspace = ";
    full_tensor_planner_print_bytes(os, cpu_batched_workspace_bytes); os << "\n";
    os << "reassoc output        = ";
    full_tensor_planner_print_bytes(os, reassociated_output_bytes); os << "\n";
    os << "canonical output      = ";
    full_tensor_planner_print_bytes(os, canonical_output_bytes);
    if (canonical_output_overflows_u64) os << " (saturated/overflow)";
    os << "\n";
    os << "canonical guard       = "
       << (canonical_output_exceeds_guard ? "EXCEEDED" : "ok") << "\n";
    os << "resident fast peak    = ";
    full_tensor_planner_print_bytes(os, resident_fast_peak_bytes);
    os << "  [" << (resident_fast_fits ? "fits" : "does not fit") << "]\n";
    os << "resident B-only peak  = ";
    full_tensor_planner_print_bytes(os, resident_boosters_only_peak_bytes);
    os << "  [" << (resident_boosters_fit ? "fits" : "does not fit") << "]\n";
    os << "output layout         = "
       << full_tensor_reassociated_auto_layout_name(output_plan.layout) << "\n"
       << "output values_count   = " << output_plan.values_count << "\n";
}

double full_tensor_value_checksum(const double* values, std::size_t count) noexcept
{
    // A deterministic, order-sensitive linear checksum suitable for benchmark
    // comparisons.  It is not a replacement for elementwise correctness tests,
    // but catches layout/tile omissions while avoiding a second output buffer.
    long double sum = 0.0L;
    long double correction = 0.0L;
    for (std::size_t i = 0; i < count; ++i) {
        const long double weight = 1.0L + static_cast<long double>(i % 1021u) / 1021.0L;
        const long double term = static_cast<long double>(values[i]) * weight;
        const long double y = term - correction;
        const long double t = sum + y;
        correction = (t - sum) - y;
        sum = t;
    }
    return static_cast<double>(sum);
}

std::size_t full_tensor_choose_next_x_slab_end(
    const B4Spins& spins,
    int Dl,
    const FullTensorChannels& full_channels,
    const FullTensorExecutionPlan& plan,
    const FullTensorAutoOptions& options,
    std::size_t x_begin,
    FullTensorInputSupportPlan& chosen_support,
    std::uint64_t* chosen_peak_bytes)
{
    const std::size_t X = full_channels.xvals.size();
    if (x_begin >= X) {
        chosen_support = FullTensorInputSupportPlan{};
        if (chosen_peak_bytes) *chosen_peak_bytes = 0;
        return x_begin;
    }

    std::size_t max_count = X - x_begin;
    if (options.streamed_x_per_slab != 0) {
        max_count = std::min<std::size_t>(max_count, options.streamed_x_per_slab);
    }

    // Try large slabs first, then shrink by halves.  This keeps normal streamed
    // cases efficient while still finding a one-x fallback under an artificial
    // tiny budget.  If even one x does not fit, we still return it and the caller
    // decides whether to throw; that gives a precise diagnostic instead of a
    // mysterious allocation failure.
    std::size_t count = std::max<std::size_t>(1, max_count);
    FullTensorInputSupportPlan best;
    std::uint64_t best_peak = std::numeric_limits<std::uint64_t>::max();

    while (count >= 1) {
        FullTensorChannels slab_ch = full_tensor_channels_with_x_slice(full_channels, x_begin, x_begin + count);
        FullTensorInputSupportPlan slab_support = build_full_tensor_input_support_plan(
            spins, Dl, slab_ch);
        const std::uint64_t peak = full_tensor_estimate_sj_streamed_peak_bytes(
            plan, slab_support, options);

        if (peak <= plan.memory_budget_bytes || count == 1) {
            best = std::move(slab_support);
            best_peak = peak;
            break;
        }
        count = (count + 1) / 2;
        if (count == max_count) --count;
    }

    chosen_support = std::move(best);
    if (chosen_peak_bytes) *chosen_peak_bytes = best_peak;
    return std::min<std::size_t>(X, x_begin + std::max<std::size_t>(1, count));
}

void full_tensor_add_local_canonical_tile_to_global(
    const std::vector<double>& local,
    Tensor<double, 5>& global,
    const std::array<std::size_t,5>& global_dims,
    std::size_t i1_begin,
    std::size_t i2_begin,
    std::size_t i3_begin,
    std::size_t i4_begin,
    std::size_t i5_begin,
    std::size_t I1t,
    std::size_t I2t,
    std::size_t I3t,
    std::size_t I4t,
    std::size_t I5t)
{
    const std::size_t GI1 = global_dims[0];
    const std::size_t GI2 = global_dims[1];
    const std::size_t GI3 = global_dims[2];
    const std::size_t GI4 = global_dims[3];

    for (std::size_t ii5 = 0; ii5 < I5t; ++ii5) {
        for (std::size_t ii4 = 0; ii4 < I4t; ++ii4) {
            for (std::size_t ii3 = 0; ii3 < I3t; ++ii3) {
                for (std::size_t ii2 = 0; ii2 < I2t; ++ii2) {
                    for (std::size_t ii1 = 0; ii1 < I1t; ++ii1) {
                        const std::size_t src = full_tensor_canonical_flat_index(
                            ii1, ii2, ii3, ii4, ii5, I1t, I2t, I3t, I4t);
                        const std::size_t dst = full_tensor_canonical_flat_index(
                            i1_begin + ii1,
                            i2_begin + ii2,
                            i3_begin + ii3,
                            i4_begin + ii4,
                            i5_begin + ii5,
                            GI1, GI2, GI3, GI4);
                        global[dst] += local[src];
                    }
                }
            }
        }
    }
}

FullTensorReassociatedAutoOutputPlan
full_tensor_expected_reassociated_output_plan_from_channels(
    const std::array<int, 10>& spins,
    const FullTensorChannels& channels)
{
    FullTensorReassociatedAutoOutputPlan plan;
    plan.I1 = channels.i1.size();
    plan.M = channels.i2.size() * channels.i3.size();
    plan.N = channels.i4.size() * channels.i5.size();

    const bool reflection_channels_match =
        channels.i2 == channels.i5 && channels.i3 == channels.i4;
    plan.reflection_compatible = reflection_channels_match;
    plan.reverse_column_compatible = reflection_channels_match;
    if (reflection_channels_match) {
        plan.compact_representatives =
            full_tensor_triangular_upper_size(plan.M);
    }

    const bool use_reflection_compact =
        full_tensor_reflection_15432_preserves_spins(spins) &&
        reflection_channels_match && plan.M == plan.N;
    if (use_reflection_compact) {
        plan.layout =
            FullTensorReassociatedAutoOutputLayout::ReflectionCompactTriangular;
        plan.values_count = plan.I1 * plan.compact_representatives;
    } else {
        plan.layout = FullTensorReassociatedAutoOutputLayout::RectangularFull;
        plan.values_count = plan.I1 * plan.M * plan.N;
    }
    return plan;
}

FullTensorExistingDiskCheck full_tensor_check_existing_disk_file(
    const std::filesystem::path& file,
    tensor_disk::LayoutCode expected_layout,
    std::uint32_t expected_ndim,
    const std::array<int, 10>& expected_spins,
    std::size_t expected_Dl,
    double expected_gamma,
    const std::array<std::size_t, 5>& expected_canonical_dims,
    std::uint64_t expected_values_count_or_zero,
    sl2cfoamxx::SimplicityConstraint expected_simplicity)
{
    FullTensorExistingDiskCheck out{};
    out.exists = std::filesystem::exists(file);
    if (!out.exists) {
        out.reason = "file does not exist";
        return out;
    }
    if (!std::filesystem::is_regular_file(file)) {
        out.reason = "path exists but is not a regular file";
        return out;
    }

    try {
        out.header = tensor_disk::read_header_file(file);
    } catch (const std::exception& e) {
        out.reason = std::string("cannot read tensor header: ") + e.what();
        return out;
    }

    if (out.header.ndim != expected_ndim) {
        out.reason = "rank mismatch in tensor header";
        return out;
    }
    if (out.header.scalar_code != static_cast<std::uint32_t>(tensor_disk::ScalarCodeOf<double>::value) ||
        out.header.sizeof_scalar != sizeof(double)) {
        out.reason = "scalar type mismatch in tensor header";
        return out;
    }
    if (out.header.layout_code != static_cast<std::uint32_t>(expected_layout)) {
        out.reason = "layout code mismatch in tensor header";
        return out;
    }

    if (expected_ndim == 1) {
        if (expected_values_count_or_zero != 0 && out.header.values_count != expected_values_count_or_zero) {
            out.reason = "value count mismatch in tensor header";
            return out;
        }
    } else if (expected_ndim == 5) {
        for (std::size_t i = 0; i < 5; ++i) {
            if (out.header.dims[i] != static_cast<std::uint64_t>(expected_canonical_dims[i])) {
                out.reason = "canonical dimension mismatch in tensor header";
                return out;
            }
        }
        if (expected_values_count_or_zero != 0 && out.header.values_count != expected_values_count_or_zero) {
            out.reason = "canonical value count mismatch in tensor header";
            return out;
        }
    }

    const std::uint64_t payload_offset = out.header.reserved_u64[0] != 0
        ? out.header.reserved_u64[0]
        : tensor_disk::default_single_file_payload_offset();
    const std::uint64_t expected_file_bytes = full_tensor_planner_sat_add_u64(payload_offset, out.header.payload_bytes);
    try {
        const std::uint64_t actual_file_bytes = static_cast<std::uint64_t>(std::filesystem::file_size(file));
        if (actual_file_bytes < expected_file_bytes) {
            out.reason = "file is shorter than header payload size";
            return out;
        }
    } catch (const std::exception& e) {
        out.reason = std::string("cannot stat tensor file: ") + e.what();
        return out;
    }

    if (!read_full_tensor_binary_tag(out.header, out.tag)) {
        out.reason = "full-tensor metadata tag is missing or invalid";
        return out;
    }
    if (out.tag.layout_code != static_cast<std::uint32_t>(expected_layout)) {
        out.reason = "layout code mismatch in full-tensor tag";
        return out;
    }
    if (out.tag.Dl != static_cast<std::uint64_t>(expected_Dl)) {
        out.reason = "Dl mismatch in full-tensor tag";
        return out;
    }
    if (!full_tensor_gamma_matches(out.tag.gamma, expected_gamma)) {
        out.reason = "gamma mismatch in full-tensor tag";
        return out;
    }
    if (!full_tensor_simplicity_constraint_matches(
            out.tag, expected_simplicity)) {
        out.reason = "simplicity-constraint mismatch in full-tensor tag";
        return out;
    }
    for (std::size_t i = 0; i < 10; ++i) {
        if (out.tag.spins[i] != static_cast<std::int32_t>(expected_spins[i])) {
            out.reason = "doubled-spin mismatch in full-tensor tag";
            return out;
        }
    }
    for (std::size_t i = 0; i < 5; ++i) {
        if (out.tag.canonical_dims[i] != static_cast<std::uint64_t>(expected_canonical_dims[i])) {
            out.reason = "canonical dimension mismatch in full-tensor tag";
            return out;
        }
    }
    if (out.tag.values_count != out.header.values_count) {
        out.reason = "tag/header value-count mismatch";
        return out;
    }
    if (expected_values_count_or_zero != 0 && out.tag.values_count != expected_values_count_or_zero) {
        out.reason = "expected value-count mismatch in full-tensor tag";
        return out;
    }

    if (expected_layout == tensor_disk::LayoutCode::FullTensorReassociatedReflectionCompactByI1 &&
        full_tensor_reassociated_storage_order(out.tag) == FullTensorReassociatedStorageOrder::Unknown)
    {
        out.reason = "missing reassociated storage-order tag; delete/regenerate this old cache file";
        return out;
    }

    out.reusable = true;
    out.reason = "metadata matches requested tensor";
    return out;
}

FullTensorExistingDiskCheck full_tensor_check_existing_reassociated_file(
    const std::filesystem::path& file,
    const std::array<int, 10>& spins,
    std::size_t Dl,
    double gamma,
    const std::array<std::size_t, 5>& canonical_dims,
    const FullTensorReassociatedAutoOutputPlan& expected_output_plan,
    sl2cfoamxx::SimplicityConstraint simplicity)
{
    const std::uint64_t expected_values_count =
        static_cast<std::uint64_t>(expected_output_plan.values_count);
    FullTensorExistingDiskCheck out =
        full_tensor_check_existing_reassociated_file(
            file,
            spins,
            Dl,
            gamma,
            canonical_dims,
            expected_values_count,
            simplicity);
    if (!out.reusable) return out;

    const FullTensorReassociatedStorageOrder expected_order =
        full_tensor_reassociated_storage_order_for_output_plan(
            expected_output_plan);
    const FullTensorReassociatedStorageOrder actual_order =
        full_tensor_reassociated_storage_order(out.tag);
    if (actual_order != expected_order) {
        out.reusable = false;
        out.reason =
            "reassociated storage-order mismatch in full-tensor tag";
    }
    return out;
}

bool full_tensor_should_use_disk_reassociated_buffer(
    const FullTensorExecutionPlan& plan,
    const FullTensorAutoOptions& options,
    std::uint64_t backend_peak_without_reassoc) noexcept
{
    switch (options.output_mode) {
        case FullTensorOutputMode::ReassociatedOnDisk:
            return true;
        case FullTensorOutputMode::CanonicalDenseOnDisk:
            return full_tensor_planner_sat_add_u64(
                       backend_peak_without_reassoc, plan.reassociated_output_bytes) > plan.memory_budget_bytes;
        case FullTensorOutputMode::BenchmarkOnly:
            return full_tensor_planner_sat_add_u64(
                       backend_peak_without_reassoc, plan.reassociated_output_bytes) > plan.memory_budget_bytes;
        case FullTensorOutputMode::ReassociatedAuto:
            return options.save_reassociated_auto_to_disk ||
                   full_tensor_planner_sat_add_u64(
                       backend_peak_without_reassoc, plan.reassociated_output_bytes) > plan.memory_budget_bytes;
        case FullTensorOutputMode::ReassociatedOnly:
        case FullTensorOutputMode::CanonicalDenseInMemory:
            return false;
    }
    return false;
}

void full_tensor_add_local_canonical_tile_to_global_ptr(
    const std::vector<double>& local,
    double* global,
    const std::array<std::size_t,5>& global_dims,
    std::size_t i1_begin,
    std::size_t i2_begin,
    std::size_t i3_begin,
    std::size_t i4_begin,
    std::size_t i5_begin,
    std::size_t I1t,
    std::size_t I2t,
    std::size_t I3t,
    std::size_t I4t,
    std::size_t I5t)
{
    if (global == nullptr) return;
    const std::size_t GI1 = global_dims[0];
    const std::size_t GI2 = global_dims[1];
    const std::size_t GI3 = global_dims[2];
    const std::size_t GI4 = global_dims[3];

    for (std::size_t ii5 = 0; ii5 < I5t; ++ii5) {
        for (std::size_t ii4 = 0; ii4 < I4t; ++ii4) {
            for (std::size_t ii3 = 0; ii3 < I3t; ++ii3) {
                for (std::size_t ii2 = 0; ii2 < I2t; ++ii2) {
                    for (std::size_t ii1 = 0; ii1 < I1t; ++ii1) {
                        const std::size_t src = full_tensor_canonical_flat_index(
                            ii1, ii2, ii3, ii4, ii5, I1t, I2t, I3t, I4t);
                        const std::size_t dst = full_tensor_canonical_flat_index(
                            i1_begin + ii1,
                            i2_begin + ii2,
                            i3_begin + ii3,
                            i4_begin + ii4,
                            i5_begin + ii5,
                            GI1, GI2, GI3, GI4);
                        global[dst] += local[src];
                    }
                }
            }
        }
    }
}

FullTensorReflectionCompactWeightedProgressOptions
full_tensor_auto_progress_options_for_backend(
    const FullTensorAutoOptions& options,
    const char* label,
    const char* unit_label,
    std::uint64_t update_granularity)
{
    FullTensorReflectionCompactWeightedProgressOptions out = options.progress_options;
    if (label != nullptr && *label != '\0') {
        out.label = label;
    }
    if (unit_label != nullptr && *unit_label != '\0') {
        out.unit_label = unit_label;
    }
    if (update_granularity != 0) {
        out.update_granularity = update_granularity;
    }
    out.phase = sl2cfoamxx::ProgressPhase::contraction;
    return out;
}

std::uint64_t full_tensor_fully_streamed_output_peak_bytes_for_mode(
    const FullTensorExecutionPlan& plan,
    const FullTensorAutoOptions& options) noexcept
{
    switch (options.output_mode) {
        case FullTensorOutputMode::BenchmarkOnly:
        case FullTensorOutputMode::ReassociatedAuto:
        case FullTensorOutputMode::ReassociatedOnDisk:
        case FullTensorOutputMode::CanonicalDenseOnDisk:
            return 0;
        case FullTensorOutputMode::CanonicalDenseInMemory:
            return plan.canonical_output_bytes;
        case FullTensorOutputMode::ReassociatedOnly:
            return plan.reassociated_output_bytes;
    }
    return 0;
}

std::uint64_t full_tensor_estimate_fully_streamed_tile_peak_bytes(
    const FullTensorExecutionPlan& plan,
    const FullTensorInputSupportPlan& tile_fullx_support,
    const FullTensorInputSupportPlan& tile_x_support,
    const FullTensorReassociatedAutoOutputPlan& tile_output_plan,
    const FullTensorAutoOptions& options) noexcept
{
    std::uint64_t peak = plan.external_fixed_bytes;
    // The current dispatcher still materializes the global support before this
    // backend is selected.  Fully implicit support is the next redesign step;
    // until then, include it explicitly in the peak-memory estimate.
    peak = full_tensor_planner_sat_add_u64(peak, plan.support_metadata_bytes);
    peak = full_tensor_planner_sat_add_u64(peak, tile_fullx_support.total_metadata_bytes());
    peak = full_tensor_planner_sat_add_u64(peak, tile_x_support.total_metadata_bytes());
    peak = full_tensor_planner_sat_add_u64(peak, full_tensor_support_booster_payload_bytes(tile_fullx_support));
    // Direct leg-local generation keeps earlier packed panels alive while one
    // canonical rank-5 leg is harvested.  Account for the largest such dense
    // transient; the previous estimate omitted all four canonical tensors and
    // could therefore select a panel which was impossible in practice.
    const auto dense_leg_bytes = [](std::initializer_list<std::uint64_t> dims) noexcept {
        std::uint64_t values = 1;
        for (std::uint64_t d : dims) values = full_tensor_planner_sat_mul_u64(values, d);
        return full_tensor_planner_sat_mul_u64(values, sizeof(double));
    };
    const auto& ch = tile_fullx_support.channels;
    const std::uint64_t dense_B2 = dense_leg_bytes({
        tile_fullx_support.l23.size(), tile_fullx_support.l24.size(), tile_fullx_support.l25.size(),
        ch.i2.size(), ch.k2.size()});
    const std::uint64_t dense_B3 = dense_leg_bytes({
        tile_fullx_support.l34.size(), tile_fullx_support.l35.size(), tile_fullx_support.l23.size(),
        ch.i3.size(), ch.k3.size()});
    const std::uint64_t dense_B4 = dense_leg_bytes({
        tile_fullx_support.l45.size(), tile_fullx_support.l24.size(), tile_fullx_support.l34.size(),
        ch.i4.size(), ch.k4.size()});
    const std::uint64_t dense_B5 = dense_leg_bytes({
        tile_fullx_support.l25.size(), tile_fullx_support.l35.size(), tile_fullx_support.l45.size(),
        ch.i5.size(), ch.k5.size()});
    peak = full_tensor_planner_sat_add_u64(
        peak, std::max({dense_B2, dense_B3, dense_B4, dense_B5}));
    peak = full_tensor_planner_sat_add_u64(peak, full_tensor_support_sj_payload_bytes(tile_x_support));
    peak = full_tensor_planner_sat_add_u64(
        peak,
        static_cast<std::uint64_t>(tile_output_plan.values_count) * static_cast<std::uint64_t>(sizeof(double)));
    peak = full_tensor_planner_sat_add_u64(
        peak, full_tensor_fully_streamed_output_peak_bytes_for_mode(plan, options));
    peak = full_tensor_planner_sat_add_u64(
        peak, full_tensor_planner_total_contraction_scratch_bytes(options.planner));
    return peak;
}

std::uint64_t full_tensor_estimate_fully_streamed_peak_for_panel(
    const B4Spins& spins,
    int Dl,
    const FullTensorChannels& FTC,
    const FullTensorExecutionPlan& plan,
    const FullTensorAutoOptions& options,
    std::size_t panel_size)
{
    const std::size_t I1 = FTC.i1.size();
    const std::size_t I2 = FTC.i2.size();
    const std::size_t I3 = FTC.i3.size();
    const std::size_t I4 = FTC.i4.size();
    const std::size_t I5 = FTC.i5.size();
    const std::size_t X  = FTC.xvals.size();
    const std::size_t P = std::max<std::size_t>(1, panel_size);

    FullTensorChannels tile_ch = full_tensor_channels_with_boundary_slices(
        FTC,
        0, std::min<std::size_t>(I1, P),
        0, std::min<std::size_t>(I2, P),
        0, std::min<std::size_t>(I3, P),
        0, std::min<std::size_t>(I4, P),
        0, std::min<std::size_t>(I5, P));

    FullTensorInputSupportPlan tile_fullx_support = build_full_tensor_input_support_plan(
        spins, Dl, tile_ch);

    const std::size_t x_count = options.streamed_x_per_slab == 0
        ? 1
        : std::max<std::size_t>(1, options.streamed_x_per_slab);
    FullTensorChannels tile_x_ch = tile_ch;
    tile_x_ch.xvals = full_tensor_slice_axis(FTC.xvals, 0, std::min<std::size_t>(X, x_count));
    FullTensorInputSupportPlan tile_x_support = build_full_tensor_input_support_plan(
        spins, Dl, tile_x_ch);

    const FullTensorReassociatedAutoOutputPlan tile_output_plan =
        make_full_tensor_reassociated_auto_output_plan(tile_fullx_support);

    return full_tensor_estimate_fully_streamed_tile_peak_bytes(
        plan, tile_fullx_support, tile_x_support, tile_output_plan, options);
}

std::size_t full_tensor_choose_fully_streamed_boundary_panel_size(
    const B4Spins& spins,
    int Dl,
    const FullTensorChannels& FTC,
    const FullTensorExecutionPlan& plan,
    const FullTensorAutoOptions& options,
    std::uint64_t* chosen_peak_bytes)
{
    const std::size_t pmax = std::max<std::size_t>(1, std::min({
        FTC.i1.size(), FTC.i2.size(), FTC.i3.size(), FTC.i4.size(), FTC.i5.size()}));

    if (options.fully_streamed_boundary_panel_size != 0) {
        const std::size_t p = std::max<std::size_t>(1, std::min<std::size_t>(pmax, options.fully_streamed_boundary_panel_size));
        if (chosen_peak_bytes != nullptr) {
            *chosen_peak_bytes = full_tensor_estimate_fully_streamed_peak_for_panel(
                spins, Dl, FTC, plan, options, p);
        }
        return p;
    }

    std::size_t lo = 1;
    std::size_t hi = pmax;
    std::size_t best = 1;
    std::uint64_t best_peak = 0;

    while (lo <= hi) {
        const std::size_t mid = lo + (hi - lo) / 2;
        const std::uint64_t peak = full_tensor_estimate_fully_streamed_peak_for_panel(
            spins, Dl, FTC, plan, options, mid);
        if (peak <= plan.memory_budget_bytes) {
            best = mid;
            best_peak = peak;
            lo = mid + 1;
        } else {
            if (mid == 0) break;
            hi = mid - 1;
        }
    }

    if (best_peak == 0) {
        best_peak = full_tensor_estimate_fully_streamed_peak_for_panel(
            spins, Dl, FTC, plan, options, best);
    }
    if (chosen_peak_bytes != nullptr) {
        *chosen_peak_bytes = best_peak;
    }
    return best;
}

FullTensorAutoResult compute_full_tensor_resident_fast(
    const B4Spins& spins,
    double gamma,
    const std::vector<double>& FactTable,
    const FullTensorChannels& FTC,
    const FullTensorInputSupportPlan& support,
    const FullTensorExecutionPlan& execution_plan,
    const FullTensorAutoOptions& options)
{
    FullTensorAutoResult result;
    result.output_mode = options.output_mode;
    result.execution_plan = execution_plan;
    result.output_plan = execution_plan.output_plan;
    result.canonical_dims = {{FTC.i1.size(), FTC.i2.size(), FTC.i3.size(), FTC.i4.size(), FTC.i5.size()}};

    if (!options.allow_resident_fast) {
        full_tensor_throw_unavailable_backend(execution_plan);
    }

    if (execution_plan.mode != FullTensorPlannerMode::ResidentFast &&
        execution_plan.mode != FullTensorPlannerMode::ResidentFastNoCanonicalScatter) {
        full_tensor_throw_unavailable_backend(execution_plan);
    }

    PackedFullTensorSJTensors packed_full_sj;
    {
        FullTensorAutoScopedTimer timer(result.timings.sj_fill_seconds);
        fill_packed_full_tensor_sj_tensors_rec(spins, support, packed_full_sj);
    }

    PackedFullTensorBoosters packed_full_B;
    {
        FullTensorAutoScopedTimer timer(result.timings.booster_fill_seconds);
        const std::array<std::size_t, 6> Dim_ls{{
            support.l23.size(), support.l24.size(), support.l25.size(),
            support.l34.size(), support.l35.size(), support.l45.size()
        }};
        fill_packed_full_tensor_boosters_via_canonical(
            spins, Dim_ls, gamma, FactTable, support, packed_full_B,
            options.simplicity_constraint, options.booster_rho_tile);
    }

    const std::size_t output_size = full_tensor_checked_size_t_from_u64(
        execution_plan.reassociated_values, "reassociated full-tensor output size");

    const bool reassoc_on_disk = full_tensor_should_use_disk_reassociated_buffer(
        execution_plan, options, execution_plan.resident_fast_peak_bytes);
    const bool reassoc_is_canonical_scratch =
        reassoc_on_disk &&
        options.output_mode == FullTensorOutputMode::CanonicalDenseOnDisk;

    Tensor<double, 1> reassoc_memory;
    FullTensorScratchFileCleanup reassoc_scratch_cleanup;
    Tensor<double, 1> reassoc_disk;
    double* reassoc_data = nullptr;

    {
        FullTensorAutoScopedTimer timer(result.timings.output_allocate_seconds);
        if (reassoc_on_disk) {
            full_tensor_throw_if_disk_output_exceeds_guard(
                execution_plan.reassociated_output_bytes,
                options.max_reassociated_output_disk_bytes,
                "Reassociated full tensor");
            result.reassociated_disk_root = reassoc_is_canonical_scratch
                ? full_tensor_canonical_reassociated_scratch_root(
                      options,
                      full_tensor_spins_array_from_b4spins(spins),
                      full_tensor_Dl_from_support(support),
                      gamma)
                : full_tensor_reassociated_disk_root(
                      options,
                      full_tensor_spins_array_from_b4spins(spins),
                      full_tensor_Dl_from_support(support),
                      gamma);
            if (reassoc_is_canonical_scratch) {
                reassoc_scratch_cleanup.arm(result.reassociated_disk_root);
            }
            const auto tag = make_full_tensor_binary_tag_reassociated_with_storage_order(
                full_tensor_spins_array_from_b4spins(spins),
                (support.l23.empty() ? std::size_t{0} : support.l23.size() - 1),
                gamma,
                result.canonical_dims,
                execution_plan.reassociated_values,
                full_tensor_reassociated_storage_order_for_output_plan(
                    execution_plan.output_plan),
                options.simplicity_constraint);
            reassoc_disk = Tensor<double, 1>::create_file(
                result.reassociated_disk_root,
                std::array<std::size_t, 1>{{output_size}},
                // Fresh mapped files are logically zero after truncation;
                // avoid faulting and writing the complete output up front.
                TensorInit::Uninitialized,
                tensor_disk::LayoutCode::FullTensorReassociatedReflectionCompactByI1,
                tag);
            reassoc_data = reassoc_disk.data();
            result.reassociated_materialized_on_disk = true;
        } else {
            reassoc_memory = Tensor<double, 1>(
                {{output_size}},
                TensorInit::zero,
                TensorLayout::FullTensorReassociatedReflectionCompactByI1);
            reassoc_data = reassoc_memory.data();
            result.reassociated_materialized = true;
        }
    }

    // Both allocation paths above provide a fresh, logically-zero output.
    // Accumulate directly instead of performing a second whole-output clear.
    FullTensorPackedContractionL25ScalarOptions resident_copt =
        options.contraction_options;
    resident_copt.zero_output = false;

    {
        FullTensorAutoScopedTimer timer(result.timings.contraction_seconds);
        result.output_plan =
            contract_full_tensor_packed_fused_l25_stacked_scalar_profiled_reassociated_auto_scheduled_by_i1_layout(
                support, packed_full_sj, packed_full_B, reassoc_data,
                resident_copt, result.contraction_profile,
                options.triangular_options, options.schedule_options,
                full_tensor_auto_progress_options_for_backend(
                    options, "Full tensor resident", "sched-work",
                    options.progress_options.update_granularity),
                &result.schedule_balance);
    }
    result.boundary_tiles_completed = 1;

    result.ran_resident_backend = true;

    const bool canonical_output_requested =
        options.output_mode == FullTensorOutputMode::CanonicalDenseInMemory ||
        options.output_mode == FullTensorOutputMode::CanonicalDenseOnDisk;
    if (!canonical_output_requested) {
        result.value_checksum = full_tensor_value_checksum(
            reassoc_data, output_size);
    }

    if (canonical_output_requested) {
        if (execution_plan.canonical_output_exceeds_guard) {
            std::ostringstream oss;
            oss << "Refusing canonical dense scatter: output is "
                << full_tensor_format_bytes_string(execution_plan.canonical_output_bytes)
                << ", above the configured guard.";
            throw b4utils::ResourceLimitFailure(oss.str());
        }

        if (options.output_mode == FullTensorOutputMode::CanonicalDenseInMemory) {
            {
                FullTensorAutoScopedTimer timer(result.timings.output_allocate_seconds);
                (void)full_tensor_checked_size_t_from_u64(
                    execution_plan.canonical_values,
                    "canonical full-tensor output size");
                result.canonical_output = Tensor<double, 5>(
                    result.canonical_dims,
                    // Canonical scatter assigns every element exactly once.
                    TensorInit::Uninitialized,
                    TensorLayout::CanonicalDenseColumnMajor);
            }

            {
                FullTensorAutoScopedTimer timer(result.timings.canonical_scatter_seconds);
                scatter_full_tensor_reassociated_auto_by_i1_to_canonical(
                    support,
                    result.output_plan,
                    reassoc_data,
                    result.canonical_output.data());
            }
            result.canonical_materialized = true;
            result.value_checksum = full_tensor_value_checksum(
                result.canonical_output.data(), result.canonical_output.size());
        } else {
            full_tensor_throw_if_disk_output_exceeds_guard(
                execution_plan.canonical_output_bytes,
                options.max_canonical_output_disk_bytes,
                "Canonical dense full tensor");
            Tensor<double, 5> canonical_disk;
            {
                FullTensorAutoScopedTimer timer(result.timings.output_allocate_seconds);
                result.canonical_disk_root = full_tensor_canonical_disk_root(
                options,
                full_tensor_spins_array_from_b4spins(spins),
                full_tensor_Dl_from_support(support),
                gamma);
                const auto canonical_tag = make_full_tensor_binary_tag_canonical(
                    full_tensor_spins_array_from_b4spins(spins),
                    full_tensor_Dl_from_support(support),
                    gamma,
                    result.canonical_dims,
                    execution_plan.canonical_values,
                    options.simplicity_constraint);
                canonical_disk = Tensor<double, 5>::create_file(
                    result.canonical_disk_root,
                    result.canonical_dims,
                    // Canonical scatter assigns every element exactly once.
                    TensorInit::Uninitialized,
                    tensor_disk::LayoutCode::CanonicalDenseColumnMajor,
                    canonical_tag);
            }
            {
                FullTensorAutoScopedTimer timer(result.timings.canonical_scatter_seconds);
                scatter_full_tensor_reassociated_auto_by_i1_to_canonical(
                    support,
                    result.output_plan,
                    reassoc_data,
                    canonical_disk.data());
                result.value_checksum = full_tensor_value_checksum(
                    canonical_disk.data(),
                    full_tensor_checked_size_t_from_u64(
                        execution_plan.canonical_values,
                        "canonical full-tensor checksum size"));
                canonical_disk.flush();
            }
            result.canonical_materialized_on_disk = true;
        }
    }

    if (reassoc_on_disk) {
        reassoc_disk.flush();
        if (reassoc_scratch_cleanup.armed()) {
            reassoc_disk.clear();
            reassoc_data = nullptr;
            reassoc_scratch_cleanup.remove_now();
            result.reassociated_materialized_on_disk = false;
            result.reassociated_disk_root.clear();
        }
    } else {
        const bool keep_reassoc =
            (options.output_mode == FullTensorOutputMode::ReassociatedOnly) ||
            (options.output_mode == FullTensorOutputMode::ReassociatedAuto) ||
            (options.output_mode == FullTensorOutputMode::CanonicalDenseInMemory &&
             options.keep_reassociated_output);
        if (keep_reassoc && options.output_mode != FullTensorOutputMode::BenchmarkOnly) {
            result.reassociated_output = std::move(reassoc_memory);
        }
    }

    if (options.output_mode == FullTensorOutputMode::BenchmarkOnly) {
        result.reassociated_output.clear();
        result.canonical_output.clear();
        result.reassociated_materialized = false;
        result.canonical_materialized = false;
    }

    return result;
}

FullTensorAutoResult compute_full_tensor_sj_streamed(
    const B4Spins& spins,
    std::size_t Dl,
    double gamma,
    const std::vector<double>& FactTable,
    const FullTensorChannels& FTC,
    const FullTensorInputSupportPlan& full_support,
    const FullTensorExecutionPlan& execution_plan,
    const FullTensorAutoOptions& options)
{
    if (!options.allow_sj_streaming) {
        full_tensor_throw_unavailable_backend(execution_plan);
    }
    if (execution_plan.mode != FullTensorPlannerMode::NeedsSjStreaming &&
        execution_plan.mode != FullTensorPlannerMode::ResidentFast &&
        execution_plan.mode != FullTensorPlannerMode::ResidentFastNoCanonicalScatter) {
        full_tensor_throw_unavailable_backend(execution_plan);
    }

    FullTensorAutoResult result;
    result.output_mode = options.output_mode;
    result.execution_plan = execution_plan;
    result.output_plan = execution_plan.output_plan;
    result.canonical_dims = {{FTC.i1.size(), FTC.i2.size(), FTC.i3.size(), FTC.i4.size(), FTC.i5.size()}};

    PackedFullTensorBoosters packed_full_B;
    {
        FullTensorAutoScopedTimer timer(result.timings.booster_fill_seconds);
        const std::array<std::size_t, 6> Dim_ls{{
            full_support.l23.size(), full_support.l24.size(), full_support.l25.size(),
            full_support.l34.size(), full_support.l35.size(), full_support.l45.size()
        }};
        fill_packed_full_tensor_boosters_via_canonical(
            spins, Dim_ls, gamma, FactTable, full_support,
            packed_full_B, options.simplicity_constraint,
            options.booster_rho_tile);
    }

    const std::size_t output_size = full_tensor_checked_size_t_from_u64(
        execution_plan.reassociated_values, "reassociated full-tensor output size");

    const bool reassoc_on_disk = full_tensor_should_use_disk_reassociated_buffer(
        execution_plan, options, execution_plan.resident_boosters_only_peak_bytes);
    const bool reassoc_is_canonical_scratch =
        reassoc_on_disk &&
        options.output_mode == FullTensorOutputMode::CanonicalDenseOnDisk;

    Tensor<double, 1> reassoc_memory;
    FullTensorScratchFileCleanup reassoc_scratch_cleanup;
    Tensor<double, 1> reassoc_disk;
    double* reassoc_data = nullptr;

    {
        FullTensorAutoScopedTimer timer(result.timings.output_allocate_seconds);
        if (reassoc_on_disk) {
            full_tensor_throw_if_disk_output_exceeds_guard(
                execution_plan.reassociated_output_bytes,
                options.max_reassociated_output_disk_bytes,
                "Reassociated full tensor");
            result.reassociated_disk_root = reassoc_is_canonical_scratch
                ? full_tensor_canonical_reassociated_scratch_root(
                      options,
                      full_tensor_spins_array_from_b4spins(spins),
                      Dl,
                      gamma)
                : full_tensor_reassociated_disk_root(
                      options,
                      full_tensor_spins_array_from_b4spins(spins),
                      Dl,
                      gamma);
            if (reassoc_is_canonical_scratch) {
                reassoc_scratch_cleanup.arm(result.reassociated_disk_root);
            }
            const auto tag = make_full_tensor_binary_tag_reassociated_with_storage_order(
                full_tensor_spins_array_from_b4spins(spins),
                Dl,
                gamma,
                result.canonical_dims,
                execution_plan.reassociated_values,
                full_tensor_reassociated_storage_order_for_output_plan(
                    execution_plan.output_plan),
                options.simplicity_constraint);
            reassoc_disk = Tensor<double, 1>::create_file(
                result.reassociated_disk_root,
                std::array<std::size_t, 1>{{output_size}},
                // Fresh mapped files are logically zero after truncation.
                TensorInit::Uninitialized,
                tensor_disk::LayoutCode::FullTensorReassociatedReflectionCompactByI1,
                tag);
            reassoc_data = reassoc_disk.data();
            result.reassociated_materialized_on_disk = true;
        } else {
            reassoc_memory = Tensor<double, 1>(
                {{output_size}},
                TensorInit::zero,
                TensorLayout::FullTensorReassociatedReflectionCompactByI1);
            reassoc_data = reassoc_memory.data();
            result.reassociated_materialized = true;
        }
    }

    Tensor<double, 5> canonical_disk;
    double* canonical_disk_data = nullptr;

    if (options.output_mode == FullTensorOutputMode::CanonicalDenseInMemory ||
        options.output_mode == FullTensorOutputMode::CanonicalDenseOnDisk) {
        if (execution_plan.canonical_output_exceeds_guard) {
            std::ostringstream oss;
            oss << "Refusing canonical dense output: output is "
                << full_tensor_format_bytes_string(execution_plan.canonical_output_bytes)
                << ", above the configured guard.";
            throw b4utils::ResourceLimitFailure(oss.str());
        }
        if (options.output_mode == FullTensorOutputMode::CanonicalDenseInMemory) {
            FullTensorAutoScopedTimer timer(result.timings.output_allocate_seconds);
            (void)full_tensor_checked_size_t_from_u64(
                execution_plan.canonical_values,
                "canonical full-tensor output size");
            result.canonical_output = Tensor<double, 5>(
                result.canonical_dims,
                // The final scatter assigns the complete canonical tensor.
                TensorInit::Uninitialized,
                TensorLayout::CanonicalDenseColumnMajor);
        } else {
            full_tensor_throw_if_disk_output_exceeds_guard(
                execution_plan.canonical_output_bytes,
                options.max_canonical_output_disk_bytes,
                "Canonical dense full tensor");
            FullTensorAutoScopedTimer timer(result.timings.output_allocate_seconds);
            result.canonical_disk_root = full_tensor_canonical_disk_root(
                options,
                full_tensor_spins_array_from_b4spins(spins),
                Dl,
                gamma);
            const auto canonical_tag = make_full_tensor_binary_tag_canonical(
                full_tensor_spins_array_from_b4spins(spins),
                Dl,
                gamma,
                result.canonical_dims,
                execution_plan.canonical_values,
                options.simplicity_constraint);
            canonical_disk = Tensor<double, 5>::create_file(
                result.canonical_disk_root,
                result.canonical_dims,
                // The final scatter assigns the complete canonical tensor.
                TensorInit::Uninitialized,
                tensor_disk::LayoutCode::CanonicalDenseColumnMajor,
                canonical_tag);
            canonical_disk_data = canonical_disk.data();
            result.canonical_materialized_on_disk = true;
        }
    }

    const std::size_t X = FTC.xvals.size();
    std::size_t ix = 0;

    auto sj_progress_opts = full_tensor_auto_progress_options_for_backend(
        options, "Full tensor sj-stream", "x", 1);
    sj_progress_opts.phase =
        sl2cfoamxx::ProgressPhase::streamed_computation;
    FullTensorReflectionCompactWeightedProgressBar sj_progress(
        static_cast<std::uint64_t>(std::max<std::size_t>(1, X)), sj_progress_opts);
    sj_progress.start();

    while (ix < X) {
        FullTensorInputSupportPlan slab_support;
        std::uint64_t slab_peak = 0;
        const std::size_t ix_next = full_tensor_choose_next_x_slab_end(
            spins, static_cast<int>(Dl), FTC, execution_plan, options,
            ix, slab_support, &slab_peak);

        if (slab_peak > execution_plan.memory_budget_bytes &&
            options.planner.memory_budget_bytes != 0) {
            std::ostringstream oss;
            oss << "SJ-streamed full-tensor backend cannot fit even one x slab. "
                << "slab peak = " << full_tensor_format_bytes_string(slab_peak)
                << ", budget = " << full_tensor_format_bytes_string(execution_plan.memory_budget_bytes)
                << ". This case needs q-slicing inside the full-tensor kernel, not just x-slicing.";
            throw b4utils::ResourceLimitFailure(oss.str());
        }

        if (options.print_streaming_plan && options.log) {
            *options.log << "[full tensor sj-stream] x slab [" << ix << "," << ix_next
                         << ") peak=" << full_tensor_format_bytes_string(slab_peak) << "\n";
        }

        PackedFullTensorSJTensors slab_sj;
        {
            FullTensorAutoScopedTimer timer(result.timings.sj_fill_seconds);
            fill_packed_full_tensor_sj_tensors_rec(
                spins, slab_support, slab_sj);
        }
        ++result.sj_slab_builds;

        FullTensorPackedContractionL25ScalarProfile slab_profile;
        FullTensorScheduledBalanceProfile slab_balance;
        FullTensorPackedContractionL25ScalarOptions slab_copt = options.contraction_options;
        slab_copt.zero_output = false; // output was explicitly initialized once above
        FullTensorReflectionCompactWeightedProgressOptions slab_progress{};
        slab_progress.enabled = false; // outer x-slab progress is handled by the dispatcher

        {
            FullTensorAutoScopedTimer timer(result.timings.contraction_seconds);
            result.output_plan =
                contract_full_tensor_packed_fused_l25_stacked_scalar_profiled_reassociated_auto_scheduled_by_i1_layout(
                    slab_support, slab_sj, packed_full_B, reassoc_data,
                    slab_copt, slab_profile, options.triangular_options,
                    options.schedule_options, slab_progress, &slab_balance);
        }
        full_tensor_auto_add_profile(result.contraction_profile, slab_profile);
        result.schedule_balance = slab_balance; // last slab diagnostics; aggregate scheduling is future bookkeeping

        if (sj_progress.enabled()) {
            sj_progress.add(static_cast<std::uint64_t>(ix_next - ix));
        }
        ix = ix_next;
    }
    sj_progress.stop();
    result.boundary_tiles_completed = 1;

    result.ran_resident_backend = false;
    result.output_plan = execution_plan.output_plan;

    const bool canonical_output_requested =
        options.output_mode == FullTensorOutputMode::CanonicalDenseInMemory ||
        options.output_mode == FullTensorOutputMode::CanonicalDenseOnDisk;
    if (!canonical_output_requested) {
        result.value_checksum = full_tensor_value_checksum(
            reassoc_data, output_size);
    }

    if (canonical_output_requested) {
        FullTensorAutoScopedTimer timer(result.timings.canonical_scatter_seconds);
        scatter_full_tensor_reassociated_auto_by_i1_to_canonical(
            full_support,
            result.output_plan,
            reassoc_data,
            options.output_mode == FullTensorOutputMode::CanonicalDenseInMemory
                ? result.canonical_output.data()
                : canonical_disk_data);
        if (options.output_mode == FullTensorOutputMode::CanonicalDenseInMemory) {
            result.canonical_materialized = true;
            result.value_checksum = full_tensor_value_checksum(
                result.canonical_output.data(), result.canonical_output.size());
        } else {
            result.value_checksum = full_tensor_value_checksum(
                canonical_disk_data,
                full_tensor_checked_size_t_from_u64(
                    execution_plan.canonical_values,
                    "canonical full-tensor checksum size"));
            canonical_disk.flush();
            result.canonical_materialized_on_disk = true;
        }
    }

    if (reassoc_on_disk) {
        reassoc_disk.flush();
        if (reassoc_scratch_cleanup.armed()) {
            reassoc_disk.clear();
            reassoc_data = nullptr;
            reassoc_scratch_cleanup.remove_now();
            result.reassociated_materialized_on_disk = false;
            result.reassociated_disk_root.clear();
        }
    } else {
        const bool keep_reassoc =
            (options.output_mode == FullTensorOutputMode::ReassociatedOnly) ||
            (options.output_mode == FullTensorOutputMode::ReassociatedAuto) ||
            (options.output_mode == FullTensorOutputMode::CanonicalDenseInMemory &&
             options.keep_reassociated_output);
        if (keep_reassoc && options.output_mode != FullTensorOutputMode::BenchmarkOnly) {
            result.reassociated_output = std::move(reassoc_memory);
        }
    }

    if (options.output_mode == FullTensorOutputMode::BenchmarkOnly) {
        result.reassociated_output.clear();
        result.canonical_output.clear();
        result.reassociated_materialized = false;
        result.canonical_materialized = false;
    }

    return result;
}

FullTensorAutoResult compute_full_tensor_fully_streamed(
    const B4Spins& spins,
    std::size_t Dl,
    double gamma,
    const std::vector<double>& FactTable,
    const FullTensorChannels& FTC,
    const FullTensorExecutionPlan& execution_plan,
    const FullTensorAutoOptions& options)
{
    if (!options.allow_booster_streaming) {
        full_tensor_throw_unavailable_backend(execution_plan);
    }

    if (options.output_mode == FullTensorOutputMode::ReassociatedOnly ||
        options.output_mode == FullTensorOutputMode::ReassociatedOnDisk) {
        throw b4utils::UnsupportedOperationFailure(
            "The fully-streamed backend cannot currently return a global reassociated buffer: "
            "boundary intertwiner tiling changes the local reassociated layout. Use ReassociatedAuto "
            "to fall back to canonical disk output, CanonicalDenseOnDisk explicitly, "
            "CanonicalDenseInMemory for small tests, or BenchmarkOnly for path validation.");
    }

    FullTensorAutoResult result;
    result.output_mode = options.output_mode;
    result.execution_plan = execution_plan;
    result.output_plan = execution_plan.output_plan;
    result.canonical_dims = {{FTC.i1.size(), FTC.i2.size(), FTC.i3.size(), FTC.i4.size(), FTC.i5.size()}};

    const bool fully_stream_canonical_memory =
        options.output_mode == FullTensorOutputMode::CanonicalDenseInMemory;
    const bool fully_stream_canonical_disk =
        options.output_mode == FullTensorOutputMode::CanonicalDenseOnDisk ||
        options.output_mode == FullTensorOutputMode::ReassociatedAuto;

    Tensor<double, 5> canonical_disk;
    double* canonical_disk_data = nullptr;

    if (fully_stream_canonical_memory || fully_stream_canonical_disk) {
        if (execution_plan.canonical_output_exceeds_guard) {
            std::ostringstream oss;
            oss << "Refusing fully-streamed canonical output: output is "
                << full_tensor_format_bytes_string(execution_plan.canonical_output_bytes)
                << ", above the configured guard.";
            throw b4utils::ResourceLimitFailure(oss.str());
        }
        if (fully_stream_canonical_memory) {
            FullTensorAutoScopedTimer timer(result.timings.output_allocate_seconds);
            (void)full_tensor_checked_size_t_from_u64(
                execution_plan.canonical_values,
                "canonical full-tensor output size");
            result.canonical_output = Tensor<double, 5>(
                result.canonical_dims,
                TensorInit::Zero,
                TensorLayout::CanonicalDenseColumnMajor);
            result.canonical_materialized = true;
        } else {
            full_tensor_throw_if_disk_output_exceeds_guard(
                execution_plan.canonical_output_bytes,
                options.max_canonical_output_disk_bytes,
                "Canonical dense full tensor");
            FullTensorAutoScopedTimer timer(result.timings.output_allocate_seconds);
            result.canonical_disk_root = full_tensor_canonical_disk_root(
                options,
                full_tensor_spins_array_from_b4spins(spins),
                Dl,
                gamma);
            const auto canonical_tag = make_full_tensor_binary_tag_canonical(
                full_tensor_spins_array_from_b4spins(spins),
                Dl,
                gamma,
                result.canonical_dims,
                execution_plan.canonical_values,
                options.simplicity_constraint);
            canonical_disk = Tensor<double, 5>::create_file(
                result.canonical_disk_root,
                result.canonical_dims,
                TensorInit::Zero,
                tensor_disk::LayoutCode::CanonicalDenseColumnMajor,
                canonical_tag);
            canonical_disk_data = canonical_disk.data();
            result.canonical_materialized_on_disk = true;
        }
    }

    const std::size_t I1 = FTC.i1.size();
    const std::size_t I2 = FTC.i2.size();
    const std::size_t I3 = FTC.i3.size();
    const std::size_t I4 = FTC.i4.size();
    const std::size_t I5 = FTC.i5.size();
    const std::size_t X  = FTC.xvals.size();
    FullTensorExecutionPlan panel_admission_plan = execution_plan;
    if (options.fully_streamed_common_sj_cache_bytes != 0 &&
        options.fully_streamed_common_sj_cache_bytes < panel_admission_plan.memory_budget_bytes) {
        // An explicit cache request is a tuning decision: reserve it before
        // choosing P so the cache and the selected tile are jointly admissible.
        panel_admission_plan.memory_budget_bytes -=
            options.fully_streamed_common_sj_cache_bytes;
    }
    std::uint64_t selected_panel_peak = 0;
    const std::size_t P = full_tensor_choose_fully_streamed_boundary_panel_size(
        spins, static_cast<int>(Dl), FTC, panel_admission_plan, options, &selected_panel_peak);

    if (selected_panel_peak > execution_plan.memory_budget_bytes &&
        options.planner.memory_budget_bytes != 0) {
        std::ostringstream oss;
        oss << "Fully-streamed full-tensor backend cannot fit even a single boundary intertwiner tile. "
            << "tile peak = " << full_tensor_format_bytes_string(selected_panel_peak)
            << ", budget = " << full_tensor_format_bytes_string(execution_plan.memory_budget_bytes)
            << ". Reduce the safety margin for artificial tests, use BenchmarkOnly, "
               "or wait for the disk/tile-sink + implicit-support backend.";
        throw b4utils::ResourceLimitFailure(oss.str());
    }

    const std::uint64_t n1_tiles = full_tensor_panel_tile_count(I1, P);
    const std::uint64_t n2_tiles = full_tensor_panel_tile_count(I2, P);
    const std::uint64_t n3_tiles = full_tensor_panel_tile_count(I3, P);
    const std::uint64_t n4_tiles = full_tensor_panel_tile_count(I4, P);
    const std::uint64_t n5_tiles = full_tensor_panel_tile_count(I5, P);
    std::uint64_t total_boundary_tiles = 1;
    total_boundary_tiles = full_tensor_planner_sat_mul_u64(total_boundary_tiles, n1_tiles);
    total_boundary_tiles = full_tensor_planner_sat_mul_u64(total_boundary_tiles, n2_tiles);
    total_boundary_tiles = full_tensor_planner_sat_mul_u64(total_boundary_tiles, n3_tiles);
    total_boundary_tiles = full_tensor_planner_sat_mul_u64(total_boundary_tiles, n4_tiles);
    total_boundary_tiles = full_tensor_planner_sat_mul_u64(total_boundary_tiles, n5_tiles);
    if (total_boundary_tiles == 0) total_boundary_tiles = 1;

    if (options.print_streaming_plan && options.log) {
        *options.log << "[full tensor fully-streamed] boundary intertwiner panel size=" << P
                     << " boundary intertwiner tiles=" << total_boundary_tiles
                     << " selected tile peak=" << full_tensor_format_bytes_string(selected_panel_peak)
                     << " x_per_slab=" << (options.streamed_x_per_slab == 0 ? 1 : options.streamed_x_per_slab)
                     << " output=" << full_tensor_output_mode_name(options.output_mode) << "\n";
    }

    auto fs_progress_opts = full_tensor_auto_progress_options_for_backend(
        options, "Full tensor fully-streamed", "boundary intertwiner-tile", 1);
    fs_progress_opts.phase =
        sl2cfoamxx::ProgressPhase::streamed_computation;
    FullTensorReflectionCompactWeightedProgressBar fs_progress(total_boundary_tiles, fs_progress_opts);
    fs_progress.start();

    // Booster legs are boundary intertwiner-separable.  The historical loop order placed i1
    // outermost and regenerated all B2..B5 for every five-dimensional boundary intertwiner
    // tile.  Generate each leg at its natural nesting level instead, keep just
    // four panel payloads alive, and put i1 innermost (boosters do not depend on
    // it).  This changes booster work from 4*N1*N2*N3*N4*N5 panel builds to
    // N2 + N2*N3 + N2*N3*N4 + N2*N3*N4*N5 without increasing peak storage.
    const std::size_t i1_ref_e = std::min<std::size_t>(I1, P);
    const std::size_t i2_ref_e = std::min<std::size_t>(I2, P);
    const std::size_t i3_ref_e = std::min<std::size_t>(I3, P);
    const std::size_t i4_ref_e = std::min<std::size_t>(I4, P);
    const std::size_t i5_ref_e = std::min<std::size_t>(I5, P);

    const auto build_panel_support = [&](std::size_t i1b, std::size_t i1e,
                                         std::size_t i2b, std::size_t i2e,
                                         std::size_t i3b, std::size_t i3e,
                                         std::size_t i4b, std::size_t i4e,
                                         std::size_t i5b, std::size_t i5e) {
        FullTensorChannels panel_ch = full_tensor_channels_with_boundary_slices(
            FTC, i1b, i1e, i2b, i2e, i3b, i3e, i4b, i4e, i5b, i5e);
        return build_full_tensor_input_support_plan(
            spins, static_cast<int>(Dl), panel_ch);
    };

    // sjB/sjC/sjD have no boundary intertwiner index.  In the fully streamed route the same
    // recurrence slabs would otherwise be regenerated for every Cartesian
    // boundary intertwiner tile.  Cache as many x slabs as fit in *confirmed* planner
    // headroom.  Uncached slabs retain the original bounded producer, so this
    // is safe on small-memory machines and useful on large-memory nodes.
    struct CommonSjCacheSlab {
        std::size_t ix_begin = 0;
        std::size_t ix_end = 0;
        bool populated = false;
        PackedFullTensorSJTensors values;
    };
    const std::size_t x_per_slab = options.streamed_x_per_slab == 0
        ? 1
        : std::max<std::size_t>(1, options.streamed_x_per_slab);
    const std::size_t x_slab_count = (X + x_per_slab - 1) / x_per_slab;
    std::vector<CommonSjCacheSlab> common_sj_cache(x_slab_count);

    const std::uint64_t cache_headroom = execution_plan.memory_budget_bytes > selected_panel_peak
        ? execution_plan.memory_budget_bytes - selected_panel_peak
        : 0;
    const std::uint64_t auto_cache_ceiling = std::min<std::uint64_t>(
        1ull * 1024ull * 1024ull * 1024ull,
        execution_plan.memory_budget_bytes / 8ull);
    const std::uint64_t requested_cache = options.fully_streamed_common_sj_cache_bytes == 0
        ? auto_cache_ceiling
        : options.fully_streamed_common_sj_cache_bytes;
    const std::uint64_t common_cache_budget = std::min(cache_headroom, requested_cache);

    if (common_cache_budget != 0 && total_boundary_tiles > 1 && X != 0) {
        FullTensorChannels reference_ch = full_tensor_channels_with_boundary_slices(
            FTC,
            0, i1_ref_e,
            0, i2_ref_e,
            0, i3_ref_e,
            0, i4_ref_e,
            0, i5_ref_e);

        for (std::size_t slab = 0, ix = 0; ix < X; ++slab, ix += x_per_slab) {
            CommonSjCacheSlab& entry = common_sj_cache[slab];
            entry.ix_begin = ix;
            entry.ix_end = std::min<std::size_t>(X, ix + x_per_slab);

            FullTensorChannels common_x_ch = reference_ch;
            common_x_ch.xvals = full_tensor_slice_axis(FTC.xvals, entry.ix_begin, entry.ix_end);
            FullTensorInputSupportPlan common_x_support;
            {
                FullTensorAutoScopedTimer timer(result.timings.streamed_support_seconds);
                common_x_support = build_full_tensor_input_support_plan(
                    spins, static_cast<int>(Dl), common_x_ch);
            }

            std::uint64_t slab_bytes = 0;
            slab_bytes = full_tensor_planner_sat_add_u64(
                slab_bytes, common_x_support.sjB.payload_bytes());
            slab_bytes = full_tensor_planner_sat_add_u64(
                slab_bytes, common_x_support.sjC.payload_bytes());
            slab_bytes = full_tensor_planner_sat_add_u64(
                slab_bytes, common_x_support.sjD.payload_bytes());
            if (slab_bytes > common_cache_budget - result.sj_common_cache_bytes) {
                continue;
            }

            {
                FullTensorAutoScopedTimer timer(result.timings.sj_fill_seconds);
                fill_packed_full_tensor_sj_tensors_rec(
                    spins, common_x_support, entry.values,
                    FullTensorSjBoundaryIndependent);
            }
            entry.populated = true;
            result.sj_common_cache_bytes += slab_bytes;
            ++result.sj_common_cache_slabs;
            ++result.sj_slab_builds;
        }
    }

    for (std::size_t i2b = 0; i2b < I2; i2b += P) {
        const std::size_t i2e = std::min<std::size_t>(I2, i2b + P);
        PackedFullTensorBoosters B2_panel;
        {
            FullTensorInputSupportPlan leg_support;
            {
                FullTensorAutoScopedTimer timer(result.timings.streamed_support_seconds);
                leg_support = build_panel_support(
                    0, i1_ref_e, i2b, i2e, 0, i3_ref_e, 0, i4_ref_e, 0, i5_ref_e);
            }
            FullTensorAutoScopedTimer timer(result.timings.booster_fill_seconds);
            fill_packed_full_tensor_booster_leg_via_canonical(
                spins, gamma, FactTable, leg_support,
                FullTensorBoosterLeg::B2, B2_panel,
                options.simplicity_constraint, options.booster_rho_tile);
            ++result.booster_leg_builds;
        }

        for (std::size_t i3b = 0; i3b < I3; i3b += P) {
            const std::size_t i3e = std::min<std::size_t>(I3, i3b + P);
            PackedFullTensorBoosters B3_panel;
            {
                FullTensorInputSupportPlan leg_support;
                {
                    FullTensorAutoScopedTimer timer(result.timings.streamed_support_seconds);
                    leg_support = build_panel_support(
                        0, i1_ref_e, i2b, i2e, i3b, i3e, 0, i4_ref_e, 0, i5_ref_e);
                }
                FullTensorAutoScopedTimer timer(result.timings.booster_fill_seconds);
                fill_packed_full_tensor_booster_leg_via_canonical(
                    spins, gamma, FactTable, leg_support,
                    FullTensorBoosterLeg::B3, B3_panel,
                    options.simplicity_constraint, options.booster_rho_tile);
                ++result.booster_leg_builds;
            }

            for (std::size_t i4b = 0; i4b < I4; i4b += P) {
                const std::size_t i4e = std::min<std::size_t>(I4, i4b + P);
                PackedFullTensorBoosters B4_panel;
                {
                    FullTensorInputSupportPlan leg_support;
                    {
                        FullTensorAutoScopedTimer timer(result.timings.streamed_support_seconds);
                        leg_support = build_panel_support(
                            0, i1_ref_e, i2b, i2e, i3b, i3e, i4b, i4e, 0, i5_ref_e);
                    }
                    FullTensorAutoScopedTimer timer(result.timings.booster_fill_seconds);
                    fill_packed_full_tensor_booster_leg_via_canonical(
                        spins, gamma, FactTable, leg_support,
                        FullTensorBoosterLeg::B4, B4_panel,
                        options.simplicity_constraint, options.booster_rho_tile);
                    ++result.booster_leg_builds;
                }

                for (std::size_t i5b = 0; i5b < I5; i5b += P) {
                    const std::size_t i5e = std::min<std::size_t>(I5, i5b + P);
                    PackedFullTensorBoosters B5_panel;
                    {
                        FullTensorInputSupportPlan leg_support;
                        {
                            FullTensorAutoScopedTimer timer(result.timings.streamed_support_seconds);
                            leg_support = build_panel_support(
                                0, i1_ref_e, i2b, i2e, i3b, i3e, i4b, i4e, i5b, i5e);
                        }
                        FullTensorAutoScopedTimer timer(result.timings.booster_fill_seconds);
                        fill_packed_full_tensor_booster_leg_via_canonical(
                            spins, gamma, FactTable, leg_support,
                            FullTensorBoosterLeg::B5, B5_panel,
                            options.simplicity_constraint,
                            options.booster_rho_tile);
                        ++result.booster_leg_builds;
                    }

                    PackedFullTensorBoosters tile_B_view;
                    assemble_packed_full_tensor_booster_panels(
                        B2_panel, B3_panel, B4_panel, B5_panel, tile_B_view);

                    for (std::size_t i1b = 0; i1b < I1; i1b += P) {
                        const std::size_t i1e = std::min<std::size_t>(I1, i1b + P);
                        FullTensorChannels tile_ch = full_tensor_channels_with_boundary_slices(
                            FTC, i1b, i1e, i2b, i2e, i3b, i3e, i4b, i4e, i5b, i5e);

                        FullTensorInputSupportPlan tile_fullx_support;
                        {
                            FullTensorAutoScopedTimer timer(result.timings.streamed_support_seconds);
                            tile_fullx_support = build_full_tensor_input_support_plan(
                                spins, static_cast<int>(Dl), tile_ch);
                        }

                        const FullTensorReassociatedAutoOutputPlan tile_plan =
                            make_full_tensor_reassociated_auto_output_plan(tile_fullx_support);
                        std::vector<double> tile_reassoc(tile_plan.values_count, 0.0);

                        std::size_t ix = 0;
                        std::size_t slab_index = 0;
                        while (ix < X) {
                            const std::size_t ix_next = std::min<std::size_t>(X, ix + x_per_slab);

                            FullTensorChannels tile_x_ch = tile_ch;
                            tile_x_ch.xvals = full_tensor_slice_axis(FTC.xvals, ix, ix_next);
                            FullTensorInputSupportPlan tile_x_support;
                            {
                                FullTensorAutoScopedTimer timer(result.timings.streamed_support_seconds);
                                tile_x_support = build_full_tensor_input_support_plan(
                                    spins, static_cast<int>(Dl), tile_x_ch);
                            }

                            PackedFullTensorSJTensors tile_sj;
                            PackedFullTensorSJTensors* cached_common = nullptr;
                            {
                                FullTensorAutoScopedTimer timer(result.timings.sj_fill_seconds);
                                if (slab_index < common_sj_cache.size() &&
                                    common_sj_cache[slab_index].populated) {
                                    cached_common = &common_sj_cache[slab_index].values;
                                    fill_packed_full_tensor_sj_tensors_rec(
                                        spins, tile_x_support, tile_sj,
                                        FullTensorSjBoundaryDependent);
                                    if (cached_common->sjB.size() !=
                                            static_cast<std::size_t>(tile_x_support.sjB.packed_values) ||
                                        cached_common->sjC.size() !=
                                            static_cast<std::size_t>(tile_x_support.sjC.packed_values) ||
                                        cached_common->sjD.size() !=
                                            static_cast<std::size_t>(tile_x_support.sjD.packed_values)) {
                                        throw std::logic_error(
                                            "Boundary-independent full-tensor 6j cache layout mismatch");
                                    }
                                    tile_sj.sjB.swap(cached_common->sjB);
                                    tile_sj.sjC.swap(cached_common->sjC);
                                    tile_sj.sjD.swap(cached_common->sjD);
                                    ++result.sj_common_cache_hits;
                                } else {
                                    fill_packed_full_tensor_sj_tensors_rec(
                                        spins, tile_x_support, tile_sj);
                                }
                            }
                            ++result.sj_slab_builds;

                            FullTensorPackedContractionL25ScalarProfile tile_profile;
                            FullTensorPackedContractionL25ScalarOptions tile_copt = options.contraction_options;
                            tile_copt.zero_output = false;
                            FullTensorScheduledBalanceProfile tile_balance;

                            {
                                FullTensorAutoScopedTimer timer(result.timings.contraction_seconds);
                                (void)contract_full_tensor_packed_fused_l25_stacked_scalar_profiled_reassociated_auto_scheduled_by_i1_layout(
                                    tile_x_support, tile_sj, tile_B_view,
                                    tile_reassoc.data(), tile_copt, tile_profile,
                                    options.triangular_options, options.schedule_options,
                                    FullTensorReflectionCompactWeightedProgressOptions{}, &tile_balance);
                            }
                            full_tensor_auto_add_profile(result.contraction_profile, tile_profile);
                            result.schedule_balance = tile_balance;
                            if (cached_common != nullptr) {
                                // Return the immutable common payload to its
                                // cache entry before the next boundary intertwiner tile.
                                tile_sj.sjB.swap(cached_common->sjB);
                                tile_sj.sjC.swap(cached_common->sjC);
                                tile_sj.sjD.swap(cached_common->sjD);
                            }
                            ix = ix_next;
                            ++slab_index;
                        }

                        if (fully_stream_canonical_memory || fully_stream_canonical_disk) {
                            const std::size_t I1t = i1e - i1b;
                            const std::size_t I2t = i2e - i2b;
                            const std::size_t I3t = i3e - i3b;
                            const std::size_t I4t = i4e - i4b;
                            const std::size_t I5t = i5e - i5b;
                            std::vector<double> tile_canonical(I1t * I2t * I3t * I4t * I5t, 0.0);
                            {
                                FullTensorAutoScopedTimer timer(result.timings.canonical_scatter_seconds);
                                scatter_full_tensor_reassociated_auto_by_i1_to_canonical(
                                    tile_fullx_support, tile_plan,
                                    tile_reassoc.data(), tile_canonical.data());
                                if (fully_stream_canonical_memory) {
                                    full_tensor_add_local_canonical_tile_to_global(
                                        tile_canonical, result.canonical_output, result.canonical_dims,
                                        i1b, i2b, i3b, i4b, i5b,
                                        I1t, I2t, I3t, I4t, I5t);
                                } else {
                                    full_tensor_add_local_canonical_tile_to_global_ptr(
                                        tile_canonical, canonical_disk_data, result.canonical_dims,
                                        i1b, i2b, i3b, i4b, i5b,
                                        I1t, I2t, I3t, I4t, I5t);
                                }
                            }
                            result.value_checksum += full_tensor_value_checksum(
                                tile_canonical.data(), tile_canonical.size());
                        } else {
                            result.value_checksum += full_tensor_value_checksum(
                                tile_reassoc.data(), tile_reassoc.size());
                        }

                        ++result.boundary_tiles_completed;
                        if (fs_progress.enabled()) fs_progress.add(1);
                    }
                }
            }
        }
    }

    fs_progress.stop();
    if (fully_stream_canonical_disk) {
        result.value_checksum = full_tensor_value_checksum(
            canonical_disk_data,
            full_tensor_checked_size_t_from_u64(
                execution_plan.canonical_values,
                "canonical full-tensor checksum size"));
        canonical_disk.flush();
        result.canonical_materialized_on_disk = true;
    } else if (fully_stream_canonical_memory) {
        result.value_checksum = full_tensor_value_checksum(
            result.canonical_output.data(), result.canonical_output.size());
    }

    result.ran_resident_backend = false;
    return result;
}

FullTensorAutoResult compute_full_tensor_auto(
    const std::array<int, 10>& SpinsArray,
    std::size_t Dl,
    double gamma,
    const std::vector<double>& FactTable,
    FullTensorAutoOptions options)
{
    using clock = std::chrono::high_resolution_clock;
    const auto total_start = clock::now();

    FullTensorAutoResult result;
    result.output_mode = options.output_mode;

    const B4Spins spins(SpinsArray);

    FullTensorChannels FTC;
    {
        FullTensorAutoScopedTimer timer(result.timings.channels_seconds);
        FTC = make_FullTensorChannels(spins, Dl);
    }

    // Apply the convenience memory-budget override before doing any planning.
    // This is mainly for tests: set it artificially low to force SJ or booster
    // streaming on a small problem.
    if (options.memory_budget_bytes != 0) {
        options.planner.memory_budget_bytes = options.memory_budget_bytes;
    }

    // Count only outputs that must be resident in RAM.  Auto/disk modes are
    // planned as disk-backed so the dispatcher can select the fastest internal
    // backend that fits, then use mapped Tensor storage if the output buffer would
    // blow the budget.
    options.planner.include_reassociated_output =
        full_tensor_output_mode_forces_reassoc_memory(options.output_mode);
    options.planner.include_canonical_output_in_peak =
        full_tensor_output_mode_needs_canonical_memory(options.output_mode);

    const bool canonical_output_explicitly_requested =
        full_tensor_output_mode_needs_canonical_memory(options.output_mode) ||
        full_tensor_output_mode_needs_canonical_disk(options.output_mode);
    options.planner.refuse_large_canonical_output =
        options.planner.refuse_large_canonical_output && canonical_output_explicitly_requested;

    {
        FullTensorAutoScopedTimer timer(result.timings.preflight_seconds);
        result.preflight = preflight_full_tensor_output_from_channels(FTC, options.planner);
    }

    if (options.print_preflight && options.log) {
        result.preflight.print(*options.log);
    }

    result.canonical_dims = {{FTC.i1.size(), FTC.i2.size(), FTC.i3.size(), FTC.i4.size(), FTC.i5.size()}};
    const FullTensorReassociatedAutoOutputPlan
        expected_reassociated_output_plan =
            full_tensor_expected_reassociated_output_plan_from_channels(
                SpinsArray, FTC);

    // Fast cache hit path.  This intentionally checks the binary header/tag,
    // not merely the directory and file name.  A file moved into the wrong
    // Dspins_* folder will therefore be rejected and recomputed.  We do this
    // before materializing support, because for very large tensors the support
    // metadata itself can already be a RAM problem.
    if (options.reuse_existing_disk_output &&
        !options.force_recompute_existing_output &&
        options.output_mode != FullTensorOutputMode::BenchmarkOnly) {
        if (full_tensor_auto_primary_disk_cache_is_reassociated(options)) {
            const auto path = full_tensor_reassociated_disk_root(options, SpinsArray, Dl, gamma);
            const auto chk = full_tensor_check_existing_reassociated_file(
                path, SpinsArray, Dl, gamma, result.canonical_dims,
                expected_reassociated_output_plan,
                options.simplicity_constraint);
            if (chk.reusable) {
                result.output_plan = expected_reassociated_output_plan;
                result.reassociated_disk_root = path;
                result.reassociated_materialized_on_disk = true;
                result.reused_existing_disk_output = true;
                result.reused_existing_disk_path = path;
                result.reused_existing_disk_layout = "reassociated";
                result.timings.total_seconds = std::chrono::duration<double>(clock::now() - total_start).count();
                if (options.print_existing_output_status && options.log) {
                    *options.log << "[full tensor cache] using existing reassociated tensor: "
                                 << path << "\n";
                }
                if (options.print_timing && options.log) {
                    result.timings.print(*options.log);
                }
                return result;
            } else if (chk.exists && options.print_existing_output_status && options.log) {
                *options.log << "[full tensor cache] existing reassociated file rejected: "
                             << path << " (" << chk.reason << ")\n";
            }
        } else if (full_tensor_auto_primary_disk_cache_is_canonical(options)) {
            const auto path = full_tensor_canonical_disk_root(options, SpinsArray, Dl, gamma);
            const auto chk = full_tensor_check_existing_canonical_file(
                path, SpinsArray, Dl, gamma, result.canonical_dims,
                result.preflight.canonical_values,
                options.simplicity_constraint);
            if (chk.reusable) {
                result.canonical_disk_root = path;
                result.canonical_materialized_on_disk = true;
                result.reused_existing_disk_output = true;
                result.reused_existing_disk_path = path;
                result.reused_existing_disk_layout = "canonical";
                result.timings.total_seconds = std::chrono::duration<double>(clock::now() - total_start).count();
                if (options.print_existing_output_status && options.log) {
                    *options.log << "[full tensor cache] using existing canonical tensor: "
                                 << path << "\n";
                }
                if (options.print_timing && options.log) {
                    result.timings.print(*options.log);
                }
                return result;
            } else if (chk.exists && options.print_existing_output_status && options.log) {
                *options.log << "[full tensor cache] existing canonical file rejected: "
                             << path << " (" << chk.reason << ")\n";
            }
        }
    }

    if (options.planner.refuse_large_canonical_output &&
        result.preflight.canonical_output_exceeds_guard) {
        std::ostringstream oss;
        oss << "Canonical full tensor output would require "
            << full_tensor_format_bytes_string(result.preflight.canonical_output_bytes)
            << ", exceeding the configured guard of "
            << full_tensor_format_bytes_string(options.planner.max_canonical_output_bytes)
            << ". Use selected entries, boundary contraction, sparse/thresholded output, "
               "or a disk/tile streaming backend.";
        throw b4utils::ResourceLimitFailure(oss.str());
    }

    FullTensorInputSupportPlan support;
    {
        FullTensorAutoScopedTimer timer(result.timings.support_seconds);
        support = build_full_tensor_input_support_plan(
            spins, static_cast<int>(Dl), FTC);
    }

    // Admit the exact P_batch-dependent L/R allocation used by the scheduled
    // CPU contraction before selecting resident or streamed execution.  This
    // is separate from caller-supplied generic scratch and therefore enters
    // every peak estimate exactly once.
    const FullTensorReflectionCompactTriangularCacheInfo cache =
        full_tensor_detect_cache_info_triangular_batched();
    const FullTensorCpuBatchedWorkspaceAdmission batch_admission =
        make_full_tensor_cpu_batched_workspace_admission(
            support,
            options.contraction_options,
            options.triangular_options,
            cache);
    options.planner.cpu_batched_workspace_bytes =
        batch_admission.total_bytes;

    {
        FullTensorAutoScopedTimer timer(result.timings.planning_seconds);
        result.execution_plan = make_full_tensor_execution_plan_from_support(
            support, options.planner);
    }

    if (options.schedule_options.max_tls_output_scratch_bytes == 0) {
        // Do not let an imbalanced x-split schedule silently replicate every
        // i1 output panel for every worker.  The locked exclusive-panel fallback
        // uses one partial panel per active worker and remains correct under the
        // same scheduler.  Cap TLS to one eighth of the admitted host budget,
        // with a 2-GiB ceiling so large-memory nodes do not turn spare RAM into
        // a benchmark dependency.
        options.schedule_options.max_tls_output_scratch_bytes = std::min<std::uint64_t>(
            2ull * 1024ull * 1024ull * 1024ull,
            result.execution_plan.memory_budget_bytes / 8ull);
    }

    if (options.print_execution_plan && options.log) {
        result.execution_plan.print(*options.log);
    }

    if (result.execution_plan.mode == FullTensorPlannerMode::RefuseOutputTooLarge ||
        result.execution_plan.mode == FullTensorPlannerMode::BudgetBelowFixedCost) {
        full_tensor_throw_unavailable_backend(result.execution_plan);
    }

    if (result.execution_plan.mode == FullTensorPlannerMode::ResidentFast ||
        result.execution_plan.mode == FullTensorPlannerMode::ResidentFastNoCanonicalScatter) {
        FullTensorAutoResult resident = compute_full_tensor_resident_fast(
            spins,
            gamma,
            FactTable,
            FTC,
            support,
            result.execution_plan,
            options);

        resident.preflight = result.preflight;
        resident.timings.channels_seconds = result.timings.channels_seconds;
        resident.timings.preflight_seconds = result.timings.preflight_seconds;
        resident.timings.support_seconds = result.timings.support_seconds;
        resident.timings.planning_seconds = result.timings.planning_seconds;
        resident.timings.total_seconds = std::chrono::duration<double>(clock::now() - total_start).count();

        if (options.print_result_summary && options.log) {
            resident.print_summary(*options.log);
        }
        if (options.print_timing && options.log) {
            resident.timings.print(*options.log);
        }
        return resident;
    }

    if (result.execution_plan.mode == FullTensorPlannerMode::NeedsSjStreaming) {
        if (!options.allow_sj_streaming) {
            full_tensor_throw_unavailable_backend(result.execution_plan);
        }

        FullTensorAutoResult streamed = compute_full_tensor_sj_streamed(
            spins, Dl, gamma, FactTable, FTC, support,
            result.execution_plan, options);

        streamed.preflight = result.preflight;
        streamed.timings.channels_seconds = result.timings.channels_seconds;
        streamed.timings.preflight_seconds = result.timings.preflight_seconds;
        streamed.timings.support_seconds = result.timings.support_seconds;
        streamed.timings.planning_seconds = result.timings.planning_seconds;
        streamed.timings.total_seconds = std::chrono::duration<double>(clock::now() - total_start).count();

        if (options.print_result_summary && options.log) {
            streamed.print_summary(*options.log);
        }
        if (options.print_timing && options.log) {
            streamed.timings.print(*options.log);
        }
        return streamed;
    }

    if (result.execution_plan.mode == FullTensorPlannerMode::NeedsBoosterStreamingOrImplicitSupport) {
        if (!options.allow_booster_streaming) {
            full_tensor_throw_unavailable_backend(result.execution_plan);
        }

        FullTensorAutoResult streamed = compute_full_tensor_fully_streamed(
            spins, Dl, gamma, FactTable, FTC,
            result.execution_plan, options);

        streamed.preflight = result.preflight;
        streamed.timings.channels_seconds = result.timings.channels_seconds;
        streamed.timings.preflight_seconds = result.timings.preflight_seconds;
        streamed.timings.support_seconds = result.timings.support_seconds;
        streamed.timings.planning_seconds = result.timings.planning_seconds;
        streamed.timings.total_seconds = std::chrono::duration<double>(clock::now() - total_start).count();

        if (options.print_result_summary && options.log) {
            streamed.print_summary(*options.log);
        }
        if (options.print_timing && options.log) {
            streamed.timings.print(*options.log);
        }
        return streamed;
    }

    full_tensor_throw_unavailable_backend(result.execution_plan);
}

} // namespace b4support
