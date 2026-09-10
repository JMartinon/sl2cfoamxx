// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "full_tensor/CpuContraction.hpp"
#include "core/CpuBlas.hpp"

#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

namespace b4support {

#if defined(__APPLE__)
bool full_tensor_triangular_read_sysctl_size(
    const char* name,
    std::size_t& value) noexcept
{
    std::size_t tmp = 0;
    std::size_t len = sizeof(tmp);
    if (sysctlbyname(name, &tmp, &len, nullptr, 0) == 0 && tmp > 0) {
        value = tmp;
        return true;
    }
    return false;
}
#endif

#if defined(__linux__)
bool full_tensor_triangular_read_first_line(
    const std::string& path,
    std::string& out)
{
    std::ifstream f(path);
    if (!f) {
        return false;
    }
    std::getline(f, out);
    return true;
}
#endif

void FullTensorScalarSupportProfile::add(const FullTensorScalarSupportProfile& other) noexcept
{
    q_blocks_seen         += other.q_blocks_seen;
    q_blocks_nonempty_sjC += other.q_blocks_nonempty_sjC;
    q_blocks_with_T2      += other.q_blocks_with_T2;
    q_blocks_with_T4      += other.q_blocks_with_T4;
    q_blocks_accumulated  += other.q_blocks_accumulated;

    sum_nk3 += other.sum_nk3;
    sum_nk4 += other.sum_nk4;
    max_nk3 = std::max(max_nk3, other.max_nk3);
    max_nk4 = std::max(max_nk4, other.max_nk4);

    T1_seed_values_total += other.T1_seed_values_total;
    T1_seed_values_nnz   += other.T1_seed_values_nnz;
    T3_seed_values_total += other.T3_seed_values_total;
    T3_seed_values_nnz   += other.T3_seed_values_nnz;

    T2_values_total  += other.T2_values_total;
    T2_values_nnz    += other.T2_values_nnz;
    T4_values_total  += other.T4_values_total;
    T4_values_nnz    += other.T4_values_nnz;
    CT2_values_total += other.CT2_values_total;
    CT2_values_nnz   += other.CT2_values_nnz;

    T1_triple_terms   += other.T1_triple_terms;
    T3_triple_terms   += other.T3_triple_terms;
    T2_dot_terms      += other.T2_dot_terms;
    T4_dot_terms      += other.T4_dot_terms;
    CT2_rank1_updates += other.CT2_rank1_updates;
    final_dot_terms   += other.final_dot_terms;
}

void FullTensorPackedContractionL25ScalarProfile::add(const FullTensorPackedContractionL25ScalarProfile& other) noexcept
{
    zero_output.add(other.zero_output);
    T1_seed.add(other.T1_seed);
    T3_seed.add(other.T3_seed);
    T2_zero.add(other.T2_zero);
    T2_stack.add(other.T2_stack);
    T4_zero.add(other.T4_zero);
    T4_stack.add(other.T4_stack);
    CT2_zero.add(other.CT2_zero);
    sjC_access.add(other.sjC_access);
    CT2_scalar.add(other.CT2_scalar);
    final_scalar.add(other.final_scalar);
    total_wall.add(other.total_wall);
    support.add(other.support);
    used_cpu_q_reflection_orbits =
        used_cpu_q_reflection_orbits ||
        other.used_cpu_q_reflection_orbits;
    cpu_q_logical_members_covered +=
        other.cpu_q_logical_members_covered;
    cpu_q_orbits_executed += other.cpu_q_orbits_executed;
    cpu_q_paired_orbits_executed +=
        other.cpu_q_paired_orbits_executed;
    cpu_q_self_orbits_executed +=
        other.cpu_q_self_orbits_executed;
    cpu_q_unpaired_blocks_executed +=
        other.cpu_q_unpaired_blocks_executed;
    cpu_vendor_blas_final_calls +=
        other.cpu_vendor_blas_final_calls;
    cpu_tiled_final_calls += other.cpu_tiled_final_calls;
}

void FullTensorPackedContractionL25ScalarProfile::print(std::ostream& os) const
{
    const auto print_phase = [&](const FullTensorScalarPhaseProfile& p) {
        os << "  " << std::left << std::setw(14) << p.name
           << std::right
           << " time = " << std::setw(12) << std::fixed << std::setprecision(6) << p.seconds << " s"
           << " | calls = " << std::setw(12) << p.calls
           << " | approx FLOPs = " << std::scientific << std::setprecision(4)
           << static_cast<double>(p.approx_flops)
           << " | approx GF/s = " << std::fixed << std::setprecision(3)
           << std::setw(10) << p.gflops_per_s()
           << "\n";
    };

    os << "\nFull tensor L25-stacked scalar contraction profile\n";
    os << "=================================================\n";
    print_phase(zero_output);
    print_phase(T1_seed);
    print_phase(T3_seed);
    print_phase(T2_zero);
    print_phase(T2_stack);
    print_phase(T4_zero);
    print_phase(T4_stack);
    print_phase(CT2_zero);
    print_phase(sjC_access);
    print_phase(CT2_scalar);
    print_phase(final_scalar);
    print_phase(total_wall);

    const double phase_sum = zero_output.seconds + T1_seed.seconds + T3_seed.seconds
                           + T2_zero.seconds + T2_stack.seconds
                           + T4_zero.seconds + T4_stack.seconds
                           + CT2_zero.seconds + sjC_access.seconds + CT2_scalar.seconds
                           + final_scalar.seconds;

    os << "\nPhase fractions, using summed per-thread phase time:\n";
    const auto print_frac = [&](const FullTensorScalarPhaseProfile& p) {
        os << "  " << std::left << std::setw(14) << p.name
           << std::right << std::setw(8) << std::fixed << std::setprecision(2)
           << pct(p.seconds, phase_sum) << " %\n";
    };
    print_frac(zero_output);
    print_frac(T1_seed);
    print_frac(T3_seed);
    print_frac(T2_zero);
    print_frac(T2_stack);
    print_frac(T4_zero);
    print_frac(T4_stack);
    print_frac(CT2_zero);
    print_frac(sjC_access);
    print_frac(CT2_scalar);
    print_frac(final_scalar);

    os << "\nSupport / density diagnostics\n";
    os << "-----------------------------\n";
    os << "  q blocks seen              = " << support.q_blocks_seen << "\n";
    os << "  q blocks nonempty sjC      = " << support.q_blocks_nonempty_sjC << "\n";
    os << "  q blocks with T2           = " << support.q_blocks_with_T2 << "\n";
    os << "  q blocks with T4           = " << support.q_blocks_with_T4 << "\n";
    os << "  q blocks accumulated       = " << support.q_blocks_accumulated << "\n";
    os << "  CPU q-orbit reduction      = "
       << (used_cpu_q_reflection_orbits ? "enabled" : "logical fallback")
       << "\n";
    os << "  CPU logical/orbit q        = "
       << cpu_q_logical_members_covered << " / "
       << cpu_q_orbits_executed
       << " (paired/self/unpaired="
       << cpu_q_paired_orbits_executed << "/"
       << cpu_q_self_orbits_executed << "/"
       << cpu_q_unpaired_blocks_executed << ")\n";
    os << "  CPU final BLAS/tiled calls = "
       << cpu_vendor_blas_final_calls << " / "
       << cpu_tiled_final_calls << "\n";

    const double avg_nk3 = support.q_blocks_nonempty_sjC > 0
        ? static_cast<double>(support.sum_nk3) / static_cast<double>(support.q_blocks_nonempty_sjC)
        : 0.0;
    const double avg_nk4 = support.q_blocks_nonempty_sjC > 0
        ? static_cast<double>(support.sum_nk4) / static_cast<double>(support.q_blocks_nonempty_sjC)
        : 0.0;

    os << "  avg/max nk3                = " << std::fixed << std::setprecision(3)
       << avg_nk3 << " / " << support.max_nk3 << "\n";
    os << "  avg/max nk4                = " << std::fixed << std::setprecision(3)
       << avg_nk4 << " / " << support.max_nk4 << "\n";

    os << "\nPanel densities\n";
    os << "---------------\n";
    os << "  T1 seed nnz / total        = " << support.T1_seed_values_nnz << " / "
       << support.T1_seed_values_total << " = " << std::fixed << std::setprecision(4)
       << density(support.T1_seed_values_nnz, support.T1_seed_values_total) << "\n";
    os << "  T3 seed nnz / total        = " << support.T3_seed_values_nnz << " / "
       << support.T3_seed_values_total << " = " << std::fixed << std::setprecision(4)
       << density(support.T3_seed_values_nnz, support.T3_seed_values_total) << "\n";
    os << "  T2 stack nnz / total       = " << support.T2_values_nnz << " / "
       << support.T2_values_total << " = " << std::fixed << std::setprecision(4)
       << density(support.T2_values_nnz, support.T2_values_total) << "\n";
    os << "  T4 stack nnz / total       = " << support.T4_values_nnz << " / "
       << support.T4_values_total << " = " << std::fixed << std::setprecision(4)
       << density(support.T4_values_nnz, support.T4_values_total) << "\n";
    os << "  CT2 stack nnz / total      = " << support.CT2_values_nnz << " / "
       << support.CT2_values_total << " = " << std::fixed << std::setprecision(4)
       << density(support.CT2_values_nnz, support.CT2_values_total) << "\n";

    os << "\nTerm counters\n";
    os << "-------------\n";
    os << "  T1 triple terms            = " << support.T1_triple_terms << "\n";
    os << "  T3 triple terms            = " << support.T3_triple_terms << "\n";
    os << "  T2 dot terms               = " << support.T2_dot_terms << "\n";
    os << "  T4 dot terms               = " << support.T4_dot_terms << "\n";
    os << "  CT2 rank-1 row updates     = " << support.CT2_rank1_updates << "\n";
    os << "  final dot terms            = " << support.final_dot_terms << "\n";
}

void FullTensorPackedContractionL25ScalarWorkspace::ensure_seed_caches(const FullTensorInputSupportPlan& support)
{
    const std::size_t I2  = support.channels.i2.size();
    const std::size_t I5  = support.channels.i5.size();
    const std::size_t K3d = support.channels.k3.size();
    const std::size_t K4d = support.channels.k4.size();
    const std::size_t L23 = support.l23.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L45 = support.l45.size();

    T1_cache.resize(L25 * L35 * K4d * L45 * I5);
    T3_cache.resize(L25 * L24 * L23 * K3d * I2);
}

FullTensorReflectionCompactWeightedProgressBar::FullTensorReflectionCompactWeightedProgressBar(
        std::uint64_t total_work,
        FullTensorReflectionCompactWeightedProgressOptions options)
    : total_work_(total_work),
          options_(std::move(options)),
          done_(0),
          eta_done_(0),
          started_(false),
          interactive_(false),
          owns_interactive_line_(false)
{
    if (options_.update_granularity == 0) {
        options_.update_granularity = 1;
    }
    if (options_.refresh_seconds <= 0.0) {
        options_.refresh_seconds = 1.0;
    }
    if (options_.stream == nullptr) {
        options_.stream = &std::cerr;
    }
    if (options_.bar_width == 0) {
        options_.bar_width = 22;
    }
    if (options_.eta_smoothing <= 0.0 || options_.eta_smoothing > 1.0) {
        options_.eta_smoothing = 0.20;
    }
    if (options_.eta_min_elapsed_seconds < 0.0) {
        options_.eta_min_elapsed_seconds = 0.0;
    }
    if (options_.eta_min_fraction < 0.0) {
        options_.eta_min_fraction = 0.0;
    }
    if (options_.eta_max_display_seconds <= 0.0) {
        options_.eta_max_display_seconds = 30.0 * 24.0 * 3600.0;
    }
    if (options_.unit_label.empty()) {
        options_.unit_label = "work";
    }
}

void FullTensorReflectionCompactWeightedProgressBar::start() noexcept
{
    std::lock_guard lifecycle_lock(lifecycle_mutex_);
    if (!options_.enabled || started_) {
        return;
    }
    try {
        done_.store(0, std::memory_order_relaxed);
        eta_done_.store(0, std::memory_order_relaxed);
        {
            std::lock_guard wake_lock(wake_mutex_);
            stop_requested_ = false;
        }
        start_time_ = clock::now();
        rate_start_time_ = start_time_;
        eta_estimator_.reset();
        next_plain_milestone_ = 1;
        last_plain_reported_done_ = 0;
        last_callback_reported_done_ = 0;
        attempt_ = 1;
        last_interactive_width_ = 0;
        interactive_ = options_.render_output &&
            options_.presentation.interactive;
        owns_interactive_line_ = false;
        if (interactive_) {
            owns_interactive_line_ =
                sl2cfoamxx::detail::try_acquire_interactive_progress();
            interactive_ = owns_interactive_line_;
        }
        monitor_ = std::thread([this]() { this->monitor_loop(); });
        started_ = true;
        print_initial();
    } catch (...) {
        release_interactive_line();
        started_ = false;
        options_.enabled = false;
    }
}

void FullTensorReflectionCompactWeightedProgressBar::reset() noexcept
{
    std::lock_guard lifecycle_lock(lifecycle_mutex_);
    if (!started_) return;
    std::lock_guard render_lock(render_mutex_);
    try {
        const auto now = clock::now();
        done_.store(0, std::memory_order_relaxed);
        eta_done_.store(0, std::memory_order_relaxed);
        rate_start_time_ = now;
        eta_estimator_.reset();
        next_plain_milestone_ = 1;
        last_plain_reported_done_ = 0;
        last_callback_reported_done_ = 0;
        ++attempt_;

        const Snapshot current{
            .done = 0,
            .elapsed = std::chrono::duration<double>(
                now - start_time_).count(),
        };
        emit_callback(current, sl2cfoamxx::ProgressStatus::restarted);
        if (!options_.render_output) return;
        if (interactive_) {
            print_interactive_line(current, false, false);
        } else {
            print_plain_line(current, false, false);
        }
    } catch (...) {
        // A progress reset is informational and must never make a
        // recoverable backend retry fail.
    }
}

void FullTensorReflectionCompactWeightedProgressBar::stop(bool completed) noexcept
{
    std::lock_guard lifecycle_lock(lifecycle_mutex_);
    if (!started_) {
        return;
    }

    {
        std::lock_guard wake_lock(wake_mutex_);
        stop_requested_ = true;
    }
    wake_.notify_all();
    if (monitor_.joinable()) {
        monitor_.join();
    }
    print_final(completed);
    release_interactive_line();
    started_ = false;
}

void FullTensorReflectionCompactWeightedProgressBar::monitor_loop() noexcept
{
    std::unique_lock lock(wake_mutex_);
    while (!stop_requested_) {
        const bool stopped = wake_.wait_for(
            lock,
            std::chrono::duration<double>(options_.refresh_seconds),
            [this] {
                return stop_requested_;
            });
        if (stopped) break;
        lock.unlock();
        print_update();
        lock.lock();
    }
}

[[nodiscard]] FullTensorReflectionCompactWeightedProgressBar::Snapshot FullTensorReflectionCompactWeightedProgressBar::snapshot(bool force_done) noexcept
{
    const auto now = clock::now();
    const double elapsed = std::chrono::duration<double>(now - start_time_).count();
    const double rate_elapsed =
        std::chrono::duration<double>(now - rate_start_time_).count();

    std::uint64_t done = done_.load(std::memory_order_relaxed);
    if (force_done) {
        done = total_work_;
    }
    if (done > total_work_) {
        done = total_work_;
    }
    if (force_done) {
        return Snapshot{done, elapsed, 0.0, false};
    }
    const std::uint64_t independent_eta_total =
        eta_total_work_.load(std::memory_order_relaxed);
    const std::uint64_t eta_total = independent_eta_total != 0
        ? independent_eta_total
        : total_work_;
    const std::uint64_t eta_done = independent_eta_total != 0
        ? std::min(
              eta_done_.load(std::memory_order_relaxed),
              independent_eta_total)
        : done;
    const sl2cfoamxx::detail::ProgressEtaEstimate eta =
        eta_estimator_.estimate(
            eta_done,
            eta_total,
            rate_elapsed,
            options_.eta_min_elapsed_seconds,
            options_.eta_min_fraction,
            options_.eta_smoothing,
            options_.eta_max_display_seconds);
    return Snapshot{done, elapsed, eta.seconds, eta.available};
}

[[nodiscard]] std::string FullTensorReflectionCompactWeightedProgressBar::transient_prefix() const
{
    const char* separator = options_.presentation.utf8 ? " │ " : " | ";
    const std::size_t terminal_columns = transient_terminal_columns();
    const std::size_t columns = terminal_columns > 0
        ? terminal_columns - 1
        : 0;
    constexpr std::size_t minimum_progress_width = 24;
    const auto fits = [&](const std::string& prefix) {
        return display_width(prefix) + minimum_progress_width <= columns;
    };

    std::string full(sl2cfoamxx::detail::product_display_name);
    if (!options_.operation_label.empty()) {
        full += separator;
        full += options_.operation_label;
    }
    if (!options_.backend.empty()) {
        full += separator;
        full += options_.backend;
    }
    full += separator;
    if (fits(full)) return full;

    std::string compact;
    if (!options_.operation_label.empty()) {
        compact += options_.operation_label;
    }
    if (!options_.backend.empty()) {
        if (!compact.empty()) compact += separator;
        compact += options_.backend;
    }
    if (!compact.empty()) compact += separator;
    if (fits(compact)) return compact;

    std::string backend;
    if (!options_.backend.empty()) {
        backend = options_.backend;
        backend += separator;
    }
    return fits(backend) ? backend : std::string{};
}

[[nodiscard]] std::string FullTensorReflectionCompactWeightedProgressBar::render_interactive_line_text(
        const Snapshot& current,
        bool completed_successfully,
        bool stopped) const
{
    sl2cfoamxx::detail::OutputPresentation presentation =
        options_.presentation;
    std::string prefix;
    if (uses_transient_live_line()) {
        prefix = transient_prefix();
        // Reserve the terminal's last column so rewriting or clearing the
        // row cannot trigger automatic line wrapping. Unknown widths use
        // the conservative fallback from transient_terminal_columns().
        const std::size_t terminal_columns = transient_terminal_columns();
        const std::size_t usable_columns = terminal_columns > 0
            ? terminal_columns - 1
            : 0;
        const std::size_t prefix_width = display_width(prefix);
        presentation.columns = usable_columns > prefix_width
            ? usable_columns - prefix_width
            : 1;
    }

    std::string line = sl2cfoamxx::detail::render_interactive_progress(
        render_state(current, completed_successfully, stopped),
        presentation,
        options_.bar_width);
    if (uses_transient_live_line()) {
        const std::size_t content = line.find_first_not_of(' ');
        if (content == std::string::npos) {
            line.clear();
        } else if (content != 0) {
            line.erase(0, content);
        }
        line.insert(0, prefix);
    }
    return line;
}

[[nodiscard]] sl2cfoamxx::detail::ProgressRenderState FullTensorReflectionCompactWeightedProgressBar::render_state(
        const Snapshot& current,
        bool completed_successfully,
        bool stopped) const noexcept
{
    return sl2cfoamxx::detail::ProgressRenderState{
        .label = options_.label,
        .unit_label = public_unit_label(),
        .completed = current.done,
        .total = total_work_,
        .elapsed_seconds = current.elapsed,
        .eta_seconds = current.eta,
        .eta_available = current.eta_available,
        .completed_successfully = completed_successfully,
        .stopped = stopped,
    };
}

void FullTensorReflectionCompactWeightedProgressBar::emit_callback(
        const Snapshot& current,
        sl2cfoamxx::ProgressStatus status) noexcept
{
    if (!options_.callback) return;
    try {
        options_.callback(sl2cfoamxx::ProgressEvent{
            .operation = options_.operation,
            .phase = options_.phase,
            .backend = options_.backend_kind,
            .status = status,
            .attempt = attempt_,
            .completed = current.done,
            .total = total_work_,
            .elapsed_seconds = current.elapsed,
            .eta_seconds = current.eta,
            .eta_available = current.eta_available,
        });
    } catch (...) {
        // User callbacks are informational and cannot fail a computation.
    }
}

void FullTensorReflectionCompactWeightedProgressBar::print_initial() noexcept
{
    std::lock_guard render_lock(render_mutex_);
    try {
        const Snapshot current = snapshot(false);
        emit_callback(current, sl2cfoamxx::ProgressStatus::started);
        if (!options_.render_output) return;
        if (interactive_) {
            if (uses_transient_live_line()) {
                const std::string line = render_interactive_line_text(
                    current, false, false);
                last_interactive_width_ = display_width(line);
                std::lock_guard output_lock(
                    sl2cfoamxx::detail::informational_output_mutex());
                (*options_.stream) << '\r' << line << std::flush;
                return;
            }
            std::ostringstream header;
            const char* separator = options_.presentation.utf8
                ? " │ "
                : " | ";
            header << sl2cfoamxx::detail::product_display_name << separator
                   << options_.operation_label;
            if (!options_.backend.empty()) {
                header << separator << options_.backend;
            }
            header << separator << phase_label() << '\n';

            const std::string line = render_interactive_line_text(
                current, false, false);
            last_interactive_width_ = display_width(line);
            std::lock_guard output_lock(
                sl2cfoamxx::detail::informational_output_mutex());
            (*options_.stream) << header.str() << '\r' << line
                               << std::flush;
        } else {
            print_plain_line(current, false, false);
        }
    } catch (...) {
        // Progress is informational and must never fail a computation,
        // including when a caller-supplied stream enables exceptions.
    }
}

void FullTensorReflectionCompactWeightedProgressBar::print_update() noexcept
{
    std::lock_guard render_lock(render_mutex_);
    try {
        const Snapshot current = snapshot(false);
        if (current.done != last_callback_reported_done_) {
            last_callback_reported_done_ = current.done;
            emit_callback(current, sl2cfoamxx::ProgressStatus::running);
        }
        if (!options_.render_output) return;
        if (interactive_) {
            print_interactive_line(current, false, false);
            return;
        }

        const bool crossed_milestone =
            full_tensor_progress_advance_milestones(
                current.done, total_work_, next_plain_milestone_);
        if (!crossed_milestone ||
            current.done == last_plain_reported_done_) {
            return;
        }
        last_plain_reported_done_ = current.done;
        print_plain_line(current, false, false);
    } catch (...) {
        // Progress is informational and must never fail a computation.
    }
}

void FullTensorReflectionCompactWeightedProgressBar::print_final(bool completed_successfully) noexcept
{
    std::lock_guard render_lock(render_mutex_);
    try {
        const Snapshot current = snapshot(completed_successfully);
        emit_callback(
            current,
            completed_successfully
                ? sl2cfoamxx::ProgressStatus::completed
                : sl2cfoamxx::ProgressStatus::stopped);
        if (!options_.render_output) return;
        if (interactive_) {
            if (completed_successfully && uses_transient_live_line()) {
                clear_interactive_line();
                return;
            }
            print_interactive_line(
                current,
                completed_successfully,
                !completed_successfully,
                true);
        } else {
            print_plain_line(
                current,
                completed_successfully,
                !completed_successfully);
        }
    } catch (...) {
        // Progress is informational and must never fail a computation,
        // including when a caller-supplied stream enables exceptions.
    }
}

void FullTensorReflectionCompactWeightedProgressBar::print_interactive_line(
        const Snapshot& current,
        bool completed_successfully,
        bool stopped,
        bool terminate_line)
{
    const std::string line = render_interactive_line_text(
        current, completed_successfully, stopped);
    const std::size_t width = display_width(line);
    const std::size_t padding = last_interactive_width_ > width
        ? last_interactive_width_ - width
        : 0;
    last_interactive_width_ = width;

    std::lock_guard output_lock(
        sl2cfoamxx::detail::informational_output_mutex());
    (*options_.stream) << '\r' << line << std::string(padding, ' ');
    if (terminate_line) (*options_.stream) << '\n';
    (*options_.stream) << std::flush;
}

void FullTensorReflectionCompactWeightedProgressBar::print_plain_line(
        const Snapshot& current,
        bool completed_successfully,
        bool stopped)
{
    const std::string line = sl2cfoamxx::detail::render_plain_progress(
        render_state(current, completed_successfully, stopped));
    std::lock_guard output_lock(
        sl2cfoamxx::detail::informational_output_mutex());
    if (options_.presentation.terminal_control &&
        sl2cfoamxx::detail::interactive_progress_is_active()) {
        const std::size_t columns = options_.presentation.columns != 0
            ? options_.presentation.columns
            : 120;
        (*options_.stream) << '\r' << std::string(columns, ' ') << '\r';
    }
    (*options_.stream) << line << '\n' << std::flush;
}

void FullTensorTriangularBatchedLRWorkspace::reset(std::size_t p_capacity, std::size_t M_)
{
    p_cap = std::max<std::size_t>(1, p_capacity);
    p_used = 0;
    M = M_;
    // Pad to a multiple of 8 so all currently supported tile shapes
    // (4x4, 4x8, 8x4) can blindly load their full tile without reading
    // past the padded column storage.
    M_pad = full_tensor_triangular_round_up_multiple(M, 8);

    // Use assign to guarantee the padded areas are exactly 0.0
    L.assign(p_cap * M_pad, 0.0);
    Rperm.assign(p_cap * M_pad, 0.0);
}

void FullTensorTriangularBatchedLRWorkspace::skip_planned_rows(std::uint64_t rows) noexcept
{
    if (progress == nullptr || rows == 0) return;
    const std::uint64_t maximum =
        std::numeric_limits<std::uint64_t>::max();
    skipped_rows_pending = maximum - skipped_rows_pending < rows
        ? maximum
        : skipped_rows_pending + rows;
    skipped_rows_total = maximum - skipped_rows_total < rows
        ? maximum
        : skipped_rows_total + rows;
    if (progress != nullptr &&
        skipped_rows_pending >= static_cast<std::uint64_t>(p_cap)) {
        progress->add(skipped_rows_pending);
        skipped_rows_pending = 0;
    }
}

void FullTensorReassociatedRectangularBatchedLRWorkspace::reset(std::size_t p_capacity, std::size_t M_, std::size_t N_)
{
    p_cap = std::max<std::size_t>(1, p_capacity);
    p_used = 0;
    M = M_;
    N = N_;
    M_pad = full_tensor_triangular_round_up_multiple(M, 4);
    N_pad = full_tensor_triangular_round_up_multiple(N, 4);

    // assign(), rather than resize(), deliberately zeros the padded columns.
    // The final tile kernel can then read those columns unconditionally.
    L.assign(p_cap * M_pad, 0.0);
    R.assign(p_cap * N_pad, 0.0);
}

void FullTensorReassociatedRectangularBatchedLRWorkspace::skip_planned_rows(std::uint64_t rows) noexcept
{
    if (progress == nullptr || rows == 0) return;
    const std::uint64_t maximum =
        std::numeric_limits<std::uint64_t>::max();
    skipped_rows_pending = maximum - skipped_rows_pending < rows
        ? maximum
        : skipped_rows_pending + rows;
    skipped_rows_total = maximum - skipped_rows_total < rows
        ? maximum
        : skipped_rows_total + rows;
    if (progress != nullptr &&
        skipped_rows_pending >= static_cast<std::uint64_t>(p_cap)) {
        progress->add(skipped_rows_pending);
        skipped_rows_pending = 0;
    }
}

void FullTensorScheduledBalanceProfile::reset(std::size_t I1_, std::size_t X_, std::size_t workers_, std::size_t n_tasks)
{
    I1 = I1_;
    X = X_;
    workers = workers_;
    selected_p_batch = 0;
    selected_final_tile_shape =
        FullTensorTriangularFinalTileShape::Tile4x4;
    selected_final_tile_applicable = false;
    whole_i1_tasks = 0;
    split_i1_panels = 0;
    selected_mode = FullTensorAdaptiveScheduleMode::Auto;
    auto_selected_i1_only = false;
    i1_tasks_per_worker = 0.0;
    predicted_i1_max_over_mean = 0.0;
    predicted_i1_max_over_target = 0.0;
    predicted_i1_max_work = 0.0;
    heavy_i1_forced_split = false;
    inner_tile_parallel_requested = false;
    inner_tile_parallel_enabled = false;
    used_thread_local_i1_panels = false;
    tls_panels_touched = 0;
    tls_output_bytes = 0;
    tls_reduction_seconds = 0.0;
    target_task_work = 0.0;
    total_predicted_work = 0.0;
    tasks.assign(n_tasks, FullTensorScheduledTaskRecord{});
}

void FullTensorScheduledBalanceProfile::print(std::ostream& os, std::size_t top_n) const
{
    double min_s = std::numeric_limits<double>::infinity();
    double max_s = 0.0;
    double sum_s = 0.0;
    double sum2_s = 0.0;
    std::size_t nonzero = 0;
    for (const auto& r : tasks) {
        if (r.seconds <= 0.0) continue;
        ++nonzero;
        min_s = std::min(min_s, r.seconds);
        max_s = std::max(max_s, r.seconds);
        sum_s += r.seconds;
        sum2_s += r.seconds * r.seconds;
    }
    if (nonzero == 0) min_s = 0.0;
    const double n = static_cast<double>(nonzero);
    const double mean = n > 0.0 ? sum_s / n : 0.0;
    const double var = n > 0.0 ? std::max(0.0, sum2_s / n - mean * mean) : 0.0;
    const double sd = std::sqrt(var);
    const double imbalance = mean > 0.0 ? max_s / mean : 0.0;

    os << "\nAdaptive scheduled task balance profile\n";
    os << "=======================================\n";
    os << "I1, X, workers          = " << I1 << ", " << X << ", " << workers << "\n";
    os << "tasks                  = " << tasks.size() << "\n";
    os << "selected P_batch       = " << selected_p_batch << "\n";
    os << "selected final tile    = "
       << (selected_final_tile_applicable
               ? full_tensor_triangular_final_tile_shape_name(
                     selected_final_tile_shape)
               : "not applicable (rectangular output)")
       << "\n";
    os << "whole-i1 tasks          = " << whole_i1_tasks << "\n";
    os << "split i1 panels         = " << split_i1_panels << "\n";
    os << "target predicted work   = " << std::scientific << std::setprecision(6)
       << target_task_work << "\n";
    os << "total predicted work    = " << std::scientific << std::setprecision(6)
       << total_predicted_work << "\n";
    os << "selected scheduler mode = "
       << full_tensor_adaptive_schedule_mode_name(selected_mode) << "\n";
    os << "auto picked i1-only     = " << (auto_selected_i1_only ? "YES" : "NO") << "\n";
    os << "I1/tasks per worker     = " << std::fixed << std::setprecision(3)
       << i1_tasks_per_worker << "\n";
    os << "pred i1 max/mean        = " << std::fixed << std::setprecision(3)
       << predicted_i1_max_over_mean << "\n";
    os << "pred i1 max/target      = " << std::fixed << std::setprecision(3)
       << predicted_i1_max_over_target << "\n";
    os << "pred i1 max work        = " << std::scientific << std::setprecision(6)
       << predicted_i1_max_work << "\n";
    os << "heavy-i1 split guard    = " << (heavy_i1_forced_split ? "triggered" : "not triggered") << "\n";
    os << "inner tile fallback     = "
       << (inner_tile_parallel_enabled ? "enabled" : (inner_tile_parallel_requested ? "requested, not implemented" : "not requested"))
       << "\n";
    os << "task seconds min/mean/max = " << std::fixed << std::setprecision(6)
       << min_s << " / " << mean << " / " << max_s << "\n";
    os << "task seconds stddev     = " << std::fixed << std::setprecision(6) << sd << "\n";
    os << "max/mean imbalance      = " << std::fixed << std::setprecision(3)
       << imbalance << "\n";
    os << "TLS i1 panels           = " << (used_thread_local_i1_panels ? "enabled" : "disabled") << "\n";
    os << "TLS panels touched      = " << tls_panels_touched << "\n";
    os << "TLS output memory       = " << std::fixed << std::setprecision(2)
       << (static_cast<double>(tls_output_bytes) / (1024.0 * 1024.0)) << " MiB\n";
    os << "TLS final reduction     = " << std::fixed << std::setprecision(6)
       << tls_reduction_seconds << " s\n";

    std::vector<double> i1_seconds(I1, 0.0);
    std::vector<double> i1_pred(I1, 0.0);
    std::vector<std::size_t> i1_tasks(I1, 0);
    std::vector<std::uint64_t> i1_final(I1, 0);
    std::vector<std::uint64_t> i1_q(I1, 0);
    for (const auto& r : tasks) {
        if (r.ii1 >= I1) continue;
        i1_seconds[r.ii1] += r.seconds;
        i1_pred[r.ii1] += r.predicted_work;
        i1_tasks[r.ii1] += 1;
        i1_final[r.ii1] += r.final_dot_terms;
        i1_q[r.ii1] += r.q_blocks_accumulated;
    }

    os << "\nPer-i1 scheduled cost\n";
    os << "---------------------\n";
    os << "  ii1   tasks      predicted      seconds      q_accum       final_terms\n";
    for (std::size_t ii1 = 0; ii1 < I1; ++ii1) {
        os << "  " << std::setw(3) << ii1
           << "  " << std::setw(6) << i1_tasks[ii1]
           << "  " << std::setw(13) << std::scientific << std::setprecision(4) << i1_pred[ii1]
           << "  " << std::setw(11) << std::fixed << std::setprecision(6) << i1_seconds[ii1]
           << "  " << std::setw(11) << i1_q[ii1]
           << "  " << std::setw(16) << i1_final[ii1]
           << "\n";
    }

    int max_thread = 0;
    for (const auto& r : tasks) max_thread = std::max(max_thread, r.thread_id);
    std::vector<double> thread_seconds(static_cast<std::size_t>(max_thread + 1), 0.0);
    std::vector<std::size_t> thread_tasks(static_cast<std::size_t>(max_thread + 1), 0);
    for (const auto& r : tasks) {
        const std::size_t tid = static_cast<std::size_t>(std::max(0, r.thread_id));
        if (tid < thread_seconds.size()) {
            thread_seconds[tid] += r.seconds;
            thread_tasks[tid] += 1;
        }
    }

    os << "\nPer-thread scheduled ownership\n";
    os << "------------------------------\n";
    os << "  tid     tasks      seconds\n";
    for (std::size_t tid = 0; tid < thread_seconds.size(); ++tid) {
        os << "  " << std::setw(3) << tid
           << "  " << std::setw(8) << thread_tasks[tid]
           << "  " << std::setw(12) << std::fixed << std::setprecision(6)
           << thread_seconds[tid] << "\n";
    }

    std::vector<std::size_t> order(tasks.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return tasks[a].seconds > tasks[b].seconds;
    });
    top_n = std::min<std::size_t>(top_n, order.size());
    os << "\nTop " << top_n << " slowest scheduled tasks\n";
    os << "--------------------------------\n";
    os << "  rank  ii1  x_begin  x_end   tid   whole      pred_work      seconds      q_accum       final_terms\n";
    for (std::size_t rank = 0; rank < top_n; ++rank) {
        const auto& r = tasks[order[rank]];
        os << "  " << std::setw(4) << rank
           << "  " << std::setw(3) << r.ii1
           << "  " << std::setw(7) << r.ix_begin
           << "  " << std::setw(5) << r.ix_end
           << "  " << std::setw(4) << r.thread_id
           << "  " << std::setw(5) << (r.whole_i1 ? "yes" : "no")
           << "  " << std::setw(13) << std::scientific << std::setprecision(4) << r.predicted_work
           << "  " << std::setw(11) << std::fixed << std::setprecision(6) << r.seconds
           << "  " << std::setw(11) << r.q_blocks_accumulated
           << "  " << std::setw(16) << r.final_dot_terms
           << "\n";
    }
}

const double* full_tensor_sjC_logical_block_base_l25_scalar(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const FullTensorSjCHalfBlockMap& map,
    std::vector<double>& C_block)
{
    const Range2D& r = map.logical;
    if (r.empty()) {
        return nullptr;
    }

    const Range2D& r_store = support.sjC.blocks[static_cast<std::size_t>(map.storage_block)];
    if (r_store.empty()) {
        return nullptr;
    }

    if (!map.transpose_from_storage) {
        return sj.sjC.data() + r_store.offset;
    }

    const std::size_t nk3 = static_cast<std::size_t>(r.i_end - r.i_begin);
    const std::size_t nk4 = static_cast<std::size_t>(r.j_end - r.j_begin);
    C_block.assign(nk3 * nk4, 0.0);

    const std::size_t n_store_i =
        static_cast<std::size_t>(r_store.i_end - r_store.i_begin);
    const double* base = sj.sjC.data() + r_store.offset;

    for (std::uint32_t ik4 = r.j_begin; ik4 < r.j_end; ++ik4) {
        const int mapped_i_signed = support.sjC_k4_to_k3[ik4];
        if (mapped_i_signed < 0) {
            continue;
        }
        const std::uint32_t mapped_i = static_cast<std::uint32_t>(mapped_i_signed);
        if (mapped_i < r_store.i_begin || mapped_i >= r_store.i_end) {
            continue;
        }

        for (std::uint32_t ik3 = r.i_begin; ik3 < r.i_end; ++ik3) {
            const int mapped_j_signed = support.sjC_k3_to_k4[ik3];
            if (mapped_j_signed < 0) {
                continue;
            }
            const std::uint32_t mapped_j = static_cast<std::uint32_t>(mapped_j_signed);
            if (mapped_j < r_store.j_begin || mapped_j >= r_store.j_end) {
                continue;
            }

            const std::size_t local_i = static_cast<std::size_t>(ik3 - r.i_begin);
            const std::size_t local_j = static_cast<std::size_t>(ik4 - r.j_begin);
            const std::size_t store_i = static_cast<std::size_t>(mapped_i - r_store.i_begin);
            const std::size_t store_j = static_cast<std::size_t>(mapped_j - r_store.j_begin);

            C_block[local_i + nk3 * local_j] =
                base[store_i + n_store_i * store_j];
        }
    }

    return C_block.data();
}

void build_full_T1_seed_panel_scalar_profiled(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    std::size_t ii1,
    std::size_t ix,
    std::size_t il25,
    std::size_t il35,
    double* B4FULL_RESTRICT seed,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt)
{
    FullTensorScalarScopedTimer timer(profile.T1_seed);

    const std::size_t X   = support.channels.xvals.size();
    const std::size_t I5  = support.channels.i5.size();
    const std::size_t K4d = support.channels.k4.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L45 = support.l45.size();

    assert(seed != nullptr);
    full_scalar_profile_zero(seed, K4d * L45 * I5);
    profile.support.T1_seed_values_total += static_cast<std::uint64_t>(K4d * L45 * I5);

    const std::size_t sjA_block = idx3(ii1, ix, il25, X, L25);
    const Range1D& r_sjA = support.sjA.blocks[sjA_block];
    if (r_sjA.empty()) {
        return;
    }

    const double* B4FULL_RESTRICT sjA_base = sj.sjA.data() + r_sjA.offset;

    std::uint64_t local_terms = 0;
    std::uint64_t local_nnz = 0;

    for (std::size_t ii5 = 0; ii5 < I5; ++ii5) {
        for (std::size_t ik4 = 0; ik4 < K4d; ++ik4) {
            for (std::size_t il45 = 0; il45 < L45; ++il45) {
                const std::size_t sjB_block = idx4(ix, ik4, il35, il45, K4d, L35, L45);
                const Range1D& r_sjB = support.sjB.blocks[sjB_block];
                if (r_sjB.empty()) {
                    continue;
                }

                const std::size_t B5_block = idx4(ii5, il25, il35, il45, L25, L35, L45);
                const Range1D& r_B5 = support.B5.blocks[B5_block];
                if (r_B5.empty()) {
                    continue;
                }

                const double* B4FULL_RESTRICT sjB_base = sj.sjB.data() + r_sjB.offset;
                const double* B4FULL_RESTRICT B5_base  = B.B5_data()  + r_B5.offset;

                double s = full_scalar_profile_triple_dot_intersection(
                    r_sjA, sjA_base,
                    r_sjB, sjB_base,
                    r_B5,  B5_base,
                    &local_terms);

                seed[il45 + L45 * (ik4 + K4d * ii5)] = s;
                if (opt.count_seed_nnz && s != 0.0) {
                    ++local_nnz;
                }
            }
        }
    }

    profile.support.T1_triple_terms += local_terms;
    profile.support.T1_seed_values_nnz += local_nnz;
    profile.T1_seed.approx_flops += 4.0L * static_cast<long double>(local_terms);
}

void build_full_T3_seed_panel_scalar_profiled(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    std::size_t ii1,
    std::size_t ix,
    std::size_t il24,
    std::size_t il25,
    double* B4FULL_RESTRICT seed,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt)
{
    FullTensorScalarScopedTimer timer(profile.T3_seed);

    const std::size_t X   = support.channels.xvals.size();
    const std::size_t I2  = support.channels.i2.size();
    const std::size_t K3d = support.channels.k3.size();
    const std::size_t L23 = support.l23.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();

    assert(seed != nullptr);
    full_scalar_profile_zero(seed, L23 * K3d * I2);
    profile.support.T3_seed_values_total += static_cast<std::uint64_t>(L23 * K3d * I2);

    const std::size_t sjE_block = idx3(ii1, ix, il25, X, L25);
    const Range1D& r_sjE = support.sjE.blocks[sjE_block];
    if (r_sjE.empty()) {
        return;
    }

    const double* B4FULL_RESTRICT sjE_base = sj.sjE.data() + r_sjE.offset;

    std::uint64_t local_terms = 0;
    std::uint64_t local_nnz = 0;

    for (std::size_t ii2 = 0; ii2 < I2; ++ii2) {
        for (std::size_t ik3 = 0; ik3 < K3d; ++ik3) {
            for (std::size_t il23 = 0; il23 < L23; ++il23) {
                const std::size_t sjD_block = idx4(ix, ik3, il23, il24, K3d, L23, L24);
                const Range1D& r_sjD = support.sjD.blocks[sjD_block];
                if (r_sjD.empty()) {
                    continue;
                }

                const std::size_t B2_block = idx4(ii2, il23, il24, il25, L23, L24, L25);
                const Range1D& r_B2 = support.B2.blocks[B2_block];
                if (r_B2.empty()) {
                    continue;
                }

                const double* B4FULL_RESTRICT sjD_base = sj.sjD.data() + r_sjD.offset;
                const double* B4FULL_RESTRICT B2_base  = B.B2_data()  + r_B2.offset;

                double s = full_scalar_profile_triple_dot_intersection(
                    r_sjE, sjE_base,
                    r_sjD, sjD_base,
                    r_B2,  B2_base,
                    &local_terms);

                seed[il23 + L23 * (ik3 + K3d * ii2)] = s;
                if (opt.count_seed_nnz && s != 0.0) {
                    ++local_nnz;
                }
            }
        }
    }

    profile.support.T3_triple_terms += local_terms;
    profile.support.T3_seed_values_nnz += local_nnz;
    profile.T3_seed.approx_flops += 4.0L * static_cast<long double>(local_terms);
}

void build_full_seed_caches_l25_scalar_profiled(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    std::size_t ii1,
    std::size_t ix,
    FullTensorPackedContractionL25ScalarWorkspace& ws,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt)
{
    const std::size_t I2  = support.channels.i2.size();
    const std::size_t I5  = support.channels.i5.size();
    const std::size_t K3d = support.channels.k3.size();
    const std::size_t K4d = support.channels.k4.size();
    const std::size_t L23 = support.l23.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L45 = support.l45.size();

    const std::size_t T1_seed_size = K4d * L45 * I5;
    const std::size_t T3_seed_size = L23 * K3d * I2;

    for (std::size_t il25 = 0; il25 < L25; ++il25) {
        for (std::size_t il35 = 0; il35 < L35; ++il35) {
            double* B4FULL_RESTRICT T1_seed =
                ws.T1_cache.data() + (il25 * L35 + il35) * T1_seed_size;

            build_full_T1_seed_panel_scalar_profiled(
                support, sj, B, ii1, ix, il25, il35,
                T1_seed, profile, opt);
        }

        for (std::size_t il24 = 0; il24 < L24; ++il24) {
            double* B4FULL_RESTRICT T3_seed =
                ws.T3_cache.data() + (il25 * L24 + il24) * T3_seed_size;

            build_full_T3_seed_panel_scalar_profiled(
                support, sj, B, ii1, ix, il24, il25,
                T3_seed, profile, opt);
        }
    }
}

void scatter_full_output_by_i1_to_canonical_scalar_profiled(
    const FullTensorInputSupportPlan& support,
    const double* B4FULL_RESTRICT A_by_i1,
    double* B4FULL_RESTRICT A_canonical)
{
    const std::size_t I1 = support.channels.i1.size();
    const std::size_t I2 = support.channels.i2.size();
    const std::size_t I3 = support.channels.i3.size();
    const std::size_t I4 = support.channels.i4.size();
    const std::size_t I5 = support.channels.i5.size();

    const std::size_t M = I2 * I3;
    const std::size_t N = I4 * I5;

#ifdef _OPENMP
#pragma omp parallel for collapse(5) schedule(static)
#endif
    for (std::ptrdiff_t ii5_s = 0; ii5_s < static_cast<std::ptrdiff_t>(I5); ++ii5_s) {
        for (std::size_t ii4 = 0; ii4 < I4; ++ii4) {
            for (std::size_t ii3 = 0; ii3 < I3; ++ii3) {
                for (std::size_t ii2 = 0; ii2 < I2; ++ii2) {
                    for (std::size_t ii1 = 0; ii1 < I1; ++ii1) {
                        const std::size_t row = ii2 + I2 * ii3;
                        const std::size_t col = ii4 + I4 * static_cast<std::size_t>(ii5_s);
                        const std::size_t src = row + M * (col + N * ii1);
                        const std::size_t dst =
                            ii1 + I1 * (ii2 + I2 * (ii3 + I3 * (ii4 + I4 * static_cast<std::size_t>(ii5_s))));
                        A_canonical[dst] = A_by_i1[src];
                    }
                }
            }
        }
    }
}

FullTensorReflectionOutputMap build_full_tensor_reflection_output_map(
    const FullTensorInputSupportPlan& support)
{
    FullTensorReflectionOutputMap map;

    const auto& i2 = support.channels.i2;
    const auto& i3 = support.channels.i3;
    const auto& i4 = support.channels.i4;
    const auto& i5 = support.channels.i5;

    map.I2 = i2.size();
    map.I3 = i3.size();
    map.I4 = i4.size();
    map.I5 = i5.size();
    map.M  = map.I2 * map.I3;
    map.N  = map.I4 * map.I5;

    map.rep_index.assign(map.M * map.N, 0);

    if (map.M == 0 || map.N == 0) {
        map.compatible = true;
        return map;
    }

    // The reflected tuple is
    //   (i1,i2,i3,i4,i5) -> (i1,i5,i4,i3,i2).
    // Therefore every i5 value must exist in channel i2, every i4 value in i3,
    // every i3 value in i4, and every i2 value in i5.
    for (int v : i5) {
        if (full_tensor_reflection_find_value_index(i2, v) < 0) {
            map.compatible = false;
            return map;
        }
    }
    for (int v : i4) {
        if (full_tensor_reflection_find_value_index(i3, v) < 0) {
            map.compatible = false;
            return map;
        }
    }
    for (int v : i3) {
        if (full_tensor_reflection_find_value_index(i4, v) < 0) {
            map.compatible = false;
            return map;
        }
    }
    for (int v : i2) {
        if (full_tensor_reflection_find_value_index(i5, v) < 0) {
            map.compatible = false;
            return map;
        }
    }

    map.representatives.reserve((map.M * map.N + 1) / 2);

    for (std::size_t ii5 = 0; ii5 < map.I5; ++ii5) {
        for (std::size_t ii4 = 0; ii4 < map.I4; ++ii4) {
            for (std::size_t ii3 = 0; ii3 < map.I3; ++ii3) {
                for (std::size_t ii2 = 0; ii2 < map.I2; ++ii2) {
                    const std::size_t m = ii2 + map.I2 * ii3;
                    const std::size_t n = ii4 + map.I4 * ii5;
                    const std::size_t flat = m + map.M * n;

                    const int ji2_s = full_tensor_reflection_find_value_index(i2, i5[ii5]);
                    const int ji3_s = full_tensor_reflection_find_value_index(i3, i4[ii4]);
                    const int ji4_s = full_tensor_reflection_find_value_index(i4, i3[ii3]);
                    const int ji5_s = full_tensor_reflection_find_value_index(i5, i2[ii2]);

                    if (ji2_s < 0 || ji3_s < 0 || ji4_s < 0 || ji5_s < 0) {
                        map.compatible = false;
                        return map;
                    }

                    const std::size_t ji2 = static_cast<std::size_t>(ji2_s);
                    const std::size_t ji3 = static_cast<std::size_t>(ji3_s);
                    const std::size_t ji4 = static_cast<std::size_t>(ji4_s);
                    const std::size_t ji5 = static_cast<std::size_t>(ji5_s);

                    const std::size_t pm = ji2 + map.I2 * ji3;
                    const std::size_t pn = ji4 + map.I4 * ji5;
                    const std::size_t pflat = pm + map.M * pn;

                    const std::size_t rflat = std::min(flat, pflat);
                    map.rep_index[flat] = rflat;

                    if (flat <= pflat) {
                        map.representatives.push_back(FullTensorReflectionPair{m, n, pm, pn});
                    }
                }
            }
        }
    }

    map.compatible = true;
    return map;
}

FullTensorReflectionCompactOutputMap
build_full_tensor_reflection_compact_output_map(
    const FullTensorInputSupportPlan& support)
{
    FullTensorReflectionCompactOutputMap cmap;
    cmap.full_map = build_full_tensor_reflection_output_map(support);
    cmap.compatible = cmap.full_map.compatible;

    if (!cmap.compatible) {
        return cmap;
    }

    const std::size_t full = cmap.full_map.full_entries();
    cmap.compact_rep_of_full.assign(full, static_cast<std::size_t>(-1));

    for (std::size_t rep = 0; rep < cmap.full_map.representatives.size(); ++rep) {
        const FullTensorReflectionPair& p = cmap.full_map.representatives[rep];
        const std::size_t flat0 = p.m + cmap.full_map.M * p.n;
        const std::size_t flat1 = p.partner_m + cmap.full_map.M * p.partner_n;

        cmap.compact_rep_of_full[flat0] = rep;
        cmap.compact_rep_of_full[flat1] = rep;
    }

    // Defensive check: every full entry must be represented exactly by one orbit.
    for (std::size_t flat = 0; flat < full; ++flat) {
        if (cmap.compact_rep_of_full[flat] == static_cast<std::size_t>(-1)) {
            cmap.compatible = false;
            return cmap;
        }
    }

    return cmap;
}

void scatter_full_output_reflection_compact_by_i1_to_canonical(
    const FullTensorInputSupportPlan& support,
    const double* B4FULL_RESTRICT A_tri_by_i1,
    double* B4FULL_RESTRICT A_canonical)
{
    const FullTensorReflectionCompactOutputMap compact_map =
        build_full_tensor_reflection_compact_output_map(support);

    if (!compact_map.compatible || compact_map.M() != compact_map.N()) {
        throw b4utils::InternalFailure(
            "Cannot scatter triangular compact output: boundary intertwiner channels are not reflection-compatible.");
    }

    const auto& i2 = support.channels.i2;
    const auto& i3 = support.channels.i3;
    const auto& i4 = support.channels.i4;
    const auto& i5 = support.channels.i5;

    const std::size_t I1 = support.channels.i1.size();
    const std::size_t I2 = i2.size();
    const std::size_t I3 = i3.size();
    const std::size_t I4 = i4.size();
    const std::size_t M  = compact_map.M();
    const std::size_t N  = compact_map.N();
    const std::size_t tri_size = M * (M + 1u) / 2u;

    const auto upper_index = [](std::size_t i, std::size_t j) noexcept {
        if (i > j) std::swap(i, j);
        return j * (j + 1u) / 2u + i;
    };

    std::vector<std::size_t> n_to_s(N, static_cast<std::size_t>(-1));
    for (std::size_t n = 0; n < N; ++n) {
        const std::size_t ii4 = n % I4;
        const std::size_t ii5 = n / I4;
        const int ji2_s = full_tensor_reflection_find_value_index(i2, i5[ii5]);
        const int ji3_s = full_tensor_reflection_find_value_index(i3, i4[ii4]);
        if (ji2_s < 0 || ji3_s < 0) {
            throw b4utils::InternalFailure(
                "Invalid triangular compact reflected-column map.");
        }
        n_to_s[n] = static_cast<std::size_t>(ji2_s) + I2 * static_cast<std::size_t>(ji3_s);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t ii1_s = 0; ii1_s < static_cast<std::ptrdiff_t>(I1); ++ii1_s) {
        const std::size_t ii1 = static_cast<std::size_t>(ii1_s);
        const double* B4FULL_RESTRICT A_tri_panel = A_tri_by_i1 + tri_size * ii1;

        for (const FullTensorReflectionPair& p : compact_map.full_map.representatives) {
            const std::size_t s = n_to_s[p.n];
            const double v = A_tri_panel[upper_index(p.m, s)];

            const auto write_one = [&](std::size_t m, std::size_t n) {
                const std::size_t ii2 = m % I2;
                const std::size_t ii3 = m / I2;
                const std::size_t ii4 = n % I4;
                const std::size_t ii5 = n / I4;

                assert(ii3 < I3);
                assert(ii5 < i5.size());

                const std::size_t dst =
                    ii1 + I1 * (ii2 + I2 * (ii3 + I3 * (ii4 + I4 * ii5)));
                A_canonical[dst] = v;
            };

            write_one(p.m, p.n);
            if (p.m != p.partner_m || p.n != p.partner_n) {
                write_one(p.partner_m, p.partner_n);
            }
        }
    }
}

bool full_tensor_progress_advance_milestones(
    std::uint64_t completed,
    std::uint64_t total,
    std::size_t& next_milestone) noexcept
{
    bool crossed = false;
    while (next_milestone < 4) {
        const std::uint64_t quotient = total / 4;
        const std::uint64_t remainder = total % 4;
        const std::uint64_t threshold =
            quotient * next_milestone +
            (remainder * next_milestone + 3) / 4;
        if (completed < threshold) break;
        ++next_milestone;
        crossed = true;
    }
    return crossed;
}

#if defined(__linux__)
std::size_t full_tensor_triangular_parse_cache_size(std::string s)
{
    s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }), s.end());
    if (s.empty()) {
        return 0;
    }

    std::size_t mult = 1;
    const char suffix = s.back();
    if (suffix == 'K' || suffix == 'k') {
        mult = 1024u;
        s.pop_back();
    } else if (suffix == 'M' || suffix == 'm') {
        mult = 1024u * 1024u;
        s.pop_back();
    }

    if (s.empty()) {
        return 0;
    }
    return static_cast<std::size_t>(std::stoull(s)) * mult;
}
#endif

FullTensorReflectionCompactTriangularCacheInfo
full_tensor_detect_cache_info_triangular_batched()
{
    FullTensorReflectionCompactTriangularCacheInfo info;

#if defined(__APPLE__)
    info.detected_l1d = full_tensor_triangular_read_sysctl_size(
        "hw.l1dcachesize", info.l1d_bytes);
    info.detected_l2 = full_tensor_triangular_read_sysctl_size(
        "hw.l2cachesize", info.l2_bytes);
    info.detected_l3 = full_tensor_triangular_read_sysctl_size(
        "hw.l3cachesize", info.l3_bytes);

    // Apple Silicon exposes per-performance-level cache sizes on many macOS
    // versions.  The names are not guaranteed on every machine, so this is
    // deliberately best-effort.
    info.detected_apple_perf_l2 = full_tensor_triangular_read_sysctl_size(
        "hw.perflevel0.l2cachesize", info.apple_perf_l2_bytes);
    info.detected_apple_eff_l2 = full_tensor_triangular_read_sysctl_size(
        "hw.perflevel1.l2cachesize", info.apple_eff_l2_bytes);

    if (info.detected_apple_perf_l2) {
        info.l2_bytes = std::max(info.l2_bytes, info.apple_perf_l2_bytes);
        info.detected_l2 = true;
    }
#elif defined(__linux__)
    for (int idx = 0; idx < 16; ++idx) {
        const std::string base =
            "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(idx) + "/";

        std::string level_s, type_s, size_s;
        if (!full_tensor_triangular_read_first_line(base + "level", level_s)) continue;
        if (!full_tensor_triangular_read_first_line(base + "type",  type_s)) continue;
        if (!full_tensor_triangular_read_first_line(base + "size",  size_s)) continue;

        const int level = std::stoi(level_s);
        const std::size_t bytes = full_tensor_triangular_parse_cache_size(size_s);
        if (bytes == 0) continue;

        if (level == 1 && type_s.find("Data") != std::string::npos) {
            info.l1d_bytes = bytes;
            info.detected_l1d = true;
        } else if (level == 2 && type_s.find("Unified") != std::string::npos) {
            info.l2_bytes = bytes;
            info.detected_l2 = true;
        } else if (level == 3 && type_s.find("Unified") != std::string::npos) {
            info.l3_bytes = bytes;
            info.detected_l3 = true;
        }
    }
#endif

    return info;
}

std::size_t choose_full_tensor_triangular_p_batch(
    std::size_t M,
    std::size_t active_outer_workers,
    const FullTensorReflectionCompactTriangularCacheInfo& cache,
    const FullTensorReflectionCompactTriangularBatchedOptions& topt)
{
    if (topt.p_batch != 0) {
        return std::max<std::size_t>(1, topt.p_batch);
    }

    double frac = topt.l2_fraction;
    if (!(frac > 0.0) || frac > 1.0) {
        frac = 0.35;
    }

    (void)active_outer_workers;

    const std::size_t tri_elems = full_tensor_triangular_upper_size(M);
    const std::size_t tri_bytes = tri_elems * sizeof(double);

    std::size_t target_bytes = static_cast<std::size_t>(
        frac * static_cast<double>(cache.l2_bytes));
    target_bytes = std::max(target_bytes, topt.min_worker_batch_bytes);
    target_bytes = std::min(target_bytes, topt.max_worker_batch_bytes);

    // Per worker scratch is roughly:
    //   B_upper[M(M+1)/2] + L[p,M] + Rperm[p,M]
    // so p ~= (target - triangular_output) / (2*M*sizeof(double)).
    std::size_t p = topt.min_p_batch;
    if (target_bytes > tri_bytes && M > 0) {
        p = (target_bytes - tri_bytes) / (2u * M * sizeof(double));
    }

    p = std::max(p, topt.min_p_batch);
    p = std::min(p, topt.max_p_batch);
    p = full_tensor_triangular_round_down_multiple(p, topt.p_batch_round_multiple);
    if (p < topt.min_p_batch) {
        p = topt.min_p_batch;
    }
    return std::max<std::size_t>(1, p);
}

FullTensorTriangularReverseColumnMap
build_full_tensor_triangular_reverse_column_map(
    const FullTensorInputSupportPlan& support)
{
    FullTensorTriangularReverseColumnMap map;

    const auto& i2 = support.channels.i2;
    const auto& i3 = support.channels.i3;
    const auto& i4 = support.channels.i4;
    const auto& i5 = support.channels.i5;

    const std::size_t I2 = i2.size();
    const std::size_t I3 = i3.size();
    const std::size_t I4 = i4.size();
    const std::size_t I5 = i5.size();
    map.M = I2 * I3;
    map.N = I4 * I5;

    if (map.M != map.N) {
        map.compatible = false;
        return map;
    }

    map.n_to_s.assign(map.N, static_cast<std::size_t>(-1));
    map.s_to_n.assign(map.M, static_cast<std::size_t>(-1));

    for (std::size_t ii5 = 0; ii5 < I5; ++ii5) {
        for (std::size_t ii4 = 0; ii4 < I4; ++ii4) {
            const int ii2_s = full_tensor_reflection_find_value_index(i2, i5[ii5]);
            const int ii3_s = full_tensor_reflection_find_value_index(i3, i4[ii4]);
            if (ii2_s < 0 || ii3_s < 0) {
                map.compatible = false;
                return map;
            }

            const std::size_t n = ii4 + I4 * ii5;
            const std::size_t s = static_cast<std::size_t>(ii2_s)
                                + I2 * static_cast<std::size_t>(ii3_s);
            if (s >= map.M) {
                map.compatible = false;
                return map;
            }
            map.n_to_s[n] = s;
            map.s_to_n[s] = n;
        }
    }

    for (std::size_t s = 0; s < map.M; ++s) {
        if (map.s_to_n[s] == static_cast<std::size_t>(-1)) {
            map.compatible = false;
            return map;
        }
    }

    map.compatible = true;
    return map;
}

void full_tensor_triangular_batched_update_upper_tile4x4(
                                                                const double* B4FULL_RESTRICT L_batch,
                                                                const double* B4FULL_RESTRICT Rperm_batch,
                                                                double* B4FULL_RESTRICT B_upper,
                                                                std::size_t P,
                                                                std::size_t ldP,
                                                                std::size_t M,
                                                                std::size_t M_pad) // Accept the padded dimension
{
    if (P == 0 || M == 0) {
        return;
    }

    // Loop over the PADDED dimensions so we never hit a boundary
    for (std::size_t s0 = 0; s0 < M_pad; s0 += 4) {
        for (std::size_t m0 = 0; m0 <= s0; m0 += 4) {

            // No ternary operators. We blindly grab all 4 pointers.
            // If they are in the padding zone, they safely point to 0.0s.
            const double* B4FULL_RESTRICT l0 = L_batch + ldP * (m0 + 0);
            const double* B4FULL_RESTRICT l1 = L_batch + ldP * (m0 + 1);
            const double* B4FULL_RESTRICT l2 = L_batch + ldP * (m0 + 2);
            const double* B4FULL_RESTRICT l3 = L_batch + ldP * (m0 + 3);

            const double* B4FULL_RESTRICT r0 = Rperm_batch + ldP * (s0 + 0);
            const double* B4FULL_RESTRICT r1 = Rperm_batch + ldP * (s0 + 1);
            const double* B4FULL_RESTRICT r2 = Rperm_batch + ldP * (s0 + 2);
            const double* B4FULL_RESTRICT r3 = Rperm_batch + ldP * (s0 + 3);

            double acc00 = 0.0, acc01 = 0.0, acc02 = 0.0, acc03 = 0.0;
            double acc10 = 0.0, acc11 = 0.0, acc12 = 0.0, acc13 = 0.0;
            double acc20 = 0.0, acc21 = 0.0, acc22 = 0.0, acc23 = 0.0;
            double acc30 = 0.0, acc31 = 0.0, acc32 = 0.0, acc33 = 0.0;

            // This is now a pure, hardware-friendly vector loop
#ifdef _OPENMP
#pragma omp simd reduction(+:acc00,acc01,acc02,acc03,acc10,acc11,acc12,acc13,acc20,acc21,acc22,acc23,acc30,acc31,acc32,acc33)
#endif
            for (std::size_t p = 0; p < P; ++p) {
                const double v0 = l0[p];
                const double v1 = l1[p];
                const double v2 = l2[p];
                const double v3 = l3[p];

                const double w0 = r0[p];
                const double w1 = r1[p];
                const double w2 = r2[p];
                const double w3 = r3[p];

                acc00 += v0 * w0; acc01 += v0 * w1; acc02 += v0 * w2; acc03 += v0 * w3;
                acc10 += v1 * w0; acc11 += v1 * w1; acc12 += v1 * w2; acc13 += v1 * w3;
                acc20 += v2 * w0; acc21 += v2 * w1; acc22 += v2 * w2; acc23 += v2 * w3;
                acc30 += v3 * w0; acc31 += v3 * w1; acc32 += v3 * w2; acc33 += v3 * w3;
            }

            // Boundary checks happen HERE, safely outside the heavy loop
            const auto store_if_upper = [&](std::size_t dm, std::size_t ds, double value) {
                const std::size_t m = m0 + dm;
                const std::size_t s = s0 + ds;
                if (m <= s && m < M && s < M) {
                    B_upper[full_tensor_triangular_upper_index(m, s)] += value;
                }
            };

            store_if_upper(0, 0, acc00); store_if_upper(0, 1, acc01);
            store_if_upper(0, 2, acc02); store_if_upper(0, 3, acc03);
            store_if_upper(1, 0, acc10); store_if_upper(1, 1, acc11);
            store_if_upper(1, 2, acc12); store_if_upper(1, 3, acc13);
            store_if_upper(2, 0, acc20); store_if_upper(2, 1, acc21);
            store_if_upper(2, 2, acc22); store_if_upper(2, 3, acc23);
            store_if_upper(3, 0, acc30); store_if_upper(3, 1, acc31);
            store_if_upper(3, 2, acc32); store_if_upper(3, 3, acc33);
        }
    }
}

void full_tensor_triangular_batched_update_upper_tile4x8(
    const double* B4FULL_RESTRICT L_batch,
    const double* B4FULL_RESTRICT Rperm_batch,
    double* B4FULL_RESTRICT B_upper,
    std::size_t P,
    std::size_t ldP,
    std::size_t M,
    std::size_t M_pad)
{
    if (P == 0 || M == 0) {
        return;
    }

    // Rectangular padded triangular traversal.  The m-loop must run while
    // m0 < s0 + NR, not merely m0 <= s0, otherwise rectangular diagonal
    // tiles would miss valid upper-triangular entries.
    for (std::size_t s0 = 0; s0 < M_pad; s0 += 8) {
        for (std::size_t m0 = 0; m0 < M_pad && m0 < s0 + 8; m0 += 4) {
            const double* B4FULL_RESTRICT l0 = L_batch + ldP * (m0 + 0);
            const double* B4FULL_RESTRICT l1 = L_batch + ldP * (m0 + 1);
            const double* B4FULL_RESTRICT l2 = L_batch + ldP * (m0 + 2);
            const double* B4FULL_RESTRICT l3 = L_batch + ldP * (m0 + 3);

            const double* B4FULL_RESTRICT r0 = Rperm_batch + ldP * (s0 + 0);
            const double* B4FULL_RESTRICT r1 = Rperm_batch + ldP * (s0 + 1);
            const double* B4FULL_RESTRICT r2 = Rperm_batch + ldP * (s0 + 2);
            const double* B4FULL_RESTRICT r3 = Rperm_batch + ldP * (s0 + 3);
            const double* B4FULL_RESTRICT r4 = Rperm_batch + ldP * (s0 + 4);
            const double* B4FULL_RESTRICT r5 = Rperm_batch + ldP * (s0 + 5);
            const double* B4FULL_RESTRICT r6 = Rperm_batch + ldP * (s0 + 6);
            const double* B4FULL_RESTRICT r7 = Rperm_batch + ldP * (s0 + 7);

            double acc00 = 0.0, acc01 = 0.0, acc02 = 0.0, acc03 = 0.0, acc04 = 0.0, acc05 = 0.0, acc06 = 0.0, acc07 = 0.0;
            double acc10 = 0.0, acc11 = 0.0, acc12 = 0.0, acc13 = 0.0, acc14 = 0.0, acc15 = 0.0, acc16 = 0.0, acc17 = 0.0;
            double acc20 = 0.0, acc21 = 0.0, acc22 = 0.0, acc23 = 0.0, acc24 = 0.0, acc25 = 0.0, acc26 = 0.0, acc27 = 0.0;
            double acc30 = 0.0, acc31 = 0.0, acc32 = 0.0, acc33 = 0.0, acc34 = 0.0, acc35 = 0.0, acc36 = 0.0, acc37 = 0.0;

#ifdef _OPENMP
#pragma omp simd reduction(+:acc00,acc01,acc02,acc03,acc04,acc05,acc06,acc07,acc10,acc11,acc12,acc13,acc14,acc15,acc16,acc17,acc20,acc21,acc22,acc23,acc24,acc25,acc26,acc27,acc30,acc31,acc32,acc33,acc34,acc35,acc36,acc37)
#endif
            for (std::size_t p = 0; p < P; ++p) {
                const double v0 = l0[p];
                const double v1 = l1[p];
                const double v2 = l2[p];
                const double v3 = l3[p];
                const double w0 = r0[p];
                const double w1 = r1[p];
                const double w2 = r2[p];
                const double w3 = r3[p];
                const double w4 = r4[p];
                const double w5 = r5[p];
                const double w6 = r6[p];
                const double w7 = r7[p];

                acc00 += v0 * w0; acc01 += v0 * w1; acc02 += v0 * w2; acc03 += v0 * w3;
                acc04 += v0 * w4; acc05 += v0 * w5; acc06 += v0 * w6; acc07 += v0 * w7;
                acc10 += v1 * w0; acc11 += v1 * w1; acc12 += v1 * w2; acc13 += v1 * w3;
                acc14 += v1 * w4; acc15 += v1 * w5; acc16 += v1 * w6; acc17 += v1 * w7;
                acc20 += v2 * w0; acc21 += v2 * w1; acc22 += v2 * w2; acc23 += v2 * w3;
                acc24 += v2 * w4; acc25 += v2 * w5; acc26 += v2 * w6; acc27 += v2 * w7;
                acc30 += v3 * w0; acc31 += v3 * w1; acc32 += v3 * w2; acc33 += v3 * w3;
                acc34 += v3 * w4; acc35 += v3 * w5; acc36 += v3 * w6; acc37 += v3 * w7;
            }

            const auto store_if_upper = [&](std::size_t dm, std::size_t ds, double value) {
                const std::size_t m = m0 + dm;
                const std::size_t s = s0 + ds;
                if (m <= s && m < M && s < M) {
                    B_upper[full_tensor_triangular_upper_index(m, s)] += value;
                }
            };

            store_if_upper(0, 0, acc00); store_if_upper(0, 1, acc01);
            store_if_upper(0, 2, acc02); store_if_upper(0, 3, acc03);
            store_if_upper(0, 4, acc04); store_if_upper(0, 5, acc05);
            store_if_upper(0, 6, acc06); store_if_upper(0, 7, acc07);
            store_if_upper(1, 0, acc10); store_if_upper(1, 1, acc11);
            store_if_upper(1, 2, acc12); store_if_upper(1, 3, acc13);
            store_if_upper(1, 4, acc14); store_if_upper(1, 5, acc15);
            store_if_upper(1, 6, acc16); store_if_upper(1, 7, acc17);
            store_if_upper(2, 0, acc20); store_if_upper(2, 1, acc21);
            store_if_upper(2, 2, acc22); store_if_upper(2, 3, acc23);
            store_if_upper(2, 4, acc24); store_if_upper(2, 5, acc25);
            store_if_upper(2, 6, acc26); store_if_upper(2, 7, acc27);
            store_if_upper(3, 0, acc30); store_if_upper(3, 1, acc31);
            store_if_upper(3, 2, acc32); store_if_upper(3, 3, acc33);
            store_if_upper(3, 4, acc34); store_if_upper(3, 5, acc35);
            store_if_upper(3, 6, acc36); store_if_upper(3, 7, acc37);
        }
    }
}

void full_tensor_triangular_batched_update_upper_tile8x4(
    const double* B4FULL_RESTRICT L_batch,
    const double* B4FULL_RESTRICT Rperm_batch,
    double* B4FULL_RESTRICT B_upper,
    std::size_t P,
    std::size_t ldP,
    std::size_t M,
    std::size_t M_pad)
{
    if (P == 0 || M == 0) {
        return;
    }

    // Rectangular padded triangular traversal.  The m-loop must run while
    // m0 < s0 + NR, not merely m0 <= s0, otherwise rectangular diagonal
    // tiles would miss valid upper-triangular entries.
    for (std::size_t s0 = 0; s0 < M_pad; s0 += 4) {
        for (std::size_t m0 = 0; m0 < M_pad && m0 < s0 + 4; m0 += 8) {
            const double* B4FULL_RESTRICT l0 = L_batch + ldP * (m0 + 0);
            const double* B4FULL_RESTRICT l1 = L_batch + ldP * (m0 + 1);
            const double* B4FULL_RESTRICT l2 = L_batch + ldP * (m0 + 2);
            const double* B4FULL_RESTRICT l3 = L_batch + ldP * (m0 + 3);
            const double* B4FULL_RESTRICT l4 = L_batch + ldP * (m0 + 4);
            const double* B4FULL_RESTRICT l5 = L_batch + ldP * (m0 + 5);
            const double* B4FULL_RESTRICT l6 = L_batch + ldP * (m0 + 6);
            const double* B4FULL_RESTRICT l7 = L_batch + ldP * (m0 + 7);

            const double* B4FULL_RESTRICT r0 = Rperm_batch + ldP * (s0 + 0);
            const double* B4FULL_RESTRICT r1 = Rperm_batch + ldP * (s0 + 1);
            const double* B4FULL_RESTRICT r2 = Rperm_batch + ldP * (s0 + 2);
            const double* B4FULL_RESTRICT r3 = Rperm_batch + ldP * (s0 + 3);

            double acc00 = 0.0, acc01 = 0.0, acc02 = 0.0, acc03 = 0.0, acc10 = 0.0, acc11 = 0.0, acc12 = 0.0, acc13 = 0.0;
            double acc20 = 0.0, acc21 = 0.0, acc22 = 0.0, acc23 = 0.0, acc30 = 0.0, acc31 = 0.0, acc32 = 0.0, acc33 = 0.0;
            double acc40 = 0.0, acc41 = 0.0, acc42 = 0.0, acc43 = 0.0, acc50 = 0.0, acc51 = 0.0, acc52 = 0.0, acc53 = 0.0;
            double acc60 = 0.0, acc61 = 0.0, acc62 = 0.0, acc63 = 0.0, acc70 = 0.0, acc71 = 0.0, acc72 = 0.0, acc73 = 0.0;

#ifdef _OPENMP
#pragma omp simd reduction(+:acc00,acc01,acc02,acc03,acc10,acc11,acc12,acc13,acc20,acc21,acc22,acc23,acc30,acc31,acc32,acc33,acc40,acc41,acc42,acc43,acc50,acc51,acc52,acc53,acc60,acc61,acc62,acc63,acc70,acc71,acc72,acc73)
#endif
            for (std::size_t p = 0; p < P; ++p) {
                const double v0 = l0[p];
                const double v1 = l1[p];
                const double v2 = l2[p];
                const double v3 = l3[p];
                const double v4 = l4[p];
                const double v5 = l5[p];
                const double v6 = l6[p];
                const double v7 = l7[p];
                const double w0 = r0[p];
                const double w1 = r1[p];
                const double w2 = r2[p];
                const double w3 = r3[p];

                acc00 += v0 * w0; acc01 += v0 * w1; acc02 += v0 * w2; acc03 += v0 * w3;
                acc10 += v1 * w0; acc11 += v1 * w1; acc12 += v1 * w2; acc13 += v1 * w3;
                acc20 += v2 * w0; acc21 += v2 * w1; acc22 += v2 * w2; acc23 += v2 * w3;
                acc30 += v3 * w0; acc31 += v3 * w1; acc32 += v3 * w2; acc33 += v3 * w3;
                acc40 += v4 * w0; acc41 += v4 * w1; acc42 += v4 * w2; acc43 += v4 * w3;
                acc50 += v5 * w0; acc51 += v5 * w1; acc52 += v5 * w2; acc53 += v5 * w3;
                acc60 += v6 * w0; acc61 += v6 * w1; acc62 += v6 * w2; acc63 += v6 * w3;
                acc70 += v7 * w0; acc71 += v7 * w1; acc72 += v7 * w2; acc73 += v7 * w3;
            }

            const auto store_if_upper = [&](std::size_t dm, std::size_t ds, double value) {
                const std::size_t m = m0 + dm;
                const std::size_t s = s0 + ds;
                if (m <= s && m < M && s < M) {
                    B_upper[full_tensor_triangular_upper_index(m, s)] += value;
                }
            };

            store_if_upper(0, 0, acc00); store_if_upper(0, 1, acc01);
            store_if_upper(0, 2, acc02); store_if_upper(0, 3, acc03);
            store_if_upper(1, 0, acc10); store_if_upper(1, 1, acc11);
            store_if_upper(1, 2, acc12); store_if_upper(1, 3, acc13);
            store_if_upper(2, 0, acc20); store_if_upper(2, 1, acc21);
            store_if_upper(2, 2, acc22); store_if_upper(2, 3, acc23);
            store_if_upper(3, 0, acc30); store_if_upper(3, 1, acc31);
            store_if_upper(3, 2, acc32); store_if_upper(3, 3, acc33);
            store_if_upper(4, 0, acc40); store_if_upper(4, 1, acc41);
            store_if_upper(4, 2, acc42); store_if_upper(4, 3, acc43);
            store_if_upper(5, 0, acc50); store_if_upper(5, 1, acc51);
            store_if_upper(5, 2, acc52); store_if_upper(5, 3, acc53);
            store_if_upper(6, 0, acc60); store_if_upper(6, 1, acc61);
            store_if_upper(6, 2, acc62); store_if_upper(6, 3, acc63);
            store_if_upper(7, 0, acc70); store_if_upper(7, 1, acc71);
            store_if_upper(7, 2, acc72); store_if_upper(7, 3, acc73);
        }
    }
}

void flush_full_tensor_triangular_batched_lr(
                                                    FullTensorTriangularBatchedLRWorkspace& batch,
                                                    double* B4FULL_RESTRICT B_upper,
                                                    FullTensorPackedContractionL25ScalarProfile& profile)
{
    if (batch.skipped_rows_pending != 0) {
        if (batch.progress != nullptr) {
            batch.progress->add(batch.skipped_rows_pending);
        }
        batch.skipped_rows_pending = 0;
    }
    if (batch.p_used == 0) {
        return;
    }

    // Keep the profiler/counts based on the real compact triangular work,
    // not on the padded M dimension. Padding is only an implementation detail.
    const std::uint64_t terms =
    static_cast<std::uint64_t>(batch.p_used)
    * static_cast<std::uint64_t>(
                                 full_tensor_triangular_upper_size(batch.M)
                                 );

    {
        FullTensorScalarScopedTimer timer(profile.final_scalar);

        switch (batch.final_tile_shape) {
            case FullTensorTriangularFinalTileShape::Tile4x4:
                full_tensor_triangular_batched_update_upper_tile4x4(
                    batch.L.data(), batch.Rperm.data(), B_upper,
                    batch.p_used, batch.p_cap, batch.M, batch.M_pad);
                break;
            case FullTensorTriangularFinalTileShape::Tile4x8:
                full_tensor_triangular_batched_update_upper_tile4x8(
                    batch.L.data(), batch.Rperm.data(), B_upper,
                    batch.p_used, batch.p_cap, batch.M, batch.M_pad);
                break;
            case FullTensorTriangularFinalTileShape::Tile8x4:
                full_tensor_triangular_batched_update_upper_tile8x4(
                    batch.L.data(), batch.Rperm.data(), B_upper,
                    batch.p_used, batch.p_cap, batch.M, batch.M_pad);
                break;
        }
    }

    profile.support.final_dot_terms += terms;
    profile.final_scalar.approx_flops +=
    2.0L * static_cast<long double>(terms);
    if (batch.progress != nullptr) {
        batch.progress->add(static_cast<std::uint64_t>(batch.p_used));
    }

    batch.clear();
}

bool build_full_T2_l25_stack_reassociated_profiled(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorBoosters& B,
    const FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    const Range2D& rC,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* B4FULL_RESTRICT T2_row,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt)
{
    const std::size_t I4  = support.channels.i4.size();
    const std::size_t I5  = support.channels.i5.size();
    const std::size_t K4d = support.channels.k4.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L45 = support.l45.size();

    const std::size_t nk4 = static_cast<std::size_t>(rC.j_end - rC.j_begin);
    const std::size_t N   = I4 * I5;
    const std::size_t K4  = nk4 * L25;

    {
        FullTensorScalarScopedTimer timer(profile.T2_zero);
        full_scalar_profile_zero(T2_row, K4 * N);
    }

    profile.support.T2_values_total += static_cast<std::uint64_t>(K4 * N);

    const std::size_t T1_seed_size = K4d * L45 * I5;
    local_ws.T2_B4_panel.resize(L45 * I4);

    bool any = false;
    std::uint64_t local_terms = 0;
    std::uint64_t local_nnz = 0;

    {
        FullTensorScalarScopedTimer timer(profile.T2_stack);

        // B4(i4,l45) depends on k4, l24 and l34, but not on l25.  Keep k4
        // outside the l25 loop so this small support-aware panel is packed once
        // and reused for every l25 row.
        for (std::size_t jb = 0; jb < nk4; ++jb) {
            const std::size_t ik4 = static_cast<std::size_t>(rC.j_begin) + jb;

            std::fill(local_ws.T2_B4_panel.begin(), local_ws.T2_B4_panel.end(), 0.0);
            for (std::size_t ii4 = 0; ii4 < I4; ++ii4) {
                const std::size_t B4_block = idx4(ii4, ik4, il24, il34, K4d, L24, L34);
                const Range1D& r_B4 = support.B4.blocks[B4_block];
                if (r_B4.empty()) {
                    continue;
                }
                const double* B4FULL_RESTRICT B4_base = B.B4_data() + r_B4.offset;
                const std::size_t len = static_cast<std::size_t>(r_B4.end - r_B4.begin);
                local_terms += static_cast<std::uint64_t>(len) *
                               static_cast<std::uint64_t>(I5) *
                               static_cast<std::uint64_t>(L25);
                for (std::size_t il45 = r_B4.begin; il45 < r_B4.end; ++il45) {
                    local_ws.T2_B4_panel[il45 * I4 + ii4] = B4_base[il45 - r_B4.begin];
                }
            }

            for (std::size_t il25 = 0; il25 < L25; ++il25) {
                const double* B4FULL_RESTRICT T1_seed =
                    seed_ws.T1_cache.data() + (il25 * L35 + il35) * T1_seed_size;
                double* B4FULL_RESTRICT row = T2_row + (il25 * nk4 + jb) * N;

                // T2_row(i4,i5) += B4(i4,l45) * T1(l45,i5).
                // For fixed l45 and i5, the ii4 loop is contiguous in the
                // resident row-major panel n = ii4 + I4*ii5.
                for (std::size_t il45 = 0; il45 < L45; ++il45) {
                    const double* B4FULL_RESTRICT k4_col = local_ws.T2_B4_panel.data() + il45 * I4;
                    for (std::size_t ii5 = 0; ii5 < I5; ++ii5) {
                        const double t1 = T1_seed[L45 * (ik4 + K4d * ii5) + il45];
                        if (t1 == 0.0) {
                            continue;
                        }
                        double* B4FULL_RESTRICT out = row + I4 * ii5;
#ifdef _OPENMP
#pragma omp simd
#endif
                        for (std::size_t ii4 = 0; ii4 < I4; ++ii4) {
                            out[ii4] += k4_col[ii4] * t1;
                        }
                    }
                }

                if (opt.count_stack_nnz) {
                    for (std::size_t n = 0; n < N; ++n) {
                        if (row[n] != 0.0) {
                            ++local_nnz;
                        }
                    }
                }
            }
        }
    }

    if (!opt.count_stack_nnz) {
        for (std::size_t i = 0; i < K4 * N; ++i) {
            if (T2_row[i] != 0.0) {
                any = true;
                break;
            }
        }
    } else {
        any = (local_nnz != 0);
    }

    profile.support.T2_dot_terms += local_terms;
    profile.support.T2_values_nnz += local_nnz;
    profile.T2_stack.approx_flops += 2.0L * static_cast<long double>(local_terms);

    return any;
}

bool build_full_T4_l25_stack_reassociated_profiled(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorBoosters& B,
    const FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    const Range2D& rC,
    std::size_t il24,
    std::size_t il34,
    std::size_t il35,
    double* B4FULL_RESTRICT T4_row,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt)
{
    const std::size_t I2  = support.channels.i2.size();
    const std::size_t I3  = support.channels.i3.size();
    const std::size_t K3d = support.channels.k3.size();
    const std::size_t L23 = support.l23.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();

    const std::size_t nk3 = static_cast<std::size_t>(rC.i_end - rC.i_begin);
    const std::size_t M   = I2 * I3;
    const std::size_t K3  = nk3 * L25;

    {
        FullTensorScalarScopedTimer timer(profile.T4_zero);
        full_scalar_profile_zero(T4_row, K3 * M);
    }

    profile.support.T4_values_total += static_cast<std::uint64_t>(K3 * M);

    const std::size_t T3_seed_size = L23 * K3d * I2;
    local_ws.T4_B3_panel.resize(L23 * I3);
    local_ws.T4_T3_col.resize(I2);

    bool any = false;
    std::uint64_t local_terms = 0;
    std::uint64_t local_nnz = 0;

    {
        FullTensorScalarScopedTimer timer(profile.T4_stack);

        // B3(i3,l23) depends on k3, l34 and l35, but not on l25.  Hoist
        // support lookup and packing so all l25 rows share one packed panel.
        for (std::size_t ib = 0; ib < nk3; ++ib) {
            const std::size_t ik3 = static_cast<std::size_t>(rC.i_begin) + ib;

            std::fill(local_ws.T4_B3_panel.begin(), local_ws.T4_B3_panel.end(), 0.0);
            for (std::size_t ii3 = 0; ii3 < I3; ++ii3) {
                const std::size_t B3_block = idx4(ii3, ik3, il34, il35, K3d, L34, L35);
                const Range1D& r_B3 = support.B3.blocks[B3_block];
                if (r_B3.empty()) {
                    continue;
                }
                const double* B4FULL_RESTRICT B3_base = B.B3_data() + r_B3.offset;
                const std::size_t len = static_cast<std::size_t>(r_B3.end - r_B3.begin);
                local_terms += static_cast<std::uint64_t>(len) *
                               static_cast<std::uint64_t>(I2) *
                               static_cast<std::uint64_t>(L25);
                for (std::size_t il23 = r_B3.begin; il23 < r_B3.end; ++il23) {
                    local_ws.T4_B3_panel[il23 * I3 + ii3] = B3_base[il23 - r_B3.begin];
                }
            }

            for (std::size_t il25 = 0; il25 < L25; ++il25) {
                const double* B4FULL_RESTRICT T3_seed =
                    seed_ws.T3_cache.data() + (il25 * L24 + il24) * T3_seed_size;
                double* B4FULL_RESTRICT row = T4_row + (il25 * nk3 + ib) * M;

                // T4_row(i2,i3) += T3(i2,l23) * B3(i3,l23).
                // For fixed l23 and i3, the ii2 loop is contiguous in
                // m = ii2 + I2*ii3.
                for (std::size_t il23 = 0; il23 < L23; ++il23) {
                    for (std::size_t ii2 = 0; ii2 < I2; ++ii2) {
                        local_ws.T4_T3_col[ii2] = T3_seed[L23 * (ik3 + K3d * ii2) + il23];
                    }
                    for (std::size_t ii3 = 0; ii3 < I3; ++ii3) {
                        const double k3 = local_ws.T4_B3_panel[il23 * I3 + ii3];
                        if (k3 == 0.0) {
                            continue;
                        }
                        double* B4FULL_RESTRICT out = row + I2 * ii3;
#ifdef _OPENMP
#pragma omp simd
#endif
                        for (std::size_t ii2 = 0; ii2 < I2; ++ii2) {
                            out[ii2] += local_ws.T4_T3_col[ii2] * k3;
                        }
                    }
                }

                if (opt.count_stack_nnz) {
                    for (std::size_t m = 0; m < M; ++m) {
                        if (row[m] != 0.0) {
                            ++local_nnz;
                        }
                    }
                }
            }
        }
    }

    if (!opt.count_stack_nnz) {
        for (std::size_t i = 0; i < K3 * M; ++i) {
            if (T4_row[i] != 0.0) {
                any = true;
                break;
            }
        }
    } else {
        any = (local_nnz != 0);
    }

    profile.support.T4_dot_terms += local_terms;
    profile.support.T4_values_nnz += local_nnz;
    profile.T4_stack.approx_flops += 2.0L * static_cast<long double>(local_terms);

    return any;
}

void build_full_D4_l25_stack_reassociated_profiled(
    const Range2D& rC,
    const double* B4FULL_RESTRICT C_base,
    const double* B4FULL_RESTRICT T4_row,
    double* B4FULL_RESTRICT D4_row,
    std::size_t L25,
    std::size_t M,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt)
{
    const std::size_t nk3 = static_cast<std::size_t>(rC.i_end - rC.i_begin);
    const std::size_t nk4 = static_cast<std::size_t>(rC.j_end - rC.j_begin);
    const std::size_t K4 = nk4 * L25;

    const long double dense_flops_ld =
        2.0L * static_cast<long double>(L25) *
        static_cast<long double>(nk4) * static_cast<long double>(nk3) *
        static_cast<long double>(M);
    const bool blas_dims_fit =
        nk3 <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
        nk4 <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
        M <= static_cast<std::size_t>(std::numeric_limits<int>::max());
    const bool use_vendor_blas =
        opt.use_vendor_blas_CT2 && blas_dims_fit &&
        nk3 != 0 && nk4 != 0 && M != 0 && L25 != 0 &&
        dense_flops_ld >= static_cast<long double>(opt.vendor_blas_CT2_min_flops);

    // Every l25 slice is fully overwritten by beta=0 DGEMM.  The scalar
    // fallback accumulates with += and therefore retains the explicit zero.
    if (!use_vendor_blas) {
        FullTensorScalarScopedTimer timer(profile.CT2_zero);
        full_scalar_profile_zero(D4_row, K4 * M);
    }

    // Reuse the CT2 support counters for the reassociated D4 intermediate.
    profile.support.CT2_values_total += static_cast<std::uint64_t>(K4 * M);

    std::uint64_t local_updates = 0;
    {
        FullTensorScalarScopedTimer timer(profile.CT2_scalar);

        if (use_vendor_blas) {
            // C is column-major [nk3,nk4], which is the same byte layout as a
            // row-major [nk4,nk3] C^T.  T4 and D4 are row-major by construction,
            // so every l25 slice is a zero-pack DGEMM.
            for (std::size_t il25 = 0; il25 < L25; ++il25) {
                const double* t4 = T4_row + il25 * nk3 * M;
                double* d4 = D4_row + il25 * nk4 * M;
                cblas_dgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    static_cast<int>(nk4), static_cast<int>(M), static_cast<int>(nk3),
                    1.0, C_base, static_cast<int>(nk3),
                    t4, static_cast<int>(M),
                    0.0, d4, static_cast<int>(M));
            }
            local_updates = static_cast<std::uint64_t>(L25) *
                            static_cast<std::uint64_t>(nk4) *
                            static_cast<std::uint64_t>(nk3) *
                            static_cast<std::uint64_t>(M);
        } else {
            for (std::size_t il25 = 0; il25 < L25; ++il25) {
                for (std::size_t jb = 0; jb < nk4; ++jb) {
                    double* B4FULL_RESTRICT drow = D4_row + (il25 * nk4 + jb) * M;

                    for (std::size_t ib = 0; ib < nk3; ++ib) {
                        const double c = C_base[ib + nk3 * jb];
                        if (c == 0.0) {
                            continue;
                        }
                        const double* B4FULL_RESTRICT t4row = T4_row + (il25 * nk3 + ib) * M;
                        local_updates += static_cast<std::uint64_t>(M);
#ifdef _OPENMP
#pragma omp simd
#endif
                        for (std::size_t m = 0; m < M; ++m) {
                            drow[m] += c * t4row[m];
                        }
                    }
                }
            }
        }
    }

    profile.support.CT2_rank1_updates += local_updates;
    profile.CT2_scalar.approx_flops += 2.0L * static_cast<long double>(local_updates);

    if (opt.count_CT2_nnz) {
        std::uint64_t local_nnz = 0;
        for (std::size_t i = 0; i < K4 * M; ++i) {
            if (D4_row[i] != 0.0) {
                ++local_nnz;
            }
        }
        profile.support.CT2_values_nnz += local_nnz;
    }
}

void append_q_block_to_full_tensor_triangular_batched_lr_reassociated(
    const double* B4FULL_RESTRICT D4_row,
    const double* B4FULL_RESTRICT T2_row,
    std::size_t K,
    std::size_t M,
    const FullTensorTriangularReverseColumnMap& reverse_map,
    FullTensorTriangularBatchedLRWorkspace& batch,
    double* B4FULL_RESTRICT B_upper,
    FullTensorPackedContractionL25ScalarProfile& profile)
{
    if (K == 0) {
        return;
    }

    if (K > batch.p_cap) {
        flush_full_tensor_triangular_batched_lr(batch, B_upper, profile);
        batch.reset(K, M);
    } else if (batch.p_used + K > batch.p_cap) {
        flush_full_tensor_triangular_batched_lr(batch, B_upper, profile);
    }

    const std::size_t p0 = batch.p_used;

    // Pack D4 rows into column-major L_batch[p,m].
    for (std::size_t m = 0; m < M; ++m) {
        double* B4FULL_RESTRICT dst = batch.L.data() + p0 + batch.p_cap * m;
        for (std::size_t p = 0; p < K; ++p) {
            dst[p] = D4_row[p * M + m];
        }
    }

    // Pack T2 rows into the reflected column order Rperm[p,s].
    for (std::size_t s = 0; s < M; ++s) {
        const std::size_t n = reverse_map.s_to_n[s];
        double* B4FULL_RESTRICT dst = batch.Rperm.data() + p0 + batch.p_cap * s;
        for (std::size_t p = 0; p < K; ++p) {
            dst[p] = T2_row[p * M + n];
        }
    }

    batch.p_used += K;
}

void append_q_orbit_to_full_tensor_triangular_batched_lr_reassociated(
    const double* B4FULL_RESTRICT D4_row,
    const double* B4FULL_RESTRICT T2_row,
    std::size_t K,
    std::size_t M,
    const FullTensorTriangularReverseColumnMap& reverse_map,
    FullTensorQOrbitKind orbit_kind,
    FullTensorTriangularBatchedLRWorkspace& batch,
    double* B4FULL_RESTRICT B_upper,
    FullTensorPackedContractionL25ScalarProfile& profile)
{
    if (orbit_kind == FullTensorQOrbitKind::Unpaired) {
        append_q_block_to_full_tensor_triangular_batched_lr_reassociated(
            D4_row, T2_row, K, M, reverse_map, batch, B_upper, profile);
        return;
    }
    if (K == 0) {
        return;
    }

    const double orbit_scale =
        orbit_kind == FullTensorQOrbitKind::SelfReflected ? 0.5 : 1.0;

    const auto append_orientation = [&](bool reflected_output) {
        if (K > batch.p_cap) {
            flush_full_tensor_triangular_batched_lr(batch, B_upper, profile);
            batch.reset(K, M);
        } else if (batch.p_used + K > batch.p_cap) {
            flush_full_tensor_triangular_batched_lr(batch, B_upper, profile);
        }

        const std::size_t p0 = batch.p_used;
        for (std::size_t m = 0; m < M; ++m) {
            const std::size_t reflected_n = reverse_map.s_to_n[m];
            double* B4FULL_RESTRICT dst =
                batch.L.data() + p0 + batch.p_cap * m;
            for (std::size_t p = 0; p < K; ++p) {
                const double value = reflected_output
                    ? T2_row[p * M + reflected_n]
                    : D4_row[p * M + m];
                dst[p] = orbit_scale * value;
            }
        }

        for (std::size_t s = 0; s < M; ++s) {
            const std::size_t reflected_n = reverse_map.s_to_n[s];
            double* B4FULL_RESTRICT dst =
                batch.Rperm.data() + p0 + batch.p_cap * s;
            for (std::size_t p = 0; p < K; ++p) {
                dst[p] = reflected_output
                    ? D4_row[p * M + s]
                    : T2_row[p * M + reflected_n];
            }
        }
        batch.p_used += K;
    };

    append_orientation(false);
    append_orientation(true);
}

void process_full_tensor_one_q_block_triangular_batched_reassociated(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    const FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    FullTensorTriangularBatchedLRWorkspace& batch,
    const FullTensorTriangularReverseColumnMap& reverse_map,
    std::size_t ix,
    std::size_t il35,
    std::size_t il24,
    std::size_t il34,
    double* B4FULL_RESTRICT B_upper,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt,
    FullTensorQOrbitKind orbit_kind)
{
    const std::size_t I2  = support.channels.i2.size();
    const std::size_t I3  = support.channels.i3.size();
    const std::size_t I4  = support.channels.i4.size();
    const std::size_t I5  = support.channels.i5.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t M = I2 * I3;
    const std::size_t N = I4 * I5;

    ++profile.support.q_blocks_seen;

    if (M != N) {
        throw b4utils::InternalFailure(
            "Reassociated triangular backend requires I2*I3 == I4*I5.");
    }

    const std::size_t C_block = idx4(ix, il24, il34, il35, L24, L34, L35);
    const FullTensorSjCHalfBlockMap& c_map = support.sjC_map[C_block];
    const Range2D& rC = c_map.logical;
    if (rC.empty()) {
        return;
    }

    ++profile.support.q_blocks_nonempty_sjC;
    const std::size_t nk3 = static_cast<std::size_t>(rC.i_end - rC.i_begin);
    const std::size_t nk4 = static_cast<std::size_t>(rC.j_end - rC.j_begin);
    const std::uint64_t planned_final_rows =
        full_tensor_cpu_saturating_mul_u64(
            full_tensor_cpu_saturating_mul_u64(
                static_cast<std::uint64_t>(nk4),
                static_cast<std::uint64_t>(L25)),
            orbit_kind == FullTensorQOrbitKind::Unpaired ? 1u : 2u);
    profile.support.sum_nk3 += static_cast<std::uint64_t>(nk3);
    profile.support.sum_nk4 += static_cast<std::uint64_t>(nk4);
    profile.support.max_nk3 = std::max<std::uint64_t>(profile.support.max_nk3, nk3);
    profile.support.max_nk4 = std::max<std::uint64_t>(profile.support.max_nk4, nk4);

    local_ws.ensure_stacks(nk3, nk4, L25, M, N);
    local_ws.D4_stack.resize(nk4 * L25 * M);

    const bool has_T2 = build_full_T2_l25_stack_reassociated_profiled(
        support, B, seed_ws, local_ws, rC, il24, il34, il35,
        local_ws.T2_stack.data(), profile, opt);
    if (!has_T2) {
        batch.skip_planned_rows(planned_final_rows);
        return;
    }
    ++profile.support.q_blocks_with_T2;

    const bool has_T4 = build_full_T4_l25_stack_reassociated_profiled(
        support, B, seed_ws, local_ws, rC, il24, il34, il35,
        local_ws.T4_stack.data(), profile, opt);
    if (!has_T4) {
        batch.skip_planned_rows(planned_final_rows);
        return;
    }
    ++profile.support.q_blocks_with_T4;

    const double* B4FULL_RESTRICT C_base = nullptr;
    {
        FullTensorScalarScopedTimer timer(profile.sjC_access);
        C_base = full_tensor_sjC_logical_block_base_l25_scalar(
            support, sj, c_map, local_ws.C_block);
    }
    if (C_base == nullptr) {
        batch.skip_planned_rows(planned_final_rows);
        return;
    }

    build_full_D4_l25_stack_reassociated_profiled(
        rC, C_base, local_ws.T4_stack.data(), local_ws.D4_stack.data(),
        L25, M, profile, opt);

    const std::size_t K = nk4 * L25;
    append_q_orbit_to_full_tensor_triangular_batched_lr_reassociated(
        local_ws.D4_stack.data(), local_ws.T2_stack.data(), K, M,
        reverse_map, orbit_kind, batch, B_upper, profile);

    ++profile.support.q_blocks_accumulated;
}

void process_full_tensor_one_q_orbit_triangular_batched_reassociated(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    const FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    FullTensorTriangularBatchedLRWorkspace& batch,
    const FullTensorTriangularReverseColumnMap& reverse_map,
    const FullTensorQOrbitDescriptor& orbit,
    bool canonical_orbit,
    double* B4FULL_RESTRICT B_upper,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt)
{
    const FullTensorQOrbitMember& member = canonical_orbit
        ? full_tensor_select_left_associated_cpu_q_member(orbit)
        : orbit.representative;
    const FullTensorQOrbitKind kind = canonical_orbit
        ? orbit.kind
        : FullTensorQOrbitKind::Unpaired;

    profile.used_cpu_q_reflection_orbits =
        profile.used_cpu_q_reflection_orbits || canonical_orbit;
    ++profile.cpu_q_orbits_executed;
    if (kind == FullTensorQOrbitKind::Paired) {
        profile.cpu_q_logical_members_covered += 2;
        ++profile.cpu_q_paired_orbits_executed;
    } else if (kind == FullTensorQOrbitKind::SelfReflected) {
        ++profile.cpu_q_logical_members_covered;
        ++profile.cpu_q_self_orbits_executed;
    } else {
        ++profile.cpu_q_logical_members_covered;
        ++profile.cpu_q_unpaired_blocks_executed;
    }

    process_full_tensor_one_q_block_triangular_batched_reassociated(
        support, sj, B, seed_ws, local_ws, batch, reverse_map,
        member.ix, member.il35, member.il24, member.il34,
        B_upper, profile, opt, kind);
}

std::uint64_t full_tensor_reflection_compact_triangular_batched_reassociated_q_weight_units(
    const FullTensorInputSupportPlan& support,
    const FullTensorReflectionCompactOutputMap& compact_map,
    std::size_t ix,
    std::size_t il35,
    std::size_t il24,
    std::size_t il34)
{
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::uint64_t M = static_cast<std::uint64_t>(compact_map.M());
    const std::uint64_t tri_entries = static_cast<std::uint64_t>(
        compact_map.independent_entries());

    const std::size_t C_block = idx4(ix, il24, il34, il35, L24, L34, L35);
    const FullTensorSjCHalfBlockMap& c_map = support.sjC_map[C_block];
    const Range2D& rC = c_map.logical;
    if (rC.empty()) {
        return 1;
    }

    const std::uint64_t nk3 = static_cast<std::uint64_t>(rC.i_end - rC.i_begin);
    const std::uint64_t nk4 = static_cast<std::uint64_t>(rC.j_end - rC.j_begin);
    const std::uint64_t l25 = static_cast<std::uint64_t>(L25);
    const std::uint64_t d4_units = (nk3 * nk4 * l25 * M) / 64u;
    const std::uint64_t final_units = (nk4 * l25 * tri_entries) / 256u;
    return 1u + d4_units + final_units;
}

std::uint64_t build_full_tensor_reflection_compact_triangular_batched_reassociated_q_weights(
    const FullTensorInputSupportPlan& support,
    const FullTensorReflectionCompactOutputMap& compact_map,
    std::vector<std::uint64_t>& q_weights)
{
    const std::size_t X   = support.channels.xvals.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    q_weights.assign(X * L35 * L24 * L34, 1u);

    std::uint64_t total_one_i1 = 0;
    for (std::size_t ix = 0; ix < X; ++ix) {
        for (std::size_t il35 = 0; il35 < L35; ++il35) {
            for (std::size_t il24 = 0; il24 < L24; ++il24) {
                for (std::size_t il34 = 0; il34 < L34; ++il34) {
                    const std::uint64_t w = full_tensor_reflection_compact_triangular_batched_reassociated_q_weight_units(
                        support, compact_map, ix, il35, il24, il34);
                    const std::size_t flat = full_tensor_reflection_compact_q_weight_index(
                        ix, il35, il24, il34, L35, L24, L34);
                    q_weights[flat] = w;
                    total_one_i1 += w;
                }
            }
        }
    }

    const std::uint64_t I1 = static_cast<std::uint64_t>(support.channels.i1.size());
    return total_one_i1 * I1;
}

std::uint64_t
build_full_tensor_cpu_q_orbit_triangular_batched_reassociated_q_weights(
    const FullTensorInputSupportPlan& support,
    const FullTensorReflectionCompactOutputMap& compact_map,
    const FullTensorQReflectionOrbitPlan& orbit_plan,
    std::vector<std::uint64_t>& q_weights)
{
    const std::size_t X = support.channels.xvals.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::uint64_t L25 =
        static_cast<std::uint64_t>(support.l25.size());
    const std::uint64_t M = static_cast<std::uint64_t>(compact_map.M());
    const std::uint64_t tri_entries = static_cast<std::uint64_t>(
        compact_map.independent_entries());
    q_weights.assign(X * L35 * L24 * L34, 0u);

    std::uint64_t total_one_i1 = 0;
    for (const FullTensorQOrbitDescriptor& orbit : orbit_plan.orbits) {
        const FullTensorQOrbitMember& member =
            full_tensor_select_left_associated_cpu_q_member(orbit);
        const std::uint64_t nk3 = member.nk3();
        const std::uint64_t nk4 = member.nk4();
        const std::uint64_t d4_units =
            (nk3 * nk4 * L25 * M) / 64u;
        const std::uint64_t final_units =
            (nk4 * L25 * tri_entries) / 256u;
        const std::uint64_t reflected_copies =
            orbit.kind == FullTensorQOrbitKind::Unpaired ? 1u : 2u;
        const std::uint64_t weight =
            1u + d4_units + reflected_copies * final_units;
        const std::size_t flat =
            full_tensor_reflection_compact_q_weight_index(
                member.ix, member.il35, member.il24, member.il34,
                L35, L24, L34);
        q_weights[flat] = weight;
        total_one_i1 += weight;
    }

    const std::uint64_t I1 =
        static_cast<std::uint64_t>(support.channels.i1.size());
    return total_one_i1 * I1;
}

bool full_tensor_reassociated_rectangular_use_vendor_blas(
    const FullTensorReassociatedRectangularBatchedLRWorkspace& batch) noexcept
{
    if (!batch.use_vendor_blas || batch.p_used == 0 ||
        batch.M == 0 || batch.N == 0) {
        return false;
    }

    // This source uses the portable LP64 CBLAS interface.  Retain the local
    // kernel rather than narrowing dimensions if a future panel exceeds it.
    const std::size_t blas_int_max =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (batch.p_used > blas_int_max || batch.p_cap > blas_int_max ||
        batch.M > blas_int_max || batch.N > blas_int_max) {
        return false;
    }

    if (std::min(batch.M, batch.N) < batch.vendor_blas_min_dimension) {
        return false;
    }

    const long double dense_flops =
        2.0L * static_cast<long double>(batch.p_used) *
        static_cast<long double>(batch.M) *
        static_cast<long double>(batch.N);
    return dense_flops >=
        static_cast<long double>(batch.vendor_blas_min_flops);
}

void full_tensor_reassociated_rectangular_batched_update_tile4x4(
    const double* B4FULL_RESTRICT L_batch,
    const double* B4FULL_RESTRICT R_batch,
    double* B4FULL_RESTRICT A_panel,
    std::size_t P,
    std::size_t ldP,
    std::size_t M,
    std::size_t N,
    std::size_t M_pad,
    std::size_t N_pad)
{
    if (P == 0 || M == 0 || N == 0) {
        return;
    }

    // Full rectangular padded traversal.  The hot p-loop has no boundary
    // conditionals.  Only the final store checks whether we are in the real
    // M x N panel or in padded zero columns.
    for (std::size_t n0 = 0; n0 < N_pad; n0 += 4) {
        const double* B4FULL_RESTRICT r0 = R_batch + ldP * (n0 + 0);
        const double* B4FULL_RESTRICT r1 = R_batch + ldP * (n0 + 1);
        const double* B4FULL_RESTRICT r2 = R_batch + ldP * (n0 + 2);
        const double* B4FULL_RESTRICT r3 = R_batch + ldP * (n0 + 3);

        for (std::size_t m0 = 0; m0 < M_pad; m0 += 4) {
            const double* B4FULL_RESTRICT l0 = L_batch + ldP * (m0 + 0);
            const double* B4FULL_RESTRICT l1 = L_batch + ldP * (m0 + 1);
            const double* B4FULL_RESTRICT l2 = L_batch + ldP * (m0 + 2);
            const double* B4FULL_RESTRICT l3 = L_batch + ldP * (m0 + 3);

            double acc00 = 0.0, acc01 = 0.0, acc02 = 0.0, acc03 = 0.0;
            double acc10 = 0.0, acc11 = 0.0, acc12 = 0.0, acc13 = 0.0;
            double acc20 = 0.0, acc21 = 0.0, acc22 = 0.0, acc23 = 0.0;
            double acc30 = 0.0, acc31 = 0.0, acc32 = 0.0, acc33 = 0.0;

#ifdef _OPENMP
#pragma omp simd reduction(+:acc00,acc01,acc02,acc03,acc10,acc11,acc12,acc13,acc20,acc21,acc22,acc23,acc30,acc31,acc32,acc33)
#endif
            for (std::size_t p = 0; p < P; ++p) {
                const double v0 = l0[p];
                const double v1 = l1[p];
                const double v2 = l2[p];
                const double v3 = l3[p];
                const double w0 = r0[p];
                const double w1 = r1[p];
                const double w2 = r2[p];
                const double w3 = r3[p];

                acc00 += v0 * w0; acc01 += v0 * w1; acc02 += v0 * w2; acc03 += v0 * w3;
                acc10 += v1 * w0; acc11 += v1 * w1; acc12 += v1 * w2; acc13 += v1 * w3;
                acc20 += v2 * w0; acc21 += v2 * w1; acc22 += v2 * w2; acc23 += v2 * w3;
                acc30 += v3 * w0; acc31 += v3 * w1; acc32 += v3 * w2; acc33 += v3 * w3;
            }

            const auto store_if_real = [&](std::size_t dm, std::size_t dn, double value) {
                const std::size_t m = m0 + dm;
                const std::size_t n = n0 + dn;
                if (m < M && n < N) {
                    A_panel[m + M * n] += value;
                }
            };

            store_if_real(0, 0, acc00); store_if_real(0, 1, acc01);
            store_if_real(0, 2, acc02); store_if_real(0, 3, acc03);
            store_if_real(1, 0, acc10); store_if_real(1, 1, acc11);
            store_if_real(1, 2, acc12); store_if_real(1, 3, acc13);
            store_if_real(2, 0, acc20); store_if_real(2, 1, acc21);
            store_if_real(2, 2, acc22); store_if_real(2, 3, acc23);
            store_if_real(3, 0, acc30); store_if_real(3, 1, acc31);
            store_if_real(3, 2, acc32); store_if_real(3, 3, acc33);
        }
    }
}

void flush_full_tensor_reassociated_rectangular_batched_lr(
    FullTensorReassociatedRectangularBatchedLRWorkspace& batch,
    double* B4FULL_RESTRICT A_panel,
    FullTensorPackedContractionL25ScalarProfile& profile)
{
    if (batch.skipped_rows_pending != 0) {
        if (batch.progress != nullptr) {
            batch.progress->add(batch.skipped_rows_pending);
        }
        batch.skipped_rows_pending = 0;
    }
    if (batch.p_used == 0) {
        return;
    }

    const std::uint64_t terms =
        static_cast<std::uint64_t>(batch.p_used) *
        static_cast<std::uint64_t>(batch.M) *
        static_cast<std::uint64_t>(batch.N);

    {
        FullTensorScalarScopedTimer timer(profile.final_scalar);
        if (full_tensor_reassociated_rectangular_use_vendor_blas(batch)) {
            // L and R are already column-major P-by-M/P-by-N panels and the
            // public rectangular output is column-major M-by-N.  No transpose
            // buffer or output conversion is required.
            cblas_dgemm(
                CblasColMajor, CblasTrans, CblasNoTrans,
                static_cast<int>(batch.M), static_cast<int>(batch.N),
                static_cast<int>(batch.p_used),
                1.0, batch.L.data(), static_cast<int>(batch.p_cap),
                batch.R.data(), static_cast<int>(batch.p_cap),
                1.0, A_panel, static_cast<int>(batch.M));
            profile.cpu_vendor_blas_final_calls += 1;
        } else {
            full_tensor_reassociated_rectangular_batched_update_tile4x4(
                batch.L.data(), batch.R.data(), A_panel,
                batch.p_used, batch.p_cap,
                batch.M, batch.N, batch.M_pad, batch.N_pad);
            profile.cpu_tiled_final_calls += 1;
        }
    }

    profile.support.final_dot_terms += terms;
    profile.final_scalar.approx_flops += 2.0L * static_cast<long double>(terms);
    if (batch.progress != nullptr) {
        batch.progress->add(static_cast<std::uint64_t>(batch.p_used));
    }
    batch.clear();
}

void append_q_block_to_full_tensor_reassociated_rectangular_batched_lr(
    const double* B4FULL_RESTRICT D4_row,
    const double* B4FULL_RESTRICT T2_row,
    std::size_t K,
    std::size_t M,
    std::size_t N,
    FullTensorReassociatedRectangularBatchedLRWorkspace& batch,
    double* B4FULL_RESTRICT A_panel,
    FullTensorPackedContractionL25ScalarProfile& profile)
{
    if (K == 0) {
        return;
    }

    if (K > batch.p_cap || batch.M != M || batch.N != N) {
        flush_full_tensor_reassociated_rectangular_batched_lr(batch, A_panel, profile);
        batch.reset(K, M, N);
    } else if (batch.p_used + K > batch.p_cap) {
        flush_full_tensor_reassociated_rectangular_batched_lr(batch, A_panel, profile);
    }

    const std::size_t p0 = batch.p_used;

    // Pack D4 rows into column-major L_batch[p,m].
    for (std::size_t m = 0; m < M; ++m) {
        double* B4FULL_RESTRICT dst = batch.L.data() + p0 + batch.p_cap * m;
        for (std::size_t p = 0; p < K; ++p) {
            dst[p] = D4_row[p * M + m];
        }
    }

    // Pack T2 rows into column-major R_batch[p,n].
    for (std::size_t n = 0; n < N; ++n) {
        double* B4FULL_RESTRICT dst = batch.R.data() + p0 + batch.p_cap * n;
        for (std::size_t p = 0; p < K; ++p) {
            dst[p] = T2_row[p * N + n];
        }
    }

    batch.p_used += K;
}

void process_full_tensor_one_q_block_reassociated_rectangular_batched(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    const FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    FullTensorReassociatedRectangularBatchedLRWorkspace& batch,
    std::size_t ix,
    std::size_t il35,
    std::size_t il24,
    std::size_t il34,
    double* B4FULL_RESTRICT A_panel,
    FullTensorPackedContractionL25ScalarProfile& profile,
    const FullTensorPackedContractionL25ScalarOptions& opt)
{
    const std::size_t I2  = support.channels.i2.size();
    const std::size_t I3  = support.channels.i3.size();
    const std::size_t I4  = support.channels.i4.size();
    const std::size_t I5  = support.channels.i5.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t M = I2 * I3;
    const std::size_t N = I4 * I5;

    ++profile.support.q_blocks_seen;

    const std::size_t C_block = idx4(ix, il24, il34, il35, L24, L34, L35);
    const FullTensorSjCHalfBlockMap& c_map = support.sjC_map[C_block];
    const Range2D& rC = c_map.logical;
    if (rC.empty()) {
        return;
    }

    ++profile.support.q_blocks_nonempty_sjC;
    const std::size_t nk3 = static_cast<std::size_t>(rC.i_end - rC.i_begin);
    const std::size_t nk4 = static_cast<std::size_t>(rC.j_end - rC.j_begin);
    const std::uint64_t planned_final_rows =
        full_tensor_cpu_saturating_mul_u64(
            static_cast<std::uint64_t>(nk4),
            static_cast<std::uint64_t>(L25));
    profile.support.sum_nk3 += static_cast<std::uint64_t>(nk3);
    profile.support.sum_nk4 += static_cast<std::uint64_t>(nk4);
    profile.support.max_nk3 = std::max<std::uint64_t>(profile.support.max_nk3, nk3);
    profile.support.max_nk4 = std::max<std::uint64_t>(profile.support.max_nk4, nk4);

    local_ws.ensure_stacks(nk3, nk4, L25, M, N);
    local_ws.D4_stack.resize(nk4 * L25 * M);

    const bool has_T2 = build_full_T2_l25_stack_reassociated_profiled(
        support, B, seed_ws, local_ws, rC, il24, il34, il35,
        local_ws.T2_stack.data(), profile, opt);
    if (!has_T2) {
        batch.skip_planned_rows(planned_final_rows);
        return;
    }
    ++profile.support.q_blocks_with_T2;

    const bool has_T4 = build_full_T4_l25_stack_reassociated_profiled(
        support, B, seed_ws, local_ws, rC, il24, il34, il35,
        local_ws.T4_stack.data(), profile, opt);
    if (!has_T4) {
        batch.skip_planned_rows(planned_final_rows);
        return;
    }
    ++profile.support.q_blocks_with_T4;

    const double* B4FULL_RESTRICT C_base = nullptr;
    {
        FullTensorScalarScopedTimer timer(profile.sjC_access);
        C_base = full_tensor_sjC_logical_block_base_l25_scalar(
            support, sj, c_map, local_ws.C_block);
    }
    if (C_base == nullptr) {
        batch.skip_planned_rows(planned_final_rows);
        return;
    }

    build_full_D4_l25_stack_reassociated_profiled(
        rC, C_base, local_ws.T4_stack.data(), local_ws.D4_stack.data(),
        L25, M, profile, opt);

    const std::size_t K = nk4 * L25;
    append_q_block_to_full_tensor_reassociated_rectangular_batched_lr(
        local_ws.D4_stack.data(), local_ws.T2_stack.data(), K, M, N,
        batch, A_panel, profile);

    ++profile.support.q_blocks_accumulated;
}

std::uint64_t full_tensor_reassociated_rectangular_batched_q_weight_units(
    const FullTensorInputSupportPlan& support,
    std::size_t ix,
    std::size_t il35,
    std::size_t il24,
    std::size_t il34)
{
    const std::size_t I2  = support.channels.i2.size();
    const std::size_t I3  = support.channels.i3.size();
    const std::size_t I4  = support.channels.i4.size();
    const std::size_t I5  = support.channels.i5.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    const std::uint64_t M = static_cast<std::uint64_t>(I2 * I3);
    const std::uint64_t N = static_cast<std::uint64_t>(I4 * I5);

    const std::size_t C_block = idx4(ix, il24, il34, il35, L24, L34, L35);
    const FullTensorSjCHalfBlockMap& c_map = support.sjC_map[C_block];
    const Range2D& rC = c_map.logical;
    if (rC.empty()) {
        return 1;
    }

    const std::uint64_t nk3 = static_cast<std::uint64_t>(rC.i_end - rC.i_begin);
    const std::uint64_t nk4 = static_cast<std::uint64_t>(rC.j_end - rC.j_begin);
    const std::uint64_t l25 = static_cast<std::uint64_t>(L25);
    const std::uint64_t d4_units = (nk3 * nk4 * l25 * M) / 64u;
    const std::uint64_t final_units = (nk4 * l25 * M * N) / 512u;
    return 1u + d4_units + final_units;
}

std::uint64_t build_full_tensor_reassociated_rectangular_batched_q_weights(
    const FullTensorInputSupportPlan& support,
    std::vector<std::uint64_t>& q_weights)
{
    const std::size_t X   = support.channels.xvals.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    q_weights.assign(X * L35 * L24 * L34, 1u);

    std::uint64_t total_one_i1 = 0;
    for (std::size_t ix = 0; ix < X; ++ix) {
        for (std::size_t il35 = 0; il35 < L35; ++il35) {
            for (std::size_t il24 = 0; il24 < L24; ++il24) {
                for (std::size_t il34 = 0; il34 < L34; ++il34) {
                    const std::uint64_t w = full_tensor_reassociated_rectangular_batched_q_weight_units(
                        support, ix, il35, il24, il34);
                    const std::size_t flat = full_tensor_reflection_compact_q_weight_index(
                        ix, il35, il24, il34, L35, L24, L34);
                    q_weights[flat] = w;
                    total_one_i1 += w;
                }
            }
        }
    }

    const std::uint64_t I1 = static_cast<std::uint64_t>(support.channels.i1.size());
    return total_one_i1 * I1;
}

std::size_t choose_full_tensor_reassociated_rectangular_p_batch(
    std::size_t M,
    std::size_t N,
    std::size_t active_outer_workers,
    const FullTensorReflectionCompactTriangularCacheInfo& cache,
    const FullTensorReflectionCompactTriangularBatchedOptions& topt)
{
    if (topt.p_batch != 0) {
        return std::max<std::size_t>(1, topt.p_batch);
    }

    (void)active_outer_workers;
    double frac = topt.l2_fraction;
    if (!(frac > 0.0) || frac > 1.0) {
        frac = 0.35;
    }

    const std::size_t M_pad = full_tensor_triangular_round_up_multiple(M, 4);
    const std::size_t N_pad = full_tensor_triangular_round_up_multiple(N, 4);
    const std::size_t panel_bytes = M * N * sizeof(double);

    std::size_t target_bytes = static_cast<std::size_t>(
        frac * static_cast<double>(cache.l2_bytes));
    target_bytes = std::max(target_bytes, topt.min_worker_batch_bytes);
    target_bytes = std::min(target_bytes, topt.max_worker_batch_bytes);

    std::size_t p = topt.min_p_batch;
    if (target_bytes > panel_bytes && (M_pad + N_pad) > 0) {
        p = (target_bytes - panel_bytes) / ((M_pad + N_pad) * sizeof(double));
    }
    p = std::max(p, topt.min_p_batch);
    p = std::min(p, topt.max_p_batch);
    p = full_tensor_triangular_round_down_multiple(p, topt.p_batch_round_multiple);
    if (p < topt.min_p_batch) {
        p = topt.min_p_batch;
    }
    return std::max<std::size_t>(1, p);
}

FullTensorReassociatedAutoOutputPlan
make_full_tensor_reassociated_auto_output_plan(
    const FullTensorInputSupportPlan& support)
{
    FullTensorReassociatedAutoOutputPlan plan;
    plan.I1 = support.channels.i1.size();
    plan.M  = support.channels.i2.size() * support.channels.i3.size();
    plan.N  = support.channels.i4.size() * support.channels.i5.size();

    const FullTensorReflectionCompactOutputMap compact_map =
        build_full_tensor_reflection_compact_output_map(support);
    const FullTensorTriangularReverseColumnMap reverse_map =
        build_full_tensor_triangular_reverse_column_map(support);

    plan.reflection_compatible = compact_map.compatible;
    plan.reverse_column_compatible = reverse_map.compatible;

    if (compact_map.compatible) {
        plan.compact_representatives = compact_map.independent_entries();
    }

    const bool compact_triangular_is_valid =
        support.reflection_15432_preserves_spins &&
        compact_map.compatible &&
        reverse_map.compatible &&
        reverse_map.M == compact_map.M() &&
        compact_map.M() == compact_map.N() &&
        compact_map.independent_entries() ==
            full_tensor_triangular_upper_size(compact_map.M());

    if (compact_triangular_is_valid) {
        plan.layout = FullTensorReassociatedAutoOutputLayout::ReflectionCompactTriangular;
        plan.values_count = full_tensor_reflection_compact_values_count(support);
    } else {
        plan.layout = FullTensorReassociatedAutoOutputLayout::RectangularFull;
        plan.values_count = full_tensor_reassociated_rectangular_by_i1_values_count(support);
    }

    return plan;
}

FullTensorCpuBatchedWorkspaceAdmission
make_full_tensor_cpu_batched_workspace_admission(
    const FullTensorInputSupportPlan& support,
    const FullTensorPackedContractionL25ScalarOptions& opt,
    const FullTensorReflectionCompactTriangularBatchedOptions& topt,
    const FullTensorReflectionCompactTriangularCacheInfo& cache)
{
    FullTensorCpuBatchedWorkspaceAdmission admission;
    const FullTensorReassociatedAutoOutputPlan output_plan =
        make_full_tensor_reassociated_auto_output_plan(support);

    const std::size_t I1 = support.channels.i1.size();
    const std::size_t X = support.channels.xvals.size();
    const std::size_t task_ceiling =
        I1 != 0 && X > std::numeric_limits<std::size_t>::max() / I1
            ? std::numeric_limits<std::size_t>::max()
            : std::max<std::size_t>(1, I1 * X);

#ifdef _OPENMP
    const std::size_t omp_threads = opt.use_openmp
        ? static_cast<std::size_t>(std::max(1, omp_get_max_threads()))
        : 1u;
#else
    (void)opt;
    const std::size_t omp_threads = 1u;
#endif

    admission.worker_ceiling = std::min(task_ceiling, omp_threads);
    if (topt.max_outer_workers != 0) {
        admission.worker_ceiling = std::min(
            admission.worker_ceiling, topt.max_outer_workers);
    }
    admission.worker_ceiling = std::max<std::size_t>(
        1, admission.worker_ceiling);

    if (output_plan.uses_reflection_compact()) {
        admission.M_pad = full_tensor_triangular_round_up_multiple(
            output_plan.M, 8);
        admission.p_batch = choose_full_tensor_triangular_p_batch(
            output_plan.M, admission.worker_ceiling, cache, topt);
        const std::uint64_t one_panel_values =
            full_tensor_cpu_saturating_mul_u64(
                static_cast<std::uint64_t>(admission.p_batch),
                static_cast<std::uint64_t>(admission.M_pad));
        admission.per_worker_bytes = full_tensor_cpu_saturating_mul_u64(
            full_tensor_cpu_saturating_mul_u64(one_panel_values, 2u),
            static_cast<std::uint64_t>(sizeof(double)));
    } else {
        admission.M_pad = full_tensor_triangular_round_up_multiple(
            output_plan.M, 4);
        admission.N_pad = full_tensor_triangular_round_up_multiple(
            output_plan.N, 4);
        admission.p_batch =
            choose_full_tensor_reassociated_rectangular_p_batch(
                output_plan.M, output_plan.N, admission.worker_ceiling,
                cache, topt);
        const std::uint64_t padded_columns =
            full_tensor_cpu_saturating_add_u64(
                static_cast<std::uint64_t>(admission.M_pad),
                static_cast<std::uint64_t>(admission.N_pad));
        admission.per_worker_bytes = full_tensor_cpu_saturating_mul_u64(
            full_tensor_cpu_saturating_mul_u64(
                static_cast<std::uint64_t>(admission.p_batch),
                padded_columns),
            static_cast<std::uint64_t>(sizeof(double)));
    }

    admission.total_bytes = full_tensor_cpu_saturating_mul_u64(
        admission.per_worker_bytes,
        static_cast<std::uint64_t>(admission.worker_ceiling));
    return admission;
}

FullTensorTlsPanelReductionStats
full_tensor_reduce_tls_i1_panels_deterministic(
    const std::vector<std::vector<double>>& tls_output,
    std::size_t active_workers,
    std::size_t I1,
    std::size_t panel_elems,
    double* B4FULL_RESTRICT output_by_i1,
    bool use_openmp,
    std::size_t parallel_min_output_values)
{
    FullTensorTlsPanelReductionStats stats;
    if (active_workers == 0 || I1 == 0 || panel_elems == 0) {
        return stats;
    }

    assert(output_by_i1 != nullptr);
    assert(active_workers <= std::numeric_limits<std::size_t>::max() / I1);
    assert(tls_output.size() >= active_workers * I1);

    // Compact the sparse lazy-panel table once.  Appending in tid order
    // preserves the previous cross-thread floating-point reduction order for
    // a given set of TLS panels; task-to-thread assignment remains a property
    // of the dynamic scheduler.
    std::vector<std::size_t> source_offsets(I1 + 1, 0);
    std::vector<const double*> sources;
    sources.reserve(active_workers * I1);
    for (std::size_t ii1 = 0; ii1 < I1; ++ii1) {
        for (std::size_t tid = 0; tid < active_workers; ++tid) {
            const std::vector<double>& panel = tls_output[tid * I1 + ii1];
            if (panel.empty()) continue;
            assert(panel.size() == panel_elems);
            sources.push_back(panel.data());
            ++stats.panels_touched;
            stats.panel_bytes += static_cast<std::uint64_t>(panel.size())
                               * static_cast<std::uint64_t>(sizeof(double));
        }
        source_offsets[ii1 + 1] = sources.size();
    }
    if (sources.empty()) {
        return stats;
    }

    constexpr std::size_t block_values = 4096;
    const std::size_t blocks_per_panel =
        1 + (panel_elems - 1) / block_values;
    assert(I1 <= std::numeric_limits<std::size_t>::max() / blocks_per_panel);
    const std::size_t total_blocks = I1 * blocks_per_panel;
    assert(total_blocks <=
           static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()));

    const auto reduce_block = [&](std::size_t block_index) {
        const std::size_t ii1 = block_index / blocks_per_panel;
        const std::size_t local_block = block_index - ii1 * blocks_per_panel;
        const std::size_t begin = local_block * block_values;
        const std::size_t end = std::min(panel_elems, begin + block_values);
        double* const dst = output_by_i1 + ii1 * panel_elems;
        const std::size_t source_begin = source_offsets[ii1];
        const std::size_t source_end = source_offsets[ii1 + 1];
        if (source_begin == source_end) return;
        if (source_end == source_begin + 1) {
            const double* B4FULL_RESTRICT src = sources[source_begin];
#ifdef _OPENMP
#pragma omp simd
#endif
            for (std::size_t i = begin; i < end; ++i) {
                dst[i] += src[i];
            }
            return;
        }
        for (std::size_t i = begin; i < end; ++i) {
            double value = dst[i];
            for (std::size_t source = source_begin;
                 source < source_end; ++source) {
                value += sources[source][i];
            }
            dst[i] = value;
        }
    };

    assert(I1 <= std::numeric_limits<std::size_t>::max() / panel_elems);
    const std::size_t output_values = I1 * panel_elems;
#ifdef _OPENMP
    if (use_openmp && active_workers > 1 &&
        output_values >= parallel_min_output_values) {
#pragma omp parallel for schedule(static) num_threads(active_workers)
        for (std::ptrdiff_t block = 0;
             block < static_cast<std::ptrdiff_t>(total_blocks); ++block) {
            reduce_block(static_cast<std::size_t>(block));
        }
        return stats;
    }
#else
    (void)use_openmp;
    (void)parallel_min_output_values;
#endif

    for (std::size_t block = 0; block < total_blocks; ++block) {
        reduce_block(block);
    }
    return stats;
}

std::vector<std::uint64_t>
build_full_tensor_cpu_planned_final_rows_by_x(
    const FullTensorInputSupportPlan& support,
    const FullTensorReassociatedAutoOutputPlan& plan,
    const FullTensorQReflectionOrbitPlan& orbit_plan,
    bool use_canonical_q_orbits)
{
    const std::size_t X = support.channels.xvals.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L25 = support.l25.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t L35 = support.l35.size();
    std::vector<std::uint64_t> rows(X, 0);

    const auto add_rows = [&](std::size_t ix,
                              std::uint64_t nk4,
                              std::uint64_t copies) {
        if (ix >= rows.size()) return;
        const std::uint64_t contribution =
            full_tensor_cpu_saturating_mul_u64(
                full_tensor_cpu_saturating_mul_u64(
                    nk4, static_cast<std::uint64_t>(L25)),
                copies);
        rows[ix] = full_tensor_cpu_saturating_add_u64(
            rows[ix], contribution);
    };

    if (plan.uses_reflection_compact() && use_canonical_q_orbits &&
        orbit_plan.canonical_orbits_enabled) {
        for (const FullTensorQOrbitDescriptor& orbit : orbit_plan.orbits) {
            const FullTensorQOrbitMember& member =
                full_tensor_select_left_associated_cpu_q_member(orbit);
            const std::uint64_t copies =
                orbit.kind == FullTensorQOrbitKind::Unpaired ? 1u : 2u;
            add_rows(member.ix, member.nk4(), copies);
        }
        return rows;
    }

    for (std::size_t ix = 0; ix < X; ++ix) {
        for (std::size_t il35 = 0; il35 < L35; ++il35) {
            for (std::size_t il24 = 0; il24 < L24; ++il24) {
                for (std::size_t il34 = 0; il34 < L34; ++il34) {
                    const std::size_t block = idx4(
                        ix, il24, il34, il35, L24, L34, L35);
                    const Range2D& range = support.sjC_map[block].logical;
                    if (range.empty()) continue;
                    add_rows(
                        ix,
                        static_cast<std::uint64_t>(
                            range.j_end - range.j_begin),
                        1u);
                }
            }
        }
    }
    return rows;
}

double full_tensor_scheduled_seed_support_units(
    const FullTensorInputSupportPlan& support,
    std::size_t ii1,
    std::size_t ix,
    const FullTensorAdaptiveScheduleOptions& sopt)
{
    if (!sopt.use_seed_support_predictor) {
        return 1.0;
    }

    const std::size_t X   = support.channels.xvals.size();
    const std::size_t L25 = support.l25.size();
    double sjA_units = 0.0;
    double sjE_units = 0.0;
    for (std::size_t il25 = 0; il25 < L25; ++il25) {
        const std::size_t block = idx3(ii1, ix, il25, X, L25);
        sjA_units += static_cast<double>(support.sjA.blocks[block].count());
        sjE_units += static_cast<double>(support.sjE.blocks[block].count());
    }

    // The two branch seeds both need support.  The product term is deliberately
    // softened by L25 so this remains a predictor rather than a sledgehammer.
    double units = 1.0 + sjA_units + sjE_units;
    if (L25 != 0) {
        units += (sjA_units * sjE_units) / static_cast<double>(L25);
    }
    if (sopt.seed_support_power != 1.0) {
        units = std::pow(std::max(1.0, units), sopt.seed_support_power);
    }
    return std::max(1.0, units);
}

std::vector<double> full_tensor_scheduled_x_q_work(
    const FullTensorInputSupportPlan& support,
    const std::vector<std::uint64_t>& q_weights)
{
    const std::size_t X   = support.channels.xvals.size();
    const std::size_t L35 = support.l35.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    std::vector<double> x_work(X, 1.0);
    for (std::size_t ix = 0; ix < X; ++ix) {
        double s = 0.0;
        for (std::size_t il35 = 0; il35 < L35; ++il35) {
            for (std::size_t il24 = 0; il24 < L24; ++il24) {
                for (std::size_t il34 = 0; il34 < L34; ++il34) {
                    const std::size_t idx = full_tensor_reflection_compact_q_weight_index(
                        ix, il35, il24, il34, L35, L24, L34);
                    s += q_weights.empty() ? 1.0 : static_cast<double>(q_weights[idx]);
                }
            }
        }
        x_work[ix] = std::max(1.0, s);
    }
    return x_work;
}

std::vector<FullTensorScheduledTask> build_full_tensor_adaptive_scheduled_tasks(
    const FullTensorInputSupportPlan& support,
    const FullTensorReassociatedAutoOutputPlan& plan,
    const std::vector<std::uint64_t>& q_weights,
    std::size_t active_workers,
    std::size_t p_batch,
    const FullTensorAdaptiveScheduleOptions& sopt,
    double& total_predicted_work,
    double& target_task_work,
    std::size_t& whole_i1_tasks,
    std::size_t& split_i1_panels,
    FullTensorAdaptiveScheduleMode selected_mode,
    double& i1_tasks_per_worker,
    double& predicted_i1_max_over_mean,
    double& predicted_i1_max_over_target,
    double& predicted_i1_max_work)
{
    (void)plan;
    const std::size_t I1 = support.channels.i1.size();
    const std::size_t X  = support.channels.xvals.size();
    const std::vector<double> x_q = full_tensor_scheduled_x_q_work(support, q_weights);

    std::vector<std::vector<double>> w(I1, std::vector<double>(X, 1.0));
    std::vector<double> i1_sum(I1, 0.0);
    total_predicted_work = 0.0;

    for (std::size_t ii1 = 0; ii1 < I1; ++ii1) {
        for (std::size_t ix = 0; ix < X; ++ix) {
            const double seed = full_tensor_scheduled_seed_support_units(support, ii1, ix, sopt);
            const double ww = std::max(1.0, x_q[ix] * seed);
            w[ii1][ix] = ww;
            i1_sum[ii1] += ww;
            total_predicted_work += ww;
        }
    }

    const double i1_mean = I1 != 0 ? total_predicted_work / static_cast<double>(I1) : 0.0;
    double i1_max = 0.0;
    for (double v : i1_sum) i1_max = std::max(i1_max, v);
    predicted_i1_max_over_mean = i1_mean > 0.0 ? i1_max / i1_mean : 0.0;
    i1_tasks_per_worker = static_cast<double>(I1) / static_cast<double>(std::max<std::size_t>(1, active_workers));

    const double worker_count = static_cast<double>(std::max<std::size_t>(1, active_workers));
    const double oversub = std::max(1.0, sopt.oversubscription);
    target_task_work = total_predicted_work / (worker_count * oversub);

    // Tie the target to p_batch so tasks are not microscopic relative to the
    // inner final accumulator.  This is deliberately a soft lower bound in the
    // same arbitrary predictor units, not an exact flop model.
    const double avg_x = X != 0 ? total_predicted_work / static_cast<double>(I1 * X) : 1.0;
    const double p_soft = avg_x * std::max<double>(1.0, static_cast<double>(p_batch) / 512.0);
    target_task_work = std::max(target_task_work, p_soft);
    if (sopt.max_predicted_work_per_task > 0.0) {
        target_task_work = std::min(target_task_work, sopt.max_predicted_work_per_task);
    }

    predicted_i1_max_work = i1_max;
    predicted_i1_max_over_target = target_task_work > 0.0 ? i1_max / target_task_work : 0.0;

    std::vector<FullTensorScheduledTask> tasks;
    whole_i1_tasks = 0;
    split_i1_panels = 0;

    const auto make_units = [&](double predicted) -> std::uint64_t {
        // Progress bars do not need astrophysical integers.  Normalize later-ish
        // by using target-size units while preserving ordering well enough.
        const double scale = std::max(1.0, total_predicted_work / 1000000000.0);
        const double u = std::max(1.0, predicted / scale);
        if (u > static_cast<double>(std::numeric_limits<std::uint64_t>::max() / 4ull)) {
            return std::numeric_limits<std::uint64_t>::max() / 4ull;
        }
        return static_cast<std::uint64_t>(std::llround(u));
    };

    const auto fill_task_progress_units = [&](FullTensorScheduledTask& t) {
        std::uint64_t sum_units = 0;
        for (std::size_t ix = t.ix_begin; ix < t.ix_end; ++ix) {
            std::uint64_t u = make_units(w[t.ii1][ix]);
            if (u == 0) u = 1;
            sum_units = full_tensor_cpu_saturating_add_u64(sum_units, u);
        }
        t.predicted_units = std::max<std::uint64_t>(1, sum_units);
    };

    const bool force_whole_i1 = (selected_mode == FullTensorAdaptiveScheduleMode::WeightedI1Only);

    for (std::size_t ii1 = 0; ii1 < I1; ++ii1) {
        const bool keep_whole = force_whole_i1 ||
            i1_sum[ii1] <= sopt.split_threshold_factor * target_task_work || X <= 1;

        if (keep_whole) {
            FullTensorScheduledTask t;
            t.ii1 = ii1;
            t.ix_begin = 0;
            t.ix_end = X;
            t.predicted_work = i1_sum[ii1];
            t.whole_i1 = true;
            fill_task_progress_units(t);
            tasks.push_back(t);
            ++whole_i1_tasks;
            continue;
        }

        ++split_i1_panels;
        const std::size_t before = tasks.size();
        std::size_t begin = 0;
        double acc = 0.0;
        std::size_t tasks_for_i1 = 0;
        const std::size_t max_tasks = sopt.max_tasks_per_i1 == 0
            ? std::numeric_limits<std::size_t>::max()
            : sopt.max_tasks_per_i1;

        for (std::size_t ix = 0; ix < X; ++ix) {
            acc += w[ii1][ix];
            const std::size_t len = ix + 1 - begin;
            const bool enough_x = len >= std::max<std::size_t>(1, sopt.min_x_per_task);
            const bool reached_target = acc >= target_task_work;
            const bool must_leave_room = (X - (ix + 1)) >= std::max<std::size_t>(1, sopt.min_x_per_task);
            const bool under_cap = (tasks_for_i1 + 1) < max_tasks;
            if (reached_target && enough_x && must_leave_room && under_cap) {
                FullTensorScheduledTask t;
                t.ii1 = ii1;
                t.ix_begin = begin;
                t.ix_end = ix + 1;
                t.predicted_work = acc;
                t.whole_i1 = false;
                fill_task_progress_units(t);
                tasks.push_back(t);
                ++tasks_for_i1;
                begin = ix + 1;
                acc = 0.0;
            }
        }

        if (begin < X) {
            // Merge a tiny tail into the previous task for this same i1.
            if (tasks.size() > before && acc < sopt.tail_merge_fraction * target_task_work) {
                FullTensorScheduledTask& prev = tasks.back();
                prev.ix_end = X;
                prev.predicted_work += acc;
                fill_task_progress_units(prev);
            } else {
                FullTensorScheduledTask t;
                t.ii1 = ii1;
                t.ix_begin = begin;
                t.ix_end = X;
                t.predicted_work = acc;
                t.whole_i1 = false;
                fill_task_progress_units(t);
                tasks.push_back(t);
            }
        }
    }

    if (sopt.sort_descending_predicted_work) {
        std::sort(tasks.begin(), tasks.end(), [](const FullTensorScheduledTask& a,
                                                 const FullTensorScheduledTask& b) {
            return a.predicted_work > b.predicted_work;
        });
    }

    // Fill a cheap p-row predictor for diagnostics.  The exact p rows depend on
    // surviving T2/T4 support, but ix-range size times p_batch scale is enough
    // to detect crumb-sized tasks.
    for (auto& t : tasks) {
        const double frac = total_predicted_work > 0.0 ? t.predicted_work / total_predicted_work : 0.0;
        t.predicted_p_rows = static_cast<std::uint64_t>(std::max(1.0,
            frac * static_cast<double>(I1 * X) * static_cast<double>(p_batch)));
    }

    return tasks;
}

void print_full_tensor_adaptive_schedule_summary(
    const FullTensorReassociatedAutoOutputPlan& plan,
    const FullTensorReflectionCompactTriangularCacheInfo& cache,
    const std::vector<FullTensorScheduledTask>& tasks,
    std::size_t I1,
    std::size_t X,
    std::size_t M,
    std::size_t N,
    std::size_t p_batch,
    std::size_t active_workers,
    double total_predicted_work,
    double target_task_work,
    std::size_t whole_i1_tasks,
    std::size_t split_i1_panels,
    FullTensorAdaptiveScheduleMode selected_mode,
    double i1_tasks_per_worker,
    double predicted_i1_max_over_mean,
    double predicted_i1_max_over_target,
    double predicted_i1_max_work,
    bool heavy_i1_forced_split,
    bool inner_tile_parallel_requested,
    bool inner_tile_parallel_enabled,
    const FullTensorReflectionCompactTriangularBatchedOptions& topt,
    const FullTensorAdaptiveScheduleOptions& sopt,
    std::ostream& os)
{
    const double mib = 1024.0 * 1024.0;
    const std::size_t M_pad = full_tensor_triangular_round_up_multiple(M, 4);
    const std::size_t N_pad = plan.uses_reflection_compact()
        ? M_pad
        : full_tensor_triangular_round_up_multiple(N, 4);
    const std::size_t out_elems = plan.uses_reflection_compact()
        ? full_tensor_triangular_upper_size(M)
        : M * N;
    const double out_mb = static_cast<double>(out_elems * sizeof(double)) / mib;
    const double l_mb = static_cast<double>(p_batch * M_pad * sizeof(double)) / mib;
    const double r_mb = static_cast<double>(p_batch * N_pad * sizeof(double)) / mib;

    os << "\nReassociated adaptive scheduled-parallel summary\n";
    os << "================================================\n";
    os << "layout                 = " << full_tensor_reassociated_auto_layout_name(plan.layout) << "\n";
    os << "parallel tasks         = weighted (i1,x-range), count = " << tasks.size() << "\n";
    os << "I1, X                  = " << I1 << ", " << X << "\n";
    os << "whole-i1 tasks         = " << whole_i1_tasks << "\n";
    os << "split i1 panels        = " << split_i1_panels << "\n";
    os << "active workers         = " << active_workers << "\n";
    os << "scheduler requested    = " << full_tensor_adaptive_schedule_mode_name(sopt.mode) << "\n";
    os << "scheduler selected     = " << full_tensor_adaptive_schedule_mode_name(selected_mode) << "\n";
    os << "I1/tasks per worker    = " << std::fixed << std::setprecision(3) << i1_tasks_per_worker << "\n";
    os << "pred i1 max/mean       = " << std::fixed << std::setprecision(3) << predicted_i1_max_over_mean << "\n";
    os << "pred i1 max/target     = " << std::fixed << std::setprecision(3) << predicted_i1_max_over_target << "\n";
    os << "pred i1 max work       = " << std::scientific << std::setprecision(6) << predicted_i1_max_work << "\n";
    os << "heavy-i1 split guard   = " << (heavy_i1_forced_split ? "triggered" : "not triggered") << "\n";
    os << "max task work cap      = " << std::scientific << std::setprecision(6) << sopt.max_predicted_work_per_task << "\n";
    os << "inner tile fallback    = "
       << (inner_tile_parallel_enabled ? "enabled" : (inner_tile_parallel_requested ? "requested but not implemented" : "not requested"))
       << "\n";
    os << "oversubscription       = " << std::fixed << std::setprecision(2) << sopt.oversubscription << "\n";
    os << "target predicted work  = " << std::scientific << std::setprecision(6) << target_task_work << "\n";
    os << "total predicted work   = " << std::scientific << std::setprecision(6) << total_predicted_work << "\n";
    os << "L1D = " << cache.l1d_bytes / 1024 << " KiB"
       << (cache.detected_l1d ? "" : " (fallback)") << "\n";
    os << "L2  = " << cache.l2_bytes / 1024 << " KiB"
       << (cache.detected_l2 ? "" : " (fallback)") << "\n";
    if (cache.detected_apple_perf_l2) {
        os << "Apple perf L2 = " << cache.apple_perf_l2_bytes / 1024 << " KiB\n";
    }
    if (cache.detected_apple_eff_l2) {
        os << "Apple eff  L2 = " << cache.apple_eff_l2_bytes / 1024 << " KiB\n";
    }
    os << "M = I2*I3            = " << M << "\n";
    os << "N = I4*I5            = " << N << "\n";
    os << "M_pad                = " << M_pad << "\n";
    os << "N_pad                = " << N_pad << "\n";
    os << "selected P_batch     = " << p_batch << "\n";
    os << "private output panel = " << std::fixed << std::setprecision(2)
       << out_mb << " MiB per worker\n";
    os << "L_batch              = " << std::fixed << std::setprecision(2) << l_mb << " MiB\n";
    os << "R_batch              = " << std::fixed << std::setprecision(2) << r_mb << " MiB\n";
    os << "approx per-worker scratch = " << std::fixed << std::setprecision(2)
       << (out_mb + l_mb + r_mb) << " MiB\n";
    os << "reassociation         = enabled: final = (C^T*T4)^T*T2\n";
    os << "final tile shape      = ";
    switch (topt.final_tile_shape) {
        case FullTensorTriangularFinalTileShape::Tile4x4: os << "4x4"; break;
        case FullTensorTriangularFinalTileShape::Tile4x8: os << "4x8"; break;
        case FullTensorTriangularFinalTileShape::Tile8x4: os << "8x4"; break;
    }
    os << "\n";
}

void process_full_tensor_xrange_task_triangular_batched_reassociated(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    std::size_t ii1,
    std::size_t ix_begin,
    std::size_t ix_end,
    double* B4FULL_RESTRICT B_upper_task,
    const FullTensorReflectionCompactOutputMap& compact_map,
    const FullTensorTriangularReverseColumnMap& reverse_map,
    const FullTensorQReflectionOrbitPlan& q_orbit_plan,
    FullTensorPackedContractionL25ScalarOptions opt,
    const FullTensorReflectionCompactTriangularBatchedOptions& topt,
    FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    FullTensorTriangularBatchedLRWorkspace& batch,
    FullTensorPackedContractionL25ScalarProfile& profile)
{
    const std::size_t L35 = support.l35.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t M = compact_map.M();
    const std::size_t tri_size = full_tensor_triangular_upper_size(M);

    {
        FullTensorScalarScopedTimer timer(profile.zero_output);
        full_scalar_profile_zero(B_upper_task, tri_size);
    }

    batch.begin_task();
    batch.final_tile_shape = topt.final_tile_shape;

    for (std::size_t ix = ix_begin; ix < ix_end; ++ix) {
        if (!full_tensor_cpu_i1_x_has_seed_wings(support, ii1, ix)) {
            continue;
        }
        build_full_seed_caches_l25_scalar_profiled(
            support, sj, B, ii1, ix, seed_ws, profile, opt);

        const bool use_canonical_q_orbits =
            opt.use_cpu_q_reflection_orbits &&
            q_orbit_plan.canonical_orbits_enabled;
        if (use_canonical_q_orbits) {
            const std::size_t begin = q_orbit_plan.x_offsets[ix];
            const std::size_t end = q_orbit_plan.x_offsets[ix + 1];
            for (std::size_t ordinal = begin; ordinal < end; ++ordinal) {
                process_full_tensor_one_q_orbit_triangular_batched_reassociated(
                    support, sj, B, seed_ws, local_ws, batch, reverse_map,
                    q_orbit_plan.orbits[ordinal], true,
                    B_upper_task, profile, opt);
            }
        } else {
            for (std::size_t il35 = 0; il35 < L35; ++il35) {
                for (std::size_t il24 = 0; il24 < L24; ++il24) {
                    for (std::size_t il34 = 0; il34 < L34; ++il34) {
                        process_full_tensor_one_q_block_triangular_batched_reassociated(
                            support, sj, B, seed_ws, local_ws, batch, reverse_map,
                            ix, il35, il24, il34,
                            B_upper_task, profile, opt);
                    }
                }
            }
        }
    }

    flush_full_tensor_triangular_batched_lr(batch, B_upper_task, profile);
}

void process_full_tensor_xrange_task_reassociated_rectangular_batched(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    std::size_t ii1,
    std::size_t ix_begin,
    std::size_t ix_end,
    double* B4FULL_RESTRICT A_panel_task,
    FullTensorPackedContractionL25ScalarOptions opt,
    const FullTensorReflectionCompactTriangularBatchedOptions& topt,
    FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    FullTensorReassociatedRectangularBatchedLRWorkspace& batch,
    FullTensorPackedContractionL25ScalarProfile& profile)
{
    (void)topt;
    const std::size_t L35 = support.l35.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();
    const std::size_t M = support.channels.i2.size() * support.channels.i3.size();
    const std::size_t N = support.channels.i4.size() * support.channels.i5.size();

    {
        FullTensorScalarScopedTimer timer(profile.zero_output);
        full_scalar_profile_zero(A_panel_task, M * N);
    }

    batch.begin_task();

    for (std::size_t ix = ix_begin; ix < ix_end; ++ix) {
        if (!full_tensor_cpu_i1_x_has_seed_wings(support, ii1, ix)) {
            continue;
        }
        build_full_seed_caches_l25_scalar_profiled(
            support, sj, B, ii1, ix, seed_ws, profile, opt);

        for (std::size_t il35 = 0; il35 < L35; ++il35) {
            for (std::size_t il24 = 0; il24 < L24; ++il24) {
                for (std::size_t il34 = 0; il34 < L34; ++il34) {
                    process_full_tensor_one_q_block_reassociated_rectangular_batched(
                        support, sj, B, seed_ws, local_ws, batch,
                        ix, il35, il24, il34,
                        A_panel_task, profile, opt);
                }
            }
        }
    }

    flush_full_tensor_reassociated_rectangular_batched_lr(batch, A_panel_task, profile);
}

void process_full_tensor_xrange_task_triangular_batched_reassociated_add(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    std::size_t ii1,
    std::size_t ix_begin,
    std::size_t ix_end,
    double* B4FULL_RESTRICT B_upper_task,
    const FullTensorReflectionCompactOutputMap& compact_map,
    const FullTensorTriangularReverseColumnMap& reverse_map,
    const FullTensorQReflectionOrbitPlan& q_orbit_plan,
    FullTensorPackedContractionL25ScalarOptions opt,
    const FullTensorReflectionCompactTriangularBatchedOptions& topt,
    FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    FullTensorTriangularBatchedLRWorkspace& batch,
    FullTensorPackedContractionL25ScalarProfile& profile)
{
    (void)compact_map;
    const std::size_t L35 = support.l35.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();

    batch.begin_task();
    batch.final_tile_shape = topt.final_tile_shape;

    for (std::size_t ix = ix_begin; ix < ix_end; ++ix) {
        if (!full_tensor_cpu_i1_x_has_seed_wings(support, ii1, ix)) {
            continue;
        }
        build_full_seed_caches_l25_scalar_profiled(
            support, sj, B, ii1, ix, seed_ws, profile, opt);

        const bool use_canonical_q_orbits =
            opt.use_cpu_q_reflection_orbits &&
            q_orbit_plan.canonical_orbits_enabled;
        if (use_canonical_q_orbits) {
            const std::size_t begin = q_orbit_plan.x_offsets[ix];
            const std::size_t end = q_orbit_plan.x_offsets[ix + 1];
            for (std::size_t ordinal = begin; ordinal < end; ++ordinal) {
                process_full_tensor_one_q_orbit_triangular_batched_reassociated(
                    support, sj, B, seed_ws, local_ws, batch, reverse_map,
                    q_orbit_plan.orbits[ordinal], true,
                    B_upper_task, profile, opt);
            }
        } else {
            for (std::size_t il35 = 0; il35 < L35; ++il35) {
                for (std::size_t il24 = 0; il24 < L24; ++il24) {
                    for (std::size_t il34 = 0; il34 < L34; ++il34) {
                        process_full_tensor_one_q_block_triangular_batched_reassociated(
                            support, sj, B, seed_ws, local_ws, batch, reverse_map,
                            ix, il35, il24, il34,
                            B_upper_task, profile, opt);
                    }
                }
            }
        }
    }

    flush_full_tensor_triangular_batched_lr(batch, B_upper_task, profile);
}

void process_full_tensor_xrange_task_reassociated_rectangular_batched_add(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    std::size_t ii1,
    std::size_t ix_begin,
    std::size_t ix_end,
    double* B4FULL_RESTRICT A_panel_task,
    FullTensorPackedContractionL25ScalarOptions opt,
    const FullTensorReflectionCompactTriangularBatchedOptions& topt,
    FullTensorPackedContractionL25ScalarWorkspace& seed_ws,
    FullTensorPackedContractionL25ScalarWorkspace& local_ws,
    FullTensorReassociatedRectangularBatchedLRWorkspace& batch,
    FullTensorPackedContractionL25ScalarProfile& profile)
{
    (void)topt;
    const std::size_t L35 = support.l35.size();
    const std::size_t L24 = support.l24.size();
    const std::size_t L34 = support.l34.size();

    batch.begin_task();

    for (std::size_t ix = ix_begin; ix < ix_end; ++ix) {
        if (!full_tensor_cpu_i1_x_has_seed_wings(support, ii1, ix)) {
            continue;
        }
        build_full_seed_caches_l25_scalar_profiled(
            support, sj, B, ii1, ix, seed_ws, profile, opt);

        for (std::size_t il35 = 0; il35 < L35; ++il35) {
            for (std::size_t il24 = 0; il24 < L24; ++il24) {
                for (std::size_t il34 = 0; il34 < L34; ++il34) {
                    process_full_tensor_one_q_block_reassociated_rectangular_batched(
                        support, sj, B, seed_ws, local_ws, batch,
                        ix, il35, il24, il34,
                        A_panel_task, profile, opt);
                }
            }
        }
    }

    flush_full_tensor_reassociated_rectangular_batched_lr(batch, A_panel_task, profile);
}

b4gpu::FullTensorGpuNativeOutputLayout
make_full_tensor_gpu_native_output_layout(
    const FullTensorInputSupportPlan& support,
    const FullTensorReassociatedAutoOutputPlan& plan)
{
    b4gpu::FullTensorGpuNativeOutputLayout layout;
    layout.M = plan.M;
    layout.N = plan.N;

    if (!plan.uses_reflection_compact()) {
        // General spins use the retained materialized GPU contraction with
        // every logical q block and a rectangular destination. Reflection
        // metadata is only needed for the compact schedule and epilogue.
        layout.kind = b4gpu::FullTensorGpuNativeOutputKind::RectangularByI1;
        return layout;
    }
    if (!support.reflection_15432_preserves_spins) {
        throw b4utils::InternalFailure(
            "CUDA native compact output requires reflection-compatible spins");
    }
    if (plan.M > std::numeric_limits<std::uint32_t>::max() ||
        plan.N > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error(
            "CUDA native compact full-tensor representatives exceed uint32_t");
    }

    const FullTensorTriangularReverseColumnMap reverse_map =
        build_full_tensor_triangular_reverse_column_map(support);
    if (!reverse_map.compatible || reverse_map.M != plan.M ||
        reverse_map.N != plan.N || plan.M != plan.N ||
        reverse_map.s_to_n.size() != plan.M ||
        reverse_map.n_to_s.size() != plan.N) {
        throw b4utils::InternalFailure(
            "CUDA native full-tensor compact output has no compatible reverse-column map");
    }

    layout.kind = b4gpu::FullTensorGpuNativeOutputKind::CompactByI1;
    layout.representative_m.reserve(plan.compact_representatives);
    layout.representative_n.reserve(plan.compact_representatives);
    layout.representative_partner_m.reserve(plan.compact_representatives);
    layout.representative_partner_n.reserve(plan.compact_representatives);

    const std::size_t I2 = support.channels.i2.size();
    const std::size_t I3 = support.channels.i3.size();
    const std::size_t I4 = support.channels.i4.size();
    const std::size_t I5 = support.channels.i5.size();

    // Public compact storage is formula-indexed upper triangular: upper(m,s),
    // with s the reflected right-column index.  Populate direct and reflected
    // rectangular coordinates in exactly that arithmetic order.
    for (std::size_t s = 0; s < plan.M; ++s) {
        const std::size_t n = reverse_map.s_to_n[s];
        if (n >= plan.N || reverse_map.n_to_s[n] != s) {
            throw b4utils::InternalFailure(
                "CUDA native compact reverse-column map is not bijective");
        }
        for (std::size_t m = 0; m <= s; ++m) {
            const std::size_t partner_m = s;
            const std::size_t partner_n = reverse_map.s_to_n[m];
            if (partner_n >= plan.N ||
                reverse_map.n_to_s[partner_n] != m) {
                throw b4utils::InternalFailure(
                    "CUDA native compact partner map is not involutive");
            }

            // Validate the coordinate formula against channel values, not
            // merely against equal extents.  This catches a future change in
            // channel ordering before a paired-q epilogue can corrupt output.
            if (plan.M != 0) {
                if (I2 == 0 || I3 == 0 || I4 == 0 || I5 == 0 ||
                    m / I2 >= I3 || partner_m / I2 >= I3 ||
                    n / I4 >= I5 || partner_n / I4 >= I5 ||
                    support.channels.i2[partner_m % I2] !=
                        support.channels.i5[n / I4] ||
                    support.channels.i3[partner_m / I2] !=
                        support.channels.i4[n % I4] ||
                    support.channels.i4[partner_n % I4] !=
                        support.channels.i3[m / I2] ||
                    support.channels.i5[partner_n / I4] !=
                        support.channels.i2[m % I2]) {
                    throw b4utils::InternalFailure(
                        "CUDA native compact partner coordinates do not implement reflection 15432");
                }
            }

            layout.representative_m.push_back(static_cast<std::uint32_t>(m));
            layout.representative_n.push_back(static_cast<std::uint32_t>(n));
            layout.representative_partner_m.push_back(
                static_cast<std::uint32_t>(partner_m));
            layout.representative_partner_n.push_back(
                static_cast<std::uint32_t>(partner_n));
        }
    }

    if (layout.representative_m.size() != plan.compact_representatives) {
        throw b4utils::InternalFailure(
            "CUDA native compact representative cardinality mismatch");
    }
    layout.reflection_orbit_compatible = true;
    if (!layout.has_valid_reflection_orbit_metadata()) {
        throw b4utils::InternalFailure(
            "CUDA native compact reflection metadata failed validation");
    }
    return layout;
}

FullTensorReassociatedAutoOutputPlan
contract_full_tensor_packed_fused_l25_stacked_scalar_profiled_reassociated_auto_scheduled_by_i1_layout(
    const FullTensorInputSupportPlan& support,
    const PackedFullTensorSJTensors& sj,
    const PackedFullTensorBoosters& B,
    double* B4FULL_RESTRICT A_auto_by_i1,
    FullTensorPackedContractionL25ScalarOptions opt,
    FullTensorPackedContractionL25ScalarProfile& profile,
    FullTensorReflectionCompactTriangularBatchedOptions topt,
    FullTensorAdaptiveScheduleOptions sopt,
    FullTensorReflectionCompactWeightedProgressOptions progress_options,
    FullTensorScheduledBalanceProfile* balance_profile)
{
    profile.CT2_zero.name = "D4_zero";
    profile.CT2_scalar.name = "D4_stack";

    const FullTensorReassociatedAutoOutputPlan plan =
        make_full_tensor_reassociated_auto_output_plan(support);

    const std::size_t I1 = support.channels.i1.size();
    const std::size_t X = support.channels.xvals.size();
    const std::size_t M = support.channels.i2.size() * support.channels.i3.size();
    const std::size_t N = support.channels.i4.size() * support.channels.i5.size();
    assert(A_auto_by_i1 != nullptr);

    const auto wall_start = std::chrono::high_resolution_clock::now();

    const FullTensorReflectionCompactTriangularCacheInfo cache =
        full_tensor_detect_cache_info_triangular_batched();

    // Use the same conservative ceiling and P_batch calculation as host-memory
    // admission.  The task builder may cap this further, but can never exceed
    // the number of worker batches admitted by the planner.
    const FullTensorCpuBatchedWorkspaceAdmission batch_admission =
        make_full_tensor_cpu_batched_workspace_admission(
            support, opt, topt, cache);
    std::size_t active_workers = batch_admission.worker_ceiling;
    const std::size_t p_batch = batch_admission.p_batch;

    FullTensorReflectionCompactOutputMap compact_map;
    FullTensorTriangularReverseColumnMap reverse_map;
    std::size_t tri_size = 0;
    if (plan.uses_reflection_compact()) {
        compact_map = build_full_tensor_reflection_compact_output_map(support);
        reverse_map = build_full_tensor_triangular_reverse_column_map(support);
        if (!compact_map.compatible || !reverse_map.compatible || reverse_map.M != compact_map.M()) {
            throw b4utils::InternalFailure(
                "scheduled compact path selected but compact/reverse maps are not compatible.");
        }
        tri_size = full_tensor_triangular_upper_size(compact_map.M());
        if (compact_map.independent_entries() != tri_size) {
            throw b4utils::InternalFailure(
                "scheduled compact path selected but triangular representative count is inconsistent.");
        }
    }

    const FullTensorQReflectionOrbitPlan q_orbit_plan =
        build_full_tensor_q_reflection_orbit_plan(support);

    std::vector<std::uint64_t> q_weights;
    if (plan.uses_reflection_compact()) {
        if (opt.use_cpu_q_reflection_orbits &&
            q_orbit_plan.canonical_orbits_enabled) {
            (void)build_full_tensor_cpu_q_orbit_triangular_batched_reassociated_q_weights(
                support, compact_map, q_orbit_plan, q_weights);
        } else {
            (void)build_full_tensor_reflection_compact_triangular_batched_reassociated_q_weights(
                support, compact_map, q_weights);
        }
    } else {
        (void)build_full_tensor_reassociated_rectangular_batched_q_weights(
            support, q_weights);
    }

    double total_predicted_work = 0.0;
    double target_task_work = 0.0;
    std::size_t whole_i1_tasks = 0;
    std::size_t split_i1_panels = 0;
    double i1_tasks_per_worker = 0.0;
    double predicted_i1_max_over_mean = 0.0;
    double predicted_i1_max_over_target = 0.0;
    double predicted_i1_max_work = 0.0;
    bool heavy_i1_forced_split = false;

    FullTensorAdaptiveScheduleMode selected_mode = sopt.mode;
    if (selected_mode == FullTensorAdaptiveScheduleMode::Auto) {
        const double ratio = static_cast<double>(I1) / static_cast<double>(std::max<std::size_t>(1, active_workers));
        // Initial decision from ratio. The predicted imbalance is computed inside
        // the builder, so the ratio rule is intentionally conservative. If I1
        // gives at least about 1.5 tasks per worker, whole-i1 tends to preserve
        // enough locality to beat chunking on small CPUs.
        selected_mode = (ratio >= sopt.i1_only_min_tasks_per_worker)
            ? FullTensorAdaptiveScheduleMode::WeightedI1Only
            : FullTensorAdaptiveScheduleMode::WeightedXRangeTLS;
    }

    std::vector<FullTensorScheduledTask> tasks = build_full_tensor_adaptive_scheduled_tasks(
        support, plan, q_weights, active_workers, p_batch, sopt,
        total_predicted_work, target_task_work, whole_i1_tasks, split_i1_panels,
        selected_mode, i1_tasks_per_worker, predicted_i1_max_over_mean,
        predicted_i1_max_over_target, predicted_i1_max_work);

    // If Auto selected whole-i1 but a full i1 panel is predicted to be huge,
    // rebuild with weighted x-range TLS chunks.  This catches large-Dl / large-M
    // cases where I1/tasks-per-worker looks fine, but each i1 task would run for
    // minutes before producing progress or sharing work.
    const bool heavy_i1_by_absolute_work =
        sopt.split_heavy_i1_by_predicted_work &&
        sopt.heavy_i1_predicted_work_threshold > 0.0 &&
        predicted_i1_max_work > sopt.heavy_i1_predicted_work_threshold &&
        predicted_i1_max_over_target > sopt.i1_only_max_work_over_target;

    const bool extreme_i1_imbalance =
        predicted_i1_max_over_mean > sopt.i1_only_max_imbalance &&
        i1_tasks_per_worker < 2.0;

    if (sopt.mode == FullTensorAdaptiveScheduleMode::Auto &&
        selected_mode == FullTensorAdaptiveScheduleMode::WeightedI1Only &&
        X > 1 && (heavy_i1_by_absolute_work || extreme_i1_imbalance)) {
        heavy_i1_forced_split = heavy_i1_by_absolute_work;
        selected_mode = FullTensorAdaptiveScheduleMode::WeightedXRangeTLS;
        tasks = build_full_tensor_adaptive_scheduled_tasks(
            support, plan, q_weights, active_workers, p_batch, sopt,
            total_predicted_work, target_task_work, whole_i1_tasks, split_i1_panels,
            selected_mode, i1_tasks_per_worker, predicted_i1_max_over_mean,
            predicted_i1_max_over_target, predicted_i1_max_work);
    }

    std::uint64_t task_final_rows = 0;
    if (progress_options.enabled) {
        const bool use_canonical_q_orbits =
            plan.uses_reflection_compact() &&
            opt.use_cpu_q_reflection_orbits &&
            q_orbit_plan.canonical_orbits_enabled;
        const std::vector<std::uint64_t> planned_final_rows_by_x =
            build_full_tensor_cpu_planned_final_rows_by_x(
                support, plan, q_orbit_plan, use_canonical_q_orbits);
        for (FullTensorScheduledTask& task : tasks) {
            std::uint64_t rows = 0;
            for (std::size_t ix = task.ix_begin; ix < task.ix_end; ++ix) {
                if (!full_tensor_cpu_i1_x_has_seed_wings(
                        support, task.ii1, ix)) {
                    continue;
                }
                rows = full_tensor_cpu_saturating_add_u64(
                    rows, planned_final_rows_by_x[ix]);
            }
            task.planned_final_rows = rows;
            task_final_rows = full_tensor_cpu_saturating_add_u64(
                task_final_rows, rows);
        }
    }

    const bool inner_tile_parallel_requested = (tasks.size() < active_workers);
    const bool inner_tile_parallel_enabled = false; // reserved for a later backend

    // The current implementation avoids oversubscribing outer OpenMP workers.
    // If there are fewer scheduled tasks than available workers, the summary
    // reports that inner-tile fallback would be useful, then caps active_workers
    // to the number of outer tasks. A later backend can fill this branch with
    // final/D4 tile-level parallelism.
    active_workers = std::min<std::size_t>(active_workers, std::max<std::size_t>(1, tasks.size()));

    if (topt.print_cache_info || sopt.print_schedule_summary) {
        print_full_tensor_adaptive_schedule_summary(
            plan, cache, tasks, I1, X, M, N, p_batch, active_workers,
            total_predicted_work, target_task_work, whole_i1_tasks,
            split_i1_panels, selected_mode, i1_tasks_per_worker,
            predicted_i1_max_over_mean, predicted_i1_max_over_target,
            predicted_i1_max_work, heavy_i1_forced_split,
            inner_tile_parallel_requested, inner_tile_parallel_enabled,
            topt, sopt, std::cout);
    }

    const std::size_t output_values = plan.values_count;
    if (opt.zero_output) {
        FullTensorScalarScopedTimer timer(profile.zero_output);
        full_scalar_profile_zero(A_auto_by_i1, output_values);
    }

    // Final-matrix row work accounts for nearly all measured contraction time
    // and is published after each expensive batch flush returns. This provides
    // many truthful samples during the first OpenMP task wave. Reserve a small
    // final fraction until TLS reduction/locked cleanup has also finished.
    const std::uint64_t finalization_progress_units = std::max<std::uint64_t>(
        1, task_final_rows / 1000u);
    const std::uint64_t total_progress_units =
        full_tensor_cpu_saturating_add_u64(
            task_final_rows, finalization_progress_units);
    progress_options.unit_label = "row work";
    // Fine-grained executed/skipped row accounting makes cumulative progress
    // representative from the first second. A cumulative deadline is more
    // stable here than a short rolling window, which would mistake OpenMP tail
    // drain and bursts of structurally skipped rows for throughput changes.
    progress_options.eta_min_elapsed_seconds = 1.0;
    progress_options.eta_min_fraction = 0.0;
    FullTensorReflectionCompactWeightedProgressBar progress_bar(
        total_progress_units, std::move(progress_options));
    progress_bar.start();
    FullTensorReflectionCompactWeightedProgressBar* progress_ptr =
        progress_bar.enabled() ? &progress_bar : nullptr;

    std::vector<FullTensorPackedContractionL25ScalarProfile> task_profiles(tasks.size());
    FullTensorScheduledBalanceProfile* balance = balance_profile;
    if (balance != nullptr) {
        balance->reset(I1, X, active_workers, tasks.size());
        balance->selected_p_batch = p_batch;
        balance->selected_final_tile_shape = topt.final_tile_shape;
        balance->selected_final_tile_applicable =
            plan.uses_reflection_compact();
        balance->target_task_work = target_task_work;
        balance->total_predicted_work = total_predicted_work;
        balance->whole_i1_tasks = whole_i1_tasks;
        balance->split_i1_panels = split_i1_panels;
        balance->selected_mode = selected_mode;
        balance->auto_selected_i1_only = (sopt.mode == FullTensorAdaptiveScheduleMode::Auto &&
                                          selected_mode == FullTensorAdaptiveScheduleMode::WeightedI1Only);
        balance->i1_tasks_per_worker = i1_tasks_per_worker;
        balance->predicted_i1_max_over_mean = predicted_i1_max_over_mean;
        balance->predicted_i1_max_over_target = predicted_i1_max_over_target;
        balance->predicted_i1_max_work = predicted_i1_max_work;
        balance->heavy_i1_forced_split = heavy_i1_forced_split;
        balance->inner_tile_parallel_requested = inner_tile_parallel_requested;
        balance->inner_tile_parallel_enabled = inner_tile_parallel_enabled;
    }


    const std::size_t panel_elems = plan.uses_reflection_compact() ? tri_size : (M * N);
    const std::uint64_t worst_tls_bytes = static_cast<std::uint64_t>(active_workers)
        * static_cast<std::uint64_t>(I1)
        * static_cast<std::uint64_t>(panel_elems)
        * static_cast<std::uint64_t>(sizeof(double));

    bool use_tls_panels = sopt.use_thread_local_i1_panels;
    if (sopt.max_tls_output_scratch_bytes != 0 &&
        worst_tls_bytes > sopt.max_tls_output_scratch_bytes) {
        use_tls_panels = false;
    }

    std::size_t tls_panels_touched = 0;
    std::uint64_t tls_output_bytes = 0;
    double tls_reduction_seconds = 0.0;

    if (use_tls_panels) {
        // Lazy thread-local panels: slot (tid,ii1) is allocated only if the
        // dynamic scheduler actually assigns at least one task for that i1 to
        // that thread. This avoids the worst-case memory cost while eliminating
        // per-task global reductions and locks.
        std::vector<std::vector<double>> tls_output(active_workers * I1);

#ifdef _OPENMP
#pragma omp parallel num_threads(active_workers) if(opt.use_openmp && active_workers > 1)
#endif
        {
            FullTensorPackedContractionL25ScalarWorkspace seed_ws;
            seed_ws.ensure_seed_caches(support);
            FullTensorPackedContractionL25ScalarWorkspace local_ws;
            FullTensorTriangularBatchedLRWorkspace tri_batch;
            FullTensorReassociatedRectangularBatchedLRWorkspace rect_batch;

            if (plan.uses_reflection_compact()) {
                tri_batch.reset(p_batch, M);
                tri_batch.final_tile_shape = topt.final_tile_shape;
                tri_batch.progress = progress_ptr;
            } else {
                rect_batch.reset(p_batch, M, N);
                rect_batch.configure_cpu_final_accumulator(opt);
                rect_batch.progress = progress_ptr;
            }

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
            for (std::ptrdiff_t task_s = 0; task_s < static_cast<std::ptrdiff_t>(tasks.size()); ++task_s) {
                const std::size_t task_index = static_cast<std::size_t>(task_s);
                const FullTensorScheduledTask& task = tasks[task_index];
#ifdef _OPENMP
                const int tid = omp_get_thread_num();
#else
                const int tid = 0;
#endif
                FullTensorPackedContractionL25ScalarProfile& task_profile = task_profiles[task_index];
                task_profile.CT2_zero.name = "D4_zero";
                task_profile.CT2_scalar.name = "D4_stack";

                const auto task_start = std::chrono::high_resolution_clock::now();

                std::vector<double>& out_panel = tls_output[static_cast<std::size_t>(tid) * I1 + task.ii1];
                if (out_panel.empty()) {
                    FullTensorScalarScopedTimer zero_timer(task_profile.zero_output);
                    out_panel.assign(panel_elems, 0.0);
                }

                if (plan.uses_reflection_compact()) {
                    process_full_tensor_xrange_task_triangular_batched_reassociated_add(
                        support, sj, B, task.ii1, task.ix_begin, task.ix_end,
                        out_panel.data(), compact_map, reverse_map, q_orbit_plan,
                        opt, topt, seed_ws, local_ws, tri_batch, task_profile);
                } else {
                    process_full_tensor_xrange_task_reassociated_rectangular_batched_add(
                        support, sj, B, task.ii1, task.ix_begin, task.ix_end,
                        out_panel.data(), opt, topt,
                        seed_ws, local_ws, rect_batch, task_profile);
                }

                // Batch flushes report executed final rows, while q support
                // checks report skipped planned rows. Reconcile any residual
                // metadata-only gap so the fixed work denominator stays exact.
                if (progress_ptr != nullptr) {
                    const std::uint64_t actual_rows = panel_elems != 0
                        ? task_profile.support.final_dot_terms /
                              static_cast<std::uint64_t>(panel_elems)
                        : 0;
                    const std::uint64_t skipped_rows =
                        plan.uses_reflection_compact()
                            ? tri_batch.skipped_rows_total
                            : rect_batch.skipped_rows_total;
#ifndef NDEBUG
                    const std::uint64_t accounted_rows =
                        full_tensor_cpu_saturating_add_u64(
                            actual_rows, skipped_rows);
                    assert(accounted_rows <= task.planned_final_rows);
#endif
                    progress_ptr->add(
                        full_tensor_cpu_progress_residual_rows(
                            task.planned_final_rows,
                            actual_rows,
                            skipped_rows));
                }

                const auto task_end = std::chrono::high_resolution_clock::now();
                const double task_seconds = std::chrono::duration<double>(task_end - task_start).count();
                if (balance != nullptr) {
                    FullTensorScheduledTaskRecord& rec = balance->tasks[task_index];
                    rec.ii1 = task.ii1;
                    rec.ix_begin = task.ix_begin;
                    rec.ix_end = task.ix_end;
                    rec.thread_id = tid;
                    rec.whole_i1 = task.whole_i1;
                    rec.predicted_work = task.predicted_work;
                    rec.predicted_units = task.predicted_units;
                    rec.predicted_p_rows = task.predicted_p_rows;
                    rec.seconds = task_seconds;
                    rec.q_blocks_accumulated = task_profile.support.q_blocks_accumulated;
                    rec.final_dot_terms = task_profile.support.final_dot_terms;
                    rec.T2_dot_terms = task_profile.support.T2_dot_terms;
                    rec.T4_dot_terms = task_profile.support.T4_dot_terms;
                    rec.middle_updates = task_profile.support.CT2_rank1_updates;
                    rec.approx_flops = task_profile.T1_seed.approx_flops
                                     + task_profile.T3_seed.approx_flops
                                     + task_profile.T2_stack.approx_flops
                                     + task_profile.T4_stack.approx_flops
                                     + task_profile.CT2_scalar.approx_flops
                                     + task_profile.final_scalar.approx_flops;
                }
            }
        }

        const auto red_start = std::chrono::high_resolution_clock::now();
        const FullTensorTlsPanelReductionStats reduction_stats =
            full_tensor_reduce_tls_i1_panels_deterministic(
                tls_output, active_workers, I1, panel_elems,
                A_auto_by_i1, opt.use_openmp);
        tls_panels_touched = reduction_stats.panels_touched;
        tls_output_bytes = reduction_stats.panel_bytes;
        const auto red_end = std::chrono::high_resolution_clock::now();
        tls_reduction_seconds = std::chrono::duration<double>(red_end - red_start).count();
    } else {
        // Fallback path: reduce each task immediately under an i1 lock. This is
        // less scalable, but has a lower peak scratch footprint when the user
        // sets a tight max_tls_output_scratch_bytes budget.
#ifdef _OPENMP
        std::vector<omp_lock_t> i1_locks(I1);
        if (opt.use_openmp && active_workers > 1) {
            for (std::size_t ii1 = 0; ii1 < I1; ++ii1) {
                omp_init_lock(&i1_locks[ii1]);
            }
        }
#endif

#ifdef _OPENMP
#pragma omp parallel num_threads(active_workers) if(opt.use_openmp && active_workers > 1)
#endif
        {
            FullTensorPackedContractionL25ScalarWorkspace seed_ws;
            seed_ws.ensure_seed_caches(support);
            FullTensorPackedContractionL25ScalarWorkspace local_ws;
            FullTensorTriangularBatchedLRWorkspace tri_batch;
            FullTensorReassociatedRectangularBatchedLRWorkspace rect_batch;
            std::vector<double> partial_output;

            if (plan.uses_reflection_compact()) {
                partial_output.assign(tri_size, 0.0);
                tri_batch.reset(p_batch, M);
                tri_batch.final_tile_shape = topt.final_tile_shape;
                tri_batch.progress = progress_ptr;
            } else {
                partial_output.assign(M * N, 0.0);
                rect_batch.reset(p_batch, M, N);
                rect_batch.configure_cpu_final_accumulator(opt);
                rect_batch.progress = progress_ptr;
            }

#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
            for (std::ptrdiff_t task_s = 0; task_s < static_cast<std::ptrdiff_t>(tasks.size()); ++task_s) {
                const std::size_t task_index = static_cast<std::size_t>(task_s);
                const FullTensorScheduledTask& task = tasks[task_index];
#ifdef _OPENMP
                const int tid = omp_get_thread_num();
#else
                const int tid = 0;
#endif
                FullTensorPackedContractionL25ScalarProfile& task_profile = task_profiles[task_index];
                task_profile.CT2_zero.name = "D4_zero";
                task_profile.CT2_scalar.name = "D4_stack";

                const auto task_start = std::chrono::high_resolution_clock::now();

                if (plan.uses_reflection_compact()) {
                    process_full_tensor_xrange_task_triangular_batched_reassociated(
                        support, sj, B, task.ii1, task.ix_begin, task.ix_end,
                        partial_output.data(), compact_map, reverse_map,
                        q_orbit_plan,
                        opt, topt, seed_ws, local_ws, tri_batch, task_profile);

#ifdef _OPENMP
                    if (opt.use_openmp && active_workers > 1) omp_set_lock(&i1_locks[task.ii1]);
#endif
                    full_tensor_i1x_add_vector_inplace(
                        A_auto_by_i1 + task.ii1 * tri_size,
                        partial_output.data(), tri_size);
#ifdef _OPENMP
                    if (opt.use_openmp && active_workers > 1) omp_unset_lock(&i1_locks[task.ii1]);
#endif
                } else {
                    process_full_tensor_xrange_task_reassociated_rectangular_batched(
                        support, sj, B, task.ii1, task.ix_begin, task.ix_end,
                        partial_output.data(), opt, topt,
                        seed_ws, local_ws, rect_batch, task_profile);

#ifdef _OPENMP
                    if (opt.use_openmp && active_workers > 1) omp_set_lock(&i1_locks[task.ii1]);
#endif
                    full_tensor_i1x_add_vector_inplace(
                        A_auto_by_i1 + (M * N) * task.ii1,
                        partial_output.data(), M * N);
#ifdef _OPENMP
                    if (opt.use_openmp && active_workers > 1) omp_unset_lock(&i1_locks[task.ii1]);
#endif
                }

                // On the bounded-memory fallback, reconciliation occurs only
                // after the locked merge into the shared output panel above.
                if (progress_ptr != nullptr) {
                    const std::uint64_t actual_rows = panel_elems != 0
                        ? task_profile.support.final_dot_terms /
                              static_cast<std::uint64_t>(panel_elems)
                        : 0;
                    const std::uint64_t skipped_rows =
                        plan.uses_reflection_compact()
                            ? tri_batch.skipped_rows_total
                            : rect_batch.skipped_rows_total;
#ifndef NDEBUG
                    const std::uint64_t accounted_rows =
                        full_tensor_cpu_saturating_add_u64(
                            actual_rows, skipped_rows);
                    assert(accounted_rows <= task.planned_final_rows);
#endif
                    progress_ptr->add(
                        full_tensor_cpu_progress_residual_rows(
                            task.planned_final_rows,
                            actual_rows,
                            skipped_rows));
                }

                const auto task_end = std::chrono::high_resolution_clock::now();
                const double task_seconds = std::chrono::duration<double>(task_end - task_start).count();
                if (balance != nullptr) {
                    FullTensorScheduledTaskRecord& rec = balance->tasks[task_index];
                    rec.ii1 = task.ii1;
                    rec.ix_begin = task.ix_begin;
                    rec.ix_end = task.ix_end;
                    rec.thread_id = tid;
                    rec.whole_i1 = task.whole_i1;
                    rec.predicted_work = task.predicted_work;
                    rec.predicted_units = task.predicted_units;
                    rec.predicted_p_rows = task.predicted_p_rows;
                    rec.seconds = task_seconds;
                    rec.q_blocks_accumulated = task_profile.support.q_blocks_accumulated;
                    rec.final_dot_terms = task_profile.support.final_dot_terms;
                    rec.T2_dot_terms = task_profile.support.T2_dot_terms;
                    rec.T4_dot_terms = task_profile.support.T4_dot_terms;
                    rec.middle_updates = task_profile.support.CT2_rank1_updates;
                    rec.approx_flops = task_profile.T1_seed.approx_flops
                                     + task_profile.T3_seed.approx_flops
                                     + task_profile.T2_stack.approx_flops
                                     + task_profile.T4_stack.approx_flops
                                     + task_profile.CT2_scalar.approx_flops
                                     + task_profile.final_scalar.approx_flops;
                }
            }
        }

#ifdef _OPENMP
        if (opt.use_openmp && active_workers > 1) {
            for (std::size_t ii1 = 0; ii1 < I1; ++ii1) {
                omp_destroy_lock(&i1_locks[ii1]);
            }
        }
#endif
    }

    if (balance != nullptr) {
        balance->used_thread_local_i1_panels = use_tls_panels;
        balance->tls_panels_touched = tls_panels_touched;
        balance->tls_output_bytes = tls_output_bytes;
        balance->tls_reduction_seconds = tls_reduction_seconds;
    }
    if (progress_ptr != nullptr) {
        progress_ptr->add(finalization_progress_units);
    }
    progress_bar.stop();

    // Compact reductions already accumulate directly into the public
    // formula-indexed triangular panels.  No output-sized staging buffer or
    // final copy is required.

    for (std::size_t task = 0; task < tasks.size(); ++task) {
        profile.add(task_profiles[task]);
    }
    profile.CT2_zero.name = "D4_zero";
    profile.CT2_scalar.name = "D4_stack";

    const auto wall_end = std::chrono::high_resolution_clock::now();
    profile.total_wall.seconds += std::chrono::duration<double>(wall_end - wall_start).count();
    profile.total_wall.calls += 1;
    profile.total_wall.approx_flops = profile.T1_seed.approx_flops
                                    + profile.T3_seed.approx_flops
                                    + profile.T2_stack.approx_flops
                                    + profile.T4_stack.approx_flops
                                    + profile.CT2_scalar.approx_flops
                                    + profile.final_scalar.approx_flops;

    return plan;
}

} // namespace b4support
