// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "core/OutputPresentation.hpp"
#include "sl2cfoamxx/progress.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>

namespace sl2cfoamxx::detail {

// Operation-neutral facade over the shared progress engine. Numerical
// backends report only completed units; presentation, ETA estimation, retry
// accounting, and terminal coordination remain centralized here.
struct ProgressReporterOptions {
    bool enabled = false;
    bool render_output = true;
    ProgressCallback callback{};
    ProgressOperation operation = ProgressOperation::coherent_amplitude;
    ProgressPhase phase = ProgressPhase::contraction;
    Backend backend_kind = Backend::CPU;
    std::uint64_t update_granularity = 1;
    double refresh_seconds = 0.25;
    std::string label = "operation";
    std::string unit_label = "work";
    std::ostream* stream = nullptr;
    OutputPresentation presentation{};
    std::string operation_label;
    std::string backend;
    std::string phase_label = "contract";
    std::size_t bar_width = 22;
    bool clear_on_success = false;
    double eta_smoothing = 0.20;
    double eta_min_elapsed_seconds = 2.0;
    double eta_min_fraction = 0.01;
    double eta_max_display_seconds = 30.0 * 24.0 * 3600.0;
};

class ProgressReporter {
public:
    ProgressReporter(
        std::uint64_t total_work,
        ProgressReporterOptions options);
    ~ProgressReporter();

    ProgressReporter(const ProgressReporter&) = delete;
    ProgressReporter& operator=(const ProgressReporter&) = delete;

    [[nodiscard]] bool enabled() const noexcept;
    [[nodiscard]] std::uint64_t update_granularity() const noexcept;

    void set_total(std::uint64_t total_work) noexcept;
    void start() noexcept;
    void add(std::uint64_t completed_units) noexcept;
    void reset() noexcept;
    void stop(bool completed = true) noexcept;

private:
    class Impl;
    std::unique_ptr<Impl> implementation_;
};

} // namespace sl2cfoamxx::detail
