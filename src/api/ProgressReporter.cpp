// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/ProgressReporter.hpp"

#include "full_tensor/CpuContraction.hpp"

#include <utility>

namespace sl2cfoamxx::detail {
namespace {

[[nodiscard]] b4support::FullTensorReflectionCompactWeightedProgressOptions
convert_options(ProgressReporterOptions options)
{
    b4support::FullTensorReflectionCompactWeightedProgressOptions converted;
    converted.enabled = options.enabled;
    converted.render_output = options.render_output;
    converted.callback = std::move(options.callback);
    converted.operation = options.operation;
    converted.phase = options.phase;
    converted.backend_kind = options.backend_kind;
    converted.update_granularity = options.update_granularity;
    converted.refresh_seconds = options.refresh_seconds;
    converted.label = std::move(options.label);
    converted.unit_label = std::move(options.unit_label);
    converted.stream = options.stream;
    converted.presentation = options.presentation;
    converted.operation_label = std::move(options.operation_label);
    converted.backend = std::move(options.backend);
    converted.phase_label = std::move(options.phase_label);
    converted.bar_width = options.bar_width;
    converted.clear_on_success = options.clear_on_success;
    converted.eta_smoothing = options.eta_smoothing;
    converted.eta_min_elapsed_seconds = options.eta_min_elapsed_seconds;
    converted.eta_min_fraction = options.eta_min_fraction;
    converted.eta_max_display_seconds = options.eta_max_display_seconds;
    return converted;
}

} // namespace

class ProgressReporter::Impl {
public:
    Impl(std::uint64_t total_work, ProgressReporterOptions options)
        : reporter_(total_work, convert_options(std::move(options)))
    {
    }

    b4support::FullTensorReflectionCompactWeightedProgressBar reporter_;
};

ProgressReporter::ProgressReporter(
    std::uint64_t total_work,
    ProgressReporterOptions options)
    : implementation_(
          std::make_unique<Impl>(total_work, std::move(options)))
{
}

ProgressReporter::~ProgressReporter() = default;

bool ProgressReporter::enabled() const noexcept
{
    return implementation_->reporter_.enabled();
}

std::uint64_t ProgressReporter::update_granularity() const noexcept
{
    return implementation_->reporter_.update_granularity();
}

void ProgressReporter::set_total(std::uint64_t total_work) noexcept
{
    implementation_->reporter_.set_total(total_work);
}

void ProgressReporter::start() noexcept
{
    implementation_->reporter_.start();
}

void ProgressReporter::add(std::uint64_t completed_units) noexcept
{
    implementation_->reporter_.add(completed_units);
}

void ProgressReporter::reset() noexcept
{
    implementation_->reporter_.reset();
}

void ProgressReporter::stop(bool completed) noexcept
{
    implementation_->reporter_.stop(completed);
}

} // namespace sl2cfoamxx::detail
