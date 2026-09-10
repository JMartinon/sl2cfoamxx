// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace sl2cfoamxx::detail {

struct ProgressEtaEstimate {
    double seconds = 0.0;
    bool available = false;
};

// Progress backends publish completed work in irregular chunks.  Estimating
// directly from completed / elapsed at every renderer refresh makes the ETA
// rise while a chunk is in flight and collapse when that chunk completes.
// Instead, update a predicted completion time only when new work arrives and
// count down from the smoothed prediction between completion events.
class ProgressEtaEstimator {
public:
    void reset() noexcept
    {
        last_observed_done_ = 0;
        meaningful_observations_ = 0;
        completion_seconds_ = 0.0;
        has_completion_ = false;
    }

    [[nodiscard]] ProgressEtaEstimate estimate(
        std::uint64_t completed,
        std::uint64_t total,
        double elapsed_seconds,
        double minimum_elapsed_seconds,
        double minimum_fraction,
        double smoothing,
        double maximum_eta_seconds) noexcept
    {
        if (total == 0 || !std::isfinite(elapsed_seconds) ||
            elapsed_seconds < 0.0) {
            return {};
        }
        completed = std::min(completed, total);

        // A backend retry is expected to call reset(), but defend against a
        // decreasing counter so stale ETA history can never cross attempts.
        if (completed < last_observed_done_) {
            reset();
        }

        if (completed > last_observed_done_) {
            last_observed_done_ = completed;
            const double fraction = static_cast<double>(completed) /
                static_cast<double>(total);
            const bool meaningful =
                completed < total &&
                elapsed_seconds >= std::max(0.0, minimum_elapsed_seconds) &&
                fraction >= std::max(0.0, minimum_fraction);
            if (meaningful) {
                const double raw_completion_seconds = elapsed_seconds /
                    fraction;
                const double raw_eta_seconds =
                    raw_completion_seconds - elapsed_seconds;
                if (std::isfinite(raw_completion_seconds) &&
                    std::isfinite(raw_eta_seconds) &&
                    raw_eta_seconds >= 0.0 &&
                    maximum_eta_seconds > 0.0 &&
                    raw_eta_seconds <= maximum_eta_seconds) {
                    observe_completion(raw_completion_seconds, smoothing);
                }
            }
        }

        if (!has_completion_ || completed == 0 || completed >= total) {
            return {};
        }
        const double eta_seconds = completion_seconds_ - elapsed_seconds;
        if (!std::isfinite(eta_seconds) || eta_seconds <= 0.0 ||
            maximum_eta_seconds <= 0.0 ||
            eta_seconds > maximum_eta_seconds) {
            return {};
        }
        return ProgressEtaEstimate{eta_seconds, true};
    }

private:
    std::uint64_t last_observed_done_ = 0;
    std::size_t meaningful_observations_ = 0;
    double completion_seconds_ = 0.0;
    bool has_completion_ = false;

    void observe_completion(
        double raw_completion_seconds,
        double smoothing) noexcept
    {
        ++meaningful_observations_;
        if (meaningful_observations_ == 1) {
            return;
        }
        if (meaningful_observations_ == 2) {
            // The second cumulative observation contains strictly more work
            // than the first. Use that newer deadline instead of averaging in
            // a tiny early completion whose extrapolation can be hundreds of
            // seconds too pessimistic for an otherwise short parallel run.
            completion_seconds_ = raw_completion_seconds;
            has_completion_ = true;
            return;
        }

        const double configured_alpha =
            smoothing > 0.0 && smoothing <= 1.0 ? smoothing : 0.20;
        completion_seconds_ =
            (1.0 - configured_alpha) * completion_seconds_ +
            configured_alpha * raw_completion_seconds;
    }
};

} // namespace sl2cfoamxx::detail
