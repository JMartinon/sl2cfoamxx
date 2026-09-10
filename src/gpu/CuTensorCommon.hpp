// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cutensor.h>

#include <mutex>
#include <stdexcept>
#include <string>

namespace b4gpu::cutensor_detail {

inline void check(cutensorStatus_t status, const char* operation)
{
    if (status != CUTENSOR_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed: " + cutensorGetErrorString(status));
    }
}

// cuTENSOR's JIT cache is process-global. Both coherent-amplitude and
// full-tensor plan construction must therefore use this same mutex; separate
// per-backend mutexes would still allow the vendor planner to run concurrently.
inline std::mutex& planning_mutex()
{
    static std::mutex mutex;
    return mutex;
}

inline const char* algorithm_name(cutensorAlgo_t algorithm) noexcept
{
    switch (algorithm) {
    case CUTENSOR_ALGO_DEFAULT: return "default";
    case CUTENSOR_ALGO_GETT: return "gett";
    case CUTENSOR_ALGO_TGETT: return "tgett";
    case CUTENSOR_ALGO_TTGT: return "ttgt";
    default: return "unknown";
    }
}

} // namespace b4gpu::cutensor_detail
