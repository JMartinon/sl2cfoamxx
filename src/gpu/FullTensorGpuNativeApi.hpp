// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "gpu/FullTensorGpuApi.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace b4gpu {

// Storage descriptor shared by the CPU-side compact-layout constructor and
// the retained semi-fused production CUDA bridge.
enum class FullTensorGpuNativeOutputKind : std::uint8_t {
    RectangularByI1,
    CompactByI1
};

struct FullTensorGpuNativeOutputLayout {
    FullTensorGpuNativeOutputKind kind =
        FullTensorGpuNativeOutputKind::RectangularByI1;
    std::size_t M = 0;
    std::size_t N = 0;
    std::vector<std::uint32_t> representative_m;
    std::vector<std::uint32_t> representative_n;
    bool reflection_orbit_compatible = false;
    std::vector<std::uint32_t> representative_partner_m;
    std::vector<std::uint32_t> representative_partner_n;

    [[nodiscard]] bool has_valid_reflection_orbit_metadata() const noexcept
    {
        if (kind != FullTensorGpuNativeOutputKind::CompactByI1 ||
            !reflection_orbit_compatible ||
            representative_m.size() != representative_n.size() ||
            representative_m.size() != representative_partner_m.size() ||
            representative_m.size() != representative_partner_n.size()) {
            return false;
        }
        for (std::size_t i = 0; i < representative_m.size(); ++i) {
            if (representative_m[i] >= M ||
                representative_partner_m[i] >= M ||
                representative_n[i] >= N ||
                representative_partner_n[i] >= N) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] std::size_t values_per_i1() const noexcept
    {
        if (kind != FullTensorGpuNativeOutputKind::RectangularByI1) {
            return representative_m.size();
        }
        return M != 0 && N > std::numeric_limits<std::size_t>::max() / M
            ? 0
            : M * N;
    }
};

} // namespace b4gpu
