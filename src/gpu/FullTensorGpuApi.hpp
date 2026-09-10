// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>

namespace b4gpu {

// Mathematical contraction tree used by the retained semi-fused production
// backend.  GPU residency and scheduling are owned by that backend.
enum class FullTensorGpuNativeContractionTree : std::uint8_t {
    Automatic,
    LogicalQ,
    CanonicalReflectionOrbits
};

[[nodiscard]] constexpr const char* full_tensor_gpu_native_tree_name(
    FullTensorGpuNativeContractionTree tree) noexcept
{
    switch (tree) {
    case FullTensorGpuNativeContractionTree::Automatic:
        return "auto";
    case FullTensorGpuNativeContractionTree::LogicalQ:
        return "logical-q";
    case FullTensorGpuNativeContractionTree::CanonicalReflectionOrbits:
        return "canonical-reflection-orbits";
    }
    return "unknown";
}

} // namespace b4gpu
