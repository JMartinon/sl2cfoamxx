// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "sl2cfoamxx/configuration.hpp"

#include <cstdint>

namespace sl2cfoamxx::detail {

#if defined(__APPLE__) && defined(__aarch64__)
inline constexpr std::uint32_t default_booster_rho_tile = 16;
#else
inline constexpr std::uint32_t default_booster_rho_tile = 4;
#endif

inline constexpr std::uint32_t default_coherent_gpu_pipeline_slots = 3;
inline constexpr std::uint32_t
    default_full_tensor_gpu_producer_waves_per_sm = 8;
inline constexpr std::uint32_t default_full_tensor_cpu_p_batch = 512;

[[nodiscard]] inline constexpr bool is_supported_booster_rho_tile(
    std::uint32_t value) noexcept
{
    return value == 0 || value == 4 || value == 8 || value == 16;
}

[[nodiscard]] inline constexpr std::uint32_t effective_booster_rho_tile(
    std::uint32_t requested) noexcept
{
    return requested == 0 ? default_booster_rho_tile : requested;
}

[[nodiscard]] inline constexpr bool
is_supported_coherent_gpu_pipeline_slots(std::uint32_t value) noexcept
{
    return value <= 3;
}

[[nodiscard]] inline constexpr std::uint32_t
effective_coherent_gpu_pipeline_slots(std::uint32_t requested) noexcept
{
    return requested == 0
        ? default_coherent_gpu_pipeline_slots
        : requested;
}

[[nodiscard]] inline constexpr bool
is_supported_full_tensor_gpu_producer_waves_per_sm(
    std::uint32_t value) noexcept
{
    return value <= 64;
}

[[nodiscard]] inline constexpr std::uint32_t
effective_full_tensor_gpu_producer_waves_per_sm(
    std::uint32_t requested) noexcept
{
    return requested == 0
        ? default_full_tensor_gpu_producer_waves_per_sm
        : requested;
}

[[nodiscard]] inline constexpr bool
is_supported_full_tensor_cpu_p_batch(std::uint32_t value) noexcept
{
    return value == 0 || value == 128 || value == 256 || value == 512 ||
           value == 1024 || value == 2048;
}

[[nodiscard]] inline constexpr std::uint32_t
effective_full_tensor_cpu_p_batch(std::uint32_t requested) noexcept
{
    return requested == 0 ? default_full_tensor_cpu_p_batch : requested;
}

[[nodiscard]] inline constexpr bool is_supported_full_tensor_cpu_final_tile(
    FullTensorCpuFinalTile value) noexcept
{
    switch (value) {
    case FullTensorCpuFinalTile::automatic:
    case FullTensorCpuFinalTile::tile_4x4:
    case FullTensorCpuFinalTile::tile_4x8:
    case FullTensorCpuFinalTile::tile_8x4:
        return true;
    }
    return false;
}

[[nodiscard]] inline constexpr FullTensorCpuFinalTile
effective_full_tensor_cpu_final_tile(
    FullTensorCpuFinalTile requested) noexcept
{
    return requested == FullTensorCpuFinalTile::automatic
        ? FullTensorCpuFinalTile::tile_4x4
        : requested;
}

[[nodiscard]] inline constexpr const char* full_tensor_cpu_final_tile_name(
    FullTensorCpuFinalTile value) noexcept
{
    switch (value) {
    case FullTensorCpuFinalTile::automatic: return "automatic";
    case FullTensorCpuFinalTile::tile_4x4: return "4x4";
    case FullTensorCpuFinalTile::tile_4x8: return "4x8";
    case FullTensorCpuFinalTile::tile_8x4: return "8x4";
    }
    return "unknown";
}

} // namespace sl2cfoamxx::detail
