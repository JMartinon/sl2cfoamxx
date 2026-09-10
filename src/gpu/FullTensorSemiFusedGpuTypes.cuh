// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace b4gpu::semifused_detail {

constexpr unsigned kWarpSize = 32;
constexpr unsigned kWarpsPerBlock = 8;
constexpr unsigned kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr unsigned kBoundaryTile = 8;
constexpr unsigned kBoundaryTileValues = kBoundaryTile * kBoundaryTile;
constexpr unsigned kMmaM = 8;
constexpr unsigned kMmaN = 8;
constexpr unsigned kMmaK = 4;
constexpr unsigned kVirtualSourceCapacity = 64;
constexpr unsigned kVirtualOutputTile = 8;

// Plain U/V producers need only one 8(virtual intertwiner) x 64(boundary intertwiner) transpose tile.
constexpr std::uint64_t kPlainOutputValues =
    static_cast<std::uint64_t>(kWarpsPerBlock) * kBoundaryTileValues;
constexpr std::uint64_t kPlainProducerSharedBytes =
    kPlainOutputValues * sizeof(double);

// C-transform producers retain a complete source-virtual intertwiner x 64 boundary intertwiner panel,
// one padded 8 x source-virtual intertwiner C tile, and one 8 x 64 output tile.
constexpr std::uint64_t kSourceValues =
    static_cast<std::uint64_t>(kVirtualSourceCapacity) * kBoundaryTileValues;
constexpr std::uint64_t kCStageValues =
    static_cast<std::uint64_t>(kVirtualOutputTile) * kVirtualSourceCapacity;
constexpr std::uint64_t kTransformOutputValues =
    static_cast<std::uint64_t>(kVirtualOutputTile) * kBoundaryTileValues;
constexpr std::uint64_t kTransformProducerSharedValues =
    kSourceValues + kCStageValues + kTransformOutputValues;
constexpr std::uint64_t kTransformProducerSharedBytes =
    kTransformProducerSharedValues * sizeof(double);

__host__ __device__ constexpr std::uint32_t min_u32(
    std::uint32_t a, std::uint32_t b) noexcept
{
    return a < b ? a : b;
}

__host__ __device__ constexpr std::uint64_t round_up_u64(
    std::uint64_t value, std::uint64_t alignment) noexcept
{
    return (value + alignment - 1u) / alignment * alignment;
}

struct SemiFusedDims {
    std::uint64_t I1 = 0, I2 = 0, I3 = 0, I4 = 0, I5 = 0;
    std::uint64_t X = 0, K2 = 0, K3 = 0, K4 = 0, K5 = 0;
    std::uint64_t U = 0, Q = 0, P = 0, R = 0, S = 0, T = 0;
    std::uint64_t Upad = 0, Tpad = 0;
    std::uint64_t M = 0, N = 0;
    // Logical value counts exclude tile/K padding; T1/T3 are the actual
    // contraction-native storage values per i1.
    std::uint64_t T1Logical = 0, T3Logical = 0;
    std::uint64_t T1 = 0, T3 = 0;
    // Explicit virtual intertwiner-fast pointwise operands consumed by the binary cuTENSOR
    // contractions. They are one-x buffers and do not scale with i1.
    std::uint64_t T1Operand = 0, T3Operand = 0;
    std::uint64_t B3Mma = 0, B4Mma = 0;
    std::uint64_t Cq = 0, Cx = 0;
    std::uint64_t tile2 = 0, tile3 = 0, tile4 = 0, tile5 = 0;
};

struct SemiFusedQDescriptor {
    std::uint64_t row_prefix = 0;
    std::uint32_t il24 = 0;
    std::uint32_t il34 = 0;
    std::uint32_t il35 = 0;
    std::uint32_t k3_begin = 0;
    std::uint32_t k3_count = 0;
    std::uint32_t k4_begin = 0;
    std::uint32_t k4_count = 0;
    // 0: L=C U and R=V.  1: L=U and R=C^T V.
    std::uint8_t association = 0;
    // 0 unpaired, 1 self-reflected, 2 paired.
    std::uint8_t orbit_kind = 0;
};

static_assert(std::is_trivially_copyable_v<SemiFusedDims> &&
              std::is_standard_layout_v<SemiFusedDims>);
static_assert(std::is_trivially_copyable_v<SemiFusedQDescriptor> &&
              std::is_standard_layout_v<SemiFusedQDescriptor>);

__host__ __device__ inline std::uint64_t idx4(
    std::uint64_t i0,
    std::uint64_t i1,
    std::uint64_t i2,
    std::uint64_t i3,
    std::uint64_t D1,
    std::uint64_t D2,
    std::uint64_t D3) noexcept
{
    return ((i0 * D1 + i1) * D2 + i2) * D3 + i3;
}

// All route-native producer packs retain the contracted coordinate as stride
// one. The lane/tile axes group eight neighbouring boundary intertwiner values for either a
// complete WMMA tile or the architecture-neutral SIMT fallback.
__host__ __device__ inline std::uint64_t booster3_mma_tile_offset(
    const SemiFusedDims& d,
    std::uint64_t i3_tile,
    std::uint64_t ik3,
    std::uint64_t il34,
    std::uint64_t il35) noexcept
{
    return d.Upad * kBoundaryTile *
        (i3_tile + d.tile3 * (ik3 + d.K3 * (il34 + d.R * il35)));
}

__host__ __device__ inline std::uint64_t booster4_mma_tile_offset(
    const SemiFusedDims& d,
    std::uint64_t i4_tile,
    std::uint64_t ik4,
    std::uint64_t il24,
    std::uint64_t il34) noexcept
{
    return d.Tpad * kBoundaryTile *
        (i4_tile + d.tile4 * (ik4 + d.K4 * (il24 + d.Q * il34)));
}

__host__ __device__ inline std::uint64_t t1_mma_tile_offset(
    const SemiFusedDims& d,
    std::uint64_t i5_tile,
    std::uint64_t ik4,
    std::uint64_t il25,
    std::uint64_t il35) noexcept
{
    return d.Tpad * kBoundaryTile *
        (i5_tile + d.tile5 * (ik4 + d.K4 * (il25 + d.P * il35)));
}

__host__ __device__ inline std::uint64_t t3_mma_tile_offset(
    const SemiFusedDims& d,
    std::uint64_t i2_tile,
    std::uint64_t ik3,
    std::uint64_t il25,
    std::uint64_t il24) noexcept
{
    return d.Upad * kBoundaryTile *
        (i2_tile + d.tile2 * (ik3 + d.K3 * (il25 + d.P * il24)));
}

} // namespace b4gpu::semifused_detail
