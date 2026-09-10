// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "amplitude/AmplitudeInputSupport.hpp"
#include "gpu/FullTensorSemiFusedGpuTypes.cuh"

#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <type_traits>

namespace b4gpu::semifused_detail {

inline unsigned launch_blocks(
    std::uint64_t count, unsigned threads = 256) noexcept
{
    if (count == 0) return 1;
    const std::uint64_t blocks = (count + threads - 1u) / threads;
    return static_cast<unsigned>(blocks > 65535u ? 65535u : blocks);
}

// Expand one packed booster leg into the dense source layout consumed by the
// production MMA packing kernels.
__global__ void expand_packed_ranges_kernel(
    const b4support::Range1D* __restrict__ ranges,
    const double* __restrict__ packed,
    double* __restrict__ dense,
    std::uint64_t outer_count,
    std::uint64_t inner_count)
{
    for (std::uint64_t outer = blockIdx.x; outer < outer_count;
         outer += static_cast<std::uint64_t>(gridDim.x)) {
        const b4support::Range1D range = ranges[outer];
        for (std::uint64_t inner = threadIdx.x; inner < inner_count;
             inner += blockDim.x) {
            dense[outer * inner_count + inner] =
                inner >= range.begin && inner < range.end
                ? packed[range.offset + inner - range.begin]
                : 0.0;
        }
    }
}

// Canonical source B3 is [u,k3,r,s,i3] with u stride one.  The MMA pack is
// [u,i3_lane,i3_tile,k3,r,s], still with u stride one.
__global__ void pack_booster3_mma_kernel(
    const double* __restrict__ dense,
    double* __restrict__ packed,
    SemiFusedDims d)
{
    for (std::uint64_t linear =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < d.B3Mma;
         linear += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        std::uint64_t t = linear;
        const std::uint64_t u = t % d.Upad; t /= d.Upad;
        const std::uint64_t lane = t % kBoundaryTile; t /= kBoundaryTile;
        const std::uint64_t tile = t % d.tile3; t /= d.tile3;
        const std::uint64_t ik3 = t % d.K3; t /= d.K3;
        const std::uint64_t il34 = t % d.R; t /= d.R;
        const std::uint64_t il35 = t;
        const std::uint64_t ii3 = tile * kBoundaryTile + lane;
        double value = 0.0;
        if (u < d.U && ii3 < d.I3) {
            const std::uint64_t outer = idx4(
                ii3, ik3, il34, il35, d.K3, d.R, d.S);
            value = dense[outer * d.U + u];
        }
        packed[linear] = value;
    }
}

// Canonical source B4 is [t,k4,q,r,i4] with t stride one.  The MMA pack is
// [t,i4_lane,i4_tile,k4,q,r], still with t stride one.
__global__ void pack_booster4_mma_kernel(
    const double* __restrict__ dense,
    double* __restrict__ packed,
    SemiFusedDims d)
{
    for (std::uint64_t linear =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < d.B4Mma;
         linear += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        std::uint64_t t = linear;
        const std::uint64_t il45 = t % d.Tpad; t /= d.Tpad;
        const std::uint64_t lane = t % kBoundaryTile; t /= kBoundaryTile;
        const std::uint64_t tile = t % d.tile4; t /= d.tile4;
        const std::uint64_t ik4 = t % d.K4; t /= d.K4;
        const std::uint64_t il24 = t % d.Q; t /= d.Q;
        const std::uint64_t il34 = t;
        const std::uint64_t ii4 = tile * kBoundaryTile + lane;
        double value = 0.0;
        if (il45 < d.T && ii4 < d.I4) {
            const std::uint64_t outer = idx4(
                ii4, ik4, il24, il34, d.K4, d.Q, d.R);
            value = dense[outer * d.T + il45];
        }
        packed[linear] = value;
    }
}

// cuTENSOR T1/T3 packs.  The recurrence kernels intentionally emit their
// own consumer-native virtual_index recurrence order.  These tiny transforms make the
// next contraction's k5/k2 reduction stride one. The resulting T1/T3 retain
// l45/l23 at stride one for the producer.
__global__ void pack_boundary_virtual_fast_kernel(
    const double* __restrict__ source,
    double* __restrict__ destination,
    std::uint64_t a_count,
    std::uint64_t virtual_count,
    std::uint64_t p_count)
{
    const std::uint64_t total = a_count * virtual_count * p_count;
    for (std::uint64_t linear =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        std::uint64_t t = linear;
        const std::uint64_t virtual_index = t % virtual_count; t /= virtual_count;
        const std::uint64_t a = t % a_count; t /= a_count;
        const std::uint64_t p = t;
        destination[linear] = source[a + a_count * (virtual_index + virtual_count * p)];
    }
}

// Source D is [k3,k2,l23,l24,x-slot] with k3 stride one.  cuTENSOR sees
// [k2,k3,l23,l24,x-slot] so the shared k2 reduction is contiguous.
__global__ void pack_t1_cutensor_output_kernel(
    const double* __restrict__ logical,
    double* __restrict__ padded,
    SemiFusedDims d,
    std::uint64_t i1_count)
{
    const std::uint64_t total = i1_count * d.T1;
    for (std::uint64_t linear =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint64_t ii1 = linear / d.T1;
        std::uint64_t x = linear - ii1 * d.T1;
        const std::uint64_t il45 = x % d.Tpad; x /= d.Tpad;
        const std::uint64_t lane = x % kBoundaryTile; x /= kBoundaryTile;
        const std::uint64_t tile = x % d.tile5; x /= d.tile5;
        const std::uint64_t ik4 = x % d.K4; x /= d.K4;
        const std::uint64_t il25 = x % d.P; x /= d.P;
        const std::uint64_t il35 = x;
        const std::uint64_t ii5 = tile * kBoundaryTile + lane;
        double value = 0.0;
        if (il45 < d.T && ii5 < d.I5) {
            const std::uint64_t source = il45 + d.T *
                (ii5 + d.I5 * (ik4 + d.K4 *
                (il25 + d.P * (il35 + d.S * ii1))));
            value = logical[source];
        }
        padded[linear] = value;
    }
}

// Convert packed logical cuTENSOR output [l23,i2,k3,l25,l24,i1]
// to the downstream padded producer layout.
__global__ void pack_t3_cutensor_output_kernel(
    const double* __restrict__ logical,
    double* __restrict__ padded,
    SemiFusedDims d,
    std::uint64_t i1_count)
{
    const std::uint64_t total = i1_count * d.T3;
    for (std::uint64_t linear =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint64_t ii1 = linear / d.T3;
        std::uint64_t x = linear - ii1 * d.T3;
        const std::uint64_t il23 = x % d.Upad; x /= d.Upad;
        const std::uint64_t lane = x % kBoundaryTile; x /= kBoundaryTile;
        const std::uint64_t tile = x % d.tile2; x /= d.tile2;
        const std::uint64_t ik3 = x % d.K3; x /= d.K3;
        const std::uint64_t il25 = x % d.P; x /= d.P;
        const std::uint64_t il24 = x;
        const std::uint64_t ii2 = tile * kBoundaryTile + lane;
        double value = 0.0;
        if (il23 < d.U && ii2 < d.I2) {
            const std::uint64_t source = il23 + d.U *
                (ii2 + d.I2 * (ik3 + d.K3 *
                (il25 + d.P * (il24 + d.Q * ii1))));
            value = logical[source];
        }
        padded[linear] = value;
    }
}

// The packed producer operands keep the contracted coordinate contiguous.
// Ampere and newer consume complete 8x8 tiles through FP64 WMMA below.  Older
// architectures use these scalar dot products over the same storage, so the
// host-side preparation and the downstream slab layout remain identical.
__device__ inline double compute_u_value_simt(
    const double* __restrict__ B3,
    const double* __restrict__ T3,
    const SemiFusedDims& d,
    std::uint32_t ik3,
    std::uint32_t il24,
    std::uint32_t il34,
    std::uint32_t il35,
    std::uint32_t il25,
    std::uint32_t i2_tile,
    std::uint32_t i3_tile,
    std::uint32_t li2,
    std::uint32_t li3)
{
    const double* const k3_tile = B3 + booster3_mma_tile_offset(
        d, i3_tile, ik3, il34, il35);
    const double* const t3_tile = T3 + t3_mma_tile_offset(
        d, i2_tile, ik3, il25, il24);
    double value = 0.0;
    for (std::uint64_t u = 0; u < d.U; ++u) {
        value += k3_tile[static_cast<std::uint64_t>(li3) * d.Upad + u] *
                 t3_tile[static_cast<std::uint64_t>(li2) * d.Upad + u];
    }
    return value;
}

__device__ inline double compute_v_value_simt(
    const double* __restrict__ B4,
    const double* __restrict__ T1,
    const SemiFusedDims& d,
    std::uint32_t ik4,
    std::uint32_t il24,
    std::uint32_t il34,
    std::uint32_t il35,
    std::uint32_t il25,
    std::uint32_t i4_tile,
    std::uint32_t i5_tile,
    std::uint32_t li4,
    std::uint32_t li5)
{
    const double* const k4_tile = B4 + booster4_mma_tile_offset(
        d, i4_tile, ik4, il24, il34);
    const double* const t1_tile = T1 + t1_mma_tile_offset(
        d, i5_tile, ik4, il25, il35);
    double value = 0.0;
    for (std::uint64_t t = 0; t < d.T; ++t) {
        value += t1_tile[static_cast<std::uint64_t>(li5) * d.Tpad + t] *
                 k4_tile[static_cast<std::uint64_t>(li4) * d.Tpad + t];
    }
    return value;
}

__device__ inline void generate_u_source_simt(
    const double* __restrict__ B3,
    const double* __restrict__ T3,
    double* __restrict__ source,
    const SemiFusedDims& d,
    std::uint32_t virtual_begin,
    std::uint32_t virtual_count,
    std::uint32_t il24,
    std::uint32_t il34,
    std::uint32_t il35,
    std::uint32_t il25,
    std::uint32_t i2_tile,
    std::uint32_t i3_tile)
{
    const std::uint64_t values =
        static_cast<std::uint64_t>(virtual_count) * kBoundaryTileValues;
    for (std::uint64_t linear = threadIdx.x; linear < values;
         linear += blockDim.x) {
        const std::uint32_t virtual_index = static_cast<std::uint32_t>(
            linear / kBoundaryTileValues);
        const std::uint32_t boundary_index = static_cast<std::uint32_t>(
            linear % kBoundaryTileValues);
        const std::uint32_t li3 = boundary_index / kBoundaryTile;
        const std::uint32_t li2 = boundary_index % kBoundaryTile;
        source[linear] = compute_u_value_simt(
            B3, T3, d, virtual_begin + virtual_index, il24, il34, il35, il25,
            i2_tile, i3_tile, li2, li3);
    }
    __syncthreads();
}

__device__ inline void generate_v_source_simt(
    const double* __restrict__ B4,
    const double* __restrict__ T1,
    double* __restrict__ source,
    const SemiFusedDims& d,
    std::uint32_t virtual_begin,
    std::uint32_t virtual_count,
    std::uint32_t il24,
    std::uint32_t il34,
    std::uint32_t il35,
    std::uint32_t il25,
    std::uint32_t i4_tile,
    std::uint32_t i5_tile)
{
    const std::uint64_t values =
        static_cast<std::uint64_t>(virtual_count) * kBoundaryTileValues;
    for (std::uint64_t linear = threadIdx.x; linear < values;
         linear += blockDim.x) {
        const std::uint32_t virtual_index = static_cast<std::uint32_t>(
            linear / kBoundaryTileValues);
        const std::uint32_t boundary_index = static_cast<std::uint32_t>(
            linear % kBoundaryTileValues);
        const std::uint32_t li5 = boundary_index / kBoundaryTile;
        const std::uint32_t li4 = boundary_index % kBoundaryTile;
        source[linear] = compute_v_value_simt(
            B4, T1, d, virtual_begin + virtual_index, il24, il34, il35, il25,
            i4_tile, i5_tile, li4, li5);
    }
    __syncthreads();
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

using namespace nvcuda;
using MmaArow = wmma::fragment<
    wmma::matrix_a, kMmaM, kMmaN, kMmaK, double, wmma::row_major>;
using MmaBrow = wmma::fragment<
    wmma::matrix_b, kMmaM, kMmaN, kMmaK, double, wmma::row_major>;
using MmaBcol = wmma::fragment<
    wmma::matrix_b, kMmaM, kMmaN, kMmaK, double, wmma::col_major>;
using MmaAcc = wmma::fragment<
    wmma::accumulator, kMmaM, kMmaN, kMmaK, double>;

__device__ inline void scale_accumulator(MmaAcc& fragment, double scale)
{
    if (scale == 1.0) return;
    for (int i = 0; i < fragment.num_elements; ++i) {
        fragment.x[i] *= scale;
    }
}

__device__ inline void compute_u_fragment(
    const double* __restrict__ B3,
    const double* __restrict__ T3,
    const SemiFusedDims& d,
    std::uint32_t ik3,
    std::uint32_t il24,
    std::uint32_t il34,
    std::uint32_t il35,
    std::uint32_t il25,
    std::uint32_t i2_tile,
    std::uint32_t i3_tile,
    MmaAcc& accum)
{
    const double* const a = B3 + booster3_mma_tile_offset(
        d, i3_tile, ik3, il34, il35);
    const double* const b = T3 + t3_mma_tile_offset(
        d, i2_tile, ik3, il25, il24);
    wmma::fill_fragment(accum, 0.0);
    for (std::uint64_t k = 0; k < d.Upad; k += kMmaK) {
        MmaArow af;
        MmaBcol bf;
        wmma::load_matrix_sync(af, a + k, d.Upad);
        wmma::load_matrix_sync(bf, b + k, d.Upad);
        wmma::mma_sync(accum, af, bf, accum);
    }
}

__device__ inline void compute_v_fragment(
    const double* __restrict__ B4,
    const double* __restrict__ T1,
    const SemiFusedDims& d,
    std::uint32_t ik4,
    std::uint32_t il24,
    std::uint32_t il34,
    std::uint32_t il35,
    std::uint32_t il25,
    std::uint32_t i4_tile,
    std::uint32_t i5_tile,
    MmaAcc& accum)
{
    const double* const a = T1 + t1_mma_tile_offset(
        d, i5_tile, ik4, il25, il35);
    const double* const b = B4 + booster4_mma_tile_offset(
        d, i4_tile, ik4, il24, il34);
    wmma::fill_fragment(accum, 0.0);
    for (std::uint64_t k = 0; k < d.Tpad; k += kMmaK) {
        MmaArow af;
        MmaBcol bf;
        wmma::load_matrix_sync(af, a + k, d.Tpad);
        wmma::load_matrix_sync(bf, b + k, d.Tpad);
        wmma::mma_sync(accum, af, bf, accum);
    }
}

__device__ inline void generate_u_source(
    const double* __restrict__ B3,
    const double* __restrict__ T3,
    double* __restrict__ source,
    const SemiFusedDims& d,
    std::uint32_t virtual_begin,
    std::uint32_t virtual_count,
    std::uint32_t il24,
    std::uint32_t il34,
    std::uint32_t il35,
    std::uint32_t il25,
    std::uint32_t i2_tile,
    std::uint32_t i3_tile)
{
    const unsigned warp = threadIdx.x / kWarpSize;
    const std::uint32_t padded = static_cast<std::uint32_t>(
        round_up_u64(virtual_count, kMmaK));
    for (std::uint64_t i = threadIdx.x;
         i < static_cast<std::uint64_t>(padded) * kBoundaryTileValues;
         i += blockDim.x) {
        source[i] = 0.0;
    }
    __syncthreads();
    for (std::uint32_t group = 0; group < virtual_count;
         group += kWarpsPerBlock) {
        const std::uint32_t local = group + warp;
        if (local < virtual_count) {
            MmaAcc accum;
            compute_u_fragment(
                B3, T3, d, virtual_begin + local, il24, il34, il35,
                il25, i2_tile, i3_tile, accum);
            wmma::store_matrix_sync(
                source + static_cast<std::uint64_t>(local) * kBoundaryTileValues,
                accum, kBoundaryTile, wmma::mem_row_major);
        }
    }
    __syncthreads();
}

__device__ inline void generate_v_source(
    const double* __restrict__ B4,
    const double* __restrict__ T1,
    double* __restrict__ source,
    const SemiFusedDims& d,
    std::uint32_t virtual_begin,
    std::uint32_t virtual_count,
    std::uint32_t il24,
    std::uint32_t il34,
    std::uint32_t il35,
    std::uint32_t il25,
    std::uint32_t i4_tile,
    std::uint32_t i5_tile)
{
    const unsigned warp = threadIdx.x / kWarpSize;
    const std::uint32_t padded = static_cast<std::uint32_t>(
        round_up_u64(virtual_count, kMmaK));
    for (std::uint64_t i = threadIdx.x;
         i < static_cast<std::uint64_t>(padded) * kBoundaryTileValues;
         i += blockDim.x) {
        source[i] = 0.0;
    }
    __syncthreads();
    for (std::uint32_t group = 0; group < virtual_count;
         group += kWarpsPerBlock) {
        const std::uint32_t local = group + warp;
        if (local < virtual_count) {
            MmaAcc accum;
            compute_v_fragment(
                B4, T1, d, virtual_begin + local, il24, il34, il35,
                il25, i4_tile, i5_tile, accum);
            wmma::store_matrix_sync(
                source + static_cast<std::uint64_t>(local) * kBoundaryTileValues,
                accum, kBoundaryTile, wmma::mem_row_major);
        }
    }
    __syncthreads();
}

#endif // FP64 WMMA helpers for sm_80+

__device__ inline const double* c_block_pointer(
    const double* __restrict__ c_base,
    const SemiFusedQDescriptor& q,
    const SemiFusedDims& d,
    bool resident_c,
    std::uint32_t streamed_l34_begin,
    std::uint32_t streamed_l34_count) noexcept
{
    const std::uint64_t z = resident_c
        ? static_cast<std::uint64_t>(q.il34) + d.R * q.il35
        : static_cast<std::uint64_t>(q.il34 - streamed_l34_begin) +
              static_cast<std::uint64_t>(streamed_l34_count) * q.il35;
    return resident_c
        ? c_base + static_cast<std::uint64_t>(q.il24) * d.Cq +
              d.K4 * d.K3 * z
        : c_base + d.K4 * d.K3 * z;
}

// Association 1: write plain U into L.  This kernel has no C staging and no
// association branch.  Ampere uses one warp per virtual_index row; the pre-Ampere path
// assigns scalar dot products cooperatively across the CTA.
__global__ void produce_left_u_dmma_kernel(
    const double* __restrict__ B3,
    const double* __restrict__ T3_all,
    const SemiFusedQDescriptor* __restrict__ descriptors,
    std::uint64_t descriptor_count,
    double* __restrict__ L,
    SemiFusedDims d,
    std::uint64_t i1_count,
    std::uint64_t capacity,
    std::uint64_t row_base,
    std::uint64_t prefix_base)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(32) double output_stage[];
    const std::uint64_t m_tiles = d.tile2 * d.tile3;
    const std::uint64_t task_count =
        i1_count * descriptor_count * d.P * m_tiles;
    const unsigned warp = threadIdx.x / kWarpSize;

    for (std::uint64_t task = blockIdx.x; task < task_count;
         task += static_cast<std::uint64_t>(gridDim.x)) {
        std::uint64_t t = task;
        const std::uint64_t m_tile = t % m_tiles; t /= m_tiles;
        const std::uint32_t p = static_cast<std::uint32_t>(t % d.P); t /= d.P;
        const std::uint64_t qi = t % descriptor_count; t /= descriptor_count;
        const std::uint64_t i1 = t;
        const SemiFusedQDescriptor q = descriptors[qi];
        const std::uint32_t tile3 = static_cast<std::uint32_t>(m_tile % d.tile3);
        const std::uint32_t tile2 = static_cast<std::uint32_t>(m_tile / d.tile3);
        const double* const T3 = T3_all + i1 * d.T3;
        const double orbit_scale = q.orbit_kind == 1u ? 0.5 : 1.0;
        const std::uint64_t qrow = row_base + (q.row_prefix - prefix_base);

        for (std::uint32_t i0 = 0; i0 < q.k3_count;
             i0 += kWarpsPerBlock) {
            const std::uint32_t icount = min_u32(
                kWarpsPerBlock, q.k3_count - i0);
            const std::uint32_t local = i0 + warp;
            if (warp < icount) {
                MmaAcc accum;
                compute_u_fragment(
                    B3, T3, d, q.k3_begin + local,
                    q.il24, q.il34, q.il35, p, tile2, tile3, accum);
                scale_accumulator(accum, orbit_scale);
                wmma::store_matrix_sync(
                    output_stage + warp * kBoundaryTileValues,
                    accum, kBoundaryTile, wmma::mem_row_major);
            }
            __syncthreads();
            for (std::uint64_t linear = threadIdx.x;
                 linear < static_cast<std::uint64_t>(icount) * kBoundaryTileValues;
                 linear += blockDim.x) {
                const std::uint64_t boundary_index = linear / icount;
                const std::uint64_t ib = linear % icount;
                const std::uint64_t li3 = boundary_index / kBoundaryTile;
                const std::uint64_t li2 = boundary_index % kBoundaryTile;
                const std::uint64_t ii2 = tile2 * kBoundaryTile + li2;
                const std::uint64_t ii3 = tile3 * kBoundaryTile + li3;
                if (ii2 < d.I2 && ii3 < d.I3) {
                    const std::uint64_t m = ii2 + d.I2 * ii3;
                    const std::uint64_t row = qrow +
                        static_cast<std::uint64_t>(p) * q.k3_count + i0 + ib;
                    L[i1 * capacity * d.M + row + capacity * m] =
                        output_stage[ib * kBoundaryTileValues + boundary_index];
                }
            }
            __syncthreads();
        }
    }
#else
    const std::uint64_t m_tiles = d.tile2 * d.tile3;
    const std::uint64_t task_count =
        i1_count * descriptor_count * d.P * m_tiles;

    for (std::uint64_t task = blockIdx.x; task < task_count;
         task += static_cast<std::uint64_t>(gridDim.x)) {
        std::uint64_t t = task;
        const std::uint64_t m_tile = t % m_tiles; t /= m_tiles;
        const std::uint32_t p = static_cast<std::uint32_t>(t % d.P); t /= d.P;
        const std::uint64_t qi = t % descriptor_count; t /= descriptor_count;
        const std::uint64_t i1 = t;
        const SemiFusedQDescriptor q = descriptors[qi];
        const std::uint32_t tile3 = static_cast<std::uint32_t>(m_tile % d.tile3);
        const std::uint32_t tile2 = static_cast<std::uint32_t>(m_tile / d.tile3);
        const double* const T3 = T3_all + i1 * d.T3;
        const double orbit_scale = q.orbit_kind == 1u ? 0.5 : 1.0;
        const std::uint64_t qrow = row_base + (q.row_prefix - prefix_base);
        const std::uint64_t values =
            static_cast<std::uint64_t>(q.k3_count) * kBoundaryTileValues;

        for (std::uint64_t linear = threadIdx.x; linear < values;
             linear += blockDim.x) {
            const std::uint64_t boundary_index = linear / q.k3_count;
            const std::uint32_t ib = static_cast<std::uint32_t>(
                linear % q.k3_count);
            const std::uint32_t li3 = static_cast<std::uint32_t>(
                boundary_index / kBoundaryTile);
            const std::uint32_t li2 = static_cast<std::uint32_t>(
                boundary_index % kBoundaryTile);
            const std::uint64_t ii2 = tile2 * kBoundaryTile + li2;
            const std::uint64_t ii3 = tile3 * kBoundaryTile + li3;
            if (ii2 < d.I2 && ii3 < d.I3) {
                const std::uint64_t m = ii2 + d.I2 * ii3;
                const std::uint64_t row = qrow +
                    static_cast<std::uint64_t>(p) * q.k3_count + ib;
                L[i1 * capacity * d.M + row + capacity * m] =
                    orbit_scale * compute_u_value_simt(
                        B3, T3, d, q.k3_begin + ib,
                        q.il24, q.il34, q.il35, p,
                        tile2, tile3, li2, li3);
            }
        }
    }
#endif
}

// Association-0 fused producer.  It forms B3*T3 fragments and contracts them
// with C directly into the descriptor's materialized L slab.  The compact
// source prefixes are independent of the final row prefixes because nk3 and
// nk4 need not match.
__global__ void produce_left_cu_dmma_kernel(
    const double* __restrict__ B3,
    const double* __restrict__ T3_all,
    const double* __restrict__ c_base,
    const SemiFusedQDescriptor* __restrict__ descriptors,
    std::uint64_t descriptor_count,
    double* __restrict__ L,
    SemiFusedDims d,
    std::uint64_t i1_count,
    std::uint64_t capacity,
    std::uint64_t row_base,
    std::uint64_t prefix_base,
    bool resident_c,
    std::uint32_t streamed_l34_begin,
    std::uint32_t streamed_l34_count)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(32) double shared[];
    double* const source = shared;
    double* const c_stage = source + kSourceValues;
    double* const output_stage = c_stage + kCStageValues;
    const std::uint64_t m_tiles = d.tile2 * d.tile3;
    const std::uint64_t task_count =
        i1_count * descriptor_count * d.P * m_tiles;
    const unsigned warp = threadIdx.x / kWarpSize;

    for (std::uint64_t task = blockIdx.x; task < task_count;
         task += static_cast<std::uint64_t>(gridDim.x)) {
        std::uint64_t t = task;
        const std::uint64_t m_tile = t % m_tiles; t /= m_tiles;
        const std::uint32_t p = static_cast<std::uint32_t>(t % d.P); t /= d.P;
        const std::uint64_t qi = t % descriptor_count; t /= descriptor_count;
        const std::uint64_t i1 = t;
        const SemiFusedQDescriptor q = descriptors[qi];
        const std::uint32_t tile3 = static_cast<std::uint32_t>(m_tile % d.tile3);
        const std::uint32_t tile2 = static_cast<std::uint32_t>(m_tile / d.tile3);
        const double* const T3 = T3_all + i1 * d.T3;
        const double* const C = c_block_pointer(
            c_base, q, d, resident_c,
            streamed_l34_begin, streamed_l34_count);
        const double orbit_scale = q.orbit_kind == 1u ? 0.5 : 1.0;
        const std::uint64_t qrow = row_base + (q.row_prefix - prefix_base);
        const bool source_fits = q.k3_count <= kVirtualSourceCapacity;
        if (source_fits) {
            generate_u_source(
                B3, T3, source, d, q.k3_begin, q.k3_count,
                q.il24, q.il34, q.il35, p, tile2, tile3);
        }

        for (std::uint32_t j0 = 0; j0 < q.k4_count;
             j0 += kVirtualOutputTile) {
            const std::uint32_t jcount = min_u32(
                kVirtualOutputTile, q.k4_count - j0);
            MmaAcc accum;
            wmma::fill_fragment(accum, 0.0);

            const std::uint32_t source_chunks = source_fits
                ? 1u
                : static_cast<std::uint32_t>(
                    (q.k3_count + kVirtualSourceCapacity - 1u) /
                    kVirtualSourceCapacity);
            for (std::uint32_t chunk = 0; chunk < source_chunks; ++chunk) {
                const std::uint32_t i0 = source_fits
                    ? 0u : chunk * kVirtualSourceCapacity;
                const std::uint32_t icount = source_fits
                    ? q.k3_count
                    : min_u32(kVirtualSourceCapacity, q.k3_count - i0);
                if (!source_fits) {
                    generate_u_source(
                        B3, T3, source, d, q.k3_begin + i0, icount,
                        q.il24, q.il34, q.il35, p, tile2, tile3);
                }
                for (std::uint64_t linear = threadIdx.x;
                     linear < kCStageValues; linear += blockDim.x) {
                    const std::uint32_t jr = static_cast<std::uint32_t>(
                        linear / kVirtualSourceCapacity);
                    const std::uint32_t ik = static_cast<std::uint32_t>(
                        linear % kVirtualSourceCapacity);
                    c_stage[linear] = jr < jcount && ik < icount
                        ? C[(q.k4_begin + j0 + jr) + d.K4 *
                            (q.k3_begin + i0 + ik)]
                        : 0.0;
                }
                __syncthreads();
                const std::uint32_t padded = static_cast<std::uint32_t>(
                    round_up_u64(icount, kMmaK));
                for (std::uint32_t k = 0; k < padded; k += kMmaK) {
                    MmaArow af;
                    MmaBrow bf;
                    wmma::load_matrix_sync(
                        af, c_stage + k, kVirtualSourceCapacity);
                    wmma::load_matrix_sync(
                        bf, source + static_cast<std::uint64_t>(k) *
                                         kBoundaryTileValues + warp * kBoundaryTile,
                        kBoundaryTileValues);
                    wmma::mma_sync(accum, af, bf, accum);
                }
                __syncthreads();
            }

            scale_accumulator(accum, orbit_scale);
            wmma::store_matrix_sync(
                output_stage + warp * kBoundaryTile,
                accum, kBoundaryTileValues, wmma::mem_row_major);
            __syncthreads();
            for (std::uint64_t linear = threadIdx.x;
                 linear < static_cast<std::uint64_t>(jcount) * kBoundaryTileValues;
                 linear += blockDim.x) {
                const std::uint64_t boundary_index = linear / jcount;
                const std::uint64_t jb = linear % jcount;
                const std::uint64_t li3 = boundary_index / kBoundaryTile;
                const std::uint64_t li2 = boundary_index % kBoundaryTile;
                const std::uint64_t ii2 = tile2 * kBoundaryTile + li2;
                const std::uint64_t ii3 = tile3 * kBoundaryTile + li3;
                if (ii2 < d.I2 && ii3 < d.I3) {
                    const std::uint64_t m = ii2 + d.I2 * ii3;
                    const std::uint64_t row = qrow +
                        static_cast<std::uint64_t>(p) * q.k4_count + j0 + jb;
                    L[i1 * capacity * d.M + row + capacity * m] =
                        output_stage[jb * kBoundaryTileValues + boundary_index];
                }
            }
            __syncthreads();
        }
    }
#else
    extern __shared__ __align__(32) double shared[];
    double* const source = shared;
    double* const output_stage =
        source + kSourceValues + kCStageValues;
    const std::uint64_t m_tiles = d.tile2 * d.tile3;
    const std::uint64_t task_count =
        i1_count * descriptor_count * d.P * m_tiles;

    for (std::uint64_t task = blockIdx.x; task < task_count;
         task += static_cast<std::uint64_t>(gridDim.x)) {
        std::uint64_t t = task;
        const std::uint64_t m_tile = t % m_tiles; t /= m_tiles;
        const std::uint32_t p = static_cast<std::uint32_t>(t % d.P); t /= d.P;
        const std::uint64_t qi = t % descriptor_count; t /= descriptor_count;
        const std::uint64_t i1 = t;
        const SemiFusedQDescriptor q = descriptors[qi];
        const std::uint32_t tile3 = static_cast<std::uint32_t>(m_tile % d.tile3);
        const std::uint32_t tile2 = static_cast<std::uint32_t>(m_tile / d.tile3);
        const double* const T3 = T3_all + i1 * d.T3;
        const double* const C = c_block_pointer(
            c_base, q, d, resident_c,
            streamed_l34_begin, streamed_l34_count);
        const double orbit_scale = q.orbit_kind == 1u ? 0.5 : 1.0;
        const std::uint64_t qrow = row_base + (q.row_prefix - prefix_base);
        const bool source_fits = q.k3_count <= kVirtualSourceCapacity;
        if (source_fits) {
            generate_u_source_simt(
                B3, T3, source, d, q.k3_begin, q.k3_count,
                q.il24, q.il34, q.il35, p, tile2, tile3);
        }

        for (std::uint32_t j0 = 0; j0 < q.k4_count;
             j0 += kVirtualOutputTile) {
            const std::uint32_t jcount = min_u32(
                kVirtualOutputTile, q.k4_count - j0);
            const std::uint64_t output_values =
                static_cast<std::uint64_t>(jcount) * kBoundaryTileValues;
            for (std::uint64_t linear = threadIdx.x;
                 linear < output_values; linear += blockDim.x) {
                output_stage[linear] = 0.0;
            }
            __syncthreads();

            const std::uint32_t source_chunks = source_fits
                ? 1u
                : static_cast<std::uint32_t>(
                    (q.k3_count + kVirtualSourceCapacity - 1u) /
                    kVirtualSourceCapacity);
            for (std::uint32_t chunk = 0; chunk < source_chunks; ++chunk) {
                const std::uint32_t i0 = source_fits
                    ? 0u : chunk * kVirtualSourceCapacity;
                const std::uint32_t icount = source_fits
                    ? q.k3_count
                    : min_u32(kVirtualSourceCapacity, q.k3_count - i0);
                if (!source_fits) {
                    generate_u_source_simt(
                        B3, T3, source, d, q.k3_begin + i0, icount,
                        q.il24, q.il34, q.il35, p, tile2, tile3);
                }
                for (std::uint64_t linear = threadIdx.x;
                     linear < output_values; linear += blockDim.x) {
                    const std::uint32_t jb = static_cast<std::uint32_t>(
                        linear / kBoundaryTileValues);
                    const std::uint32_t boundary_index = static_cast<std::uint32_t>(
                        linear % kBoundaryTileValues);
                    double partial = 0.0;
                    for (std::uint32_t ik = 0; ik < icount; ++ik) {
                        partial += C[(q.k4_begin + j0 + jb) + d.K4 *
                                     (q.k3_begin + i0 + ik)] *
                                   source[static_cast<std::uint64_t>(ik) *
                                          kBoundaryTileValues + boundary_index];
                    }
                    output_stage[linear] += partial;
                }
                __syncthreads();
            }

            for (std::uint64_t linear = threadIdx.x;
                 linear < output_values; linear += blockDim.x) {
                const std::uint32_t jb = static_cast<std::uint32_t>(
                    linear / kBoundaryTileValues);
                const std::uint32_t boundary_index = static_cast<std::uint32_t>(
                    linear % kBoundaryTileValues);
                const std::uint32_t li3 = boundary_index / kBoundaryTile;
                const std::uint32_t li2 = boundary_index % kBoundaryTile;
                const std::uint64_t ii2 = tile2 * kBoundaryTile + li2;
                const std::uint64_t ii3 = tile3 * kBoundaryTile + li3;
                if (ii2 < d.I2 && ii3 < d.I3) {
                    const std::uint64_t m = ii2 + d.I2 * ii3;
                    const std::uint64_t row = qrow +
                        static_cast<std::uint64_t>(p) * q.k4_count + j0 + jb;
                    L[i1 * capacity * d.M + row + capacity * m] =
                        orbit_scale * output_stage[linear];
                }
            }
            __syncthreads();
        }
    }
#endif
}

// ---------------------------------------------------------------------------
__global__ void produce_right_v_dmma_kernel(
    const double* __restrict__ B4,
    const double* __restrict__ T1_all,
    const SemiFusedQDescriptor* __restrict__ descriptors,
    std::uint64_t descriptor_count,
    double* __restrict__ R,
    SemiFusedDims d,
    std::uint64_t i1_count,
    std::uint64_t capacity,
    std::uint64_t row_base,
    std::uint64_t prefix_base)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(32) double output_stage[];
    const std::uint64_t n_tiles = d.tile4 * d.tile5;
    const std::uint64_t task_count =
        i1_count * descriptor_count * d.P * n_tiles;
    const unsigned warp = threadIdx.x / kWarpSize;

    for (std::uint64_t task = blockIdx.x; task < task_count;
         task += static_cast<std::uint64_t>(gridDim.x)) {
        std::uint64_t t = task;
        const std::uint64_t n_tile = t % n_tiles; t /= n_tiles;
        const std::uint32_t p = static_cast<std::uint32_t>(t % d.P); t /= d.P;
        const std::uint64_t qi = t % descriptor_count; t /= descriptor_count;
        const std::uint64_t i1 = t;
        const SemiFusedQDescriptor q = descriptors[qi];
        const std::uint32_t tile5 = static_cast<std::uint32_t>(n_tile % d.tile5);
        const std::uint32_t tile4 = static_cast<std::uint32_t>(n_tile / d.tile5);
        const double* const T1 = T1_all + i1 * d.T1;
        const std::uint64_t qrow = row_base + (q.row_prefix - prefix_base);

        for (std::uint32_t j0 = 0; j0 < q.k4_count;
             j0 += kWarpsPerBlock) {
            const std::uint32_t jcount = min_u32(
                kWarpsPerBlock, q.k4_count - j0);
            const std::uint32_t local = j0 + warp;
            if (warp < jcount) {
                MmaAcc accum;
                compute_v_fragment(
                    B4, T1, d, q.k4_begin + local,
                    q.il24, q.il34, q.il35, p, tile4, tile5, accum);
                wmma::store_matrix_sync(
                    output_stage + warp * kBoundaryTileValues,
                    accum, kBoundaryTile, wmma::mem_row_major);
            }
            __syncthreads();
            for (std::uint64_t linear = threadIdx.x;
                 linear < static_cast<std::uint64_t>(jcount) * kBoundaryTileValues;
                 linear += blockDim.x) {
                const std::uint64_t boundary_index = linear / jcount;
                const std::uint64_t jb = linear % jcount;
                const std::uint64_t li5 = boundary_index / kBoundaryTile;
                const std::uint64_t li4 = boundary_index % kBoundaryTile;
                const std::uint64_t ii4 = tile4 * kBoundaryTile + li4;
                const std::uint64_t ii5 = tile5 * kBoundaryTile + li5;
                if (ii4 < d.I4 && ii5 < d.I5) {
                    const std::uint64_t n = ii4 + d.I4 * ii5;
                    const std::uint64_t row = qrow +
                        static_cast<std::uint64_t>(p) * q.k4_count + j0 + jb;
                    R[i1 * capacity * d.N + row + capacity * n] =
                        output_stage[jb * kBoundaryTileValues + boundary_index];
                }
            }
            __syncthreads();
        }
    }
#else
    const std::uint64_t n_tiles = d.tile4 * d.tile5;
    const std::uint64_t task_count =
        i1_count * descriptor_count * d.P * n_tiles;

    for (std::uint64_t task = blockIdx.x; task < task_count;
         task += static_cast<std::uint64_t>(gridDim.x)) {
        std::uint64_t t = task;
        const std::uint64_t n_tile = t % n_tiles; t /= n_tiles;
        const std::uint32_t p = static_cast<std::uint32_t>(t % d.P); t /= d.P;
        const std::uint64_t qi = t % descriptor_count; t /= descriptor_count;
        const std::uint64_t i1 = t;
        const SemiFusedQDescriptor q = descriptors[qi];
        const std::uint32_t tile5 = static_cast<std::uint32_t>(n_tile % d.tile5);
        const std::uint32_t tile4 = static_cast<std::uint32_t>(n_tile / d.tile5);
        const double* const T1 = T1_all + i1 * d.T1;
        const std::uint64_t qrow = row_base + (q.row_prefix - prefix_base);
        const std::uint64_t values =
            static_cast<std::uint64_t>(q.k4_count) * kBoundaryTileValues;

        for (std::uint64_t linear = threadIdx.x; linear < values;
             linear += blockDim.x) {
            const std::uint64_t boundary_index = linear / q.k4_count;
            const std::uint32_t jb = static_cast<std::uint32_t>(
                linear % q.k4_count);
            const std::uint32_t li5 = static_cast<std::uint32_t>(
                boundary_index / kBoundaryTile);
            const std::uint32_t li4 = static_cast<std::uint32_t>(
                boundary_index % kBoundaryTile);
            const std::uint64_t ii4 = tile4 * kBoundaryTile + li4;
            const std::uint64_t ii5 = tile5 * kBoundaryTile + li5;
            if (ii4 < d.I4 && ii5 < d.I5) {
                const std::uint64_t n = ii4 + d.I4 * ii5;
                const std::uint64_t row = qrow +
                    static_cast<std::uint64_t>(p) * q.k4_count + jb;
                R[i1 * capacity * d.N + row + capacity * n] =
                    compute_v_value_simt(
                        B4, T1, d, q.k4_begin + jb,
                        q.il24, q.il34, q.il35, p,
                        tile4, tile5, li4, li5);
            }
        }
    }
#endif
}

// Association 1: form V on chip, then C^T V, and write only transformed R.
__global__ void produce_right_cv_dmma_kernel(
    const double* __restrict__ B4,
    const double* __restrict__ T1_all,
    const double* __restrict__ c_base,
    const SemiFusedQDescriptor* __restrict__ descriptors,
    std::uint64_t descriptor_count,
    double* __restrict__ R,
    SemiFusedDims d,
    std::uint64_t i1_count,
    std::uint64_t capacity,
    std::uint64_t row_base,
    std::uint64_t prefix_base,
    bool resident_c,
    std::uint32_t streamed_l34_begin,
    std::uint32_t streamed_l34_count)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    extern __shared__ __align__(32) double shared[];
    double* const source = shared;
    double* const c_stage = source + kSourceValues;
    double* const output_stage = c_stage + kCStageValues;
    const std::uint64_t n_tiles = d.tile4 * d.tile5;
    const std::uint64_t task_count =
        i1_count * descriptor_count * d.P * n_tiles;
    const unsigned warp = threadIdx.x / kWarpSize;

    for (std::uint64_t task = blockIdx.x; task < task_count;
         task += static_cast<std::uint64_t>(gridDim.x)) {
        std::uint64_t t = task;
        const std::uint64_t n_tile = t % n_tiles; t /= n_tiles;
        const std::uint32_t p = static_cast<std::uint32_t>(t % d.P); t /= d.P;
        const std::uint64_t qi = t % descriptor_count; t /= descriptor_count;
        const std::uint64_t i1 = t;
        const SemiFusedQDescriptor q = descriptors[qi];
        const std::uint32_t tile5 = static_cast<std::uint32_t>(n_tile % d.tile5);
        const std::uint32_t tile4 = static_cast<std::uint32_t>(n_tile / d.tile5);
        const double* const T1 = T1_all + i1 * d.T1;
        const double* const C = c_block_pointer(
            c_base, q, d, resident_c,
            streamed_l34_begin, streamed_l34_count);
        const std::uint64_t qrow = row_base + (q.row_prefix - prefix_base);
        const bool source_fits = q.k4_count <= kVirtualSourceCapacity;
        if (source_fits) {
            generate_v_source(
                B4, T1, source, d, q.k4_begin, q.k4_count,
                q.il24, q.il34, q.il35, p, tile4, tile5);
        }

        for (std::uint32_t i0 = 0; i0 < q.k3_count;
             i0 += kVirtualOutputTile) {
            const std::uint32_t icount = min_u32(
                kVirtualOutputTile, q.k3_count - i0);
            MmaAcc accum;
            wmma::fill_fragment(accum, 0.0);

            const std::uint32_t source_chunks = source_fits
                ? 1u
                : static_cast<std::uint32_t>(
                    (q.k4_count + kVirtualSourceCapacity - 1u) /
                    kVirtualSourceCapacity);
            for (std::uint32_t chunk = 0; chunk < source_chunks; ++chunk) {
                const std::uint32_t j0 = source_fits
                    ? 0u : chunk * kVirtualSourceCapacity;
                const std::uint32_t jcount = source_fits
                    ? q.k4_count
                    : min_u32(kVirtualSourceCapacity, q.k4_count - j0);
                if (!source_fits) {
                    generate_v_source(
                        B4, T1, source, d, q.k4_begin + j0, jcount,
                        q.il24, q.il34, q.il35, p, tile4, tile5);
                }
                for (std::uint64_t linear = threadIdx.x;
                     linear < kCStageValues; linear += blockDim.x) {
                    const std::uint32_t ir = static_cast<std::uint32_t>(
                        linear / kVirtualSourceCapacity);
                    const std::uint32_t jk = static_cast<std::uint32_t>(
                        linear % kVirtualSourceCapacity);
                    c_stage[linear] = ir < icount && jk < jcount
                        ? C[(q.k4_begin + j0 + jk) + d.K4 *
                            (q.k3_begin + i0 + ir)]
                        : 0.0;
                }
                __syncthreads();
                const std::uint32_t padded = static_cast<std::uint32_t>(
                    round_up_u64(jcount, kMmaK));
                for (std::uint32_t k = 0; k < padded; k += kMmaK) {
                    MmaArow af;
                    MmaBrow bf;
                    wmma::load_matrix_sync(
                        af, c_stage + k, kVirtualSourceCapacity);
                    wmma::load_matrix_sync(
                        bf, source + static_cast<std::uint64_t>(k) *
                                         kBoundaryTileValues + warp * kBoundaryTile,
                        kBoundaryTileValues);
                    wmma::mma_sync(accum, af, bf, accum);
                }
                __syncthreads();
            }

            wmma::store_matrix_sync(
                output_stage + warp * kBoundaryTile,
                accum, kBoundaryTileValues, wmma::mem_row_major);
            __syncthreads();
            for (std::uint64_t linear = threadIdx.x;
                 linear < static_cast<std::uint64_t>(icount) * kBoundaryTileValues;
                 linear += blockDim.x) {
                const std::uint64_t boundary_index = linear / icount;
                const std::uint64_t ib = linear % icount;
                const std::uint64_t li5 = boundary_index / kBoundaryTile;
                const std::uint64_t li4 = boundary_index % kBoundaryTile;
                const std::uint64_t ii4 = tile4 * kBoundaryTile + li4;
                const std::uint64_t ii5 = tile5 * kBoundaryTile + li5;
                if (ii4 < d.I4 && ii5 < d.I5) {
                    const std::uint64_t n = ii4 + d.I4 * ii5;
                    const std::uint64_t row = qrow +
                        static_cast<std::uint64_t>(p) * q.k3_count + i0 + ib;
                    R[i1 * capacity * d.N + row + capacity * n] =
                        output_stage[ib * kBoundaryTileValues + boundary_index];
                }
            }
            __syncthreads();
        }
    }
#else
    extern __shared__ __align__(32) double shared[];
    double* const source = shared;
    double* const output_stage =
        source + kSourceValues + kCStageValues;
    const std::uint64_t n_tiles = d.tile4 * d.tile5;
    const std::uint64_t task_count =
        i1_count * descriptor_count * d.P * n_tiles;

    for (std::uint64_t task = blockIdx.x; task < task_count;
         task += static_cast<std::uint64_t>(gridDim.x)) {
        std::uint64_t t = task;
        const std::uint64_t n_tile = t % n_tiles; t /= n_tiles;
        const std::uint32_t p = static_cast<std::uint32_t>(t % d.P); t /= d.P;
        const std::uint64_t qi = t % descriptor_count; t /= descriptor_count;
        const std::uint64_t i1 = t;
        const SemiFusedQDescriptor q = descriptors[qi];
        const std::uint32_t tile5 = static_cast<std::uint32_t>(n_tile % d.tile5);
        const std::uint32_t tile4 = static_cast<std::uint32_t>(n_tile / d.tile5);
        const double* const T1 = T1_all + i1 * d.T1;
        const double* const C = c_block_pointer(
            c_base, q, d, resident_c,
            streamed_l34_begin, streamed_l34_count);
        const std::uint64_t qrow = row_base + (q.row_prefix - prefix_base);
        const bool source_fits = q.k4_count <= kVirtualSourceCapacity;
        if (source_fits) {
            generate_v_source_simt(
                B4, T1, source, d, q.k4_begin, q.k4_count,
                q.il24, q.il34, q.il35, p, tile4, tile5);
        }

        for (std::uint32_t i0 = 0; i0 < q.k3_count;
             i0 += kVirtualOutputTile) {
            const std::uint32_t icount = min_u32(
                kVirtualOutputTile, q.k3_count - i0);
            const std::uint64_t output_values =
                static_cast<std::uint64_t>(icount) * kBoundaryTileValues;
            for (std::uint64_t linear = threadIdx.x;
                 linear < output_values; linear += blockDim.x) {
                output_stage[linear] = 0.0;
            }
            __syncthreads();

            const std::uint32_t source_chunks = source_fits
                ? 1u
                : static_cast<std::uint32_t>(
                    (q.k4_count + kVirtualSourceCapacity - 1u) /
                    kVirtualSourceCapacity);
            for (std::uint32_t chunk = 0; chunk < source_chunks; ++chunk) {
                const std::uint32_t j0 = source_fits
                    ? 0u : chunk * kVirtualSourceCapacity;
                const std::uint32_t jcount = source_fits
                    ? q.k4_count
                    : min_u32(kVirtualSourceCapacity, q.k4_count - j0);
                if (!source_fits) {
                    generate_v_source_simt(
                        B4, T1, source, d, q.k4_begin + j0, jcount,
                        q.il24, q.il34, q.il35, p, tile4, tile5);
                }
                for (std::uint64_t linear = threadIdx.x;
                     linear < output_values; linear += blockDim.x) {
                    const std::uint32_t ib = static_cast<std::uint32_t>(
                        linear / kBoundaryTileValues);
                    const std::uint32_t boundary_index = static_cast<std::uint32_t>(
                        linear % kBoundaryTileValues);
                    double partial = 0.0;
                    for (std::uint32_t jk = 0; jk < jcount; ++jk) {
                        partial += C[(q.k4_begin + j0 + jk) + d.K4 *
                                     (q.k3_begin + i0 + ib)] *
                                   source[static_cast<std::uint64_t>(jk) *
                                          kBoundaryTileValues + boundary_index];
                    }
                    output_stage[linear] += partial;
                }
                __syncthreads();
            }

            for (std::uint64_t linear = threadIdx.x;
                 linear < output_values; linear += blockDim.x) {
                const std::uint32_t ib = static_cast<std::uint32_t>(
                    linear / kBoundaryTileValues);
                const std::uint32_t boundary_index = static_cast<std::uint32_t>(
                    linear % kBoundaryTileValues);
                const std::uint32_t li5 = boundary_index / kBoundaryTile;
                const std::uint32_t li4 = boundary_index % kBoundaryTile;
                const std::uint64_t ii4 = tile4 * kBoundaryTile + li4;
                const std::uint64_t ii5 = tile5 * kBoundaryTile + li5;
                if (ii4 < d.I4 && ii5 < d.I5) {
                    const std::uint64_t n = ii4 + d.I4 * ii5;
                    const std::uint64_t row = qrow +
                        static_cast<std::uint64_t>(p) * q.k3_count + i0 + ib;
                    R[i1 * capacity * d.N + row + capacity * n] =
                        output_stage[linear];
                }
            }
            __syncthreads();
        }
    }
#endif
}

__global__ void accumulate_reflection_compact_batched_kernel(
    const double* __restrict__ dense,
    const std::uint32_t* __restrict__ rep_m,
    const std::uint32_t* __restrict__ rep_n,
    const std::uint32_t* __restrict__ partner_m,
    const std::uint32_t* __restrict__ partner_n,
    double* __restrict__ compact,
    std::uint64_t compact_count,
    std::uint64_t M,
    std::uint64_t N,
    std::uint64_t i1_count)
{
    const std::uint64_t total = i1_count * compact_count;
    for (std::uint64_t linear =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint64_t slot = linear % compact_count;
        const std::uint64_t i1 = linear / compact_count;
        const double* const A = dense + i1 * M * N;
        double value = A[rep_m[slot] + M * rep_n[slot]];
        value += A[partner_m[slot] + M * partner_n[slot]];
        compact[linear] += value;
    }
}

} // namespace b4gpu::semifused_detail
