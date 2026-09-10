// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace b4gpu::cuda_detail {

struct SupportIndexRange {
    std::uint32_t begin;
    std::uint32_t end;
};

__device__ __forceinline__ int support_iabs(int value) noexcept
{
    return value < 0 ? -value : value;
}

__device__ __forceinline__ bool support_triangle(
    int a, int b, int c) noexcept
{
    return ((a ^ b ^ c) & 1) == 0 &&
           c >= support_iabs(a - b) && c <= a + b;
}

// Every virtual intertwiner and lifted-spin axis accepted by the device recurrence is a
// regular doubled-spin axis with step two. Return the exact half-open index
// interval intersecting [lo,hi] and the requested parity.
__device__ __forceinline__ SupportIndexRange support_axis_interval(
    int first,
    std::uint64_t count,
    int lo,
    int hi,
    int parity) noexcept
{
    if (count == 0 || lo > hi || (first & 1) != (parity & 1)) {
        return SupportIndexRange{0, 0};
    }
    const int last = first + 2 * static_cast<int>(count - 1);
    lo = lo > first ? lo : first;
    hi = hi < last ? hi : last;
    if (lo > hi) return SupportIndexRange{0, 0};
    const int begin = lo <= first ? 0 : (lo - first + 1) / 2;
    const int end = (hi - first) / 2 + 1;
    return SupportIndexRange{
        static_cast<std::uint32_t>(begin),
        static_cast<std::uint32_t>(end)};
}

__device__ __forceinline__ SupportIndexRange support_two_triangles(
    int first,
    std::uint64_t count,
    int a,
    int b,
    int c,
    int d) noexcept
{
    const int parity_ab = (a + b) & 1;
    if (parity_ab != ((c + d) & 1)) {
        return SupportIndexRange{0, 0};
    }
    const int lo_ab = support_iabs(a - b);
    const int lo_cd = support_iabs(c - d);
    const int hi_ab = a + b;
    const int hi_cd = c + d;
    return support_axis_interval(
        first, count,
        lo_ab > lo_cd ? lo_ab : lo_cd,
        hi_ab < hi_cd ? hi_ab : hi_cd,
        parity_ab);
}

__device__ __forceinline__ SupportIndexRange support_one_triangle(
    int first,
    std::uint64_t count,
    int a,
    int b) noexcept
{
    return support_axis_interval(
        first, count, support_iabs(a - b), a + b, (a + b) & 1);
}

// Compact descriptors are generated once per X value (p/v support) and once
// per fixed-(q,r-slab) tile (i/j support). They keep the measured fused tail
// memory-bounded while keeping the production slab layout compact.
static __global__ void support_q_slab_build_x_descriptors_kernel(
    SupportIndexRange* __restrict__ p_ranges_by_v,
    std::uint8_t* __restrict__ v_x_active,
    int Dx,
    int Dk5_first,
    int Dj14,
    int Dj15,
    int Dj25,
    std::uint64_t V,
    std::uint64_t P)
{
    const std::uint64_t begin =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    // launch_blocks deliberately caps CUDA grids for portability.  A
    // grid-stride walk keeps descriptor construction correct even for a very
    // long, skinny support domain whose logical length exceeds that cap.
    for (std::uint64_t v = begin; v < V; v += stride) {
        const int Dk5 = Dk5_first + 2 * static_cast<int>(v);
        p_ranges_by_v[v] = support_one_triangle(Dj25, P, Dk5, Dj15);
        v_x_active[v] = static_cast<std::uint8_t>(
            support_triangle(Dk5, Dx, Dj14));
    }
}

static __global__ void support_q_slab_build_z_descriptors_kernel(
    SupportIndexRange* __restrict__ i_ranges_by_z,
    SupportIndexRange* __restrict__ j_ranges_by_z,
    int Dx,
    int Dk3_first,
    int Dk4_first,
    int Dj24,
    int Dj34,
    int Dj35,
    std::uint64_t q,
    std::uint64_t r_begin,
    std::uint64_t Rt,
    std::uint64_t I,
    std::uint64_t J,
    std::uint64_t S)
{
    const std::uint64_t begin =
        static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride =
        static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    const std::uint64_t Z = Rt * S;
    for (std::uint64_t z = begin; z < Z; z += stride) {
        const std::uint64_t rr = z % Rt;
        const std::uint64_t s = z / Rt;
        const int Dl24 = Dj24 + 2 * static_cast<int>(q);
        const int Dl34 = Dj34 + 2 * static_cast<int>(r_begin + rr);
        const int Dl35 = Dj35 + 2 * static_cast<int>(s);
        i_ranges_by_z[z] = support_two_triangles(
            Dk3_first, I, Dl34, Dl35, Dx, Dl24);
        j_ranges_by_z[z] = support_two_triangles(
            Dk4_first, J, Dl34, Dl24, Dx, Dl35);
    }
}

// Support-aware fused X5/final microkernel. It keeps the measured T8 warp
// organization, but intersects all exact sjC/sjB/B4/B5 triangle masks before
// entering the inner p loop. Each CTA then deterministically reduces its
// complete 32xT8 tile and emits one scalar partial instead of up to 256 values.
static __global__ void support_q_slab_tail_partials_kernel(
    const double* __restrict__ X4re,
    const double* __restrict__ X4im,
    const double* __restrict__ B5re,
    const double* __restrict__ B5im,
    const double* __restrict__ sjB_packed,
    const cuDoubleComplex* __restrict__ B4,
    cuDoubleComplex* __restrict__ partials,
    const SupportIndexRange* __restrict__ i_ranges_by_z,
    const SupportIndexRange* __restrict__ j_ranges_by_z,
    const SupportIndexRange* __restrict__ p_ranges_by_v,
    const std::uint8_t* __restrict__ v_x_active,
    int Dk4_first,
    int Dk5_first,
    int Dj14,
    int Dj35,
    int Dj45,
    std::uint64_t q,
    std::uint64_t r_begin,
    std::uint64_t Rt,
    std::uint64_t V,
    std::uint64_t J,
    std::uint64_t Rdim,
    std::uint64_t P,
    std::uint64_t S,
    std::uint64_t T,
    std::uint64_t x4_z_pitch,
    std::uint64_t x4_v_pitch,
    std::uint64_t x4_p_pitch,
    std::uint64_t b_tile_size)
{
    if (blockDim.x != 32 || blockDim.y != 1 || blockDim.z != 1 ||
        T == 0 || b_tile_size == 0) return;
    constexpr unsigned warp_size = 32;
    constexpr unsigned t_tile = 8;
    constexpr unsigned full_mask = 0xffffffffu;
    const unsigned lane = threadIdx.x;
    const std::uint64_t JR = J * Rt;
    const std::uint64_t SV = S * V;
    const std::uint64_t jr = blockIdx.x * warp_size + lane;
    const bool valid_jr = jr < JR;
    const std::uint64_t j_index = valid_jr ? jr % J : 0;
    const std::uint64_t rr = valid_jr ? jr / J : 0;
    const int Dk4 = Dk4_first + 2 * static_cast<int>(j_index);

    const std::uint64_t b_begin = blockIdx.y * b_tile_size;
    const std::uint64_t b_end =
        (b_begin + b_tile_size < SV) ? b_begin + b_tile_size : SV;
    const std::uint64_t t_begin = blockIdx.z * t_tile;
    if (b_begin >= b_end || t_begin >= T) return;
    const unsigned tile_T = static_cast<unsigned>(
        T - t_begin < t_tile ? T - t_begin : t_tile);

    extern __shared__ double shared[];
    double* const sum_re = shared;
    double* const sum_im = shared + warp_size * t_tile;
#pragma unroll
    for (unsigned tt = 0; tt < t_tile; ++tt) {
        const unsigned slot = lane + warp_size * tt;
        sum_re[slot] = 0.0;
        sum_im[slot] = 0.0;
    }

    for (std::uint64_t b = b_begin; b < b_end; ++b) {
        const std::uint64_t s_index = b % S;
        const std::uint64_t v_index = b / S;
        const int Dl35 = Dj35 + 2 * static_cast<int>(s_index);
        const int Dk5 = Dk5_first + 2 * static_cast<int>(v_index);
        const std::uint64_t z = rr + Rt * s_index;
        const SupportIndexRange i_range = valid_jr
            ? i_ranges_by_z[z]
            : SupportIndexRange{0, 0};
        const SupportIndexRange j_range = valid_jr
            ? j_ranges_by_z[z]
            : SupportIndexRange{0, 0};
        const bool s_active = valid_jr &&
            i_range.begin < i_range.end &&
            j_index >= j_range.begin && j_index < j_range.end;
        if (__ballot_sync(full_mask, s_active) == 0) continue;

        const bool k5_x = v_x_active[v_index] != 0;
        unsigned lane_k5_t = 0;
        if (lane < tile_T) {
            const int Dl45 = Dj45 +
                2 * static_cast<int>(t_begin + lane);
            lane_k5_t = support_triangle(Dk5, Dl35, Dl45) ? 1u : 0u;
        }

        bool active_t[t_tile];
        bool lane_active = false;
#pragma unroll
        for (unsigned tt = 0; tt < t_tile; ++tt) {
            bool active = false;
            const bool k5_t = __shfl_sync(
                full_mask, lane_k5_t, tt) != 0;
            if (tt < tile_T && s_active && k5_x && k5_t) {
                const int Dl45 = Dj45 +
                    2 * static_cast<int>(t_begin + tt);
                active = support_triangle(Dk4, Dl45, Dj14);
            }
            active_t[tt] = active;
            lane_active = lane_active || active;
        }
        if (__ballot_sync(full_mask, lane_active) == 0) continue;

        const SupportIndexRange p_range = p_ranges_by_v[v_index];
        if (p_range.begin >= p_range.end) continue;
        double dot_re[t_tile];
        double dot_im[t_tile];
#pragma unroll
        for (unsigned tt = 0; tt < t_tile; ++tt) {
            dot_re[tt] = 0.0;
            dot_im[tt] = 0.0;
        }

        for (std::uint64_t p = p_range.begin; p < p_range.end; ++p) {
            double ar = 0.0;
            double ai = 0.0;
            if (lane_active) {
                const std::uint64_t index = j_index +
                    x4_z_pitch * z + x4_v_pitch * v_index +
                    x4_p_pitch * p;
                ar = X4re[index];
                ai = X4im[index];
            }
            double lane_br = 0.0;
            double lane_bi = 0.0;
            if (lane < tile_T) {
                const std::uint64_t t = t_begin + lane;
                const std::uint64_t k5_index = t + T * (p + P * b);
                lane_br = B5re[k5_index];
                lane_bi = B5im[k5_index];
            }
#pragma unroll
            for (unsigned tt = 0; tt < t_tile; ++tt) {
                // Every lane named by full_mask must execute each shuffle.
                // active_t is lane-dependent because j/r support differs
                // across the warp, so only guard the arithmetic below.
                const double br = __shfl_sync(full_mask, lane_br, tt);
                const double bi = __shfl_sync(full_mask, lane_bi, tt);
                if (active_t[tt]) {
                    dot_re[tt] = fma(ar, br, dot_re[tt]);
                    dot_re[tt] = fma(-ai, bi, dot_re[tt]);
                    dot_im[tt] = fma(ar, bi, dot_im[tt]);
                    dot_im[tt] = fma(ai, br, dot_im[tt]);
                }
            }
        }

#pragma unroll
        for (unsigned tt = 0; tt < t_tile; ++tt) {
            if (active_t[tt]) {
                const std::uint64_t t = t_begin + tt;
                const double weight = sjB_packed[
                    j_index + J * (b + SV * t)];
                const unsigned slot = lane + warp_size * tt;
                sum_re[slot] = fma(dot_re[tt], weight, sum_re[slot]);
                sum_im[slot] = fma(dot_im[tt], weight, sum_im[slot]);
            }
        }
    }

    double local_re = 0.0;
    double local_im = 0.0;
    if (valid_jr) {
        const std::uint64_t r = r_begin + rr;
#pragma unroll
        for (unsigned tt = 0; tt < t_tile; ++tt) {
            if (tt < tile_T) {
                const std::uint64_t t = t_begin + tt;
                const unsigned slot = lane + warp_size * tt;
                const cuDoubleComplex k4 = B4[
                    j_index + J * (t + T * (r + Rdim * q))];
                local_re = fma(sum_re[slot], k4.x, local_re);
                local_re = fma(-sum_im[slot], k4.y, local_re);
                local_im = fma(sum_re[slot], k4.y, local_im);
                local_im = fma(sum_im[slot], k4.x, local_im);
            }
        }
    }
    for (unsigned offset = 16; offset != 0; offset >>= 1) {
        local_re += __shfl_down_sync(full_mask, local_re, offset);
        local_im += __shfl_down_sync(full_mask, local_im, offset);
    }
    if (lane == 0) {
        const std::uint64_t partial = blockIdx.x +
            gridDim.x * (blockIdx.y +
                         static_cast<std::uint64_t>(gridDim.y) * blockIdx.z);
        partials[partial] = make_cuDoubleComplex(local_re, local_im);
    }
}

} // namespace b4gpu::cuda_detail

