// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "CoherentGpuApi.hpp"

namespace b4gpu::cuda_detail {
__global__ void weight_and_transpose_sjA_kernel(
    const double *__restrict__ sjA, const Complex64 *__restrict__ c1,
    double *__restrict__ weighted, std::uint64_t A, std::uint64_t V,
    std::uint64_t P, bool imaginary) {
  const std::uint64_t total = A * V * P;
  for (std::uint64_t linear =
           blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
       linear < total;
       linear += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    std::uint64_t t = linear;
    const std::uint64_t v = t % V;
    t /= V;
    const std::uint64_t a = t % A;
    const std::uint64_t p = t / A;
    const double coeff = imaginary ? c1[a].im : c1[a].re;
    weighted[v + V * (a + A * p)] = coeff * sjA[a + A * (v + V * p)];
  }
}

__global__ void interleave_complex_kernel(const double *__restrict__ re,
                                          const double *__restrict__ im,
                                          cuDoubleComplex *__restrict__ out,
                                          std::uint64_t count) {
  for (std::uint64_t i =
           blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
       i < count; i += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    out[i] = make_cuDoubleComplex(re[i], im[i]);
  }
}
// The fused tail broadcasts consecutive t values across a warp.  Its B5
// representation therefore makes t, rather than p, the unit-stride axis:
// [t,p,b], b=s+S*v, making the fused warp loads coalesced.
__global__ void pack_q_slab_booster5_fused_split_kernel(
    const cuDoubleComplex *__restrict__ B5, double *__restrict__ packed_re,
    double *__restrict__ packed_im, std::uint64_t V, std::uint64_t P,
    std::uint64_t T, std::uint64_t S) {
  const std::uint64_t total = V * P * T * S;
  for (std::uint64_t linear =
           blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
       linear < total;
       linear += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    std::uint64_t rem = linear;
    const std::uint64_t t = rem % T;
    rem /= T;
    const std::uint64_t p = rem % P;
    const std::uint64_t b = rem / P;
    const std::uint64_t s = b % S;
    const std::uint64_t v = b / S;
    const cuDoubleComplex value = B5[v + V * (p + P * (t + T * s))];
    packed_re[linear] = value.x;
    packed_im[linear] = value.y;
  }
}

// Reorder canonical sjB[v,j,s,t] to [j,b,t], b=s+S*v.  The fused q-slab
// tail below assigns consecutive j/r pairs to a warp.  Making j the unit-
// stride index turns its sjB reads into coalesced transactions; retaining the
// canonical v-fast layout here would give adjacent lanes a V-double stride.
// sjB changes with x, so it must be packed once after each wing generation,
// outside the q/r loops.  It is not a device-lifetime fixed input.
__global__ void pack_sjB_for_fused_q_slab_kernel(
    const double *__restrict__ sjB, double *__restrict__ packed,
    std::uint64_t V, std::uint64_t J, std::uint64_t S, std::uint64_t T) {
  const std::uint64_t SV = S * V;
  const std::uint64_t total = J * SV * T;
  for (std::uint64_t linear =
           blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
       linear < total;
       linear += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    std::uint64_t rem = linear;
    const std::uint64_t j = rem % J;
    rem /= J;
    const std::uint64_t b = rem % SV;
    const std::uint64_t t = rem / SV;
    const std::uint64_t s = b % S;
    const std::uint64_t v = b / S;
    packed[linear] = sjB[v + V * (j + J * (s + S * t))];
  }
}
// The exact-support fused tail in CoherentGpuSupportKernels.cuh retains the
// measured one-warp T8 organization.  Keep its tile width here because the
// backend also uses it when sizing bounded workspaces and launch grids.
constexpr unsigned fused_q_slab_t_tile = 8;

// Direct-complex pointwise operand used by the production cuTENSOR X2 stage.
// H[k,u,i,p] = sjD[i,k,u,q] * B2[k,p,u,q], with k contiguous.
__global__ void build_q_slab_h_complex_kernel(
    const double *__restrict__ sjD, const cuDoubleComplex *__restrict__ B2,
    cuDoubleComplex *__restrict__ H, std::uint64_t q, std::uint64_t I,
    std::uint64_t K, std::uint64_t P, std::uint64_t U) {
  const std::uint64_t total = K * U * I * P;
  for (std::uint64_t linear =
           blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
       linear < total;
       linear += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
    std::uint64_t rem = linear;
    const std::uint64_t k = rem % K;
    rem /= K;
    const std::uint64_t u = rem % U;
    rem /= U;
    const std::uint64_t i = rem % I;
    const std::uint64_t p = rem / I;
    const double d = sjD[i + I * (k + K * (u + U * q))];
    const cuDoubleComplex b = B2[k + K * (p + P * (u + U * q))];
    H[linear] = make_cuDoubleComplex(d * b.x, d * b.y);
  }
}
__global__ void
reduce_q_slab_partials_kernel(const cuDoubleComplex *__restrict__ partials,
                              std::uint64_t count,
                              cuDoubleComplex *__restrict__ output) {
  double sum_re = 0.0;
  double sum_im = 0.0;
  for (std::uint64_t i = threadIdx.x; i < count; i += blockDim.x) {
    sum_re += partials[i].x;
    sum_im += partials[i].y;
  }
  extern __shared__ double scratch[];
  double *sre = scratch;
  double *sim = scratch + blockDim.x;
  sre[threadIdx.x] = sum_re;
  sim[threadIdx.x] = sum_im;
  __syncthreads();
  for (unsigned offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      sre[threadIdx.x] += sre[threadIdx.x + offset];
      sim[threadIdx.x] += sim[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    output[0] = make_cuDoubleComplex(sre[0], sim[0]);
  }
}

// First stage for reducing the larger partial array produced by the fused
// tail.  Unlike reduce_q_slab_partials_kernel, this launch uses many CTAs and
// can therefore consume the partial array at device-memory bandwidth.  The
// caller chooses gridDim.x (normally a few multiples of the SM count),
// provides gridDim.x output elements, then invokes
// reduce_q_slab_partials_kernel on those compact block sums.
//
// blockDim.x must be a multiple of 32 and no greater than 1024.  There are no
// atomics: every CTA owns exactly one output, and a fixed launch geometry has
// a fixed summation tree.
__global__ void reduce_q_slab_fused_block_sums_kernel(
    const cuDoubleComplex *__restrict__ partials, std::uint64_t count,
    cuDoubleComplex *__restrict__ block_sums) {
  if ((blockDim.x & 31u) != 0 || blockDim.x > 1024)
    return;

  double sum_re = 0.0;
  double sum_im = 0.0;
  const std::uint64_t first =
      blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
  for (std::uint64_t i = first; i < count; i += stride) {
    sum_re += partials[i].x;
    sum_im += partials[i].y;
  }

  constexpr unsigned full_mask = 0xffffffffu;
  for (unsigned offset = 16; offset > 0; offset >>= 1) {
    sum_re += __shfl_down_sync(full_mask, sum_re, offset);
    sum_im += __shfl_down_sync(full_mask, sum_im, offset);
  }

  __shared__ double warp_re[32];
  __shared__ double warp_im[32];
  const unsigned lane = threadIdx.x & 31u;
  const unsigned warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_re[warp] = sum_re;
    warp_im[warp] = sum_im;
  }
  __syncthreads();

  if (warp == 0) {
    const unsigned warp_count = (blockDim.x + 31u) >> 5;
    sum_re = lane < warp_count ? warp_re[lane] : 0.0;
    sum_im = lane < warp_count ? warp_im[lane] : 0.0;
    for (unsigned offset = 16; offset > 0; offset >>= 1) {
      sum_re += __shfl_down_sync(full_mask, sum_re, offset);
      sum_im += __shfl_down_sync(full_mask, sum_im, offset);
    }
    if (lane == 0) {
      block_sums[blockIdx.x] = make_cuDoubleComplex(sum_re, sum_im);
    }
  }
}
} // namespace b4gpu::cuda_detail

