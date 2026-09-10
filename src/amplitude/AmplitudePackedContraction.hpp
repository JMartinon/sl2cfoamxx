// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "amplitude/AmplitudeInputSupport.hpp"
#include "amplitude/AmplitudePackedSJTensors.hpp"
#include "amplitude/AmplitudePackedBoosters.hpp"

#ifndef B4PACK_RESTRICT
#  if defined(__GNUC__) || defined(__clang__)
#    define B4PACK_RESTRICT __restrict__
#  elif defined(_MSC_VER)
#    define B4PACK_RESTRICT __restrict
#  else
#    define B4PACK_RESTRICT
#  endif
#endif

#ifndef B4PACK_FORCE_INLINE
#  if defined(__GNUC__) || defined(__clang__)
#    define B4PACK_FORCE_INLINE inline __attribute__((always_inline))
#  elif defined(_MSC_VER)
#    define B4PACK_FORCE_INLINE __forceinline
#  else
#    define B4PACK_FORCE_INLINE inline
#  endif
#endif

namespace b4support {

struct PackedContractionOptions {
    bool use_openmp = true;
    std::size_t parallel_threshold = 32768;
};

inline Range1D intersect_2_ranges(const Range1D& a,
                                  const Range1D& b) noexcept
{
    const std::uint32_t begin = std::max(a.begin, b.begin);
    const std::uint32_t end   = std::min(a.end,   b.end);
    
    if (begin >= end) {
        return empty_range_1d();
    }
    
    return Range1D{begin, end, 0ull};
}

inline void zero_range(double* B4PACK_RESTRICT y,
                       std::size_t n) noexcept
{
#ifdef _OPENMP
#pragma omp simd
#endif
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = 0.0;
    }
}

template <std::size_t N>
B4PACK_FORCE_INLINE void axpy_fixed_n(double alpha,
                                     const double* B4PACK_RESTRICT x,
                                     double* B4PACK_RESTRICT y) noexcept
{
    // N is a compile-time constant so the T2/T4 AXPY loop can be unrolled and
    // vectorized without a runtime-size branch.
    for (std::size_t i = 0; i < N; ++i) {
        y[i] += alpha * x[i];
    }
}

B4PACK_FORCE_INLINE void axpy_range(double alpha,
                                   const double* B4PACK_RESTRICT x,
                                   double* B4PACK_RESTRICT y,
                                   std::size_t n) noexcept
{
    // Tiny l25 AXPY kernel. Only n <= 4 is manually specialized.
    // For n >= 5, keep the old SIMD loop. The symmetric benchmark has plenty
    // of medium K, and over-unrolling those made performance worse. Naturally.
    switch (n) {
        case 0:  return;
        case 1:  axpy_fixed_n<1>(alpha, x, y); return;
        case 2:  axpy_fixed_n<2>(alpha, x, y); return;
        case 3:  axpy_fixed_n<3>(alpha, x, y); return;
        case 4:  axpy_fixed_n<4>(alpha, x, y); return;
        default:
            break;
    }

#ifdef _OPENMP
#pragma omp simd
#endif
    for (std::size_t i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}


struct ReusableDoubleBuffer {
    std::unique_ptr<double[]> data;
    std::size_t size = 0;
    std::size_t capacity = 0;
    
    void ensure(std::size_t n) {
        if (n > capacity) {
            data.reset(new double[n]);
            capacity = n;
        }
        
        size = n;
    }
    
    double* ptr() noexcept {
        return data.get();
    }
    
    const double* ptr() const noexcept {
        return data.get();
    }
};

void fill_T1_values_packed_slice(
                                        const AmplitudeInputSupportPlan& support,
                                        const PackedAmplitudeSJTensors& sj,
                                        const PackedAmplitudeBoosters& B,
                                        const RangeBlockSupport1D& T1_support,
                                        std::size_t ix,
                                        double* B4PACK_RESTRICT T1_values,
                                        PackedContractionOptions opt = {});

void fill_T3_values_packed_slice(
                                        const AmplitudeInputSupportPlan& support,
                                        const PackedAmplitudeSJTensors& sj,
                                        const PackedAmplitudeBoosters& B,
                                        const RangeBlockSupport1D& T3_support,
                                        std::size_t ix,
                                        double* B4PACK_RESTRICT T3_values,
                                        PackedContractionOptions opt = {});

} // namespace b4support

#undef B4PACK_RESTRICT
#undef B4PACK_FORCE_INLINE
