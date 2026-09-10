// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "CoherentGpuRecurrence.cuh"

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace b4gpu::cuda_detail {
namespace {

constexpr unsigned recurrence_block_threads = 32;
constexpr unsigned fused_operand_block_threads = 256;

struct PreparedShared {
    double* coupling;
    double* Y_base;
    double* Y_f4;
    double* values;
};

struct AxisIntersection {
    std::uint64_t begin;
    std::uint64_t end;
};

__device__ __forceinline__ int iabs_device(int value) noexcept
{
    return value < 0 ? -value : value;
}

__device__ __forceinline__ int imax_device(int lhs, int rhs) noexcept
{
    return lhs > rhs ? lhs : rhs;
}

__device__ __forceinline__ int imin_device(int lhs, int rhs) noexcept
{
    return lhs < rhs ? lhs : rhs;
}

__device__ __forceinline__ bool triangle_device(int a, int b, int c) noexcept
{
    if (a < 0 || b < 0 || c < 0) return false;
    const std::int64_t aa = a;
    const std::int64_t bb = b;
    const std::int64_t cc = c;
    if (((aa + bb + cc) & 1) != 0) return false;
    return aa + bb >= cc && aa + cc >= bb && bb + cc >= aa;
}

__device__ __forceinline__ bool even_phase_device(std::int64_t Dsum,
                                                   double& phase) noexcept
{
    if ((Dsum & 1) != 0) return false;
    std::int64_t half = Dsum / 2;
    if (half < 0) half = -half;
    phase = (half & 1) != 0 ? -1.0 : 1.0;
    return true;
}

__device__ __forceinline__ int doubled_sum_parity_device(
    std::int64_t Dsum) noexcept
{
    const int remainder = static_cast<int>(Dsum % 2);
    return remainder < 0 ? -remainder : remainder;
}

__device__ __forceinline__ double half_spin_device(int doubled) noexcept
{
    return 0.5 * static_cast<double>(doubled);
}

__device__ __forceinline__ void compensated_sqrt_product_device(
    double lhs,
    double rhs,
    double& value,
    double& correction) noexcept
{
    double product = lhs * rhs;
    double product_error = fma(lhs, rhs, -product);
    if (product < 0.0) {
        product = -product;
        product_error = -product_error;
    }

    value = sqrt(product);
    if (value == 0.0) {
        correction = 0.0;
        return;
    }

    // One Newton step using both halves of the error-free product.  The
    // source-local CUDA options keep FP64 FMA, division, square root and
    // denormals precise even when the rest of the target uses fast math.
    const double residual = fma(-value, value, product) + product_error;
    correction = residual / (2.0 * value);
}

__device__ __forceinline__ double coupled_pair_factor_device(
    int Da,
    int Db,
    int Dc,
    int Dd,
    int Dvar) noexcept
{
    // WIGSGLL's coefficient layout evaluates the complete square-root
    // coefficient once per recurrence point.  In doubled-spin integers,
    //
    // E(j) = 1/16 sqrt(|(D^2-(Da-Db)^2)
    //                    ((Da+Db+2)^2-D^2)
    //                    (D^2-(Dc-Dd)^2)
    //                    ((Dc+Dd+2)^2-D^2)|).
    //
    // Factoring every difference of squares preserves exact triangle-boundary
    // zeros.  The product cannot overflow double for non-negative int32 spin
    // labels.  Keep one square root, but retain the low part of the final
    // product with an FMA and apply one Newton correction to the root.  This is
    // the GPU analogue of the compensated CPU pair-factor path without paying
    // for a second square root or storing a second shared coefficient array.
    const double D = static_cast<double>(Dvar);
    const double difference_ab =
        static_cast<double>(Da) - static_cast<double>(Db);
    const double sum_ab =
        static_cast<double>(Da) + static_cast<double>(Db) + 2.0;
    const double difference_cd =
        static_cast<double>(Dc) - static_cast<double>(Dd);
    const double sum_cd =
        static_cast<double>(Dc) + static_cast<double>(Dd) + 2.0;
    const double low_ab = (D - difference_ab) * (D + difference_ab);
    const double high_ab = (sum_ab - D) * (sum_ab + D);
    const double low_cd = (D - difference_cd) * (D + difference_cd);
    const double high_cd = (sum_cd - D) * (sum_cd + D);
    const double pair_ab = low_ab * high_ab;
    const double pair_cd = low_cd * high_cd;
    // Scaling the compensated square root by 1/16 is exact in binary.  At the
    // physical spin range used by this project (j <= 400, hence doubled labels
    // <= 800), each pair product is itself exact in binary64; the correction
    // therefore recovers the complete radicand to double-double precision
    // before the final rounding.
    double value = 0.0;
    double correction = 0.0;
    compensated_sqrt_product_device(
        pair_ab, pair_cd, value, correction);

    // E = sqrt(product) / 16.  The FMA retains the low root component when it
    // affects the correctly rounded coefficient.
    return fma(0.0625, value, 0.0625 * correction);
}

__device__ __forceinline__ double zero_branch_z_device(
    int Dj2,
    int Dj3,
    int Dj5,
    int Dj6,
    int Dvar) noexcept
{
    // j * sqrt(|(j^2-(j2+j3+1)^2)(j^2-(j5+j6+1)^2)|), written
    // entirely in doubled-spin factors.  D/8 is (D/2) times the 1/4
    // introduced by the two squared half-spin differences.
    const double D = static_cast<double>(Dvar);
    const double sum23 =
        static_cast<double>(Dj2) + static_cast<double>(Dj3) + 2.0;
    const double sum56 =
        static_cast<double>(Dj5) + static_cast<double>(Dj6) + 2.0;
    const double first = (sum23 - D) * (sum23 + D);
    const double second = (sum56 - D) * (sum56 + D);
    double value = 0.0;
    double correction = 0.0;
    compensated_sqrt_product_device(first, second, value, correction);
    const double scale = 0.125 * D;
    return fma(scale, value, scale * correction);
}

__device__ __forceinline__ double prepared_x_device(
    int Dmin,
    bool zero_branch,
    std::uint32_t i,
    const PreparedShared& shared) noexcept
{
    if (zero_branch) return shared.coupling[i + 1];
    const double j = half_spin_device(Dmin + 2 * static_cast<int>(i));
    return j * shared.coupling[i + 1];
}

__device__ __forceinline__ double prepared_z_device(
    int Dmin,
    bool zero_branch,
    std::uint32_t i,
    const PreparedShared& shared) noexcept
{
    if (zero_branch) return shared.coupling[i];
    const double j = half_spin_device(Dmin + 2 * static_cast<int>(i));
    return (j + 1.0) * shared.coupling[i];
}

__device__ __forceinline__ void record_failure(
    std::uint64_t* failure_count) noexcept
{
    if (failure_count != nullptr) {
        atomicAdd(reinterpret_cast<unsigned long long*>(failure_count), 1ULL);
    }
}

__device__ __forceinline__ AxisIntersection intersect_regular_axis(
    const int* axis,
    std::uint64_t count,
    int Dmin,
    int Dmax) noexcept
{
    if (axis == nullptr || count == 0) return AxisIntersection{0, 0};

    const int first = axis[0];
    const int last = axis[count - 1];
    if ((first & 1) != (Dmin & 1)) return AxisIntersection{0, 0};
    int lo = imax_device(Dmin, first);
    int hi = imin_device(Dmax, last);
    if (lo > hi) return AxisIntersection{0, 0};

    const int parity = Dmin & 1;
    if ((lo & 1) != parity) ++lo;
    if ((hi & 1) != parity) --hi;
    if (lo > hi || lo < first) return AxisIntersection{0, 0};

    const std::uint64_t begin =
        static_cast<std::uint64_t>((lo - first) / 2);
    const std::uint64_t end =
        static_cast<std::uint64_t>((hi - first) / 2 + 1);
    return AxisIntersection{begin, end < count ? end : count};
}

__device__ bool prepare_coefficients(
    int Dmin,
    int Dmax,
    int Dj2,
    int Dj3,
    int Dj5,
    int Dj6,
    std::uint32_t capacity,
    const PreparedShared& shared,
    std::uint32_t& n) noexcept
{
    if (Dmin > Dmax || ((Dmax - Dmin) & 1) != 0) {
        n = 0;
        return false;
    }

    const std::uint64_t n64 =
        static_cast<std::uint64_t>((Dmax - Dmin) / 2) + 1;
    if (n64 == 0 || n64 > capacity || n64 > 0xffffffffULL) {
        n = 0;
        return false;
    }
    n = static_cast<std::uint32_t>(n64);

    const double j2 = half_spin_device(Dj2);
    const double j3 = half_spin_device(Dj3);
    const double j5 = half_spin_device(Dj5);
    const double j6 = half_spin_device(Dj6);
    const double f2 = j2 * (j2 + 1.0);
    const double f3 = j3 * (j3 + 1.0);
    const double f5 = j5 * (j5 + 1.0);
    const double f6 = j6 * (j6 + 1.0);
    const bool zero_branch = Dmin == 0;

    for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const int D = Dmin + 2 * static_cast<int>(i);
        const double j = half_spin_device(D);
        const double q = j * (j + 1.0);
        const double two_j_plus_one = static_cast<double>(D + 1);

        if (zero_branch) {
            shared.Y_base[i] =
                two_j_plus_one * (-q + 2.0 * (f2 + f5));
            shared.Y_f4[i] = -2.0 * two_j_plus_one;
            shared.coupling[i] = zero_branch_z_device(
                Dj2, Dj3, Dj5, Dj6, D);
        } else {
            const double Y1_base = f2 + f3 + f5 + f6;
            const double Y0_base = (f5 - f6) * (f2 - f3);
            shared.Y_base[i] = two_j_plus_one *
                fma(q, Y1_base - q, Y0_base);
            shared.Y_f4[i] = two_j_plus_one * (-2.0 * q);
            shared.coupling[i] = coupled_pair_factor_device(
                Dj2, Dj3, Dj5, Dj6, D);
        }
    }
    __syncthreads();
    return true;
}

__device__ bool solve_prepared_series(
    int Dmin,
    std::uint32_t n,
    int Dj2,
    int Dj3,
    int Dj4,
    int Dj5,
    int Dj6,
    std::uint32_t lane,
    std::uint32_t lanes,
    const PreparedShared& shared,
    double& final_scale) noexcept
{
    double* const values = shared.values;
#define B4GPU_RECURRENCE_VALUE(i) \
    values[static_cast<std::size_t>(i) * lanes + lane]
#define B4GPU_RECURRENCE_Y(i) \
    fma(f4, shared.Y_f4[(i)], shared.Y_base[(i)])
#define B4GPU_RECURRENCE_X(i) \
    prepared_x_device(Dmin, zero_branch, (i), shared)
#define B4GPU_RECURRENCE_Z(i) \
    prepared_z_device(Dmin, zero_branch, (i), shared)

    const std::int64_t Dsum = static_cast<std::int64_t>(Dj2) + Dj3 + Dj5 + Dj6;
    double endpoint_phase = 0.0;
    if (!even_phase_device(Dsum, endpoint_phase)) return false;

    if (n == 1) {
        const double denominator = static_cast<double>(Dmin + 1) *
                                   static_cast<double>(Dj4 + 1);
        if (!(denominator > 0.0) || !isfinite(denominator)) return false;
        B4GPU_RECURRENCE_VALUE(0) = endpoint_phase / sqrt(denominator);
        final_scale = 1.0;
        return isfinite(B4GPU_RECURRENCE_VALUE(0));
    }

    const double j4 = half_spin_device(Dj4);
    const double f4 = j4 * (j4 + 1.0);
    const double tiny = sqrt(sqrt(DBL_MIN));
    const bool zero_branch = Dmin == 0;

    // Lower non-classical tail. Store only ratios with magnitude at most one;
    // lower_seed is the first value that will be anchored at unity.
    std::uint32_t lower_seed = 0;
    const double ymin = B4GPU_RECURRENCE_Y(0);
    if (fabs(ymin) >= tiny) {
        double sn = -B4GPU_RECURRENCE_X(0) / ymin;
        std::uint32_t i = 0;
        while (isfinite(sn) && fabs(sn) <= 1.0) {
            B4GPU_RECURRENCE_VALUE(i) = sn;
            lower_seed = i + 1;
            if (lower_seed >= n - 1) break;
            i = lower_seed;
            const double denominator = fma(
                B4GPU_RECURRENCE_Z(i), sn, B4GPU_RECURRENCE_Y(i));
            if (fabs(denominator) < tiny) break;
            sn = -B4GPU_RECURRENCE_X(i) / denominator;
        }
    }

    // Upper non-classical tail. `n` means that no stable upper ratio was
    // accepted and the ordinary recurrence will run to the endpoint.
    std::uint32_t upper_ratio_begin = n;
    const std::uint32_t last = n - 1;
    const double ymax = B4GPU_RECURRENCE_Y(last);
    if (fabs(ymax) >= tiny) {
        double rn = -B4GPU_RECURRENCE_Z(last) / ymax;
        std::uint32_t i = last;
        while (isfinite(rn) && fabs(rn) <= 1.0) {
            B4GPU_RECURRENCE_VALUE(i) = rn;
            upper_ratio_begin = i;
            if (i == 0) break;
            --i;
            if (i < lower_seed || i == 0) break;
            const double denominator = fma(
                B4GPU_RECURRENCE_X(i), rn, B4GPU_RECURRENCE_Y(i));
            if (fabs(denominator) < tiny) break;
            rn = -B4GPU_RECURRENCE_Z(i) / denominator;
        }
    }

    // WIGSGLL/Luscombe-Luban reconstruction: anchor once after the lower
    // tail, advance through the classical region, then attach the stored upper
    // ratios to the value that arrives there.  This avoids independently
    // anchoring both tails, a potentially tiny bridge division, and an O(n)
    // bridge-rescaling pass.
    B4GPU_RECURRENCE_VALUE(lower_seed) = 1.0;
    for (std::uint32_t k = lower_seed; k > 0; --k) {
        B4GPU_RECURRENCE_VALUE(k - 1) *= B4GPU_RECURRENCE_VALUE(k);
    }

    std::uint32_t k = lower_seed;
    if (k == 0) {
        const double X0 = B4GPU_RECURRENCE_X(0);
        if (fabs(X0) < tiny) return false;
        B4GPU_RECURRENCE_VALUE(1) =
            (-B4GPU_RECURRENCE_Y(0) * B4GPU_RECURRENCE_VALUE(0)) / X0;
        k = 1;
    }
    for (; k + 1 < upper_ratio_begin && k + 1 < n; ++k) {
        const double Xk = B4GPU_RECURRENCE_X(k);
        if (fabs(Xk) < tiny) return false;
        const double numerator = fma(
            B4GPU_RECURRENCE_Y(k), B4GPU_RECURRENCE_VALUE(k),
            B4GPU_RECURRENCE_Z(k) * B4GPU_RECURRENCE_VALUE(k - 1));
        B4GPU_RECURRENCE_VALUE(k + 1) = -numerator / Xk;
    }
    for (; k + 1 < n; ++k) {
        B4GPU_RECURRENCE_VALUE(k + 1) *= B4GPU_RECURRENCE_VALUE(k);
    }

    double norm = 0.0;
    for (std::uint32_t i = 0; i < n; ++i) {
        const int D = Dmin + 2 * static_cast<int>(i);
        const double v = B4GPU_RECURRENCE_VALUE(i);
        if (!isfinite(v)) return false;
        norm = fma(static_cast<double>(D + 1), v * v, norm);
    }
    norm *= static_cast<double>(Dj4 + 1);
    if (!(norm > 0.0) || !isfinite(norm)) return false;

    int parity = static_cast<int>((Dsum / 2) & 1);
    // A forbidden endpoint may underflow to -0; its sign still fixes the row.
    if (signbit(B4GPU_RECURRENCE_VALUE(n - 1))) parity ^= 1;
    final_scale = (parity != 0 ? -1.0 : 1.0) / sqrt(norm);
    const bool finite = isfinite(final_scale);
#undef B4GPU_RECURRENCE_VALUE
#undef B4GPU_RECURRENCE_Y
#undef B4GPU_RECURRENCE_X
#undef B4GPU_RECURRENCE_Z
    return finite;
}

__device__ __forceinline__ PreparedShared shared_layout(
    std::uint32_t capacity,
    std::uint32_t lanes) noexcept
{
    extern __shared__ double storage[];
    PreparedShared shared{};
    shared.coupling = storage;
    shared.Y_base = shared.coupling + capacity;
    shared.Y_f4 = shared.Y_base + capacity;
    shared.values = shared.Y_f4 + capacity;
    (void)lanes;
    return shared;
}

__global__ void generate_sjA_kernel(
    DeviceCoherentRecurrenceProblem problem,
    std::uint64_t x_index,
    double* output,
    std::uint64_t* failure_count,
    RecurrenceLaunchConfig config)
{
    const std::uint64_t p_index = blockIdx.x;
    const int Dx = problem.x[x_index];
    const int Dl25 = problem.Dj25 + 2 * static_cast<int>(p_index);
    if (((problem.Dj15 + Dl25) & 1) != ((Dx + problem.Dj14) & 1)) return;

    const int Dmin = imax_device(
        iabs_device(problem.Dj15 - Dl25), iabs_device(Dx - problem.Dj14));
    const int Dmax = imin_device(
        problem.Dj15 + Dl25, Dx + problem.Dj14);
    const AxisIntersection virtual_index = intersect_regular_axis(
        problem.k5, problem.dimensions.k5, Dmin, Dmax);
    if (virtual_index.begin >= virtual_index.end) return;

    const PreparedShared shared = shared_layout(config.capacity, config.lanes);
    std::uint32_t n = 0;
    if (!prepare_coefficients(
            Dmin, Dmax, problem.Dj15, Dl25, Dx, problem.Dj14,
            config.capacity, shared, n)) {
        if (threadIdx.x == 0) record_failure(failure_count);
        return;
    }

    const std::uint64_t A = problem.dimensions.a;
    const std::uint64_t V = problem.dimensions.k5;
    for (std::uint64_t batch = 0; batch < A; batch += config.lanes) {
        const std::uint32_t lane = threadIdx.x;
        const std::uint64_t a_index = batch + lane;
        if (lane < config.lanes && a_index < A) {
            const int Di1 = problem.i1[a_index];
            if (triangle_device(Di1, problem.Dj15, problem.Dj14) &&
                triangle_device(Di1, Dx, Dl25)) {
                double scale = 0.0;
                if (solve_prepared_series(
                        Dmin, n, problem.Dj15, Dl25, Di1, Dx, problem.Dj14,
                        lane, config.lanes, shared, scale)) {
                    const double common =
                        sqrt(static_cast<double>(Di1 + 1)) *
                        static_cast<double>(Dx + 1);
                    for (std::uint64_t v_index = virtual_index.begin;
                         v_index < virtual_index.end; ++v_index) {
                        const int Dk5 = problem.k5[v_index];
                        double phase = 0.0;
                        if (!even_phase_device(
                                static_cast<std::int64_t>(Di1) + Dk5 + Dl25 +
                                    problem.Dj15 -
                                    doubled_sum_parity_device(
                                        problem.Dj14 + problem.Dj15),
                                phase)) {
                            record_failure(failure_count);
                            break;
                        }
                        const std::uint32_t recurrence_index =
                            static_cast<std::uint32_t>((Dk5 - Dmin) / 2);
                        const double raw = shared.values[
                            static_cast<std::size_t>(recurrence_index) *
                                config.lanes + lane];
                        output[a_index + A * (v_index + V * p_index)] =
                            phase * common * sqrt(static_cast<double>(Dk5 + 1)) *
                            scale * raw;
                    }
                } else {
                    record_failure(failure_count);
                }
            }
        }
        __syncthreads();
    }
}

__global__ void generate_sjB_kernel(
    DeviceCoherentRecurrenceProblem problem,
    std::uint64_t x_index,
    double* output,
    std::uint64_t* failure_count,
    RecurrenceLaunchConfig config)
{
    const std::uint64_t S = problem.dimensions.l35;
    const std::uint64_t st = blockIdx.x;
    const std::uint64_t s_index = st % S;
    const std::uint64_t t_index = st / S;
    const int Dx = problem.x[x_index];
    const int Dl35 = problem.Dj35 + 2 * static_cast<int>(s_index);
    const int Dl45 = problem.Dj45 + 2 * static_cast<int>(t_index);
    if (((Dx + problem.Dj14) & 1) != ((Dl45 + Dl35) & 1)) return;

    const int Dmin = imax_device(
        iabs_device(Dx - problem.Dj14), iabs_device(Dl45 - Dl35));
    const int Dmax = imin_device(Dx + problem.Dj14, Dl45 + Dl35);
    const AxisIntersection virtual_index = intersect_regular_axis(
        problem.k5, problem.dimensions.k5, Dmin, Dmax);
    if (virtual_index.begin >= virtual_index.end) return;

    const PreparedShared shared = shared_layout(config.capacity, config.lanes);
    std::uint32_t n = 0;
    if (!prepare_coefficients(
            Dmin, Dmax, Dx, problem.Dj14, Dl45, Dl35,
            config.capacity, shared, n)) {
        if (threadIdx.x == 0) record_failure(failure_count);
        return;
    }

    const std::uint64_t V = problem.dimensions.k5;
    const std::uint64_t J = problem.dimensions.k4;
    for (std::uint64_t batch = 0; batch < J; batch += config.lanes) {
        const std::uint32_t lane = threadIdx.x;
        const std::uint64_t j_index = batch + lane;
        if (lane < config.lanes && j_index < J) {
            const int Dk4 = problem.k4[j_index];
            if (triangle_device(Dk4, Dx, Dl35) &&
                triangle_device(Dk4, Dl45, problem.Dj14)) {
                double scale = 0.0;
                if (solve_prepared_series(
                        Dmin, n, Dx, problem.Dj14, Dk4, Dl45, Dl35,
                        lane, config.lanes, shared, scale)) {
                    double phase = 0.0;
                    if (!even_phase_device(
                            static_cast<std::int64_t>(Dk4) + Dl45 +
                                problem.Dj14 + Dl35 + problem.Dj35,
                            phase)) {
                        record_failure(failure_count);
                    } else {
                        const double common =
                            phase * sqrt(static_cast<double>(Dk4 + 1)) * scale;
                        for (std::uint64_t v_index = virtual_index.begin;
                             v_index < virtual_index.end; ++v_index) {
                            const int Dk5 = problem.k5[v_index];
                            const std::uint32_t recurrence_index =
                                static_cast<std::uint32_t>((Dk5 - Dmin) / 2);
                            const double raw = shared.values[
                                static_cast<std::size_t>(recurrence_index) *
                                    config.lanes + lane];
                            output[v_index + V *
                                (j_index + J * (s_index + S * t_index))] =
                                common * raw;
                        }
                    }
                } else {
                    record_failure(failure_count);
                }
            }
        }
        __syncthreads();
    }
}

// Fused full-tensor W1 producer. One 256-thread block owns one
// (l35,l45,l25,k4-batch) tuple. The first warp solves the virtual_index recurrences;
// the whole block then writes the virtual_index-fast cuTENSOR operand.
//
//   W1[k5,l45,i5,k4,l25,l35] = sjB[k5,k4,l35,l45]
//                                * B5[k5,l45,l35,l25,i5].
//
// Cooperative output is important: k5 is the contiguous coordinate, so
// adjacent lanes write adjacent doubles instead of each recurrence lane
// scattering a private k5 string through a 50-MiB operand.
__global__ void generate_sjB_times_B5_operand_kernel(
    DeviceCoherentRecurrenceProblem problem,
    std::uint64_t x_index,
    const double* __restrict__ B5,
    std::uint64_t I5,
    double* __restrict__ W1,
    std::uint64_t* failure_count,
    RecurrenceLaunchConfig config)
{
    const std::uint64_t S = problem.dimensions.l35;
    const std::uint64_t T = problem.dimensions.l45;
    const std::uint64_t J = problem.dimensions.k4;
    const std::uint64_t P = problem.dimensions.l25;
    const std::uint64_t j_batches =
        J / config.lanes + (J % config.lanes != 0 ? 1u : 0u);

    // Split the fanout over l25 as well as the k4 recurrence batch.  The
    // recurrence is consequently recomputed P times, but it is tiny compared
    // with writing the multi-million-value W1 operand.  This exposes hundreds
    // of 256-thread blocks instead of leaving half an A100 idle behind 49
    // one-warp blocks.
    std::uint64_t block = blockIdx.x;
    const std::uint64_t p_index = block % P; block /= P;
    const std::uint64_t j_batch = block % j_batches; block /= j_batches;
    const std::uint64_t s_index = block % S;
    const std::uint64_t t_index = block / S;
    const std::uint64_t batch = j_batch * config.lanes;

    const int Dx = problem.x[x_index];
    const int Dl35 = problem.Dj35 + 2 * static_cast<int>(s_index);
    const int Dl45 = problem.Dj45 + 2 * static_cast<int>(t_index);
    const int Dmin = imax_device(
        iabs_device(Dx - problem.Dj14), iabs_device(Dl45 - Dl35));
    const int Dmax = imin_device(Dx + problem.Dj14, Dl45 + Dl35);
    const bool parity_valid =
        ((Dx + problem.Dj14) & 1) == ((Dl45 + Dl35) & 1);
    AxisIntersection virtual_index{0, 0};
    if (parity_valid && Dmin <= Dmax) {
        virtual_index = intersect_regular_axis(
            problem.k5, problem.dimensions.k5, Dmin, Dmax);
    }

    const PreparedShared shared = shared_layout(config.capacity, config.lanes);
    std::uint32_t n = 0;
    bool coefficients_ready = false;
    if (virtual_index.begin < virtual_index.end) {
        coefficients_ready = prepare_coefficients(
            Dmin, Dmax, Dx, problem.Dj14, Dl45, Dl35,
            config.capacity, shared, n);
        if (!coefficients_ready && threadIdx.x == 0) {
            record_failure(failure_count);
        }
    }

    const std::uint64_t V = problem.dimensions.k5;
    const std::uint64_t batch_count =
        (J - batch) < config.lanes ? (J - batch) : config.lanes;

    int lane_valid = 0;
    if (coefficients_ready && threadIdx.x < batch_count) {
        const std::uint32_t lane = threadIdx.x;
        const std::uint64_t j_index = batch + lane;
        const int Dk4 = problem.k4[j_index];
        if (triangle_device(Dk4, Dx, Dl35) &&
            triangle_device(Dk4, Dl45, problem.Dj14)) {
            double scale = 0.0;
            if (solve_prepared_series(
                    Dmin, n, Dx, problem.Dj14, Dk4, Dl45, Dl35,
                    lane, config.lanes, shared, scale)) {
                double phase = 0.0;
                if (!even_phase_device(
                        static_cast<std::int64_t>(Dk4) + Dl45 +
                            problem.Dj14 + Dl35 + problem.Dj35,
                        phase)) {
                    record_failure(failure_count);
                } else {
                    const double common =
                        phase * sqrt(static_cast<double>(Dk4 + 1)) * scale;
                    for (std::uint64_t v_index = virtual_index.begin;
                         v_index < virtual_index.end; ++v_index) {
                        const int Dk5 = problem.k5[v_index];
                        const std::uint32_t recurrence_index =
                            static_cast<std::uint32_t>((Dk5 - Dmin) / 2);
                        shared.values[
                            static_cast<std::size_t>(recurrence_index) *
                                config.lanes + lane] *= common;
                    }
                    lane_valid = 1;
                }
            } else {
                record_failure(failure_count);
            }
        }
    }
    // All recurrence lanes must finish before publishing validity to the
    // extra shared-memory lane table reserved by the fused launch family.
    __syncthreads();
    double* const lane_flags = shared.values +
        static_cast<std::size_t>(config.capacity) * config.lanes;
    if (threadIdx.x < config.lanes) {
        lane_flags[threadIdx.x] = lane_valid != 0 ? 1.0 : 0.0;
    }
    __syncthreads();

    // Every logical operand element is written exactly once, including
    // structural zeros.  That removes the preceding full-buffer memset.
    const std::uint64_t output_count = V * I5 * batch_count;
    for (std::uint64_t linear = threadIdx.x; linear < output_count;
         linear += blockDim.x) {
        std::uint64_t item = linear;
        const std::uint64_t v_index = item % V; item /= V;
        const std::uint64_t i5_index = item % I5; item /= I5;
        const std::uint64_t j_local = item;
        const std::uint64_t j_index = batch + j_local;

        double value = 0.0;
        if (lane_flags[j_local] != 0.0 &&
            v_index >= virtual_index.begin && v_index < virtual_index.end) {
            const int Dk5 = problem.k5[v_index];
            const std::uint32_t recurrence_index =
                static_cast<std::uint32_t>((Dk5 - Dmin) / 2);
            const double sjB = shared.values[
                static_cast<std::size_t>(recurrence_index) * config.lanes +
                j_local];
            const std::uint64_t k5_offset = v_index + V *
                (t_index + T *
                (s_index + S * (p_index + P * i5_index)));
            value = sjB * B5[k5_offset];
        }
        const std::uint64_t w1_offset = v_index + V *
            (t_index + T *
            (i5_index + I5 *
            (j_index + J * (p_index + P * s_index))));
        W1[w1_offset] = value;
    }
}

__global__ void generate_sjD_kernel(
    DeviceCoherentRecurrenceProblem problem,
    std::uint64_t x_index,
    double* output,
    std::uint64_t* failure_count,
    RecurrenceLaunchConfig config)
{
    const std::uint64_t U = problem.dimensions.l23;
    const std::uint64_t uq = blockIdx.x;
    const std::uint64_t u_index = uq % U;
    const std::uint64_t q_index = uq / U;
    const int Dx = problem.x[x_index];
    const int Dl23 = problem.Dj23 + 2 * static_cast<int>(u_index);
    const int Dl24 = problem.Dj24 + 2 * static_cast<int>(q_index);
    if (((Dl23 + Dl24) & 1) != ((Dx + problem.Dj13) & 1)) return;

    const int Dmin = imax_device(
        iabs_device(Dl23 - Dl24), iabs_device(Dx - problem.Dj13));
    const int Dmax = imin_device(Dl23 + Dl24, Dx + problem.Dj13);
    const AxisIntersection virtual_index = intersect_regular_axis(
        problem.k2, problem.dimensions.k2, Dmin, Dmax);
    if (virtual_index.begin >= virtual_index.end) return;

    const PreparedShared shared = shared_layout(config.capacity, config.lanes);
    std::uint32_t n = 0;
    if (!prepare_coefficients(
            Dmin, Dmax, Dl23, Dl24, Dx, problem.Dj13,
            config.capacity, shared, n)) {
        if (threadIdx.x == 0) record_failure(failure_count);
        return;
    }

    const std::uint64_t I = problem.dimensions.k3;
    const std::uint64_t K = problem.dimensions.k2;
    for (std::uint64_t batch = 0; batch < I; batch += config.lanes) {
        const std::uint32_t lane = threadIdx.x;
        const std::uint64_t i_index = batch + lane;
        if (lane < config.lanes && i_index < I) {
            const int Dk3 = problem.k3[i_index];
            if (triangle_device(Dk3, Dl23, problem.Dj13) &&
                triangle_device(Dk3, Dx, Dl24)) {
                double scale = 0.0;
                if (solve_prepared_series(
                        Dmin, n, Dl23, Dl24, Dk3, Dx, problem.Dj13,
                        lane, config.lanes, shared, scale)) {
                    const double sqrt_k3 = sqrt(static_cast<double>(Dk3 + 1));
                    for (std::uint64_t k_index = virtual_index.begin;
                         k_index < virtual_index.end; ++k_index) {
                        const int Dk2 = problem.k2[k_index];
                        double phase = 0.0;
                        if (!even_phase_device(
                                static_cast<std::int64_t>(Dk2) + Dk3 + Dl23 +
                                    problem.Dj13 +
                                    Dl24 + problem.Dj24 -
                                    doubled_sum_parity_device(
                                        problem.Dj23 + problem.Dj24),
                                phase)) {
                            record_failure(failure_count);
                            break;
                        }
                        const std::uint32_t recurrence_index =
                            static_cast<std::uint32_t>((Dk2 - Dmin) / 2);
                        const double raw = shared.values[
                            static_cast<std::size_t>(recurrence_index) *
                                config.lanes + lane];
                        output[i_index + I *
                            (k_index + K * (u_index + U * q_index))] =
                            phase * sqrt(static_cast<double>(Dk2 + 1)) *
                            sqrt_k3 * scale * raw;
                    }
                } else {
                    record_failure(failure_count);
                }
            }
        }
        __syncthreads();
    }
}

// Fused full-tensor W3 producer.  One block owns one (l23,l24) pair.  The
// normalized sjD strings are completed in shared memory, then multiplied by
// the virtual_index-contiguous B2 tensor while the warp writes
//
//   W3[k2,l23,i2,k3,l25,l24]
//
// in the exact layout consumed by the binary cuTENSOR T3 contraction.
__global__ void generate_sjD_times_B2_operand_kernel(
    DeviceCoherentRecurrenceProblem problem,
    std::uint64_t x_index,
    const double* __restrict__ B2,
    std::uint64_t I2,
    double* __restrict__ W3,
    std::uint64_t* failure_count,
    RecurrenceLaunchConfig config)
{
    const std::uint64_t U = problem.dimensions.l23;
    const std::uint64_t Q = problem.dimensions.l24;
    const std::uint64_t I = problem.dimensions.k3;
    const std::uint64_t P = problem.dimensions.l25;
    const std::uint64_t i_batches =
        I / config.lanes + (I % config.lanes != 0 ? 1u : 0u);

    std::uint64_t block = blockIdx.x;
    const std::uint64_t p_index = block % P; block /= P;
    const std::uint64_t i_batch = block % i_batches; block /= i_batches;
    const std::uint64_t u_index = block % U;
    const std::uint64_t q_index = block / U;
    const std::uint64_t batch = i_batch * config.lanes;

    const int Dx = problem.x[x_index];
    const int Dl23 = problem.Dj23 + 2 * static_cast<int>(u_index);
    const int Dl24 = problem.Dj24 + 2 * static_cast<int>(q_index);
    const int Dmin = imax_device(
        iabs_device(Dl23 - Dl24), iabs_device(Dx - problem.Dj13));
    const int Dmax = imin_device(Dl23 + Dl24, Dx + problem.Dj13);
    const bool parity_valid =
        ((Dl23 + Dl24) & 1) == ((Dx + problem.Dj13) & 1);
    AxisIntersection virtual_index{0, 0};
    if (parity_valid && Dmin <= Dmax) {
        virtual_index = intersect_regular_axis(
            problem.k2, problem.dimensions.k2, Dmin, Dmax);
    }

    const PreparedShared shared = shared_layout(config.capacity, config.lanes);
    std::uint32_t n = 0;
    bool coefficients_ready = false;
    if (virtual_index.begin < virtual_index.end) {
        coefficients_ready = prepare_coefficients(
            Dmin, Dmax, Dl23, Dl24, Dx, problem.Dj13,
            config.capacity, shared, n);
        if (!coefficients_ready && threadIdx.x == 0) {
            record_failure(failure_count);
        }
    }

    const std::uint64_t K = problem.dimensions.k2;
    const std::uint64_t batch_count =
        (I - batch) < config.lanes ? (I - batch) : config.lanes;

    int lane_valid = 0;
    if (coefficients_ready && threadIdx.x < batch_count) {
        const std::uint32_t lane = threadIdx.x;
        const std::uint64_t i_index = batch + lane;
        const int Dk3 = problem.k3[i_index];
        if (triangle_device(Dk3, Dl23, problem.Dj13) &&
            triangle_device(Dk3, Dx, Dl24)) {
            double scale = 0.0;
            if (solve_prepared_series(
                    Dmin, n, Dl23, Dl24, Dk3, Dx, problem.Dj13,
                    lane, config.lanes, shared, scale)) {
                const double common =
                    sqrt(static_cast<double>(Dk3 + 1)) * scale;
                bool valid_phase = true;
                for (std::uint64_t k_index = virtual_index.begin;
                     k_index < virtual_index.end; ++k_index) {
                    const int Dk2 = problem.k2[k_index];
                    double phase = 0.0;
                    if (!even_phase_device(
                            static_cast<std::int64_t>(Dk2) + Dk3 + Dl23 +
                                problem.Dj13 + Dl24 + problem.Dj24 -
                                doubled_sum_parity_device(
                                    problem.Dj23 + problem.Dj24),
                            phase)) {
                        valid_phase = false;
                        record_failure(failure_count);
                        break;
                    }
                    const std::uint32_t recurrence_index =
                        static_cast<std::uint32_t>((Dk2 - Dmin) / 2);
                    shared.values[
                        static_cast<std::size_t>(recurrence_index) *
                            config.lanes + lane] *=
                        phase * sqrt(static_cast<double>(Dk2 + 1)) * common;
                }
                lane_valid = valid_phase ? 1 : 0;
            } else {
                record_failure(failure_count);
            }
        }
    }
    __syncthreads();
    double* const lane_flags = shared.values +
        static_cast<std::size_t>(config.capacity) * config.lanes;
    if (threadIdx.x < config.lanes) {
        lane_flags[threadIdx.x] = lane_valid != 0 ? 1.0 : 0.0;
    }
    __syncthreads();

    const std::uint64_t output_count = K * I2 * batch_count;
    for (std::uint64_t linear = threadIdx.x; linear < output_count;
         linear += blockDim.x) {
        std::uint64_t item = linear;
        const std::uint64_t k_index = item % K; item /= K;
        const std::uint64_t i2_index = item % I2; item /= I2;
        const std::uint64_t i_local = item;
        const std::uint64_t i_index = batch + i_local;

        double value = 0.0;
        if (lane_flags[i_local] != 0.0 &&
            k_index >= virtual_index.begin && k_index < virtual_index.end) {
            const int Dk2 = problem.k2[k_index];
            const std::uint32_t recurrence_index =
                static_cast<std::uint32_t>((Dk2 - Dmin) / 2);
            const double sjD = shared.values[
                static_cast<std::size_t>(recurrence_index) * config.lanes +
                i_local];
            const std::uint64_t k2_offset = k_index + K *
                (p_index + P *
                (q_index + Q * (u_index + U * i2_index)));
            value = sjD * B2[k2_offset];
        }
        const std::uint64_t w3_offset = k_index + K *
            (u_index + U *
            (i2_index + I2 *
            (i_index + I * (p_index + P * q_index))));
        W3[w3_offset] = value;
    }
}

__global__ void generate_sjE_kernel(
    DeviceCoherentRecurrenceProblem problem,
    std::uint64_t x_index,
    double* output,
    std::uint64_t* failure_count,
    RecurrenceLaunchConfig config)
{
    const std::uint64_t p_index = blockIdx.x;
    const int Dx = problem.x[x_index];
    const int Dl25 = problem.Dj25 + 2 * static_cast<int>(p_index);
    if (((problem.Dj12 + Dl25) & 1) != ((Dx + problem.Dj13) & 1)) return;

    const int Dmin = imax_device(
        iabs_device(problem.Dj12 - Dl25), iabs_device(Dx - problem.Dj13));
    const int Dmax = imin_device(
        problem.Dj12 + Dl25, Dx + problem.Dj13);
    const AxisIntersection virtual_index = intersect_regular_axis(
        problem.k2, problem.dimensions.k2, Dmin, Dmax);
    if (virtual_index.begin >= virtual_index.end) return;

    const PreparedShared shared = shared_layout(config.capacity, config.lanes);
    std::uint32_t n = 0;
    if (!prepare_coefficients(
            Dmin, Dmax, problem.Dj12, Dl25, Dx, problem.Dj13,
            config.capacity, shared, n)) {
        if (threadIdx.x == 0) record_failure(failure_count);
        return;
    }

    const int parity_a = doubled_sum_parity_device(
        problem.Dj14 + problem.Dj15);
    const int parity_d = doubled_sum_parity_device(
        problem.Dj23 + problem.Dj24);
    const std::int64_t DprefE =
        static_cast<std::int64_t>(problem.Dj12) - problem.Dj35 -
        problem.Dj34 - problem.Dj24;
    const int parity_e = doubled_sum_parity_device(DprefE);

    double prefE = 0.0;
    if (!even_phase_device(
            DprefE - parity_e,
            prefE)) {
        if (threadIdx.x == 0) record_failure(failure_count);
        return;
    }
    double phase_correction = 0.0;
    if (!even_phase_device(parity_a + parity_d + parity_e,
                           phase_correction)) {
        if (threadIdx.x == 0) record_failure(failure_count);
        return;
    }
    prefE *= phase_correction;

    const std::uint64_t A = problem.dimensions.a;
    const std::uint64_t K = problem.dimensions.k2;
    for (std::uint64_t batch = 0; batch < A; batch += config.lanes) {
        const std::uint32_t lane = threadIdx.x;
        const std::uint64_t a_index = batch + lane;
        if (lane < config.lanes && a_index < A) {
            const int Di1 = problem.i1[a_index];
            if (triangle_device(Di1, problem.Dj12, problem.Dj13) &&
                triangle_device(Di1, Dx, Dl25)) {
                double scale = 0.0;
                if (solve_prepared_series(
                        Dmin, n, problem.Dj12, Dl25, Di1, Dx, problem.Dj13,
                        lane, config.lanes, shared, scale)) {
                    for (std::uint64_t k_index = virtual_index.begin;
                         k_index < virtual_index.end; ++k_index) {
                        const int Dk2 = problem.k2[k_index];
                        const std::uint32_t recurrence_index =
                            static_cast<std::uint32_t>((Dk2 - Dmin) / 2);
                        const double raw = shared.values[
                            static_cast<std::size_t>(recurrence_index) *
                                config.lanes + lane];
                        output[a_index + A * (k_index + K * p_index)] =
                            prefE * scale * raw;
                    }
                } else {
                    record_failure(failure_count);
                }
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void generate_sjC_point(
    DeviceCoherentRecurrenceProblem problem,
    std::uint64_t x_index,
    std::uint64_t q_index,
    std::uint64_t r_index,
    std::uint64_t s_index,
    std::uint64_t output_index,
    std::uint64_t output_J,
    std::uint64_t output_I,
    double* output,
    std::uint64_t* failure_count,
    RecurrenceLaunchConfig config)
{
    const int Dx = problem.x[x_index];
    const int Dl24 = problem.Dj24 + 2 * static_cast<int>(q_index);
    const int Dl34 = problem.Dj34 + 2 * static_cast<int>(r_index);
    const int Dl35 = problem.Dj35 + 2 * static_cast<int>(s_index);
    if (((Dl34 + Dl35) & 1) != ((Dx + Dl24) & 1)) return;

    const int Dmin = imax_device(
        iabs_device(Dl34 - Dl35), iabs_device(Dx - Dl24));
    const int Dmax = imin_device(Dl34 + Dl35, Dx + Dl24);
    const AxisIntersection virtual_index = intersect_regular_axis(
        problem.k3, problem.dimensions.k3, Dmin, Dmax);
    if (virtual_index.begin >= virtual_index.end) return;

    const PreparedShared shared = shared_layout(config.capacity, config.lanes);
    std::uint32_t n = 0;
    if (!prepare_coefficients(
            Dmin, Dmax, Dl34, Dl35, Dx, Dl24,
            config.capacity, shared, n)) {
        if (threadIdx.x == 0) record_failure(failure_count);
        return;
    }

    double prefC = 0.0;
    if (!even_phase_device(
            static_cast<std::int64_t>(Dl34) + problem.Dj34, prefC)) {
        if (threadIdx.x == 0) record_failure(failure_count);
        return;
    }

    const std::uint64_t J = problem.dimensions.k4;
    for (std::uint64_t batch = 0; batch < J; batch += config.lanes) {
        const std::uint32_t lane = threadIdx.x;
        const std::uint64_t j_index = batch + lane;
        if (lane < config.lanes && j_index < J) {
            const int Dk4 = problem.k4[j_index];
            if (triangle_device(Dk4, Dl34, Dl24) &&
                triangle_device(Dk4, Dx, Dl35)) {
                double scale = 0.0;
                if (solve_prepared_series(
                        Dmin, n, Dl34, Dl35, Dk4, Dx, Dl24,
                        lane, config.lanes, shared, scale)) {
                    for (std::uint64_t i_index = virtual_index.begin;
                         i_index < virtual_index.end; ++i_index) {
                        const int Dk3 = problem.k3[i_index];
                        const std::uint32_t recurrence_index =
                            static_cast<std::uint32_t>((Dk3 - Dmin) / 2);
                        const double raw = shared.values[
                            static_cast<std::size_t>(recurrence_index) *
                                config.lanes + lane];
                        output[j_index + output_J *
                            (i_index + output_I * output_index)] =
                            prefC * scale * raw;
                    }
                } else {
                    record_failure(failure_count);
                }
            }
        }
        __syncthreads();
    }
}

__global__ void generate_sjC_qr_slab_kernel(
    DeviceCoherentRecurrenceProblem problem,
    std::uint64_t x_index,
    std::uint64_t q_index,
    std::uint64_t r_begin,
    std::uint64_t r_count,
    double* output,
    std::uint64_t* failure_count,
    RecurrenceLaunchConfig config)
{
    const std::uint64_t z = blockIdx.x;
    const std::uint64_t Z = r_count * problem.dimensions.l35;
    if (z >= Z) return;
    const std::uint64_t rr = z % r_count;
    const std::uint64_t s_index = z / r_count;
    generate_sjC_point(
        problem, x_index, q_index, r_begin + rr, s_index, z,
        problem.dimensions.k4, problem.dimensions.k3, output,
        failure_count, config);
}

bool checked_bytes(std::initializer_list<std::uint64_t> extents,
                   std::size_t& bytes) noexcept
{
    std::uint64_t count = 1;
    for (const std::uint64_t extent : extents) {
        if (extent != 0 &&
            count > std::numeric_limits<std::uint64_t>::max() / extent) {
            return false;
        }
        count *= extent;
    }
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        return false;
    }
    bytes = static_cast<std::size_t>(count) * sizeof(double);
    return true;
}

bool valid_problem(const DeviceCoherentRecurrenceProblem& problem,
                   std::uint64_t x_index,
                   const RecurrenceLaunchConfig& config) noexcept
{
    const DenseCoherentDimensions& d = problem.dimensions;
    return config.capacity != 0 && config.lanes != 0 && config.lanes <= 32 &&
           x_index < problem.x_count && problem.x != nullptr &&
           problem.i1 != nullptr && problem.k2 != nullptr &&
           problem.k3 != nullptr && problem.k4 != nullptr &&
           problem.k5 != nullptr && d.a != 0 && d.k2 != 0 && d.k3 != 0 &&
           d.k4 != 0 && d.k5 != 0 && d.l23 != 0 && d.l24 != 0 &&
           d.l25 != 0 && d.l34 != 0 && d.l35 != 0 && d.l45 != 0;
}

bool grid_extent(std::uint64_t count, unsigned& grid) noexcept
{
    if (count == 0 || count > static_cast<std::uint64_t>(
                                  std::numeric_limits<int>::max())) {
        return false;
    }
    grid = static_cast<unsigned>(count);
    return true;
}

bool checked_product(std::uint64_t lhs,
                     std::uint64_t rhs,
                     std::uint64_t& product) noexcept
{
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        return false;
    }
    product = lhs * rhs;
    return true;
}

std::size_t recurrence_shared_bytes(const RecurrenceLaunchConfig& config) noexcept
{
    const std::uint64_t arrays = static_cast<std::uint64_t>(config.lanes) + 3;
    if (config.capacity != 0 &&
        arrays > std::numeric_limits<std::uint64_t>::max() / config.capacity) {
        return 0;
    }
    const std::uint64_t doubles = arrays * config.capacity;
    if (doubles > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        return 0;
    }
    return static_cast<std::size_t>(doubles) * sizeof(double);
}

std::size_t family_shared_bytes(
    const RecurrenceLaunchConfig& config,
    RecurrenceKernelFamily family) noexcept
{
    const std::size_t recurrence_bytes = recurrence_shared_bytes(config);
    if (recurrence_bytes == 0) return 0;
    if (family != RecurrenceKernelFamily::fused_full_tensor_operands) {
        return recurrence_bytes;
    }
    const std::size_t flags =
        static_cast<std::size_t>(config.lanes) * sizeof(double);
    if (recurrence_bytes > std::numeric_limits<std::size_t>::max() - flags) {
        return 0;
    }
    return recurrence_bytes + flags;
}

cudaError_t family_occupancy(RecurrenceKernelFamily family,
                             std::size_t shared_bytes,
                             int& blocks) noexcept
{
    blocks = 0;
    if (family == RecurrenceKernelFamily::c_tile) {
        return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, generate_sjC_qr_slab_kernel,
            recurrence_block_threads, shared_bytes);
    }

    if (family == RecurrenceKernelFamily::fused_full_tensor_operands) {
        int w1_blocks = 0;
        int w3_blocks = 0;
        cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &w1_blocks, generate_sjB_times_B5_operand_kernel,
            fused_operand_block_threads, shared_bytes);
        if (status != cudaSuccess) return status;
        status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &w3_blocks, generate_sjD_times_B2_operand_kernel,
            fused_operand_block_threads, shared_bytes);
        if (status != cudaSuccess) return status;
        blocks = w1_blocks < w3_blocks ? w1_blocks : w3_blocks;
        return cudaSuccess;
    }

    int family_minimum = std::numeric_limits<int>::max();
    int current = 0;
    cudaError_t status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &current, generate_sjA_kernel, recurrence_block_threads, shared_bytes);
    if (status != cudaSuccess) return status;
    family_minimum = current < family_minimum ? current : family_minimum;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &current, generate_sjB_kernel, recurrence_block_threads, shared_bytes);
    if (status != cudaSuccess) return status;
    family_minimum = current < family_minimum ? current : family_minimum;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &current, generate_sjD_kernel, recurrence_block_threads, shared_bytes);
    if (status != cudaSuccess) return status;
    family_minimum = current < family_minimum ? current : family_minimum;
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &current, generate_sjE_kernel, recurrence_block_threads, shared_bytes);
    if (status != cudaSuccess) return status;
    blocks = current < family_minimum ? current : family_minimum;
    return cudaSuccess;
}

cudaError_t set_family_dynamic_shared_limit(RecurrenceKernelFamily family,
                                            int bytes) noexcept
{
    if (family == RecurrenceKernelFamily::c_tile) {
        return cudaFuncSetAttribute(
            generate_sjC_qr_slab_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    }
    if (family == RecurrenceKernelFamily::fused_full_tensor_operands) {
        cudaError_t status = cudaFuncSetAttribute(
            generate_sjB_times_B5_operand_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
        if (status != cudaSuccess) return status;
        return cudaFuncSetAttribute(
            generate_sjD_times_B2_operand_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    }
    cudaError_t status = cudaFuncSetAttribute(
        generate_sjA_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        bytes);
    if (status != cudaSuccess) return status;
    status = cudaFuncSetAttribute(
        generate_sjB_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        bytes);
    if (status != cudaSuccess) return status;
    status = cudaFuncSetAttribute(
        generate_sjD_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        bytes);
    if (status != cudaSuccess) return status;
    return cudaFuncSetAttribute(
        generate_sjE_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        bytes);
}

} // namespace

cudaError_t select_coherent_recurrence_launch_config(
    std::uint32_t capacity,
    std::uint32_t maximum_lanes,
    RecurrenceKernelFamily family,
    RecurrenceLaunchConfig* config,
    int* resident_blocks_per_sm) noexcept
{
    if (config == nullptr || capacity == 0 || maximum_lanes == 0) {
        return cudaErrorInvalidValue;
    }
    maximum_lanes = maximum_lanes > recurrence_block_threads
        ? recurrence_block_threads
        : maximum_lanes;
    *config = RecurrenceLaunchConfig{0, 0};
    if (resident_blocks_per_sm != nullptr) *resident_blocks_per_sm = 0;

    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) return status;
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) return status;

    const auto search = [&](std::uint64_t shared_limit) {
        std::uint64_t best_series = 0;
        int best_blocks = 0;
        std::uint32_t best_lanes = 0;
        for (std::uint32_t lanes = maximum_lanes; lanes != 0; --lanes) {
            const RecurrenceLaunchConfig candidate{capacity, lanes};
            const std::size_t shared_bytes = family_shared_bytes(candidate, family);
            if (shared_bytes == 0 || shared_bytes > shared_limit) continue;
            int blocks = 0;
            const cudaError_t occupancy_status =
                family_occupancy(family, shared_bytes, blocks);
            if (occupancy_status != cudaSuccess || blocks <= 0) continue;
            const std::uint64_t resident_series =
                static_cast<std::uint64_t>(blocks) * lanes;
            if (resident_series > best_series ||
                (resident_series == best_series && blocks > best_blocks) ||
                (resident_series == best_series && blocks == best_blocks &&
                 lanes > best_lanes)) {
                best_series = resident_series;
                best_blocks = blocks;
                best_lanes = lanes;
            }
        }
        if (best_lanes == 0) return false;
        *config = RecurrenceLaunchConfig{capacity, best_lanes};
        if (resident_blocks_per_sm != nullptr) {
            *resident_blocks_per_sm = best_blocks;
        }
        return true;
    };

    std::uint64_t shared_limit =
        static_cast<std::uint64_t>(properties.sharedMemPerBlock);
#if defined(CUDART_VERSION) && CUDART_VERSION >= 9000
    int optin_bytes = 0;
    status = cudaDeviceGetAttribute(
        &optin_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status == cudaSuccess && optin_bytes > 0 &&
        static_cast<std::uint64_t>(optin_bytes) > shared_limit) {
        const cudaError_t attribute_status =
            set_family_dynamic_shared_limit(family, optin_bytes);
        if (attribute_status == cudaSuccess) {
            shared_limit = static_cast<std::uint64_t>(optin_bytes);
        }
    }
#endif
    if (search(shared_limit)) return cudaSuccess;
    return cudaErrorInvalidConfiguration;
}

cudaError_t launch_generate_coherent_wings(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    double* sjA,
    double* sjB,
    double* sjD,
    double* sjE,
    std::uint64_t* device_failure_count,
    const RecurrenceLaunchConfig& config,
    cudaStream_t stream) noexcept
{
    cudaError_t status = launch_generate_coherent_common_wings(
        problem, x_index, sjB, sjD, device_failure_count, config, stream);
    if (status != cudaSuccess) return status;
    return launch_generate_coherent_boundary_wings(
        problem, x_index, sjA, sjE, device_failure_count, config, stream);
}

cudaError_t launch_generate_coherent_boundary_wings(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    double* sjA,
    double* sjE,
    std::uint64_t* device_failure_count,
    const RecurrenceLaunchConfig& config,
    cudaStream_t stream) noexcept
{
    if (!valid_problem(problem, x_index, config) || sjA == nullptr ||
        sjE == nullptr || device_failure_count == nullptr) {
        return cudaErrorInvalidValue;
    }
    const std::size_t shared_bytes = recurrence_shared_bytes(config);
    if (shared_bytes == 0) return cudaErrorInvalidValue;

    const DenseCoherentDimensions& d = problem.dimensions;
    std::size_t bytesA = 0, bytesE = 0;
    if (!checked_bytes({d.a, d.k5, d.l25}, bytesA) ||
        !checked_bytes({d.a, d.k2, d.l25}, bytesE)) {
        return cudaErrorInvalidValue;
    }
    cudaError_t status = cudaMemsetAsync(sjA, 0, bytesA, stream);
    if (status != cudaSuccess) return status;
    status = cudaMemsetAsync(sjE, 0, bytesE, stream);
    if (status != cudaSuccess) return status;

    unsigned gridA = 0, gridE = 0;
    if (!grid_extent(d.l25, gridA) || !grid_extent(d.l25, gridE)) {
        return cudaErrorInvalidValue;
    }
    generate_sjA_kernel<<<gridA, recurrence_block_threads, shared_bytes, stream>>>(
        problem, x_index, sjA, device_failure_count, config);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    generate_sjE_kernel<<<gridE, recurrence_block_threads, shared_bytes, stream>>>(
        problem, x_index, sjE, device_failure_count, config);
    return cudaGetLastError();
}

cudaError_t launch_generate_coherent_common_wings(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    double* sjB,
    double* sjD,
    std::uint64_t* device_failure_count,
    const RecurrenceLaunchConfig& config,
    cudaStream_t stream) noexcept
{
    if (!valid_problem(problem, x_index, config) || sjB == nullptr ||
        sjD == nullptr || device_failure_count == nullptr) {
        return cudaErrorInvalidValue;
    }
    const std::size_t shared_bytes = recurrence_shared_bytes(config);
    if (shared_bytes == 0) return cudaErrorInvalidValue;

    const DenseCoherentDimensions& d = problem.dimensions;
    std::size_t bytesB = 0, bytesD = 0;
    if (!checked_bytes({d.k5, d.k4, d.l35, d.l45}, bytesB) ||
        !checked_bytes({d.k3, d.k2, d.l23, d.l24}, bytesD)) {
        return cudaErrorInvalidValue;
    }
    cudaError_t status = cudaMemsetAsync(sjB, 0, bytesB, stream);
    if (status != cudaSuccess) return status;
    status = cudaMemsetAsync(sjD, 0, bytesD, stream);
    if (status != cudaSuccess) return status;

    std::uint64_t gridB_count = 0, gridD_count = 0;
    unsigned gridB = 0, gridD = 0;
    if (!checked_product(d.l35, d.l45, gridB_count) ||
        !grid_extent(gridB_count, gridB) ||
        !checked_product(d.l23, d.l24, gridD_count) ||
        !grid_extent(gridD_count, gridD)) {
        return cudaErrorInvalidValue;
    }
    generate_sjB_kernel<<<gridB, recurrence_block_threads, shared_bytes, stream>>>(
        problem, x_index, sjB, device_failure_count, config);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    generate_sjD_kernel<<<gridD, recurrence_block_threads, shared_bytes, stream>>>(
        problem, x_index, sjD, device_failure_count, config);
    return cudaGetLastError();
}

cudaError_t launch_generate_coherent_sjB_times_B5_operand(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    const double* dense_B5,
    std::uint64_t i5_count,
    double* W1,
    std::uint64_t* device_failure_count,
    const RecurrenceLaunchConfig& config,
    cudaStream_t stream) noexcept
{
    if (!valid_problem(problem, x_index, config) || dense_B5 == nullptr ||
        W1 == nullptr || device_failure_count == nullptr || i5_count == 0) {
        return cudaErrorInvalidValue;
    }
    const DenseCoherentDimensions& d = problem.dimensions;
    std::size_t bytes = 0;
    if (!checked_bytes(
            {d.k5, d.l45, i5_count, d.k4, d.l25, d.l35}, bytes)) {
        return cudaErrorInvalidValue;
    }
    (void)bytes; // the fused kernel writes structural zeros explicitly

    const std::uint64_t j_batches =
        d.k4 / config.lanes + (d.k4 % config.lanes != 0 ? 1u : 0u);
    std::uint64_t grid_count = 0, tmp = 0;
    unsigned grid = 0;
    const std::size_t shared_bytes = family_shared_bytes(
        config, RecurrenceKernelFamily::fused_full_tensor_operands);
    if (shared_bytes == 0 ||
        !checked_product(d.l35, d.l45, tmp) ||
        !checked_product(tmp, j_batches, tmp) ||
        !checked_product(tmp, d.l25, grid_count) ||
        !grid_extent(grid_count, grid)) {
        return cudaErrorInvalidValue;
    }
    generate_sjB_times_B5_operand_kernel<<<
        grid, fused_operand_block_threads, shared_bytes, stream>>>(
        problem, x_index, dense_B5, i5_count, W1,
        device_failure_count, config);
    return cudaGetLastError();
}

cudaError_t launch_generate_coherent_sjD_times_B2_operand(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    const double* dense_B2,
    std::uint64_t i2_count,
    double* W3,
    std::uint64_t* device_failure_count,
    const RecurrenceLaunchConfig& config,
    cudaStream_t stream) noexcept
{
    if (!valid_problem(problem, x_index, config) || dense_B2 == nullptr ||
        W3 == nullptr || device_failure_count == nullptr || i2_count == 0) {
        return cudaErrorInvalidValue;
    }
    const DenseCoherentDimensions& d = problem.dimensions;
    std::size_t bytes = 0;
    if (!checked_bytes(
            {d.k2, d.l23, i2_count, d.k3, d.l25, d.l24}, bytes)) {
        return cudaErrorInvalidValue;
    }
    (void)bytes; // the fused kernel writes structural zeros explicitly

    const std::uint64_t i_batches =
        d.k3 / config.lanes + (d.k3 % config.lanes != 0 ? 1u : 0u);
    std::uint64_t grid_count = 0, tmp = 0;
    unsigned grid = 0;
    const std::size_t shared_bytes = family_shared_bytes(
        config, RecurrenceKernelFamily::fused_full_tensor_operands);
    if (shared_bytes == 0 ||
        !checked_product(d.l23, d.l24, tmp) ||
        !checked_product(tmp, i_batches, tmp) ||
        !checked_product(tmp, d.l25, grid_count) ||
        !grid_extent(grid_count, grid)) {
        return cudaErrorInvalidValue;
    }
    generate_sjD_times_B2_operand_kernel<<<
        grid, fused_operand_block_threads, shared_bytes, stream>>>(
        problem, x_index, dense_B2, i2_count, W3,
        device_failure_count, config);
    return cudaGetLastError();
}

cudaError_t launch_generate_coherent_c_qr_slab(
    const DeviceCoherentRecurrenceProblem& problem,
    std::uint64_t x_index,
    std::uint64_t q_index,
    std::uint64_t r_begin,
    std::uint64_t r_count,
    double* sjC,
    std::uint64_t* device_failure_count,
    const RecurrenceLaunchConfig& config,
    cudaStream_t stream) noexcept
{
    if (!valid_problem(problem, x_index, config) || sjC == nullptr ||
        device_failure_count == nullptr) {
        return cudaErrorInvalidValue;
    }
    const DenseCoherentDimensions& d = problem.dimensions;
    if (q_index >= d.l24 || r_begin > d.l34 ||
        r_count > d.l34 - r_begin) {
        return cudaErrorInvalidValue;
    }
    if (r_count == 0) return cudaSuccess;
    std::uint64_t Z = 0;
    if (!checked_product(r_count, d.l35, Z)) return cudaErrorInvalidValue;

    std::size_t bytes = 0;
    if (!checked_bytes({d.k4, d.k3, Z}, bytes)) {
        return cudaErrorInvalidValue;
    }
    cudaError_t status = cudaMemsetAsync(sjC, 0, bytes, stream);
    if (status != cudaSuccess) return status;

    const std::size_t shared_bytes = recurrence_shared_bytes(config);
    unsigned grid = 0;
    if (shared_bytes == 0 || !grid_extent(Z, grid)) {
        return cudaErrorInvalidValue;
    }
    generate_sjC_qr_slab_kernel<<<
        grid, recurrence_block_threads, shared_bytes, stream>>>(
        problem, x_index, q_index, r_begin, r_count, sjC,
        device_failure_count, config);
    return cudaGetLastError();
}

} // namespace b4gpu::cuda_detail
