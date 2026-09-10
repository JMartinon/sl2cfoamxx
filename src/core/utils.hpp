// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "core/Complex64.hpp"

// ============================================================================
// utils.hpp - high-performance complex scalar / complex tensor utilities
// ============================================================================
// Design goals:
//   - no std::complex in hot paths;
//   - scalar complex values are tiny PODs;
//   - complex work buffers use SoA storage: re[] and im[];
//   - storage is 64-byte aligned and can be allocated uninitialized;
//   - kernels expose raw double* pointers with restrict for vectorization.
//
// Recommended usage:
//   - use Complex64 for scalar temporaries and final amplitudes;
//   - use ComplexBuffer for large intermediates;
//   - use raw pointers inside contraction kernels.
// ============================================================================

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef B4UTILS_CACHELINE_BYTES
#define B4UTILS_CACHELINE_BYTES 64
#endif

#ifndef B4UTILS_FORCE_INLINE
#  if defined(__GNUC__) || defined(__clang__)
#    define B4UTILS_FORCE_INLINE inline __attribute__((always_inline))
#  elif defined(_MSC_VER)
#    define B4UTILS_FORCE_INLINE __forceinline
#  else
#    define B4UTILS_FORCE_INLINE inline
#  endif
#endif

namespace b4utils {

// ============================================================================
// Basic alignment helpers
// ============================================================================

constexpr std::size_t default_alignment = B4UTILS_CACHELINE_BYTES;

B4UTILS_FORCE_INLINE constexpr std::size_t round_up_to_multiple(std::size_t n,
                                                                 std::size_t a) noexcept
{
    return ((n + a - 1) / a) * a;
}

template <class T>
B4UTILS_FORCE_INLINE constexpr std::size_t padded_element_count(std::size_t n,
                                                                std::size_t alignment = default_alignment) noexcept
{
    const std::size_t elems_per_align = alignment / sizeof(T);
    if constexpr (sizeof(T) >= default_alignment) {
        return n;
    } else {
        return round_up_to_multiple(n, elems_per_align == 0 ? 1 : elems_per_align);
    }
}

inline void* aligned_malloc_bytes(std::size_t bytes,
                                  std::size_t alignment = default_alignment)
{
    if (bytes == 0) return nullptr;
    bytes = round_up_to_multiple(bytes, alignment);

#if defined(_MSC_VER)
    void* ptr = _aligned_malloc(bytes, alignment);
    if (!ptr) throw std::bad_alloc();
    return ptr;
#else
    void* ptr = nullptr;
    const int err = ::posix_memalign(&ptr, alignment, bytes);
    if (err != 0 || ptr == nullptr) throw std::bad_alloc();
    return ptr;
#endif
}

inline void aligned_free(void* ptr) noexcept
{
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

template <class T>
B4UTILS_FORCE_INLINE T* assume_aligned(T* p,
                                       std::size_t alignment = default_alignment) noexcept
{
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<T*>(__builtin_assume_aligned(p, alignment));
#else
    (void)alignment;
    return p;
#endif
}

template <class T>
B4UTILS_FORCE_INLINE const T* assume_aligned(const T* p,
                                             std::size_t alignment = default_alignment) noexcept
{
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<const T*>(__builtin_assume_aligned(p, alignment));
#else
    (void)alignment;
    return p;
#endif
}

// ============================================================================
// Move-only aligned raw buffer.
//
// Important: resize_discard() does NOT preserve old values. That is intentional.
// This is for contraction workspaces and packed tensors, not general-purpose
// containers whose resize operation preserves existing values.
// ============================================================================

template <class T, std::size_t Alignment = default_alignment>
class AlignedBuffer {
    static_assert(std::is_trivially_copyable_v<T>,
                  "AlignedBuffer is intended for trivially copyable numeric types");

public:
    AlignedBuffer() = default;

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept
    {
        move_from(std::move(other));
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept
    {
        if (this != &other) {
            release();
            move_from(std::move(other));
        }
        return *this;
    }

    ~AlignedBuffer() { release(); }

    void release() noexcept
    {
        aligned_free(data_);
        data_ = nullptr;
        size_ = 0;
        capacity_ = 0;
    }

    // Allocate enough storage for n elements, discard previous contents.
    void resize_discard(std::size_t n)
    {
        if (n <= capacity_) {
            size_ = n;
            return;
        }

        release();
        const std::size_t padded_n = padded_element_count<T>(n, Alignment);
        data_ = static_cast<T*>(aligned_malloc_bytes(padded_n * sizeof(T), Alignment));
        size_ = n;
        capacity_ = padded_n;
    }

    void zero() noexcept
    {
        if (data_ == nullptr || size_ == 0) {
            return;
        }

        const std::size_t byte_count = size_ * sizeof(T);
#ifdef _OPENMP
        constexpr std::size_t parallel_threshold_bytes =
            32ULL * 1024ULL * 1024ULL;
        constexpr std::size_t minimum_bytes_per_thread =
            8ULL * 1024ULL * 1024ULL;

        // A contiguous static partition gives large shared buffers portable
        // first-touch placement when the application binds its OpenMP team.
        // Avoid nested teams and cap workers so small chunks do not turn a
        // bandwidth operation into parallel-region overhead.
        if (byte_count >= parallel_threshold_bytes && !omp_in_parallel()) {
            const int maximum_threads = omp_get_max_threads();
            if (maximum_threads > 1) {
                const std::size_t wanted_threads =
                    1 + (byte_count - 1) / minimum_bytes_per_thread;
                const std::size_t thread_count_size =
                    wanted_threads < static_cast<std::size_t>(maximum_threads)
                    ? wanted_threads
                    : static_cast<std::size_t>(maximum_threads);
                const int thread_count = static_cast<int>(thread_count_size);
                if (thread_count > 1) {
#pragma omp parallel num_threads(thread_count)
                    {
                        const std::size_t thread =
                            static_cast<std::size_t>(omp_get_thread_num());
                        const std::size_t threads =
                            static_cast<std::size_t>(omp_get_num_threads());
                        const std::size_t base = size_ / threads;
                        const std::size_t remainder = size_ % threads;
                        const std::size_t begin =
                            thread * base +
                            (thread < remainder ? thread : remainder);
                        const std::size_t count =
                            base + (thread < remainder ? 1u : 0u);
                        std::memset(data_ + begin, 0, count * sizeof(T));
                    }
                    return;
                }
            }
        }
#endif

        std::memset(data_, 0, byte_count);
    }

    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    T* aligned_data() noexcept { return assume_aligned(data_, Alignment); }
    const T* aligned_data() const noexcept { return assume_aligned(data_, Alignment); }

    std::size_t size() const noexcept { return size_; }
    std::size_t bytes() const noexcept { return size_ * sizeof(T); }
    std::size_t capacity_bytes() const noexcept { return capacity_ * sizeof(T); }

private:
    void move_from(AlignedBuffer&& other) noexcept
    {
        data_ = other.data_;
        size_ = other.size_;
        capacity_ = other.capacity_;
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    T* data_ = nullptr;
    std::size_t size_ = 0;
    std::size_t capacity_ = 0;
};

// Scalar helpers. These are deliberately tiny free functions/operators.
// They are fine for scalar coefficient code and debugging. In hot tensor loops,
// prefer explicit raw-pointer loops, where the compiler sees plain double
// streams and can vectorize them directly.

B4UTILS_FORCE_INLINE constexpr Complex64 cmul(Complex64 a, Complex64 b) noexcept
{
    return Complex64{a.re * b.re - a.im * b.im,
                     a.re * b.im + a.im * b.re};
}

B4UTILS_FORCE_INLINE constexpr Complex64 cscale(Complex64 z, double x) noexcept
{
    return Complex64{x * z.re, x * z.im};
}

B4UTILS_FORCE_INLINE constexpr Complex64 cconj(Complex64 a) noexcept
{
    return Complex64{a.re, -a.im};
}

B4UTILS_FORCE_INLINE constexpr double norm2(Complex64 z) noexcept
{
    return z.re * z.re + z.im * z.im;
}

B4UTILS_FORCE_INLINE double abs(Complex64 z) noexcept
{
    return std::sqrt(norm2(z));
}

B4UTILS_FORCE_INLINE double arg(Complex64 z) noexcept
{
    return std::atan2(z.im, z.re);
}

B4UTILS_FORCE_INLINE constexpr Complex64 cdiv(Complex64 a, Complex64 b) noexcept
{
    const double d = norm2(b);
    return Complex64{(a.re * b.re + a.im * b.im) / d,
                     (a.im * b.re - a.re * b.im) / d};
}

// Lightweight scalar operators. These do not change the layout/triviality of
// Complex64 because they are free functions. Use them where readability matters;
// use the raw kernels below in performance-critical loops.
B4UTILS_FORCE_INLINE constexpr Complex64 operator+(Complex64 a, Complex64 b) noexcept
{
    return Complex64{a.re + b.re, a.im + b.im};
}

B4UTILS_FORCE_INLINE constexpr Complex64 operator-(Complex64 a, Complex64 b) noexcept
{
    return Complex64{a.re - b.re, a.im - b.im};
}

B4UTILS_FORCE_INLINE constexpr Complex64 operator-(Complex64 a) noexcept
{
    return Complex64{-a.re, -a.im};
}

B4UTILS_FORCE_INLINE constexpr Complex64 operator*(Complex64 a, Complex64 b) noexcept
{
    return cmul(a, b);
}

B4UTILS_FORCE_INLINE constexpr Complex64 operator*(double x, Complex64 z) noexcept
{
    return cscale(z, x);
}

B4UTILS_FORCE_INLINE constexpr Complex64 operator*(Complex64 z, double x) noexcept
{
    return cscale(z, x);
}

B4UTILS_FORCE_INLINE constexpr Complex64 operator/(Complex64 a, Complex64 b) noexcept
{
    return cdiv(a, b);
}

B4UTILS_FORCE_INLINE constexpr Complex64 operator/(Complex64 z, double x) noexcept
{
    return Complex64{z.re / x, z.im / x};
}

B4UTILS_FORCE_INLINE Complex64& operator+=(Complex64& a, Complex64 b) noexcept
{
    a.re += b.re;
    a.im += b.im;
    return a;
}

B4UTILS_FORCE_INLINE Complex64& operator-=(Complex64& a, Complex64 b) noexcept
{
    a.re -= b.re;
    a.im -= b.im;
    return a;
}

B4UTILS_FORCE_INLINE Complex64& operator*=(Complex64& a, Complex64 b) noexcept
{
    a = cmul(a, b);
    return a;
}

B4UTILS_FORCE_INLINE Complex64& operator*=(Complex64& z, double x) noexcept
{
    z.re *= x;
    z.im *= x;
    return z;
}

B4UTILS_FORCE_INLINE Complex64& operator/=(Complex64& a, Complex64 b) noexcept
{
    a = cdiv(a, b);
    return a;
}

B4UTILS_FORCE_INLINE Complex64& operator/=(Complex64& z, double x) noexcept
{
    z.re /= x;
    z.im /= x;
    return z;
}

B4UTILS_FORCE_INLINE constexpr bool operator==(Complex64 a, Complex64 b) noexcept
{
    return a.re == b.re && a.im == b.im;
}

B4UTILS_FORCE_INLINE constexpr bool operator!=(Complex64 a, Complex64 b) noexcept
{
    return !(a == b);
}

// ============================================================================
// Owning complex buffer: SoA re[] + im[].
// ============================================================================

template <std::size_t Alignment = default_alignment>
class ComplexBuffer {
public:
    ComplexBuffer() = default;

    ComplexBuffer(const ComplexBuffer&) = delete;
    ComplexBuffer& operator=(const ComplexBuffer&) = delete;
    ComplexBuffer(ComplexBuffer&&) noexcept = default;
    ComplexBuffer& operator=(ComplexBuffer&&) noexcept = default;

    void resize_discard(std::size_t n)
    {
        re_.resize_discard(n);
        im_.resize_discard(n);
    }

    void zero() noexcept
    {
        re_.zero();
        im_.zero();
    }

    void release() noexcept
    {
        re_.release();
        im_.release();
    }

    std::size_t size() const noexcept { return re_.size(); }
    std::size_t bytes() const noexcept { return re_.bytes() + im_.bytes(); }
    std::size_t capacity_bytes() const noexcept
    {
        return re_.capacity_bytes() + im_.capacity_bytes();
    }

    double* re_data() noexcept { return re_.data(); }
    double* im_data() noexcept { return im_.data(); }
    const double* re_data() const noexcept { return re_.data(); }
    const double* im_data() const noexcept { return im_.data(); }

private:
    AlignedBuffer<double, Alignment> re_;
    AlignedBuffer<double, Alignment> im_;
};

} // namespace b4utils
