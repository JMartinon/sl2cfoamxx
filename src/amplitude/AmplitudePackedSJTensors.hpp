// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "amplitude/AmplitudeInputSupport.hpp"
#include "wigner/SixJTensors.hpp"


#include <iostream>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

namespace b4support {


// ============================================================================
// Packed value storage.
//
// std::vector<double>::resize(n) value-initializes newly exposed entries, i.e.
// it touches and zeroes the whole payload. For the packed 6j tensors this is a
// wasted full-memory pass: the fill kernels overwrite every packed slot.
//
// PackedDoubleBuffer exposes the tiny subset of std::vector used by the packed
// tensor code (.resize/.clear/.data/.size/.begin/.end), but resize() allocates
// uninitialized aligned storage and does not preserve old values. This is
// intentional. These buffers are scratch/output payloads, not semantic vectors.
// ============================================================================

class PackedDoubleBuffer {
public:
    using value_type = double;
    using size_type = std::size_t;

    PackedDoubleBuffer() noexcept = default;

    explicit PackedDoubleBuffer(size_type n) {
        resize(n);
    }

    PackedDoubleBuffer(const PackedDoubleBuffer& other) {
        resize(other.size_);
        if (size_ != 0) {
            std::memcpy(data_.get(), other.data_.get(), size_ * sizeof(double));
        }
    }

    PackedDoubleBuffer& operator=(const PackedDoubleBuffer& other) {
        if (this == &other) {
            return *this;
        }
        resize(other.size_);
        if (size_ != 0) {
            std::memcpy(data_.get(), other.data_.get(), size_ * sizeof(double));
        }
        return *this;
    }

    PackedDoubleBuffer(PackedDoubleBuffer&&) noexcept = default;
    PackedDoubleBuffer& operator=(PackedDoubleBuffer&&) noexcept = default;
    ~PackedDoubleBuffer() = default;

    void resize(size_type n) {
        // Unlike std::vector<double>::resize, this deliberately does not zero
        // initialize and does not preserve old payload values.
        if (n <= capacity_) {
            size_ = n;
            return;
        }

        data_.reset(allocate_raw(n));
        capacity_ = n;
        size_ = n;
    }

    void reserve(size_type n) {
        if (n <= capacity_) {
            return;
        }
        data_.reset(allocate_raw(n));
        capacity_ = n;
        if (size_ > capacity_) {
            size_ = capacity_;
        }
    }

    void clear() noexcept {
        // Match std::vector::clear semantics: keep capacity for reuse.
        size_ = 0;
    }

    void release() noexcept {
        data_.reset();
        size_ = 0;
        capacity_ = 0;
    }

    double* data() noexcept { return data_.get(); }
    const double* data() const noexcept { return data_.get(); }

    double* begin() noexcept { return data_.get(); }
    double* end() noexcept { return data_.get() + size_; }
    const double* begin() const noexcept { return data_.get(); }
    const double* end() const noexcept { return data_.get() + size_; }

    size_type size() const noexcept { return size_; }
    size_type capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }

    double& operator[](size_type i) noexcept { return data_.get()[i]; }
    const double& operator[](size_type i) const noexcept { return data_.get()[i]; }

    void fill(double value) {
        std::fill_n(data_.get(), size_, value);
    }

private:
    struct FreeDeleter {
        void operator()(double* p) const noexcept {
            std::free(p);
        }
    };

    using Ptr = std::unique_ptr<double, FreeDeleter>;

    static double* allocate_raw(size_type n) {
        if (n == 0) {
            return nullptr;
        }

        constexpr std::size_t alignment = 64;
        void* p = nullptr;
        std::size_t bytes = n * sizeof(double);
        bytes = (bytes + alignment - 1u) & ~(alignment - 1u);

        const int rc = ::posix_memalign(&p, alignment, bytes);
        if (rc != 0) {
            throw std::bad_alloc();
        }
        return static_cast<double*>(p);
    }

    Ptr data_{nullptr};
    size_type size_ = 0;
    size_type capacity_ = 0;
};

struct PackedAmplitudeSJTensors {
    PackedDoubleBuffer sjA;
    PackedDoubleBuffer sjB;
    PackedDoubleBuffer sjC;
    PackedDoubleBuffer sjD;
    PackedDoubleBuffer sjE;
    
    void allocate_from_support(const AmplitudeInputSupportPlan& support) {
        // PackedDoubleBuffer::resize allocates uninitialized storage by default.
        sjA.resize(static_cast<std::size_t>(support.sjA.packed_values));
        sjB.resize(static_cast<std::size_t>(support.sjB.packed_values));
        sjC.resize(static_cast<std::size_t>(support.sjC.packed_values));
        sjD.resize(static_cast<std::size_t>(support.sjD.packed_values));
        sjE.resize(static_cast<std::size_t>(support.sjE.packed_values));
    }
    
    void clear() {
        sjA.clear();
        sjB.clear();
        sjC.clear();
        sjD.clear();
        sjE.clear();
    }
    
    std::uint64_t payload_bytes() const noexcept {
        return static_cast<std::uint64_t>(
                                          (sjA.size() + sjB.size() + sjC.size() + sjD.size() + sjE.size())
                                          * sizeof(double)
                                          );
    }
    
    void print(std::ostream& os) const {
        const double mib = 1024.0 * 1024.0;
        
        os << "\nPacked amplitude sj tensors\n";
        os << "===========================\n";
        os << "sjA values = " << sjA.size()
        << " | " << static_cast<double>(sjA.size() * sizeof(double)) / mib
        << " MiB\n";
        os << "sjB values = " << sjB.size()
        << " | " << static_cast<double>(sjB.size() * sizeof(double)) / mib
        << " MiB\n";
        os << "sjC values = " << sjC.size()
        << " | " << static_cast<double>(sjC.size() * sizeof(double)) / mib
        << " MiB\n";
        os << "sjD values = " << sjD.size()
        << " | " << static_cast<double>(sjD.size() * sizeof(double)) / mib
        << " MiB\n";
        os << "sjE values = " << sjE.size()
        << " | " << static_cast<double>(sjE.size() * sizeof(double)) / mib
        << " MiB\n";
        os << "total      = "
        << static_cast<double>(payload_bytes()) / mib
        << " MiB\n";
    }
};

inline void packed_store_1d(
                            const RangeBlockSupport1D& support,
                            std::size_t block,
                            std::size_t inner_index,
                            double* values,
                            double value)
{
    const Range1D& r = support.blocks[block];
    
#ifndef NDEBUG
    if (!(inner_index >= r.begin && inner_index < r.end)) {
        std::cerr << "packed_store_1d: index outside support for "
        << support.name << "\n"
        << "block = " << block
        << ", inner = " << inner_index
        << ", range = [" << r.begin << "," << r.end << ")\n";
        std::abort();
    }
#endif
    
    values[r.offset + (inner_index - r.begin)] = value;
}

inline void packed_store_2d(
                            const RangeBlockSupport2D& support,
                            std::size_t block,
                            std::size_t i_index,
                            std::size_t j_index,
                            double* values,
                            double value)
{
    const Range2D& r = support.blocks[block];
    
#ifndef NDEBUG
    if (!(i_index >= r.i_begin && i_index < r.i_end &&
          j_index >= r.j_begin && j_index < r.j_end))
    {
        std::cerr << "packed_store_2d: index outside support for "
        << support.name << "\n"
        << "block = " << block
        << ", i = " << i_index
        << ", j = " << j_index
        << ", i-range = [" << r.i_begin << "," << r.i_end << ")"
        << ", j-range = [" << r.j_begin << "," << r.j_end << ")\n";
        std::abort();
    }
#endif
    
    const std::size_t ni =
    static_cast<std::size_t>(r.i_end - r.i_begin);
    
    const std::size_t local_i = i_index - r.i_begin;
    const std::size_t local_j = j_index - r.j_begin;
    
    values[r.offset + local_i + ni * local_j] = value;
}


inline int max_value_in_vector(const std::vector<int>& v) {
    if (v.empty()) {
        return 0;
    }
    
    return *std::max_element(v.begin(), v.end());
}

inline std::vector<int> make_value_to_index_map(
                                                const std::vector<int>& values,
                                                int max_value)
{
    std::vector<int> map(static_cast<std::size_t>(max_value + 1), -1);
    
    for (std::size_t i = 0; i < values.size(); ++i) {
        const int v = values[i];
        
        if (v >= 0 && v <= max_value) {
            map[static_cast<std::size_t>(v)] = static_cast<int>(i);
        }
    }
    
    return map;
}

} // namespace b4support


