// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include "core/OpenMp.hpp"
#endif

#include "boosters/B4_function.hpp"

namespace b4support {

// ============================================================================
// Basic range-block metadata
// ============================================================================

inline void print_support_memory(std::ostream& out, long double bytes)
{
    constexpr long double mib = 1024.0L * 1024.0L;
    constexpr long double gib = 1024.0L * mib;
    const long double value_mib = bytes / mib;

    if (value_mib > 1000.0L) {
        out << static_cast<double>(bytes / gib) << " GiB";
    } else {
        out << static_cast<double>(value_mib) << " MiB";
    }
}

struct Range1D {
    std::uint32_t begin = 0;
    std::uint32_t end   = 0;
    std::uint64_t offset = 0;
    
    std::uint32_t count() const noexcept {
        return end > begin ? end - begin : 0u;
    }
    
    bool empty() const noexcept {
        return begin >= end;
    }
};

struct Range2D {
    std::uint32_t i_begin = 0;
    std::uint32_t i_end   = 0;
    
    std::uint32_t j_begin = 0;
    std::uint32_t j_end   = 0;
    
    std::uint64_t offset = 0;
    
    std::uint64_t count() const noexcept {
        const std::uint64_t ni =
        i_end > i_begin ? static_cast<std::uint64_t>(i_end - i_begin) : 0ull;
        
        const std::uint64_t nj =
        j_end > j_begin ? static_cast<std::uint64_t>(j_end - j_begin) : 0ull;
        
        return ni * nj;
    }
    
    bool empty() const noexcept {
        return i_begin >= i_end || j_begin >= j_end;
    }
};



struct RangeBlockSupport1D {
    std::string name;
    
    // Outer block dimensions. The interpretation depends on the tensor.
    //
    // Example:
    //   sjB outer_dims = {Dim_x, Dim_k4, Dim_l35, Dim_l45}
    //   inner axis     = k5
    std::vector<std::size_t> outer_dims;
    
    std::size_t inner_dim = 0;
    
    std::vector<Range1D> blocks;
    
    // Number of doubles needed to store the packed tensor payload.
    std::uint64_t packed_values = 0;
    
    std::size_t num_blocks() const noexcept {
        return blocks.size();
    }
    
    std::uint64_t metadata_bytes() const noexcept {
        return static_cast<std::uint64_t>(blocks.size() * sizeof(Range1D));
    }
    
    std::uint64_t payload_bytes() const noexcept {
        return packed_values * static_cast<std::uint64_t>(sizeof(double));
    }
    
    std::uint64_t total_bytes() const noexcept {
        return metadata_bytes() + payload_bytes();
    }
    
    void print(std::ostream& os) const {
        std::uint64_t dense_values = inner_dim;
        for (std::size_t d : outer_dims) {
            dense_values *= d;
        }

        const long double dense_bytes =
            static_cast<long double>(dense_values) * sizeof(double);
        const long double packed_bytes =
            static_cast<long double>(payload_bytes());
        const long double meta_bytes =
            static_cast<long double>(metadata_bytes());

        const double frac =
            dense_values == 0
                ? 0.0
                : 100.0 * static_cast<double>(packed_values)
                    / static_cast<double>(dense_values);

        os << name << "\n";
        os << "  blocks        = " << blocks.size() << "\n";
        os << "  inner_dim     = " << inner_dim << "\n";
        os << "  dense values  = " << dense_values << "\n";
        os << "  packed values = " << packed_values
           << " (" << std::fixed << std::setprecision(4) << frac << " %)\n";

        os << "  dense payload = ";
        print_support_memory(os, dense_bytes);
        os << "\n";

        os << "  packed payload= ";
        print_support_memory(os, packed_bytes);
        os << "\n";

        os << "  metadata      = ";
        print_support_memory(os, meta_bytes);
        os << "\n";
    }
};

struct RangeBlockSupport2D {
    std::string name;
    
    // Example for sjC:
    //   outer_dims = {Dim_x, Dim_l24, Dim_l34, Dim_l35}
    //   inner_i    = k3
    //   inner_j    = k4
    std::vector<std::size_t> outer_dims;
    
    std::size_t inner_i_dim = 0;
    std::size_t inner_j_dim = 0;
    
    std::vector<Range2D> blocks;
    
    std::uint64_t packed_values = 0;
    
    std::size_t num_blocks() const noexcept {
        return blocks.size();
    }
    
    std::uint64_t metadata_bytes() const noexcept {
        return static_cast<std::uint64_t>(blocks.size() * sizeof(Range2D));
    }
    
    std::uint64_t payload_bytes() const noexcept {
        return packed_values * static_cast<std::uint64_t>(sizeof(double));
    }
    
    std::uint64_t total_bytes() const noexcept {
        return metadata_bytes() + payload_bytes();
    }
    
    void print(std::ostream& os) const {
        std::uint64_t dense_values = inner_i_dim * inner_j_dim;
        for (std::size_t d : outer_dims) {
            dense_values *= d;
        }

        const double frac =
            dense_values == 0
                ? 0.0
                : 100.0 * static_cast<double>(packed_values)
                    / static_cast<double>(dense_values);

        os << name << "\n";
        os << "  blocks        = " << blocks.size() << "\n";
        os << "  inner_i_dim   = " << inner_i_dim << "\n";
        os << "  inner_j_dim   = " << inner_j_dim << "\n";
        os << "  dense values  = " << dense_values << "\n";
        os << "  packed values = " << packed_values
           << " (" << std::fixed << std::setprecision(4) << frac << " %)\n";

        os << "  dense payload = ";
        print_support_memory(
            os, static_cast<long double>(dense_values) * sizeof(double));
        os << "\n";

        os << "  packed payload= ";
        print_support_memory(os, static_cast<long double>(payload_bytes()));
        os << "\n";

        os << "  metadata      = ";
        print_support_memory(os, static_cast<long double>(metadata_bytes()));
        os << "\n";
    }
};


struct AmplitudeInputSupportPlan {
    AmplitudeChannels channels;
    
    // Lifted ell axes, stored as doubled-spin values.
    std::vector<int> l23;
    std::vector<int> l24;
    std::vector<int> l25;
    std::vector<int> l34;
    std::vector<int> l35;
    std::vector<int> l45;
    
    // 6j tensor supports.
    RangeBlockSupport1D sjA; // block (x,l25)          -> k5 range
    RangeBlockSupport1D sjB; // block (x,k4,l35,l45)   -> k5 range
    RangeBlockSupport2D sjC; // block (x,l24,l34,l35)  -> k3 x k4 rectangle
    RangeBlockSupport1D sjD; // block (x,k3,l23,l24)   -> k2 range
    RangeBlockSupport1D sjE; // block (x,l25)          -> k2 range
    
    // Booster tensor supports.
    RangeBlockSupport1D B2;  // block (l23,l24,l25)    -> k2 range
    RangeBlockSupport1D B3;  // block (k3,l34,l35)     -> l23 range
    RangeBlockSupport1D B4;  // block (k4,l24,l34)     -> l45 range
    RangeBlockSupport1D B5;  // block (l25,l35,l45)    -> k5 range
    
    std::uint64_t total_payload_bytes() const noexcept {
        return sjA.payload_bytes()
        + sjB.payload_bytes()
        + sjC.payload_bytes()
        + sjD.payload_bytes()
        + sjE.payload_bytes()
        + B2.payload_bytes()
        + B3.payload_bytes()
        + B4.payload_bytes()
        + B5.payload_bytes();
    }
    
    std::uint64_t total_metadata_bytes() const noexcept {
        return sjA.metadata_bytes()
        + sjB.metadata_bytes()
        + sjC.metadata_bytes()
        + sjD.metadata_bytes()
        + sjE.metadata_bytes()
        + B2.metadata_bytes()
        + B3.metadata_bytes()
        + B4.metadata_bytes()
        + B5.metadata_bytes();
    }
    
    void print(std::ostream& os) const {
        os << "\nAmplitude input support plan\n";
        os << "============================\n\n";
        
        sjA.print(os); os << "\n";
        sjB.print(os); os << "\n";
        sjC.print(os); os << "\n";
        sjD.print(os); os << "\n";
        sjE.print(os); os << "\n";
        
        B2.print(os); os << "\n";
        B3.print(os); os << "\n";
        B4.print(os); os << "\n";
        B5.print(os); os << "\n";
        
        os << "Total packed payload = ";
        print_support_memory(os, static_cast<long double>(total_payload_bytes()));
        os << "\n";

        os << "Total metadata       = ";
        print_support_memory(os, static_cast<long double>(total_metadata_bytes()));
        os << "\n";
    }
};

// ============================================================================
// Helper functions
// ============================================================================

inline std::vector<int> make_l_axis(int Dj, int Dl) {
    std::vector<int> vals;
    vals.reserve(static_cast<std::size_t>(Dl + 1));
    
    for (int i = 0; i <= Dl; ++i) {
        vals.push_back(Dj + 2 * i);
    }
    
    return vals;
}

inline std::size_t product_dims(const std::vector<std::size_t>& dims) {
    std::size_t p = 1;
    
    for (std::size_t d : dims) {
        p *= d;
    }
    
    return p;
}

inline Range1D empty_range_1d() noexcept {
    return Range1D{0u, 0u, 0ull};
}

inline Range2D empty_range_2d() noexcept {
    return Range2D{0u, 0u, 0u, 0u, 0ull};
}

// Returns a range in axis-index space, not doubled-spin-value space.
//
// axis must be sorted and should contain one parity only, which is true for
// all b/x/l/a channels generated here.
inline bool axis_is_uniform(const std::vector<int>& axis) noexcept {
    if (axis.size() <= 2) {
        return true;
    }
    
    const int step = axis[1] - axis[0];
    
    for (std::size_t i = 2; i < axis.size(); ++i) {
        if (axis[i] - axis[i - 1] != step) {
            return false;
        }
    }
    
    return true;
}

inline Range1D axis_range_from_interval_binary(
                                               const std::vector<int>& axis,
                                               int lo,
                                               int hi,
                                               int required_parity)
{
    if (axis.empty() || lo > hi) {
        return empty_range_1d();
    }
    
    if ((axis.front() & 1) != (required_parity & 1)) {
        return empty_range_1d();
    }
    
    const auto it_begin =
    std::lower_bound(axis.begin(), axis.end(), lo);
    
    const auto it_end =
    std::upper_bound(axis.begin(), axis.end(), hi);
    
    const std::uint32_t begin =
    static_cast<std::uint32_t>(it_begin - axis.begin());
    
    const std::uint32_t end =
    static_cast<std::uint32_t>(it_end - axis.begin());
    
    if (begin >= end) {
        return empty_range_1d();
    }
    
    return Range1D{begin, end, 0ull};
}

inline Range1D axis_range_from_interval(
                                        const std::vector<int>& axis,
                                        int lo,
                                        int hi,
                                        int required_parity)
{
    if (axis.empty() || lo > hi) {
        return empty_range_1d();
    }
    
    if ((axis.front() & 1) != (required_parity & 1)) {
        return empty_range_1d();
    }
    
    // Fast path for the normal case:
    //   axis = first, first+2, first+4, ...
    //
    // This is true for all l-axes and should be true for tight b-channels.
    if (axis.size() == 1) {
        const int v = axis.front();
        
        if (v >= lo && v <= hi) {
            return Range1D{0u, 1u, 0ull};
        }
        
        return empty_range_1d();
    }
    
    const int first = axis.front();
    const int last  = axis.back();
    const int step  = axis[1] - axis[0];
    
    if (step <= 0 || !axis_is_uniform(axis)) {
        return axis_range_from_interval_binary(axis, lo, hi, required_parity);
    }
    
    const int lo_clamped = std::max(lo, first);
    const int hi_clamped = std::min(hi, last);
    
    if (lo_clamped > hi_clamped) {
        return empty_range_1d();
    }
    
    // Since parity was already checked and all values in the axis have the
    // same parity, we can convert directly.
    const int begin_i =
    (lo_clamped <= first)
    ? 0
    : (lo_clamped - first + step - 1) / step;
    
    const int end_i =
    (hi_clamped - first) / step + 1;
    
    if (begin_i >= end_i) {
        return empty_range_1d();
    }
    
    return Range1D{
        static_cast<std::uint32_t>(begin_i),
        static_cast<std::uint32_t>(end_i),
        0ull
    };
}

inline Range1D axis_range_for_triangle(
                                       const std::vector<int>& axis,
                                       int a,
                                       int b)
{
    const int lo = iabs_int(a - b);
    const int hi = a + b;
    const int required_parity = (a + b) & 1;
    
    return axis_range_from_interval(axis, lo, hi, required_parity);
}

inline Range1D axis_range_for_two_triangles(
                                            const std::vector<int>& axis,
                                            int a,
                                            int b,
                                            int c,
                                            int d)
{
    const int parity1 = (a + b) & 1;
    const int parity2 = (c + d) & 1;
    
    if (parity1 != parity2) {
        return empty_range_1d();
    }
    
    const int lo =
    std::max(iabs_int(a - b), iabs_int(c - d));
    
    const int hi =
    std::min(a + b, c + d);
    
    return axis_range_from_interval(axis, lo, hi, parity1);
}

void finalize_offsets(RangeBlockSupport1D& s);

void finalize_offsets(RangeBlockSupport2D& s);

inline RangeBlockSupport1D make_support_1d(
                                           const std::string& name,
                                           std::vector<std::size_t> outer_dims,
                                           std::size_t inner_dim)
{
    RangeBlockSupport1D s;
    s.name = name;
    s.outer_dims = std::move(outer_dims);
    s.inner_dim = inner_dim;
    s.blocks.resize(product_dims(s.outer_dims));
    return s;
}

inline RangeBlockSupport2D make_support_2d(
                                           const std::string& name,
                                           std::vector<std::size_t> outer_dims,
                                           std::size_t inner_i_dim,
                                           std::size_t inner_j_dim)
{
    RangeBlockSupport2D s;
    s.name = name;
    s.outer_dims = std::move(outer_dims);
    s.inner_i_dim = inner_i_dim;
    s.inner_j_dim = inner_j_dim;
    s.blocks.resize(product_dims(s.outer_dims));
    return s;
}

// ============================================================================
// Index helpers matching the block layouts used above.
// ============================================================================

inline std::size_t idx2(
                        std::size_t i0,
                        std::size_t i1,
                        std::size_t D1) noexcept
{
    return i0 * D1 + i1;
}

inline std::size_t idx3(
                        std::size_t i0,
                        std::size_t i1,
                        std::size_t i2,
                        std::size_t D1,
                        std::size_t D2) noexcept
{
    return (i0 * D1 + i1) * D2 + i2;
}

inline std::size_t idx4(
                        std::size_t i0,
                        std::size_t i1,
                        std::size_t i2,
                        std::size_t i3,
                        std::size_t D1,
                        std::size_t D2,
                        std::size_t D3) noexcept
{
    return ((i0 * D1 + i1) * D2 + i2) * D3 + i3;
}

// ============================================================================
// Main builder
// ============================================================================
//
// This builds triangle-only input supports.
//
// Important:
//   We deliberately ignore the extra booster symmetry/parity zero rules here.
//   Therefore B2/B3/B4/B5 supports may slightly overestimate the true nonzero
//   structure. That is intentional for now.
// ============================================================================

AmplitudeInputSupportPlan build_amplitude_input_support_plan(
                                                                    const B4Spins& s,
                                                                    int Dl,
                                                                    const AmplitudeChannels& AmC);


// ============================================================================
// Plan-A intermediate support
// ============================================================================
//
// Current contraction tree:
//
//   T1(l25,l45,k4,l35,x)
//      = sum_k5 sjA(k5,l25,x) sjB(k5,l45,k4,l35,x) B5(k5,l45,l25,l35)
//
//   T2(l25,k4,l24,l34,l35,x)
//      = sum_l45 T1(l25,l45,k4,l35,x) B4(l45,k4,l24,l34)
//
//   T3(l25,l23,k3,l24,x)
//      = sum_k2 sjE(k2,l25,x) sjD(k2,l23,k3,l24,x) B2(k2,l23,l25,l24)
//
//   T4(l25,k3,l24,l34,l35,x)
//      = sum_l23 T3(l25,l23,k3,l24,x) B3(l23,k3,l34,l35)
//
// We store each intermediate as range blocks over l25:
//
//   T1 block (x,l45,k4,l35)       -> l25 range
//   T2 block (x,k4,l24,l34,l35)   -> l25 range
//   T3 block (x,l23,k3,l24)       -> l25 range
//   T4 block (x,k3,l24,l34,l35)   -> l25 range
//
// ============================================================================


// ============================================================================
// Range utilities
// ============================================================================


inline bool ranges_intersect(const Range1D& a, const Range1D& b) noexcept {
    return std::max(a.begin, b.begin) < std::min(a.end, b.end);
}

inline bool ranges_intersect(
                             const Range1D& a,
                             const Range1D& b,
                             const Range1D& c) noexcept
{
    const std::uint32_t begin =
    std::max({a.begin, b.begin, c.begin});
    
    const std::uint32_t end =
    std::min({a.end, b.end, c.end});
    
    return begin < end;
}

struct CoverRangeBuilder {
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
    bool has_value = false;
    
    void add(std::size_t i) noexcept {
        const std::uint32_t u =
        static_cast<std::uint32_t>(i);
        
        if (!has_value) {
            begin = u;
            end = u + 1u;
            has_value = true;
            return;
        }
        
        begin = std::min(begin, u);
        end = std::max(end, u + 1u);
    }
    
    Range1D finish() const noexcept {
        if (!has_value) {
            return empty_range_1d();
        }
        
        return Range1D{begin, end, 0ull};
    }
};

enum class WingExecutionOrder {
    RightThenLeft, // Plan A: T1 -> T2, then T3 -> T4
    LeftThenRight  // Plan B: T3 -> T4, then T1 -> T2
};

inline const char* to_string(WingExecutionOrder order) noexcept {
    switch (order) {
        case WingExecutionOrder::RightThenLeft:
            return "Plan A: right wing first, left wing second";
        case WingExecutionOrder::LeftThenRight:
            return "Plan B: left wing first, right wing second";
    }
    
    return "Unknown";
}







} // namespace b4support
