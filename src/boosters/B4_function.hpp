// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

// contraction_example.cpp
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring> // for std::memcpy
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "wigner/ThreeJRecurrence.hpp"

#include <fstream>
#include <iomanip>
#include <string>

#include <cmath>
#include <complex>
#include <limits>
#include <initializer_list>

#include "wigner/ln_gamma.hpp"
// #include <quadmath.h>

using C = std::complex<double>;

// ----------------------------------------------------------------------------
// PUBLIC TENSOR-FILL WRAPPERS
// ----------------------------------------------------------------------------
struct B4Spins {
    int Dj12;
    int Dj13;
    int Dj14;
    int Dj15;
    
    int Dj23;
    int Dj24;
    int Dj25;
    
    int Dj34;
    int Dj35;
    int Dj45;
    
    explicit B4Spins(const std::array<int, 10>& j)
    : Dj12(j[0]),
    Dj13(j[1]),
    Dj14(j[2]),
    Dj15(j[3]),
    Dj23(j[4]),
    Dj24(j[5]),
    Dj25(j[6]),
    Dj34(j[7]),
    Dj35(j[8]),
    Dj45(j[9])
    {}
    
    int max() const {
        return std::max({
            Dj12, Dj13, Dj14, Dj15,
            Dj23, Dj24, Dj25,
            Dj34, Dj35,
            Dj45
        });
    }
};

// Triangle condition for doubled spins
inline bool triangle_condition(int a, int b, int c) {
    // Halve the inputs for the actual triangle condition
    return (a + b >= c) && (a + c >= b) && (b + c >= a) && ((a + b + c) % 2 == 0);
}

inline int iabs_int(int x) noexcept {
    return x < 0 ? -x : x;
}

inline double pm1_pow_int(int n) noexcept {
    return (iabs_int(n) & 1) ? -1.0 : 1.0;
}

inline double zsign_as_double(int Dnu1, int Dnu2, int DJ1, int DJ2) noexcept {
    return pm1_pow_int((Dnu1 + Dnu2 + DJ1 + DJ2) / 2);
}

inline double z_to_k_factor(int DJ1, int DJ2, int DJ, int Dnu1, int Dnu2) noexcept {
    return 1.0 /
    (
     zsign_as_double(Dnu1, Dnu2, DJ1, DJ2) *
     std::sqrt((DJ1 + 1.0) * (DJ2 + 1.0) * (DJ + 1.0))
     );
}

inline int find_or_push_int(std::vector<int>& xs, int x) {
    for (int i = 0; i < static_cast<int>(xs.size()); ++i) {
        if (xs[i] == x) return i;
    }
    
    xs.push_back(x);
    return static_cast<int>(xs.size()) - 1;
}

inline std::vector<int> get_valid_i(int Dj1, int Dj2, int Dj3, int Dj4) {
    int i_min = std::max(iabs_int(Dj1 - Dj2), iabs_int(Dj3 - Dj4));
    int i_max = std::min(Dj1 + Dj2, Dj3 + Dj4);
    int parity = (Dj1 + Dj2) & 1;
    
    std::vector<int> res;
    
    if (parity != ((Dj3 + Dj4) & 1)) return res;
    
    if ((i_min & 1) != parity) ++i_min;
    
    for (int i = i_min; i <= i_max; i += 2) {
        res.push_back(i);
    }
    
    return res;
}

struct DRange {
    int lo;
    int hi;
};

inline int min_abs_diff_between_doubled_ranges(DRange r1, DRange r2) noexcept {
    // Values in each DRange are assumed to advance by 2.
    //
    // If the integer intervals do not overlap, the minimum distance is the gap.
    // If they overlap:
    //   - same parity ranges can meet exactly -> 0
    //   - opposite parity ranges can get distance 1
    if (r1.hi < r2.lo) {
        return r2.lo - r1.hi;
    }
    
    if (r2.hi < r1.lo) {
        return r1.lo - r2.hi;
    }
    
    return ((r1.lo ^ r2.lo) & 1) ? 1 : 0;
}

inline std::vector<int> get_valid_k_tight(
                                          DRange r1,
                                          DRange r2,
                                          DRange r3,
                                          DRange r4)
{
    std::vector<int> res;
    
    const int parity12 = (r1.lo + r2.lo) & 1;
    const int parity34 = (r3.lo + r4.lo) & 1;
    
    if (parity12 != parity34) {
        return res;
    }
    
    int k_min = std::max(
                         min_abs_diff_between_doubled_ranges(r1, r2),
                         min_abs_diff_between_doubled_ranges(r3, r4)
                         );
    
    const int k_max = std::min(
                               r1.hi + r2.hi,
                               r3.hi + r4.hi
                               );
    
    if (k_min > k_max) {
        return res;
    }
    
    if ((k_min & 1) != parity12) {
        ++k_min;
    }
    
    res.reserve(static_cast<std::size_t>((k_max - k_min) / 2 + 1));
    
    for (int k = k_min; k <= k_max; k += 2) {
        res.push_back(k);
    }
    
    return res;
}

inline void mark_x_interval(
                            std::vector<uint8_t>& mask,
                            int x_min,
                            int x_max,
                            int parity)
{
    if (mask.empty()) return;
    
    const int max_valid_x =
    static_cast<int>(mask.size()) - 1;
    
    if (x_max < 0 || x_min > max_valid_x) {
        return;
    }
    
    x_min = std::max(x_min, 0);
    x_max = std::min(x_max, max_valid_x);
    
    if ((x_min & 1) != parity) {
        ++x_min;
    }
    
    for (int x = x_min; x <= x_max; x += 2) {
        mask[static_cast<std::size_t>(x)] = 1;
    }
}


inline bool exists_value_in_sorted_interval_parity(
                                                   const std::vector<int>& vals,
                                                   int lo,
                                                   int hi,
                                                   int parity)
{
    if (vals.empty() || lo > hi) {
        return false;
    }
    
    auto it = std::lower_bound(vals.begin(), vals.end(), lo);
    
    while (it != vals.end() && *it <= hi) {
        if (((*it) & 1) == parity) {
            return true;
        }
        ++it;
    }
    
    return false;
}

inline bool exists_value_for_two_triangles(
                                           const std::vector<int>& vals,
                                           int a,
                                           int b,
                                           int c,
                                           int d)
{
    // Need one value v such that:
    //
    //   triangle(v,a,b)
    //   triangle(v,c,d)
    //
    // Therefore:
    //
    //   max(|a-b|, |c-d|) <= v <= min(a+b, c+d)
    //
    // and both parity constraints must agree.
    
    const int parity1 = (a + b) & 1;
    const int parity2 = (c + d) & 1;
    
    if (parity1 != parity2) {
        return false;
    }
    
    const int lo = std::max(iabs_int(a - b), iabs_int(c - d));
    const int hi = std::min(a + b, c + d);
    
    return exists_value_in_sorted_interval_parity(vals, lo, hi, parity1);
}

inline bool x_supported_by_A_fast(
                                  int Dx,
                                  const B4Spins& spins,
                                  int Dl,
                                  const std::vector<int>& i1,
                                  const std::vector<int>& k5)
{
    const int Dj14 = spins.Dj14;
    const int Dj15 = spins.Dj15;
    
    const int Dl25_max = spins.Dj25 + 2 * Dl;
    
    for (int Dl25 = spins.Dj25; Dl25 <= Dl25_max; Dl25 += 2) {
        const bool has_i1 =
        exists_value_for_two_triangles(
                                       i1,
                                       Dj14, Dj15,   // triangle(i1,Dj14,Dj15)
                                       Dl25, Dx      // triangle(i1,Dl25,x)
                                       );
        
        if (!has_i1) {
            continue;
        }
        
        const bool has_k5 =
        exists_value_for_two_triangles(
                                       k5,
                                       Dl25, Dj15,   // triangle(k5,Dl25,Dj15)
                                       Dj14, Dx      // triangle(k5,Dj14,x)
                                       );
        
        if (has_k5) {
            return true;
        }
    }
    
    return false;
}

inline bool x_supported_by_B_fast(
                                  int Dx,
                                  const B4Spins& spins,
                                  int Dl,
                                  const std::vector<int>& k4,
                                  const std::vector<int>& k5)
{
    const int Dj14 = spins.Dj14;
    
    const int Dl35_max = spins.Dj35 + 2 * Dl;
    const int Dl45_max = spins.Dj45 + 2 * Dl;
    
    for (int Dl35 = spins.Dj35; Dl35 <= Dl35_max; Dl35 += 2) {
        for (int Dl45 = spins.Dj45; Dl45 <= Dl45_max; Dl45 += 2) {
            const bool has_k5 =
            exists_value_for_two_triangles(
                                           k5,
                                           Dl35, Dl45,  // triangle(k5,Dl35,Dl45)
                                           Dj14, Dx     // triangle(k5,Dj14,x)
                                           );
            
            if (!has_k5) {
                continue;
            }
            
            const bool has_k4 =
            exists_value_for_two_triangles(
                                           k4,
                                           Dj14, Dl45,  // triangle(k4,Dj14,Dl45)
                                           Dl35, Dx     // triangle(k4,Dl35,x)
                                           );
            
            if (has_k4) {
                return true;
            }
        }
    }
    
    return false;
}

inline bool x_supported_by_C_fast(
                                  int Dx,
                                  const B4Spins& spins,
                                  int Dl,
                                  const std::vector<int>& k3,
                                  const std::vector<int>& k4)
{
    const int Dl24_max = spins.Dj24 + 2 * Dl;
    const int Dl34_max = spins.Dj34 + 2 * Dl;
    const int Dl35_max = spins.Dj35 + 2 * Dl;
    
    for (int Dl35 = spins.Dj35; Dl35 <= Dl35_max; Dl35 += 2) {
        for (int Dl34 = spins.Dj34; Dl34 <= Dl34_max; Dl34 += 2) {
            for (int Dl24 = spins.Dj24; Dl24 <= Dl24_max; Dl24 += 2) {
                const bool has_k3 =
                exists_value_for_two_triangles(
                                               k3,
                                               Dl35, Dl34,  // triangle(k3,Dl35,Dl34)
                                               Dl24, Dx     // triangle(k3,Dl24,x)
                                               );
                
                if (!has_k3) {
                    continue;
                }
                
                const bool has_k4 =
                exists_value_for_two_triangles(
                                               k4,
                                               Dl24, Dl34,  // triangle(k4,Dl24,Dl34)
                                               Dl35, Dx     // triangle(k4,Dl35,x)
                                               );
                
                if (has_k4) {
                    return true;
                }
            }
        }
    }
    
    return false;
}

inline bool x_supported_by_D_fast(
                                  int Dx,
                                  const B4Spins& spins,
                                  int Dl,
                                  const std::vector<int>& k2,
                                  const std::vector<int>& k3)
{
    const int Dj13 = spins.Dj13;
    
    const int Dl23_max = spins.Dj23 + 2 * Dl;
    const int Dl24_max = spins.Dj24 + 2 * Dl;
    
    for (int Dl24 = spins.Dj24; Dl24 <= Dl24_max; Dl24 += 2) {
        for (int Dl23 = spins.Dj23; Dl23 <= Dl23_max; Dl23 += 2) {
            const bool has_k2 =
            exists_value_for_two_triangles(
                                           k2,
                                           Dl24, Dl23,  // triangle(k2,Dl24,Dl23)
                                           Dj13, Dx     // triangle(k2,Dj13,x)
                                           );
            
            if (!has_k2) {
                continue;
            }
            
            const bool has_k3 =
            exists_value_for_two_triangles(
                                           k3,
                                           Dj13, Dl23,  // triangle(k3,Dj13,Dl23)
                                           Dl24, Dx     // triangle(k3,Dl24,x)
                                           );
            
            if (has_k3) {
                return true;
            }
        }
    }
    
    return false;
}

inline bool x_supported_by_E_fast(
                                  int Dx,
                                  const B4Spins& spins,
                                  int Dl,
                                  const std::vector<int>& i1,
                                  const std::vector<int>& k2)
{
    const int Dj12 = spins.Dj12;
    const int Dj13 = spins.Dj13;
    
    const int Dl25_max = spins.Dj25 + 2 * Dl;
    
    for (int Dl25 = spins.Dj25; Dl25 <= Dl25_max; Dl25 += 2) {
        const bool has_i1 =
        exists_value_for_two_triangles(
                                       i1,
                                       Dj13, Dj12,  // triangle(i1,Dj13,Dj12)
                                       Dl25, Dx     // triangle(i1,Dl25,x)
                                       );
        
        if (!has_i1) {
            continue;
        }
        
        const bool has_k2 =
        exists_value_for_two_triangles(
                                       k2,
                                       Dl25, Dj12,  // triangle(k2,Dl25,Dj12)
                                       Dj13, Dx     // triangle(k2,Dj13,x)
                                       );
        
        if (has_k2) {
            return true;
        }
    }
    
    return false;
}

inline std::vector<int> make_tight_xvals(
                                         const B4Spins& spins,
                                         int Dl,
                                         const std::vector<int>& i1,
                                         const std::vector<int>& k2,
                                         const std::vector<int>& k3,
                                         const std::vector<int>& k4,
                                         const std::vector<int>& k5)
{
    std::vector<int> xvals;
    
    if (i1.empty() || k2.empty() || k3.empty() ||
        k4.empty() || k5.empty())
    {
        return xvals;
    }
    
    const int Dl24_max = spins.Dj24 + 2 * Dl;
    const int Dl25_max = spins.Dj25 + 2 * Dl;
    const int Dl35_max = spins.Dj35 + 2 * Dl;
    
    const int x_max = std::min({
        i1.back() + Dl25_max,
        k5.back() + spins.Dj14,
        k4.back() + Dl35_max,
        k3.back() + Dl24_max,
        k2.back() + spins.Dj13
    });
    
    if (x_max < 0) {
        return xvals;
    }
    
    xvals.reserve(static_cast<std::size_t>(x_max / 2 + 1));
    
    // Candidate parity is actually fixed if the whole network is nonzero.
    // Using all x remains safe and still cheap. But we can start from the
    // A/E parity to skip half immediately.
    const int x_parity = (i1.front() + spins.Dj25) & 1;
    
    for (int Dx = x_parity; Dx <= x_max; Dx += 2) {
        if (!x_supported_by_A_fast(Dx, spins, Dl, i1, k5)) {
            continue;
        }
        
        if (!x_supported_by_B_fast(Dx, spins, Dl, k4, k5)) {
            continue;
        }
        
        if (!x_supported_by_C_fast(Dx, spins, Dl, k3, k4)) {
            continue;
        }
        
        if (!x_supported_by_D_fast(Dx, spins, Dl, k2, k3)) {
            continue;
        }
        
        if (!x_supported_by_E_fast(Dx, spins, Dl, i1, k2)) {
            continue;
        }
        
        xvals.push_back(Dx);
    }
    
    return xvals;
}

struct FullTensorChannels {
    
    // Boundary intertwiner channels i1,...,i5.
    std::vector<int> i1;
    std::vector<int> i2;
    std::vector<int> i3;
    std::vector<int> i4;
    std::vector<int> i5;
    
    
    // Virtual intertwiner channels k2,...,k5.
    // k1 is fixed by i1 after gauge fixing.
    std::vector<int> k2;
    std::vector<int> k3;
    std::vector<int> k4;
    std::vector<int> k5;
    
    
    std::vector<int> xvals;
    
    
};

struct AmplitudeChannels {
    std::array<int, 5> i;
    
    std::vector<int> k2;
    std::vector<int> k3;
    std::vector<int> k4;
    std::vector<int> k5;
    
    std::vector<int> xvals;
};


// Channels for the coherent-boundary single-amplitude contraction where
// legs 2,3,4,5 have already been projected into coherent booster tensors and
// leg 1 is still summed as
//
//   A_coh = sum_{i1} C1[i1] A(i1).
//
// Unlike AmplitudeChannels, this is not a fixed-boundary intertwiner object.  The i1 axis is
// an active set, usually the nonzero iotas carried by C1, filtered through the
// physical intertwiner range.  The virtual intertwiner and x axes are the tight union axes
// needed by all active i1 values. It is a separate type so `fixed_intertwiners[0]` retains
// its single-channel meaning throughout the ordinary amplitude path.
struct CoherentAmplitudeChannels {
    std::vector<int> i1;

    // Position in the original C1 input array for each i1 entry above.  This is
    // useful when the C1 input order is not sorted.  Duplicate iotas are folded
    // by keeping the first position; callers that need duplicate accumulation
    // should combine coefficients before calling the contraction routine.
    std::vector<std::size_t> c1_positions;

    std::vector<int> k2;
    std::vector<int> k3;
    std::vector<int> k4;
    std::vector<int> k5;

    std::vector<int> xvals;

    bool empty() const noexcept {
        return i1.empty() || k2.empty() || k3.empty() || k4.empty() || k5.empty() || xvals.empty();
    }
};

inline CoherentAmplitudeChannels make_CoherentAmplitudeChannels(
                                                                const B4Spins& spins,
                                                                int Dl,
                                                                const std::vector<int>& C1_doubled_iotas)
{
    CoherentAmplitudeChannels ch;

    const std::vector<int> valid_i1 =
    get_valid_i(spins.Dj14, spins.Dj15,
                spins.Dj13, spins.Dj12);

    if (valid_i1.empty() || C1_doubled_iotas.empty()) {
        return ch;
    }

    // Build sorted active i1 = intersection(C1 iotas, physical valid i1).
    // Keep the first original C1 position for each active i1.
    std::vector<std::pair<int, std::size_t>> requested;
    requested.reserve(C1_doubled_iotas.size());
    for (std::size_t i = 0; i < C1_doubled_iotas.size(); ++i) {
        requested.push_back({C1_doubled_iotas[i], i});
    }
    std::stable_sort(requested.begin(), requested.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });

    int last_added = std::numeric_limits<int>::min();
    for (const auto& [Di1, pos] : requested) {
        if (Di1 == last_added) {
            continue;
        }
        if (std::binary_search(valid_i1.begin(), valid_i1.end(), Di1)) {
            ch.i1.push_back(Di1);
            ch.c1_positions.push_back(pos);
            last_added = Di1;
        }
    }

    if (ch.i1.empty()) {
        return ch;
    }

    const DRange r23{spins.Dj23, spins.Dj23 + 2 * Dl};
    const DRange r24{spins.Dj24, spins.Dj24 + 2 * Dl};
    const DRange r25{spins.Dj25, spins.Dj25 + 2 * Dl};

    const DRange r34{spins.Dj34, spins.Dj34 + 2 * Dl};
    const DRange r35{spins.Dj35, spins.Dj35 + 2 * Dl};
    const DRange r45{spins.Dj45, spins.Dj45 + 2 * Dl};

    const DRange j12{spins.Dj12, spins.Dj12};
    const DRange j13{spins.Dj13, spins.Dj13};
    const DRange j14{spins.Dj14, spins.Dj14};
    const DRange j15{spins.Dj15, spins.Dj15};

    // The coherent boosters B2..B5 have already summed over i2..i5.  Therefore
    // the virtual intertwiner ranges are the full tight ranges, not fixed-boundary intertwiner-pruned ranges.
    ch.k2 = get_valid_k_tight(r23, r24, r25, j12);
    ch.k3 = get_valid_k_tight(r35, r34, r23, j13);
    ch.k4 = get_valid_k_tight(r45, j14, r24, r34);
    ch.k5 = get_valid_k_tight(r25, j15, r35, r45);

    if (ch.k2.empty() || ch.k3.empty() || ch.k4.empty() || ch.k5.empty()) {
        ch.k2.clear(); ch.k3.clear(); ch.k4.clear(); ch.k5.clear(); ch.xvals.clear();
        return ch;
    }

    ch.xvals = make_tight_xvals(
        spins,
        Dl,
        ch.i1,
        ch.k2,
        ch.k3,
        ch.k4,
        ch.k5
    );

    if (ch.xvals.empty()) {
        ch.k2.clear(); ch.k3.clear(); ch.k4.clear(); ch.k5.clear();
    }

    return ch;
}

inline CoherentAmplitudeChannels make_CoherentAmplitudeChannels(
                                                                const B4Spins& spins,
                                                                int Dl,
                                                                std::initializer_list<int> C1_doubled_iotas)
{
    return make_CoherentAmplitudeChannels(spins, Dl, std::vector<int>(C1_doubled_iotas));
}

inline FullTensorChannels make_FullTensorChannels(
                                         const B4Spins& spins,
                                         int Dl)
{
    FullTensorChannels ch;
    
    
    ch.i1 = get_valid_i(spins.Dj14, spins.Dj15, spins.Dj13, spins.Dj12);
    
    ch.i2 = get_valid_i(spins.Dj23, spins.Dj24, spins.Dj25, spins.Dj12);
    
    ch.i3 = get_valid_i(spins.Dj34, spins.Dj35, spins.Dj13, spins.Dj23);
    
    ch.i4 = get_valid_i(spins.Dj45, spins.Dj14, spins.Dj24, spins.Dj34);
    
    ch.i5 = get_valid_i(spins.Dj15, spins.Dj25, spins.Dj35, spins.Dj45);
    
    
    const DRange r23{spins.Dj23, spins.Dj23 + 2 * Dl};
    const DRange r24{spins.Dj24, spins.Dj24 + 2 * Dl};
    const DRange r25{spins.Dj25, spins.Dj25 + 2 * Dl};
    
    const DRange r34{spins.Dj34, spins.Dj34 + 2 * Dl};
    const DRange r35{spins.Dj35, spins.Dj35 + 2 * Dl};
    const DRange r45{spins.Dj45, spins.Dj45 + 2 * Dl};
    
    const DRange j12{spins.Dj12, spins.Dj12};
    const DRange j13{spins.Dj13, spins.Dj13};
    const DRange j14{spins.Dj14, spins.Dj14};
    const DRange j15{spins.Dj15, spins.Dj15};
    
    
    ch.k2 = get_valid_k_tight(r23, r24, r25, j12);
    
    ch.k3 = get_valid_k_tight(r35, r34, r23, j13);
    
    ch.k4 = get_valid_k_tight(r45, j14, r24, r34);
    
    ch.k5 = get_valid_k_tight(r25, j15, r35, r45);
    
    
    if (!ch.i1.empty() && !ch.k2.empty() && !ch.k3.empty() && !ch.k4.empty() && !ch.k5.empty()) {
        
        ch.xvals = make_tight_xvals(
                                    spins,
                                    Dl,
                                    ch.i1,
                                    ch.k2,
                                    ch.k3,
                                    ch.k4,
                                    ch.k5
                                    );

    }
    
    return ch;
}


inline bool vector_contains_sorted(const std::vector<int>& v, int x) {
    return std::binary_search(v.begin(), v.end(), x);
}

inline AmplitudeChannels make_AmplitudeChannels(
                                                const B4Spins& spins,
                                                int Dl,
                                                const std::array<int, 5>& fixed_intertwiners)
{
    AmplitudeChannels ch;
    
    // Store the requested fixed boundary intertwiner channels.
    ch.i = fixed_intertwiners;
    
    // ---------------------------------------------------------------------
    // Build and validate the allowed boundary intertwiner channels.
    //
    // This is not optional. If one fixed boundary intertwiner is invalid, the amplitude is
    // identically zero, so virtual intertwiner/x support construction can be skipped.
    // ---------------------------------------------------------------------
    
    const std::vector<int> valid_i1 =
    get_valid_i(spins.Dj14, spins.Dj15,
                spins.Dj13, spins.Dj12);
    
    const std::vector<int> valid_i2 =
    get_valid_i(spins.Dj23, spins.Dj24,
                spins.Dj25, spins.Dj12);
    
    const std::vector<int> valid_i3 =
    get_valid_i(spins.Dj34, spins.Dj35,
                spins.Dj13, spins.Dj23);
    
    const std::vector<int> valid_i4 =
    get_valid_i(spins.Dj45, spins.Dj14,
                spins.Dj24, spins.Dj34);
    
    const std::vector<int> valid_i5 =
    get_valid_i(spins.Dj15, spins.Dj25,
                spins.Dj35, spins.Dj45);
    
    if (!vector_contains_sorted(valid_i1, fixed_intertwiners[0]) ||
        !vector_contains_sorted(valid_i2, fixed_intertwiners[1]) ||
        !vector_contains_sorted(valid_i3, fixed_intertwiners[2]) ||
        !vector_contains_sorted(valid_i4, fixed_intertwiners[3]) ||
        !vector_contains_sorted(valid_i5, fixed_intertwiners[4]))
    {
        // Leave b/x channels empty. Caller can interpret this as zero amplitude.
        return ch;
    }
    // ---------------------------------------------------------------------
    // Lifted-spin ranges.
    //
    // Dl counts ordinary spin increments, so doubled spins increase by 2*Dl.
    // ---------------------------------------------------------------------
    
    const DRange r23{spins.Dj23, spins.Dj23 + 2 * Dl};
    const DRange r24{spins.Dj24, spins.Dj24 + 2 * Dl};
    const DRange r25{spins.Dj25, spins.Dj25 + 2 * Dl};
    
    const DRange r34{spins.Dj34, spins.Dj34 + 2 * Dl};
    const DRange r35{spins.Dj35, spins.Dj35 + 2 * Dl};
    const DRange r45{spins.Dj45, spins.Dj45 + 2 * Dl};
    
    const DRange j12{spins.Dj12, spins.Dj12};
    const DRange j13{spins.Dj13, spins.Dj13};
    const DRange j14{spins.Dj14, spins.Dj14};
    const DRange j15{spins.Dj15, spins.Dj15};
    
    // ---------------------------------------------------------------------
    // Fixed boundary intertwiner singleton vectors.
    //
    // This is the important part.
    //
    // For amplitude mode, each booster only sees one boundary intertwiner i_n.
    // Therefore, if the booster parity rule is active, roughly half of the
    // corresponding virtual k_n values can disappear.
    // ---------------------------------------------------------------------
    
    const std::vector<int> i1_fixed{fixed_intertwiners[0]};
    // Build virtual intertwiner channels before deriving the support-aware x axis.
    ch.k2 = get_valid_k_tight(r23, r24, r25, j12);
    ch.k3 = get_valid_k_tight(r35, r34, r23, j13);
    ch.k4 = get_valid_k_tight(r45, j14, r24, r34);
    ch.k5 = get_valid_k_tight(r25, j15, r35, r45);

    if (ch.k2.empty() ||
        ch.k3.empty() ||
        ch.k4.empty() ||
        ch.k5.empty())
    {
        ch.k2.clear();
        ch.k3.clear();
        ch.k4.clear();
        ch.k5.clear();
        ch.xvals.clear();
        return ch;
    }
    
    // ---------------------------------------------------------------------
    // Build x after virtual intertwiner pruning.
    //
    // This must happen after the booster selection has removed virtual intertwiner values,
    // because x support depends on k2,k3,k4,k5.
    // ---------------------------------------------------------------------
    
    ch.xvals = make_tight_xvals(
                                spins,
                                Dl,
                                i1_fixed,
                                ch.k2,
                                ch.k3,
                                ch.k4,
                                ch.k5
                                );
    
    if (ch.xvals.empty()) {
        ch.k2.clear();
        ch.k3.clear();
        ch.k4.clear();
        ch.k5.clear();
    }
    
    return ch;
}




// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------


// (-1)^(Dpower/2), where Dpower is twice an integer or half-integer
// exponent.  Minimal kappa seeds can carry a +/-i phase when their coupled
// representation is half-integer, so reducing this phase to a real sign
// would silently discard valid odd doubled labels.
inline C neg_one_to_half_integer_power(int Dpower) noexcept {
  const int residue = ((Dpower % 4) + 4) % 4;
  switch (residue) {
  case 0: return C( 1.0,  0.0);
  case 1: return C( 0.0,  1.0);
  case 2: return C(-1.0,  0.0);
  default: return C(0.0, -1.0);
  }
}

inline const std::vector<double> LogFactTable(int max_init) {
  std::vector<double> toReturn(max_init + 1);

  for (int i = 0; i <= max_init; ++i) {

    toReturn[i] = lgamma(i + 1);
  }

  return toReturn;
}

//=======================================================================================================================================
//=======================================================================================================================================

static inline double log_cosh(double x) {
  double ax = std::abs(x);
  if (ax < 20.0)
    return std::log(std::cosh(ax));
  // cosh(ax) = 0.5*e^{ax}(1+e^{-2ax})
  return ax - M_LN2 + std::log1p(std::exp(-2.0 * ax));
}

static inline double log_sinh_pos(double x) { // x >= 0
  if (x < 20.0)
    return std::log(std::sinh(x));
  // sinh(x) = 0.5*e^{x}(1-e^{-2x})
  return x - M_LN2 + std::log1p(-std::exp(-2.0 * x));
}

struct PhaseLog {
  C phase;       // unit complex: ±1 or ±i
  double logabs; // log(|value|)
};

// returns invSin = 1/sin(pi/2*(m + i y)) = phase * exp(logabs)
static inline PhaseLog inv_sin_pi_over2_int_iy(int m, double y) {
  const double a = 0.5 * M_PI * y; // = (pi y)/2

  if ((m & 1) == 0) {
    // m = 2q: sin = i (-1)^q sinh(a)
    int q = m >> 1;
    double sgn = (y >= 0.0) ? 1.0 : -1.0; // sign(sinh(a)) = sign(y)
    double logabs = -log_sinh_pos(std::abs(a));

    // 1/sin = (-i)(-1)^q / sinh(a)
    C phase = C(0.0, -1.0) * (((q & 1) ? -1.0 : 1.0) * sgn);
    return {phase, logabs};
  } else {
    // m = 2q+1: sin = (-1)^q cosh(a)
    int q = (m - 1) >> 1;
    double logabs = -log_cosh(a);
    C phase = C(((q & 1) ? -1.0 : 1.0), 0.0); // ±1
    return {phase, logabs};
  }
}

// helper: fast complex ratio (nr+i ni)/(dr+i di)
static inline C cdiv(double nr, double ni, double dr, double di) noexcept {
  const double inv = 1.0 / (dr * dr + di * di);
  return C((nr * dr + ni * di) * inv, (ni * dr - nr * di) * inv);
}

// helper: exp(a + i b) with one exp + one sincos
static inline C exp_c(C z) noexcept {
  double s, c;
  detail::sincos_fast(z.imag(), s, c);
  double ea = std::exp(z.real());
  return C(ea * c, ea * s);
}

inline std::vector<double> GetVec3j(int Dj1, int Dj2, int Dk3,
                                   int Dj3)
{
    int Dnmin = 0;
    int Dnmax = -2;
    std::vector<double> vec3j;
    if (!threejrec::fill_wigner3j_magnetic_row(
            Dj1, Dj2, Dj3, Dk3, Dnmin, Dnmax, vec3j)) {
        return {};
    }
    return vec3j;
}

// The complete magnetic row is generated without external seeds.  The
// recurrence implementation normalizes the row and fixes its phase
// internally, including short and classically forbidden rows.

inline C kappaMinVec(double rho1, int Dj1, double rho2, int Dj2, double rho3, int Dk3,
           int Dj3, const std::vector<double> &Vec3j, const std::vector<double> &LF,
           double logDen) {
    const double k3 = 0.5 * double(Dk3);
    const double j3 = 0.5 * double(Dj3);
    // =========================================================================
    // Algorithmic Early Bail-Outs
    // If the Wigner 3j symbol violates the triangle condition, it is identically
    // zero. Bailing out here skips 7 expensive complex Gamma function calls!
    // =========================================================================
    if (std::abs(Dk3) > Dj3 || !triangle_condition(Dj1, Dj2, Dj3)) {
        return C(0.0, 0.0);
    }
    
    int Dnmin = std::max(-Dj1, -Dj2 + Dk3);
    int Dnmax = std::min(Dj1, Dj2 + Dk3);
    if (Dnmin > Dnmax)
        return C(0.0, 0.0);

    const std::size_t expected_terms =
        static_cast<std::size_t>((Dnmax - Dnmin) / 2 + 1);
    if (Vec3j.size() != expected_terms) {
        return C(0.0, 0.0);
    }
    
    
    
    // ---- basic scalars
    const double j1 = 0.5 * double(Dj1);
    const double j2 = 0.5 * double(Dj2);
    
    // ---- frequently used rho combinations
    const double rhoY = rho1 - rho2 - rho3;
    const double rhoP = rho1 - rho2 + rho3;
    
    // ---- phase tracking (Kept separate to preserve floating-point precision reduction)
    const double phaseArg = ln_gamma(1.0 + double(j3), rho3).imag();
    const C phase = std::polar(1.0, phaseArg);
    
    const int Dm = 2 - Dj1 + Dj2 + Dk3;
    if (Dm & 1) return C(0.0, 0.0);
    const int m = (Dm >> 1);
    
    PhaseLog invSin = inv_sin_pi_over2_int_iy(m, rhoY);
    
    // (-1)^(2*j1 + j3 - 2*k3 + 2*nmin).  The doubled exponent
    // remains exact for both integer and half-integer coupled labels.
    const int DsignExponent =
        2 * Dj1 + Dj3 - 2 * Dk3 + 2 * Dnmin;
    const C prefPhase =
        invSin.phase * neg_one_to_half_integer_power(DsignExponent);
    
    const double prefLogAbs = std::log(M_PI) - 0.5 * std::log(2.0 * double(j3) + 1.0) + invSin.logabs;
    
    // ---- logG block (Identical graph to prevent phase drift)
    const double lg1 = ln_gamma_logabs(1.0 + j1, std::abs(rho1));
    const double lg2 = ln_gamma_logabs(1.0 + j2, std::abs(rho2));
    
    const C gA = ln_gamma(0.5 * (1.0 + j1 + j2 - k3), -0.5 * (rho1 + rho2 - rho3));
    const C gB = ln_gamma(0.5 * (1.0 - j1 - j2 - k3), -0.5 * (rho1 + rho2 + rho3));
    const C gC = ln_gamma(1.0 + j1 + j2, rho3);
    
    C logG = C(0.5 * (LF[Dj1] + LF[Dj2]) - lg1 - lg2, 0.0) + gA - gB - gC + C(prefLogAbs - 0.5 * logDen, 0.0);
    // ---- n-sum recurrence variables
    const double Da_r = 1.0 + j1 + j2 - k3;
    const double Da_i = rhoP;
    const double Db_r = 1.0 - j1 - j2 - k3;
    const double Db_i = rhoY;
    
    C logF = C(0.5 * (-LF[(Dj1 - Dnmin) / 2] - LF[(Dj2 + Dk3 - Dnmin) / 2] -
                      LF[(Dj1 + Dnmin) / 2] - LF[(Dj2 - Dk3 + Dnmin) / 2]), 0.0) +
    ln_gamma(0.5 * (1.0 + j1 + j2 - k3 + Dnmin), 0.5 * rhoP) -
    ln_gamma(0.5 * (1.0 - j1 - j2 - k3 + Dnmin), 0.5 * rhoY);
    
    
    const int nTerms = static_cast<int>(Vec3j.size());
    
    const double* __restrict w = Vec3j.data();

/*
    C F = exp_c(logF);
    
    C sum_n(0.0, 0.0);
    
    int Dn = Dnmin;
    
    int idx = 0;
    
    for (int idx = 0; idx < nTerms - 1; ++idx, Dn += 2) {
        
        sum_n += w[idx] * F;
        
        const double num =
        double(Dj1 - Dn) * double(Dj2 + Dk3 - Dn);
        
        const double den =
        double(Dj1 + Dn + 2) * double(Dj2 - Dk3 + Dn + 2);
        
        const double srt = std::sqrt(num / den);
        
        const C ratio =
        cdiv(Da_r + double(Dn), Da_i,
             Db_r + double(Dn), Db_i);
        
        F *= srt * ratio;
    }
    
    sum_n += w[nTerms - 1] * F;
    
    return sum_n * GammasTimesPref * phase;*/

   double log_scale = logF.real();
C F = exp_c(logF - C(log_scale, 0.0));

C sum_n(0.0, 0.0);

auto rescale_pair = [&]() {
    const double aF = std::hypot(F.real(), F.imag());
    const double aS = std::hypot(sum_n.real(), sum_n.imag());
    const double a = std::max(aF, aS);

    if (!std::isfinite(a) || a == 0.0) {
        return;
    }

    if (a > 1e100 || a < 1e-100) {
        F /= a;
        sum_n /= a;
        log_scale += std::log(a);
    }
};

int Dn = Dnmin;

for (int idx = 0; idx < nTerms - 1; ++idx, Dn += 2) {
    sum_n += w[idx] * F;

    const double num =
        double(Dj1 - Dn) * double(Dj2 + Dk3 - Dn);

    const double den =
        double(Dj1 + Dn + 2) * double(Dj2 - Dk3 + Dn + 2);

    const double srt = std::sqrt(num / den);

    const C ratio =
        cdiv(Da_r + double(Dn), Da_i,
             Db_r + double(Dn), Db_i);

    F *= srt * ratio;

    rescale_pair();
}

sum_n += w[nTerms - 1] * F;
rescale_pair();

const double sum_abs = std::hypot(sum_n.real(), sum_n.imag());

if (sum_abs == 0.0 || !std::isfinite(sum_abs)) {
    return C(0.0, 0.0);
}

const double final_log_abs =
    logG.real()
  + log_scale
  + std::log(sum_abs);

constexpr double LOG_DBL_TRUE_MIN = -744.4400719213812;
constexpr double LOG_DBL_MAX      =  709.7827128933840;

if (final_log_abs < LOG_DBL_TRUE_MIN) {
    return C(0.0, 0.0);
}

if (final_log_abs > LOG_DBL_MAX) {
    return C(
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity()
    );
}

const C sum_phase = sum_n / sum_abs;
const C gamma_phase = std::polar(1.0, logG.imag());

return
    std::exp(final_log_abs)
  * sum_phase
  * gamma_phase
  * prefPhase
  * phase;
}


// Reference implementation
inline C kappaMin(double rho1, int Dj1, double rho2, int Dj2, double rho3, int Dk3,
           int Dj3, const std::vector<double> &LF,
           double logDen) {
    // ---- basic scalars
    const double j1 = 0.5 * double(Dj1);
    const double j2 = 0.5 * double(Dj2);
    const double k3 = 0.5 * double(Dk3);
    const double j3 = 0.5 * double(Dj3);

    if (std::abs(Dk3) > Dj3 || !triangle_condition(Dj1, Dj2, Dj3)) {
        return C(0.0, 0.0);
    }
    
    // ---- n-range (doubled)
    int Dnmin = std::max(-Dj1, -Dj2 + Dk3);
    int Dnmax = std::min(Dj1, Dj2 + Dk3);
    if (Dnmin > Dnmax)
        return C(0.0, 0.0);
    
    // ---- frequently used rho combinations
    const double rhoY = rho1 - rho2 - rho3;   // appears in 1/sin
    const double rhoP = rho1 - rho2 + rho3;   // appears in Da/ gamma
    
    // ---- phase from Gamma(1+j3+i rho3)/|...|
    const double phaseArg = ln_gamma(1.0 + double(j3), rho3).imag();
    const C phase = std::polar(1.0, phaseArg);
    
    // =========================================================================
    // prefactor: (pi / sqrt(2 j3+1)) * (-1)^q / sin(pi/2*(m + i y))
    // Here m is integral; compute it exactly from doubled labels.
    // =========================================================================
    
    // m = 1 - j1 + j2 + k3.  In doubled units: 2m = 2 - Dj1 + Dj2 + 2k3
    const int Dm = 2 - Dj1 + Dj2 + Dk3;
    if (Dm & 1) return C(0.0, 0.0);
    const int m = (Dm >> 1);
    
    PhaseLog invSin = inv_sin_pi_over2_int_iy(m, rhoY);
    
    const int DsignExponent =
        2 * Dj1 + Dj3 - 2 * Dk3 + 2 * Dnmin;
    const C prefPhase =
        invSin.phase * neg_one_to_half_integer_power(DsignExponent);
    
    const double prefLogAbs =
    std::log(M_PI) - 0.5 * std::log(2.0 * double(j3) + 1.0) + invSin.logabs;
    
    // =========================================================================
    // logDen = log(cosh(x) + cos(pi*(j1+j2+k3))) with cos = ±1 exactly
    // x = pi*(rho1+rho2+rho3)
    // =========================================================================

    // =========================================================================
    // Build logG (complex): everything that stays outside the n-sum,
    // and fold all magnitudes BEFORE exp.
    // =========================================================================
    
    // Only the real components of the first two ln_gamma terms contribute,
    // matching the Mathematica |Gamma| factors.
    const double lg1 =
    ln_gamma_logabs(1.0 + j1, std::abs(rho1)); // real part even in imag
    const double lg2 = ln_gamma_logabs(1.0 + j2, std::abs(rho2));
    
    const C gA =
    ln_gamma(0.5 * (1.0 + j1 + j2 - k3), -0.5 * (rho1 + rho2 - rho3));
    const C gB =
    ln_gamma(0.5 * (1.0 - j1 - j2 - k3), -0.5 * (rho1 + rho2 + rho3));
    const C gC = ln_gamma(1.0 + j1 + j2, rho3);
    
    C logG = C(0.5 * (LF[Dj1] + LF[Dj2]) - lg1 - lg2, 0.0) + gA - gB - gC +
    C(prefLogAbs - 0.5 * logDen, 0.0);
    
    // exp(logG) (faster than std::exp(complex) in many libcs)
    const C GammasTimesPref = exp_c(logG) * prefPhase;
    
    // =========================================================================
    // n-sum seed F and recurrence
    // =========================================================================
    
    // Da = (1+j1+j2-k3) + i(rho1-rho2+rho3)
    // Db = (1-j1-j2-k3) + i(rho1-rho2-rho3)
    const double Da_r = 1.0 + j1 + j2 - k3;
    const double Da_i = rhoP;
    const double Db_r = 1.0 - j1 - j2 - k3;
    const double Db_i = rhoY; // rho1-rho2-rho3
    
    // initial F at Dn = Dnmin
    C logF = C(0.5 * (-LF[(Dj1 - Dnmin) / 2] - LF[(Dj2 + Dk3 - Dnmin) / 2] -
                      LF[(Dj1 + Dnmin) / 2] - LF[(Dj2 - Dk3 + Dnmin) / 2]),
               0.0) +
    ln_gamma(0.5 * (1.0 + j1 + j2 - k3 + Dnmin), 0.5 * (rhoP)) -
    ln_gamma(0.5 * (1.0 - j1 - j2 - k3 + Dnmin), 0.5 * (rhoY));
    
    C F = exp_c(logF);

    const std::vector<double> vec3j = GetVec3j(Dj1, Dj2, Dk3, Dj3);
    if (vec3j.size() !=
        static_cast<std::size_t>((Dnmax - Dnmin) / 2 + 1)) {
        return C(0.0, 0.0);
    }

    C sum_n(0.0, 0.0);
    std::size_t threej_index = 0;
    
    for (int Dn = Dnmin; Dn <= Dnmax; Dn += 2) {
        
        const double wig3j = vec3j[threej_index++];
        // often faster to just do it (branchless) than if(wig3j!=0)
        sum_n += wig3j * F;
        
        if (Dn != Dnmax) {
            const double num = double(Dj1 - Dn) * double(Dj2 + Dk3 - Dn);
            const double den = double(Dj1 + Dn + 2) * double(Dj2 - Dk3 + Dn + 2);
            const double srt = std::sqrt(num / den);
            
            const double shift = double(Dn);
            const C ratio = cdiv(Da_r + shift, Da_i, Db_r + shift, Db_i);
            
            F *= (srt * ratio);
        }
    }
    
    return sum_n * GammasTimesPref * phase;
}
