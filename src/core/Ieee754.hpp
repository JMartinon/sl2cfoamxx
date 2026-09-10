// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstdint>
#include <cstring>

namespace b4utils::ieee754 {

// These checks intentionally inspect the representation. Under fast-math,
// compilers may otherwise assume that a floating-point value is always finite
// and discard recurrence failure handling.
inline std::uint64_t double_bits(double value) noexcept
{
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

inline double double_from_bits(std::uint64_t bits) noexcept
{
    double value = 0.0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

inline bool finite_double(double value) noexcept
{
    volatile std::uint64_t bits = double_bits(value);
    return (bits & 0x7ff0000000000000ULL) != 0x7ff0000000000000ULL;
}

inline bool negative_double(double value) noexcept
{
    volatile std::uint64_t bits = double_bits(value);
    return (bits & 0x8000000000000000ULL) != 0;
}

inline bool zero_double(double value) noexcept
{
    volatile std::uint64_t bits = double_bits(value);
    return (bits & 0x7fffffffffffffffULL) == 0;
}

inline double preserve_zero_sign(double value, bool negative) noexcept
{
    volatile std::uint64_t observed = double_bits(value);
    std::uint64_t bits = observed;
    if ((bits & 0x7fffffffffffffffULL) == 0) {
        bits = negative ? 0x8000000000000000ULL : 0ULL;
        return double_from_bits(bits);
    }
    return value;
}

} // namespace b4utils::ieee754
