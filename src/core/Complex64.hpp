// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <type_traits>

namespace b4utils {

// Scalar complex values cross the host/CUDA boundary as two adjacent doubles.
// Keep arithmetic as free functions in utils.hpp so this representation stays
// trivial, standard-layout, and safe for byte-for-byte device transfers.
struct alignas(16) Complex64 {
    double re;
    double im;
};

static_assert(std::is_trivially_copyable_v<Complex64>,
              "Complex64 must stay trivially copyable");
static_assert(std::is_trivial_v<Complex64> &&
              std::is_standard_layout_v<Complex64>,
              "Complex64 must remain a trivial standard-layout type");
static_assert(sizeof(Complex64) == 2 * sizeof(double),
              "Complex64 must be exactly two doubles");
static_assert(alignof(Complex64) >= 2 * sizeof(double),
              "Complex64 must be at least 16-byte aligned");

} // namespace b4utils
