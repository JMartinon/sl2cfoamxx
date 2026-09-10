// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "full_tensor/FullTensorPackedContainers.hpp"
#include <cstdint>

namespace b4support {

// ============================================================================
// Packed full-tensor 6j tensors
// ============================================================================
//
// Dense logical tensors:
//
//   sjA(x,i1,k5,l25)
//   sjB(x,k5,k4,l35,l45)
//   sjC(x,k4,k3,l24,l34,l35)
//   sjD(x,k3,k2,l23,l24)
//   sjE(x,i1,k2,l25)
//
// Packed tensors:
//
//   sjA: block (i1,x,l25)      -> k5
//   sjB: block (x,k4,l35,l45)  -> k5
//   sjC: half-stored block (x,l24,l34,l35) -> k3 x k4
//   sjD: block (x,k3,l23,l24)  -> k2
//   sjE: block (i1,x,l25)      -> k2
// ============================================================================

enum FullTensorSjComponent : std::uint8_t {
    FullTensorSjA = 1u << 0,
    FullTensorSjB = 1u << 1,
    FullTensorSjC = 1u << 2,
    FullTensorSjD = 1u << 3,
    FullTensorSjE = 1u << 4,
    FullTensorSjBoundaryDependent = FullTensorSjA | FullTensorSjE,
    FullTensorSjBoundaryIndependent = FullTensorSjB | FullTensorSjC | FullTensorSjD,
    FullTensorSjAll = FullTensorSjBoundaryDependent | FullTensorSjBoundaryIndependent
};

void fill_packed_full_tensor_sj_tensors_rec(
    const B4Spins& s,
    const FullTensorInputSupportPlan& support,
    PackedFullTensorSJTensors& out,
    std::uint8_t components = FullTensorSjAll);

} // namespace b4support
