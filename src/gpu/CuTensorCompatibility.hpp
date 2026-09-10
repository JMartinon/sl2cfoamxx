// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cuda_runtime.h>
#include <cutensor.h>

namespace b4gpu::cutensor_compat {

// cuTENSOR 2.3 changed tensor descriptors from cutensorDataType_t to
// cudaDataType_t.  Keep the call sites strongly typed for both sides of that
// source-compatible API boundary; the constants have the same semantics but
// C++ does not implicitly convert between the two enum types.
#if CUTENSOR_VERSION >= 20300
using DataType = cudaDataType_t;
inline constexpr DataType kReal64 = CUDA_R_64F;
inline constexpr DataType kComplex64 = CUDA_C_64F;
#else
using DataType = cutensorDataType_t;
inline constexpr DataType kReal64 = CUTENSOR_R_64F;
inline constexpr DataType kComplex64 = CUTENSOR_C_64F;
#endif

} // namespace b4gpu::cutensor_compat
