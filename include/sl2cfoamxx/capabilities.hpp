// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include "sl2cfoamxx/error.hpp"
#include "sl2cfoamxx/export.hpp"

#include "sl2cfoamxx/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace sl2cfoamxx {

// Immutable facts about the loaded sl2cfoam++ shared library. String views
// refer to process-lifetime storage owned by the library.
struct BuildInfo {
    std::string_view version{};
    std::string_view compiler_id{};
    std::string_view compiler_version{};
    std::string_view build_type{};
    // The FindBLAS selector requested at configure time (for example All,
    // AOCL, or OpenBLAS) and the provider/link description CMake resolved.
    // They differ when the portable selector All is used.
    std::string_view blas_vendor_selector{};
    std::string_view blas_provider{};
    bool native_arch_enabled = false;
    bool cuda_backend_built = false;
    // True when ENABLE_FAST_MATH requested fast CUDA kernels. Numerically
    // sensitive recurrence translation units retain precise div/sqrt/FTZ
    // semantics even in such a build.
    bool cuda_fast_math_requested = false;
    std::string_view cuda_toolkit_version{};
    std::string_view cuda_architectures{};
    std::string_view cutensor_version{};
};

// Properties of one CUDA device visible to this process. The ordinal follows
// CUDA_VISIBLE_DEVICES remapping and is the value accepted by the CUDA runtime.
struct CudaDeviceInfo {
    std::size_t ordinal = 0;
    std::string name{};
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    std::uint64_t total_memory_bytes = 0;
    int multiprocessor_count = 0;
    int warp_size = 0;
    int max_threads_per_block = 0;
    // CUDA's stable domain:bus:device identifier, useful for mapping a GPU to
    // its nearest CPU NUMA node through external topology tools.
    std::string pci_bus_id{};
};

// Version of the loaded shared library, not merely the headers used to compile
// the caller.
[[nodiscard]] SL2CFOAMXX_API std::string_view version() noexcept;

[[nodiscard]] SL2CFOAMXX_API BuildInfo build_info() noexcept;

// CPU is always available. GPU is available only when this build contains the
// CUDA backend and at least one CUDA device is currently visible. This is a
// lightweight availability probe; it cannot guarantee that a future workload
// fits device memory or that every requested kernel launch will succeed.
[[nodiscard]] SL2CFOAMXX_API bool backend_available(Backend backend) noexcept;

// A CPU-only build reports zero devices. A CUDA build reports CUDA runtime or
// driver initialization failures as ErrorCode::backend_failure instead of
// silently treating a broken runtime as an empty machine.
[[nodiscard]] SL2CFOAMXX_API std::size_t cuda_device_count();

// Query one CUDA-visible ordinal without changing the caller's active device.
// An invalid ordinal throws std::out_of_range. A CPU-only build throws
// ErrorCode::backend_unavailable; CUDA runtime/driver failures throw
// ErrorCode::backend_failure.
[[nodiscard]] SL2CFOAMXX_API CudaDeviceInfo cuda_device_info(
    std::size_t ordinal);

} // namespace sl2cfoamxx
