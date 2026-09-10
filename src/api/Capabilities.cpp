// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "sl2cfoamxx/capabilities.hpp"
#include "sl2cfoamxx/error.hpp"

#if !defined(SL2CFOAMXX_COMPILED_WITH_CUDA)
#define SL2CFOAMXX_COMPILED_WITH_CUDA 0
#endif

#if SL2CFOAMXX_COMPILED_WITH_CUDA
#include "gpu/CudaIntrospection.hpp"
#endif

#include <stdexcept>

#ifndef SL2CFOAMXX_VERSION_STRING
#define SL2CFOAMXX_VERSION_STRING "unknown"
#endif
#ifndef SL2CFOAMXX_COMPILER_ID_STRING
#define SL2CFOAMXX_COMPILER_ID_STRING "unknown"
#endif
#ifndef SL2CFOAMXX_COMPILER_VERSION_STRING
#define SL2CFOAMXX_COMPILER_VERSION_STRING "unknown"
#endif
#ifndef SL2CFOAMXX_BUILD_TYPE_STRING
#define SL2CFOAMXX_BUILD_TYPE_STRING "unknown"
#endif
#ifndef SL2CFOAMXX_BLAS_VENDOR_STRING
#define SL2CFOAMXX_BLAS_VENDOR_STRING "unknown"
#endif
#ifndef SL2CFOAMXX_BLAS_PROVIDER_STRING
#define SL2CFOAMXX_BLAS_PROVIDER_STRING "unknown"
#endif
#ifndef SL2CFOAMXX_NATIVE_ARCH_ENABLED
#define SL2CFOAMXX_NATIVE_ARCH_ENABLED 0
#endif
#ifndef SL2CFOAMXX_CUDA_FAST_MATH_ENABLED
#define SL2CFOAMXX_CUDA_FAST_MATH_ENABLED 0
#endif
#ifndef SL2CFOAMXX_CUDA_TOOLKIT_VERSION_STRING
#define SL2CFOAMXX_CUDA_TOOLKIT_VERSION_STRING ""
#endif
#ifndef SL2CFOAMXX_CUDA_ARCHITECTURES_STRING
#define SL2CFOAMXX_CUDA_ARCHITECTURES_STRING ""
#endif
#ifndef SL2CFOAMXX_CUTENSOR_VERSION_STRING
#define SL2CFOAMXX_CUTENSOR_VERSION_STRING ""
#endif

namespace sl2cfoamxx {

std::string_view version() noexcept
{
    return SL2CFOAMXX_VERSION_STRING;
}

BuildInfo build_info() noexcept
{
    return BuildInfo{
        version(),
        SL2CFOAMXX_COMPILER_ID_STRING,
        SL2CFOAMXX_COMPILER_VERSION_STRING,
        SL2CFOAMXX_BUILD_TYPE_STRING,
        SL2CFOAMXX_BLAS_VENDOR_STRING,
        SL2CFOAMXX_BLAS_PROVIDER_STRING,
        SL2CFOAMXX_NATIVE_ARCH_ENABLED != 0,
        SL2CFOAMXX_COMPILED_WITH_CUDA != 0,
        SL2CFOAMXX_CUDA_FAST_MATH_ENABLED != 0,
        SL2CFOAMXX_CUDA_TOOLKIT_VERSION_STRING,
        SL2CFOAMXX_CUDA_ARCHITECTURES_STRING,
        SL2CFOAMXX_CUTENSOR_VERSION_STRING};
}

bool backend_available(Backend backend) noexcept
{
    switch (backend) {
    case Backend::CPU:
        return true;
    case Backend::GPU:
        try {
            return cuda_device_count() != 0;
        } catch (...) {
            return false;
        }
    }
    return false;
}

std::size_t cuda_device_count()
{
#if SL2CFOAMXX_COMPILED_WITH_CUDA
    return detail::cuda_device_count_impl();
#else
    return 0;
#endif
}

CudaDeviceInfo cuda_device_info(std::size_t ordinal)
{
#if SL2CFOAMXX_COMPILED_WITH_CUDA
    return detail::cuda_device_info_impl(ordinal);
#else
    (void)ordinal;
    throw Error(
        ErrorCode::backend_unavailable,
        "sl2cfoam++: CUDA device information is unavailable because this "
        "library was built with ENABLE_CUDA=OFF");
#endif
}

} // namespace sl2cfoamxx
