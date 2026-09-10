// SPDX-FileCopyrightText: 2026 Jeremy Martinon
// SPDX-License-Identifier: GPL-3.0-or-later

#include "gpu/CudaIntrospection.hpp"

#include "sl2cfoamxx/error.hpp"

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace sl2cfoamxx::detail {
namespace {

[[noreturn]] void throw_cuda_error(cudaError_t status, const char* operation)
{
    const char* description = cudaGetErrorString(status);
    throw Error(
        ErrorCode::backend_failure,
        std::string("sl2cfoam++: ") + operation + " failed: " +
        (description != nullptr ? description : "unknown CUDA error"));
}

void check_cuda(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw_cuda_error(status, operation);
    }
}

} // namespace

std::size_t cuda_device_count_impl()
{
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count < 0) {
        throw Error(
            ErrorCode::backend_failure,
            "sl2cfoam++: cudaGetDeviceCount returned a negative count");
    }
    return static_cast<std::size_t>(count);
}

CudaDeviceInfo cuda_device_info_impl(std::size_t ordinal)
{
    const std::size_t count = cuda_device_count_impl();
    if (ordinal >= count) {
        throw std::out_of_range(
            "sl2cfoam++: CUDA device ordinal " + std::to_string(ordinal) +
            " is outside the visible range [0," + std::to_string(count) + ")");
    }

    cudaDeviceProp property{};
    check_cuda(
        cudaGetDeviceProperties(&property, static_cast<int>(ordinal)),
        "cudaGetDeviceProperties");

    CudaDeviceInfo info;
    info.ordinal = ordinal;
    info.name = property.name;
    info.compute_capability_major = property.major;
    info.compute_capability_minor = property.minor;
    info.total_memory_bytes =
        static_cast<std::uint64_t>(property.totalGlobalMem);
    info.multiprocessor_count = property.multiProcessorCount;
    info.warp_size = property.warpSize;
    info.max_threads_per_block = property.maxThreadsPerBlock;
    char pci_bus_id[32]{};
    check_cuda(
        cudaDeviceGetPCIBusId(
            pci_bus_id, sizeof(pci_bus_id), static_cast<int>(ordinal)),
        "cudaDeviceGetPCIBusId");
    info.pci_bus_id = pci_bus_id;
    return info;
}

} // namespace sl2cfoamxx::detail
