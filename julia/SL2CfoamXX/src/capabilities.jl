# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later

struct BuildInfo
    version::String
    compiler_id::String
    compiler_version::String
    build_type::String
    blas_vendor_selector::String
    blas_provider::String
    native_arch_enabled::Bool
    cuda_backend_built::Bool
    cuda_fast_math_requested::Bool
    cuda_toolkit_version::String
    cuda_architectures::String
    cutensor_version::String
end

struct _BuildInfo
    native_arch_enabled::Int32
    cuda_backend_built::Int32
    cuda_fast_math_requested::Int32
    version::Ptr{UInt8}
    compiler_id::Ptr{UInt8}
    compiler_version::Ptr{UInt8}
    build_type::Ptr{UInt8}
    blas_vendor_selector::Ptr{UInt8}
    blas_provider::Ptr{UInt8}
    cuda_toolkit_version::Ptr{UInt8}
    cuda_architectures::Ptr{UInt8}
    cutensor_version::Ptr{UInt8}
end

function build_info()
    output = Ref{_BuildInfo}()
    _check_error(ccall(_sym(:sjx_build_info), Ptr{Cvoid}, (Ref{_BuildInfo},), output))
    try
        raw = output[]
        return BuildInfo(_string(raw.version), _string(raw.compiler_id),
            _string(raw.compiler_version), _string(raw.build_type),
            _string(raw.blas_vendor_selector), _string(raw.blas_provider),
            raw.native_arch_enabled != 0, raw.cuda_backend_built != 0,
            raw.cuda_fast_math_requested != 0, _string(raw.cuda_toolkit_version),
            _string(raw.cuda_architectures), _string(raw.cutensor_version))
    finally
        ccall(_sym(:sjx_build_info_dispose), Cvoid, (Ref{_BuildInfo},), output)
    end
end

version() = build_info().version
function backend_available(backend::Backend)
    output = Ref{Int32}(0)
    _check_error(ccall(_sym(:sjx_backend_available), Ptr{Cvoid},
                       (Int32, Ref{Int32}), Int32(backend), output))
    return output[] != 0
end
function cuda_device_count()
    output = Ref{UInt64}(0)
    _check_error(ccall(_sym(:sjx_cuda_device_count), Ptr{Cvoid}, (Ref{UInt64},), output))
    return Int(output[])
end

struct CudaDeviceInfo
    ordinal::Int
    name::String
    compute_capability_major::Int
    compute_capability_minor::Int
    total_memory_bytes::UInt64
    multiprocessor_count::Int
    warp_size::Int
    max_threads_per_block::Int
    pci_bus_id::String
end
struct _DeviceInfo
    ordinal::UInt64
    total_memory_bytes::UInt64
    compute_capability_major::Int32
    compute_capability_minor::Int32
    multiprocessor_count::Int32
    warp_size::Int32
    max_threads_per_block::Int32
    name::Ptr{UInt8}
    pci_bus_id::Ptr{UInt8}
end
"""Query a CUDA-visible device ordinal (CUDA ordinals remain zero-based)."""
function cuda_device_info(ordinal=0)
    index = _nonnegative_int(ordinal, UInt64, "device ordinal")
    output = Ref{_DeviceInfo}()
    _check_error(ccall(_sym(:sjx_cuda_device_info), Ptr{Cvoid},
                       (UInt64, Ref{_DeviceInfo}), index, output))
    try
        raw = output[]
        return CudaDeviceInfo(Int(raw.ordinal), _string(raw.name),
            Int(raw.compute_capability_major), Int(raw.compute_capability_minor),
            raw.total_memory_bytes, Int(raw.multiprocessor_count),
            Int(raw.warp_size), Int(raw.max_threads_per_block), _string(raw.pci_bus_id))
    finally
        ccall(_sym(:sjx_device_info_dispose), Cvoid, (Ref{_DeviceInfo},), output)
    end
end
