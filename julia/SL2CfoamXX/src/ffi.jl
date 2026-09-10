# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later

const _library_handle = Ref{Ptr{Cvoid}}(C_NULL)
const _mkl_dependency_handle = Ref{Ptr{Cvoid}}(C_NULL)
const _symbols = Dict{Symbol,Ptr{Cvoid}}()
const _symbol_names = (
    :sjx_abi_version, :sjx_error_kind, :sjx_error_code, :sjx_error_system_code,
    :sjx_error_message, :sjx_error_path1, :sjx_error_path2, :sjx_error_free,
    :sjx_free, :sjx_default_config, :sjx_current_config, :sjx_configure,
    :sjx_config_dispose, :sjx_build_info, :sjx_build_info_dispose,
    :sjx_backend_available, :sjx_cuda_device_count, :sjx_cuda_device_info,
    :sjx_device_info_dispose, :sjx_progress_create, :sjx_progress_poll,
    :sjx_progress_free, :sjx_indices_from_intertwiners, :sjx_compute_amplitude,
    :sjx_compute_coherent_amplitude, :sjx_compute_full_tensor,
    :sjx_compute_coherent_coefficients, :sjx_check_closure, :sjx_tensor_create,
    :sjx_tensor_create_mapped, :sjx_tensor_load, :sjx_tensor_map, :sjx_tensor_free,
    :sjx_tensor_info, :sjx_tensor_info_dispose, :sjx_tensor_file_info,
    :sjx_tensor_get, :sjx_tensor_set, :sjx_tensor_get_flat, :sjx_tensor_set_flat,
    :sjx_tensor_payload, :sjx_tensor_copy_values, :sjx_tensor_clone,
    :sjx_tensor_dense_copy, :sjx_tensor_fill, :sjx_tensor_save, :sjx_tensor_flush,
    :sjx_tensor_convert_storage, :sjx_tensor_format, :sjx_tensor_tag,
)

_sym(name::Symbol) = _symbols[name]
_string(pointer::Ptr) = pointer == C_NULL ? "" : unsafe_string(pointer)
_nullable_string(pointer::Ptr) = (text = _string(pointer); isempty(text) ? nothing : text)

function __init__()
    library = get(ENV, "SL2CFOAMXX_JULIA_LIBRARY", "")
    if isempty(library)
        # Resolve from the module's runtime location, including when a
        # precompiled installation has moved since its cache was created.
        source = pathof(@__MODULE__)
        source === nothing && error("load SL2CfoamXX as a package or set SL2CFOAMXX_JULIA_LIBRARY")
        library = joinpath(dirname(source), "..", "lib",
                           "libsl2cfoamxx_julia." * Libdl.dlext)
    end
    handle = Libdl.dlopen(abspath(library))
    abi = ccall(Libdl.dlsym(handle, :sjx_abi_version), UInt32, ())
    abi == 1 || error("unsupported sl2cfoam++ Julia bridge ABI $abi (expected 1)")
    for name in _symbol_names
        _symbols[name] = Libdl.dlsym(handle, name)
    end
    _library_handle[] = handle
    if Sys.islinux() && occursin("mkl", lowercase(build_info().blas_provider))
        # MKL's lazily loaded kernels need the linked dependency symbols in
        # global scope. Promote this library only, retaining its handle.
        _mkl_dependency_handle[] = Libdl.dlopen(abspath(library),
                                               Libdl.RTLD_NOLOAD | Libdl.RTLD_GLOBAL)
    end
end

"""An operational failure from the native library, with a stable `code`."""
struct Error <: Exception
    code::ErrorCode
    message::String
end
Base.showerror(io::IO, error::Error) = print(io, error.message, " [", error.code, "]")

struct FilesystemError <: Exception
    message::String
    system_code::Int
    path1::Union{Nothing,String}
    path2::Union{Nothing,String}
end
Base.showerror(io::IO, error::FilesystemError) = print(io, error.message)

function _check_error(pointer::Ptr{Cvoid})
    pointer == C_NULL && return nothing
    exception = try
        kind = ccall(_sym(:sjx_error_kind), Int32, (Ptr{Cvoid},), pointer)
        message = _string(ccall(_sym(:sjx_error_message), Ptr{UInt8}, (Ptr{Cvoid},), pointer))
        if kind == 1
            ArgumentError(message)
        elseif kind == 2
            BoundsError(message)
        elseif kind == 3
            OverflowError(message)
        elseif kind == 4
            OutOfMemoryError()
        elseif kind == 5
            FilesystemError(message,
                Int(ccall(_sym(:sjx_error_system_code), Int32, (Ptr{Cvoid},), pointer)),
                _nullable_string(ccall(_sym(:sjx_error_path1), Ptr{UInt8}, (Ptr{Cvoid},), pointer)),
                _nullable_string(ccall(_sym(:sjx_error_path2), Ptr{UInt8}, (Ptr{Cvoid},), pointer)))
        elseif kind == 6
            Error(ErrorCode(ccall(_sym(:sjx_error_code), Int32, (Ptr{Cvoid},), pointer)), message)
        else
            ErrorException(message)
        end
    finally
        ccall(_sym(:sjx_error_free), Cvoid, (Ptr{Cvoid},), pointer)
    end
    throw(exception)
end
_check_error(error::Ref{Ptr{Cvoid}}) = _check_error(error[])

function _nonnegative_int(value, ::Type{T}, name::AbstractString) where {T<:Integer}
    value isa Integer && !(value isa Bool) ||
        throw(ArgumentError("$name must be an integer (not Bool)"))
    value >= 0 || throw(ArgumentError("$name must be nonnegative"))
    value <= typemax(T) || throw(OverflowError("$name exceeds $(T)"))
    return T(value)
end

function _path(value)
    value === nothing && return ""
    value isa AbstractString || throw(ArgumentError("path must be a string or nothing"))
    occursin('\0', value) && throw(ArgumentError("path must not contain NUL"))
    return String(value)
end

function _labels(values, length_required::Int, name::AbstractString)
    (values isa Tuple || values isa AbstractVector) ||
        throw(ArgumentError("$name must be a vector or tuple"))
    length(values) == length_required ||
        throw(ArgumentError("$name must have $length_required entries"))
    return Int32[_nonnegative_int(value, Int32, name) for value in values]
end

function _angles(values, rows::Int)
    result = Vector{Float64}(undef, 2rows)
    if values isa AbstractMatrix
        size(values) == (rows, 2) || throw(ArgumentError("angles must have size ($rows, 2)"))
        for row in 1:rows, column in 1:2
            result[2(row - 1) + column] = _real(values[row, column], "angle")
        end
    elseif values isa Tuple || values isa AbstractVector
        length(values) == rows || throw(ArgumentError("angles must have $rows rows"))
        for row in 1:rows
            pair = values[row]
            (pair isa Tuple || pair isa AbstractVector) && length(pair) == 2 ||
                throw(ArgumentError("each angle row must contain theta and phi"))
            for column in 1:2
                result[2(row - 1) + column] = _real(pair[column], "angle")
            end
        end
    else
        throw(ArgumentError("angles must be a matrix or a sequence of pairs"))
    end
    return result
end

function _real(value, name::AbstractString)
    value isa Real && !(value isa Bool) || throw(ArgumentError("$name must be real"))
    result = Float64(value)
    isfinite(result) || throw(ArgumentError("$name must be finite"))
    return result
end
