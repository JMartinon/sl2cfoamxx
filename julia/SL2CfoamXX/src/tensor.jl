# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later

"""Scientific inputs attached to a tensor returned by `compute_full_tensor`."""
struct FullTensorScientificMetadata
    spins::NTuple{10,Int32}
    Dl::Int
    gamma::Float64
    simplicity_constraint::SimplicityConstraint
end

"""Logical shape, physical representation, backing, and optional scientific inputs."""
struct FullTensorMetadata
    scientific::Union{Nothing,FullTensorScientificMetadata}
    logical_shape::NTuple{5,Int}
    logical_values::Int
    stored_values::Int
    storage::TensorStorage
    layout::TensorLayout
    reassociated_layout::Union{Nothing,FullTensorReassociatedLayout}
    mapped::Bool
    writable::Bool
    path::Union{Nothing,String}
end

"""Options for `format_tensor`; `max_values=typemax(UInt64)` prints all entries."""
Base.@kwdef struct FullTensorPrintOptions
    max_values::UInt64 = 4096
    precision::Int32 = 17
    include_metadata::Bool = true
end

# This private POD mirrors SJXTensorInfo exactly, including its owned path.
struct _CTensorInfo
    shape::NTuple{5,UInt64}
    strides::NTuple{5,UInt64}
    logical_values::UInt64
    stored_values::UInt64
    stored_bytes::UInt64
    dense_bytes::UInt64
    Dl::UInt64
    gamma::Float64
    spins::NTuple{10,Int32}
    has_scientific::Int32
    simplicity_constraint::Int32
    storage::Int32
    layout::UInt32
    reassociated_layout::Int32
    mapped::Int32
    writable::Int32
    symmetry_compressed::Int32
    dense::Int32
    path::Ptr{UInt8}
end

_empty_tensor_info() = _CTensorInfo(
    ntuple(_ -> UInt64(0), 5), ntuple(_ -> UInt64(0), 5),
    0, 0, 0, 0, 0, 0.0, ntuple(_ -> Int32(0), 10),
    0, 0, 0, 0, -1, 0, 0, 0, 0, C_NULL)

mutable struct _TensorOwner
    handle::Ptr{Cvoid}
    destroy::Ptr{Cvoid}
end

function _finalize_tensor(owner::_TensorOwner)
    handle = owner.handle
    if handle != C_NULL
        owner.handle = C_NULL
        # Resolve the function before registering the finalizer. Finalization
        # performs only native destruction, with no lookup, logging, or yield.
        ccall(owner.destroy, Cvoid, (Ptr{Cvoid},), handle)
    end
    nothing
end

# @threadcall must finish before Julia unwinds the caller's argument roots or
# releases native output records. Defer Ctrl-C through completion and error
# translation; no native worker is allowed to call back into Julia.
function _tensor_threaded(operation)
    Base.disable_sigint() do
        _check_error(operation())
    end
end

function _tensor_result(operation)
    result = Ref{Ptr{Cvoid}}(C_NULL)
    release = _sym(:sjx_tensor_free)
    try
        Base.disable_sigint() do
            _check_error(operation(result))
            handle = result[]
            result[] = C_NULL
            # Ownership transfers before signals are reenabled, so an
            # interruption at return cannot orphan a newly allocated handle.
            _take_tensor(handle, release)
        end
    finally
        result[] == C_NULL || ccall(release, Cvoid, (Ptr{Cvoid},), result[])
    end
end

"""
    FullTensor(shape; init=TensorInit.zero,
               layout=TensorLayout.canonical_dense_column_major)

A rank-five `AbstractArray{Float64,5}` backed by the C++ `FullTensor`.
Julia indexing is one-based and always follows canonical `(i1,i2,i3,i4,i5)`
order, including linear indexing. `length` counts logical entries; use
`stored_size` or `physical_view` to inspect the physical payload.

`Array(t)` makes an independent Julia-owned canonical array. `copy(t)` keeps
the native representation in independent owned memory. Ordinary `view` and
`reshape` retain the native owner and preserve read-only mapping protection.
"""
struct FullTensor <: AbstractArray{Float64,5}
    _owner::_TensorOwner
    _metadata::FullTensorMetadata
    _logical_strides::NTuple{5,Int}
    _stored_bytes::Int
    _dense_bytes::Int
    _dense::Bool
    _symmetry_compressed::Bool
end

function _tensor_int(value::UInt64, name::AbstractString)
    value <= typemax(Int) || throw(OverflowError("$name does not fit Julia Int"))
    Int(value)
end

function _decode_tensor_info(info::_CTensorInfo)
    shape = map(value -> _tensor_int(value, "tensor dimension"), info.shape)
    scientific = info.has_scientific == 0 ? nothing : FullTensorScientificMetadata(
        info.spins, _tensor_int(info.Dl, "tensor cutoff"), info.gamma,
        SimplicityConstraint(info.simplicity_constraint))
    path = info.path == C_NULL ? nothing : unsafe_string(info.path)
    path == "" && (path = nothing)
    FullTensorMetadata(scientific, shape,
        _tensor_int(info.logical_values, "logical tensor size"),
        _tensor_int(info.stored_values, "stored tensor size"),
        TensorStorage(info.storage), TensorLayout(info.layout),
        info.reassociated_layout < 0 ? nothing :
            FullTensorReassociatedLayout(info.reassociated_layout),
        info.mapped != 0, info.writable != 0, path)
end

# Adopts a newly returned native handle, including handles from computation.
# Neither resizing nor explicit closing is exposed: extant views stay valid.
function _take_tensor(handle::Ptr{Cvoid}, destroy::Ptr{Cvoid}=_sym(:sjx_tensor_free))
    handle != C_NULL || error("native tensor operation returned a null handle")
    owner = nothing
    info = nothing
    dispose = Ptr{Cvoid}(C_NULL)
    try
        # Consume the handle before the first owner/record allocation, also
        # covering allocation failure and failed finalizer registration.
        owner = _TensorOwner(handle, destroy)
        finalizer(_finalize_tensor, owner)
        dispose = _sym(:sjx_tensor_info_dispose)
        info = Ref(_empty_tensor_info())
        GC.@preserve owner begin
            _check_error(ccall(_sym(:sjx_tensor_info), Ptr{Cvoid},
                (Ptr{Cvoid}, Ref{_CTensorInfo}), handle, info))
        end
        native = info[]
        FullTensor(owner, _decode_tensor_info(native),
            map(value -> _tensor_int(value, "tensor stride"), native.strides),
            _tensor_int(native.stored_bytes, "stored tensor bytes"),
            _tensor_int(native.dense_bytes, "dense tensor bytes"),
            native.dense != 0, native.symmetry_compressed != 0)
    catch
        if owner === nothing
            ccall(destroy, Cvoid, (Ptr{Cvoid},), handle)
        else
            _finalize_tensor(owner)
        end
        rethrow()
    finally
        info === nothing || ccall(dispose, Cvoid, (Ref{_CTensorInfo},), info)
    end
end

function _tensor_shape(shape)
    length(shape) == 5 || throw(ArgumentError("a full tensor requires five dimensions"))
    dimensions = ntuple(i -> _nonnegative_int(shape[i], UInt64, "tensor dimension"), 5)
    # Enforce Julia's dimension and length limits before asking C++ to allocate.
    count = 1
    for extent in dimensions
        count = Base.checked_mul(count, _tensor_int(extent, "tensor dimension"))
    end
    dimensions
end

function FullTensor(shape; init::TensorInit=TensorInit.zero,
                    layout::TensorLayout=TensorLayout.canonical_dense_column_major)
    dimensions = Ref(_tensor_shape(shape))
    function_pointer = _sym(:sjx_tensor_create)
    _tensor_result() do result
        Base.@threadcall(function_pointer, Ptr{Cvoid},
            (Ptr{NTuple{5,UInt64}}, Int32, UInt32, Ptr{Ptr{Cvoid}}),
            dimensions, Int32(init), UInt32(layout), result)
    end
end

FullTensor() = FullTensor((0, 0, 0, 0, 0))

"""Create a writable, zero-filled mapped tensor at `path` (unless `init` overrides it)."""
function create_mapped(path, shape; init::TensorInit=TensorInit.zero,
                       layout::TensorLayout=TensorLayout.canonical_dense_column_major)
    file = _path(path)
    dimensions = Ref(_tensor_shape(shape))
    function_pointer = _sym(:sjx_tensor_create_mapped)
    _tensor_result() do result
        Base.@threadcall(function_pointer, Ptr{Cvoid},
            (Cstring, Ptr{NTuple{5,UInt64}}, Int32, UInt32, Ptr{Ptr{Cvoid}}),
            file, dimensions, Int32(init), UInt32(layout), result)
    end
end

"""Load a persisted `.sl2cxx` tensor into independent, writable native memory."""
function load_tensor(path)
    file = _path(path)
    function_pointer = _sym(:sjx_tensor_load)
    _tensor_result() do result
        Base.@threadcall(function_pointer, Ptr{Cvoid},
            (Cstring, Ptr{Ptr{Cvoid}}), file, result)
    end
end

"""Map a persisted tensor; the default `TensorAccess.read_only` forbids mutation."""
function map_tensor(path; access::TensorAccess=TensorAccess.read_only)
    file = _path(path)
    function_pointer = _sym(:sjx_tensor_map)
    _tensor_result() do result
        Base.@threadcall(function_pointer, Ptr{Cvoid},
            (Cstring, Int32, Ptr{Ptr{Cvoid}}), file, Int32(access), result)
    end
end

metadata(tensor::FullTensor) = getfield(tensor, :_metadata)

"""Inspect a persisted full tensor's validated metadata without copying its payload."""
function metadata(path::AbstractString)
    file = _path(path)
    info = Ref(_empty_tensor_info())
    function_pointer = _sym(:sjx_tensor_file_info)
    dispose = _sym(:sjx_tensor_info_dispose)
    try
        _tensor_threaded() do
            Base.@threadcall(function_pointer, Ptr{Cvoid},
                (Cstring, Ptr{_CTensorInfo}), file, info)
        end
        _decode_tensor_info(info[])
    finally
        ccall(dispose, Cvoid, (Ref{_CTensorInfo},), info)
    end
end

Base.size(tensor::FullTensor) = metadata(tensor).logical_shape
Base.axes(tensor::FullTensor) = map(Base.OneTo, size(tensor))
Base.length(tensor::FullTensor) = metadata(tensor).logical_values
Base.IndexStyle(::Type{FullTensor}) = IndexCartesian()

function Base.getindex(tensor::FullTensor, indices::Vararg{Int,5})
    @boundscheck checkbounds(tensor, indices...)
    native_indices = ntuple(i -> UInt64(indices[i] - 1), 5)
    result = Ref{Float64}(0)
    GC.@preserve tensor begin
        _check_error(ccall(_sym(:sjx_tensor_get), Ptr{Cvoid},
            (Ptr{Cvoid}, Ref{NTuple{5,UInt64}}, Ref{Float64}),
            getfield(tensor, :_owner).handle, native_indices, result))
    end
    result[]
end

function Base.setindex!(tensor::FullTensor, value, indices::Vararg{Int,5})
    @boundscheck checkbounds(tensor, indices...)
    native_indices = ntuple(i -> UInt64(indices[i] - 1), 5)
    scalar = convert(Float64, value)
    GC.@preserve tensor begin
        _check_error(ccall(_sym(:sjx_tensor_set), Ptr{Cvoid},
            (Ptr{Cvoid}, Ref{NTuple{5,UInt64}}, Float64),
            getfield(tensor, :_owner).handle, native_indices, scalar))
    end
    tensor
end

# Generic slicing, broadcasting, and reductions allocate ordinary Julia arrays.
Base.similar(::FullTensor, ::Type{T}, dimensions::Dims) where {T} =
    Array{T}(undef, dimensions)

function Base.Array(tensor::FullTensor)
    output = Array{Float64}(undef, size(tensor))
    function_pointer = _sym(:sjx_tensor_copy_values)
    GC.@preserve tensor begin
        _tensor_threaded() do
            Base.@threadcall(function_pointer, Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Ptr{Float64}, UInt64),
                getfield(tensor, :_owner).handle, Int32(1), output, UInt64(length(output)))
        end
    end
    output
end

Base.Array{T}(tensor::FullTensor) where {T} = convert(Array{T,5}, Array(tensor))
Base.Array{T,5}(tensor::FullTensor) where {T} = convert(Array{T,5}, Array(tensor))

function Base.copy(tensor::FullTensor)
    function_pointer = _sym(:sjx_tensor_clone)
    GC.@preserve tensor begin
        _tensor_result() do result
            Base.@threadcall(function_pointer, Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), getfield(tensor, :_owner).handle, result)
        end
    end
end

clone(tensor::FullTensor) = copy(tensor)

function Base.deepcopy_internal(tensor::FullTensor, copies::IdDict)
    get!(copies, tensor) do
        copy(tensor)
    end
end

"""Return an independent, writable C++ tensor in canonical dense storage."""
function dense_copy(tensor::FullTensor)
    function_pointer = _sym(:sjx_tensor_dense_copy)
    GC.@preserve tensor begin
        _tensor_result() do result
            Base.@threadcall(function_pointer, Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Ptr{Cvoid}}), getfield(tensor, :_owner).handle, result)
        end
    end
end

function Base.fill!(tensor::FullTensor, value)
    scalar = convert(Float64, value)
    function_pointer = _sym(:sjx_tensor_fill)
    GC.@preserve tensor begin
        _tensor_threaded() do
            Base.@threadcall(function_pointer, Ptr{Cvoid},
                (Ptr{Cvoid}, Float64), getfield(tensor, :_owner).handle, scalar)
        end
    end
    tensor
end

"""Atomically save a tensor in its current physical representation."""
function save_tensor(tensor::FullTensor, path)
    file = _path(path)
    function_pointer = _sym(:sjx_tensor_save)
    GC.@preserve tensor begin
        _tensor_threaded() do
            Base.@threadcall(function_pointer, Ptr{Cvoid},
                (Ptr{Cvoid}, Cstring), getfield(tensor, :_owner).handle, file)
        end
    end
    nothing
end

"""Flush a mapped tensor's pending writes; owned tensors need no action."""
function Base.flush(tensor::FullTensor)
    function_pointer = _sym(:sjx_tensor_flush)
    GC.@preserve tensor begin
        _tensor_threaded() do
            Base.@threadcall(function_pointer, Ptr{Cvoid},
                (Ptr{Cvoid},), getfield(tensor, :_owner).handle)
        end
    end
    nothing
end

"""Losslessly convert a mapped tensor into the requested on-disk storage mode."""
function convert_storage(tensor::FullTensor, storage::TensorStorage, path;
                         overwrite_existing::Bool=false)
    file = _path(path)
    function_pointer = _sym(:sjx_tensor_convert_storage)
    GC.@preserve tensor begin
        _tensor_result() do result
            Base.@threadcall(function_pointer, Ptr{Cvoid},
                (Ptr{Cvoid}, Int32, Cstring, Int32, Ptr{Ptr{Cvoid}}),
                getfield(tensor, :_owner).handle, Int32(storage), file,
                Int32(overwrite_existing), result)
        end
    end
end

"""Format canonical entries using the native bounded diagnostic printer."""
function format_tensor(tensor::FullTensor,
                       options::FullTensorPrintOptions=FullTensorPrintOptions())
    result = Ref{Ptr{UInt8}}(C_NULL)
    function_pointer = _sym(:sjx_tensor_format)
    release = _sym(:sjx_free)
    try
        GC.@preserve tensor begin
            _tensor_threaded() do
                Base.@threadcall(function_pointer, Ptr{Cvoid},
                    (Ptr{Cvoid}, UInt64, Int32, Int32, Ptr{Ptr{UInt8}}),
                    getfield(tensor, :_owner).handle, options.max_values,
                    options.precision, Int32(options.include_metadata), result)
            end
        end
        result[] == C_NULL ? "" : unsafe_string(result[])
    finally
        ccall(release, Cvoid, (Ptr{Cvoid},), result[])
    end
end

"""Return an independent copy of the tensor's 1024-byte binary metadata tag."""
function tensor_tag(tensor::FullTensor)
    output = Vector{UInt8}(undef, 1024)
    GC.@preserve tensor begin
        _check_error(ccall(_sym(:sjx_tensor_tag), Ptr{Cvoid},
            (Ptr{Cvoid}, Ptr{UInt8}, UInt64),
            getfield(tensor, :_owner).handle, output, UInt64(length(output))))
    end
    output
end

"""
An owner-retaining vector over a tensor's physical payload. It may contain
compressed representatives or reassociated panels rather than canonical
entries. Use `Array(tensor)` for a canonical Julia array.
"""
struct TensorPhysicalView <: AbstractVector{Float64}
    _owner::_TensorOwner
    _data::Ptr{Float64}
    _length::Int
    _writable::Bool
end

"""Expose physical values without copying; this view and its descendants pin native storage."""
function physical_view(tensor::FullTensor)
    data = Ref{Ptr{Float64}}(C_NULL)
    count = Ref{UInt64}(0)
    writable = Ref{Int32}(0)
    owner = getfield(tensor, :_owner)
    GC.@preserve tensor begin
        _check_error(ccall(_sym(:sjx_tensor_payload), Ptr{Cvoid},
            (Ptr{Cvoid}, Ref{Ptr{Float64}}, Ref{UInt64}, Ref{Int32}),
            owner.handle, data, count, writable))
    end
    TensorPhysicalView(owner, data[], _tensor_int(count[], "stored tensor size"), writable[] != 0)
end

stored_view(tensor::FullTensor) = physical_view(tensor)

Base.size(values::TensorPhysicalView) = (values._length,)
Base.IndexStyle(::Type{TensorPhysicalView}) = IndexLinear()
Base.similar(::TensorPhysicalView, ::Type{T}, dimensions::Dims) where {T} =
    Array{T}(undef, dimensions)

function Base.deepcopy_internal(values::TensorPhysicalView, copies::IdDict)
    get!(copies, values) do
        copy(values)
    end
end

function Base.getindex(values::TensorPhysicalView, index::Int)
    @boundscheck checkbounds(values, index)
    GC.@preserve values unsafe_load(values._data, index)
end

function Base.setindex!(values::TensorPhysicalView, value, index::Int)
    @boundscheck checkbounds(values, index)
    scalar = convert(Float64, value)
    GC.@preserve values begin
        if !values._writable
            # Ask the public container to classify the rejected write exactly
            # as it would through canonical indexing, without touching memory.
            _check_error(ccall(_sym(:sjx_tensor_set_flat), Ptr{Cvoid},
                (Ptr{Cvoid}, UInt64, Float64), values._owner.handle,
                UInt64(index - 1), scalar))
        else
            unsafe_store!(values._data, scalar, index)
        end
    end
    values
end

function Base.getproperty(tensor::FullTensor, name::Symbol)
    name === :shape && return size(tensor)
    name === :strides && return getfield(tensor, :_logical_strides)
    name === :stored_size && return metadata(tensor).stored_values
    (name === :logical_size || name === :dense_size) && return length(tensor)
    (name === :stored_bytes || name === :size_bytes) && return getfield(tensor, :_stored_bytes)
    name === :dense_bytes && return getfield(tensor, :_dense_bytes)
    name === :is_dense && return getfield(tensor, :_dense)
    name === :is_reassociated && return !getfield(tensor, :_dense)
    name === :is_symmetry_compressed && return getfield(tensor, :_symmetry_compressed)
    name === :is_mapped && return metadata(tensor).mapped
    name === :is_read_only && return !metadata(tensor).writable
    name === :writable && return metadata(tensor).writable
    name === :storage && return metadata(tensor).storage
    name === :layout && return metadata(tensor).layout
    name === :reassociated_layout && return metadata(tensor).reassociated_layout
    name === :path && return metadata(tensor).path
    name === :tag && return tensor_tag(tensor)
    getfield(tensor, name)
end

function Base.propertynames(::FullTensor, private::Bool=false)
    public = (:shape, :strides, :stored_size, :logical_size, :dense_size,
        :stored_bytes, :size_bytes, :dense_bytes, :is_dense, :is_reassociated,
        :is_symmetry_compressed, :is_mapped, :is_read_only, :writable,
        :storage, :layout, :reassociated_layout, :path, :tag)
    private ? (public..., fieldnames(FullTensor)...) : public
end

function Base.show(io::IO, ::MIME"text/plain", tensor::FullTensor)
    print(io, "FullTensor", size(tensor), " with ", tensor.stored_size,
        " stored Float64 values (", tensor.storage, ")")
    tensor.is_read_only && print(io, ", read-only")
end
