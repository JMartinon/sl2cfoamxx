# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later


# Each enum has its own namespace, so ordinary names such as `summary` and
# `zero` never replace Base bindings. Only these package-owned types gain
# qualified value access (Backend.CPU, TensorStorage.in_memory_sym, ...).
module _Backend
@enum Value::Int32 CPU=0 GPU=1
end
module _TensorStorage
@enum Value::Int32 in_memory_sym=0 in_memory_dense=1 on_disk_sym=2 on_disk_dense=3
end
module _SimplicityConstraint
@enum Value::Int32 gamma_j=0 gamma_j_p1=1
end
module _AmplitudeCachePolicy
@enum Value::Int32 disabled=0 prefer_full_tensor=1
end
module _Verbosity
@enum Value::Int32 silent=0 summary=1 timing=2 detailed=3
end
module _PrintingStyle
@enum Value::Int32 UTF8=0 ASCII=1
end
module _FullTensorCpuFinalTile
@enum Value::Int32 automatic=0 tile_4x4=1 tile_4x8=2 tile_8x4=3
end
module _TensorInit
@enum Value::Int32 zero=0 uninitialized=1
end
module _TensorAccess
@enum Value::Int32 read_only=0 read_write=1
end
module _TensorLayout
@enum Value::Int32 generic_column_major=0 canonical_dense_column_major=1 full_tensor_reassociated_by_i1=2
const GenericColumnMajor = generic_column_major
const CanonicalDenseColumnMajor = canonical_dense_column_major
const full_tensor_reassociated_by_a1 = full_tensor_reassociated_by_i1
const FullTensorReassociatedByA1 = full_tensor_reassociated_by_i1
const FullTensorReassociatedReflectionCompactByA1 = full_tensor_reassociated_by_i1
const FullTensorReassociatedByI1 = full_tensor_reassociated_by_i1
const FullTensorReassociatedReflectionCompactByI1 = full_tensor_reassociated_by_i1
end
module _FullTensorReassociatedLayout
@enum Value::Int32 reflection_triangular_by_i1=0 rectangular_by_i1=1
const reflection_triangular_by_a1 = reflection_triangular_by_i1
const rectangular_by_a1 = rectangular_by_i1
end
module _ProgressOperation
@enum Value::Int32 coherent_amplitude=0 full_tensor=1
end
module _ProgressPhase
@enum Value::Int32 contraction=0 streamed_computation=1
end
module _ProgressStatus
@enum Value::Int32 started=0 running=1 restarted=2 completed=3 stopped=4
end
module _ErrorCode
@enum Value::Int32 backend_unavailable=1 backend_failure=2 invalid_tensor_file=3 cache_failure=4 resource_limit=5 numerical_failure=6 internal_failure=7 unsupported_operation=8
end

const Backend = _Backend.Value
const TensorStorage = _TensorStorage.Value
const SimplicityConstraint = _SimplicityConstraint.Value
const AmplitudeCachePolicy = _AmplitudeCachePolicy.Value
const Verbosity = _Verbosity.Value
const PrintingStyle = _PrintingStyle.Value
const printing_style = PrintingStyle
const FullTensorCpuFinalTile = _FullTensorCpuFinalTile.Value
const TensorInit = _TensorInit.Value
const TensorAccess = _TensorAccess.Value
const TensorLayout = _TensorLayout.Value
const FullTensorReassociatedLayout = _FullTensorReassociatedLayout.Value
const ProgressOperation = _ProgressOperation.Value
const ProgressPhase = _ProgressPhase.Value
const ProgressStatus = _ProgressStatus.Value
const ErrorCode = _ErrorCode.Value

for (type, namespace) in ((Backend, _Backend), (TensorStorage, _TensorStorage),
        (SimplicityConstraint, _SimplicityConstraint),
        (AmplitudeCachePolicy, _AmplitudeCachePolicy), (Verbosity, _Verbosity),
        (PrintingStyle, _PrintingStyle), (FullTensorCpuFinalTile, _FullTensorCpuFinalTile),
        (TensorInit, _TensorInit), (TensorAccess, _TensorAccess),
        (TensorLayout, _TensorLayout),
        (FullTensorReassociatedLayout, _FullTensorReassociatedLayout),
        (ProgressOperation, _ProgressOperation), (ProgressPhase, _ProgressPhase),
        (ProgressStatus, _ProgressStatus), (ErrorCode, _ErrorCode))
    @eval function Base.getproperty(::Type{$type}, name::Symbol)
        if isdefined($namespace, name)
            value = getfield($namespace, name)
            value isa $type && return value
        end
        return getfield($type, name)
    end
    @eval Base.propertynames(::Type{$type}) = Tuple(Symbol(value) for value in instances($type))
    @eval to_string(value::$type) = string(value)
end
