# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Julia access to the sl2cfoam++ public C++ API.

Spins and intertwiners are doubled integers; angles are in radians. Tensor
indexing follows Julia's one-based, column-major array conventions. Native
computations yield the Julia scheduler through `@threadcall`.
"""
module SL2CfoamXX

using Libdl

include("enums.jl")
include("ffi.jl")
include("boundary_angles.jl")
include("configuration.jl")
include("capabilities.jl")
include("tensor.jl")
include("computations.jl")

export Backend, TensorStorage, SimplicityConstraint, AmplitudeCachePolicy,
       Verbosity, PrintingStyle, printing_style, FullTensorCpuFinalTile,
       TensorInit, TensorAccess, TensorLayout, FullTensorReassociatedLayout,
       ErrorCode, Error, FilesystemError, to_string,
       TensorCacheOptions, LowLevelOptions, CoherentAmplitudeOptions, LibraryConfig,
       default_config, current_config, configure,
       BuildInfo, CudaDeviceInfo, version, build_info, backend_available,
       cuda_device_count, cuda_device_info,
       ProgressOperation, ProgressPhase, ProgressStatus, ProgressEvent,
       ClosureNodeResult, ClosureCheckResult, check_closure,
       lorentzian_boundary_angles, euclidean_boundary_angles,
       indices_from_intertwiners, compute_amplitude, compute_coherent_amplitude,
       compute_full_tensor, compute_coherent_coefficients,
       FullTensor, FullTensorScientificMetadata, FullTensorMetadata, FullTensorPrintOptions,
       TensorPhysicalView, metadata, tensor_tag, create_mapped, map_tensor, load_tensor,
       save_tensor, flush, convert_storage, dense_copy, clone, physical_view,
       stored_view, format_tensor

end # module
