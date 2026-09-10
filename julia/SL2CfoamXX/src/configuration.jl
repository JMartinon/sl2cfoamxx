# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later

Base.@kwdef mutable struct TensorCacheOptions
    directory::Union{Nothing,String} = nothing
    storage::TensorStorage = TensorStorage.on_disk_sym
    output_path::Union{Nothing,String} = nothing
    reuse_existing::Bool = true
    maximum_payload_size_bytes::UInt64 = UInt64(100) * 1024^3
end

Base.@kwdef mutable struct LowLevelOptions
    booster_rho_tile::UInt32 = 0
    full_tensor_gpu_row_capacity::UInt64 = 0
    coherent_gpu_pipeline_slots::UInt32 = 0
    full_tensor_gpu_producer_waves_per_sm::UInt32 = 0
    full_tensor_cpu_p_batch::UInt32 = 0
    full_tensor_cpu_final_tile::FullTensorCpuFinalTile = FullTensorCpuFinalTile.automatic
end

Base.@kwdef mutable struct CoherentAmplitudeOptions
    check_closure::Bool = false
    pruning_ratio::Float64 = 1.0e-8
end

Base.@kwdef mutable struct LibraryConfig
    tensors::TensorCacheOptions = TensorCacheOptions()
    backend::Backend = Backend.CPU
    progress_bar::Bool = true
    simplicity_constraint::SimplicityConstraint = SimplicityConstraint.gamma_j
    cpu_threads::UInt64 = 0
    memory_budget_bytes::UInt64 = 0
    verbosity::Verbosity = Verbosity.silent
    printing_style::PrintingStyle = PrintingStyle.UTF8
    low_level::LowLevelOptions = LowLevelOptions()
    coherent_amplitude::CoherentAmplitudeOptions = CoherentAmplitudeOptions()
end

Base.copy(config::LibraryConfig) = deepcopy(config)
Base.copy(options::TensorCacheOptions) = deepcopy(options)
Base.copy(options::LowLevelOptions) = deepcopy(options)
Base.copy(options::CoherentAmplitudeOptions) = deepcopy(options)

# Keep field order and widths identical to SJXConfig in the private bridge.
struct _Config
    backend::Int32
    progress_bar::Int32
    simplicity_constraint::Int32
    verbosity::Int32
    printing_style::Int32
    tensor_storage::Int32
    reuse_existing::Int32
    check_closure::Int32
    full_tensor_cpu_final_tile::Int32
    booster_rho_tile::UInt32
    coherent_gpu_pipeline_slots::UInt32
    full_tensor_gpu_producer_waves_per_sm::UInt32
    full_tensor_cpu_p_batch::UInt32
    cpu_threads::UInt64
    memory_budget_bytes::UInt64
    maximum_payload_size_bytes::UInt64
    full_tensor_gpu_row_capacity::UInt64
    pruning_ratio::Float64
    directory::Ptr{UInt8}
    output_path::Ptr{UInt8}
end

function _config_from_native(raw::_Config)
    return LibraryConfig(
        tensors=TensorCacheOptions(directory=_nullable_string(raw.directory),
            storage=TensorStorage(raw.tensor_storage),
            output_path=_nullable_string(raw.output_path),
            reuse_existing=raw.reuse_existing != 0,
            maximum_payload_size_bytes=raw.maximum_payload_size_bytes),
        backend=Backend(raw.backend), progress_bar=raw.progress_bar != 0,
        simplicity_constraint=SimplicityConstraint(raw.simplicity_constraint),
        cpu_threads=raw.cpu_threads, memory_budget_bytes=raw.memory_budget_bytes,
        verbosity=Verbosity(raw.verbosity), printing_style=PrintingStyle(raw.printing_style),
        low_level=LowLevelOptions(booster_rho_tile=raw.booster_rho_tile,
            full_tensor_gpu_row_capacity=raw.full_tensor_gpu_row_capacity,
            coherent_gpu_pipeline_slots=raw.coherent_gpu_pipeline_slots,
            full_tensor_gpu_producer_waves_per_sm=raw.full_tensor_gpu_producer_waves_per_sm,
            full_tensor_cpu_p_batch=raw.full_tensor_cpu_p_batch,
            full_tensor_cpu_final_tile=FullTensorCpuFinalTile(raw.full_tensor_cpu_final_tile)),
        coherent_amplitude=CoherentAmplitudeOptions(check_closure=raw.check_closure != 0,
                                                   pruning_ratio=raw.pruning_ratio))
end

function _get_config(name::Symbol)
    result = Ref{_Config}()
    _check_error(ccall(_sym(name), Ptr{Cvoid}, (Ref{_Config},), result))
    try
        return _config_from_native(result[])
    finally
        ccall(_sym(:sjx_config_dispose), Cvoid, (Ref{_Config},), result)
    end
end

"""Return native defaults without creating the tensor cache directory."""
default_config() = _get_config(:sjx_default_config)
"""Return an independent copy of the normalized active configuration."""
current_config() = _get_config(:sjx_current_config)

function configure(config::LibraryConfig)
    snapshot = deepcopy(config)
    directory = _path(snapshot.tensors.directory)
    output_path = _path(snapshot.tensors.output_path)
    GC.@preserve directory output_path begin
        native = Ref(_Config(Int32(snapshot.backend), Int32(snapshot.progress_bar),
            Int32(snapshot.simplicity_constraint), Int32(snapshot.verbosity),
            Int32(snapshot.printing_style), Int32(snapshot.tensors.storage),
            Int32(snapshot.tensors.reuse_existing), Int32(snapshot.coherent_amplitude.check_closure),
            Int32(snapshot.low_level.full_tensor_cpu_final_tile),
            snapshot.low_level.booster_rho_tile, snapshot.low_level.coherent_gpu_pipeline_slots,
            snapshot.low_level.full_tensor_gpu_producer_waves_per_sm,
            snapshot.low_level.full_tensor_cpu_p_batch, snapshot.cpu_threads,
            snapshot.memory_budget_bytes, snapshot.tensors.maximum_payload_size_bytes,
            snapshot.low_level.full_tensor_gpu_row_capacity,
            snapshot.coherent_amplitude.pruning_ratio, pointer(directory), pointer(output_path)))
        _check_error(ccall(_sym(:sjx_configure), Ptr{Cvoid}, (Ref{_Config},), native))
    end
    return nothing
end
