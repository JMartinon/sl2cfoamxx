# SPDX-FileCopyrightText: 2026 Jeremy Martinon
# SPDX-License-Identifier: GPL-3.0-or-later

struct ProgressEvent
    operation::ProgressOperation
    phase::ProgressPhase
    backend::Backend
    status::ProgressStatus
    attempt::UInt32
    completed::UInt64
    total::UInt64
    elapsed_seconds::Float64
    eta_seconds::Float64
    eta_available::Bool
end
struct _ProgressEvent
    operation::Int32
    phase::Int32
    backend::Int32
    status::Int32
    attempt::UInt32
    eta_available::Int32
    completed::UInt64
    total::UInt64
    elapsed_seconds::Float64
    eta_seconds::Float64
end
_progress_event(raw::_ProgressEvent) = ProgressEvent(ProgressOperation(raw.operation),
    ProgressPhase(raw.phase), Backend(raw.backend), ProgressStatus(raw.status),
    raw.attempt, raw.completed, raw.total, raw.elapsed_seconds, raw.eta_seconds,
    raw.eta_available != 0)

function _backend_argument(backend, progress)
    backend === nothing || backend isa Backend ||
        throw(ArgumentError("backend must be a Backend value or nothing"))
    if progress !== nothing
        backend === nothing && throw(ArgumentError("progress requires an explicit backend"))
        example = ProgressEvent(ProgressOperation.full_tensor, ProgressPhase.contraction,
            backend, ProgressStatus.started, UInt32(1), UInt64(0), UInt64(0), 0.0, 0.0, false)
        applicable(progress, example) || throw(ArgumentError("progress must accept a ProgressEvent"))
    end
    return backend === nothing ? Ptr{Int32}(C_NULL) : Ref(Int32(backend))
end

# A long foreign call must finish before Ctrl-C can unwind the @threadcall
# argument/result roots. The normal path waits on libuv without polling.
# Only an explicit Julia callback creates an event-draining task/loop.
function _run_native(call, callback=nothing)
    Base.disable_sigint() do
        if callback === nothing
            _check_error(call(Ptr{Cvoid}(C_NULL)))
            return Ptr{Cvoid}(C_NULL)
        end
        queue_output = Ref{Ptr{Cvoid}}(C_NULL)
        _check_error(ccall(_sym(:sjx_progress_create), Ptr{Cvoid},
                           (Ref{Ptr{Cvoid}},), queue_output))
        queue = queue_output[]
        try
            events = Vector{_ProgressEvent}(undef, 64)
            count = Ref{UInt64}(0)
            pending = nothing
            function drain()
                while true
                    _check_error(ccall(_sym(:sjx_progress_poll), Ptr{Cvoid},
                        (Ptr{Cvoid}, Ptr{_ProgressEvent}, UInt64, Ref{UInt64}),
                        queue, events, UInt64(length(events)), count))
                    count[] == 0 && break
                    for index in 1:Int(count[])
                        if pending === nothing
                            try
                                callback(_progress_event(events[index]))
                            catch exception
                                pending = exception
                            end
                        end
                    end
                end
            end
            worker = @async call(queue)
            while !istaskdone(worker)
                try
                    drain()
                    sleep(0.01)
                catch exception
                    pending === nothing && (pending = exception)
                end
            end
            native_error = fetch(worker)
            try
                try
                    drain()
                catch exception
                    pending === nothing && (pending = exception)
                end
                pending === nothing || throw(pending)
                owned_error = native_error
                native_error = C_NULL
                _check_error(owned_error)
                return Ptr{Cvoid}(C_NULL)
            finally
                native_error == C_NULL ||
                    ccall(_sym(:sjx_error_free), Cvoid, (Ptr{Cvoid},), native_error)
            end
        finally
            ccall(_sym(:sjx_progress_free), Cvoid, (Ptr{Cvoid},), queue)
        end
    end
end

"""Translate doubled intertwiners to the five one-based Julia tensor indices."""
function indices_from_intertwiners(spins, intertwiners)
    labels = _labels(spins, 10, "spins")
    fixed = _labels(intertwiners, 5, "intertwiners")
    output = Vector{UInt64}(undef, 5)
    _check_error(ccall(_sym(:sjx_indices_from_intertwiners), Ptr{Cvoid},
        (Ptr{Int32}, Ptr{Int32}, Ptr{UInt64}), labels, fixed, output))
    return ntuple(index -> Base.checked_add(Int(output[index]), 1), 5)
end

function compute_amplitude(spins, intertwiners, Dl, gamma;
                           cache_policy::AmplitudeCachePolicy=AmplitudeCachePolicy.disabled)
    labels = _labels(spins, 10, "spins")
    fixed = _labels(intertwiners, 5, "intertwiners")
    cutoff = _nonnegative_int(Dl, UInt64, "Dl")
    parameter = _real(gamma, "gamma")
    result = Ref{Float64}(0.0)
    function_pointer = _sym(:sjx_compute_amplitude)
    _check_error(_run_native() do _
        Base.@threadcall(function_pointer, Ptr{Cvoid},
            (Ptr{Int32}, Ptr{Int32}, UInt64, Float64, Int32, Ptr{Float64}),
            labels, fixed, cutoff, parameter, Int32(cache_policy), result)
    end)
    return result[]
end
compute_amplitude(spins, intertwiners, Dl, gamma, policy::AmplitudeCachePolicy) =
    compute_amplitude(spins, intertwiners, Dl, gamma; cache_policy=policy)

function compute_coherent_amplitude(spins, Dl, gamma, angles; backend=nothing, progress=nothing)
    labels = _labels(spins, 10, "spins")
    cutoff = _nonnegative_int(Dl, UInt64, "Dl")
    parameter = _real(gamma, "gamma")
    normals = _angles(angles, 20)
    selected_backend = _backend_argument(backend, progress)
    result = Ref{ComplexF64}(0.0)
    function_pointer = _sym(:sjx_compute_coherent_amplitude)
    _check_error(_run_native(progress) do queue
        Base.@threadcall(function_pointer, Ptr{Cvoid},
            (Ptr{Int32}, UInt64, Float64, Ptr{Float64}, Ptr{Int32}, Ptr{Cvoid}, Ptr{ComplexF64}),
            labels, cutoff, parameter, normals, selected_backend, queue, result)
    end)
    return result[]
end
compute_coherent_amplitude(spins, Dl, gamma, angles, backend; progress=nothing) =
    compute_coherent_amplitude(spins, Dl, gamma, angles; backend=backend, progress=progress)

function compute_full_tensor(spins, Dl, gamma; backend=nothing, progress=nothing)
    labels = _labels(spins, 10, "spins")
    cutoff = _nonnegative_int(Dl, UInt64, "Dl")
    parameter = _real(gamma, "gamma")
    selected_backend = _backend_argument(backend, progress)
    function_pointer = _sym(:sjx_compute_full_tensor)
    return _tensor_result() do result
        _run_native(progress) do queue
            Base.@threadcall(function_pointer, Ptr{Cvoid},
                (Ptr{Int32}, UInt64, Float64, Ptr{Int32}, Ptr{Cvoid}, Ptr{Ptr{Cvoid}}),
                labels, cutoff, parameter, selected_backend, queue, result)
        end
    end
end
compute_full_tensor(spins, Dl, gamma, backend; progress=nothing) =
    compute_full_tensor(spins, Dl, gamma; backend=backend, progress=progress)

function compute_coherent_coefficients(spins, angles)
    labels = _labels(spins, 4, "tetrahedron spins")
    normals = _angles(angles, 4)
    data = Ref{Ptr{ComplexF64}}(C_NULL)
    count = Ref{UInt64}(0)
    function_pointer = _sym(:sjx_compute_coherent_coefficients)
    try
        _check_error(_run_native() do _
            Base.@threadcall(function_pointer, Ptr{Cvoid},
                (Ptr{Int32}, Ptr{Float64}, Ptr{Ptr{ComplexF64}}, Ptr{UInt64}),
                labels, normals, data, count)
        end)
        count[] <= typemax(Int) || throw(OverflowError("coefficient count exceeds Int"))
        result = Vector{ComplexF64}(undef, Int(count[]))
        if !isempty(result)
            GC.@preserve result unsafe_copyto!(pointer(result), data[], length(result))
        end
        return result
    finally
        data[] != C_NULL && ccall(_sym(:sjx_free), Cvoid, (Ptr{Cvoid},), data[])
    end
end

struct ClosureNodeResult
    vector::NTuple{3,Float64}
    residual_norm::Float64
    scale::Float64
    relative_residual::Float64
    closed::Bool
end
struct ClosureCheckResult
    nodes::NTuple{5,ClosureNodeResult}
    all_closed::Bool
end
Base.Bool(result::ClosureCheckResult) = result.all_closed
struct _ClosureNode
    vector::NTuple{3,Float64}
    residual_norm::Float64
    scale::Float64
    relative_residual::Float64
    closed::Int32
end
struct _ClosureResult
    nodes::NTuple{5,_ClosureNode}
    all_closed::Int32
end

function check_closure(spins, angles; absolute_tolerance=1.0e-12, relative_tolerance=1.0e-12)
    labels = _labels(spins, 10, "spins")
    normals = _angles(angles, 20)
    absolute = _real(absolute_tolerance, "absolute_tolerance")
    relative = _real(relative_tolerance, "relative_tolerance")
    result = Ref{_ClosureResult}()
    _check_error(ccall(_sym(:sjx_check_closure), Ptr{Cvoid},
        (Ptr{Int32}, Ptr{Float64}, Float64, Float64, Ref{_ClosureResult}),
        labels, normals, absolute, relative, result))
    raw = result[]
    return ClosureCheckResult(ntuple(5) do index
        node = raw.nodes[index]
        ClosureNodeResult(node.vector, node.residual_norm, node.scale,
                          node.relative_residual, node.closed != 0)
    end, raw.all_closed != 0)
end
